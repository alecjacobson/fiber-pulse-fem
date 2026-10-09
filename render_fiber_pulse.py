#!/usr/bin/env python3
"""Render animations and figures from the cached fiber_pulse.m run.

Inputs (all written by fiber_pulse.m):
  fiber_pulse_mesh_v6.mat       mesh (metres), core mask, time-averaged <S>, mode data
  fiber_pulse_fields/frame_*.mat  per-frame E (vertices), H and S (triangles)

The mesh file is a v6 .mat so scipy can read it without h5py.

Outputs:
  fiber_pulse_hd.mp4           E_z, |H| with H arrows, |S| with Poynting arrows
  fiber_pulse_apex.mp4         close-up of the pulse at the top of the bump
  fiber_pulse_poynting_hd.png  time-averaged Poynting field + mode cross-section

Usage: python3 render_fiber_pulse.py [hd] [apex] [png]   (default: all three)
"""
import glob
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from matplotlib import tri as mtri
from matplotlib.colors import LinearSegmentedColormap

BG, FG = "#0a0a12", "#ebebf2"
CM_E = LinearSegmentedColormap.from_list(
    "bko", ["#59bfff", "#0d4090", "#000000", "#8c3305", "#ffb326"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.titlecolor": FG,
    "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
})
CM_H = LinearSegmentedColormap.from_list(
    "kteal", ["#000000", "#06424d", "#12898a", "#7fd6b0", "#f4ffd8"])

M = sio.loadmat("fiber_pulse_mesh_v6.mat", squeeze_me=True)
V_um = M["V"] * 1e6
F = M["F"].astype(int) - 1
IN_CORE = M["in_core"].astype(bool)
LX_UM = float(M["Lx"]) * 1e6
E0, NEFF, Z0 = float(M["E0"]), float(M["neff"]), float(M["Z0"])
S0, H0 = E0**2 * NEFF / Z0, E0 * NEFF / Z0
CENTROID = V_um[F].mean(axis=1)
_e1, _e2 = V_um[F[:, 1]] - V_um[F[:, 0]], V_um[F[:, 2]] - V_um[F[:, 0]]
AREA = 0.5 * np.abs(_e1[:, 0] * _e2[:, 1] - _e1[:, 1] * _e2[:, 0])
FRAMES = sorted(glob.glob("fiber_pulse_fields/frame_*.mat"))
W_ABS_RIGHT = 3.0  # um, absorber thickness (not interesting to look at)


class Raster:
    """Resample mesh data onto a regular pixel grid (um)."""

    def __init__(self, xr, yr, dx):
        self.x = np.arange(xr[0], xr[1] + dx / 2, dx)
        self.y = np.arange(yr[0], yr[1] + dx / 2, dx)
        X, Y = np.meshgrid(self.x, self.y)
        self.shape, self.extent = X.shape, (xr[0], xr[1], yr[0], yr[1])
        pad = 0.5
        keep = np.where((CENTROID[:, 0] > xr[0] - pad) & (CENTROID[:, 0] < xr[1] + pad)
                        & (CENTROID[:, 1] > yr[0] - pad) & (CENTROID[:, 1] < yr[1] + pad))[0]
        finder = mtri.Triangulation(V_um[:, 0], V_um[:, 1], F[keep]).get_trifinder()
        t = finder(X.ravel(), Y.ravel())
        if np.any(t < 0):
            raise RuntimeError("pixel outside mesh")
        self.tri = keep[t]
        A, B, C = (V_um[F[self.tri, i]] for i in range(3))
        P = np.column_stack([X.ravel(), Y.ravel()])
        d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
        w0 = ((B[:, 1] - C[:, 1]) * (P[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (P[:, 1] - C[:, 1])) / d
        w1 = ((C[:, 1] - A[:, 1]) * (P[:, 0] - C[:, 0]) + (A[:, 0] - C[:, 0]) * (P[:, 1] - C[:, 1])) / d
        W = np.column_stack([w0, w1, 1 - w0 - w1])
        n = P.shape[0]
        self.M = sp.csr_matrix((W.ravel(), (np.repeat(np.arange(n), 3), F[self.tri].ravel())),
                               shape=(n, V_um.shape[0]))
        self.X, self.Y = X, Y
        self.core = IN_CORE[self.tri].reshape(self.shape).astype(float)
        # sparse set of arrow anchor triangles in the window
        self.window_tris = keep[(CENTROID[keep, 0] >= xr[0]) & (CENTROID[keep, 0] <= xr[1])
                                & (CENTROID[keep, 1] >= yr[0]) & (CENTROID[keep, 1] <= yr[1])]

    def node(self, f):
        return (self.M @ f).reshape(self.shape)

    def elem(self, f):
        return f[self.tri].reshape(self.shape)

    def arrow_tris(self, cell, mask=None):
        """Triangles under a regular grid of arrow anchors; returns (triangle ids, anchor xy)."""
        step = max(1, int(round(cell / (self.x[1] - self.x[0]))))
        sl = (slice(step // 2, None, step), slice(step // 2, None, step))
        ids = self.tri.reshape(self.shape)[sl].ravel()
        xy = np.column_stack([self.X[sl].ravel(), self.Y[sl].ravel()])
        if mask is not None:
            keep = mask[ids]
            ids, xy = ids[keep], xy[keep]
        return ids, xy


def nodal_average(f):
    """Area-weighted average of per-triangle data onto vertices."""
    num = np.bincount(F.ravel(), np.repeat(AREA * f, 3), V_um.shape[0])
    den = np.bincount(F.ravel(), np.repeat(AREA, 3), V_um.shape[0])
    return num / den


def load_frame(path):
    f = sio.loadmat(path, squeeze_me=True)
    return (f["fr_E"].astype(float), f["fr_Hx"].astype(float), f["fr_Hy"].astype(float),
            f["fr_Sx"].astype(float), f["fr_Sy"].astype(float), float(f["fr_t"]))


class Mp4:
    def __init__(self, path, fig, fps=30):
        w, h = (int(v) for v in fig.get_size_inches() * fig.dpi)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "17", path], stdin=subprocess.PIPE)
        self.fig = fig

    def write(self):
        self.fig.canvas.draw()
        self.p.stdin.write(np.asarray(self.fig.canvas.buffer_rgba()).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def sqrt_ticks(cb, vals, scale, signed):
    vals = np.asarray(vals, float)
    cb.set_ticks(np.sign(vals) * np.sqrt(np.abs(vals)) if signed else np.sqrt(vals))
    cb.set_ticklabels([f"{v * scale:.2g}" for v in vals])


def render_video(out, xr, yr, dx, cell, three_panel):
    R = Raster(xr, yr, dx)
    frames_active = []
    for p in FRAMES:
        E = load_frame(p)[0]
        if np.abs(R.node(E)).max() > 0.02 * E0:
            frames_active.append(p)
    at, axy = R.arrow_tris(cell)
    alen = 0.85 * cell
    npanel = 3 if three_panel else 1
    figsize = (25.6, 16.0) if three_panel else (25.6, 6.6)
    fig = plt.figure(figsize=figsize, dpi=100)
    fw, fh = figsize
    aspect = (xr[1] - xr[0]) / (yr[1] - yr[0])
    pw = 0.70                                   # panel width as a fraction of the figure
    left = (1 - pw - 0.04) / 2
    ph = pw * fw / aspect / fh                  # panel height giving true 1:1 data aspect
    gap = 0.058
    top = 0.925 if three_panel else 0.5 + ph / 2 + 0.02
    axs, caxs = [], []
    for k in range(npanel):
        b = top - (k + 1) * ph - k * gap
        axs.append(fig.add_axes([left, b, pw, ph]))
        caxs.append(fig.add_axes([left + pw + 0.012, b, 0.011, ph]))
    axs = np.array(axs)
    ims = []
    tv = [-1, -.5, -.1, 0, .1, .5, 1]
    titles = ["Electric field $E_z$ (out of plane)",
              "Magnetic field $H$ (in plane; arrows: direction)",
              "Poynting vector $S = E\\times H$ (arrows: direction)"]
    cmaps = [CM_E, CM_H, "magma"]
    for k, ax in enumerate(axs):
        lo = -1 if k == 0 else 0
        im = ax.imshow(np.zeros(R.shape), extent=R.extent, origin="lower", cmap=cmaps[k],
                       vmin=lo, vmax=1, interpolation="bilinear", aspect="auto")
        ax.contour(R.X, R.Y, R.core, [0.5], colors=["#c0c0c8"], linewidths=0.6)
        ax.set_title(titles[k] if three_panel else
                     "$E_z$ (colour) and Poynting vector $S$ (arrows) near the top of the bump")
        ax.set_ylabel("y [µm]")
        cb = fig.colorbar(im, cax=caxs[k])
        if k == 0:
            sqrt_ticks(cb, tv, E0 / 1e6, True); cb.set_label("$E_z$ [MV/m]", labelpad=28)
        elif k == 1:
            sqrt_ticks(cb, tv[3:], H0 / 1e3, False); cb.set_label("|H| [kA/m]", labelpad=28)
        else:
            sqrt_ticks(cb, tv[3:], S0 / 1e9, False); cb.set_label("|S| [GW/m²]", labelpad=28)
        ims.append(im)
    axs[-1].set_xlabel("x [µm]")
    qkw = dict(angles="xy", scale_units="xy", scale=1, pivot="mid", width=0.0017,
               headwidth=4.5, headlength=5, minlength=0)
    if three_panel:
        qH = axs[1].quiver(axy[:, 0], axy[:, 1], 0 * axy[:, 0], 0 * axy[:, 0], color="white", **qkw)
        qS = axs[2].quiver(axy[:, 0], axy[:, 1], 0 * axy[:, 0], 0 * axy[:, 0], color="white", **qkw)
    else:
        qS = axs[0].quiver(axy[:, 0], axy[:, 1], 0 * axy[:, 0], 0 * axy[:, 0], color="#ffe680", **qkw)
    stamp = fig.text(0.5, 0.995, "", ha="center", va="top", fontsize=26)

    def arrows(q, vx, vy, scale):
        mag = np.hypot(vx, vy)
        s = np.where(mag > 0.02 * scale, alen * np.sqrt(mag / scale) / np.maximum(mag, 1e-300), 0.0)
        q.set_UVC(s * vx, s * vy)

    mp4 = Mp4(out, fig)
    for p in frames_active:
        E, Hx, Hy, Sx, Sy, t = load_frame(p)
        Eg = R.node(E) / E0
        ims[0].set_data(np.sign(Eg) * np.sqrt(np.minimum(np.abs(Eg), 1)))
        if three_panel:
            ims[1].set_data(np.sqrt(np.minimum(R.elem(np.hypot(Hx, Hy)) / H0, 1)))
            ims[2].set_data(np.sqrt(np.minimum(R.elem(np.hypot(Sx, Sy)) / S0, 1)))
            arrows(qH, Hx[at], Hy[at], H0)
        arrows(qS, Sx[at], Sy[at], S0)
        stamp.set_text(f"λ₀ = 532 nm   TE₀ pulse in a Si₃N₄ / SiO₂ channel   t = {t * 1e15:6.1f} fs")
        mp4.write()
    mp4.close()
    plt.close(fig)
    print(f"wrote {out} ({len(frames_active)} frames)")


def mode_psi(q, kappa, gamma, d, order):
    inside = np.abs(q) <= d / 2
    if order % 2 == 0:
        edge, fin, sgn = np.cos(kappa * d / 2), np.cos(kappa * q), 1.0
    else:
        edge, fin, sgn = np.sin(kappa * d / 2), np.sin(kappa * q), np.sign(q)
    return np.where(inside, fin, sgn * edge * np.exp(-gamma * (np.abs(q) - d / 2)))


def olic(U, V, length_px, density=0.004, dot_sigma=1.4, seed=1):
    """Oriented line integral convolution: sparse noise dots smeared *downstream* along the flow.

    Each noise dot becomes a comet whose bright head is at the dot and whose tail fades in the
    flow direction, so the texture shows both the flow lines and which way the flow goes.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    ny, nx = U.shape
    mag = np.hypot(U, V)
    ux, uy = U / np.maximum(mag, 1e-300), V / np.maximum(mag, 1e-300)
    rng = np.random.default_rng(seed)
    noise = (rng.random((ny, nx)) < density) * (0.5 + 0.5 * rng.random((ny, nx)))
    noise = gaussian_filter(noise, dot_sigma)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    acc = np.zeros((ny, nx))
    x, y = xx.copy(), yy.copy()
    for k in range(length_px):
        acc += (1.0 - k / length_px) ** 1.6 * map_coordinates(noise, [y, x], order=1, mode="nearest")
        # step upstream: a pixel is lit by dots lying behind it
        x -= map_coordinates(ux, [y, x], order=1, mode="nearest")
        y -= map_coordinates(uy, [y, x], order=1, mode="nearest")
    return acc


def render_png(out):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    Sx, Sy = M["Sx_avg"], M["Sy_avg"]
    Sxn, Syn = nodal_average(Sx), nodal_average(Sy)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(plt.get_cmap("magma")(np.linspace(0.0, 0.93, 256)))
    yc_apex = (0.4166666666666 + 1 / 6) * LX_UM

    def yc_of(x):  # centre line of the first half of the channel [um]
        X = 6 * x / LX_UM
        return (X**2 / 3 - 2 * X**3 / 27) / 6 * LX_UM + 0.4166666666666 * LX_UM

    views = {
        "whole": ((0, LX_UM), (9.9, 15.1), 0.008, 1.6),
        "launch": ((1.4, 5.4), (yc_of(3.4) - 1.0, yc_of(3.4) + 1.0), 0.002, 0.5),
        "apex": ((10.5, 14.5), (yc_apex - 1.0, yc_apex + 1.0), 0.002, 0.5),
    }
    rast = {k: Raster(v[0], v[1], v[2]) for k, v in views.items()}
    fld = {k: (r.node(Sxn), r.node(Syn)) for k, r in rast.items()}
    mag_whole = np.hypot(*fld["whole"])
    hi = np.log10(np.percentile(mag_whole, 99.95))
    lo = hi - 2.6                       # show 2.6 decades of intensity; below that is black
    norm = Normalize(lo, hi)
    T = float(M["T_total"]) * 1e15

    def lic_rgb(key, gain=1.0):
        r, (U, V) = rast[key], fld[key]
        m = np.hypot(U, V)
        acc = olic(U, V, int(views[key][3] / views[key][2]),
                   density=0.004 if key != "whole" else 0.003, dot_sigma=1.4)
        w = m > 10**lo
        bright = np.clip(acc / np.percentile(acc[w], 99.7), 0, 1) ** 0.6
        bright = 0.05 + 0.95 * bright
        return cmap(norm(np.log10(np.maximum(m, 1e-300))))[..., :3] * np.clip(gain * bright, 0, 1)[..., None]

    def finish(ax, key, title, xlabel=False):
        r = rast[key]
        ax.set_xlim(r.extent[0], r.extent[1]); ax.set_ylim(r.extent[2], r.extent[3])
        ax.set_aspect("equal"); ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel("y [µm]")
        if xlabel:
            ax.set_xlabel("x [µm]")
        ax.contour(r.X, r.Y, r.core, [0.5], colors=["#9aa0b0"], linewidths=1.0, linestyles="--", zorder=5)

    fig = plt.figure(figsize=(24, 12.8), dpi=160, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    axes = []

    ax = fig.add_subplot(gs[0, :]); axes.append(ax)
    ax.imshow(lic_rgb("whole", gain=1.0), extent=rast["whole"].extent, origin="lower",
              interpolation="bilinear")
    finish(ax, "whole", "Energy flow: oriented LIC of ⟨S⟩  (comet heads → tails point downstream, colour = intensity)")

    for col, key, title in ((0, "launch", "Zoom: launch region (source sheet at x = 1.5 µm)"),
                            (1, "apex", "Zoom: apex of the bump")):
        ax = fig.add_subplot(gs[1, col]); axes.append(ax)
        ax.imshow(lic_rgb(key, gain=1.0), extent=rast[key].extent, origin="lower",
                  interpolation="bilinear")
        finish(ax, key, title, xlabel=True)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.7, pad=0.01, aspect=30)
    decades = [(j, m) for j in range(0, 12) for m in (1, 3) if lo <= np.log10(m * 10**j * 1e0) <= hi]
    tick_w = [m * 10**j for j, m in decades]            # W/m^2
    cb.set_ticks([np.log10(v) for v in tick_w])
    cb.set_ticklabels([f"{v / 1e6:g}" for v in tick_w])
    cb.set_label("|⟨S⟩|  [MW/m²]")
    fig.suptitle(f"Time-averaged Poynting vector over a {T:.0f} fs pulse   "
                 "(λ₀ = 532 nm, TE₀ mode, Si₃N₄ channel in SiO₂)", fontsize=30)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["png", "apex", "hd"]
    if "png" in what:
        render_png("fiber_pulse_poynting_hd.png")
    if "apex" in what:
        render_video("fiber_pulse_apex.mp4", (8, 17), (13.7, 15.2), 0.01, 0.3, False)
    if "hd" in what:
        render_video("fiber_pulse_hd.mp4", (0, LX_UM), (9.75, 15.25), 0.02, 0.5, True)
