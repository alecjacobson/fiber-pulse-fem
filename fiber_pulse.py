#!/usr/bin/env python3
"""Visible-light pulse in a curved dielectric channel (2D slab-waveguide analogue).

Python port of fiber_pulse.m. Geometry input, meshing and operator construction use
libigl (``igl.triangle.triangulate``, ``igl.winding_number``, ``igl.grad``, ...); the cubic
Bezier spline helpers that libigl does not provide (offsetting, adaptive flattening) are
implemented here on top of ``igl.cubic_is_flat`` / ``igl.cubic_split``.

Model (SI units, out-of-plane E_z, TE polarisation):

    eps dE/dt   = d_x H_y - d_y H_x - eps*alpha*E - J_z
    mu0 dH_x/dt = -d_y E - mu0*alpha*H_x
    mu0 dH_y/dt =  d_x E - mu0*alpha*H_y - M_y

E_z is piecewise linear on mesh vertices (lumped mass), H is piecewise constant on triangles,
explicit leapfrog in time. The pulse is the fundamental TE mode of the straight slab, injected
one-way by matched electric + magnetic current sheets. Impedance-matched absorbing layers
(equal electric and magnetic loss rates) close the domain.

Writes into --out (default: current directory), in the same layout as fiber_pulse.m so that
render_fiber_pulse.py can render either:
  fiber_pulse_mesh_v6.mat            mesh (F is 1-based), core mask, time-averaged <S>, mode data
  fiber_pulse_fields/frame_NNNN.mat  fr_E, fr_Hx, fr_Hy, fr_Sx, fr_Sy (single), fr_t

Usage: python3 fiber_pulse.py [--quick] [--out DIR]
"""
import argparse
import os
import time

import igl
import igl.triangle
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

C0 = 299792458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)
Z0 = MU0 * C0


# --------------------------------------------------------------------------------------------
# Cubic Bezier spline helpers (ports of the gptoolbox functions of the same names)
# --------------------------------------------------------------------------------------------
def cubic_eval(C, t):
    t = np.asarray(t, float).reshape(-1, 1)
    return ((1 - t) ** 3 * C[0] + 3 * (1 - t) ** 2 * t * C[1]
            + 3 * (1 - t) * t**2 * C[2] + t**3 * C[3])


def cubic_tangent(C, t):
    t = np.asarray(t, float).reshape(-1, 1)
    return (3 * (1 - t) ** 2 * (C[1] - C[0]) + 6 * (1 - t) * t * (C[2] - C[1])
            + 3 * t**2 * (C[3] - C[2]))


def point_cubic_squared_distance(q, C, samples=256, newton_iters=6):
    """Squared distance from point q to the cubic C (dense sampling + Newton refinement)."""
    t = np.linspace(0.0, 1.0, samples)
    d = np.sum((cubic_eval(C, t) - q) ** 2, axis=1)
    t0 = t[np.argmin(d)]
    for _ in range(newton_iters):
        p = cubic_eval(C, t0)[0] - q
        d1 = cubic_tangent(C, t0)[0]
        # second derivative of a cubic Bezier
        s = t0
        d2 = 6 * (1 - s) * (C[2] - 2 * C[1] + C[0]) + 6 * s * (C[3] - 2 * C[2] + C[1])
        g, h = p @ d1, d1 @ d1 + p @ d2
        if abs(h) < 1e-300:
            break
        t0 = float(np.clip(t0 - g / h, 0.0, 1.0))
    return float(np.sum((cubic_eval(C, t0)[0] - q) ** 2))


def spline_robust_endpoint_tangent_directions(C, epsilon=1e-7):
    """End tangent directions of one cubic, falling back to later control points if degenerate."""
    def helper(a, b, c, d):
        for v in (b - a, c - a, d - a):
            if np.sum(v**2) >= epsilon:
                return v
        return d - a
    return helper(C[0], C[1], C[2], C[3]), -helper(C[3], C[2], C[1], C[0])


def spline_endpoint_curvature(C):
    """Curvature at both end points of one cubic (signed; positive = turning left)."""
    def dblarea(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    s1 = np.linalg.norm(C[1] - C[0]) ** 3
    s4 = np.linalg.norm(C[3] - C[2]) ** 3
    return (4.0 / 6.0) * np.array([dblarea(C[0], C[1], C[2]) / s1, dblarea(C[1], C[2], C[3]) / s4])


def _normalize(v):
    return v / np.linalg.norm(v)


def _offset_cubic(C, th):
    T0, T1 = spline_robust_endpoint_tangent_directions(C)
    N0 = _normalize(np.array([-T0[1], T0[0]]))
    N1 = _normalize(np.array([-T1[1], T1[0]]))
    oC = np.full((4, C.shape[1]), np.nan)
    oC[0] = C[0] + th * N0
    oC[3] = C[3] + th * N1
    D0 = cubic_tangent(C, 0.0)[0]
    D1 = cubic_tangent(C, 1.0)[0]
    k = spline_endpoint_curvature(C)
    oC[1] = oC[0] + (D0 - th * k[0] * D0) / 3
    oC[2] = oC[3] - (D1 - th * k[1] * D1) / 3
    return oC


def _offset_cubic_recursive(C, th, tol):
    oP = _offset_cubic(C, th)
    om = cubic_eval(oP, 0.5)[0]
    dm = np.sqrt(point_cubic_squared_distance(om, C))
    if abs(dm - abs(th)) < tol:
        return oP, np.array([[0, 1, 2, 3]]), np.array([0.0, 1.0])
    C1, C2 = igl.cubic_split(C, 0.5)
    oP1, oC1, oT1 = _offset_cubic_recursive(C1, th, tol)
    oP2, oC2, oT2 = _offset_cubic_recursive(C2, th, tol)
    oP = np.vstack([oP1[:-1], oP2])
    oC = np.vstack([oC1, oC2 + oP1.shape[0] - 1])
    return oP, oC, np.concatenate([oT1 * 0.5, oT2 * 0.5 + 0.5])


def spline_offset(P, C, th, tol):
    """Offset a cubic Bezier spline by signed distance th (positive = left of travel).

    Returns (oP, oC, oI, oT) as in gptoolbox's spline_offset.m (0-based indices).
    """
    oPs, oCs, oIs, oTs, oJs = [], [], [], [], []
    count = 0
    for i in range(C.shape[0]):
        oPi, oCi, oTi = _offset_cubic_recursive(P[C[i]], th, tol)
        oCs.append(oCi + count)
        oPs.append(oPi)
        oJi = np.full(oPi.shape[0], -1, dtype=int)
        oJi[0], oJi[-1] = C[i, 0], C[i, 3]
        oJs.append(oJi)
        oIs.append(np.full(oCi.shape[0], i))
        oTs.append(oTi)
        count += oPi.shape[0]
    oP, oC = np.vstack(oPs), np.vstack(oCs)
    oJ = np.concatenate(oJs)
    # merge the shared end points of consecutive curves (entries with oJ >= 0), keep the rest
    key = np.where(oJ >= 0, oJ, -1 - np.arange(oJ.size))
    _, ia, ic = np.unique(key, return_index=True, return_inverse=True)
    # 'stable' ordering: order unique entries by first occurrence
    order = np.argsort(ia)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return oP[np.sort(ia)], remap[ic][oC], np.concatenate(oIs), np.concatenate(oTs)


def cubic_flat_eval(C, tol):
    """Adaptively subdivide a cubic until each piece is flat to tol; returns (points, params)."""
    if np.any(np.isnan(C)) or igl.cubic_is_flat(C, tol * tol):
        return C[[0, 3]], np.array([0.0, 1.0])
    C1, C2 = igl.cubic_split(C, 0.5)
    P1, T1 = cubic_flat_eval(C1, tol)
    P2, T2 = cubic_flat_eval(C2, tol)
    return np.vstack([P1, P2[1:]]), np.concatenate([T1 * 0.5, 0.5 + 0.5 * T2[1:]])


def spline_to_poly(P, C, tol):
    """Flatten a cubic Bezier spline to a polyline (V, E) that is flat to tol per segment."""
    V = [P]
    n = P.shape[0]
    edges = []
    for c in range(C.shape[0]):
        Pc, _ = cubic_flat_eval(P[C[c]], tol)
        J = np.concatenate([[C[c, 0]], n + np.arange(Pc.shape[0] - 2), [C[c, 3]]])
        edges.append(np.column_stack([J[:-1], J[1:]]))
        V.append(Pc[1:-1])
        n += Pc.shape[0] - 2
    V, E = np.vstack(V), np.vstack(edges)
    keep = np.unique(E)                      # remove_unreferenced, preserving order
    remap = np.full(V.shape[0], -1)
    remap[keep] = np.arange(keep.size)
    return V[keep], remap[E]


# --------------------------------------------------------------------------------------------
# Slab mode
# --------------------------------------------------------------------------------------------
def slab_mode(k0, d, n1, n2, order):
    """Guided TE mode of a symmetric slab of thickness d (E_z polarisation)."""
    v = k0 * d / 2 * np.sqrt(n1**2 - n2**2)
    if order >= 2 * v / np.pi:
        raise ValueError(f"slab with V = {v:.2f} supports only {int(np.ceil(2 * v / np.pi))} TE "
                         f"modes; requested order {order}")
    lo = order * np.pi / 2 + 1e-9
    hi = min((order + 1) * np.pi / 2, v) - 1e-9
    if order % 2 == 0:
        f = lambda u: u * np.tan(u) - np.sqrt(max(v**2 - u**2, 0.0))
    else:
        f = lambda u: -u / np.tan(u) - np.sqrt(max(v**2 - u**2, 0.0))
    u = brentq(f, lo, hi)
    kappa = 2 * u / d
    gamma = 2 * np.sqrt(v**2 - u**2) / d
    beta = np.sqrt((k0 * n1) ** 2 - kappa**2)
    return dict(V=v, order=order, d=d, kappa=kappa, gamma=gamma, beta=beta, neff=beta / k0)


def mode_profile(q, m):
    """Transverse profile of E_z (peak ~1) and its q-derivative."""
    inside = np.abs(q) <= m["d"] / 2
    if m["order"] % 2 == 0:
        edge = np.cos(m["kappa"] * m["d"] / 2)
        fin, dfin, sgn = np.cos(m["kappa"] * q), -m["kappa"] * np.sin(m["kappa"] * q), 1.0
    else:
        edge = np.sin(m["kappa"] * m["d"] / 2)
        fin, dfin, sgn = np.sin(m["kappa"] * q), m["kappa"] * np.cos(m["kappa"] * q), np.sign(q)
    dec = edge * np.exp(-m["gamma"] * (np.abs(q) - m["d"] / 2))
    return np.where(inside, fin, sgn * dec), np.where(inside, dfin, -m["gamma"] * dec)


# --------------------------------------------------------------------------------------------
# Geometry and mesh
# --------------------------------------------------------------------------------------------
def channel_outline(lx, core_width, sptol=2e-5, tol=1e-2):
    """Closed outline (unit square coordinates) of the channel, as in test_single.m."""
    P = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [4, 1], [5, 0], [6, 0]], float) / 6 \
        + np.array([0, 0.4166666666666])
    C = np.array([[0, 1, 2, 3], [3, 4, 5, 6]])
    th = 0.5 * core_width / lx
    oP, oC, _, _ = spline_offset(P, C, th, tol)
    uP, uC, _, _ = spline_offset(P, C, -th, tol)
    oC = oC[::-1, ::-1]

    def to_path(Pc, Cc):
        Vp, Ep = spline_to_poly(Pc, Cc, sptol)
        I, _, K = igl.edges_to_path(Ep)
        if K[0] == 1:                      # first edge traversed against its orientation
            I = I[::-1]
        return Vp[I]

    oV, uV = to_path(oP, oC), to_path(uP, uC)
    aV = np.vstack([oV, uV])
    aV[np.abs(aV[:, 0] - 0) < 1e-6, 0] = 0
    aV[np.abs(aV[:, 0] - 1) < 1e-6, 0] = 1
    return P, C, aV


def subdivide_loop(V, h):
    """Subdivide each edge of a closed polygon to length <= h, keeping the input vertices."""
    n = V.shape[0]
    parts = []
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / h)))
        parts.append(a + (np.arange(k)[:, None] / k) * (b - a))
    RV = np.vstack(parts)
    idx = np.arange(RV.shape[0])
    return RV, np.column_stack([idx, (idx + 1) % RV.shape[0]])


def build_mesh(aV, lx, h_core, h_clad):
    RV, RE = subdivide_loop(aV, h_core / lx)
    hu = h_core / lx
    nx = int(np.floor(1.0 / hu + 1e-9)) + 1
    step_y = hu * np.sqrt(3) / 2
    ny = int(np.floor((0.62 - 0.38) / step_y + 1e-9)) + 1
    X, Y = np.meshgrid(np.arange(nx) * hu, 0.38 + np.arange(ny) * step_y)
    X[1::2, :] += hu / 2
    Q = np.column_stack([X.ravel(order="F"), Y.ravel(order="F")])   # column-major like MATLAB
    Q = Q[np.abs(igl.winding_number(RV, RE, Q)) > 0.5]
    Q = Q[igl.point_mesh_squared_distance(Q, RV, RE)[0] > (0.6 * hu) ** 2]
    max_area = 0.5 * (h_clad / lx) ** 2
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    V, F = igl.triangle.triangulate(np.vstack([RV, Q, corners]), RE,
                                    flags=f"-Qcq30a{max_area:.17f}")[:2]
    in_core = np.abs(igl.winding_number(RV, RE, igl.barycenter(V, F))) > 0.5
    return V, F, in_core


def sponge_rate(x, y, lx, w, scale):
    """Quadratic loss ramp inside a layer along each domain edge [1/s]."""
    def ramp(depth, wd):
        return scale / wd * (np.maximum(depth, 0) / wd) ** 2
    return (ramp(w["left"] - x, w["left"]) + ramp(x - (lx - w["right"]), w["right"])
            + ramp(w["bottom"] - y, w["bottom"]) + ramp(y - (lx - w["top"]), w["top"]))


def make_src(pts, c, tan_dir, nrm_dir, md, x_src, sigma_g):
    rel = pts - c
    s = rel @ tan_dir
    psi, dpsi = mode_profile(rel @ nrm_dir, md)
    g = np.exp(-(pts[:, 0] - x_src) ** 2 / (2 * sigma_g**2)) / (sigma_g * np.sqrt(2 * np.pi))
    return dict(s=s, psi=psi, dpsi=dpsi, g=g)


def sheet_fields(t, s, p):
    """Forward-propagating mode fields (E_z, H_y) on the source sheet at time t."""
    tau = t - p["t0"] - s["s"] / p["vg"]
    env = np.exp(-tau**2 / (2 * p["sigma"] ** 2))
    ph = p["omega"] * (t - p["t0"]) - p["beta"] * s["s"]
    ez = p["E0"] * s["psi"] * env * np.cos(ph)
    hp = -(p["beta"] / (p["omega"] * MU0)) * ez
    hd = -p["E0"] * s["dpsi"] * env * np.sin(ph) / (MU0 * p["omega"])
    return ez, hp * p["cos"] + hd * p["sin"]


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="coarse, short preview run")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--mesh-from", metavar="MAT", help="reuse the mesh (V in metres, 1-based F, "
                    "in_core) from a fiber_pulse_mesh_v6.mat instead of meshing; for comparisons")
    args = ap.parse_args()

    lx, core_width, lambda0 = 25e-6, 0.5e-6, 532e-9
    n_core, n_clad, mode_order = 2.04, 1.46, 0
    eps_core, eps_clad = n_core**2, n_clad**2
    E0, pulse_sigma, t_end, frame_dt = 1e6, 8e-15, 280e-15, 1.5e-15
    h_core, h_clad = 0.025e-6, 0.060e-6
    if args.quick:
        h_core, h_clad, frame_dt, t_end = 0.05e-6, 0.15e-6, 4e-15, 120e-15
    x_src, sigma_g = 1.5e-6, 0.1e-6
    w_abs = dict(left=0.8e-6, right=3e-6, bottom=3e-6, top=3e-6)
    refl = 1e-4
    omega = 2 * np.pi * C0 / lambda0
    k0 = omega / C0

    md = slab_mode(k0, core_width, n_core, n_clad, mode_order)
    dk = 1e-4 * k0
    v_group = C0 * 2 * dk / (slab_mode(k0 + dk, core_width, n_core, n_clad, mode_order)["beta"]
                             - slab_mode(k0 - dk, core_width, n_core, n_clad, mode_order)["beta"])
    print(f"slab V = {md['V']:.2f} ({int(np.ceil(2 * md['V'] / np.pi))} guided TE modes), "
          f"launching TE{mode_order}: n_eff = {md['neff']:.4f}, n_g = {C0 / v_group:.4f}")

    # --- geometry and mesh (libigl) ---
    P, C, aV = channel_outline(lx, core_width)
    if args.mesh_from:
        m = sio.loadmat(args.mesh_from)
        V, F, in_core = m["V"], m["F"].astype(np.int64) - 1, m["in_core"].ravel() > 0.5
    else:
        V, F, in_core = build_mesh(aV, lx, h_core, h_clad)
        V = V * lx
    print(f"mesh: {V.shape[0]} vertices, {F.shape[0]} triangles, {in_core.sum()} core triangles")
    BC = igl.barycenter(V, F)
    nV, nF = V.shape[0], F.shape[0]
    el = igl.edge_lengths(V, F)
    print(f"edge lengths: min {1e9 * el.min():.1f} nm, max {1e9 * el.max():.1f} nm")

    # --- operators (libigl) ---
    a = np.abs(igl.doublearea(V, F)) / 2
    epsr = np.where(in_core, eps_core, eps_clad)
    G = igl.grad(V, F).tocsr()
    Gx, Gy = G[:nF], G[nF:]
    Wx = (Gx.T @ sp.diags(a)).tocsr()
    Wy = (Gy.T @ sp.diags(a)).tocsr()
    vol = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
    incidence = sp.csr_matrix((np.ones(3 * nF), (F.ravel(), np.repeat(np.arange(nF), 3))), shape=(nV, nF))
    me = EPS0 * (incidence @ (a * epsr / 3))
    avg = sp.csr_matrix((np.full(3 * nF, 1 / 3), (np.repeat(np.arange(nF), 3), F.ravel())), shape=(nF, nV))

    # leapfrog stability: dt <= 2 / sqrt(lambda_max(me^-1 L)), L = (Gx'aGx + Gy'aGy)/mu0
    L = (Wx @ Gx + Wy @ Gy) / MU0
    dinv = sp.diags(1 / np.sqrt(me))
    lam = spla.eigsh(dinv @ L @ dinv, k=1, which="LA", tol=1e-4, return_eigenvectors=False)[0]
    dt = 0.95 * 2 / np.sqrt(lam)
    nsteps = int(np.ceil(t_end / dt))
    print(f"dt = {dt * 1e15:.3g} fs, {nsteps} steps")

    # --- absorbing layers ---
    scale = 3 * (C0 / n_core) * np.log(1 / refl) / 2
    alpha_e = sponge_rate(V[:, 0], V[:, 1], lx, w_abs, scale)
    alpha_h = sponge_rate(BC[:, 0], BC[:, 1], lx, w_abs, scale)
    cE = (1 - alpha_e * dt / 2) / (1 + alpha_e * dt / 2)
    dE = dt / (me * (1 + alpha_e * dt / 2))
    cH = (1 - alpha_h * dt / 2) / (1 + alpha_h * dt / 2)
    dH = (dt / MU0) / (1 + alpha_h * dt / 2)

    # --- one-way mode-matched sheet source ---
    t = np.linspace(0, 1, 20001)[:, None]
    B = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t**2 * P[2] + t**3 * P[3]
    dB = 3 * (1 - t) ** 2 * (P[1] - P[0]) + 6 * (1 - t) * t * (P[2] - P[1]) + 3 * t**2 * (P[3] - P[2])
    yc = np.interp(x_src / lx, B[:, 0], B[:, 1]) * lx
    tan_dir = np.array([np.interp(x_src / lx, B[:, 0], dB[:, 0]), np.interp(x_src / lx, B[:, 0], dB[:, 1])])
    tan_dir /= np.linalg.norm(tan_dir)
    nrm_dir = np.array([-tan_dir[1], tan_dir[0]])
    print(f"source at x = {x_src * 1e6:.1f} um, guide tilt {np.degrees(np.arctan2(tan_dir[1], tan_dir[0])):.1f} deg")
    q_max = core_width / 2 + np.log(1e4) / md["gamma"]
    c_src = np.array([x_src, yc])
    src_nodes = np.where((np.abs(V[:, 0] - x_src) < 4 * sigma_g) & (np.abs((V - c_src) @ nrm_dir) < q_max))[0]
    src_tris = np.where((np.abs(BC[:, 0] - x_src) < 4 * sigma_g) & (np.abs((BC - c_src) @ nrm_dir) < q_max))[0]
    sn = make_src(V[src_nodes], c_src, tan_dir, nrm_dir, md, x_src, sigma_g)
    st = make_src(BC[src_tris], c_src, tan_dir, nrm_dir, md, x_src, sigma_g)
    pulse = dict(t0=4.5 * pulse_sigma, sigma=pulse_sigma, omega=omega, vg=v_group, E0=E0,
                 beta=md["beta"], cos=tan_dir[0], sin=tan_dir[1])

    # --- time stepping ---
    out_fields = os.path.join(args.out, "fiber_pulse_fields")
    os.makedirs(out_fields, exist_ok=True)
    E = np.zeros(nV)
    Hx, Hy = np.zeros(nF), np.zeros(nF)
    Sx_acc, Sy_acc = np.zeros(nF), np.zeros(nF)
    My = np.zeros(nF)
    frame_every = max(1, int(round(frame_dt / dt)))
    kf = 0
    t0 = time.time()
    for n in range(nsteps):
        t_n = n * dt
        ez_des, _ = sheet_fields(t_n, st, pulse)
        My[src_tris] = ez_des * st["g"]
        Hx = cH * Hx - dH * (Gy @ E)
        Hy = cH * Hy + dH * (Gx @ E - My)
        _, hy_des = sheet_fields(t_n + dt / 2, sn, pulse)
        rhs = Wy @ Hx - Wx @ Hy
        rhs[src_nodes] -= vol[src_nodes] * hy_des * sn["g"]
        E_old = E
        E = cE * E + dE * rhs
        em = avg @ (0.5 * (E + E_old))
        Sx, Sy = -em * Hy, em * Hx
        Sx_acc += dt * Sx
        Sy_acc += dt * Sy
        if n % frame_every == 0:
            kf += 1
            sio.savemat(os.path.join(out_fields, f"frame_{kf:04d}.mat"),
                        dict(fr_E=E.astype(np.float32)[:, None], fr_Hx=Hx.astype(np.float32)[:, None],
                             fr_Hy=Hy.astype(np.float32)[:, None], fr_Sx=Sx.astype(np.float32)[:, None],
                             fr_Sy=Sy.astype(np.float32)[:, None], fr_t=t_n + dt), do_compression=False)
        if n % 500 == 0:
            emax = np.abs(E).max()
            print(f"step {n}/{nsteps}  t = {t_n * 1e15:.1f} fs  max|E| = {emax:.3g} V/m  ({time.time() - t0:.0f} s)")
            if not np.isfinite(emax) or emax > 1e3 * E0:
                raise RuntimeError(f"simulation blew up (max|E| = {emax:g})")
    t_total = nsteps * dt
    sx_avg, sy_avg = Sx_acc / t_total, Sy_acc / t_total

    # --- energy-flux sanity check: fluence through vertical strips ---
    for xc in np.array([0.9, 3, 10, 18, 21]) * 1e-6:
        strip = np.abs(BC[:, 0] - xc) < 0.15e-6
        print(f"net x-fluence through x = {xc * 1e6:4.1f} um: {np.sum(Sx_acc[strip] * a[strip]) / 0.3e-6:9.3e} J/m")
    print(f"max|E| at end = {np.abs(E).max():.3g} V/m (should be ~0 after absorption)")

    sio.savemat(os.path.join(args.out, "fiber_pulse_mesh_v6.mat"), dict(
        V=V, F=F + 1, in_core=in_core.astype(float), Sx_avg=sx_avg, Sy_avg=sy_avg, T_total=t_total,
        Lx=lx, E0=E0, neff=md["neff"], beta=md["beta"], kappa=md["kappa"], gamma=md["gamma"],
        core_width=core_width, Z0=Z0, lambda0=lambda0, order=mode_order, dt=dt))
    print(f"wrote {args.out}/fiber_pulse_mesh_v6.mat and {kf} frames in {out_fields}")


if __name__ == "__main__":
    main()
