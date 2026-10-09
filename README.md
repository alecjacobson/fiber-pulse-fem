# fiber-pulse-fem

Time-domain finite-element electromagnetic simulation of a visible-light pulse in a curved
dielectric channel (a 2D slab-waveguide analogue of an optical fiber), with animations and
visualizations of E, H and the Poynting vector.

## Model

* 25 µm × 25 µm domain with a 0.5 µm thick Si₃N₄ channel (n = 2.04) in SiO₂ (n = 1.46).
* Out-of-plane E_z (TE polarisation); first-order Maxwell system in SI units.
* E_z is piecewise linear on mesh vertices, H is piecewise constant on triangles, and the
  Poynting vector S = E × H is evaluated per triangle. Explicit leapfrog time stepping with a
  lumped mass matrix.
* The pulse (λ₀ = 532 nm, ~19 fs FWHM Gaussian envelope) is the fundamental TE mode of the
  straight slab, injected one-way by a matched electric + magnetic current sheet.
* Impedance-matched absorbing layers (equal electric and magnetic loss rates) on all edges.

## Usage

Two equivalent implementations of the simulation write the same cache layout:

* `fiber_pulse.py` (Python, libigl): meshing, outline/offset of the cubic Bézier channel and
  operator construction use libigl; the spline helpers that libigl lacks (offsetting, adaptive
  flattening) are ported on top of `igl.cubic_is_flat` / `igl.cubic_split`.

      python3 fiber_pulse.py --out run            # full resolution
      python3 fiber_pulse.py --quick --out run    # coarse, short preview (seconds)

* `fiber_pulse.m` (MATLAB, gptoolbox): the original implementation; also writes an mp4 and a
  summary png.

      matlab -nodisplay -batch fiber_pulse
      matlab -nodisplay -batch "quick=true; fiber_pulse"

Both write `fiber_pulse_mesh_v6.mat` (mesh, core mask, time-averaged ⟨S⟩, mode data) and one
`fiber_pulse_fields/frame_*.mat` per animation frame (about 2 GB at full resolution).

`render_fiber_pulse.py` renders from the cached frames without rerunning the simulation:

    python3 render_fiber_pulse.py --dir run [png] [apex] [hd]

* `fiber_pulse_hd.mp4`: E_z, |H| and |S| animation
* `fiber_pulse_apex.mp4`: close-up of the pulse at the top of the bump
* `fiber_pulse_poynting_hd.png`: time-averaged Poynting vector as oriented line integral
  convolution

### Agreement between the two implementations

The Python port reproduces the MATLAB geometry exactly (offset spline, flattened outline) and,
given the same mesh (`--mesh-from`), the fields to about 1e-7 relative error (the cache is
stored in single precision) and the time-averaged ⟨S⟩ to about 3e-9. Meshes generated
independently by the two can differ slightly: the hex-lattice points differ by one rounding
error, which changes how Triangle breaks ties among cocircular points. Fed MATLAB's exact
lattice points, libigl's `triangle.triangulate` returns an identical mesh.

## Dependencies

* Python: `pip install -r requirements.txt` (libigl ≥ 2.6 for `igl.triangle`), and `ffmpeg` on the
  path for the animations.
* MATLAB version only: [gptoolbox](https://github.com/alecjacobson/gptoolbox) on the path
  (`triangulate`, `grad`, `spline_offset`, `spline_to_poly`, ...).

Written with Claude Code.
