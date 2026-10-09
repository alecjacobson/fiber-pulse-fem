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

1. `fiber_pulse.m` (MATLAB) meshes the domain, runs the simulation and writes
   `fiber_pulse.mat`, `fiber_pulse_mesh_v6.mat` and one `fiber_pulse_fields/frame_*.mat` per
   animation frame (about 2 GB at full resolution).

       matlab -nodisplay -batch fiber_pulse            % full resolution (~12 min)
       matlab -nodisplay -batch "quick=true; fiber_pulse"   % coarse preview

2. `render_fiber_pulse.py` renders from the cached frames without rerunning the simulation:

       python3 render_fiber_pulse.py [png] [apex] [hd]

   * `fiber_pulse_hd.mp4`: E_z, |H| and |S| animation
   * `fiber_pulse_apex.mp4`: close-up of the pulse at the top of the bump
   * `fiber_pulse_poynting_hd.png`: time-averaged Poynting vector as oriented line integral
     convolution

## Dependencies

* MATLAB with [gptoolbox](https://github.com/alecjacobson/gptoolbox) on the path (meshing via
  `triangulate`, `grad`, `spline_to_poly`, ...), plus `spline_offset.m`, which offsets the cubic
  Bézier channel centre line and is not included here.
* Python 3 with numpy, scipy, matplotlib, and `ffmpeg` on the path.

Written with Claude Code.
