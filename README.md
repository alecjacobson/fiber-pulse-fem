# fiber-pulse-fem

A graph-capturable GPU Maxwell solver for a visible-light pulse in a conforming,
Bézier-shaped 2D slab waveguide. The original Python/MATLAB implementation is retained below.

## Warp FEM solver

```bash
python -m pip install -r requirements.txt
# Limit host BLAS/OpenMP threads for reproducible setup timings.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python warp_waveguide.py --h 0.04 --out results/preview
python warp_waveguide.py --h 0.01 --frames --out results/quality
python render_warp.py --run results/quality --out artifacts
python -m unittest -v test_waveguide
```

`--h` is the requested near-guide edge spacing **in µm**, not a grid dimension.
The Triangle maximum-area constraint is `0.5*(h*clad_ratio)^2`; actual element sizes
vary near the curved interface. Arbitrary positive spacings are accepted, subject to
host/device memory. `--clad-ratio` and `--halo` control static domain refinement.
`--mesh-from results/quality/mesh.npz` reuses exactly the same host mesh.
The mesh archive stores positions in µm, zero-based triangles, material labels,
the outline/centerline, and geometry configuration.

CUDA with conditional graph support (CUDA 12.4+, compatible NVIDIA driver) is
required for the full-device loop. Tested with Warp 1.15.0 and an NVIDIA L40.
`--no-graph` queues ordinary Warp launches for debugging/comparison. It runs the
same kernels but incurs Python launch overhead. No CPU time-domain solver or
SciPy sparse operation is used by the production trajectory.

### Physical and numerical model

* 532 nm carrier; nondispersive, lossless core index 2.04 and cladding index 1.46.
  These are an idealized material model, not a prediction including measured
  material dispersion or absorption. The core supports three guided E_z slab modes.
* An 18 × 8 µm box, 0.5 µm core, and straight leads at y = 3 µm. Two cubic Bézier
  segments between x = 6 and 12 µm form a 1.5 µm bump, with horizontal endpoint
  tangents. This tighter bend makes mode conversion and radiation measurable.
  `--bend 0` produces the analytic straight-guide control.
* Matched electric and magnetic current sheets at x = 2.5 µm launch TE₀ to the
  right. The default Gaussian **field** envelope has sigma = 4 fs (9.42 fs field
  FWHM; 6.66 fs intensity FWHM), centered at 18 fs. `--sigma 8` restores the
  original approximately 19 fs field FWHM. Pulse width changes the experiment.
* Continuous P1 E_z and discontinuous P0 H_x,H_y. `warp.fem.Trimesh2D`, polynomial
  spaces, and weak-form integration assemble electric lumped mass, magnetic cell
  mass, and the derivative matrices entirely on the GPU. Fused Warp CSR kernels
  apply these operators in a staggered leapfrog update. P1 row-sum lumping is
  positive; naive triangular P2 row-sum lumping is not a safe higher-order upgrade
  (vertex integrals vanish), so this implementation deliberately retains P1/P0.
* Crank–Nicolson local damping with equal electric/magnetic loss rates in a
  1.5 µm cubic sponge at all edges. This is a lossy sponge, **not a PML**. The
  outer weak boundary is reflecting; the sponge attenuates waves before return.
* Internal coordinates are x/(1 µm), time is c₀t/(1 µm), electric field is E/E₀,
  and magnetic field is Z₀H/E₀, with E₀ = 1 MV/m. This nondimensionalization is
  exactly equivalent to SI Maxwell equations and avoids poorly scaled fp32 values.
  Output times are fs, beta is rad/µm, power is W/m of invariant slab depth, and
  pulse fluence is J/m. Frame archives are normalized, with SI conversion in the renderer.

The time step is a requested fraction of `2/sqrt(B)`, where B is a conservative
triangle-wise absolute-row-sum bound on the spectral radius of M⁻¹K. It accounts
for the smallest/most distorted cells, not just the nominal h. `--cfl` must lie in
(0,1]; decreasing it isolates temporal error. This bound can be more restrictive
than the exact eigenvalue-based limit used by the original solver.

### Device execution and measurements

Setup creates the host mesh, material/source coefficients, a conservative CFL
bound, and exact triangle/monitor intersections. After FEM assembly and kernel
warm-up, **one CUDA graph launch advances the entire trajectory**. A conditional
while-node reads a device step counter. Field updates, analytic source evaluation,
monitor reductions, frame rasterization, and counter increments all run on device.
There are no allocations, host callbacks, field transfers, convergence checks, or
host synchronizations inside the trajectory. A single completion synchronization
precedes result readback. `trajectory.dot` records the actual captured graph.

Monitors use three-point Gaussian quadrature along exact vertical cuts through
the triangles. E is centered in time to match staggered H before computing flux.
Nine input-lead probes (x = 4.0…5.6 µm) fit beta from the carrier-frequency complex
phase. The analytic beta selects the phase-unwrapping branch; the fit slope is
free. Input/output measurements are at x = 4.6 and 14.5 µm, outside the sponge.

The saved metrics distinguish:

* TE₀ forward pulse-energy transmission, TE₀ carrier-frequency transmission, and
  transmission into TE₁ and TE₂, using modal E/H overlaps. Their admittances and
  profiles are evaluated at the carrier, a narrowband approximation.
* Total net Poynting fluence through the whole domain cross-section, which
  includes radiation and other guided modes; a ±0.9 µm window is also reported.
  Radiation absorbed at the top/bottom before the output cut is absent from this metric.
* Hilbert-envelope intensity-centroid arrival time and input-to-output delay.
  Straight-guide delay is compared with analytic n_g Δx/c₀. For the bend, the
  geometric n_g L/c₀ estimate is only a reference scale: curvature, mode conversion,
  and spectral filtering can change the physical delay. Bend accuracy is assessed
  against a finer numerical run.

`metrics.json` includes configuration, source SHA256 hashes, mesh sizes, dt, timing,
phase-fit residual, final-field amplitude, and output-window tail fraction. Mesh,
FEM setup/capture, and GPU trajectory times are separate; disk compression and
rendering are excluded from those timings. All trace samples remain on device
until completion. Animation raster frames also remain on device; only their
post-run archive uses fp16 to save space. Physics and quantitative traces use fp32,
with fp64 accumulation in the port reductions.

### Reproduce the studies

```bash
python sweep_waveguide.py --study resolution
python sweep_waveguide.py --study parameters
python sweep_waveguide.py --study reference
python render_warp.py --study results/study --out artifacts
```

The runner uses one GPU simulation at a time and resumes completed cases; pass
`--rerun` after code/configuration changes. The 80 nm case runs for 480 fs, the
60 nm case for 280 fs, and finer resolution cases for 240 fs to capture slow
coarse-mesh tails. Compare runtimes with those different durations in mind.
Full field/mesh archives stay under ignored `results/`; compact measurements and
figures are committed under `artifacts/`. See the measured results below.

## Original implementation

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

* Python: `pip install -r requirements.txt` (libigl ≥ 2.6.3 for `igl.triangle`), and `ffmpeg` on the
  path for the animations.
* MATLAB version only: [gptoolbox](https://github.com/alecjacobson/gptoolbox) on the path
  (`triangulate`, `grad`, `spline_offset`, `spline_to_poly`, ...).

Written with Claude Code.
