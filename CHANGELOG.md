# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-25

### Changed

- **Package renamed to match the distribution.** The code previously lived in
  `src/neurosim`, so `pip install jaxphys` 0.1.x installed a wheel without any
  code. The import package is now `jaxphys`.

  Migration:

  | 0.1.x | 0.2.0 |
  |---|---|
  | `import neurosim as ns` | `import jaxphys as jp` |
  | `from neurosim.quantum import ...` | `from jaxphys.quantum import ...` |
  | `neurosim.NeurosimError` | `jaxphys.JaxphysError` |

  There is no `neurosim` compatibility shim: that name belongs to an unrelated
  PyPI project.
- `jaxphys.__version__` is read from the installed package metadata.
- Every solver's time loop is a single compiled `lax.scan` that only stores
  the saved snapshots (previously every step was stored and subsampled
  afterwards, which used gigabytes for field solvers). Reverse-mode gradients
  rematerialize the steps between snapshots.
- `Params` and all result containers (`Trajectory`, `NBodyTrajectory`,
  `QuantumResult`, ...) are JAX pytrees, so `simulate()` works under
  `jax.jit`, `jax.vmap` and `jax.grad` (including gradients with respect to
  `Params` fields). Repeated `simulate()` calls reuse the compiled loop.
- `sweep_temperatures` runs all temperatures in one `jax.vmap`-ed call, and
  Metropolis sweeps on even lattices use checkerboard updates.
- The 2D and 3D FDTD solvers use a Berenger split-field PML and reject time
  steps that violate the CFL limit. Snapshot times are reported exactly.
- `SpinChain` builds the Hamiltonian directly in the S^z basis (no
  Kronecker products) and is limited to 14 sites, the largest size the dense
  diagonalization can hold in memory.
- `optimize()` compiles the gradient once instead of re-tracing per
  iteration; `projectile()` returns a pytree `ProjectileResult`.
- Minimum supported JAX is 0.4.30.
- Repeated calls never recompile. `sweep_temperatures` built a new jitted
  closure per call, and `RigidBody`, `ChargeSystem` and `SPHFluid` baked
  their constants into per-instance executables; all of them now run
  module-level compiled loops that take the system data as traced
  arguments, so new parameter values (inertia, charges, sound speed,
  temperatures, ...) reuse the executable. `optimize()` compiles the
  gradient once per objective, and `TightBinding.bands` is compiled.
  `tests/test_performance.py` checks zero compilations on repeated calls
  for every solver.
- Faster kernels: the N-body force uses per-component planes and `rsqrt`
  (2.6x to 6x faster for N = 64 to 1000 on CPU), and the LBM keeps its
  populations population-major inside the loop (1.3x to 2x). Results agree
  with the previous kernels to round-off (max relative difference 3e-14).
- Hamiltonian, Lagrangian and rigid-body params that user code needs as
  concrete values (for example `if params.k > 0:`) no longer fail with a
  tracer error: they are compiled in as constants, cached per value.

### Added

- `solve_fdfd`: 2D TM frequency-domain solver with a stretched-coordinate
  PML, differentiable with respect to the permittivity (inverse design).
- `SPHFluid`: 2D weakly compressible SPH with a cell-list neighbour search.
- `solve_euler_1d`: MUSCL-HLLC finite-volume solver for the compressible
  Euler equations.
- `TightBinding` (chain, square, honeycomb and custom lattices), with Bloch
  bands, real-space supercells and `k_path`.
- `solve_schrodinger_2d` and `GaussianWavepacket2D`.
- `LBMGrid` now honours `boundary="no_slip"` and `"free_slip"` for the
  y walls.
- `examples/bench.py`, which checks every benchmarked solver against a NumPy
  reference before timing it.
- CI matrix for Python 3.11-3.13, a build-and-install wheel smoke job, and a
  tag-triggered release workflow using PyPI trusted publishing.
- `benchmarks/`: a reproducible benchmark and validation suite
  (`python -m benchmarks.run [--quick]`) that times every solver through
  its public API against a NumPy implementation of the same scheme,
  measures vmap scaling, and runs physics validation experiments; plus
  `benchmarks/compare_revisions.py` for before/after comparisons between
  git revisions. Results and methodology are in `docs/benchmarks.md`.
  CI runs the quick suite.

### Fixed

- `yoshida4` evaluated forces at the wrong sub-step times for time-dependent
  systems (first-order error).
- `adaptive_rk45` returned a state and time from different step sizes after
  `max_reject` rejections.
- `ChargeSystem` used velocity Verlet with a stale velocity in `v x B`, so
  particle speed drifted in a pure magnetic field; it now uses Boris pushes.
- The 2D FDTD "PML" was a single damping factor that reflected strongly, and
  the 3D conductivity profile had the wrong units (the layer acted as a hard
  wall). `"reflecting"` boundaries now implement conducting walls.
- `solve_schrodinger` included the periodic endpoint twice (wrong `dx` and
  group velocity), used a `linspace` time axis that was wrong when `t_span`
  was not a multiple of `dt`, and measured transmission from a fixed offset
  instead of the barrier edge.
- `lindblad_evolve` had the same time-axis bug.
- `circular_aperture` used a truncated power series for J1 that diverged
  for large arguments (values up to 1e95).
- `TraceResult.image_distance` returned 0 or None instead of the image
  distance `-B/D`.
- The Wolff update re-sampled bonds during cluster growth and capped the
  number of growth steps, violating detailed balance.
- `metropolis_step` converted a traced value with `bool()`, so it could not
  run under `jax.jit`.
- `entropy()` ignored degeneracies inside the logarithm.
- `LBMGrid` bounce-back applied collision at solid nodes and ignored the
  `boundary` argument; `NavierStokesSolver` vorticity ignored `dx`.
- N-body and Coulomb accelerations produced NaN gradients without softening.
- `optimize(method="adam")` reported one more iteration than it performed.
- `LBMGrid` gained mass without bound: the zero-gradient outlet does not
  fix the pressure, so with a velocity inlet the mean density grew (1.00 to
  2.26 in 20000 steps in a 160x34 channel) and long runs diverged. The
  outlet is now a Zou-He pressure boundary (rho = 1); a no-slip channel
  reaches a steady Poiseuille state with stationary mass. Free-slip wall
  nodes start at the local equilibrium, which removes a start-up
  transient.
- `NBody` and `RigidBody` can be constructed inside `jax.jit`/`jax.vmap`
  (their argument checks raised `TracerBoolConversionError`).

