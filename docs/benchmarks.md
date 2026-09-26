# Benchmarks and validation

This page describes how jaxphys is benchmarked and validated, how to
reproduce the numbers, and what the committed results show.

> **Hardware scope.** All committed results were measured **on CPU only, in a
> shared cloud container** (4 logical CPUs, `Intel(R) Xeon(R) Processor @
> 2.80GHz`, JAX 0.10.2 `cpu` backend, float64 enabled). **No GPU was
> available or used**, so nothing here says anything about GPU performance.
> The container is shared with other jobs, so timings carry run-to-run
> noise; the tables report the interquartile range (IQR) so it is visible.

## Reproduce

From the repository root, with the project installed
(`pip install -e ".[all]"`, which includes SciPy for the FDFD, SPH and Sod
references):

```bash
export JAX_PLATFORMS=cpu

# Full suite: performance + vmap scaling + accuracy (about 14 min here)
python -m benchmarks.run

# CI-sized run: same checks, smaller sizes (about 115 s here)
python -m benchmarks.run --quick

# One suite only
python -m benchmarks.run --suite perf        # or: scaling, accuracy

# Before/after comparison of two git revisions on identical workloads
python -m benchmarks.compare_revisions --base e51cc7c [--head <rev>]
```

Every run writes `benchmarks/results/<run_id>.json` and a Markdown report
generated from it, `benchmarks/results/<run_id>.md`. The JSON holds the raw
timing samples, all measured values and the environment: Python / JAX /
jaxlib / NumPy / SciPy / jaxphys versions, `jax.devices()`, the default
backend, the CPU model and count, `XLA_FLAGS`, the git commit (and whether
the tree was dirty), a UTC timestamp and the random seeds.
`python -m benchmarks.run` exits non-zero if any agreement or validation
check fails. CI runs `python -m benchmarks.run --quick` on Python 3.12.

## Methodology

### Timing

* **Public API.** Every performance case calls a public entry point
  (`HamiltonianSystem.simulate`, `EMGrid.simulate`, `solve_schrodinger`,
  ...) exactly as a user would, so timings include argument validation and
  the host-side NaN checks.
* **Compile time is measured, not inferred.** JAX reports the duration of
  every trace, MLIR lowering and XLA backend compilation through
  `jax.monitoring`; the suite sums those events during the first call
  (`compile` column). `first call` is the wall time of that call.
* **Steady state.** After the first call and one more warm-up call, the
  case is called `repeats` times (7 in the full run, 5 with `--quick`).
  Every call is synchronized with `block_until_ready` on all output arrays.
  The tables report the median and the IQR; min/max/mean and all samples
  are in the JSON. The number of XLA compilations during the timed calls
  is recorded (`steady_compiles`); it is 0 for every case.
* **Throughput** is work / median time with a unit per solver: steps/s
  (ODE solvers), pair interactions/s (N-body, N^2 per step), point-steps/s
  (Schrödinger), cell updates/s (FDTD, Navier-Stokes, Euler), lattice-site
  updates/s (LBM; 1e6/s = 1 MLUPS), particle-steps/s (SPH), unknowns/s
  (FDFD), k-points/s (tight binding), spin-flip attempts/s (Metropolis),
  cluster updates/s (Wolff).

### NumPy reference baselines

`benchmarks/numpy_ref.py` re-implements each solver's numerical scheme in
plain vectorized NumPy with a Python time loop: same update equations,
same boundary handling, same recording convention. Where jaxphys
differentiates a user function (Hamiltonian / Lagrangian systems) the
reference uses the closed-form derivatives. The SPH reference finds
neighbours with SciPy's compiled `cKDTree` and the FDFD reference solves the
same sparse system with SciPy's SuperLU, i.e. the tools one would normally
use. Each reference is both the "no JAX" baseline and an independent
correctness check:

* deterministic solvers: the saved outputs are compared, and the maximum
  absolute difference divided by the maximum absolute reference value must
  be below a per-case tolerance (1e-12 to 1e-8, listed in the table);
* Monte Carlo (Ising): the RNG streams differ (and jaxphys uses
  checkerboard Metropolis sweeps on even lattices), so the mean energy per
  spin is compared statistically (|z| <= 5, errors from block averages).

The references are baselines, not tuned competitors (no Numba or C).

### vmap batch scaling

The public entry points run under `jax.jit(jax.vmap(...))` (host-side
checks are skipped for traced values), so the scaling curves batch the
public API over initial conditions or parameters. For each batch size B the
table gives the median time of one batched call, trajectories/s, and the
efficiency `B * t(1) / t(B)` (> 1: one batched call beats B sequential
single calls). XLA:CPU splits large fused loops across its thread pool;
for per-step kernels of roughly 10^3-10^5 elements the synchronization cost
can exceed the gain, and batching changes which loops are split, so
efficiency below 1 is expected for some kernels on CPU.

### Before/after comparison

`benchmarks/compare_revisions.py` exports `src/` of two git revisions with
`git archive`, then runs each public-API workload in a fresh Python process
with that tree first on `PYTHONPATH`, alternating base/head for several
rounds so that drift in machine load affects both equally. It reports the
first-call time, the median of 5 warm calls, XLA compilations per warm
call, and the process peak RSS. Solvers added after the base revision have
no "before" and are only in the main suite. Between the base (`e51cc7c`)
and this code several solvers were also *corrected* (split-field PML,
Boris pushes, Schrödinger grid, Wolff cluster rule, LBM boundaries), so the
comparison measures the code a user gets, not identical kernels.

### Accuracy and validation experiments

| experiment | setup | pass criterion |
|---|---|---|
| Symplectic vs RK4 energy error | Kepler problem, e = 0.5, 200 steps/orbit, 1000 orbits (100 in `--quick`), energy sampled 20x per orbit | leapfrog / Yoshida-4: max error in the 2nd half <= 1.5x the 1st half (bounded); RK4: 2nd half > 1.5x 1st half (secular drift) |
| Convergence order vs dt | harmonic oscillator (exact, t = 10) and Kepler e = 0.5 (analytic orbit from Kepler's equation, t = 0.37 T); dt halved 6 times (4 in `--quick`); all five integrators | least-squares order over the three finest dt with error > 1e-10 within 0.3 of the theoretical order |
| Schrödinger norm and free packet | norm: 2048 points, square barrier, 20000 steps (5000); free packet: V = 0, x0 = -10, k0 = 3, sigma0 = 1, t = 5, 1024 and 4096 points (1024) vs the exact solution | max abs(norm - 1) < 1e-10; psi relative L2 error < 1e-8 |
| FDTD PEC cavity | 2D TM grid, two PEC walls 0.60 m apart, periodic in x, soft line source; probe FFT (Hann window, quadratic peak interpolation) | modes 1-4 within 5e-4 of the Yee dispersion relation sin(w dt/2) = (c dt/dx) sin(k dx/2) |
| Ising T_c and Wolff | Metropolis chains vmapped over 11 temperatures in [2.15, 2.40]; L = 8, 16, 32 (8, 16); Binder cumulant crossings with jackknife errors. Wolff vs Metropolis energy per spin on 16x16 at T = 2.0 and 3.0 | largest-size crossing within 3 jackknife errors + 1% of Onsager's 2.269185; Wolff and Metropolis within 5 sigma |
| LBM channel | (a) decay of the fundamental shear mode between bounce-back walls, H = 16, 32, 64 (16, 32), vs exp(-nu (pi/H)^2 t); (b) inflow-driven 160x34 no-slip channel, velocity inlet + pressure outlet, 30000 steps (12000) | (a) error < 5%; (b) mass change < 1e-4 over the last quarter of the run, every section carries the inflow to 1e-4, profile within 0.5% of the parabola, dp/dx within 5% of -12 mu u_mean / H^2 |
| Gradients | `jax.grad` through `simulate()` (Hamiltonian params and initial state, N-body initial velocity, Lagrangian pendulum parameter) vs central finite differences | relative difference < 1e-6 |
| Sod shock tube | `solve_euler_1d`, 100-1600 cells (100-400), t = 0.2, vs the exact Riemann solution | density L1 error < 5e-3 from 400 cells, decreasing under refinement |
| FDFD Green's function | unit line current, 15/30/60 points per wavelength (15/30), 12-cell PML, vs -(omega mu0 / 4) H0(k r) along an axis and a diagonal | max relative error < 2% at >= 30 points per wavelength; order within 0.5 of 2 |
| SPH sound wave | standing acoustic wave in a periodic box, 1600 particles (576), alpha = 0 | kinetic-energy minimum within 3% of L / (4 c0); momentum < 1e-12; energy change < 1e-3 of the initial kinetic energy |

## Results

Committed result files (all in `benchmarks/results/`, produced on the code
of this release; `src/` is that of commit `dcb0e5a`):

| file | what |
|---|---|
| `20260925T233416Z_full.{json,md}` | full suite: performance, vmap scaling, accuracy (825 s) |
| `20260926T001310Z_quick.{json,md}` | `--quick` run of the same code, the CI configuration (113 s) |
| `20260926T001258Z_before_after.{json,md}` | `compare_revisions --base e51cc7c` (0.1.x code before the 0.2.0 work) vs this code |
| `20260926T002258Z_before_after.{json,md}` | `compare_revisions --base 22abad2` on the cases the performance port touched |

The two `compare_revisions` files record `git_dirty: true` because
CHANGELOG/README/docs edits were uncommitted while they ran; `src/` was
identical to `dcb0e5a`. Every number below is copied from these files.

### Before/after: e51cc7c vs this code

Median over 3 interleaved process rounds; each round is the median of 5
warm calls after one first call. The speedups combine the 0.2.0 rewrite of
the time loops (snapshot-only rollouts, per-module compiled loops,
checkerboard Metropolis, corrected Wolff) with the later performance port
(no recompilation for new systems or parameter values, N-body and LBM
kernels); the next table isolates the latter.

| case | first call before | first call after | warm before | warm after | speedup | compiles/warm call before -> after | peak RSS before -> after |
|---|---|---|---|---|---|---|---|
| HamiltonianSystem leapfrog, 1 dof, 20000 steps, save_every=100 | 0.961 s | 0.238 s | 0.1188 s | 0.0006 s | 214.2x | 1 -> 0 | 383 -> 334 MB |
| LagrangianSystem double pendulum rk4, 20000 steps, save_every=100 | 2.616 s | 0.902 s | 0.9691 s | 0.1138 s | 8.5x | 1 -> 0 | 450 -> 362 MB |
| NBody N=64, 20000 steps, save_every=100 | 4.029 s | 1.307 s | 2.7876 s | 0.9850 s | 2.8x | 1 -> 0 | 669 -> 365 MB |
| RigidBody rk4, 20000 steps | 0.640 s | 0.333 s | 0.2847 s | 0.0189 s | 15.1x | 1 -> 0 | 402 -> 353 MB |
| ChargeSystem 2 charges, 20000 steps, save_every=100 | 1.812 s | 0.244 s | 0.2032 s | 0.0092 s | 22.2x | 1 -> 0 | 443 -> 346 MB |
| solve_schrodinger 2048 points, 20000 steps, save_every=100 | 2.497 s | 1.572 s | 1.4824 s | 0.5888 s | 2.5x | 1 -> 0 | 1694 -> 399 MB |
| EMGrid 200x200, 856 steps, save_every=50 | 1.346 s | 1.242 s | 0.7035 s | 0.2082 s | 3.4x | 1 -> 0 | 1262 -> 456 MB |
| EMGrid3D 40^3, 157 steps, save_every=20 | 1.394 s | 1.793 s | 0.5791 s | 0.1393 s | 4.2x | 1 -> 0 | 934 -> 572 MB |
| LBMGrid 200x80, 3000 steps, save_every=100 | 8.976 s | 3.894 s | 7.7943 s | 2.1099 s | 3.7x | 1 -> 0 | 1614 -> 460 MB |
| NavierStokesSolver 64x64, 2000 steps, 50 Jacobi its, save_every=100 | 1.152 s | 0.812 s | 0.8785 s | 0.2852 s | 3.1x | 1 -> 0 | 607 -> 374 MB |
| IsingLattice.run_metropolis 16x16, 50+200 sweeps | 3.130 s | 1.293 s | 1.5437 s | 0.0220 s | 70.1x | 0 -> 0 | 404 -> 380 MB |
| sweep_temperatures Wolff 16x16, 1 temperature, 20+100 updates | 34.284 s | 1.457 s | 34.9473 s | 0.0089 s | 3942.8x | 120 -> 0 | 2773 -> 402 MB |
| lindblad_evolve 2-level, 20000 steps, save_every=100 | 1.065 s | 0.769 s | 0.3779 s | 0.1836 s | 2.1x | 1 -> 0 | 410 -> 359 MB |

* **Compilations per warm call** (1 -> 0 for every solver except
  `run_metropolis`, which was already 0): at `e51cc7c` every solver rebuilt
  its jitted/scanned closure on each call, so each repeated call re-traced
  and recompiled; the Wolff sampler was dispatched eagerly (120
  compilations per call).
* **Peak RSS**: `e51cc7c` stored every step and subsampled afterwards;
  the rollouts now store only the saved snapshots.
* Several of these solvers were also corrected in between (Boris pushes for
  charges, split-field PML, periodic Schrödinger grid, Wolff cluster rule,
  checkerboard Metropolis, LBM boundaries), so not every row compares the
  same arithmetic; the workloads are identical.

### Effect of the performance port alone: 22abad2 vs this code

Same method, base = `22abad2` (0.2.0 before this port), on the workloads
the port changed. These workloads reuse one system object, so they show
the kernel changes and the `sweep_temperatures` recompilation fix; the
recompilation fixes for new `RigidBody`/`ChargeSystem`/`SPHFluid` objects
and new parameter values do not show up here (at `22abad2` each new object
cost one compilation, 0.3-2 s; `tests/test_performance.py` covers them).

| case | first call before | first call after | warm before | warm after | speedup | compiles/warm call before -> after | peak RSS before -> after |
|---|---|---|---|---|---|---|---|
| NBody N=64, 20000 steps, save_every=100 | 2.960 s | 1.289 s | 2.6955 s | 0.9497 s | 2.8x | 0 -> 0 | 365 -> 362 MB |
| RigidBody rk4, 20000 steps | 0.306 s | 0.326 s | 0.0173 s | 0.0184 s | 0.9x | 0 -> 0 | 359 -> 353 MB |
| ChargeSystem 2 charges, 20000 steps, save_every=100 | 0.236 s | 0.264 s | 0.0091 s | 0.0090 s | 1.0x | 0 -> 0 | 350 -> 350 MB |
| LBMGrid 200x80, 3000 steps, save_every=100 | 5.867 s | 3.724 s | 3.6719 s | 2.2315 s | 1.6x | 0 -> 0 | 489 -> 463 MB |
| sweep_temperatures Wolff 16x16, 1 temperature, 20+100 updates | 1.492 s | 1.510 s | 0.8323 s | 0.0091 s | 91.5x | 1 -> 0 | 430 -> 404 MB |

### Performance (full run `20260925T233416Z_full`)

Compile = trace + lower + XLA compile time measured during the first call.
Steady = median (IQR) of 7 synchronized calls after one warm-up call. NumPy
= the same scheme in plain NumPy (median of 5). The number of XLA
compilations during the timed calls was 0 for every row.

| solver | size | first call | compile | steady median (IQR) | throughput | NumPy median | speedup | agreement |
|---|---|---|---|---|---|---|---|---|
| HamiltonianSystem (leapfrog, 1 dof) | 20000 steps | 302.07 ms | 289.69 ms | 897.1 us (92.7 us) | 22.29M steps/s | 2.74 ms | 3.1x | PASS: max rel 3.1e-15 (tol 1e-12) |
| HamiltonianSystem (leapfrog, 1 dof) | 200000 steps | 258.38 ms | 249.49 ms | 2.28 ms (129.4 us) | 87.62M steps/s | 28.65 ms | 12.6x | PASS: max rel 2.9e-14 (tol 1e-12) |
| LagrangianSystem (double pendulum, rk4) | 5000 steps | 798.66 ms | 759.86 ms | 30.41 ms (1.06 ms) | 164.44k steps/s | 289.61 ms | 9.5x | PASS: max rel 2.2e-15 (tol 1e-09) |
| LagrangianSystem (double pendulum, rk4) | 20000 steps | 878.64 ms | 753.14 ms | 119.59 ms (32.11 ms) | 167.24k steps/s | 1.143 s | 9.6x | PASS: max rel 4.4e-15 (tol 1e-09) |
| NBody (velocity Verlet, O(N^2)) | N=16, 5000 steps | 479.97 ms | 471.87 ms | 9.94 ms (941.7 us) | 128.72M pair-interactions/s | 163.87 ms | 16.5x | PASS: max rel 1.2e-16 (tol 1e-09) |
| NBody (velocity Verlet, O(N^2)) | N=64, 5000 steps | 571.47 ms | 497.60 ms | 243.20 ms (24.87 ms) | 84.21M pair-interactions/s | 1.086 s | 4.5x | PASS: max rel 2.0e-16 (tol 1e-09) |
| NBody (velocity Verlet, O(N^2)) | N=256, 1000 steps | 765.16 ms | 454.74 ms | 361.15 ms (26.72 ms) | 181.47M pair-interactions/s | 3.222 s | 8.9x | PASS: max rel 3.1e-16 (tol 1e-09) |
| NBody (velocity Verlet, O(N^2)) | N=1000, 200 steps | 1.181 s | 498.06 ms | 798.37 ms (53.27 ms) | 250.51M pair-interactions/s | 12.082 s | 15.1x | PASS: max rel 4.8e-16 (tol 1e-09) |
| RigidBody (Euler eqs + quaternion, rk4) | 5000 steps | 398.84 ms | 387.48 ms | 5.20 ms (545.4 us) | 961.90k steps/s | 235.79 ms | 45.4x | PASS: max rel 6.7e-15 (tol 1e-10) |
| RigidBody (Euler eqs + quaternion, rk4) | 50000 steps | 338.59 ms | 269.05 ms | 44.13 ms (13.43 ms) | 1.13M steps/s | 2.377 s | 53.9x | PASS: max rel 2.5e-12 (tol 1e-10) |
| ChargeSystem (Coulomb + uniform B, Boris) | 3 charges, 5000 steps | 288.08 ms | 279.60 ms | 4.51 ms (601.2 us) | 1.11M steps/s | 959.67 ms | 213.0x | PASS: max rel 9.3e-15 (tol 1e-10) |
| ChargeSystem (Coulomb + uniform B, Boris) | 3 charges, 50000 steps | 315.58 ms | 274.07 ms | 38.10 ms (841.3 us) | 1.31M steps/s | 9.358 s | 245.6x | PASS: max rel 3.3e-15 (tol 1e-10) |
| solve_schrodinger (split-operator FFT) | 512 points, 5000 steps | 1.717 s | 1.648 s | 44.50 ms (1.32 ms) | 57.53M point-steps/s | 147.42 ms | 3.3x | PASS: max rel 1.7e-13 (tol 1e-09) |
| solve_schrodinger (split-operator FFT) | 2048 points, 5000 steps | 1.301 s | 1.239 s | 155.64 ms (1.49 ms) | 65.79M point-steps/s | 353.92 ms | 2.3x | PASS: max rel 1.2e-13 (tol 1e-09) |
| solve_schrodinger (split-operator FFT) | 8192 points, 5000 steps | 1.739 s | 1.253 s | 577.18 ms (20.24 ms) | 70.97M point-steps/s | 1.241 s | 2.2x | PASS: max rel 2.3e-13 (tol 1e-09) |
| solve_schrodinger_2d (split-operator FFT) | 128x128, 500 steps | 1.768 s | 1.696 s | 173.18 ms (13.77 ms) | 47.30M point-steps/s | 226.17 ms | 1.3x | PASS: max rel 4.1e-15 (tol 1e-09) |
| solve_schrodinger_2d (split-operator FFT) | 256x256, 500 steps | 2.504 s | 1.633 s | 894.12 ms (122.85 ms) | 36.65M point-steps/s | 1.234 s | 1.4x | PASS: max rel 3.5e-15 (tol 1e-09) |
| lindblad_evolve (rk4) | d=2, 20000 steps | 679.61 ms | 657.54 ms | 182.68 ms (3.74 ms) | 109.48k steps/s | 2.465 s | 13.5x | PASS: max rel 7.8e-16 (tol 1e-10) |
| lindblad_evolve (rk4) | d=8, 5000 steps | 393.34 ms | 384.67 ms | 258.07 ms (9.83 ms) | 19.37k steps/s | 858.37 ms | 3.3x | PASS: max rel 3.1e-16 (tol 1e-10) |
| TightBinding.bands (graphene) | 1000 k-points | 369.52 ms | 275.96 ms | 2.28 ms (721.5 us) | 438.00k k-points/s | 971.0 us | 0.4x | PASS: max rel 3.0e-16 (tol 1e-12) |
| TightBinding.bands (graphene) | 10000 k-points | 337.73 ms | 317.19 ms | 9.95 ms (2.00 ms) | 1.01M k-points/s | 6.61 ms | 0.7x | PASS: max rel 3.3e-16 (tol 1e-12) |
| EMGrid (2D FDTD TM, split-field PML) | 100x100, 500 steps | 1.304 s | 1.254 s | 18.75 ms (1.23 ms) | 266.64M cell-updates/s | 98.47 ms | 5.3x | PASS: max rel 7.2e-14 (tol 1e-10) |
| EMGrid (2D FDTD TM, split-field PML) | 200x200, 500 steps | 1.259 s | 1.204 s | 101.98 ms (3.02 ms) | 196.12M cell-updates/s | 373.77 ms | 3.7x | PASS: max rel 7.0e-14 (tol 1e-10) |
| EMGrid (2D FDTD TM, split-field PML) | 400x400, 500 steps | 1.310 s | 1.010 s | 404.17 ms (81.05 ms) | 197.94M cell-updates/s | 1.854 s | 4.6x | PASS: max rel 6.9e-14 (tol 1e-10) |
| EMGrid3D (3D FDTD, split-field PML) | 24^3, 100 steps | 1.804 s | 1.749 s | 30.61 ms (1.27 ms) | 45.16M cell-updates/s | 109.41 ms | 3.6x | PASS: max rel 6.6e-15 (tol 1e-10) |
| EMGrid3D (3D FDTD, split-field PML) | 40^3, 100 steps | 1.614 s | 1.567 s | 109.09 ms (5.13 ms) | 58.67M cell-updates/s | 542.34 ms | 5.0x | PASS: max rel 5.1e-15 (tol 1e-10) |
| EMGrid3D (3D FDTD, split-field PML) | 64^3, 100 steps | 1.869 s | 1.594 s | 327.18 ms (29.22 ms) | 80.12M cell-updates/s | 2.934 s | 9.0x | PASS: max rel 8.4e-15 (tol 1e-10) |
| solve_fdfd (2D TM, stretched-coordinate PML) | 100x100 | 1.047 s | 1.008 s | 72.79 ms (9.26 ms) | 137.38k unknowns/s | 81.38 ms | 1.1x | PASS: max rel 1.9e-15 (tol 1e-08) |
| solve_fdfd (2D TM, stretched-coordinate PML) | 200x200 | 971.93 ms | 935.71 ms | 752.30 ms (49.22 ms) | 53.17k unknowns/s | 687.20 ms | 0.9x | PASS: max rel 3.8e-15 (tol 1e-08) |
| LBMGrid (D2Q9 BGK) | 100x40, 500 steps | 2.060 s | 2.002 s | 122.00 ms (3.37 ms) | 16.39M lattice-updates/s | 579.10 ms | 4.7x | PASS: max rel 4.9e-14 (tol 1e-10) |
| LBMGrid (D2Q9 BGK) | 200x80, 500 steps | 2.188 s | 1.753 s | 321.53 ms (113.79 ms) | 24.88M lattice-updates/s | 2.582 s | 8.0x | PASS: max rel 6.6e-14 (tol 1e-10) |
| LBMGrid (D2Q9 BGK) | 400x160, 500 steps | 3.140 s | 2.000 s | 1.494 s (118.49 ms) | 21.41M lattice-updates/s | 12.092 s | 8.1x | PASS: max rel 1.2e-13 (tol 1e-10) |
| NavierStokesSolver (vorticity-streamfunction) | 32x32, 200 steps | 697.26 ms | 680.68 ms | 6.26 ms (1.12 ms) | 32.71M cell-steps/s | 178.98 ms | 28.6x | PASS: max rel 6.9e-18 (tol 1e-10) |
| NavierStokesSolver (vorticity-streamfunction) | 64x64, 200 steps | 577.23 ms | 568.58 ms | 36.23 ms (3.31 ms) | 22.61M cell-steps/s | 336.89 ms | 9.3x | PASS: max rel 6.9e-18 (tol 1e-10) |
| NavierStokesSolver (vorticity-streamfunction) | 128x128, 200 steps | 668.68 ms | 509.96 ms | 221.05 ms (15.38 ms) | 14.82M cell-steps/s | 961.04 ms | 4.3x | PASS: max rel 6.9e-18 (tol 1e-10) |
| SPHFluid (weakly compressible, cell list) | 1024 particles, 200 steps | 2.991 s | 2.059 s | 999.82 ms (28.29 ms) | 204.84k particle-steps/s | 580.88 ms | 0.6x | PASS: max rel 1.3e-12 (tol 1e-09) |
| SPHFluid (weakly compressible, cell list) | 4096 particles, 200 steps | 5.027 s | 1.745 s | 3.494 s (75.33 ms) | 234.48k particle-steps/s | 2.315 s | 0.7x | PASS: max rel 2.7e-12 (tol 1e-09) |
| solve_euler_1d (MUSCL-HLLC, SSP-RK2) | 400 cells, 250 steps (Sod) | 1.455 s | 1.414 s | 15.11 ms (375.0 us) | 6.62M cell-steps/s | 83.87 ms | 5.6x | PASS: max rel 4.1e-15 (tol 1e-10) |
| solve_euler_1d (MUSCL-HLLC, SSP-RK2) | 1600 cells, 950 steps (Sod) | 1.638 s | 1.590 s | 294.85 ms (13.23 ms) | 5.16M cell-steps/s | 592.86 ms | 2.0x | PASS: max rel 4.7e-14 (tol 1e-10) |
| IsingLattice.run_metropolis (checkerboard) | 8x8, 200 sweeps | 1.617 s | 1.585 s | 9.71 ms (876.9 us) | 1.65M spin-flip attempts/s | 67.62 ms | 7.0x | PASS: z=0.1 (<= 5) |
| IsingLattice.run_metropolis (checkerboard) | 16x16, 200 sweeps | 864.56 ms | 831.88 ms | 21.13 ms (2.00 ms) | 3.03M spin-flip attempts/s | 234.71 ms | 11.1x | PASS: z=0.7 (<= 5) |
| IsingLattice.run_metropolis (checkerboard) | 32x32, 200 sweeps | 888.90 ms | 840.19 ms | 30.65 ms (4.96 ms) | 8.35M spin-flip attempts/s | 941.35 ms | 30.7x | PASS: z=0.2 (<= 5) |
| sweep_temperatures (Wolff cluster) | 8x8, 400 cluster updates | 1.552 s | 1.525 s | 11.76 ms (219.2 us) | 38.25k cluster updates/s | 27.86 ms | 2.4x | PASS: z=0.0 (<= 5) |
| sweep_temperatures (Wolff cluster) | 16x16, 400 cluster updates | 914.54 ms | 892.01 ms | 18.03 ms (1.54 ms) | 24.96k cluster updates/s | 28.82 ms | 1.6x | PASS: z=1.3 (<= 5) |
| sweep_temperatures (Wolff cluster) | 32x32, 400 cluster updates | 930.12 ms | 895.60 ms | 29.51 ms (1.55 ms) | 15.25k cluster updates/s | 29.44 ms | 1.0x | PASS: z=0.4 (<= 5) |

Observations:

* Compile time (0.25-2.1 s per solver and shape) dominates a single short
  run. It is paid once per system structure and shape; repeated calls,
  including calls with new parameter values or new systems of the same
  shape, reuse the executable (enforced by `tests/test_performance.py`).
* JAX is faster than the NumPy reference for every ODE, N-body, field and
  grid-fluid solver (1.3x-246x). The gap is largest for small-state solvers where NumPy
  pays Python overhead per step, and smallest for large stencils and FFTs
  that NumPy already vectorizes.
* It is slower where the reference uses compiled library code with a
  better algorithm for the job: SPH (0.6-0.7x against SciPy's `cKDTree`
  neighbour search), FDFD (0.9-1.1x against SciPy's sparse SuperLU; jaxphys
  uses dense block elimination, which is differentiable and vmappable), and
  tight-binding bands at small sizes (0.4-0.7x against a closed-form 2x2
  NumPy formula).
* The Monte Carlo samplers are now faster than the NumPy references
  (checkerboard Metropolis 7-31x; Wolff 1.0-2.4x against a stack-based
  NumPy implementation, whose cost per update grows with the cluster size
  in the same way).

### vmap batch scaling (full run, CPU)

Efficiency `B * t(1) / t(B)`; > 1 means one batched call beats B
sequential single calls.

| kernel | work per trajectory | t(B=1) ms | eff. B=4 | eff. B=16 | eff. B=64 | trajectories/s at B=64 |
|---|---|---|---|---|---|---|
| HamiltonianSystem (anharmonic, leapfrog) | 20000 steps | 0.19 | 3.25 | 1.80 | 5.68 | 29313.4 |
| NBody (N=32) | 2000 steps | 7.89 | 0.34 | 0.46 | 0.55 | 69.1 |
| solve_schrodinger (1024 points) | 2000 steps | 31.59 | 1.43 | 1.07 | 1.09 | 34.6 |
| EMGrid 2D FDTD (100x100), batch of source frequencies | 300 steps | 9.42 | 0.50 | 0.67 | 0.67 | 70.7 |
| LBMGrid (100x40), batch of initial perturbations | 300 steps | 68.39 | 1.10 | 1.37 | 1.01 | 14.7 |
| sweep_temperatures Metropolis (16x16), batch of temperatures | 20 sweeps | 1.99 | 2.53 | 2.47 | 3.99 | 2010.5 |

Batching helps most where the per-trajectory work is tiny (a 1-dof ODE,
16x16 Metropolis sweeps) and gives little for kernels that already keep
the CPU busy (FFTs, stencils). XLA:CPU's intra-op threading makes batched
N-body and small FDTD kernels slower than sequential calls at these sizes.
These are CPU results only.

### Accuracy / validation (full run)

| check | result | status |
|---|---|---|
| Energy error, Kepler e = 0.5, 1000 orbits | max abs(dE/E): leapfrog 2.69e-3, Yoshida-4 9.24e-6, the same maximum in both halves of the run (bounded); RK4 4.65e-4 in the 1st half, 9.28e-4 in the 2nd (linear drift, -9.27e-7 per orbit) | PASS |
| Convergence order | 1.02 / 1.01 / 2.00 / 4.00 / 4.01 (oscillator) and 1.00 / 1.00 / 2.00 / 4.00 / 4.02 (Kepler) for euler / symplectic_euler / leapfrog / yoshida4 / rk4 | PASS |
| Schrödinger norm | max deviation 2.50e-12 over 20000 steps | PASS |
| Free Gaussian vs exact | psi relative L2 error 3.7e-13 (1024 points), 3.8e-13 (4096 points) | PASS |
| FDTD PEC cavity | modes 1-4 within 6.7e-5 of the Yee dispersion relation (continuum deviation -9.4e-4 at mode 4, the expected Yee dispersion) | PASS |
| Ising T_c (Binder) | 2.2559 +- 0.0069 (L = 8/16), 2.2683 +- 0.0104 (L = 16/32) vs Onsager 2.269185 | PASS |
| Wolff vs Metropolis, 16x16 | energy per spin -1.7447 vs -1.7443 at T = 2 (z = 0.23), -0.8155 vs -0.8201 at T = 3 (z = 0.93) | PASS |
| LBM shear-mode decay | rate error +0.22% / +0.05% / +0.01% for H = 16 / 32 / 64 (order 2.04; walls exactly half-way) | PASS |
| LBM driven Poiseuille channel | mean density 1.0118 and constant (relative mass change 1.7e-9 over the last 7500 steps); every section carries the inflow to 2.0e-11; profile within 7.7e-4 of the parabola; dp/dx -4.865e-5 vs -4.826e-5 from -12 mu u / H^2 (+0.81%) | PASS |
| jax.grad vs central FD | relative differences 9.2e-10, 5.8e-10, 1.3e-8, 2.5e-10 | PASS |
| Sod shock tube | density L1 error 7.9e-3 / 4.3e-3 / 2.4e-3 / 1.4e-3 / 7.8e-4 for 100 / 200 / 400 / 800 / 1600 cells (order 0.83) | PASS |
| FDFD vs Hankel Green's function | max relative error 3.2% / 0.73% / 0.30% at 15 / 30 / 60 points per wavelength (order 1.72) | PASS |
| SPH sound wave, 1600 particles | quarter period 0.02490 s vs L/(4 c0) = 0.025 s (-0.39%); momentum 2.3e-15; energy change 1.5e-4 of the initial kinetic energy | PASS |

### Checks that were expected failures on the performance branch

The performance branch's version of this suite marked three checks as
known issues. All three are fixed in this code and are ordinary, passing
checks with tighter criteria:

1. **Wolff cluster updates** re-drew rejected boundary bonds during cluster
   growth (energy per spin -2.000 vs -0.814 from Metropolis at T = 3).
   Fixed in 0.2.0: each bond is sampled once and the cluster is grown to
   convergence. Now within 1 sigma of Metropolis and within 1.3 sigma of a
   textbook NumPy Wolff implementation (perf table).
2. **Schrödinger k-grid**: `x` used `linspace(x_min, x_max, n)` (spacing
   L/(n-1)) while the FFT wavenumbers used L/n, an effective mass
   ((n-1)/n)^2 (free-packet error 4.8e-2 at 1024 points). Fixed in 0.2.0
   (periodic grid, endpoint excluded); the error is now 3.7e-13.
3. **LBM inlet/outlet mass growth**: the zero-gradient outlet let the mean
   density grow without bound (1.00 -> 2.26 in 20000 steps in a 160x34
   channel on 22abad2; long runs reach NaN). Fixed here with a Zou-He
   pressure outlet; the channel now reaches a steady Poiseuille state
   (last row of the accuracy table), and
   `tests/test_regressions.py::test_lbm_inflow_outflow_conserves_mass_and_matches_poiseuille`
   guards it. The first-order wall offset the perf branch measured in the
   shear-decay test came from colliding solid nodes, which 0.2.0 had
   already removed (the error now converges at second order).

