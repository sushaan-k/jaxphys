# Physics Modules

`neurosim` is organized around a small set of public physics domains.
Each domain uses the same JAX-backed array model, so the outputs can be
combined, differentiated, and visualized with the same tooling.

## Classical Mechanics

Use `neurosim.classical` for Lagrangian and Hamiltonian mechanics,
rigid-body rotation, and direct N-body gravity.

Common entry points:

- `LagrangianSystem`
- `HamiltonianSystem`
- `RigidBody`
- `NBody`
- `velocity_verlet`
- `yoshida4`

## Electromagnetism

Use `neurosim.em` for Maxwell-style field problems and charged-particle
dynamics.

Common entry points:

- `EMGrid`
- `PlaneWave`
- `Wall`
- `ChargeSystem`
- `RectangularWaveguide`

The FDTD grid supports absorbing, periodic, and reflecting boundaries.

## Quantum Mechanics

Use `neurosim.quantum` for 1D wave mechanics and matrix-based quantum
systems.

Common entry points:

- `solve_schrodinger`
- `GaussianWavepacket`
- `SquareBarrier`
- `HarmonicPotential`
- `DoubleWellPotential`
- `solve_eigenvalue_problem`
- `SpinChain`

## Statistical Mechanics

Use `neurosim.statmech` for Ising and Boltzmann-style calculations.

Common entry points:

- `IsingLattice`
- `sweep_temperatures`
- `metropolis_step`
- `wolff_step`
- `boltzmann_distribution`

`sweep_temperatures(..., algorithm="wolff_cluster")` now uses the
Wolff cluster update path directly for zero-field Ising systems.

## Optics

Use `neurosim.optics` for geometric ray tracing and scalar diffraction.

Common entry points:

- `trace_system`
- `single_slit`
- `double_slit`
- `circular_aperture`

## Optimization

Use `neurosim.optimize` when you want to solve inverse problems or tune
simulation inputs to match a target outcome.

Common entry points:

- `optimize`
- `sensitivity`
- `projectile`

## Practical Notes

- All public modules use JAX arrays.
- Many solvers can run on CPU or GPU depending on the installed JAX
  backend.
- The repository enables 64-bit mode at import time to keep numerical
  results stable across the examples and tests.
