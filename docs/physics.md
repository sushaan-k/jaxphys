# Mathematical Foundations

This document explains the mathematical framework behind jaxphys's
physics engines.  It covers the core formalisms and the numerical
methods used to solve them.

---

## 1. Lagrangian Mechanics

### The Principle of Least Action

A mechanical system with generalized coordinates **q** evolves so as
to make the *action* functional stationary:

    S[q] = integral from t1 to t2 of L(q, dq/dt, t) dt

where L = T - V is the Lagrangian (kinetic minus potential energy).

### Euler-Lagrange Equations

The stationary-action requirement yields the Euler-Lagrange equations:

    d/dt (dL/d(dq_i/dt)) - dL/dq_i = 0      for i = 1 ... n

Expanding via the chain rule and solving for the acceleration:

    M * qddot = dL/dq - (d^2L / (d(dq/dt) dq)) * dq/dt

where M = d^2L / d(dq/dt)^2 is the *mass matrix* (Hessian of L with
respect to the generalized velocities).

jaxphys derives M, dL/dq, and the mixed Hessian automatically using
JAX autodiff (`jax.grad`, `jax.hessian`, `jax.jacfwd`), then solves
the linear system at each timestep.

### Implementation

`LagrangianSystem` takes a user-defined Python function `L(q, qdot,
params) -> scalar` and produces the equations of motion without
symbolic algebra:

```python
dL_dq     = jax.grad(L, argnums=0)
dL_dqdot  = jax.grad(L, argnums=1)
M         = jax.hessian(L, argnums=1)
mixed     = jax.jacfwd(dL_dqdot, argnums=0)

qddot = jnp.linalg.solve(M(q, qdot, p), dL_dq(q, qdot, p) - mixed(q, qdot, p) @ qdot)
```

---

## 2. Hamiltonian Mechanics

### Hamilton's Equations

Given a Hamiltonian H(q, p) the equations of motion are:

    dq/dt =  dH/dp
    dp/dt = -dH/dq

These equations preserve phase-space volume (Liouville's theorem) and
are naturally suited to symplectic integrators.

jaxphys derives the right-hand sides via `jax.grad`:

```python
dq_dt =  jax.grad(H, argnums=1)(q, p, params)
dp_dt = -jax.grad(H, argnums=0)(q, p, params)
```

### Legendre Transform

The Hamiltonian is related to the Lagrangian by:

    H(q, p) = p . dq/dt - L(q, dq/dt)

where p = dL/d(dq/dt) are the conjugate momenta.

---

## 3. Numerical Integrators

### 3.1 Symplectic Euler (1st order)

    p_{n+1} = p_n + dt * dp/dt(q_n, p_n)
    q_{n+1} = q_n + dt * dq/dt(q_n, p_{n+1})

First-order, preserves the symplectic 2-form.

### 3.2 Leapfrog / Stormer-Verlet (2nd order)

    p_{1/2} = p_n     + (dt/2) * dp/dt(q_n)
    q_{n+1} = q_n     +  dt    * dq/dt(p_{1/2})
    p_{n+1} = p_{1/2} + (dt/2) * dp/dt(q_{n+1})

Second-order, time-reversible, symplectic.  The workhorse for
long-time Hamiltonian simulations because it keeps energy drift
bounded over exponentially long times.

### 3.3 Yoshida 4th-order

Composes three leapfrog steps with coefficients:

    w1 = 1 / (2 - 2^{1/3})
    w0 = -2^{1/3} / (2 - 2^{1/3})

so that the combined step is accurate to O(dt^4) while remaining
symplectic.  Reference: Yoshida (1990).

### 3.4 Classical Runge-Kutta (RK4)

The standard four-stage explicit method:

    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt/2 * k1)
    k3 = f(t_n + dt/2, y_n + dt/2 * k2)
    k4 = f(t_n + dt,   y_n + dt   * k3)
    y_{n+1} = y_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)

Fourth-order accurate but *not* symplectic.  Best for non-Hamiltonian
systems or Lagrangian formulations where the EOM are not separable.

### 3.5 Adaptive RK45 (Dormand-Prince)

Embeds a 4th-order solution inside a 5th-order Runge-Kutta formula.
The difference provides a local error estimate:

    err = |y5 - y4|

The step is accepted if the scaled error norm is below 1; otherwise it
is retried with

    h_new = h * safety * err_norm^{-1/4}

`adaptive_rk45` returns one accepted step (of size at most the `dt` it was
given, reported through the returned time). Its accept/reject loop runs
on the host, so call it from a Python loop; it cannot drive the compiled
`simulate()` loops.

---

## 4. Split-Operator Method (Quantum)

For the time-dependent Schrodinger equation:

    i * hbar * d|psi>/dt = H |psi>

with H = T + V (kinetic + potential), jaxphys uses the split-operator
(Strang splitting) approach:

    |psi(t + dt)> = exp(-i V dt/2) * F^{-1}[ exp(-i T_k dt) * F[ exp(-i V dt/2) |psi(t)> ] ]

where F denotes the Fourier transform and T_k = hbar^2 k^2 / (2m) is
the kinetic energy in momentum space.

This is second-order accurate in dt and exactly unitary, so
probability is conserved to machine precision. The grid is periodic
(`x` samples `[x_min, x_max)`), which is what the FFT assumes.
`solve_schrodinger_2d` applies the same scheme with a 2D FFT and
`|k|^2 = kx^2 + ky^2`.

---

## 5. FDTD Maxwell Solver

The Finite-Difference Time-Domain method solves Maxwell's curl
equations on a staggered Yee grid:

    dE/dt = (1/eps) * curl(H) - J/eps
    dH/dt = -(1/mu) * curl(E)

The electric and magnetic fields are offset by half a grid cell and
half a time step, giving second-order accuracy in both space and time.

The Courant stability condition requires:

    dt <= dx / (c * sqrt(D))

where D is the spatial dimension and c = 1/sqrt(eps * mu).

Absorbing boundaries use Berenger's split-field Perfectly Matched Layer:
each field component is split into the parts driven by its two curl
terms, and each part decays with the conductivity along the axis of its
derivative. With the matched magnetic conductivity sigma* = sigma mu0/eps0
the layer is reflectionless at the continuum level for every angle of
incidence. The conductivity is graded as sigma(d) = sigma_max (d/L)^3 with
sigma_max = 0.8 (m + 1) / (eta0 dx), m = 3, and the layer is backed by
perfectly conducting walls. Both `EMGrid` (2D TM) and `EMGrid3D` use it;
the tests compare a PML-terminated grid against a much larger grid and
require agreement to 5%.

## 5b. FDFD (frequency domain)

For a harmonic source J exp(-i omega t), `solve_fdfd` solves the 2D TM
Helmholtz equation

    (1/s_x) d/dx (1/s_x) dEz/dx + (1/s_y) d/dy (1/s_y) dEz/dy
        + k0^2 eps_r Ez = -i omega mu0 Jz

with stretched-coordinate PML factors s = 1 + i sigma / (omega eps0). The
5-point system is block tridiagonal and is solved exactly by block LU
elimination (a `lax.scan` of dense solves), so the solve is jit-able,
vmap-able and differentiable with respect to eps_r. A unit line current
reproduces the free-space Green's function -(omega mu0 / 4) H0^(1)(k r).

---

## 6. Monte Carlo Methods (Statistical Mechanics)

### Metropolis Algorithm

For the Ising model with energy E = -J * sum_{<i,j>} s_i * s_j:

1. Pick a random spin s_i.
2. Compute the energy change dE from flipping it.
3. Accept the flip with probability min(1, exp(-dE / (k_B * T))).

On lattices with even side lengths a sweep updates the two checkerboard
sublattices in turn: spins of one colour do not interact, so all of them
can be proposed simultaneously while keeping detailed balance. Odd
lattices use N random single-site proposals per sweep. The whole chain
is one compiled loop, and `sweep_temperatures` runs all temperatures as
independent chains in one `jax.vmap`-ed call.

### Wolff Cluster Algorithm

Near the critical temperature T_c, single-spin Metropolis suffers
from critical slowing down.  The Wolff algorithm builds clusters of
aligned spins and flips them collectively:

1. Pick a random seed spin.
2. Activate every satisfied bond independently with probability
   p = 1 - exp(-2J / (k_B * T)) (each bond is sampled exactly once).
3. Grow the cluster of active bonds containing the seed to convergence.
4. Flip all spins in the cluster.

This dramatically reduces autocorrelation times near T_c.

---

## 7. Fluids

### Lattice Boltzmann (D2Q9, BGK)

`LBMGrid` streams and collides nine populations per node with relaxation
time tau = 3 nu + 1/2. The x direction is a channel with a Zou-He velocity
inlet and a zero-gradient outlet; the y edges are periodic, no-slip
(full-way bounce-back) or free-slip (specular reflection). A no-slip
channel develops the parabolic Poiseuille profile.

### Vorticity-streamfunction Navier-Stokes

`NavierStokesSolver` advances the vorticity with explicit Euler, solves
laplacian(psi) = -omega with Jacobi iterations, and imposes Thom's wall
vorticity for the lid-driven cavity.

### Weakly compressible SPH

`SPHFluid` uses the 2D cubic spline kernel, summation density, the Tait
equation of state p = B((rho/rho0)^gamma - 1), the symmetric pressure
force (exact momentum conservation), Monaghan's artificial viscosity and
kick-drift-kick leapfrog. Neighbours come from a cell list rebuilt every
step. A standing sound wave oscillates with the analytic period L/c0.

### Compressible Euler (1D)

`solve_euler_1d` is a conservative finite-volume scheme: MUSCL (minmod)
reconstruction of the primitive variables, HLLC fluxes and SSP-RK2 time
stepping. It reproduces the exact Sod shock-tube solution and conserves
mass, momentum and energy to round-off with periodic or reflective
boundaries.

---

## 8. Tight-Binding Models

`TightBinding` stores on-site energies and hoppings t_ij(R) between
orbital i in the home cell and orbital j in the cell at lattice vector R.
The Bloch Hamiltonian

    H_ij(k) = eps_i delta_ij + sum_R t_ij(R) exp(i k.R) + h.c.

is diagonalized per k (vmapped). Built-in models: chain
(E = eps - 2t cos ka), square lattice and honeycomb (graphene, with Dirac
points at the zone corners). `finite_hamiltonian` builds open or
periodic real-space supercells. Models are pytrees, so band energies are
differentiable with respect to the hoppings.

---

## 9. Coupled Oscillators and Normal Modes

For N identical masses connected by springs (stiffness k, mass m)
with a fixed wall on the left and a free end on the right, the
potential energy is:

    V = (1/2) k q_0^2 + sum_{i=1}^{N-1} (1/2) k (q_i - q_{i-1})^2

The normal-mode frequencies are:

    omega_j = 2 * sqrt(k/m) * sin((2j - 1) * pi / (4N + 2))

for j = 1, 2, ..., N.

The `coupled_oscillators(n, k, m)` function builds the Lagrangian
automatically, and `normal_mode_frequencies(n, k, m)` returns the
analytical eigenfrequencies.

---

## References

- Goldstein, Poole, Safko. "Classical Mechanics", 3rd ed. (2002)
- Arnold. "Mathematical Methods of Classical Mechanics" (1989)
- Hairer, Lubich, Wanner. "Geometric Numerical Integration" (2006)
- Yoshida. "Construction of higher order symplectic integrators",
  Physics Letters A 150(5-7), 262-268 (1990)
- Dormand, Prince. "A family of embedded Runge-Kutta formulae",
  J. Comput. Appl. Math. 6(1), 19-26 (1980)
- Taflove, Hagness. "Computational Electrodynamics: The Finite-
  Difference Time-Domain Method" (2005)
- Newman. "Monte Carlo Methods in Statistical Physics" (1999)
- Berenger. "A perfectly matched layer for the absorption of
  electromagnetic waves", J. Comput. Phys. 114, 185-200 (1994)
- Monaghan. "Smoothed particle hydrodynamics", Annu. Rev. Astron.
  Astrophys. 30, 543-574 (1992)
- Toro. "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009)
- Kruger et al. "The Lattice Boltzmann Method" (2017)
