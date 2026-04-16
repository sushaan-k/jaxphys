# Mathematical Foundations

This document explains the mathematical framework behind neurosim's
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

neurosim derives M, dL/dq, and the mixed Hessian automatically using
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

neurosim derives the right-hand sides via `jax.grad`:

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

The step is accepted if the scaled error norm is below 1.  The step
size is then updated:

    h_new = h * safety * err_norm^{-1/5}   (accepted)
    h_new = h * safety * err_norm^{-1/4}   (rejected)

This gives high accuracy with fewer total function evaluations than
a fixed-step method at the same tolerance.

---

## 4. Split-Operator Method (Quantum)

For the time-dependent Schrodinger equation:

    i * hbar * d|psi>/dt = H |psi>

with H = T + V (kinetic + potential), neurosim uses the split-operator
(Strang splitting) approach:

    |psi(t + dt)> = exp(-i V dt/2) * F^{-1}[ exp(-i T_k dt) * F[ exp(-i V dt/2) |psi(t)> ] ]

where F denotes the Fourier transform and T_k = hbar^2 k^2 / (2m) is
the kinetic energy in momentum space.

This is second-order accurate in dt and exactly unitary, so
probability is conserved to machine precision.

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

Absorbing boundary conditions use a Perfectly Matched Layer (PML) to
prevent reflections from the grid edges.

---

## 6. Monte Carlo Methods (Statistical Mechanics)

### Metropolis Algorithm

For the Ising model with energy E = -J * sum_{<i,j>} s_i * s_j:

1. Pick a random spin s_i.
2. Compute the energy change dE from flipping it.
3. Accept the flip with probability min(1, exp(-dE / (k_B * T))).

### Wolff Cluster Algorithm

Near the critical temperature T_c, single-spin Metropolis suffers
from critical slowing down.  The Wolff algorithm builds clusters of
aligned spins and flips them collectively:

1. Pick a random seed spin.
2. Add each aligned neighbor with probability p = 1 - exp(-2J / (k_B * T)).
3. Recursively grow the cluster.
4. Flip all spins in the cluster.

This dramatically reduces autocorrelation times near T_c.

---

## 7. Coupled Oscillators and Normal Modes

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
