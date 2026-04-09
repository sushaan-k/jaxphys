# jaxphys

[![CI](https://github.com/sushaan-k/jaxphys/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/jaxphys/actions)
[![PyPI](https://img.shields.io/pypi/v/jaxphys.svg)](https://pypi.org/project/jaxphys/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/jaxphys.svg)](https://pypi.org/project/jaxphys/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Differentiable JAX physics engine — classical mechanics, quantum, EM, and fluid dynamics in one API.**

`jaxphys` is a fully differentiable physics simulator built on JAX. Every simulation is end-to-end differentiable: you can `jax.grad` through a rigid body trajectory, `jax.vmap` over parameter ensembles, and `jax.jit` compile entire simulation loops to XLA — including your loss function.

---

## The Problem

Physics simulation for ML is fragmented. MuJoCo handles rigid bodies. Schrödinger solvers are standalone scripts. FDTD codes are Fortran. None of them are differentiable end-to-end in the same framework, which means you can't jointly optimize across physics domains. Robotics, materials science, and scientific ML all need this — and nobody has built the unified differentiable layer.

## Solution

```python
import jax.numpy as jnp
from jaxphys import System, RigidBody, Gravity, simulate

# Define a double pendulum system
system = System([
    RigidBody(mass=1.0, length=1.0, q0=jnp.array([jnp.pi/4, 0.0])),
    RigidBody(mass=1.0, length=1.0, q0=jnp.array([jnp.pi/3, 0.0])),
], forces=[Gravity(g=9.81)])

# Simulate — returns full trajectory, fully differentiable
traj = simulate(system, dt=0.01, steps=1000)

# Gradient of final kinetic energy w.r.t. initial angle
import jax
grad_fn = jax.grad(lambda q0: simulate(system.replace_q0(q0), dt=0.01, steps=1000).ke[-1])
dKE_dtheta = grad_fn(jnp.array([jnp.pi/4, 0.0]))
```

## At a Glance

- **Classical mechanics** — rigid bodies, constraints, Hamiltonian integration
- **Quantum** — 1D/2D Schrödinger solver, tight-binding models
- **Electromagnetics** — FDTD solver, waveguide propagation
- **Fluid dynamics** — SPH and Euler fluid on GPU
- **Fully differentiable** — `jax.grad`, `jax.vmap`, `jax.jit` work through everything
- **XLA-compiled** — 100x–1000x faster than NumPy equivalents on GPU/TPU

## Install

```bash
pip install jaxphys
# GPU support:
pip install jaxphys "jax[cuda12]"
```

## Benchmark

| Simulation | NumPy | PyTorch | **jaxphys (JIT)** |
|---|---|---|---|
| N-body (N=1000, 1000 steps) | 4.2s | 0.8s | **0.04s** |
| Schrödinger 2D (256×256) | 11.3s | 2.1s | **0.09s** |
| SPH fluid (5000 particles) | 18.7s | 3.4s | **0.18s** |

## Architecture

```
System
 ├── RigidBodySolver   # symplectic Euler / Runge-Kutta integrators
 ├── QuantumSolver     # split-operator Schrödinger, tight-binding
 ├── EMSolver          # FDTD with PML boundary conditions
 ├── FluidSolver       # SPH + Euler finite difference
 └── Renderer          # matplotlib / manim visualization helpers
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐
