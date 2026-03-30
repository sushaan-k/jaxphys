# neurosim

## GPU-Accelerated Differentiable Physics Engine

### The Problem

Physics simulation libraries fall into two camps:
1. **Research-grade** (FEniCS, OpenFOAM, COMSOL) — powerful but massive, C++/Fortran, impossible to install, not differentiable
2. **Educational** (VPython, PhysicsJS) — toy-level, CPU-only, not useful for real computation

There's a huge gap for a **modern, GPU-accelerated, differentiable physics library in Python** that's actually usable for:
- Solving olympiad and research-level physics problems computationally
- Running parameter sweeps and optimization (because it's differentiable)
- Visualizing results beautifully
- Teaching physics through interactive computation

JAX made differentiable programming accessible. Nobody has built a serious physics library on top of it that covers classical mechanics, E&M, quantum, and statistical mechanics with a clean API.

### The Solution

`neurosim` is a JAX-based differentiable physics engine for simulating systems across classical mechanics, electromagnetism, quantum mechanics, and statistical mechanics — with GPU acceleration and automatic differentiation built in.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                      neurosim                         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │                  Physics Modules                 │  │
│  │                                                  │  │
│  │  ┌────────────┐ ┌────────────┐ ┌─────────────┐  │  │
│  │  │ Classical  │ │ E&M        │ │ Quantum     │  │  │
│  │  │ Mechanics  │ │            │ │ Mechanics   │  │  │
│  │  │            │ │ - Maxwell  │ │             │  │  │
│  │  │ - Lagrange │ │   solver   │ │ - Schröding │  │  │
│  │  │ - Hamilton │ │ - FDTD     │ │   er solver │  │  │
│  │  │ - N-body  │ │ - Charge   │ │ - Density   │  │  │
│  │  │ - Rigid   │ │   dynamics │ │   matrices  │  │  │
│  │  │   body    │ │ - Wave     │ │ - Spin      │  │  │
│  │  │           │ │   guides   │ │   chains    │  │  │
│  │  └────────────┘ └────────────┘ └─────────────┘  │  │
│  │                                                  │  │
│  │  ┌────────────┐ ┌────────────┐                   │  │
│  │  │ StatMech   │ │ Optics     │                   │  │
│  │  │            │ │            │                   │  │
│  │  │ - Ising    │ │ - Ray      │                   │  │
│  │  │ - Boltz-   │ │   tracing  │                   │  │
│  │  │   mann     │ │ - Diffrac- │                   │  │
│  │  │ - Monte    │ │   tion     │                   │  │
│  │  │   Carlo    │ │ - Interfer │                   │  │
│  │  └────────────┘ └────────────┘                   │  │
│  └─────────────────────────────────────────────────┘  │
│                          │                             │
│                          ▼                             │
│  ┌─────────────────────────────────────────────────┐  │
│  │               JAX Backend                        │  │
│  │                                                  │  │
│  │  - Automatic differentiation (jax.grad)          │  │
│  │  - GPU/TPU acceleration (jax.jit)                │  │
│  │  - Vectorized simulation (jax.vmap)              │  │
│  │  - Parallel parameter sweeps (jax.pmap)          │  │
│  └─────────────────────────────────────────────────┘  │
│                          │                             │
│                          ▼                             │
│  ┌─────────────────────────────────────────────────┐  │
│  │             Visualization Layer                   │  │
│  │                                                  │  │
│  │  - Real-time 3D rendering (matplotlib / plotly)  │  │
│  │  - Animation export (mp4, gif)                   │  │
│  │  - Interactive exploration                        │  │
│  │  - Phase space plots                             │  │
│  │  - Field visualizations (vector fields, contour) │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Core Modules

#### 1. Classical Mechanics

**Lagrangian Mechanics Engine**:
Define a system by its Lagrangian, and neurosim derives the equations of motion automatically using JAX's autodiff:

```python
import neurosim as ns
import jax.numpy as jnp

# Double pendulum — define the Lagrangian, neurosim does the rest
def lagrangian(q, qdot, params):
    theta1, theta2 = q
    omega1, omega2 = qdot
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    T = (0.5 * m1 * (l1 * omega1)**2 +
         0.5 * m2 * ((l1 * omega1)**2 + (l2 * omega2)**2 +
         2 * l1 * l2 * omega1 * omega2 * jnp.cos(theta1 - theta2)))
    V = (-(m1 + m2) * g * l1 * jnp.cos(theta1) -
         m2 * g * l2 * jnp.cos(theta2))
    return T - V

system = ns.LagrangianSystem(lagrangian, n_dof=2)
params = ns.Params(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)

# Simulate
trajectory = system.simulate(
    q0=[jnp.pi/4, jnp.pi/2],     # initial angles
    qdot0=[0.0, 0.0],              # initial velocities
    t_span=(0, 30),                # 30 seconds
    dt=0.001,                      # 1ms timestep
    params=params,
    integrator="rk4",              # non-symplectic; best for Lagrangian systems
)

# Visualize
ns.animate_pendulum(trajectory, save="double_pendulum.mp4")
ns.plot_phase_space(trajectory, coords=[0, 1])
ns.plot_energy(trajectory)  # verify energy conservation
```

**N-Body Simulator**:
```python
# Gravitational N-body problem — GPU accelerated
system = ns.NBody(
    masses=[1.0, 0.001, 0.0003],  # Sun, Jupiter-like, Earth-like
    positions=[[0,0,0], [5.2,0,0], [1,0,0]],
    velocities=[[0,0,0], [0,2.75,0], [0,6.28,0]],
    G=1.0,
)

# Simulate 1000 years with 1M timesteps on GPU
trajectory = system.simulate(t_span=(0, 1000), n_steps=1_000_000)
ns.animate_3d(trajectory, trails=True)
```

#### 2. Electromagnetism

**FDTD Maxwell Solver**:
```python
# Simulate electromagnetic wave propagation through a slit
grid = ns.EMGrid(
    size=(200, 200),
    resolution=0.01,        # 1cm cells
    boundary="absorbing",   # PML boundaries
)

# Add a conducting wall with a slit
grid.add_conductor(ns.Wall(y=100, gap_start=90, gap_end=110))

# Add a plane wave source
grid.add_source(ns.PlaneWave(frequency=3e9, y=20))  # 3 GHz

# Simulate
fields = grid.simulate(t_span=(0, 1e-8), dt=1e-11)
ns.animate_field(fields, component="Ez", save="diffraction.mp4")
```

#### 3. Quantum Mechanics

**Schrödinger Equation Solver**:
```python
# Quantum tunneling through a barrier
potential = ns.SquareBarrier(height=5.0, width=1.0, center=10.0)

psi0 = ns.GaussianWavepacket(x0=5.0, k0=3.0, sigma=0.5)

result = ns.solve_schrodinger(
    psi0=psi0,
    potential=potential,
    x_range=(-5, 25),
    t_span=(0, 10),
    n_points=1000,
)

ns.animate_wavefunction(result, show_probability=True, show_potential=True)
print(f"Transmission coefficient: {result.transmission_coefficient:.4f}")
```

#### 4. Statistical Mechanics

**Ising Model with GPU-Accelerated Monte Carlo**:
```python
# 2D Ising model — phase transition
lattice = ns.IsingLattice(size=(256, 256))

# Sweep temperature across the critical point
results = ns.sweep_temperatures(
    lattice,
    temperatures=jnp.linspace(1.0, 4.0, 100),
    n_sweeps=10000,
    algorithm="wolff_cluster",
)

ns.plot_phase_transition(results)  # magnetization vs temperature
ns.plot_specific_heat(results)     # specific heat peak at T_c
```

### The Differentiable Advantage

Because everything runs on JAX, you get automatic differentiation for free:

```python
# Inverse problem: what initial velocity makes a projectile hit a target?
def miss_distance(v0):
    trajectory = ns.projectile(v0=v0, angle=45, g=9.81)
    return (trajectory.final_position - target)**2

# Gradient descent to find optimal v0
optimal_v0 = ns.optimize(miss_distance, initial_guess=10.0)

# Parameter sensitivity: how does changing mass affect the double pendulum?
sensitivity = jax.jacobian(system.simulate)(params)
```

This enables:
- **Inverse problems**: Find parameters that produce desired behavior
- **Sensitivity analysis**: How does changing one parameter affect the whole system?
- **Optimization**: Find optimal configurations (spacecraft trajectories, lens designs, etc.)
- **Neural ODEs**: Combine physics with learned dynamics

### Technical Stack

- **Language**: Python 3.11+
- **Computation**: JAX (GPU/TPU acceleration, autodiff)
- **Numerics**: `jax.numpy`, custom symplectic integrators
- **Visualization**: `matplotlib` (static/animated)

### What Makes This Novel

1. **Differentiable physics** — not just simulation, but gradients through the simulation (inverse problems, optimization)
2. **GPU-accelerated** — 1000x speedup over CPU for large systems (N-body, FDTD, Monte Carlo)
3. **USAPhO-level problems as code** — your physics competition background expressed as engineering
4. **Lagrangian mechanics from first principles** — define L, get equations of motion automatically
5. **Cross-domain** — classical, E&M, quantum, stat mech in one library with a unified API

### Repo Structure

```
neurosim/
├── README.md
├── pyproject.toml
├── src/
│   └── neurosim/
│       ├── __init__.py
│       ├── classical/
│       │   ├── lagrangian.py       # Lagrangian mechanics
│       │   ├── hamiltonian.py      # Hamiltonian mechanics
│       │   ├── nbody.py            # N-body simulation
│       │   ├── rigid_body.py       # Rigid body dynamics
│       │   └── integrators.py      # Symplectic integrators
│       ├── em/
│       │   ├── fdtd.py             # FDTD Maxwell solver
│       │   ├── charges.py          # Charge dynamics
│       │   └── waveguides.py       # Waveguide simulation
│       ├── quantum/
│       │   ├── schrodinger.py      # Time-dependent Schrödinger
│       │   ├── stationary.py       # Time-independent (eigenvalues)
│       │   ├── spin.py             # Spin chain dynamics
│       │   └── density_matrix.py   # Open quantum systems
│       ├── statmech/
│       │   ├── ising.py            # Ising model
│       │   ├── monte_carlo.py      # MC methods
│       │   └── boltzmann.py        # Boltzmann distribution
│       ├── optics/
│       │   ├── ray_tracing.py      # Geometric optics
│       │   └── diffraction.py      # Wave optics
│       ├── viz/
│       │   ├── animate.py          # Animation utilities
│       │   ├── fields.py           # Field visualization
│       │   ├── phase_space.py      # Phase space plots
│       │   └── interactive.py      # Interactive visualization
│       └── optimize.py             # Inverse problems, optimization
├── tests/
├── examples/
│   ├── double_pendulum.py
│   ├── three_body.py
│   ├── quantum_tunneling.py
│   ├── em_diffraction.py
│   ├── ising_phase_transition.py
│   └── spacecraft_trajectory.py
├── notebooks/
│   ├── classical_mechanics.ipynb
│   ├── electromagnetism.ipynb
│   └── quantum_mechanics.ipynb
└── docs/
    ├── physics.md
    ├── differentiable.md
    └── gpu_acceleration.md
```

### Research References

- JAX documentation (jax.readthedocs.io)
- "Differentiable Physics Simulation" (NeurIPS 2020, foundational)
- "Neural ODEs" (Chen et al., 2018)
- Feynman Lectures on Physics (pedagogical design inspiration)
- USAPhO past problems (problem set inspiration)
- "Hamiltonian Neural Networks" (Greydanus et al., 2019)
