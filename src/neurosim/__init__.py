"""neurosim: GPU-accelerated differentiable physics engine.

A JAX-based physics simulation library covering classical mechanics,
electromagnetism, quantum mechanics, and statistical mechanics with
automatic differentiation and GPU acceleration.

Example:
    >>> import neurosim as ns
    >>> import jax.numpy as jnp
    >>> def lagrangian(q, qdot, params):
    ...     T = 0.5 * params.m * (params.l * qdot[0])**2
    ...     V = -params.m * params.g * params.l * jnp.cos(q[0])
    ...     return T - V
    >>> system = ns.LagrangianSystem(lagrangian, n_dof=1)
    >>> params = ns.Params(m=1.0, g=9.81, l=1.0)
    >>> traj = system.simulate(q0=[0.3], qdot0=[0.0],
    ...     t_span=(0, 10), dt=0.01, params=params)
"""

# Enforce 64-bit precision. JAX defaults to float32 which silently
# truncates the float64 dtypes used throughout this library.
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

__version__ = "0.1.0"

# Core configuration
from neurosim.classical.hamiltonian import HamiltonianSystem
from neurosim.classical.integrators import (
    euler,
    leapfrog,
    rk4,
    stormer_verlet,
    symplectic_euler,
    velocity_verlet,
    yoshida4,
)

# Classical mechanics
from neurosim.classical.lagrangian import LagrangianSystem
from neurosim.classical.nbody import NBody
from neurosim.classical.rigid_body import RigidBody
from neurosim.config import (
    EMConfig,
    FluidConfig,
    IsingConfig,
    NBodyConfig,
    Params,
    QuantumConfig,
    SimulationConfig,
)
from neurosim.em.charges import ChargeSystem, PointCharge

# Electromagnetism
from neurosim.em.fdtd import EMGrid, PlaneWave, Wall
from neurosim.em.fdtd3d import DielectricRegion, EMGrid3D, PointSource3D
from neurosim.em.waveguides import RectangularWaveguide

# Fluid dynamics
from neurosim.fluids.lbm import D2Q9, LBMGrid, Obstacle
from neurosim.fluids.navier_stokes import NavierStokesSolver

# Exceptions
from neurosim.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DimensionError,
    NeurosimError,
    NumericalInstabilityError,
    PhysicsError,
    SimulationError,
    VisualizationError,
)
from neurosim.optics.diffraction import (
    circular_aperture,
    double_slit,
    single_slit,
)

# Optics
from neurosim.optics.ray_tracing import (
    FlatMirror,
    Ray,
    SphericalMirror,
    ThinLens,
    trace_system,
)

# Optimization
from neurosim.optimize import optimize, projectile, sensitivity
from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve

# Quantum mechanics
from neurosim.quantum.schrodinger import (
    DoubleWellPotential,
    GaussianWavepacket,
    HarmonicPotential,
    SquareBarrier,
    solve_schrodinger,
)
from neurosim.quantum.spin import SpinChain
from neurosim.quantum.stationary import solve_eigenvalue_problem

# State representations
from neurosim.state import (
    EMFieldHistory,
    EMFieldHistory3D,
    EMFieldState,
    FluidHistory,
    FluidState,
    IsingResult,
    NBodyState,
    NBodyTrajectory,
    PhaseState,
    QuantumResult,
    QuantumState,
    Trajectory,
)
from neurosim.statmech.boltzmann import (
    boltzmann_distribution,
    partition_function,
)

# Statistical mechanics
from neurosim.statmech.ising import (
    IsingLattice,
    sweep_temperatures,
    vmap_temperatures,
)

# Visualization (lazy import — only loaded if matplotlib available)
try:
    from neurosim.viz.animate import (
        animate_3d,
        animate_pendulum,
        animate_wavefunction,
    )
    from neurosim.viz.fields import animate_field, plot_field_snapshot
    from neurosim.viz.phase_space import (
        plot_energy,
        plot_phase_space,
        plot_phase_transition,
        plot_specific_heat,
    )
except ImportError:
    pass

__all__ = [
    # Version
    "__version__",
    # Config
    "Params",
    "SimulationConfig",
    "NBodyConfig",
    "EMConfig",
    "FluidConfig",
    "QuantumConfig",
    "IsingConfig",
    # State
    "PhaseState",
    "Trajectory",
    "NBodyState",
    "NBodyTrajectory",
    "EMFieldState",
    "EMFieldHistory",
    "EMFieldHistory3D",
    "FluidState",
    "FluidHistory",
    "QuantumState",
    "QuantumResult",
    "IsingResult",
    # Exceptions
    "NeurosimError",
    "SimulationError",
    "NumericalInstabilityError",
    "ConfigurationError",
    "DimensionError",
    "PhysicsError",
    "ConvergenceError",
    "VisualizationError",
    # Classical
    "LagrangianSystem",
    "HamiltonianSystem",
    "NBody",
    "RigidBody",
    "euler",
    "symplectic_euler",
    "leapfrog",
    "velocity_verlet",
    "stormer_verlet",
    "yoshida4",
    "rk4",
    # EM
    "EMGrid",
    "PlaneWave",
    "Wall",
    "EMGrid3D",
    "PointSource3D",
    "DielectricRegion",
    "PointCharge",
    "ChargeSystem",
    "RectangularWaveguide",
    # Fluids
    "D2Q9",
    "LBMGrid",
    "Obstacle",
    "NavierStokesSolver",
    # Quantum
    "solve_schrodinger",
    "GaussianWavepacket",
    "SquareBarrier",
    "HarmonicPotential",
    "DoubleWellPotential",
    "solve_eigenvalue_problem",
    "SpinChain",
    "DensityMatrix",
    "lindblad_evolve",
    # StatMech
    "IsingLattice",
    "sweep_temperatures",
    "vmap_temperatures",
    "boltzmann_distribution",
    "partition_function",
    # Optics
    "Ray",
    "ThinLens",
    "FlatMirror",
    "SphericalMirror",
    "trace_system",
    "single_slit",
    "double_slit",
    "circular_aperture",
    # Optimization
    "optimize",
    "projectile",
    "sensitivity",
    # Visualization
    "plot_phase_space",
    "plot_energy",
    "plot_phase_transition",
    "plot_specific_heat",
    "animate_pendulum",
    "animate_wavefunction",
    "animate_3d",
    "animate_field",
    "plot_field_snapshot",
]
