"""jaxphys: GPU-accelerated differentiable physics engine.

A JAX-based physics simulation library covering classical mechanics,
electromagnetism, quantum mechanics, and statistical mechanics with
automatic differentiation and GPU acceleration.

Example:
    >>> import jaxphys as jp
    >>> import jax.numpy as jnp
    >>> def lagrangian(q, qdot, params):
    ...     T = 0.5 * params.m * (params.l * qdot[0])**2
    ...     V = -params.m * params.g * params.l * jnp.cos(q[0])
    ...     return T - V
    >>> system = jp.LagrangianSystem(lagrangian, n_dof=1)
    >>> params = jp.Params(m=1.0, g=9.81, l=1.0)
    >>> traj = system.simulate(q0=[0.3], qdot0=[0.0],
    ...     t_span=(0, 10), dt=0.01, params=params)
"""

# Enforce 64-bit precision. JAX defaults to float32 which silently
# truncates the float64 dtypes used throughout this library.
from importlib.metadata import PackageNotFoundError, version

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

try:
    __version__ = version("jaxphys")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

# Core configuration
# Classical mechanics
from jaxphys.classical.coupled_oscillators import (
    coupled_oscillators,
    normal_mode_frequencies,
)
from jaxphys.classical.hamiltonian import HamiltonianSystem
from jaxphys.classical.integrators import (
    adaptive_rk45,
    euler,
    leapfrog,
    rk4,
    stormer_verlet,
    symplectic_euler,
    velocity_verlet,
    yoshida4,
)
from jaxphys.classical.lagrangian import LagrangianSystem
from jaxphys.classical.nbody import NBody
from jaxphys.classical.rigid_body import RigidBody
from jaxphys.config import (
    EMConfig,
    FluidConfig,
    IsingConfig,
    NBodyConfig,
    Params,
    QuantumConfig,
    SimulationConfig,
)
from jaxphys.em.charges import ChargeSystem, PointCharge

# Electromagnetism
from jaxphys.em.fdtd import EMGrid, PlaneWave, Wall
from jaxphys.em.fdtd3d import DielectricRegion, EMGrid3D, PointSource3D
from jaxphys.em.waveguides import RectangularWaveguide

# Exceptions
from jaxphys.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DimensionError,
    JaxphysError,
    NumericalInstabilityError,
    PhysicsError,
    SimulationError,
    VisualizationError,
)

# Fluid dynamics
from jaxphys.fluids.lbm import D2Q9, LBMGrid, Obstacle
from jaxphys.fluids.navier_stokes import NavierStokesSolver
from jaxphys.optics.diffraction import (
    circular_aperture,
    double_slit,
    single_slit,
)

# Optics
from jaxphys.optics.ray_tracing import (
    FlatMirror,
    Ray,
    SphericalMirror,
    ThinLens,
    trace_system,
)

# Optimization
from jaxphys.optimize import (
    ParameterGrid,
    ParameterSweepResult,
    RefinedSweepCandidate,
    SweepRefinementResult,
    make_parameter_grid,
    optimize,
    parameter_sweep,
    projectile,
    refine_parameter_sweep,
    sensitivity,
)
from jaxphys.quantum.density_matrix import DensityMatrix, lindblad_evolve

# Quantum mechanics
from jaxphys.quantum.schrodinger import (
    DoubleWellPotential,
    GaussianWavepacket,
    HarmonicPotential,
    SquareBarrier,
    solve_schrodinger,
)
from jaxphys.quantum.spin import SpinChain
from jaxphys.quantum.stationary import solve_eigenvalue_problem

# State representations
from jaxphys.state import (
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
from jaxphys.statmech.boltzmann import (
    boltzmann_distribution,
    partition_function,
)

# Statistical mechanics
from jaxphys.statmech.ising import (
    IsingLattice,
    sweep_temperatures,
    vmap_temperatures,
)

# Visualization (lazy import — only loaded if matplotlib available)
try:
    from jaxphys.viz.animate import (
        animate_3d,
        animate_pendulum,
        animate_wavefunction,
    )
    from jaxphys.viz.fields import animate_field, plot_field_snapshot
    from jaxphys.viz.phase_space import (
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
    "JaxphysError",
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
    "adaptive_rk45",
    "coupled_oscillators",
    "normal_mode_frequencies",
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
    "ParameterGrid",
    "ParameterSweepResult",
    "RefinedSweepCandidate",
    "SweepRefinementResult",
    "make_parameter_grid",
    "optimize",
    "parameter_sweep",
    "projectile",
    "refine_parameter_sweep",
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
