"""Classical mechanics module.

Provides Lagrangian and Hamiltonian mechanics, N-body simulation,
rigid body dynamics, and symplectic integrators.
"""

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

__all__ = [
    "LagrangianSystem",
    "HamiltonianSystem",
    "NBody",
    "RigidBody",
    "coupled_oscillators",
    "normal_mode_frequencies",
    "euler",
    "symplectic_euler",
    "leapfrog",
    "velocity_verlet",
    "stormer_verlet",
    "yoshida4",
    "rk4",
    "adaptive_rk45",
]
