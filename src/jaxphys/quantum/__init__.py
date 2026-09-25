"""Quantum mechanics module.

Provides time-dependent and time-independent Schrodinger equation solvers,
spin chain dynamics, and open quantum system simulation via density matrices.
"""

from jaxphys.quantum.density_matrix import DensityMatrix, lindblad_evolve
from jaxphys.quantum.schrodinger import (
    GaussianWavepacket,
    SquareBarrier,
    solve_schrodinger,
)
from jaxphys.quantum.spin import SpinChain
from jaxphys.quantum.stationary import solve_eigenvalue_problem

__all__ = [
    "solve_schrodinger",
    "GaussianWavepacket",
    "SquareBarrier",
    "solve_eigenvalue_problem",
    "SpinChain",
    "DensityMatrix",
    "lindblad_evolve",
]
