"""Quantum mechanics module.

Provides time-dependent and time-independent Schrodinger equation solvers,
spin chain dynamics, and open quantum system simulation via density matrices.
"""

from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve
from neurosim.quantum.schrodinger import (
    GaussianWavepacket,
    SquareBarrier,
    solve_schrodinger,
)
from neurosim.quantum.spin import SpinChain
from neurosim.quantum.stationary import solve_eigenvalue_problem

__all__ = [
    "solve_schrodinger",
    "GaussianWavepacket",
    "SquareBarrier",
    "solve_eigenvalue_problem",
    "SpinChain",
    "DensityMatrix",
    "lindblad_evolve",
]
