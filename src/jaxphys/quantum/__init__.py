"""Quantum mechanics module.

Provides time-dependent and time-independent Schrodinger equation solvers,
tight-binding band structures, spin chain dynamics, and open quantum system
simulation via density matrices.
"""

from jaxphys.quantum.density_matrix import DensityMatrix, lindblad_evolve
from jaxphys.quantum.schrodinger import (
    GaussianWavepacket,
    GaussianWavepacket2D,
    SquareBarrier,
    solve_schrodinger,
    solve_schrodinger_2d,
)
from jaxphys.quantum.spin import SpinChain
from jaxphys.quantum.stationary import solve_eigenvalue_problem
from jaxphys.quantum.tight_binding import TightBinding, k_path

__all__ = [
    "solve_schrodinger",
    "solve_schrodinger_2d",
    "GaussianWavepacket",
    "GaussianWavepacket2D",
    "SquareBarrier",
    "solve_eigenvalue_problem",
    "SpinChain",
    "TightBinding",
    "k_path",
    "DensityMatrix",
    "lindblad_evolve",
]
