"""Fluid dynamics module.

Provides Lattice Boltzmann and vorticity-streamfunction Navier-Stokes
solvers for 2D incompressible flow simulation.
"""

from neurosim.fluids.lbm import D2Q9, LBMGrid
from neurosim.fluids.navier_stokes import NavierStokesSolver

__all__ = [
    "D2Q9",
    "LBMGrid",
    "NavierStokesSolver",
]
