"""Fluid dynamics module.

Provides Lattice Boltzmann and vorticity-streamfunction Navier-Stokes
solvers for 2D incompressible flow, weakly compressible SPH, and a
finite-volume solver for the 1D compressible Euler equations.
"""

from jaxphys.fluids.euler import EulerResult, solve_euler_1d
from jaxphys.fluids.lbm import D2Q9, LBMGrid
from jaxphys.fluids.navier_stokes import NavierStokesSolver
from jaxphys.fluids.sph import SPHFluid, SPHTrajectory

__all__ = [
    "D2Q9",
    "LBMGrid",
    "NavierStokesSolver",
    "SPHFluid",
    "SPHTrajectory",
    "solve_euler_1d",
    "EulerResult",
]
