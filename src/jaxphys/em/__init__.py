"""Electromagnetism module.

Provides FDTD (2D and 3D) and FDFD Maxwell solvers, charge dynamics, and
waveguide analysis.
"""

from jaxphys.em.charges import ChargeSystem, PointCharge
from jaxphys.em.fdfd import FDFDResult, solve_fdfd
from jaxphys.em.fdtd import EMGrid, PlaneWave, Wall
from jaxphys.em.fdtd3d import DielectricRegion, EMGrid3D, PointSource3D
from jaxphys.em.waveguides import RectangularWaveguide

__all__ = [
    "EMGrid",
    "PlaneWave",
    "Wall",
    "EMGrid3D",
    "PointSource3D",
    "DielectricRegion",
    "solve_fdfd",
    "FDFDResult",
    "PointCharge",
    "ChargeSystem",
    "RectangularWaveguide",
]
