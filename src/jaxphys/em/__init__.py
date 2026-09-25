"""Electromagnetism module.

Provides FDTD Maxwell solver (2D and 3D), charge dynamics, and waveguide simulation.
"""

from jaxphys.em.charges import ChargeSystem, PointCharge
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
    "PointCharge",
    "ChargeSystem",
    "RectangularWaveguide",
]
