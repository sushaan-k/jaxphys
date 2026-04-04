"""Electromagnetism module.

Provides FDTD Maxwell solver (2D and 3D), charge dynamics, and waveguide simulation.
"""

from neurosim.em.charges import ChargeSystem, PointCharge
from neurosim.em.fdtd import EMGrid, PlaneWave, Wall
from neurosim.em.fdtd3d import DielectricRegion, EMGrid3D, PointSource3D
from neurosim.em.waveguides import RectangularWaveguide

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
