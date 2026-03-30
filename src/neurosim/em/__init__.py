"""Electromagnetism module.

Provides FDTD Maxwell solver, charge dynamics, and waveguide simulation.
"""

from neurosim.em.charges import ChargeSystem, PointCharge
from neurosim.em.fdtd import EMGrid, PlaneWave, Wall
from neurosim.em.waveguides import RectangularWaveguide

__all__ = [
    "EMGrid",
    "PlaneWave",
    "Wall",
    "PointCharge",
    "ChargeSystem",
    "RectangularWaveguide",
]
