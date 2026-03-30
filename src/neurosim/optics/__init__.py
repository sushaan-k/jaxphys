"""Optics module.

Provides geometric ray tracing and wave optics diffraction simulation.
"""

from neurosim.optics.diffraction import (
    circular_aperture,
    double_slit,
    single_slit,
)
from neurosim.optics.ray_tracing import (
    FlatMirror,
    Ray,
    SphericalMirror,
    ThinLens,
    trace_system,
)

__all__ = [
    "Ray",
    "ThinLens",
    "FlatMirror",
    "SphericalMirror",
    "trace_system",
    "single_slit",
    "double_slit",
    "circular_aperture",
]
