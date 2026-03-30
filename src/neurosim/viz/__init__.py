"""Visualization module.

Provides plotting and animation utilities for physics simulations.
Requires the 'viz' optional dependency group (matplotlib, plotly).
"""

from neurosim.viz.animate import animate_pendulum, animate_wavefunction
from neurosim.viz.fields import animate_field, plot_field_snapshot
from neurosim.viz.phase_space import plot_energy, plot_phase_space

__all__ = [
    "plot_phase_space",
    "plot_energy",
    "animate_pendulum",
    "animate_wavefunction",
    "animate_field",
    "plot_field_snapshot",
]
