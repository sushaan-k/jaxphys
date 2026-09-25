"""Statistical mechanics module.

Provides Ising model simulation, Monte Carlo methods,
and Boltzmann distribution utilities.
"""

from jaxphys.statmech.boltzmann import (
    boltzmann_distribution,
    partition_function,
)
from jaxphys.statmech.ising import (
    IsingLattice,
    sweep_temperatures,
    vmap_temperatures,
)
from jaxphys.statmech.monte_carlo import metropolis_step, wolff_step

__all__ = [
    "IsingLattice",
    "sweep_temperatures",
    "vmap_temperatures",
    "metropolis_step",
    "wolff_step",
    "boltzmann_distribution",
    "partition_function",
]
