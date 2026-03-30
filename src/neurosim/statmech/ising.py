"""2D Ising model simulation.

Implements the 2D square-lattice Ising model with Metropolis and
Wolff cluster Monte Carlo algorithms. The Hamiltonian is:

    H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i

where s_i in {-1, +1} and the first sum runs over nearest-neighbor
pairs on a 2D square lattice with periodic boundary conditions.

The exact critical temperature (Onsager, 1944) for h=0 is:
    T_c = 2J / ln(1 + sqrt(2)) ~ 2.269 J/kB

References:
    - Onsager. "Crystal statistics I" (1944)
    - Wolff. "Collective Monte Carlo updating for spin systems" (1989)
    - Newman & Barkema. "Monte Carlo Methods in Statistical Physics" (1999)
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import IsingConfig
from neurosim.exceptions import ConfigurationError
from neurosim.state import IsingResult
from neurosim.statmech.monte_carlo import wolff_step

logger = logging.getLogger(__name__)

# Onsager critical temperature for J=1
T_CRITICAL = 2.0 / jnp.log(1.0 + jnp.sqrt(2.0))


class IsingLattice:
    """2D Ising model on a square lattice.

    Supports Metropolis single-spin-flip and Wolff cluster updates.
    All operations are JIT-compiled for GPU acceleration.

    Example:
        >>> lattice = IsingLattice(size=(64, 64))
        >>> result = lattice.run_metropolis(
        ...     temperature=2.269, n_sweeps=10000,
        ...     key=jax.random.PRNGKey(0),
        ... )

    Args:
        size: Lattice dimensions (Lx, Ly).
        J: Coupling constant. Positive = ferromagnetic.
        h: External magnetic field.
    """

    def __init__(
        self,
        size: tuple[int, int] = (64, 64),
        J: float = 1.0,
        h: float = 0.0,
    ) -> None:
        if size[0] < 2 or size[1] < 2:
            raise ConfigurationError(f"Lattice must be at least 2x2, got {size}")
        self._Lx, self._Ly = size
        self._config = IsingConfig(J=J, h=h)

    @property
    def size(self) -> tuple[int, int]:
        """Lattice dimensions."""
        return (self._Lx, self._Ly)

    @property
    def n_spins(self) -> int:
        """Total number of spins."""
        return self._Lx * self._Ly

    def random_state(self, key: Array) -> Array:
        """Generate a random spin configuration.

        Args:
            key: JAX PRNG key.

        Returns:
            Spin array of +1/-1, shape (Lx, Ly).
        """
        return (
            2 * jax.random.bernoulli(key, shape=(self._Lx, self._Ly)).astype(jnp.int32)
            - 1
        )

    def _compute_energy(self, spins: Array) -> float:
        """Compute total energy of the spin configuration.

        Args:
            spins: Spin array, shape (Lx, Ly).

        Returns:
            Total energy.
        """
        J = self._config.J
        h = self._config.h
        # Nearest-neighbor interaction with periodic BC
        energy = -J * jnp.sum(
            spins * jnp.roll(spins, 1, axis=0) + spins * jnp.roll(spins, 1, axis=1)
        )
        energy = energy - h * jnp.sum(spins)
        return float(energy)

    def _compute_magnetization(self, spins: Array) -> float:
        """Compute magnetization per spin.

        Args:
            spins: Spin array, shape (Lx, Ly).

        Returns:
            Absolute magnetization per spin.
        """
        return float(jnp.abs(jnp.mean(spins)))

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def _metropolis_sweep(
        spins: Array,
        key: Array,
        temperature: float,
        Lx: int,
        Ly: int,
        J: float,
        h: float,
    ) -> tuple[Array, Array]:
        """Perform one full Metropolis sweep over all spins.

        A sweep consists of N = Lx * Ly single-spin-flip proposals.

        Args:
            spins: Current spin configuration.
            key: PRNG key.
            temperature: Temperature (in units of J/kB).
            Lx: Lattice size x.
            Ly: Lattice size y.
            J: Coupling constant.
            h: External field.

        Returns:
            (new_spins, new_key).
        """
        beta = 1.0 / temperature
        n_spins = Lx * Ly

        def single_flip(
            carry: tuple[Array, Array], _: None
        ) -> tuple[tuple[Array, Array], None]:
            s, k = carry
            k, k1, k2, k3 = jax.random.split(k, 4)

            # Random site
            ix = jax.random.randint(k1, (), 0, Lx)
            iy = jax.random.randint(k2, (), 0, Ly)

            # Local field from neighbors (periodic BC)
            nn_sum = (
                s[(ix + 1) % Lx, iy]
                + s[(ix - 1) % Lx, iy]
                + s[ix, (iy + 1) % Ly]
                + s[ix, (iy - 1) % Ly]
            )

            # Energy change from flipping spin at (ix, iy)
            dE = 2.0 * J * s[ix, iy] * nn_sum + 2.0 * h * s[ix, iy]

            # Metropolis acceptance
            accept = (dE < 0) | (jax.random.uniform(k3) < jnp.exp(-beta * dE))
            new_spin = jnp.where(accept, -s[ix, iy], s[ix, iy])
            s = s.at[ix, iy].set(new_spin)

            return (s, k), None

        (spins_new, key_new), _ = jax.lax.scan(
            single_flip, (spins, key), None, length=n_spins
        )
        return spins_new, key_new

    def run_metropolis(
        self,
        temperature: float,
        n_sweeps: int = 10000,
        n_warmup: int = 1000,
        key: Array | None = None,
    ) -> dict[str, float]:
        """Run Metropolis MC simulation at a given temperature.

        Args:
            temperature: Temperature in units of J/kB.
            n_sweeps: Number of measurement sweeps.
            n_warmup: Number of warmup sweeps (discarded).
            key: PRNG key. Uses random seed if None.

        Returns:
            Dictionary with mean energy, magnetization, specific heat,
            and susceptibility per spin.
        """
        if temperature <= 0:
            raise ConfigurationError(f"Temperature must be positive, got {temperature}")

        if key is None:
            key = jax.random.PRNGKey(42)

        key, init_key = jax.random.split(key)
        spins = self.random_state(init_key)

        J = self._config.J
        h = self._config.h
        Lx, Ly = self._Lx, self._Ly
        N = self.n_spins

        # Warmup
        for _ in range(n_warmup):
            spins, key = self._metropolis_sweep(spins, key, temperature, Lx, Ly, J, h)

        # Measurement
        energies = []
        magnetizations = []
        for _ in range(n_sweeps):
            spins, key = self._metropolis_sweep(spins, key, temperature, Lx, Ly, J, h)
            energies.append(self._compute_energy(spins) / N)
            magnetizations.append(self._compute_magnetization(spins))

        e_arr = jnp.array(energies)
        m_arr = jnp.array(magnetizations)

        beta = 1.0 / temperature

        return {
            "energy": float(jnp.mean(e_arr)),
            "magnetization": float(jnp.mean(m_arr)),
            "specific_heat": float(beta**2 * N * jnp.var(e_arr)),
            "susceptibility": float(beta * N * jnp.var(m_arr)),
        }


def sweep_temperatures(
    lattice: IsingLattice,
    temperatures: Array,
    n_sweeps: int = 10000,
    n_warmup: int = 1000,
    algorithm: str = "metropolis",
    key: Array | None = None,
) -> IsingResult:
    """Run Ising model simulations across a range of temperatures.

    Iterates over each temperature sequentially, running a full
    Metropolis or Wolff cluster Monte Carlo simulation at each one.

    Args:
        lattice: IsingLattice instance.
        temperatures: Array of temperatures.
        n_sweeps: Measurement sweeps per temperature.
        n_warmup: Warmup sweeps per temperature.
        algorithm: "metropolis" or "wolff_cluster". Wolff updates are
            implemented for zero-field Ising models.
        key: PRNG key.

    Returns:
        IsingResult with thermodynamic quantities vs temperature.
    """
    if algorithm not in ("metropolis", "wolff_cluster"):
        raise ConfigurationError(
            f"Unknown algorithm '{algorithm}'. Choose 'metropolis' or 'wolff_cluster'."
        )

    if key is None:
        key = jax.random.PRNGKey(0)

    temperatures = jnp.asarray(temperatures)
    n_temps = temperatures.shape[0]

    logger.info(
        "Running temperature sweep: n_temps=%d, n_sweeps=%d, lattice=%dx%d",
        n_temps,
        n_sweeps,
        lattice.size[0],
        lattice.size[1],
    )

    energies = []
    magnetizations = []
    specific_heats = []
    susceptibilities = []

    for i in range(n_temps):
        T = float(temperatures[i])
        key, subkey = jax.random.split(key)
        if algorithm == "metropolis":
            result = lattice.run_metropolis(
                temperature=T,
                n_sweeps=n_sweeps,
                n_warmup=n_warmup,
                key=subkey,
            )
        else:
            result = _run_wolff_temperature(
                lattice=lattice,
                temperature=T,
                n_sweeps=n_sweeps,
                n_warmup=n_warmup,
                key=subkey,
            )
        energies.append(result["energy"])
        magnetizations.append(result["magnetization"])
        specific_heats.append(result["specific_heat"])
        susceptibilities.append(result["susceptibility"])

    return IsingResult(
        temperatures=temperatures,
        magnetizations=jnp.array(magnetizations),
        energies=jnp.array(energies),
        specific_heats=jnp.array(specific_heats),
        susceptibilities=jnp.array(susceptibilities),
    )


def _run_wolff_temperature(
    lattice: IsingLattice,
    temperature: float,
    n_sweeps: int,
    n_warmup: int,
    key: Array,
) -> dict[str, float]:
    """Run a Wolff-cluster Monte Carlo simulation at one temperature."""
    if lattice._config.h != 0.0:
        raise ConfigurationError(
            "wolff_cluster in sweep_temperatures currently requires h=0.0"
        )

    if temperature <= 0:
        raise ConfigurationError(f"Temperature must be positive, got {temperature}")

    key, init_key = jax.random.split(key)
    spins = lattice.random_state(init_key)
    J = lattice._config.J
    N = lattice.n_spins

    for _ in range(n_warmup):
        spins, key = wolff_step(spins, temperature, J, key)

    energies = []
    magnetizations = []
    for _ in range(n_sweeps):
        spins, key = wolff_step(spins, temperature, J, key)
        energies.append(lattice._compute_energy(spins) / N)
        magnetizations.append(lattice._compute_magnetization(spins))

    e_arr = jnp.array(energies)
    m_arr = jnp.array(magnetizations)
    beta = 1.0 / temperature

    return {
        "energy": float(jnp.mean(e_arr)),
        "magnetization": float(jnp.mean(m_arr)),
        "specific_heat": float(beta**2 * N * jnp.var(e_arr)),
        "susceptibility": float(beta * N * jnp.var(m_arr)),
    }


def vmap_temperatures(
    lattice: IsingLattice,
    temperatures: Array,
    n_sweeps: int = 10000,
    n_warmup: int = 1000,
    algorithm: str = "metropolis",
    key: Array | None = None,
) -> IsingResult:
    """Deprecated: use ``sweep_temperatures`` instead.

    This function was misleadingly named -- it uses a sequential
    Python for-loop, not ``jax.vmap``.  It is kept as an alias for
    backwards compatibility but will be removed in a future release.
    """
    import warnings

    warnings.warn(
        "vmap_temperatures is deprecated and will be removed in a "
        "future release. Use sweep_temperatures instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sweep_temperatures(
        lattice,
        temperatures,
        n_sweeps=n_sweeps,
        n_warmup=n_warmup,
        algorithm=algorithm,
        key=key,
    )
