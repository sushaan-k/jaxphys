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
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys.config import IsingConfig
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import IsingResult
from jaxphys.statmech.monte_carlo import wolff_step

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
        return _random_spins(key, (self._Lx, self._Ly))

    def run_metropolis(
        self,
        temperature: float,
        n_sweeps: int = 10000,
        n_warmup: int = 1000,
        key: Array | None = None,
    ) -> dict[str, float]:
        """Run Metropolis MC simulation at a given temperature.

        The whole chain (warm-up and measurement) runs as one compiled
        loop. On lattices with even side lengths a sweep is a checkerboard
        update: all spins of one sublattice are proposed simultaneously,
        which is valid because same-colour spins do not interact. Odd
        lattices fall back to N random single-site proposals per sweep.

        Args:
            temperature: Temperature in units of J/kB.
            n_sweeps: Number of measurement sweeps.
            n_warmup: Number of warmup sweeps (discarded).
            key: PRNG key. Uses ``PRNGKey(42)`` if None.

        Returns:
            Dictionary with mean energy, magnetization, specific heat,
            and susceptibility per spin.
        """
        if temperature <= 0:
            raise ConfigurationError(f"Temperature must be positive, got {temperature}")

        if key is None:
            key = jax.random.PRNGKey(42)

        key, init_key = jax.random.split(key)
        energies, mags = _metropolis_chain(
            self.random_state(init_key),
            key,
            1.0 / temperature,
            self._config.J,
            self._config.h,
            n_warmup=n_warmup,
            n_sweeps=n_sweeps,
        )
        stats = _thermo_stats(energies, mags, temperature, self.n_spins)
        return {name: float(value) for name, value in stats.items()}


def _random_spins(key: Array, shape: tuple[int, int]) -> Array:
    """Uniformly random +1/-1 spins (int32)."""
    return 2 * jax.random.bernoulli(key, shape=shape).astype(jnp.int32) - 1


def _energy(spins: Array, J: float | Array, h: float | Array) -> Array:
    """Total Ising energy with periodic boundaries."""
    bonds = spins * jnp.roll(spins, 1, axis=0) + spins * jnp.roll(spins, 1, axis=1)
    return -J * jnp.sum(bonds) - h * jnp.sum(spins)


def _observables(
    spins: Array, J: float | Array, h: float | Array
) -> tuple[Array, Array]:
    """Energy per spin and absolute magnetization per spin."""
    return _energy(spins, J, h) / spins.size, jnp.abs(jnp.mean(spins, dtype=float))


def _thermo_stats(
    energies: Array, mags: Array, temperature: float | Array, n_spins: int
) -> dict[str, Array]:
    """Means of e/N and |m| and their fluctuation estimates of C_v and chi.

    Reduces over the last axis, so a batch of chains gives one value each.
    """
    beta = 1.0 / jnp.asarray(temperature)
    return {
        "energy": jnp.mean(energies, axis=-1),
        "magnetization": jnp.mean(mags, axis=-1),
        "specific_heat": beta**2 * n_spins * jnp.var(energies, axis=-1),
        "susceptibility": beta * n_spins * jnp.var(mags, axis=-1),
    }


def _checkerboard_sweep(
    spins: Array, key: Array, beta: Array | float, J: float, h: float
) -> tuple[Array, Array]:
    """One Metropolis sweep as two sublattice (checkerboard) half-sweeps."""
    Lx, Ly = spins.shape
    parity = (jnp.arange(Lx)[:, None] + jnp.arange(Ly)[None, :]) % 2
    for color in (0, 1):
        key, sub = jax.random.split(key)
        nn_sum = (
            jnp.roll(spins, 1, axis=0)
            + jnp.roll(spins, -1, axis=0)
            + jnp.roll(spins, 1, axis=1)
            + jnp.roll(spins, -1, axis=1)
        )
        dE = 2.0 * spins * (J * nn_sum + h)
        accept = jax.random.uniform(sub, spins.shape) < jnp.exp(-beta * dE)
        spins = jnp.where((parity == color) & accept, -spins, spins)
    return spins, key


def _random_site_sweep(
    spins: Array, key: Array, beta: Array | float, J: float, h: float
) -> tuple[Array, Array]:
    """One Metropolis sweep of N sequential random single-site proposals."""
    Lx, Ly = spins.shape

    def single_flip(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        s, k = carry
        k, k1, k2, k3 = jax.random.split(k, 4)
        ix = jax.random.randint(k1, (), 0, Lx)
        iy = jax.random.randint(k2, (), 0, Ly)
        nn_sum = (
            s[(ix + 1) % Lx, iy]
            + s[(ix - 1) % Lx, iy]
            + s[ix, (iy + 1) % Ly]
            + s[ix, (iy - 1) % Ly]
        )
        dE = 2.0 * s[ix, iy] * (J * nn_sum + h)
        accept = jax.random.uniform(k3) < jnp.exp(-beta * dE)
        return s.at[ix, iy].set(jnp.where(accept, -s[ix, iy], s[ix, iy])), k

    out: tuple[Array, Array] = jax.lax.fori_loop(0, Lx * Ly, single_flip, (spins, key))
    return out


@partial(jax.jit, static_argnames=("n_warmup", "n_sweeps"))
def _metropolis_chain(
    spins: Array,
    key: Array,
    beta: float,
    J: float,
    h: float,
    *,
    n_warmup: int,
    n_sweeps: int,
) -> tuple[Array, Array]:
    """Warm up, then record (energy/N, |m|) after each measurement sweep."""
    Lx, Ly = spins.shape
    sweep = _checkerboard_sweep if Lx % 2 == 0 and Ly % 2 == 0 else _random_site_sweep

    def advance(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return sweep(carry[0], carry[1], beta, J, h)

    carry = jax.lax.fori_loop(0, n_warmup, advance, (spins, key))

    def measure(carry: tuple[Array, Array], _: None) -> tuple[Any, Any]:
        carry = advance(0, carry)
        return carry, _observables(carry[0], J, h)

    _, (energies, mags) = jax.lax.scan(measure, carry, None, length=n_sweeps)
    return energies, mags


@partial(jax.jit, static_argnames=("n_warmup", "n_sweeps"))
def _wolff_chain(
    spins: Array,
    key: Array,
    temperature: float,
    J: float,
    *,
    n_warmup: int,
    n_sweeps: int,
) -> tuple[Array, Array]:
    """Wolff analogue of :func:`_metropolis_chain` (one cluster flip per sweep)."""

    def advance(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        out: tuple[Array, Array] = wolff_step(carry[0], temperature, J, carry[1])
        return out

    carry = jax.lax.fori_loop(0, n_warmup, advance, (spins, key))

    def measure(carry: tuple[Array, Array], _: None) -> tuple[Any, Any]:
        carry = advance(0, carry)
        return carry, _observables(carry[0], J, 0.0)

    _, (energies, mags) = jax.lax.scan(measure, carry, None, length=n_sweeps)
    return energies, mags


@partial(jax.jit, static_argnames=("shape", "algorithm", "n_warmup", "n_sweeps"))
def _sweep_chains(
    keys: Array,
    temperatures: Array,
    J: float,
    h: float,
    *,
    shape: tuple[int, int],
    algorithm: str,
    n_warmup: int,
    n_sweeps: int,
) -> tuple[Array, Array]:
    """One independent chain per temperature, vmapped and compiled once.

    Compiled once per (lattice shape, algorithm, n_warmup, n_sweeps, number
    of temperatures); temperatures, couplings and keys are traced.
    """

    def one_temperature(k: Array, T: Array) -> tuple[Array, Array]:
        k, init_key = jax.random.split(k)
        spins = _random_spins(init_key, shape)
        chain: tuple[Array, Array]
        if algorithm == "metropolis":
            chain = _metropolis_chain(
                spins, k, 1.0 / T, J, h, n_warmup=n_warmup, n_sweeps=n_sweeps
            )
        else:
            chain = _wolff_chain(spins, k, T, J, n_warmup=n_warmup, n_sweeps=n_sweeps)
        return chain

    out: tuple[Array, Array] = jax.vmap(one_temperature)(keys, temperatures)
    return out


def sweep_temperatures(
    lattice: IsingLattice,
    temperatures: Array,
    n_sweeps: int = 10000,
    n_warmup: int = 1000,
    algorithm: str = "metropolis",
    key: Array | None = None,
) -> IsingResult:
    """Run Ising model simulations across a range of temperatures.

    All temperatures run as independent chains in a single ``jax.vmap``-ed,
    compiled call, so the sweep is parallel across temperatures.

    Args:
        lattice: IsingLattice instance.
        temperatures: Array of temperatures.
        n_sweeps: Measurement sweeps per temperature.
        n_warmup: Warmup sweeps per temperature.
        algorithm: "metropolis" or "wolff_cluster". Wolff updates are
            implemented for zero-field Ising models; one Wolff "sweep" is
            one cluster flip.
        key: PRNG key.

    Returns:
        IsingResult with thermodynamic quantities vs temperature.
    """
    if algorithm not in ("metropolis", "wolff_cluster"):
        raise ConfigurationError(
            f"Unknown algorithm '{algorithm}'. Choose 'metropolis' or 'wolff_cluster'."
        )
    if algorithm == "wolff_cluster" and lattice._config.h != 0.0:
        raise ConfigurationError("wolff_cluster requires h=0.0")

    if key is None:
        key = jax.random.PRNGKey(0)

    temperatures = jnp.asarray(temperatures, dtype=jnp.float64)
    if temperatures.ndim != 1 or temperatures.shape[0] == 0:
        raise ConfigurationError("temperatures must be a non-empty 1D array")
    if bool(jnp.any(temperatures <= 0)):
        raise ConfigurationError("All temperatures must be positive")

    logger.info(
        "Running temperature sweep: n_temps=%d, n_sweeps=%d, lattice=%dx%d",
        temperatures.shape[0],
        n_sweeps,
        lattice.size[0],
        lattice.size[1],
    )

    keys = jax.random.split(key, temperatures.shape[0])
    energies, mags = _sweep_chains(
        keys,
        temperatures,
        lattice._config.J,
        lattice._config.h,
        shape=lattice.size,
        algorithm=algorithm,
        n_warmup=n_warmup,
        n_sweeps=n_sweeps,
    )
    stats = _thermo_stats(energies, mags, temperatures, lattice.n_spins)
    return IsingResult(
        temperatures=temperatures,
        magnetizations=stats["magnetization"],
        energies=stats["energy"],
        specific_heats=stats["specific_heat"],
        susceptibilities=stats["susceptibility"],
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
        raise ConfigurationError("wolff_cluster requires h=0.0")
    if temperature <= 0:
        raise ConfigurationError(f"Temperature must be positive, got {temperature}")

    key, init_key = jax.random.split(key)
    energies, mags = _wolff_chain(
        lattice.random_state(init_key),
        key,
        temperature,
        lattice._config.J,
        n_warmup=n_warmup,
        n_sweeps=n_sweeps,
    )
    stats = _thermo_stats(energies, mags, temperature, lattice.n_spins)
    return {name: float(value) for name, value in stats.items()}


def vmap_temperatures(
    lattice: IsingLattice,
    temperatures: Array,
    n_sweeps: int = 10000,
    n_warmup: int = 1000,
    algorithm: str = "metropolis",
    key: Array | None = None,
) -> IsingResult:
    """Deprecated alias of :func:`sweep_temperatures`.

    Kept for backwards compatibility; it will be removed in a future
    release.
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
