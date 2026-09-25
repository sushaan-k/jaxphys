"""Boltzmann distribution utilities.

Provides functions for computing partition functions, free energies,
and Boltzmann-weighted averages for discrete and continuous energy
spectra.

The Boltzmann distribution assigns probability:
    P(E_i) = exp(-E_i / kT) / Z

where Z = sum_i exp(-E_i / kT) is the partition function.

References:
    - Pathria & Beale. "Statistical Mechanics" (2011)
    - Reif. "Fundamentals of Statistical and Thermal Physics" (1965)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys.exceptions import ConfigurationError


def partition_function(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> float:
    """Compute the canonical partition function.

    Z = sum_i g_i * exp(-E_i / kT)

    where g_i are degeneracies (default 1).

    Args:
        energies: Energy levels, shape (n_levels,).
        temperature: Temperature (kB = 1 units).
        degeneracies: Optional degeneracy factors, shape (n_levels,).

    Returns:
        Partition function Z.

    Raises:
        ConfigurationError: If temperature is non-positive.
    """
    log_Z = _log_partition_function(energies, temperature, degeneracies)
    return float(jnp.exp(log_Z))


def boltzmann_distribution(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> Array:
    """Compute Boltzmann probability distribution over energy levels.

    P(E_i) = g_i * exp(-E_i / kT) / Z

    Args:
        energies: Energy levels, shape (n_levels,).
        temperature: Temperature (kB = 1 units).
        degeneracies: Optional degeneracy factors.

    Returns:
        Probability array, shape (n_levels,). Sums to 1.

    Raises:
        ConfigurationError: If temperature is non-positive.
    """
    log_weights = _log_weights(energies, temperature, degeneracies)
    return jnp.exp(log_weights - jax_logsumexp(log_weights))


def jax_logsumexp(x: Array) -> Array:
    """Numerically stable log-sum-exp (thin wrapper over ``jax.nn.logsumexp``)."""
    return jax.nn.logsumexp(x)


def _log_weights(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> Array:
    """Unnormalized log-probabilities ``ln g_i - E_i / kT`` of each level."""
    if temperature <= 0:
        raise ConfigurationError(f"Temperature must be positive, got {temperature}")
    log_weights = -jnp.asarray(energies) / temperature
    if degeneracies is not None:
        log_weights = log_weights + jnp.log(jnp.asarray(degeneracies))
    return log_weights


def _log_partition_function(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> Array:
    """Compute ``log(Z)`` stably for the canonical ensemble."""
    return jax_logsumexp(_log_weights(energies, temperature, degeneracies))


def mean_energy(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> float:
    """Compute mean energy <E> = sum_i E_i * P(E_i).

    Args:
        energies: Energy levels.
        temperature: Temperature.
        degeneracies: Optional degeneracy factors.

    Returns:
        Mean energy.
    """
    probs = boltzmann_distribution(energies, temperature, degeneracies)
    return float(jnp.sum(jnp.asarray(energies) * probs))


def free_energy(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> float:
    """Compute Helmholtz free energy F = -kT * ln(Z).

    Args:
        energies: Energy levels.
        temperature: Temperature.
        degeneracies: Optional degeneracy factors.

    Returns:
        Free energy F.
    """
    log_Z = _log_partition_function(energies, temperature, degeneracies)
    return -temperature * float(log_Z)


def entropy(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> float:
    """Compute the canonical entropy S = -sum_states p ln p.

    With level probabilities ``P_i`` and degeneracies ``g_i`` each of the
    ``g_i`` states has probability ``P_i / g_i``, so
    ``S = -sum_i P_i * ln(P_i / g_i)`` (equal to ``(U - F) / T``).

    Args:
        energies: Energy levels.
        temperature: Temperature.
        degeneracies: Optional degeneracy factors.

    Returns:
        Entropy in natural units (kB = 1).
    """
    probs = boltzmann_distribution(energies, temperature, degeneracies)
    per_state = probs if degeneracies is None else probs / jnp.asarray(degeneracies)
    # Avoid log(0); those terms contribute 0 * log(0) = 0.
    safe = jnp.clip(per_state, 1e-300, None)
    return float(-jnp.sum(probs * jnp.log(safe)))
