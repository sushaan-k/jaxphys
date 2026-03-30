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

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError


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
    if temperature <= 0:
        raise ConfigurationError(f"Temperature must be positive, got {temperature}")

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
    if temperature <= 0:
        raise ConfigurationError(f"Temperature must be positive, got {temperature}")

    energies = jnp.asarray(energies)
    beta = 1.0 / temperature

    e_shifted = energies - jnp.min(energies)
    log_probs = -beta * e_shifted

    if degeneracies is not None:
        degeneracies = jnp.asarray(degeneracies)
        log_probs = log_probs + jnp.log(degeneracies)

    # Log-sum-exp for numerical stability
    log_Z = jax_logsumexp(log_probs)
    probs = jnp.exp(log_probs - log_Z)

    return probs


def jax_logsumexp(x: Array) -> Array:
    """Numerically stable log-sum-exp.

    log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Args:
        x: Input array.

    Returns:
        Scalar log-sum-exp value.
    """
    x_max = jnp.max(x)
    return x_max + jnp.log(jnp.sum(jnp.exp(x - x_max)))


def _log_partition_function(
    energies: Array,
    temperature: float,
    degeneracies: Array | None = None,
) -> Array:
    """Compute ``log(Z)`` stably for the canonical ensemble."""
    energies = jnp.asarray(energies)
    beta = 1.0 / temperature
    e_min = jnp.min(energies)
    shifted = -beta * (energies - e_min)

    if degeneracies is not None:
        degeneracies = jnp.asarray(degeneracies)
        shifted = shifted + jnp.log(degeneracies)

    return -beta * e_min + jax_logsumexp(shifted)


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
    return float(jnp.sum(energies * probs))


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
    """Compute entropy S = -sum_i P_i * ln(P_i).

    Args:
        energies: Energy levels.
        temperature: Temperature.
        degeneracies: Optional degeneracy factors.

    Returns:
        Entropy in natural units.
    """
    probs = boltzmann_distribution(energies, temperature, degeneracies)
    # Avoid log(0)
    safe_probs = jnp.clip(probs, 1e-30, None)
    return float(-jnp.sum(probs * jnp.log(safe_probs)))
