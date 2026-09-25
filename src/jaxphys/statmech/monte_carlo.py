"""General-purpose Monte Carlo methods.

Provides building blocks for Markov Chain Monte Carlo (MCMC)
simulations, including Metropolis-Hastings and Wolff cluster updates.

These are low-level functions intended to be composed into
higher-level simulation pipelines.

References:
    - Metropolis et al. "Equation of State Calculations by Fast
      Computing Machines" (1953)
    - Wolff. "Collective Monte Carlo Updating for Spin Systems" (1989)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

EnergyFn: TypeAlias = Callable[[Array], Array]
ProposalFn: TypeAlias = Callable[[Array, Array], Array]


def metropolis_step(
    energy_fn: EnergyFn,
    state: Array,
    proposal_fn: ProposalFn,
    temperature: float,
    key: Array,
) -> tuple[Array, Array, Array]:
    """Perform one Metropolis-Hastings step.

    Args:
        energy_fn: Function state -> scalar energy.
        state: Current state (arbitrary JAX array).
        proposal_fn: Function (state, key) -> proposed_state.
        temperature: Temperature (kB = 1 units).
        key: PRNG key.

    Returns:
        Tuple of (new_state, new_key, accepted), where ``accepted`` is a
        boolean scalar array (so the step can run under ``jax.jit``).
    """
    key, k1, k2 = jax.random.split(key, 3)

    proposed = proposal_fn(state, k1)
    dE = energy_fn(proposed) - energy_fn(state)
    beta = 1.0 / temperature

    accept = (dE < 0) | (jax.random.uniform(k2) < jnp.exp(-beta * dE))
    new_state = jnp.where(accept, proposed, state)

    return new_state, key, accept


@jax.jit
def wolff_step(
    spins: Array,
    temperature: float,
    J: float,
    key: Array,
) -> tuple[Array, Array]:
    """Perform one Wolff cluster flip on a 2D Ising lattice.

    The Wolff algorithm:
    1. Pick a random seed spin.
    2. Activate each satisfied bond (J * s_i * s_j > 0) independently with
       probability p = 1 - exp(-2*beta*|J|).
    3. Flip the connected cluster of active bonds containing the seed.

    Every bond is sampled exactly once, which is what detailed balance
    requires; the cluster is then grown to convergence with a
    ``lax.while_loop`` flood fill, so arbitrarily shaped clusters are
    captured. This eliminates critical slowing down near T_c.

    Args:
        spins: 2D spin array (+1/-1), shape (Lx, Ly), periodic boundaries.
        temperature: Temperature.
        J: Coupling constant.
        key: PRNG key.

    Returns:
        (new_spins, new_key).
    """
    Lx, Ly = spins.shape
    beta = 1.0 / temperature
    p_add = 1.0 - jnp.exp(-2.0 * beta * jnp.abs(J))

    key, k_seed, k_x, k_y = jax.random.split(key, 4)
    seed = jax.random.randint(k_seed, (), 0, Lx * Ly)

    # bond_x[i, j] links (i, j)-(i+1, j); bond_y[i, j] links (i, j)-(i, j+1).
    def bonds(k: Array, axis: int) -> Array:
        satisfied = J * spins * jnp.roll(spins, -1, axis=axis) > 0
        return satisfied & (jax.random.uniform(k, spins.shape) < p_add)

    bond_x = bonds(k_x, 0)
    bond_y = bonds(k_y, 1)

    def grow(cluster: Array) -> Array:
        return (
            cluster
            | jnp.roll(cluster & bond_x, 1, axis=0)
            | (jnp.roll(cluster, -1, axis=0) & bond_x)
            | jnp.roll(cluster & bond_y, 1, axis=1)
            | (jnp.roll(cluster, -1, axis=1) & bond_y)
        )

    def not_converged(state: tuple[Array, Array]) -> Array:
        return state[1]

    def body(state: tuple[Array, Array]) -> tuple[Array, Array]:
        grown = grow(state[0])
        return grown, jnp.any(grown != state[0])

    cluster0 = (jnp.arange(Lx * Ly) == seed).reshape(Lx, Ly)
    cluster, _ = jax.lax.while_loop(not_converged, body, (cluster0, jnp.array(True)))
    return jnp.where(cluster, -spins, spins), key
