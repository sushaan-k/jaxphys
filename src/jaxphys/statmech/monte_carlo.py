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
) -> tuple[Array, Array, bool]:
    """Perform one Metropolis-Hastings step.

    Args:
        energy_fn: Function state -> scalar energy.
        state: Current state (arbitrary JAX array).
        proposal_fn: Function (state, key) -> proposed_state.
        temperature: Temperature (kB = 1 units).
        key: PRNG key.

    Returns:
        Tuple of (new_state, new_key, accepted).
    """
    key, k1, k2 = jax.random.split(key, 3)

    proposed = proposal_fn(state, k1)
    dE = energy_fn(proposed) - energy_fn(state)
    beta = 1.0 / temperature

    accept = (dE < 0) | (jax.random.uniform(k2) < jnp.exp(-beta * dE))
    new_state = jnp.where(accept, proposed, state)

    return new_state, key, bool(accept)


def wolff_step(
    spins: Array,
    temperature: float,
    J: float,
    key: Array,
) -> tuple[Array, Array]:
    """Perform one Wolff cluster flip on a 2D Ising lattice.

    The Wolff algorithm:
    1. Pick a random seed spin.
    2. Grow a cluster by adding aligned neighbors with probability
       p = 1 - exp(-2*beta*J).
    3. Flip the entire cluster.

    This eliminates critical slowing down near T_c.

    Args:
        spins: 2D spin array (+1/-1), shape (Lx, Ly).
        temperature: Temperature.
        J: Coupling constant.
        key: PRNG key.

    Returns:
        (new_spins, new_key).
    """
    Lx, Ly = spins.shape
    beta = 1.0 / temperature
    p_add = 1.0 - jnp.exp(-2.0 * beta * J)

    key, k_seed_x, k_seed_y = jax.random.split(key, 3)
    # Random seed site
    seed_x = jax.random.randint(k_seed_x, (), 0, Lx)
    seed_y = jax.random.randint(k_seed_y, (), 0, Ly)
    seed_spin = spins[seed_x, seed_y]

    # BFS cluster growth using a fixed-size visited mask
    cluster = jnp.zeros((Lx, Ly), dtype=bool)
    cluster = cluster.at[seed_x, seed_y].set(True)

    # Use a scan-based approach for JIT compatibility
    def grow_step(
        carry: tuple[Array, Array, Array],
        _: None,
    ) -> tuple[tuple[Array, Array, Array], None]:
        cl, sp, k = carry
        k, k1 = jax.random.split(k)

        # For each cluster spin, try adding its neighbors
        # Shift cluster mask to find potential additions
        up = jnp.roll(cl, 1, axis=0)
        down = jnp.roll(cl, -1, axis=0)
        left = jnp.roll(cl, 1, axis=1)
        right = jnp.roll(cl, -1, axis=1)

        # Neighbors of cluster members that are not yet in cluster
        candidates = (up | down | left | right) & ~cl

        # Only accept aligned spins with probability p_add
        aligned = sp == seed_spin
        rands = jax.random.uniform(k1, shape=(Lx, Ly))
        accept = candidates & aligned & (rands < p_add)

        cl = cl | accept
        return (cl, sp, k), None

    # Iterate enough times for cluster to potentially span lattice.
    # The cluster can grow at most one layer per iteration, so we need
    # at least (Lx + Ly) iterations to span the lattice diagonally.
    n_growth_steps = Lx + Ly
    (cluster, _, key), _ = jax.lax.scan(
        grow_step, (cluster, spins, key), None, length=n_growth_steps
    )

    # Flip the cluster
    flipped = jnp.where(cluster, -spins, spins)
    return flipped, key
