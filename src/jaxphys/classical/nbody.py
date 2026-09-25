"""N-body gravitational simulation.

GPU-accelerated N-body simulator using direct O(N^2) pairwise
force computation. Employs velocity Verlet integration for
symplectic time evolution.

The gravitational acceleration on body i is:
    a_i = -G * sum_{j != i} m_j * (r_i - r_j) / |r_i - r_j|^3

A softening parameter epsilon prevents divergence at close approach:
    a_i = -G * sum_{j != i} m_j * (r_i - r_j) / (|r_i - r_j|^2 + eps^2)^{3/2}

References:
    - Aarseth. "Gravitational N-Body Simulations" (2003)
    - Dehnen & Read. "N-body simulations of gravitational dynamics" (2011)
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys._rollout import is_traced, strided_rollout
from jaxphys.config import NBodyConfig
from jaxphys.exceptions import (
    ConfigurationError,
    NumericalInstabilityError,
)
from jaxphys.state import NBodyTrajectory

logger = logging.getLogger(__name__)


class NBody:
    """N-body gravitational simulator.

    Computes pairwise gravitational forces between all bodies and
    integrates orbits using velocity Verlet. All force computations
    are JIT-compiled and vectorized for GPU acceleration.

    Example:
        >>> system = NBody(
        ...     masses=[1.0, 0.001],
        ...     positions=[[0, 0, 0], [1, 0, 0]],
        ...     velocities=[[0, 0, 0], [0, 1, 0]],
        ...     G=1.0,
        ... )
        >>> traj = system.simulate(t_span=(0, 100), n_steps=100000)

    Args:
        masses: List or array of particle masses.
        positions: Initial positions, shape (n, 3).
        velocities: Initial velocities, shape (n, 3).
        G: Gravitational constant. Default 1.0.
        softening: Softening length to prevent singularities.
    """

    def __init__(
        self,
        masses: list[float] | Array,
        positions: list[list[float]] | Array,
        velocities: list[list[float]] | Array,
        G: float = 1.0,
        softening: float = 1e-4,
    ) -> None:
        self._masses = jnp.asarray(masses, dtype=jnp.float64)
        self._positions = jnp.asarray(positions, dtype=jnp.float64)
        self._velocities = jnp.asarray(velocities, dtype=jnp.float64)
        self._config = NBodyConfig(G=G, softening=softening, theta=0.5)

        n = self._masses.shape[0]
        if self._positions.shape != (n, 3):
            raise ConfigurationError(
                f"positions shape {self._positions.shape} != expected ({n}, 3)"
            )
        if self._velocities.shape != (n, 3):
            raise ConfigurationError(
                f"velocities shape {self._velocities.shape} != expected ({n}, 3)"
            )
        if jnp.any(self._masses <= 0):
            raise ConfigurationError("All masses must be positive")

    @property
    def n_bodies(self) -> int:
        """Number of bodies in the system."""
        return int(self._masses.shape[0])

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 100.0),
        n_steps: int = 100000,
        save_every: int = 100,
    ) -> NBodyTrajectory:
        """Simulate the N-body system using velocity Verlet.

        Args:
            t_span: (t_start, t_end) time interval.
            n_steps: Total number of integration steps.
            save_every: Save snapshot every N steps.

        Returns:
            NBodyTrajectory with full orbital history.

        Raises:
            NumericalInstabilityError: If NaN values detected.
        """
        if n_steps < 1:
            raise ConfigurationError(f"n_steps must be >= 1, got {n_steps}")
        if save_every < 1:
            raise ConfigurationError(f"save_every must be >= 1, got {save_every}")
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps

        logger.info(
            "Starting N-body simulation: n=%d, n_steps=%d, dt=%.2e",
            self.n_bodies,
            n_steps,
            dt,
        )

        pos_hist, vel_hist, t_hist, energy = _nbody_rollout(
            self._positions,
            self._velocities,
            self._masses,
            jnp.asarray(t_start, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
            self._config.G,
            self._config.softening,
            n_steps=n_steps,
            save_every=save_every,
        )

        if not is_traced(pos_hist) and bool(jnp.any(jnp.isnan(pos_hist))):
            raise NumericalInstabilityError(
                "NaN detected in N-body simulation. Try increasing "
                "n_steps or the softening parameter."
            )

        return NBodyTrajectory(
            t=t_hist,
            positions=pos_hist,
            velocities=vel_hist,
            masses=self._masses,
            energy=energy,
        )


def gravitational_accelerations(
    positions: Array, masses: Array, G: float | Array, softening: float | Array
) -> Array:
    """Softened pairwise gravitational accelerations, shape ``(n, 3)``.

    a_i = G * sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^{3/2}
    """
    # dr[i, j] = r_j - r_i, shape (n, n, 3)
    dr = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]
    # Pair weights m_j / d_ij^3 with the self-interaction removed. The
    # diagonal distance is replaced before the power so gradients stay
    # finite even without softening.
    self_pair = jnp.eye(positions.shape[0], dtype=bool)
    dist_sq = jnp.where(self_pair, 1.0, jnp.sum(dr**2, axis=-1) + softening**2)
    w = jnp.where(self_pair, 0.0, masses[jnp.newaxis, :] * dist_sq**-1.5)
    return G * jnp.einsum("ij,ijk->ik", w, dr)


def total_energy(
    positions: Array,
    velocities: Array,
    masses: Array,
    G: float | Array,
    softening: float | Array,
) -> Array:
    """Kinetic plus softened pairwise potential energy of an N-body state."""
    kinetic = 0.5 * jnp.sum(masses[:, None] * velocities**2)
    dr = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]
    upper = jnp.triu(jnp.ones(masses.shape * 2, dtype=bool), k=1)
    dist = jnp.sqrt(jnp.where(upper, jnp.sum(dr**2, axis=-1) + softening**2, 1.0))
    pair = jnp.where(upper, masses[:, None] * masses[None, :] / dist, 0.0)
    return kinetic - G * jnp.sum(pair)


@partial(jax.jit, static_argnames=("n_steps", "save_every"))
def _nbody_rollout(
    positions: Array,
    velocities: Array,
    masses: Array,
    t0: Array,
    dt: Array,
    G: float,
    softening: float,
    *,
    n_steps: int,
    save_every: int,
) -> tuple[Array, Array, Array, Array]:
    """Velocity-Verlet rollout saving every ``save_every`` steps (incl. t0)."""

    def step(
        carry: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        pos, vel, acc, t = carry
        pos = pos + vel * dt + 0.5 * acc * dt**2
        acc_new = gravitational_accelerations(pos, masses, G, softening)
        vel = vel + 0.5 * (acc + acc_new) * dt
        return pos, vel, acc_new, t + dt

    def observe(carry: tuple[Array, Array, Array, Array]) -> tuple[Array, ...]:
        pos, vel, _, t = carry
        return pos, vel, t, total_energy(pos, vel, masses, G, softening)

    acc0 = gravitational_accelerations(positions, masses, G, softening)
    out: tuple[Array, Array, Array, Array] = strided_rollout(
        step, (positions, velocities, acc0, t0), n_steps, save_every, observe
    )
    return out
