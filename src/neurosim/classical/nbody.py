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

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import NBodyConfig
from neurosim.exceptions import (
    ConfigurationError,
    NumericalInstabilityError,
)
from neurosim.state import NBodyTrajectory

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

    @staticmethod
    @jax.jit
    def _compute_accelerations(
        positions: Array,
        masses: Array,
        G: float,
        softening: float,
    ) -> Array:
        """Compute gravitational accelerations on all bodies.

        Uses vectorized pairwise computation for GPU efficiency.

        Args:
            positions: Shape (n, 3).
            masses: Shape (n,).
            G: Gravitational constant.
            softening: Softening length.

        Returns:
            Accelerations, shape (n, 3).
        """
        # Pairwise displacement vectors: r_ij = r_j - r_i
        # Shape: (n, n, 3)
        dr = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]

        # Pairwise distances with softening
        # Shape: (n, n)
        dist_sq = jnp.sum(dr**2, axis=-1) + softening**2
        inv_dist_cube = dist_sq ** (-1.5)

        # Zero self-interaction
        inv_dist_cube = inv_dist_cube.at[jnp.diag_indices(positions.shape[0])].set(0.0)

        # Acceleration: a_i = G * sum_j m_j * (r_j - r_i) / |r_ij|^3
        # Shape: (n, 3)
        accel = G * jnp.einsum("j,ijk,ij->ik", masses, dr, inv_dist_cube)
        return accel

    def _kinetic_energy(self, velocities: Array, masses: Array) -> Array:
        """Compute total kinetic energy: sum(0.5 * m * v^2)."""
        return 0.5 * jnp.sum(masses[:, None] * velocities**2)

    def _potential_energy(
        self, positions: Array, masses: Array, G: float, softening: float
    ) -> Array:
        """Compute total gravitational potential energy.

        U = -G * sum_{i<j} m_i * m_j / |r_i - r_j|
        """
        dr = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + softening**2)
        # Mass product matrix
        mass_prod = masses[:, None] * masses[None, :]
        # Upper triangle sum (avoid double-counting and self)
        n = masses.shape[0]
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        return -G * jnp.sum(mask * mass_prod / dist)

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
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        masses = self._masses
        G = self._config.G
        softening = self._config.softening

        logger.info(
            "Starting N-body simulation: n=%d, n_steps=%d, dt=%.2e",
            self.n_bodies,
            n_steps,
            dt,
        )

        def verlet_step(
            carry: tuple[Array, Array, Array, float],
            _: None,
        ) -> tuple[
            tuple[Array, Array, Array, float],
            tuple[Array, Array, Array],
        ]:
            pos, vel, acc, t = carry
            # Velocity Verlet
            pos_new = pos + vel * dt + 0.5 * acc * dt**2
            acc_new = NBody._compute_accelerations(pos_new, masses, G, softening)
            vel_new = vel + 0.5 * (acc + acc_new) * dt
            return (pos_new, vel_new, acc_new, t + dt), (
                pos_new,
                vel_new,
                jnp.asarray(t + dt),
            )

        # Initial acceleration
        acc0 = NBody._compute_accelerations(self._positions, masses, G, softening)

        init_carry = (self._positions, self._velocities, acc0, t_start)
        _, (pos_hist, vel_hist, t_hist) = jax.lax.scan(
            verlet_step, init_carry, None, length=n_steps
        )

        # Prepend initial state
        pos_hist = jnp.concatenate(
            [self._positions[jnp.newaxis, :, :], pos_hist], axis=0
        )
        vel_hist = jnp.concatenate(
            [self._velocities[jnp.newaxis, :, :], vel_hist], axis=0
        )
        t_hist = jnp.concatenate([jnp.array([t_start]), t_hist], axis=0)

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps + 1, save_every)
            pos_hist = pos_hist[indices]
            vel_hist = vel_hist[indices]
            t_hist = t_hist[indices]

        # Compute energy at saved steps
        def compute_energy(pos: Array, vel: Array) -> Array:
            ke = self._kinetic_energy(vel, masses)
            pe = self._potential_energy(pos, masses, G, softening)
            return ke + pe

        energy = jax.vmap(compute_energy)(pos_hist, vel_hist)

        if jnp.any(jnp.isnan(pos_hist)):
            raise NumericalInstabilityError(
                "NaN detected in N-body simulation. Try increasing "
                "n_steps or the softening parameter."
            )

        return NBodyTrajectory(
            t=t_hist,
            positions=pos_hist,
            velocities=vel_hist,
            masses=masses,
            energy=energy,
        )
