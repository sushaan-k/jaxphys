"""Charge dynamics simulation.

Simulates the motion of charged particles in electric and magnetic fields
using the Lorentz force law:

    F = q * (E + v x B)

Supports both prescribed external fields and self-consistent particle-particle
Coulomb interactions.

References:
    - Griffiths. "Introduction to Electrodynamics" (2017), Ch. 2, 5
    - Boris pusher: Boris (1970), "Relativistic plasma simulation"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError
from neurosim.state import NBodyTrajectory

logger = logging.getLogger(__name__)

# Coulomb constant in SI units
K_COULOMB = 8.9875517873681764e9  # N m^2 / C^2


@dataclass(frozen=True)
class PointCharge:
    """A point charge specification.

    Attributes:
        charge: Charge in Coulombs.
        mass: Mass in kg.
        position: Initial position (x, y, z) in meters.
        velocity: Initial velocity (vx, vy, vz) in m/s.
    """

    charge: float
    mass: float
    position: list[float] | Array
    velocity: list[float] | Array


class ChargeSystem:
    """System of charged particles with Coulomb interactions.

    Simulates charged particle dynamics under mutual Coulomb forces
    and optional external electric/magnetic fields.

    Example:
        >>> q1 = PointCharge(charge=1e-6, mass=1e-3,
        ...     position=[0, 0, 0], velocity=[0, 0, 0])
        >>> q2 = PointCharge(charge=-1e-6, mass=1e-3,
        ...     position=[0.1, 0, 0], velocity=[0, 0, 0])
        >>> system = ChargeSystem(charges=[q1, q2])
        >>> traj = system.simulate(t_span=(0, 1e-3), n_steps=10000)

    Args:
        charges: List of PointCharge objects.
        E_external: External electric field function
            (position, t) -> E_vector, or constant vector.
        B_external: External magnetic field function
            (position, t) -> B_vector, or constant vector.
        softening: Softening parameter for close-range interactions.
    """

    def __init__(
        self,
        charges: list[PointCharge],
        E_external: Array | Callable[[Array, float], Array] | None = None,
        B_external: Array | Callable[[Array, float], Array] | None = None,
        softening: float = 1e-10,
    ) -> None:
        if len(charges) < 1:
            raise ConfigurationError("Need at least one charge")

        self._n = len(charges)
        self._charges = jnp.array([c.charge for c in charges])
        self._masses = jnp.array([c.mass for c in charges])
        self._positions = jnp.array([c.position for c in charges], dtype=jnp.float64)
        self._velocities = jnp.array([c.velocity for c in charges], dtype=jnp.float64)
        self._softening = softening

        self._E_ext = E_external
        self._B_ext = B_external

    @property
    def n_charges(self) -> int:
        """Number of charges."""
        return self._n

    def _compute_accelerations(
        self,
        positions: Array,
        velocities: Array,
        charges: Array,
        masses: Array,
        softening: float,
        E_ext: Array | Callable[[Array, float], Array] | None,
        B_ext: Array | Callable[[Array, float], Array] | None,
        t: float,
    ) -> Array:
        """Compute accelerations from Coulomb + Lorentz forces.

        Args:
            positions: Shape (n, 3).
            velocities: Shape (n, 3).
            charges: Shape (n,).
            masses: Shape (n,).
            softening: Softening length.
            E_ext: External E field, shape (3,) or callable.
            B_ext: External B field, shape (3,) or callable.
            t: Current simulation time.

        Returns:
            Accelerations, shape (n, 3).
        """
        n = charges.shape[0]

        # Coulomb force
        dr = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]
        dist_sq = jnp.sum(dr**2, axis=-1) + softening**2
        inv_dist_cube = dist_sq ** (-1.5)
        inv_dist_cube = inv_dist_cube.at[jnp.diag_indices(n)].set(0.0)

        # F_i = k * q_i * sum_j q_j * (r_i - r_j) / |r_ij|^3
        # Note: dr[i,j] = r_j - r_i, so force on i from j is
        # k * q_i * q_j * (-dr[i,j]) / |r_ij|^3
        force_coulomb = -K_COULOMB * jnp.einsum(
            "i,j,ijk,ij->ik", charges, charges, dr, inv_dist_cube
        )

        def evaluate_field(
            field: Array | Callable[[Array, float], Array] | None,
        ) -> Array:
            if field is None:
                return jnp.zeros_like(positions)
            if callable(field):
                try:
                    value = jnp.asarray(field(positions, t))
                except TypeError:
                    value = jax.vmap(lambda pos: jnp.asarray(field(pos, t)))(positions)
                if value.shape == (3,):
                    return jnp.broadcast_to(value, positions.shape)
                if value.shape != positions.shape:
                    raise ConfigurationError(
                        "External field callable must return shape (3,) or "
                        f"{positions.shape}, got {value.shape}"
                    )
                return value

            value = jnp.asarray(field)
            if value.shape != (3,):
                raise ConfigurationError(
                    f"External field vector must have shape (3,), got {value.shape}"
                )
            return jnp.broadcast_to(value, positions.shape)

        e_field = evaluate_field(E_ext)
        b_field = evaluate_field(B_ext)

        # Lorentz force: F = q * (E + v x B)
        force_ext = charges[:, None] * (e_field + jnp.cross(velocities, b_field))

        total_force = force_coulomb + force_ext
        return total_force / masses[:, None]

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 1e-3),
        n_steps: int = 10000,
        save_every: int = 10,
    ) -> NBodyTrajectory:
        """Simulate the charge system using velocity Verlet.

        Args:
            t_span: Time interval (seconds).
            n_steps: Number of integration steps.
            save_every: Save every N steps.

        Returns:
            NBodyTrajectory with positions and velocities over time.
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        charges = self._charges
        masses = self._masses
        softening = self._softening
        E_ext = self._E_ext
        B_ext = self._B_ext

        logger.info("Starting charge simulation: n=%d, n_steps=%d", self._n, n_steps)

        def verlet_step(
            carry: tuple[Array, Array, Array, float],
            _: None,
        ) -> tuple[
            tuple[Array, Array, Array, float],
            tuple[Array, Array, Array],
        ]:
            pos, vel, acc, t = carry
            pos_new = pos + vel * dt + 0.5 * acc * dt**2
            acc_new = self._compute_accelerations(
                pos_new, vel, charges, masses, softening, E_ext, B_ext, t + dt
            )
            vel_new = vel + 0.5 * (acc + acc_new) * dt
            return (pos_new, vel_new, acc_new, t + dt), (
                pos_new,
                vel_new,
                jnp.asarray(t + dt),
            )

        acc0 = self._compute_accelerations(
            self._positions,
            self._velocities,
            charges,
            masses,
            softening,
            E_ext,
            B_ext,
            t_start,
        )

        init = (self._positions, self._velocities, acc0, t_start)
        _, (pos_hist, vel_hist, t_hist) = jax.lax.scan(
            verlet_step, init, None, length=n_steps
        )

        # Prepend initial
        pos_hist = jnp.concatenate([self._positions[None, :, :], pos_hist], axis=0)
        vel_hist = jnp.concatenate([self._velocities[None, :, :], vel_hist], axis=0)
        t_hist = jnp.concatenate([jnp.array([t_start]), t_hist])

        if save_every > 1:
            indices = jnp.arange(0, n_steps + 1, save_every)
            pos_hist = pos_hist[indices]
            vel_hist = vel_hist[indices]
            t_hist = t_hist[indices]

        return NBodyTrajectory(
            t=t_hist,
            positions=pos_hist,
            velocities=vel_hist,
            masses=masses,
        )
