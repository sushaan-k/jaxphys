"""Charge dynamics simulation.

Simulates the motion of charged particles in electric and magnetic fields
using the Lorentz force law:

    F = q * (E + v x B)

Supports both prescribed external fields and self-consistent particle-particle
Coulomb interactions. Time integration uses Boris velocity kicks in a
kick-drift-kick (velocity-Verlet-like) scheme.

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

from jaxphys._rollout import strided_rollout
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import NBodyTrajectory

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
        self._charges = jnp.array([c.charge for c in charges], dtype=jnp.float64)
        self._masses = jnp.array([c.mass for c in charges], dtype=jnp.float64)
        self._positions = jnp.array([c.position for c in charges], dtype=jnp.float64)
        self._velocities = jnp.array([c.velocity for c in charges], dtype=jnp.float64)
        self._softening = softening

        self._E_ext = E_external
        self._B_ext = B_external
        # Compiled once per system; reused by every simulate() call with the
        # same (n_steps, save_every).
        self._rollout = jax.jit(
            self._rollout_impl, static_argnames=("n_steps", "save_every")
        )

    @property
    def n_charges(self) -> int:
        """Number of charges."""
        return self._n

    def _fields(self, positions: Array, t: Array | float) -> tuple[Array, Array]:
        """Electric field (Coulomb + external) and magnetic field at each charge.

        Args:
            positions: Shape (n, 3).
            t: Current simulation time.

        Returns:
            ``(E, B)``, each of shape (n, 3).
        """
        n = self._n

        # Coulomb field at charge i from all j != i:
        #   E_i = k * sum_j q_j (r_i - r_j) / |r_ij|^3,  dr[i, j] = r_j - r_i
        dr = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]
        self_pair = jnp.eye(n, dtype=bool)
        dist_sq = jnp.where(
            self_pair, 1.0, jnp.sum(dr**2, axis=-1) + self._softening**2
        )
        w = jnp.where(self_pair, 0.0, self._charges[jnp.newaxis, :] * dist_sq**-1.5)
        e_coulomb = -K_COULOMB * jnp.einsum("ij,ijk->ik", w, dr)

        def evaluate_field(
            field: Array | Callable[[Array, float], Array] | None,
        ) -> Array:
            if field is None:
                return jnp.zeros_like(positions)
            if callable(field):
                try:
                    value = jnp.asarray(field(positions, t))  # type: ignore[arg-type]
                except TypeError:
                    value = jax.vmap(lambda pos: jnp.asarray(field(pos, t)))(  # type: ignore[arg-type]
                        positions
                    )
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

        return e_coulomb + evaluate_field(self._E_ext), evaluate_field(self._B_ext)

    def _boris_kick(
        self, positions: Array, velocities: Array, t: Array, h: Array
    ) -> Array:
        """Advance velocities by ``h`` under the Lorentz force (Boris rotation).

        Half electric kick, exact-norm magnetic rotation, half electric kick.
        With E = 0 the speed of every particle is conserved to round-off.
        """
        e_field, b_field = self._fields(positions, t)
        qm = (self._charges / self._masses)[:, None]
        v_minus = velocities + qm * e_field * (0.5 * h)
        tvec = qm * b_field * (0.5 * h)
        v_prime = v_minus + jnp.cross(v_minus, tvec)
        svec = 2.0 * tvec / (1.0 + jnp.sum(tvec**2, axis=-1, keepdims=True))
        v_plus = v_minus + jnp.cross(v_prime, svec)
        return v_plus + qm * e_field * (0.5 * h)

    def _rollout_impl(
        self, t0: Array, dt: Array, n_steps: int, save_every: int
    ) -> tuple[Array, Array, Array]:
        def step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            pos, vel, t = carry
            # Kick-drift-kick with Boris kicks: second order, time
            # reversible, and energy-conserving in a pure magnetic field.
            vel = self._boris_kick(pos, vel, t, 0.5 * dt)
            pos = pos + dt * vel
            vel = self._boris_kick(pos, vel, t + dt, 0.5 * dt)
            return pos, vel, t + dt

        out: tuple[Array, Array, Array] = strided_rollout(
            step,
            (self._positions, self._velocities, t0),
            n_steps,
            save_every,
            lambda carry: carry,
        )
        return out

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 1e-3),
        n_steps: int = 10000,
        save_every: int = 10,
    ) -> NBodyTrajectory:
        """Simulate the charge system.

        Uses a kick-drift-kick scheme whose velocity kicks are Boris
        pushes (Boris 1970), the standard integrator for the Lorentz force:
        unlike velocity Verlet it treats the velocity-dependent magnetic
        force consistently and conserves kinetic energy in a pure B field.

        Args:
            t_span: Time interval (seconds).
            n_steps: Number of integration steps.
            save_every: Save every N steps.

        Returns:
            NBodyTrajectory with positions and velocities over time.
        """
        if n_steps < 1:
            raise ConfigurationError(f"n_steps must be >= 1, got {n_steps}")
        if save_every < 1:
            raise ConfigurationError(f"save_every must be >= 1, got {save_every}")
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps

        logger.info("Starting charge simulation: n=%d, n_steps=%d", self._n, n_steps)

        pos_hist, vel_hist, t_hist = self._rollout(
            jnp.asarray(t_start, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
            n_steps=n_steps,
            save_every=save_every,
        )

        return NBodyTrajectory(
            t=t_hist,
            positions=pos_hist,
            velocities=vel_hist,
            masses=self._masses,
        )
