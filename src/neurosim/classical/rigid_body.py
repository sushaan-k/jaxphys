"""Rigid body dynamics.

Simulates rotational dynamics of rigid bodies using Euler's equations
and quaternion-based orientation tracking.

Euler's equations for a torque-free rigid body:
    I_1 * dwx/dt = (I_2 - I_3) * wy * wz
    I_2 * dwy/dt = (I_3 - I_1) * wz * wx
    I_3 * dwz/dt = (I_1 - I_2) * wx * wy

where I_1, I_2, I_3 are the principal moments of inertia and
(wx, wy, wz) is the angular velocity in the body frame.

References:
    - Goldstein, Poole, Safko. "Classical Mechanics" (2002), Ch. 5
    - Diebel. "Representing Attitude: Euler Angles, Unit Quaternions,
      and Rotation Vectors" (2006)
"""

from __future__ import annotations

import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError
from neurosim.state import Trajectory

logger = logging.getLogger(__name__)


class RigidBody:
    """Rigid body dynamics simulator.

    Tracks rotational state using quaternions for singularity-free
    orientation representation and integrates Euler's equations.

    Args:
        inertia: Principal moments of inertia (I1, I2, I3).
        torque_fn: Optional external torque function
            (omega, t, params) -> torque_vector.

    Example:
        >>> body = RigidBody(inertia=[1.0, 2.0, 3.0])
        >>> traj = body.simulate(
        ...     omega0=[1.0, 0.1, 0.0],
        ...     t_span=(0, 50), dt=0.01,
        ... )
    """

    def __init__(
        self,
        inertia: list[float] | Array,
        torque_fn: Any | None = None,
    ) -> None:
        self._inertia = jnp.asarray(inertia, dtype=jnp.float64)
        if self._inertia.shape != (3,):
            raise ConfigurationError(
                f"inertia must have shape (3,), got {self._inertia.shape}"
            )
        if jnp.any(self._inertia <= 0):
            raise ConfigurationError(
                "All principal moments of inertia must be positive"
            )
        self._torque_fn = torque_fn

    @property
    def inertia(self) -> Array:
        """Principal moments of inertia."""
        return self._inertia

    def _euler_equations(self, omega: Array, t: float, params: Any) -> Array:
        """Euler's equations for rigid body rotation.

        Args:
            omega: Angular velocity in body frame, shape (3,).
            t: Current time.
            params: Optional parameters for torque function.

        Returns:
            Angular acceleration domega/dt, shape (3,).
        """
        inertia = self._inertia
        wx, wy, wz = omega[0], omega[1], omega[2]

        domega = jnp.array(
            [
                (inertia[1] - inertia[2]) * wy * wz / inertia[0],
                (inertia[2] - inertia[0]) * wz * wx / inertia[1],
                (inertia[0] - inertia[1]) * wx * wy / inertia[2],
            ]
        )

        if self._torque_fn is not None:
            tau = self._torque_fn(omega, t, params)
            domega = domega + jnp.asarray(tau) / inertia

        return domega

    def _quaternion_deriv(self, quat: Array, omega: Array) -> Array:
        """Time derivative of the orientation quaternion.

        dq/dt = 0.5 * q * omega_quat

        where omega_quat = (0, wx, wy, wz) is the angular velocity
        as a pure quaternion.

        Args:
            quat: Unit quaternion (w, x, y, z), shape (4,).
            omega: Angular velocity, shape (3,).

        Returns:
            dq/dt, shape (4,).
        """
        w, x, y, z = quat
        wx, wy, wz = omega

        return 0.5 * jnp.array(
            [
                -x * wx - y * wy - z * wz,
                w * wx + y * wz - z * wy,
                w * wy + z * wx - x * wz,
                w * wz + x * wy - y * wx,
            ]
        )

    def _normalize_quaternion(self, quat: Array) -> Array:
        """Normalize a quaternion to unit length."""
        return cast(Array, quat / jnp.linalg.norm(quat))

    def rotational_energy(self, omega: Array) -> Array:
        """Compute rotational kinetic energy: T = 0.5 * I . omega^2.

        Args:
            omega: Angular velocity, shape (3,).

        Returns:
            Scalar kinetic energy.
        """
        return 0.5 * jnp.sum(self._inertia * omega**2)

    def angular_momentum(self, omega: Array) -> Array:
        """Compute angular momentum L = I * omega in body frame.

        Args:
            omega: Angular velocity, shape (3,).

        Returns:
            Angular momentum vector, shape (3,).
        """
        return self._inertia * omega

    def simulate(
        self,
        omega0: list[float] | Array,
        t_span: tuple[float, float] = (0.0, 10.0),
        dt: float = 0.01,
        params: Any = None,
        quat0: list[float] | Array | None = None,
    ) -> Trajectory:
        """Simulate rigid body rotation.

        Uses RK4 integration for both Euler's equations (angular velocity)
        and quaternion evolution (orientation).

        Args:
            omega0: Initial angular velocity in body frame.
            t_span: Time interval.
            dt: Time step.
            params: Parameters for external torque function.
            quat0: Initial quaternion (w, x, y, z). Default is identity.

        Returns:
            Trajectory where q stores quaternions (n_steps, 4) and
            p stores angular velocities (n_steps, 3).
        """
        omega = jnp.asarray(omega0, dtype=jnp.float64)
        if omega.shape != (3,):
            raise ConfigurationError(f"omega0 must have shape (3,), got {omega.shape}")

        if quat0 is None:
            quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat = jnp.asarray(quat0, dtype=jnp.float64)
            quat = self._normalize_quaternion(quat)

        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        logger.info(
            "Starting rigid body simulation: I=%s, n_steps=%d",
            self._inertia,
            n_steps,
        )

        def rk4_step(
            carry: tuple[Array, Array, float],
            _: None,
        ) -> tuple[
            tuple[Array, Array, float],
            tuple[Array, Array, Array, Array],
        ]:
            quat_c, omega_c, t_c = carry

            # RK4 for omega (Euler's equations)
            k1_o = self._euler_equations(omega_c, t_c, params)
            k2_o = self._euler_equations(
                omega_c + 0.5 * dt * k1_o, t_c + 0.5 * dt, params
            )
            k3_o = self._euler_equations(
                omega_c + 0.5 * dt * k2_o, t_c + 0.5 * dt, params
            )
            k4_o = self._euler_equations(omega_c + dt * k3_o, t_c + dt, params)
            omega_new = omega_c + (dt / 6.0) * (k1_o + 2 * k2_o + 2 * k3_o + k4_o)

            # RK4 for quaternion
            k1_q = self._quaternion_deriv(quat_c, omega_c)
            k2_q = self._quaternion_deriv(
                quat_c + 0.5 * dt * k1_q,
                omega_c + 0.5 * dt * k1_o,
            )
            k3_q = self._quaternion_deriv(
                quat_c + 0.5 * dt * k2_q,
                omega_c + 0.5 * dt * k2_o,
            )
            k4_q = self._quaternion_deriv(
                quat_c + dt * k3_q,
                omega_c + dt * k3_o,
            )
            quat_new = quat_c + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            quat_new = self._normalize_quaternion(quat_new)

            e = self.rotational_energy(omega_new)
            return (quat_new, omega_new, t_c + dt), (
                quat_new,
                omega_new,
                jnp.asarray(t_c + dt),
                e,
            )

        init = (quat, omega, t_start)
        _, (q_hist, o_hist, t_hist, e_hist) = jax.lax.scan(
            rk4_step, init, None, length=n_steps
        )

        # Prepend initial state
        e0 = self.rotational_energy(omega)
        q_hist = jnp.concatenate([quat[None, :], q_hist], axis=0)
        o_hist = jnp.concatenate([omega[None, :], o_hist], axis=0)
        t_hist = jnp.concatenate([jnp.array([t_start]), t_hist])
        e_hist = jnp.concatenate([jnp.array([e0]), e_hist])

        return Trajectory(
            t=t_hist,
            q=q_hist,
            p=o_hist,
            energy=e_hist,
        )
