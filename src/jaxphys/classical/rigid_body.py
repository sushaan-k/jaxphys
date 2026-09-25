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

from jaxphys._rollout import call_with_params, strided_rollout
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import Trajectory

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

    def _euler_equations(self, omega: Array, t: Array | float, params: Any) -> Array:
        """Euler's equations for rigid body rotation.

        Args:
            omega: Angular velocity in body frame, shape (3,).
            t: Current time.
            params: Optional parameters for torque function.

        Returns:
            Angular acceleration domega/dt, shape (3,).
        """
        return _euler_rhs(self._inertia, self._torque_fn, omega, t, params)

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
        return _quaternion_deriv(quat, omega)

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

    def _rk4_step(
        self, carry: tuple[Array, Array, Array], dt: Array, params: Any
    ) -> tuple[Array, Array, Array]:
        """One RK4 step of the coupled (quaternion, omega) system."""
        return _rk4_step(self._inertia, self._torque_fn, carry, dt, params)

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
        if dt <= 0:
            raise ConfigurationError(f"dt must be positive, got {dt}")
        n_steps = int((t_end - t_start) / dt)

        logger.info(
            "Starting rigid body simulation: I=%s, n_steps=%d",
            self._inertia,
            n_steps,
        )

        out = call_with_params(
            _rollout,
            _rollout_impl,
            params,
            _STATIC,
            inertia=self._inertia,
            quat=quat,
            omega=omega,
            t0=jnp.asarray(t_start, dtype=jnp.float64),
            dt=jnp.asarray(dt, dtype=jnp.float64),
            torque_fn=self._torque_fn,
            n_steps=n_steps,
        )
        q_hist, o_hist, t_hist, e_hist = out

        return Trajectory(
            t=t_hist,
            q=q_hist,
            p=o_hist,
            energy=e_hist,
        )


def _euler_rhs(
    inertia: Array, torque_fn: Any | None, omega: Array, t: Array | float, params: Any
) -> Array:
    """Euler's equations ``I domega/dt = (I omega) x omega + tau`` (body frame)."""
    wx, wy, wz = omega[0], omega[1], omega[2]

    domega = jnp.array(
        [
            (inertia[1] - inertia[2]) * wy * wz / inertia[0],
            (inertia[2] - inertia[0]) * wz * wx / inertia[1],
            (inertia[0] - inertia[1]) * wx * wy / inertia[2],
        ]
    )

    if torque_fn is not None:
        tau = torque_fn(omega, t, params)
        domega = domega + jnp.asarray(tau) / inertia

    return domega


def _quaternion_deriv(quat: Array, omega: Array) -> Array:
    """``dq/dt = 0.5 * q * (0, omega)`` for a unit quaternion ``(w, x, y, z)``."""
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


def _rk4_step(
    inertia: Array,
    torque_fn: Any | None,
    carry: tuple[Array, Array, Array],
    dt: Array,
    params: Any,
) -> tuple[Array, Array, Array]:
    """One RK4 step of the coupled (quaternion, omega) system."""
    quat_c, omega_c, t_c = carry

    def f(omega: Array, t: Array) -> Array:
        return _euler_rhs(inertia, torque_fn, omega, t, params)

    k1_o = f(omega_c, t_c)
    k2_o = f(omega_c + 0.5 * dt * k1_o, t_c + 0.5 * dt)
    k3_o = f(omega_c + 0.5 * dt * k2_o, t_c + 0.5 * dt)
    k4_o = f(omega_c + dt * k3_o, t_c + dt)
    omega_new = omega_c + (dt / 6.0) * (k1_o + 2 * k2_o + 2 * k3_o + k4_o)

    k1_q = _quaternion_deriv(quat_c, omega_c)
    k2_q = _quaternion_deriv(quat_c + 0.5 * dt * k1_q, omega_c + 0.5 * dt * k1_o)
    k3_q = _quaternion_deriv(quat_c + 0.5 * dt * k2_q, omega_c + 0.5 * dt * k2_o)
    k4_q = _quaternion_deriv(quat_c + dt * k3_q, omega_c + dt * k3_o)
    quat_new = quat_c + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    return cast(Array, quat_new / jnp.linalg.norm(quat_new)), omega_new, t_c + dt


def _rollout_impl(
    inertia: Array,
    quat: Array,
    omega: Array,
    t0: Array,
    dt: Array,
    params: Any,
    torque_fn: Any | None,
    n_steps: int,
) -> tuple[Array, Array, Array, Array]:
    """RK4 rollout saving every step; returns (quat, omega, t, energy)."""

    def observe(carry: tuple[Array, Array, Array]) -> tuple[Array, ...]:
        _, omega_c, _ = carry
        return (*carry, 0.5 * jnp.sum(inertia * omega_c**2))

    out: tuple[Array, Array, Array, Array] = strided_rollout(
        lambda carry: _rk4_step(inertia, torque_fn, carry, dt, params),
        (quat, omega, t0),
        n_steps,
        1,
        observe,
    )
    return out


# Compiled once per (torque_fn, n_steps); the inertia, initial state, dt
# and array-valued params are traced, so new values do not recompile.
_STATIC = ("torque_fn", "n_steps")
_rollout = jax.jit(_rollout_impl, static_argnames=_STATIC)
