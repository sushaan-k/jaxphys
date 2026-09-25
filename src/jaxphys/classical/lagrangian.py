"""Lagrangian mechanics engine.

Given a Lagrangian L(q, qdot, params), derives the Euler-Lagrange
equations of motion using JAX automatic differentiation and integrates
them forward in time.

The Euler-Lagrange equations are:
    d/dt (dL/dqdot) - dL/dq = 0

Expanding via the chain rule:
    (d^2L/dqdot dq) * qdot + (d^2L/dqdot dqdot) * qddot - dL/dq = 0

Solving for qddot:
    qddot = M^{-1} @ (dL/dq - (d^2L/dqdot dq) @ qdot)

where M = d^2L/dqdot^2 is the mass matrix (Hessian of L w.r.t. qdot).

References:
    - Goldstein, Poole, Safko. "Classical Mechanics" (2002), Ch. 2
    - Feynman. "The Feynman Lectures on Physics", Vol. II, Ch. 19
"""

from __future__ import annotations

import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys._rollout import is_traced
from jaxphys.classical.integrators import integrate_system
from jaxphys.exceptions import (
    ConfigurationError,
    NumericalInstabilityError,
)
from jaxphys.state import Trajectory
from jaxphys.types import LagrangianFn

logger = logging.getLogger(__name__)

# Integrators that assume separable Hamiltonian structure (dp/dt depends
# only on q, dq/dt depends only on p).  LagrangianSystem._deriv_fn
# returns (qdot, qddot) where qddot depends on *both* q and qdot,
# which breaks the symplectic splitting.
_SYMPLECTIC_INTEGRATORS = frozenset(
    {"symplectic_euler", "leapfrog", "stormer_verlet", "yoshida4"}
)


class LagrangianSystem:
    """A mechanical system defined by its Lagrangian.

    Automatically derives equations of motion from the Lagrangian
    using JAX autodiff, then integrates using symplectic or
    general-purpose integrators.

    Example:
        >>> import jax.numpy as jnp
        >>> def lagrangian(q, qdot, params):
        ...     m, g, l = params.m, params.g, params.l
        ...     T = 0.5 * m * (l * qdot[0])**2
        ...     V = -m * g * l * jnp.cos(q[0])
        ...     return T - V
        >>> system = LagrangianSystem(lagrangian, n_dof=1)
        >>> params = Params(m=1.0, g=9.81, l=1.0)
        >>> traj = system.simulate(
        ...     q0=[0.3], qdot0=[0.0], t_span=(0, 10),
        ...     dt=0.01, params=params
        ... )

    Args:
        lagrangian: Function L(q, qdot, params) -> scalar.
        n_dof: Number of degrees of freedom.
    """

    def __init__(self, lagrangian: LagrangianFn, n_dof: int) -> None:
        if n_dof < 1:
            raise ConfigurationError(f"n_dof must be >= 1, got {n_dof}")
        self._lagrangian = lagrangian
        self._n_dof = n_dof
        self._build_eom()

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._n_dof

    def _build_eom(self) -> None:
        """Build the equations of motion using JAX autodiff.

        Computes the acceleration qddot by solving:
            M @ qddot = dL/dq - (d^2L/dqdot dq) @ qdot

        where M = d^2L/dqdot^2 is the generalized mass matrix.
        """
        L = self._lagrangian

        # Partial derivatives of L
        dL_dq = jax.grad(L, argnums=0)
        dL_dqdot = jax.grad(L, argnums=1)

        # Hessian of L w.r.t. qdot (mass matrix)
        d2L_dqdot2 = jax.hessian(L, argnums=1)

        # Mixed partial: d^2L / (dqdot dq)
        d2L_dqdot_dq = jax.jacfwd(dL_dqdot, argnums=0)

        def compute_acceleration(q: Array, qdot: Array, params: Any) -> Array:
            """Compute generalized acceleration from the Lagrangian.

            Solves M @ qddot = rhs for qddot.
            """
            mass_matrix = d2L_dqdot2(q, qdot, params)
            grad_q = dL_dq(q, qdot, params)
            mixed = d2L_dqdot_dq(q, qdot, params)

            rhs = grad_q - mixed @ qdot
            qddot = jnp.linalg.solve(mass_matrix, rhs)
            return cast(Array, qddot)

        self._compute_accel = compute_acceleration

    def acceleration(self, q: Array, qdot: Array, params: Any) -> Array:
        """Compute the generalized acceleration.

        Args:
            q: Generalized coordinates, shape (n_dof,).
            qdot: Generalized velocities, shape (n_dof,).
            params: System parameters.

        Returns:
            Generalized acceleration qddot, shape (n_dof,).
        """
        return self._compute_accel(q, qdot, params)

    def energy(self, q: Array, qdot: Array, params: Any) -> Array:
        """Compute total energy E = qdot . dL/dqdot - L (Jacobi integral).

        For natural Lagrangians (T - V with T quadratic in qdot),
        this equals T + V.

        Args:
            q: Generalized coordinates, shape (n_dof,).
            qdot: Generalized velocities, shape (n_dof,).
            params: System parameters.

        Returns:
            Scalar energy value.
        """
        dL_dqdot = jax.grad(self._lagrangian, argnums=1)
        L_val = self._lagrangian(q, qdot, params)
        p = dL_dqdot(q, qdot, params)
        return jnp.dot(qdot, p) - L_val

    def _deriv_fn(
        self, q: Array, p: Array, t: float, params: Any
    ) -> tuple[Array, Array]:
        """Derivative function for integrators.

        Maps (q, qdot, t) -> (dq/dt, dqdot/dt) = (qdot, qddot).
        """
        qddot = self._compute_accel(q, p, params)
        return p, qddot

    def simulate(
        self,
        q0: Array | list[float],
        qdot0: Array | list[float],
        t_span: tuple[float, float],
        dt: float = 0.001,
        params: Any = None,
        integrator: str = "rk4",
        save_every: int = 1,
    ) -> Trajectory:
        """Simulate the system forward in time.

        Args:
            q0: Initial generalized coordinates.
            qdot0: Initial generalized velocities.
            t_span: (t_start, t_end) time interval.
            dt: Time step size.
            params: System parameters (Params object or similar).
            integrator: Integrator name. Options: "euler",
                "symplectic_euler", "leapfrog", "rk4", "yoshida4".
            save_every: Save state every N steps.

        Returns:
            Trajectory containing the full time evolution.

        Raises:
            ConfigurationError: If parameters are invalid.
            NumericalInstabilityError: If NaN/Inf values are detected.
        """
        q = jnp.asarray(q0, dtype=jnp.float64)
        qdot = jnp.asarray(qdot0, dtype=jnp.float64)

        if q.shape != (self._n_dof,):
            raise ConfigurationError(f"q0 shape {q.shape} != expected ({self._n_dof},)")
        if qdot.shape != (self._n_dof,):
            raise ConfigurationError(
                f"qdot0 shape {qdot.shape} != expected ({self._n_dof},)"
            )

        if integrator in _SYMPLECTIC_INTEGRATORS:
            raise ConfigurationError(
                f"Integrator '{integrator}' is not compatible with "
                "LagrangianSystem because the derived equations of motion "
                "are not separable into position-only and momentum-only "
                "updates. Use 'rk4' or a Hamiltonian formulation instead."
            )

        logger.info(
            "Starting Lagrangian simulation: n_dof=%d, integrator=%s",
            self._n_dof,
            integrator,
        )
        q_hist, p_hist, t_hist, e_hist = integrate_system(
            integrator,
            self._deriv_fn,
            self.energy,
            q,
            qdot,
            t_span,
            dt,
            params,
            save_every,
        )

        if not is_traced(q_hist) and bool(jnp.any(jnp.isnan(q_hist))):
            raise NumericalInstabilityError(
                f"NaN detected in trajectory. Try reducing dt "
                f"(currently {dt}) or using a symplectic integrator."
            )

        return Trajectory(
            t=t_hist,
            q=q_hist,
            p=p_hist,
            energy=e_hist,
        )
