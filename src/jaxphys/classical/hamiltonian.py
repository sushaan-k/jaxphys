"""Hamiltonian mechanics engine.

Given a Hamiltonian H(q, p, params), derives Hamilton's equations
using JAX automatic differentiation and integrates them.

Hamilton's equations:
    dq/dt =  dH/dp
    dp/dt = -dH/dq

These are inherently symplectic, making this formulation ideal for
long-time integrations with symplectic integrators.

References:
    - Goldstein, Poole, Safko. "Classical Mechanics" (2002), Ch. 8
    - Arnold. "Mathematical Methods of Classical Mechanics" (1989)
"""

from __future__ import annotations

import logging
from typing import Any

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
from jaxphys.types import HamiltonianFn

logger = logging.getLogger(__name__)


class HamiltonianSystem:
    """A mechanical system defined by its Hamiltonian.

    Automatically derives Hamilton's equations from H(q, p, params)
    using JAX autodiff and integrates them forward in time.

    Example:
        >>> import jax.numpy as jnp
        >>> def hamiltonian(q, p, params):
        ...     return p[0]**2 / (2 * params.m) + 0.5 * params.k * q[0]**2
        >>> system = HamiltonianSystem(hamiltonian, n_dof=1)
        >>> params = Params(m=1.0, k=4.0)
        >>> traj = system.simulate(
        ...     q0=[1.0], p0=[0.0], t_span=(0, 10),
        ...     dt=0.01, params=params
        ... )

    Args:
        hamiltonian: Function H(q, p, params) -> scalar.
        n_dof: Number of degrees of freedom.
    """

    def __init__(self, hamiltonian: HamiltonianFn, n_dof: int) -> None:
        if n_dof < 1:
            raise ConfigurationError(f"n_dof must be >= 1, got {n_dof}")
        self._hamiltonian = hamiltonian
        self._n_dof = n_dof
        self._build_eom()

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._n_dof

    def _build_eom(self) -> None:
        """Build Hamilton's equations using JAX autodiff."""
        H = self._hamiltonian
        self._dH_dq = jax.grad(H, argnums=0)
        self._dH_dp = jax.grad(H, argnums=1)

    def energy(self, q: Array, p: Array, params: Any) -> Array:
        """Evaluate the Hamiltonian (total energy).

        Args:
            q: Generalized coordinates, shape (n_dof,).
            p: Generalized momenta, shape (n_dof,).
            params: System parameters.

        Returns:
            Scalar energy value.
        """
        return self._hamiltonian(q, p, params)

    def _deriv_fn(
        self, q: Array, p: Array, t: float, params: Any
    ) -> tuple[Array, Array]:
        """Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq."""
        dq_dt = self._dH_dp(q, p, params)
        dp_dt = -self._dH_dq(q, p, params)
        return dq_dt, dp_dt

    def simulate(
        self,
        q0: Array | list[float],
        p0: Array | list[float],
        t_span: tuple[float, float],
        dt: float = 0.001,
        params: Any = None,
        integrator: str = "leapfrog",
        save_every: int = 1,
    ) -> Trajectory:
        """Simulate the Hamiltonian system forward in time.

        Args:
            q0: Initial generalized coordinates.
            p0: Initial generalized momenta.
            t_span: (t_start, t_end) time interval.
            dt: Time step size.
            params: System parameters.
            integrator: Integrator name. Symplectic integrators
                recommended for long-time accuracy.
            save_every: Save state every N steps.

        Returns:
            Trajectory with full time evolution.

        Raises:
            ConfigurationError: If parameters are invalid.
            NumericalInstabilityError: If NaN values are detected.
        """
        q = jnp.asarray(q0, dtype=jnp.float64)
        p = jnp.asarray(p0, dtype=jnp.float64)

        if q.shape != (self._n_dof,):
            raise ConfigurationError(f"q0 shape {q.shape} != expected ({self._n_dof},)")
        if p.shape != (self._n_dof,):
            raise ConfigurationError(f"p0 shape {p.shape} != expected ({self._n_dof},)")

        logger.info(
            "Starting Hamiltonian simulation: n_dof=%d, integrator=%s",
            self._n_dof,
            integrator,
        )
        q_hist, p_hist, t_hist, e_hist = integrate_system(
            integrator,
            self._deriv_fn,
            self.energy,
            q,
            p,
            t_span,
            dt,
            params,
            save_every,
        )

        if not is_traced(q_hist) and bool(jnp.any(jnp.isnan(q_hist))):
            raise NumericalInstabilityError(
                f"NaN detected in Hamiltonian trajectory. "
                f"Try reducing dt (currently {dt})."
            )

        return Trajectory(
            t=t_hist,
            q=q_hist,
            p=p_hist,
            energy=e_hist,
        )
