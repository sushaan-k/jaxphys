"""Coupled oscillators convenience builder.

Constructs a Lagrangian for N masses connected by springs in a
one-dimensional chain and returns a ready-to-simulate
:class:`LagrangianSystem`.

The system consists of N masses m_1, ..., m_N connected by springs
of stiffness k between each adjacent pair.  The first mass is
anchored to a fixed wall by a spring, and the last mass has a free
end (unless the caller adds boundary conditions via the returned
Lagrangian).

The Lagrangian is:
    L = T - V
    T = sum_i 0.5 * m_i * qdot_i^2
    V = 0.5 * k * q_0^2  +  sum_{i=1}^{N-1} 0.5 * k * (q_i - q_{i-1})^2

The normal-mode frequencies for equal masses (m_i = m) with a fixed
wall on the left and a free end on the right are:

    omega_j = 2 * sqrt(k/m) * sin((2j - 1) * pi / (4N + 2))   for j = 1 ... N

References:
    - Goldstein, Poole, Safko. "Classical Mechanics" (2002), Ch. 6
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from neurosim.classical.lagrangian import LagrangianSystem
from neurosim.exceptions import ConfigurationError


def coupled_oscillators(
    n: int,
    k: float = 1.0,
    m: float = 1.0,
) -> LagrangianSystem:
    """Build a 1-D coupled-oscillator chain.

    Args:
        n: Number of masses (degrees of freedom).
        k: Spring constant (uniform across all springs).
        m: Mass of each particle (uniform).

    Returns:
        A :class:`LagrangianSystem` ready for simulation.  Pass
        ``params=None`` to :meth:`simulate` since all physical
        parameters are baked into the Lagrangian closure.

    Raises:
        ConfigurationError: If *n* < 1, *k* <= 0, or *m* <= 0.

    Example:
        >>> system = coupled_oscillators(3, k=2.0, m=1.0)
        >>> traj = system.simulate(
        ...     q0=[0.1, 0.0, 0.0],
        ...     qdot0=[0.0, 0.0, 0.0],
        ...     t_span=(0, 20),
        ...     dt=0.01,
        ...     params=None,
        ... )
    """
    if n < 1:
        raise ConfigurationError(f"n must be >= 1, got {n}")
    if k <= 0:
        raise ConfigurationError(f"k must be > 0, got {k}")
    if m <= 0:
        raise ConfigurationError(f"m must be > 0, got {m}")

    def lagrangian(q: Array, qdot: Array, _params: Any) -> Array:
        # Kinetic energy: sum 0.5 * m * qdot_i^2
        T = 0.5 * m * jnp.sum(qdot**2)

        # Potential energy: wall spring + inter-mass springs
        V_wall = 0.5 * k * q[0] ** 2
        if n > 1:
            diffs = q[1:] - q[:-1]
            V_springs = 0.5 * k * jnp.sum(diffs**2)
        else:
            V_springs = 0.0
        V = V_wall + V_springs

        return T - V

    return LagrangianSystem(lagrangian, n_dof=n)


def normal_mode_frequencies(
    n: int,
    k: float = 1.0,
    m: float = 1.0,
) -> Array:
    """Compute analytical normal-mode frequencies for the coupled chain.

    Returns the *n* eigenfrequencies (angular) for the fixed-wall /
    free-end boundary condition used by :func:`coupled_oscillators`.

    The formula is:
        omega_j = 2 * sqrt(k/m) * sin((2j - 1) * pi / (4N + 2))

    Args:
        n: Number of masses.
        k: Spring constant.
        m: Mass per particle.

    Returns:
        Array of shape (n,) with frequencies sorted ascending.
    """
    j = jnp.arange(1, n + 1)
    return 2.0 * jnp.sqrt(k / m) * jnp.sin(
        (2 * j - 1) * jnp.pi / (4 * n + 2)
    )
