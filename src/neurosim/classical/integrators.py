"""Symplectic and general-purpose numerical integrators.

Provides a suite of integrators for Hamiltonian and general ODE systems.
Symplectic integrators (leapfrog, Stormer-Verlet, Yoshida) preserve the
geometric structure of Hamiltonian flow, yielding bounded energy error
over exponentially long times.

All integrators are JIT-compatible and differentiable through JAX.

References:
    - Hairer, Lubich, Wanner. "Geometric Numerical Integration" (2006)
    - Yoshida. "Construction of higher order symplectic integrators" (1990)
"""

from collections.abc import Callable
from typing import Any

from jax import Array

# Type alias for a derivative function: (q, p, t, params) -> (dq/dt, dp/dt)
DerivFn = Callable[[Array, Array, float, Any], tuple[Array, Array]]

# Type alias for acceleration: (q, v, t, params) -> a
AccelFn = Callable[[Array, Array, float, Any], Array]


def euler(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Forward Euler integrator.

    First-order, non-symplectic. Useful only as a baseline;
    not recommended for production simulations.

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, p_new, t_new).
    """
    dq, dp = deriv_fn(q, p, t, params)
    return q + dt * dq, p + dt * dp, t + dt


def symplectic_euler(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Symplectic Euler (semi-implicit Euler) integrator.

    First-order symplectic. Updates p first, then uses the new p
    to update q. Preserves phase-space volume.

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, p_new, t_new).
    """
    _, dp = deriv_fn(q, p, t, params)
    p_new = p + dt * dp
    dq, _ = deriv_fn(q, p_new, t, params)
    q_new = q + dt * dq
    return q_new, p_new, t + dt


def leapfrog(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Leapfrog (Stormer-Verlet) integrator.

    Second-order symplectic. The workhorse of Hamiltonian simulation.
    Time-reversible with O(dt^2) local error and bounded energy drift.

    The leapfrog scheme:
        p_{1/2} = p_n + (dt/2) * dp/dt(q_n, p_n)
        q_{n+1} = q_n + dt * dq/dt(q_n, p_{1/2})
        p_{n+1} = p_{1/2} + (dt/2) * dp/dt(q_{n+1}, p_{1/2})

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, p_new, t_new).
    """
    _, dp1 = deriv_fn(q, p, t, params)
    p_half = p + 0.5 * dt * dp1

    dq, _ = deriv_fn(q, p_half, t + 0.5 * dt, params)
    q_new = q + dt * dq

    _, dp2 = deriv_fn(q_new, p_half, t + dt, params)
    p_new = p_half + 0.5 * dt * dp2

    return q_new, p_new, t + dt


def velocity_verlet(
    accel_fn: AccelFn,
    q: Array,
    v: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Velocity Verlet integrator.

    Second-order symplectic, equivalent to leapfrog but formulated
    in terms of positions and velocities with an explicit acceleration
    function. Preferred for N-body problems.

    The scheme:
        q_{n+1} = q_n + v_n * dt + 0.5 * a_n * dt^2
        a_{n+1} = accel(q_{n+1})
        v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt

    Args:
        accel_fn: Function (q, v, t, params) -> acceleration.
        q: Positions.
        v: Velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, v_new, t_new).
    """
    a = accel_fn(q, v, t, params)
    q_new = q + v * dt + 0.5 * a * dt**2

    a_new = accel_fn(q_new, v, t + dt, params)
    v_new = v + 0.5 * (a + a_new) * dt

    return q_new, v_new, t + dt


def stormer_verlet(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Stormer-Verlet integrator (position form).

    Identical to leapfrog but written in the traditional Stormer-Verlet
    form. Included for API clarity.

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, p_new, t_new).
    """
    return leapfrog(deriv_fn, q, p, t, dt, params)


def yoshida4(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Fourth-order Yoshida symplectic integrator.

    Constructed by composing three leapfrog steps with specially chosen
    substep sizes. O(dt^4) local error with symplectic structure preserved.

    The Yoshida coefficients satisfy:
        c1 + c2 + c3 = 1  (consistency)
        c1^3 + c2^3 + c3^3 = 0  (fourth-order)

    Reference:
        Yoshida, H. "Construction of higher order symplectic integrators"
        Physics Letters A, 150(5-7), 262-268 (1990).

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, p_new, t_new).
    """
    # Yoshida fourth-order coefficients
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)

    d1 = w1
    d2 = w0
    d3 = w1
    c1 = w1 / 2.0
    c2 = (w0 + w1) / 2.0
    c3 = c2
    c4 = c1

    # Step 1
    dq, _ = deriv_fn(q, p, t, params)
    q = q + c1 * dt * dq
    _, dp = deriv_fn(q, p, t + c1 * dt, params)
    p = p + d1 * dt * dp

    # Step 2
    dq, _ = deriv_fn(q, p, t + (c1 + d1) * dt, params)
    q = q + c2 * dt * dq
    _, dp = deriv_fn(q, p, t + (c1 + d1 + c2) * dt, params)
    p = p + d2 * dt * dp

    # Step 3
    dq, _ = deriv_fn(q, p, t + (c1 + d1 + c2 + d2) * dt, params)
    q = q + c3 * dt * dq
    _, dp = deriv_fn(q, p, t + (c1 + d1 + c2 + d2 + c3) * dt, params)
    p = p + d3 * dt * dp

    # Final position update
    dq, _ = deriv_fn(q, p, t + dt, params)
    q = q + c4 * dt * dq

    return q, p, t + dt


def rk4(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
) -> tuple[Array, Array, float]:
    """Classical fourth-order Runge-Kutta integrator.

    Fourth-order accurate but NOT symplectic. Use for non-Hamiltonian
    systems or when high accuracy per step matters more than long-time
    energy conservation.

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Time step.
        params: System parameters.

    Returns:
        Tuple of (q_new, p_new, t_new).
    """
    dq1, dp1 = deriv_fn(q, p, t, params)

    dq2, dp2 = deriv_fn(
        q + 0.5 * dt * dq1,
        p + 0.5 * dt * dp1,
        t + 0.5 * dt,
        params,
    )

    dq3, dp3 = deriv_fn(
        q + 0.5 * dt * dq2,
        p + 0.5 * dt * dp2,
        t + 0.5 * dt,
        params,
    )

    dq4, dp4 = deriv_fn(
        q + dt * dq3,
        p + dt * dp3,
        t + dt,
        params,
    )

    q_new = q + (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
    p_new = p + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)

    return q_new, p_new, t + dt


# Registry mapping integrator names to functions
INTEGRATORS: dict[str, Callable[..., tuple[Array, Array, float]]] = {
    "euler": euler,
    "symplectic_euler": symplectic_euler,
    "leapfrog": leapfrog,
    "velocity_verlet": velocity_verlet,
    "stormer_verlet": stormer_verlet,
    "yoshida4": yoshida4,
    "rk4": rk4,
}


def get_integrator(
    name: str,
) -> Callable[..., tuple[Array, Array, float]]:
    """Look up an integrator by name.

    Args:
        name: Integrator name (e.g., "rk4", "leapfrog", "yoshida4").

    Returns:
        The integrator function.

    Raises:
        ValueError: If the integrator name is not recognized.
    """
    if name not in INTEGRATORS:
        available = ", ".join(sorted(INTEGRATORS.keys()))
        raise ValueError(f"Unknown integrator '{name}'. Available: {available}")
    return INTEGRATORS[name]
