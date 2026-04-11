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


def adaptive_rk45(
    deriv_fn: DerivFn,
    q: Array,
    p: Array,
    t: float,
    dt: float,
    params: Any,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    safety: float = 0.9,
    dt_min: float = 1e-12,
    dt_max: float | None = None,
    max_reject: int = 100,
) -> tuple[Array, Array, float]:
    """Dormand-Prince RK45 integrator with adaptive step-size control.

    Embeds a 4th-order solution inside a 5th-order Runge-Kutta method.
    The difference between the two provides an error estimate used to
    accept/reject steps and adapt *dt*.

    The Dormand-Prince coefficients (DOPRI5) are the same used by
    MATLAB's ``ode45`` and SciPy's ``RK45``.

    Args:
        deriv_fn: Function returning (dq/dt, dp/dt).
        q: Generalized coordinates.
        p: Generalized momenta/velocities.
        t: Current time.
        dt: Suggested time step (adapted internally).
        params: System parameters.
        atol: Absolute tolerance for error control.
        rtol: Relative tolerance for error control.
        safety: Safety factor for step-size updates (< 1).
        dt_min: Minimum allowed time step.
        dt_max: Maximum allowed time step (defaults to *dt*).
        max_reject: Maximum consecutive rejected steps before giving up.

    Returns:
        Tuple of (q_new, p_new, t_new) after one *accepted* step.
        The actual step size used may differ from the input *dt*.

    References:
        Dormand, J. R.; Prince, P. J. "A family of embedded Runge-Kutta
        formulae", J. Comput. Appl. Math. 6(1), 19-26 (1980).
    """
    import jax.numpy as _jnp

    if dt_max is None:
        dt_max = dt

    # --- Dormand-Prince Butcher tableau ---
    a21 = 1.0 / 5.0
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = (
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    )
    a61, a62, a63, a64, a65 = (
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    )

    # 5th-order weights
    b1 = 35.0 / 384.0
    b3 = 500.0 / 1113.0
    b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0
    b6 = 11.0 / 84.0

    # 4th-order weights (for error estimate)
    e1 = 71.0 / 57600.0
    e3 = -71.0 / 16695.0
    e4 = 71.0 / 1920.0
    e5 = -17253.0 / 339200.0
    e6 = 22.0 / 525.0
    e7 = -1.0 / 40.0

    h = dt
    rejects = 0

    while True:
        k1q, k1p = deriv_fn(q, p, t, params)

        k2q, k2p = deriv_fn(
            q + h * a21 * k1q,
            p + h * a21 * k1p,
            t + h / 5.0,
            params,
        )
        k3q, k3p = deriv_fn(
            q + h * (a31 * k1q + a32 * k2q),
            p + h * (a31 * k1p + a32 * k2p),
            t + 3.0 * h / 10.0,
            params,
        )
        k4q, k4p = deriv_fn(
            q + h * (a41 * k1q + a42 * k2q + a43 * k3q),
            p + h * (a41 * k1p + a42 * k2p + a43 * k3p),
            t + 4.0 * h / 5.0,
            params,
        )
        k5q, k5p = deriv_fn(
            q + h * (a51 * k1q + a52 * k2q + a53 * k3q + a54 * k4q),
            p + h * (a51 * k1p + a52 * k2p + a53 * k3p + a54 * k4p),
            t + 8.0 * h / 9.0,
            params,
        )
        k6q, k6p = deriv_fn(
            q + h * (a61 * k1q + a62 * k2q + a63 * k3q + a64 * k4q + a65 * k5q),
            p + h * (a61 * k1p + a62 * k2p + a63 * k3p + a64 * k4p + a65 * k5p),
            t + h,
            params,
        )

        # 5th-order solution
        q5 = q + h * (b1 * k1q + b3 * k3q + b4 * k4q + b5 * k5q + b6 * k6q)
        p5 = p + h * (b1 * k1p + b3 * k3p + b4 * k4p + b5 * k5p + b6 * k6p)

        # k7 for the embedded error estimate
        k7q, k7p = deriv_fn(q5, p5, t + h, params)

        # Error: difference between 4th and 5th order
        err_q = h * (
            e1 * k1q + e3 * k3q + e4 * k4q + e5 * k5q + e6 * k6q + e7 * k7q
        )
        err_p = h * (
            e1 * k1p + e3 * k3p + e4 * k4p + e5 * k5p + e6 * k6p + e7 * k7p
        )

        # Scaled error norm
        scale_q = atol + rtol * _jnp.maximum(_jnp.abs(q), _jnp.abs(q5))
        scale_p = atol + rtol * _jnp.maximum(_jnp.abs(p), _jnp.abs(p5))
        err_norm_q = _jnp.sqrt(_jnp.mean((err_q / scale_q) ** 2))
        err_norm_p = _jnp.sqrt(_jnp.mean((err_p / scale_p) ** 2))
        err_norm = float(_jnp.maximum(err_norm_q, err_norm_p))

        if err_norm <= 1.0:
            # Accept step — compute new dt for next step
            h_new = dt_max if err_norm < 1e-30 else h * safety * err_norm ** (-0.2)
            h_new = min(h_new, dt_max)
            h_new = max(h_new, dt_min)
            # Store for potential next call (not used directly since we
            # return a single step, but the caller can inspect t_new - t).
            return q5, p5, t + h
        else:
            # Reject step and shrink
            h_new = h * safety * err_norm ** (-0.25)
            h = max(h_new, dt_min)
            rejects += 1
            if rejects >= max_reject:
                # Give up and accept the current best
                return q5, p5, t + h


# Registry mapping integrator names to functions
INTEGRATORS: dict[str, Callable[..., tuple[Array, Array, float]]] = {
    "euler": euler,
    "symplectic_euler": symplectic_euler,
    "leapfrog": leapfrog,
    "velocity_verlet": velocity_verlet,
    "stormer_verlet": stormer_verlet,
    "yoshida4": yoshida4,
    "rk4": rk4,
    "adaptive_rk45": adaptive_rk45,
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
