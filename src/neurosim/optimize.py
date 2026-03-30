"""Optimization and inverse problem solvers.

Provides gradient-based optimization utilities that leverage JAX's
automatic differentiation for solving inverse physics problems:

- Find initial conditions that produce desired outcomes
- Optimize system parameters to match observations
- Sensitivity analysis via Jacobians

All functions are differentiable and JIT-compatible.

References:
    - Nocedal & Wright. "Numerical Optimization" (2006)
    - Baydin et al. "Automatic differentiation in machine learning" (2018)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizeResult:
    """Result of an optimization run.

    Attributes:
        x: Optimal parameter value.
        fun: Objective function value at optimum.
        n_iterations: Number of iterations performed.
        converged: Whether the optimizer converged.
        trajectory: History of parameter values (if tracked).
    """

    x: Array
    fun: float
    n_iterations: int
    converged: bool
    trajectory: Array | None = None


def optimize(
    objective: Any,
    initial_guess: Array | float,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    method: str = "gradient_descent",
    track_trajectory: bool = False,
) -> OptimizeResult:
    """Minimize an objective function using gradient-based optimization.

    Leverages JAX's automatic differentiation to compute gradients
    of the objective function, then applies gradient descent or
    Adam to find the minimum.

    Example:
        >>> def miss_distance(v0):
        ...     # Simulate projectile and return distance to target
        ...     return (v0 * jnp.sin(jnp.pi/4) * 2 * v0 * jnp.sin(jnp.pi/4) / 9.81 - 100)**2
        >>> result = optimize(miss_distance, initial_guess=10.0)

    Args:
        objective: Scalar-valued function to minimize.
            Must be differentiable via JAX.
        initial_guess: Starting parameter value.
        learning_rate: Step size for optimization.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence threshold on gradient norm.
        method: "gradient_descent" or "adam".
        track_trajectory: Whether to save parameter history.

    Returns:
        OptimizeResult with optimal parameters.
    """
    x = jnp.asarray(initial_guess, dtype=jnp.float64)
    grad_fn = jax.grad(objective)

    logger.info(
        "Starting optimization: method=%s, lr=%.2e, max_iter=%d",
        method,
        learning_rate,
        max_iterations,
    )

    if method == "adam":
        return _adam_optimize(
            objective,
            grad_fn,
            x,
            learning_rate,
            max_iterations,
            tolerance,
            track_trajectory,
        )

    # Gradient descent
    trajectory_list = [x] if track_trajectory else []

    for i in range(max_iterations):
        g = grad_fn(x)
        grad_norm = float(jnp.sqrt(jnp.sum(g**2)))

        if grad_norm < tolerance:
            logger.info("Converged at iteration %d, grad_norm=%.2e", i, grad_norm)
            traj = jnp.stack(trajectory_list) if track_trajectory else None
            return OptimizeResult(
                x=x,
                fun=float(objective(x)),
                n_iterations=i,
                converged=True,
                trajectory=traj,
            )

        x = x - learning_rate * g
        if track_trajectory:
            trajectory_list.append(x)

    logger.warning("Optimization did not converge after %d iterations", max_iterations)
    traj = jnp.stack(trajectory_list) if track_trajectory else None
    return OptimizeResult(
        x=x,
        fun=float(objective(x)),
        n_iterations=max_iterations,
        converged=False,
        trajectory=traj,
    )


def _adam_optimize(
    objective: Any,
    grad_fn: Any,
    x0: Array,
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
    track_trajectory: bool,
) -> OptimizeResult:
    """Adam optimizer implementation.

    Adam combines momentum and adaptive learning rates:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        x_t = x_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

    Args:
        objective: Objective function.
        grad_fn: Gradient function.
        x0: Initial parameters.
        learning_rate: Learning rate.
        max_iterations: Max iterations.
        tolerance: Convergence threshold.
        track_trajectory: Whether to track history.

    Returns:
        OptimizeResult.
    """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    x = x0
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)

    trajectory_list = [x] if track_trajectory else []

    for i in range(1, max_iterations + 1):
        g = grad_fn(x)
        grad_norm = float(jnp.sqrt(jnp.sum(g**2)))

        if grad_norm < tolerance:
            traj = jnp.stack(trajectory_list) if track_trajectory else None
            return OptimizeResult(
                x=x,
                fun=float(objective(x)),
                n_iterations=i,
                converged=True,
                trajectory=traj,
            )

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)

        x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        if track_trajectory:
            trajectory_list.append(x)

    traj = jnp.stack(trajectory_list) if track_trajectory else None
    return OptimizeResult(
        x=x,
        fun=float(objective(x)),
        n_iterations=max_iterations,
        converged=False,
        trajectory=traj,
    )


def sensitivity(
    simulation_fn: Any,
    params: Array,
) -> Array:
    """Compute parameter sensitivity (Jacobian) of a simulation.

    Uses JAX's automatic Jacobian computation to determine how
    each output depends on each input parameter.

    Args:
        simulation_fn: Function params -> outputs.
        params: Parameter values at which to evaluate.

    Returns:
        Jacobian matrix d(outputs)/d(params).
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    return cast(Array, jax.jacobian(simulation_fn)(params))


def projectile(
    v0: float | Array,
    angle: float = 45.0,
    g: float = 9.81,
    dt: float = 0.01,
) -> Any:
    """Simple projectile simulation for optimization demos.

    Simulates a projectile under constant gravity and returns
    a result object with the final position.

    Args:
        v0: Initial speed.
        angle: Launch angle in degrees.
        g: Gravitational acceleration.
        dt: Time step.

    Returns:
        Object with final_position attribute.
    """
    angle_rad = jnp.radians(angle)
    vx = v0 * jnp.cos(angle_rad)
    vy = v0 * jnp.sin(angle_rad)

    # Time of flight: t = 2 * vy / g
    t_flight = 2.0 * vy / g

    # Range: R = vx * t_flight
    range_val = vx * t_flight

    @dataclass
    class _ProjectileResult:
        final_position: Any
        range: Any
        time_of_flight: Any

    return _ProjectileResult(
        final_position=range_val,
        range=range_val,
        time_of_flight=t_flight,
    )
