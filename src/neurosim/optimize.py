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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError

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


@dataclass(frozen=True)
class ParameterGrid:
    """Cartesian parameter grid for reproducible simulation sweeps.

    Attributes:
        names: Parameter names in column order.
        values: Grid values, shape ``(n_points, n_parameters)``.
        axes: Original one-dimensional axis values by parameter name.
    """

    names: tuple[str, ...]
    values: Array
    axes: dict[str, Array]

    @property
    def n_points(self) -> int:
        """Number of parameter combinations in the grid."""
        return int(self.values.shape[0])

    @property
    def n_parameters(self) -> int:
        """Number of named parameters per grid point."""
        return int(self.values.shape[1])

    def as_dict(self, index: int) -> dict[str, float]:
        """Return one grid row as a plain ``dict`` of parameter values."""
        row = self.values[index]
        return {name: float(row[i]) for i, name in enumerate(self.names)}


@dataclass(frozen=True)
class ParameterSweepResult:
    """Result of evaluating a simulation over a parameter grid.

    Attributes:
        parameters: Input parameters, shape ``(n_evaluations, ...)``.
        values: Raw simulation outputs, first axis aligned to parameters.
        scores: Scalar objective scores used for ranking.
        minimize: Whether lower scores are considered better.
        metadata: Optional caller-supplied context.
    """

    parameters: Array
    values: Array
    scores: Array
    minimize: bool = True
    metadata: dict[str, Any] | None = None

    @property
    def n_evaluations(self) -> int:
        """Number of parameter points evaluated."""
        return int(self.parameters.shape[0])

    @property
    def best_index(self) -> int:
        """Index of the best-scoring parameter point."""
        index = jnp.argmin(self.scores) if self.minimize else jnp.argmax(self.scores)
        return int(index)

    @property
    def best_parameters(self) -> Array:
        """Parameters for the best-scoring simulation."""
        return self.parameters[self.best_index]

    @property
    def best_value(self) -> Array:
        """Raw simulation output for the best-scoring parameter point."""
        return self.values[self.best_index]

    @property
    def best_score(self) -> float:
        """Best scalar objective score."""
        return float(self.scores[self.best_index])

    def top_k(self, k: int) -> dict[str, Array]:
        """Return the top ``k`` ranked parameters, values, and scores."""
        if k <= 0:
            raise ConfigurationError("k must be positive")

        n = min(k, self.n_evaluations)
        order = jnp.argsort(self.scores)
        if not self.minimize:
            order = order[::-1]
        indices = order[:n]
        return {
            "indices": indices,
            "parameters": self.parameters[indices],
            "values": self.values[indices],
            "scores": self.scores[indices],
        }

    def summary(self) -> dict[str, Any]:
        """Return a compact, serializable summary of the sweep."""
        return {
            "n_evaluations": self.n_evaluations,
            "parameter_shape": tuple(int(dim) for dim in self.parameters.shape),
            "value_shape": tuple(int(dim) for dim in self.values.shape),
            "minimize": self.minimize,
            "best_index": self.best_index,
            "best_score": self.best_score,
            "score_min": float(jnp.min(self.scores)),
            "score_max": float(jnp.max(self.scores)),
            "score_mean": float(jnp.mean(self.scores)),
        }


@dataclass(frozen=True)
class RefinedSweepCandidate:
    """One local optimization run started from a sweep candidate.

    Attributes:
        initial_parameters: Parameter values selected from the coarse sweep.
        initial_score: Coarse sweep score for the selected point.
        result: Local optimizer result started from ``initial_parameters``.
    """

    initial_parameters: Array
    initial_score: float
    result: OptimizeResult

    @property
    def refined_parameters(self) -> Array:
        """Parameters returned by the local optimizer."""
        return self.result.x

    @property
    def refined_score(self) -> float:
        """Objective value returned by the local optimizer."""
        return self.result.fun

    @property
    def improvement(self) -> float:
        """Positive value when refinement lowered the objective."""
        return self.initial_score - self.refined_score


@dataclass(frozen=True)
class SweepRefinementResult:
    """Local optimization results seeded by a coarse parameter sweep.

    Attributes:
        sweep: Original coarse sweep used to choose starting points.
        candidates: Refined candidates sorted by final objective value.
    """

    sweep: ParameterSweepResult
    candidates: tuple[RefinedSweepCandidate, ...]

    @property
    def n_refinements(self) -> int:
        """Number of local optimization runs performed."""
        return len(self.candidates)

    @property
    def best_candidate(self) -> RefinedSweepCandidate:
        """Best local refinement by final objective value."""
        if not self.candidates:
            raise ConfigurationError("Sweep refinement result has no candidates")
        return self.candidates[0]

    @property
    def best_parameters(self) -> Array:
        """Best refined parameters."""
        return self.best_candidate.refined_parameters

    @property
    def best_score(self) -> float:
        """Best refined objective value."""
        return self.best_candidate.refined_score

    def summary(self) -> dict[str, Any]:
        """Return a compact, serializable summary of the refinement."""
        best = self.best_candidate
        return {
            "n_refinements": self.n_refinements,
            "sweep_best_score": self.sweep.best_score,
            "best_initial_score": best.initial_score,
            "best_refined_score": best.refined_score,
            "best_improvement": best.improvement,
            "best_converged": best.result.converged,
            "best_iterations": best.result.n_iterations,
        }


def make_parameter_grid(
    axes: Mapping[str, Sequence[float] | Array],
) -> ParameterGrid:
    """Build a Cartesian grid from named one-dimensional parameter axes.

    Example:
        >>> grid = make_parameter_grid({
        ...     "v0": jnp.linspace(20.0, 40.0, 5),
        ...     "angle": [30.0, 45.0, 60.0],
        ... })
        >>> grid.values.shape
        (15, 2)

    Args:
        axes: Mapping of parameter name to one-dimensional values.

    Returns:
        ``ParameterGrid`` with rows ordered by the mapping's insertion order.

    Raises:
        ConfigurationError: If no axes are provided, an axis is empty, or an
            axis is not one-dimensional.
    """
    if not axes:
        raise ConfigurationError("At least one parameter axis is required")

    names = tuple(axes.keys())
    axis_arrays: dict[str, Array] = {}
    for name in names:
        values = jnp.asarray(axes[name], dtype=jnp.float64)
        if values.ndim != 1:
            raise ConfigurationError(f"Parameter axis '{name}' must be one-dimensional")
        if int(values.shape[0]) == 0:
            raise ConfigurationError(f"Parameter axis '{name}' must not be empty")
        axis_arrays[name] = values

    meshes = jnp.meshgrid(*(axis_arrays[name] for name in names), indexing="ij")
    values = jnp.stack([mesh.reshape(-1) for mesh in meshes], axis=-1)
    return ParameterGrid(names=names, values=values, axes=axis_arrays)


def parameter_sweep(
    simulation_fn: Callable[[Array], Array],
    parameters: Sequence[float] | Sequence[Sequence[float]] | Array,
    *,
    objective: Callable[[Array], Array] | None = None,
    minimize: bool = True,
    vectorized: bool = True,
    batch_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParameterSweepResult:
    """Evaluate a simulation over candidate parameters and rank results.

    ``parameter_sweep`` is useful before gradient optimization, when mapping a
    stable basin or comparing solver settings matters more than following one
    local gradient. Inputs can be a one-dimensional list of scalar parameters or
    a two-dimensional array where each row is a parameter vector.

    Args:
        simulation_fn: Function mapping one parameter point to a JAX array.
        parameters: Parameter candidates. The leading axis enumerates trials.
        objective: Optional function mapping one simulation output to a scalar
            score. If omitted, each simulation output must already be scalar.
        minimize: Whether lower scores are better. Set ``False`` to maximize.
        vectorized: Use ``jax.vmap`` for evaluation. Set ``False`` for
            debugging functions with Python-side effects.
        batch_size: Optional chunk size for large vectorized sweeps.
        metadata: Optional caller-supplied context copied into the result.

    Returns:
        ``ParameterSweepResult`` with raw outputs, scalar scores, and helpers
        for best/top-k inspection.

    Raises:
        ConfigurationError: If parameters are empty, batch size is invalid, or
            outputs/objective scores are not scalar per evaluation.
    """
    parameter_array = jnp.asarray(parameters, dtype=jnp.float64)
    if parameter_array.ndim == 0:
        parameter_array = parameter_array.reshape(1)
    if int(parameter_array.shape[0]) == 0:
        raise ConfigurationError("Parameter sweep requires at least one candidate")
    if batch_size is not None and batch_size <= 0:
        raise ConfigurationError("batch_size must be positive")

    values = _evaluate_sweep(
        simulation_fn=simulation_fn,
        parameters=parameter_array,
        vectorized=vectorized,
        batch_size=batch_size,
    )
    scores = _score_sweep_values(values, objective=objective, vectorized=vectorized)
    return ParameterSweepResult(
        parameters=parameter_array,
        values=values,
        scores=scores,
        minimize=minimize,
        metadata=metadata,
    )


def refine_parameter_sweep(
    objective: Callable[[Array], Array],
    sweep: ParameterSweepResult,
    *,
    top_k: int = 1,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    method: str = "gradient_descent",
    track_trajectory: bool = False,
) -> SweepRefinementResult:
    """Refine the best coarse sweep candidates with local optimization.

    This helper implements the common global-to-local workflow: first map a
    broad parameter space with ``parameter_sweep``, then run gradient descent
    from the best basin or basins. The ``objective`` should be the scalar
    minimization objective over one parameter point.

    Args:
        objective: Differentiable scalar objective accepting one parameter
            point, shaped like a single row from ``sweep.parameters``.
        sweep: Coarse parameter sweep result used to select starting points.
        top_k: Number of ranked sweep candidates to refine.
        learning_rate: Step size passed to ``optimize``.
        max_iterations: Maximum local optimization iterations.
        tolerance: Gradient norm convergence threshold.
        method: Optimizer method passed to ``optimize``.
        track_trajectory: Whether each local optimizer records its trajectory.

    Returns:
        ``SweepRefinementResult`` with refined candidates sorted by final
        objective value.

    Raises:
        ConfigurationError: If ``top_k`` is not positive, or if ``sweep`` was
            ranked as a maximization search.
    """
    if top_k <= 0:
        raise ConfigurationError("top_k must be positive")
    if not sweep.minimize:
        raise ConfigurationError(
            "refine_parameter_sweep requires a minimization sweep; "
            "negate reward objectives before sweeping and refining"
        )

    ranked = sweep.top_k(top_k)
    parameters = ranked["parameters"]
    scores = ranked["scores"]

    candidates: list[RefinedSweepCandidate] = []
    for index in range(int(parameters.shape[0])):
        seed = parameters[index]
        result = optimize(
            objective,
            initial_guess=seed,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            method=method,
            track_trajectory=track_trajectory,
        )
        candidates.append(
            RefinedSweepCandidate(
                initial_parameters=seed,
                initial_score=float(scores[index]),
                result=result,
            )
        )

    candidates.sort(key=lambda candidate: candidate.refined_score)
    return SweepRefinementResult(sweep=sweep, candidates=tuple(candidates))


def _evaluate_sweep(
    simulation_fn: Callable[[Array], Array],
    parameters: Array,
    *,
    vectorized: bool,
    batch_size: int | None,
) -> Array:
    """Evaluate a sweep in one vectorized call or in bounded chunks."""
    if not vectorized:
        return jnp.stack([jnp.asarray(simulation_fn(point)) for point in parameters])

    batched_fn = jax.vmap(simulation_fn)
    if batch_size is None or batch_size >= int(parameters.shape[0]):
        return batched_fn(parameters)

    chunks = []
    for start in range(0, int(parameters.shape[0]), batch_size):
        stop = min(start + batch_size, int(parameters.shape[0]))
        chunks.append(batched_fn(parameters[start:stop]))
    return jnp.concatenate(chunks, axis=0)


def _score_sweep_values(
    values: Array,
    *,
    objective: Callable[[Array], Array] | None,
    vectorized: bool,
) -> Array:
    """Convert raw simulation outputs into one scalar score per trial."""
    if objective is None:
        scores = values
    elif vectorized:
        scores = jax.vmap(objective)(values)
    else:
        scores = jnp.stack([jnp.asarray(objective(value)) for value in values])

    if scores.ndim == 0:
        scores = scores.reshape(1)
    if scores.ndim != 1:
        raise ConfigurationError(
            "Parameter sweep objective must return one scalar score per candidate"
        )
    if int(scores.shape[0]) != int(values.shape[0]):
        raise ConfigurationError(
            "Parameter sweep scores must align with the number of outputs"
        )
    return scores


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
