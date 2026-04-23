"""Tests for optimization and inverse problem solvers."""

import jax
import jax.numpy as jnp
import pytest

import neurosim as ns
from neurosim.exceptions import ConfigurationError
from neurosim.optimize import make_parameter_grid, optimize, parameter_sweep, projectile

jax.config.update("jax_enable_x64", True)


class TestOptimize:
    """Tests for gradient-based optimization."""

    def test_quadratic_minimum(self) -> None:
        """Optimizer should find the minimum of a simple quadratic."""

        def objective(x):
            return (x - 3.0) ** 2

        result = optimize(
            objective,
            initial_guess=0.0,
            learning_rate=0.1,
            max_iterations=1000,
        )

        assert result.converged
        assert float(result.x) == pytest.approx(3.0, abs=1e-4)
        assert result.fun < 1e-6

    def test_adam_optimizer(self) -> None:
        """Adam should also find the minimum."""

        def objective(x):
            return (x - 5.0) ** 2 + 1.0

        result = optimize(
            objective,
            initial_guess=0.0,
            learning_rate=0.1,
            max_iterations=1000,
            method="adam",
        )

        assert float(result.x) == pytest.approx(5.0, abs=0.1)

    def test_trajectory_tracking(self) -> None:
        """When track_trajectory=True, should record parameter history."""

        def objective(x):
            return x**2

        result = optimize(
            objective,
            initial_guess=10.0,
            learning_rate=0.1,
            max_iterations=100,
            track_trajectory=True,
        )

        assert result.trajectory is not None
        assert result.trajectory.shape[0] > 1

    def test_multivariate(self) -> None:
        """Should work with vector inputs."""

        def objective(x):
            return jnp.sum((x - jnp.array([1.0, 2.0])) ** 2)

        result = optimize(
            objective,
            initial_guess=jnp.array([0.0, 0.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )

        assert jnp.allclose(result.x, jnp.array([1.0, 2.0]), atol=0.1)


class TestProjectile:
    """Tests for the projectile utility."""

    def test_basic_projectile(self) -> None:
        result = projectile(v0=10.0, angle=45.0)
        # Range = v^2 * sin(2*theta) / g
        expected_range = 10.0**2 * jnp.sin(jnp.radians(90.0)) / 9.81
        assert float(result.range) == pytest.approx(float(expected_range), rel=0.01)

    def test_projectile_differentiable(self) -> None:
        """projectile should be differentiable w.r.t. v0."""

        def range_fn(v0):
            return projectile(v0=v0, angle=45.0).range

        grad = jax.grad(range_fn)(10.0)
        # d(range)/d(v0) = 2*v0*sin(2*theta)/g at 45 deg
        expected_grad = 2 * 10.0 / 9.81
        assert float(grad) == pytest.approx(expected_grad, rel=0.01)


class TestParameterSweep:
    """Tests for grid-based parameter sweeps."""

    def test_scalar_parameter_sweep_finds_best_loss(self) -> None:
        """parameter_sweep should rank scalar candidates by scalar output."""

        candidates = jnp.linspace(-2.0, 4.0, 13)
        result = parameter_sweep(lambda x: (x - 1.5) ** 2, candidates)

        assert result.n_evaluations == 13
        assert result.best_score == pytest.approx(0.0)
        assert float(result.best_parameters) == pytest.approx(1.5)
        assert float(result.best_value) == pytest.approx(0.0)
        assert result.summary()["best_index"] == result.best_index

    def test_vector_parameter_grid_with_objective(self) -> None:
        """Named grids should drive vector-valued simulations plus objective."""

        grid = make_parameter_grid(
            {
                "v0": jnp.array([20.0, 30.0, 40.0]),
                "angle": jnp.array([30.0, 45.0, 60.0]),
            }
        )

        target_range = 90.0
        result = parameter_sweep(
            lambda params: projectile(v0=params[0], angle=params[1]).range,
            grid.values,
            objective=lambda value: (value - target_range) ** 2,
        )

        best = grid.as_dict(result.best_index)
        assert result.n_evaluations == 9
        assert best["v0"] == pytest.approx(30.0)
        assert best["angle"] == pytest.approx(45.0)
        assert result.best_score < 5.0

    def test_top_k_respects_maximization(self) -> None:
        """top_k should sort descending when minimize=False."""

        result = parameter_sweep(
            lambda x: -((x - 2.0) ** 2),
            jnp.array([0.0, 1.0, 2.0, 3.0]),
            minimize=False,
            vectorized=False,
            batch_size=None,
        )
        top = result.top_k(3)

        assert top["indices"].shape == (3,)
        assert float(top["parameters"][0]) == pytest.approx(2.0)
        assert float(top["scores"][0]) == pytest.approx(0.0)
        assert float(top["scores"][1]) == pytest.approx(-1.0)

    def test_chunked_vectorized_sweep_matches_single_batch(self) -> None:
        """Chunking should preserve output and scores."""

        params = jnp.linspace(0.0, 1.0, 11)
        single = parameter_sweep(lambda x: x**2, params)
        chunked = parameter_sweep(lambda x: x**2, params, batch_size=4)

        assert jnp.allclose(chunked.values, single.values)
        assert jnp.allclose(chunked.scores, single.scores)

    def test_rejects_invalid_grid_and_scores(self) -> None:
        """Invalid sweep inputs should fail with domain errors."""

        with pytest.raises(ConfigurationError, match="At least one"):
            make_parameter_grid({})

        with pytest.raises(ConfigurationError, match="one scalar score"):
            parameter_sweep(lambda x: jnp.array([x, x**2]), jnp.array([1.0, 2.0]))

        result = parameter_sweep(lambda x: x, jnp.array([1.0]))
        with pytest.raises(ConfigurationError, match="positive"):
            result.top_k(0)

    def test_public_exports_available_from_package_root(self) -> None:
        """The sweep API should be reachable from the top-level package."""

        assert ns.parameter_sweep is parameter_sweep
        assert ns.make_parameter_grid is make_parameter_grid
        assert ns.ParameterGrid.__name__ == "ParameterGrid"
        assert ns.ParameterSweepResult.__name__ == "ParameterSweepResult"
