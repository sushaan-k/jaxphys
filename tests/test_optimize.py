"""Tests for optimization and inverse problem solvers."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.optimize import optimize, projectile

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
