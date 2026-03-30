"""Tests for numerical integrators.

Verifies correctness of each integrator against known analytical
solutions, and checks symplectic integrators for energy conservation.
"""

import jax
import jax.numpy as jnp
import pytest

from neurosim.classical.integrators import (
    euler,
    get_integrator,
    leapfrog,
    rk4,
    velocity_verlet,
    yoshida4,
)

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


def harmonic_deriv(q, p, t, params):
    """Simple harmonic oscillator: dq/dt = p, dp/dt = -q."""
    return p, -q


def harmonic_accel(q, v, t, params):
    """Acceleration for harmonic oscillator: a = -q."""
    return -q


class TestEuler:
    """Tests for forward Euler integrator."""

    def test_single_step(self) -> None:
        q = jnp.array([1.0])
        p = jnp.array([0.0])
        q_new, _p_new, t_new = euler(harmonic_deriv, q, p, 0.0, 0.01, None)
        assert q_new.shape == (1,)
        assert t_new == pytest.approx(0.01)

    def test_first_order(self) -> None:
        """Euler should be first-order accurate."""
        q0 = jnp.array([1.0])
        p0 = jnp.array([0.0])

        # Run with dt and dt/2
        q1, _p1, _ = euler(harmonic_deriv, q0, p0, 0.0, 0.1, None)

        q_half, p_half, _ = euler(harmonic_deriv, q0, p0, 0.0, 0.05, None)
        q2, _p2, _ = euler(harmonic_deriv, q_half, p_half, 0.05, 0.05, None)

        # Errors should differ by factor of ~2 (first order)
        exact_q = jnp.cos(0.1)
        err1 = jnp.abs(q1[0] - exact_q)
        err2 = jnp.abs(q2[0] - exact_q)
        assert err2 < err1


class TestLeapfrog:
    """Tests for leapfrog integrator."""

    def test_harmonic_oscillator(self) -> None:
        """Leapfrog should approximately conserve energy for SHO."""
        q = jnp.array([1.0])
        p = jnp.array([0.0])
        dt = 0.01

        # Run 1000 steps (one full period at omega=1 is 2*pi ~ 6.28)
        for _ in range(1000):
            q, p, _ = leapfrog(harmonic_deriv, q, p, 0.0, dt, None)

        # Energy should be close to initial (0.5)
        energy = 0.5 * p[0] ** 2 + 0.5 * q[0] ** 2
        assert energy == pytest.approx(0.5, abs=1e-4)


class TestYoshida4:
    """Tests for fourth-order Yoshida integrator."""

    def test_higher_order_accuracy(self) -> None:
        """Yoshida4 should be more accurate than leapfrog for same dt."""
        q0 = jnp.array([1.0])
        p0 = jnp.array([0.0])
        dt = 0.1

        # One step with leapfrog
        q_lf, p_lf, _ = leapfrog(harmonic_deriv, q0, p0, 0.0, dt, None)

        # One step with yoshida4
        q_y4, p_y4, _ = yoshida4(harmonic_deriv, q0, p0, 0.0, dt, None)

        exact_q = jnp.cos(dt)
        exact_p = -jnp.sin(dt)

        err_lf = jnp.abs(q_lf[0] - exact_q) + jnp.abs(p_lf[0] - exact_p)
        err_y4 = jnp.abs(q_y4[0] - exact_q) + jnp.abs(p_y4[0] - exact_p)

        assert err_y4 < err_lf


class TestRK4:
    """Tests for fourth-order Runge-Kutta integrator."""

    def test_harmonic_oscillator(self) -> None:
        """RK4 should track the exact solution closely."""
        q = jnp.array([1.0])
        p = jnp.array([0.0])
        dt = 0.01
        t = 0.0

        for _ in range(628):  # ~ one period
            q, p, t = rk4(harmonic_deriv, q, p, t, dt, None)

        # Should return near initial condition after one period
        assert q[0] == pytest.approx(1.0, abs=1e-2)
        assert p[0] == pytest.approx(0.0, abs=1e-2)


class TestVelocityVerlet:
    """Tests for velocity Verlet integrator."""

    def test_single_step(self) -> None:
        q = jnp.array([1.0])
        v = jnp.array([0.0])
        q_new, _v_new, t_new = velocity_verlet(harmonic_accel, q, v, 0.0, 0.01, None)
        assert q_new.shape == (1,)
        assert t_new == pytest.approx(0.01)


class TestIntegratorRegistry:
    """Tests for the integrator lookup."""

    def test_valid_lookup(self) -> None:
        fn = get_integrator("rk4")
        assert fn is rk4

    def test_invalid_lookup(self) -> None:
        with pytest.raises(ValueError, match="Unknown integrator"):
            get_integrator("nonexistent")

    def test_all_registered(self) -> None:
        for name in [
            "euler",
            "symplectic_euler",
            "leapfrog",
            "velocity_verlet",
            "stormer_verlet",
            "yoshida4",
            "rk4",
        ]:
            fn = get_integrator(name)
            assert callable(fn)
