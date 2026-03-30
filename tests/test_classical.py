"""Tests for classical mechanics modules."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.classical.hamiltonian import HamiltonianSystem
from neurosim.classical.lagrangian import LagrangianSystem
from neurosim.classical.nbody import NBody
from neurosim.classical.rigid_body import RigidBody
from neurosim.config import Params
from neurosim.exceptions import ConfigurationError

jax.config.update("jax_enable_x64", True)


class TestLagrangianSystem:
    """Tests for the Lagrangian mechanics engine."""

    def test_simple_pendulum(self) -> None:
        """Verify energy conservation for a simple pendulum."""

        def lagrangian(q, qdot, params):
            m, g, length = params.m, params.g, params.l
            T = 0.5 * m * (length * qdot[0]) ** 2
            V = -m * g * length * jnp.cos(q[0])
            return T - V

        system = LagrangianSystem(lagrangian, n_dof=1)
        params = Params(m=1.0, g=9.81, l=1.0)

        traj = system.simulate(
            q0=[0.3],
            qdot0=[0.0],
            t_span=(0, 5),
            dt=0.01,
            params=params,
            integrator="rk4",
        )

        assert traj.n_steps > 0
        assert traj.n_dof == 1
        assert traj.energy is not None
        # Energy conservation: drift should be small for RK4
        assert traj.energy_drift() < 1e-6

    def test_harmonic_oscillator_frequency(self) -> None:
        """Check that a harmonic oscillator has the correct period."""

        def lagrangian(q, qdot, params):
            return 0.5 * params.m * qdot[0] ** 2 - 0.5 * params.k * q[0] ** 2

        system = LagrangianSystem(lagrangian, n_dof=1)
        params = Params(m=1.0, k=4.0)  # omega = 2, period = pi

        traj = system.simulate(
            q0=[1.0],
            qdot0=[0.0],
            t_span=(0, jnp.pi),
            dt=0.001,
            params=params,
            integrator="rk4",
        )

        # After one full period, should return to initial conditions
        assert traj.final_position[0] == pytest.approx(1.0, abs=1e-3)

    def test_invalid_n_dof(self) -> None:
        with pytest.raises(ConfigurationError):
            LagrangianSystem(lambda q, qdot, p: 0.0, n_dof=0)

    def test_shape_mismatch(self) -> None:
        def L(q, qdot, p):
            return 0.5 * qdot[0] ** 2

        system = LagrangianSystem(L, n_dof=1)
        with pytest.raises(ConfigurationError):
            system.simulate(
                q0=[1.0, 2.0],
                qdot0=[0.0],
                t_span=(0, 1),
                params=None,
            )

    def test_symplectic_integrator_rejected(self) -> None:
        def L(q, qdot, p):
            return 0.5 * qdot[0] ** 2 - 0.5 * q[0] ** 2

        system = LagrangianSystem(L, n_dof=1)

        with pytest.raises(ConfigurationError, match="not compatible"):
            system.simulate(
                q0=[1.0],
                qdot0=[0.0],
                t_span=(0, 1),
                params=None,
                integrator="leapfrog",
            )


class TestHamiltonianSystem:
    """Tests for the Hamiltonian mechanics engine."""

    def test_harmonic_oscillator(self) -> None:
        """Test H = p^2/(2m) + kx^2/2."""

        def hamiltonian(q, p, params):
            return p[0] ** 2 / (2 * params.m) + 0.5 * params.k * q[0] ** 2

        system = HamiltonianSystem(hamiltonian, n_dof=1)
        params = Params(m=1.0, k=1.0)

        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 6.28),
            dt=0.01,
            params=params,
            integrator="leapfrog",
        )

        # Leapfrog should conserve energy well
        assert traj.energy_drift() < 1e-6

    def test_symplectic_bounded_drift(self) -> None:
        """Symplectic integrator should have bounded energy drift."""

        def H(q, p, params):
            return p[0] ** 2 / 2 + q[0] ** 2 / 2

        params = Params()
        system = HamiltonianSystem(H, n_dof=1)

        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 100),
            dt=0.01,
            params=params,
            integrator="leapfrog",
        )

        # Leapfrog energy drift should be small and bounded
        assert traj.energy_drift() < 1e-4


class TestNBody:
    """Tests for the N-body simulator."""

    def test_two_body_circular(self) -> None:
        """Two equal masses in circular orbit should conserve energy."""
        system = NBody(
            masses=[1.0, 1.0],
            positions=[[-0.5, 0, 0], [0.5, 0, 0]],
            velocities=[[0, -0.5, 0], [0, 0.5, 0]],
            G=1.0,
            softening=1e-6,
        )

        traj = system.simulate(t_span=(0, 10), n_steps=10000, save_every=100)

        assert traj.n_bodies == 2
        assert traj.energy is not None
        # Energy should be approximately conserved
        e0 = float(traj.energy[0])
        ef = float(traj.energy[-1])
        assert abs((ef - e0) / abs(e0)) < 0.01

    def test_invalid_shapes(self) -> None:
        with pytest.raises(ConfigurationError):
            NBody(
                masses=[1.0],
                positions=[[0, 0]],  # Wrong shape
                velocities=[[0, 0, 0]],
            )


class TestRigidBody:
    """Tests for rigid body dynamics."""

    def test_torque_free(self) -> None:
        """Torque-free rotation should conserve energy and angular momentum."""
        body = RigidBody(inertia=[1.0, 2.0, 3.0])
        traj = body.simulate(
            omega0=[1.0, 0.1, 0.0],
            t_span=(0, 10),
            dt=0.01,
        )

        assert traj.energy is not None
        assert traj.energy_drift() < 1e-4

    def test_invalid_inertia(self) -> None:
        with pytest.raises(ConfigurationError):
            RigidBody(inertia=[1.0, -2.0, 3.0])
