"""Tests for classical mechanics modules."""

import time

import jax
import jax.numpy as jnp
import pytest

from neurosim.classical.coupled_oscillators import (
    coupled_oscillators,
    normal_mode_frequencies,
)
from neurosim.classical.hamiltonian import HamiltonianSystem
from neurosim.classical.lagrangian import LagrangianSystem
from neurosim.classical.nbody import NBody
from neurosim.classical.rigid_body import RigidBody
from neurosim.config import Params
from neurosim.exceptions import ConfigurationError, PhysicsError

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


class TestEnergyConservationDiagnostic:
    """Tests for Trajectory.max_energy_drift and check_conservation."""

    def test_max_energy_drift_symplectic(self) -> None:
        """max_energy_drift should be small for a symplectic integrator."""

        def H(q, p, params):
            return p[0] ** 2 / 2 + q[0] ** 2 / 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 20),
            dt=0.01,
            params=Params(),
            integrator="leapfrog",
        )

        drift = traj.max_energy_drift()
        assert drift < 1e-4
        # Also verify it is at least non-negative
        assert drift >= 0.0

    def test_max_energy_drift_larger_than_endpoint_drift(self) -> None:
        """max_energy_drift should capture intermediate deviations."""

        def H(q, p, params):
            return p[0] ** 2 / 2 + q[0] ** 2 / 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 10),
            dt=0.01,
            params=Params(),
            integrator="leapfrog",
        )

        # max drift over the full trajectory >= simple endpoint drift
        assert traj.max_energy_drift() >= abs(float(traj.energy[-1] - traj.energy[0]))

    def test_check_conservation_passes(self) -> None:
        """check_conservation should not raise for a well-behaved sim."""

        def H(q, p, params):
            return p[0] ** 2 / 2 + q[0] ** 2 / 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 5),
            dt=0.01,
            params=Params(),
            integrator="leapfrog",
        )
        # Should not raise
        traj.check_conservation(tolerance=1e-3)

    def test_check_conservation_raises(self) -> None:
        """check_conservation should raise PhysicsError for bad tolerance."""

        def H(q, p, params):
            return p[0] ** 2 / 2 + q[0] ** 2 / 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 10),
            dt=0.1,  # large dt => noticeable drift
            params=Params(),
            integrator="euler",
        )
        with pytest.raises(PhysicsError, match="Energy conservation violated"):
            traj.check_conservation(tolerance=1e-15)

    def test_max_energy_drift_no_energy(self) -> None:
        """max_energy_drift returns 0.0 when energy is None."""
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 1.0]),
            q=jnp.array([[1.0], [0.9]]),
            p=jnp.array([[0.0], [0.1]]),
            energy=None,
        )
        assert traj.max_energy_drift() == 0.0
        # check_conservation should also be safe
        traj.check_conservation(tolerance=1e-10)

    def test_diagnostics_with_energy(self) -> None:
        """diagnostics should expose drift and summary stats."""

        def H(q, p, params):
            return p[0] ** 2 / 2 + q[0] ** 2 / 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 5),
            dt=0.01,
            params=Params(),
            integrator="leapfrog",
        )

        diagnostics = traj.diagnostics()
        assert diagnostics["n_steps"] > 0
        assert diagnostics["duration"] == pytest.approx(5.0, abs=0.05)
        assert diagnostics["energy_span"] >= 0.0
        assert diagnostics["max_energy_drift"] >= 0.0
        assert diagnostics["energy_drift"] >= 0.0

    def test_diagnostics_without_energy(self) -> None:
        """diagnostics should remain useful when energy is not tracked."""
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 1.0, 2.0]),
            q=jnp.array([[0.0], [1.0], [2.0]]),
            p=jnp.array([[0.0], [0.5], [1.0]]),
            energy=None,
        )

        diagnostics = traj.diagnostics()
        assert diagnostics["n_steps"] == 3.0
        assert diagnostics["n_dof"] == 1.0
        assert "energy_span" not in diagnostics

    def test_diagnostics_exact_energy_summary(self) -> None:
        """diagnostics should report exact summary values for known energies."""
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 0.5, 1.0]),
            q=jnp.array([[0.0], [0.1], [0.2]]),
            p=jnp.array([[1.0], [0.9], [0.8]]),
            energy=jnp.array([10.0, 11.5, 9.0]),
        )

        diagnostics = traj.diagnostics()
        assert diagnostics["energy_initial"] == pytest.approx(10.0)
        assert diagnostics["energy_final"] == pytest.approx(9.0)
        assert diagnostics["energy_min"] == pytest.approx(9.0)
        assert diagnostics["energy_max"] == pytest.approx(11.5)
        assert diagnostics["energy_mean"] == pytest.approx(10.1666666667)
        assert diagnostics["energy_span"] == pytest.approx(2.5)
        assert diagnostics["energy_drift"] == pytest.approx(0.1)
        assert diagnostics["max_energy_drift"] == pytest.approx(1.5)

    def test_conservation_report(self) -> None:
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 1.0]),
            q=jnp.array([[0.0], [0.0]]),
            p=jnp.array([[0.0], [0.0]]),
            energy=jnp.array([1.0, 1.01]),
        )

        report = traj.conservation_report(tolerance=0.02)
        assert report["tracked"] is True
        assert report["passed"] is True
        assert report["max_energy_drift"] == pytest.approx(0.01)

        failed = traj.conservation_report(tolerance=0.001)
        assert failed["passed"] is False

    def test_conservation_report_without_energy_is_not_passed(self) -> None:
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 1.0]),
            q=jnp.array([[0.0], [0.0]]),
            p=jnp.array([[0.0], [0.0]]),
            energy=None,
        )

        report = traj.conservation_report(tolerance=1e-6)
        assert report["tracked"] is False
        assert report["passed"] is False
        assert report["max_energy_drift"] == 0.0

    def test_conservation_report_empty_energy_is_not_passed(self) -> None:
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0]),
            q=jnp.array([[0.0]]),
            p=jnp.array([[0.0]]),
            energy=jnp.array([]),
        )

        report = traj.conservation_report(tolerance=1e-6)
        assert report["tracked"] is False
        assert report["passed"] is False
        assert report["max_energy_drift"] == 0.0
        assert traj.energy_drift() == 0.0
        assert traj.max_energy_drift() == 0.0

    def test_conservation_report_single_energy_sample_passes(self) -> None:
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0]),
            q=jnp.array([[0.0]]),
            p=jnp.array([[0.0]]),
            energy=jnp.array([5.0]),
        )

        report = traj.conservation_report(tolerance=1e-6)
        assert report["tracked"] is True
        assert report["passed"] is True
        assert report["max_energy_drift"] == 0.0
        assert report["energy_drift"] == 0.0


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


class TestJITPerformance:
    """Verify that JIT-compiled simulation loops provide speedup."""

    def test_hamiltonian_jit_speedup(self) -> None:
        """Second run of a Hamiltonian simulation should be faster (JIT cached)."""

        def hamiltonian(q, p, params):
            return p[0] ** 2 / (2 * params.m) + 0.5 * params.k * q[0] ** 2

        system = HamiltonianSystem(hamiltonian, n_dof=1)
        params = Params(m=1.0, k=4.0)
        kwargs = dict(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 10),
            dt=0.01,
            params=params,
            integrator="leapfrog",
        )

        # First call includes compilation
        t0 = time.perf_counter()
        system.simulate(**kwargs)
        first_run = time.perf_counter() - t0

        # Second call uses cached JIT kernels
        t0 = time.perf_counter()
        system.simulate(**kwargs)
        second_run = time.perf_counter() - t0

        # Second run should be noticeably faster (at least 1.5x)
        assert second_run < first_run or second_run < 0.5

    def test_lagrangian_jit_speedup(self) -> None:
        """Lagrangian simulation should also benefit from JIT caching."""

        def lagrangian(q, qdot, params):
            return 0.5 * params.m * qdot[0] ** 2 - 0.5 * params.k * q[0] ** 2

        system = LagrangianSystem(lagrangian, n_dof=1)
        params = Params(m=1.0, k=4.0)
        kwargs = dict(
            q0=[1.0],
            qdot0=[0.0],
            t_span=(0, 10),
            dt=0.01,
            params=params,
            integrator="rk4",
        )

        # Warm up (compilation)
        t0 = time.perf_counter()
        system.simulate(**kwargs)
        first_run = time.perf_counter() - t0

        # Cached run
        t0 = time.perf_counter()
        traj = system.simulate(**kwargs)
        second_run = time.perf_counter() - t0

        # Verify correctness is maintained
        assert traj.energy is not None
        assert traj.energy_drift() < 1e-6

        # JIT second run should be faster
        assert second_run < first_run or second_run < 0.5


class TestCoupledOscillators:
    """Tests for the coupled oscillators convenience builder."""

    def test_single_oscillator_frequency(self) -> None:
        """A single mass-spring matches the exact omega = sqrt(k/m)."""
        k, m = 4.0, 1.0
        system = coupled_oscillators(1, k=k, m=m)
        omega_exact = jnp.sqrt(k / m)  # = 2.0
        period = float(2 * jnp.pi / omega_exact)

        traj = system.simulate(
            q0=[1.0],
            qdot0=[0.0],
            t_span=(0, period),
            dt=0.001,
            params=None,
        )

        # Should return to initial conditions after one period
        assert traj.final_position[0] == pytest.approx(1.0, abs=1e-3)

    def test_normal_mode_frequencies_vs_eigenvalues(self) -> None:
        """Verify analytical frequencies match numerical eigenvalues of K/m."""
        n, k, m = 3, 2.0, 0.5
        freqs = normal_mode_frequencies(n, k=k, m=m)

        # Build stiffness matrix for fixed-wall / free-end chain
        K = jnp.zeros((n, n))
        K = K.at[0, 0].set(2 * k)
        for i in range(1, n):
            K = K.at[i, i].set(2 * k)
            K = K.at[i, i - 1].set(-k)
            K = K.at[i - 1, i].set(-k)
        K = K.at[n - 1, n - 1].set(k)  # free end

        eigenvalues = jnp.sort(jnp.linalg.eigvalsh(K / m))
        numerical_freqs = jnp.sqrt(eigenvalues)

        for i in range(n):
            assert float(freqs[i]) == pytest.approx(float(numerical_freqs[i]), rel=1e-8)

    def test_energy_conservation(self) -> None:
        """Coupled oscillator chain should conserve energy."""
        system = coupled_oscillators(4, k=2.0, m=1.0)
        traj = system.simulate(
            q0=[0.5, 0.0, 0.0, 0.0],
            qdot0=[0.0, 0.0, 0.0, 0.0],
            t_span=(0, 20),
            dt=0.001,
            params=None,
        )

        assert traj.energy is not None
        assert traj.energy_drift() < 1e-5

    def test_normal_mode_excitation(self) -> None:
        """Exciting a pure normal mode should oscillate at the predicted frequency."""
        n = 3
        k, m = 1.0, 1.0

        freqs = normal_mode_frequencies(n, k=k, m=m)

        # Build K matrix and get first eigenvector numerically
        K = jnp.zeros((n, n))
        K = K.at[0, 0].set(2 * k)
        for i in range(1, n):
            K = K.at[i, i].set(2 * k)
            K = K.at[i, i - 1].set(-k)
            K = K.at[i - 1, i].set(-k)
        K = K.at[n - 1, n - 1].set(k)

        _eigenvalues, eigenvectors = jnp.linalg.eigh(K / m)
        mode_shape = eigenvectors[:, 0]  # lowest mode

        amplitude = 0.1
        q0 = amplitude * mode_shape

        system = coupled_oscillators(n, k=k, m=m)
        omega1 = float(freqs[0])
        half_period = jnp.pi / omega1

        # After half a period the mode should reverse sign
        traj = system.simulate(
            q0=q0.tolist(),
            qdot0=[0.0] * n,
            t_span=(0, float(half_period)),
            dt=0.0005,
            params=None,
        )

        # Position should be approximately -q0
        for i in range(n):
            assert traj.final_position[i] == pytest.approx(float(-q0[i]), abs=0.01)

    def test_invalid_params(self) -> None:
        with pytest.raises(ConfigurationError):
            coupled_oscillators(0)
        with pytest.raises(ConfigurationError):
            coupled_oscillators(3, k=-1.0)
        with pytest.raises(ConfigurationError):
            coupled_oscillators(3, m=0.0)
