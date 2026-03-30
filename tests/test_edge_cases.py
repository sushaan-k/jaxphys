"""Comprehensive edge case, accuracy, and advanced tests for neurosim.

Covers:
- Energy conservation over long simulations (symplectic integrators)
- Numerical accuracy against analytical solutions
- Edge cases: zero coupling, single-particle N-body, tiny/huge timesteps
- Visualization module smoke tests
- Config validation (invalid params, negative masses, etc.)
- Rigid body quaternion normalization
- Diffraction pattern symmetry
- Density matrix trace preservation
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Energy conservation tests: symplectic integrators on Hamiltonian systems
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """Verify symplectic integrators conserve energy over long simulations."""

    @staticmethod
    def _sho_hamiltonian(q, p, params):
        """Simple harmonic oscillator H = p^2/2 + q^2/2."""
        return 0.5 * p[0] ** 2 + 0.5 * q[0] ** 2

    def test_leapfrog_long_run_energy_bounded(self) -> None:
        """Leapfrog energy error stays bounded over 10000 steps."""
        from neurosim.classical.hamiltonian import HamiltonianSystem

        system = HamiltonianSystem(self._sho_hamiltonian, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 100),
            dt=0.01,
            params=None,
            integrator="leapfrog",
        )
        # Energy drift should stay below 1e-4 over 10000 steps
        max_drift = float(
            jnp.max(jnp.abs(traj.energy - traj.energy[0])) / jnp.abs(traj.energy[0])
        )
        assert max_drift < 1e-4

    def test_yoshida4_long_run_energy_bounded(self) -> None:
        """Yoshida4 energy error bounded over 10000 steps."""
        from neurosim.classical.hamiltonian import HamiltonianSystem

        system = HamiltonianSystem(self._sho_hamiltonian, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 100),
            dt=0.01,
            params=None,
            integrator="yoshida4",
        )
        max_drift = float(
            jnp.max(jnp.abs(traj.energy - traj.energy[0])) / jnp.abs(traj.energy[0])
        )
        # Yoshida4 should be even better than leapfrog
        assert max_drift < 1e-6

    def test_symplectic_euler_energy_bounded(self) -> None:
        """Symplectic Euler energy error bounded (less tight than leapfrog)."""
        from neurosim.classical.hamiltonian import HamiltonianSystem

        system = HamiltonianSystem(self._sho_hamiltonian, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 50),
            dt=0.01,
            params=None,
            integrator="symplectic_euler",
        )
        max_drift = float(
            jnp.max(jnp.abs(traj.energy - traj.energy[0])) / jnp.abs(traj.energy[0])
        )
        assert max_drift < 0.01  # 1st order so looser

    def test_euler_energy_drifts(self) -> None:
        """Non-symplectic Euler should show unbounded energy drift."""
        from neurosim.classical.hamiltonian import HamiltonianSystem

        system = HamiltonianSystem(self._sho_hamiltonian, n_dof=1)
        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, 50),
            dt=0.01,
            params=None,
            integrator="euler",
        )
        # Euler grows energy; drift should be large
        drift = traj.energy_drift()
        assert drift > 0.01  # noticeable drift

    def test_nbody_two_body_energy_conservation(self) -> None:
        """Two-body problem energy should be well-conserved."""
        from neurosim.classical.nbody import NBody

        system = NBody(
            masses=[1.0, 1.0],
            positions=[[-0.5, 0, 0], [0.5, 0, 0]],
            velocities=[[0, -0.5, 0], [0, 0.5, 0]],
            G=1.0,
            softening=1e-6,
        )
        traj = system.simulate(t_span=(0, 50), n_steps=50000, save_every=500)
        e0 = float(traj.energy[0])
        ef = float(traj.energy[-1])
        assert abs((ef - e0) / abs(e0)) < 0.01

    def test_rigid_body_torque_free_energy_conservation(self) -> None:
        """Torque-free rigid body should conserve rotational energy."""
        from neurosim.classical.rigid_body import RigidBody

        body = RigidBody(inertia=[1.0, 2.0, 3.0])
        traj = body.simulate(
            omega0=[5.0, 0.5, 0.1],
            t_span=(0, 50),
            dt=0.005,
        )
        assert traj.energy_drift() < 1e-4


# ---------------------------------------------------------------------------
# Numerical accuracy tests: compare against analytical solutions
# ---------------------------------------------------------------------------


class TestNumericalAccuracy:
    """Compare simulations against known analytical results."""

    def test_harmonic_oscillator_period(self) -> None:
        """SHO with omega=2 has period pi. Position should return after T."""
        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.config import Params

        def H(q, p, params):
            return p[0] ** 2 / (2 * params.m) + 0.5 * params.k * q[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        params = Params(m=1.0, k=4.0)  # omega=2, T=pi

        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, jnp.pi),
            dt=0.001,
            params=params,
            integrator="leapfrog",
        )
        # After one full period, q should return to ~1.0
        assert float(traj.final_position[0]) == pytest.approx(1.0, abs=5e-3)

    def test_harmonic_oscillator_half_period(self) -> None:
        """After half period, position should be ~-1.0 (for cos motion)."""
        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.config import Params

        def H(q, p, params):
            return p[0] ** 2 / (2 * params.m) + 0.5 * params.k * q[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        params = Params(m=1.0, k=4.0)  # omega=2, T=pi

        traj = system.simulate(
            q0=[1.0],
            p0=[0.0],
            t_span=(0, jnp.pi / 2),
            dt=0.001,
            params=params,
            integrator="leapfrog",
        )
        # After half period, q should be ~-1.0
        assert float(traj.final_position[0]) == pytest.approx(-1.0, abs=5e-3)

    def test_kepler_orbit_period(self) -> None:
        """Circular orbit period: T = 2*pi*r^{3/2}/sqrt(GM)."""
        from neurosim.classical.nbody import NBody

        # Central mass = 1.0 at origin, test mass orbiting at r=1
        # Circular velocity: v = sqrt(GM/r) = 1.0
        system = NBody(
            masses=[1.0, 1e-6],
            positions=[[0, 0, 0], [1, 0, 0]],
            velocities=[[0, 0, 0], [0, 1, 0]],
            G=1.0,
            softening=1e-8,
        )
        # Kepler period T = 2*pi for r=1, M=1
        T = 2 * jnp.pi
        traj = system.simulate(
            t_span=(0, float(T)),
            n_steps=50000,
            save_every=500,
        )
        # Test particle should return close to (1,0,0)
        final_pos = traj.positions[-1, 1, :]
        assert float(final_pos[0]) == pytest.approx(1.0, abs=0.05)
        assert float(final_pos[1]) == pytest.approx(0.0, abs=0.05)

    def test_qho_eigenvalues_accuracy(self) -> None:
        """QHO eigenvalues E_n = (n+0.5)*hbar*omega with high accuracy."""
        from neurosim.quantum.schrodinger import HarmonicPotential
        from neurosim.quantum.stationary import solve_eigenvalue_problem

        result = solve_eigenvalue_problem(
            potential=HarmonicPotential(k=1.0),
            x_range=(-15, 15),
            n_points=1000,
            n_states=8,
            hbar=1.0,
            mass=1.0,
        )
        for n in range(8):
            expected = n + 0.5
            assert float(result.energies[n]) == pytest.approx(expected, abs=0.02)

    def test_free_particle_wavepacket_spreading(self) -> None:
        """Free Gaussian wavepacket width should grow as sqrt(1 + (t/tau)^2)."""
        from neurosim.quantum.schrodinger import (
            GaussianWavepacket,
            solve_schrodinger,
        )

        class ZeroPotential:
            def __call__(self, x):
                return jnp.zeros_like(x)

        sigma0 = 1.0
        hbar = 1.0
        mass = 1.0
        psi0 = GaussianWavepacket(x0=0.0, k0=0.0, sigma=sigma0)

        result = solve_schrodinger(
            psi0=psi0,
            potential=ZeroPotential(),
            x_range=(-30, 30),
            t_span=(0, 5),
            n_points=1000,
            dt=0.005,
            hbar=hbar,
            mass=mass,
            save_every=100,
        )

        # At the final time, check wavepacket has spread
        prob_final = jnp.abs(result.psi[-1]) ** 2
        prob_initial = jnp.abs(result.psi[0]) ** 2

        # Width of final distribution (second moment) should be larger
        x = result.x
        width_initial = jnp.sqrt(
            jnp.trapezoid(x**2 * prob_initial, x)
            - jnp.trapezoid(x * prob_initial, x) ** 2
        )
        width_final = jnp.sqrt(
            jnp.trapezoid(x**2 * prob_final, x) - jnp.trapezoid(x * prob_final, x) ** 2
        )
        assert float(width_final) > float(width_initial)

    def test_single_slit_first_minimum(self) -> None:
        """First minimum of single slit at sin(theta) = lambda/a."""
        from neurosim.optics.diffraction import single_slit

        a = 1e-4  # slit width
        lam = 500e-9  # wavelength
        result = single_slit(
            slit_width=a, wavelength=lam, n_points=10001, theta_max=0.02
        )
        # First minimum: sin(theta) = lambda/a
        sin_min = lam / a
        theta_min = jnp.arcsin(sin_min)
        # Find closest point to the first minimum
        idx_min = jnp.argmin(jnp.abs(result.theta - theta_min))
        # Intensity at first minimum should be very small
        assert float(result.intensity[idx_min]) < 0.01

    def test_waveguide_cutoff_ordering(self) -> None:
        """Lower-order modes should have lower cutoff frequencies."""
        from neurosim.em.waveguides import RectangularWaveguide

        wg = RectangularWaveguide(a=0.02286, b=0.01016)
        fc_10 = wg.cutoff_frequency(1, 0)
        fc_20 = wg.cutoff_frequency(2, 0)
        fc_01 = wg.cutoff_frequency(0, 1)
        assert fc_10 < fc_20
        assert fc_10 < fc_01


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: zero coupling, single particle, tiny/huge timesteps."""

    def test_zero_coupling_ising(self) -> None:
        """With J=0, magnetization should be near zero at any T."""
        from neurosim.statmech.ising import IsingLattice

        lattice = IsingLattice(size=(8, 8), J=0.0)
        result = lattice.run_metropolis(
            temperature=1.0,
            n_sweeps=1000,
            n_warmup=500,
            key=jax.random.PRNGKey(0),
        )
        # With no coupling, spins are independent; <|m|> ~ 1/sqrt(N)
        assert result["magnetization"] < 0.5

    def test_single_body_nbody(self) -> None:
        """Single particle N-body should just stay in place (no forces)."""
        from neurosim.classical.nbody import NBody

        system = NBody(
            masses=[1.0],
            positions=[[1.0, 2.0, 3.0]],
            velocities=[[0.1, 0.2, 0.3]],
            G=1.0,
        )
        traj = system.simulate(t_span=(0, 1), n_steps=100, save_every=10)
        # With constant velocity, final position = initial + v*t
        expected = jnp.array([1.1, 2.2, 3.3])
        assert jnp.allclose(traj.positions[-1, 0, :], expected, atol=1e-3)

    def test_negative_mass_rejected(self) -> None:
        """NBody should reject negative masses."""
        from neurosim.classical.nbody import NBody
        from neurosim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="positive"):
            NBody(
                masses=[-1.0],
                positions=[[0, 0, 0]],
                velocities=[[0, 0, 0]],
            )

    def test_zero_mass_rejected(self) -> None:
        """NBody should reject zero masses."""
        from neurosim.classical.nbody import NBody
        from neurosim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="positive"):
            NBody(
                masses=[0.0],
                positions=[[0, 0, 0]],
                velocities=[[0, 0, 0]],
            )

    def test_small_timestep_accuracy(self) -> None:
        """Very small dt should give very accurate single step."""
        from neurosim.classical.integrators import leapfrog

        q = jnp.array([1.0])
        p = jnp.array([0.0])
        dt = 1e-6

        def deriv(q, p, t, params):
            return p, -q

        q_new, p_new, _ = leapfrog(deriv, q, p, 0.0, dt, None)
        # Exact: q = cos(dt), p = -sin(dt)
        assert float(q_new[0]) == pytest.approx(float(jnp.cos(dt)), abs=1e-12)
        assert float(p_new[0]) == pytest.approx(float(-jnp.sin(dt)), abs=1e-12)

    def test_large_timestep_leapfrog_does_not_crash(self) -> None:
        """Leapfrog with large dt should not crash (may be inaccurate)."""
        from neurosim.classical.integrators import leapfrog

        q = jnp.array([1.0])
        p = jnp.array([0.0])
        dt = 1.0  # large

        def deriv(q, p, t, params):
            return p, -q

        q_new, p_new, t_new = leapfrog(deriv, q, p, 0.0, dt, None)
        assert jnp.isfinite(q_new[0])
        assert jnp.isfinite(p_new[0])
        assert t_new == pytest.approx(1.0)

    def test_stormer_verlet_is_leapfrog(self) -> None:
        """Stormer-Verlet should produce identical results to leapfrog."""
        from neurosim.classical.integrators import leapfrog, stormer_verlet

        q = jnp.array([1.0, 0.5])
        p = jnp.array([0.3, -0.2])

        def deriv(q, p, t, params):
            return p, -q

        q_lf, p_lf, t_lf = leapfrog(deriv, q, p, 0.0, 0.01, None)
        q_sv, p_sv, t_sv = stormer_verlet(deriv, q, p, 0.0, 0.01, None)
        assert jnp.allclose(q_lf, q_sv, atol=1e-15)
        assert jnp.allclose(p_lf, p_sv, atol=1e-15)
        assert t_lf == t_sv

    def test_lagrangian_invalid_tspan(self) -> None:
        """t_end <= t_start should raise ConfigurationError."""
        from neurosim.classical.lagrangian import LagrangianSystem
        from neurosim.exceptions import ConfigurationError

        def L(q, qdot, p):
            return 0.5 * qdot[0] ** 2

        system = LagrangianSystem(L, n_dof=1)
        with pytest.raises(ConfigurationError, match="must be > t_start"):
            system.simulate(q0=[0.0], qdot0=[0.0], t_span=(5, 3))

    def test_lagrangian_negative_dt(self) -> None:
        """Negative dt should raise ConfigurationError."""
        from neurosim.classical.lagrangian import LagrangianSystem
        from neurosim.exceptions import ConfigurationError

        def L(q, qdot, p):
            return 0.5 * qdot[0] ** 2

        system = LagrangianSystem(L, n_dof=1)
        with pytest.raises(ConfigurationError, match="positive"):
            system.simulate(q0=[0.0], qdot0=[0.0], t_span=(0, 1), dt=-0.01)

    def test_hamiltonian_invalid_tspan(self) -> None:
        """Hamiltonian t_end <= t_start should raise."""
        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.exceptions import ConfigurationError

        def H(q, p, params):
            return 0.5 * p[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        with pytest.raises(ConfigurationError, match="must be > t_start"):
            system.simulate(q0=[0.0], p0=[0.0], t_span=(5, 3), params=None)

    def test_hamiltonian_shape_mismatch_q(self) -> None:
        """Wrong q0 shape should raise ConfigurationError."""
        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.exceptions import ConfigurationError

        def H(q, p, params):
            return 0.5 * p[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        with pytest.raises(ConfigurationError, match="q0 shape"):
            system.simulate(q0=[0.0, 1.0], p0=[0.0], t_span=(0, 1), params=None)

    def test_hamiltonian_shape_mismatch_p(self) -> None:
        """Wrong p0 shape should raise ConfigurationError."""
        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.exceptions import ConfigurationError

        def H(q, p, params):
            return 0.5 * p[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        with pytest.raises(ConfigurationError, match="p0 shape"):
            system.simulate(q0=[0.0], p0=[0.0, 1.0], t_span=(0, 1), params=None)

    def test_hamiltonian_invalid_n_dof(self) -> None:
        """n_dof=0 should raise ConfigurationError."""
        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.exceptions import ConfigurationError

        def H(q, p, params):
            return 0.0

        with pytest.raises(ConfigurationError, match="n_dof"):
            HamiltonianSystem(H, n_dof=0)


# ---------------------------------------------------------------------------
# Visualization module smoke tests
# ---------------------------------------------------------------------------


class TestVisualization:
    """Test that visualization functions produce valid figures."""

    @staticmethod
    def _require_matplotlib():
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        return matplotlib

    def test_plot_phase_space_runs(self) -> None:
        """plot_phase_space should produce a matplotlib figure."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.viz.phase_space import plot_phase_space

        def H(q, p, params):
            return 0.5 * p[0] ** 2 + 0.5 * q[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0], p0=[0.0], t_span=(0, 6.28), dt=0.01, params=None
        )
        fig = plot_phase_space(traj)
        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_energy_runs(self) -> None:
        """plot_energy should produce a figure with 2 subplots."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.classical.hamiltonian import HamiltonianSystem
        from neurosim.viz.phase_space import plot_energy

        def H(q, p, params):
            return 0.5 * p[0] ** 2 + 0.5 * q[0] ** 2

        system = HamiltonianSystem(H, n_dof=1)
        traj = system.simulate(
            q0=[1.0], p0=[0.0], t_span=(0, 6.28), dt=0.01, params=None
        )
        fig = plot_energy(traj)
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_energy_no_energy_raises(self) -> None:
        """plot_energy should raise if trajectory has no energy data."""
        self._require_matplotlib()

        from neurosim.exceptions import VisualizationError
        from neurosim.state import Trajectory
        from neurosim.viz.phase_space import plot_energy

        traj = Trajectory(
            t=jnp.array([0.0, 1.0]),
            q=jnp.array([[0.0], [1.0]]),
            p=jnp.array([[1.0], [0.0]]),
            energy=None,
        )
        with pytest.raises(VisualizationError, match="energy"):
            plot_energy(traj)

    def test_plot_phase_space_multi_dof(self) -> None:
        """plot_phase_space with multiple coords should create subplots."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.state import Trajectory
        from neurosim.viz.phase_space import plot_phase_space

        traj = Trajectory(
            t=jnp.linspace(0, 1, 50),
            q=jnp.zeros((50, 2)),
            p=jnp.ones((50, 2)),
        )
        fig = plot_phase_space(traj, coords=[0, 1])
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_animate_pendulum_runs(self) -> None:
        """animate_pendulum should return an animation object."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.state import Trajectory
        from neurosim.viz.animate import animate_pendulum

        traj = Trajectory(
            t=jnp.linspace(0, 1, 20),
            q=jnp.tile(jnp.array([0.3]), (20, 1)),
            p=jnp.zeros((20, 1)),
        )
        anim = animate_pendulum(traj)
        assert anim is not None
        plt.close("all")

    def test_animate_wavefunction_runs(self) -> None:
        """animate_wavefunction should return an animation object."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.state import QuantumResult
        from neurosim.viz.animate import animate_wavefunction

        x = jnp.linspace(-5, 5, 100)
        psi = jnp.exp(-(x**2) / 2.0).astype(jnp.complex128)
        psi = psi / jnp.sqrt(jnp.trapezoid(jnp.abs(psi) ** 2, x))

        result = QuantumResult(
            t=jnp.array([0.0, 1.0]),
            psi=jnp.stack([psi, psi]),
            x=x,
            potential=jnp.zeros_like(x),
        )
        anim = animate_wavefunction(result)
        assert anim is not None
        plt.close("all")

    def test_plot_field_snapshot_runs(self) -> None:
        """plot_field_snapshot should create a figure."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.state import EMFieldHistory
        from neurosim.viz.fields import plot_field_snapshot

        # Create minimal field data
        fields = EMFieldHistory(
            t=jnp.array([0.0]),
            ez=jnp.zeros((1, 10, 10)),
            hx=jnp.zeros((1, 10, 10)),
            hy=jnp.zeros((1, 10, 10)),
            grid_x=jnp.arange(10) * 0.01,
            grid_y=jnp.arange(10) * 0.01,
        )
        fig = plot_field_snapshot(fields, step=0)
        assert fig is not None
        plt.close(fig)

    def test_plot_field_snapshot_invalid_component(self) -> None:
        """plot_field_snapshot with bad component should raise."""
        self._require_matplotlib()

        from neurosim.exceptions import VisualizationError
        from neurosim.state import EMFieldHistory
        from neurosim.viz.fields import plot_field_snapshot

        fields = EMFieldHistory(
            t=jnp.array([0.0]),
            ez=jnp.zeros((1, 10, 10)),
            hx=jnp.zeros((1, 10, 10)),
            hy=jnp.zeros((1, 10, 10)),
            grid_x=jnp.arange(10) * 0.01,
            grid_y=jnp.arange(10) * 0.01,
        )
        with pytest.raises(VisualizationError, match="Unknown component"):
            plot_field_snapshot(fields, component="Bz")

    def test_plot_phase_transition_runs(self) -> None:
        """plot_phase_transition should produce a figure."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.state import IsingResult
        from neurosim.viz.phase_space import plot_phase_transition

        result = IsingResult(
            temperatures=jnp.linspace(1, 4, 5),
            magnetizations=jnp.array([0.9, 0.7, 0.3, 0.1, 0.05]),
            energies=jnp.array([-2.0, -1.5, -1.0, -0.5, -0.3]),
            specific_heats=jnp.array([0.1, 0.5, 2.0, 0.5, 0.1]),
            susceptibilities=jnp.array([0.1, 0.5, 3.0, 0.5, 0.1]),
        )
        fig = plot_phase_transition(result)
        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_specific_heat_runs(self) -> None:
        """plot_specific_heat should produce a figure."""
        self._require_matplotlib()
        import matplotlib.pyplot as plt

        from neurosim.state import IsingResult
        from neurosim.viz.phase_space import plot_specific_heat

        result = IsingResult(
            temperatures=jnp.linspace(1, 4, 5),
            magnetizations=jnp.zeros(5),
            energies=jnp.zeros(5),
            specific_heats=jnp.array([0.1, 0.5, 2.0, 0.5, 0.1]),
            susceptibilities=jnp.zeros(5),
        )
        fig = plot_specific_heat(result)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Config validation: invalid params, negative masses, etc."""

    def test_simulation_config_t_end_must_be_positive(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import SimulationConfig

        with pytest.raises(ValidationError):
            SimulationConfig(t_end=0.0)

    def test_simulation_config_dt_must_be_positive(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import SimulationConfig

        with pytest.raises(ValidationError):
            SimulationConfig(t_end=1.0, dt=0.0)

    def test_simulation_config_save_every_must_be_ge1(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import SimulationConfig

        with pytest.raises(ValidationError):
            SimulationConfig(t_end=1.0, save_every=0)

    def test_nbody_config_negative_softening(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import NBodyConfig

        with pytest.raises(ValidationError):
            NBodyConfig(softening=-1.0)

    def test_nbody_config_zero_G(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import NBodyConfig

        with pytest.raises(ValidationError):
            NBodyConfig(G=0.0)

    def test_em_config_zero_resolution(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import EMConfig

        with pytest.raises(ValidationError):
            EMConfig(resolution=0.0)

    def test_em_config_invalid_courant(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import EMConfig

        with pytest.raises(ValidationError):
            EMConfig(courant_number=1.5)

    def test_quantum_config_negative_hbar(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import QuantumConfig

        with pytest.raises(ValidationError):
            QuantumConfig(hbar=-1.0)

    def test_quantum_config_negative_mass(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import QuantumConfig

        with pytest.raises(ValidationError):
            QuantumConfig(mass=-1.0)

    def test_quantum_config_too_few_points(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import QuantumConfig

        with pytest.raises(ValidationError):
            QuantumConfig(n_points=5)

    def test_quantum_config_invalid_method(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import QuantumConfig

        with pytest.raises(ValidationError):
            QuantumConfig(method="euler")

    def test_ising_config_invalid_algorithm(self) -> None:
        from pydantic import ValidationError

        from neurosim.config import IsingConfig

        with pytest.raises(ValidationError):
            IsingConfig(algorithm="invalid")

    def test_params_access_nonexistent(self) -> None:
        from neurosim.config import Params

        p = Params(x=1.0)
        with pytest.raises(AttributeError):
            _ = p.y

    def test_params_private_attr_raises(self) -> None:
        from neurosim.config import Params

        p = Params(x=1.0)
        with pytest.raises(AttributeError):
            _ = p._private


# ---------------------------------------------------------------------------
# Rigid body quaternion normalization tests
# ---------------------------------------------------------------------------


class TestRigidBodyQuaternion:
    """Test quaternion normalization during rigid body simulation."""

    def test_quaternion_stays_normalized(self) -> None:
        """Quaternion norm should stay 1 throughout the simulation."""
        from neurosim.classical.rigid_body import RigidBody

        body = RigidBody(inertia=[1.0, 2.0, 3.0])
        traj = body.simulate(
            omega0=[5.0, 0.5, 0.1],
            t_span=(0, 20),
            dt=0.01,
        )
        # q stores quaternions: shape (n_steps, 4)
        quat_norms = jnp.linalg.norm(traj.q, axis=1)
        assert jnp.allclose(quat_norms, 1.0, atol=1e-8)

    def test_custom_initial_quaternion(self) -> None:
        """Non-identity initial quaternion should be normalized."""
        from neurosim.classical.rigid_body import RigidBody

        body = RigidBody(inertia=[1.0, 1.0, 1.0])
        # Unnormalized quaternion
        traj = body.simulate(
            omega0=[0.0, 0.0, 1.0],
            quat0=[2.0, 0.0, 0.0, 0.0],
            t_span=(0, 5),
            dt=0.01,
        )
        # First quaternion should have been normalized
        assert float(jnp.linalg.norm(traj.q[0])) == pytest.approx(1.0, abs=1e-10)

    def test_angular_momentum_conservation_torque_free(self) -> None:
        """Torque-free: |L| should be conserved."""
        from neurosim.classical.rigid_body import RigidBody

        body = RigidBody(inertia=[1.0, 2.0, 3.0])
        traj = body.simulate(
            omega0=[1.0, 0.5, 0.3],
            t_span=(0, 20),
            dt=0.005,
        )
        # p stores omega; compute |L| = |I*omega| at each step
        inertia = jnp.array([1.0, 2.0, 3.0])
        L_norms = jnp.linalg.norm(traj.p * inertia, axis=1)
        # |L| should be constant (conserved in body frame for torque-free)
        assert jnp.allclose(L_norms, L_norms[0], rtol=1e-3)

    def test_rigid_body_invalid_inertia_shape(self) -> None:
        """Wrong shape inertia should raise."""
        from neurosim.classical.rigid_body import RigidBody
        from neurosim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="shape"):
            RigidBody(inertia=[1.0, 2.0])

    def test_rigid_body_invalid_omega_shape(self) -> None:
        """Wrong omega0 shape should raise."""
        from neurosim.classical.rigid_body import RigidBody
        from neurosim.exceptions import ConfigurationError

        body = RigidBody(inertia=[1.0, 2.0, 3.0])
        with pytest.raises(ConfigurationError, match="shape"):
            body.simulate(omega0=[1.0, 0.5])

    def test_symmetric_top_precession(self) -> None:
        """Symmetric top (I1=I2!=I3) should show regular precession."""
        from neurosim.classical.rigid_body import RigidBody

        body = RigidBody(inertia=[1.0, 1.0, 2.0])
        traj = body.simulate(
            omega0=[0.5, 0.0, 3.0],
            t_span=(0, 10),
            dt=0.005,
        )
        # Energy should be conserved
        assert traj.energy_drift() < 1e-5
        # omega_z should stay constant for torque-free symmetric top
        omega_z = traj.p[:, 2]
        assert jnp.allclose(omega_z, omega_z[0], atol=1e-4)


# ---------------------------------------------------------------------------
# Diffraction pattern symmetry tests
# ---------------------------------------------------------------------------


class TestDiffractionSymmetry:
    """Verify diffraction patterns have expected symmetries."""

    def test_single_slit_even_symmetry(self) -> None:
        """Single slit pattern I(theta) = I(-theta)."""
        from neurosim.optics.diffraction import single_slit

        result = single_slit(slit_width=5e-5, wavelength=600e-9, n_points=2001)
        n = result.intensity.shape[0]
        left = result.intensity[: n // 2]
        right = jnp.flip(result.intensity[n // 2 + 1 :])
        assert jnp.allclose(left, right, atol=1e-12)

    def test_double_slit_even_symmetry(self) -> None:
        """Double slit pattern should also be symmetric."""
        from neurosim.optics.diffraction import double_slit

        result = double_slit(
            slit_width=5e-5,
            slit_separation=2e-4,
            wavelength=600e-9,
            n_points=2001,
        )
        n = result.intensity.shape[0]
        left = result.intensity[: n // 2]
        right = jnp.flip(result.intensity[n // 2 + 1 :])
        assert jnp.allclose(left, right, atol=1e-12)

    def test_circular_aperture_even_symmetry(self) -> None:
        """Circular aperture Airy pattern should be symmetric."""
        from neurosim.optics.diffraction import circular_aperture

        result = circular_aperture(diameter=1e-3, wavelength=500e-9, n_points=2001)
        n = result.intensity.shape[0]
        left = result.intensity[: n // 2]
        right = jnp.flip(result.intensity[n // 2 + 1 :])
        assert jnp.allclose(left, right, atol=1e-10)

    def test_single_slit_peak_value_one(self) -> None:
        """Central peak I(0)=1 for all slit widths and wavelengths."""
        from neurosim.optics.diffraction import single_slit

        for a, lam in [(1e-4, 500e-9), (5e-5, 700e-9), (2e-4, 400e-9)]:
            result = single_slit(slit_width=a, wavelength=lam, n_points=1001)
            center = result.intensity.shape[0] // 2
            assert float(result.intensity[center]) == pytest.approx(1.0, abs=1e-10)

    def test_double_slit_has_multiple_peaks(self) -> None:
        """Double slit should have interference maxima (multiple peaks)."""
        from neurosim.optics.diffraction import double_slit

        result = double_slit(
            slit_width=5e-5,
            slit_separation=5e-4,
            wavelength=500e-9,
            n_points=5000,
            theta_max=0.05,
        )
        # Count peaks (local maxima) above 0.1
        intensity = result.intensity
        is_peak = (intensity[1:-1] > intensity[:-2]) & (intensity[1:-1] > intensity[2:])
        n_peaks = int(jnp.sum(is_peak & (intensity[1:-1] > 0.1)))
        assert n_peaks >= 3  # at least 3 visible peaks

    def test_single_slit_invalid_negative_width(self) -> None:
        from neurosim.exceptions import ConfigurationError
        from neurosim.optics.diffraction import single_slit

        with pytest.raises(ConfigurationError):
            single_slit(slit_width=-1e-4, wavelength=500e-9)

    def test_single_slit_invalid_negative_wavelength(self) -> None:
        from neurosim.exceptions import ConfigurationError
        from neurosim.optics.diffraction import single_slit

        with pytest.raises(ConfigurationError):
            single_slit(slit_width=1e-4, wavelength=-500e-9)

    def test_double_slit_separation_less_than_width_error(self) -> None:
        from neurosim.exceptions import ConfigurationError
        from neurosim.optics.diffraction import double_slit

        with pytest.raises(ConfigurationError, match="separation"):
            double_slit(
                slit_width=1e-3,
                slit_separation=1e-4,
                wavelength=500e-9,
            )

    def test_circular_aperture_invalid_diameter(self) -> None:
        from neurosim.exceptions import ConfigurationError
        from neurosim.optics.diffraction import circular_aperture

        with pytest.raises(ConfigurationError):
            circular_aperture(diameter=-1, wavelength=500e-9)

    def test_angle_degrees_property(self) -> None:
        """DiffractionResult.angle_degrees should convert correctly."""
        from neurosim.optics.diffraction import single_slit

        result = single_slit(slit_width=1e-4, wavelength=500e-9, n_points=101)
        degs = result.angle_degrees
        assert degs.shape == result.theta.shape
        assert jnp.allclose(degs, jnp.degrees(result.theta), atol=1e-12)


# ---------------------------------------------------------------------------
# Density matrix trace preservation tests
# ---------------------------------------------------------------------------


class TestDensityMatrixTracePreservation:
    """Verify density matrix trace is preserved during Lindblad evolution."""

    def test_trace_preserved_unitary(self) -> None:
        """Unitary evolution (no dissipation) should preserve Tr(rho)=1."""
        from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve

        psi = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)

        # Non-zero Hamiltonian, no Lindblad operators
        H = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
        result = lindblad_evolve(dm, H, [], [], t_span=(0, 5), dt=0.01)
        # Check trace at every saved step
        for i in range(result.rho.shape[0]):
            trace = jnp.real(jnp.trace(result.rho[i]))
            assert float(trace) == pytest.approx(1.0, abs=1e-6)

    def test_trace_preserved_dissipative(self) -> None:
        """Dissipative Lindblad evolution should still preserve trace."""
        from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve

        psi = jnp.array([0.0, 1.0], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)

        H = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
        L = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)

        result = lindblad_evolve(dm, H, [L], [0.5], t_span=(0, 10), dt=0.01)
        for i in range(result.rho.shape[0]):
            trace = jnp.real(jnp.trace(result.rho[i]))
            assert float(trace) == pytest.approx(1.0, abs=1e-4)

    def test_purity_decreases_under_dissipation(self) -> None:
        """Purity should decrease from 1 under dissipative evolution."""
        from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve

        psi = jnp.array([0.0, 1.0], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)

        H = jnp.zeros((2, 2), dtype=jnp.complex128)
        L = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)

        result = lindblad_evolve(dm, H, [L], [1.0], t_span=(0, 5), dt=0.01)
        # Initial purity = 1 (pure state)
        assert float(result.purity[0]) == pytest.approx(1.0, abs=1e-6)
        # Purity should drop below 1 mid-evolution
        assert float(jnp.min(result.purity)) < 0.99

    def test_purity_preserved_unitary(self) -> None:
        """Unitary evolution should preserve purity exactly."""
        from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve

        psi = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)

        H = jnp.array([[1.0, 0.5], [0.5, -1.0]], dtype=jnp.complex128)
        result = lindblad_evolve(dm, H, [], [], t_span=(0, 10), dt=0.01)
        # All purities should be ~1
        assert jnp.allclose(result.purity, 1.0, atol=1e-5)

    def test_thermal_state_is_mixed(self) -> None:
        """Thermal state at finite T should have purity < 1."""
        from neurosim.quantum.density_matrix import DensityMatrix

        H = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
        dm = DensityMatrix.thermal_state(H, temperature=1.0)
        assert dm.purity() < 1.0
        assert float(jnp.real(jnp.trace(dm.rho))) == pytest.approx(1.0, abs=1e-10)

    def test_thermal_state_negative_temp_raises(self) -> None:
        """Negative temperature should raise ConfigurationError."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.quantum.density_matrix import DensityMatrix

        H = jnp.eye(2, dtype=jnp.complex128)
        with pytest.raises(ConfigurationError, match="positive"):
            DensityMatrix.thermal_state(H, temperature=-1.0)

    def test_expectation_value(self) -> None:
        """Tr(rho * sigma_z) for |0> should be 1."""
        from neurosim.quantum.density_matrix import DensityMatrix

        psi = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)
        sigma_z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
        val = dm.expectation(sigma_z)
        assert float(jnp.real(val)) == pytest.approx(1.0, abs=1e-10)

    def test_dimension_property(self) -> None:
        """DensityMatrix.dimension should return correct Hilbert space dim."""
        from neurosim.quantum.density_matrix import DensityMatrix

        rho = jnp.eye(4, dtype=jnp.complex128) / 4
        dm = DensityMatrix(rho=rho)
        assert dm.dimension == 4

    def test_lindblad_mismatched_ops_rates(self) -> None:
        """Mismatched Lindblad ops and rates should raise."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve

        dm = DensityMatrix.from_pure_state(jnp.array([1.0, 0.0], dtype=jnp.complex128))
        H = jnp.zeros((2, 2), dtype=jnp.complex128)
        L = jnp.eye(2, dtype=jnp.complex128)
        with pytest.raises(ConfigurationError, match="match"):
            lindblad_evolve(dm, H, [L], [0.1, 0.2])


# ---------------------------------------------------------------------------
# Quantum Schrodinger edge cases
# ---------------------------------------------------------------------------


class TestSchrodingerEdgeCases:
    """Edge cases for the Schrodinger solver."""

    def test_invalid_x_range(self) -> None:
        """x_max <= x_min should raise."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.quantum.schrodinger import (
            GaussianWavepacket,
            HarmonicPotential,
            solve_schrodinger,
        )

        with pytest.raises(ConfigurationError, match="x_max"):
            solve_schrodinger(
                psi0=GaussianWavepacket(x0=0, k0=0, sigma=1),
                potential=HarmonicPotential(k=1.0),
                x_range=(5, -5),
            )

    def test_harmonic_potential_symmetry(self) -> None:
        """V(x) = V(-x) for symmetric harmonic potential."""
        from neurosim.quantum.schrodinger import HarmonicPotential

        V = HarmonicPotential(k=2.0, x0=0.0)
        x = jnp.linspace(-5, 5, 101)
        vals = V(x)
        assert jnp.allclose(vals, jnp.flip(vals), atol=1e-12)

    def test_double_well_symmetry(self) -> None:
        """Double well V(x) = a(x^2-b)^2 should be symmetric."""
        from neurosim.quantum.schrodinger import DoubleWellPotential

        V = DoubleWellPotential(a=1.0, b=1.0)
        x = jnp.linspace(-3, 3, 201)
        vals = V(x)
        assert jnp.allclose(vals, jnp.flip(vals), atol=1e-12)

    def test_norm_preservation_harmonic(self) -> None:
        """Norm should be preserved for harmonic potential evolution."""
        from neurosim.quantum.schrodinger import (
            GaussianWavepacket,
            HarmonicPotential,
            solve_schrodinger,
        )

        result = solve_schrodinger(
            psi0=GaussianWavepacket(x0=0.0, k0=0.0, sigma=1.0),
            potential=HarmonicPotential(k=1.0),
            x_range=(-15, 15),
            t_span=(0, 5),
            n_points=500,
            dt=0.01,
            save_every=50,
        )
        for i in range(result.n_steps):
            norm = float(jnp.trapezoid(jnp.abs(result.psi[i]) ** 2, result.x))
            assert norm == pytest.approx(1.0, abs=1e-3)

    def test_stationary_x_range_invalid(self) -> None:
        """Stationary solver with invalid x_range should raise."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.quantum.schrodinger import HarmonicPotential
        from neurosim.quantum.stationary import solve_eigenvalue_problem

        with pytest.raises(ConfigurationError, match="x_max"):
            solve_eigenvalue_problem(
                potential=HarmonicPotential(k=1.0),
                x_range=(10, -10),
            )

    def test_stationary_n_states_exceeds_n_points(self) -> None:
        """n_states > n_points should raise."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.quantum.schrodinger import HarmonicPotential
        from neurosim.quantum.stationary import solve_eigenvalue_problem

        with pytest.raises(ConfigurationError, match="n_states"):
            solve_eigenvalue_problem(
                potential=HarmonicPotential(k=1.0),
                n_points=50,
                n_states=100,
            )


# ---------------------------------------------------------------------------
# Spin chain tests
# ---------------------------------------------------------------------------


class TestSpinChainEdgeCases:
    """Edge cases for spin chain module."""

    def test_min_sites(self) -> None:
        """Minimum valid chain is 2 sites."""
        from neurosim.quantum.spin import SpinChain

        chain = SpinChain(n_sites=2, J=1.0)
        result = chain.diagonalize(n_states=4)
        assert result.energies.shape[0] == 4
        assert result.n_sites == 2

    def test_single_site_rejected(self) -> None:
        """1 site should raise ConfigurationError."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.quantum.spin import SpinChain

        with pytest.raises(ConfigurationError, match="n_sites"):
            SpinChain(n_sites=1)

    def test_periodic_vs_open_chain(self) -> None:
        """Periodic and open BC should give different ground state energies."""
        from neurosim.quantum.spin import SpinChain

        open_chain = SpinChain(n_sites=4, J=1.0, periodic=False)
        periodic_chain = SpinChain(n_sites=4, J=1.0, periodic=True)
        e_open = open_chain.diagonalize(n_states=1).energies[0]
        e_periodic = periodic_chain.diagonalize(n_states=1).energies[0]
        # Different boundary conditions give different energies
        assert not jnp.isclose(e_open, e_periodic)

    def test_hamiltonian_is_hermitian(self) -> None:
        """Spin chain Hamiltonian should be Hermitian."""
        from neurosim.quantum.spin import SpinChain

        chain = SpinChain(n_sites=3, J=1.0, h=0.5)
        H = chain.build_hamiltonian()
        assert jnp.allclose(H, H.conj().T, atol=1e-12)

    def test_zero_field_magnetization(self) -> None:
        """Ground state of AFM (J<0) at h=0 should have ~0 magnetization."""
        from neurosim.quantum.spin import SpinChain

        chain = SpinChain(n_sites=4, J=-1.0, h=0.0)
        result = chain.diagonalize(n_states=1)
        # AFM ground state has near-zero net magnetization per site
        assert abs(float(result.magnetization[0])) < 0.1


# ---------------------------------------------------------------------------
# Ray tracing edge cases
# ---------------------------------------------------------------------------


class TestRayTracingEdgeCases:
    """Edge cases for geometric optics."""

    def test_zero_focal_length_raises(self) -> None:
        """Thin lens with f=0 should raise."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.optics.ray_tracing import ThinLens

        lens = ThinLens(f=0)
        with pytest.raises(ConfigurationError, match="Focal length"):
            lens.matrix()

    def test_spherical_mirror_zero_R_raises(self) -> None:
        """Spherical mirror with R=0 should raise."""
        from neurosim.exceptions import ConfigurationError
        from neurosim.optics.ray_tracing import SphericalMirror

        mirror = SphericalMirror(R=0)
        with pytest.raises(ConfigurationError, match="Radius"):
            mirror.matrix()

    def test_diverging_lens(self) -> None:
        """Negative focal length (diverging lens) should work."""
        from neurosim.optics.ray_tracing import Ray, ThinLens, trace_system

        ray = Ray(y=1.0, theta=0.0)
        lens = ThinLens(f=-0.5, position=0.0)
        result = trace_system(ray, [lens])
        # Diverging lens: theta should be positive (deflects away from axis)
        assert result.angles[-1] > 0

    def test_ray_vector_conversion(self) -> None:
        """Ray.to_vector should give correct [y, theta]."""
        from neurosim.optics.ray_tracing import Ray

        ray = Ray(y=2.5, theta=0.03)
        v = ray.to_vector()
        assert float(v[0]) == pytest.approx(2.5)
        assert float(v[1]) == pytest.approx(0.03)

    def test_system_matrix_determinant(self) -> None:
        """System matrix should have det=1 (symplecticity of ray optics)."""
        from neurosim.optics.ray_tracing import Ray, ThinLens, trace_system

        ray = Ray(y=1.0, theta=0.0)
        elements = [
            ThinLens(f=0.5, position=0.0),
            ThinLens(f=0.3, position=1.0),
        ]
        result = trace_system(ray, elements)
        det = jnp.linalg.det(result.system_matrix)
        assert float(det) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Waveguide edge cases
# ---------------------------------------------------------------------------


class TestWaveguideEdgeCases:
    """Edge cases for waveguide module."""

    def test_negative_dimensions_raises(self) -> None:
        from neurosim.em.waveguides import RectangularWaveguide
        from neurosim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="positive"):
            RectangularWaveguide(a=-0.01, b=0.01)

    def test_zero_dimensions_raises(self) -> None:
        from neurosim.em.waveguides import RectangularWaveguide
        from neurosim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="positive"):
            RectangularWaveguide(a=0.0, b=0.01)

    def test_invalid_mode_type(self) -> None:
        from neurosim.em.waveguides import RectangularWaveguide
        from neurosim.exceptions import ConfigurationError

        wg = RectangularWaveguide(a=0.02, b=0.01)
        with pytest.raises(ConfigurationError, match="TE"):
            wg.compute_mode("XX", 1, 0)

    def test_tm_mode_requires_mn_ge_1(self) -> None:
        from neurosim.em.waveguides import RectangularWaveguide
        from neurosim.exceptions import ConfigurationError

        wg = RectangularWaveguide(a=0.02, b=0.01)
        with pytest.raises(ConfigurationError, match="m >= 1"):
            wg.compute_mode("TM", 0, 1)


# ---------------------------------------------------------------------------
# EM charges edge cases
# ---------------------------------------------------------------------------


class TestChargesEdgeCases:
    """Edge cases for charge dynamics."""

    def test_single_charge_no_interaction(self) -> None:
        """Single charge should move freely (no Coulomb interaction)."""
        from neurosim.em.charges import ChargeSystem, PointCharge

        q = PointCharge(charge=1e-6, mass=1e-3, position=[0, 0, 0], velocity=[1, 0, 0])
        system = ChargeSystem(charges=[q])
        traj = system.simulate(t_span=(0, 0.001), n_steps=100, save_every=10)
        # Should move with constant velocity
        final_x = float(traj.positions[-1, 0, 0])
        expected_x = 0.001 * 1  # v * t
        assert final_x == pytest.approx(expected_x, abs=1e-5)

    def test_empty_charges_raises(self) -> None:
        from neurosim.em.charges import ChargeSystem
        from neurosim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="at least one"):
            ChargeSystem(charges=[])


# ---------------------------------------------------------------------------
# Optimization edge cases
# ---------------------------------------------------------------------------


class TestOptimizeEdgeCases:
    """Edge cases for the optimizer."""

    def test_already_at_minimum(self) -> None:
        """Starting at the minimum should converge immediately."""
        from neurosim.optimize import optimize

        def obj(x):
            return x**2

        result = optimize(
            obj,
            initial_guess=0.0,
            learning_rate=0.1,
            tolerance=1e-8,
        )
        assert result.converged
        assert result.n_iterations == 0

    def test_non_convergence(self) -> None:
        """With too few iterations, should not converge."""
        from neurosim.optimize import optimize

        def obj(x):
            return (x - 100.0) ** 2

        result = optimize(
            obj,
            initial_guess=0.0,
            learning_rate=0.001,
            max_iterations=5,
        )
        assert not result.converged
        assert result.n_iterations == 5

    def test_sensitivity_function(self) -> None:
        """Sensitivity (Jacobian) should match expected gradient."""
        from neurosim.optimize import sensitivity

        def sim_fn(params):
            return params**2

        s = sensitivity(sim_fn, jnp.array([3.0]))
        # d/d(params) of params^2 = 2*params = 6
        # sensitivity returns a Jacobian matrix; for 1D->1D it's shape (1,1)
        assert float(s[0, 0]) == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# State container tests
# ---------------------------------------------------------------------------


class TestStateContainers:
    """Tests for state dataclasses."""

    def test_trajectory_energy_drift_no_energy(self) -> None:
        """energy_drift should return 0.0 when energy is None."""
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 1.0]),
            q=jnp.array([[0.0], [1.0]]),
            p=jnp.array([[1.0], [0.0]]),
            energy=None,
        )
        assert traj.energy_drift() == 0.0

    def test_trajectory_energy_drift_near_zero_energy(self) -> None:
        """When E0 ~ 0, energy_drift should use absolute measure."""
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.array([0.0, 1.0]),
            q=jnp.array([[0.0], [1.0]]),
            p=jnp.array([[0.0], [0.0]]),
            energy=jnp.array([0.0, 1e-16]),
        )
        drift = traj.energy_drift()
        assert drift < 1e-14

    def test_trajectory_properties(self) -> None:
        from neurosim.state import Trajectory

        traj = Trajectory(
            t=jnp.linspace(0, 10, 100),
            q=jnp.zeros((100, 3)),
            p=jnp.ones((100, 3)),
        )
        assert traj.n_steps == 100
        assert traj.n_dof == 3
        assert traj.duration == pytest.approx(10.0, abs=0.2)
        assert traj.final_position.shape == (3,)
        assert traj.final_momentum.shape == (3,)

    def test_nbody_trajectory_properties(self) -> None:
        from neurosim.state import NBodyTrajectory

        traj = NBodyTrajectory(
            t=jnp.linspace(0, 1, 10),
            positions=jnp.zeros((10, 3, 3)),
            velocities=jnp.zeros((10, 3, 3)),
            masses=jnp.ones(3),
        )
        assert traj.n_steps == 10
        assert traj.n_bodies == 3
        assert traj.final_position.shape == (3, 3)

    def test_quantum_result_probability(self) -> None:
        """QuantumResult.probability should be |psi|^2."""
        from neurosim.state import QuantumResult

        x = jnp.linspace(-1, 1, 10)
        psi = jnp.ones((2, 10), dtype=jnp.complex128)
        result = QuantumResult(
            t=jnp.array([0.0, 1.0]),
            psi=psi,
            x=x,
            potential=jnp.zeros(10),
        )
        assert jnp.allclose(result.probability, 1.0)


# ---------------------------------------------------------------------------
# FDTD edge cases
# ---------------------------------------------------------------------------


class TestFDTDEdgeCases:
    """Edge cases for FDTD Maxwell solver."""

    def test_source_out_of_bounds_raises(self) -> None:
        from neurosim.em.fdtd import EMGrid, PlaneWave
        from neurosim.exceptions import ConfigurationError

        grid = EMGrid(size=(50, 50))
        with pytest.raises(ConfigurationError, match="out of grid"):
            grid.add_source(PlaneWave(frequency=1e9, y=100))

    def test_conductor_out_of_bounds_raises(self) -> None:
        from neurosim.em.fdtd import EMGrid, Wall
        from neurosim.exceptions import ConfigurationError

        grid = EMGrid(size=(50, 50))
        with pytest.raises(ConfigurationError, match="out of grid"):
            grid.add_conductor(Wall(y=100))

    def test_em_grid_properties(self) -> None:
        from neurosim.em.fdtd import EMGrid

        grid = EMGrid(size=(100, 80))
        assert grid.size == (100, 80)


# ---------------------------------------------------------------------------
# Ising model edge cases
# ---------------------------------------------------------------------------


class TestIsingEdgeCases:
    """Edge cases for Ising model."""

    def test_negative_temperature_raises(self) -> None:
        from neurosim.exceptions import ConfigurationError
        from neurosim.statmech.ising import IsingLattice

        lattice = IsingLattice(size=(4, 4))
        with pytest.raises(ConfigurationError, match="positive"):
            lattice.run_metropolis(temperature=-1.0)

    def test_lattice_properties(self) -> None:
        from neurosim.statmech.ising import IsingLattice

        lattice = IsingLattice(size=(10, 8))
        assert lattice.size == (10, 8)
        assert lattice.n_spins == 80

    def test_external_field_ising(self) -> None:
        """Strong external field should align spins."""
        from neurosim.statmech.ising import IsingLattice

        lattice = IsingLattice(size=(4, 4), J=1.0, h=10.0)
        result = lattice.run_metropolis(
            temperature=1.0,
            n_sweeps=500,
            n_warmup=500,
            key=jax.random.PRNGKey(0),
        )
        # Strong field should give high magnetization
        assert result["magnetization"] > 0.8


# ---------------------------------------------------------------------------
# Integrator convergence order tests
# ---------------------------------------------------------------------------


class TestIntegratorConvergenceOrder:
    """Verify convergence rates match theoretical orders."""

    @staticmethod
    def _run_sho_steps(integrator_fn, dt, n_steps):
        """Run SHO with given integrator and return final state."""

        def deriv(q, p, t, params):
            return p, -q

        q = jnp.array([1.0])
        p = jnp.array([0.0])
        t = 0.0
        for _ in range(n_steps):
            q, p, t = integrator_fn(deriv, q, p, t, dt, None)
        return q, p, t

    def test_rk4_fourth_order(self) -> None:
        """RK4 error should decrease by ~16x when dt halved."""
        from neurosim.classical.integrators import rk4

        dt1 = 0.1
        dt2 = 0.05
        T = 1.0
        n1 = int(T / dt1)
        n2 = int(T / dt2)

        q1, _p1, _ = self._run_sho_steps(rk4, dt1, n1)
        q2, _p2, _ = self._run_sho_steps(rk4, dt2, n2)

        exact_q = jnp.cos(T)
        err1 = float(jnp.abs(q1[0] - exact_q))
        err2 = float(jnp.abs(q2[0] - exact_q))

        # Ratio should be ~16 for 4th order
        if err2 > 1e-14:
            ratio = err1 / err2
            assert ratio > 10  # should be ~16 in theory

    def test_leapfrog_second_order(self) -> None:
        """Leapfrog error should decrease by ~4x when dt halved."""
        from neurosim.classical.integrators import leapfrog

        dt1 = 0.1
        dt2 = 0.05
        T = 1.0
        n1 = int(T / dt1)
        n2 = int(T / dt2)

        q1, _p1, _ = self._run_sho_steps(leapfrog, dt1, n1)
        q2, _p2, _ = self._run_sho_steps(leapfrog, dt2, n2)

        exact_q = jnp.cos(T)
        err1 = float(jnp.abs(q1[0] - exact_q))
        err2 = float(jnp.abs(q2[0] - exact_q))

        # Ratio should be ~4 for 2nd order
        if err2 > 1e-14:
            ratio = err1 / err2
            assert ratio > 3  # should be ~4 in theory
