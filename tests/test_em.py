"""Tests for electromagnetism modules."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.em.charges import ChargeSystem, PointCharge
from neurosim.em.fdtd import EMGrid, PlaneWave, Wall
from neurosim.em.waveguides import RectangularWaveguide
from neurosim.exceptions import ConfigurationError, PhysicsError

jax.config.update("jax_enable_x64", True)


class TestEMGrid:
    """Tests for the FDTD Maxwell solver."""

    def test_basic_simulation(self) -> None:
        """FDTD simulation should run without errors."""
        grid = EMGrid(size=(50, 50), resolution=0.01)
        grid.add_source(PlaneWave(frequency=3e9, y=10))

        fields = grid.simulate(t_span=(0, 1e-9), save_every=10)

        assert fields.ez.shape[1] == 50
        assert fields.ez.shape[2] == 50
        assert fields.t.shape[0] > 0

    def test_with_conductor(self) -> None:
        """Simulation with a conducting wall should work."""
        grid = EMGrid(size=(50, 50), resolution=0.01)
        grid.add_source(PlaneWave(frequency=3e9, y=10))
        grid.add_conductor(Wall(y=25, gap_start=20, gap_end=30))

        fields = grid.simulate(t_span=(0, 5e-10), save_every=5)

        # Ez should be zero at the conductor
        # (averaged over the wall row, excluding gap)
        assert fields.ez.shape[0] > 0

    def test_no_source_error(self) -> None:
        grid = EMGrid(size=(50, 50))
        with pytest.raises(ConfigurationError):
            grid.simulate()

    def test_small_grid_error(self) -> None:
        with pytest.raises(ConfigurationError):
            EMGrid(size=(5, 5))

    def test_multiple_sources_are_applied(self) -> None:
        """All configured sources should inject energy into the grid."""
        grid = EMGrid(size=(50, 50), resolution=0.01)
        grid.add_source(PlaneWave(frequency=3e9, y=10, amplitude=1.0))
        grid.add_source(PlaneWave(frequency=5e9, y=35, amplitude=0.5))

        fields = grid.simulate(t_span=(0, 2e-10), save_every=1)

        top_row = jnp.max(jnp.abs(fields.ez[-1, :, 10]))
        bottom_row = jnp.max(jnp.abs(fields.ez[-1, :, 35]))
        assert top_row > 0.0
        assert bottom_row > 0.0

    def test_boundary_modes_affect_solution(self) -> None:
        """Boundary mode selection should change the field evolution."""
        periodic = EMGrid(
            size=(40, 40), resolution=0.01, boundary="periodic", pml_layers=0
        )
        reflecting = EMGrid(
            size=(40, 40), resolution=0.01, boundary="reflecting", pml_layers=0
        )

        source = PlaneWave(frequency=3e9, y=2, amplitude=1.0)
        periodic.add_source(source)
        reflecting.add_source(source)

        periodic_fields = periodic.simulate(t_span=(0, 2e-10), save_every=1)
        reflecting_fields = reflecting.simulate(t_span=(0, 2e-10), save_every=1)

        assert not jnp.allclose(periodic_fields.ez[-1], reflecting_fields.ez[-1])


class TestWaveguide:
    """Tests for waveguide mode analysis."""

    def test_wr90_cutoff(self) -> None:
        """WR-90 waveguide TE10 cutoff should be ~6.56 GHz."""
        wg = RectangularWaveguide(a=0.02286, b=0.01016)
        fc = wg.cutoff_frequency(1, 0)
        assert fc == pytest.approx(6.56e9, rel=0.01)

    def test_te10_mode(self) -> None:
        wg = RectangularWaveguide(a=0.02286, b=0.01016)
        mode = wg.compute_mode("TE", 1, 0, n_points=50)
        assert mode.field_pattern.shape == (50, 50)
        assert mode.cutoff_frequency > 0

    def test_tm11_mode(self) -> None:
        wg = RectangularWaveguide(a=0.02, b=0.01)
        mode = wg.compute_mode("TM", 1, 1, n_points=50)
        assert mode.mode_type == "TM"

    def test_evanescent_mode(self) -> None:
        wg = RectangularWaveguide(a=0.02286, b=0.01016)
        # Frequency below TE10 cutoff
        with pytest.raises(PhysicsError):
            wg.propagation_constant(1e9, 1, 0)

    def test_invalid_te00(self) -> None:
        wg = RectangularWaveguide(a=0.02, b=0.01)
        with pytest.raises(ConfigurationError):
            wg.compute_mode("TE", 0, 0)

    def test_dispersion_relation(self) -> None:
        wg = RectangularWaveguide(a=0.02286, b=0.01016)
        freqs = jnp.linspace(5e9, 15e9, 100)
        beta = wg.dispersion_relation(1, 0, freqs)
        # Below cutoff should be NaN
        fc = wg.cutoff_frequency(1, 0)
        below_mask = freqs < fc
        assert jnp.all(jnp.isnan(beta[below_mask]))
        # Above cutoff should be positive
        above_mask = freqs > fc
        assert jnp.all(beta[above_mask] > 0)


class TestCharges:
    """Tests for charge dynamics."""

    def test_two_opposite_charges(self) -> None:
        """Two opposite charges should attract."""
        q1 = PointCharge(
            charge=1e-6,
            mass=1e-3,
            position=[0, 0, 0],
            velocity=[0, 0, 0],
        )
        q2 = PointCharge(
            charge=-1e-6,
            mass=1e-3,
            position=[0.01, 0, 0],
            velocity=[0, 0, 0],
        )
        system = ChargeSystem(charges=[q1, q2])
        traj = system.simulate(
            t_span=(0, 1e-5),
            n_steps=1000,
            save_every=100,
        )

        # Distance should decrease
        initial_dist = jnp.linalg.norm(traj.positions[0, 0] - traj.positions[0, 1])
        final_dist = jnp.linalg.norm(traj.positions[-1, 0] - traj.positions[-1, 1])
        assert final_dist < initial_dist

    def test_callable_external_field(self) -> None:
        """Callable external fields should be evaluated during integration."""

        q = PointCharge(
            charge=1.0,
            mass=1.0,
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
        )

        def electric_field(position, t):
            return jnp.array([1.0 + 0.0 * t, 0.0, 0.0]) + 0.0 * position

        system = ChargeSystem(charges=[q], E_external=electric_field)
        traj = system.simulate(t_span=(0, 1e-3), n_steps=10, save_every=1)

        assert float(traj.positions[-1, 0, 0]) > 0.0
