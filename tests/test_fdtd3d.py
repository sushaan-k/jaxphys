"""Tests for 3D FDTD electromagnetic solver."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.em.fdtd3d import DielectricRegion, EMGrid3D, PointSource3D
from neurosim.exceptions import ConfigurationError

jax.config.update("jax_enable_x64", True)


class TestEMGrid3D:
    """Tests for the 3D FDTD Maxwell solver."""

    def test_basic_simulation(self) -> None:
        """3D FDTD should run without errors."""
        grid = EMGrid3D(size=(20, 20, 20), resolution=0.01)
        grid.add_source(PointSource3D(frequency=3e9, position=(10, 10, 10)))

        fields = grid.simulate(t_span=(0, 5e-10), save_every=10)

        assert fields.ez.shape[1] == 20
        assert fields.ez.shape[2] == 20
        assert fields.ez.shape[3] == 20
        assert fields.t.shape[0] > 0

    def test_source_excites_field(self) -> None:
        """Source should inject energy into the grid."""
        grid = EMGrid3D(size=(20, 20, 20), resolution=0.01)
        grid.add_source(
            PointSource3D(frequency=3e9, position=(10, 10, 10), amplitude=1.0)
        )

        fields = grid.simulate(t_span=(0, 3e-10), save_every=5)

        max_ez = float(jnp.max(jnp.abs(fields.ez[-1])))
        assert max_ez > 0.0

    def test_all_six_components(self) -> None:
        """All six field components should be present."""
        grid = EMGrid3D(size=(16, 16, 16), resolution=0.01)
        grid.add_source(PointSource3D(frequency=5e9, position=(8, 8, 8)))

        fields = grid.simulate(t_span=(0, 2e-10), save_every=5)

        assert fields.ex.shape == fields.ey.shape == fields.ez.shape
        assert fields.hx.shape == fields.hy.shape == fields.hz.shape
        assert fields.ex.shape[1:] == (16, 16, 16)

    def test_polarization_x(self) -> None:
        """X-polarized source should excite Ex component."""
        grid = EMGrid3D(size=(16, 16, 16), resolution=0.01)
        grid.add_source(
            PointSource3D(
                frequency=3e9, position=(8, 8, 8), polarization="x"
            )
        )

        fields = grid.simulate(t_span=(0, 2e-10), save_every=5)
        assert float(jnp.max(jnp.abs(fields.ex[-1]))) > 0.0

    def test_polarization_y(self) -> None:
        """Y-polarized source should excite Ey component."""
        grid = EMGrid3D(size=(16, 16, 16), resolution=0.01)
        grid.add_source(
            PointSource3D(
                frequency=3e9, position=(8, 8, 8), polarization="y"
            )
        )

        fields = grid.simulate(t_span=(0, 2e-10), save_every=5)
        assert float(jnp.max(jnp.abs(fields.ey[-1]))) > 0.0

    def test_dielectric_material(self) -> None:
        """Dielectric region should affect field propagation."""
        grid_free = EMGrid3D(size=(16, 16, 16), resolution=0.01)
        grid_free.add_source(PointSource3D(frequency=5e9, position=(8, 8, 8)))

        grid_dielectric = EMGrid3D(size=(16, 16, 16), resolution=0.01)
        grid_dielectric.add_source(PointSource3D(frequency=5e9, position=(8, 8, 8)))

        # Add a dielectric slab close to the source
        mask = jnp.zeros((16, 16, 16), dtype=bool)
        mask = mask.at[10:, :, :].set(True)
        grid_dielectric.add_material(DielectricRegion(mask=mask, epsilon_r=4.0))

        # Run long enough for the wave to reach the dielectric
        fields_free = grid_free.simulate(t_span=(0, 1e-9), save_every=20)
        fields_diel = grid_dielectric.simulate(t_span=(0, 1e-9), save_every=20)

        # Fields should differ due to dielectric
        assert not jnp.allclose(fields_free.ez[-1], fields_diel.ez[-1])

    def test_periodic_boundary(self) -> None:
        """Periodic boundary should produce different results than absorbing."""
        grid_abs = EMGrid3D(
            size=(16, 16, 16), resolution=0.01, boundary="absorbing"
        )
        grid_per = EMGrid3D(
            size=(16, 16, 16), resolution=0.01, boundary="periodic", pml_layers=0
        )

        source = PointSource3D(frequency=5e9, position=(8, 8, 8))
        grid_abs.add_source(source)
        grid_per.add_source(source)

        fields_abs = grid_abs.simulate(t_span=(0, 2e-10), save_every=5)
        fields_per = grid_per.simulate(t_span=(0, 2e-10), save_every=5)

        assert not jnp.allclose(fields_abs.ez[-1], fields_per.ez[-1])

    def test_no_source_error(self) -> None:
        grid = EMGrid3D(size=(16, 16, 16))
        with pytest.raises(ConfigurationError):
            grid.simulate()

    def test_small_grid_error(self) -> None:
        with pytest.raises(ConfigurationError):
            EMGrid3D(size=(4, 4, 4))

    def test_source_out_of_bounds(self) -> None:
        grid = EMGrid3D(size=(16, 16, 16))
        with pytest.raises(ConfigurationError):
            grid.add_source(PointSource3D(frequency=1e9, position=(20, 8, 8)))

    def test_material_shape_mismatch(self) -> None:
        grid = EMGrid3D(size=(16, 16, 16))
        bad_mask = jnp.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ConfigurationError):
            grid.add_material(DielectricRegion(mask=bad_mask))

    def test_grid_coordinates(self) -> None:
        """Grid coordinates should match resolution."""
        grid = EMGrid3D(size=(16, 16, 16), resolution=0.02)
        grid.add_source(PointSource3D(frequency=1e9, position=(8, 8, 8)))
        fields = grid.simulate(t_span=(0, 1e-10), save_every=5)
        assert fields.grid_x.shape == (16,)
        assert float(fields.grid_x[1] - fields.grid_x[0]) == pytest.approx(0.02)
