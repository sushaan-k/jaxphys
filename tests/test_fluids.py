"""Tests for fluid dynamics modules."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.exceptions import ConfigurationError
from neurosim.fluids.lbm import D2Q9, LBMGrid, Obstacle
from neurosim.fluids.navier_stokes import NavierStokesSolver

jax.config.update("jax_enable_x64", True)


class TestD2Q9:
    """Tests for the D2Q9 lattice constants."""

    def test_weights_sum_to_one(self) -> None:
        lattice = D2Q9()
        assert jnp.sum(lattice.w) == pytest.approx(1.0, rel=1e-12)

    def test_velocity_count(self) -> None:
        lattice = D2Q9()
        assert lattice.c.shape == (9, 2)

    def test_opposite_directions(self) -> None:
        """Each direction's opposite should reverse the velocity."""
        lattice = D2Q9()
        for i, opp_i in enumerate(lattice.opposite):
            assert jnp.allclose(lattice.c[i] + lattice.c[opp_i], 0), (
                f"Direction {i} and its opposite {opp_i} don't sum to zero"
            )


class TestLBMGrid:
    """Tests for the Lattice Boltzmann solver."""

    def test_basic_simulation(self) -> None:
        """LBM simulation should run without errors."""
        grid = LBMGrid(size=(40, 20), viscosity=0.05)
        result = grid.simulate(n_steps=100, u_inlet=0.04, save_every=50)

        assert result.rho.shape[1] == 40
        assert result.rho.shape[2] == 20
        assert result.t.shape[0] > 0

    def test_mass_conservation(self) -> None:
        """Total mass should be approximately conserved."""
        grid = LBMGrid(size=(40, 20), viscosity=0.05)
        result = grid.simulate(n_steps=200, u_inlet=0.02, save_every=100)

        initial_mass = float(jnp.sum(result.rho[0]))
        final_mass = float(jnp.sum(result.rho[-1]))
        # Allow some tolerance due to open boundaries
        assert abs(final_mass - initial_mass) / initial_mass < 0.1

    def test_with_obstacle(self) -> None:
        """Simulation with an obstacle should run and perturb the flow."""
        nx, ny = 50, 30
        grid = LBMGrid(size=(nx, ny), viscosity=0.05)
        x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
        cylinder = ((x - 15) ** 2 + (y - 15) ** 2) < 5**2
        grid.add_obstacle(Obstacle(mask=cylinder))

        result = grid.simulate(n_steps=100, u_inlet=0.04, save_every=50)
        assert result.uy.shape[0] > 0

    def test_vorticity_computed(self) -> None:
        """Vorticity should be computed in the result."""
        grid = LBMGrid(size=(30, 20), viscosity=0.05)
        result = grid.simulate(n_steps=50, u_inlet=0.04, save_every=25)
        assert result.vorticity is not None
        assert result.vorticity.shape == result.ux.shape

    def test_speed_property(self) -> None:
        """FluidHistory.speed should return the magnitude."""
        grid = LBMGrid(size=(30, 20), viscosity=0.05)
        result = grid.simulate(n_steps=50, u_inlet=0.04, save_every=25)
        speed = result.speed
        expected = jnp.sqrt(result.ux**2 + result.uy**2)
        assert jnp.allclose(speed, expected)

    def test_small_grid_error(self) -> None:
        with pytest.raises(ConfigurationError):
            LBMGrid(size=(2, 2))

    def test_high_velocity_error(self) -> None:
        grid = LBMGrid(size=(20, 20), viscosity=0.05)
        with pytest.raises(ConfigurationError):
            grid.simulate(n_steps=10, u_inlet=0.5)

    def test_low_viscosity_error(self) -> None:
        """Viscosity that makes tau <= 0.5 should raise error."""
        with pytest.raises(Exception):
            LBMGrid(size=(20, 20), viscosity=-0.1)

    def test_obstacle_shape_mismatch(self) -> None:
        grid = LBMGrid(size=(20, 20), viscosity=0.05)
        bad_mask = jnp.zeros((10, 10), dtype=bool)
        with pytest.raises(ConfigurationError):
            grid.add_obstacle(Obstacle(mask=bad_mask))


class TestNavierStokesSolver:
    """Tests for the vorticity-streamfunction solver."""

    def test_basic_simulation(self) -> None:
        """Navier-Stokes solver should run without errors."""
        solver = NavierStokesSolver(size=(20, 20), viscosity=0.01)
        result = solver.simulate(
            n_steps=100, dt=0.01, lid_velocity=0.5, save_every=50
        )

        assert result.ux.shape[1] == 20
        assert result.ux.shape[2] == 20
        assert result.t.shape[0] > 0

    def test_lid_velocity_on_boundary(self) -> None:
        """Top boundary should have the lid velocity."""
        solver = NavierStokesSolver(size=(20, 20), viscosity=0.01)
        result = solver.simulate(
            n_steps=100, dt=0.01, lid_velocity=1.0, save_every=50
        )
        # Top wall ux should be lid_velocity
        top_ux = result.ux[-1, :, -1]
        assert jnp.allclose(top_ux, 1.0)

    def test_no_slip_walls(self) -> None:
        """Bottom and side walls should have zero velocity."""
        solver = NavierStokesSolver(size=(20, 20), viscosity=0.01)
        result = solver.simulate(
            n_steps=100, dt=0.01, lid_velocity=1.0, save_every=50
        )
        # Bottom wall
        assert jnp.allclose(result.ux[-1, :, 0], 0.0)
        # Left wall vy
        assert jnp.allclose(result.uy[-1, 0, :], 0.0)

    def test_cfl_error(self) -> None:
        solver = NavierStokesSolver(size=(20, 20), viscosity=0.01)
        with pytest.raises(ConfigurationError):
            solver.simulate(n_steps=10, dt=10.0, lid_velocity=1.0)

    def test_small_grid_error(self) -> None:
        with pytest.raises(ConfigurationError):
            NavierStokesSolver(size=(2, 2))
