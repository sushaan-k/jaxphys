"""Finite-Difference Time-Domain (FDTD) Maxwell solver.

Implements the Yee algorithm for solving Maxwell's equations on a
staggered grid. Supports TM polarization (Ez, Hx, Hy) in 2D with
perfectly matched layer (PML) absorbing boundaries.

The Yee update equations (TM mode, 2D):

    Hx^{n+1/2} = Hx^{n-1/2} - (dt/mu0) * dEz/dy
    Hy^{n+1/2} = Hy^{n-1/2} + (dt/mu0) * dEz/dx
    Ez^{n+1}   = Ez^{n} + (dt/eps0) * (dHy/dx - dHx/dy)

The CFL stability condition requires:
    dt <= dx / (c * sqrt(2))  for 2D

References:
    - Yee, K.S. "Numerical solution of initial boundary value problems
      involving Maxwell's equations in isotropic media" (1966)
    - Taflove & Hagness. "Computational Electrodynamics" (2005)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import EMConfig
from neurosim.exceptions import ConfigurationError
from neurosim.state import EMFieldHistory

logger = logging.getLogger(__name__)

# Physical constants (SI)
C0 = 299792458.0  # speed of light (m/s)
MU0 = 4.0e-7 * jnp.pi  # permeability of free space (H/m)
EPS0 = 1.0 / (MU0 * C0**2)  # permittivity of free space (F/m)


@dataclass(frozen=True)
class PlaneWave:
    """Plane wave source specification.

    Attributes:
        frequency: Wave frequency in Hz.
        y: Source y-position (grid index).
        amplitude: Peak electric field amplitude (V/m).
    """

    frequency: float
    y: int
    amplitude: float = 1.0


@dataclass(frozen=True)
class Wall:
    """Conducting wall with optional gap (slit).

    Attributes:
        y: Wall y-position (grid index).
        gap_start: Start of gap (grid index). None for solid wall.
        gap_end: End of gap (grid index). None for solid wall.
    """

    y: int
    gap_start: int | None = None
    gap_end: int | None = None


class EMGrid:
    """2D electromagnetic FDTD simulation grid.

    Implements TM polarization (Ez, Hx, Hy) with PML absorbing
    boundaries. Sources and conductors can be added before simulation.

    Example:
        >>> grid = EMGrid(size=(200, 200), resolution=0.01)
        >>> grid.add_source(PlaneWave(frequency=3e9, y=20))
        >>> grid.add_conductor(Wall(y=100, gap_start=90, gap_end=110))
        >>> fields = grid.simulate(t_span=(0, 1e-8), dt=1e-11)

    Args:
        size: Grid dimensions (nx, ny) in cells.
        resolution: Cell size in meters.
        boundary: Boundary condition type.
        pml_layers: Number of PML layers.
    """

    def __init__(
        self,
        size: tuple[int, int] = (200, 200),
        resolution: float = 0.01,
        boundary: Literal["absorbing", "periodic", "reflecting"] = "absorbing",
        pml_layers: int = 10,
    ) -> None:
        self._nx, self._ny = size
        self._dx = resolution
        self._config = EMConfig(
            resolution=resolution,
            courant_number=0.5,
            boundary=boundary,
            pml_layers=pml_layers,
        )
        self._sources: list[PlaneWave] = []
        self._conductors: list[Wall] = []

        # Validate grid size
        if self._nx < 10 or self._ny < 10:
            raise ConfigurationError(f"Grid must be at least 10x10, got {size}")

    @property
    def size(self) -> tuple[int, int]:
        """Grid dimensions (nx, ny)."""
        return (self._nx, self._ny)

    def add_source(self, source: PlaneWave) -> None:
        """Add a plane wave source to the grid.

        Args:
            source: PlaneWave specification.
        """
        if source.y < 0 or source.y >= self._ny:
            raise ConfigurationError(
                f"Source y={source.y} out of grid bounds [0, {self._ny})"
            )
        self._sources.append(source)

    def add_conductor(self, wall: Wall) -> None:
        """Add a conducting wall (with optional slit) to the grid.

        Args:
            wall: Wall specification.
        """
        if wall.y < 0 or wall.y >= self._ny:
            raise ConfigurationError(
                f"Wall y={wall.y} out of grid bounds [0, {self._ny})"
            )
        self._conductors.append(wall)

    def _build_conductor_mask(self) -> Array:
        """Build a boolean mask for conducting regions.

        Returns:
            Boolean array, shape (nx, ny). True where Ez is forced to 0.
        """
        mask = jnp.zeros((self._nx, self._ny), dtype=bool)
        for wall in self._conductors:
            row = jnp.ones(self._nx, dtype=bool)
            if wall.gap_start is not None and wall.gap_end is not None:
                gap = jnp.arange(self._nx)
                row = (gap < wall.gap_start) | (gap >= wall.gap_end)
            mask = mask.at[:, wall.y].set(row)
        return mask

    def _build_pml_sigma(self) -> tuple[Array, Array]:
        """Build PML conductivity profiles for absorbing boundaries.

        Uses a polynomial grading for the PML conductivity:
            sigma(d) = sigma_max * (d / thickness)^3

        Returns:
            Tuple of (sigma_x, sigma_y), each shape (nx, ny).
        """
        n_pml = self._config.pml_layers
        if n_pml == 0:
            zeros = jnp.zeros((self._nx, self._ny))
            return zeros, zeros

        sigma_max = 0.8 * (3 + 1) / (self._dx * jnp.sqrt(1.0))

        # x-direction PML
        sigma_x = jnp.zeros(self._nx)
        for i in range(n_pml):
            val = sigma_max * ((n_pml - i) / n_pml) ** 3
            sigma_x = sigma_x.at[i].set(val)
            sigma_x = sigma_x.at[self._nx - 1 - i].set(val)
        sigma_x = jnp.broadcast_to(sigma_x[:, jnp.newaxis], (self._nx, self._ny))

        # y-direction PML
        sigma_y = jnp.zeros(self._ny)
        for i in range(n_pml):
            val = sigma_max * ((n_pml - i) / n_pml) ** 3
            sigma_y = sigma_y.at[i].set(val)
            sigma_y = sigma_y.at[self._ny - 1 - i].set(val)
        sigma_y = jnp.broadcast_to(sigma_y[jnp.newaxis, :], (self._nx, self._ny))

        return sigma_x, sigma_y

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 1e-8),
        dt: float | None = None,
        save_every: int = 10,
    ) -> EMFieldHistory:
        """Run the FDTD simulation.

        Args:
            t_span: (t_start, t_end) in seconds.
            dt: Time step. If None, computed from CFL condition.
            save_every: Save field snapshot every N steps.

        Returns:
            EMFieldHistory with time-series of field snapshots.
        """
        dx = self._dx
        nx, ny = self._nx, self._ny
        boundary = self._config.boundary

        # CFL condition: dt <= dx / (c * sqrt(2))
        dt = float(0.99 * dx / (C0 * float(jnp.sqrt(2.0)))) if dt is None else float(dt)

        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        if not self._sources:
            raise ConfigurationError(
                "At least one source must be added before simulation"
            )

        logger.info(
            "Starting FDTD simulation: grid=%dx%d, dt=%.2e, n_steps=%d",
            nx,
            ny,
            float(dt),
            n_steps,
        )

        # Initialize fields
        ez = jnp.zeros((nx, ny))
        hx = jnp.zeros((nx, ny))
        hy = jnp.zeros((nx, ny))

        # Build conductor mask and PML
        conductor_mask = self._build_conductor_mask()
        sigma_x, sigma_y = self._build_pml_sigma()

        # PML damping profile
        sigma_avg = 0.5 * (sigma_x + sigma_y)

        # Source parameters
        source_omegas = [2.0 * jnp.pi * source.frequency for source in self._sources]

        # Spatial coordinates for output
        grid_x = jnp.arange(nx) * dx
        grid_y = jnp.arange(ny) * dx

        # Precompute PML coefficient slices for H and E updates
        # Use simple dt/(mu*dx) and dt/(eps*dx) without PML for now,
        # and apply PML via damping on the fields.
        dt_over_mu_dx = dt / (float(MU0) * dx)
        dt_over_eps_dx = dt / (float(EPS0) * dx)

        def add_sources(ez_field: Array, t_value: float) -> Array:
            """Apply all soft sources to the Ez field."""
            for source, omega in zip(self._sources, source_omegas, strict=True):
                source_val = source.amplitude * jnp.sin(omega * t_value)
                ez_field = ez_field.at[:, source.y].add(source_val)
            return ez_field

        def apply_reflecting_boundaries(
            ez_field: Array, hx_field: Array, hy_field: Array
        ) -> tuple[Array, Array, Array]:
            """Approximate a reflecting boundary by mirroring edge values."""
            ez_field = ez_field.at[0, :].set(ez_field[1, :])
            ez_field = ez_field.at[-1, :].set(ez_field[-2, :])
            ez_field = ez_field.at[:, 0].set(ez_field[:, 1])
            ez_field = ez_field.at[:, -1].set(ez_field[:, -2])

            hx_field = hx_field.at[0, :].set(hx_field[1, :])
            hx_field = hx_field.at[-1, :].set(hx_field[-2, :])
            hx_field = hx_field.at[:, 0].set(hx_field[:, 1])
            hx_field = hx_field.at[:, -1].set(hx_field[:, -2])

            hy_field = hy_field.at[0, :].set(hy_field[1, :])
            hy_field = hy_field.at[-1, :].set(hy_field[-2, :])
            hy_field = hy_field.at[:, 0].set(hy_field[:, 1])
            hy_field = hy_field.at[:, -1].set(hy_field[:, -2])
            return ez_field, hx_field, hy_field

        def apply_periodic_boundaries(
            ez_field: Array, hx_field: Array, hy_field: Array
        ) -> tuple[Array, Array, Array]:
            """Wrap the field values around the domain edges."""
            ez_field = ez_field.at[0, :].set(ez_field[-2, :])
            ez_field = ez_field.at[-1, :].set(ez_field[1, :])
            ez_field = ez_field.at[:, 0].set(ez_field[:, -2])
            ez_field = ez_field.at[:, -1].set(ez_field[:, 1])

            hx_field = hx_field.at[0, :].set(hx_field[-2, :])
            hx_field = hx_field.at[-1, :].set(hx_field[1, :])
            hx_field = hx_field.at[:, 0].set(hx_field[:, -2])
            hx_field = hx_field.at[:, -1].set(hx_field[:, 1])

            hy_field = hy_field.at[0, :].set(hy_field[-2, :])
            hy_field = hy_field.at[-1, :].set(hy_field[1, :])
            hy_field = hy_field.at[:, 0].set(hy_field[:, -2])
            hy_field = hy_field.at[:, -1].set(hy_field[:, 1])
            return ez_field, hx_field, hy_field

        def step(
            carry: tuple[Array, Array, Array, int],
            _: None,
        ) -> tuple[
            tuple[Array, Array, Array, int],
            tuple[Array, Array, Array, Array],
        ]:
            ez_c, hx_c, hy_c, step_idx = carry
            t_c = t_start + step_idx * dt

            if boundary == "periodic":
                # Periodic curls wrap around the domain.
                hx_new = hx_c - dt_over_mu_dx * (jnp.roll(ez_c, -1, axis=1) - ez_c)
                hy_new = hy_c + dt_over_mu_dx * (jnp.roll(ez_c, -1, axis=0) - ez_c)

                dhy_dx = hy_new - jnp.roll(hy_new, 1, axis=0)
                dhx_dy = hx_new - jnp.roll(hx_new, 1, axis=1)
                ez_new = ez_c + dt_over_eps_dx * (dhy_dx - dhx_dy)
            else:
                # Update Hx: Hx -= dt/(mu*dx) * (Ez[i,j+1] - Ez[i,j])
                # Interior points only; boundary stays zero unless reflected.
                hx_new = hx_c.at[:, :-1].set(
                    hx_c[:, :-1] - dt_over_mu_dx * (ez_c[:, 1:] - ez_c[:, :-1])
                )

                # Update Hy: Hy += dt/(mu*dx) * (Ez[i+1,j] - Ez[i,j])
                hy_new = hy_c.at[:-1, :].set(
                    hy_c[:-1, :] + dt_over_mu_dx * (ez_c[1:, :] - ez_c[:-1, :])
                )

                # Update Ez: Ez += dt/(eps*dx) * (dHy/dx - dHx/dy)
                dhy_dx = hy_new[1:, :] - hy_new[:-1, :]
                dhx_dy = hx_new[:, 1:] - hx_new[:, :-1]

                ez_new = ez_c.at[1:, 1:].set(
                    ez_c[1:, 1:] + dt_over_eps_dx * (dhy_dx[:, 1:] - dhx_dy[1:, :])
                )

            if boundary == "absorbing":
                # Apply PML damping only for absorbing boundaries.
                ez_new = ez_new * jnp.exp(-sigma_avg * dt / float(EPS0))
            elif boundary == "reflecting":
                ez_new, hx_new, hy_new = apply_reflecting_boundaries(
                    ez_new, hx_new, hy_new
                )
            elif boundary == "periodic":
                ez_new, hx_new, hy_new = apply_periodic_boundaries(
                    ez_new, hx_new, hy_new
                )

            ez_new = add_sources(ez_new, t_c)

            # Apply conductor boundary (Ez = 0 in conductors)
            ez_new = jnp.where(conductor_mask, 0.0, ez_new)

            return (ez_new, hx_new, hy_new, step_idx + 1), (
                ez_new,
                hx_new,
                hy_new,
                jnp.asarray(t_c + dt),
            )

        init = (ez, hx, hy, 0)
        _, (ez_all, hx_all, hy_all, t_all) = jax.lax.scan(
            step, init, None, length=n_steps
        )

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            ez_all = ez_all[indices]
            hx_all = hx_all[indices]
            hy_all = hy_all[indices]
            t_all = t_all[indices]

        return EMFieldHistory(
            t=t_all,
            ez=ez_all,
            hx=hx_all,
            hy=hy_all,
            grid_x=grid_x,
            grid_y=grid_y,
        )
