"""3D Finite-Difference Time-Domain (FDTD) Maxwell solver.

Extends the 2D Yee algorithm to the full 3D case, solving Maxwell's
curl equations for all six field components (Ex, Ey, Ez, Hx, Hy, Hz).

The 3D Yee update equations:

    H^{n+1/2} = H^{n-1/2} - (dt/mu0) * curl(E^n)
    E^{n+1}   = E^{n}     + (dt/eps0) * curl(H^{n+1/2})

The CFL stability condition in 3D requires:
    dt <= dx / (c * sqrt(3))

References:
    - Yee, K.S. "Numerical solution of initial boundary value problems
      involving Maxwell's equations in isotropic media" (1966)
    - Taflove & Hagness. "Computational Electrodynamics" (2005), Ch. 3-5
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
from neurosim.state import EMFieldHistory3D

logger = logging.getLogger(__name__)

# Physical constants (SI)
C0 = 299792458.0  # speed of light (m/s)
MU0 = 4.0e-7 * jnp.pi  # permeability of free space (H/m)
EPS0 = 1.0 / (MU0 * C0**2)  # permittivity of free space (F/m)


@dataclass(frozen=True)
class PointSource3D:
    """Point source specification for 3D FDTD.

    Attributes:
        frequency: Source frequency in Hz.
        position: Source location as (ix, iy, iz) grid indices.
        amplitude: Peak electric field amplitude (V/m).
        polarization: Which E-field component to drive.
    """

    frequency: float
    position: tuple[int, int, int]
    amplitude: float = 1.0
    polarization: Literal["x", "y", "z"] = "z"


@dataclass(frozen=True)
class DielectricRegion:
    """Region with custom permittivity (for dispersive media).

    Attributes:
        mask: Boolean array, shape (nx, ny, nz). True where material is.
        epsilon_r: Relative permittivity of the region.
    """

    mask: Array
    epsilon_r: float = 1.0


class EMGrid3D:
    """3D electromagnetic FDTD simulation grid.

    Implements the full 3D Yee algorithm with all six field components.
    Supports PML absorbing boundaries, point sources, and dielectric
    materials.

    Example:
        >>> grid = EMGrid3D(size=(60, 60, 60), resolution=0.01)
        >>> grid.add_source(PointSource3D(frequency=3e9, position=(30, 30, 10)))
        >>> fields = grid.simulate(t_span=(0, 5e-9), save_every=20)

    Args:
        size: Grid dimensions (nx, ny, nz) in cells.
        resolution: Cell size in meters (uniform in all directions).
        boundary: Boundary condition type.
        pml_layers: Number of PML absorbing layers.
    """

    def __init__(
        self,
        size: tuple[int, int, int] = (60, 60, 60),
        resolution: float = 0.01,
        boundary: Literal["absorbing", "periodic"] = "absorbing",
        pml_layers: int = 8,
    ) -> None:
        self._nx, self._ny, self._nz = size
        self._dx = resolution
        self._boundary = boundary
        self._pml_layers = pml_layers
        self._sources: list[PointSource3D] = []
        self._materials: list[DielectricRegion] = []

        if min(size) < 8:
            raise ConfigurationError(
                f"Grid must be at least 8x8x8, got {size}"
            )

    @property
    def size(self) -> tuple[int, int, int]:
        """Grid dimensions (nx, ny, nz)."""
        return (self._nx, self._ny, self._nz)

    def add_source(self, source: PointSource3D) -> None:
        """Add a point source to the grid."""
        ix, iy, iz = source.position
        if not (0 <= ix < self._nx and 0 <= iy < self._ny and 0 <= iz < self._nz):
            raise ConfigurationError(
                f"Source position {source.position} out of grid bounds "
                f"({self._nx}, {self._ny}, {self._nz})"
            )
        self._sources.append(source)

    def add_material(self, material: DielectricRegion) -> None:
        """Add a dielectric region to the grid."""
        expected = (self._nx, self._ny, self._nz)
        if material.mask.shape != expected:
            raise ConfigurationError(
                f"Material mask shape {material.mask.shape} doesn't match "
                f"grid size {expected}"
            )
        self._materials.append(material)

    def _build_epsilon_r(self) -> Array:
        """Build relative permittivity grid from materials."""
        eps_r = jnp.ones((self._nx, self._ny, self._nz))
        for mat in self._materials:
            eps_r = jnp.where(mat.mask, mat.epsilon_r, eps_r)
        return eps_r

    def _build_pml_sigma_3d(self) -> tuple[Array, Array, Array]:
        """Build 3D PML conductivity profiles."""
        nx, ny, nz = self._nx, self._ny, self._nz
        n_pml = self._pml_layers
        dx = self._dx

        if n_pml == 0:
            z = jnp.zeros((nx, ny, nz))
            return z, z, z

        sigma_max = 0.8 * 4.0 / (dx * 1.0)

        def build_1d(n: int) -> Array:
            s = jnp.zeros(n)
            for i in range(n_pml):
                val = sigma_max * ((n_pml - i) / n_pml) ** 3
                s = s.at[i].set(val)
                s = s.at[n - 1 - i].set(val)
            return s

        sx = build_1d(nx)
        sy = build_1d(ny)
        sz = build_1d(nz)

        sigma_x = jnp.broadcast_to(
            sx[:, jnp.newaxis, jnp.newaxis], (nx, ny, nz)
        )
        sigma_y = jnp.broadcast_to(
            sy[jnp.newaxis, :, jnp.newaxis], (nx, ny, nz)
        )
        sigma_z = jnp.broadcast_to(
            sz[jnp.newaxis, jnp.newaxis, :], (nx, ny, nz)
        )

        return sigma_x, sigma_y, sigma_z

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 5e-9),
        dt: float | None = None,
        save_every: int = 20,
    ) -> EMFieldHistory3D:
        """Run the 3D FDTD simulation.

        Args:
            t_span: (t_start, t_end) in seconds.
            dt: Time step. If None, computed from CFL condition.
            save_every: Save field snapshot every N steps.

        Returns:
            EMFieldHistory3D with time-series of all six field components.
        """
        dx = self._dx
        nx, ny, nz = self._nx, self._ny, self._nz
        boundary = self._boundary

        # CFL condition: dt <= dx / (c * sqrt(3)) for 3D
        if dt is None:
            dt = float(0.99 * dx / (C0 * float(jnp.sqrt(3.0))))
        dt = float(dt)

        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        if not self._sources:
            raise ConfigurationError(
                "At least one source must be added before simulation"
            )

        logger.info(
            "Starting 3D FDTD: grid=%dx%dx%d, dt=%.2e, n_steps=%d",
            nx, ny, nz, dt, n_steps,
        )

        # Initialize fields
        ex = jnp.zeros((nx, ny, nz))
        ey = jnp.zeros((nx, ny, nz))
        ez = jnp.zeros((nx, ny, nz))
        hx = jnp.zeros((nx, ny, nz))
        hy = jnp.zeros((nx, ny, nz))
        hz = jnp.zeros((nx, ny, nz))

        eps_r = self._build_epsilon_r()
        sigma_x, sigma_y, sigma_z = self._build_pml_sigma_3d()
        sigma_avg = (sigma_x + sigma_y + sigma_z) / 3.0

        dt_mu_dx = dt / (float(MU0) * dx)
        dt_eps_dx = dt / (float(EPS0) * dx)

        # Source parameters
        source_omegas = [
            2.0 * jnp.pi * s.frequency for s in self._sources
        ]

        grid_x = jnp.arange(nx) * dx
        grid_y = jnp.arange(ny) * dx
        grid_z = jnp.arange(nz) * dx

        def add_sources(
            ex_f: Array, ey_f: Array, ez_f: Array, t_val: float
        ) -> tuple[Array, Array, Array]:
            for source, omega in zip(self._sources, source_omegas, strict=True):
                ix, iy, iz = source.position
                val = source.amplitude * jnp.sin(omega * t_val)
                if source.polarization == "z":
                    ez_f = ez_f.at[ix, iy, iz].add(val)
                elif source.polarization == "x":
                    ex_f = ex_f.at[ix, iy, iz].add(val)
                else:
                    ey_f = ey_f.at[ix, iy, iz].add(val)
            return ex_f, ey_f, ez_f

        if boundary == "periodic":

            def curl_e_periodic(
                ex_f: Array, ey_f: Array, ez_f: Array
            ) -> tuple[Array, Array, Array]:
                # curl_x = dEz/dy - dEy/dz
                curl_x = (
                    (jnp.roll(ez_f, -1, axis=1) - ez_f)
                    - (jnp.roll(ey_f, -1, axis=2) - ey_f)
                )
                # curl_y = dEx/dz - dEz/dx
                curl_y = (
                    (jnp.roll(ex_f, -1, axis=2) - ex_f)
                    - (jnp.roll(ez_f, -1, axis=0) - ez_f)
                )
                # curl_z = dEy/dx - dEx/dy
                curl_z = (
                    (jnp.roll(ey_f, -1, axis=0) - ey_f)
                    - (jnp.roll(ex_f, -1, axis=1) - ex_f)
                )
                return curl_x, curl_y, curl_z

            def curl_h_periodic(
                hx_f: Array, hy_f: Array, hz_f: Array
            ) -> tuple[Array, Array, Array]:
                curl_x = (
                    (hz_f - jnp.roll(hz_f, 1, axis=1))
                    - (hy_f - jnp.roll(hy_f, 1, axis=2))
                )
                curl_y = (
                    (hx_f - jnp.roll(hx_f, 1, axis=2))
                    - (hz_f - jnp.roll(hz_f, 1, axis=0))
                )
                curl_z = (
                    (hy_f - jnp.roll(hy_f, 1, axis=0))
                    - (hx_f - jnp.roll(hx_f, 1, axis=1))
                )
                return curl_x, curl_y, curl_z

        def step(
            carry: tuple[Array, Array, Array, Array, Array, Array, int],
            _: None,
        ) -> tuple[
            tuple[Array, Array, Array, Array, Array, Array, int],
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c, step_idx = carry
            t_c = t_start + step_idx * dt

            if boundary == "periodic":
                curl_ex, curl_ey, curl_ez = curl_e_periodic(ex_c, ey_c, ez_c)

                hx_new = hx_c - dt_mu_dx * curl_ex
                hy_new = hy_c - dt_mu_dx * curl_ey
                hz_new = hz_c - dt_mu_dx * curl_ez

                curl_hx, curl_hy, curl_hz = curl_h_periodic(hx_new, hy_new, hz_new)

                ex_new = ex_c + (dt_eps_dx / eps_r) * curl_hx
                ey_new = ey_c + (dt_eps_dx / eps_r) * curl_hy
                ez_new = ez_c + (dt_eps_dx / eps_r) * curl_hz
            else:
                # H update: interior finite differences
                # Hx: dEz/dy - dEy/dz
                hx_new = hx_c.at[:, :-1, :-1].set(
                    hx_c[:, :-1, :-1]
                    - dt_mu_dx * (
                        (ez_c[:, 1:, :-1] - ez_c[:, :-1, :-1])
                        - (ey_c[:, :-1, 1:] - ey_c[:, :-1, :-1])
                    )
                )
                # Hy: dEx/dz - dEz/dx
                hy_new = hy_c.at[:-1, :, :-1].set(
                    hy_c[:-1, :, :-1]
                    - dt_mu_dx * (
                        (ex_c[:-1, :, 1:] - ex_c[:-1, :, :-1])
                        - (ez_c[1:, :, :-1] - ez_c[:-1, :, :-1])
                    )
                )
                # Hz: dEy/dx - dEx/dy
                hz_new = hz_c.at[:-1, :-1, :].set(
                    hz_c[:-1, :-1, :]
                    - dt_mu_dx * (
                        (ey_c[1:, :-1, :] - ey_c[:-1, :-1, :])
                        - (ex_c[:-1, 1:, :] - ex_c[:-1, :-1, :])
                    )
                )

                # E update using curl of H
                # Ex: dHz/dy - dHy/dz
                ex_new = ex_c.at[:, 1:, 1:].set(
                    ex_c[:, 1:, 1:]
                    + (dt_eps_dx / eps_r[:, 1:, 1:]) * (
                        (hz_new[:, 1:, 1:] - hz_new[:, :-1, 1:])
                        - (hy_new[:, 1:, 1:] - hy_new[:, 1:, :-1])
                    )
                )
                # Ey: dHx/dz - dHz/dx
                ey_new = ey_c.at[1:, :, 1:].set(
                    ey_c[1:, :, 1:]
                    + (dt_eps_dx / eps_r[1:, :, 1:]) * (
                        (hx_new[1:, :, 1:] - hx_new[1:, :, :-1])
                        - (hz_new[1:, :, 1:] - hz_new[:-1, :, 1:])
                    )
                )
                # Ez: dHy/dx - dHx/dy
                ez_new = ez_c.at[1:, 1:, :].set(
                    ez_c[1:, 1:, :]
                    + (dt_eps_dx / eps_r[1:, 1:, :]) * (
                        (hy_new[1:, 1:, :] - hy_new[:-1, 1:, :])
                        - (hx_new[1:, 1:, :] - hx_new[1:, :-1, :])
                    )
                )

            # PML damping (absorbing boundary)
            if boundary == "absorbing":
                damping = jnp.exp(-sigma_avg * dt / float(EPS0))
                ex_new = ex_new * damping
                ey_new = ey_new * damping
                ez_new = ez_new * damping

            # Add sources
            ex_new, ey_new, ez_new = add_sources(ex_new, ey_new, ez_new, t_c)

            return (
                ex_new, ey_new, ez_new, hx_new, hy_new, hz_new, step_idx + 1
            ), (
                ex_new, ey_new, ez_new, hx_new, hy_new, hz_new,
                jnp.asarray(t_c + dt),
            )

        init = (ex, ey, ez, hx, hy, hz, 0)
        _, (
            ex_all, ey_all, ez_all,
            hx_all, hy_all, hz_all,
            t_all,
        ) = jax.lax.scan(step, init, None, length=n_steps)

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            ex_all = ex_all[indices]
            ey_all = ey_all[indices]
            ez_all = ez_all[indices]
            hx_all = hx_all[indices]
            hy_all = hy_all[indices]
            hz_all = hz_all[indices]
            t_all = t_all[indices]

        return EMFieldHistory3D(
            t=t_all,
            ex=ex_all,
            ey=ey_all,
            ez=ez_all,
            hx=hx_all,
            hy=hy_all,
            hz=hz_all,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
        )
