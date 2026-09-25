"""3D Finite-Difference Time-Domain (FDTD) Maxwell solver.

Extends the 2D Yee algorithm to the full 3D case, solving Maxwell's
curl equations for all six field components (Ex, Ey, Ez, Hx, Hy, Hz).

The 3D Yee update equations:

    H^{n+1/2} = H^{n-1/2} - (dt/mu0) * curl(E^n)
    E^{n+1}   = E^{n}     + (dt/eps) * curl(H^{n+1/2})

The CFL stability condition in 3D requires:
    dt <= dx / (c * sqrt(3))

Absorbing boundaries use the same Berenger split-field PML as the 2D
solver: every component is split into the two parts driven by its two
curl terms, and each part decays with the conductivity along the axis of
its derivative. The layer is backed by perfectly conducting walls.

References:
    - Yee, K.S. "Numerical solution of initial boundary value problems
      involving Maxwell's equations in isotropic media" (1966)
    - Berenger, J.-P. "A perfectly matched layer for the absorption of
      electromagnetic waves" (1994)
    - Taflove & Hagness. "Computational Electrodynamics" (2005), Ch. 3-7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxphys._rollout import strided_rollout
from jaxphys.em.fdtd import (
    C0,
    EPS0,
    MU0,
    _decay,
    _pml_loss,
    diff_backward,
    diff_forward,
    edge_mask,
)
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import EMFieldHistory3D

logger = logging.getLogger(__name__)

_POLARIZATION_INDEX = {"x": 0, "y": 1, "z": 2}


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
    """Region with a constant relative permittivity.

    Attributes:
        mask: Boolean array, shape (nx, ny, nz). True where material is.
        epsilon_r: Relative permittivity of the region.
    """

    mask: Array
    epsilon_r: float = 1.0


class EMGrid3D:
    """3D electromagnetic FDTD simulation grid.

    Implements the full 3D Yee algorithm with all six field components.
    Boundaries: ``"absorbing"`` (split-field PML of ``pml_layers`` cells
    backed by conducting walls) or ``"periodic"``. Supports soft point
    sources and dielectric regions (keep materials out of the PML, which is
    matched to vacuum).

    Example:
        >>> grid = EMGrid3D(size=(60, 60, 60), resolution=0.01)
        >>> grid.add_source(PointSource3D(frequency=3e9, position=(30, 30, 10)))
        >>> fields = grid.simulate(t_span=(0, 5e-9), save_every=20)

    Args:
        size: Grid dimensions (nx, ny, nz) in cells.
        resolution: Cell size in meters (uniform in all directions).
        boundary: Boundary condition type.
        pml_layers: PML thickness in cells (used when ``boundary`` is
            ``"absorbing"``).
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
            raise ConfigurationError(f"Grid must be at least 8x8x8, got {size}")

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

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 5e-9),
        dt: float | None = None,
        save_every: int = 20,
    ) -> EMFieldHistory3D:
        """Run the 3D FDTD simulation.

        Args:
            t_span: (t_start, t_end) in seconds.
            dt: Time step in seconds. Defaults to 0.99x the 3D CFL limit
                ``dx / (c * sqrt(3))``; larger values are rejected.
            save_every: Save field snapshot every N steps.

        Returns:
            EMFieldHistory3D with snapshots after steps 1, 1 + save_every,
            1 + 2*save_every, ... (``t`` holds the matching times).

        Raises:
            ConfigurationError: If no source was added, ``dt`` violates the
                CFL condition, or the time span holds less than one step.
        """
        dx = self._dx
        nx, ny, nz = self._nx, self._ny, self._nz

        if not self._sources:
            raise ConfigurationError(
                "At least one source must be added before simulation"
            )
        dt_max = dx / (C0 * float(np.sqrt(3.0)))
        dt = 0.99 * dt_max if dt is None else float(dt)
        if not 0 < dt <= dt_max:
            raise ConfigurationError(
                f"dt={dt:.3e} s violates the 3D CFL condition "
                f"0 < dt <= dx/(c*sqrt(3)) = {dt_max:.3e} s"
            )
        if save_every < 1:
            raise ConfigurationError(f"save_every must be >= 1, got {save_every}")

        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        if n_steps < 1:
            raise ConfigurationError(
                f"t_span {t_span} is shorter than one time step ({dt:.3e} s)"
            )

        logger.info(
            "Starting 3D FDTD: grid=%dx%dx%d, dt=%.2e, n_steps=%d",
            nx,
            ny,
            nz,
            dt,
            n_steps,
        )

        n_pml = self._pml_layers if self._boundary == "absorbing" else 0
        # (loss at integer nodes, loss at half nodes) for each axis, shaped
        # to broadcast along that axis.
        losses = []
        for axis, n in enumerate((nx, ny, nz)):
            shape = [1, 1, 1]
            shape[axis] = n
            node, half = _pml_loss(n, n_pml, dx, dt)
            losses.append((node.reshape(shape), half.reshape(shape)))

        sources = self._sources
        fields = _fdtd3d_rollout(
            self._build_epsilon_r(),
            tuple(losses),
            jnp.array([src.position for src in sources]).T,
            jnp.array([_POLARIZATION_INDEX[src.polarization] for src in sources]),
            jnp.array([2.0 * np.pi * src.frequency for src in sources]),
            jnp.array([src.amplitude for src in sources], dtype=jnp.float64),
            jnp.asarray(t_start, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
            dt / (MU0 * dx),
            dt / (EPS0 * dx),
            periodic=self._boundary == "periodic",
            n_steps=n_steps,
            save_every=save_every,
        )
        ex, ey, ez, hx, hy, hz = fields
        t = t_start + dt * (1 + save_every * jnp.arange(ex.shape[0]))
        return EMFieldHistory3D(
            t=t,
            ex=ex,
            ey=ey,
            ez=ez,
            hx=hx,
            hy=hy,
            hz=hz,
            grid_x=jnp.arange(nx) * dx,
            grid_y=jnp.arange(ny) * dx,
            grid_z=jnp.arange(nz) * dx,
        )


@partial(jax.jit, static_argnames=("periodic", "n_steps", "save_every"))
def _fdtd3d_rollout(
    eps_r: Array,
    losses: tuple[tuple[Array, Array], ...],
    src_pos: Array,
    src_pol: Array,
    src_omega: Array,
    src_amp: Array,
    t0: Array,
    dt: Array,
    h_coef: float,
    e_coef: float,
    *,
    periodic: bool,
    n_steps: int,
    save_every: int,
) -> tuple[Array, ...]:
    """Split-field Yee rollout; returns (Ex, Ey, Ez, Hx, Hy, Hz) snapshots.

    Each component ``F`` is stored as ``F_a + F_b``, the parts driven by the
    derivative along axis ``a`` and ``b``. E parts decay with the node loss
    and H parts with the half-node loss of their derivative axis (H and the
    derivative of E live half a cell forward along that axis).
    """
    e_decay = [_decay(node) for node, _ in losses]
    h_decay = [_decay(half) for _, half in losses]
    ec = e_coef / eps_r
    pec = None if periodic else edge_mask(eps_r.shape)

    def dfwd(a: Array, axis: int) -> Array:
        return diff_forward(a, axis, periodic)

    def dbwd(a: Array, axis: int) -> Array:
        return diff_backward(a, axis, periodic)

    def part(
        old: Array, decay: tuple[Array, Array], coef: Array | float, drive: Array
    ) -> Array:
        a, b = decay
        return a * old + coef * b * drive

    def step(carry: tuple[Array, ...]) -> tuple[Array, ...]:
        exy, exz, eyz, eyx, ezx, ezy, hxy, hxz, hyz, hyx, hzx, hzy, n = carry
        ex, ey, ez = exy + exz, eyz + eyx, ezx + ezy
        # H = H - (dt/mu0) curl E, split by derivative axis.
        hxy = part(hxy, h_decay[1], -h_coef, dfwd(ez, 1))
        hxz = part(hxz, h_decay[2], h_coef, dfwd(ey, 2))
        hyz = part(hyz, h_decay[2], -h_coef, dfwd(ex, 2))
        hyx = part(hyx, h_decay[0], h_coef, dfwd(ez, 0))
        hzx = part(hzx, h_decay[0], -h_coef, dfwd(ey, 0))
        hzy = part(hzy, h_decay[1], h_coef, dfwd(ex, 1))
        hx, hy, hz = hxy + hxz, hyz + hyx, hzx + hzy
        # E = E + (dt/eps) curl H.
        exy = part(exy, e_decay[1], ec, dbwd(hz, 1))
        exz = part(exz, e_decay[2], -ec, dbwd(hy, 2))
        eyz = part(eyz, e_decay[2], ec, dbwd(hx, 2))
        eyx = part(eyx, e_decay[0], -ec, dbwd(hz, 0))
        ezx = part(ezx, e_decay[0], ec, dbwd(hy, 0))
        ezy = part(ezy, e_decay[1], -ec, dbwd(hx, 1))
        # Soft point sources evaluated at t_n (added after the update).
        value = src_amp * jnp.sin(src_omega * (t0 + n * dt))
        ix, iy, iz = src_pos
        exy = exy.at[ix, iy, iz].add(jnp.where(src_pol == 0, value, 0.0))
        eyz = eyz.at[ix, iy, iz].add(jnp.where(src_pol == 1, value, 0.0))
        ezx = ezx.at[ix, iy, iz].add(jnp.where(src_pol == 2, value, 0.0))
        e_parts: tuple[Array, ...] = (exy, exz, eyz, eyx, ezx, ezy)
        if pec is not None:
            e_parts = tuple(jnp.where(pec, 0.0, e) for e in e_parts)
        return (*e_parts, hxy, hxz, hyz, hyx, hzx, hzy, n + 1)

    def observe(carry: tuple[Array, ...]) -> tuple[Array, ...]:
        parts = carry[:-1]
        return tuple(parts[i] + parts[i + 1] for i in range(0, 12, 2))

    zeros = jnp.zeros(eps_r.shape)
    first = step((zeros,) * 12 + (jnp.asarray(0),))
    out: tuple[Array, ...] = strided_rollout(
        step, first, n_steps - 1, save_every, observe
    )
    return out
