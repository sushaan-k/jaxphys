"""Finite-Difference Time-Domain (FDTD) Maxwell solver.

Implements the Yee algorithm for solving Maxwell's equations on a
staggered grid. Supports TM polarization (Ez, Hx, Hy) in 2D with a
Berenger split-field perfectly matched layer (PML), perfectly conducting
(reflecting) walls, or periodic boundaries.

The Yee update equations (TM mode, 2D):

    Hx^{n+1/2} = Hx^{n-1/2} - (dt/mu0) * dEz/dy
    Hy^{n+1/2} = Hy^{n-1/2} + (dt/mu0) * dEz/dx
    Ez^{n+1}   = Ez^{n} + (dt/eps0) * (dHy/dx - dHx/dy)

The CFL stability condition requires:
    dt <= dx / (c * sqrt(2))  for 2D

Inside the PML, Ez is split into Ezx + Ezy; each part and the matching H
component decay with conductivities sigma and sigma* = sigma * mu0/eps0,
which makes the layer reflectionless at the continuum level for any angle
of incidence. The conductivity is graded as sigma_max * (depth/d)^3 with
the usual optimum sigma_max = 0.8 * (m + 1) / (eta0 * dx), m = 3.

References:
    - Yee, K.S. "Numerical solution of initial boundary value problems
      involving Maxwell's equations in isotropic media" (1966)
    - Taflove & Hagness. "Computational Electrodynamics" (2005)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxphys._rollout import strided_rollout
from jaxphys.config import EMConfig
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import EMFieldHistory

logger = logging.getLogger(__name__)

# Physical constants (SI)
C0 = 299792458.0  # speed of light (m/s)
MU0 = 4.0e-7 * np.pi  # permeability of free space (H/m)
EPS0 = 1.0 / (MU0 * C0**2)  # permittivity of free space (F/m)
ETA0 = MU0 * C0  # impedance of free space (ohm)


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

    Implements TM polarization (Ez, Hx, Hy). Boundaries: ``"absorbing"``
    (split-field PML of ``pml_layers`` cells backed by conducting walls),
    ``"reflecting"`` (perfectly conducting walls, Ez = 0) or ``"periodic"``.
    Sources and conductors can be added before simulation.

    Example:
        >>> grid = EMGrid(size=(200, 200), resolution=0.01)
        >>> grid.add_source(PlaneWave(frequency=3e9, y=20))
        >>> grid.add_conductor(Wall(y=100, gap_start=90, gap_end=110))
        >>> fields = grid.simulate(t_span=(0, 1e-8), dt=1e-11)

    Args:
        size: Grid dimensions (nx, ny) in cells.
        resolution: Cell size dx = dy in meters.
        boundary: Boundary condition type.
        pml_layers: PML thickness in cells (used when ``boundary`` is
            ``"absorbing"``; 0 gives bare conducting walls).
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

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 1e-8),
        dt: float | None = None,
        save_every: int = 10,
    ) -> EMFieldHistory:
        """Run the FDTD simulation.

        Args:
            t_span: (t_start, t_end) in seconds.
            dt: Time step in seconds. Defaults to 0.99x the 2D CFL limit
                ``dx / (c * sqrt(2))``; larger values are rejected.
            save_every: Save field snapshot every N steps.

        Returns:
            EMFieldHistory with snapshots after steps 1, 1 + save_every,
            1 + 2*save_every, ... (``t`` holds the matching times).

        Raises:
            ConfigurationError: If no source was added, ``dt`` violates the
                CFL condition, or the time span holds less than one step.
        """
        dx = self._dx
        nx, ny = self._nx, self._ny
        boundary = self._config.boundary

        if not self._sources:
            raise ConfigurationError(
                "At least one source must be added before simulation"
            )
        dt_max = dx / (C0 * float(np.sqrt(2.0)))
        dt = 0.99 * dt_max if dt is None else float(dt)
        if not 0 < dt <= dt_max:
            raise ConfigurationError(
                f"dt={dt:.3e} s violates the 2D CFL condition "
                f"0 < dt <= dx/(c*sqrt(2)) = {dt_max:.3e} s"
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
            "Starting FDTD simulation: grid=%dx%d, dt=%.2e, n_steps=%d",
            nx,
            ny,
            dt,
            n_steps,
        )

        n_pml = self._config.pml_layers if boundary == "absorbing" else 0
        # Normalised PML loss a = sigma*dt/eps0 at E nodes (i) and H nodes
        # (i + 1/2) along each axis.
        loss_e_x, loss_h_x = _pml_loss(nx, n_pml, dx, dt)
        loss_e_y, loss_h_y = _pml_loss(ny, n_pml, dx, dt)

        ez, hx, hy = _fdtd2d_rollout(
            self._build_conductor_mask(),
            loss_e_x[:, None],
            loss_e_y[None, :],
            loss_h_x[:, None],
            loss_h_y[None, :],
            jnp.array([src.y for src in self._sources]),
            jnp.array([2.0 * np.pi * src.frequency for src in self._sources]),
            jnp.array([src.amplitude for src in self._sources], dtype=jnp.float64),
            jnp.asarray(t_start, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
            dt / (MU0 * dx),
            dt / (EPS0 * dx),
            periodic=boundary == "periodic",
            n_steps=n_steps,
            save_every=save_every,
        )

        t = t_start + dt * (1 + save_every * jnp.arange(ez.shape[0]))
        return EMFieldHistory(
            t=t,
            ez=ez,
            hx=hx,
            hy=hy,
            grid_x=jnp.arange(nx) * dx,
            grid_y=jnp.arange(ny) * dx,
        )


def _pml_loss(
    n: int, n_pml: int, dx: float, dt: float, order: int = 3
) -> tuple[Array, Array]:
    """Per-step PML loss ``sigma*dt/eps0`` on integer and half-integer nodes.

    Returns ``(at_nodes, at_half_nodes)``, each of shape ``(n,)``; the second
    is sampled at ``i + 1/2`` where the staggered H components live.
    """
    if n_pml == 0:
        zeros = jnp.zeros(n)
        return zeros, zeros
    sigma_max = 0.8 * (order + 1) / (ETA0 * dx)

    def profile(pos: Array) -> Array:
        # Depth into the layer, 0 at the inner PML interface, 1 at the wall.
        depth = jnp.maximum(n_pml - pos, pos - (n - 1 - n_pml)) / n_pml
        return sigma_max * jnp.clip(depth, 0.0, 1.0) ** order * dt / EPS0

    idx = jnp.arange(n, dtype=jnp.float64)
    return profile(idx), profile(idx + 0.5)


def _decay(loss: Array) -> tuple[Array, Array]:
    """Exponential time-stepping factors ``exp(-a)`` and ``(1 - exp(-a)) / a``."""
    safe = jnp.where(loss > 0, loss, 1.0)
    return jnp.exp(-loss), jnp.where(loss > 0, -jnp.expm1(-safe) / safe, 1.0)


def diff_forward(a: Array, axis: int, periodic: bool) -> Array:
    """``a[i+1] - a[i]`` along ``axis``; zero on the last plane unless periodic."""
    if periodic:
        return jnp.roll(a, -1, axis=axis) - a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (0, 1)
    return jnp.pad(jnp.diff(a, axis=axis), pad)


def diff_backward(a: Array, axis: int, periodic: bool) -> Array:
    """``a[i] - a[i-1]`` along ``axis``; zero on the first plane unless periodic."""
    if periodic:
        return a - jnp.roll(a, 1, axis=axis)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (1, 0)
    return jnp.pad(jnp.diff(a, axis=axis), pad)


def edge_mask(shape: tuple[int, ...]) -> Array:
    """Boolean mask of the outermost cells of a grid (PEC walls)."""
    mask = jnp.zeros(shape, dtype=bool)
    for axis in range(len(shape)):
        index: list[Any] = [slice(None)] * len(shape)
        for edge in (0, -1):
            index[axis] = edge
            mask = mask.at[tuple(index)].set(True)
    return mask


@partial(jax.jit, static_argnames=("periodic", "n_steps", "save_every"))
def _fdtd2d_rollout(
    conductor_mask: Array,
    loss_e_x: Array,
    loss_e_y: Array,
    loss_h_x: Array,
    loss_h_y: Array,
    src_rows: Array,
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
) -> tuple[Array, Array, Array]:
    """Split-field TM Yee rollout; returns (Ez, Hx, Hy) snapshots."""
    nx, ny = conductor_mask.shape
    pec = conductor_mask if periodic else conductor_mask | edge_mask((nx, ny))
    # Matched decay factors: sigma*/mu0 = sigma/eps0, so E and H share
    # exp(-a); the curl coefficients carry (1 - exp(-a)) / a.
    ea_x, eb_x = _decay(loss_e_x)
    ea_y, eb_y = _decay(loss_e_y)
    ha_x, hb_x = _decay(loss_h_x)
    ha_y, hb_y = _decay(loss_h_y)

    def step(carry: tuple[Array, ...]) -> tuple[Array, ...]:
        ezx, ezy, hx, hy, n = carry
        ez = ezx + ezy
        hx = ha_y * hx - h_coef * hb_y * diff_forward(ez, 1, periodic)
        hy = ha_x * hy + h_coef * hb_x * diff_forward(ez, 0, periodic)
        ezx = ea_x * ezx + e_coef * eb_x * diff_backward(hy, 0, periodic)
        ezy = ea_y * ezy - e_coef * eb_y * diff_backward(hx, 1, periodic)
        # Soft line sources evaluated at t_n (added after the update).
        t_n = t0 + n * dt
        ezx = ezx.at[:, src_rows].add(src_amp * jnp.sin(src_omega * t_n))
        ezx = jnp.where(pec, 0.0, ezx)
        ezy = jnp.where(pec, 0.0, ezy)
        return ezx, ezy, hx, hy, n + 1

    zeros = jnp.zeros((nx, ny))
    first = step((zeros, zeros, zeros, zeros, jnp.asarray(0)))
    out: tuple[Array, Array, Array] = strided_rollout(
        step,
        first,
        n_steps - 1,
        save_every,
        lambda c: (c[0] + c[1], c[2], c[3]),
    )
    return out
