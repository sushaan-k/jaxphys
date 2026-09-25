"""Weakly compressible smoothed-particle hydrodynamics (SPH) in 2D.

The fluid is a set of particles of equal mass ``m``. Density follows from
kernel summation and pressure from the Tait equation of state:

    rho_i = sum_j m W(|r_i - r_j|, h)
    p_i   = B ((rho_i / rho0)^gamma - 1),   B = rho0 c0^2 / gamma

and the symmetric momentum equation (Monaghan 1992)

    dv_i/dt = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij) grad_i W_ij + g

conserves linear momentum exactly. ``Pi_ij`` is Monaghan's artificial
viscosity (strength ``alpha``). Time stepping is kick-drift-kick leapfrog.

Neighbours are found with a cell list (cells of side >= 2h, the kernel
support), so the cost per step is O(N) for a roughly uniform fluid; the
list is rebuilt every step inside the compiled loop. Small boxes (fewer
than 3 cells per axis) and traced inputs fall back to all pairs.

Boundaries are periodic, or reflecting walls that mirror a particle's
position and normal velocity. Reflecting walls carry no boundary particles,
so densities are underestimated within ``2h`` of a wall.

References:
    - Monaghan. "Smoothed particle hydrodynamics", ARA&A 30 (1992)
    - Monaghan. "Simulating free surface flows with SPH", JCP 110 (1994)
    - Price. "Smoothed particle hydrodynamics and magnetohydrodynamics",
      JCP 231 (2012)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys._rollout import is_traced, strided_rollout
from jaxphys.exceptions import ConfigurationError, NumericalInstabilityError

logger = logging.getLogger(__name__)


def cubic_spline_kernel(r: Array, h: float | Array) -> Array:
    """2D cubic spline kernel ``W(r, h)`` with support ``2h`` (unit integral)."""
    q = r / h
    sigma = 10.0 / (7.0 * jnp.pi * h**2)
    inner = 1.0 - 1.5 * q**2 + 0.75 * q**3
    outer = 0.25 * jnp.clip(2.0 - q, 0.0) ** 3
    return sigma * jnp.where(q < 1.0, inner, outer)


def cubic_spline_gradient(r: Array, h: float | Array) -> Array:
    """Radial derivative ``dW/dr`` of :func:`cubic_spline_kernel`."""
    q = r / h
    sigma = 10.0 / (7.0 * jnp.pi * h**2)
    inner = -3.0 * q + 2.25 * q**2
    outer = -0.75 * jnp.clip(2.0 - q, 0.0) ** 2
    return sigma / h * jnp.where(q < 1.0, inner, outer)


@dataclass(frozen=True)
class SPHTrajectory:
    """Saved SPH particle states.

    Attributes:
        t: Times, shape (n_saved,).
        positions: Shape (n_saved, n_particles, 2).
        velocities: Shape (n_saved, n_particles, 2).
        density: Summation density, shape (n_saved, n_particles).
        mass: Particle mass.
    """

    t: Array
    positions: Array
    velocities: Array
    density: Array
    mass: float

    @property
    def kinetic_energy(self) -> Array:
        """Total kinetic energy at each saved time."""
        return 0.5 * self.mass * jnp.sum(self.velocities**2, axis=(-2, -1))

    @property
    def momentum(self) -> Array:
        """Total linear momentum at each saved time, shape (n_saved, 2)."""
        return self.mass * jnp.sum(self.velocities, axis=-2)


jax.tree_util.register_dataclass(
    SPHTrajectory,
    data_fields=["t", "positions", "velocities", "density"],
    meta_fields=["mass"],
)


class SPHFluid:
    """2D weakly compressible SPH fluid in a rectangular box.

    Example:
        >>> dx = 0.02
        >>> xs = jnp.arange(0.0, 1.0, dx) + dx / 2
        >>> pos = jnp.stack(jnp.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
        >>> fluid = SPHFluid(mass=1000.0 * dx**2, smoothing_length=1.3 * dx,
        ...                  box=(1.0, 1.0), sound_speed=20.0)
        >>> traj = fluid.simulate(pos, jnp.zeros_like(pos), (0.0, 0.1), dt=2e-4)

    Args:
        mass: Mass of every particle (``rho0 * dx**2`` for spacing ``dx``).
        smoothing_length: Kernel smoothing length ``h`` (``~1.3 dx``).
        box: Domain size ``(Lx, Ly)``; particles live in ``[0, Lx) x [0, Ly)``.
        rest_density: Reference density ``rho0``.
        sound_speed: Numerical sound speed ``c0``; keep it ~10x the fastest
            flow speed so density fluctuations stay ~1%.
        gamma: Tait exponent.
        alpha: Artificial viscosity strength (0 disables it).
        gravity: Body acceleration ``(gx, gy)``.
        boundary: ``"periodic"`` or ``"reflecting"``.
    """

    def __init__(
        self,
        mass: float,
        smoothing_length: float,
        box: tuple[float, float],
        rest_density: float = 1000.0,
        sound_speed: float = 10.0,
        gamma: float = 7.0,
        alpha: float = 0.1,
        gravity: tuple[float, float] = (0.0, 0.0),
        boundary: Literal["periodic", "reflecting"] = "periodic",
    ) -> None:
        if min(mass, smoothing_length, rest_density, sound_speed, gamma) <= 0:
            raise ConfigurationError(
                "mass, smoothing_length, rest_density, sound_speed and gamma "
                "must be positive"
            )
        if min(box) <= 4 * smoothing_length:
            raise ConfigurationError(f"box {box} must exceed 4 * smoothing_length")
        if boundary not in ("periodic", "reflecting"):
            raise ConfigurationError(f"Unknown boundary '{boundary}'")
        self.mass = float(mass)
        self.h = float(smoothing_length)
        self.box = (float(box[0]), float(box[1]))
        self.rest_density = float(rest_density)
        self.sound_speed = float(sound_speed)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.gravity = (float(gravity[0]), float(gravity[1]))
        self.boundary = boundary

    def _grid_shape(self) -> tuple[int, int]:
        """Cells of side >= 2h (the kernel support) along each axis."""
        return (
            int(self.box[0] // (2.0 * self.h)),
            int(self.box[1] // (2.0 * self.h)),
        )

    def cell_capacity(self, positions: Array) -> int | None:
        """Slots per cell for the neighbour search, or ``None`` for all pairs.

        Uses twice the current maximum cell occupancy (at least 8). Returns
        ``None`` when the box holds fewer than 3 cells along an axis, where a
        cell list cannot beat the all-pairs search.
        """
        nx, ny = self._grid_shape()
        if min(nx, ny) < 3:
            return None
        cells = _cell_ids(jnp.asarray(positions), self.box, (nx, ny))
        occupancy = int(jnp.max(jnp.bincount(cells, length=nx * ny)))
        return max(8, 2 * occupancy)

    def _neighbours(
        self, positions: Array, capacity: int | None
    ) -> tuple[Array, Array, Array, Array]:
        """Candidate neighbours of every particle and their separations.

        Returns ``(idx, mask, dr, dist)`` of shapes (n, k), (n, k), (n, k, 2)
        and (n, k): candidate indices, validity (excludes padding and the
        particle itself), ``r_i - r_j`` (minimum image if periodic) and
        ``|r_i - r_j|``. Candidates come from the 3x3 surrounding cells of a
        cell list when ``capacity`` is given, else from all particles.
        """
        n = positions.shape[0]
        if capacity is None:
            idx = jnp.broadcast_to(jnp.arange(n), (n, n))
            overflow = jnp.array(False)
        else:
            idx, overflow = _cell_list_candidates(
                positions,
                self.box,
                self._grid_shape(),
                capacity,
                self.boundary == "periodic",
            )
        mask = (idx < n) & (idx != jnp.arange(n)[:, None])
        dr = positions[:, None, :] - positions[jnp.minimum(idx, n - 1)]
        if self.boundary == "periodic":
            L = jnp.asarray(self.box)
            dr = dr - L * jnp.round(dr / L)
        dist_sq = jnp.sum(dr**2, axis=-1)
        # Offset masked and coincident pairs before the sqrt so gradients
        # stay finite.
        ok = mask & (dist_sq > 0)
        dist = jnp.where(ok, jnp.sqrt(jnp.where(ok, dist_sq, 1.0)), 0.0)
        # An overflowing cell list misses neighbours: poison the state with
        # NaN so simulate() reports it instead of returning wrong physics.
        dist = jnp.where(overflow, jnp.nan, dist)
        return idx, mask, dr, dist

    def _density(self, mask: Array, dist: Array) -> Array:
        w = jnp.where(mask, cubic_spline_kernel(dist, self.h), 0.0)
        self_term = cubic_spline_kernel(jnp.zeros(()), self.h)
        return self.mass * (jnp.sum(w, axis=1) + self_term)

    def density(self, positions: Array) -> Array:
        """Summation density at each particle, shape (n,)."""
        positions = jnp.asarray(positions, dtype=jnp.float64)
        _, mask, _, dist = self._neighbours(positions, self._capacity(positions))
        return self._density(mask, dist)

    def _capacity(self, positions: Array) -> int | None:
        # Traced positions (inside jit/vmap/grad) fall back to all pairs.
        return None if is_traced(positions) else self.cell_capacity(positions)

    def pressure(self, density: Array) -> Array:
        """Tait equation of state."""
        B = self.rest_density * self.sound_speed**2 / self.gamma
        return B * ((density / self.rest_density) ** self.gamma - 1.0)

    def internal_energy(self, density: Array) -> Array:
        """Specific internal energy ``u(rho)`` with ``du/drho = p / rho^2``."""
        rho0, g = self.rest_density, self.gamma
        B = rho0 * self.sound_speed**2 / g
        return B * ((density / rho0) ** (g - 1.0) / (rho0 * (g - 1.0)) + 1.0 / density)

    def total_energy(self, positions: Array, velocities: Array) -> Array:
        """Kinetic plus internal energy (conserved when alpha = 0, g = 0)."""
        rho = self.density(positions)
        kinetic = 0.5 * self.mass * jnp.sum(jnp.asarray(velocities) ** 2)
        return kinetic + self.mass * jnp.sum(self.internal_energy(rho))

    def acceleration(
        self, positions: Array, velocities: Array, capacity: int | None = None
    ) -> tuple[Array, Array]:
        """Particle accelerations and summation densities.

        Args:
            positions: Shape (n, 2).
            velocities: Shape (n, 2).
            capacity: Cell-list slots per cell (see :meth:`cell_capacity`);
                ``None`` evaluates all pairs.
        """
        n = positions.shape[0]
        idx, mask, dr, dist = self._neighbours(positions, capacity)
        rho = self._density(mask, dist)
        p_over_rho2 = self.pressure(rho) / rho**2
        j = jnp.minimum(idx, n - 1)
        # grad_i W_ij = W'(r) (r_i - r_j) / r
        safe = jnp.where(dist > 0, dist, 1.0)
        grad_w = (cubic_spline_gradient(dist, self.h) / safe)[..., None] * dr
        grad_w = jnp.where((mask & (dist > 0))[..., None], grad_w, 0.0)

        dv = velocities[:, None, :] - velocities[j]
        vr = jnp.sum(dv * dr, axis=-1)
        mu = self.h * vr / (dist**2 + 0.01 * self.h**2)
        rho_bar = 0.5 * (rho[:, None] + rho[j])
        visc = jnp.where(vr < 0, -self.alpha * self.sound_speed * mu / rho_bar, 0.0)

        coeff = p_over_rho2[:, None] + p_over_rho2[j] + visc
        acc = -self.mass * jnp.einsum("ij,ijk->ik", coeff, grad_w)
        return acc + jnp.asarray(self.gravity), rho

    def _wrap(self, positions: Array, velocities: Array) -> tuple[Array, Array]:
        L = jnp.asarray(self.box)
        if self.boundary == "periodic":
            return jnp.mod(positions, L), velocities
        below, above = positions < 0, positions > L
        positions = jnp.where(
            below, -positions, jnp.where(above, 2 * L - positions, positions)
        )
        velocities = jnp.where(below | above, -velocities, velocities)
        return positions, velocities

    def simulate(
        self,
        positions: Array,
        velocities: Array,
        t_span: tuple[float, float],
        dt: float,
        save_every: int = 10,
    ) -> SPHTrajectory:
        """Integrate the particles with kick-drift-kick leapfrog.

        Args:
            positions: Initial positions, shape (n, 2).
            velocities: Initial velocities, shape (n, 2).
            t_span: ``(t_start, t_end)``.
            dt: Time step. Stability needs ``dt <~ 0.25 h / c0``.
            save_every: Save every N steps.

        Returns:
            SPHTrajectory saved at ``t_start + k * save_every * dt``.

        Raises:
            ConfigurationError: For invalid shapes, ``dt`` or ``save_every``.
        """
        positions = jnp.asarray(positions, dtype=jnp.float64)
        velocities = jnp.asarray(velocities, dtype=jnp.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ConfigurationError(f"positions must be (n, 2), got {positions.shape}")
        if velocities.shape != positions.shape:
            raise ConfigurationError("velocities must match positions in shape")
        if dt <= 0 or save_every < 1:
            raise ConfigurationError("dt must be positive and save_every >= 1")
        if dt > 0.4 * self.h / self.sound_speed:
            logger.warning(
                "dt=%.3g exceeds the acoustic limit 0.4 h/c0 = %.3g",
                dt,
                0.4 * self.h / self.sound_speed,
            )
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        capacity = self._capacity(positions)
        logger.info(
            "Starting SPH: n=%d, n_steps=%d, cell capacity=%s",
            positions.shape[0],
            n_steps,
            capacity,
        )
        pos, vel, rho = _sph_rollout(
            self,
            positions,
            velocities,
            dt,
            capacity=capacity,
            n_steps=n_steps,
            save_every=save_every,
        )
        if not is_traced(rho) and bool(jnp.any(~jnp.isfinite(rho))):
            raise NumericalInstabilityError(
                "SPH state became non-finite (a cell-list overflow poisons the "
                "state): reduce dt, raise sound_speed, or pass positions that "
                "are less clustered."
            )
        return SPHTrajectory(
            t=t_start + dt * jnp.arange(0, n_steps + 1, save_every),
            positions=pos,
            velocities=vel,
            density=rho,
            mass=self.mass,
        )


@partial(jax.jit, static_argnames=("fluid", "capacity", "n_steps", "save_every"))
def _sph_rollout(
    fluid: SPHFluid,
    positions: Array,
    velocities: Array,
    dt: float,
    *,
    capacity: int | None,
    n_steps: int,
    save_every: int,
) -> tuple[Array, Array, Array]:
    """Kick-drift-kick rollout; returns (positions, velocities, density)."""

    def step(
        carry: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        pos, vel, acc, _ = carry
        vel_half = vel + 0.5 * dt * acc
        pos, vel_half = fluid._wrap(pos + dt * vel_half, vel_half)
        # The viscous force depends on velocity: evaluate it at the
        # half-step velocity (standard for explicit SPH leapfrog).
        acc, rho = fluid.acceleration(pos, vel_half, capacity)
        return pos, vel_half + 0.5 * dt * acc, acc, rho

    acc0, rho0 = fluid.acceleration(positions, velocities, capacity)
    out: tuple[Array, Array, Array] = strided_rollout(
        step,
        (positions, velocities, acc0, rho0),
        n_steps,
        save_every,
        lambda c: (c[0], c[1], c[3]),
    )
    return out


def _cell_ids(
    positions: Array, box: tuple[float, float], shape: tuple[int, int]
) -> Array:
    """Flat cell index of each particle (cells are box / shape in size)."""
    ij = jnp.floor(positions / (jnp.asarray(box) / jnp.asarray(shape))).astype(int)
    ij = jnp.clip(ij, 0, jnp.asarray(shape) - 1)
    return ij[:, 0] * shape[1] + ij[:, 1]


def _cell_list_candidates(
    positions: Array,
    box: tuple[float, float],
    shape: tuple[int, int],
    capacity: int,
    periodic: bool,
) -> tuple[Array, Array]:
    """Particles in the 3x3 cells around each particle's cell.

    Returns ``(idx, overflow)``: indices of shape (n, 9 * capacity), padded
    with ``n``, and a flag set when a cell holds more than ``capacity``
    particles (the candidate lists are then incomplete).
    """
    n = positions.shape[0]
    nx, ny = shape
    cell = _cell_ids(positions, box, shape)
    order = jnp.argsort(cell)
    sorted_cell = cell[order]
    counts = jnp.bincount(cell, length=nx * ny)
    starts = jnp.cumsum(counts) - counts
    rank = jnp.arange(n) - starts[sorted_cell]
    # Table of particle indices per cell; the extra last row is an empty
    # cell used for out-of-range neighbours of non-periodic boxes.
    table = jnp.full((nx * ny + 1, capacity), n)
    table = table.at[sorted_cell, rank].set(order, mode="drop")

    cx, cy = cell // ny, cell % ny
    neighbour_cells = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ni, nj = cx + di, cy + dj
            if periodic:
                flat = (ni % nx) * ny + nj % ny
            else:
                inside = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
                flat = jnp.where(inside, ni * ny + nj, nx * ny)
            neighbour_cells.append(flat)
    idx = table[jnp.stack(neighbour_cells, axis=1)].reshape(n, 9 * capacity)
    return idx, jnp.max(counts) > capacity
