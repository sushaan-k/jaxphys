"""Lattice Boltzmann Method (LBM) solver for 2D incompressible flow.

Implements the D2Q9 lattice Boltzmann scheme with BGK (single relaxation
time) collision operator. The method solves the weakly-compressible
Navier-Stokes equations in the low-Mach-number limit.

The D2Q9 lattice uses 9 velocity directions:

    6  2  5
     \\ | /
    3--0--1
     / | \\
    7  4  8

The BGK collision operator relaxes f toward the equilibrium f_eq:

    f_i(x + c_i*dt, t + dt) = f_i(x, t) - (f_i - f_eq_i) / tau

where tau = 3*nu + 0.5 is the relaxation time and nu is the kinematic
viscosity in lattice units.

References:
    - Succi. "The Lattice Boltzmann Equation" (2001)
    - Kruger et al. "The Lattice Boltzmann Method" (2017)
    - Chen & Doolen. "Lattice Boltzmann method for fluid flows" (1998)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import FluidConfig
from neurosim.exceptions import ConfigurationError, NumericalInstabilityError
from neurosim.state import FluidHistory

logger = logging.getLogger(__name__)


class D2Q9:
    """D2Q9 lattice constants for LBM.

    Precomputed velocity vectors and weights for the 9-velocity
    2D lattice.
    """

    # Lattice velocity vectors: (9, 2)
    c: Array = jnp.array(
        [
            [0, 0],   # 0: rest
            [1, 0],   # 1: east
            [0, 1],   # 2: north
            [-1, 0],  # 3: west
            [0, -1],  # 4: south
            [1, 1],   # 5: NE
            [-1, 1],  # 6: NW
            [-1, -1], # 7: SW
            [1, -1],  # 8: SE
        ]
    )

    # Lattice weights
    w: Array = jnp.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )

    # Opposite direction indices (for bounce-back)
    opposite: tuple[int, ...] = (0, 3, 4, 1, 2, 7, 8, 5, 6)


@dataclass(frozen=True)
class Obstacle:
    """Boolean mask defining solid regions on the grid.

    Attributes:
        mask: Boolean array, shape (nx, ny). True where solid.
    """

    mask: Array


class LBMGrid:
    """2D Lattice Boltzmann simulation grid.

    Implements the D2Q9 BGK scheme for incompressible flow with
    optional solid obstacles (bounce-back boundaries).

    Example:
        >>> grid = LBMGrid(size=(200, 100), viscosity=0.02)
        >>> # Add a cylindrical obstacle
        >>> import jax.numpy as jnp
        >>> x, y = jnp.meshgrid(jnp.arange(200), jnp.arange(100), indexing='ij')
        >>> cylinder = (x - 50)**2 + (y - 50)**2 < 15**2
        >>> grid.add_obstacle(Obstacle(mask=cylinder))
        >>> result = grid.simulate(
        ...     n_steps=5000, u_inlet=0.04, save_every=100,
        ... )

    Args:
        size: Grid dimensions (nx, ny) in lattice units.
        viscosity: Kinematic viscosity in lattice units. Must satisfy
            tau = 3*nu + 0.5 > 0.5 for stability.
        boundary: Boundary condition type for domain edges.
    """

    def __init__(
        self,
        size: tuple[int, int] = (200, 100),
        viscosity: float = 0.02,
        boundary: Literal["periodic", "no_slip", "free_slip"] = "periodic",
    ) -> None:
        self._nx, self._ny = size
        self._config = FluidConfig(
            viscosity=viscosity,
            method="lbm",
            boundary=boundary,
        )
        self._obstacles: list[Obstacle] = []
        self._lattice = D2Q9()

        # Relaxation time
        self._tau = 3.0 * viscosity + 0.5
        if self._tau <= 0.5:
            raise ConfigurationError(
                f"Relaxation time tau={self._tau:.4f} must be > 0.5. "
                f"Increase viscosity (got {viscosity})."
            )

        if self._nx < 4 or self._ny < 4:
            raise ConfigurationError(f"Grid must be at least 4x4, got {size}")

    @property
    def size(self) -> tuple[int, int]:
        """Grid dimensions (nx, ny)."""
        return (self._nx, self._ny)

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return self._tau

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add a solid obstacle to the grid.

        Args:
            obstacle: Obstacle with boolean mask of shape (nx, ny).
        """
        if obstacle.mask.shape != (self._nx, self._ny):
            raise ConfigurationError(
                f"Obstacle mask shape {obstacle.mask.shape} doesn't match "
                f"grid size ({self._nx}, {self._ny})"
            )
        self._obstacles.append(obstacle)

    def _build_obstacle_mask(self) -> Array:
        """Combine all obstacles into a single boolean mask."""
        mask = jnp.zeros((self._nx, self._ny), dtype=bool)
        for obs in self._obstacles:
            mask = mask | obs.mask
        return mask

    def simulate(
        self,
        n_steps: int = 5000,
        u_inlet: float = 0.04,
        save_every: int = 100,
        initial_rho: Array | None = None,
        initial_ux: Array | None = None,
        initial_uy: Array | None = None,
    ) -> FluidHistory:
        """Run the LBM simulation.

        Initializes with uniform flow in the x-direction at the given
        inlet velocity. Uses the BGK collision operator with bounce-back
        on obstacles.

        Args:
            n_steps: Number of time steps to simulate.
            u_inlet: Inlet velocity magnitude (lattice units). Should be
                << 1/sqrt(3) ~ 0.577 for stability.
            save_every: Save snapshots every N steps.
            initial_rho: Optional initial density, shape (nx, ny).
            initial_ux: Optional initial x-velocity, shape (nx, ny).
            initial_uy: Optional initial y-velocity, shape (nx, ny).

        Returns:
            FluidHistory with density, velocity, and vorticity snapshots.
        """
        if u_inlet >= 0.3:
            raise ConfigurationError(
                f"Inlet velocity {u_inlet} too high for LBM stability. "
                "Keep u_inlet << 0.577 (speed of sound). Recommended < 0.1."
            )

        nx, ny = self._nx, self._ny
        tau = self._tau
        omega = 1.0 / tau
        lattice = self._lattice
        c = lattice.c
        w = lattice.w
        opp = jnp.array(lattice.opposite)

        obstacle_mask = self._build_obstacle_mask()

        logger.info(
            "Starting LBM simulation: grid=%dx%d, tau=%.3f, u_inlet=%.4f, "
            "n_steps=%d",
            nx, ny, tau, u_inlet, n_steps,
        )

        # Initialize macroscopic fields
        if initial_rho is None:
            rho = jnp.ones((nx, ny))
        else:
            rho = initial_rho

        if initial_ux is None:
            ux = jnp.full((nx, ny), u_inlet)
        else:
            ux = initial_ux

        if initial_uy is None:
            uy = jnp.zeros((nx, ny))
        else:
            uy = initial_uy

        # Initialize distribution to equilibrium
        cx = jnp.array([int(c[i, 0]) for i in range(9)])
        cy = jnp.array([int(c[i, 1]) for i in range(9)])
        f = _compute_equilibrium(rho, ux, uy, cx, cy, w)

        # Zero velocity inside obstacles
        f = jnp.where(obstacle_mask[..., jnp.newaxis], w[jnp.newaxis, jnp.newaxis, :], f)

        grid_x = jnp.arange(nx, dtype=jnp.float64)
        grid_y = jnp.arange(ny, dtype=jnp.float64)

        # Precompute integer velocity components for use in JAX-traced code
        cx = jnp.array([int(c[i, 0]) for i in range(9)])
        cy = jnp.array([int(c[i, 1]) for i in range(9)])

        # Precompute streaming shift tuples (static, not traced)
        stream_shifts = [(int(c[i, 0]), int(c[i, 1])) for i in range(9)]

        def step(
            carry: tuple[Array, int],
            _: None,
        ) -> tuple[tuple[Array, int], tuple[Array, Array, Array]]:
            f_c, step_idx = carry

            # Collision: BGK
            rho_c = jnp.sum(f_c, axis=-1)
            ux_c = jnp.sum(f_c * cx, axis=-1) / rho_c
            uy_c = jnp.sum(f_c * cy, axis=-1) / rho_c

            f_eq = _compute_equilibrium(rho_c, ux_c, uy_c, cx, cy, w)
            f_out = f_c - omega * (f_c - f_eq)

            # Bounce-back on obstacles: swap populations to opposite direction
            f_bounce = f_out[..., opp]
            f_out = jnp.where(obstacle_mask[..., jnp.newaxis], f_bounce, f_out)

            # Streaming: shift each population by its velocity vector
            f_new = _stream(f_out, stream_shifts)

            # Inlet boundary (Zou-He, left wall): fixed velocity
            # Directions: 0=rest, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE
            rho_inlet = (
                (f_new[0, :, 0] + f_new[0, :, 2] + f_new[0, :, 4])
                + 2.0 * (f_new[0, :, 3] + f_new[0, :, 6] + f_new[0, :, 7])
            ) / (1.0 - u_inlet)
            f_new = f_new.at[0, :, 1].set(
                f_new[0, :, 3] + (2.0 / 3.0) * rho_inlet * u_inlet
            )
            f_new = f_new.at[0, :, 5].set(
                f_new[0, :, 7]
                - 0.5 * (f_new[0, :, 2] - f_new[0, :, 4])
                + (1.0 / 6.0) * rho_inlet * u_inlet
            )
            f_new = f_new.at[0, :, 8].set(
                f_new[0, :, 6]
                + 0.5 * (f_new[0, :, 2] - f_new[0, :, 4])
                + (1.0 / 6.0) * rho_inlet * u_inlet
            )

            # Outlet boundary (zero-gradient, right wall)
            f_new = f_new.at[-1, :, :].set(f_new[-2, :, :])

            # Recompute macroscopic for output
            rho_out = jnp.sum(f_new, axis=-1)
            ux_out = jnp.sum(f_new * c[:, 0], axis=-1) / rho_out
            uy_out = jnp.sum(f_new * c[:, 1], axis=-1) / rho_out

            return (f_new, step_idx + 1), (rho_out, ux_out, uy_out)

        init = (f, 0)
        _, (rho_all, ux_all, uy_all) = jax.lax.scan(step, init, None, length=n_steps)

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            rho_all = rho_all[indices]
            ux_all = ux_all[indices]
            uy_all = uy_all[indices]

        # Compute vorticity: d(uy)/dx - d(ux)/dy
        vort = jnp.gradient(uy_all, axis=1) - jnp.gradient(ux_all, axis=2)

        t = jnp.arange(rho_all.shape[0], dtype=jnp.float64) * save_every

        return FluidHistory(
            t=t,
            rho=rho_all,
            ux=ux_all,
            uy=uy_all,
            vorticity=vort,
            grid_x=grid_x,
            grid_y=grid_y,
        )


def _compute_equilibrium(
    rho: Array, ux: Array, uy: Array, cx: Array, cy: Array, w: Array
) -> Array:
    """Compute the equilibrium distribution function.

    f_eq_i = w_i * rho * (1 + 3*(c_i . u) + 4.5*(c_i . u)^2 - 1.5*u^2)

    Args:
        rho: Density field, shape (nx, ny).
        ux: x-velocity, shape (nx, ny).
        uy: y-velocity, shape (nx, ny).
        cx: x-component of lattice velocities, shape (9,).
        cy: y-component of lattice velocities, shape (9,).
        w: Lattice weights, shape (9,).

    Returns:
        Equilibrium distribution, shape (nx, ny, 9).
    """
    u_sq = ux**2 + uy**2  # (nx, ny)
    # c_i . u for each direction: (nx, ny, 9)
    cu = cx * ux[..., jnp.newaxis] + cy * uy[..., jnp.newaxis]

    f_eq = w * rho[..., jnp.newaxis] * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq[..., jnp.newaxis]
    )
    return f_eq


def _stream(f: Array, shifts: list[tuple[int, int]]) -> Array:
    """Stream populations along their velocity vectors.

    Each population f[..., i] is shifted by the corresponding
    velocity using periodic boundary conditions (jnp.roll).

    Args:
        f: Distribution function, shape (nx, ny, 9).
        shifts: List of (dx, dy) integer shifts for each direction.

    Returns:
        Streamed distribution, shape (nx, ny, 9).
    """
    f_new = jnp.zeros_like(f)
    for i, (sx, sy) in enumerate(shifts):
        f_new = f_new.at[..., i].set(
            jnp.roll(jnp.roll(f[..., i], sx, axis=0), sy, axis=1)
        )
    return f_new
