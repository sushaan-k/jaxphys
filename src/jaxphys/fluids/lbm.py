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
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxphys._rollout import strided_rollout
from jaxphys.config import FluidConfig
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import FluidHistory

logger = logging.getLogger(__name__)


class D2Q9:
    """D2Q9 lattice constants for LBM.

    Precomputed velocity vectors and weights for the 9-velocity
    2D lattice.
    """

    # Lattice velocity vectors: (9, 2)
    c: Array = jnp.array(
        [
            [0, 0],  # 0: rest
            [1, 0],  # 1: east
            [0, 1],  # 2: north
            [-1, 0],  # 3: west
            [0, -1],  # 4: south
            [1, 1],  # 5: NE
            [-1, 1],  # 6: NW
            [-1, -1],  # 7: SW
            [1, -1],  # 8: SE
        ]
    )

    # Lattice weights
    w: Array = jnp.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )

    # Opposite direction indices (for bounce-back)
    opposite: tuple[int, ...] = (0, 3, 4, 1, 2, 7, 8, 5, 6)

    # Direction indices with the y-component reversed (specular reflection)
    mirror_y: tuple[int, ...] = (0, 1, 4, 3, 2, 8, 7, 6, 5)


@dataclass(frozen=True)
class Obstacle:
    """Boolean mask defining solid regions on the grid.

    Attributes:
        mask: Boolean array, shape (nx, ny). True where solid.
    """

    mask: Array


class LBMGrid:
    """2D Lattice Boltzmann simulation grid.

    Implements the D2Q9 BGK scheme for channel flow in x: a Zou-He
    velocity inlet at ``x = 0`` and a zero-gradient outlet at ``x = nx - 1``.
    Solid obstacles use full-way bounce-back. ``boundary`` selects the
    y edges: ``"periodic"``, ``"no_slip"`` (bounce-back walls on the first
    and last rows) or ``"free_slip"`` (specular-reflection walls).

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
        boundary: Boundary condition for the y edges (see above).
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
        inlet velocity (unless initial fields are given) and applies the
        BGK collision operator with bounce-back on solids.

        Args:
            n_steps: Number of time steps to simulate.
            u_inlet: Inlet velocity magnitude (lattice units). Should be
                << 1/sqrt(3) ~ 0.577 for stability.
            save_every: Save snapshots every N steps.
            initial_rho: Optional initial density, shape (nx, ny).
            initial_ux: Optional initial x-velocity, shape (nx, ny).
            initial_uy: Optional initial y-velocity, shape (nx, ny).

        Returns:
            FluidHistory with snapshots at lattice times
            ``t = k * save_every`` for ``k = 0 .. n_steps // save_every``.
            Velocities are zero on solid nodes.
        """
        if u_inlet >= 0.3:
            raise ConfigurationError(
                f"Inlet velocity {u_inlet} too high for LBM stability. "
                "Keep u_inlet << 0.577 (speed of sound). Recommended < 0.1."
            )
        if save_every < 1:
            raise ConfigurationError(f"save_every must be >= 1, got {save_every}")

        nx, ny = self._nx, self._ny
        logger.info(
            "Starting LBM simulation: grid=%dx%d, tau=%.3f, u_inlet=%.4f, n_steps=%d",
            nx,
            ny,
            self._tau,
            u_inlet,
            n_steps,
        )

        lattice = self._lattice
        cx, cy = lattice.c[:, 0], lattice.c[:, 1]
        rho = jnp.ones((nx, ny)) if initial_rho is None else initial_rho
        ux = jnp.full((nx, ny), u_inlet) if initial_ux is None else initial_ux
        uy = jnp.zeros((nx, ny)) if initial_uy is None else initial_uy

        obstacle = self._build_obstacle_mask()
        wall = jnp.zeros((nx, ny), dtype=bool)
        if self._config.boundary != "periodic":
            wall = wall.at[:, 0].set(True).at[:, -1].set(True)
        free_slip = self._config.boundary == "free_slip"
        solid = obstacle | wall

        f0 = _compute_equilibrium(rho, ux, uy, cx, cy, lattice.w)
        f0 = jnp.where(solid[..., None], lattice.w, f0)

        rho_h, ux_h, uy_h, vort_h = _lbm_rollout(
            f0,
            obstacle,
            wall,
            1.0 / self._tau,
            u_inlet,
            free_slip=free_slip,
            n_steps=n_steps,
            save_every=save_every,
        )
        return FluidHistory(
            t=save_every * jnp.arange(rho_h.shape[0], dtype=jnp.float64),
            rho=rho_h,
            ux=ux_h,
            uy=uy_h,
            vorticity=vort_h,
            grid_x=jnp.arange(nx, dtype=jnp.float64),
            grid_y=jnp.arange(ny, dtype=jnp.float64),
        )


# Lattice tables as static Python data (no device reads while tracing).
_C = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))


@partial(jax.jit, static_argnames=("free_slip", "n_steps", "save_every"))
def _lbm_rollout(
    f0: Array,
    obstacle: Array,
    wall: Array,
    omega: float,
    u_inlet: float,
    *,
    free_slip: bool,
    n_steps: int,
    save_every: int,
) -> tuple[Array, Array, Array, Array]:
    """BGK rollout; returns (rho, ux, uy, vorticity) snapshots.

    ``f0`` has the public ``(nx, ny, 9)`` layout. Inside the loop the
    distribution is population-major, ``(9, nx, ny)``: each population is a
    contiguous plane, so streaming is nine plane rolls and the moments are
    explicit plane sums instead of length-9 minor-axis reductions and
    strided rolls, which dominated the step time on CPU.
    """
    w = D2Q9.w[:, None, None]
    cx = jnp.array([c[0] for c in _C])[:, None, None]
    cy = jnp.array([c[1] for c in _C])[:, None, None]
    opposite = np.asarray(D2Q9.opposite)
    mirror = np.asarray(D2Q9.mirror_y)
    solid = obstacle | wall
    # Solid nodes that bounce populations back (all of them unless the y
    # walls are free-slip, which reflect specularly instead).
    bounce = obstacle if free_slip else solid
    fluid_inlet = ~solid[0, :]

    def macroscopic(g: Array) -> tuple[Array, Array, Array]:
        rho = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8]
        ux = (g[1] - g[3] + g[5] - g[6] - g[7] + g[8]) / rho
        uy = (g[2] - g[4] + g[5] + g[6] - g[7] - g[8]) / rho
        return rho, ux, uy

    def step(g: Array) -> Array:
        rho, ux, uy = macroscopic(g)
        u_sq = ux**2 + uy**2
        cu = cx * ux + cy * uy
        g_eq = w * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
        g_post = g - omega * (g - g_eq)
        # Full-way bounce-back / specular reflection: the populations that
        # streamed into a solid node are sent back unchanged (no collision).
        g_post = jnp.where(bounce, g[opposite], g_post)
        if free_slip:
            g_post = jnp.where(wall & ~obstacle, g[mirror], g_post)
        g = jnp.stack(
            [jnp.roll(g_post[i], shift, axis=(0, 1)) for i, shift in enumerate(_C)]
        )

        # Zou-He velocity inlet on the fluid nodes of the left edge.
        # Directions: 0=rest, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE
        col = g[:, 0, :]
        rho_in = (col[0] + col[2] + col[4] + 2.0 * (col[3] + col[6] + col[7])) / (
            1.0 - u_inlet
        )
        shear = 0.5 * (col[2] - col[4])
        inlet = (
            col.at[1]
            .set(col[3] + (2.0 / 3.0) * rho_in * u_inlet)
            .at[5]
            .set(col[7] - shear + (1.0 / 6.0) * rho_in * u_inlet)
            .at[8]
            .set(col[6] + shear + (1.0 / 6.0) * rho_in * u_inlet)
        )
        g = g.at[:, 0, :].set(jnp.where(fluid_inlet, inlet, col))
        # Zero-gradient outlet on the right edge.
        return g.at[:, -1, :].set(g[:, -2, :])

    def observe(g: Array) -> tuple[Array, Array, Array, Array]:
        rho, ux, uy = macroscopic(g)
        ux = jnp.where(solid, 0.0, ux)
        uy = jnp.where(solid, 0.0, uy)
        duy_dx: Array = jnp.gradient(uy, axis=0)  # type: ignore[assignment]
        dux_dy: Array = jnp.gradient(ux, axis=1)  # type: ignore[assignment]
        vort = duy_dx - dux_dy
        return rho, ux, uy, vort

    out: tuple[Array, Array, Array, Array] = strided_rollout(
        step, jnp.moveaxis(f0, -1, 0), n_steps, save_every, observe
    )
    return out


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

    f_eq = (
        w
        * rho[..., jnp.newaxis]
        * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq[..., jnp.newaxis])
    )
    return f_eq


def _stream(f: Array) -> Array:
    """Stream each population one lattice step along its velocity.

    Uses periodic wrap-around (``jnp.roll``); the inlet, outlet and wall
    treatments overwrite the wrapped values where needed.

    Args:
        f: Distribution function, shape (nx, ny, 9).

    Returns:
        Streamed distribution, shape (nx, ny, 9).
    """
    shifts = [(int(sx), int(sy)) for sx, sy in np.asarray(D2Q9.c)]
    return jnp.stack(
        [jnp.roll(f[..., i], shift, axis=(0, 1)) for i, shift in enumerate(shifts)],
        axis=-1,
    )
