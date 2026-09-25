"""Vorticity-streamfunction Navier-Stokes solver for 2D incompressible flow.

Solves the 2D incompressible Navier-Stokes equations in the
vorticity-streamfunction formulation:

    dw/dt + u * dw/dx + v * dw/dy = nu * (d²w/dx² + d²w/dy²)

    d²psi/dx² + d²psi/dy² = -w

where w is vorticity, psi is the streamfunction, and (u, v) are
velocity components derived from psi:

    u = dpsi/dy,   v = -dpsi/dx

The Poisson equation for psi is solved iteratively using Jacobi
relaxation. Time integration uses explicit Euler; wall vorticity follows
Thom's formula.

References:
    - Peyret & Taylor. "Computational Methods for Fluid Flow" (1983)
    - Chorin & Marsden. "A Mathematical Introduction to Fluid Mechanics" (2000)
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys._rollout import strided_rollout
from jaxphys.config import FluidConfig
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import FluidHistory

logger = logging.getLogger(__name__)


class NavierStokesSolver:
    """2D vorticity-streamfunction Navier-Stokes solver.

    Solves incompressible 2D flow on a uniform grid using the
    vorticity-streamfunction formulation with explicit time stepping
    and Jacobi iteration for the Poisson equation.

    Example:
        >>> solver = NavierStokesSolver(size=(128, 128), viscosity=0.001)
        >>> # Lid-driven cavity: top wall moves at u=1
        >>> result = solver.simulate(
        ...     n_steps=10000, dt=0.001,
        ...     lid_velocity=1.0, save_every=500,
        ... )

    Args:
        size: Grid dimensions (nx, ny).
        viscosity: Kinematic viscosity.
        dx: Grid spacing. Default 1.0 (lattice units).
    """

    def __init__(
        self,
        size: tuple[int, int] = (128, 128),
        viscosity: float = 0.001,
        dx: float = 1.0,
    ) -> None:
        self._nx, self._ny = size
        self._dx = dx
        self._config = FluidConfig(
            viscosity=viscosity,
            method="navier_stokes",
            boundary="no_slip",
        )

        if self._nx < 4 or self._ny < 4:
            raise ConfigurationError(f"Grid must be at least 4x4, got {size}")

    @property
    def size(self) -> tuple[int, int]:
        """Grid dimensions."""
        return (self._nx, self._ny)

    def simulate(
        self,
        n_steps: int = 10000,
        dt: float = 0.001,
        lid_velocity: float = 1.0,
        poisson_iters: int = 50,
        save_every: int = 100,
        initial_omega: Array | None = None,
    ) -> FluidHistory:
        """Run the lid-driven cavity simulation.

        Args:
            n_steps: Number of time steps.
            dt: Time step size.
            lid_velocity: Velocity of the top lid (x-direction).
            poisson_iters: Number of Jacobi iterations per step for the
                Poisson solve (warm-started from the previous step).
            save_every: Save snapshots every N steps.
            initial_omega: Optional initial vorticity, shape (nx, ny).

        Returns:
            FluidHistory with snapshots at ``t = k * save_every * dt`` for
            ``k = 0 .. n_steps // save_every``. ``vorticity`` is the solver's
            vorticity field and ``rho`` is identically 1 (incompressible).

        Raises:
            ConfigurationError: If the advective CFL number exceeds 0.5, the
                explicit diffusion limit ``nu*dt/dx^2 <= 0.25`` is violated,
                or ``save_every < 1``.
        """
        nx, ny = self._nx, self._ny
        dx = self._dx
        nu = self._config.viscosity

        max_velocity = max(abs(lid_velocity), 0.1)
        cfl = max_velocity * dt / dx
        if cfl > 0.5:
            raise ConfigurationError(
                f"CFL number {cfl:.3f} exceeds 0.5. Reduce dt or increase dx."
            )
        diffusion = nu * dt / dx**2
        if diffusion > 0.25:
            raise ConfigurationError(
                f"Diffusion number nu*dt/dx^2 = {diffusion:.3f} exceeds 0.25 "
                "(explicit Euler limit). Reduce dt or increase dx."
            )
        if save_every < 1:
            raise ConfigurationError(f"save_every must be >= 1, got {save_every}")

        logger.info(
            "Starting Navier-Stokes: grid=%dx%d, nu=%.4f, dt=%.4f, n_steps=%d",
            nx,
            ny,
            nu,
            dt,
            n_steps,
        )

        omega0 = (
            jnp.zeros((nx, ny))
            if initial_omega is None
            else jnp.asarray(initial_omega, dtype=jnp.float64)
        )
        if omega0.shape != (nx, ny):
            raise ConfigurationError(
                f"initial_omega shape {omega0.shape} != grid size {(nx, ny)}"
            )

        omega, ux, uy = _vorticity_rollout(
            omega0,
            dx,
            dt,
            nu,
            lid_velocity,
            poisson_iters=poisson_iters,
            n_steps=n_steps,
            save_every=save_every,
        )
        return FluidHistory(
            t=dt * save_every * jnp.arange(omega.shape[0], dtype=jnp.float64),
            rho=jnp.ones_like(ux),
            ux=ux,
            uy=uy,
            vorticity=omega,
            grid_x=jnp.arange(nx, dtype=jnp.float64) * dx,
            grid_y=jnp.arange(ny, dtype=jnp.float64) * dx,
        )


def _solve_poisson(psi: Array, omega: Array, dx: float, n_iters: int) -> Array:
    """Jacobi iterations for laplacian(psi) = -omega with psi = 0 on walls."""

    def jacobi(_: int, p: Array) -> Array:
        interior = (
            0.25 * (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2])
            + 0.25 * dx**2 * omega[1:-1, 1:-1]
        )
        return jnp.zeros_like(p).at[1:-1, 1:-1].set(interior)

    out: Array = jax.lax.fori_loop(0, n_iters, jacobi, psi)
    return out


def _velocity(psi: Array, dx: float, lid_velocity: float) -> tuple[Array, Array]:
    """u = dpsi/dy, v = -dpsi/dx (central differences) with wall values."""
    u = (
        jnp.zeros_like(psi)
        .at[1:-1, 1:-1]
        .set((psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx))
    )
    v = (
        jnp.zeros_like(psi)
        .at[1:-1, 1:-1]
        .set(-(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx))
    )
    return u.at[:, -1].set(lid_velocity), v


@partial(jax.jit, static_argnames=("poisson_iters", "n_steps", "save_every"))
def _vorticity_rollout(
    omega0: Array,
    dx: float,
    dt: float,
    nu: float,
    lid_velocity: float,
    *,
    poisson_iters: int,
    n_steps: int,
    save_every: int,
) -> tuple[Array, Array, Array]:
    """Explicit-Euler vorticity transport; returns (omega, u, v) snapshots.

    The carry holds ``(omega, psi)`` with ``psi`` solved from ``omega``, so
    every snapshot is self-consistent.
    """

    def step(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        omega, psi = carry
        u, v = _velocity(psi, dx, lid_velocity)

        c = omega[1:-1, 1:-1]
        domega_dx = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * dx)
        domega_dy = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * dx)
        laplacian = (
            omega[2:, 1:-1] + omega[:-2, 1:-1] + omega[1:-1, 2:] + omega[1:-1, :-2]
        ) / dx**2 - 4.0 * c / dx**2
        rhs = -u[1:-1, 1:-1] * domega_dx - v[1:-1, 1:-1] * domega_dy + nu * laplacian
        omega = omega.at[1:-1, 1:-1].set(c + dt * rhs)

        # Thom's wall vorticity (no-slip walls, moving lid at the top).
        omega = omega.at[:, -1].set(-2.0 * psi[:, -2] / dx**2 - 2.0 * lid_velocity / dx)
        omega = omega.at[:, 0].set(-2.0 * psi[:, 1] / dx**2)
        omega = omega.at[0, :].set(-2.0 * psi[1, :] / dx**2)
        omega = omega.at[-1, :].set(-2.0 * psi[-2, :] / dx**2)
        return omega, _solve_poisson(psi, omega, dx, poisson_iters)

    def observe(carry: tuple[Array, Array]) -> tuple[Array, Array, Array]:
        omega, psi = carry
        return (omega, *_velocity(psi, dx, lid_velocity))

    psi0 = _solve_poisson(jnp.zeros_like(omega0), omega0, dx, poisson_iters)
    out: tuple[Array, Array, Array] = strided_rollout(
        step, (omega0, psi0), n_steps, save_every, observe
    )
    return out
