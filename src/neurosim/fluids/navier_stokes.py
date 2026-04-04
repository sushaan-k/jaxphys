"""Vorticity-streamfunction Navier-Stokes solver for 2D incompressible flow.

Solves the 2D incompressible Navier-Stokes equations in the
vorticity-streamfunction formulation:

    dw/dt + u * dw/dx + v * dw/dy = nu * (d²w/dx² + d²w/dy²)

    d²psi/dx² + d²psi/dy² = -w

where w is vorticity, psi is the streamfunction, and (u, v) are
velocity components derived from psi:

    u = dpsi/dy,   v = -dpsi/dx

The Poisson equation for psi is solved iteratively using Jacobi
relaxation. Time integration uses explicit Euler.

References:
    - Peyret & Taylor. "Computational Methods for Fluid Flow" (1983)
    - Chorin & Marsden. "A Mathematical Introduction to Fluid Mechanics" (2000)
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.config import FluidConfig
from neurosim.exceptions import ConfigurationError
from neurosim.state import FluidHistory

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
                Poisson solve.
            save_every: Save snapshots every N steps.
            initial_omega: Optional initial vorticity, shape (nx, ny).

        Returns:
            FluidHistory with velocity and vorticity snapshots.
        """
        nx, ny = self._nx, self._ny
        dx = self._dx
        nu = self._config.viscosity

        # CFL check
        max_velocity = max(abs(lid_velocity), 0.1)
        cfl = max_velocity * dt / dx
        if cfl > 0.5:
            raise ConfigurationError(
                f"CFL number {cfl:.3f} exceeds 0.5. Reduce dt or increase dx."
            )

        logger.info(
            "Starting Navier-Stokes: grid=%dx%d, nu=%.4f, dt=%.4f, n_steps=%d",
            nx, ny, nu, dt, n_steps,
        )

        if initial_omega is None:
            omega = jnp.zeros((nx, ny))
        else:
            omega = initial_omega

        psi = jnp.zeros((nx, ny))

        grid_x = jnp.arange(nx, dtype=jnp.float64) * dx
        grid_y = jnp.arange(ny, dtype=jnp.float64) * dx

        dx2 = dx * dx
        dt_nu_dx2 = dt * nu / dx2
        dt_over_2dx = dt / (2.0 * dx)

        def poisson_step(psi_in: Array, omega_in: Array) -> Array:
            """One Jacobi iteration for the Poisson equation."""
            psi_new = 0.25 * (
                jnp.roll(psi_in, 1, axis=0)
                + jnp.roll(psi_in, -1, axis=0)
                + jnp.roll(psi_in, 1, axis=1)
                + jnp.roll(psi_in, -1, axis=1)
                + dx2 * omega_in
            )
            # Enforce psi = 0 on all walls
            psi_new = psi_new.at[0, :].set(0.0)
            psi_new = psi_new.at[-1, :].set(0.0)
            psi_new = psi_new.at[:, 0].set(0.0)
            psi_new = psi_new.at[:, -1].set(0.0)
            return psi_new

        def solve_poisson(psi_in: Array, omega_in: Array) -> Array:
            """Solve Poisson equation with multiple Jacobi iterations."""

            def body(carry: tuple[Array, int], _: None) -> tuple[tuple[Array, int], None]:
                p, i = carry
                p = poisson_step(p, omega_in)
                return (p, i + 1), None

            (psi_out, _), _ = jax.lax.scan(
                body, (psi_in, 0), None, length=poisson_iters
            )
            return psi_out

        def step(
            carry: tuple[Array, Array, int],
            _: None,
        ) -> tuple[tuple[Array, Array, int], tuple[Array, Array, Array]]:
            omega_c, psi_c, step_idx = carry

            # Solve Poisson for streamfunction
            psi_c = solve_poisson(psi_c, omega_c)

            # Velocity from streamfunction
            u = (jnp.roll(psi_c, -1, axis=1) - jnp.roll(psi_c, 1, axis=1)) / (2.0 * dx)
            v = -(jnp.roll(psi_c, -1, axis=0) - jnp.roll(psi_c, 1, axis=0)) / (2.0 * dx)

            # Advection (central differences)
            domega_dx = (
                jnp.roll(omega_c, -1, axis=0) - jnp.roll(omega_c, 1, axis=0)
            ) / (2.0 * dx)
            domega_dy = (
                jnp.roll(omega_c, -1, axis=1) - jnp.roll(omega_c, 1, axis=1)
            ) / (2.0 * dx)

            # Diffusion (Laplacian)
            laplacian_omega = (
                jnp.roll(omega_c, 1, axis=0)
                + jnp.roll(omega_c, -1, axis=0)
                + jnp.roll(omega_c, 1, axis=1)
                + jnp.roll(omega_c, -1, axis=1)
                - 4.0 * omega_c
            ) / dx2

            # Time step
            omega_new = omega_c + dt * (
                -u * domega_dx - v * domega_dy + nu * laplacian_omega
            )

            # Boundary conditions: no-slip walls
            # Top wall (lid): omega from lid velocity
            omega_new = omega_new.at[:, -1].set(
                -2.0 * psi_c[:, -2] / dx2 - 2.0 * lid_velocity / dx
            )
            # Bottom wall
            omega_new = omega_new.at[:, 0].set(-2.0 * psi_c[:, 1] / dx2)
            # Left wall
            omega_new = omega_new.at[0, :].set(-2.0 * psi_c[1, :] / dx2)
            # Right wall
            omega_new = omega_new.at[-1, :].set(-2.0 * psi_c[-2, :] / dx2)

            # Recompute velocity for output
            u_out = (
                jnp.roll(psi_c, -1, axis=1) - jnp.roll(psi_c, 1, axis=1)
            ) / (2.0 * dx)
            v_out = -(
                jnp.roll(psi_c, -1, axis=0) - jnp.roll(psi_c, 1, axis=0)
            ) / (2.0 * dx)

            # Enforce no-slip on velocity
            u_out = u_out.at[0, :].set(0.0)
            u_out = u_out.at[-1, :].set(0.0)
            u_out = u_out.at[:, 0].set(0.0)
            u_out = u_out.at[:, -1].set(lid_velocity)
            v_out = v_out.at[0, :].set(0.0)
            v_out = v_out.at[-1, :].set(0.0)
            v_out = v_out.at[:, 0].set(0.0)
            v_out = v_out.at[:, -1].set(0.0)

            rho_out = jnp.ones((nx, ny))

            return (omega_new, psi_c, step_idx + 1), (rho_out, u_out, v_out)

        init = (omega, psi, 0)
        _, (rho_all, ux_all, uy_all) = jax.lax.scan(step, init, None, length=n_steps)

        # Subsample
        if save_every > 1:
            indices = jnp.arange(0, n_steps, save_every)
            rho_all = rho_all[indices]
            ux_all = ux_all[indices]
            uy_all = uy_all[indices]

        # Compute vorticity from saved velocity
        vort = jnp.gradient(uy_all, axis=1) - jnp.gradient(ux_all, axis=2)

        t = jnp.arange(rho_all.shape[0], dtype=jnp.float64) * save_every * dt

        return FluidHistory(
            t=t,
            rho=rho_all,
            ux=ux_all,
            uy=uy_all,
            vorticity=vort,
            grid_x=grid_x,
            grid_y=grid_y,
        )
