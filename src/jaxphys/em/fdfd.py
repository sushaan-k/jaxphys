"""Finite-Difference Frequency-Domain (FDFD) Maxwell solver.

Solves the 2D TM (Ez) Helmholtz equation for a time-harmonic source
``Jz exp(-i omega t)``:

    d/dx (1/s_x) d/dx Ez / s_x + d/dy (1/s_y) d/dy Ez / s_y
        + k0^2 eps_r Ez = -i omega mu0 Jz

on the Yee grid (Ez on integer nodes, derivatives on half nodes). The
outer ``pml_layers`` cells form a stretched-coordinate PML with
``s = 1 + i sigma / (omega eps0)`` and the same graded conductivity
profile as the FDTD solver; the domain is terminated by Ez = 0.

The 5-point system is block tridiagonal (one block per grid column), so
it is solved exactly by block LU elimination: a ``lax.scan`` of dense
``n x n`` solves with ``n = min(nx, ny)``, costing O(max * min^3). The
solver is pure JAX: it runs under ``jax.jit`` and ``jax.vmap`` and is
differentiable with respect to ``eps_r`` and the source, which makes it
suitable for inverse design.

References:
    - Shin & Fan. "Choice of the perfectly matched layer boundary condition
      for frequency-domain Maxwell's equations solvers", JCP 231 (2012)
    - Hughes et al. "Adjoint method and inverse design for nonlinear
      nanophotonic devices", ACS Photonics 5 (2018)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxphys.em.fdtd import EPS0, MU0, _pml_loss
from jaxphys.exceptions import ConfigurationError


@dataclass(frozen=True)
class FDFDResult:
    """Steady-state field of an FDFD solve.

    Attributes:
        ez: Complex Ez phasor, shape (nx, ny).
        grid_x: x coordinates of the Ez nodes, shape (nx,).
        grid_y: y coordinates of the Ez nodes, shape (ny,).
        frequency: Source frequency in Hz.
    """

    ez: Array
    grid_x: Array
    grid_y: Array
    frequency: float

    @property
    def intensity(self) -> Array:
        """Time-averaged ``|Ez|^2``."""
        return jnp.abs(self.ez) ** 2


def solve_fdfd(
    eps_r: Array,
    source: Array,
    frequency: float,
    resolution: float,
    pml_layers: int = 10,
) -> FDFDResult:
    """Solve for the steady-state TM field driven by a harmonic current.

    Example:
        >>> eps = jnp.ones((120, 120))
        >>> J = jnp.zeros((120, 120)).at[60, 60].set(1.0)
        >>> field = solve_fdfd(eps, J, frequency=3e9, resolution=0.004)

    Args:
        eps_r: Relative permittivity on the Ez nodes, shape (nx, ny). May be
            complex (``Im eps_r > 0`` is loss for this time convention).
        source: Current density Jz (A/m^2) on the Ez nodes, shape (nx, ny).
            A point current ``I`` is ``I / resolution**2`` at one node.
        frequency: Frequency in Hz.
        resolution: Cell size dx = dy in meters.
        pml_layers: PML thickness in cells on every side.

    Returns:
        FDFDResult with the complex Ez phasor.

    Raises:
        ConfigurationError: If shapes or parameters are invalid.
    """
    # Explicit dtypes drop weak typing (e.g. from jnp.full(shape, 2.0)), so
    # equal-shaped inputs always reuse the same compiled operations.
    eps_r = jnp.asarray(eps_r)
    eps_r = jnp.asarray(eps_r, dtype=eps_r.dtype)
    source = jnp.asarray(source)
    source = jnp.asarray(source, dtype=source.dtype)
    if eps_r.ndim != 2 or source.shape != eps_r.shape:
        raise ConfigurationError(
            f"eps_r and source must be 2D with equal shapes, got "
            f"{eps_r.shape} and {source.shape}"
        )
    nx, ny = eps_r.shape
    if min(nx, ny) <= 2 * pml_layers + 2:
        raise ConfigurationError(
            f"Grid {eps_r.shape} is too small for {pml_layers} PML layers"
        )
    if frequency <= 0 or resolution <= 0:
        raise ConfigurationError("frequency and resolution must be positive")

    omega = 2.0 * np.pi * frequency
    k0_sq = omega**2 * MU0 * EPS0

    # Stretch factors s = 1 + i sigma/(omega eps0) on nodes and half nodes
    # (_pml_loss with dt = 1 returns sigma / eps0).
    def stretch(n: int) -> tuple[Array, Array]:
        node, half = _pml_loss(n, pml_layers, resolution, 1.0)
        return 1.0 + 1j * node / omega, 1.0 + 1j * half / omega

    sx_e, sx_h = stretch(nx)
    sy_e, sy_h = stretch(ny)
    inv_dx2 = 1.0 / resolution**2
    # Coupling to the lower / upper neighbour along each axis; the half-node
    # stretch at i - 1/2 is stored at index i - 1.
    x_lo = inv_dx2 / (sx_e * jnp.concatenate([sx_h[:1], sx_h[:-1]]))
    x_hi = inv_dx2 / (sx_e * sx_h)
    y_lo = inv_dx2 / (sy_e * jnp.concatenate([sy_h[:1], sy_h[:-1]]))
    y_hi = inv_dx2 / (sy_e * sy_h)
    X_lo, X_hi = x_lo[:, None], x_hi[:, None]
    Y_lo, Y_hi = y_lo[None, :], y_hi[None, :]
    center = k0_sq * eps_r - (X_lo + X_hi + Y_lo + Y_hi)
    rhs = (-1j * omega * MU0 * source).astype(jnp.complex128)
    ez = _solve_block_tridiagonal(
        jnp.broadcast_to(center, (nx, ny)).astype(jnp.complex128),
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        rhs,
    )
    return FDFDResult(
        ez=ez,
        grid_x=jnp.arange(nx) * resolution,
        grid_y=jnp.arange(ny) * resolution,
        frequency=frequency,
    )


@jax.jit
def _solve_block_tridiagonal(
    center: Array, x_lo: Array, x_hi: Array, y_lo: Array, y_hi: Array, rhs: Array
) -> Array:
    """Solve the 5-point system ``A u = rhs`` on an (nx, ny) grid.

    ``A u = center*u + x_lo*u[i-1] + x_hi*u[i+1] + y_lo*u[j-1] + y_hi*u[j+1]``
    with zero values outside the grid. Eliminates along the longer axis so
    that the dense blocks have the size of the shorter one.
    """
    if center.shape[1] > center.shape[0]:
        transposed: Array = _solve_block_tridiagonal(
            center.T, y_lo, y_hi, x_lo, x_hi, rhs.T
        )
        return transposed.T
    n = center.shape[1]
    eye = jnp.eye(n, dtype=center.dtype)

    def column_block(i: Array) -> Array:
        # Tridiagonal coupling within column i (along the second axis).
        return (
            jnp.diag(center[i])
            + jnp.diag(y_hi[:-1], 1).astype(center.dtype)
            + jnp.diag(y_lo[1:], -1).astype(center.dtype)
        )

    def forward(
        carry: tuple[Array, Array], i: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        prev_c, prev_d = carry
        m = column_block(i) - x_lo[i] * prev_c
        sol = jnp.linalg.solve(
            m, jnp.concatenate([x_hi[i] * eye, (rhs[i] - x_lo[i] * prev_d)[:, None]], 1)
        )
        c, d = sol[:, :n], sol[:, n]
        return (c, d), (c, d)

    zeros_c = jnp.zeros((n, n), dtype=center.dtype)
    zeros_d = jnp.zeros(n, dtype=center.dtype)
    _, (cs, ds) = jax.lax.scan(forward, (zeros_c, zeros_d), jnp.arange(center.shape[0]))

    def backward(next_u: Array, cd: tuple[Array, Array]) -> tuple[Array, Array]:
        c, d = cd
        u = d - c @ next_u
        return u, u

    _, u = jax.lax.scan(backward, zeros_d, (cs, ds), reverse=True)
    return u
