"""Time-independent Schrodinger equation solver.

Solves the eigenvalue problem:
    H|psi> = E|psi>

where H = -hbar^2/(2m) * d^2/dx^2 + V(x)

Uses a finite-difference discretization of the kinetic energy operator
and JAX/numpy eigensolvers to find bound state energies and wavefunctions.

The Hamiltonian matrix in the position basis:
    H_ij = T_ij + V_i * delta_ij

where T uses the three-point stencil:
    T_ij = -hbar^2/(2m*dx^2) * (-2*delta_{i,j} + delta_{i,j+1} + delta_{i,j-1})

References:
    - Griffiths. "Introduction to Quantum Mechanics" (2018), Ch. 2
    - Numerical Recipes, Ch. 18 (eigenvalue problems)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EigenResult:
    """Result of a stationary Schrodinger equation solve.

    Attributes:
        energies: Eigenvalues (energy levels), shape (n_states,).
        wavefunctions: Eigenvectors (wavefunctions), shape (n_states, n_points).
        x: Spatial grid, shape (n_points,).
        potential: Potential on the grid, shape (n_points,).
    """

    energies: Array
    wavefunctions: Array
    x: Array
    potential: Array

    @property
    def n_states(self) -> int:
        """Number of eigenstates computed."""
        return int(self.energies.shape[0])


def solve_eigenvalue_problem(
    potential: Any,
    x_range: tuple[float, float] = (-10.0, 10.0),
    n_points: int = 500,
    n_states: int = 10,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> EigenResult:
    """Solve the time-independent Schrodinger equation.

    Finds the n_states lowest energy eigenvalues and eigenfunctions
    using finite-difference discretization.

    Args:
        potential: Callable V(x) -> potential energy, or array.
        x_range: Spatial domain (x_min, x_max).
        n_points: Number of grid points.
        n_states: Number of lowest eigenstates to compute.
        hbar: Reduced Planck constant.
        mass: Particle mass.

    Returns:
        EigenResult with energies and wavefunctions.

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    x_min, x_max = x_range
    if x_max <= x_min:
        raise ConfigurationError(f"x_max ({x_max}) must be > x_min ({x_min})")
    if n_states > n_points:
        raise ConfigurationError(
            f"n_states ({n_states}) cannot exceed n_points ({n_points})"
        )

    x = jnp.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]

    # Potential on grid
    V = potential(x) if callable(potential) else jnp.asarray(potential)

    logger.info(
        "Solving eigenvalue problem: n_points=%d, n_states=%d",
        n_points,
        n_states,
    )

    # Build Hamiltonian matrix using three-point stencil
    prefactor = -(hbar**2) / (2.0 * mass * dx**2)

    # Kinetic energy: tridiagonal matrix
    diag = -2.0 * prefactor * jnp.ones(n_points) + V
    off_diag = prefactor * jnp.ones(n_points - 1)

    H = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)

    # Select lowest n_states
    energies = eigenvalues[:n_states]
    wavefunctions = eigenvectors[:, :n_states].T  # shape (n_states, n_points)

    # Normalize each wavefunction
    for i in range(n_states):
        norm = jnp.sqrt(jnp.trapezoid(jnp.abs(wavefunctions[i]) ** 2, x))
        wavefunctions = wavefunctions.at[i].set(wavefunctions[i] / norm)

    return EigenResult(
        energies=energies,
        wavefunctions=wavefunctions,
        x=x,
        potential=V,
    )
