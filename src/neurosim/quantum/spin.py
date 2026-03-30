"""Spin chain dynamics.

Simulates quantum spin-1/2 chains using exact diagonalization.
Implements the Heisenberg model:

    H = -J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
        - h * sum_i Sz_i

where Sx, Sy, Sz are the Pauli spin-1/2 operators and the sums
run over nearest-neighbor pairs on a 1D chain.

The Hilbert space dimension is 2^N, so exact methods are limited
to chains of length N ~ 20.

References:
    - Sachdev. "Quantum Phase Transitions" (2011)
    - Schollwock. "The density-matrix renormalization group" (2005)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Determine complex dtype based on x64 mode
_X64_ENABLED = bool(getattr(jax.config, "jax_enable_x64", False))
_COMPLEX_DTYPE = jnp.complex128 if _X64_ENABLED else jnp.complex64


def _pauli_matrices() -> tuple[Array, Array, Array, Array]:
    """Construct Pauli matrices using the current dtype setting."""
    dtype = jnp.complex128 if _X64_ENABLED else jnp.complex64
    sx = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    sy = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=dtype)
    sz = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    ident = jnp.eye(2, dtype=dtype)
    return sx, sy, sz, ident


# Module-level references (will be recomputed if needed)
SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY_2 = _pauli_matrices()


def _tensor_product_operator(op: Array, site: int, n_sites: int) -> Array:
    """Embed a single-site operator into the full Hilbert space.

    Constructs I_1 (x) ... (x) op_site (x) ... (x) I_N.

    Args:
        op: Single-site operator, shape (2, 2).
        site: Site index (0-based).
        n_sites: Total number of sites.

    Returns:
        Full-space operator, shape (2^N, 2^N).
    """
    result = jnp.array([[1.0]], dtype=jnp.complex128)
    for i in range(n_sites):
        m = op if i == site else IDENTITY_2
        result = jnp.kron(result, m)
    return result


@dataclass(frozen=True)
class SpinChainResult:
    """Result of a spin chain computation.

    Attributes:
        energies: Energy eigenvalues, shape (n_states,).
        states: Eigenstates, shape (n_states, 2^N).
        n_sites: Number of spin sites.
        magnetization: Expectation value of total Sz per site.
    """

    energies: Array
    states: Array
    n_sites: int
    magnetization: Array


class SpinChain:
    """Quantum spin-1/2 chain with Heisenberg interactions.

    Builds the full Hamiltonian matrix via exact diagonalization.
    Limited to small chains (N <= 16) due to exponential Hilbert space.

    Example:
        >>> chain = SpinChain(n_sites=8, J=1.0, h=0.0)
        >>> result = chain.diagonalize(n_states=10)
        >>> print(f"Ground state energy: {result.energies[0]:.6f}")

    Args:
        n_sites: Number of spin-1/2 sites.
        J: Heisenberg coupling constant. J > 0 is ferromagnetic.
        h: External magnetic field strength along z.
        periodic: Whether to use periodic boundary conditions.
    """

    def __init__(
        self,
        n_sites: int,
        J: float = 1.0,
        h: float = 0.0,
        periodic: bool = False,
    ) -> None:
        if n_sites < 2:
            raise ConfigurationError(f"n_sites must be >= 2, got {n_sites}")
        if n_sites > 16:
            raise ConfigurationError(
                f"n_sites={n_sites} gives Hilbert space dim 2^{n_sites}="
                f"{2**n_sites}. Max supported is 16 (65536 states)."
            )
        self._n_sites = n_sites
        self._J = J
        self._h = h
        self._periodic = periodic
        self._dim = 2**n_sites

    @property
    def n_sites(self) -> int:
        """Number of spin sites."""
        return self._n_sites

    @property
    def hilbert_dim(self) -> int:
        """Dimension of the Hilbert space (2^N)."""
        return int(self._dim)

    def build_hamiltonian(self) -> Array:
        """Construct the full Hamiltonian matrix.

        Returns:
            Hamiltonian matrix, shape (2^N, 2^N).
        """
        N = self._n_sites
        dim = self._dim
        H = jnp.zeros((dim, dim), dtype=jnp.complex128)

        # Heisenberg interaction: -J * sum (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
        n_bonds = N if self._periodic else N - 1
        for i in range(n_bonds):
            j = (i + 1) % N
            for sigma in [SIGMA_X, SIGMA_Y, SIGMA_Z]:
                Si = _tensor_product_operator(sigma, i, N)
                Sj = _tensor_product_operator(sigma, j, N)
                H = H - self._J * 0.25 * Si @ Sj  # Factor of 1/4 for spin-1/2

        # External field: -h * sum Sz_i
        if self._h != 0.0:
            for i in range(N):
                Szi = _tensor_product_operator(SIGMA_Z, i, N)
                H = H - self._h * 0.5 * Szi  # Factor of 1/2 for spin-1/2

        return H

    def diagonalize(self, n_states: int = 10) -> SpinChainResult:
        """Diagonalize the Hamiltonian and return lowest eigenstates.

        Args:
            n_states: Number of lowest eigenstates to return.

        Returns:
            SpinChainResult with energies and states.
        """
        if n_states > self._dim:
            n_states = self._dim

        logger.info(
            "Diagonalizing spin chain: N=%d, dim=%d, J=%.2f, h=%.2f",
            self._n_sites,
            self._dim,
            self._J,
            self._h,
        )

        H = self.build_hamiltonian()
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)

        energies = eigenvalues[:n_states]
        states = eigenvectors[:, :n_states].T  # (n_states, dim)

        # Compute magnetization per site for each state
        total_Sz = jnp.zeros((self._dim, self._dim), dtype=jnp.complex128)
        for i in range(self._n_sites):
            total_Sz = total_Sz + 0.5 * _tensor_product_operator(
                SIGMA_Z, i, self._n_sites
            )

        magnetization = jnp.array(
            [
                jnp.real(state.conj() @ total_Sz @ state) / self._n_sites
                for state in states
            ]
        )

        return SpinChainResult(
            energies=energies,
            states=states,
            n_sites=self._n_sites,
            magnetization=magnetization,
        )
