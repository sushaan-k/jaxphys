"""Spin chain dynamics.

Simulates quantum spin-1/2 chains using exact diagonalization.
Implements the Heisenberg model:

    H = -J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
        - h * sum_i Sz_i

where Sx, Sy, Sz are the Pauli spin-1/2 operators and the sums
run over nearest-neighbor pairs on a 1D chain.

The Hilbert space dimension is 2^N and the Hamiltonian is diagonalized as
a dense matrix, so chains are limited to N <= 14 (a 16384 x 16384 matrix).

References:
    - Sachdev. "Quantum Phase Transitions" (2011)
    - Schollwock. "The density-matrix renormalization group" (2005)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxphys.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_MAX_SITES = 14


def _spin_z(n_sites: int) -> np.ndarray:
    """Pauli-z eigenvalues ``z[s, i]`` of site ``i`` in basis state ``s``.

    Site 0 is the most significant bit, matching the Kronecker ordering
    ``I (x) ... (x) op_site (x) ... (x) I``; bit value 0 is spin up (z = +1).
    """
    states = np.arange(2**n_sites)[:, None]
    bits = (states >> (n_sites - 1 - np.arange(n_sites))) & 1
    return 1 - 2 * bits


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
    Limited to small chains (N <= 14): the dense 2^N x 2^N Hamiltonian
    takes 16 * 4^N bytes (4.3 GB at N = 14).

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
        if n_sites > _MAX_SITES:
            raise ConfigurationError(
                f"n_sites={n_sites} gives Hilbert space dim 2^{n_sites}="
                f"{2**n_sites}. Max supported is {_MAX_SITES} (dense diagonalization)."
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

        Built directly in the S^z product basis: each bond contributes
        ``z_i z_j`` on the diagonal and a spin exchange (matrix element 2 in
        units of the Pauli operators) between states whose spins i and j
        differ, so no 2^N x 2^N Kronecker intermediates are formed.

        Returns:
            Hamiltonian matrix, shape (2^N, 2^N).
        """
        N = self._n_sites
        z = _spin_z(N)
        states = np.arange(self._dim)
        n_bonds = N if self._periodic else N - 1

        # H = -J/4 sum_<ij> sigma_i . sigma_j - h/2 sum_i sigma^z_i
        diag = np.zeros(self._dim)
        rows, cols = [], []
        for i in range(n_bonds):
            j = (i + 1) % N
            diag += z[:, i] * z[:, j]
            flip = z[:, i] != z[:, j]
            rows.append(states[flip])
            cols.append(states[flip] ^ ((1 << (N - 1 - i)) | (1 << (N - 1 - j))))
        H = jnp.diag(-0.25 * self._J * jnp.asarray(diag) - 0.5 * self._h * z.sum(1))
        H = H.at[np.concatenate(rows), np.concatenate(cols)].add(-0.5 * self._J)
        return H.astype(jnp.complex128)

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

        # <S^z_total> / N for each state; S^z_total is diagonal in this basis.
        total_sz = 0.5 * _spin_z(self._n_sites).sum(axis=1)
        magnetization = (jnp.abs(states) ** 2 @ total_sz) / self._n_sites

        return SpinChainResult(
            energies=energies,
            states=states,
            n_sites=self._n_sites,
            magnetization=magnetization,
        )
