"""Tight-binding models on periodic lattices.

A model is a set of orbitals in a unit cell with on-site energies
``eps_i`` and hopping amplitudes ``t_ij(R) = <i, 0|H|j, R>`` between orbital
``i`` in the home cell and orbital ``j`` in the cell displaced by the
lattice vector ``R``. Each bond is listed once; its Hermitian conjugate is
added automatically. The Bloch Hamiltonian is

    H_ij(k) = eps_i delta_ij + sum_R t_ij(R) exp(i k . R) + h.c.

(the "lattice-vector" gauge; eigenvalues do not depend on the gauge).

Models are JAX pytrees whose on-site energies and hoppings are leaves, so
band energies can be differentiated with respect to model parameters and
evaluated under ``jax.jit``/``jax.vmap``.

References:
    - Ashcroft & Mermin. "Solid State Physics" (1976), Ch. 10
    - Castro Neto et al. "The electronic properties of graphene",
      Rev. Mod. Phys. 81, 109 (2009)
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxphys.exceptions import ConfigurationError


@dataclass(frozen=True)
class TightBinding:
    """Tight-binding model (a JAX pytree).

    Example:
        >>> chain = TightBinding.chain(t=1.0)
        >>> k = jnp.linspace(-jnp.pi, jnp.pi, 5)[:, None]
        >>> energies = chain.bands(k)[:, 0]  # -2 cos(k): 2, 0, -2, 0, 2

    Attributes:
        lattice_vectors: Primitive vectors as rows, shape (d, d).
        onsite: On-site energies, shape (n_orbitals,).
        amplitudes: Hopping amplitudes ``t_ij(R)``, shape (n_hoppings,).
        sources: Orbital ``i`` of each hopping.
        targets: Orbital ``j`` of each hopping.
        cells: Cell offset ``R`` of each hopping in lattice-vector units.
    """

    lattice_vectors: Array
    onsite: Array
    amplitudes: Array
    sources: tuple[int, ...] = field(default=())
    targets: tuple[int, ...] = field(default=())
    cells: tuple[tuple[int, ...], ...] = field(default=())

    def __post_init__(self) -> None:
        # Skip validation when JAX rebuilds the pytree with non-array leaves.
        if not isinstance(self.onsite, jax.Array | np.ndarray):
            return
        d = np.shape(self.lattice_vectors)
        if len(d) != 2 or d[0] != d[1]:
            raise ConfigurationError(f"lattice_vectors must be (d, d), got {d}")
        n_orb, n_hop = np.shape(self.onsite)[0], np.shape(self.amplitudes)[0]
        if not len(self.sources) == len(self.targets) == len(self.cells) == n_hop:
            raise ConfigurationError("One source, target and cell per amplitude")
        for i, j, cell in zip(self.sources, self.targets, self.cells, strict=True):
            if not (0 <= i < n_orb and 0 <= j < n_orb):
                raise ConfigurationError(f"Orbital index out of range in ({i}, {j})")
            if len(cell) != d[0]:
                raise ConfigurationError(f"Cell offset {cell} must have {d[0]} entries")
            if i == j and not any(cell):
                raise ConfigurationError(
                    f"Hopping ({i}, {j}, {cell}) is an on-site term; use onsite"
                )

    @classmethod
    def from_hoppings(
        cls,
        lattice_vectors: Sequence[Sequence[float]] | Array,
        onsite: Sequence[float] | Array,
        hoppings: Sequence[tuple[int, int, Sequence[int], float | complex]],
    ) -> TightBinding:
        """Build a model from ``(i, j, R, t_ij(R))`` tuples."""
        amplitudes = [t for *_, t in hoppings]
        dtype = (
            jnp.complex128 if any(isinstance(t, complex) for t in amplitudes) else None
        )
        return cls(
            lattice_vectors=jnp.asarray(lattice_vectors, dtype=jnp.float64),
            onsite=jnp.asarray(onsite, dtype=jnp.float64),
            amplitudes=jnp.asarray(amplitudes, dtype=dtype or jnp.float64),
            sources=tuple(int(h[0]) for h in hoppings),
            targets=tuple(int(h[1]) for h in hoppings),
            cells=tuple(tuple(int(c) for c in h[2]) for h in hoppings),
        )

    @classmethod
    def chain(cls, t: float = 1.0, onsite: float = 0.0, a: float = 1.0) -> TightBinding:
        """1D chain, ``E(k) = onsite - 2 t cos(k a)``."""
        return cls.from_hoppings([[a]], [onsite], [(0, 0, (1,), -t)])

    @classmethod
    def square_lattice(cls, t: float = 1.0, a: float = 1.0) -> TightBinding:
        """Square lattice, ``E(k) = -2 t (cos kx a + cos ky a)``."""
        return cls.from_hoppings(
            [[a, 0.0], [0.0, a]], [0.0], [(0, 0, (1, 0), -t), (0, 0, (0, 1), -t)]
        )

    @classmethod
    def honeycomb(cls, t: float = 1.0, a: float = 1.0) -> TightBinding:
        """Graphene-like honeycomb lattice with lattice constant ``a``.

        Two orbitals (A, B); ``E(k) = +- t |1 + exp(-i k.a1) + exp(-i k.a2)|``
        with Dirac points at the Brillouin-zone corners.
        """
        vectors = [[a, 0.0], [0.5 * a, 0.5 * np.sqrt(3.0) * a]]
        bonds = [(0, 1, (0, 0), -t), (0, 1, (-1, 0), -t), (0, 1, (0, -1), -t)]
        return cls.from_hoppings(vectors, [0.0, 0.0], bonds)

    @property
    def n_orbitals(self) -> int:
        """Number of orbitals per unit cell."""
        return int(self.onsite.shape[0])

    @property
    def dim(self) -> int:
        """Spatial dimension of the lattice."""
        return int(self.lattice_vectors.shape[0])

    @property
    def reciprocal_vectors(self) -> Array:
        """Reciprocal primitive vectors ``b_i`` (rows), ``a_i . b_j = 2 pi delta_ij``."""
        inverse: Array = jnp.linalg.inv(self.lattice_vectors)
        return 2.0 * jnp.pi * inverse.T

    def bloch_hamiltonian(self, k: Array) -> Array:
        """Bloch Hamiltonian ``H(k)`` for a Cartesian wavevector ``k`` (d,)."""
        n = self.n_orbitals
        H = jnp.diag(self.onsite.astype(jnp.complex128))
        if self.amplitudes.shape[0] == 0:
            return H
        R = jnp.asarray(self.cells, dtype=jnp.float64) @ self.lattice_vectors
        terms = self.amplitudes * jnp.exp(1j * (R @ jnp.asarray(k)))
        hop = jnp.zeros((n, n), dtype=jnp.complex128)
        hop = hop.at[np.array(self.sources), np.array(self.targets)].add(terms)
        return H + hop + hop.conj().T

    def bands(self, k_points: Array) -> Array:
        """Band energies (ascending) at each wavevector, shape (n_k, n_orbitals)."""
        k_points = jnp.atleast_2d(jnp.asarray(k_points, dtype=jnp.float64))
        if k_points.shape[-1] != self.dim:
            raise ConfigurationError(
                f"k_points must have {self.dim} components, got {k_points.shape[-1]}"
            )
        energies: Array = jax.vmap(
            lambda k: jnp.linalg.eigvalsh(self.bloch_hamiltonian(k))
        )(k_points)
        return energies

    def finite_hamiltonian(
        self, n_cells: Sequence[int], periodic: bool = False
    ) -> Array:
        """Real-space Hamiltonian of an ``n_cells`` supercell.

        Basis ordering is ``(cell index in C order, orbital)``. With
        ``periodic=False`` bonds leaving the supercell are dropped (open
        boundaries); with ``periodic=True`` they wrap around.

        Returns:
            Hermitian matrix of shape ``(N * n_orbitals, N * n_orbitals)``
            with ``N = prod(n_cells)``.
        """
        shape = tuple(int(n) for n in n_cells)
        if len(shape) != self.dim or min(shape) < 1:
            raise ConfigurationError(f"n_cells must be {self.dim} positive ints")
        n_orb = self.n_orbitals
        size = int(np.prod(shape)) * n_orb
        rows, cols, which = [], [], []
        for cell in itertools.product(*(range(n) for n in shape)):
            home = np.ravel_multi_index(cell, shape)
            for h, (i, j, R) in enumerate(
                zip(self.sources, self.targets, self.cells, strict=True)
            ):
                other = np.add(cell, R)
                if periodic:
                    other = np.mod(other, shape)
                elif np.any(other < 0) or np.any(other >= shape):
                    continue
                rows.append(home * n_orb + i)
                cols.append(np.ravel_multi_index(tuple(other), shape) * n_orb + j)
                which.append(h)
        onsite = jnp.tile(self.onsite, size // n_orb).astype(jnp.complex128)
        H = jnp.diag(onsite)
        if rows:
            hop = jnp.zeros((size, size), dtype=jnp.complex128)
            hop = hop.at[np.array(rows), np.array(cols)].add(
                self.amplitudes[np.array(which)]
            )
            H = H + hop + hop.conj().T
        return H


jax.tree_util.register_dataclass(
    TightBinding,
    data_fields=["lattice_vectors", "onsite", "amplitudes"],
    meta_fields=["sources", "targets", "cells"],
)


def k_path(
    points: Sequence[Sequence[float]] | Array, n_per_segment: int = 50
) -> tuple[Array, Array]:
    """Straight-line path through high-symmetry points for band plots.

    Args:
        points: Corner wavevectors, shape (n_points, d).
        n_per_segment: Samples per segment (the end point is included once).

    Returns:
        ``(k, distance)``: wavevectors of shape (n, d) and the cumulative
        path length at each, for use as the x-axis of a band plot.
    """
    corners = jnp.asarray(points, dtype=jnp.float64)
    if corners.ndim != 2 or corners.shape[0] < 2:
        raise ConfigurationError("k_path needs at least two points of shape (d,)")
    s = jnp.linspace(0.0, 1.0, n_per_segment, endpoint=False)[:, None]
    segments = [
        corners[i] + s * (corners[i + 1] - corners[i]) for i in range(len(corners) - 1)
    ]
    k = jnp.concatenate([*segments, corners[-1:]], axis=0)
    steps = jnp.linalg.norm(jnp.diff(k, axis=0), axis=-1)
    return k, jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps)])
