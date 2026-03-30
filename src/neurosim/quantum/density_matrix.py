"""Density matrix formalism for open quantum systems.

Implements the Lindblad master equation for modeling decoherence
and dissipation in quantum systems:

    drho/dt = -i/hbar * [H, rho] + sum_k gamma_k * D[L_k](rho)

where the dissipator superoperator is:
    D[L](rho) = L rho L^dag - 0.5 * {L^dag L, rho}

References:
    - Lindblad. "On the generators of quantum dynamical semigroups" (1976)
    - Breuer & Petruccione. "Theory of Open Quantum Systems" (2002)
    - Nielsen & Chuang. "Quantum Computation and Quantum Information" (2010)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DensityMatrix:
    """Density matrix state representation.

    Attributes:
        rho: Density matrix, shape (d, d). Must be Hermitian,
            positive semi-definite, with trace 1.
    """

    rho: Array

    def __post_init__(self) -> None:
        rho = jnp.asarray(self.rho, dtype=jnp.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ConfigurationError(
                f"Density matrix must be square, got shape {rho.shape}"
            )
        if not bool(jnp.all(jnp.isfinite(rho))):
            raise ConfigurationError("Density matrix contains non-finite values")

        hermitian_error = jnp.max(jnp.abs(rho - rho.conj().T))
        if float(hermitian_error) > 1e-8:
            raise ConfigurationError("Density matrix must be Hermitian")

        trace = jnp.trace(rho)
        if (
            float(jnp.abs(jnp.real(trace) - 1.0)) > 1e-8
            or float(jnp.abs(jnp.imag(trace))) > 1e-8
        ):
            raise ConfigurationError(
                f"Density matrix must have trace 1, got {complex(trace)}"
            )

        eigenvalues = jnp.linalg.eigvalsh(rho)
        min_eig = float(jnp.min(jnp.real(eigenvalues)))
        if min_eig < -1e-8:
            raise ConfigurationError(
                "Density matrix must be positive semi-definite "
                f"(minimum eigenvalue {min_eig:.3e})"
            )

        object.__setattr__(self, "rho", rho)

    @staticmethod
    def from_pure_state(psi: Array) -> DensityMatrix:
        """Create a density matrix from a pure state.

        rho = |psi><psi|

        Args:
            psi: State vector, shape (d,).

        Returns:
            DensityMatrix for the pure state.
        """
        psi = jnp.asarray(psi, dtype=jnp.complex128)
        psi = psi / jnp.linalg.norm(psi)
        rho = jnp.outer(psi, jnp.conj(psi))
        return DensityMatrix(rho=rho)

    @staticmethod
    def thermal_state(hamiltonian: Array, temperature: float) -> DensityMatrix:
        """Create a thermal equilibrium state.

        rho = exp(-H / kT) / Z

        where Z = Tr(exp(-H / kT)) is the partition function.

        Args:
            hamiltonian: Hamiltonian matrix, shape (d, d).
            temperature: Temperature (in units where kB = 1).

        Returns:
            Thermal density matrix.
        """
        if temperature <= 0:
            raise ConfigurationError(f"Temperature must be positive, got {temperature}")
        hamiltonian = jnp.asarray(hamiltonian, dtype=jnp.complex128)
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ConfigurationError(
                f"Hamiltonian must be square, got shape {hamiltonian.shape}"
            )
        if not bool(jnp.all(jnp.isfinite(hamiltonian))):
            raise ConfigurationError("Hamiltonian contains non-finite values")
        beta = 1.0 / temperature
        eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian)
        boltzmann = jnp.exp(-beta * (eigenvalues - eigenvalues[0]))
        Z = jnp.sum(boltzmann)
        rho = eigenvectors @ jnp.diag(boltzmann / Z) @ eigenvectors.conj().T
        return DensityMatrix(rho=rho)

    @property
    def dimension(self) -> int:
        """Hilbert space dimension."""
        return int(self.rho.shape[0])

    def purity(self) -> float:
        """Compute purity Tr(rho^2). Equals 1 for pure states."""
        return float(jnp.real(jnp.trace(self.rho @ self.rho)))

    def von_neumann_entropy(self) -> float:
        """Compute von Neumann entropy S = -Tr(rho ln rho).

        Returns:
            Entropy in nats.
        """
        eigenvalues = jnp.linalg.eigvalsh(self.rho)
        # Clip small negatives from numerical error
        eigenvalues = jnp.clip(eigenvalues, 1e-30, None)
        return float(-jnp.sum(eigenvalues * jnp.log(eigenvalues)))

    def expectation(self, operator: Array) -> complex:
        """Compute expectation value Tr(rho * O).

        Args:
            operator: Observable matrix, shape (d, d).

        Returns:
            Expectation value <O>.
        """
        return complex(jnp.trace(self.rho @ operator))


@dataclass(frozen=True)
class LindbladResult:
    """Result of Lindblad master equation evolution.

    Attributes:
        t: Time values, shape (n_steps,).
        rho: Density matrices at each time, shape (n_steps, d, d).
        purity: Purity at each time step, shape (n_steps,).
    """

    t: Array
    rho: Array
    purity: Array


def _lindblad_rhs(
    rho: Array,
    hamiltonian: Array,
    lindblad_ops: list[Array],
    rates: Array,
    hbar: float,
) -> Array:
    """Compute the right-hand side of the Lindblad master equation.

    drho/dt = -i/hbar [H, rho] + sum_k gamma_k D[L_k](rho)

    Args:
        rho: Current density matrix.
        hamiltonian: System Hamiltonian.
        lindblad_ops: List of Lindblad (jump) operators.
        rates: Dissipation rates for each operator.
        hbar: Reduced Planck constant.

    Returns:
        Time derivative of the density matrix.
    """
    # Unitary part: -i/hbar * [H, rho]
    commutator = hamiltonian @ rho - rho @ hamiltonian
    drho = -1j / hbar * commutator

    # Dissipative part
    for k, L in enumerate(lindblad_ops):
        Ldag = L.conj().T
        LdagL = Ldag @ L
        drho = drho + rates[k] * (L @ rho @ Ldag - 0.5 * (LdagL @ rho + rho @ LdagL))

    return drho


def lindblad_evolve(
    rho0: DensityMatrix,
    hamiltonian: Array,
    lindblad_ops: list[Array],
    rates: list[float] | Array,
    t_span: tuple[float, float] = (0.0, 10.0),
    dt: float = 0.01,
    hbar: float = 1.0,
    save_every: int = 1,
) -> LindbladResult:
    """Evolve a density matrix under the Lindblad master equation.

    Uses fourth-order Runge-Kutta for time integration.

    Args:
        rho0: Initial density matrix.
        hamiltonian: System Hamiltonian, shape (d, d).
        lindblad_ops: List of Lindblad operators.
        rates: Dissipation rate for each Lindblad operator.
        t_span: Time interval.
        dt: Time step.
        hbar: Reduced Planck constant.
        save_every: Save every N steps.

    Returns:
        LindbladResult with density matrix history.
    """
    rates_arr = jnp.asarray(rates)
    if len(lindblad_ops) != rates_arr.shape[0]:
        raise ConfigurationError(
            f"Number of Lindblad operators ({len(lindblad_ops)}) must match "
            f"number of rates ({rates_arr.shape[0]})"
        )
    if bool(jnp.any(rates_arr < 0)):
        raise ConfigurationError("Lindblad rates must be non-negative")

    hamiltonian = jnp.asarray(hamiltonian, dtype=jnp.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ConfigurationError(
            f"Hamiltonian must be square, got shape {hamiltonian.shape}"
        )

    dim = rho0.dimension
    if hamiltonian.shape != (dim, dim):
        raise ConfigurationError(
            f"Hamiltonian shape {hamiltonian.shape} must match density matrix "
            f"dimension {(dim, dim)}"
        )

    validated_ops: list[Array] = []
    for idx, op in enumerate(lindblad_ops):
        op_arr = jnp.asarray(op, dtype=jnp.complex128)
        if op_arr.shape != (dim, dim):
            raise ConfigurationError(
                f"Lindblad operator {idx} shape {op_arr.shape} must match "
                f"density matrix dimension {(dim, dim)}"
            )
        validated_ops.append(op_arr)

    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    logger.info(
        "Starting Lindblad evolution: dim=%d, n_steps=%d, n_lindblad_ops=%d",
        rho0.dimension,
        n_steps,
        len(lindblad_ops),
    )

    def rhs(rho: Array) -> Array:
        return _lindblad_rhs(rho, hamiltonian, validated_ops, rates_arr, hbar)

    def rk4_step(rho: Array, _: None) -> tuple[Array, tuple[Array, Array]]:
        k1 = rhs(rho)
        k2 = rhs(rho + 0.5 * dt * k1)
        k3 = rhs(rho + 0.5 * dt * k2)
        k4 = rhs(rho + dt * k3)
        rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        trace = jnp.trace(rho_new)
        rho_new = rho_new / trace
        purity: Array = jnp.real(jnp.trace(rho_new @ rho_new))
        return rho_new, (rho_new, purity)

    _, (rho_hist, purity_hist) = jax.lax.scan(rk4_step, rho0.rho, None, length=n_steps)

    # Prepend initial state
    purity0: Array = jnp.real(jnp.trace(rho0.rho @ rho0.rho))
    rho_hist = jnp.concatenate([rho0.rho[None, :, :], rho_hist], axis=0)
    purity_hist = jnp.concatenate([jnp.array([purity0]), purity_hist])
    t_array = jnp.linspace(t_start, t_end, n_steps + 1)

    if save_every > 1:
        indices = jnp.arange(0, n_steps + 1, save_every)
        rho_hist = rho_hist[indices]
        purity_hist = purity_hist[indices]
        t_array = t_array[indices]

    return LindbladResult(
        t=t_array,
        rho=rho_hist,
        purity=purity_hist,
    )
