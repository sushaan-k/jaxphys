"""Time-dependent Schrodinger equation solver.

Solves the 1D time-dependent Schrodinger equation using the
split-operator Fourier method:

    i*hbar * dpsi/dt = H*psi = (-hbar^2/(2m) * d^2/dx^2 + V(x)) * psi

The split-operator method factorizes the time evolution operator:

    U(dt) = exp(-i*V*dt/(2*hbar)) * exp(-i*T*dt/hbar) * exp(-i*V*dt/(2*hbar))

where T is the kinetic energy operator applied in momentum space via FFT.
This is second-order accurate in dt and exactly unitary (norm-preserving).

References:
    - Feit, Fleck, Steiger. "Solution of the Schrodinger equation by a
      spectral method" (1982)
    - Griffiths. "Introduction to Quantum Mechanics" (2018), Ch. 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError
from neurosim.state import QuantumResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SquareBarrier:
    """Square potential barrier.

    V(x) = height for |x - center| < width/2, else 0.

    Attributes:
        height: Barrier height in energy units.
        width: Barrier width in spatial units.
        center: Barrier center position.
    """

    height: float
    width: float
    center: float

    def __call__(self, x: Array) -> Array:
        """Evaluate the potential at positions x."""
        return jnp.where(
            jnp.abs(x - self.center) < self.width / 2.0,
            self.height,
            0.0,
        )


@dataclass(frozen=True)
class HarmonicPotential:
    """Quantum harmonic oscillator potential.

    V(x) = 0.5 * k * (x - x0)^2

    Attributes:
        k: Spring constant.
        x0: Equilibrium position.
    """

    k: float
    x0: float = 0.0

    def __call__(self, x: Array) -> Array:
        """Evaluate the potential at positions x."""
        return 0.5 * self.k * (x - self.x0) ** 2


@dataclass(frozen=True)
class DoubleWellPotential:
    """Double-well potential for tunneling studies.

    V(x) = a * (x^2 - b)^2

    Attributes:
        a: Potential depth parameter.
        b: Well separation parameter.
    """

    a: float = 1.0
    b: float = 1.0

    def __call__(self, x: Array) -> Array:
        """Evaluate the potential at positions x."""
        return self.a * (x**2 - self.b) ** 2


@dataclass(frozen=True)
class GaussianWavepacket:
    """Gaussian wavepacket initial condition.

    psi(x) = (2*pi*sigma^2)^{-1/4} * exp(-(x-x0)^2 / (4*sigma^2)) * exp(i*k0*x)

    Attributes:
        x0: Center position.
        k0: Central wavenumber (determines mean momentum p = hbar * k0).
        sigma: Width parameter.
    """

    x0: float
    k0: float
    sigma: float

    def __call__(self, x: Array) -> Array:
        """Evaluate the wavepacket at positions x."""
        norm = (2.0 * jnp.pi * self.sigma**2) ** (-0.25)
        gaussian = jnp.exp(-((x - self.x0) ** 2) / (4.0 * self.sigma**2))
        phase = jnp.exp(1j * self.k0 * x)
        return cast(Array, norm * gaussian * phase)


def solve_schrodinger(
    psi0: GaussianWavepacket | Array,
    potential: SquareBarrier | HarmonicPotential | DoubleWellPotential,
    x_range: tuple[float, float] = (-10.0, 10.0),
    t_span: tuple[float, float] = (0.0, 10.0),
    n_points: int = 1000,
    dt: float = 0.01,
    hbar: float = 1.0,
    mass: float = 1.0,
    save_every: int = 10,
) -> QuantumResult:
    """Solve the 1D time-dependent Schrodinger equation.

    Uses the split-operator Fourier method for exact unitarity.

    Args:
        psi0: Initial wavefunction (callable or array).
        potential: Potential energy function V(x).
        x_range: Spatial domain (x_min, x_max).
        t_span: Time interval.
        n_points: Number of spatial grid points.
        dt: Time step.
        hbar: Reduced Planck constant.
        mass: Particle mass.
        save_every: Save wavefunction every N steps.

    Returns:
        QuantumResult with wavefunction history and diagnostics.

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    x_min, x_max = x_range
    if x_max <= x_min:
        raise ConfigurationError(f"x_max ({x_max}) must be > x_min ({x_min})")

    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    # Spatial grid
    dx = (x_max - x_min) / n_points
    x = jnp.linspace(x_min, x_max, n_points)

    # Momentum grid (for FFT)
    k = jnp.fft.fftfreq(n_points, d=dx) * 2.0 * jnp.pi

    # Potential on grid
    V = potential(x)

    # Initial wavefunction
    if callable(psi0):
        psi = psi0(x).astype(jnp.complex128)
    else:
        psi = jnp.asarray(psi0, dtype=jnp.complex128)

    # Normalize
    norm = jnp.sqrt(jnp.trapezoid(jnp.abs(psi) ** 2, x))
    psi = psi / norm

    # Split-operator propagators
    # V half-step: exp(-i * V * dt / (2 * hbar))
    exp_V_half = jnp.exp(-1j * V * dt / (2.0 * hbar))

    # T full step: exp(-i * hbar * k^2 * dt / (2 * mass))
    exp_T = jnp.exp(-1j * hbar * k**2 * dt / (2.0 * mass))

    logger.info(
        "Starting Schrodinger solver: n_points=%d, n_steps=%d, method=split_operator",
        n_points,
        n_steps,
    )

    def split_step(psi_c: Array, _: None) -> tuple[Array, Array]:
        """One split-operator time step."""
        # Half-step in V
        psi_v = exp_V_half * psi_c
        # Full step in T (momentum space)
        psi_k = jnp.fft.fft(psi_v)
        psi_k = exp_T * psi_k
        psi_x = jnp.fft.ifft(psi_k)
        # Half-step in V
        psi_new = exp_V_half * psi_x
        return psi_new, psi_new

    _, psi_history = jax.lax.scan(split_step, psi, None, length=n_steps)

    # Prepend initial state
    psi_history = jnp.concatenate([psi[None, :], psi_history], axis=0)
    t_array = jnp.linspace(t_start, t_end, n_steps + 1)

    # Subsample
    if save_every > 1:
        indices = jnp.arange(0, n_steps + 1, save_every)
        psi_history = psi_history[indices]
        t_array = t_array[indices]

    # Compute transmission coefficient (for barrier problems)
    # Fraction of probability density past the barrier center
    barrier_center = None
    if isinstance(potential, SquareBarrier) or hasattr(potential, "center"):
        barrier_center = potential.center

    transmission = None
    if barrier_center is not None:
        final_prob = jnp.abs(psi_history[-1]) ** 2
        mask = x > barrier_center + 1.0  # past the barrier
        transmission = float(jnp.trapezoid(final_prob * mask, x))

    return QuantumResult(
        t=t_array,
        psi=psi_history,
        x=x,
        potential=V,
        transmission_coefficient=transmission,
    )
