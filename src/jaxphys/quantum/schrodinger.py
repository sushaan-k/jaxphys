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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys._rollout import is_traced, strided_rollout
from jaxphys.exceptions import ConfigurationError
from jaxphys.state import QuantumResult, QuantumResult2D

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

    Uses the split-operator Fourier method for exact unitarity. The domain
    is periodic: ``x`` samples ``[x_min, x_max)`` with spacing
    ``(x_max - x_min) / n_points``.

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
        QuantumResult with the wavefunction at ``t_start + k*save_every*dt``.
        For potentials with a ``center`` (e.g. :class:`SquareBarrier`), the
        transmission coefficient is the final probability beyond
        ``center + width/2``.

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    x_min, x_max = x_range
    if x_max <= x_min:
        raise ConfigurationError(f"x_max ({x_max}) must be > x_min ({x_min})")

    t_start, t_end = t_span
    if dt <= 0:
        raise ConfigurationError(f"dt must be positive, got {dt}")
    if save_every < 1:
        raise ConfigurationError(f"save_every must be >= 1, got {save_every}")
    n_steps = int((t_end - t_start) / dt)

    # Periodic spatial grid: the FFT treats x_max as the image of x_min, so
    # the endpoint is excluded and dx = L / n_points exactly.
    x = jnp.linspace(x_min, x_max, n_points, endpoint=False)
    dx = (x_max - x_min) / n_points

    # Angular wavenumbers matching the FFT ordering.
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n_points, d=dx)

    V = jnp.asarray(potential(x))

    if callable(psi0):
        psi = jnp.asarray(psi0(x), dtype=jnp.complex128)
    else:
        psi = jnp.asarray(psi0, dtype=jnp.complex128)
    if psi.shape != (n_points,):
        raise ConfigurationError(f"psi0 must have shape ({n_points},), got {psi.shape}")

    # Normalize with the same discrete norm the propagator conserves.
    psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx)

    logger.info(
        "Starting Schrodinger solver: n_points=%d, n_steps=%d, method=split_operator",
        n_points,
        n_steps,
    )

    psi_history = _split_operator_rollout(
        psi, V, k**2, dt, hbar, mass, n_steps=n_steps, save_every=save_every
    )
    t_array = t_start + dt * jnp.arange(0, n_steps + 1, save_every)

    # Transmission coefficient for barrier problems: probability found past
    # the far edge of the barrier at the final saved time.
    transmission: float | Array | None = None
    center = getattr(potential, "center", None)
    if center is not None:
        edge = center + 0.5 * getattr(potential, "width", 0.0)
        final_prob = jnp.abs(psi_history[-1]) ** 2
        transmitted = jnp.sum(jnp.where(x > edge, final_prob, 0.0)) * dx
        transmission = transmitted if is_traced(transmitted) else float(transmitted)

    return QuantumResult(
        t=t_array,
        psi=psi_history,
        x=x,
        potential=V,
        transmission_coefficient=transmission,
    )


@dataclass(frozen=True)
class GaussianWavepacket2D:
    """Isotropic 2D Gaussian wavepacket initial condition.

    psi(x, y) = (2*pi*sigma^2)^{-1/2}
                * exp(-((x-x0)^2 + (y-y0)^2) / (4*sigma^2))
                * exp(i*(kx*x + ky*y))

    Attributes:
        x0: Center x position.
        y0: Center y position.
        kx: Central wavenumber along x.
        ky: Central wavenumber along y.
        sigma: Position standard deviation along each axis.
    """

    x0: float
    y0: float
    kx: float
    ky: float
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        """Evaluate the wavepacket on (broadcastable) coordinates x, y."""
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2
        norm = (2.0 * jnp.pi * self.sigma**2) ** (-0.5)
        phase = jnp.exp(1j * (self.kx * x + self.ky * y))
        return cast(Array, norm * jnp.exp(-r2 / (4.0 * self.sigma**2)) * phase)


def solve_schrodinger_2d(
    psi0: Callable[[Array, Array], Array] | Array,
    potential: Callable[[Array, Array], Array] | Array,
    x_range: tuple[float, float] = (-10.0, 10.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    n_points: tuple[int, int] = (128, 128),
    t_span: tuple[float, float] = (0.0, 1.0),
    dt: float = 0.01,
    hbar: float = 1.0,
    mass: float = 1.0,
    save_every: int = 10,
) -> QuantumResult2D:
    """Solve the 2D time-dependent Schrodinger equation.

    Same Strang split-operator scheme as :func:`solve_schrodinger`, with a
    2D FFT for the kinetic step: second order in ``dt``, spectrally accurate
    in space and exactly unitary. The domain is periodic in both directions
    (``x`` samples ``[x_min, x_max)``, likewise ``y``). The whole propagation
    is one compiled loop, so it runs under ``jax.jit`` and is differentiable
    with respect to array-valued ``psi0`` and ``potential``.

    Args:
        psi0: Initial wavefunction: a callable ``psi0(X, Y)`` evaluated on
            the ``indexing="ij"`` meshgrid, or an array of shape ``n_points``.
            It is normalized to unit probability.
        potential: ``V(X, Y)`` callable or array of shape ``n_points``.
        x_range: Domain ``(x_min, x_max)``.
        y_range: Domain ``(y_min, y_max)``.
        n_points: Grid size ``(nx, ny)``.
        t_span: Time interval.
        dt: Time step.
        hbar: Reduced Planck constant.
        mass: Particle mass.
        save_every: Save the wavefunction every N steps.

    Returns:
        QuantumResult2D with ``psi`` of shape ``(n_saved, nx, ny)`` at
        ``t_start + k*save_every*dt``.

    Raises:
        ConfigurationError: If the domain, time step or shapes are invalid.
    """
    (x_min, x_max), (y_min, y_max) = x_range, y_range
    if x_max <= x_min or y_max <= y_min:
        raise ConfigurationError(f"Invalid domain x={x_range}, y={y_range}")
    if dt <= 0:
        raise ConfigurationError(f"dt must be positive, got {dt}")
    if save_every < 1:
        raise ConfigurationError(f"save_every must be >= 1, got {save_every}")
    nx, ny = n_points
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    x = jnp.linspace(x_min, x_max, nx, endpoint=False)
    y = jnp.linspace(y_min, y_max, ny, endpoint=False)
    dx, dy = (x_max - x_min) / nx, (y_max - y_min) / ny
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2

    V = jnp.asarray(potential(X, Y) if callable(potential) else potential)
    psi = jnp.asarray(psi0(X, Y) if callable(psi0) else psi0, dtype=jnp.complex128)
    for name, arr in (("psi0", psi), ("potential", V)):
        if arr.shape != (nx, ny):
            raise ConfigurationError(
                f"{name} must have shape {(nx, ny)}, got {arr.shape}"
            )
    psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx * dy)

    logger.info(
        "Starting 2D Schrodinger solver: grid=%dx%d, n_steps=%d", nx, ny, n_steps
    )
    psi_history = _split_operator_rollout(
        psi, V, k2, dt, hbar, mass, n_steps=n_steps, save_every=save_every
    )
    return QuantumResult2D(
        t=t_start + dt * jnp.arange(0, n_steps + 1, save_every),
        psi=psi_history,
        x=x,
        y=y,
        potential=V,
    )


@partial(jax.jit, static_argnames=("n_steps", "save_every"))
def _split_operator_rollout(
    psi: Array,
    V: Array,
    k2: Array,
    dt: float,
    hbar: float,
    mass: float,
    *,
    n_steps: int,
    save_every: int,
) -> Array:
    """Strang-split propagation, saving every ``save_every`` steps (incl. t0).

    Works in any dimension: ``k2`` is ``|k|^2`` on the FFT grid of ``psi``.
    """
    exp_V_half = jnp.exp(-1j * V * dt / (2.0 * hbar))
    exp_T = jnp.exp(-1j * hbar * k2 * dt / (2.0 * mass))

    def split_step(psi_c: Array) -> Array:
        psi_k = exp_T * jnp.fft.fftn(exp_V_half * psi_c)
        return exp_V_half * jnp.fft.ifftn(psi_k)

    history: Array = strided_rollout(split_step, psi, n_steps, save_every, lambda p: p)
    return history
