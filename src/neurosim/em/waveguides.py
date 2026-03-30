"""Rectangular waveguide mode analysis.

Computes TE and TM mode profiles, cutoff frequencies, and dispersion
relations for rectangular metallic waveguides.

For a rectangular waveguide of dimensions a x b (a > b):

    TE_mn cutoff: f_c = (c/2) * sqrt((m/a)^2 + (n/b)^2)
    TM_mn cutoff: same formula, but m >= 1 and n >= 1

    Propagation constant: beta = sqrt(k^2 - k_c^2)
    where k = omega/c and k_c = 2*pi*f_c / c

References:
    - Griffiths. "Introduction to Electrodynamics" (2017), Ch. 9
    - Pozar. "Microwave Engineering" (2012), Ch. 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError, PhysicsError

logger = logging.getLogger(__name__)

C0 = 299792458.0  # speed of light (m/s)


@dataclass(frozen=True)
class WaveguideMode:
    """A waveguide mode solution.

    Attributes:
        mode_type: "TE" or "TM".
        m: Mode index along x.
        n: Mode index along y.
        cutoff_frequency: Cutoff frequency in Hz.
        field_pattern: 2D field pattern, shape (nx, ny).
        x: x-coordinates, shape (nx,).
        y: y-coordinates, shape (ny,).
    """

    mode_type: str
    m: int
    n: int
    cutoff_frequency: float
    field_pattern: Array
    x: Array
    y: Array


class RectangularWaveguide:
    """Rectangular metallic waveguide analyzer.

    Computes mode profiles and dispersion for a waveguide of
    dimensions a x b with perfectly conducting walls.

    Example:
        >>> wg = RectangularWaveguide(a=0.02286, b=0.01016)  # WR-90
        >>> mode = wg.compute_mode("TE", 1, 0, n_points=100)
        >>> print(f"TE10 cutoff: {mode.cutoff_frequency/1e9:.3f} GHz")

    Args:
        a: Width (larger dimension) in meters.
        b: Height (smaller dimension) in meters.
    """

    def __init__(self, a: float, b: float) -> None:
        if a <= 0 or b <= 0:
            raise ConfigurationError(
                f"Waveguide dimensions must be positive: a={a}, b={b}"
            )
        self._a = a
        self._b = b

    @property
    def a(self) -> float:
        """Waveguide width."""
        return self._a

    @property
    def b(self) -> float:
        """Waveguide height."""
        return self._b

    def cutoff_frequency(self, m: int, n: int) -> float:
        """Compute cutoff frequency for mode (m, n).

        Args:
            m: Mode index along x (>= 0 for TE, >= 1 for TM).
            n: Mode index along y (>= 0 for TE, >= 1 for TM).

        Returns:
            Cutoff frequency in Hz.
        """
        return float((C0 / 2.0) * jnp.sqrt((m / self._a) ** 2 + (n / self._b) ** 2))

    def propagation_constant(self, frequency: float, m: int, n: int) -> float:
        """Compute propagation constant beta for a given mode and frequency.

        Args:
            frequency: Operating frequency in Hz.
            m: Mode index along x.
            n: Mode index along y.

        Returns:
            Propagation constant beta in rad/m.

        Raises:
            PhysicsError: If frequency is below cutoff (evanescent mode).
        """
        fc = self.cutoff_frequency(m, n)
        if frequency < fc:
            raise PhysicsError(
                f"Frequency {frequency:.3e} Hz is below cutoff "
                f"{fc:.3e} Hz for mode ({m},{n}). Mode is evanescent."
            )
        k = 2.0 * jnp.pi * frequency / C0
        kc = 2.0 * jnp.pi * fc / C0
        return float(jnp.sqrt(k**2 - kc**2))

    def compute_mode(
        self,
        mode_type: str,
        m: int,
        n: int,
        n_points: int = 100,
    ) -> WaveguideMode:
        """Compute the transverse field pattern for a waveguide mode.

        For TE modes, computes the Hz pattern:
            Hz = cos(m*pi*x/a) * cos(n*pi*y/b)

        For TM modes, computes the Ez pattern:
            Ez = sin(m*pi*x/a) * sin(n*pi*y/b)

        Args:
            mode_type: "TE" or "TM".
            m: Mode index along x.
            n: Mode index along y.
            n_points: Number of grid points per dimension.

        Returns:
            WaveguideMode with the field pattern.

        Raises:
            ConfigurationError: If mode indices are invalid.
        """
        if mode_type not in ("TE", "TM"):
            raise ConfigurationError(
                f"mode_type must be 'TE' or 'TM', got '{mode_type}'"
            )
        if mode_type == "TE" and m == 0 and n == 0:
            raise ConfigurationError("TE00 mode does not exist")
        if mode_type == "TM" and (m < 1 or n < 1):
            raise ConfigurationError(
                f"TM modes require m >= 1 and n >= 1, got ({m},{n})"
            )

        x = jnp.linspace(0, self._a, n_points)
        y = jnp.linspace(0, self._b, n_points)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        if mode_type == "TE":
            pattern = jnp.cos(m * jnp.pi * X / self._a) * jnp.cos(
                n * jnp.pi * Y / self._b
            )
        else:
            pattern = jnp.sin(m * jnp.pi * X / self._a) * jnp.sin(
                n * jnp.pi * Y / self._b
            )

        fc = self.cutoff_frequency(m, n)

        return WaveguideMode(
            mode_type=mode_type,
            m=m,
            n=n,
            cutoff_frequency=fc,
            field_pattern=pattern,
            x=x,
            y=y,
        )

    def dispersion_relation(
        self,
        m: int,
        n: int,
        frequencies: Array,
    ) -> Array:
        """Compute beta(f) for a range of frequencies.

        Returns NaN for frequencies below cutoff.

        Args:
            m: Mode index along x.
            n: Mode index along y.
            frequencies: Array of frequencies in Hz.

        Returns:
            Propagation constants, same shape as frequencies.
        """
        fc = self.cutoff_frequency(m, n)
        k = 2.0 * jnp.pi * frequencies / C0
        kc = 2.0 * jnp.pi * fc / C0
        beta_sq = k**2 - kc**2
        return jnp.where(beta_sq > 0, jnp.sqrt(beta_sq), jnp.nan)
