"""Wave optics: Fraunhofer diffraction patterns.

Computes far-field (Fraunhofer) diffraction patterns for standard
apertures using the Fourier transform relationship between the
aperture function and the diffracted field.

The Fraunhofer diffraction integral gives:
    U(x) ~ FT{aperture(x')} evaluated at fx = x / (lambda * z)

For a single slit of width a:
    I(theta) = I_0 * sinc^2(pi * a * sin(theta) / lambda)

For a double slit (width a, separation d):
    I(theta) = I_0 * sinc^2(pi*a*sin(theta)/lambda) * cos^2(pi*d*sin(theta)/lambda)

References:
    - Hecht. "Optics" (2017), Ch. 10
    - Goodman. "Introduction to Fourier Optics" (2017)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError


@dataclass(frozen=True)
class DiffractionResult:
    """Result of a diffraction calculation.

    Attributes:
        theta: Diffraction angles in radians.
        intensity: Normalized intensity pattern.
        wavelength: Wavelength used.
    """

    theta: Array
    intensity: Array
    wavelength: float

    @property
    def angle_degrees(self) -> Array:
        """Angles in degrees."""
        return jnp.degrees(self.theta)


def single_slit(
    slit_width: float,
    wavelength: float,
    n_points: int = 1000,
    theta_max: float = 0.1,
) -> DiffractionResult:
    """Compute Fraunhofer diffraction pattern from a single slit.

    I(theta) = I_0 * [sin(beta)/beta]^2
    where beta = pi * a * sin(theta) / lambda

    Args:
        slit_width: Slit width in meters.
        wavelength: Light wavelength in meters.
        n_points: Number of angle points.
        theta_max: Maximum angle in radians.

    Returns:
        DiffractionResult with intensity pattern.

    Raises:
        ConfigurationError: If parameters are non-positive.
    """
    if slit_width <= 0:
        raise ConfigurationError(f"Slit width must be positive, got {slit_width}")
    if wavelength <= 0:
        raise ConfigurationError(f"Wavelength must be positive, got {wavelength}")

    theta = jnp.linspace(-theta_max, theta_max, n_points)
    beta = jnp.pi * slit_width * jnp.sin(theta) / wavelength

    # sinc function: sin(x)/x, handling x=0
    intensity = jnp.where(
        jnp.abs(beta) < 1e-15,
        1.0,
        (jnp.sin(beta) / beta) ** 2,
    )

    return DiffractionResult(
        theta=theta,
        intensity=intensity,
        wavelength=wavelength,
    )


def double_slit(
    slit_width: float,
    slit_separation: float,
    wavelength: float,
    n_points: int = 1000,
    theta_max: float = 0.1,
) -> DiffractionResult:
    """Compute Fraunhofer diffraction pattern from a double slit.

    I(theta) = I_0 * sinc^2(beta) * cos^2(delta)
    where beta = pi * a * sin(theta) / lambda
          delta = pi * d * sin(theta) / lambda

    Args:
        slit_width: Individual slit width in meters.
        slit_separation: Center-to-center separation in meters.
        wavelength: Light wavelength in meters.
        n_points: Number of angle points.
        theta_max: Maximum angle in radians.

    Returns:
        DiffractionResult with intensity pattern.
    """
    if slit_width <= 0 or slit_separation <= 0 or wavelength <= 0:
        raise ConfigurationError("All parameters must be positive")
    if slit_separation < slit_width:
        raise ConfigurationError("Slit separation must be >= slit width")

    theta = jnp.linspace(-theta_max, theta_max, n_points)

    # Single-slit envelope
    beta = jnp.pi * slit_width * jnp.sin(theta) / wavelength
    envelope = jnp.where(
        jnp.abs(beta) < 1e-15,
        1.0,
        (jnp.sin(beta) / beta) ** 2,
    )

    # Double-slit interference
    delta = jnp.pi * slit_separation * jnp.sin(theta) / wavelength
    interference = jnp.cos(delta) ** 2

    intensity = envelope * interference

    return DiffractionResult(
        theta=theta,
        intensity=intensity,
        wavelength=wavelength,
    )


def circular_aperture(
    diameter: float,
    wavelength: float,
    n_points: int = 1000,
    theta_max: float = 0.05,
) -> DiffractionResult:
    """Compute Airy diffraction pattern from a circular aperture.

    I(theta) = I_0 * [2 * J_1(x) / x]^2
    where x = pi * D * sin(theta) / lambda

    Uses a polynomial approximation to J_1 for JAX compatibility.

    Args:
        diameter: Aperture diameter in meters.
        wavelength: Light wavelength in meters.
        n_points: Number of angle points.
        theta_max: Maximum angle in radians.

    Returns:
        DiffractionResult with Airy pattern.
    """
    if diameter <= 0 or wavelength <= 0:
        raise ConfigurationError("Diameter and wavelength must be positive")

    theta = jnp.linspace(-theta_max, theta_max, n_points)
    x = jnp.pi * diameter * jnp.sin(theta) / wavelength

    # Bessel J_1 approximation using series expansion
    # J_1(z) = z/2 - z^3/16 + z^5/384 - z^7/18432 + ...
    # For small x, use series; for larger x, use jnp.
    # JAX doesn't have jnp.j1, so we use the series for moderate x.
    def j1_approx(z: Array) -> Array:
        """Bessel J_1 via power series (converges for |z| < ~15)."""
        result = jnp.zeros_like(z)
        term = z / 2.0
        result = result + term
        for k in range(1, 20):
            term = term * (-(z**2)) / (4.0 * k * (k + 1))
            result = result + term
        return result

    j1 = j1_approx(x)

    intensity = jnp.where(
        jnp.abs(x) < 1e-15,
        1.0,
        (2.0 * j1 / x) ** 2,
    )

    return DiffractionResult(
        theta=theta,
        intensity=intensity,
        wavelength=wavelength,
    )
