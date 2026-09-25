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

from jaxphys.exceptions import ConfigurationError


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

    J_1 is evaluated with rational/asymptotic approximations accurate to
    ~1e-8 for all arguments (the Airy argument reaches hundreds for
    millimetre apertures at optical wavelengths).

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

    j1 = _bessel_j1(x)

    safe_x = jnp.where(jnp.abs(x) < 1e-15, 1.0, x)
    intensity = jnp.where(jnp.abs(x) < 1e-15, 1.0, (2.0 * j1 / safe_x) ** 2)

    return DiffractionResult(
        theta=theta,
        intensity=intensity,
        wavelength=wavelength,
    )


def _bessel_j1(x: Array) -> Array:
    """Bessel function of the first kind, order one.

    Rational approximation for |x| < 8 and the Hankel asymptotic form for
    |x| >= 8 (Numerical Recipes, 3rd ed., section 6.5); absolute error is
    below 1e-8 everywhere. A truncated power series is unusable here: it
    diverges catastrophically for |x| >~ 20.
    """
    ax = jnp.abs(x)
    y = x * x
    small = (
        x
        * (
            72362614232.0
            + y
            * (
                -7895059235.0
                + y
                * (
                    242396853.1
                    + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606)))
                )
            )
        )
        / (
            144725228442.0
            + y
            * (
                2300535178.0
                + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
            )
        )
    )

    big_x = jnp.where(ax < 8.0, 8.0, ax)  # keep the unused branch finite
    z = 8.0 / big_x
    z2 = z * z
    xx = big_x - 2.356194491
    p1 = 1.0 + z2 * (
        0.183105e-2
        + z2 * (-0.3516396496e-4 + z2 * (0.2457520174e-5 + z2 * (-0.240337019e-6)))
    )
    q1 = 0.04687499995 + z2 * (
        -0.2002690873e-3
        + z2 * (0.8449199096e-5 + z2 * (-0.88228987e-6 + z2 * 0.105787412e-6))
    )
    large = (
        jnp.sign(x)
        * jnp.sqrt(0.636619772 / big_x)
        * (jnp.cos(xx) * p1 - z * jnp.sin(xx) * q1)
    )
    return jnp.where(ax < 8.0, small, large)
