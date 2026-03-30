"""Geometric optics ray tracing.

Implements paraxial ray tracing through optical systems using
the ABCD (ray transfer) matrix formalism. A ray is characterized
by its height y and angle theta at each optical plane.

The ray transfer matrix relates input to output:
    [y_out]     [A  B] [y_in ]
    [     ]  =  [    ] [     ]
    [t_out]     [C  D] [t_in ]

where t = n * theta (reduced angle).

Standard ABCD matrices:
    Free space (d):      [[1, d], [0, 1]]
    Thin lens (f):       [[1, 0], [-1/f, 1]]
    Flat mirror:         [[1, 0], [0, 1]]  (reflection reverses z)
    Spherical mirror (R): [[1, 0], [-2/R, 1]]

References:
    - Hecht. "Optics" (2017), Ch. 6
    - Saleh & Teich. "Fundamentals of Photonics" (2019), Ch. 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from neurosim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ray:
    """A paraxial ray characterized by height and angle.

    Attributes:
        y: Ray height above the optical axis (meters).
        theta: Ray angle with respect to the optical axis (radians).
    """

    y: float
    theta: float

    def to_vector(self) -> Array:
        """Convert to column vector [y, theta]."""
        return jnp.array([self.y, self.theta])


@dataclass(frozen=True)
class ThinLens:
    """Thin lens optical element.

    Attributes:
        f: Focal length in meters. Positive for converging.
        position: Position along the optical axis.
    """

    f: float
    position: float = 0.0

    def matrix(self) -> Array:
        """ABCD matrix for a thin lens."""
        if self.f == 0:
            raise ConfigurationError("Focal length cannot be zero")
        return jnp.array([[1.0, 0.0], [-1.0 / self.f, 1.0]])


@dataclass(frozen=True)
class FlatMirror:
    """Flat mirror (plane reflector).

    Attributes:
        position: Position along the optical axis.
    """

    position: float = 0.0

    def matrix(self) -> Array:
        """ABCD matrix for a flat mirror (identity in paraxial approx)."""
        return jnp.eye(2)


@dataclass(frozen=True)
class SphericalMirror:
    """Spherical mirror.

    Attributes:
        R: Radius of curvature. Positive for concave.
        position: Position along the optical axis.
    """

    R: float
    position: float = 0.0

    def matrix(self) -> Array:
        """ABCD matrix for a spherical mirror."""
        if self.R == 0:
            raise ConfigurationError("Radius of curvature cannot be zero")
        return jnp.array([[1.0, 0.0], [-2.0 / self.R, 1.0]])


def _free_space_matrix(d: float) -> Array:
    """ABCD matrix for free space propagation.

    Args:
        d: Propagation distance in meters.

    Returns:
        2x2 transfer matrix.
    """
    return jnp.array([[1.0, d], [0.0, 1.0]])


@dataclass(frozen=True)
class TraceResult:
    """Result of ray tracing through an optical system.

    Attributes:
        positions: Position of each element along optical axis.
        heights: Ray height at each element.
        angles: Ray angle at each element.
        system_matrix: Total ABCD matrix of the system.
    """

    positions: list[float]
    heights: list[float]
    angles: list[float]
    system_matrix: Array

    @property
    def image_distance(self) -> float | None:
        """Compute image distance from the last element.

        For a system matrix [[A, B], [C, D]], the image forms where
        B = 0 (all rays from a point converge). If B != 0, returns None.
        """
        B = float(self.system_matrix[0, 1])
        if abs(B) < 1e-10:
            return 0.0
        # For a lens at the end: image at d where A + B*d_obj = 0
        # This is a simplified check
        return None


def trace_system(
    ray: Ray,
    elements: list[ThinLens | FlatMirror | SphericalMirror],
) -> TraceResult:
    """Trace a ray through a sequence of optical elements.

    Elements are assumed to be ordered by position along the optical
    axis. Free-space propagation is automatically inserted between
    elements.

    Args:
        ray: Input ray.
        elements: Ordered list of optical elements.

    Returns:
        TraceResult with ray state at each element.

    Raises:
        ConfigurationError: If elements are not in order.
    """
    if not elements:
        raise ConfigurationError("Need at least one optical element")

    # Sort by position
    sorted_elements = sorted(elements, key=lambda e: e.position)

    v = ray.to_vector()
    total_matrix = jnp.eye(2)
    current_pos = 0.0

    positions = [current_pos]
    heights = [float(v[0])]
    angles = [float(v[1])]

    for elem in sorted_elements:
        # Free space to element
        d = elem.position - current_pos
        if d < 0:
            raise ConfigurationError(
                f"Element at position {elem.position} is behind "
                f"current position {current_pos}"
            )
        if d > 0:
            M_space = _free_space_matrix(d)
            v = M_space @ v
            total_matrix = M_space @ total_matrix

        # Apply element
        M_elem = elem.matrix()
        v = M_elem @ v
        total_matrix = M_elem @ total_matrix
        current_pos = elem.position

        positions.append(current_pos)
        heights.append(float(v[0]))
        angles.append(float(v[1]))

    return TraceResult(
        positions=positions,
        heights=heights,
        angles=angles,
        system_matrix=total_matrix,
    )
