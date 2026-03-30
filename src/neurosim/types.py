"""Core type definitions for neurosim.

Defines type aliases and protocols used throughout the library
to ensure consistent type annotations.
"""

from collections.abc import Callable
from typing import Any, TypeAlias

import jax
from jax import Array

# Core JAX array type
Scalar: TypeAlias = float | Array
Vec3: TypeAlias = Array  # shape (3,)
VecN: TypeAlias = Array  # shape (n,)
MatNM: TypeAlias = Array  # shape (n, m)

# Function signatures for physics
LagrangianFn: TypeAlias = Callable[[Array, Array, Any], Array]
HamiltonianFn: TypeAlias = Callable[[Array, Array, Any], Array]
ForceFn: TypeAlias = Callable[[Array, Array, float, Any], Array]
PotentialFn: TypeAlias = Callable[[Array], Array]

# RNG key type
PRNGKey: TypeAlias = jax.Array

# Time types
TimeSpan: TypeAlias = tuple[float, float]

__all__ = [
    "Scalar",
    "Vec3",
    "VecN",
    "MatNM",
    "LagrangianFn",
    "HamiltonianFn",
    "ForceFn",
    "PotentialFn",
    "PRNGKey",
    "TimeSpan",
]
