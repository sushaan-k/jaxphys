"""Custom exception classes for neurosim.

Provides domain-specific exceptions for physics simulation errors,
configuration issues, and numerical instabilities.
"""


class NeurosimError(Exception):
    """Base exception for all neurosim errors."""


class SimulationError(NeurosimError):
    """Raised when a simulation fails to converge or produces invalid results."""


class NumericalInstabilityError(SimulationError):
    """Raised when a numerical integration becomes unstable.

    Common causes include timesteps that are too large for the chosen
    integrator, or stiff systems that require implicit methods.
    """


class ConfigurationError(NeurosimError):
    """Raised when simulation parameters are invalid or inconsistent."""


class DimensionError(NeurosimError):
    """Raised when array dimensions are incompatible."""


class PhysicsError(NeurosimError):
    """Raised when a physical constraint is violated.

    Examples include negative masses, superluminal velocities in
    relativistic contexts, or non-Hermitian Hamiltonians.
    """


class ConvergenceError(SimulationError):
    """Raised when an iterative solver fails to converge."""


class VisualizationError(NeurosimError):
    """Raised when visualization encounters an error."""
