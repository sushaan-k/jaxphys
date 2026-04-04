"""Configuration models for neurosim simulations.

Uses pydantic for runtime validation of simulation parameters,
ensuring physically meaningful values before computation begins.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Params(BaseModel):
    """Generic parameter container for physics simulations.

    Accepts arbitrary keyword arguments and exposes them as attributes.
    Validated at construction time for NaN/Inf safety.

    Example:
        >>> params = Params(m1=1.0, m2=2.0, g=9.81)
        >>> params.m1
        1.0
    """

    model_config = {"extra": "allow"}

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access for extra fields."""
        if name.startswith("_"):
            raise AttributeError(name)
        extra = self.__pydantic_extra__
        if extra and name in extra:
            return extra[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class SimulationConfig(BaseModel):
    """Configuration for time-stepping simulations.

    Attributes:
        t_start: Start time of the simulation.
        t_end: End time of the simulation.
        dt: Time step size. Must be positive.
        n_steps: Number of time steps. Computed from t_span and dt if
            not provided.
        integrator: Name of the numerical integrator to use.
        save_every: Save state every N steps. 1 saves every step.
    """

    t_start: float = 0.0
    t_end: float = Field(..., gt=0.0)
    dt: float = Field(0.001, gt=0.0)
    n_steps: int | None = None
    integrator: str = "rk4"
    save_every: int = Field(1, ge=1)

    @field_validator("t_end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        """Validate that t_end > t_start."""
        t_start = info.data.get("t_start", 0.0)
        if v <= t_start:
            raise ValueError(f"t_end ({v}) must be greater than t_start ({t_start})")
        return v

    @property
    def t_span(self) -> tuple[float, float]:
        """Return the time span as a tuple."""
        return (self.t_start, self.t_end)

    @property
    def total_steps(self) -> int:
        """Compute total number of steps."""
        if self.n_steps is not None:
            return self.n_steps
        return int((self.t_end - self.t_start) / self.dt)


class NBodyConfig(BaseModel):
    """Configuration for N-body gravitational simulations.

    Attributes:
        G: Gravitational constant.
        softening: Softening length to prevent singularities at close
            approach. Should be small relative to typical separations.
        use_barnes_hut: Whether to use the Barnes-Hut approximation for
            O(N log N) force computation.
        theta: Barnes-Hut opening angle parameter.
    """

    G: float = Field(1.0, gt=0.0)
    softening: float = Field(1e-4, ge=0.0)
    use_barnes_hut: bool = False
    theta: float = Field(0.5, ge=0.0, le=1.0)


class EMConfig(BaseModel):
    """Configuration for electromagnetic FDTD simulations.

    Attributes:
        resolution: Spatial grid resolution (cell size in meters).
        courant_number: Courant-Friedrichs-Lewy number. Must be <= 1/sqrt(2)
            for 2D stability, <= 1/sqrt(3) for 3D.
        boundary: Boundary condition type.
        pml_layers: Number of PML absorbing layers at boundaries.
    """

    resolution: float = Field(0.01, gt=0.0)
    courant_number: float = Field(0.5, gt=0.0, le=1.0)
    boundary: Literal["absorbing", "periodic", "reflecting"] = "absorbing"
    pml_layers: int = Field(10, ge=0)


class QuantumConfig(BaseModel):
    """Configuration for quantum mechanical simulations.

    Attributes:
        hbar: Reduced Planck constant. Default is 1.0 (natural units).
        mass: Particle mass. Default is 1.0 (natural units).
        n_points: Number of spatial grid points.
        method: Numerical method for time evolution.
    """

    hbar: float = Field(1.0, gt=0.0)
    mass: float = Field(1.0, gt=0.0)
    n_points: int = Field(1000, ge=10)
    method: Literal["split_operator", "crank_nicolson"] = "split_operator"


class FluidConfig(BaseModel):
    """Configuration for fluid dynamics simulations.

    Attributes:
        viscosity: Kinematic viscosity (in lattice units for LBM).
        method: Simulation method.
        boundary: Boundary condition type.
    """

    viscosity: float = Field(0.1, gt=0.0)
    method: Literal["lbm", "navier_stokes"] = "lbm"
    boundary: Literal["periodic", "no_slip", "free_slip"] = "periodic"


class IsingConfig(BaseModel):
    """Configuration for Ising model simulations.

    Attributes:
        J: Coupling constant. Positive for ferromagnetic.
        h: External magnetic field strength.
        algorithm: Monte Carlo update algorithm.
    """

    J: float = Field(1.0)
    h: float = Field(0.0)
    algorithm: Literal["metropolis", "wolff_cluster"] = "metropolis"
