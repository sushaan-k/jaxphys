"""State representations for physics simulations.

Provides immutable data containers for simulation states and trajectories,
designed to be compatible with JAX transformations (jit, vmap, grad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class PhaseState:
    """Phase-space state for classical mechanical systems.

    Represents a point in phase space (q, p) or configuration space
    (q, qdot) at a given time.

    Attributes:
        q: Generalized coordinates, shape (n_dof,).
        p: Generalized momenta or velocities, shape (n_dof,).
        t: Current time.
    """

    q: Array
    p: Array
    t: float


@dataclass(frozen=True)
class Trajectory:
    """Time series of simulation states.

    Stores the full history of a simulation for analysis and
    visualization. All arrays have time along axis 0.

    Attributes:
        t: Time values, shape (n_steps,).
        q: Generalized coordinates, shape (n_steps, n_dof).
        p: Generalized momenta/velocities, shape (n_steps, n_dof).
        energy: Total energy at each step, shape (n_steps,), or None.
        metadata: Additional simulation metadata.
    """

    t: Array
    q: Array
    p: Array
    energy: Array | None = None
    metadata: dict[str, Any] | None = None

    @property
    def n_steps(self) -> int:
        """Number of time steps in the trajectory."""
        return int(self.t.shape[0])

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        if self.q.ndim == 1:
            return 1
        return int(self.q.shape[1])

    @property
    def duration(self) -> float:
        """Total simulation time."""
        return float(self.t[-1] - self.t[0])

    @property
    def final_position(self) -> Array:
        """Position at the final time step."""
        return self.q[-1]

    @property
    def final_momentum(self) -> Array:
        """Momentum/velocity at the final time step."""
        return self.p[-1]

    def energy_drift(self) -> float:
        """Compute relative energy drift over the trajectory.

        Returns:
            Relative change in energy: |E_final - E_initial| / |E_initial|.
            Returns 0.0 if energy was not tracked.
        """
        if self.energy is None:
            return 0.0
        e0 = self.energy[0]
        ef = self.energy[-1]
        if jnp.abs(e0) < 1e-15:
            return float(jnp.abs(ef - e0))
        return float(jnp.abs((ef - e0) / e0))


@dataclass(frozen=True)
class NBodyState:
    """State of an N-body gravitational system.

    Attributes:
        positions: Particle positions, shape (n_bodies, 3).
        velocities: Particle velocities, shape (n_bodies, 3).
        masses: Particle masses, shape (n_bodies,).
        t: Current time.
    """

    positions: Array
    velocities: Array
    masses: Array
    t: float


@dataclass(frozen=True)
class NBodyTrajectory:
    """Trajectory of an N-body system.

    Attributes:
        t: Time values, shape (n_steps,).
        positions: Positions at each step, shape (n_steps, n_bodies, 3).
        velocities: Velocities at each step, shape (n_steps, n_bodies, 3).
        masses: Particle masses, shape (n_bodies,).
        energy: Total energy at each step, shape (n_steps,), or None.
    """

    t: Array
    positions: Array
    velocities: Array
    masses: Array
    energy: Array | None = None

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.t.shape[0])

    @property
    def n_bodies(self) -> int:
        """Number of bodies."""
        return int(self.positions.shape[1])

    @property
    def final_position(self) -> Array:
        """Positions at the final time step."""
        return self.positions[-1]


@dataclass(frozen=True)
class EMFieldState:
    """State of electromagnetic fields on a 2D Yee grid.

    The Yee grid staggers E and H fields for second-order accuracy
    in the FDTD method.

    Attributes:
        ex: x-component of electric field.
        ey: y-component of electric field.
        ez: z-component of electric field (TM mode).
        hx: x-component of magnetic field.
        hy: y-component of magnetic field.
        hz: z-component of magnetic field (TE mode).
        t: Current time.
    """

    ex: Array
    ey: Array
    ez: Array
    hx: Array
    hy: Array
    hz: Array
    t: float


@dataclass(frozen=True)
class EMFieldHistory:
    """Time series of electromagnetic field snapshots.

    Attributes:
        t: Time values, shape (n_snapshots,).
        ez: Ez field snapshots, shape (n_snapshots, nx, ny).
        hx: Hx field snapshots, shape (n_snapshots, nx, ny).
        hy: Hy field snapshots, shape (n_snapshots, nx, ny).
        grid_x: Spatial x-coordinates.
        grid_y: Spatial y-coordinates.
    """

    t: Array
    ez: Array
    hx: Array
    hy: Array
    grid_x: Array
    grid_y: Array


@dataclass(frozen=True)
class QuantumState:
    """State of a quantum mechanical wavefunction.

    Attributes:
        psi: Complex wavefunction, shape (n_points,).
        x: Spatial grid, shape (n_points,).
        t: Current time.
        potential: Potential energy on the grid, shape (n_points,).
    """

    psi: Array
    x: Array
    t: float
    potential: Array | None = None


@dataclass(frozen=True)
class QuantumResult:
    """Result of a quantum mechanical simulation.

    Attributes:
        t: Time values, shape (n_steps,).
        psi: Wavefunction history, shape (n_steps, n_points).
        x: Spatial grid, shape (n_points,).
        potential: Potential on the grid, shape (n_points,).
        transmission_coefficient: Fraction transmitted past a barrier.
    """

    t: Array
    psi: Array
    x: Array
    potential: Array
    transmission_coefficient: float | None = None

    @property
    def probability(self) -> Array:
        """Probability density |psi|^2 at each time step."""
        return jnp.abs(self.psi) ** 2

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.t.shape[0])


@dataclass(frozen=True)
class FluidState:
    """State of a fluid simulation on a 2D grid.

    Attributes:
        rho: Density field, shape (nx, ny).
        ux: x-velocity field, shape (nx, ny).
        uy: y-velocity field, shape (nx, ny).
        t: Current time.
    """

    rho: Array
    ux: Array
    uy: Array
    t: float


@dataclass(frozen=True)
class FluidHistory:
    """Time series of fluid field snapshots.

    Attributes:
        t: Time values, shape (n_snapshots,).
        rho: Density snapshots, shape (n_snapshots, nx, ny).
        ux: x-velocity snapshots, shape (n_snapshots, nx, ny).
        uy: y-velocity snapshots, shape (n_snapshots, nx, ny).
        vorticity: Vorticity snapshots, shape (n_snapshots, nx, ny), or None.
        grid_x: Spatial x-coordinates.
        grid_y: Spatial y-coordinates.
    """

    t: Array
    rho: Array
    ux: Array
    uy: Array
    vorticity: Array | None = None
    grid_x: Array | None = None
    grid_y: Array | None = None

    @property
    def n_snapshots(self) -> int:
        """Number of saved snapshots."""
        return int(self.t.shape[0])

    @property
    def speed(self) -> Array:
        """Speed field |u| at each snapshot."""
        return jnp.sqrt(self.ux**2 + self.uy**2)


@dataclass(frozen=True)
class EMFieldHistory3D:
    """Time series of 3D electromagnetic field snapshots.

    Attributes:
        t: Time values, shape (n_snapshots,).
        ex: Ex field snapshots, shape (n_snapshots, nx, ny, nz).
        ey: Ey field snapshots, shape (n_snapshots, nx, ny, nz).
        ez: Ez field snapshots, shape (n_snapshots, nx, ny, nz).
        hx: Hx field snapshots, shape (n_snapshots, nx, ny, nz).
        hy: Hy field snapshots, shape (n_snapshots, nx, ny, nz).
        hz: Hz field snapshots, shape (n_snapshots, nx, ny, nz).
        grid_x: Spatial x-coordinates.
        grid_y: Spatial y-coordinates.
        grid_z: Spatial z-coordinates.
    """

    t: Array
    ex: Array
    ey: Array
    ez: Array
    hx: Array
    hy: Array
    hz: Array
    grid_x: Array
    grid_y: Array
    grid_z: Array


@dataclass(frozen=True)
class IsingResult:
    """Result of an Ising model Monte Carlo simulation.

    Attributes:
        temperatures: Temperature values, shape (n_temps,).
        magnetizations: Mean absolute magnetization per spin.
        energies: Mean energy per spin.
        specific_heats: Specific heat per spin.
        susceptibilities: Magnetic susceptibility per spin.
    """

    temperatures: Array
    magnetizations: Array
    energies: Array
    specific_heats: Array
    susceptibilities: Array
