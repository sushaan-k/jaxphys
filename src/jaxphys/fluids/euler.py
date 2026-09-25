"""Finite-volume solver for the 1D compressible Euler equations.

Solves the conservation laws for an ideal gas

    d/dt (rho, rho u, E) + d/dx (rho u, rho u^2 + p, (E + p) u) = 0,
    E = p / (gamma - 1) + rho u^2 / 2,

with a second-order Godunov-type scheme: minmod-limited (MUSCL) linear
reconstruction of the primitive variables, the HLLC approximate Riemann
solver at cell faces, and two-stage strong-stability-preserving
Runge-Kutta (Heun) time stepping. The scheme is conservative, so shocks
move at the correct speed.

The whole run is a compiled ``lax.scan``; results are differentiable with
respect to the initial state (e.g. for shock-tube parameter estimation).

References:
    - Toro. "Riemann Solvers and Numerical Methods for Fluid Dynamics"
      (2009), Ch. 10 (HLLC) and Ch. 13-14 (MUSCL)
    - Sod. "A survey of several finite difference methods for systems of
      nonlinear hyperbolic conservation laws", JCP 27 (1978)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from jaxphys._rollout import is_traced, strided_rollout
from jaxphys.exceptions import ConfigurationError, NumericalInstabilityError

logger = logging.getLogger(__name__)

Boundary = Literal["transmissive", "reflective", "periodic"]


@dataclass(frozen=True)
class EulerResult:
    """Saved states of a 1D Euler simulation (cell averages).

    Attributes:
        t: Times, shape (n_saved,).
        x: Cell centres, shape (n_cells,).
        rho: Density, shape (n_saved, n_cells).
        u: Velocity, shape (n_saved, n_cells).
        p: Pressure, shape (n_saved, n_cells).
        gamma: Ratio of specific heats.
    """

    t: Array
    x: Array
    rho: Array
    u: Array
    p: Array
    gamma: float

    @property
    def energy(self) -> Array:
        """Total energy density ``E``."""
        return self.p / (self.gamma - 1.0) + 0.5 * self.rho * self.u**2

    @property
    def sound_speed(self) -> Array:
        """Adiabatic sound speed ``sqrt(gamma p / rho)``."""
        return jnp.sqrt(self.gamma * self.p / self.rho)


jax.tree_util.register_dataclass(
    EulerResult, data_fields=["t", "x", "rho", "u", "p"], meta_fields=["gamma"]
)


def solve_euler_1d(
    rho: Array,
    u: Array,
    p: Array,
    dx: float,
    t_end: float,
    gamma: float = 1.4,
    dt: float | None = None,
    cfl: float = 0.4,
    boundary: Boundary = "transmissive",
    save_every: int = 10,
) -> EulerResult:
    """Evolve an initial state of the 1D Euler equations to ``t_end``.

    Example (Sod shock tube):
        >>> x = (jnp.arange(400) + 0.5) / 400
        >>> left = x < 0.5
        >>> result = solve_euler_1d(
        ...     jnp.where(left, 1.0, 0.125), jnp.zeros(400),
        ...     jnp.where(left, 1.0, 0.1), dx=1 / 400, t_end=0.2,
        ... )

    Args:
        rho: Initial density per cell, shape (n,).
        u: Initial velocity per cell, shape (n,).
        p: Initial pressure per cell, shape (n,).
        dx: Cell width.
        t_end: Final time (reached exactly).
        gamma: Ratio of specific heats.
        dt: Time step. If None, ``cfl * dx / max(|u| + c)`` of the initial
            state, reduced so that ``t_end`` is a whole number of
            ``save_every``-step intervals (so ``t_end`` is always saved).
            Pass ``dt`` explicitly to call this under ``jax.jit``/``grad``.
        cfl: Courant number used when ``dt`` is None. Waves generated later
            (e.g. shocks) can be faster than the initial ones, hence the
            conservative default.
        boundary: ``"transmissive"`` (zero-gradient outflow),
            ``"reflective"`` (solid walls) or ``"periodic"``.
        save_every: Save every N steps (the initial state is always saved).

    Returns:
        EulerResult with the saved primitive variables.

    Raises:
        ConfigurationError: For invalid shapes or parameters.
        NumericalInstabilityError: If the density or pressure becomes
            non-positive or non-finite (reduce ``dt``/``cfl``).
    """
    rho, u, p = (jnp.asarray(a, dtype=jnp.float64) for a in (rho, u, p))
    if rho.ndim != 1 or u.shape != rho.shape or p.shape != rho.shape:
        raise ConfigurationError("rho, u and p must be 1D arrays of equal length")
    if rho.shape[0] < 4:
        raise ConfigurationError("Need at least 4 cells")
    if boundary not in ("transmissive", "reflective", "periodic"):
        raise ConfigurationError(f"Unknown boundary '{boundary}'")
    if dx <= 0 or t_end <= 0 or gamma <= 1.0 or save_every < 1:
        raise ConfigurationError(
            "dx and t_end must be positive, gamma > 1 and save_every >= 1"
        )

    if dt is None:
        if is_traced(rho) or is_traced(u) or is_traced(p):
            raise ConfigurationError("Pass dt explicitly when tracing the inputs")
        c = jnp.sqrt(gamma * p / rho)
        dt = cfl * dx / float(jnp.max(jnp.abs(u) + c))
    # Round the step count up to a multiple of save_every so that the final
    # saved state is exactly t_end.
    n_steps = save_every * max(1, math.ceil(t_end / dt / save_every - 1e-9))
    dt = t_end / n_steps

    logger.info("Starting 1D Euler: n_cells=%d, n_steps=%d", rho.shape[0], n_steps)
    U0 = jnp.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2])
    history = _euler_rollout(
        U0, dt / dx, gamma, boundary=boundary, n_steps=n_steps, save_every=save_every
    )
    rho_h = history[:, 0]
    u_h = history[:, 1] / rho_h
    p_h = (gamma - 1.0) * (history[:, 2] - 0.5 * rho_h * u_h**2)
    if not is_traced(p_h) and not bool(
        jnp.all(jnp.isfinite(p_h)) & jnp.all(p_h > 0) & jnp.all(rho_h > 0)
    ):
        raise NumericalInstabilityError(
            "Non-positive density or pressure; reduce dt or cfl."
        )
    return EulerResult(
        t=dt * jnp.arange(0, n_steps + 1, save_every),
        x=(jnp.arange(rho.shape[0]) + 0.5) * dx,
        rho=rho_h,
        u=u_h,
        p=p_h,
        gamma=gamma,
    )


def _primitive(U: Array, gamma: float) -> Array:
    rho = U[0]
    u = U[1] / rho
    return jnp.stack([rho, u, (gamma - 1.0) * (U[2] - 0.5 * rho * u**2)])


def _with_ghosts(W: Array, boundary: str) -> Array:
    """Pad primitive variables with two ghost cells on each side."""
    if boundary == "periodic":
        return jnp.concatenate([W[:, -2:], W, W[:, :2]], axis=1)
    left, right = W[:, 1::-1], W[:, :-3:-1]  # mirrored edge cells
    if boundary == "transmissive":
        left, right = W[:, :1].repeat(2, 1), W[:, -1:].repeat(2, 1)
    else:  # reflective: mirror with reversed velocity
        flip = jnp.array([1.0, -1.0, 1.0])[:, None]
        left, right = left * flip, right * flip
    return jnp.concatenate([left, W, right], axis=1)


def _minmod(a: Array, b: Array) -> Array:
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def _hllc_flux(WL: Array, WR: Array, gamma: float) -> Array:
    """HLLC numerical flux from left/right primitive states (Toro 10.4)."""
    rL, uL, pL = WL
    rR, uR, pR = WR
    cL, cR = jnp.sqrt(gamma * pL / rL), jnp.sqrt(gamma * pR / rR)
    EL = pL / (gamma - 1.0) + 0.5 * rL * uL**2
    ER = pR / (gamma - 1.0) + 0.5 * rR * uR**2
    # Davis wave-speed estimates.
    SL = jnp.minimum(uL - cL, uR - cR)
    SR = jnp.maximum(uL + cL, uR + cR)
    S_star = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (
        rL * (SL - uL) - rR * (SR - uR)
    )

    def flux(r: Array, v: Array, pr: Array, E: Array) -> Array:
        return jnp.stack([r * v, r * v**2 + pr, (E + pr) * v])

    def star(r: Array, v: Array, pr: Array, E: Array, S: Array) -> Array:
        factor = r * (S - v) / (S - S_star)
        energy = E / r + (S_star - v) * (S_star + pr / (r * (S - v)))
        out: Array = factor * jnp.stack([jnp.ones_like(r), S_star, energy])
        return out

    UL = jnp.stack([rL, rL * uL, EL])
    UR = jnp.stack([rR, rR * uR, ER])
    FL, FR = flux(rL, uL, pL, EL), flux(rR, uR, pR, ER)
    FL_star = FL + SL * (star(rL, uL, pL, EL, SL) - UL)
    FR_star = FR + SR * (star(rR, uR, pR, ER, SR) - UR)
    return jnp.where(
        SL >= 0,
        FL,
        jnp.where(S_star >= 0, FL_star, jnp.where(SR > 0, FR_star, FR)),
    )


def _rhs(U: Array, gamma: float, boundary: str) -> Array:
    """``-(F_{i+1/2} - F_{i-1/2})`` (to be scaled by dt/dx)."""
    W = _with_ghosts(_primitive(U, gamma), boundary)
    slope = _minmod(W[:, 1:-1] - W[:, :-2], W[:, 2:] - W[:, 1:-1])
    # Face i+1/2 between padded cells k and k+1, for k = 1 .. n + 1.
    WL = W[:, 1:-2] + 0.5 * slope[:, :-1]
    WR = W[:, 2:-1] - 0.5 * slope[:, 1:]
    F = _hllc_flux(WL, WR, gamma)
    return -(F[:, 1:] - F[:, :-1])


@partial(jax.jit, static_argnames=("boundary", "n_steps", "save_every"))
def _euler_rollout(
    U0: Array,
    dt_dx: float,
    gamma: float,
    *,
    boundary: str,
    n_steps: int,
    save_every: int,
) -> Array:
    """SSP-RK2 rollout of the conserved variables, shape (n_saved, 3, n)."""

    def step(U: Array) -> Array:
        U1 = U + dt_dx * _rhs(U, gamma, boundary)
        return 0.5 * (U + U1 + dt_dx * _rhs(U1, gamma, boundary))

    out: Array = strided_rollout(step, U0, n_steps, save_every, lambda U: U)
    return out
