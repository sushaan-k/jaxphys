"""Differentiable spacecraft launch targeting.

Uses the projectile helper and gradient-based optimizer to solve a
simple trajectory design problem: pick a launch speed that reaches a
specified downrange target under lunar gravity.

Usage:
    python examples/spacecraft_trajectory.py
"""

from __future__ import annotations

import jax.numpy as jnp

import neurosim as ns


def main() -> None:
    """Solve the launch-speed targeting problem."""
    target_range = 1200.0
    launch_angle = 35.0
    lunar_gravity = 1.62

    def objective(v0: float | jnp.ndarray) -> jnp.ndarray:
        return (ns.projectile(v0=v0, angle=launch_angle, g=lunar_gravity).range - target_range) ** 2

    result = ns.optimize(
        objective,
        initial_guess=45.0,
        learning_rate=0.01,
        max_iterations=200,
        tolerance=1.0,
        method="adam",
    )

    final = ns.projectile(v0=result.x, angle=launch_angle, g=lunar_gravity)
    miss_distance = float(final.range - target_range)

    print(f"Target range: {target_range:.1f} m")
    print(f"Optimal launch speed: {float(result.x):.3f} m/s")
    print(f"Range achieved: {float(final.range):.3f} m")
    print(f"Miss distance: {miss_distance:.6f} m")
    print(f"Time of flight: {float(final.time_of_flight):.3f} s")
    print(f"Converged: {result.converged} after {result.n_iterations} iterations")

    speeds = jnp.linspace(float(result.x) * 0.75, float(result.x) * 1.25, 50)
    ranges = jnp.array([float(ns.projectile(v0=s, angle=launch_angle, g=lunar_gravity).range) for s in speeds])
    print(f"Range sweep span: {float(ranges.min()):.1f} m .. {float(ranges.max()):.1f} m")


if __name__ == "__main__":
    main()
