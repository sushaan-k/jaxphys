"""Three-body gravitational simulation.

Simulates a Sun-Jupiter-Earth-like system using the N-body engine
with velocity Verlet integration. Demonstrates GPU-accelerated
orbital mechanics.

Usage:
    python examples/three_body.py
"""

import jax

import neurosim as ns

jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Run the three-body simulation."""
    # Sun-Jupiter-Earth system (natural units: G=1)
    system = ns.NBody(
        masses=[1.0, 0.001, 0.0003],
        positions=[
            [0.0, 0.0, 0.0],   # Sun at origin
            [5.2, 0.0, 0.0],   # Jupiter-like orbit
            [1.0, 0.0, 0.0],   # Earth-like orbit
        ],
        velocities=[
            [0.0, 0.0, 0.0],   # Sun stationary
            [0.0, 0.44, 0.0],  # Jupiter circular velocity
            [0.0, 1.0, 0.0],   # Earth circular velocity
        ],
        G=1.0,
        softening=1e-6,
    )

    print(f"Simulating {system.n_bodies}-body system...")

    trajectory = system.simulate(
        t_span=(0, 100),
        n_steps=500000,
        save_every=500,
    )

    print(f"Simulation complete: {trajectory.n_steps} saved snapshots")
    print(f"Energy drift: {trajectory.positions.shape}")

    # Check energy conservation
    e0 = float(trajectory.energy[0])
    ef = float(trajectory.energy[-1])
    drift = abs((ef - e0) / abs(e0))
    print(f"Initial energy: {e0:.6f}")
    print(f"Final energy:   {ef:.6f}")
    print(f"Relative drift: {drift:.2e}")


if __name__ == "__main__":
    main()
