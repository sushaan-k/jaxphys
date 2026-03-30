"""Double pendulum simulation.

Demonstrates chaotic dynamics of a double pendulum using the
Lagrangian mechanics engine. The system exhibits sensitive
dependence on initial conditions — a hallmark of chaos.

Usage:
    python examples/double_pendulum.py
"""

import jax
import jax.numpy as jnp

import neurosim as ns

jax.config.update("jax_enable_x64", True)


def lagrangian(q: jnp.ndarray, qdot: jnp.ndarray, params: ns.Params) -> jnp.ndarray:
    """Lagrangian for the double pendulum.

    L = T - V where:
        T = 0.5*m1*(l1*w1)^2 + 0.5*m2*[(l1*w1)^2 + (l2*w2)^2
            + 2*l1*l2*w1*w2*cos(t1-t2)]
        V = -(m1+m2)*g*l1*cos(t1) - m2*g*l2*cos(t2)
    """
    theta1, theta2 = q
    omega1, omega2 = qdot
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    T = 0.5 * m1 * (l1 * omega1) ** 2 + 0.5 * m2 * (
        (l1 * omega1) ** 2
        + (l2 * omega2) ** 2
        + 2 * l1 * l2 * omega1 * omega2 * jnp.cos(theta1 - theta2)
    )
    V = -(m1 + m2) * g * l1 * jnp.cos(theta1) - m2 * g * l2 * jnp.cos(
        theta2
    )
    return T - V


def main() -> None:
    """Run the double pendulum simulation."""
    system = ns.LagrangianSystem(lagrangian, n_dof=2)
    params = ns.Params(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)

    # Initial conditions: both arms at angles, released from rest
    trajectory = system.simulate(
        q0=[jnp.pi / 4, jnp.pi / 2],
        qdot0=[0.0, 0.0],
        t_span=(0, 30),
        dt=0.001,
        params=params,
        integrator="rk4",
        save_every=10,
    )

    print(f"Simulation complete: {trajectory.n_steps} saved steps")
    print(f"Duration: {trajectory.duration:.1f} s")
    print(f"Energy drift: {trajectory.energy_drift():.2e}")

    # Try to visualize if matplotlib is available
    try:
        fig = ns.plot_phase_space(trajectory, coords=[0, 1])
        fig.savefig("double_pendulum_phase.png", dpi=150)
        print("Phase space plot saved to double_pendulum_phase.png")

        fig = ns.plot_energy(trajectory)
        fig.savefig("double_pendulum_energy.png", dpi=150)
        print("Energy plot saved to double_pendulum_energy.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()
