"""Quantum tunneling through a potential barrier.

Demonstrates the wave nature of quantum particles by simulating
a Gaussian wavepacket incident on a square potential barrier.
Part of the wavepacket tunnels through, part reflects.

Usage:
    python examples/quantum_tunneling.py
"""

import jax
import jax.numpy as jnp

import neurosim as ns

jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Run the quantum tunneling simulation."""
    # Define the potential barrier
    barrier = ns.SquareBarrier(height=5.0, width=1.0, center=10.0)

    # Gaussian wavepacket approaching the barrier
    psi0 = ns.GaussianWavepacket(x0=5.0, k0=3.0, sigma=0.5)

    print("Solving time-dependent Schrodinger equation...")
    print(f"Barrier: height={barrier.height}, width={barrier.width}")
    print(f"Wavepacket: x0={psi0.x0}, k0={psi0.k0}, sigma={psi0.sigma}")
    print(f"Incident energy: E = hbar^2 k^2 / (2m) = {0.5 * psi0.k0**2:.2f}")
    print(f"Barrier height: V0 = {barrier.height:.2f}")
    print(f"E {'>' if 0.5 * psi0.k0**2 > barrier.height else '<'} V0 "
          f"({'classically allowed' if 0.5 * psi0.k0**2 > barrier.height else 'tunneling regime'})")

    result = ns.solve_schrodinger(
        psi0=psi0,
        potential=barrier,
        x_range=(-5, 25),
        t_span=(0, 8),
        n_points=1000,
        dt=0.005,
        save_every=20,
    )

    print(f"\nSimulation complete: {result.n_steps} snapshots saved")

    if result.transmission_coefficient is not None:
        print(f"Transmission coefficient: {result.transmission_coefficient:.4f}")
        print(f"Reflection coefficient:   {1 - result.transmission_coefficient:.4f}")

    # Verify norm conservation
    for i in [0, result.n_steps // 2, -1]:
        norm = float(jnp.trapezoid(jnp.abs(result.psi[i]) ** 2, result.x))
        print(f"  Norm at step {i}: {norm:.8f}")

    # Also solve the eigenvalue problem for the harmonic oscillator
    print("\n--- Harmonic oscillator eigenvalues ---")
    ho = ns.HarmonicPotential(k=1.0)
    eigen = ns.solve_eigenvalue_problem(
        potential=ho, x_range=(-10, 10), n_points=500, n_states=5
    )
    for n, E in enumerate(eigen.energies):
        print(f"  E_{n} = {float(E):.4f}  (exact: {n + 0.5:.4f})")


if __name__ == "__main__":
    main()
