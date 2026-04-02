#!/usr/bin/env python3
"""Offline demo for neurosim."""

from __future__ import annotations

import jax.numpy as jnp

import neurosim as ns


def harmonic_hamiltonian(
    q: jnp.ndarray, p: jnp.ndarray, params: ns.Params
) -> jnp.ndarray:
    return p[0] ** 2 / (2 * params.m) + 0.5 * params.k * q[0] ** 2


def main() -> None:
    system = ns.HamiltonianSystem(harmonic_hamiltonian, n_dof=1)
    params = ns.Params(m=1.0, k=4.0)
    trajectory = system.simulate(
        q0=[1.0],
        p0=[0.0],
        t_span=(0.0, 6.0),
        dt=0.01,
        params=params,
        integrator="leapfrog",
    )
    pattern = ns.single_slit(
        slit_width=80e-6,
        wavelength=532e-9,
        n_points=256,
        theta_max=0.04,
    )

    print("neurosim demo")
    print(f"steps saved: {trajectory.n_steps}")
    print(f"energy drift: {trajectory.energy_drift():.2e}")
    print(f"final position: {float(trajectory.final_position[0]):.4f}")
    print(f"peak diffraction intensity: {float(pattern.intensity.max()):.4f}")


if __name__ == "__main__":
    main()
