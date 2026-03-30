"""Ising model phase transition sweep.

Runs a temperature sweep across the 2D Ising critical region and
reports the magnetization and susceptibility peak. The example keeps
the workload small enough to run quickly as a script.

Usage:
    python examples/ising_phase_transition.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import neurosim as ns

jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Run the Ising temperature sweep."""
    lattice = ns.IsingLattice(size=(12, 12), J=1.0)
    temperatures = jnp.linspace(1.7, 3.1, 6)

    print("Running temperature sweep...")

    result = ns.sweep_temperatures(
        lattice,
        temperatures,
        n_sweeps=12,
        n_warmup=20,
        algorithm="metropolis",
        key=jax.random.PRNGKey(0),
    )

    tc = float(2.0 / jnp.log(1.0 + jnp.sqrt(2.0)))
    peak_idx = int(jnp.argmax(result.susceptibilities))

    print(f"Onsager critical temperature: {tc:.4f}")
    print(f"Peak susceptibility at T = {float(result.temperatures[peak_idx]):.4f}")
    for temp, mag, susc in zip(
        result.temperatures,
        result.magnetizations,
        result.susceptibilities,
        strict=True,
    ):
        print(
            f"T={float(temp):.3f}  "
            f"|m|={float(mag):.3f}  "
            f"chi={float(susc):.3f}"
        )

    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(result.temperatures, result.magnetizations, "o-", label="|m|")
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Magnetization")
        ax1.axvline(tc, color="gray", linestyle="--", linewidth=1)
        ax2 = ax1.twinx()
        ax2.plot(
            result.temperatures,
            result.susceptibilities,
            "s-",
            color="tab:red",
            label="chi",
        )
        ax2.set_ylabel("Susceptibility")
        fig.tight_layout()
        fig.savefig("ising_phase_transition.png", dpi=150)
        print("Saved ising_phase_transition.png")
    except Exception as exc:
        print(f"Visualization skipped: {exc}")


if __name__ == "__main__":
    main()
