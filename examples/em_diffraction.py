"""Electromagnetic diffraction through a slit.

Demonstrates the 2D FDTD solver by launching a plane wave toward a
conducting screen with a single slit. The resulting field snapshot is
a simple diffraction pattern that can be visualized if matplotlib is
available.

Usage:
    python examples/em_diffraction.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import neurosim as ns

jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Run the slit diffraction simulation."""
    grid = ns.EMGrid(size=(60, 60), resolution=0.01, pml_layers=8)
    grid.add_source(ns.PlaneWave(frequency=3.0e9, y=6, amplitude=1.0))
    grid.add_conductor(ns.Wall(y=30, gap_start=26, gap_end=34))

    fields = grid.simulate(t_span=(0.0, 1.5e-9), save_every=20)

    final_ez = fields.ez[-1]
    screen_slice = jnp.abs(final_ez[:, 35])

    print(f"Saved snapshots: {fields.t.shape[0]}")
    print(f"Peak |Ez| on final frame: {float(jnp.max(jnp.abs(final_ez))):.3e}")
    print(f"Screen intensity proxy: {float(jnp.mean(screen_slice ** 2)):.3e}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(
            jnp.abs(final_ez).T,
            origin="lower",
            cmap="magma",
            aspect="auto",
        )
        ax.set_title("EM diffraction snapshot")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, label="|Ez|")
        fig.tight_layout()
        fig.savefig("em_diffraction.png", dpi=150)
        print("Saved em_diffraction.png")
    except Exception as exc:
        print(f"Visualization skipped: {exc}")


if __name__ == "__main__":
    main()
