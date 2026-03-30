"""Field visualization utilities.

Provides functions for plotting electromagnetic fields, potential
energy surfaces, and vector fields.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neurosim.exceptions import VisualizationError
from neurosim.state import EMFieldHistory

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise VisualizationError(
            "matplotlib required. Install with: pip install neurosim[viz]"
        )


def plot_field_snapshot(
    fields: EMFieldHistory,
    step: int = -1,
    component: str = "Ez",
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "RdBu_r",
) -> Any:
    """Plot a snapshot of an electromagnetic field component.

    Args:
        fields: EMFieldHistory from FDTD simulation.
        step: Time step index to plot (-1 for last).
        component: Field component ("Ez", "Hx", "Hy").
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        matplotlib Figure object.
    """
    _check_matplotlib()

    component_map = {
        "Ez": fields.ez,
        "Hx": fields.hx,
        "Hy": fields.hy,
    }
    if component not in component_map:
        raise VisualizationError(
            f"Unknown component '{component}'. Available: {list(component_map.keys())}"
        )

    data = component_map[component][step]
    vmax = float(jnp.max(jnp.abs(data)))
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data.T,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=(
            float(fields.grid_x[0]),
            float(fields.grid_x[-1]),
            float(fields.grid_y[0]),
            float(fields.grid_y[-1]),
        ),
    )
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(
        f"{component} at t = {float(fields.t[step]):.2e} s",
        fontsize=13,
    )
    fig.colorbar(im, ax=ax, label=f"{component} (a.u.)")

    fig.tight_layout()
    return fig


def animate_field(
    fields: EMFieldHistory,
    component: str = "Ez",
    save: str | None = None,
    fps: int = 30,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "RdBu_r",
) -> Any:
    """Animate an electromagnetic field evolving in time.

    Args:
        fields: EMFieldHistory from FDTD simulation.
        component: Field component to animate.
        save: File path to save animation.
        fps: Frames per second.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        matplotlib FuncAnimation object.
    """
    _check_matplotlib()

    component_map = {
        "Ez": fields.ez,
        "Hx": fields.hx,
        "Hy": fields.hy,
    }
    if component not in component_map:
        raise VisualizationError(f"Unknown component '{component}'")

    data = component_map[component]
    vmax = float(jnp.max(jnp.abs(data)))
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data[0].T,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=(
            float(fields.grid_x[0]),
            float(fields.grid_x[-1]),
            float(fields.grid_y[0]),
            float(fields.grid_y[-1]),
        ),
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = ax.set_title(f"{component} at t = 0")
    fig.colorbar(im, ax=ax)

    n_frames = data.shape[0]

    def update(frame: int) -> list[Any]:
        im.set_data(data[frame].T)
        title.set_text(f"{component} at t = {float(fields.t[frame]):.2e} s")
        return [im, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    if save:
        anim.save(save, fps=fps, dpi=100)

    return anim
