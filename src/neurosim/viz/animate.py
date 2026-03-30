"""Animation utilities for physics simulations.

Provides functions for creating animated visualizations of
trajectories, wavefunctions, and other time-dependent data.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neurosim.exceptions import VisualizationError
from neurosim.state import QuantumResult, Trajectory

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise VisualizationError(
            "matplotlib is required for animation. "
            "Install with: pip install neurosim[viz]"
        )


def animate_pendulum(
    trajectory: Trajectory,
    lengths: list[float] | None = None,
    save: str | None = None,
    fps: int = 30,
    figsize: tuple[float, float] = (6, 6),
) -> Any:
    """Create an animation of a pendulum system.

    Supports single and double pendulums. Angles are taken from
    the trajectory's q array.

    Args:
        trajectory: Trajectory with angle data in q.
        lengths: Pendulum arm lengths. Defaults to [1.0] per DOF.
        save: If provided, save animation to this file path.
        fps: Frames per second.
        figsize: Figure size.

    Returns:
        matplotlib FuncAnimation object.
    """
    _check_matplotlib()

    n_dof = trajectory.n_dof
    if lengths is None:
        lengths = [1.0] * n_dof

    total_length = sum(lengths)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-total_length * 1.3, total_length * 1.3)
    ax.set_ylim(-total_length * 1.3, total_length * 1.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Pendulum Simulation", fontsize=13)

    (line,) = ax.plot([], [], "o-", lw=2, markersize=8, color="steelblue")
    (trail,) = ax.plot([], [], "-", lw=0.5, alpha=0.3, color="red")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    trail_x: list[float] = []
    trail_y: list[float] = []

    def init() -> tuple[Any, ...]:
        line.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return line, trail, time_text

    # Subsample frames for smooth animation
    n_frames = min(trajectory.n_steps, 5000)
    frame_indices = jnp.linspace(0, trajectory.n_steps - 1, n_frames).astype(int)

    def update(frame: int) -> tuple[Any, ...]:
        idx = int(frame_indices[frame])
        angles = trajectory.q[idx]

        # Convert angles to Cartesian
        xs = [0.0]
        ys = [0.0]
        for j in range(n_dof):
            xs.append(xs[-1] + lengths[j] * float(jnp.sin(angles[j])))
            ys.append(ys[-1] - lengths[j] * float(jnp.cos(angles[j])))

        line.set_data(xs, ys)

        # Trail of the last bob
        trail_x.append(xs[-1])
        trail_y.append(ys[-1])
        trail.set_data(trail_x[-2000:], trail_y[-2000:])

        time_text.set_text(f"t = {float(trajectory.t[idx]):.2f}")
        return line, trail, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
    )

    if save:
        anim.save(save, fps=fps, dpi=100)

    return anim


def animate_wavefunction(
    result: QuantumResult,
    show_probability: bool = True,
    show_potential: bool = True,
    save: str | None = None,
    fps: int = 30,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Animate a quantum wavefunction evolving in time.

    Shows |psi(x,t)|^2 and optionally the potential V(x).

    Args:
        result: QuantumResult from solve_schrodinger.
        show_probability: Whether to show probability density.
        show_potential: Whether to show the potential.
        save: File path to save animation.
        fps: Frames per second.
        figsize: Figure size.

    Returns:
        matplotlib FuncAnimation object.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    prob = jnp.abs(result.psi) ** 2
    max_prob = float(jnp.max(prob)) * 1.2

    ax.set_xlim(float(result.x[0]), float(result.x[-1]))
    ax.set_ylim(0, max_prob)
    ax.set_xlabel("Position $x$", fontsize=12)
    ax.set_ylabel("$|\\psi(x)|^2$", fontsize=12)
    ax.set_title("Wavefunction Evolution", fontsize=13)
    ax.grid(True, alpha=0.3)

    if show_potential and result.potential is not None:
        v_max = float(jnp.max(jnp.abs(result.potential)) + 1e-30)
        v_scaled = result.potential / v_max * max_prob * 0.3
        ax.fill_between(
            result.x,
            0,
            v_scaled,
            alpha=0.2,
            color="gray",
            label="$V(x)$",
        )

    (line,) = ax.plot([], [], lw=1.5, color="steelblue", label="$|\\psi|^2$")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def init() -> tuple[Any, ...]:
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(frame: int) -> tuple[Any, ...]:
        p = prob[frame]
        line.set_data(result.x, p)
        time_text.set_text(f"t = {float(result.t[frame]):.3f}")
        return line, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=result.n_steps,
        interval=1000 / fps,
        blit=True,
    )

    if save:
        anim.save(save, fps=fps, dpi=100)

    return anim


def animate_3d(
    trajectory: Any,
    trails: bool = True,
    save: str | None = None,
    fps: int = 30,
) -> Any:
    """Animate a 3D N-body trajectory.

    Args:
        trajectory: NBodyTrajectory.
        trails: Whether to show orbital trails.
        save: File path to save.
        fps: Frames per second.

    Returns:
        matplotlib FuncAnimation object.
    """
    _check_matplotlib()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Determine axis limits from trajectory
    pos = trajectory.positions
    margin = 0.2
    lim = float(jnp.max(jnp.abs(pos))) * (1 + margin)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    n_bodies = trajectory.n_bodies
    colors = plt.get_cmap("Set1")(jnp.linspace(0, 1, n_bodies))

    # Create plot elements
    scatters = [
        ax.plot([], [], [], "o", color=colors[i], ms=8)[0] for i in range(n_bodies)
    ]
    trail_lines = [
        ax.plot([], [], [], "-", color=colors[i], alpha=0.3, lw=0.5)[0]
        for i in range(n_bodies)
    ]

    n_frames = min(trajectory.n_steps, 3000)
    frame_indices = jnp.linspace(0, trajectory.n_steps - 1, n_frames).astype(int)

    def update(frame: int) -> list[Any]:
        idx = int(frame_indices[frame])
        artists = []
        for i in range(n_bodies):
            x = float(pos[idx, i, 0])
            y = float(pos[idx, i, 1])
            z = float(pos[idx, i, 2])
            scatters[i].set_data([x], [y])
            scatters[i].set_3d_properties([z])
            artists.append(scatters[i])

            if trails:
                trail_len = min(idx + 1, 500)
                start = max(0, idx - trail_len)
                tx = pos[start : idx + 1, i, 0]
                ty = pos[start : idx + 1, i, 1]
                tz = pos[start : idx + 1, i, 2]
                trail_lines[i].set_data(tx, ty)
                trail_lines[i].set_3d_properties(tz)
                artists.append(trail_lines[i])

        return artists

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
