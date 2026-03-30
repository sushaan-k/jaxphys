"""Phase space and energy plotting utilities.

Provides functions for visualizing trajectories in phase space
and monitoring energy conservation during simulations.
"""

from __future__ import annotations

from typing import Any

from neurosim.exceptions import VisualizationError
from neurosim.state import Trajectory

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    """Raise an error if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise VisualizationError(
            "matplotlib is required for plotting. "
            "Install it with: pip install neurosim[viz]"
        )


def plot_phase_space(
    trajectory: Trajectory,
    coords: list[int] | None = None,
    figsize: tuple[float, float] = (8, 6),
    **kwargs: Any,
) -> Any:
    """Plot a trajectory in phase space (q vs p).

    Args:
        trajectory: Simulation trajectory.
        coords: Indices of degrees of freedom to plot.
            Defaults to [0].
        figsize: Figure size in inches.
        **kwargs: Additional matplotlib plot kwargs.

    Returns:
        matplotlib Figure object.

    Raises:
        VisualizationError: If matplotlib is not installed.
    """
    _check_matplotlib()

    if coords is None:
        coords = [0]

    fig, axes = plt.subplots(1, len(coords), figsize=figsize, squeeze=False)

    for i, coord_idx in enumerate(coords):
        ax = axes[0, i]
        q = trajectory.q[:, coord_idx]
        p = trajectory.p[:, coord_idx]

        ax.plot(q, p, linewidth=0.5, alpha=0.8, **kwargs)
        ax.set_xlabel(f"$q_{{{coord_idx}}}$", fontsize=12)
        ax.set_ylabel(f"$\\dot{{q}}_{{{coord_idx}}}$", fontsize=12)
        ax.set_title(f"Phase Space (DOF {coord_idx})", fontsize=13)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_energy(
    trajectory: Trajectory,
    figsize: tuple[float, float] = (10, 4),
    **kwargs: Any,
) -> Any:
    """Plot energy vs time to verify conservation.

    Args:
        trajectory: Simulation trajectory (must have energy computed).
        figsize: Figure size.
        **kwargs: Additional matplotlib plot kwargs.

    Returns:
        matplotlib Figure object.

    Raises:
        VisualizationError: If energy is not available.
    """
    _check_matplotlib()

    if trajectory.energy is None:
        raise VisualizationError("Trajectory does not contain energy data")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Absolute energy
    ax1.plot(trajectory.t, trajectory.energy, linewidth=0.8, **kwargs)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Energy", fontsize=12)
    ax1.set_title("Total Energy", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Relative energy drift
    e0 = trajectory.energy[0]
    drift = (trajectory.energy - e0) / (abs(e0) + 1e-30)
    ax2.plot(trajectory.t, drift, linewidth=0.8, color="red", **kwargs)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("$(E - E_0) / |E_0|$", fontsize=12)
    ax2.set_title("Relative Energy Drift", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))

    fig.tight_layout()
    return fig


def plot_phase_transition(
    result: Any,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot Ising model magnetization vs temperature.

    Args:
        result: IsingResult from temperature sweep.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Magnetization
    axes[0, 0].plot(result.temperatures, result.magnetizations, "o-", ms=2)
    axes[0, 0].set_xlabel("Temperature $T$")
    axes[0, 0].set_ylabel("$|m|$")
    axes[0, 0].set_title("Magnetization")
    axes[0, 0].axvline(x=2.269, color="red", linestyle="--", alpha=0.5, label="$T_c$")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Energy
    axes[0, 1].plot(result.temperatures, result.energies, "o-", ms=2)
    axes[0, 1].set_xlabel("Temperature $T$")
    axes[0, 1].set_ylabel("$E/N$")
    axes[0, 1].set_title("Energy per Spin")
    axes[0, 1].grid(True, alpha=0.3)

    # Specific heat
    axes[1, 0].plot(result.temperatures, result.specific_heats, "o-", ms=2)
    axes[1, 0].set_xlabel("Temperature $T$")
    axes[1, 0].set_ylabel("$C_V$")
    axes[1, 0].set_title("Specific Heat")
    axes[1, 0].axvline(x=2.269, color="red", linestyle="--", alpha=0.5, label="$T_c$")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Susceptibility
    axes[1, 1].plot(result.temperatures, result.susceptibilities, "o-", ms=2)
    axes[1, 1].set_xlabel("Temperature $T$")
    axes[1, 1].set_ylabel("$\\chi$")
    axes[1, 1].set_title("Susceptibility")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_specific_heat(
    result: Any,
    figsize: tuple[float, float] = (8, 5),
) -> Any:
    """Plot specific heat vs temperature.

    Args:
        result: IsingResult from temperature sweep.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.temperatures, result.specific_heats, "o-", ms=3)
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Specific Heat $C_V$", fontsize=12)
    ax.set_title("Specific Heat vs Temperature", fontsize=13)
    ax.axvline(
        x=2.269, color="red", linestyle="--", alpha=0.5, label="$T_c \\approx 2.269$"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
