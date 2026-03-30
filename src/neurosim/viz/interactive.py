"""Interactive Jupyter widget utilities.

Provides helper functions for creating interactive parameter
exploration widgets in Jupyter notebooks.

Requires ipywidgets (optional dependency).
"""

from __future__ import annotations

from typing import Any

from neurosim.exceptions import VisualizationError

try:
    import ipywidgets as widgets

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


def _check_widgets() -> None:
    if not HAS_WIDGETS:
        raise VisualizationError(
            "ipywidgets required for interactive features. "
            "Install with: pip install ipywidgets"
        )


def parameter_slider(
    name: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float = 0.01,
    description: str = "",
) -> Any:
    """Create a parameter slider widget.

    Args:
        name: Parameter name (used as label).
        min_val: Minimum value.
        max_val: Maximum value.
        default: Default value.
        step: Step size.
        description: Optional description text.

    Returns:
        ipywidgets FloatSlider.
    """
    _check_widgets()

    return widgets.FloatSlider(
        value=default,
        min=min_val,
        max=max_val,
        step=step,
        description=description if description else name,
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )


def interactive_simulation(
    simulate_fn: Any,
    **slider_params: dict[str, Any],
) -> Any:
    """Create an interactive simulation with parameter sliders.

    Args:
        simulate_fn: Function that takes keyword args and returns
            a matplotlib figure.
        **slider_params: Keyword arguments defining sliders.
            Each value should be a dict with keys:
            min, max, default, step (optional).

    Returns:
        ipywidgets interactive output.
    """
    _check_widgets()

    sliders = {}
    for name, config in slider_params.items():
        sliders[name] = widgets.FloatSlider(
            value=config.get("default", 1.0),
            min=config.get("min", 0.0),
            max=config.get("max", 10.0),
            step=config.get("step", 0.1),
            description=name,
            continuous_update=False,
            style={"description_width": "initial"},
        )

    return widgets.interactive(simulate_fn, **sliders)
