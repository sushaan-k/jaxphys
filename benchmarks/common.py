"""Shared helpers: environment capture, timing, compile accounting."""

from __future__ import annotations

import datetime as _dt
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import jax
import numpy as np
from jax import monitoring

# ---------------------------------------------------------------------------
# Compile-time accounting.  JAX reports the duration of every trace, lowering
# and XLA backend compilation through jax.monitoring; summing those events
# around a call measures compile overhead directly (no subtraction of
# steady-state time needed).
# ---------------------------------------------------------------------------

_COMPILE_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower",
    "/jax/core/compile/backend_compile_duration": "backend",
}
_EVENTS: dict[str, float] = {"trace": 0.0, "lower": 0.0, "backend": 0.0, "n": 0.0}


def _listener(event: str, duration: float, **_: Any) -> None:
    kind = _COMPILE_EVENTS.get(event)
    if kind is not None:
        _EVENTS[kind] += duration
        if kind == "backend":
            _EVENTS["n"] += 1


monitoring.register_event_duration_secs_listener(_listener)


@contextmanager
def compile_accounting() -> Iterator[dict[str, float]]:
    """Collect trace/lower/backend-compile seconds and compile count."""
    start = dict(_EVENTS)
    out: dict[str, float] = {}
    try:
        yield out
    finally:
        for k in _EVENTS:
            out[k] = _EVENTS[k] - start[k]
        out["compile_s"] = out["trace"] + out["lower"] + out["backend"]


def block(tree: Any) -> Any:
    """block_until_ready on every array leaf of a pytree or dataclass."""
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def time_calls(fn: Callable[[], Any], repeats: int) -> list[float]:
    """Wall time of ``repeats`` calls of ``fn`` (each fully synchronized)."""
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        block(fn())
        out.append(time.perf_counter() - t0)
    return out


def summarize(samples: list[float]) -> dict[str, float]:
    """Median, interquartile range and extremes of timing samples."""
    arr = np.asarray(samples)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "median_s": float(q50),
        "iqr_s": float(q75 - q25),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "n": len(samples),
        "mean_s": float(statistics.fmean(samples)),
    }


def measure_jax(fn: Callable[[], Any], repeats: int, warmup: int = 1) -> dict[str, Any]:
    """First call (with compile accounting), warm-up, then timed repeats."""
    with compile_accounting() as comp:
        t0 = time.perf_counter()
        result = block(fn())
        first = time.perf_counter() - t0
    for _ in range(warmup):
        block(fn())
    with compile_accounting() as steady_comp:
        samples = time_calls(fn, repeats)
    return {
        "first_call_s": first,
        "compile": comp,
        "steady": summarize(samples),
        "steady_compiles": steady_comp["n"],
        "result": result,
    }


def measure_numpy(fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    result = fn()
    samples = time_calls(fn, repeats)
    return {"steady": summarize(samples), "result": result}


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _git(*args: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", *args],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    import jaxlib

    import jaxphys

    try:
        import scipy

        scipy_version: str | None = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "jaxphys": getattr(jaxphys, "__version__", None),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def max_rel_diff(a: Any, b: Any) -> tuple[float, float]:
    """Return (max |a - b|, max |a - b| / max |b|) over matching arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    return diff, (diff / scale if scale else diff)
