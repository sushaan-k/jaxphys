"""Shared time-stepping helpers used by every solver.

All solvers advance a state with a pure ``step`` function and only keep
every ``save_every``-th state. Stacking every step and subsampling
afterwards costs ``n_steps`` times the state size in memory (gigabytes for
field solvers), so the loop is nested instead: an outer ``lax.scan`` over
snapshots and an inner ``fori_loop`` over the steps between them. The
outer body is wrapped in ``jax.checkpoint`` so reverse-mode gradients only
store one carry per snapshot and recompute the inner steps.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

C = TypeVar("C")


def strided_rollout(
    step: Callable[[C], C],
    init: C,
    n_steps: int,
    save_every: int,
    observe: Callable[[C], Any],
) -> Any:
    """Return ``observe(state)`` at steps ``0, save_every, 2*save_every, ...``.

    Snapshot ``k`` is the state after ``k * save_every`` applications of
    ``step``; the initial state is snapshot 0, so ``n_steps // save_every + 1``
    snapshots are returned (the same rows as ``history[::save_every]`` of the
    full trajectory). Steps after the last snapshot are not executed because
    nothing observes them.
    """

    @jax.checkpoint
    def chunk(carry: C, _: None) -> tuple[C, Any]:
        carry = jax.lax.fori_loop(0, save_every, lambda _, c: step(c), carry)
        return carry, observe(carry)

    first = observe(init)
    _, rest = jax.lax.scan(chunk, init, None, length=n_steps // save_every)
    return jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([jnp.asarray(a)[None], b]), first, rest
    )


def is_traced(x: Any) -> bool:
    """True when ``x`` is an abstract value inside ``jit``/``vmap``/``grad``.

    Host-side validation (NaN checks, logging of values) is skipped for
    traced values so simulations stay composable with JAX transformations.
    """
    return isinstance(x, jax.core.Tracer)


def is_array_tree(tree: Any) -> bool:
    """True when every leaf of ``tree`` can be passed as a traced jit argument."""
    return all(
        isinstance(leaf, numbers.Number | np.ndarray | np.generic | jax.Array)
        for leaf in jax.tree_util.tree_leaves(tree)
    )
