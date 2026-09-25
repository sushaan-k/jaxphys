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
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
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


def any_traced(tree: Any) -> bool:
    """True when any leaf of ``tree`` is a tracer (see :func:`is_traced`)."""
    return any(is_traced(leaf) for leaf in jax.tree_util.tree_leaves(tree))


def is_array_tree(tree: Any) -> bool:
    """True when every leaf of ``tree`` can be passed as a traced jit argument."""
    return all(
        isinstance(leaf, numbers.Number | np.ndarray | np.generic | jax.Array)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _concrete_key(tree: Any) -> Any:
    """Hashable key of a pytree's structure and concrete leaf values, or None."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    key: list[Any] = [treedef]
    for leaf in leaves:
        if is_traced(leaf):
            return None
        if isinstance(leaf, np.ndarray | np.generic | jax.Array):
            arr = np.asarray(leaf)
            key.append((arr.dtype.str, arr.shape, arr.tobytes()))
        else:
            try:
                hash(leaf)
            except TypeError:
                return None
            key.append((type(leaf), leaf))
    return tuple(key)


_CONSTANT_PARAM_CACHE: OrderedDict[Any, Callable[..., Any]] = OrderedDict()
_CONSTANT_PARAM_CACHE_SIZE = 64


def call_with_params(
    jitted: Callable[..., Any],
    impl: Callable[..., Any],
    params: Any,
    static_argnames: tuple[str, ...],
    **kwargs: Any,
) -> Any:
    """Run a compiled rollout with ``params`` as a traced argument if possible.

    ``jitted`` must be ``jax.jit(impl, static_argnames=static_argnames)``,
    created once at module level, so new parameter *values* reuse the
    compiled loop. Two cases cannot be traced and fall back to compiling
    ``impl`` with ``params`` closed over as constants:

    * leaves that are not numbers or arrays (arbitrary Python objects);
    * user physics that needs concrete values (``if params.k > 0:``), which
      raises a ``JAXTypeError`` when traced.

    Fallback executables are cached on the concrete parameter values, so
    repeating a call with equal parameters does not compile again.
    """
    if is_array_tree(params):
        try:
            return jitted(params=params, **kwargs)
        except jax.errors.JAXTypeError:
            pass
    static = {k: kwargs.pop(k) for k in static_argnames if k in kwargs}
    params_key = _concrete_key(params)
    key = None
    if params_key is not None:
        try:
            key = (impl, tuple(sorted(static.items())), params_key)
            hash(key)
        except TypeError:
            key = None
    run = _CONSTANT_PARAM_CACHE.get(key) if key is not None else None
    if run is None:
        run = jax.jit(partial(impl, params=params, **static))
        if key is not None:
            _CONSTANT_PARAM_CACHE[key] = run
            if len(_CONSTANT_PARAM_CACHE) > _CONSTANT_PARAM_CACHE_SIZE:
                _CONSTANT_PARAM_CACHE.popitem(last=False)
    else:
        _CONSTANT_PARAM_CACHE.move_to_end(key)
    return run(**kwargs)
