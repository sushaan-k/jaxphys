"""The shared strided rollout used by every solver's time loop."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxphys._rollout import call_with_params, strided_rollout


def _step(c: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    x, t = c
    return jnp.sin(x) * 0.9 + 0.1 * jnp.cos(3.0 * x) + 0.01 * t, t + 0.1


def _flat(c: Any, n_steps: int) -> Any:
    def body(c: Any, _: None) -> tuple[Any, Any]:
        c = _step(c)
        return c, c

    _, hist = jax.lax.scan(body, c, None, length=n_steps)
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b]), c, hist)


@pytest.mark.parametrize("n_steps", [0, 1, 7, 23])
@pytest.mark.parametrize("save_every", [1, 3, 50])
def test_matches_flat_scan_subsample(n_steps: int, save_every: int) -> None:
    c0 = (jnp.linspace(-1.0, 1.0, 5), jnp.asarray(0.0))
    xs, ts = jax.jit(_flat, static_argnums=1)(c0, n_steps)
    run = jax.jit(lambda c: strided_rollout(_step, c, n_steps, save_every, lambda c: c))
    rx, rt = run(c0)
    idx = np.arange(0, n_steps + 1, save_every)
    np.testing.assert_array_equal(np.asarray(rx), np.asarray(xs)[idx])
    np.testing.assert_array_equal(np.asarray(rt), np.asarray(ts)[idx])


def test_checkpointed_gradient_matches_flat_scan() -> None:
    def loss_strided(x0: jax.Array) -> jax.Array:
        xs, _ = strided_rollout(_step, (x0, jnp.asarray(0.0)), 40, 8, lambda c: c)
        return jnp.sum(xs**2)

    def loss_flat(x0: jax.Array) -> jax.Array:
        xs, _ = _flat((x0, jnp.asarray(0.0)), 40)
        return jnp.sum(xs[::8] ** 2)

    x0 = jnp.linspace(-1.0, 1.0, 5)
    np.testing.assert_allclose(
        np.asarray(jax.grad(loss_strided)(x0)),
        np.asarray(jax.grad(loss_flat)(x0)),
        rtol=1e-12,
    )


def _scaled_impl(x: jax.Array, params: Any, n: int) -> jax.Array:
    scale = params["a"] if params["a"] > 0 else -params["a"]
    return x * scale + n


_scaled = jax.jit(_scaled_impl, static_argnames=("n",))


def test_call_with_params_falls_back_to_constants_and_caches() -> None:
    x = jnp.arange(3.0)
    # Traced path for plain numbers ...
    out = call_with_params(_scaled, _scaled_impl, {"a": -2.0}, ("n",), x=x, n=1)
    np.testing.assert_array_equal(np.asarray(out), [1.0, 3.0, 5.0])
    # ... which needs a concrete value here, so it is compiled in as a constant
    # and the executable is reused for equal parameter values.
    from jaxphys import _rollout

    before = len(_rollout._CONSTANT_PARAM_CACHE)
    call_with_params(_scaled, _scaled_impl, {"a": -2.0}, ("n",), x=x, n=1)
    assert len(_rollout._CONSTANT_PARAM_CACHE) == before
    call_with_params(_scaled, _scaled_impl, {"a": -3.0}, ("n",), x=x, n=1)
    assert len(_rollout._CONSTANT_PARAM_CACHE) == before + 1
