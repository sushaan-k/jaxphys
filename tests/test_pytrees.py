"""Parameter, state and solver containers behave as JAX pytrees."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

import jaxphys as jp
import jaxphys.state as state
from jaxphys.config import Params


def test_params_roundtrip_through_tree_util() -> None:
    p = Params(m=1.0, k=4.0, name="spring")
    leaves, treedef = jax.tree_util.tree_flatten(p)
    assert leaves == [4.0, 1.0, "spring"]  # one leaf per field, sorted by name
    q = jax.tree_util.tree_unflatten(treedef, [8.0, 2.0, "spring"])
    assert isinstance(q, Params)
    assert (q.m, q.k, q.name) == (2.0, 8.0, "spring")


def test_params_pass_through_jit_grad_and_vmap() -> None:
    def energy(params: Params) -> jax.Array:
        return 0.5 * params.k * params.x**2

    p = Params(k=4.0, x=0.5)
    assert float(jax.jit(energy)(p)) == 0.5
    g = jax.grad(energy)(p)
    assert isinstance(g, Params)
    assert float(g.k) == 0.125
    assert float(g.x) == 2.0
    batch = Params(k=jnp.array([1.0, 2.0, 3.0]), x=jnp.array([1.0, 1.0, 1.0]))
    assert jnp.allclose(jax.vmap(energy)(batch), jnp.array([0.5, 1.0, 1.5]))


def test_trajectory_is_a_pytree_usable_under_vmap() -> None:
    def make(q0: jax.Array) -> state.Trajectory:
        t = jnp.linspace(0.0, 1.0, 5)
        q = q0 * jnp.cos(t)[:, None]
        return state.Trajectory(
            t=t, q=q, p=-q0 * jnp.sin(t)[:, None], energy=0.5 * q0**2
        )

    batch = jax.vmap(make)(jnp.array([[1.0], [2.0], [3.0]]))
    assert isinstance(batch, state.Trajectory)
    assert batch.q.shape == (3, 5, 1)
    doubled = jax.tree_util.tree_map(lambda x: 2 * x, make(jnp.array([1.0])))
    assert isinstance(doubled, state.Trajectory)
    assert float(doubled.q[0, 0]) == 2.0
    assert doubled.metadata is None


def test_all_result_containers_flatten() -> None:
    for name in (
        "PhaseState",
        "Trajectory",
        "NBodyState",
        "NBodyTrajectory",
        "EMFieldState",
        "EMFieldHistory",
        "EMFieldHistory3D",
        "QuantumState",
        "QuantumResult",
        "QuantumResult2D",
        "FluidState",
        "FluidHistory",
        "IsingResult",
    ):
        cls = getattr(state, name)
        values = {f.name: jnp.ones(2) for f in dataclasses.fields(cls)}
        values.pop("metadata", None)  # static metadata, not a leaf
        obj = cls(**values)
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        assert len(leaves) == len(values), name
        assert type(jax.tree_util.tree_unflatten(treedef, leaves)) is cls, name


def test_sph_fluid_constants_are_leaves_and_grid_settings_static() -> None:
    fluid = jp.SPHFluid(
        mass=0.5, smoothing_length=0.05, box=(1.0, 2.0), gravity=(0.0, -9.8)
    )
    leaves, treedef = jax.tree_util.tree_flatten(fluid)
    assert leaves == [0.5, 1000.0, 10.0, 7.0, 0.1, 0.0, -9.8]
    rebuilt = jax.tree_util.tree_unflatten(treedef, [2 * x for x in leaves])
    assert isinstance(rebuilt, jp.SPHFluid)
    assert (rebuilt.mass, rebuilt.gravity) == (1.0, (0.0, -19.6))
    assert (rebuilt.h, rebuilt.box, rebuilt.boundary) == (0.05, (1.0, 2.0), "periodic")
    # d(pressure)/d(sound speed) through the pytree.
    grad = jax.grad(lambda f: f.pressure(jnp.asarray(1010.0)))(fluid)
    assert float(grad.sound_speed) > 0.0
