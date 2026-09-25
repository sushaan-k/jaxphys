"""vmap batch-scaling curves.

Every public ``simulate`` / ``solve_*`` entry point runs under ``jax.vmap``
(host-side checks are skipped for traced values), so the curves batch the
public API directly with ``jax.jit(jax.vmap(...))`` over initial
conditions or parameters, which is how batched studies are written with
jaxphys. The Ising curve batches temperatures through
``sweep_temperatures``, which vmaps its chains internally.

For each batch size B we report the median wall time of one batched call
and the throughput in trajectories per second; ``efficiency`` is
``B * t(1) / t(B)`` (values > 1 mean batching beats running B single
trajectories back to back).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxphys as jp
from benchmarks.common import block, compile_accounting, summarize, time_calls
from jaxphys.em.fdtd import C0

Kernel = tuple[str, str, Callable[[int], Callable[[], Any]]]


def _hamiltonian_kernel() -> Kernel:
    def h(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
        return 0.5 * p[0] ** 2 + 0.5 * params.k * q[0] ** 2 + 0.25 * q[0] ** 4

    system = jp.HamiltonianSystem(h, n_dof=1)

    def make(batch: int) -> Callable[[], Any]:
        q0 = jnp.linspace(0.1, 2.0, batch)[:, None]
        ks = jnp.linspace(1.0, 4.0, batch)

        def one(q: jax.Array, k: jax.Array) -> Any:
            return system.simulate(
                q, [0.0], (0.0, 20.0005), 1e-3, jp.Params(k=k), save_every=100
            ).q

        fn = jax.jit(jax.vmap(one))
        return lambda: fn(q0, ks)

    return "HamiltonianSystem (anharmonic, leapfrog)", "20000 steps", make


def _nbody_kernel() -> Kernel:
    n = 32
    # NumPy (concrete) masses: NBody validates them on construction.
    pos = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (n, 3)))
    m = np.full(n, 1.0 / n)

    def make(batch: int) -> Callable[[], Any]:
        vel = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (batch, n, 3))

        def one(v: jax.Array) -> Any:
            system = jp.NBody(m, pos, v, softening=0.05)
            return system.simulate((0.0, 0.2), n_steps=2000, save_every=100).positions

        fn = jax.jit(jax.vmap(one))
        return lambda: fn(vel)

    return "NBody (N=32)", "2000 steps", make


def _schrodinger_kernel() -> Kernel:
    n = 1024
    x = np.linspace(-20.0, 20.0, n, endpoint=False)

    def make(batch: int) -> Callable[[], Any]:
        x0 = np.linspace(-5.0, 5.0, batch)[:, None]
        psi = jnp.asarray(np.exp(-((x[None, :] - x0) ** 2)) + 0j)

        def one(p: jax.Array) -> Any:
            return jp.solve_schrodinger(
                p,
                jp.HarmonicPotential(k=0.1),
                (-20.0, 20.0),
                (0.0, 2.0005),
                n_points=n,
                dt=1e-3,
                save_every=100,
            ).psi

        fn = jax.jit(jax.vmap(one))
        return lambda: fn(psi)

    return "solve_schrodinger (1024 points)", "2000 steps", make


def _fdtd_kernel() -> Kernel:
    n, dx = 100, 0.01
    dt = 0.99 * dx / (C0 * float(np.sqrt(2.0)))

    def make(batch: int) -> Callable[[], Any]:
        freqs = jnp.linspace(2e9, 4e9, batch)

        def one(f: jax.Array) -> Any:
            grid = jp.EMGrid(size=(n, n), resolution=dx)
            grid.add_source(jp.PlaneWave(frequency=f, y=20))  # type: ignore[arg-type]
            grid.add_conductor(jp.Wall(y=50, gap_start=45, gap_end=55))
            return grid.simulate((0.0, 300.5 * dt), dt=dt, save_every=50).ez

        fn = jax.jit(jax.vmap(one))
        return lambda: fn(freqs)

    return "EMGrid 2D FDTD (100x100), batch of source frequencies", "300 steps", make


def _lbm_kernel() -> Kernel:
    nx, ny = 100, 40
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    grid = jp.LBMGrid(size=(nx, ny), viscosity=0.05)
    grid.add_obstacle(jp.Obstacle(mask=jnp.asarray((x - 25) ** 2 + (y - 20) ** 2 < 25)))

    def make(batch: int) -> Callable[[], Any]:
        amp = jnp.linspace(0.0, 0.01, batch)
        shape = jnp.asarray(np.sin(2 * np.pi * y / ny))

        def one(a: jax.Array) -> Any:
            return grid.simulate(
                n_steps=300,
                u_inlet=0.04,
                save_every=100,
                initial_uy=a * shape,
            ).ux

        fn = jax.jit(jax.vmap(one))
        return lambda: fn(amp)

    return "LBMGrid (100x40), batch of initial perturbations", "300 steps", make


def _ising_kernel() -> Kernel:
    lattice = jp.IsingLattice(size=(16, 16))

    def make(batch: int) -> Callable[[], Any]:
        temps = jnp.linspace(1.5, 3.5, batch)
        return lambda: (
            jp.sweep_temperatures(
                lattice, temps, n_sweeps=20, n_warmup=0, key=jax.random.PRNGKey(0)
            ).energies
        )

    return (
        "sweep_temperatures Metropolis (16x16), batch of temperatures",
        "20 sweeps",
        make,
    )


KERNELS = [
    _hamiltonian_kernel,
    _nbody_kernel,
    _schrodinger_kernel,
    _fdtd_kernel,
    _lbm_kernel,
    _ising_kernel,
]


def run_scaling(batches: list[int], repeats: int) -> list[dict[str, Any]]:
    out = []
    for build in KERNELS:
        name, work, make = build()
        rows = []
        for b in batches:
            fn = make(b)
            with compile_accounting() as comp:
                block(fn())
            block(fn())
            stats = summarize(time_calls(fn, repeats))
            rows.append(
                {
                    "batch": b,
                    "compile_s": comp["compile_s"],
                    "steady": stats,
                    "trajectories_per_s": b / stats["median_s"],
                }
            )
        t1 = rows[0]["steady"]["median_s"] / rows[0]["batch"]
        for r in rows:
            r["efficiency"] = r["batch"] * t1 / r["steady"]["median_s"]
        out.append({"kernel": name, "work_per_trajectory": work, "rows": rows})
    return out
