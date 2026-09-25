"""Before/after comparison of two jaxphys revisions on identical workloads.

Usage (from the repository root)::

    JAX_PLATFORMS=cpu python -m benchmarks.compare_revisions --base e51cc7c
    JAX_PLATFORMS=cpu python -m benchmarks.compare_revisions --base e51cc7c --head HEAD

``--base``/``--head`` are git revisions (``--head`` defaults to the working
tree).  Each revision's ``src/`` is exported with ``git archive`` to a
temporary directory; every case then runs in a fresh Python process with
that tree first on ``PYTHONPATH``, alternating base/head for ``--rounds``
rounds so that machine-load drift affects both equally. Only the public
API is used, so the same workload runs unchanged on both revisions. Solvers
added after the base revision (FDFD, SPH, 1D Euler, tight-binding, 2D
Schrodinger) have no "before" and are covered by ``benchmarks.run`` instead.

The workloads are identical, but the algorithms are not always: between
e51cc7c and 0.2.0 several solvers were corrected (split-field PML in the
FDTD solvers, Boris pushes for charges, the Schrodinger grid, the Wolff
cluster rule, LBM boundaries). The table measures the cost of the code a
user gets before and after, not a like-for-like kernel comparison.

Per case and revision the result is the median over rounds of: the first
call (trace + compile + run), the median of 5 warm calls, XLA compilations
per warm call, and the process peak RSS.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cases() -> dict[str, tuple[str, Callable[[], Any]]]:
    import jax
    import jax.numpy as jnp

    import jaxphys as jp

    def ho(q: Any, p: Any, params: Any) -> Any:
        return 0.5 * p[0] ** 2 + 0.5 * q[0] ** 2

    def dp_lag(q: Any, qd: Any, params: Any) -> Any:
        t1, t2 = q
        w1, w2 = qd
        T = 0.5 * w1**2 + 0.5 * (w1**2 + w2**2 + 2 * w1 * w2 * jnp.cos(t1 - t2))
        V = -2 * 9.81 * jnp.cos(t1) - 9.81 * jnp.cos(t2)
        return T - V

    key = jax.random.PRNGKey(0)
    hs = jp.HamiltonianSystem(ho, n_dof=1)
    ls = jp.LagrangianSystem(dp_lag, n_dof=2)
    nb = jp.NBody(
        jnp.ones(64) / 64,
        jax.random.normal(key, (64, 3)),
        jnp.zeros((64, 3)),
        softening=0.05,
    )
    rb = jp.RigidBody([1.0, 2.0, 3.0])
    cs = jp.ChargeSystem(
        [
            jp.PointCharge(1e-6, 1e-3, [0, 0, 0], [0, 0, 0]),
            jp.PointCharge(-1e-6, 1e-3, [0.1, 0, 0], [0, 1, 0]),
        ]
    )
    g2 = jp.EMGrid(size=(200, 200), resolution=0.01)
    g2.add_source(jp.PlaneWave(frequency=3e9, y=20))
    dt2 = 0.99 * 0.01 / (299792458.0 * 2.0**0.5)  # default dt on both revisions
    g3 = jp.EMGrid3D(size=(40, 40, 40), resolution=0.01)
    g3.add_source(jp.PointSource3D(frequency=3e9, position=(20, 20, 20)))
    dt3 = 0.99 * 0.01 / (299792458.0 * 3.0**0.5)
    lb = jp.LBMGrid(size=(200, 80), viscosity=0.02)
    ns = jp.NavierStokesSolver(size=(64, 64), viscosity=0.01)
    il = jp.IsingLattice(size=(16, 16))
    rho0 = jp.DensityMatrix.from_pure_state(jnp.array([1.0, 0.0], dtype=jnp.complex128))
    H = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    L = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)
    return {
        "hamiltonian": (
            "HamiltonianSystem leapfrog, 1 dof, 20000 steps, save_every=100",
            lambda: hs.simulate([1.0], [0.0], (0, 200.0), dt=0.01, save_every=100),
        ),
        "lagrangian": (
            "LagrangianSystem double pendulum rk4, 20000 steps, save_every=100",
            lambda: ls.simulate(
                [0.5, 1.0], [0.0, 0.0], (0, 20.0), dt=0.001, save_every=100
            ),
        ),
        "nbody": (
            "NBody N=64, 20000 steps, save_every=100",
            lambda: nb.simulate((0, 1.0), n_steps=20000, save_every=100),
        ),
        "rigid_body": (
            "RigidBody rk4, 20000 steps",
            lambda: rb.simulate([1.0, 0.1, 0.0], (0, 200.0), dt=0.01),
        ),
        "charges": (
            "ChargeSystem 2 charges, 20000 steps, save_every=100",
            lambda: cs.simulate((0, 1e-2), n_steps=20000, save_every=100),
        ),
        "schrodinger": (
            "solve_schrodinger 2048 points, 20000 steps, save_every=100",
            lambda: jp.solve_schrodinger(
                jp.GaussianWavepacket(-5.0, 2.0, 0.5),
                jp.HarmonicPotential(k=0.0),
                (-20, 20),
                (0, 20.0),
                n_points=2048,
                dt=0.001,
                save_every=100,
            ),
        ),
        "fdtd2d": (
            "EMGrid 200x200, 856 steps, save_every=50",
            lambda: g2.simulate((0, 2e-8), dt=dt2, save_every=50),
        ),
        "fdtd3d": (
            "EMGrid3D 40^3, 157 steps, save_every=20",
            lambda: g3.simulate((0, 3e-9), dt=dt3, save_every=20),
        ),
        "lbm": (
            "LBMGrid 200x80, 3000 steps, save_every=100",
            lambda: lb.simulate(n_steps=3000, u_inlet=0.04, save_every=100),
        ),
        "navier_stokes": (
            "NavierStokesSolver 64x64, 2000 steps, 50 Jacobi its, save_every=100",
            lambda: ns.simulate(n_steps=2000, dt=0.001, save_every=100),
        ),
        "ising_metropolis": (
            "IsingLattice.run_metropolis 16x16, 50+200 sweeps",
            lambda: il.run_metropolis(
                2.3, n_sweeps=200, n_warmup=50, key=jax.random.PRNGKey(1)
            ),
        ),
        "ising_wolff": (
            "sweep_temperatures Wolff 16x16, 1 temperature, 20+100 updates",
            lambda: jp.sweep_temperatures(
                il,
                jnp.array([2.3]),
                n_sweeps=100,
                n_warmup=20,
                algorithm="wolff_cluster",
                key=jax.random.PRNGKey(1),
            ),
        ),
        "lindblad": (
            "lindblad_evolve 2-level, 20000 steps, save_every=100",
            lambda: jp.lindblad_evolve(
                rho0, H, [L], [0.1], (0, 200.0), dt=0.01, save_every=100
            ),
        ),
    }


def _run_case(name: str) -> dict[str, Any]:
    """Run one case in this process (called in a subprocess)."""
    import jax
    from jax import monitoring

    count = [0]

    def listener(event: str, duration: float, **_: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            count[0] += 1

    monitoring.register_event_duration_secs_listener(listener)
    desc, fn = _cases()[name]

    def call() -> None:
        res = fn()
        leaves = (
            [getattr(res, f) for f in res.__dataclass_fields__]
            if hasattr(res, "__dataclass_fields__")
            else jax.tree_util.tree_leaves(res)
        )
        for leaf in leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    t0 = time.perf_counter()
    call()
    first = time.perf_counter() - t0
    before = count[0]
    warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        call()
        warm.append(time.perf_counter() - t0)
    return {
        "case": name,
        "description": desc,
        "first_call_s": first,
        "warm_median_s": statistics.median(warm),
        "warm_samples_s": warm,
        "compiles_per_warm_call": (count[0] - before) / 5,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def _export(rev: str | None, dest: str) -> str:
    if rev is None:
        return os.path.join(REPO, "src")
    os.makedirs(dest, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", rev, "src"], cwd=REPO, check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=archive, check=True)
    return os.path.join(dest, "src")


def _rev_id(rev: str | None) -> str:
    if rev is None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        return f"working tree (HEAD {head[:12]})"
    return subprocess.run(
        ["git", "rev-parse", rev], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--case", action="append", help="subset of cases")
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    parser.add_argument("--run-case", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.run_case:
        print(json.dumps(_run_case(args.run_case)))
        return 0

    from benchmarks.common import environment

    names = args.case or list(_cases().keys())
    results: dict[str, dict[str, list[dict[str, Any]]]] = {
        n: {"base": [], "head": []} for n in names
    }
    with tempfile.TemporaryDirectory() as tmp:
        trees = {
            "base": _export(args.base, os.path.join(tmp, "base")),
            "head": _export(args.head, os.path.join(tmp, "head")),
        }
        for rnd in range(args.rounds):
            for name in names:
                for side in ("base", "head"):
                    env = dict(
                        os.environ, PYTHONPATH=os.pathsep.join([trees[side], REPO])
                    )
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.compare_revisions",
                            "--base",
                            "-",
                            "--run-case",
                            name,
                        ],
                        cwd=REPO,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0:
                        print(proc.stderr[-2000:], file=sys.stderr)
                        raise SystemExit(f"case {name} failed on {side}")
                    results[name][side].append(
                        json.loads(proc.stdout.strip().splitlines()[-1])
                    )
                    r = results[name][side][-1]
                    print(
                        f"round {rnd} {name:18s} {side}: warm {r['warm_median_s']:.4f}s",
                        flush=True,
                    )

    rows = []
    for name in names:
        row: dict[str, Any] = {
            "case": name,
            "description": results[name]["base"][0]["description"],
        }
        for side in ("base", "head"):
            rs = results[name][side]
            row[side] = {
                "first_call_s": statistics.median(r["first_call_s"] for r in rs),
                "warm_median_s": statistics.median(r["warm_median_s"] for r in rs),
                "warm_medians_per_round_s": [r["warm_median_s"] for r in rs],
                "compiles_per_warm_call": statistics.median(
                    r["compiles_per_warm_call"] for r in rs
                ),
                "peak_rss_mb": statistics.median(r["peak_rss_mb"] for r in rs),
            }
        row["warm_speedup"] = (
            row["base"]["warm_median_s"] / row["head"]["warm_median_s"]
        )
        rows.append(row)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_before_after"
    out = {
        "run_id": run_id,
        "base": _rev_id(args.base),
        "head": _rev_id(args.head),
        "rounds": args.rounds,
        "environment": environment(),
        "rows": rows,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{run_id}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    lines = [
        f"# Before/after: `{out['base'][:12]}` -> {out['head']}",
        "",
        f"CPU: {out['environment']['cpu_model']} ({out['environment']['cpu_count']} "
        f"logical CPUs), JAX {out['environment']['jax']} on "
        f"{', '.join(out['environment']['jax_devices'])}; no GPU. "
        f"Median over {args.rounds} interleaved process rounds; each round = "
        "median of 5 warm calls after one first call.",
        "",
        "| case | first call before | first call after | warm before | warm after | "
        "speedup | compiles/warm call before -> after | peak RSS before -> after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b, h = r["base"], r["head"]
        lines.append(
            f"| {r['description']} | {b['first_call_s']:.3f} s | {h['first_call_s']:.3f} s | "
            f"{b['warm_median_s']:.4f} s | {h['warm_median_s']:.4f} s | "
            f"{r['warm_speedup']:.1f}x | {b['compiles_per_warm_call']:g} -> "
            f"{h['compiles_per_warm_call']:g} | {b['peak_rss_mb']:.0f} -> "
            f"{h['peak_rss_mb']:.0f} MB |"
        )
    with open(os.path.join(args.out, f"{run_id}.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
