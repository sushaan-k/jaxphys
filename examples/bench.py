"""Benchmark jaxphys solvers against straightforward NumPy implementations.

Each row runs the same numerical scheme twice: through the public jaxphys
API (JIT-compiled; the first call, which compiles, is excluded) and through
a plain NumPy reference written here. The final states must agree to a
tight tolerance before a timing is reported. Timings are the best of
``--repeats`` runs, with ``block_until_ready()`` on the JAX results.

Usage:
    python examples/bench.py            # sizes used for the README table
    python examples/bench.py --quick    # small sizes, finishes in seconds
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from scipy.spatial import cKDTree

import jaxphys as jp
from jaxphys.em.fdtd import C0, EPS0, ETA0, MU0

# --------------------------------------------------------------- utilities


def best_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Best wall time of ``repeats`` calls of ``fn`` and its last result."""
    best, out = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        out = jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - start)
    return best, out


def hardware_summary() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            names = [
                line.split(":", 1)[1].strip() for line in f if "model name" in line
            ]
        cpu = names[0] if names else cpu
    except OSError:
        pass
    devices = ", ".join(str(d) for d in jax.devices())
    return (
        f"CPU: {cpu} ({os.cpu_count()} logical cores)\n"
        f"JAX devices: {devices}\n"
        f"Python {platform.python_version()}, jax {jax.__version__}, "
        f"numpy {np.__version__}, jaxphys {jp.__version__}"
    )


# ------------------------------------------------------------------ N-body


def nbody_numpy(pos, vel, masses, G, eps, dt, n_steps):
    def accel(p):
        dr = p[None, :, :] - p[:, None, :]
        d2 = np.sum(dr**2, axis=-1) + eps**2
        np.fill_diagonal(d2, 1.0)
        w = masses[None, :] * d2**-1.5
        np.fill_diagonal(w, 0.0)
        return G * np.einsum("ij,ijk->ik", w, dr)

    a = accel(pos)
    for _ in range(n_steps):
        pos = pos + vel * dt + 0.5 * a * dt**2
        a_new = accel(pos)
        vel = vel + 0.5 * (a + a_new) * dt
        a = a_new
    return pos


def bench_nbody(n: int, n_steps: int, repeats: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    pos, vel = rng.normal(size=(n, 3)), 0.1 * rng.normal(size=(n, 3))
    masses, G, eps, dt = np.full(n, 1.0 / n), 1.0, 0.05, 1e-3
    system = jp.NBody(masses, pos, vel, G=G, softening=eps)

    def run_jax():
        return system.simulate((0.0, n_steps * dt), n_steps=n_steps, save_every=n_steps)

    run_jax()  # compile
    t_jax, traj = best_time(run_jax, repeats)
    t_np, ref = best_time(lambda: nbody_numpy(pos, vel, masses, G, eps, dt, n_steps), 1)
    np.testing.assert_allclose(np.asarray(traj.positions[-1]), ref, rtol=0, atol=1e-9)
    return {
        "name": f"N-body, velocity Verlet (N={n}, {n_steps} steps)",
        "numpy": t_np,
        "jax": t_jax,
    }


# ---------------------------------------------------------- Schrodinger 2D


def schrodinger_numpy(psi, V, k2, dt, n_steps):
    exp_v = np.exp(-0.5j * V * dt)
    exp_t = np.exp(-0.5j * k2 * dt)
    for _ in range(n_steps):
        psi = exp_v * np.fft.ifft2(exp_t * np.fft.fft2(exp_v * psi))
    return psi


def bench_schrodinger(n: int, n_steps: int, repeats: int) -> dict[str, Any]:
    L, dt = 20.0, 1e-3
    packet = jp.GaussianWavepacket2D(x0=-2.0, y0=0.5, kx=3.0, ky=-1.0, sigma=0.8)

    def potential(X, Y):
        return 0.5 * (X**2 + 2.0 * Y**2)

    def run_jax():
        return jp.solve_schrodinger_2d(
            packet,
            potential,
            x_range=(-L / 2, L / 2),
            y_range=(-L / 2, L / 2),
            n_points=(n, n),
            t_span=(0.0, n_steps * dt * (1 + 1e-9)),
            dt=dt,
            save_every=n_steps,
        )

    run_jax()
    t_jax, result = best_time(run_jax, repeats)
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    psi0 = np.asarray(result.psi[0])
    t_np, ref = best_time(
        lambda: schrodinger_numpy(psi0, potential(X, Y), k2, dt, n_steps), 1
    )
    np.testing.assert_allclose(np.asarray(result.psi[-1]), ref, atol=1e-10)
    return {
        "name": f"Schrodinger 2D, split-operator ({n}x{n}, {n_steps} steps)",
        "numpy": t_np,
        "jax": t_jax,
    }


# --------------------------------------------------------------------- SPH


def sph_numpy(fluid: jp.SPHFluid, pos, vel, dt, n_steps):
    h, m, L = fluid.h, fluid.mass, np.asarray(fluid.box)
    sigma = 10.0 / (7.0 * np.pi * h**2)
    B = fluid.rest_density * fluid.sound_speed**2 / fluid.gamma

    def acceleration(p, v):
        pairs = cKDTree(p, boxsize=L).query_pairs(2 * h, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        dr = p[i] - p[j]
        dr -= L * np.round(dr / L)
        r = np.linalg.norm(dr, axis=1)
        q = r / h
        w = sigma * np.where(q < 1, 1 - 1.5 * q**2 + 0.75 * q**3, 0.25 * (2 - q) ** 3)
        dw = sigma / h * np.where(q < 1, -3 * q + 2.25 * q**2, -0.75 * (2 - q) ** 2)
        n = len(p)
        rho = m * (sigma + np.bincount(i, w, n) + np.bincount(j, w, n))
        pr = B * ((rho / fluid.rest_density) ** fluid.gamma - 1)
        vr = np.sum((v[i] - v[j]) * dr, axis=1)
        mu = h * vr / (r**2 + 0.01 * h**2)
        visc = np.where(
            vr < 0,
            -fluid.alpha * fluid.sound_speed * mu / (0.5 * (rho[i] + rho[j])),
            0.0,
        )
        f = (-m * (pr[i] / rho[i] ** 2 + pr[j] / rho[j] ** 2 + visc) * dw / r)[
            :, None
        ] * dr
        acc = np.stack(
            [np.bincount(i, f[:, d], n) - np.bincount(j, f[:, d], n) for d in (0, 1)], 1
        )
        return acc

    a = acceleration(pos, vel)
    for _ in range(n_steps):
        v_half = vel + 0.5 * dt * a
        pos = np.mod(pos + dt * v_half, L)
        a = acceleration(pos, v_half)
        vel = v_half + 0.5 * dt * a
    return pos


def bench_sph(n_side: int, n_steps: int, repeats: int) -> dict[str, Any]:
    dx = 1.0 / n_side
    xs = (np.arange(n_side) + 0.5) * dx
    pos = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    vel = 0.05 * np.random.default_rng(1).normal(size=pos.shape)
    fluid = jp.SPHFluid(
        mass=1000.0 * dx**2, smoothing_length=1.3 * dx, box=(1.0, 1.0), sound_speed=10.0
    )
    dt = 0.2 * fluid.h / fluid.sound_speed

    def run_jax():
        return fluid.simulate(
            pos, vel, (0.0, n_steps * dt * (1 + 1e-9)), dt, save_every=n_steps
        )

    run_jax()
    t_jax, traj = best_time(run_jax, repeats)
    t_np, ref = best_time(lambda: sph_numpy(fluid, pos, vel, dt, n_steps), 1)
    np.testing.assert_allclose(np.asarray(traj.positions[-1]), ref, rtol=0, atol=1e-9)
    return {
        "name": f"SPH, weakly compressible ({n_side**2} particles, {n_steps} steps)",
        "numpy": t_np,
        "jax": t_jax,
    }


# ----------------------------------------------------------------- FDTD 3D


def fdtd3d_numpy(n, n_pml, dx, dt, src, freq, n_steps):
    def loss(pos):
        sigma_max = 0.8 * 4 / (ETA0 * dx)
        depth = np.clip(np.maximum(n_pml - pos, pos - (n - 1 - n_pml)) / n_pml, 0, 1)
        return sigma_max * depth**3 * dt / EPS0

    def decay(a):
        safe = np.where(a > 0, a, 1.0)
        return np.exp(-a), np.where(a > 0, -np.expm1(-safe) / safe, 1.0)

    idx = np.arange(n, dtype=float)
    shapes = [(n, 1, 1), (1, n, 1), (1, 1, n)]
    e_dec = [tuple(x.reshape(s) for x in decay(loss(idx))) for s in shapes]
    h_dec = [tuple(x.reshape(s) for x in decay(loss(idx + 0.5))) for s in shapes]
    hc, ec = dt / (MU0 * dx), dt / (EPS0 * dx)

    def fwd(a, ax):
        d = np.zeros_like(a)
        sl = [slice(None)] * 3
        sl[ax] = slice(0, -1)
        d[tuple(sl)] = np.diff(a, axis=ax)
        return d

    def bwd(a, ax):
        d = np.zeros_like(a)
        sl = [slice(None)] * 3
        sl[ax] = slice(1, None)
        d[tuple(sl)] = np.diff(a, axis=ax)
        return d

    def part(old, dec, coef, drive):
        return dec[0] * old + coef * dec[1] * drive

    edge = np.zeros((n, n, n), bool)
    edge[[0, -1]] = edge[:, [0, -1]] = edge[:, :, [0, -1]] = True
    f = [np.zeros((n, n, n)) for _ in range(12)]
    for step in range(n_steps):
        exy, exz, eyz, eyx, ezx, ezy, hxy, hxz, hyz, hyx, hzx, hzy = f
        ex, ey, ez = exy + exz, eyz + eyx, ezx + ezy
        hxy = part(hxy, h_dec[1], -hc, fwd(ez, 1))
        hxz = part(hxz, h_dec[2], hc, fwd(ey, 2))
        hyz = part(hyz, h_dec[2], -hc, fwd(ex, 2))
        hyx = part(hyx, h_dec[0], hc, fwd(ez, 0))
        hzx = part(hzx, h_dec[0], -hc, fwd(ey, 0))
        hzy = part(hzy, h_dec[1], hc, fwd(ex, 1))
        hx, hy, hz = hxy + hxz, hyz + hyx, hzx + hzy
        exy = part(exy, e_dec[1], ec, bwd(hz, 1))
        exz = part(exz, e_dec[2], -ec, bwd(hy, 2))
        eyz = part(eyz, e_dec[2], ec, bwd(hx, 2))
        eyx = part(eyx, e_dec[0], -ec, bwd(hz, 0))
        ezx = part(ezx, e_dec[0], ec, bwd(hy, 0))
        ezy = part(ezy, e_dec[1], -ec, bwd(hx, 1))
        ezx[src] += np.sin(2 * np.pi * freq * step * dt)
        e = [np.where(edge, 0.0, a) for a in (exy, exz, eyz, eyx, ezx, ezy)]
        f = [*e, hxy, hxz, hyz, hyx, hzx, hzy]
    return f[4] + f[5]


def bench_fdtd3d(n: int, n_steps: int, repeats: int) -> dict[str, Any]:
    dx, freq, n_pml = 0.01, 3e9, 8
    dt = 0.99 * dx / (C0 * np.sqrt(3))
    src = (n // 2, n // 2, n // 2)
    grid = jp.EMGrid3D(size=(n, n, n), resolution=dx, pml_layers=n_pml)
    grid.add_source(jp.PointSource3D(frequency=freq, position=src))

    def run_jax():
        # Snapshots follow steps 1, 1 + save_every, ...: save after step n_steps.
        return grid.simulate(
            t_span=(0.0, n_steps * dt * (1 + 1e-9)), dt=dt, save_every=n_steps - 1
        )

    run_jax()
    t_jax, fields = best_time(run_jax, repeats)
    t_np, ref = best_time(lambda: fdtd3d_numpy(n, n_pml, dx, dt, src, freq, n_steps), 1)
    scale = np.max(np.abs(ref))
    np.testing.assert_allclose(
        np.asarray(fields.ez[-1]), ref, rtol=0, atol=1e-9 * scale
    )
    return {
        "name": f"FDTD 3D Yee + split-field PML ({n}^3, {n_steps} steps)",
        "numpy": t_np,
        "jax": t_jax,
    }


# -------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small problem sizes")
    parser.add_argument("--repeats", type=int, default=3, help="timed JAX runs per row")
    args = parser.parse_args()

    # (N, steps) per benchmark; the NumPy reference runs once per row.
    sizes = (
        {"nbody": (256, 50), "schrodinger": (64, 50), "sph": (24, 20), "fdtd": (24, 20)}
        if args.quick
        else {
            "nbody": (1000, 200),
            "schrodinger": (256, 500),
            "sph": (64, 200),
            "fdtd": (64, 200),
        }
    )
    print(hardware_summary())
    print()
    rows = [
        bench_nbody(*sizes["nbody"], args.repeats),
        bench_schrodinger(*sizes["schrodinger"], args.repeats),
        bench_sph(*sizes["sph"], args.repeats),
        bench_fdtd3d(*sizes["fdtd"], args.repeats),
    ]
    print("| Simulation | NumPy | jaxphys (JIT) | Speedup |")
    print("|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['name']} | {row['numpy']:.2f} s | {row['jax']:.2f} s | "
            f"{row['numpy'] / row['jax']:.1f}x |"
        )
    print("\nAll jaxphys results matched the NumPy references.")


if __name__ == "__main__":
    main()
