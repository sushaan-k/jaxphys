"""Per-solver performance cases: JAX (public API) vs a NumPy reference.

Every case times the *public* jaxphys entry point (``simulate`` /
``solve_schrodinger`` / ...) exactly as a user calls it, so the numbers
include input validation and the host-side NaN checks. JIT compile time is
measured separately from steady-state time (see ``common.measure_jax``).
The NumPy reference runs the same configuration and its output is compared
with the JAX output; the comparison tolerance is stated per case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxphys as jp
from benchmarks import numpy_ref as ref
from benchmarks.common import max_rel_diff, measure_jax, measure_numpy
from jaxphys.em.fdtd import C0


@dataclass
class Case:
    solver: str
    size: str
    params: dict[str, Any]
    jax_fn: Callable[[], Any]
    numpy_fn: Callable[[], Any]
    compare: Callable[[Any, Any], dict[str, Any]]
    work: float
    work_unit: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _allclose_check(pairs: list[tuple[str, Any, Any]], rtol: float) -> dict[str, Any]:
    """Compare arrays; relative error is normalized by max |reference|."""
    worst_abs, worst_rel, detail = 0.0, 0.0, {}
    for name, a, b in pairs:
        a_np, b_np = np.asarray(a), np.asarray(b)
        if a_np.shape != b_np.shape:
            return {
                "passed": False,
                "reason": f"shape mismatch for {name}: {a_np.shape} vs {b_np.shape}",
            }
        d_abs, d_rel = max_rel_diff(a_np, b_np)
        detail[name] = {"max_abs": d_abs, "max_rel": d_rel}
        worst_abs, worst_rel = max(worst_abs, d_abs), max(worst_rel, d_rel)
    return {
        "kind": "deterministic",
        "rtol": rtol,
        "max_abs": worst_abs,
        "max_rel": worst_rel,
        "passed": bool(worst_rel <= rtol),
        "detail": detail,
    }


def _statistical_check(
    jax_mean: float, jax_err: float, np_samples: np.ndarray, n_sigma: float = 5.0
) -> dict[str, Any]:
    """Two-sample agreement of a Monte Carlo mean within n_sigma."""
    np_mean = float(np.mean(np_samples))
    np_err = _blocked_error(np_samples)
    sigma = float(np.hypot(jax_err, np_err))
    z = abs(jax_mean - np_mean) / sigma if sigma > 0 else float("inf")
    return {
        "kind": "statistical",
        "jax_mean": jax_mean,
        "jax_err": jax_err,
        "numpy_mean": np_mean,
        "numpy_err": np_err,
        "z_score": z,
        "n_sigma": n_sigma,
        "passed": bool(z <= n_sigma),
    }


def _blocked_error(x: np.ndarray, n_blocks: int = 10) -> float:
    """Standard error of the mean from block averages (handles autocorrelation)."""
    x = np.asarray(x, dtype=float)
    m = len(x) // n_blocks
    if m == 0:
        return float(np.std(x) / np.sqrt(max(len(x), 1)))
    blocks = x[: m * n_blocks].reshape(n_blocks, m).mean(axis=1)
    return float(np.std(blocks, ddof=1) / np.sqrt(n_blocks))


# ---------------------------------------------------------------------------
# Classical mechanics
# ---------------------------------------------------------------------------


def _oscillator(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
    return 0.5 * p[0] ** 2 + 0.5 * params.k * q[0] ** 2


def hamiltonian_cases(steps_list: list[int]) -> list[Case]:
    out = []
    system = jp.HamiltonianSystem(_oscillator, n_dof=1)
    params = jp.Params(k=4.0)
    dt, save = 1e-3, 100
    for n in steps_list:
        t_end = (n + 0.5) * dt  # exactly n steps

        def jax_fn(t_end: float = t_end) -> Any:
            return system.simulate(
                q0=[1.0],
                p0=[0.0],
                t_span=(0.0, t_end),
                dt=dt,
                params=params,
                integrator="leapfrog",
                save_every=save,
            )

        def np_fn(n: int = n) -> Any:
            return ref.oscillator_leapfrog(1.0, 0.0, 4.0, dt, n, save)

        out.append(
            Case(
                solver="HamiltonianSystem (leapfrog, 1 dof)",
                size=f"{n} steps",
                params={"n_steps": n, "dt": dt, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [("q", a.q[:, 0], b["q"]), ("p", a.p[:, 0], b["p"])], 1e-12
                ),
                work=n,
                work_unit="steps/s",
            )
        )
    return out


def _double_pendulum(q: jax.Array, qd: jax.Array, params: Any) -> jax.Array:
    t1, t2 = q
    w1, w2 = qd
    g = 9.81
    T = 0.5 * w1**2 + 0.5 * (w1**2 + w2**2 + 2 * w1 * w2 * jnp.cos(t1 - t2))
    V = -2.0 * g * jnp.cos(t1) - g * jnp.cos(t2)
    return T - V


def lagrangian_cases(steps_list: list[int]) -> list[Case]:
    out = []
    system = jp.LagrangianSystem(_double_pendulum, n_dof=2)
    dt, save = 1e-3, 100
    for n in steps_list:
        t_end = (n + 0.5) * dt

        def jax_fn(t_end: float = t_end) -> Any:
            return system.simulate(
                q0=[0.3, 0.2],
                qdot0=[0.0, 0.0],
                t_span=(0.0, t_end),
                dt=dt,
                integrator="rk4",
                save_every=save,
            )

        def np_fn(n: int = n) -> Any:
            return ref.double_pendulum_rk4((0.3, 0.2), (0.0, 0.0), 9.81, dt, n, save)

        out.append(
            Case(
                solver="LagrangianSystem (double pendulum, rk4)",
                size=f"{n} steps",
                params={"n_steps": n, "dt": dt, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [("q", a.q, b["q"]), ("qdot", a.p, b["p"])], 1e-9
                ),
                work=n,
                work_unit="steps/s",
                notes="NumPy uses the closed-form equations of motion; JAX derives "
                "them by autodiff (mass matrix + mixed Hessian per step).",
            )
        )
    return out


T_END_NBODY, SAVE_NBODY = 0.25, 100


def nbody_cases(sizes: list[tuple[int, int]]) -> list[Case]:
    out = []
    for n, steps in sizes:
        key = jax.random.PRNGKey(n)
        pos = np.asarray(jax.random.normal(key, (n, 3)))
        vel = 0.1 * np.asarray(jax.random.normal(jax.random.fold_in(key, 1), (n, 3)))
        m = np.full(n, 1.0 / n)
        system = jp.NBody(m, pos, vel, G=1.0, softening=0.05)
        dt = T_END_NBODY / steps

        def jax_fn(system: Any = system, steps: int = steps) -> Any:
            return system.simulate(
                t_span=(0.0, T_END_NBODY), n_steps=steps, save_every=SAVE_NBODY
            )

        def np_fn(
            pos: np.ndarray = pos,
            vel: np.ndarray = vel,
            m: np.ndarray = m,
            steps: int = steps,
        ) -> Any:
            return ref.nbody_verlet(
                pos, vel, m, 1.0, 0.05, T_END_NBODY / steps, steps, SAVE_NBODY
            )

        out.append(
            Case(
                solver="NBody (velocity Verlet, O(N^2))",
                size=f"N={n}, {steps} steps",
                params={
                    "n_bodies": n,
                    "n_steps": steps,
                    "dt": dt,
                    "save_every": SAVE_NBODY,
                },
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [
                        ("positions", a.positions, b["positions"]),
                        ("velocities", a.velocities, b["velocities"]),
                    ],
                    1e-9,
                ),
                work=float(n * n * steps),
                work_unit="pair-interactions/s",
            )
        )
    return out


def rigid_body_cases(steps_list: list[int]) -> list[Case]:
    out = []
    body = jp.RigidBody(inertia=[1.0, 2.0, 3.0])
    dt = 1e-2
    for n in steps_list:
        t_end = (n + 0.5) * dt

        def jax_fn(t_end: float = t_end) -> Any:
            return body.simulate(omega0=[1.0, 0.1, 0.05], t_span=(0.0, t_end), dt=dt)

        def np_fn(n: int = n) -> Any:
            return ref.rigid_body_rk4(
                np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.1, 0.05]), dt, n
            )

        out.append(
            Case(
                solver="RigidBody (Euler eqs + quaternion, rk4)",
                size=f"{n} steps",
                params={"n_steps": n, "dt": dt},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [("quaternion", a.q, b["q"]), ("omega", a.p, b["p"])], 1e-10
                ),
                work=n,
                work_unit="steps/s",
            )
        )
    return out


def charges_cases(steps_list: list[int]) -> list[Case]:
    out = []
    bfield = np.array([0.0, 0.0, 0.5])
    system = jp.ChargeSystem(
        [
            jp.PointCharge(1e-6, 1e-3, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            jp.PointCharge(-1e-6, 1e-3, [0.1, 0.0, 0.0], [0.0, 1.0, 0.0]),
            jp.PointCharge(2e-6, 2e-3, [0.0, 0.2, 0.05], [0.5, 0.0, 0.0]),
        ],
        B_external=jnp.asarray(bfield),
    )
    t_end, save = 1e-2, 100
    for n in steps_list:

        def jax_fn(n: int = n) -> Any:
            return system.simulate(t_span=(0.0, t_end), n_steps=n, save_every=save)

        def np_fn(n: int = n) -> Any:
            return ref.charges_boris(
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.05]]),
                np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0]]),
                np.array([1e-6, -1e-6, 2e-6]),
                np.array([1e-3, 1e-3, 2e-3]),
                bfield,
                1e-10,
                t_end / n,
                n,
                save,
            )

        out.append(
            Case(
                solver="ChargeSystem (Coulomb + uniform B, Boris)",
                size=f"3 charges, {n} steps",
                params={"n_steps": n, "dt": t_end / n, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [
                        ("positions", a.positions, b["positions"]),
                        ("velocities", a.velocities, b["velocities"]),
                    ],
                    1e-10,
                ),
                work=n,
                work_unit="steps/s",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Quantum mechanics
# ---------------------------------------------------------------------------


def schrodinger_cases(points: list[int], steps: int) -> list[Case]:
    out = []
    dt, save, x_range = 1e-3, 100, (-20.0, 20.0)
    packet = jp.GaussianWavepacket(x0=-5.0, k0=2.0, sigma=0.7)
    barrier = jp.SquareBarrier(height=2.0, width=1.0, center=0.0)
    t_end = (steps + 0.5) * dt
    for n in points:

        def jax_fn(n: int = n) -> Any:
            return jp.solve_schrodinger(
                packet,
                barrier,
                x_range,
                (0.0, t_end),
                n_points=n,
                dt=dt,
                save_every=save,
            )

        def np_fn(n: int = n) -> Any:
            # Periodic grid: n points on [x_min, x_max), dx = L / n.
            x = np.linspace(x_range[0], x_range[1], n, endpoint=False)
            dx = (x_range[1] - x_range[0]) / n
            psi = (
                (2 * np.pi * packet.sigma**2) ** -0.25
                * np.exp(-((x - packet.x0) ** 2) / (4 * packet.sigma**2))
                * np.exp(1j * packet.k0 * x)
            )
            psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
            V = np.where(
                np.abs(x - barrier.center) < barrier.width / 2, barrier.height, 0.0
            )
            k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
            return ref.split_operator(psi, V, k**2, dt, steps, save)

        out.append(
            Case(
                solver="solve_schrodinger (split-operator FFT)",
                size=f"{n} points, {steps} steps",
                params={"n_points": n, "n_steps": steps, "dt": dt, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("psi", a.psi, b["psi"])], 1e-9),
                work=float(n * steps),
                work_unit="point-steps/s",
            )
        )
    return out


def schrodinger_2d_cases(sizes: list[int], steps: int) -> list[Case]:
    out = []
    L, dt, save = 20.0, 1e-3, 100
    packet = jp.GaussianWavepacket2D(x0=-2.0, y0=0.5, kx=3.0, ky=-1.0, sigma=0.8)

    def potential(X: Any, Y: Any) -> Any:
        return 0.5 * (X**2 + 2.0 * Y**2)

    t_end = (steps + 0.5) * dt
    for n in sizes:

        def jax_fn(n: int = n) -> Any:
            return jp.solve_schrodinger_2d(
                packet,
                potential,
                x_range=(-L / 2, L / 2),
                y_range=(-L / 2, L / 2),
                n_points=(n, n),
                t_span=(0.0, t_end),
                dt=dt,
                save_every=save,
            )

        def np_fn(n: int = n) -> Any:
            x = np.linspace(-L / 2, L / 2, n, endpoint=False)
            dx = L / n
            X, Y = np.meshgrid(x, x, indexing="ij")
            r2 = (X - packet.x0) ** 2 + (Y - packet.y0) ** 2
            psi = (
                (2 * np.pi * packet.sigma**2) ** -0.5
                * np.exp(-r2 / (4 * packet.sigma**2))
                * np.exp(1j * (packet.kx * X + packet.ky * Y))
            )
            psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dx)
            k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
            k2 = k[:, None] ** 2 + k[None, :] ** 2
            return ref.split_operator(psi, potential(X, Y), k2, dt, steps, save)

        out.append(
            Case(
                solver="solve_schrodinger_2d (split-operator FFT)",
                size=f"{n}x{n}, {steps} steps",
                params={"grid": [n, n], "n_steps": steps, "dt": dt, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("psi", a.psi, b["psi"])], 1e-9),
                work=float(n * n * steps),
                work_unit="point-steps/s",
            )
        )
    return out


def lindblad_cases(configs: list[tuple[int, int]]) -> list[Case]:
    out = []
    dt, save = 1e-2, 100
    for d, steps in configs:
        rng = np.random.default_rng(d)
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H = (a + a.conj().T) / 2
        lower = np.diag(np.ones(d - 1), k=1).astype(complex)
        deph = np.diag(np.arange(d)).astype(complex)
        psi0 = np.zeros(d, complex)
        psi0[-1] = 1.0
        rho0 = jp.DensityMatrix.from_pure_state(jnp.asarray(psi0))
        t_end = (steps + 0.5) * dt

        def jax_fn(
            rho0: Any = rho0,
            H: Any = H,
            lower: Any = lower,
            deph: Any = deph,
            t_end: float = t_end,
        ) -> Any:
            return jp.lindblad_evolve(
                rho0,
                jnp.asarray(H),
                [jnp.asarray(lower), jnp.asarray(deph)],
                [0.1, 0.05],
                t_span=(0.0, t_end),
                dt=dt,
                save_every=save,
            )

        def np_fn(
            psi0: Any = psi0,
            H: Any = H,
            lower: Any = lower,
            deph: Any = deph,
            steps: int = steps,
        ) -> Any:
            return ref.lindblad_rk4(
                np.outer(psi0, psi0.conj()),
                H,
                [lower, deph],
                [0.1, 0.05],
                dt,
                steps,
                save,
            )

        out.append(
            Case(
                solver="lindblad_evolve (rk4)",
                size=f"d={d}, {steps} steps",
                params={"dim": d, "n_steps": steps, "dt": dt, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("rho", a.rho, b["rho"])], 1e-10),
                work=steps,
                work_unit="steps/s",
            )
        )
    return out


def tight_binding_cases(n_k_list: list[int]) -> list[Case]:
    out = []
    model = jp.TightBinding.honeycomb(t=2.7)
    for n_k in n_k_list:
        k = np.asarray(jax.random.uniform(jax.random.PRNGKey(n_k), (n_k, 2))) * 4.0

        def jax_fn(k: np.ndarray = k) -> Any:
            return model.bands(jnp.asarray(k))

        def np_fn(k: np.ndarray = k) -> Any:
            return ref.honeycomb_bands(2.7, 1.0, k)

        out.append(
            Case(
                solver="TightBinding.bands (graphene)",
                size=f"{n_k} k-points",
                params={"n_k": n_k},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("bands", a, b)], 1e-12),
                work=n_k,
                work_unit="k-points/s",
                notes="NumPy uses the closed-form 2x2 Bloch Hamiltonian; jaxphys "
                "assembles H(k) from the hopping list.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Electromagnetism
# ---------------------------------------------------------------------------


def fdtd2d_cases(sizes: list[int], steps: int) -> list[Case]:
    out = []
    dx, save, freq = 0.01, 50, 3e9
    dt = 0.99 * dx / (C0 * float(np.sqrt(2.0)))
    for n in sizes:
        grid = jp.EMGrid(size=(n, n), resolution=dx)
        grid.add_source(jp.PlaneWave(frequency=freq, y=n // 5))
        t_end = (steps + 0.5) * dt

        def jax_fn(grid: Any = grid, t_end: float = t_end) -> Any:
            return grid.simulate(t_span=(0.0, t_end), dt=dt, save_every=save)

        def np_fn(n: int = n) -> Any:
            return ref.fdtd2d_pml(n, n, dx, dt, steps, save, n // 5, freq)

        out.append(
            Case(
                solver="EMGrid (2D FDTD TM, split-field PML)",
                size=f"{n}x{n}, {steps} steps",
                params={"grid": [n, n], "n_steps": steps, "dt": dt, "save_every": save},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("ez", a.ez, b["ez"])], 1e-10),
                work=float(n * n * steps),
                work_unit="cell-updates/s",
            )
        )
    return out


def fdtd3d_cases(sizes: list[int], steps: int) -> list[Case]:
    out = []
    dx, save, freq = 0.01, 20, 3e9
    dt = 0.99 * dx / (C0 * float(np.sqrt(3.0)))
    for n in sizes:
        src = (n // 2, n // 2, n // 2)
        grid = jp.EMGrid3D(size=(n, n, n), resolution=dx)
        grid.add_source(jp.PointSource3D(frequency=freq, position=src))
        t_end = (steps + 0.5) * dt

        def jax_fn(grid: Any = grid, t_end: float = t_end) -> Any:
            return grid.simulate(t_span=(0.0, t_end), dt=dt, save_every=save)

        def np_fn(n: int = n, src: tuple[int, int, int] = src) -> Any:
            return ref.fdtd3d_pml(n, dx, dt, steps, save, src, freq)

        out.append(
            Case(
                solver="EMGrid3D (3D FDTD, split-field PML)",
                size=f"{n}^3, {steps} steps",
                params={
                    "grid": [n, n, n],
                    "n_steps": steps,
                    "dt": dt,
                    "save_every": save,
                },
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("ez", a.ez, b["ez"])], 1e-10),
                work=float(n**3 * steps),
                work_unit="cell-updates/s",
            )
        )
    return out


def fdfd_cases(sizes: list[int]) -> list[Case]:
    out = []
    freq = 3e9
    dx = C0 / freq / 20
    for n in sizes:
        x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        eps = np.where(
            (x - n // 2) ** 2 + (y - 2 * n // 3) ** 2 < (n // 8) ** 2, 4.0, 1.0
        )
        src = np.zeros((n, n))
        src[n // 2, n // 4] = 1.0 / dx**2

        def jax_fn(eps: np.ndarray = eps, src: np.ndarray = src) -> Any:
            return jp.solve_fdfd(jnp.asarray(eps), jnp.asarray(src), freq, dx, 12)

        def np_fn(eps: np.ndarray = eps, src: np.ndarray = src) -> Any:
            return ref.fdfd_tm(eps, src, freq, dx, 12)

        out.append(
            Case(
                solver="solve_fdfd (2D TM, stretched-coordinate PML)",
                size=f"{n}x{n}",
                params={"grid": [n, n], "pml_layers": 12, "points_per_wavelength": 20},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("ez", a.ez, b["ez"])], 1e-8),
                work=float(n * n),
                work_unit="unknowns/s",
                notes="NumPy side: SciPy sparse assembly + spsolve (SuperLU); "
                "jaxphys: dense block-tridiagonal elimination with lax.scan.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Fluids
# ---------------------------------------------------------------------------


def lbm_cases(sizes: list[tuple[int, int]], steps: int) -> list[Case]:
    out = []
    nu, u_in, save = 0.05, 0.04, 100
    for nx, ny in sizes:
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        mask = (x - nx // 4) ** 2 + (y - ny // 2) ** 2 < (ny // 8) ** 2
        grid = jp.LBMGrid(size=(nx, ny), viscosity=nu)
        grid.add_obstacle(jp.Obstacle(mask=jnp.asarray(mask)))

        def jax_fn(grid: Any = grid) -> Any:
            return grid.simulate(n_steps=steps, u_inlet=u_in, save_every=save)

        def np_fn(nx: int = nx, ny: int = ny, mask: np.ndarray = mask) -> Any:
            return ref.lbm_d2q9(nx, ny, 3 * nu + 0.5, u_in, steps, save, mask)

        out.append(
            Case(
                solver="LBMGrid (D2Q9 BGK)",
                size=f"{nx}x{ny}, {steps} steps",
                params={
                    "grid": [nx, ny],
                    "n_steps": steps,
                    "save_every": save,
                    "nu": nu,
                },
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [
                        ("rho", a.rho, b["rho"]),
                        ("ux", a.ux, b["ux"]),
                        ("uy", a.uy, b["uy"]),
                    ],
                    1e-10,
                ),
                work=float(nx * ny * steps),
                work_unit="lattice-updates/s",
                notes="Throughput in lattice-site updates/s (1e6 = 1 MLUPS).",
            )
        )
    return out


def navier_stokes_cases(sizes: list[int], steps: int) -> list[Case]:
    out = []
    nu, dt, lid, iters, save = 0.01, 0.01, 1.0, 50, 20
    for n in sizes:
        solver = jp.NavierStokesSolver(size=(n, n), viscosity=nu)

        def jax_fn(solver: Any = solver) -> Any:
            return solver.simulate(
                n_steps=steps,
                dt=dt,
                lid_velocity=lid,
                poisson_iters=iters,
                save_every=save,
            )

        def np_fn(n: int = n) -> Any:
            return ref.vorticity_streamfunction(n, nu, dt, lid, iters, steps, save)

        out.append(
            Case(
                solver="NavierStokesSolver (vorticity-streamfunction)",
                size=f"{n}x{n}, {steps} steps",
                params={
                    "grid": [n, n],
                    "n_steps": steps,
                    "poisson_iters": iters,
                    "save_every": save,
                },
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check([("ux", a.ux, b["ux"])], 1e-10),
                work=float(n * n * steps),
                work_unit="cell-steps/s",
            )
        )
    return out


def sph_cases(sides: list[int], steps: int) -> list[Case]:
    out = []
    for n_side in sides:
        dx = 1.0 / n_side
        xs = (np.arange(n_side) + 0.5) * dx
        pos = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
        vel = 0.05 * np.random.default_rng(1).normal(size=pos.shape)
        fluid = jp.SPHFluid(
            mass=1000.0 * dx**2, smoothing_length=1.3 * dx, box=(1.0, 1.0)
        )
        dt = 0.2 * fluid.h / fluid.sound_speed
        t_end = (steps + 0.5) * dt

        def jax_fn(
            fluid: Any = fluid,
            pos: Any = pos,
            vel: Any = vel,
            t_end: float = t_end,
            dt: float = dt,
        ) -> Any:
            return fluid.simulate(pos, vel, (0.0, t_end), dt, save_every=steps)

        def np_fn(
            fluid: Any = fluid, pos: Any = pos, vel: Any = vel, dt: float = dt
        ) -> Any:
            return ref.sph_periodic(
                pos,
                vel,
                fluid.mass,
                fluid.h,
                fluid.box,
                fluid.rest_density,
                fluid.sound_speed,
                fluid.gamma,
                fluid.alpha,
                dt,
                steps,
            )

        out.append(
            Case(
                solver="SPHFluid (weakly compressible, cell list)",
                size=f"{n_side**2} particles, {steps} steps",
                params={"n_particles": n_side**2, "n_steps": steps, "dt": dt},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [
                        ("positions", a.positions[-1], b["positions"]),
                        ("velocities", a.velocities[-1], b["velocities"]),
                    ],
                    1e-9,
                ),
                work=float(n_side**2 * steps),
                work_unit="particle-steps/s",
                notes="NumPy side finds neighbours with SciPy's compiled cKDTree.",
            )
        )
    return out


def euler_cases(cells: list[int]) -> list[Case]:
    out = []
    t_end, save = 0.2, 50
    for n in cells:
        x = (np.arange(n) + 0.5) / n
        rho = np.where(x < 0.5, 1.0, 0.125)
        p = np.where(x < 0.5, 1.0, 0.1)
        u = np.zeros(n)
        # Same dt rule as solve_euler_1d's default (CFL 0.4 on the initial
        # state, rounded to a whole number of save intervals), fixed here.
        dt0 = 0.4 / n / float(np.max(np.sqrt(1.4 * p / rho)))
        n_steps = save * int(np.ceil(t_end / dt0 / save - 1e-9))
        dt = t_end / n_steps

        def jax_fn(
            rho: Any = rho, u: Any = u, p: Any = p, n: int = n, n_steps: int = n_steps
        ) -> Any:
            return jp.solve_euler_1d(
                jnp.asarray(rho),
                jnp.asarray(u),
                jnp.asarray(p),
                dx=1.0 / n,
                t_end=t_end,
                dt=t_end / n_steps,
                save_every=save,
            )

        def np_fn(
            rho: Any = rho, u: Any = u, p: Any = p, n: int = n, n_steps: int = n_steps
        ) -> Any:
            return ref.euler_muscl_hllc(rho, u, p, (t_end / n_steps) * n, 1.4, n_steps)

        out.append(
            Case(
                solver="solve_euler_1d (MUSCL-HLLC, SSP-RK2)",
                size=f"{n} cells, {n_steps} steps (Sod)",
                params={"n_cells": n, "n_steps": n_steps, "dt": dt, "t_end": t_end},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=lambda a, b: _allclose_check(
                    [
                        ("rho", a.rho[-1], b["rho"]),
                        ("u", a.u[-1], b["u"]),
                        ("p", a.p[-1], b["p"]),
                    ],
                    1e-10,
                ),
                work=float(n * n_steps),
                work_unit="cell-steps/s",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Statistical mechanics
# ---------------------------------------------------------------------------


def ising_cases(sizes: list[int], sweeps: int, warmup: int) -> list[Case]:
    out = []
    T = 3.5  # disordered phase: short autocorrelation, clean statistics
    for L in sizes:
        lattice = jp.IsingLattice(size=(L, L))

        def jax_fn(lattice: Any = lattice) -> Any:
            return lattice.run_metropolis(
                T, n_sweeps=sweeps, n_warmup=warmup, key=jax.random.PRNGKey(7)
            )

        def np_fn(L: int = L) -> Any:
            return ref.ising_metropolis(L, T, sweeps, warmup, seed=7)

        def compare(a: Any, b: Any, L: int = L) -> dict[str, Any]:
            var_e = a["specific_heat"] * T**2 / (L * L)
            err = float(np.sqrt(var_e * 10.0 / sweeps))  # tau_int <= 5 sweeps
            return _statistical_check(a["energy"], err, b["energy"])

        out.append(
            Case(
                solver="IsingLattice.run_metropolis (checkerboard)",
                size=f"{L}x{L}, {sweeps} sweeps",
                params={"L": L, "T": T, "n_sweeps": sweeps, "n_warmup": warmup},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=compare,
                work=float(L * L * (sweeps + warmup)),
                work_unit="spin-flip attempts/s",
                notes="Different RNG streams and update orders (checkerboard vs "
                "random site): mean energy per spin compared statistically "
                "(5 sigma, blocked errors).",
            )
        )
    return out


def wolff_cases(sizes: list[int], updates: int, warmup: int) -> list[Case]:
    out = []
    T = 3.5
    for L in sizes:
        lattice = jp.IsingLattice(size=(L, L))

        def jax_fn(lattice: Any = lattice) -> Any:
            return jp.sweep_temperatures(
                lattice,
                jnp.array([T]),
                n_sweeps=updates,
                n_warmup=warmup,
                algorithm="wolff_cluster",
                key=jax.random.PRNGKey(11),
            )

        def np_fn(L: int = L) -> Any:
            return ref.ising_wolff(L, T, updates, warmup, seed=11)

        def compare(a: Any, b: Any, L: int = L) -> dict[str, Any]:
            var_e = float(a.specific_heats[0]) * T**2 / (L * L)
            err = float(np.sqrt(max(var_e, 1e-300) * 10.0 / updates))
            return _statistical_check(float(a.energies[0]), err, b["energy"])

        out.append(
            Case(
                solver="sweep_temperatures (Wolff cluster)",
                size=f"{L}x{L}, {updates} cluster updates",
                params={"L": L, "T": T, "n_updates": updates, "n_warmup": warmup},
                jax_fn=jax_fn,
                numpy_fn=np_fn,
                compare=compare,
                work=float(updates + warmup),
                work_unit="cluster updates/s",
                notes="Compared statistically against a textbook (stack-based) "
                "Wolff implementation (5 sigma).",
            )
        )
    return out


def all_cases(quick: bool) -> list[Case]:
    if quick:
        return [
            *hamiltonian_cases([20000]),
            *lagrangian_cases([2000]),
            *nbody_cases([(16, 2000)]),
            *rigid_body_cases([5000]),
            *charges_cases([5000]),
            *schrodinger_cases([512], 2000),
            *schrodinger_2d_cases([64], 200),
            *lindblad_cases([(2, 5000)]),
            *tight_binding_cases([1000]),
            *fdtd2d_cases([100], 200),
            *fdtd3d_cases([24], 60),
            *fdfd_cases([60]),
            *lbm_cases([(100, 40)], 300),
            *navier_stokes_cases([32], 100),
            *sph_cases([16], 50),
            *euler_cases([200]),
            *ising_cases([8], 200, 50),
            *wolff_cases([8], 200, 20),
        ]
    return [
        *hamiltonian_cases([20000, 200000]),
        *lagrangian_cases([5000, 20000]),
        *nbody_cases([(16, 5000), (64, 5000), (256, 1000), (1000, 200)]),
        *rigid_body_cases([5000, 50000]),
        *charges_cases([5000, 50000]),
        *schrodinger_cases([512, 2048, 8192], 5000),
        *schrodinger_2d_cases([128, 256], 500),
        *lindblad_cases([(2, 20000), (8, 5000)]),
        *tight_binding_cases([1000, 10000]),
        *fdtd2d_cases([100, 200, 400], 500),
        *fdtd3d_cases([24, 40, 64], 100),
        *fdfd_cases([100, 200]),
        *lbm_cases([(100, 40), (200, 80), (400, 160)], 500),
        *navier_stokes_cases([32, 64, 128], 200),
        *sph_cases([32, 64], 200),
        *euler_cases([400, 1600]),
        *ising_cases([8, 16, 32], 200, 50),
        *wolff_cases([8, 16, 32], 400, 50),
    ]


def run_case(case: Case, repeats: int, numpy_repeats: int) -> dict[str, Any]:
    jax_meas = measure_jax(case.jax_fn, repeats=repeats)
    np_meas = measure_numpy(case.numpy_fn, repeats=numpy_repeats)
    check = case.compare(jax_meas.pop("result"), np_meas.pop("result"))
    jax_med = jax_meas["steady"]["median_s"]
    np_med = np_meas["steady"]["median_s"]
    return {
        "solver": case.solver,
        "size": case.size,
        "params": case.params,
        "jax": jax_meas,
        "numpy": np_meas,
        "throughput_unit": case.work_unit,
        "jax_throughput": case.work / jax_med,
        "numpy_throughput": case.work / np_med,
        "speedup_vs_numpy": np_med / jax_med,
        "agreement": check,
        "notes": case.notes,
        **case.extra,
    }
