"""Accuracy and physics-validation experiments.

Each experiment returns a JSON-serializable dict with the measured values,
the analytic (or independent) reference, the pass criterion and whether it
passed.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxphys as jp
from jaxphys.em.fdtd import C0, MU0
from jaxphys.statmech.ising import _metropolis_chain, _wolff_chain

ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def _fit_order(dts: list[float], errs: list[float], floor: float = 1e-13) -> float:
    """Least-squares slope of log(err) vs log(dt), ignoring round-off floor."""
    pts = [(np.log(d), np.log(e)) for d, e in zip(dts, errs, strict=True) if e > floor]
    if len(pts) < 2:
        return float("nan")
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# 1. Symplectic vs RK4 long-time energy behaviour (Kepler problem)
# ---------------------------------------------------------------------------


def _kepler(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
    return 0.5 * jnp.sum(p**2) - 1.0 / jnp.sqrt(jnp.sum(q**2))


def _kepler_ic(e: float) -> tuple[list[float], list[float]]:
    return [1.0 - e, 0.0], [0.0, float(np.sqrt((1.0 + e) / (1.0 - e)))]


def energy_drift(n_orbits: int, steps_per_orbit: int = 200) -> dict[str, Any]:
    e = 0.5
    q0, p0 = _kepler_ic(e)
    period = 2.0 * np.pi
    dt = period / steps_per_orbit
    n_steps = n_orbits * steps_per_orbit
    system = jp.HamiltonianSystem(_kepler, n_dof=2)
    rows = []
    for integ in ("symplectic_euler", "leapfrog", "yoshida4", "rk4"):
        tr = system.simulate(
            q0,
            p0,
            (0.0, (n_steps + 0.5) * dt),
            dt=dt,
            integrator=integ,
            save_every=steps_per_orbit // 20,
        )
        energy = np.asarray(tr.energy)
        rel = (energy - energy[0]) / abs(energy[0])
        orbits = np.asarray(tr.t) / period
        half = len(rel) // 2
        slope = float(np.polyfit(orbits, rel, 1)[0])
        rows.append(
            {
                "integrator": integ,
                "max_abs_rel_energy_error": float(np.max(np.abs(rel))),
                "final_rel_energy_error": float(rel[-1]),
                "max_error_first_half": float(np.max(np.abs(rel[:half]))),
                "max_error_second_half": float(np.max(np.abs(rel[half:]))),
                "linear_drift_per_orbit": slope,
            }
        )
    by = {r["integrator"]: r for r in rows}
    # Symplectic: error bounded (second half no worse than 1.5x first half).
    # RK4: secular drift (monotone growth, second half clearly larger).
    symplectic_bounded = all(
        by[i]["max_error_second_half"] <= 1.5 * by[i]["max_error_first_half"]
        for i in ("leapfrog", "yoshida4")
    )
    rk4_secular = (
        by["rk4"]["max_error_second_half"] > 1.5 * by["rk4"]["max_error_first_half"]
    )
    return {
        "name": "Energy error over long runs: symplectic vs RK4 (Kepler, e=0.5)",
        "setup": {
            "eccentricity": e,
            "orbits": n_orbits,
            "steps_per_orbit": steps_per_orbit,
            "dt": dt,
            "energy_sampled_every_steps": steps_per_orbit // 20,
        },
        "rows": rows,
        "criterion": "leapfrog/yoshida4 max error in 2nd half <= 1.5x 1st half "
        "(bounded); rk4 2nd half > 1.5x 1st half (secular drift)",
        "passed": bool(symplectic_bounded and rk4_secular),
    }


# ---------------------------------------------------------------------------
# 2. Measured convergence order vs dt
# ---------------------------------------------------------------------------

ROUNDOFF_FLOOR = 1e-10

EXPECTED_ORDER = {
    "euler": 1,
    "symplectic_euler": 1,
    "leapfrog": 2,
    "yoshida4": 4,
    "rk4": 4,
}


def _oscillator(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
    return 0.5 * p[0] ** 2 + 0.5 * q[0] ** 2


def _kepler_exact(t: float, e: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Kepler orbit (a = GM = 1, perihelion at t = 0) via Kepler's eq."""
    mean_anomaly = t  # n = sqrt(GM / a^3) = 1
    E = mean_anomaly
    for _ in range(100):
        E = E - (E - e * np.sin(E) - mean_anomaly) / (1.0 - e * np.cos(E))
    b = np.sqrt(1.0 - e**2)
    e_dot = 1.0 / (1.0 - e * np.cos(E))
    q = np.array([np.cos(E) - e, b * np.sin(E)])
    p = np.array([-np.sin(E) * e_dot, b * np.cos(E) * e_dot])
    return q, p


def convergence_order(n_refinements: int) -> dict[str, Any]:
    results = []
    # Harmonic oscillator: q(t) = cos t, p(t) = -sin t, compared at t = 10.
    # Kepler: compared with the analytic orbit at t = 0.37 T (a generic phase;
    # at exactly one period some first-order errors cancel by symmetry).
    ho = jp.HamiltonianSystem(_oscillator, n_dof=1)
    kep = jp.HamiltonianSystem(_kepler, n_dof=2)
    q0k, p0k = _kepler_ic(0.5)
    problems = [
        ("harmonic oscillator (t=10)", ho, [1.0], [0.0], 10.0, 100),
        ("Kepler e=0.5 (t=0.37 T)", kep, q0k, p0k, 0.37 * 2.0 * np.pi, 400),
    ]
    for label, system, q0, p0, t_final, n0 in problems:
        for integ, expected in EXPECTED_ORDER.items():
            dts, errs = [], []
            for r in range(n_refinements):
                n = n0 * 2**r
                dt = t_final / n
                tr = system.simulate(
                    q0,
                    p0,
                    (0.0, (n + 0.5) * dt),
                    dt=dt,
                    integrator=integ,
                    save_every=n,
                )
                q, p = np.asarray(tr.q[-1]), np.asarray(tr.p[-1])
                if system is ho:
                    q_ex, p_ex = (
                        np.array([np.cos(t_final)]),
                        np.array([-np.sin(t_final)]),
                    )
                else:
                    q_ex, p_ex = _kepler_exact(float(tr.t[-1]), 0.5)
                err = float(np.max(np.abs(np.concatenate([q - q_ex, p - p_ex]))))
                dts.append(dt)
                errs.append(err)
            # Fit on the three finest step sizes whose error is well above the
            # accumulated round-off level (asymptotic, round-off-free regime).
            usable = [
                (d, e) for d, e in zip(dts, errs, strict=True) if e > ROUNDOFF_FLOOR
            ]
            order = _fit_order([d for d, _ in usable[-3:]], [e for _, e in usable[-3:]])
            results.append(
                {
                    "problem": label,
                    "integrator": integ,
                    "expected_order": expected,
                    "measured_order": order,
                    "dts": dts,
                    "errors": errs,
                    "passed": bool(abs(order - expected) <= 0.3),
                }
            )
    return {
        "name": "Convergence order vs dt",
        "rows": results,
        "criterion": "|measured - expected| <= 0.3 (log-log least-squares fit "
        "over the three finest dt with error > 1e-10, i.e. above round-off)",
        "passed": all(r["passed"] for r in results),
    }


# ---------------------------------------------------------------------------
# 3. Schrodinger: norm conservation and free Gaussian wavepacket
# ---------------------------------------------------------------------------


def _free_gaussian(
    x: np.ndarray, t: float, x0: float, k0: float, s0: float, m: float = 1.0
) -> np.ndarray:
    a = 1.0 + 1j * t / (2.0 * m * s0**2)
    return (
        (2.0 * np.pi * s0**2) ** -0.25
        / np.sqrt(a)
        * np.exp(
            -((x - x0 - k0 * t / m) ** 2) / (4.0 * s0**2 * a)
            + 1j * k0 * (x - k0 * t / (2.0 * m))
        )
    )


def schrodinger_validation(points: list[int], norm_steps: int) -> dict[str, Any]:
    # Norm conservation over a long run with a barrier (reflection + transmission).
    res = jp.solve_schrodinger(
        jp.GaussianWavepacket(x0=-5.0, k0=2.0, sigma=0.7),
        jp.SquareBarrier(height=2.0, width=1.0, center=0.0),
        x_range=(-40.0, 40.0),
        t_span=(0.0, (norm_steps + 0.5) * 1e-3),
        n_points=2048,
        dt=1e-3,
        save_every=max(norm_steps // 100, 1),
    )
    # The split-operator step is unitary for the discrete L2 norm
    # sum |psi|^2 dx on the periodic grid; that is the conserved quantity.
    # (A trapezoid integral differs from it by the end-point half weights,
    # which matters once the packet wraps around the periodic boundary.)
    x = np.asarray(res.x)
    norms = np.sum(np.abs(np.asarray(res.psi)) ** 2, axis=1) * (x[1] - x[0])
    norm_dev = float(np.max(np.abs(norms / norms[0] - 1.0)))

    rows = []
    x0, k0, s0, t_final = -10.0, 3.0, 1.0, 5.0
    for n in points:
        r = jp.solve_schrodinger(
            jp.GaussianWavepacket(x0=x0, k0=k0, sigma=s0),
            jp.HarmonicPotential(k=0.0),
            x_range=(-40.0, 40.0),
            t_span=(0.0, t_final),
            n_points=n,
            dt=1e-3,
            save_every=5000,
        )
        xg = np.asarray(r.x)
        psi = np.asarray(r.psi[-1])
        t = float(r.t[-1])
        exact = _free_gaussian(xg, t, x0, k0, s0)
        dens, dens_ex = np.abs(psi) ** 2, np.abs(exact) ** 2
        rows.append(
            {
                "n_points": n,
                "t": t,
                "psi_rel_l2_error_vs_exact": float(
                    np.linalg.norm(psi - exact) / np.linalg.norm(exact)
                ),
                "density_rel_l2_error_vs_exact": float(
                    np.linalg.norm(dens - dens_ex) / np.linalg.norm(dens_ex)
                ),
                "center_measured": float(np.trapezoid(xg * dens, xg)),
                "center_exact": float(x0 + k0 * t),
                "width_measured": float(
                    np.sqrt(
                        np.trapezoid(xg**2 * dens, xg)
                        - np.trapezoid(xg * dens, xg) ** 2
                    )
                ),
                "width_exact": float(s0 * np.sqrt(1.0 + (t / (2.0 * s0**2)) ** 2)),
            }
        )
    return {
        "name": "Schrodinger: norm conservation + free Gaussian wavepacket",
        "norm": {
            "n_points": 2048,
            "n_steps": norm_steps,
            "max_abs_norm_deviation": norm_dev,
            "passed": bool(norm_dev < 1e-10),
        },
        "free_gaussian": rows,
        "criterion": "norm deviation < 1e-10; free packet: psi relative L2 "
        "error vs the exact solution < 1e-8 at t=5 on every grid (V = 0, so the "
        "split-operator step is exact in time and the FFT grid is spectrally "
        "accurate)",
        "passed": bool(
            norm_dev < 1e-10
            and all(r["psi_rel_l2_error_vs_exact"] < 1e-8 for r in rows)
        ),
    }


# ---------------------------------------------------------------------------
# 4. FDTD: PEC cavity mode frequencies vs analytic
# ---------------------------------------------------------------------------


def fdtd_cavity(n_steps: int) -> dict[str, Any]:
    """1D PEC cavity inside the 2D TM solver.

    Two solid conducting walls (Ez = 0) at rows a and b, periodic in x, a
    uniform line source between them: only kx = 0 modes are excited, with
    ky = m pi / L, L = (b - a) dx.  Mode frequencies are read from the FFT of
    a probe (Hann window, quadratic peak interpolation) and compared with the
    continuum value m c / (2 L) and with the Yee-scheme dispersion relation
    sin(w dt / 2) = (c dt / dx) sin(ky dx / 2).
    """
    dx, a, b, ny, nx = 0.01, 5, 65, 70, 10
    grid = jp.EMGrid(size=(nx, ny), resolution=dx, boundary="periodic")
    grid.add_conductor(jp.Wall(y=a))
    grid.add_conductor(jp.Wall(y=b))
    grid.add_source(jp.PlaneWave(frequency=0.37e9, y=a + 7))
    dt = 0.99 * dx / (C0 * float(np.sqrt(2.0)))
    save = 4
    hist = grid.simulate(t_span=(0.0, (n_steps + 0.5) * dt), save_every=save)
    probe = np.asarray(hist.ez[:, nx // 2, a + 23])
    sig = (probe - probe.mean()) * np.hanning(len(probe))
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), dt * save)
    df = freqs[1] - freqs[0]
    length = (b - a) * dx
    rows = []
    for m in range(1, 5):
        f_cont = m * C0 / (2.0 * length)
        ky = m * np.pi / length
        f_yee = np.arcsin(C0 * dt / dx * np.sin(ky * dx / 2.0)) / (np.pi * dt)
        window = (freqs > 0.8 * f_cont) & (freqs < 1.2 * f_cont)
        i = int(np.argmax(np.where(window, spec, 0.0)))
        y0, y1, y2 = np.log(spec[i - 1 : i + 2])
        f_meas = freqs[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * df
        rows.append(
            {
                "mode": m,
                "measured_hz": float(f_meas),
                "analytic_continuum_hz": float(f_cont),
                "analytic_yee_dispersion_hz": float(f_yee),
                "rel_error_vs_continuum": float((f_meas - f_cont) / f_cont),
                "rel_error_vs_yee": float((f_meas - f_yee) / f_yee),
                "fft_bin_rel": float(df / f_cont),
            }
        )
    worst = max(abs(r["rel_error_vs_yee"]) for r in rows)
    return {
        "name": "FDTD PEC cavity mode frequencies",
        "setup": {
            "cavity_length_m": length,
            "dx": dx,
            "dt": dt,
            "n_steps": n_steps,
            "probe_sampled_every_steps": save,
        },
        "rows": rows,
        "criterion": "|f_measured / f_yee - 1| < 5e-4 for modes 1-4",
        "passed": bool(worst < 5e-4),
    }


# ---------------------------------------------------------------------------
# 5. Ising critical temperature (Binder cumulant crossing) vs Onsager
# ---------------------------------------------------------------------------


def _metropolis_samples(
    L: int, temps: np.ndarray, n_sweeps: int, n_warmup: int, seed: int
) -> np.ndarray:
    """|m| per sweep for a batch of temperatures (vmapped library chain)."""

    def chain(key: jax.Array, T: jax.Array) -> jax.Array:
        k0, k1 = jax.random.split(key)
        spins = 2 * jax.random.bernoulli(k0, shape=(L, L)).astype(jnp.int32) - 1
        _, mags = _metropolis_chain(
            spins, k1, 1.0 / T, 1.0, 0.0, n_warmup=n_warmup, n_sweeps=n_sweeps
        )
        return mags

    keys = jax.random.split(jax.random.PRNGKey(seed), len(temps))
    return np.asarray(jax.jit(jax.vmap(chain))(keys, jnp.asarray(temps)))


def _blocked_mean(x: np.ndarray, n_blocks: int = 20) -> tuple[float, float]:
    """Mean and its standard error from block averages."""
    m = len(x) // n_blocks
    blocks = np.asarray(x[: m * n_blocks]).reshape(n_blocks, m).mean(axis=1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(n_blocks))


def wolff_vs_metropolis(L: int, temps: list[float], n_samples: int) -> dict[str, Any]:
    """Energy per spin from Wolff and Metropolis chains at the same T."""
    rows = []
    for i, T in enumerate(temps):
        k0, k1, k2 = jax.random.split(jax.random.PRNGKey(100 + i), 3)
        spins = 2 * jax.random.bernoulli(k0, shape=(L, L)).astype(jnp.int32) - 1
        e_w, _ = _wolff_chain(
            spins, k1, T, 1.0, n_warmup=n_samples // 10, n_sweeps=n_samples
        )
        e_m, _ = _metropolis_chain(
            spins, k2, 1.0 / T, 1.0, 0.0, n_warmup=n_samples // 10, n_sweeps=n_samples
        )
        (mw, sw), (mm, sm) = (
            _blocked_mean(np.asarray(e_w)),
            _blocked_mean(np.asarray(e_m)),
        )
        z = abs(mw - mm) / float(np.hypot(sw, sm))
        rows.append(
            {
                "temperature": T,
                "wolff_energy": mw,
                "wolff_err": sw,
                "metropolis_energy": mm,
                "metropolis_err": sm,
                "z_score": z,
                "passed": bool(z <= 5.0),
            }
        )
    return {
        "lattice": L,
        "samples_per_chain": n_samples,
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }


def _binder(m: np.ndarray) -> np.ndarray:
    return 1.0 - np.mean(m**4, axis=-1) / (3.0 * np.mean(m**2, axis=-1) ** 2)


def _crossing(temps: np.ndarray, ua: np.ndarray, ub: np.ndarray) -> float:
    """Crossing of two Binder curves from cubic least-squares fits in T.

    Fitting smooths the Monte Carlo noise of the individual points; the
    crossing is the root of the fitted difference inside the T window (the
    one closest to the window centre if the cubic has several).
    """
    diff = np.polyfit(temps, ua, 3) - np.polyfit(temps, ub, 3)
    roots = np.roots(diff)
    real = roots[np.abs(roots.imag) < 1e-12].real
    inside = real[(real >= temps[0]) & (real <= temps[-1])]
    if len(inside) == 0:
        return float("nan")
    centre = 0.5 * (temps[0] + temps[-1])
    return float(inside[np.argmin(np.abs(inside - centre))])


def ising_tc(
    sizes: list[int], n_sweeps: int, n_warmup: int, n_wolff_samples: int
) -> dict[str, Any]:
    temps = np.linspace(2.15, 2.40, 11)
    samples = {
        L: _metropolis_samples(L, temps, n_sweeps, n_warmup, seed=L) for L in sizes
    }
    binder = {L: _binder(samples[L]) for L in sizes}
    n_blocks = 10
    crossings = []
    for la, lb in itertools.pairwise(sizes):
        est = _crossing(temps, binder[la], binder[lb])
        # Jackknife over blocks of the Monte Carlo time series.
        jk = []
        for j in range(n_blocks):

            def drop(m: np.ndarray, j: int = j) -> np.ndarray:
                blocks = np.array_split(m, n_blocks, axis=-1)
                return np.concatenate(
                    [b for i, b in enumerate(blocks) if i != j], axis=-1
                )

            jk.append(
                _crossing(temps, _binder(drop(samples[la])), _binder(drop(samples[lb])))
            )
        jk_arr = np.array(jk)
        err = float(
            np.sqrt(
                (n_blocks - 1)
                / n_blocks
                * np.nansum((jk_arr - np.nanmean(jk_arr)) ** 2)
            )
        )
        crossings.append(
            {
                "sizes": [la, lb],
                "tc_estimate": est,
                "jackknife_error": err,
                "deviation_from_onsager": est - ONSAGER_TC,
                "rel_deviation": (est - ONSAGER_TC) / ONSAGER_TC,
            }
        )
    best = crossings[-1]
    tolerance = 3.0 * best["jackknife_error"] + 0.01 * ONSAGER_TC
    passed = bool(abs(best["deviation_from_onsager"]) <= tolerance)

    wolff = wolff_vs_metropolis(16, [2.0, 3.0], n_wolff_samples)
    return {
        "name": "Ising critical temperature (Binder cumulant crossing, Metropolis)",
        "setup": {
            "temperatures": temps.tolist(),
            "sizes": sizes,
            "n_sweeps": n_sweeps,
            "n_warmup": n_warmup,
            "onsager_tc": ONSAGER_TC,
        },
        "binder": {str(L): binder[L].tolist() for L in sizes},
        "crossings": crossings,
        "criterion": "largest-size-pair crossing (cubic fits of U_L(T)) within 3 "
        "jackknife errors + 1% (finite-size allowance) of Onsager T_c; Wolff and "
        "Metropolis mean energies agree within 5 sigma (blocked errors) at "
        "T = 2.0 and 3.0",
        "passed": bool(passed and wolff["passed"]),
        "tc_passed": passed,
        "wolff_vs_metropolis": wolff,
    }


# ---------------------------------------------------------------------------
# 6. LBM channel (plane Poiseuille geometry) vs analytic
# ---------------------------------------------------------------------------


def lbm_channel(widths: list[int], driven_steps: int) -> dict[str, Any]:
    """Two LBM channel checks.

    (a) Decay of the fundamental shear mode between two no-slip walls. The
        flow u_y(x) = U sin(pi (x - x_w) / H) is an exact solution of the
        incompressible Navier-Stokes equations that decays as
        exp(-nu (pi/H)^2 t). The walls (obstacle columns) isolate it from the
        solver's inlet/outlet columns, so only bulk + bounce-back physics is
        tested.
    (b) Inflow-driven plane Poiseuille flow: Zou-He velocity inlet, pressure
        outlet, no-slip walls. At steady state the mass must be stationary,
        every cross-section must carry the inflow, the profile must be the
        parabola and the pressure gradient dp/dx = -12 mu u_mean / H^2.
    """
    nu = 0.1
    decay_rows = []
    for H in widths:
        nx, ny = H + 4, 8
        mask = np.zeros((nx, ny), bool)
        mask[1, :] = True
        mask[H + 2, :] = True
        x = np.arange(nx)
        fluid = (x >= 2) & (x <= H + 1)
        shape = np.where(fluid, np.sin(np.pi * (x - 1.5) / H), 0.0)
        grid = jp.LBMGrid(size=(nx, ny), viscosity=nu)
        grid.add_obstacle(jp.Obstacle(mask=jnp.asarray(mask)))
        n = int(H**2 / (nu * np.pi**2))
        hist = grid.simulate(
            n_steps=n,
            u_inlet=0.0,
            save_every=max(n // 10, 1),
            initial_ux=jnp.zeros((nx, ny)),
            initial_uy=jnp.asarray(np.repeat((0.01 * shape)[:, None], ny, axis=1)),
        )
        uy = np.asarray(hist.uy)[:, :, ny // 2]
        t = np.asarray(hist.t)  # snapshot k is after k * save_every steps
        f = slice(2, H + 2)
        amp = uy[:, f] @ shape[f] / (shape[f] @ shape[f])
        rate = -float(np.polyfit(t[1:], np.log(amp[1:]), 1)[0])
        analytic = nu * (np.pi / H) ** 2
        decay_rows.append(
            {
                "channel_width_nodes": H,
                "measured_decay_rate": rate,
                "analytic_decay_rate_halfway_walls": analytic,
                "rel_error": (rate - analytic) / analytic,
                "effective_width_from_rate": float(np.pi * np.sqrt(nu / rate)),
                "max_mode_shape_residual": float(
                    max(
                        np.linalg.norm(uy[k, f] - amp[k] * shape[f])
                        / np.linalg.norm(amp[k] * shape[f])
                        for k in range(len(t))
                    )
                ),
            }
        )
    errs = [abs(r["rel_error"]) for r in decay_rows]
    order = _fit_order([1.0 / r["channel_width_nodes"] for r in decay_rows], errs, 0.0)

    # (b) inflow-driven channel
    nx, ny, u_in = 160, 34, 0.04
    grid = jp.LBMGrid(size=(nx, ny), viscosity=nu, boundary="no_slip")
    save = driven_steps // 4
    hist = grid.simulate(n_steps=driven_steps, u_inlet=u_in, save_every=save)
    ux = np.asarray(hist.ux)
    rho = np.asarray(hist.rho)
    Hc = ny - 2  # walls half-way between rows 0/1 and -2/-1
    y = np.arange(1, ny - 1)
    prof = ux[-1, int(0.75 * nx), 1:-1]
    parab = 6.0 * prof.mean() * (y - 0.5) * (Hc + 0.5 - y) / Hc**2
    mass = rho[:, :, 1:-1].sum(axis=(1, 2))
    inflow = float(np.sum(rho[-1, 0, 1:-1] * u_in))
    flux_err = max(
        abs(float(np.sum(rho[-1, i, 1:-1] * ux[-1, i, 1:-1])) / inflow - 1.0)
        for i in range(1, nx)
    )
    xs = np.arange(nx // 2, nx - 10)
    dp_dx = float(np.polyfit(xs, rho[-1, xs, 1:-1].mean(axis=1), 1)[0]) / 3.0
    mu = nu * float(rho[-1, nx // 2, 1:-1].mean())
    dp_dx_exact = -12.0 * mu * float(ux[-1, nx // 2, 1:-1].mean()) / Hc**2
    driven = {
        "grid": [nx, ny],
        "u_inlet": u_in,
        "n_steps": driven_steps,
        "profile_max_error_vs_parabola": float(
            np.max(np.abs(prof - parab)) / prof.max()
        ),
        "profile_rel_l2_error_vs_parabola": float(
            np.linalg.norm(prof - parab) / np.linalg.norm(parab)
        ),
        "mean_density_by_snapshot": [float(r[:, 1:-1].mean()) for r in rho],
        "snapshot_steps": [int(v) for v in np.asarray(hist.t)],
        "mass_change_last_interval": float(abs(mass[-1] - mass[-2]) / mass[-1]),
        "max_flux_error_vs_inflow": flux_err,
        "dp_dx_measured": dp_dx,
        "dp_dx_poiseuille": dp_dx_exact,
        "dp_dx_rel_error": dp_dx / dp_dx_exact - 1.0,
    }
    driven["passed"] = bool(
        driven["mass_change_last_interval"] < 1e-4
        and flux_err < 1e-4
        and driven["profile_max_error_vs_parabola"] < 5e-3
        and abs(driven["dp_dx_rel_error"]) < 0.05
    )
    decay_passed = bool(max(errs) < 0.05)
    return {
        "name": "LBM plane channel vs analytic",
        "decay": decay_rows,
        "decay_error_convergence_order_in_1_over_H": order,
        "driven": driven,
        "criterion": "(a) shear-mode decay-rate error < 5% at every width; "
        "(b) steady driven channel: relative mass change < 1e-4 over the last "
        "snapshot interval, every cross-section carries the inflow to 1e-4, "
        "profile within 0.5% of the parabola, dp/dx within 5% of "
        "-12 mu u_mean / H^2",
        "passed": bool(decay_passed and driven["passed"]),
    }


# ---------------------------------------------------------------------------
# 7. jax.grad through rollouts vs central finite differences
# ---------------------------------------------------------------------------


def gradients() -> dict[str, Any]:
    rows = []

    def h(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
        return 0.5 * p[0] ** 2 + 0.5 * params.k * q[0] ** 2 + 0.25 * q[0] ** 4

    ham = jp.HamiltonianSystem(h, n_dof=1)

    def q_final_k(k: float) -> jax.Array:
        return ham.simulate(
            [1.0], [0.0], (0.0, 10.0), dt=1e-3, params=jp.Params(k=k), save_every=100
        ).q[-1, 0]

    def q_final_q0(q0: float) -> jax.Array:
        return ham.simulate(
            jnp.array([q0]),
            [0.0],
            (0.0, 10.0),
            dt=1e-3,
            params=jp.Params(k=2.0),
            save_every=100,
        ).q[-1, 0]

    n = 8
    pos = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (n, 3)))
    masses = np.full(n, 1.0 / n)
    v0 = 0.1 * np.asarray(jax.random.normal(jax.random.PRNGKey(1), (n, 3)))

    def nbody_r2(vx0: float) -> jax.Array:
        v = jnp.asarray(v0).at[0, 0].set(vx0)
        tr = jp.NBody(masses, pos, v, softening=0.1).simulate(
            (0.0, 1.0), n_steps=2000, save_every=100
        )
        return jnp.sum(tr.positions[-1] ** 2)

    def lag_final(g: float) -> jax.Array:
        def lag(q: jax.Array, qd: jax.Array, params: Any) -> jax.Array:
            return 0.5 * qd[0] ** 2 + params.g * jnp.cos(q[0])

        return (
            jp.LagrangianSystem(lag, n_dof=1)
            .simulate(
                [0.5], [0.0], (0.0, 5.0), dt=1e-3, params=jp.Params(g=g), save_every=100
            )
            .q[-1, 0]
        )

    cases = [
        ("HamiltonianSystem: d q(10) / d k (anharmonic, 10000 steps)", q_final_k, 2.0),
        ("HamiltonianSystem: d q(10) / d q0", q_final_q0, 1.0),
        ("NBody: d sum|x(1)|^2 / d vx0 (N=8, 2000 steps)", nbody_r2, float(v0[0, 0])),
        (
            "LagrangianSystem pendulum: d theta(5) / d g (rk4, 5000 steps)",
            lag_final,
            9.81,
        ),
    ]
    for label, fn, x0 in cases:
        grad = float(jax.grad(fn)(x0))
        hstep = 1e-5 * max(abs(x0), 1.0)
        fd = float((fn(x0 + hstep) - fn(x0 - hstep)) / (2 * hstep))
        rel = abs(grad - fd) / max(abs(fd), 1e-300)
        rows.append(
            {
                "case": label,
                "at": x0,
                "autodiff": grad,
                "central_fd": fd,
                "fd_step": hstep,
                "rel_diff": rel,
                "passed": bool(rel < 1e-6),
            }
        )
    return {
        "name": "jax.grad through simulate() vs central finite differences",
        "rows": rows,
        "criterion": "relative difference < 1e-6 (central FD truncation error ~h^2)",
        "passed": all(r["passed"] for r in rows),
    }


# ---------------------------------------------------------------------------
# 8. Compressible Euler: Sod shock tube vs the exact Riemann solution
# ---------------------------------------------------------------------------


def _exact_sod_density(x: np.ndarray, t: float, g: float = 1.4) -> np.ndarray:
    """Exact density of Sod's problem (Toro, Riemann Solvers, Ch. 4)."""
    from scipy.optimize import brentq

    rl, pl, rr, pr = 1.0, 1.0, 0.125, 0.1
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def f(p: float, r: float, pk: float, c: float) -> float:
        if p > pk:
            a, b = 2 / ((g + 1) * r), (g - 1) / (g + 1) * pk
            return float((p - pk) * np.sqrt(a / (p + b)))
        return float(2 * c / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1))

    ps = brentq(lambda p: f(p, rl, pl, cl) + f(p, rr, pr, cr), 1e-8, 10)
    us = 0.5 * (f(ps, rr, pr, cr) - f(ps, rl, pl, cl))
    r_left_star = rl * (ps / pl) ** (1 / g)
    ratio = ps / pr
    r_right_star = rr * (ratio + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ratio + 1)
    shock = cr * np.sqrt((g + 1) / (2 * g) * ratio + (g - 1) / (2 * g))
    tail = us - cl * (ps / pl) ** ((g - 1) / (2 * g))
    xi = (x - 0.5) / t
    fan_c = 2 / (g + 1) * (cl - (g - 1) / 2 * xi)
    return np.select(
        [xi < -cl, xi < tail, xi < us, xi < shock],
        [rl, rl * (fan_c / cl) ** (2 / (g - 1)), r_left_star, r_right_star],
        rr,
    )


def sod_shock_tube(cells: list[int]) -> dict[str, Any]:
    rows = []
    for n in cells:
        x = (np.arange(n) + 0.5) / n
        left = jnp.asarray(x < 0.5)
        res = jp.solve_euler_1d(
            jnp.where(left, 1.0, 0.125),
            jnp.zeros(n),
            jnp.where(left, 1.0, 0.1),
            dx=1.0 / n,
            t_end=0.2,
            save_every=10,
        )
        err = float(
            np.mean(np.abs(np.asarray(res.rho[-1]) - _exact_sod_density(x, 0.2)))
        )
        rows.append({"n_cells": n, "density_l1_error": err, "t": float(res.t[-1])})
    order = _fit_order(
        [1.0 / r["n_cells"] for r in rows], [r["density_l1_error"] for r in rows]
    )
    return {
        "name": "Compressible Euler: Sod shock tube vs exact Riemann solution",
        "rows": rows,
        "observed_order_in_dx": order,
        "criterion": "density L1 error < 5e-3 at 400 cells and decreasing under "
        "refinement (discontinuities limit the order to <= 1)",
        "passed": bool(
            all(r["density_l1_error"] < 5e-3 for r in rows if r["n_cells"] >= 400)
            and all(
                b["density_l1_error"] < a["density_l1_error"]
                for a, b in itertools.pairwise(rows)
            )
        ),
    }


# ---------------------------------------------------------------------------
# 9. FDFD: line source vs the 2D free-space Green's function
# ---------------------------------------------------------------------------


def fdfd_green_function(points_per_wavelength: list[int]) -> dict[str, Any]:
    from scipy.special import hankel1

    frequency = 3e9
    k = 2 * np.pi * frequency / C0
    omega = 2 * np.pi * frequency
    rows = []
    for ppw in points_per_wavelength:
        dx = C0 / frequency / ppw
        n = round(3.4 * ppw) | 1  # odd, ~3.4 wavelengths across
        c = n // 2
        source = jnp.zeros((n, n)).at[c, c].set(1.0 / dx**2)  # unit line current
        ez = np.asarray(jp.solve_fdfd(jnp.ones((n, n)), source, frequency, dx, 12).ez)
        # Compare between 5 cells from the source and the PML.
        offsets = np.arange(5, c - 13)
        errs = []
        for field, r in (
            (ez[c + offsets, c], offsets * dx),
            (ez[c + offsets, c + offsets], np.sqrt(2) * offsets * dx),
        ):
            exact = -(omega * MU0 / 4) * hankel1(0, k * r)
            errs.append(float(np.max(np.abs(field - exact)) / np.max(np.abs(exact))))
        rows.append(
            {
                "points_per_wavelength": ppw,
                "grid": n,
                "max_rel_error_axis": errs[0],
                "max_rel_error_diagonal": errs[1],
            }
        )
    order = _fit_order(
        [1.0 / r["points_per_wavelength"] for r in rows],
        [max(r["max_rel_error_axis"], r["max_rel_error_diagonal"]) for r in rows],
    )
    return {
        "name": "FDFD line source vs 2D Green's function (Hankel H0)",
        "rows": rows,
        "observed_order_in_dx": order,
        "criterion": "max relative error < 2% at 30 points per wavelength, "
        "second-order convergence (order within 0.5 of 2)",
        "passed": bool(
            all(
                max(r["max_rel_error_axis"], r["max_rel_error_diagonal"]) < 0.02
                for r in rows
                if r["points_per_wavelength"] >= 30
            )
            and abs(order - 2.0) < 0.5
        ),
    }


# ---------------------------------------------------------------------------
# 10. SPH: standing acoustic wave period
# ---------------------------------------------------------------------------


def sph_sound_wave(n_side: int) -> dict[str, Any]:
    dx = 1.0 / n_side
    xs = jnp.arange(n_side) * dx + dx / 2
    pos = jnp.stack(jnp.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    c0 = 10.0
    probe = jp.SPHFluid(mass=1000.0 * dx**2, smoothing_length=1.3 * dx, box=(1, 1))
    fluid = jp.SPHFluid(
        mass=1000.0 * dx**2,
        smoothing_length=1.3 * dx,
        box=(1.0, 1.0),
        rest_density=float(jnp.mean(probe.density(pos))),
        sound_speed=c0,
        alpha=0.0,
    )
    vel = jnp.stack([0.05 * jnp.sin(2 * jnp.pi * pos[:, 0]), jnp.zeros(len(pos))], -1)
    dt = 0.2 * fluid.h / c0
    traj = fluid.simulate(pos, vel, (0.0, 0.04), dt, save_every=1)
    ke = np.asarray(traj.kinetic_energy)
    t = np.asarray(traj.t)
    i = int(np.argmin(ke))
    # Parabolic interpolation of the kinetic-energy minimum.
    y0, y1, y2 = ke[i - 1 : i + 2]
    t_min = float(t[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (t[1] - t[0]))
    e0 = float(fluid.total_energy(traj.positions[0], traj.velocities[0]))
    e1 = float(fluid.total_energy(traj.positions[-1], traj.velocities[-1]))
    expected = 1.0 / (4 * c0)
    return {
        "name": "SPH standing sound wave (quarter period vs L / 4 c0)",
        "n_particles": n_side**2,
        "quarter_period_measured": t_min,
        "quarter_period_expected": expected,
        "rel_error": t_min / expected - 1.0,
        "max_abs_momentum": float(jnp.max(jnp.abs(traj.momentum))),
        "energy_change_over_initial_ke": abs(e1 - e0) / float(ke[0]),
        "criterion": "kinetic-energy minimum within 3% of L / (4 c0); total "
        "momentum < 1e-12; energy change < 1e-3 of the initial kinetic energy",
        "passed": bool(
            abs(t_min / expected - 1.0) < 0.03
            and float(jnp.max(jnp.abs(traj.momentum))) < 1e-12
            and abs(e1 - e0) < 1e-3 * float(ke[0])
        ),
    }


def run_accuracy(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [
            energy_drift(n_orbits=100),
            convergence_order(n_refinements=4),
            schrodinger_validation([1024], norm_steps=5000),
            fdtd_cavity(n_steps=12000),
            ising_tc([8, 16], n_sweeps=3000, n_warmup=500, n_wolff_samples=2000),
            lbm_channel([16, 32], driven_steps=12000),
            gradients(),
            sod_shock_tube([100, 200, 400]),
            fdfd_green_function([15, 30]),
            sph_sound_wave(24),
        ]
    return [
        energy_drift(n_orbits=1000),
        convergence_order(n_refinements=6),
        schrodinger_validation([1024, 4096], norm_steps=20000),
        fdtd_cavity(n_steps=40000),
        ising_tc([8, 16, 32], n_sweeps=20000, n_warmup=2000, n_wolff_samples=20000),
        lbm_channel([16, 32, 64], driven_steps=30000),
        gradients(),
        sod_shock_tube([100, 200, 400, 800, 1600]),
        fdfd_green_function([15, 30, 60]),
        sph_sound_wave(40),
    ]
