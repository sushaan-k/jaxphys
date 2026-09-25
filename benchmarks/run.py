"""Run the jaxphys benchmark suite and write JSON + Markdown results.

Examples (from the repository root)::

    JAX_PLATFORMS=cpu python -m benchmarks.run            # full suite
    JAX_PLATFORMS=cpu python -m benchmarks.run --quick    # CI-sized run
    JAX_PLATFORMS=cpu python -m benchmarks.run --suite perf --suite scaling

Each run writes ``<out>/<run_id>.json`` (all measurements plus environment
metadata) and ``<out>/<run_id>.md`` (tables generated from that JSON).  The
exit status is non-zero if any agreement or validation check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import jax

import jaxphys  # noqa: F401  (enables float64 before anything else runs)
from benchmarks.common import environment

SEEDS = {
    "perf.nbody_initial_conditions": "jax.random.PRNGKey(N) / fold_in(., 1)",
    "perf.ising_metropolis": "JAX PRNGKey(7); NumPy default_rng(7)",
    "perf.ising_wolff": "JAX PRNGKey(11); NumPy default_rng(11)",
    "perf.lindblad_hamiltonian": "NumPy default_rng(d)",
    "perf.sph_velocities": "NumPy default_rng(1)",
    "perf.tight_binding_k_points": "jax.random.PRNGKey(n_k)",
    "accuracy.ising_tc": "jax.random.PRNGKey(L) split per temperature",
    "accuracy.wolff_vs_metropolis": "jax.random.PRNGKey(100 + i) per temperature",
    "accuracy.gradients.nbody": "jax.random.PRNGKey(0), PRNGKey(1)",
    "scaling": "jax.random.PRNGKey(0) / PRNGKey(1)",
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def _status(check: dict[str, Any]) -> str:
    return "PASS" if check.get("passed") else "FAIL"


def _fmt_s(x: float) -> str:
    if x >= 1.0:
        return f"{x:.3f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f} ms"
    return f"{x * 1e6:.1f} us"


def _fmt_rate(x: float) -> str:
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= scale:
            return f"{x / scale:.2f}{suffix}"
    return f"{x:.1f}"


def markdown(result: dict[str, Any]) -> str:
    env = result["environment"]
    lines = [
        f"# jaxphys benchmark results: `{result['run_id']}`",
        "",
        f"- mode: **{result['mode']}**; suites: {', '.join(result['suites'])}",
        f"- timestamp (UTC): {env['timestamp_utc']}; git commit: `{env['git_commit']}`"
        + (" (dirty)" if env["git_dirty"] else ""),
        f"- CPU: {env['cpu_model']} ({env['cpu_count']} logical CPUs); "
        f"JAX devices: {', '.join(env['jax_devices'])} (backend "
        f"`{env['jax_default_backend']}`); no GPU",
        f"- python {env['python']}, jax {env['jax']}, jaxlib {env['jaxlib']}, "
        f"numpy {env['numpy']}, jaxphys {env['jaxphys']}; x64={env['jax_enable_x64']}",
        f"- wall time of this run: {result['wall_time_s']:.0f} s",
        "",
    ]
    if "perf" in result:
        lines += [
            "## Performance (public API, CPU)",
            "",
            "Compile = trace + lower + XLA compile time measured during the "
            "first call. Steady = median (IQR) of "
            f"{result['repeats']} synchronized calls after one warm-up call. "
            "NumPy = same configuration in plain NumPy "
            f"(median of {result['numpy_repeats']}).",
            "",
            "| solver | size | first call | compile | steady median (IQR) | "
            "throughput | NumPy median | speedup | agreement |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in result["perf"]:
            j, a = r["jax"], r["agreement"]
            if a.get("kind") == "statistical":
                agree = f"z={a['z_score']:.1f} (<= {a['n_sigma']:.0f})"
            else:
                agree = f"max rel {a.get('max_rel', float('nan')):.1e} (tol {a.get('rtol')})"
            status = _status(a)
            lines.append(
                f"| {r['solver']} | {r['size']} | {_fmt_s(j['first_call_s'])} | "
                f"{_fmt_s(j['compile']['compile_s'])} | "
                f"{_fmt_s(j['steady']['median_s'])} ({_fmt_s(j['steady']['iqr_s'])}) | "
                f"{_fmt_rate(r['jax_throughput'])} {r['throughput_unit']} | "
                f"{_fmt_s(r['numpy']['steady']['median_s'])} | "
                f"{r['speedup_vs_numpy']:.1f}x | {status}: {agree} |"
            )
        lines.append("")
    if "scaling" in result:
        lines += [
            "## vmap batch scaling (public API under jax.jit(jax.vmap(...)), CPU)",
            "",
            "Efficiency = B * t(1) / t(B); > 1 means one batched call beats B "
            "sequential single calls.",
            "",
        ]
        for k in result["scaling"]:
            lines += [
                f"### {k['kernel']} ({k['work_per_trajectory']} per trajectory)",
                "",
                "| batch | median (IQR) | trajectories/s | efficiency |",
                "|---|---|---|---|",
            ]
            for row in k["rows"]:
                st = row["steady"]
                lines.append(
                    f"| {row['batch']} | {_fmt_s(st['median_s'])} "
                    f"({_fmt_s(st['iqr_s'])}) | {row['trajectories_per_s']:.1f} | "
                    f"{row['efficiency']:.2f} |"
                )
            lines.append("")
    if "accuracy" in result:
        lines += ["## Accuracy / validation", ""]
        lines += _accuracy_markdown(result["accuracy"])
    return "\n".join(lines) + "\n"


def _accuracy_markdown(items: list[dict[str, Any]]) -> list[str]:
    out = []
    for item in items:
        status = _status(item)
        out += [
            f"### {item['name']}: {status}",
            "",
            f"Criterion: {item['criterion']}",
            "",
        ]
        name = item["name"]
        if name.startswith("Energy error"):
            s = item["setup"]
            out += [
                f"{s['orbits']} orbits, {s['steps_per_orbit']} steps/orbit, dt={s['dt']:.4g}.",
                "",
                "| integrator | max abs(dE/E) | final dE/E | max, 1st half | "
                "max, 2nd half | linear drift / orbit |",
                "|---|---|---|---|---|---|",
            ]
            for r in item["rows"]:
                out.append(
                    f"| {r['integrator']} | {r['max_abs_rel_energy_error']:.3e} | "
                    f"{r['final_rel_energy_error']:.3e} | {r['max_error_first_half']:.3e} | "
                    f"{r['max_error_second_half']:.3e} | {r['linear_drift_per_orbit']:.3e} |"
                )
        elif name.startswith("Convergence"):
            out += [
                "| problem | integrator | expected | measured | finest dt | error at finest dt | status |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in item["rows"]:
                out.append(
                    f"| {r['problem']} | {r['integrator']} | {r['expected_order']} | "
                    f"{r['measured_order']:.2f} | {r['dts'][-1]:.3e} | {r['errors'][-1]:.3e} | "
                    f"{'PASS' if r['passed'] else 'FAIL'} |"
                )
        elif name.startswith("Schrodinger"):
            nm = item["norm"]
            out += [
                f"Norm: max abs(norm - 1) = {nm['max_abs_norm_deviation']:.2e} over "
                f"{nm['n_steps']} steps ({nm['n_points']} points, barrier).",
                "",
                "| n_points | psi rel L2 err vs exact | density rel L2 err | "
                "center (meas / exact) | width (meas / exact) |",
                "|---|---|---|---|---|",
            ]
            for r in item["free_gaussian"]:
                out.append(
                    f"| {r['n_points']} | {r['psi_rel_l2_error_vs_exact']:.3e} | "
                    f"{r['density_rel_l2_error_vs_exact']:.3e} | "
                    f"{r['center_measured']:.5f} / {r['center_exact']:.5f} | "
                    f"{r['width_measured']:.5f} / {r['width_exact']:.5f} |"
                )
        elif name.startswith("FDTD"):
            out += [
                "| mode | measured (GHz) | continuum (GHz) | Yee dispersion (GHz) | "
                "rel err vs continuum | rel err vs Yee | FFT bin / f |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in item["rows"]:
                out.append(
                    f"| {r['mode']} | {r['measured_hz'] / 1e9:.6f} | "
                    f"{r['analytic_continuum_hz'] / 1e9:.6f} | "
                    f"{r['analytic_yee_dispersion_hz'] / 1e9:.6f} | "
                    f"{r['rel_error_vs_continuum']:.2e} | {r['rel_error_vs_yee']:.2e} | "
                    f"{r['fft_bin_rel']:.1e} |"
                )
        elif name.startswith("Ising"):
            s = item["setup"]
            out += [
                f"Sizes {s['sizes']}, {s['n_sweeps']} sweeps (+{s['n_warmup']} warm-up) "
                f"at each of {len(s['temperatures'])} temperatures in "
                f"[{s['temperatures'][0]}, {s['temperatures'][-1]}]. Onsager "
                f"T_c = {s['onsager_tc']:.6f}.",
                "",
                "| sizes | T_c estimate | jackknife error | deviation | rel. deviation |",
                "|---|---|---|---|---|",
            ]
            for r in item["crossings"]:
                out.append(
                    f"| {r['sizes'][0]} / {r['sizes'][1]} | {r['tc_estimate']:.4f} | "
                    f"{r['jackknife_error']:.4f} | {r['deviation_from_onsager']:+.4f} | "
                    f"{r['rel_deviation']:+.2%} |"
                )
            w = item["wolff_vs_metropolis"]
            out += [
                "",
                f"Wolff vs Metropolis ({w['lattice']}x{w['lattice']}, "
                f"{w['samples_per_chain']} samples per chain), energy per spin:",
                "",
                "| T | Wolff | Metropolis | z | status |",
                "|---|---|---|---|---|",
            ]
            for r in w["rows"]:
                out.append(
                    f"| {r['temperature']} | {r['wolff_energy']:.4f} +- "
                    f"{r['wolff_err']:.4f} | {r['metropolis_energy']:.4f} +- "
                    f"{r['metropolis_err']:.4f} | {r['z_score']:.2f} | "
                    f"{'PASS' if r['passed'] else 'FAIL'} |"
                )
        elif name.startswith("LBM"):
            out += [
                "(a) Fundamental shear-mode decay between no-slip walls:",
                "",
                "| channel width H | measured rate | analytic nu (pi/H)^2 | rel error | "
                "effective width | mode-shape residual |",
                "|---|---|---|---|---|---|",
            ]
            for r in item["decay"]:
                out.append(
                    f"| {r['channel_width_nodes']} | {r['measured_decay_rate']:.4e} | "
                    f"{r['analytic_decay_rate_halfway_walls']:.4e} | {r['rel_error']:+.2%} | "
                    f"{r['effective_width_from_rate']:.2f} | {r['max_mode_shape_residual']:.1e} |"
                )
            d = item["driven"]
            dens = ", ".join(f"{v:.4f}" for v in d["mean_density_by_snapshot"])
            out += [
                "",
                f"Observed convergence order of the decay-rate error in 1/H: "
                f"{item['decay_error_convergence_order_in_1_over_H']:.2f}.",
                "",
                f"(b) Inflow-driven channel {d['grid'][0]}x{d['grid'][1]}, "
                f"u_in={d['u_inlet']}, {d['n_steps']} steps: "
                f"{'PASS' if d['passed'] else 'FAIL'}",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| mean density at steps {d['snapshot_steps']} | {dens} |",
                f"| relative mass change over the last interval | "
                f"{d['mass_change_last_interval']:.1e} |",
                f"| max mass-flux error vs inflow (all sections) | "
                f"{d['max_flux_error_vs_inflow']:.1e} |",
                f"| profile max error vs parabola (same flow rate) | "
                f"{d['profile_max_error_vs_parabola']:.2e} |",
                f"| dp/dx measured / Poiseuille | {d['dp_dx_measured']:.4e} / "
                f"{d['dp_dx_poiseuille']:.4e} ({d['dp_dx_rel_error']:+.2%}) |",
            ]
        elif name.startswith("jax.grad"):
            out += [
                "| case | autodiff | central FD | FD step | rel diff | status |",
                "|---|---|---|---|---|---|",
            ]
            for r in item["rows"]:
                out.append(
                    f"| {r['case']} | {r['autodiff']:.10e} | {r['central_fd']:.10e} | "
                    f"{r['fd_step']:.0e} | {r['rel_diff']:.2e} | "
                    f"{'PASS' if r['passed'] else 'FAIL'} |"
                )
        elif name.startswith("Compressible Euler"):
            out += [
                f"Observed order in dx: {item['observed_order_in_dx']:.2f}.",
                "",
                "| cells | density L1 error |",
                "|---|---|",
            ]
            for r in item["rows"]:
                out.append(f"| {r['n_cells']} | {r['density_l1_error']:.3e} |")
        elif name.startswith("FDFD"):
            out += [
                f"Observed order in dx: {item['observed_order_in_dx']:.2f}.",
                "",
                "| points / wavelength | grid | max rel err (axis) | max rel err (diagonal) |",
                "|---|---|---|---|",
            ]
            for r in item["rows"]:
                out.append(
                    f"| {r['points_per_wavelength']} | {r['grid']}^2 | "
                    f"{r['max_rel_error_axis']:.2e} | {r['max_rel_error_diagonal']:.2e} |"
                )
        elif name.startswith("SPH"):
            out += [
                f"{item['n_particles']} particles: quarter period "
                f"{item['quarter_period_measured']:.5f} s vs L/(4 c0) = "
                f"{item['quarter_period_expected']:.5f} s ({item['rel_error']:+.2%}); "
                f"max |momentum| {item['max_abs_momentum']:.1e}; energy change "
                f"{item['energy_change_over_initial_ke']:.1e} of the initial KE.",
            ]
        out.append("")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small CI-sized run")
    parser.add_argument(
        "--suite",
        action="append",
        choices=["perf", "scaling", "accuracy"],
        help="suites to run (repeatable; default: all)",
    )
    parser.add_argument(
        "--repeats", type=int, default=None, help="timed JAX repeats (>=5)"
    )
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    parser.add_argument("--label", default="", help="suffix for the run id")
    args = parser.parse_args(argv)

    suites = args.suite or ["perf", "scaling", "accuracy"]
    repeats = max(args.repeats or (5 if args.quick else 7), 5)
    numpy_repeats = 3 if args.quick else 5
    mode = "quick" if args.quick else "full"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{mode}"
    if args.label:
        run_id += f"_{args.label}"

    t_start = time.perf_counter()
    result: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "suites": suites,
        "repeats": repeats,
        "numpy_repeats": numpy_repeats,
        "seeds": SEEDS,
        "environment": environment(),
    }
    failures: list[str] = []

    if "perf" in suites:
        from benchmarks.perf import all_cases, run_case

        rows = []
        for case in all_cases(args.quick):
            print(f"[perf] {case.solver} :: {case.size}", flush=True)
            row = run_case(case, repeats=repeats, numpy_repeats=numpy_repeats)
            rows.append(row)
            if _status(row["agreement"]) == "FAIL":
                failures.append(f"perf {case.solver} {case.size}")
        result["perf"] = rows

    if "scaling" in suites:
        from benchmarks.scaling import run_scaling

        print("[scaling] vmap batch scaling", flush=True)
        batches = [1, 4, 16] if args.quick else [1, 2, 4, 8, 16, 32, 64]
        result["scaling"] = run_scaling(batches, repeats)

    if "accuracy" in suites:
        from benchmarks.accuracy import run_accuracy

        print("[accuracy] validation experiments", flush=True)
        items = run_accuracy(args.quick)
        for item in items:
            if _status(item) == "FAIL":
                failures.append(f"accuracy {item['name']}")
        result["accuracy"] = items

    result["wall_time_s"] = time.perf_counter() - t_start
    result["summary"] = {"failures": failures}
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"{run_id}.json")
    md_path = os.path.join(args.out, f"{run_id}.md")
    clean = _jsonable(result)
    with open(json_path, "w") as fh:
        json.dump(clean, fh, indent=1)
    with open(md_path, "w") as fh:
        fh.write(markdown(clean))
    print(f"wrote {json_path} and {md_path}")
    print(f"devices: {jax.devices()}")
    for f in failures:
        print(f"FAIL: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
