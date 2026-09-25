"""Tests for the SPH and compressible Euler solvers against analytic results."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxphys as jp
from jaxphys.exceptions import ConfigurationError
from jaxphys.fluids.sph import cubic_spline_gradient, cubic_spline_kernel

# --------------------------------------------------------------------- SPH


def _lattice(n_side: int, box: float = 1.0) -> tuple[jax.Array, float]:
    dx = box / n_side
    xs = jnp.arange(n_side) * dx + dx / 2
    grid = jnp.stack(jnp.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    return grid, dx


def test_kernel_is_normalized_and_gradient_consistent() -> None:
    h = 0.7
    r = np.linspace(0.0, 2 * h, 20001)
    w = np.asarray(cubic_spline_kernel(jnp.asarray(r), h))
    assert np.trapezoid(w * 2 * np.pi * r, r) == pytest.approx(1.0, abs=1e-8)
    for x in (0.2, 0.9, 1.3):
        auto = jax.grad(lambda d: cubic_spline_kernel(d, h))(x)
        assert float(cubic_spline_gradient(x, h)) == pytest.approx(float(auto))


def test_uniform_lattice_density_and_cell_list_agree_with_all_pairs() -> None:
    pos, dx = _lattice(30)
    fluid = jp.SPHFluid(mass=1000.0 * dx**2, smoothing_length=1.3 * dx, box=(1, 1))
    rho = fluid.density(pos)
    assert float(jnp.max(jnp.abs(rho / 1000.0 - 1.0))) < 1e-3
    assert fluid.cell_capacity(pos) is not None
    vel = 0.1 * jax.random.normal(jax.random.PRNGKey(0), pos.shape)
    acc_cells, _ = fluid.acceleration(pos, vel, fluid.cell_capacity(pos))
    acc_pairs, _ = fluid.acceleration(pos, vel, None)
    np.testing.assert_allclose(acc_cells, acc_pairs, atol=1e-9)


def test_sound_wave_period_and_conservation() -> None:
    """A standing acoustic wave's kinetic energy first vanishes at t = L/(4 c0)."""
    pos, dx = _lattice(40)
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
    traj = fluid.simulate(pos, vel, (0.0, 0.04), dt, save_every=2)
    ke = np.asarray(traj.kinetic_energy)
    t_min = float(traj.t[int(np.argmin(ke))])
    assert t_min == pytest.approx(1.0 / (4 * c0), rel=0.03)
    assert float(jnp.max(jnp.abs(traj.momentum))) < 1e-12
    e0 = fluid.total_energy(traj.positions[0], traj.velocities[0])
    e1 = fluid.total_energy(traj.positions[-1], traj.velocities[-1])
    assert abs(float(e1 - e0)) < 1e-3 * ke[0]


def test_reflecting_box_keeps_particles_inside_and_is_differentiable() -> None:
    pos, dx = _lattice(12, box=0.5)
    fluid = jp.SPHFluid(
        mass=1000.0 * dx**2,
        smoothing_length=1.3 * dx,
        box=(0.5, 0.5),
        sound_speed=20.0,
        gravity=(0.0, -9.81),
        boundary="reflecting",
    )
    dt = 0.2 * fluid.h / fluid.sound_speed
    traj = fluid.simulate(pos, jnp.zeros_like(pos), (0.0, 0.05), dt, save_every=25)
    assert bool(jnp.all((traj.positions >= 0) & (traj.positions <= 0.5)))

    def final_ke(speed):
        vel = jnp.zeros_like(pos).at[:, 0].set(speed)
        out = fluid.simulate(pos, vel, (0.0, 20 * dt), dt, save_every=20)
        return out.kinetic_energy[-1]

    grad = jax.grad(final_ke)(0.1)
    fd = (final_ke(0.1 + 1e-5) - final_ke(0.1 - 1e-5)) / 2e-5
    assert float(grad) == pytest.approx(float(fd), rel=1e-4)


def test_sph_validation() -> None:
    with pytest.raises(ConfigurationError):
        jp.SPHFluid(mass=1.0, smoothing_length=0.5, box=(1.0, 1.0))
    with pytest.raises(ConfigurationError):
        jp.SPHFluid(mass=1.0, smoothing_length=0.01, box=(1, 1), boundary="open")


# ------------------------------------------------------------------- Euler


def _exact_sod(x: np.ndarray, t: float, g: float = 1.4) -> np.ndarray:
    """Exact density of Sod's problem (Toro, Ch. 4)."""
    from scipy.optimize import brentq

    rl, pl, rr, pr = 1.0, 1.0, 0.125, 0.1
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def f(p, r, pk, c):
        if p > pk:
            a, b = 2 / ((g + 1) * r), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(a / (p + b))
        return 2 * c / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

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


def _sod(n: int, **kwargs):
    x = (np.arange(n) + 0.5) / n
    left = jnp.asarray(x < 0.5)
    return jp.solve_euler_1d(
        jnp.where(left, 1.0, 0.125),
        jnp.zeros(n),
        jnp.where(left, 1.0, 0.1),
        dx=1.0 / n,
        **kwargs,
    )


def test_sod_shock_tube_matches_exact_solution() -> None:
    result = _sod(400, t_end=0.2, save_every=50)
    assert float(result.t[-1]) == pytest.approx(0.2)
    error = np.mean(
        np.abs(np.asarray(result.rho[-1]) - _exact_sod(np.asarray(result.x), 0.2))
    )
    assert error < 5e-3
    coarse = _sod(100, t_end=0.2, save_every=10)
    coarse_error = np.mean(
        np.abs(np.asarray(coarse.rho[-1]) - _exact_sod(np.asarray(coarse.x), 0.2))
    )
    assert error < 0.5 * coarse_error  # converges under refinement


@pytest.mark.parametrize("boundary", ["periodic", "reflective"])
def test_euler_conserves_mass_and_energy(boundary: str) -> None:
    n = 200
    x = (jnp.arange(n) + 0.5) / n
    u0 = 0.3 if boundary == "periodic" else 0.0
    result = jp.solve_euler_1d(
        1 + 0.2 * jnp.sin(2 * jnp.pi * x),
        jnp.full(n, u0),
        1 + 0.1 * jnp.cos(2 * jnp.pi * x),
        dx=1.0 / n,
        t_end=0.3,
        boundary=boundary,
    )
    for total in (result.rho.sum(1), result.energy.sum(1)):
        assert float(jnp.max(jnp.abs(total - total[0])) / total[0]) < 1e-12


def test_euler_is_differentiable_and_jittable() -> None:
    def run(p_left):
        n = 100
        x = (jnp.arange(n) + 0.5) / n
        out = jp.solve_euler_1d(
            jnp.where(x < 0.5, 1.0, 0.125),
            jnp.zeros(n),
            jnp.where(x < 0.5, p_left, 0.1),
            dx=1.0 / n,
            t_end=0.1,
            dt=1e-3,
            save_every=100,
        )
        return out.p[-1].mean()

    grad = jax.grad(run)(1.0)
    fd = (run(1.0 + 1e-5) - run(1.0 - 1e-5)) / 2e-5
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)
    assert float(jax.jit(run)(1.0)) == pytest.approx(float(run(1.0)))
    with pytest.raises(ConfigurationError, match="dt"):
        jax.jit(lambda p: jp.solve_euler_1d(p, p, p, 0.1, 1.0).p)(jnp.ones(8))
