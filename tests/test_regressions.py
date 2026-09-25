"""Regression tests for bugs fixed in 0.2.0.

Each test pins down one previously incorrect behaviour against an analytic
or independently computed reference.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxphys as jp
from jaxphys.classical.integrators import adaptive_rk45, yoshida4
from jaxphys.classical.nbody import gravitational_accelerations, total_energy
from jaxphys.em.fdtd import C0
from jaxphys.exceptions import ConfigurationError
from jaxphys.optics.diffraction import _bessel_j1
from jaxphys.statmech.boltzmann import entropy, free_energy, mean_energy
from jaxphys.statmech.ising import _run_wolff_temperature
from jaxphys.statmech.monte_carlo import metropolis_step


def _oscillator(q, p, params):
    return 0.5 * p[0] ** 2 + 0.5 * params.k * q[0] ** 2


# ---------------------------------------------------------------- classical


def test_yoshida4_evaluates_forces_at_drift_times() -> None:
    """A time-dependent force must be sampled at the correct sub-step times."""

    def deriv(q, p, t, _):
        return p, jnp.cos(t) * jnp.ones_like(q)

    q, p, t, dt = jnp.zeros(1), jnp.zeros(1), 0.0, 0.1
    for _ in range(50):
        q, p, t = yoshida4(deriv, q, p, t, dt, None)
    # Exact: p = sin t, q = 1 - cos t. The old time bookkeeping gave O(dt) error.
    assert float(jnp.abs(p[0] - np.sin(t))) < 1e-6
    assert float(jnp.abs(q[0] - (1 - np.cos(t)))) < 1e-6


def test_adaptive_rk45_give_up_returns_consistent_step() -> None:
    """After max_reject the returned state must belong to the returned time."""

    def deriv(q, p, t, _):
        return p, -q

    q, _, t_new = adaptive_rk45(
        deriv,
        jnp.ones(1),
        jnp.zeros(1),
        0.0,
        1.0,
        None,
        atol=1e-16,
        rtol=1e-16,
        max_reject=1,
    )
    assert 0.0 < t_new < 1.0
    assert float(jnp.abs(q[0] - np.cos(t_new))) < 1e-4


def test_simulate_is_jittable_and_differentiable_in_params() -> None:
    system = jp.HamiltonianSystem(_oscillator, n_dof=1)

    def final_q(k):
        return system.simulate([1.0], [0.0], (0, 1), 0.001, jp.Params(k=k)).q[-1, 0]

    grad = jax.grad(final_q)(4.0)
    # q(t) = cos(sqrt(k) t) => dq/dk = -t sin(sqrt(k) t) / (2 sqrt(k)) at t=1.
    assert float(grad) == pytest.approx(-np.sin(2.0) / 4.0, rel=1e-4)
    traj = jax.jit(
        lambda k: system.simulate([1.0], [0.0], (0, 1), 0.01, jp.Params(k=k))
    )(4.0)
    assert isinstance(traj, jp.Trajectory)
    batched = jax.vmap(final_q)(jnp.array([1.0, 4.0]))
    assert batched.shape == (2,)


def test_save_every_matches_subsampled_full_trajectory() -> None:
    system = jp.HamiltonianSystem(_oscillator, n_dof=1)
    params = jp.Params(k=2.0)
    full = system.simulate([1.0], [0.0], (0, 1), 0.01, params, save_every=1)
    sub = system.simulate([1.0], [0.0], (0, 1), 0.01, params, save_every=7)
    np.testing.assert_allclose(sub.q, full.q[::7], atol=1e-12)
    np.testing.assert_allclose(sub.t, full.t[::7], atol=1e-12)


def test_unscannable_integrators_raise_clear_error() -> None:
    system = jp.HamiltonianSystem(_oscillator, n_dof=1)
    for name in ("velocity_verlet", "adaptive_rk45"):
        with pytest.raises(ConfigurationError, match=name):
            system.simulate(
                [1.0], [0.0], (0, 1), 0.01, jp.Params(k=1.0), integrator=name
            )


def test_nbody_gradients_finite_without_softening() -> None:
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = jnp.array([1.0, 2.0, 3.0])
    g_acc = jax.jacobian(gravitational_accelerations)(pos, masses, 1.0, 0.0)
    g_energy = jax.grad(total_energy)(pos, jnp.zeros_like(pos), masses, 1.0, 0.0)
    assert bool(jnp.all(jnp.isfinite(g_acc)))
    # -dU/dr is the force m * a.
    np.testing.assert_allclose(
        -g_energy, masses[:, None] * gravitational_accelerations(pos, masses, 1.0, 0.0)
    )


def test_nbody_rejects_invalid_step_counts() -> None:
    body = jp.NBody(masses=[1.0], positions=[[0, 0, 0]], velocities=[[0, 0, 0]])
    with pytest.raises(ConfigurationError):
        body.simulate(n_steps=0)


# ---------------------------------------------------------------------- EM


def test_charge_speed_conserved_in_magnetic_field() -> None:
    """Boris pushes conserve |v| exactly in a pure magnetic field."""
    charge = jp.PointCharge(
        charge=1.0, mass=1.0, position=[0, 0, 0], velocity=[1, 0, 0]
    )
    system = jp.ChargeSystem([charge], B_external=jnp.array([0.0, 0.0, 1.0]))
    traj = system.simulate((0, 100.0), n_steps=5000, save_every=100)
    speed = jnp.linalg.norm(traj.velocities[:, 0], axis=-1)
    assert float(jnp.max(jnp.abs(speed - 1.0))) < 1e-10
    # Gyration radius m v / (q B) = 1 about the guiding centre (0, -1).
    radius = jnp.linalg.norm(traj.positions[:, 0, :2] - jnp.array([0.0, -1.0]), axis=-1)
    assert float(jnp.max(jnp.abs(radius - 1.0))) < 1e-3


def _probe_2d(ny: int, pml: int, boundary: str = "absorbing", steps: int = 300):
    dx = 0.01
    grid = jp.EMGrid(size=(60, ny), resolution=dx, boundary=boundary, pml_layers=pml)
    grid.add_source(jp.PlaneWave(frequency=3e9, y=ny // 2))
    dt = 0.99 * dx / (C0 * np.sqrt(2))
    fields = grid.simulate(t_span=(0, steps * dt * (1 + 1e-9)), dt=dt, save_every=1)
    return fields.ez[:, 30, ny // 2 + 15]


def test_fdtd_2d_pml_matches_unbounded_reference() -> None:
    """Fields next to a small PML-terminated grid match a much larger grid."""
    reference = _probe_2d(800, 10)
    scale = float(jnp.max(jnp.abs(reference)))
    pml_error = float(jnp.max(jnp.abs(_probe_2d(60, 10) - reference))) / scale
    pec_error = float(jnp.max(jnp.abs(_probe_2d(60, 10, "reflecting") - reference)))
    assert pml_error < 0.05
    assert pec_error / scale > 1.0


def test_fdtd_3d_pml_matches_unbounded_reference() -> None:
    dx = 0.01
    dt = 0.99 * dx / (C0 * np.sqrt(3))

    def probe(n: int, pml: int):
        grid = jp.EMGrid3D(size=(n, n, n), resolution=dx, pml_layers=pml)
        c = n // 2
        grid.add_source(jp.PointSource3D(frequency=3e9, position=(c, c, c)))
        fields = grid.simulate(t_span=(0, 110 * dt * (1 + 1e-9)), dt=dt, save_every=1)
        return fields.ez[:, c + 6, c, c]

    reference = probe(72, 8)
    scale = float(jnp.max(jnp.abs(reference)))
    assert float(jnp.max(jnp.abs(probe(32, 8) - reference))) / scale < 0.05
    assert float(jnp.max(jnp.abs(probe(32, 0) - reference))) / scale > 0.5


def test_fdtd_rejects_cfl_violation_and_reports_times() -> None:
    grid = jp.EMGrid(size=(20, 20), resolution=0.01)
    grid.add_source(jp.PlaneWave(frequency=3e9, y=10))
    with pytest.raises(ConfigurationError, match="CFL"):
        grid.simulate(t_span=(0, 1e-9), dt=1e-10)
    dt = 1e-11
    fields = grid.simulate(t_span=(0, 20 * dt * (1 + 1e-9)), dt=dt, save_every=5)
    np.testing.assert_allclose(fields.t, dt * np.array([1, 6, 11, 16]))
    grid3 = jp.EMGrid3D(size=(8, 8, 8), resolution=0.01)
    grid3.add_source(jp.PointSource3D(frequency=3e9, position=(4, 4, 4)))
    with pytest.raises(ConfigurationError, match="CFL"):
        grid3.simulate(t_span=(0, 1e-9), dt=1e-10)


# ------------------------------------------------------------------ optics


def test_bessel_j1_accurate_for_large_arguments() -> None:
    scipy_special = pytest.importorskip("scipy.special")
    x = np.linspace(-500.0, 500.0, 20001)
    np.testing.assert_allclose(
        _bessel_j1(jnp.asarray(x)), scipy_special.j1(x), atol=1e-7
    )


def test_image_distance_from_system_matrix() -> None:
    # Object at z = 0, lens f = 0.1 at z = 0.3: 1/0.3 + 1/d = 1/0.1 => d = 0.15.
    result = jp.trace_system(
        jp.Ray(y=0.01, theta=0.0), [jp.ThinLens(f=0.1, position=0.3)]
    )
    assert result.image_distance == pytest.approx(0.15)


# ----------------------------------------------------------------- quantum


def test_schrodinger_group_velocity_and_time_grid() -> None:
    packet = jp.GaussianWavepacket(x0=-5.0, k0=2.0, sigma=1.0)
    result = jp.solve_schrodinger(
        packet,
        jp.HarmonicPotential(k=0.0),
        x_range=(-20, 20),
        t_span=(0, 5),
        n_points=256,
        dt=0.01,
        save_every=100,
    )
    prob = jnp.abs(result.psi[-1]) ** 2
    mean_x = float(jnp.sum(prob * result.x) / jnp.sum(prob))
    assert mean_x == pytest.approx(-5.0 + 2.0 * 5.0, abs=1e-3)
    # t_span not divisible by dt: times are multiples of dt, not a linspace.
    short = jp.solve_schrodinger(
        packet,
        jp.HarmonicPotential(k=0.0),
        x_range=(-20, 20),
        t_span=(0, 1.005),
        n_points=64,
        dt=0.01,
        save_every=1,
    )
    np.testing.assert_allclose(short.t, 0.01 * np.arange(101), atol=1e-12)


def test_lindblad_time_grid_and_amplitude_damping() -> None:
    gamma = 0.5
    rho0 = jp.DensityMatrix.from_pure_state(jnp.array([0.0, 1.0]))  # excited
    lowering = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    result = jp.lindblad_evolve(
        rho0,
        jnp.zeros((2, 2)),
        [lowering],
        [gamma],
        t_span=(0, 2.005),
        dt=0.01,
        save_every=10,
    )
    np.testing.assert_allclose(result.t, 0.1 * np.arange(21), atol=1e-12)
    excited = jnp.real(result.rho[:, 1, 1])
    np.testing.assert_allclose(excited, np.exp(-gamma * result.t), atol=1e-8)


def test_spin_hamiltonian_matches_kronecker_construction() -> None:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1.0, -1.0]).astype(complex)

    def site(op, i, n):
        out = np.eye(1)
        for k in range(n):
            out = np.kron(out, op if k == i else np.eye(2))
        return out

    n, J, h = 4, 0.7, 0.3
    expected = sum(
        -0.25 * J * site(s, i, n) @ site(s, (i + 1) % n, n)
        for i in range(n)
        for s in (sx, sy, sz)
    ) - 0.5 * h * sum(site(sz, i, n) for i in range(n))
    H = jp.SpinChain(n, J=J, h=h, periodic=True).build_hamiltonian()
    np.testing.assert_allclose(H, expected, atol=1e-12)
    with pytest.raises(ConfigurationError):
        jp.SpinChain(15)


# ---------------------------------------------------------------- statmech


def _exact_ising(L_x: int, L_y: int, T: float) -> tuple[float, float]:
    states = np.array(list(itertools.product([-1, 1], repeat=L_x * L_y)))
    states = states.reshape(-1, L_x, L_y)
    energy = -(states * np.roll(states, 1, 1) + states * np.roll(states, 1, 2)).sum(
        (1, 2)
    )
    weights = np.exp(-(energy - energy.min()) / T)
    weights /= weights.sum()
    n = L_x * L_y
    return float(weights @ energy) / n, float(weights @ np.abs(states.mean((1, 2))))


@pytest.mark.parametrize("size", [(4, 4), (3, 3)])
def test_metropolis_matches_exact_enumeration(size) -> None:
    exact_e, exact_m = _exact_ising(*size, 2.5)
    lattice = jp.IsingLattice(size)
    result = lattice.run_metropolis(2.5, 20000, 500, jax.random.PRNGKey(3))
    assert result["energy"] == pytest.approx(exact_e, abs=0.03)
    assert result["magnetization"] == pytest.approx(exact_m, abs=0.03)


def test_wolff_matches_exact_enumeration() -> None:
    exact_e, exact_m = _exact_ising(4, 4, 2.5)
    result = _run_wolff_temperature(
        jp.IsingLattice((4, 4)), 2.5, 20000, 500, jax.random.PRNGKey(5)
    )
    assert result["energy"] == pytest.approx(exact_e, abs=0.03)
    assert result["magnetization"] == pytest.approx(exact_m, abs=0.03)


def test_sweep_temperatures_matches_single_temperature_runs() -> None:
    lattice = jp.IsingLattice((6, 6))
    temps = jnp.array([1.5, 3.0])
    key = jax.random.PRNGKey(11)
    sweep = jp.sweep_temperatures(lattice, temps, n_sweeps=200, n_warmup=50, key=key)
    for i, k in enumerate(jax.random.split(key, 2)):
        single = lattice.run_metropolis(float(temps[i]), 200, 50, k)
        assert float(sweep.energies[i]) == pytest.approx(single["energy"], abs=1e-12)


def test_metropolis_step_is_jittable() -> None:
    step = jax.jit(
        lambda s, k: metropolis_step(
            lambda x: jnp.sum(x**2), s, lambda x, kk: x + jax.random.normal(kk), 1.0, k
        )
    )
    _, _, accepted = step(jnp.array(0.5), jax.random.PRNGKey(0))
    assert accepted.dtype == jnp.bool_


def test_entropy_with_degeneracies_satisfies_thermodynamic_identity() -> None:
    energies, g, T = jnp.array([0.0, 1.0, 2.5]), jnp.array([1.0, 3.0, 5.0]), 1.3
    s = entropy(energies, T, g)
    assert s == pytest.approx(
        (mean_energy(energies, T, g) - free_energy(energies, T, g)) / T
    )


# ------------------------------------------------------------------ fluids


def test_lbm_no_slip_channel_develops_poiseuille_profile() -> None:
    nx, ny = 40, 21
    grid = jp.LBMGrid(size=(nx, ny), viscosity=0.1, boundary="no_slip")
    u = grid.simulate(n_steps=6000, u_inlet=0.02, save_every=6000).ux[-1, nx // 2]
    # Full-way bounce-back puts the walls half-way between nodes 0/1 and -2/-1.
    y = np.arange(ny)
    width = ny - 2
    profile = 4 * float(u.max()) * (y - 0.5) * (ny - 1.5 - y) / width**2
    assert float(jnp.max(jnp.abs(u[1:-1] - profile[1:-1]))) / float(u.max()) < 0.01
    free = jp.LBMGrid(size=(nx, ny), viscosity=0.1, boundary="free_slip")
    u_free = free.simulate(n_steps=2000, u_inlet=0.02, save_every=2000).ux[-1, nx // 2]
    np.testing.assert_allclose(u_free[1:-1], 0.02, rtol=1e-6)


def test_navier_stokes_vorticity_uses_grid_spacing() -> None:
    dx = 0.05
    solver = jp.NavierStokesSolver(size=(24, 24), viscosity=0.01, dx=dx)
    result = solver.simulate(n_steps=400, dt=0.005, lid_velocity=1.0, save_every=100)
    assert result.t.shape == (5,)
    assert float(result.t[0]) == 0.0
    u, v, w = result.ux[-1], result.uy[-1], result.vorticity[-1]
    curl = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx) - (u[1:-1, 2:] - u[1:-1, :-2]) / (
        2 * dx
    )
    inner = (slice(4, -4), slice(4, -4))
    np.testing.assert_allclose(curl[inner], w[1:-1, 1:-1][inner], rtol=0.05, atol=0.05)


# ---------------------------------------------------------------- optimize


def test_iteration_counts_agree_between_methods() -> None:
    for method in ("gradient_descent", "adam"):
        result = jp.optimize(lambda x: (x - 1.0) ** 2, 1.0, method=method)
        assert result.converged and result.n_iterations == 0
