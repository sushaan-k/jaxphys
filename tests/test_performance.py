"""Performance regression guards.

These tests do not time anything (timings are noisy in CI). They check the
structural property the benchmarks in ``benchmarks/`` rely on: a repeated
call with the same shapes, or with new parameter *values*, reuses the
compiled loop instead of triggering another XLA compilation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import monitoring

import jaxphys as jp
from jaxphys.em.fdtd import C0

_COMPILES = [0]


def _listener(event: str, duration: float, **_: Any) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES[0] += 1


monitoring.register_event_duration_secs_listener(_listener)


@contextmanager
def count_compiles() -> Iterator[list[int]]:
    """Yield a one-element list that holds the number of XLA compilations."""
    start = _COMPILES[0]
    box = [0]
    try:
        yield box
    finally:
        box[0] = _COMPILES[0] - start


def assert_no_recompile(*calls: Callable[[], Any]) -> None:
    """Warm up with the first call; the others must not compile anything."""
    jax.block_until_ready(jax.tree_util.tree_leaves(calls[0]()))
    for call in calls[1:] or calls:
        with count_compiles() as n:
            jax.block_until_ready(jax.tree_util.tree_leaves(call()))
        assert n[0] == 0, f"repeated call triggered {n[0]} XLA compilation(s)"


def _oscillator(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
    return p[0] ** 2 / (2 * params.m) + 0.5 * params.k * q[0] ** 2


# ------------------------------------------------------------- classical


def test_hamiltonian_new_param_values_do_not_recompile() -> None:
    system = jp.HamiltonianSystem(_oscillator, n_dof=1)

    def run(k: float, q0: float = 1.0) -> Any:
        params = jp.Params(m=1.0, k=k)
        return system.simulate([q0], [0.0], (0, 1), 0.01, params, save_every=10).q

    assert_no_recompile(lambda: run(4.0), lambda: run(9.0), lambda: run(2.0, 0.5))


def test_hamiltonian_falls_back_when_params_must_be_concrete() -> None:
    def branching(q: jax.Array, p: jax.Array, params: Any) -> jax.Array:
        # Python control flow on a parameter value needs a concrete float.
        k = params.k if params.k > 0 else 1.0
        return 0.5 * p[0] ** 2 + 0.5 * k * q[0] ** 2

    kwargs: dict[str, Any] = dict(q0=[1.0], p0=[0.0], t_span=(0.0, 1.0), dt=0.01)
    system = jp.HamiltonianSystem(branching, n_dof=1)
    a = system.simulate(params=jp.Params(k=4.0), **kwargs)
    b = jp.HamiltonianSystem(_oscillator, n_dof=1).simulate(
        params=jp.Params(m=1.0, k=4.0), **kwargs
    )
    np.testing.assert_allclose(np.asarray(a.q), np.asarray(b.q), rtol=1e-12)
    # Such params are compiled in as constants, cached per value.
    assert_no_recompile(lambda: system.simulate(params=jp.Params(k=4.0), **kwargs).q)
    # Non-numeric parameter values are closed over as constants.
    c = jp.HamiltonianSystem(_oscillator, n_dof=1).simulate(
        params=jp.Params(m=1.0, k=4.0, label="spring"), **kwargs
    )
    np.testing.assert_allclose(np.asarray(c.q), np.asarray(b.q), rtol=1e-12)


def test_lagrangian_new_param_values_do_not_recompile() -> None:
    def lagrangian(q: jax.Array, qdot: jax.Array, params: Any) -> jax.Array:
        return 0.5 * params.m * qdot[0] ** 2 - 0.5 * params.k * q[0] ** 2

    system = jp.LagrangianSystem(lagrangian, n_dof=1)

    def run(k: float) -> Any:
        params = jp.Params(m=1.0, k=k)
        return system.simulate([1.0], [0.0], (0, 1), 0.01, params, save_every=10).q

    assert_no_recompile(lambda: run(4.0), lambda: run(9.0))


def test_nbody_new_systems_do_not_recompile() -> None:
    pos = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    a = jp.NBody(jnp.ones(5), pos, jnp.zeros((5, 3)), softening=0.1)
    b = jp.NBody(2 * jnp.ones(5), pos + 1, jnp.ones((5, 3)), G=2.0, softening=0.2)
    assert_no_recompile(
        lambda: a.simulate(t_span=(0.0, 0.1), n_steps=40, save_every=7).positions,
        lambda: b.simulate(t_span=(0.0, 0.3), n_steps=40, save_every=7).positions,
    )


def test_nbody_force_kernel_matches_reference_formula() -> None:
    from jaxphys.classical.nbody import gravitational_accelerations

    pos = jax.random.normal(jax.random.PRNGKey(2), (33, 3))
    masses = jax.random.uniform(jax.random.PRNGKey(3), (33,)) + 0.1
    got = gravitational_accelerations(pos, masses, 1.3, 0.05)
    dr = np.asarray(pos)[None, :, :] - np.asarray(pos)[:, None, :]
    inv = (np.sum(dr**2, axis=-1) + 0.05**2) ** -1.5
    np.fill_diagonal(inv, 0.0)
    want = 1.3 * np.einsum("j,ijk,ij->ik", np.asarray(masses), dr, inv)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-15)


def test_rigid_bodies_share_one_compiled_loop() -> None:
    a = jp.RigidBody(inertia=[1.0, 2.0, 3.0])
    b = jp.RigidBody(inertia=[1.5, 2.0, 2.5])
    assert_no_recompile(
        lambda: a.simulate(omega0=[1.0, 0.1, 0.0], t_span=(0.0, 1.0), dt=0.01).q,
        lambda: a.simulate(omega0=[0.5, 0.1, 0.0], t_span=(0.0, 1.0), dt=0.01).q,
        lambda: b.simulate(omega0=[1.0, 0.1, 0.0], t_span=(0.0, 1.0), dt=0.01).q,
    )


def test_charge_systems_share_one_compiled_loop() -> None:
    def system(q: float, bz: float) -> jp.ChargeSystem:
        return jp.ChargeSystem(
            [
                jp.PointCharge(q, 1e-3, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                jp.PointCharge(-1e-6, 1e-3, [0.1, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ],
            B_external=jnp.array([0.0, 0.0, bz]),
        )

    a, b = system(1e-6, 1.0), system(2e-6, 0.5)
    assert_no_recompile(
        lambda: a.simulate(t_span=(0.0, 1e-4), n_steps=50, save_every=7).positions,
        lambda: b.simulate(t_span=(0.0, 2e-4), n_steps=50, save_every=7).positions,
    )


def test_optimize_reuses_the_compiled_gradient() -> None:
    def objective(v: jax.Array) -> jax.Array:
        return (2.0 * v - 3.0) ** 2

    assert_no_recompile(
        lambda: jp.optimize(objective, 1.0, learning_rate=0.1, max_iterations=5).x,
        lambda: jp.optimize(objective, 5.0, learning_rate=0.1, max_iterations=5).x,
    )


# ---------------------------------------------------------------- fields


def test_fdtd2d_new_sources_do_not_recompile() -> None:
    def grid(frequency: float, y: int) -> jp.EMGrid:
        g = jp.EMGrid(size=(24, 20), resolution=0.01)
        g.add_source(jp.PlaneWave(frequency=frequency, y=y))
        return g

    a, b = grid(3e9, 4), grid(4e9, 6)
    assert_no_recompile(
        lambda: a.simulate(t_span=(0.0, 5e-10), save_every=4).ez,
        lambda: b.simulate(t_span=(0.0, 5e-10), save_every=4).ez,
    )


def test_fdtd3d_new_sources_do_not_recompile() -> None:
    def grid(frequency: float, x: int) -> jp.EMGrid3D:
        g = jp.EMGrid3D(size=(10, 10, 10), resolution=0.01, pml_layers=2)
        g.add_source(jp.PointSource3D(frequency=frequency, position=(x, 5, 5)))
        return g

    a, b = grid(3e9, 5), grid(4e9, 4)
    assert_no_recompile(
        lambda: a.simulate(t_span=(0.0, 2e-10), save_every=3).ez,
        lambda: b.simulate(t_span=(0.0, 2e-10), save_every=3).ez,
    )


def test_fdfd_new_permittivity_and_frequency_do_not_recompile() -> None:
    n, dx = 40, C0 / 3e9 / 20
    source = jnp.zeros((n, n)).at[20, 20].set(1.0)
    eps_a, eps_b = jnp.ones((n, n)), 2.0 * jnp.ones((n, n))
    assert_no_recompile(
        lambda: jp.solve_fdfd(eps_a, source, 3e9, dx, 8).ez,
        lambda: jp.solve_fdfd(eps_b, source, 3.2e9, dx, 8).ez,
    )


# ---------------------------------------------------------------- fluids


@pytest.mark.parametrize("boundary", ["periodic", "no_slip", "free_slip"])
def test_lbm_new_viscosity_and_inlet_do_not_recompile(boundary: str) -> None:
    a = jp.LBMGrid(size=(16, 8), viscosity=0.05, boundary=boundary)  # type: ignore[arg-type]
    b = jp.LBMGrid(size=(16, 8), viscosity=0.08, boundary=boundary)  # type: ignore[arg-type]
    assert_no_recompile(
        lambda: a.simulate(n_steps=20, u_inlet=0.04, save_every=5).ux,
        lambda: b.simulate(n_steps=20, u_inlet=0.05, save_every=5).ux,
    )


def test_navier_stokes_new_parameters_do_not_recompile() -> None:
    a = jp.NavierStokesSolver(size=(12, 12), viscosity=0.01)
    b = jp.NavierStokesSolver(size=(12, 12), viscosity=0.02)
    assert_no_recompile(
        lambda: a.simulate(n_steps=10, dt=0.01, poisson_iters=5, save_every=3).ux,
        lambda: (
            b.simulate(
                n_steps=10, dt=0.02, lid_velocity=0.5, poisson_iters=5, save_every=3
            ).ux
        ),
    )


def test_sph_fluids_with_new_constants_do_not_recompile() -> None:
    n = 12
    dx = 1.0 / n
    xs = jnp.arange(n) * dx + dx / 2
    pos = jnp.stack(jnp.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    vel_a, vel_b = jnp.zeros_like(pos), jnp.full(pos.shape, 0.01)

    def fluid(sound_speed: float, alpha: float) -> jp.SPHFluid:
        return jp.SPHFluid(
            mass=1000 * dx**2,
            smoothing_length=1.3 * dx,
            box=(1.0, 1.0),
            sound_speed=sound_speed,
            alpha=alpha,
        )

    a, b = fluid(10.0, 0.1), fluid(12.0, 0.0)
    assert_no_recompile(
        lambda: a.simulate(pos, vel_a, (0.0, 0.002), 0.001, 1).positions,
        lambda: b.simulate(pos, vel_b, (0.0, 0.002), 0.001, 1).positions,
    )


def test_euler_new_states_and_gamma_do_not_recompile() -> None:
    x = (jnp.arange(64) + 0.5) / 64
    rho = jnp.where(x < 0.5, 1.0, 0.125)
    p_a, p_b = jnp.where(x < 0.5, 1.0, 0.1), jnp.where(x < 0.5, 2.0, 0.1)
    u = jnp.zeros(64)

    def run(p: jax.Array, gamma: float) -> Any:
        return jp.solve_euler_1d(
            rho, u, p, dx=1 / 64, t_end=0.02, gamma=gamma, dt=0.002, save_every=5
        ).rho

    assert_no_recompile(lambda: run(p_a, 1.4), lambda: run(p_b, 5.0 / 3.0))


# --------------------------------------------------------------- quantum


def test_schrodinger_new_potential_does_not_recompile() -> None:
    def run(potential: Any) -> Any:
        return jp.solve_schrodinger(
            jp.GaussianWavepacket(x0=-3.0, k0=2.0, sigma=0.5),
            potential,
            n_points=128,
            t_span=(0.0, 0.5),
            dt=0.01,
            save_every=10,
        ).psi

    assert_no_recompile(
        lambda: run(jp.HarmonicPotential(k=1.0)),
        lambda: run(jp.HarmonicPotential(k=2.0)),
    )


def test_schrodinger_2d_new_initial_state_does_not_recompile() -> None:
    def run(kx: float) -> Any:
        return jp.solve_schrodinger_2d(
            jp.GaussianWavepacket2D(0.0, 0.0, kx, 0.0, 1.0),
            lambda X, Y: 0.5 * (X**2 + Y**2),
            n_points=(32, 32),
            t_span=(0.0, 0.1),
            dt=0.01,
            save_every=5,
        ).psi

    assert_no_recompile(lambda: run(1.0), lambda: run(2.0))


def test_lindblad_new_rates_do_not_recompile() -> None:
    rho0 = jp.DensityMatrix.from_pure_state(jnp.array([1.0, 0.0], dtype=complex))
    h = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    lop = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    def run(rate: float) -> Any:
        return jp.lindblad_evolve(
            rho0, h, [lop], [rate], t_span=(0.0, 1.0), dt=0.01, save_every=10
        ).rho

    assert_no_recompile(lambda: run(0.1), lambda: run(0.3))


def test_tight_binding_new_hoppings_do_not_recompile() -> None:
    k = jnp.linspace(-np.pi, np.pi, 11)[:, None]
    assert_no_recompile(
        lambda: jp.TightBinding.chain(t=1.0).bands(k),
        lambda: jp.TightBinding.chain(t=1.3, onsite=0.2).bands(k),
    )


# --------------------------------------------------------------- statmech


def test_wolff_step_compiles_once_across_temperatures() -> None:
    from jaxphys.statmech.monte_carlo import wolff_step

    spins = jnp.ones((8, 8), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    assert_no_recompile(
        lambda: wolff_step(spins, 2.0, 1.0, key),
        lambda: wolff_step(spins, 1.5, 1.0, key),
        lambda: wolff_step(spins, 3.0, 1.0, key),
    )


def test_metropolis_new_temperature_does_not_recompile() -> None:
    lattice = jp.IsingLattice(size=(6, 6))
    assert_no_recompile(
        lambda: lattice.run_metropolis(2.0, n_sweeps=5, n_warmup=2),
        lambda: lattice.run_metropolis(3.0, n_sweeps=5, n_warmup=2),
    )


@pytest.mark.parametrize("algorithm", ["metropolis", "wolff_cluster"])
def test_sweep_temperatures_does_not_recompile(algorithm: str) -> None:
    lattice = jp.IsingLattice(size=(6, 6))

    def run(temperatures: list[float]) -> Any:
        return jp.sweep_temperatures(
            lattice,
            jnp.array(temperatures),
            n_sweeps=5,
            n_warmup=2,
            algorithm=algorithm,
        ).energies

    assert_no_recompile(lambda: run([1.0, 1.1, 1.2]), lambda: run([2.0, 2.5, 3.0]))
