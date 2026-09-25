"""Tests for the FDFD solver, tight-binding models and the 2D Schrodinger solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxphys as jp
from jaxphys.em.fdtd import C0, MU0
from jaxphys.exceptions import ConfigurationError

# -------------------------------------------------------------------- FDFD


def _point_source_field(n: int, points_per_wavelength: int = 30):
    frequency = 3e9
    dx = C0 / frequency / points_per_wavelength
    c = n // 2
    eps = jnp.ones((n, n))
    source = jnp.zeros((n, n)).at[c, c].set(1.0 / dx**2)  # unit line current
    return jp.solve_fdfd(eps, source, frequency, dx, pml_layers=12), dx, c


def test_fdfd_line_source_matches_hankel_green_function() -> None:
    special = pytest.importorskip("scipy.special")
    result, dx, c = _point_source_field(101)
    k = 2 * np.pi * result.frequency / C0
    omega = 2 * np.pi * result.frequency
    offsets = np.arange(5, 35)
    for field, r in (
        (result.ez[c + offsets, c], offsets * dx),
        (result.ez[c + offsets, c + offsets], np.sqrt(2) * offsets * dx),
    ):
        exact = -(omega * MU0 / 4) * special.hankel1(0, k * r)
        error = np.max(np.abs(np.asarray(field) - exact)) / np.max(np.abs(exact))
        assert error < 0.02


def test_fdfd_gradient_wrt_permittivity() -> None:
    n, frequency = 61, 3e9
    dx = C0 / frequency / 20
    source = jnp.zeros((n, n)).at[30, 30].set(1.0)

    def probe(eps):
        return jnp.abs(jp.solve_fdfd(eps, source, frequency, dx, 10).ez[40, 30]) ** 2

    eps = jnp.ones((n, n))
    grad = jax.grad(probe)(eps)
    bump = jnp.zeros((n, n)).at[35, 32].set(1e-4)
    fd = (probe(eps + bump) - probe(eps - bump)) / 2e-4
    assert float(grad[35, 32]) == pytest.approx(float(fd), rel=1e-4)
    # Batched solves over several permittivity maps.
    batch = jax.vmap(lambda e: jp.solve_fdfd(e, source, frequency, dx, 10).ez)(
        jnp.stack([eps, 2.0 * eps])
    )
    assert batch.shape == (2, n, n)


def test_fdfd_rectangular_grid_and_validation() -> None:
    frequency = 3e9
    dx = C0 / frequency / 20
    source = jnp.zeros((50, 70)).at[25, 35].set(1.0)
    wide = jp.solve_fdfd(jnp.ones((50, 70)), source, frequency, dx, 10).ez
    tall = jp.solve_fdfd(jnp.ones((70, 50)), source.T, frequency, dx, 10).ez
    np.testing.assert_allclose(wide, tall.T, rtol=1e-8, atol=1e-12)
    with pytest.raises(ConfigurationError):
        jp.solve_fdfd(jnp.ones((20, 20)), jnp.zeros((20, 20)), frequency, dx, 10)


# ------------------------------------------------------------ tight binding


def test_tight_binding_bands_match_analytic_dispersions() -> None:
    k = jnp.linspace(-np.pi, np.pi, 41)
    chain = jp.TightBinding.chain(t=1.3, onsite=0.2)
    np.testing.assert_allclose(chain.bands(k[:, None])[:, 0], 0.2 - 2.6 * np.cos(k))

    kx, ky = np.meshgrid(k, k[::3], indexing="ij")
    kk = jnp.stack([kx.ravel(), ky.ravel()], -1)
    square = jp.TightBinding.square_lattice(t=0.5)
    np.testing.assert_allclose(
        square.bands(kk)[:, 0], -(np.cos(kk[:, 0]) + np.cos(kk[:, 1])), atol=1e-12
    )

    graphene = jp.TightBinding.honeycomb(t=2.7)
    a1, a2 = np.asarray(graphene.lattice_vectors)
    f = 1 + np.exp(-1j * kk @ a1) + np.exp(-1j * kk @ a2)
    expected = np.stack([-2.7 * np.abs(f), 2.7 * np.abs(f)], -1)
    np.testing.assert_allclose(graphene.bands(kk), expected, atol=1e-10)
    # Dirac point at the zone corner K; bandwidth 3t at Gamma.
    corner = (2 * graphene.reciprocal_vectors[0] + graphene.reciprocal_vectors[1]) / 3
    np.testing.assert_allclose(graphene.bands(corner), 0.0, atol=1e-12)
    np.testing.assert_allclose(graphene.bands(jnp.zeros(2)), [[-8.1, 8.1]])


def test_finite_chain_spectrum() -> None:
    chain = jp.TightBinding.chain(t=1.0)
    n = 12
    open_levels = jnp.linalg.eigvalsh(chain.finite_hamiltonian((n,)))
    exact = -2 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))
    np.testing.assert_allclose(open_levels, np.sort(exact), atol=1e-12)
    ring = jnp.linalg.eigvalsh(chain.finite_hamiltonian((n,), periodic=True))
    np.testing.assert_allclose(
        ring, np.sort(-2 * np.cos(2 * np.pi * np.arange(n) / n)), atol=1e-12
    )
    # A periodic honeycomb flake reproduces the Bloch bands on its k-grid.
    graphene = jp.TightBinding.honeycomb(t=1.0)
    flake = jnp.linalg.eigvalsh(graphene.finite_hamiltonian((4, 4), periodic=True))
    m = np.arange(4) / 4
    ks = np.stack(np.meshgrid(m, m, indexing="ij"), -1).reshape(-1, 2)
    ks = ks @ np.asarray(graphene.reciprocal_vectors)
    np.testing.assert_allclose(flake, np.sort(np.ravel(graphene.bands(ks))), atol=1e-10)


def test_tight_binding_is_a_differentiable_pytree() -> None:
    chain = jp.TightBinding.chain(t=1.0)

    def energy(model, k):
        return model.bands(jnp.array([[k]]))[0, 0]

    grad = jax.grad(energy)(chain, 0.3)
    # E = 2 * amplitude * cos(k), amplitude = -t.
    assert float(grad.amplitudes[0]) == pytest.approx(2 * np.cos(0.3))
    assert float(grad.onsite[0]) == pytest.approx(1.0)
    jitted = jax.jit(lambda model: model.bands(jnp.zeros((3, 1))))(chain)
    assert jitted.shape == (3, 1)
    k, distance = jp.k_path([[0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi]], 10)
    assert k.shape == (21, 2)
    assert float(distance[-1]) == pytest.approx(2 * np.pi)
    with pytest.raises(ConfigurationError):
        jp.TightBinding.from_hoppings([[1.0]], [0.0], [(0, 0, (0,), 1.0)])


# ------------------------------------------------------------- Schrodinger 2D


def test_free_packet_2d_moves_and_spreads_analytically() -> None:
    packet = jp.GaussianWavepacket2D(x0=-3.0, y0=1.0, kx=1.5, ky=-0.5, sigma=0.7)
    result = jp.solve_schrodinger_2d(
        packet,
        lambda x, y: 0.0 * x,
        x_range=(-15, 15),
        y_range=(-15, 15),
        n_points=(96, 96),
        t_span=(0, 2),
        dt=0.01,
        save_every=50,
    )
    X, Y = jnp.meshgrid(result.x, result.y, indexing="ij")
    cell = (30 / 96) ** 2
    for t, prob in zip(np.asarray(result.t), result.probability, strict=True):
        norm = float(jnp.sum(prob) * cell)
        mean_x = float(jnp.sum(prob * X) * cell)
        mean_y = float(jnp.sum(prob * Y) * cell)
        var_x = float(jnp.sum(prob * X**2) * cell) - mean_x**2
        assert norm == pytest.approx(1.0, abs=1e-12)
        assert mean_x == pytest.approx(-3.0 + 1.5 * t, abs=1e-8)
        assert mean_y == pytest.approx(1.0 - 0.5 * t, abs=1e-8)
        # sigma(t)^2 = sigma^2 (1 + (t / (2 sigma^2))^2) for hbar = m = 1.
        assert var_x == pytest.approx(0.49 * (1 + (t / 0.98) ** 2), rel=1e-6)


def test_separable_2d_problem_matches_product_of_1d_solutions() -> None:
    x = jnp.linspace(-8, 8, 64, endpoint=False)
    kwargs = {"t_span": (0, 1), "dt": 0.01, "save_every": 100}
    result = jp.solve_schrodinger_2d(
        lambda X, Y: jnp.exp(-((X - 1) ** 2) / 2 - (Y + 0.5) ** 2),
        lambda X, Y: X**2 + 0.5 * Y**2,
        x_range=(-8, 8),
        y_range=(-8, 8),
        n_points=(64, 64),
        **kwargs,
    )
    along_x = jp.solve_schrodinger(
        jnp.exp(-((x - 1) ** 2) / 2),
        jp.HarmonicPotential(k=2.0),
        x_range=(-8, 8),
        n_points=64,
        **kwargs,
    )
    along_y = jp.solve_schrodinger(
        jnp.exp(-((x + 0.5) ** 2)),
        jp.HarmonicPotential(k=1.0),
        x_range=(-8, 8),
        n_points=64,
        **kwargs,
    )
    product = along_x.psi[-1][:, None] * along_y.psi[-1][None, :]
    np.testing.assert_allclose(result.psi[-1], product, atol=1e-12)
    with pytest.raises(ConfigurationError):
        jp.solve_schrodinger_2d(jnp.ones((4, 4)), jnp.zeros((4, 4)), n_points=(8, 8))
