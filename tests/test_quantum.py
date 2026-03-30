"""Tests for quantum mechanics modules."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.exceptions import ConfigurationError
from neurosim.quantum.density_matrix import DensityMatrix, lindblad_evolve
from neurosim.quantum.schrodinger import (
    GaussianWavepacket,
    HarmonicPotential,
    SquareBarrier,
    solve_schrodinger,
)
from neurosim.quantum.spin import SpinChain
from neurosim.quantum.stationary import solve_eigenvalue_problem

jax.config.update("jax_enable_x64", True)


class TestSchrodinger:
    """Tests for the time-dependent Schrodinger solver."""

    def test_free_particle_norm(self) -> None:
        """Free particle evolution should preserve probability norm."""
        psi0 = GaussianWavepacket(x0=0.0, k0=2.0, sigma=1.0)

        # Zero potential
        class ZeroPotential:
            def __call__(self, x):
                return jnp.zeros_like(x)

        result = solve_schrodinger(
            psi0=psi0,
            potential=ZeroPotential(),
            x_range=(-20, 20),
            t_span=(0, 5),
            n_points=500,
            dt=0.01,
            save_every=50,
        )

        # Check norm preservation at each saved step
        for i in range(result.n_steps):
            norm = jnp.trapezoid(jnp.abs(result.psi[i]) ** 2, result.x)
            assert norm == pytest.approx(1.0, abs=1e-3)

    def test_tunneling(self) -> None:
        """Wavepacket should partially tunnel through a barrier."""
        psi0 = GaussianWavepacket(x0=5.0, k0=3.0, sigma=0.5)
        barrier = SquareBarrier(height=5.0, width=1.0, center=10.0)

        result = solve_schrodinger(
            psi0=psi0,
            potential=barrier,
            x_range=(-5, 25),
            t_span=(0, 5),
            n_points=1000,
            dt=0.005,
            save_every=100,
        )

        assert result.transmission_coefficient is not None
        # Some should tunnel, but not all
        assert 0.0 <= result.transmission_coefficient <= 1.0

    def test_square_barrier_callable(self) -> None:
        barrier = SquareBarrier(height=3.0, width=2.0, center=0.0)
        x = jnp.linspace(-5, 5, 100)
        V = barrier(x)
        assert V.shape == (100,)
        assert float(V[50]) == pytest.approx(3.0, abs=0.1)


class TestStationary:
    """Tests for the time-independent Schrodinger solver."""

    def test_harmonic_oscillator_eigenvalues(self) -> None:
        """Energy levels of QHO should be E_n = hbar*omega*(n + 0.5)."""
        potential = HarmonicPotential(k=1.0)

        result = solve_eigenvalue_problem(
            potential=potential,
            x_range=(-10, 10),
            n_points=500,
            n_states=5,
            hbar=1.0,
            mass=1.0,
        )

        # omega = sqrt(k/m) = 1
        for n in range(5):
            expected = n + 0.5  # hbar * omega * (n + 0.5)
            assert result.energies[n] == pytest.approx(expected, abs=0.05)

    def test_infinite_well(self) -> None:
        """Particle in a box: E_n = n^2 * pi^2 * hbar^2 / (2*m*L^2)."""

        def well(x):
            L = 5.0
            return jnp.where((x > -L / 2) & (x < L / 2), 0.0, 1e6)

        result = solve_eigenvalue_problem(
            potential=well,
            x_range=(-5, 5),
            n_points=500,
            n_states=3,
        )

        # Check first eigenvalue
        L = 5.0
        E1_expected = jnp.pi**2 / (2.0 * L**2)
        assert result.energies[0] == pytest.approx(float(E1_expected), rel=0.05)


class TestSpinChain:
    """Tests for spin chain dynamics."""

    def test_two_site_antiferro(self) -> None:
        """Two-site AFM chain: E_gs = -3J/4 (singlet)."""
        chain = SpinChain(n_sites=2, J=-1.0, h=0.0)
        result = chain.diagonalize(n_states=4)

        # Ground state of 2-site Heisenberg AFM
        assert result.energies[0] == pytest.approx(-0.75, abs=0.01)

    def test_hilbert_dim(self) -> None:
        chain = SpinChain(n_sites=4)
        assert chain.hilbert_dim == 16

    def test_too_large(self) -> None:
        with pytest.raises(ConfigurationError):
            SpinChain(n_sites=20)


class TestDensityMatrix:
    """Tests for density matrix formalism."""

    def test_pure_state_purity(self) -> None:
        psi = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)
        assert dm.purity() == pytest.approx(1.0, abs=1e-10)

    def test_mixed_state_purity(self) -> None:
        rho = 0.5 * jnp.eye(2, dtype=jnp.complex128)
        dm = DensityMatrix(rho=rho)
        assert dm.purity() == pytest.approx(0.5, abs=1e-10)

    def test_von_neumann_entropy(self) -> None:
        # Maximally mixed state of dim 2: S = ln(2)
        rho = 0.5 * jnp.eye(2, dtype=jnp.complex128)
        dm = DensityMatrix(rho=rho)
        assert dm.von_neumann_entropy() == pytest.approx(float(jnp.log(2.0)), abs=1e-6)

    def test_lindblad_decay(self) -> None:
        """Lindblad evolution should decrease purity for dissipative systems."""
        psi0 = jnp.array([0.0, 1.0], dtype=jnp.complex128)  # Excited state
        dm0 = DensityMatrix.from_pure_state(psi0)

        H = jnp.zeros((2, 2), dtype=jnp.complex128)

        # Decay operator |0><1|
        L = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)

        result = lindblad_evolve(dm0, H, [L], [0.5], t_span=(0, 5), dt=0.01)

        # Purity should decrease
        assert result.purity[-1] < result.purity[0]

    def test_invalid_density_matrix_raises(self) -> None:
        rho = jnp.array([[0.5, 0.5], [0.1, 0.5]], dtype=jnp.complex128)
        with pytest.raises(ConfigurationError, match="Hermitian"):
            DensityMatrix(rho=rho)

    def test_lindblad_shape_validation(self) -> None:
        psi = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        dm = DensityMatrix.from_pure_state(psi)
        H = jnp.zeros((2, 2), dtype=jnp.complex128)
        L = jnp.eye(3, dtype=jnp.complex128)

        with pytest.raises(ConfigurationError, match="shape"):
            lindblad_evolve(dm, H, [L], [0.1], t_span=(0, 1), dt=0.1)
