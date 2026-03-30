"""Tests for statistical mechanics modules."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.exceptions import ConfigurationError
from neurosim.statmech.boltzmann import (
    boltzmann_distribution,
    entropy,
    free_energy,
    mean_energy,
    partition_function,
)
from neurosim.statmech.ising import IsingLattice, sweep_temperatures

jax.config.update("jax_enable_x64", True)


class TestBoltzmann:
    """Tests for Boltzmann distribution utilities."""

    def test_partition_function(self) -> None:
        energies = jnp.array([0.0, 1.0, 2.0])
        Z = partition_function(energies, temperature=1.0)
        expected = 1.0 + jnp.exp(-1.0) + jnp.exp(-2.0)
        assert pytest.approx(float(expected), rel=1e-6) == Z

    def test_partition_function_energy_shift_invariant(self) -> None:
        energies = jnp.array([10.0, 11.0])
        Z = partition_function(energies, temperature=1.0)
        expected = jnp.exp(-10.0) + jnp.exp(-11.0)
        assert pytest.approx(float(expected), rel=1e-6) == Z

    def test_distribution_sums_to_one(self) -> None:
        energies = jnp.array([0.0, 1.0, 2.0, 3.0])
        probs = boltzmann_distribution(energies, temperature=2.0)
        assert float(jnp.sum(probs)) == pytest.approx(1.0, abs=1e-10)

    def test_ground_state_dominance(self) -> None:
        """At very low temperature, ground state should dominate."""
        energies = jnp.array([0.0, 1.0, 5.0])
        probs = boltzmann_distribution(energies, temperature=0.01)
        assert probs[0] > 0.99

    def test_high_temperature_equipartition(self) -> None:
        """At very high temperature, all states should be equally likely."""
        energies = jnp.array([0.0, 1.0, 2.0])
        probs = boltzmann_distribution(energies, temperature=1e6)
        assert probs[0] == pytest.approx(1.0 / 3.0, abs=1e-3)

    def test_with_degeneracies(self) -> None:
        energies = jnp.array([0.0, 1.0])
        degens = jnp.array([1.0, 3.0])
        probs = boltzmann_distribution(energies, 1.0, degens)
        assert float(jnp.sum(probs)) == pytest.approx(1.0, abs=1e-10)
        # Degenerate level should have higher probability
        assert probs[1] > probs[0] * jnp.exp(-1.0)  # Check ratio

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ConfigurationError):
            partition_function(jnp.array([0.0]), temperature=-1.0)

    def test_mean_energy(self) -> None:
        energies = jnp.array([0.0, 1.0])
        E = mean_energy(energies, temperature=1e10)
        assert pytest.approx(0.5, abs=0.01) == E

    def test_free_energy(self) -> None:
        energies = jnp.array([0.0])
        F = free_energy(energies, temperature=1.0)
        # F = -T * ln(Z) = -1 * ln(1) = 0
        assert pytest.approx(0.0, abs=1e-6) == F

    def test_free_energy_energy_shift_invariant(self) -> None:
        energies = jnp.array([10.0, 11.0])
        F = free_energy(energies, temperature=1.0)
        expected = 10.0 - jnp.log1p(jnp.exp(-1.0))
        assert pytest.approx(float(expected), rel=1e-6) == F

    def test_entropy(self) -> None:
        # Two equal-energy states: S = ln(2)
        energies = jnp.array([0.0, 0.0])
        S = entropy(energies, temperature=1.0)
        assert pytest.approx(float(jnp.log(2.0)), abs=1e-6) == S


class TestIsingLattice:
    """Tests for the Ising model."""

    def test_random_state(self) -> None:
        lattice = IsingLattice(size=(16, 16))
        key = jax.random.PRNGKey(0)
        spins = lattice.random_state(key)
        assert spins.shape == (16, 16)
        # All values should be +1 or -1
        assert jnp.all((spins == 1) | (spins == -1))

    def test_low_temperature_order(self) -> None:
        """At low T, magnetization should be near 1."""
        lattice = IsingLattice(size=(8, 8), J=1.0)
        result = lattice.run_metropolis(
            temperature=0.5,
            n_sweeps=2000,
            n_warmup=5000,
            key=jax.random.PRNGKey(42),
        )
        assert result["magnetization"] > 0.5

    def test_high_temperature_disorder(self) -> None:
        """At high T, magnetization should be near 0."""
        lattice = IsingLattice(size=(8, 8), J=1.0)
        result = lattice.run_metropolis(
            temperature=10.0,
            n_sweeps=500,
            n_warmup=500,
            key=jax.random.PRNGKey(42),
        )
        assert result["magnetization"] < 0.5

    def test_small_lattice_error(self) -> None:
        with pytest.raises(ConfigurationError):
            IsingLattice(size=(1, 1))

    def test_wolff_cluster_sweep_uses_cluster_updates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wolff sweeps should not route through Metropolis."""
        lattice = IsingLattice(size=(4, 4), J=1.0)

        def fail_run_metropolis(*args: object, **kwargs: object) -> dict[str, float]:
            raise AssertionError("run_metropolis should not be called")

        monkeypatch.setattr(IsingLattice, "run_metropolis", fail_run_metropolis)

        result = sweep_temperatures(
            lattice,
            jnp.array([2.0, 2.5]),
            n_sweeps=4,
            n_warmup=2,
            algorithm="wolff_cluster",
            key=jax.random.PRNGKey(0),
        )

        assert result.temperatures.shape == (2,)
        assert result.energies.shape == (2,)
        assert result.magnetizations.shape == (2,)
