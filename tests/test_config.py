"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from neurosim.config import (
    EMConfig,
    IsingConfig,
    NBodyConfig,
    Params,
    QuantumConfig,
    SimulationConfig,
)


class TestParams:
    """Tests for the generic Params container."""

    def test_basic_params(self) -> None:
        p = Params(m=1.0, g=9.81)
        assert p.m == 1.0
        assert p.g == 9.81

    def test_attribute_error(self) -> None:
        p = Params(m=1.0)
        with pytest.raises(AttributeError):
            _ = p.nonexistent

    def test_multiple_params(self) -> None:
        p = Params(m1=1.0, m2=2.0, l1=1.5, l2=0.8, g=9.81)
        assert p.m1 == 1.0
        assert p.l2 == 0.8


class TestSimulationConfig:
    """Tests for SimulationConfig validation."""

    def test_valid_config(self) -> None:
        cfg = SimulationConfig(t_end=10.0, dt=0.01)
        assert cfg.total_steps == 1000
        assert cfg.t_span == (0.0, 10.0)

    def test_invalid_t_end(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfig(t_end=-1.0)

    def test_invalid_dt(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfig(t_end=10.0, dt=-0.01)

    def test_n_steps_override(self) -> None:
        cfg = SimulationConfig(t_end=10.0, n_steps=500)
        assert cfg.total_steps == 500


class TestNBodyConfig:
    """Tests for NBodyConfig."""

    def test_defaults(self) -> None:
        cfg = NBodyConfig()
        assert cfg.G == 1.0
        assert cfg.softening == 1e-4

    def test_invalid_G(self) -> None:
        with pytest.raises(ValidationError):
            NBodyConfig(G=-1.0)


class TestEMConfig:
    """Tests for EMConfig."""

    def test_defaults(self) -> None:
        cfg = EMConfig()
        assert cfg.boundary == "absorbing"
        assert cfg.pml_layers == 10

    def test_invalid_boundary(self) -> None:
        with pytest.raises(ValidationError):
            EMConfig(boundary="invalid")


class TestQuantumConfig:
    """Tests for QuantumConfig."""

    def test_defaults(self) -> None:
        cfg = QuantumConfig()
        assert cfg.hbar == 1.0
        assert cfg.mass == 1.0
        assert cfg.n_points == 1000


class TestIsingConfig:
    """Tests for IsingConfig."""

    def test_defaults(self) -> None:
        cfg = IsingConfig()
        assert cfg.J == 1.0
        assert cfg.h == 0.0
        assert cfg.algorithm == "metropolis"
