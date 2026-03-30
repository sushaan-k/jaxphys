"""Tests for optics modules."""

import jax
import jax.numpy as jnp
import pytest

from neurosim.exceptions import ConfigurationError
from neurosim.optics.diffraction import (
    circular_aperture,
    double_slit,
    single_slit,
)
from neurosim.optics.ray_tracing import (
    FlatMirror,
    Ray,
    SphericalMirror,
    ThinLens,
    trace_system,
)

jax.config.update("jax_enable_x64", True)


class TestRayTracing:
    """Tests for geometric optics ray tracing."""

    def test_thin_lens_focusing(self) -> None:
        """Parallel ray through a thin lens should focus at f."""
        ray = Ray(y=0.5, theta=0.0)
        lens = ThinLens(f=0.1, position=0.0)
        result = trace_system(ray, [lens])

        # After lens, theta should be -y/f
        assert result.angles[-1] == pytest.approx(-0.5 / 0.1, abs=1e-10)

    def test_thin_lens_matrix(self) -> None:
        lens = ThinLens(f=0.2)
        M = lens.matrix()
        assert M[0, 0] == pytest.approx(1.0)
        assert M[1, 0] == pytest.approx(-1.0 / 0.2)

    def test_flat_mirror(self) -> None:
        mirror = FlatMirror()
        M = mirror.matrix()
        assert jnp.allclose(M, jnp.eye(2))

    def test_spherical_mirror(self) -> None:
        mirror = SphericalMirror(R=0.5)
        M = mirror.matrix()
        assert M[1, 0] == pytest.approx(-2.0 / 0.5)

    def test_two_lens_system(self) -> None:
        """Two thin lenses: verify ray trace returns expected heights."""
        ray = Ray(y=1.0, theta=0.0)
        elements = [
            ThinLens(f=0.5, position=0.0),
            ThinLens(f=0.3, position=1.0),
        ]
        result = trace_system(ray, elements)

        # Should have initial + 2 element positions = 3 entries
        assert len(result.heights) == 3
        assert len(result.angles) == 3

    def test_no_elements_error(self) -> None:
        ray = Ray(y=1.0, theta=0.0)
        with pytest.raises(ConfigurationError):
            trace_system(ray, [])


class TestDiffraction:
    """Tests for wave optics diffraction."""

    def test_single_slit_peak(self) -> None:
        """Single slit should have maximum at theta=0."""
        result = single_slit(
            slit_width=1e-4,
            wavelength=500e-9,
            n_points=1000,
        )
        # Maximum should be at the center (theta=0)
        center_idx = result.intensity.shape[0] // 2
        assert result.intensity[center_idx] == pytest.approx(1.0, abs=0.01)

    def test_single_slit_symmetry(self) -> None:
        """Pattern should be symmetric about theta=0."""
        result = single_slit(
            slit_width=1e-4,
            wavelength=500e-9,
            n_points=1001,
        )
        n = result.intensity.shape[0]
        left = result.intensity[: n // 2]
        right = jnp.flip(result.intensity[n // 2 + 1 :])
        assert jnp.allclose(left, right, atol=1e-10)

    def test_double_slit(self) -> None:
        """Double slit pattern should be modulated by interference."""
        result = double_slit(
            slit_width=5e-5,
            slit_separation=2e-4,
            wavelength=500e-9,
        )
        # Should have multiple peaks
        assert result.intensity.shape[0] == 1000

    def test_circular_aperture(self) -> None:
        """Airy pattern should have central maximum = 1."""
        result = circular_aperture(
            diameter=1e-3,
            wavelength=500e-9,
        )
        center = result.intensity.shape[0] // 2
        assert result.intensity[center] == pytest.approx(1.0, abs=0.05)

    def test_invalid_params(self) -> None:
        with pytest.raises(ConfigurationError):
            single_slit(slit_width=-1, wavelength=500e-9)
