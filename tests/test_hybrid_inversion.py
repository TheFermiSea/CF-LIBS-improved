"""
Tests for hybrid inversion (manifold + gradient descent).

These tests validate the two-stage inversion approach:
1. Coarse search: Manifold nearest-neighbor
2. Fine tuning: JAX autodiff + L-BFGS optimization

Requirements: JAX, h5py
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Mark entire module as requiring JAX
pytestmark = pytest.mark.requires_jax

# Skip all tests if JAX is not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

h5py = pytest.importorskip("h5py")

from cflibs.inversion.hybrid import (  # noqa: E402
    HybridInverter,
    HybridInversionResult,
    SpectralFitter,
)


@pytest.fixture
def mock_manifold_file():
    """Create a minimal mock manifold HDF5 file."""
    import os

    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)  # Close file descriptor to prevent leaks

    with h5py.File(path, "w") as f:
        # Create wavelength grid
        wavelength = np.linspace(300, 600, 100)
        f.create_dataset("wavelength", data=wavelength)

        # Create synthetic spectra at different conditions
        n_spectra = 20
        elements = ["Fe", "Cu"]

        spectra = []
        params = []

        for i in range(n_spectra):
            # Vary T and n_e
            T_eV = 0.5 + 2.0 * i / n_spectra  # 0.5 to 2.5 eV
            n_e = 1e16 * (10 ** (2.0 * i / n_spectra))  # 1e16 to 1e18

            # Concentration depends on index
            c_Fe = 0.3 + 0.5 * (i % 5) / 4.0
            c_Cu = 1.0 - c_Fe

            # Generate simple spectrum (sum of Gaussians)
            spectrum = np.zeros(len(wavelength))
            for j, (el, c, center) in enumerate([("Fe", c_Fe, 400), ("Cu", c_Cu, 500)]):
                # Intensity depends on concentration and temperature
                intensity = c * 1000 * np.exp(-3.0 / T_eV)
                sigma = 10.0 * np.sqrt(T_eV)
                profile = np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
                spectrum += intensity * profile

            spectra.append(spectrum)
            params.append([T_eV, n_e, c_Fe, c_Cu])

        f.create_dataset("spectra", data=np.array(spectra))
        f.create_dataset("params", data=np.array(params))

        # Metadata
        f.attrs["elements"] = elements
        f.attrs["wavelength_range"] = [300.0, 600.0]
        f.attrs["temperature_range"] = [0.5, 2.5]
        f.attrs["density_range"] = [1e16, 1e18]

    yield path

    Path(path).unlink()


@pytest.fixture
def mock_manifold(mock_manifold_file):
    """Load mock manifold."""
    from cflibs.manifold.loader import ManifoldLoader

    return ManifoldLoader(mock_manifold_file)


@pytest.mark.requires_jax
class TestHybridInversionResult:
    """Tests for the HybridInversionResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = HybridInversionResult(
            temperature_eV=1.2,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "Cu": 0.3},
            coarse_temperature_eV=1.0,
            coarse_electron_density_cm3=1e17,
            coarse_concentrations={"Fe": 0.6, "Cu": 0.4},
            coarse_similarity=0.95,
            final_residual=100.0,
            converged=True,
            iterations=15,
        )

        assert result.temperature_eV == 1.2
        assert result.electron_density_cm3 == 1e17
        assert result.concentrations["Fe"] == 0.7
        assert result.converged

    def test_temperature_K_property(self):
        """Test temperature Kelvin property."""
        result = HybridInversionResult(
            temperature_eV=1.0,  # ~11604 K
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            coarse_temperature_eV=1.0,
            coarse_electron_density_cm3=1e17,
            coarse_concentrations={"Fe": 1.0},
            coarse_similarity=0.9,
            final_residual=0.0,
            converged=True,
            iterations=0,
        )

        # 1 eV ≈ 11604 K
        np.testing.assert_allclose(result.temperature_K, 11604.5, rtol=0.01)

    def test_summary(self):
        """Test result summary generation."""
        result = HybridInversionResult(
            temperature_eV=1.2,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "Cu": 0.3},
            coarse_temperature_eV=1.0,
            coarse_electron_density_cm3=1e17,
            coarse_concentrations={"Fe": 0.6, "Cu": 0.4},
            coarse_similarity=0.95,
            final_residual=100.0,
            converged=True,
            iterations=15,
        )

        summary = result.summary()
        assert "Hybrid Inversion Result" in summary
        assert "Coarse" in summary
        assert "Fine" in summary
        assert "Fe" in summary


@pytest.mark.requires_jax
class TestHybridInverter:
    """Tests for the HybridInverter class."""

    def test_init(self, mock_manifold):
        """Test HybridInverter initialization."""
        inverter = HybridInverter(mock_manifold)

        assert inverter.n_elements == 2
        assert "Fe" in inverter.elements
        assert "Cu" in inverter.elements
        assert len(inverter.wavelength) == 100

    def test_invert_with_manifold_init(self, mock_manifold):
        """Test inversion with manifold initialization."""
        inverter = HybridInverter(mock_manifold)

        # Create a test spectrum similar to manifold data
        wavelength = mock_manifold.wavelength
        T_true = 1.5
        c_Fe_true = 0.6

        spectrum = np.zeros(len(wavelength))
        for c, center in [(c_Fe_true, 400), (1.0 - c_Fe_true, 500)]:
            intensity = c * 1000 * np.exp(-3.0 / T_true)
            sigma = 10.0 * np.sqrt(T_true)
            profile = np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
            spectrum += intensity * profile

        result = inverter.invert(spectrum, use_manifold_init=True)

        # Should return valid result
        assert result.temperature_eV > 0
        assert result.electron_density_cm3 > 0
        assert sum(result.concentrations.values()) > 0.99

        # Coarse similarity should be reasonable
        assert result.coarse_similarity > 0.5

    def test_default_lbfgsb_path_performs_real_optimization(self, mock_manifold):
        """Default hybrid inversion should do real fine tuning instead of falling back."""
        wavelength = mock_manifold.wavelength

        def forward_model(T_eV, n_e, conc, wl):
            center = 380.0 + 40.0 * conc[0]
            sigma = 6.0 + 2.0 * T_eV
            amplitude = 600.0 * conc[0] + 250.0 * conc[1]
            return amplitude * jnp.exp(-0.5 * ((wl - center) / sigma) ** 2)

        inverter = HybridInverter(mock_manifold, forward_model=forward_model, max_iterations=30)
        measured = np.array(
            forward_model(1.4, 1e17, jnp.array([0.75, 0.25]), jnp.array(wavelength))
        )

        result = inverter.invert(
            measured,
            use_manifold_init=False,
            initial_guess={"T_eV": 0.8, "n_e": 5e16, "Fe": 0.45, "Cu": 0.55},
        )

        assert result.iterations > 0
        assert result.metadata["optimizer_backend"] in {"jax", "scipy"}

    def test_invert_without_manifold_init(self, mock_manifold):
        """Test inversion with default initialization (no manifold)."""
        inverter = HybridInverter(mock_manifold)

        # Create a test spectrum
        wavelength = mock_manifold.wavelength
        spectrum = np.exp(-((wavelength - 450) ** 2) / 1000)

        result = inverter.invert(
            spectrum,
            use_manifold_init=False,
            initial_guess={"T_eV": 1.0, "n_e": 1e17},
        )

        assert result.temperature_eV > 0
        assert result.coarse_similarity == 0.0  # No manifold lookup

    def test_invert_with_uncertainties(self, mock_manifold):
        """Test inversion with explicit uncertainties."""
        inverter = HybridInverter(mock_manifold)

        wavelength = mock_manifold.wavelength
        spectrum = np.exp(-((wavelength - 450) ** 2) / 1000) * 100 + 10
        uncertainties = np.sqrt(spectrum)  # Poisson-like

        result = inverter.invert(
            spectrum,
            uncertainties=uncertainties,
            use_manifold_init=True,
        )

        assert result.temperature_eV > 0

    def test_pack_unpack_params(self, mock_manifold):
        """Test parameter packing and unpacking."""
        inverter = HybridInverter(mock_manifold)

        T_eV = 1.5
        n_e = 5e17
        conc = {"Fe": 0.7, "Cu": 0.3}

        packed = inverter._pack_params(T_eV, n_e, conc)
        T_out, ne_out, conc_out = inverter._unpack_params(packed)

        # Temperature and density should round-trip
        np.testing.assert_allclose(float(T_out), T_eV, rtol=0.01)
        np.testing.assert_allclose(float(ne_out), n_e, rtol=0.01)

        # Concentrations should sum to 1 (softmax)
        np.testing.assert_allclose(float(jnp.sum(conc_out)), 1.0, rtol=0.01)


@pytest.mark.requires_jax
class TestSpectralFitter:
    """Tests for the SpectralFitter class."""

    def test_init(self):
        """Test SpectralFitter initialization."""

        def forward_model(T, ne, conc, wl):
            return jnp.zeros_like(wl)

        wavelength = np.linspace(300, 600, 100)
        fitter = SpectralFitter(forward_model, ["Fe", "Cu"], wavelength)

        assert fitter.n_elements == 2
        assert len(fitter.wavelength) == 100

    def test_fit_simple(self):
        """Test fitting with simple forward model."""
        wavelength = np.linspace(300, 600, 100)

        # Simple forward model: single Gaussian peak
        def forward_model(T, ne, conc, wl):
            center = 400 + 100 * conc[0]  # Peak position depends on Fe conc
            sigma = 10 * jnp.sqrt(T)
            return 1000 * conc[0] * jnp.exp(-0.5 * ((wl - center) / sigma) ** 2)

        fitter = SpectralFitter(forward_model, ["Fe", "Cu"], wavelength)

        # Generate synthetic data
        T_true = 1.5
        c_Fe_true = 0.7
        true_spectrum = np.array(
            forward_model(T_true, 1e17, jnp.array([c_Fe_true, 0.3]), jnp.array(wavelength))
        )

        # Add noise
        true_spectrum += np.random.normal(0, 10, len(wavelength))
        true_spectrum = np.maximum(true_spectrum, 1.0)

        result = fitter.fit(
            true_spectrum,
            initial_T_eV=1.0,
            initial_n_e=1e17,
            initial_concentrations={"Fe": 0.5, "Cu": 0.5},
        )

        # Should return valid result
        assert result.temperature_eV > 0
        assert result.electron_density_cm3 > 0

    def test_fit_with_uncertainties(self):
        """Test fitting with explicit uncertainties."""
        wavelength = np.linspace(300, 600, 100)

        def forward_model(T, ne, conc, wl):
            return 100 * jnp.exp(-((wl - 450) ** 2) / 1000)

        fitter = SpectralFitter(forward_model, ["Fe", "Cu"], wavelength)

        spectrum = np.array(forward_model(1.0, 1e17, jnp.array([0.5, 0.5]), jnp.array(wavelength)))
        uncertainties = np.sqrt(np.maximum(spectrum, 1.0))

        result = fitter.fit(spectrum, uncertainties=uncertainties)

        assert result.temperature_eV > 0

    def test_fit_supports_lbfgsb_via_fallback_backend(self):
        """SpectralFitter should support L-BFGS-B through the fallback backend."""
        wavelength = np.linspace(300, 600, 100)

        def forward_model(T, ne, conc, wl):
            center = 420.0 + 20.0 * conc[0]
            sigma = 8.0 + T
            return 400.0 * conc[0] * jnp.exp(-0.5 * ((wl - center) / sigma) ** 2)

        fitter = SpectralFitter(forward_model, ["Fe", "Cu"], wavelength)
        spectrum = np.array(forward_model(1.2, 1e17, jnp.array([0.7, 0.3]), jnp.array(wavelength)))

        result = fitter.fit(
            spectrum,
            initial_T_eV=0.9,
            initial_n_e=8e16,
            initial_concentrations={"Fe": 0.5, "Cu": 0.5},
            method="L-BFGS-B",
            max_iterations=25,
        )

        assert result.iterations > 0
        assert result.metadata["optimizer_backend"] == "scipy"


@pytest.mark.requires_jax
class TestHybridInverterEdgeCases:
    """Edge case tests for HybridInverter."""

    def test_spectrum_length_mismatch(self, mock_manifold):
        """Test error handling for mismatched spectrum length."""
        inverter = HybridInverter(mock_manifold)

        # Spectrum with wrong length
        bad_spectrum = np.zeros(50)  # Manifold has 100 points

        with pytest.raises(ValueError, match="does not match"):
            inverter.invert(bad_spectrum)

    def test_zero_spectrum(self, mock_manifold):
        """Test handling of zero/very small spectrum."""
        inverter = HybridInverter(mock_manifold)

        # Near-zero spectrum
        spectrum = np.ones(100) * 1e-10

        # Should not crash, may not converge well
        result = inverter.invert(spectrum)
        assert result.temperature_eV > 0

    def test_very_noisy_spectrum(self, mock_manifold):
        """Test handling of very noisy spectrum."""
        inverter = HybridInverter(mock_manifold)

        # Pure noise
        rng = np.random.default_rng(42)
        spectrum = rng.normal(100, 50, 100)
        spectrum = np.maximum(spectrum, 1.0)

        # Should return some result, may not converge
        result = inverter.invert(spectrum)
        assert result.temperature_eV > 0
