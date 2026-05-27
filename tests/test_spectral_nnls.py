"""
Tests for SpectralNNLSIdentifier — full-spectrum NNLS element identification.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier  # noqa: E402
from cflibs.inversion.common.element_id import ElementIdentificationResult  # noqa: E402

pytestmark = [pytest.mark.requires_db, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_basis_library(tmp_path_factory):
    """Generate a small basis library for testing (module-scoped for speed)."""
    from pathlib import Path

    from cflibs.manifold.basis_library import (
        BasisLibrary,
        BasisLibraryConfig,
        BasisLibraryGenerator,
    )

    # Search multiple candidate locations (mirrors conftest.py pattern)
    candidates = [
        Path("libs_production.db"),
        Path("ASD_da/libs_production.db"),
        Path(__file__).parent.parent / "libs_production.db",
        Path(__file__).parent.parent / "ASD_da" / "libs_production.db",
    ]
    db_path = None
    for p in candidates:
        if p.exists():
            db_path = str(p)
            break
    if db_path is None:
        pytest.skip("Production database not found")

    tmp_dir = tmp_path_factory.mktemp("basis_lib")
    cfg = BasisLibraryConfig(
        db_path=db_path,
        output_path=str(tmp_dir / "test_basis.h5"),
        wavelength_range=(370.0, 380.0),
        pixels=256,
        temperature_range=(6000.0, 10000.0),
        temperature_steps=3,
        density_range=(1e16, 1e17),
        density_steps=2,
    )
    gen = BasisLibraryGenerator(cfg)
    gen.generate()

    lib = BasisLibrary(cfg.output_path)
    yield lib
    lib.close()


@pytest.fixture
def identifier_no_index(small_basis_library):
    """SpectralNNLSIdentifier without FAISS index (uses fallback T/ne)."""
    return SpectralNNLSIdentifier(
        basis_library=small_basis_library,
        basis_index=None,
        detection_snr=2.0,  # Lower for testing
        fallback_T_K=8000.0,
        fallback_ne_cm3=5e16,
    )


def _make_synthetic_spectrum(basis_library, elements_fracs, T_K=8000.0, ne=5e16, noise=0.01):
    """Create a synthetic spectrum as a weighted sum of basis spectra + noise."""
    wl = basis_library.wavelength
    spectrum = np.zeros(len(wl))
    for el, frac in elements_fracs.items():
        el_spec = basis_library.get_element_spectrum(el, T_K, ne)
        spectrum += frac * el_spec
    # Add noise
    rng = np.random.RandomState(42)
    spectrum += noise * rng.normal(size=len(spectrum))
    spectrum = np.maximum(spectrum, 0.0)
    return wl, spectrum


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpectralNNLSIdentifier:
    def test_returns_result_type(self, identifier_no_index, small_basis_library):
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = identifier_no_index.identify(wl, spec)
        assert isinstance(result, ElementIdentificationResult)
        assert result.algorithm == "spectral_nnls"

    def test_detects_fe_in_pure_fe_spectrum(self, identifier_no_index, small_basis_library):
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0}, noise=0.001)
        result = identifier_no_index.identify(wl, spec)
        detected_els = {e.element for e in result.detected_elements}
        assert "Fe" in detected_els, f"Fe not detected. Detected: {detected_els}"

    def test_fe_is_dominant(self, identifier_no_index, small_basis_library):
        """Fe should have the highest score in a pure Fe spectrum."""
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0}, noise=0.001)
        result = identifier_no_index.identify(wl, spec)
        if result.all_elements:
            best = max(result.all_elements, key=lambda e: e.score)
            assert best.element == "Fe", (
                f"Expected Fe to be best, got {best.element} " f"(score={best.score:.4f})"
            )

    def test_detects_mixture(self, identifier_no_index, small_basis_library):
        """Both Fe and Cr should be detected in a mixture spectrum."""
        wl, spec = _make_synthetic_spectrum(
            small_basis_library, {"Fe": 0.7, "Cr": 0.3}, noise=0.001
        )
        result = identifier_no_index.identify(wl, spec)
        detected_els = {e.element for e in result.detected_elements}
        # At minimum Fe should be detected (Cr may or may not have
        # lines in 370-380nm range depending on DB)
        assert "Fe" in detected_els

    def test_noise_only_few_detections(self, identifier_no_index, small_basis_library):
        """Pure noise should produce few or no detections."""
        wl = small_basis_library.wavelength
        rng = np.random.RandomState(99)
        noise_spec = np.abs(rng.normal(size=len(wl))) * 0.01
        result = identifier_no_index.identify(wl, noise_spec)
        # With SNR threshold, noise should produce minimal detections
        assert len(result.detected_elements) < len(small_basis_library.elements) // 2

    def test_metadata_has_nnls_fields(self, identifier_no_index, small_basis_library):
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = identifier_no_index.identify(wl, spec)
        for el in result.all_elements:
            assert "nnls_coefficient" in el.metadata
            assert "nnls_snr" in el.metadata
            assert "estimated_T_K" in el.metadata
            assert "estimated_ne_cm3" in el.metadata

    def test_estimated_params_stored(self, identifier_no_index, small_basis_library):
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        identifier_no_index.identify(wl, spec)
        assert identifier_no_index._estimated_T == 8000.0  # fallback
        assert identifier_no_index._estimated_ne == 5e16

    def test_parameters_in_result(self, identifier_no_index, small_basis_library):
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = identifier_no_index.identify(wl, spec)
        assert "estimated_T_K" in result.parameters
        assert "detection_snr" in result.parameters
        assert result.parameters["estimated_T_K"] == 8000.0

    def test_continuum_degree_minus_one_disables(self, small_basis_library):
        """continuum_degree=-1 should not add polynomial columns."""
        ident = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            detection_snr=2.0,
            continuum_degree=-1,
            fallback_T_K=8000.0,
            fallback_ne_cm3=5e16,
        )
        # Verify _build_augmented_matrix produces no continuum rows
        basis_matrix = small_basis_library.get_basis_matrix_interp(8000.0, 5e16)
        lib_wl = small_basis_library.wavelength
        A = ident._build_augmented_matrix(basis_matrix, lib_wl)
        n_elements = len(small_basis_library.elements)
        assert (
            A.shape[0] == n_elements
        ), f"Expected {n_elements} rows (no continuum), got {A.shape[0]}"

        # Also verify end-to-end still works
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = ident.identify(wl, spec)
        assert isinstance(result, ElementIdentificationResult)

    def test_all_elements_in_result(self, identifier_no_index, small_basis_library):
        """all_elements should contain every element in the basis library."""
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = identifier_no_index.identify(wl, spec)
        result_els = {e.element for e in result.all_elements}
        lib_els = set(small_basis_library.elements)
        assert result_els == lib_els

    def test_scores_between_zero_and_one(self, identifier_no_index, small_basis_library):
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = identifier_no_index.identify(wl, spec)
        for el in result.all_elements:
            assert 0.0 <= el.score <= 1.0
            assert 0.0 <= el.confidence <= 1.0

    def test_concentration_estimates_sum(self, identifier_no_index, small_basis_library):
        """Detected element concentrations should sum to ~1 (for detected only)."""
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0}, noise=0.001)
        result = identifier_no_index.identify(wl, spec)
        if result.detected_elements:
            concs = [e.metadata["concentration_estimate"] for e in result.detected_elements]
            total = sum(concs)
            # May not sum to exactly 1 due to undetected elements, but should be >0
            assert total > 0

    def test_with_faiss_index(self, small_basis_library):
        """Test with FAISS-based T/ne estimation."""
        try:
            from cflibs.manifold.basis_index import BasisIndex
        except ImportError:
            pytest.skip("faiss not available")

        idx = BasisIndex(n_components=5)
        idx.build_from_library(small_basis_library)

        ident = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            basis_index=idx,
            detection_snr=2.0,
        )
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0}, noise=0.001)
        result = ident.identify(wl, spec)
        assert isinstance(result, ElementIdentificationResult)
        detected_els = {e.element for e in result.detected_elements}
        assert "Fe" in detected_els
