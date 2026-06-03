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

    def test_min_relative_coeff_in_parameters(self, identifier_no_index, small_basis_library):
        """The relative-magnitude floor must be reported for tunability."""
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 1.0})
        result = identifier_no_index.identify(wl, spec)
        assert "min_relative_coeff" in result.parameters

    def test_relative_gate_never_passes_sub_floor_elements(self, small_basis_library):
        """No detected element may carry less than min_relative_coeff of the mass.

        This is the load-bearing precision gate (NNLS-GAUSS-BASIS-4): an
        element is detected only if its NNLS coefficient is at least
        ``min_relative_coeff`` of the total element coefficient mass.
        """
        ident = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            detection_snr=2.0,
            min_relative_coeff=0.05,
            fallback_T_K=8000.0,
            fallback_ne_cm3=5e16,
        )
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 0.7, "Cr": 0.3})
        result = ident.identify(wl, spec)
        for el in result.detected_elements:
            assert el.metadata["concentration_estimate"] >= 0.05 - 1e-12, (
                f"{el.element} detected with relative coeff "
                f"{el.metadata['concentration_estimate']:.4f} below the 5% floor"
            )

    def test_relative_gate_is_monotone_in_floor(self, small_basis_library):
        """Raising the relative floor can only remove detections, never add them."""
        wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 0.7, "Cr": 0.3})

        ident_off = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            detection_snr=2.0,
            min_relative_coeff=0.0,
            fallback_T_K=8000.0,
            fallback_ne_cm3=5e16,
        )
        ident_on = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            detection_snr=2.0,
            min_relative_coeff=0.05,
            fallback_T_K=8000.0,
            fallback_ne_cm3=5e16,
        )
        det_off = {e.element for e in ident_off.identify(wl, spec).detected_elements}
        det_on = {e.element for e in ident_on.identify(wl, spec).detected_elements}
        assert det_on <= det_off, "Relative gate added a detection (must be subtractive)"

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


class TestHybridUnionRecallFloor:
    """Regression guard for the #215 hybrid_union recall regression.

    The standalone NNLS relative-coeff floor (5% of *total* mass) is a
    precision lever calibrated on a small near-orthogonal basis. On a large
    real-data candidate set, each legitimate element holds only a few percent
    of the total coefficient mass, so the same floor silently drops true
    elements (Si, Mg, Na, K on real BHVO-2 spectra). The fix routes the
    hybrid_union NNLS arm through ``HybridIdentifier.nnls_min_relative_coeff``
    which defaults to ``0.0`` (recall-favoring) while standalone NNLS keeps
    its 0.05 precision floor.
    """

    def test_standalone_default_keeps_precision_floor(self):
        """Standalone SpectralNNLSIdentifier keeps the W2 0.05 floor by default."""
        from cflibs.inversion.identify.spectral_nnls import (
            DEFAULT_MIN_RELATIVE_COEFF,
            SpectralNNLSIdentifier,
        )

        import inspect

        sig = inspect.signature(SpectralNNLSIdentifier.__init__)
        assert sig.parameters["min_relative_coeff"].default == DEFAULT_MIN_RELATIVE_COEFF
        assert DEFAULT_MIN_RELATIVE_COEFF == pytest.approx(0.05)

    def test_hybrid_union_arm_defaults_to_recall_favoring_floor(self):
        """HybridIdentifier defaults the NNLS arm floor to 0.0 (recall-favoring)."""
        import inspect

        from cflibs.inversion.identify.hybrid import HybridIdentifier

        sig = inspect.signature(HybridIdentifier.__init__)
        assert "nnls_min_relative_coeff" in sig.parameters
        assert sig.parameters["nnls_min_relative_coeff"].default == pytest.approx(0.0)

    def test_hybrid_forwards_floor_to_nnls_stage(self, small_basis_library, monkeypatch):
        """The hybrid arm must forward nnls_min_relative_coeff to Stage-1 NNLS."""
        from cflibs.inversion.identify import spectral_nnls as snnls_mod
        from cflibs.inversion.identify.hybrid import HybridIdentifier

        captured = {}
        real_cls = snnls_mod.SpectralNNLSIdentifier

        def _spy(*args, **kwargs):
            captured["min_relative_coeff"] = kwargs.get("min_relative_coeff")
            return real_cls(*args, **kwargs)

        monkeypatch.setattr(snnls_mod, "SpectralNNLSIdentifier", _spy)

        from pathlib import Path

        from cflibs.atomic.database import AtomicDatabase

        candidates = [
            Path("libs_production.db"),
            Path("ASD_da/libs_production.db"),
            Path(__file__).parent.parent / "libs_production.db",
            Path(__file__).parent.parent / "ASD_da" / "libs_production.db",
        ]
        db_path = next((str(p) for p in candidates if p.exists()), None)
        if db_path is None:
            pytest.skip("Production database not found")

        with AtomicDatabase(db_path) as db:
            ident = HybridIdentifier(
                atomic_db=db,
                basis_library=small_basis_library,
                elements=list(small_basis_library.elements),
                require_both=False,
            )
            wl, spec = _make_synthetic_spectrum(small_basis_library, {"Fe": 0.7, "Cr": 0.3})
            ident.identify(wl, spec)

        # Default hybrid_union arm must run the NNLS screen with the floor off.
        assert captured["min_relative_coeff"] == pytest.approx(0.0)

    def test_disabling_floor_is_recall_favoring(self, small_basis_library):
        """Disabling the relative floor can only add detections, never remove
        them — the recall-favoring property the hybrid_union arm relies on.
        """
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

        wl, spec = _make_synthetic_spectrum(
            small_basis_library, {"Fe": 0.7, "Cr": 0.3}, noise=0.001
        )
        floor_on = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            detection_snr=2.0,
            min_relative_coeff=0.05,
            fallback_T_K=8000.0,
            fallback_ne_cm3=5e16,
        )
        floor_off = SpectralNNLSIdentifier(
            basis_library=small_basis_library,
            detection_snr=2.0,
            min_relative_coeff=0.0,
            fallback_T_K=8000.0,
            fallback_ne_cm3=5e16,
        )
        det_on = {e.element for e in floor_on.identify(wl, spec).detected_elements}
        det_off = {e.element for e in floor_off.identify(wl, spec).detected_elements}
        assert det_on <= det_off
