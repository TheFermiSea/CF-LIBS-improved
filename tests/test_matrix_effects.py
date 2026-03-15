"""
Tests for matrix effect correction framework.

Tests cover:
- MatrixType enum
- MatrixEffectCorrector classification and correction
- InternalStandardizer normalization
- CorrectionFactorDB storage and retrieval
- Integration with CFLIBSResult
"""

import json
import tempfile
from pathlib import Path

import pytest

from cflibs.inversion.matrix_effects import (
    MatrixType,
    CorrectionFactor,
    CorrectionFactorDB,
    MatrixEffectCorrector,
    InternalStandardizer,
    combine_corrections,
)
from cflibs.inversion.solver import CFLIBSResult

# ============================================================================
# MatrixType Enum Tests
# ============================================================================


class TestMatrixType:
    """Tests for MatrixType enumeration."""

    def test_all_types_defined(self):
        """Verify all expected matrix types exist."""
        expected = {"METALLIC", "OXIDE", "ORGANIC", "GEOLOGICAL", "GLASS", "LIQUID", "UNKNOWN"}
        actual = {mt.name for mt in MatrixType}
        assert actual == expected

    def test_types_are_unique(self):
        """Verify each type has a unique value."""
        values = [mt.value for mt in MatrixType]
        assert len(values) == len(set(values))

    def test_type_from_name(self):
        """Test lookup by name."""
        assert MatrixType["METALLIC"] == MatrixType.METALLIC
        assert MatrixType["ORGANIC"] == MatrixType.ORGANIC

    def test_invalid_type_raises(self):
        """Test invalid name raises KeyError."""
        with pytest.raises(KeyError):
            _ = MatrixType["INVALID"]


# ============================================================================
# CorrectionFactor Tests
# ============================================================================


class TestCorrectionFactor:
    """Tests for CorrectionFactor dataclass."""

    def test_default_values(self):
        """Test default correction factor (no correction)."""
        factor = CorrectionFactor(element="Fe", matrix_type=MatrixType.METALLIC)
        assert factor.multiplicative == 1.0
        assert factor.additive == 0.0
        assert factor.uncertainty == 0.1

    def test_apply_multiplicative_only(self):
        """Test multiplicative correction."""
        factor = CorrectionFactor(
            element="Fe",
            matrix_type=MatrixType.METALLIC,
            multiplicative=1.10,
            uncertainty=0.05,
        )
        corrected, uncert = factor.apply(0.50)
        assert corrected == pytest.approx(0.55, rel=1e-6)
        assert uncert == pytest.approx(0.55 * 0.05, rel=1e-6)

    def test_apply_with_additive(self):
        """Test combined multiplicative and additive correction."""
        factor = CorrectionFactor(
            element="C",
            matrix_type=MatrixType.ORGANIC,
            multiplicative=0.80,
            additive=0.02,
            uncertainty=0.10,
        )
        # 0.30 * 0.80 + 0.02 = 0.26
        corrected, uncert = factor.apply(0.30)
        assert corrected == pytest.approx(0.26, rel=1e-6)
        assert uncert == pytest.approx(0.026, rel=1e-6)

    def test_apply_zero_concentration(self):
        """Test correction with zero input."""
        factor = CorrectionFactor(
            element="Fe",
            matrix_type=MatrixType.METALLIC,
            multiplicative=1.10,
            additive=0.01,
        )
        corrected, uncert = factor.apply(0.0)
        assert corrected == pytest.approx(0.01, rel=1e-6)

    def test_valid_range(self):
        """Test valid range attribute."""
        factor = CorrectionFactor(
            element="Si",
            matrix_type=MatrixType.OXIDE,
            valid_range=(0.10, 0.50),
        )
        assert factor.valid_range == (0.10, 0.50)


# ============================================================================
# CorrectionFactorDB Tests
# ============================================================================


class TestCorrectionFactorDB:
    """Tests for CorrectionFactorDB class."""

    def test_init_populates_defaults(self):
        """Test default factors are populated."""
        db = CorrectionFactorDB()

        # Check some expected defaults exist
        assert db.get_factor("Fe", MatrixType.METALLIC) is not None
        assert db.get_factor("Si", MatrixType.OXIDE) is not None
        assert db.get_factor("C", MatrixType.ORGANIC) is not None

    def test_get_nonexistent_factor(self):
        """Test getting a factor that doesn't exist."""
        db = CorrectionFactorDB()
        factor = db.get_factor("Unobtainium", MatrixType.METALLIC)
        assert factor is None

    def test_add_factor(self):
        """Test adding a custom factor."""
        db = CorrectionFactorDB()
        custom = CorrectionFactor(
            element="Au",
            matrix_type=MatrixType.METALLIC,
            multiplicative=1.03,
            source="custom_calibration",
        )
        db.add_factor(custom)

        retrieved = db.get_factor("Au", MatrixType.METALLIC)
        assert retrieved is not None
        assert retrieved.multiplicative == 1.03
        assert retrieved.source == "custom_calibration"

    def test_add_factor_overwrites_existing(self):
        """Test that adding overwrites existing factor."""
        db = CorrectionFactorDB()

        # Get original
        original = db.get_factor("Fe", MatrixType.METALLIC)
        original_mult = original.multiplicative

        # Add new with different value
        new = CorrectionFactor(
            element="Fe",
            matrix_type=MatrixType.METALLIC,
            multiplicative=1.25,
        )
        db.add_factor(new)

        retrieved = db.get_factor("Fe", MatrixType.METALLIC)
        assert retrieved.multiplicative == 1.25
        assert retrieved.multiplicative != original_mult

    def test_get_factors_for_matrix(self):
        """Test getting all factors for a matrix type."""
        db = CorrectionFactorDB()
        factors = db.get_factors_for_matrix(MatrixType.METALLIC)

        assert isinstance(factors, dict)
        assert "Fe" in factors
        assert "Cr" in factors

    def test_elements_property(self):
        """Test elements property returns all elements."""
        db = CorrectionFactorDB()
        elements = db.elements

        assert isinstance(elements, set)
        assert "Fe" in elements
        assert "Si" in elements
        assert "C" in elements

    def test_save_and_load_json(self):
        """Test JSON serialization round-trip."""
        db = CorrectionFactorDB()

        # Add a custom factor
        custom = CorrectionFactor(
            element="Pt",
            matrix_type=MatrixType.METALLIC,
            multiplicative=0.98,
            additive=0.001,
            uncertainty=0.03,
            source="test",
            valid_range=(0.0, 0.10),
        )
        db.add_factor(custom)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            db.save_to_json(filepath)

            # Load into new database
            db2 = CorrectionFactorDB()
            db2.load_from_json(filepath)

            # Check custom factor was loaded
            loaded = db2.get_factor("Pt", MatrixType.METALLIC)
            assert loaded is not None
            assert loaded.multiplicative == pytest.approx(0.98)
            assert loaded.additive == pytest.approx(0.001)
            assert loaded.uncertainty == pytest.approx(0.03)
            assert loaded.valid_range == (0.0, 0.10)
        finally:
            filepath.unlink()

    def test_load_json_unknown_matrix_type(self):
        """Test loading JSON with unknown matrix type logs warning."""
        data = {"NONEXISTENT_TYPE": {"Fe": {"multiplicative": 1.0}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            filepath = Path(f.name)

        try:
            db = CorrectionFactorDB()
            db.load_from_json(filepath)  # Should not raise, just warn
        finally:
            filepath.unlink()


# ============================================================================
# MatrixEffectCorrector Classification Tests
# ============================================================================


class TestMatrixClassification:
    """Tests for matrix type classification."""

    def test_classify_metallic(self):
        """Test classification of metallic alloy."""
        corrector = MatrixEffectCorrector()
        concentrations = {
            "Fe": 0.70,
            "Cr": 0.18,
            "Ni": 0.08,
            "Mn": 0.02,
            "Si": 0.01,
            "C": 0.01,
        }
        result = corrector.classify(concentrations)

        assert result.matrix_type == MatrixType.METALLIC
        assert result.confidence >= 0.5

    def test_classify_organic(self):
        """Test classification of organic material."""
        corrector = MatrixEffectCorrector()
        concentrations = {
            "C": 0.45,
            "H": 0.06,
            "O": 0.40,
            "N": 0.05,
            "Ca": 0.02,
            "K": 0.01,
            "P": 0.01,
        }
        result = corrector.classify(concentrations)

        assert result.matrix_type == MatrixType.ORGANIC
        assert result.confidence >= 0.5

    def test_classify_geological(self):
        """Test classification of geological sample."""
        corrector = MatrixEffectCorrector()
        concentrations = {
            "Si": 0.25,
            "O": 0.45,
            "Al": 0.08,
            "Fe": 0.05,
            "Ca": 0.04,
            "Mg": 0.03,
            "Na": 0.02,
            "K": 0.02,
            "Ti": 0.01,
        }
        result = corrector.classify(concentrations)

        assert result.matrix_type == MatrixType.GEOLOGICAL
        assert result.confidence >= 0.5

    def test_classify_oxide(self):
        """Test classification of metal oxide."""
        corrector = MatrixEffectCorrector()
        concentrations = {
            "Fe": 0.70,
            "O": 0.28,
            "Si": 0.02,
        }
        result = corrector.classify(concentrations)

        assert result.matrix_type == MatrixType.OXIDE

    def test_classify_glass(self):
        """Test classification of glass."""
        corrector = MatrixEffectCorrector()
        concentrations = {
            "Si": 0.35,
            "O": 0.50,
            "Na": 0.08,
            "Ca": 0.02,
        }
        result = corrector.classify(concentrations)

        # Glass or oxide are both reasonable
        assert result.matrix_type in (MatrixType.GLASS, MatrixType.OXIDE)

    def test_classify_known_type_override(self):
        """Test that known_type parameter overrides classification."""
        corrector = MatrixEffectCorrector()
        concentrations = {"Fe": 0.70, "Cr": 0.20, "Ni": 0.10}

        result = corrector.classify(concentrations, known_type=MatrixType.ORGANIC)

        assert result.matrix_type == MatrixType.ORGANIC
        assert result.confidence == 1.0
        assert "user" in result.notes.lower()

    def test_classify_empty_concentrations(self):
        """Test classification with empty concentrations."""
        corrector = MatrixEffectCorrector()
        result = corrector.classify({})

        assert result.matrix_type == MatrixType.UNKNOWN
        assert result.confidence == 0.0

    def test_classify_zero_total(self):
        """Test classification when all concentrations are zero."""
        corrector = MatrixEffectCorrector()
        result = corrector.classify({"Fe": 0.0, "Cr": 0.0})

        assert result.matrix_type == MatrixType.UNKNOWN


# ============================================================================
# MatrixEffectCorrector Correction Tests
# ============================================================================


class TestMatrixCorrection:
    """Tests for matrix effect correction application."""

    @pytest.fixture
    def steel_result(self) -> CFLIBSResult:
        """Create a typical steel CF-LIBS result."""
        return CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={
                "Fe": 0.70,
                "Cr": 0.18,
                "Ni": 0.08,
                "Mn": 0.02,
                "Si": 0.015,
                "C": 0.005,
            },
            concentration_uncertainties={},
            iterations=5,
            converged=True,
        )

    def test_correct_metallic_sample(self, steel_result):
        """Test correction of metallic sample."""
        corrector = MatrixEffectCorrector()
        result = corrector.correct(steel_result)

        assert result.matrix_type == MatrixType.METALLIC
        assert len(result.corrected_concentrations) == len(result.original_concentrations)
        # Corrections should be applied
        assert "Fe" in result.factors_applied

    def test_correct_with_explicit_type(self, steel_result):
        """Test correction with explicitly specified matrix type."""
        corrector = MatrixEffectCorrector()
        result = corrector.correct(steel_result, matrix_type=MatrixType.OXIDE)

        assert result.matrix_type == MatrixType.OXIDE

    def test_correct_unknown_type_no_correction(self):
        """Test that explicitly unknown type applies no corrections."""
        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"X": 0.5, "Y": 0.3, "Z": 0.2},  # Unknown elements
            concentration_uncertainties={},
            iterations=1,
            converged=True,
        )

        corrector = MatrixEffectCorrector()
        # Explicitly specify UNKNOWN type to bypass classification
        corrected = corrector.correct(result, matrix_type=MatrixType.UNKNOWN)

        assert corrected.matrix_type == MatrixType.UNKNOWN
        assert len(corrected.factors_applied) == 0
        # Concentrations unchanged
        assert corrected.corrected_concentrations == corrected.original_concentrations

    def test_renormalization(self, steel_result):
        """Test that concentrations are renormalized by default."""
        corrector = MatrixEffectCorrector(renormalize=True)
        result = corrector.correct(steel_result)

        total = sum(result.corrected_concentrations.values())
        assert total == pytest.approx(1.0, rel=1e-6)
        assert result.renormalized is True

    def test_no_renormalization(self, steel_result):
        """Test that renormalization can be disabled."""
        corrector = MatrixEffectCorrector(renormalize=False)
        result = corrector.correct(steel_result)

        # Sum may not be exactly 1.0
        assert result.renormalized is False

    def test_correct_concentrations_method(self):
        """Test the convenience method for raw concentrations."""
        corrector = MatrixEffectCorrector()
        concentrations = {"Fe": 0.65, "Cr": 0.20, "Ni": 0.10, "Mn": 0.05}

        result = corrector.correct_concentrations(concentrations)

        assert result.matrix_type == MatrixType.METALLIC
        assert "Fe" in result.corrected_concentrations

    def test_custom_correction_db(self, steel_result):
        """Test using a custom correction database."""
        custom_db = CorrectionFactorDB()
        # Override Fe factor
        custom_db.add_factor(
            CorrectionFactor(
                element="Fe",
                matrix_type=MatrixType.METALLIC,
                multiplicative=1.50,  # Large correction for testing
                uncertainty=0.01,
            )
        )

        corrector = MatrixEffectCorrector(correction_db=custom_db)
        result = corrector.correct(steel_result)

        # Check custom factor was used
        assert result.factors_applied["Fe"].multiplicative == 1.50

    def test_correction_uncertainties_propagated(self, steel_result):
        """Test that correction uncertainties are calculated."""
        corrector = MatrixEffectCorrector()
        result = corrector.correct(steel_result)

        # Should have uncertainties for corrected elements
        for element in result.factors_applied:
            assert element in result.correction_uncertainties
            assert result.correction_uncertainties[element] >= 0


# ============================================================================
# InternalStandardizer Tests
# ============================================================================


class TestInternalStandardizer:
    """Tests for InternalStandardizer class."""

    def test_init_valid(self):
        """Test valid initialization."""
        standardizer = InternalStandardizer("Fe", 0.70)
        assert standardizer.standard_element == "Fe"
        assert standardizer.known_concentration == 0.70

    def test_init_invalid_concentration_zero(self):
        """Test invalid zero concentration."""
        with pytest.raises(ValueError, match="must be in"):
            InternalStandardizer("Fe", 0.0)

    def test_init_invalid_concentration_negative(self):
        """Test invalid negative concentration."""
        with pytest.raises(ValueError, match="must be in"):
            InternalStandardizer("Fe", -0.1)

    def test_init_invalid_concentration_over_one(self):
        """Test invalid concentration > 1."""
        with pytest.raises(ValueError, match="must be in"):
            InternalStandardizer("Fe", 1.5)

    def test_standardize_basic(self):
        """Test basic standardization."""
        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.50, "Cr": 0.10, "Ni": 0.05},
            concentration_uncertainties={},
            iterations=1,
            converged=True,
        )

        standardizer = InternalStandardizer("Fe", known_concentration=0.70)
        std_result = standardizer.standardize(result)

        # Scale factor should be 0.70 / 0.50 = 1.4
        assert std_result.scale_factor == pytest.approx(1.4, rel=1e-6)

        # Fe should now be 0.70
        assert std_result.standardized_concentrations["Fe"] == pytest.approx(0.70, rel=1e-6)

        # Others should be scaled
        assert std_result.standardized_concentrations["Cr"] == pytest.approx(0.14, rel=1e-6)
        assert std_result.standardized_concentrations["Ni"] == pytest.approx(0.07, rel=1e-6)

    def test_standardize_missing_element(self):
        """Test standardization with missing standard element."""
        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"Cr": 0.10, "Ni": 0.05},
            concentration_uncertainties={},
            iterations=1,
            converged=True,
        )

        standardizer = InternalStandardizer("Fe", 0.70)
        with pytest.raises(ValueError, match="not found"):
            standardizer.standardize(result)

    def test_standardize_zero_standard(self):
        """Test standardization with zero standard concentration."""
        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.0, "Cr": 0.10},
            concentration_uncertainties={},
            iterations=1,
            converged=True,
        )

        standardizer = InternalStandardizer("Fe", 0.70)
        with pytest.raises(ValueError, match="zero or negative"):
            standardizer.standardize(result)

    def test_standardize_concentrations_method(self):
        """Test convenience method for raw concentrations."""
        standardizer = InternalStandardizer("Fe", 0.65)
        result = standardizer.standardize_concentrations({"Fe": 0.50, "Cr": 0.10})

        assert result.scale_factor == pytest.approx(1.3, rel=1e-6)
        assert result.standardized_concentrations["Fe"] == pytest.approx(0.65, rel=1e-6)

    def test_compute_ratios(self):
        """Test concentration ratio computation."""
        standardizer = InternalStandardizer("Fe", 0.70)
        concentrations = {"Fe": 0.50, "Cr": 0.10, "Ni": 0.05}

        ratios = standardizer.compute_ratios(concentrations)

        assert ratios["Fe"] == pytest.approx(1.0, rel=1e-6)
        assert ratios["Cr"] == pytest.approx(0.20, rel=1e-6)
        assert ratios["Ni"] == pytest.approx(0.10, rel=1e-6)

    def test_compute_ratios_missing_standard(self):
        """Test ratio computation with missing standard."""
        standardizer = InternalStandardizer("Fe", 0.70)

        with pytest.raises(ValueError, match="not in concentrations"):
            standardizer.compute_ratios({"Cr": 0.10})

    def test_compute_ratios_zero_standard(self):
        """Test ratio computation with zero standard."""
        standardizer = InternalStandardizer("Fe", 0.70)

        with pytest.raises(ValueError, match="zero or negative"):
            standardizer.compute_ratios({"Fe": 0.0, "Cr": 0.10})


# ============================================================================
# Integration Tests
# ============================================================================


class TestCombineCorrections:
    """Tests for combine_corrections convenience function."""

    @pytest.fixture
    def result(self) -> CFLIBSResult:
        """Create a test CF-LIBS result."""
        return CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={
                "Fe": 0.50,
                "Cr": 0.12,
                "Ni": 0.06,
                "Mn": 0.02,
            },
            concentration_uncertainties={},
            iterations=5,
            converged=True,
        )

    def test_combine_no_corrections(self, result):
        """Test with no corrections specified."""
        corrected = combine_corrections(result)

        # Should still apply matrix correction
        assert sum(corrected.values()) == pytest.approx(1.0, rel=1e-3)

    def test_combine_internal_standard_only(self, result):
        """Test with internal standardization only."""
        corrected = combine_corrections(
            result,
            internal_standard="Fe",
            standard_concentration=0.70,
        )

        # Fe should be at known concentration before matrix correction
        # Matrix correction then renormalizes
        assert sum(corrected.values()) == pytest.approx(1.0, rel=1e-3)

    def test_combine_matrix_type_only(self, result):
        """Test with explicit matrix type only."""
        corrected = combine_corrections(
            result,
            matrix_type=MatrixType.METALLIC,
        )

        assert sum(corrected.values()) == pytest.approx(1.0, rel=1e-3)

    def test_combine_all_corrections(self, result):
        """Test with both internal standard and matrix correction."""
        corrected = combine_corrections(
            result,
            internal_standard="Fe",
            standard_concentration=0.65,
            matrix_type=MatrixType.METALLIC,
        )

        assert sum(corrected.values()) == pytest.approx(1.0, rel=1e-3)

    def test_combine_custom_db(self, result):
        """Test with custom correction database."""
        custom_db = CorrectionFactorDB()
        custom_db.add_factor(
            CorrectionFactor(
                element="Fe",
                matrix_type=MatrixType.METALLIC,
                multiplicative=1.0,  # No correction
                uncertainty=0.0,
            )
        )

        corrected = combine_corrections(
            result,
            matrix_type=MatrixType.METALLIC,
            correction_db=custom_db,
        )

        assert "Fe" in corrected


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_element(self):
        """Test with single element concentration."""
        corrector = MatrixEffectCorrector()
        concentrations = {"Fe": 1.0}
        result = corrector.correct_concentrations(concentrations)

        # Single element should remain 1.0 after renormalization
        assert result.corrected_concentrations["Fe"] == pytest.approx(1.0, rel=1e-6)

    def test_very_small_concentrations(self):
        """Test with trace element concentrations."""
        corrector = MatrixEffectCorrector()
        concentrations = {
            "Fe": 0.999,
            "Cr": 0.0005,
            "Ni": 0.0003,
            "Mn": 0.0002,
        }
        result = corrector.correct_concentrations(concentrations)

        assert sum(result.corrected_concentrations.values()) == pytest.approx(1.0, rel=1e-4)

    def test_many_elements(self):
        """Test with many elements."""
        corrector = MatrixEffectCorrector()
        # Simulate complex geological sample
        concentrations = {
            "Si": 0.20,
            "O": 0.45,
            "Al": 0.08,
            "Fe": 0.05,
            "Ca": 0.04,
            "Mg": 0.03,
            "Na": 0.02,
            "K": 0.02,
            "Ti": 0.01,
            "Mn": 0.005,
            "P": 0.003,
            "S": 0.002,
        }
        result = corrector.correct_concentrations(concentrations)

        assert len(result.corrected_concentrations) == 12
        assert sum(result.corrected_concentrations.values()) == pytest.approx(1.0, rel=1e-4)

    def test_negative_concentration_passthrough(self):
        """Test handling of negative concentrations (should pass through)."""
        # Negative concentrations are physically impossible but might occur
        # from unconstrained fitting
        corrector = MatrixEffectCorrector(renormalize=False)
        concentrations = {"Fe": 0.90, "Cr": -0.05, "Ni": 0.15}
        result = corrector.correct_concentrations(concentrations)

        # Should not crash, negative value passes through
        assert "Cr" in result.corrected_concentrations

    def test_all_zero_concentrations(self):
        """Test with all zero concentrations."""
        corrector = MatrixEffectCorrector()
        result = corrector.correct_concentrations({"Fe": 0.0, "Cr": 0.0})

        assert result.matrix_type == MatrixType.UNKNOWN

    def test_reproducibility(self):
        """Test that corrections are reproducible."""
        corrector = MatrixEffectCorrector()
        concentrations = {"Fe": 0.70, "Cr": 0.18, "Ni": 0.08}

        result1 = corrector.correct_concentrations(concentrations)
        result2 = corrector.correct_concentrations(concentrations)

        for el in concentrations:
            assert result1.corrected_concentrations[el] == pytest.approx(
                result2.corrected_concentrations[el], rel=1e-10
            )


# ============================================================================
# Import Tests
# ============================================================================


class TestImports:
    """Tests for module imports from cflibs.inversion."""

    def test_import_from_inversion_package(self):
        """Test that all classes can be imported from cflibs.inversion."""
        from cflibs.inversion import (  # noqa: F401
            MatrixType as MT,
            MatrixClassificationResult,
            CorrectionFactor as CF,
            CorrectionFactorDB as CFDB,
            MatrixCorrectionResult,
            MatrixEffectCorrector as MEC,
            InternalStandardResult,
            InternalStandardizer as IS,
            combine_corrections as cc,
        )

        # Verify they're the correct types
        assert MT.METALLIC is not None
        assert MEC is not None
        assert IS is not None
        # Verify unused imports are accessible
        assert MatrixClassificationResult is not None
        assert CF is not None
        assert CFDB is not None
        assert MatrixCorrectionResult is not None
        assert InternalStandardResult is not None
        assert cc is not None