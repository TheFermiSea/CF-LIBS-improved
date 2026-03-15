"""
Tests for interpretable ML module.

These tests validate:
1. PhysicsGuidedFeatureExtractor functionality
2. SpectralExplainer (SHAP/LIME/permutation methods)
3. ExplanationValidator against spectroscopic knowledge
4. InterpretableModel wrapper
"""

import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path

# Mark entire module for interpretable tests
pytestmark = pytest.mark.unit


@pytest.fixture
def interpretable_db():
    """Create a database with spectral lines for interpretable ML tests."""
    import os

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)  # Close file descriptor to prevent leaks

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL
        )
    """)

    # Insert Fe spectral lines
    fe_lines = [
        ("Fe", 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000),
        ("Fe", 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500),
        ("Fe", 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200),
        ("Fe", 1, 438.35, 5.0e6, 0.0, 4.47, 9, 9, 800),
        ("Fe", 2, 238.20, 3.0e8, 0.0, 5.22, 10, 10, 600),
        ("Fe", 2, 259.94, 2.2e8, 0.0, 4.77, 8, 10, 400),
    ]

    # Insert Cu spectral lines
    cu_lines = [
        ("Cu", 1, 324.75, 1.4e8, 0.0, 3.82, 2, 4, 2000),
        ("Cu", 1, 327.40, 1.4e8, 0.0, 3.79, 2, 2, 1000),
        ("Cu", 1, 510.55, 2.0e6, 1.39, 3.82, 4, 4, 300),
    ]

    # Insert Ni spectral lines
    ni_lines = [
        ("Ni", 1, 341.48, 4.0e7, 0.0, 3.63, 9, 9, 700),
        ("Ni", 1, 352.45, 1.5e7, 0.02, 3.54, 11, 9, 350),
    ]

    all_lines = fe_lines + cu_lines + ni_lines

    conn.executemany(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        all_lines,
    )

    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink()


@pytest.fixture
def synthetic_spectrum():
    """Create a synthetic spectrum with known peaks."""
    wavelengths = np.linspace(300, 600, 3000)

    # Create spectrum with peaks at known positions
    spectrum = np.ones(len(wavelengths)) * 100  # Background

    # Add peaks at Fe, Cu, Ni lines
    peak_centers = [324.75, 327.40, 341.48, 352.45, 371.99, 373.49, 438.35, 510.55]
    peak_heights = [2000, 1000, 700, 350, 1000, 500, 800, 300]

    for center, height in zip(peak_centers, peak_heights):
        # Gaussian peak
        sigma = 0.2  # Peak width in nm
        peak = height * np.exp(-((wavelengths - center) ** 2) / (2 * sigma**2))
        spectrum += peak

    # Add some noise
    rng = np.random.default_rng(42)
    spectrum += rng.normal(0, 10, len(spectrum))
    spectrum = np.maximum(spectrum, 0)

    return wavelengths, spectrum


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_feature_types_exist(self):
        """Test that all expected feature types exist."""
        from cflibs.inversion.interpretable import FeatureType

        assert FeatureType.PEAK_INTENSITY.value == "peak_intensity"
        assert FeatureType.PEAK_AREA.value == "peak_area"
        assert FeatureType.PEAK_WIDTH.value == "peak_width"
        assert FeatureType.REGION_SUM.value == "region_sum"
        assert FeatureType.RATIO.value == "ratio"
        assert FeatureType.BASELINE.value == "baseline"


class TestSpectralFeature:
    """Tests for SpectralFeature dataclass."""

    def test_spectral_feature_creation(self):
        """Test creating a SpectralFeature."""
        from cflibs.inversion.interpretable import SpectralFeature, FeatureType

        feature = SpectralFeature(
            name="Fe_I_371.99nm_intensity",
            feature_type=FeatureType.PEAK_INTENSITY,
            wavelength_nm=371.99,
            element="Fe",
            ionization_stage=0,
            value=1000.0,
            uncertainty=50.0,
            wavelength_range=(371.5, 372.5),
        )

        assert feature.name == "Fe_I_371.99nm_intensity"
        assert feature.feature_type == FeatureType.PEAK_INTENSITY
        assert feature.wavelength_nm == pytest.approx(371.99)
        assert feature.element == "Fe"
        assert feature.ionization_stage == 0
        assert feature.value == 1000.0
        assert feature.uncertainty == 50.0


class TestPhysicsGuidedFeatureExtractor:
    """Tests for PhysicsGuidedFeatureExtractor class."""

    def test_initialization(self, interpretable_db):
        """Test extractor initialization."""
        from cflibs.inversion.interpretable import PhysicsGuidedFeatureExtractor

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        assert extractor.elements == ["Fe", "Cu"]
        assert len(extractor.known_lines) > 0

    def test_extract_features_from_spectrum(self, interpretable_db, synthetic_spectrum):
        """Test feature extraction from synthetic spectrum."""
        from cflibs.inversion.interpretable import PhysicsGuidedFeatureExtractor

        wavelengths, spectrum = synthetic_spectrum

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Fe", "Cu", "Ni"],
            wavelength_tolerance_nm=0.5,
        )

        result = extractor.extract(wavelengths, spectrum)

        # Should find features
        assert len(result.features) > 0
        assert len(result.feature_names) > 0
        assert result.feature_matrix.shape[0] == 1  # Single sample
        assert result.feature_matrix.shape[1] == len(result.features)

    def test_extract_peak_features(self, interpretable_db, synthetic_spectrum):
        """Test that peak features are extracted correctly."""
        from cflibs.inversion.interpretable import (
            PhysicsGuidedFeatureExtractor,
            FeatureType,
        )

        wavelengths, spectrum = synthetic_spectrum

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Cu"],
            wavelength_tolerance_nm=0.5,
        )

        result = extractor.extract(wavelengths, spectrum)

        # Should find Cu peaks
        intensity_features = [
            f for f in result.features if f.feature_type == FeatureType.PEAK_INTENSITY
        ]
        area_features = [f for f in result.features if f.feature_type == FeatureType.PEAK_AREA]

        # At least one intensity and area feature for Cu
        cu_intensity = [f for f in intensity_features if f.element == "Cu"]
        cu_area = [f for f in area_features if f.element == "Cu"]

        assert len(cu_intensity) > 0, "Should find Cu intensity features"
        assert len(cu_area) > 0, "Should find Cu area features"

    def test_extract_ratio_features(self, interpretable_db, synthetic_spectrum):
        """Test that ratio features are extracted between elements."""
        from cflibs.inversion.interpretable import (
            PhysicsGuidedFeatureExtractor,
            FeatureType,
        )

        wavelengths, spectrum = synthetic_spectrum

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
            wavelength_tolerance_nm=0.5,
        )

        result = extractor.extract(wavelengths, spectrum)

        # Should have ratio features
        ratio_features = [f for f in result.features if f.feature_type == FeatureType.RATIO]

        # If both Fe and Cu are detected, should have ratio
        if "Fe" in result.elements_detected and "Cu" in result.elements_detected:
            assert len(ratio_features) > 0, "Should find ratio features"

    def test_extract_batch(self, interpretable_db, synthetic_spectrum):
        """Test batch feature extraction."""
        from cflibs.inversion.interpretable import PhysicsGuidedFeatureExtractor

        wavelengths, spectrum = synthetic_spectrum

        # Create multiple spectra with different intensities
        spectra = np.vstack(
            [
                spectrum,
                spectrum * 0.5,
                spectrum * 1.5,
            ]
        )

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        result = extractor.extract_batch(wavelengths, spectra)

        assert result.feature_matrix.shape[0] == 3  # 3 samples
        assert result.feature_matrix.shape[1] > 0  # Has features


class TestSpectralExplainer:
    """Tests for SpectralExplainer class."""

    @pytest.fixture
    def simple_model(self, synthetic_spectrum):
        """Create a simple linear model for testing."""
        wavelengths, _ = synthetic_spectrum

        class SimpleModel:
            def __init__(self, wavelengths):
                self.wavelengths = wavelengths
                # Model that weights high wavelengths
                self.weights = np.linspace(0, 1, len(wavelengths))

            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return np.sum(X * self.weights, axis=1)

        return SimpleModel(wavelengths)

    def test_explainer_initialization(self, synthetic_spectrum, simple_model):
        """Test explainer initialization."""
        from cflibs.inversion.interpretable import SpectralExplainer

        wavelengths, _ = synthetic_spectrum

        explainer = SpectralExplainer(simple_model, wavelengths)

        assert len(explainer.wavelengths) == len(wavelengths)
        assert len(explainer.feature_names) == len(wavelengths)

    def test_explain_permutation(self, synthetic_spectrum, simple_model):
        """Test permutation importance explanation."""
        from cflibs.inversion.interpretable import SpectralExplainer

        wavelengths, spectrum = synthetic_spectrum

        explainer = SpectralExplainer(simple_model, wavelengths)
        explanation = explainer._explain_permutation(spectrum, n_samples=10)

        assert explanation.method == "permutation"
        assert len(explanation.feature_importances) > 0
        assert len(explanation.wavelength_importances) == len(wavelengths)
        assert len(explanation.top_features) <= 20

    def test_explain_lime(self, synthetic_spectrum, simple_model):
        """Test LIME explanation."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import SpectralExplainer

        wavelengths, spectrum = synthetic_spectrum

        explainer = SpectralExplainer(simple_model, wavelengths)
        explanation = explainer.explain_lime(spectrum, n_samples=50)

        assert explanation.method == "lime"
        assert len(explanation.feature_importances) > 0
        assert explanation.prediction is not None

    def test_compute_saliency_map(self, synthetic_spectrum, simple_model):
        """Test saliency map computation."""
        from cflibs.inversion.interpretable import SpectralExplainer

        wavelengths, spectrum = synthetic_spectrum

        explainer = SpectralExplainer(simple_model, wavelengths)
        saliency = explainer.compute_saliency_map(spectrum, smooth_sigma=0)

        assert saliency.shape == spectrum.shape
        assert np.all(saliency >= 0)  # Should be non-negative
        assert np.max(saliency) <= 1.0  # Should be normalized


class TestExplanationValidator:
    """Tests for ExplanationValidator class."""

    def test_validator_initialization(self, interpretable_db):
        """Test validator initialization."""
        from cflibs.inversion.interpretable import ExplanationValidator

        validator = ExplanationValidator(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        assert validator.elements == ["Fe", "Cu"]
        assert len(validator.known_lines) > 0

    def test_validate_good_explanation(self, interpretable_db, synthetic_spectrum):
        """Test validation of explanation matching known lines."""
        from cflibs.inversion.interpretable import (
            ExplanationValidator,
            SpectralExplanation,
        )

        wavelengths, _ = synthetic_spectrum

        validator = ExplanationValidator(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        # Create explanation with features at known Cu line positions
        feature_importances = {
            "324.75nm": 0.3,  # Cu I line
            "327.40nm": 0.2,  # Cu I line
            "371.99nm": 0.15,  # Fe I line
            "400.00nm": 0.01,  # Unknown
        }

        explanation = SpectralExplanation(
            method="test",
            feature_importances=feature_importances,
            wavelength_importances=np.zeros(len(wavelengths)),
            top_features=list(feature_importances.items()),
            prediction=1.0,
        )

        result = validator.validate(explanation, wavelengths)

        # Should have high score since most features match known lines
        assert result.score > 0.5
        assert len(result.matched_lines) > 0

    def test_validate_spurious_explanation(self, interpretable_db, synthetic_spectrum):
        """Test detection of spurious correlations."""
        from cflibs.inversion.interpretable import (
            ExplanationValidator,
            SpectralExplanation,
        )

        wavelengths, _ = synthetic_spectrum

        validator = ExplanationValidator(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        # Create explanation with features at unknown positions
        feature_importances = {
            "400.00nm": 0.3,  # Unknown wavelength
            "450.00nm": 0.25,  # Unknown wavelength
            "500.00nm": 0.2,  # Unknown wavelength
            "324.75nm": 0.05,  # Cu I line (low importance)
        }

        explanation = SpectralExplanation(
            method="test",
            feature_importances=feature_importances,
            wavelength_importances=np.zeros(len(wavelengths)),
            top_features=list(feature_importances.items()),
            prediction=1.0,
        )

        result = validator.validate(explanation, wavelengths)

        # Should have low score due to unmatched features
        assert result.score < 0.5
        assert len(result.spurious_correlations) > 0

    def test_generate_report(self, interpretable_db, synthetic_spectrum):
        """Test report generation."""
        from cflibs.inversion.interpretable import (
            ExplanationValidator,
            SpectralExplanation,
        )

        wavelengths, _ = synthetic_spectrum

        validator = ExplanationValidator(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        feature_importances = {
            "324.75nm": 0.3,
            "371.99nm": 0.2,
        }

        explanation = SpectralExplanation(
            method="test",
            feature_importances=feature_importances,
            wavelength_importances=np.zeros(len(wavelengths)),
            top_features=list(feature_importances.items()),
        )

        result = validator.validate(explanation, wavelengths)
        report = validator.generate_report(result)

        assert "Validation Report" in report
        assert "Alignment Score" in report


class TestInterpretableModel:
    """Tests for InterpretableModel class."""

    def test_model_creation(self, interpretable_db):
        """Test interpretable model creation."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import InterpretableModel

        model = InterpretableModel(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
            model_type="ridge",
        )

        assert model.model_type == "ridge"
        assert model.elements == ["Fe", "Cu"]
        assert not model.is_fitted_

    def test_invalid_model_type(self, interpretable_db):
        """Test that invalid model type raises error."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import InterpretableModel

        with pytest.raises(ValueError, match="model_type must be one of"):
            InterpretableModel(
                db_path=interpretable_db,
                elements=["Fe", "Cu"],
                model_type="invalid_model",
            )

    def test_fit_and_predict(self, interpretable_db, synthetic_spectrum):
        """Test model fitting and prediction."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import InterpretableModel

        wavelengths, spectrum = synthetic_spectrum

        # Create training data
        n_samples = 20
        rng = np.random.default_rng(42)
        spectra = np.vstack([spectrum * (0.5 + rng.random()) for _ in range(n_samples)])
        targets = rng.random(n_samples)

        model = InterpretableModel(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
            model_type="ridge",
        )

        # Fit
        model.fit(wavelengths, spectra, targets)
        assert model.is_fitted_
        assert model.feature_names_ is not None

        # Predict
        predictions = model.predict(wavelengths, spectra[:5])
        assert predictions.shape == (5,)

    def test_predict_explain(self, interpretable_db, synthetic_spectrum):
        """Test prediction with explanations."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import InterpretableModel

        wavelengths, spectrum = synthetic_spectrum

        # Create training data
        n_samples = 20
        rng = np.random.default_rng(42)
        spectra = np.vstack([spectrum * (0.5 + rng.random()) for _ in range(n_samples)])
        targets = rng.random(n_samples)

        model = InterpretableModel(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
            model_type="ridge",
        )

        model.fit(wavelengths, spectra, targets)

        # Predict with explanations
        predictions, explanations = model.predict_explain(wavelengths, spectra[:3])

        assert len(predictions) == 3
        assert len(explanations) == 3
        assert all(isinstance(exp, dict) for exp in explanations)

    def test_feature_importance_table(self, interpretable_db, synthetic_spectrum):
        """Test feature importance table generation."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import InterpretableModel

        wavelengths, spectrum = synthetic_spectrum

        n_samples = 20
        rng = np.random.default_rng(42)
        spectra = np.vstack([spectrum * (0.5 + rng.random()) for _ in range(n_samples)])
        targets = rng.random(n_samples)

        model = InterpretableModel(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
            model_type="ridge",
        )

        model.fit(wavelengths, spectra, targets)
        table = model.get_feature_importance_table()

        assert "Feature Importances" in table
        assert "Rank" in table

    def test_different_model_types(self, interpretable_db, synthetic_spectrum):
        """Test different model types."""
        pytest.importorskip("sklearn")

        from cflibs.inversion.interpretable import InterpretableModel

        wavelengths, spectrum = synthetic_spectrum

        n_samples = 20
        rng = np.random.default_rng(42)
        spectra = np.vstack([spectrum * (0.5 + rng.random()) for _ in range(n_samples)])
        targets = rng.random(n_samples)

        for model_type in ["ridge", "lasso", "random_forest"]:
            model = InterpretableModel(
                db_path=interpretable_db,
                elements=["Fe", "Cu"],
                model_type=model_type,
            )

            model.fit(wavelengths, spectra, targets)
            predictions = model.predict(wavelengths, spectra[:2])

            assert model.is_fitted_
            assert len(predictions) == 2


class TestDataclasses:
    """Tests for result dataclasses."""

    def test_feature_extraction_result(self):
        """Test FeatureExtractionResult dataclass."""
        from cflibs.inversion.interpretable import (
            FeatureExtractionResult,
            SpectralFeature,
            FeatureType,
        )

        feature = SpectralFeature(
            name="test",
            feature_type=FeatureType.PEAK_INTENSITY,
            wavelength_nm=500.0,
            element="Fe",
            ionization_stage=0,
            value=100.0,
        )

        result = FeatureExtractionResult(
            features=[feature],
            feature_matrix=np.array([[100.0]]),
            feature_names=["test"],
            wavelengths_used=[500.0],
            elements_detected=["Fe"],
        )

        assert len(result.features) == 1
        assert result.feature_matrix.shape == (1, 1)

    def test_spectral_explanation(self):
        """Test SpectralExplanation dataclass."""
        from cflibs.inversion.interpretable import SpectralExplanation

        explanation = SpectralExplanation(
            method="shap",
            feature_importances={"a": 0.5, "b": 0.3},
            wavelength_importances=np.array([0.5, 0.3]),
            top_features=[("a", 0.5), ("b", 0.3)],
            base_value=0.1,
            prediction=1.5,
        )

        assert explanation.method == "shap"
        assert explanation.prediction == 1.5
        assert len(explanation.top_features) == 2

    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        from cflibs.inversion.interpretable import ValidationResult

        result = ValidationResult(
            is_valid=True,
            score=0.8,
            matched_lines=[("Fe", 371.99, 0.3)],
            unmatched_features=[("unknown", 0.05)],
            spurious_correlations=[],
            recommendations=["Consider adding more elements"],
        )

        assert result.is_valid
        assert result.score == 0.8
        assert len(result.matched_lines) == 1
        assert len(result.recommendations) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_spectrum(self, interpretable_db):
        """Test handling of empty/zero spectrum."""
        from cflibs.inversion.interpretable import PhysicsGuidedFeatureExtractor

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        wavelengths = np.linspace(300, 600, 1000)
        spectrum = np.zeros(len(wavelengths))

        result = extractor.extract(wavelengths, spectrum)

        # Should handle gracefully (may find no features)
        assert result.feature_matrix is not None

    def test_single_element(self, interpretable_db, synthetic_spectrum):
        """Test with single element."""
        from cflibs.inversion.interpretable import PhysicsGuidedFeatureExtractor

        wavelengths, spectrum = synthetic_spectrum

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Cu"],  # Single element
        )

        result = extractor.extract(wavelengths, spectrum)

        # Should find Cu features but no ratio features
        assert result.feature_matrix is not None

    def test_out_of_range_wavelengths(self, interpretable_db):
        """Test with wavelengths outside database range."""
        from cflibs.inversion.interpretable import PhysicsGuidedFeatureExtractor

        extractor = PhysicsGuidedFeatureExtractor(
            db_path=interpretable_db,
            elements=["Fe", "Cu"],
        )

        # Wavelengths outside typical LIBS range
        wavelengths = np.linspace(100, 200, 1000)
        spectrum = np.random.default_rng(42).random(len(wavelengths)) * 100

        result = extractor.extract(wavelengths, spectrum)

        # Should handle gracefully (may find no features matching known lines)
        assert result.feature_matrix is not None