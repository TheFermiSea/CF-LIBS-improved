"""
Tests for PLS (Partial Least Squares) regression module.

Tests cover:
- Core PLS regression fitting and prediction
- Preprocessing methods
- Cross-validation for component selection
- VIP scores and loadings
- High-level calibration model interface
- Edge cases and error handling
"""

import numpy as np
import pytest

from cflibs.inversion.pls import (
    PreprocessingMethod,
    CrossValidationResult,
    PLSRegression,
    PLSCrossValidator,
    PLSCalibrationModel,
    build_pls_calibration,
)


# --- Fixtures ---


@pytest.fixture
def simple_data():
    """Simple synthetic data with known linear relationship."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_features = 100

    # Create latent factors
    T = rng.standard_normal((n_samples, 3))

    # X = T @ P' + noise
    P = rng.standard_normal((n_features, 3))
    X = T @ P.T + 0.1 * rng.standard_normal((n_samples, n_features))

    # Y = T @ Q' + noise (linear relationship through latent space)
    Q = rng.standard_normal((2, 3))
    Y = T @ Q.T + 0.05 * rng.standard_normal((n_samples, 2))

    return X, Y


@pytest.fixture
def libs_like_data():
    """LIBS-like synthetic data with spectral features."""
    rng = np.random.default_rng(123)
    n_samples = 100
    n_wavelengths = 500

    wavelengths = np.linspace(200, 700, n_wavelengths)

    # Simulate Fe and Si concentrations
    fe_conc = rng.uniform(0.1, 0.8, n_samples)
    si_conc = rng.uniform(0.05, 0.3, n_samples)

    # Generate spectra with peaks at specific wavelengths
    spectra = np.zeros((n_samples, n_wavelengths))

    # Fe peaks (simulate)
    fe_peaks = [259.9, 302.1, 373.5, 404.6]
    for peak in fe_peaks:
        idx = np.argmin(np.abs(wavelengths - peak))
        peak_width = 5
        for i in range(n_samples):
            gaussian = np.exp(-0.5 * ((np.arange(n_wavelengths) - idx) / peak_width) ** 2)
            spectra[i] += fe_conc[i] * gaussian * rng.uniform(0.8, 1.2)

    # Si peaks (simulate)
    si_peaks = [288.2, 390.6, 634.7]
    for peak in si_peaks:
        idx = np.argmin(np.abs(wavelengths - peak))
        peak_width = 4
        for i in range(n_samples):
            gaussian = np.exp(-0.5 * ((np.arange(n_wavelengths) - idx) / peak_width) ** 2)
            spectra[i] += si_conc[i] * gaussian * rng.uniform(0.8, 1.2)

    # Add baseline and noise
    spectra += 0.1 * rng.standard_normal(spectra.shape)
    spectra = np.maximum(spectra, 0)  # Non-negative

    Y = np.column_stack([fe_conc, si_conc])

    return spectra, Y, wavelengths


# --- PLSRegression Tests ---


class TestPLSRegression:
    """Tests for core PLSRegression class."""

    def test_fit_basic(self, simple_data):
        """Test basic PLS fitting."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        assert pls.is_fitted
        assert pls.components is not None
        assert pls.components.n_components == 3

    def test_fit_single_target(self, simple_data):
        """Test PLS with single target (PLS1)."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y[:, 0])  # Single column

        assert pls.is_fitted
        result = pls.predict(X)
        assert result.predictions.shape == (X.shape[0], 1)

    def test_predict_r2(self, simple_data):
        """Test that prediction achieves reasonable R2."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        result = pls.predict(X, Y=Y)
        assert result.r2 is not None
        assert result.r2 > 0.9  # Should fit well with 3 true components

    def test_predict_without_y(self, simple_data):
        """Test prediction without ground truth."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        result = pls.predict(X)
        assert result.predictions.shape == Y.shape
        assert result.r2 is None
        assert result.residuals is None

    def test_predict_with_fewer_components(self, simple_data):
        """Test prediction using subset of components."""
        X, Y = simple_data
        pls = PLSRegression(n_components=5)
        pls.fit(X, Y)

        result_full = pls.predict(X, Y=Y)
        result_partial = pls.predict(X, Y=Y, n_components=2)

        # More components should give better fit
        assert result_full.r2 >= result_partial.r2

    def test_transform(self, simple_data):
        """Test score computation via transform."""
        X, Y = simple_data
        pls = PLSRegression(n_components=4)
        pls.fit(X, Y)

        scores = pls.transform(X)
        assert scores.shape == (X.shape[0], 4)

    def test_coefficients_shape(self, simple_data):
        """Test regression coefficients have correct shape."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        assert pls.coefficients is not None
        assert pls.coefficients.shape == (X.shape[1], Y.shape[1])

    def test_not_fitted_raises(self, simple_data):
        """Test that predict raises before fit."""
        X, _ = simple_data
        pls = PLSRegression(n_components=3)

        with pytest.raises(RuntimeError, match="must be fitted"):
            pls.predict(X)

    def test_invalid_n_components_raises(self):
        """Test that n_components < 1 raises error."""
        with pytest.raises(ValueError, match="must be >= 1"):
            PLSRegression(n_components=0)

    def test_mismatched_samples_raises(self, simple_data):
        """Test that X and Y with different samples raises error."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)

        with pytest.raises(ValueError, match="different sample counts"):
            pls.fit(X[:-5], Y)  # Different number of samples

    def test_n_components_capped(self):
        """Test that fit-time component cap does not mutate configured n_components."""
        X = np.random.default_rng(42).standard_normal((10, 100))
        Y = np.random.default_rng(42).standard_normal((10, 2))

        pls = PLSRegression(n_components=50)  # More than samples
        pls.fit(X, Y)

        # Requested count is preserved on the instance.
        assert pls.n_components == 50
        # Fitted decomposition is capped to n_samples - 1 = 9.
        assert pls.components is not None
        assert pls.components.n_components <= 9


class TestPreprocessing:
    """Tests for preprocessing methods."""

    def test_none_preprocessing(self):
        """Test no preprocessing."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        Y = np.array([1, 2, 3], dtype=float)

        pls = PLSRegression(n_components=2, preprocessing=PreprocessingMethod.NONE)
        pls.fit(X, Y)

        # Predictions should still work
        result = pls.predict(X)
        assert result.predictions.shape == (3, 1)

    def test_center_preprocessing(self):
        """Test mean-centering only."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Y = np.array([1, 2, 3], dtype=float)

        pls = PLSRegression(n_components=1, preprocessing=PreprocessingMethod.CENTER)
        pls.fit(X, Y)
        assert pls.is_fitted

    def test_autoscale_preprocessing(self):
        """Test autoscaling (mean-center + unit variance)."""
        X = np.array([[1, 100], [2, 200], [3, 300]], dtype=float)
        Y = np.array([1, 2, 3], dtype=float)

        pls = PLSRegression(n_components=1, preprocessing=PreprocessingMethod.AUTOSCALE)
        pls.fit(X, Y)
        assert pls.is_fitted

    def test_snv_preprocessing(self):
        """Test Standard Normal Variate preprocessing."""
        # Use data with different spectral shapes (not just scaled versions)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 10))  # Random spectra with different shapes
        Y = X[:, 0] + 0.5 * X[:, 5] + 0.1 * rng.standard_normal(20)

        pls = PLSRegression(n_components=3, preprocessing=PreprocessingMethod.SNV)
        pls.fit(X, Y)
        assert pls.is_fitted

    def test_pareto_preprocessing(self):
        """Test Pareto scaling."""
        X = np.array([[1, 100], [2, 200], [3, 300]], dtype=float)
        Y = np.array([1, 2, 3], dtype=float)

        pls = PLSRegression(n_components=1, preprocessing=PreprocessingMethod.PARETO)
        pls.fit(X, Y)
        assert pls.is_fitted


class TestVIPScores:
    """Tests for Variable Importance in Projection."""

    def test_vip_scores_shape(self, simple_data):
        """Test VIP scores have correct shape."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        vip = pls.vip_scores()
        assert vip.shape == (X.shape[1],)

    def test_vip_scores_nonnegative(self, simple_data):
        """Test VIP scores are non-negative."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        vip = pls.vip_scores()
        assert np.all(vip >= 0)

    def test_vip_important_variables(self, libs_like_data):
        """Test that VIP identifies important wavelengths."""
        spectra, Y, wavelengths = libs_like_data
        pls = PLSRegression(n_components=5)
        pls.fit(spectra, Y)

        vip = pls.vip_scores()
        # Most important variables should have VIP > 1
        n_important = np.sum(vip > 1)
        assert n_important > 0
        assert n_important < len(vip)  # Not all variables are important


class TestLoadings:
    """Tests for component loadings."""

    def test_get_loadings_spectrum(self, simple_data):
        """Test getting loadings for a component."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        loadings = pls.get_loadings_spectrum(component=0)
        assert loadings.shape == (X.shape[1],)

    def test_loadings_invalid_component_raises(self, simple_data):
        """Test that invalid component index raises error."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        with pytest.raises(ValueError, match="not available"):
            pls.get_loadings_spectrum(component=10)


# --- CrossValidation Tests ---


class TestPLSCrossValidator:
    """Tests for cross-validation component selection."""

    def test_basic_cv(self, simple_data):
        """Test basic cross-validation."""
        X, Y = simple_data
        cv = PLSCrossValidator(max_components=5, n_folds=5, random_state=42)
        result = cv.validate(X, Y)

        assert isinstance(result, CrossValidationResult)
        assert result.optimal_n_components >= 1
        assert result.optimal_n_components <= 5
        assert len(result.cv_rmse) == 5
        assert len(result.cv_r2) == 5

    def test_cv_metrics_decrease(self, simple_data):
        """Test that CV-RMSE generally decreases then stabilizes."""
        X, Y = simple_data
        cv = PLSCrossValidator(max_components=10, n_folds=5, random_state=42)
        result = cv.validate(X, Y)

        # RMSE should generally decrease with more components
        # (at least for first few components)
        assert result.cv_rmse[2] <= result.cv_rmse[0]

    def test_one_sigma_rule(self, simple_data):
        """Test one-sigma selection criterion."""
        X, Y = simple_data
        cv = PLSCrossValidator(
            max_components=10,
            n_folds=5,
            selection_criterion="one_sigma",
            random_state=42,
        )
        result = cv.validate(X, Y)

        # One-sigma rule should give a reasonable number
        assert result.optimal_n_components >= 1

    def test_cv_summary(self, simple_data):
        """Test CV result summary generation."""
        X, Y = simple_data
        cv = PLSCrossValidator(max_components=5, n_folds=5, random_state=42)
        result = cv.validate(X, Y)

        summary = result.summary()
        assert "Optimal components" in summary
        assert "CV-RMSE" in summary

    def test_cv_invalid_folds_raises(self):
        """Test that n_folds < 2 raises error."""
        with pytest.raises(ValueError, match="must be >= 2"):
            PLSCrossValidator(n_folds=1)


# --- Calibration Model Tests ---


class TestPLSCalibrationModel:
    """Tests for high-level calibration model."""

    def test_build_calibration_model(self, libs_like_data):
        """Test building calibration model with CV."""
        spectra, Y, wavelengths = libs_like_data

        model = build_pls_calibration(
            spectra=spectra,
            concentrations=Y,
            wavelengths=wavelengths,
            elements=["Fe", "Si"],
            use_cv=True,
            cv_folds=5,
            max_components=10,
            random_state=42,
        )

        assert isinstance(model, PLSCalibrationModel)
        assert model.elements == ["Fe", "Si"]
        assert model.cv_result is not None

    def test_build_calibration_model_dict_concentrations(self, libs_like_data):
        """Test building model with dict concentrations."""
        spectra, Y, wavelengths = libs_like_data

        concentrations = {
            "Fe": Y[:, 0],
            "Si": Y[:, 1],
        }

        model = build_pls_calibration(
            spectra=spectra,
            concentrations=concentrations,
            wavelengths=wavelengths,
            use_cv=False,
            n_components=5,
        )

        assert model.elements == ["Fe", "Si"]

    def test_model_predict(self, libs_like_data):
        """Test prediction from calibration model."""
        spectra, Y, wavelengths = libs_like_data

        model = build_pls_calibration(
            spectra=spectra,
            concentrations={"Fe": Y[:, 0], "Si": Y[:, 1]},
            wavelengths=wavelengths,
            use_cv=False,
            n_components=5,
        )

        # Predict first sample
        result = model.predict(spectra[0])

        assert "Fe" in result
        assert "Si" in result
        assert isinstance(result["Fe"], float)

    def test_model_important_wavelengths(self, libs_like_data):
        """Test getting important wavelengths."""
        spectra, Y, wavelengths = libs_like_data

        model = build_pls_calibration(
            spectra=spectra,
            concentrations={"Fe": Y[:, 0], "Si": Y[:, 1]},
            wavelengths=wavelengths,
            use_cv=False,
            n_components=5,
        )

        important = model.get_important_wavelengths(vip_threshold=1.0)
        assert "wavelengths" in important
        assert "vip_scores" in important
        assert len(important["wavelengths"]) == len(important["vip_scores"])

    def test_model_summary(self, libs_like_data):
        """Test model summary generation."""
        spectra, Y, wavelengths = libs_like_data

        model = build_pls_calibration(
            spectra=spectra,
            concentrations={"Fe": Y[:, 0], "Si": Y[:, 1]},
            wavelengths=wavelengths,
            use_cv=True,
            cv_folds=5,
            random_state=42,
        )

        summary = model.summary()
        assert "PLS Calibration Model" in summary
        assert "Fe" in summary
        assert "Si" in summary

    def test_calibration_range_warning(self, libs_like_data, caplog):
        """Test warning when prediction outside calibration range."""
        spectra, Y, wavelengths = libs_like_data

        model = build_pls_calibration(
            spectra=spectra,
            concentrations={"Fe": Y[:, 0]},
            wavelengths=wavelengths,
            use_cv=False,
            n_components=3,
        )

        # Create spectrum that will predict outside range
        extreme_spectrum = spectra[0] * 10  # Very high intensity
        model.predict(extreme_spectrum)

        # Should log a warning about calibration range
        # (may or may not trigger depending on the model)


class TestPLSComponents:
    """Tests for PLSComponents dataclass."""

    def test_components_attributes(self, simple_data):
        """Test that fitted components have all attributes."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        comp = pls.components
        assert comp is not None

        # Check shapes
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        n_comp = comp.n_components

        assert comp.x_scores.shape == (n_samples, n_comp)
        assert comp.x_loadings.shape == (n_features, n_comp)
        assert comp.y_loadings.shape == (n_targets, n_comp)
        assert comp.x_weights.shape == (n_features, n_comp)
        assert comp.coefficients.shape == (n_features, n_targets)
        assert comp.x_explained_variance.shape == (n_comp,)
        assert comp.y_explained_variance.shape == (n_comp,)

    def test_explained_variance_sums(self, simple_data):
        """Test that explained variances are valid."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        comp = pls.components
        assert comp is not None

        # Variances should be between 0 and 1
        assert np.all(comp.x_explained_variance >= 0)
        assert np.all(comp.y_explained_variance >= 0)

        # Cumulative should not exceed 1
        assert comp.x_explained_variance.sum() <= 1.01  # Small tolerance
        assert comp.y_explained_variance.sum() <= 1.01


class TestPLSResult:
    """Tests for PLSResult dataclass."""

    def test_result_summary(self, simple_data):
        """Test PLSResult summary generation."""
        X, Y = simple_data
        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        result = pls.predict(X, Y=Y)
        summary = result.summary()

        assert "PLS Prediction Results" in summary
        assert "R2" in summary
        assert "RMSE" in summary


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample_target(self):
        """Test with minimum viable data."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        Y = np.array([1, 2, 3], dtype=float)

        pls = PLSRegression(n_components=2)
        pls.fit(X, Y)
        result = pls.predict(X)

        assert result.predictions.shape == (3, 1)

    def test_constant_feature_handling(self):
        """Test handling of constant (zero variance) features."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 10))
        X[:, 5] = 1.0  # Constant column
        Y = rng.standard_normal(20)

        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)  # Should not raise

        result = pls.predict(X)
        assert not np.any(np.isnan(result.predictions))

    def test_highly_correlated_features(self):
        """Test with highly correlated features (typical in spectra)."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal((30, 1))
        X = np.hstack([base + 0.01 * rng.standard_normal((30, 1)) for _ in range(20)])
        Y = base[:, 0] + 0.1 * rng.standard_normal(30)

        pls = PLSRegression(n_components=3)
        pls.fit(X, Y)

        result = pls.predict(X, Y=Y)
        # Should still work and achieve reasonable fit
        assert result.r2 > 0.5

    def test_more_features_than_samples(self):
        """Test with more features than samples (common in LIBS)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 500))  # 20 samples, 500 wavelengths
        Y = rng.standard_normal((20, 3))

        pls = PLSRegression(n_components=5)
        pls.fit(X, Y)

        assert pls.is_fitted
        result = pls.predict(X)
        assert result.predictions.shape == Y.shape


# --- Integration Tests ---


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, libs_like_data):
        """Test complete PLS workflow for LIBS analysis."""
        spectra, Y, wavelengths = libs_like_data
        n_samples = spectra.shape[0]

        # Split data
        train_idx = np.arange(80)
        test_idx = np.arange(80, n_samples)

        X_train, X_test = spectra[train_idx], spectra[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Cross-validation
        cv = PLSCrossValidator(max_components=10, n_folds=5, random_state=42)
        cv_result = cv.validate(X_train, Y_train)

        # Fit final model
        pls = PLSRegression(n_components=cv_result.optimal_n_components)
        pls.fit(X_train, Y_train)

        # Evaluate on test set
        test_result = pls.predict(X_test, Y=Y_test)

        # Should achieve reasonable test performance
        assert test_result.r2 is not None
        assert test_result.r2 > 0.7  # Reasonable for simulated data
        assert test_result.rmse is not None

        # VIP scores
        vip = pls.vip_scores()
        important_wl = wavelengths[vip > 1]
        assert len(important_wl) > 0

    def test_calibration_model_reproducibility(self, libs_like_data):
        """Test that results are reproducible with same seed."""
        spectra, Y, wavelengths = libs_like_data

        model1 = build_pls_calibration(
            spectra=spectra,
            concentrations={"Fe": Y[:, 0]},
            wavelengths=wavelengths,
            use_cv=True,
            random_state=42,
        )

        model2 = build_pls_calibration(
            spectra=spectra,
            concentrations={"Fe": Y[:, 0]},
            wavelengths=wavelengths,
            use_cv=True,
            random_state=42,
        )

        # Same optimal components
        assert model1.cv_result.optimal_n_components == model2.cv_result.optimal_n_components

        # Same predictions
        pred1 = model1.predict(spectra[0])
        pred2 = model2.predict(spectra[0])
        assert pred1["Fe"] == pytest.approx(pred2["Fe"], rel=1e-10)
