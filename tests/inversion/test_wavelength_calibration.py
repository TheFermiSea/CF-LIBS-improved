"""Tests for cflibs.inversion.preprocess.wavelength_calibration helpers."""

import numpy as np
import pytest

from cflibs.inversion.preprocess.wavelength_calibration import (
    WavelengthCalibrationResult,
    _compute_bic,
    _dedupe_one_to_one,
    _eval_model,
    _fit_model,
    _is_monotonic_on_grid,
    _model_min_points,
    _model_param_count,
)


class TestModelMetadata:
    def test_min_points(self):
        assert _model_min_points("shift") == 1
        assert _model_min_points("affine") == 2
        assert _model_min_points("quadratic") == 3
        assert _model_min_points("none") == 1

    def test_param_count(self):
        assert _model_param_count("shift") == 1
        assert _model_param_count("affine") == 2
        assert _model_param_count("quadratic") == 3
        assert _model_param_count("none") == 0


class TestEvalModel:
    def test_shift(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _eval_model(x, "shift", [0.5])
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5])

    def test_affine(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _eval_model(x, "affine", [2.0, 1.0])
        np.testing.assert_allclose(result, [3.0, 5.0, 7.0])

    def test_quadratic(self):
        x = np.array([0.0, 1.0, 2.0])
        result = _eval_model(x, "quadratic", [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, [3.0, 6.0, 11.0])

    def test_none_returns_x(self):
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(_eval_model(x, "none", []), x)


class TestFitModel:
    def test_fit_shift(self):
        x = np.array([1.0, 2.0, 3.0])
        y = x + 0.5
        coef = _fit_model(x, y, "shift")
        assert coef is not None
        assert coef[0] == pytest.approx(0.5)

    def test_fit_affine(self):
        x = np.linspace(0, 10, 20)
        y = 1.5 * x + 0.3
        coef = _fit_model(x, y, "affine")
        assert coef is not None
        assert coef[0] == pytest.approx(1.5, abs=1e-6)
        assert coef[1] == pytest.approx(0.3, abs=1e-6)

    def test_fit_quadratic(self):
        x = np.linspace(-5, 5, 30)
        y = 0.5 * x**2 + 1.0 * x + 2.0
        coef = _fit_model(x, y, "quadratic")
        assert coef is not None
        assert coef[0] == pytest.approx(0.5, abs=1e-6)
        assert coef[1] == pytest.approx(1.0, abs=1e-6)
        assert coef[2] == pytest.approx(2.0, abs=1e-6)

    def test_fit_insufficient_points_returns_none(self):
        x = np.array([1.0])
        y = np.array([1.0])
        assert _fit_model(x, y, "affine") is None
        assert _fit_model(x, y, "quadratic") is None

    def test_fit_with_weights(self):
        x = np.array([1.0, 2.0, 3.0, 100.0])
        y = np.array([1.5, 2.5, 3.5, 0.0])
        weights = np.array([1.0, 1.0, 1.0, 1e-6])
        coef = _fit_model(x, y, "shift", weights=weights)
        assert coef is not None
        assert coef[0] == pytest.approx(0.5, abs=1e-3)

    def test_fit_unsupported_model_raises(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unsupported"):
            _fit_model(x, y, "cubic")  # type: ignore[arg-type]


class TestIsMonotonic:
    def test_identity_is_monotonic(self):
        wl = np.linspace(400.0, 500.0, 100)
        assert _is_monotonic_on_grid("shift", [0.0], wl) is True

    def test_positive_slope_affine_monotonic(self):
        wl = np.linspace(400.0, 500.0, 100)
        assert _is_monotonic_on_grid("affine", [1.001, 0.5], wl) is True

    def test_negative_slope_not_monotonic(self):
        wl = np.linspace(400.0, 500.0, 100)
        assert _is_monotonic_on_grid("affine", [-1.0, 1000.0], wl) is False

    def test_quadratic_can_be_non_monotonic(self):
        wl = np.linspace(-10.0, 10.0, 100)
        assert _is_monotonic_on_grid("quadratic", [-1.0, 0.0, 0.0], wl) is False


class TestDedupeOneToOne:
    def test_keeps_best_match_per_peak(self):
        residuals = np.array([0.1, 0.05, 0.2])
        peak_ids = np.array([0, 0, 1])
        line_ids = np.array([10, 11, 12])
        inlier_mask = np.array([True, True, True])
        selected = _dedupe_one_to_one(residuals, peak_ids, line_ids, inlier_mask)
        assert 1 in selected
        assert 0 not in selected
        assert 2 in selected

    def test_keeps_best_match_per_line(self):
        residuals = np.array([0.1, 0.05])
        peak_ids = np.array([0, 1])
        line_ids = np.array([10, 10])
        inlier_mask = np.array([True, True])
        selected = _dedupe_one_to_one(residuals, peak_ids, line_ids, inlier_mask)
        assert len(selected) == 1
        assert 1 in selected

    def test_empty_when_no_inliers(self):
        residuals = np.array([0.1, 0.2])
        peak_ids = np.array([0, 1])
        line_ids = np.array([0, 1])
        inlier_mask = np.array([False, False])
        selected = _dedupe_one_to_one(residuals, peak_ids, line_ids, inlier_mask)
        assert selected.size == 0


class TestComputeBIC:
    def test_finite_for_valid_inputs(self):
        bic = _compute_bic(rss=1.0, n=100, k=2)
        assert np.isfinite(bic)

    def test_inf_when_n_le_k(self):
        assert _compute_bic(rss=1.0, n=2, k=2) == float("inf")
        assert _compute_bic(rss=1.0, n=0, k=2) == float("inf")

    def test_finite_with_larger_n(self):
        bic_small_n = _compute_bic(rss=10.0, n=100, k=2)
        bic_large_n = _compute_bic(rss=100.0, n=1000, k=2)
        assert np.isfinite(bic_small_n)
        assert np.isfinite(bic_large_n)


class TestCalibrationResultDataclass:
    def test_construct(self):
        wl = np.linspace(400.0, 500.0, 10)
        result = WavelengthCalibrationResult(
            success=True,
            model="affine",
            coefficients=(1.0, 0.0),
            corrected_wavelength=wl,
            bic=10.0,
            rmse_nm=0.01,
            n_inliers=15,
            n_peaks=20,
            n_candidates=50,
            matched_peak_fraction=0.75,
            quality_passed=True,
            quality_reason="passed",
        )
        assert result.success is True
        assert result.model == "affine"
        assert result.details == {}

    def test_details_default_factory_independent(self):
        wl = np.empty(0)
        r1 = WavelengthCalibrationResult(
            success=False, model="none", coefficients=(),
            corrected_wavelength=wl, bic=float("inf"), rmse_nm=float("inf"),
            n_inliers=0, n_peaks=0, n_candidates=0,
            matched_peak_fraction=0.0, quality_passed=False, quality_reason="x",
        )
        r2 = WavelengthCalibrationResult(
            success=False, model="none", coefficients=(),
            corrected_wavelength=wl, bic=float("inf"), rmse_nm=float("inf"),
            n_inliers=0, n_peaks=0, n_candidates=0,
            matched_peak_fraction=0.0, quality_passed=False, quality_reason="x",
        )
        r1.details["foo"] = "bar"
        assert "foo" not in r2.details
