"""Tests for cflibs.inversion.preprocess.wavelength_calibration helpers."""

import numpy as np
import pytest

from cflibs.inversion.preprocess import wavelength_calibration as wcal_mod
from cflibs.inversion.preprocess.wavelength_calibration import (
    WavelengthCalibrationResult,
    _apply_segment_coverage_gate,
    _compute_bic,
    _coverage_extrapolation_nm,
    _dedupe_one_to_one,
    _dense_anchor_hull,
    _eval_model,
    _fit_model,
    _is_monotonic_on_grid,
    _model_min_points,
    _model_param_count,
    detect_ccd_seams,
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
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wl,
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=0,
            n_candidates=0,
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="x",
        )
        r2 = WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wl,
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=0,
            n_candidates=0,
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="x",
        )
        r1.details["foo"] = "bar"
        assert "foo" not in r2.details


class TestDetectCcdSeams:
    def test_uniform_axis_has_no_seams(self):
        wl = np.linspace(200.0, 900.0, 5000)
        seams = detect_ccd_seams(wl)
        assert seams.size == 0

    def test_single_real_seam_detected(self):
        # Two contiguous channels with a clear gap between them.
        left = np.arange(200.0, 300.0, 0.05)
        right = np.arange(305.0, 400.0, 0.05)  # 5 nm gap >> 0.05 nm step
        wl = np.concatenate([left, right])
        seams = detect_ccd_seams(wl)
        assert seams.size == 1
        # Seam index points at the last sample of the left channel.
        assert seams[0] == left.size - 1

    def test_gradual_dispersion_change_is_not_a_seam(self):
        # A within-axis dispersion change (blue dense, red coarse) must NOT be
        # shattered into thousands of false seams the way diff > k*median would.
        blue = np.arange(200.0, 400.0, 0.05)
        # Red channel sampled 5x coarser but still contiguous (no gap).
        red = np.arange(blue[-1] + 0.25, 700.0, 0.25)
        wl = np.concatenate([blue, red])
        seams = detect_ccd_seams(wl)
        # At most the single dispersion-change boundary, never thousands.
        assert seams.size <= 1

    def test_multiple_seams_detected(self):
        segs = []
        start = 200.0
        for _ in range(4):
            seg = np.arange(start, start + 50.0, 0.05)
            segs.append(seg)
            start = seg[-1] + 3.0  # 3 nm inter-channel gap
        wl = np.concatenate(segs)
        seams = detect_ccd_seams(wl)
        assert seams.size == 3

    def test_too_short_axis_returns_empty(self):
        assert detect_ccd_seams(np.array([200.0, 200.1])).size == 0

    def test_seam_indices_are_sorted_and_in_range(self):
        segs = [np.arange(200.0, 250.0, 0.05), np.arange(253.0, 300.0, 0.05)]
        wl = np.concatenate(segs)
        seams = detect_ccd_seams(wl)
        assert np.all(np.diff(seams) > 0) if seams.size > 1 else True
        assert np.all(seams >= 0)
        assert np.all(seams < wl.size - 1)


# ---------------------------------------------------------------------------
# Bead ye6t: affine coverage gate (never extrapolate a slope past its anchors)
# ---------------------------------------------------------------------------


def _result(model, coefficients, seg_wl, anchors, n_inliers=20, rmse=0.02):
    return WavelengthCalibrationResult(
        success=True,
        model=model,
        coefficients=tuple(coefficients),
        corrected_wavelength=wcal_mod._eval_model(seg_wl, model, coefficients),
        bic=-100.0,
        rmse_nm=rmse,
        n_inliers=n_inliers,
        n_peaks=50,
        n_candidates=80,
        matched_peak_fraction=0.5,
        quality_passed=True,
        quality_reason="passed",
        details={"inlier_anchor_wl_nm": sorted(float(a) for a in anchors)},
    )


class TestDenseAnchorHull:
    def test_uniform_anchors_cover_their_span(self):
        anchors = np.linspace(500.0, 900.0, 30)
        lo, hi = _dense_anchor_hull(anchors)
        # 80% of 30 = 24 anchors -> ~80% of the span.
        assert (hi - lo) / 400.0 == pytest.approx(0.8, abs=0.05)

    def test_stray_anchors_do_not_stretch_the_hull(self):
        """A handful of circularly-matched red anchors (the ChemCam VNIR
        failure mode) must not fake full-span coverage."""
        anchors = np.sort(np.concatenate([np.linspace(475.0, 650.0, 26), [854.3, 877.6, 892.6]]))
        lo, hi = _dense_anchor_hull(anchors)
        assert lo == pytest.approx(475.0)
        assert hi <= 660.0  # the dense mass, not the stray tail

    def test_empty_returns_nan(self):
        lo, hi = _dense_anchor_hull(np.array([]))
        assert np.isnan(lo) and np.isnan(hi)


class TestCoverageExtrapolation:
    def test_shift_model_never_extrapolates(self):
        seg_wl = np.linspace(473.0, 906.0, 100)
        assert _coverage_extrapolation_nm(seg_wl, "shift", (-0.1,), 475.0, 650.0) == 0.0

    def test_affine_drift_is_slope_times_uncovered_distance(self):
        seg_wl = np.linspace(473.0, 906.0, 100)
        a_minus_1 = -6.0e-4
        drift = _coverage_extrapolation_nm(seg_wl, "affine", (1.0 + a_minus_1, 0.2), 475.0, 650.0)
        assert drift == pytest.approx(abs(a_minus_1) * (906.0 - 650.0), rel=1e-6)

    def test_hull_covering_segment_has_no_drift(self):
        seg_wl = np.linspace(473.0, 906.0, 100)
        drift = _coverage_extrapolation_nm(seg_wl, "affine", (0.9994, 0.2), 473.0, 906.0)
        assert drift == pytest.approx(0.0, abs=1e-12)


class TestSegmentCoverageGate:
    """Rigged anchors spanning half the segment -> shift model chosen."""

    SEG_WL = np.linspace(473.2, 905.6, 2048)  # ~0.21 nm/px, like ChemCam VNIR

    def _gate(self, seg_cal, monkeypatch, shift_trusted=True):
        shift_result = _result("shift", (-0.13,), self.SEG_WL, [500.0, 600.0], n_inliers=30)

        def _fake_calibrate(**kwargs):
            assert tuple(kwargs["candidate_models"]) == ("shift",)
            return shift_result

        monkeypatch.setattr(wcal_mod, "calibrate_wavelength_axis", _fake_calibrate)
        return _apply_segment_coverage_gate(
            seg_cal,
            self.SEG_WL,
            np.ones_like(self.SEG_WL),
            atomic_db=None,
            elements=["Fe"],
            coverage_min_anchor_span_fraction=0.6,
            coverage_max_extrapolation_px=1.0,
            inlier_tolerance_nm=0.08,
            max_pair_window_nm=2.0,
            random_seed=42,
            calibrate_kwargs={},
        )

    def test_half_segment_anchors_degrade_affine_to_shift(self, monkeypatch):
        anchors = np.linspace(475.0, 650.0, 26).tolist() + [854.3, 877.6, 892.6]
        seg_cal = _result("affine", (0.999396, 0.1896), self.SEG_WL, anchors, n_inliers=29)
        gated, status, extrap = self._gate(seg_cal, monkeypatch)
        assert status == "degraded_to_shift"
        assert gated.model == "shift"
        assert extrap > 0.1  # the ~0.12 nm VNIR red-end drift

    def test_full_coverage_affine_passes(self, monkeypatch):
        anchors = np.linspace(475.0, 904.0, 30)
        seg_cal = _result("affine", (1.0 - 2.0e-5, 0.01), self.SEG_WL, anchors)
        gated, status, _extrap = self._gate(seg_cal, monkeypatch)
        assert status == "passed"
        assert gated is seg_cal
