"""
Robust wavelength-axis calibration helpers for real LIBS spectra.

This module estimates a calibration mapping from measured peak wavelengths to
reference transition wavelengths using robust line matching with outlier
rejection. Supported mappings:

- ``shift``: y = x + b
- ``affine``: y = a*x + b
- ``quadratic``: y = c2*x^2 + c1*x + c0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.preprocess.preprocessing import detect_peaks_auto

logger = get_logger("inversion.wavelength_calibration")

CalibrationModel = Literal["none", "shift", "affine", "quadratic"]


@dataclass
class WavelengthCalibrationResult:
    """
    Result of robust wavelength calibration.

    Attributes
    ----------
    success : bool
        Whether calibration was successfully estimated.
    model : CalibrationModel
        Calibration model used.
    coefficients : Tuple[float, ...]
        Fitted model coefficients.
    corrected_wavelength : np.ndarray
        Corrected wavelength axis.
    bic : float
        Bayesian information criterion for selected fit.
    rmse_nm : float
        Inlier RMSE in nm.
    n_inliers : int
        Number of robust inlier peak-line pairs.
    n_peaks : int
        Number of detected peaks in the measured spectrum.
    n_candidates : int
        Number of candidate peak-line pairs considered.
    matched_peak_fraction : float
        Fraction of detected peaks that matched candidate lines.
    quality_passed : bool
        Whether quality-gate criteria were satisfied.
    quality_reason : str
        Quality-gate pass/fail reason.
    details : Dict[str, Any]
        Additional diagnostics and quality metrics.
    """

    success: bool
    model: CalibrationModel
    coefficients: Tuple[float, ...]
    corrected_wavelength: np.ndarray
    bic: float
    rmse_nm: float
    n_inliers: int
    n_peaks: int
    n_candidates: int
    matched_peak_fraction: float
    quality_passed: bool
    quality_reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ModelFit:
    model: CalibrationModel
    coefficients: Tuple[float, ...]
    bic: float
    rmse_nm: float
    n_inliers: int
    inlier_indices: np.ndarray
    matched_peak_fraction: float


def _max_abs_correction_nm(
    wavelength: np.ndarray, model: CalibrationModel, coefficients: Sequence[float]
) -> float:
    correction = _eval_model(wavelength, model, coefficients) - wavelength
    return float(np.max(np.abs(correction))) if correction.size else 0.0


def _quality_gate_check(
    fit: _ModelFit,
    wavelength: np.ndarray,
    peak_wl: np.ndarray,
    peak_ids: np.ndarray,
    min_inliers: int,
    min_peak_match_fraction: float,
    max_rmse_nm: float,
    min_inlier_span_fraction: float,
    max_abs_correction_nm: float,
) -> Tuple[bool, str, Dict[str, float]]:
    inlier_peak_ids = (
        np.unique(peak_ids[fit.inlier_indices]) if fit.inlier_indices.size else np.array([])
    )
    peak_span_nm = float(np.ptp(peak_wl[inlier_peak_ids])) if inlier_peak_ids.size >= 2 else 0.0
    full_span_nm = max(float(np.ptp(wavelength)), 1e-9)
    inlier_span_fraction = peak_span_nm / full_span_nm
    max_abs_correction = _max_abs_correction_nm(wavelength, fit.model, fit.coefficients)

    metrics = {
        "quality_n_inliers": float(fit.n_inliers),
        "quality_peak_match_fraction": float(fit.matched_peak_fraction),
        "quality_rmse_nm": float(fit.rmse_nm),
        "quality_inlier_span_fraction": float(inlier_span_fraction),
        "quality_max_abs_correction_nm": float(max_abs_correction),
    }

    if fit.n_inliers < min_inliers:
        return False, "insufficient_inliers", metrics
    if fit.matched_peak_fraction < min_peak_match_fraction:
        return False, "low_peak_match_fraction", metrics
    if fit.rmse_nm > max_rmse_nm:
        return False, "rmse_too_high", metrics
    if inlier_span_fraction < min_inlier_span_fraction:
        return False, "insufficient_span", metrics
    if max_abs_correction > max_abs_correction_nm:
        return False, "correction_too_large", metrics
    return True, "passed", metrics


def _model_min_points(model: CalibrationModel) -> int:
    if model == "shift":
        return 1
    if model == "affine":
        return 2
    if model == "quadratic":
        return 3
    return 1


def _model_param_count(model: CalibrationModel) -> int:
    if model == "shift":
        return 1
    if model == "affine":
        return 2
    if model == "quadratic":
        return 3
    return 0


def _eval_model(x: np.ndarray, model: CalibrationModel, coef: Sequence[float]) -> np.ndarray:
    if model == "shift":
        return x + float(coef[0])
    if model == "affine":
        a, b = float(coef[0]), float(coef[1])
        return a * x + b
    if model == "quadratic":
        c2, c1, c0 = float(coef[0]), float(coef[1]), float(coef[2])
        return c2 * x * x + c1 * x + c0
    return x


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    model: CalibrationModel,
    weights: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, ...]]:
    if x.size < _model_min_points(model):
        return None

    if weights is None:
        weights = np.ones_like(x, dtype=float)
    weights = np.clip(np.asarray(weights, dtype=float), 1e-8, None)
    sqrt_w = np.sqrt(weights)

    try:
        if model == "shift":
            shift = float(np.average(y - x, weights=weights))
            return (shift,)
        if model == "affine":
            X = np.column_stack([x, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
            return (float(coef[0]), float(coef[1]))
        if model == "quadratic":
            X = np.column_stack([x * x, x, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
            return (float(coef[0]), float(coef[1]), float(coef[2]))
    except np.linalg.LinAlgError:
        return None
    raise ValueError(f"Unsupported calibration model: {model}")


def _is_monotonic_on_grid(
    model: CalibrationModel,
    coef: Sequence[float],
    wavelength: np.ndarray,
) -> bool:
    corrected = _eval_model(wavelength, model, coef)
    diffs = np.diff(corrected)
    return bool(np.all(diffs > 0))


def _dedupe_one_to_one(
    residuals: np.ndarray,
    peak_ids: np.ndarray,
    line_ids: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    idx = np.where(inlier_mask)[0]
    if idx.size == 0:
        return idx
    order = idx[np.argsort(residuals[idx])]
    used_peaks = set()
    used_lines = set()
    selected: List[int] = []
    for i in order:
        p = int(peak_ids[i])
        line_id = int(line_ids[i])
        if p in used_peaks or line_id in used_lines:
            continue
        used_peaks.add(p)
        used_lines.add(line_id)
        selected.append(int(i))
    return np.array(selected, dtype=int)


def _compute_bic(rss: float, n: int, k: int) -> float:
    if n <= k or n <= 0:
        return float("inf")
    rss = max(float(rss), 1e-12)
    return float(n * np.log(rss / n) + k * np.log(n))


def _select_sample_indices(
    rng: np.random.Generator,
    n_items: int,
    n_needed: int,
    x_vals: np.ndarray,
    probs: np.ndarray,
) -> Optional[np.ndarray]:
    if n_items < n_needed:
        return None
    for _ in range(25):
        idx = rng.choice(n_items, size=n_needed, replace=False, p=probs)
        if np.unique(x_vals[idx]).size >= n_needed:
            return np.asarray(idx, dtype=int)
    return None


def _ransac_search(
    model: CalibrationModel,
    x: np.ndarray,
    y: np.ndarray,
    peak_ids: np.ndarray,
    line_ids: np.ndarray,
    weights: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
    min_pts: int,
    inlier_tolerance_nm: float,
    iterations: int,
) -> Tuple[Optional[Tuple[float, ...]], np.ndarray]:
    """Run the RANSAC sampling loop, returning the best coefficients/inliers."""
    best_coef: Optional[Tuple[float, ...]] = None
    best_inliers: np.ndarray = np.array([], dtype=int)
    best_score = (-1, np.inf)  # maximize n_inliers, then minimize median residual

    for _ in range(iterations):
        sample = _select_sample_indices(rng, x.size, min_pts, x, probs)
        if sample is None:
            continue
        coef = _fit_model(x[sample], y[sample], model, weights=weights[sample])
        if coef is None:
            continue

        pred = _eval_model(x, model, coef)
        residual = np.abs(pred - y)
        inlier_mask = residual <= inlier_tolerance_nm
        selected = _dedupe_one_to_one(residual, peak_ids, line_ids, inlier_mask)
        n_in = int(selected.size)
        if n_in == 0:
            continue
        med = float(np.median(residual[selected]))
        score = (n_in, med)
        if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_coef = coef
            best_inliers = selected

    return best_coef, best_inliers


def _refine_robust_inliers(
    model: CalibrationModel,
    x: np.ndarray,
    y: np.ndarray,
    peak_ids: np.ndarray,
    line_ids: np.ndarray,
    weights: np.ndarray,
    best_inliers: np.ndarray,
    min_pts: int,
    inlier_tolerance_nm: float,
) -> Optional[np.ndarray]:
    """Refit on best inliers and recompute the robust inlier set."""
    refit_coef = _fit_model(x[best_inliers], y[best_inliers], model, weights=weights[best_inliers])
    if refit_coef is None:
        return None

    pred_all = _eval_model(x, model, refit_coef)
    residual_all = np.abs(pred_all - y)
    inlier_mask_all = residual_all <= inlier_tolerance_nm
    robust_inliers = _dedupe_one_to_one(residual_all, peak_ids, line_ids, inlier_mask_all)

    if robust_inliers.size < min_pts:
        return None
    return robust_inliers


def _build_model_fit(
    model: CalibrationModel,
    x: np.ndarray,
    y: np.ndarray,
    peak_ids: np.ndarray,
    weights: np.ndarray,
    robust_inliers: np.ndarray,
    n_peaks_total: int,
) -> Optional[_ModelFit]:
    """Final refine on robust inliers and assemble the :class:`_ModelFit`."""
    final_coef = _fit_model(
        x[robust_inliers], y[robust_inliers], model, weights=weights[robust_inliers]
    )
    if final_coef is None:
        return None

    pred = _eval_model(x[robust_inliers], model, final_coef)
    res = pred - y[robust_inliers]
    rmse = float(np.sqrt(np.mean(res**2)))
    rss = float(np.sum(res**2))
    bic = _compute_bic(rss, int(robust_inliers.size), _model_param_count(model))
    matched_peak_fraction = float(np.unique(peak_ids[robust_inliers]).size / max(1, n_peaks_total))

    return _ModelFit(
        model=model,
        coefficients=tuple(float(c) for c in final_coef),
        bic=bic,
        rmse_nm=rmse,
        n_inliers=int(robust_inliers.size),
        inlier_indices=robust_inliers,
        matched_peak_fraction=matched_peak_fraction,
    )


def _ransac_fit(
    model: CalibrationModel,
    x: np.ndarray,
    y: np.ndarray,
    peak_ids: np.ndarray,
    line_ids: np.ndarray,
    weights: np.ndarray,
    n_peaks_total: int,
    inlier_tolerance_nm: float,
    iterations: int,
    seed: int,
) -> Optional[_ModelFit]:
    min_pts = _model_min_points(model)
    if x.size < min_pts:
        return None

    rng = np.random.default_rng(seed + _model_param_count(model))
    probs = weights / np.sum(weights)

    best_coef, best_inliers = _ransac_search(
        model,
        x,
        y,
        peak_ids,
        line_ids,
        weights,
        probs,
        rng,
        min_pts,
        inlier_tolerance_nm,
        iterations,
    )
    if best_coef is None or best_inliers.size < min_pts:
        return None

    robust_inliers = _refine_robust_inliers(
        model, x, y, peak_ids, line_ids, weights, best_inliers, min_pts, inlier_tolerance_nm
    )
    if robust_inliers is None:
        return None

    return _build_model_fit(model, x, y, peak_ids, weights, robust_inliers, n_peaks_total)


def _build_reference_line_pool(
    atomic_db: AtomicDatabase,
    elements: Sequence[str],
    wavelength_min: float,
    wavelength_max: float,
    max_lines_per_element: int,
    min_aki_gk: float,
    reference_temperature_K: float,
) -> Tuple[np.ndarray, np.ndarray]:
    line_wl: List[float] = []
    line_strength: List[float] = []
    kT = KB_EV * reference_temperature_K

    for element in elements:
        transitions = atomic_db.get_transitions(
            element, wavelength_min=wavelength_min, wavelength_max=wavelength_max
        )
        if min_aki_gk > 0:
            transitions = [t for t in transitions if (t.A_ki * t.g_k) >= min_aki_gk]
        if not transitions:
            continue
        ranked = sorted(
            transitions,
            key=lambda t: (t.A_ki * t.g_k) * np.exp(-max(t.E_k_ev, 0.0) / max(kT, 1e-9)),
            reverse=True,
        )[:max_lines_per_element]
        for t in ranked:
            line_wl.append(float(t.wavelength_nm))
            # positive floor for weighting
            strength = max((t.A_ki * t.g_k) * np.exp(-max(t.E_k_ev, 0.0) / max(kT, 1e-9)), 1e-12)
            line_strength.append(float(strength))

    if not line_wl:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.asarray(line_wl, dtype=float), np.asarray(line_strength, dtype=float)


def _build_candidate_pairs(
    peak_wl: np.ndarray,
    peak_amp: np.ndarray,
    line_wl: np.ndarray,
    line_strength: np.ndarray,
    max_pair_window_nm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the candidate peak-line pair pool as parallel numpy arrays."""
    cand_x: List[float] = []
    cand_y: List[float] = []
    cand_peak_id: List[int] = []
    cand_line_id: List[int] = []
    cand_weight: List[float] = []

    line_strength_n = line_strength / max(np.max(line_strength), 1e-12)
    peak_amp_n = peak_amp / max(np.max(peak_amp), 1e-12)

    for p_i, (pw, pa) in enumerate(zip(peak_wl, peak_amp_n, strict=False)):
        delta = np.abs(line_wl - pw)
        hit = np.where(delta <= max_pair_window_nm)[0]
        for l_i in hit:
            cand_x.append(float(pw))
            cand_y.append(float(line_wl[l_i]))
            cand_peak_id.append(int(p_i))
            cand_line_id.append(int(l_i))
            # Blend peak prominence and line observability.
            w = np.sqrt(max(float(pa), 1e-6) * max(float(line_strength_n[l_i]), 1e-6))
            cand_weight.append(w)

    return (
        np.asarray(cand_x, dtype=float),
        np.asarray(cand_y, dtype=float),
        np.asarray(cand_peak_id, dtype=int),
        np.asarray(cand_line_id, dtype=int),
        np.asarray(cand_weight, dtype=float),
    )


def _fit_candidate_models(
    models: List[CalibrationModel],
    x: np.ndarray,
    y: np.ndarray,
    peak_ids: np.ndarray,
    line_ids: np.ndarray,
    weights: np.ndarray,
    wavelength: np.ndarray,
    n_peaks_total: int,
    inlier_tolerance_nm: float,
    ransac_iterations: int,
    random_seed: int,
) -> List[_ModelFit]:
    """RANSAC-fit each candidate model, keeping only monotonic fits."""
    fits: List[_ModelFit] = []
    for m in models:
        fit = _ransac_fit(
            model=m,
            x=x,
            y=y,
            peak_ids=peak_ids,
            line_ids=line_ids,
            weights=weights,
            n_peaks_total=n_peaks_total,
            inlier_tolerance_nm=inlier_tolerance_nm,
            iterations=ransac_iterations,
            seed=random_seed,
        )
        if fit is None:
            continue
        if not _is_monotonic_on_grid(m, fit.coefficients, wavelength):
            logger.debug("Rejected non-monotonic %s calibration fit: %s", m, fit.coefficients)
            continue
        fits.append(fit)
    return fits


def _assemble_calibration_result(
    fits: List[_ModelFit],
    best: _ModelFit,
    wavelength: np.ndarray,
    peak_wl: np.ndarray,
    line_wl: np.ndarray,
    peak_ids: np.ndarray,
    x: np.ndarray,
    apply_quality_gate: bool,
    quality_min_inliers: int,
    quality_min_peak_match_fraction: float,
    quality_max_rmse_nm: float,
    quality_min_inlier_span_fraction: float,
    quality_max_abs_correction_nm: float,
) -> WavelengthCalibrationResult:
    """Run the quality gate, build diagnostics, and assemble the final result."""
    quality_passed = True
    quality_reason = "passed"
    quality_metrics: Dict[str, float] = {}
    if apply_quality_gate:
        quality_passed, quality_reason, quality_metrics = _quality_gate_check(
            fit=best,
            wavelength=wavelength,
            peak_wl=peak_wl,
            peak_ids=peak_ids,
            min_inliers=quality_min_inliers,
            min_peak_match_fraction=quality_min_peak_match_fraction,
            max_rmse_nm=quality_max_rmse_nm,
            min_inlier_span_fraction=quality_min_inlier_span_fraction,
            max_abs_correction_nm=quality_max_abs_correction_nm,
        )

    corrected = (
        _eval_model(wavelength, best.model, best.coefficients)
        if quality_passed
        else wavelength.copy()
    )
    inlier_x = x[best.inlier_indices] if best.inlier_indices.size else np.array([], dtype=float)
    details = {
        "peak_count": float(peak_wl.size),
        "line_pool_size": float(line_wl.size),
        "candidate_count": float(x.size),
        # Inlier anchor positions (measured peak wavelengths of the robust
        # inliers). Consumers use these to gate slope models on anchor
        # *coverage*: a slope fitted from anchors spanning only part of the
        # axis extrapolates its dispersion correction past the anchored range
        # (audit bead ye6t: the ChemCam VNIR affine, anchored mostly at
        # 475-650 nm, was wrong by ~0.2 nm at 877-905 nm against the Ca II IR
        # triplet — and the few red "inliers" were the disputed contaminated
        # peaks themselves, so a min/max hull cannot detect the gap).
        "inlier_anchor_min_nm": float(np.min(inlier_x)) if inlier_x.size else float("nan"),
        "inlier_anchor_max_nm": float(np.max(inlier_x)) if inlier_x.size else float("nan"),
        "inlier_anchor_wl_nm": sorted(float(v) for v in inlier_x),
        "selected_model_bic": float(best.bic),
        "quality_gate_enabled": bool(apply_quality_gate),
        "quality_passed": bool(quality_passed),
        "quality_reason": quality_reason,
        "quality_min_inliers": float(quality_min_inliers),
        "quality_min_peak_match_fraction": float(quality_min_peak_match_fraction),
        "quality_max_rmse_nm": float(quality_max_rmse_nm),
        "quality_min_inlier_span_fraction": float(quality_min_inlier_span_fraction),
        "quality_max_abs_correction_nm": float(quality_max_abs_correction_nm),
    }
    for fit in fits:
        details[f"{fit.model}_bic"] = float(fit.bic)
        details[f"{fit.model}_rmse_nm"] = float(fit.rmse_nm)
        details[f"{fit.model}_n_inliers"] = float(fit.n_inliers)
    details.update(quality_metrics)

    return WavelengthCalibrationResult(
        success=True,
        model=best.model,
        coefficients=best.coefficients,
        corrected_wavelength=corrected,
        bic=float(best.bic),
        rmse_nm=float(best.rmse_nm),
        n_inliers=int(best.n_inliers),
        n_peaks=int(peak_wl.size),
        n_candidates=int(x.size),
        matched_peak_fraction=float(best.matched_peak_fraction),
        quality_passed=quality_passed,
        quality_reason=quality_reason,
        details=details,
    )


def calibrate_wavelength_axis(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: Sequence[str],
    mode: Literal["auto", "shift", "affine", "quadratic"] = "auto",
    candidate_models: Sequence[CalibrationModel] = ("shift", "affine", "quadratic"),
    max_pair_window_nm: float = 2.0,
    inlier_tolerance_nm: float = 0.08,
    max_lines_per_element: int = 60,
    min_aki_gk: float = 3e3,
    reference_temperature_K: float = 10000.0,
    threshold_factor: float = 4.0,
    ransac_iterations: int = 600,
    random_seed: int = 42,
    apply_quality_gate: bool = True,
    quality_min_inliers: int = 12,
    quality_min_peak_match_fraction: float = 0.0,
    quality_max_rmse_nm: float = 0.10,
    quality_min_inlier_span_fraction: float = 0.25,
    quality_max_abs_correction_nm: float = 2.5,
) -> WavelengthCalibrationResult:
    """
    Estimate and apply robust wavelength calibration using matched strong lines.

    Parameters
    ----------
    wavelength, intensity
        Spectrum axis and intensity.
    atomic_db : AtomicDatabase
        Atomic transition database.
    elements : Sequence[str]
        Candidate elements to build reference line pool.
    mode : {"auto", "shift", "affine", "quadratic"}
        Calibration model selection mode.
    candidate_models : sequence
        Candidate models considered when ``mode='auto'``.
    max_pair_window_nm : float
        Candidate pair window |line - peak| <= window (nm).
    inlier_tolerance_nm : float
        Inlier threshold for robust fitting.
    max_lines_per_element : int
        Strong-line cap per element.
    min_aki_gk : float
        Minimum line observability filter.
    reference_temperature_K : float
        Reference temperature for line ranking.
    threshold_factor : float
        Peak detection threshold for candidate peaks.
    ransac_iterations : int
        Number of robust fit iterations per candidate model.
    random_seed : int
        RNG seed for reproducibility.
    apply_quality_gate : bool
        If true, reject fits that fail quality thresholds.
    quality_min_inliers : int
        Minimum robust inlier pairs required.
    quality_min_peak_match_fraction : float
        Minimum fraction of detected peaks that must be matched. Defaults to
        ``0.0`` (disabled): in a dense real LIBS spectrum the large majority of
        detected peaks are legitimately unmatched (continuum ripple, molecular
        bands, weak/absent reference lines, off-list elements), so requiring a
        high matched fraction rejects perfectly good calibrations (it killed the
        CSA BHVO-2 fit at 0.35 and forced a no-op). Robust confidence is instead
        governed by ``quality_min_inliers`` + ``quality_max_rmse_nm``. Raise this
        only for clean synthetic spectra where a high match fraction is expected.
    quality_max_rmse_nm : float
        Maximum inlier RMSE allowed for accepted calibration.
    quality_min_inlier_span_fraction : float
        Minimum inlier peak span as a fraction of full wavelength span.
    quality_max_abs_correction_nm : float
        Maximum absolute correction magnitude over the wavelength axis.

    Returns
    -------
    WavelengthCalibrationResult
        Calibration result with corrected wavelength axis, fit diagnostics,
        and quality-gate outcome.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if wavelength.size < 4 or intensity.size < 4:
        return WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wavelength.copy(),
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=0,
            n_candidates=0,
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="spectrum_too_short",
            details={"reason": "spectrum_too_short"},
        )

    peaks, _baseline, _noise = detect_peaks_auto(
        wavelength, intensity, threshold_factor=threshold_factor
    )
    if not peaks:
        return WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wavelength.copy(),
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=0,
            n_candidates=0,
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="no_peaks_detected",
            details={"reason": "no_peaks_detected"},
        )

    peak_idx = np.asarray([p[0] for p in peaks], dtype=int)
    peak_wl = np.asarray([p[1] for p in peaks], dtype=float)
    peak_amp = np.maximum(intensity[peak_idx], 1e-12)

    line_wl, line_strength = _build_reference_line_pool(
        atomic_db=atomic_db,
        elements=elements,
        wavelength_min=float(np.min(wavelength)) - max_pair_window_nm,
        wavelength_max=float(np.max(wavelength)) + max_pair_window_nm,
        max_lines_per_element=max_lines_per_element,
        min_aki_gk=min_aki_gk,
        reference_temperature_K=reference_temperature_K,
    )

    if line_wl.size == 0:
        return WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wavelength.copy(),
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=int(peak_wl.size),
            n_candidates=0,
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="no_reference_lines",
            details={"reason": "no_reference_lines"},
        )

    # Build candidate pair pool
    x, y, peak_ids, line_ids, weights = _build_candidate_pairs(
        peak_wl, peak_amp, line_wl, line_strength, max_pair_window_nm
    )

    if x.size == 0:
        return WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wavelength.copy(),
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=int(peak_wl.size),
            n_candidates=0,
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="no_candidate_pairs",
            details={"reason": "no_candidate_pairs"},
        )

    if mode == "auto":
        models = [m for m in candidate_models if m in {"shift", "affine", "quadratic"}]
    else:
        models = [mode]

    fits = _fit_candidate_models(
        models=models,
        x=x,
        y=y,
        peak_ids=peak_ids,
        line_ids=line_ids,
        weights=weights,
        wavelength=wavelength,
        n_peaks_total=int(peak_wl.size),
        inlier_tolerance_nm=inlier_tolerance_nm,
        ransac_iterations=ransac_iterations,
        random_seed=random_seed,
    )

    if not fits:
        return WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wavelength.copy(),
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=int(peak_wl.size),
            n_candidates=int(x.size),
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason="no_valid_model_fit",
            details={"reason": "no_valid_model_fit"},
        )

    fits.sort(key=lambda f: f.bic)
    best = fits[0]

    return _assemble_calibration_result(
        fits=fits,
        best=best,
        wavelength=wavelength,
        peak_wl=peak_wl,
        line_wl=line_wl,
        peak_ids=peak_ids,
        x=x,
        apply_quality_gate=apply_quality_gate,
        quality_min_inliers=quality_min_inliers,
        quality_min_peak_match_fraction=quality_min_peak_match_fraction,
        quality_max_rmse_nm=quality_max_rmse_nm,
        quality_min_inlier_span_fraction=quality_min_inlier_span_fraction,
        quality_max_abs_correction_nm=quality_max_abs_correction_nm,
    )


def detect_ccd_seams(
    wavelength: np.ndarray,
    ratio_threshold: float = 3.0,
    window: int = 51,
    min_local_window: int = 5,
) -> np.ndarray:
    """
    Detect CCD/spectrometer seam indices in a stitched wavelength axis.

    Many real LIBS instruments stitch several detectors (or spectrometer
    channels) into one wavelength axis. Each channel can carry an independent
    dispersion error, so a single global calibration model cannot represent the
    piecewise mapping. Seams appear as a sample-to-sample gap ``wl[i+1]-wl[i]``
    that is large *relative to the local dispersion*, distinct from a gradual
    within-channel dispersion change.

    A simple ``diff > k * median(diff)`` test fails on instruments whose
    dispersion varies across the axis (e.g. a red channel with much coarser
    sampling than the blue): the global median is dominated by the dense channel
    and the coarse channel is shattered into thousands of false seams. This
    detector instead compares each gap to a *local* rolling median of the
    neighbouring gaps, so a smoothly varying dispersion produces ratio ~1 and
    only true discontinuities exceed ``ratio_threshold``.

    Parameters
    ----------
    wavelength : np.ndarray
        Monotonic wavelength axis (nm).
    ratio_threshold : float
        A gap is a seam when ``gap / local_median_gap > ratio_threshold``.
    window : int
        Half-width (in samples) of the local rolling-median window over gaps.
    min_local_window : int
        Minimum number of neighbouring gaps required to estimate a local median.

    Returns
    -------
    np.ndarray
        Sorted array of integer seam indices ``i`` such that a discontinuity
        occurs between samples ``i`` and ``i+1`` (empty if none / too short).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.size < max(3, 2 * min_local_window):
        return np.array([], dtype=int)

    dl = np.diff(wavelength)
    if dl.size == 0:
        return np.array([], dtype=int)

    n = dl.size
    local_med = np.empty(n, dtype=float)
    w = max(int(window), int(min_local_window))
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        local_med[i] = np.median(dl[lo:hi])

    local_med = np.maximum(local_med, 1e-12)
    ratio = dl / local_med
    seams = np.where(ratio > float(ratio_threshold))[0]
    return np.asarray(seams, dtype=int)


def _no_op_result(wavelength: np.ndarray, reason: str) -> WavelengthCalibrationResult:
    return WavelengthCalibrationResult(
        success=False,
        model="none",
        coefficients=(),
        corrected_wavelength=np.asarray(wavelength, dtype=float).copy(),
        bic=float("inf"),
        rmse_nm=float("inf"),
        n_inliers=0,
        n_peaks=0,
        n_candidates=0,
        matched_peak_fraction=0.0,
        quality_passed=False,
        quality_reason=reason,
        details={"reason": reason},
    )


#: Fraction of inlier anchors that must fit inside the dense anchor hull
#: (shortest interval). 0.8 tolerates up to 20% stray/circular anchors while
#: still keying the hull on where the anchors actually concentrate.
_COVERAGE_DENSE_HULL_ALPHA = 0.8


def _dense_anchor_hull(
    anchors_sorted: np.ndarray, alpha: float = _COVERAGE_DENSE_HULL_ALPHA
) -> Tuple[float, float]:
    """Shortest interval containing ``ceil(alpha * n)`` of the sorted anchors.

    A robust anchor hull: the plain min/max hull is stretched arbitrarily by a
    handful of stray inliers (on ChemCam BHVO-2 the VNIR affine's few red
    "anchors" are the disputed contaminated peaks themselves, matched
    circularly), whereas the highest-density interval tracks where the anchor
    mass actually sits (475-650 nm).
    """
    n = int(anchors_sorted.size)
    if n == 0:
        return float("nan"), float("nan")
    k = max(int(np.ceil(alpha * n)), 1)
    if k >= n:
        return float(anchors_sorted[0]), float(anchors_sorted[-1])
    widths = anchors_sorted[k - 1 :] - anchors_sorted[: n - k + 1]
    i = int(np.argmin(widths))
    return float(anchors_sorted[i]), float(anchors_sorted[i + k - 1])


def _coverage_extrapolation_nm(
    seg_wl: np.ndarray,
    model: CalibrationModel,
    coefficients: Sequence[float],
    hull_lo_nm: float,
    hull_hi_nm: float,
) -> float:
    """Worst-case correction drift from the inlier-anchor hull to a segment edge.

    For a slope-bearing model (``affine``/``quadratic``) the fitted correction
    ``corr(x) = model(x) - x`` keeps changing outside the inlier anchor hull,
    i.e. the dispersion slope is *extrapolated* past its anchors. This returns
    ``max(|corr(edge) - corr(nearest hull edge)|)`` over both segment ends —
    the extra correction the model invents where it has no anchors. A pure
    ``shift`` model (constant correction) always returns 0. Hull edges are
    clamped into the segment so a hull that covers an edge contributes 0.
    """
    if model == "shift" or seg_wl.size == 0:
        return 0.0
    lo, hi = float(seg_wl[0]), float(seg_wl[-1])
    if not (np.isfinite(hull_lo_nm) and np.isfinite(hull_hi_nm)):
        return float("inf")
    drift = 0.0
    for edge, hull in ((lo, min(max(hull_lo_nm, lo), hi)), (hi, min(max(hull_hi_nm, lo), hi))):
        corr_edge = float(_eval_model(np.asarray([edge]), model, coefficients)[0]) - edge
        corr_hull = float(_eval_model(np.asarray([hull]), model, coefficients)[0]) - hull
        drift = max(drift, abs(corr_edge - corr_hull))
    return drift


@dataclass
class _SegmentOutcome:
    status: str
    diag: Dict[str, Any]
    n_inliers: int
    rmse_nm: Optional[float]


def _segment_fit_trusted(
    seg_cal: WavelengthCalibrationResult,
    segment_min_inliers: int,
    segment_max_rmse_nm: float,
) -> bool:
    """Per-segment confidence gate (inlier count + inlier RMSE)."""
    return (
        seg_cal.success
        and seg_cal.model != "none"
        and int(seg_cal.n_inliers) >= segment_min_inliers
        and float(seg_cal.rmse_nm) <= segment_max_rmse_nm
    )


def _segment_anchor_coverage(
    seg_cal: WavelengthCalibrationResult,
    seg_wl: np.ndarray,
) -> Tuple[float, float, float]:
    """Dense-hull anchor coverage of a segment fit.

    Returns ``(span_fraction, extrapolation_nm, extrapolation_px)``:
    the dense anchor hull's width as a fraction of the segment span, the
    implied correction drift from the dense-hull edges to the segment edges,
    and that drift in local pixels.
    """
    anchors = np.asarray(seg_cal.details.get("inlier_anchor_wl_nm", []), dtype=float)
    hull_lo, hull_hi = _dense_anchor_hull(np.sort(anchors))
    seg_span = max(float(seg_wl[-1] - seg_wl[0]), 1e-9) if seg_wl.size >= 2 else 1e-9
    span_fraction = (
        (hull_hi - hull_lo) / seg_span if np.isfinite(hull_lo) and np.isfinite(hull_hi) else 0.0
    )
    extrap_nm = _coverage_extrapolation_nm(
        seg_wl, seg_cal.model, seg_cal.coefficients, hull_lo, hull_hi
    )
    local_px = float(np.median(np.diff(seg_wl))) if seg_wl.size >= 2 else 0.0
    extrap_px = extrap_nm / max(local_px, 1e-9)
    return float(span_fraction), float(extrap_nm), float(extrap_px)


def _apply_segment_coverage_gate(
    seg_cal: WavelengthCalibrationResult,
    seg_wl: np.ndarray,
    seg_in: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: Sequence[str],
    *,
    coverage_min_anchor_span_fraction: float,
    coverage_max_extrapolation_px: float,
    inlier_tolerance_nm: float,
    max_pair_window_nm: float,
    random_seed: int,
    calibrate_kwargs: Dict[str, Any],
) -> Tuple[WavelengthCalibrationResult, str, float]:
    """Degrade a slope model to ``shift`` when its anchors do not cover the segment.

    Never extrapolate a dispersion slope past its anchors (audit bead ye6t):
    the ChemCam VNIR affine, anchored mostly at 475-650 nm, overcorrected the
    red end by ~0.2 nm at 877-905 nm, flipping the Al I 877 doublet to the
    wrong member. A slope model is kept only when its *dense* inlier-anchor
    hull (shortest interval holding 80% of anchors — robust against the
    handful of circularly-matched stray anchors that stretch a min/max hull)
    covers at least ``coverage_min_anchor_span_fraction`` of the segment AND
    the implied correction drift past the hull stays within
    ``coverage_max_extrapolation_px`` local pixels. Otherwise the segment is
    refit with a pure ``shift`` model (a constant correction is safe to extend
    beyond its anchors).

    Returns ``(seg_cal, coverage_status, extrapolation_nm)``; ``seg_cal`` is
    the (possibly replaced) calibration whose trust gates the caller re-checks.
    """
    span_fraction, extrap_nm, extrap_px = _segment_anchor_coverage(seg_cal, seg_wl)
    if (
        span_fraction >= coverage_min_anchor_span_fraction
        and extrap_px <= coverage_max_extrapolation_px
    ):
        return seg_cal, "passed", extrap_nm

    shift_cal = calibrate_wavelength_axis(
        wavelength=seg_wl,
        intensity=seg_in,
        atomic_db=atomic_db,
        elements=elements,
        mode="auto",
        candidate_models=("shift",),
        inlier_tolerance_nm=inlier_tolerance_nm,
        max_pair_window_nm=max_pair_window_nm,
        apply_quality_gate=False,
        random_seed=random_seed,
        **calibrate_kwargs,
    )
    logger.info(
        "Segment coverage gate: %s model anchors cover %.0f%% of the segment "
        "(min %.0f%%) with %.3f nm (%.2f px) extrapolated correction drift "
        "past the dense anchor hull (max %.2f px); degraded to shift model.",
        seg_cal.model,
        100.0 * span_fraction,
        100.0 * coverage_min_anchor_span_fraction,
        extrap_nm,
        extrap_px,
        coverage_max_extrapolation_px,
    )
    return shift_cal, "degraded_to_shift", extrap_nm


def _fit_one_segment(
    index: int,
    a: int,
    b: int,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    corrected: np.ndarray,
    global_offset: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: Sequence[str],
    *,
    min_segment_points: int,
    sparse_segment_points: int,
    sparse_segment_max_models: Sequence[CalibrationModel],
    candidate_models: Sequence[CalibrationModel],
    inlier_tolerance_nm: float,
    max_pair_window_nm: float,
    segment_min_inliers: int,
    segment_max_rmse_nm: float,
    segment_max_global_disagreement_nm: float,
    affine_coverage_gate: bool,
    coverage_min_anchor_span_fraction: float,
    coverage_max_extrapolation_px: float,
    random_seed: int,
    calibrate_kwargs: Dict[str, Any],
) -> _SegmentOutcome:
    """Calibrate one segment, writing its correction into ``corrected[a:b]``."""
    seg_wl = wavelength[a:b]
    seg_in = intensity[a:b]
    n_pts = int(seg_wl.size)
    status = "global"
    seg_model = "none"
    seg_n_in = 0
    seg_rmse = float("inf")
    accepted_inliers = 0
    accepted_rmse: Optional[float] = None
    coverage_status = "not_applicable"
    coverage_extrap_nm = 0.0
    global_disagreement_nm = 0.0

    if n_pts >= min_segment_points:
        models = sparse_segment_max_models if n_pts < sparse_segment_points else candidate_models
        seg_cal = calibrate_wavelength_axis(
            wavelength=seg_wl,
            intensity=seg_in,
            atomic_db=atomic_db,
            elements=elements,
            mode="auto",
            candidate_models=tuple(models),
            inlier_tolerance_nm=inlier_tolerance_nm,
            max_pair_window_nm=max_pair_window_nm,
            apply_quality_gate=False,
            random_seed=random_seed + index,
            **calibrate_kwargs,
        )
        trusted = _segment_fit_trusted(seg_cal, segment_min_inliers, segment_max_rmse_nm)
        if trusted and affine_coverage_gate and seg_cal.model != "shift":
            seg_cal, coverage_status, coverage_extrap_nm = _apply_segment_coverage_gate(
                seg_cal,
                seg_wl,
                seg_in,
                atomic_db,
                elements,
                coverage_min_anchor_span_fraction=coverage_min_anchor_span_fraction,
                coverage_max_extrapolation_px=coverage_max_extrapolation_px,
                inlier_tolerance_nm=inlier_tolerance_nm,
                max_pair_window_nm=max_pair_window_nm,
                random_seed=random_seed + index,
                calibrate_kwargs=calibrate_kwargs,
            )
            trusted = _segment_fit_trusted(seg_cal, segment_min_inliers, segment_max_rmse_nm)
            if coverage_status == "degraded_to_shift" and not trusted:
                coverage_status = "degraded_shift_untrusted"
        # Global-disagreement plausibility gate: a per-segment RANSAC fit over
        # a short channel against a dense line catalog can lock onto a
        # self-coherent but wrong registration ~1-2 nm away (excellent inlier
        # RMSE, wrong lines — measured on CSA planetary spectra: a 13-inlier
        # rmse-0.025 segment fit at -1.58 nm vs real channel offsets of
        # -0.03..-0.38 nm). The global fit is anchored across ALL channels, so
        # a trusted segment correction that departs from the global correction
        # by more than ``segment_max_global_disagreement_nm`` is a catalog
        # alias, not a real channel registration error; demote it to the
        # global fallback ("no segment worse than the global model").
        if trusted:
            seg_offset_med = float(np.median(seg_cal.corrected_wavelength - seg_wl))
            global_offset_med = float(np.median(global_offset[a:b]))
            global_disagreement_nm = abs(seg_offset_med - global_offset_med)
            if global_disagreement_nm > segment_max_global_disagreement_nm:
                trusted = False
                logger.info(
                    "Segment %d (%.1f-%.1f nm): fit offset %+.3f nm disagrees with "
                    "the global correction %+.3f nm by %.3f nm (max %.3f nm); "
                    "likely catalog alias -- demoted to global fallback.",
                    index,
                    float(seg_wl.min()),
                    float(seg_wl.max()),
                    seg_offset_med,
                    global_offset_med,
                    global_disagreement_nm,
                    segment_max_global_disagreement_nm,
                )
        seg_model = seg_cal.model
        seg_n_in = int(seg_cal.n_inliers)
        seg_rmse = float(seg_cal.rmse_nm)
        if trusted:
            corrected[a:b] = seg_cal.corrected_wavelength
            status = "fit"
            accepted_inliers = seg_n_in
            accepted_rmse = seg_rmse

    if status != "fit":
        # Fallback 1: global single-axis correction over this segment.
        corrected[a:b] = seg_wl + global_offset[a:b]

    diag = {
        "index": index,
        "wl_min": float(seg_wl.min()) if n_pts else 0.0,
        "wl_max": float(seg_wl.max()) if n_pts else 0.0,
        "n_points": n_pts,
        "model": seg_model,
        "n_inliers": seg_n_in,
        "rmse_nm": seg_rmse,
        "status": status,
        "coverage_gate": coverage_status,
        "coverage_extrapolation_nm": float(coverage_extrap_nm),
        "global_disagreement_nm": float(global_disagreement_nm),
    }
    return _SegmentOutcome(
        status=status, diag=diag, n_inliers=accepted_inliers, rmse_nm=accepted_rmse
    )


def _run_segments(
    bounds: List[int],
    wavelength: np.ndarray,
    intensity: np.ndarray,
    corrected: np.ndarray,
    global_offset: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: Sequence[str],
    *,
    min_segment_points: int,
    sparse_segment_points: int,
    sparse_segment_max_models: Sequence[CalibrationModel],
    candidate_models: Sequence[CalibrationModel],
    inlier_tolerance_nm: float,
    max_pair_window_nm: float,
    segment_min_inliers: int,
    segment_max_rmse_nm: float,
    segment_max_global_disagreement_nm: float,
    affine_coverage_gate: bool,
    coverage_min_anchor_span_fraction: float,
    coverage_max_extrapolation_px: float,
    random_seed: int,
    calibrate_kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], int, List[float]]:
    """Calibrate every segment, accumulating diagnostics and inlier stats."""
    seg_diag: List[Dict[str, Any]] = []
    seg_status: List[str] = []  # "fit" | "global" | "neighbor" per segment
    total_inliers = 0
    rmse_accum: List[float] = []

    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        outcome = _fit_one_segment(
            i,
            a,
            b,
            wavelength,
            intensity,
            corrected,
            global_offset,
            atomic_db,
            elements,
            min_segment_points=min_segment_points,
            sparse_segment_points=sparse_segment_points,
            sparse_segment_max_models=sparse_segment_max_models,
            candidate_models=candidate_models,
            inlier_tolerance_nm=inlier_tolerance_nm,
            max_pair_window_nm=max_pair_window_nm,
            segment_min_inliers=segment_min_inliers,
            segment_max_rmse_nm=segment_max_rmse_nm,
            segment_max_global_disagreement_nm=segment_max_global_disagreement_nm,
            affine_coverage_gate=affine_coverage_gate,
            coverage_min_anchor_span_fraction=coverage_min_anchor_span_fraction,
            coverage_max_extrapolation_px=coverage_max_extrapolation_px,
            random_seed=random_seed,
            calibrate_kwargs=calibrate_kwargs,
        )
        if outcome.status == "fit":
            total_inliers += outcome.n_inliers
            if outcome.rmse_nm is not None:
                rmse_accum.append(outcome.rmse_nm)
        seg_diag.append(outcome.diag)
        seg_status.append(outcome.status)

    return seg_diag, seg_status, total_inliers, rmse_accum


def _apply_neighbor_fallback(
    bounds: List[int],
    wavelength: np.ndarray,
    corrected: np.ndarray,
    seg_diag: List[Dict[str, Any]],
    seg_status: List[str],
) -> None:
    """Fallback 2: borrow the nearest accepted-fit neighbour's median offset."""
    fit_idx = [i for i, s in enumerate(seg_status) if s == "fit"]
    if not fit_idx:
        return
    for i, s in enumerate(seg_status):
        if s == "fit":
            continue
        nearest = min(fit_idx, key=lambda j: abs(j - i))
        a, b = bounds[i], bounds[i + 1]
        na, nb = bounds[nearest], bounds[nearest + 1]
        neighbor_offset = float(np.median(corrected[na:nb] - wavelength[na:nb]))
        corrected[a:b] = wavelength[a:b] + neighbor_offset
        seg_diag[i]["status"] = "neighbor"
        seg_status[i] = "neighbor"


def _restore_seam_monotonicity(
    bounds: List[int], corrected: np.ndarray
) -> Tuple[int, float, float]:
    """Shift downstream segments up to remove seam-boundary overlaps."""
    n_clamped = 0
    max_clamp_nm = 0.0
    cumulative_shift = 0.0
    for k in range(1, len(bounds) - 1):
        prev_end = bounds[k] - 1  # last sample of previous segment
        cur_start = bounds[k]  # first sample of this segment
        deficit = corrected[prev_end] - corrected[cur_start]
        if deficit >= 0:
            shift = deficit + 1e-6
            cumulative_shift += shift
            corrected[bounds[k] : bounds[k + 1]] += shift
            max_clamp_nm = max(max_clamp_nm, shift)
            n_clamped += 1
    return n_clamped, max_clamp_nm, cumulative_shift


def _revert_segmented_to_global(
    global_result: WavelengthCalibrationResult,
    bounds: List[int],
    seams: np.ndarray,
    reason: str,
) -> WavelengthCalibrationResult:
    """Tag ``global_result`` with revert diagnostics and return it."""
    global_result.details["segments"] = len(bounds) - 1
    global_result.details["seam_count"] = int(seams.size)
    global_result.details["segmented_reverted"] = reason
    return global_result


def _build_segmented_result(
    global_result: WavelengthCalibrationResult,
    corrected: np.ndarray,
    bounds: List[int],
    seams: np.ndarray,
    seg_diag: List[Dict[str, Any]],
    seg_status: List[str],
    total_inliers: int,
    rmse_accum: List[float],
    n_clamped: int,
    max_clamp_nm: float,
    segment_min_inliers: int,
    segment_max_rmse_nm: float,
) -> WavelengthCalibrationResult:
    """Assemble the aggregate segmented calibration result."""
    n_fit = sum(1 for s in seg_status if s == "fit")
    n_seg = len(bounds) - 1
    agg_rmse = float(np.mean(rmse_accum)) if rmse_accum else global_result.rmse_nm

    # When EVERY segment fell back to the global correction, the stitched axis
    # IS the global fit -- so the result must inherit the global fit's quality
    # verdict, exactly as the seam-free path does. Declaring it failed (the
    # old behaviour) made callers discard a quality-passed global correction
    # and re-detect on the raw axis whenever the per-segment fits were all
    # demoted (e.g. by the catalog-alias disagreement gate).
    if n_fit > 0:
        quality_passed = True
        quality_reason = "segmented_passed"
    else:
        quality_passed = bool(global_result.quality_passed)
        quality_reason = f"segmented_all_fallback_global_{global_result.quality_reason}"
    details = {
        "segments": n_seg,
        "seam_count": int(seams.size),
        "n_segments_fit": n_fit,
        "n_segments_fallback": n_seg - n_fit,
        "global_model": global_result.model,
        "global_rmse_nm": float(global_result.rmse_nm),
        "segment_min_inliers": float(segment_min_inliers),
        "segment_max_rmse_nm": float(segment_max_rmse_nm),
        "seam_boundary_clamps": int(n_clamped),
        "max_seam_clamp_nm": float(max_clamp_nm),
        "segment_diagnostics": seg_diag,
    }

    return WavelengthCalibrationResult(
        success=True,
        model="segmented",
        coefficients=(),
        corrected_wavelength=corrected,
        bic=float(global_result.bic),
        rmse_nm=agg_rmse,
        n_inliers=int(total_inliers),
        n_peaks=int(global_result.n_peaks),
        n_candidates=int(global_result.n_candidates),
        matched_peak_fraction=float(global_result.matched_peak_fraction),
        quality_passed=quality_passed,
        quality_reason=quality_reason,
        details=details,
    )


def calibrate_wavelength_axis_segmented(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: Sequence[str],
    *,
    seam_ratio_threshold: float = 3.0,
    seam_window: int = 51,
    min_segment_points: int = 16,
    segment_min_inliers: int = 10,
    segment_max_rmse_nm: float = 0.06,
    segment_max_global_disagreement_nm: float = 0.5,
    inlier_tolerance_nm: float = 0.08,
    max_pair_window_nm: float = 2.0,
    candidate_models: Sequence[CalibrationModel] = ("shift", "affine"),
    sparse_segment_max_models: Sequence[CalibrationModel] = ("shift",),
    sparse_segment_points: int = 400,
    affine_coverage_gate: bool = True,
    coverage_min_anchor_span_fraction: float = 0.6,
    coverage_max_extrapolation_px: float = 1.0,
    random_seed: int = 42,
    fallback_to_global: bool = True,
    **calibrate_kwargs: Any,
) -> WavelengthCalibrationResult:
    """
    Per-segment robust wavelength calibration for stitched multi-channel axes.

    Detects CCD seams (:func:`detect_ccd_seams`) and calibrates each segment
    independently with the existing RANSAC calibrator. Each segment's fit is
    accepted only when it clears per-segment confidence gates
    (``segment_min_inliers`` inliers AND inlier RMSE ``<= segment_max_rmse_nm``);
    otherwise the segment falls back, in order, to (1) the global single-model
    fit over the whole axis, and (2) the median per-sample correction of the
    nearest accepted neighbour segment. This guarantees no segment is left
    *worse* than the global model and that under-populated channels are never
    corrupted by an overfit local model.

    A single-segment axis (no seams) degrades to the ordinary global
    :func:`calibrate_wavelength_axis` (auto mode), so this is safe to call on
    instruments that do not need segmentation.

    Parameters
    ----------
    wavelength, intensity
        Spectrum axis and intensity.
    atomic_db : AtomicDatabase
        Atomic transition database.
    elements : sequence of str
        Candidate elements for the reference line pool.
    seam_ratio_threshold, seam_window
        Seam-detector parameters (see :func:`detect_ccd_seams`).
    min_segment_points : int
        Segments shorter than this are not fit independently; they take the
        fallback correction directly.
    segment_min_inliers : int
        Minimum robust inlier pairs for a segment fit to be trusted.
    segment_max_rmse_nm : float
        Maximum inlier RMSE for a segment fit to be trusted.
    segment_max_global_disagreement_nm : float
        Maximum allowed difference between a trusted segment fit's median
        correction and the global fit's median correction over the same
        samples (default 0.5 nm). A short channel fit against a dense line
        catalog can lock onto a self-coherent but *wrong* registration 1-2 nm
        away (excellent inlier RMSE on the wrong lines); real channel
        registration errors on stitched instruments are a few local pixels.
        Segments beyond this bound are demoted to the global fallback offset
        instead of being trusted (and instead of later tearing the stitched
        axis apart at the seams, which previously forced a wholesale revert
        of *all* segment fits). When the global fit failed, the bound acts as
        a maximum absolute segment correction.
    inlier_tolerance_nm, max_pair_window_nm
        Robust-fit tolerances passed through to the per-segment calibrator.
    candidate_models : sequence
        Models tried for well-populated segments. Quadratic is intentionally
        excluded by default: a free quadratic over a short channel overfits and
        produces wild corrections at the band edges.
    sparse_segment_max_models : sequence
        Models tried for sparse segments (``< sparse_segment_points``). Kept to
        ``shift`` only to avoid slope/curvature overfitting on few points.
    sparse_segment_points : int
        Point-count threshold separating sparse from well-populated segments.
    affine_coverage_gate : bool
        If True (default), gate each accepted slope-model (``affine``/
        ``quadratic``) segment fit on inlier anchor *coverage*: when the
        dense anchor hull (shortest interval holding 80% of the inlier
        anchors) covers less than ``coverage_min_anchor_span_fraction`` of the
        segment, or the implied correction drift past the hull exceeds
        ``coverage_max_extrapolation_px`` local pixels, the segment is refit
        with a pure ``shift`` model. Never extrapolate a dispersion slope past
        its anchors (bead ye6t: the ChemCam VNIR affine, anchored mostly at
        475-650 nm, was wrong by ~0.2 nm at 877-905 nm).
    coverage_min_anchor_span_fraction : float
        Minimum dense-anchor-hull width as a fraction of the segment span for
        a slope model to be kept (default 0.6; a uniformly anchored segment
        measures ~0.8, the defective ChemCam VNIR affine ~0.53).
    coverage_max_extrapolation_px : float
        Maximum tolerated dense-hull-to-segment-edge correction drift, in
        local pixels (default 1.0).
    fallback_to_global : bool
        If True, low-confidence segments use the global single-axis fit before
        falling back to a neighbour offset.
    **calibrate_kwargs
        Extra keyword args forwarded to :func:`calibrate_wavelength_axis`.

    Returns
    -------
    WavelengthCalibrationResult
        Aggregate result. ``corrected_wavelength`` is the piecewise-corrected
        axis; ``model`` is ``"segmented"`` when more than one segment was
        present, else the single global model. ``details["segments"]`` carries
        per-segment diagnostics.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if wavelength.size < 4 or intensity.size < 4:
        return _no_op_result(wavelength, "spectrum_too_short")

    seams = detect_ccd_seams(wavelength, ratio_threshold=seam_ratio_threshold, window=seam_window)

    # Always compute a global single-axis fit. For single-segment axes this IS
    # the answer; for multi-segment axes it is the per-segment fallback.
    global_result = calibrate_wavelength_axis(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=elements,
        mode="auto",
        candidate_models=tuple(candidate_models),
        inlier_tolerance_nm=inlier_tolerance_nm,
        max_pair_window_nm=max_pair_window_nm,
        random_seed=random_seed,
        **calibrate_kwargs,
    )

    # ye6t coverage gate on the GLOBAL fit. The global fit is both the
    # seam-free answer and the per-segment fallback offset source, so an
    # affine anchored in one region of the axis must be degraded HERE —
    # otherwise its extrapolated correction drift leaks through both paths
    # (single-channel instruments hit the same failure class as the ChemCam
    # VNIR segment: anchors at 420-600 nm extrapolating ~1 px at the red end).
    if affine_coverage_gate and global_result.success and global_result.model != "shift":
        span_fraction, extrap_nm, extrap_px = _segment_anchor_coverage(global_result, wavelength)
        if (
            span_fraction < coverage_min_anchor_span_fraction
            or extrap_px > coverage_max_extrapolation_px
        ):
            logger.info(
                "Global coverage gate: %s model anchors cover %.0f%% of the axis "
                "(min %.0f%%) with %.3f nm (%.2f px) extrapolated correction drift "
                "past the dense anchor hull (max %.2f px); degraded to shift model.",
                global_result.model,
                100.0 * span_fraction,
                100.0 * coverage_min_anchor_span_fraction,
                extrap_nm,
                extrap_px,
                coverage_max_extrapolation_px,
            )
            global_result = calibrate_wavelength_axis(
                wavelength=wavelength,
                intensity=intensity,
                atomic_db=atomic_db,
                elements=elements,
                mode="auto",
                candidate_models=("shift",),
                inlier_tolerance_nm=inlier_tolerance_nm,
                max_pair_window_nm=max_pair_window_nm,
                random_seed=random_seed,
                **calibrate_kwargs,
            )
            global_result.details["coverage_gate"] = "degraded_to_shift"
            global_result.details["coverage_extrapolation_nm"] = float(extrap_nm)
        else:
            global_result.details["coverage_gate"] = "passed"
            global_result.details["coverage_extrapolation_nm"] = float(extrap_nm)

    if seams.size == 0:
        # No seams: degrade to the ordinary global calibration (safe path).
        global_result.details["segments"] = 1
        global_result.details["seam_count"] = 0
        return global_result

    bounds = [0] + [int(s) + 1 for s in seams] + [int(wavelength.size)]
    corrected = wavelength.copy()
    global_corrected = (
        global_result.corrected_wavelength
        if (global_result.success and fallback_to_global)
        else wavelength
    )
    global_offset = global_corrected - wavelength

    seg_diag, seg_status, total_inliers, rmse_accum = _run_segments(
        bounds,
        wavelength,
        intensity,
        corrected,
        global_offset,
        atomic_db,
        elements,
        min_segment_points=min_segment_points,
        sparse_segment_points=sparse_segment_points,
        sparse_segment_max_models=sparse_segment_max_models,
        candidate_models=candidate_models,
        inlier_tolerance_nm=inlier_tolerance_nm,
        max_pair_window_nm=max_pair_window_nm,
        segment_min_inliers=segment_min_inliers,
        segment_max_rmse_nm=segment_max_rmse_nm,
        segment_max_global_disagreement_nm=segment_max_global_disagreement_nm,
        affine_coverage_gate=affine_coverage_gate,
        coverage_min_anchor_span_fraction=coverage_min_anchor_span_fraction,
        coverage_max_extrapolation_px=coverage_max_extrapolation_px,
        random_seed=random_seed,
        calibrate_kwargs=calibrate_kwargs,
    )

    # Fallback 2: if the global fit was unavailable (offset all zero) for a
    # segment that also failed its own fit, borrow the median per-sample offset
    # of the nearest accepted-fit neighbour so no channel is left uncorrected
    # while its neighbours are shifted (which would tear lines across the seam).
    if not (global_result.success and fallback_to_global):
        _apply_neighbor_fallback(bounds, wavelength, corrected, seg_diag, seg_status)

    # Each accepted per-segment model is monotonic on its own grid (the
    # underlying calibrator rejects non-monotonic fits), so within a segment the
    # corrected axis is strictly increasing. The only way the stitched axis can
    # go non-monotonic is a boundary overlap *at a seam* when two adjacent
    # channels are corrected by different offsets. A seam is a genuine detector
    # gap, not real spectral data, so we restore monotonicity by shifting the
    # *entire* offending downstream segment up by a constant deficit (which
    # preserves its internal sample spacing exactly) rather than discarding the
    # per-channel correction or reordering samples. The shift cascades forward.
    n_clamped, max_clamp_nm, cumulative_shift = _restore_seam_monotonicity(bounds, corrected)

    # If restoring monotonicity required a large cumulative shift, the segments
    # genuinely disagree (not just touch at seams) and the segmentation is
    # untrustworthy -> revert to the safe global single-model fit.
    if cumulative_shift > 0.5:
        logger.warning(
            "Segmented calibration required a large cumulative seam shift "
            "(%.3f nm); reverting to global single-model fit.",
            cumulative_shift,
        )
        return _revert_segmented_to_global(global_result, bounds, seams, "large_seam_shift")

    # Sanity: a residual intra-segment non-monotonicity should be impossible.
    if not bool(np.all(np.diff(corrected) > 0)):
        logger.warning(
            "Segmented calibration left a non-monotonic axis after seam "
            "alignment; reverting to global single-model fit."
        )
        return _revert_segmented_to_global(global_result, bounds, seams, "residual_non_monotonic")

    return _build_segmented_result(
        global_result,
        corrected,
        bounds,
        seams,
        seg_diag,
        seg_status,
        total_inliers,
        rmse_accum,
        n_clamped,
        max_clamp_nm,
        segment_min_inliers,
        segment_max_rmse_nm,
    )
