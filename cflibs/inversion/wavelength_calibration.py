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
from cflibs.inversion.preprocessing import detect_peaks_auto

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

    if best_coef is None or best_inliers.size < min_pts:
        return None

    # Refit on robust inliers
    refit_coef = _fit_model(x[best_inliers], y[best_inliers], model, weights=weights[best_inliers])
    if refit_coef is None:
        return None

    pred_all = _eval_model(x, model, refit_coef)
    residual_all = np.abs(pred_all - y)
    inlier_mask_all = residual_all <= inlier_tolerance_nm
    robust_inliers = _dedupe_one_to_one(residual_all, peak_ids, line_ids, inlier_mask_all)

    if robust_inliers.size < min_pts:
        return None

    # Final refine
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
    quality_min_peak_match_fraction: float = 0.35,
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
        Minimum fraction of detected peaks that must be matched.
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

    if not cand_x:
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

    x = np.asarray(cand_x, dtype=float)
    y = np.asarray(cand_y, dtype=float)
    peak_ids = np.asarray(cand_peak_id, dtype=int)
    line_ids = np.asarray(cand_line_id, dtype=int)
    weights = np.asarray(cand_weight, dtype=float)

    if mode == "auto":
        models = [m for m in candidate_models if m in {"shift", "affine", "quadratic"}]
    else:
        models = [mode]

    fits: List[_ModelFit] = []
    for m in models:
        fit = _ransac_fit(
            model=m,
            x=x,
            y=y,
            peak_ids=peak_ids,
            line_ids=line_ids,
            weights=weights,
            n_peaks_total=int(peak_wl.size),
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
    details = {
        "peak_count": float(peak_wl.size),
        "line_pool_size": float(line_wl.size),
        "candidate_count": float(x.size),
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
