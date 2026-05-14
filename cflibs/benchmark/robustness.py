"""
Robustness perturbation battery for CF-LIBS benchmark validation.

This module implements the Tier-2 robustness checks specified in
``docs/VALIDATION_METRICS.md`` §3 and ``validation/protocol.yaml``
``robustness`` block:

* **Line-dropout** -- remove the top-N highest-leverage lines from a
  spectrum, recompute the composition, require
  :math:`\\Delta d_A < 0.02`.
* **Outlier injection** -- inject :math:`N(0, k\\sigma)` noise into a
  fraction of channels, recompute the composition, require
  :math:`\\Delta d_A < 0.05`.

Per Matsumura et al. (*ACS Earth Space Chem* 2024), robust likelihoods
(Student-t) should beat :math:`L_2` under outlier perturbation.  This
harness makes that claim testable.

The module is deliberately decoupled from any specific inversion
pipeline: callers pass a ``pipeline_fn`` callable that maps a
:class:`~cflibs.benchmark.dataset.BenchmarkSpectrum` to a predicted
composition (``Dict[str, float]``).  This keeps the robustness battery
reusable across L\\ :sub:`2`/Student-t/Bayesian pipelines.

References
----------
* Matsumura et al., *ACS Earth Space Chem* 2024 -- robust Student-t
  Boltzmann fitting under outliers.
* Völker & Gornushkin, *JAAS* 2023 -- the 1% T-RSD threshold for the
  self-absorption robustness companion check (handled elsewhere).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from cflibs.benchmark.composition_metrics import aitchison_distance
from cflibs.benchmark.dataset import BenchmarkSpectrum

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

CompositionDict = Dict[str, float]
PipelineFn = Callable[[BenchmarkSpectrum], CompositionDict]
PerturbationFn = Callable[[BenchmarkSpectrum], BenchmarkSpectrum]


# Default thresholds mirror ``validation/protocol.yaml`` ``robustness``.
LINE_DROPOUT_DELTA_DA_MAX: float = 0.02
OUTLIER_INJECTION_DELTA_DA_MAX: float = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _copy_spectrum(spectrum: BenchmarkSpectrum) -> BenchmarkSpectrum:
    """Return a deep copy of a :class:`BenchmarkSpectrum` with independent arrays.

    The dataclass has NumPy arrays as attributes; ``copy.copy`` would share
    the underlying buffers and any in-place mutation would leak back into the
    caller's spectrum.  ``copy.deepcopy`` is used to keep the helper
    bullet-proof against future field additions, and the wavelength /
    intensity arrays are then re-assigned with ``np.array(...)`` so that the
    perturbation routines below can mutate them safely.
    """
    new = copy.deepcopy(spectrum)
    new.wavelength_nm = np.array(spectrum.wavelength_nm, copy=True)
    new.intensity = np.array(spectrum.intensity, copy=True)
    if spectrum.intensity_uncertainty is not None:
        new.intensity_uncertainty = np.array(spectrum.intensity_uncertainty, copy=True)
    return new


def _detect_peak_indices(
    intensity: np.ndarray,
    min_separation: int = 5,
) -> np.ndarray:
    """Return indices of local maxima in ``intensity``.

    A point ``i`` is a local maximum iff
    ``intensity[i] > intensity[i-1]`` AND ``intensity[i] >= intensity[i+1]``.
    The endpoints are excluded.  Empty / very short arrays return an empty
    index array.

    Peaks that lie within ``min_separation`` samples of a *taller* peak are
    suppressed -- this prevents a single physical line from being reported
    as multiple "peaks" because of noise on the line's flanks.

    Parameters
    ----------
    intensity : np.ndarray
        Spectrum intensity array.
    min_separation : int, optional
        Minimum spacing (in samples) between any two reported peaks
        (default ``5``).  Set to ``0`` to disable suppression.
    """
    n = intensity.size
    if n < 3:
        return np.empty(0, dtype=np.int64)
    left = intensity[1:-1] > intensity[:-2]
    right = intensity[1:-1] >= intensity[2:]
    mask = left & right
    candidates = np.where(mask)[0] + 1
    if candidates.size == 0 or min_separation <= 0:
        return candidates

    # Suppress peaks that are within `min_separation` of a *higher* peak.
    order = np.argsort(intensity[candidates])[::-1]
    keep_mask = np.ones(candidates.size, dtype=bool)
    accepted: List[int] = []
    for rank in order:
        idx = int(candidates[rank])
        if all(abs(idx - other) >= min_separation for other in accepted):
            accepted.append(idx)
        else:
            keep_mask[rank] = False
    accepted_arr = candidates[keep_mask]
    accepted_arr.sort()
    return accepted_arr


def _leverage_scores(
    intensity: np.ndarray,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-peak leverage scores for line-dropout perturbation.

    Parameters
    ----------
    intensity : np.ndarray
        Spectrum intensity array.
    metric : str
        ``"intensity"`` -- score is the peak intensity itself.  This is the
        canonical leverage in CF-LIBS: a Boltzmann/Saha-Boltzmann fit weighs
        bright lines most, so removing them moves the recovered T / n_e the
        farthest.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected local maxima, shape ``(K,)``.
    scores : np.ndarray
        Leverage scores, shape ``(K,)``.

    Raises
    ------
    ValueError
        If ``metric`` is not recognised.
    """
    if metric != "intensity":
        raise ValueError(f"Unknown leverage_metric={metric!r}. " "Supported: 'intensity'.")
    peaks = _detect_peak_indices(intensity)
    if peaks.size == 0:
        return peaks, np.empty(0, dtype=intensity.dtype)
    return peaks, intensity[peaks]


def _peak_window(
    wavelength_nm: np.ndarray,
    peak_index: int,
    intensity: np.ndarray,
    drop_fraction: float = 0.1,
    max_half_width: int = 200,
) -> Tuple[int, int]:
    """Return the half-open ``[lo, hi)`` index window around ``peak_index``.

    The window is bounded on each side by the first index at which the
    intensity falls below ``drop_fraction * intensity[peak_index]``.  This
    is a noise-robust replacement for a strict "walk down" approach, which
    would terminate at the first local minimum even when that minimum is
    only one sample wide and still well above the continuum.

    The half-width is capped by ``max_half_width`` to prevent runaway
    windows on broad / merged features (each peak contributes at most
    ``2 * max_half_width + 1`` channels to the dropout mask).

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength grid (used only for shape).
    peak_index : int
        Index of the peak whose window we want.
    intensity : np.ndarray
        Spectrum intensity array.
    drop_fraction : float, optional
        Fraction of peak height at which the window terminates
        (default ``0.1`` -- well below the peak but still inside the
        line shape so the dropout actually removes the line).
    max_half_width : int, optional
        Hard cap on the half-window in samples (default ``200``).

    Returns
    -------
    lo, hi : int
        Half-open window indices.
    """
    n = intensity.size
    if n == 0:
        return 0, 0
    peak_value = float(intensity[peak_index])
    threshold = drop_fraction * peak_value if peak_value > 0 else 0.0

    # Walk left while above threshold.
    lo = peak_index
    while lo > 0 and (peak_index - lo) < max_half_width and intensity[lo - 1] >= threshold:
        lo -= 1

    # Walk right while above threshold.
    hi = peak_index + 1
    while hi < n and (hi - peak_index - 1) < max_half_width and intensity[hi] >= threshold:
        hi += 1
    return lo, hi


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------


def line_dropout_perturbation(
    spectrum: BenchmarkSpectrum,
    top_n: int = 3,
    leverage_metric: str = "intensity",
) -> BenchmarkSpectrum:
    """Remove the top-N highest-leverage lines from a spectrum.

    The implementation:

    1. Detects local maxima ("lines") in the intensity array.
    2. Ranks them by ``leverage_metric`` (currently always peak intensity).
    3. Identifies the index window for each of the top-N peaks (bounded by
       the nearest local minima on either side).
    4. Sets the intensity inside each window to a baseline equal to the
       lower of the two boundary minima (a flat continuum patch).

    Removing the *window* rather than the single peak sample reflects what
    "deleting a line" physically means and ensures the perturbed spectrum
    has no residual peak signal at that location.

    Parameters
    ----------
    spectrum : BenchmarkSpectrum
        Original spectrum.
    top_n : int, optional
        Number of leverage lines to remove (default ``3`` -- matches
        ``validation/protocol.yaml`` ``robustness.line_dropout``).
    leverage_metric : str, optional
        Currently only ``"intensity"`` is implemented.

    Returns
    -------
    BenchmarkSpectrum
        New spectrum (deep copy) with the top-N lines removed.

    Notes
    -----
    * If fewer than ``top_n`` peaks are detected, all available peaks are
      removed.  This keeps the function robust on synthetic / sparse spectra
      where there may not be ``top_n`` peaks.
    * ``top_n=0`` is a no-op (returns a deep copy).
    """
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")

    out = _copy_spectrum(spectrum)

    if top_n == 0:
        return out

    peaks, scores = _leverage_scores(out.intensity, leverage_metric)
    if peaks.size == 0:
        return out

    # Argsort descending; pick top_n.
    order = np.argsort(scores)[::-1]
    chosen = peaks[order[:top_n]]

    # Apply window-based dropout.
    intensity = out.intensity
    for peak_index in chosen:
        lo, hi = _peak_window(out.wavelength_nm, int(peak_index), intensity)
        # Use the lower of the two boundary samples as the continuum patch.
        left_floor = intensity[lo] if lo < intensity.size else 0.0
        right_floor = intensity[hi - 1] if 0 <= hi - 1 < intensity.size else 0.0
        # The window's boundary intensities are the surrounding minima; using
        # the smaller of the two avoids leaving a step.
        baseline = float(min(left_floor, right_floor))
        intensity[lo:hi] = baseline
    out.intensity = intensity
    return out


def outlier_injection_perturbation(
    spectrum: BenchmarkSpectrum,
    fraction: float = 0.05,
    sigma_multiplier: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> BenchmarkSpectrum:
    """Inject Gaussian outliers into a fraction of channels.

    A subset of size ``round(fraction * n_channels)`` is sampled uniformly
    (without replacement) and additive noise drawn from
    :math:`\\mathcal{N}(0, (k\\sigma)^2)` is added to those channels, where
    :math:`\\sigma` is the standard deviation of the *unperturbed* intensity
    array and :math:`k` is ``sigma_multiplier``.

    Parameters
    ----------
    spectrum : BenchmarkSpectrum
        Original spectrum.
    fraction : float, optional
        Fraction of channels to corrupt (``0.05`` per protocol).
    sigma_multiplier : float, optional
        Multiplier ``k`` on the spectrum-wide :math:`\\sigma` (``5.0`` per
        protocol).
    rng : np.random.Generator, optional
        RNG used for both the channel selection and the noise draw.  If
        ``None``, a fresh ``np.random.default_rng()`` is created (NOT
        reproducible -- supply your own RNG for reproducibility).

    Returns
    -------
    BenchmarkSpectrum
        New spectrum (deep copy) with outliers injected.

    Notes
    -----
    Reproducibility: passing the same ``rng`` (or a freshly-seeded
    ``np.random.default_rng(seed)``) to two calls produces identical
    perturbations -- enforced by ``test_outlier_injection_reproducible``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if sigma_multiplier < 0:
        raise ValueError(f"sigma_multiplier must be >= 0, got {sigma_multiplier}")

    out = _copy_spectrum(spectrum)
    n = out.intensity.size

    if n == 0 or fraction <= 0.0 or sigma_multiplier <= 0.0:
        return out

    if rng is None:
        rng = np.random.default_rng()

    sigma = float(np.std(out.intensity))
    if sigma < 1e-12:
        # Fall back to a fraction of the mean if std is effectively zero
        # (constant spectrum). This still injects detectable outliers.
        sigma = float(np.mean(np.abs(out.intensity))) or 1.0

    n_corrupt = int(round(fraction * n))
    if n_corrupt == 0:
        return out

    indices = rng.choice(n, size=n_corrupt, replace=False)
    noise = rng.normal(loc=0.0, scale=sigma_multiplier * sigma, size=n_corrupt)
    out.intensity[indices] = out.intensity[indices] + noise
    return out


# ---------------------------------------------------------------------------
# Battery harness
# ---------------------------------------------------------------------------


@dataclass
class PerturbationResult:
    """Single (spectrum, perturbation) result."""

    spectrum_id: str
    perturbation: str
    d_a_unperturbed: float
    d_a_perturbed: float
    delta_d_a: float
    threshold: Optional[float] = None
    passes_threshold: Optional[bool] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationSummary:
    """Aggregate Δ$d_A$ statistics for a single perturbation type."""

    perturbation: str
    n_spectra: int
    mean_delta_d_a: float
    median_delta_d_a: float
    max_delta_d_a: float
    bootstrap_ci_lo: float
    bootstrap_ci_hi: float
    threshold: Optional[float] = None
    fraction_passing: Optional[float] = None
    fraction_failing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationReport:
    """Full report from :func:`run_perturbation_battery`.

    Attributes
    ----------
    results : List[PerturbationResult]
        Per-spectrum, per-perturbation rows.
    perturbation_specs : Dict[str, Dict[str, Any]]
        The spec each perturbation was run with (for provenance).
    """

    results: List[PerturbationResult] = field(default_factory=list)
    perturbation_specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def reduce_to_summary(
        self,
        bootstrap_iterations: int = 1000,
        bootstrap_alpha: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, PerturbationSummary]:
        """Aggregate per-perturbation Δ$d_A$ with bootstrap CI.

        Parameters
        ----------
        bootstrap_iterations : int, optional
            Number of bootstrap resamples (default ``1000``).  Set to ``0``
            to skip the bootstrap (lo/hi will mirror the mean).
        bootstrap_alpha : float, optional
            CI level: ``0.05`` -> 95% CI.
        rng : np.random.Generator, optional
            RNG for the bootstrap.  If ``None``, defaults are used (not
            reproducible).

        Returns
        -------
        Dict[str, PerturbationSummary]
            Keyed by perturbation name.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Group the (delta, threshold) values by perturbation, ignoring rows
        # where the pipeline raised an error.
        by_pert: Dict[str, List[Tuple[float, Optional[float]]]] = {}
        for row in self.results:
            if row.error is not None:
                continue
            by_pert.setdefault(row.perturbation, []).append((row.delta_d_a, row.threshold))

        summaries: Dict[str, PerturbationSummary] = {}
        for name, rows in by_pert.items():
            deltas = np.asarray([r[0] for r in rows], dtype=float)
            thresholds = [r[1] for r in rows if r[1] is not None]
            threshold = thresholds[0] if thresholds else None
            n = deltas.size

            if n == 0:
                continue

            mean_d = float(np.mean(deltas))
            median_d = float(np.median(deltas))
            max_d = float(np.max(deltas))

            if bootstrap_iterations > 0 and n > 1:
                idx = rng.integers(0, n, size=(bootstrap_iterations, n))
                boot = np.mean(deltas[idx], axis=1)
                lo = float(np.quantile(boot, bootstrap_alpha / 2.0))
                hi = float(np.quantile(boot, 1.0 - bootstrap_alpha / 2.0))
            else:
                lo = mean_d
                hi = mean_d

            if threshold is not None:
                fraction_passing = float(np.mean(deltas < threshold))
                fraction_failing = 1.0 - fraction_passing
            else:
                fraction_passing = None
                fraction_failing = None

            summaries[name] = PerturbationSummary(
                perturbation=name,
                n_spectra=n,
                mean_delta_d_a=mean_d,
                median_delta_d_a=median_d,
                max_delta_d_a=max_d,
                bootstrap_ci_lo=lo,
                bootstrap_ci_hi=hi,
                threshold=threshold,
                fraction_passing=fraction_passing,
                fraction_failing=fraction_failing,
            )

        return summaries

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a JSON-friendly dict, including summary."""
        summary = self.reduce_to_summary()
        return {
            "results": [r.to_dict() for r in self.results],
            "perturbation_specs": self.perturbation_specs,
            "summary": {k: v.to_dict() for k, v in summary.items()},
        }

    def save_json(self, path: Union[str, Path]) -> Path:
        """Write the full report (incl. summary) to ``path`` as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return out_path


# ---------------------------------------------------------------------------
# Built-in perturbation registry
# ---------------------------------------------------------------------------


def default_perturbations(
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the protocol-aligned default perturbations for the battery.

    Each value is a dict with keys:

    * ``fn`` -- a :data:`PerturbationFn` callable.
    * ``threshold`` -- the protocol Δ$d_A$ threshold (``None`` if unset).
    * ``spec`` -- a JSON-friendly dict capturing the parameters.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Used by stochastic perturbations (currently outlier injection).
        A fresh default RNG is created if ``None``.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _line_dropout(s: BenchmarkSpectrum) -> BenchmarkSpectrum:
        return line_dropout_perturbation(s, top_n=3, leverage_metric="intensity")

    def _outlier_injection(s: BenchmarkSpectrum) -> BenchmarkSpectrum:
        return outlier_injection_perturbation(s, fraction=0.05, sigma_multiplier=5.0, rng=rng)

    return {
        "line_dropout_top3": {
            "fn": _line_dropout,
            "threshold": LINE_DROPOUT_DELTA_DA_MAX,
            "spec": {
                "type": "line_dropout",
                "top_n": 3,
                "leverage_metric": "intensity",
                "delta_d_a_max": LINE_DROPOUT_DELTA_DA_MAX,
            },
        },
        "outlier_injection_5pct_5sigma": {
            "fn": _outlier_injection,
            "threshold": OUTLIER_INJECTION_DELTA_DA_MAX,
            "spec": {
                "type": "outlier_injection",
                "fraction": 0.05,
                "sigma_multiplier": 5.0,
                "delta_d_a_max": OUTLIER_INJECTION_DELTA_DA_MAX,
            },
        },
    }


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def run_perturbation_battery(
    pipeline_fn: PipelineFn,
    spectra: Sequence[BenchmarkSpectrum],
    perturbations: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ground_truth: Optional[Mapping[str, CompositionDict]] = None,
    rng: Optional[np.random.Generator] = None,
) -> PerturbationReport:
    """Run the full perturbation battery.

    For each ``(spectrum, perturbation)`` pair this:

    1. Calls ``pipeline_fn`` on the *unperturbed* spectrum to get
       :math:`\\hat C_0`.
    2. Applies the perturbation.
    3. Calls ``pipeline_fn`` on the perturbed spectrum to get
       :math:`\\hat C_p`.
    4. Computes :math:`\\Delta d_A`:

       * If ``ground_truth`` is provided and contains the spectrum,
         :math:`\\Delta d_A = |d_A(\\hat C_p, C^*) - d_A(\\hat C_0, C^*)|`.
       * Otherwise, :math:`\\Delta d_A = d_A(\\hat C_p, \\hat C_0)` --
         the Aitchison drift the perturbation induced.

    The unperturbed pipeline call is cached so each spectrum is run once
    on the unperturbed input regardless of how many perturbations are
    evaluated against it.

    Parameters
    ----------
    pipeline_fn : Callable[[BenchmarkSpectrum], Dict[str, float]]
        A function that returns a predicted composition for a spectrum.
    spectra : Sequence[BenchmarkSpectrum]
        Spectra to evaluate.
    perturbations : Mapping[str, Mapping[str, Any]], optional
        Perturbations to apply.  Keys are perturbation names, values are
        dicts with at least ``"fn"`` (a :data:`PerturbationFn`).
        Optional keys: ``"threshold"`` (float Δ$d_A$ threshold from the
        protocol) and ``"spec"`` (JSON-friendly metadata).  Defaults to
        :func:`default_perturbations` (uses ``rng``).
    ground_truth : Mapping[str, Dict[str, float]], optional
        Optional spectrum_id -> ground truth composition mapping.  When
        absent, Δ$d_A$ is computed against the unperturbed prediction
        instead of against ground truth.
    rng : np.random.Generator, optional
        Default RNG threaded into :func:`default_perturbations`.

    Returns
    -------
    PerturbationReport
        Per-row results plus the spec for each perturbation.
    """
    if perturbations is None:
        perturbations = default_perturbations(rng=rng)

    report = PerturbationReport()
    for name, entry in perturbations.items():
        report.perturbation_specs[name] = dict(entry.get("spec", {}))

    for spec in spectra:
        try:
            base_pred = pipeline_fn(spec)
        except Exception as exc:  # pragma: no cover -- defensive
            for name in perturbations:
                report.results.append(
                    PerturbationResult(
                        spectrum_id=spec.spectrum_id,
                        perturbation=name,
                        d_a_unperturbed=float("nan"),
                        d_a_perturbed=float("nan"),
                        delta_d_a=float("nan"),
                        threshold=perturbations[name].get("threshold"),
                        passes_threshold=None,
                        error=f"unperturbed-pipeline: {exc!r}",
                    )
                )
            continue

        truth: Optional[CompositionDict] = None
        if ground_truth is not None and spec.spectrum_id in ground_truth:
            truth = dict(ground_truth[spec.spectrum_id])
        elif ground_truth is None:
            # The benchmark's own true_composition is also acceptable when
            # the caller supplied no override.
            truth = dict(spec.true_composition) if spec.true_composition else None

        d_a_unperturbed: float
        if truth is not None and truth:
            d_a_unperturbed = aitchison_distance(truth, base_pred)
        else:
            d_a_unperturbed = 0.0  # unperturbed-vs-itself

        for name, entry in perturbations.items():
            fn: PerturbationFn = entry["fn"]
            threshold: Optional[float] = entry.get("threshold")
            try:
                perturbed_spec = fn(spec)
                pert_pred = pipeline_fn(perturbed_spec)
            except Exception as exc:  # pragma: no cover -- defensive
                report.results.append(
                    PerturbationResult(
                        spectrum_id=spec.spectrum_id,
                        perturbation=name,
                        d_a_unperturbed=d_a_unperturbed,
                        d_a_perturbed=float("nan"),
                        delta_d_a=float("nan"),
                        threshold=threshold,
                        passes_threshold=None,
                        error=f"perturbation/pipeline: {exc!r}",
                    )
                )
                continue

            if truth is not None and truth:
                d_a_perturbed = aitchison_distance(truth, pert_pred)
                delta = abs(d_a_perturbed - d_a_unperturbed)
            else:
                d_a_perturbed = aitchison_distance(base_pred, pert_pred)
                delta = d_a_perturbed

            passes = bool(delta < threshold) if threshold is not None else None

            report.results.append(
                PerturbationResult(
                    spectrum_id=spec.spectrum_id,
                    perturbation=name,
                    d_a_unperturbed=float(d_a_unperturbed),
                    d_a_perturbed=float(d_a_perturbed),
                    delta_d_a=float(delta),
                    threshold=threshold,
                    passes_threshold=passes,
                    error=None,
                )
            )

    return report


__all__ = [
    "LINE_DROPOUT_DELTA_DA_MAX",
    "OUTLIER_INJECTION_DELTA_DA_MAX",
    "PerturbationFn",
    "PerturbationResult",
    "PerturbationSummary",
    "PerturbationReport",
    "PipelineFn",
    "default_perturbations",
    "line_dropout_perturbation",
    "outlier_injection_perturbation",
    "run_perturbation_battery",
]
