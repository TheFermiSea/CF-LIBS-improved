"""
Synthetic corpus identifier benchmark utilities.

Runs the full identifier stack on synthetic benchmark spectra and produces
element-level + peak-level diagnostics suitable for regression tracking.

The default suite is the three peak-matching identifiers (ALIAS, Comb,
Correlation), which share the ``cls(db, elements=..., **kwargs)`` constructor
shape. Two full-spectrum / composite identifiers are *opt-in* because they
need a pre-computed :class:`~cflibs.manifold.basis_library.BasisLibrary`:

* ``spectral_nnls`` — NNLS decomposition into single-element basis spectra
  (:class:`~cflibs.inversion.identify.spectral_nnls.SpectralNNLSIdentifier`).
* ``hybrid_union`` — the NNLS ∪ ALIAS union
  (:class:`~cflibs.inversion.identify.hybrid.HybridIdentifier` with
  ``require_both=False``), reusing the existing two-stage building block
  rather than reinventing the union logic.

Pass ``include_nnls=True`` and a ``basis_library`` to :func:`evaluate_dataset`
/ :func:`run_synthetic_benchmark` to enable them. When the basis library is
unavailable the two extra identifiers are silently skipped, so the historical
three-identifier behavior (and the unit tests) are unaffected by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import csv
import json
import logging
import os
import re

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.dataset import BenchmarkDataset, BenchmarkSpectrum
from cflibs.benchmark.loaders import load_benchmark
from cflibs.inversion.identify.alias import STRICT_DETECTION_THRESHOLD, ALIASIdentifier
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.inversion.identify.correlation import CorrelationIdentifier
from cflibs.inversion.common.element_id import ElementIdentification, ElementIdentificationResult
from cflibs.inversion.preprocess.wavelength_calibration import calibrate_wavelength_axis

if TYPE_CHECKING:
    from cflibs.manifold.basis_library import BasisLibrary

logger = logging.getLogger(__name__)

# Identifier name constants. The peak-matching trio is always run; the two
# basis-dependent identifiers are appended only when a basis library is built.
ALGO_ALIAS = "ALIAS"
ALGO_COMB = "Comb"
ALGO_CORRELATION = "Correlation"
ALGO_SPECTRAL_NNLS = "spectral_nnls"
ALGO_HYBRID_UNION = "hybrid_union"
ALGO_FORWARD_FIT = "forward_fit"

_BASE_ALGORITHMS: Tuple[str, ...] = (ALGO_ALIAS, ALGO_COMB, ALGO_CORRELATION)
_BASIS_ALGORITHMS: Tuple[str, ...] = (ALGO_SPECTRAL_NNLS, ALGO_HYBRID_UNION)
# J10 population forward-fitter (JAX-only); opt-in via ``with_forward_fit``.
_FORWARD_FIT_ALGORITHMS: Tuple[str, ...] = (ALGO_FORWARD_FIT,)


def derive_truth_elements(
    composition: Dict[str, float], presence_threshold: float = 1e-4
) -> Set[str]:
    """Return set of elements considered present in ground truth (>= floor)."""
    return {el for el, frac in composition.items() if float(frac) >= float(presence_threshold)}


def derive_ignore_elements(
    composition: Dict[str, float], presence_threshold: float = 1e-4
) -> Set[str]:
    """Return in-recipe elements BELOW the detection floor — the "don't-care" band.

    Detection-floor semantics (3-way scoring): an element is *present* if its
    concentration is >= ``presence_threshold``, *absent* if it is not in the
    recipe at all (fraction == 0), and *don't-care* if it is in the recipe but
    below the floor (0 < frac < presence_threshold). Don't-care elements are
    removed from the scoring panel so that detecting a real-but-sub-detection-
    limit trace is neither rewarded (TP) nor penalised (FP), and missing it is
    not an FN. At the legacy default (1e-4) this band is empty, so behaviour is
    unchanged unless a meaningful floor is set.
    """
    floor = float(presence_threshold)
    return {el for el, frac in composition.items() if 0.0 < float(frac) < floor}


def compute_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute precision/recall/FPR/F1/accuracy from confusion counts."""
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def confusion_counts(
    true_elements: Set[str],
    predicted_elements: Set[str],
    candidate_elements: Sequence[str],
) -> Dict[str, int]:
    """Return TP/FP/FN/TN counts for one sample."""
    tp = fp = fn = tn = 0
    for element in candidate_elements:
        truth = element in true_elements
        pred = element in predicted_elements
        if truth and pred:
            tp += 1
        elif (not truth) and pred:
            fp += 1
        elif truth and (not pred):
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


_RECIPE_PROVENANCE_RE = re.compile(r"recipe=([^;]+)")
_SAMPLE_SUFFIX_RE = re.compile(r"_\d+$")


def _derive_recipe(
    spec: BenchmarkSpectrum,
    manifest_by_sample: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Resolve the synthetic recipe label for a spectrum.

    Resolution ladder (first non-empty wins):

    1. ``manifest_by_sample[spectrum_id]["recipe"]`` — when a manifest ships.
    2. ``recipe=<name>`` parsed out of ``spec.metadata.provenance`` — the
       manifest-less corpus (``data/synthetic_corpus/corpus.json``) stores the
       recipe only here, e.g. ``"...; recipe=binary_Fe_Ni; scenario=3"``.
    3. The ``spectrum_id`` prefix with any trailing ``_<digits>`` stripped,
       e.g. ``"binary_Fe_Ni_0007" -> "binary_Fe_Ni"``.

    Returns an empty string only if all three fail (no id at all).
    """
    if manifest_by_sample:
        entry = manifest_by_sample.get(spec.spectrum_id)
        if entry:
            recipe = entry.get("recipe")
            if recipe:
                return str(recipe)

    provenance = ""
    metadata = getattr(spec, "metadata", None)
    if metadata is not None:
        provenance = str(getattr(metadata, "provenance", "") or "")
    match = _RECIPE_PROVENANCE_RE.search(provenance)
    if match:
        return match.group(1).strip()

    sample_id = str(spec.spectrum_id or "")
    return _SAMPLE_SUFFIX_RE.sub("", sample_id)


def _adjust_alloc(
    alloc: List[int],
    raw: Sequence[float],
    target: int,
    sizes: Sequence[int],
) -> None:
    """Trim/grow ``alloc`` in place so it sums to exactly ``target``.

    Uses the largest-remainder method: when the floor allocation overshoots
    (because each non-empty stratum is forced to >=1) drops are taken from the
    strata with the smallest fractional remainders first (never below 1 while
    other strata still have surplus); when it undershoots, extra picks go to
    the largest fractional remainders. Allocation per stratum is also clamped
    to its available pool size.
    """
    n = len(alloc)
    if n == 0:
        return
    # remainder used as the tie-break key for grow/shrink.
    remainders = [float(r) - float(int(np.floor(r))) for r in raw]
    current = sum(alloc)

    # Grow: hand extra picks to the largest fractional remainder, respecting pool size.
    grow_order = sorted(range(n), key=lambda i: remainders[i], reverse=True)
    while current < target:
        progressed = False
        for i in grow_order:
            if current >= target:
                break
            if alloc[i] < int(sizes[i]):
                alloc[i] += 1
                current += 1
                progressed = True
        if not progressed:
            break  # every stratum saturated; cannot reach target

    # Shrink: remove from smallest fractional remainder first, never below 1.
    shrink_order = sorted(range(n), key=lambda i: remainders[i])
    while current > target:
        progressed = False
        for i in shrink_order:
            if current <= target:
                break
            if alloc[i] > 1:
                alloc[i] -= 1
                current -= 1
                progressed = True
        if not progressed:
            # All strata at the >=1 floor but still over target: drop floors as
            # a last resort (only happens when n_strata > target).
            for i in shrink_order:
                if current <= target:
                    break
                if alloc[i] > 0:
                    alloc[i] -= 1
                    current -= 1
            break


def _select_spectra(
    spectra: Sequence[BenchmarkSpectrum],
    max_spectra: Optional[int],
    sampling: str,
    seed: int,
    manifest_by_sample: Optional[Dict[str, Dict[str, Any]]],
) -> List[BenchmarkSpectrum]:
    """Select up to ``max_spectra`` spectra under the requested sampling mode.

    Modes:

    * ``"sorted"`` — legacy: alpha-sorted by ``spectrum_id`` then truncated.
      Preserved for exact reproduction of historical runs (biased: with the
      4-recipe corpus any cap <=72 yields 100% of the alphabetically-first
      recipe).
    * ``"shuffle"`` — seeded uniform random draw (then re-sorted for stable
      output ordering).
    * ``"stratified"`` (default) — proportional, largest-remainder allocation
      over ``(recipe, label_cardinality)`` strata with >=1 per non-empty
      stratum, so every recipe/cardinality combination is represented.

    The full sorted list is always the determinism floor; the returned list is
    sorted by ``spectrum_id`` so downstream output ordering is reproducible.
    """
    ordered = sorted(spectra, key=lambda s: s.spectrum_id)
    if max_spectra is None or int(max_spectra) >= len(ordered):
        return ordered
    n = int(max_spectra)
    if n <= 0:
        return []
    if sampling == "sorted":
        return ordered[:n]

    rng = np.random.default_rng(seed)
    if sampling == "shuffle":
        idx = rng.permutation(len(ordered))[:n]
        return [ordered[i] for i in sorted(idx)]

    if sampling == "stratified":
        strata: Dict[Tuple[str, int], List[BenchmarkSpectrum]] = {}
        for s in ordered:
            key = (_derive_recipe(s, manifest_by_sample), int(s.label_cardinality or 0))
            strata.setdefault(key, []).append(s)
        keys = sorted(strata)
        sizes = [len(strata[k]) for k in keys]
        total = sum(sizes)
        raw = [n * sz / total for sz in sizes]
        alloc = [max(1, int(np.floor(r))) for r in raw]
        _adjust_alloc(alloc, raw, n, sizes)
        picked: List[BenchmarkSpectrum] = []
        for k, a in zip(keys, alloc):
            pool = strata[k]
            take = min(int(a), len(pool))
            sel = rng.permutation(len(pool))[:take]
            picked.extend(pool[i] for i in sel)
        return sorted(picked, key=lambda s: s.spectrum_id)

    raise ValueError(f"unknown sampling mode {sampling!r}")


def resolving_power_from_spectrum(spec: BenchmarkSpectrum, fallback: float = 900.0) -> float:
    """
    Infer resolving power from benchmark spectrum conditions.

    Uses mean wavelength / spectral_resolution_nm when available.
    """
    resolution_nm = float(spec.conditions.spectral_resolution_nm)
    if not np.isfinite(resolution_nm) or resolution_nm <= 0:
        return float(fallback)
    mean_wl = float(np.mean(spec.wavelength_nm))
    rp = mean_wl / resolution_nm
    return float(np.clip(rp, 100.0, 20000.0))


def build_corpus_basis_library(
    db_path: str,
    output_path: str,
    elements: Sequence[str],
    wavelength_range: Tuple[float, float] = (222.0, 267.0),
    pixels: int = 2560,
    temperature_range: Tuple[float, float] = (9000.0, 21000.0),
    temperature_steps: int = 8,
    density_range: Tuple[float, float] = (1e16, 1e18),
    density_steps: int = 5,
    ionization_stages: Tuple[int, ...] = (1, 2),
    instrument_fwhm_nm: float = 0.3,
    overwrite: bool = False,
) -> Optional["BasisLibrary"]:
    """Build a small single-element basis library for a synthetic corpus.

    Mirrors :class:`cflibs.manifold.basis_library.BasisLibraryGenerator` but
    restricts the element set (and grid) to *exactly* the corpus candidates,
    keeping the file tiny and the build deterministic & fast (seconds, not
    minutes). The full :meth:`BasisLibraryGenerator.generate` builds every
    element present in the atomic DB, which is unnecessary here.

    The defaults span the ps-LIBS regime of the workhorse synthetic corpora
    (T ~ 0.8-1.8 eV, n_e ~ 1e16-1e18 cm^-3) over the deep-UV iron-group
    window. The element order is sorted for determinism so repeat builds are
    byte-stable, which is what a controlled before/after measurement needs.

    Parameters
    ----------
    db_path : str
        Atomic database path.
    output_path : str
        HDF5 destination. Reused if it already exists (unless ``overwrite``).
    elements : sequence of str
        Candidate elements to include as basis vectors.
    wavelength_range, pixels, temperature_range, temperature_steps,
    density_range, density_steps, ionization_stages, instrument_fwhm_nm
        Grid knobs (see :class:`BasisLibraryConfig`).
    overwrite : bool
        Rebuild even if ``output_path`` already exists.

    Returns
    -------
    BasisLibrary or None
        Loaded basis library, or ``None`` if generation is impossible
        (e.g. ``h5py`` unavailable). Callers treat ``None`` as "skip the
        basis-dependent identifiers".
    """
    try:
        import h5py  # noqa: F401

        from cflibs.manifold.basis_library import (
            BasisLibrary,
            BasisLibraryConfig,
            BasisLibraryGenerator,
        )
    except Exception:  # noqa: BLE001 - h5py / basis module unavailable
        logger.warning(
            "Cannot build basis library (h5py or basis module unavailable); "
            "spectral_nnls / hybrid_union will be skipped.",
            exc_info=True,
        )
        return None

    out = Path(output_path)
    sorted_elements = sorted({str(e) for e in elements})

    if out.exists() and not overwrite:
        logger.info("Reusing existing basis library at %s", out)
        return BasisLibrary(str(out))

    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = BasisLibraryConfig(
        db_path=str(db_path),
        output_path=str(out),
        wavelength_range=wavelength_range,
        pixels=pixels,
        temperature_range=temperature_range,
        temperature_steps=temperature_steps,
        density_range=density_range,
        density_steps=density_steps,
        ionization_stages=tuple(ionization_stages),
        instrument_fwhm_nm=instrument_fwhm_nm,
    )
    cfg.validate()

    generator = BasisLibraryGenerator(cfg)
    wl_grid, sigma, params = BasisLibraryGenerator._build_grids(cfg)
    n_grid = params.shape[0]
    spectra = np.zeros((len(sorted_elements), n_grid, cfg.pixels), dtype=np.float32)
    for idx, element in enumerate(sorted_elements):
        spectra[idx, :, :] = generator._compute_element_spectra(element, wl_grid, sigma, params)

    with h5py.File(str(out), "w") as f:
        f.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
        f.create_dataset("params", data=params)
        f.create_dataset("wavelength", data=wl_grid)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("elements", data=np.array(sorted_elements, dtype=object), dtype=dt)

    logger.info(
        "Built corpus basis library: %s (%d elements, %d grid points)",
        out,
        len(sorted_elements),
        n_grid,
    )
    return BasisLibrary(str(out))


def _element_by_name(result: ElementIdentificationResult) -> Dict[str, ElementIdentification]:
    return {e.element: e for e in result.all_elements}


# A "runner" is a zero-argument callable that constructs an identifier and
# returns its ElementIdentificationResult for the current spectrum. Using a
# per-entry factory (instead of the old (name, cls, kwargs) tuple) lets the
# basis-dependent identifiers — whose constructor (basis_library=...) and
# identify() signatures differ from the cls(db, elements, **kwargs) trio —
# slot in without special-casing the dispatch loop.
IdentifierRunner = Callable[[], ElementIdentificationResult]


def build_identifier_runners(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
    resolving_power: float,
    basis_library: Optional["BasisLibrary"] = None,
    with_forward_fit: bool = False,
) -> List[Tuple[str, IdentifierRunner]]:
    """Build the ordered list of (name, runner) identifier factories.

    The peak-matching trio (ALIAS/Comb/Correlation) is always present. When
    ``basis_library`` is provided, ``spectral_nnls`` and ``hybrid_union`` are
    appended; otherwise they are omitted entirely (graceful skip). When
    ``with_forward_fit`` is True, the J10 population forward-fitter
    (:class:`cflibs.jitpipe.forward_id_identifier.ForwardFitIdentifier`) is
    appended — JAX-only, so it is opt-in and runs after the catalog identifiers.
    """

    def _run_alias() -> ElementIdentificationResult:
        identifier = ALIASIdentifier(
            db,
            elements=elements,
            resolving_power=resolving_power,
            intensity_threshold_factor=3.0,
            detection_threshold=0.5,  # paper C_th (Noel 2025 sec 3.8): k_det presence threshold
            chance_window_scale=0.3,
        )
        return identifier.identify(wavelength, intensity)

    def _run_comb() -> ElementIdentificationResult:
        identifier = CombIdentifier(
            db,
            elements=elements,
            resolving_power=resolving_power,
            min_correlation=0.08,
            tooth_activation_threshold=0.35,
            relative_threshold_scale=1.4,
            min_aki_gk=3000.0,
        )
        return identifier.identify(wavelength, intensity)

    def _run_correlation() -> ElementIdentificationResult:
        identifier = CorrelationIdentifier(
            db,
            elements=elements,
            resolving_power=resolving_power,
            min_confidence=0.008,
            relative_threshold_scale=1.2,
            min_line_strength=1000.0,
            T_range_K=(5000, 15000),
            T_steps=7,
            n_e_range_cm3=(1e15, 5e17),
            n_e_steps=4,
        )
        return identifier.identify(wavelength, intensity, mode="classic")

    runners: List[Tuple[str, IdentifierRunner]] = [
        (ALGO_ALIAS, _run_alias),
        (ALGO_COMB, _run_comb),
        (ALGO_CORRELATION, _run_correlation),
    ]

    if basis_library is not None:
        # NNLS decomposes against the basis library's own element set; pass
        # the corpus candidates as the fallback grid so they line up. Mirror
        # the production kwargs from cflibs/benchmark/unified.py.
        def _run_nnls() -> ElementIdentificationResult:
            from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

            identifier = SpectralNNLSIdentifier(
                basis_library=basis_library,
                detection_snr=3.0,
                continuum_degree=3,
            )
            return identifier.identify(wavelength, intensity)

        # hybrid_union == HybridIdentifier(require_both=False), i.e. the
        # NNLS ∪ ALIAS union. Reuses the two-stage building block rather than
        # reinventing the union; lenient NNLS screen + ALIAS confirmation,
        # then the union of the two detection sets.
        def _run_hybrid_union() -> ElementIdentificationResult:
            from cflibs.inversion.identify.hybrid import HybridIdentifier

            identifier = HybridIdentifier(
                atomic_db=db,
                basis_library=basis_library,
                elements=elements,
                resolving_power=resolving_power,
                nnls_detection_snr=1.5,
                # Paper strict C_th (k_det > C_th); old 0.05 CL-floor was
                # "accept everything" on the k_det scale, flooding the union
                # branch with ALIAS false positives.
                alias_detection_threshold=STRICT_DETECTION_THRESHOLD,
                require_both=False,
            )
            return identifier.identify(wavelength, intensity)

        runners.append((ALGO_SPECTRAL_NNLS, _run_nnls))
        runners.append((ALGO_HYBRID_UNION, _run_hybrid_union))

    if with_forward_fit:

        def _run_forward_fit() -> ElementIdentificationResult:
            from cflibs.jitpipe.forward_id_identifier import ForwardFitIdentifier
            from cflibs.jitpipe.snapshot import PipelineSnapshot

            # db.snapshot() returns the raw AtomicSnapshot (no element_symbols); the
            # adapter needs the unified PipelineSnapshot — convert before passing.
            asnap = db.snapshot(
                elements=elements,
                wavelength_range=(float(np.min(wavelength)), float(np.max(wavelength))),
                include_levels=True,
            )
            # Tuning knobs. Defaults = the AC1-PASSING config (diagnostic weights ON,
            # gamma=2.0, presence_threshold=0.02, n_configs=1024): on the ak3.1.3 corpus
            # forward_fit P=0.514 R=0.463 F1=0.487, beating the best baseline Comb
            # (F1=0.442) on all three metrics — see docs/jitpipe/J10-forward-id-scope.md.
            # CFLIBS_FF_PRESENCE_THRESHOLD = presence gate; CFLIBS_FF_N_CONFIGS = population;
            # CFLIBS_FF_DIAG_WEIGHTS ('1' on/'0' off) = rank-1 diagnostic per-wavelength
            # weights (host-only; frozen core untouched); CFLIBS_FF_WEIGHT_GAMMA = distinctness exponent.
            # CFLIBS_FF_IEF ('1' on/'0' off) = inverse-element-frequency (TF-IDF) crowding penalty
            # multiplied into the diagnostic weight; CFLIBS_FF_IEF_FLOOR = own-peak fraction a bin
            # must reach to count an element as emitting there (Amato et al. 2010).
            # CFLIBS_FF_REQUIRE_BIC ('1' on/'0' off, default off) = BIC-margin presence
            # gate (AND on top of the correlation gap; only removes calls => raises
            # precision); CFLIBS_FF_BIC_MARGIN = required BIC improvement (default 0.0).
            ff_threshold = float(os.environ.get("CFLIBS_FF_PRESENCE_THRESHOLD", "0.02"))
            ff_n_configs = int(os.environ.get("CFLIBS_FF_N_CONFIGS", "1024"))
            ff_diag_weights = os.environ.get("CFLIBS_FF_DIAG_WEIGHTS", "1") != "0"
            ff_weight_gamma = float(os.environ.get("CFLIBS_FF_WEIGHT_GAMMA", "2.0"))
            ff_use_ief = os.environ.get("CFLIBS_FF_IEF", "1") != "0"
            ff_ief_floor = float(os.environ.get("CFLIBS_FF_IEF_FLOOR", "0.25"))
            ff_require_bic = os.environ.get("CFLIBS_FF_REQUIRE_BIC", "0") == "1"
            ff_bic_margin = float(os.environ.get("CFLIBS_FF_BIC_MARGIN", "0.0"))
            identifier = ForwardFitIdentifier(
                elements,
                snapshot=PipelineSnapshot.from_atomic_snapshot(asnap),
                resolving_power=resolving_power,
                n_configs=ff_n_configs,
                presence_threshold=ff_threshold,
                use_diagnostic_weights=ff_diag_weights,
                weight_gamma=ff_weight_gamma,
                use_ief=ff_use_ief,
                ief_floor_frac=ff_ief_floor,
                require_bic=ff_require_bic,
                bic_margin=ff_bic_margin,
            )
            return identifier.identify(wavelength, intensity)

        runners.append((ALGO_FORWARD_FIT, _run_forward_fit))

    return runners


def _identifier_suite(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
    resolving_power: float,
    basis_library: Optional["BasisLibrary"] = None,
    with_forward_fit: bool = False,
) -> Dict[str, Optional[ElementIdentificationResult]]:
    results: Dict[str, Optional[ElementIdentificationResult]] = {}
    runners = build_identifier_runners(
        wavelength=wavelength,
        intensity=intensity,
        db=db,
        elements=elements,
        resolving_power=resolving_power,
        basis_library=basis_library,
        with_forward_fit=with_forward_fit,
    )
    for name, runner in runners:
        try:
            results[name] = runner()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Identifier %s failed during synthetic benchmark run", name, exc_info=True
            )
            results[name] = None
    return results


@dataclass
class CalibrationOptions:
    """Options for optional wavelength-calibration preprocessing."""

    mode: str = "none"
    max_pair_window_nm: float = 2.0
    inlier_tolerance_nm: float = 0.08
    apply_quality_gate: bool = True
    quality_min_inliers: int = 12
    quality_min_peak_match_fraction: float = 0.35
    quality_max_rmse_nm: float = 0.10
    quality_min_inlier_span_fraction: float = 0.25
    quality_max_abs_correction_nm: float = 2.5


def _apply_calibration(
    wl: np.ndarray,
    intensity: np.ndarray,
    db: AtomicDatabase,
    candidate_elements: List[str],
    calibration: CalibrationOptions,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Optionally calibrate the wavelength axis, returning (wl, cal_meta)."""
    cal_meta: Dict[str, Any] = {
        "calibration_mode": calibration.mode,
        "calibration_applied": False,
        "calibration_success": False,
        "calibration_quality_passed": False,
        "calibration_reason": "",
        "calibration_model": "none",
    }
    if calibration.mode == "none":
        return wl, cal_meta

    cal = calibrate_wavelength_axis(
        wavelength=wl,
        intensity=intensity,
        atomic_db=db,
        elements=candidate_elements,
        mode=calibration.mode,  # type: ignore[arg-type]
        max_pair_window_nm=calibration.max_pair_window_nm,
        inlier_tolerance_nm=calibration.inlier_tolerance_nm,
        apply_quality_gate=calibration.apply_quality_gate,
        quality_min_inliers=calibration.quality_min_inliers,
        quality_min_peak_match_fraction=calibration.quality_min_peak_match_fraction,
        quality_max_rmse_nm=calibration.quality_max_rmse_nm,
        quality_min_inlier_span_fraction=calibration.quality_min_inlier_span_fraction,
        quality_max_abs_correction_nm=calibration.quality_max_abs_correction_nm,
    )
    cal_meta.update(
        {
            "calibration_success": bool(cal.success),
            "calibration_quality_passed": bool(cal.quality_passed),
            "calibration_reason": cal.quality_reason or str(cal.details.get("reason", "")),
            "calibration_model": cal.model,
            "calibration_rmse_nm": float(cal.rmse_nm),
            "calibration_n_inliers": int(cal.n_inliers),
            "calibration_n_candidates": int(cal.n_candidates),
            "calibration_peak_match_fraction": float(cal.matched_peak_fraction),
        }
    )
    if cal.success and cal.quality_passed:
        wl = cal.corrected_wavelength
        cal_meta["calibration_applied"] = True
    return wl, cal_meta


def _build_failed_row(
    spec: BenchmarkSpectrum,
    algo_name: str,
    true_elements: Set[str],
    candidate_elements: List[str],
    resolving_power: float,
    manifest_meta: Dict[str, Any],
    perturb: Dict[str, Any],
    cal_meta: Dict[str, Any],
    recipe: str,
    ignore_elements: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Build the row dict emitted when an identifier failed for a spectrum."""
    return {
        "sample_id": spec.spectrum_id,
        "algorithm": algo_name,
        "failed": True,
        "true_elements": sorted(true_elements),
        "predicted_elements": [],
        # Per-spectrum "don't-care" band (sub-detection-floor traces); carried
        # on the row so per-element/confounder re-aggregation can exclude them
        # the same way the stored confusion counts already do.
        "ignore_elements": sorted(ignore_elements or set()),
        "tp": 0,
        "fp": 0,
        "fn": len(true_elements),
        "tn": max(len(candidate_elements) - len(true_elements), 0),
        "n_peaks": 0,
        "n_matched_peaks": 0,
        "n_unmatched_peaks": 0,
        "peak_match_rate": 0.0,
        "matched_lines_true_elements": 0,
        "total_lines_true_elements": 0,
        "matched_lines_absent_elements": 0,
        "resolving_power": resolving_power,
        "recipe": recipe,
        "label_cardinality": int(spec.label_cardinality or 0),
        "snr_db": perturb.get("snr_db"),
        "continuum_level": perturb.get("continuum_level"),
        "shift_nm": perturb.get("shift_nm"),
        "warp_quadratic_nm": perturb.get("warp_quadratic_nm"),
        **cal_meta,
    }


def _count_line_matches(
    result: ElementIdentificationResult,
    true_elements: Set[str],
    candidate_elements: List[str],
) -> Tuple[int, int, int]:
    """Tally matched/total true-element lines and matched absent-element lines."""
    by_element = _element_by_name(result)
    matched_true = 0
    total_true = 0
    matched_absent = 0
    for element in candidate_elements:
        elem_result = by_element.get(element)
        if elem_result is None:
            continue
        if element in true_elements:
            matched_true += int(elem_result.n_matched_lines)
            total_true += int(elem_result.n_total_lines)
        else:
            matched_absent += int(elem_result.n_matched_lines)
    return matched_true, total_true, matched_absent


def _build_success_row(
    spec: BenchmarkSpectrum,
    algo_name: str,
    result: ElementIdentificationResult,
    true_elements: Set[str],
    candidate_elements: List[str],
    resolving_power: float,
    manifest_meta: Dict[str, Any],
    perturb: Dict[str, Any],
    cal_meta: Dict[str, Any],
    recipe: str,
    ignore_elements: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Build the row dict emitted when an identifier succeeded for a spectrum."""
    predicted_elements = {e.element for e in result.detected_elements}
    counts = confusion_counts(true_elements, predicted_elements, candidate_elements)
    matched_true, total_true, matched_absent = _count_line_matches(
        result, true_elements, candidate_elements
    )
    return {
        "sample_id": spec.spectrum_id,
        "algorithm": algo_name,
        "failed": False,
        "true_elements": sorted(true_elements),
        "predicted_elements": sorted(predicted_elements),
        # Per-spectrum "don't-care" band (sub-detection-floor traces); carried
        # on the row so per-element/confounder re-aggregation can exclude them
        # the same way the stored confusion counts already do.
        "ignore_elements": sorted(ignore_elements or set()),
        **counts,
        "n_peaks": int(result.n_peaks),
        "n_matched_peaks": int(result.n_matched_peaks),
        "n_unmatched_peaks": int(result.n_unmatched_peaks),
        "peak_match_rate": float(result.n_matched_peaks / max(result.n_peaks, 1)),
        "matched_lines_true_elements": int(matched_true),
        "total_lines_true_elements": int(total_true),
        "matched_lines_absent_elements": int(matched_absent),
        "resolving_power": float(resolving_power),
        "recipe": recipe,
        "label_cardinality": int(spec.label_cardinality or 0),
        "snr_db": perturb.get("snr_db"),
        "continuum_level": perturb.get("continuum_level"),
        "shift_nm": perturb.get("shift_nm"),
        "warp_quadratic_nm": perturb.get("warp_quadratic_nm"),
        **cal_meta,
    }


def _evaluate_spectrum_rows(
    spec: BenchmarkSpectrum,
    db: AtomicDatabase,
    candidate_elements: List[str],
    algorithms: List[str],
    presence_threshold: float,
    calibration: CalibrationOptions,
    basis_library: Optional["BasisLibrary"],
    manifest_by_sample: Optional[Dict[str, Dict[str, Any]]],
    with_forward_fit: bool = False,
) -> List[Dict[str, Any]]:
    """Run all identifiers on one spectrum and build its per-algorithm rows."""
    true_elements = derive_truth_elements(
        spec.true_composition, presence_threshold=presence_threshold
    )
    # 3-way detection-floor scoring: in-recipe elements below the floor are
    # "don't-care" — dropped from the SCORING panel (identifiers still search
    # the full candidate panel, since a real instrument cannot know a priori
    # which constituents are sub-detection-limit).
    ignore_elements = derive_ignore_elements(
        spec.true_composition, presence_threshold=presence_threshold
    )
    scoring_panel = [e for e in candidate_elements if e not in ignore_elements]
    wl = np.asarray(spec.wavelength_nm, dtype=float)
    intensity = np.asarray(spec.intensity, dtype=float)

    wl, cal_meta = _apply_calibration(wl, intensity, db, candidate_elements, calibration)

    resolving_power = resolving_power_from_spectrum(spec)
    results = _identifier_suite(
        wavelength=wl,
        intensity=intensity,
        db=db,
        elements=candidate_elements,
        resolving_power=resolving_power,
        basis_library=basis_library,
        with_forward_fit=with_forward_fit,
    )

    manifest_meta = (manifest_by_sample or {}).get(spec.spectrum_id, {})
    perturb = manifest_meta.get("perturbation", {})
    # Prefer an explicit manifest recipe; fall back to provenance / sample_id so
    # per-recipe + stratified reporting works on the manifest-less corpus.
    recipe = manifest_meta.get("recipe") or _derive_recipe(spec, manifest_by_sample)

    spectrum_rows: List[Dict[str, Any]] = []
    for algo_name in algorithms:
        result = results.get(algo_name)
        if result is None:
            spectrum_rows.append(
                _build_failed_row(
                    spec,
                    algo_name,
                    true_elements,
                    scoring_panel,
                    resolving_power,
                    manifest_meta,
                    perturb,
                    cal_meta,
                    recipe,
                    ignore_elements=ignore_elements,
                )
            )
            continue
        spectrum_rows.append(
            _build_success_row(
                spec,
                algo_name,
                result,
                true_elements,
                scoring_panel,
                resolving_power,
                manifest_meta,
                perturb,
                cal_meta,
                recipe,
                ignore_elements=ignore_elements,
            )
        )
    return spectrum_rows


def evaluate_dataset(
    dataset: BenchmarkDataset,
    db: AtomicDatabase,
    candidate_elements: List[str],
    manifest_by_sample: Optional[Dict[str, Dict[str, Any]]] = None,
    presence_threshold: float = 1e-4,
    max_spectra: Optional[int] = None,
    calibration: Optional[CalibrationOptions] = None,
    basis_library: Optional["BasisLibrary"] = None,
    with_forward_fit: bool = False,
    sampling: str = "stratified",
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate all identifiers on synthetic benchmark dataset.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Synthetic benchmark dataset to evaluate.
    db : AtomicDatabase
        Atomic database used by the identifiers.
    candidate_elements : List[str]
        Candidate elements considered for detection.
    manifest_by_sample : dict, optional
        Optional per-sample perturbation metadata.
    presence_threshold : float
        Minimum composition fraction to consider an element present.
    max_spectra : int, optional
        Optional cap on number of spectra evaluated.
    calibration : CalibrationOptions, optional
        Optional wavelength calibration settings.
    basis_library : BasisLibrary, optional
        When provided, the basis-dependent identifiers ``spectral_nnls`` and
        ``hybrid_union`` are added to the suite. When ``None`` (default) only
        the peak-matching trio runs.
    sampling : str
        Spectrum-selection mode when ``max_spectra`` caps the set: one of
        ``"stratified"`` (default; proportional over ``(recipe, cardinality)``
        strata, >=1 per non-empty stratum), ``"shuffle"`` (seeded uniform), or
        ``"sorted"`` (legacy alpha-sort + truncate, kept for reproducibility).
    seed : int
        Seed for the ``stratified`` / ``shuffle`` draws (reproducible).

    Returns
    -------
    Dict[str, Any]
        Dictionary with per-spectrum rows, aggregate summaries, the realized
        sampling provenance, and the ever-present companion / confounder
        diagnostics.
    """
    if calibration is None:
        calibration = CalibrationOptions()

    spectra = _select_spectra(
        dataset.spectra,
        max_spectra=max_spectra,
        sampling=sampling,
        seed=seed,
        manifest_by_sample=manifest_by_sample,
    )

    rows: List[Dict[str, Any]] = []
    algorithms: List[str] = list(_BASE_ALGORITHMS)
    if basis_library is not None:
        algorithms.extend(_BASIS_ALGORITHMS)
    if with_forward_fit:
        algorithms.extend(_FORWARD_FIT_ALGORITHMS)

    for idx, spec in enumerate(spectra, start=1):
        rows.extend(
            _evaluate_spectrum_rows(
                spec=spec,
                db=db,
                candidate_elements=candidate_elements,
                algorithms=algorithms,
                presence_threshold=presence_threshold,
                calibration=calibration,
                basis_library=basis_library,
                manifest_by_sample=manifest_by_sample,
                with_forward_fit=with_forward_fit,
            )
        )

        if idx % 20 == 0 or idx == len(spectra):
            print(f"[synthetic-benchmark] processed {idx}/{len(spectra)} spectra")

    # Elements that actually occur in the (sampled) truth — the realistic panel.
    # Intersected with ``candidate_elements``: an above-floor truth element that
    # was never offered as a candidate could never be predicted, so counting it
    # as a perpetual FN in the ever-present aggregate would penalise the
    # identifier for a panel it was never given.
    candidate_set = set(candidate_elements)
    ever_present = sorted(
        {
            el
            for spec in spectra
            for el, frac in spec.true_composition.items()
            if float(frac) >= float(presence_threshold) and el in candidate_set
        }
    )

    aggregate = summarize_aggregate(rows, candidate_elements)
    aggregate_ever_present = summarize_aggregate(recount_rows(rows, ever_present), ever_present)
    per_element = summarize_per_element(rows, candidate_elements)
    group_metrics = summarize_by_group(rows, candidate_elements)
    confounders = summarize_confounders(rows, candidate_elements)

    sampled_recipe_counts: Dict[str, int] = {}
    sampled_cardinality_counts: Dict[str, int] = {}
    for spec in spectra:
        recipe = _derive_recipe(spec, manifest_by_sample)
        sampled_recipe_counts[recipe] = sampled_recipe_counts.get(recipe, 0) + 1
        card = str(int(spec.label_cardinality or 0))
        sampled_cardinality_counts[card] = sampled_cardinality_counts.get(card, 0) + 1

    return {
        "rows": rows,
        "aggregate_metrics": aggregate,
        "aggregate_metrics_ever_present": aggregate_ever_present,
        "per_element_metrics": per_element,
        "group_metrics": group_metrics,
        "confounder_summary": confounders,
        "candidate_elements": list(candidate_elements),
        "ever_present_elements": ever_present,
        "sampling": sampling,
        "seed": int(seed),
        "sampled_recipe_counts": sampled_recipe_counts,
        "sampled_cardinality_counts": sampled_cardinality_counts,
    }


def recount_rows(rows: List[Dict[str, Any]], panel_elements: Sequence[str]) -> List[Dict[str, Any]]:
    """Re-derive TP/FP/FN/TN against a restricted candidate panel.

    The stored ``true_elements`` / ``predicted_elements`` lists are recomputed
    against ``panel_elements`` only, so absent (never-truth) candidates stop
    inflating TN/FP. Used for the ``ever_present`` companion aggregate. Failed
    rows are passed through unchanged (their counts are already zero), and all
    non-confusion fields are preserved.

    Each row's per-spectrum "don't-care" band (sub-detection-floor traces) is
    removed from the panel for that row, even though the same element may be
    above-floor — and thus a legitimate panel member — for a *different*
    spectrum. Without this, predicting a real-but-sub-floor trace would be
    re-counted as an FP here, contradicting the stored row counts and the
    per-element metrics that already exclude it.
    """
    panel = list(panel_elements)
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        if not row.get("failed"):
            truth = set(row.get("true_elements", []))
            pred = set(row.get("predicted_elements", []))
            ignore = set(row.get("ignore_elements", ()))
            row_panel = [el for el in panel if el not in ignore] if ignore else panel
            counts = confusion_counts(truth, pred, row_panel)
            new_row.update(counts)
        out.append(new_row)
    return out


def summarize_confounders(
    rows: List[Dict[str, Any]],
    candidate_elements: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Rank the FP/FN magnets and flag never-truth candidates per algorithm.

    For each algorithm, sums per-element FP and FN across its (non-failed)
    rows, returning the nonzero entries sorted descending. ``never_truth``
    lists candidate elements that never appear in any truth set across the
    sample — the one-glance hygiene tripwire for absent-but-scored elements
    (e.g. the 7/11 candidates that never occur in any recipe).
    """
    out: Dict[str, Dict[str, Any]] = {}
    algorithms = sorted({row["algorithm"] for row in rows})
    # never-truth is algorithm-independent (truth is the same across algos), but
    # report it per-algorithm for a self-contained block per algorithm.
    truth_seen: Set[str] = set()
    for row in rows:
        truth_seen.update(row.get("true_elements", []))
    never_truth = sorted(el for el in candidate_elements if el not in truth_seen)

    for algorithm in algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm and not row["failed"]]
        fp_by_el: Dict[str, int] = {}
        fn_by_el: Dict[str, int] = {}
        confusion = _confusion_by_element(subset, candidate_elements)
        for element in candidate_elements:
            _, fp, fn, _ = confusion[element]
            if fp:
                fp_by_el[element] = int(fp)
            if fn:
                fn_by_el[element] = int(fn)
        top_fp = sorted(fp_by_el.items(), key=lambda kv: (-kv[1], kv[0]))
        top_fn = sorted(fn_by_el.items(), key=lambda kv: (-kv[1], kv[0]))
        out[algorithm] = {
            "top_fp": [[el, c] for el, c in top_fp],
            "top_fn": [[el, c] for el, c in top_fn],
            "never_truth": never_truth,
        }
    return out


def summarize_aggregate(
    rows: List[Dict[str, Any]], candidate_elements: List[str]
) -> List[Dict[str, Any]]:
    """Aggregate confusion + peak metrics by algorithm."""
    out: List[Dict[str, Any]] = []
    algorithms = sorted({row["algorithm"] for row in rows})
    for algorithm in algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm and not row["failed"]]
        failed_count = sum(1 for row in rows if row["algorithm"] == algorithm and row["failed"])
        tp = sum(int(r["tp"]) for r in subset)
        fp = sum(int(r["fp"]) for r in subset)
        fn = sum(int(r["fn"]) for r in subset)
        tn = sum(int(r["tn"]) for r in subset)
        metrics = compute_binary_metrics(tp, fp, fn, tn)
        out.append(
            {
                "algorithm": algorithm,
                "n_spectra": len(subset),
                "n_failed": failed_count,
                "n_candidate_elements": len(candidate_elements),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **metrics,
                "mean_peak_match_rate": float(
                    np.mean([float(r["peak_match_rate"]) for r in subset]) if subset else 0.0
                ),
                "mean_n_peaks": float(
                    np.mean([float(r["n_peaks"]) for r in subset]) if subset else 0.0
                ),
                "mean_n_matched_peaks": float(
                    np.mean([float(r["n_matched_peaks"]) for r in subset]) if subset else 0.0
                ),
                "mean_true_line_match_fraction": float(
                    np.mean(
                        [
                            float(r["matched_lines_true_elements"])
                            / max(float(r["total_lines_true_elements"]), 1.0)
                            for r in subset
                        ]
                    )
                    if subset
                    else 0.0
                ),
                "mean_absent_matched_lines": float(
                    np.mean([float(r["matched_lines_absent_elements"]) for r in subset])
                    if subset
                    else 0.0
                ),
            }
        )
    return out


def _confusion_by_element(
    subset: List[Dict[str, Any]], candidate_elements: Sequence[str]
) -> Dict[str, Tuple[int, int, int, int]]:
    """Tally ``{element: (tp, fp, fn, tn)}`` for every candidate in one pass.

    Each row's truth/prediction sets are built once (not once per element), so
    the cost is O(rows) set builds rather than O(rows x elements). Rows whose
    per-spectrum "don't-care" band (sub-detection-floor traces) contains an
    element are skipped for that element, so detecting/missing a real-but-sub-
    floor trace is neither rewarded nor penalised — matching the scoring-panel
    semantics already baked into each row's stored confusion counts.
    """
    counts: Dict[str, List[int]] = {el: [0, 0, 0, 0] for el in candidate_elements}
    for row in subset:
        ignore = set(row.get("ignore_elements", ()))
        truth = set(row["true_elements"])
        pred = set(row["predicted_elements"])
        for element in candidate_elements:
            if element in ignore:
                continue
            c = counts[element]
            t = element in truth
            p = element in pred
            if t and p:
                c[0] += 1
            elif p:
                c[1] += 1
            elif t:
                c[2] += 1
            else:
                c[3] += 1
    return {el: (c[0], c[1], c[2], c[3]) for el, c in counts.items()}


def summarize_per_element(
    rows: List[Dict[str, Any]],
    candidate_elements: List[str],
) -> List[Dict[str, Any]]:
    """Compute per-element confusion metrics for each algorithm."""
    out: List[Dict[str, Any]] = []
    algorithms = sorted({row["algorithm"] for row in rows})
    for algorithm in algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm and not row["failed"]]
        confusion = _confusion_by_element(subset, candidate_elements)
        for element in candidate_elements:
            tp, fp, fn, tn = confusion[element]
            metrics = compute_binary_metrics(tp, fp, fn, tn)
            out.append(
                {
                    "algorithm": algorithm,
                    "element": element,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    **metrics,
                }
            )
    return out


def summarize_by_group(
    rows: List[Dict[str, Any]],
    candidate_elements: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate metrics by recipe, label cardinality, and perturbation axis values."""
    group_fields = [
        "recipe",
        "label_cardinality",
        "snr_db",
        "continuum_level",
        "shift_nm",
        "warp_quadratic_nm",
    ]
    output: Dict[str, List[Dict[str, Any]]] = {}
    algorithms = sorted({row["algorithm"] for row in rows})

    for field in group_fields:
        grouped_rows: List[Dict[str, Any]] = []
        values = sorted({value for row in rows if (value := row.get(field)) is not None})
        for value in values:
            for algorithm in algorithms:
                subset = [
                    row
                    for row in rows
                    if row["algorithm"] == algorithm
                    and not row["failed"]
                    and row.get(field) == value
                ]
                if not subset:
                    continue
                tp = sum(int(r["tp"]) for r in subset)
                fp = sum(int(r["fp"]) for r in subset)
                fn = sum(int(r["fn"]) for r in subset)
                tn = sum(int(r["tn"]) for r in subset)
                metrics = compute_binary_metrics(tp, fp, fn, tn)
                grouped_rows.append(
                    {
                        "group_field": field,
                        "group_value": value,
                        "algorithm": algorithm,
                        "n_rows": len(subset),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        **metrics,
                        "mean_peak_match_rate": float(
                            np.mean([float(r["peak_match_rate"]) for r in subset])
                        ),
                    }
                )
        output[field] = grouped_rows
    return output


def _load_manifest_index(manifest_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if manifest_path is None:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
    else:
        rows = json.loads(path.read_text())
    return {row["sample_id"]: row for row in rows if "sample_id" in row}


def run_synthetic_benchmark(
    dataset_path: str,
    db_path: str,
    output_dir: str,
    candidate_elements: Optional[List[str]] = None,
    manifest_path: Optional[str] = None,
    presence_threshold: float = 1e-4,
    max_spectra: Optional[int] = None,
    calibration: Optional[CalibrationOptions] = None,
    include_nnls: bool = False,
    basis_library: Optional["BasisLibrary"] = None,
    basis_library_path: Optional[str] = None,
    basis_instrument_fwhm_nm: float = 0.3,
    with_forward_fit: bool = False,
    sampling: str = "stratified",
    seed: int = 0,
    panel: str = "full",
) -> Dict[str, Any]:
    """Load data, run identifier benchmark, and write artifacts.

    Parameters
    ----------
    include_nnls : bool
        Opt-in switch for the basis-dependent identifiers ``spectral_nnls``
        and ``hybrid_union``. When ``True``, a basis library is used (taken
        from ``basis_library`` if given, else built/loaded at
        ``basis_library_path``). If a basis library cannot be obtained, the
        two extra identifiers are silently skipped and the run proceeds with
        the peak-matching trio only.
    basis_library : BasisLibrary, optional
        A pre-built basis library. Takes precedence over
        ``basis_library_path`` so a controlled before/after can reuse one
        deterministically-built library across runs.
    basis_library_path : str, optional
        HDF5 path to build (if missing) or load the corpus basis library
        from. Only consulted when ``include_nnls`` is True and
        ``basis_library`` is None.
    basis_instrument_fwhm_nm : float
        Instrument FWHM (nm) for the built basis library.
    with_forward_fit : bool
        Opt-in switch for the J10 population forward-fitter
        (:class:`cflibs.jitpipe.forward_id_identifier.ForwardFitIdentifier`).
        Requires JAX; when JAX is unavailable the runner raises and that
        spectrum's ``forward_fit`` row is recorded as failed (graceful skip).
    sampling : str
        Spectrum-selection mode when ``max_spectra`` is set: ``"stratified"``
        (default), ``"shuffle"``, or ``"sorted"`` (legacy). See
        :func:`evaluate_dataset`.
    seed : int
        Seed for the stratified/shuffle draw (reproducible).
    panel : str
        Which aggregate is the headline in ``summary.json``'s
        ``aggregate_metrics``: ``"full"`` (default; all candidate elements) or
        ``"ever_present"`` (only elements that occur in the sampled truth). The
        full-panel and ever-present aggregates are *both* always emitted; this
        only chooses which one is the headline, never silently redefining it.
    """
    dataset = load_benchmark(dataset_path)
    elements = candidate_elements if candidate_elements else list(dataset.elements)
    manifest_index = _load_manifest_index(manifest_path)
    db = AtomicDatabase(db_path)

    active_basis: Optional["BasisLibrary"] = None
    if include_nnls:
        if basis_library is not None:
            active_basis = basis_library
        elif basis_library_path is not None:
            active_basis = build_corpus_basis_library(
                db_path=db_path,
                output_path=basis_library_path,
                elements=elements,
                instrument_fwhm_nm=basis_instrument_fwhm_nm,
            )
        else:
            logger.warning(
                "include_nnls=True but no basis_library or basis_library_path "
                "given; spectral_nnls / hybrid_union will be skipped."
            )
        if active_basis is None:
            logger.warning("Basis library unavailable; running peak-matching trio only.")

    evaluated = evaluate_dataset(
        dataset=dataset,
        db=db,
        candidate_elements=elements,
        manifest_by_sample=manifest_index,
        presence_threshold=presence_threshold,
        max_spectra=max_spectra,
        calibration=calibration,
        basis_library=active_basis,
        with_forward_fit=with_forward_fit,
        sampling=sampling,
        seed=seed,
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluated["rows"]
    aggregate = evaluated["aggregate_metrics"]
    aggregate_ever_present = evaluated["aggregate_metrics_ever_present"]
    per_element = evaluated["per_element_metrics"]
    group_metrics = evaluated["group_metrics"]
    confounder_summary = evaluated["confounder_summary"]
    ever_present_elements = evaluated["ever_present_elements"]
    sampled_recipe_counts = evaluated["sampled_recipe_counts"]
    sampled_cardinality_counts = evaluated["sampled_cardinality_counts"]

    # The headline aggregate honors --panel without silently redefining it: both
    # the full-panel and ever-present aggregates are always emitted below.
    headline_aggregate = aggregate_ever_present if str(panel) == "ever_present" else aggregate

    # summary json
    summary = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "db_path": str(Path(db_path).resolve()),
        "n_rows": len(rows),
        "n_spectra": len({row["sample_id"] for row in rows}),
        "candidate_elements": elements,
        "ever_present_elements": ever_present_elements,
        "presence_threshold": float(presence_threshold),
        "max_spectra": int(max_spectra) if max_spectra is not None else None,
        "sampling": sampling,
        "seed": int(seed),
        "panel": str(panel),
        "sampled_recipe_counts": sampled_recipe_counts,
        "sampled_cardinality_counts": sampled_cardinality_counts,
        "include_nnls": bool(include_nnls),
        "with_forward_fit": bool(with_forward_fit),
        "basis_library_active": active_basis is not None,
        "calibration": (
            calibration.__dict__ if calibration is not None else CalibrationOptions().__dict__
        ),
        "aggregate_metrics": headline_aggregate,
        "aggregate_metrics_full": aggregate,
        "aggregate_metrics_ever_present": aggregate_ever_present,
        "confounder_summary": confounder_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "group_metrics.json").write_text(json.dumps(group_metrics, indent=2))

    with (out_dir / "per_spectrum.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    _write_csv(out_dir / "aggregate_metrics.csv", aggregate)
    _write_csv(out_dir / "aggregate_metrics_ever_present.csv", aggregate_ever_present)
    _write_csv(out_dir / "per_element_metrics.csv", per_element)
    for field, metrics in group_metrics.items():
        _write_csv(out_dir / f"group_metrics_{field}.csv", metrics)

    return {
        "summary": summary,
        "output_dir": str(out_dir),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
