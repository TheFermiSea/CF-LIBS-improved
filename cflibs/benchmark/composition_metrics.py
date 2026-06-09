"""
Compositional accuracy metrics for CF-LIBS benchmark evaluation.

All functions operate on composition vectors represented as dictionaries
mapping element symbols to number fractions (or mass fractions).  The
functions handle edge cases such as zero or negative values by clamping
to a small epsilon before taking logarithms.

References
----------
- Aitchison, J. (1986) "The Statistical Analysis of Compositional Data"
- Egozcue et al. (2003) "Isometric Logratio Transformations for
  Compositional Data Analysis"
- Greenacre (2018) "Compositional Data Analysis in Practice" — pairwise
  log-ratios as subcompositional invariants.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Minimum value used to replace zeros/negatives before log transforms.
_EPSILON = 1e-12

# Default canonical pair list (used as fallback when protocol.yaml cannot be
# located).  The authoritative source is validation/protocol.yaml.
_DEFAULT_SUBCOMPOSITIONAL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Fe", "Si"),
    ("Mg", "Si"),
    ("Ca", "Si"),
    ("Al", "Si"),
)


def _to_positive_array(
    composition: Dict[str, float],
    elements: list[str],
) -> np.ndarray:
    """Extract a positive numpy array from a composition dict.

    Parameters
    ----------
    composition : Dict[str, float]
        Element symbol -> fraction mapping.
    elements : list[str]
        Ordered element keys to extract.

    Returns
    -------
    np.ndarray
        1-D array of positive values (zeros/negatives clipped to epsilon).
    """
    arr = np.array([composition.get(el, 0.0) for el in elements], dtype=np.float64)
    return np.clip(arr, _EPSILON, None)


# ---------------------------------------------------------------------------
# Aitchison distance
# ---------------------------------------------------------------------------


def aitchison_distance(
    c_true: Dict[str, float],
    c_pred: Dict[str, float],
) -> float:
    """Compute the Aitchison distance between two compositions.

    The Aitchison distance is the Euclidean norm of the difference of
    centered log-ratio (CLR) vectors.  It is scale-invariant: multiplying
    all components by a constant does not change the distance.

    .. math::

        d_A(x, y) = \\sqrt{\\sum_i \\left(
            \\ln \\frac{x_i}{g(x)} - \\ln \\frac{y_i}{g(y)}
        \\right)^2}

    where :math:`g(\\cdot)` is the geometric mean.

    Parameters
    ----------
    c_true : Dict[str, float]
        Ground truth composition.
    c_pred : Dict[str, float]
        Predicted composition.

    Returns
    -------
    float
        Aitchison distance (non-negative).
    """
    elements = sorted(set(c_true.keys()) | set(c_pred.keys()))
    if not elements:
        return 0.0

    x = _to_positive_array(c_true, elements)
    y = _to_positive_array(c_pred, elements)

    clr_x = np.log(x) - np.mean(np.log(x))
    clr_y = np.log(y) - np.mean(np.log(y))

    return float(np.sqrt(np.sum((clr_x - clr_y) ** 2)))


# ---------------------------------------------------------------------------
# CLR transform
# ---------------------------------------------------------------------------


def clr_transform(c: Dict[str, float]) -> Dict[str, float]:
    """Centered log-ratio (CLR) transform.

    .. math::

        \\text{clr}(x)_i = \\ln \\frac{x_i}{g(x)}

    Parameters
    ----------
    c : Dict[str, float]
        Composition (element -> fraction).

    Returns
    -------
    Dict[str, float]
        CLR-transformed values keyed by element symbol.
    """
    elements = sorted(c.keys())
    if not elements:
        return {}

    arr = _to_positive_array(c, elements)
    log_arr = np.log(arr)
    clr_vals = log_arr - np.mean(log_arr)

    return {el: float(v) for el, v in zip(elements, clr_vals)}


# ---------------------------------------------------------------------------
# ILR transform
# ---------------------------------------------------------------------------


def _helmert_basis(d: int) -> np.ndarray:
    """Construct a (d-1) x d Helmert sub-composition basis.

    Each row i (0-indexed) contrasts the first i+1 components against
    the (i+2)-th component, with appropriate normalisation so the rows
    are orthonormal in the Aitchison inner product.

    Parameters
    ----------
    d : int
        Number of compositional parts.

    Returns
    -------
    np.ndarray
        Shape ``(d-1, d)`` orthonormal contrast matrix.
    """
    basis = np.zeros((d - 1, d))
    for i in range(d - 1):
        r = i + 1
        basis[i, :r] = 1.0 / r
        basis[i, r] = -1.0
        basis[i] *= np.sqrt(r / (r + 1.0))
    return basis


def ilr_transform(c: Dict[str, float]) -> np.ndarray:
    """Isometric log-ratio (ILR) transform using the Helmert sub-composition.

    Maps a D-part composition to a (D-1)-dimensional real vector that
    lives in an unconstrained Euclidean space.  Euclidean distance in
    ILR coordinates equals the Aitchison distance.

    Parameters
    ----------
    c : Dict[str, float]
        Composition (element -> fraction).

    Returns
    -------
    np.ndarray
        ILR coordinates, shape ``(D-1,)``.
    """
    elements = sorted(c.keys())
    d = len(elements)
    if d <= 1:
        return np.array([], dtype=np.float64)

    arr = _to_positive_array(c, elements)
    log_arr = np.log(arr)

    basis = _helmert_basis(d)
    return basis @ log_arr


def ilr_inverse(coords: np.ndarray, elements: list[str]) -> Dict[str, float]:
    """Inverse ILR transform: map ILR coordinates back to a composition.

    Parameters
    ----------
    coords : np.ndarray
        ILR coordinates, shape ``(D-1,)``.
    elements : list[str]
        Ordered element symbols (length D).

    Returns
    -------
    Dict[str, float]
        Composition that sums to 1.
    """
    d = len(elements)
    if d <= 1:
        return {elements[0]: 1.0} if elements else {}

    basis = _helmert_basis(d)
    log_arr = basis.T @ coords
    arr = np.exp(log_arr)
    arr /= arr.sum()
    return {el: float(v) for el, v in zip(elements, arr)}


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------


def rmse_composition(
    c_true: Dict[str, float],
    c_pred: Dict[str, float],
) -> float:
    """Root mean squared error between two compositions.

    Parameters
    ----------
    c_true : Dict[str, float]
        Ground truth composition.
    c_pred : Dict[str, float]
        Predicted composition.

    Returns
    -------
    float
        RMSE (non-negative).
    """
    elements = sorted(set(c_true.keys()) | set(c_pred.keys()))
    if not elements:
        return 0.0

    true_arr = np.array([c_true.get(el, 0.0) for el in elements], dtype=np.float64)
    pred_arr = np.array([c_pred.get(el, 0.0) for el in elements], dtype=np.float64)

    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def per_element_error(
    c_true: Dict[str, float],
    c_pred: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """Per-element absolute and relative error.

    Parameters
    ----------
    c_true : Dict[str, float]
        Ground truth composition.
    c_pred : Dict[str, float]
        Predicted composition.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Mapping of element -> (absolute_error, relative_error).
        Relative error is ``inf`` when the true value is zero.
    """
    elements = sorted(set(c_true.keys()) | set(c_pred.keys()))
    result: Dict[str, Tuple[float, float]] = {}

    for el in elements:
        t = c_true.get(el, 0.0)
        p = c_pred.get(el, 0.0)
        abs_err = abs(p - t)
        rel_err = abs_err / t if t > 0 else float("inf") if abs_err > 0 else 0.0
        result[el] = (abs_err, rel_err)

    return result


# ---------------------------------------------------------------------------
# Subcompositional ratio errors (Tier-1 gate companion)
# ---------------------------------------------------------------------------


def _resolve_protocol_path(protocol_path: Optional[Path]) -> Optional[Path]:
    """Resolve the protocol.yaml path, walking upward when not given explicitly."""
    if protocol_path is not None:
        return Path(protocol_path)
    here = Path(__file__).resolve()
    for parent in here.parents:
        guess = parent / "validation" / "protocol.yaml"
        if guess.is_file():
            return guess
    return None


def _parse_subcompositional_pairs(raw_pairs: Any) -> List[Tuple[str, str]]:
    """Convert raw YAML pair entries into validated ``(num, den)`` tuples."""
    pairs: List[Tuple[str, str]] = []
    for entry in raw_pairs:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            pairs.append((str(entry[0]), str(entry[1])))
    return pairs


def load_subcompositional_pairs(
    protocol_path: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """Load the canonical subcompositional pair list from the validation protocol.

    The pair list is read from ``validation/protocol.yaml`` under the
    ``subcompositional_pairs`` key.  When ``protocol_path`` is ``None`` the
    function searches upwards from this module for the first ``validation``
    directory containing ``protocol.yaml``.

    Falls back to the canonical ``[(Fe, Si), (Mg, Si), (Ca, Si), (Al, Si)]``
    list when the file cannot be located or parsed (e.g. in installed-package
    contexts where the protocol file is not co-located).

    Parameters
    ----------
    protocol_path : Path, optional
        Explicit path to ``protocol.yaml``.  When ``None`` the file is
        located by walking upward from this module.

    Returns
    -------
    list[tuple[str, str]]
        Ordered ``(numerator, denominator)`` pairs.
    """
    candidate = _resolve_protocol_path(protocol_path)

    if candidate is None or not candidate.is_file():
        return list(_DEFAULT_SUBCOMPOSITIONAL_PAIRS)

    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(candidate.read_text())
    except Exception:
        return list(_DEFAULT_SUBCOMPOSITIONAL_PAIRS)

    raw_pairs = data.get("subcompositional_pairs") if isinstance(data, dict) else None
    if not raw_pairs:
        return list(_DEFAULT_SUBCOMPOSITIONAL_PAIRS)

    pairs = _parse_subcompositional_pairs(raw_pairs)
    return pairs or list(_DEFAULT_SUBCOMPOSITIONAL_PAIRS)


def subcompositional_ratio_errors(
    predicted_composition: Mapping[str, float],
    true_composition: Mapping[str, float],
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """Per-pair |log(r̂/r*)| subcompositional ratio errors.

    For each ``(numerator, denominator)`` pair, computes
    ``abs(log(predicted_ratio) - log(true_ratio))`` where each ratio is
    ``composition[numerator] / composition[denominator]``.  Zero / negative
    components are clipped to ``_EPSILON`` before taking the logarithm so a
    completely missing element produces a large (but finite) error rather
    than NaN.

    A pair is only scored when **both** elements appear in ``true_composition``
    with strictly positive values; otherwise the pair is reported as ``NaN``
    and downstream aggregators must filter accordingly.

    Subcompositional ratios are scale-invariant under closure: missing
    detection of a VUV-only element inflates all detected concentrations
    proportionally, but the ratios between detected elements remain
    physically correct.  This is the Aitchison subcompositional-coherence
    property — the metric that surfaces ratio-only violations cleanly when
    the global Aitchison distance is dominated by closure noise.

    Parameters
    ----------
    predicted_composition : Mapping[str, float]
        Predicted composition (element symbol -> fraction).
    true_composition : Mapping[str, float]
        Ground-truth composition.
    pairs : Sequence[Tuple[str, str]], optional
        Pair list as ``(numerator, denominator)`` tuples.  When ``None``,
        loads from ``validation/protocol.yaml`` via
        :func:`load_subcompositional_pairs`.

    Returns
    -------
    Dict[str, float]
        Mapping of ``"<num>/<den>"`` to ``|log(r̂/r*)|``.  Pairs with
        ill-defined truth ratios map to ``float('nan')``.
    """
    if pairs is None:
        pairs = load_subcompositional_pairs()

    errors: Dict[str, float] = {}
    for numerator, denominator in pairs:
        key = f"{numerator}/{denominator}"

        true_num = float(true_composition.get(numerator, 0.0))
        true_den = float(true_composition.get(denominator, 0.0))

        if true_num <= 0.0 or true_den <= 0.0:
            errors[key] = float("nan")
            continue

        pred_num = max(float(predicted_composition.get(numerator, 0.0)), _EPSILON)
        pred_den = max(float(predicted_composition.get(denominator, 0.0)), _EPSILON)

        true_ratio = true_num / true_den
        pred_ratio = pred_num / pred_den
        errors[key] = float(abs(np.log(pred_ratio) - np.log(true_ratio)))

    return errors


# ---------------------------------------------------------------------------
# Concentration stratification (majors / minors / traces)
# ---------------------------------------------------------------------------


# Default thresholds matching validation/protocol.yaml composition_strata.
# Concentrations are mass fractions (i.e. 0.05 == 5 wt%).
DEFAULT_STRATA_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "majors": {
        "concentration_floor": 0.05,
        "rd_max": 0.05,
    },
    "minors": {
        "concentration_floor": 0.001,
        "concentration_ceiling": 0.05,
        "rd_max": 0.20,
    },
    "traces": {
        "concentration_ceiling": 0.001,
        # No rd_max — traces are MDL-bounded, not RD-bounded.  The pass
        # criterion uses an ``mdl_factor`` multiple of the LOQ when supplied.
        "mdl_factor": 3.0,
    },
}


def classify_stratum(
    certified_concentration: float,
    thresholds: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> str:
    """Classify a concentration into ``majors`` / ``minors`` / ``traces``.

    Boundary convention (matches docs/VALIDATION_METRICS.md §2.1):
    - ``majors``: ``c > majors.concentration_floor`` (strictly greater)
    - ``minors``: ``minors.concentration_floor <= c <= minors.concentration_ceiling``
    - ``traces``: ``c < traces.concentration_ceiling``

    Note that the boundary at exactly the floor (5 wt%) falls into ``minors``
    and the boundary at exactly the ceiling (0.1 wt%) also falls into
    ``minors``.  This is the inclusive-minors convention used by Tier-1 gate
    review, where a borderline value's stricter (5%) and looser (20%) RD
    requirements both apply, and we elect to enforce the looser one.

    Parameters
    ----------
    certified_concentration : float
        Mass fraction of the certified value (NOT predicted).
    thresholds : Mapping, optional
        Stratum thresholds; falls back to :data:`DEFAULT_STRATA_THRESHOLDS`.

    Returns
    -------
    str
        ``"majors"``, ``"minors"``, or ``"traces"``.
    """
    cfg = thresholds or DEFAULT_STRATA_THRESHOLDS
    majors_floor = float(cfg.get("majors", {}).get("concentration_floor", 0.05))
    traces_ceiling = float(cfg.get("traces", {}).get("concentration_ceiling", 0.001))

    if certified_concentration > majors_floor:
        return "majors"
    if certified_concentration < traces_ceiling:
        return "traces"
    return "minors"


def _bucket_records(
    per_element_records: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Mapping[str, float]],
) -> Dict[str, List[Tuple[str, float, float]]]:
    """Group records into majors/minors/traces buckets keyed by certified concentration.

    Each bucket entry is ``(element, true, rd)``.  Records with ``true <= 0``
    are dropped because they cannot be stratified by certified concentration.
    """
    buckets: Dict[str, List[Tuple[str, float, float]]] = {
        "majors": [],
        "minors": [],
        "traces": [],
    }
    for record in per_element_records:
        true = float(record.get("true", 0.0))
        if true <= 0.0:
            continue
        pred = float(record.get("predicted", 0.0))
        element = str(record.get("element", "?"))
        rd = abs(pred - true) / true
        stratum = classify_stratum(true, cfg)
        buckets[stratum].append((element, true, rd))
    return buckets


def _empty_stratum_summary() -> Dict[str, Any]:
    """Summary for a stratum with no observations (vacuously passes)."""
    return {
        "n_elements": 0,
        "n_spectra": 0,
        "mean_rd": float("nan"),
        "median_rd": float("nan"),
        "p95_rd": float("nan"),
        "pass": True,  # vacuously passes when no observations
    }


def _traces_passed(
    entries: List[Tuple[str, float, float]],
    cfg: Mapping[str, Mapping[str, float]],
    loq_lookup: Optional[Mapping[str, float]],
) -> bool:
    """Pass criterion for the MDL-bounded ``traces`` stratum."""
    # traces: MDL-bounded.  When a per-element LOQ table is provided,
    # require mean_rd to stay within (mdl_factor * LOQ / mean(true)).
    mdl_factor = float(cfg.get("traces", {}).get("mdl_factor", 3.0))
    if not loq_lookup:
        # No LOQ table available: do not gate (return informational
        # pass=True so the absence of LOQ data does not silently fail).
        return True
    bounded = []
    for element, true, rd in entries:
        loq = float(loq_lookup.get(element, float("inf")))
        if true > 0:
            bounded.append(rd <= mdl_factor * (loq / true))
    return bool(bounded) and all(bounded)


def _stratum_passed(
    stratum: str,
    mean_rd: float,
    entries: List[Tuple[str, float, float]],
    cfg: Mapping[str, Mapping[str, float]],
    loq_lookup: Optional[Mapping[str, float]],
) -> bool:
    """Determine whether a populated stratum passes its gate criterion."""
    if stratum in ("majors", "minors"):
        rd_max = float(cfg.get(stratum, {}).get("rd_max", 0.20))
        return mean_rd <= rd_max
    return _traces_passed(entries, cfg, loq_lookup)


def stratify_per_element_errors(
    per_element_records: Iterable[Mapping[str, Any]],
    thresholds: Optional[Mapping[str, Mapping[str, float]]] = None,
    loq_lookup: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-element / per-spectrum records into stratum summaries.

    Each input record is a mapping with at least the keys:

    - ``element``: element symbol (str)
    - ``true``: certified concentration (mass fraction)
    - ``predicted``: predicted concentration (mass fraction)

    The relative deviation (RD) is ``|predicted - true| / true``.  Records
    with ``true <= 0`` are dropped (cannot be stratified by certified
    concentration).

    Stratification is by **certified** concentration, not predicted, so a
    PR cannot move an element between strata by inflating its prediction.

    Pass / fail per stratum:
    - ``majors``: ``mean_rd <= majors.rd_max`` (default 5%)
    - ``minors``: ``mean_rd <= minors.rd_max`` (default 20%)
    - ``traces``: ``mean_rd <= mdl_factor * (LOQ_for_element / true)`` when
      ``loq_lookup`` is supplied; otherwise ``True`` (informational only).

    Parameters
    ----------
    per_element_records : Iterable[Mapping]
        Per-spectrum records with ``element``, ``true``, ``predicted`` keys.
    thresholds : Mapping, optional
        Override default stratum thresholds.
    loq_lookup : Mapping[str, float], optional
        Element symbol -> LOQ (mass fraction) for trace MDL gating.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        ``{stratum: {n_elements, n_spectra, mean_rd, median_rd, p95_rd, pass}}``.
    """
    cfg = thresholds or DEFAULT_STRATA_THRESHOLDS
    buckets = _bucket_records(per_element_records, cfg)

    summary: Dict[str, Dict[str, Any]] = {}
    for stratum, entries in buckets.items():
        if not entries:
            summary[stratum] = _empty_stratum_summary()
            continue

        rds = np.array([rd for _, _, rd in entries], dtype=np.float64)
        elements = sorted({el for el, _, _ in entries})

        mean_rd = float(np.mean(rds))
        median_rd = float(np.median(rds))
        p95_rd = float(np.percentile(rds, 95))

        passed = _stratum_passed(stratum, mean_rd, entries, cfg, loq_lookup)

        summary[stratum] = {
            "n_elements": len(elements),
            "n_spectra": len(entries),
            "mean_rd": mean_rd,
            "median_rd": median_rd,
            "p95_rd": p95_rd,
            "pass": passed,
        }

    return summary
