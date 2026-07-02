"""Robust, conditioning-gated one-point calibration (OPC) for known-matrix mode.

This module promotes the winning real-steel lever combination
(``tests/benchmarks/real_steel/lever_l3b_robust_opc.py`` + ``best_config_v2.py``,
honest held-out RMSEP 10.12 wt%, a 4x reduction over the calibration-free
baseline) into the shipped, physics-only CF-LIBS pipeline as an **opt-in**
known-matrix calibration step.

The default calibration-free path is byte-for-byte unchanged: nothing here runs
unless a caller explicitly builds an :class:`OPCCalibration` from certified
standards and applies it. The DED real goal *is* a known matrix (Ti-6Al-4V,
steel), so this is the deployment sweet spot rather than a corner case.

Two phases (Cavalcanti et al. SAB 2013 OPC + Zhao et al. PST 2018 optimal-``T``):

1. **Calibration** (offline, from certified standards only): :func:`calibrate_opc`
   selects well-conditioned standards via an in-sample-only conditioning gate,
   fixes the plasma temperature at the mean of the selected standards' optimal
   ``T*``, and derives one robust per-element relative-sensitivity factor ``F``
   by the *geometric mean* of ``C_true / C_rec`` over the selected standards
   (clamp-saturated / degenerate factors filtered out).
2. **Inference** (online, on unknowns): :func:`apply_opc` rescales each line
   observation's intensity by ``F[element]`` in place, after which the caller
   solves at ``calibration.robust_T_K``.

Structural honesty (the property the 2026-06-09 audit demanded):

* :func:`calibrate_opc` sees *only* standards — it cannot peek at unknowns, so
  held-out leakage is impossible by construction.
* :func:`apply_opc` takes *only* observations + a calibration; it never reads a
  recovered composition, so there is no positive-feedback loop.

Physics-only: pure NumPy arithmetic (geometric mean, clamp, conditioning gate)
plus a caller-injected composition-recovery callback that wraps the existing
Saha-Boltzmann solver. No banned ML API is imported (ruff TID251 + the AST
blocklist scanner both pass).

References
----------
Cavalcanti et al., Spectrochim. Acta B 87 (2013) 51-56,
doi:10.1016/j.sab.2013.05.016.
Zhao et al., Plasma Sci. Technol. 20 (2018) 035502,
doi:10.1088/2058-6272/aa9b1f.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

from cflibs.core.constants import HC_EV_NM, KB_EV

__all__ = [
    "OPCCalibration",
    "Standard",
    "StandardRecovery",
    "ConditioningAssessment",
    "choose_optimal_temperature",
    "assess_conditioning",
    "calibrate_opc",
    "apply_opc",
    "measure_line_width",
    "select_optically_thin_lines",
    "cdsb_raw_ordinate",
    "cdsb_global_scale",
    "apply_cdsb_matrix",
    "DEFAULT_COND_T_K",
    "DEFAULT_T_GRID",
    "F_MIN",
    "F_MAX",
    "MIN_WT",
    "MAX_INSAMPLE_RMSEP",
    "MAX_NONMATRIX_FRAC",
    "THIN_FILTER_MIN_CORR_LINES",
    "THIN_FILTER_SNR_MIN",
    "THIN_FILTER_THICK_RATIO",
    "THIN_FILTER_INSTR_WIDTH_PCT",
    "THIN_FILTER_WIDTH_HALF_NM",
    "THIN_FILTER_MIN_KEEP",
    "CDSB_COG_EXP",
    "CDSB_THIN_REF_WIDTH_NM",
    "CDSB_LAMBDA_REF_NM",
    "CDSB_KL_MAX",
    "CDSB_KL_FLOOR",
]

# --- a-priori constants (lifted unchanged from the winning benchmark lever) ---

#: Lower / upper clamp band for the OPC relative-sensitivity factor ``F``. A
#: factor that saturates either edge means the standard did not usefully recover
#: that element; such values are filtered out of the robust geometric mean.
F_MIN: float = 1e-2
F_MAX: float = 1e2
#: Floor on a wt% before it enters the ``F = C_true / C_rec`` ratio (guards the
#: degenerate case where the uncorrected solve recovers ~0 for an element).
MIN_WT: float = 1e-3
#: Fixed reference temperature (K) for the conditioning gate — a physically
#: reasonable steel/alloy-plasma value, never tuned to held-out truth.
DEFAULT_COND_T_K: float = 9000.0
#: A standard whose own uncorrected in-sample RMSEP exceeds this (wt%) is
#: keystone-collapsed and is dropped from the calibration.
MAX_INSAMPLE_RMSEP: float = 20.0
#: A *non-matrix* element recovered above this fraction (%) signals a collapsed
#: closure (a trace minor soaking the composition) and drops the standard.
MAX_NONMATRIX_FRAC: float = 60.0
#: Coarse optimal-``T`` scan grid (K); the robust mean is insensitive to step.
DEFAULT_T_GRID: tuple[float, ...] = tuple(float(t) for t in range(7000, 12001, 1000))

# --- a-priori constants for the optically-thin line filter (real-steel L5) ---
# Physical El Sherbini 2005 / IRSAC width-gating values, lifted unchanged from the
# winning benchmark lever (``tests/benchmarks/real_steel/lever_l5_fe_selfabs.py``,
# mode="thin_filter"). NOT tuned to the held-out gate.

#: An element is considered for filtering only with >= this many well-measured lines.
THIN_FILTER_MIN_CORR_LINES: int = 3
#: Peak-SNR floor for a line to count toward the width statistics / filter.
THIN_FILTER_SNR_MIN: float = 3.0
#: An element is "optically thick" only when its MEDIAN measured width exceeds the
#: per-spectrum instrument width by this factor (below it the element is thin and
#: all its lines are kept -- this is what leaves optically-thin trace minors alone).
THIN_FILTER_THICK_RATIO: float = 1.15
#: Percentile of the well-measured line widths taken as the instrument width (the
#: optically-thin floor of the width distribution).
THIN_FILTER_INSTR_WIDTH_PCT: float = 10.0
#: Half-window (nm) for the second-moment width measurement: wide enough to capture
#: a self-absorption-broadened profile at moderate LIBS resolution.
THIN_FILTER_WIDTH_HALF_NM: float = 0.6
#: A thick element is filtered only if at least this many optically-thin anchor
#: lines remain (else its lines are kept as-is, so a noisy spectrum never strips
#: the matrix anchor entirely).
THIN_FILTER_MIN_KEEP: int = 2

#: Relative (fractional) floor on a line's intensity uncertainty when an ordinate
#: is replaced/re-extracted (CD-SB injection, constrained peak-locked extraction):
#: ``sigma_I = max(I * RELATIVE_INTENSITY_UNCERTAINTY_FLOOR, 1e-12)``.
RELATIVE_INTENSITY_UNCERTAINTY_FLOOR: float = 0.05


# --- public data structures --------------------------------------------------


@dataclass(frozen=True)
class StandardRecovery:
    """Result of recovering a standard's composition at a fixed temperature.

    Attributes
    ----------
    composition : Mapping[str, float]
        Recovered wt% per element from the *uncorrected* (no-OPC) solve.
    converged : bool
        Whether the underlying solver converged.
    degenerate : bool
        Whether the solver flagged a degenerate composition (e.g.
        ``quality_metrics['degenerate_composition']``).
    """

    composition: Mapping[str, float]
    converged: bool = True
    degenerate: bool = False


@dataclass(frozen=True)
class Standard:
    """A certified calibration standard plus its composition-recovery callback.

    Attributes
    ----------
    name : str
        Standard identifier (provenance only).
    certified : Mapping[str, float]
        Certified composition (wt%) over the modeled element set. Renormalized
        to 100% internally, so absolute scaling is irrelevant.
    recover : Callable[[float], StandardRecovery]
        Uncorrected composition recovery at a fixed temperature (K). This
        closure must depend only on *this* standard's own spectrum — which is
        what makes the calibration un-peekable. The shipped pipeline supplies a
        closure wrapping the Saha-Boltzmann solver; tests supply a synthetic one.
    """

    name: str
    certified: Mapping[str, float]
    recover: Callable[[float], StandardRecovery]


def _recover_or_degenerate(standard: "Standard", T_K: float) -> StandardRecovery:
    """Recover a standard at fixed ``T_K``, representing strict failures as data.

    ``calibrate_opc``'s conditioning sweep deliberately probes off-optimal
    temperatures where closure collapse is *expected*; ``StandardRecovery``
    carries an explicit ``degenerate`` flag for exactly that outcome, and the
    conditioning gate then excludes the standard. Under strict / no-fallback
    mode (``CFLIBS_NO_FALLBACK``) the solver raises a typed
    :class:`~cflibs.inversion.common.strict.SolverFailure` instead of flagging —
    which, uncaught, lets one bad standard abort the whole calibration. This
    wrapper converts that typed failure into the caller's explicit degenerate
    contract: the failure is *represented and acted on* (standard excluded),
    not silently substituted — the strict-mode semantics for a caller that
    handles the outcome.
    """
    from cflibs.inversion.common.strict import SolverFailure

    try:
        return standard.recover(float(T_K))
    except SolverFailure:
        return StandardRecovery(composition={}, converged=False, degenerate=True)


@dataclass(frozen=True)
class ConditioningAssessment:
    """Per-standard conditioning-gate diagnostics (in-sample only)."""

    passed: bool
    converged: bool
    degenerate: bool
    in_sample_rmsep: float
    nonmatrix_max_pct: float
    matrix_element: str | None


@dataclass(frozen=True)
class OPCCalibration:
    """A frozen known-matrix OPC calibration (tiny, JSON-serializable payload).

    Attributes
    ----------
    robust_T_K : float
        Fixed plasma temperature (K) = mean of the selected standards' optimal
        ``T*``. The inference solve is held at this temperature.
    F : dict[str, float]
        Per-element relative-sensitivity factor (geometric mean over selected
        standards of ``C_true / C_rec``). Multiply each observation's intensity
        by ``F[element]`` before solving.
    selected_standards : list[str]
        Names of the standards that passed the conditioning gate (provenance).
    conditioning_rule : str
        Human-readable description of the exact a-priori gate used.
    cdsb_scale : float
        Global Columnar-Density Saha-Boltzmann (CD-SB) unit scale ``S`` derived
        from the standards only (geometric mean of ``geomean(measured matrix
        areas) / geomean(CD-SB raw)``). A single constant for every spectrum;
        consumed by :func:`apply_cdsb_matrix` only when the opt-in CD-SB matrix
        mode is enabled. Defaults to ``1.0`` (CD-SB off / plain OPC, byte-
        identical). See :func:`apply_cdsb_matrix` for the physics.
    """

    robust_T_K: float
    F: dict[str, float]
    selected_standards: list[str]
    conditioning_rule: str
    cdsb_scale: float = 1.0


class _OPCObservation(Protocol):
    """Structural type for an OPC-rescalable line observation."""

    element: str
    intensity: float


class _LineWidthObservation(Protocol):
    """Structural type for a line observation the thin filter can measure."""

    element: str
    wavelength_nm: float


#: Bound to :class:`_LineWidthObservation` so :func:`select_optically_thin_lines`
#: returns the same concrete observation type it was given (a subset).
_ThinObsT = TypeVar("_ThinObsT", bound=_LineWidthObservation)


# --- internal helpers (pure NumPy; lifted from the benchmark lever) ----------


def _renorm100(comp: Mapping[str, float]) -> dict[str, float]:
    """Renormalize positive, finite wt% entries to sum to 100%."""
    vals = {e: float(v) for e, v in comp.items() if np.isfinite(float(v)) and float(v) > 0.0}
    tot = sum(vals.values())
    if tot <= 0.0:
        return {}
    return {e: v / tot * 100.0 for e, v in vals.items()}


def _composition_rmsep(
    certified_norm: Mapping[str, float], recovered: Mapping[str, float]
) -> float:
    """Overall RMSEP (wt%) of a recovered composition vs renormalized certified.

    Mirrors the benchmark ``harness.score`` for a single sample: the recovered
    composition is renormalized to 100% over all finite entries, then per-element
    errors against ``certified_norm`` are root-mean-squared.
    """
    ps = sum(v for v in recovered.values() if np.isfinite(v))
    errs: list[float] = []
    for e, tv in certified_norm.items():
        pn = (recovered.get(e, 0.0) / ps * 100.0) if ps > 0 else float("nan")
        err = pn - tv
        if np.isfinite(err):
            errs.append(err)
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(np.asarray(errs) ** 2)))


def _geomean(values: Sequence[float]) -> float:
    """Geometric mean ``exp(mean(log v))`` of an already-filtered positive sequence.

    Callers filter to strictly-positive finite values first (the degenerate /
    clamp-saturation rules differ between callers); an empty sequence is the
    caller's degenerate case and must not be passed.
    """
    return float(np.exp(np.mean(np.log(values))))


def _geomean_F(F_list: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Per-element geometric mean of OPC factor dicts, clamp-saturated filtered.

    DEGENERATE-VALUE FILTER (the key robustness step from lever L3b): a factor
    that saturated the clamp band (``F_e == F_MIN`` or ``F_MAX``) means that
    standard did not usefully recover element ``e`` (``C_rec_e ~ 0`` blew the
    ratio up to the clamp). Such a value carries no relative-sensitivity
    information; averaging it in lets one degenerate standard inflate ``F_e`` and
    make ``e`` soak the closure on unknowns. We therefore drop clamp-saturated,
    non-finite, and non-positive values before the geometric mean (multiplicative
    correction -> geometric, not arithmetic). If every standard is degenerate for
    ``e`` we apply no correction (``F_e = 1``).
    """
    if not F_list:
        return {}
    eps = 1e-6
    keys: set[str] = set()
    for F in F_list:
        keys.update(F.keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [
            v
            for F in F_list
            for v in (float(F.get(k, 1.0)),)
            if np.isfinite(v) and v > 0.0 and abs(v - F_MIN) > eps and abs(v - F_MAX) > eps
        ]
        if not vals:
            out[k] = 1.0
            continue
        gm = float(np.exp(np.mean(np.log(vals))))
        out[k] = float(np.clip(gm, F_MIN, F_MAX))
    return out


def _derive_F(standard: Standard, T_K: float) -> dict[str, float]:
    """Per-element OPC factor ``F_e = C_true_e / C_rec_e`` for one standard.

    ``C_rec`` is the standard's *uncorrected* recovery at ``T_K``, renormalized
    to its own modeled basis; ``C_true`` is the renormalized certified
    composition. Both ratio terms are floored at :data:`MIN_WT` and the result is
    clamped to ``[F_MIN, F_MAX]``.
    """
    certified_norm = _renorm100(standard.certified)
    rec_norm = _renorm100(_recover_or_degenerate(standard, T_K).composition)
    F: dict[str, float] = {}
    for e, tv in certified_norm.items():
        cn = rec_norm.get(e, 0.0)
        f = max(tv, MIN_WT) / max(cn, MIN_WT)
        F[e] = float(np.clip(f, F_MIN, F_MAX))
    return F


# --- public API --------------------------------------------------------------


def choose_optimal_temperature(
    standard: Standard, t_grid: Sequence[float] = DEFAULT_T_GRID
) -> float:
    """Scan ``t_grid`` and return the ``T*`` minimizing the standard's RMSEP.

    Zhao et al. 2018: CF-LIBS composition is hypersensitive to plasma ``T``; the
    optimal fixed temperature is the one that best reproduces a matrix-matched
    standard's *own* certified composition. Uses only the standard's own data and
    truth — never any held-out sample.

    Parameters
    ----------
    standard : Standard
        The matrix-matched calibration standard.
    t_grid : Sequence[float]
        Candidate temperatures (K).

    Returns
    -------
    float
        The temperature (K) minimizing the standard's in-sample RMSEP.
    """
    grid = list(t_grid)
    if not grid:
        raise ValueError("t_grid must be non-empty")
    certified_norm = _renorm100(standard.certified)
    best_T = float(grid[0])
    best_err = float("inf")
    for T_K in grid:
        rec = _recover_or_degenerate(standard, T_K).composition
        err = _composition_rmsep(certified_norm, rec)
        if np.isfinite(err) and err < best_err:
            best_err = err
            best_T = float(T_K)
    return best_T


def assess_conditioning(
    standard: Standard,
    *,
    cond_T_K: float = DEFAULT_COND_T_K,
    max_insample_rmsep: float = MAX_INSAMPLE_RMSEP,
    max_nonmatrix_frac: float = MAX_NONMATRIX_FRAC,
) -> ConditioningAssessment:
    """Judge a standard's conditioning from its OWN data + truth only.

    A standard passes when its uncorrected fixed-``T`` solve (at ``cond_T_K``):

    * converged;
    * is not flagged degenerate;
    * has in-sample RMSEP below ``max_insample_rmsep`` (keystone-collapsed
      standards have a huge in-sample RMSEP); and
    * has no *non-matrix* element recovered above ``max_nonmatrix_frac`` (a trace
      minor recovered as a large fraction is the keystone-collapse signature; the
      matrix element — argmax of the certified truth — is exempt because it is
      genuinely dominant).

    This is the un-peekable selection criterion: it uses no other sample's data.
    """
    certified_norm = _renorm100(standard.certified)
    rec = _recover_or_degenerate(standard, cond_T_K)
    converged = bool(rec.converged)
    degenerate = bool(rec.degenerate)
    in_rmsep = _composition_rmsep(certified_norm, rec.composition)
    matrix_el = max(certified_norm, key=lambda e: certified_norm[e]) if certified_norm else None
    rec_norm = _renorm100(rec.composition)
    nonmatrix_max = 0.0
    for e, v in rec_norm.items():
        if e == matrix_el:
            continue
        nonmatrix_max = max(nonmatrix_max, v)
    passed = bool(
        converged
        and (not degenerate)
        and np.isfinite(in_rmsep)
        and in_rmsep < max_insample_rmsep
        and nonmatrix_max < max_nonmatrix_frac
    )
    return ConditioningAssessment(
        passed=passed,
        converged=converged,
        degenerate=degenerate,
        in_sample_rmsep=float(in_rmsep),
        nonmatrix_max_pct=float(nonmatrix_max),
        matrix_element=matrix_el,
    )


def calibrate_opc(
    standards: Sequence[Standard],
    *,
    cond_T_K: float = DEFAULT_COND_T_K,
    t_grid: Sequence[float] = DEFAULT_T_GRID,
    max_insample_rmsep: float = MAX_INSAMPLE_RMSEP,
    max_nonmatrix_frac: float = MAX_NONMATRIX_FRAC,
) -> OPCCalibration:
    """Build a robust known-matrix OPC calibration from certified standards only.

    Implements the winning lever-L3b algorithm exactly:

    1. **Conditioning gate** (in-sample only): keep the standards that pass
       :func:`assess_conditioning`. If none pass, fall back to the single best
       in-sample standard (least keystone-collapsed).
    2. **Robust temperature**: ``robust_T_K = mean(T*)`` over the selected
       standards, each ``T*`` from :func:`choose_optimal_temperature`.
    3. **Robust factor**: ``F_e = geomean(C_true_e / C_rec_e)`` over the selected
       standards (derived at ``robust_T_K``), with clamp-saturated / degenerate
       per-element factors filtered out (:func:`_geomean_F`).

    Structural honesty: this function receives *only* standards, so it cannot
    observe any unknown sample.

    Parameters
    ----------
    standards : Sequence[Standard]
        Certified calibration standards (at least one).
    cond_T_K : float
        Fixed temperature (K) for the conditioning gate.
    t_grid : Sequence[float]
        Optimal-``T`` scan grid (K).
    max_insample_rmsep, max_nonmatrix_frac : float
        A-priori conditioning thresholds.

    Returns
    -------
    OPCCalibration
        The frozen calibration (robust ``T``, per-element ``F``, provenance).
    """
    stds = list(standards)
    if not stds:
        raise ValueError("calibrate_opc requires at least one standard")

    assessments = [
        assess_conditioning(
            s,
            cond_T_K=cond_T_K,
            max_insample_rmsep=max_insample_rmsep,
            max_nonmatrix_frac=max_nonmatrix_frac,
        )
        for s in stds
    ]
    selected = [i for i, a in enumerate(assessments) if a.passed]
    if not selected:  # nothing passed -> fall back to the best in-sample standard
        selected = [
            min(
                range(len(stds)),
                key=lambda i: (
                    assessments[i].in_sample_rmsep
                    if np.isfinite(assessments[i].in_sample_rmsep)
                    else float("inf")
                ),
            )
        ]

    T_stars = [choose_optimal_temperature(stds[i], t_grid) for i in selected]
    robust_T = float(np.mean(T_stars))

    F_per_std = [_derive_F(stds[i], robust_T) for i in selected]
    robust_F = _geomean_F(F_per_std)

    rule = (
        "converged AND not degenerate AND in-sample uncorrected rmsep < "
        f"{max_insample_rmsep:g} wt% AND non-matrix recovered fraction < "
        f"{max_nonmatrix_frac:g}% (gate at fixed T={cond_T_K:.0f} K)"
    )
    return OPCCalibration(
        robust_T_K=robust_T,
        F=robust_F,
        selected_standards=[stds[i].name for i in selected],
        conditioning_rule=rule,
    )


def apply_opc(observations: Iterable[_OPCObservation], calibration: OPCCalibration) -> None:
    """Rescale each observation's intensity by ``F[element]`` in place.

    This is the inference-phase step: after calling it, the caller solves the
    constrained Saha-Boltzmann balance at ``calibration.robust_T_K``.

    STRUCTURAL HONESTY: this function reads *only* the observations and the
    calibration. It never reads a recovered composition — there is no
    positive-feedback loop (the failure the 2026-06-09 audit condemned). Its
    signature is exactly ``(observations, calibration)``.

    Parameters
    ----------
    observations : Iterable[_OPCObservation]
        Line observations with mutable ``element`` and ``intensity`` (and an
        optional ``intensity_uncertainty``). Modified in place.
    calibration : OPCCalibration
        The known-matrix calibration whose per-element ``F`` is applied.
    """
    F = calibration.F
    for o in observations:
        f = float(F.get(o.element, 1.0))
        o.intensity = float(o.intensity) * f
        unc = getattr(o, "intensity_uncertainty", None)
        if unc is not None:
            setattr(o, "intensity_uncertainty", max(float(unc) * f, 1e-12))


# --- optically-thin line filter (opt-in known-matrix pre-OPC selection) ------
#
# Real-steel L5 winning lever (``tests/benchmarks/real_steel/lever_l5_fe_selfabs``
# mode="thin_filter"; docs/research/real-steel-opc-promotion.md). The matrix
# element (Fe in steel, 85-95 wt%) is optically thick, so its strong lines are
# self-absorbed -- width-broadened and line-center-saturated -- by a *different*
# amount in every spectrum (variable Fe content). A single shared OPC scalar
# ``F_Fe`` captures only the AVERAGE matrix self-absorption bias, not the
# per-spectrum variation.
#
# Instead of *correcting* the self-absorbed line intensities -- which double-counts
# the OPC scalar ``F`` (a uniform per-element intensity boost lives in the SAME
# scalar subspace as ``F`` and only re-centers it; benchmarked regression) -- this
# filter DROPS the width-broadened lines of each optically-thick element and
# anchors the closure on the optically-thin subset (lines at the instrument-width
# floor). Because it is a line-SELECTION change, not an intensity scaling, it
# composes cleanly with the OPC ``F`` (no double-correction).
#
# Held-out real-steel gate (36 samples, honest leave-one-out): off 10.124 ->
# thin_filter 9.561 wt% overall (Fe 20.66 -> 19.62). It reads ONLY measured line
# widths / intensities -- never a recovered composition -- so there is no
# positive-feedback loop. The width-ratio criterion and constants are lifted
# unchanged from the lever.


def _local_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Robust median-edge local baseline over a window.

    A straight line interpolated between the median of the first and last ~1/6 of
    the window (a per-window continuum estimate robust to single-sample noise).
    Shared by :func:`measure_line_width` (OPC width measure) and the constrained
    peak-locked extractor; the line-detection extractor deliberately uses a
    different plain-trapz baseline and is NOT routed through here.
    """
    k = max(1, x.size // 6)
    return np.interp(x, [x[0], x[-1]], [float(np.median(y[:k])), float(np.median(y[-k:]))])


def measure_line_width(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    half_nm: float = THIN_FILTER_WIDTH_HALF_NM,
) -> tuple[float, float]:
    """Robust FWHM (nm) and peak SNR of the line at ``center_nm`` from the spectrum.

    Second-moment width after a robust median-edge local baseline (same baseline
    policy as the line extractor), with the peak height measured against the
    window-edge noise. Returns ``(nan, 0.0)`` when the line cannot be measured.

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured spectrum axes (nm, intensity).
    center_nm : float
        Line-center wavelength (nm) to measure around.
    half_nm : float
        Half-window (nm) over which the local profile is integrated.

    Returns
    -------
    tuple of float
        ``(fwhm_nm, peak_snr)``; ``(nan, 0.0)`` for an unmeasurable line.
    """
    wl = np.asarray(wavelength, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    mask = (wl >= center_nm - half_nm) & (wl <= center_nm + half_nm)
    if int(mask.sum()) < 5:
        return float("nan"), 0.0
    x = wl[mask]
    y = inten[mask].astype(float)
    k = max(1, x.size // 6)
    y = y - _local_baseline(x, y)
    peak = float(y.max())
    if peak <= 0:
        return float("nan"), 0.0
    edge = np.concatenate([y[:k], y[-k:]])
    noise = float(np.std(edge)) if edge.size > 1 else 0.0
    snr = peak / noise if noise > 0 else float("inf")
    yp = np.clip(y, 0.0, None)
    tot = float(yp.sum())
    if tot <= 0:
        return float("nan"), 0.0
    mu = float(np.sum(x * yp) / tot)
    var = float(np.sum((x - mu) ** 2 * yp) / tot)
    fwhm = 2.354820045 * np.sqrt(max(var, 1e-12))
    return fwhm, snr


def _assess_thickness(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence[_LineWidthObservation],
    *,
    min_corr_lines: int,
    snr_min: float,
    thick_ratio: float,
    instr_width_pct: float,
    width_half_nm: float,
) -> tuple[dict[int, float], dict[int, float], dict[str, list[int]], float, dict[str, float]]:
    """Measure per-line widths/SNRs and identify the optically-thick elements.

    Shared a-priori front-end of :func:`select_optically_thin_lines` and
    :func:`cdsb_raw_ordinate`: per-line FWHM + peak SNR, the per-spectrum
    instrument width (a low percentile of the well-measured widths), and the set
    of optically-thick elements (>= ``min_corr_lines`` well-measured lines whose
    median width exceeds ``thick_ratio * instr``).

    Returns ``(widths, snrs, by_el, instr, thick)`` where ``thick`` maps each
    optically-thick element to its ``median_width / instr`` ratio. ``instr`` is
    ``0.0`` and ``thick`` empty when no line is well measured (the callers treat
    that as "leave everything untouched").
    """
    widths: dict[int, float] = {}
    snrs: dict[int, float] = {}
    by_el: dict[str, list[int]] = {}
    for i, o in enumerate(observations):
        fwhm, snr = measure_line_width(wavelength, intensity, o.wavelength_nm, width_half_nm)
        widths[i] = fwhm
        snrs[i] = snr
        by_el.setdefault(o.element, []).append(i)

    good_all = [widths[i] for i in widths if np.isfinite(widths[i]) and snrs[i] >= snr_min]
    if not good_all:
        return widths, snrs, by_el, 0.0, {}
    instr = float(np.percentile(good_all, instr_width_pct))
    if instr <= 0:
        return widths, snrs, by_el, 0.0, {}

    thick: dict[str, float] = {}
    for el, idxs in by_el.items():
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        if len(good) < min_corr_lines:
            continue
        median_w = float(np.median([widths[i] for i in good]))
        if median_w / instr < thick_ratio:
            continue
        thick[el] = median_w / instr
    return widths, snrs, by_el, instr, thick


def select_optically_thin_lines(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence[_ThinObsT],
    *,
    min_corr_lines: int = THIN_FILTER_MIN_CORR_LINES,
    snr_min: float = THIN_FILTER_SNR_MIN,
    thick_ratio: float = THIN_FILTER_THICK_RATIO,
    instr_width_pct: float = THIN_FILTER_INSTR_WIDTH_PCT,
    width_half_nm: float = THIN_FILTER_WIDTH_HALF_NM,
    min_keep: int = THIN_FILTER_MIN_KEEP,
) -> list[_ThinObsT]:
    """Drop the width-broadened (self-absorbed) lines of each optically-thick element.

    IRSAC / SC-LIBS anchor strategy: instead of *correcting* the intensities of
    self-absorbed matrix (e.g. Fe) lines, **exclude** them and keep only the
    optically-thin subset (lines whose measured width is at the per-spectrum
    instrument-width floor). This is a line-*selection* change -- not an intensity
    scaling -- so it cannot live in the same scalar subspace as the OPC ``F`` and
    therefore cannot double-correct with it.

    Per spectrum: measure each line's FWHM + peak SNR; take the instrument width as
    a low percentile of the well-measured widths; an element with at least
    ``min_corr_lines`` well-measured lines whose MEDIAN width exceeds
    ``thick_ratio * instr`` is optically thick, and its lines wider than
    ``thick_ratio * instr`` are dropped (keeping the thin anchors). An optically
    thin element is left untouched, and a thick element is only filtered if at
    least ``min_keep`` thin anchors remain (else its lines are kept as-is so a
    noisy spectrum never strips the matrix anchor entirely). Reads only measured
    widths -- no composition feedback.

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured spectrum axes (nm, intensity) the widths are measured from.
    observations : Sequence
        Line observations, each exposing ``element`` and ``wavelength_nm``.
    min_corr_lines, snr_min, thick_ratio, instr_width_pct, width_half_nm, min_keep
        A-priori filter parameters (defaults are the lifted lever constants).

    Returns
    -------
    list
        The kept subset of ``observations`` (the same objects, optically-thick
        broadened lines removed). When no line is well measured the full input is
        returned unchanged.
    """
    widths, snrs, by_el, instr, thick = _assess_thickness(
        wavelength,
        intensity,
        observations,
        min_corr_lines=min_corr_lines,
        snr_min=snr_min,
        thick_ratio=thick_ratio,
        instr_width_pct=instr_width_pct,
        width_half_nm=width_half_nm,
    )
    if instr <= 0:
        return list(observations)

    drop: set[int] = set()
    for el in thick:
        idxs = by_el[el]
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        thin = [i for i in good if widths[i] / instr < thick_ratio]
        if len(thin) < min_keep:
            continue  # too few thin anchors -> keep the element's lines as-is
        keep = set(thin)
        for i in idxs:
            if i not in keep:
                drop.add(i)
    return [o for i, o in enumerate(observations) if i not in drop]


# --- Columnar-Density Saha-Boltzmann (CD-SB) matrix ordinate (opt-in) ---------
#
# Real-steel L6 winning lever (``tests/benchmarks/real_steel/lever_l6_cdsb.py``
# mode="cdsb"; Cristoforetti & Tognoni 2013, Spectrochim. Acta B 86, 51; the
# Pisa-group curve-of-growth relation). The optically-thick matrix element (Fe in
# steel, 85-95 wt%) is self-absorbed by a *different* amount in every spectrum
# (variable matrix content), so a single shared OPC scalar ``F`` cannot carry the
# per-spectrum variation -- and the L5 thin filter *discards* those lines entirely.
#
# CD-SB instead KEEPS the self-absorbed matrix lines and REPLACES each line's
# solver ordinate with a columnar-density value read from the line's WIDTH (not
# its suppressed peak intensity). In classical CF-LIBS the Boltzmann-plot ordinate
# ``ln(I*lambda/(g_k*A_ki))`` is proportional to the upper-level density ``n_k`` and
# is only valid for optically-thin lines. CD-SB rewrites the plot in terms of the
# lower-level *columnar density* ``n_i^l`` (Cristoforetti & Tognoni eqn 12), read
# from the integrated absorption coefficient ``kl``. ``kl`` is recovered from the
# measured self-absorption broadening via the Pisa-group curve-of-growth relation
# (eqn 13): ``Delta_lambda / Delta_lambda_0 = ((1 - e^{-kl}) / kl)^{-COG_EXP}``,
# where ``Delta_lambda_0`` is the optically-thin reference FWHM.
#
# The shipped Saha-Boltzmann solver ingests per-line *intensities* and internally
# forms ``n_k`` from ``I*lambda/(g_k*A_ki)``. So to put the thick element on the
# CD-SB ordinate we inject, for each of its lines, the intensity the solver needs
# to recover the CD-SB columnar density; after the ``A_ki`` and per-element
# constants cancel this is
#
#     I_cdsb(line)  proportional_to  g_k * kl * exp(-(E_k - E_i)/(k_B T)) / lambda^5
#
# with ``E_k - E_i = hc/lambda`` the photon energy. The MEASURED (self-absorbed)
# intensity does NOT enter -- only the WIDTH does. This is an ordinate REPLACEMENT,
# not an intensity scale: it does not live in the same multiplicative subspace as
# the OPC scalar ``F`` (which would double-correct), so it composes cleanly with
# OPC. A single global unit scale ``S`` (standards only) places the width-derived
# ordinate at the measured matrix lines' average brightness; ``F`` is then
# RE-DERIVED on this CD-SB pipeline so it corrects only the static residual, never
# the per-spectrum self-absorption the CD-SB ordinate already encodes.
#
# Held-out real-steel gate (36 samples, honest leave-one-out): plain OPC 10.124,
# L5 thin_filter 9.561 -> CD-SB 8.383 wt% overall (Fe 19.6 -> 16.5). It reads ONLY
# measured line widths -- never a recovered composition -- so there is no
# positive-feedback loop. The COG exponent (0.56) and the optically-thin reference
# width (0.08 nm) are physical CD-SB / El Sherbini constants, not gate-tuned.

#: Pisa-group curve-of-growth exponent: ``dl/dl0 = ((1-e^-kl)/kl)^-CDSB_COG_EXP``.
CDSB_COG_EXP: float = 0.56
#: Optically-thin intrinsic reference width ``Delta_lambda_0`` (nm), El Sherbini 2005.
CDSB_THIN_REF_WIDTH_NM: float = 0.08
#: Numeric reference for the ``(lref/lambda)^5`` factor (cancels into ``S``/``F``).
CDSB_LAMBDA_REF_NM: float = 370.0
#: Cap on the recovered optical depth ``kl`` (saturated self-absorption guard).
CDSB_KL_MAX: float = 50.0
#: Floor so a marginally-broadened thick-element line still casts a columnar vote.
CDSB_KL_FLOOR: float = 0.1


class _CDSBObservation(Protocol):
    """Structural type for a CD-SB-replaceable line observation."""

    element: str
    wavelength_nm: float
    g_k: float
    intensity: float


def _cdsb_geomean(vals: Sequence[float]) -> float:
    """Geometric mean of strictly-positive finite values (1.0 if none)."""
    v = [float(x) for x in vals if np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else 1.0


def _solve_cdsb_kl_from_ratio(
    ratio: float,
    *,
    cog_exp: float = CDSB_COG_EXP,
    kl_max: float = CDSB_KL_MAX,
    n_iter: int = 60,
) -> float:
    """Invert the COG relation ``dl/dl0 = ((1-e^-kl)/kl)^-cog_exp`` for ``kl``.

    ``ratio`` is the measured self-absorption broadening
    ``Delta_lambda/Delta_lambda_0`` (>= 1). Returns 0 for an unbroadened
    (optically-thin) line and a bisection root otherwise. ``g(kl) =
    (1-e^-kl)/kl`` decreases monotonically from 1 (kl->0) to 0, so the root is
    bracketed on ``[0, kl_max]`` and found by bisection.
    """
    if not np.isfinite(ratio) or ratio <= 1.0:
        return 0.0
    target = ratio ** (-1.0 / cog_exp)  # g(kl) we must match, in (0, 1)

    def g(kl: float) -> float:
        return 1.0 if kl <= 0.0 else (1.0 - np.exp(-kl)) / kl

    if g(kl_max) >= target:  # even at kl_max not thick enough -> cap
        return kl_max
    lo, hi = 0.0, kl_max
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if g(mid) > target:
            lo = mid  # g too high -> need larger kl
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cdsb_raw_ordinate(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence[_CDSBObservation],
    T_K: float,
    *,
    thin_ref_width_nm: float = CDSB_THIN_REF_WIDTH_NM,
    min_corr_lines: int = THIN_FILTER_MIN_CORR_LINES,
    snr_min: float = THIN_FILTER_SNR_MIN,
    thick_ratio: float = THIN_FILTER_THICK_RATIO,
    instr_width_pct: float = THIN_FILTER_INSTR_WIDTH_PCT,
    width_half_nm: float = THIN_FILTER_WIDTH_HALF_NM,
) -> tuple[dict[int, float], dict[str, float]]:
    """Per-line CD-SB raw columnar-density ordinate for every optically-thick element.

    Returns ``({obs_index: raw_value}, diagnostics)``. ``raw`` (before the global
    unit scale ``S``) is ``g_k * kl * exp(-(E_k-E_i)/(k_B T)) * (lref/lambda)^5``
    with ``kl`` from the COG inversion (:func:`_solve_cdsb_kl_from_ratio`) of the
    instrument-deconvolved line width. Only the WIDTH enters -- never the measured
    (self-absorbed) intensity. Optically-thin elements (median width at the
    instrument floor) get no entries (the caller leaves them on measured
    intensity). The thickness gate is identical to
    :func:`select_optically_thin_lines` (same a-priori width-ratio constants).

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured spectrum axes (nm, intensity) the widths are measured from.
    observations : Sequence
        Line observations, each exposing ``element``, ``wavelength_nm`` and
        ``g_k`` (upper-level statistical weight).
    T_K : float
        Plasma temperature (K) for the ``exp(-(E_k-E_i)/(k_B T))`` Boltzmann factor.
    thin_ref_width_nm, min_corr_lines, snr_min, thick_ratio, instr_width_pct, width_half_nm
        A-priori CD-SB / width-gating parameters (defaults are the lifted lever
        constants).

    Returns
    -------
    tuple of (dict, dict)
        ``(raw_by_index, thickness_diag)``: per-line raw ordinate keyed by the
        observation index, plus a ``{element: median_width / instr}`` diagnostic
        for each thick element corrected.
    """
    widths, snrs, by_el, instr, thick = _assess_thickness(
        wavelength,
        intensity,
        observations,
        min_corr_lines=min_corr_lines,
        snr_min=snr_min,
        thick_ratio=thick_ratio,
        instr_width_pct=instr_width_pct,
        width_half_nm=width_half_nm,
    )
    if instr <= 0:
        return {}, {}
    ref = max(float(thin_ref_width_nm), 1e-6)

    raw: dict[int, float] = {}
    diag: dict[str, float] = {}
    for el, ratio in thick.items():
        idxs = by_el[el]
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        median_w = float(np.median([widths[i] for i in good]))
        diag[el] = ratio
        for i in idxs:
            # Marginal / unmeasurable lines fall back to the element median width so
            # every thick-element line still gets a (floored) columnar-density vote.
            w = widths[i] if (np.isfinite(widths[i]) and snrs[i] >= snr_min) else median_w
            intrinsic = float(np.sqrt(max(w**2 - instr**2, 0.0)))
            kl = max(_solve_cdsb_kl_from_ratio(intrinsic / ref), CDSB_KL_FLOOR)
            o = observations[i]
            lam = float(o.wavelength_nm)
            photon_ev = HC_EV_NM / lam
            boltz = float(np.exp(-photon_ev / (KB_EV * T_K)))
            raw_i = float(o.g_k) * kl * boltz * (CDSB_LAMBDA_REF_NM / lam) ** 5
            if np.isfinite(raw_i) and raw_i > 0:
                raw[i] = raw_i
    return raw, diag


def cdsb_global_scale(observations: Sequence[_CDSBObservation], raw: Mapping[int, float]) -> float:
    """Per-standard CD-SB unit factor ``geomean(measured thick areas) / geomean(raw)``.

    Placing the CD-SB ordinate at the same average brightness as the measured
    matrix (e.g. Fe) lines ON THE STANDARD keeps the re-derived ``F`` near O(1)
    (inside the OPC clamp band). The run-level scale ``S`` is the geometric mean
    of this factor across the selected standards and is applied UNCHANGED to every
    spectrum, so it never removes the per-spectrum ``kl`` magnitude (the
    matrix-content adaptivity the CD-SB ordinate carries).

    Returns ``1.0`` when there is no CD-SB-replaced line.
    """
    if not raw:
        return 1.0
    meas = [float(observations[i].intensity) for i in raw]
    return _cdsb_geomean(meas) / max(_cdsb_geomean([raw[i] for i in raw]), 1e-300)


def apply_cdsb_matrix(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence[_CDSBObservation],
    T_K: float,
    scale: float,
) -> tuple[int, float]:
    """In-place: replace each thick-element line's intensity with its CD-SB ordinate.

    The self-absorbed matrix lines are KEPT (vs :func:`select_optically_thin_lines`,
    which drops them) and placed on the columnar-density ordinate ``scale * raw``
    (width-derived via :func:`cdsb_raw_ordinate`, NOT the measured intensity).
    Optically-thin elements are untouched. After calling this the caller applies
    the OPC ``F`` rescale (:func:`apply_opc`) and solves at the calibration's
    ``robust_T_K``. Reads only the measured line widths -- never a recovered
    composition -- so there is no positive-feedback loop.

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured spectrum axes (nm, intensity) the widths are measured from.
    observations : Sequence
        Line observations with mutable ``intensity`` (and an optional
        ``intensity_uncertainty``); thick-element entries are modified in place.
    T_K : float
        Plasma temperature (K) for the CD-SB Boltzmann factor (the calibration's
        ``robust_T_K``).
    scale : float
        Global CD-SB unit scale ``S`` (standards only; ``OPCCalibration.cdsb_scale``).

    Returns
    -------
    tuple of (int, float)
        ``(n_replaced, max_raw)``: number of lines moved onto the CD-SB ordinate
        and the maximum raw (pre-scale) columnar value (diagnostic).
    """
    raw, _diag = cdsb_raw_ordinate(wavelength, intensity, observations, T_K)
    if not raw:
        return 0, 0.0
    n = 0
    mx = 0.0
    for i, raw_i in raw.items():
        o = observations[i]
        o.intensity = float(scale) * raw_i
        unc = getattr(o, "intensity_uncertainty", None)
        if unc is not None:
            setattr(
                o,
                "intensity_uncertainty",
                max(o.intensity * RELATIVE_INTENSITY_UNCERTAINTY_FLOOR, 1e-12),
            )
        n += 1
        mx = max(mx, raw_i)
    return n, mx
