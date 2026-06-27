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
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

__all__ = [
    "OPCCalibration",
    "Standard",
    "StandardRecovery",
    "ConditioningAssessment",
    "choose_optimal_temperature",
    "assess_conditioning",
    "calibrate_opc",
    "apply_opc",
    "DEFAULT_COND_T_K",
    "DEFAULT_T_GRID",
    "F_MIN",
    "F_MAX",
    "MIN_WT",
    "MAX_INSAMPLE_RMSEP",
    "MAX_NONMATRIX_FRAC",
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
    """

    robust_T_K: float
    F: dict[str, float]
    selected_standards: list[str]
    conditioning_rule: str


class _OPCObservation(Protocol):
    """Structural type for an OPC-rescalable line observation."""

    element: str
    intensity: float


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
    rec = standard.recover(float(T_K)).composition
    tot = sum(v for v in rec.values() if np.isfinite(v) and v > 0)
    F: dict[str, float] = {}
    for e, tv in certified_norm.items():
        rv = rec.get(e, 0.0)
        cn = (rv / tot * 100.0) if (tot > 0 and np.isfinite(rv)) else 0.0
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
        rec = standard.recover(float(T_K)).composition
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
    rec = standard.recover(float(cond_T_K))
    converged = bool(rec.converged)
    degenerate = bool(rec.degenerate)
    in_rmsep = _composition_rmsep(certified_norm, rec.composition)
    matrix_el = max(certified_norm, key=lambda e: certified_norm[e]) if certified_norm else None
    ps = sum(v for v in rec.composition.values() if np.isfinite(v) and v > 0)
    nonmatrix_max = 0.0
    if ps > 0:
        for e, v in rec.composition.items():
            if e == matrix_el or not np.isfinite(v):
                continue
            nonmatrix_max = max(nonmatrix_max, v / ps * 100.0)
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
