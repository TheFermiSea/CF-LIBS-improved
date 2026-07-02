"""In-plasma relative-g·A self-calibration (physics-first-principles audit Issue 1a).

Physics
-------
Two spectral lines *i*, *j* that share the **same upper level** *k* have an
intensity ratio that is *exactly* fixed by their atomic constants::

    I_i / I_j = (g_k A_i / lambda_i) / (g_k A_j / lambda_j)
              = (A_i lambda_j) / (A_j lambda_i)

The shared ``g_k`` and — crucially — the shared upper-level population ``n_k``
cancel. The identity therefore holds **independently of** temperature, electron
density, the level populations, LTE, and even the self-absorption-free-ness of
the *level* (both lines see the same ``n_k``). It is one of the very few
model-free statements available in CF-LIBS.

Consequently, in the Boltzmann ordinate ``y = ln(I lambda / (g A))`` every line
that shares an upper level must land on the **same y-value** (same ``E_k`` ->
same point on the Boltzmann plane), for perfect data. Any spread *within* a
shared-upper-level group is a direct measurement of the **relative** ``A_ki``
error of that group (plus intensity noise and *differential* self-absorption).

This module measures that spread and refines the per-line ``A_ki`` used by the
Boltzmann fit so the group agrees. It is a *relative* calibration: it removes
line-to-line ``A_ki`` scatter within a group, but it **cannot** recover the
group's absolute ``A_ki`` scale (that is the lifetime/branching-fraction
anchoring sub-step, audit Issue 1b — out of scope here). This is exactly why an
internally-consistent single-source set ("Kurucz beat NIST") outperforms a
many-source set: it reproduces that benefit from the plasma itself, with no
external standard.

References
----------
* Physics-first-principles audit, Issue 1 / atomic #2 (in-plasma relative-gf
  self-calibration).
* Aragón & Aguilera (2008); Lawler/Den Hartog Wisconsin-group branching-
  fraction/lifetime practice (relative branching fractions accurate to a few %).

Design notes
------------
* Pure-physics NumPy only (no ML libraries — the repository physics-only
  constraint applies to this shipped module).
* Grouping uses the DB ``upp_level_id`` when an :class:`AtomicDatabase` is
  supplied and the line can be resolved, and otherwise falls back to the
  ``(element, stage, E_k, g_k)`` fingerprint — which *is* the physical identity
  of the upper level (same species, same upper energy, same degeneracy).
* Differential self-absorption contaminates the ratio, so lines flagged
  optically thick (via ``exclude_mask``) are excluded from the group anchor and
  never receive a g·A correction.
* Corrections are shrunk toward 1 by the per-line noise estimate (empirical
  Bayes) so that pure intensity noise is not chased.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.data_structures import LineObservation

logger = get_logger("inversion.ga_selfcal")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SharedUpperGroup:
    """A set of observed lines that share the same upper level.

    Attributes
    ----------
    element, ionization_stage
        Emitting species shared by every member.
    level_key
        String identity of the upper level: the DB ``upp_level_id`` when
        resolved, otherwise the ``E_k``/``g_k`` fingerprint.
    indices
        Indices into the original ``observations`` list.
    E_k_ev, g_k
        Upper-level energy (eV) and degeneracy shared by the group (from the
        first member; members agree by construction).
    resolved_by_id
        True if the group was formed from the DB ``upp_level_id`` rather than
        the ``E_k``/``g_k`` fingerprint fallback.
    """

    element: str
    ionization_stage: int
    level_key: str
    indices: list[int]
    E_k_ev: float
    g_k: float
    resolved_by_id: bool = False


@dataclass
class RelativeGAResidual:
    """Per-line relative-g·A residual measured within a shared-upper group.

    Attributes
    ----------
    index
        Index into the original ``observations`` list.
    element, ionization_stage, level_key
        Identity of the line's shared-upper group.
    residual
        ``r_i = y_i - <y>_group`` in the natural-log Boltzmann ordinate. Equal
        to ``-ln(A_i^db / A_i^true)`` up to the (unmeasurable) group mean, i.e.
        a direct estimate of the *relative* A_ki error of this line.
    sigma
        1-sigma noise on ``residual`` propagated from the intensity uncertainty
        (log-domain), ``sqrt(sigma_y_i^2 - 1/sum_w)``.
    group_size
        Number of lines contributing to the group anchor.
    excluded
        True if the line was excluded from its group anchor (optically thick /
        differential-SA suspect); such lines receive no correction.
    """

    index: int
    element: str
    ionization_stage: int
    level_key: str
    residual: float
    sigma: float
    group_size: int
    excluded: bool = False


@dataclass
class RelativeGACalibration:
    """Outcome of a relative-g·A self-calibration pass.

    Attributes
    ----------
    residuals
        One :class:`RelativeGAResidual` per line that belonged to a usable
        multi-line group.
    corrections
        ``{obs_index: multiplicative g·A correction}`` (shrunk). Lines absent
        from the map are uncorrected (factor 1.0).
    correction_sigma
        ``{obs_index: measured relative-A_ki uncertainty}`` (fractional), the
        physically-derived replacement for the fabricated rel_int->grade value.
    tau2
        Method-of-moments estimate of the population variance of the log-g·A
        error (shrinkage hyper-parameter).
    n_groups
        Number of usable multi-line groups.
    n_lines_corrected
        Number of lines that received a (non-unit) correction.
    """

    residuals: list[RelativeGAResidual] = field(default_factory=list)
    corrections: dict[int, float] = field(default_factory=dict)
    correction_sigma: dict[int, float] = field(default_factory=dict)
    tau2: float = 0.0
    n_groups: int = 0
    n_lines_corrected: int = 0


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def _fingerprint_key(obs: LineObservation, ek_decimals: int) -> str:
    """(E_k, g_k) fingerprint of an upper level — its physical identity."""
    return (
        f"ek{round(float(obs.E_k_ev), ek_decimals):.{ek_decimals}f}|g{int(round(float(obs.g_k)))}"
    )


def _resolve_upper_level_ids(
    observations: Sequence[LineObservation],
    atomic_db,
    wl_tol_nm: float,
    ek_tol_ev: float,
) -> dict[int, str]:
    """Best-effort attach of DB ``upp_level_id`` to each observation.

    Returns ``{obs_index: upp_level_id}`` for lines resolved within tolerance.
    Any failure (missing method, missing column, no match) simply omits that
    line, which then falls back to the fingerprint grouping.
    """
    resolved: dict[int, str] = {}
    if atomic_db is None:
        return resolved
    getconn = getattr(atomic_db, "_get_connection", None)
    if getconn is None:
        return resolved

    # Cache the DB line table per (element, stage) so we query once per species.
    species_cache: dict[tuple[str, int], list[tuple[float, float, str]]] = {}
    try:
        for i, obs in enumerate(observations):
            key = (obs.element, int(obs.ionization_stage))
            if key not in species_cache:
                rows: list[tuple[float, float, str]] = []
                try:
                    with getconn() as conn:
                        cur = conn.execute(
                            "SELECT wavelength_nm, ek_ev, upp_level_id FROM lines "
                            "WHERE element = ? AND sp_num = ? "
                            "AND upp_level_id IS NOT NULL AND upp_level_id != ''",
                            (obs.element, int(obs.ionization_stage)),
                        )
                        for wl, ek, uid in cur.fetchall():
                            if wl is None or ek is None or uid is None:
                                continue
                            rows.append((float(wl), float(ek), str(uid)))
                except Exception:  # noqa: BLE001 - best-effort; fingerprint fallback
                    rows = []
                species_cache[key] = rows
            rows = species_cache[key]
            if not rows:
                continue
            # Nearest line by wavelength that also matches E_k within tolerance.
            best_uid: Optional[str] = None
            best_dwl = wl_tol_nm
            for wl, ek, uid in rows:
                if abs(ek - float(obs.E_k_ev)) > ek_tol_ev:
                    continue
                dwl = abs(wl - float(obs.wavelength_nm))
                if dwl <= best_dwl:
                    best_dwl = dwl
                    best_uid = uid
            if best_uid is not None:
                resolved[i] = best_uid
    except Exception:  # noqa: BLE001 - never let self-cal grouping crash the solve
        return resolved
    return resolved


def find_shared_upper_groups(
    observations: Sequence[LineObservation],
    atomic_db=None,
    *,
    min_group_size: int = 2,
    ek_tol_ev: float = 1e-3,
    wl_tol_nm: float = 0.05,
) -> list[SharedUpperGroup]:
    """Group observed lines by shared upper level (element, stage, level).

    Parameters
    ----------
    observations
        Line observations to group.
    atomic_db
        Optional :class:`~cflibs.atomic.database.AtomicDatabase`. When supplied,
        each line is matched to the DB ``upp_level_id`` (the complete DB
        populates this column for ~87% of lines and 95-99% of Ti/Al/V/Fe/Cr
        lines). Lines that cannot be resolved fall back to the ``(E_k, g_k)``
        fingerprint.
    min_group_size
        Groups with fewer members than this are dropped (a group of one carries
        no relative information).
    ek_tol_ev
        Tolerance (eV) used both for the fingerprint rounding and for the DB
        ``E_k`` match.
    wl_tol_nm
        Wavelength tolerance for the DB ``upp_level_id`` match.

    Returns
    -------
    list[SharedUpperGroup]
        Sorted by (element, stage, level_key) for determinism.
    """
    ek_decimals = max(0, int(round(-np.log10(ek_tol_ev)))) if ek_tol_ev > 0 else 3
    id_map = _resolve_upper_level_ids(observations, atomic_db, wl_tol_nm, ek_tol_ev)

    buckets: dict[tuple[str, int, str], list[int]] = {}
    resolved_flag: dict[tuple[str, int, str], bool] = {}
    for i, obs in enumerate(observations):
        # Only usable Boltzmann points can form a group.
        if not (np.isfinite(obs.intensity) and obs.intensity > 0):
            continue
        if not (np.isfinite(obs.A_ki) and obs.A_ki > 0):
            continue
        if i in id_map:
            level_key = f"id:{id_map[i]}"
            by_id = True
        else:
            level_key = f"fp:{_fingerprint_key(obs, ek_decimals)}"
            by_id = False
        gkey = (obs.element, int(obs.ionization_stage), level_key)
        buckets.setdefault(gkey, []).append(i)
        resolved_flag[gkey] = by_id

    groups: list[SharedUpperGroup] = []
    for (element, stage, level_key), idxs in buckets.items():
        if len(idxs) < min_group_size:
            continue
        first = observations[idxs[0]]
        groups.append(
            SharedUpperGroup(
                element=element,
                ionization_stage=stage,
                level_key=level_key,
                indices=sorted(idxs),
                E_k_ev=float(first.E_k_ev),
                g_k=float(first.g_k),
                resolved_by_id=resolved_flag[(element, stage, level_key)],
            )
        )
    groups.sort(key=lambda g: (g.element, g.ionization_stage, g.level_key))
    return groups


# ---------------------------------------------------------------------------
# Residual measurement
# ---------------------------------------------------------------------------
def measure_relative_ga_residuals(
    observations: Sequence[LineObservation],
    groups: Sequence[SharedUpperGroup],
    *,
    exclude_mask: Optional[Sequence[bool]] = None,
    sigma_floor: float = 1e-3,
) -> list[RelativeGAResidual]:
    """Measure the relative-g·A residual of every line in each group.

    For each group the anchor is the inverse-variance weighted mean of the
    Boltzmann ordinate ``y = ln(I lambda / (g A))``. The residual
    ``r_i = y_i - <y>`` is a direct estimate of the line's relative A_ki error
    (up to the group's unmeasurable absolute scale). Its noise is propagated
    analytically from the intensity uncertainties.

    Parameters
    ----------
    observations
        Line observations (indexed by ``group.indices``).
    groups
        Output of :func:`find_shared_upper_groups`.
    exclude_mask
        Optional boolean per-observation mask; ``True`` marks a line as
        optically thick / differential-SA suspect. Such lines are excluded from
        the anchor and flagged ``excluded`` (no correction applied).
    sigma_floor
        Minimum per-line log-domain sigma, to keep weights finite when the
        intensity uncertainty is zero/unknown.

    Returns
    -------
    list[RelativeGAResidual]
    """
    residuals: list[RelativeGAResidual] = []
    excl = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)

    for grp in groups:
        idxs = grp.indices
        y = np.array([observations[i].y_value for i in idxs], dtype=float)
        sig = np.array(
            [max(float(observations[i].y_uncertainty), sigma_floor) for i in idxs],
            dtype=float,
        )
        is_excl = np.array([bool(excl[i]) if excl is not None else False for i in idxs], dtype=bool)
        finite = np.isfinite(y) & np.isfinite(sig) & (sig > 0)
        anchor = finite & (~is_excl)
        if int(np.sum(anchor)) < 1:
            # No clean anchor lines; cannot define a group reference.
            continue
        w = np.zeros_like(sig)
        w[anchor] = 1.0 / sig[anchor] ** 2
        sum_w = float(np.sum(w))
        if sum_w <= 0:
            continue
        mu = float(np.sum(w * y) / sum_w)

        for local, i in enumerate(idxs):
            if not finite[local]:
                continue
            r = float(y[local] - mu)
            # Var(r_i) = sigma_i^2 - 1/sum_w for an anchor line (exact for IVW);
            # for an excluded line (not in the mean) Var(r_i) = sigma_i^2 + 1/sum_w.
            if anchor[local]:
                var_r = sig[local] ** 2 - 1.0 / sum_w
            else:
                var_r = sig[local] ** 2 + 1.0 / sum_w
            sigma_r = float(np.sqrt(max(var_r, sigma_floor**2)))
            residuals.append(
                RelativeGAResidual(
                    index=i,
                    element=grp.element,
                    ionization_stage=grp.ionization_stage,
                    level_key=grp.level_key,
                    residual=r,
                    sigma=sigma_r,
                    group_size=int(np.sum(anchor)),
                    excluded=bool(is_excl[local]),
                )
            )
    return residuals


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------
def _estimate_tau2(residuals: Sequence[RelativeGAResidual]) -> float:
    """Method-of-moments population variance of the log-g·A error.

    ``E[r_i^2] = tau^2 + sigma_i^2`` -> ``tau2 = max(0, mean(r_i^2 - sigma_i^2))``
    over the correctable (non-excluded) lines.
    """
    r = np.array([x.residual for x in residuals if not x.excluded], dtype=float)
    s = np.array([x.sigma for x in residuals if not x.excluded], dtype=float)
    if r.size == 0:
        return 0.0
    return float(max(0.0, np.mean(r**2 - s**2)))


def apply_relative_ga_correction(
    observations: Sequence[LineObservation],
    residuals: Sequence[RelativeGAResidual],
    *,
    min_group_size: int = 2,
    shrinkage: "str | float | None" = "empirical_bayes",
    max_log_correction: float = np.log(5.0),
) -> tuple[list[LineObservation], RelativeGACalibration]:
    """Refine per-line ``A_ki`` from measured relative-g·A residuals.

    The multiplicative correction is ``A_i^corr = A_i * exp(s_i * r_i)`` where
    ``s_i`` is the shrinkage factor. With ``shrinkage="empirical_bayes"`` the
    factor is the James-Stein posterior weight ``tau2 / (tau2 + sigma_i^2)`` so
    that residuals dominated by intensity noise are shrunk toward zero (no
    correction) while genuine A_ki errors are largely retained. A fixed float in
    ``[0, 1]`` applies a global shrinkage; ``None`` / ``0`` disables correction.

    For every corrected line the (previously fabricated rel_int->grade) A_ki
    uncertainty is replaced by the measured per-line residual sigma.

    Returns
    -------
    (corrected_observations, calibration)
        ``corrected_observations`` is a new list; unmodified lines are the same
        objects. ``calibration`` records residuals, correction factors, and the
        estimated shrinkage hyper-parameter.
    """
    obs_out = list(observations)
    calib = RelativeGACalibration(residuals=list(residuals))

    usable = [r for r in residuals if not r.excluded and r.group_size >= min_group_size]
    if not usable:
        return obs_out, calib

    if shrinkage == "empirical_bayes":
        tau2 = _estimate_tau2(usable)
        calib.tau2 = tau2

        def shrink_factor(sig: float) -> float:
            denom = tau2 + sig**2
            return 0.0 if denom <= 0 else tau2 / denom

    elif shrinkage is None:
        return obs_out, calib
    else:
        s_const = float(shrinkage)
        if not (0.0 <= s_const <= 1.0):
            raise ValueError("shrinkage float must be in [0, 1]")
        if s_const == 0.0:
            return obs_out, calib

        def shrink_factor(sig: float) -> float:  # noqa: ARG001 - constant shrinkage
            return s_const

    n_corrected = 0
    for r in usable:
        s_i = shrink_factor(r.sigma)
        log_corr = float(np.clip(s_i * r.residual, -max_log_correction, max_log_correction))
        if log_corr == 0.0:
            continue
        factor = float(np.exp(log_corr))
        i = r.index
        old = obs_out[i]
        # Measured relative-A_ki uncertainty replaces the fabricated one.
        new_aki_unc = float(r.sigma)
        obs_out[i] = replace(
            old,
            A_ki=float(old.A_ki) * factor,
            aki_uncertainty=new_aki_unc,
        )
        calib.corrections[i] = factor
        calib.correction_sigma[i] = new_aki_unc
        n_corrected += 1

    calib.n_groups = len({r.level_key for r in usable})
    calib.n_lines_corrected = n_corrected
    return obs_out, calib


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def self_calibrate_relative_ga(
    observations: Sequence[LineObservation],
    atomic_db=None,
    *,
    exclude_mask: Optional[Sequence[bool]] = None,
    min_group_size: int = 2,
    shrinkage: "str | float | None" = "empirical_bayes",
    ek_tol_ev: float = 1e-3,
) -> tuple[list[LineObservation], RelativeGACalibration]:
    """End-to-end relative-g·A self-calibration of a line list.

    Convenience wrapper: :func:`find_shared_upper_groups` ->
    :func:`measure_relative_ga_residuals` -> :func:`apply_relative_ga_correction`.

    Applied ONCE, BEFORE the Boltzmann / Saha-Boltzmann-graph fit, as a pure
    observation transform (like the observable self-absorption corrector). No
    recovered composition is read.
    """
    groups = find_shared_upper_groups(
        observations, atomic_db, min_group_size=min_group_size, ek_tol_ev=ek_tol_ev
    )
    if not groups:
        return list(observations), RelativeGACalibration()
    residuals = measure_relative_ga_residuals(observations, groups, exclude_mask=exclude_mask)
    corrected, calib = apply_relative_ga_correction(
        observations, residuals, min_group_size=min_group_size, shrinkage=shrinkage
    )
    if calib.n_lines_corrected:
        logger.info(
            "relative-gA self-cal: %d group(s), corrected %d line(s), tau=%.3f",
            calib.n_groups,
            calib.n_lines_corrected,
            float(np.sqrt(calib.tau2)),
        )
    return corrected, calib
