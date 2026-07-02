"""Absolute A_ki anchoring via radiative lifetimes / branching fractions.

Implements Fix #1b of the physics-first-principles audit (Issue 1, cluster A):
*atomic-data g*A error is a systematic, source-correlated **scale bias**, not
random variance.* The shipped pipeline models the D/E-grade A_ki error as an
independent Gaussian and down-weights poorly graded lines (WLS). You cannot
average your way out of a scale error, so this module provides the audit's
remedy plus the diagnostics that quantify it:

1. :func:`species_scale_diagnostic` -- per-species/stage exposure to the
   systematic bias, computed from the DB's own NIST accuracy grades. The
   grade-weighted RMS fractional A_ki error is, to first order, exactly the
   within-species Boltzmann-plot RMS residual *floor* attributable to A_ki
   (a multiplicative A error ``delta`` shifts ``y = ln(I*lambda/gA)`` by
   ``-ln(delta) ~ -(delta-1)``). This is the falsifiable "scale-spread table."

2. :func:`branching_fractions` / :func:`anchor_aki_to_lifetimes` -- the physics
   change: renormalize each upper level's A_ki set to a measured radiative
   lifetime, ``A_ki = BR_ki / tau_k`` (Lawler/Den Hartog Wisconsin-group
   practice). Relative branching fractions and laser-measured lifetimes are
   accurate to a few percent, so the per-species intercept then carries a
   ~few-% scale error instead of a D/E-grade (50-100%) one. Corrected values
   are written to NEW columns (``aki_anchored``, ``aki_anchor_source``); the
   original ``aki`` is never overwritten.

3. :func:`compare_aki_sources` -- cross-DB per-species scale comparison
   (``ln(A_a / A_b)`` over shared transitions). This is the "why did Kurucz
   beat NIST" measurement; it becomes usable the moment an independent Kurucz
   or VALD DB is ingested (``scripts/ingest_kurucz_atomic.py``). NOTE: the
   production DB's ``log_gf`` column is NIST's *own* (same ASD dump as ``aki``),
   so comparing ``aki`` against ``log_gf`` in a single DB is circular -- an
   independent second source is required.

Physics-only: numpy + sqlite3 only (no ML libraries).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from cflibs.atomic.reference_data import NIST_GRADE_UNCERTAINTY
from cflibs.core.constants import C_LIGHT, E_CHARGE, M_E

# Vacuum permittivity (SI); not in core.constants, defined here for the gf<->A map.
_EPSILON_0 = 8.8541878128e-12  # F/m

# g_k * A_ki = GF_TO_GKA * gf / lambda[Angstrom]^2   (classical oscillator relation)
# = (2*pi*e^2)/(m_e*c*eps0) * 1e20 (m^-2 -> Angstrom^-2). Evaluates to ~6.6702e15.
GF_TO_GKA_ANGSTROM = (2.0 * math.pi * E_CHARGE**2) / (M_E * C_LIGHT * _EPSILON_0) * 1e20

# Target quantitation species for CF-LIBS (steel / DED / geology minors).
DEFAULT_TARGET_SPECIES: tuple[tuple[str, int], ...] = tuple(
    (el, sp)
    for el in ("Ti", "Al", "V", "Fe", "Cr", "Mn", "Ni", "Cu", "Si", "Mo")
    for sp in (1, 2)
)


def a_from_log_gf(log_gf: float, gk: float, wavelength_nm: float) -> float:
    """Transition probability ``A_ki`` (s^-1) from ``log(gf)``, ``g_k`` and lambda.

    Uses ``g_k A_ki = 6.6702e15 * gf / lambda[Angstrom]^2``. Provided so an
    independent Kurucz/VALD ``log_gf`` set can be converted to A_ki on the same
    footing as NIST for :func:`compare_aki_sources`.
    """
    if gk <= 0 or wavelength_nm <= 0:
        raise ValueError("gk and wavelength_nm must be positive")
    lambda_ang = wavelength_nm * 10.0
    gf = 10.0**log_gf
    return GF_TO_GKA_ANGSTROM * gf / (gk * lambda_ang**2)


@dataclass
class SpeciesScaleReport:
    """Per-species exposure to the systematic A_ki scale bias (Issue 1)."""

    element: str
    sp_num: int
    n_lines: int
    n_with_aki: int
    grade_counts: dict[str, int]
    frac_de_grade: float
    """Fraction of A_ki-bearing lines that are D/E grade (the 50-100% biased set)."""
    sigma_scale_rms: float
    """Grade-weighted RMS fractional A_ki error == predicted within-species
    Boltzmann-plot RMS residual floor attributable to A_ki bias."""
    sigma_scale_median: float
    n_shared_upper_levels: int
    """Upper levels with >=2 observed lines (candidates for branching-fraction /
    lifetime anchoring and in-plasma relative-gA self-calibration)."""
    n_lines_in_shared_levels: int

    def as_dict(self) -> dict:
        return {
            "species": f"{self.element} {self.sp_num}",
            "element": self.element,
            "sp_num": self.sp_num,
            "n_lines": self.n_lines,
            "n_with_aki": self.n_with_aki,
            "grade_counts": self.grade_counts,
            "frac_de_grade": round(self.frac_de_grade, 4),
            "sigma_scale_rms": round(self.sigma_scale_rms, 4),
            "sigma_scale_median": round(self.sigma_scale_median, 4),
            "n_shared_upper_levels": self.n_shared_upper_levels,
            "n_lines_in_shared_levels": self.n_lines_in_shared_levels,
        }


def _grade_sigma(grade: str | None) -> float:
    """Fractional A_ki uncertainty for a NIST accuracy grade.

    Missing/unknown grade -> worst case (1.0), matching the shipped fallback.
    """
    if grade is None:
        return NIST_GRADE_UNCERTAINTY["E"]
    return NIST_GRADE_UNCERTAINTY.get(grade.strip(), NIST_GRADE_UNCERTAINTY["E"])


def species_scale_diagnostic(
    db_path: str,
    species: tuple[tuple[str, int], ...] = DEFAULT_TARGET_SPECIES,
) -> list[SpeciesScaleReport]:
    """Per-species systematic-bias-exposure table from the atomic DB.

    Parameters
    ----------
    db_path:
        Path to the atomic SQLite DB (the shipped ``lines`` schema).
    species:
        Iterable of ``(element, sp_num)`` pairs. Defaults to the CF-LIBS
        quantitation targets (Ti/Al/V/Fe/Cr/Mn/Ni/Cu/Si/Mo, stages I/II).

    Returns
    -------
    list[SpeciesScaleReport]
        One report per species that has any lines, sorted by descending
        ``sigma_scale_rms`` (worst systematic-bias exposure first).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    reports: list[SpeciesScaleReport] = []
    try:
        for element, sp_num in species:
            rows = conn.execute(
                "SELECT aki, accuracy_grade, upp_level_id "
                "FROM lines WHERE element = ? AND sp_num = ?",
                (element, sp_num),
            ).fetchall()
            if not rows:
                continue
            grade_counts: dict[str, int] = {}
            sigmas: list[float] = []
            level_line_counts: dict[str, int] = {}
            n_with_aki = 0
            n_de = 0
            for r in rows:
                aki = r["aki"]
                if aki is None or aki <= 0:
                    continue
                n_with_aki += 1
                grade = (r["accuracy_grade"] or "").strip() or "(none)"
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
                sig = _grade_sigma(r["accuracy_grade"])
                sigmas.append(sig)
                if sig >= NIST_GRADE_UNCERTAINTY["D"]:  # D, D+, E
                    n_de += 1
                uid = r["upp_level_id"]
                if uid:
                    level_line_counts[uid] = level_line_counts.get(uid, 0) + 1
            if n_with_aki == 0:
                continue
            arr = np.asarray(sigmas, dtype=float)
            shared = {k: v for k, v in level_line_counts.items() if v >= 2}
            reports.append(
                SpeciesScaleReport(
                    element=element,
                    sp_num=sp_num,
                    n_lines=len(rows),
                    n_with_aki=n_with_aki,
                    grade_counts=grade_counts,
                    frac_de_grade=n_de / n_with_aki,
                    sigma_scale_rms=float(np.sqrt(np.mean(arr**2))),
                    sigma_scale_median=float(np.median(arr)),
                    n_shared_upper_levels=len(shared),
                    n_lines_in_shared_levels=int(sum(shared.values())),
                )
            )
    finally:
        conn.close()
    reports.sort(key=lambda x: x.sigma_scale_rms, reverse=True)
    return reports


@dataclass
class UpperLevelGroup:
    """Observed lines sharing an upper level, with DB-derived branching fractions."""

    upp_level_id: str
    line_ids: list[int]
    wavelengths_nm: list[float]
    aki: list[float]
    branching_fractions: list[float]
    """A_ki / sum(A_ki over the *observed* lines from this level). Sums to 1."""
    observed_sum_aki: float
    """sum(A_ki over observed lines) -- a LOWER bound on 1/tau_k (unobserved
    IR/UV branches are missing from the DB)."""


def branching_fractions(
    db_path: str, element: str, sp_num: int, min_lines: int = 2
) -> list[UpperLevelGroup]:
    """Group a species' observed lines by upper level and compute branching fractions.

    ``BR_ki = A_ki / sum_i A_ki`` over the lines *present in the DB* for that
    upper level. Because unobserved decay channels are missing, ``observed_sum_aki``
    is a lower bound on ``1/tau_k``; see :func:`anchor_aki_to_lifetimes` for how
    the residual-branch issue is handled honestly.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    groups: dict[str, list[sqlite3.Row]] = {}
    try:
        rows = conn.execute(
            "SELECT id, wavelength_nm, aki, upp_level_id FROM lines "
            "WHERE element = ? AND sp_num = ? AND aki > 0 "
            "AND upp_level_id IS NOT NULL AND upp_level_id != ''",
            (element, sp_num),
        ).fetchall()
    finally:
        conn.close()
    for r in rows:
        groups.setdefault(r["upp_level_id"], []).append(r)

    out: list[UpperLevelGroup] = []
    for uid, grp in groups.items():
        if len(grp) < min_lines:
            continue
        aki = np.asarray([r["aki"] for r in grp], dtype=float)
        s = float(aki.sum())
        if s <= 0:
            continue
        out.append(
            UpperLevelGroup(
                upp_level_id=uid,
                line_ids=[int(r["id"]) for r in grp],
                wavelengths_nm=[float(r["wavelength_nm"]) for r in grp],
                aki=aki.tolist(),
                branching_fractions=(aki / s).tolist(),
                observed_sum_aki=s,
            )
        )
    return out


@dataclass
class AnchorResult:
    """Outcome of a lifetime-anchoring pass over one species."""

    element: str
    sp_num: int
    n_levels_anchored: int
    n_lines_anchored: int
    per_line: list[dict] = field(default_factory=list)
    """One entry per anchored line: id, wavelength_nm, aki_original, aki_anchored,
    scale_ratio (anchored/original)."""
    wrote_to_db: bool = False

    @property
    def mean_ln_scale_shift(self) -> float:
        """Mean |ln(A_anchored / A_original)| -- how far anchoring moved the scale."""
        if not self.per_line:
            return 0.0
        return float(
            np.mean([abs(math.log(p["scale_ratio"])) for p in self.per_line if p["scale_ratio"] > 0])
        )


def _ensure_anchor_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(lines)")}
    if "aki_anchored" not in cols:
        conn.execute("ALTER TABLE lines ADD COLUMN aki_anchored REAL")
    if "aki_anchor_source" not in cols:
        conn.execute("ALTER TABLE lines ADD COLUMN aki_anchor_source TEXT")


def anchor_aki_to_lifetimes(
    db_path: str,
    element: str,
    sp_num: int,
    level_lifetimes: dict[str, float],
    *,
    observed_branch_fraction: dict[str, float] | None = None,
    anchor_source: str = "lifetime",
    write: bool = False,
) -> AnchorResult:
    """Renormalize a species' A_ki to measured radiative lifetimes: ``A_ki = BR_ki/tau_k``.

    For each upper level ``k`` in ``level_lifetimes`` we take the branching
    fractions among the *observed* lines from the DB (accurate to a few %,
    per Lawler/Den Hartog), and distribute the measured total decay rate
    ``1/tau_k`` across them::

        A_ki = BR_ki * f_obs_k / tau_k

    where ``BR_ki = A_ki / sum_i A_ki`` (observed lines) and ``f_obs_k`` is the
    fraction of the level's total decay captured by the observed lines
    (default 1.0). **Residual-branch honesty:** with ``f_obs_k = 1`` the full
    ``1/tau_k`` is attributed to the observed lines, so each anchored ``A_ki``
    is an *upper bound*; supply ``observed_branch_fraction`` (from measured
    branching-fraction tables) to correct for unobserved IR/UV channels.

    Parameters
    ----------
    level_lifetimes:
        ``{upp_level_id: tau_seconds}``. Only levels present here are anchored.
    observed_branch_fraction:
        Optional ``{upp_level_id: f_obs in (0, 1]}``.
    write:
        If True, persist to new columns ``aki_anchored`` / ``aki_anchor_source``
        (the original ``aki`` is never modified). If False (default), a dry run.

    Returns
    -------
    AnchorResult
    """
    mode = "rwc" if write else "ro"
    conn = sqlite3.connect(f"file:{db_path}?mode={mode}", uri=True)
    conn.row_factory = sqlite3.Row
    per_line: list[dict] = []
    n_levels = 0
    try:
        if write:
            _ensure_anchor_columns(conn)
        for uid, tau in level_lifetimes.items():
            if tau <= 0:
                continue
            rows = conn.execute(
                "SELECT id, wavelength_nm, aki FROM lines "
                "WHERE element = ? AND sp_num = ? AND upp_level_id = ? AND aki > 0",
                (element, sp_num, uid),
            ).fetchall()
            if not rows:
                continue
            aki = np.asarray([r["aki"] for r in rows], dtype=float)
            s = float(aki.sum())
            if s <= 0:
                continue
            f_obs = 1.0
            if observed_branch_fraction is not None:
                f_obs = float(observed_branch_fraction.get(uid, 1.0))
            br = aki / s
            a_anchored = br * f_obs / tau
            n_levels += 1
            for r, a_orig, a_new in zip(rows, aki.tolist(), a_anchored.tolist()):
                per_line.append(
                    {
                        "id": int(r["id"]),
                        "wavelength_nm": float(r["wavelength_nm"]),
                        "aki_original": float(a_orig),
                        "aki_anchored": float(a_new),
                        "scale_ratio": float(a_new / a_orig) if a_orig > 0 else float("nan"),
                    }
                )
                if write:
                    conn.execute(
                        "UPDATE lines SET aki_anchored = ?, aki_anchor_source = ? WHERE id = ?",
                        (float(a_new), anchor_source, int(r["id"])),
                    )
        if write:
            conn.commit()
    finally:
        conn.close()
    return AnchorResult(
        element=element,
        sp_num=sp_num,
        n_levels_anchored=n_levels,
        n_lines_anchored=len(per_line),
        per_line=per_line,
        wrote_to_db=write,
    )


@dataclass
class SourceComparison:
    """Per-species ``ln(A_a / A_b)`` scale comparison between two atomic DBs."""

    element: str
    sp_num: int
    n_shared: int
    ln_ratio_median: float
    ln_ratio_mad: float
    """Median absolute deviation of ln(A_a/A_b) -- the incoherent (per-line) spread."""
    ln_ratio_mean: float
    ln_ratio_std: float

    def as_dict(self) -> dict:
        return {
            "species": f"{self.element} {self.sp_num}",
            "n_shared": self.n_shared,
            "ln_ratio_median": round(self.ln_ratio_median, 4),
            "ln_ratio_mad": round(self.ln_ratio_mad, 4),
            "ln_ratio_mean": round(self.ln_ratio_mean, 4),
            "ln_ratio_std": round(self.ln_ratio_std, 4),
            "scale_factor_median": round(math.exp(self.ln_ratio_median), 4),
        }


def _load_aki_by_transition(db_path: str, element: str, sp_num: int) -> dict[tuple, float]:
    """Map (low_level_id, upp_level_id) -> A_ki for a species (transition-keyed)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[tuple, float] = {}
    try:
        rows = conn.execute(
            "SELECT low_level_id, upp_level_id, wavelength_nm, aki FROM lines "
            "WHERE element = ? AND sp_num = ? AND aki > 0",
            (element, sp_num),
        ).fetchall()
    finally:
        conn.close()
    for r in rows:
        low, upp = r["low_level_id"], r["upp_level_id"]
        if low and upp:
            out[(low, upp)] = float(r["aki"])
    return out


def compare_aki_sources(
    db_a: str,
    db_b: str,
    species: tuple[tuple[str, int], ...] = DEFAULT_TARGET_SPECIES,
) -> list[SourceComparison]:
    """Per-species ``ln(A_a / A_b)`` over transitions shared by two atomic DBs.

    Cross-matches on the ``(low_level_id, upp_level_id)`` transition key (stable
    across sources that carry NIST level ids). This is the audit's "why did
    Kurucz beat NIST" measurement: a single-source set has one coherent
    (closure-absorbable) scale error, whereas a mix of per-source offsets does
    not. ``db_a`` = reference (e.g. NIST production DB), ``db_b`` = comparison
    (e.g. a Kurucz DB from ``scripts/ingest_kurucz_atomic.py``).

    NOTE: passing the same DB twice, or a DB's ``log_gf``-derived A vs its own
    ``aki``, yields ~0 by construction (circular); an *independent* second
    source is required for a meaningful number.
    """
    out: list[SourceComparison] = []
    for element, sp_num in species:
        a = _load_aki_by_transition(db_a, element, sp_num)
        b = _load_aki_by_transition(db_b, element, sp_num)
        shared = set(a) & set(b)
        ln_ratios = [
            math.log(a[k] / b[k]) for k in shared if a[k] > 0 and b[k] > 0
        ]
        if not ln_ratios:
            continue
        arr = np.asarray(ln_ratios, dtype=float)
        med = float(np.median(arr))
        out.append(
            SourceComparison(
                element=element,
                sp_num=sp_num,
                n_shared=len(ln_ratios),
                ln_ratio_median=med,
                ln_ratio_mad=float(np.median(np.abs(arr - med))),
                ln_ratio_mean=float(np.mean(arr)),
                ln_ratio_std=float(np.std(arr)),
            )
        )
    return out
