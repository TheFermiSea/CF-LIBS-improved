#!/usr/bin/env python3
"""Regenerate the physically-impossible partition-function polynomial rows.

Background (composition-pipeline diagnosis 2026-06-03, § 2.1, blocker PF-1/PF-2)
-------------------------------------------------------------------------------
Direct summation U(T) = Σ gᵢ exp(-Eᵢ / kT) over the ``energy_levels`` table is a
*strict lower bound* on the true partition function (you can only add more
levels, never fewer).  Therefore a stored polynomial that evaluates **below**
the DB's own direct-sum at any temperature is physically impossible — it is a
bad fit.

The 2026-06-03 audit found **28 of 146** polynomial rows with
``min(U_poly / U_directsum) < 0.80`` somewhere in 6000–12000 K (the ps-LIBS
band).  The defect is strongly source-correlated:

* ``Irwin1981``        — 11 / 12 rows bad (fit to truncated, observed-only Q(T) tables)
* ``NIST_ASD_fit``     — 17 / 99 rows bad (stale reference coefficients)
* ``direct_sum_fit_v1`` — 0 / 35 rows bad (fit directly to the current EL table)

The fix is **data regeneration**, not a code/math change: re-fit the flagged
rows with the *same recipe* that produced the clean ``direct_sum_fit_v1`` rows.

What this script does
---------------------
1. Identify the flagged rows (``min(U_poly / U_directsum) < THRESHOLD`` over the
   6000–12000 K band, where ``U_directsum`` is computed from ``energy_levels``).
2. Re-fit each flagged row:
   * direct-sum U(T) on a 2000–25000 K grid (KEY_TEMPS over-weighted);
   * fit ``ln U = Σ aₙ (ln T)ⁿ`` (natural-log basis, degree 4);
   * **monotone-constrained**: if the resulting polynomial turns over inside
     the default ``[2000, 25000] K`` window, tighten ``t_min`` / ``t_max`` past
     the (cold/hot) turnover so the evaluator's clamp guard holds U flat in the
     physically-flat tails (the LIBS band 5800–15000 K is never clipped);
   * apply a small positive ln-offset so ``U_poly ≥ U_directsum`` across
     6000–12000 K (removes the physically-impossible undershoot);
   * source-tag ``direct_sum_fit_v1``.
3. Verify each regenerated row: ``poly / directsum ≥ 1.0`` across 6000–12000 K
   **and** within ~5–10 % of direct-sum.
4. Species whose ``energy_levels`` are too incomplete to fit (e.g. Cr II /
   hot-edge Ti I per the diagnosis) are **left as-is and LOGGED** — we never
   fabricate levels.
5. Print a before/after table and ``INSERT OR REPLACE`` the regenerated rows
   (unless ``--dry-run``).

Usage
-----
    python scripts/archive/migrations/regenerate_partition_functions.py --db-path ASD_da/libs_production.db
    python scripts/archive/migrations/regenerate_partition_functions.py --db-path ASD_da/libs_production.db --dry-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KB_EV = 8.617333262145e-5  # Boltzmann constant in eV/K

# Fit grid (matches the ``direct_sum_fit_v1`` recipe in
# scripts/archive/migrations/populate_partition_functions.py, extended to start at 2000 K).
T_FIT_MIN = 2000.0
T_FIT_MAX = 25000.0
N_FIT_POINTS = 60
KEY_TEMPS = (5000.0, 10000.0, 15000.0, 20000.0)

# DB-stored validity window defaults (may be tightened per-species below).
T_MIN_DB = 2000.0
T_MAX_DB = 25000.0

# LIBS validation band (ps-LIBS regime 5800–15000 K; audit checks 6000–12000 K).
BAND_LO = 6000.0
BAND_HI = 12000.0
N_BAND = 25

# A row is flagged for regeneration when poly drops below this fraction of
# direct-sum anywhere in the LIBS band.  Direct-sum is a strict lower bound on
# the true U, so ANY row below it is physically impossible; the audit headline
# (28 rows at < 0.80) used the gross-defect threshold, but we regenerate every
# row that meaningfully undershoots the floor (< 0.95) so the STORED DB row and
# the load-time direct-sum fallback agree — this also catches the borderline
# Al I (Irwin1981, min 0.851) the gross threshold left below the floor.
FLAG_THRESHOLD = 0.95

# Acceptance tolerance for the regenerated fit (poly within this of direct-sum).
ACCEPT_REL_TOL = 0.10

SOURCE_TAG = "direct_sum_fit_v1"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def fetch_levels(cur: sqlite3.Cursor, element: str, sp_num: int):
    rows = cur.execute(
        "SELECT g_level, energy_ev FROM energy_levels "
        "WHERE element=? AND sp_num=? ORDER BY energy_ev",
        (element, sp_num),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([])
    g = np.array([r[0] for r in rows], dtype=np.float64)
    e = np.array([r[1] for r in rows], dtype=np.float64)
    return g, e


def fetch_ip(cur: sqlite3.Cursor, element: str, sp_num: int):
    row = cur.execute(
        "SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?",
        (element, sp_num),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def direct_sum_U(T_K: float, g: np.ndarray, e_ev: np.ndarray, ip_ev: float) -> float:
    """U(T) = Σ gᵢ exp(-Eᵢ / kT), masked to Eᵢ < IP. Floored at 1.0."""
    mask = e_ev < ip_ev
    if not mask.any():
        return 1.0
    U = float(np.sum(g[mask] * np.exp(-e_ev[mask] / (KB_EV * T_K))))
    return max(U, 1.0)


def poly_U(coeffs, T_K: float) -> float:
    """U(T) = exp(Σ aₙ (ln T)ⁿ) — natural-log basis (no clamp/floor)."""
    ln_T = np.log(T_K)
    return float(np.exp(sum(a * ln_T**i for i, a in enumerate(coeffs))))


def fit_lnT_polynomial(T: np.ndarray, Q: np.ndarray, degree: int = 4) -> list[float]:
    """Fit ``ln Q = Σ aₙ (ln T)ⁿ`` (10× weight on KEY_TEMPS), pad to length 5."""
    lnT = np.log(T)
    lnQ = np.log(np.maximum(Q, 1e-12))
    w = np.ones_like(T)
    for kt in KEY_TEMPS:
        w[np.isclose(T, kt)] = 10.0
    coeffs = np.polynomial.polynomial.polyfit(lnT, lnQ, degree, w=w)
    coeffs = list(coeffs) + [0.0] * (5 - len(coeffs))
    return coeffs[:5]


def is_monotone(coeffs, t_min: float, t_max: float, n: int = 400) -> bool:
    """True if U(T) is non-decreasing across [t_min, t_max]."""
    Tm = np.linspace(t_min, t_max, n)
    Um = np.array([poly_U(coeffs, T) for T in Tm])
    return bool(np.all(np.diff(Um) >= -1e-9))


def tighten_window(coeffs, t_min: float, t_max: float) -> tuple[float, float]:
    """Tighten [t_min, t_max] past the cold/hot turnover so U is monotone within.

    The polynomial is clamped to this window at evaluation time
    (see :func:`cflibs.plasma.partition.polynomial_partition_function`), so any
    non-monotone tail outside the returned window is held flat — physically
    correct for species whose U is flat at the cold/hot edge.  The ps-LIBS band
    (5800–15000 K) is never inside a turnover for the flagged species.
    """
    Tm = np.linspace(t_min, t_max, 600)
    Um = np.array([poly_U(coeffs, T) for T in Tm])
    dec = np.where(np.diff(Um) < 0)[0]
    if len(dec) == 0:
        return t_min, t_max
    mid = len(Tm) // 2
    cold = [i for i in dec if i < mid]
    hot = [i for i in dec if i >= mid]
    lo = Tm[max(cold) + 1] if cold else t_min
    hi = Tm[min(hot)] if hot else t_max
    return float(lo), float(hi)


def refit_species(g: np.ndarray, e: np.ndarray, ip_ev: float):
    """Refit one species via the monotone direct-sum recipe.

    Returns ``(coeffs, t_min, t_max, min_ratio, max_ratio)`` or ``None`` if the
    energy_levels are too sparse to support a fit (caller LOGS and skips).
    """
    if len(g) < 2:
        return None

    T_grid = np.unique(
        np.concatenate([np.linspace(T_FIT_MIN, T_FIT_MAX, N_FIT_POINTS), np.array(KEY_TEMPS)])
    )
    Q_grid = np.array([direct_sum_U(T, g, e, ip_ev) for T in T_grid])

    # Degenerate (flat, U≈g0) species: a polynomial fit is unstable; the
    # cold-flat tail dominates.  Still fit degree 4, then enforce monotonicity
    # via the window tightening below.
    coeffs = fit_lnT_polynomial(T_grid, Q_grid, degree=4)

    t_min, t_max = T_MIN_DB, T_MAX_DB
    if not is_monotone(coeffs, t_min, t_max):
        t_min, t_max = tighten_window(coeffs, t_min, t_max)
        # Guard: never tighten so far that the LIBS band is clipped.
        t_min = min(t_min, BAND_LO)
        t_max = max(t_max, BAND_HI)

    # Lift the polynomial so U_poly ≥ U_directsum across the LIBS band
    # (removes the physically-impossible undershoot).  A pure ln-offset on a0
    # preserves the fit shape and monotonicity.
    T_band = np.linspace(BAND_LO, BAND_HI, N_BAND)
    ratios = np.array([poly_U(coeffs, T) / direct_sum_U(T, g, e, ip_ev) for T in T_band])
    deficit = max(0.0, float(np.log(1.0 / min(ratios))))
    coeffs[0] += deficit

    ratios2 = np.array([poly_U(coeffs, T) / direct_sum_U(T, g, e, ip_ev) for T in T_band])
    return coeffs, t_min, t_max, float(ratios2.min()), float(ratios2.max())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def find_flagged_rows(cur: sqlite3.Cursor):
    """Return list of (element, sp_num, source, min_ratio, n_levels) flagged rows."""
    rows = cur.execute(
        "SELECT element, sp_num, a0, a1, a2, a3, a4, source FROM partition_functions "
        "ORDER BY element, sp_num"
    ).fetchall()
    T_band = np.linspace(BAND_LO, BAND_HI, 13)
    flagged = []
    for element, sp_num, a0, a1, a2, a3, a4, source in rows:
        g, e = fetch_levels(cur, element, sp_num)
        if len(g) == 0:
            continue  # no levels → polynomial is the only option, can't verify
        ip_ev = fetch_ip(cur, element, sp_num)
        if ip_ev is None:
            ip_ev = float(np.max(e)) + 5.0
        coeffs = [a0, a1, a2, a3, a4]
        min_ratio = min(poly_U(coeffs, T) / direct_sum_U(T, g, e, ip_ev) for T in T_band)
        if min_ratio < FLAG_THRESHOLD:
            flagged.append((element, sp_num, source, float(min_ratio), len(g)))
    return flagged


def regenerate(db_path: Path, dry_run: bool = False) -> int:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    flagged = find_flagged_rows(cur)
    print("=" * 92)
    print("PARTITION-FUNCTION REGENERATION")
    print("=" * 92)
    print(
        f"Flagged rows (min(U_poly/U_directsum) < {FLAG_THRESHOLD:.2f} in "
        f"{BAND_LO:.0f}-{BAND_HI:.0f} K): {len(flagged)}"
    )
    from collections import Counter

    print("By source:", dict(Counter(f[2] for f in flagged)))
    print()

    header = (
        f"{'Species':<8s} {'old src':<16s} {'before':>8s} -> "
        f"{'after_min':>9s} {'after_max':>9s} {'window':>16s} {'status':>10s}"
    )
    print(header)
    print("-" * len(header))

    rows_to_insert: list[tuple] = []
    skipped: list[tuple] = []
    n_fixed = 0

    for element, sp_num, source, before_min, n_levels in flagged:
        g, e = fetch_levels(cur, element, sp_num)
        ip_ev = fetch_ip(cur, element, sp_num)
        if ip_ev is None:
            ip_ev = float(np.max(e)) + 5.0

        result = refit_species(g, e, ip_ev)
        sp_label = f"{element} {sp_num}"
        if result is None:
            skipped.append((element, sp_num, source, n_levels, "too few levels"))
            print(
                f"{sp_label:<8s} {source:<16s} {before_min:8.3f} -> "
                f"{'--':>9s} {'--':>9s} {'--':>16s} {'SKIP':>10s}"
            )
            continue

        coeffs, t_min, t_max, after_min, after_max = result
        # Acceptance: poly ≥ directsum across the band AND within tolerance.
        ok = (after_min >= 1.0 - 1e-6) and (after_max <= 1.0 + ACCEPT_REL_TOL)
        status = "FIXED" if ok else "WEAK*"
        win = f"[{t_min:.0f},{t_max:.0f}]"
        print(
            f"{sp_label:<8s} {source:<16s} {before_min:8.3f} -> "
            f"{after_min:9.4f} {after_max:9.4f} {win:>16s} {status:>10s}"
        )
        if not ok:
            # Incomplete level table (e.g. Cr II / hot-edge Ti I): the refit
            # cannot beat the underlying completeness.  LOG and leave as-is
            # rather than ship a fit that still violates the band tolerance.
            skipped.append(
                (element, sp_num, source, n_levels, f"refit out of tol (max {after_max:.3f})")
            )
            continue
        rows_to_insert.append((element, sp_num, *coeffs, float(t_min), float(t_max), SOURCE_TAG))
        n_fixed += 1

    print()
    if dry_run:
        print(f"[DRY-RUN] Would INSERT OR REPLACE {len(rows_to_insert)} regenerated rows.")
    else:
        cur.executemany(
            "INSERT OR REPLACE INTO partition_functions "
            "(element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows_to_insert,
        )
        conn.commit()
        print(f"Wrote {len(rows_to_insert)} regenerated rows (source='{SOURCE_TAG}').")

    if skipped:
        print()
        print("SKIPPED (incomplete energy_levels — left as-is, NOT fabricated):")
        for element, sp_num, source, n_levels, reason in skipped:
            print(f"  {element} {sp_num:<2d} src={source:<16s} n_levels={n_levels:<4d} ({reason})")

    print()
    print(f"Summary: {n_fixed} fixed, {len(skipped)} skipped, of {len(flagged)} flagged.")
    conn.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db-path", default="ASD_da/libs_production.db")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 1
    print(f"Regenerating partition_functions in {db_path}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}\n")
    return regenerate(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
