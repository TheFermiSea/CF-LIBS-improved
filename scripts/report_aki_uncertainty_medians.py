#!/usr/bin/env python3
"""Report median sigma(Aki) per ion for key steel/silicate ions.

Queries the generated SQLite atomic database and prints the median fractional
aki_uncertainty for Fe I, Fe II, Cr II, and Mn II — the four ions that
dominate steel and silicate CF-LIBS fits.  Optionally writes a JSON summary
so the before/after delta can be tracked across DB refreshes.

Usage::

    python scripts/report_aki_uncertainty_medians.py \\
        --db ASD_da/libs_production.db \\
        --output output/validation/aki_uncertainty_medians.json

The JSON output can be committed or diff-ed to document the improvement from
ingesting Kramida 2024 numerical uncertainties (see bead CF-LIBS-improved-f25n).
"""

import argparse
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

# Ions of interest as (element, sp_num, label) tuples.
# sp_num follows NIST convention: 1 = neutral (I), 2 = singly ionised (II).
IONS_OF_INTEREST = [
    ("Fe", 1, "Fe I"),
    ("Fe", 2, "Fe II"),
    ("Cr", 2, "Cr II"),
    ("Mn", 2, "Mn II"),
]


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def compute_medians(db_path: str) -> dict:
    """Return per-ion statistics on aki_uncertainty from *db_path*."""
    conn = sqlite3.connect(db_path)
    try:
        results: dict[str, dict] = {}
        for element, sp_num, label in IONS_OF_INTEREST:
            rows = conn.execute(
                "SELECT aki_uncertainty, accuracy_grade "
                "FROM lines "
                "WHERE element = ? AND sp_num = ? "
                "  AND aki IS NOT NULL AND aki_uncertainty IS NOT NULL",
                (element, sp_num),
            ).fetchall()

            uncertainties = [r[0] for r in rows if r[0] is not None]
            grades = [r[1] for r in rows if r[1] is not None]

            # Distinguish numerically ingested uncertainties from grade-backfilled ones.
            # Rows where accuracy_grade is NULL had uncertainty assigned numerically
            # (Kramida 2024 data) since the grade was not stored; rows with a grade
            # were assigned via the grade→sigma heuristic or ingested alongside a grade.
            numerical_rows = conn.execute(
                "SELECT aki_uncertainty "
                "FROM lines "
                "WHERE element = ? AND sp_num = ? "
                "  AND aki IS NOT NULL AND aki_uncertainty IS NOT NULL "
                "  AND accuracy_grade IS NULL",
                (element, sp_num),
            ).fetchall()
            numerical_unc = [r[0] for r in numerical_rows if r[0] is not None]

            grade_rows = conn.execute(
                "SELECT aki_uncertainty "
                "FROM lines "
                "WHERE element = ? AND sp_num = ? "
                "  AND aki IS NOT NULL AND aki_uncertainty IS NOT NULL "
                "  AND accuracy_grade IS NOT NULL",
                (element, sp_num),
            ).fetchall()
            grade_unc = [r[0] for r in grade_rows if r[0] is not None]

            unique_grades: dict[str, int] = {}
            for g in grades:
                unique_grades[g] = unique_grades.get(g, 0) + 1

            results[label] = {
                "element": element,
                "sp_num": sp_num,
                "n_lines_with_uncertainty": len(uncertainties),
                "n_numerical_uncertainty": len(numerical_unc),
                "n_grade_backfilled": len(grade_unc),
                "median_aki_uncertainty": _median_or_none(uncertainties),
                "median_numerical_uncertainty": _median_or_none(numerical_unc),
                "median_grade_backfilled_uncertainty": _median_or_none(grade_unc),
                "grade_distribution": unique_grades,
            }
        return results
    finally:
        conn.close()


def print_report(medians: dict) -> None:
    """Print a human-readable table of the median sigma(Aki) per ion."""
    header = f"{'Ion':<8} {'N lines':>8} {'N numerical':>12} {'Median σ(Aki)':>14} {'Median σ numerical':>20} {'Median σ grade-fill':>22}"
    print()
    print("Aki Uncertainty Medians by Ion")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, stats in medians.items():
        med_all = stats["median_aki_uncertainty"]
        med_num = stats["median_numerical_uncertainty"]
        med_grd = stats["median_grade_backfilled_uncertainty"]
        print(
            f"{label:<8} "
            f"{stats['n_lines_with_uncertainty']:>8} "
            f"{stats['n_numerical_uncertainty']:>12} "
            f"{med_all if med_all is not None else 'N/A':>14.4f} "
            f"{med_num if med_num is not None else 'N/A':>20} "
            f"{med_grd if med_grd is not None else 'N/A':>22}"
        )
    print()
    print("Grade distributions:")
    for label, stats in medians.items():
        grades = stats["grade_distribution"]
        grade_str = ", ".join(f"{g}={n}" for g, n in sorted(grades.items()))
        print(f"  {label}: {grade_str or '(none)'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report median sigma(Aki) for Fe I, Fe II, Cr II, Mn II.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default="ASD_da/libs_production.db",
        help="Path to the SQLite atomic database (default: ASD_da/libs_production.db)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for machine-readable results",
    )
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).is_file():
        print(f"Error: database not found at '{db_path}'")
        print("Generate it first with:  cflibs generate-db  or  python datagen_v2.py")
        return 1

    print(f"Database: {db_path}")
    medians = compute_medians(db_path)
    print_report(medians)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_meta": {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "db_path": str(Path(db_path).resolve()),
                "description": (
                    "Median fractional aki_uncertainty per ion after "
                    "Kramida 2024 NIST Aki uncertainty refresh (CF-LIBS-improved-f25n)."
                ),
            },
            "ions": medians,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Written JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
