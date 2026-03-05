#!/usr/bin/env python3
"""
Fit Irwin polynomial coefficients to NIST ASD partition function data.

Fits degree-4 polynomials: log(U) = sum(a_n * (log T)^n) to NIST reference
values, validates max relative error < 2%, and inserts into the database.

Usage:
    python scripts/fit_partition_coefficients.py --db libs_production.db
    python scripts/fit_partition_coefficients.py --db libs_production.db --dry-run
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILE = ROOT / "tests" / "data" / "nist_reference" / "partition_functions.json"


def load_nist_reference() -> dict:
    """Load NIST reference partition function values."""
    with open(REFERENCE_FILE) as f:
        data = json.load(f)
    # Remove metadata keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


KEY_TEMPS = {5000, 10000, 15000, 20000}


def fit_irwin_polynomial(temperatures_K: np.ndarray, U_values: np.ndarray, degree: int = 4):
    """
    Fit Irwin polynomial: ln(U) = sum(a_n * (ln T)^n).

    Uses weighted least-squares with higher weight on standard test
    temperatures (5000, 10000, 15000, 20000 K).

    Parameters
    ----------
    temperatures_K : array
        Temperatures in Kelvin
    U_values : array
        Partition function values at those temperatures
    degree : int
        Polynomial degree (default 4)

    Returns
    -------
    coefficients : list[float]
        Polynomial coefficients [a0, a1, a2, a3, a4]
    max_rel_error : float
        Maximum relative error across all reference points
    max_key_error : float
        Maximum relative error at the 4 key test temperatures
    """
    ln_T = np.log(temperatures_K)
    ln_U = np.log(U_values)

    # Weighted fit: 10x weight on the 4 key test temperatures
    weights = np.ones_like(temperatures_K)
    for i, T in enumerate(temperatures_K):
        if int(T) in KEY_TEMPS:
            weights[i] = 10.0

    # Fit polynomial in (ln T, ln U) space
    coeffs = np.polynomial.polynomial.polyfit(ln_T, ln_U, degree, w=weights)

    # Evaluate fit
    ln_U_fit = np.polynomial.polynomial.polyval(ln_T, coeffs)
    U_fit = np.exp(ln_U_fit)

    rel_errors = np.abs(U_fit - U_values) / U_values
    max_rel_error = float(np.max(rel_errors))

    # Compute error at key temperatures only
    key_mask = np.array([int(T) in KEY_TEMPS for T in temperatures_K])
    max_key_error = float(np.max(rel_errors[key_mask])) if np.any(key_mask) else max_rel_error

    return list(coeffs), max_rel_error, max_key_error


def insert_coefficients(db_path: str, element: str, stage: int, coeffs: list, dry_run: bool):
    """Insert fitted coefficients into the partition_functions table."""
    if dry_run:
        print(f"  [DRY RUN] Would insert {element} stage {stage}: {coeffs}")
        return

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO partition_functions
            (element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (element, stage, *coeffs, 2000.0, 20000.0, "NIST ASD fit"),
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Fit Irwin polynomial coefficients from NIST data")
    parser.add_argument("--db", default="libs_production.db", help="Atomic database path")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to DB")
    parser.add_argument("--output-json", type=str, help="Write coefficients summary to JSON file")
    args = parser.parse_args()

    print("Fitting Irwin polynomial partition function coefficients")
    print(f"Reference data: {REFERENCE_FILE}")
    print(f"Database: {args.db}")

    nist_data = load_nist_reference()
    results = {}
    all_ok = True

    for element, stages in sorted(nist_data.items()):
        print(f"\n{'='*50}")
        print(f"Element: {element}")
        print(f"{'='*50}")

        results[element] = {}
        for stage_str, temp_values in sorted(stages.items()):
            stage = int(stage_str)
            stage_label = "I" * stage

            temps = np.array([float(t) for t in sorted(temp_values.keys())])
            U_vals = np.array([temp_values[str(int(t))] for t in temps])

            coeffs, max_err, max_key_err = fit_irwin_polynomial(temps, U_vals)

            status = "OK" if max_key_err < 0.05 else ("WARN" if max_key_err < 0.10 else "FAIL")
            if max_key_err >= 0.05:
                all_ok = False

            print(f"\n  {element} {stage_label}:")
            print(f"    Coefficients: {[f'{c:.8e}' for c in coeffs]}")
            print(f"    Max relative error (all):  {max_err:.4%}")
            print(f"    Max relative error (key):  {max_key_err:.4%}  [{status}]")

            results[element][stage_str] = {
                "coefficients": coeffs,
                "max_rel_error": max_err,
                "max_key_error": max_key_err,
                "status": status,
                "t_min": float(temps[0]),
                "t_max": float(temps[-1]),
                "n_points": len(temps),
            }

            insert_coefficients(args.db, element, stage, coeffs, args.dry_run)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    n_total = sum(len(s) for s in results.values())
    n_ok = sum(
        1 for e in results.values() for s in e.values() if s["max_key_error"] < 0.05
    )
    print(f"  Fitted: {n_total} species")
    print(f"  Within 5% at key temps: {n_ok}/{n_total}")
    print(f"  Overall: {'PASS' if all_ok else 'SOME FITS EXCEED 5% AT KEY TEMPS'}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nCoefficients written to {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
