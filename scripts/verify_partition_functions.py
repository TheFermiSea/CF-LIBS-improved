#!/usr/bin/env python3
"""
Compare CF-LIBS partition functions against NIST ASD reference values.

Evaluates Irwin polynomial partition functions at LIBS-relevant
temperatures and reports percentage differences vs NIST published data.

Usage:
    python scripts/verify_partition_functions.py --element Fe --db libs_production.db
    python scripts/verify_partition_functions.py --element Fe Cu Al
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import EV_TO_K


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILE = ROOT / "tests" / "data" / "nist_reference" / "partition_functions.json"

TEMPERATURES_K = [5000, 10000, 15000, 20000]


def _load_nist_reference() -> dict:
    """Load NIST partition function reference from the canonical JSON fixture."""
    if not REFERENCE_FILE.is_file():
        raise SystemExit(f"NIST reference file not found: {REFERENCE_FILE}")
    with open(REFERENCE_FILE) as f:
        data = json.load(f)
    # Convert to {element: {stage_int: {T_int: U}}} format
    result = {}
    for elem, stages in data.items():
        if elem.startswith("_"):
            continue
        result[elem] = {}
        for stage_str, temps in stages.items():
            result[elem][int(stage_str)] = {int(t): v for t, v in temps.items()}
    return result


def verify_element(
    solver: SahaBoltzmannSolver,
    element: str,
    nist_ref: dict,
) -> dict[int, dict]:
    """Compare partition functions for one element."""
    ref = nist_ref.get(element, {})
    if not ref:
        print(f"  No NIST reference data for {element}")
        return {}

    results = {}
    for stage in sorted(ref.keys()):
        stage_ref = ref[stage]
        stage_label = f"{element} {'I' * stage}"
        print(f"\n  {stage_label}:")
        print(f"    {'T (K)':<10} {'CF-LIBS':<12} {'NIST':<12} {'Diff %':<10} {'Status'}")
        print(f"    {'-'*54}")

        results[stage] = {}
        for T_K in TEMPERATURES_K:
            T_eV = T_K / EV_TO_K
            our_U = solver.calculate_partition_function(element, stage, T_eV)
            nist_U = stage_ref.get(T_K)

            if nist_U is not None:
                pct_diff = (our_U - nist_U) / nist_U * 100
                status = "OK" if abs(pct_diff) < 5 else ("WARN" if abs(pct_diff) < 20 else "FAIL")
                print(f"    {T_K:<10} {our_U:<12.3f} {nist_U:<12.3f} {pct_diff:+8.2f}%  {status}")
                results[stage][T_K] = {
                    "cflibs": our_U,
                    "nist": nist_U,
                    "pct_diff": pct_diff,
                    "status": status,
                }
            else:
                print(f"    {T_K:<10} {our_U:<12.3f} {'N/A':<12}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CF-LIBS partition functions vs NIST ASD")
    parser.add_argument("--element", nargs="+", default=["Fe"], help="Element(s) to verify")
    parser.add_argument("--db", default="libs_production.db", help="Atomic database path")
    args = parser.parse_args()

    print("CF-LIBS Partition Function Verification vs NIST ASD")
    print(f"Database: {args.db}")
    print(f"Temperatures: {TEMPERATURES_K} K")

    nist_ref = _load_nist_reference()
    db = AtomicDatabase(args.db)
    solver = SahaBoltzmannSolver(db)

    all_results = {}
    for element in args.element:
        print(f"\n{'='*60}")
        print(f"Element: {element}")
        print(f"{'='*60}")
        all_results[element] = verify_element(solver, element, nist_ref)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    n_ok = n_warn = n_fail = 0
    for element, stages in all_results.items():
        for stage, temps in stages.items():
            for T_K, data in temps.items():
                s = data.get("status", "")
                if s == "OK":
                    n_ok += 1
                elif s == "WARN":
                    n_warn += 1
                elif s == "FAIL":
                    n_fail += 1

    print(f"  OK (<5%):   {n_ok}")
    print(f"  WARN (5-20%): {n_warn}")
    print(f"  FAIL (>20%):  {n_fail}")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
