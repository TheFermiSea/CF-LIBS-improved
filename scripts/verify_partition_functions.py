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
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import EV_TO_K


# NIST ASD reference partition function values
# Source: NIST Atomic Spectra Database (https://www.nist.gov/pml/atomic-spectra-database)
# These are approximate values from NIST tables for comparison.
NIST_PARTITION_FUNCTIONS = {
    "Fe": {
        1: {  # Fe I (neutral)
            5000: 25.07,
            10000: 41.95,
            15000: 64.38,
            20000: 98.36,
        },
        2: {  # Fe II (singly ionized)
            5000: 30.42,
            10000: 43.38,
            15000: 55.02,
            20000: 69.80,
        },
        3: {  # Fe III (doubly ionized)
            5000: 23.87,
            10000: 30.58,
            15000: 38.55,
            20000: 47.32,
        },
    },
    "Cu": {
        1: {
            5000: 2.03,
            10000: 3.81,
            15000: 6.55,
            20000: 10.16,
        },
        2: {
            5000: 1.00,
            10000: 1.08,
            15000: 1.55,
            20000: 2.42,
        },
    },
    "Al": {
        1: {
            5000: 5.84,
            10000: 5.91,
            15000: 6.06,
            20000: 6.38,
        },
        2: {
            5000: 1.00,
            10000: 1.00,
            15000: 1.01,
            20000: 1.03,
        },
    },
}

TEMPERATURES_K = [5000, 10000, 15000, 20000]


def verify_element(
    solver: SahaBoltzmannSolver,
    element: str,
) -> dict[str, dict]:
    """Compare partition functions for one element."""
    ref = NIST_PARTITION_FUNCTIONS.get(element, {})
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

    db = AtomicDatabase(args.db)
    solver = SahaBoltzmannSolver(db)

    all_results = {}
    for element in args.element:
        print(f"\n{'='*60}")
        print(f"Element: {element}")
        print(f"{'='*60}")
        all_results[element] = verify_element(solver, element)

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
