#!/usr/bin/env python3
"""
Expand partition function coverage to all elements in the atomic database.

Computes U(T) by direct summation over energy levels in the database,
fits Irwin polynomials (ln U = Σ aₙ (ln T)ⁿ), and inserts into the
partition_functions table.

For elements with missing ground state data (E_min > 0.5 eV), adds
the known ground state degeneracy from NIST before computing U(T).

Usage:
    python scripts/expand_partition_functions.py --db libs_production.db
    python scripts/expand_partition_functions.py --db libs_production.db --dry-run
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Allow running as standalone script without full package install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cflibs.core.constants import EV_TO_K  # noqa: E402

# Temperatures for computing U(T) — 2000 to 20000 K in 1000 K steps
TEMPERATURES = np.arange(2000.0, 20001.0, 1000.0)

# Key test temperatures for error evaluation
KEY_TEMPS = {5000, 10000, 15000, 20000}

# Ground state degeneracies (2J+1) for elements/stages where our
# energy_levels table is missing the ground state (E_min > 0.5 eV).
# Values from NIST Atomic Spectra Database ground configurations.
GROUND_STATE_DEGENERACIES = {
    # (element, sp_num): (g_ground, E_ground_eV)
    ("Ag", 2): (1, 0.0),  # 4d^10 ^1S_0
    ("As", 1): (4, 0.0),  # 4s^2 4p^3 ^4S_3/2
    ("Br", 1): (4, 0.0),  # 4s^2 4p^5 ^2P_3/2
    ("Cl", 1): (4, 0.0),  # 3s^2 3p^5 ^2P_3/2
    ("Ge", 2): (2, 0.0),  # 4s^2 4p ^2P_1/2
    ("Hg", 2): (2, 0.0),  # 5d^10 6s ^2S_1/2
    ("I", 1): (4, 0.0),  # 5s^2 5p^5 ^2P_3/2
    ("Kr", 1): (1, 0.0),  # 4s^2 4p^6 ^1S_0
    ("N", 1): (4, 0.0),  # 2s^2 2p^3 ^4S_3/2
    ("Ni", 2): (6, 0.0),  # 3d^9 ^2D_5/2
    ("P", 1): (4, 0.0),  # 3s^2 3p^3 ^4S_3/2
    ("Pt", 2): (6, 0.0),  # 5d^9 ^2D_5/2
    ("Ru", 2): (10, 0.0),  # 4d^7 ^4F_9/2
    ("S", 1): (5, 0.0),  # 3s^2 3p^4 ^3P_2
    ("Si", 2): (2, 0.0),  # 3s^2 3p ^2P_1/2
    ("Xe", 1): (1, 0.0),  # 5s^2 5p^6 ^1S_0
}


def compute_partition_function_from_levels(
    levels: list[tuple[float, float]],
    T_K: float,
    ip_ev: float | None = None,
) -> float:
    """Compute U(T) = Σ g_i exp(-E_i / kT) from energy levels.

    Parameters
    ----------
    levels : list of (g, energy_ev) tuples
    T_K : float
        Temperature in Kelvin
    ip_ev : float or None
        Ionization potential; levels above 0.98*IP are excluded

    Returns
    -------
    float
        Partition function value
    """
    kT_eV = T_K / EV_TO_K
    max_e = ip_ev * 0.98 if ip_ev else 50.0

    U = 0.0
    for g, e in levels:
        if e <= max_e:
            U += g * np.exp(-e / kT_eV)

    return max(U, 1.0)


def fit_irwin_polynomial(temperatures_K: np.ndarray, U_values: np.ndarray, degree: int = 4):
    """Fit Irwin polynomial: ln(U) = Σ aₙ (ln T)ⁿ.

    Returns (coefficients, max_rel_error, max_key_error).
    """
    ln_T = np.log(temperatures_K)
    ln_U = np.log(U_values)

    # Weighted fit: 10x weight on key temperatures
    weights = np.ones_like(temperatures_K)
    for i, T in enumerate(temperatures_K):
        if int(T) in KEY_TEMPS:
            weights[i] = 10.0

    coeffs = np.polynomial.polynomial.polyfit(ln_T, ln_U, degree, w=weights)

    # Evaluate fit
    ln_U_fit = np.polynomial.polynomial.polyval(ln_T, coeffs)
    U_fit = np.exp(ln_U_fit)

    rel_errors = np.abs(U_fit - U_values) / U_values
    max_rel_error = float(np.max(rel_errors))

    key_mask = np.array([int(T) in KEY_TEMPS for T in temperatures_K])
    max_key_error = float(np.max(rel_errors[key_mask])) if np.any(key_mask) else max_rel_error

    return list(coeffs), max_rel_error, max_key_error


def get_all_species(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Get all element-stage combos with energy levels."""
    cursor = conn.execute(
        "SELECT DISTINCT element, sp_num FROM energy_levels ORDER BY element, sp_num"
    )
    return [(r[0], r[1]) for r in cursor]


def get_energy_levels(
    conn: sqlite3.Connection, element: str, sp_num: int
) -> list[tuple[float, float]]:
    """Get (g, energy_ev) pairs for an element-stage combo."""
    cursor = conn.execute(
        "SELECT g_level, energy_ev FROM energy_levels WHERE element=? AND sp_num=? ORDER BY energy_ev",
        (element, sp_num),
    )
    return [(float(r[0]), float(r[1])) for r in cursor]


def get_ionization_potential(conn: sqlite3.Connection, element: str, sp_num: int) -> float | None:
    """Get ionization potential from species_physics."""
    cursor = conn.execute(
        "SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?",
        (element, sp_num),
    )
    row = cursor.fetchone()
    return float(row[0]) if row else None


def main():
    parser = argparse.ArgumentParser(description="Expand partition function coverage")
    parser.add_argument("--db", default="libs_production.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--output-json", type=str, help="Write results summary to JSON")
    parser.add_argument(
        "--max-error", type=float, default=0.02, help="Max acceptable relative error"
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    species = get_all_species(conn)

    print(f"Found {len(species)} element-stage combinations with energy levels")
    print(
        f"Temperature grid: {TEMPERATURES[0]:.0f} - {TEMPERATURES[-1]:.0f} K ({len(TEMPERATURES)} points)"
    )
    print(f"Max acceptable error: {args.max_error:.1%}")
    print()

    results = {}
    n_fitted = 0
    n_skipped = 0
    n_warnings = 0

    all_coefficients = []  # For database insertion and code generation

    for element, sp_num in species:
        levels = get_energy_levels(conn, element, sp_num)
        ip_ev = get_ionization_potential(conn, element, sp_num)

        # Check if ground state is missing and add it
        key = (element, sp_num)
        e_min = min(e for _, e in levels) if levels else 999.0
        if e_min > 0.5 and key in GROUND_STATE_DEGENERACIES:
            g_ground, e_ground = GROUND_STATE_DEGENERACIES[key]
            levels.insert(0, (g_ground, e_ground))

        if not levels or len(levels) < 2:
            print(f"  SKIP {element:3s} stage {sp_num}: insufficient levels ({len(levels)})")
            n_skipped += 1
            continue

        # Compute U(T) at all temperatures
        U_values = np.array(
            [compute_partition_function_from_levels(levels, T, ip_ev) for T in TEMPERATURES]
        )

        # Fit polynomial
        coeffs, max_err, max_key_err = fit_irwin_polynomial(TEMPERATURES, U_values)

        status = "OK" if max_key_err < args.max_error else "WARN"
        if status == "WARN":
            n_warnings += 1

        stage_label = ["", "I", "II", "III", "IV"][sp_num] if sp_num <= 4 else str(sp_num)
        marker = " (ground added)" if e_min > 0.5 else ""
        print(
            f"  {element:3s} {stage_label:3s}  "
            f"max_err={max_err:7.4%}  key_err={max_key_err:7.4%}  "
            f"U(5kK)={U_values[3]:10.4f}  U(10kK)={U_values[8]:10.4f}  "
            f"U(20kK)={U_values[-1]:10.4f}  [{status}]{marker}"
        )

        results[f"{element}_{sp_num}"] = {
            "element": element,
            "sp_num": sp_num,
            "coefficients": coeffs,
            "max_rel_error": max_err,
            "max_key_error": max_key_err,
            "status": status,
            "n_levels": len(levels),
            "ground_state_added": e_min > 0.5,
            "U_values": {str(int(T)): float(U) for T, U in zip(TEMPERATURES, U_values)},
        }

        all_coefficients.append((element, sp_num, coeffs, max_key_err))
        n_fitted += 1

    # Insert into database
    if not args.dry_run:
        cursor = conn.cursor()
        data_to_insert = [
            (element, sp_num, *coeffs) for element, sp_num, coeffs, _ in all_coefficients
        ]
        cursor.executemany(
            """INSERT OR REPLACE INTO partition_functions
               (element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 2000.0, 20000.0, 'NIST ASD energy level sum')""",
            data_to_insert,
        )
        conn.commit()
        print(f"\nInserted {n_fitted} coefficient sets into database")
    else:
        print(f"\n[DRY RUN] Would insert {n_fitted} coefficient sets")

    conn.close()

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total element-stage combos found: {len(species)}")
    print(f"  Fitted: {n_fitted}")
    print(f"  Skipped: {n_skipped}")
    print(f"  Warnings (>{args.max_error:.1%} error): {n_warnings}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"  Results written to {out}")

    return 0 if n_warnings == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
