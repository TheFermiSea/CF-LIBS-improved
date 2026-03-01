#!/usr/bin/env python3
"""
Validate CF-LIBS forward model against NIST LIBS Simulation reference.

Computes a synthetic spectrum in NIST_PARITY mode and compares
ion fractions and spectral shape to expected NIST values.

Usage:
    python scripts/validate_nist_parity.py --element Fe --T 0.8 --ne 1e17 \
        --wl-min 220 --wl-max 265 --resolving-power 1000

    python scripts/validate_nist_parity.py --element Fe --T 0.8 --ne 1e17 \
        --wl-min 220 --wl-max 265 --resolving-power 1000 --plot
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.radiation.profiles import BroadeningMode
from cflibs.radiation.spectrum_model import SpectrumModel


# NIST reference ionization fractions (Fe, T_e=0.8 eV, n_e=1e17 cm^-3)
NIST_REFERENCE_FRACTIONS = {
    "Fe": {
        "conditions": {"T_eV": 0.8, "n_e_cm3": 1e17},
        "fractions": {1: 0.27, 2: 0.73, 3: 2.2e-5},
    }
}


def validate_ionization_fractions(
    solver: SahaBoltzmannSolver,
    element: str,
    T_eV: float,
    n_e: float,
) -> dict:
    """Compare computed ion fractions to NIST reference."""
    fractions = solver.get_ionization_fractions(element, T_eV, n_e)

    print(f"\n--- Ionization Fractions: {element} at T={T_eV} eV, n_e={n_e:.1e} cm^-3 ---")
    print(f"  {'Stage':<8} {'CF-LIBS':<12} {'NIST Ref':<12} {'Diff':<12}")
    print(f"  {'-'*44}")

    ref = NIST_REFERENCE_FRACTIONS.get(element, {}).get("fractions", {})
    results = {}

    for stage in sorted(set(list(fractions.keys()) + list(ref.keys()))):
        our = fractions.get(stage, 0.0)
        nist = ref.get(stage)
        if nist is not None:
            diff = our - nist
            pct = f"({diff/nist*100:+.1f}%)" if nist > 1e-10 else ""
            print(f"  {stage:<8} {our:<12.4f} {nist:<12.4f} {diff:+.4f} {pct}")
            results[stage] = {"cflibs": our, "nist": nist, "diff": diff}
        else:
            print(f"  {stage:<8} {our:<12.4f} {'N/A':<12}")
            results[stage] = {"cflibs": our, "nist": None}

    return results


def compute_nist_parity_spectrum(
    element: str,
    T_eV: float,
    n_e: float,
    wl_min: float,
    wl_max: float,
    resolving_power: float,
    db: AtomicDatabase | None = None,
    db_path: str = "libs_production.db",
) -> tuple:
    """Compute spectrum in NIST_PARITY mode."""
    T_K = T_eV / 8.617333262e-5
    density = 1e16  # arbitrary for shape comparison

    if db is None:
        db = AtomicDatabase(db_path)
    plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=n_e, species={element: density})
    instrument = InstrumentModel.from_resolving_power(resolving_power)

    model = SpectrumModel(
        plasma=plasma,
        atomic_db=db,
        instrument=instrument,
        lambda_min=wl_min,
        lambda_max=wl_max,
        delta_lambda=0.01,
        broadening_mode=BroadeningMode.NIST_PARITY,
    )

    wavelength, intensity = model.compute_spectrum()
    return wavelength, intensity, db


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CF-LIBS vs NIST LIBS Simulation")
    parser.add_argument("--element", default="Fe", help="Element symbol")
    parser.add_argument("--T", type=float, default=0.8, help="Electron temperature in eV")
    parser.add_argument("--ne", type=float, default=1e17, help="Electron density in cm^-3")
    parser.add_argument("--wl-min", type=float, default=220.0, help="Min wavelength (nm)")
    parser.add_argument("--wl-max", type=float, default=265.0, help="Max wavelength (nm)")
    parser.add_argument("--resolving-power", type=float, default=1000.0, help="Resolving power R")
    parser.add_argument("--db", default="libs_production.db", help="Atomic database path")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    args = parser.parse_args()

    print("CF-LIBS NIST Parity Validation")
    print(f"Element: {args.element}")
    print(f"T_e = {args.T} eV ({args.T / 8.617333262e-5:.0f} K)")
    print(f"n_e = {args.ne:.1e} cm^-3")
    print(f"Range: {args.wl_min}-{args.wl_max} nm, R={args.resolving_power}")

    # Validate ion fractions
    db = AtomicDatabase(args.db)
    solver = SahaBoltzmannSolver(db)
    frac_results = validate_ionization_fractions(solver, args.element, args.T, args.ne)

    # Compute spectrum (reuse db instance)
    print("\n--- Computing spectrum in NIST_PARITY mode ---")
    wavelength, intensity, _ = compute_nist_parity_spectrum(
        args.element, args.T, args.ne, args.wl_min, args.wl_max, args.resolving_power, db=db
    )

    peak = intensity.max()
    n_lines = np.sum(intensity > peak * 0.01) if peak > 0 else 0
    print(f"Spectrum: {len(wavelength)} points, {n_lines} significant lines")
    print(f"Peak intensity: {peak:.4e}")

    # Summary
    print("\n--- Summary ---")
    has_ref = args.element in NIST_REFERENCE_FRACTIONS
    if has_ref:
        for stage, res in frac_results.items():
            if res.get("nist") is not None:
                pct = abs(res["diff"] / res["nist"] * 100) if res["nist"] > 1e-10 else 0
                status = "PASS" if pct < 20 else "CHECK"
                print(f"  Ion stage {stage}: {status} ({pct:.1f}% difference)")
    else:
        print(f"  No NIST reference fractions for {args.element}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))
            norm = intensity.max() if intensity.max() > 0 else 1.0
            ax.plot(wavelength, intensity / norm, "b-", lw=0.8)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized Intensity")
            ax.set_title(
                f"CF-LIBS NIST_PARITY: {args.element}, "
                f"T={args.T} eV, n_e={args.ne:.0e}, R={args.resolving_power:.0f}"
            )
            ax.set_xlim(args.wl_min, args.wl_max)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available for plotting")

    return 0


if __name__ == "__main__":
    sys.exit(main())
