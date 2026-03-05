#!/usr/bin/env python3
"""
Fetch NIST LIBS reference spectra for test fixture generation.

Queries the NIST ASD LIBS interface for each element at standard conditions
(T=1.0 eV, n_e=1e17 cm^-3, R=1000) and saves stick spectra as CSV fixtures.

Usage:
    python scripts/fetch_nist_reference_spectra.py
    python scripts/fetch_nist_reference_spectra.py --elements Fe Cu Al
    python scripts/fetch_nist_reference_spectra.py --dry-run
"""

import argparse
import ast
import csv
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "tests" / "data" / "nist_reference"

logger = logging.getLogger(__name__)

ELEMENTS = ["Fe", "Cu", "Al", "Ni", "Ti", "Cr"]
DEFAULT_CONDITIONS = {
    "temp_eV": 1.0,
    "eden_cm3": 1e17,
    "resolution": 1000,
    "low_w": 200.0,
    "upp_w": 800.0,
}


def fetch_nist_lines(
    element: str,
    low_w: float,
    upp_w: float,
    resolution: float,
    temp_eV: float,
    eden_cm3: float,
) -> tuple[list[float], list[float]]:
    """Fetch line list from NIST LIBS interface.

    Reuses the URL pattern from generate_nist_reference_spectra.py.
    """
    params = {
        "libs": "1",
        "composition": f"{element}:100",
        "spectra": f"{element}0-2",
        "low_w": f"{low_w:.6f}",
        "upp_w": f"{upp_w:.6f}",
        "show_av": "2",
        "unit": "1",
        "resolution": f"{resolution:.6f}",
        "temp": f"{temp_eV:.6f}",
        "eden": f"{eden_cm3:.6e}",
        "maxcharge": "2",
        "min_rel_int": "0.01",
        "int_scale": "1",
    }
    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CF-LIBS-nist-audit/1.0 (contact: maintainers@thefermisea.org)",
        },
    )
    logger.info("Fetching NIST lines for %s: T=%.2f eV, ne=%.2e", element, temp_eV, eden_cm3)
    with urllib.request.urlopen(req, timeout=45) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    match = re.search(r"var lines = (\[\[.*?\]\]);", html, re.S)
    if not match:
        raise RuntimeError(f"Could not parse NIST lines array for {element}")

    lines = ast.literal_eval(match.group(1))
    wavelengths = [row[0] for row in lines]
    strengths = [row[1] for row in lines]
    return wavelengths, strengths


def save_csv(element: str, wavelengths: list, strengths: list, conditions: dict) -> Path:
    """Save stick spectrum as CSV fixture."""
    T = conditions["temp_eV"]
    ne = conditions["eden_cm3"]
    R = int(conditions["resolution"])
    filename = f"{element}_T{T}eV_ne{ne:.0e}_R{R}.csv"
    filepath = OUTPUT_DIR / filename

    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength_nm", "strength"])
        for wl, s in zip(wavelengths, strengths):
            writer.writerow([f"{wl:.8f}", f"{s:.8e}"])

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Fetch NIST LIBS reference spectra")
    parser.add_argument("--elements", nargs="+", default=ELEMENTS, help="Elements to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without fetching")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = DEFAULT_CONDITIONS

    for element in args.elements:
        if args.dry_run:
            print(f"[DRY RUN] Would fetch {element} spectrum at T={conditions['temp_eV']} eV")
            continue

        try:
            wavelengths, strengths = fetch_nist_lines(
                element=element,
                low_w=conditions["low_w"],
                upp_w=conditions["upp_w"],
                resolution=conditions["resolution"],
                temp_eV=conditions["temp_eV"],
                eden_cm3=conditions["eden_cm3"],
            )
            filepath = save_csv(element, wavelengths, strengths, conditions)
            print(f"  {element}: {len(wavelengths)} lines -> {filepath}")
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            print(f"  {element}: FAILED - {e}", file=sys.stderr)
            continue

    print(f"\nDone. Fixtures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
