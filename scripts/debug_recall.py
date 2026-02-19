#!/usr/bin/env python
"""Quick diagnostic: show pre-gate and post-gate CL values for ALIAS failures."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import numpy as np
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.alias_identifier import ALIASIdentifier

# Import loaders from validate script
sys.path.insert(0, str(Path(__file__).parent))
from validate_real_data import load_netcdf, load_hdf5, load_scipp, estimate_resolving_power

db = AtomicDatabase("ASD_da/libs_production.db")

CASES = [
    (
        "steel_245nm",
        "steel_245nm.nc",
        "netcdf",
        ["Fe", "Cr", "Ni", "Mn", "Cu", "Ti", "Si"],
        ["Fe", "Cr", "Ni", "Mn"],
        None,
    ),
    ("Ni_245nm", "Ni_245nm", "hdf5", ["Fe", "Ni", "Cr", "Mn", "Cu", "Ti", "Si"], ["Ni"], None),
    ("FeNi_380nm", "FeNi_380nm.nc", "netcdf", ["Fe", "Ni", "Cr", "Mn"], ["Fe", "Ni"], None),
    (
        "Ti6Al4V",
        "Ti6Al4V_substrate.h5",
        "scipp",
        ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        ["Ti", "Al", "V"],
        500,
    ),
]

for name, path, loader, elements, expected, rp in CASES:
    data_path = Path("data") / path
    if not data_path.exists():
        print(f"SKIP {name}: {data_path} not found")
        continue

    if loader == "netcdf":
        wavelength, data, _ = load_netcdf(str(data_path))
    elif loader == "hdf5":
        wavelength, data, _ = load_hdf5(str(data_path))
    elif loader == "scipp":
        wavelength, data, _ = load_scipp(str(data_path))
    else:
        raise ValueError(f"Unsupported loader '{loader}' for case {name}")

    # Get mean spectrum
    if data.ndim == 3:
        spectrum = np.mean(data, axis=(0, 1))
    elif data.ndim == 2:
        spectrum = np.mean(data, axis=0)
    else:
        spectrum = data

    if rp is None:
        rp = estimate_resolving_power(wavelength, spectrum)

    identifier = ALIASIdentifier(db, elements=elements, resolving_power=rp)
    result = identifier.identify(wavelength, spectrum)

    print(f"\n{'='*100}")
    print(f"Dataset: {name}  (RP={rp:.0f}, expected={expected})")
    print(f"{'='*100}")
    print(
        f"{'Element':<8} {'Det':<5} {'k_det':<8} {'CL':<8} {'k_sim':<7} {'k_rate':<7} {'k_shift':<7} {'P_maj':<7} {'P_local':<8} {'P_mix':<8} {'R_rat':<7} {'P_SNR':<7} {'P_ab':<6} {'N_m/N_e':<8}"
    )
    print("-" * 110)
    for e in sorted(result.all_elements, key=lambda x: x.confidence, reverse=True):
        m = e.metadata
        det = "YES" if e.detected else "no"
        if e.element in expected:
            det = f"*{det}*" if e.detected else "MISS"
        print(
            f"{e.element:<8} {det:<5} {m.get('k_det', 0):<8.3f} {e.confidence:<8.4f} "
            f"{m.get('k_sim', 0):<7.3f} {m.get('k_rate', 0):<7.3f} {m.get('k_shift', 0):<7.3f} "
            f"{m.get('P_maj', 0):<7.3f} {m.get('P_local', 0):<8.3f} {m.get('P_mix', 0):<8.3f} "
            f"{m.get('R_rat', 0):<7.3f} ",
            end="",
        )
        # Reconstruct P_SNR from CL/k_det/P_maj/P_ab and gates
        k_det = m.get("k_det", 0)
        P_ab = m.get("P_ab", 1.0)
        P_maj = m.get("P_maj", 0.5)
        P_local = m.get("P_local", 1.0)
        P_mix = m.get("P_mix", 1.0)
        R_rat = m.get("R_rat", 0.5)
        # CL = k_det * P_SNR * P_maj * P_ab * gate1 * gate2 * gate3
        gate1 = float(np.clip(2.0 * P_local, 0.1, 1.0))
        gate2 = 0.1 + 0.9 * min(P_mix, 1.0)
        gate3 = 0.5 + 0.5 * R_rat
        denom = k_det * P_maj * P_ab * gate1 * gate2 * gate3
        P_SNR = e.confidence / denom if denom > 1e-10 else 0.0
        print(f"{P_SNR:<7.3f} {P_ab:<6.2f} {e.n_matched_lines}/{e.n_total_lines}")
