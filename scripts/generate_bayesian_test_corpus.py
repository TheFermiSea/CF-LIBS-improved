#!/usr/bin/env python3
"""
Generate synthetic LIBS test corpus for Bayesian sparse inversion validation.

Produces 48-96 spectra with known ground-truth plasma parameters, covering:
- Pure single-element (8-12 spectra)
- Binary mixtures (8-12)
- Steel-like compositions (8-12)
- 4-element mixtures (8-12)

Plasma parameters are calibrated for ps-LIBS (1ps, 1040nm Yb:fiber laser):
- T: 0.5-1.0 eV (cooler than ns-LIBS due to minimal laser-plasma reheating)
- ne: 1e16-1e17 cm^-3 (coupled: ne = 10^(15 + 2*T_eV))
- Noise: detector-limited (less continuum from ps pulse)
- Matrix effects: reduced (non-thermal ps ablation → stoichiometry ≈ bulk)

Parameter grid validated by Gemini 3.1 Pro (2026-04-13) against:
- McWhirter LTE criterion
- ps-LIBS literature values
- Physical T-ne coupling for picosecond-ablated plasmas

Usage:
    python scripts/generate_bayesian_test_corpus.py --db-path ASD_da/libs_production.db
    python scripts/generate_bayesian_test_corpus.py --db-path ASD_da/libs_production.db --output-dir output/bayesian_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# ps-LIBS Parameter Grid (Gemini-validated for 1ps/1040nm)
# ---------------------------------------------------------------------------

# Temperature grid (eV) — ps-LIBS is cooler than ns-LIBS
T_EV_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Electron density coupling: ne = 10^(15 + 2*T_eV)
# This yields ne from 1e16 (at 0.5 eV) to 1e17 (at 1.0 eV)


def coupled_ne(T_eV: float) -> float:
    """Compute electron density from temperature using ps-LIBS coupling."""
    return 10 ** (15.0 + 2.0 * T_eV)


# Resolving power grid
RP_GRID = [300, 600, 1000]

# SNR grid (dB) — ps-LIBS is more detector-limited
SNR_DB_GRID = [20, 50, 100]


# ---------------------------------------------------------------------------
# Composition Categories
# ---------------------------------------------------------------------------


def pure_element_compositions() -> List[Dict[str, Any]]:
    """Pure single-element spectra with 0.1% dark-element residual.

    The residual tests the algorithm's false-positive rejection: a trace
    amount of a second element is present but should NOT dominate the
    identification result.
    """
    return [
        {"label": "pure_Fe", "composition": {"Fe": 0.999, "Mn": 0.001}},
        {"label": "pure_Al", "composition": {"Al": 0.999, "Mg": 0.001}},
        {"label": "pure_Cu", "composition": {"Cu": 0.999, "Zn": 0.001}},
        {"label": "pure_Ti", "composition": {"Ti": 0.999, "V": 0.001}},
        {"label": "pure_Ni", "composition": {"Ni": 0.999, "Co": 0.001}},
        {"label": "pure_Cr", "composition": {"Cr": 0.999, "Mo": 0.001}},
        {"label": "pure_Si", "composition": {"Si": 0.999, "C": 0.001}},
        {"label": "pure_W", "composition": {"W": 0.999, "Re": 0.001}},
    ]


def binary_compositions() -> List[Dict[str, Any]]:
    """Binary mixtures spanning different element pairs."""
    return [
        {"label": "AlCu_50_50", "composition": {"Al": 0.50, "Cu": 0.50}},
        {"label": "FeNi_80_20", "composition": {"Fe": 0.80, "Ni": 0.20}},
        {"label": "TiAl_90_10", "composition": {"Ti": 0.90, "Al": 0.10}},
        {"label": "CuZn_70_30", "composition": {"Cu": 0.70, "Zn": 0.30}},
        {"label": "FeCr_85_15", "composition": {"Fe": 0.85, "Cr": 0.15}},
        {"label": "AlSi_75_25", "composition": {"Al": 0.75, "Si": 0.25}},
        {"label": "NiCo_60_40", "composition": {"Ni": 0.60, "Co": 0.40}},
        {"label": "FeMn_95_05", "composition": {"Fe": 0.95, "Mn": 0.05}},
    ]


def steel_compositions() -> List[Dict[str, Any]]:
    """Steel-like alloys matching common CRM compositions.

    ps-LIBS matrix effects are reduced (non-thermal ablation), so
    the plasma stoichiometry closely mirrors the bulk composition.
    """
    return [
        {
            "label": "ss304",
            "composition": {
                "Fe": 0.695,
                "Cr": 0.185,
                "Ni": 0.085,
                "Mn": 0.020,
                "Si": 0.010,
                "C": 0.005,
            },
        },
        {
            "label": "ss316",
            "composition": {
                "Fe": 0.655,
                "Cr": 0.170,
                "Ni": 0.120,
                "Mo": 0.025,
                "Mn": 0.020,
                "Si": 0.010,
            },
        },
        {
            "label": "mild_steel",
            "composition": {"Fe": 0.980, "C": 0.005, "Mn": 0.010, "Si": 0.005},
        },
        {
            "label": "tool_steel",
            "composition": {
                "Fe": 0.820,
                "Cr": 0.050,
                "W": 0.060,
                "V": 0.020,
                "Mo": 0.040,
                "C": 0.010,
            },
        },
        {
            "label": "duplex_ss",
            "composition": {
                "Fe": 0.630,
                "Cr": 0.225,
                "Ni": 0.060,
                "Mo": 0.032,
                "Mn": 0.020,
                "N": 0.002,
                "Si": 0.031,
            },
        },
        {
            "label": "maraging_steel",
            "composition": {"Fe": 0.670, "Ni": 0.180, "Co": 0.090, "Mo": 0.050, "Ti": 0.010},
        },
        {
            "label": "hsla_steel",
            "composition": {
                "Fe": 0.970,
                "Mn": 0.015,
                "Si": 0.005,
                "Nb": 0.005,
                "V": 0.003,
                "Ti": 0.002,
            },
        },
        {
            "label": "hadfield_steel",
            "composition": {"Fe": 0.860, "Mn": 0.120, "C": 0.012, "Si": 0.008},
        },
    ]


def four_element_compositions() -> List[Dict[str, Any]]:
    """4-element mixtures covering diverse matrix types."""
    return [
        {"label": "soil_like", "composition": {"Si": 0.40, "Al": 0.20, "Ca": 0.20, "Fe": 0.20}},
        {"label": "brass_like", "composition": {"Cu": 0.60, "Zn": 0.30, "Pb": 0.05, "Sn": 0.05}},
        {"label": "inconel_like", "composition": {"Ni": 0.55, "Cr": 0.22, "Fe": 0.18, "Mo": 0.05}},
        {"label": "bronze_like", "composition": {"Cu": 0.88, "Sn": 0.07, "Zn": 0.03, "Pb": 0.02}},
        {"label": "AlTiVFe", "composition": {"Al": 0.50, "Ti": 0.25, "V": 0.15, "Fe": 0.10}},
        {"label": "NiCrCoW", "composition": {"Ni": 0.45, "Cr": 0.25, "Co": 0.20, "W": 0.10}},
        {"label": "FeCuMnSi", "composition": {"Fe": 0.70, "Cu": 0.15, "Mn": 0.10, "Si": 0.05}},
        {"label": "CrNiMoV", "composition": {"Cr": 0.40, "Ni": 0.30, "Mo": 0.20, "V": 0.10}},
    ]


# ---------------------------------------------------------------------------
# Spectrum Generation
# ---------------------------------------------------------------------------


def generate_corpus(
    db_path: str,
    wavelength_range: Tuple[float, float] = (200.0, 450.0),
    pixels: int = 2048,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate the full test corpus using the forward model.

    Parameters
    ----------
    db_path : str
        Path to the atomic database (ASD_da/libs_production.db).
    wavelength_range : tuple
        Wavelength range in nm.
    pixels : int
        Number of spectral pixels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict contains: wavelength, intensity, ground_truth, label, metadata.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.spectrum_model import SpectrumModel

    rng = np.random.default_rng(seed)
    delta_lambda = (wavelength_range[1] - wavelength_range[0]) / pixels
    corpus: List[Dict[str, Any]] = []

    # Collect all compositions
    categories = [
        ("pure", pure_element_compositions()),
        ("binary", binary_compositions()),
        ("steel", steel_compositions()),
        ("four_element", four_element_compositions()),
    ]

    # Typical total heavy-particle density for ps-LIBS (lower than ns-LIBS)
    total_species_density = 1e19  # cm^-3

    with AtomicDatabase(db_path) as db:
        for category_name, compositions in categories:
            for comp_spec in compositions:
                label = comp_spec["label"]
                composition = comp_spec["composition"]

                # Select 2 (T, ne) points per composition for diversity
                t_indices = rng.choice(len(T_EV_GRID), size=2, replace=False)
                for ti in t_indices:
                    T_eV = T_EV_GRID[ti]
                    T_K = T_eV / 8.617333262e-5  # eV to K
                    ne = coupled_ne(T_eV)

                    # Add ±0.3 dex scatter to ne
                    ne_scattered = ne * 10 ** rng.normal(0, 0.3)
                    ne_scattered = np.clip(ne_scattered, 1e15, 1e18)

                    # Select RP and SNR
                    rp = int(rng.choice(RP_GRID))
                    snr_db = int(rng.choice(SNR_DB_GRID))
                    snr_linear = 10 ** (snr_db / 20.0)

                    # Build plasma state from number fractions
                    plasma = SingleZoneLTEPlasma.from_number_fractions(
                        T_e=T_K,
                        n_e=ne_scattered,
                        number_fractions=composition,
                        total_species_density_cm3=total_species_density,
                    )

                    # Build instrument model with resolving power
                    instrument = InstrumentModel(resolving_power=rp)

                    # Generate clean spectrum via forward model
                    try:
                        model = SpectrumModel(
                            plasma=plasma,
                            atomic_db=db,
                            instrument=instrument,
                            lambda_min=wavelength_range[0],
                            lambda_max=wavelength_range[1],
                            delta_lambda=delta_lambda,
                        )
                        wavelength_out, intensity_out = model.compute_spectrum()
                    except Exception as exc:
                        print(f"  SKIP {label} T={T_eV:.1f}eV: {exc}", file=sys.stderr)
                        continue

                    clean_spectrum = intensity_out

                    # Apply noise model (detector-limited for ps-LIBS)
                    # Shot noise (Poisson) + readout noise (Gaussian)
                    signal_max = np.max(clean_spectrum)
                    if signal_max <= 0:
                        continue

                    # Scale so peak = snr_linear * readout_noise
                    readout_noise = 1.0  # arbitrary units
                    scale_factor = snr_linear * readout_noise / signal_max
                    scaled = clean_spectrum * scale_factor

                    # Poisson shot noise (variance = signal)
                    noisy = rng.poisson(np.maximum(scaled, 0).astype(int)).astype(np.float64)
                    # Add Gaussian readout noise
                    noisy += rng.normal(0, readout_noise, size=len(noisy))
                    # Add small dark current baseline
                    dark_current = 0.1 * readout_noise
                    noisy += dark_current

                    spectrum_label = f"{category_name}/{label}_T{T_eV:.1f}_ne{np.log10(ne_scattered):.1f}_rp{rp}_snr{snr_db}"

                    elements = list(composition.keys())
                    corpus.append(
                        {
                            "wavelength_nm": wavelength_out.tolist(),
                            "intensity": noisy.tolist(),
                            "ground_truth": {
                                "temperature_K": float(T_K),
                                "temperature_eV": float(T_eV),
                                "electron_density_cm3": float(ne_scattered),
                                "concentrations": {k: float(v) for k, v in composition.items()},
                                "elements_present": elements,
                            },
                            "label": spectrum_label,
                            "metadata": {
                                "category": category_name,
                                "composition_label": label,
                                "resolving_power": int(rp),
                                "snr_db": int(snr_db),
                                "laser": "1ps_1040nm_Yb_fiber",
                                "plasma_regime": "ps-LIBS",
                                "noise_model": "poisson_shot + gaussian_readout + dark_current",
                            },
                        }
                    )

    print(f"Generated {len(corpus)} spectra across {len(categories)} categories")
    return corpus


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Bayesian test corpus for ps-LIBS")
    parser.add_argument("--db-path", required=True, help="Path to atomic database")
    parser.add_argument("--output-dir", default="output/bayesian_corpus", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wl-min", type=float, default=200.0, help="Min wavelength (nm)")
    parser.add_argument("--wl-max", type=float, default=450.0, help="Max wavelength (nm)")
    parser.add_argument("--pixels", type=int, default=2048, help="Number of spectral pixels")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = generate_corpus(
        db_path=args.db_path,
        wavelength_range=(args.wl_min, args.wl_max),
        pixels=args.pixels,
        seed=args.seed,
    )

    # Save corpus
    corpus_path = output_dir / "bayesian_test_corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {len(corpus)} spectra to {corpus_path}")

    # Save metadata summary
    meta = {
        "n_spectra": len(corpus),
        "categories": {},
        "parameter_grid": {
            "T_eV": T_EV_GRID,
            "ne_coupling": "10^(15 + 2*T_eV)",
            "RP": RP_GRID,
            "SNR_dB": SNR_DB_GRID,
        },
        "laser": "1ps 1040nm Yb:fiber",
        "plasma_regime": "ps-LIBS",
        "seed": args.seed,
    }
    for entry in corpus:
        cat = entry["metadata"]["category"]
        meta["categories"][cat] = meta["categories"].get(cat, 0) + 1

    meta_path = output_dir / "corpus_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
