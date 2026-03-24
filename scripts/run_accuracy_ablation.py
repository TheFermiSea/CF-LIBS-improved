#!/usr/bin/env python
"""
Accuracy ablation benchmark for CF-LIBS inversion pipeline.

Runs the full CF-LIBS inversion on Aalto mineral spectra with 16 different
configurations (2^4 factorial design), toggling 4 accuracy improvements:

    1. Voigt deconvolution: disabled (trapezoid integration) vs enabled (Voigt fit)
    2. Boltzmann fitting: OLS (no outlier rejection) vs Huber M-estimation
    3. Closure mode: "standard" (direct normalization) vs "ilr" (Isometric Log-Ratio)
    4. Self-absorption: disabled vs CD-SB correction enabled

Measures composition recovery accuracy against known stoichiometric compositions.

Usage:
    python scripts/run_accuracy_ablation.py \
        --db ASD_da/libs_production.db \
        --output validation/accuracy/results/ablation_results.json

    # Run a single config for debugging:
    python scripts/run_accuracy_ablation.py \
        --db ASD_da/libs_production.db \
        --config-index 0

    # Use only 3 factors (skip Voigt, 8 configs):
    python scripts/run_accuracy_ablation.py \
        --db ASD_da/libs_production.db \
        --skip-voigt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.core.logging_config import get_logger  # noqa: E402
from cflibs.inversion.boltzmann import (  # noqa: E402
    BoltzmannPlotFitter,
    FitMethod,
    LineObservation,
)
from cflibs.inversion.cdsb import CDSBLineObservation, CDSBPlotter  # noqa: E402
from cflibs.inversion.closure import ClosureEquation  # noqa: E402
from cflibs.inversion.line_detection import detect_line_observations  # noqa: E402
from cflibs.inversion.solver import CFLIBSResult, IterativeCFLIBSSolver  # noqa: E402

logger = get_logger("accuracy_ablation")


# =============================================================================
# Stoichiometric compositions (mass fractions) for minerals
# =============================================================================
# Computed from idealized mineral formulas. Only LIBS-detectable elements are
# included (O, H, F, Cl excluded). Fractions are renormalized to sum to 1.0
# over detectable elements.

# Atomic masses for stoichiometry calculations
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.065,
    "Cl": 35.453,
    "K": 39.098,
    "Ca": 40.078,
    "Ti": 47.867,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.380,
    "Mo": 95.950,
    "Zr": 91.224,
    "Hg": 200.59,
    "Pb": 207.20,
    "Ta": 180.95,
}

# Elements that LIBS can actually detect (excludes O, H, F, Cl, etc.)
LIBS_DETECTABLE: Set[str] = {
    "Li",
    "Be",
    "B",
    "C",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
}


def _compute_mass_fractions(
    formula: Dict[str, int],
    detectable_only: bool = True,
) -> Dict[str, float]:
    """Compute mass fractions from a chemical formula {element: atom_count}.

    If detectable_only is True, renormalize over LIBS-detectable elements only.
    """
    total_mass = 0.0
    element_masses: Dict[str, float] = {}
    for el, count in formula.items():
        mass = ATOMIC_MASSES.get(el, 0.0) * count
        element_masses[el] = mass
        total_mass += mass

    if total_mass == 0:
        return {}

    fractions: Dict[str, float] = {}
    if detectable_only:
        det_total = sum(m for el, m in element_masses.items() if el in LIBS_DETECTABLE)
        if det_total == 0:
            return {}
        for el, mass in element_masses.items():
            if el in LIBS_DETECTABLE:
                fractions[el] = mass / det_total
    else:
        for el, mass in element_masses.items():
            fractions[el] = mass / total_mass

    return fractions


# Mineral formulas: {element: atom_count_per_formula_unit}
# Idealized compositions; real minerals have solid-solution substitution.
MINERAL_FORMULAS: Dict[str, Dict[str, int]] = {
    # K-feldspars: KAlSi3O8
    "adularia": {"K": 1, "Al": 1, "Si": 3, "O": 8},
    # Aegirine: NaFeSi2O6
    "aegerine": {"Na": 1, "Fe": 1, "Si": 2, "O": 6},
    # Almandine: Fe3Al2Si3O12
    "almandine": {"Fe": 3, "Al": 2, "Si": 3, "O": 12},
    # Apatite: Ca5(PO4)3(OH) — simplified, ignoring OH
    "apatite": {"Ca": 5, "P": 3, "O": 13},
    # Augite: Ca(Mg,Fe)(SiO3)2 — use Ca Mg Fe Si average
    "augite": {"Ca": 1, "Mg": 1, "Fe": 1, "Al": 0, "Si": 2, "O": 6},
    # Beryl: Be3Al2Si6O18
    "beryl": {"Be": 3, "Al": 2, "Si": 6, "O": 18},
    # Biotite: K(Mg,Fe)3AlSi3O10(OH)2 — use K Mg Fe Al Si average
    "biotite": {"K": 1, "Mg": 1, "Fe": 2, "Al": 1, "Si": 3, "O": 12},
    # Chalcopyrite: CuFeS2
    "chalcopyrite": {"Cu": 1, "Fe": 1, "S": 2},
    # Cinnabar: HgS
    "cinnabar": {"Hg": 1, "S": 1},
    # Cordierite: Mg2Al4Si5O18
    "cordierite": {"Mg": 2, "Al": 4, "Si": 5, "O": 18},
    # Corundum: Al2O3
    "corundum": {"Al": 2, "O": 3},
    # Diopside: CaMgSi2O6
    "diopside": {"Ca": 1, "Mg": 1, "Si": 2, "O": 6},
    # Fluorite: CaF2
    "fluorite": {"Ca": 1, "F": 2},
    # Galena: PbS
    "galena": {"Pb": 1, "S": 1},
    # Garnet (almandine-pyrope average): (Fe,Mg,Ca)3Al2Si3O12
    "garnet": {"Fe": 1, "Mg": 1, "Ca": 1, "Al": 2, "Si": 3, "O": 12},
    # Gypsum: CaSO4 * 2H2O
    "gypsum": {"Ca": 1, "S": 1, "O": 6, "H": 4},
    # Hematite: Fe2O3
    "hematite": {"Fe": 2, "O": 3},
    # Hornblende: Ca2(Mg,Fe)4Al(Si7Al)O22(OH)2 — simplified
    "hornblende": {"Ca": 2, "Mg": 2, "Fe": 2, "Al": 2, "Si": 7, "O": 24},
    # Hypersthene: (Mg,Fe)SiO3
    "hypersthene": {"Fe": 1, "Mg": 1, "Si": 2, "O": 6},
    # Kaolinite: Al2Si2O5(OH)4
    "kaolinite": {"Al": 2, "Si": 2, "O": 9, "H": 4},
    # Kyanite: Al2SiO5
    "kyanite": {"Al": 2, "Si": 1, "O": 5},
    # Lepidolite: KLi2AlSi4O10(F,OH)2 — simplified
    "lepidolite": {"K": 1, "Li": 2, "Al": 1, "Si": 4, "O": 12},
    # Magnesite: MgCO3
    "magnesite": {"Mg": 1, "C": 1, "O": 3},
    # Magnetite: Fe3O4
    "magnetite": {"Fe": 3, "O": 4},
    # Microcline: KAlSi3O8
    "microcline": {"K": 1, "Al": 1, "Si": 3, "O": 8},
    # Molybdenite: MoS2
    "molybdenite": {"Mo": 1, "S": 2},
    # Muscovite: KAl2(AlSi3O10)(OH)2
    "muscovite": {"K": 1, "Al": 3, "Si": 3, "O": 12, "H": 2},
    # Olivine: (Mg,Fe)2SiO4
    "olivine": {"Mg": 1, "Fe": 1, "Si": 1, "O": 4},
    # Orthoclase: KAlSi3O8
    "orthoclase": {"K": 1, "Al": 1, "Si": 3, "O": 8},
    # Pentlandite: (Fe,Ni)9S8
    "pentlandite": {"Fe": 5, "Ni": 4, "S": 8},
    # Phlogopite: KMg3AlSi3O10(OH)2
    "phlogopite": {"K": 1, "Mg": 3, "Al": 1, "Si": 3, "O": 12, "H": 2},
    # Plagioclase: (Na0.5Ca0.5)(Al1.5Si2.5)O8 — labradorite-like
    "plagioclase": {"Na": 1, "Ca": 1, "Al": 2, "Si": 3, "O": 8},
    # Pyrite: FeS2
    "pyrite": {"Fe": 1, "S": 2},
    # Pyrrhotite: Fe7S8 (common superstructure)
    "pyrrhotite": {"Fe": 7, "S": 8},
    # Quartz: SiO2
    "quartz": {"Si": 1, "O": 2},
    # Scapolite: Na4Al3Si9O24Cl — simplified meionite-marialite
    "scapolite": {"Na": 2, "Ca": 2, "Al": 4, "Si": 8, "O": 24},
    # Serpentine: Mg3Si2O5(OH)4
    "serpentine": {"Mg": 3, "Si": 2, "O": 9, "H": 4},
    # Siderite: FeCO3
    "siderite": {"Fe": 1, "C": 1, "O": 3},
    # Sphalerite: ZnS
    "sphalerite": {"Zn": 1, "S": 1},
    # Sphene (Titanite): CaTiSiO5
    "sphene": {"Ca": 1, "Ti": 1, "Si": 1, "O": 5},
    # Spodumene: LiAlSi2O6
    "spodumene": {"Li": 1, "Al": 1, "Si": 2, "O": 6},
    # Staurolite: Fe2Al9Si4O23(OH)
    "staurolite": {"Fe": 2, "Al": 9, "Si": 4, "O": 24, "H": 1},
    # Talc: Mg3Si4O10(OH)2
    "talc": {"Mg": 3, "Si": 4, "O": 12, "H": 2},
    # Topaz: Al2SiO4(F,OH)2
    "topaz": {"Al": 2, "Si": 1, "O": 6},
    # Tourmaline: NaMg3Al6B3Si6O27(OH)4 — dravite end-member
    "tourmaline": {"Na": 1, "Mg": 3, "Al": 6, "Si": 6, "B": 3, "O": 31, "H": 4},
    # Tremolite: Ca2Mg5Si8O22(OH)2
    "tremolite": {"Ca": 2, "Mg": 5, "Si": 8, "O": 24, "H": 2},
    # Wollastonite: CaSiO3
    "wollastonite": {"Ca": 1, "Si": 1, "O": 3},
    # Zircon: ZrSiO4
    "zircon": {"Zr": 1, "Si": 1, "O": 4},
    # Mn-Tantalite: MnTa2O6
    "mntantalite": {"Mn": 1, "Ta": 2, "O": 6},
}

# Precompute stoichiometric mass fractions (LIBS-detectable only)
MINERAL_STOICHIOMETRIC: Dict[str, Dict[str, float]] = {
    name: _compute_mass_fractions(formula, detectable_only=True)
    for name, formula in MINERAL_FORMULAS.items()
}


def get_mineral_name(filename: str) -> str:
    """Extract mineral name from filename like 'adulariaE11_spectrum'."""
    import re

    name = filename.replace("_spectrum", "")
    match = re.match(r"([a-zA-Z]+)", name)
    return match.group(1).lower() if match else name.lower()


# =============================================================================
# Ablation configuration
# =============================================================================


@dataclass
class AblationConfig:
    """A single ablation configuration."""

    name: str
    use_deconvolution: bool  # True = Voigt deconvolution, False = trapezoid
    outlier_method: Optional[str]  # None = OLS, "huber" = Huber M-estimation
    closure_mode: str  # "standard" or "ilr"
    self_absorption: bool  # True = CD-SB correction

    @property
    def short_label(self) -> str:
        parts = []
        parts.append("Voigt" if self.use_deconvolution else "Trap")
        parts.append("Huber" if self.outlier_method == "huber" else "OLS")
        parts.append("ILR" if self.closure_mode == "ilr" else "Std")
        parts.append("SA" if self.self_absorption else "noSA")
        return "_".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "use_deconvolution": self.use_deconvolution,
            "outlier_method": self.outlier_method,
            "closure_mode": self.closure_mode,
            "self_absorption": self.self_absorption,
        }


def build_configs(skip_voigt: bool = False) -> List[AblationConfig]:
    """Build all factorial combinations of ablation factors."""
    if skip_voigt:
        voigt_levels = [False]
    else:
        voigt_levels = [False, True]

    outlier_levels = [None, "huber"]
    closure_levels = ["standard", "ilr"]
    sa_levels = [False, True]

    configs = []
    for voigt, outlier, closure, sa in product(
        voigt_levels, outlier_levels, closure_levels, sa_levels
    ):
        cfg = AblationConfig(
            name="",
            use_deconvolution=voigt,
            outlier_method=outlier,
            closure_mode=closure,
            self_absorption=sa,
        )
        cfg.name = cfg.short_label
        configs.append(cfg)

    return configs


# =============================================================================
# Solver wrapper with ILR closure support
# =============================================================================


class AblationSolver(IterativeCFLIBSSolver):
    """Extended solver that routes 'ilr' closure mode to ClosureEquation.apply_ilr."""

    def solve(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        **closure_kwargs,
    ) -> CFLIBSResult:
        """Solve with ILR closure support added."""
        if closure_mode == "ilr":
            # Run the parent solve with standard closure, then re-do closure
            # with ILR on the converged state.
            # We patch the closure step by temporarily replacing the closure
            # dispatch inside the iteration loop. The simplest approach: run
            # standard first to get converged T, ne, then redo with ILR closure.
            result = super().solve(observations, closure_mode="standard", **closure_kwargs)

            # Now re-run closure with ILR using the converged parameters
            from collections import defaultdict as _dd

            obs_by_element = _dd(list)
            for obs in observations:
                obs_by_element[obs.element].append(obs)

            elements = list(obs_by_element.keys())
            T_K = result.temperature_K
            n_e = result.electron_density_cm3

            ips = {}
            for el in elements:
                ip = self.atomic_db.get_ionization_potential(el, 1)
                ips[el] = ip if ip is not None else 15.0
            effective_ips = self._compute_effective_ips(ips, n_e, T_K)

            partition_funcs = {}
            partition_funcs_II = {}
            for el in elements:
                partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
                partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)

            corrected_obs_map = self._apply_saha_correction(
                dict(obs_by_element), T_K, n_e, effective_ips
            )
            common_fit = self._fit_common_boltzmann_plane(corrected_obs_map)
            if common_fit is None:
                return result

            intercepts = common_fit.intercepts
            abundance_multipliers = self._compute_abundance_multipliers(
                list(intercepts.keys()),
                T_K,
                n_e,
                partition_funcs,
                partition_funcs_II,
                effective_ips,
            )

            closure_res = ClosureEquation.apply_ilr(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

            return CFLIBSResult(
                temperature_K=result.temperature_K,
                temperature_uncertainty_K=result.temperature_uncertainty_K,
                electron_density_cm3=result.electron_density_cm3,
                concentrations=closure_res.concentrations,
                concentration_uncertainties=result.concentration_uncertainties,
                iterations=result.iterations,
                converged=result.converged,
                quality_metrics=result.quality_metrics,
                electron_density_uncertainty_cm3=result.electron_density_uncertainty_cm3,
                boltzmann_covariance=result.boltzmann_covariance,
            )
        else:
            return super().solve(observations, closure_mode=closure_mode, **closure_kwargs)


# =============================================================================
# Self-absorption correction helper
# =============================================================================


def apply_cdsb_correction(
    observations: List[LineObservation],
    atomic_db: AtomicDatabase,
    n_e: float = 1e17,
    initial_T_K: float = 10000.0,
) -> List[LineObservation]:
    """Apply CD-SB self-absorption correction to line observations.

    Converts LineObservation to CDSBLineObservation (using DB transitions for
    lower-level info), runs the CD-SB correction, and returns corrected
    LineObservation list.
    """
    if len(observations) < 3:
        return observations

    cdsb_obs: List[CDSBLineObservation] = []
    for obs in observations:
        # Query transition for lower-level info
        transitions = atomic_db.get_transitions(
            obs.element,
            wavelength_min=obs.wavelength_nm - 0.05,
            wavelength_max=obs.wavelength_nm + 0.05,
        )
        # Find best matching transition
        E_i_ev = 0.0
        g_i = 1
        is_resonance = False
        if transitions:
            best = min(transitions, key=lambda t: abs(t.wavelength_nm - obs.wavelength_nm))
            E_i_ev = best.E_i_ev
            g_i = best.g_i
            is_resonance = E_i_ev < 0.1

        cdsb_obs.append(
            CDSBLineObservation(
                wavelength_nm=obs.wavelength_nm,
                intensity=obs.intensity,
                intensity_uncertainty=obs.intensity_uncertainty,
                element=obs.element,
                ionization_stage=obs.ionization_stage,
                E_k_ev=obs.E_k_ev,
                g_k=obs.g_k,
                A_ki=obs.A_ki,
                E_i_ev=E_i_ev,
                g_i=g_i,
                is_resonance=is_resonance,
            )
        )

    try:
        plotter = CDSBPlotter(
            plasma_length_cm=0.1,
            max_iterations=10,
            convergence_tolerance=0.02,
        )
        result = plotter.fit(cdsb_obs, n_e=n_e, initial_T_K=initial_T_K)

        if result.n_points < 2:
            return observations

        # The CD-SB plotter returns corrected observations internally via
        # _apply_tau_correction. We re-apply the final tau corrections to get
        # corrected LineObservations.
        tau_values = {od.wavelength_nm: od.tau_final for od in result.optical_depths}
        corrected = plotter._apply_tau_correction(cdsb_obs, tau_values)
        if len(corrected) >= 2:
            return corrected
        return observations
    except Exception as exc:
        logger.warning("CD-SB correction failed: %s", exc)
        return observations


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class SpectrumResult:
    """Result for a single spectrum under a single configuration."""

    mineral: str
    spectrum_file: str
    config_name: str
    success: bool
    error_msg: str = ""
    temperature_K: float = 0.0
    electron_density_cm3: float = 0.0
    converged: bool = False
    iterations: int = 0
    r_squared: float = 0.0
    n_observations: int = 0
    n_elements_detected: int = 0
    detected_elements: List[str] = field(default_factory=list)
    recovered_concentrations: Dict[str, float] = field(default_factory=dict)
    stoichiometric_concentrations: Dict[str, float] = field(default_factory=dict)
    per_element_relative_error: Dict[str, float] = field(default_factory=dict)
    rmsep: float = float("nan")
    element_recall: float = 0.0
    element_precision: float = 0.0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mineral": self.mineral,
            "spectrum_file": self.spectrum_file,
            "config_name": self.config_name,
            "success": self.success,
            "error_msg": self.error_msg,
            "temperature_K": self.temperature_K,
            "electron_density_cm3": self.electron_density_cm3,
            "converged": self.converged,
            "iterations": self.iterations,
            "r_squared": self.r_squared,
            "n_observations": self.n_observations,
            "n_elements_detected": self.n_elements_detected,
            "detected_elements": self.detected_elements,
            "recovered_concentrations": self.recovered_concentrations,
            "stoichiometric_concentrations": self.stoichiometric_concentrations,
            "per_element_relative_error": self.per_element_relative_error,
            "rmsep": self.rmsep if np.isfinite(self.rmsep) else None,
            "element_recall": self.element_recall,
            "element_precision": self.element_precision,
            "elapsed_s": self.elapsed_s,
        }


def compute_rmsep(
    recovered: Dict[str, float],
    stoichiometric: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Compute RMSEP and per-element relative errors.

    Compares recovered vs stoichiometric over the union of detectable elements
    in the stoichiometric composition.
    """
    per_element_error: Dict[str, float] = {}
    squared_errors = []

    for el, true_frac in stoichiometric.items():
        pred_frac = recovered.get(el, 0.0)
        # Absolute error for RMSEP
        squared_errors.append((pred_frac - true_frac) ** 2)
        # Relative error (guard against zero)
        if true_frac > 0.01:
            per_element_error[el] = abs(pred_frac - true_frac) / true_frac
        else:
            per_element_error[el] = abs(pred_frac - true_frac)

    if not squared_errors:
        return float("nan"), per_element_error

    rmsep = float(np.sqrt(np.mean(squared_errors)))
    return rmsep, per_element_error


def compute_element_metrics(
    detected: Set[str],
    expected: Set[str],
) -> Tuple[float, float]:
    """Compute element detection recall and precision."""
    if not expected:
        return 0.0, 0.0
    tp = len(detected & expected)
    recall = tp / len(expected) if expected else 0.0
    precision = tp / len(detected) if detected else 0.0
    return recall, precision


# =============================================================================
# Pipeline runner
# =============================================================================


def run_single_spectrum(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    mineral: str,
    spectrum_file: str,
    config: AblationConfig,
    db: AtomicDatabase,
    stoichiometric: Dict[str, float],
    expected_elements: Set[str],
    alias_result_cache: Dict[str, Any],
) -> SpectrumResult:
    """Run one configuration on one spectrum."""
    t0 = time.perf_counter()

    result = SpectrumResult(
        mineral=mineral,
        spectrum_file=spectrum_file,
        config_name=config.name,
        success=False,
        stoichiometric_concentrations=stoichiometric,
    )

    try:
        # Step 1: Element identification (shared across configs)
        cache_key = spectrum_file
        if cache_key not in alias_result_cache:
            from cflibs.inversion.alias_identifier import ALIASIdentifier

            identifier = ALIASIdentifier(db)
            alias_res = identifier.identify(wavelength, intensity)
            alias_result_cache[cache_key] = alias_res
        alias_res = alias_result_cache[cache_key]

        if alias_res is None or not alias_res.detected_elements:
            result.error_msg = "no_elements_detected"
            result.elapsed_s = time.perf_counter() - t0
            return result

        detected_elements = [d.element for d in alias_res.detected_elements]
        result.detected_elements = detected_elements
        result.n_elements_detected = len(detected_elements)

        detected_set = set(detected_elements)
        result.element_recall, result.element_precision = compute_element_metrics(
            detected_set,
            expected_elements,
        )

        # Step 2: Build LineObservations
        det = detect_line_observations(
            wavelength,
            intensity,
            db,
            elements=detected_elements,
            wavelength_tolerance_nm=0.1,
            min_peak_height=0.01,
            peak_width_nm=0.2,
            use_deconvolution=config.use_deconvolution,
        )

        observations = det.observations
        if len(observations) < 3:
            result.error_msg = f"too_few_observations ({len(observations)})"
            result.n_observations = len(observations)
            result.elapsed_s = time.perf_counter() - t0
            return result

        result.n_observations = len(observations)

        # Step 3: Self-absorption correction (optional)
        if config.self_absorption:
            observations = apply_cdsb_correction(observations, db)
            if len(observations) < 3:
                result.error_msg = "too_few_observations_after_sa_correction"
                result.elapsed_s = time.perf_counter() - t0
                return result

        # Step 4: Configure and run solver
        solver = AblationSolver(db, max_iterations=25)

        # Set Boltzmann fitter method
        if config.outlier_method == "huber":
            solver.boltzmann_fitter = BoltzmannPlotFitter(
                method=FitMethod.HUBER,
                outlier_sigma=2.5,
            )
        else:
            # OLS: sigma_clip with very high sigma = effectively no rejection
            solver.boltzmann_fitter = BoltzmannPlotFitter(
                method=FitMethod.SIGMA_CLIP,
                outlier_sigma=100.0,  # effectively no clipping
            )

        # Step 5: Solve
        cflibs_result = solver.solve(observations, closure_mode=config.closure_mode)

        result.temperature_K = cflibs_result.temperature_K
        result.electron_density_cm3 = cflibs_result.electron_density_cm3
        result.converged = cflibs_result.converged
        result.iterations = cflibs_result.iterations
        result.r_squared = cflibs_result.quality_metrics.get("r_squared_last", 0.0)
        result.recovered_concentrations = dict(cflibs_result.concentrations)

        # Step 6: Compare with stoichiometric
        # The solver returns number fractions. Stoichiometric are mass fractions.
        # Convert recovered number fractions to mass fractions for comparison.
        recovered_mass = _number_to_mass_fractions(cflibs_result.concentrations, db)
        result.recovered_concentrations = recovered_mass

        rmsep, per_el_err = compute_rmsep(recovered_mass, stoichiometric)
        result.rmsep = rmsep
        result.per_element_relative_error = per_el_err
        result.success = True

    except Exception as exc:
        result.error_msg = f"exception: {exc}"
        logger.debug(
            "Config %s on %s failed: %s", config.name, spectrum_file, traceback.format_exc()
        )

    result.elapsed_s = time.perf_counter() - t0
    return result


def _number_to_mass_fractions(
    number_fractions: Dict[str, float],
    db: AtomicDatabase,
) -> Dict[str, float]:
    """Convert CF-LIBS number fractions to mass fractions."""
    total = 0.0
    masses: Dict[str, float] = {}
    for el, nf in number_fractions.items():
        aw = ATOMIC_MASSES.get(el, 0.0)
        if aw == 0.0:
            # Try to get from DB
            sp = db.get_species_physics(el, 1)
            if sp and hasattr(sp, "atomic_mass") and sp.atomic_mass:
                aw = sp.atomic_mass
            else:
                aw = 50.0  # Rough fallback
        m = nf * aw
        masses[el] = m
        total += m

    if total == 0:
        return number_fractions

    return {el: m / total for el, m in masses.items()}


# =============================================================================
# Aggregation and reporting
# =============================================================================


@dataclass
class ConfigSummary:
    """Aggregate metrics for one configuration across all spectra."""

    config: AblationConfig
    n_spectra: int = 0
    n_success: int = 0
    n_converged: int = 0
    mean_rmsep: float = float("nan")
    median_rmsep: float = float("nan")
    std_rmsep: float = float("nan")
    mean_recall: float = 0.0
    mean_precision: float = 0.0
    mean_r_squared: float = 0.0
    mean_temperature_K: float = 0.0
    total_elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_spectra": self.n_spectra,
            "n_success": self.n_success,
            "n_converged": self.n_converged,
            "mean_rmsep": self.mean_rmsep if np.isfinite(self.mean_rmsep) else None,
            "median_rmsep": self.median_rmsep if np.isfinite(self.median_rmsep) else None,
            "std_rmsep": self.std_rmsep if np.isfinite(self.std_rmsep) else None,
            "mean_recall": self.mean_recall,
            "mean_precision": self.mean_precision,
            "mean_r_squared": self.mean_r_squared,
            "mean_temperature_K": self.mean_temperature_K,
            "total_elapsed_s": self.total_elapsed_s,
        }


def summarize_config(
    config: AblationConfig,
    results: List[SpectrumResult],
) -> ConfigSummary:
    """Compute aggregate metrics for one configuration."""
    summary = ConfigSummary(config=config)
    summary.n_spectra = len(results)
    summary.n_success = sum(1 for r in results if r.success)
    summary.n_converged = sum(1 for r in results if r.converged)
    summary.total_elapsed_s = sum(r.elapsed_s for r in results)

    successful = [r for r in results if r.success]
    if successful:
        rmseps = [r.rmsep for r in successful if np.isfinite(r.rmsep)]
        if rmseps:
            summary.mean_rmsep = float(np.mean(rmseps))
            summary.median_rmsep = float(np.median(rmseps))
            summary.std_rmsep = float(np.std(rmseps))

        summary.mean_recall = float(np.mean([r.element_recall for r in successful]))
        summary.mean_precision = float(np.mean([r.element_precision for r in successful]))
        summary.mean_r_squared = float(np.mean([r.r_squared for r in successful]))
        temps = [r.temperature_K for r in successful if r.temperature_K > 0]
        summary.mean_temperature_K = float(np.mean(temps)) if temps else 0.0

    return summary


def print_summary_table(summaries: List[ConfigSummary]) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 110)
    print("ACCURACY ABLATION RESULTS")
    print("=" * 110)
    print(
        f"{'Config':<30s} {'OK':>4s} {'Conv':>5s} "
        f"{'RMSEP':>8s} {'med':>8s} {'std':>8s} "
        f"{'Recall':>7s} {'Prec':>7s} {'R2':>6s} {'T(K)':>8s} {'Time':>7s}"
    )
    print("-" * 110)

    sorted_summaries = sorted(
        summaries,
        key=lambda s: s.mean_rmsep if np.isfinite(s.mean_rmsep) else 999.0,
    )

    for s in sorted_summaries:
        rmsep_str = f"{s.mean_rmsep:.4f}" if np.isfinite(s.mean_rmsep) else "   N/A"
        med_str = f"{s.median_rmsep:.4f}" if np.isfinite(s.median_rmsep) else "   N/A"
        std_str = f"{s.std_rmsep:.4f}" if np.isfinite(s.std_rmsep) else "   N/A"
        print(
            f"{s.config.short_label:<30s} {s.n_success:>4d} {s.n_converged:>5d} "
            f"{rmsep_str:>8s} {med_str:>8s} {std_str:>8s} "
            f"{s.mean_recall:>7.3f} {s.mean_precision:>7.3f} {s.mean_r_squared:>6.3f} "
            f"{s.mean_temperature_K:>8.0f} {s.total_elapsed_s:>6.1f}s"
        )

    # Best and worst
    valid = [s for s in sorted_summaries if np.isfinite(s.mean_rmsep)]
    if valid:
        best = valid[0]
        worst = valid[-1]
        print(f"\nBest config:  {best.config.short_label} (RMSEP = {best.mean_rmsep:.4f})")
        print(f"Worst config: {worst.config.short_label} (RMSEP = {worst.mean_rmsep:.4f})")


def print_mineral_breakdown(
    results_by_config: Dict[str, List[SpectrumResult]],
    config_name: str,
    label: str,
) -> None:
    """Print per-mineral breakdown for a given configuration."""
    results = results_by_config.get(config_name, [])
    if not results:
        return

    print(f"\n--- Per-mineral breakdown: {label} ({config_name}) ---")
    print(f"{'Mineral':<20s} {'RMSEP':>8s} {'Recall':>7s} {'T(K)':>8s} {'Conv':>5s} {'nObs':>5s}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x.mineral):
        if not r.success:
            print(f"{r.mineral:<20s} {'FAIL':>8s}   {r.error_msg[:30]}")
            continue
        rmsep_str = f"{r.rmsep:.4f}" if np.isfinite(r.rmsep) else "   N/A"
        print(
            f"{r.mineral:<20s} {rmsep_str:>8s} {r.element_recall:>7.3f} "
            f"{r.temperature_K:>8.0f} {'Y' if r.converged else 'N':>5s} "
            f"{r.n_observations:>5d}"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CF-LIBS accuracy ablation benchmark on Aalto mineral spectra."
    )
    parser.add_argument(
        "--db",
        default="ASD_da/libs_production.db",
        help="Path to atomic database",
    )
    parser.add_argument(
        "--data-dir",
        default="data/aalto_libs/minerals",
        help="Path to mineral spectra directory",
    )
    parser.add_argument(
        "--output",
        default="validation/accuracy/results/ablation_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--skip-voigt",
        action="store_true",
        help="Skip Voigt factor (8 configs instead of 16)",
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help="Run only a single config by index (for debugging)",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        default=None,
        help="Limit number of spectra (for debugging)",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Build configs
    configs = build_configs(skip_voigt=args.skip_voigt)
    if args.config_index is not None:
        if 0 <= args.config_index < len(configs):
            configs = [configs[args.config_index]]
        else:
            print(f"ERROR: config-index {args.config_index} out of range [0, {len(configs)-1}]")
            sys.exit(1)

    print(f"Ablation benchmark: {len(configs)} configurations")
    for i, cfg in enumerate(configs):
        print(f"  [{i:2d}] {cfg.short_label}")

    # Load DB
    db = AtomicDatabase(args.db)
    print(f"\nDatabase: {args.db}")

    # Load mineral spectra
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    spectra: List[Dict[str, Any]] = []
    for f in sorted(data_dir.glob("*_spectrum.csv")):
        mineral = get_mineral_name(f.stem)
        stoich = MINERAL_STOICHIOMETRIC.get(mineral, {})
        if not stoich:
            logger.debug("Skipping %s: no stoichiometric data", f.stem)
            continue

        expected_elements = {el for el in stoich.keys() if el in LIBS_DETECTABLE}
        if not expected_elements:
            continue

        try:
            df = pd.read_csv(f)
            wl = df.iloc[:, 0].values.astype(float)
            inten = df.iloc[:, 1].values.astype(float)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", f, exc)
            continue

        spectra.append(
            {
                "wavelength": wl,
                "intensity": inten,
                "mineral": mineral,
                "file": f.stem,
                "stoichiometric": stoich,
                "expected_elements": expected_elements,
            }
        )

    if args.max_spectra:
        spectra = spectra[: args.max_spectra]

    print(f"Loaded {len(spectra)} mineral spectra")

    if not spectra:
        print("ERROR: No spectra loaded")
        sys.exit(1)

    # Run ablation
    all_results: Dict[str, List[SpectrumResult]] = defaultdict(list)
    alias_cache: Dict[str, Any] = {}  # shared ALIAS results across configs

    total_runs = len(configs) * len(spectra)
    completed = 0

    for cfg_idx, config in enumerate(configs):
        print(f"\n[{cfg_idx+1}/{len(configs)}] Running config: {config.short_label}")

        for spec_idx, spec in enumerate(spectra):
            result = run_single_spectrum(
                wavelength=spec["wavelength"],
                intensity=spec["intensity"],
                mineral=spec["mineral"],
                spectrum_file=spec["file"],
                config=config,
                db=db,
                stoichiometric=spec["stoichiometric"],
                expected_elements=spec["expected_elements"],
                alias_result_cache=alias_cache,
            )
            all_results[config.name].append(result)
            completed += 1

            status = "OK" if result.success else f"FAIL({result.error_msg[:20]})"
            if completed % 20 == 0 or completed == total_runs:
                print(f"  [{completed}/{total_runs}] {spec['mineral']:>15s} " f"-> {status}")

    # Compute summaries
    summaries: List[ConfigSummary] = []
    for config in configs:
        summary = summarize_config(config, all_results[config.name])
        summaries.append(summary)

    # Print results
    print_summary_table(summaries)

    # Per-mineral breakdown for best and worst
    valid_summaries = [s for s in summaries if np.isfinite(s.mean_rmsep)]
    if valid_summaries:
        best = min(valid_summaries, key=lambda s: s.mean_rmsep)
        worst = max(valid_summaries, key=lambda s: s.mean_rmsep)
        print_mineral_breakdown(all_results, best.config.name, "BEST")
        print_mineral_breakdown(all_results, worst.config.name, "WORST")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_configs": len(configs),
            "n_spectra": len(spectra),
            "n_total_runs": total_runs,
            "total_elapsed_s": time.perf_counter() - t_start,
            "database": args.db,
            "data_dir": args.data_dir,
            "skip_voigt": args.skip_voigt,
            "factors": {
                "voigt_deconvolution": "Voigt deconvolution vs trapezoid integration",
                "boltzmann_fitting": "Huber M-estimation vs OLS (no outlier rejection)",
                "closure_mode": "ILR (Isometric Log-Ratio) vs standard normalization",
                "self_absorption": "CD-SB self-absorption correction vs disabled",
            },
        },
        "config_summaries": [s.to_dict() for s in summaries],
        "per_spectrum_results": {
            cfg_name: [r.to_dict() for r in results] for cfg_name, results in all_results.items()
        },
        "mineral_stoichiometric": {
            name: fracs for name, fracs in MINERAL_STOICHIOMETRIC.items() if fracs  # skip empties
        },
    }

    out_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
