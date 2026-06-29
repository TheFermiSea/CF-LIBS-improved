#!/usr/bin/env python3
"""Complete the line-broadening columns on the complete (ASD59) production DB.

Why
---
After the M5 gold-standard ASD59 ingest the ``lines`` table grew to 203 695
rows, but only the 28 683 lines carried over from the pre-reset DB had
``stark_w`` populated (14.1 %).  The STARK-B literature ingest
(``stark_w_source='stark_b'``, 693 anchor lines) and the legacy
λ²-scaling fill survived, but the bulk of the new ASD59 lines -- the higher
ionization stages (II/III, 96 k lines) and the long tail of minor species --
were left with NULL ``stark_w`` and only a per-species *placeholder*
``gamma_vdw_log`` / ``gamma_self_log`` (83 distinct values across 158 species;
constant within a species, i.e. not physical).

This script finishes the deliverable with **physically-correct, provenance-tagged
semi-empirical estimates** -- not a flat band-aid.  It NEVER touches the
``stark_w_source='stark_b'`` literature rows (the acceptance-test anchors) and
NEVER overwrites an already-populated ``stark_w`` (so the validated forward-model
behaviour on covered species is preserved); it only *fills* NULL ``stark_w``.
The two γ columns are inert metadata (the radiation forward model consumes only
``stark_w``; jitpipe carries ``line_gamma_vdw_log`` but the JIT core does not use
it), so they are recomputed for *every* line to replace the placeholder with a
single uniform, documented convention.

Physics & references
--------------------
All reference conditions: T = 10 000 K, n_e = 1e17 cm^-3 (the convention the
stored ``stark_w`` is tabulated at -- see ``cflibs/radiation/stark.py``).

1. Stark electron-impact FWHM ``stark_w`` (nm), NULL rows only, in priority:

   (a) ``konjevic_lambda_sq_scaled`` -- quadratic Stark λ²-scaling
       ``w(λ) = k_species · λ²`` where the per-species coefficient
       ``k = w_ref/λ_ref²`` comes from the *median of that species' STARK-B
       literature widths* when available, else from the Konjević 2002 /
       STARK-B critical-compilation reference table.  The λ² law is the
       standard quadratic-Stark scaling (Griem 1974; Konjević & Wiese 1990;
       STARK-B, Sahal-Bréchot et al. 2014; macroscopic form Δλ₀ = w_s·n_e).

   (b) ``interpolated`` (curated cross-ion) -- Konjević-2002 reference width
       for the opposite charge state, λ²-scaled (Lesage 1996 for a few).

   (c) ``interpolated`` (isoelectronic / charge-class) -- for species with no
       literature reference at all (mostly stages III+ and exotic neutrals),
       λ²-scaling from the *same element's* nearest reference charge state where
       available (scaled by the (Z_ref/Z)² isonuclear-sequence trend: widths
       decrease with increasing ionic charge -- Wiese & Konjević 1990
       regularities), else from the median width coefficient of that charge
       *class* (neutral / singly-ionized / multiply-ionized) over the Konjević
       2002 compilation.  This is the second sanctioned fallback ("λ²-scaling
       from a species/isoelectronic reference"); it is preferred over the pure
       Griem n*⁴ absolute formula (Griem 1968, Phys. Rev. 165, 258) because a
       single-anchor n*⁴ extrapolation over-predicts neutral widths by ~10× --
       the isoelectronic scaling keeps the absolute scale literature-anchored
       per charge class.

   (d) ``hydrogenic`` -- Griem 1974 hydrogenic approximation for the hydrogen
       isotopes (H/D/T) and the noble gases lacking any compilation (Ne/Kr/Xe).

   (e) ``lanthanide_default`` -- mild λ²-scaled flat γ_e for lanthanides with
       neither literature nor usable atomic data.

2. van der Waals ``gamma_vdw_log`` (ALL rows) -- Unsöld (1955) approximation.
   C₆ = 6.46e-34 · Δ⟨r²⟩ with the hydrogenic mean-square radius
   ⟨r²⟩ = (n*²/2Z²)[5n*² + 1 − 3l(l+1)] (l neglected; upper minus lower level),
   and the classical Unsöld broadening rate per perturber
   ``γ₆/N = 17 · v_rel^(3/5) · C₆^(2/5)`` (Unsöld 1955; Aller 1963; Gray 2005),
   v_rel = √(8kT/πμ).  Perturber = neutral hydrogen (the universal VALD/Kurucz
   line-list tabulation convention).  Stored as
   ``log10(γ₆/N / [rad·s⁻¹·cm³])`` at T = 10 000 K (typical ≈ −8).

3. self / resonance ``gamma_self_log`` (ALL rows):
   * Resonance lines (lower level = ground, E_i ≲ 0.05 eV, or is_resonance):
     **Ali-Griem (1965, 1966) resonance broadening** (NIST Atomic Spectroscopy
     compendium, §"Line Broadening"):
     Δλ_R = 8.6e-30·(g_i/g_k)^½·λ²·λ_r·f_r·N_i (Å, cm⁻³).  Converting the
     wavelength FWHM to an angular rate (γ = 2πc·Δλ/λ², the λ² cancels) gives the
     density-normalised coefficient
     ``γ_R/N = 1.62e-10 · (g_i/g_k)^½ · λ_r[Å] · f_r``  [rad·s⁻¹·cm³],
     with f_r the resonance-line absorption oscillator strength
     (f = 1.49919e-16·(g_k/g_i)·λ_Å²·A_ki).
   * Non-resonance lines: resonance broadening is undefined, so the documented
     convention is Unsöld vdW *self*-broadening (perturber = same-species
     ground-state neutral, C₆ scaled by α_self/α_H).  Stored the same way as (2).

The DB change is the artifact: run, VACUUM, commit ``ASD_da/libs_production.db``.

Usage
-----
    PYTHONPATH=$PWD python scripts/fill_broadening_complete_db.py \
        --db-path ASD_da/libs_production.db
    PYTHONPATH=$PWD python scripts/fill_broadening_complete_db.py \
        --db-path ASD_da/libs_production.db --dry-run
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RYD_EV = 13.605693  # Rydberg energy (eV)
HC_NM_EV = 1239.84193  # h·c in nm·eV
A0_CM = 5.29177210903e-9  # Bohr radius (cm)
C_ANG_S = 2.99792458e18  # speed of light (Å/s)
KB_ERG = 1.380649e-16  # Boltzmann constant (erg/K)
U_G = 1.66053906660e-24  # atomic mass unit (g)
ALPHA_H_CM3 = 0.666793e-24  # neutral hydrogen polarizability (cm^3)

T_REF_K = 10000.0
NE_REF_CM3 = 1.0e17

# Unsöld C6 constant: C6 = C6_UNSOLD * Δ<r^2>(a0^2)  (cgs cm^6 s^-1, H perturber)
C6_UNSOLD = 6.46e-34
# Classical Unsöld broadening-rate prefactor: γ6/N = VDW_K * v^(3/5) * C6^(2/5)
VDW_K = 17.0
# Ali-Griem resonance coefficient (NIST compendium, density-normalised, rad cm^3/s).
# γ_R/N = 2π·c·8.6e-30·(g_i/g_k)^½·λ_r[Å]·f_r  with c in Å/s.
ALI_GRIEM_K = 2.0 * math.pi * C_ANG_S * 8.6e-30  # ≈ 1.62e-10

STARK_W_MIN_PM = 0.3
STARK_W_MAX_PM = 250.0

# ---------------------------------------------------------------------------
# Konjević 2002 / STARK-B critical-compilation reference widths
#   (element, ion_stage): (w_ref_pm, lambda_ref_nm, alpha, d_over_w)
# (carried over verbatim from the validated populate_stark_widths.py)
# ---------------------------------------------------------------------------
REFERENCE_STARK_WIDTHS: dict[tuple[str, int], tuple[float, float, float, float]] = {
    ("Fe", 2): (18.0, 260.0, 0.08, 0.40),
    ("Fe", 1): (12.0, 380.0, 0.06, 0.35),
    ("Cr", 2): (22.0, 283.0, 0.09, 0.42),
    ("Cr", 1): (15.0, 425.0, 0.06, 0.38),
    ("Mn", 2): (32.0, 257.0, 0.10, 0.45),
    ("Mn", 1): (20.0, 403.0, 0.07, 0.40),
    ("Ti", 2): (18.0, 320.0, 0.07, 0.38),
    ("Ti", 1): (12.0, 500.0, 0.05, 0.32),
    ("Ca", 2): (8.0, 393.0, 0.05, 0.30),
    ("Ca", 1): (10.0, 423.0, 0.05, 0.30),
    ("Mg", 2): (6.0, 280.0, 0.04, 0.25),
    ("Mg", 1): (8.0, 285.0, 0.04, 0.25),
    ("Si", 1): (12.0, 288.0, 0.06, 0.35),
    ("Si", 2): (10.0, 413.0, 0.06, 0.35),
    ("Al", 1): (4.0, 396.0, 0.04, 0.25),
    ("Al", 2): (5.0, 358.0, 0.04, 0.25),
    ("Na", 1): (6.0, 589.0, 0.05, 0.30),
    ("K", 1): (32.0, 766.0, 0.10, 0.40),
    ("Cu", 1): (12.0, 324.0, 0.06, 0.32),
    ("Cu", 2): (15.0, 271.0, 0.07, 0.35),
    ("Ni", 1): (10.0, 341.0, 0.05, 0.30),
    ("Ni", 2): (14.0, 226.0, 0.07, 0.35),
    ("Co", 1): (12.0, 350.0, 0.06, 0.32),
    ("Co", 2): (15.0, 230.0, 0.07, 0.35),
    ("V", 1): (14.0, 410.0, 0.06, 0.32),
    ("V", 2): (18.0, 290.0, 0.08, 0.38),
    ("Mo", 1): (15.0, 380.0, 0.07, 0.35),
    ("W", 1): (10.0, 400.0, 0.06, 0.30),
    ("Zn", 1): (8.0, 213.0, 0.05, 0.28),
    ("Zn", 2): (12.0, 250.0, 0.06, 0.32),
    ("Pb", 1): (15.0, 405.0, 0.07, 0.35),
    ("Sn", 1): (12.0, 380.0, 0.06, 0.32),
    ("Ba", 2): (10.0, 455.0, 0.05, 0.30),
    ("Ba", 1): (12.0, 553.0, 0.05, 0.30),
    ("Sr", 2): (8.0, 408.0, 0.05, 0.28),
    ("Sr", 1): (10.0, 461.0, 0.05, 0.28),
    ("Li", 1): (5.0, 670.8, 0.04, 0.25),
    ("H", 1): (50.0, 656.3, 0.20, 0.10),
    ("He", 1): (15.0, 587.6, 0.08, 0.30),
    ("C", 1): (10.0, 247.9, 0.05, 0.30),
    ("C", 2): (12.0, 426.7, 0.06, 0.32),
    ("N", 1): (8.0, 411.0, 0.05, 0.28),
    ("N", 2): (10.0, 500.5, 0.05, 0.30),
    ("O", 1): (8.0, 777.4, 0.05, 0.28),
    ("O", 2): (10.0, 441.5, 0.05, 0.30),
    ("S", 1): (10.0, 469.5, 0.05, 0.30),
    ("Cl", 1): (8.0, 837.6, 0.05, 0.28),
    ("Ar", 1): (12.0, 696.5, 0.06, 0.32),
    ("F", 1): (8.0, 685.6, 0.05, 0.28),
    ("Br", 1): (10.0, 478.6, 0.05, 0.30),
    ("I", 1): (12.0, 540.7, 0.06, 0.32),
    ("As", 1): (10.0, 235.0, 0.05, 0.30),
    ("Sb", 1): (12.0, 287.8, 0.06, 0.32),
    ("Bi", 1): (15.0, 306.7, 0.07, 0.35),
    ("Hg", 1): (10.0, 253.7, 0.05, 0.30),
    ("Cd", 1): (12.0, 228.8, 0.06, 0.32),
    ("Ag", 1): (12.0, 328.1, 0.06, 0.32),
    ("Au", 1): (10.0, 242.8, 0.05, 0.30),
    ("Be", 1): (4.0, 234.9, 0.04, 0.25),
    ("B", 1): (6.0, 249.7, 0.04, 0.27),
    ("P", 1): (10.0, 213.6, 0.05, 0.30),
    ("Y", 2): (15.0, 360.0, 0.07, 0.35),
    ("Zr", 2): (12.0, 327.0, 0.06, 0.32),
    ("Nb", 2): (15.0, 405.0, 0.07, 0.35),
    ("La", 2): (12.0, 408.0, 0.06, 0.32),
    ("Ce", 2): (15.0, 418.0, 0.07, 0.35),
    ("Nd", 2): (15.0, 401.0, 0.07, 0.35),
    ("Sm", 2): (15.0, 442.0, 0.07, 0.35),
    ("Eu", 2): (15.0, 412.0, 0.07, 0.35),
    ("Gd", 2): (15.0, 442.0, 0.07, 0.35),
    ("Dy", 2): (15.0, 421.0, 0.07, 0.35),
    ("Er", 2): (15.0, 350.0, 0.07, 0.35),
    ("Lu", 2): (12.0, 350.0, 0.06, 0.32),
    ("Hf", 2): (12.0, 277.0, 0.06, 0.32),
    ("Ta", 2): (12.0, 271.0, 0.06, 0.32),
    ("Re", 2): (12.0, 346.0, 0.06, 0.32),
    ("Ir", 1): (12.0, 263.9, 0.06, 0.32),
    ("Pt", 1): (12.0, 265.9, 0.06, 0.32),
    ("Pd", 1): (12.0, 340.4, 0.06, 0.32),
    ("Rh", 1): (12.0, 343.4, 0.06, 0.32),
    ("Ru", 1): (12.0, 372.8, 0.06, 0.32),
    ("Ga", 1): (8.0, 417.2, 0.05, 0.28),
    ("Ge", 1): (10.0, 265.1, 0.05, 0.30),
    ("Tl", 1): (12.0, 351.9, 0.06, 0.32),
    ("In", 1): (10.0, 451.1, 0.05, 0.30),
    ("Cs", 1): (15.0, 852.1, 0.07, 0.35),
    ("Rb", 1): (12.0, 780.0, 0.06, 0.32),
    ("Tm", 2): (12.0, 384.8, 0.06, 0.32),
    ("Pr", 2): (15.0, 422.0, 0.07, 0.35),
    ("Ho", 2): (15.0, 405.4, 0.07, 0.35),
    ("Yb", 2): (12.0, 369.4, 0.06, 0.32),
    ("Sc", 2): (15.0, 363.0, 0.07, 0.35),
}

# Curated cross-ion references (Konjević 2002 / Lesage 1996) -> 'interpolated'.
INTERPOLATED_REFS: dict[tuple[str, int], tuple[float, float, float, float]] = {
    ("Ar", 2): (26.0, 480.0, 0.08, 0.40),
    ("S", 2): (15.0, 545.4, 0.06, 0.35),
    ("In", 2): (14.0, 256.0, 0.06, 0.32),
    ("La", 1): (15.0, 550.0, 0.07, 0.35),
    ("Hg", 2): (12.0, 194.2, 0.06, 0.32),
    ("Sc", 1): (16.0, 391.2, 0.07, 0.35),
    ("W", 2): (14.0, 200.6, 0.06, 0.32),
    ("Cl", 2): (12.0, 481.0, 0.06, 0.32),
    ("Ag", 2): (14.0, 232.0, 0.06, 0.32),
    ("Ta", 1): (15.0, 362.7, 0.07, 0.35),
    ("Y", 1): (14.0, 410.2, 0.06, 0.32),
    ("Eu", 1): (15.0, 459.4, 0.07, 0.35),
    ("Hf", 1): (14.0, 368.2, 0.06, 0.32),
    ("B", 2): (8.0, 412.2, 0.05, 0.30),
    ("Ir", 2): (14.0, 224.3, 0.06, 0.32),
    ("Pt", 2): (14.0, 224.6, 0.06, 0.32),
    ("Be", 2): (1.6, 313.0, 0.04, 0.25),
    ("Cd", 2): (15.0, 226.5, 0.07, 0.35),
    ("Sn", 2): (14.0, 215.0, 0.06, 0.32),
    ("Ce", 1): (15.0, 520.0, 0.07, 0.35),
    ("P", 2): (12.0, 604.0, 0.06, 0.32),
    ("Ge", 2): (12.0, 273.0, 0.06, 0.32),
    ("Mo", 2): (14.0, 290.0, 0.06, 0.32),
    ("Ga", 2): (12.0, 245.0, 0.06, 0.32),
    ("Pb", 2): (14.0, 220.4, 0.06, 0.32),
    ("Tl", 2): (14.0, 351.9, 0.06, 0.32),
    ("Zr", 1): (12.0, 360.0, 0.06, 0.32),
}

# Hydrogenic noble gases (Griem 1974) -> 'hydrogenic'; k tuned to Konjević Ar I.
HYDROGENIC_NOBLE: dict[tuple[str, int], float] = {
    ("Ne", 1): 0.10,
    ("Kr", 1): 0.40,
    ("Xe", 1): 0.50,
    ("Ne", 2): 0.20,
    ("Kr", 2): 0.60,
    ("Xe", 2): 0.75,
}
H_ALPHA_NM = 656.3
H_ALPHA_W_PM = 49.0

LANTHANIDES = {
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
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
}
# Lanthanide-class median λ²-coefficient (pm/nm²) from the Konjević 2002
# lanthanide-II compilation entries (≈13 pm at 400 nm) -- default for the
# handful of lanthanides (Pm, Tb...) absent from the table at every charge state.
LANTHANIDE_DEFAULT_K = 8.0e-5
HYDROGEN_ISOTOPES = {"H", "D", "T"}

# Ground-state polarizabilities (cm^3, CRC Handbook 95th ed.) for self-broadening.
ALPHA_SELF_CM3 = {
    "H": 0.667e-24,
    "He": 0.205e-24,
    "Li": 24.3e-24,
    "Be": 5.6e-24,
    "B": 3.0e-24,
    "C": 1.76e-24,
    "N": 1.10e-24,
    "O": 0.802e-24,
    "F": 0.557e-24,
    "Ne": 0.396e-24,
    "Na": 24.1e-24,
    "Mg": 10.6e-24,
    "Al": 6.8e-24,
    "Si": 5.4e-24,
    "P": 3.63e-24,
    "S": 2.90e-24,
    "Cl": 2.18e-24,
    "Ar": 1.64e-24,
    "K": 43.4e-24,
    "Ca": 22.8e-24,
    "Sc": 17.8e-24,
    "Ti": 14.6e-24,
    "V": 12.4e-24,
    "Cr": 11.6e-24,
    "Mn": 9.4e-24,
    "Fe": 8.4e-24,
    "Co": 7.5e-24,
    "Ni": 6.8e-24,
    "Cu": 6.7e-24,
    "Zn": 5.75e-24,
    "Ga": 8.12e-24,
    "Ge": 5.84e-24,
    "As": 4.31e-24,
    "Se": 3.77e-24,
    "Br": 3.05e-24,
    "Kr": 2.484e-24,
    "Rb": 47.4e-24,
    "Sr": 27.6e-24,
    "Y": 22.7e-24,
    "Zr": 17.9e-24,
    "Nb": 15.7e-24,
    "Mo": 12.8e-24,
    "Tc": 11.4e-24,
    "Ru": 9.6e-24,
    "Rh": 8.6e-24,
    "Pd": 4.8e-24,
    "Ag": 7.2e-24,
    "Cd": 7.36e-24,
    "In": 10.2e-24,
    "Sn": 7.7e-24,
    "Sb": 6.6e-24,
    "Te": 5.5e-24,
    "I": 5.35e-24,
    "Xe": 4.044e-24,
    "Cs": 59.3e-24,
    "Ba": 39.7e-24,
    "La": 31.1e-24,
    "Ce": 29.6e-24,
    "Pr": 28.2e-24,
    "Nd": 31.4e-24,
    "Sm": 28.3e-24,
    "Eu": 30.1e-24,
    "Gd": 23.5e-24,
    "Tb": 25.5e-24,
    "Dy": 24.5e-24,
    "Ho": 23.6e-24,
    "Er": 22.7e-24,
    "Tm": 21.8e-24,
    "Yb": 20.9e-24,
    "Lu": 21.9e-24,
    "Hf": 16.2e-24,
    "Ta": 13.1e-24,
    "W": 11.1e-24,
    "Re": 9.7e-24,
    "Os": 8.5e-24,
    "Ir": 7.6e-24,
    "Pt": 6.5e-24,
    "Au": 5.8e-24,
    "Hg": 5.02e-24,
    "Tl": 7.6e-24,
    "Pb": 6.98e-24,
    "Bi": 7.4e-24,
    "Th": 32.1e-24,
    "U": 27.4e-24,
}
ATOMIC_MASS_U = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Po": 209.0,
    "At": 210.0,
    "Rn": 222.0,
    "Fr": 223.0,
    "Ra": 226.0,
    "Ac": 227.0,
    "Th": 232.04,
    "Pa": 231.04,
    "U": 238.03,
    "Np": 237.0,
    "Pu": 244.0,
    "Am": 243.0,
    "Cm": 247.0,
    "Bk": 247.0,
    "Cf": 251.0,
    "Es": 252.0,
}
# Neutral first ionization energies (eV) -- fallback when species_physics lacks IP.
IP_NEUTRAL_EV = {
    "H": 13.598,
    "He": 24.587,
    "Li": 5.392,
    "Be": 9.323,
    "B": 8.298,
    "C": 11.260,
    "N": 14.534,
    "O": 13.618,
    "F": 17.423,
    "Ne": 21.565,
    "Na": 5.139,
    "Mg": 7.646,
    "Al": 5.986,
    "Si": 8.152,
    "P": 10.487,
    "S": 10.360,
    "Cl": 12.968,
    "Ar": 15.760,
    "K": 4.341,
    "Ca": 6.113,
    "Sc": 6.561,
    "Ti": 6.828,
    "V": 6.746,
    "Cr": 6.767,
    "Mn": 7.434,
    "Fe": 7.902,
    "Co": 7.881,
    "Ni": 7.640,
    "Cu": 7.726,
    "Zn": 9.394,
    "Ga": 5.999,
    "Ge": 7.900,
    "As": 9.789,
    "Se": 9.752,
    "Br": 11.814,
    "Kr": 14.000,
    "Rb": 4.177,
    "Sr": 5.695,
    "Y": 6.217,
    "Zr": 6.634,
    "Nb": 6.759,
    "Mo": 7.092,
    "Tc": 7.119,
    "Ru": 7.361,
    "Rh": 7.459,
    "Pd": 8.337,
    "Ag": 7.576,
    "Cd": 8.994,
    "In": 5.786,
    "Sn": 7.344,
    "Sb": 8.609,
    "Te": 9.010,
    "I": 10.451,
    "Xe": 12.130,
    "Cs": 3.894,
    "Ba": 5.212,
    "La": 5.577,
    "Ce": 5.539,
    "Pr": 5.473,
    "Nd": 5.525,
    "Pm": 5.55,
    "Sm": 5.644,
    "Eu": 5.670,
    "Gd": 6.150,
    "Tb": 5.864,
    "Dy": 5.939,
    "Ho": 6.022,
    "Er": 6.108,
    "Tm": 6.184,
    "Yb": 6.254,
    "Lu": 5.426,
    "Hf": 6.825,
    "Ta": 7.550,
    "W": 7.864,
    "Re": 7.834,
    "Os": 8.438,
    "Ir": 8.967,
    "Pt": 8.959,
    "Au": 9.226,
    "Hg": 10.438,
    "Tl": 6.108,
    "Pb": 7.417,
    "Bi": 7.286,
    "Po": 8.414,
    "At": 9.318,
    "Rn": 10.748,
    "Fr": 4.073,
    "Ra": 5.278,
    "Ac": 5.380,
    "Th": 6.307,
    "Pa": 5.89,
    "U": 6.194,
    "Np": 6.266,
    "Pu": 6.026,
    "Am": 5.974,
    "Cm": 5.991,
    "Bk": 6.198,
    "Cf": 6.282,
    "Es": 6.368,
}


def base_element(elem: str) -> str:
    """Strip an isotope prefix (e.g. '12C' -> 'C', '198Hg' -> 'Hg'); D/T -> H."""
    s = elem.lstrip("0123456789")
    if s in ("D", "T"):
        return "H"
    return s


def n_eff_upper(ip_ev: float, e_upper_ev: float, sp_num: int) -> float:
    """Effective principal quantum number of the upper level (clamped)."""
    de = max(ip_ev - e_upper_ev, 0.5)
    n2 = RYD_EV * sp_num * sp_num / de
    return max(min(math.sqrt(n2), 20.0), 1.0)


def mean_sq_radius_a0(n_star: float, sp_num: int) -> float:
    """Hydrogenic <r^2> in a0^2, l neglected (Unsöld): (n*^2/2Z^2)(5n*^2+1)."""
    z2 = float(sp_num * sp_num)
    return (n_star * n_star) / (2.0 * z2) * (5.0 * n_star * n_star + 1.0)


def log_gamma_vdw_perN(c6_cgs: float, mass_a_u: float, mass_p_u: float) -> float:
    """log10(γ6/N / [rad·s⁻¹·cm³]) = log10(17·v_rel^0.6·C6^0.4) at T_REF."""
    c6 = max(c6_cgs, 1e-50)
    mu_g = (mass_a_u * mass_p_u) / (mass_a_u + mass_p_u) * U_G
    v_rel = math.sqrt(8.0 * KB_ERG * T_REF_K / (math.pi * mu_g))  # cm/s
    gamma = VDW_K * v_rel**0.6 * c6**0.4
    return math.log10(max(gamma, 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db-path", default="ASD_da/libs_production.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 1
    print(f"DB: {db_path}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}\n")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # IP per (element, sp_num) from species_physics (preferred for high stages).
    ip_db: dict[tuple[str, int], float] = {}
    mass_db: dict[str, float] = {}
    for el, sp, ip, am in cur.execute(
        "SELECT element, sp_num, ip_ev, atomic_mass FROM species_physics"
    ):
        if ip is not None:
            ip_db[(el, int(sp))] = float(ip)
        if am is not None and el not in mass_db:
            mass_db[el] = float(am)

    def ip_for(base: str, sp: int, raw_elem: str) -> float | None:
        for key in ((raw_elem, sp), (base, sp)):
            if key in ip_db:
                return ip_db[key]
        # Fallback: neutral IP only meaningful for sp==1.
        if sp == 1 and base in IP_NEUTRAL_EV:
            return IP_NEUTRAL_EV[base]
        return None

    def mass_for(base: str, raw_elem: str) -> float:
        return mass_db.get(raw_elem) or mass_db.get(base) or ATOMIC_MASS_U.get(base, 50.0)

    # Per-species λ²-scaling coefficient k (pm/nm^2): prefer STARK-B median.
    species_k: dict[tuple[str, int], tuple[float, float, float]] = {}
    rows = cur.execute(
        "SELECT element, sp_num, wavelength_nm, stark_w, stark_alpha FROM lines "
        "WHERE stark_w_source='stark_b' AND stark_w IS NOT NULL "
        "AND wavelength_nm IS NOT NULL AND wavelength_nm>0"
    ).fetchall()
    sb: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for el, sp, wl, sw, sa in rows:
        sb.setdefault((base_element(el), int(sp)), []).append(
            (sw * 1e3 / (wl * wl), sa if sa is not None else 0.07)
        )
    for key, vals in sb.items():
        ks = np.array([v[0] for v in vals])
        al = np.array([v[1] for v in vals])
        species_k[key] = (float(np.median(ks)), float(np.median(al)), 0.35)
    for (el, sp), (w_ref, lam_ref, alpha, dw) in REFERENCE_STARK_WIDTHS.items():
        species_k.setdefault((el, sp), (w_ref / (lam_ref * lam_ref), alpha, dw))
    interp_k = {
        (el, sp): (w_ref / (lam_ref * lam_ref), alpha, dw)
        for (el, sp), (w_ref, lam_ref, alpha, dw) in INTERPOLATED_REFS.items()
    }

    # Same-element charge-state reference map and per-charge-class default
    # coefficients (median k over the Konjević 2002 compilation), for the
    # isoelectronic / charge-class λ²-scaling fallback.
    element_refs: dict[str, dict[int, float]] = {}
    class_k: dict[int, list[float]] = {}
    for (el, sp), (k, _a, _d) in species_k.items():
        element_refs.setdefault(el, {})[sp] = k
        class_k.setdefault(min(sp, 3), []).append(k)
    k_class = {c: float(np.median(v)) for c, v in class_k.items()}
    k_neutral = k_class.get(1, 8.0e-5)
    k_ion = k_class.get(2, 2.0e-4)

    def isoelectronic_k(el: str, sp: int) -> float:
        """λ²-coefficient (pm/nm²) for a species with no direct reference.

        Prefer the same element's nearest reference charge state, scaled by the
        (Z_ref/Z)² isonuclear-sequence trend (widths shrink with charge); else
        fall back to the charge-class median.
        """
        refs = element_refs.get(el)
        if refs:
            sp_ref = min(refs, key=lambda s: abs(s - sp))
            return refs[sp_ref] * (sp_ref * sp_ref) / float(sp * sp)
        if sp <= 1:
            return k_neutral
        # singly-ionized baseline, narrowing for higher stages.
        return k_ion * (4.0 / float(sp * sp))

    # ----- Stark width: fill NULL stark_w rows only -----------------------
    null_rows = cur.execute(
        "SELECT id, element, sp_num, wavelength_nm, ei_ev, ek_ev "
        "FROM lines WHERE stark_w IS NULL AND wavelength_nm IS NOT NULL "
        "AND wavelength_nm>0"
    ).fetchall()

    stark_updates: list[tuple] = []
    counts = {
        "konjevic_lambda_sq_scaled": 0,
        "interpolated": 0,
        "hydrogenic": 0,
        "lanthanide_default": 0,
    }
    bucket_w: dict[str, list[float]] = {k: [] for k in counts}
    n_stark_skip = 0

    for line_id, elem, sp, wl, ei, ek in null_rows:
        sp = int(sp)
        base = base_element(elem)
        w_pm: float | None = None
        alpha = 0.07
        dw = 0.30
        src = None

        if base in HYDROGEN_ISOTOPES or (base, sp) in HYDROGENIC_NOBLE:
            if base in HYDROGEN_ISOTOPES:
                w_ref, lam_ref = REFERENCE_STARK_WIDTHS[("H", 1)][:2]
                w_pm = w_ref * (wl / lam_ref) ** 2
                alpha, dw = 0.20, 0.10
            else:
                k = HYDROGENIC_NOBLE[(base, sp)]
                w_pm = k * H_ALPHA_W_PM * (wl / H_ALPHA_NM) ** 2
                alpha, dw = 0.07, 0.30
            src = "hydrogenic"
        elif (base, sp) in species_k:
            k, alpha, dw = species_k[(base, sp)]
            w_pm = k * wl * wl
            src = "konjevic_lambda_sq_scaled"
        elif (base, sp) in interp_k:
            k, alpha, dw = interp_k[(base, sp)]
            w_pm = k * wl * wl
            src = "interpolated"
        elif base in LANTHANIDES and base not in element_refs:
            # Lanthanide with no reference at any charge state.
            w_pm = LANTHANIDE_DEFAULT_K * wl * wl
            alpha, dw = 0.07, 0.40
            src = "lanthanide_default"
        else:
            # Isoelectronic / charge-class λ²-scaling (literature-anchored).
            w_pm = isoelectronic_k(base, max(sp, 1)) * wl * wl
            alpha, dw = 0.07, 0.30
            src = "interpolated"

        if w_pm is None or not math.isfinite(w_pm) or w_pm <= 0:
            n_stark_skip += 1
            continue
        w_pm = min(max(w_pm, STARK_W_MIN_PM), STARK_W_MAX_PM)
        w_nm = w_pm * 1e-3
        shift_nm = w_nm * dw
        counts[src] += 1
        bucket_w[src].append(w_pm)
        stark_updates.append((w_nm, alpha, shift_nm, src, line_id))

    # ----- vdW + self: recompute for ALL lines (replace placeholders) -----
    all_rows = cur.execute(
        "SELECT id, element, sp_num, wavelength_nm, ei_ev, ek_ev, gi, gk, aki, "
        "log_gf, osc_str, is_resonance FROM lines "
        "WHERE wavelength_nm IS NOT NULL AND wavelength_nm>0"
    ).fetchall()

    gamma_updates: list[tuple] = []
    n_vdw = n_self = n_resonance = n_gamma_skip = 0

    for line_id, elem, sp, wl, ei, ek, gi, gk, aki, log_gf, osc_str, is_res in all_rows:
        sp = int(sp)
        base = base_element(elem)
        ip = ip_for(base, sp, elem)
        if ip is None:
            n_gamma_skip += 1
            continue
        e_up = (
            float(ek)
            if ek is not None
            else ((float(ei) if ei is not None else 0.0) + HC_NM_EV / wl)
        )
        e_lo = float(ei) if ei is not None else 0.0
        n_up = n_eff_upper(ip, e_up, sp)
        n_lo = n_eff_upper(ip, min(e_lo, ip - 0.5), sp)
        d_r2 = max(mean_sq_radius_a0(n_up, sp) - mean_sq_radius_a0(n_lo, sp), 1e-3)
        mass_a = mass_for(base, elem)

        # van der Waals (hydrogen perturber).
        c6_h = C6_UNSOLD * d_r2
        vdw_log = log_gamma_vdw_perN(c6_h, mass_a, 1.008)

        # Self / resonance broadening.
        is_resonance = (is_res == 1) or (ei is not None and float(ei) < 0.05)
        f_res = None
        if is_resonance and gi and gk and gi > 0 and gk > 0:
            lam_ang = wl * 10.0
            if aki is not None and aki > 0:
                f_res = 1.49919e-16 * (gk / gi) * lam_ang * lam_ang * float(aki)
            elif log_gf is not None:
                f_res = 10.0 ** float(log_gf) / gi
            elif osc_str is not None and osc_str > 0:
                f_res = float(osc_str)
        if f_res is not None and f_res > 0:
            lam_ang = wl * 10.0
            gamma_self = ALI_GRIEM_K * math.sqrt(gi / gk) * lam_ang * f_res
            self_log = math.log10(max(gamma_self, 1e-30))
            n_resonance += 1
        else:
            alpha_self = ALPHA_SELF_CM3.get(base, 5.0e-24)
            c6_self = C6_UNSOLD * d_r2 * (alpha_self / ALPHA_H_CM3)
            self_log = log_gamma_vdw_perN(c6_self, mass_a, mass_a)

        if not (math.isfinite(vdw_log) and math.isfinite(self_log)):
            n_gamma_skip += 1
            continue
        gamma_updates.append((vdw_log, self_log, line_id))
        n_vdw += 1
        n_self += 1

    # ----- write -----------------------------------------------------------
    cur.execute("SELECT COUNT(*) FROM lines")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM lines WHERE stark_w IS NOT NULL")
    n_sw_before = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM lines WHERE gamma_vdw_log IS NOT NULL AND NOT "
        "(wavelength_nm IS NOT NULL AND wavelength_nm>0)"
    )
    gamma_nonupdatable_existing = cur.fetchone()[0]

    if not args.dry_run:
        cur.executemany(
            "UPDATE lines SET stark_w=?, stark_alpha=?, stark_shift=?, "
            "stark_w_source=? WHERE id=?",
            stark_updates,
        )
        cur.executemany(
            "UPDATE lines SET gamma_vdw_log=?, gamma_self_log=? WHERE id=?",
            gamma_updates,
        )
        conn.commit()

    if args.dry_run:
        # Projected (exact: we fill only NULL stark_w; recompute all gamma).
        n_sw = n_sw_before + len(stark_updates)
        n_vd = n_se = len(gamma_updates) + gamma_nonupdatable_existing
        cur.execute(
            "SELECT stark_w_source, COUNT(*) FROM lines "
            "WHERE stark_w_source='stark_b' GROUP BY stark_w_source"
        )
        breakdown = cur.fetchall() + [("(projected fills below)", "")]
        for prov in counts:
            breakdown.append((prov, counts[prov]))
    else:
        cur.execute("SELECT COUNT(*) FROM lines WHERE stark_w IS NOT NULL")
        n_sw = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM lines WHERE gamma_vdw_log IS NOT NULL")
        n_vd = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM lines WHERE gamma_self_log IS NOT NULL")
        n_se = cur.fetchone()[0]
        cur.execute(
            "SELECT stark_w_source, COUNT(*) FROM lines GROUP BY stark_w_source " "ORDER BY 2 DESC"
        )
        breakdown = cur.fetchall()

    print("=" * 78)
    print("BROADENING FILL SUMMARY")
    print("=" * 78)
    print(f"total lines:                 {total}")
    print(f"NULL stark_w considered:     {len(null_rows)}")
    print("stark_w fill by provenance (NULL rows only):")
    for prov in counts:
        med = float(np.median(bucket_w[prov])) if bucket_w[prov] else float("nan")
        print(f"  {prov:28s} {counts[prov]:>8d}   median {med:6.1f} pm")
    print(f"  stark_w skipped (no fill):   {n_stark_skip}")
    print(
        f"gamma recomputed (all lines): vdw={n_vdw} self={n_self} "
        f"(resonance via Ali-Griem={n_resonance}, skipped={n_gamma_skip})"
    )
    print()
    print(f"{'COVERAGE (after)':24s}{'count':>10s}{'pct':>8s}{'target':>9s}")
    for label, n, tgt in (
        ("stark_w", n_sw, 95.0),
        ("gamma_vdw_log", n_vd, 80.0),
        ("gamma_self_log", n_se, 80.0),
    ):
        pct = 100.0 * n / max(total, 1)
        flag = "OK" if pct >= tgt else "FAIL"
        print(f"{label:24s}{n:>10d}{pct:>7.1f}%{tgt:>7.0f}%  {flag}")
    print("\nstark_w_source breakdown (whole table):")
    for src, n in breakdown:
        print(f"  {str(src):28s} {n}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
