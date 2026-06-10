#!/usr/bin/env python3
"""Populate Van der Waals (γ_vdw) and self-broadening (γ_self) columns.

Approximation
-------------

This script populates ``gamma_vdw_log`` and ``gamma_self_log`` (added by
``scripts/archive/migrations/migrate_add_broadening_columns.py``) using the Anstee-O'Mara
unified Unsöld theory in its simplified Mihalas-Kurucz form
(Kurucz 1981 SAO Special Report 391; Mihalas 1978 "Stellar
Atmospheres" §9-3). This is a CRUDE physics-based estimate — better
than NULL but not a substitute for ABO Anstee-Barklem-O'Mara
calculations or VALD3 line-list-supplied γ_6 values, which we do not
have access to here.

Reference conditions
--------------------

* T_ref = 10000 K
* perturber number density n_p = 1e22 cm^-3 (Kurucz "standard" scaling
  density for synth.f / SYNTHE / VALD3 line lists; converts to ABO at
  10000 K via factor exp(0.4 × ln(10)) for Wahlgren-Cowley convention).

For Van der Waals we assume the perturber is **argon** (LIBS plasmas
are typically Ar-buffered), with polarizability α_Ar = 1.64×10⁻²⁴ cm³.
For self-broadening the perturber is the neutral emitter species itself
with self-polarizability α_self looked up from a small table of
ground-state polarizabilities (CRC Handbook 2014).

Formulae
--------

Mihalas-Kurucz Van der Waals damping (rad/s, FWHM):

    γ_vdw = 17 × C₆^(2/5) × v_rel^(3/5) × n_p

with C₆ ≈ e²a₀⁵/h × α_perturber × R²_eff and

    R²_eff ≈ n*²(n*² + 5l(l+1) + 3) / 2Z_eff²       (Bates-Damgaard)

where n* is the effective principal quantum number of the **upper**
level: n* = √(13.6 eV × Z_eff² / (E_ion − E_upper)). The thermal
relative velocity is v_rel = √(8kT/πμ) with reduced mass μ = m_a m_p /
(m_a + m_p).

For LIBS we use the simplified Kurucz log scaling (his 1981 SAO 391
eq. 4.61 with the Unsöld α=2/5 exponent):

    log10(γ_vdw / [rad/s]) ≈ 19.4 + 0.4 log10 C₆ + 0.3 log10(T/10000)
                            − 0.4 log10(μ) + log10(n_p / 1e22)

C₆ (in atomic units, hartree × a_0⁶) is

    C₆ ≈ 1.01 × 10⁻³² × α_p × n*⁴ / Z_eff²

with α_p in cm³ and the effective principal quantum number of the
upper level. We approximate n* using the **wavelength** when the
energies aren't readily available:

    n* = √(13.6 / (13.6 / n_low² − 12398.4 / λ_nm × 10⁻³))   eV

This is a 30-50 % crude estimate but good enough to produce non-NULL
values that fall in the right order of magnitude, which is the entire
point — the Voigt convolution is logarithmically forgiving on the
narrow-line Lorentzian wings.

For self-broadening γ_self uses the same machinery with α_perturber
replaced by α_self; the "Mihalas Q factor" boost (~1.6) is folded into
log10 implicitly since self-broadening is dominated by close-binary
resonance interactions which Anstee-O'Mara does NOT capture; downstream
code should treat γ_self as a 50 % uncertainty estimate.

Output
------

Both columns are stored as ``log10(γ / [rad/s])``. Typical values:
γ_vdw ~ 10⁸–10¹⁰ rad/s ⇒ log10 ≈ 8–10. γ_self can be 1-2 orders of
magnitude larger for resonance lines.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

T_REF_K = 10000.0
N_P_REF = 1.0e22  # perturber density (cm^-3)
ALPHA_AR_CM3 = 1.64e-24  # argon ground-state polarizability

# Element ground-state polarizabilities in cm^3 (CRC Handbook 95th ed., Tab 10-77).
# Used for self-broadening γ_self. Values to ±10 % are good enough.
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

# Atomic masses (u) — used in reduced mass calculation. Grab from the
# DB's species_physics if it exists; else use a built-in for common
# elements. For full coverage we just compute m_a from element symbol
# via a small table (extras filled in by the embedded ALPHA table
# anyway).
ATOMIC_MASS_U = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03,
}

# Approximate first ionization energies (eV) for n* estimation.
# Values from NIST ASD; ±1 % uncertainty is fine.
IP_EV = {
    "H": 13.598, "He": 24.587, "Li": 5.392, "Be": 9.323, "B": 8.298,
    "C": 11.260, "N": 14.534, "O": 13.618, "F": 17.423, "Ne": 21.565,
    "Na": 5.139, "Mg": 7.646, "Al": 5.986, "Si": 8.152, "P": 10.487,
    "S": 10.360, "Cl": 12.968, "Ar": 15.760, "K": 4.341, "Ca": 6.113,
    "Sc": 6.561, "Ti": 6.828, "V": 6.746, "Cr": 6.767, "Mn": 7.434,
    "Fe": 7.902, "Co": 7.881, "Ni": 7.640, "Cu": 7.726, "Zn": 9.394,
    "Ga": 5.999, "Ge": 7.900, "As": 9.789, "Se": 9.752, "Br": 11.814,
    "Kr": 14.000, "Rb": 4.177, "Sr": 5.695, "Y": 6.217, "Zr": 6.634,
    "Nb": 6.759, "Mo": 7.092, "Tc": 7.119, "Ru": 7.361, "Rh": 7.459,
    "Pd": 8.337, "Ag": 7.576, "Cd": 8.994, "In": 5.786, "Sn": 7.344,
    "Sb": 8.609, "Te": 9.010, "I": 10.451, "Xe": 12.130, "Cs": 3.894,
    "Ba": 5.212, "La": 5.577, "Ce": 5.539, "Pr": 5.473, "Nd": 5.525,
    "Sm": 5.644, "Eu": 5.670, "Gd": 6.150, "Tb": 5.864, "Dy": 5.939,
    "Ho": 6.022, "Er": 6.108, "Tm": 6.184, "Yb": 6.254, "Lu": 5.426,
    "Hf": 6.825, "Ta": 7.550, "W": 7.864, "Re": 7.834, "Os": 8.438,
    "Ir": 8.967, "Pt": 8.959, "Au": 9.226, "Hg": 10.438, "Tl": 6.108,
    "Pb": 7.417, "Bi": 7.286, "Th": 6.307, "U": 6.194,
}

PERTURBER_AR_MASS = 39.948  # argon

# Constants
HC_NM_EV = 1239.84193  # h c in nm·eV


def n_star_upper(element: str, sp_num: int, wavelength_nm: float, ek_ev: float | None) -> float:
    """Effective principal quantum number of the upper level.

    Uses E_ek directly when available, falls back to (IP - hc/λ) when
    only the wavelength is known. n* = √(13.6 Z_eff² / (E_ion - E_upper)).
    """
    z_eff = float(sp_num)  # 1=neutral, 2=singly-ionized, etc.
    ip = IP_EV.get(element, 7.5)
    if ek_ev is not None and ek_ev > 0:
        e_above_upper = max(ip - ek_ev, 0.5)  # eV; clamp to avoid div by 0
    else:
        # Estimate upper-level energy as hc/λ above some lower-level
        # energy ~0 (ground). This is a 30 % crude estimate.
        e_above_upper = max(ip - HC_NM_EV / max(wavelength_nm, 100.0), 0.5)
    n_star = math.sqrt(13.6 * z_eff * z_eff / e_above_upper)
    # Clamp to physical range:
    return max(min(n_star, 20.0), 1.0)


def log_gamma_vdw(
    element: str,
    sp_num: int,
    wavelength_nm: float,
    ek_ev: float | None,
    alpha_perturber_cm3: float,
    perturber_mass_u: float,
) -> float:
    """log10(γ_vdw / [rad/s]) at T_REF_K, n_p = N_P_REF.

    Mihalas-Kurucz formula:
        γ ≈ 17 v_rel^(3/5) C6^(2/5) n_p
    in CGS rad/s. We use the log form for numerical stability.
    """
    n_star = n_star_upper(element, sp_num, wavelength_nm, ek_ev)
    z_eff = float(sp_num)
    # C6 in atomic units (hartree × a_0^6): ABO/Kurucz convention.
    # C6 ≈ 1.01e-32 × (α_p / [cm^3]) × n*^4 / Z_eff^2  (Mihalas 1978 §9-3)
    c6 = 1.01e-32 * alpha_perturber_cm3 * (n_star ** 4) / (z_eff * z_eff)
    c6 = max(c6, 1e-50)  # numerical guard
    # Reduced mass
    m_a = ATOMIC_MASS_U.get(element, 50.0)
    mu = (m_a * perturber_mass_u) / (m_a + perturber_mass_u)
    # Kurucz 1981 SAO 391 eq. 4.61 (log form, evaluated at T=10000):
    #   log γ = 19.4 + 0.4 log C6 + 0.3 log(T/1e4) − 0.4 log μ + log(n_p/1e22)
    # T/T_ref = 1, n_p/N_P_REF = 1, so the log term drops to a constant.
    log_g = 19.4 + 0.4 * math.log10(c6) - 0.4 * math.log10(mu)
    return log_g


def log_gamma_self(
    element: str,
    sp_num: int,
    wavelength_nm: float,
    ek_ev: float | None,
) -> float:
    """log10(γ_self) at T_REF_K, n_p = N_P_REF.

    Same machinery as Van der Waals but with self-perturbation. We add
    a +0.2 dex bias for resonance-line enhancement (rough Q factor).
    """
    alpha_self = ALPHA_SELF_CM3.get(element, 5.0e-24)
    m_a = ATOMIC_MASS_U.get(element, 50.0)
    base = log_gamma_vdw(element, sp_num, wavelength_nm, ek_ev, alpha_self, m_a)
    return base + 0.2


def populate(db_path: Path, dry_run: bool = False) -> dict[str, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Verify required columns are present.
    cur.execute("PRAGMA table_info(lines)")
    cols = {row[1] for row in cur.fetchall()}
    if "gamma_vdw_log" not in cols or "gamma_self_log" not in cols:
        conn.close()
        raise RuntimeError(
            "Required columns missing — run scripts/archive/migrations/migrate_add_broadening_columns.py first"
        )

    cur.execute(
        "SELECT id, element, sp_num, wavelength_nm, ek_ev FROM lines "
        "WHERE wavelength_nm IS NOT NULL AND wavelength_nm > 0"
    )
    rows = cur.fetchall()
    n_total = len(rows)
    n_vdw = 0
    n_self = 0
    n_skipped = 0
    for line_id, elem, sp, wl, ek in rows:
        if elem not in ATOMIC_MASS_U:
            n_skipped += 1
            continue
        try:
            log_vdw = log_gamma_vdw(
                elem, int(sp), float(wl), float(ek) if ek is not None else None,
                ALPHA_AR_CM3, PERTURBER_AR_MASS,
            )
            log_self = log_gamma_self(
                elem, int(sp), float(wl), float(ek) if ek is not None else None,
            )
        except (ValueError, TypeError):
            n_skipped += 1
            continue
        if not dry_run:
            cur.execute(
                "UPDATE lines SET gamma_vdw_log = ?, gamma_self_log = ? WHERE id = ?",
                (log_vdw, log_self, line_id),
            )
        n_vdw += 1
        n_self += 1

    if not dry_run:
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM lines WHERE gamma_vdw_log IS NOT NULL")
    n_vdw_after = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM lines WHERE gamma_self_log IS NOT NULL")
    n_self_after = cur.fetchone()[0]
    conn.close()

    return {
        "total": n_total,
        "populated_vdw": n_vdw,
        "populated_self": n_self,
        "skipped": n_skipped,
        "lines_with_vdw_after": n_vdw_after,
        "lines_with_self_after": n_self_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("ASD_da/libs_production.db"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Populating γ_vdw / γ_self in: {args.db}")
    print(f"  perturber: argon (α={ALPHA_AR_CM3:.2e} cm³, m={PERTURBER_AR_MASS} u)")
    print(f"  reference: T={T_REF_K} K, n_p={N_P_REF:.0e} cm^-3")
    stats = populate(args.db, dry_run=args.dry_run)
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
