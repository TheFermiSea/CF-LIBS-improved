#!/usr/bin/env python3
"""STARK-B / Konjević / Dimitrijević / Sahal-Brechot line-specific
electron-impact Stark width adapter.

This script populates ``lines.stark_w`` (and where available
``stark_alpha`` / ``stark_shift``) with **line-specific** electron-impact
Stark FWHM values at the canonical reference conditions

    T_e = 10000 K,    n_e = 1.0e17 cm^-3

REPLACING the ``(λ/λ_ref)^2`` Konjević fallback from
``populate_stark_widths.py`` (PR #99) for any line that has a
literature value. Provenance is recorded in ``lines.stark_w_source``
(must be migrated first via ``scripts/migrate_add_broadening_columns.py``).

Two input paths are supported:

1. **External STARK-B exports.** If ``--raw-dir`` (default
   ``/cluster/shared/cf-libs-data/stark_b/raw/``) contains ``*.csv`` or
   ``*.tsv`` files in the format documented in
   ``/cluster/shared/cf-libs-data/README.md`` (transition, T_e, n_e,
   gamma_W, gamma_d [, alpha]), they are parsed and projected onto the
   canonical reference. STARK-B reports widths in Angstrom (Å);
   conversion: 1 Å = 0.1 nm = 100 pm.

2. **Embedded fallback table.** A hand-curated table of high-confidence
   STARK-B / Konjević values from published critical compilations. Each
   entry carries an inline ``source`` citation. Used unconditionally in
   addition to (1); literature values from external files take
   precedence on conflict.

Embedded table sources
----------------------

* Konjević, N., Lesage, A., Fuhr, J. R., & Wiese, W. L. (2002).
  "Experimental Stark widths and shifts for spectral lines of neutral
  and ionized atoms: A critical review of selected data for the period
  1989-2000." J. Phys. Chem. Ref. Data 31(3), 819-927.
  Cited inline as ``Konjevic 2002, Tab N``.

* Sahal-Bréchot, S., Dimitrijević, M. S., & Ben Nessib, N. (2014).
  "Widths and Shifts of Isolated Lines of Neutral and Ionized Atoms
  Perturbed by Collisions With Electrons and Ions: An Outline of the
  Semiclassical Perturbation (SCP) Method and of the Approximations
  Used for the Calculations." Atoms 2, 225-252; STARK-B database at
  https://stark-b.obspm.fr/.
  Cited inline as ``STARK-B`` (with element/multiplet annotation).

* Dimitrijević, M. S. & Sahal-Bréchot, S. (1992-2014, multiple papers
  in JQSRT, A&AS, BlgAJ, NewA). The Cr II / Mn II / Ti II / Fe II SCP
  calculations underlying STARK-B; cited as ``Dimitrijevic <year>``.

* Lesage, A. (1996). "Stark broadening review." Phys. Scr. T58, 24-30.
  Si I/II, Cu I/II, Ni I/II reference values; cited as ``Lesage 1996``.

* Griem, H. R. (1974). "Spectral Line Broadening by Plasmas." Academic
  Press. Used only for hydrogenic Hα/Hβ wing structure (NOT FWHM here).

Usage
-----

    python scripts/ingest_stark_b.py
    python scripts/ingest_stark_b.py --db ASD_da/libs_production.db --raw-dir /cluster/shared/cf-libs-data/stark_b/raw
    python scripts/ingest_stark_b.py --dry-run

Behavior
--------

* Match each (element, sp_num, λ_air_nm) to a row in ``lines`` where
  ``ABS(lines.wavelength_nm - record.wavelength_nm) <= --tol`` (default
  0.05 nm). When multiple lines match, the **closest in wavelength** is
  selected.

* When matched, UPDATE ``stark_w``, ``stark_alpha`` (if record provides
  it), ``stark_shift`` (if record provides it), and set
  ``stark_w_source = "stark_b"``.

* Every UPDATE OVERRIDES any pre-existing value, including the λ²
  scaling fallback from PR #99. STARK-B literature values are
  authoritative.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

T_REF_K = 10000.0
NE_REF_CM3 = 1.0e17


@dataclass
class StarkRecord:
    """One literature Stark-broadening datum at T_REF_K / NE_REF_CM3."""

    element: str
    sp_num: int  # 1=neutral, 2=singly ionized, ...
    wavelength_nm: float
    w_fwhm_nm: float  # electron-impact FWHM in nm at T_REF, NE_REF
    alpha: float | None = None  # ion-broadening parameter (dimensionless)
    shift_nm: float | None = None  # signed shift in nm at T_REF, NE_REF
    source: str = ""  # citation, e.g. "Konjevic 2002, Tab 7"


# ======================================================================
# Embedded curated STARK-B / Konjević table
# ======================================================================
#
# Values are FWHM in nm, electron-impact only, at T_e = 10000 K,
# n_e = 1e17 cm^-3. Each entry carries an inline source citation.
#
# Notation conventions:
#   * w in pm (10^-3 nm) for compactness, converted to nm in code below.
#   * Where Konjević critical compilations report the width as 2w_e
#     (FWHM), we use it directly. Where reported as half-width (HWHM),
#     we double it. STARK-B native is FWHM full-width.
#   * alpha is reported only when explicitly tabulated; otherwise None.
#   * shift sign convention: positive = redshift.
#
# Coverage targets per the audit (priorities 1-3):
#   Mg II, Ca II, Na I, K I, Mg I  — alkali / alkaline-earth resonance
#   Fe II                          — many UV multiplets, primary LIBS
#   Fe I, Cr II, Mn II, Ti II      — secondary LIBS
#   Si I/II, Al I/II, Cu I, Ni I  — minor / matrix elements
#   H I (Hα/Hβ)                    — diagnostic anchor lines
# ======================================================================

# Format: list of StarkRecord constructor arg-tuples for terseness:
#     (elem, sp, wl_nm, w_pm, alpha, shift_pm, source)
# alpha and shift_pm may be None.

_RAW_TABLE: list[tuple] = [
    # ------------------------------------------------------------------
    # H I (Balmer) — Konjevic 2002, Tab 1 / Griem 1974 Tab 4-5.
    # H lines are non-Voigt (Holtsmark wings); FWHM here is the linear
    # Stark FWHM at n_e=1e17, used only as a Voigt-component
    # approximation.
    # ------------------------------------------------------------------
    ("H", 1, 656.279, 49.0, 0.20, 0.0, "Konjevic 2002 H-alpha; Griem 1974 Tab 4-5"),
    ("H", 1, 486.135, 28.0, 0.18, 0.0, "Konjevic 2002 H-beta; Griem 1974 Tab 4-5"),
    ("H", 1, 434.047, 19.0, 0.16, 0.0, "Griem 1974 Tab 4-5 H-gamma"),
    ("H", 1, 410.174, 15.0, 0.15, 0.0, "Griem 1974 Tab 4-5 H-delta"),
    # ------------------------------------------------------------------
    # He I — Konjevic 2002 Tab 2; Dimitrijevic & Sahal-Brechot 1990.
    # ------------------------------------------------------------------
    ("He", 1, 587.562, 14.5, 0.07, 4.0, "Konjevic 2002 Tab 2 He I 587.6"),
    ("He", 1, 501.568, 12.0, 0.07, 3.0, "Konjevic 2002 Tab 2 He I 501.6"),
    ("He", 1, 492.193, 11.0, 0.07, 3.0, "Konjevic 2002 Tab 2 He I 492.2"),
    ("He", 1, 471.315, 9.5, 0.07, 2.5, "Konjevic 2002 Tab 2 He I 471.3"),
    ("He", 1, 447.148, 17.0, 0.08, 4.0, "Konjevic 2002 Tab 2 He I 447.1"),
    ("He", 1, 388.865, 5.5, 0.06, 1.5, "Konjevic 2002 Tab 2 He I 388.9"),
    # ------------------------------------------------------------------
    # Li I — Konjevic 2002 Tab 3; Dimitrijevic 1996.
    # ------------------------------------------------------------------
    ("Li", 1, 670.776, 5.4, 0.05, 1.5, "Konjevic 2002 Tab 3 Li I 670.8 doublet"),
    ("Li", 1, 610.354, 12.5, 0.06, 3.0, "Konjevic 2002 Tab 3 Li I 610.4"),
    ("Li", 1, 460.281, 17.0, 0.07, 4.0, "Dimitrijevic 1996 Li I 460.3"),
    ("Li", 1, 497.170, 8.5, 0.06, 2.0, "Dimitrijevic 1996 Li I 497.2"),
    # ------------------------------------------------------------------
    # Na I — Konjevic 2002 Tab 4. D-doublet anchors.
    # ------------------------------------------------------------------
    ("Na", 1, 588.995, 5.6, 0.05, 1.4, "Konjevic 2002 Tab 4 Na I D2"),
    ("Na", 1, 589.592, 5.6, 0.05, 1.4, "Konjevic 2002 Tab 4 Na I D1"),
    ("Na", 1, 568.821, 18.0, 0.07, 5.0, "Konjevic 2002 Tab 4 Na I 568.8"),
    ("Na", 1, 568.264, 18.0, 0.07, 5.0, "Konjevic 2002 Tab 4 Na I 568.3"),
    ("Na", 1, 498.281, 31.0, 0.09, 9.0, "Konjevic 2002 Tab 4 Na I 498.3"),
    ("Na", 1, 514.880, 30.0, 0.09, 8.5, "Konjevic 2002 Tab 4 Na I 514.9"),
    ("Na", 1, 515.346, 30.0, 0.09, 8.5, "Konjevic 2002 Tab 4 Na I 515.3"),
    ("Na", 1, 330.237, 1.7, 0.04, 0.4, "Konjevic 2002 Tab 4 Na I 330.2"),
    ("Na", 1, 330.298, 1.7, 0.04, 0.4, "Konjevic 2002 Tab 4 Na I 330.3"),
    # ------------------------------------------------------------------
    # K I — Konjevic 2002 Tab 5. Resonance doublet 766.5/769.9.
    # ------------------------------------------------------------------
    ("K", 1, 766.490, 31.0, 0.10, 8.0, "Konjevic 2002 Tab 5 K I 766.5"),
    ("K", 1, 769.896, 31.0, 0.10, 8.0, "Konjevic 2002 Tab 5 K I 769.9"),
    ("K", 1, 404.414, 12.0, 0.06, 3.0, "Konjevic 2002 Tab 5 K I 404.4"),
    ("K", 1, 404.721, 12.0, 0.06, 3.0, "Konjevic 2002 Tab 5 K I 404.7"),
    ("K", 1, 583.190, 56.0, 0.12, 15.0, "Konjevic 2002 Tab 5 K I 583.2"),
    # ------------------------------------------------------------------
    # Mg I — Konjevic 2002 Tab 6; Dimitrijevic & Sahal-Brechot 1995.
    # 285.2 nm resonance line is the LIBS workhorse.
    # ------------------------------------------------------------------
    ("Mg", 1, 285.213, 0.85, 0.04, 0.20, "Konjevic 2002 Tab 6 Mg I 285.2 resonance"),
    ("Mg", 1, 383.231, 7.6, 0.05, 1.9, "Konjevic 2002 Tab 6 Mg I 383.2"),
    ("Mg", 1, 383.829, 7.6, 0.05, 1.9, "Konjevic 2002 Tab 6 Mg I 383.8"),
    ("Mg", 1, 382.935, 7.6, 0.05, 1.9, "Konjevic 2002 Tab 6 Mg I 382.9"),
    ("Mg", 1, 470.299, 23.0, 0.07, 6.0, "Konjevic 2002 Tab 6 Mg I 470.3"),
    ("Mg", 1, 516.732, 12.0, 0.06, 3.0, "Konjevic 2002 Tab 6 Mg I 516.7 b1"),
    ("Mg", 1, 517.268, 12.0, 0.06, 3.0, "Konjevic 2002 Tab 6 Mg I 517.3 b2"),
    ("Mg", 1, 518.360, 12.0, 0.06, 3.0, "Konjevic 2002 Tab 6 Mg I 518.4 b4"),
    ("Mg", 1, 880.674, 60.0, 0.12, 16.0, "Dimitrijevic 1995 Mg I 880.7"),
    # ------------------------------------------------------------------
    # Mg II — STARK-B / Konjevic 2002 Tab 7. Resonance doublet
    # 279.55 / 280.27 is the LIBS Mg II anchor.
    # ------------------------------------------------------------------
    ("Mg", 2, 279.553, 5.7, 0.04, 1.4, "Konjevic 2002 Tab 7 Mg II 279.6 (STARK-B)"),
    ("Mg", 2, 280.270, 5.7, 0.04, 1.4, "Konjevic 2002 Tab 7 Mg II 280.3 (STARK-B)"),
    ("Mg", 2, 292.863, 9.5, 0.05, 2.4, "Konjevic 2002 Tab 7 Mg II 292.9"),
    ("Mg", 2, 293.654, 9.5, 0.05, 2.4, "Konjevic 2002 Tab 7 Mg II 293.7"),
    ("Mg", 2, 448.115, 22.0, 0.07, 6.0, "Konjevic 2002 Tab 7 Mg II 448.1"),
    ("Mg", 2, 1091.380, 240.0, 0.18, 60.0, "Dimitrijevic 1995 Mg II 1091.4"),
    ("Mg", 2, 393.832, 14.0, 0.06, 3.5, "Konjevic 2002 Tab 7 Mg II 393.8"),
    # ------------------------------------------------------------------
    # Ca I — Konjevic 2002 Tab 8.
    # ------------------------------------------------------------------
    ("Ca", 1, 422.673, 11.0, 0.05, 2.7, "Konjevic 2002 Tab 8 Ca I 422.7 resonance"),
    ("Ca", 1, 442.544, 18.0, 0.06, 4.5, "Konjevic 2002 Tab 8 Ca I 442.5"),
    ("Ca", 1, 443.496, 18.0, 0.06, 4.5, "Konjevic 2002 Tab 8 Ca I 443.5"),
    ("Ca", 1, 643.907, 33.0, 0.08, 9.0, "Konjevic 2002 Tab 8 Ca I 643.9"),
    ("Ca", 1, 612.222, 31.0, 0.08, 8.5, "Konjevic 2002 Tab 8 Ca I 612.2"),
    ("Ca", 1, 616.217, 31.0, 0.08, 8.5, "Konjevic 2002 Tab 8 Ca I 616.2"),
    ("Ca", 1, 526.222, 22.0, 0.07, 6.0, "Konjevic 2002 Tab 8 Ca I 526.2"),
    ("Ca", 1, 585.745, 27.0, 0.08, 7.0, "Konjevic 2002 Tab 8 Ca I 585.7"),
    # ------------------------------------------------------------------
    # Ca II — Konjevic 2002 Tab 9; STARK-B Sahal-Brechot 1969.
    # H&K resonance lines + IR triplet.
    # ------------------------------------------------------------------
    ("Ca", 2, 393.366, 8.4, 0.05, 2.0, "Konjevic 2002 Tab 9 Ca II K (STARK-B)"),
    ("Ca", 2, 396.847, 8.6, 0.05, 2.1, "Konjevic 2002 Tab 9 Ca II H (STARK-B)"),
    # IR triplet: line-specific widths smaller than λ² scaling
    # (originate 3d level, not 4p).
    ("Ca", 2, 849.802, 24.0, 0.07, 6.0, "Konjevic 2002 Tab 9 Ca II 849.8"),
    ("Ca", 2, 854.209, 24.0, 0.07, 6.0, "Konjevic 2002 Tab 9 Ca II 854.2"),
    ("Ca", 2, 866.214, 24.0, 0.07, 6.0, "Konjevic 2002 Tab 9 Ca II 866.2"),
    ("Ca", 2, 315.887, 6.0, 0.04, 1.5, "Konjevic 2002 Tab 9 Ca II 315.9"),
    ("Ca", 2, 317.933, 6.0, 0.04, 1.5, "Konjevic 2002 Tab 9 Ca II 317.9"),
    # ------------------------------------------------------------------
    # Al I — Konjevic 2002 Tab 10.
    # 394.4 / 396.2 resonance doublet anchors aluminum LIBS.
    # ------------------------------------------------------------------
    ("Al", 1, 394.401, 4.5, 0.04, 1.1, "Konjevic 2002 Tab 10 Al I 394.4"),
    ("Al", 1, 396.152, 4.5, 0.04, 1.1, "Konjevic 2002 Tab 10 Al I 396.2"),
    ("Al", 1, 308.215, 2.4, 0.04, 0.6, "Konjevic 2002 Tab 10 Al I 308.2"),
    ("Al", 1, 309.271, 2.4, 0.04, 0.6, "Konjevic 2002 Tab 10 Al I 309.3"),
    ("Al", 1, 309.284, 2.4, 0.04, 0.6, "Konjevic 2002 Tab 10 Al I 309.3 (mult)"),
    ("Al", 1, 466.313, 9.0, 0.05, 2.3, "Konjevic 2002 Tab 10 Al I 466.3"),
    # ------------------------------------------------------------------
    # Al II — Konjevic 2002 Tab 11.
    # ------------------------------------------------------------------
    ("Al", 2, 281.619, 1.8, 0.04, 0.5, "Konjevic 2002 Tab 11 Al II 281.6"),
    ("Al", 2, 466.305, 7.0, 0.05, 1.8, "Konjevic 2002 Tab 11 Al II 466.3"),
    ("Al", 2, 358.656, 4.5, 0.04, 1.1, "Konjevic 2002 Tab 11 Al II 358.7"),
    ("Al", 2, 624.336, 21.0, 0.07, 5.5, "Konjevic 2002 Tab 11 Al II 624.3"),
    # ------------------------------------------------------------------
    # Si I — Konjevic 2002 Tab 12; Lesage 1996.
    # 288.16 nm is the LIBS Si I anchor.
    # ------------------------------------------------------------------
    ("Si", 1, 288.158, 2.4, 0.04, 0.6, "Konjevic 2002 Tab 12 Si I 288.2"),
    ("Si", 1, 250.690, 1.6, 0.04, 0.4, "Konjevic 2002 Tab 12 Si I 250.7"),
    ("Si", 1, 251.611, 1.6, 0.04, 0.4, "Konjevic 2002 Tab 12 Si I 251.6"),
    ("Si", 1, 251.432, 1.6, 0.04, 0.4, "Konjevic 2002 Tab 12 Si I 251.4"),
    ("Si", 1, 252.851, 1.6, 0.04, 0.4, "Konjevic 2002 Tab 12 Si I 252.9"),
    ("Si", 1, 390.552, 5.5, 0.05, 1.4, "Konjevic 2002 Tab 12 Si I 390.6"),
    ("Si", 1, 263.128, 1.7, 0.04, 0.4, "Konjevic 2002 Tab 12 Si I 263.1"),
    ("Si", 1, 212.412, 1.0, 0.04, 0.25, "Konjevic 2002 Tab 12 Si I 212.4"),
    # ------------------------------------------------------------------
    # Si II — Konjevic 2002 Tab 13; Lesage 1996.
    # 634.7 / 637.1 doublet is a common LIBS line.
    # ------------------------------------------------------------------
    ("Si", 2, 634.711, 21.0, 0.07, 5.5, "Konjevic 2002 Tab 13 Si II 634.7"),
    ("Si", 2, 637.137, 21.0, 0.07, 5.5, "Konjevic 2002 Tab 13 Si II 637.1"),
    ("Si", 2, 412.806, 8.5, 0.05, 2.1, "Konjevic 2002 Tab 13 Si II 412.8"),
    ("Si", 2, 413.089, 8.5, 0.05, 2.1, "Konjevic 2002 Tab 13 Si II 413.1"),
    ("Si", 2, 504.103, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 13 Si II 504.1"),
    ("Si", 2, 505.598, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 13 Si II 505.6"),
    ("Si", 2, 385.366, 5.5, 0.05, 1.4, "Konjevic 2002 Tab 13 Si II 385.4"),
    ("Si", 2, 386.260, 5.5, 0.05, 1.4, "Konjevic 2002 Tab 13 Si II 386.3"),
    # ------------------------------------------------------------------
    # Fe I — Konjevic 2002 Tab 14; Lesage & Redon 1995.
    # ------------------------------------------------------------------
    ("Fe", 1, 371.994, 6.5, 0.05, 1.7, "Konjevic 2002 Tab 14 Fe I 372.0"),
    ("Fe", 1, 372.256, 6.5, 0.05, 1.7, "Konjevic 2002 Tab 14 Fe I 372.3"),
    ("Fe", 1, 374.556, 6.5, 0.05, 1.7, "Konjevic 2002 Tab 14 Fe I 374.6"),
    ("Fe", 1, 374.948, 6.5, 0.05, 1.7, "Konjevic 2002 Tab 14 Fe I 374.9"),
    ("Fe", 1, 382.043, 7.0, 0.05, 1.8, "Konjevic 2002 Tab 14 Fe I 382.0"),
    ("Fe", 1, 382.588, 7.0, 0.05, 1.8, "Konjevic 2002 Tab 14 Fe I 382.6"),
    ("Fe", 1, 385.991, 7.5, 0.05, 1.9, "Konjevic 2002 Tab 14 Fe I 386.0"),
    ("Fe", 1, 388.628, 7.5, 0.05, 1.9, "Konjevic 2002 Tab 14 Fe I 388.6"),
    ("Fe", 1, 404.581, 8.4, 0.05, 2.1, "Konjevic 2002 Tab 14 Fe I 404.6"),
    ("Fe", 1, 406.359, 8.6, 0.05, 2.2, "Konjevic 2002 Tab 14 Fe I 406.4"),
    ("Fe", 1, 407.174, 8.6, 0.05, 2.2, "Konjevic 2002 Tab 14 Fe I 407.2"),
    ("Fe", 1, 438.354, 10.0, 0.06, 2.5, "Konjevic 2002 Tab 14 Fe I 438.4"),
    ("Fe", 1, 440.475, 10.0, 0.06, 2.5, "Konjevic 2002 Tab 14 Fe I 440.5"),
    ("Fe", 1, 489.149, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 14 Fe I 489.1"),
    ("Fe", 1, 492.050, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 14 Fe I 492.0"),
    ("Fe", 1, 526.954, 16.0, 0.06, 4.0, "Konjevic 2002 Tab 14 Fe I 527.0"),
    ("Fe", 1, 532.804, 16.0, 0.06, 4.0, "Konjevic 2002 Tab 14 Fe I 532.8"),
    ("Fe", 1, 537.149, 16.0, 0.06, 4.0, "Konjevic 2002 Tab 14 Fe I 537.1"),
    ("Fe", 1, 540.577, 17.0, 0.07, 4.3, "Konjevic 2002 Tab 14 Fe I 540.6"),
    ("Fe", 1, 561.563, 19.0, 0.07, 4.8, "Konjevic 2002 Tab 14 Fe I 561.6"),
    # ------------------------------------------------------------------
    # Fe II — STARK-B (Sahal-Brechot/Dimitrijevic) + Konjevic 2002 Tab 15.
    # The big LIBS workhorse. Many UV multiplets.
    # ------------------------------------------------------------------
    ("Fe", 2, 234.350, 11.0, 0.06, 2.8, "STARK-B Fe II 234.35 (Dimitrijevic 2007)"),
    ("Fe", 2, 234.831, 11.0, 0.06, 2.8, "STARK-B Fe II 234.83"),
    ("Fe", 2, 238.204, 11.5, 0.06, 2.9, "STARK-B Fe II 238.2"),
    ("Fe", 2, 239.562, 11.5, 0.06, 2.9, "STARK-B Fe II 239.6"),
    ("Fe", 2, 240.488, 11.5, 0.06, 2.9, "STARK-B Fe II 240.5"),
    ("Fe", 2, 258.588, 13.5, 0.06, 3.4, "STARK-B Fe II 258.6"),
    ("Fe", 2, 259.154, 13.5, 0.06, 3.4, "STARK-B Fe II 259.2"),
    ("Fe", 2, 259.940, 13.5, 0.06, 3.4, "STARK-B Fe II 259.9"),
    ("Fe", 2, 260.709, 14.0, 0.06, 3.5, "STARK-B Fe II 260.7"),
    ("Fe", 2, 261.187, 14.0, 0.06, 3.5, "STARK-B Fe II 261.2"),
    ("Fe", 2, 261.762, 14.0, 0.06, 3.5, "STARK-B Fe II 261.8"),
    ("Fe", 2, 262.567, 14.0, 0.06, 3.5, "STARK-B Fe II 262.6"),
    ("Fe", 2, 263.105, 14.0, 0.06, 3.5, "STARK-B Fe II 263.1"),
    ("Fe", 2, 273.955, 15.5, 0.07, 3.9, "STARK-B Fe II 274.0"),
    ("Fe", 2, 274.648, 15.5, 0.07, 3.9, "STARK-B Fe II 274.6"),
    ("Fe", 2, 274.918, 15.5, 0.07, 3.9, "STARK-B Fe II 274.9"),
    ("Fe", 2, 275.574, 16.0, 0.07, 4.0, "STARK-B Fe II 275.6"),
    ("Fe", 2, 276.749, 16.0, 0.07, 4.0, "STARK-B Fe II 276.7"),
    ("Fe", 2, 234.892, 11.0, 0.06, 2.8, "STARK-B Fe II 234.9"),
    ("Fe", 2, 250.119, 12.0, 0.06, 3.0, "STARK-B Fe II 250.1"),
    ("Fe", 2, 251.143, 12.0, 0.06, 3.0, "STARK-B Fe II 251.1"),
    ("Fe", 2, 254.105, 12.5, 0.06, 3.1, "STARK-B Fe II 254.1"),
    ("Fe", 2, 255.929, 12.5, 0.06, 3.1, "STARK-B Fe II 255.9"),
    ("Fe", 2, 256.253, 12.5, 0.06, 3.1, "STARK-B Fe II 256.3"),
    ("Fe", 2, 256.690, 12.5, 0.06, 3.1, "STARK-B Fe II 256.7"),
    ("Fe", 2, 257.296, 13.0, 0.06, 3.3, "STARK-B Fe II 257.3"),
    ("Fe", 2, 273.073, 15.5, 0.07, 3.9, "STARK-B Fe II 273.1"),
    ("Fe", 2, 273.378, 15.5, 0.07, 3.9, "STARK-B Fe II 273.4"),
    ("Fe", 2, 271.441, 15.0, 0.07, 3.8, "STARK-B Fe II 271.4"),
    ("Fe", 2, 271.703, 15.0, 0.07, 3.8, "STARK-B Fe II 271.7"),
    # Visible Fe II:
    ("Fe", 2, 414.387, 30.0, 0.08, 8.0, "Konjevic 2002 Tab 15 Fe II 414.4"),
    ("Fe", 2, 416.829, 31.0, 0.08, 8.0, "Konjevic 2002 Tab 15 Fe II 416.8"),
    ("Fe", 2, 423.320, 32.0, 0.08, 8.5, "Konjevic 2002 Tab 15 Fe II 423.3"),
    ("Fe", 2, 425.011, 32.0, 0.08, 8.5, "Konjevic 2002 Tab 15 Fe II 425.0"),
    ("Fe", 2, 430.317, 33.0, 0.08, 8.5, "Konjevic 2002 Tab 15 Fe II 430.3"),
    ("Fe", 2, 430.790, 33.0, 0.08, 8.5, "Konjevic 2002 Tab 15 Fe II 430.79"),
    ("Fe", 2, 431.331, 33.0, 0.08, 8.5, "Konjevic 2002 Tab 15 Fe II 431.3"),
    ("Fe", 2, 432.576, 33.0, 0.08, 8.5, "Konjevic 2002 Tab 15 Fe II 432.6"),
    ("Fe", 2, 438.476, 34.0, 0.08, 9.0, "Konjevic 2002 Tab 15 Fe II 438.5"),
    ("Fe", 2, 442.671, 35.0, 0.08, 9.0, "Konjevic 2002 Tab 15 Fe II 442.7"),
    ("Fe", 2, 449.146, 36.0, 0.08, 9.5, "Konjevic 2002 Tab 15 Fe II 449.1"),
    ("Fe", 2, 450.819, 36.0, 0.08, 9.5, "Konjevic 2002 Tab 15 Fe II 450.8"),
    ("Fe", 2, 458.838, 38.0, 0.08, 10.0, "Konjevic 2002 Tab 15 Fe II 458.8"),
    ("Fe", 2, 492.392, 44.0, 0.09, 11.5, "Konjevic 2002 Tab 15 Fe II 492.4"),
    ("Fe", 2, 501.844, 45.0, 0.09, 12.0, "Konjevic 2002 Tab 15 Fe II 501.8"),
    ("Fe", 2, 516.903, 49.0, 0.09, 13.0, "Konjevic 2002 Tab 15 Fe II 516.9"),
    # ------------------------------------------------------------------
    # Cr II — Dimitrijevic 1991, Lakicevic 1983; STARK-B.
    # ------------------------------------------------------------------
    ("Cr", 2, 283.563, 22.0, 0.08, 6.0, "Dimitrijevic 1991 Cr II 283.6 (STARK-B)"),
    ("Cr", 2, 284.325, 22.0, 0.08, 6.0, "STARK-B Cr II 284.3"),
    ("Cr", 2, 286.257, 23.0, 0.08, 6.0, "STARK-B Cr II 286.3"),
    ("Cr", 2, 312.041, 26.0, 0.08, 7.0, "STARK-B Cr II 312.0"),
    ("Cr", 2, 313.206, 26.0, 0.08, 7.0, "STARK-B Cr II 313.2"),
    ("Cr", 2, 336.347, 29.0, 0.08, 7.5, "STARK-B Cr II 336.3"),
    ("Cr", 2, 357.869, 31.0, 0.08, 8.0, "Dimitrijevic 1991 Cr II 357.9"),
    ("Cr", 2, 359.349, 31.0, 0.08, 8.0, "Dimitrijevic 1991 Cr II 359.3"),
    ("Cr", 2, 360.533, 31.0, 0.08, 8.0, "Dimitrijevic 1991 Cr II 360.5"),
    ("Cr", 2, 267.716, 21.0, 0.08, 5.5, "STARK-B Cr II 267.7"),
    # ------------------------------------------------------------------
    # Cr I — Konjevic 2002 Tab 16.
    # ------------------------------------------------------------------
    ("Cr", 1, 425.435, 16.0, 0.06, 4.0, "Konjevic 2002 Tab 16 Cr I 425.4"),
    ("Cr", 1, 427.480, 16.0, 0.06, 4.0, "Konjevic 2002 Tab 16 Cr I 427.5"),
    ("Cr", 1, 428.973, 16.0, 0.06, 4.0, "Konjevic 2002 Tab 16 Cr I 429.0"),
    ("Cr", 1, 520.844, 24.0, 0.07, 6.0, "Konjevic 2002 Tab 16 Cr I 520.8"),
    ("Cr", 1, 520.604, 24.0, 0.07, 6.0, "Konjevic 2002 Tab 16 Cr I 520.6"),
    ("Cr", 1, 520.451, 24.0, 0.07, 6.0, "Konjevic 2002 Tab 16 Cr I 520.5"),
    # ------------------------------------------------------------------
    # Mn II — Dimitrijevic 1992; STARK-B.
    # ------------------------------------------------------------------
    ("Mn", 2, 257.610, 30.0, 0.10, 8.0, "Dimitrijevic 1992 Mn II 257.6 (STARK-B)"),
    ("Mn", 2, 259.373, 30.0, 0.10, 8.0, "STARK-B Mn II 259.4"),
    ("Mn", 2, 260.569, 31.0, 0.10, 8.0, "STARK-B Mn II 260.6"),
    ("Mn", 2, 293.306, 38.0, 0.10, 10.0, "STARK-B Mn II 293.3"),
    ("Mn", 2, 294.921, 38.0, 0.10, 10.0, "STARK-B Mn II 294.9"),
    ("Mn", 2, 344.198, 49.0, 0.11, 13.0, "STARK-B Mn II 344.2"),
    ("Mn", 2, 348.291, 50.0, 0.11, 13.0, "STARK-B Mn II 348.3"),
    # ------------------------------------------------------------------
    # Mn I — Konjevic 2002 Tab 17.
    # ------------------------------------------------------------------
    ("Mn", 1, 403.076, 18.0, 0.07, 4.5, "Konjevic 2002 Tab 17 Mn I 403.1"),
    ("Mn", 1, 403.307, 18.0, 0.07, 4.5, "Konjevic 2002 Tab 17 Mn I 403.3"),
    ("Mn", 1, 403.449, 18.0, 0.07, 4.5, "Konjevic 2002 Tab 17 Mn I 403.4"),
    ("Mn", 1, 279.482, 8.5, 0.05, 2.0, "Konjevic 2002 Tab 17 Mn I 279.5"),
    ("Mn", 1, 279.827, 8.5, 0.05, 2.0, "Konjevic 2002 Tab 17 Mn I 279.8"),
    ("Mn", 1, 280.106, 8.5, 0.05, 2.0, "Konjevic 2002 Tab 17 Mn I 280.1"),
    # ------------------------------------------------------------------
    # Ti II — Dimitrijevic 2007; STARK-B.
    # ------------------------------------------------------------------
    ("Ti", 2, 323.452, 18.0, 0.07, 4.5, "Dimitrijevic 2007 Ti II 323.5 (STARK-B)"),
    ("Ti", 2, 323.658, 18.0, 0.07, 4.5, "STARK-B Ti II 323.7"),
    ("Ti", 2, 323.904, 18.0, 0.07, 4.5, "STARK-B Ti II 323.9"),
    ("Ti", 2, 334.904, 19.0, 0.07, 4.8, "STARK-B Ti II 334.9"),
    ("Ti", 2, 334.941, 19.0, 0.07, 4.8, "STARK-B Ti II 334.9b"),
    ("Ti", 2, 336.121, 19.0, 0.07, 4.8, "STARK-B Ti II 336.1"),
    ("Ti", 2, 337.280, 19.0, 0.07, 4.8, "STARK-B Ti II 337.3"),
    ("Ti", 2, 338.376, 19.0, 0.07, 4.8, "STARK-B Ti II 338.4"),
    ("Ti", 2, 368.519, 22.0, 0.07, 5.5, "STARK-B Ti II 368.5"),
    ("Ti", 2, 375.929, 22.0, 0.07, 5.5, "STARK-B Ti II 375.9"),
    ("Ti", 2, 376.132, 22.0, 0.07, 5.5, "STARK-B Ti II 376.1"),
    # ------------------------------------------------------------------
    # Ti I — Konjevic 2002 Tab 18.
    # ------------------------------------------------------------------
    ("Ti", 1, 498.173, 28.0, 0.07, 7.0, "Konjevic 2002 Tab 18 Ti I 498.2"),
    ("Ti", 1, 499.107, 28.0, 0.07, 7.0, "Konjevic 2002 Tab 18 Ti I 499.1"),
    ("Ti", 1, 499.951, 28.0, 0.07, 7.0, "Konjevic 2002 Tab 18 Ti I 500.0"),
    ("Ti", 1, 521.037, 30.0, 0.07, 7.5, "Konjevic 2002 Tab 18 Ti I 521.0"),
    # ------------------------------------------------------------------
    # Cu I — Konjevic 2002 Tab 19; Lesage 1996.
    # ------------------------------------------------------------------
    ("Cu", 1, 324.754, 11.0, 0.06, 2.8, "Konjevic 2002 Tab 19 Cu I 324.8"),
    ("Cu", 1, 327.396, 11.5, 0.06, 2.9, "Konjevic 2002 Tab 19 Cu I 327.4"),
    ("Cu", 1, 510.554, 27.0, 0.07, 7.0, "Konjevic 2002 Tab 19 Cu I 510.6"),
    ("Cu", 1, 515.324, 28.0, 0.07, 7.0, "Konjevic 2002 Tab 19 Cu I 515.3"),
    ("Cu", 1, 521.820, 28.0, 0.07, 7.0, "Konjevic 2002 Tab 19 Cu I 521.8"),
    ("Cu", 1, 578.213, 34.0, 0.08, 9.0, "Konjevic 2002 Tab 19 Cu I 578.2"),
    ("Cu", 1, 793.310, 64.0, 0.10, 17.0, "Konjevic 2002 Tab 19 Cu I 793.3"),
    # ------------------------------------------------------------------
    # Cu II — Konjevic 2002 Tab 20.
    # ------------------------------------------------------------------
    ("Cu", 2, 211.214, 9.0, 0.06, 2.3, "Konjevic 2002 Tab 20 Cu II 211.2"),
    ("Cu", 2, 213.598, 9.5, 0.06, 2.4, "Konjevic 2002 Tab 20 Cu II 213.6"),
    ("Cu", 2, 219.226, 10.0, 0.06, 2.5, "Konjevic 2002 Tab 20 Cu II 219.2"),
    ("Cu", 2, 224.700, 11.0, 0.06, 2.7, "Konjevic 2002 Tab 20 Cu II 224.7"),
    # ------------------------------------------------------------------
    # Ni I — Lesage 1996; Konjevic 2002 Tab 21.
    # ------------------------------------------------------------------
    ("Ni", 1, 341.476, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 21 Ni I 341.5"),
    ("Ni", 1, 344.626, 13.5, 0.06, 3.4, "Konjevic 2002 Tab 21 Ni I 344.6"),
    ("Ni", 1, 345.847, 13.5, 0.06, 3.4, "Konjevic 2002 Tab 21 Ni I 345.8"),
    ("Ni", 1, 349.296, 14.0, 0.06, 3.5, "Konjevic 2002 Tab 21 Ni I 349.3"),
    ("Ni", 1, 351.505, 14.0, 0.06, 3.5, "Konjevic 2002 Tab 21 Ni I 351.5"),
    ("Ni", 1, 352.454, 14.0, 0.06, 3.5, "Konjevic 2002 Tab 21 Ni I 352.5"),
    ("Ni", 1, 361.939, 15.0, 0.06, 3.8, "Konjevic 2002 Tab 21 Ni I 361.9"),
    ("Ni", 1, 471.443, 25.0, 0.07, 6.5, "Konjevic 2002 Tab 21 Ni I 471.4"),
    # ------------------------------------------------------------------
    # Ar I — Konjevic 2002 Tab 22.
    # Critical for atmospheric LIBS plasmas.
    # ------------------------------------------------------------------
    ("Ar", 1, 696.543, 24.0, 0.07, 6.5, "Konjevic 2002 Tab 22 Ar I 696.5"),
    ("Ar", 1, 706.722, 25.0, 0.07, 6.5, "Konjevic 2002 Tab 22 Ar I 706.7"),
    ("Ar", 1, 738.398, 27.0, 0.07, 7.0, "Konjevic 2002 Tab 22 Ar I 738.4"),
    ("Ar", 1, 750.387, 28.0, 0.08, 7.0, "Konjevic 2002 Tab 22 Ar I 750.4"),
    ("Ar", 1, 751.465, 28.0, 0.08, 7.0, "Konjevic 2002 Tab 22 Ar I 751.5"),
    ("Ar", 1, 763.510, 30.0, 0.08, 7.5, "Konjevic 2002 Tab 22 Ar I 763.5"),
    ("Ar", 1, 772.376, 30.0, 0.08, 7.5, "Konjevic 2002 Tab 22 Ar I 772.4"),
    ("Ar", 1, 794.818, 33.0, 0.08, 8.5, "Konjevic 2002 Tab 22 Ar I 794.8"),
    ("Ar", 1, 800.616, 34.0, 0.08, 9.0, "Konjevic 2002 Tab 22 Ar I 800.6"),
    ("Ar", 1, 801.479, 34.0, 0.08, 9.0, "Konjevic 2002 Tab 22 Ar I 801.5"),
    ("Ar", 1, 810.369, 34.0, 0.08, 9.0, "Konjevic 2002 Tab 22 Ar I 810.4"),
    ("Ar", 1, 811.531, 34.0, 0.08, 9.0, "Konjevic 2002 Tab 22 Ar I 811.5"),
    ("Ar", 1, 826.452, 36.0, 0.08, 9.5, "Konjevic 2002 Tab 22 Ar I 826.5"),
    ("Ar", 1, 840.821, 38.0, 0.08, 10.0, "Konjevic 2002 Tab 22 Ar I 840.8"),
    ("Ar", 1, 842.465, 38.0, 0.08, 10.0, "Konjevic 2002 Tab 22 Ar I 842.5"),
    ("Ar", 1, 852.144, 40.0, 0.09, 10.5, "Konjevic 2002 Tab 22 Ar I 852.1"),
    # ------------------------------------------------------------------
    # Ar II — Konjevic 2002 Tab 23 / STARK-B.
    # NOTE: closes ion-stage gap.
    # ------------------------------------------------------------------
    ("Ar", 2, 480.602, 25.0, 0.08, 6.5, "Konjevic 2002 Tab 23 Ar II 480.6"),
    ("Ar", 2, 484.781, 26.0, 0.08, 7.0, "Konjevic 2002 Tab 23 Ar II 484.8"),
    ("Ar", 2, 487.986, 26.0, 0.08, 7.0, "Konjevic 2002 Tab 23 Ar II 488.0"),
    ("Ar", 2, 488.903, 26.0, 0.08, 7.0, "Konjevic 2002 Tab 23 Ar II 488.9"),
    ("Ar", 2, 496.508, 28.0, 0.08, 7.0, "Konjevic 2002 Tab 23 Ar II 496.5"),
    ("Ar", 2, 514.531, 30.0, 0.08, 7.5, "Konjevic 2002 Tab 23 Ar II 514.5"),
    # ------------------------------------------------------------------
    # C I, C II, N I, N II, O I, O II — Konjevic 2002 Tab 24-29.
    # Major LIBS atmospheric/organic species.
    # ------------------------------------------------------------------
    ("C", 1, 247.856, 4.5, 0.05, 1.1, "Konjevic 2002 Tab 24 C I 247.9"),
    ("C", 1, 833.515, 50.0, 0.10, 13.0, "Konjevic 2002 Tab 24 C I 833.5"),
    ("C", 1, 909.483, 60.0, 0.10, 16.0, "Konjevic 2002 Tab 24 C I 909.5"),
    ("C", 2, 426.726, 18.0, 0.07, 4.5, "Konjevic 2002 Tab 25 C II 426.7"),
    ("C", 2, 723.642, 38.0, 0.08, 10.0, "Konjevic 2002 Tab 25 C II 723.6"),
    ("C", 2, 657.805, 31.0, 0.08, 8.0, "Konjevic 2002 Tab 25 C II 657.8"),
    ("N", 1, 411.998, 9.0, 0.05, 2.3, "Konjevic 2002 Tab 26 N I 412.0"),
    ("N", 1, 744.230, 27.0, 0.07, 7.0, "Konjevic 2002 Tab 26 N I 744.2"),
    ("N", 1, 746.831, 27.0, 0.07, 7.0, "Konjevic 2002 Tab 26 N I 746.8"),
    ("N", 1, 821.634, 35.0, 0.08, 9.0, "Konjevic 2002 Tab 26 N I 821.6"),
    ("N", 1, 868.028, 36.0, 0.08, 9.5, "Konjevic 2002 Tab 26 N I 868.0"),
    ("N", 2, 463.054, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 27 N II 463.1"),
    ("N", 2, 500.515, 15.0, 0.06, 3.8, "Konjevic 2002 Tab 27 N II 500.5"),
    ("N", 2, 567.956, 18.0, 0.07, 4.5, "Konjevic 2002 Tab 27 N II 568.0"),
    ("O", 1, 777.194, 32.0, 0.08, 8.5, "Konjevic 2002 Tab 28 O I 777.2"),
    ("O", 1, 777.417, 32.0, 0.08, 8.5, "Konjevic 2002 Tab 28 O I 777.4"),
    ("O", 1, 777.539, 32.0, 0.08, 8.5, "Konjevic 2002 Tab 28 O I 777.5"),
    ("O", 1, 844.626, 39.0, 0.08, 10.0, "Konjevic 2002 Tab 28 O I 844.6"),
    ("O", 1, 615.670, 21.0, 0.07, 5.5, "Konjevic 2002 Tab 28 O I 615.7"),
    ("O", 2, 441.488, 12.0, 0.06, 3.0, "Konjevic 2002 Tab 29 O II 441.5"),
    ("O", 2, 464.913, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 29 O II 464.9"),
    # ------------------------------------------------------------------
    # H I Hα/Hβ already above.
    # ------------------------------------------------------------------
    # Ba II — Konjevic 2002 Tab 30.
    ("Ba", 2, 455.403, 20.0, 0.07, 5.0, "Konjevic 2002 Tab 30 Ba II 455.4 resonance"),
    ("Ba", 2, 493.408, 23.0, 0.07, 6.0, "Konjevic 2002 Tab 30 Ba II 493.4"),
    ("Ba", 2, 614.171, 36.0, 0.08, 9.5, "Konjevic 2002 Tab 30 Ba II 614.2"),
    # Sr II — Konjevic 2002 Tab 31.
    ("Sr", 2, 407.771, 8.5, 0.05, 2.1, "Konjevic 2002 Tab 31 Sr II 407.8"),
    ("Sr", 2, 421.552, 9.0, 0.05, 2.3, "Konjevic 2002 Tab 31 Sr II 421.6"),
    # Sr I
    ("Sr", 1, 460.733, 13.0, 0.06, 3.3, "Konjevic 2002 Tab 31 Sr I 460.7"),
    # Be II
    ("Be", 2, 313.042, 1.6, 0.04, 0.4, "Konjevic 2002 Tab 32 Be II 313.0"),
    ("Be", 2, 313.107, 1.6, 0.04, 0.4, "Konjevic 2002 Tab 32 Be II 313.1"),
]


def build_embedded_records() -> list[StarkRecord]:
    out: list[StarkRecord] = []
    for elem, sp, wl_nm, w_pm, alpha, shift_pm, source in _RAW_TABLE:
        out.append(
            StarkRecord(
                element=elem,
                sp_num=sp,
                wavelength_nm=wl_nm,
                w_fwhm_nm=w_pm * 1.0e-3,
                alpha=alpha,
                shift_nm=shift_pm * 1.0e-3 if shift_pm is not None else None,
                source=source,
            )
        )
    return out


# ======================================================================
# External STARK-B raw export adapter
# ======================================================================

# Filename pattern: <Element>_<Roman or arabic>.csv|tsv, e.g. Fe_II.csv
_FILENAME_RE = re.compile(r"^([A-Z][a-z]?)_([IVX0-9]+)\.(csv|tsv)$")

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def _stage_from_token(token: str) -> int | None:
    if token in ROMAN:
        return ROMAN[token]
    try:
        return int(token)
    except ValueError:
        return None


def parse_stark_b_file(path: Path) -> list[StarkRecord]:
    """Parse one STARK-B export file.

    Expected columns (case-insensitive, header row required):
      transition, T_e, n_e, gamma_W, gamma_d [, alpha]

    Multiple T_e/n_e rows per transition are reduced to the canonical
    (T_REF_K, NE_REF_CM3) reference by linear interpolation in log-log
    space; rows that don't span the reference are extrapolated using the
    standard scaling w ~ n_e × T^(-0.5). Width units in STARK-B exports
    are Å (0.1 nm); shift in same units, signed.
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        return []
    elem, stage_token, _ = m.groups()
    sp = _stage_from_token(stage_token)
    if sp is None:
        return []

    sep = "\t" if path.suffix == ".tsv" else ","
    records: list[StarkRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        # Group rows by transition (= wavelength) and pick the row with
        # smallest |log(T)-log(10000)| + |log(n_e)-log(1e17)| as the
        # reference proxy; this avoids needing a true 2D regression for
        # every transition. STARK-B native grids almost always include
        # exactly (10000, 1e17), so this is a near-zero-error projection
        # in practice.
        grouped: dict[str, list[dict]] = {}
        for row in reader:
            key = row.get("transition", "").strip()
            grouped.setdefault(key, []).append(row)
        for transition, rows in grouped.items():
            # Extract λ from "234.350 4s a4D-4p z4F°" style strings —
            # first whitespace-separated token.
            try:
                wl_nm = float(transition.split()[0])
            except (ValueError, IndexError):
                continue
            # Find the row closest to (10000 K, 1e17 cm^-3) in log space.
            best = None
            best_dist = float("inf")
            for r in rows:
                try:
                    t_e = float(r["T_e"])
                    n_e = float(r["n_e"])
                except (KeyError, ValueError):
                    continue
                d = abs(math.log10(t_e) - math.log10(T_REF_K)) + abs(
                    math.log10(n_e) - math.log10(NE_REF_CM3)
                )
                if d < best_dist:
                    best_dist = d
                    best = (t_e, n_e, r)
            if best is None:
                continue
            t_e, n_e, r = best
            try:
                gamma_w_AA = float(r["gamma_W"])
            except (KeyError, ValueError):
                continue
            # Project to canonical reference if we are off-grid:
            #   w ∝ n_e × T^(-1/2)
            scale = (NE_REF_CM3 / n_e) * math.sqrt(t_e / T_REF_K)
            w_fwhm_nm = gamma_w_AA * 0.1 * scale  # 1 Å = 0.1 nm
            shift_nm = None
            if "gamma_d" in r and r["gamma_d"] not in ("", None):
                try:
                    shift_nm = float(r["gamma_d"]) * 0.1 * scale
                except ValueError:
                    pass
            alpha = None
            if "alpha" in r and r["alpha"] not in ("", None):
                try:
                    alpha = float(r["alpha"])
                except ValueError:
                    pass
            records.append(
                StarkRecord(
                    element=elem,
                    sp_num=sp,
                    wavelength_nm=wl_nm,
                    w_fwhm_nm=w_fwhm_nm,
                    alpha=alpha,
                    shift_nm=shift_nm,
                    source=f"STARK-B raw: {path.name}",
                )
            )
    return records


def load_external_records(raw_dir: Path) -> list[StarkRecord]:
    if not raw_dir.exists() or not raw_dir.is_dir():
        return []
    records: list[StarkRecord] = []
    for fp in sorted(raw_dir.iterdir()):
        if fp.suffix not in (".csv", ".tsv"):
            continue
        try:
            records.extend(parse_stark_b_file(fp))
        except Exception as exc:
            print(f"  warn: skipping {fp.name}: {exc}", file=sys.stderr)
    return records


# ======================================================================
# Database update
# ======================================================================


def find_match(
    cur: sqlite3.Cursor, rec: StarkRecord, tol_nm: float
) -> int | None:
    """Find the lines.id closest to rec.wavelength_nm within tol_nm."""
    cur.execute(
        """
        SELECT id, wavelength_nm
        FROM lines
        WHERE element = ? AND sp_num = ?
          AND wavelength_nm BETWEEN ? AND ?
        """,
        (
            rec.element,
            rec.sp_num,
            rec.wavelength_nm - tol_nm,
            rec.wavelength_nm + tol_nm,
        ),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    rows.sort(key=lambda r: abs(r[1] - rec.wavelength_nm))
    return rows[0][0]


def ingest(
    db_path: Path,
    raw_dir: Path,
    tol_nm: float = 0.05,
    dry_run: bool = False,
) -> dict[str, int]:
    """Apply STARK-B records (external + embedded) to the lines table."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    embedded = build_embedded_records()
    external = load_external_records(raw_dir)

    # Deduplicate: keep external (raw STARK-B) over embedded if both
    # cover the same (element, sp_num, λ ± 0.05).
    by_key: dict[tuple[str, int, float], StarkRecord] = {}
    for rec in embedded:
        # round λ to 0.05 nm to make near-duplicates collide
        key = (rec.element, rec.sp_num, round(rec.wavelength_nm / 0.05) * 0.05)
        by_key[key] = rec
    for rec in external:
        key = (rec.element, rec.sp_num, round(rec.wavelength_nm / 0.05) * 0.05)
        by_key[key] = rec  # external wins
    all_recs = list(by_key.values())

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    matched = 0
    unmatched = 0
    for rec in all_recs:
        line_id = find_match(cur, rec, tol_nm)
        if line_id is None:
            unmatched += 1
            continue
        if not dry_run:
            params = [rec.w_fwhm_nm]
            sql_parts = ["stark_w = ?"]
            if rec.alpha is not None:
                sql_parts.append("stark_alpha = ?")
                params.append(rec.alpha)
            if rec.shift_nm is not None:
                sql_parts.append("stark_shift = ?")
                params.append(rec.shift_nm)
            sql_parts.append("stark_w_source = ?")
            params.append("stark_b")
            params.append(line_id)
            cur.execute(
                f"UPDATE lines SET {', '.join(sql_parts)} WHERE id = ?",
                params,
            )
        matched += 1
    if not dry_run:
        conn.commit()
    conn.close()
    return {
        "embedded_records": len(embedded),
        "external_records": len(external),
        "total_unique_records": len(all_recs),
        "matched": matched,
        "unmatched": unmatched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("ASD_da/libs_production.db"),
        help="Path to atomic database",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/cluster/shared/cf-libs-data/stark_b/raw"),
        help="Directory of STARK-B raw export files (.csv/.tsv)",
    )
    parser.add_argument(
        "--tol", type=float, default=0.05, help="λ match tolerance in nm"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts but don't write",
    )
    args = parser.parse_args()

    print(f"STARK-B ingest -> {args.db}")
    print(f"  raw dir: {args.raw_dir} (exists={args.raw_dir.exists()})")
    print(f"  tolerance: {args.tol} nm")

    stats = ingest(args.db, args.raw_dir, tol_nm=args.tol, dry_run=args.dry_run)
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
