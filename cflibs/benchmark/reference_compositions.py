"""Certified compositions for community CF-LIBS reference materials.

These are the truth-side lookup tables for published reference materials
that the CF-LIBS literature uses to validate composition retrieval. Once
spectra of these materials are ingested into ``data/``, the benchmark
gate's identification recall floor and per-element relative-error
strata become enforceable against community-comparable ground truth.

References
----------
* BHVO-2 (USGS Hawaiian basalt): USGS preliminary compilation by
  Jochum et al. (2005) "GeoReM: A New Geochemical Database for Reference
  Materials and Isotopic Standards", *Geostandards and Geoanalytical
  Research* 29(3):333-338, plus USGS Reference Materials Program
  (https://crustal.usgs.gov/geochemical_reference_standards/).
* BIR-1 (USGS Icelandic basalt): same compilation; certified by
  USGS Reference Materials Program.
* NIST SRM 612 (trace-element glass): NIST Standard Reference Materials
  Catalog; Pearce et al. (1997) "A compilation of new and published
  major and trace element data for NIST SRM 610 and NIST SRM 612 glass
  reference materials", *Geostandards Newsletter* 21(1):115-144.

Values are mass fractions (NOT wt% — multiply by 100 if reading the
literature; we store as 0-1 fractions to match unified-runner's
``true_composition`` schema).

Compositions here are MAJOR + MINOR elements only; trace elements
(< 0.001 mass fraction) are reported as ``trace_below_loq`` and
omitted from the dict to avoid coercing < LOQ values into pseudo-
certainties. The benchmark gate's composition_strata configuration
(majors > 0.05, minors 0.001-0.05, traces < 0.001) consumes this
shape correctly.
"""

from __future__ import annotations

from typing import Mapping

# ---------------------------------------------------------------------------
# BHVO-2 (USGS Hawaiian basalt SRM)
# ---------------------------------------------------------------------------
# Reference: GeoReM compilation, Jochum 2005. Converted from oxide wt%
# to elemental mass fractions assuming stoichiometry (e.g. SiO2 wt% × 0.4675
# = Si mass fraction). Original oxide values:
#   SiO2  49.9 wt%
#   TiO2   2.73
#   Al2O3 13.5
#   Fe2O3  12.3  (total Fe expressed as Fe2O3)
#   MnO    0.17
#   MgO    7.23
#   CaO   11.4
#   Na2O   2.22
#   K2O    0.52
#   P2O5   0.27
# Converted to elemental mass fractions:
BHVO2_BASALT_USGS: Mapping[str, float] = {
    "Si": 0.2333,  # SiO2 49.9% × (28.09 / 60.08) = 0.2333
    "Ti": 0.0164,  # TiO2 2.73% × (47.87 / 79.87) = 0.01636
    "Al": 0.0714,  # Al2O3 13.5% × (53.96 / 101.96) = 0.07145
    "Fe": 0.0861,  # Fe2O3 12.3% × (111.69 / 159.69) = 0.08603
    "Mn": 0.0013,  # MnO 0.17% × (54.94 / 70.94) = 0.001317
    "Mg": 0.0436,  # MgO 7.23% × (24.31 / 40.30) = 0.04361
    "Ca": 0.0815,  # CaO 11.4% × (40.08 / 56.08) = 0.08149
    "Na": 0.0165,  # Na2O 2.22% × (45.98 / 61.98) = 0.01647
    "K": 0.00432,  # K2O 0.52% × (78.20 / 94.20) = 0.004317
    "P": 0.00118,  # P2O5 0.27% × (61.94 / 141.94) = 0.001178
}


# ---------------------------------------------------------------------------
# BIR-1 (USGS Icelandic basalt SRM)
# ---------------------------------------------------------------------------
# Reference: GeoReM, Flanagan 1984 oxide compilation. Original:
#   SiO2  47.8
#   TiO2   0.96
#   Al2O3 15.4
#   Fe2O3 11.3
#   MnO    0.17
#   MgO    9.7
#   CaO   13.3
#   Na2O   1.81
#   K2O    0.027  (very low K — useful trace-element calibration)
#   P2O5   0.03
BIR1_BASALT_USGS: Mapping[str, float] = {
    "Si": 0.2234,
    "Ti": 0.00575,
    "Al": 0.0815,
    "Fe": 0.0791,
    "Mn": 0.00132,
    "Mg": 0.0585,
    "Ca": 0.0951,
    "Na": 0.01343,
    "K": 0.000224,  # near LOQ — minor element
    "P": 0.000131,
}


# ---------------------------------------------------------------------------
# NIST SRM 612 (trace-element doped glass)
# ---------------------------------------------------------------------------
# Reference: Pearce et al. 1997 "A compilation of new and published
# major and trace element data for NIST SRM 610 and 612". SRM 612 is
# nominally 41 trace elements doped at ~38 ppm in a SiO2-Al2O3-CaO-Na2O
# matrix. Majors only listed here; trace dopants below LOQ are omitted.
# Original major-element oxide composition:
#   SiO2  72   (matrix glass)
#   Al2O3 2.0
#   CaO  12.0
#   Na2O 14.0
NIST_SRM_612_GLASS: Mapping[str, float] = {
    "Si": 0.3366,  # SiO2 72% × 0.4675
    "Al": 0.01059,  # Al2O3 2% × 0.5293
    "Ca": 0.08574,  # CaO 12% × 0.7147
    "Na": 0.10388,  # Na2O 14% × 0.7419
    # Trace dopants ~38 ppm = 0.0000038 — below LOQ for CF-LIBS;
    # omitted to match composition_strata.traces.bound = "mdl".
}


# ---------------------------------------------------------------------------
# Registry — dataset_id → certified composition
# ---------------------------------------------------------------------------
# Map ingest-time dataset_ids to the certified composition for the
# physical sample. The benchmark loader (cflibs/benchmark/loaders.py)
# uses this to populate each spectrum's ``true_composition`` when the
# spectra are ingested. Empty for now; populated by the BHVO-2 / NIST
# SRM 612 ingest PR (CF-LIBS-improved bd issue).
REFERENCE_COMPOSITIONS: Mapping[str, Mapping[str, float]] = {
    "bhvo2_usgs": BHVO2_BASALT_USGS,
    "bir1_usgs": BIR1_BASALT_USGS,
    "nist_srm_612": NIST_SRM_612_GLASS,
}


def get_reference_composition(dataset_id: str) -> Mapping[str, float] | None:
    """Return certified composition (mass fractions) for a known CRM.

    Returns None if the dataset_id is not in the registry — the caller
    should fall through to the dataset's own ``true_composition`` field
    (legacy path for aalto_libs / aa1100_substrate).

    Once spectra of these materials are ingested into ``data/`` and
    registered in ``cflibs/benchmark/loaders.py``, the benchmark gate's
    per-element-%RE and identification-recall floor become enforceable
    against community-comparable ground truth.
    """
    return REFERENCE_COMPOSITIONS.get(dataset_id)
