"""
MSL ChemCam preflight calibration spectra helper (PDS CALIB directory).

Files (under ``data/chemcam_calib/``, PDS dataset
``MSL-M-CHEMCAM-LIBS-4/5-RDR-V1.0``, CALIB directory):

- ``msl_ccam_libs_calib.csv``: preflight cleanroom spectra of 66 standards.
  Row 1 is a header of sample names (3-4 replicate columns per standard);
  remaining 6144 rows are ``wavelength_nm, radiance...`` with radiance in
  photons/shot/mm^2/sr/nm (Wiens et al. 2013, Spectrochim. Acta B; observed
  at 3 m through 7 Torr CO2). Wavelengths are strictly increasing, 240.8 to
  905.6 nm with the two physical inter-spectrometer gaps (341-382, 469-473).
- ``ccam_calibration_compositions.csv``: target compositions; columns
  ``Spectrum Name`` (join key against the spectra header) and the oxide panel
  SiO2/TiO2/Al2O3/FeOT/MnO/MgO/CaO/Na2O/K2O in wt% plus ``MOC total``.

Truth notes
-----------
- 65 of the 66 spectrum names join to a composition row; ``M6-HAGGERTY`` has
  none and is skipped with a log message. Duplicate composition rows (SGR1
  appears under two targets with identical values) take the first occurrence.
- Standards whose certified oxide panel covers < 50 wt% of the sample (e.g.
  the GYP* gypsum series, where S and structural water are uncertified) are
  skipped with a log message.
- Resolving-power hint: ChemCam FWHM is approximately 0.15/0.20/0.65 nm in
  the UV/VIO/VNIR bands (Wiens et al. 2012, Space Sci. Rev.), i.e. R is
  about 2000/2100/1050 at band centre; 2000.0 is recorded as a single-number
  hint with the per-band detail in the notes.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

from cflibs.benchmark.datasets.usgs import OXIDE_TO_ELEMENT_FACTOR

logger = logging.getLogger(__name__)

OXIDE_COLUMNS = ("SiO2", "TiO2", "Al2O3", "FeOT", "MnO", "MgO", "CaO", "Na2O", "K2O")

MIN_PANEL_COVERAGE_WT = 50.0
"""Minimum certified oxide-panel total (wt%) for a standard to be yielded."""

RESOLVING_POWER_HINT = 2000.0

# FeOT = total iron expressed as FeO -> stoichiometric FeO factor.
_FACTORS: Dict[str, tuple[str, float]] = {
    **OXIDE_TO_ELEMENT_FACTOR,
    "FeOT": ("Fe", OXIDE_TO_ELEMENT_FACTOR["FeO"][1]),
}


def _load_compositions(comp_csv: Path) -> Dict[str, dict]:
    """Parse the compositions CSV into ``spectrum_name -> sample dict``."""
    table: Dict[str, dict] = {}
    with open(comp_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Spectrum Name") or "").strip()
            if not name or name in table:  # first occurrence wins (dups identical)
                continue
            oxides: Dict[str, float] = {}
            for oxide in OXIDE_COLUMNS:
                cell = (row.get(oxide) or "").strip()
                try:
                    oxides[oxide] = float(cell)
                except ValueError:
                    continue
            table[name] = {"target": (row.get("Target") or "").strip(), "oxides": oxides}
    return table


def iter_spectra(root: Path) -> Iterator[tuple]:
    """Yield ``SpectrumRecord`` tuples for the ChemCam preflight calibration set."""
    from cflibs.benchmark.adapters_extended import (
        PRESENCE_CUTOFF_WT,
        SpectrumTruth,
        enforce_strictly_increasing,
    )
    from cflibs.benchmark.scoreboard_registry import presence_set

    comp_table = _load_compositions(root / "ccam_calibration_compositions.csv")

    spectra_csv = root / "msl_ccam_libs_calib.csv"
    with open(spectra_csv) as fh:
        column_names = fh.readline().rstrip("\n").split(",")[1:]
    data = np.loadtxt(spectra_csv, delimiter=",", skiprows=1)
    wavelength = data[:, 0]

    truths: Dict[str, SpectrumTruth] = {}
    skipped: set[str] = set()
    occurrence: Dict[str, int] = {}
    for j, name in enumerate(column_names):
        rep = occurrence.get(name, 0)
        occurrence[name] = rep + 1
        if name in skipped:
            continue
        truth = truths.get(name)
        if truth is None:
            sample = comp_table.get(name)
            if sample is None:
                logger.warning(
                    "ChemCam spectrum %r has no composition row in "
                    "ccam_calibration_compositions.csv; skipping its replicates.",
                    name,
                )
                skipped.add(name)
                continue
            coverage = sum(sample["oxides"].values())
            if coverage < MIN_PANEL_COVERAGE_WT:
                logger.warning(
                    "ChemCam standard %r skipped: certified oxide panel covers "
                    "only %.1f wt%% of sample mass.",
                    name,
                    coverage,
                )
                skipped.add(name)
                continue
            element_wt: Dict[str, float] = {}
            for oxide, wt in sample["oxides"].items():
                element, factor = _FACTORS[oxide]
                element_wt[element] = element_wt.get(element, 0.0) + wt * factor
            present = presence_set(element_wt)
            notes = (
                "MSL ChemCam preflight cleanroom calibration spectrum (PDS "
                "MSL-M-CHEMCAM-LIBS-4/5-RDR-V1.0, MSL_CCAM_LIBS_CALIB.CSV, "
                f"column {name!r}, target {sample['target']!r}); radiance in "
                "photons/shot/mm^2/sr/nm at 3 m through 7 Torr CO2 (Wiens et "
                "al. 2013); composition from CCAM_CALIBRATION_COMPOSITIONS.CSV "
                "oxide columns SiO2/TiO2/Al2O3/FeOT/MnO/MgO/CaO/Na2O/K2O "
                f"converted to element wt% (O excluded); certified panel totals "
                f"{coverage:.1f} wt% -- elements outside the panel (S, P, C, "
                f"H2O, trace metals) may appear in the spectrum; presence "
                f"cutoff {PRESENCE_CUTOFF_WT} wt%. Resolving-power hint "
                "2000 (band FWHM about 0.15/0.20/0.65 nm in UV/VIO/VNIR, "
                "Wiens et al. 2012); spectra contain real gaps at 341-382 and "
                "469-473 nm."
            )
            truth = SpectrumTruth(
                elements_present=present,
                composition_wt={el: round(wt, 6) for el, wt in sorted(element_wt.items())},
                composition_basis="element_wt",
                resolving_power=RESOLVING_POWER_HINT,
                notes=notes,
            )
            truths[name] = truth
        wl, inten = enforce_strictly_increasing(wavelength, data[:, j + 1])
        yield f"chemcam/{name}/{rep}", wl, inten, truth
