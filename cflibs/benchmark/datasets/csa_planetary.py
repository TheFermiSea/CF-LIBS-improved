"""
CSA (Canadian Space Agency) open planetary-analogue LIBS dataset helper.

Layout (under ``data/csa_planetary_libs/``)::

    LIBSOpenDatacsv.7z                          # source archive (~5.5 MB)
    Sample_Composition_Data_LargeSet.csv        # oxide certs, large set
    Sample_Composition_Data_SubSet.csv          # element certs ("Others" table)
    extracted/
        csv Material Large Set 200pulseaverage/<Name>_200AVG.csv      # 69 spectra
        csv Certified Samples Subset 1000pulseaverage/<Name>_1000AVG.csv  # 41 spectra

Each spectrum CSV is headerless ``wavelength_nm,intensity`` (13490 points,
198.066-970.142 nm, strictly increasing; raw pulse-averaged counts).

Truth
-----
``Sample_Composition_Data_LargeSet.csv`` certifies mass fractions for the
panel Al2O3, CaO, FeO, K2O, MgO, MnO, Na2O, SiO2, TiO2, P2O5 and elemental C.
Values are fractions of 1 except QuartzSARM49's SiO2 (99.6), which can only be
read as wt% (a mass fraction cannot exceed 1) and is divided by 100. Cell
prefixes follow the file's own legend: ``-`` unavailable, ``~`` uncertified
(kept, flagged in notes), ``< x`` below detection (omitted), ``* x`` footnote
value (kept; the footnote concerns Mn speciation in rutile only).

The certified-subset spectra reuse the same table through
:data:`SUBSET_NAME_ALIASES`; ``Graphite`` comes from the element-basis
"Others" row (``Graphite KD2``) of ``Sample_Composition_Data_SubSet.csv``.
``SOre`` has no composition row anywhere and is skipped with a log message.

Truth-quality guards (both logged when they fire):

- *Coverage*: samples whose certified panel covers < 50% of sample mass are
  skipped (e.g. galena, stibnite, barite, gypsum, chalcopyrite -- their major
  constituents are outside the certified panel).
- *Known missing major emitter*: samples whose name declares a mineral whose
  dominant element is uncertified are skipped even when coverage passes:
  fluorite (F), pyrite (S), chromite (Cr), limestone and travertine (C).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

from cflibs.benchmark.datasets.usgs import OXIDE_TO_ELEMENT_FACTOR

logger = logging.getLogger(__name__)

LARGE_SET_DIR = "csv Material Large Set 200pulseaverage"
SUBSET_DIR = "csv Certified Samples Subset 1000pulseaverage"

#: Certified-subset spectrum stem -> large-set composition row name.
#: ``Sediments`` == ``Alumosilicate`` (identical certified values in the
#: dataset's own Sample_Composition_Data.xls subset sheet).
SUBSET_NAME_ALIASES: Dict[str, str] = {
    "Anortho2120": "Anorthosite2120",
    "AnorthoMO11": "AnorthositeMO11",
    "Sediments": "Alumosilicate",
}

#: Mineral keywords whose dominant element is absent from the certified panel.
KNOWN_MISSING_MAJOR: Dict[str, str] = {
    "fluorite": "F",
    "pyrite": "S",
    "chromite": "Cr",
    "limestone": "C",
    "travertine": "C",
}

MIN_PANEL_COVERAGE = 0.5
"""Minimum certified mass fraction (oxides + C) for a sample to be yielded."""

_HAND_SAMPLE_RE = re.compile(r"^(hand sample\d+)\b")

# Local extension of the shared oxide->element table: elemental carbon passes
# through unchanged (factor 1.0).
_FACTORS: Dict[str, tuple[str, float]] = {**OXIDE_TO_ELEMENT_FACTOR, "C": ("C", 1.0)}


def _parse_cell(cell: str) -> tuple[Optional[float], bool]:
    """
    Parse one composition cell into ``(mass_fraction, uncertified_flag)``.

    Returns ``(None, False)`` for unavailable (``-``), empty, or
    below-detection (``< x``) cells. ``~``-prefixed values are returned with
    the uncertified flag set; ``*``-prefixed footnote values are kept.
    Values > 1.5 are wt% (only QuartzSARM49's SiO2) and are divided by 100.
    """
    text = cell.strip()
    if not text or text == "-" or text.startswith("<"):
        return None, False
    uncertified = text.startswith("~")
    text = text.lstrip("~*").strip()
    try:
        value = float(text)
    except ValueError:
        return None, False
    if value > 1.5:  # wt% slipped into a mass-fraction column
        value /= 100.0
    return value, uncertified


def _load_large_set_table(comp_csv: Path) -> Dict[str, dict]:
    """Parse the large-set oxide certificate CSV into ``name -> sample dict``."""
    table: Dict[str, dict] = {}
    with open(comp_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        analytes = [col.strip() for col in header[1:]]
        for row in reader:
            name = row[0].strip()
            if not name or name.startswith("*"):
                continue
            oxides: Dict[str, float] = {}
            uncertified: list[str] = []
            for analyte, cell in zip(analytes, row[1:]):
                value, flag = _parse_cell(cell)
                if value is None:
                    continue
                oxides[analyte] = value
                if flag:
                    uncertified.append(analyte)
            table[name] = {"oxides": oxides, "uncertified": uncertified}
    return table


def _load_graphite_row(subset_csv: Path) -> Optional[dict]:
    """Parse the element-basis ``Graphite KD2`` row of the SubSet CSV."""
    if not subset_csv.is_file():
        return None
    with open(subset_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        elements = [col.strip() for col in header[1:]]
        for row in reader:
            if row and row[0].strip().startswith("Graphite"):
                element_wt: Dict[str, float] = {}
                for element, cell in zip(elements, row[1:]):
                    value, _ = _parse_cell(cell)
                    if value is not None:
                        element_wt[element] = value * 100.0
                return {"name": row[0].strip(), "element_wt": element_wt}
    return None


def _resolve_comp_name(stem: str, table: Dict[str, dict]) -> Optional[str]:
    """Map a spectrum-file stem to its composition-table row name."""
    stem = SUBSET_NAME_ALIASES.get(stem, stem)
    if stem in table:
        return stem
    match = _HAND_SAMPLE_RE.match(stem)
    if match:
        prefix = match.group(1)
        for name in table:
            if _HAND_SAMPLE_RE.match(name) and name.split()[1] == prefix.split()[1]:
                return name
    return None


def _oxides_to_elements(oxides: Dict[str, float]) -> tuple[Dict[str, float], float]:
    """Convert oxide mass fractions to element wt%; also return panel coverage."""
    element_wt: Dict[str, float] = {}
    coverage = 0.0
    for analyte, fraction in oxides.items():
        coverage += fraction
        if analyte not in _FACTORS:
            continue
        element, factor = _FACTORS[analyte]
        element_wt[element] = element_wt.get(element, 0.0) + fraction * 100.0 * factor
    return element_wt, coverage


def iter_spectra(root: Path) -> Iterator[tuple]:
    """Yield ``SpectrumRecord`` tuples for the CSA dataset (see contract)."""
    from cflibs.benchmark.adapters_extended import (
        PRESENCE_CUTOFF_WT,
        SpectrumTruth,
        enforce_strictly_increasing,
    )

    table = _load_large_set_table(root / "Sample_Composition_Data_LargeSet.csv")
    graphite = _load_graphite_row(root / "Sample_Composition_Data_SubSet.csv")

    sets = (
        ("large200", root / "extracted" / LARGE_SET_DIR, "_200AVG"),
        ("subset1000", root / "extracted" / SUBSET_DIR, "_1000AVG"),
    )
    for set_id, directory, suffix in sets:
        if not directory.is_dir():
            logger.warning("CSA set %r missing at %s; skipping.", set_id, directory)
            continue
        for path in sorted(directory.glob("*.csv")):
            stem = path.stem.replace(suffix, "")
            if stem == "Graphite" and graphite is not None:
                element_wt = dict(graphite["element_wt"])
                comp_name = graphite["name"]
                uncertified: list[str] = []
                coverage = sum(element_wt.values()) / 100.0
                source = "Sample_Composition_Data_SubSet.csv (element basis)"
            else:
                comp_name = _resolve_comp_name(stem, table)
                if comp_name is None:
                    logger.warning("CSA spectrum %s has no composition row; skipping.", path.name)
                    continue
                sample = table[comp_name]
                element_wt, coverage = _oxides_to_elements(sample["oxides"])
                uncertified = sample["uncertified"]
                source = "Sample_Composition_Data_LargeSet.csv (oxide basis)"

            lowered = comp_name.lower()
            missing_major = next(
                ((kw, el) for kw, el in KNOWN_MISSING_MAJOR.items() if kw in lowered),
                None,
            )
            if missing_major is not None:
                logger.warning(
                    "CSA sample %r skipped: major element %s of %s is not in the "
                    "certified panel.",
                    comp_name,
                    missing_major[1],
                    missing_major[0],
                )
                continue
            if coverage < MIN_PANEL_COVERAGE:
                logger.warning(
                    "CSA sample %r skipped: certified panel covers only %.1f%% of " "sample mass.",
                    comp_name,
                    coverage * 100.0,
                )
                continue

            present = frozenset(el for el, wt in element_wt.items() if wt >= PRESENCE_CUTOFF_WT)
            if not present:
                logger.warning("CSA sample %r skipped: empty element panel.", comp_name)
                continue

            data = np.loadtxt(path, delimiter=",")
            wl, inten = enforce_strictly_increasing(data[:, 0], data[:, 1])
            notes = (
                "CSA open planetary-analogue LIBS dataset (LIBSOpenDatacsv.7z, "
                f"{set_id} set, file {path.name!r}); composition row "
                f"{comp_name!r} from {source}, oxide panel Al2O3/CaO/FeO/K2O/"
                "MgO/MnO/Na2O/SiO2/TiO2/P2O5 + elemental C converted to element "
                f"wt% (O excluded); certified panel covers {coverage * 100.0:.1f}% "
                f"of sample mass -- elements outside the panel (e.g. S, Cu, Zn, "
                f"volatiles) may appear in the spectrum; presence cutoff "
                f"{PRESENCE_CUTOFF_WT} wt%; intensities are raw "
                f"{'200' if set_id == 'large200' else '1000'}-pulse averages; "
                "instrument resolving power not stated in the dataset files."
            )
            if uncertified:
                notes += f" Uncertified (~) analytes kept: {sorted(uncertified)}."
            truth = SpectrumTruth(
                elements_present=present,
                composition_wt={el: round(wt, 6) for el, wt in sorted(element_wt.items())},
                composition_basis="element_wt",
                resolving_power=None,
                notes=notes,
            )
            yield f"csa/{set_id}/{stem}", wl, inten, truth
