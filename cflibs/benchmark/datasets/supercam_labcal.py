"""
SuperCam laboratory LIBS calibration database helper (Anderson et al. 2022).

Single-file adapter over the PDS4 delimited table
``data/supercam_calib/raw/labcal/libs_spectral_library_reference.csv``
(721 MB; PDS LID ``...:calibration_supercam:libs_spectral_library_reference::1.1``,
Anderson et al. 2022, doi:10.1016/j.sab.2021.106347) — the spectral library used
to train the Mars-2020 SuperCam Major Oxide Composition (MOC) models.

File format
-----------
8,353 physical lines = 2 header rows + 8,351 data records, 8,060 columns:

- Row 1: column-category tags ``meta`` (44) / ``comp`` (83) / ``wvl`` (7,933).
- Row 2: column names. The 7,933 ``wvl`` names ARE the wavelength axis in nm
  (243.789-852.772, strictly increasing, with the two real inter-spectrometer
  gaps 341.4-379.3 and 464.5-537.6 nm).
- Rows 3+: one record per (acquisition x wavelength-shift variant). Each base
  acquisition appears 7x with ``shift`` in {0, +/-1, +/-2, +/-3} pixels
  (training-set augmentation); **only shift == 0 rows are yielded** (1,193
  base spectra of 334 unique ``Target_Name`` standards).
- The file contains a few latin-1 bytes (degree signs in location strings),
  so it is decoded as latin-1.

Truth notes
-----------
- Major oxides SiO2/TiO2/Al2O3/FeOT/MnO/MgO/CaO/Na2O/K2O/P2O5 in wt% are
  converted to element wt% via the stoichiometric factors in
  :data:`cflibs.benchmark.datasets.usgs.OXIDE_TO_ELEMENT_FACTOR`; FeOT is
  total iron expressed as FeO (FeO factor). The PDS4 label names the K2O
  field ``K20`` (a label typo) while the CSV header row says ``K2O`` — both
  spellings are accepted.
- wt%-reported volatiles: S directly (falling back to SO3T x 0.40046 when S
  is blank), C from C_total (falling back to C, then CO2T/CO2 x 0.27292),
  H from H2O+_structural x 0.11191. ``loss_on_ignition`` /
  ``loss_on_drying`` are bulk measurements, not element-mappable: skipped.
- ppm-reported analytes (N, Cl, F and the ~60 trace elements) are converted
  ppm -> wt% (/1e4) and included only at/above
  :data:`~cflibs.benchmark.datasets._common.PRESENCE_CUTOFF_WT` (0.01 wt% =
  100 ppm). ``Th`` is certified for some standards but is not in
  :data:`cflibs.atomic.masses.STANDARD_ATOMIC_MASSES`, so it can never be a
  pipeline candidate and is excluded.
- Blank cells = unreported -> omitted, never zero. Certified zeros in
  wt%-reported columns are kept in ``composition_wt`` (informative) but
  excluded from ``elements_present`` by the presence cutoff.
- Rows flagged ``Remove_from_all == 'Remove'`` (54 at shift 0; mission-flagged
  gross outliers) are skipped and counted in one summary log line.
- **Pseudo-replication:** multiple observation points (rows) share one
  certified truth row per ``Target_Name``; honest train/test splits MUST
  group by ``Target_Name`` (recorded in every spectrum's notes; see also
  :func:`spectrum_target_names`).

Caveats
-------
Spectra were acquired with SuperCam EQM/FM-representative hardware in a
~7 mbar CO2 Mars-atmosphere chamber ("fmeqm" in source filenames) at
1545-4250 mm standoff; intensities are instrument-response-corrected but in
arbitrary radiometric units — CF-LIBS must use relative line intensities.
The ambient CO2 chamber means C emission appears regardless of sample carbon
content. Resolving power hint 2400 (FWHM about 0.12 nm for the UV and VIO
spectrometers, 0.35 nm for the transmission spectrometer; Manrique et al.
2020, Space Sci. Rev. 216:138, sect. 2.6).
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES
from cflibs.benchmark.datasets._common import (
    PRESENCE_CUTOFF_WT,
    SpectrumTruth,
    enforce_strictly_increasing,
    presence_set,
)
from cflibs.benchmark.datasets.usgs import OXIDE_TO_ELEMENT_FACTOR

logger = logging.getLogger(__name__)

LABCAL_CSV_RELPATH = Path("raw") / "labcal" / "libs_spectral_library_reference.csv"
"""Location of the lab table below the ``data/supercam_calib/`` dataset root."""

CSV_ENCODING = "latin-1"
"""The PDS4 table contains a few non-UTF8 bytes (degree signs in locations)."""

MAJOR_OXIDES = ("SiO2", "TiO2", "Al2O3", "FeOT", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5")

#: The PDS4 label calls the K2O field "K20" (typo); the CSV header says "K2O".
_COMP_NAME_ALIASES = {"K20": "K2O"}

#: The 8 MOC oxides carrying mission *_Train / *_Folds / *_outliers columns.
MOC_SPLIT_OXIDES = ("SiO2", "TiO2", "Al2O3", "FeOT", "MgO", "CaO", "Na2O", "K2O")

# FeOT = total iron expressed as FeO -> stoichiometric FeO factor.
_OXIDE_FACTORS: Dict[str, Tuple[str, float]] = {
    **OXIDE_TO_ELEMENT_FACTOR,
    "FeOT": ("Fe", OXIDE_TO_ELEMENT_FACTOR["FeO"][1]),
}

# Stoichiometric volatile -> element mass-fraction factors (CIAAW 2021 masses).
S_FROM_SO3T = 32.06 / (32.06 + 3 * 15.999)  # 0.40046
C_FROM_CO2 = 12.011 / (12.011 + 2 * 15.999)  # 0.27292
H_FROM_H2O = (2 * 1.008) / (2 * 1.008 + 15.999)  # 0.11191

#: Comp columns handled by dedicated rules (everything else is a ppm column).
_WT_PCT_VOLATILES = ("SO3T", "S", "C_total", "CO2T", "CO2", "C", "H2O+_structural")
_NON_ELEMENT_COLUMNS = ("loss_on_ignition", "loss_on_drying")

RESOLVING_POWER_HINT = 2400.0
"""FWHM 0.12 nm (UV/VIO) and 0.35 nm (TSPEC) per Manrique et al. 2020 sect. 2.6
-> R about 2400/3500/2000 at band centres; one mid-range hint is recorded."""

EXPECTED_META_COLS = 44
EXPECTED_COMP_COLS = 83
EXPECTED_WVL_COLS = 7933

#: Strips the trailing 2-digit chip number: flight and lab SCCT chips of one
#: target come from the same homogeneous batch (e.g. Mars ``SCCT_LBHVO20406``
#: <-> lab ``LBHVO20401`` both -> ``LBHVO204``).
_CHIP_SUFFIX_RE = re.compile(r"\d{2}$")


def chip_base_name(target_name: str) -> str:
    """SCCT target identity ignoring the trailing 2-digit chip number."""
    return _CHIP_SUFFIX_RE.sub("", target_name.strip().upper())


@dataclass(frozen=True)
class _Columns:
    """Resolved column layout of the lab table (validated against the label)."""

    meta: Dict[str, int]
    comp: Dict[str, int]  # canonical names (K20 -> K2O)
    wvl_start: int
    wavelength_nm: np.ndarray


def _parse_header(categories: List[str], names: List[str]) -> _Columns:
    """Resolve the two header rows into a validated column layout."""
    meta: Dict[str, int] = {}
    comp: Dict[str, int] = {}
    wvl_start = -1
    for i, (cat, name) in enumerate(zip(categories, names)):
        if cat == "meta":
            meta[name] = i
        elif cat == "comp":
            comp[_COMP_NAME_ALIASES.get(name, name)] = i
        elif cat == "wvl":
            if wvl_start < 0:
                wvl_start = i
        else:
            raise ValueError(f"Unknown column category {cat!r} at index {i}")
    n_wvl = len(categories) - wvl_start
    if (len(meta), len(comp), n_wvl) != (
        EXPECTED_META_COLS,
        EXPECTED_COMP_COLS,
        EXPECTED_WVL_COLS,
    ):
        raise ValueError(
            f"Unexpected lab-table layout: {len(meta)} meta / {len(comp)} comp / "
            f"{n_wvl} wvl columns (expected {EXPECTED_META_COLS}/"
            f"{EXPECTED_COMP_COLS}/{EXPECTED_WVL_COLS})"
        )
    wavelength_nm = np.array(names[wvl_start:], dtype=float)
    if not np.all(np.diff(wavelength_nm) > 0):
        raise ValueError("Lab-table wavelength header is not strictly increasing")
    return _Columns(meta=meta, comp=comp, wvl_start=wvl_start, wavelength_nm=wavelength_nm)


def _iter_table(csv_path: Path) -> Iterator[Tuple[_Columns, List[str]]]:
    """Stream the lab table one record at a time (never loads the 721 MB file)."""
    with open(csv_path, newline="", encoding=CSV_ENCODING) as fh:
        reader = csv.reader(fh)
        columns = _parse_header(next(reader), next(reader))
        for row in reader:
            yield columns, row


def _cell(row: List[str], section: Dict[str, int], name: str) -> str:
    return row[section[name]].strip()


def _parse_wt(cell: str) -> float | None:
    """Parse one comp cell; blank/unparseable = unreported -> None (never 0)."""
    cell = cell.strip()
    if not cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _sulfur_wt(comp_cells: Dict[str, str]) -> Optional[float]:
    """S wt% from the S column, else converted from SO3T."""
    sulfur = _parse_wt(comp_cells.get("S", ""))
    if sulfur is not None:
        return sulfur
    so3t = _parse_wt(comp_cells.get("SO3T", ""))
    return so3t * S_FROM_SO3T if so3t is not None else None


def _carbon_wt(comp_cells: Dict[str, str]) -> Optional[float]:
    """C wt% from C_total/C, else converted from CO2T/CO2."""
    for column in ("C_total", "C"):
        carbon = _parse_wt(comp_cells.get(column, ""))
        if carbon is not None:
            return carbon
    for column in ("CO2T", "CO2"):
        co2 = _parse_wt(comp_cells.get(column, ""))
        if co2 is not None:
            return co2 * C_FROM_CO2
    return None


def _trace_element_wt(comp_cells: Dict[str, str], handled: set) -> Dict[str, float]:
    """ppm-reported trace analytes -> wt%, cutoff-gated, pipeline elements only."""
    out: Dict[str, float] = {}
    for name, cell in comp_cells.items():
        if name in handled or name not in STANDARD_ATOMIC_MASSES:
            continue
        ppm = _parse_wt(cell)
        if ppm is None:
            continue
        wt = ppm / 1.0e4
        if wt >= PRESENCE_CUTOFF_WT:  # ppm-reported analytes are cutoff-gated
            out[name] = wt
    return out


def comp_row_to_element_wt(comp_cells: Dict[str, str]) -> Dict[str, float]:
    """
    Convert one record's raw comp-column strings to ELEMENT wt% (O excluded).

    This is the single truth-conversion rule shared by the lab-calibration
    adapter and the Mars SCCT adapter (which joins flight targets to these
    lab rows). See the module docstring for the per-column rules.

    Parameters
    ----------
    comp_cells : dict[str, str]
        Canonical comp-column name -> raw CSV cell (may be blank).

    Returns
    -------
    dict[str, float]
        Element symbol -> wt%, rounded to 6 decimals, sorted by symbol.
    """
    element_wt: Dict[str, float] = {}

    for oxide in MAJOR_OXIDES:
        value = _parse_wt(comp_cells.get(oxide, ""))
        if value is None:
            continue
        element, factor = _OXIDE_FACTORS[oxide]
        element_wt[element] = element_wt.get(element, 0.0) + value * factor

    sulfur = _sulfur_wt(comp_cells)
    if sulfur is not None:
        element_wt["S"] = element_wt.get("S", 0.0) + sulfur

    carbon = _carbon_wt(comp_cells)
    if carbon is not None:
        element_wt["C"] = element_wt.get("C", 0.0) + carbon

    water = _parse_wt(comp_cells.get("H2O+_structural", ""))
    if water is not None:
        element_wt["H"] = element_wt.get("H", 0.0) + water * H_FROM_H2O

    handled = set(MAJOR_OXIDES) | set(_WT_PCT_VOLATILES) | set(_NON_ELEMENT_COLUMNS)
    for name, wt in _trace_element_wt(comp_cells, handled).items():
        element_wt[name] = element_wt.get(name, 0.0) + wt

    return {el: round(wt, 6) for el, wt in sorted(element_wt.items())}


def _comp_cells(columns: _Columns, row: List[str]) -> Dict[str, str]:
    return {name: row[idx] for name, idx in columns.comp.items()}


def _row_notes(columns: _Columns, row: List[str], element_wt: Dict[str, float]) -> str:
    """Per-spectrum provenance + stratifier metadata string."""

    def meta(name: str) -> str:
        return _cell(row, columns.meta, name)

    train = ",".join(meta(f"{ox}_Train") or "-" for ox in MOC_SPLIT_OXIDES)
    folds = ",".join(meta(f"{ox}_Folds") or "-" for ox in MOC_SPLIT_OXIDES)
    outliers = [ox for ox in MOC_SPLIT_OXIDES if meta(f"{ox}_outliers").lower() == "remove"]
    panel_total = sum(element_wt.values())
    return (
        "SuperCam lab LIBS calibration spectrum (PDS urn:nasa:pds:mars2020_supercam:"
        "calibration_supercam, libs_spectral_library_reference.csv, file="
        f"{meta('file')!r}, shift=0 of 7 augmentation variants; Anderson et al. 2022, "
        f"doi:10.1016/j.sab.2021.106347). Target_Name={meta('Target_Name')!r} — honest "
        "splits MUST group by Target_Name (multiple observation points share one "
        f"certified truth row). distance_mm={meta('distance_mm') or '?'}; "
        f"SCCT_chip={meta('SCCT')}; composition_type={meta('composition_type') or '?'}; "
        f"composition_source={meta('composition_source') or '?'}. Truth: comp columns -> "
        "element wt% (O excluded): major oxides SiO2..P2O5 (FeOT = total Fe as FeO; "
        "label field 'K20' = K2O), S/C/H from wt% volatile columns, ppm analytes only at "
        f">= {PRESENCE_CUTOFF_WT} wt%; blanks unreported (omitted); certified panel "
        f"totals {panel_total:.1f} wt%. Mission split: Train[{train}]; Folds[{folds}]; "
        f"per-oxide outlier Remove flags: {','.join(outliers) if outliers else 'none'}. "
        "Lab conditions: EQM-representative hardware, ~7 mbar CO2 chamber (ambient C "
        "emission regardless of sample C), response-corrected RELATIVE intensity units. "
        f"Resolving-power hint {RESOLVING_POWER_HINT:.0f} (FWHM 0.12 nm UV/VIO, 0.35 nm "
        "VNIR; Manrique et al. 2020); real spectral gaps at 341.4-379.3 and "
        "464.5-537.6 nm."
    )


def iter_spectra(root: Path) -> Iterator[tuple]:
    """
    Yield ``(spectrum_id, wavelength_nm, intensity, truth)`` for the lab table.

    Streams the 721 MB CSV one record at a time (peak memory ~one record).
    ``spectrum_id`` is the record's ``file`` column (the source FITS name,
    unique across the 1,193 shift==0 records).
    """
    csv_path = root / LABCAL_CSV_RELPATH
    n_yielded = n_removed = n_no_truth = 0
    for columns, row in _iter_table(csv_path):
        if _cell(row, columns.meta, "shift") != "0":
            continue
        if _cell(row, columns.meta, "Remove_from_all").lower() == "remove":
            n_removed += 1
            continue
        element_wt = comp_row_to_element_wt(_comp_cells(columns, row))
        elements_present = presence_set(element_wt)
        if not elements_present:
            n_no_truth += 1
            logger.warning(
                "supercam_labcal: row %r has no scoreable certified analyte; skipping.",
                _cell(row, columns.meta, "file"),
            )
            continue
        intensity = np.array(row[columns.wvl_start :], dtype=float)
        wl, inten = enforce_strictly_increasing(columns.wavelength_nm, intensity)
        truth = SpectrumTruth(
            elements_present=elements_present,
            composition_wt=element_wt,
            resolving_power=RESOLVING_POWER_HINT,
            notes=_row_notes(columns, row, element_wt),
        )
        n_yielded += 1
        yield _cell(row, columns.meta, "file"), wl, inten, truth
    logger.info(
        "supercam_labcal: yielded %d shift==0 spectra (skipped %d Remove_from_all rows, "
        "%d rows without scoreable truth).",
        n_yielded,
        n_removed,
        n_no_truth,
    )


def spectrum_target_names(root: Path) -> Dict[str, str]:
    """
    Map shift==0 ``spectrum_id`` (file column) -> ``Target_Name``.

    The split-grouping helper: rows sharing a ``Target_Name`` share one
    certified truth row, so honest train/test splits must keep them together.
    One streaming pass; the 7,933 spectral cells are never converted.
    """
    csv_path = root / LABCAL_CSV_RELPATH
    groups: Dict[str, str] = {}
    for columns, row in _iter_table(csv_path):
        if _cell(row, columns.meta, "shift") != "0":
            continue
        groups[_cell(row, columns.meta, "file")] = _cell(row, columns.meta, "Target_Name")
    return groups


# ---------------------------------------------------------------------------
# SCCT truth table (shared with cflibs.benchmark.datasets.supercam_scct)
# ---------------------------------------------------------------------------

_SCCT_TRUTH_CACHE: Dict[Tuple[str, int, int], Dict[str, Dict[str, Any]]] = {}


def load_scct_truth_table(root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Truth rows of the SCCT == 'Yes' lab standards, keyed by chip base name.

    Returns ``chip_base_name -> {"target_names": [lab Target_Name, ...],
    "element_wt": {...}, "composition_type": str, "composition_source": str}``.
    Flight SCCT targets join to these rows by :func:`chip_base_name` (flight
    and lab chips of one target are cut from the same homogeneous batch).

    One streaming pass over the lab CSV (spectral cells never converted);
    results are cached per (path, mtime, size).
    """
    csv_path = root / LABCAL_CSV_RELPATH
    stat = csv_path.stat()
    cache_key = (str(csv_path.resolve()), stat.st_mtime_ns, stat.st_size)
    if cache_key in _SCCT_TRUTH_CACHE:
        return _SCCT_TRUTH_CACHE[cache_key]

    table: Dict[str, Dict[str, Any]] = {}
    for columns, row in _iter_table(csv_path):
        if _cell(row, columns.meta, "shift") != "0":
            continue
        if _cell(row, columns.meta, "SCCT").lower() != "yes":
            continue
        target = _cell(row, columns.meta, "Target_Name")
        base = chip_base_name(target)
        element_wt = comp_row_to_element_wt(_comp_cells(columns, row))
        entry = table.get(base)
        if entry is None:
            table[base] = {
                "target_names": [target],
                "element_wt": element_wt,
                "composition_type": _cell(row, columns.meta, "composition_type"),
                "composition_source": _cell(row, columns.meta, "composition_source"),
            }
            continue
        if target not in entry["target_names"]:
            entry["target_names"].append(target)
            if element_wt != entry["element_wt"]:
                logger.warning(
                    "supercam_labcal: SCCT chips %s share base %r but differ in "
                    "certified composition; keeping the first chip's panel.",
                    entry["target_names"],
                    base,
                )
    for entry in table.values():
        entry["target_names"].sort()
    _SCCT_TRUTH_CACHE[cache_key] = table
    return table
