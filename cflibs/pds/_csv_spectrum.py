"""Shared metadata-CSV parsing for PDS LIBS data products.

ChemCam CL5 and SuperCam calibrated products share an identical
header/data CSV layout: a block of ``#``- or ``"``-prefixed (or ``key = value``)
metadata header lines, an optional column-header row, then numeric
``wavelength,intensity`` rows. The two instruments differ only in the
wavelength gap that separates their three spectrometers (ChemCam's
smallest gap is ~5 nm; SuperCam's VIO->VNIR gap is ~70 nm), which is
exposed here as the single ``gap_threshold_nm`` parameter.

This module concentrates that parse behind one deep interface,
:func:`parse_metadata_csv`, so the instrument adapters
(``cflibs.pds.chemcam`` / ``cflibs.pds.supercam``) reduce to thin wrappers
that supply their gap threshold and instrument-specific sol extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ParsedPdsSpectrum:
    """Result of parsing a PDS metadata-CSV spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array in nm (concatenated across all spectrometers).
    intensity : np.ndarray
        Calibrated intensity values, aligned with ``wavelength``.
    metadata : Dict
        Header-derived metadata (``key = value`` pairs from the header block,
        with values coerced to ``int``/``float`` where possible).
    spectrometer_ranges : List[Tuple[int, int]]
        Index ranges ``(start, end)`` for each spectrometer in the
        concatenated arrays, derived from wavelength gaps.
    """

    wavelength: np.ndarray
    intensity: np.ndarray
    metadata: Dict = field(default_factory=dict)
    spectrometer_ranges: List[Tuple[int, int]] = field(default_factory=list)


def _parse_header_line(line: str, metadata: Dict) -> None:
    """Extract a ``key = value`` metadata pair from one header line."""
    clean = line.lstrip("#").strip().strip('"')
    if "=" in clean:
        key, _, val = clean.partition("=")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip().strip('"')
        try:
            metadata[key] = int(val)
        except ValueError:
            try:
                metadata[key] = float(val)
            except ValueError:
                metadata[key] = val


def _parse_header(lines: List[str], metadata: Dict) -> int:
    """Parse header lines into ``metadata`` and return the data start index.

    Lines starting with ``#`` or ``"`` are metadata header lines. The first
    comma-separated line whose first field parses as a float marks the start
    of the numeric data; a non-numeric comma-separated line is treated as a
    column-header row.
    """
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            _parse_header_line(stripped, metadata)
            data_start = i + 1
        elif "," in stripped:
            parts = stripped.split(",")
            try:
                float(parts[0])
                data_start = i
                break
            except ValueError:
                data_start = i + 1
        else:
            data_start = i + 1
    return data_start


def _parse_numeric_data(data_lines: List[str]) -> Tuple[List[float], List[float]]:
    """Parse numeric ``wavelength,intensity`` rows from the data section."""
    wavelengths: List[float] = []
    intensities: List[float] = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(",")
        if len(parts) >= 2:
            try:
                wl = float(parts[0])
                intensity = float(parts[1])
                wavelengths.append(wl)
                intensities.append(intensity)
            except ValueError:
                continue
    return wavelengths, intensities


def _find_spectrometer_ranges(
    wavelength: np.ndarray, gap_threshold_nm: float
) -> List[Tuple[int, int]]:
    """Identify spectrometer boundaries from wavelength gaps.

    A gap larger than ``gap_threshold_nm`` between consecutive wavelengths is
    treated as the boundary between two spectrometers.
    """
    if len(wavelength) < 2:
        return [(0, len(wavelength))]

    diffs = np.diff(wavelength)
    gap_indices = np.where(diffs > gap_threshold_nm)[0]

    ranges = []
    start = 0
    for gap_idx in gap_indices:
        ranges.append((start, gap_idx + 1))
        start = gap_idx + 1
    ranges.append((start, len(wavelength)))

    return ranges


def parse_metadata_csv(text: str, *, gap_threshold_nm: float) -> ParsedPdsSpectrum:
    """Parse a PDS metadata-CSV spectrum from raw file text.

    This is the single deep parse shared by the ChemCam and SuperCam
    adapters. It performs header detection, numeric-row parsing, and
    spectrometer-gap detection; the only instrument-dependent input is
    ``gap_threshold_nm``.

    Parameters
    ----------
    text : str
        Raw file contents (will be stripped and split on newlines).
    gap_threshold_nm : float
        Minimum wavelength gap (nm) treated as a spectrometer boundary.

    Returns
    -------
    ParsedPdsSpectrum
        Parsed wavelengths, intensities, metadata, and spectrometer ranges.
        ``wavelength``/``intensity`` are empty arrays if no numeric rows are
        found; callers decide how to handle that (e.g. raise ``ValueError``).
    """
    lines = text.strip().split("\n")

    metadata: Dict = {}
    data_start = _parse_header(lines, metadata)
    wavelengths, intensities = _parse_numeric_data(lines[data_start:])

    wl_arr = np.array(wavelengths)
    int_arr = np.array(intensities)
    spec_ranges = _find_spectrometer_ranges(wl_arr, gap_threshold_nm)

    return ParsedPdsSpectrum(
        wavelength=wl_arr,
        intensity=int_arr,
        metadata=metadata,
        spectrometer_ranges=spec_ranges,
    )
