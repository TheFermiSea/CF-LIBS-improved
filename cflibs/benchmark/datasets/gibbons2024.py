"""
Gibbons et al. nitrate-in-Mars-simulant LIBS helper (N quantitation).

File (under ``data/gibbons2024_nitrogen_libs/``): ``SI_Raw_Spectral_Data.xlsx``
(supporting information of Gibbons et al., corresponding author Erin Gibbons,
McGill). Sheet ``Nitrates_UnnormalizedSpectra`` holds 180 spectra: columns
``File name``, ``wt. % NO3- ion``, ``Nitrate Salt Cation``, ``Scan
Replicate``, then 12288 wavelength columns (185.678-1048.998 nm; a handful of
duplicated channel-boundary wavelengths are deduplicated). Per the in-file
ReadMe: intensities are arbitrary-unit accumulations of 100 consecutive
shots; the 213 nm Nd:YAG laser line is recorded in-range; atmosphere is He.

Rows fall into three groups:

- **Mixtures** (150 rows): nitrate salt (cation Ca/Fe/K/Mg/Na) doped into
  Mars Global Simulant at NO3- ion levels 0.5-30 wt%. Quantitative truth:
  ``N wt% = (wt% NO3-) * M_N / M_NO3 = wt% * 0.225905`` -- a pure
  stoichiometric identity, independent of salt hydration state. Element
  panel: {N, cation}.
- **Pure-salt end-members** (25 rows, wt column reads "100 (Pure Salt
  End-member)"): presence-only truth {N, cation} -- the NO3- mass fraction
  of the pure salt depends on the (unreported) hydration state, so no
  quantitative value is derived.
- **MGS blanks** (5 rows, "0 (Mars Global Simulant)"): skipped with a log
  message -- the simulant's composition is not in the file, so no truth is
  derivable.

The matrix (MGS) composition is uncertified throughout, so mixture panels are
partial: matrix elements (Si, Fe, Mg, Al, Ca, O, ...) will appear in the
spectra. Recorded in notes; suitable for recall-on-panel and N-quantitation
scoring. The ``ClayMineral_UnnormalizedSpectra`` sheet (ammoniated clays) is
*not* adapted: its truth is a mineral/ammoniation label whose elemental
composition is not in the file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

SHEET = "Nitrates_UnnormalizedSpectra"

CATIONS = frozenset({"Ca", "Fe", "K", "Mg", "Na"})

# M_N / M_NO3 = 14.007 / (14.007 + 3 * 15.999) with CIAAW 2021 masses.
NO3_TO_N = 14.007 / (14.007 + 3 * 15.999)


def iter_spectra(root: Path) -> Iterator[tuple]:
    """Yield ``SpectrumRecord`` tuples for the Gibbons nitrate series."""
    import openpyxl

    from cflibs.benchmark.adapters_extended import (
        SpectrumTruth,
        enforce_strictly_increasing,
    )

    workbook = openpyxl.load_workbook(
        root / "SI_Raw_Spectral_Data.xlsx", read_only=True, data_only=True
    )
    try:
        sheet = workbook[SHEET]
        rows = sheet.iter_rows(values_only=True)
        next(rows)  # banner row
        header = next(rows)
        wl_columns = [i for i, v in enumerate(header) if i >= 4 and isinstance(v, (int, float))]
        wavelength = np.array([header[i] for i in wl_columns], dtype=float)

        common_notes = (
            "Gibbons et al. nitrate-doped Mars Global Simulant LIBS spectrum "
            "(SI_Raw_Spectral_Data.xlsx, sheet Nitrates_UnnormalizedSpectra, "
            "row {fname!r}); 100-shot accumulated raw intensity, He atmosphere, "
            "185.7-1049.0 nm (duplicate channel-boundary wavelengths "
            "deduplicated); the 213 nm Nd:YAG laser line is recorded in-range; "
            "matrix = Mars Global Simulant, composition not in file, so the "
            "element panel is partial (matrix Si/Fe/Mg/Al/Ca/O/... will appear "
            "in the spectrum) -- recall-on-panel scoring only; instrument "
            "resolving power not stated in the file. {truth_detail}"
        )

        for row in rows:
            fname = row[0]
            if fname is None:
                continue
            wt_no3, cation = row[1], row[2]
            intensities = np.array(
                [
                    row[i] if i < len(row) and isinstance(row[i], (int, float)) else np.nan
                    for i in wl_columns
                ],
                dtype=float,
            )
            if cation not in CATIONS:
                logger.warning(
                    "Gibbons row %r skipped: no nitrate cation (matrix blank or "
                    "unrecognized cation %r).",
                    fname,
                    cation,
                )
                continue
            if isinstance(wt_no3, (int, float)) and wt_no3 > 0:
                n_wt = float(wt_no3) * NO3_TO_N
                truth_detail = (
                    f"Quantitative N: 'wt. % NO3- ion' column = {float(wt_no3)} "
                    f"wt% NO3-, x M_N/M_NO3 ({NO3_TO_N:.6f}) = {n_wt:.4f} wt% N "
                    "(stoichiometric identity, hydration-independent); cation "
                    f"{cation} certified present via the 'Nitrate Salt Cation' "
                    "column (its wt% depends on uncertified hydration state and "
                    "matrix content, so it is presence-only)."
                )
                truth = SpectrumTruth(
                    elements_present=frozenset({"N", str(cation)}),
                    composition_wt={"N": round(n_wt, 6)},
                    composition_basis="element_wt",
                    resolving_power=None,
                    notes=common_notes.format(fname=fname, truth_detail=truth_detail),
                )
            else:
                # "100 (Pure Salt End-member)" rows: presence-only.
                truth_detail = (
                    f"Pure {cation}-nitrate salt end-member ('wt. % NO3- ion' "
                    f"column reads {wt_no3!r}); NO3- mass fraction of the pure "
                    "salt depends on the unreported hydration state, so truth "
                    "is presence-only {N, cation}."
                )
                truth = SpectrumTruth(
                    elements_present=frozenset({"N", str(cation)}),
                    composition_wt=None,
                    composition_basis="presence_only",
                    resolving_power=None,
                    notes=common_notes.format(fname=fname, truth_detail=truth_detail),
                )
            wl, inten = enforce_strictly_increasing(wavelength, intensities)
            yield f"gibbons2024/{fname}", wl, inten, truth
    finally:
        workbook.close()
