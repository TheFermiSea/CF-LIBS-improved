"""
Silva et al. 2022 tropical-soil LIBS dataset helper.

Files (under ``data/silva2022_tropical_soils/``):

- ``LIBS_data.txt``: tab-separated; row 1 = sample IDs (1-102), row 2 =
  field number (1 or 2), remaining 53717 rows = ``wavelength_nm`` followed by
  one raw intensity per sample. Wavelengths run 200.005-779.992 nm and are
  strictly increasing.
- ``soil_fertility_data.txt``: tab-separated per-sample fertility panel with
  columns ``ID, Field, Clay, OM, CEC, pH, V, exP, exK, exCa, exMg``.

Truth
-----
Presence-only. The fertility panel certifies *exchangeable/extractable*
concentrations -- exP in mg/dm3; exK/exCa/exMg in mmolc/dm3 (standard
Brazilian soil-fertility protocol units) -- which cannot be converted to
element wt% without the unreported soil bulk density, so no quantitative
truth is emitted. A positive exchangeable concentration nevertheless
certifies the element is present, giving the panel {P, K, Ca, Mg}.

Caveats recorded in notes:

- Column ``V`` is base saturation (%), **not** vanadium.
- The panel is partial: tropical-soil matrix elements (Si, Al, Fe, Ti, O,
  C, ...) are uncertified, so the truth supports recall-on-panel scoring
  only.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

from cflibs.benchmark.datasets._common import SpectrumTruth, enforce_strictly_increasing

logger = logging.getLogger(__name__)

ANALYTE_TO_ELEMENT = {"exP": "P", "exK": "K", "exCa": "Ca", "exMg": "Mg"}


def _load_fertility(fert_txt: Path) -> Dict[int, dict]:
    """Parse the fertility table into ``sample_id -> row dict``."""
    table: Dict[int, dict] = {}
    with open(fert_txt, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                sample_id = int(row["ID"])
            except (KeyError, ValueError):
                continue
            table[sample_id] = row
    return table


def iter_spectra(root: Path) -> Iterator[tuple]:
    """Yield ``SpectrumRecord`` tuples for the Silva 2022 tropical soils."""
    fertility = _load_fertility(root / "soil_fertility_data.txt")

    with open(root / "LIBS_data.txt") as fh:
        id_row = fh.readline().rstrip("\n").split("\t")
        field_row = fh.readline().rstrip("\n").split("\t")
        raw = np.loadtxt(fh, delimiter="\t", dtype=float)
    sample_ids = [int(v) for v in id_row[1:]]
    fields = [int(v) for v in field_row[1:]]
    wavelength = raw[:, 0]

    for col, (sample_id, field_id) in enumerate(zip(sample_ids, fields), start=1):
        row = fertility.get(int(sample_id))
        if row is None:
            logger.warning(
                "Silva 2022 sample %d has no soil_fertility_data.txt row; skipping.",
                sample_id,
            )
            continue
        present = set()
        values: Dict[str, str] = {}
        for analyte, element in ANALYTE_TO_ELEMENT.items():
            try:
                value = float(row[analyte])
            except (KeyError, TypeError, ValueError):
                continue
            values[analyte] = row[analyte]
            if value > 0.0:
                present.add(element)
        if not present:
            logger.warning(
                "Silva 2022 sample %d skipped: no positive fertility analytes.",
                sample_id,
            )
            continue
        wl, inten = enforce_strictly_increasing(wavelength, raw[:, col])
        truth = SpectrumTruth(
            elements_present=frozenset(present),
            composition_wt=None,
            resolving_power=None,
            notes=(
                "Silva et al. 2022 tropical-soil LIBS spectrum (LIBS_data.txt, "
                f"sample ID {int(sample_id)}, field {int(field_id)}); raw "
                "intensities, 200-780 nm; presence truth from "
                "soil_fertility_data.txt exchangeable panel "
                f"exP={values.get('exP')} mg/dm3, exK={values.get('exK')}, "
                f"exCa={values.get('exCa')}, exMg={values.get('exMg')} mmolc/dm3 "
                "-- exchangeable concentrations are not convertible to wt% "
                "without bulk density, hence presence_only; column 'V' is base "
                "saturation (%), not vanadium; matrix elements (Si, Al, Fe, Ti, "
                "O, C, ...) are uncertified -- recall-on-panel scoring only; "
                "instrument resolving power not stated in the files."
            ),
        )
        yield f"silva2022/field{int(field_id)}/sample{int(sample_id)}", wl, inten, truth
