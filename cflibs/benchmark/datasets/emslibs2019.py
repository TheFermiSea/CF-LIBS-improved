"""
EMSLIBS 2019 contest benchmark helper (Vrabel/Kepes et al. 2020).

Files (under ``data/vrabel2020_soil_benchmark/``):

- ``train.h5``: groups ``Wavelengths/1`` (40002 channels, 200-1000 nm,
  strictly increasing), ``Spectra/001..100`` (each ``(40002, 500)`` -- 500
  single-shot spectra per sample) and ``Class/1`` (50000 class labels, 1-12,
  constant within each 500-shot block). Layout documented by the dataset's
  own ``readInH5.py``.
- ``support_tables.xlsx``: sheet ``MIXED_composition`` holds element wt%
  (Al/Ca/Cr/Cu/Fe/K/Mg/Na/Pb/Si/Ti) for all 138 contest samples plus their
  ``Class ID``; sheet ``OREAS`` names the ore type of each class (the samples
  are mixtures of OREAS certified reference ores).
- ``test.h5`` + ``test_labels.csv``: test split, class labels only.

Why presence-only truth
-----------------------
The 100 training samples are a per-class subset of the 138 certificated
samples, but *which* member of each class each ``Spectra/NNN`` block is
cannot be recovered from the published files (verified: the h5 class
sequence matches no contiguous Sample-ID prefix). Per-spectrum quantitative
truth is therefore not derivable without inventing a mapping. Instead each
class gets a presence-only panel: the intersection over all class members of
analytes certified at or above the presence cutoff. Zeros in
``MIXED_composition`` mean *uncertified* (the OREAS sheet shows them as
em-dashes), not absent, so matrix elements such as Si are often missing from
the panel -- the truth is suitable for recall-on-certified-panel scoring.

The test split is not yielded: its labels are class-level too and add no
truth the train split lacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

logger = logging.getLogger(__name__)

ANALYTE_COLUMNS = ("Al", "Ca", "Cr", "Cu", "Fe", "K", "Mg", "Na", "Pb", "Si", "Ti")

SHOTS_PER_SAMPLE_CAP = 3
"""Default number of single-shot spectra yielded per training sample."""


def _class_panels(
    support_xlsx: Path, cutoff_wt: float
) -> tuple[Dict[int, frozenset], Dict[int, str], Dict[int, int]]:
    """
    Build per-class presence panels from ``MIXED_composition``.

    Returns ``(class -> frozenset(elements), class -> ore name,
    class -> member count)``. An element enters a class panel only when every
    member sample of that class certifies it at >= ``cutoff_wt`` wt%.
    """
    import pandas as pd

    comp = pd.read_excel(support_xlsx, sheet_name="MIXED_composition")
    oreas = pd.read_excel(support_xlsx, sheet_name="OREAS")
    ore_names: Dict[int, str] = {}
    for _, row in oreas.dropna(subset=["Class"]).iterrows():
        ore_names.setdefault(int(row["Class"]), str(row["Ore"]))

    panels: Dict[int, frozenset] = {}
    members: Dict[int, int] = {}
    for class_id, group in comp.groupby(comp["Class ID"].astype(int)):
        present = set(ANALYTE_COLUMNS)
        for col in ANALYTE_COLUMNS:
            if (group[col].astype(float) < cutoff_wt).any():
                present.discard(col)
        panels[int(class_id)] = frozenset(present)
        members[int(class_id)] = len(group)
    return panels, ore_names, members


def iter_spectra(root: Path, shots_per_sample: int = SHOTS_PER_SAMPLE_CAP) -> Iterator[tuple]:
    """Yield ``SpectrumRecord`` tuples for the EMSLIBS 2019 train split."""
    import h5py

    from cflibs.benchmark.adapters_extended import PRESENCE_CUTOFF_WT, SpectrumTruth

    panels, ore_names, members = _class_panels(root / "support_tables.xlsx", PRESENCE_CUTOFF_WT)

    truths: Dict[int, SpectrumTruth] = {}
    with h5py.File(root / "train.h5", "r") as h5:
        wl_group = h5["Wavelengths"]
        wavelength = np.asarray(wl_group[next(iter(wl_group))][...], dtype=float).ravel()
        classes = np.asarray(h5["Class"]["1"][...], dtype=int)
        sample_keys = sorted(h5["Spectra"].keys())
        for index, key in enumerate(sample_keys):
            class_id = int(classes[index * 500])
            panel = panels.get(class_id)
            if not panel:
                logger.warning(
                    "EMSLIBS train sample %s (class %d) skipped: empty or missing " "class panel.",
                    key,
                    class_id,
                )
                continue
            truth = truths.get(class_id)
            if truth is None:
                truth = SpectrumTruth(
                    elements_present=panel,
                    composition_wt=None,
                    composition_basis="presence_only",
                    resolving_power=None,
                    notes=(
                        "EMSLIBS 2019 contest train spectrum (Vrabel/Kepes et al. "
                        "2020, train.h5); single-shot, 40002 channels, 200-1000 "
                        f"nm; class {class_id} ({ore_names.get(class_id, 'unknown ore')}); "
                        "presence panel = intersection of analytes certified >= "
                        f"{PRESENCE_CUTOFF_WT} wt% over all {members.get(class_id, 0)} "
                        "class members in support_tables.xlsx sheet "
                        "MIXED_composition (columns Al/Ca/Cr/Cu/Fe/K/Mg/Na/Pb/Si/"
                        "Ti); zeros there mean uncertified, not absent, so matrix "
                        "elements (often Si, O, S) are outside the panel -- use "
                        "for recall-on-panel scoring; per-sample quantitative "
                        "truth is not recoverable from the published files; "
                        "instrument resolving power not stated in the files."
                    ),
                )
                truths[class_id] = truth
            n_shots = min(shots_per_sample, 500)
            block = np.asarray(h5["Spectra"][key][:, :n_shots], dtype=float)
            for shot in range(n_shots):
                yield (
                    f"emslibs2019/train/s{key}/shot{shot}",
                    wavelength,
                    block[:, shot],
                    truth,
                )
