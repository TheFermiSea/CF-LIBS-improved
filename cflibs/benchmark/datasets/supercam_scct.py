"""
SuperCam Calibration Target (SCCT) Mars-surface LIBS spectra helper.

Adapter over the 547 calibrated LIBS products (CL1 FITS + PDS4 labels) under
``data/supercam_calib/raw/scct/cl1/sol_XXXXX/`` — every SuperCam LIBS
observation of the onboard calibration targets, 23 distinct targets across
sols 13-1694 (PDS bundle ``urn:nasa:pds:mars2020_supercam``,
``data_calibrated_spectra`` collection). These are **real Mars** spectra:
true Mars atmosphere and dust at the ~3 m fixed SCCT standoff.

FITS layout (per the SuperCam EDR/RDR SIS)
------------------------------------------
- ``WAVELENGTH`` HDU: columns ``Wavelength`` (nm, 7,933 channels, identical
  axis to the lab table) and ``IRF`` (instrument response).
- ``SPECTRA`` HDU: up to 10 columns ``Spectrum0..Spectrum9`` — the burst is
  split into shot groups (some products carry fewer columns). The adapter
  yields the per-channel **mean over the available sub-spectra**; the count
  is recorded in the notes.
- ``SATURATION`` HDU: per-channel, per-sub-spectrum saturation flags. Values
  are **kept, not masked** (NaN axes break the pipeline); the flagged
  fraction is recorded in the notes.

Truth join (Manrique et al. 2020; Anderson et al. 2022)
-------------------------------------------------------
Flight target names (FITS ``TARGETNM``, e.g. ``SCCT_LBHVO20406``) are
stripped of the ``SCCT_`` prefix and matched to lab-table ``Target_Name``
ignoring the trailing 2-digit chip number
(:func:`cflibs.benchmark.datasets.supercam_labcal.chip_base_name`): flight
and lab chips are cut from the same homogeneous material batch (verified for
all 23 LIBS-observed flight targets except TITANIUM). Truth VALUES come from
the lab CSV comp columns via the shared converter
:func:`cflibs.benchmark.datasets.supercam_labcal.comp_row_to_element_wt`.

``SCCT_TITANIUM`` has no lab-table row: it is the Ti6Al4V wavelength-
calibration plate (Manrique et al. 2020, Space Sci. Rev. 216:138, sect. 2.6).
It is yielded **presence-only** with the alloy's nominal constituents
{Ti, Al, V} — no certified quantitative panel exists. (Note the TAPAG/TAPAX
targets, sometimes assumed to be metal plates, are fluoro-chloro-hydro
apatite chips with full EMPA truth in the lab table.)

Requires :mod:`astropy` (``pip install cflibs[fits]``); the adapter wrapper
skips with a log message when it is missing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from cflibs.benchmark.datasets._common import (
    PRESENCE_CUTOFF_WT,
    SpectrumTruth,
    enforce_strictly_increasing,
    presence_set,
)
from cflibs.benchmark.datasets.supercam_labcal import (
    RESOLVING_POWER_HINT,
    chip_base_name,
    load_scct_truth_table,
)

logger = logging.getLogger(__name__)

CL1_RELPATH = Path("raw") / "scct" / "cl1"
"""CL1 product tree below the ``data/supercam_calib/`` dataset root."""

#: Nominal constituents of the Ti6Al4V wavelength-calibration plate
#: (Manrique et al. 2020 sect. 2.6) — presence-only, no certified panel.
TITANIUM_PRESENCE = frozenset({"Ti", "Al", "V"})

#: CL1 filename: scam_<sol>_<sclk>_<ms>_cl1_<seqid>_<target>___<point>p<version>
_CL1_STEM_RE = re.compile(
    r"^scam_(?P<sol>\d+)_(?P<sclk>\d+)_(?P<ms>\d+)_cl1_(?P<seqid>[a-z0-9]+)_"
    r"(?P<target>.+?)_*(?P<point>\d+)p(?P<version>\d+)$"
)


def _parse_stem(stem: str) -> Dict[str, str]:
    match = _CL1_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Unrecognized CL1 filename {stem!r}")
    return match.groupdict()


def truth_for_flight_target(
    flight_target: str,
    truth_table: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[SpectrumTruth, str]]:
    """
    Build the (truth, lab-provenance) pair for one flight SCCT target name.

    ``flight_target`` is the FITS ``TARGETNM`` (e.g. ``SCCT_LBHVO20406``).
    Returns ``None`` when no defensible truth exists (unknown target);
    TITANIUM gets the presence-only Ti6Al4V panel.
    """
    name = flight_target.strip().upper()
    if name.startswith("SCCT_"):
        name = name[len("SCCT_") :]
    if name == "TITANIUM":
        truth = SpectrumTruth(
            elements_present=TITANIUM_PRESENCE,
            composition_wt=None,
            resolving_power=RESOLVING_POWER_HINT,
            notes=(
                "Ti6Al4V wavelength-calibration plate (Manrique et al. 2020, Space "
                "Sci. Rev. 216:138, sect. 2.6); no lab-table row and no certified "
                "quantitative panel — presence-only truth from the nominal alloy "
                "constituents Ti/Al/V."
            ),
        )
        return truth, "Ti6Al4V plate (no lab chip row)"
    entry = truth_table.get(chip_base_name(name))
    if entry is None:
        return None
    element_wt = dict(entry["element_wt"])
    elements_present = presence_set(element_wt)
    if not elements_present:
        return None
    provenance = (
        f"lab chip(s) {', '.join(entry['target_names'])} "
        f"(type={entry['composition_type'] or '?'}, "
        f"source={entry['composition_source'] or '?'})"
    )
    truth = SpectrumTruth(
        elements_present=elements_present,
        composition_wt=element_wt,
        resolving_power=RESOLVING_POWER_HINT,
        notes=provenance,  # replaced by the full per-spectrum notes at yield time
    )
    return truth, provenance


def _mean_subspectra(spectra_hdu: Any) -> Tuple[np.ndarray, int, int]:
    """Mean over the available ``Spectrum*`` columns, never producing NaN.

    Returns ``(mean_intensity, n_columns_used, n_channels_without_data)``;
    channels with no finite value in any sub-spectrum are set to 0.0.
    """
    columns = [c for c in spectra_hdu.columns.names if c.startswith("Spectrum")]
    stack = np.vstack([np.asarray(spectra_hdu.data[c], dtype=float) for c in columns])
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stack, 0.0).sum(axis=0)
    mean = sums / np.maximum(counts, 1)
    n_empty = int(np.sum(counts == 0))
    return mean, len(columns), n_empty


def _saturation_summary(hdul: Any) -> Tuple[float, int]:
    """(flagged-cell fraction, channels flagged in >=1 sub-spectrum) or zeros."""
    if "SATURATION" not in hdul:
        return 0.0, 0
    data = hdul["SATURATION"].data
    masks = np.vstack(
        [np.asarray(data[c]) != 0 for c in data.columns.names if c.startswith("SaturationMask")]
    )
    if masks.size == 0:
        return 0.0, 0
    return float(masks.mean()), int(masks.any(axis=0).sum())


def iter_spectra(root: Path) -> Iterator[tuple]:
    """
    Yield ``(spectrum_id, wavelength_nm, intensity, truth)`` for the CL1 set.

    ``root`` is the ``data/supercam_calib/`` dataset root (must contain both
    ``raw/scct/cl1/`` and the lab CSV, which supplies the truth values).
    ``spectrum_id`` is the CL1 filename stem. Files are processed in sorted
    order; unreadable or truth-less products are skipped with a log message.
    """
    from astropy.io import fits

    truth_table = load_scct_truth_table(root)
    paths = sorted(p for p in (root / CL1_RELPATH).glob("sol_*/*.fits") if "_cl1_" in p.name)
    n_yielded = n_skipped = 0
    for path in paths:
        try:
            record = _read_product(fits, path, truth_table)
        except Exception as exc:  # noqa: BLE001 — one bad product must not kill the run
            logger.warning("supercam_scct: failed to read %s: %s", path.name, exc)
            n_skipped += 1
            continue
        if record is None:
            n_skipped += 1
            continue
        n_yielded += 1
        yield record
    logger.info(
        "supercam_scct: yielded %d of %d CL1 products (%d skipped).",
        n_yielded,
        len(paths),
        n_skipped,
    )


def _read_product(fits: Any, path: Path, truth_table: Dict[str, Dict[str, Any]]) -> Optional[tuple]:
    """Parse one CL1 FITS product into an adapter record (None = skip, logged)."""
    stem = path.stem
    parts = _parse_stem(stem)
    with fits.open(path) as hdul:
        header = hdul[0].header
        flight_target = str(header.get("TARGETNM", "")).strip() or parts["target"].upper()
        joined = truth_for_flight_target(flight_target, truth_table)
        if joined is None:
            logger.warning(
                "supercam_scct: %s target %r has no lab-table truth; skipping.",
                stem,
                flight_target,
            )
            return None
        base_truth, lab_provenance = joined
        wavelength = np.asarray(hdul["WAVELENGTH"].data["Wavelength"], dtype=float)
        intensity, n_cols, n_empty = _mean_subspectra(hdul["SPECTRA"])
        sat_fraction, sat_channels = _saturation_summary(hdul)

    wl, inten = enforce_strictly_increasing(wavelength, intensity)
    notes = (
        "Mars-surface SuperCam SCCT LIBS spectrum (PDS urn:nasa:pds:mars2020_supercam "
        f"data_calibrated_spectra CL1 product {stem!r}; sol {int(parts['sol'])}, target "
        f"{flight_target}, point {int(parts['point'])}); intensity = mean of {n_cols} "
        "Spectrum0..9 sub-spectra (shot groups, dark-subtracted, denoised, "
        "IRF-corrected, RELATIVE units)"
        + (f"; {n_empty} channels had no finite sub-spectrum value (set to 0)" if n_empty else "")
        + f"; saturation flags: {sat_fraction:.4f} of channel/shot cells, {sat_channels} "
        "channels flagged in >=1 sub-spectrum (values KEPT, not masked). Truth: "
        + (
            f"joined to SuperCam lab calibration table {lab_provenance} via chip base name "
            "(flight/lab chips share one material batch; Anderson et al. 2022 "
            "libs_spectral_library_reference.csv comp columns -> element wt%, O excluded; "
            f"ppm analytes only at >= {PRESENCE_CUTOFF_WT} wt%)."
            if base_truth.composition_wt is not None
            else base_truth.notes
        )
        + " True Mars atmosphere/dust at ~3 m fixed SCCT standoff; CL1 averaging already "
        f"discards first dust-sampling shots per SIS. Resolving-power hint "
        f"{RESOLVING_POWER_HINT:.0f} (FWHM 0.12 nm UV/VIO, 0.35 nm VNIR; Manrique et al. "
        "2020); real spectral gaps at 341.4-379.3 and 464.5-537.6 nm."
    )
    truth = SpectrumTruth(
        elements_present=base_truth.elements_present,
        composition_wt=base_truth.composition_wt,
        resolving_power=base_truth.resolving_power,
        notes=notes,
    )
    return stem, wl, inten, truth
