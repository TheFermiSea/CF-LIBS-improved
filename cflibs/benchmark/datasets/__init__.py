"""
External benchmark dataset adapters.

This sub-package hosts thin loaders for community LIBS benchmark datasets that
plug into :mod:`cflibs.benchmark`'s :class:`BenchmarkSpectrum` /
:class:`BenchmarkDataset` data model.

Adapters
--------
- :mod:`cflibs.benchmark.datasets.nist_steel` -- NIST SRM 1261a-1265a low-alloy
  steel reference materials. Compositions hard-coded from publicly available
  NIST Certificates of Analysis; spectra are user-supplied (see module docstring
  for the expected file layout under ``data/nist_steel/``).
- :mod:`cflibs.benchmark.datasets.usgs` -- USGS BHVO-2 / AGV-2 / BCR-2 / G-2
  silicate-rock geochemical reference materials spanning the basalt-andesite-
  granite range. Major-oxide compositions hard-coded from the USGS Reference
  Material Information Sheets / Jochum et al. (2016); spectra are
  user-supplied (see module docstring for the expected file layout under
  ``data/usgs_geostandards/``).

Extended-adapter helpers (bead A2)
----------------------------------
The following modules back the :data:`cflibs.benchmark.adapters_extended.MANIFEST`
generators. They are intentionally *not* imported here: the adapters lazy-import
them at iteration time so importing this package stays free of heavy I/O.

- :mod:`cflibs.benchmark.datasets.csa_planetary` -- CSA open planetary-analogue
  LIBS set (oxide certificates -> element wt%).
- :mod:`cflibs.benchmark.datasets.chemcam_calib` -- MSL ChemCam preflight
  cleanroom calibration spectra (PDS CALIB directory).
- :mod:`cflibs.benchmark.datasets.emslibs2019` -- EMSLIBS 2019 contest train
  split (class-level presence truth).
- :mod:`cflibs.benchmark.datasets.silva2022` -- Silva et al. 2022 tropical
  soils (presence-only exchangeable-fertility panel).
- :mod:`cflibs.benchmark.datasets.gibbons2024` -- Gibbons et al. nitrate-doped
  Mars Global Simulant series (quantitative N).
- :mod:`cflibs.benchmark.datasets.supercam_labcal` -- SuperCam laboratory LIBS
  calibration database (Anderson et al. 2022; oxide certificates in the same
  table -> element wt%), streamed row-by-row from the 721 MB PDS4 CSV.
- :mod:`cflibs.benchmark.datasets.supercam_scct` -- SuperCam onboard
  calibration-target spectra from the Mars surface (CL1 FITS products; truth
  joined to the lab table's SCCT chip rows). Requires astropy.
"""

from cflibs.benchmark.datasets.nist_steel import (
    CERTIFIED_COMPOSITIONS as NIST_STEEL_CERTIFIED_COMPOSITIONS,
    NISTSteelComposition,
    NISTSteelDataset,
)
from cflibs.benchmark.datasets.usgs import (
    CERTIFIED_COMPOSITIONS as USGS_CERTIFIED_COMPOSITIONS,
    USGSDataset,
    USGSStandardComposition,
    oxide_to_element_wt,
)

# Backwards-compatible alias: nist_steel was the first adapter and exported
# CERTIFIED_COMPOSITIONS unprefixed. Keep that name pointing at NIST so
# existing callers don't break.
CERTIFIED_COMPOSITIONS = NIST_STEEL_CERTIFIED_COMPOSITIONS

__all__ = [
    "CERTIFIED_COMPOSITIONS",
    "NIST_STEEL_CERTIFIED_COMPOSITIONS",
    "NISTSteelComposition",
    "NISTSteelDataset",
    "USGS_CERTIFIED_COMPOSITIONS",
    "USGSDataset",
    "USGSStandardComposition",
    "oxide_to_element_wt",
]
