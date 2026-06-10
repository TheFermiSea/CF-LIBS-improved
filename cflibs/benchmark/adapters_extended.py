"""
Extended real-data benchmark adapters (bead A2: dataset unlock).

This module exposes truth-bearing LIBS datasets that CF-LIBS had never
exploited, under one shared contract so the scoreboard can multiply its
real-data coverage. Each adapter is a zero-argument generator function
yielding ``(spectrum_id, wavelength_nm, intensity, truth)`` tuples where
``truth`` is a :class:`SpectrumTruth`.

Adapter contract
----------------
- **Lazy**: no heavy I/O happens at import time; all parsing happens when the
  generator is iterated.
- **Deterministic**: file lists are sorted, caps are fixed constants, and the
  same records appear in the same order on every run.
- **Skip-with-log**: when a dataset's files are absent the generator logs a
  warning and yields nothing instead of raising.
- Element symbols use standard capitalization (``Fe``, ``Si``). Analytes
  below :data:`PRESENCE_CUTOFF_WT` (0.01 wt%) are excluded from
  ``elements_present``; the cutoff is recorded in ``truth.notes``.
- Truth provenance: every composition number comes from the dataset's own
  files; the source file and column are cited in ``truth.notes``. Nothing is
  invented; uncertified analytes are *omitted*, never guessed.

Dataset status (2026-06)
------------------------
=====================  ========  ==========================================
Dataset                Status    Truth
=====================  ========  ==========================================
CSA planetary LIBS     working   element wt% (oxide certs converted);
                                 ~99 pulse-averaged spectra, 198-970 nm
ChemCam calibration    working   element wt% (oxide certs converted);
                                 ~250 preflight spectra, 240-906 nm
EMSLIBS 2019 (Vrabel)  working   presence-only (class-level intersection of
                                 certified analytes); 100 train samples
Silva 2022 soils       working   presence-only (exchangeable P/K/Ca/Mg);
                                 102 spectra, 200-780 nm
Gibbons 2024 nitrogen  working   element wt% for N (nitrate-doped MGS-1);
                                 ~175 spectra, 186-1049 nm
SuperCam lab calib     working   element wt% (oxide certs converted);
                                 1,193 shift==0 spectra, 244-853 nm
SuperCam SCCT (Mars)   working   element wt% joined from lab chip rows;
                                 547 CL1 FITS products, sols 13-1694
=====================  ========  ==========================================

Per-dataset parsing lives in small helper modules under
:mod:`cflibs.benchmark.datasets`; this module owns the thin generator
wrappers and :data:`MANIFEST`, the registration list the scoreboard
maintainer wires at integration time. The contract types come from
:mod:`cflibs.benchmark.scoreboard_registry` (re-exported here).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

__all__ = [
    "DATA_ROOT",
    "MANIFEST",
    "PRESENCE_CUTOFF_WT",
    "SpectrumRecord",
    "SpectrumTruth",
    "enforce_strictly_increasing",
    "iter_chemcam_calibration_spectra",
    "iter_csa_planetary_spectra",
    "iter_emslibs2019_spectra",
    "iter_gibbons2024_spectra",
    "iter_silva2022_spectra",
    "iter_supercam_labcal_spectra",
    "iter_supercam_scct_spectra",
]

logger = logging.getLogger(__name__)


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
"""Repository ``data/`` directory holding (symlinked) external datasets."""


# The shared contract types and the presence cutoff live in the scoreboard
# registry (bead A1); the wavelength-grid helper in datasets._common. All are
# re-exported here so adapter callers and tests can import either path.
from cflibs.benchmark.scoreboard_registry import (  # noqa: E402
    PRESENCE_CUTOFF_WT,
    AdapterFactory,
    AdapterYield,
    SpectrumTruth,
)
from cflibs.benchmark.datasets._common import enforce_strictly_increasing  # noqa: E402

SpectrumRecord = AdapterYield
"""One adapter yield: ``(spectrum_id, wavelength_nm, intensity, truth)``."""


def _require(dataset: str, needs: str, *checks: tuple[Path, bool]) -> bool:
    """Skip-with-log presence guard: every ``(path, present)`` check must pass.

    Logs the first missing path together with the dataset's acquisition hint
    and returns False so the wrapper can yield nothing (adapter contract).
    """
    for path, present in checks:
        if not present:
            logger.warning("Skipping %s adapter: %s not found. %s", dataset, path, needs)
            return False
    return True


# ---------------------------------------------------------------------------
# Generator wrappers (the per-dataset parser modules stay lazy-imported so
# importing this module never touches their heavy dependencies).
# ---------------------------------------------------------------------------


def iter_csa_planetary_spectra(data_dir: Path | None = None) -> Iterator[SpectrumRecord]:
    """
    CSA (Canadian Space Agency) open planetary-analogue LIBS dataset.

    ~99 pulse-averaged spectra (198-970 nm) of geological reference materials
    and hand samples, with certified oxide compositions converted to element
    wt%. Requires ``data/csa_planetary_libs/extracted/`` (run
    ``7z x LIBSOpenDatacsv.7z -oextracted`` first).
    """
    from cflibs.benchmark.datasets import csa_planetary

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "csa_planetary_libs"
    extracted = root / "extracted"
    comp_csv = root / "Sample_Composition_Data_LargeSet.csv"
    if not _require(
        "csa_planetary",
        "Extract LIBSOpenDatacsv.7z into data/csa_planetary_libs/extracted/ "
        "(needs the two 'csv ...pulseaverage' folders) and keep "
        "Sample_Composition_Data_LargeSet.csv next to the archive.",
        (extracted, extracted.is_dir()),
        (comp_csv, comp_csv.is_file()),
    ):
        return
    yield from csa_planetary.iter_spectra(root)


def iter_chemcam_calibration_spectra(data_dir: Path | None = None) -> Iterator[SpectrumRecord]:
    """
    MSL ChemCam preflight calibration spectra (PDS ``MSL_CCAM_LIBS_CALIB.CSV``).

    ~250 cleanroom radiance spectra (240-906 nm, 3-4 replicates each) of 66
    geochemical standards, joined to oxide compositions from
    ``CCAM_CALIBRATION_COMPOSITIONS.CSV``. Same instrument family as the
    BHVO-2 gate.
    """
    from cflibs.benchmark.datasets import chemcam_calib

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "chemcam_calib"
    spectra_csv = root / "msl_ccam_libs_calib.csv"
    comp_csv = root / "ccam_calibration_compositions.csv"
    if not _require(
        "chemcam_calib",
        "Download MSL_CCAM_LIBS_CALIB.CSV and CCAM_CALIBRATION_COMPOSITIONS.CSV "
        "from the PDS Geosciences node (MSL ChemCam LIBS RDR CALIB directory, "
        "msl-m-chemcam-libs-4_5-rdr-v1) into data/chemcam_calib/.",
        (spectra_csv, spectra_csv.is_file()),
        (comp_csv, comp_csv.is_file()),
    ):
        return
    yield from chemcam_calib.iter_spectra(root)


def iter_emslibs2019_spectra(
    data_dir: Path | None = None, shots_per_sample: int = 3
) -> Iterator[SpectrumRecord]:
    """
    EMSLIBS 2019 contest benchmark (Vrabel et al. 2020 classification dataset).

    Train-set spectra (200-1000 nm, 40002 channels) for 100 ore samples in 12
    classes, capped at ``shots_per_sample`` of the 500 shots per sample.
    Truth is presence-only at class level (see helper module docstring).
    Requires ``train.h5`` (~7.5 GB) and ``support_tables.xlsx``.
    """
    from cflibs.benchmark.datasets import emslibs2019

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "vrabel2020_soil_benchmark"
    train_h5 = root / "train.h5"
    support = root / "support_tables.xlsx"
    if not _require(
        "emslibs2019",
        "Download train.h5 and support_tables.xlsx from the EMSLIBS 2019 "
        "contest archive (Kepes/Vrabel et al. 2020, "
        "https://doi.org/10.6084/m9.figshare.c.4768790) into "
        "data/vrabel2020_soil_benchmark/.",
        (train_h5, train_h5.is_file()),
        (support, support.is_file()),
    ):
        return
    try:
        import h5py  # noqa: F401
    except ImportError:
        logger.warning("Skipping emslibs2019 adapter: h5py is not installed.")
        return
    yield from emslibs2019.iter_spectra(root, shots_per_sample=shots_per_sample)


def iter_silva2022_spectra(data_dir: Path | None = None) -> Iterator[SpectrumRecord]:
    """
    Silva et al. 2022 tropical-soil LIBS dataset (102 Brazilian soil samples).

    One spectrum per sample (200-780 nm), presence-only truth for the
    exchangeable-fertility panel P/K/Ca/Mg from ``soil_fertility_data.txt``.
    """
    from cflibs.benchmark.datasets import silva2022

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "silva2022_tropical_soils"
    libs_txt = root / "LIBS_data.txt"
    fert_txt = root / "soil_fertility_data.txt"
    if not _require(
        "silva2022",
        "Place LIBS_data.txt and soil_fertility_data.txt (Silva et al. 2022 "
        "tropical-soils deposit) into data/silva2022_tropical_soils/.",
        (libs_txt, libs_txt.is_file()),
        (fert_txt, fert_txt.is_file()),
    ):
        return
    yield from silva2022.iter_spectra(root)


def iter_gibbons2024_spectra(data_dir: Path | None = None) -> Iterator[SpectrumRecord]:
    """
    Gibbons et al. nitrate-in-Mars-simulant LIBS series (N quantitation).

    ~175 spectra (186-1049 nm, 100-shot accumulations in He) of nitrate salts
    doped into Mars Global Simulant plus pure-salt end-members. N wt% derived
    stoichiometrically from the certified NO3- ion wt% column.
    """
    from cflibs.benchmark.datasets import gibbons2024

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "gibbons2024_nitrogen_libs"
    xlsx = root / "SI_Raw_Spectral_Data.xlsx"
    if not _require(
        "gibbons2024",
        "Place SI_Raw_Spectral_Data.xlsx (Gibbons et al. supporting "
        "information) into data/gibbons2024_nitrogen_libs/.",
        (xlsx, xlsx.is_file()),
    ):
        return
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        logger.warning("Skipping gibbons2024 adapter: openpyxl is not installed.")
        return
    yield from gibbons2024.iter_spectra(root)


def iter_supercam_labcal_spectra(data_dir: Path | None = None) -> Iterator[SpectrumRecord]:
    """
    SuperCam laboratory LIBS calibration database (Anderson et al. 2022).

    1,193 shift==0 spectra (244-853 nm, 7,933 channels) of 334 geological
    standards, streamed row-by-row from the 721 MB PDS4 table with certified
    compositions in the same rows (oxide wt% -> element wt%). Honest splits
    must group by Target_Name (see the helper module docstring).
    """
    from cflibs.benchmark.datasets import supercam_labcal

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "supercam_calib"
    labcal_csv = root / supercam_labcal.LABCAL_CSV_RELPATH
    if not _require(
        "supercam_labcal",
        "Download the SuperCam lab calibration table (PDS Geosciences node, "
        "urn:nasa:pds:mars2020_supercam:calibration_supercam, "
        "libs_spectral_library_reference.csv) into "
        "data/supercam_calib/raw/labcal/.",
        (labcal_csv, labcal_csv.is_file()),
    ):
        return
    yield from supercam_labcal.iter_spectra(root)


def iter_supercam_scct_spectra(data_dir: Path | None = None) -> Iterator[SpectrumRecord]:
    """
    SuperCam Calibration Target LIBS spectra from the Mars surface (CL1).

    547 calibrated FITS products across 23 onboard cal targets, sols 13-1694;
    truth joined to the lab calibration table's SCCT chip rows (same material
    batch). Real-Mars spectra: HOLDOUT tier (adoption-gate material).
    Requires astropy and both raw/scct/cl1/ and the lab CSV.
    """
    from cflibs.benchmark.datasets import supercam_labcal, supercam_scct

    root = Path(data_dir) if data_dir is not None else DATA_ROOT / "supercam_calib"
    cl1_dir = root / supercam_scct.CL1_RELPATH
    labcal_csv = root / supercam_labcal.LABCAL_CSV_RELPATH
    if not _require(
        "supercam_scct",
        "Download the SCCT CL1 FITS products (PDS urn:nasa:pds:mars2020_supercam "
        "data_calibrated_spectra, files matching *scct*cl1*) into "
        "data/supercam_calib/raw/scct/cl1/sol_XXXXX/ and the lab table "
        "libs_spectral_library_reference.csv (truth source) into "
        "data/supercam_calib/raw/labcal/.",
        (cl1_dir, cl1_dir.is_dir()),
        (labcal_csv, labcal_csv.is_file()),
    ):
        return
    try:
        import astropy  # noqa: F401
    except ImportError:
        logger.warning(
            "Skipping supercam_scct adapter: astropy is not installed "
            "(pip install 'cflibs[fits]')."
        )
        return
    yield from supercam_scct.iter_spectra(root)


# ---------------------------------------------------------------------------
# Registration manifest -- the scoreboard maintainer wires these entries into
# the registry at integration time. Order = value order from bead A2. Tiers
# mirror the campaign split design (design 2.1; splits.py derives its name
# sets from these registrations): emslibs2019 is HOLDOUT (adoption gate),
# gibbons2024 is VAULT (end-of-program only — never run by the scoreboard).
# ---------------------------------------------------------------------------

MANIFEST: list[tuple[str, AdapterFactory, tuple[str, ...], str, str]] = [
    (
        "csa_planetary",
        iter_csa_planetary_spectra,
        ("real", "geological", "element_wt", "broadband"),
        "optimization",
        "CSA open planetary-analogue LIBS: ~99 pulse-averaged spectra "
        "(198-970 nm) of certified geological standards + hand samples; "
        "element-wt truth from oxide certificates; requires one-time 7z "
        "extraction of LIBSOpenDatacsv.7z.",
    ),
    (
        "chemcam_calib",
        iter_chemcam_calibration_spectra,
        ("real", "geological", "element_wt", "chemcam"),
        "optimization",
        "MSL ChemCam preflight cleanroom calibration: ~250 radiance spectra "
        "(240-906 nm) of 66 standards with oxide certificates; same "
        "instrument family as the BHVO-2 gate.",
    ),
    (
        "emslibs2019",
        iter_emslibs2019_spectra,
        ("real", "ore", "presence_only", "classification"),
        "holdout",
        "EMSLIBS 2019 contest (Vrabel et al. 2020): 100 train ore samples x "
        "3 shots (200-1000 nm); presence-only truth at class level because "
        "the train-sample <-> certificate mapping is not in the files.",
    ),
    (
        "silva2022",
        iter_silva2022_spectra,
        ("real", "soil", "presence_only"),
        "optimization",
        "Silva et al. 2022 tropical soils: 102 spectra (200-780 nm); "
        "presence-only truth for exchangeable P/K/Ca/Mg (units are mg/dm3 / "
        "mmolc/dm3, not convertible to wt% without bulk density).",
    ),
    (
        "gibbons2024",
        iter_gibbons2024_spectra,
        ("real", "planetary", "element_wt", "nitrogen"),
        "vault",
        "Gibbons et al. nitrate-doped Mars Global Simulant: ~175 spectra "
        "(186-1049 nm, He atmosphere); quantitative N wt% from certified "
        "NO3- ion wt%; matrix elements uncertified (partial panel).",
    ),
    (
        "supercam_labcal",
        iter_supercam_labcal_spectra,
        ("real", "geological", "element_wt", "supercam", "mars_analog"),
        "optimization",
        "SuperCam lab LIBS calibration database (Anderson et al. 2022, PDS "
        "calibration_supercam): 1,193 shift==0 spectra (244-853 nm) of 334 "
        "certified standards in a ~7 mbar CO2 chamber; element-wt truth from "
        "the same table's oxide/volatile/trace columns; 54 Remove_from_all "
        "rows skipped; multiple observation points share one Target_Name "
        "truth row — honest splits must group by Target_Name; relative "
        "intensity units.",
    ),
    (
        "supercam_scct",
        iter_supercam_scct_spectra,
        ("real", "geological", "element_wt", "supercam", "mars"),
        "holdout",
        "SuperCam onboard calibration targets shot on Mars (PDS CL1 products): "
        "547 spectra of 23 targets, sols 13-1694, true Mars atmosphere/dust "
        "at ~3 m standoff; element-wt truth joined from the lab table's SCCT "
        "chip rows (same material batch, chip number ignored); TITANIUM is "
        "presence-only (Ti6Al4V plate). Real-Mars adoption-gate material.",
    ),
]


def register_extended_adapters(*, replace: bool = True) -> None:
    """Register every MANIFEST dataset into the scoreboard registry.

    Idempotent like :func:`cflibs.benchmark.adapters_core.register_core_adapters`.
    """
    from cflibs.benchmark.scoreboard_registry import register_dataset

    for name, factory, tags, tier, notes in MANIFEST:
        register_dataset(name, factory, tags=tags, tier=tier, notes=notes, replace=replace)
