"""
Curated PDS evaluation corpus for ChemCam and SuperCam LIBS data.

Defines concrete subsets of publicly available PDS data products for
reproducible validation of the CF-LIBS inversion pipeline against
real Martian spectra and onboard calibration targets.

Corpus selection criteria:
- Calibration targets with independently characterized compositions
- Multiple shots per target for noise averaging
- Coverage across the three ChemCam/SuperCam spectrometer ranges
- Small enough for CI/local validation but representative enough
  to expose parser and algorithm issues

References
----------
- Wiens et al. (2012), "The ChemCam Instrument Suite on the Mars Science
  Laboratory (MSL) Rover", Space Science Reviews 170, 167-227
- Maurice et al. (2021), "The SuperCam Instrument Suite on the Mars 2020
  Rover", Space Science Reviews 217, 47
- ChemCam PDS Archive: https://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/
- SuperCam PDS Archive: https://pds-geosciences.wustl.edu/m2020/m2020-m-supercam-libs-5-rdr-v1/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Instrument(Enum):
    """Mars LIBS instrument identifier."""

    CHEMCAM = "chemcam"
    SUPERCAM = "supercam"


class TargetClass(Enum):
    """Classification of observation target type."""

    CALIBRATION = "calibration"  # Onboard calibration target (known composition)
    ROCK = "rock"  # Natural rock target (expected elements from context)
    SOIL = "soil"  # Soil/regolith target


@dataclass(frozen=True)
class CorpusEntry:
    """A single curated PDS observation for evaluation.

    Attributes
    ----------
    entry_id : str
        Unique identifier for this corpus entry.
    instrument : Instrument
        Which instrument produced this observation.
    target_name : str
        Name of the observation target (e.g., "CCCT5", "Novarupta").
    target_class : TargetClass
        Classification of target type.
    sol : int
        Mars sol number of the observation.
    product_id : str
        PDS product identifier for the calibrated spectrum file.
    expected_elements : Dict[str, Optional[float]]
        Elements expected in this target. Value is known weight fraction
        (0-1) if independently measured, None if only qualitatively expected.
    wavelength_ranges_nm : List[tuple[float, float]]
        Wavelength ranges covered by the spectrometers.
    n_shots : int
        Number of laser shots in this observation.
    notes : str
        Human-readable context about the observation.
    pds_base_url : str
        Base URL for the PDS data archive.
    relative_path : str
        Path within the PDS archive to the data product.
    """

    entry_id: str
    instrument: Instrument
    target_name: str
    target_class: TargetClass
    sol: int
    product_id: str
    expected_elements: Dict[str, Optional[float]]
    wavelength_ranges_nm: List[tuple[float, float]] = field(default_factory=list)
    n_shots: int = 30
    notes: str = ""
    pds_base_url: str = ""
    relative_path: str = ""


# ============================================================================
# ChemCam Calibration Targets (CCCT)
# ============================================================================
# The ChemCam Calibration Target (CCCT) assembly contains 10 targets of
# known composition mounted on the rover deck. These are the gold standard
# for validating LIBS analysis pipelines.
#
# Target compositions from Fabre et al. (2011) and Wiens et al. (2013).
# All values in weight fraction. Only major elements listed; trace
# elements omitted for simplicity.

_CHEMCAM_BASE = (
    "https://pds-geosciences.wustl.edu/msl/" "msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx/"
)

# ChemCam wavelength ranges (3 spectrometers):
_CHEMCAM_WL_RANGES = [
    (240.0, 342.0),  # UV spectrometer
    (382.0, 469.0),  # VIO spectrometer
    (474.0, 906.0),  # VNIR spectrometer
]

# CCCT compositions (weight fractions of major oxides → elements)
# Reference: Fabre et al. (2011), "In Situ Calibration Using Univariate
# Analyses Based on the Onboard ChemCam Targets"
_CCCT_COMPOSITIONS: Dict[str, Dict[str, Optional[float]]] = {
    # CCCT-1: Macusanite (obsidian glass, Si-rich)
    "CCCT1": {
        "Si": 0.347,
        "Al": 0.080,
        "Na": 0.030,
        "K": 0.039,
        "Fe": 0.007,
        "Ca": 0.003,
        "Mg": 0.001,
        "O": None,
    },
    # CCCT-2: Norite (pyroxene + plagioclase)
    "CCCT2": {
        "Si": 0.243,
        "Al": 0.083,
        "Ca": 0.069,
        "Mg": 0.053,
        "Fe": 0.053,
        "Na": 0.016,
        "K": 0.003,
        "Ti": 0.003,
        "O": None,
    },
    # CCCT-3: Picrite basalt (Mg-rich)
    "CCCT3": {
        "Si": 0.207,
        "Al": 0.049,
        "Mg": 0.110,
        "Fe": 0.086,
        "Ca": 0.053,
        "Na": 0.009,
        "K": 0.002,
        "Ti": 0.005,
        "O": None,
    },
    # CCCT-4: Shergottite (basalt, Mars meteorite analog)
    "CCCT4": {
        "Si": 0.232,
        "Al": 0.050,
        "Fe": 0.127,
        "Mg": 0.049,
        "Ca": 0.068,
        "Na": 0.010,
        "K": 0.001,
        "Ti": 0.004,
        "O": None,
    },
    # CCCT-5: Graphite-doped glass ceramic (with trace Li, Sr, Mn)
    "CCCT5": {
        "Si": 0.270,
        "Al": 0.107,
        "Ca": 0.034,
        "Na": 0.035,
        "K": 0.023,
        "Fe": 0.017,
        "Mg": 0.012,
        "Ti": 0.005,
        "O": None,
    },
    # CCCT-9: Ti alloy (Ti-6Al-4V) — Fabre/Wiens (2011/2013) CCCT designation.
    # NOTE (verified 2026-06: full MSL-CHEMCAM-LIBS RDR archive scan): ChemCam does
    # NOT publish this Ti plate's LIBS spectra (fired for wavelength-cal only,
    # consumed internally to derive wave-cal coefficients). The PDS archive's
    # "Cal Target 9" is a SILICATE (the archive's "Cal Target N" numbering differs
    # from this Fabre/Wiens CCCT numbering). The composition below is correct for
    # the Ti6Al4V plate, but the only OPEN real Ti-6Al-4V LIBS spectra are SuperCam
    # SCCT_TITANIUM (data/supercam_calib/raw/scct/) — see benchmark_chemcam_ccct9.py.
    "CCCT9": {
        "Ti": 0.895,
        "Al": 0.061,
        "V": 0.040,
        "Fe": 0.004,
    },
}


def _chemcam_entry(
    target: str,
    sol: int,
    product_id: str,
    relative_path: str,
    n_shots: int = 30,
    notes: str = "",
) -> CorpusEntry:
    """Helper to create a ChemCam corpus entry."""
    return CorpusEntry(
        entry_id=f"chemcam_{target.lower()}_sol{sol}",
        instrument=Instrument.CHEMCAM,
        target_name=target,
        target_class=TargetClass.CALIBRATION,
        sol=sol,
        product_id=product_id,
        expected_elements=dict(_CCCT_COMPOSITIONS.get(target, {})),
        wavelength_ranges_nm=list(_CHEMCAM_WL_RANGES),
        n_shots=n_shots,
        notes=notes,
        pds_base_url=_CHEMCAM_BASE,
        relative_path=relative_path,
    )


# ============================================================================
# SuperCam Calibration Targets (SCCT)
# ============================================================================
# SuperCam has its own calibration targets on the Perseverance rover deck.
# SCCT compositions from Manrique et al. (2020) and Cousin et al. (2022).

_SUPERCAM_BASE = (
    "https://pds-geosciences.wustl.edu/m2020/"
    "m2020-m-supercam-libs-5-rdr-v1/m2020_supercam_libs_1xxx/"
)

_SUPERCAM_WL_RANGES = [
    (245.0, 340.0),  # UV spectrometer
    (385.0, 465.0),  # VIO spectrometer
    (536.0, 853.0),  # VNIR transmission spectrometer
]

_SCCT_COMPOSITIONS: Dict[str, Dict[str, Optional[float]]] = {
    # SCCT-5: Picrite basalt (similar to CCCT-3, Mg-rich)
    "SCCT5": {
        "Si": 0.207,
        "Al": 0.049,
        "Mg": 0.110,
        "Fe": 0.086,
        "Ca": 0.053,
        "Na": 0.009,
        "Ti": 0.005,
        "O": None,
    },
    # SCCT-7: Norite (pyroxene + plagioclase, similar to CCCT-2)
    "SCCT7": {
        "Si": 0.243,
        "Al": 0.083,
        "Ca": 0.069,
        "Mg": 0.053,
        "Fe": 0.053,
        "Na": 0.016,
        "Ti": 0.003,
        "O": None,
    },
}


def _supercam_entry(
    target: str,
    sol: int,
    product_id: str,
    relative_path: str,
    n_shots: int = 30,
    notes: str = "",
) -> CorpusEntry:
    """Helper to create a SuperCam corpus entry."""
    return CorpusEntry(
        entry_id=f"supercam_{target.lower()}_sol{sol}",
        instrument=Instrument.SUPERCAM,
        target_name=target,
        target_class=TargetClass.CALIBRATION,
        sol=sol,
        product_id=product_id,
        expected_elements=dict(_SCCT_COMPOSITIONS.get(target, {})),
        wavelength_ranges_nm=list(_SUPERCAM_WL_RANGES),
        n_shots=n_shots,
        notes=notes,
        pds_base_url=_SUPERCAM_BASE,
        relative_path=relative_path,
    )


# ============================================================================
# Curated Corpus Definition
# ============================================================================


class PDSCorpus:
    """Curated evaluation corpus of PDS LIBS observations.

    The corpus is small (< 20 observations) for repeatable CI/local
    validation but covers:
    - Multiple instruments (ChemCam and SuperCam)
    - Calibration targets with known compositions
    - Varied mineralogy (silicate glass, basalt, pyroxenite, Ti alloy)
    - All three spectrometer wavelength ranges

    Examples
    --------
    >>> corpus = PDSCorpus()
    >>> for entry in corpus.chemcam_entries():
    ...     print(f"{entry.target_name}: {list(entry.expected_elements.keys())}")
    """

    def __init__(self) -> None:
        self._entries: List[CorpusEntry] = self._build_default_corpus()

    @staticmethod
    def _build_default_corpus() -> List[CorpusEntry]:
        """Build the default curated corpus."""
        entries: List[CorpusEntry] = []

        # --- ChemCam calibration target observations ---
        # Selected from early mission sols when targets were freshest.

        # CCCT-1: Macusanite obsidian (Si-rich, low Fe)
        entries.append(
            _chemcam_entry(
                "CCCT1",
                sol=69,
                product_id="CL5_398755580RCE_F0050104CCAM01069P1",
                relative_path="data/sol00069/",
                notes="Early-mission CCCT-1, Si-dominated glass",
            )
        )

        # CCCT-3: Picrite basalt (Mg-rich, Fe-rich)
        entries.append(
            _chemcam_entry(
                "CCCT3",
                sol=69,
                product_id="CL5_398755715RCE_F0050104CCAM03069P1",
                relative_path="data/sol00069/",
                notes="Early-mission CCCT-3, Mg/Fe-rich basalt",
            )
        )

        # CCCT-4: Shergottite (Mars meteorite analog)
        entries.append(
            _chemcam_entry(
                "CCCT4",
                sol=69,
                product_id="CL5_398755850RCE_F0050104CCAM04069P1",
                relative_path="data/sol00069/",
                notes="Shergottite analog, Fe-rich basalt",
            )
        )

        # CCCT-5: Glass ceramic (intermediate composition)
        entries.append(
            _chemcam_entry(
                "CCCT5",
                sol=69,
                product_id="CL5_398756000RCE_F0050104CCAM05069P1",
                relative_path="data/sol00069/",
                notes="Glass ceramic with trace Li/Sr/Mn",
            )
        )

        # CCCT-9: Ti alloy (metallic target, not a rock)
        entries.append(
            _chemcam_entry(
                "CCCT9",
                sol=69,
                product_id="CL5_398756800RCE_F0050104CCAM09069P1",
                relative_path="data/sol00069/",
                notes="Ti-6Al-4V alloy, sharp metallic lines",
            )
        )

        # --- SuperCam calibration target observations ---
        # Selected from early commissioning phase.

        # SCCT-5: Picrite basalt (Mg-rich)
        entries.append(
            _supercam_entry(
                "SCCT5",
                sol=82,
                product_id="SC3_0082_0684979964_052RAS_N0031416SRLC08025_0000LMJ01",
                relative_path="data_calibrated/sol00082/",
                notes="Early SuperCam SCCT-5 calibration",
            )
        )

        # SCCT-7: Norite
        entries.append(
            _supercam_entry(
                "SCCT7",
                sol=82,
                product_id="SC3_0082_0684980099_052RAS_N0031416SRLC08025_0000LMJ01",
                relative_path="data_calibrated/sol00082/",
                notes="Norite calibration target",
            )
        )

        return entries

    @property
    def entries(self) -> List[CorpusEntry]:
        """All corpus entries."""
        return list(self._entries)

    def chemcam_entries(self) -> List[CorpusEntry]:
        """ChemCam-only entries."""
        return [e for e in self._entries if e.instrument == Instrument.CHEMCAM]

    def supercam_entries(self) -> List[CorpusEntry]:
        """SuperCam-only entries."""
        return [e for e in self._entries if e.instrument == Instrument.SUPERCAM]

    def calibration_entries(self) -> List[CorpusEntry]:
        """Calibration target entries only (known compositions)."""
        return [e for e in self._entries if e.target_class == TargetClass.CALIBRATION]

    def get_entry(self, entry_id: str) -> Optional[CorpusEntry]:
        """Look up an entry by ID."""
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def summary(self) -> str:
        """Human-readable corpus summary."""
        lines = [
            "PDS Evaluation Corpus",
            "=" * 60,
            f"Total entries: {len(self._entries)}",
            f"  ChemCam: {len(self.chemcam_entries())}",
            f"  SuperCam: {len(self.supercam_entries())}",
            f"  Calibration: {len(self.calibration_entries())}",
            "",
        ]
        for entry in self._entries:
            elements = ", ".join(sorted(entry.expected_elements.keys()))
            known = sum(1 for v in entry.expected_elements.values() if v is not None)
            lines.append(
                f"  {entry.entry_id}: {entry.target_name} (sol {entry.sol}) "
                f"[{known} quantified elements: {elements}]"
            )
        return "\n".join(lines)
