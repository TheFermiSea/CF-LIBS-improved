"""
USGS geochemical reference materials benchmark adapter.

This module hard-codes the certified major-oxide compositions for four widely
used USGS silicate-rock geochemical reference materials that span the
ultramafic-to-felsic composition space and are the de-facto standard for
geochemistry cross-validation:

==========  ===============================  ===========================
Standard    Material                         Composition class
==========  ===============================  ===========================
BHVO-2      Hawaiian Volcano Obs. basalt     Tholeiitic basalt
AGV-2       Guano Valley andesite (Oregon)   Calc-alkaline andesite
BCR-2       Columbia River basalt (Oregon)   Continental flood basalt
G-2         Westerly granite (Rhode Island)  Granite
==========  ===============================  ===========================

Composition source
------------------
Major-oxide mass fractions (wt %) and 95% confidence-level uncertainties are
transcribed from the USGS Reference Material Information Sheets (revised
June 2022 for the three currently distributed materials), which in turn cite
Jochum et al. (2016), Table 3:

    Jochum, K.P., Weis, U., Schwager, B., Stoll, B., Wilson, S.A., Haug, G.H.,
    Andreae, M.O., and Enzweiler, J., 2016, "Reference values following ISO
    guidelines for frequently requested rock reference materials."
    Geostandards and Geoanalytical Research, v. 40, p. 333-350.
    https://doi.org/10.1111/j.1751-908X.2015.00392.x

USGS Information Sheets:

- BHVO-2:  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/USGS-BHVO-2-IS_2022-508.pdf
- AGV-2:   https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/USGS-AGV-2-IS_2022-508_4.pdf
- BCR-2:   https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/USGS%20BCR-2%20IS_2022.pdf
- G-2:     The USGS no longer distributes G-2 (Jochum et al. 2016, Table 1),
           so values come directly from Table 3 of the same paper.

Iron is reported as Fe2O3T (total iron expressed as Fe2O3), per Jochum 2016.
We retain that convention in :data:`OXIDE_TO_ELEMENT_FACTOR` and convert with
the stoichiometric factor 2 * M_Fe / M(Fe2O3) = 0.6994.

Spectra source
--------------
The author searched Mendeley Data, Zenodo, the NASA PDS ChemCam archives, MIT
LIBS DB, ASU LIBS DB, and recent Spectrochim. Acta B / JAAS / Talanta /
Appl. Geochem. issues for openly redistributable LIBS spectra of bulk powder
samples of BHVO-2, AGV-2, BCR-2, and G-2. None were found that ship without a
login wall or a non-redistribution clause:

- The Mars ChemCam CCCT calibration target series uses ceramic pucks doped from
  USGS feedstocks, not the bulk powders themselves; spectra (PDS-hosted) are
  for the puck composition, not BHVO-2/etc.
- Most published LIBS papers using these standards (e.g., Anderson et al.,
  Cousin et al., Wiens et al. and follow-ups) do not deposit raw spectra.
- The MIT LIBS DB requires an account; ASU's LIBS database overlaps with
  ChemCam CCCT pucks rather than USGS bulk powders.

Users with their own measurements should drop a CSV per standard into
``data/usgs_geostandards/`` named ``{standard_id}.csv`` (two columns:
``wavelength`` / ``wavelength_nm`` and ``intensity``);
:meth:`USGSDataset.get_spectrum` returns ``None`` when no such file is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cflibs.benchmark.dataset import (
    BenchmarkSpectrum,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
    TruthType,
)
from cflibs.io.spectrum import load_spectrum

__all__ = [
    "CERTIFIED_COMPOSITIONS",
    "DEFAULT_DATA_DIR",
    "OXIDE_TO_ELEMENT_FACTOR",
    "USGSStandardComposition",
    "USGSDataset",
    "oxide_to_element_wt",
]


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "usgs_geostandards"
"""Default directory inspected by :meth:`USGSDataset.get_spectrum`."""


# ---------------------------------------------------------------------------
# Stoichiometric oxide -> cation mass-fraction conversion factors.
# ---------------------------------------------------------------------------
#
# For an oxide ``X_a O_b`` with formula mass M_oxide, the cation mass fraction
# is ``a * M_X / M_oxide``. Atomic weights are CIAAW 2021 standard values.
# The constants below are computed in closed form (see ``OXIDE_DERIVATION``
# below) and stored to 5-6 significant figures.
#
# Iron special case: Jochum (2016) and the USGS sheets report Fe2O3T (total
# iron expressed as Fe2O3); converting Fe2O3T -> Fe with the stoichiometric
# Fe2O3 factor gives total Fe mass fraction, regardless of redox state.

OXIDE_DERIVATION: dict[str, str] = {
    # oxide  -> "(a * M_X) / M_oxide  with masses (g/mol)"
    "SiO2": "1 * 28.085 / (28.085 + 2 * 15.999)        = 0.46744",
    "TiO2": "1 * 47.867 / (47.867 + 2 * 15.999)        = 0.59940",
    "Al2O3": "2 * 26.982 / (2 * 26.982 + 3 * 15.999)   = 0.52925",
    "Fe2O3": "2 * 55.845 / (2 * 55.845 + 3 * 15.999)   = 0.69944",
    "FeO": "1 * 55.845 / (55.845 + 15.999)             = 0.77731",
    "MnO": "1 * 54.938 / (54.938 + 15.999)             = 0.77445",
    "MgO": "1 * 24.305 / (24.305 + 15.999)             = 0.60304",
    "CaO": "1 * 40.078 / (40.078 + 15.999)             = 0.71470",
    "Na2O": "2 * 22.990 / (2 * 22.990 + 15.999)        = 0.74186",
    "K2O": "2 * 39.098 / (2 * 39.098 + 15.999)         = 0.83014",
    "P2O5": "2 * 30.974 / (2 * 30.974 + 5 * 15.999)    = 0.43643",
}

OXIDE_TO_ELEMENT_FACTOR: dict[str, tuple[str, float]] = {
    "SiO2": ("Si", 0.46744),
    "TiO2": ("Ti", 0.59940),
    "Al2O3": ("Al", 0.52925),
    "Fe2O3": ("Fe", 0.69944),
    "FeO": ("Fe", 0.77731),
    "Fe2O3T": ("Fe", 0.69944),  # total iron expressed as Fe2O3
    "MnO": ("Mn", 0.77445),
    "MgO": ("Mg", 0.60304),
    "CaO": ("Ca", 0.71470),
    "Na2O": ("Na", 0.74186),
    "K2O": ("K", 0.83014),
    "P2O5": ("P", 0.43643),
}
"""Map oxide formula -> (element symbol, cation mass fraction)."""


def oxide_to_element_wt(oxide_wt_percent: dict[str, float]) -> dict[str, float]:
    """
    Convert a major-oxide mass-percent dict to elemental cation mass percents.

    Oxygen bound in each oxide is implicit; the returned dict contains only
    the cation contributions. Multiple oxides containing the same cation
    (e.g. ``"FeO"`` and ``"Fe2O3"``) are summed.

    Parameters
    ----------
    oxide_wt_percent : dict[str, float]
        Oxide formula -> mass percent (g / 100 g sample). Unknown oxides are
        ignored with no error so callers can pass through LOI / H2O / CO2.

    Returns
    -------
    dict[str, float]
        Element symbol -> mass percent (g / 100 g sample). Sum is < the input
        sum because oxygen is excluded.

    Examples
    --------
    >>> # 100% pure SiO2 -> 46.744 wt% Si.
    >>> oxide_to_element_wt({"SiO2": 100.0})  # doctest: +ELLIPSIS
    {'Si': 46.74...}
    """
    out: dict[str, float] = {}
    for oxide, wt in oxide_wt_percent.items():
        try:
            element, factor = OXIDE_TO_ELEMENT_FACTOR[oxide]
        except KeyError:
            continue
        out[element] = out.get(element, 0.0) + float(wt) * factor
    return out


# ---------------------------------------------------------------------------
# Composition dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class USGSStandardComposition:
    """
    Certified major-oxide composition of one USGS geochemical reference rock.

    Attributes
    ----------
    standard_id : str
        Catalogue identifier, e.g. ``"BHVO-2"``.
    material : str
        Human-readable description (e.g. ``"Hawaiian Volcano Observatory basalt"``).
    rock_type : str
        Petrologic classification (``"basalt"``, ``"andesite"``, ``"granite"``).
    oxide_wt_percent : dict[str, float]
        Oxide formula (e.g. ``"SiO2"``, ``"Fe2O3T"``) -> mass percent
        (g / 100 g sample). Iron is reported as ``"Fe2O3T"`` (total iron
        expressed as Fe2O3) per Jochum et al. (2016).
    oxide_uncertainty_wt_percent : dict[str, float]
        Oxide formula -> 95% confidence-level uncertainty in mass percent,
        per Table 3 of Jochum et al. (2016).
    locality : str
        Sample collection locality.
    information_sheet_url : str
        URL of the USGS Reference Material Information Sheet (kept for
        traceability). Empty for G-2 (no current USGS sheet; values come
        from Jochum et al. 2016 directly).
    notes : str
        Free-form provenance / interpretation notes.
    """

    standard_id: str
    material: str
    rock_type: str
    oxide_wt_percent: dict[str, float]
    oxide_uncertainty_wt_percent: dict[str, float] = field(default_factory=dict)
    locality: str = ""
    information_sheet_url: str = ""
    notes: str = ""

    @property
    def total_oxide_wt_percent(self) -> float:
        """Sum of oxide mass percents (excludes LOI / H2O / CO2)."""
        return float(sum(self.oxide_wt_percent.values()))

    @property
    def element_wt_percent(self) -> dict[str, float]:
        """Cation mass percents (oxygen-free) derived from ``oxide_wt_percent``."""
        return oxide_to_element_wt(self.oxide_wt_percent)

    @property
    def element_uncertainty_wt_percent(self) -> dict[str, float]:
        """
        Cation mass-percent uncertainties propagated from the oxide values.

        The conversion is a constant scalar (the stoichiometric cation mass
        fraction), so the relative uncertainty is preserved:
        ``sigma_X = factor * sigma_oxide``. Uncertainties for distinct oxides
        sharing a cation are added in quadrature (independent measurements).
        """
        out: dict[str, float] = {}
        # Track variance contributions, then sqrt at the end.
        variance: dict[str, float] = {}
        for oxide, sigma in self.oxide_uncertainty_wt_percent.items():
            try:
                element, factor = OXIDE_TO_ELEMENT_FACTOR[oxide]
            except KeyError:
                continue
            variance[element] = variance.get(element, 0.0) + (factor * float(sigma)) ** 2
        for element, var in variance.items():
            out[element] = float(np.sqrt(var))
        return out

    def as_mass_fractions(self) -> dict[str, float]:
        """
        Return cation composition as mass *fractions* (wt% / 100).

        Sum is < 1.0 because oxygen, LOI, H2O, CO2 are excluded; callers that
        need a closed composition should treat the missing balance as oxygen.
        """
        return {el: v / 100.0 for el, v in self.element_wt_percent.items()}


# ---------------------------------------------------------------------------
# Certified compositions transcribed from USGS Information Sheets / Jochum
# (2016) Table 3. All values are 95% CL uncertainties.
# ---------------------------------------------------------------------------

_BHVO_2 = USGSStandardComposition(
    standard_id="BHVO-2",
    material="Hawaiian Volcano Observatory basalt",
    rock_type="basalt",
    oxide_wt_percent={
        "SiO2": 49.60,
        "TiO2": 2.731,
        "Al2O3": 13.44,
        "Fe2O3T": 12.39,
        "MnO": 0.1690,
        "MgO": 7.257,
        "CaO": 11.40,
        "Na2O": 2.219,
        "K2O": 0.5130,
        "P2O5": 0.2685,
    },
    oxide_uncertainty_wt_percent={
        "SiO2": 0.14,
        "TiO2": 0.018,
        "Al2O3": 0.06,
        "Fe2O3T": 0.09,
        "MnO": 0.0019,
        "MgO": 0.042,
        "CaO": 0.06,
        "Na2O": 0.048,
        "K2O": 0.0037,
        "P2O5": 0.0050,
    },
    locality="Halemaumau Crater, Kilauea Caldera, Hawaii (1919 pahoehoe flow)",
    information_sheet_url=(
        "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/"
        "s3fs-public/media/files/USGS-BHVO-2-IS_2022-508.pdf"
    ),
    notes=(
        "Tholeiitic basalt. Values from USGS Information Sheet (Rev. June 2022), "
        "Table 1, citing Jochum et al. (2016) Table 3. Fe2O3T is total iron "
        "expressed as Fe2O3."
    ),
)


_AGV_2 = USGSStandardComposition(
    standard_id="AGV-2",
    material="Guano Valley andesite",
    rock_type="andesite",
    oxide_wt_percent={
        "SiO2": 59.14,
        "TiO2": 1.051,
        "Al2O3": 17.03,
        "Fe2O3T": 6.78,
        "MnO": 0.1004,
        "MgO": 1.80,
        "CaO": 5.15,
        "Na2O": 4.204,
        "K2O": 2.898,
        "P2O5": 0.483,
    },
    oxide_uncertainty_wt_percent={
        "SiO2": 0.58,
        "TiO2": 0.023,
        "Al2O3": 0.12,
        "Fe2O3T": 0.17,
        "MnO": 0.0026,
        "MgO": 0.15,
        "CaO": 0.10,
        "Na2O": 0.080,
        "K2O": 0.033,
        "P2O5": 0.043,
    },
    locality="Guano Valley, Lake County, Oregon",
    information_sheet_url=(
        "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/"
        "s3fs-public/media/files/USGS-AGV-2-IS_2022-508_4.pdf"
    ),
    notes=(
        "Calc-alkaline andesite, second-generation companion to AGV-1. Values "
        "from USGS Information Sheet (Rev. June 2022), Table 1, citing Jochum "
        "et al. (2016) Table 3. Fe2O3T is total iron expressed as Fe2O3."
    ),
)


_BCR_2 = USGSStandardComposition(
    standard_id="BCR-2",
    material="Columbia River basalt",
    rock_type="basalt",
    oxide_wt_percent={
        "SiO2": 54.00,
        "TiO2": 2.265,
        "Al2O3": 13.48,
        "Fe2O3T": 13.77,
        "MnO": 0.1966,
        "MgO": 3.599,
        "CaO": 7.114,
        "Na2O": 3.120,
        "K2O": 1.774,
        "P2O5": 0.3593,
    },
    oxide_uncertainty_wt_percent={
        "SiO2": 0.20,
        "TiO2": 0.024,
        "Al2O3": 0.12,
        "Fe2O3T": 0.19,
        "MnO": 0.0030,
        "MgO": 0.044,
        "CaO": 0.075,
        "Na2O": 0.042,
        "K2O": 0.019,
        "P2O5": 0.0095,
    },
    locality="Bridal Veil Flow Quarry, ~26 mi east of Portland, Oregon (1996)",
    information_sheet_url=(
        "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/"
        "s3fs-public/media/files/USGS%20BCR-2%20IS_2022.pdf"
    ),
    notes=(
        "Continental flood basalt; second-generation companion to BCR-1. Values "
        "from USGS Information Sheet (Rev. June 2022), Table 1, citing Jochum "
        "et al. (2016) Table 3. Fe2O3T is total iron expressed as Fe2O3."
    ),
)


_G_2 = USGSStandardComposition(
    standard_id="G-2",
    material="Westerly granite",
    rock_type="granite",
    oxide_wt_percent={
        "SiO2": 68.74,
        "TiO2": 0.4799,
        "Al2O3": 15.31,
        "Fe2O3T": 2.644,
        "MnO": 0.0306,
        "MgO": 0.754,
        "CaO": 1.910,
        "Na2O": 4.045,
        "K2O": 4.500,
        "P2O5": 0.129,
    },
    oxide_uncertainty_wt_percent={
        "SiO2": 0.47,
        "TiO2": 0.0089,
        "Al2O3": 0.17,
        "Fe2O3T": 0.048,
        "MnO": 0.0008,
        "MgO": 0.029,
        "CaO": 0.037,
        "Na2O": 0.059,
        "K2O": 0.061,
        "P2O5": 0.022,
    },
    locality="Sullivan quarry, Bradford, Rhode Island",
    information_sheet_url="",
    notes=(
        "Granite. The USGS no longer distributes G-2 (Jochum et al. 2016), so "
        "no current USGS Information Sheet exists; values are from Jochum et "
        "al. (2016) Table 3 directly. Fe2O3T is total iron expressed as Fe2O3."
    ),
)


CERTIFIED_COMPOSITIONS: dict[str, USGSStandardComposition] = {
    comp.standard_id: comp for comp in (_BHVO_2, _AGV_2, _BCR_2, _G_2)
}
"""All four USGS rock-standard compositions, keyed by standard id."""


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


_MATRIX_BY_ROCK_TYPE = {
    "basalt": MatrixType.GEOLOGICAL,
    "andesite": MatrixType.GEOLOGICAL,
    "granite": MatrixType.GEOLOGICAL,
}


class USGSDataset:
    """
    Benchmark adapter for USGS BHVO-2, AGV-2, BCR-2, and G-2 reference rocks.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory inspected by :meth:`get_spectrum` for ``{standard_id}.csv``
        files (or any extension supported by
        :func:`cflibs.io.spectrum.load_spectrum`). Defaults to
        ``<repo>/data/usgs_geostandards/``.

    Notes
    -----
    Mirrors the call surface of :class:`NISTSteelDataset` and the existing
    Aalto adapter (:func:`cflibs.benchmark.unified.load_aalto_id_dataset`):

    - :meth:`available_samples` enumerates known standard ids.
    - :meth:`get_composition` returns the certified composition record (always
      available -- the data is hard-coded from USGS / Jochum 2016).
    - :meth:`get_spectrum` returns a :class:`BenchmarkSpectrum` if a CSV is
      cached in ``data_dir``, otherwise ``None``. USGS does not publish LIBS
      spectra alongside its Information Sheets; users supply their own.

    Examples
    --------
    >>> ds = USGSDataset()
    >>> sorted(ds.available_samples())
    ['AGV-2', 'BCR-2', 'BHVO-2', 'G-2']
    >>> bhvo2 = ds.get_composition("BHVO-2")
    >>> round(bhvo2.element_wt_percent["Si"], 2)
    23.18
    """

    DATASET_NAME = "usgs_geostandards"
    INSTRUMENT_ID = "user_supplied"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    # ------------------------------------------------------------------ API

    def available_samples(self) -> list[str]:
        """Return all standard ids covered by the certified-composition table."""
        return sorted(CERTIFIED_COMPOSITIONS.keys())

    def get_composition(self, standard_id: str) -> USGSStandardComposition:
        """
        Return the certified composition for ``standard_id``.

        Raises
        ------
        KeyError
            If ``standard_id`` is not one of the four supported standards.
        """
        try:
            return CERTIFIED_COMPOSITIONS[standard_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown USGS standard id {standard_id!r}; "
                f"available: {self.available_samples()}"
            ) from exc

    def get_spectrum(self, standard_id: str) -> Optional[BenchmarkSpectrum]:
        """
        Return the cached LIBS spectrum for ``standard_id`` if one is on disk.

        Looks for ``data_dir/{standard_id}.csv`` (and a few common alternates)
        and wraps the result in a :class:`BenchmarkSpectrum` whose
        ``true_composition`` is the cation mass-fraction vector derived from
        the certified oxide composition. Returns ``None`` -- not raising --
        when no spectrum file is present, so callers can iterate over
        :meth:`available_samples` without try/except.

        Raises
        ------
        KeyError
            If ``standard_id`` is not one of the four supported standards.
        """
        if standard_id not in CERTIFIED_COMPOSITIONS:
            raise KeyError(
                f"Unknown USGS standard id {standard_id!r}; "
                f"available: {self.available_samples()}"
            )

        spectrum_path = self._locate_spectrum_file(standard_id)
        if spectrum_path is None:
            return None

        wavelength, intensity = load_spectrum(str(spectrum_path))
        composition = self.get_composition(standard_id)
        wavelength = np.asarray(wavelength, dtype=float)
        intensity = np.asarray(intensity, dtype=float)

        conditions = InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=0.0,
            spectral_range_nm=(float(wavelength.min()), float(wavelength.max())),
            spectral_resolution_nm=0.05,
            spectrometer_type="user_supplied",
            detector_type="unknown",
            atmosphere="air",
            notes=f"User-supplied spectrum from {spectrum_path}",
        )
        metadata = SampleMetadata(
            sample_id=f"USGS_{standard_id}",
            sample_type=SampleType.CRM,
            matrix_type=_MATRIX_BY_ROCK_TYPE.get(composition.rock_type, MatrixType.GEOLOGICAL),
            crm_name=f"USGS {standard_id}",
            crm_source="USGS",
            preparation="powder",
            surface_condition="pressed_pellet",
            doi=composition.information_sheet_url or "",
            provenance=(
                f"Composition: USGS Information Sheet / Jochum et al. (2016); "
                f"spectrum file: {spectrum_path.name}"
            ),
        )
        composition_uncertainty_fraction = {
            el: float(u) / 100.0 for el, u in composition.element_uncertainty_wt_percent.items()
        }
        return BenchmarkSpectrum(
            spectrum_id=f"usgs_{standard_id.lower()}",
            wavelength_nm=wavelength,
            intensity=intensity,
            true_composition=composition.as_mass_fractions(),
            composition_uncertainty=composition_uncertainty_fraction,
            conditions=conditions,
            metadata=metadata,
            dataset_id=self.DATASET_NAME,
            group_id=standard_id,
            specimen_id=standard_id,
            instrument_id=self.INSTRUMENT_ID,
            truth_type=TruthType.ASSAY,
            spectrum_kind="geostandard",
            annotations={
                "standard_id": standard_id,
                "material": composition.material,
                "rock_type": composition.rock_type,
                "locality": composition.locality,
                "information_sheet_url": composition.information_sheet_url,
            },
        )

    # ------------------------------------------------------------- helpers

    def _locate_spectrum_file(self, standard_id: str) -> Optional[Path]:
        """Look up a cached spectrum file for ``standard_id`` in ``data_dir``."""
        if not self.data_dir.is_dir():
            return None
        # Try a few common naming conventions; first hit wins. Hyphens are
        # often dropped or replaced with underscores in filenames.
        flat = standard_id.replace("-", "")
        under = standard_id.replace("-", "_")
        candidates = [
            self.data_dir / f"{standard_id}.csv",
            self.data_dir / f"{flat}.csv",
            self.data_dir / f"{under}.csv",
            self.data_dir / f"USGS_{standard_id}.csv",
            self.data_dir / f"usgs_{standard_id}.csv",
            self.data_dir / f"{standard_id}.txt",
            self.data_dir / f"{standard_id}.tsv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None
