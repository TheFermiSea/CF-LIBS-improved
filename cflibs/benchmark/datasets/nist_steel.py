"""
NIST SRM 1261a-1265a low-alloy steel benchmark adapter.

This module hard-codes the certified mass-fraction compositions for the five
"1200 series" NIST Standard Reference Materials commonly used as LIBS
calibration standards:

==========  =========================================  ============================
SRM ID      Material                                   AISI / informal name
==========  =========================================  ============================
1261a       Low-alloy AISI 4340 steel                  Cr-Mo-Ni structural steel
1262a       Low-alloy AISI 94B17 (modified) steel      B-alloyed case-hardening
1263a       Cr-V steel (modified)                      Tool / spring steel proxy
1264a       High-carbon steel (modified)               Tool steel, ~0.87 wt% C
1265a       Electrolytic iron                          High-purity Fe matrix blank
==========  =========================================  ============================

Composition source
------------------
Values are transcribed verbatim from the public NIST Certificates of Analysis:

- SRM 1261a -- ``https://tsapps.nist.gov/srmext/certificates/archives/1261a.pdf``
  (Issue 25 Aug 2008; original 24 Feb 1981).
- SRM 1262a -- ``https://tsapps.nist.gov/srmext/certificates/archives/1262a.pdf``
  (24 Feb 1981, reproduced in the National Bureau of Standards format).
- SRM 1263a -- ``https://tsapps.nist.gov/srmext/certificates/archives/1263a.pdf``
  (24 Feb 1981).
- SRM 1264a -- ``https://tsapps.nist.gov/srmext/certificates/1264a.pdf``
  (Revision 22 Feb 2019).
- SRM 1265a -- ``https://tsapps.nist.gov/srmext/certificates/1265a.pdf``
  (Revision 26 Jul 2019).

Uncertainty convention
----------------------
- Modern certificates (1264a, 1265a) report a combined standard uncertainty
  ``u_c`` (1-sigma, k=1) in mass percent. We store that value directly in
  :attr:`NISTSteelComposition.uncertainty_wt_pct`.
- Legacy certificates (1261a, 1262a, 1263a) state: "the value listed is not
  expected to deviate from the true value by more than +/- 1 in the last
  significant figure reported; for a subscript figure, the deviation is not
  expected to be more than +/- 5". We translate that prose convention into a
  symmetric +/- bound by inspecting the trailing digit / subscript and storing
  half of the bound as a 1-sigma proxy is *not* attempted here -- we keep the
  certificate's own +/- bound verbatim so users can decide how to interpret it.
- "Information" (non-certified) values from each certificate are stored under
  :attr:`NISTSteelComposition.information_values` without uncertainty.

Iron is provided implicitly. Where a certificate gives an "Iron (by difference)"
information value we record it in ``information_values["Fe"]`` so the certified
table sums to <100 wt% by construction; ``total_certified_wt_pct`` exposes the
sum for sanity checks.

Spectra source
--------------
NIST does not publish LIBS spectra alongside its SRM Certificates of Analysis.
The author searched Mendeley Data, Zenodo, the Open LIBS DB, and recent
SpectroChim. Acta B / JAAS issues for redistributable spectra of the 1261a-1265a
series and found none that are unencumbered by login walls or non-redistribution
licences (e.g., the Optica AS 2023 "Adversarial Data Augmentation" paper uses
this series but ships no public CSVs). Users with their own measurements should
drop a CSV per SRM into ``data/nist_steel/`` named ``{srm_id}.csv`` (two
columns: wavelength_nm, intensity); :meth:`NISTSteelDataset.get_spectrum`
returns ``None`` when no such file is present.
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
    "NISTSteelComposition",
    "NISTSteelDataset",
]


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "nist_steel"
"""Default directory inspected by :meth:`NISTSteelDataset.get_spectrum`."""


@dataclass(frozen=True)
class NISTSteelComposition:
    """
    Certified composition of one NIST SRM 1261a-1265a steel/iron disk.

    Attributes
    ----------
    srm_id : str
        Catalogue number, e.g. ``"1261a"``.
    material : str
        Human-readable description (e.g. ``"AISI 4340 Steel"``).
    certified_wt_pct : dict[str, float]
        Element symbol -> certified mass percent (sum is <= 100 because Fe is
        the matrix balance and is not always certified).
    uncertainty_wt_pct : dict[str, float]
        Element symbol -> 1-sigma combined standard uncertainty in wt% (where
        the certificate reports one). Legacy certificates report a +/- bound
        in the last significant digit; that bound is stored verbatim and is
        *not* divided by any coverage factor.
    information_values : dict[str, float]
        Non-certified ("information only") mass percents from the certificate's
        secondary table. ``"Fe"`` typically lives here as an "iron by
        difference" sanity check.
    iron_balance : bool
        ``True`` when Fe is treated as the matrix balance and is *not* in
        ``certified_wt_pct``. All five SRMs in this series satisfy this.
    certificate_url : str
        Public URL of the Certificate of Analysis these numbers were
        transcribed from (kept on the dataclass for traceability).
    notes : str
        Free-form provenance/expansion notes.
    """

    srm_id: str
    material: str
    certified_wt_pct: dict[str, float]
    uncertainty_wt_pct: dict[str, float] = field(default_factory=dict)
    information_values: dict[str, float] = field(default_factory=dict)
    iron_balance: bool = True
    certificate_url: str = ""
    notes: str = ""

    @property
    def total_certified_wt_pct(self) -> float:
        """Sum of certified mass percents (Fe excluded when ``iron_balance``)."""
        return float(sum(self.certified_wt_pct.values()))

    @property
    def implied_iron_wt_pct(self) -> float:
        """100 - sum(certified) when Fe is the matrix balance, else NaN."""
        if not self.iron_balance:
            return float("nan")
        return 100.0 - self.total_certified_wt_pct

    def as_mass_fractions(self, include_iron: bool = True) -> dict[str, float]:
        """
        Return composition as mass *fractions* (sum ~= 1.0 when iron included).

        Parameters
        ----------
        include_iron : bool, default True
            Add Fe = (100 - sum(certified)) / 100 to the dict so the result
            sums to ~1.0. Only meaningful when ``iron_balance`` is set.
        """
        out = {el: float(v) / 100.0 for el, v in self.certified_wt_pct.items()}
        if include_iron and self.iron_balance:
            out["Fe"] = self.implied_iron_wt_pct / 100.0
        return out


# ---------------------------------------------------------------------------
# Certified compositions transcribed from public NIST Certificates of Analysis.
# ---------------------------------------------------------------------------
#
# Legacy certificates (1261a, 1262a, 1263a) use the convention "+/- 1 in the
# last significant digit; for a subscript digit, +/- 5". We translate that into
# the absolute uncertainty in wt% by reading the printed precision. Subscript
# digits in the original (e.g. 0.39_1) appear as the trailing digit here and
# carry an uncertainty of 5 in the next decimal place.
#
# Modern certificates (1264a, 1265a) report combined standard uncertainties u_c
# explicitly; those are reproduced verbatim.

_SRM_1261A = NISTSteelComposition(
    srm_id="1261a",
    material="AISI 4340 Steel (low-alloy)",
    certified_wt_pct={
        "C": 0.391,
        "Mn": 0.67,
        "P": 0.016,
        "S": 0.015,
        "Si": 0.228,
        "Cu": 0.042,
        "Ni": 2.00,
        "Cr": 0.693,
        "V": 0.011,
        "Mo": 0.19,
        "W": 0.017,
        "Co": 0.032,
        "Ti": 0.020,
        "As": 0.017,
        "Sn": 0.010,
        "Al": 0.021,
        "Nb": 0.022,
        "Ta": 0.020,
        "B": 0.0005,
        "Pb": 0.000025,
        "Zr": 0.009,
        "Sb": 0.0042,
        "Bi": 0.0004,
        "Ag": 0.0004,
        "Ca": 0.000028,
        "Mg": 0.00018,
        "Se": 0.004,
        "Te": 0.0006,
        "Ce": 0.0014,
        "La": 0.0004,
        "Nd": 0.00029,
    },
    uncertainty_wt_pct={
        "C": 0.005,
        "Mn": 0.01,
        "P": 0.001,
        "S": 0.001,
        "Si": 0.001,
        "Cu": 0.001,
        "Ni": 0.01,
        "Cr": 0.005,
        "V": 0.001,
        "Mo": 0.01,
        "W": 0.001,
        "Co": 0.001,
        "Ti": 0.001,
        "As": 0.001,
        "Sn": 0.001,
        "Al": 0.001,
        "Nb": 0.001,
        "Ta": 0.001,
        "B": 0.0001,
        "Pb": 0.000005,
        "Zr": 0.001,
        "Sb": 0.0001,
        "Bi": 0.0001,
        "Ag": 0.0001,
        "Ca": 0.000005,
        "Mg": 0.00001,
        "Se": 0.001,
        "Te": 0.0001,
        "Ce": 0.0001,
        "La": 0.0001,
        "Nd": 0.00005,
    },
    information_values={
        "Au": 0.00005,  # reported as "< 0.00005"
        "Zn": 0.0001,
        "Pr": 0.00014,
        "Hf": 0.0002,
        "N": 0.0037,
        "O": 0.0009,
        "H": 0.0005,  # "< 0.0005"
        "Sr": 0.0005,  # "< 0.0005"
        "Ge": 0.006,
        "Fe": 95.6,  # "Iron (by difference)"
    },
    iron_balance=True,
    certificate_url="https://tsapps.nist.gov/srmext/certificates/archives/1261a.pdf",
    notes=(
        "Legacy NBS/NIST 1981 certificate, editorial revision 2008-08-25. "
        "Uncertainties follow the certificate's own +/- bound on the last "
        "significant digit (subscript digits => +/- 5 in the next place)."
    ),
)


_SRM_1262A = NISTSteelComposition(
    srm_id="1262a",
    material="AISI 94B17 Steel (Modified)",
    certified_wt_pct={
        "C": 0.163,
        "Mn": 1.05,
        "P": 0.044,
        "S": 0.037,
        "Si": 0.40,
        "Cu": 0.51,
        "Ni": 0.60,
        "Cr": 0.30,
        "V": 0.041,
        "Mo": 0.070,
        "W": 0.20,
        "Co": 0.30,
        "Ti": 0.085,
        "As": 0.095,
        "Sn": 0.016,
        "Al": 0.095,
        "Nb": 0.30,
        "Ta": 0.21,
        "B": 0.0025,
        "Pb": 0.00043,
        "Zr": 0.20,
        "Sb": 0.0120,
        "Ag": 0.0011,
        "Ca": 0.00014,
        "Mg": 0.00062,
        "Te": 0.0011,
        "Ce": 0.0015,
        "La": 0.0004,
        "Nd": 0.00064,
    },
    uncertainty_wt_pct={
        "C": 0.005,
        "Mn": 0.01,
        "P": 0.001,
        "S": 0.001,
        "Si": 0.01,
        "Cu": 0.01,
        "Ni": 0.01,
        "Cr": 0.01,
        "V": 0.005,
        "Mo": 0.005,
        "W": 0.01,
        "Co": 0.01,
        "Ti": 0.001,
        "As": 0.005,
        "Sn": 0.001,
        "Al": 0.005,
        "Nb": 0.01,
        "Ta": 0.01,
        "B": 0.0001,
        "Pb": 0.00005,
        "Zr": 0.01,
        "Sb": 0.0005,
        "Ag": 0.0001,
        "Ca": 0.00001,
        "Mg": 0.00001,
        "Te": 0.0001,
        "Ce": 0.0001,
        "La": 0.0001,
        "Nd": 0.00005,
    },
    information_values={
        "Bi": 0.002,
        "Au": 0.00005,  # "< 0.00005"
        "Se": 0.0012,
        "Zn": 0.0005,
        "Pr": 0.00012,
        "Hf": 0.0003,
        "N": 0.00404,
        "O": 0.00107,
        "H": 0.0005,  # "< 0.0005"
        "Sr": 0.0005,  # "< 0.0005"
        "Ge": 0.002,
        "Fe": 95.3,  # "Iron (by difference)"
    },
    iron_balance=True,
    certificate_url="https://tsapps.nist.gov/srmext/certificates/archives/1262a.pdf",
    notes=(
        "Original NBS certificate dated 1981-02-24. Uncertainties follow the "
        "certificate's own +/- bound on the last significant digit."
    ),
)


_SRM_1263A = NISTSteelComposition(
    srm_id="1263a",
    material="Cr-V Steel (Modified)",
    certified_wt_pct={
        "C": 0.626,
        "Mn": 1.50,
        "P": 0.029,
        "S": 0.0057,
        "Si": 0.74,
        "Cu": 0.098,
        "Ni": 0.32,
        "Cr": 1.31,
        "V": 0.31,
        "Mo": 0.030,
        "W": 0.046,
        "Co": 0.048,
        "Ti": 0.050,
        "As": 0.010,
        "Sn": 0.104,
        "Al": 0.24,
        "Nb": 0.049,
        "B": 0.00091,
        "Pb": 0.0022,
        "Zr": 0.050,
        "Sb": 0.002,
        "Ag": 0.0037,
        "Au": 0.0005,
        "Ca": 0.00013,
        "Mg": 0.00049,
        "Te": 0.0009,
        "Ce": 0.0014,
        "La": 0.0006,
        "Nd": 0.00060,
    },
    uncertainty_wt_pct={
        "C": 0.005,
        "Mn": 0.01,
        "P": 0.001,
        "S": 0.0005,
        "Si": 0.01,
        "Cu": 0.005,
        "Ni": 0.01,
        "Cr": 0.01,
        "V": 0.01,
        "Mo": 0.001,
        "W": 0.001,
        "Co": 0.001,
        "Ti": 0.001,
        "As": 0.001,
        "Sn": 0.005,
        "Al": 0.01,
        "Nb": 0.001,
        "B": 0.00005,
        "Pb": 0.0001,
        "Zr": 0.001,
        "Sb": 0.001,
        "Ag": 0.0001,
        "Au": 0.0001,
        "Ca": 0.00001,
        "Mg": 0.00001,
        "Te": 0.0001,
        "Ce": 0.0001,
        "La": 0.0001,
        "Nd": 0.00005,
    },
    information_values={
        "Ta": 0.053,
        "Bi": 0.0008,
        "Se": 0.00016,
        "Zn": 0.0004,
        "Pr": 0.00018,
        "Hf": 0.0005,
        "N": 0.0041,
        "O": 0.00066,
        "H": 0.0005,  # "< 0.0005"
        "Sr": 0.0005,  # "< 0.0005"
        "Ge": 0.010,
        "Fe": 94.4,  # "Iron (by difference)"
    },
    iron_balance=True,
    certificate_url="https://tsapps.nist.gov/srmext/certificates/archives/1263a.pdf",
    notes=(
        "Original NBS certificate dated 1981-02-24. Uncertainties follow the "
        "certificate's own +/- bound on the last significant digit."
    ),
)


_SRM_1264A = NISTSteelComposition(
    srm_id="1264a",
    material="High-Carbon Steel (Modified)",
    certified_wt_pct={
        "Sb": 0.034,
        "As": 0.052,
        "C": 0.871,
        "Ca": 0.00004,
        "Ce": 0.00022,
        "Cr": 0.066,
        "Co": 0.15,
        "Cu": 0.250,
        "La": 0.00007,
        "Pb": 0.024,
        "Mg": 0.00015,
        "Mn": 0.258,
        "Mo": 0.49,
        "Nd": 0.00007,
        "Ni": 0.142,
        "Nb": 0.157,
        "P": 0.010,
        "Si": 0.067,
        "S": 0.025,
        "Ta": 0.11,
        "Te": 0.00018,
        "Ti": 0.24,
        "W": 0.102,
        "V": 0.106,
        "Zr": 0.069,
    },
    uncertainty_wt_pct={
        "Sb": 0.001,
        "As": 0.005,
        "C": 0.005,
        "Ca": 0.00001,
        "Ce": 0.00005,
        "Cr": 0.005,
        "Co": 0.01,
        "Cu": 0.005,
        "La": 0.00001,
        "Pb": 0.001,
        "Mg": 0.00001,
        "Mn": 0.005,
        "Mo": 0.01,
        "Nd": 0.00001,
        "Ni": 0.005,
        "Nb": 0.005,
        "P": 0.001,
        "Si": 0.001,
        "S": 0.001,
        "Ta": 0.01,
        "Te": 0.00001,
        "Ti": 0.01,
        "W": 0.005,
        "V": 0.005,
        "Zr": 0.001,
    },
    information_values={
        "Al": 0.008,
        "Bi": 0.0009,
        "B": 0.011,
        "Ge": 0.003,
        "Au": 0.0001,
        "Hf": 0.0013,
        "H": 0.0005,  # "< 0.0005"
        "N": 0.0032,
        "O": 0.0010,
        "Pr": 0.00003,
        "Se": 0.00021,
        "Ag": 0.00002,
        "Sr": 0.0005,
        "Sn": 0.008,
        "Zn": 0.001,
        "Fe": 96.7,  # informational matrix value
    },
    iron_balance=True,
    certificate_url="https://tsapps.nist.gov/srmext/certificates/1264a.pdf",
    notes=(
        "Revision dated 2019-02-22. Uncertainties are combined standard "
        "uncertainties u_c (k=1, ~68% confidence) per the JCGM Guide."
    ),
)


_SRM_1265A = NISTSteelComposition(
    srm_id="1265a",
    material="Electrolytic Iron",
    certified_wt_pct={
        "B": 0.00013,
        "C": 0.0067,
        "Cr": 0.0072,
        "Co": 0.0070,
        "Cu": 0.0058,
        "Pb": 0.000015,
        "Mn": 0.0057,
        "Mo": 0.0050,
        "Ni": 0.041,
        "P": 0.0011,
        "Si": 0.0080,
        "S": 0.0055,
        "V": 0.0006,
    },
    uncertainty_wt_pct={
        "B": 0.00001,
        "C": 0.0003,
        "Cr": 0.0005,
        "Co": 0.0005,
        "Cu": 0.0001,
        "Pb": 0.000005,
        "Mn": 0.0001,
        "Mo": 0.0001,
        "Ni": 0.001,
        "P": 0.0001,
        "Si": 0.0005,
        "S": 0.0003,
        "V": 0.0001,
    },
    information_values={
        "Al": 0.0007,
        "Sb": 0.00005,  # "< 0.00005"
        "As": 0.0002,
        "Ge": 0.0014,
        "H": 0.0005,  # "< 0.0005"
        "N": 0.0011,
        "O": 0.0063,
        "Ag": 0.000002,
        "Sn": 0.0002,
        "Ti": 0.0001,
        "W": 0.00004,
        "Zn": 0.0001,  # "< 0.0001"
        "Fe": 99.9,
    },
    iron_balance=True,
    certificate_url="https://tsapps.nist.gov/srmext/certificates/1265a.pdf",
    notes=(
        "Revision dated 2019-07-26. Uncertainties are combined standard "
        "uncertainties u_c (k=1, ~68% confidence) per the JCGM Guide."
    ),
)


CERTIFIED_COMPOSITIONS: dict[str, NISTSteelComposition] = {
    comp.srm_id: comp for comp in (_SRM_1261A, _SRM_1262A, _SRM_1263A, _SRM_1264A, _SRM_1265A)
}
"""All five NIST SRM 1200-series compositions, keyed by SRM id."""


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class NISTSteelDataset:
    """
    Benchmark adapter for the NIST SRM 1261a-1265a low-alloy steel series.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory inspected by :meth:`get_spectrum` for ``{srm_id}.csv``
        files (or any extension supported by :func:`cflibs.io.spectrum.load_spectrum`).
        Defaults to ``<repo>/data/nist_steel/``.

    Notes
    -----
    Mirrors the call surface of the existing Aalto adapter
    (:func:`cflibs.benchmark.unified.load_aalto_id_dataset`):

    - :meth:`available_samples` enumerates known SRM ids.
    - :meth:`get_composition` returns the certified composition record (always
      available -- the data is hard-coded).
    - :meth:`get_spectrum` returns a :class:`BenchmarkSpectrum` if a CSV is
      cached in ``data_dir``, otherwise ``None`` (NIST does not publish LIBS
      spectra alongside the chemical certificates).
    """

    DATASET_NAME = "nist_steel_1200_series"
    INSTRUMENT_ID = "user_supplied"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    # ------------------------------------------------------------------ API

    def available_samples(self) -> list[str]:
        """Return all SRM ids covered by the certified-composition table."""
        return sorted(CERTIFIED_COMPOSITIONS.keys())

    def get_composition(self, srm_id: str) -> NISTSteelComposition:
        """
        Return the certified composition for ``srm_id``.

        Raises
        ------
        KeyError
            If ``srm_id`` is not one of the five 1200-series SRMs.
        """
        try:
            return CERTIFIED_COMPOSITIONS[srm_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown NIST steel SRM id {srm_id!r}; " f"available: {self.available_samples()}"
            ) from exc

    def get_spectrum(self, srm_id: str) -> Optional[BenchmarkSpectrum]:
        """
        Return the cached LIBS spectrum for ``srm_id`` if one is on disk.

        Looks for ``data_dir/{srm_id}.csv`` (and a few common alternates) and
        wraps the result in a :class:`BenchmarkSpectrum` whose
        ``true_composition`` is the certified mass-fraction vector. Returns
        ``None`` -- not raising -- when no spectrum file is present, so callers
        can iterate over :meth:`available_samples` without try/except.
        """
        if srm_id not in CERTIFIED_COMPOSITIONS:
            raise KeyError(
                f"Unknown NIST steel SRM id {srm_id!r}; " f"available: {self.available_samples()}"
            )

        spectrum_path = self._locate_spectrum_file(srm_id)
        if spectrum_path is None:
            return None

        wavelength, intensity = load_spectrum(str(spectrum_path))
        composition = self.get_composition(srm_id)
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
            sample_id=f"NIST_SRM_{srm_id}",
            sample_type=SampleType.CRM,
            matrix_type=MatrixType.METAL_ALLOY,
            crm_name=f"NIST SRM {srm_id}",
            crm_source="NIST",
            preparation="polished disk",
            surface_condition="polished",
            doi=composition.certificate_url,
            provenance=(
                f"Composition: {composition.certificate_url}; "
                f"spectrum file: {spectrum_path.name}"
            ),
        )
        return BenchmarkSpectrum(
            spectrum_id=f"nist_srm_{srm_id}",
            wavelength_nm=wavelength,
            intensity=intensity,
            true_composition=composition.as_mass_fractions(include_iron=True),
            composition_uncertainty={
                el: float(u) / 100.0 for el, u in composition.uncertainty_wt_pct.items()
            },
            conditions=conditions,
            metadata=metadata,
            dataset_id=self.DATASET_NAME,
            group_id=srm_id,
            specimen_id=srm_id,
            instrument_id=self.INSTRUMENT_ID,
            truth_type=TruthType.ASSAY,
            spectrum_kind="alloy",
            annotations={
                "srm_id": srm_id,
                "material": composition.material,
                "certificate_url": composition.certificate_url,
            },
        )

    # ------------------------------------------------------------- helpers

    def _locate_spectrum_file(self, srm_id: str) -> Optional[Path]:
        """Look up a cached spectrum file for ``srm_id`` in ``data_dir``."""
        if not self.data_dir.is_dir():
            return None
        # Try a few common naming conventions; first hit wins.
        candidates = [
            self.data_dir / f"{srm_id}.csv",
            self.data_dir / f"SRM_{srm_id}.csv",
            self.data_dir / f"nist_{srm_id}.csv",
            self.data_dir / f"{srm_id}.txt",
            self.data_dir / f"{srm_id}.tsv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None
