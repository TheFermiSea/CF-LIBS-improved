"""
Spectral-response (radiometric) correction for measured spectra.

CF-LIBS is built on *relative line intensities across wide wavelength spans*:
the Boltzmann ordinate ``ln(I*lambda / g*A)`` compares lines that may sit
hundreds of nm apart, so the wavelength-dependent detection efficiency
``E(lambda)`` of the collection + spectrometer + detector chain enters the
intercepts directly (``delta_q = ln E``) and rotates the Boltzmann slope
(temperature bias). Correcting the measured intensities for ``E(lambda)`` is
listed among the experimental prerequisites of the method (Ciucci et al. 1999;
Tognoni et al., Spectrochim. Acta B 65 (2010) 1-14, section 2). See audit
finding F5 in ``docs/audit/2026-06-09-overhaul/02-inversion-solver.md``.

Convention
----------
Only the **relative** spectral response matters for CF-LIBS: the closure
equation (sum of concentrations = 1) cancels any wavelength-independent scale
factor. The curve is therefore normalized to ``max(E) = 1`` on load -- the
same convention as the forward model's
:meth:`cflibs.instrument.model.InstrumentModel.apply_response` -- and an
absolutely-calibrated (irradiance) curve works unchanged.

The correction is its own inverse pair with the forward model: the forward
model multiplies synthetic spectra by ``E(lambda)``; this module divides
measured spectra by the same curve representation (an ``(N, 2)`` array of
``(wavelength_nm, relative_efficiency)`` rows, CSV-loadable exactly like
``instrument.response_curve``).

When to use
-----------
Any spectrometer without upstream radiometric calibration -- e.g. an in-house
ps-LIBS instrument. ChemCam/SuperCam CCS data arrive with the response
correction already applied (see ``cflibs/pds/chemcam.py``), so the hook
defaults to identity (``None``) and must stay off for those datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.preprocess.response_correction")

__all__ = [
    "ResponseCurveCoverageError",
    "SpectralResponseCorrection",
    "apply_response_correction",
    "derive_response_from_argon_branching_ratios",
    "load_response_curve",
]


class ResponseCurveCoverageError(ValueError):
    """Raised when a response curve does not cover the measured spectrum range."""


def _load_curve_csv(path: Path) -> np.ndarray:
    """Load a ``wavelength_nm,relative_efficiency`` CSV (header row optional)."""
    try:
        data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    except ValueError:
        # Tolerate a single text header row ("wavelength_nm,relative_efficiency").
        data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2, skiprows=1)
    return data


def _load_curve_yaml(path: Path) -> np.ndarray:
    """Load a YAML mapping with ``wavelength_nm`` / ``relative_efficiency`` lists."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Response-curve YAML {path} must be a mapping with "
            "'wavelength_nm' and 'relative_efficiency' lists."
        )
    missing = {"wavelength_nm", "relative_efficiency"} - set(payload)
    if missing:
        raise ValueError(f"Response-curve YAML {path} missing key(s): {sorted(missing)}")
    wl = np.asarray(payload["wavelength_nm"], dtype=float)
    eff = np.asarray(payload["relative_efficiency"], dtype=float)
    if wl.shape != eff.shape:
        raise ValueError(
            f"Response-curve YAML {path}: 'wavelength_nm' ({wl.size} values) and "
            f"'relative_efficiency' ({eff.size} values) must have equal length."
        )
    return np.column_stack([wl, eff])


def load_response_curve(path: Union[str, Path]) -> np.ndarray:
    """
    Load a spectral response curve from CSV or YAML.

    The returned representation is the **same** ``(N, 2)`` array of
    ``(wavelength_nm, relative_efficiency)`` rows used by
    :attr:`cflibs.instrument.model.InstrumentModel.response_curve`, so a single
    file serves both the forward model (multiply) and the inversion (divide).

    Parameters
    ----------
    path : str or Path
        ``.csv`` file with two comma-separated columns
        (``wavelength_nm,relative_efficiency``; ``#`` comments and an optional
        header row are tolerated), or a ``.yaml``/``.yml`` mapping with
        ``wavelength_nm`` and ``relative_efficiency`` lists.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` float array sorted by wavelength.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has fewer than 2 points, non-finite values, duplicate
        wavelengths, or non-positive efficiencies.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Response curve file not found: {path}")

    if path.suffix.lower() in (".yaml", ".yml"):
        data = _load_curve_yaml(path)
    else:
        data = _load_curve_csv(path)

    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Response curve {path} must have two columns "
            f"(wavelength_nm, relative_efficiency); got shape {data.shape}."
        )
    curve = data[:, :2]
    _validate_curve(curve, source=str(path))
    return curve[np.argsort(curve[:, 0])]


def _validate_curve(curve: np.ndarray, source: str = "") -> None:
    """Validate an ``(N, 2)`` response-curve array (shape/finite/positive/unique)."""
    label = f" ({source})" if source else ""
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(f"Response curve{label} must be an (N, 2) array; got {curve.shape}.")
    if curve.shape[0] < 2:
        raise ValueError(
            f"Response curve{label} needs at least 2 points; got {curve.shape[0]}."
        )
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"Response curve{label} contains non-finite values.")
    wl = curve[:, 0]
    if np.unique(wl).size != wl.size:
        raise ValueError(f"Response curve{label} contains duplicate wavelengths.")
    eff = curve[:, 1]
    if np.any(eff <= 0.0):
        raise ValueError(
            f"Response curve{label} contains non-positive efficiencies; the inversion "
            "divides by E(lambda), so every tabulated efficiency must be > 0. Trim the "
            "curve (and the spectrum) to the wavelength range the instrument actually "
            "detects."
        )


@dataclass
class SpectralResponseCorrection:
    """
    Divide measured intensities by the relative detection efficiency E(lambda).

    Inverse of the forward model's
    :meth:`cflibs.instrument.model.InstrumentModel.apply_response` (which
    multiplies synthetic spectra by the same curve). Applied to the measured
    spectrum *before* line detection / observation building, so every
    downstream quantity -- integrated line intensities, their shot-noise
    uncertainties, Boltzmann ordinates -- is computed from response-corrected
    data.

    Because the correction is a deterministic multiplicative factor, a line
    intensity ``I`` and its 1-sigma uncertainty ``sigma_I`` transform
    identically (``I -> I/E``, ``sigma_I -> sigma_I/E``); relative
    uncertainties are preserved. :meth:`apply` accepts an optional uncertainty
    array for callers that hold explicit ``(I, sigma_I)`` pairs.

    Attributes
    ----------
    response_curve : np.ndarray
        ``(N, 2)`` array of ``(wavelength_nm, relative_efficiency)`` rows.
        Normalized to ``max(E) = 1`` on construction (relative convention --
        CF-LIBS closure cancels any absolute scale).
    extrapolation_margin_nm : float
        Maximum distance (nm) the spectrum may extend beyond the tabulated
        curve on either side. Within the margin the edge value is held
        (constant extrapolation) and a warning is logged; beyond it,
        :class:`ResponseCurveCoverageError` is raised listing the coverage.
    source : str
        Provenance label (file path) used in log/error messages.
    """

    response_curve: np.ndarray
    extrapolation_margin_nm: float = 5.0
    source: str = ""
    _wl: np.ndarray = field(init=False, repr=False)
    _eff: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        curve = np.asarray(self.response_curve, dtype=float)
        _validate_curve(curve, source=self.source)
        curve = curve[np.argsort(curve[:, 0])]
        # Relative convention: normalize to max = 1 (same as the forward
        # model's InstrumentModel.apply_response).
        curve = curve.copy()
        curve[:, 1] = curve[:, 1] / curve[:, 1].max()
        self.response_curve = curve
        self._wl = curve[:, 0]
        self._eff = curve[:, 1]
        if float(self._eff.min()) < 1e-3:
            logger.warning(
                "Response curve%s drops below 0.1%% of peak efficiency "
                "(min E = %.2e). Dividing by E(lambda) there amplifies noise "
                "by >1000x; consider restricting the analysis window.",
                f" ({self.source})" if self.source else "",
                float(self._eff.min()),
            )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        extrapolation_margin_nm: float = 5.0,
    ) -> "SpectralResponseCorrection":
        """Load a correction from a CSV/YAML file (see :func:`load_response_curve`)."""
        curve = load_response_curve(path)
        return cls(
            response_curve=curve,
            extrapolation_margin_nm=extrapolation_margin_nm,
            source=str(path),
        )

    @property
    def coverage_nm(self) -> Tuple[float, float]:
        """Tabulated wavelength coverage ``(min_nm, max_nm)`` of the curve."""
        return float(self._wl[0]), float(self._wl[-1])

    def validate_coverage(self, wavelength: np.ndarray) -> None:
        """
        Check that the curve covers the spectrum's wavelength range.

        Raises
        ------
        ResponseCurveCoverageError
            If the spectrum extends more than ``extrapolation_margin_nm``
            beyond the tabulated curve on either side. The message lists the
            curve coverage and the spectrum range so the gap is actionable.
        """
        wl = np.asarray(wavelength, dtype=float)
        if wl.size == 0:
            return
        spec_min, spec_max = float(np.min(wl)), float(np.max(wl))
        cov_min, cov_max = self.coverage_nm
        margin = float(self.extrapolation_margin_nm)
        label = f" ({self.source})" if self.source else ""

        under = cov_min - spec_min
        over = spec_max - cov_max
        if under > margin or over > margin:
            raise ResponseCurveCoverageError(
                f"Response curve{label} does not cover the measured spectrum: "
                f"curve coverage [{cov_min:.2f}, {cov_max:.2f}] nm vs spectrum "
                f"[{spec_min:.2f}, {spec_max:.2f}] nm "
                f"(uncovered: {max(under, 0.0):.2f} nm below, "
                f"{max(over, 0.0):.2f} nm above; allowed extrapolation margin "
                f"{margin:.2f} nm). Provide a curve measured over the full "
                "spectral range, or restrict the spectrum to the covered range."
            )
        if under > 0.0 or over > 0.0:
            logger.warning(
                "Response curve%s extrapolated (edge-hold) beyond its tabulated "
                "range: curve [%.2f, %.2f] nm vs spectrum [%.2f, %.2f] nm "
                "(within the %.2f nm margin). Efficiencies outside the table "
                "are held at the edge value.",
                label,
                cov_min,
                cov_max,
                spec_min,
                spec_max,
                margin,
            )

    def efficiency(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Interpolate the relative efficiency E(lambda) onto ``wavelength``.

        Linear interpolation; outside the tabulated range the edge value is
        held (``np.interp`` semantics). Coverage must be validated separately
        (:meth:`validate_coverage`) or via :meth:`apply`.
        """
        wl = np.asarray(wavelength, dtype=float)
        return np.interp(wl, self._wl, self._eff)

    def apply(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        intensity_uncertainty: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return the response-corrected intensity ``I / E(lambda)``.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength axis (nm).
        intensity : np.ndarray
            Measured intensities on that axis.
        intensity_uncertainty : np.ndarray, optional
            1-sigma intensity uncertainties; divided by the same E(lambda)
            (deterministic multiplicative correction -- relative uncertainty
            is preserved).

        Returns
        -------
        np.ndarray or (np.ndarray, np.ndarray)
            Corrected intensity, or ``(intensity, uncertainty)`` when an
            uncertainty array is given.

        Raises
        ------
        ResponseCurveCoverageError
            If the curve does not cover the spectrum (see
            :meth:`validate_coverage`).
        """
        self.validate_coverage(wavelength)
        eff = self.efficiency(wavelength)
        corrected = np.asarray(intensity, dtype=float) / eff
        if intensity_uncertainty is None:
            return corrected
        corrected_unc = np.asarray(intensity_uncertainty, dtype=float) / eff
        return corrected, corrected_unc


def apply_response_correction(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    correction: Optional[SpectralResponseCorrection] = None,
) -> np.ndarray:
    """
    Apply an optional spectral-response correction (identity when ``None``).

    The identity path returns the *same* ``intensity`` object unchanged --
    pinned by a regression test so the default pipeline behaviour is
    bit-identical with and without the hook.
    """
    if correction is None:
        return intensity
    result = correction.apply(wavelength, intensity)
    assert isinstance(result, np.ndarray)
    return result


def derive_response_from_argon_branching_ratios(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    *,
    branching_ratio_table: Optional[np.ndarray] = None,
) -> SpectralResponseCorrection:
    """
    Lamp-free relative spectral response from argon branching ratios (stub).

    Placeholder API for internal (standards-free) calibration: lines sharing
    an upper level have intensity ratios fixed by their transition
    probabilities (branching ratios), independent of the plasma state. The
    deviation of measured Ar line ratios from the tabulated branching ratios
    therefore measures the relative response E(lambda) at the line
    wavelengths -- no calibrated lamp required. Validated against a quartz
    tungsten halogen lamp above 350 nm in the LIBS context.

    Intended use on the 1 ps / 1040 nm instrument: record an LIBS plasma in an
    argon atmosphere (or an Ar-filled hollow-cathode lamp), pass the spectrum
    here, and feed the returned :class:`SpectralResponseCorrection` to the
    inversion pipeline (``--response-curve`` accepts a file produced from it).

    References
    ----------
    "Relative spectral response calibration of a spectrometer system for laser
    induced breakdown spectroscopy using the argon branching ratio method",
    J. Anal. At. Spectrom. 29 (2014) 657-664, DOI 10.1039/C3JA50371B.

    Whaling et al., "Argon branching ratios for spectrometer response
    calibration", J. Quant. Spectrosc. Radiat. Transf. 50 (1993) 7-18
    (branching-ratio tables spanning 210-4591 nm).

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured argon emission spectrum.
    branching_ratio_table : np.ndarray, optional
        Override table of ``(wavelength_nm, upper_level_id, branching_ratio)``
        rows; defaults to the Whaling 1993 Ar I/Ar II tables once implemented.

    Raises
    ------
    NotImplementedError
        Always -- this is a designed-but-unimplemented surface. Tracked as
        follow-up to bead CF-LIBS-improved-gzwd.
    """
    raise NotImplementedError(
        "Argon branching-ratio internal calibration is not implemented yet. "
        "Planned method: J. Anal. At. Spectrom. 29 (2014) 657-664, "
        "DOI 10.1039/C3JA50371B, using the Whaling et al. JQSRT 50 (1993) 7-18 "
        "Ar branching-ratio tables. Until then, supply a measured response "
        "curve (calibrated lamp) via SpectralResponseCorrection.from_file()."
    )
