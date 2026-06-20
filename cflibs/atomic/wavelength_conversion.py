"""Air ↔ vacuum wavelength conversion.

Atomic line lists do not agree on the wavelength medium:

- **NIST ASD** tabulates *observed* wavelengths in **air** for 200 nm < λ < 2000 nm and in
  **vacuum** outside that band (the standard spectroscopic convention).
- **Kurucz** and **VALD3** ship line wavelengths in **vacuum**.

Mixing the two without converting introduces a systematic offset of ~+0.14 nm at 500 nm
(``n_air − 1 ≈ 2.8e-4``), which is several resolution elements for a typical LIBS spectrometer.
That offset corrupts every cross-database line match, the wavelength solution, and the forward
model unless one medium is enforced everywhere (Ciucci/Tognoni). This module is that single
converter: apply it at ingest so the line list, the wavelength solution, and the forward model
share one convention.

The refractive index of standard air follows Morton (2000) / Birch & Downs (1994) — the IAU
convention used by NIST and VALD. Two mutually-consistent closed forms are used so neither
direction requires iteration:

- ``vacuum_to_air_nm`` uses the vacuum-input dispersion (Morton 2000, Eq. given for λ > 2000 Å).
- ``air_to_vacuum_nm`` uses the explicit air-input inverse dispersion (Morton 2000; the same
  coefficients adopted by SDSS), so it is the analytic inverse rather than a fixed-point iterate.

Below ``boundary_nm`` (default 200 nm) both line lists are tabulated in vacuum, so the
conversion is the identity there — applying an air correction to an already-vacuum value would
be wrong. The boundary is handled element-wise for array inputs.

References
----------
Morton, D. C. (2000), ApJS 130, 403 (atomic data tables; air/vacuum dispersion relations).
Birch, K. P. & Downs, M. J. (1994), Metrologia 31, 315 (updated Edlén refractive index of air).
"""

from __future__ import annotations

import numpy as np

__all__ = ["air_to_vacuum_nm", "vacuum_to_air_nm", "AIR_VACUUM_BOUNDARY_NM"]

# Below this wavelength NIST/Kurucz/VALD all tabulate in vacuum, so air<->vacuum is the identity.
AIR_VACUUM_BOUNDARY_NM: float = 200.0


def _wavenumber_um_inv(wavelength_nm: np.ndarray) -> np.ndarray:
    """Vacuum-ish wavenumber σ in μm⁻¹ used by the dispersion polynomials (σ = 1000 / λ[nm])."""
    return 1.0e3 / wavelength_nm


def air_to_vacuum_nm(wavelength_air_nm, boundary_nm: float = AIR_VACUUM_BOUNDARY_NM):
    """Convert air wavelength(s) to vacuum (nm).

    Uses the Morton (2000) air-input dispersion (also used by SDSS), which is the analytic
    inverse of :func:`vacuum_to_air_nm` and needs no iteration. Wavelengths at or below
    ``boundary_nm`` are returned unchanged (already vacuum by convention).

    Parameters
    ----------
    wavelength_air_nm : float or array_like
        Air wavelength(s) in nm.
    boundary_nm : float, optional
        Convention boundary (default 200 nm); values ``<= boundary_nm`` pass through unchanged.

    Returns
    -------
    float or numpy.ndarray
        Vacuum wavelength(s) in nm (scalar in → scalar out).
    """
    wl = np.asarray(wavelength_air_nm, dtype=float)
    s2 = _wavenumber_um_inv(wl) ** 2
    n = (
        1.0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (130.1065924 - s2)
        + 1.599740894897e-4 / (38.92568793293 - s2)
    )
    vac = wl * n
    out = np.where(wl <= boundary_nm, wl, vac)
    return float(out) if np.isscalar(wavelength_air_nm) or np.ndim(wavelength_air_nm) == 0 else out


def vacuum_to_air_nm(wavelength_vacuum_nm, boundary_nm: float = AIR_VACUUM_BOUNDARY_NM):
    """Convert vacuum wavelength(s) to air (nm).

    Uses the Morton (2000) vacuum-input dispersion (the NIST/VALD convention for λ > 200 nm).
    Wavelengths at or below ``boundary_nm`` are returned unchanged (reported in vacuum by
    convention).

    Parameters
    ----------
    wavelength_vacuum_nm : float or array_like
        Vacuum wavelength(s) in nm.
    boundary_nm : float, optional
        Convention boundary (default 200 nm); values ``<= boundary_nm`` pass through unchanged.

    Returns
    -------
    float or numpy.ndarray
        Air wavelength(s) in nm (scalar in → scalar out).
    """
    wl = np.asarray(wavelength_vacuum_nm, dtype=float)
    s2 = _wavenumber_um_inv(wl) ** 2
    n = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
    air = wl / n
    out = np.where(wl <= boundary_nm, wl, air)
    return (
        float(out)
        if np.isscalar(wavelength_vacuum_nm) or np.ndim(wavelength_vacuum_nm) == 0
        else out
    )
