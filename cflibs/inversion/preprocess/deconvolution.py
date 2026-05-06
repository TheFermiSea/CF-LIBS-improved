"""Voigt deconvolution for overlapping spectral peaks.

Replaces simple trapezoidal integration with multi-peak Voigt fitting to
recover accurate line intensities (areas) even when peaks are blended.

Two backends are provided:

- **JAX** (preferred): Uses ``jax.scipy.optimize.minimize`` (L-BFGS) for
  fast, differentiable fitting with the Weideman Faddeeva approximation.
- **SciPy** (fallback): Uses ``scipy.optimize.curve_fit`` with the
  ``voigt_profile`` from :mod:`cflibs.radiation.profiles`.

The public entry points are :func:`group_peaks` and :func:`deconvolve_peaks`.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.deconvolution")

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    import jax.scipy.optimize  # noqa: F401  needed at runtime by `jax.scipy.optimize.minimize` below

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

# Lazy-cached import: profiles.py creates Weideman coefficients at import
# time, so we defer until first use to ensure jax_enable_x64 is already set.
_voigt_jax_fn = None


def _get_voigt_profile_jax():
    global _voigt_jax_fn
    if _voigt_jax_fn is None:
        from cflibs.radiation.profiles import voigt_profile_jax

        _voigt_jax_fn = voigt_profile_jax
    return _voigt_jax_fn


try:
    from scipy.optimize import curve_fit
    from scipy.special import wofz as scipy_wofz

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VoigtFitResult:
    """Result for a single fitted Voigt peak.

    Attributes
    ----------
    center_nm : float
        Fitted peak center wavelength in nm.
    amplitude : float
        Fitted peak integrated area.
    sigma_gaussian : float
        Gaussian standard deviation (nm).
    gamma_lorentzian : float
        Lorentzian HWHM (nm).
    area : float
        Analytically computed peak area (equals ``amplitude``).
    area_uncertainty : float
        Estimated 1-sigma uncertainty on ``area``.
    residual_rms : float
        RMS residual of the fit in the peak region.
    """

    center_nm: float
    amplitude: float
    sigma_gaussian: float
    gamma_lorentzian: float
    area: float
    area_uncertainty: float
    residual_rms: float


@dataclass
class DeconvolutionResult:
    """Result of multi-peak Voigt deconvolution for a spectral region.

    Attributes
    ----------
    fit_results : List[VoigtFitResult]
        Per-peak fitting results.
    wavelength : np.ndarray
        Wavelength grid used for fitting.
    fitted_spectrum : np.ndarray
        Sum of all fitted Voigt profiles.
    residuals : np.ndarray
        Observed minus fitted spectrum.
    rms_residual : float
        Root-mean-square of ``residuals``.
    """

    fit_results: List[VoigtFitResult]
    wavelength: np.ndarray
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    rms_residual: float


# ---------------------------------------------------------------------------
# Peak grouping
# ---------------------------------------------------------------------------


def group_peaks(
    peak_wavelengths: np.ndarray,
    fwhm_estimate: float,
    grouping_factor: float = 2.0,
) -> List[List[int]]:
    """Group nearby peaks that may overlap.

    Peaks separated by less than ``grouping_factor * fwhm_estimate`` are
    placed in the same group so they can be fit simultaneously.

    Parameters
    ----------
    peak_wavelengths : np.ndarray
        Wavelength positions of detected peaks in nm.
    fwhm_estimate : float
        Estimated full-width at half-maximum (nm) for a single peak.
    grouping_factor : float
        Peaks within ``grouping_factor * fwhm_estimate`` of any group
        member are merged into that group (default 2.0).

    Returns
    -------
    List[List[int]]
        Each inner list contains the indices (into *peak_wavelengths*)
        of peaks belonging to one group.
    """
    if len(peak_wavelengths) == 0:
        return []

    sorted_indices = np.argsort(peak_wavelengths)
    threshold = grouping_factor * fwhm_estimate

    groups: List[List[int]] = [[int(sorted_indices[0])]]
    for i in range(1, len(sorted_indices)):
        idx = int(sorted_indices[i])
        prev_idx = groups[-1][-1]
        if peak_wavelengths[idx] - peak_wavelengths[prev_idx] <= threshold:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    return groups


# ---------------------------------------------------------------------------
# Voigt model helpers (NumPy / SciPy)
# ---------------------------------------------------------------------------


def _voigt_scipy(
    wavelength: np.ndarray, center: float, sigma: float, gamma: float, amplitude: float
) -> np.ndarray:
    """Single Voigt profile using scipy.special.wofz."""
    sigma = max(sigma, 1e-12)
    gamma = max(gamma, 1e-12)
    z = (wavelength - center + 1j * gamma) / (sigma * np.sqrt(2.0))
    w_z = scipy_wofz(z)
    return amplitude * w_z.real / (sigma * np.sqrt(2.0 * np.pi))


def _multi_voigt_scipy(wavelength: np.ndarray, *params) -> np.ndarray:
    """Sum of N Voigt profiles.  *params* is a flat array of
    [center, sigma, gamma, amplitude] x N_peaks."""
    n_peaks = len(params) // 4
    result = np.zeros_like(wavelength)
    for k in range(n_peaks):
        c, s, g, a = params[4 * k : 4 * k + 4]
        result += _voigt_scipy(wavelength, c, s, g, a)
    return result


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

if HAS_JAX:

    def _voigt_jax_single(
        wavelength: jnp.ndarray, center: float, sigma: float, gamma: float, amplitude: float
    ) -> jnp.ndarray:
        """Single Voigt profile using the Weideman Faddeeva approximation.

        Uses a lazily-imported ``voigt_profile_jax`` from
        :mod:`cflibs.radiation.profiles` to ensure ``jax_enable_x64`` is
        already set before Weideman coefficients are created.
        """
        return _get_voigt_profile_jax()(wavelength, center, sigma, gamma, amplitude)

    def _multi_voigt_jax(wavelength: jnp.ndarray, params: jnp.ndarray, n_peaks: int) -> jnp.ndarray:
        """Sum of *n_peaks* Voigt profiles from a flat parameter vector."""
        result = jnp.zeros_like(wavelength)
        for k in range(n_peaks):
            c = params[4 * k]
            s = params[4 * k + 1]
            g = params[4 * k + 2]
            a = params[4 * k + 3]
            result = result + _voigt_jax_single(wavelength, c, s, g, a)
        return result

    def _fit_group_jax(
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peak_centers: np.ndarray,
        sigma_init: float,
        gamma_init: float,
    ) -> List[VoigtFitResult]:
        """Fit a group of peaks using JAX L-BFGS optimisation."""
        n_peaks = len(peak_centers)
        wl_jax = jnp.asarray(wavelength)
        int_jax = jnp.asarray(intensity)

        # Initial parameter vector: [center, sigma, gamma, amplitude] x n
        # amplitude is area, not peak height
        p0 = []
        for c in peak_centers:
            peak_height = float(intensity[np.argmin(np.abs(wavelength - c))])
            amp_init = max(peak_height * sigma_init * np.sqrt(2 * np.pi), 1.0)
            p0.extend([c, sigma_init, gamma_init, amp_init])
        p0 = jnp.asarray(p0)

        def loss(params):
            model = _multi_voigt_jax(wl_jax, params, n_peaks)
            return jnp.sum((int_jax - model) ** 2)

        result = jax.scipy.optimize.minimize(loss, p0, method="BFGS")
        popt = np.asarray(result.x)

        fitted = np.asarray(_multi_voigt_jax(wl_jax, jnp.asarray(popt), n_peaks))
        residuals = intensity - fitted
        rms = float(np.sqrt(np.mean(residuals**2)))

        # Build per-peak results
        results: List[VoigtFitResult] = []
        for k in range(n_peaks):
            c = float(popt[4 * k])
            s = float(np.abs(popt[4 * k + 1]))
            g = float(np.abs(popt[4 * k + 2]))
            a = float(np.abs(popt[4 * k + 3]))

            # Area = amplitude (by definition of Voigt normalisation)
            area = a
            # Rough uncertainty from RMS and number of data points
            n_data = len(wavelength)
            area_unc = (
                rms * float(np.sqrt(n_data)) * abs(wavelength[-1] - wavelength[0]) / max(n_data, 1)
            )

            results.append(
                VoigtFitResult(
                    center_nm=c,
                    amplitude=a,
                    sigma_gaussian=s,
                    gamma_lorentzian=g,
                    area=area,
                    area_uncertainty=max(area_unc, 1e-10),
                    residual_rms=rms,
                )
            )
        return results


# ---------------------------------------------------------------------------
# SciPy backend
# ---------------------------------------------------------------------------


def _fit_group_scipy(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    peak_centers: np.ndarray,
    sigma_init: float,
    gamma_init: float,
) -> List[VoigtFitResult]:
    """Fit a group of peaks using scipy.optimize.curve_fit."""
    if not HAS_SCIPY:
        raise ImportError("SciPy is required for the fallback deconvolution backend.")

    n_peaks = len(peak_centers)
    wl_step = float(np.median(np.abs(np.diff(wavelength)))) if len(wavelength) > 1 else 1e-3

    # Build initial guesses and bounds.
    # The amplitude parameter is *area* (integrated), not peak height.
    # Peak height ~ area / (sigma * sqrt(2*pi)), so:
    #   area_init ~ peak_height * sigma * sqrt(2*pi)
    p0: List[float] = []
    lower: List[float] = []
    upper: List[float] = []

    for c in peak_centers:
        peak_height = float(intensity[np.argmin(np.abs(wavelength - c))])
        amp_init = max(peak_height * sigma_init * np.sqrt(2 * np.pi), 1.0)
        # Center: allow drift up to half the FWHM
        fwhm_approx = 2.355 * sigma_init + 2.0 * gamma_init
        center_tol = max(fwhm_approx, wl_step * 5)
        p0.extend([c, sigma_init, gamma_init, amp_init])
        lower.extend([c - center_tol, 1e-6, 1e-6, 0.0])
        upper.extend([c + center_tol, sigma_init * 10, gamma_init * 10, amp_init * 20])

    try:
        popt, pcov = curve_fit(
            _multi_voigt_scipy,
            wavelength,
            intensity,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except RuntimeError as exc:
        logger.warning("curve_fit failed: %s -- returning initial guesses", exc)
        popt = np.array(p0)
        pcov = np.full((len(p0), len(p0)), np.inf)

    fitted = _multi_voigt_scipy(wavelength, *popt)
    residuals = intensity - fitted
    rms = float(np.sqrt(np.mean(residuals**2)))

    # Extract per-peak uncertainties from covariance diagonal
    perr = np.sqrt(np.abs(np.diag(pcov))) if np.isfinite(pcov).all() else np.full(len(popt), np.inf)

    results: List[VoigtFitResult] = []
    for k in range(n_peaks):
        c = float(popt[4 * k])
        s = float(popt[4 * k + 1])
        g = float(popt[4 * k + 2])
        a = float(popt[4 * k + 3])
        a_err = float(perr[4 * k + 3])

        results.append(
            VoigtFitResult(
                center_nm=c,
                amplitude=a,
                sigma_gaussian=s,
                gamma_lorentzian=g,
                area=a,
                area_uncertainty=max(a_err, 1e-10),
                residual_rms=rms,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public deconvolution entry point
# ---------------------------------------------------------------------------


def deconvolve_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    peak_wavelengths: np.ndarray,
    fwhm_estimate: float,
    grouping_factor: float = 2.0,
    margin_factor: float = 3.0,
    use_jax: Optional[bool] = None,
) -> DeconvolutionResult:
    """Deconvolve overlapping peaks via multi-peak Voigt fitting.

    Groups nearby peaks (within ``grouping_factor * fwhm_estimate``) and
    fits each group with a sum of Voigt profiles.  Uses JAX L-BFGS when
    available, falling back to ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength grid in nm.
    intensity : np.ndarray
        Baseline-subtracted intensity array.
    peak_wavelengths : np.ndarray
        Detected peak centre wavelengths in nm.
    fwhm_estimate : float
        Estimated FWHM (nm) for a typical peak.  Used for grouping and
        initial width guesses.
    grouping_factor : float
        Peaks within ``grouping_factor * fwhm_estimate`` are fit together
        (default 2.0).
    margin_factor : float
        Fitting window extends ``margin_factor * fwhm_estimate`` beyond
        outermost peaks in each group (default 3.0).
    use_jax : bool, optional
        Force JAX (True) or SciPy (False) backend.  ``None`` auto-detects.

    Returns
    -------
    DeconvolutionResult
        Aggregated fitting results across all peak groups.
    """
    if wavelength.ndim != 1 or intensity.ndim != 1 or wavelength.shape != intensity.shape:
        raise ValueError("wavelength and intensity must be 1-D arrays of equal length")
    if fwhm_estimate <= 0:
        raise ValueError(f"fwhm_estimate must be > 0, got {fwhm_estimate}")
    if grouping_factor <= 0:
        raise ValueError(f"grouping_factor must be > 0, got {grouping_factor}")
    if margin_factor <= 0:
        raise ValueError(f"margin_factor must be > 0, got {margin_factor}")

    if len(wavelength) == 0 or len(peak_wavelengths) == 0:
        return DeconvolutionResult(
            fit_results=[],
            wavelength=wavelength,
            fitted_spectrum=np.zeros_like(wavelength),
            residuals=intensity.copy() if len(intensity) > 0 else np.array([]),
            rms_residual=0.0,
        )

    peak_wl = np.asarray(peak_wavelengths, dtype=float)
    groups = group_peaks(peak_wl, fwhm_estimate, grouping_factor)

    # Initial width guesses from FWHM estimate
    sigma_init = fwhm_estimate / 2.355  # Gaussian sigma from FWHM
    gamma_init = fwhm_estimate / 2.0  # Lorentzian HWHM from FWHM

    # Choose backend
    if use_jax is None:
        backend_jax = HAS_JAX
    else:
        backend_jax = use_jax and HAS_JAX

    if not backend_jax and not HAS_SCIPY:
        raise ImportError(
            "Voigt deconvolution requires JAX or SciPy. "
            "Install one with: pip install jax  or  pip install scipy"
        )

    fit_func = _fit_group_jax if backend_jax else _fit_group_scipy

    all_results: List[VoigtFitResult] = []
    fitted_spectrum = np.zeros_like(wavelength, dtype=float)

    for group in groups:
        centers = peak_wl[group]
        margin = margin_factor * fwhm_estimate

        wl_lo = float(np.min(centers)) - margin
        wl_hi = float(np.max(centers)) + margin
        mask = (wavelength >= wl_lo) & (wavelength <= wl_hi)

        if np.sum(mask) < 4:
            logger.debug(
                "Skipping group at %.2f nm: too few data points (%d)",
                float(np.mean(centers)),
                int(np.sum(mask)),
            )
            continue

        wl_seg = wavelength[mask]
        int_seg = intensity[mask]

        try:
            group_results = fit_func(wl_seg, int_seg, centers, sigma_init, gamma_init)
        except Exception as exc:
            logger.warning(
                "Deconvolution failed for group at %.2f nm: %s",
                float(np.mean(centers)),
                exc,
            )
            continue

        all_results.extend(group_results)

        # Accumulate the fitted spectrum
        if backend_jax and HAS_JAX:
            params_flat = []
            for r in group_results:
                params_flat.extend(
                    [
                        r.center_nm,
                        r.sigma_gaussian,
                        r.gamma_lorentzian,
                        r.amplitude,
                    ]
                )
            params_arr = jnp.asarray(params_flat)
            fitted_seg = np.asarray(
                _multi_voigt_jax(jnp.asarray(wl_seg), params_arr, len(group_results))
            )
        else:
            params_flat = []
            for r in group_results:
                params_flat.extend(
                    [
                        r.center_nm,
                        r.sigma_gaussian,
                        r.gamma_lorentzian,
                        r.amplitude,
                    ]
                )
            fitted_seg = _multi_voigt_scipy(wl_seg, *params_flat)

        fitted_spectrum[mask] += fitted_seg

    residuals = intensity - fitted_spectrum
    rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 0.0

    return DeconvolutionResult(
        fit_results=all_results,
        wavelength=wavelength,
        fitted_spectrum=fitted_spectrum,
        residuals=residuals,
        rms_residual=rms,
    )
