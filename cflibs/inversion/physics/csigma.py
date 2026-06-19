"""
C-sigma (Csigma) generalized curve-of-growth for CF-LIBS plasma characterization.

This is a **standalone, strictly opt-in** module.  Nothing here is imported by
the default inversion pipeline; the production composition path remains the
Boltzmann / Saha-Boltzmann solver.  Importing and calling the public functions
below is the only way to exercise the C-sigma method.

Overview
--------
The C-sigma (Cσ) method of Aragón & Aguilera generalises the single-element
curve of growth (COG) so that **lines of different elements that share the same
ionization stage** can be pooled onto a *single* graph.  Each line contributes
one point whose

* abscissa is ``x = log10(C · σ)`` — concentration ``C`` times the line
  **absorption cross-section** ``σ`` (computed from the atomic data, ``T`` and
  ``n_e``), and
* ordinate is ``y = log10(I / B(λ, T))`` — the measured **integrated line
  intensity** ``I`` divided by the Planck blackbody radiance ``B(λ, T)`` at the
  line wavelength (the "blackbody factor").

Because the ordinate is built on the **full radiative-transfer relation** for a
homogeneous LTE slab, optically *thin* and optically *thick* lines lie on the
*same* master curve.  In the thin limit the curve is a straight line of slope
+1 (``I ∝ C·σ``); as ``C·σ`` grows the points saturate (slope → +1/2, then
flatten), exactly the COG behaviour.  A common fit of all pooled lines to this
master curve therefore yields, simultaneously,

1. the temperature ``T`` (the abscissa scaling, i.e. the level-population
   Boltzmann factors, are consistent only at the correct ``T``), and
2. the *relative* concentrations of the elements (a per-element horizontal
   offset of each element's sub-cloud of points).

Physics / radiative transfer
----------------------------
For a homogeneous slab of geometric length ``ℓ`` in LTE, the emergent spectral
radiance integrated over the line profile is (Aragón & Aguilera 2014, Eqs. 1-7;
see also Aragón & Aguilera, Spectrochim. Acta B 63 (2008) 893, the COG review)

    I = G · B(λ, T) · ∫ [1 - exp(-τ(λ'))] dλ'                       (RT slab)

where ``B(λ, T)`` is the Planck function (the LTE line source function),
``G`` is a constant geometric/instrumental efficiency factor common to all
lines, and ``τ(λ')`` is the optical depth profile.  The optical-depth at line
centre obeys

    τ0 = k0 · ℓ ∝ C · σ                                            (Beer-Lambert)

so the abscissa ``C·σ`` is, up to the common slab length ``ℓ``, the line-centre
optical depth.  Define the dimensionless curve-of-growth function for a Gaussian
(Doppler) profile of ``1/e`` half-width ``Δλ_D``:

    W(τ0) = (1/Δλ_D) ∫ [1 - exp(-τ0 · exp(-(Δλ/Δλ_D)^2))] dΔλ      (COG)

Limiting forms (used as the analytic check in the unit test):

    τ0 ≪ 1 :  W(τ0) → sqrt(pi) · τ0                (linear, optically thin)
    τ0 ≫ 1 :  W(τ0) → 2 · sqrt(ln τ0)              (square-root, "flat" part)

Hence the ordinate

    y = log10( I / [G · B(λ, T) · Δλ_D] ) = log10( W(τ0) )

is a *universal* function of ``log10(C·σ)`` once the per-line constants
(``Δλ_D``, ``B``) are divided out.  This is precisely what makes the multi-line,
multi-element pooling possible.

Absorption cross-section
------------------------
The integrated absorption cross-section that sets the abscissa is, in LTE
(Aragón & Aguilera 2014, Eq. 6; standard line-absorption coefficient, e.g.
Thorne, *Spectrophysics*):

    σ = (π e² / (4 πε0 m_e c²)) · λ² · (g_k / g_i) · A_ki
        · (g_i / U(T)) · exp(-E_i / k_B T) · [1 - exp(-ΔE / k_B T)]

The pre-factor groups into the classical electron radius; for the C-sigma
*relative* analysis the absolute pre-factor is a constant common to every line
and cancels in the common-line fit.  What matters line-to-line is the
*relative* cross-section

    σ_rel ∝ λ² · g_k · A_ki · exp(-E_i / k_B T) / U(T)             (relative σ)

(in the small-stimulated-emission limit ``ΔE ≫ k_B T`` typical of UV/visible
LIBS lines, the ``[1 - exp(-ΔE/k_BT)]`` factor → 1; it is retained in the code
for completeness).  Note ``g_i exp(-E_i/k_BT)`` is the *lower*-level Boltzmann
factor — the absorbing population — which is what distinguishes σ from the
*emission* coefficient ``∝ g_k A_ki exp(-E_k/k_BT)``.

References
----------
- C. Aragón, J. A. Aguilera, "CSigma graphs: A new approach for plasma
  characterization in laser-induced breakdown spectroscopy",
  J. Quant. Spectrosc. Radiat. Transfer 149 (2014) 90-102,
  doi:10.1016/j.jqsrt.2014.07.026.
- C. Aragón, J. A. Aguilera, "Characterization of laser-induced plasmas by
  optical emission spectroscopy: A review of experiments and methods",
  Spectrochim. Acta B 63 (2008) 893-916 (curve-of-growth foundations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

from cflibs.core.constants import C_LIGHT, H_PLANCK, KB, KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.data_structures import LineObservation

logger = get_logger("inversion.csigma")

# Photon energy conversion (J -> eV); local to avoid importing the SI charge
# under a different alias.
_J_TO_EV = 1.0 / 1.602176634e-19


# ---------------------------------------------------------------------------
# Curve-of-growth function W(tau0) for a Gaussian (Doppler) line profile.
# ---------------------------------------------------------------------------
def cog_function(tau0: float) -> float:
    """Dimensionless curve-of-growth function ``W(τ0)`` for a Gaussian profile.

    Implements (Aragón & Aguilera 2014, COG kernel; Thorne, *Spectrophysics*)

    .. math::

        W(\\tau_0) = \\int_{-\\infty}^{\\infty}
            \\bigl[1 - e^{-\\tau_0\\, e^{-u^2}}\\bigr]\\, du

    with ``u = Δλ / Δλ_D`` the profile-normalised wavelength offset, so ``W`` is
    the equivalent width of ``[1 - exp(-τ)]`` in units of the Doppler width
    ``Δλ_D``.  This is the universal master curve every line collapses onto.

    Limiting behaviour (verified in the unit test):

    * ``τ0 → 0``  : ``W → sqrt(pi) · τ0``     (optically thin, slope 1)
    * ``τ0 → ∞``  : ``W → 2 · sqrt(ln τ0)``   (optically thick, flat part)

    Parameters
    ----------
    tau0 : float
        Optical depth at line centre (``≥ 0``).

    Returns
    -------
    float
        ``W(τ0)`` (dimensionless, ``≥ 0``).
    """
    if tau0 <= 0.0:
        return 0.0
    # Integrand is even in u; integrate [0, inf) and double.  For very small
    # tau0 use the analytic thin limit to avoid catastrophic cancellation.
    if tau0 < 1e-8:
        return np.sqrt(np.pi) * tau0

    def integrand(u: float) -> float:
        return 1.0 - np.exp(-tau0 * np.exp(-(u * u)))

    val, _ = quad(integrand, 0.0, np.inf, limit=200)
    return 2.0 * val


# Vectorised helper for arrays of optical depths.
_cog_vec = np.vectorize(cog_function, otypes=[float])


@dataclass
class CsigmaPoint:
    """One line's contribution to a C-sigma graph.

    Attributes
    ----------
    element : str
        Emitting element symbol.
    ionization_stage : int
        Ionization stage (0 = neutral, 1 = singly ionized, ...).  All points on
        one C-sigma graph must share this value.
    wavelength_nm : float
        Line wavelength (nm).
    log_sigma_rel : float
        ``log10`` of the abscissa ``C · σ_rel`` (the σ common pre-factor is
        dropped; see module docstring).
    log_intensity_ratio : float
        Ordinate ``y = log10(I / B(λ, T))`` — measured integrated intensity
        divided by the Planck blackbody factor.
    """

    element: str
    ionization_stage: int
    wavelength_nm: float
    log_sigma_rel: float
    log_intensity_ratio: float


@dataclass
class CsigmaFitResult:
    """Result of a C-sigma common-line fit.

    Attributes
    ----------
    temperature_K : float
        Best-fit excitation/ionization temperature (K).
    relative_concentrations : dict[str, float]
        Per-element relative concentration, normalised so the values sum to 1
        over the elements present in the graph.
    scale : float
        Common multiplicative scale (``log10`` offset) absorbing the geometric
        slab length, instrumental efficiency and the σ pre-factor.
    residual_rms : float
        Root-mean-square residual of ``y_model - y_obs`` (dex).
    n_points : int
        Number of pooled line points used in the fit.
    n_thick : int
        Number of points whose fitted line-centre optical depth ``τ0 > 1``
        (i.e. the optically-thick lines the method explicitly handles).
    converged : bool
        Whether the least-squares optimiser reported convergence.
    """

    temperature_K: float
    relative_concentrations: dict[str, float]
    scale: float
    residual_rms: float
    n_points: int
    n_thick: int
    converged: bool = True
    points: list[CsigmaPoint] | None = field(default=None, repr=False)


def _planck_radiance(wavelength_nm: float, temperature_K: float) -> float:
    """Planck spectral radiance ``B(λ, T)`` (W m^-3 sr^-1, per-wavelength form).

    .. math::

        B(\\lambda, T) = \\frac{2 h c^2}{\\lambda^5}
            \\frac{1}{e^{hc / \\lambda k_B T} - 1}

    Only relative values matter in the C-sigma ordinate, but the physical form
    is used so the blackbody-factor division is exact.
    """
    lam = wavelength_nm * 1e-9  # nm -> m
    x = (H_PLANCK * C_LIGHT) / (lam * KB * temperature_K)
    # Guard against overflow for the UV / cool-plasma corner.
    return (2.0 * H_PLANCK * C_LIGHT**2) / (lam**5) / np.expm1(x)


def _log_sigma_rel(obs: LineObservation, temperature_K: float) -> float:
    """``log10`` of the relative absorption cross-section for one line.

    Uses the relative cross-section (module docstring, "relative σ"):

    .. math::

        \\sigma_{rel} \\propto \\lambda^2\\, g_k\\, A_{ki}\\,
            e^{-E_i / k_B T} \\bigl[1 - e^{-\\Delta E / k_B T}\\bigr]

    The lower-level energy is ``E_i = E_k - ΔE`` with ``ΔE = hc/λ`` the photon
    energy.  The species partition function ``U(T)`` is common to every line of
    a given element, so it acts as a per-element offset and is folded into the
    relative-concentration fit; it is omitted from the abscissa here precisely
    so the fit can recover the per-element offsets.  ``g_k`` and ``A_ki`` come
    from the atomic data carried on the :class:`LineObservation`.
    """
    lam_nm = obs.wavelength_nm
    lam_m = lam_nm * 1e-9
    photon_eV = (H_PLANCK * C_LIGHT) / lam_m * _J_TO_EV
    e_lower_eV = obs.E_k_ev - photon_eV
    kt_eV = KB_EV * temperature_K
    boltz_lower = np.exp(-e_lower_eV / kt_eV)
    stim = 1.0 - np.exp(-photon_eV / kt_eV)
    sigma_rel = (lam_nm**2) * obs.g_k * obs.A_ki * boltz_lower * stim
    return float(np.log10(sigma_rel))


def build_csigma_graph(
    observations: Sequence[LineObservation],
    temperature_K: float,
    concentrations: Mapping[str, float] | None = None,
) -> list[CsigmaPoint]:
    """Build the C-sigma graph points for a set of same-stage line observations.

    Each observation becomes one :class:`CsigmaPoint` with abscissa
    ``log10(C · σ_rel)`` and ordinate ``log10(I / B(λ, T))`` (Aragón & Aguilera
    2014).  All observations must share one ionization stage (a C-sigma graph is
    *per ionization stage*); a ``ValueError`` is raised otherwise.

    Parameters
    ----------
    observations : sequence of LineObservation
        Measured lines of one ionization stage, carrying ``intensity``,
        ``wavelength_nm``, ``E_k_ev``, ``g_k`` and ``A_ki``.
    temperature_K : float
        Temperature used to evaluate the Boltzmann factor and blackbody factor.
    concentrations : mapping str -> float, optional
        Relative concentration per element used for the abscissa ``C·σ``.  When
        ``None`` (default), ``C = 1`` for every element (the abscissa is then
        the pure cross-section ``σ``); the common-line fit recovers the
        concentrations as per-element offsets.

    Returns
    -------
    list[CsigmaPoint]
    """
    if not observations:
        raise ValueError("build_csigma_graph requires at least one observation")
    stages = {o.ionization_stage for o in observations}
    if len(stages) != 1:
        raise ValueError(
            f"C-sigma graph mixes ionization stages {sorted(stages)}; " "build one graph per stage."
        )
    stage = stages.pop()
    points: list[CsigmaPoint] = []
    for obs in observations:
        if obs.intensity <= 0.0:
            logger.debug("Skipping non-positive intensity line at %.3f nm", obs.wavelength_nm)
            continue
        conc = 1.0 if concentrations is None else float(concentrations.get(obs.element, 1.0))
        log_sigma = _log_sigma_rel(obs, temperature_K)
        x = float(np.log10(conc) + log_sigma)
        b = _planck_radiance(obs.wavelength_nm, temperature_K)
        y = float(np.log10(obs.intensity / b))
        points.append(
            CsigmaPoint(
                element=obs.element,
                ionization_stage=stage,
                wavelength_nm=obs.wavelength_nm,
                log_sigma_rel=x,
                log_intensity_ratio=y,
            )
        )
    if not points:
        raise ValueError("No usable (positive-intensity) lines for C-sigma graph")
    return points


def _csigma_model(
    log_temperature_K: float,
    log_conc: np.ndarray,
    scale: float,
    observations: Sequence[LineObservation],
    elem_index: np.ndarray,
) -> np.ndarray:
    """Predict the C-sigma ordinate ``y_i`` for each line given fit parameters.

    Combines the relative cross-section, concentration offsets and the COG
    function into

    .. math::

        y_i = \\log_{10} W(\\tau_{0,i}),\\quad
        \\tau_{0,i} = 10^{\\,scale + \\log_{10} C_{e(i)} + \\log_{10}\\sigma_i(T)}

    The free parameters are ``T`` (through ``log10 T``), one log-concentration
    per element and a single common ``scale`` (slab length × efficiency × σ
    pre-factor).  This is the heart of the common-line fit.
    """
    temperature_K = 10.0**log_temperature_K
    log_sigma = np.array([_log_sigma_rel(o, temperature_K) for o in observations])
    log_tau0 = scale + log_conc[elem_index] + log_sigma
    tau0 = np.power(10.0, log_tau0)
    w = _cog_vec(tau0)
    # log10 of W; guard the (numerically) zero-W floor.
    return np.log10(np.maximum(w, 1e-300))


def fit_csigma(
    observations: Sequence[LineObservation],
    *,
    t_init_K: float = 10000.0,
    t_bounds_K: tuple[float, float] = (3000.0, 30000.0),
    return_points: bool = True,
) -> CsigmaFitResult:
    """Common-line fit of a C-sigma graph -> temperature and relative concentrations.

    All observations must belong to **one ionization stage** (Aragón & Aguilera
    2014: a C-sigma graph pools different *elements* but a single ionic stage).
    The model fits, by non-linear least squares on the ordinate
    ``y = log10(I / B(λ, T))``,

    * one temperature ``T`` (shared),
    * one relative concentration per element, and
    * one common scale (geometry × efficiency × σ pre-factor),

    using the full radiative-transfer COG ordinate ``log10 W(τ0)`` so that
    optically thin *and* thick lines are fit together.

    Parameters
    ----------
    observations : sequence of LineObservation
        Measured same-stage lines from >= 2 elements (more than one element is
        what makes the per-element concentrations identifiable; a single element
        still yields ``T`` and an overall scale).
    t_init_K : float, optional
        Initial temperature guess (K).  Default 10000.
    t_bounds_K : tuple, optional
        Lower/upper temperature bounds (K).  Default ``(3000, 30000)``.
    return_points : bool, optional
        Attach the built :class:`CsigmaPoint` list to the result.  Default True.

    Returns
    -------
    CsigmaFitResult
    """
    stages = {o.ionization_stage for o in observations}
    if len(stages) != 1:
        raise ValueError(f"fit_csigma requires a single ionization stage, got {sorted(stages)}")
    usable = [o for o in observations if o.intensity > 0.0]
    if len(usable) < 3:
        raise ValueError("fit_csigma needs at least 3 positive-intensity lines")

    elements = sorted({o.element for o in usable})
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    elem_index = np.array([elem_to_idx[o.element] for o in usable], dtype=int)
    n_elem = len(elements)

    def residuals(params: np.ndarray) -> np.ndarray:
        log_T = params[0]
        temperature_K = 10.0**log_T
        # First element's concentration is the reference (log_conc[0] = 0); the
        # remaining n_elem - 1 are free, absorbed together with the common scale.
        log_conc = np.concatenate(([0.0], params[1 : 1 + (n_elem - 1)]))
        scale = params[1 + (n_elem - 1)]
        # Observed ordinate uses B(lambda, T) at the *current* T (the blackbody
        # factor moves the data, so it must be inside the residual).
        y_obs = np.array(
            [
                np.log10(o.intensity / _planck_radiance(o.wavelength_nm, temperature_K))
                for o in usable
            ]
        )
        y_model = _csigma_model(log_T, log_conc, scale, usable, elem_index)
        return y_model - y_obs

    # Parameter vector: [log10 T, log_conc_2..n (n_elem-1), scale]
    p0 = np.concatenate(([np.log10(t_init_K)], np.zeros(n_elem - 1), [0.0]))
    lb = np.concatenate(([np.log10(t_bounds_K[0])], np.full(n_elem - 1, -6.0), [-50.0]))
    ub = np.concatenate(([np.log10(t_bounds_K[1])], np.full(n_elem - 1, 6.0), [50.0]))

    sol = least_squares(residuals, p0, bounds=(lb, ub), method="trf", max_nfev=4000)

    log_T = sol.x[0]
    temperature_K = float(10.0**log_T)
    log_conc = np.concatenate(([0.0], sol.x[1 : 1 + (n_elem - 1)]))
    scale = float(sol.x[1 + (n_elem - 1)])

    conc = np.power(10.0, log_conc)
    conc = conc / conc.sum()
    relative_concentrations = {e: float(c) for e, c in zip(elements, conc)}

    res = sol.fun
    residual_rms = float(np.sqrt(np.mean(res**2))) if res.size else 0.0

    # Count optically-thick lines (tau0 > 1) at the solution.
    log_sigma = np.array([_log_sigma_rel(o, temperature_K) for o in usable])
    log_tau0 = scale + log_conc[elem_index] + log_sigma
    n_thick = int(np.sum(log_tau0 > 0.0))

    points = build_csigma_graph(usable, temperature_K) if return_points else None

    return CsigmaFitResult(
        temperature_K=temperature_K,
        relative_concentrations=relative_concentrations,
        scale=scale,
        residual_rms=residual_rms,
        n_points=len(usable),
        n_thick=n_thick,
        converged=bool(sol.success),
        points=points,
    )
