"""
Unit tests for the OPT-IN Columnar-Density Saha-Boltzmann (CD-SB) path in
``cflibs.inversion.physics.self_absorption``.

Reference
---------
A. Safi, S. H. Tavassoli, G. Cristoforetti, S. Legnaioli, V. Palleschi,
F. Rezaei, E. Tognoni, "Determination of excitation temperature in
laser-induced plasmas using columnar density Saha-Boltzmann plot",
J. Adv. Res. 18 (2019) 1-10, https://doi.org/10.1016/j.jare.2019.01.008.

Acceptance property (specification B7-cdsb)
-------------------------------------------
On a synthetic line set WITH self-absorption (known column density + T),
CD-SB recovers T and relative column densities CLOSER TO TRUTH than the
thin-line Boltzmann fit on the SAME (self-absorbed) lines.

The synthetic generator is built directly on the module's own forward optical
-depth physics (``_PI_E2_OVER_MEC_CGS``, the Doppler line-center formula and
the 1.4992 f_lu conversion) so the test is a true round-trip of the CD-SB
inversion, with no real database and no network.
"""

import numpy as np
import pytest

from cflibs.core.constants import C_LIGHT, KB, KB_EV, M_PROTON
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.self_absorption import (
    _PI_E2_OVER_MEC_CGS,
    CDSBLineInput,
    CDSBResult,
    cdsb_quantify,
    column_density_from_optical_depth,
)

# ---------------------------------------------------------------------------
# Synthetic, physics-consistent line generator (no DB, no network)
# ---------------------------------------------------------------------------


def _forward_tau0(n_i_l, A_ki, g_k, g_i, wavelength_nm, T_K, mass_amu):
    """Forward line-center Doppler optical depth (inverse of CD-SB recovery)."""
    lambda_cm = wavelength_nm * 1e-7
    f_lu = 1.4992 * (lambda_cm**2) * A_ki * (g_k / g_i)
    mass_kg = mass_amu * M_PROTON
    v_th = np.sqrt(2.0 * KB * T_K / mass_kg)
    nu_0 = 2.99792458e10 / lambda_cm  # c[cm/s] / lambda[cm]
    delta_nu_D = nu_0 * (v_th / C_LIGHT)
    phi_nu0 = 1.0 / (np.sqrt(np.pi) * delta_nu_D)
    return _PI_E2_OVER_MEC_CGS * f_lu * n_i_l * phi_nu0


def _escape(tau):
    """Photon escape factor f(tau) = (1 - exp(-tau)) / tau."""
    if tau < 1e-10:
        return 1.0
    return (1.0 - np.exp(-tau)) / tau


def _build_synthetic_lines(T_true_K, N_total_l, U_T, mass_amu=56.0):
    """
    Build a Mn/Fe-like neutral multiplet with a known T and known lower-level
    columnar densities n_i*l following a Boltzmann distribution, plus the
    self-absorbed measured intensity for each line.

    Returns
    -------
    cdsb_inputs : list[CDSBLineInput]
    thin_obs : list[LineObservation]   (self-absorbed measured intensities)
    truth : dict with n_i_l_true, T_true_K
    """
    # (wavelength_nm, A_ki, g_k, g_i, E_i_ev, E_k_ev)
    # Spread over lower-level energies; lower lines (resonance, E_i~0) are the
    # strongly self-absorbed ones, exactly the CD-SB use case.
    specs = [
        (279.5, 3.6e8, 6, 6, 0.00, 4.435),  # resonance: heavily self-absorbed
        (280.1, 2.5e8, 4, 4, 0.00, 4.429),  # resonance: heavily self-absorbed
        (357.9, 1.5e8, 8, 6, 0.96, 4.428),  # mildly thick
        (404.1, 7.9e7, 8, 8, 2.11, 5.179),  # near-thin
        (478.3, 5.0e7, 6, 6, 2.92, 5.512),  # thin
        (542.0, 3.0e7, 4, 4, 3.41, 5.700),  # thin
    ]
    T_eV = T_true_K * KB_EV
    cdsb_inputs = []
    thin_obs = []
    n_true = {}
    for wl, A, gk, gi, Ei, Ek in specs:
        # Lower-level columnar density from Boltzmann: n_i l = N l (g_i/U) e^{-E_i/kT}
        n_i_l = N_total_l * (gi / U_T) * np.exp(-Ei / T_eV)
        n_true[wl] = n_i_l

        tau0 = _forward_tau0(n_i_l, A, gk, gi, wl, T_true_K, mass_amu)

        # True optically-thin emission intensity (Boltzmann on the UPPER level):
        # I_thin ∝ (g_k A / lambda) * exp(-E_k / kT). The proportionality
        # constant is common to all lines, so its value is irrelevant to slopes.
        I_thin = (gk * A / wl) * np.exp(-Ek / T_eV)
        # Self-absorption suppresses the measured intensity.
        I_obs = I_thin * _escape(tau0)

        cdsb_inputs.append(
            CDSBLineInput(
                wavelength_nm=wl,
                tau_0=tau0,
                A_ki=A,
                g_k=gk,
                g_i=gi,
                E_i_ev=Ei,
                element="Fe",
                ionization_stage=1,
                mass_amu=mass_amu,
            )
        )
        thin_obs.append(
            LineObservation(
                wavelength_nm=wl,
                intensity=I_obs,
                intensity_uncertainty=0.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=Ek,
                g_k=gk,
                A_ki=A,
            )
        )
    return cdsb_inputs, thin_obs, {"n_i_l_true": n_true, "T_true_K": T_true_K}


def _boltzmann_T_from_intensities(obs):
    """
    Classic thin-line Boltzmann fit: slope of ln(I lambda / (g_k A)) vs E_k.
    T = -1/(k_B * slope). Returns NaN if degenerate.
    """
    x = np.array([o.E_k_ev for o in obs])
    y = np.array([o.y_value for o in obs])  # ln(I lambda / (g_k A))
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    if slope >= 0:
        return np.nan
    return -1.0 / (KB_EV * slope)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_column_density_roundtrip_is_exact():
    """The CD-SB column-density recovery exactly inverts the forward tau_0."""
    n_true = 5.0e15  # cm^-2
    A, gk, gi, wl, T, M = 3.6e8, 6, 6, 279.5, 9000.0, 56.0
    tau0 = _forward_tau0(n_true, A, gk, gi, wl, T, M)
    n_rec = column_density_from_optical_depth(
        tau_0=tau0, A_ki=A, g_k=gk, g_i=gi, wavelength_nm=wl, temperature_K=T, mass_amu=M
    )
    # Algebraic inverse -> machine-precision agreement.
    assert n_rec == pytest.approx(n_true, rel=1e-9)


def test_cdsb_recovers_temperature_better_than_thin_fit():
    """
    ACCEPTANCE: with self-absorption present, CD-SB recovers T closer to truth
    than the thin-line Boltzmann fit on the SAME self-absorbed lines.
    """
    T_true = 9000.0
    # Realistic LIBS regime: resonance lines moderately-to-strongly thick
    # (tau_0 ~ a few), weak high-E_i lines near-thin (tau_0 < 0.1). This is
    # exactly the regime where excluding thick lines wastes information and
    # CD-SB pays off.
    cdsb_inputs, thin_obs, truth = _build_synthetic_lines(
        T_true_K=T_true, N_total_l=3.0e13, U_T=25.0
    )

    # Sanity: the resonance lines really are self-absorbed (tau_0 > 1).
    assert max(ln.tau_0 for ln in cdsb_inputs) > 3.0
    # ... and the weakest line is essentially thin (so the set spans the COG).
    assert min(ln.tau_0 for ln in cdsb_inputs) < 0.1

    # Thin-line Boltzmann fit on the suppressed intensities -> biased T.
    T_thin = _boltzmann_T_from_intensities(thin_obs)
    assert np.isfinite(T_thin)

    # CD-SB fit.
    result = cdsb_quantify(cdsb_inputs, temperature_guess_K=8000.0)
    assert isinstance(result, CDSBResult)
    T_cdsb = result.temperature_K

    err_thin = abs(T_thin - T_true)
    err_cdsb = abs(T_cdsb - T_true)

    # CD-SB must be strictly closer to truth, and the thin fit must actually
    # be biased (otherwise the comparison is vacuous).
    assert err_thin > 200.0, f"thin fit not biased enough: T_thin={T_thin:.0f}"
    assert err_cdsb < err_thin, (
        f"CD-SB ({T_cdsb:.0f} K, err {err_cdsb:.0f}) not closer to truth "
        f"({T_true:.0f} K) than thin fit ({T_thin:.0f} K, err {err_thin:.0f})"
    )
    # CD-SB recovers temperature to good accuracy.
    assert err_cdsb < 0.05 * T_true, f"CD-SB T error too large: {err_cdsb:.0f} K"


def test_cdsb_recovers_relative_column_densities_better_than_thin_fit():
    """
    ACCEPTANCE: CD-SB recovers RELATIVE lower-level column densities closer to
    truth than the thin-line proxy (which uses suppressed intensities).
    """
    T_true = 9000.0
    cdsb_inputs, thin_obs, truth = _build_synthetic_lines(
        T_true_K=T_true, N_total_l=3.0e13, U_T=25.0
    )
    n_true = truth["n_i_l_true"]

    result = cdsb_quantify(cdsb_inputs, temperature_guess_K=8000.0)
    n_cdsb = result.column_densities_cm2

    # Build aligned arrays keyed on wavelength.
    wls = [ln.wavelength_nm for ln in cdsb_inputs]
    true_vec = np.array([n_true[w] for w in wls])
    cdsb_vec = np.array([n_cdsb[w] for w in wls])

    # Thin-line proxy for the lower-level column density: under the optically
    # -thin assumption an analyst would take n_i_l_proxy ∝ I_obs / (g_k A)
    # * exp(+E_k/kT) brought to the lower level via the Boltzmann relation.
    # With self-absorption the strong (resonance) lines are suppressed, so the
    # proxy under-estimates exactly the lines CD-SB is meant to rescue.
    T_eV = T_true * KB_EV
    proxy_vec = []
    for ln, o in zip(cdsb_inputs, thin_obs):
        # n_k_proxy ∝ I lambda / (g_k A); n_i ∝ n_k * (g_i/g_k) exp((E_k-E_i)/kT)
        n_k_proxy = o.intensity * ln.wavelength_nm / (ln.g_k * ln.A_ki)
        n_i_proxy = n_k_proxy * (ln.g_i / ln.g_k) * np.exp((o.E_k_ev - ln.E_i_ev) / T_eV)
        proxy_vec.append(n_i_proxy)
    proxy_vec = np.array(proxy_vec)

    # Normalise to compare RELATIVE column densities (overall scale is free in
    # both methods; CD-SB carries an absolute scale but we compare patterns).
    def _norm(v):
        return v / np.sum(v)

    rel_true = _norm(true_vec)
    rel_cdsb = _norm(cdsb_vec)
    rel_proxy = _norm(proxy_vec)

    err_cdsb = float(np.sum(np.abs(rel_cdsb - rel_true)))
    err_proxy = float(np.sum(np.abs(rel_proxy - rel_true)))

    assert err_cdsb < err_proxy, (
        f"CD-SB relative column densities (L1 err {err_cdsb:.3f}) not closer to "
        f"truth than thin-line proxy (L1 err {err_proxy:.3f})"
    )
    # CD-SB recovers the relative pattern to high fidelity.
    assert err_cdsb < 0.02, f"CD-SB relative column-density error too large: {err_cdsb:.3f}"


def test_cdsb_requires_two_lines_and_energy_spread():
    """CD-SB raises on degenerate inputs (default-path safety, not a no-op)."""
    one = [
        CDSBLineInput(
            wavelength_nm=279.5,
            tau_0=2.0,
            A_ki=3.6e8,
            g_k=6,
            g_i=6,
            E_i_ev=0.0,
            element="Fe",
        )
    ]
    with pytest.raises(ValueError):
        cdsb_quantify(one)

    # Two lines, identical lower-level energy -> no slope.
    same_E = [
        CDSBLineInput(
            wavelength_nm=279.5, tau_0=2.0, A_ki=3.6e8, g_k=6, g_i=6, E_i_ev=0.0, element="Fe"
        ),
        CDSBLineInput(
            wavelength_nm=280.1, tau_0=1.5, A_ki=2.5e8, g_k=4, g_i=4, E_i_ev=0.0, element="Fe"
        ),
    ]
    with pytest.raises(ValueError):
        cdsb_quantify(same_E)
