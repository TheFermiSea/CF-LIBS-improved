"""Unit tests for the observable-gated self-absorption correction (bead 0jvr).

Covers the three rungs of the correction ladder:

1. Doublet intensity ratio (Pace 2025; machinery from bead 1fcg) — recovery
   of a hand-computed tau from a rigged same-upper-level pair.
2. Planck-ceiling closed form (Völker & Gornushkin 2023, JAAS,
   doi:10.1039/D2JA00352J) — tau from the measured peak spectral radiance
   relative to B_lambda(T), with the published tau <= 3 validity gate; plus a
   rigged optically-thick synthetic generated through the SAME slab
   radiative-transfer formula as the Wave-1 forward kernel
   (cflibs.radiation.kernels._apply_self_absorption, I = B(1 - exp(-kL))),
   verifying the correction moves intensities TOWARD the thin truth and never
   beyond physical bounds.
3. SA-suspect flagging — bright low-E_i lines without observables are
   down-weighted (uncertainty inflated), never boosted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.self_absorption import _escape_factor
from cflibs.inversion.physics.self_absorption_observable import (
    DOUBLET_TAU_VALIDITY_MAX,
    PLANCK_TAU_VALIDITY_MAX,
    ObservableSelfAbsorptionCorrector,
    correct_intensity_planck,
    doppler_cog_escape_factor,
    normalize_self_absorption_mode,
    planck_ceiling_optical_depth,
    planck_spectral_radiance,
)

pytestmark = pytest.mark.unit


def _obs(wl, intensity, element="Fe", stage=1, E_k=5.0, g_k=4, A_ki=1e8, unc_frac=0.01):
    return LineObservation(
        wavelength_nm=wl,
        intensity=intensity,
        intensity_uncertainty=unc_frac * intensity,
        element=element,
        ionization_stage=stage,
        E_k_ev=E_k,
        g_k=g_k,
        A_ki=A_ki,
    )


def _rigged_doublet(tau_1: float, lam1=400.0, lam2=500.0, E_k=5.0, g_k=4, A_ki=1e8):
    """Same-upper-level pair attenuated with hand-computed escape factors.

    Thin emission ratio r_thin = lam2/lam1 (equal gA); optical-depth link
    rho = (lam1/lam2)^3 so tau_2 = tau_1/rho.
    """
    r_thin = lam2 / lam1
    rho = (lam1 / lam2) ** 3
    tau_2 = tau_1 / rho
    i1 = r_thin * _escape_factor(tau_1)
    i2 = 1.0 * _escape_factor(tau_2)
    line1 = _obs(lam1, i1, E_k=E_k, g_k=g_k, A_ki=A_ki)
    line2 = _obs(lam2, i2, E_k=E_k, g_k=g_k, A_ki=A_ki)
    return line1, line2, tau_2, r_thin


# ---------------------------------------------------------------------------
# (a) doublet-ratio rung
# ---------------------------------------------------------------------------


class TestDoubletRung:
    def test_hand_computed_tau_recovered_and_intensities_restored(self):
        """tau_1 = 1.5 rigged pair -> corrector recovers tau and thin truth."""
        tau_1 = 1.5
        line1, line2, tau_2, r_thin = _rigged_doublet(tau_1)
        # High-E_k third line, optically thin, untouched.
        bystander = _obs(600.0, 0.3, E_k=7.5)

        corrector = ObservableSelfAbsorptionCorrector()
        result = corrector.correct([line1, line2, bystander])

        assert result.n_corrected == 2
        assert result.n_suspect == 0
        # max tau is the pair's larger (longer-wavelength) member: tau_2
        assert result.max_tau == pytest.approx(tau_2, rel=1e-3)

        out = {o.wavelength_nm: o for o in result.observations}
        # thin truth: I1 = r_thin, I2 = 1.0
        assert out[400.0].intensity == pytest.approx(r_thin, rel=1e-3)
        assert out[500.0].intensity == pytest.approx(1.0, rel=1e-3)
        # bystander untouched (bit-identical object)
        assert out[600.0] is bystander

        rec = result.corrections[400.0]
        assert rec.method == "doublet"
        assert rec.tau == pytest.approx(tau_1, rel=1e-3)
        assert rec.correction_factor == pytest.approx(1.0 / _escape_factor(tau_1), rel=1e-3)

    def test_corrected_uncertainty_scales_with_factor(self):
        line1, line2, _, _ = _rigged_doublet(2.0)
        result = ObservableSelfAbsorptionCorrector().correct([line1, line2])
        out = {o.wavelength_nm: o for o in result.observations}
        for wl, original in ((400.0, line1), (500.0, line2)):
            factor = result.corrections[wl].correction_factor
            assert out[wl].intensity_uncertainty == pytest.approx(
                original.intensity_uncertainty * factor, rel=1e-9
            )

    def test_thin_pair_is_cleared_not_corrected(self):
        """A pair measured at exactly the thin ratio: no correction, cleared."""
        r_thin = 500.0 / 400.0
        line1 = _obs(400.0, r_thin)
        line2 = _obs(500.0, 1.0)
        result = ObservableSelfAbsorptionCorrector().correct([line1, line2])
        assert result.n_corrected == 0
        assert result.n_suspect == 0
        assert result.corrections[400.0].method == "doublet-thin"
        assert result.observations[0].intensity == pytest.approx(r_thin)

    def test_noise_insignificant_deviation_not_corrected(self):
        """Ratio deviation within 1 sigma of intensity noise -> treated thin."""
        r_thin = 500.0 / 400.0
        # 2% deviation, 10% intensity uncertainties -> ~0.14 sigma
        line1 = _obs(400.0, r_thin * 0.98, unc_frac=0.10)
        line2 = _obs(500.0, 1.0, unc_frac=0.10)
        result = ObservableSelfAbsorptionCorrector().correct([line1, line2])
        assert result.n_corrected == 0
        assert result.corrections[400.0].method == "doublet-thin"

    def test_beyond_validity_ceiling_flags_suspect_instead_of_boosting(self):
        """Pair implying tau above the validity ceiling is down-weighted."""
        tau_1 = 8.0  # tau_2 = 8 / 0.512 ~ 15.6 >> ceiling
        line1, line2, _, _ = _rigged_doublet(tau_1)
        result = ObservableSelfAbsorptionCorrector().correct([line1, line2])
        assert result.n_corrected == 0
        assert result.n_suspect == 2
        out = {o.wavelength_nm: o for o in result.observations}
        # No boost: intensities unchanged, uncertainties inflated.
        assert out[400.0].intensity == pytest.approx(line1.intensity)
        assert out[400.0].intensity_uncertainty > line1.intensity_uncertainty
        assert result.corrections[400.0].suspect is True
        assert result.max_tau > DOUBLET_TAU_VALIDITY_MAX


# ---------------------------------------------------------------------------
# (b) Planck-ceiling rung (Völker & Gornushkin 2023)
# ---------------------------------------------------------------------------


class TestPlanckRung:
    T_K = 10000.0
    WL = 400.0

    def test_tau_roundtrip_from_slab_peak(self):
        """tau_0 = -ln(1 - I_peak/B) inverts the slab solution exactly."""
        b = planck_spectral_radiance(self.WL, self.T_K)
        for tau in (0.1, 1.0, 2.9):
            peak = b * (1.0 - math.exp(-tau))
            assert planck_ceiling_optical_depth(peak, self.WL, self.T_K) == pytest.approx(
                tau, rel=1e-9
            )

    def test_peak_at_or_above_ceiling_is_invalid(self):
        b = planck_spectral_radiance(self.WL, self.T_K)
        assert planck_ceiling_optical_depth(b, self.WL, self.T_K) is None
        assert planck_ceiling_optical_depth(1.5 * b, self.WL, self.T_K) is None

    def test_validity_gate_tau_above_3(self):
        """Völker & Gornushkin validity: tau_0 <= ~3 (10% RSD budget)."""
        b = planck_spectral_radiance(self.WL, self.T_K)
        peak_thick = b * (1.0 - math.exp(-4.5))
        pc = correct_intensity_planck(1.0, peak_thick, self.WL, self.T_K)
        assert not pc.valid
        assert pc.correction_factor == 1.0
        assert pc.tau_0 > PLANCK_TAU_VALIDITY_MAX

    def test_doppler_cog_escape_factor_matches_numerical_integral(self):
        for tau in (0.3, 1.0, 2.5):
            x = np.linspace(-8.0, 8.0, 200_001)
            profile = tau * np.exp(-(x**2))
            numeric = np.trapezoid(1.0 - np.exp(-profile), x) / (math.sqrt(math.pi) * tau)
            assert doppler_cog_escape_factor(tau) == pytest.approx(numeric, rel=1e-8)

    def test_slab_rt_thick_spectrum_corrected_toward_thin_truth(self):
        """Rigged optically-thick synthetic through the Wave-1 slab RT path.

        Generates a Gaussian emission line through the SAME radiative-transfer
        formula as cflibs.radiation.kernels._apply_self_absorption
        (I = B(1 - exp(-kappa L)), kappa = emissivity/B), then corrects the
        integrated intensity with the Planck-ceiling method using only the two
        observables (peak spectral radiance, temperature). The corrected
        intensity must move TOWARD the optically-thin truth (emissivity * L)
        and never beyond physical bounds.
        """
        T_K = self.T_K
        wl0 = self.WL
        sigma_nm = 0.02
        path_m = 0.01
        b = planck_spectral_radiance(wl0, T_K)
        tau_0 = 2.0

        wl = np.linspace(wl0 - 1.0, wl0 + 1.0, 40_001)
        # Emissivity profile chosen so the line-center optical depth is tau_0.
        kappa_per_m = (tau_0 / path_m) * np.exp(-0.5 * ((wl - wl0) / sigma_nm) ** 2)
        emissivity = kappa_per_m * b  # W m^-2 nm^-1 sr^-1 per metre

        # Wave-1 slab RT (kernels.py): I = B * (1 - exp(-kappa L))
        i_thick = b * (1.0 - np.exp(-kappa_per_m * path_m))
        i_thin_truth = emissivity * path_m  # optically-thin limit

        area_thick = float(np.trapezoid(i_thick, wl))
        area_thin = float(np.trapezoid(i_thin_truth, wl))
        assert area_thick < area_thin  # SA attenuates

        peak_measured = float(i_thick.max())
        pc = correct_intensity_planck(area_thick, peak_measured, wl0, T_K)

        assert pc.valid
        assert pc.tau_0 == pytest.approx(tau_0, rel=1e-6)
        # Moves toward the thin truth...
        assert abs(pc.corrected_intensity - area_thin) < abs(area_thick - area_thin)
        # ...recovers it to better than 1% (exact up to grid integration error
        # because f_G is the exact Gaussian-profile attenuation)...
        assert pc.corrected_intensity == pytest.approx(area_thin, rel=1e-2)
        # ...and never beyond physical bounds: above the observed, at most the
        # thin truth (within numerical tolerance).
        assert pc.corrected_intensity >= area_thick
        assert pc.corrected_intensity <= area_thin * (1.0 + 1e-2)

    def test_corrector_uses_planck_observable_when_supplied(self):
        """Lines with calibrated peak radiance + T go through rung (b)."""
        b = planck_spectral_radiance(self.WL, self.T_K)
        tau = 1.2
        thin_area = 5.0e-3
        observed = thin_area * doppler_cog_escape_factor(tau)
        line = _obs(self.WL, observed, E_k=8.0)  # high E_i: not a suspect
        peak = b * (1.0 - math.exp(-tau))

        corrector = ObservableSelfAbsorptionCorrector()
        result = corrector.correct(
            [line],
            temperature_K=self.T_K,
            peak_spectral_radiance={self.WL: peak},
        )
        assert result.n_corrected == 1
        assert result.corrections[self.WL].method == "planck"
        assert result.observations[0].intensity == pytest.approx(thin_area, rel=1e-6)


# ---------------------------------------------------------------------------
# (c) SA-suspect rung
# ---------------------------------------------------------------------------


class TestSuspectRung:
    def test_bright_resonance_line_without_observable_down_weighted(self):
        # Resonance line: E_k = photon energy -> E_i = 0 (1239.84/400 = 3.0996)
        photon_ev = 1239.841984 / 400.0
        resonance = _obs(400.0, 100.0, E_k=photon_ev, unc_frac=0.02)
        weak_high = _obs(450.0, 1.0, E_k=7.0)
        result = ObservableSelfAbsorptionCorrector().correct([resonance, weak_high])

        assert result.n_suspect == 1
        assert result.corrections[400.0].suspect is True
        out = {o.wavelength_nm: o for o in result.observations}
        # Down-weighted: uncertainty inflated by the configured factor...
        assert out[400.0].intensity_uncertainty == pytest.approx(
            resonance.intensity_uncertainty * 3.0
        )
        # ...intensity NEVER boosted.
        assert out[400.0].intensity == pytest.approx(resonance.intensity)
        # High-E_i line untouched.
        assert out[450.0] is weak_high

    def test_high_E_i_lines_are_not_suspects(self):
        lines = [_obs(400.0 + i, 10.0 * (i + 1), E_k=7.0 + 0.2 * i) for i in range(4)]
        result = ObservableSelfAbsorptionCorrector().correct(lines)
        assert result.n_suspect == 0
        assert result.n_corrected == 0
        assert all(a is b for a, b in zip(result.observations, lines))

    def test_doublet_cleared_line_exempt_from_suspicion(self):
        """A resonance-energy pair whose measured ratio is thin is cleared."""
        photon_ev = 1239.841984 / 400.0
        r_thin = 500.0 / 400.0
        line1 = _obs(400.0, 100.0 * r_thin, E_k=photon_ev)
        # Same upper level => second line has E_i = E_k - hc/lambda > 0
        line2 = _obs(500.0, 100.0, E_k=photon_ev)
        result = ObservableSelfAbsorptionCorrector().correct([line1, line2])
        assert result.n_suspect == 0
        assert result.corrections[400.0].method == "doublet-thin"

    def test_zero_uncertainty_suspect_still_down_weighted(self):
        photon_ev = 1239.841984 / 400.0
        suspect = LineObservation(400.0, 100.0, 0.0, "Fe", 1, photon_ev, 4, 1e8)
        result = ObservableSelfAbsorptionCorrector().correct([suspect])
        out = result.observations[0]
        # sigma_y = k -> weight 1/k^2 relative to the unit weight sigma=0 maps to
        assert out.intensity_uncertainty == pytest.approx(100.0 * 3.0)


# ---------------------------------------------------------------------------
# mode normalization
# ---------------------------------------------------------------------------


class TestModeNormalization:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (False, "off"),
            (None, "off"),
            ("off", "off"),
            (True, "observable"),
            ("observable", "observable"),
            ("OBSERVABLE", "observable"),
        ],
    )
    def test_valid_values(self, value, expected):
        assert normalize_self_absorption_mode(value) == expected

    @pytest.mark.parametrize("bad", ["feedback", "on", 2, 1.5])
    def test_invalid_values_raise(self, bad):
        with pytest.raises(ValueError, match="apply_self_absorption"):
            normalize_self_absorption_mode(bad)
