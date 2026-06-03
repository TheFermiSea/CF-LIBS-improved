"""
Tests for Stark broadening calculations.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from cflibs.radiation.stark import (
    stark_hwhm,
    stark_width,
    stark_shift,
    estimate_stark_parameter,
    StarkBroadeningCalculator,
    stark_hwhm_jax,
)
from cflibs.core.constants import EV_TO_K


class TestStarkCalculation:
    def test_stark_at_reference_conditions_returns_stored_fwhm(self):
        """At the reference conditions (REF_NE=1e17, T=10000 K) the FWHM must
        equal the stored ``stark_w`` exactly — the stored value IS the FWHM at
        those conditions (single-source-of-truth convention, A4-CONV-2)."""
        from cflibs.radiation.stark import REF_NE, REF_T_K

        w_ref = 0.1  # nm — stored FWHM at REF_NE, T=REF_T_K
        fwhm = stark_width(REF_NE, REF_T_K, w_ref, stark_alpha=0.5)
        assert np.isclose(fwhm, w_ref)
        # HWHM is exactly half the FWHM.
        hwhm = stark_hwhm(REF_NE, REF_T_K, w_ref, stark_alpha=0.5)
        assert np.isclose(hwhm, 0.5 * w_ref)

    def test_stark_scaling(self):
        """Test Stark width scaling with density and temperature.

        ``stark_w`` is the stored electron-impact FWHM at REF_NE=1e17 cm^-3,
        T=10000 K. ``stark_hwhm`` returns HALF of that, scaled by
        ``(n_e/1e17) * (T/10000)^(-alpha)``; ``stark_width`` doubles back to
        the FWHM.
        """
        w_ref = 0.1  # nm (stored FWHM at REF_NE=1e17)
        n_e = 2.0e16
        T = 40000.0

        # hwhm = 0.5 * 0.1 * (2e16/1e17) * (40000/10000)^(-0.5)
        #      = 0.5 * 0.1 * 0.2 * 0.5 = 0.005
        hwhm = stark_hwhm(n_e, T, w_ref, stark_alpha=0.5)
        assert np.isclose(hwhm, 0.005)

        # FWHM = 2 * HWHM = 0.01
        fwhm = stark_width(n_e, T, w_ref, stark_alpha=0.5)
        assert np.isclose(fwhm, 0.01)

    def test_stark_shift(self):
        """Test Stark shift scaling (stored d_ref is at REF_NE=1e17)."""
        d_ref = 0.05
        n_e = 0.5e17

        shift = stark_shift(n_e, d_ref)
        # 0.05 * (0.5e17/1e17) = 0.05 * 0.5 = 0.025
        assert np.isclose(shift, 0.025)

    def test_estimation(self):
        """Test fallback estimation."""
        # Visible line, neutral
        wl = 500.0
        E_up = 10.0
        IP = 13.6
        stage = 1

        w_est = estimate_stark_parameter(wl, E_up, IP, stage)
        assert 0.0001 <= w_est <= 0.5

        # Check scaling
        # Lower binding energy (closer to continuum) -> larger width
        w_est_high = estimate_stark_parameter(wl, 13.0, IP, stage)  # E_up=13
        w_est_low = estimate_stark_parameter(wl, 5.0, IP, stage)  # E_up=5
        assert w_est_high > w_est_low

    def test_calculator(self):
        """Test Calculator with mock DB."""
        db = MagicMock()
        # Mock get_stark_parameters returning (w, alpha, d)
        db.get_stark_parameters.return_value = (0.01, 0.5, 0.002)

        calc = StarkBroadeningCalculator(db)

        fwhm = calc.get_stark_width("Fe", 1, 500.0, n_e_cm3=1e16, T_e_eV=1.0)
        # stark_w=0.01 is the stored FWHM at REF_NE=1e17. At n_e=1e16:
        #   FWHM = 0.01 * (1e16/1e17) * (11604/10000)^-0.5 ~ 0.01 * 0.1 * 0.928
        # i.e. small but strictly positive.

        assert fwhm > 0.0
        db.get_stark_parameters.assert_called_once()

    def test_calculator_fallback(self):
        """Test Calculator fallback when DB returns None."""
        db = MagicMock()
        db.get_stark_parameters.return_value = (None, None, None)
        db.get_ionization_potential.return_value = 10.0

        calc = StarkBroadeningCalculator(db)

        fwhm = calc.get_stark_width("Fe", 1, 500.0, n_e_cm3=1e16, T_e_eV=1.0, upper_energy_ev=8.0)
        # Should fallback to estimate
        assert fwhm > 0.0

    @pytest.mark.requires_jax
    def test_jax_imports(self):
        """Test JAX implementation matches standard."""
        pytest.importorskip("jax")

        w_ref = 0.1
        n_e = 2.0e16
        T_eV = 1.0
        alpha = 0.5

        hwhm_jax = stark_hwhm_jax(n_e, T_eV, w_ref, alpha)

        # Compare with standard
        T_K = T_eV * EV_TO_K
        hwhm_std = stark_hwhm(n_e, T_K, w_ref, alpha)

        # Note: JAX uses REF_T_EV = 0.86173 vs T_K=10000
        # 10000 * KB_EV = 0.86173...
        assert np.isclose(float(hwhm_jax), hwhm_std, rtol=1e-3)


# Canonical convention anchors. The stored ``lines.stark_w`` is the published
# electron-impact FWHM at the reference conditions n_e=1e17 cm^-3, T=10000 K
# (Konjević et al. 2002; see scripts/ingest_stark_b.py and
# tests/test_stark_provenance.py). At those exact conditions the runtime
# omega_stark (FWHM) must therefore equal the stored value EXACTLY — not the
# x20-inflated value the pre-A4-CONV-2 runtime produced
# (REF_NE=1e16 plus an extra HWHM->FWHM doubling: x10 * x2 = x20).
#
# (stored_fwhm_pm, stark_alpha, old_buggy_x20_pm)
_CONVENTION_ANCHORS = [
    ("Al I 396.15", 4.5, 0.04, 90.0),
    ("Ca II 393.37", 8.4, 0.05, 168.0),
    ("Fe II 430.317", 33.0, 0.08, 660.0),
]


@pytest.mark.parametrize("name,stored_pm,alpha,old_buggy_pm", _CONVENTION_ANCHORS)
def test_stark_convention_anchor_no_20x_inflation(name, stored_pm, alpha, old_buggy_pm):
    """omega_stark at the reference (n_e=1e17, T=10000 K) equals the stored
    FWHM, killing the x20 density/HWHM convention mismatch (A4-CONV-2)."""
    n_e = 1.0e17
    T_K = 10000.0
    stored_nm = stored_pm * 1e-3

    omega_stark_pm = stark_width(n_e, T_K, stored_nm, stark_alpha=alpha) * 1e3

    # The convention-correct omega_stark IS the stored FWHM at reference.
    assert np.isclose(omega_stark_pm, stored_pm, rtol=1e-6), (
        f"{name}: omega_stark={omega_stark_pm:.3f} pm != stored {stored_pm} pm "
        "— Stark width convention regressed."
    )
    # And it must be the x20-corrected value, far below the historical bug.
    assert np.isclose(omega_stark_pm, old_buggy_pm / 20.0, rtol=1e-6), (
        f"{name}: expected ~{old_buggy_pm/20.0:.2f} pm (old buggy {old_buggy_pm} pm / 20), "
        f"got {omega_stark_pm:.3f} pm."
    )
