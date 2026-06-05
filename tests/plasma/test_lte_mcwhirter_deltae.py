"""Validation gate for the McWhirter delta_E energy gap (audit Family K).

The McWhirter LTE criterion

    n_e >= 1.6e12 * sqrt(T_K) * delta_E_eV^3        [cm^-3, eV]

requires delta_E to be the *largest energy gap in the atomic term scheme* —
conventionally the resonance (ground -> first-excited) transition, which is
~3-5 eV for typical LIBS species. See Cristoforetti, G. et al. (2010)
Spectrochim. Acta B 65, 86-95, which codifies the McWhirter floor.

The previous implementation used delta_E = the largest gap between *adjacent
sorted observed upper-level energies* of the fitted lines (floored at 0.1 eV).
Observed E_k cluster narrowly, so that delta_E was ~0.2 eV — orders of
magnitude too small. Because n_e_required scales as delta_E^3, this
under-estimated the required density by 1-2 orders of magnitude and let
non-LTE plasmas pass the gate.

These tests use an INDEPENDENT oracle: delta_E and n_e_required are computed
directly from the documented McWhirter formula and the term-scheme convention,
not from any helper in the code under test.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from cflibs.plasma.lte_validator import LTEValidator

# Independent reproduction of the McWhirter constant (cm^-3, ΔE in eV, T in K).
MCWHIRTER_CONST = 1.6e12


def _make_obs(e_k_values):
    """Build observation stand-ins carrying only upper-level energies (eV)."""
    return [MagicMock(E_k_ev=float(e)) for e in e_k_values]


class TestMcWhirterTermSchemeDeltaE:
    """delta_E must be the term-scheme span, not the adjacent observed gap."""

    # Clustered upper-level energies, all near 3 eV. The adjacent gaps are
    # only 0.2 eV, but the term-scheme span (max E_k) is 3.6 eV.
    CLUSTERED_EK = [3.0, 3.2, 3.4, 3.6]
    T_K = 1.0e4

    def test_delta_e_is_term_scheme_scale_not_adjacent_gap(self):
        """The delta_E used must be the term-scheme scale (>= ~3 eV)."""
        obs = _make_obs(self.CLUSTERED_EK)
        report = LTEValidator().validate(T_K=self.T_K, n_e_cm3=1e17, observations=obs)

        # Recover the delta_E actually used by inverting the McWhirter formula
        # with an independent constant: n_e_required = C * sqrt(T) * dE^3.
        delta_e_used = (report.mcwhirter.n_e_required / (MCWHIRTER_CONST * np.sqrt(self.T_K))) ** (
            1.0 / 3.0
        )

        # Term-scheme / resonance scale: must be >= ~3 eV, NOT the ~0.2 eV
        # adjacent-gap value the buggy implementation produced.
        assert delta_e_used >= 3.0
        # And not the 0.2 eV adjacent gap (with a wide margin).
        adjacent_gap = max(
            self.CLUSTERED_EK[i + 1] - self.CLUSTERED_EK[i]
            for i in range(len(self.CLUSTERED_EK) - 1)
        )
        assert delta_e_used > 10.0 * adjacent_gap

    def test_n_e_required_three_orders_larger_than_adjacent_gap_result(self):
        """n_e_required must be ~3 orders of magnitude above the buggy value."""
        obs = _make_obs(self.CLUSTERED_EK)
        report = LTEValidator().validate(T_K=self.T_K, n_e_cm3=1e17, observations=obs)

        adjacent_gap = 0.2  # eV, the gap the buggy code would have used
        buggy_required = MCWHIRTER_CONST * np.sqrt(self.T_K) * adjacent_gap**3
        term_scheme_required = MCWHIRTER_CONST * np.sqrt(self.T_K) * 3.6**3

        # The fix should match the term-scheme oracle, not the buggy one.
        assert report.mcwhirter.n_e_required == pytest.approx(term_scheme_required, rel=1e-4)
        # And that is ~3 orders of magnitude larger than the adjacent-gap value:
        # (3.6 / 0.2)^3 = 18^3 = 5832 ~ 3.8 decades.
        ratio = report.mcwhirter.n_e_required / buggy_required
        assert ratio > 1.0e3

    def test_plasma_that_previously_passed_now_fails(self):
        """A plasma at n_e=1e15 passed under the adjacent gap but must now fail.

        With the buggy adjacent gap (0.2 eV): required ~ 1.6e12 * 100 * 0.008
        = 1.28e9 cm^-3, so n_e=1e15 PASSES.
        With the term-scheme span (3.6 eV): required ~ 1.6e12 * 100 * 46.66
        = 7.46e15 cm^-3, so n_e=1e15 FAILS.
        """
        obs = _make_obs(self.CLUSTERED_EK)

        # Independent sanity check that the buggy gap would have passed.
        buggy_required = MCWHIRTER_CONST * np.sqrt(self.T_K) * 0.2**3
        assert 1e15 >= buggy_required  # buggy code: PASS

        report = LTEValidator().validate(T_K=self.T_K, n_e_cm3=1e15, observations=obs)
        assert not report.mcwhirter.satisfied  # fixed code: FAIL
        assert not report.overall_satisfied
        assert len(report.warnings) > 0

    def test_term_scheme_plasma_with_adequate_density_passes(self):
        """A genuinely LTE plasma (high n_e) still passes under the term gap."""
        obs = _make_obs(self.CLUSTERED_EK)
        # required ~ 7.46e15; n_e=1e17 comfortably exceeds it.
        report = LTEValidator().validate(T_K=self.T_K, n_e_cm3=1e17, observations=obs)
        assert report.mcwhirter.satisfied
        assert report.overall_satisfied
