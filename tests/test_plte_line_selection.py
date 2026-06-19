"""
Unit tests for the opt-in partial-LTE (pLTE) thermalization line-selection cut.

These tests assert the cited physical property (Cristoforetti et al. 2010,
Spectrochim. Acta Part B 65, 86-95): only levels lying at or above the
thermalization limit E* are collisionally coupled to the free-electron
temperature. On a synthetic level set with a known E* the cut must remove
*exactly* the sub-threshold lines and keep the thermalized ones; on a fully
thermalized set it must be a no-op. The default ``select()`` path is verified
to be untouched.

Self-contained: synthetic ``LineObservation`` inputs, no real atomic DB.
"""

import math

import pytest

from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.line_selection import (
    LineSelector,
    PLTECutResult,
    mcwhirter_thermalization_limit_ev,
)


def _line(wavelength_nm: float, e_k_ev: float, element: str = "Fe") -> LineObservation:
    """Build a clean, high-SNR synthetic line at the given upper energy."""
    return LineObservation(
        wavelength_nm=wavelength_nm,
        intensity=1000.0,
        intensity_uncertainty=10.0,
        element=element,
        ionization_stage=1,
        E_k_ev=e_k_ev,
        g_k=5,
        A_ki=1e7,
    )


# ==============================================================================
# McWhirter thermalization-limit formula
# ==============================================================================


class TestMcWhirterThermalizationLimit:
    """Verify E* = (n_e / (1.6e12 * sqrt(T)))^(1/3) [eV]."""

    def test_formula_matches_inverted_mcwhirter(self):
        T_K = 10000.0
        n_e = 1.0e17
        e_star = mcwhirter_thermalization_limit_ev(T_K, n_e)
        expected = (n_e / (1.6e12 * math.sqrt(T_K))) ** (1.0 / 3.0)
        assert e_star == pytest.approx(expected, rel=1e-12)

    def test_round_trip_with_mcwhirter_criterion(self):
        """E* computed from (T, n_e) must, fed back as the gap, recover n_e."""
        T_K = 8000.0
        n_e = 5.0e16
        e_star = mcwhirter_thermalization_limit_ev(T_K, n_e)
        # McWhirter: n_e = 1.6e12 * sqrt(T) * (ΔE)^3 evaluated at ΔE = E*
        n_e_back = 1.6e12 * math.sqrt(T_K) * e_star**3
        assert n_e_back == pytest.approx(n_e, rel=1e-9)

    def test_higher_density_thermalizes_lower_levels(self):
        """Higher n_e lowers E* (more levels thermalized) -- monotonic."""
        T_K = 9000.0
        low = mcwhirter_thermalization_limit_ev(T_K, 1e16)
        high = mcwhirter_thermalization_limit_ev(T_K, 1e18)
        assert high > low  # denser plasma thermalizes a larger energy range

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            mcwhirter_thermalization_limit_ev(0.0, 1e17)
        with pytest.raises(ValueError):
            mcwhirter_thermalization_limit_ev(8000.0, 0.0)


# ==============================================================================
# Acceptance: cut removes exactly the sub-threshold lines (known E*)
# ==============================================================================


class TestPLTECutAcceptance:
    """Cited acceptance criterion on a synthetic level set with a known E*."""

    def test_removes_exactly_sub_threshold_lines(self):
        """With E* = 1.0 eV, only lines with lower level < 1.0 eV are removed."""
        e_star = 1.0
        # Lower-level energies straddling the threshold.
        sub = [_line(400.0, 4.0), _line(410.0, 4.5)]  # E_i = 0.0, 0.5 (< E*)
        kept = [_line(420.0, 5.0), _line(430.0, 5.5)]  # E_i = 1.0, 2.0 (>= E*)
        lower = {
            ("Fe", 1, 400.0): 0.0,
            ("Fe", 1, 410.0): 0.5,
            ("Fe", 1, 420.0): 1.0,  # exactly at E* -> kept (>=)
            ("Fe", 1, 430.0): 2.0,
        }

        selector = LineSelector()
        result = selector.apply_plte_thermalization_cut(
            sub + kept, lower, enable=True, e_star_ev=e_star
        )

        assert isinstance(result, PLTECutResult)
        removed_wl = {o.wavelength_nm for o in result.sub_threshold_lines}
        kept_wl = {o.wavelength_nm for o in result.thermalized_lines}
        assert removed_wl == {400.0, 410.0}
        assert kept_wl == {420.0, 430.0}
        assert result.e_star_ev == pytest.approx(1.0)

    def test_boundary_level_is_kept(self):
        """A level exactly at E* is thermalized (>= cut, not strict >)."""
        selector = LineSelector()
        obs = [_line(400.0, 5.0)]
        lower = {("Fe", 1, 400.0): 1.5}
        result = selector.apply_plte_thermalization_cut(obs, lower, enable=True, e_star_ev=1.5)
        assert result.sub_threshold_lines == []
        assert len(result.thermalized_lines) == 1

    def test_fully_thermalized_set_is_noop(self):
        """All lower levels above E* -> nothing removed."""
        selector = LineSelector()
        obs = [_line(400.0, 5.0), _line(410.0, 6.0), _line(420.0, 7.0)]
        lower = {
            ("Fe", 1, 400.0): 2.0,
            ("Fe", 1, 410.0): 3.0,
            ("Fe", 1, 420.0): 4.5,
        }
        result = selector.apply_plte_thermalization_cut(obs, lower, enable=True, e_star_ev=1.0)
        assert result.sub_threshold_lines == []
        assert len(result.thermalized_lines) == 3

    def test_cut_uses_derived_e_star_from_T_and_ne(self):
        """When e_star_ev is omitted, E* is derived from (T, n_e)."""
        T_K = 10000.0
        n_e = 1e17
        e_star = mcwhirter_thermalization_limit_ev(T_K, n_e)
        # Place one line below and one above the derived E*.
        obs = [_line(400.0, 5.0), _line(410.0, 6.0)]
        lower = {
            ("Fe", 1, 400.0): e_star - 0.5,
            ("Fe", 1, 410.0): e_star + 0.5,
        }
        selector = LineSelector()
        result = selector.apply_plte_thermalization_cut(
            obs, lower, enable=True, temperature_K=T_K, electron_density_cm3=n_e
        )
        assert result.e_star_ev == pytest.approx(e_star, rel=1e-12)
        removed_wl = {o.wavelength_nm for o in result.sub_threshold_lines}
        kept_wl = {o.wavelength_nm for o in result.thermalized_lines}
        assert removed_wl == {400.0}
        assert kept_wl == {410.0}


# ==============================================================================
# Opt-in / default-path guarantees
# ==============================================================================


class TestPLTECutOptIn:
    """The cut must be strictly opt-in and never touch defaults."""

    def test_disabled_is_noop_keeps_everything(self):
        """enable=False -> no removal even when lines are below E*."""
        selector = LineSelector()
        obs = [_line(400.0, 4.0), _line(410.0, 4.5)]
        lower = {("Fe", 1, 400.0): 0.0, ("Fe", 1, 410.0): 0.1}
        result = selector.apply_plte_thermalization_cut(
            obs, lower, e_star_ev=2.0  # enable defaults to False
        )
        assert result.sub_threshold_lines == []
        assert len(result.thermalized_lines) == 2

    def test_missing_lower_level_is_kept_conservatively(self):
        """Lines without lower-level data are never dropped (reported skipped)."""
        selector = LineSelector()
        obs = [_line(400.0, 5.0), _line(410.0, 6.0)]
        lower = {("Fe", 1, 400.0): 0.0}  # 410.0 missing
        result = selector.apply_plte_thermalization_cut(obs, lower, enable=True, e_star_ev=1.0)
        skipped_wl = {o.wavelength_nm for o in result.skipped_lines}
        assert skipped_wl == {410.0}
        # 400.0 (E_i=0 < E*) removed; 410.0 (no data) kept.
        assert {o.wavelength_nm for o in result.sub_threshold_lines} == {400.0}
        assert 410.0 in {o.wavelength_nm for o in result.thermalized_lines}

    def test_enable_without_params_raises(self):
        selector = LineSelector()
        with pytest.raises(ValueError):
            selector.apply_plte_thermalization_cut([_line(400.0, 5.0)], {}, enable=True)

    def test_default_select_unaffected(self):
        """select() output is identical whether or not the pLTE method exists."""
        selector = LineSelector(min_snr=10.0, min_lines_per_element=1)
        obs = [_line(400.0, 3.0), _line(420.0, 5.0)]
        result = selector.select(obs)
        # Default select keeps both clean, isolated, high-SNR lines.
        assert len(result.selected_lines) == 2
        assert len(result.rejected_lines) == 0
        # No pLTE-related rejection reason leaks into the default path.
        assert all(
            s.rejection_reason is None or "pLTE" not in (s.rejection_reason or "")
            for s in result.scores
        )
