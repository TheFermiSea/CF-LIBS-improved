"""
Tests for LTE validity checker.

Reference values computed from McWhirter criterion:
  n_e_min = 1.6e12 * sqrt(T_K) * delta_E_eV^3
"""

import pytest
from cflibs.plasma.lte_validator import LTEValidator, MCWHIRTER_C


class TestMcWhirterCheck:
    """Tests for the McWhirter criterion implementation."""

    def test_canonical_libs_conditions_pass(self):
        """Typical LIBS plasma should satisfy McWhirter."""
        # T=10000 K, n_e=1e17, delta_E=2 eV: n_e_min ~ 1.6e12 * 100 * 8 = 1.28e15
        result = LTEValidator.check_mcwhirter(T_K=10000.0, n_e_cm3=1e17, delta_E_eV=2.0)
        assert result.satisfied
        assert result.criterion == "mcwhirter"
        assert result.ratio > 1.0
        assert result.n_e_required == pytest.approx(
            MCWHIRTER_C * (10000.0**0.5) * (2.0**3), rel=1e-6
        )

    def test_low_density_fails(self):
        """Electron density below threshold should fail."""
        # At T=5000 K, delta_E=3 eV: n_e_min = 1.6e12 * sqrt(5000) * 27 ~ 3.05e15
        result = LTEValidator.check_mcwhirter(T_K=5000.0, n_e_cm3=1e14, delta_E_eV=3.0)
        assert not result.satisfied
        assert result.ratio < 1.0

    def test_ratio_matches_formula(self):
        """Ratio should equal n_e_actual / n_e_required."""
        T_K = 8000.0
        n_e = 5e16
        delta_E = 1.5
        result = LTEValidator.check_mcwhirter(T_K, n_e, delta_E)
        expected_required = MCWHIRTER_C * (T_K**0.5) * (delta_E**3)
        assert result.n_e_required == pytest.approx(expected_required, rel=1e-6)
        assert result.ratio == pytest.approx(n_e / expected_required, rel=1e-6)

    def test_high_delta_e_raises_threshold(self):
        """Large energy gaps require higher n_e (cubic dependence)."""
        low = LTEValidator.check_mcwhirter(T_K=10000.0, n_e_cm3=1e17, delta_E_eV=2.0)
        high = LTEValidator.check_mcwhirter(T_K=10000.0, n_e_cm3=1e17, delta_E_eV=4.0)
        # delta_E increases 2x -> threshold increases 8x
        assert high.n_e_required == pytest.approx(8 * low.n_e_required, rel=1e-4)


class TestTemporalCheck:
    """Tests for the temporal equilibration check."""

    def test_dense_plasma_passes(self):
        """Dense plasma at LIBS conditions equilibrates much faster than plasma lifetime."""
        # T=10000 K (0.86 eV), n_e=1e17, lifetime=1000 ns
        # tau_ee ~ 3.44e13 * 0.86^1.5 / 1e17 ~ 2.74e-4 ns << 0.1 * 1000 = 100 ns
        result = LTEValidator.check_temporal(T_K=10000.0, n_e_cm3=1e17, plasma_lifetime_ns=1000.0)
        assert result.satisfied
        assert result.criterion == "temporal"

    def test_tenuous_hot_plasma_fails(self):
        """Very hot, dilute plasma: long equilibration time relative to short lifetime."""
        # T=1e6 K (86 eV), n_e=1e5 cm^-3, lifetime=100 ns
        # tau_ee ~ 3.44e13 * 86^1.5 / 1e5 ~ 2.75e11 ns >> 0.1 * 100 = 10 ns
        result = LTEValidator.check_temporal(T_K=1e6, n_e_cm3=1e5, plasma_lifetime_ns=100.0)
        assert not result.satisfied


class TestValidate:
    """Tests for the full validate() method."""

    def test_returns_report_with_metrics(self):
        """validate() should return LTEReport with quality_metrics dict."""
        validator = LTEValidator()
        report = validator.validate(T_K=10000.0, n_e_cm3=1e17, delta_E_eV=2.0)

        assert hasattr(report, "mcwhirter")
        assert "lte_mcwhirter_satisfied" in report.quality_metrics
        assert "lte_n_e_ratio" in report.quality_metrics
        assert "lte_n_e_required_cm3" in report.quality_metrics

    def test_overall_satisfied_when_mcwhirter_passes(self):
        validator = LTEValidator()
        report = validator.validate(T_K=10000.0, n_e_cm3=1e17, delta_E_eV=2.0)
        assert report.mcwhirter.satisfied
        assert report.overall_satisfied

    def test_overall_not_satisfied_when_mcwhirter_fails(self):
        validator = LTEValidator()
        report = validator.validate(T_K=10000.0, n_e_cm3=1e10, delta_E_eV=2.0)
        assert not report.mcwhirter.satisfied
        assert not report.overall_satisfied
        assert len(report.warnings) > 0

    def test_delta_e_extracted_from_observations(self):
        """When delta_E_eV is not given, uses max adjacent gap (not total span)."""
        from unittest.mock import MagicMock

        # Energies [1, 3, 5] eV: adjacent gaps are [2, 2] -> max adjacent gap = 2.0 eV
        # (NOT the total span 5-1=4 eV)
        obs = [MagicMock(E_k_ev=1.0), MagicMock(E_k_ev=3.0), MagicMock(E_k_ev=5.0)]
        validator = LTEValidator()
        report = validator.validate(T_K=10000.0, n_e_cm3=1e17, observations=obs)

        expected_required = MCWHIRTER_C * (10000.0**0.5) * (2.0**3)
        assert report.mcwhirter.n_e_required == pytest.approx(expected_required, rel=1e-4)

    def test_with_temporal_check(self):
        """Temporal check should appear in report when requested."""
        validator = LTEValidator()
        report = validator.validate(
            T_K=10000.0,
            n_e_cm3=1e17,
            delta_E_eV=2.0,
            check_temporal=True,
            plasma_lifetime_ns=1000.0,
        )
        assert report.temporal is not None
        assert "lte_temporal_satisfied" in report.quality_metrics

    def test_no_observations_uses_default_delta_e(self):
        """With neither delta_E_eV nor observations, a default of 2.0 eV is used."""
        validator = LTEValidator()
        # Should not raise
        report = validator.validate(T_K=10000.0, n_e_cm3=1e17)
        expected_required = MCWHIRTER_C * (10000.0**0.5) * (2.0**3)
        assert report.mcwhirter.n_e_required == pytest.approx(expected_required, rel=1e-4)
