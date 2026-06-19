"""
Tests for the opt-in Cristoforetti relaxation-time LTE criterion.

The relaxation criterion (Cristoforetti et al. 2010, Spectrochim. Acta B 65,
86-95, Eq. 4; after Griem 1964) is a temporal criterion *beyond* the
McWhirter density floor: the time to reach excitation/ionization equilibrium,

    tau_rel ~ (6.3e4 / (n_e <g> f_nm)) * dE_nm * sqrt(kT) * exp(dE_nm / kT)

(n_e in cm^-3, dE_nm and kT in eV, tau_rel in seconds), must be much shorter
than the plasma evolution timescale. McWhirter is necessary but not
sufficient; a low-n_e, fast-evolving plasma can clear the steady-state
McWhirter floor yet fail this transient criterion.

These tests assert the physical acceptance properties:
  1. A low-n_e, fast-evolving case that PASSES McWhirter is FLAGGED as
     LTE-questionable by the relaxation criterion.
  2. A dense, steady case passes BOTH criteria.
And that the criterion is strictly opt-in (default behaviour unchanged).
"""

import pytest

from cflibs.core.constants import KB_EV
from cflibs.plasma.lte_validator import LTEValidator


class TestRelaxationFormula:
    """The bare relaxation-time formula matches the cited literature value."""

    def test_matches_carbon_calibration_point(self):
        """Reproduce the worked LIBS example used to verify Eq. 4.

        Colliding-carbon plasma: dE = 2.9 eV, f12 = 0.914,
        n_e = 3e16 cm^-3, kT = 1.45 eV, <g> ~ 1  ->  tau_rel ~ 6e-11 s.
        """
        kT_eV = 1.45
        T_K = kT_eV / KB_EV
        result = LTEValidator.check_relaxation(
            T_K=T_K,
            n_e_cm3=3e16,
            delta_E_eV=2.9,
            oscillator_strength=0.914,
            gaunt_factor=1.0,
            plasma_lifetime_ns=1000.0,
        )
        tau_rel_s = result.n_e_actual * 1e-9  # n_e_actual stores tau_rel in ns
        assert tau_rel_s == pytest.approx(6e-11, rel=0.05)
        assert result.criterion == "relaxation"

    def test_tau_rel_grows_as_density_drops(self):
        """tau_rel ~ 1/n_e: rarefying the plasma slows equilibration."""
        dense = LTEValidator.check_relaxation(8000.0, 1e17, 2.0)
        dilute = LTEValidator.check_relaxation(8000.0, 1e15, 2.0)
        assert dilute.n_e_actual == pytest.approx(100.0 * dense.n_e_actual, rel=1e-6)

    def test_tau_rel_grows_with_energy_gap(self):
        """Larger gaps thermalise more slowly (dE * exp(dE/kT) factor)."""
        small_gap = LTEValidator.check_relaxation(8000.0, 1e16, 1.0)
        large_gap = LTEValidator.check_relaxation(8000.0, 1e16, 3.0)
        assert large_gap.n_e_actual > small_gap.n_e_actual


class TestAcceptanceCriteria:
    """The two acceptance scenarios from the specification."""

    def test_low_ne_fast_evolving_passes_mcwhirter_but_flagged_by_relaxation(self):
        """ACCEPTANCE 1: McWhirter PASS, relaxation FLAGS LTE-questionable.

        Cool (T=6000 K), low-density (n_e=5e15 cm^-3), fast-evolving
        (tau_evol=100 ns) plasma with a dE=3.0 eV resonance gap. n_e sits
        above the McWhirter floor (necessary condition met), but the
        relaxation time (~18 ns) is not << tau_evol/10 = 10 ns, so the plasma
        has not had time to reach equilibrium -> transient non-LTE.
        """
        T_K = 6000.0
        n_e_cm3 = 5.0e15
        delta_E_eV = 3.0
        lifetime_ns = 100.0

        mcwhirter = LTEValidator.check_mcwhirter(T_K, n_e_cm3, delta_E_eV)
        relaxation = LTEValidator.check_relaxation(
            T_K, n_e_cm3, delta_E_eV, plasma_lifetime_ns=lifetime_ns
        )

        # McWhirter (necessary, steady-state) is satisfied ...
        assert mcwhirter.satisfied, "case must clear the McWhirter floor"
        # ... but the relaxation (transient) criterion flags it.
        assert not relaxation.satisfied, "relaxation must flag the fast-evolving case"
        # tau_rel exceeds the tolerable tau_evol/margin.
        assert relaxation.n_e_actual > relaxation.n_e_required

    def test_dense_steady_plasma_passes_both(self):
        """ACCEPTANCE 2: dense steady plasma passes McWhirter AND relaxation.

        Canonical LIBS conditions (T=10000 K, n_e=1e17 cm^-3) observed at a
        long gate (tau_evol=1000 ns): tau_rel is sub-ns, far below tau_evol/10.
        """
        T_K = 10000.0
        n_e_cm3 = 1.0e17
        delta_E_eV = 2.0
        lifetime_ns = 1000.0

        mcwhirter = LTEValidator.check_mcwhirter(T_K, n_e_cm3, delta_E_eV)
        relaxation = LTEValidator.check_relaxation(
            T_K, n_e_cm3, delta_E_eV, plasma_lifetime_ns=lifetime_ns
        )

        assert mcwhirter.satisfied
        assert relaxation.satisfied
        assert relaxation.n_e_actual < relaxation.n_e_required


class TestOptInWiring:
    """The criterion is wired into validate() but strictly opt-in."""

    def test_default_validate_does_not_run_relaxation(self):
        """Default validate() leaves relaxation unrun (default unchanged)."""
        validator = LTEValidator()
        report = validator.validate(T_K=10000.0, n_e_cm3=1e17, delta_E_eV=2.0)
        assert report.relaxation is None
        assert "lte_relaxation_satisfied" not in report.quality_metrics

    def test_opt_in_relaxation_appears_in_report(self):
        """With check_relaxation=True the result and metrics are surfaced."""
        validator = LTEValidator()
        report = validator.validate(
            T_K=10000.0,
            n_e_cm3=1e17,
            delta_E_eV=2.0,
            check_relaxation=True,
            plasma_lifetime_ns=1000.0,
        )
        assert report.relaxation is not None
        assert report.relaxation.satisfied
        assert "lte_relaxation_satisfied" in report.quality_metrics
        assert "lte_relaxation_tau_ns" in report.quality_metrics

    def test_opt_in_failed_relaxation_flips_overall_and_warns(self):
        """A flagged relaxation check fails overall_satisfied and warns,
        EVEN when McWhirter passes (the whole point of going beyond McWhirter).
        """
        validator = LTEValidator()
        report = validator.validate(
            T_K=6000.0,
            n_e_cm3=5.0e15,
            delta_E_eV=3.0,
            check_relaxation=True,
            plasma_lifetime_ns=100.0,
        )
        assert report.mcwhirter.satisfied
        assert report.relaxation is not None and not report.relaxation.satisfied
        assert not report.overall_satisfied
        assert any("Relaxation" in w for w in report.warnings)

    def test_overall_satisfied_unaffected_when_not_opted_in(self):
        """The flagged case still reports overall_satisfied=True by DEFAULT,
        because the relaxation check is not run unless opted in.
        """
        validator = LTEValidator()
        report = validator.validate(T_K=6000.0, n_e_cm3=5.0e15, delta_E_eV=3.0)
        assert report.mcwhirter.satisfied
        assert report.relaxation is None
        assert report.overall_satisfied
