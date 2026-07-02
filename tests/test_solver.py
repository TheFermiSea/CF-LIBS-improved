"""
Tests for iterative solver.
"""

import pytest
from unittest.mock import MagicMock
import numpy as np
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3


@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    # Setup standard IP
    db.get_ionization_potential.return_value = 7.0  # eV

    # Setup simple partition coeffs (constant)
    # log(25) = 3.2188
    coeffs_I = [3.2188, 0, 0, 0, 0]
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs_I,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


def test_solver_basic(mock_db):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5)

    # Create synthetic data for T=10000K (0.86 eV)
    # Element A (Neutral)
    # y = -E/T + const
    T_eV = 0.8617

    obs = []
    # Neutral lines (Stage 1)
    for E in [1.0, 2.0, 3.0]:
        # y = -E/0.8617 + 10.0
        y = -E / T_eV + 10.0
        # Inverse: I = exp(y) * gA / lambda (simplification)
        # We set y_value directly via intensity manipulation
        # y = ln(I*lam/gA) => I = exp(y) if lam=gA=1
        intensity = np.exp(y)
        obs.append(
            LineObservation(
                wavelength_nm=500.0,
                intensity=intensity,
                intensity_uncertainty=0.1,
                element="A",
                ionization_stage=1,
                E_k_ev=E,
                g_k=1,
                A_ki=1e8,
            )
        )

    res = solver.solve(obs)

    assert res.converged
    assert abs(res.temperature_K - 10000.0) < 500.0
    assert "A" in res.concentrations
    assert res.concentrations["A"] == 1.0


def test_solver_multielement(mock_db):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)

    # Elements A and B, equal concentration
    T_eV = 1.0  # ~11600 K

    obs = []
    # Element A
    for E in [1, 3, 5]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500, np.exp(y), 0.1, "A", 1, E, 1, 1e8))

    # Element B
    for E in [1, 3, 5]:
        y = -E / T_eV + 10.0  # Same intercept -> same concentration if U is same
        obs.append(LineObservation(500, np.exp(y), 0.1, "B", 1, E, 1, 1e8))

    res = solver.solve(obs)

    assert abs(res.temperature_K - 11604.0) < 500.0
    assert abs(res.concentrations["A"] - 0.5) < 0.05
    assert abs(res.concentrations["B"] - 0.5) < 0.05


def test_clean_boltzmann_fit_is_not_gated(mock_db):
    """A clean (high-R^2, negative-slope) fit must update T normally and converge.

    Regression guard for the slope-R^2 conditioning gate: it must be a no-op on
    well-conditioned data so the round-trip/NIST-parity behaviour is preserved.
    """
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
    T_eV = 1.0  # ~11604 K
    obs = []
    for E in [1.0, 2.0, 3.0, 4.0, 5.0]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500.0, np.exp(y), 0.1, "A", 1, E, 1, 1e8))

    res = solver.solve(obs)
    assert res.converged is True
    assert res.quality_metrics["r_squared_last"] > 0.99
    assert abs(res.temperature_K - 11604.0) < 600.0


def test_degenerate_boltzmann_fit_holds_T_and_reports_unconverged(mock_db):
    """A flat / near-zero-slope Boltzmann plane must NOT run T to the 50000 K
    clamp and must NOT be reported as converged.

    This is the conditioning that prevents the closure-degeneracy collapse: at a
    runaway temperature exp(-E_k/kT) -> 1 for every line and the closure becomes
    a raw-intensity softmax that the brightest element wins. The gate holds T at
    the prior (initial 10000 K) and marks the solve non-converged so the garbage
    composition is flagged rather than trusted.
    """
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
    obs = []
    # Flat intensities (no E_k dependence) => zero/non-negative slope: an
    # unphysical Boltzmann plane the slope-sign gate must reject regardless of
    # R^2. Higher-E_k lines even *rise* slightly so the fitted slope is >= 0.
    for el in ("A", "B"):
        for E in (1.0, 2.0, 3.0, 4.0, 5.0):
            y = 10.0 + 0.05 * E  # populations rise with E_k: unphysical
            obs.append(LineObservation(500.0, np.exp(y), 0.1, el, 1, E, 1, 1e8))

    res = solver.solve(obs)
    assert res.converged is False
    # T held near the 10000 K prior, NOT the legacy 50000 K clamp.
    assert res.temperature_K < 30000.0


def test_min_boltzmann_r2_gate_is_configurable(mock_db):
    """Setting min_boltzmann_r2=0.0 with a negative slope disables the R^2 gate
    (slope-sign gate still applies)."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5, min_boltzmann_r2=0.0)
    assert solver.min_boltzmann_r2 == pytest.approx(0.0, abs=1e-12)
    T_eV = 1.0
    obs = []
    for E in [1.0, 2.0, 3.0]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500.0, np.exp(y), 0.1, "A", 1, E, 1, 1e8))
    res = solver.solve(obs)
    assert res.converged is True


def test_saha_correction_maps_mixed_stage_lines_to_common_boltzmann_plane(mock_db):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)

    T_eV = 1.0
    T_K = T_eV * EV_TO_K
    n_e = 1.0e17
    ip = 7.0
    wavelength_nm = 500.0

    # The classic Saha-Boltzmann transform should subtract only the ionic
    # prefactor from y while carrying IP on the x-axis.
    saha_offset = np.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5))
    common_intercept = 8.0

    observations = []

    for E_k in [1.0, 2.0, 3.0]:
        y = common_intercept - E_k / T_eV
        intensity = np.exp(y) / wavelength_nm
        observations.append(
            LineObservation(
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=max(intensity * 0.01, 1e-8),
                element="A",
                ionization_stage=1,
                E_k_ev=E_k,
                g_k=1,
                A_ki=1.0,
            )
        )

    for E_k in [4.0, 5.0, 6.0]:
        y = common_intercept + saha_offset - (ip + E_k) / T_eV
        intensity = np.exp(y) / wavelength_nm
        observations.append(
            LineObservation(
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=max(intensity * 0.01, 1e-8),
                element="A",
                ionization_stage=2,
                E_k_ev=E_k,
                g_k=1,
                A_ki=1.0,
            )
        )

    corrected = solver._apply_saha_correction({"A": observations}, T_K, n_e, {"A": ip})["A"]

    for raw, mapped in zip(observations, corrected):
        expected_y = common_intercept - mapped.E_k_ev / T_eV
        assert mapped.y_value == pytest.approx(expected_y, rel=1e-10, abs=1e-10)
        if raw.ionization_stage == 2:
            assert mapped.y_uncertainty == pytest.approx(raw.y_uncertainty, rel=1e-10)
        else:
            assert mapped.intensity == pytest.approx(raw.intensity, rel=1e-12)

    xs = np.array([obs.E_k_ev for obs in corrected])
    ys = np.array([obs.y_value for obs in corrected])
    slope, intercept = np.polyfit(xs, ys, deg=1)

    assert slope == pytest.approx(-1.0 / T_eV, rel=1e-10, abs=1e-10)
    assert intercept == pytest.approx(common_intercept, rel=1e-10, abs=1e-10)


def test_solver_quality_metrics_contain_lte(mock_db):
    """Verify solve() populates LTE quality metrics in quality_metrics."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5)
    T_eV = 0.8617

    obs = []
    for E in [1.0, 2.0, 3.0]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500.0, np.exp(y), 0.1, "A", 1, E, 1, 1e8))

    res = solver.solve(obs)
    assert "lte_mcwhirter_satisfied" in res.quality_metrics
    assert "lte_n_e_ratio" in res.quality_metrics


def test_solver_ipd_flag_runs_without_error(mock_db):
    """Verify apply_ipd=True completes solve without raising."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5, apply_ipd=True)
    T_eV = 0.8617

    obs = []
    for E in [1.0, 2.0, 3.0]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500.0, np.exp(y), 0.1, "A", 1, E, 1, 1e8))

    res = solver.solve(obs)
    assert res.converged


def test_solver_electron_density_pressure_balance(mock_db):
    """Verify n_e converges to a self-consistent value from pressure balance.

    Generate synthetic observations at the equilibrium n_e implied by
    P = n_tot * kT * (1 + Z_avg), then check the solver recovers that n_e.

    This test targets the *pressure-balance* n_e path by name, so the solver is
    constructed with ``prefer_sb_offset_ne=False``: on this branch the default
    precedence (Issue 4) measures n_e from the Saha-Boltzmann inter-stage
    intercept offset whenever an element exposes both a neutral and an ion
    stage — which these synthetic observations do — and that measured n_e would
    intercept before pressure balance ever runs. Disabling it forces the path
    under test.

    The reference iteration mirrors the solver's *three-stage* charge balance
    (Issue 5): free electrons per atom eps = (S1 + 2·S1·S2)/(1 + S1 + S1·S2),
    matching ``_pressure_balance_ne``. S2 is fetched exactly as
    ``_second_saha_ratio`` computes it from the mock DB — which returns a finite
    IP_II (7.0 eV) and U_III (=25) for every stage, so S2 is NON-zero here and
    the legacy two-stage form (avg_Z = S/(1+S)) no longer matches.
    """
    from cflibs.core.constants import KB, STP_PRESSURE

    T_eV = 1.0
    T_K = T_eV * EV_TO_K
    ip = 7.0

    # Compute self-consistent n_e at STP for a single element, mirroring the
    # solver's three-stage ionization ladder exactly.
    #   S1 = n_II/n_I, S2 = n_III/n_II  (both from the Saha equation)
    #   avg_Z = (S1 + 2·S1·S2) / (1 + S1 + S1·S2)
    #   n_tot = P/(kT·(1 + avg_Z)),  n_e = avg_Z · n_tot
    # The mock DB returns log(25) for U(I)=U(II)=U(III) and IP=7.0 eV for every
    # stage, so U_III/U_II = 1 and IP_II = 7.0 (see _second_saha_ratio).
    U_I = 25.0  # matches mock_db partition function
    U_II = 25.0
    U_III = 25.0  # mock_db returns log(25) for every stage
    ip_II = 7.0  # mock_db.get_ionization_potential returns 7.0 for every stage
    n_e_eq = 1e17  # initial guess
    for _ in range(100):
        S1 = (SAHA_CONST_CM3 / n_e_eq) * (T_eV**1.5) * (U_II / U_I) * np.exp(-ip / T_eV)
        S2 = (SAHA_CONST_CM3 / n_e_eq) * (T_eV**1.5) * (U_III / U_II) * np.exp(-ip_II / T_eV)
        ladder = 1.0 + S1 + S1 * S2
        avg_Z = (S1 + 2.0 * S1 * S2) / ladder
        n_tot = STP_PRESSURE / (KB * T_K * (1.0 + avg_Z)) * 1e-6  # cm^-3
        n_e_new = avg_Z * n_tot
        if abs(n_e_new - n_e_eq) / n_e_eq < 1e-6:
            break
        n_e_eq = 0.5 * n_e_eq + 0.5 * n_e_new

    # Generate synthetic data at this equilibrium n_e
    saha_offset = np.log((SAHA_CONST_CM3 / n_e_eq) * (T_eV**1.5))
    common_intercept = 8.0
    wavelength_nm = 500.0

    observations = []
    for E_k in [1.0, 2.0, 3.0, 4.0]:
        y = common_intercept - E_k / T_eV
        intensity = np.exp(y) / wavelength_nm
        observations.append(
            LineObservation(
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=max(intensity * 0.01, 1e-8),
                element="A",
                ionization_stage=1,
                E_k_ev=E_k,
                g_k=1,
                A_ki=1.0,
            )
        )

    for E_k in [2.0, 3.0, 4.0, 5.0]:
        y = common_intercept + saha_offset - (ip + E_k) / T_eV
        intensity = np.exp(y) / wavelength_nm
        observations.append(
            LineObservation(
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=max(intensity * 0.01, 1e-8),
                element="A",
                ionization_stage=2,
                E_k_ev=E_k,
                g_k=1,
                A_ki=1.0,
            )
        )

    solver = IterativeCFLIBSSolver(
        mock_db, max_iterations=20, pressure_pa=STP_PRESSURE, prefer_sb_offset_ne=False
    )
    res = solver.solve(observations)

    assert res.converged
    assert res.quality_metrics.get("ne_source") == "pressure_balance_imputed"
    assert res.electron_density_cm3 == pytest.approx(n_e_eq, rel=0.25)
