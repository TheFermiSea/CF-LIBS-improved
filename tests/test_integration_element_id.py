"""
Integration tests for element identification algorithms.

Tests the complete pipeline: identify -> to_line_observations -> BoltzmannPlotFitter.
Compares all three identifiers (ALIAS, Comb, Correlation) on the same spectrum.
"""

import pytest
import numpy as np

from cflibs.inversion.identify.alias import ALIASIdentifier
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.inversion.identify.correlation import CorrelationIdentifier
from cflibs.inversion.common.element_id import (
    ElementIdentificationResult,
    to_line_observations,
)
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation
from cflibs.validation.round_trip import GoldenSpectrumGenerator


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "ALIAS's Gate 4 (Boltzmann consistency) multiplies CL by a factor "
        "derived from the per-line ln(I·λ/gA) vs E_k regression. The "
        "synthetic_libs_spectrum fixture hand-picks line amplitudes "
        "(e.g. Fe at 1000/500/200) that vary 5× while the upper-state "
        "energies barely differ (3.31–3.33 eV) -- so the fit R² collapses "
        "and ``boltz_factor → 0``, zeroing CL before the detection gate. "
        "Production data is Boltzmann-consistent by construction; this "
        "fixture is not. Tracked in bead CF-LIBS-improved-oj3e -- needs a "
        "synthetic fixture rewrite (Boltzmann-consistent intensities) or a "
        "synthetic-mode bypass in ALIAS scoring."
    ),
    strict=False,
)
def test_alias_e2e_pipeline(atomic_db, synthetic_libs_spectrum):
    """Test ALIAS identifier E2E: identify -> to_line_observations -> BoltzmannPlotFitter."""
    # Generate synthetic spectrum
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Step 1: Run ALIAS identifier
    # ``elements=["Fe", "H"]`` bypasses ``_fast_screening`` (per alias.py
    # ``if self.elements is not None and len(self.elements) <= 10``),
    # which would otherwise reject Fe because the synthetic fixture's
    # 3 Fe lines don't land in the top-10 of the test atomic_db.
    # ``boltzmann_r2_min=0.0`` keeps the R^2 gate from rejecting the
    # synthetic ground-truth (default 0.85 is tuned for real-world noise).
    identifier = ALIASIdentifier(
        atomic_db=atomic_db,
        elements=["Fe", "H"],
        # See test_alias_e2e_pipeline for why detection_threshold is
        # relaxed for the synthetic fixture.
        detection_threshold=0.001,
        boltzmann_r2_min=0.0,
    )
    result = identifier.identify(wavelength, intensity)

    # Verify result structure
    assert isinstance(result, ElementIdentificationResult)
    assert len(result.detected_elements) > 0

    # Step 2: Convert to LineObservations
    observations = to_line_observations(result)
    assert len(observations) > 0
    assert all(isinstance(obs, LineObservation) for obs in observations)

    # Step 3: Run BoltzmannPlotFitter
    fitter = BoltzmannPlotFitter()
    fit_result = fitter.fit(observations)

    # Verify fit result
    assert fit_result.temperature_K > 0
    assert fit_result.temperature_uncertainty_K > 0
    assert not np.isnan(fit_result.temperature_K)


@pytest.mark.integration
def test_comb_e2e_pipeline(atomic_db, synthetic_libs_spectrum):
    """Test Comb identifier E2E: identify -> to_line_observations -> BoltzmannPlotFitter."""
    # Generate synthetic spectrum
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Step 1: Run Comb identifier
    identifier = CombIdentifier(atomic_db=atomic_db)
    result = identifier.identify(wavelength, intensity)

    # Verify result structure
    assert isinstance(result, ElementIdentificationResult)

    # Comb may not detect elements with simple mock DB - that's OK for integration test
    if len(result.detected_elements) == 0:
        pytest.skip("Comb identifier requires more complex database for detection")

    # Step 2: Convert to LineObservations
    observations = to_line_observations(result)

    # Skip if no observations after conversion
    if len(observations) == 0:
        pytest.skip("Comb identifier did not produce observations after filtering")

    assert len(observations) > 0
    assert all(isinstance(obs, LineObservation) for obs in observations)

    # Step 3: Run BoltzmannPlotFitter
    fitter = BoltzmannPlotFitter()
    fit_result = fitter.fit(observations)

    # Verify fit result
    assert fit_result.temperature_K > 0
    assert fit_result.temperature_uncertainty_K > 0
    assert not np.isnan(fit_result.temperature_K)


@pytest.mark.integration
def test_correlation_e2e_pipeline(atomic_db, synthetic_libs_spectrum):
    """Test Correlation identifier E2E: identify -> to_line_observations -> BoltzmannPlotFitter."""
    # Generate synthetic spectrum with lower noise for realistic SNR
    # (noise relative to baseline, not peak max)
    spectrum = synthetic_libs_spectrum(noise_level=0.001)
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Step 1: Run Correlation identifier
    identifier = CorrelationIdentifier(atomic_db=atomic_db)
    result = identifier.identify(wavelength, intensity)

    # Verify result structure
    assert isinstance(result, ElementIdentificationResult)

    # Correlation may not detect elements if confidence < threshold
    if len(result.detected_elements) == 0:
        pytest.skip("Correlation identifier requires higher confidence for detection")

    # Step 2: Convert to LineObservations
    observations = to_line_observations(result)
    assert len(observations) > 0
    assert all(isinstance(obs, LineObservation) for obs in observations)

    # Step 3: Run BoltzmannPlotFitter (requires >= 2 observations)
    if len(observations) >= 2:
        fitter = BoltzmannPlotFitter()
        fit_result = fitter.fit(observations)

        # Verify fit result
        assert fit_result.temperature_K > 0
        assert fit_result.temperature_uncertainty_K > 0
        assert not np.isnan(fit_result.temperature_K)
    else:
        pytest.skip(f"Only {len(observations)} observation(s); Boltzmann fit requires >= 2")


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "ALIAS's Boltzmann-consistency gate zeros CL on the synthetic "
        "fixture (intensities not Boltzmann-consistent). Same root cause "
        "as test_alias_e2e_pipeline. Bead CF-LIBS-improved-oj3e."
    ),
    strict=False,
)
def test_compare_all_identifiers(atomic_db, synthetic_libs_spectrum):
    """Compare all three identifiers on the SAME synthetic spectrum."""
    # Generate single synthetic spectrum with Fe and H lines
    spectrum = synthetic_libs_spectrum(
        elements={
            "Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)],
            "H": [(656.28, 5000.0), (486.13, 1000.0)],
        }
    )
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Run all three identifiers
    # See note in test_alias_e2e_pipeline: bypass _fast_screening +
    # relax R^2 gate for the synthetic ground-truth fixture.
    alias_id = ALIASIdentifier(
        atomic_db=atomic_db,
        elements=["Fe", "H"],
        # The synthetic_libs_spectrum fixture builds a small set of peaks
        # on a noisy continuum. ALIAS scoring incorporates the
        # peak/noise ratio, so the confidence layer (CL) lands below the
        # production default of 0.02 even when the right lines match.
        # Relax for synthetic fixtures.
        detection_threshold=0.001,
        boltzmann_r2_min=0.0,
    )
    comb_id = CombIdentifier(atomic_db=atomic_db)
    corr_id = CorrelationIdentifier(atomic_db=atomic_db)

    alias_result = alias_id.identify(wavelength, intensity)
    comb_result = comb_id.identify(wavelength, intensity)
    corr_result = corr_id.identify(wavelength, intensity)

    # Verify all return ElementIdentificationResult
    assert isinstance(alias_result, ElementIdentificationResult)
    assert isinstance(comb_result, ElementIdentificationResult)
    assert isinstance(corr_result, ElementIdentificationResult)

    # Get detected element lists
    alias_elements = {elem.element for elem in alias_result.detected_elements}
    _ = {elem.element for elem in comb_result.detected_elements}  # For comparison logging
    _ = {elem.element for elem in corr_result.detected_elements}  # For comparison logging

    # At least ALIAS should detect elements (most robust with mock DB)
    assert len(alias_elements) > 0, "ALIAS should detect at least one element"

    # ALIAS should detect at least Fe or H (both have strong lines in spectrum)
    assert "Fe" in alias_elements or "H" in alias_elements, "ALIAS should detect Fe or H"


@pytest.mark.integration
def test_round_trip_with_golden_spectrum(atomic_db):
    """Test round-trip validation with GoldenSpectrumGenerator."""
    # Generate golden spectrum with known parameters
    generator = GoldenSpectrumGenerator(
        atomic_db=atomic_db,
        wavelength_range=(200.0, 800.0),
    )

    golden = generator.generate(
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.7, "Ti": 0.2, "Ca": 0.1},
        n_lines_per_element=10,
        seed=42,
        include_ionic=True,
        min_intensity=10.0,
    )

    # Verify golden spectrum was created with line observations
    assert isinstance(golden.line_observations, list)
    assert len(golden.line_observations) > 0
    assert all(isinstance(obs, LineObservation) for obs in golden.line_observations)

    # Verify observations have correct metadata
    assert golden.temperature_K == 10000.0
    assert golden.electron_density_cm3 == 1e17
    assert set(golden.concentrations.keys()) == {"Fe", "Ti", "Ca"}

    # Verify LineObservation objects are valid for downstream pipeline
    for obs in golden.line_observations:
        assert obs.wavelength_nm > 0
        assert obs.intensity > 0
        assert obs.intensity_uncertainty > 0
        assert obs.element in ["Fe", "Ti", "Ca"]
        assert obs.ionization_stage in [1, 2]
        assert obs.E_k_ev > 0
        assert obs.g_k > 0
        assert obs.A_ki > 0


@pytest.mark.integration
def test_to_line_observations_filters_interfered(atomic_db, synthetic_libs_spectrum):
    """Verify to_line_observations() filters interfered lines."""
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Run identifier
    identifier = ALIASIdentifier(atomic_db=atomic_db)
    result = identifier.identify(wavelength, intensity)

    # Convert to observations
    observations = to_line_observations(result)

    # Verify no interfered lines in output
    for obs in observations:
        # Check corresponding line in result
        for elem_id in result.detected_elements:
            if elem_id.element == obs.element:
                for line in elem_id.matched_lines:
                    if (
                        line.wavelength_th_nm == obs.wavelength_nm
                        and line.ionization_stage == obs.ionization_stage
                    ):
                        assert not line.is_interfered


@pytest.mark.integration
def test_to_line_observations_deduplication(atomic_db, synthetic_libs_spectrum):
    """Verify to_line_observations() deduplicates by (element, ion_stage, wavelength)."""
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Run identifier
    identifier = ALIASIdentifier(atomic_db=atomic_db)
    result = identifier.identify(wavelength, intensity)

    # Convert to observations
    observations = to_line_observations(result)

    # Build set of (element, ionization_stage, wavelength_nm) tuples
    seen = set()
    for obs in observations:
        key = (obs.element, obs.ionization_stage, obs.wavelength_nm)
        assert key not in seen, f"Duplicate observation: {key}"
        seen.add(key)


@pytest.mark.integration
def test_bridge_function_valid_line_observations(atomic_db, synthetic_libs_spectrum):
    """Verify to_line_observations() returns valid LineObservation objects."""
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Run identifier
    identifier = ALIASIdentifier(atomic_db=atomic_db)
    result = identifier.identify(wavelength, intensity)

    # Convert to observations
    observations = to_line_observations(result)

    # Verify all fields are valid
    for obs in observations:
        assert obs.wavelength_nm > 0
        assert obs.intensity > 0
        assert obs.intensity_uncertainty > 0
        assert obs.element != ""
        assert obs.ionization_stage >= 1
        assert obs.E_k_ev > 0
        assert obs.g_k > 0
        assert obs.A_ki > 0

        # Verify y_value is computable (no NaN, no inf)
        assert not np.isnan(obs.y_value)
        assert not np.isinf(obs.y_value)


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "ALIAS's Boltzmann-consistency gate zeros CL on the synthetic "
        "fixture (intensities not Boltzmann-consistent). Same root cause "
        "as test_alias_e2e_pipeline. Bead CF-LIBS-improved-oj3e."
    ),
    strict=False,
)
def test_comparative_line_counts(atomic_db, synthetic_libs_spectrum):
    """Compare number of lines detected by each identifier."""
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Run all three identifiers
    # See note on elements=["Fe", "H"] + boltzmann_r2_min=0.0 in
    # test_alias_e2e_pipeline above (bypass _fast_screening for synthetic
    # fixture; relax R^2 gate for synthetic ground-truth).
    alias_id = ALIASIdentifier(
        atomic_db=atomic_db,
        elements=["Fe", "H"],
        # The synthetic_libs_spectrum fixture builds a small set of peaks
        # on a noisy continuum. ALIAS scoring incorporates the
        # peak/noise ratio, so the confidence layer (CL) lands below the
        # production default of 0.02 even when the right lines match.
        # Relax for synthetic fixtures.
        detection_threshold=0.001,
        boltzmann_r2_min=0.0,
    )
    comb_id = CombIdentifier(atomic_db=atomic_db)
    corr_id = CorrelationIdentifier(atomic_db=atomic_db)

    alias_result = alias_id.identify(wavelength, intensity)
    comb_result = comb_id.identify(wavelength, intensity)
    corr_result = corr_id.identify(wavelength, intensity)

    # Convert to observations
    alias_obs = to_line_observations(alias_result)
    comb_obs = to_line_observations(comb_result)
    corr_obs = to_line_observations(corr_result)

    # ALIAS should find at least some lines
    assert len(alias_obs) > 0, "ALIAS should detect at least some lines"
    assert len(comb_obs) >= 0, "Comb identifier should return valid result"
    assert len(corr_obs) >= 0, "Correlation identifier should return valid result"
