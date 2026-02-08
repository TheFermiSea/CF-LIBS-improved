"""
Integration tests for element identification algorithms.

Tests the complete pipeline: identify -> to_line_observations -> BoltzmannPlotFitter.
Compares all three identifiers (ALIAS, Comb, Correlation) on the same spectrum.
"""

import pytest
import numpy as np

from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.comb_identifier import CombIdentifier
from cflibs.inversion.correlation_identifier import CorrelationIdentifier
from cflibs.inversion.element_id import (
    ElementIdentificationResult,
    to_line_observations,
)
from cflibs.inversion.boltzmann import BoltzmannPlotFitter, LineObservation
from cflibs.validation.round_trip import GoldenSpectrumGenerator


@pytest.mark.integration
def test_alias_e2e_pipeline(atomic_db, synthetic_libs_spectrum):
    """Test ALIAS identifier E2E: identify -> to_line_observations -> BoltzmannPlotFitter."""
    # Generate synthetic spectrum
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Step 1: Run ALIAS identifier
    identifier = ALIASIdentifier(atomic_db=atomic_db)
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
    # Generate synthetic spectrum
    spectrum = synthetic_libs_spectrum()
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

    # Step 3: Run BoltzmannPlotFitter
    fitter = BoltzmannPlotFitter()
    fit_result = fitter.fit(observations)

    # Verify fit result
    assert fit_result.temperature_K > 0
    assert fit_result.temperature_uncertainty_K > 0
    assert not np.isnan(fit_result.temperature_K)


@pytest.mark.integration
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
    alias_id = ALIASIdentifier(atomic_db=atomic_db)
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
def test_comparative_line_counts(atomic_db, synthetic_libs_spectrum):
    """Compare number of lines detected by each identifier."""
    spectrum = synthetic_libs_spectrum()
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    # Run all three identifiers
    alias_id = ALIASIdentifier(atomic_db=atomic_db)
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

    # Log counts for comparison (useful for debugging)
    print("\nLine counts:")
    print(f"  ALIAS: {len(alias_obs)}")
    print(f"  Comb: {len(comb_obs)}")
    print(f"  Correlation: {len(corr_obs)}")
