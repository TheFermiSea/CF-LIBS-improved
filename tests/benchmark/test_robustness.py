import numpy as np
import pytest

from cflibs.benchmark.dataset import BenchmarkSpectrum, InstrumentalConditions, SampleMetadata
from cflibs.benchmark.robustness import (
    line_dropout_perturbation,
    outlier_injection_perturbation,
    run_perturbation_battery,
)


@pytest.fixture
def dummy_spectrum():
    """
    Create a deterministic BenchmarkSpectrum used for tests.

    Returns:
        BenchmarkSpectrum: A spectrum with id "test_spec_1", wavelengths linearly spaced from 200 to 300 nm (100 points), intensities linearly increasing from 1 to 100 (so the top three intensities are the final three channels), true composition {"Fe": 0.8, "Cr": 0.2}, fixed InstrumentalConditions (1064 nm laser, 50.0 mJ, 5.0 ns pulse width, 10.0 Hz repetition, 100.0 µm spot), and SampleMetadata(sample_id="dummy_1").
    """
    return BenchmarkSpectrum(
        spectrum_id="test_spec_1",
        wavelength_nm=np.linspace(200, 300, 100),
        intensity=np.linspace(1, 100, 100),  # increasing intensity, top 3 are at the end
        true_composition={"Fe": 0.8, "Cr": 0.2},
        conditions=InstrumentalConditions(
            laser_wavelength_nm=1064,
            laser_energy_mj=50.0,
            laser_pulse_width_ns=5.0,
            repetition_rate_hz=10.0,
            spot_diameter_um=100.0,
        ),
        metadata=SampleMetadata(sample_id="dummy_1"),
    )


@pytest.mark.unit
def test_line_dropout_perturbation(dummy_spectrum):
    top_n = 3
    perturbed = line_dropout_perturbation(dummy_spectrum, top_n=top_n)

    assert np.allclose(perturbed.wavelength_nm, dummy_spectrum.wavelength_nm)

    # Check that top 3 intensities (98, 99, 100 at indices 97, 98, 99) are 0
    assert perturbed.intensity[-1] == pytest.approx(0.0)
    assert perturbed.intensity[-2] == pytest.approx(0.0)
    assert perturbed.intensity[-3] == pytest.approx(0.0)

    # Check other values are unmodified
    assert np.allclose(perturbed.intensity[:-3], dummy_spectrum.intensity[:-3])


@pytest.mark.unit
def test_outlier_injection_perturbation(dummy_spectrum):
    rng = np.random.default_rng(42)
    fraction = 0.05
    sigma_mult = 5.0

    perturbed = outlier_injection_perturbation(
        dummy_spectrum, fraction=fraction, sigma_mult=sigma_mult, rng=rng
    )

    n_channels = len(dummy_spectrum.intensity)
    n_inject = int(n_channels * fraction)

    # Find indices where intensity was changed
    diff = perturbed.intensity != dummy_spectrum.intensity
    changed_indices = np.nonzero(diff)[0]

    assert len(changed_indices) == n_inject

    # Check noise statistics roughly if n_inject was larger, but with 5 items we can just verify differences
    noise_added = perturbed.intensity[changed_indices] - dummy_spectrum.intensity[changed_indices]
    assert np.any(np.abs(noise_added) > 0.0)


@pytest.mark.unit
def test_outlier_injection_validates_inputs(dummy_spectrum):
    with pytest.raises(ValueError, match="fraction"):
        outlier_injection_perturbation(dummy_spectrum, fraction=1.5)

    with pytest.raises(ValueError, match="sigma_mult"):
        outlier_injection_perturbation(dummy_spectrum, sigma_mult=-1.0)


@pytest.mark.unit
def test_outlier_injection_handles_constant_integer_intensity(dummy_spectrum):
    dummy_spectrum.intensity = np.ones_like(dummy_spectrum.intensity, dtype=np.int64)

    perturbed = outlier_injection_perturbation(
        dummy_spectrum,
        fraction=0.2,
        sigma_mult=5.0,
        rng=np.random.default_rng(7),
    )

    assert np.issubdtype(perturbed.intensity.dtype, np.floating)
    assert np.any(perturbed.intensity != dummy_spectrum.intensity)
    assert np.all(perturbed.intensity >= 0.0)


@pytest.mark.unit
def test_run_perturbation_battery(dummy_spectrum):
    """
    Validates that run_perturbation_battery produces expected baseline, perturbed, and delta error metrics when a line-dropout perturbation is applied.

    Sets up a dummy pipeline that returns a "perfect" composition when the total intensity exceeds a threshold and a "bad" composition otherwise, applies a line dropout perturbation that reduces the total intensity, runs the perturbation battery on a single test spectrum, and asserts:
    - the returned report uses the expected pipeline name;
    - the baseline absolute difference for the test spectrum is 0.0 (perfect match);
    - the perturbed absolute difference for the test spectrum under "line_dropout" is greater than 0.0 (degraded match);
    - the delta absolute difference for the test spectrum under "line_dropout" is greater than 0.0.
    """

    def dummy_pipeline(wavelengths, intensities, elements):
        # A simple dummy pipeline that returns a composition based on the sum of intensities
        # If intensity is unmodified, sum is 5050
        s = np.sum(intensities)
        if s > 5000:
            return {"concentrations": {"Fe": 0.8, "Cr": 0.2}}  # Perfect match
        else:
            return {"concentrations": {"Fe": 0.5, "Cr": 0.5}}  # Bad match

    perturbations = {
        "line_dropout": lambda spec: line_dropout_perturbation(spec, top_n=50),
    }

    report = run_perturbation_battery(dummy_pipeline, [dummy_spectrum], perturbations)

    assert report.pipeline_name == "pipeline"
    assert "test_spec_1" in report.baseline_d_a
    assert report.baseline_d_a["test_spec_1"] == pytest.approx(0.0, abs=1e-10)

    assert "test_spec_1" in report.perturbed_d_a["line_dropout"]
    assert report.perturbed_d_a["line_dropout"]["test_spec_1"] > 0.0

    assert "test_spec_1" in report.delta_d_a["line_dropout"]
    assert report.delta_d_a["line_dropout"]["test_spec_1"] > 0.0
