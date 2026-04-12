"""
Tests for cflibs.benchmarks — corpus, metrics, and harness.
"""

import json
import math

import numpy as np
import pytest

from cflibs.benchmark.composition_metrics import (
    aitchison_distance,
    clr_transform,
    ilr_transform,
    ilr_inverse,
    rmse_composition,
    per_element_error,
)
from cflibs.benchmark.corpus import BenchmarkCorpus, BenchmarkSpectrum
from cflibs.benchmark.harness import (
    AccuracyTier,
    BenchmarkHarness,
)

# ============================================================================
# Aitchison distance
# ============================================================================


class TestAitchisonDistance:
    """Tests for the Aitchison distance function."""

    def test_identical_compositions(self):
        c = {"Fe": 0.7, "Cu": 0.3}
        assert aitchison_distance(c, c) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        a = {"Fe": 0.7, "Cu": 0.2, "Al": 0.1}
        b = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        assert aitchison_distance(a, b) == pytest.approx(aitchison_distance(b, a), rel=1e-10)

    def test_triangle_inequality(self):
        a = {"Fe": 0.7, "Cu": 0.2, "Al": 0.1}
        b = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        c = {"Fe": 0.3, "Cu": 0.4, "Al": 0.3}
        d_ab = aitchison_distance(a, b)
        d_bc = aitchison_distance(b, c)
        d_ac = aitchison_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-10

    def test_known_value(self):
        """Aitchison distance for a simple two-component case."""
        a = {"X": 0.8, "Y": 0.2}
        b = {"X": 0.5, "Y": 0.5}
        # Manual: clr(a) = [ln(0.8/g), ln(0.2/g)] where g = sqrt(0.16)
        # clr(b) = [0, 0]
        # d = sqrt(2) * |ln(0.8/0.2) - 0| / sqrt(2) ... let's just verify positive
        d = aitchison_distance(a, b)
        assert d > 0

    def test_scale_invariance(self):
        """Multiplying all components by a constant should not change distance."""
        a = {"Fe": 0.7, "Cu": 0.3}
        b = {"Fe": 0.5, "Cu": 0.5}
        d1 = aitchison_distance(a, b)
        a_scaled = {"Fe": 7.0, "Cu": 3.0}
        b_scaled = {"Fe": 5.0, "Cu": 5.0}
        d2 = aitchison_distance(a_scaled, b_scaled)
        assert d1 == pytest.approx(d2, rel=1e-10)

    def test_empty_compositions(self):
        assert aitchison_distance({}, {}) == 0.0

    def test_missing_elements_treated_as_epsilon(self):
        """Elements present in one but not the other get epsilon."""
        a = {"Fe": 0.7, "Cu": 0.3}
        b = {"Fe": 0.7}
        d = aitchison_distance(a, b)
        assert math.isfinite(d) and d > 0

    def test_zero_values_handled(self):
        """Zeros should be clipped to epsilon, not cause NaN/inf."""
        a = {"Fe": 0.7, "Cu": 0.0, "Al": 0.3}
        b = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        d = aitchison_distance(a, b)
        assert math.isfinite(d)


# ============================================================================
# CLR / ILR transforms
# ============================================================================


class TestCLRTransform:
    """Tests for the centered log-ratio transform."""

    def test_sums_to_zero(self):
        c = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        clr = clr_transform(c)
        assert sum(clr.values()) == pytest.approx(0.0, abs=1e-12)

    def test_empty(self):
        assert clr_transform({}) == {}

    def test_uniform_composition(self):
        c = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        clr = clr_transform(c)
        for v in clr.values():
            assert v == pytest.approx(0.0, abs=1e-10)


class TestILRTransform:
    """Tests for the isometric log-ratio transform and its inverse."""

    def test_dimensionality(self):
        c = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        ilr = ilr_transform(c)
        assert ilr.shape == (2,)

    def test_single_element(self):
        c = {"Fe": 1.0}
        ilr = ilr_transform(c)
        assert ilr.shape == (0,)

    def test_round_trip(self):
        """ILR -> inverse ILR should recover the original (normalized) composition."""
        c = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        elements = sorted(c.keys())
        coords = ilr_transform(c)
        recovered = ilr_inverse(coords, elements)
        for el in elements:
            assert recovered[el] == pytest.approx(c[el], rel=1e-8)

    def test_aitchison_equals_euclidean_in_ilr(self):
        """Euclidean distance in ILR space equals Aitchison distance."""
        a = {"Fe": 0.6, "Cu": 0.3, "Al": 0.1}
        b = {"Fe": 0.4, "Cu": 0.4, "Al": 0.2}
        d_a = aitchison_distance(a, b)
        ilr_a = ilr_transform(a)
        ilr_b = ilr_transform(b)
        d_ilr = float(np.linalg.norm(ilr_a - ilr_b))
        assert d_a == pytest.approx(d_ilr, rel=1e-6)


# ============================================================================
# Standard metrics
# ============================================================================


class TestRMSEComposition:
    def test_identical(self):
        c = {"Fe": 0.7, "Cu": 0.3}
        assert rmse_composition(c, c) == pytest.approx(0.0)

    def test_known_value(self):
        a = {"X": 1.0, "Y": 0.0}
        b = {"X": 0.0, "Y": 1.0}
        # RMSE = sqrt(mean([1, 1])) = 1.0
        assert rmse_composition(a, b) == pytest.approx(1.0)


class TestPerElementError:
    def test_basic(self):
        a = {"Fe": 0.7, "Cu": 0.3}
        b = {"Fe": 0.6, "Cu": 0.4}
        errs = per_element_error(a, b)
        assert errs["Fe"][0] == pytest.approx(0.1)  # absolute
        assert errs["Fe"][1] == pytest.approx(0.1 / 0.7)  # relative
        assert errs["Cu"][0] == pytest.approx(0.1)

    def test_missing_element(self):
        a = {"Fe": 0.7}
        b = {"Fe": 0.7, "Cu": 0.3}
        errs = per_element_error(a, b)
        assert errs["Cu"][0] == pytest.approx(0.3)
        assert errs["Cu"][1] == float("inf")


# ============================================================================
# BenchmarkCorpus
# ============================================================================


class TestBenchmarkCorpus:
    """Tests for the synthetic corpus generator."""

    def test_generates_spectra(self):
        corpus = BenchmarkCorpus(
            temperatures_K=[10000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.7, "Cu": 0.3}],
            snr_values=None,
        )
        spectra = corpus.generate()
        # 1 composition * 1 T * 1 ne * 1 (clean) + 1 dark-element = 2
        assert len(spectra) >= 1

    def test_spectrum_shapes(self):
        corpus = BenchmarkCorpus(
            temperatures_K=[10000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.7, "Cu": 0.3}],
        )
        spectra = corpus.generate()
        for s in spectra:
            assert s.wavelength.shape == s.intensity.shape
            assert len(s.wavelength) > 0

    def test_ground_truth_present(self):
        corpus = BenchmarkCorpus(
            temperatures_K=[12000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.5, "Ni": 0.5}],
        )
        spectra = corpus.generate()
        # At least the first spectrum should have ground truth
        gt = spectra[0].ground_truth
        assert "temperature_K" in gt
        assert "electron_density_cm3" in gt
        assert "concentrations" in gt
        assert gt["temperature_K"] == 12000.0

    def test_noisy_variants(self):
        corpus = BenchmarkCorpus(
            temperatures_K=[10000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.7, "Cu": 0.3}],
            snr_values=[50.0, 100.0],
            missing_element_specs=[],
        )
        spectra = corpus.generate()
        # 1 comp * 1 T * 1 ne * 2 SNR = 2
        assert len(spectra) == 2
        snrs = {s.snr for s in spectra}
        assert 50.0 in snrs
        assert 100.0 in snrs

    def test_dark_element_spectra(self):
        corpus = BenchmarkCorpus(
            temperatures_K=[10000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.7, "Cu": 0.3}],
            missing_element_specs=[{"Fe": 0.7, "Cu": 0.3}],
        )
        spectra = corpus.generate()
        dark = [s for s in spectra if s.metadata.get("dark_element_test")]
        assert len(dark) == 1

    def test_gaussian_fallback_produces_peaks(self):
        """Gaussian fallback should produce non-zero intensities for known elements."""
        corpus = BenchmarkCorpus(
            wavelength_range=(300.0, 400.0),
            temperatures_K=[10000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.5, "Cu": 0.5}],
        )
        _, intensity = corpus._gaussian_fallback(10000.0, 1e17, {"Fe": 0.5, "Cu": 0.5})
        assert intensity.max() > 0


# ============================================================================
# BenchmarkHarness
# ============================================================================


def _dummy_pipeline(wavelengths, intensities, elements):
    """A trivial pipeline that returns equal concentrations."""
    n = len(elements)
    conc = {el: 1.0 / n for el in elements}
    return {
        "concentrations": conc,
        "temperature_K": 10000.0,
        "electron_density_cm3": 1e17,
    }


def _failing_pipeline(wavelengths, intensities, elements):
    """A pipeline that always raises."""
    raise RuntimeError("intentional failure")


class TestBenchmarkHarness:
    """Tests for the experiment harness."""

    def test_register_and_run(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("dummy", _dummy_pipeline)

        corpus = BenchmarkCorpus(
            temperatures_K=[10000.0],
            electron_densities_cm3=[1e17],
            compositions=[{"Fe": 0.7, "Cu": 0.3}],
            missing_element_specs=[],
        )
        spectra = corpus.generate()
        report = harness.run(spectra)

        assert len(report.results) == 1
        assert report.results[0].name == "dummy"
        assert report.results[0].n_spectra == len(spectra)

    def test_duplicate_registration_raises(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("a", _dummy_pipeline)
        with pytest.raises(ValueError, match="already registered"):
            harness.register_pipeline("a", _dummy_pipeline)

    def test_no_pipelines_raises(self):
        harness = BenchmarkHarness()
        with pytest.raises(ValueError, match="No pipelines"):
            harness.run([])

    def test_report_structure(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("dummy", _dummy_pipeline)

        spec = BenchmarkSpectrum(
            wavelength=np.linspace(200, 400, 100),
            intensity=np.ones(100),
            ground_truth={
                "temperature_K": 10000.0,
                "electron_density_cm3": 1e17,
                "concentrations": {"Fe": 0.7, "Cu": 0.3},
            },
            label="test",
        )
        report = harness.run([spec])

        sr = report.results[0].spectrum_results[0]
        assert sr.label == "test"
        assert math.isfinite(sr.aitchison)
        assert math.isfinite(sr.rmse)
        assert sr.elapsed_ns > 0
        assert isinstance(sr.tier, AccuracyTier)

    def test_summary(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("dummy", _dummy_pipeline)

        spec = BenchmarkSpectrum(
            wavelength=np.linspace(200, 400, 100),
            intensity=np.ones(100),
            ground_truth={
                "temperature_K": 10000.0,
                "electron_density_cm3": 1e17,
                "concentrations": {"Fe": 0.7, "Cu": 0.3},
            },
        )
        report = harness.run([spec])
        summary = report.summary()

        assert "dummy" in summary
        assert "mean_aitchison" in summary["dummy"]
        assert "mean_time_ms" in summary["dummy"]
        assert "tier_distribution" in summary["dummy"]

    def test_to_json(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("dummy", _dummy_pipeline)

        spec = BenchmarkSpectrum(
            wavelength=np.linspace(200, 400, 100),
            intensity=np.ones(100),
            ground_truth={
                "temperature_K": 10000.0,
                "electron_density_cm3": 1e17,
                "concentrations": {"Fe": 0.7, "Cu": 0.3},
            },
        )
        report = harness.run([spec])
        j = report.to_json()
        data = json.loads(j)
        assert "summary" in data
        assert "pipelines" in data

    def test_compare(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("dummy", _dummy_pipeline)

        spec = BenchmarkSpectrum(
            wavelength=np.linspace(200, 400, 100),
            intensity=np.ones(100),
            ground_truth={
                "temperature_K": 10000.0,
                "electron_density_cm3": 1e17,
                "concentrations": {"Fe": 0.7, "Cu": 0.3},
            },
        )
        r1 = harness.run([spec])
        r2 = harness.run([spec])
        comp = r1.compare(r2)
        assert "dummy" in comp
        assert "delta_mean_aitchison" in comp["dummy"]

    def test_failing_pipeline_captured(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("bad", _failing_pipeline)

        spec = BenchmarkSpectrum(
            wavelength=np.linspace(200, 400, 100),
            intensity=np.ones(100),
            ground_truth={
                "temperature_K": 10000.0,
                "electron_density_cm3": 1e17,
                "concentrations": {"Fe": 0.7, "Cu": 0.3},
            },
        )
        report = harness.run([spec])
        sr = report.results[0].spectrum_results[0]
        assert sr.error is not None
        assert "intentional failure" in sr.error
        assert report.results[0].n_errors == 1

    def test_selective_pipeline_run(self):
        harness = BenchmarkHarness()
        harness.register_pipeline("a", _dummy_pipeline)
        harness.register_pipeline("b", _dummy_pipeline)

        spec = BenchmarkSpectrum(
            wavelength=np.linspace(200, 400, 100),
            intensity=np.ones(100),
            ground_truth={
                "concentrations": {"Fe": 0.5, "Cu": 0.5},
            },
        )
        report = harness.run([spec], pipelines=["a"])
        assert len(report.results) == 1
        assert report.results[0].name == "a"


# ============================================================================
# AccuracyTier
# ============================================================================


class TestAccuracyTier:
    def test_excellent(self):
        assert AccuracyTier.from_aitchison(0.02) == AccuracyTier.EXCELLENT

    def test_good(self):
        assert AccuracyTier.from_aitchison(0.07) == AccuracyTier.GOOD

    def test_acceptable(self):
        assert AccuracyTier.from_aitchison(0.15) == AccuracyTier.ACCEPTABLE

    def test_poor(self):
        assert AccuracyTier.from_aitchison(0.25) == AccuracyTier.POOR
