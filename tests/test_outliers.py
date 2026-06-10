"""
Tests for outlier detection methods (SAM and MAD).

Tests cover:
- SAM: Basic angle computation, pairwise matrix, outlier detection
- MAD: 1D, spectrum, and channel mode outlier detection
- Edge cases and error handling
"""

import numpy as np
import pytest

from cflibs.inversion.preprocess.outliers import (
    SAMResult,
    SpectralAngleMapper,
    sam_distance,
    detect_outlier_spectra,
    MADResult,
    MADOutlierDetector,
    mad_outliers_1d,
    mad_outliers_spectra,
    mad_clean_channels,
    MAD_SCALE_FACTOR,
)


class TestSpectralAngle:
    """Tests for basic SAM angle computation."""

    def test_identical_spectra_zero_angle(self):
        """Identical spectra should have zero angle."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sam = SpectralAngleMapper()
        angle = sam.spectral_angle(spectrum, spectrum)
        assert angle == pytest.approx(0.0, abs=1e-10)

    def test_scaled_spectra_zero_angle(self):
        """Scaled spectra should have zero angle (SAM is scale-invariant)."""
        spectrum1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectrum2 = spectrum1 * 10.0  # Same shape, different scale
        sam = SpectralAngleMapper()
        angle = sam.spectral_angle(spectrum1, spectrum2)
        assert angle == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_spectra_90_degrees(self):
        """Orthogonal spectra should have 90 degree angle."""
        spectrum1 = np.array([1.0, 0.0, 0.0])
        spectrum2 = np.array([0.0, 1.0, 0.0])
        sam = SpectralAngleMapper()
        angle = sam.spectral_angle(spectrum1, spectrum2)
        assert angle == pytest.approx(np.pi / 2, abs=1e-10)

    def test_opposite_spectra_180_degrees(self):
        """Opposite spectra should have 180 degree angle."""
        spectrum1 = np.array([1.0, 0.0, 0.0])
        spectrum2 = np.array([-1.0, 0.0, 0.0])
        sam = SpectralAngleMapper()
        angle = sam.spectral_angle(spectrum1, spectrum2)
        assert angle == pytest.approx(np.pi, abs=1e-10)

    def test_45_degree_angle(self):
        """Test known 45 degree angle case."""
        spectrum1 = np.array([1.0, 0.0])
        spectrum2 = np.array([1.0, 1.0])
        sam = SpectralAngleMapper()
        angle = sam.spectral_angle(spectrum1, spectrum2)
        assert angle == pytest.approx(np.pi / 4, abs=1e-10)

    def test_symmetry(self):
        """SAM(a, b) should equal SAM(b, a)."""
        spectrum1 = np.array([1.0, 2.0, 3.0, 4.0])
        spectrum2 = np.array([4.0, 3.0, 2.0, 1.0])
        sam = SpectralAngleMapper()
        angle1 = sam.spectral_angle(spectrum1, spectrum2)
        angle2 = sam.spectral_angle(spectrum2, spectrum1)
        assert angle1 == pytest.approx(angle2, abs=1e-10)

    def test_shape_mismatch_raises(self):
        """Different length spectra should raise ValueError."""
        spectrum1 = np.array([1.0, 2.0, 3.0])
        spectrum2 = np.array([1.0, 2.0])
        sam = SpectralAngleMapper()
        with pytest.raises(ValueError, match="shape mismatch"):
            sam.spectral_angle(spectrum1, spectrum2)

    def test_zero_spectrum_raises(self):
        """Zero spectrum should raise ValueError."""
        spectrum1 = np.array([0.0, 0.0, 0.0])
        spectrum2 = np.array([1.0, 2.0, 3.0])
        sam = SpectralAngleMapper()
        with pytest.raises(ValueError, match="zero or near-zero"):
            sam.spectral_angle(spectrum1, spectrum2)


class TestSamDistanceFunction:
    """Tests for the convenience sam_distance function."""

    def test_sam_distance_identical(self):
        """Identical spectra should have zero distance."""
        spectrum = np.array([1.0, 2.0, 3.0])
        dist = sam_distance(spectrum, spectrum)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_sam_distance_returns_float(self):
        """sam_distance should return a Python float."""
        spectrum1 = np.array([1.0, 2.0, 3.0])
        spectrum2 = np.array([3.0, 2.0, 1.0])
        dist = sam_distance(spectrum1, spectrum2)
        assert isinstance(dist, float)


class TestPairwiseAngles:
    """Tests for pairwise angle matrix computation."""

    def test_pairwise_diagonal_zero(self):
        """Diagonal of pairwise matrix should be zero."""
        spectra = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        sam = SpectralAngleMapper()
        angles = sam.pairwise_angles(spectra)
        np.testing.assert_array_almost_equal(np.diag(angles), 0.0)

    def test_pairwise_symmetric(self):
        """Pairwise matrix should be symmetric."""
        spectra = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 3.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        )
        sam = SpectralAngleMapper()
        angles = sam.pairwise_angles(spectra)
        np.testing.assert_array_almost_equal(angles, angles.T)

    def test_pairwise_shape(self):
        """Pairwise matrix should have correct shape."""
        n_spectra = 5
        n_wavelengths = 100
        spectra = np.random.default_rng(42).random((n_spectra, n_wavelengths))
        sam = SpectralAngleMapper()
        angles = sam.pairwise_angles(spectra)
        assert angles.shape == (n_spectra, n_spectra)

    def test_pairwise_nonnegative(self):
        """All angles should be non-negative."""
        spectra = np.random.default_rng(42).random((10, 50))
        sam = SpectralAngleMapper()
        angles = sam.pairwise_angles(spectra)
        assert np.all(angles >= 0)


class TestAnglesFromReference:
    """Tests for computing angles from a reference spectrum."""

    def test_angles_from_mean_reference(self):
        """Test angles computed from mean reference."""
        spectra = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],  # Same as first
                [1.0, 2.0, 3.0],  # Same as first
            ]
        )
        sam = SpectralAngleMapper(reference_method="mean")
        angles, ref = sam.angles_from_reference(spectra)
        # All spectra are identical to mean, angles should be 0
        np.testing.assert_array_almost_equal(angles, 0.0)

    def test_angles_from_median_reference(self):
        """Test angles computed from median reference."""
        spectra = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [100.0, 200.0, 300.0],  # Outlier (same shape, different scale)
            ]
        )
        sam = SpectralAngleMapper(reference_method="median")
        angles, ref = sam.angles_from_reference(spectra)
        # Median should be [1, 2, 3], all have same shape so angle = 0
        np.testing.assert_array_almost_equal(angles, 0.0, decimal=5)

    def test_custom_reference(self):
        """Test angles from custom reference spectrum."""
        spectra = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        reference = np.array([1.0, 0.0, 0.0])
        sam = SpectralAngleMapper()
        angles, ref = sam.angles_from_reference(spectra, reference=reference)

        # First spectrum matches reference
        assert angles[0] == pytest.approx(0.0, abs=1e-10)
        # Others are orthogonal
        assert angles[1] == pytest.approx(np.pi / 2, abs=1e-10)
        assert angles[2] == pytest.approx(np.pi / 2, abs=1e-10)


class TestOutlierDetection:
    """Tests for outlier detection functionality."""

    def test_no_outliers_identical_spectra(self):
        """Identical spectra should have no outliers."""
        base_spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.tile(base_spectrum, (10, 1))

        sam = SpectralAngleMapper()
        result = sam.detect_outliers(spectra)

        assert result.n_outliers == 0
        assert np.all(~result.outlier_mask)

    def test_obvious_outlier_detected(self):
        """An obviously different spectrum should be detected."""
        # Create 9 similar spectra and 1 very different one
        base_spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.tile(base_spectrum, (10, 1))
        # Make the last spectrum orthogonal
        spectra[9] = np.array([5.0, 4.0, 3.0, 2.0, 1.0]) * 0.01 + np.array(
            [0.0, 0.0, 10.0, 0.0, 0.0]
        )

        sam = SpectralAngleMapper(threshold_sigma=2.0)
        result = sam.detect_outliers(spectra)

        assert result.n_outliers >= 1
        assert 9 in result.outlier_indices

    def test_fixed_threshold(self):
        """Test fixed threshold method."""
        spectra = np.random.default_rng(42).random((20, 50))
        threshold = 0.5  # radians (~28 degrees)

        sam = SpectralAngleMapper(threshold_method="fixed", threshold_fixed=threshold)
        result = sam.detect_outliers(spectra)

        assert result.threshold == threshold
        assert result.method == "fixed"

    def test_percentile_threshold(self):
        """Test percentile threshold method."""
        spectra = np.random.default_rng(42).random((20, 50))

        sam = SpectralAngleMapper(threshold_method="percentile", threshold_percentile=90.0)
        result = sam.detect_outliers(spectra)

        assert result.method == "percentile"
        # At least 10% should be flagged (those above 90th percentile)
        assert result.n_outliers >= 2

    def test_mad_threshold(self):
        """Test MAD threshold method."""
        spectra = np.random.default_rng(42).random((20, 50))

        sam = SpectralAngleMapper(threshold_method="mad", threshold_sigma=3.0)
        result = sam.detect_outliers(spectra)

        assert result.method == "mad"
        # 3-sigma should catch only extreme outliers
        assert result.n_outliers < len(spectra)

    def test_override_threshold(self):
        """Test overriding automatic threshold."""
        spectra = np.random.default_rng(42).random((20, 50))

        sam = SpectralAngleMapper(threshold_method="mad")
        result = sam.detect_outliers(spectra, threshold=0.1)  # Override

        assert result.threshold == 0.1
        assert result.method == "fixed"  # Method changes to "fixed" when overridden


class TestSAMResult:
    """Tests for SAMResult dataclass."""

    def test_result_properties(self):
        """Test SAMResult computed properties."""
        result = SAMResult(
            angles=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            outlier_mask=np.array([False, False, False, False, True]),
            threshold=0.45,
            reference_spectrum=np.array([1.0, 2.0, 3.0]),
            n_outliers=1,
            inlier_indices=np.array([0, 1, 2, 3]),
            outlier_indices=np.array([4]),
            method="mad",
        )

        assert result.n_inliers == 4
        assert result.outlier_fraction == pytest.approx(0.2)
        assert result.angles_degrees()[0] == pytest.approx(np.degrees(0.1))

    def test_result_summary(self):
        """Test SAMResult summary string generation."""
        result = SAMResult(
            angles=np.array([0.1, 0.2, 0.3]),
            outlier_mask=np.array([False, False, True]),
            threshold=0.25,
            reference_spectrum=np.array([1.0, 2.0]),
            n_outliers=1,
            inlier_indices=np.array([0, 1]),
            outlier_indices=np.array([2]),
            method="mad",
        )

        summary = result.summary()
        assert "SAM Outlier Detection" in summary
        assert "Total spectra: 3" in summary
        assert "Outliers: 1" in summary


class TestFilterSpectra:
    """Tests for spectrum filtering functionality."""

    def test_filter_removes_outliers(self):
        """Filtering should remove outlier spectra."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectra = np.array(
            [
                base,
                base * 1.01,  # Similar
                base * 0.99,  # Similar
                np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # Very different
            ]
        )

        sam = SpectralAngleMapper(threshold_sigma=2.0)
        filtered, result = sam.filter_spectra(spectra)

        # Filtered should have fewer spectra if outlier was detected
        if result.n_outliers > 0:
            assert len(filtered) < len(spectra)
            assert len(filtered) == result.n_inliers


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_spectrum(self):
        """Single spectrum should return no outliers."""
        spectra = np.array([[1.0, 2.0, 3.0]])
        sam = SpectralAngleMapper()
        result = sam.detect_outliers(spectra)

        assert result.n_outliers == 0

    def test_two_spectra(self):
        """Two spectra should work correctly."""
        spectra = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
            ]
        )
        sam = SpectralAngleMapper()
        result = sam.detect_outliers(spectra)

        # Should run without error
        assert len(result.angles) == 2

    def test_invalid_threshold_method(self):
        """Invalid threshold method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown threshold_method"):
            SpectralAngleMapper(threshold_method="invalid")

    def test_invalid_reference_method(self):
        """Invalid reference method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown reference_method"):
            SpectralAngleMapper(reference_method="invalid")

    def test_1d_array_raises(self):
        """1D array should raise ValueError."""
        spectra = np.array([1.0, 2.0, 3.0])
        sam = SpectralAngleMapper()
        with pytest.raises(ValueError, match="Expected 2D array"):
            sam.detect_outliers(spectra)


class TestDetectOutlierSpectraConvenience:
    """Tests for the convenience function detect_outlier_spectra."""

    def test_detect_outlier_spectra_basic(self):
        """Test basic usage of convenience function."""
        spectra = np.random.default_rng(42).random((10, 50))
        result = detect_outlier_spectra(spectra)

        assert isinstance(result, SAMResult)
        assert len(result.angles) == 10

    def test_detect_outlier_spectra_with_params(self):
        """Test convenience function with parameters."""
        spectra = np.random.default_rng(42).random((10, 50))
        result = detect_outlier_spectra(
            spectra,
            threshold_sigma=2.5,
            reference_method="median",
        )

        assert result.method == "mad"


class TestRealisticLIBSScenario:
    """Tests simulating realistic LIBS measurement scenarios."""

    def test_shot_to_shot_variation(self):
        """Test with simulated shot-to-shot intensity variation."""
        rng = np.random.default_rng(42)

        # Base spectrum (simulated LIBS with peaks)
        wavelengths = np.linspace(300, 400, 1000)
        base_spectrum = np.zeros_like(wavelengths)
        # Add some peaks
        for peak_wl in [320, 340, 360, 380]:
            base_spectrum += 100 * np.exp(-((wavelengths - peak_wl) ** 2) / (2 * 2**2))
        base_spectrum += 10  # Background

        # Create 20 spectra with shot-to-shot variation (just scaling)
        spectra = []
        for i in range(20):
            scale = rng.uniform(0.8, 1.2)  # +/- 20% intensity variation
            noise = rng.normal(0, 2, len(wavelengths))  # Small noise
            spectra.append(base_spectrum * scale + noise)

        # Add one outlier (different peak ratios - indicates plasma issue)
        outlier = base_spectrum.copy()
        outlier[300:400] *= 3  # Different peak ratio
        spectra.append(outlier)

        spectra = np.array(spectra)

        sam = SpectralAngleMapper(threshold_sigma=3.0)
        result = sam.detect_outliers(spectra)

        # The outlier should be detected
        assert 20 in result.outlier_indices

    def test_plasma_failure_detection(self):
        """Test detection of spectra where plasma didn't form properly."""
        rng = np.random.default_rng(42)

        # Normal spectra with emission lines
        wavelengths = np.linspace(300, 400, 500)
        base_spectrum = 20 + 80 * np.exp(-((wavelengths - 350) ** 2) / 50)

        spectra = []
        for _ in range(19):
            noise = rng.normal(0, 2, len(wavelengths))
            spectra.append(base_spectrum + noise)

        # Add spectrum where plasma didn't form (just background)
        failed_spectrum = 20 + rng.normal(0, 2, len(wavelengths))
        spectra.append(failed_spectrum)

        spectra = np.array(spectra)

        result = detect_outlier_spectra(spectra, threshold_sigma=2.5)

        # The failed plasma spectrum should be detected
        assert 19 in result.outlier_indices


# =============================================================================
# MAD (Median Absolute Deviation) Tests
# =============================================================================


class TestMADBasics:
    """Tests for basic MAD computation."""

    def test_mad_scale_factor(self):
        """MAD scale factor should be approximately 1.4826."""
        assert MAD_SCALE_FACTOR == pytest.approx(1.4826, rel=1e-4)

    def test_mad_of_constant_is_zero(self):
        """MAD of constant data should be zero."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        detector = MADOutlierDetector(mode="1d")
        median, mad = detector.compute_mad(data)
        assert mad == 0.0
        assert median == 5.0

    def test_mad_symmetric_data(self):
        """Test MAD on symmetric data around median."""
        # Data: [-2, -1, 0, 1, 2] has median=0, deviations=[2,1,0,1,2], MAD=1
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        detector = MADOutlierDetector(mode="1d")
        median, mad = detector.compute_mad(data)
        assert median == 0.0
        assert mad == 1.0

    def test_mad_with_outlier(self):
        """MAD should be robust to outliers."""
        # Normal data plus one extreme outlier
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
        detector = MADOutlierDetector(mode="1d")
        median, mad = detector.compute_mad(data)
        # Median should be (3+4)/2 = 3.5
        assert median == pytest.approx(3.5)
        # Deviations from 3.5: [2.5, 1.5, 0.5, 0.5, 1.5, 996.5]
        # MAD = median([2.5, 1.5, 0.5, 0.5, 1.5, 996.5]) = (1.5+1.5)/2 = 1.5
        assert mad == pytest.approx(1.5)


class TestMAD1DMode:
    """Tests for MAD in 1D mode."""

    def test_detect_obvious_outlier(self):
        """Obvious outlier should be detected."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = mad_outliers_1d(data, threshold=3.0)

        assert result.n_outliers >= 1
        assert 5 in result.outlier_indices  # Index of 100.0

    def test_no_outliers_in_normal_data(self):
        """Normally distributed data with 3-sigma threshold should have few outliers."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=1000)

        result = mad_outliers_1d(data, threshold=3.0)

        # With 3-sigma, expect < 1% outliers for normal distribution
        assert result.outlier_fraction < 0.02

    def test_result_properties(self):
        """Test MADResult properties."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad_outliers_1d(data, threshold=3.0)

        assert result.mode == "1d"
        assert result.n_inliers + result.n_outliers == len(data)
        assert len(result.inlier_indices) + len(result.outlier_indices) == len(data)

    def test_1d_wrong_shape_raises(self):
        """2D data in 1D mode should raise."""
        data = np.array([[1, 2], [3, 4]])
        detector = MADOutlierDetector(mode="1d")
        with pytest.raises(ValueError, match="Expected 1D array"):
            detector.detect_outliers(data)


class TestMADSpectrumMode:
    """Tests for MAD in spectrum mode."""

    def test_detect_bright_spectrum(self):
        """Abnormally bright spectrum should be detected."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(100, 10, size=(20, 100))
        # Make one spectrum 10x brighter
        spectra[5] *= 10

        result = mad_outliers_spectra(spectra, threshold=3.0, statistic="total_intensity")

        assert 5 in result.outlier_indices

    def test_detect_dim_spectrum(self):
        """Abnormally dim spectrum should be detected."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(100, 10, size=(20, 100))
        # Make one spectrum 10x dimmer
        spectra[10] /= 10

        result = mad_outliers_spectra(spectra, threshold=3.0, statistic="total_intensity")

        assert 10 in result.outlier_indices

    def test_different_statistics(self):
        """Test different summary statistics."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(100, 10, size=(10, 50))

        for stat in ["total_intensity", "max_intensity", "mean_intensity", "std_intensity"]:
            detector = MADOutlierDetector(mode="spectrum", statistic=stat)
            result = detector.detect_outliers(spectra)
            assert result.statistic == stat
            assert result.mode == "spectrum"

    def test_filter_spectra(self):
        """Test filter_spectra convenience method."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(100, 10, size=(10, 50))
        spectra[0] *= 100  # Make first spectrum an outlier

        detector = MADOutlierDetector(mode="spectrum", threshold=3.0)
        filtered, result = detector.filter_spectra(spectra)

        assert filtered.shape[0] < spectra.shape[0]
        assert 0 not in result.inlier_indices

    def test_spectrum_mode_wrong_shape_raises(self):
        """1D data in spectrum mode should raise."""
        data = np.array([1.0, 2.0, 3.0])
        detector = MADOutlierDetector(mode="spectrum")
        with pytest.raises(ValueError, match="Expected 2D array"):
            detector.detect_outliers(data)


class TestMADChannelMode:
    """Tests for MAD in channel mode."""

    def test_detect_spike(self):
        """Spike in single channel should be detected."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(100, 5, size=(10, 50))
        # Add a spike to one spectrum at one channel
        spectra[3, 25] = 1000

        detector = MADOutlierDetector(mode="channel", threshold=3.0)
        result = detector.detect_outliers(spectra)

        assert result.outlier_mask[3, 25]
        assert result.n_outliers >= 1

    def test_channel_mode_shape(self):
        """Channel mode outlier mask should have same shape as input."""
        spectra = np.random.default_rng(42).normal(100, 5, size=(10, 50))
        detector = MADOutlierDetector(mode="channel")
        result = detector.detect_outliers(spectra)

        assert result.outlier_mask.shape == spectra.shape

    def test_clean_channels_median(self):
        """Test cleaning channels with median replacement."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(100, 5, size=(10, 50))
        spectra[3, 25] = 1000  # Spike

        cleaned, result = mad_clean_channels(spectra, threshold=3.0, replacement="median")

        # Spike should be replaced with median
        assert cleaned[3, 25] != 1000
        assert cleaned[3, 25] == pytest.approx(result.median[25], rel=0.01)

    def test_clean_channels_nan(self):
        """Test cleaning channels with NaN replacement."""
        spectra = np.array([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 1000.0]])

        cleaned, result = mad_clean_channels(spectra, threshold=2.0, replacement="nan")

        # The spike should become NaN
        assert np.isnan(cleaned[2, 2])

    def test_clean_channels_interpolate(self):
        """Test cleaning channels with interpolation."""
        # Need more spectra for robust MAD estimation
        spectra = np.array(
            [
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [10.0, 20.0, 1000.0, 40.0, 50.0],  # Spike at index 2
            ]
        )

        cleaned, result = mad_clean_channels(spectra, threshold=2.0, replacement="interpolate")

        # The spike should be interpolated from neighbors (20, 40)
        assert cleaned[5, 2] != 1000.0
        assert cleaned[5, 2] == pytest.approx(30.0, rel=0.1)  # Interpolated value


class TestMADEdgeCases:
    """Tests for MAD edge cases."""

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            MADOutlierDetector(mode="invalid")

    def test_invalid_statistic_raises(self):
        """Invalid statistic should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            MADOutlierDetector(statistic="invalid")

    def test_negative_threshold_raises(self):
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="Threshold must be positive"):
            MADOutlierDetector(threshold=-1.0)

    def test_filter_spectra_wrong_mode_raises(self):
        """filter_spectra in non-spectrum mode should raise."""
        detector = MADOutlierDetector(mode="1d")
        with pytest.raises(ValueError, match="filter_spectra only works"):
            detector.filter_spectra(np.array([[1, 2], [3, 4]]))

    def test_clean_channels_wrong_mode_raises(self):
        """clean_channels in non-channel mode should raise."""
        detector = MADOutlierDetector(mode="spectrum")
        with pytest.raises(ValueError, match="clean_channels only works"):
            detector.clean_channels(np.array([[1, 2], [3, 4]]))

    def test_constant_data_no_outliers(self):
        """Constant data should have no outliers (MAD=0)."""
        data = np.ones(100)
        result = mad_outliers_1d(data)
        assert result.n_outliers == 0

    def test_two_values(self):
        """Should work with just two values."""
        data = np.array([1.0, 100.0])
        result = mad_outliers_1d(data, threshold=3.0)
        # With only 2 values, MAD is |1-50.5| = |100-50.5| = 49.5
        # Both are equidistant from median
        assert result.n_outliers == 0 or result.n_outliers == 2


class TestMADResult:
    """Tests for MADResult dataclass."""

    def test_summary_1d(self):
        """Test summary output for 1D mode."""
        result = MADResult(
            outlier_mask=np.array([False, False, True]),
            median=np.array([2.0]),
            mad=np.array([1.0]),
            threshold=3.0,
            n_outliers=1,
            mode="1d",
            statistic="",
            inlier_indices=np.array([0, 1]),
            outlier_indices=np.array([2]),
        )
        summary = result.summary()
        assert "MAD Outlier Detection" in summary
        assert "1d mode" in summary
        assert "Outliers: 1" in summary

    def test_summary_spectrum(self):
        """Test summary output for spectrum mode."""
        result = MADResult(
            outlier_mask=np.array([False, False, True]),
            median=np.array([100.0]),
            mad=np.array([10.0]),
            threshold=3.0,
            n_outliers=1,
            mode="spectrum",
            statistic="total_intensity",
            inlier_indices=np.array([0, 1]),
            outlier_indices=np.array([2]),
        )
        summary = result.summary()
        assert "spectrum mode" in summary
        assert "total_intensity" in summary

    def test_outlier_fraction(self):
        """Test outlier_fraction property."""
        result = MADResult(
            outlier_mask=np.array([False, True, False, True]),
            median=np.array([0.0]),
            mad=np.array([1.0]),
            threshold=3.0,
            n_outliers=2,
            mode="1d",
            inlier_indices=np.array([0, 2]),
            outlier_indices=np.array([1, 3]),
        )
        assert result.outlier_fraction == 0.5


class TestMADRealisticScenarios:
    """Tests with realistic LIBS scenarios."""

    def test_laser_energy_fluctuation(self):
        """Test with simulated laser energy fluctuations."""
        rng = np.random.default_rng(42)

        # Base intensities for 50 channels
        base = np.linspace(50, 150, 50)

        # 20 spectra with normal variation
        spectra = []
        for _ in range(20):
            scale = rng.uniform(0.9, 1.1)  # 10% energy fluctuation
            noise = rng.normal(0, 5, 50)
            spectra.append(base * scale + noise)

        # Add one spectrum with failed shot (very low intensity)
        spectra.append(base * 0.1 + rng.normal(0, 5, 50))

        spectra = np.array(spectra)

        # Should detect the failed shot
        result = mad_outliers_spectra(spectra, threshold=3.0)
        assert 20 in result.outlier_indices

    def test_cosmic_ray_spike(self):
        """Test detection of cosmic ray spikes in CCD data."""
        rng = np.random.default_rng(42)

        # Normal spectra
        spectra = rng.normal(100, 10, size=(10, 100))

        # Add cosmic ray spikes (single channel, very high)
        spectra[2, 50] = 10000  # Cosmic ray
        spectra[7, 30] = 8000  # Another cosmic ray

        cleaned, result = mad_clean_channels(spectra, threshold=3.0)

        # Both spikes should be detected
        assert result.outlier_mask[2, 50]
        assert result.outlier_mask[7, 30]

        # Cleaned values should be close to median
        assert cleaned[2, 50] < 1000
        assert cleaned[7, 30] < 1000
