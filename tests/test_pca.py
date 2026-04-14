"""
Tests for PCA pipeline.

Tests cover:
- Basic PCA fitting and transformation
- Variance explanation and component selection
- Reconstruction and denoising
- Edge cases and error handling
- JAX backend (when available)
"""

import numpy as np
import pytest

from cflibs.inversion.pca import (
    PCAResult,
    PCAPipeline,
    fit_pca,
    denoise_spectra,
    explained_variance_curve,
    HAS_JAX,
)

# --- Fixtures ---


@pytest.fixture
def simple_data():
    """Simple 2D data with clear principal direction."""
    rng = np.random.default_rng(42)
    n_samples = 100

    # Data along primary axis with some noise
    x = rng.normal(0, 3, n_samples)
    y = rng.normal(0, 0.5, n_samples)

    # Stack into 2D array
    data = np.column_stack([x, y])
    return data


@pytest.fixture
def spectral_data():
    """Simulated spectral data with known structure."""
    rng = np.random.default_rng(123)
    n_samples = 50
    n_wavelengths = 200

    # Create base signal patterns (principal components)
    wavelengths = np.linspace(400, 800, n_wavelengths)

    # Component 1: broad Gaussian peak
    pc1 = np.exp(-((wavelengths - 550) ** 2) / (2 * 50**2))

    # Component 2: sharp peak
    pc2 = np.exp(-((wavelengths - 650) ** 2) / (2 * 10**2))

    # Component 3: baseline slope
    pc3 = (wavelengths - 400) / 400

    # Generate spectra as linear combinations + noise
    scores1 = rng.normal(10, 3, n_samples)
    scores2 = rng.normal(5, 1, n_samples)
    scores3 = rng.normal(2, 0.5, n_samples)

    spectra = (
        np.outer(scores1, pc1)
        + np.outer(scores2, pc2)
        + np.outer(scores3, pc3)
        + rng.normal(0, 0.1, (n_samples, n_wavelengths))
    )

    return spectra


@pytest.fixture
def noisy_spectral_data():
    """Spectral data with higher noise level for denoising tests."""
    rng = np.random.default_rng(456)
    n_samples = 30
    n_wavelengths = 100

    wavelengths = np.linspace(400, 700, n_wavelengths)
    pc1 = np.exp(-((wavelengths - 500) ** 2) / (2 * 30**2))
    pc2 = np.exp(-((wavelengths - 600) ** 2) / (2 * 20**2))

    scores1 = rng.normal(10, 2, n_samples)
    scores2 = rng.normal(5, 1, n_samples)

    signal = np.outer(scores1, pc1) + np.outer(scores2, pc2)
    noise = rng.normal(0, 0.5, (n_samples, n_wavelengths))

    return signal + noise, signal  # noisy data and clean signal


# --- Basic PCA Tests ---


class TestPCAPipelineBasics:
    """Tests for basic PCA fitting and transformation."""

    def test_fit_returns_result(self, spectral_data):
        """fit() should return PCAResult."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        assert isinstance(result, PCAResult)
        assert pca.is_fitted

    def test_components_shape(self, spectral_data):
        """Components should have correct shape."""
        n_components = 5
        pca = PCAPipeline(n_components=n_components)
        result = pca.fit(spectral_data)

        assert result.components.shape == (n_components, spectral_data.shape[1])
        assert result.n_components == n_components
        assert result.n_features == spectral_data.shape[1]
        assert result.n_samples == spectral_data.shape[0]

    def test_transform_shape(self, spectral_data):
        """Transform should produce correct shape."""
        n_components = 5
        pca = PCAPipeline(n_components=n_components)
        result = pca.fit(spectral_data)

        scores = result.transform(spectral_data)
        assert scores.shape == (spectral_data.shape[0], n_components)

    def test_fit_transform_equivalent(self, spectral_data):
        """fit_transform should equal fit then transform."""
        pca1 = PCAPipeline(n_components=5)
        scores1 = pca1.fit_transform(spectral_data)

        pca2 = PCAPipeline(n_components=5)
        result2 = pca2.fit(spectral_data)
        scores2 = result2.transform(spectral_data)

        np.testing.assert_allclose(scores1, scores2, rtol=1e-10)

    def test_inverse_transform_reconstruction(self, spectral_data):
        """inverse_transform(transform(X)) should approximate X."""
        # Use more components for better reconstruction
        pca = PCAPipeline(n_components=30)
        result = pca.fit(spectral_data)

        scores = result.transform(spectral_data)
        reconstructed = result.inverse_transform(scores)

        # With many components, reconstruction should be close
        # Use relative tolerance for values and absolute for near-zero values
        np.testing.assert_allclose(reconstructed, spectral_data, rtol=0.15, atol=0.2)

    def test_mean_centering(self, spectral_data):
        """Data should be centered around mean."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        np.testing.assert_allclose(result.mean, np.mean(spectral_data, axis=0), rtol=1e-10)


class TestVarianceExplanation:
    """Tests for variance explanation and component selection."""

    def test_variance_ratio_sums_to_one(self, spectral_data):
        """With all components, variance ratio should sum to ~1."""
        n_samples, n_features = spectral_data.shape
        max_components = min(n_samples, n_features)

        pca = PCAPipeline(n_components=max_components)
        result = pca.fit(spectral_data)

        # May not be exactly 1 due to numerical precision
        assert 0.99 <= result.cumulative_variance_ratio[-1] <= 1.01

    def test_variance_decreasing(self, spectral_data):
        """Explained variance should decrease with each component."""
        pca = PCAPipeline(n_components=10)
        result = pca.fit(spectral_data)

        # Each component should explain less variance than previous
        for i in range(1, len(result.explained_variance)):
            assert result.explained_variance[i] <= result.explained_variance[i - 1] + 1e-10

    def test_n_components_for_variance(self, spectral_data):
        """n_components_for_variance should find correct count."""
        pca = PCAPipeline(n_components=20)
        result = pca.fit(spectral_data)

        # Should return first component index where cumvar >= threshold
        for threshold in [0.8, 0.9, 0.95]:
            n = result.n_components_for_variance(threshold)
            assert result.cumulative_variance_ratio[n - 1] >= threshold
            if n > 1:
                assert result.cumulative_variance_ratio[n - 2] < threshold

    def test_variance_threshold_component_selection(self):
        """Using float n_components should select by variance."""
        # Create data where multiple components are needed for 95% variance
        rng = np.random.default_rng(999)
        n_samples = 100
        n_features = 50

        # Create data with more distributed variance
        # PC1: 40%, PC2: 30%, PC3: 15%, PC4: 10%, rest: 5%
        pc1 = rng.normal(0, 1, (n_samples, 1)) @ rng.normal(0, 1, (1, n_features))
        pc2 = rng.normal(0, 0.86, (n_samples, 1)) @ rng.normal(0, 1, (1, n_features))
        pc3 = rng.normal(0, 0.61, (n_samples, 1)) @ rng.normal(0, 1, (1, n_features))
        pc4 = rng.normal(0, 0.5, (n_samples, 1)) @ rng.normal(0, 1, (1, n_features))
        noise = rng.normal(0, 0.35, (n_samples, n_features))

        data = pc1 + pc2 + pc3 + pc4 + noise

        # Force NumPy backend for consistency
        pca = PCAPipeline(n_components=0.95, use_jax=False)
        result = pca.fit(data)

        # Must explain at least the requested variance
        assert result.cumulative_variance_ratio[-1] >= 0.95

        # Should need multiple components for this data
        assert result.n_components > 1


class TestReconstruction:
    """Tests for reconstruction and error computation."""

    def test_reconstruction_error_decreases(self, spectral_data):
        """Reconstruction error should decrease with more components."""
        errors = []

        for n_comp in [1, 3, 5, 10]:
            pca = PCAPipeline(n_components=n_comp)
            result = pca.fit(spectral_data)
            _, rmse = result.reconstruction_error(spectral_data)
            errors.append(rmse)

        # Each should be less than or equal to previous
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] + 1e-6

    def test_reconstruction_error_zero_with_all_components(self, spectral_data):
        """Full reconstruction should have near-zero error."""
        n_samples, n_features = spectral_data.shape
        max_components = min(n_samples, n_features)

        pca = PCAPipeline(n_components=max_components)
        result = pca.fit(spectral_data)

        _, rmse = result.reconstruction_error(spectral_data)
        # Allow small numerical error from floating point operations
        assert rmse < 1e-5

    def test_per_sample_error_shape(self, spectral_data):
        """Per-sample error should have correct shape."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        per_sample, _ = result.reconstruction_error(spectral_data)
        assert per_sample.shape == (spectral_data.shape[0],)


class TestDenoising:
    """Tests for spectral denoising."""

    def test_denoise_reduces_noise(self, noisy_spectral_data):
        """Denoising should reduce RMSE vs clean signal."""
        noisy, clean = noisy_spectral_data

        denoised, _ = denoise_spectra(noisy, n_components=3)

        # Error vs clean should be lower after denoising
        rmse_before = np.sqrt(np.mean((noisy - clean) ** 2))
        rmse_after = np.sqrt(np.mean((denoised - clean) ** 2))

        assert rmse_after < rmse_before

    def test_denoise_preserves_shape(self, spectral_data):
        """Denoised output should have same shape as input."""
        denoised, _ = denoise_spectra(spectral_data, n_components=5)
        assert denoised.shape == spectral_data.shape


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_fit_pca_function(self, spectral_data):
        """fit_pca convenience function should work."""
        result = fit_pca(spectral_data, n_components=5)

        assert isinstance(result, PCAResult)
        assert result.n_components == 5

    def test_explained_variance_curve(self, spectral_data):
        """explained_variance_curve should return valid data."""
        cumvar, var = explained_variance_curve(spectral_data)

        assert len(cumvar) == len(var)
        assert cumvar[-1] >= cumvar[0]
        np.testing.assert_allclose(np.cumsum(var), cumvar, rtol=1e-10)


# --- Edge Cases and Error Handling ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_feature_data(self):
        """Should handle single feature data."""
        data = np.random.randn(50, 1)
        pca = PCAPipeline(n_components=1)
        result = pca.fit(data)

        assert result.n_components == 1
        assert result.n_features == 1

    def test_two_samples(self):
        """Should work with minimum samples."""
        data = np.random.randn(2, 10)
        pca = PCAPipeline(n_components=1)
        result = pca.fit(data)

        assert result.n_samples == 2

    def test_transform_single_sample(self, spectral_data):
        """Transform should work with single sample."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        single_sample = spectral_data[0]
        scores = result.transform(single_sample)

        assert scores.shape == (1, 5)

    def test_invalid_n_components_negative(self):
        """Should raise on negative n_components."""
        with pytest.raises(ValueError, match="must be >= 1"):
            pca = PCAPipeline(n_components=-1)
            pca.fit(np.random.randn(10, 5))

    def test_invalid_n_components_too_large(self):
        """Should raise when n_components exceeds max."""
        data = np.random.randn(10, 5)
        pca = PCAPipeline(n_components=20)

        with pytest.raises(ValueError, match="exceeds max"):
            pca.fit(data)

    def test_invalid_n_components_float_out_of_range(self):
        """Should raise for float n_components outside (0, 1)."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            pca = PCAPipeline(n_components=1.5)
            pca.fit(np.random.randn(10, 5))

    def test_invalid_variance_threshold(self, spectral_data):
        """Should raise for invalid variance threshold."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        with pytest.raises(ValueError, match="must be in"):
            result.n_components_for_variance(0)

        with pytest.raises(ValueError, match="must be in"):
            result.n_components_for_variance(1.5)

    def test_feature_mismatch_in_transform(self, spectral_data):
        """Transform should raise on feature count mismatch."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        wrong_features = np.random.randn(10, spectral_data.shape[1] + 5)
        with pytest.raises(ValueError, match="Feature count mismatch"):
            result.transform(wrong_features)

    def test_transform_before_fit_raises(self):
        """Transform before fit should raise."""
        pca = PCAPipeline(n_components=5)

        with pytest.raises(RuntimeError, match="Must call fit"):
            pca.transform(np.random.randn(10, 5))

    def test_1d_input_raises(self):
        """1D input to fit should raise."""
        pca = PCAPipeline(n_components=5)

        with pytest.raises(ValueError, match="Expected 2D"):
            pca.fit(np.random.randn(10))

    def test_single_sample_raises(self):
        """Single sample should raise (need at least 2 for variance)."""
        pca = PCAPipeline(n_components=1)

        with pytest.raises(ValueError, match="at least 2 samples"):
            pca.fit(np.random.randn(1, 10))


class TestPCAResult:
    """Tests for PCAResult methods and properties."""

    def test_scree_data(self, spectral_data):
        """scree_data should return plotting data."""
        pca = PCAPipeline(n_components=10)
        result = pca.fit(spectral_data)

        indices, var_ratio, cum_ratio = result.scree_data()

        assert len(indices) == 10
        assert indices[0] == 1  # 1-based indexing
        assert len(var_ratio) == 10
        assert len(cum_ratio) == 10
        np.testing.assert_allclose(np.cumsum(var_ratio), cum_ratio, rtol=1e-10)

    def test_summary_string(self, spectral_data):
        """summary() should return valid string."""
        pca = PCAPipeline(n_components=5)
        result = pca.fit(spectral_data)

        summary = result.summary()

        assert isinstance(summary, str)
        assert "PCA Results" in summary
        assert "PC1" in summary

    def test_reconstruction_error_with_fewer_components(self, spectral_data):
        """Should compute error with subset of components."""
        pca = PCAPipeline(n_components=10)
        result = pca.fit(spectral_data)

        # Error with 3 components should be computable
        _, rmse_3 = result.reconstruction_error(spectral_data, n_components=3)
        _, rmse_10 = result.reconstruction_error(spectral_data, n_components=10)

        # More components = less error
        assert rmse_3 >= rmse_10


# --- JAX Backend Tests ---


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestJAXBackend:
    """Tests for JAX backend functionality."""

    def test_jax_fit_produces_same_result(self, spectral_data):
        """JAX and NumPy backends should produce similar results."""
        pca_np = PCAPipeline(n_components=5, use_jax=False)
        result_np = pca_np.fit(spectral_data)

        pca_jax = PCAPipeline(n_components=5, use_jax=True)
        result_jax = pca_jax.fit(spectral_data)

        # Components may have sign flips, so compare absolute values
        # Use looser tolerance due to float32 vs float64 differences
        np.testing.assert_allclose(
            np.abs(result_np.components),
            np.abs(result_jax.components),
            rtol=1e-3,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            result_np.explained_variance_ratio,
            result_jax.explained_variance_ratio,
            rtol=1e-3,
        )

    def test_jax_transform_produces_same_result(self, spectral_data):
        """JAX transform should match NumPy transform."""
        pca_np = PCAPipeline(n_components=5, use_jax=False)
        pca_jax = PCAPipeline(n_components=5, use_jax=True)

        result_np = pca_np.fit(spectral_data)
        result_jax = pca_jax.fit(spectral_data)

        scores_np = result_np.transform(spectral_data)
        scores_jax = result_jax.transform(spectral_data)

        # Allow sign flips in components and looser tolerance for float32/64 differences
        np.testing.assert_allclose(
            np.abs(scores_np),
            np.abs(scores_jax),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_jax_accelerated_functions(self, spectral_data):
        """JAX-specific functions should work."""
        from cflibs.inversion.pca import (
            pca_transform_jax,
            pca_inverse_transform_jax,
            pca_reconstruction_error_jax,
        )
        import jax.numpy as jnp

        pca = PCAPipeline(n_components=5, use_jax=True)
        result = pca.fit(spectral_data)

        # Convert to JAX arrays
        X_jax = jnp.array(spectral_data)
        components_jax = jnp.array(result.components)
        mean_jax = jnp.array(result.mean)

        # Test transform
        scores = pca_transform_jax(X_jax, components_jax, mean_jax)
        assert scores.shape == (spectral_data.shape[0], 5)

        # Test inverse transform
        reconstructed = pca_inverse_transform_jax(scores, components_jax, mean_jax)
        assert reconstructed.shape == spectral_data.shape

        # Test reconstruction error
        errors = pca_reconstruction_error_jax(X_jax, components_jax, mean_jax)
        assert errors.shape == (spectral_data.shape[0],)


class TestNoJAXBackend:
    """Tests for behavior when JAX is requested but not available."""

    def test_jax_requested_without_jax_raises(self, monkeypatch):
        """Should raise if JAX requested but not available."""
        # Temporarily pretend JAX isn't available.
        # Must patch the canonical module where PCAPipeline reads HAS_JAX,
        # not the backward-compat shim (cflibs.inversion.pca).
        import cflibs.inversion.common.pca as pca_module

        monkeypatch.setattr(pca_module, "HAS_JAX", False)

        with pytest.raises(ImportError, match="JAX requested"):
            PCAPipeline(n_components=5, use_jax=True)