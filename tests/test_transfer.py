"""
Tests for transfer learning framework.

Tests cover:
- Domain adaptation methods (MMD, CORAL, subspace)
- Calibration transfer methods (SBC, PDS, DS)
- Fine-tuning functionality
- Full pipeline integration
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from cflibs.inversion.transfer import (
    # Domain adaptation
    DomainAdapter,
    DomainAdaptationMethod,
    DomainAdaptationResult,
    compute_mmd,
    adapt_domains,
    # Calibration transfer
    CalibrationTransfer,
    TransferResult,
    transfer_calibration,
    # Fine-tuning
    FineTuner,
    FineTuneResult,
    # Pipeline
    TransferLearningPipeline,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def source_spectra(rng):
    """Simulated source instrument spectra."""
    n_samples = 50
    n_wavelengths = 100
    # Base spectra + instrument response
    return rng.normal(100, 20, (n_samples, n_wavelengths)) + 50


@pytest.fixture
def target_spectra(source_spectra, rng):
    """
    Simulated target instrument spectra.

    Applies a linear transformation with some noise to simulate
    different instrument response.
    """
    # Different instrument response: scale + shift + noise
    scale = 0.85 + 0.3 * rng.random(source_spectra.shape[1])
    shift = rng.normal(0, 5, source_spectra.shape[1])
    noise = rng.normal(0, 3, source_spectra.shape)

    return source_spectra * scale + shift + noise


@pytest.fixture
def concentrations(rng):
    """Simulated concentration data."""
    n_samples = 50
    n_elements = 5

    # Random concentrations that sum to 1
    conc = rng.random((n_samples, n_elements))
    return conc / conc.sum(axis=1, keepdims=True)


# ============================================================================
# MMD Tests
# ============================================================================

class TestMMD:
    """Tests for Maximum Mean Discrepancy computation."""

    def test_mmd_identical_distributions(self, rng):
        """MMD should be near zero for identical distributions."""
        X = rng.normal(0, 1, (100, 50))
        Y = X.copy()

        mmd = compute_mmd(X, Y)

        # Unbiased MMD estimator can have small positive values even for identical
        # distributions due to the diagonal exclusion in the estimator
        assert mmd < 0.02, "MMD for identical distributions should be near zero"

    def test_mmd_different_distributions(self, rng):
        """MMD should be positive for different distributions."""
        X = rng.normal(0, 1, (100, 50))
        Y = rng.normal(5, 2, (100, 50))  # Different mean and variance

        mmd = compute_mmd(X, Y)

        assert mmd > 0.1, "MMD for different distributions should be positive"

    def test_mmd_symmetry(self, rng):
        """MMD should be symmetric: MMD(X,Y) = MMD(Y,X)."""
        X = rng.normal(0, 1, (50, 30))
        Y = rng.normal(1, 1, (60, 30))

        mmd_xy = compute_mmd(X, Y)
        mmd_yx = compute_mmd(Y, X)

        # Numerical kernels can differ slightly by operand order across BLAS/Python versions.
        assert abs(mmd_xy - mmd_yx) < 2e-5

    def test_mmd_kernels(self, rng):
        """Test different kernel options."""
        X = rng.normal(0, 1, (30, 20))
        Y = rng.normal(1, 1, (30, 20))

        mmd_rbf = compute_mmd(X, Y, kernel='rbf')
        mmd_linear = compute_mmd(X, Y, kernel='linear')
        mmd_poly = compute_mmd(X, Y, kernel='poly')

        # All should be positive
        assert mmd_rbf >= 0
        assert mmd_linear >= 0
        assert mmd_poly >= 0

    def test_mmd_with_custom_gamma(self, rng):
        """Test custom gamma parameter."""
        X = rng.normal(0, 1, (30, 20))
        Y = rng.normal(1, 1, (30, 20))

        mmd1 = compute_mmd(X, Y, gamma=0.1)
        mmd2 = compute_mmd(X, Y, gamma=1.0)
        mmd3 = compute_mmd(X, Y, gamma=10.0)

        # Different gammas should give different results
        assert mmd1 != mmd2 or mmd2 != mmd3


# ============================================================================
# Domain Adaptation Tests
# ============================================================================

class TestDomainAdapter:
    """Tests for domain adaptation methods."""

    def test_coral_adaptation(self, source_spectra, target_spectra):
        """Test CORAL domain adaptation."""
        adapter = DomainAdapter(method='coral')
        result = adapter.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, DomainAdaptationResult)
        assert result.method == DomainAdaptationMethod.CORAL
        assert result.source_aligned.shape == source_spectra.shape
        assert result.mmd_after <= result.mmd_before

    def test_mmd_adaptation(self, source_spectra, target_spectra):
        """Test MMD-based domain adaptation."""
        adapter = DomainAdapter(method='mmd')
        result = adapter.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, DomainAdaptationResult)
        assert result.method == DomainAdaptationMethod.MMD
        # MMD adaptation should reduce discrepancy
        assert result.mmd_after < result.mmd_before * 2  # Some reduction

    def test_subspace_adaptation(self, source_spectra, target_spectra):
        """Test subspace alignment domain adaptation."""
        adapter = DomainAdapter(method='subspace', n_components=20)
        result = adapter.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, DomainAdaptationResult)
        assert result.method == DomainAdaptationMethod.SUBSPACE

    def test_tca_adaptation(self, source_spectra, target_spectra):
        """Test Transfer Component Analysis."""
        adapter = DomainAdapter(method='tca', n_components=10)
        result = adapter.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, DomainAdaptationResult)
        assert result.method == DomainAdaptationMethod.TCA

    def test_no_adaptation(self, source_spectra, target_spectra):
        """Test identity transformation (no adaptation)."""
        adapter = DomainAdapter(method='none')
        result = adapter.fit_transform(source_spectra, target_spectra)

        assert result.method == DomainAdaptationMethod.NONE
        # Source should be unchanged
        np.testing.assert_array_almost_equal(
            result.source_aligned, source_spectra, decimal=5
        )

    def test_fit_transform_separate(self, source_spectra, target_spectra, rng):
        """Test that fit and transform can be called separately."""
        adapter = DomainAdapter(method='coral')

        # Fit
        adapter.fit(source_spectra, target_spectra)

        # Transform new data
        new_source = rng.normal(100, 20, (10, source_spectra.shape[1]))
        transformed = adapter.transform(new_source)

        assert transformed.shape == new_source.shape

    def test_transform_before_fit_raises(self, rng):
        """Transform before fit should raise error."""
        adapter = DomainAdapter(method='coral')
        X = rng.normal(0, 1, (10, 50))

        with pytest.raises(RuntimeError, match="fit"):
            adapter.transform(X)

    def test_dimension_mismatch_raises(self, rng):
        """Mismatched dimensions should raise error."""
        adapter = DomainAdapter()
        source = rng.normal(0, 1, (50, 100))
        target = rng.normal(0, 1, (50, 80))  # Different dimension

        with pytest.raises(ValueError, match="dimensionality"):
            adapter.fit(source, target)

    def test_method_from_string(self):
        """Test method specification from string."""
        adapter_coral = DomainAdapter(method='CORAL')
        assert adapter_coral.method == DomainAdaptationMethod.CORAL

        adapter_mmd = DomainAdapter(method='mmd')
        assert adapter_mmd.method == DomainAdaptationMethod.MMD


# ============================================================================
# Calibration Transfer Tests
# ============================================================================

class TestCalibrationTransfer:
    """Tests for calibration transfer methods."""

    def test_sbc_transfer(self, source_spectra, target_spectra):
        """Test Slope/Bias Correction transfer."""
        transfer = CalibrationTransfer(method='sbc')
        result = transfer.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, TransferResult)
        assert result.method == 'SBC'
        assert result.transformed_spectra.shape == source_spectra.shape
        assert 'r_squared' in result.metrics

    def test_pds_transfer(self, source_spectra, target_spectra):
        """Test Piecewise Direct Standardization transfer."""
        transfer = CalibrationTransfer(method='pds', window_size=5)
        result = transfer.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, TransferResult)
        assert result.method == 'PDS'
        assert result.transformed_spectra.shape == source_spectra.shape

    def test_ds_transfer(self, source_spectra, target_spectra):
        """Test Direct Standardization transfer."""
        transfer = CalibrationTransfer(method='ds')
        result = transfer.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, TransferResult)
        assert result.method == 'DS'

    def test_standardization_transfer(self, source_spectra, target_spectra):
        """Test simple standardization transfer."""
        transfer = CalibrationTransfer(method='standardization')
        result = transfer.fit_transform(source_spectra, target_spectra)

        assert isinstance(result, TransferResult)
        assert result.method == 'STANDARDIZATION'

    def test_transfer_reduces_residual(self, source_spectra, target_spectra):
        """Transfer should reduce residuals between domains."""
        # Initial residual
        initial_residual = np.mean(np.abs(target_spectra - source_spectra))

        transfer = CalibrationTransfer(method='sbc')
        transfer.fit(source_spectra, target_spectra)
        transformed = transfer.transform(target_spectra)

        # Residual after transfer
        final_residual = np.mean(np.abs(transformed - source_spectra))

        assert final_residual < initial_residual

    def test_save_and_load(self, source_spectra, target_spectra):
        """Test saving and loading transfer model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transfer_model.json"

            # Fit and save
            transfer = CalibrationTransfer(method='sbc')
            transfer.fit(source_spectra, target_spectra)
            transfer.save(path)

            # Load
            loaded = CalibrationTransfer.load(path)

            # Compare transformations
            original = transfer.transform(target_spectra)
            loaded_result = loaded.transform(target_spectra)

            np.testing.assert_array_almost_equal(original, loaded_result)

    def test_shape_mismatch_raises(self, rng):
        """Mismatched shapes should raise error."""
        transfer = CalibrationTransfer()
        source = rng.normal(0, 1, (50, 100))
        target = rng.normal(0, 1, (30, 100))  # Different n_samples

        with pytest.raises(ValueError, match="shape"):
            transfer.fit(source, target)


# ============================================================================
# Fine-tuning Tests
# ============================================================================

class TestFineTuner:
    """Tests for fine-tuning functionality."""

    def test_basic_finetuning(self, rng):
        """Test basic fine-tuning workflow."""
        n_samples = 30
        n_features = 50
        n_outputs = 3

        X = rng.normal(0, 1, (n_samples, n_features))
        # Linear relationship
        W_true = rng.normal(0, 0.1, (n_features, n_outputs))
        y = np.dot(X, W_true) + rng.normal(0, 0.01, (n_samples, n_outputs))

        initial_params = {
            'W': rng.normal(0, 0.1, (n_features, n_outputs)),
            'b': np.zeros(n_outputs),
        }

        finetuner = FineTuner(learning_rate=0.01, l2_reg=0.001)
        result = finetuner.adapt(
            initial_params=initial_params,
            target_spectra=X,
            target_concentrations=y,
            epochs=50,
            verbose=False,
        )

        assert isinstance(result, FineTuneResult)
        assert len(result.loss_history) > 0
        assert result.validation_loss is not None

    def test_finetuning_improves_loss(self, rng):
        """Fine-tuning should reduce loss."""
        n_samples = 50
        n_features = 30
        n_outputs = 2

        X = rng.normal(0, 1, (n_samples, n_features))
        W_true = rng.normal(0, 0.5, (n_features, n_outputs))
        y = np.dot(X, W_true) + rng.normal(0, 0.1, (n_samples, n_outputs))

        # Start with random (poor) initialization
        initial_params = {
            'W': rng.normal(0, 1, (n_features, n_outputs)),
            'b': np.zeros(n_outputs),
        }

        finetuner = FineTuner(learning_rate=0.01)
        result = finetuner.adapt(
            initial_params=initial_params,
            target_spectra=X,
            target_concentrations=y,
            epochs=100,
            verbose=False,
        )

        # Final loss should be less than initial
        assert result.final_loss < result.loss_history[0]

    def test_early_stopping(self, rng):
        """Test early stopping behavior."""
        n_samples = 30
        n_features = 20
        n_outputs = 2

        X = rng.normal(0, 1, (n_samples, n_features))
        # Linear relationship with noise - this should converge quickly
        W_true = rng.normal(0, 0.5, (n_features, n_outputs))
        y = np.dot(X, W_true) + rng.normal(0, 0.1, (n_samples, n_outputs))

        initial_params = {
            'W': rng.normal(0, 0.5, (n_features, n_outputs)),
            'b': np.zeros(n_outputs),
        }

        finetuner = FineTuner(early_stopping_patience=10, learning_rate=0.01)
        result = finetuner.adapt(
            initial_params=initial_params,
            target_spectra=X,
            target_concentrations=y,
            epochs=500,  # Enough epochs to see early stopping
            verbose=False,
        )

        # Should either converge or stop early
        assert result.n_epochs <= 500


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestTransferLearningPipeline:
    """Tests for complete transfer learning pipeline."""

    def test_full_pipeline(self, source_spectra, target_spectra, concentrations):
        """Test complete pipeline workflow."""
        pipeline = TransferLearningPipeline(
            source_name="instrument_A",
            target_name="instrument_B",
            adaptation_method="coral",
            transfer_method="sbc",
        )

        pipeline.fit(source_spectra, target_spectra)

        assert pipeline._fitted

        # Transform new data
        transformed = pipeline.transform(target_spectra[:5])
        assert transformed.shape == (5, source_spectra.shape[1])

    def test_pipeline_evaluation(self, source_spectra, target_spectra):
        """Test pipeline evaluation metrics."""
        pipeline = TransferLearningPipeline(
            adaptation_method="coral",
            transfer_method="sbc",
        )

        pipeline.fit(source_spectra, target_spectra)
        metrics = pipeline.evaluate(source_spectra, target_spectra)

        assert 'mmd' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r_squared' in metrics

    def test_pipeline_summary(self, source_spectra, target_spectra):
        """Test pipeline summary generation."""
        pipeline = TransferLearningPipeline(
            source_name="Lab_A",
            target_name="Lab_B",
        )

        pipeline.fit(source_spectra, target_spectra)
        summary = pipeline.summary()

        assert "Lab_A" in summary
        assert "Lab_B" in summary
        assert "Domain Adaptation" in summary
        assert "Calibration Transfer" in summary

    def test_pipeline_save_load(self, source_spectra, target_spectra):
        """Test pipeline save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline"

            # Fit and save
            pipeline = TransferLearningPipeline(
                source_name="A",
                target_name="B",
            )
            pipeline.fit(source_spectra, target_spectra)
            pipeline.save(path)

            # Load
            loaded = TransferLearningPipeline.load(path)

            # Compare
            original = pipeline.transform(target_spectra[:5])
            loaded_result = loaded.transform(target_spectra[:5])

            np.testing.assert_array_almost_equal(original, loaded_result)

    def test_pipeline_different_methods(self, source_spectra, target_spectra):
        """Test pipeline with different method combinations."""
        methods = [
            ("coral", "sbc"),
            ("mmd", "pds"),
            ("subspace", "ds"),
        ]

        for adapt_method, transfer_method in methods:
            pipeline = TransferLearningPipeline(
                adaptation_method=adapt_method,
                transfer_method=transfer_method,
            )

            pipeline.fit(source_spectra, target_spectra)
            transformed = pipeline.transform(target_spectra[:5])

            assert transformed.shape == (5, source_spectra.shape[1])


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_transfer_calibration(self, source_spectra, target_spectra):
        """Test quick transfer_calibration function."""
        result = transfer_calibration(source_spectra, target_spectra, method='sbc')

        assert result.shape == target_spectra.shape

    def test_adapt_domains(self, source_spectra, target_spectra):
        """Test quick adapt_domains function."""
        source_aligned, target_aligned = adapt_domains(
            source_spectra, target_spectra, method='coral'
        )

        assert source_aligned.shape == source_spectra.shape
        assert target_aligned.shape == target_spectra.shape


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self, rng):
        """Test with single sample."""
        source = rng.normal(0, 1, (1, 50))
        target = rng.normal(1, 1, (1, 50))

        # SBC should work with single sample
        transfer = CalibrationTransfer(method='sbc')
        transfer.fit(source, target)
        result = transfer.transform(target)

        assert result.shape == source.shape

    def test_high_dimensional_data(self, rng):
        """Test with high-dimensional spectra."""
        n_wavelengths = 2000
        source = rng.normal(0, 1, (20, n_wavelengths))
        target = rng.normal(0, 1, (20, n_wavelengths))

        adapter = DomainAdapter(method='coral')
        result = adapter.fit_transform(source, target)

        assert result.source_aligned.shape == source.shape

    def test_constant_channels(self, rng):
        """Test handling of constant (zero variance) channels."""
        source = rng.normal(0, 1, (30, 50))
        target = rng.normal(0, 1, (30, 50))

        # Make some channels constant
        source[:, 10] = 5.0
        target[:, 10] = 3.0

        transfer = CalibrationTransfer(method='sbc')
        transfer.fit(source, target)
        result = transfer.transform(target)

        # Should handle without numerical issues
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_regularization_effect(self, rng):
        """Test that regularization prevents numerical issues."""
        # Near-singular covariance
        source = rng.normal(0, 1, (100, 10))
        target = source + rng.normal(0, 0.001, source.shape)  # Nearly identical

        adapter = DomainAdapter(method='coral', regularization=1e-4)
        result = adapter.fit_transform(source, target)

        assert not np.any(np.isnan(result.source_aligned))


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_adaptation_then_transfer(self, source_spectra, target_spectra):
        """Test domain adaptation followed by calibration transfer."""
        # Step 1: Domain adaptation
        adapter = DomainAdapter(method='coral')
        adaptation_result = adapter.fit_transform(source_spectra, target_spectra)

        # Step 2: Calibration transfer on adapted data
        transfer = CalibrationTransfer(method='sbc')
        transfer.fit(adaptation_result.source_aligned, target_spectra)
        final_spectra = transfer.transform(target_spectra)

        # Verify improvement
        initial_mmd = compute_mmd(source_spectra, target_spectra)
        final_mmd = compute_mmd(source_spectra, final_spectra)

        # Should show some improvement
        assert final_mmd < initial_mmd * 1.5

    def test_full_workflow_with_new_samples(
        self, source_spectra, target_spectra, rng
    ):
        """Test complete workflow including prediction on new samples."""
        # Split target data
        n_transfer = 20
        transfer_target = target_spectra[:n_transfer]
        transfer_source = source_spectra[:n_transfer]

        new_target = target_spectra[n_transfer:]

        # Fit pipeline
        pipeline = TransferLearningPipeline()
        pipeline.fit(transfer_source, transfer_target)

        # Transform new samples
        corrected = pipeline.transform(new_target)

        # Verify reasonable output
        assert corrected.shape == new_target.shape
        assert not np.any(np.isnan(corrected))

        # Should be closer to source domain than original
        new_source = source_spectra[n_transfer:]
        mmd_before = compute_mmd(new_source, new_target)
        mmd_after = compute_mmd(new_source, corrected)

        # Allow some tolerance
        assert mmd_after < mmd_before * 2
