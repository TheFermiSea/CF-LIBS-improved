import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from cflibs.benchmark.unified import _fit_bayesian_pipeline, UnifiedBenchmarkContext
from cflibs.benchmark.dataset import BenchmarkSpectrum, InstrumentalConditions, SampleMetadata

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=UnifiedBenchmarkContext)
    ctx.db_path = "/tmp/mock.db"
    return ctx

@pytest.fixture
def mock_spectrum():
    wl = np.linspace(200, 400, 500)
    return BenchmarkSpectrum(
        spectrum_id="test_spec",
        wavelength_nm=wl,
        intensity=np.ones_like(wl),
        true_composition={"Fe": 1.0},
        conditions=InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=50.0,
            spectral_range_nm=(200, 400),
            spectral_resolution_nm=0.1,
            spectrometer_type="mock",
            detector_type="mock",
            atmosphere="air",
        ),
        metadata=SampleMetadata(
            sample_id="mock_sample",
            sample_type="synthetic",
            matrix_type="geological",
            provenance="mock",
        ),
        dataset_id="mock_ds",
        group_id="mock_group",
        specimen_id="mock_specimen",
        instrument_id="mock_inst",
        truth_type="synthetic",
    )

def test_bayesian_pipeline_config_applied(mock_context):
    """Verify that _fit_bayesian_pipeline applies JAX/NumPyro configs.

    CF-LIBS-improved-xsuj (PR #186) replaced the prior
    ``numpyro.set_host_device_count(num_chains)`` call -- which had no
    effect because it was invoked after JAX initialized -- with the
    ``MCMCSampler(chain_method="vectorized")`` default, which batches
    chains into a single JIT'd kernel on the current device. This test
    therefore no longer asserts that ``set_host_device_count`` is
    called; it asserts the surviving config touches and pins the
    documented behaviour change as a regression guard.
    """
    with patch("jax.config.update") as mock_jax_update, \
         patch("numpyro.set_platform") as mock_set_platform:

        # Simulate CUDA environment
        with patch.dict(os.environ, {"JAX_PLATFORMS": "cuda"}):
            _fit_bayesian_pipeline(mock_context, [], {"num_chains": 2})

            # Check JAX configs
            mock_jax_update.assert_any_call("jax_enable_x64", True)
            mock_jax_update.assert_any_call("jax_platform_name", "gpu")

            # Check NumPyro configs
            mock_set_platform.assert_called_with("gpu")

    # Regression guard for the xsuj change: confirm
    # ``MCMCSampler.run``'s default ``chain_method`` is ``"vectorized"``
    # so the cluster benchmark gets multi-chain NUTS on a single GPU
    # without ``set_host_device_count`` (which previously was the only
    # NumPyro device-count knob in this code path).
    import inspect
    from cflibs.inversion.solve.bayesian.samplers import MCMCSampler

    sig = inspect.signature(MCMCSampler.run)
    assert sig.parameters["chain_method"].default == "vectorized", (
        "MCMCSampler.run default chain_method must remain 'vectorized' "
        "for single-GPU multi-chain NUTS (CF-LIBS-improved-xsuj)"
    )

def test_bayesian_predictor_caching(mock_context, mock_spectrum):
    """Verify that the Bayesian predictor caches its sampler across spectra."""
    config = {"num_warmup": 10, "num_samples": 10, "pixels": 512}
    
    with patch("cflibs.inversion.solve.bayesian.BayesianForwardModel") as mock_model_cls, \
         patch("cflibs.inversion.solve.bayesian.MCMCSampler") as mock_sampler_cls:
        
        # Mock sampler instance
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.run.return_value = MagicMock()
        mock_model = mock_model_cls.return_value
        mock_model.wavelength = np.linspace(200, 400, 512)
        
        predictor = _fit_bayesian_pipeline(mock_context, [], config)
        
        # First call: should instantiate model and sampler
        predictor(mock_spectrum, ["Fe"], None)
        assert mock_model_cls.call_count == 1
        assert mock_sampler_cls.call_count == 1
        
        # Second call with same elements: should hit cache
        predictor(mock_spectrum, ["Fe"], None)
        assert mock_model_cls.call_count == 1
        assert mock_sampler_cls.call_count == 1
        
        # Third call with different elements: should miss cache
        predictor(mock_spectrum, ["Fe", "Mg"], None)
        assert mock_model_cls.call_count == 2
        assert mock_sampler_cls.call_count == 2
