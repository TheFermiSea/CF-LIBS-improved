import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

# Mocking dependencies that might not be present in the test environment
import sys
mock_jax = MagicMock()
mock_numpyro = MagicMock()
sys.modules["jax"] = mock_jax
sys.modules["numpyro"] = mock_numpyro

from cflibs.benchmark.unified import (
    _fit_bayesian_pipeline,
    UnifiedBenchmarkContext,
    BenchmarkSpectrum,
    InstrumentalConditions,
    SampleMetadata,
)

class TestBayesianCompositionGPUPinning(unittest.TestCase):
    def setUp(self):
        self.context = UnifiedBenchmarkContext(db_path=Path("/tmp/fake.db"))
        self.config = {
            "num_warmup": 10,
            "num_samples": 20,
            "num_chains": 1,
            "seed": 42,
        }
        self.spectrum = BenchmarkSpectrum(
            spectrum_id="test_spec",
            wavelength_nm=np.linspace(200, 400, 100),
            intensity=np.random.rand(100),
            true_composition={"Fe": 1.0},
            conditions=InstrumentalConditions(
                laser_wavelength_nm=1064.0,
                laser_energy_mj=0.0,
                spectral_range_nm=(200.0, 400.0),
                spectral_resolution_nm=0.1,
                spectrometer_type="test",
                detector_type="test",
                atmosphere="air",
            ),
            metadata=SampleMetadata(
                sample_id="test_sample",
                sample_type="test",
                matrix_type="test",
                provenance="test",
            ),
            dataset_id="test_ds",
            group_id="test_group",
            specimen_id="test_specimen",
            instrument_id="test_inst",
            truth_type="test",
            rp_estimate=1000.0,
            label_cardinality=1,
            spectrum_kind="test",
        )

    @patch("cflibs.benchmark.unified.HAS_JAX", True)
    @patch("cflibs.benchmark.unified.HAS_NUMPYRO", True)
    @patch("cflibs.inversion.solve.bayesian.BayesianForwardModel")
    @patch("cflibs.inversion.solve.bayesian.MCMCSampler")
    def test_bayesian_pipeline_initialization(self, mock_sampler_cls, mock_model_cls):
        # Verify that _fit_bayesian_pipeline calls the expected JAX/NumPyro config functions
        with patch("jax.config.update") as mock_jax_update, \
             patch("numpyro.set_platform") as mock_set_platform:
            
            _fit_bayesian_pipeline(self.context, [], self.config)
            
            mock_set_platform.assert_called_with("gpu")
            mock_jax_update.assert_called_with("jax_enable_x64", True)

    @patch("cflibs.benchmark.unified.HAS_JAX", True)
    @patch("cflibs.benchmark.unified.HAS_NUMPYRO", True)
    @patch("cflibs.inversion.solve.bayesian.BayesianForwardModel")
    @patch("cflibs.inversion.solve.bayesian.MCMCSampler")
    def test_predictor_gpu_context(self, mock_sampler_cls, mock_model_cls):
        # Verify that the predictor attempts to use jax.default_device if a GPU is found
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.run.return_value = MagicMock(
            concentrations_mean={"Fe": 1.0},
            samples={},
            T_K_mean=10000.0,
            n_samples=20,
            n_chains=1,
            n_warmup=10,
            convergence_status=MagicMock(value="converged")
        )
        
        predictor = _fit_bayesian_pipeline(self.context, [], self.config)
        
        with patch("jax.devices") as mock_devices, \
             patch("jax.default_device") as mock_default_device:
            
            mock_gpu = MagicMock()
            mock_devices.return_value = [mock_gpu]
            
            predictor(self.spectrum, ["Fe"], None)
            
            mock_devices.assert_called_with("gpu")
            mock_default_device.assert_called_with(mock_gpu)
            mock_sampler.run.assert_called()

if __name__ == "__main__":
    unittest.main()
