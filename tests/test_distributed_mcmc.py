"""Tests for HPC distributed MCMC and GPU configuration.

These tests use mocked MPI and do not require actual GPU hardware.
"""

import numpy as np
from unittest.mock import MagicMock, patch

# ============================================================================
# GPU Configuration Tests
# ============================================================================


class TestGPUConfig:
    """Tests for GPU configuration utilities."""

    def test_configure_gpu_sets_env_vars(self):
        """configure_gpu sets CUDA_VISIBLE_DEVICES and memory fraction."""
        import os

        with patch.dict(os.environ, {}, clear=False):
            from cflibs.hpc.gpu_config import configure_gpu

            # Should not crash even without GPU
            configure_gpu(device_id=2, memory_fraction=0.75)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
            assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.75"

    def test_configure_gpu_returns_info_on_cpu(self):
        """On CPU, configure_gpu returns GPUInfo with platform='cpu'."""
        from cflibs.hpc.gpu_config import configure_gpu, GPUInfo

        result = configure_gpu()
        # Will return CPU info on test machines without GPU
        if result is not None:
            assert isinstance(result, GPUInfo)
            assert result.platform in ("cpu", "gpu", "METAL")

    def test_gpu_info_dataclass(self):
        """GPUInfo dataclass works correctly."""
        from cflibs.hpc.gpu_config import GPUInfo

        info = GPUInfo(device_id=0, name="Test GPU", memory_bytes=8_000_000_000, platform="gpu")
        assert info.device_id == 0
        assert info.name == "Test GPU"
        assert info.memory_bytes == 8_000_000_000


# ============================================================================
# Distributed MCMC Config Tests
# ============================================================================


class TestDistributedMCMCConfig:
    """Tests for DistributedMCMCConfig dataclass."""

    def test_default_values(self):
        from cflibs.hpc.distributed_mcmc import DistributedMCMCConfig

        cfg = DistributedMCMCConfig()
        assert cfg.chains_per_rank == 1
        assert cfg.num_warmup == 500
        assert cfg.num_samples == 1000
        assert cfg.use_gpu is False
        assert cfg.target_accept_prob == 0.8

    def test_custom_values(self):
        from cflibs.hpc.distributed_mcmc import DistributedMCMCConfig

        cfg = DistributedMCMCConfig(
            chains_per_rank=2,
            num_warmup=100,
            num_samples=200,
            use_gpu=True,
        )
        assert cfg.chains_per_rank == 2
        assert cfg.use_gpu is True


class TestDistributedMCMCResult:
    """Tests for DistributedMCMCResult dataclass."""

    def test_creation(self):
        from cflibs.hpc.distributed_mcmc import DistributedMCMCResult

        result = DistributedMCMCResult(
            samples={"T_eV": np.array([1.0, 1.1, 1.2])},
            r_hat={"T_eV": 1.001},
            ess={"T_eV": 500.0},
            total_chains=4,
            total_samples=4000,
            rank_chain_counts=[1, 1, 1, 1],
        )
        assert result.total_chains == 4
        assert "T_eV" in result.r_hat
        assert result.r_hat["T_eV"] < 1.01


# ============================================================================
# Distributed MCMC Sampler Tests (Mocked MPI)
# ============================================================================


class TestDistributedMCMCSamplerMocked:
    """Tests with mocked MPI for the distributed sampler."""

    def test_merge_results(self):
        """Test that _merge_results correctly combines samples from ranks."""
        from cflibs.hpc.distributed_mcmc import (
            DistributedMCMCSampler,
            DistributedMCMCConfig,
        )

        # Mock MPI communicator
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 2

        # Mock the forward model and model_fn
        mock_fm = MagicMock()
        mock_model_fn = MagicMock()

        with (
            patch("cflibs.hpc.distributed_mcmc.HAS_MPI", True),
            patch("cflibs.hpc.distributed_mcmc.HAS_NUMPYRO", True),
            patch("cflibs.hpc.distributed_mcmc.HAS_JAX", True),
            patch("cflibs.hpc.distributed_mcmc.MPI", MagicMock()),
        ):

            cfg = DistributedMCMCConfig(chains_per_rank=1, num_samples=100)
            sampler = DistributedMCMCSampler(
                mock_fm,
                mock_model_fn,
                config=cfg,
                comm=mock_comm,
            )

            # Simulate gathered results from 2 ranks
            # Shape: (chains, samples) for each rank
            rank0_samples = {"T_eV": np.random.randn(1, 100)}
            rank1_samples = {"T_eV": np.random.randn(1, 100)}

            result = sampler._merge_results([rank0_samples, rank1_samples], cfg)

            assert result.total_chains == 2
            assert result.rank_chain_counts == [1, 1]
            assert "T_eV" in result.samples
            assert result.samples["T_eV"].shape[0] == 200  # 2 chains × 100 samples


# ============================================================================
# SLURM Integration Tests
# ============================================================================


class TestCreateDistributedMCMCJob:
    """Tests for SLURM script generation."""

    def test_generates_valid_script(self, tmp_path):
        from cflibs.hpc.slurm import create_distributed_mcmc_job

        script_path = str(tmp_path / "run_mcmc.sh")
        content = create_distributed_mcmc_job(
            script_path=script_path,
            db_path="/data/atomic.db",
            elements=["Fe", "Cu"],
            wavelength_range=(200.0, 600.0),
            observed_path="/data/observed.npy",
            output_dir="/results",
            ntasks=8,
            gpus_per_task=1,
            partition="gpu",
            account="myacct",
        )

        assert "#!/bin/bash" in content
        assert "#SBATCH --ntasks=8" in content
        assert "#SBATCH --gpus-per-task=1" in content
        assert "#SBATCH --account=myacct" in content
        assert "JAX_PLATFORMS=gpu" in content
        assert "mpirun -n 8" in content
        assert "DistributedMCMCSampler" in content

        # Verify file was written
        with open(script_path) as f:
            assert f.read() == content

    def test_cpu_only_script(self, tmp_path):
        from cflibs.hpc.slurm import create_distributed_mcmc_job

        script_path = str(tmp_path / "run_mcmc_cpu.sh")
        content = create_distributed_mcmc_job(
            script_path=script_path,
            db_path="/data/atomic.db",
            elements=["Fe"],
            wavelength_range=(300.0, 500.0),
            observed_path="/data/obs.npy",
            ntasks=4,
            gpus_per_task=0,
        )

        assert "JAX_PLATFORMS=cpu" in content
        assert "#SBATCH --gpus" not in content

    def test_with_modules_and_conda(self, tmp_path):
        from cflibs.hpc.slurm import create_distributed_mcmc_job

        script_path = str(tmp_path / "run.sh")
        content = create_distributed_mcmc_job(
            script_path=script_path,
            db_path="/data/atomic.db",
            elements=["Fe"],
            wavelength_range=(300.0, 500.0),
            observed_path="/data/obs.npy",
            ntasks=2,
            modules=["cuda/12.0", "python/3.12"],
            conda_env="cflibs-env",
        )

        assert "module load cuda/12.0" in content
        assert "module load python/3.12" in content
        assert "conda activate cflibs-env" in content
