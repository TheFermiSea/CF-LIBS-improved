"""
HPC utilities for CF-LIBS computation.

Provides SLURM job management, GPU configuration, and distributed MCMC
sampling for HPC cluster deployment.
"""

from cflibs.hpc.slurm import (
    ArrayJobConfig,
    SlurmJobConfig,
    SlurmJobManager,
    SlurmJobState,
    SlurmJobStatus,
    create_distributed_mcmc_job,
    generate_distributed_mcmc_script,
)
from cflibs.hpc.gpu_config import GPUInfo, configure_gpu, pin_to_device

__all__ = [
    "SlurmJobConfig",
    "ArrayJobConfig",
    "SlurmJobStatus",
    "SlurmJobState",
    "SlurmJobManager",
    "create_distributed_mcmc_job",
    "generate_distributed_mcmc_script",
    "GPUInfo",
    "configure_gpu",
    "pin_to_device",
]
