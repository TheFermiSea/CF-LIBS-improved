"""GPU configuration utilities for JAX-based CF-LIBS computation.

Provides helpers for configuring JAX GPU backends (CUDA) with appropriate
settings for plasma-physics computations that require float64 precision.

CPU test isolation is preserved: ``conftest.py`` sets ``JAX_PLATFORMS=cpu``
which takes precedence over any GPU configuration done here.
"""

import os
from dataclasses import dataclass
from typing import Optional

from cflibs.core.logging_config import get_logger

logger = get_logger("hpc.gpu_config")


@dataclass
class GPUInfo:
    """Summary of the active GPU device.

    Attributes
    ----------
    device_id : int
        CUDA device index.
    name : str
        Device name (e.g. ``"NVIDIA A100"``).
    memory_bytes : int
        Total device memory in bytes (0 if unknown).
    platform : str
        JAX platform string (``"gpu"``, ``"cpu"``, etc.).
    """

    device_id: int
    name: str
    memory_bytes: int
    platform: str


def configure_gpu(
    device_id: int = 0,
    enable_x64: bool = True,
    memory_fraction: float = 0.9,
) -> Optional[GPUInfo]:
    """Configure JAX for GPU computation.

    Must be called **before** any JAX computation is triggered
    (i.e. before importing modules that create JAX arrays at import time).

    Parameters
    ----------
    device_id : int
        CUDA device index (default: 0).
    enable_x64 : bool
        Enable float64 precision — required for plasma-physics accuracy
        (default: True).
    memory_fraction : float
        Fraction of GPU memory JAX is allowed to pre-allocate
        (default: 0.9).

    Returns
    -------
    GPUInfo or None
        Device information if a GPU was configured, ``None`` if GPU is
        unavailable (falls back to CPU silently).
    """
    # Set environment variables *before* JAX init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    try:
        import jax

        if enable_x64:
            jax.config.update("jax_enable_x64", True)

        devices = jax.devices()
        if not devices:
            logger.warning("No JAX devices found; falling back to CPU")
            return None

        dev = devices[0]
        platform = dev.platform

        if platform == "cpu":
            logger.info("JAX using CPU (no GPU detected or JAX_PLATFORMS=cpu)")
            return GPUInfo(
                device_id=0,
                name="CPU",
                memory_bytes=0,
                platform="cpu",
            )

        # Attempt to read memory info (CUDA only)
        mem_bytes = 0
        try:
            mem_stats = dev.memory_stats()
            if mem_stats and "bytes_limit" in mem_stats:
                mem_bytes = int(mem_stats["bytes_limit"])
        except Exception:
            pass

        info = GPUInfo(
            device_id=device_id,
            name=str(getattr(dev, "device_kind", "unknown")),
            memory_bytes=mem_bytes,
            platform=platform,
        )

        logger.info(
            f"GPU configured: device={device_id}, name={info.name}, "
            f"memory={mem_bytes / 1e9:.1f} GB, x64={enable_x64}"
        )
        return info

    except ImportError:
        logger.warning("JAX not installed; GPU configuration skipped")
        return None
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")
        return None


def pin_to_device(local_rank: int) -> Optional[GPUInfo]:
    """Pin the current process to a specific GPU by local MPI rank.

    Useful in multi-GPU nodes where each MPI rank should use a
    different GPU.  Computes ``device_id = local_rank % num_gpus``.

    Parameters
    ----------
    local_rank : int
        Local MPI rank on this node (e.g. from ``MPI.COMM_WORLD.Get_rank()``
        modulo processes-per-node, or ``SLURM_LOCALID``).

    Returns
    -------
    GPUInfo or None
        Device info for the pinned GPU, or ``None`` on failure.
    """
    try:
        # Derive device_id from SLURM env or local_rank without importing JAX,
        # so CUDA_VISIBLE_DEVICES is set before JAX initialises.
        gpus_on_node = os.environ.get("SLURM_GPUS_ON_NODE")
        if gpus_on_node is not None:
            device_id = local_rank % int(gpus_on_node)
        else:
            # Best-effort: assume one GPU per rank
            device_id = local_rank

        return configure_gpu(device_id=device_id, enable_x64=True)
    except Exception as e:
        logger.warning(f"Device pinning failed for rank {local_rank}: {e}")
        return configure_gpu(device_id=0, enable_x64=True)
