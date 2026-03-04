"""Platform-aware JAX configuration for CF-LIBS.

Detects the runtime platform (macOS/Linux) and configures JAX with the
appropriate backend and precision settings.  Must be called **before** any
JAX arrays are created (i.e. before importing ``cflibs.radiation.profiles``).

Key design decisions
--------------------
* **macOS always uses CPU** -- the Metal backend is abandoned (jax-metal last
  release Oct 2024, incompatible with JAX >= 0.6) and does not support
  float64, which CF-LIBS requires for Saha-Boltzmann accuracy.
* **Linux tries CUDA first**, falls back to CPU.
* **float64 is always enabled** -- plasma-physics calculations are not
  accurate in float32 (Saha exponentials, partition-function polynomials,
  Weideman Faddeeva coefficients all require >7 significant digits).
"""

import os
import sys
import platform as _platform
from enum import Enum
from cflibs.core.logging_config import get_logger

logger = get_logger("core.platform_config")


class AcceleratorBackend(Enum):
    """JAX accelerator backend in use.

    Attributes
    ----------
    CPU : str
        CPU-only backend (always available).
    CUDA : str
        NVIDIA CUDA GPU backend (Linux only).
    """

    CPU = "cpu"
    CUDA = "cuda"


def configure_jax(
    enable_x64: bool = True,
    prefer_gpu: bool = True,
) -> AcceleratorBackend:
    """Configure JAX for CF-LIBS computation.

    Must be called **before** any JAX import or computation that creates
    JAX arrays at import time (e.g. ``cflibs.radiation.profiles``).
    If JAX has already been imported, the environment-variable based
    platform selection will not take effect and a warning is logged.

    Parameters
    ----------
    enable_x64 : bool
        Enable float64 precision.  Must be ``True`` for CF-LIBS physics
        correctness (default).
    prefer_gpu : bool
        Attempt GPU acceleration if available (default ``True``).

    Returns
    -------
    AcceleratorBackend
        The backend that was configured.
    """
    system = _platform.system()

    # Warn if JAX was already imported — env var changes won't affect backend.
    # Use .get() to avoid false positives from negative cache entries (None).
    if sys.modules.get("jax") is not None:
        logger.warning(
            "configure_jax() called after JAX was already imported. "
            "JAX_PLATFORMS changes will not take effect. "
            "Call configure_jax() before importing JAX for reliable configuration."
        )

    # On macOS, Metal cannot support float64 — force CPU before JAX init.
    # Unconditional assignment: the Metal backend must never be used.
    if system == "Darwin":
        os.environ["JAX_PLATFORMS"] = "cpu"

    try:
        import jax

        if enable_x64:
            jax.config.update("jax_enable_x64", True)

        if system == "Darwin":
            logger.info(
                "macOS detected: using JAX CPU backend "
                "(Metal lacks float64 support required for CF-LIBS physics)"
            )
            return AcceleratorBackend.CPU

        # On Linux, try CUDA if requested.
        # jax.devices("gpu") raises on failure (never returns []),
        # so the call succeeding means a GPU is available.
        if prefer_gpu and system == "Linux":
            try:
                dev = jax.devices("gpu")[0]
                logger.info(f"CUDA GPU detected: {dev}, x64={enable_x64}")
                return AcceleratorBackend.CUDA
            except Exception:
                pass

        logger.info("Using JAX CPU backend")
        return AcceleratorBackend.CPU

    except ImportError:
        logger.warning("JAX not installed; JAX-accelerated operations unavailable")
        return AcceleratorBackend.CPU
