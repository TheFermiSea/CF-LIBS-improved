"""
CF-LIBS: Computational Framework for Laser-Induced Breakdown Spectroscopy

A production-grade Python library for forward modeling, inversion, and analysis
of LIBS plasmas with emphasis on rigorous physics, high-performance numerics,
and reproducible workflows.
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "TheFermiSea"

__all__ = [
    "constants",
    "units",
]


def __getattr__(name: str):
    """Lazy-load lightweight convenience exports."""
    if name in {"constants", "units"}:
        module = import_module(f"cflibs.core.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
