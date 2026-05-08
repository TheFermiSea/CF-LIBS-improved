"""
Instrument response and detector modeling.

This module provides:
- Instrument response functions
- Detector models
- Wavelength calibration and rebinning tools
- Echellogram extraction for 2D spectral images
"""

from cflibs.instrument.model import InstrumentModel, InstrumentModelJax
from cflibs.instrument.convolution import apply_instrument_function, apply_instrument_function_jax
from cflibs.instrument.echelle import EchelleExtractor

__all__ = [
    "InstrumentModel",
    "InstrumentModelJax",
    "apply_instrument_function",
    "apply_instrument_function_jax",
    "EchelleExtractor",
]
