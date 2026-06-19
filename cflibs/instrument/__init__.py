"""
Instrument response modeling.

This module provides:
- Instrument response functions
- Instrument-function convolution (fixed FWHM / resolving-power modes)
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
