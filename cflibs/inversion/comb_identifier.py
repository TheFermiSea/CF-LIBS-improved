"""Backward-compatible shim — use cflibs.inversion.identify.comb."""

from cflibs.inversion.identify.comb import *  # noqa: F401,F403

# Default parameters for trace-element optimized identification
DEFAULT_MIN_CORRELATION = 0.05
DEFAULT_MIN_ACTIVE_TEETH = 2
