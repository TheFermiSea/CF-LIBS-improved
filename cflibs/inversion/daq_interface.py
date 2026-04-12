"""PERMANENT shim — external rust-daq plugin depends on this path.

Do NOT remove this file. The rust-daq plugin imports from
cflibs.inversion.daq_interface directly.
"""
from cflibs.inversion.runtime.daq_interface import *  # noqa: F401,F403
