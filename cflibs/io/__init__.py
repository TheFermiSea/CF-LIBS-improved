"""
Input/output utilities.

This module provides:
- Standardized file formats for spectra (load/save)
- Export tools for analysis results (CSV, HDF5, JSON)
"""

from cflibs.io.spectrum import load_spectrum, save_spectrum
from cflibs.io.exporters import (
    Exporter,
    CSVExporter,
    HDF5Exporter,
    JSONExporter,
    ExportMetadata,
    create_exporter,
    export_to_csv,
    export_to_json,
)
from cflibs.io.opc import load_opc_calibration, save_opc_calibration

__all__ = [
    # Spectrum I/O
    "load_spectrum",
    "save_spectrum",
    # Exporters
    "Exporter",
    "CSVExporter",
    "HDF5Exporter",
    "JSONExporter",
    "ExportMetadata",
    "create_exporter",
    "export_to_csv",
    "export_to_json",
    # OPC calibration persistence
    "load_opc_calibration",
    "save_opc_calibration",
]
