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
]
