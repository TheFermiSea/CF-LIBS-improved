"""
Export tools for CF-LIBS analysis results.

This module provides exporters for common data formats:
- CSV: Spectrum data, analysis results with configurable columns
- HDF5: Complete analysis state with metadata and compression
- JSON: Structured results for API consumption

All exporters follow the Exporter ABC interface and support metadata
preservation including timestamps, version info, and analysis parameters.

Example
-------
>>> from cflibs.io.exporters import create_exporter, CSVExporter
>>> result = solver.solve(observations)
>>>
>>> # Export to CSV
>>> exporter = create_exporter("csv")
>>> exporter.export(result, "analysis_results.csv")
>>>
>>> # Export to HDF5 with compression
>>> exporter = create_exporter("hdf5", compression="gzip", compression_opts=4)
>>> exporter.export(result, "analysis_state.h5")
>>>
>>> # Export to JSON for API
>>> exporter = create_exporter("json", indent=2)
>>> exporter.export(result, "results.json")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np

from cflibs.core.logging_config import get_logger

# Import result types for ExportData alias (lazy to avoid circular imports)
from cflibs.inversion.solver import CFLIBSResult
from cflibs.inversion.bayesian import MCMCResult, NestedSamplingResult

logger = get_logger("io.exporters")

# Package version for metadata
try:
    from cflibs import __version__
except ImportError:
    __version__ = "unknown"


# --- Type aliases ---
PathLike = Union[str, Path]
ExportData = Union[CFLIBSResult, MCMCResult, NestedSamplingResult, Dict[str, Any]]


@dataclass
class ExportMetadata:
    """
    Metadata for exported data.

    Attributes
    ----------
    timestamp : str
        ISO format timestamp of export
    version : str
        CF-LIBS package version
    format : str
        Export format (csv, hdf5, json)
    source_type : str
        Type of source data (CFLIBSResult, MCMCResult, etc.)
    parameters : Dict[str, Any]
        Analysis parameters used
    """

    timestamp: str
    version: str
    format: str
    source_type: str
    parameters: Dict[str, Any]

    @classmethod
    def create(
        cls,
        format: str,
        source_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "ExportMetadata":
        """Create metadata with current timestamp and version."""
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            version=__version__,
            format=format,
            source_type=source_type,
            parameters=parameters or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "version": self.version,
            "format": self.format,
            "source_type": self.source_type,
            "parameters": self.parameters,
        }


class Exporter(ABC):
    """
    Abstract base class for data exporters.

    All exporters must implement the export() method to write data
    to the specified format. Exporters should preserve metadata
    and handle numpy arrays appropriately.

    Subclasses
    ----------
    CSVExporter : Export to CSV format
    HDF5Exporter : Export to HDF5 format
    JSONExporter : Export to JSON format
    """

    @abstractmethod
    def export(
        self,
        data: ExportData,
        path: PathLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Export data to file.

        Parameters
        ----------
        data : ExportData
            Data to export (CFLIBSResult, MCMCResult, dict, etc.)
        path : PathLike
            Output file path
        metadata : Dict[str, Any], optional
            Additional metadata to include

        Raises
        ------
        ValueError
            If data format is not supported
        IOError
            If file cannot be written
        """
        pass

    @abstractmethod
    def get_format(self) -> str:
        """Return the export format name (csv, hdf5, json)."""
        pass

    def _normalize_data(self, data: ExportData) -> Dict[str, Any]:
        """
        Normalize input data to dictionary format.

        Parameters
        ----------
        data : ExportData
            Input data (dataclass, dict, etc.)

        Returns
        -------
        Dict[str, Any]
            Normalized dictionary representation
        """
        if isinstance(data, dict):
            return data
        elif is_dataclass(data) and not isinstance(data, type):
            return self._dataclass_to_dict(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data).__name__}")

    def _dataclass_to_dict(self, obj: Any, max_depth: int = 10) -> Any:
        """
        Convert dataclass to dictionary recursively.

        Handles nested dataclasses and numpy arrays. Returns ``str(obj)``
        when ``max_depth`` is exhausted, so the recursion terminates safely
        on cyclic references.
        """
        if max_depth <= 0:
            return str(obj)

        result: Dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = self._convert_value(value, max_depth - 1)
        return result

    def _convert_value(self, value: Any, max_depth: int) -> Any:
        """Convert a single value for serialization."""
        if value is None:
            return None
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, dict):
            return {k: self._convert_value(v, max_depth) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._convert_value(v, max_depth) for v in value]
        elif is_dataclass(value) and not isinstance(value, type):
            return self._dataclass_to_dict(value, max_depth)
        elif hasattr(value, "__dict__") and not callable(value):
            # Handle enum or other objects with __dict__
            if hasattr(value, "value"):
                return value.value
            return str(value)
        else:
            return value

    def _get_source_type(self, data: ExportData) -> str:
        """Get the type name of the source data."""
        if isinstance(data, dict):
            return "dict"
        return type(data).__name__

    # --- Type detection helpers (shared by all exporters) ---

    def _is_cflibs_result(self, data: Dict[str, Any]) -> bool:
        """Check if data looks like CFLIBSResult."""
        return "temperature_K" in data and "concentrations" in data and "converged" in data

    def _is_mcmc_result(self, data: Dict[str, Any]) -> bool:
        """Check if data looks like MCMCResult."""
        return "T_eV_mean" in data and "concentrations_mean" in data and "samples" in data

    def _is_nested_sampling_result(self, data: Dict[str, Any]) -> bool:
        """Check if data looks like NestedSamplingResult."""
        return "log_evidence" in data and "weights" in data

    def _is_spectrum_data(self, data: Dict[str, Any]) -> bool:
        """Check if data is spectrum (wavelength/intensity arrays)."""
        return "wavelength" in data and "intensity" in data


class CSVExporter(Exporter):
    """
    Export analysis results to CSV format.

    Supports exporting CFLIBSResult, MCMCResult, spectrum data, and
    concentration tables with configurable column names and formatting.

    Parameters
    ----------
    columns : List[str], optional
        Column names to export (default: all available)
    delimiter : str
        Field delimiter (default: ",")
    include_header : bool
        Include column header row (default: True)
    include_metadata : bool
        Include metadata as comment lines (default: True)
    float_format : str
        Format string for floating point numbers (default: "%.6g")

    Example
    -------
    >>> exporter = CSVExporter(columns=["element", "concentration", "uncertainty"])
    >>> exporter.export(result, "concentrations.csv")
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        delimiter: str = ",",
        include_header: bool = True,
        include_metadata: bool = True,
        float_format: str = "%.6g",
    ):
        self.columns = columns
        self.delimiter = delimiter
        self.include_header = include_header
        self.include_metadata = include_metadata
        self.float_format = float_format

    def get_format(self) -> str:
        return "csv"

    def export(
        self,
        data: ExportData,
        path: PathLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Export data to CSV file.

        The export format depends on the data type:
        - CFLIBSResult: Concentration table with uncertainties
        - MCMCResult: Summary statistics and concentration table
        - NestedSamplingResult: Summary statistics and evidence
        - dict with 'wavelength'/'intensity': Spectrum data
        - dict: General key-value pairs
        """
        path = Path(path)
        data_dict = self._normalize_data(data)
        source_type = self._get_source_type(data)

        # Create metadata
        export_meta = ExportMetadata.create(
            format="csv",
            source_type=source_type,
            parameters=metadata,
        )

        lines = []

        # Add metadata as comments
        if self.include_metadata:
            lines.append("# CF-LIBS Export")
            lines.append(f"# Timestamp: {export_meta.timestamp}")
            lines.append(f"# Version: {export_meta.version}")
            lines.append(f"# Source: {export_meta.source_type}")
            if metadata:
                for key, value in metadata.items():
                    lines.append(f"# {key}: {value}")
            lines.append("#")

        # Export based on data structure
        # Note: Check nested sampling BEFORE MCMC since nested results
        # may contain MCMC-like fields (T_eV_mean, etc.)
        if self._is_cflibs_result(data_dict):
            lines.extend(self._export_cflibs_result(data_dict))
        elif self._is_nested_sampling_result(data_dict):
            lines.extend(self._export_nested_sampling_result(data_dict))
        elif self._is_mcmc_result(data_dict):
            lines.extend(self._export_mcmc_result(data_dict))
        elif self._is_spectrum_data(data_dict):
            lines.extend(self._export_spectrum(data_dict))
        else:
            lines.extend(self._export_dict(data_dict))

        # Write file
        with open(path, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")

        logger.info(f"Exported {source_type} to CSV: {path}")

    def _export_cflibs_result(self, data: Dict[str, Any]) -> List[str]:
        """Export CFLIBSResult to CSV lines."""
        lines = []

        # Summary section
        lines.append("# Summary")
        if self.include_header:
            lines.append(self.delimiter.join(["parameter", "value", "uncertainty", "unit"]))

        T_K = data.get("temperature_K", 0)
        T_err = data.get("temperature_uncertainty_K", 0)
        lines.append(
            self.delimiter.join(
                ["temperature", self.float_format % T_K, self.float_format % T_err, "K"]
            )
        )

        ne = data.get("electron_density_cm3", 0)
        ne_err = data.get("electron_density_uncertainty_cm3", 0)
        lines.append(
            self.delimiter.join(
                ["electron_density", self.float_format % ne, self.float_format % ne_err, "cm^-3"]
            )
        )

        lines.append(self.delimiter.join(["iterations", str(data.get("iterations", 0)), "", ""]))

        lines.append(self.delimiter.join(["converged", str(data.get("converged", False)), "", ""]))

        lines.append("#")

        # Concentration table
        lines.append("# Concentrations")
        if self.include_header:
            lines.append(self.delimiter.join(["element", "concentration", "uncertainty"]))

        concentrations = data.get("concentrations", {})
        uncertainties = data.get("concentration_uncertainties", {})

        for element in sorted(concentrations.keys()):
            conc = concentrations[element]
            uncert = uncertainties.get(element, 0)
            lines.append(
                self.delimiter.join(
                    [
                        element,
                        self.float_format % conc,
                        self.float_format % uncert,
                    ]
                )
            )

        # Quality metrics
        if "quality_metrics" in data and data["quality_metrics"]:
            lines.append("#")
            lines.append("# Quality Metrics")
            if self.include_header:
                lines.append(self.delimiter.join(["metric", "value"]))
            for metric, value in data["quality_metrics"].items():
                if isinstance(value, float):
                    lines.append(self.delimiter.join([metric, self.float_format % value]))
                else:
                    lines.append(self.delimiter.join([metric, str(value)]))

        return lines

    def _export_mcmc_result(self, data: Dict[str, Any]) -> List[str]:
        """Export MCMCResult to CSV lines."""
        lines = []

        # Summary statistics
        lines.append("# MCMC Summary Statistics")
        if self.include_header:
            lines.append(self.delimiter.join(["parameter", "mean", "std", "q025", "q975"]))

        lines.append(
            self.delimiter.join(
                [
                    "T_eV",
                    self.float_format % data.get("T_eV_mean", 0),
                    self.float_format % data.get("T_eV_std", 0),
                    self.float_format % data.get("T_eV_q025", 0),
                    self.float_format % data.get("T_eV_q975", 0),
                ]
            )
        )

        lines.append(
            self.delimiter.join(
                [
                    "log_ne",
                    self.float_format % data.get("log_ne_mean", 0),
                    self.float_format % data.get("log_ne_std", 0),
                    self.float_format % data.get("log_ne_q025", 0),
                    self.float_format % data.get("log_ne_q975", 0),
                ]
            )
        )

        lines.append("#")

        # Concentration table
        lines.append("# Concentrations")
        if self.include_header:
            lines.append(self.delimiter.join(["element", "mean", "std", "q025", "q975"]))

        conc_mean = data.get("concentrations_mean", {})
        conc_std = data.get("concentrations_std", {})
        conc_q025 = data.get("concentrations_q025", {})
        conc_q975 = data.get("concentrations_q975", {})

        for element in sorted(conc_mean.keys()):
            lines.append(
                self.delimiter.join(
                    [
                        element,
                        self.float_format % conc_mean.get(element, 0),
                        self.float_format % conc_std.get(element, 0),
                        self.float_format % conc_q025.get(element, 0),
                        self.float_format % conc_q975.get(element, 0),
                    ]
                )
            )

        # Convergence diagnostics
        if "r_hat" in data and data["r_hat"]:
            lines.append("#")
            lines.append("# Convergence Diagnostics")
            if self.include_header:
                lines.append(self.delimiter.join(["parameter", "r_hat", "ess"]))
            for param in data["r_hat"]:
                rhat = data["r_hat"].get(param, 0)
                ess = data.get("ess", {}).get(param, 0)
                lines.append(
                    self.delimiter.join(
                        [
                            param,
                            self.float_format % rhat,
                            self.float_format % ess,
                        ]
                    )
                )

        return lines

    def _export_nested_sampling_result(self, data: Dict[str, Any]) -> List[str]:
        """Export NestedSamplingResult to CSV lines."""
        lines = []

        # Evidence
        lines.append("# Model Evidence")
        if self.include_header:
            lines.append(self.delimiter.join(["quantity", "value", "uncertainty"]))

        lines.append(
            self.delimiter.join(
                [
                    "log_evidence",
                    self.float_format % data.get("log_evidence", 0),
                    self.float_format % data.get("log_evidence_err", 0),
                ]
            )
        )

        lines.append(
            self.delimiter.join(
                [
                    "information",
                    self.float_format % data.get("information", 0),
                    "",
                ]
            )
        )

        lines.append("#")

        # Summary statistics
        lines.append("# Summary Statistics")
        if self.include_header:
            lines.append(self.delimiter.join(["parameter", "mean", "std"]))

        lines.append(
            self.delimiter.join(
                [
                    "T_eV",
                    self.float_format % data.get("T_eV_mean", 0),
                    self.float_format % data.get("T_eV_std", 0),
                ]
            )
        )

        lines.append(
            self.delimiter.join(
                [
                    "log_ne",
                    self.float_format % data.get("log_ne_mean", 0),
                    self.float_format % data.get("log_ne_std", 0),
                ]
            )
        )

        lines.append("#")

        # Concentrations
        lines.append("# Concentrations")
        if self.include_header:
            lines.append(self.delimiter.join(["element", "mean", "std"]))

        conc_mean = data.get("concentrations_mean", {})
        conc_std = data.get("concentrations_std", {})

        for element in sorted(conc_mean.keys()):
            lines.append(
                self.delimiter.join(
                    [
                        element,
                        self.float_format % conc_mean.get(element, 0),
                        self.float_format % conc_std.get(element, 0),
                    ]
                )
            )

        return lines

    def _export_spectrum(self, data: Dict[str, Any]) -> List[str]:
        """Export spectrum data to CSV lines."""
        lines = []

        wavelength = np.asarray(data["wavelength"])
        intensity = np.asarray(data["intensity"])

        # Determine available columns
        available_columns = ["wavelength", "intensity"]
        arrays = {"wavelength": wavelength, "intensity": intensity}

        if "uncertainty" in data:
            available_columns.append("uncertainty")
            arrays["uncertainty"] = np.asarray(data["uncertainty"])

        # Filter columns if specified
        if self.columns:
            columns = [c for c in self.columns if c in available_columns]
        else:
            columns = available_columns

        if self.include_header:
            lines.append(self.delimiter.join(columns))

        for i in range(len(wavelength)):
            row = []
            for col in columns:
                val = arrays[col][i]
                row.append(self.float_format % val)
            lines.append(self.delimiter.join(row))

        return lines

    def _export_dict(self, data: Dict[str, Any]) -> List[str]:
        """Export generic dictionary to CSV lines."""
        lines = []

        if self.include_header:
            lines.append(self.delimiter.join(["key", "value"]))

        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                # Skip arrays for simple key-value export
                continue
            elif isinstance(value, dict):
                # Flatten nested dicts
                for subkey, subval in value.items():
                    if not isinstance(subval, (list, dict, np.ndarray)):
                        lines.append(
                            self.delimiter.join(
                                [
                                    f"{key}.{subkey}",
                                    str(subval),
                                ]
                            )
                        )
            else:
                if isinstance(value, float):
                    lines.append(self.delimiter.join([key, self.float_format % value]))
                else:
                    lines.append(self.delimiter.join([key, str(value)]))

        return lines


class HDF5Exporter(Exporter):
    """
    Export analysis results to HDF5 format with hierarchical storage.

    HDF5 provides efficient storage for large arrays (spectrum, samples)
    with optional compression. The hierarchical structure preserves
    the complete analysis state for reproducibility.

    Parameters
    ----------
    compression : str, optional
        Compression algorithm ("gzip", "lzf", None)
    compression_opts : int, optional
        Compression level (1-9 for gzip)
    chunks : bool or tuple, optional
        Enable chunked storage (default: True)

    Example
    -------
    >>> exporter = HDF5Exporter(compression="gzip", compression_opts=4)
    >>> exporter.export(mcmc_result, "analysis.h5")
    >>>
    >>> # Read back
    >>> import h5py
    >>> with h5py.File("analysis.h5", "r") as f:
    ...     T_samples = f["samples/T_eV"][:]
    ...     metadata = dict(f.attrs)

    Notes
    -----
    Requires h5py: pip install h5py
    """

    def __init__(
        self,
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
        chunks: Union[bool, tuple] = True,
    ):
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunks = chunks
        self._h5py = None

    def _import_h5py(self):
        """Lazy import h5py."""
        if self._h5py is None:
            try:
                import h5py

                self._h5py = h5py
            except ImportError:
                raise ImportError("h5py required for HDF5 export. Install with: pip install h5py")
        return self._h5py

    def get_format(self) -> str:
        return "hdf5"

    def export(
        self,
        data: ExportData,
        path: PathLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Export data to HDF5 file.

        Creates hierarchical structure:
        - /metadata: Export metadata as attributes
        - /summary: Summary statistics
        - /concentrations: Concentration data
        - /samples: MCMC samples (if available)
        - /spectrum: Spectrum data (if available)
        """
        h5py = self._import_h5py()
        path = Path(path)
        data_dict = self._normalize_data(data)
        source_type = self._get_source_type(data)

        # Create metadata
        export_meta = ExportMetadata.create(
            format="hdf5",
            source_type=source_type,
            parameters=metadata,
        )

        with h5py.File(path, "w") as f:
            # Store metadata as root attributes
            f.attrs["timestamp"] = export_meta.timestamp
            f.attrs["version"] = export_meta.version
            f.attrs["source_type"] = export_meta.source_type
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        f.attrs[key] = value

            # Export based on data structure
            # Note: Check nested sampling BEFORE MCMC since nested results
            # may contain MCMC-like fields (T_eV_mean, etc.)
            if self._is_cflibs_result(data_dict):
                self._export_cflibs_result_hdf5(f, data_dict)
            elif self._is_nested_sampling_result(data_dict):
                self._export_nested_sampling_result_hdf5(f, data_dict)
            elif self._is_mcmc_result(data_dict):
                self._export_mcmc_result_hdf5(f, data_dict)
            else:
                self._export_dict_hdf5(f, data_dict)

        logger.info(f"Exported {source_type} to HDF5: {path}")

    def _create_dataset(self, group, name: str, data: np.ndarray):
        """Create dataset with compression settings."""
        data = np.asarray(data)

        # Skip compression for small datasets
        if data.size < 1000:
            group.create_dataset(name, data=data)
        else:
            group.create_dataset(
                name,
                data=data,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=self.chunks if data.ndim > 0 else None,
            )

    def _export_cflibs_result_hdf5(self, f, data: Dict[str, Any]) -> None:
        """Export CFLIBSResult to HDF5."""
        # Summary group
        summary = f.create_group("summary")
        summary.attrs["temperature_K"] = data.get("temperature_K", 0)
        summary.attrs["temperature_uncertainty_K"] = data.get("temperature_uncertainty_K", 0)
        summary.attrs["electron_density_cm3"] = data.get("electron_density_cm3", 0)
        summary.attrs["electron_density_uncertainty_cm3"] = data.get(
            "electron_density_uncertainty_cm3", 0
        )
        summary.attrs["iterations"] = data.get("iterations", 0)
        summary.attrs["converged"] = data.get("converged", False)

        # Concentrations group
        conc_group = f.create_group("concentrations")
        concentrations = data.get("concentrations", {})
        uncertainties = data.get("concentration_uncertainties", {})

        elements = sorted(concentrations.keys())
        if elements:
            conc_group.create_dataset(
                "elements",
                data=np.array(elements, dtype="S10"),
            )
            conc_group.create_dataset(
                "values",
                data=np.array([concentrations[el] for el in elements]),
            )
            conc_group.create_dataset(
                "uncertainties",
                data=np.array([uncertainties.get(el, 0) for el in elements]),
            )

        # Quality metrics
        if "quality_metrics" in data and data["quality_metrics"]:
            metrics = f.create_group("quality_metrics")
            for key, value in data["quality_metrics"].items():
                if isinstance(value, (int, float, bool)):
                    metrics.attrs[key] = value

        # Boltzmann covariance
        if "boltzmann_covariance" in data and data["boltzmann_covariance"] is not None:
            cov = np.asarray(data["boltzmann_covariance"])
            self._create_dataset(f, "boltzmann_covariance", cov)

    def _export_mcmc_result_hdf5(self, f, data: Dict[str, Any]) -> None:
        """Export MCMCResult to HDF5."""
        # Summary statistics
        summary = f.create_group("summary")
        for key in [
            "T_eV_mean",
            "T_eV_std",
            "T_eV_q025",
            "T_eV_q975",
            "log_ne_mean",
            "log_ne_std",
            "log_ne_q025",
            "log_ne_q975",
            "n_samples",
            "n_chains",
            "n_warmup",
        ]:
            if key in data:
                summary.attrs[key] = data[key]

        # Convergence status
        if "convergence_status" in data:
            status = data["convergence_status"]
            if isinstance(status, str):
                summary.attrs["convergence_status"] = status
            else:
                summary.attrs["convergence_status"] = str(status)

        # Concentrations
        conc_group = f.create_group("concentrations")
        conc_mean = data.get("concentrations_mean", {})
        conc_std = data.get("concentrations_std", {})
        conc_q025 = data.get("concentrations_q025", {})
        conc_q975 = data.get("concentrations_q975", {})

        elements = sorted(conc_mean.keys())
        if elements:
            conc_group.create_dataset("elements", data=np.array(elements, dtype="S10"))
            conc_group.create_dataset("mean", data=np.array([conc_mean[el] for el in elements]))
            conc_group.create_dataset(
                "std", data=np.array([conc_std.get(el, 0) for el in elements])
            )
            conc_group.create_dataset(
                "q025", data=np.array([conc_q025.get(el, 0) for el in elements])
            )
            conc_group.create_dataset(
                "q975", data=np.array([conc_q975.get(el, 0) for el in elements])
            )

        # Convergence diagnostics
        if "r_hat" in data and data["r_hat"]:
            diag = f.create_group("diagnostics")
            params = list(data["r_hat"].keys())
            diag.create_dataset("parameters", data=np.array(params, dtype="S20"))
            diag.create_dataset("r_hat", data=np.array([data["r_hat"][p] for p in params]))
            if "ess" in data:
                diag.create_dataset("ess", data=np.array([data["ess"].get(p, 0) for p in params]))

        # Samples (large arrays - use compression)
        if "samples" in data and data["samples"]:
            samples_group = f.create_group("samples")
            for key, value in data["samples"].items():
                arr = np.asarray(value)
                self._create_dataset(samples_group, key, arr)

    def _export_nested_sampling_result_hdf5(self, f, data: Dict[str, Any]) -> None:
        """Export NestedSamplingResult to HDF5."""
        # Evidence
        evidence = f.create_group("evidence")
        evidence.attrs["log_evidence"] = data.get("log_evidence", 0)
        evidence.attrs["log_evidence_err"] = data.get("log_evidence_err", 0)
        evidence.attrs["information"] = data.get("information", 0)

        # Summary
        summary = f.create_group("summary")
        for key in [
            "T_eV_mean",
            "T_eV_std",
            "log_ne_mean",
            "log_ne_std",
            "n_live",
            "n_iterations",
            "n_calls",
        ]:
            if key in data:
                summary.attrs[key] = data[key]

        # Concentrations
        conc_group = f.create_group("concentrations")
        conc_mean = data.get("concentrations_mean", {})
        conc_std = data.get("concentrations_std", {})

        elements = sorted(conc_mean.keys())
        if elements:
            conc_group.create_dataset("elements", data=np.array(elements, dtype="S10"))
            conc_group.create_dataset("mean", data=np.array([conc_mean[el] for el in elements]))
            conc_group.create_dataset(
                "std", data=np.array([conc_std.get(el, 0) for el in elements])
            )

        # Samples and weights
        if "samples" in data:
            samples_group = f.create_group("samples")
            for key, value in data["samples"].items():
                arr = np.asarray(value)
                self._create_dataset(samples_group, key, arr)

        if "weights" in data:
            self._create_dataset(f, "weights", np.asarray(data["weights"]))

    def _export_dict_hdf5(self, f, data: Dict[str, Any]) -> None:
        """Export generic dictionary to HDF5."""
        for key, value in data.items():
            if isinstance(value, (np.ndarray, list)):
                arr = np.asarray(value)
                if arr.dtype.kind == "U":  # Unicode string array
                    arr = arr.astype("S")
                self._create_dataset(f, key, arr)
            elif isinstance(value, dict):
                group = f.create_group(key)
                for subkey, subval in value.items():
                    if isinstance(subval, (int, float, bool, str)):
                        group.attrs[subkey] = subval
                    elif isinstance(subval, (np.ndarray, list)):
                        arr = np.asarray(subval)
                        self._create_dataset(group, subkey, arr)
            elif isinstance(value, (int, float, bool, str)):
                f.attrs[key] = value


class JSONExporter(Exporter):
    """
    Export analysis results to JSON format.

    Provides human-readable, API-compatible output with proper handling
    of numpy arrays and special float values (NaN, Inf).

    Parameters
    ----------
    indent : int, optional
        JSON indentation level (default: 2)
    sort_keys : bool
        Sort dictionary keys (default: True)
    allow_nan : bool
        Allow NaN/Inf values in output (default: True)

    Example
    -------
    >>> exporter = JSONExporter(indent=4)
    >>> exporter.export(result, "analysis.json")
    >>>
    >>> # Read back
    >>> import json
    >>> with open("analysis.json") as f:
    ...     data = json.load(f)
    """

    def __init__(
        self,
        indent: Optional[int] = 2,
        sort_keys: bool = True,
        allow_nan: bool = True,
    ):
        self.indent = indent
        self.sort_keys = sort_keys
        self.allow_nan = allow_nan

    def get_format(self) -> str:
        return "json"

    def export(
        self,
        data: ExportData,
        path: PathLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export data to JSON file."""
        path = Path(path)
        data_dict = self._normalize_data(data)
        source_type = self._get_source_type(data)

        # Create metadata
        export_meta = ExportMetadata.create(
            format="json",
            source_type=source_type,
            parameters=metadata,
        )

        # Build output structure
        output = {
            "metadata": export_meta.to_dict(),
            "data": self._prepare_for_json(data_dict),
        }

        # Write JSON
        with open(path, "w") as f:
            json.dump(
                output,
                f,
                indent=self.indent,
                sort_keys=self.sort_keys,
                allow_nan=self.allow_nan,
                default=self._json_serializer,
            )

        logger.info(f"Exported {source_type} to JSON: {path}")

    def _prepare_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for JSON serialization.

        Handles numpy arrays, special float values, and nested structures.
        """
        result = {}
        for key, value in data.items():
            result[key] = self._convert_for_json(value)
        return result

    def _convert_for_json(self, value: Any) -> Any:
        """Convert a single value for JSON serialization."""
        if value is None:
            return None
        elif isinstance(value, np.ndarray):
            return self._array_to_json(value)
        elif isinstance(value, (np.integer,)):
            return int(value)
        elif isinstance(value, (np.floating,)):
            return self._float_to_json(float(value))
        elif isinstance(value, (np.bool_,)):
            return bool(value)
        elif isinstance(value, float):
            return self._float_to_json(value)
        elif isinstance(value, dict):
            return {k: self._convert_for_json(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._convert_for_json(v) for v in value]
        elif hasattr(value, "value"):  # Enum
            return value.value
        else:
            return value

    def _float_to_json(self, value: float) -> Union[float, str, None]:
        """Convert float handling NaN and Inf."""
        if np.isnan(value):
            return "NaN" if self.allow_nan else None
        elif np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    def _array_to_json(self, arr: np.ndarray) -> Union[List, Dict]:
        """
        Convert numpy array to JSON-compatible format.

        For large arrays, stores shape and data separately.
        For small arrays, converts directly to list.
        """
        arr = np.asarray(arr)

        # Small arrays: direct conversion
        if arr.size < 10000:
            return self._convert_for_json(arr.tolist())

        # Large arrays: store with metadata
        return {
            "_type": "ndarray",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": self._convert_for_json(arr.tolist()),
        }

    def _json_serializer(self, obj: Any) -> Any:
        """Default JSON serializer for unknown types."""
        if isinstance(obj, np.ndarray):
            return self._array_to_json(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return self._float_to_json(float(obj))
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif hasattr(obj, "value"):  # Enum
            return obj.value
        elif hasattr(obj, "__dict__"):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --- Factory Function ---


def create_exporter(
    format: str,
    **kwargs,
) -> Exporter:
    """
    Factory function to create exporters by format name.

    Parameters
    ----------
    format : str
        Export format: "csv", "hdf5", "h5", "json"
    **kwargs
        Format-specific options passed to exporter constructor

    Returns
    -------
    Exporter
        Configured exporter instance

    Raises
    ------
    ValueError
        If format is not supported

    Example
    -------
    >>> exporter = create_exporter("csv", delimiter=";")
    >>> exporter = create_exporter("hdf5", compression="gzip")
    >>> exporter = create_exporter("json", indent=4)
    """
    format_lower = format.lower()

    if format_lower == "csv":
        return CSVExporter(**kwargs)
    elif format_lower in ("hdf5", "h5"):
        return HDF5Exporter(**kwargs)
    elif format_lower == "json":
        return JSONExporter(**kwargs)
    else:
        supported = ["csv", "hdf5", "h5", "json"]
        raise ValueError(
            f"Unsupported export format: '{format}'. " f"Supported formats: {supported}"
        )


# --- Convenience functions ---


def export_to_csv(
    data: ExportData,
    path: PathLike,
    **kwargs,
) -> None:
    """
    Convenience function to export data to CSV.

    Parameters
    ----------
    data : ExportData
        Data to export
    path : PathLike
        Output file path
    **kwargs
        Options passed to CSVExporter
    """
    exporter = CSVExporter(**kwargs)
    exporter.export(data, path)


def export_to_hdf5(
    data: ExportData,
    path: PathLike,
    **kwargs,
) -> None:
    """
    Convenience function to export data to HDF5.

    Parameters
    ----------
    data : ExportData
        Data to export
    path : PathLike
        Output file path
    **kwargs
        Options passed to HDF5Exporter

    Notes
    -----
    Requires h5py: pip install h5py
    """
    exporter = HDF5Exporter(**kwargs)
    exporter.export(data, path)


def export_to_json(
    data: ExportData,
    path: PathLike,
    **kwargs,
) -> None:
    """
    Convenience function to export data to JSON.

    Parameters
    ----------
    data : ExportData
        Data to export
    path : PathLike
        Output file path
    **kwargs
        Options passed to JSONExporter
    """
    exporter = JSONExporter(**kwargs)
    exporter.export(data, path)
