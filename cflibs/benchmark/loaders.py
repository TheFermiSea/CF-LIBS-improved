"""
I/O utilities for benchmark datasets.

This module provides functions for loading and saving benchmark datasets
in various formats (JSON, HDF5) for portability and efficient storage.

Supported Formats
-----------------
- JSON: Human-readable, portable, good for small-medium datasets
- HDF5: Efficient for large datasets with many spectra, supports compression

Example
-------
>>> from cflibs.benchmark import load_benchmark, save_benchmark
>>>
>>> # Load from JSON
>>> dataset = load_benchmark("steel_crm.json")
>>>
>>> # Save to HDF5
>>> save_benchmark(dataset, "steel_crm.h5", format="hdf5")
>>>
>>> # Auto-detect format from extension
>>> dataset = load_benchmark("steel_crm.h5")
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union
import json

from cflibs.benchmark.dataset import BenchmarkDataset
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("benchmark.loaders")


class BenchmarkFormat(Enum):
    """Supported benchmark file formats."""

    JSON = "json"
    HDF5 = "hdf5"


# Type alias
PathLike = Union[str, Path]


def load_benchmark(
    path: PathLike,
    format: Optional[BenchmarkFormat] = None,
) -> BenchmarkDataset:
    """
    Load a benchmark dataset from file.

    Parameters
    ----------
    path : PathLike
        Path to benchmark file
    format : BenchmarkFormat, optional
        File format (auto-detected from extension if not specified)

    Returns
    -------
    BenchmarkDataset
        Loaded benchmark dataset

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If format is not supported or file is corrupt
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    # Auto-detect format from extension
    if format is None:
        ext = path.suffix.lower()
        if ext == ".json":
            format = BenchmarkFormat.JSON
        elif ext in (".h5", ".hdf5"):
            format = BenchmarkFormat.HDF5
        else:
            raise ValueError(
                f"Cannot detect format from extension: {ext}. Please specify format explicitly."
            )

    if format == BenchmarkFormat.JSON:
        return _load_json(path)
    elif format == BenchmarkFormat.HDF5:
        return _load_hdf5(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_benchmark(
    dataset: "BenchmarkDataset",
    path: PathLike,
    format: Optional[BenchmarkFormat] = None,
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
) -> None:
    """
    Save a benchmark dataset to file.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Dataset to save
    path : PathLike
        Output file path
    format : BenchmarkFormat, optional
        File format (auto-detected from extension if not specified)
    compression : str, optional
        Compression algorithm for HDF5 ("gzip", "lzf", None)
    compression_level : int
        Compression level for gzip (1-9, default: 4)
    """
    path = Path(path)

    # Auto-detect format from extension
    if format is None:
        ext = path.suffix.lower()
        if ext == ".json":
            format = BenchmarkFormat.JSON
        elif ext in (".h5", ".hdf5"):
            format = BenchmarkFormat.HDF5
        else:
            raise ValueError(
                f"Cannot detect format from extension: {ext}. Please specify format explicitly."
            )

    if format == BenchmarkFormat.JSON:
        _save_json(dataset, path)
    elif format == BenchmarkFormat.HDF5:
        _save_hdf5(dataset, path, compression, compression_level)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved benchmark dataset to {path}")


def _load_json(path: Path) -> "BenchmarkDataset":
    """Load dataset from JSON file."""
    from cflibs.benchmark.dataset import BenchmarkDataset

    with open(path, "r") as f:
        data = json.load(f)

    return BenchmarkDataset.from_dict(data)


def _save_json(dataset: "BenchmarkDataset", path: Path) -> None:
    """Save dataset to JSON file."""
    data = dataset.to_dict()

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_hdf5(path: Path) -> "BenchmarkDataset":
    """Load dataset from HDF5 file."""
    from cflibs.benchmark.dataset import (
        BenchmarkDataset,
        BenchmarkSpectrum,
        InstrumentalConditions,
        SampleMetadata,
        SampleType,
        MatrixType,
        DataSplit,
    )

    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required for HDF5 support. Install with: pip install h5py") from e

    with h5py.File(path, "r") as f:
        # Read dataset-level metadata
        name = f.attrs["name"]
        version = f.attrs["version"]
        elements = list(f.attrs["elements"])
        description = f.attrs.get("description", "")
        citation = f.attrs.get("citation", "")
        license_ = f.attrs.get("license", "CC-BY-4.0")
        created_date = f.attrs.get("created_date", "")
        contributors = list(f.attrs.get("contributors", []))

        # Read spectra
        spectra = []
        spectra_group = f["spectra"]

        for spec_id in spectra_group.keys():
            spec_group = spectra_group[spec_id]

            # Read arrays
            wavelength_nm = spec_group["wavelength_nm"][:]
            intensity = spec_group["intensity"][:]
            intensity_uncertainty = None
            if "intensity_uncertainty" in spec_group:
                intensity_uncertainty = spec_group["intensity_uncertainty"][:]

            # Read composition
            comp_group = spec_group["composition"]
            true_composition = {}
            composition_uncertainty = {}
            for elem in comp_group.attrs.get("elements", []):
                elem_str = elem.decode() if isinstance(elem, bytes) else elem
                true_composition[elem_str] = comp_group.attrs[f"{elem_str}_value"]
                if f"{elem_str}_uncertainty" in comp_group.attrs:
                    composition_uncertainty[elem_str] = comp_group.attrs[f"{elem_str}_uncertainty"]

            # Read instrumental conditions
            cond_group = spec_group["conditions"]
            conditions = InstrumentalConditions(
                laser_wavelength_nm=cond_group.attrs["laser_wavelength_nm"],
                laser_energy_mj=cond_group.attrs["laser_energy_mj"],
                laser_pulse_width_ns=cond_group.attrs.get("laser_pulse_width_ns", 10.0),
                repetition_rate_hz=cond_group.attrs.get("repetition_rate_hz", 10.0),
                spot_diameter_um=cond_group.attrs.get("spot_diameter_um", 100.0),
                fluence_j_cm2=cond_group.attrs.get("fluence_j_cm2"),
                gate_delay_us=cond_group.attrs.get("gate_delay_us", 1.0),
                gate_width_us=cond_group.attrs.get("gate_width_us", 10.0),
                spectrometer_type=_decode_string(
                    cond_group.attrs.get("spectrometer_type", "Echelle")
                ),
                spectral_range_nm=tuple(cond_group.attrs.get("spectral_range_nm", [200.0, 900.0])),
                spectral_resolution_nm=cond_group.attrs.get("spectral_resolution_nm", 0.05),
                detector_type=_decode_string(cond_group.attrs.get("detector_type", "ICCD")),
                accumulations=cond_group.attrs.get("accumulations", 1),
                atmosphere=_decode_string(cond_group.attrs.get("atmosphere", "air")),
                pressure_mbar=cond_group.attrs.get("pressure_mbar", 1013.25),
                standoff_distance_m=cond_group.attrs.get("standoff_distance_m"),
                notes=_decode_string(cond_group.attrs.get("notes", "")),
            )

            # Read sample metadata
            meta_group = spec_group["metadata"]
            metadata = SampleMetadata(
                sample_id=_decode_string(meta_group.attrs["sample_id"]),
                sample_type=SampleType(_decode_string(meta_group.attrs.get("sample_type", "crm"))),
                matrix_type=MatrixType(
                    _decode_string(meta_group.attrs.get("matrix_type", "metal_alloy"))
                ),
                crm_name=_decode_string_optional(meta_group.attrs.get("crm_name")),
                crm_source=_decode_string_optional(meta_group.attrs.get("crm_source")),
                preparation=_decode_string(meta_group.attrs.get("preparation", "polished")),
                surface_condition=_decode_string(
                    meta_group.attrs.get("surface_condition", "polished")
                ),
                measurement_date=_decode_string_optional(meta_group.attrs.get("measurement_date")),
                laboratory=_decode_string_optional(meta_group.attrs.get("laboratory")),
                doi=_decode_string_optional(meta_group.attrs.get("doi")),
                provenance=_decode_string(meta_group.attrs.get("provenance", "")),
            )

            spectrum = BenchmarkSpectrum(
                spectrum_id=spec_id,
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=intensity_uncertainty,
                true_composition=true_composition,
                composition_uncertainty=composition_uncertainty,
                conditions=conditions,
                metadata=metadata,
                plasma_temperature_K=spec_group.attrs.get("plasma_temperature_K"),
                electron_density_cm3=spec_group.attrs.get("electron_density_cm3"),
                quality_flag=spec_group.attrs.get("quality_flag", 0),
            )
            spectra.append(spectrum)

        # Read splits
        splits = {}
        if "splits" in f:
            splits_group = f["splits"]
            for split_name in splits_group.keys():
                split_group = splits_group[split_name]
                train_ids = [_decode_string(s) for s in split_group["train_ids"][:]]
                test_ids = [_decode_string(s) for s in split_group["test_ids"][:]]
                validation_ids = None
                if "validation_ids" in split_group:
                    validation_ids = [_decode_string(s) for s in split_group["validation_ids"][:]]

                splits[split_name] = DataSplit(
                    name=split_name,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    validation_ids=validation_ids,
                    description=_decode_string(split_group.attrs.get("description", "")),
                    random_seed=split_group.attrs.get("random_seed"),
                )

    return BenchmarkDataset(
        name=name,
        version=version,
        spectra=spectra,
        elements=elements,
        splits=splits,
        description=description,
        citation=citation,
        license=license_,
        created_date=created_date,
        contributors=contributors,
    )


def _save_hdf5(
    dataset: "BenchmarkDataset",
    path: Path,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Save dataset to HDF5 file."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required for HDF5 support. Install with: pip install h5py") from e

    with h5py.File(path, "w") as f:
        # Write dataset-level metadata
        f.attrs["name"] = dataset.name
        f.attrs["version"] = dataset.version
        f.attrs["elements"] = dataset.elements
        f.attrs["description"] = dataset.description
        f.attrs["citation"] = dataset.citation
        f.attrs["license"] = dataset.license
        f.attrs["created_date"] = dataset.created_date
        f.attrs["contributors"] = dataset.contributors

        # Write spectra
        spectra_group = f.create_group("spectra")

        for spectrum in dataset.spectra:
            spec_group = spectra_group.create_group(spectrum.spectrum_id)

            # Write arrays with optional compression
            _create_dataset(
                spec_group, "wavelength_nm", spectrum.wavelength_nm, compression, compression_level
            )
            _create_dataset(
                spec_group, "intensity", spectrum.intensity, compression, compression_level
            )
            if spectrum.intensity_uncertainty is not None:
                _create_dataset(
                    spec_group,
                    "intensity_uncertainty",
                    spectrum.intensity_uncertainty,
                    compression,
                    compression_level,
                )

            # Write scalar attributes
            if spectrum.plasma_temperature_K is not None:
                spec_group.attrs["plasma_temperature_K"] = spectrum.plasma_temperature_K
            if spectrum.electron_density_cm3 is not None:
                spec_group.attrs["electron_density_cm3"] = spectrum.electron_density_cm3
            spec_group.attrs["quality_flag"] = spectrum.quality_flag

            # Write composition
            comp_group = spec_group.create_group("composition")
            comp_group.attrs["elements"] = list(spectrum.true_composition.keys())
            for elem, value in spectrum.true_composition.items():
                comp_group.attrs[f"{elem}_value"] = value
                if elem in spectrum.composition_uncertainty:
                    comp_group.attrs[f"{elem}_uncertainty"] = spectrum.composition_uncertainty[elem]

            # Write conditions
            cond_group = spec_group.create_group("conditions")
            cond = spectrum.conditions
            cond_group.attrs["laser_wavelength_nm"] = cond.laser_wavelength_nm
            cond_group.attrs["laser_energy_mj"] = cond.laser_energy_mj
            cond_group.attrs["laser_pulse_width_ns"] = cond.laser_pulse_width_ns
            cond_group.attrs["repetition_rate_hz"] = cond.repetition_rate_hz
            cond_group.attrs["spot_diameter_um"] = cond.spot_diameter_um
            if cond.fluence_j_cm2 is not None:
                cond_group.attrs["fluence_j_cm2"] = cond.fluence_j_cm2
            cond_group.attrs["gate_delay_us"] = cond.gate_delay_us
            cond_group.attrs["gate_width_us"] = cond.gate_width_us
            cond_group.attrs["spectrometer_type"] = cond.spectrometer_type
            cond_group.attrs["spectral_range_nm"] = list(cond.spectral_range_nm)
            cond_group.attrs["spectral_resolution_nm"] = cond.spectral_resolution_nm
            cond_group.attrs["detector_type"] = cond.detector_type
            cond_group.attrs["accumulations"] = cond.accumulations
            cond_group.attrs["atmosphere"] = cond.atmosphere
            cond_group.attrs["pressure_mbar"] = cond.pressure_mbar
            if cond.standoff_distance_m is not None:
                cond_group.attrs["standoff_distance_m"] = cond.standoff_distance_m
            cond_group.attrs["notes"] = cond.notes

            # Write metadata
            meta_group = spec_group.create_group("metadata")
            meta = spectrum.metadata
            meta_group.attrs["sample_id"] = meta.sample_id
            meta_group.attrs["sample_type"] = meta.sample_type.value
            meta_group.attrs["matrix_type"] = meta.matrix_type.value
            if meta.crm_name is not None:
                meta_group.attrs["crm_name"] = meta.crm_name
            if meta.crm_source is not None:
                meta_group.attrs["crm_source"] = meta.crm_source
            meta_group.attrs["preparation"] = meta.preparation
            meta_group.attrs["surface_condition"] = meta.surface_condition
            if meta.measurement_date is not None:
                meta_group.attrs["measurement_date"] = meta.measurement_date
            if meta.laboratory is not None:
                meta_group.attrs["laboratory"] = meta.laboratory
            if meta.doi is not None:
                meta_group.attrs["doi"] = meta.doi
            meta_group.attrs["provenance"] = meta.provenance

        # Write splits
        if dataset.splits:
            splits_group = f.create_group("splits")
            for name, split in dataset.splits.items():
                split_group = splits_group.create_group(name)
                # Use variable-length strings
                dt = h5py.string_dtype(encoding="utf-8")
                split_group.create_dataset("train_ids", data=split.train_ids, dtype=dt)
                split_group.create_dataset("test_ids", data=split.test_ids, dtype=dt)
                if split.validation_ids:
                    split_group.create_dataset(
                        "validation_ids", data=split.validation_ids, dtype=dt
                    )
                split_group.attrs["description"] = split.description
                if split.random_seed is not None:
                    split_group.attrs["random_seed"] = split.random_seed


def _create_dataset(
    group,
    name: str,
    data: np.ndarray,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Create HDF5 dataset with optional compression."""
    if compression and len(data) > 1000:
        group.create_dataset(
            name,
            data=data,
            compression=compression,
            compression_opts=compression_level if compression == "gzip" else None,
        )
    else:
        group.create_dataset(name, data=data)


def _decode_string(value) -> str:
    """Decode HDF5 string (bytes or str) to str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_string_optional(value) -> Optional[str]:
    """Decode optional HDF5 string."""
    if value is None:
        return None
    return _decode_string(value)


def validate_benchmark_file(path: PathLike) -> Dict[str, bool]:
    """
    Validate a benchmark file without fully loading it.

    Parameters
    ----------
    path : PathLike
        Path to benchmark file

    Returns
    -------
    Dict[str, bool]
        Validation results {check_name: passed}
    """
    path = Path(path)
    results = {
        "file_exists": path.exists(),
        "readable": False,
        "valid_structure": False,
        "has_spectra": False,
        "has_splits": False,
        "compositions_valid": False,
    }

    if not results["file_exists"]:
        return results

    try:
        dataset = load_benchmark(path)
        results["readable"] = True
        results["valid_structure"] = True
        results["has_spectra"] = dataset.n_spectra > 0
        results["has_splits"] = len(dataset.splits) > 0

        # Check composition validity
        all_valid = True
        for spec in dataset.spectra:
            total = sum(spec.true_composition.values())
            if abs(total - 1.0) > 0.1:
                all_valid = False
                break
        results["compositions_valid"] = all_valid

    except Exception as e:
        logger.warning(f"Validation failed: {e}")

    return results
