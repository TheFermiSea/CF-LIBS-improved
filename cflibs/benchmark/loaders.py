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
from typing import Dict, List, Optional, Union
import json

from cflibs.benchmark.dataset import BenchmarkDataset, TruthType
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
                dataset_id=_decode_string_optional(spec_group.attrs.get("dataset_id")),
                group_id=_decode_string_optional(spec_group.attrs.get("group_id")),
                specimen_id=_decode_string_optional(spec_group.attrs.get("specimen_id")),
                instrument_id=_decode_string_optional(spec_group.attrs.get("instrument_id")),
                truth_type=_decode_string(
                    spec_group.attrs.get("truth_type", TruthType.ASSAY.value)
                ),
                rp_estimate=spec_group.attrs.get("rp_estimate"),
                label_cardinality=spec_group.attrs.get("label_cardinality"),
                spectrum_kind=_decode_string_optional(spec_group.attrs.get("spectrum_kind")),
                annotations=json.loads(
                    _decode_string(spec_group.attrs.get("annotations_json", "{}"))
                ),
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
                    metadata=json.loads(
                        _decode_string(split_group.attrs.get("metadata_json", "{}"))
                    ),
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
            if spectrum.dataset_id is not None:
                spec_group.attrs["dataset_id"] = spectrum.dataset_id
            if spectrum.group_id is not None:
                spec_group.attrs["group_id"] = spectrum.group_id
            if spectrum.specimen_id is not None:
                spec_group.attrs["specimen_id"] = spectrum.specimen_id
            if spectrum.instrument_id is not None:
                spec_group.attrs["instrument_id"] = spectrum.instrument_id
            spec_group.attrs["truth_type"] = spectrum.truth_type.value
            if spectrum.rp_estimate is not None:
                spec_group.attrs["rp_estimate"] = spectrum.rp_estimate
            if spectrum.label_cardinality is not None:
                spec_group.attrs["label_cardinality"] = spectrum.label_cardinality
            if spectrum.spectrum_kind is not None:
                spec_group.attrs["spectrum_kind"] = spectrum.spectrum_kind
            if spectrum.annotations:
                spec_group.attrs["annotations_json"] = json.dumps(spectrum.annotations)

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
                if split.metadata:
                    split_group.attrs["metadata_json"] = json.dumps(split.metadata)


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


# ---------------------------------------------------------------------------
# Community CRM dataset loaders: BHVO-2 (USGS) and NIST SRM 612
# ---------------------------------------------------------------------------
#
# These loaders scan a flat CSV directory for all *.csv files, wrap each file
# as a BenchmarkSpectrum, and set true_composition from the certified-
# composition table in reference_compositions.py.  The directory is populated
# by the spectra-ingest pipeline; the loaders return None gracefully when the
# directory is absent so the benchmark registry degrades without error before
# ingest has run.
#
# File format: any CSV accepted by cflibs.io.spectrum.load_spectrum — i.e.
# two columns named ``wavelength`` / ``wavelength_nm`` and ``intensity``.

_REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_crm_dataset(
    dataset_id: str,
    data_dir: Optional[Path] = None,
) -> Optional[BenchmarkDataset]:
    """
    Generic loader for community CRM datasets stored as flat CSV collections.

    Each ``*.csv`` file in *data_dir / dataset_id* is loaded as one
    :class:`~cflibs.benchmark.dataset.BenchmarkSpectrum`; ``true_composition``
    is drawn from
    :func:`~cflibs.benchmark.reference_compositions.get_reference_composition`
    so the certified table in that module is the single source of truth —
    never hardcoded here.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier, e.g. ``"bhvo2_usgs"`` or ``"nist_srm_612"``.
        Must be a key in ``REFERENCE_COMPOSITIONS``.
    data_dir : Path, optional
        Root data directory.  Defaults to ``<repo>/data/``.

    Returns
    -------
    BenchmarkDataset or None
        Loaded dataset, or ``None`` if the directory is absent or empty.

    Raises
    ------
    ValueError
        If *dataset_id* is not registered in ``REFERENCE_COMPOSITIONS``.
    """
    from cflibs.benchmark.dataset import (
        BenchmarkSpectrum,
        InstrumentalConditions,
        MatrixType,
        SampleMetadata,
        SampleType,
    )
    from cflibs.benchmark.reference_compositions import get_reference_composition
    from cflibs.io.spectrum import load_spectrum

    if data_dir is None:
        data_dir = _REPO_DATA_DIR
    spectra_dir = Path(data_dir) / dataset_id

    if not spectra_dir.is_dir():
        logger.debug("CRM dataset directory not found: %s (skipped)", spectra_dir)
        return None

    true_composition_mapping = get_reference_composition(dataset_id)
    if true_composition_mapping is None:
        raise ValueError(
            f"No reference composition registered for dataset_id={dataset_id!r}. "
            "Add it to cflibs/benchmark/reference_compositions.py first."
        )
    # Convert to a plain mutable dict so BenchmarkSpectrum's validator is happy.
    true_composition: Dict[str, float] = dict(true_composition_mapping)

    csv_paths = sorted(spectra_dir.glob("*.csv"))
    if not csv_paths:
        logger.debug("CRM dataset directory is empty: %s (skipped)", spectra_dir)
        return None

    spectra = []
    for csv_path in csv_paths:
        try:
            wavelength, intensity = load_spectrum(str(csv_path))
        except Exception as exc:  # pragma: no cover — bad files are warned, not fatal
            logger.warning("Failed to load spectrum %s: %s", csv_path.name, exc)
            continue

        wavelength = np.asarray(wavelength, dtype=float)
        intensity = np.asarray(intensity, dtype=float)

        conditions = InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=0.0,
            spectral_range_nm=(float(wavelength.min()), float(wavelength.max())),
            spectral_resolution_nm=0.05,
            spectrometer_type="user_supplied",
            detector_type="unknown",
            atmosphere="air",
            notes=f"Loaded from {csv_path.name}",
        )
        metadata = SampleMetadata(
            sample_id=csv_path.stem,
            sample_type=SampleType.CRM,
            matrix_type=MatrixType.GEOLOGICAL,
            crm_name=dataset_id,
            crm_source="community_crm",
            preparation="unknown",
            surface_condition="unknown",
            provenance=(
                f"Spectrum file: {csv_path.name}; "
                "certified composition: cflibs/benchmark/reference_compositions.py"
            ),
        )
        spectra.append(
            BenchmarkSpectrum(
                spectrum_id=f"{dataset_id}_{csv_path.stem}",
                wavelength_nm=wavelength,
                intensity=intensity,
                true_composition=true_composition,
                conditions=conditions,
                metadata=metadata,
                dataset_id=dataset_id,
                group_id=dataset_id,
                specimen_id=csv_path.stem,
                instrument_id="user_supplied",
                truth_type=TruthType.ASSAY,
                spectrum_kind="geostandard",
                annotations={
                    "source_file": csv_path.name,
                    "dataset_id": dataset_id,
                },
            )
        )

    if not spectra:
        return None

    elements = sorted(true_composition.keys())
    return BenchmarkDataset(
        name=dataset_id,
        version="v1",
        spectra=spectra,
        elements=elements,
        description=(
            f"Community CF-LIBS reference material: {dataset_id}. "
            "Certified compositions from reference_compositions.py."
        ),
        citation="See cflibs/benchmark/reference_compositions.py for citations.",
        contributors=["CF-LIBS"],
    )


def _load_bhvo2_usgs(data_dir: Optional[Path] = None) -> Optional[BenchmarkDataset]:
    """
    Load USGS BHVO-2 Hawaiian basalt LIBS spectra from ``data/bhvo2_usgs/``.

    Each ``*.csv`` file in the directory is treated as one acquisition
    (shot-averaged or single-shot, depending on how the ingest pipeline wrote
    the files).  ``true_composition`` is set to
    ``REFERENCE_COMPOSITIONS["bhvo2_usgs"]`` — the Jochum 2005 GeoReM oxide
    compilation converted to cation mass fractions.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.  Defaults to ``<repo>/data/``.

    Returns
    -------
    BenchmarkDataset or None
        Returns ``None`` if ``data/bhvo2_usgs/`` is absent or empty (i.e.
        before the spectra-ingest pipeline has run).
    """
    return _load_crm_dataset("bhvo2_usgs", data_dir)


def _load_nist_srm_612(data_dir: Optional[Path] = None) -> Optional[BenchmarkDataset]:
    """
    Load NIST SRM 612 trace-element glass LIBS spectra from
    ``data/nist_srm_612/``.

    Each ``*.csv`` file in the directory is one acquisition.
    ``true_composition`` is set to ``REFERENCE_COMPOSITIONS["nist_srm_612"]``
    — the Pearce et al. 1997 major-element glass matrix composition (Si, Al,
    Ca, Na cation mass fractions; trace dopants omitted as below-LOQ).

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.  Defaults to ``<repo>/data/``.

    Returns
    -------
    BenchmarkDataset or None
        Returns ``None`` if ``data/nist_srm_612/`` is absent or empty.
    """
    return _load_crm_dataset("nist_srm_612", data_dir)


def _load_vrabel2020_soils(
    data_dir: Optional[Path] = None,
    max_spectra_per_sample: Optional[int] = 50,
) -> Optional[BenchmarkDataset]:
    """
    Load the Vrabel et al. 2020 LIBS soil benchmark dataset.

    Reference
    ---------
    Vrabel, J., et al. (2020). "Benchmark classification dataset for
    laser-induced breakdown spectroscopy." *Scientific Data* 7:175.
    doi:10.1038/s41597-020-0396-8

    The published benchmark contains 100 soil/ore samples × 500 spectra each
    in the training set (50,000 total) and ~20,000 spectra in the test set,
    measured at 40,002 wavelength bins from 200–1000 nm. Each sample has a
    certified composition for 10 elements (Al, Ca, Cr, Cu, Fe, K, Mg, Na, Pb, Si)
    with reported uncertainties.

    Data location
    -------------
    ``<repo>/data/vrabel2020_soil_benchmark/``:

    * ``train.h5`` — 7.1 GB HDF5 with /Spectra/NNN groups (one per sample,
      each shape (40002, 500)).
    * ``test.h5`` — 3.0 GB HDF5 with /UNKNOWN/* test groups.
    * ``support_tables.xlsx`` — MIXED_composition sheet (100 samples × 10 elements
      in weight-percent) and MIXED_uncertainty sheet.
    * ``test_labels.csv`` — 20,000 integer class labels (1–12).

    Memory note
    -----------
    Loading all 50,000 train spectra at full resolution would require ~16 GB in
    memory.  ``max_spectra_per_sample`` (default 50) caps the per-sample shot
    count so the default load is ~5,000 spectra × ~1.6 GB.  Pass ``None`` to
    load the full 50k.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.  Defaults to ``<repo>/data/``.
    max_spectra_per_sample : int or None, default 50
        Cap shots loaded per sample.  ``None`` loads everything (50k spectra,
        ~16 GB RAM).  Lower values produce smaller / faster benchmarks.

    Returns
    -------
    BenchmarkDataset or None
        Returns ``None`` if the Vrabel data is not present in tree.
    """
    from cflibs.benchmark.dataset import (
        BenchmarkSpectrum,
        InstrumentalConditions,
        MatrixType,
        SampleMetadata,
        SampleType,
    )

    if data_dir is None:
        data_dir = _REPO_DATA_DIR
    vrabel_dir = Path(data_dir) / "vrabel2020_soil_benchmark"
    train_path = vrabel_dir / "train.h5"
    support_path = vrabel_dir / "support_tables.xlsx"

    if not (train_path.is_file() and support_path.is_file()):
        logger.debug(
            "Vrabel 2020 data not found in %s (skipped; need train.h5 + support_tables.xlsx)",
            vrabel_dir,
        )
        return None

    try:
        import h5py
        import openpyxl
    except ImportError as exc:  # pragma: no cover — optional deps
        logger.warning("Vrabel loader needs h5py + openpyxl: %s", exc)
        return None

    # Read certified compositions from the MIXED_composition sheet:
    # row = sample ID (1..100), columns = Class ID, Al, Ca, Cr, Cu, Fe, K, Mg, Na, Pb, Si.
    # Values are weight-percent; divide by 100 for mass fraction.
    wb = openpyxl.load_workbook(support_path, read_only=True, data_only=True)
    ws = wb["MIXED_composition"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    element_cols = {h: i for i, h in enumerate(header) if h not in ("Sample ID", "Class ID")}
    elements = sorted(element_cols.keys())

    sample_compositions: Dict[int, Dict[str, float]] = {}
    sample_class: Dict[int, int] = {}
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        sample_id = int(row[0])
        sample_class[sample_id] = int(row[1])
        comp = {
            el: float(row[idx]) / 100.0 if row[idx] is not None else 0.0
            for el, idx in element_cols.items()
        }
        sample_compositions[sample_id] = comp
    wb.close()

    spectra = []
    with h5py.File(train_path, "r") as f:
        # /Wavelengths/1: shape (40002,), 200–1000 nm.
        wavelength_nm = np.asarray(f["Wavelengths"]["1"][:], dtype=float)
        spectral_range = (float(wavelength_nm.min()), float(wavelength_nm.max()))

        # /Spectra/NNN: shape (40002, 500), one per sample.
        spectra_grp = f["Spectra"]
        sample_keys = sorted(spectra_grp.keys(), key=lambda s: int(s))

        for sk in sample_keys:
            sample_id = int(sk)
            if sample_id not in sample_compositions:
                logger.debug("Vrabel sample %d has no composition; skipped", sample_id)
                continue

            arr = spectra_grp[sk]  # shape (40002, N_shots)
            n_shots = arr.shape[1]
            if max_spectra_per_sample is not None:
                n_load = min(max_spectra_per_sample, n_shots)
            else:
                n_load = n_shots

            # Load the per-shot intensities (40002, n_load) once and slice per shot.
            shots = np.asarray(arr[:, :n_load], dtype=float)

            comp = sample_compositions[sample_id]
            cls = sample_class.get(sample_id, 0)

            conditions = InstrumentalConditions(
                laser_wavelength_nm=1064.0,
                laser_energy_mj=0.0,
                spectral_range_nm=spectral_range,
                spectral_resolution_nm=0.02,
                spectrometer_type="echelle",
                detector_type="ICCD",
                atmosphere="air",
                notes=(
                    "Vrabel et al. 2020 Sci Data benchmark; mixed soil/ore samples; "
                    f"sample {sample_id} (class {cls})"
                ),
            )

            for shot_idx in range(n_load):
                metadata = SampleMetadata(
                    sample_id=f"vrabel_sample_{sample_id:03d}",
                    sample_type=SampleType.CRM,
                    matrix_type=MatrixType.GEOLOGICAL,
                    crm_name=f"vrabel2020_sample_{sample_id:03d}",
                    crm_source="Vrabel 2020 Sci Data benchmark",
                    preparation="pressed pellet",
                    surface_condition="prepared",
                    provenance=(
                        f"Vrabel et al. 2020 Sci Data 7:175; sample {sample_id}, "
                        f"shot {shot_idx + 1}/{n_load} (of {n_shots} available); "
                        "composition: support_tables.xlsx MIXED_composition sheet"
                    ),
                )
                spectra.append(
                    BenchmarkSpectrum(
                        spectrum_id=f"vrabel2020_s{sample_id:03d}_shot{shot_idx:03d}",
                        wavelength_nm=wavelength_nm,
                        intensity=shots[:, shot_idx],
                        true_composition=comp,
                        conditions=conditions,
                        metadata=metadata,
                        dataset_id="vrabel2020_soil_benchmark",
                        group_id=f"vrabel_sample_{sample_id:03d}",
                        specimen_id=f"vrabel_sample_{sample_id:03d}",
                        instrument_id="vrabel2020_echelle_iccd",
                        truth_type=TruthType.ASSAY,
                        spectrum_kind="geostandard",
                        annotations={
                            "vrabel_sample_id": sample_id,
                            "vrabel_class_id": cls,
                            "shot_index": shot_idx,
                            "n_shots_total": n_shots,
                        },
                    )
                )

    if not spectra:
        return None

    return BenchmarkDataset(
        name="vrabel2020_soil_benchmark",
        version="v1",
        spectra=spectra,
        elements=elements,
        description=(
            "Vrabel et al. 2020 *Scientific Data* peer-reviewed LIBS benchmark. "
            "100 mixed soil/ore samples × per-sample shots × certified compositions "
            "for Al, Ca, Cr, Cu, Fe, K, Mg, Na, Pb, Si. "
            f"This load: {len(spectra)} spectra "
            f"(cap = {max_spectra_per_sample} per sample)."
        ),
        citation=(
            "Vrabel, J., et al. (2020). Benchmark classification dataset for "
            "laser-induced breakdown spectroscopy. Scientific Data 7:175. "
            "doi:10.1038/s41597-020-0396-8"
        ),
        contributors=["Vrabel et al."],
    )


def build_dataset_registry(
    data_dir: Optional[Path] = None,
) -> List[BenchmarkDataset]:
    """
    Return a list of all available community CRM datasets.

    Scans *data_dir* for ``bhvo2_usgs`` and ``nist_srm_612`` subdirectories
    and returns a :class:`BenchmarkDataset` for each that is present and
    non-empty.  Datasets whose directories are absent are silently omitted so
    the registry degrades gracefully before the spectra-ingest pipeline has
    run.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.  Defaults to ``<repo>/data/``.

    Returns
    -------
    list of BenchmarkDataset

    Examples
    --------
    >>> from cflibs.benchmark.loaders import build_dataset_registry
    >>> datasets = build_dataset_registry()
    >>> for ds in datasets:
    ...     print(ds.name, ds.n_spectra)
    """
    datasets: List[BenchmarkDataset] = []
    for loader in (_load_bhvo2_usgs, _load_nist_srm_612, _load_vrabel2020_soils):
        ds = loader(data_dir)
        if ds is not None:
            datasets.append(ds)
    return datasets
