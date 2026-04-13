"""
Unified grouped benchmark runner for LIBS workflow comparison.

This module centralizes:
- dataset adapters for real, assay-backed, blind, and synthetic corpora
- leakage-safe grouped split construction
- workflow registries for element identification and composition estimation
- nested evaluation, metrics, summary tables, and statistical tests
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import importlib.util
import json
import sys
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cflibs.benchmark.dataset import (
    BenchmarkDataset,
    BenchmarkSpectrum,
    DataSplit,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
    TruthType,
)
from cflibs.benchmark.loaders import load_benchmark
from cflibs.benchmark.composition_metrics import (
    aitchison_distance,
    per_element_error,
    rmse_composition,
)
from cflibs.core.logging_config import get_logger

if TYPE_CHECKING:
    from cflibs.inversion.element_id import ElementIdentificationResult

logger = get_logger("benchmark.unified")

try:
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover - optional dependency
    find_peaks = None

try:
    from scipy.stats import chi2, friedmanchisquare, studentized_range
except ImportError:  # pragma: no cover - optional dependency
    chi2 = None
    friedmanchisquare = None
    studentized_range = None


RP_BUCKETS: List[Tuple[float, float, str, float]] = [
    (0.0, 500.0, "rp_lt_500", 350.0),
    (500.0, 1000.0, "rp_500_999", 750.0),
    (1000.0, 3000.0, "rp_1000_2999", 2000.0),
    (3000.0, float("inf"), "rp_ge_3000", 5000.0),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_MODULE_CACHE: Dict[str, Any] = {}


def _load_repo_script_module(module_name: str) -> Any:
    if module_name in _SCRIPT_MODULE_CACHE:
        return _SCRIPT_MODULE_CACHE[module_name]

    script_path = _REPO_ROOT / "scripts" / f"{module_name}.py"
    if not script_path.exists():
        raise ImportError(
            f"Unified benchmark helper '{module_name}' requires {script_path} to be present."
        )

    spec = importlib.util.spec_from_file_location(
        f"_cflibs_repo_scripts_{module_name}", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SCRIPT_MODULE_CACHE[module_name] = module
    return module


def _aalto_search_elements() -> List[str]:
    return list(_load_repo_script_module("calibrate_alias").AALTO_SEARCH_ELEMENTS)


def _select_aalto_cases_lazy(data_dir: Path):
    return _load_repo_script_module("calibrate_alias")._select_aalto_cases(data_dir)


def _estimate_effective_rp_lazy(wavelength_nm: np.ndarray, intensity: np.ndarray) -> float:
    return float(
        _load_repo_script_module("calibrate_alias")._estimate_effective_rp(wavelength_nm, intensity)
    )


def _load_real_spectra_lazy(data_dir: Path) -> List[Dict[str, Any]]:
    return _load_repo_script_module("run_comprehensive_benchmark").load_real_spectra(data_dir)


def _run_boltzmann_pipeline_lazy(
    spectrum: Dict[str, Any],
    db: Any,
    fit_method: Any,
    closure_mode: str,
    elements: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    return _load_repo_script_module("run_comprehensive_benchmark").run_boltzmann_pipeline(
        spectrum,
        db=db,
        fit_method=fit_method,
        closure_mode=closure_mode,
        elements=elements,
    )


def _pipeline_joint_softmax_lazy(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    elements: List[str],
) -> Dict[str, Any]:
    return _load_repo_script_module("run_experiments_advanced")._pipeline_joint_softmax(
        wavelength_nm, intensity, elements
    )


def _pipeline_hybrid_manifold_lazy(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    elements: List[str],
) -> Dict[str, Any]:
    return _load_repo_script_module("run_experiments_advanced")._pipeline_hybrid_manifold(
        wavelength_nm, intensity, elements
    )


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _rp_bucket(rp: Optional[float]) -> str:
    if rp is None or not np.isfinite(rp):
        return "rp_unknown"
    for lo, hi, label, _mid in RP_BUCKETS:
        if lo <= rp < hi:
            return label
    return "rp_unknown"


def _rp_bucket_midpoint(bucket: str) -> float:
    for _lo, _hi, label, midpoint in RP_BUCKETS:
        if bucket == label:
            return midpoint
    return float("nan")


def _sorted_positive_elements(composition: Dict[str, float], threshold: float = 1e-9) -> List[str]:
    return sorted([el for el, value in composition.items() if float(value) > threshold])


def derive_truth_elements(
    spectrum: BenchmarkSpectrum,
    presence_threshold: float = 1e-9,
) -> List[str]:
    if spectrum.truth_type == TruthType.BLIND:
        return []
    annotated = spectrum.annotations.get("expected_elements")
    if annotated:
        return sorted(str(el) for el in annotated)
    return _sorted_positive_elements(spectrum.true_composition, threshold=presence_threshold)


def _clone_spectrum(spectrum: BenchmarkSpectrum, **updates: Any) -> BenchmarkSpectrum:
    payload = spectrum.to_dict()
    payload.update(updates)
    return BenchmarkSpectrum.from_dict(payload)


def subset_dataset(
    dataset: BenchmarkDataset, spectrum_ids: Sequence[str], name: str
) -> BenchmarkDataset:
    spectra = [dataset.get_spectrum(spec_id) for spec_id in spectrum_ids]
    return BenchmarkDataset(
        name=name,
        version=dataset.version,
        spectra=spectra,
        elements=list(dataset.elements),
        description=dataset.description,
        citation=dataset.citation,
        license=dataset.license,
        created_date=dataset.created_date,
        contributors=list(dataset.contributors),
    )


def _mineral_group_from_name(case_name: str) -> str:
    stem = case_name.removeprefix("aalto_")
    if stem.startswith("elem_"):
        return stem.removeprefix("elem_")
    trimmed = stem.rstrip("0123456789")
    if trimmed.endswith("E"):
        trimmed = trimmed[:-1]
    return trimmed.lower()


def load_aalto_id_dataset(data_dir: Path) -> BenchmarkDataset:
    """Load the 74-spectrum Aalto low-RP identification benchmark."""
    cases = _select_aalto_cases_lazy(data_dir)
    spectra: List[BenchmarkSpectrum] = []

    for case in cases:
        expected = sorted(case.expected)
        if not expected:
            continue
        is_pure = case.name.startswith("aalto_elem_")
        if is_pure:
            specimen_id = case.name.removeprefix("aalto_elem_")
            group_id = specimen_id
            matrix_type = MatrixType.METAL_PURE
            spectrum_kind = "pure_element"
            truth_type = TruthType.ASSAY
        else:
            specimen_id = case.path.stem.replace("_spectrum", "")
            group_id = _mineral_group_from_name(case.name)
            matrix_type = MatrixType.GEOLOGICAL
            spectrum_kind = "mineral"
            truth_type = TruthType.FORMULA_PROXY

        composition = {el: 1.0 / len(expected) for el in expected}
        mean_wl = float(np.mean(case.wavelength))
        resolution_nm = mean_wl / max(case.resolving_power, 1.0)
        conditions = InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=0.0,
            spectral_range_nm=(float(case.wavelength.min()), float(case.wavelength.max())),
            spectral_resolution_nm=resolution_nm,
            spectrometer_type="Aalto",
            detector_type="unknown",
            atmosphere="air",
            notes="Imported from Aalto low-RP element identification benchmark",
        )
        metadata = SampleMetadata(
            sample_id=specimen_id,
            sample_type=SampleType.FIELD,
            matrix_type=matrix_type,
            provenance=f"Aalto benchmark source file: {case.path.name}",
        )
        spectra.append(
            BenchmarkSpectrum(
                spectrum_id=case.name,
                wavelength_nm=case.wavelength,
                intensity=case.spectrum,
                true_composition=composition,
                conditions=conditions,
                metadata=metadata,
                dataset_id="aalto_libs",
                group_id=group_id,
                specimen_id=specimen_id,
                instrument_id="aalto_laser_libs",
                truth_type=truth_type,
                rp_estimate=float(case.resolving_power),
                label_cardinality=len(expected),
                spectrum_kind=spectrum_kind,
                annotations={"expected_elements": expected, "path": str(case.path)},
            )
        )

    return BenchmarkDataset(
        name="aalto_libs",
        version="v2",
        spectra=spectra,
        elements=_aalto_search_elements(),
        description="Aalto low-resolution LIBS element-identification benchmark.",
        citation="Aalto University LIBS spectral library benchmark.",
        contributors=["CF-LIBS"],
    )


def _real_dataset_id(label: str, has_ground_truth: bool) -> str:
    if label.startswith("AA1100_"):
        return "aa1100_substrate"
    if label.startswith("Ti6Al4V_"):
        return "ti6al4v_substrate"
    if label.startswith("20shot_"):
        return "20shot_blind"
    return "assay_substrates" if has_ground_truth else "blind_stress"


def _build_real_benchmark_spectrum(record: Dict[str, Any]) -> BenchmarkSpectrum:
    wavelength = np.asarray(record["wavelength"], dtype=float)
    intensity = np.asarray(record["intensity"], dtype=float)
    label = str(record["label"])
    ground_truth = dict(record.get("ground_truth") or {})
    source = str(record.get("source", "experimental"))
    specimen_id = label.split("_shot")[0].split("_pos")[0]
    group_id = specimen_id if ground_truth else label
    rp_estimate = _estimate_effective_rp_lazy(wavelength, intensity)
    resolution_nm = float(np.mean(wavelength) / max(rp_estimate, 1.0))
    conditions = InstrumentalConditions(
        laser_wavelength_nm=1064.0,
        laser_energy_mj=0.0,
        spectral_range_nm=(float(wavelength.min()), float(wavelength.max())),
        spectral_resolution_nm=resolution_nm,
        spectrometer_type="Scipp-HDF5",
        detector_type="unknown",
        atmosphere="air",
        notes=f"Imported from {source}",
    )
    metadata = SampleMetadata(
        sample_id=specimen_id,
        sample_type=SampleType.CRM if ground_truth else SampleType.FIELD,
        matrix_type=MatrixType.METAL_ALLOY,
        provenance=f"Real substrate spectrum: {label}",
    )
    return BenchmarkSpectrum(
        spectrum_id=label,
        wavelength_nm=wavelength,
        intensity=intensity,
        true_composition=ground_truth,
        conditions=conditions,
        metadata=metadata,
        dataset_id=_real_dataset_id(label, bool(ground_truth)),
        group_id=group_id,
        specimen_id=specimen_id,
        instrument_id="scipp_substrate",
        truth_type=TruthType.ASSAY if ground_truth else TruthType.BLIND,
        rp_estimate=rp_estimate,
        label_cardinality=len(_sorted_positive_elements(ground_truth)),
        spectrum_kind="substrate_shot" if ground_truth else "blind_stress",
        annotations={"source": source},
    )


def load_assay_and_blind_datasets(data_dir: Path) -> Dict[str, BenchmarkDataset]:
    """Load assay-backed substrate shots and blind stress spectra."""
    records = _load_real_spectra_lazy(data_dir)
    assay_spectra: List[BenchmarkSpectrum] = []
    blind_spectra: List[BenchmarkSpectrum] = []

    for record in records:
        spectrum = _build_real_benchmark_spectrum(record)
        if spectrum.true_composition:
            assay_spectra.append(spectrum)
        else:
            blind_spectra.append(spectrum)

    datasets: Dict[str, BenchmarkDataset] = {}
    if assay_spectra:
        elements = sorted({el for spec in assay_spectra for el in spec.true_composition})
        datasets["assay_substrates"] = BenchmarkDataset(
            name="assay_substrates",
            version="v2",
            spectra=assay_spectra,
            elements=elements,
            description="AA1100 and Ti6Al4V substrate shots with quantitative compositions.",
            citation="Internal assay-backed substrate benchmark.",
            contributors=["CF-LIBS"],
        )
    if blind_spectra:
        blind_elements = sorted({el for spec in blind_spectra for el in spec.true_composition})
        datasets["blind_stress"] = BenchmarkDataset(
            name="blind_stress",
            version="v2",
            spectra=blind_spectra,
            elements=blind_elements,
            description="Blind stress spectra excluded from supervised scoring.",
            citation="Internal blind stress dataset.",
            contributors=["CF-LIBS"],
        )
    return datasets


def _load_manifest_rows(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    if manifest_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(manifest_path.read_text())
    manifest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            manifest[str(sample_id)] = row
    return manifest


def load_manifest_synthetic_dataset(
    corpus_path: Path, manifest_path: Optional[Path] = None
) -> BenchmarkDataset:
    """Load the manifest-backed synthetic ID benchmark corpus."""
    dataset = load_benchmark(corpus_path)
    if manifest_path is None:
        manifest_path = corpus_path.with_name("manifest.json")
    manifest = _load_manifest_rows(manifest_path)
    spectra: List[BenchmarkSpectrum] = []

    for spectrum in dataset.spectra:
        row = manifest.get(spectrum.spectrum_id, {})
        perturb = row.get("perturbation", {})
        rp_estimate = float(perturb.get("resolving_power", spectrum.rp_estimate or 1000.0))
        spectra.append(
            _clone_spectrum(
                spectrum,
                dataset_id=dataset.name,
                group_id=row.get("recipe", spectrum.group_id or spectrum.spectrum_id),
                specimen_id=row.get("recipe", spectrum.specimen_id or spectrum.spectrum_id),
                instrument_id="synthetic_forward_model",
                truth_type=TruthType.SYNTHETIC.value,
                rp_estimate=rp_estimate,
                label_cardinality=len(_sorted_positive_elements(spectrum.true_composition)),
                spectrum_kind="synthetic",
                annotations={
                    **dict(spectrum.annotations),
                    "recipe": row.get("recipe"),
                    "present_elements": row.get("present_elements", []),
                    "perturbation": perturb,
                },
            )
        )

    return BenchmarkDataset(
        name=dataset.name,
        version=dataset.version,
        spectra=spectra,
        elements=list(dataset.elements),
        description=dataset.description,
        citation=dataset.citation,
        license=dataset.license,
        created_date=dataset.created_date,
        contributors=list(dataset.contributors),
    )


def load_default_datasets(
    data_dir: Path,
    synthetic_corpus_path: Optional[Path] = None,
) -> Dict[str, BenchmarkDataset]:
    datasets: Dict[str, BenchmarkDataset] = {"aalto_libs": load_aalto_id_dataset(data_dir)}
    datasets.update(load_assay_and_blind_datasets(data_dir))
    if synthetic_corpus_path is not None and synthetic_corpus_path.exists():
        datasets["synthetic_id"] = load_manifest_synthetic_dataset(synthetic_corpus_path)
    return datasets


def build_outer_splits(dataset: BenchmarkDataset) -> List[DataSplit]:
    """Construct grouped outer splits according to dataset type."""
    group_count = len(dataset._group_spectrum_ids("group_id"))
    if dataset.name == "aalto_libs":
        return dataset.create_grouped_loocv_splits(group_by="group_id", name_prefix="outer")
    if group_count <= 5:
        return dataset.create_grouped_loocv_splits(group_by="group_id", name_prefix="outer")
    return dataset.create_grouped_kfold_splits(
        n_folds=min(5, group_count),
        random_seed=42,
        group_by="group_id",
        stratify_by="label_cardinality",
        name_prefix="outer",
    )


def build_inner_splits(train_dataset: BenchmarkDataset) -> List[DataSplit]:
    group_count = len(train_dataset._group_spectrum_ids("group_id"))
    if group_count <= 1:
        return []
    if group_count < 5:
        return train_dataset.create_grouped_loocv_splits(group_by="group_id", name_prefix="inner")
    return train_dataset.create_grouped_kfold_splits(
        n_folds=min(5, group_count),
        random_seed=43,
        group_by="group_id",
        stratify_by="label_cardinality",
        name_prefix="inner",
    )


@dataclass
class IDWorkflowSpec:
    name: str
    parameter_grid: List[Dict[str, Any]]
    build_predictor: Callable[
        ["UnifiedBenchmarkContext", List[str], Dict[str, Any]],
        Callable[[BenchmarkSpectrum], ElementIdentificationResult],
    ]
    config_name: Callable[[Dict[str, Any]], str]


@dataclass
class CompositionWorkflowSpec:
    name: str
    parameter_grid: List[Dict[str, Any]]
    fit_predictor: Callable[
        ["UnifiedBenchmarkContext", Sequence[BenchmarkSpectrum], Dict[str, Any]],
        Callable[
            [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]],
            Dict[str, Any],
        ],
    ]
    config_name: Callable[[Dict[str, Any]], str]
    requires_training: bool = False


@dataclass
class IDEvaluationRecord:
    dataset_id: str
    spectrum_id: str
    group_id: Optional[str]
    specimen_id: Optional[str]
    instrument_id: Optional[str]
    truth_type: str
    rp_estimate: Optional[float]
    label_cardinality: Optional[int]
    spectrum_kind: Optional[str]
    workflow_name: str
    outer_split_id: str
    tuning_split_id: Optional[str]
    config_name: str
    elapsed_seconds: float
    true_elements: List[str]
    predicted_elements: List[str]
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    jaccard: float
    hamming_loss: float
    exact_match: bool
    false_positives_per_spectrum: int
    scored: bool = True
    failure_reason: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionEvaluationRecord:
    dataset_id: str
    spectrum_id: str
    group_id: Optional[str]
    specimen_id: Optional[str]
    instrument_id: Optional[str]
    truth_type: str
    rp_estimate: Optional[float]
    label_cardinality: Optional[int]
    spectrum_kind: Optional[str]
    id_workflow_name: str
    composition_workflow_name: str
    outer_split_id: str
    tuning_split_id: Optional[str]
    id_config_name: str
    composition_config_name: str
    elapsed_seconds: float
    candidate_elements: List[str]
    true_composition: Dict[str, float]
    predicted_composition: Dict[str, float]
    aitchison: Optional[float]
    rmse: Optional[float]
    temperature_error_frac: Optional[float]
    ne_error_frac: Optional[float]
    closure_residual: Optional[float]
    error_tier: Optional[str] = None
    per_element_absolute_error: Dict[str, float] = field(default_factory=dict)
    per_element_relative_error: Dict[str, float] = field(default_factory=dict)
    scored: bool = True
    failure_reason: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


def _empty_identification_result(
    algorithm: str,
    warnings: Optional[List[str]] = None,
) -> ElementIdentificationResult:
    from cflibs.inversion.element_id import ElementIdentificationResult

    return ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm=algorithm,
        parameters={},
        warnings=warnings or [],
    )


class UnifiedBenchmarkContext:
    def __init__(self, db_path: Path, basis_dir: Optional[Path] = None):
        self.db_path = Path(db_path)
        self.basis_dir = Path(basis_dir) if basis_dir is not None else None
        self._basis_cache: Dict[str, Any] = {}
        self._basis_manifest: Optional[List[Tuple[float, Path]]] = None

    def _basis_files(self) -> List[Tuple[float, Path]]:
        if self._basis_manifest is not None:
            return self._basis_manifest
        files: List[Tuple[float, Path]] = []
        if self.basis_dir is not None and self.basis_dir.exists():
            for path in sorted(self.basis_dir.glob("basis_fwhm_*nm.h5")):
                match = re.search(r"basis_fwhm_([0-9.]+)nm\.h5$", path.name)
                if match:
                    files.append((float(match.group(1)), path))
        self._basis_manifest = files
        return files

    def basis_for_rp(self, rp_estimate: Optional[float]) -> Tuple[Any, float, float]:
        from cflibs.manifold.basis_library import BasisLibrary

        basis_files = self._basis_files()
        if not basis_files:
            raise FileNotFoundError("No basis libraries found in basis_dir")

        target_fwhm = 0.5 if rp_estimate is None or rp_estimate <= 0 else 550.0 / float(rp_estimate)
        selected_fwhm, selected_path = min(basis_files, key=lambda item: abs(item[0] - target_fwhm))
        cache_key = str(selected_path.resolve())
        if cache_key not in self._basis_cache:
            self._basis_cache[cache_key] = BasisLibrary(str(selected_path))
        return self._basis_cache[cache_key], selected_fwhm, abs(selected_fwhm - target_fwhm)


def _alias_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    if quick:
        return [
            {
                "detection_threshold": 0.03,
                "intensity_threshold_factor": 3.0,
                "chance_window_scale": 0.3,
                "max_lines_per_element": 30,
            },
            {
                "detection_threshold": 0.05,
                "intensity_threshold_factor": 3.0,
                "chance_window_scale": 0.4,
                "max_lines_per_element": 30,
            },
        ]
    configs: List[Dict[str, Any]] = []
    for dt in [0.02, 0.03, 0.05]:
        for itf in [3.0, 3.5]:
            for cws in [0.3, 0.4]:
                configs.append(
                    {
                        "detection_threshold": dt,
                        "intensity_threshold_factor": itf,
                        "chance_window_scale": cws,
                        "max_lines_per_element": 30,
                    }
                )
    return configs


def _comb_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    if quick:
        return [
            {
                "min_correlation": 0.08,
                "tooth_activation_threshold": 0.35,
                "relative_threshold_scale": 1.4,
            }
        ]
    configs: List[Dict[str, Any]] = []
    for min_correlation in [0.05, 0.08]:
        for tooth_activation_threshold in [0.30, 0.35, 0.40]:
            for relative_threshold_scale in [1.2, 1.4]:
                configs.append(
                    {
                        "min_correlation": min_correlation,
                        "tooth_activation_threshold": tooth_activation_threshold,
                        "relative_threshold_scale": relative_threshold_scale,
                    }
                )
    return configs


def _correlation_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    if quick:
        return [
            {"min_confidence": 0.008, "relative_threshold_scale": 1.2, "min_line_strength": 1000.0}
        ]
    configs: List[Dict[str, Any]] = []
    for min_confidence in [0.005, 0.008, 0.015]:
        for threshold_scale in [1.0, 1.2]:
            configs.append(
                {
                    "min_confidence": min_confidence,
                    "relative_threshold_scale": threshold_scale,
                    "min_line_strength": 1000.0,
                }
            )
    return configs


def _nnls_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    snrs = [1.0, 1.5, 2.0] if quick else [1.0, 1.5, 2.0, 2.5, 3.0]
    cdegs = [2, 3] if quick else [2, 3, 4]
    temps = [8000.0] if quick else [6000.0, 8000.0, 10000.0]
    configs: List[Dict[str, Any]] = []
    for snr in snrs:
        for cdeg in cdegs:
            for temp in temps:
                configs.append(
                    {
                        "detection_snr": snr,
                        "continuum_degree": cdeg,
                        "fallback_T_K": temp,
                    }
                )
    return configs


def _hybrid_workflow_configs(require_both: bool, quick: bool) -> List[Dict[str, Any]]:
    nnls_snrs = [1.0, 1.5] if quick else [1.0, 1.5, 2.0]
    alias_thresholds = [0.03, 0.05] if quick else [0.03, 0.05, 0.10]
    return [
        {
            "nnls_detection_snr": nnls_snr,
            "alias_detection_threshold": alias_threshold,
            "require_both": require_both,
        }
        for nnls_snr in nnls_snrs
        for alias_threshold in alias_thresholds
    ]


def _nnls_concentration_configs(quick: bool) -> List[Dict[str, Any]]:
    thresholds = [0.005, 0.02] if quick else [0.001, 0.005, 0.01, 0.02, 0.05]
    cdegs = [2] if quick else [2, 3]
    return [
        {"concentration_threshold": threshold, "continuum_degree": cdeg}
        for threshold in thresholds
        for cdeg in cdegs
    ]


def _config_name(config: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(config):
        parts.append(f"{key}={config[key]}")
    return ",".join(parts)


def _estimate_rp_for_spectrum(spectrum: BenchmarkSpectrum) -> float:
    if spectrum.rp_estimate is not None and np.isfinite(spectrum.rp_estimate):
        return float(spectrum.rp_estimate)
    return _estimate_effective_rp_lazy(spectrum.wavelength_nm, spectrum.intensity)


def _build_alias_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                intensity_threshold_factor=float(config["intensity_threshold_factor"]),
                detection_threshold=float(config["detection_threshold"]),
                chance_window_scale=float(config["chance_window_scale"]),
                max_lines_per_element=int(config["max_lines_per_element"]),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_comb_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.comb_identifier import CombIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = CombIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                min_correlation=float(config["min_correlation"]),
                tooth_activation_threshold=float(config["tooth_activation_threshold"]),
                relative_threshold_scale=float(config["relative_threshold_scale"]),
                min_aki_gk=3000.0,
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_correlation_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.correlation_identifier import CorrelationIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = CorrelationIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                min_confidence=float(config["min_confidence"]),
                relative_threshold_scale=float(config["relative_threshold_scale"]),
                min_line_strength=float(config["min_line_strength"]),
                T_range_K=(5000, 15000),
                T_steps=7,
                n_e_range_cm3=(1e15, 5e17),
                n_e_steps=4,
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity, mode="classic")
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_nnls_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=float(config["detection_snr"]),
            continuum_degree=int(config["continuum_degree"]),
            fallback_T_K=float(config["fallback_T_K"]),
            fallback_ne_cm3=1e17,
        )
        result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
        result.parameters["basis_fwhm_nm"] = basis_fwhm
        result.parameters["basis_fwhm_mismatch_nm"] = mismatch
        result.parameters["candidate_elements"] = list(candidate_elements)
        return result

    return predictor


def _build_hybrid_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.hybrid_identifier import HybridIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
            identifier = HybridIdentifier(
                atomic_db=db,
                basis_library=basis,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                nnls_detection_snr=float(config["nnls_detection_snr"]),
                alias_detection_threshold=float(config["alias_detection_threshold"]),
                require_both=bool(config["require_both"]),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["basis_fwhm_nm"] = basis_fwhm
            result.parameters["basis_fwhm_mismatch_nm"] = mismatch
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_voigt_alias_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    from cflibs.inversion.deconvolution import deconvolve_peaks
    from cflibs.inversion.preprocessing import estimate_baseline

    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            baseline = estimate_baseline(spectrum.wavelength_nm, spectrum.intensity)
            corrected = np.maximum(spectrum.intensity - baseline, 0.0)
            if find_peaks is None:
                cleaned = corrected
            else:
                threshold = (
                    np.percentile(corrected[corrected > 0], 70) if np.any(corrected > 0) else 0.0
                )
                peak_indices, _ = find_peaks(corrected, height=threshold, distance=5)
                if len(peak_indices) == 0:
                    cleaned = corrected
                else:
                    peak_wls = spectrum.wavelength_nm[peak_indices]
                    rp = _estimate_rp_for_spectrum(spectrum)
                    fwhm_est = float(np.median(spectrum.wavelength_nm) / max(rp, 1.0))
                    try:
                        deconv = deconvolve_peaks(
                            spectrum.wavelength_nm,
                            corrected,
                            peak_wls,
                            fwhm_est,
                            grouping_factor=2.0,
                            margin_factor=3.0,
                            use_jax=False,
                        )
                        cleaned = np.maximum(deconv.fitted_spectrum, 0.0)
                    except Exception:  # noqa: BLE001
                        cleaned = corrected

            rp = _estimate_rp_for_spectrum(spectrum)
            alias = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=rp,
                intensity_threshold_factor=3.0,
                detection_threshold=float(config["detection_threshold"]),
                chance_window_scale=0.4,
                max_lines_per_element=30,
            )
            result = alias.identify(
                spectrum.wavelength_nm,
                cleaned + np.median(spectrum.intensity) * 0.01,
            )
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_nnls_concentration_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=0.0,
            continuum_degree=int(config["continuum_degree"]),
            fallback_T_K=8000.0,
            fallback_ne_cm3=1e17,
        )
        result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
        threshold = float(config["concentration_threshold"])
        for element in result.all_elements:
            concentration = float(element.metadata.get("concentration_estimate", 0.0))
            element.detected = concentration >= threshold
        result.detected_elements = [element for element in result.all_elements if element.detected]
        result.rejected_elements = [
            element for element in result.all_elements if not element.detected
        ]
        result.algorithm = "nnls_concentration_threshold"
        result.parameters["concentration_threshold"] = threshold
        result.parameters["basis_fwhm_nm"] = basis_fwhm
        result.parameters["basis_fwhm_mismatch_nm"] = mismatch
        result.parameters["candidate_elements"] = list(candidate_elements)
        return result

    return predictor


def build_id_workflow_registry(quick: bool = False) -> Dict[str, IDWorkflowSpec]:
    return {
        "alias": IDWorkflowSpec(
            "alias", _alias_workflow_configs(quick), _build_alias_predictor, _config_name
        ),
        "comb": IDWorkflowSpec(
            "comb", _comb_workflow_configs(quick), _build_comb_predictor, _config_name
        ),
        "correlation": IDWorkflowSpec(
            "correlation",
            _correlation_workflow_configs(quick),
            _build_correlation_predictor,
            _config_name,
        ),
        "spectral_nnls": IDWorkflowSpec(
            "spectral_nnls", _nnls_workflow_configs(quick), _build_nnls_predictor, _config_name
        ),
        "hybrid_intersect": IDWorkflowSpec(
            "hybrid_intersect",
            _hybrid_workflow_configs(True, quick),
            _build_hybrid_predictor,
            _config_name,
        ),
        "hybrid_union": IDWorkflowSpec(
            "hybrid_union",
            _hybrid_workflow_configs(False, quick),
            _build_hybrid_predictor,
            _config_name,
        ),
        "voigt_alias": IDWorkflowSpec(
            "voigt_alias",
            (
                [{"detection_threshold": 0.03}]
                if quick
                else [{"detection_threshold": 0.03}, {"detection_threshold": 0.05}]
            ),
            _build_voigt_alias_predictor,
            _config_name,
        ),
        "nnls_concentration_threshold": IDWorkflowSpec(
            "nnls_concentration_threshold",
            _nnls_concentration_configs(quick),
            _build_nnls_concentration_predictor,
            _config_name,
        ),
        **_bayesian_sparse_entry(quick),
    }


def _bayesian_sparse_entry(quick: bool) -> Dict[str, "IDWorkflowSpec"]:
    """Lazy-load the bayesian_sparse workflow (requires JAX + NumPyro)."""
    try:
        # Gate on actual heavy deps, not just the wrapper module
        import jax  # noqa: F401
        import numpyro  # noqa: F401

        from cflibs.benchmark.bayesian_sparse_id import (
            bayesian_sparse_config_name,
            bayesian_sparse_workflow_configs,
            build_bayesian_sparse_predictor,
        )

        return {
            "bayesian_sparse": IDWorkflowSpec(
                "bayesian_sparse",
                bayesian_sparse_workflow_configs(quick),
                build_bayesian_sparse_predictor,
                bayesian_sparse_config_name,
            )
        }
    except ImportError:
        return {}


def _fit_iterative_pipeline(
    _context: UnifiedBenchmarkContext,
    _train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
]:
    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional[ElementIdentificationResult],
    ) -> Dict[str, Any]:
        from cflibs.atomic.database import AtomicDatabase

        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for iterative composition workflow")
        with AtomicDatabase(str(_context.db_path)) as db:
            result = _run_boltzmann_pipeline_lazy(
                {
                    "wavelength": spectrum.wavelength_nm,
                    "intensity": spectrum.intensity,
                    "ground_truth": spectrum.true_composition,
                },
                db=db,
                fit_method=config["fit_method"],
                closure_mode=str(config["closure_mode"]),
                elements=elements,
            )
        if result is None:
            raise RuntimeError("Iterative composition workflow failed")
        return {"concentrations": result}

    return predictor


def _fit_joint_optimizer_pipeline(
    _context: UnifiedBenchmarkContext,
    _train_spectra: Sequence[BenchmarkSpectrum],
    _config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
]:
    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional[ElementIdentificationResult],
    ) -> Dict[str, Any]:
        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for joint optimizer")
        return _pipeline_joint_softmax_lazy(spectrum.wavelength_nm, spectrum.intensity, elements)

    return predictor


def _fit_hybrid_manifold_pipeline(
    _context: UnifiedBenchmarkContext,
    _train_spectra: Sequence[BenchmarkSpectrum],
    _config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
]:
    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional[ElementIdentificationResult],
    ) -> Dict[str, Any]:
        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for hybrid manifold workflow")
        return _pipeline_hybrid_manifold_lazy(spectrum.wavelength_nm, spectrum.intensity, elements)

    return predictor


def _fit_pls_pipeline(
    _context: UnifiedBenchmarkContext,
    train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
]:
    from cflibs.inversion.pls import PLSRegression

    train_spectra = [spec for spec in train_spectra if spec.truth_type != TruthType.BLIND]
    if len(train_spectra) < 2:
        raise ValueError("Need at least two supervised spectra to train PLS")

    elements = sorted({el for spec in train_spectra for el in spec.true_composition.keys()})
    X_train = np.array([spec.intensity for spec in train_spectra], dtype=float)
    Y_train = np.array(
        [[spec.true_composition.get(el, 0.0) for el in elements] for spec in train_spectra],
        dtype=float,
    )
    max_components = min(
        int(config["n_components"]), X_train.shape[0] - 1, X_train.shape[1], len(elements)
    )
    max_components = max(max_components, 1)
    pls = PLSRegression(n_components=max_components)
    pls.fit(X_train, Y_train)

    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional[ElementIdentificationResult],
    ) -> Dict[str, Any]:
        result = pls.predict(np.asarray(spectrum.intensity, dtype=float).reshape(1, -1))
        prediction = np.clip(result.predictions[0], 0.0, None)
        concentrations = {el: float(prediction[idx]) for idx, el in enumerate(elements)}
        if candidate_elements:
            candidate_set = set(candidate_elements)
            concentrations = {
                el: value if el in candidate_set else 0.0 for el, value in concentrations.items()
            }
        total = sum(concentrations.values())
        if total > 0:
            concentrations = {el: value / total for el, value in concentrations.items()}
        return {"concentrations": concentrations}

    return predictor


def build_composition_workflow_registry(quick: bool = False) -> Dict[str, CompositionWorkflowSpec]:
    from cflibs.inversion.boltzmann import FitMethod

    iterative_configs = [
        {"fit_method": FitMethod.SIGMA_CLIP, "closure_mode": "standard"},
        {"fit_method": FitMethod.SIGMA_CLIP, "closure_mode": "ilr"},
    ]
    if not quick:
        iterative_configs.extend(
            [
                {"fit_method": FitMethod.RANSAC, "closure_mode": "standard"},
                {"fit_method": FitMethod.RANSAC, "closure_mode": "ilr"},
                {"fit_method": FitMethod.HUBER, "closure_mode": "standard"},
                {"fit_method": FitMethod.HUBER, "closure_mode": "ilr"},
            ]
        )

    return {
        "iterative": CompositionWorkflowSpec(
            "iterative", iterative_configs, _fit_iterative_pipeline, _config_name
        ),
        "joint_softmax": CompositionWorkflowSpec(
            "joint_softmax", [{}], _fit_joint_optimizer_pipeline, _config_name
        ),
        "hybrid_manifold": CompositionWorkflowSpec(
            "hybrid_manifold", [{}], _fit_hybrid_manifold_pipeline, _config_name
        ),
        "pls": CompositionWorkflowSpec(
            "pls",
            [{"n_components": 3}, {"n_components": 10}] if not quick else [{"n_components": 3}],
            _fit_pls_pipeline,
            _config_name,
            requires_training=True,
        ),
    }


def _composition_error_tier(aitchison: Optional[float]) -> Optional[str]:
    if aitchison is None or not np.isfinite(aitchison):
        return None
    if aitchison <= 0.15:
        return "excellent"
    if aitchison <= 0.35:
        return "good"
    if aitchison <= 0.60:
        return "fair"
    return "poor"


def _score_identification(
    spectrum: BenchmarkSpectrum,
    workflow_name: str,
    outer_split_id: str,
    tuning_split_id: Optional[str],
    config_name: str,
    result: ElementIdentificationResult,
    elapsed_seconds: float,
) -> IDEvaluationRecord:
    true_elements = derive_truth_elements(spectrum)
    predicted_elements = sorted({element.element for element in result.detected_elements})
    candidate_elements = list(getattr(result, "parameters", {}).get("candidate_elements", []))
    if not candidate_elements:
        candidate_elements = (
            list(result.parameters.get("elements", [])) if hasattr(result, "parameters") else []
        )
    if not candidate_elements:
        candidate_elements = list(spectrum.annotations.get("candidate_elements", []))
    if not candidate_elements:
        candidate_elements = list(spectrum.true_composition.keys()) or []
    if not candidate_elements:
        candidate_elements = list(predicted_elements)
    evaluated_space = set(
        candidate_elements
        or predicted_elements
        or true_elements
        or spectrum.true_composition.keys()
    )

    if spectrum.truth_type == TruthType.BLIND:
        return IDEvaluationRecord(
            dataset_id=spectrum.dataset_id or "unknown",
            spectrum_id=spectrum.spectrum_id,
            group_id=spectrum.group_id,
            specimen_id=spectrum.specimen_id,
            instrument_id=spectrum.instrument_id,
            truth_type=spectrum.truth_type.value,
            rp_estimate=spectrum.rp_estimate,
            label_cardinality=spectrum.label_cardinality,
            spectrum_kind=spectrum.spectrum_kind,
            workflow_name=workflow_name,
            outer_split_id=outer_split_id,
            tuning_split_id=tuning_split_id,
            config_name=config_name,
            elapsed_seconds=elapsed_seconds,
            true_elements=[],
            predicted_elements=predicted_elements,
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            jaccard=0.0,
            hamming_loss=0.0,
            exact_match=False,
            false_positives_per_spectrum=0,
            scored=False,
            annotations=dict(result.parameters),
        )

    tp = len(set(predicted_elements) & set(true_elements))
    fp = len(set(predicted_elements) - set(true_elements))
    fn = len(set(true_elements) - set(predicted_elements))
    tn = len(evaluated_space - set(true_elements) - set(predicted_elements))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    union = len(set(predicted_elements) | set(true_elements))
    jaccard = _safe_ratio(tp, union)
    hamming_loss = _safe_ratio(fp + fn, max(len(evaluated_space), 1))

    return IDEvaluationRecord(
        dataset_id=spectrum.dataset_id or "unknown",
        spectrum_id=spectrum.spectrum_id,
        group_id=spectrum.group_id,
        specimen_id=spectrum.specimen_id,
        instrument_id=spectrum.instrument_id,
        truth_type=spectrum.truth_type.value,
        rp_estimate=spectrum.rp_estimate,
        label_cardinality=spectrum.label_cardinality,
        spectrum_kind=spectrum.spectrum_kind,
        workflow_name=workflow_name,
        outer_split_id=outer_split_id,
        tuning_split_id=tuning_split_id,
        config_name=config_name,
        elapsed_seconds=elapsed_seconds,
        true_elements=true_elements,
        predicted_elements=predicted_elements,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        hamming_loss=hamming_loss,
        exact_match=set(predicted_elements) == set(true_elements),
        false_positives_per_spectrum=fp,
        scored=True,
        annotations=dict(result.parameters),
    )


def evaluate_id_workflow(
    spectra: Sequence[BenchmarkSpectrum],
    workflow: IDWorkflowSpec,
    predictor: Callable[[BenchmarkSpectrum], ElementIdentificationResult],
    outer_split_id: str,
    tuning_split_id: Optional[str],
    config_name: str,
) -> List[IDEvaluationRecord]:
    records: List[IDEvaluationRecord] = []
    for spectrum in spectra:
        start = time.perf_counter()
        try:
            result = predictor(spectrum)
        except Exception as exc:  # noqa: BLE001
            records.append(
                IDEvaluationRecord(
                    dataset_id=spectrum.dataset_id or "unknown",
                    spectrum_id=spectrum.spectrum_id,
                    group_id=spectrum.group_id,
                    specimen_id=spectrum.specimen_id,
                    instrument_id=spectrum.instrument_id,
                    truth_type=spectrum.truth_type.value,
                    rp_estimate=spectrum.rp_estimate,
                    label_cardinality=spectrum.label_cardinality,
                    spectrum_kind=spectrum.spectrum_kind,
                    workflow_name=workflow.name,
                    outer_split_id=outer_split_id,
                    tuning_split_id=tuning_split_id,
                    config_name=config_name,
                    elapsed_seconds=time.perf_counter() - start,
                    true_elements=derive_truth_elements(spectrum),
                    predicted_elements=[],
                    tp=0,
                    fp=0,
                    fn=0,
                    tn=0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    jaccard=0.0,
                    hamming_loss=0.0,
                    exact_match=False,
                    false_positives_per_spectrum=0,
                    scored=spectrum.truth_type != TruthType.BLIND,
                    failure_reason=str(exc),
                )
            )
            continue
        elapsed = time.perf_counter() - start
        records.append(
            _score_identification(
                spectrum=spectrum,
                workflow_name=workflow.name,
                outer_split_id=outer_split_id,
                tuning_split_id=tuning_split_id,
                config_name=config_name,
                result=result,
                elapsed_seconds=elapsed,
            )
        )
    return records


def tune_id_workflow(
    context: UnifiedBenchmarkContext,
    workflow: IDWorkflowSpec,
    train_dataset: BenchmarkDataset,
    candidate_elements: List[str],
    outer_split_id: str,
) -> Tuple[Dict[str, Any], str, Optional[str], List[Dict[str, Any]]]:
    inner_splits = build_inner_splits(train_dataset)
    tuning_split_id = f"{outer_split_id}__inner" if inner_splits else None
    if not inner_splits or len(workflow.parameter_grid) == 1:
        config = dict(workflow.parameter_grid[0])
        return config, workflow.config_name(config), tuning_split_id, []

    scores: List[Dict[str, Any]] = []
    best_config = dict(workflow.parameter_grid[0])
    best_score = float("-inf")
    best_tiebreak = float("-inf")

    for config in workflow.parameter_grid:
        fold_scores: List[float] = []
        precisions: List[float] = []
        for split in inner_splits:
            validation_dataset = subset_dataset(
                train_dataset,
                split.test_ids,
                name=f"{train_dataset.name}_{split.name}_validation",
            )
            predictor = workflow.build_predictor(context, candidate_elements, config)
            records = evaluate_id_workflow(
                validation_dataset.spectra,
                workflow=workflow,
                predictor=predictor,
                outer_split_id=split.name,
                tuning_split_id=tuning_split_id,
                config_name=workflow.config_name(config),
            )
            overall = summarize_id_records(records)["overall"]
            if workflow.name not in overall:
                continue
            summary = overall[workflow.name]
            fold_scores.append(float(summary["micro_f1"]))
            precisions.append(float(summary["micro_precision"]))
        score = float(np.mean(fold_scores)) if fold_scores else 0.0
        tie = float(np.mean(precisions)) if precisions else 0.0
        config_name = workflow.config_name(config)
        scores.append(
            {"config_name": config_name, "mean_inner_f1": score, "mean_inner_precision": tie}
        )
        if score > best_score or (math.isclose(score, best_score) and tie > best_tiebreak):
            best_score = score
            best_tiebreak = tie
            best_config = dict(config)

    return best_config, workflow.config_name(best_config), tuning_split_id, scores


def _coerce_composition_prediction(
    prediction: Dict[str, Any],
    candidate_elements: Sequence[str],
) -> Dict[str, float]:
    concentrations = dict(prediction.get("concentrations", {}))
    if candidate_elements:
        candidate_set = set(candidate_elements)
        concentrations = {
            element: value if element in candidate_set else 0.0
            for element, value in concentrations.items()
        }
    total = sum(float(value) for value in concentrations.values())
    if total > 0:
        concentrations = {
            element: float(value) / total for element, value in concentrations.items()
        }
    return concentrations


def _compute_fractional_error(
    observed_value: Optional[float],
    predicted_value: Optional[float],
) -> Optional[float]:
    if not observed_value or not predicted_value:
        return None
    return abs(float(predicted_value) - float(observed_value)) / max(float(observed_value), 1e-12)


def _build_composition_success_record(
    spectrum: BenchmarkSpectrum,
    id_workflow_name: str,
    id_config_name: str,
    composition_workflow_name: str,
    composition_config_name: str,
    outer_split_id: str,
    tuning_split_id: Optional[str],
    elapsed_seconds: float,
    candidate_elements: Sequence[str],
    concentrations: Dict[str, float],
    prediction: Dict[str, Any],
) -> CompositionEvaluationRecord:
    true_comp = dict(spectrum.true_composition)
    aitchison = float(aitchison_distance(true_comp, concentrations))
    rmse = float(rmse_composition(true_comp, concentrations))
    per_element = per_element_error(true_comp, concentrations)
    return CompositionEvaluationRecord(
        dataset_id=spectrum.dataset_id or "unknown",
        spectrum_id=spectrum.spectrum_id,
        group_id=spectrum.group_id,
        specimen_id=spectrum.specimen_id,
        instrument_id=spectrum.instrument_id,
        truth_type=spectrum.truth_type.value,
        rp_estimate=spectrum.rp_estimate,
        label_cardinality=spectrum.label_cardinality,
        spectrum_kind=spectrum.spectrum_kind,
        id_workflow_name=id_workflow_name,
        composition_workflow_name=composition_workflow_name,
        outer_split_id=outer_split_id,
        tuning_split_id=tuning_split_id,
        id_config_name=id_config_name,
        composition_config_name=composition_config_name,
        elapsed_seconds=elapsed_seconds,
        candidate_elements=list(candidate_elements),
        true_composition=true_comp,
        predicted_composition=concentrations,
        aitchison=aitchison,
        rmse=rmse,
        temperature_error_frac=_compute_fractional_error(
            spectrum.plasma_temperature_K, prediction.get("temperature_K")
        ),
        ne_error_frac=_compute_fractional_error(
            spectrum.electron_density_cm3, prediction.get("electron_density_cm3")
        ),
        closure_residual=abs(sum(concentrations.values()) - 1.0),
        error_tier=_composition_error_tier(aitchison),
        per_element_absolute_error={
            element: float(errors[0]) for element, errors in per_element.items()
        },
        per_element_relative_error={
            element: float(errors[1]) for element, errors in per_element.items()
        },
        annotations={
            key: value for key, value in prediction.items() if key not in {"concentrations"}
        },
    )


def _build_composition_failure_record(
    spectrum: BenchmarkSpectrum,
    id_workflow_name: str,
    id_config_name: str,
    composition_workflow_name: str,
    composition_config_name: str,
    outer_split_id: str,
    tuning_split_id: Optional[str],
    elapsed_seconds: float,
    failure_reason: str,
) -> CompositionEvaluationRecord:
    return CompositionEvaluationRecord(
        dataset_id=spectrum.dataset_id or "unknown",
        spectrum_id=spectrum.spectrum_id,
        group_id=spectrum.group_id,
        specimen_id=spectrum.specimen_id,
        instrument_id=spectrum.instrument_id,
        truth_type=spectrum.truth_type.value,
        rp_estimate=spectrum.rp_estimate,
        label_cardinality=spectrum.label_cardinality,
        spectrum_kind=spectrum.spectrum_kind,
        id_workflow_name=id_workflow_name,
        composition_workflow_name=composition_workflow_name,
        outer_split_id=outer_split_id,
        tuning_split_id=tuning_split_id,
        id_config_name=id_config_name,
        composition_config_name=composition_config_name,
        elapsed_seconds=elapsed_seconds,
        candidate_elements=[],
        true_composition=dict(spectrum.true_composition),
        predicted_composition={},
        aitchison=None,
        rmse=None,
        temperature_error_frac=None,
        ne_error_frac=None,
        closure_residual=None,
        error_tier=None,
        failure_reason=failure_reason,
    )


def evaluate_composition_workflow(
    spectra: Sequence[BenchmarkSpectrum],
    id_workflow_name: str,
    id_config_name: str,
    id_predictor: Callable[[BenchmarkSpectrum], ElementIdentificationResult],
    composition_workflow: CompositionWorkflowSpec,
    composition_predictor: Callable[
        [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
    ],
    outer_split_id: str,
    tuning_split_id: Optional[str],
    composition_config_name: str,
) -> List[CompositionEvaluationRecord]:
    records: List[CompositionEvaluationRecord] = []

    for spectrum in spectra:
        if spectrum.truth_type == TruthType.BLIND:
            continue
        start = time.perf_counter()
        try:
            id_result = id_predictor(spectrum)
            candidate_elements = sorted(
                {element.element for element in id_result.detected_elements}
            )
            if not candidate_elements:
                raise ValueError(
                    "No identified candidate elements available for composition estimation"
                )
            prediction = composition_predictor(spectrum, candidate_elements, id_result)
            concentrations = _coerce_composition_prediction(prediction, candidate_elements)
            if not concentrations:
                raise ValueError("Composition workflow returned no concentrations")
            records.append(
                _build_composition_success_record(
                    spectrum=spectrum,
                    id_workflow_name=id_workflow_name,
                    id_config_name=id_config_name,
                    composition_workflow_name=composition_workflow.name,
                    composition_config_name=composition_config_name,
                    outer_split_id=outer_split_id,
                    tuning_split_id=tuning_split_id,
                    elapsed_seconds=time.perf_counter() - start,
                    candidate_elements=candidate_elements,
                    concentrations=concentrations,
                    prediction=prediction,
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                _build_composition_failure_record(
                    spectrum=spectrum,
                    id_workflow_name=id_workflow_name,
                    id_config_name=id_config_name,
                    composition_workflow_name=composition_workflow.name,
                    composition_config_name=composition_config_name,
                    outer_split_id=outer_split_id,
                    tuning_split_id=tuning_split_id,
                    elapsed_seconds=time.perf_counter() - start,
                    failure_reason=str(exc),
                )
            )
    return records


def tune_composition_workflow(
    context: UnifiedBenchmarkContext,
    workflow: CompositionWorkflowSpec,
    train_dataset: BenchmarkDataset,
    id_workflow_name: str,
    id_config_name: str,
    id_predictor: Callable[[BenchmarkSpectrum], ElementIdentificationResult],
    outer_split_id: str,
) -> Tuple[Dict[str, Any], str, Optional[str], List[Dict[str, Any]]]:
    inner_splits = build_inner_splits(train_dataset)
    tuning_split_id = f"{outer_split_id}__composition_inner" if inner_splits else None
    if not inner_splits or len(workflow.parameter_grid) == 1:
        config = dict(workflow.parameter_grid[0])
        return config, workflow.config_name(config), tuning_split_id, []

    scores: List[Dict[str, Any]] = []
    best_config = dict(workflow.parameter_grid[0])
    best_score = float("inf")
    best_tiebreak = float("inf")

    for config in workflow.parameter_grid:
        fold_aitchisons: List[float] = []
        fold_rmses: List[float] = []
        failed_folds = 0
        for split in inner_splits:
            inner_train = subset_dataset(
                train_dataset,
                split.train_ids,
                name=f"{train_dataset.name}_{split.name}_train",
            )
            validation = [
                train_dataset.get_spectrum(spec_id)
                for spec_id in split.test_ids
                if train_dataset.get_spectrum(spec_id).truth_type != TruthType.BLIND
            ]
            if not validation:
                continue
            try:
                predictor = workflow.fit_predictor(
                    context,
                    [spec for spec in inner_train.spectra if spec.truth_type != TruthType.BLIND],
                    config,
                )
            except Exception:  # noqa: BLE001
                failed_folds += 1
                continue
            records = evaluate_composition_workflow(
                validation,
                id_workflow_name=id_workflow_name,
                id_config_name=id_config_name,
                id_predictor=id_predictor,
                composition_workflow=workflow,
                composition_predictor=predictor,
                outer_split_id=split.name,
                tuning_split_id=tuning_split_id,
                composition_config_name=workflow.config_name(config),
            )
            scored = [
                record for record in records if record.scored and record.aitchison is not None
            ]
            if not scored:
                failed_folds += 1
                continue
            fold_aitchisons.append(float(np.mean([record.aitchison for record in scored])))
            fold_rmses.append(
                float(np.mean([record.rmse for record in scored if record.rmse is not None]))
            )

        score = float(np.mean(fold_aitchisons)) if fold_aitchisons else float("inf")
        tie = float(np.mean(fold_rmses)) if fold_rmses else float("inf")
        config_name = workflow.config_name(config)
        scores.append(
            {
                "config_name": config_name,
                "mean_inner_aitchison": score,
                "mean_inner_rmse": tie,
                "failed_folds": failed_folds,
            }
        )
        if score < best_score or (math.isclose(score, best_score) and tie < best_tiebreak):
            best_score = score
            best_tiebreak = tie
            best_config = dict(config)

    return best_config, workflow.config_name(best_config), tuning_split_id, scores


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.zeros(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        selection = rng.integers(0, array.size, size=array.size)
        means[idx] = float(np.mean(array[selection]))
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.mean(array)),
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1.0 - alpha))),
    )


def _compute_workflow_aggregates(
    workflow_records: Sequence[IDEvaluationRecord],
) -> Dict[str, Any]:
    tp = sum(record.tp for record in workflow_records)
    fp = sum(record.fp for record in workflow_records)
    fn = sum(record.fn for record in workflow_records)
    tn = sum(record.tn for record in workflow_records)
    micro_precision = _safe_ratio(tp, tp + fp)
    micro_recall = _safe_ratio(tp, tp + fn)
    return {
        "n_spectra": len(workflow_records),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "macro_precision": (
            float(np.mean([record.precision for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "macro_recall": (
            float(np.mean([record.recall for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "macro_f1": (
            float(np.mean([record.f1 for record in workflow_records])) if workflow_records else 0.0
        ),
        "fpr": _safe_ratio(fp, fp + tn),
        "exact_match_rate": (
            float(np.mean([record.exact_match for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "jaccard": (
            float(np.mean([record.jaccard for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "hamming_loss": (
            float(np.mean([record.hamming_loss for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "false_positives_per_spectrum": (
            float(np.mean([record.false_positives_per_spectrum for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "latency_mean_s": (
            float(np.mean([record.elapsed_seconds for record in workflow_records]))
            if workflow_records
            else 0.0
        ),
        "latency_p95_s": (
            float(np.percentile([record.elapsed_seconds for record in workflow_records], 95))
            if workflow_records
            else 0.0
        ),
        "bootstrap_f1": bootstrap_ci([record.f1 for record in workflow_records]),
        "bootstrap_precision": bootstrap_ci([record.precision for record in workflow_records]),
    }


def _compute_per_element_stats(
    workflow_records: Sequence[IDEvaluationRecord],
) -> Dict[str, Dict[str, float]]:
    element_summary: Dict[str, Dict[str, float]] = {}
    element_names = sorted(
        {
            element
            for record in workflow_records
            for element in (record.true_elements + record.predicted_elements)
        }
    )
    for element in element_names:
        el_tp = sum(
            1
            for record in workflow_records
            if element in record.true_elements and element in record.predicted_elements
        )
        el_fp = sum(
            1
            for record in workflow_records
            if element not in record.true_elements and element in record.predicted_elements
        )
        el_fn = sum(
            1
            for record in workflow_records
            if element in record.true_elements and element not in record.predicted_elements
        )
        support = sum(1 for record in workflow_records if element in record.true_elements)
        element_summary[element] = {
            "precision": _safe_ratio(el_tp, el_tp + el_fp),
            "recall": _safe_ratio(el_tp, el_tp + el_fn),
            "support": support,
            "false_positives": el_fp,
        }
    return element_summary


def _compute_id_stratified_buckets(
    workflow_records: Sequence[IDEvaluationRecord],
    field_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    stratified: Dict[str, Dict[str, Any]] = {}
    for field_name in field_names:
        buckets: Dict[str, List[IDEvaluationRecord]] = {}
        for record in workflow_records:
            key = (
                _rp_bucket(record.rp_estimate)
                if field_name == "rp_bucket"
                else str(getattr(record, field_name))
            )
            buckets.setdefault(key, []).append(record)
        stratified[field_name] = {
            key: {
                "micro_f1": float(np.mean([rec.f1 for rec in bucket_records])),
                "micro_precision": float(np.mean([rec.precision for rec in bucket_records])),
                "micro_recall": float(np.mean([rec.recall for rec in bucket_records])),
                "n_spectra": len(bucket_records),
            }
            for key, bucket_records in buckets.items()
        }
    return stratified


def summarize_id_records(records: Sequence[IDEvaluationRecord]) -> Dict[str, Any]:
    scored = [record for record in records if record.scored]
    by_workflow: Dict[str, List[IDEvaluationRecord]] = {}
    for record in scored:
        by_workflow.setdefault(record.workflow_name, []).append(record)

    overall: Dict[str, Dict[str, Any]] = {}
    per_element: Dict[str, Dict[str, Dict[str, float]]] = {}
    stratified: Dict[str, Dict[str, Dict[str, Any]]] = {
        "dataset_id": {},
        "truth_type": {},
        "rp_bucket": {},
        "spectrum_kind": {},
        "label_cardinality": {},
    }

    for workflow_name, workflow_records in by_workflow.items():
        overall[workflow_name] = _compute_workflow_aggregates(workflow_records)
        per_element[workflow_name] = _compute_per_element_stats(workflow_records)
        workflow_buckets = _compute_id_stratified_buckets(workflow_records, stratified.keys())
        for field_name, bucket_data in workflow_buckets.items():
            stratified[field_name][workflow_name] = bucket_data

    return {"overall": overall, "per_element": per_element, "stratified": stratified}


def _compute_composition_overall(
    pair_records: Sequence[CompositionEvaluationRecord],
) -> Dict[str, Any]:
    aitchisons = [
        float(record.aitchison) for record in pair_records if record.aitchison is not None
    ]
    rmses = [float(record.rmse) for record in pair_records if record.rmse is not None]
    closure_residuals = [
        float(record.closure_residual)
        for record in pair_records
        if record.closure_residual is not None
    ]
    temperature_errors = [
        float(record.temperature_error_frac)
        for record in pair_records
        if record.temperature_error_frac is not None
    ]
    ne_errors = [
        float(record.ne_error_frac) for record in pair_records if record.ne_error_frac is not None
    ]
    tier_distribution: Dict[str, int] = {}
    for record in pair_records:
        if record.error_tier is not None:
            tier_distribution[record.error_tier] = tier_distribution.get(record.error_tier, 0) + 1
    return {
        "n_spectra": len(pair_records),
        "mean_aitchison": float(np.mean(aitchisons)) if aitchisons else float("inf"),
        "median_aitchison": float(np.median(aitchisons)) if aitchisons else float("inf"),
        "p95_aitchison": float(np.percentile(aitchisons, 95)) if aitchisons else float("inf"),
        "mean_rmse": float(np.mean(rmses)) if rmses else float("inf"),
        "mean_closure_residual": (
            float(np.mean(closure_residuals)) if closure_residuals else float("inf")
        ),
        "mean_temperature_error_frac": (
            float(np.mean(temperature_errors)) if temperature_errors else None
        ),
        "mean_ne_error_frac": float(np.mean(ne_errors)) if ne_errors else None,
        "latency_mean_s": (
            float(np.mean([record.elapsed_seconds for record in pair_records]))
            if pair_records
            else 0.0
        ),
        "latency_p95_s": (
            float(np.percentile([record.elapsed_seconds for record in pair_records], 95))
            if pair_records
            else 0.0
        ),
        "bootstrap_aitchison": bootstrap_ci(aitchisons),
        "tier_distribution": tier_distribution,
    }


def _compute_composition_per_element(
    pair_records: Sequence[CompositionEvaluationRecord],
) -> Dict[str, Dict[str, float]]:
    element_names = sorted(
        {
            element
            for record in pair_records
            for element in (
                list(record.per_element_absolute_error.keys())
                + list(record.per_element_relative_error.keys())
            )
        }
    )
    per_element: Dict[str, Dict[str, float]] = {}
    for element in element_names:
        abs_errors = [
            record.per_element_absolute_error[element]
            for record in pair_records
            if element in record.per_element_absolute_error
        ]
        rel_errors = [
            record.per_element_relative_error[element]
            for record in pair_records
            if element in record.per_element_relative_error
            and np.isfinite(record.per_element_relative_error[element])
        ]
        per_element[element] = {
            "mean_absolute_error": float(np.mean(abs_errors)) if abs_errors else float("nan"),
            "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else float("nan"),
            "support": len(abs_errors),
        }
    return per_element


def _compute_composition_stratified_buckets(
    pair_records: Sequence[CompositionEvaluationRecord],
    field_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    stratified: Dict[str, Dict[str, Any]] = {}
    for field_name in field_names:
        buckets: Dict[str, List[CompositionEvaluationRecord]] = {}
        for record in pair_records:
            bucket_key = (
                _rp_bucket(record.rp_estimate)
                if field_name == "rp_bucket"
                else str(getattr(record, field_name))
            )
            buckets.setdefault(bucket_key, []).append(record)
        stratified[field_name] = {
            bucket_key: {
                "mean_aitchison": float(
                    np.mean([rec.aitchison for rec in bucket_records if rec.aitchison is not None])
                ),
                "mean_rmse": float(
                    np.mean([rec.rmse for rec in bucket_records if rec.rmse is not None])
                ),
                "n_spectra": len(bucket_records),
            }
            for bucket_key, bucket_records in buckets.items()
        }
    return stratified


def summarize_composition_records(records: Sequence[CompositionEvaluationRecord]) -> Dict[str, Any]:
    scored = [record for record in records if record.scored and record.aitchison is not None]
    by_pair: Dict[str, List[CompositionEvaluationRecord]] = {}
    for record in scored:
        key = f"{record.id_workflow_name}__{record.composition_workflow_name}"
        by_pair.setdefault(key, []).append(record)

    summary: Dict[str, Any] = {}
    per_element: Dict[str, Dict[str, Dict[str, float]]] = {}
    stratified: Dict[str, Dict[str, Dict[str, Any]]] = {
        "dataset_id": {},
        "truth_type": {},
        "rp_bucket": {},
        "spectrum_kind": {},
        "label_cardinality": {},
    }
    for key, pair_records in by_pair.items():
        summary[key] = _compute_composition_overall(pair_records)
        per_element[key] = _compute_composition_per_element(pair_records)
        pair_buckets = _compute_composition_stratified_buckets(pair_records, stratified.keys())
        for field_name, bucket_data in pair_buckets.items():
            stratified[field_name][key] = bucket_data
    return {"overall": summary, "per_element": per_element, "stratified": stratified}


def mcnemar_test(
    left_records: Sequence[IDEvaluationRecord],
    right_records: Sequence[IDEvaluationRecord],
) -> Dict[str, float]:
    if chi2 is None:
        return {"b": 0, "c": 0, "chi2": float("nan"), "p_value": float("nan")}
    left_map = {
        (record.outer_split_id, record.spectrum_id): record
        for record in left_records
        if record.scored
    }
    right_map = {
        (record.outer_split_id, record.spectrum_id): record
        for record in right_records
        if record.scored
    }
    keys = sorted(set(left_map) & set(right_map))
    b = c = 0
    for key in keys:
        left_success = left_map[key].exact_match
        right_success = right_map[key].exact_match
        if left_success and not right_success:
            b += 1
        elif not left_success and right_success:
            c += 1
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2.sf(statistic, 1))
    return {"b": b, "c": c, "chi2": float(statistic), "p_value": p_value}


def friedman_nemenyi(
    blocks: Dict[str, Dict[str, float]],
    higher_is_better: bool,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    if friedmanchisquare is None or studentized_range is None:
        return {}
    workflows = sorted({workflow for values in blocks.values() for workflow in values.keys()})
    if len(workflows) < 3:
        return {}
    valid_blocks = [
        block for block in blocks.values() if all(workflow in block for workflow in workflows)
    ]
    if len(valid_blocks) < 2:
        return {}

    matrix = np.array(
        [[block[workflow] for workflow in workflows] for block in valid_blocks],
        dtype=float,
    )
    ranks = np.zeros_like(matrix)
    for row_idx, row in enumerate(matrix):
        order = np.argsort(-row if higher_is_better else row)
        row_ranks = np.empty_like(order, dtype=float)
        row_ranks[order] = np.arange(1, len(workflows) + 1, dtype=float)
        ranks[row_idx] = row_ranks
    friedman = friedmanchisquare(*[matrix[:, idx] for idx in range(matrix.shape[1])])
    avg_ranks = {workflow: float(np.mean(ranks[:, idx])) for idx, workflow in enumerate(workflows)}
    q_alpha = float(studentized_range.ppf(1.0 - alpha, len(workflows), np.inf) / math.sqrt(2.0))
    cd = q_alpha * math.sqrt(len(workflows) * (len(workflows) + 1) / (6.0 * len(valid_blocks)))
    significant_pairs: List[Dict[str, Any]] = []
    for idx_left, workflow_left in enumerate(workflows):
        for idx_right in range(idx_left + 1, len(workflows)):
            workflow_right = workflows[idx_right]
            diff = abs(avg_ranks[workflow_left] - avg_ranks[workflow_right])
            significant_pairs.append(
                {
                    "left": workflow_left,
                    "right": workflow_right,
                    "rank_diff": diff,
                    "significant": diff > cd,
                }
            )
    return {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "average_ranks": avg_ranks,
        "critical_difference": float(cd),
        "pairs": significant_pairs,
    }


def _records_to_rows(records: Sequence[Any]) -> List[Dict[str, Any]]:
    return [asdict(record) for record in records]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _plot_per_element_heatmap(
    per_element: Dict[str, Dict[str, Dict[str, float]]], output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    workflows = sorted(per_element.keys())
    elements = sorted(
        {element for workflow in workflows for element in per_element[workflow].keys()}
    )
    if not workflows or not elements:
        return
    data = np.zeros((len(elements), len(workflows)), dtype=float)
    for workflow_idx, workflow in enumerate(workflows):
        for element_idx, element in enumerate(elements):
            data[element_idx, workflow_idx] = float(
                per_element.get(workflow, {}).get(element, {}).get("precision", 0.0)
            )

    fig, ax = plt.subplots(figsize=(max(8, len(workflows) * 1.2), max(6, len(elements) * 0.3)))
    image = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(workflows)))
    ax.set_xticklabels(workflows, rotation=45, ha="right")
    ax.set_yticks(range(len(elements)))
    ax.set_yticklabels(elements)
    fig.colorbar(image, ax=ax, label="Precision")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_recall(summary: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not summary:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for workflow, metrics in summary.items():
        ax.scatter(metrics["micro_recall"], metrics["micro_precision"], label=workflow, s=60)
        ax.text(metrics["micro_recall"], metrics["micro_precision"], workflow, fontsize=8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Trade-off")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_rp(
    stratified: Dict[str, Dict[str, Dict[str, Any]]],
    metric_name: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not stratified:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for workflow, buckets in stratified.items():
        points = []
        for bucket, metrics in buckets.items():
            midpoint = _rp_bucket_midpoint(bucket)
            if np.isfinite(midpoint):
                points.append((midpoint, float(metrics.get(metric_name, 0.0))))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=workflow,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Resolving Power (bucket midpoint)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} vs RP")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_failure_table(records: Sequence[IDEvaluationRecord], output_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    by_workflow: Dict[str, List[IDEvaluationRecord]] = {}
    for record in records:
        if record.scored:
            by_workflow.setdefault(record.workflow_name, []).append(record)
    for workflow, workflow_records in by_workflow.items():
        for element in ["Mn", "Na"]:
            false_positives = sum(
                1
                for record in workflow_records
                if element in record.predicted_elements and element not in record.true_elements
            )
            rows.append(
                {
                    "workflow": workflow,
                    "failure_mode": f"{element}_false_positive",
                    "count": false_positives,
                    "note": "Systematic low-RP false-positive source",
                }
            )
        basis_mismatch = sum(
            1
            for record in workflow_records
            if float(record.annotations.get("basis_fwhm_mismatch_nm", 0.0)) > 0.25
        )
        rows.append(
            {
                "workflow": workflow,
                "failure_mode": "basis_mismatch",
                "count": basis_mismatch,
                "note": "Selected basis FWHM differs materially from RP-implied target",
            }
        )
        label_caveats = sum(
            1 for record in workflow_records if record.truth_type == TruthType.FORMULA_PROXY.value
        )
        rows.append(
            {
                "workflow": workflow,
                "failure_mode": "formula_proxy_labels",
                "count": label_caveats,
                "note": "Mineral truth labels come from formulas rather than specimen assays",
            }
        )
    _write_csv(output_path, rows)


class UnifiedBenchmarkRunner:
    def __init__(
        self,
        db_path: Path,
        basis_dir: Optional[Path] = None,
        quick: bool = False,
    ):
        self.context = UnifiedBenchmarkContext(db_path=db_path, basis_dir=basis_dir)
        self.id_registry = build_id_workflow_registry(quick=quick)
        self.composition_registry = build_composition_workflow_registry(quick=quick)

    def run_identification(
        self,
        datasets: Sequence[BenchmarkDataset],
        workflow_names: Sequence[str],
        max_outer_folds: Optional[int] = None,
    ) -> Tuple[List[IDEvaluationRecord], List[Dict[str, Any]]]:
        records: List[IDEvaluationRecord] = []
        selections: List[Dict[str, Any]] = []
        for dataset in datasets:
            outer_splits = build_outer_splits(dataset)
            if max_outer_folds is not None:
                outer_splits = outer_splits[: int(max_outer_folds)]
            for outer_split in outer_splits:
                train_dataset = subset_dataset(
                    dataset, outer_split.train_ids, f"{dataset.name}_{outer_split.name}_train"
                )
                test_spectra = [dataset.get_spectrum(spec_id) for spec_id in outer_split.test_ids]
                for workflow_name in workflow_names:
                    workflow = self.id_registry[workflow_name]
                    best_config, config_name, tuning_split_id, tuning_scores = tune_id_workflow(
                        self.context,
                        workflow,
                        train_dataset,
                        list(dataset.elements),
                        outer_split.name,
                    )
                    selections.append(
                        {
                            "dataset_id": dataset.name,
                            "workflow_name": workflow_name,
                            "outer_split_id": outer_split.name,
                            "tuning_split_id": tuning_split_id,
                            "config_name": config_name,
                            "config": best_config,
                            "inner_scores": tuning_scores,
                        }
                    )
                    predictor = workflow.build_predictor(
                        self.context, list(dataset.elements), best_config
                    )
                    records.extend(
                        evaluate_id_workflow(
                            test_spectra,
                            workflow=workflow,
                            predictor=predictor,
                            outer_split_id=outer_split.name,
                            tuning_split_id=tuning_split_id,
                            config_name=config_name,
                        )
                    )
        return records, selections

    def run_composition(
        self,
        datasets: Sequence[BenchmarkDataset],
        id_workflow_names: Sequence[str],
        composition_workflow_names: Sequence[str],
        max_outer_folds: Optional[int] = None,
    ) -> Tuple[List[CompositionEvaluationRecord], List[Dict[str, Any]]]:
        records: List[CompositionEvaluationRecord] = []
        selections: List[Dict[str, Any]] = []

        for dataset in datasets:
            supervised = [spec for spec in dataset.spectra if spec.truth_type != TruthType.BLIND]
            if not supervised:
                continue
            outer_splits = build_outer_splits(dataset)
            if max_outer_folds is not None:
                outer_splits = outer_splits[: int(max_outer_folds)]
            for outer_split in outer_splits:
                train_dataset = subset_dataset(
                    dataset, outer_split.train_ids, f"{dataset.name}_{outer_split.name}_train"
                )
                test_spectra = [dataset.get_spectrum(spec_id) for spec_id in outer_split.test_ids]
                for id_workflow_name in id_workflow_names:
                    id_workflow = self.id_registry[id_workflow_name]
                    best_config, config_name, tuning_split_id, tuning_scores = tune_id_workflow(
                        self.context,
                        id_workflow,
                        train_dataset,
                        list(dataset.elements),
                        outer_split.name,
                    )
                    selections.append(
                        {
                            "dataset_id": dataset.name,
                            "id_workflow_name": id_workflow_name,
                            "outer_split_id": outer_split.name,
                            "tuning_split_id": tuning_split_id,
                            "config_name": config_name,
                            "config": best_config,
                            "inner_scores": tuning_scores,
                        }
                    )
                    id_predictor = id_workflow.build_predictor(
                        self.context, list(dataset.elements), best_config
                    )
                    for composition_name in composition_workflow_names:
                        composition_workflow = self.composition_registry[composition_name]
                        (
                            composition_config,
                            composition_config_name,
                            composition_tuning_split_id,
                            composition_scores,
                        ) = tune_composition_workflow(
                            self.context,
                            composition_workflow,
                            train_dataset,
                            id_workflow_name=id_workflow_name,
                            id_config_name=config_name,
                            id_predictor=id_predictor,
                            outer_split_id=outer_split.name,
                        )
                        selections.append(
                            {
                                "dataset_id": dataset.name,
                                "id_workflow_name": id_workflow_name,
                                "composition_workflow_name": composition_name,
                                "outer_split_id": outer_split.name,
                                "tuning_split_id": composition_tuning_split_id,
                                "config_name": composition_config_name,
                                "config": composition_config,
                                "inner_scores": composition_scores,
                            }
                        )
                        composition_predictor = composition_workflow.fit_predictor(
                            self.context,
                            [
                                spec
                                for spec in train_dataset.spectra
                                if spec.truth_type != TruthType.BLIND
                            ],
                            composition_config,
                        )
                        records.extend(
                            evaluate_composition_workflow(
                                test_spectra,
                                id_workflow_name=id_workflow_name,
                                id_config_name=config_name,
                                id_predictor=id_predictor,
                                composition_workflow=composition_workflow,
                                composition_predictor=composition_predictor,
                                outer_split_id=outer_split.name,
                                tuning_split_id=composition_tuning_split_id,
                                composition_config_name=composition_config_name,
                            )
                        )
        return records, selections

    def write_outputs(
        self,
        output_dir: Path,
        id_records: Sequence[IDEvaluationRecord],
        id_selections: Sequence[Dict[str, Any]],
        composition_records: Sequence[CompositionEvaluationRecord],
        composition_selections: Sequence[Dict[str, Any]],
    ) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        id_summary = summarize_id_records(id_records)
        composition_summary = summarize_composition_records(composition_records)

        outputs = {
            "id_records_json": output_dir / "id_records.json",
            "id_records_csv": output_dir / "id_records.csv",
            "id_summary_json": output_dir / "id_summary.json",
            "id_selections_json": output_dir / "id_config_selections.json",
            "composition_records_json": output_dir / "composition_records.json",
            "composition_records_csv": output_dir / "composition_records.csv",
            "composition_summary_json": output_dir / "composition_summary.json",
            "composition_selections_json": output_dir / "composition_config_selections.json",
            "overall_ranking_csv": output_dir / "overall_ranking.csv",
            "aitchison_summary_csv": output_dir / "aitchison_summary.csv",
            "latency_table_csv": output_dir / "latency_table.csv",
            "failure_table_csv": output_dir / "failure_table.csv",
            "statistics_json": output_dir / "statistics.json",
            "per_element_heatmap_png": figures_dir / "per_element_heatmap.png",
            "precision_recall_png": figures_dir / "precision_recall.png",
            "f1_vs_rp_png": figures_dir / "f1_vs_rp.png",
            "precision_vs_rp_png": figures_dir / "precision_vs_rp.png",
        }

        _write_json(outputs["id_records_json"], _records_to_rows(id_records))
        _write_csv(outputs["id_records_csv"], _records_to_rows(id_records))
        _write_json(outputs["id_summary_json"], id_summary)
        _write_json(outputs["id_selections_json"], list(id_selections))
        _write_json(outputs["composition_records_json"], _records_to_rows(composition_records))
        _write_csv(outputs["composition_records_csv"], _records_to_rows(composition_records))
        _write_json(outputs["composition_summary_json"], composition_summary)
        _write_json(outputs["composition_selections_json"], list(composition_selections))

        overall_rows = [
            {"workflow": workflow, **metrics}
            for workflow, metrics in sorted(id_summary["overall"].items())
        ]
        _write_csv(outputs["overall_ranking_csv"], overall_rows)
        _write_csv(
            outputs["aitchison_summary_csv"],
            [
                {"workflow_pair": workflow_pair, **metrics}
                for workflow_pair, metrics in sorted(composition_summary["overall"].items())
            ],
        )
        latency_rows = [
            {
                "kind": "identification",
                "workflow": workflow,
                "latency_mean_s": metrics["latency_mean_s"],
                "latency_p95_s": metrics["latency_p95_s"],
            }
            for workflow, metrics in sorted(id_summary["overall"].items())
        ] + [
            {
                "kind": "composition",
                "workflow": workflow_pair,
                "latency_mean_s": metrics["latency_mean_s"],
                "latency_p95_s": metrics["latency_p95_s"],
            }
            for workflow_pair, metrics in sorted(composition_summary["overall"].items())
        ]
        _write_csv(outputs["latency_table_csv"], latency_rows)
        _write_failure_table(id_records, outputs["failure_table_csv"])

        statistics: Dict[str, Any] = {"mcnemar": {}, "friedman_nemenyi": {}}
        by_workflow_records = {
            workflow: [
                record
                for record in id_records
                if record.workflow_name == workflow and record.scored
            ]
            for workflow in sorted({record.workflow_name for record in id_records})
        }
        workflows = sorted(by_workflow_records.keys())
        for idx, left_workflow in enumerate(workflows):
            for right_workflow in workflows[idx + 1 :]:
                key = f"{left_workflow}__vs__{right_workflow}"
                statistics["mcnemar"][key] = mcnemar_test(
                    by_workflow_records[left_workflow], by_workflow_records[right_workflow]
                )
        id_blocks: Dict[str, Dict[str, float]] = {}
        for record in id_records:
            if record.scored:
                id_blocks.setdefault(f"{record.outer_split_id}:{record.spectrum_id}", {})[
                    record.workflow_name
                ] = record.f1
        statistics["friedman_nemenyi"]["identification"] = friedman_nemenyi(
            id_blocks, higher_is_better=True
        )
        comp_blocks: Dict[str, Dict[str, float]] = {}
        for record in composition_records:
            if record.scored and record.aitchison is not None:
                pair_name = f"{record.id_workflow_name}__{record.composition_workflow_name}"
                comp_blocks.setdefault(f"{record.outer_split_id}:{record.spectrum_id}", {})[
                    pair_name
                ] = record.aitchison
        statistics["friedman_nemenyi"]["composition"] = friedman_nemenyi(
            comp_blocks, higher_is_better=False
        )
        _write_json(outputs["statistics_json"], statistics)

        _plot_per_element_heatmap(id_summary["per_element"], outputs["per_element_heatmap_png"])
        _plot_precision_recall(id_summary["overall"], outputs["precision_recall_png"])
        _plot_metric_vs_rp(
            id_summary["stratified"]["rp_bucket"], "micro_f1", outputs["f1_vs_rp_png"]
        )
        _plot_metric_vs_rp(
            id_summary["stratified"]["rp_bucket"],
            "micro_precision",
            outputs["precision_vs_rp_png"],
        )
        return outputs


build_default_datasets = load_default_datasets


__all__ = [
    "CompositionEvaluationRecord",
    "CompositionWorkflowSpec",
    "IDEvaluationRecord",
    "IDWorkflowSpec",
    "UnifiedBenchmarkContext",
    "UnifiedBenchmarkRunner",
    "bootstrap_ci",
    "build_composition_workflow_registry",
    "build_default_datasets",
    "build_id_workflow_registry",
    "build_inner_splits",
    "build_outer_splits",
    "derive_truth_elements",
    "friedman_nemenyi",
    "load_aalto_id_dataset",
    "load_assay_and_blind_datasets",
    "load_default_datasets",
    "load_manifest_synthetic_dataset",
    "mcnemar_test",
    "subset_dataset",
    "summarize_composition_records",
    "summarize_id_records",
]
