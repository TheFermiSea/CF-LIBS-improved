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
import inspect
import json
import os
import sys
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _jax_identifier_flags_for(cls) -> Dict[str, bool]:
    """
    Return ``{flag: True}`` for every ``use_jax_*`` kwarg accepted by *cls*.

    Gated on ``CFLIBS_USE_JAX_IDENTIFIER=1``.  When the env var is unset or
    "0", returns an empty dict (preserves the as-shipped CPU-only behavior
    of every identifier).  When set to "1", auto-detects each identifier's
    ``use_jax_*`` constructor kwargs via ``inspect.signature`` and turns
    them all on.

    This is the toggle the unified benchmark CLI flips via
    ``--jax-identifier``.  See PR #118, #119, #120, #121, #122 for the
    identifier-side opt-in flags this targets.

    **Pure function.** The previous version of this helper had a hidden
    side effect that mutated ``jax.config`` (bead
    ``CF-LIBS-improved-jbfg.1``).  That contract has been lifted to
    :func:`cflibs.core.jax_runtime.configure_for_identifiers` (called
    once by :class:`UnifiedBenchmarkRunner.__init__`) and is verified at
    each identifier constructor via :func:`check_jax64bit`.  Arch review
    #2 candidate 2.
    """
    if os.environ.get("CFLIBS_USE_JAX_IDENTIFIER", "0") != "1":
        return {}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    return {name: True for name in sig.parameters if name.startswith("use_jax_")}


# These imports sit AFTER the `_jax_param_keys` helper above to keep that
# helper trivial-to-call without paying the cost (or risking the circular
# import) of pulling in the full benchmark.dataset / composition_metrics
# graph. Each import carries a per-line E402 suppression because ruff's
# module-level-imports rule doesn't understand the intentional placement.
from cflibs.benchmark.dataset import (  # noqa: E402
    BenchmarkDataset,
    BenchmarkSpectrum,
    DataSplit,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
    TruthType,
)
from cflibs.benchmark.loaders import load_benchmark  # noqa: E402
from cflibs.benchmark.checkpoint import (  # noqa: E402
    emit_checkpoint_part,
    make_worker_slug,
    new_run_id,
)
from cflibs.benchmark.composition_eval import (  # noqa: E402,F401
    # Public surface
    CompositionEvaluationRecord,
    evaluate_composition_workflow,
    # Backward-compat: leading-underscore helpers are imported by name
    # in :mod:`tests.benchmark.test_jax_workflows`,
    # :mod:`tests.benchmark.test_posterior_metrics`, and some scripts,
    # so re-export them here verbatim.
    _build_composition_failure_record,
    _build_composition_success_record,
    _coerce_composition_prediction,
    _compose_annotations,
    _composition_error_tier,
    _compute_fractional_error,
    _maybe_compute_posterior_diagnostics,
    _spectrum_metadata_fields,
)
from cflibs.benchmark.composition_metrics import (  # noqa: E402
    aitchison_distance,
    load_subcompositional_pairs,
    stratify_per_element_errors,
)
from cflibs.core.logging_config import get_logger  # noqa: E402

if TYPE_CHECKING:
    from cflibs.inversion.common.element_id import ElementIdentificationResult

logger = get_logger("benchmark.unified")

# Backward-compat alias: ``_emit_checkpoint_part`` was the original private
# helper in this module before the checkpoint primitives moved to
# :mod:`cflibs.benchmark.checkpoint`.  Any external caller (incl. older test
# fixtures) that imports the old name still works.
_emit_checkpoint_part = emit_checkpoint_part

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


# ---------------------------------------------------------------------------
# Optional JAX / NumPyro deps for the GPU-using composition workflows
# (`bayesian` + `iterative_jax`).  Both workflows register unconditionally so
# the benchmark gate's CLI surface stays stable; missing deps are converted to
# a clear runtime warning + numpy fallback inside the workflow predictors.
# ---------------------------------------------------------------------------
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_JAX = False

try:
    import numpyro  # noqa: F401

    HAS_NUMPYRO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_NUMPYRO = False

try:  # The JAX iterative solver is being built in parallel by Agent B; the
    # benchmark gate must keep working even when it lands later.
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolverJax  # type: ignore

    HAS_JAX_ITERATIVE_SOLVER = True
except (ImportError, AttributeError):  # pragma: no cover - exercised when absent
    IterativeCFLIBSSolverJax = None  # type: ignore[assignment]
    HAS_JAX_ITERATIVE_SOLVER = False


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


SUPPORTED_BOLTZMANN_WEIGHTINGS = {None, "aki_inverse_variance", "intensity_only"}


def _validate_boltzmann_weighting(weighting: Optional[str]) -> Optional[str]:
    if weighting not in SUPPORTED_BOLTZMANN_WEIGHTINGS:
        allowed = ", ".join(sorted(w for w in SUPPORTED_BOLTZMANN_WEIGHTINGS if w is not None))
        raise ValueError(
            f"Unsupported Boltzmann weighting {weighting!r}; expected one of {allowed}"
        )
    return weighting


def _run_boltzmann_pipeline_lazy(
    spectrum: Dict[str, Any],
    db: Any,
    fit_method: Any,
    closure_mode: str,
    elements: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[Dict[str, float]]:
    if "weighting" in kwargs:
        kwargs["weighting"] = _validate_boltzmann_weighting(kwargs["weighting"])
    return _load_repo_script_module("run_comprehensive_benchmark").run_boltzmann_pipeline(
        spectrum,
        db=db,
        fit_method=fit_method,
        closure_mode=closure_mode,
        elements=elements,
        **kwargs,
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
    vrabel_max_shots_per_sample: Optional[int] = 50,
    dataset_shard: Optional[tuple[int, int]] = None,
) -> Dict[str, BenchmarkDataset]:
    """Load the default benchmark datasets, optionally split by shard.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    synthetic_corpus_path : Path, optional
        Manifest path for an optional synthetic corpus.
    vrabel_max_shots_per_sample : int, optional
        Cap shots loaded per Vrabel sample. ``None`` loads all shots
        (~50k spectra). Sharding is applied *after* this cap so that
        ``vrabel_max_shots_per_sample=None`` + ``dataset_shard=(1, 3)``
        means full corpus / 3 (not capped / 3).
    dataset_shard : tuple[int, int], optional
        ``(N, K)`` to retain only the N-th shard of K. Applied to the
        Vrabel dataset only — smaller community datasets (BHVO-2,
        NIST SRM 612) are not sharded because they're <100 spectra
        and sharding would yield too few samples per node to be useful.
        See ``docs/dataset-sharding.md`` for the math.
    """
    from cflibs.benchmark.loaders import (
        _load_bhvo2_usgs,
        _load_nist_srm_612,
        _load_vrabel2020_soils,
        apply_dataset_shard,
    )

    datasets: Dict[str, BenchmarkDataset] = {"aalto_libs": load_aalto_id_dataset(data_dir)}
    datasets.update(load_assay_and_blind_datasets(data_dir))

    # Community CRM datasets: bhvo2_usgs, nist_srm_612.
    # Returns None when data directories are absent — omit silently so the
    # benchmark registry degrades gracefully before the ingest pipeline runs.
    # NOTE: these are NOT sharded — both are <100 spectra and sharding them
    # would give each node <5 samples (statistically uninformative).
    for _crm_loader in (_load_bhvo2_usgs, _load_nist_srm_612):
        _ds = _crm_loader(data_dir)
        if _ds is not None:
            datasets[_ds.name] = _ds

    # Vrabel 2020 Sci Data peer-reviewed LIBS benchmark — 100 mixed soil/ore
    # samples × N shots/sample with certified compositions (Al, Ca, Cr, Cu, Fe,
    # K, Mg, Na, Pb, Si).  Default cap is 50 shots/sample (~5,000 spectra,
    # ~1.6 GB).  Pass vrabel_max_shots_per_sample=None for the full 50k.
    _vrabel = _load_vrabel2020_soils(data_dir, max_spectra_per_sample=vrabel_max_shots_per_sample)
    if _vrabel is not None:
        if dataset_shard is not None:
            shard_n, shard_k = dataset_shard
            _vrabel = apply_dataset_shard(_vrabel, shard_n, shard_k)
        datasets[_vrabel.name] = _vrabel

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


# ``CompositionEvaluationRecord`` lives in :mod:`cflibs.benchmark.composition_eval`
# (re-exported below near the module's checkpoint imports) so the
# per-spectrum evaluator can construct it without depending on this module.


def _empty_identification_result(
    algorithm: str,
    warnings: Optional[List[str]] = None,
) -> ElementIdentificationResult:
    from cflibs.inversion.common.element_id import ElementIdentificationResult

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


def _class_default_config(cls: Any, keys: Sequence[str]) -> Dict[str, Any]:
    """Pull keyword defaults from ``cls.__init__`` signature.

    Used by the workflow-config grids so that architect-modified class
    defaults flow into ``tune_id_workflow``'s grid search. Without this,
    ``_build_alias_predictor`` (and siblings) pass explicit grid values
    that override the defaults — diagnosed 2026-05-09 from PRs #101-#108
    when architect default-changes had no observable effect on benchmark
    predictions because the explicit grid bypassed them.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    out: Dict[str, Any] = {}
    for k in keys:
        param = sig.parameters.get(k)
        if param is None or param.default is inspect.Parameter.empty:
            continue
        out[k] = param.default
    return out


def _alias_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    from cflibs.inversion.identify.alias import ALIASIdentifier

    arch_defaults = _class_default_config(
        ALIASIdentifier,
        (
            "detection_threshold",
            "intensity_threshold_factor",
            "chance_window_scale",
            "max_lines_per_element",
            "boltzmann_r2_min",
        ),
    )
    # PR #159 changed ALIASIdentifier's threshold-kwarg defaults from
    # explicit `float = 3.0` / `0.02` to `Optional[float] = None` so the
    # constructor's high_recall=False/True preset resolves them. That
    # made `_class_default_config` read `None` for both knobs here,
    # which would let a future change to high_recall's default silently
    # flip the strict alias workflow into recall mode. Pin the strict
    # values explicitly so the benchmark contract is robust to future
    # constructor-signature drift.
    if arch_defaults.get("intensity_threshold_factor") is None:
        arch_defaults["intensity_threshold_factor"] = 3.0
    if arch_defaults.get("detection_threshold") is None:
        arch_defaults["detection_threshold"] = 0.02
    if quick:
        return [
            arch_defaults,
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
    configs: List[Dict[str, Any]] = [arch_defaults]
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


def _alias_high_recall_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    """Configs for the ``alias_high_recall`` workflow.

    Mirrors ``_alias_workflow_configs`` but deliberately leaves
    ``intensity_threshold_factor`` and ``detection_threshold`` UNSET so
    PR #159's ``ALIASIdentifier(high_recall=True)`` preset resolves them
    to its built-in recall values (2.0 / 0.01 — 33% / 50% looser than
    the strict 3.0 / 0.02 defaults). Per CF-LIBS-improved-knyz this is
    the wiring that surfaces more candidates on aa1100_substrate, where
    the strict default rejects 9 of 12 records at the identification
    stage and they fall out of composition entirely.

    Bead CF-LIBS-improved-n3rf.2 found that surfacing those extra
    candidates is insufficient on its own — the strict downstream
    R² and CL gates re-rejected most of them, dragging macro_F1 to
    0.083 (worse than strict alias at 0.139). The fix is baked into
    ``_build_alias_high_recall_predictor`` (inherits alias_v2's
    ``r2_gate_mode='adaptive_t'`` + ``relative_cl_per_ion_stage=True``),
    not in this config dict, so it cannot be tuned out by parameter-
    sweep callers.

    Other knobs (chance_window_scale, max_lines_per_element) are still
    sweepable so the workflow can be tuned independently of the strict
    alias workflow.
    """
    if quick:
        return [
            {"chance_window_scale": 0.4, "max_lines_per_element": 30},
            {"chance_window_scale": 0.3, "max_lines_per_element": 30},
        ]
    return [{"chance_window_scale": cws, "max_lines_per_element": 30} for cws in (0.3, 0.4)]


def _build_alias_high_recall_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    """Thin wrapper over :func:`_make_predictor` for back-compat.

    See ``ID_WORKFLOW_PRESETS["alias_high_recall"]`` -- ``high_recall=True``
    plus the alias_v2 gating cocktail (bead n3rf.2): the looser peak
    thresholds need ``r2_gate_mode='adaptive_t'`` and
    ``relative_cl_per_ion_stage=True`` so the strict downstream gates
    don't immediately re-reject the extra candidates. Records
    ``parameters['alias_mode']='high_recall_v2_gates'``.
    """
    return _resolve_id_workflow_preset("alias_high_recall")(context, candidate_elements, config)


def _alias_v2_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    """Configs for the ``alias_v2`` workflow — Phase D promotion of the
    ftp1+dj6y sweep winner per docs/research/findings/2026-05-14-v2-
    empirical-07-alias-fix-sweep.md.

    Like ``_alias_high_recall_workflow_configs``, this omits
    threshold kwargs so the constructor's strict defaults
    (intensity_threshold_factor=3.0, detection_threshold=0.02) apply.
    The two fix flags (r2_gate_mode='adaptive_t' + relative_cl_per_ion_stage)
    are baked into the predictor, NOT the config dict, so they cannot
    be tuned out by parameter-sweep callers.

    Empirical baseline on shard 1/3 + vrabel-max-shots 1 (per the v2
    sweep): macro_f1=0.3092, macro_precision=0.4318, macro_recall=0.2700,
    fp_per_spectrum=0.4545. That's +0.198 macro_f1 lift vs the strict
    alias's 0.1111 on the same corpus shape.
    """
    if quick:
        return [
            {"chance_window_scale": 0.4, "max_lines_per_element": 30},
            {"chance_window_scale": 0.3, "max_lines_per_element": 30},
        ]
    return [{"chance_window_scale": cws, "max_lines_per_element": 30} for cws in (0.3, 0.4)]


def _build_alias_v2_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    """Thin wrapper over :func:`_make_predictor` for back-compat.

    See ``ID_WORKFLOW_PRESETS["alias_v2"]`` -- the Phase D ftp1+dj6y
    sweep winner. Bakes in ``r2_gate_mode='adaptive_t'`` and
    ``relative_cl_per_ion_stage=True``; strict thresholds preserved by
    not passing them through the constructor. Records
    ``parameters['alias_mode']='v2_ftp1_plus_dj6y'``.
    """
    return _resolve_id_workflow_preset("alias_v2")(context, candidate_elements, config)


# ---------------------------------------------------------------------------
# Alias-fix sweep harness (Phase C of jaunty-weaving-mist).
#
# Enumerates the 2^3 = 8 combinations of the three opt-in fix flags landed in
# PRs #175 (ftp1 — adaptive_t r²-gate), #177 (762f — robust temperature
# estimator), and #176 (dj6y — per-ion-stage relative_cl_threshold). Each cell
# constructs an ``ALIASIdentifier`` with the cell-specific fix-flag kwargs,
# while pinning the strict threshold kwargs to their precision-king defaults
# (3.0 / 0.02 / 0.4 / 30) so the ONLY difference across cells is the 3 fix
# flags. Designed for the alias-fix sweep at
# ``scripts/sweep_alias_fixes.py``; expected wall-time on the cluster is
# 8 cells × 5 seeds × ~10 min/run ≈ 7 h serial, ≈ 2.5 h with 3-way parallelism.
# ---------------------------------------------------------------------------

# (cell_name, fix-flag kwargs) — order matches the plan's configuration table.
_ALIAS_SWEEP_CELLS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("baseline", {}),
    ("ftp1", {"r2_gate_mode": "adaptive_t"}),
    ("762f", {"temperature_estimator_mode": "robust"}),
    ("dj6y", {"relative_cl_per_ion_stage": True}),
    ("ftp1+762f", {"r2_gate_mode": "adaptive_t", "temperature_estimator_mode": "robust"}),
    ("ftp1+dj6y", {"r2_gate_mode": "adaptive_t", "relative_cl_per_ion_stage": True}),
    ("762f+dj6y", {"temperature_estimator_mode": "robust", "relative_cl_per_ion_stage": True}),
    (
        "all_three",
        {
            "r2_gate_mode": "adaptive_t",
            "temperature_estimator_mode": "robust",
            "relative_cl_per_ion_stage": True,
        },
    ),
)


# Strict threshold defaults shared by every sweep cell. Pinned explicitly here
# (rather than read from ``_class_default_config``) so a future change to
# ``ALIASIdentifier.__init__`` defaults can't silently flip the sweep's
# baseline. These ARE the precision-king defaults guarded by
# ``test_strict_alias_still_registered_unchanged`` for the strict alias
# workflow.
_ALIAS_SWEEP_BASE_KWARGS: Dict[str, Any] = {
    "intensity_threshold_factor": 3.0,
    "detection_threshold": 0.02,
    "chance_window_scale": 0.4,
    "max_lines_per_element": 30,
}


def _alias_sweep_workflow_configs(
    quick: bool,
) -> List[Dict[str, Any]]:  # noqa: ARG001 - signature parity
    """Single-element config grid shared by every ``alias_sweep_*`` workflow.

    All 8 cells use the SAME threshold kwargs (pinned at strict defaults via
    the predictor factory below) — the only thing that varies is the fix-flag
    kwargs baked into the per-cell predictor closure. So every cell has
    exactly one config; cross-cell variation lives in the predictor, not the
    grid. The ``quick`` parameter is accepted for signature parity with the
    other ``_<wf>_workflow_configs`` helpers but is intentionally unused.
    """
    return [{}]


def _build_alias_sweep_predictor_factory(
    cell_kwargs: Dict[str, Any],
    cell_name: str,
) -> Callable[
    ["UnifiedBenchmarkContext", List[str], Dict[str, Any]],
    Callable[[BenchmarkSpectrum], ElementIdentificationResult],
]:
    """Return a ``build_predictor`` callable for one sweep cell.

    The returned callable matches the signature expected by
    ``IDWorkflowSpec.build_predictor``. ``cell_kwargs`` is the per-cell
    fix-flag kwarg dict (one of ``_ALIAS_SWEEP_CELLS`` values); ``cell_name``
    is recorded in ``result.parameters['alias_sweep_cell']`` for downstream
    audit-ability.
    """

    def build_predictor(
        context: UnifiedBenchmarkContext,
        candidate_elements: List[str],
        config: Dict[str, Any],  # noqa: ARG001 - empty grid, see configs helper
    ) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
        def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
            from cflibs.atomic.database import AtomicDatabase
            from cflibs.inversion.identify.alias import ALIASIdentifier

            with AtomicDatabase(str(context.db_path)) as db:
                identifier = ALIASIdentifier(
                    atomic_db=db,
                    elements=candidate_elements,
                    resolving_power=_estimate_rp_for_spectrum(spectrum),
                    intensity_threshold_factor=float(
                        _ALIAS_SWEEP_BASE_KWARGS["intensity_threshold_factor"]
                    ),
                    detection_threshold=float(_ALIAS_SWEEP_BASE_KWARGS["detection_threshold"]),
                    chance_window_scale=float(_ALIAS_SWEEP_BASE_KWARGS["chance_window_scale"]),
                    max_lines_per_element=int(_ALIAS_SWEEP_BASE_KWARGS["max_lines_per_element"]),
                    **cell_kwargs,
                    **_jax_identifier_flags_for(ALIASIdentifier),
                )
                result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
                result.parameters["candidate_elements"] = list(candidate_elements)
                result.parameters["alias_sweep_cell"] = cell_name
                result.parameters["alias_sweep_fix_kwargs"] = dict(cell_kwargs)
                return result

        return predictor

    return build_predictor


def _comb_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    from cflibs.inversion.identify.comb import CombIdentifier

    arch_defaults = _class_default_config(
        CombIdentifier,
        ("min_correlation", "tooth_activation_threshold", "relative_threshold_scale"),
    )
    if quick:
        return [
            arch_defaults,
            {
                "min_correlation": 0.08,
                "tooth_activation_threshold": 0.35,
                "relative_threshold_scale": 1.4,
            },
        ]
    configs: List[Dict[str, Any]] = [arch_defaults]
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
    from cflibs.inversion.identify.correlation import CorrelationIdentifier

    arch_defaults = _class_default_config(
        CorrelationIdentifier,
        ("min_confidence", "relative_threshold_scale", "min_line_strength"),
    )
    if quick:
        return [
            arch_defaults,
            {"min_confidence": 0.008, "relative_threshold_scale": 1.2, "min_line_strength": 1000.0},
        ]
    configs: List[Dict[str, Any]] = [arch_defaults]
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


def _hybrid_consensus_2of3_workflow_configs(quick: bool) -> List[Dict[str, Any]]:
    """Configs for the ``hybrid_consensus_2of3`` workflow.

    This workflow runs ALIAS, Comb, and Correlation identifiers in parallel
    and applies a 2-of-3 majority vote. The vote threshold is fixed at 2
    (the "2-of-3" rule) — no parameter grid is needed because the consensus
    semantics are the entire point of this workflow.

    The three constituent identifiers use their default strict thresholds
    (same as the ``alias``, ``comb``, and ``correlation`` workflows) so the
    consensus gate only changes the *decision rule*, not the per-identifier
    sensitivity.
    """
    return [{}]


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


# ---------------------------------------------------------------------------
# Predictor-builder factory (arch candidate 1).
#
# Six near-identical hand-rolled builders -- ``_build_alias_predictor``,
# ``_build_alias_v2_predictor``, ``_build_alias_high_recall_predictor``, plus
# the three ``_build_hybrid_consensus_*_predictor`` variants -- shared an
# identical outer shell and differed only by:
#   (a) the ALIAS cocktail (strict / v2 / high_recall_v2)
#   (b) for consensus variants, which sibling identifiers join the vote
#   (c) for consensus variants, the voting rule (min_agreeing or weights)
#
# Beads jbfg.2 + n3rf.2 + n3rf.4 had to apply the same v2 cocktail fix in
# THREE different builder bodies. This factory + registry centralizes the
# decision so future fixes touch one place.
#
# Cocktails live as kwargs-builder callables -- each takes ``(config,
# candidate_elements)`` and returns the kwargs dict to pass to the
# identifier's constructor (minus the boilerplate ``atomic_db`` /
# ``resolving_power`` / JAX-flag plumbing handled by the factory itself).
# ---------------------------------------------------------------------------


def _alias_preset_kwargs(name: str) -> Dict[str, Any]:
    """Return a fresh copy of the named ALIAS preset from ``alias.ALIAS_PRESETS``.

    Lazy import keeps ``cflibs.benchmark.unified`` importable without
    eagerly pulling JAX-touching identifier code (mirrors the other
    ``cflibs.inversion.identify.alias`` lazy-imports in this file).
    The copy means downstream mutation (e.g. layering config-pulled
    overrides) cannot leak into the shared registry.
    """
    from cflibs.inversion.identify.alias import ALIAS_PRESETS

    return dict(ALIAS_PRESETS[name])


def _alias_cocktail_strict(
    config: Dict[str, Any],
    candidate_elements: List[str],  # noqa: ARG001 - signature parity
) -> Dict[str, Any]:
    """Strict ALIAS cocktail (workflow ``alias``).

    Starts from ``ALIAS_PRESETS["strict"]`` and overrides the 4 sweepable
    thresholds + ``boltzmann_r2_min`` from ``config`` so the standalone
    ``alias`` workflow can sweep them.  ``r2_gate_mode='fixed'`` and the
    other static kwargs come from the preset registry — keeping the
    precision-king baseline at ``.swarm/identifier-f1-baseline.json`` intact.
    """
    kwargs = _alias_preset_kwargs("strict")
    kwargs["intensity_threshold_factor"] = float(config["intensity_threshold_factor"])
    kwargs["detection_threshold"] = float(config["detection_threshold"])
    kwargs["chance_window_scale"] = float(config["chance_window_scale"])
    kwargs["max_lines_per_element"] = int(config["max_lines_per_element"])
    kwargs["boltzmann_r2_min"] = float(config.get("boltzmann_r2_min", 0.85))
    return kwargs


def _alias_cocktail_v2(
    config: Dict[str, Any],
    candidate_elements: List[str],  # noqa: ARG001 - signature parity
) -> Dict[str, Any]:
    """Phase D v2 cocktail -- the ftp1+dj6y sweep winner.

    Starts from ``ALIAS_PRESETS["v2"]`` (which bakes in
    ``r2_gate_mode='adaptive_t'`` from PR #175 gamma ftp1 and
    ``relative_cl_per_ion_stage=True`` from PR #176 epsilon dj6y) and
    overrides the two sweepable thresholds from ``config``.  Threshold
    kwargs intentionally NOT in the preset so the constructor's strict
    defaults (3.0 / 0.02) apply unless the caller pins them.
    """
    kwargs = _alias_preset_kwargs("v2")
    kwargs["chance_window_scale"] = float(config["chance_window_scale"])
    kwargs["max_lines_per_element"] = int(config["max_lines_per_element"])
    return kwargs


def _alias_cocktail_high_recall_v2(
    config: Dict[str, Any],
    candidate_elements: List[str],  # noqa: ARG001 - signature parity
) -> Dict[str, Any]:
    """High-recall ALIAS with v2 gates (bead n3rf.2 fix).

    Starts from ``ALIAS_PRESETS["high_recall_v2"]`` (which adds
    ``high_recall=True`` on top of the v2 cocktail so the looser peak
    thresholds are not immediately re-rejected) and overrides the two
    sweepable thresholds from ``config``.
    """
    kwargs = _alias_preset_kwargs("high_recall_v2")
    kwargs["chance_window_scale"] = float(config["chance_window_scale"])
    kwargs["max_lines_per_element"] = int(config["max_lines_per_element"])
    return kwargs


# ALIAS cocktail registry. Maps a stable preset name to (cocktail_fn,
# parameters_tag). ``parameters_tag`` is recorded as
# ``result.parameters['alias_mode']`` for downstream audit-ability; ``None``
# means do not record the tag (preserves the strict ``alias`` workflow's
# legacy behavior of not setting ``alias_mode``).
_ALIAS_COCKTAILS: Dict[str, Tuple[Callable[..., Dict[str, Any]], Optional[str]]] = {
    "strict": (_alias_cocktail_strict, None),
    "v2": (_alias_cocktail_v2, "v2_ftp1_plus_dj6y"),
    "high_recall_v2": (_alias_cocktail_high_recall_v2, "high_recall_v2_gates"),
}


@dataclass(frozen=True)
class _BinaryVoting:
    """Binary consensus rule: at least ``min_agreeing`` voters must fire."""

    min_agreeing: int
    voter_names: Tuple[str, ...]


@dataclass(frozen=True)
class _WeightedVoting:
    """Confidence-weighted consensus rule.

    Per-voter weight defaults live in ``default_weights``; the run-time
    ``config`` dict may override each via ``w_<name>`` keys and the
    threshold via ``weight_threshold`` (matching the existing
    ``hybrid_consensus_weighted`` knob names).
    """

    voter_names: Tuple[str, ...]
    default_weights: Dict[str, float]
    default_threshold: float


# Sibling-identifier preset signatures. Each entry is (sibling_kind, kwargs)
# where ``sibling_kind`` selects how the factory constructs that voter.
# kwargs are STATIC -- no per-call config plumbing -- mirroring the
# hand-rolled hybrid_consensus builders verbatim.
_SIBLING_COMB_STRICT: Tuple[str, Dict[str, Any]] = (
    "comb",
    {
        "min_correlation": 0.08,
        "tooth_activation_threshold": 0.35,
        "relative_threshold_scale": 1.4,
        "min_aki_gk": 3000.0,
    },
)
_SIBLING_CORRELATION_STRICT: Tuple[str, Dict[str, Any]] = (
    "correlation",
    {
        "min_confidence": 0.008,
        "relative_threshold_scale": 1.2,
        "min_line_strength": 1000.0,
        "T_range_K": (5000, 15000),
        "T_steps": 7,
        "n_e_range_cm3": (1e15, 5e17),
    },
)
_SIBLING_NNLS_DEFAULT: Tuple[str, Dict[str, Any]] = (
    "nnls",
    {
        "detection_snr": 3.0,
        "continuum_degree": 2,
        "fallback_T_K": 10000.0,
        "fallback_ne_cm3": 1e17,
    },
)


def _build_alias_voter(
    db,
    candidate_elements: List[str],
    rp: float,
    cocktail_kwargs: Dict[str, Any],
):
    """Construct one ALIAS voter for a consensus workflow."""
    from cflibs.inversion.identify.alias import ALIASIdentifier

    return ALIASIdentifier(
        atomic_db=db,
        elements=candidate_elements,
        resolving_power=rp,
        **cocktail_kwargs,
        **_jax_identifier_flags_for(ALIASIdentifier),
    )


def _build_sibling_voter(
    kind: str,
    static_kwargs: Dict[str, Any],
    *,
    db,
    candidate_elements: List[str],
    rp: float,
    nnls_basis,
):
    """Construct one non-ALIAS voter for a consensus workflow."""
    if kind == "comb":
        from cflibs.inversion.identify.comb import CombIdentifier

        return CombIdentifier(
            atomic_db=db,
            elements=candidate_elements,
            resolving_power=rp,
            **static_kwargs,
            **_jax_identifier_flags_for(CombIdentifier),
        )
    if kind == "correlation":
        from cflibs.inversion.identify.correlation import CorrelationIdentifier

        return CorrelationIdentifier(
            atomic_db=db,
            elements=candidate_elements,
            resolving_power=rp,
            **static_kwargs,
            **_jax_identifier_flags_for(CorrelationIdentifier),
        )
    if kind == "nnls":
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

        return SpectralNNLSIdentifier(
            basis_library=nnls_basis,
            **static_kwargs,
            **_jax_identifier_flags_for(SpectralNNLSIdentifier),
        )
    raise ValueError(f"Unknown sibling identifier kind: {kind!r}")


def _identify_with_voter(kind: str, identifier, spectrum: BenchmarkSpectrum):
    """Dispatch ``identifier.identify(...)`` honoring each voter's signature."""
    if kind == "alias":
        return identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
    if kind == "comb":
        return identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
    if kind == "correlation":
        return identifier.identify(spectrum.wavelength_nm, spectrum.intensity, mode="classic")
    if kind == "nnls":
        return identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
    raise ValueError(f"Unknown voter kind: {kind!r}")


def _make_predictor(
    identifier_cls,
    preset_name: str,
    voting: Optional[Any] = None,
    sibling_identifiers: Sequence[Tuple[str, Dict[str, Any]]] = (),
) -> Callable[
    ["UnifiedBenchmarkContext", List[str], Dict[str, Any]],
    Callable[[BenchmarkSpectrum], ElementIdentificationResult],
]:
    """Parameterized predictor factory replacing the 6 hand-rolled builders.

    Parameters
    ----------
    identifier_cls
        The primary identifier class.  Currently always ``ALIASIdentifier``
        (the only class the 6 collapsed builders use as their "main" voter).
        Kept as an argument so future workflows (e.g. comb-led consensus)
        can reuse the factory without further refactor.
    preset_name
        Key into ``_ALIAS_COCKTAILS`` -- one of ``"strict" | "v2" |
        "high_recall_v2"``. Drives ``identifier_cls``'s constructor kwargs.
    voting
        ``None`` for a standalone predictor (workflows ``alias`` /
        ``alias_v2`` / ``alias_high_recall``); ``_BinaryVoting`` or
        ``_WeightedVoting`` for a consensus workflow.
    sibling_identifiers
        Sequence of ``(kind, static_kwargs)`` pairs describing each
        non-main voter.  Empty for standalone predictors.

    Notes
    -----
    The ``_jax_identifier_flags_for(cls)`` side effect (JAX x64 enablement
    -- see bead jbfg.1) still fires for every voter the factory builds.
    """
    if preset_name not in _ALIAS_COCKTAILS:
        raise KeyError(
            f"Unknown ALIAS preset {preset_name!r}; expected one of " f"{sorted(_ALIAS_COCKTAILS)}"
        )

    cocktail_fn, alias_mode_tag = _ALIAS_COCKTAILS[preset_name]

    def build_predictor(
        context: UnifiedBenchmarkContext,
        candidate_elements: List[str],
        config: Dict[str, Any],
    ) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
        def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
            from cflibs.atomic.database import AtomicDatabase

            rp = _estimate_rp_for_spectrum(spectrum)
            cocktail_kwargs = cocktail_fn(config, candidate_elements)

            with AtomicDatabase(str(context.db_path)) as db:
                main_id = identifier_cls(
                    atomic_db=db,
                    elements=candidate_elements,
                    resolving_power=rp,
                    **cocktail_kwargs,
                    **_jax_identifier_flags_for(identifier_cls),
                )

                if voting is None:
                    # Standalone path -- run the main identifier alone.
                    result = main_id.identify(spectrum.wavelength_nm, spectrum.intensity)
                    result.parameters["candidate_elements"] = list(candidate_elements)
                    if alias_mode_tag is not None:
                        result.parameters["alias_mode"] = alias_mode_tag
                    return result

                # Consensus path -- build the siblings, run all voters, combine.
                from cflibs.inversion.identify.hybrid_consensus import (
                    HybridConsensusIdentifier,
                )

                # NNLS sibling needs a basis library; resolve once per call.
                nnls_basis = None
                basis_fwhm: Optional[float] = None
                basis_mismatch: Optional[float] = None
                if any(kind == "nnls" for kind, _ in sibling_identifiers):
                    nnls_basis, basis_fwhm, basis_mismatch = context.basis_for_rp(
                        spectrum.rp_estimate
                    )

                voter_kinds: List[str] = ["alias"]
                voters: List[Any] = [main_id]
                for kind, static_kwargs in sibling_identifiers:
                    voters.append(
                        _build_sibling_voter(
                            kind,
                            static_kwargs,
                            db=db,
                            candidate_elements=candidate_elements,
                            rp=rp,
                            nnls_basis=nnls_basis,
                        )
                    )
                    voter_kinds.append(kind)

                voter_results = [
                    _identify_with_voter(kind, voter, spectrum)
                    for kind, voter in zip(voter_kinds, voters)
                ]

                if isinstance(voting, _BinaryVoting):
                    consensus = HybridConsensusIdentifier(
                        identifiers=voters,
                        elements=candidate_elements,
                        min_agreeing=voting.min_agreeing,
                        names=list(voting.voter_names),
                    )
                elif isinstance(voting, _WeightedVoting):
                    weights = {
                        name: float(config.get(f"w_{name}", voting.default_weights[name]))
                        for name in voting.voter_names
                    }
                    consensus = HybridConsensusIdentifier(
                        identifiers=voters,
                        elements=candidate_elements,
                        names=list(voting.voter_names),
                        voter_weights=weights,
                        weight_threshold=float(
                            config.get("weight_threshold", voting.default_threshold)
                        ),
                    )
                else:  # pragma: no cover - defensive
                    raise TypeError(f"Unsupported voting rule: {type(voting).__name__}")

                result = consensus.combine(voter_results)
                result.parameters["candidate_elements"] = list(candidate_elements)
                if basis_fwhm is not None:
                    result.parameters["basis_fwhm_nm"] = basis_fwhm
                    result.parameters["basis_fwhm_mismatch_nm"] = basis_mismatch
                return result

        return predictor

    return build_predictor


# Workflow-name -> factory-invocation-args registry. The shape mirrors the
# arch-review doc: each entry is the kwargs that ``_make_predictor`` would
# be called with for that workflow. Resolved into a real predictor builder
# via ``_resolve_id_workflow_preset`` below (lazy import of identifier
# classes keeps cflibs.benchmark.unified importable when those modules
# would fail to load -- e.g. missing optional deps).
ID_WORKFLOW_PRESETS: Dict[str, Dict[str, Any]] = {
    "alias": {
        "identifier": "ALIASIdentifier",
        "preset_name": "strict",
        "voting": None,
        "sibling_identifiers": (),
    },
    "alias_v2": {
        "identifier": "ALIASIdentifier",
        "preset_name": "v2",
        "voting": None,
        "sibling_identifiers": (),
    },
    "alias_high_recall": {
        "identifier": "ALIASIdentifier",
        "preset_name": "high_recall_v2",
        "voting": None,
        "sibling_identifiers": (),
    },
    # bead jbfg.2 (and the docstring on _build_hybrid_consensus_2of3_predictor
    # below) flag this 3-voter variant as empirically broken; it is retained
    # for back-compat with the existing leaderboard. ALIAS here keeps the
    # STRICT cocktail (NOT v2) -- matching the pre-refactor behavior verbatim.
    "hybrid_consensus_2of3": {
        "identifier": "ALIASIdentifier",
        "preset_name": "strict_consensus",  # see _ALIAS_COCKTAILS extension below
        "voting": _BinaryVoting(
            min_agreeing=2,
            voter_names=("alias", "comb", "correlation"),
        ),
        "sibling_identifiers": (_SIBLING_COMB_STRICT, _SIBLING_CORRELATION_STRICT),
    },
    "hybrid_consensus_2of4_with_nnls": {
        "identifier": "ALIASIdentifier",
        # ``v2_consensus`` pins ``chance_window_scale`` and
        # ``max_lines_per_element`` — the consensus config grid is ``[{}]``
        # and the pre-refactor hand-rolled builder hard-coded these values.
        # Using ``"v2"`` here would raise KeyError at runtime.
        "preset_name": "v2_consensus",
        "voting": _BinaryVoting(
            min_agreeing=2,
            voter_names=("alias", "comb", "correlation", "nnls"),
        ),
        "sibling_identifiers": (
            _SIBLING_COMB_STRICT,
            _SIBLING_CORRELATION_STRICT,
            _SIBLING_NNLS_DEFAULT,
        ),
    },
    "hybrid_consensus_weighted": {
        "identifier": "ALIASIdentifier",
        # ``v2_consensus`` for the same reason as above — the weighted
        # consensus config grid is ``[{"weight_threshold": ...}]``, no
        # ALIAS thresholds, so the pinned cocktail is required.
        "preset_name": "v2_consensus",
        "voting": _WeightedVoting(
            voter_names=("alias", "comb", "correlation", "nnls"),
            default_weights={"alias": 0.30, "comb": 0.12, "correlation": 0.12, "nnls": 0.46},
            default_threshold=0.40,
        ),
        "sibling_identifiers": (
            _SIBLING_COMB_STRICT,
            _SIBLING_CORRELATION_STRICT,
            _SIBLING_NNLS_DEFAULT,
        ),
    },
}


# Consensus cocktails use the static ``ALIAS_PRESETS`` entries verbatim --
# their config grids are empty (or weight_threshold-only for the weighted
# variant), so no per-call overrides are needed.  Kept as separate
# closure-style functions because ``_ALIAS_COCKTAILS`` dispatches to
# callables (the standalone ``strict`` / ``v2`` / ``high_recall_v2`` paths
# DO read from config; aligning the consensus paths to the same shape
# keeps the factory body uniform).
def _alias_cocktail_strict_consensus(
    config: Dict[str, Any],  # noqa: ARG001 - consensus config grid is [{}]
    candidate_elements: List[str],  # noqa: ARG001 - signature parity
) -> Dict[str, Any]:
    """Strict ALIAS cocktail with thresholds PINNED (used by 2-of-3 consensus).

    Returns ``ALIAS_PRESETS["strict"]`` verbatim.  Mirrors the pre-refactor
    ``_build_hybrid_consensus_2of3_predictor`` body which used the same
    hard-coded strict thresholds the standalone ``alias`` workflow's
    defaults resolve to.
    """
    return _alias_preset_kwargs("strict")


_ALIAS_COCKTAILS["strict_consensus"] = (_alias_cocktail_strict_consensus, None)


def _alias_cocktail_v2_consensus(
    config: Dict[str, Any],  # noqa: ARG001 - consensus config grid is [{}] / weight_threshold-only
    candidate_elements: List[str],  # noqa: ARG001 - signature parity
) -> Dict[str, Any]:
    """V2 ALIAS cocktail with thresholds PINNED (used by 2-of-4 + weighted consensus).

    Returns ``ALIAS_PRESETS["consensus_voter"]`` verbatim -- the
    physics-equivalent of ``v2`` with the two sweepable thresholds pinned
    to ``chance_window_scale=0.4`` / ``max_lines_per_element=30`` (matching
    the pre-refactor ``_build_hybrid_consensus_*`` builders).
    """
    return _alias_preset_kwargs("consensus_voter")


_ALIAS_COCKTAILS["v2_consensus"] = (_alias_cocktail_v2_consensus, "v2_ftp1_plus_dj6y")


def _resolve_id_workflow_preset(
    preset_key: str,
) -> Callable[
    ["UnifiedBenchmarkContext", List[str], Dict[str, Any]],
    Callable[[BenchmarkSpectrum], ElementIdentificationResult],
]:
    """Build the ``IDWorkflowSpec.build_predictor`` callable for *preset_key*.

    Lazy-imports the identifier class so this resolver can sit at module
    scope without forcing every importer of ``cflibs.benchmark.unified`` to
    also import e.g. ``cflibs.inversion.identify.alias`` (which pulls in
    JAX-touching code paths).
    """
    spec = ID_WORKFLOW_PRESETS[preset_key]
    if spec["identifier"] == "ALIASIdentifier":
        from cflibs.inversion.identify.alias import ALIASIdentifier as _cls
    else:  # pragma: no cover - extension point
        raise NotImplementedError(
            f"Unknown identifier class for preset {preset_key!r}: {spec['identifier']!r}"
        )
    return _make_predictor(
        _cls,
        spec["preset_name"],
        voting=spec["voting"],
        sibling_identifiers=spec["sibling_identifiers"],
    )


def _build_alias_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    """Thin wrapper over :func:`_make_predictor` for back-compat.

    See ``ID_WORKFLOW_PRESETS["alias"]`` for the underlying preset --
    strict thresholds pulled from ``config`` (precision-king defaults).
    """
    return _resolve_id_workflow_preset("alias")(context, candidate_elements, config)


def _build_comb_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.identify.comb import CombIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = CombIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                min_correlation=float(config["min_correlation"]),
                tooth_activation_threshold=float(config["tooth_activation_threshold"]),
                relative_threshold_scale=float(config["relative_threshold_scale"]),
                min_aki_gk=3000.0,
                **_jax_identifier_flags_for(CombIdentifier),
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
        from cflibs.inversion.identify.correlation import CorrelationIdentifier

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
                **_jax_identifier_flags_for(CorrelationIdentifier),
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
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=float(config["detection_snr"]),
            continuum_degree=int(config["continuum_degree"]),
            fallback_T_K=float(config["fallback_T_K"]),
            fallback_ne_cm3=1e17,
            **_jax_identifier_flags_for(SpectralNNLSIdentifier),
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
        from cflibs.inversion.identify.hybrid import HybridIdentifier

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


def _build_hybrid_consensus_2of3_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    """Thin wrapper over :func:`_make_predictor` for back-compat.

    See ``ID_WORKFLOW_PRESETS["hybrid_consensus_2of3"]`` -- ALIAS
    (STRICT pinned cocktail, NOT v2), Comb, Correlation; 2-of-3 binary
    majority vote.

    .. warning:: bead ``CF-LIBS-improved-jbfg.2`` -- this 3-voter design
        is empirically broken (Phase 2 macro_F1=0.028, worse than
        ``comb`` alone at 0.014). Use the 2-of-4 variant
        (``hybrid_consensus_2of4_with_nnls``) instead.
    """
    return _resolve_id_workflow_preset("hybrid_consensus_2of3")(context, candidate_elements, config)


def _build_hybrid_consensus_2of4_with_nnls_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    """Thin wrapper over :func:`_make_predictor` for back-compat.

    See ``ID_WORKFLOW_PRESETS["hybrid_consensus_2of4_with_nnls"]`` --
    ALIAS (v2 cocktail per bead jbfg.2), Comb, Correlation, NNLS;
    2-of-4 binary majority vote. Reinstates NNLS as a voter: removing it
    from the consensus pool drove the 2-of-3 variant to F1=0.028 (worse
    than ``comb`` alone), since NNLS at F1=0.399 is the strongest single
    identifier on Vrabel rp=30k.
    """
    return _resolve_id_workflow_preset("hybrid_consensus_2of4_with_nnls")(
        context, candidate_elements, config
    )


def _build_hybrid_consensus_weighted_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    """Thin wrapper over :func:`_make_predictor` for back-compat.

    See ``ID_WORKFLOW_PRESETS["hybrid_consensus_weighted"]`` -- same 4
    voters as the 2-of-4 binary variant (ALIAS-v2, Comb, Correlation,
    NNLS) but with weighted-confidence voting. Default weights
    (alias=0.30, comb=0.12, correlation=0.12, nnls=0.46) reflect Phase 4
    standalone macro_F1; threshold defaults to 0.40 and is sweepable via
    ``config['weight_threshold']``. At threshold=0.40 the rule permits
    NNLS alone to pass (w=0.46 >= 0.40), preserving NNLS-only-TPs that
    the binary rule discards.
    """
    return _resolve_id_workflow_preset("hybrid_consensus_weighted")(
        context, candidate_elements, config
    )


def _build_voigt_alias_predictor(
    context: UnifiedBenchmarkContext,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], ElementIdentificationResult]:
    from cflibs.inversion.deconvolution import deconvolve_peaks
    from cflibs.inversion.preprocess.preprocessing import estimate_baseline

    def predictor(spectrum: BenchmarkSpectrum) -> ElementIdentificationResult:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.identify.alias import ALIASIdentifier

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
                **_jax_identifier_flags_for(ALIASIdentifier),
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
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=0.0,
            continuum_degree=int(config["continuum_degree"]),
            fallback_T_K=8000.0,
            fallback_ne_cm3=1e17,
            **_jax_identifier_flags_for(SpectralNNLSIdentifier),
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
        "alias_v2": IDWorkflowSpec(
            "alias_v2",
            _alias_v2_workflow_configs(quick),
            _build_alias_v2_predictor,
            _config_name,
        ),
        "alias_high_recall": IDWorkflowSpec(
            "alias_high_recall",
            _alias_high_recall_workflow_configs(quick),
            _build_alias_high_recall_predictor,
            _config_name,
        ),
        # 8-cell alias-fix sweep harness (Phase C, jaunty-weaving-mist):
        # enumerates 2^3 combinations of ftp1/762f/dj6y fix flags. Each cell
        # shares the strict threshold defaults (pinned in
        # ``_ALIAS_SWEEP_BASE_KWARGS``); the per-cell predictor closure
        # differs only in the fix-flag kwargs.
        **{
            f"alias_sweep_{cell_name}": IDWorkflowSpec(
                f"alias_sweep_{cell_name}",
                _alias_sweep_workflow_configs(quick),
                _build_alias_sweep_predictor_factory(cell_kwargs, cell_name),
                _config_name,
            )
            for cell_name, cell_kwargs in _ALIAS_SWEEP_CELLS
        },
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
        "hybrid_consensus_2of3": IDWorkflowSpec(
            "hybrid_consensus_2of3",
            _hybrid_consensus_2of3_workflow_configs(quick),
            _build_hybrid_consensus_2of3_predictor,
            _config_name,
        ),
        "hybrid_consensus_2of4_with_nnls": IDWorkflowSpec(
            "hybrid_consensus_2of4_with_nnls",
            [{}],  # singleton config — consensus rule is the experiment
            _build_hybrid_consensus_2of4_with_nnls_predictor,
            _config_name,
        ),
        "hybrid_consensus_weighted": IDWorkflowSpec(
            "hybrid_consensus_weighted",
            (
                [
                    # Weight grid sweeps the threshold; weights themselves fixed
                    # to Phase 4 F1 ratios. quick=True picks just the middle
                    # threshold; full grid sweeps three thresholds.
                    {"weight_threshold": 0.40},
                    {"weight_threshold": 0.35},
                    {"weight_threshold": 0.45},
                ]
                if not quick
                else [{"weight_threshold": 0.40}]
            ),
            _build_hybrid_consensus_weighted_predictor,
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
    weighting = _validate_boltzmann_weighting(config.get("weighting"))

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
            # Extract solver-specific parameters from config (e.g., two_region)
            solver_kwargs = {
                k: v for k, v in config.items() if k not in {"fit_method", "closure_mode"}
            }
            solver_kwargs["weighting"] = weighting
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
                **solver_kwargs,
            )
        if result is None:
            raise RuntimeError("Iterative composition workflow failed")

        closure_mode = str(config["closure_mode"])
        prediction = {"concentrations": result}
        if closure_mode == "dirichlet_residual":
            total_conc = sum(result.values())
            prediction["gamma_residual"] = float(max(0.0, 1.0 - total_conc))

        return prediction

    return predictor


def _fit_iterative_jax_pipeline(
    _context: UnifiedBenchmarkContext,
    _train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
]:
    """JAX-accelerated iterative CF-LIBS pipeline.

    Mirrors :func:`_fit_iterative_pipeline` but routes the inner-loop linear
    algebra through ``IterativeCFLIBSSolverJax`` so the heavy Boltzmann +
    closure passes execute on GPU when ``JAX_PLATFORMS=cuda`` is set. When
    the JAX solver is unavailable (Agent B's solver hasn't landed, or JAX
    isn't installed), the workflow logs a warning and falls back to the
    numpy ``IterativeCFLIBSSolver`` path so the benchmark gate keeps running
    end-to-end on CPU-only hosts.
    """
    use_jax = bool(HAS_JAX and HAS_JAX_ITERATIVE_SOLVER)
    if use_jax:
        import jax

        # Ensure 64-bit precision for Boltzmann fits; matches PR #269
        jax.config.update("jax_enable_x64", True)
        if os.environ.get("JAX_PLATFORMS") == "cuda":
            jax.config.update("jax_platform_name", "gpu")

    if not use_jax:
        logger.warning(
            "iterative_jax: falling back to numpy iterative pipeline "
            "(HAS_JAX=%s, HAS_JAX_ITERATIVE_SOLVER=%s)",
            HAS_JAX,
            HAS_JAX_ITERATIVE_SOLVER,
        )

    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional[ElementIdentificationResult],
    ) -> Dict[str, Any]:
        elements = list(candidate_elements)
        if not elements:
            raise ValueError(
                "No candidate elements available for iterative_jax composition workflow"
            )

        if use_jax:
            from cflibs.atomic.database import AtomicDatabase
            from cflibs.inversion.identify.line_detection import detect_line_observations

            with AtomicDatabase(str(_context.db_path)) as db:
                detection = detect_line_observations(
                    spectrum.wavelength_nm,
                    spectrum.intensity,
                    db,
                    elements=elements,
                )
                observations = detection.observations if detection is not None else []
                if not observations:
                    raise RuntimeError("iterative_jax: no matched line observations for spectrum")
                # IterativeCFLIBSSolverJax shares the same call surface as the
                # numpy IterativeCFLIBSSolver — we instantiate, then solve.
                solver = IterativeCFLIBSSolverJax(atomic_db=db)
                result = solver.solve(
                    observations,
                    closure_mode=str(config.get("closure_mode", "standard")),
                )
            if result is None or not getattr(result, "concentrations", None):
                raise RuntimeError("iterative_jax composition workflow failed")
            payload: Dict[str, Any] = {"concentrations": dict(result.concentrations)}
            t_K = getattr(result, "temperature_K", None)
            if t_K is not None:
                payload["temperature_K"] = float(t_K)
            ne = getattr(result, "electron_density_cm3", None)
            if ne is not None:
                payload["electron_density_cm3"] = float(ne)
            payload["solver_backend"] = "jax"
            return payload

        # --- numpy fallback (matches _fit_iterative_pipeline) ---
        from cflibs.atomic.database import AtomicDatabase

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
            raise RuntimeError("iterative_jax (numpy fallback) failed")
        return {"concentrations": result, "solver_backend": "numpy_fallback"}

    return predictor


def _build_checkpoint_record(
    spectrum: BenchmarkSpectrum,
    payload: Dict[str, Any],
    elapsed_s: float,
    workflow_name: str,
    config_name: str,
) -> CompositionEvaluationRecord:
    """Build a CompositionEvaluationRecord from per-spectrum data.

    Extracted helper for the bayesian predictor closure so it lives next
    to its only call site without expanding the cognitive complexity of
    that closure.

    Parameters
    ----------
    spectrum : BenchmarkSpectrum
        Input spectrum with metadata.
    payload : Dict[str, Any]
        Result dict from the predictor containing concentrations, etc.
    elapsed_s : float
        Per-spectrum elapsed time (seconds).
    workflow_name : str
        Name of the composition workflow (e.g. "bayesian").
    config_name : str
        Configuration name string.

    Returns
    -------
    CompositionEvaluationRecord
        Checkpoint record ready for emission.
    """
    return CompositionEvaluationRecord(
        **_spectrum_metadata_fields(spectrum),
        id_workflow_name="",
        composition_workflow_name=workflow_name,
        outer_split_id="",
        tuning_split_id=None,
        id_config_name="",
        composition_config_name=config_name,
        elapsed_seconds=float(elapsed_s),
        candidate_elements=list(payload.get("candidate_elements", [])),
        true_composition=dict(getattr(spectrum, "true_composition", {}) or {}),
        predicted_composition=dict(payload.get("concentrations", {})),
        aitchison=payload.get("aitchison", None),
        rmse=None,
        temperature_error_frac=None,
        ne_error_frac=None,
        closure_residual=None,
    )


def _bayesian_configure_jax_numpyro() -> None:
    """Pin JAX float64 + GPU platform, and NumPyro platform, when available.

    Pulled out of ``_fit_bayesian_pipeline`` to keep that factory's cognitive
    complexity below the SonarCloud threshold.
    """
    if HAS_JAX:
        import jax  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)
        if os.environ.get("JAX_PLATFORMS") == "cuda":
            jax.config.update("jax_platform_name", "gpu")
    if HAS_NUMPYRO:
        import numpyro  # noqa: PLC0415

        try:
            if os.environ.get("JAX_PLATFORMS") == "cuda":
                numpyro.set_platform("gpu")
        except Exception as exc:  # noqa: BLE001 - non-fatal platform pin
            logger.debug("bayesian: failed to set NumPyro platform: %s", exc)


def _bayesian_warn_missing_deps() -> None:
    """Emit a single warning at factory-build time listing missing deps."""
    if HAS_JAX and HAS_NUMPYRO:
        return
    missing = []
    if not HAS_JAX:
        missing.append("jax")
    if not HAS_NUMPYRO:
        missing.append("numpyro")
    logger.warning(
        "bayesian composition workflow: missing dependencies %s — "
        "predictor will raise on first invocation",
        ", ".join(missing),
    )


def _bayesian_get_or_build_sampler(
    cache: Dict[Tuple[Any, ...], Any],
    db_path: Any,
    elements: List[str],
    wl_min: float,
    wl_max: float,
    pixels: int,
    rp_estimate: Optional[float],
) -> Tuple[Any, Any]:
    """Cache-aware ``(BayesianForwardModel, MCMCSampler)`` factory.

    Key is ``(sorted elements, pixels, rounded wl bounds)`` so tiny float
    drift on the wavelength axis does not blow the JIT cache.
    """
    elements_key = tuple(sorted(elements))
    cache_key = (elements_key, pixels, round(wl_min, 3), round(wl_max, 3))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    from cflibs.inversion.solve.bayesian import (  # noqa: PLC0415
        BayesianForwardModel,
        MCMCSampler,
        NoiseParameters,
        PriorConfig,
    )

    resolving_power = float(rp_estimate) if rp_estimate is not None and rp_estimate > 0 else None
    forward_model = BayesianForwardModel(
        db_path=str(db_path),
        elements=elements,
        wavelength_range=(wl_min, wl_max),
        wavelength_grid=None,
        pixels=pixels,
        resolving_power=resolving_power,
    )
    sampler = MCMCSampler(
        forward_model,
        prior_config=PriorConfig(),
        noise_params=NoiseParameters(),
    )
    cache[cache_key] = (forward_model, sampler)
    return forward_model, sampler


def _bayesian_run_mcmc(
    sampler: Any,
    obs: Any,
    *,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    target_accept_prob: float,
) -> Any:
    """Run NUTS on ``sampler``, optionally pinning to a GPU device.

    The GPU pin is best-effort: if jax + a GPU are available and the
    ``JAX_PLATFORMS=cuda`` hint is set, we ``with jax.default_device(...)``
    so the kernel doesn't sporadically fall back to CPU on cluster nodes.
    """
    run_kwargs = {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "seed": seed,
        "target_accept_prob": target_accept_prob,
        "progress_bar": False,
    }
    gpu_device = None
    try:
        import jax  # noqa: PLC0415

        if os.environ.get("JAX_PLATFORMS") == "cuda":
            gpu_device = jax.devices("gpu")[0]
    except (ImportError, RuntimeError, IndexError):
        gpu_device = None

    if gpu_device is not None:
        import jax  # noqa: PLC0415

        with jax.default_device(gpu_device):
            return sampler.run(obs, **run_kwargs)
    return sampler.run(obs, **run_kwargs)


def _bayesian_extract_concentrations(result: Any, elements: List[str]) -> Dict[str, float]:
    """Posterior-mean concentrations on the (renormalised) simplex.

    Handles both the rank-2 ``(samples, elements)`` and rank-3
    ``(chains, samples, elements)`` shapes that NumPyro produces under the
    ``chain_method='vectorized'`` configuration.
    """
    samples = result.samples
    if "concentrations" in samples:
        conc_samples = np.asarray(samples["concentrations"])
        mean_concs = np.mean(conc_samples, axis=tuple(range(conc_samples.ndim - 1)))
        concentrations = {el: float(mean_concs[i]) for i, el in enumerate(elements)}
    else:
        concentrations = {el: float(result.concentrations_mean.get(el, 0.0)) for el in elements}
    concentrations = {el: max(1e-9, v) for el, v in concentrations.items()}
    total = sum(concentrations.values())
    if total > 0:
        concentrations = {el: v / total for el, v in concentrations.items()}
    return concentrations


def _bayesian_extract_posterior(result: Any) -> Tuple[Dict[str, Any], int]:
    """Best-effort ``(posterior_samples, divergent_count)`` from MCMCResult.

    Both extractions are wrapped in broad excepts because the optional
    ArviZ inference_data path is not always present, and the benchmark
    gate must never block on diagnostic prep.
    """
    posterior_samples: Dict[str, Any] = {}
    try:
        posterior_samples = {k: np.asarray(v) for k, v in result.samples.items()}
    except Exception:  # noqa: BLE001 - never block the gate on diag prep
        posterior_samples = {}

    divergent_count = 0
    try:
        inference_data = getattr(result, "inference_data", None)
        if inference_data is not None and hasattr(inference_data, "sample_stats"):
            stats = inference_data.sample_stats
            if "diverging" in stats:
                divergent_count = int(np.asarray(stats["diverging"].values).sum())
    except Exception:  # noqa: BLE001
        divergent_count = 0
    return posterior_samples, divergent_count


def _bayesian_build_payload(
    result: Any,
    concentrations: Dict[str, float],
    aitchison: Optional[float],
    posterior_samples: Dict[str, Any],
    divergent_count: int,
) -> Dict[str, Any]:
    """Assemble the predictor return dict consumed by composition_eval."""
    convergence_status = (
        result.convergence_status.value
        if hasattr(result.convergence_status, "value")
        else str(result.convergence_status)
    )
    return {
        "concentrations": concentrations,
        "predicted_composition": concentrations,
        "aitchison": aitchison,
        "posterior_samples": posterior_samples,
        "divergent_count": divergent_count,
        "temperature_K": float(result.T_K_mean) if result.T_K_mean else None,
        "electron_density_cm3": (
            float(result.n_e_mean) if getattr(result, "n_e_mean", None) else None
        ),
        "convergence_status": convergence_status,
        "n_samples": int(result.n_samples),
        "n_chains": int(result.n_chains),
        "n_warmup": int(result.n_warmup),
        "solver_backend": "numpyro_jax",
    }


def _fit_bayesian_pipeline(
    _context: UnifiedBenchmarkContext,
    _train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional[ElementIdentificationResult]], Dict[str, Any]
]:
    """Bayesian CF-LIBS composition workflow (NumPyro NUTS, JAX backend).

    Builds a :class:`cflibs.inversion.solve.bayesian.BayesianForwardModel`
    over the spectrum's wavelength range, runs short MCMC (NUTS) on the
    observed intensity, and returns a prediction dict with point-estimate
    concentrations (posterior mean) plus the raw posterior samples so
    ``_maybe_compute_posterior_diagnostics`` can compute rhat / ess_bulk /
    divergent_transitions / psis_loo_* and stash them in
    ``CompositionEvaluationRecord.annotations``.

    The MCMC budget is intentionally small (default 200 warmup / 400
    samples / 1 chain) so per-spectrum wall time stays in the 10-30s range
    on a V100 — the benchmark gate runs across many spectra and outer
    folds.  Tighten via ``--quick``-mode parameter grid if needed.

    When JAX or NumPyro isn't installed, the workflow raises a clear error
    at predictor build time so the benchmark report shows an explicit
    failure for that workflow rather than silently regressing.
    """
    _bayesian_configure_jax_numpyro()
    _bayesian_warn_missing_deps()

    num_warmup = int(config.get("num_warmup", 200))
    num_samples = int(config.get("num_samples", 400))
    num_chains = int(config.get("num_chains", 1))
    seed = int(config.get("seed", 0))
    target_accept_prob = float(config.get("target_accept_prob", 0.8))
    pixels = int(config.get("pixels", 1024))

    # Cache samplers to avoid redundant JIT compilation.
    # Keyed by (tuple(elements), pixels, wl_min, wl_max).
    sampler_cache: Dict[Tuple[Any, ...], Any] = {}

    # Per-spectrum progress + checkpoint state. We capture the wall-clock
    # start of the loop the first time the predictor is invoked so the
    # ``elapsed_s`` reported by ``logger.info`` reflects time-on-task for
    # the bayesian workflow specifically (independent of any outer
    # benchmark setup costs). Checkpoints are written atomically via
    # ``emit_checkpoint_part`` into a parts directory, with one part-file
    # per 10 spectra. Each part-file is named via worker_slug + sequence_id
    # and written atomically (staged into .tmp, then rename), so multiple
    # closures (e.g. from concurrent folds) do not collide on disk.
    progress_state: Dict[str, Any] = {
        "loop_start": None,
        "spectrum_index": 0,
    }
    # Atomic checkpoint via emit_checkpoint_part (Phase 2 pattern).
    checkpoint_parts_dir = Path("composition_records_checkpoint.parts")
    checkpoint_run_id = new_run_id()
    checkpoint_worker_slug = make_worker_slug(checkpoint_run_id)
    checkpoint_seq = 0
    checkpoint_batch: List[CompositionEvaluationRecord] = []
    composition_workflow_name = "bayesian"
    composition_config_name_str = _config_name(config)

    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional[ElementIdentificationResult],
    ) -> Dict[str, Any]:
        # ``checkpoint_seq`` lives in the enclosing scope so it persists
        # across calls (each emit advances the sequence counter).
        nonlocal checkpoint_seq
        if not HAS_JAX or not HAS_NUMPYRO:
            raise RuntimeError(
                "bayesian composition workflow requires jax + numpyro "
                "(install with: pip install jax jaxlib numpyro)"
            )
        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for bayesian composition workflow")

        # Per-spectrum progress log. ``loop_start`` is captured BEFORE the
        # first iteration so the logged ``elapsed_cumulative`` measures time
        # since the bayesian loop began. ``spectrum_start`` (below) gives
        # the per-spectrum wall time for the checkpoint record.
        if progress_state["loop_start"] is None:
            progress_state["loop_start"] = time.monotonic()
        progress_state["spectrum_index"] += 1
        spectrum_index = int(progress_state["spectrum_index"])
        elapsed_cumulative = time.monotonic() - float(progress_state["loop_start"])
        dataset_name = getattr(spectrum, "dataset_id", None) or "unknown"
        logger.info(
            f"[bayesian] {dataset_name}: spectrum {spectrum_index} "
            f"(elapsed {elapsed_cumulative:.1f}s)"
        )
        spectrum_start = time.monotonic()

        # Spectrum prep + sampler get + MCMC run, all delegated to module-
        # level helpers to keep this closure's cognitive complexity low.
        wl = np.asarray(spectrum.wavelength_nm, dtype=float)
        intensity = np.asarray(spectrum.intensity, dtype=float)
        if wl.size == 0 or intensity.size == 0:
            raise ValueError("bayesian: empty spectrum input")
        wl_min = float(wl.min())
        wl_max = float(wl.max())

        forward_model, sampler = _bayesian_get_or_build_sampler(
            sampler_cache,
            _context.db_path,
            elements,
            wl_min,
            wl_max,
            pixels,
            spectrum.rp_estimate,
        )
        obs = np.interp(np.asarray(forward_model.wavelength), wl, intensity)
        result = _bayesian_run_mcmc(
            sampler,
            obs,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            target_accept_prob=target_accept_prob,
        )

        concentrations = _bayesian_extract_concentrations(result, elements)
        aitchison = (
            aitchison_distance(spectrum.true_composition, concentrations)
            if spectrum.true_composition
            else None
        )
        posterior_samples, divergent_count = _bayesian_extract_posterior(result)
        payload = _bayesian_build_payload(
            result, concentrations, aitchison, posterior_samples, divergent_count
        )

        # Per-spectrum elapsed_seconds is just the MCMC wall time, not the
        # cumulative loop time logged above. Atomic checkpoint via
        # emit_checkpoint_part — each closure (one per dataset/fold/config)
        # writes its own part file via worker_slug + sequence_id.
        elapsed_spectrum = time.monotonic() - spectrum_start
        checkpoint_batch.append(
            _build_checkpoint_record(
                spectrum=spectrum,
                payload={
                    "concentrations": concentrations,
                    "aitchison": aitchison,
                    "candidate_elements": elements,
                },
                elapsed_s=elapsed_spectrum,
                workflow_name=composition_workflow_name,
                config_name=composition_config_name_str,
            )
        )
        if spectrum_index % 10 == 0:
            checkpoint_seq = emit_checkpoint_part(
                parts_dir=checkpoint_parts_dir,
                run_id=checkpoint_run_id,
                worker_slug=checkpoint_worker_slug,
                seq=checkpoint_seq,
                records=checkpoint_batch,
                processed=spectrum_index,
            )
            checkpoint_batch.clear()

        return payload

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


def build_composition_workflow_registry(
    quick: bool = False,
    bayesian_mcmc_override: Optional[Dict[str, int]] = None,
) -> Dict[str, CompositionWorkflowSpec]:
    from cflibs.inversion.physics.boltzmann import FitMethod

    weighting = _validate_boltzmann_weighting("aki_inverse_variance")
    iterative_configs = [
        {"fit_method": FitMethod.SIGMA_CLIP, "closure_mode": "standard", "weighting": weighting},
        {"fit_method": FitMethod.SIGMA_CLIP, "closure_mode": "ilr", "weighting": weighting},
        {
            "fit_method": FitMethod.SIGMA_CLIP,
            "closure_mode": "dirichlet_residual",
            "weighting": weighting,
        },
        {
            "fit_method": FitMethod.SIGMA_CLIP,
            "closure_mode": "standard",
            "two_region": True,
            "weighting": weighting,
        },
    ]
    if not quick:
        iterative_configs.extend(
            [
                {
                    "fit_method": FitMethod.RANSAC,
                    "closure_mode": "standard",
                    "weighting": weighting,
                },
                {"fit_method": FitMethod.RANSAC, "closure_mode": "ilr", "weighting": weighting},
                {"fit_method": FitMethod.HUBER, "closure_mode": "standard", "weighting": weighting},
                {"fit_method": FitMethod.HUBER, "closure_mode": "ilr", "weighting": weighting},
            ]
        )

    # iterative_jax mirrors the iterative parameter grid — every config
    # exposes a fit_method + closure_mode pair so the JAX path can pick the
    # same robust regression mode and the numpy fallback stays bit-compatible.
    iterative_jax_configs = [dict(c) for c in iterative_configs]

    # Bayesian: the parameter grid is shaped to keep MCMC wall time bounded
    # for the benchmark gate.  In quick mode we run a single small
    # configuration; the full grid sweeps a couple of warmup/sample points.
    if bayesian_mcmc_override is not None:
        # CLI override (--bayesian-mcmc N_WARMUP,N_SAMPLES,N_CHAINS) collapses
        # the parameter grid to a single configuration. Used by the
        # CF-LIBS-improved-4rwe before/after Stark T-factor benchmark and any
        # other ablation that needs a single pinned MCMC budget per run.
        bayesian_configs: List[Dict[str, Any]] = [
            {
                "num_warmup": int(bayesian_mcmc_override["num_warmup"]),
                "num_samples": int(bayesian_mcmc_override["num_samples"]),
                "num_chains": int(bayesian_mcmc_override["num_chains"]),
                "seed": int(bayesian_mcmc_override.get("seed", 0)),
            }
        ]
    elif quick:
        bayesian_configs = [
            {"num_warmup": 200, "num_samples": 400, "num_chains": 1, "seed": 0},
        ]
    else:
        bayesian_configs = [
            {"num_warmup": 200, "num_samples": 400, "num_chains": 1, "seed": 0},
            {"num_warmup": 500, "num_samples": 1000, "num_chains": 1, "seed": 0},
        ]

    return {
        "iterative": CompositionWorkflowSpec(
            "iterative", iterative_configs, _fit_iterative_pipeline, _config_name
        ),
        "iterative_jax": CompositionWorkflowSpec(
            "iterative_jax",
            iterative_jax_configs,
            _fit_iterative_jax_pipeline,
            _config_name,
        ),
        "bayesian": CompositionWorkflowSpec(
            "bayesian", bayesian_configs, _fit_bayesian_pipeline, _config_name
        ),
        "joint_softmax": CompositionWorkflowSpec(
            "joint_softmax", [{}], _fit_joint_optimizer_pipeline, _config_name
        ),
        "hybrid_manifold": CompositionWorkflowSpec(
            "hybrid_manifold", [{}], _fit_hybrid_manifold_pipeline, _config_name
        ),
    }


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
        "per_stratum_summary": _compute_per_stratum_summary(pair_records),
        "subcompositional_ratio_errors": _compute_subcompositional_ratio_summary(pair_records),
    }


def _compute_per_stratum_summary(
    pair_records: Sequence[CompositionEvaluationRecord],
) -> Dict[str, Dict[str, Any]]:
    """Build the majors / minors / traces stratified summary.

    Stratifies by **certified** concentration so that a model cannot move
    elements between strata by inflating its predictions.  Each (element,
    spectrum) pair becomes a single record.
    """
    stratum_records: List[Dict[str, float]] = []
    for record in pair_records:
        if not record.scored:
            continue
        for element, true_value in record.true_composition.items():
            stratum_records.append(
                {
                    "element": element,
                    "true": float(true_value),
                    "predicted": float(record.predicted_composition.get(element, 0.0)),
                }
            )
    return stratify_per_element_errors(stratum_records)


def _compute_subcompositional_ratio_summary(
    pair_records: Sequence[CompositionEvaluationRecord],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-spectrum |log(r̂/r*)| ratio errors into pair summaries.

    Returns ``{pair_key: {n, mean, median, p95, max, pass}}`` where ``pass``
    is ``True`` when ``max <= 0.20`` (per validation/protocol.yaml
    ``ratio_log_error_max``).  Pairs with zero scored spectra report
    ``pass=True`` (vacuous) with NaN summary statistics.
    """
    pair_keys: List[str] = []
    for numerator, denominator in load_subcompositional_pairs():
        pair_keys.append(f"{numerator}/{denominator}")

    aggregated: Dict[str, List[float]] = {key: [] for key in pair_keys}
    for record in pair_records:
        if not record.scored or not record.subcompositional_ratio_errors:
            continue
        for key, value in record.subcompositional_ratio_errors.items():
            if key in aggregated and value is not None and not np.isnan(float(value)):
                aggregated[key].append(float(value))

    summary: Dict[str, Dict[str, Any]] = {}
    for key, values in aggregated.items():
        if not values:
            summary[key] = {
                "n_spectra": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "max": float("nan"),
                "pass": True,
            }
            continue
        arr = np.asarray(values, dtype=np.float64)
        max_val = float(np.max(arr))
        summary[key] = {
            "n_spectra": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": max_val,
            "pass": max_val <= 0.20,
        }
    return summary


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
        bayesian_mcmc_override: Optional[Dict[str, int]] = None,
    ):
        # Session-level seam for JAX-x64 enablement (arch review #2 candidate 2,
        # bead CF-LIBS-improved-jbfg.1).  Idempotent and gated on
        # ``CFLIBS_USE_JAX_IDENTIFIER`` inside the helper.  Replaces the prior
        # hidden side effect that lived inside ``_jax_identifier_flags_for``.
        if os.environ.get("CFLIBS_USE_JAX_IDENTIFIER", "0") == "1":
            from cflibs.core.jax_runtime import configure_for_identifiers

            configure_for_identifiers()
        self.context = UnifiedBenchmarkContext(db_path=db_path, basis_dir=basis_dir)
        self.id_registry = build_id_workflow_registry(quick=quick)
        self.composition_registry = build_composition_workflow_registry(
            quick=quick,
            bayesian_mcmc_override=bayesian_mcmc_override,
        )

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
        output_format: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Write benchmark artifacts.

        Parameters
        ----------
        output_format
            One of ``"parquet"`` (default), ``"json"``, or ``"both"``.
            ``parquet`` skips the legacy per-record JSON dumps and writes
            a single ``results.parquet`` instead — see
            ``docs/results-parquet-schema.md``. ``json`` keeps the
            legacy behaviour; ``both`` writes both. When ``None`` the
            ``CFLIBS_OUTPUT_FORMAT`` env var is consulted, then the
            default ``parquet`` is used. Falls back to ``json``
            automatically if pyarrow isn't importable.
        run_metadata
            Optional dict forwarded into the parquet rows under the
            run-metadata columns (``cell``, ``identifier``,
            ``platform``, ``seed``, ``iter_index``,
            ``experiment_label``). Ignored when only JSON is written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        id_summary = summarize_id_records(id_records)
        composition_summary = summarize_composition_records(composition_records)

        if output_format is None:
            output_format = os.environ.get("CFLIBS_OUTPUT_FORMAT", "parquet")
        output_format = output_format.lower()
        if output_format not in {"parquet", "json", "both"}:
            raise ValueError(
                f"output_format must be one of 'parquet', 'json', 'both'; got {output_format!r}"
            )

        # Lazy-import results module so older test envs without pyarrow
        # can still drive the JSON path.
        from cflibs.benchmark import results as results_module

        if output_format in {"parquet", "both"} and not results_module.parquet_available():
            # Fall back to JSON if pyarrow isn't importable — better to
            # write *something* than to crash mid-run.
            print(
                "[unified] pyarrow not available; falling back to " "output_format='json' (legacy)."
            )
            output_format = "json"

        write_parquet = output_format in {"parquet", "both"}
        write_json = output_format in {"json", "both"}

        outputs: Dict[str, Path] = {
            "id_summary_json": output_dir / "id_summary.json",
            "id_selections_json": output_dir / "id_config_selections.json",
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

        if write_json:
            outputs.update(
                {
                    "id_records_json": output_dir / "id_records.json",
                    "id_records_csv": output_dir / "id_records.csv",
                    "composition_records_json": output_dir / "composition_records.json",
                    "composition_records_csv": output_dir / "composition_records.csv",
                }
            )
            _write_json(outputs["id_records_json"], _records_to_rows(id_records))
            _write_csv(outputs["id_records_csv"], _records_to_rows(id_records))
            _write_json(outputs["composition_records_json"], _records_to_rows(composition_records))
            _write_csv(outputs["composition_records_csv"], _records_to_rows(composition_records))

        if write_parquet:
            outputs["results_parquet"] = output_dir / "results.parquet"
            results_module.write_parquet(
                output_path=outputs["results_parquet"],
                id_records=id_records,
                composition_records=composition_records,
                run_metadata=run_metadata,
            )

        _write_json(outputs["id_summary_json"], id_summary)
        _write_json(outputs["id_selections_json"], list(id_selections))
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
