"""
Per-spectrum composition workflow evaluator.

This module owns the inner loop of the composition benchmark:

* :class:`CompositionEvaluationRecord` — the row dataclass written to
  ``composition_records.parquet`` / ``composition_records.json``.
* :func:`evaluate_composition_workflow` — iterate over a spectra list,
  run the ID predictor then the composition predictor on each
  non-blind spectrum, score the result, append to records, and
  incrementally checkpoint via
  :mod:`cflibs.benchmark.checkpoint`.
* Companion record-builders and small helpers used by both the
  evaluator and tuning loops in ``unified.py``.

Why a dedicated module
----------------------
Extracted from ``cflibs/benchmark/unified.py`` (which had grown to ~3.9
kLoC mixing the workflow registry, the runner class, output writers,
and this evaluator) so the evaluator can be tested + reasoned about in
isolation. ``unified.py`` re-exports every symbol from this module so
existing callers — ``UnifiedBenchmarkRunner.run_composition``,
``tune_composition_workflow``, third-party importers — keep working
verbatim.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from cflibs.benchmark.checkpoint import (
    emit_checkpoint_part,
    make_worker_slug,
    new_run_id,
)
from cflibs.benchmark.composition_metrics import (
    aitchison_distance,
    per_element_error,
    rmse_composition,
    subcompositional_ratio_errors,
)
from cflibs.benchmark.dataset import BenchmarkSpectrum, TruthType

if TYPE_CHECKING:
    from cflibs.inversion.common.element_id import ElementIdentificationResult


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


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
    # Per-pair |log(r̂/r*)| subcompositional ratio errors keyed by
    # ``"<num>/<den>"``; populated by the composition workflow runner so the
    # values land in composition_records.json.  NaN entries indicate pairs
    # that could not be scored (truth ratio undefined).
    subcompositional_ratio_errors: Dict[str, float] = field(default_factory=dict)
    scored: bool = True
    failure_reason: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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


_POSTERIOR_EXCLUDE_KEYS = frozenset(
    {
        "concentrations",
        "posterior_samples",
        "mcmc_result",
        "log_likelihood",
        "divergent_count",
    }
)


def _maybe_compute_posterior_diagnostics(
    prediction: Dict[str, Any],
    candidate_elements: Sequence[str],
    true_composition: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """Compute posterior calibration diagnostics for a Bayesian
    composition workflow, or return ``None`` if no MCMC samples were
    emitted by the workflow.

    The function looks for a ``posterior_samples`` mapping in
    ``prediction`` (or unpacks it from a ``mcmc_result`` object whose
    ``samples`` attribute mimics ``MCMCResult.samples``). The certified
    composition vector is supplied as the ``concentrations[i]`` entries
    in the certified-values mapping passed to the diagnostic.

    Returns the diagnostic dataclass as a JSON-serialisable dict so it
    can be stashed in ``CompositionEvaluationRecord.annotations`` and
    survive the ``json.dumps`` round-trip in
    ``write_outputs/composition_records.json``.
    """
    samples = prediction.get("posterior_samples")
    if samples is None:
        mcmc_result = prediction.get("mcmc_result")
        if mcmc_result is not None and hasattr(mcmc_result, "samples"):
            samples = mcmc_result.samples
    if not samples:
        return None

    log_likelihood = prediction.get("log_likelihood")
    divergent_count = int(prediction.get("divergent_count", 0))

    # Build certified_values for the parameters we have ground-truth on:
    # the concentration vector indexed by candidate-element order.
    certified: Dict[str, float] = {}
    for i, element in enumerate(candidate_elements):
        if element in true_composition:
            certified[f"concentrations[{i}]"] = float(true_composition[element])

    try:
        from cflibs.benchmark.posterior_metrics import compute_posterior_diagnostics

        diag = compute_posterior_diagnostics(
            samples,
            certified_values=certified or None,
            log_likelihood=log_likelihood,
            divergent_count=divergent_count,
        )
        return diag.as_dict()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"posterior_metrics_failed: {exc}"}


def _compose_annotations(
    prediction: Dict[str, Any], posterior_diag: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Drop bulky posterior-sample arrays out of the annotations dict and
    tack the computed posterior diagnostics on instead."""
    annotations: Dict[str, Any] = {
        key: value for key, value in prediction.items() if key not in _POSTERIOR_EXCLUDE_KEYS
    }
    if posterior_diag is not None:
        annotations["posterior_diagnostics"] = posterior_diag
    return annotations


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _spectrum_metadata_fields(spectrum: BenchmarkSpectrum) -> Dict[str, Any]:
    """Per-spectrum metadata fields shared by every CompositionEvaluationRecord builder.

    Pulled out so the success / failure / bayesian-checkpoint builders do not
    each carry their own copy of the 9-line spectrum-attribute dump. Returns
    a kwargs dict suitable for ``**``-spreading into the dataclass constructor.
    """
    truth_type_attr = getattr(spectrum, "truth_type", None)
    if truth_type_attr is None:
        truth_type_value = ""
    elif hasattr(truth_type_attr, "value"):
        truth_type_value = truth_type_attr.value
    else:
        truth_type_value = str(truth_type_attr)
    return {
        "dataset_id": getattr(spectrum, "dataset_id", None) or "unknown",
        "spectrum_id": getattr(spectrum, "spectrum_id", ""),
        "group_id": getattr(spectrum, "group_id", None),
        "specimen_id": getattr(spectrum, "specimen_id", None),
        "instrument_id": getattr(spectrum, "instrument_id", None),
        "truth_type": truth_type_value,
        "rp_estimate": getattr(spectrum, "rp_estimate", None),
        "label_cardinality": getattr(spectrum, "label_cardinality", None),
        "spectrum_kind": getattr(spectrum, "spectrum_kind", None),
    }


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
    ratio_errors = subcompositional_ratio_errors(concentrations, true_comp)
    posterior_diag = _maybe_compute_posterior_diagnostics(prediction, candidate_elements, true_comp)
    return CompositionEvaluationRecord(
        **_spectrum_metadata_fields(spectrum),
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
        subcompositional_ratio_errors=ratio_errors,
        annotations=_compose_annotations(prediction, posterior_diag),
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
        **_spectrum_metadata_fields(spectrum),
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


# ---------------------------------------------------------------------------
# Per-spectrum loop
# ---------------------------------------------------------------------------


def evaluate_composition_workflow(
    spectra: Sequence[BenchmarkSpectrum],
    id_workflow_name: str,
    id_config_name: str,
    id_predictor: Callable[[BenchmarkSpectrum], "ElementIdentificationResult"],
    composition_workflow: Any,
    composition_predictor: Callable[
        [BenchmarkSpectrum, Sequence[str], Optional["ElementIdentificationResult"]], Dict[str, Any]
    ],
    outer_split_id: str,
    tuning_split_id: Optional[str],
    composition_config_name: str,
) -> List[CompositionEvaluationRecord]:
    """Evaluate one (id_workflow, composition_workflow) pair across a
    sequence of spectra.

    Parameters mirror the in-tree ``unified.py`` contract exactly; this
    function is the implementation that
    ``cflibs.benchmark.unified.evaluate_composition_workflow`` resolves
    to via re-export.

    ``composition_workflow`` is typed loosely as ``Any`` to avoid a
    cycle on ``CompositionWorkflowSpec`` (defined in ``unified.py``);
    only ``composition_workflow.name`` is accessed.

    Checkpoint plumbing
    -------------------
    When ``CFLIBS_BENCH_CHECKPOINT_PATH`` is exported the function
    writes a part-file for every newly completed batch of
    ``CFLIBS_BENCH_CHECKPOINT_EVERY`` spectra so a SLURM timeout leaves
    something on disk. Part-files live under
    ``<checkpoint_path>.parts/part_<hostname>_<pid>_<runid>_<seq>.parquet``
    and can be merged via
    :func:`cflibs.benchmark.results.read_parquet_dir`.  See
    :mod:`cflibs.benchmark.checkpoint` for the primitives.
    """
    records: List[CompositionEvaluationRecord] = []

    checkpoint_path_str = os.environ.get("CFLIBS_BENCH_CHECKPOINT_PATH")
    try:
        checkpoint_every = int(os.environ.get("CFLIBS_BENCH_CHECKPOINT_EVERY", "1"))
    except ValueError:
        checkpoint_every = 1
    # Guard against misconfiguration: ``0`` or negative would deadlock the
    # modulo check (ZeroDivisionError) or never trigger.
    checkpoint_every = max(1, checkpoint_every)
    checkpoint_path = Path(checkpoint_path_str) if checkpoint_path_str else None
    checkpoint_parts_dir: Optional[Path] = None
    checkpoint_part_seq = 0
    checkpoint_run_id: Optional[str] = None
    checkpoint_worker_slug = ""
    if checkpoint_path is not None:
        checkpoint_parts_dir = checkpoint_path.with_name(checkpoint_path.name + ".parts")
        checkpoint_parts_dir.mkdir(parents=True, exist_ok=True)
        # ``new_run_id`` + ``make_worker_slug`` live in
        # :mod:`cflibs.benchmark.checkpoint`. The worker_slug bakes in
        # hostname + PID + first-8-chars of the run_id so the
        # part-file filename survives SLURM --requeue (same host,
        # same PID) and cross-node PID collisions without silent
        # overwrite. See ``checkpoint.make_worker_slug`` docstring.
        checkpoint_run_id = new_run_id()
        checkpoint_worker_slug = make_worker_slug(checkpoint_run_id)

    eligible_total = sum(1 for s in spectra if s.truth_type != TruthType.BLIND)
    processed = 0

    for spectrum in spectra:
        if spectrum.truth_type == TruthType.BLIND:
            continue
        processed += 1
        # Emit directly to stderr so the marker is visible even when the
        # CF-LIBS logger has not been configured (the unified-benchmark
        # runner does not call ``setup_logging`` and Python's default
        # root level is WARNING -- swallowing ``logger.info``).
        print(
            f"[progress] {spectrum.dataset_id or 'unknown'}/"
            f"{spectrum.spectrum_id} {composition_workflow.name} "
            f"spectrum {processed}/{eligible_total} "
            f"id_workflow={id_workflow_name} "
            f"comp_workflow={composition_workflow.name}",
            file=sys.stderr,
            flush=True,
        )
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

        # Incremental checkpoint -- best-effort, never blocks the gate. Each
        # checkpoint write is a fresh part-file (O(1) I/O per write), so the
        # on-disk state grows linearly with the number of completed spectra
        # instead of quadratically as a single append-rewritten parquet would.
        if (
            checkpoint_parts_dir is not None
            and records  # guard against empty failure-only batches
            and (processed % checkpoint_every == 0)
        ):
            checkpoint_part_seq = emit_checkpoint_part(
                parts_dir=checkpoint_parts_dir,
                run_id=checkpoint_run_id,
                worker_slug=checkpoint_worker_slug,
                seq=checkpoint_part_seq,
                records=records[-checkpoint_every:],
                processed=processed,
            )

    # Final-flush: capture any trailing records that didn't trip the modulo
    # gate (e.g. ``processed == 7`` with ``checkpoint_every == 5``). Without
    # this, up to ``checkpoint_every - 1`` records would be lost on a SLURM
    # timeout that fires *between* the loop end and ``write_outputs``.
    # ``emit_checkpoint_part`` owns the sequence increment, so we pass the
    # current ``checkpoint_part_seq`` unchanged -- the helper handles the
    # ``+= 1`` internally so the on-disk file numbers are gap-free.
    if checkpoint_parts_dir is not None and records:
        trailing = processed % checkpoint_every
        if trailing > 0:
            emit_checkpoint_part(
                parts_dir=checkpoint_parts_dir,
                run_id=checkpoint_run_id,
                worker_slug=checkpoint_worker_slug,
                seq=checkpoint_part_seq,
                records=records[-trailing:],
                processed=processed,
                final_flush=True,
            )

    return records
