"""Unified goal-metric scoreboard (bead A1).

ONE command that measures the only things that matter for CF-LIBS:

* **element-identification accuracy** (precision / recall / F1),
* **composition accuracy** (RMSE in element wt% against certified truth),
* **runtime** (wall-clock per pipeline stage),

across every registered truth-bearing dataset
(:mod:`cflibs.benchmark.scoreboard_registry`), running the PRODUCTION
pipeline — :func:`cflibs.inversion.pipeline.build_pipeline_config` +
:func:`cflibs.inversion.pipeline.run_pipeline` with the default
``geological`` preset — exactly as ``cflibs analyze`` would.

Reproduce::

    JAX_PLATFORMS=cpu cflibs scoreboard --output-dir output/scoreboard

Tier policy
-----------
Each registered dataset declares a tier (:class:`DatasetEntry.tier`):
``optimization`` datasets always run; ``holdout`` datasets (the campaign
adoption gate, e.g. bhvo2_chemcam, emslibs2019) are EXCLUDED unless
``include_holdout=True`` (CLI ``--include-holdout``) so casual boards cannot
leak the gate; ``vault`` datasets (gibbons2024) are never run by this
harness. Explicitly requesting an excluded dataset is a hard error, never a
silent skip.

Candidate-set policy
--------------------
For each spectrum the pipeline is given ``candidates = truth.elements_present
UNION CONFOUNDER_ELEMENTS`` (Ag/Sn/W/Bi/Th — the known false-positive
confounders from the BHVO-2 presence gate, ``scripts/measure_bhvo2_presence.py``:
neutral-resonance lines in-band, same thermal E_k band as real majors). Truth
elements make recall measurable; the fixed confounder set makes false
positives measurable. This measures *rejection* given a candidate superset,
not open-world identification over the full periodic table.

Presence rule
-------------
An element is "called present" when its solved mass fraction is
``>= PRESENCE_EPS_MASSFRAC`` (5e-3 = 0.5 wt%) — the same rule as
``scripts/measure_bhvo2_presence.py``. A spectrum whose pipeline run *fails*
calls nothing present: its truth elements all count as false negatives (the
end user gets no answer), and the failure is additionally recorded.

Aggregation
-----------
ID metrics are micro-averaged per dataset (TP/FP/FN counts summed over run
spectra, including failures). Composition RMSE is computed per spectrum over
the certified elements (element-wt% basis, oxygen excluded) and aggregated
as the median across successfully solved spectra. Runtime is the per-dataset
median of per-stage wall clocks reported by ``run_pipeline``.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.pipeline import build_pipeline_config, run_pipeline
from cflibs.benchmark.scoreboard_registry import (
    SpectrumTruth,
    ensure_default_datasets,
    iter_datasets,
    registered_names,
)
from cflibs.benchmark.synthetic_eval import compute_binary_metrics

logger = get_logger("benchmark.scoreboard")

#: Supported inversion-pipeline implementations (J12 / M3 gate).
#:   * ``reference`` — :func:`cflibs.inversion.pipeline.run_pipeline` (parity oracle).
#:   * ``jit``       — :func:`cflibs.jitpipe.run_one` (the JAX port; on-device
#:     detect/identify/solve, host-delegated calibration + Stark — the
#:     M1-validated path; see docs/jitpipe/J9-segmented-calibration-findings.md).
PIPELINE_IMPLS = ("reference", "jit")

#: Cache of jitpipe atomic snapshots keyed by DB path (built once per board).
_JIT_SNAPSHOT_CACHE: dict[str, Any] = {}


def _jit_snapshot(db_path: str) -> Any:
    """Build (and cache) the jitpipe atomic snapshot for ``db_path``."""
    snap = _JIT_SNAPSHOT_CACHE.get(db_path)
    if snap is None:
        from cflibs.jitpipe import build_snapshot

        snap = build_snapshot(db_path)
        _JIT_SNAPSHOT_CACHE[db_path] = snap
    return snap


def _run_pipeline_jit(
    wavelength: np.ndarray, intensity: np.ndarray, atomic_db: Any, pipeline: Any
) -> tuple[Any, dict[str, Any]]:
    """Run the jittable pipeline, returning ``run_pipeline``'s ``(result, diag)`` shape.

    Keeps the scorer pipeline-agnostic: builds the ``StaticConfig`` from the
    resolved ``pipeline`` (closure mode + max iters) against a cached full-DB
    snapshot, then calls :func:`cflibs.jitpipe.run_one` with the on-device
    front-end (host-delegated segmented calibration + Stark — the M1-validated
    routing, J9 finding).
    """
    from cflibs.jitpipe import PipelineParams, StaticConfig, run_one

    db_path = str(getattr(atomic_db, "db_path", "") or "")
    snap = _jit_snapshot(db_path)
    static = StaticConfig(
        bucket_id=0,
        n_species=snap.n_species,
        level_pad=snap.level_pad[1],
        closure_mode=pipeline.closure_mode,
        max_iters=int(pipeline.max_iterations),
    )
    result = run_one(
        np.asarray(intensity, dtype=float),
        np.asarray(wavelength, dtype=float),
        snap,
        PipelineParams(),
        static,
        atomic_db=atomic_db,
        pipeline_config=pipeline,
        ondevice_front_end=True,
    )
    diagnostics = {"n_observations": 0, "stage_timings": {}, "pipeline_impl": "jit"}
    return result, diagnostics


#: Fixed false-positive probe set (see scripts/measure_bhvo2_presence.py).
CONFOUNDER_ELEMENTS = ("Ag", "Sn", "W", "Bi", "Th")

#: Mass fraction above which a solved element counts as "called present"
#: (same rule as scripts/measure_bhvo2_presence.py EPS_PRESENT).
PRESENCE_EPS_MASSFRAC = 5e-3

#: Default sampling seed (date-stamped; recorded in every artifact).
DEFAULT_SEED = 20260610

#: Artifact label for ``preset=None`` (the production default). Shared with
#: the campaign objective (scripts/campaign1/objective.py) so boards from
#: both harnesses report the same provenance string.
DEFAULT_PRESET_LABEL = "geological (production default)"


# ---------------------------------------------------------------------------
# Scoring math (unit-tested without any pipeline/DB)
# ---------------------------------------------------------------------------


def presence_confusion(
    predicted_massfrac: Mapping[str, float],
    truth_elements: frozenset[str] | set[str],
    candidates: Iterable[str],
) -> dict[str, list[str]]:
    """Classify each candidate element as TP/FP/FN under the presence rule."""
    called = {
        el for el in candidates if float(predicted_massfrac.get(el, 0.0)) >= PRESENCE_EPS_MASSFRAC
    }
    truth = set(truth_elements)
    return {
        "called_present": sorted(called),
        "tp": sorted(called & truth),
        "fp": sorted(called - truth),
        "fn": sorted(truth - called),
    }


def failure_record(
    spectrum_id: str,
    truth: SpectrumTruth,
    candidates: Sequence[str],
    error: str,
    wall_s: float,
) -> dict[str, Any]:
    """Scoreboard record for a spectrum whose pipeline run produced no result.

    A failed solve calls nothing present: every truth element counts as a
    false negative (the end user gets no answer). Shared by the scoreboard's
    pipeline except-path and the campaign objective's timeout/crash records
    (``scripts/campaign1/objective.py``) so the two failure shapes can never
    drift apart.
    """
    record: dict[str, Any] = {
        "spectrum_id": spectrum_id,
        "truth_elements": sorted(truth.elements_present),
        "candidates": list(candidates),
        "composition_basis": truth.composition_basis,
        "status": "error",
        "error": error,
        "wall_s": wall_s,
    }
    record.update(presence_confusion({}, truth.elements_present, candidates))
    return record


def precision_recall_f1(n_tp: int, n_fp: int, n_fn: int) -> tuple[float, float, float]:
    """Micro precision/recall/F1 from confusion counts (0.0 on empty denominators).

    Thin delegate to :func:`cflibs.benchmark.synthetic_eval.compute_binary_metrics`
    — the single source for this metric math (TN plays no role in micro P/R/F1).
    """
    metrics = compute_binary_metrics(n_tp, n_fp, n_fn, 0)
    return metrics["precision"], metrics["recall"], metrics["f1"]


def composition_errors(
    predicted_massfrac: Mapping[str, float],
    composition_wt: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """RMSE (wt%) and per-element signed errors over the certified elements.

    Signed error = predicted_wt - certified_wt; predictions are converted
    from solver mass fractions (0-1) to wt%. Elements absent from the solver
    output count as 0 wt% (they were dropped — that error is real).
    """
    signed = {
        el: 100.0 * float(predicted_massfrac.get(el, 0.0)) - float(cert_wt)
        for el, cert_wt in composition_wt.items()
    }
    rmse = math.sqrt(sum(err**2 for err in signed.values()) / len(signed)) if signed else 0.0
    return rmse, signed


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(np.asarray(values, dtype=float))) if values else None


# ---------------------------------------------------------------------------
# Per-spectrum execution
# ---------------------------------------------------------------------------


def _score_spectrum(
    atomic_db: Any,
    spectrum_id: str,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    truth: SpectrumTruth,
    *,
    preset: Optional[str] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
    pipeline_impl: str = "reference",
) -> dict[str, Any]:
    """Run the production pipeline on one spectrum and score it against truth."""
    candidates = sorted(set(truth.elements_present) | set(CONFOUNDER_ELEMENTS))
    record: dict[str, Any] = {
        "spectrum_id": spectrum_id,
        "truth_elements": sorted(truth.elements_present),
        "candidates": candidates,
        "composition_basis": truth.composition_basis,
        "status": "ok",
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        pipeline = build_pipeline_config(
            candidates,
            preset=preset,
            resolving_power=truth.resolving_power,
            overrides=config_overrides,
        )
        if pipeline_impl == "jit":
            result, diagnostics = _run_pipeline_jit(wavelength, intensity, atomic_db, pipeline)
        else:
            result, diagnostics = run_pipeline(wavelength, intensity, atomic_db, pipeline)
    except Exception as exc:  # noqa: BLE001 — the board must never crash
        record = failure_record(
            spectrum_id,
            truth,
            candidates,
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
        logger.warning("scoreboard: %s failed: %s", spectrum_id, record["error"])
        return record

    record["wall_s"] = time.perf_counter() - t0
    predicted = dict(result.concentrations)
    record.update(presence_confusion(predicted, truth.elements_present, candidates))
    record["predicted_wt"] = {el: 100.0 * float(predicted.get(el, 0.0)) for el in candidates}
    record["temperature_K"] = float(result.temperature_K)
    record["electron_density_cm3"] = float(result.electron_density_cm3)
    record["converged"] = bool(result.converged)
    record["ne_source"] = (
        "stark" if (result.quality_metrics or {}).get("ne_from_stark") else "pressure_balance"
    )
    record["n_observations"] = int(diagnostics.get("n_observations", 0))
    record["stage_timings_s"] = dict(diagnostics.get("stage_timings", {}))

    if truth.composition_wt is not None:
        rmse_wt, signed = composition_errors(predicted, truth.composition_wt)
        record["rmse_wt"] = rmse_wt
        record["signed_errors_wt"] = signed
    return record


# ---------------------------------------------------------------------------
# Dataset + board execution
# ---------------------------------------------------------------------------


def _sample_indices(n: int, max_spectra: Optional[int], seed: int) -> Optional[list[int]]:
    """Seeded, sorted, without-replacement sample (None = run everything)."""
    if max_spectra is None or n <= max_spectra:
        return None
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=max_spectra, replace=False))


def _aggregate_dataset(
    name: str,
    tags: Sequence[str],
    records: list[dict[str, Any]],
    *,
    n_total: int,
    sampled: bool,
    notes: str,
) -> dict[str, Any]:
    """Fold per-spectrum records into the per-dataset scoreboard row."""
    ok_records = [r for r in records if r["status"] == "ok"]
    failures = {r["spectrum_id"]: r["error"] for r in records if r["status"] == "error"}

    n_tp = sum(len(r["tp"]) for r in records)
    n_fp = sum(len(r["fp"]) for r in records)
    n_fn = sum(len(r["fn"]) for r in records)
    precision, recall, f1 = precision_recall_f1(n_tp, n_fp, n_fn)

    rmse_values = [r["rmse_wt"] for r in ok_records if "rmse_wt" in r]
    composition: Optional[dict[str, Any]] = None
    if rmse_values:
        per_element: dict[str, list[float]] = {}
        for r in ok_records:
            for el, err in r.get("signed_errors_wt", {}).items():
                per_element.setdefault(el, []).append(float(err))
        composition = {
            "n_scored": len(rmse_values),
            "rmse_wt_median": _median(rmse_values),
            "rmse_wt_mean": float(np.mean(rmse_values)),
            "per_element_signed_mean_wt": {
                el: float(np.mean(errs)) for el, errs in sorted(per_element.items())
            },
        }

    stage_medians = {
        f"median_{stage}": _median(
            [
                r["stage_timings_s"][stage]
                for r in ok_records
                if stage in r.get("stage_timings_s", {})
            ]
        )
        for stage in ("calibration_s", "detection_id_s", "stark_ne_s", "solve_s", "total_s")
    }

    return {
        "name": name,
        "tags": sorted(tags),
        "status": "ok" if records else "skipped",
        "notes": notes,
        "n_total": n_total,
        "n_run": len(records),
        "n_ok": len(ok_records),
        "n_failed": len(failures),
        "sampled": sampled,
        "id_metrics": {
            "tp": n_tp,
            "fp": n_fp,
            "fn": n_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "composition": composition,
        "runtime": {
            "median_wall_s": _median([r["wall_s"] for r in ok_records]),
            **stage_medians,
        },
        "failures": failures,
        "spectra": records,
    }


def run_scoreboard(
    atomic_db: Any,
    *,
    datasets: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    max_spectra: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    preset: Optional[str] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
    include_holdout: bool = False,
    pipeline_impl: str = "reference",
) -> dict[str, Any]:
    """Run the goal-metric scoreboard over the registered datasets.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Open atomic database used by every pipeline run.
    datasets, tags : Iterable[str] or None
        Registry filters (see :func:`iter_datasets`).
    max_spectra : int or None
        Per-dataset cap. Datasets larger than this are sampled with a seeded
        ``np.random.default_rng(seed)`` without-replacement draw (sorted,
        deterministic; the sampled ids are recorded in the artifact).
    seed : int
        Sampling seed (recorded in the artifact).
    preset : str or None
        Pipeline preset override; ``None`` = the production default
        (``geological``). Exposed for ablation runs only.
    include_holdout : bool
        Run holdout-tier datasets too (deliberate adoption-gate measurement).
        Default False: holdout datasets are skipped (or, when requested by
        name, refused). Vault-tier datasets never run regardless.
    config_overrides : Mapping[str, Any] or None
        Knob overrides routed through the top precedence tier of
        :func:`cflibs.inversion.pipeline.build_pipeline_config` (validated,
        normalized and logged like every other tier). Campaign tooling only;
        recorded in the artifact so every board is attributable to an exact
        config.
    """
    import cflibs

    if pipeline_impl not in PIPELINE_IMPLS:
        raise ValueError(f"Unknown pipeline_impl {pipeline_impl!r}; choose from {PIPELINE_IMPLS}.")

    if not registered_names():  # bare interpreter: load the default board
        ensure_default_datasets()

    board: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cflibs_path": str(Path(cflibs.__file__).resolve().parent),
        "pipeline_impl": pipeline_impl,
        "preset": preset or DEFAULT_PRESET_LABEL,
        "seed": seed,
        "max_spectra": max_spectra,
        "presence_eps_massfrac": PRESENCE_EPS_MASSFRAC,
        "confounder_elements": list(CONFOUNDER_ELEMENTS),
        "candidate_policy": (
            "candidates(spectrum) = truth.elements_present UNION confounder_elements"
        ),
        "config_overrides": dict(config_overrides) if config_overrides else None,
        "include_holdout": include_holdout,
        "datasets": [],
    }

    explicitly_requested = datasets is not None
    for entry in iter_datasets(names=datasets, tags=tags):
        if entry.tier == "vault":
            if explicitly_requested:
                raise ValueError(
                    f"Dataset {entry.name!r} is VAULT tier (end-of-program material, "
                    "design 2.1); the scoreboard never runs it."
                )
            logger.info("scoreboard: skipping %s (vault tier — never run).", entry.name)
            continue
        if entry.tier == "holdout" and not include_holdout:
            if explicitly_requested:
                raise ValueError(
                    f"Dataset {entry.name!r} is HOLDOUT tier (adoption gate); pass "
                    "include_holdout=True / --include-holdout to run it deliberately."
                )
            logger.info(
                "scoreboard: skipping %s (holdout tier; use --include-holdout).", entry.name
            )
            continue
        logger.info("scoreboard: dataset %s ...", entry.name)
        items = list(entry.adapter_factory())
        n_total = len(items)
        indices = _sample_indices(n_total, max_spectra, seed)
        if indices is not None:
            items = [items[i] for i in indices]
        # Registration-time notes win; fall back to the first spectrum's truth
        # notes for adapters whose provenance is only known at run time.
        notes = entry.notes or (items[0][3].notes if items else "")
        records = [
            _score_spectrum(
                atomic_db,
                sid,
                wl,
                inten,
                truth,
                preset=preset,
                config_overrides=config_overrides,
                pipeline_impl=pipeline_impl,
            )
            for sid, wl, inten, truth in items
        ]
        dataset_row = _aggregate_dataset(
            entry.name,
            sorted(entry.tags),
            records,
            n_total=n_total,
            sampled=indices is not None,
            notes=notes,
        )
        dataset_row["tier"] = entry.tier
        board["datasets"].append(dataset_row)
        metrics = dataset_row["id_metrics"]
        logger.info(
            "scoreboard: %s done — n=%d/%d F1=%.3f failures=%d",
            entry.name,
            dataset_row["n_run"],
            n_total,
            metrics["f1"],
            dataset_row["n_failed"],
        )
    return board


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def fmt_metric(value: Optional[float], spec: str = ".3f") -> str:
    """Format an optional metric value for a report ("—" for missing)."""
    return format(value, spec) if value is not None else "—"


def render_markdown(board: Mapping[str, Any]) -> str:
    """Render the scoreboard report as a markdown document."""
    lines: list[str] = []
    lines.append("# CF-LIBS Goal-Metric Scoreboard")
    lines.append("")
    lines.append(f"Generated: {board['generated_utc']}  ")
    lines.append(f"cflibs: `{board['cflibs_path']}`  ")
    lines.append(f"Pipeline: {board.get('pipeline_impl', 'reference')}  ")
    lines.append(f"Preset: {board['preset']}  ")
    lines.append(
        f"Sampling: seed={board['seed']}, max_spectra="
        f"{board['max_spectra'] if board['max_spectra'] is not None else 'unlimited'}"
    )
    lines.append("")
    lines.append("Reproduce:")
    lines.append("")
    lines.append("```bash")
    cmd = "JAX_PLATFORMS=cpu cflibs scoreboard --output-dir output/scoreboard"
    if board["max_spectra"] is not None:
        cmd += f" --max-spectra {board['max_spectra']} --seed {board['seed']}"
    if board.get("include_holdout"):
        cmd += " --include-holdout"
    lines.append(cmd)
    lines.append("```")
    lines.append("")
    lines.append(
        "**Tier policy:** vault-tier datasets never run; holdout-tier datasets "
        "(adoption gate) run only with `--include-holdout`"
        f"{' (this board INCLUDES holdout datasets)' if board.get('include_holdout') else ''}. "
    )
    lines.append("")
    lines.append(
        f"**Candidate-set policy:** {board['candidate_policy']} "
        f"(confounders: {', '.join(board['confounder_elements'])}). "
        f"**Presence rule:** solved mass fraction >= {board['presence_eps_massfrac']} "
        "(0.5 wt%). Failed spectra call nothing present (truth counts as FN) and are "
        "also listed under failures. ID metrics are micro-averaged; RMSE is the "
        "median per-spectrum RMSE over certified elements (element-wt% basis, "
        "oxygen excluded); runtimes are per-dataset medians."
    )
    lines.append("")
    lines.append(
        "| Dataset | Tags | Spectra (run/total) | P | R | F1 | RMSE wt% (med) "
        "| s/spectrum (med) | Failures |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds in board["datasets"]:
        if ds["status"] == "skipped":
            lines.append(
                f"| {ds['name']} | {', '.join(ds['tags'])} | 0/0 (skipped) "
                "| — | — | — | — | — | — |"
            )
            continue
        m = ds["id_metrics"]
        comp = ds["composition"]
        rmse = fmt_metric(comp["rmse_wt_median"], ".2f") if comp else "—"
        lines.append(
            f"| {ds['name']} | {', '.join(ds['tags'])} | {ds['n_run']}/{ds['n_total']}"
            f"{' (sampled)' if ds['sampled'] else ''} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {rmse} | {fmt_metric(ds['runtime']['median_wall_s'], '.1f')} "
            f"| {ds['n_failed']} |"
        )
    lines.append("")

    for ds in board["datasets"]:
        lines.append(f"## {ds['name']}")
        lines.append("")
        if ds["status"] == "skipped":
            lines.append("**Skipped** — adapter yielded no spectra (see log / dataset README).")
            lines.append("")
            continue
        if ds["notes"]:
            lines.append(f"> {ds['notes']}")
            lines.append("")
        m = ds["id_metrics"]
        lines.append(
            f"- ID (micro): TP={m['tp']} FP={m['fp']} FN={m['fn']} — "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        )
        rt = ds["runtime"]
        lines.append(
            f"- Runtime medians (s): total={fmt_metric(rt['median_wall_s'], '.2f')}, "
            f"calibration={fmt_metric(rt['median_calibration_s'], '.2f')}, "
            f"detection+ID={fmt_metric(rt['median_detection_id_s'], '.2f')}, "
            f"stark n_e={fmt_metric(rt['median_stark_ne_s'], '.2f')}, "
            f"solve={fmt_metric(rt['median_solve_s'], '.2f')}"
        )
        if ds["sampled"]:
            lines.append(
                f"- Sampled {ds['n_run']} of {ds['n_total']} spectra "
                f"(seeded rng, seed={board['seed']})."
            )
        if ds["composition"]:
            comp = ds["composition"]
            lines.append(
                f"- Composition (n={comp['n_scored']}): RMSE wt% median="
                f"{fmt_metric(comp['rmse_wt_median'], '.2f')}, mean={fmt_metric(comp['rmse_wt_mean'], '.2f')}"
            )
            lines.append("")
            lines.append("| Element | mean signed error (wt%) |")
            lines.append("|---|---|")
            for el, err in comp["per_element_signed_mean_wt"].items():
                lines.append(f"| {el} | {err:+.2f} |")
        if ds["failures"]:
            lines.append("")
            lines.append("Failures:")
            lines.append("")
            for sid, err in ds["failures"].items():
                lines.append(f"- `{sid}`: {err}")
        lines.append("")
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def write_artifacts(board: Mapping[str, Any], output_dir: Path | str) -> tuple[Path, Path]:
    """Write ``scoreboard.json`` + ``scoreboard.md`` and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scoreboard.json"
    md_path = out / "scoreboard.md"
    json_path.write_text(json.dumps(board, indent=2, default=_json_default))
    md_path.write_text(render_markdown(board))
    logger.info("scoreboard: wrote %s and %s", json_path, md_path)
    return json_path, md_path
