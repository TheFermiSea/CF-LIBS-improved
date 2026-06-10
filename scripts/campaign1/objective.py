"""Campaign 1 candidate evaluation: scoreboard run + composite fitness.

One candidate = one ``config_overrides`` dict (see ``knob_space``). Evaluation
reuses the goal-metric scoreboard internals
(``cflibs.benchmark.scoreboard._score_spectrum`` / ``_aggregate_dataset``)
over the **optimization split only** (``splits``; holdout/vault material is
refused structurally), with seeded per-dataset subsampling for the per-trial
budget.

Composite fitness (design 2.2)::

    score_d  = 0.6*F1_d + 0.4*(1 - min(R_d, 10)/10)   element-wt datasets
               F1_d                                    presence-only datasets
    fitness  = sum(w_d*score_d)/sum(w_d)
               - 0.5*Var_d(score_d)                    cross-matrix overfitting
               - 0.2*max(0, t_med/5 - 1)               runtime beyond 5 s/spectrum
    weights  w_d: real element-wt 1.0, real presence-only 0.7, synthetic 0.5

Death penalties (returned as ``death=True``; the driver reports
``DEATH_FITNESS`` to Optuna instead of ``-inf`` so TPE keeps a usable
ordering):

- any *real* dataset with ``FP_d > FP_d(baseline) + 1`` (precision is the asset),
- any dataset with ``n_failed > 1.25 * baseline failures`` (no trading
  failures for F1).
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

_CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(_CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGN_DIR))

import splits  # noqa: E402

# --- fitness constants (design 2.2) ----------------------------------------
R_CAP_WT = 10.0
T_BUDGET_S = 5.0
F1_WEIGHT = 0.6
RMSE_WEIGHT = 0.4
VARIANCE_PENALTY = 0.5
RUNTIME_PENALTY = 0.2
WEIGHT_REAL_ELEMENT_WT = 1.0
WEIGHT_REAL_PRESENCE_ONLY = 0.7
WEIGHT_SYNTHETIC = 0.5
FP_DEATH_MARGIN = 1
FAILURE_DEATH_FACTOR = 1.25
#: Reported to Optuna instead of -inf (keeps TPE ordering usable).
DEATH_FITNESS = -1.0e9


@dataclass
class EvalContext:
    """Frozen evaluation context shared by every trial of a study."""

    db_path: Path
    manifest: Mapping[str, Any]
    datasets: tuple[str, ...]
    spectra_per_dataset: Optional[int] = None
    sample_seed: int = 20260610
    preset: Optional[str] = None  # None = production default (geological)
    n_procs: int = 1
    per_spectrum_timeout_s: float = 120.0


@dataclass
class FitnessReport:
    """Composite fitness with every component recorded for the trial record."""

    fitness: float
    death: bool
    death_reasons: list[str] = field(default_factory=list)
    weighted_score: float = 0.0
    variance: float = 0.0
    runtime_median_s: Optional[float] = None
    runtime_penalty: float = 0.0
    per_dataset: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset item loading (cached per worker process)
# ---------------------------------------------------------------------------

_ADAPTERS_REGISTERED = False
_ITEMS_CACHE: dict[tuple, list] = {}
_DB_CACHE: dict[str, Any] = {}


def _ensure_adapters() -> None:
    global _ADAPTERS_REGISTERED
    if not _ADAPTERS_REGISTERED:
        from cflibs.benchmark.scoreboard_registry import ensure_default_datasets

        ensure_default_datasets()
        _ADAPTERS_REGISTERED = True


def _get_db(db_path: Path | str):
    key = str(db_path)
    if key not in _DB_CACHE:
        from cflibs.atomic.database import AtomicDatabase

        _DB_CACHE[key] = AtomicDatabase(Path(db_path))
    return _DB_CACHE[key]


def _dataset_entry(name: str):
    from cflibs.benchmark.scoreboard_registry import iter_datasets

    _ensure_adapters()
    return next(iter(iter_datasets(names=[name])))


def _load_split_items(manifest: Mapping[str, Any], section: str, name: str) -> list:
    """All (sid, wl, inten, truth) items of one dataset's split, manifest order."""
    cache_key = ("items", id(manifest), section, name)
    if cache_key in _ITEMS_CACHE:
        return _ITEMS_CACHE[cache_key]
    ids: Sequence[str] = manifest[section][name]
    wanted = set(ids)
    entry = _dataset_entry(name)
    by_id = {item[0]: item for item in entry.adapter_factory() if item[0] in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise KeyError(
            f"{name}: {len(missing)} manifest spectrum ids missing from the adapter "
            f"(e.g. {sorted(missing)[:3]}) — frozen split no longer reproducible."
        )
    items = [by_id[sid] for sid in ids]
    _ITEMS_CACHE[cache_key] = items
    return items


def sample_split_ids(
    manifest: Mapping[str, Any],
    section: str,
    name: str,
    spectra_per_dataset: Optional[int],
    sample_seed: int,
) -> list[str]:
    """Seeded, sorted, per-dataset subsample of a split (frozen across trials)."""
    ids = list(manifest[section][name])
    if spectra_per_dataset is None or len(ids) <= spectra_per_dataset:
        return ids
    rng = np.random.default_rng([sample_seed, zlib.crc32(name.encode())])
    idx = sorted(int(i) for i in rng.choice(len(ids), size=spectra_per_dataset, replace=False))
    return [ids[i] for i in idx]


def _get_eval_items(ctx: EvalContext, section: str, name: str, chosen: Sequence[str]) -> list:
    """The dataset's split items restricted to the already-sampled ``chosen`` ids."""
    items = _load_split_items(ctx.manifest, section, name)
    if len(chosen) == len(items):
        return items
    by_id = {item[0]: item for item in items}
    return [by_id[sid] for sid in chosen]


# ---------------------------------------------------------------------------
# Per-spectrum scoring (with timeout), optional process pool
# ---------------------------------------------------------------------------


class _SpectrumTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover - signal path
    raise _SpectrumTimeout()


def _timeout_record(sid: str, truth: Any, timeout_s: float) -> dict[str, Any]:
    from cflibs.benchmark.scoreboard import CONFOUNDER_ELEMENTS, presence_confusion

    candidates = sorted(set(truth.elements_present) | set(CONFOUNDER_ELEMENTS))
    record: dict[str, Any] = {
        "spectrum_id": sid,
        "truth_elements": sorted(truth.elements_present),
        "candidates": candidates,
        "composition_basis": truth.composition_basis,
        "status": "error",
        "error": f"TimeoutError: per-spectrum timeout ({timeout_s:.0f}s) exceeded",
        "wall_s": timeout_s,
    }
    record.update(presence_confusion({}, truth.elements_present, candidates))
    return record


# Module globals for pool children (set by _pool_init).
_POOL_DB_PATH: Optional[str] = None


def _pool_init(db_path: str) -> None:  # pragma: no cover - exercised via pool
    global _POOL_DB_PATH
    _POOL_DB_PATH = db_path


def _score_one(payload: tuple) -> dict[str, Any]:
    """Score one spectrum (runs in the parent or in a pool child)."""
    sid, wl, inten, truth, preset, config_overrides, timeout_s, db_path = payload
    from cflibs.benchmark.scoreboard import _score_spectrum

    atomic_db = _get_db(_POOL_DB_PATH or db_path)
    use_alarm = timeout_s and hasattr(signal, "SIGALRM")
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return _score_spectrum(
            atomic_db, sid, wl, inten, truth, preset=preset, config_overrides=config_overrides
        )
    except _SpectrumTimeout:  # pragma: no cover - timing-dependent
        return _timeout_record(sid, truth, float(timeout_s))
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


#: Grace added to the in-child SIGALRM before the parent hard-kills (s).
HARD_TIMEOUT_GRACE_S = 30.0

_POOL = None
_POOL_PROCS = 0

# Tear the pool down BEFORE interpreter shutdown: a garbage-collected
# mp.Pool raises spurious AttributeErrors from Pool.__del__ at exit.
import atexit  # noqa: E402

atexit.register(lambda: _terminate_pool())


def _get_pool(n_procs: int, db_path: Path | str):
    global _POOL, _POOL_PROCS
    if _POOL is None or _POOL_PROCS != n_procs:
        _terminate_pool()
        _POOL = mp.get_context("spawn").Pool(
            processes=n_procs, initializer=_pool_init, initargs=(str(db_path),)
        )
        _POOL_PROCS = n_procs
    return _POOL


def _terminate_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL.join()
        _POOL = None


def _run_payloads(payloads: list[tuple], ctx: "EvalContext") -> list[dict[str, Any]]:
    """Score payloads with a HARD per-spectrum timeout.

    Two timeout layers (the smoke run caught the gap): the in-child SIGALRM
    interrupts Python-level overruns gracefully at ``per_spectrum_timeout_s``,
    but it cannot interrupt long GIL-released C/XLA calls (measured:
    ``use_deconvolution=True`` wedged a worker inside
    ``backend_compile_and_load`` indefinitely). So spectra always run in a
    spawn pool and the parent enforces wall deadlines via
    ``AsyncResult.get(timeout)``; a hard overrun terminates the pool (the only
    way to kill stuck C code), records a timeout failure, and resubmits the
    untouched remainder to a fresh pool.
    """
    if not payloads:
        return []
    hard = float(ctx.per_spectrum_timeout_s or 0.0)
    n_procs = max(int(ctx.n_procs), 1)
    if hard <= 0.0:  # no timeout: plain pool map
        return _get_pool(n_procs, ctx.db_path).map(_score_one, payloads)

    results: dict[int, dict[str, Any]] = {}
    pending = list(enumerate(payloads))
    while pending:
        pool = _get_pool(n_procs, ctx.db_path)
        batch_t0 = time.monotonic()
        asyncs = [(i, pool.apply_async(_score_one, (payload,))) for i, payload in pending]
        timed_out = False
        for position, (i, handle) in enumerate(asyncs):
            # Tasks queue n_procs-wide behind each other: the k-th task's
            # worst-case start is after floor(k/n_procs) full timeouts.
            deadline = batch_t0 + (position // n_procs + 1) * (hard + HARD_TIMEOUT_GRACE_S)
            try:
                results[i] = handle.get(timeout=max(deadline - time.monotonic(), 1.0))
            except mp.TimeoutError:
                sid, _wl, _inten, truth = payloads[i][:4]
                results[i] = _timeout_record(sid, truth, hard)
                results[i]["error"] += " (hard kill: stuck in C/XLA, pool terminated)"
                _terminate_pool()
                timed_out = True
                break
            except Exception as exc:  # child crashed (e.g. OOM-kill)
                sid, _wl, _inten, truth = payloads[i][:4]
                results[i] = _timeout_record(sid, truth, hard)
                results[i]["error"] = f"{type(exc).__name__}: pool child crashed: {exc}"
                _terminate_pool()
                timed_out = True
                break
        pending = [(i, payload) for i, payload in pending if i not in results]
        if not timed_out and pending:  # pragma: no cover - defensive
            raise RuntimeError("pool returned without completing all payloads")
    return [results[i] for i in range(len(payloads))]


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_overrides(
    config_overrides: Mapping[str, Any],
    ctx: EvalContext,
    *,
    section: str = "optimization",
    datasets: Optional[Iterable[str]] = None,
    allow_restricted: bool = False,
) -> dict[str, Any]:
    """Run the scoreboard for one candidate config on one split section.

    ``section='optimization'`` (the default) enforces the structural holdout
    guard: holdout-tagged datasets and holdout spectrum ids are refused.
    ``section='holdout'`` additionally requires ``allow_restricted=True`` —
    only ``holdout_eval.py`` passes it, and every such run is ledger-logged.
    Vault datasets are refused unconditionally.
    """
    names = tuple(datasets) if datasets is not None else ctx.datasets
    splits.assert_not_vault(names)
    if section == "optimization":
        # Dataset-level refusal FIRST (a holdout-only dataset has no
        # optimization split to even sample from), then id-level below.
        splits.assert_optimization_only(ctx.manifest, names)
    elif section == "holdout":
        if not allow_restricted:
            raise splits.HoldoutViolation(
                "Holdout evaluation requires allow_restricted=True (holdout_eval.py "
                "only; every holdout query is ledger-logged)."
            )
    else:
        raise ValueError(f"Unknown split section {section!r}")
    # Sample ONCE per dataset; the guard and the evaluation see the same ids.
    spectrum_ids = {
        name: sample_split_ids(
            ctx.manifest, section, name, ctx.spectra_per_dataset, ctx.sample_seed
        )
        for name in names
    }
    if section == "optimization":
        splits.assert_optimization_only(ctx.manifest, names, spectrum_ids)

    from cflibs.benchmark.scoreboard import _aggregate_dataset

    rows = []
    for name in names:
        entry = _dataset_entry(name)
        items = _get_eval_items(ctx, section, name, spectrum_ids[name])
        payloads = [
            (
                sid,
                wl,
                inten,
                truth,
                ctx.preset,
                dict(config_overrides),
                ctx.per_spectrum_timeout_s,
                str(ctx.db_path),
            )
            for sid, wl, inten, truth in items
        ]
        records = _run_payloads(payloads, ctx)
        rows.append(
            _aggregate_dataset(
                name,
                sorted(entry.tags),
                records,
                n_total=len(ctx.manifest[section][name]),
                sampled=len(items) < len(ctx.manifest[section][name]),
                notes=entry.notes or (items[0][3].notes if items else ""),
            )
        )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "section": section,
        "preset": ctx.preset or "geological (production default)",
        "sample_seed": ctx.sample_seed,
        "spectra_per_dataset": ctx.spectra_per_dataset,
        "config_overrides": dict(config_overrides),
        "datasets": rows,
    }


# ---------------------------------------------------------------------------
# Fitness math (pure functions; unit-tested on rigged metrics)
# ---------------------------------------------------------------------------


def dataset_kind(row: Mapping[str, Any]) -> str:
    """Classify a dataset row: synthetic / real_element_wt / real_presence_only."""
    if "synthetic" in row.get("tags", []):
        return "synthetic"
    basis_element_wt = any(
        r.get("composition_basis") == "element_wt" for r in row.get("spectra", [])
    )
    return "real_element_wt" if basis_element_wt else "real_presence_only"


def dataset_weight(row: Mapping[str, Any]) -> float:
    kind = dataset_kind(row)
    if kind == "synthetic":
        return WEIGHT_SYNTHETIC
    if kind == "real_element_wt":
        return WEIGHT_REAL_ELEMENT_WT
    return WEIGHT_REAL_PRESENCE_ONLY


def dataset_score(row: Mapping[str, Any]) -> float:
    """score_d per design 2.2 (missing RMSE on an element-wt dataset = worst case)."""
    f1 = float(row["id_metrics"]["f1"])
    kind = dataset_kind(row)
    if kind == "real_presence_only":
        return f1
    composition = row.get("composition")
    rmse = (
        float(composition["rmse_wt_median"])
        if composition and composition.get("rmse_wt_median") is not None
        else R_CAP_WT
    )
    return F1_WEIGHT * f1 + RMSE_WEIGHT * (1.0 - min(rmse, R_CAP_WT) / R_CAP_WT)


def pooled_runtime_median(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    walls = [
        float(r["wall_s"])
        for row in rows
        for r in row.get("spectra", [])
        if r.get("status") == "ok" and "wall_s" in r
    ]
    return float(np.median(walls)) if walls else None


def baseline_reference_from_board(board: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-dataset death-penalty reference: FP and failure counts at baseline."""
    return {
        row["name"]: {
            "fp": int(row["id_metrics"]["fp"]),
            "n_failed": int(row["n_failed"]),
            "n_run": int(row["n_run"]),
            "f1": float(row["id_metrics"]["f1"]),
        }
        for row in board["datasets"]
    }


def compute_fitness(
    rows: Sequence[Mapping[str, Any]],
    baseline_ref: Mapping[str, Mapping[str, Any]],
) -> FitnessReport:
    """Composite fitness + death penalties (design 2.2)."""
    death_reasons: list[str] = []
    scores: list[float] = []
    weights: list[float] = []
    per_dataset: dict[str, dict[str, Any]] = {}

    for row in rows:
        name = row["name"]
        if name not in baseline_ref:
            raise KeyError(f"Baseline reference missing dataset {name!r}")
        base = baseline_ref[name]
        fp = int(row["id_metrics"]["fp"])
        n_failed = int(row["n_failed"])
        kind = dataset_kind(row)
        if kind != "synthetic" and fp > int(base["fp"]) + FP_DEATH_MARGIN:
            death_reasons.append(
                f"{name}: FP {fp} > baseline {base['fp']} + {FP_DEATH_MARGIN} "
                "(precision is the asset)"
            )
        if n_failed > FAILURE_DEATH_FACTOR * float(base["n_failed"]):
            death_reasons.append(
                f"{name}: n_failed {n_failed} > {FAILURE_DEATH_FACTOR} * baseline "
                f"{base['n_failed']} (no trading failures for F1)"
            )
        score = dataset_score(row)
        weight = dataset_weight(row)
        scores.append(score)
        weights.append(weight)
        per_dataset[name] = {
            "kind": kind,
            "score": score,
            "weight": weight,
            "f1": float(row["id_metrics"]["f1"]),
            "precision": float(row["id_metrics"]["precision"]),
            "recall": float(row["id_metrics"]["recall"]),
            "fp": fp,
            "n_failed": n_failed,
            "rmse_wt_median": (row.get("composition") or {}).get("rmse_wt_median"),
        }

    t_med = pooled_runtime_median(rows)
    runtime_penalty = (
        RUNTIME_PENALTY * max(0.0, t_med / T_BUDGET_S - 1.0) if t_med is not None else 0.0
    )
    weighted = float(np.average(scores, weights=weights)) if scores else 0.0
    variance = float(np.var(scores)) if scores else 0.0
    fitness = weighted - VARIANCE_PENALTY * variance - runtime_penalty
    death = bool(death_reasons)
    return FitnessReport(
        fitness=DEATH_FITNESS if death else fitness,
        death=death,
        death_reasons=death_reasons,
        weighted_score=weighted,
        variance=variance,
        runtime_median_s=t_med,
        runtime_penalty=runtime_penalty,
        per_dataset=per_dataset,
    )


def evaluate_candidate(
    config_overrides: Mapping[str, Any],
    ctx: EvalContext,
    baseline_ref: Mapping[str, Mapping[str, Any]],
) -> tuple[FitnessReport, dict[str, Any]]:
    """Objective body: optimization-split scoreboard run -> composite fitness."""
    board = evaluate_overrides(config_overrides, ctx, section="optimization")
    report = compute_fitness(board["datasets"], baseline_ref)
    return report, board


# ---------------------------------------------------------------------------
# Paired per-spectrum bootstrap (design 2.4; used by holdout_eval)
# ---------------------------------------------------------------------------


def _confusion_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, tuple[int, ...]]]:
    out: dict[str, dict[str, tuple[int, ...]]] = {}
    for row in rows:
        out[row["name"]] = {
            r["spectrum_id"]: (len(r["tp"]), len(r["fp"]), len(r["fn"]))
            for r in row.get("spectra", [])
        }
    return out


def paired_bootstrap_delta_f1(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 20260610,
) -> dict[str, Any]:
    """Paired per-spectrum bootstrap of pooled Delta(micro-F1) (design 2.4).

    Spectra are resampled with replacement *within each dataset*; the same
    resample indices apply to candidate and baseline (paired). Returns the
    point delta and the 95% percentile CI.

    Micro-F1 comes from the scoreboard's :func:`precision_recall_f1` (the
    single source for this metric math). The summed confusion counts are
    integral by construction, so the int conversion inside it is exact.
    """
    from cflibs.benchmark.scoreboard import precision_recall_f1

    cand = _confusion_by_id(candidate_rows)
    base = _confusion_by_id(baseline_rows)
    pairs: list[np.ndarray] = []  # per dataset: (n_spectra, 6) [c_tp c_fp c_fn b_tp b_fp b_fn]
    for name in sorted(set(cand) & set(base)):
        shared = sorted(set(cand[name]) & set(base[name]))
        if set(cand[name]) != set(base[name]):
            raise ValueError(
                f"{name}: candidate and baseline ran different spectrum sets "
                "(paired bootstrap requires identical ids)."
            )
        if shared:
            pairs.append(
                np.array([[*cand[name][sid], *base[name][sid]] for sid in shared], dtype=float)
            )
    if not pairs:
        raise ValueError("No paired spectra for bootstrap")

    def pooled_delta(arrays: list[np.ndarray]) -> float:
        total = np.sum(np.vstack(arrays), axis=0)
        return precision_recall_f1(*total[:3])[2] - precision_recall_f1(*total[3:])[2]

    point = pooled_delta(pairs)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        resampled = [arr[rng.integers(0, len(arr), size=len(arr))] for arr in pairs]
        deltas[b] = pooled_delta(resampled)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_f1": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "n_spectra": int(sum(len(a) for a in pairs)),
        "seed": seed,
    }
