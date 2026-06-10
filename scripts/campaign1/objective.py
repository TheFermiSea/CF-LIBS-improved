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

Death penalties — fitness version 1 (returned as ``death=True``; the driver
reports ``DEATH_FITNESS`` to Optuna instead of ``-inf`` so TPE keeps a usable
ordering):

- any *real* dataset with ``FP_d > FP_d(baseline) + 1`` (precision is the asset),
- any dataset with ``n_failed > 1.25 * baseline failures`` (no trading
  failures for F1).

Graded penalties — fitness version 2 (FITNESS-V2)::

    fitness  = composite                       (the v1 composite above)
               - LAMBDA_FP   * sum_d excess_FP_d
               - LAMBDA_FAIL * sum_d excess_failed_d
    excess_FP_d     = max(0, FP_d - (base_FP_d + 1))      real datasets only
    excess_failed_d = max(0, n_failed_d - ceil(1.25 * base_failed_d))

Rationale (measured on the live run1 study): 79 of 80 trials hit the flat
v1 ``-1e9`` death penalty — a flat landscape giving TPE zero ranking signal,
even though several killed trials had ``weighted_score`` above the surviving
baseline (trial 1: 0.352 vs 0.3293). v2 keeps the SEARCH graded while the
hard no-regression constraint stays at ADOPTION (``holdout_eval`` G-gates).
A trial with zero excess counts scores identically under v1 and v2, so run1
and run2 numbers stay comparable for clean candidates.

Catastrophic floor: when the total graded penalty exceeds
``CATASTROPHIC_PENALTY`` (1.0 — e.g. >20 excess FPs; run1 saw FP 87 and
n_failed 73/74 trials) the fitness is the flat ``CATASTROPHIC_FITNESS``
(-1e3): far below every real composite score, far above the v1 ``-1e9`` so
the journal distinguishes v2 catastrophics from v1 deaths. Structural
failures (evaluation crash, ledger refusal, holdout violation) raise and are
recorded as Optuna FAIL states — they never get a graded fitness.
"""

from __future__ import annotations

import math
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

# --- fitness constants (FITNESS-V2: graded penalties) -----------------------
#: Penalty per excess FP beyond baseline+1 on a *real* dataset. Sized so ONE
#: excess FP costs more than any plausible single-step weighted_score gain:
#: the observed run1 weighted_score range was ~0.30-0.56, i.e. the WHOLE
#: spread over 80 trials was ~0.26 and single-knob improvements are far
#: smaller than 0.05 — a candidate can never buy one extra FP with a score
#: gain. Tuneable via the study config (recorded in the frozen manifest).
LAMBDA_FP = 0.05
#: Penalty per failure beyond ceil(1.25 x baseline failures), any dataset.
#: Same sizing argument at lower severity (a failure is one lost answer; an
#: FP is a wrong answer delivered with confidence).
LAMBDA_FAIL = 0.02
#: Above this total graded penalty the trial is catastrophic (flat floor):
#: >20 excess FPs or >50 excess failures (or a mix) — run1's genuine
#: disasters (FP 87; n_failed 73/74 on aalto) land far beyond it.
CATASTROPHIC_PENALTY = 1.0
#: Flat fitness for catastrophic trials: far below all real composite scores
#: (~[-1, 1]) so TPE deprioritizes the whole region, far above DEATH_FITNESS
#: (-1e9) so the journal distinguishes v2 catastrophics from v1 deaths.
CATASTROPHIC_FITNESS = -1.0e3
#: Fitness versions this module implements (1 = flat death penalties for
#: run1 reproducibility; 2 = graded penalties + catastrophic floor).
SUPPORTED_FITNESS_VERSIONS = (1, 2)


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
    #: Dataset at which the evaluation short-circuited on a death penalty
    #: (eff#1; its row may be partial). None = full evaluation.
    aborted_after_dataset: Optional[str] = None
    #: Which fitness math produced ``fitness`` (1 = flat death penalties,
    #: 2 = graded penalties + catastrophic floor).
    fitness_version: int = 1
    #: v2: LAMBDA_FP * sum_d excess_fp_d + LAMBDA_FAIL * sum_d excess_failed_d
    #: (always 0.0 under v1; the per-dataset breakdown is in ``per_dataset``).
    graded_total_penalty: float = 0.0
    #: v2: graded_total_penalty > CATASTROPHIC_PENALTY -> flat
    #: CATASTROPHIC_FITNESS (never True under v1).
    catastrophic: bool = False

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
    from cflibs.benchmark.scoreboard import CONFOUNDER_ELEMENTS, failure_record

    candidates = sorted(set(truth.elements_present) | set(CONFOUNDER_ELEMENTS))
    return failure_record(
        sid,
        truth,
        candidates,
        f"TimeoutError: per-spectrum timeout ({timeout_s:.0f}s) exceeded",
        timeout_s,
    )


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


def _harvest_finished(asyncs: list[tuple[int, Any]], results: dict[int, dict[str, Any]]) -> None:
    """Sweep finished-but-uncollected pool results before a pool kill (eff#2).

    When one payload hard-times-out, the other pool workers have usually
    finished several queued payloads already — work that is paid for. Collect
    every ``ready()`` handle non-blockingly so only truly-unfinished payloads
    are resubmitted to the fresh pool. A handle whose child *raised* is left
    pending (the resubmit decides its fate exactly as before).
    """
    for j, handle in asyncs:
        if j in results or not handle.ready():
            continue
        try:
            results[j] = handle.get(timeout=0)
        except Exception:  # noqa: BLE001 — crashed child: let the resubmit retry it
            pass


def _iter_payload_records(payloads: list[tuple], ctx: "EvalContext"):
    """Stream per-spectrum records for ``payloads`` in submission order.

    The whole batch (typically every dataset of a trial, merged — eff#3) is
    submitted to the pool up front; records are yielded strictly in payload
    order as they complete, so a consumer can slice them back per dataset and
    react at dataset boundaries (the eff#1 early-abort hook). A consumer that
    stops iterating early MUST call :func:`_terminate_pool` to actually kill
    the queued/in-flight remainder.

    Two timeout layers (the smoke run caught the gap): the in-child SIGALRM
    interrupts Python-level overruns gracefully at ``per_spectrum_timeout_s``,
    but it cannot interrupt long GIL-released C/XLA calls (measured: a
    ``use_deconvolution=True`` draw wedged a worker inside
    ``backend_compile_and_load`` indefinitely). So spectra always run in a
    spawn pool and the parent enforces wall deadlines via
    ``AsyncResult.get(timeout)``; a hard overrun harvests every finished
    result, terminates the pool (the only way to kill stuck C code), records
    a timeout failure, and resubmits the truly-unfinished remainder to a
    fresh pool.
    """
    hard = float(ctx.per_spectrum_timeout_s or 0.0)
    n_procs = max(int(ctx.n_procs), 1)
    if hard <= 0.0:  # no timeout: plain order-preserving pool stream
        yield from _get_pool(n_procs, ctx.db_path).imap(_score_one, payloads)
        return

    results: dict[int, dict[str, Any]] = {}
    next_emit = 0
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
                _harvest_finished(asyncs[position + 1 :], results)
                _terminate_pool()
                timed_out = True
            except Exception as exc:  # child crashed (e.g. OOM-kill)
                sid, _wl, _inten, truth = payloads[i][:4]
                results[i] = _timeout_record(sid, truth, hard)
                results[i]["error"] = f"{type(exc).__name__}: pool child crashed: {exc}"
                _harvest_finished(asyncs[position + 1 :], results)
                _terminate_pool()
                timed_out = True
            # Emit everything contiguous from the front (harvested results
            # beyond a still-pending gap wait in ``results`` until the gap
            # is filled by the resubmit round).
            while next_emit in results:
                yield results[next_emit]
                next_emit += 1
            if timed_out:
                break
        pending = [(i, payload) for i, payload in pending if i not in results]
        if not timed_out and pending:  # pragma: no cover - defensive
            raise RuntimeError("pool returned without completing all payloads")


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def _consume_dataset_records(per_dataset, record_stream, tracker, ctx, section, aggregate):
    """Drain the merged record stream per dataset, honoring the early abort.

    Returns ``(rows, aborted_at)``. Cheap per-record check (eff#1):
    counts/penalties are monotone in the accumulated records, so crossing the
    fitness version's threshold here means the full run would cross it too —
    the fitness is identical either way (see :class:`EarlyAbortTracker`).
    """
    rows: list = []
    aborted_at: Optional[str] = None
    for name, entry, items in per_dataset:
        if tracker is not None:
            tracker.start_dataset(name, "synthetic" in entry.tags)
        records: list[dict[str, Any]] = []
        dead = False
        for _ in items:
            record = next(record_stream)
            records.append(record)
            if tracker is not None and tracker.add_record(record):
                dead = True
                break
        rows.append(
            aggregate(
                name,
                sorted(entry.tags),
                records,
                n_total=len(ctx.manifest[section][name]),
                sampled=len(items) < len(ctx.manifest[section][name]),
                notes=entry.notes or (items[0][3].notes if items else ""),
            )
        )
        if dead:
            aborted_at = name
            _terminate_pool()  # kill the queued/in-flight remainder of the batch
            break
        if tracker is not None:
            tracker.finish_dataset()
    return rows, aborted_at


def _assert_section_access(
    section: str,
    names: tuple,
    ctx: "EvalContext",
    allow_restricted: bool,
) -> None:
    """Structural split-tier refusal for one evaluation request."""
    splits.assert_not_vault(names)
    if section == "optimization":
        # Dataset-level refusal FIRST (a holdout-only dataset has no
        # optimization split to even sample from), then id-level in the caller.
        splits.assert_optimization_only(ctx.manifest, names)
    elif section == "holdout":
        if not allow_restricted:
            raise splits.HoldoutViolation(
                "Holdout evaluation requires allow_restricted=True (holdout_eval.py "
                "only; every holdout query is ledger-logged)."
            )
    else:
        raise ValueError(f"Unknown split section {section!r}")


def evaluate_overrides(
    config_overrides: Mapping[str, Any],
    ctx: EvalContext,
    *,
    section: str = "optimization",
    datasets: Optional[Iterable[str]] = None,
    allow_restricted: bool = False,
    death_penalty_ref: Optional[Mapping[str, Mapping[str, Any]]] = None,
    fitness_version: int = 1,
) -> dict[str, Any]:
    """Run the scoreboard for one candidate config on one split section.

    ``section='optimization'`` (the default) enforces the structural holdout
    guard: holdout-tagged datasets and holdout spectrum ids are refused.
    ``section='holdout'`` additionally requires ``allow_restricted=True`` —
    only ``holdout_eval.py`` passes it, and every such run is ledger-logged.
    Vault datasets are refused unconditionally.

    ``death_penalty_ref`` (the per-dataset baseline FP/failure reference;
    objective trials only) arms the eff#1 early abort, version-aware via
    ``fitness_version``: v1 aborts the moment one dataset's running counts
    cross a death threshold (fitness pinned at ``DEATH_FITNESS``); v2 aborts
    the moment the ACCUMULATED graded penalty exceeds
    ``CATASTROPHIC_PENALTY`` (fitness pinned at ``CATASTROPHIC_FITNESS``).
    Either way the evaluation short-circuits, the pool is terminated
    (killing the queued remainder), the board records
    ``aborted_after_dataset``, and the returned fitness is identical to the
    full run by construction (determinism proof: :class:`EarlyAbortTracker`).
    """
    names = tuple(datasets) if datasets is not None else ctx.datasets
    _assert_section_access(section, names, ctx, allow_restricted)
    # Sample ONCE per dataset; the guard and the evaluation see the same ids.
    spectrum_ids = {
        name: sample_split_ids(
            ctx.manifest, section, name, ctx.spectra_per_dataset, ctx.sample_seed
        )
        for name in names
    }
    if section == "optimization":
        splits.assert_optimization_only(ctx.manifest, names, spectrum_ids)

    from cflibs.benchmark.scoreboard import DEFAULT_PRESET_LABEL, _aggregate_dataset

    # ONE merged pool batch for every dataset (eff#3): a single submission
    # keeps all n_procs workers busy across dataset boundaries. Records come
    # back in submission order, so each dataset's records are a contiguous
    # slice of the stream.
    per_dataset: list[tuple[str, Any, list]] = []
    payloads: list[tuple] = []
    for name in names:
        entry = _dataset_entry(name)
        items = _get_eval_items(ctx, section, name, spectrum_ids[name])
        per_dataset.append((name, entry, items))
        payloads.extend(
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
        )

    record_stream = _iter_payload_records(payloads, ctx)
    tracker = (
        EarlyAbortTracker(death_penalty_ref, fitness_version)
        if death_penalty_ref is not None
        else None
    )
    rows, aborted_at = _consume_dataset_records(
        per_dataset, record_stream, tracker, ctx, section, _aggregate_dataset
    )
    board: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "section": section,
        "preset": ctx.preset or DEFAULT_PRESET_LABEL,
        "sample_seed": ctx.sample_seed,
        "spectra_per_dataset": ctx.spectra_per_dataset,
        "config_overrides": dict(config_overrides),
        "datasets": rows,
    }
    if aborted_at is not None:
        board["aborted_after_dataset"] = aborted_at
    return board


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


def death_reasons_for_counts(
    name: str,
    synthetic: bool,
    fp: int,
    n_failed: int,
    base: Mapping[str, Any],
) -> list[str]:
    """Death-penalty reasons for one dataset's (possibly partial) counts.

    THE single source for the death thresholds: used by
    :func:`compute_fitness` on full dataset rows AND by the eff#1 early-abort
    check on running counts. Both ``fp`` and ``n_failed`` are monotone
    non-decreasing in the accumulated per-spectrum records (a record can add
    FP entries or a failure, never remove one), and ``compute_fitness`` ORs
    death over datasets — so a non-empty result on a *partial* count already
    pins the trial's fitness at ``DEATH_FITNESS`` regardless of every
    remaining spectrum and dataset. Aborting there returns a bit-identical
    fitness by construction.
    """
    reasons: list[str] = []
    if not synthetic and fp > int(base["fp"]) + FP_DEATH_MARGIN:
        reasons.append(
            f"{name}: FP {fp} > baseline {base['fp']} + {FP_DEATH_MARGIN} "
            "(precision is the asset)"
        )
    if n_failed > FAILURE_DEATH_FACTOR * float(base["n_failed"]):
        reasons.append(
            f"{name}: n_failed {n_failed} > {FAILURE_DEATH_FACTOR} * baseline "
            f"{base['n_failed']} (no trading failures for F1)"
        )
    return reasons


def graded_excess_counts(
    synthetic: bool,
    fp: int,
    n_failed: int,
    base: Mapping[str, Any],
) -> tuple[int, int]:
    """v2 ``(excess_fp, excess_failed)`` for one dataset's (possibly partial) counts.

    THE single source for the graded thresholds: used by
    :func:`compute_fitness` (v2) on full dataset rows, by the eff#1
    early-abort tracker on running counts, and by :func:`regrade_report_v2`
    on recorded v1 reports. Mirrors the v1 death conditions
    (:func:`death_reasons_for_counts`): the FP threshold applies to real
    datasets only, and the failure allowance is ``ceil(1.25 x baseline)``
    (the integer-count form of the v1 ``> 1.25 x baseline`` condition).

    Both excesses are monotone non-decreasing in ``fp`` / ``n_failed``, which
    are themselves monotone non-decreasing in the accumulated per-spectrum
    records — the property the early-abort determinism proof rests on
    (see :class:`EarlyAbortTracker`).
    """
    excess_fp = 0 if synthetic else max(0, fp - (int(base["fp"]) + FP_DEATH_MARGIN))
    allowed_failed = math.ceil(FAILURE_DEATH_FACTOR * float(base["n_failed"]))
    excess_failed = max(0, n_failed - allowed_failed)
    return excess_fp, excess_failed


def graded_penalty(excess_fp: int, excess_failed: int) -> float:
    """v2 penalty contribution of one dataset's excess counts."""
    return LAMBDA_FP * excess_fp + LAMBDA_FAIL * excess_failed


class EarlyAbortTracker:
    """Per-record decision: is this trial's fitness already pinned? (eff#1)

    Drives the early abort in :func:`evaluate_overrides` for both fitness
    versions. DETERMINISM (abort fitness == full-run fitness, for every
    aborted trial):

    - Per-spectrum records only ever ADD false positives or failures, so a
      dataset's running ``(fp, n_failed)`` counts are monotone non-decreasing
      in the accumulated records.
    - **v1**: a death threshold crossed on partial counts
      (:func:`death_reasons_for_counts`) is therefore also crossed on the
      full counts, and ``compute_fitness`` ORs death over datasets — the full
      run's fitness would be the same flat ``DEATH_FITNESS`` the abort
      assigns.
    - **v2**: :func:`graded_excess_counts` is monotone in the counts, every
      dataset's penalty is >= 0, and the total penalty is their sum — so the
      accumulated penalty (completed datasets + the current dataset's
      partial counts) is a LOWER BOUND on the full-run total penalty. The
      abort fires only when that bound already exceeds
      ``CATASTROPHIC_PENALTY``, hence the full run's total would exceed it
      too and its fitness would be the same flat ``CATASTROPHIC_FITNESS``
      the abort assigns. (Below the threshold no abort fires and the exact
      graded fitness is computed from the complete run.)

    In both versions ``compute_fitness`` on the PARTIAL board reproduces the
    pinned fitness: the aborted dataset's aggregated row carries exactly the
    counts that crossed the threshold.
    """

    def __init__(self, baseline_ref: Mapping[str, Mapping[str, Any]], fitness_version: int = 1):
        if fitness_version not in SUPPORTED_FITNESS_VERSIONS:
            raise ValueError(f"Unsupported fitness_version {fitness_version!r}")
        self._ref = baseline_ref
        self._version = int(fitness_version)
        self._completed_penalty = 0.0  # v2: sum over finished datasets
        self._name: Optional[str] = None
        self._synthetic = False
        self._base: Mapping[str, Any] = {}
        self._fp = 0
        self._failed = 0

    def start_dataset(self, name: str, synthetic: bool) -> None:
        if name not in self._ref:
            # Same contract (and message) as compute_fitness, just earlier.
            raise KeyError(f"Baseline reference missing dataset {name!r}")
        self._name, self._synthetic, self._base = name, synthetic, self._ref[name]
        self._fp = self._failed = 0

    def _dataset_penalty(self) -> float:
        assert self._name is not None
        return graded_penalty(
            *graded_excess_counts(self._synthetic, self._fp, self._failed, self._base)
        )

    def add_record(self, record: Mapping[str, Any]) -> bool:
        """Fold one per-spectrum record in; True = abort now (fitness pinned)."""
        assert self._name is not None, "start_dataset() before add_record()"
        self._fp += len(record["fp"])
        self._failed += record["status"] == "error"
        if self._version == 1:
            return bool(
                death_reasons_for_counts(
                    self._name, self._synthetic, self._fp, self._failed, self._base
                )
            )
        return self._completed_penalty + self._dataset_penalty() > CATASTROPHIC_PENALTY

    def finish_dataset(self) -> None:
        """Fold the finished dataset's penalty into the running v2 total."""
        if self._name is not None:
            self._completed_penalty += self._dataset_penalty()
        self._name = None


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
    *,
    fitness_version: int = 1,
) -> FitnessReport:
    """Composite fitness (design 2.2) + penalties.

    ``fitness_version=1``: flat death penalties (``DEATH_FITNESS`` when any
    dataset crosses a death threshold) — the run1 math, kept selectable for
    artifact reproducibility.

    ``fitness_version=2`` (FITNESS-V2): the SAME composite minus graded
    penalties (``LAMBDA_FP``/``LAMBDA_FAIL`` per excess count), with the flat
    ``CATASTROPHIC_FITNESS`` floor once the total penalty exceeds
    ``CATASTROPHIC_PENALTY``. A trial with zero excess counts scores
    identically under both versions.
    """
    if fitness_version not in SUPPORTED_FITNESS_VERSIONS:
        raise ValueError(f"Unsupported fitness_version {fitness_version!r}")
    death_reasons: list[str] = []
    scores: list[float] = []
    weights: list[float] = []
    per_dataset: dict[str, dict[str, Any]] = {}
    graded_total = 0.0

    for row in rows:
        name = row["name"]
        if name not in baseline_ref:
            raise KeyError(f"Baseline reference missing dataset {name!r}")
        base = baseline_ref[name]
        fp = int(row["id_metrics"]["fp"])
        n_failed = int(row["n_failed"])
        kind = dataset_kind(row)
        synthetic = kind == "synthetic"
        death_reasons.extend(death_reasons_for_counts(name, synthetic, fp, n_failed, base))
        excess_fp, excess_failed = graded_excess_counts(synthetic, fp, n_failed, base)
        penalty_fp, penalty_failed = LAMBDA_FP * excess_fp, LAMBDA_FAIL * excess_failed
        graded_total += penalty_fp + penalty_failed
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
            # v2 grading breakdown (recorded under v1 too — penalties only
            # enter the fitness when fitness_version == 2).
            "excess_fp": excess_fp,
            "excess_failed": excess_failed,
            "penalty_fp": penalty_fp,
            "penalty_failed": penalty_failed,
        }

    t_med = pooled_runtime_median(rows)
    runtime_penalty = (
        RUNTIME_PENALTY * max(0.0, t_med / T_BUDGET_S - 1.0) if t_med is not None else 0.0
    )
    weighted = float(np.average(scores, weights=weights)) if scores else 0.0
    variance = float(np.var(scores)) if scores else 0.0
    composite = weighted - VARIANCE_PENALTY * variance - runtime_penalty

    if fitness_version == 1:
        death = bool(death_reasons)
        fitness = DEATH_FITNESS if death else composite
        catastrophic = False
        graded_applied = 0.0
    else:
        # v2: metrics never produce a flat death — death stays False and
        # DEATH_FITNESS is reserved for structural failures (which raise and
        # land as Optuna FAIL states, never here). death_reasons are still
        # recorded as diagnostics of which v1 thresholds were crossed.
        death = False
        catastrophic = graded_total > CATASTROPHIC_PENALTY
        fitness = CATASTROPHIC_FITNESS if catastrophic else composite - graded_total
        graded_applied = graded_total
    return FitnessReport(
        fitness=fitness,
        death=death,
        death_reasons=death_reasons,
        weighted_score=weighted,
        variance=variance,
        runtime_median_s=t_med,
        runtime_penalty=runtime_penalty,
        per_dataset=per_dataset,
        fitness_version=fitness_version,
        graded_total_penalty=graded_applied,
        catastrophic=catastrophic,
    )


def regrade_report_v2(
    report: Mapping[str, Any],
    baseline_ref: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute fitness-v2 from a RECORDED fitness_report (no re-evaluation).

    The offline path behind ``rescore.py``: a journal's COMPLETE trials carry
    their v1 ``fitness_report`` in ``user_attrs`` — its ``per_dataset``
    fp/n_failed/kind counts plus the recorded ``weighted_score`` /
    ``variance`` / ``runtime_penalty`` are exactly the inputs the v2 math
    needs, so run1 can be re-ranked without re-running a single spectrum.

    Caveat (recorded as ``partial=True``): a trial whose v1 evaluation
    early-aborted (``aborted_after_dataset`` set) only recorded the datasets
    up to the abort. Its graded penalty is then a LOWER BOUND (penalties are
    monotone) and its ``weighted_score`` covers only the evaluated prefix —
    the regraded fitness is an estimate, not the exact full-run v2 value.
    """
    per_dataset: dict[str, dict[str, Any]] = {}
    graded_total = 0.0
    for name, row in report["per_dataset"].items():
        if name not in baseline_ref:
            raise KeyError(f"Baseline reference missing dataset {name!r}")
        excess_fp, excess_failed = graded_excess_counts(
            row["kind"] == "synthetic",
            int(row["fp"]),
            int(row["n_failed"]),
            baseline_ref[name],
        )
        penalty_fp, penalty_failed = LAMBDA_FP * excess_fp, LAMBDA_FAIL * excess_failed
        graded_total += penalty_fp + penalty_failed
        per_dataset[name] = {
            **row,
            "excess_fp": excess_fp,
            "excess_failed": excess_failed,
            "penalty_fp": penalty_fp,
            "penalty_failed": penalty_failed,
        }
    composite = (
        float(report["weighted_score"])
        - VARIANCE_PENALTY * float(report["variance"])
        - float(report["runtime_penalty"])
    )
    catastrophic = graded_total > CATASTROPHIC_PENALTY
    aborted = report.get("aborted_after_dataset")
    return {
        "fitness_version": 2,
        "fitness": CATASTROPHIC_FITNESS if catastrophic else composite - graded_total,
        "catastrophic": catastrophic,
        "graded_total_penalty": graded_total,
        "weighted_score": float(report["weighted_score"]),
        "variance": float(report["variance"]),
        "runtime_median_s": report.get("runtime_median_s"),
        "runtime_penalty": float(report["runtime_penalty"]),
        "per_dataset": per_dataset,
        "aborted_after_dataset": aborted,
        "partial": aborted is not None,
        "v1_fitness": report.get("fitness"),
        "v1_death": bool(report.get("death", False)),
    }


def evaluate_candidate(
    config_overrides: Mapping[str, Any],
    ctx: EvalContext,
    baseline_ref: Mapping[str, Mapping[str, Any]],
    *,
    fitness_version: int = 1,
) -> tuple[FitnessReport, dict[str, Any]]:
    """Objective body: optimization-split scoreboard run -> composite fitness.

    Passing ``baseline_ref`` as the penalty reference arms the eff#1 early
    abort (version-aware): a candidate whose fitness is already pinned (v1
    death / v2 catastrophic floor) stops paying for the remaining
    spectra/datasets, and ``report.aborted_after_dataset`` records where.
    Fitness is identical to the full run by construction.
    """
    board = evaluate_overrides(
        config_overrides,
        ctx,
        section="optimization",
        death_penalty_ref=baseline_ref,
        fitness_version=fitness_version,
    )
    report = compute_fitness(board["datasets"], baseline_ref, fitness_version=fitness_version)
    report.aborted_after_dataset = board.get("aborted_after_dataset")
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
