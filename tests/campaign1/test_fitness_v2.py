"""FITNESS-V2: graded penalties, catastrophic floor, abort==full-run determinism."""

from types import SimpleNamespace

import pytest

import objective as objective_mod


def _row(
    name,
    *,
    tags=("real",),
    f1=0.7,
    precision=1.0,
    recall=0.6,
    fp=0,
    n_failed=0,
    rmse=None,
    basis="presence_only",
    wall_s=1.0,
    n_spectra=10,
):
    spectra = [
        {
            "spectrum_id": f"{name}/s{i}",
            "status": "ok",
            "wall_s": wall_s,
            "composition_basis": basis,
            "tp": ["Fe"],
            "fp": [],
            "fn": [],
        }
        for i in range(n_spectra)
    ]
    composition = {"rmse_wt_median": rmse} if rmse is not None else None
    return {
        "name": name,
        "tags": list(tags),
        "n_run": n_spectra,
        "n_failed": n_failed,
        "id_metrics": {
            "tp": 10,
            "fp": fp,
            "fn": 4,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "composition": composition,
        "spectra": spectra,
        "failures": {},
        "runtime": {"median_wall_s": wall_s},
    }


BASE_REF = {
    "real_wt": {"fp": 0, "n_failed": 1, "n_run": 10, "f1": 0.7},
    "presence": {"fp": 1, "n_failed": 0, "n_run": 10, "f1": 0.6},
    "synth": {"fp": 3, "n_failed": 5, "n_run": 10, "f1": 0.3},
}


def _rows(**overrides):
    rows = {
        "real_wt": _row("real_wt", basis="element_wt", rmse=5.0, f1=0.7),
        "presence": _row("presence", f1=0.6, fp=1),
        "synth": _row("synth", tags=("synthetic",), basis="element_wt", rmse=20.0, f1=0.3),
    }
    rows.update(overrides)
    return list(rows.values())


# ---------------------------------------------------------------------------
# Graded math on rigged metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graded_excess_counts_exact_thresholds():
    base = {"fp": 2, "n_failed": 4}
    # FP allowance = base + 1; failure allowance = ceil(1.25 * base) = 5.
    assert objective_mod.graded_excess_counts("d", False, 3, 5, base) == (0, 0)
    assert objective_mod.graded_excess_counts("d", False, 4, 6, base) == (1, 1)
    assert objective_mod.graded_excess_counts("d", False, 9, 12, base) == (6, 7)
    # Synthetic datasets are FP-exempt (mirrors the v1 death condition).
    assert objective_mod.graded_excess_counts("d", True, 99, 6, base) == (0, 1)
    # Zero-failure baseline: ceil(0) = 0, ANY failure is excess.
    assert objective_mod.graded_excess_counts("d", False, 0, 1, {"fp": 0, "n_failed": 0}) == (0, 1)


@pytest.mark.unit
def test_graded_penalties_exact_values():
    # real_wt: fp=3 (base 0, allowed 1 -> excess 2), n_failed=4 (base 1,
    # allowed ceil(1.25)=2 -> excess 2).
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=3, n_failed=4))
    v2 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=2)
    expected_penalty = 2 * objective_mod.LAMBDA_FP + 2 * objective_mod.LAMBDA_FAIL
    assert v2.graded_total_penalty == pytest.approx(expected_penalty)
    assert v2.fitness_version == 2
    assert not v2.catastrophic and not v2.death
    pd = v2.per_dataset["real_wt"]
    assert (pd["excess_fp"], pd["excess_failed"]) == (2, 2)
    assert pd["penalty_fp"] == pytest.approx(2 * objective_mod.LAMBDA_FP)
    assert pd["penalty_failed"] == pytest.approx(2 * objective_mod.LAMBDA_FAIL)
    # fitness = the clean composite minus the graded penalty.
    clean = objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=2)
    assert v2.fitness == pytest.approx(clean.fitness - expected_penalty)
    # The same rows are a flat death under v1 — v2 keeps the gradient.
    v1 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=1)
    assert v1.fitness == objective_mod.DEATH_FITNESS
    assert v2.fitness > v1.fitness


@pytest.mark.unit
def test_synthetic_fp_not_penalized_failures_are():
    rows = _rows(
        synth=_row("synth", tags=("synthetic",), basis="element_wt", rmse=20.0, fp=50, n_failed=8)
    )
    v2 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=2)
    pd = v2.per_dataset["synth"]
    # base n_failed=5 -> allowed ceil(6.25)=7 -> excess 1; FP exempt.
    assert (pd["excess_fp"], pd["excess_failed"]) == (0, 1)
    assert v2.graded_total_penalty == pytest.approx(objective_mod.LAMBDA_FAIL)


@pytest.mark.unit
def test_clean_trial_scores_identically_under_v1_and_v2():
    """Zero excess counts: run1 (v1) and run2 (v2) numbers stay comparable."""
    v1 = objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=1)
    v2 = objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=2)
    assert not v1.death and not v2.catastrophic
    assert v2.fitness == pytest.approx(v1.fitness)
    assert v2.graded_total_penalty == pytest.approx(0.0, abs=0)
    # In-allowance counts are free: fp = base+1, n_failed = ceil(1.25*base).
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=1, n_failed=2))
    graded = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=2)
    assert graded.graded_total_penalty == pytest.approx(0.0, abs=0)


@pytest.mark.unit
def test_unsupported_fitness_version_rejected():
    with pytest.raises(ValueError):
        objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=3)
    with pytest.raises(ValueError):
        objective_mod.EarlyAbortTracker(BASE_REF, fitness_version=0)


# ---------------------------------------------------------------------------
# Catastrophic floor
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_catastrophic_floor_is_flat_and_distinct_from_v1_death():
    # 25 excess FPs -> penalty 1.25 > CATASTROPHIC_PENALTY.
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=26))
    v2 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=2)
    assert v2.catastrophic
    assert v2.fitness == objective_mod.CATASTROPHIC_FITNESS
    assert v2.graded_total_penalty == pytest.approx(25 * objective_mod.LAMBDA_FP)
    # Flat: an even worse trial scores exactly the same floor.
    worse = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=90, n_failed=9))
    assert (
        objective_mod.compute_fitness(worse, BASE_REF, fitness_version=2).fitness
        == objective_mod.CATASTROPHIC_FITNESS
    )
    # Far below all real scores, far above the v1/structural death value.
    assert objective_mod.CATASTROPHIC_FITNESS < -100
    assert objective_mod.CATASTROPHIC_FITNESS > objective_mod.DEATH_FITNESS


@pytest.mark.unit
def test_penalty_exactly_at_threshold_is_not_catastrophic():
    # Exactly 20 excess FPs -> penalty 1.0 == CATASTROPHIC_PENALTY: graded.
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=21))
    v2 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=2)
    assert v2.graded_total_penalty == pytest.approx(objective_mod.CATASTROPHIC_PENALTY)
    assert not v2.catastrophic
    assert v2.fitness > objective_mod.CATASTROPHIC_FITNESS


# ---------------------------------------------------------------------------
# Near-miss ordering (the run1 pathology: signal where v1 had a flat -1e9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_near_miss_ordering():
    ref = {"d": {"fp": 0, "n_failed": 0, "n_run": 10, "f1": 0.33}}

    def fitness(f1, fp):
        rows = [_row("d", f1=f1, fp=fp)]
        return objective_mod.compute_fitness(rows, ref, fitness_version=2).fitness

    clean_baseline = fitness(0.33, 0)  # 0.33
    near_miss = fitness(0.35, 2)  # 0.35 - 1*0.05 = 0.30
    three_excess = fitness(0.30, 4)  # 0.30 - 3*0.05 = 0.15
    catastrophic = fitness(0.56, 40)  # penalty 1.95 -> floor
    assert near_miss == pytest.approx(0.30)
    assert three_excess == pytest.approx(0.15)
    # The ordering TPE needs: one excess FP costs more than the score gain...
    assert near_miss < clean_baseline
    # ...but a near-miss still ranks above worse offenders and catastrophics.
    assert near_miss > three_excess > catastrophic
    assert catastrophic == objective_mod.CATASTROPHIC_FITNESS


# ---------------------------------------------------------------------------
# Early-abort == full-run determinism (rigged catastrophic stream)
# ---------------------------------------------------------------------------


def _ok_record(name, i, fp=()):
    return {
        "spectrum_id": f"{name}/s{i}",
        "status": "ok",
        "error": None,
        "wall_s": 0.5,
        "composition_basis": "presence_only",
        "tp": ["Fe"],
        "fp": list(fp),
        "fn": [],
    }


def _error_record(name, i):
    return {
        "spectrum_id": f"{name}/s{i}",
        "status": "error",
        "error": "RuntimeError: rigged failure",
        "wall_s": 0.1,
        "composition_basis": "presence_only",
        "tp": [],
        "fp": [],
        "fn": ["Fe"],
    }


@pytest.mark.unit
def test_tracker_abort_fitness_equals_full_run_on_catastrophic_stream():
    """Feed a rigged catastrophic record stream through the SAME accumulation
    the eff#1 abort uses; the partial board (prefix up to the abort) and the
    full board must score the identical flat CATASTROPHIC_FITNESS."""
    from cflibs.benchmark.scoreboard import _aggregate_dataset

    ref = {"d1": {"fp": 0, "n_failed": 0}, "d2": {"fp": 0, "n_failed": 0}}
    # 6 FPs per record: penalty crosses 1.0 inside d1 (allowed FP = 1).
    stream = {
        "d1": [_ok_record("d1", i, fp=["Ag", "Sn", "W", "Bi", "Th", "Zr"]) for i in range(6)],
        "d2": [_error_record("d2", i) for i in range(4)],
    }

    tracker = objective_mod.EarlyAbortTracker(ref, fitness_version=2)
    consumed: dict[str, list] = {}
    aborted_at = None
    for name, records in stream.items():
        tracker.start_dataset(name, synthetic=False)
        consumed[name] = []
        for record in records:
            consumed[name].append(record)
            if tracker.add_record(record):
                aborted_at = name
                break
        if aborted_at:
            break
        tracker.finish_dataset()

    # The abort fires mid-d1: fp accumulates 6, 12, 18, 24 -> excess 23 at
    # the 4th record (penalty 1.15 > 1.0).
    assert aborted_at == "d1"
    assert len(consumed["d1"]) == 4

    def board(records_by_name):
        return [
            _aggregate_dataset(name, ["real"], records, n_total=10, sampled=True, notes="rigged")
            for name, records in records_by_name.items()
        ]

    partial = objective_mod.compute_fitness(
        board(consumed), {k: ref[k] for k in consumed}, fitness_version=2
    )
    full = objective_mod.compute_fitness(board(stream), ref, fitness_version=2)
    assert partial.catastrophic and full.catastrophic
    assert partial.fitness == full.fitness == objective_mod.CATASTROPHIC_FITNESS


@pytest.mark.unit
def test_tracker_v2_accumulates_penalty_across_datasets():
    """Sub-catastrophic datasets must SUM: the abort can fire on a dataset
    that is individually harmless (total penalty is what matters in v2)."""
    ref = {"d1": {"fp": 0, "n_failed": 0}, "d2": {"fp": 0, "n_failed": 0}}
    tracker = objective_mod.EarlyAbortTracker(ref, fitness_version=2)
    tracker.start_dataset("d1", synthetic=False)
    # 12 excess FPs on d1: penalty 0.60 — no abort.
    assert not tracker.add_record(_ok_record("d1", 0, fp=["X"] * 13))
    tracker.finish_dataset()
    tracker.start_dataset("d2", synthetic=False)
    # 8 excess more on d2 (9 FPs, allowed 1): total 0.60 + 0.40 = 1.00 — not > 1.0.
    assert not tracker.add_record(_ok_record("d2", 0, fp=["X"] * 9))
    # One more FP tips the TOTAL over the floor.
    assert tracker.add_record(_ok_record("d2", 1, fp=["X"]))


@pytest.mark.unit
def test_tracker_v1_behavior_unchanged():
    ref = {"d1": {"fp": 0, "n_failed": 0}}
    tracker = objective_mod.EarlyAbortTracker(ref, fitness_version=1)
    tracker.start_dataset("d1", synthetic=False)
    assert not tracker.add_record(_ok_record("d1", 0, fp=["Ag"]))  # fp=1 = base+1: alive
    assert tracker.add_record(_ok_record("d1", 1, fp=["Sn"]))  # fp=2 > base+1: death
    with pytest.raises(KeyError):
        tracker.start_dataset("unknown", synthetic=False)


@pytest.mark.unit
def test_evaluate_overrides_v2_abort_equals_full_run(monkeypatch, tmp_path):
    """End-to-end through evaluate_overrides with a rigged record stream:
    the aborted board and the full board score the identical floor fitness."""
    manifest = {
        "optimization": {
            "d1": [f"d1/s{i}" for i in range(6)],
            "d2": [f"d2/s{i}" for i in range(4)],
        },
        "holdout": {},
    }
    records = [_ok_record("d1", i, fp=["Ag", "Sn", "W", "Bi", "Th", "Zr"]) for i in range(6)] + [
        _error_record("d2", i) for i in range(4)
    ]

    entry = SimpleNamespace(tags=("real",), notes="rigged")
    monkeypatch.setattr(objective_mod, "_dataset_entry", lambda name: entry)
    monkeypatch.setattr(
        objective_mod,
        "_get_eval_items",
        lambda ctx, section, name, chosen: [(sid, None, None, None) for sid in chosen],
    )
    monkeypatch.setattr(
        objective_mod, "_iter_payload_records", lambda payloads, ctx: iter(list(records))
    )
    terminated = []
    monkeypatch.setattr(objective_mod, "_terminate_pool", lambda: terminated.append(True))

    ctx = objective_mod.EvalContext(
        db_path=tmp_path / "unused.db",
        manifest=manifest,
        datasets=("d1", "d2"),
        n_procs=1,
    )
    ref = {"d1": {"fp": 0, "n_failed": 0}, "d2": {"fp": 0, "n_failed": 0}}

    aborted_board = objective_mod.evaluate_overrides(
        {}, ctx, death_penalty_ref=ref, fitness_version=2
    )
    assert aborted_board["aborted_after_dataset"] == "d1"
    assert terminated  # the queued remainder was killed
    assert [row["name"] for row in aborted_board["datasets"]] == ["d1"]
    assert aborted_board["datasets"][0]["n_run"] == 4  # abort at the 4th record

    full_board = objective_mod.evaluate_overrides({}, ctx, fitness_version=2)
    assert "aborted_after_dataset" not in full_board
    assert [row["name"] for row in full_board["datasets"]] == ["d1", "d2"]

    aborted_fit = objective_mod.compute_fitness(aborted_board["datasets"], ref, fitness_version=2)
    full_fit = objective_mod.compute_fitness(full_board["datasets"], ref, fitness_version=2)
    assert aborted_fit.catastrophic and full_fit.catastrophic
    assert aborted_fit.fitness == full_fit.fitness == objective_mod.CATASTROPHIC_FITNESS


# ---------------------------------------------------------------------------
# Offline regrade (the rescore.py math)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_regrade_recorded_v1_report_matches_direct_v2():
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=3, n_failed=4))
    recorded_v1 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=1).to_dict()
    assert recorded_v1["fitness"] == objective_mod.DEATH_FITNESS  # a run1-style death
    direct_v2 = objective_mod.compute_fitness(rows, BASE_REF, fitness_version=2)
    regraded = objective_mod.regrade_report_v2(recorded_v1, BASE_REF)
    assert regraded["fitness"] == pytest.approx(direct_v2.fitness)
    assert regraded["graded_total_penalty"] == pytest.approx(direct_v2.graded_total_penalty)
    assert regraded["catastrophic"] == direct_v2.catastrophic
    assert not regraded["partial"]
    assert regraded["v1_death"] is True
    assert regraded["v1_fitness"] == objective_mod.DEATH_FITNESS
    pd = regraded["per_dataset"]["real_wt"]
    assert (pd["excess_fp"], pd["excess_failed"]) == (2, 2)


@pytest.mark.unit
def test_regrade_flags_partial_aborted_reports():
    report = objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=1)
    report.aborted_after_dataset = "presence"
    regraded = objective_mod.regrade_report_v2(report.to_dict(), BASE_REF)
    assert regraded["partial"] is True
    assert regraded["aborted_after_dataset"] == "presence"


@pytest.mark.unit
def test_regrade_missing_baseline_dataset_raises():
    recorded = objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=1).to_dict()
    with pytest.raises(KeyError):
        objective_mod.regrade_report_v2(recorded, {"real_wt": BASE_REF["real_wt"]})


@pytest.mark.unit
def test_v2_report_records_everything_v1_did():
    v1_keys = set(objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=1).to_dict())
    v2 = objective_mod.compute_fitness(_rows(), BASE_REF, fitness_version=2).to_dict()
    assert v1_keys <= set(v2)
    assert v2["fitness_version"] == 2
    assert {"graded_total_penalty", "catastrophic"} <= set(v2)
    for row in v2["per_dataset"].values():
        assert {"excess_fp", "excess_failed", "penalty_fp", "penalty_failed"} <= set(row)
