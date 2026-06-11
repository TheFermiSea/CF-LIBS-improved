"""Fitness math on rigged metrics (death penalties) + core-hour ledger cap."""

import pytest

import objective as objective_mod
from board_fixtures import make_row as _row

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


@pytest.mark.unit
def test_composite_fitness_known_value():
    report = objective_mod.compute_fitness(_rows(), BASE_REF)
    assert not report.death
    # score: real_wt = 0.6*0.7 + 0.4*(1-5/10) = 0.62 ; presence = 0.6 ;
    # synth (rmse 20 capped at 10) = 0.6*0.3 + 0.4*0 = 0.18
    scores = {"real_wt": 0.62, "presence": 0.60, "synth": 0.18}
    weights = {"real_wt": 1.0, "presence": 0.7, "synth": 0.5}
    expected_weighted = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
    assert report.weighted_score == pytest.approx(expected_weighted)
    import numpy as np

    expected_var = float(np.var(list(scores.values())))
    assert report.variance == pytest.approx(expected_var)
    # wall 1.0 s << 5 s budget: no runtime penalty.
    assert report.runtime_penalty == pytest.approx(0.0, abs=0)
    assert report.fitness == pytest.approx(expected_weighted - 0.5 * expected_var)


@pytest.mark.unit
def test_fp_death_penalty_fires_on_real_dataset():
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=2))
    report = objective_mod.compute_fitness(rows, BASE_REF)  # baseline fp=0, margin 1
    assert report.death
    assert report.fitness == objective_mod.DEATH_FITNESS
    assert any("FP" in r for r in report.death_reasons)
    # fp = baseline + 1 is still allowed.
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=1))
    assert not objective_mod.compute_fitness(rows, BASE_REF).death


@pytest.mark.unit
def test_fp_death_penalty_does_not_fire_on_synthetic():
    rows = _rows(synth=_row("synth", tags=("synthetic",), basis="element_wt", rmse=20.0, fp=9))
    assert not objective_mod.compute_fitness(rows, BASE_REF).death


@pytest.mark.unit
def test_failure_death_penalty():
    # baseline n_failed=1 -> death iff n_failed > 1.25.
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, n_failed=2))
    report = objective_mod.compute_fitness(rows, BASE_REF)
    assert report.death and any("n_failed" in r for r in report.death_reasons)
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, n_failed=1))
    assert not objective_mod.compute_fitness(rows, BASE_REF).death
    # baseline 0 failures: ANY new failure is death (presence row, baseline 0).
    rows = _rows(presence=_row("presence", fp=1, n_failed=1))
    assert objective_mod.compute_fitness(rows, BASE_REF).death


@pytest.mark.unit
def test_runtime_penalty_beyond_budget():
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, wall_s=10.0, n_spectra=30))
    report = objective_mod.compute_fitness(rows, BASE_REF)
    # pooled median wall: 30 spectra at 10s vs 20 at 1s -> 10s; 0.2*(10/5 - 1) = 0.2
    assert report.runtime_median_s == pytest.approx(10.0)
    assert report.runtime_penalty == pytest.approx(0.2)


@pytest.mark.unit
def test_missing_baseline_dataset_raises():
    with pytest.raises(KeyError):
        objective_mod.compute_fitness(_rows(), {"real_wt": BASE_REF["real_wt"]})


@pytest.mark.unit
def test_death_reasons_helper_is_single_source():
    """compute_fitness and the eff#1 early-abort check share
    death_reasons_for_counts — identical reasons (and fitness) by construction."""
    rows = _rows(real_wt=_row("real_wt", basis="element_wt", rmse=5.0, fp=3, n_failed=4))
    report = objective_mod.compute_fitness(rows, BASE_REF)
    assert report.death
    union = []
    for row in rows:
        union += objective_mod.death_reasons_for_counts(
            row["name"],
            objective_mod.dataset_kind(row) == "synthetic",
            row["id_metrics"]["fp"],
            row["n_failed"],
            BASE_REF[row["name"]],
        )
    assert union == report.death_reasons


@pytest.mark.unit
def test_prefix_death_fitness_identical_to_full_run():
    """eff#1 monotonicity: once one dataset dies, the prefix board scores the
    exact DEATH_FITNESS the full board would (remaining datasets are moot)."""
    dying = _row("real_wt", basis="element_wt", rmse=5.0, fp=5)
    full = objective_mod.compute_fitness(_rows(real_wt=dying), BASE_REF)
    prefix = objective_mod.compute_fitness([dying], {"real_wt": BASE_REF["real_wt"]})
    assert full.death and prefix.death
    assert prefix.fitness == full.fitness == objective_mod.DEATH_FITNESS
    # And the default report records no abort.
    assert full.aborted_after_dataset is None
    assert "aborted_after_dataset" in full.to_dict()


@pytest.mark.unit
def test_paired_bootstrap_delta_f1_sign_and_ci():
    cand = [_row("d", f1=1.0)]
    base = [_row("d", f1=1.0)]
    for rec in base[0]["spectra"]:
        rec["fn"] = ["Si"]  # baseline misses Si on every spectrum
    res = objective_mod.paired_bootstrap_delta_f1(cand, base, n_boot=200, seed=1)
    assert res["delta_f1"] > 0
    assert res["ci_low"] == pytest.approx(res["delta_f1"])  # uniform improvement
    # Identical records: delta 0 with degenerate CI.
    res0 = objective_mod.paired_bootstrap_delta_f1(cand, cand, n_boot=100, seed=1)
    assert res0["delta_f1"] == pytest.approx(0.0)
    assert res0["ci_low"] == pytest.approx(0.0) and res0["ci_high"] == pytest.approx(0.0)


@pytest.mark.unit
def test_ledger_cap_refusal(tmp_path):
    pytest.importorskip("optuna")
    import driver

    assert driver.read_ledger(tmp_path) == []
    assert driver.core_hours_used(tmp_path) == pytest.approx(0.0, abs=0)
    assert not driver.budget_exceeded(tmp_path, budget_core_hours=1.0)
    # Charge 2 x 0.5 core-hour (1800 s x 1 cpu, 225 s x 8 cpus).
    driver.append_ledger(tmp_path, {"kind": "trial", "trial": 0, "wall_s": 1800.0, "cpus": 1})
    driver.append_ledger(tmp_path, {"kind": "trial", "trial": 1, "wall_s": 225.0, "cpus": 8})
    assert driver.core_hours_used(tmp_path) == pytest.approx(1.0)
    assert driver.budget_exceeded(tmp_path, budget_core_hours=1.0)
    assert not driver.budget_exceeded(tmp_path, budget_core_hours=1.5)
    entries = driver.read_ledger(tmp_path)
    assert len(entries) == 2 and all("ts" in e for e in entries)


@pytest.mark.unit
def test_holdout_quota_enforced(tmp_path):
    pytest.importorskip("optuna")
    import driver
    import holdout_eval

    holdout_eval.enforce_holdout_quota(tmp_path, force=False)  # empty ledger: fine
    driver.append_ledger(tmp_path, {"kind": "holdout_query", "wall_s": 0.0, "cpus": 1})
    with pytest.raises(SystemExit):
        holdout_eval.enforce_holdout_quota(tmp_path, force=False)
    holdout_eval.enforce_holdout_quota(tmp_path, force=True)  # human override
