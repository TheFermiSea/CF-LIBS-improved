"""rescore.py on a fabricated journal + driver warm-start / fitness-version pin.

The real run1 journal lives on the cluster only; these tests fabricate a
journal with the exact recorded shapes (COMPLETE trials carrying v1
``fitness_report`` user_attrs, including -1e9 death trials and a partial
aborted trial) so the offline regrade path is fully exercised locally. The
maintainer runs ``rescore.py`` against run1 on the cluster.
"""

import json

import pytest

optuna = pytest.importorskip("optuna")

import driver  # noqa: E402
import objective as objective_mod  # noqa: E402
import rescore  # noqa: E402
from optuna.distributions import CategoricalDistribution, FloatDistribution  # noqa: E402

BASE_REF = {
    "d1": {"fp": 0, "n_failed": 0, "n_run": 10, "f1": 0.5},
    "d2": {"fp": 0, "n_failed": 1, "n_run": 10, "f1": 0.5},
}


def _report(weighted, *, fp_d1=0, failed_d1=0, death=False, aborted=None, datasets=("d1", "d2")):
    """A recorded v1 fitness_report dict (the journal user_attrs shape)."""
    per_dataset = {}
    for name in datasets:
        fp = fp_d1 if name == "d1" else 0
        n_failed = failed_d1 if name == "d1" else 0
        per_dataset[name] = {
            "kind": "real_presence_only",
            "score": weighted,
            "weight": 1.0,
            "f1": weighted,
            "precision": 1.0,
            "recall": 1.0,
            "fp": fp,
            "n_failed": n_failed,
            "rmse_wt_median": None,
        }
    return {
        "fitness": objective_mod.DEATH_FITNESS if death else weighted,
        "death": death,
        "death_reasons": ["d1: rigged"] if death else [],
        "weighted_score": weighted,
        "variance": 0.0,
        "runtime_median_s": 1.0,
        "runtime_penalty": 0.0,
        "per_dataset": per_dataset,
        "aborted_after_dataset": aborted,
    }


def _trial(value, params, report, state=optuna.trial.TrialState.COMPLETE):
    distributions = {}
    for key, val in params.items():
        if isinstance(val, bool):
            distributions[key] = CategoricalDistribution([False, True])
        else:
            distributions[key] = FloatDistribution(0.0, 100.0)
    user_attrs = {"fitness_report": report} if report is not None else {}
    return optuna.trial.create_trial(
        state=state,
        value=value if state == optuna.trial.TrialState.COMPLETE else None,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
    )


def _fabricate_study_dir(tmp_path):
    """A study dir whose journal mirrors run1's pathology in miniature."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / driver.CONFIG_FILENAME).write_text(
        json.dumps({"study_name": "campaign1-phaseA", "fitness_version": 1})
    )
    (study_dir / driver.BASELINE_FILENAME).write_text(
        json.dumps({"reference": BASE_REF, "wall_s": 1.0})
    )
    study = optuna.create_study(
        study_name="campaign1-phaseA",
        storage=driver.study_storage(study_dir),
        direction="maximize",
    )
    # 0: the surviving clean baseline (run1: 0.3293).
    study.add_trial(_trial(0.3293, {"min_snr": 7.6}, _report(0.3293)))
    # 1: v1 death with weighted_score ABOVE the baseline (run1 trial 1:
    #    0.352 vs 0.3293) — ONE excess FP. Carries the removed
    #    use_deconvolution param (run1 predates c1-knobs-v2).
    study.add_trial(
        _trial(
            objective_mod.DEATH_FITNESS,
            {"min_snr": 3.0, "use_deconvolution": False},
            _report(0.352, fp_d1=2, death=True),
        )
    )
    # 2: genuinely catastrophic v1 death (run1: FP 87).
    study.add_trial(
        _trial(
            objective_mod.DEATH_FITNESS,
            {"min_snr": 2.1, "use_deconvolution": True},
            _report(0.56, fp_d1=87, death=True),
        )
    )
    # 3: v1 death that early-aborted (partial board: d1 only).
    study.add_trial(
        _trial(
            objective_mod.DEATH_FITNESS,
            {"min_snr": 4.2},
            _report(0.40, fp_d1=3, death=True, aborted="d1", datasets=("d1",)),
        )
    )
    # 4: COMPLETE but without a recorded report (cannot regrade -> skipped).
    study.add_trial(_trial(0.1, {"min_snr": 9.9}, None))
    # 5: FAIL state (structural failure) — never regraded.
    study.add_trial(_trial(None, {"min_snr": 5.5}, None, state=optuna.trial.TrialState.FAIL))
    return study_dir


@pytest.mark.unit
def test_rescore_fabricated_journal_ranking_and_artifacts(tmp_path, capsys):
    study_dir = _fabricate_study_dir(tmp_path)
    assert rescore.main(["--study-dir", str(study_dir), "--top-k", "3"]) == 0

    out_dir = study_dir / "rescore_v2"
    data = json.loads((out_dir / "rescore_v2.json").read_text())
    assert data["meta"]["n_regraded"] == 4
    assert data["meta"]["n_skipped"] == 1 and data["meta"]["skipped_trials"] == [4]

    by_trial = {e["trial"]: e for e in data["trials"]}
    # v2 fitness: clean 0.3293; near-miss 0.352 - 0.05 = 0.302; partial
    # 0.40 - 2*0.05 = 0.30; catastrophic floor -1e3.
    assert by_trial[0]["fitness_v2"] == pytest.approx(0.3293)
    assert by_trial[1]["fitness_v2"] == pytest.approx(0.302)
    assert by_trial[3]["fitness_v2"] == pytest.approx(0.30)
    assert by_trial[2]["fitness_v2"] == objective_mod.CATASTROPHIC_FITNESS
    # Ranked order: the near-miss now sits between baseline and the rest —
    # the gradient run1's flat -1e9 erased.
    assert [e["trial"] for e in data["trials"]] == [0, 1, 3, 2]
    assert by_trial[1]["v1_death"] and by_trial[1]["fitness_v1"] == objective_mod.DEATH_FITNESS
    assert by_trial[2]["catastrophic"] and not by_trial[1]["catastrophic"]
    assert by_trial[3]["partial"] and by_trial[3]["aborted_after_dataset"] == "d1"
    assert by_trial[1]["excess_fp_total"] == 1

    report_md = (out_dir / "rescore_v2.md").read_text()
    assert "CATASTROPHIC" in report_md and "partial" in report_md

    warm = json.loads((out_dir / "warm_start_top3.json").read_text())
    assert warm["fitness_version"] == 2
    # Catastrophic trial 2 excluded; ranked survivors in order.
    assert warm["trials"] == [0, 1, 3]
    # The removed knob is stripped from enqueueable params (knob-space v2).
    assert all("use_deconvolution" not in p for p in warm["params"])
    assert warm["params"][1]["min_snr"] == pytest.approx(3.0)


@pytest.mark.unit
def test_warm_start_enqueue_round_trip(tmp_path):
    """rescore output -> driver.enqueue_warm_starts -> queued fixed params."""
    study_dir = _fabricate_study_dir(tmp_path)
    rescore.main(["--study-dir", str(study_dir), "--top-k", "2"])
    warm_path = study_dir / "rescore_v2" / "warm_start_top2.json"

    study = optuna.create_study(direction="maximize")
    assert driver.enqueue_warm_starts(study, warm_path) == 2
    waiting = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    fixed = [t.system_attrs["fixed_params"] for t in waiting]
    assert fixed[0]["min_snr"] == pytest.approx(7.6)
    assert fixed[1]["min_snr"] == pytest.approx(3.0)
    assert all("use_deconvolution" not in p for p in fixed)


@pytest.mark.unit
def test_enqueue_drops_unknown_params_from_hand_edited_file(tmp_path, capsys):
    path = tmp_path / "warm.json"
    path.write_text(
        json.dumps({"params": [{"min_snr": 4.0, "use_deconvolution": True, "bogus_knob": 1}]})
    )
    study = optuna.create_study(direction="maximize")
    assert driver.enqueue_warm_starts(study, path) == 1
    (waiting,) = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    assert waiting.system_attrs["fixed_params"] == {"min_snr": 4.0}
    out = capsys.readouterr().out
    assert "use_deconvolution" in out and "bogus_knob" in out

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"params": "nope"}))
    with pytest.raises(SystemExit):
        driver.enqueue_warm_starts(study, bad)


@pytest.mark.unit
def test_resolve_fitness_version_pin_and_refusal(tmp_path):
    import knob_space

    # Pre-v2 study config (run1): no key -> version 1.
    assert driver.resolve_fitness_version(tmp_path, {}) == 1
    assert driver.resolve_fitness_version(tmp_path, {"fitness_version": 2}) == 2
    # Matching explicit CLI assert passes.
    assert driver.resolve_fitness_version(tmp_path, {"fitness_version": 2}, 2) == 2
    # CLI contradiction -> refusal (no mixed-fitness journals).
    with pytest.raises(SystemExit, match="refusing to append"):
        driver.resolve_fitness_version(tmp_path, {"fitness_version": 2}, 1)
    with pytest.raises(SystemExit, match="refusing to append"):
        driver.resolve_fitness_version(tmp_path, {}, 2)  # run1-style config vs v2 worker
    # Frozen-manifest pin must agree with the study config.
    manifest = tmp_path / knob_space.FROZEN_MANIFEST_FILENAME
    manifest.write_text(json.dumps({"extra": {"fitness_version": 1}}))
    with pytest.raises(SystemExit, match="inconsistent"):
        driver.resolve_fitness_version(tmp_path, {"fitness_version": 2})
    manifest.write_text(json.dumps({"extra": {"fitness_version": 2}}))
    assert driver.resolve_fitness_version(tmp_path, {"fitness_version": 2}) == 2
    # Unsupported pins are refused outright (consistent pins, unknown math).
    manifest.write_text(json.dumps({"extra": {"fitness_version": 9}}))
    with pytest.raises(SystemExit, match="unsupported"):
        driver.resolve_fitness_version(tmp_path, {"fitness_version": 9})


@pytest.mark.unit
def test_cli_fitness_version_defaults(monkeypatch):
    """init defaults to v2 for new studies; worker defaults to the study pin."""
    assert driver.DEFAULT_FITNESS_VERSION == 2
    assert set(objective_mod.SUPPORTED_FITNESS_VERSIONS) == {1, 2}
    captured = {}
    monkeypatch.setattr(driver, "cmd_init", lambda args: captured.update(vars(args)) or 0)
    assert driver.main(["init", "--study-dir", "x"]) == 0
    assert captured["fitness_version"] == 2
    assert captured["enqueue_from"] is None
    captured.clear()
    monkeypatch.setattr(driver, "cmd_worker", lambda args: captured.update(vars(args)) or 0)
    assert driver.main(["worker", "--study-dir", "x"]) == 0
    assert captured["fitness_version"] is None  # None = use the journal's pin
