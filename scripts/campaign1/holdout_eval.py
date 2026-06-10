"""Campaign 1 holdout evaluation + adoption-gate verdict (design 2.5 / 3.4).

Takes the top-K completed trials of a study, re-evaluates each (and the
production baseline) on

1. the **full optimization split** (no per-trial subsampling unless
   ``--spectra-per-dataset`` is given for smoke runs), and
2. the **holdout set** (``bhvo2_chemcam``, ``emslibs2019``, and the held-out
   40% target splits of chemcam_calib / csa_planetary / silva2022),

then applies the adoption gate:

- **G1 optimization gate:** pooled Delta(micro-F1) >= +0.02 AND the paired
  per-spectrum bootstrap 95% CI excludes 0 (design 2.4 anti-noise-mining).
- **G2 holdout gate:** pooled holdout Delta(micro-F1) >= +0.02 with the
  bootstrap CI excluding 0, and no individual holdout dataset regresses
  beyond its bootstrap noise (CI entirely below 0 = fail). BHVO-2 (n=4)
  is additionally gated point-wise, not by bootstrap.
- **G3 precision/failure gate:** zero new FP on bhvo2_chemcam, no FP
  regression and no new failures on any real holdout dataset.
- **G4 runtime gate:** pooled median s/spectrum on the holdout run
  <= 1.5x baseline.

Every holdout query is ledger-logged; the per-phase quota (1 per week,
design 2.1) is enforced — exceeding it requires an explicit ``--force``
(the overfitting kill-switch).

The verdict report is markdown; among ADOPT candidates the recommendation is
the one with the best worst-dataset score (design 3.4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import optuna

_CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(_CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGN_DIR))

import driver  # noqa: E402
import knob_space  # noqa: E402
import objective as objective_mod  # noqa: E402
import splits  # noqa: E402

MIN_DELTA_F1 = 0.02
RUNTIME_FACTOR = 1.5
HOLDOUT_QUOTA_DAYS = 7


from cflibs.benchmark.scoreboard import fmt_metric as _fmt  # noqa: E402


def top_k_trials(study: optuna.Study, k: int) -> list[optuna.trial.FrozenTrial]:
    """Top-K completed trials by fitness, deduplicated by params."""
    complete = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete.sort(key=lambda t: t.value, reverse=True)
    seen: set = set()
    picked = []
    for t in complete:
        key = tuple(sorted(t.params.items()))
        if key in seen:
            continue
        seen.add(key)
        picked.append(t)
        if len(picked) >= k:
            break
    return picked


def enforce_holdout_quota(study_dir: Path, force: bool) -> None:
    """The overfitting kill-switch: <=1 holdout query per phase per week."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=HOLDOUT_QUOTA_DAYS)
    recent = [
        e
        for e in driver.read_ledger(study_dir)
        if e.get("kind") == "holdout_query" and datetime.fromisoformat(e["ts"]) >= cutoff
    ]
    if recent and not force:
        raise SystemExit(
            f"HOLDOUT QUOTA: {len(recent)} holdout quer(ies) already logged in the last "
            f"{HOLDOUT_QUOTA_DAYS} days (design 2.1 budget: <=1/phase/week). "
            "A human must pass --force to override."
        )


def _gate_results(
    name: str,
    cand_opt: dict,
    cand_hold: dict,
    base_opt: dict,
    base_hold: dict,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Apply the adoption gates to one candidate; return verdict details."""
    opt_boot = objective_mod.paired_bootstrap_delta_f1(
        cand_opt["datasets"], base_opt["datasets"], n_boot=n_boot, seed=seed
    )
    hold_boot = objective_mod.paired_bootstrap_delta_f1(
        cand_hold["datasets"], base_hold["datasets"], n_boot=n_boot, seed=seed
    )
    g1 = opt_boot["delta_f1"] >= MIN_DELTA_F1 and opt_boot["ci_low"] > 0.0

    per_holdout: dict[str, dict[str, Any]] = {}
    g2_no_regression = True
    base_hold_rows = {r["name"]: r for r in base_hold["datasets"]}
    for row in cand_hold["datasets"]:
        base_row = base_hold_rows[row["name"]]
        ds_boot = objective_mod.paired_bootstrap_delta_f1(
            [row], [base_row], n_boot=n_boot, seed=seed
        )
        regressed = ds_boot["ci_high"] < 0.0
        if row["name"] == "bhvo2_chemcam":
            # n=4 can never pass a bootstrap alone (design 2.4): gate point-wise.
            regressed = row["id_metrics"]["f1"] < base_row["id_metrics"]["f1"]
        g2_no_regression &= not regressed
        per_holdout[row["name"]] = {
            "f1": row["id_metrics"]["f1"],
            "f1_baseline": base_row["id_metrics"]["f1"],
            "delta_f1": ds_boot["delta_f1"],
            "ci": [ds_boot["ci_low"], ds_boot["ci_high"]],
            "fp": row["id_metrics"]["fp"],
            "fp_baseline": base_row["id_metrics"]["fp"],
            "n_failed": row["n_failed"],
            "n_failed_baseline": base_row["n_failed"],
            "regressed": bool(regressed),
        }
    g2 = hold_boot["delta_f1"] >= MIN_DELTA_F1 and hold_boot["ci_low"] > 0.0 and g2_no_regression

    g3 = True
    g3_reasons: list[str] = []
    for ds_name, info in per_holdout.items():
        if info["fp"] > info["fp_baseline"]:
            g3 = False
            g3_reasons.append(f"{ds_name}: FP {info['fp']} > baseline {info['fp_baseline']}")
        if info["n_failed"] > info["n_failed_baseline"]:
            g3 = False
            g3_reasons.append(
                f"{ds_name}: failures {info['n_failed']} > baseline " f"{info['n_failed_baseline']}"
            )

    t_cand = objective_mod.pooled_runtime_median(cand_hold["datasets"])
    t_base = objective_mod.pooled_runtime_median(base_hold["datasets"])
    g4 = True
    if t_cand is not None and t_base is not None and t_base > 0:
        g4 = t_cand <= RUNTIME_FACTOR * t_base

    worst_score = min(objective_mod.dataset_score(row) for row in cand_hold["datasets"])
    return {
        "name": name,
        "opt_bootstrap": opt_boot,
        "holdout_bootstrap": hold_boot,
        "per_holdout": per_holdout,
        "runtime_median_s": t_cand,
        "runtime_median_baseline_s": t_base,
        "worst_holdout_score": worst_score,
        "gates": {
            "G1_optimization": bool(g1),
            "G2_holdout": bool(g2),
            "G3_precision_failures": bool(g3),
            "G4_runtime": bool(g4),
        },
        "g3_reasons": g3_reasons,
        "adopt": bool(g1 and g2 and g3 and g4),
    }


def render_report(study_dir: Path, results: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# Campaign 1 Holdout Verdict",
        "",
        f"Generated: {meta['generated_utc']}  ",
        f"Study: `{study_dir}`  ",
        f"Candidates: top-{meta['top_k']} by optimization fitness  ",
        f"Bootstrap: B={meta['n_boot']}, seed={meta['seed']}  ",
        f"Spectra/dataset cap: {meta['spectra_per_dataset'] or 'full split'}",
        "",
        "Adoption gate (design 2.5 + task spec): G1 optimization pooled ΔF1 ≥ "
        f"+{MIN_DELTA_F1} with 95% CI excluding 0; G2 holdout pooled ΔF1 ≥ "
        f"+{MIN_DELTA_F1} with CI excluding 0 and no per-dataset regression beyond "
        "bootstrap noise (BHVO-2 gated point-wise); G3 zero new FP / zero new "
        f"failures on real holdout datasets; G4 holdout runtime ≤ {RUNTIME_FACTOR}× "
        "baseline.",
        "",
        "| Candidate | opt ΔF1 [CI] | holdout ΔF1 [CI] | G1 | G2 | G3 | G4 | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        ob, hb = res["opt_bootstrap"], res["holdout_bootstrap"]
        gates = res["gates"]
        lines.append(
            f"| {res['name']} "
            f"| {ob['delta_f1']:+.3f} [{ob['ci_low']:+.3f}, {ob['ci_high']:+.3f}] "
            f"| {hb['delta_f1']:+.3f} [{hb['ci_low']:+.3f}, {hb['ci_high']:+.3f}] "
            f"| {'PASS' if gates['G1_optimization'] else 'fail'} "
            f"| {'PASS' if gates['G2_holdout'] else 'fail'} "
            f"| {'PASS' if gates['G3_precision_failures'] else 'fail'} "
            f"| {'PASS' if gates['G4_runtime'] else 'fail'} "
            f"| {'**ADOPT**' if res['adopt'] else 'REJECT'} |"
        )
    adopted = [r for r in results if r["adopt"]]
    lines.append("")
    if adopted:
        best = max(adopted, key=lambda r: r["worst_holdout_score"])
        lines.append(
            f"**Recommendation:** `{best['name']}` — best worst-dataset holdout score "
            f"({best['worst_holdout_score']:.3f}) among ADOPT candidates (design 3.4)."
        )
    else:
        lines.append("**Recommendation:** no candidate passes the adoption gate.")
    for res in results:
        lines += ["", f"## {res['name']}", ""]
        lines.append(
            "| Holdout dataset | F1 (base→cand) | ΔF1 [CI] | FP (base→cand) "
            "| failures (base→cand) | regressed |"
        )
        lines.append("|---|---|---|---|---|---|")
        for ds, info in res["per_holdout"].items():
            lines.append(
                f"| {ds} | {info['f1_baseline']:.3f} → {info['f1']:.3f} "
                f"| {info['delta_f1']:+.3f} [{info['ci'][0]:+.3f}, {info['ci'][1]:+.3f}] "
                f"| {info['fp_baseline']} → {info['fp']} "
                f"| {info['n_failed_baseline']} → {info['n_failed']} "
                f"| {'YES' if info['regressed'] else 'no'} |"
            )
        lines.append(
            f"\n- Runtime (holdout, pooled median s/spectrum): "
            f"{_fmt(res['runtime_median_baseline_s'], '.2f')} → "
            f"{_fmt(res['runtime_median_s'], '.2f')} (gate ≤ {RUNTIME_FACTOR}×)"
        )
        if res["g3_reasons"]:
            lines.append(f"- G3 violations: {'; '.join(res['g3_reasons'])}")
        if res.get("params_diff"):
            lines.append("- Params (diff from baseline only):")
            for key, (base_val, cand_val) in sorted(res["params_diff"].items()):
                lines.append(f"  - `{key}`: {base_val} → {cand_val}")
        elif res.get("params") is not None:
            lines.append("- Params: identical to baseline.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Report path (default in study dir)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument(
        "--spectra-per-dataset",
        type=int,
        default=None,
        help="Cap per dataset (SMOKE ONLY; default = full splits)",
    )
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Override the holdout quota")
    args = parser.parse_args(argv)

    import cflibs

    print(f"cflibs={Path(cflibs.__file__).resolve().parent}")
    study_dir = Path(args.study_dir)
    cfg = driver.load_study_config(study_dir)
    manifest = splits.load_manifest(cfg["splits_manifest_path"])

    enforce_holdout_quota(study_dir, args.force)
    driver.append_ledger(
        study_dir,
        {
            "kind": "holdout_query",
            "top_k": args.top_k,
            "cpus": driver.detect_cpus(),
            "wall_s": 0.0,  # updated by trial entries below
        },
    )

    study = optuna.load_study(study_name=cfg["study_name"], storage=driver.study_storage(study_dir))
    candidates = top_k_trials(study, args.top_k)
    if not candidates:
        raise SystemExit("No completed trials in the study.")

    holdout_datasets = tuple(manifest["holdout"].keys())
    ctx_opt = objective_mod.EvalContext(
        db_path=Path(cfg["db_path"]),
        manifest=manifest,
        datasets=tuple(cfg["datasets"]),
        spectra_per_dataset=args.spectra_per_dataset,
        sample_seed=args.seed,
        preset=cfg.get("preset"),
        n_procs=args.n_procs,
        per_spectrum_timeout_s=cfg.get("per_spectrum_timeout_s", 120.0),
    )
    ctx_hold = objective_mod.EvalContext(
        db_path=Path(cfg["db_path"]),
        manifest=manifest,
        datasets=holdout_datasets,
        spectra_per_dataset=args.spectra_per_dataset,
        sample_seed=args.seed,
        preset=cfg.get("preset"),
        n_procs=args.n_procs,
        per_spectrum_timeout_s=cfg.get("per_spectrum_timeout_s", 120.0),
    )

    def _timed_eval(label: str, overrides, ctx, section: str, allow: bool):
        t0 = time.perf_counter()
        board = objective_mod.evaluate_overrides(
            overrides, ctx, section=section, datasets=ctx.datasets, allow_restricted=allow
        )
        wall = time.perf_counter() - t0
        driver.append_ledger(
            study_dir,
            {
                "kind": "holdout_eval",
                "label": label,
                "section": section,
                "wall_s": wall,
                "cpus": driver.detect_cpus(),
            },
        )
        return board

    baseline_params = knob_space.baseline_params()
    baseline_overrides = knob_space.params_to_overrides(baseline_params)
    print("Evaluating baseline on the optimization split ...")
    base_opt = _timed_eval("baseline", baseline_overrides, ctx_opt, "optimization", False)
    print("Evaluating baseline on the holdout split ...")
    base_hold = _timed_eval("baseline", baseline_overrides, ctx_hold, "holdout", True)

    results = []
    for trial in candidates:
        name = f"trial-{trial.number}"
        print(f"Evaluating {name} (study fitness {trial.value:.4f}) ...")
        overrides = knob_space.params_to_overrides(trial.params)
        cand_opt = _timed_eval(name, overrides, ctx_opt, "optimization", False)
        cand_hold = _timed_eval(name, overrides, ctx_hold, "holdout", True)
        res = _gate_results(
            name,
            cand_opt,
            cand_hold,
            base_opt,
            base_hold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        res["study_fitness"] = trial.value
        res["params"] = dict(trial.params)
        res["params_diff"] = {
            k: (baseline_params.get(k, "<unset>"), v)
            for k, v in trial.params.items()
            if baseline_params.get(k, "<unset>") != v
        }
        results.append(res)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "spectra_per_dataset": args.spectra_per_dataset,
    }
    report = render_report(study_dir, results, meta)
    out_path = Path(args.output) if args.output else study_dir / "holdout_verdict.md"
    out_path.write_text(report)
    (study_dir / "holdout_verdict.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, default=str)
    )
    print(f"\nWrote {out_path}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
