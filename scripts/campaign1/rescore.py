"""Offline FITNESS-V2 re-scoring of an existing campaign1 study journal.

Run1 was scored with fitness version 1 (flat death penalties): 79 of 80
trials landed on -1e9, several with a recorded ``weighted_score`` ABOVE the
surviving baseline. Every COMPLETE trial's journal record already carries its
``fitness_report`` (per-dataset FP/failure/score data) in ``user_attrs`` —
so the graded v2 fitness can be recomputed **offline, from the recorded
reports, without re-evaluating a single spectrum**
(:func:`objective.regrade_report_v2`).

Outputs (under ``--output-dir``, default ``<study-dir>/rescore_v2``):

- ``rescore_v2.json``   — every regraded trial + the grading constants
- ``rescore_v2.md``     — ranked markdown report
- ``warm_start_top{K}.json`` — top-K non-catastrophic param dicts for
  ``driver.py init --enqueue-from`` (run2 warm starts). Params that the
  CURRENT knob space no longer defines (``use_deconvolution`` was removed in
  c1-knobs-v2) are stripped here so the enqueued trials match the run2 space.

Caveat: trials whose v1 evaluation early-aborted (``aborted_after_dataset``)
recorded only a PREFIX of the datasets; their regraded fitness is an
estimate (flagged ``partial``) — the penalty is a lower bound and the
weighted score covers only the evaluated prefix.

Run from the repo root (the run1 journal lives on the cluster)::

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/campaign1/rescore.py \
        --study-dir /cluster/shared/cf-libs-bench/campaign1/results/run1 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import optuna

_CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(_CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGN_DIR))

import driver  # noqa: E402
import knob_space  # noqa: E402
import objective as objective_mod  # noqa: E402


def regrade_study_trials(
    study: optuna.Study,
    baseline_ref: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Regrade every COMPLETE trial; returns (ranked entries, skipped numbers).

    Ranking: regraded v2 fitness, descending (ties: earlier trial first).
    Trials without a recorded ``fitness_report`` (e.g. hand-inserted rows)
    cannot be regraded offline and are skipped.
    """
    entries: list[dict[str, Any]] = []
    skipped: list[int] = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        report = trial.user_attrs.get("fitness_report")
        if not report:
            skipped.append(trial.number)
            continue
        regraded = objective_mod.regrade_report_v2(report, baseline_ref)
        per_dataset = regraded["per_dataset"]
        entries.append(
            {
                "trial": trial.number,
                "params": dict(trial.params),
                "fitness_v1": trial.value,
                "fitness_v2": regraded["fitness"],
                "weighted_score": regraded["weighted_score"],
                "graded_total_penalty": regraded["graded_total_penalty"],
                "excess_fp_total": sum(r["excess_fp"] for r in per_dataset.values()),
                "excess_failed_total": sum(r["excess_failed"] for r in per_dataset.values()),
                "catastrophic": regraded["catastrophic"],
                "partial": regraded["partial"],
                "aborted_after_dataset": regraded["aborted_after_dataset"],
                "v1_death": regraded["v1_death"],
                "per_dataset": per_dataset,
            }
        )
    entries.sort(key=lambda e: (-e["fitness_v2"], e["trial"]))
    return entries, skipped


def select_warm_starts(entries: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Top-K non-catastrophic entries, deduplicated by params.

    Catastrophic trials are excluded (their region is what the floor is FOR);
    partial (v1-aborted) trials stay eligible — a candidate that scored well
    on its evaluated prefix before a graded near-miss is exactly the kind of
    point run1's flat -1e9 hid from TPE.
    """
    known = {knob.param for knob in knob_space.SPACE}
    picked: list[dict[str, Any]] = []
    seen: set = set()
    for entry in entries:
        if entry["catastrophic"]:
            continue
        params = {key: v for key, v in entry["params"].items() if key in known}
        stripped = sorted(set(entry["params"]) - set(params))
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        picked.append(
            {
                "trial": entry["trial"],
                "fitness_v2": entry["fitness_v2"],
                "partial": entry["partial"],
                "stripped_params": stripped,
                "params": params,
            }
        )
        if len(picked) >= k:
            break
    return picked


def render_markdown(
    study_dir: Path,
    entries: list[dict[str, Any]],
    warm_starts: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    lines = [
        "# Campaign 1 — FITNESS-V2 offline re-score",
        "",
        f"Generated: {meta['generated_utc']}  ",
        f"Study: `{study_dir}`  ",
        f"Trials regraded: {meta['n_regraded']} (skipped without reports: "
        f"{meta['n_skipped']})  ",
        f"Constants: LAMBDA_FP={objective_mod.LAMBDA_FP}, "
        f"LAMBDA_FAIL={objective_mod.LAMBDA_FAIL}, "
        f"CATASTROPHIC_PENALTY={objective_mod.CATASTROPHIC_PENALTY}, "
        f"CATASTROPHIC_FITNESS={objective_mod.CATASTROPHIC_FITNESS}",
        "",
        "Recomputed offline from each trial's recorded `fitness_report` — no "
        "re-evaluation. `partial` trials early-aborted under v1 and recorded "
        "only a dataset prefix: their v2 fitness is an estimate (penalty is a "
        "lower bound).",
        "",
        "| rank | trial | fitness v2 | fitness v1 | weighted | penalty "
        "| excess FP | excess fail | flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, e in enumerate(entries, start=1):
        flags = [
            name
            for name, on in (
                ("CATASTROPHIC", e["catastrophic"]),
                ("partial", e["partial"]),
                ("v1-death", e["v1_death"]),
            )
            if on
        ]
        lines.append(
            f"| {rank} | {e['trial']} | {e['fitness_v2']:.4f} | {e['fitness_v1']:.4g} "
            f"| {e['weighted_score']:.4f} | {e['graded_total_penalty']:.2f} "
            f"| {e['excess_fp_total']} | {e['excess_failed_total']} "
            f"| {', '.join(flags) or '-'} |"
        )
    lines += [
        "",
        f"## Warm starts (top-{meta['top_k']}, catastrophics excluded)",
        "",
        f"Written to `{meta['warm_start_filename']}` for "
        "`driver.py init --enqueue-from`. Params removed from the current "
        f"knob space ({knob_space.KNOB_SPACE_VERSION}) are stripped.",
        "",
    ]
    for ws in warm_starts:
        stripped = (
            f" (stripped: {', '.join(ws['stripped_params'])})" if ws["stripped_params"] else ""
        )
        partial = " [partial]" if ws["partial"] else ""
        lines.append(f"- trial {ws['trial']}: fitness_v2 {ws['fitness_v2']:.4f}{partial}{stripped}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10, help="Warm-start list size")
    parser.add_argument(
        "--output-dir", default=None, help="Artifact dir (default <study-dir>/rescore_v2)"
    )
    args = parser.parse_args(argv)

    import cflibs

    print(f"cflibs={Path(cflibs.__file__).resolve().parent}")
    study_dir = Path(args.study_dir)
    cfg = driver.load_study_config(study_dir)
    baseline_ref = json.loads((study_dir / driver.BASELINE_FILENAME).read_text())["reference"]
    study = optuna.load_study(study_name=cfg["study_name"], storage=driver.study_storage(study_dir))

    entries, skipped = regrade_study_trials(study, baseline_ref)
    if not entries:
        raise SystemExit("No COMPLETE trials with recorded fitness_report in the journal.")
    warm_starts = select_warm_starts(entries, args.top_k)

    out_dir = Path(args.output_dir) if args.output_dir else study_dir / "rescore_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_start_filename = f"warm_start_top{args.top_k}.json"
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_dir": str(study_dir),
        "study_name": cfg["study_name"],
        "study_fitness_version": cfg.get("fitness_version", 1),
        "knob_space_version": knob_space.KNOB_SPACE_VERSION,
        "top_k": args.top_k,
        "n_regraded": len(entries),
        "n_skipped": len(skipped),
        "skipped_trials": skipped,
        "warm_start_filename": warm_start_filename,
        "constants": {
            "lambda_fp": objective_mod.LAMBDA_FP,
            "lambda_fail": objective_mod.LAMBDA_FAIL,
            "catastrophic_penalty": objective_mod.CATASTROPHIC_PENALTY,
            "catastrophic_fitness": objective_mod.CATASTROPHIC_FITNESS,
        },
    }

    (out_dir / "rescore_v2.json").write_text(
        json.dumps({"meta": meta, "trials": entries}, indent=2, default=str)
    )
    report = render_markdown(study_dir, entries, warm_starts, meta)
    (out_dir / "rescore_v2.md").write_text(report)
    (out_dir / warm_start_filename).write_text(
        json.dumps(
            {
                "generated_utc": meta["generated_utc"],
                "source_study": str(study_dir),
                "knob_space_version": knob_space.KNOB_SPACE_VERSION,
                "fitness_version": 2,
                "top_k": args.top_k,
                "trials": [ws["trial"] for ws in warm_starts],
                "params": [ws["params"] for ws in warm_starts],
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir / 'rescore_v2.md'}")
    print(f"Wrote {out_dir / 'rescore_v2.json'}")
    print(f"Wrote {out_dir / warm_start_filename} ({len(warm_starts)} warm starts)")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
