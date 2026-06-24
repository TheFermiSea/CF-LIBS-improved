#!/usr/bin/env python
"""Arbor evaluation script + correctness gate for CF-LIBS algorithm optimization.

Arbor (keyless mode, driven by the host Claude Code agent) runs THIS per candidate in an isolated
git worktree, then merges only if the held-out TEST score clears its margin. Arbor's built-in gate
is ACCURACY-ONLY, so correctness is enforced HERE: any candidate that fails the cflibs-formal oracle
conformance OR the physics-only import blocklist returns score = -1e9 (sentinel) and never merges --
the mechanism that auto-catches C-sigma-class silent errors (a candidate that scores well but is
physically wrong). Mirrors cflibs/evolution/evaluator.py's "fitness = -inf on violation".

Splits use the project's OWN anti-overfit tiers (no leakage):
  --split dev   -> optimization tier (supercam_labcal + aalto)   [iterate here]
  --split test  -> holdout      tier (supercam_scct + bhvo2_chemcam) [Arbor's merge gate]

Metric: median composition RMSE (wt%) from the production run_scoreboard. Reported as
score = -RMSE (HIGHER is better, so Arbor maximizes it). Emits a metrics JSON to stdout and --out.

Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/arbor/run_eval.py --split dev [--max-spectra 20] [--out metrics.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SENTINEL_BAD = -1.0e9

# datasets-with-local-data per tier (the public-data-gap datasets are skipped automatically)
SPLIT_DATASETS = {
    "dev": (["supercam_labcal", "aalto"], False),  # (datasets, include_holdout)
    "test": (["supercam_scct", "bhvo2_chemcam"], True),
}

# correctness gate: physics-only blocklist (ruff TID251) + cflibs-formal oracle conformance.
GATE_TESTS = [
    "tests/oracle/",
    "tests/inversion/physics/test_csigma_composition.py",
    "tests/inversion/physics/test_reliability.py",
    "tests/inversion/physics/test_identifiability.py",
]


def _bindir() -> str:
    return os.path.dirname(sys.executable)


#: Main checkout that holds the UNTRACKED data dirs (e.g. data/supercam_calib). Arbor worktrees
#: only check out git-TRACKED files, so the eval must symlink the rest in. Override via env.
DATA_SRC = os.environ.get(
    "CFLIBS_ARBOR_DATA_SRC", "/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5"
)


def provision_untracked_data() -> None:
    """Symlink any data/ subdir present in the main checkout but missing here (idempotent).

    A fresh Arbor worktree has only the tracked datasets (aalto, bhvo2); the SuperCam data lives in
    the UNTRACKED data/supercam_calib, so supercam_labcal/supercam_scct silently skip and the eval
    returns a sentinel. This fills the gaps so candidates are actually scored on the real splits."""
    src = os.path.join(DATA_SRC, "data")
    if not os.path.isdir(src) or os.path.realpath(os.getcwd()) == os.path.realpath(DATA_SRC):
        return
    os.makedirs("data", exist_ok=True)
    for name in os.listdir(src):
        dst = os.path.join("data", name)
        if not os.path.exists(dst):  # only fill gaps; tracked dirs are already checked out
            try:
                os.symlink(os.path.join(src, name), dst)
            except OSError:
                pass


def correctness_gate() -> tuple[bool, str]:
    """Physics-only import blocklist + verified-spec oracle conformance. (ok, reason)."""
    ruff = os.path.join(_bindir(), "ruff")
    blocklist = subprocess.run(
        [ruff, "check", "--select", "TID251", "cflibs/"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    if blocklist.returncode != 0:
        return False, "physics-only blocklist (TID251) violation:\n" + blocklist.stdout[-1500:]

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        *GATE_TESTS,
        "-q",
        "-p",
        "no:cacheprovider",
        "-m",
        "not requires_db",
        "--timeout=180",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=os.getcwd())
    res = subprocess.run(pytest_cmd, capture_output=True, text=True, env=env, cwd=os.getcwd())
    if res.returncode != 0:
        tail = (res.stdout + res.stderr)[-1500:]
        return False, "cflibs-formal oracle conformance FAILED:\n" + tail
    return True, "ok"


def score_split(
    split: str, max_spectra: int, db_path: str = "ASD_da/libs_production.db"
) -> tuple[float, dict]:
    """Median composition RMSE (wt%) over the split's datasets via run_scoreboard."""
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import ensure_default_datasets, run_scoreboard

    provision_untracked_data()  # fill in untracked data (supercam_calib) for fresh Arbor worktrees
    ensure_default_datasets()
    db = AtomicDatabase(db_path)
    datasets, include_holdout = SPLIT_DATASETS[split]
    board = run_scoreboard(
        db,
        datasets=datasets,
        max_spectra=max_spectra,
        seed=7,
        pipeline_impl="reference",
        include_holdout=include_holdout,
    )
    per = {}
    for d in board.get("datasets", []):
        m = (d.get("composition") or {}).get("rmse_wt_median")
        if m is not None:
            per[d["name"]] = float(m)
    if not per:
        return float("nan"), {}
    # aggregate across datasets: mean of per-dataset medians (equal dataset weight)
    return sum(per.values()) / len(per), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=list(SPLIT_DATASETS), default="dev")
    ap.add_argument("--max-spectra", type=int, default=20)
    ap.add_argument("--db", default="ASD_da/libs_production.db", help="atomic database to evaluate")
    ap.add_argument("--out", default="arbor_metrics.json")
    args = ap.parse_args()

    ok, reason = correctness_gate()
    if not ok:
        metrics = {
            "valid": False,
            "score": SENTINEL_BAD,
            "rmse_wt": None,
            "split": args.split,
            "reason": reason,
        }
        print(json.dumps(metrics))
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        return  # exit 0: Arbor reads the sentinel score and refuses the merge

    rmse, per = score_split(args.split, args.max_spectra, args.db)
    # score = 100/RMSE: HIGHER is better, and Arbor's "% gain" merge threshold then maps ~1:1 to
    # the RELATIVE RMSE reduction (e.g. 3.33 -> 3.23 wt% is a ~3% score gain).
    score = 100.0 / rmse if (rmse and rmse > 0 and rmse == rmse) else SENTINEL_BAD
    metrics = {
        "valid": True,
        "score": score,
        "db": args.db,
        "rmse_wt": rmse,
        "per_dataset_rmse_wt": per,
        "split": args.split,
        "max_spectra": args.max_spectra,
        "reason": "ok",
    }
    print(json.dumps(metrics))
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
