#!/bin/bash
# Full adoption gate for self-absorption "observable" mode across ALL board
# datasets (incl. holdout — this is the legitimate adoption-gate use of the
# holdout tier per ADR-0004 §8.2, one-shot). One array task per dataset; each
# runs scoreboard baseline (default) vs observable-SA and prints F1/RMSE/fail
# deltas. apply_self_absorption is pre-existing in the staged repo (no re-stage).
#
#   sbatch --export=ALL,REPO_DIR=/cluster/shared/cf-libs-bench/jitpipe-m3/repo \
#       scripts/benchmarks/sa_gate_full.sbatch
#
#SBATCH --job-name=cflibs-saf
#SBATCH --array=0-8
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/shared/cf-libs-bench/jitpipe-m3/logs/sagate/saf-%a.out

set -euo pipefail
: "${REPO_DIR:?export REPO_DIR=<staged repo>}"

DATASETS=(aalto csa_planetary chemcam_calib silva2022 synthetic_fixedforward \
          bhvo2_chemcam emslibs2019 supercam_labcal supercam_scct)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
OUT=/cluster/shared/cf-libs-bench/jitpipe-m3/sa_gate_full
mkdir -p "$OUT"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
cd "$REPO_DIR"
echo "node=$(hostname) dataset=$DS time=$(date -Is)"

exec ./.venv/bin/python -u - "$DS" "$OUT" <<'PY'
import json, sys
from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard

ds, out = sys.argv[1], sys.argv[2]
db = AtomicDatabase("ASD_da/libs_production.db")
common = dict(datasets=[ds], max_spectra=100, seed=20260610, include_holdout=True)
print(f"[saf] {ds} baseline ...", flush=True)
base = run_scoreboard(db, config_overrides=None, **common)
json.dump(base, open(f"{out}/{ds}_base.json", "w"), indent=2, default=str)
print(f"[saf] {ds} observable-SA ...", flush=True)
sa = run_scoreboard(db, config_overrides={"apply_self_absorption": "observable"}, **common)
json.dump(sa, open(f"{out}/{ds}_sa.json", "w"), indent=2, default=str)

def row(b):
    d = b["datasets"][0] if b.get("datasets") else {}
    f1 = (d.get("id_metrics") or {}).get("f1")
    comp = d.get("composition") or {}
    return f1, comp.get("rmse_wt_median"), d.get("n_failed")

bf1, brmse, bfail = row(base)
sf1, srmse, sfail = row(sa)
print(f"[saf] RESULT {ds}: F1 {bf1}->{sf1}  RMSE {brmse}->{srmse}  fail {bfail}->{sfail}", flush=True)
PY
