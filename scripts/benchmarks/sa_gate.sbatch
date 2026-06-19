#!/bin/bash
# Cluster benchmark-gate of self-absorption "observable" mode (the #1 CF-LIBS
# accuracy lever) on the composition-truth datasets, whose data lives only on
# the staged cluster repo. apply_self_absorption is PRE-EXISTING in the staged
# repo (not new session code), so NO re-stage is needed. One array task per
# dataset; each runs the scoreboard baseline (default) vs observable-SA and
# dumps both board JSONs for the parent to diff.
#
# Submit (after the full board frees the cluster):
#   sbatch --export=ALL,REPO_DIR=/cluster/shared/cf-libs-bench/jitpipe-m3/repo \
#       scripts/benchmarks/sa_gate.sbatch
#
#SBATCH --job-name=cflibs-sagate
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/shared/cf-libs-bench/jitpipe-m3/logs/sagate/sa-%a.out

set -euo pipefail
: "${REPO_DIR:?export REPO_DIR=<staged repo>}"

DATASETS=(chemcam_calib csa_planetary)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
OUT=/cluster/shared/cf-libs-bench/jitpipe-m3/sa_gate
mkdir -p "$OUT" /cluster/shared/cf-libs-bench/jitpipe-m3/logs/sagate

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
cd "$REPO_DIR"
echo "node=$(hostname) dataset=$DS time=$(date -Is)"

exec ./.venv/bin/python -u - "$DS" "$OUT" <<'PY'
import json, sys
from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard

ds, out = sys.argv[1], sys.argv[2]
db = AtomicDatabase("ASD_da/libs_production.db")
common = dict(datasets=[ds], max_spectra=100, seed=20260610)
print(f"[sa_gate] {ds} baseline ...", flush=True)
base = run_scoreboard(db, config_overrides=None, **common)
json.dump(base, open(f"{out}/{ds}_base.json", "w"), indent=2, default=str)
print(f"[sa_gate] {ds} observable-SA ...", flush=True)
sa = run_scoreboard(db, config_overrides={"apply_self_absorption": "observable"}, **common)
json.dump(sa, open(f"{out}/{ds}_sa.json", "w"), indent=2, default=str)

def row(b):
    d = b["datasets"][0] if b.get("datasets") else {}
    f1 = (d.get("id_metrics") or {}).get("f1")
    comp = d.get("composition") or {}
    return f1, comp.get("rmse_wt_median"), d.get("n_failed")

bf1, brmse, bfail = row(base)
sf1, srmse, sfail = row(sa)
print(f"[sa_gate] {ds}: F1 {bf1}->{sf1}  RMSE {brmse}->{srmse}  fail {bfail}->{sfail}", flush=True)
PY
