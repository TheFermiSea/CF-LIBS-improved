#!/bin/bash
# Campaign 1 worker array: each task self-serves Optuna trials from the shared
# journal until STOP file / core-hour ledger cap / study trial target /
# TRIALS_PER_TASK quota. Resubmit waves of this array until `driver.py status`
# reports the target reached (design 3.3 topology: 16-way, CPU-only).
#
# Submit (parameterized via --export; never edit the file per run):
#   sbatch --export=ALL,REPO_DIR=/cluster/shared/cf-libs-bench/campaign1/repo,STUDY_DIR=/cluster/shared/cf-libs-bench/campaign1/results/run1,TRIALS_PER_TASK=5 \
#       scripts/campaign1/slurm/worker_array.sbatch
#
# NO --nodelist: let SLURM schedule (the vasp cluster is shared with other
# agents). CPU-only: Campaign 1 never touches the V100S GPUs.
#SBATCH --job-name=cflibs-c1-worker
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cflibs-c1-worker-%A_%a.out

set -euo pipefail

: "${REPO_DIR:?export REPO_DIR=<staged repo root>}"
: "${STUDY_DIR:?export STUDY_DIR=<study directory (driver.py init already run)>}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-5}"

# CPU-only JAX with a PRIVATE compilation cache (never the shared one).
# $HOME, not $SLURM_TMPDIR (eff#7): SLURM_TMPDIR is wiped per job, so every
# array task recompiled XLA from scratch; a persistent per-user cache pays
# the compile cost once across all waves.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-cflibs-c1"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Synthetic corpus (staged by stage.sh; adapter skips-with-log if absent).
if [[ -f "$REPO_DIR/data/synthetic_corpus/corpus.json" ]]; then
    export CFLIBS_SCOREBOARD_SYNTH_CORPUS="$REPO_DIR/data/synthetic_corpus/corpus.json"
fi

# Kill switch + ledger live in STUDY_DIR (journal storage is NFS-safe via
# JournalFileOpenLock). Workers exit cleanly on STOP or when the core-hour
# ledger reaches the study's --budget-core-hours cap.
if [[ -f "$STUDY_DIR/STOP" ]]; then
    echo "STOP file present at $STUDY_DIR/STOP — not starting." && exit 0
fi

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR" exec "$REPO_DIR/.venv/bin/python" scripts/campaign1/driver.py worker \
    --study-dir "$STUDY_DIR" \
    --max-trials "$TRIALS_PER_TASK" \
    --worker-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-procs "${SLURM_CPUS_PER_TASK:-8}" \
    --cpus "${SLURM_CPUS_PER_TASK:-8}"
