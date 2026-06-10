#!/bin/bash
# Campaign 1 holdout evaluation (adoption gate). ONE job, run manually after
# the study plateaus — holdout queries are quota-limited (<=1/phase/week,
# ledger-enforced; design 2.1). Re-evaluates the top-K trials + baseline on
# the FULL optimization split and the holdout set, and writes the verdict
# report into the study directory.
#
# Submit:
#   sbatch --export=ALL,REPO_DIR=...,STUDY_DIR=...,TOP_K=5 \
#       scripts/campaign1/slurm/holdout_eval.sbatch
#SBATCH --job-name=cflibs-c1-holdout
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=cflibs-c1-holdout-%j.out

set -euo pipefail

: "${REPO_DIR:?export REPO_DIR=<staged repo root>}"
: "${STUDY_DIR:?export STUDY_DIR=<study directory>}"
TOP_K="${TOP_K:-5}"

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-$HOME/.cache}/jax-cflibs-c1"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
if [[ -f "$REPO_DIR/data/synthetic_corpus/corpus.json" ]]; then
    export CFLIBS_SCOREBOARD_SYNTH_CORPUS="$REPO_DIR/data/synthetic_corpus/corpus.json"
fi

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR" exec "$REPO_DIR/.venv/bin/python" scripts/campaign1/holdout_eval.py \
    --study-dir "$STUDY_DIR" \
    --top-k "$TOP_K" \
    --n-procs "${SLURM_CPUS_PER_TASK:-8}"
