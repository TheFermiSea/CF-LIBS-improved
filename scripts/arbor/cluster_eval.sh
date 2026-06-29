#!/bin/bash
# Arbor cluster-eval bridge: stage a candidate worktree to the NFS and run the FULL run_eval.py
# (correctness gate + score) on a vasp compute node via srun, printing the score JSON so Arbor's
# eval_run can extract it. The candidate's CODE is staged per-candidate; data/DBs/venv are symlinked
# from the base stage (no re-copy). Run from ai-proxy.
#
#   scripts/arbor/cluster_eval.sh <candidate_worktree> <dev|test> [id]
set -euo pipefail

CAND_WT="${1:?candidate worktree path}"
SPLIT="${2:?dev|test}"
ID="${3:-$(basename "$CAND_WT" | tr -c 'A-Za-z0-9_-' '_')}"

BASE_NODE=/cluster/shared/ai/cf-libs-bench/dbbench/repo      # data/DBs/portable .venv (node path)
CAND_WRITE="/mnt/nfs/shared/ai/cf-libs-bench/arbor/$ID/repo" # my write path (ai-proxy)
CAND_NODE="/cluster/shared/ai/cf-libs-bench/arbor/$ID/repo"  # same dir, node path

mkdir -p "$CAND_WRITE"
# Stage candidate CODE only (cflibs/tests/scripts/pyproject); base-shared heavy dirs excluded.
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude 'output' --exclude 'data' --exclude 'ASD_da' \
  --exclude '.worktrees' --exclude '__pycache__' --exclude '.arbor' "$CAND_WT/" "$CAND_WRITE/"
# Symlink shared data/DBs/venv from the base stage (node-path targets so they resolve on the node).
for L in ASD_da output data .venv; do ln -sfn "$BASE_NODE/$L" "$CAND_WRITE/$L"; done

OUT="/tmp/arbor_${ID}_${SPLIT}.json"
# Full eval (gate + score) on a compute node against the candidate's code (PYTHONPATH wins over venv).
srun --partition=normal -N1 -n1 --cpus-per-task=8 --time=00:40:00 bash -c "
  unset LD_LIBRARY_PATH
  cd '$CAND_NODE'
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=\$PWD .venv/bin/python -u \
    scripts/arbor/run_eval.py --split '$SPLIT' --max-spectra 20 --out '$OUT'
  cat '$OUT'
" 2>/dev/null | grep -E '"score"' | tail -1
