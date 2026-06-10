#!/bin/bash
# Stage the repo + atomic DB + datasets to the shared cluster filesystem and
# bootstrap a venv there. Run FROM a checkout of the campaign branch on a
# machine that can see both the data symlink targets and the cluster mount
# (or via ssh+rsync to vasp-01 — adapt DEST accordingly).
#
# Assumptions (documented per task spec):
#   - DEST default /cluster/shared/cf-libs-bench/campaign1 (the share already
#     holds prior data/ and results/ dirs from earlier benchmark waves;
#     rsync is incremental, re-running is cheap).
#   - `uv` is on PATH on the machine that runs the bootstrap step and can
#     provision Python 3.12 (`uv python install 3.12` if needed).
#   - CPU-only: install extra ".[dev]" + optuna; no CUDA wheels needed for
#     Campaign 1 (use ".[cluster]" instead if the same venv must also serve
#     GPU manifold jobs).
#
# Usage:
#   bash scripts/campaign1/slurm/stage.sh [DEST]
set -euo pipefail

DEST="${1:-/cluster/shared/cf-libs-bench/campaign1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "Staging $REPO -> $DEST"
mkdir -p "$DEST"/{repo,data,results}

# 1. Repo (no .git/.venv/output; data/ and ASD_da/ staged separately below).
rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude 'output' --exclude '.worktrees' \
    --exclude 'data' --exclude 'ASD_da' --exclude '__pycache__' \
    "$REPO/" "$DEST/repo/"

# 2. Atomic DB (frozen input: its sha256 is pinned in every study's
#    frozen_manifest.json — re-staging a different DB obsoletes in-flight runs).
mkdir -p "$DEST/repo/ASD_da"
rsync -a "$REPO/ASD_da/libs_production.db" "$DEST/repo/ASD_da/libs_production.db"

# 3. Datasets: resolve the repo's data/ symlinks into real files on the share,
#    then point the staged repo's data/ at the shared copy.
rsync -aL "$REPO/data/" "$DEST/data/"
rm -rf "$DEST/repo/data"
ln -sfn "$DEST/data" "$DEST/repo/data"

# 4. Synthetic corpus (lives outside data/ locally; the adapter finds it via
#    $CFLIBS_SCOREBOARD_SYNTH_CORPUS, exported by the sbatch templates).
SYNTH_CORPUS="${CFLIBS_SCOREBOARD_SYNTH_CORPUS:-$REPO/output/synthetic_corpus_w2/w2_fixedforward_v1/corpus.json}"
if [[ ! -f "$SYNTH_CORPUS" ]]; then
    SYNTH_CORPUS="/home/brian/code/CF-LIBS-improved/.worktrees/w1-integration/output/synthetic_corpus_w2/w2_fixedforward_v1/corpus.json"
fi
if [[ -f "$SYNTH_CORPUS" ]]; then
    mkdir -p "$DEST/data/synthetic_corpus"
    rsync -aL "$SYNTH_CORPUS" "$DEST/data/synthetic_corpus/corpus.json"
else
    echo "WARNING: synthetic corpus not found ($SYNTH_CORPUS); synthetic_fixedforward will skip."
fi

# 5. Bootstrap the venv on the share (idempotent).
cd "$DEST/repo"
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH — install uv or bootstrap the venv manually:" >&2
    echo "  python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]' optuna" >&2
    exit 1
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]" optuna

echo "Staged. Next:"
echo "  cd $DEST/repo"
echo "  JAX_PLATFORMS=cpu PYTHONPATH=\$PWD .venv/bin/python scripts/campaign1/driver.py init \\"
echo "      --study-dir $DEST/results/run1 --db ASD_da/libs_production.db"
echo "  sbatch --export=ALL,REPO_DIR=$DEST/repo,STUDY_DIR=$DEST/results/run1,TRIALS_PER_TASK=5 \\"
echo "      scripts/campaign1/slurm/worker_array.sbatch"
