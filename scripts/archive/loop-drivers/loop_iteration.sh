#!/usr/bin/env bash
# loop_iteration.sh — Run one iteration of the 24-hour analysis-pipeline loop.
#
# Each iteration runs the unified benchmark over the full Vrabel + BHVO-2
# datasets at the configured shot cap, writing results to
# output/loop-<DATE>/iter-<N>/.
#
# Usage:
#   ./scripts/loop_iteration.sh <iter_n> [<run_date>] [<vrabel_cap>]
#
# Args:
#   iter_n       Iteration number (1..24)
#   run_date     Date prefix for the output directory (default: today YYYY-MM-DD)
#   vrabel_cap   Vrabel shots/sample cap (0 = full 50k, default 0)
#
# Environment:
#   BENCH_HOST       SSH target for the run (default: root@10.0.0.22)
#   BENCH_REPO_PATH  NFS repo path on remote (default: /scratch/cf-libs-bench/repo)
#   BENCH_JAX_PLATFORMS  JAX platforms env (default: cpu; cuda for GPU)
#   BENCH_TIMEOUT_SEC    Per-iteration timeout (default: 7200 = 2hr)
#   BENCH_DATA_DIR   Dataset dir on remote (default:
#                    /cluster/shared/cf-libs-bench/data, NFS-shared; T1.3).
#                    See docs/nfs-shared-data.md.

set -euo pipefail

ITER_N="${1:?usage: loop_iteration.sh <iter_n> [run_date] [vrabel_cap]}"
RUN_DATE="${2:-$(date +%Y-%m-%d)}"
VRABEL_CAP="${3:-0}"
BENCH_HOST="${BENCH_HOST:-root@10.0.0.22}"
BENCH_REPO_PATH="${BENCH_REPO_PATH:-/scratch/cf-libs-bench/repo}"
BENCH_JAX_PLATFORMS="${BENCH_JAX_PLATFORMS:-cpu}"
BENCH_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC:-7200}"
BENCH_DATA_DIR="${BENCH_DATA_DIR:-/cluster/shared/cf-libs-bench/data}"

ITER_PADDED=$(printf "%03d" "$ITER_N")
SEED=$ITER_N
OUTPUT_REL="output/loop-${RUN_DATE}/iter-${ITER_PADDED}"
LOG_REL="output/loop-${RUN_DATE}/iter-${ITER_PADDED}.log"

echo "[loop] iter=$ITER_N date=$RUN_DATE cap=$VRABEL_CAP host=$BENCH_HOST output=$OUTPUT_REL"

ssh "$BENCH_HOST" bash <<REMOTE
set -euo pipefail
cd "$BENCH_REPO_PATH"
git fetch origin dev --quiet
git reset --hard origin/dev --quiet
SHA=\$(git rev-parse --short HEAD)
echo "[remote] sha=\$SHA cap=$VRABEL_CAP seed=$SEED platform=$BENCH_JAX_PLATFORMS data_dir=$BENCH_DATA_DIR"
# T1.3: prefer NFS-shared data dir; fall back to repo-local 'data' if NFS is unavailable.
DATA_DIR="$BENCH_DATA_DIR"
if [ ! -d "\$DATA_DIR" ]; then
  echo "[remote] WARN: \$DATA_DIR not found; falling back to repo-local 'data'" >&2
  DATA_DIR="data"
fi
export CFLIBS_DATA_DIR="\$DATA_DIR"
mkdir -p "$OUTPUT_REL"
JAX_PLATFORMS=$BENCH_JAX_PLATFORMS JAX_ENABLE_X64=1 \
  timeout $BENCH_TIMEOUT_SEC .venv/bin/python scripts/run_unified_benchmark.py \
  --quick --max-outer-folds 1 --sections all \
  --id-workflows alias comb \
  --composition-workflows iterative_jax \
  --db-path ASD_da/libs_production.db \
  --data-dir "\$DATA_DIR" \
  --vrabel-max-shots $VRABEL_CAP \
  --output-dir "$OUTPUT_REL" \
  > "$LOG_REL" 2>&1
EXIT=\$?
echo "[remote] iter=$ITER_N exit=\$EXIT sha=\$SHA data_dir=\$DATA_DIR"
exit \$EXIT
REMOTE
