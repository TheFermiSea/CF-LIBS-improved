#!/usr/bin/env bash
# loop_24h_driver.sh — Run N iterations of the analysis pipeline as fast as
# possible within a 24-hour wall budget.
#
# This script runs on the benchmark host (vasp-03) and dispatches iterations
# back-to-back via run_unified_benchmark.py. It maintains a manifest of
# completed iterations in output/loop-<DATE>/manifest.jsonl (one line per
# iteration).
#
# Usage:
#   ./scripts/loop_24h_driver.sh [run_date] [n_iter] [vrabel_cap] [platform]
#
# Args:
#   run_date    Date prefix (default: today YYYY-MM-DD)
#   n_iter      Target iteration count (default: 24)
#   vrabel_cap  Vrabel shots/sample cap; 0 = full 50k (default: 0)
#   platform    JAX_PLATFORMS value (default: cpu; cuda for GPU)
#
# Behavior:
#   - Honors a hard 24-hour wall-clock budget. Stops accepting new iterations
#     after 86400 s elapsed, regardless of how many completed.
#   - Each iteration has its own timeout (default 5400s = 90min). Iterations
#     that hit timeout are marked failed in the manifest but the loop continues.
#   - Each iteration writes to output/loop-<DATE>/iter-NNN/.
#   - Resumable: skips iter-NNN if output/loop-<DATE>/iter-NNN/_done.json exists.

set -euo pipefail

RUN_DATE="${1:-$(date +%Y-%m-%d)}"
N_ITER="${2:-24}"
VRABEL_CAP="${3:-0}"
PLATFORM="${4:-cpu}"

ROOT="/scratch/cf-libs-bench/repo"
# T1.3: data dir defaults to NFS canonical path; override via CFLIBS_DATA_DIR.
# See docs/nfs-shared-data.md.
DATA_DIR="${CFLIBS_DATA_DIR:-/cluster/shared/cf-libs-bench/data}"
if [ ! -d "$DATA_DIR" ]; then
  echo "[loop_24h_driver] WARN: data dir $DATA_DIR not found; falling back to $ROOT/data" >&2
  DATA_DIR="$ROOT/data"
fi
export CFLIBS_DATA_DIR="$DATA_DIR"
OUT_BASE="$ROOT/output/loop-$RUN_DATE"
MANIFEST="$OUT_BASE/manifest.jsonl"
WALL_BUDGET_SEC=86400  # 24 hours
PER_ITER_TIMEOUT=5400  # 90 min
START_EPOCH=$(date +%s)

mkdir -p "$OUT_BASE"
touch "$MANIFEST"

cd "$ROOT"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT_BASE/driver.log"; }
log "=== loop_24h_driver start ==="
log "RUN_DATE=$RUN_DATE N_ITER=$N_ITER VRABEL_CAP=$VRABEL_CAP PLATFORM=$PLATFORM"
log "WALL_BUDGET=${WALL_BUDGET_SEC}s PER_ITER_TIMEOUT=${PER_ITER_TIMEOUT}s"
log "SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

completed=0
failed=0
for i in $(seq 1 "$N_ITER"); do
  elapsed=$(( $(date +%s) - START_EPOCH ))
  if [ "$elapsed" -ge "$WALL_BUDGET_SEC" ]; then
    log "WALL BUDGET EXCEEDED (${elapsed}s >= ${WALL_BUDGET_SEC}s) — stopping at iter $((i-1)) / $N_ITER"
    break
  fi

  iter_padded=$(printf "%03d" "$i")
  iter_dir="$OUT_BASE/iter-$iter_padded"
  done_marker="$iter_dir/_done.json"
  log_file="$OUT_BASE/iter-$iter_padded.log"

  if [ -f "$done_marker" ]; then
    log "iter-$iter_padded already done, skipping"
    completed=$((completed + 1))
    continue
  fi

  mkdir -p "$iter_dir"
  iter_start=$(date +%s)
  log "iter-$iter_padded start (wall_elapsed=${elapsed}s)"

  set +e
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib \
  JAX_PLATFORMS="$PLATFORM" \
  JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    timeout "$PER_ITER_TIMEOUT" .venv/bin/python scripts/run_unified_benchmark.py \
    --quick --max-outer-folds 1 --sections all \
    --id-workflows alias comb \
    --composition-workflows bayesian iterative_jax \
    --db-path ASD_da/libs_production.db \
    --data-dir "$DATA_DIR" \
    --vrabel-max-shots "$VRABEL_CAP" \
    --output-dir "output/loop-$RUN_DATE/iter-$iter_padded" \
    > "$log_file" 2>&1
  exit_code=$?
  set -e

  iter_end=$(date +%s)
  iter_dur=$((iter_end - iter_start))

  if [ "$exit_code" -eq 0 ]; then
    completed=$((completed + 1))
    cat > "$done_marker" <<EOF
{"iter": $i, "exit_code": 0, "duration_sec": $iter_dur, "completed_at": "$(date -Iseconds)", "wall_elapsed_sec": $((iter_end - START_EPOCH))}
EOF
    log "iter-$iter_padded OK in ${iter_dur}s (completed=$completed, failed=$failed)"
  else
    failed=$((failed + 1))
    cat > "$iter_dir/_failed.json" <<EOF
{"iter": $i, "exit_code": $exit_code, "duration_sec": $iter_dur, "completed_at": "$(date -Iseconds)", "wall_elapsed_sec": $((iter_end - START_EPOCH))}
EOF
    log "iter-$iter_padded FAIL exit=$exit_code in ${iter_dur}s (completed=$completed, failed=$failed)"
  fi

  echo "{\"iter\": $i, \"exit_code\": $exit_code, \"duration_sec\": $iter_dur, \"end\": \"$(date -Iseconds)\"}" >> "$MANIFEST"
done

total_elapsed=$(( $(date +%s) - START_EPOCH ))
log "=== loop_24h_driver done ==="
log "completed=$completed failed=$failed total_elapsed=${total_elapsed}s"
log "manifest=$MANIFEST"
