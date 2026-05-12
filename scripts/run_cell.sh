#!/usr/bin/env bash
# run_cell.sh — run N iterations of one cell in the 6-cell parallel experiment.
#
# Usage:
#   ./scripts/run_cell.sh <cell_id> <identifier> <platform> [n_iter] [vrabel_cap]
#
# Args:
#   cell_id      e.g. C1, C2, ... (used in output path)
#   identifier   alias | comb | nnls | hybrid | correlation (one only)
#   platform     cuda | cpu
#   n_iter       default 8
#   vrabel_cap   default 10
#
# Environment expected on the executing node:
#   /scratch/cf-libs-bench/repo (the working tree)
#   .venv/bin/python (Python with JAX 0.9.2 + CUDA plugins)
#   /usr/local/cuda/lib64 (system CUDA libs)
#   /cluster/shared/cf-libs-bench/data (NFS-shared dataset directory; T1.3)
#
# Data directory selection:
#   Defaults to $CFLIBS_DATA_DIR if set, otherwise the NFS-shared
#   /cluster/shared/cf-libs-bench/data. Override by exporting
#   CFLIBS_DATA_DIR to point at a per-node /scratch copy or a custom
#   subset for offline replays. See docs/nfs-shared-data.md.
#
# Writes to: output/loop-2026-05-12/${cell_id}/iter-NNN/
# Manifest: output/loop-2026-05-12/${cell_id}/manifest.jsonl

set -euo pipefail

CELL="${1:?usage: run_cell.sh <cell_id> <identifier> <platform> [n_iter] [vrabel_cap]}"
IDENTIFIER="${2:?identifier required}"
PLATFORM="${3:?platform required}"
N_ITER="${4:-8}"
VRABEL_CAP="${5:-10}"
PER_ITER_TIMEOUT=1800   # 30 min

ROOT="/scratch/cf-libs-bench/repo"
RUN_DATE="2026-05-12"
# Data dir: prefer caller-supplied CFLIBS_DATA_DIR; otherwise use the
# NFS-shared canonical path (T1.3). Falls back to the in-tree "data"
# only if neither the env var nor the NFS path is available.
DATA_DIR="${CFLIBS_DATA_DIR:-/cluster/shared/cf-libs-bench/data}"
if [ ! -d "$DATA_DIR" ]; then
  echo "[run_cell] WARN: data dir $DATA_DIR not found; falling back to repo-local 'data'" >&2
  DATA_DIR="$ROOT/data"
fi
export CFLIBS_DATA_DIR="$DATA_DIR"
OUT_BASE="$ROOT/output/loop-$RUN_DATE/$CELL"
MANIFEST="$OUT_BASE/manifest.jsonl"
DRIVER_LOG="$OUT_BASE/driver.log"

cd "$ROOT"
mkdir -p "$OUT_BASE"
touch "$MANIFEST"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$DRIVER_LOG"; }
log "=== cell $CELL start: identifier=$IDENTIFIER platform=$PLATFORM n_iter=$N_ITER cap=$VRABEL_CAP ==="

# Compose LD_LIBRARY_PATH from venv-bundled CUDA + system CUDA.
NV_LIBS=$(find "$ROOT/.venv/lib/python3.11/site-packages/nvidia" -maxdepth 3 -name "lib" -type d 2>/dev/null | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${NV_LIBS}:/usr/local/cuda/lib64:/usr/local/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CPU thread allocation: each node runs 1 GPU + 1 CPU cell concurrently with
# 36 cores. GPU cell needs ~8 cores for data marshalling; CPU cell gets ~20
# threads; reserve 8 for system overhead.
if [ "$PLATFORM" = "cpu" ]; then
  export OMP_NUM_THREADS=20
  export OPENBLAS_NUM_THREADS=20
  export MKL_NUM_THREADS=20
  export NUMEXPR_NUM_THREADS=20
else
  export OMP_NUM_THREADS=8
  export OPENBLAS_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export NUMEXPR_NUM_THREADS=8
fi

COMPLETED=0
FAILED=0
START=$(date +%s)

for i in $(seq 1 "$N_ITER"); do
  ITER_PADDED=$(printf "%03d" "$i")
  ITER_DIR="$OUT_BASE/iter-$ITER_PADDED"
  LOG_FILE="$OUT_BASE/iter-$ITER_PADDED.log"
  if [ -f "$ITER_DIR/_done.json" ]; then
    COMPLETED=$((COMPLETED + 1))
    log "iter-$ITER_PADDED already done, skipping"
    continue
  fi
  mkdir -p "$ITER_DIR"
  iter_start=$(date +%s)
  log "iter-$ITER_PADDED start"

  set +e
  JAX_PLATFORMS="$PLATFORM" \
  JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1 \
  CFLIBS_USE_JAX_IDENTIFIER=1 \
    timeout "$PER_ITER_TIMEOUT" .venv/bin/python scripts/run_unified_benchmark.py \
    --quick --max-outer-folds 1 --sections all \
    --id-workflows "$IDENTIFIER" \
    --composition-workflows iterative_jax \
    --db-path ASD_da/libs_production.db \
    --data-dir "$DATA_DIR" \
    --vrabel-max-shots "$VRABEL_CAP" \
    --jax-identifier \
    --output-dir "output/loop-$RUN_DATE/$CELL/iter-$ITER_PADDED" \
    > "$LOG_FILE" 2>&1
  exit_code=$?
  set -e

  iter_end=$(date +%s)
  iter_dur=$((iter_end - iter_start))

  if [ "$exit_code" -eq 0 ]; then
    COMPLETED=$((COMPLETED + 1))
    cat > "$ITER_DIR/_done.json" <<EOF
{"cell":"$CELL","identifier":"$IDENTIFIER","platform":"$PLATFORM","iter":$i,"exit_code":0,"duration_sec":$iter_dur,"completed_at":"$(date -Iseconds)"}
EOF
    log "iter-$ITER_PADDED OK in ${iter_dur}s (done=$COMPLETED fail=$FAILED)"
  else
    FAILED=$((FAILED + 1))
    cat > "$ITER_DIR/_failed.json" <<EOF
{"cell":"$CELL","identifier":"$IDENTIFIER","platform":"$PLATFORM","iter":$i,"exit_code":$exit_code,"duration_sec":$iter_dur,"failed_at":"$(date -Iseconds)"}
EOF
    log "iter-$ITER_PADDED FAIL exit=$exit_code in ${iter_dur}s (done=$COMPLETED fail=$FAILED)"
    if [ "$FAILED" -ge 3 ] && [ "$COMPLETED" -eq 0 ]; then
      log "ABORT: 3 consecutive failures with 0 successes"
      break
    fi
  fi
  echo "{\"cell\":\"$CELL\",\"iter\":$i,\"exit_code\":$exit_code,\"duration_sec\":$iter_dur}" >> "$MANIFEST"
done

TOTAL=$(($(date +%s) - START))
log "=== cell $CELL done: completed=$COMPLETED failed=$FAILED total=${TOTAL}s ==="
