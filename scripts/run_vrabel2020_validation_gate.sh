#!/bin/bash
# Vrábel 2020 LIBS validation gate.
#
# Wraps both modes of run_vrabel2020_benchmark.py into a single
# pass/fail check suitable for invocation from a swarm benchmark gate
# (.swarm/benchmark.toml run_command_template) or a post-merge runner.
#
# Exit code:
#   0  → both modes passed their regression floor
#   1  → at least one mode regressed below floor
#   2  → script invocation error (data missing, etc.)
#
# Usage:
#   scripts/run_vrabel2020_validation_gate.sh                      # default tier=heavy
#   VRABEL_TIER=light  scripts/run_vrabel2020_validation_gate.sh   # shots=20
#   VRABEL_TIER=heavy  scripts/run_vrabel2020_validation_gate.sh   # shots=100
#   VRABEL_TIER=nightly scripts/run_vrabel2020_validation_gate.sh  # shots=500
#
# All artifacts (per-mode JSON, log) go under
#   benchmark_artifacts/vrabel2020-<timestamp>/
# so the orchestrator can ship them in the PR body.
#
# To wire as a per-PR gate in beefcake-swarm's benchmark gate, edit
# .swarm/benchmark.toml and set the run_command_template to call this
# script when the diff classifier emits "heavy" or "nightly". The
# default classify_command in the existing benchmark.toml will route
# any cflibs/, scripts/, or data/ touch to heavy automatically.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TIER="${VRABEL_TIER:-heavy}"
case "$TIER" in
    light)   SHOTS=20  ;;     # ~2 min, low-fidelity smoke
    heavy)   SHOTS=100 ;;     # ~10 min, real signal
    nightly) SHOTS=500 ;;     # ~30 min, full data
    *)       echo "VRABEL_TIER must be light/heavy/nightly, got '$TIER'" >&2; exit 2 ;;
esac

PYTHON="${VRABEL_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DATA_DIR="${VRABEL_DATA_DIR:-${REPO_ROOT}/data/vrabel2020_soil_benchmark}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="${REPO_ROOT}/benchmark_artifacts/vrabel2020-${TIER}-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] [vrabel2020-gate] $*"; }

# ── Preflight ─────────────────────────────────────────────────────────
[[ -x "$PYTHON" ]]                || { echo "FATAL: python not found at $PYTHON" >&2; exit 2; }
[[ -d "$DATA_DIR" ]]              || { echo "FATAL: data dir missing: $DATA_DIR" >&2; exit 2; }
[[ -f "$DATA_DIR/test.h5" ]]      || { echo "FATAL: test.h5 not in $DATA_DIR — see scripts/run_vrabel2020_benchmark.py for download" >&2; exit 2; }
[[ -f "$DATA_DIR/train.h5" ]]     || { echo "FATAL: train.h5 not in $DATA_DIR" >&2; exit 2; }
[[ -f "$DATA_DIR/test_labels.csv" ]] || { echo "FATAL: test_labels.csv missing" >&2; exit 2; }
[[ -f "$DATA_DIR/support_tables.xlsx" ]] || { echo "FATAL: support_tables.xlsx missing" >&2; exit 2; }

log "tier=$TIER shots_per_sample=$SHOTS data_dir=$DATA_DIR out_dir=$OUT_DIR"

# ── Run classification then composition; aggregate exit codes ────────
CLS_RC=0
COMP_RC=0

log "running classification mode"
"$PYTHON" "$REPO_ROOT/scripts/run_vrabel2020_benchmark.py" \
    --mode classification \
    --data-dir "$DATA_DIR" \
    --shots-per-sample "$SHOTS" \
    --out-dir "$OUT_DIR" \
    > "$OUT_DIR/classification.log" 2>&1 || CLS_RC=$?

log "classification done (rc=$CLS_RC)"
tail -3 "$OUT_DIR/classification.log" || true

log "running composition mode"
"$PYTHON" "$REPO_ROOT/scripts/run_vrabel2020_benchmark.py" \
    --mode composition \
    --data-dir "$DATA_DIR" \
    --shots-per-sample "$SHOTS" \
    --out-dir "$OUT_DIR" \
    > "$OUT_DIR/composition.log" 2>&1 || COMP_RC=$?

log "composition done (rc=$COMP_RC)"
tail -3 "$OUT_DIR/composition.log" || true

# ── Roll up ──────────────────────────────────────────────────────────
echo
log "═════ Vrábel2020 validation gate summary ═════"
log "tier=$TIER  classification=$([ $CLS_RC -eq 0 ] && echo PASS || echo FAIL)  composition=$([ $COMP_RC -eq 0 ] && echo PASS || echo FAIL)"
log "artifacts: $OUT_DIR/"

# Exit non-zero if either mode regressed below floor
if [[ $CLS_RC -ne 0 || $COMP_RC -ne 0 ]]; then
    log "RESULT: FAIL (cls=$CLS_RC, comp=$COMP_RC)"
    exit 1
fi
log "RESULT: PASS"
exit 0
