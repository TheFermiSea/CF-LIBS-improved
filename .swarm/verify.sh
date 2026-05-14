#!/usr/bin/env bash
# CF-LIBS-improved per-project verifier gate, invoked by beefcake-swarm's
# python/run.py:_gate_project_script. Contract (from beefcake-8v8q7):
#
#   $SWARM_CHANGED_FILES         newline-separated repo-relative paths
#   $SWARM_CHANGED_FILES_COUNT   integer count
#   $SWARM_WT_PATH               absolute worktree path
#   $SWARM_BASE_REF              base ref used for the diff
#   $SWARM_VERIFIER_BUDGET_SECS  this gate's timeout budget (typically 1800)
#
# Layered gating:
#
#   1. Always: pytest tests/inversion/ tests/benchmark/ tests/scripts/ -q
#      Catches broader regressions like #114 (estimate_baseline contract
#      break) and #137 (HybridIdentifier kwarg removal) that touched
#      narrow tests but broke transitively-using suites.
#
#   2. Conditional (only when identifier code changed): F1-regression
#      smoke. Triggers from .swarm/identifier-paths.txt — any changed
#      file matching ANY pattern in that file fires the smoke.
#      Baseline is .swarm/identifier-f1-baseline.json; we fail if any
#      identifier's macro_f1 drops by more than
#      tolerance_macro_f1_absolute (typically 0.02).
#
# Two failure modes:
#   - INFRA: pytest can't collect, basis dir missing, json malformed.
#     Returns exit 2 so reviewer can distinguish from a real regression.
#     The beefcake-swarm gate currently treats any non-zero as failure;
#     the exit-code split is informative for humans reading the log.
#   - REGRESSION: pytest fail OR F1 dropped beyond tolerance. Exit 1.
#
# This script is SELF_CRITICAL-protected: the swarm cannot weaken it.

set -uo pipefail

WT="${SWARM_WT_PATH:-$(pwd)}"
BUDGET="${SWARM_VERIFIER_BUDGET_SECS:-1800}"
BASELINE_FILE="$WT/.swarm/identifier-f1-baseline.json"
PATHS_FILE="$WT/.swarm/identifier-paths.txt"

log() {
  printf '[verify.sh] %s\n' "$*" >&2
}

# ─── 1. Broader regression pytest ─────────────────────────────────────────

# Only run if any of the listed test trees exist. Each one is gated
# individually so a fresh clone with a partial tree doesn't blow up.
PYTEST_TARGETS=()
for d in tests/inversion tests/benchmark tests/scripts; do
  if [[ -d "$WT/$d" ]]; then
    PYTEST_TARGETS+=("$d")
  fi
done

if (( ${#PYTEST_TARGETS[@]} > 0 )); then
  log "Running broader pytest: ${PYTEST_TARGETS[*]}"
  # Allocate ~1/3 of budget to pytest; remainder reserved for F1 smoke.
  PYTEST_TIMEOUT=$(( BUDGET / 3 ))
  if ! timeout "${PYTEST_TIMEOUT}s" python -m pytest \
        "${PYTEST_TARGETS[@]}" -q --no-header \
        --timeout=120 --timeout-method=signal 2>&1; then
    log "FAIL: broader pytest"
    exit 1
  fi
  log "PASS: broader pytest"
else
  log "skip: no broader test trees present"
fi

# ─── 2. F1-regression gate (conditional on identifier-touching change) ────

if [[ ! -f "$BASELINE_FILE" || ! -f "$PATHS_FILE" ]]; then
  log "skip: no baseline + path list — F1 gate not configured"
  exit 0
fi

# Match changed files against the patterns in identifier-paths.txt.
# `grep -E` (ERE) with one pattern per line. Empty SWARM_CHANGED_FILES
# is a valid "no diff" signal; treat as "no F1 trigger".
TRIGGERED=0
if [[ -n "${SWARM_CHANGED_FILES:-}" ]]; then
  # Build the alternation once, skipping blank/comment lines.
  PATTERN=$(grep -vE '^\s*($|#)' "$PATHS_FILE" | paste -sd'|' -)
  if [[ -n "$PATTERN" ]]; then
    if printf '%s\n' "$SWARM_CHANGED_FILES" \
        | grep -qE "($PATTERN)" 2>/dev/null; then
      TRIGGERED=1
    fi
  fi
fi

if (( TRIGGERED == 0 )); then
  log "skip: no identifier files changed — F1 gate not triggered"
  exit 0
fi

log "F1 gate TRIGGERED — identifier file(s) changed"

# Smoke prerequisites: NFS basis library directory. On a laptop or in CI
# without /cluster mount, we can't run the smoke — exit 2 (infra) rather
# than 1 (regression) so the human reviewing knows the smoke didn't
# actually run and an out-of-band manual smoke is required before merge.
BASIS_DIR="/cluster/shared/cf-libs-bench/basis_libraries"
if [[ ! -d "$BASIS_DIR" ]]; then
  log "INFRA: basis-library dir $BASIS_DIR not mounted — F1 smoke cannot run"
  log "       this is expected off-cluster; reviewer must run smoke before merge"
  exit 2
fi

OUTDIR="/tmp/verifier-smoke-$$-$(date +%s)"
mkdir -p "$OUTDIR"
trap 'rm -rf "$OUTDIR"' EXIT

# Budget for the smoke itself: whatever's left after pytest, minus 60s
# margin for json parsing + log dump.
SMOKE_TIMEOUT=$(( (BUDGET * 2 / 3) - 60 ))
(( SMOKE_TIMEOUT < 300 )) && SMOKE_TIMEOUT=300

log "Running F1 smoke (budget ${SMOKE_TIMEOUT}s) → $OUTDIR"
if ! timeout "${SMOKE_TIMEOUT}s" python "$WT/scripts/run_unified_benchmark.py" \
      --quick --max-outer-folds 1 \
      --sections id \
      --id-workflows alias comb correlation spectral_nnls hybrid_union \
      --jax-identifier \
      --basis-dir "$BASIS_DIR" \
      --vrabel-max-shots 1 \
      --dataset-shard 1/3 \
      --output-dir "$OUTDIR" 2>&1; then
  log "FAIL: F1 smoke crashed / timed out"
  exit 1
fi

SUMMARY="$OUTDIR/id_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  log "INFRA: smoke ran but no id_summary.json at $SUMMARY"
  exit 2
fi

# Compare against baseline with a small Python snippet. Returns exit 0
# on pass, exit 1 on regression. stdout is the comparison table for the
# verifier log.
#
# Schema v2 (2026-05-14, overnight-loop): MULTI-METRIC gate.
#   - macro_f1, macro_precision, macro_recall: regression = drop > tol
#   - fp_per_spectrum: regression = rise > tol (reverse polarity)
#   - any identifier MISSING from id_summary.json = regression
# The gate is conservative: ANY identifier × ANY metric beyond tolerance
# fails the PR. Improvements (drops in fp_per_spectrum, rises in the
# other three) are logged but never cause a failure.
python - "$BASELINE_FILE" "$SUMMARY" <<'PY' || exit 1
import json, sys

baseline_path, observed_path = sys.argv[1], sys.argv[2]
with open(baseline_path) as f:
    baseline = json.load(f)
with open(observed_path) as f:
    observed = json.load(f)

tol_f1 = baseline.get("tolerance_macro_f1_absolute", 0.02)
tol_p = baseline.get("tolerance_macro_precision_absolute", 0.02)
tol_r = baseline.get("tolerance_macro_recall_absolute", 0.02)
tol_fp = baseline.get("tolerance_fp_per_spectrum_absolute", 0.02)
expected = baseline.get("identifiers", {})
overall = (observed.get("overall") or {})

print(f"\n  IDENTIFIER         METRIC   BASELINE   OBSERVED     DELTA   STATUS")
print(f"  " + "-" * 68)

any_fail = False
any_run = False
any_improvement = False

# (metric_key, polarity, tolerance, label)
# polarity = +1 means higher-is-better (regression on drop), -1 means
# lower-is-better (regression on rise).
metrics = [
    ("macro_f1",         +1, tol_f1, "F1   "),
    ("macro_precision",  +1, tol_p,  "P    "),
    ("macro_recall",     +1, tol_r,  "R    "),
    ("fp_per_spectrum",  -1, tol_fp, "FP/sp"),
]

for ident, ref in sorted(expected.items()):
    obs_block = overall.get(ident)
    if obs_block is None:
        print(f"  {ident:18s} {'MISSING':>5s}     {ref['macro_f1']:.4f}        n/a       —    REGRESSION")
        any_fail = True
        continue
    any_run = True
    for key, polarity, tol, label in metrics:
        base_v = float(ref.get(key, 0.0))
        obs_v = float(obs_block.get(key, 0.0))
        delta = obs_v - base_v
        # regression: drop > tol on +polarity, rise > tol on -polarity
        if polarity > 0:
            is_regression = delta < -tol
            is_improvement = delta > tol
        else:
            is_regression = delta > tol
            is_improvement = delta < -tol
        if is_regression:
            status = "REGRESSION"
            any_fail = True
        elif is_improvement:
            status = "IMPROVEMENT"
            any_improvement = True
        else:
            status = "ok"
        print(f"  {ident:18s} {label}    {base_v:.4f}    {obs_v:.4f}   {delta:+.4f}   {status}")

print()
if not any_run:
    print("  INFRA: no identifier ran in the smoke; check id-workflows arg")
    sys.exit(2)
if any_fail:
    print(f"  REGRESSION: at least one metric beyond tolerance "
          f"(F1/P/R > {tol_f1:.3f}; FP/sp > {tol_fp:.3f}) or identifier MISSING")
    sys.exit(1)
if any_improvement:
    print(f"  PASS — at least one metric IMPROVED beyond tolerance. "
          "Consider updating .swarm/identifier-f1-baseline.json after smoke confirmation.")
else:
    print(f"  PASS — all metrics within tolerance, no regressions.")
sys.exit(0)
PY
