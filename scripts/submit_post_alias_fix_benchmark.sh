#!/usr/bin/env bash
#
# Submit a fresh unified benchmark to a vasp V100S node to measure
# whether the 4 ALIAS-line fixes that landed on 2026-05-14 closed the
# universal-miss gap on Vrabel rp=30,000:
#
#   PR #175 (ftp1)    adaptive r2_gate_mode
#   PR #176 (dj6y)    per-ion-stage relative_cl_threshold
#   PR #177 (762f)    temperature_estimator_mode robust path
#   PR #180           alias_v2 promotion (ftp1 + dj6y bundled)
#
# Bead: CF-LIBS-improved-d553. The 2026-05-14 findings characterized the
# algorithms BEFORE these landed; hybrid_union macro-F1 was 0.69, with
# 53 of 174 (spectrum, element) pairs missed by every identifier (49 on
# Vrabel alone). This run measures the post-fix delta.
#
# Usage:
#   sbatch scripts/submit_post_alias_fix_benchmark.sh [--node vasp-02]
#
#SBATCH --job-name=post-alias-fix-bench
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/post-alias-fix-%j.out
#SBATCH --error=logs/slurm/post-alias-fix-%j.err

set -euo pipefail

NODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node) NODE="$2"; shift 2 ;;
        --help|-h) grep '^#' "$0" | head -30; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
OUTPUT_DIR="${REPO_ROOT}/output/post-alias-fix-d553"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs/slurm"

# JAX on GPU, user-private compile cache (avoids the /cluster/shared/jax-cache
# uid-skew that hung jobs 1909/1914/1915 — see earlier session notes).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/home/brian/jax-cache}"

# PHASE 2: ceiling-test bead n3rf.3. Phase 1b (job 2063, 5m18s) measured
# the 3 alias variants — alias_v2 macro-F1=0.364, alias=0.139. Phase 2 adds
# the basis-driven ensemble workflows to find the new top-line ceiling. Key
# question: with alias_v2 replacing alias in the hybrid_union ensemble, does
# the new ceiling exceed the prior 0.69, and how close to the 0.85 literature
# target?
#
# Correlation is omitted (job 2062 demonstrated it dominates runtime via
# O(elements × spectra) SQLite partition-function lookups; not worth blocking
# on for a ceiling test). alias and alias_high_recall are also omitted —
# Phase 1b covered them and they're tracked in beads n3rf.1 / n3rf.2.
ID_WORKFLOWS=(
    alias_v2                    # new ALIAS line in the ensemble
    comb                        # canonical sanity (F1=0.03 known)
    spectral_nnls               # basis-driven sparse ID (F1=0.44 prior)
    hybrid_union                # 5-identifier consensus (prior best, F1=0.69)
    hybrid_consensus_2of3       # 2-of-3 majority (asta-12 alkali-FP fix)
)

# ID-only; composition deferred to a separate run.
SECTIONS=id

echo "=== Post-ALIAS-fix benchmark (bead d553) ==="
echo "Output dir:                  ${OUTPUT_DIR}"
echo "JAX_PLATFORMS:               ${JAX_PLATFORMS}"
echo "JAX_COMPILATION_CACHE_DIR:   ${JAX_COMPILATION_CACHE_DIR}"
echo "Commit:                      $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "ID workflows:                ${ID_WORKFLOWS[*]}"
echo "Sections:                    ${SECTIONS}"
echo "=================================="

cd "${REPO_ROOT}"

# Data resolution: prefer the NFS-mounted cluster share, fall back to local.
if [[ -d /cluster/shared/cf-libs-bench/data ]]; then
    DATA_DIR=/cluster/shared/cf-libs-bench/data
else
    DATA_DIR="${REPO_ROOT}/data"
fi
echo "Data dir:                    ${DATA_DIR}"

# Datasets:
# - Aalto + BHVO-2 = full (small footprint)
# - Vrabel: --vrabel-max-shots 1 (~100 spectra) — enough for the Si/Mg/Al
#   recall measurement that's the whole point of this run.
.venv/bin/python scripts/run_unified_benchmark.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --sections "${SECTIONS}" \
    --id-workflows "${ID_WORKFLOWS[@]}" \
    --quick \
    --vrabel-max-shots 1 \
    --max-outer-folds 1 \
    --output-format parquet \
    --experiment-label "post-alias-fix-d553-phase1" \
    --seed 42

echo "=== Benchmark complete. Results in ${OUTPUT_DIR} ==="
ls -la "${OUTPUT_DIR}" | head -20
