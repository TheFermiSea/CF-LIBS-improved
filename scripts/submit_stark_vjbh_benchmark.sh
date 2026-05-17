#!/usr/bin/env bash
#
# Submit one before-or-after bayesian/iterative_jax benchmark run to a
# vasp-{01,02,03} V100S node for the Stark T-factor empirical impact
# study (CF-LIBS-improved-4rwe).
#
# Usage:
#   scripts/submit_stark_vjbh_benchmark.sh --label after --node vasp-01
#   CFLIBS_DISABLE_STARK_T_FACTOR=1 scripts/submit_stark_vjbh_benchmark.sh --label before --node vasp-02
#
# The "before" / "after" labels just pin the output dir; the actual
# kernel behaviour is controlled by whether the caller pre-exports
# CFLIBS_DISABLE_STARK_T_FACTOR=1 (legacy) or leaves it unset (default fix).
#
# Designed to be self-contained: this script is the sbatch payload. It is
# invoked directly (`sbatch scripts/submit_stark_vjbh_benchmark.sh ...`)
# rather than wrapping another sbatch call, so the runner inherits the
# right node, GPU, and time budget.
#
#SBATCH --job-name=stark-vjbh-bench
#SBATCH --time=24:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/stark-vjbh-%j.out
#SBATCH --error=logs/slurm/stark-vjbh-%j.err

set -euo pipefail

LABEL=""
NODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)
            LABEL="$2"; shift 2 ;;
        --node)
            NODE="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | head -40
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$LABEL" ]]; then
    echo "ERROR: --label {before,after} is required" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
OUTPUT_DIR="${REPO_ROOT}/output/stark-fix-vjbh/${LABEL}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs/slurm"

# Pin JAX to GPU and share the persistent compile cache with the
# concurrent before/after job. The host-side env-var branch in
# _per_line_stark_gamma gives the two runs distinct jit cache keys, so
# the cache is shared safely.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/home/brian/jax-cache}"

echo "=== stark-vjbh benchmark run ==="
echo "Label:                       ${LABEL}"
echo "Node:                        ${NODE:-(scheduler)}"
echo "Output dir:                  ${OUTPUT_DIR}"
echo "JAX_PLATFORMS:               ${JAX_PLATFORMS}"
echo "JAX_COMPILATION_CACHE_DIR:   ${JAX_COMPILATION_CACHE_DIR}"
echo "CFLIBS_DISABLE_STARK_T_FACTOR: ${CFLIBS_DISABLE_STARK_T_FACTOR:-(unset → fix enabled)}"
echo "Commit:                      $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "=================================="

cd "${REPO_ROOT}"

# MCMC budget: 500 warmup / 1000 samples / 2 chains. With the post-xsuj fix
# (chain_method=vectorized + max_tree_depth=8) measured throughput on V100S is
# ~13 min/spectrum; ``--vrabel-max-shots 1`` (~100 vrabel spectra) fits the
# aalto + bhvo2 + vrabel workload into ~22h, leaving slack inside the 24h
# SLURM budget. The original ``--vrabel-max-shots 5`` setting overran by 4×.
#
# CFLIBS_BENCH_CHECKPOINT_PATH directs ``evaluate_composition_workflow`` to
# append a parquet checkpoint after every spectrum, so a SLURM timeout
# preserves completed records.
#
# --seed 0 pins the NUTS PRNG so before/after records align 1:1.
export CFLIBS_BENCH_CHECKPOINT_PATH="${OUTPUT_DIR}/composition_checkpoint.parquet"

exec .venv/bin/python scripts/run_unified_benchmark.py \
    --composition-workflows bayesian iterative_jax \
    --sections composition \
    --vrabel-max-shots 1 \
    --max-outer-folds 1 \
    --seed 0 \
    --bayesian-mcmc 500,1000,2 \
    --output-format parquet \
    --output-dir "${OUTPUT_DIR}"
