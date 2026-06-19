#!/bin/bash
# J12/M3 FULL-BOARD confirmation (all spectra, no --max-spectra), one SLURM array
# task per dataset, scheduled across vasp-01/02/03. Confirms the capped (n<=100)
# PROMOTE verdict in docs/benchmarks/SCOREBOARD-2026-06-19-jit-vs-reference.
#
# Submit from ai-proxy (the sbatch wrapper ships this to the SLURM controller;
# the job runs as root on the nodes and writes outputs under the staged repo on
# /cluster/shared, which is world-readable back on ai-proxy):
#   sbatch --export=ALL,REPO_DIR=/cluster/shared/cf-libs-bench/jitpipe-m3/repo \
#       scripts/benchmarks/m3_fullboard.sbatch
#
#SBATCH --job-name=cflibs-m3-full
#SBATCH --array=0-8
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/shared/cf-libs-bench/jitpipe-m3/logs/fullboard/m3-%a.out

set -euo pipefail
: "${REPO_DIR:?export REPO_DIR=<staged repo>}"

DATASETS=(aalto csa_planetary chemcam_calib silva2022 synthetic_fixedforward \
          bhvo2_chemcam emslibs2019 supercam_labcal supercam_scct)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export JAX_COMPILATION_CACHE_DIR=/cluster/shared/cf-libs-bench/jitpipe-m3/jaxcache
mkdir -p "$JAX_COMPILATION_CACHE_DIR" /cluster/shared/cf-libs-bench/jitpipe-m3/logs/fullboard

cd "$REPO_DIR"
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID dataset=$DS time=$(date -Is)"

# Full board: omit --max-spectra so every spectrum runs.
exec ./.venv/bin/python -u scripts/run_j12_board_compare.py \
    --datasets "$DS" --include-holdout --seed 20260610 \
    --output-dir "output/j12/fullboard/$DS"
