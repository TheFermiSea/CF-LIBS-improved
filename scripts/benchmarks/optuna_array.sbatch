#!/bin/bash
#SBATCH --job-name=ded-optuna
#SBATCH --partition=normal
#SBATCH --array=0-23%24
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/shared/cf-libs-v4/optuna_logs/worker_%a.out
# Parallel Optuna DED composition-ACCURACY sweep across vasp-01/02/03.
# Each array task runs N trials against a shared NFS-safe Journal store.
# physics-first: optuna is tuning-tooling (scripts/), never shipped cflibs/.
set -uo pipefail

REPO=/cluster/shared/cf-libs-v4
OPT=/cluster/shared/cf-libs-v4/_optlibs
PY=/cluster/shared/envs/cflibs-gpu2/bin/python3

export PYTHONPATH="$REPO:$OPT"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Per-worker DB copy on node-local /scratch (SQLite-over-NFS lock avoidance).
WDB="/scratch/ded_optuna_${SLURM_ARRAY_TASK_ID}.db"
cp "$REPO/ASD_da/libs_production.db" "$WDB"
export CFLIBS_DB="$WDB"

cd "$REPO"
echo "worker ${SLURM_ARRAY_TASK_ID} on $(hostname) cpus=${SLURM_CPUS_PER_TASK} db=$WDB"
"$PY" scripts/optuna_ded_precision.py \
  --storage "$REPO/optuna_ded.journal" \
  --study ded_precision \
  --n-trials 12

rm -f "$WDB"
echo "worker ${SLURM_ARRAY_TASK_ID} done"
