#!/bin/bash
# Atomic-DB benchmark across vasp-01/02/03: one array task per (db, dataset), reusing the prebuilt
# cluster env /cluster/shared/envs/cflibs-gpu2 (CPU; reference solver has no GPU kernel). Reads the
# staged repo at /cluster/shared/ai/cf-libs-bench/dbbench/repo; writes per-unit JSONs to results/.
# Submit from ai-proxy:  sbatch scripts/benchmarks/db_bench.sbatch
#
#SBATCH --job-name=cflibs-dbbench
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/cluster/shared/ai/cf-libs-bench/dbbench/results/db-%a.out
set -euo pipefail

REPO=/cluster/shared/ai/cf-libs-bench/dbbench/repo
PY="$REPO/.venv/bin/python"   # fresh venv built on a node (current deps), NOT the stale cflibs-gpu2
RES=/cluster/shared/ai/cf-libs-bench/dbbench/results

DBS=(ASD_da/libs_production.db output/r4_nist_vald_backfill.db output/vald_complete.db output/vald_gradeB.db)
DBN=(NIST R4 VALDcomplete VALDgradeB)
DSETS=(supercam_labcal:dev aalto:dev supercam_scct:test bhvo2_chemcam:test)

i=$SLURM_ARRAY_TASK_ID
DB="${DBS[$((i / 4))]}"; NAME="${DBN[$((i / 4))]}"
E="${DSETS[$((i % 4))]}"; DS="${E%%:*}"; SPLIT="${E##*:}"

export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
echo "node=$(hostname) task=$i db=$NAME dataset=$DS split=$SPLIT cpus=${SLURM_CPUS_PER_TASK:-8} $(date -Is)"
cd "$REPO"
# login-shell/LD_LIBRARY_PATH hygiene (project note: corrupted env forces CPU fallback otherwise)
exec bash -c "unset LD_LIBRARY_PATH; PYTHONPATH='$REPO' '$PY' -u scripts/arbor/eval_one.py \
  --db '$DB' --dataset '$DS' --split '$SPLIT' --workers ${SLURM_CPUS_PER_TASK:-8} \
  --out '$RES/${NAME}_${DS}.json'"
