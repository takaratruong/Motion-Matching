#!/bin/bash
# Exact-mesh post-audit for saved SONIC terrain rollouts.

#SBATCH --job-name=terr-man-audit
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-maneuver-audit-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROLLOUT=${TERRAIN_TRACKER_ROLLOUT:?set TERRAIN_TRACKER_ROLLOUT}
BUNDLE=${TERRAIN_TRACKER_BUNDLE:?set TERRAIN_TRACKER_BUNDLE}
OUTPUT=${TERRAIN_TRACKER_AUDIT_OUTPUT:?set TERRAIN_TRACKER_AUDIT_OUTPUT}
SHARDS=${TERRAIN_TRACKER_AUDIT_SHARDS:?set TERRAIN_TRACKER_AUDIT_SHARDS}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.audit_terrain_maneuver_tracker_rollout \
    --rollout "${ROLLOUT}" \
    --bundle "${BUNDLE}" \
    --output "${OUTPUT}/shard_${SLURM_ARRAY_TASK_ID}.json" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count "${SHARDS}"
