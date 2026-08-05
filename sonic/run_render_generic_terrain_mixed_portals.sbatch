#!/bin/bash
# Render both mixed-scene portal courses, one L40 at a time.

#SBATCH --job-name=mixed-portal-render
#SBATCH --account=move
#SBATCH --partition=humanoid
#SBATCH --qos=normal
#SBATCH --array=0-1%1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-mixed-portal-render-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROOT=${PROJECT}/artifacts/generic_terrain/unified_mixed_smoke_v1
PORTAL_DIR=${PORTAL_DIR:-${ROOT}/portals}
RENDER_DIR=${RENDER_DIR:-${PORTAL_DIR}}
EVENT=${SLURM_ARRAY_TASK_ID}
EVENT_PADDED=$(printf '%02d' "${EVENT}")
EVENT_OUTPUT=${RENDER_DIR}/event_${EVENT_PADDED}
OUTPUT=${EVENT_OUTPUT}/g1_kinematic_50fps.mp4
mkdir -p "${EVENT_OUTPUT}"
export PYTHONPATH="${PROJECT}/sonic/python"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME=/tmp/mixed-portal-render-${SLURM_JOB_ID}

/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.render_generic_terrain_portal \
    --portal-manifest "${PORTAL_DIR}/portal_manifest.json" \
    --event-index "${EVENT}" \
    --output "${OUTPUT}"

test -s "${OUTPUT}"
