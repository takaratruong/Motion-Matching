#!/bin/bash
# Headless live-steering validation for a compiled global terrain route.

#SBATCH --job-name=mb-global-route
#SBATCH --account=move
#SBATCH --partition=humanoid
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-mb-global-route-%j.out
#SBATCH --comment=heldout-live-steering-between-exact-terrain-portals

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
MOTIONBRICKS=/move/u/bodow/Projects/reference-nvidia-groot-wbc/motionbricks
DIFFUSION=/move/u/bodow/Projects/TML-BeyondMimic-autoresearch-simple-policy
MANIFEST=${TAKARA_MANIFEST:-${PROJECT}/artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/portals_v1/portal_manifest.json}
OUTPUT=${TAKARA_OUTPUT:-${PROJECT}/artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/live_rollout_seed17_v5_tangent_resume}
MAXIMUM_TICKS=${TAKARA_MAXIMUM_TICKS:-2400}

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/sonic/python:${MOTIONBRICKS}:${MOTIONBRICKS}/scripts:${DIFFUSION}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mb-global-route-mpl-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/mb-global-route-cache-${SLURM_JOB_ID}

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -B \
    -m mm_sonic.motionbricks_global_terrain_rollout \
    --portal-manifest "${MANIFEST}" \
    --motionbricks-root "${MOTIONBRICKS}" \
    --output-dir "${OUTPUT}" \
    --seed 17 \
    --maximum-ticks "${MAXIMUM_TICKS}"
