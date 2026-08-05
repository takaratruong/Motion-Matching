#!/bin/bash
# Switch-controller viewer: live MotionBricks plus bidirectional terrain portals.

#SBATCH --job-name=hiv-global-terrain
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --qos=normal
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-mb-global-terrain-viewer-%j.out
#SBATCH --comment=motionbricks-global-terrain-v6:lane-axis-jit-bidirectional

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
MOTIONBRICKS=/move/u/bodow/Projects/reference-nvidia-groot-wbc/motionbricks
DIFFUSION=/move/u/bodow/Projects/TML-BeyondMimic-autoresearch-simple-policy
FORWARD_MANIFEST=${TAKARA_FORWARD_MANIFEST:-${PROJECT}/artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_v1/portals_entry_v2/portal_manifest.json}
REVERSE_MANIFEST=${TAKARA_REVERSE_MANIFEST:-${PROJECT}/artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_reverse_v1/portals_entry_v1/portal_manifest.json}
BROWSER_PORT=${TAKARA_BROWSER_PORT:-8772}
BROWSER_TOKEN=${TAKARA_BROWSER_TOKEN:-motionbricks-global-terrain-v6}

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/sonic/python:${MOTIONBRICKS}:${MOTIONBRICKS}/scripts:${DIFFUSION}"
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/mb-global-terrain-viewer-mpl-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/mb-global-terrain-viewer-cache-${SLURM_JOB_ID}

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -B \
  -m mm_sonic.motionbricks_global_terrain_viewer \
  --terrain-route-manifest "${FORWARD_MANIFEST}" \
  --terrain-route-manifest "${REVERSE_MANIFEST}" \
  --motionbricks-root "${MOTIONBRICKS}" \
  --browser-port "${BROWSER_PORT}" \
  --browser-token "${BROWSER_TOKEN}" \
  --browser-wait-seconds 14400
