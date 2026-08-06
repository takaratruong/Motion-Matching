#!/bin/bash
# CPU-only visual audit of six representative saved SONIC terrain rollouts.

#SBATCH --job-name=terr-man-cpuvid
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-maneuver-cpuvideo-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROLLOUT=${TERRAIN_TRACKER_ROLLOUT:?set TERRAIN_TRACKER_ROLLOUT}
BUNDLE=${TERRAIN_TRACKER_BUNDLE:?set TERRAIN_TRACKER_BUNDLE}
OUTPUT=${TERRAIN_TRACKER_VIDEOS:?set TERRAIN_TRACKER_VIDEOS}
if [[ -n "${TERRAIN_TRACKER_EPISODES:-}" ]]; then
    IFS=: read -r -a EPISODES <<< "${TERRAIN_TRACKER_EPISODES}"
else
    EPISODES=(9 13 16 25 35 39)
fi
EPISODE=${EPISODES[${SLURM_ARRAY_TASK_ID}]}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.render_terrain_maneuver_tracker_rollout \
    --rollout "${ROLLOUT}" \
    --bundle "${BUNDLE}" \
    --episode "${EPISODE}" \
    --output-dir "${OUTPUT}/ep_${EPISODE}" \
    --width 480 \
    --height 270 \
    --frame-stride 2
