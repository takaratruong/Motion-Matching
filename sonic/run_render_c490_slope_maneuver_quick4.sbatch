#!/bin/bash
# Dense CPU renders for the four visually cleaner varied-terrain reversals.

#SBATCH --job-name=c490-quick4-video
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-quick4-video-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
INDICES=(79 83 121 136)
NAMES=(
    terrain_slopes__slope_019__006
    terrain_slopes__slope_030__009
    terrain_slopes__slope_127__006
    terrain_slopes__slope_176__009
)
INDEX=${INDICES[${SLURM_ARRAY_TASK_ID}]}
NAME=${NAMES[${SLURM_ARRAY_TASK_ID}]}
ROOT=${PROJECT}/artifacts/terrain_maneuver_bank/c490_slope_quick7_v2
MANIFEST=${ROOT}/clip_$(printf '%03d' "${INDEX}")_${NAME}/pilots/manifest.json

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.render_terrain_maneuver_pilots \
    --manifest "${MANIFEST}" \
    --label event_00_direct_terrain_reverse \
    --output-dir "${MANIFEST%/manifest.json}/renders" \
    --width 480 \
    --height 270 \
    --hold-window-context-frames 30
