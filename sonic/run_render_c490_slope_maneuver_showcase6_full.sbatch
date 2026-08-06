#!/bin/bash
# Full-length CPU renders for the cleaner varied-terrain reversal showcase.

#SBATCH --job-name=c490-varied-full
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-varied-full-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
INDICES=(74 87 94 119 133 142)
NAMES=(
    terrain_slopes__slope_013__004
    terrain_slopes__slope_040__008
    terrain_slopes__slope_052__004
    terrain_slopes__slope_123__005
    terrain_slopes__slope_174__001
    terrain_slopes__slope_184__009
)
INDEX=${INDICES[${SLURM_ARRAY_TASK_ID}]}
NAME=${NAMES[${SLURM_ARRAY_TASK_ID}]}
ROOT=${PROJECT}/artifacts/terrain_maneuver_bank/c490_slope_varied12_v2
MANIFEST=${ROOT}/clip_$(printf '%03d' "${INDEX}")_${NAME}/pilots/manifest.json

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.render_terrain_maneuver_pilots \
    --manifest "${MANIFEST}" \
    --label event_00_direct_terrain_reverse \
    --output-dir "${MANIFEST%/manifest.json}/full_renders" \
    --width 480 \
    --height 270
