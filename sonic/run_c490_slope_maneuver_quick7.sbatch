#!/bin/bash
# Rebuild the seven admitted varied-terrain sources with a shorter joystick
# reversal window: 16-frame decel, four-frame pivot, 16-frame acceleration.

#SBATCH --job-name=c490-man-quick7
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --array=0-6%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-man-quick7-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
INDICES=(76 79 83 97 121 136 143)
CLIP_INDEX=${INDICES[${SLURM_ARRAY_TASK_ID}]}
OUTPUT=${PROJECT}/artifacts/terrain_maneuver_bank/c490_slope_quick7_v2

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.build_c490_terrain_maneuver_bank \
    --clip-index "${CLIP_INDEX}" \
    --output-root "${OUTPUT}" \
    --ramp-frames 16 \
    --hold-frames 15 \
    --reverse-ramp-frames 16 \
    --reverse-hold-frames 4
