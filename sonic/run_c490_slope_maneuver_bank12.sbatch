#!/bin/bash
# Modest CPU-only scale-up of exact-grounded C490 slope maneuvers.

#SBATCH --job-name=c490-slope-man12
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-slope-man12-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
INDICES=(76 83 89 97 106 143 79 92 114 121 135 136)
CLIP_INDEX=${INDICES[${SLURM_ARRAY_TASK_ID}]}
OUTPUT=${PROJECT}/artifacts/terrain_maneuver_bank/c490_slope_stratified12_v1

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.build_c490_terrain_maneuver_bank \
    --clip-index "${CLIP_INDEX}" \
    --output-root "${OUTPUT}"
