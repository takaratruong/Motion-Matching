#!/bin/bash
# CPU-only qualification of additional real C490 hills, bumps, and grades.

#SBATCH --job-name=c490-varied-man12
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-varied-man12-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
# Stratified from the clean C490 archive by terrain-following pelvis profile:
# large hills, multi-crest/valley routes, locally rough routes, and sustained
# grades.  None of these geometries occur in the first 12-source scale-up.
INDICES=(103 133 74 88 94 142 87 110 119 82 127 124)
CLIP_INDEX=${INDICES[${SLURM_ARRAY_TASK_ID}]}
OUTPUT=${PROJECT}/artifacts/terrain_maneuver_bank/c490_slope_varied12_v2

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
