#!/bin/bash
# CPU-only compound maneuver sweep over every clean C490 curb/platform source.

#SBATCH --job-name=c490-curb-comp72
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --array=0-71%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-curb-comp72-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
CLIP_INDEX=${SLURM_ARRAY_TASK_ID}
OUTPUT=${C490_CURB_COMPOUND_OUTPUT:-${PROJECT}/artifacts/terrain_maneuver_bank/c490_curb_compound72_v1}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python -u -B \
    -m mm_sonic.build_c490_terrain_maneuver_bank \
    --clip-index "${CLIP_INDEX}" \
    --output-root "${OUTPUT}" \
    --allow-family c490_curb \
    --mode stop_restart \
    --mode reverse \
    --mode bounce \
    --pivot-count 3 \
    --ramp-frames 16 \
    --hold-frames 4 \
    --reverse-ramp-frames 16 \
    --reverse-hold-frames 4 \
    --pivot-speed-mps 0.20
