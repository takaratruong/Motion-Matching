#!/bin/bash
# Render one selected stairs500 kinematic pilot per array task.

#SBATCH --job-name=st500-kin-review
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move,humanoid
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-kin-review-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SELECTION=${STAIRS500_MANEUVER_SELECTION:?set STAIRS500_MANEUVER_SELECTION}
OUTPUT=${STAIRS500_MANEUVER_REVIEW_OUTPUT:?set STAIRS500_MANEUVER_REVIEW_OUTPUT}
PYTHON=/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

pilot=$(
    "${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["selections"][int(sys.argv[2])]["pilot"])' \
        "${SELECTION}" "${SLURM_ARRAY_TASK_ID}"
)
manifest=${pilot%#*}
label=${pilot##*#}
rank=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")

"${PYTHON}" -u -B -m mm_sonic.render_terrain_maneuver_pilots \
    --manifest "${manifest}" \
    --label "${label}" \
    --output-dir "${OUTPUT}/rank_${rank}" \
    --width 480 \
    --height 270 \
    --frame-stride 2
