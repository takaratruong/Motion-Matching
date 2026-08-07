#!/bin/bash
# Render one selected paired co-warp compound per CPU array task.

#SBATCH --job-name=cowarp-comp-render
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-cowarp-compound-render-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SELECTION=${COWARP_COMPOUND_REVIEW_SELECTION:?set COWARP_COMPOUND_REVIEW_SELECTION}
OUTPUT=${COWARP_COMPOUND_REVIEW_OUTPUT:?set COWARP_COMPOUND_REVIEW_OUTPUT}
ARCHIVE=${COWARP_COMPOUND_ARCHIVE:?set COWARP_COMPOUND_ARCHIVE}
VIEW=${COWARP_COMPOUND_REVIEW_VIEW:-side}
PYTHON=/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PILOT=$(
    "${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["selections"][int(sys.argv[2])]["pilot"])' \
        "${SELECTION}" "${SLURM_ARRAY_TASK_ID}"
)
MANIFEST=${PILOT%#*}
LABEL=${PILOT##*#}
RANK=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")

"${PYTHON}" -u -B -m mm_sonic.render_terrain_maneuver_pilots \
    --manifest "${MANIFEST}" \
    --label "${LABEL}" \
    --motion-archive "${ARCHIVE}" \
    --output-dir "${OUTPUT}/rank_${RANK}" \
    --width 480 \
    --height 270 \
    --frame-stride 2 \
    --view "${VIEW}"
