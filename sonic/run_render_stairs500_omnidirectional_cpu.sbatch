#!/bin/bash
# Render one exact-audited directional pilot selected by an array manifest.

#SBATCH --job-name=stairs500-omni-render
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-omni-render-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
MANIFEST=${STAIRS500_OMNI_RENDER_MANIFEST:?set STAIRS500_OMNI_RENDER_MANIFEST}
OUTPUT=${STAIRS500_OMNI_RENDER_OUTPUT:?set STAIRS500_OMNI_RENDER_OUTPUT}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

read -r REPORT LABEL < <(
    "${PYTHON}" -c \
        'import json,sys; x=json.load(open(sys.argv[1]))[int(sys.argv[2])]; print(x["report"],x["label"])' \
        "${MANIFEST}" "${SLURM_ARRAY_TASK_ID}"
)

mkdir -p "${OUTPUT}"
export PYTHONPATH="${PROJECT}/sonic/python"
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.render_stairs500_omnidirectional_report \
    --report "${REPORT}" \
    --output "${OUTPUT}/${LABEL}.mp4" \
    --frame-stride 2 \
    --width 800 \
    --height 450
