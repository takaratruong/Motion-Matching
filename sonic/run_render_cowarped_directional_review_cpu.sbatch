#!/bin/bash
# Render one selected paired co-warp review clip.

#SBATCH --job-name=cowarp-review
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-cowarp-review-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SELECTION=${COWARP_REVIEW_SELECTION:?COWARP_REVIEW_SELECTION is required}
ARCHIVE=${COWARP_REVIEW_ARCHIVE:?COWARP_REVIEW_ARCHIVE is required}
OUTPUT=${COWARP_REVIEW_OUTPUT:?COWARP_REVIEW_OUTPUT is required}
STRIDE=${COWARP_REVIEW_STRIDE:-5}
AZIMUTH=${COWARP_REVIEW_AZIMUTH_DEG:-90}
VIEW=${COWARP_REVIEW_VIEW:-side}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.render_cowarped_directional_review \
    --selection "${SELECTION}" \
    --selection-index "${SLURM_ARRAY_TASK_ID}" \
    --archive "${ARCHIVE}" \
    --output-root "${OUTPUT}" \
    --frame-stride "${STRIDE}" \
    --camera-azimuth-offset-deg "${AZIMUTH}" \
    --view "${VIEW}"
