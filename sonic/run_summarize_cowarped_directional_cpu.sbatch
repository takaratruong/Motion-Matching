#!/bin/bash
# Summarize one completed paired terrain+motion co-warp bank.

#SBATCH --job-name=cowarp-summary
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-cowarp-summary-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROOT=${COWARP_SUMMARY_ROOT:?COWARP_SUMMARY_ROOT is required}
REVIEW_COUNT=${COWARP_SUMMARY_REVIEW_COUNT:-24}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.summarize_cowarped_directional_bank \
    "${ROOT}" \
    --review-count "${REVIEW_COUNT}"
