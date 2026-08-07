#!/bin/bash
# Summarize an exact-gated co-warp compound bank and choose visual reviews.

#SBATCH --job-name=cowarp-comp-sum
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-cowarp-compound-summary-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
BANK=${COWARP_COMPOUND_BANK:?set COWARP_COMPOUND_BANK}
COUNT=${COWARP_COMPOUND_REVIEW_COUNT:-32}
MAXIMUM_PER_SOURCE=${COWARP_COMPOUND_REVIEW_MAXIMUM_PER_SOURCE:-2}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.summarize_cowarped_compound_bank \
    "${BANK}" \
    --review-count "${COUNT}" \
    --maximum-per-physical-source "${MAXIMUM_PER_SOURCE}"
