#!/bin/bash
# Select a strict, behavior-diverse review after a compound stair array.

#SBATCH --job-name=st500-comp-sel
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-compound-select-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SOURCE_SELECTION=${STAIRS500_COMPOUND_SOURCE_SELECTION:?set STAIRS500_COMPOUND_SOURCE_SELECTION}
BANK=${STAIRS500_COMPOUND_BANK:?set STAIRS500_COMPOUND_BANK}
OUTPUT=${STAIRS500_COMPOUND_REVIEW_SELECTION:?set STAIRS500_COMPOUND_REVIEW_SELECTION}
COUNT=${STAIRS500_COMPOUND_REVIEW_COUNT:-24}
MAXIMUM_PER_SOURCE=${STAIRS500_COMPOUND_REVIEW_MAXIMUM_PER_SOURCE:-1}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.select_stairs500_compound_review \
    --source-selection "${SOURCE_SELECTION}" \
    --bank-root "${BANK}" \
    --count "${COUNT}" \
    --maximum-per-physical-source "${MAXIMUM_PER_SOURCE}" \
    --output "${OUTPUT}"
