#!/bin/bash
# Select source-diverse accepted directional traversals for compound authoring.

#SBATCH --job-name=st500-comp-src
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-compound-sources-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROOT_LIST=${STAIRS500_COMPOUND_SOURCE_ROOTS:?set STAIRS500_COMPOUND_SOURCE_ROOTS}
OUTPUT=${STAIRS500_COMPOUND_SOURCE_SELECTION:?set STAIRS500_COMPOUND_SOURCE_SELECTION}
COUNT=${STAIRS500_COMPOUND_SOURCE_COUNT:-96}
MAXIMUM_PER_SOURCE=${STAIRS500_COMPOUND_MAXIMUM_PER_SOURCE:-2}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ROOT_ARGUMENTS=()
IFS=: read -r -a ROOTS <<< "${ROOT_LIST}"
for ROOT in "${ROOTS[@]}"; do
    ROOT_ARGUMENTS+=(--root "${ROOT}")
done

"${PYTHON}" -u -B -m mm_sonic.select_stairs500_compound_sources \
    "${ROOT_ARGUMENTS[@]}" \
    --count "${COUNT}" \
    --maximum-per-source "${MAXIMUM_PER_SOURCE}" \
    --output "${OUTPUT}"
