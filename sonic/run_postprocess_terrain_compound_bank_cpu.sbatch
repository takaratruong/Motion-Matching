#!/bin/bash
# Plot a compound terrain bank and select its strict dense-video review set.

#SBATCH --job-name=terrain-comp-post
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-comp-post-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
BANK=${TERRAIN_COMPOUND_BANK:?set TERRAIN_COMPOUND_BANK}
OUTPUT=${TERRAIN_COMPOUND_POST_OUTPUT:?set TERRAIN_COMPOUND_POST_OUTPUT}
COUNT=${TERRAIN_COMPOUND_REVIEW_COUNT:-24}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

mkdir -p "${OUTPUT}"
export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.plot_c490_terrain_maneuver_bank \
    --bank-root "${BANK}" \
    --output "${OUTPUT}/terrain_profiles.png"

"${PYTHON}" -u -B -m mm_sonic.select_terrain_compound_review \
    --bank-root "${BANK}" \
    --count "${COUNT}" \
    --output "${OUTPUT}/review${COUNT}.json"
