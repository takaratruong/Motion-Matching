#!/bin/bash
# Plot and export an exact-audited directional bank after symmetry expansion.

#SBATCH --job-name=stairs500-omni-post
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-omni-post-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
OUTPUT=${STAIRS500_OMNI_POST_OUTPUT:?set STAIRS500_OMNI_POST_OUTPUT}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

ROOT_ARGUMENTS=()
if [[ -n "${STAIRS500_OMNI_POST_ROOTS:-}" ]]; then
    IFS=: read -r -a ROOTS <<< "${STAIRS500_OMNI_POST_ROOTS}"
    for ROOT in "${ROOTS[@]}"; do
        ROOT_ARGUMENTS+=(--root "${ROOT}")
    done
else
    SOURCE=${STAIRS500_OMNI_POST_SOURCE:?set STAIRS500_OMNI_POST_SOURCE or STAIRS500_OMNI_POST_ROOTS}
    MIRRORS=${STAIRS500_OMNI_POST_MIRRORS:?set STAIRS500_OMNI_POST_MIRRORS or STAIRS500_OMNI_POST_ROOTS}
    ROOT_ARGUMENTS+=(--root "${SOURCE}" --root "${MIRRORS}")
fi

mkdir -p "${OUTPUT}"
export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -u -B -m mm_sonic.plot_stairs500_omnidirectional_pilots \
    "${ROOT_ARGUMENTS[@]}" \
    --output "${OUTPUT}/all_routes.png"

"${PYTHON}" -u -B -m mm_sonic.export_stairs500_directional_bank \
    "${ROOT_ARGUMENTS[@]}" \
    --output "${OUTPUT}/sonic_selection"

"${PYTHON}" -u -B -m mm_sonic.export_stairs500_directional_bank \
    "${ROOT_ARGUMENTS[@]}" \
    --quality-tier gold \
    --output "${OUTPUT}/sonic_selection_gold"

"${PYTHON}" -u -B -m mm_sonic.select_stairs500_directional_review \
    "${ROOT_ARGUMENTS[@]}" \
    --count 32 \
    --output "${OUTPUT}/review32.json"
