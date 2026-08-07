#!/bin/bash
# Add stop/restart, reversal, and bounce timing to paired co-warped sources.

#SBATCH --job-name=cowarp-compound
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-cowarp-compound-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SELECTION=${COWARP_COMPOUND_SELECTION:?set COWARP_COMPOUND_SELECTION}
OUTPUT=${COWARP_COMPOUND_OUTPUT:?set COWARP_COMPOUND_OUTPUT}
ARCHIVE=${COWARP_COMPOUND_ARCHIVE:?set COWARP_COMPOUND_ARCHIVE}
MODEL=${COWARP_COMPOUND_MODEL:-/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml}
PIVOT_COUNT=${COWARP_COMPOUND_PIVOT_COUNT:-3}
MODE_LIST=${COWARP_COMPOUND_MODES:-stop_restart:reverse:bounce}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MODE_ARGUMENTS=()
IFS=: read -r -a MODES <<< "${MODE_LIST}"
for MODE in "${MODES[@]}"; do
    MODE_ARGUMENTS+=(--mode "${MODE}")
done

"${PYTHON}" -u -B -m mm_sonic.build_cowarped_compound_maneuvers \
    --selection "${SELECTION}" \
    --row-index "${SLURM_ARRAY_TASK_ID}" \
    --output-root "${OUTPUT}" \
    --motion-archive "${ARCHIVE}" \
    --model-path "${MODEL}" \
    --pivot-count "${PIVOT_COUNT}" \
    --ramp-frames 16 \
    --hold-frames 4 \
    --reverse-ramp-frames 16 \
    --reverse-hold-frames 4 \
    "${MODE_ARGUMENTS[@]}"
