#!/bin/bash
# Contact-aware directional warps on registered C490 hills/slopes/rough ground.

#SBATCH --job-name=c490-directional
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-directional-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ARCHIVE=/move/data/terrain-aware/motion-matching/grail-c490-curb-slope-clean-50hz-v1.zarr
SELECTION=${C490_DIRECTIONAL_SELECTION:?set C490_DIRECTIONAL_SELECTION}
SOURCE_ROOT=${C490_DIRECTIONAL_SOURCE_ROOT:?set C490_DIRECTIONAL_SOURCE_ROOT}
OUTPUT=${C490_DIRECTIONAL_OUTPUT:?set C490_DIRECTIONAL_OUTPUT}
CLIP_LIST=${C490_DIRECTIONAL_CLIPS:?set C490_DIRECTIONAL_CLIPS}
MODE_LIST=${C490_DIRECTIONAL_MODES:-travel_gentle_left:travel_gentle_right:turning_micro_left:turning_micro_right:zigzag_gentle_left_right:zigzag_gentle_right_left}

CLIP_LIST=${CLIP_LIST//:/,}
IFS=',' read -r -a CLIPS <<< "${CLIP_LIST}"
CLIP_INDEX=${CLIPS[${SLURM_ARRAY_TASK_ID}]}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MODE_ARGUMENTS=()
IFS=: read -r -a MODES <<< "${MODE_LIST}"
for MODE in "${MODES[@]}"; do
    MODE_ARGUMENTS+=(--mode "${MODE}")
done

/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.build_stairs500_omnidirectional_pilots \
    --archive "${ARCHIVE}" \
    --selection "${SELECTION}" \
    --source-bank-root "${SOURCE_ROOT}" \
    --clip-index "${CLIP_INDEX}" \
    --continuous-terrain \
    --minimum-vertical-range-m 0.01 \
    --include-temporal-reverse \
    --maximum-amplitude-m 0.24 \
    --maximum-stance-sole-target-error-m 0.010 \
    "${MODE_ARGUMENTS[@]}" \
    --output "${OUTPUT}/clip_${CLIP_INDEX}"
