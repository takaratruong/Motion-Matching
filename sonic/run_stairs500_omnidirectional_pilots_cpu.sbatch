#!/bin/bash
# CPU-only exact-mesh generation for a small directional stairs500 pilot.

#SBATCH --job-name=stairs500-omni
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:45:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-omni-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
OUTPUT=${STAIRS500_OMNI_OUTPUT:?set STAIRS500_OMNI_OUTPUT}
CLIP_LIST=${STAIRS500_OMNI_CLIPS:-364,6,343,207}
MODE_LIST=${STAIRS500_OMNI_MODES:-}
INCLUDE_REVERSE=${STAIRS500_OMNI_INCLUDE_REVERSE:-0}
INCLUDE_TEMPORAL_REVERSE=${STAIRS500_OMNI_INCLUDE_TEMPORAL_REVERSE:-0}

# Colons are accepted as an sbatch --export-safe list delimiter.  Commas in an
# --export value are parsed by Slurm as new variable declarations.
CLIP_LIST=${CLIP_LIST//:/,}
MODE_LIST=${MODE_LIST//:/,}

IFS=',' read -r -a CLIPS <<< "${CLIP_LIST}"
CLIP_INDEX=${CLIPS[${SLURM_ARRAY_TASK_ID}]}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MODE_ARGUMENTS=()
if [[ -n "${MODE_LIST}" ]]; then
    IFS=',' read -r -a MODES <<< "${MODE_LIST}"
    for MODE in "${MODES[@]}"; do
        MODE_ARGUMENTS+=(--mode "${MODE}")
    done
fi

SOURCE_ARGUMENTS=()
if [[ "${INCLUDE_REVERSE}" == "1" ]]; then
    SOURCE_ARGUMENTS+=(--include-reverse)
fi
if [[ "${INCLUDE_TEMPORAL_REVERSE}" == "1" ]]; then
    SOURCE_ARGUMENTS+=(--include-temporal-reverse)
fi

/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.build_stairs500_omnidirectional_pilots \
    --clip-index "${CLIP_INDEX}" \
    "${SOURCE_ARGUMENTS[@]}" \
    --maximum-stance-sole-target-error-m 0.010 \
    "${MODE_ARGUMENTS[@]}" \
    --output "${OUTPUT}/clip_${CLIP_INDEX}"
