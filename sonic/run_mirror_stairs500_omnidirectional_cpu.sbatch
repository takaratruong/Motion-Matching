#!/bin/bash
# Exact full-G1 symmetry expansion for an accepted directional stair bank.

#SBATCH --job-name=stairs500-omni-mirror
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-omni-mirror-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SOURCE=${STAIRS500_OMNI_MIRROR_SOURCE:?set STAIRS500_OMNI_MIRROR_SOURCE}
OUTPUT=${STAIRS500_OMNI_MIRROR_OUTPUT:?set STAIRS500_OMNI_MIRROR_OUTPUT}
SHARD_COUNT=${STAIRS500_OMNI_MIRROR_SHARD_COUNT:-1}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SHARD_ARGUMENTS=()
if [[ "${SHARD_COUNT}" -gt 1 ]]; then
    SHARD_ARGUMENTS+=(
        --shard-index "${SLURM_ARRAY_TASK_ID:?array task required for sharding}"
        --shard-count "${SHARD_COUNT}"
    )
fi

/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.mirror_stairs500_omnidirectional_pilots \
    --root "${SOURCE}" \
    "${SHARD_ARGUMENTS[@]}" \
    --output "${OUTPUT}"
