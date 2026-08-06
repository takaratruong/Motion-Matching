#!/bin/bash
# CPU-only exact-mesh construction for one selected stairs500 source.

#SBATCH --job-name=stairs500-maneuver
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-maneuver-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ARCHIVE=${STAIRS500_ARCHIVE:-/move/data/terrain-aware/motion-matching/grail-stairs500-clean-50hz-v1.zarr}
SELECTION=${STAIRS500_SELECTION:?set STAIRS500_SELECTION}
OUTPUT=${STAIRS500_MANEUVER_OUTPUT:?set STAIRS500_MANEUVER_OUTPUT}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON=/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python
SHARD_COUNT=${STAIRS500_SHARD_COUNT:-0}
SOURCE_COUNT=$(
    "${PYTHON}" -c \
        'import json,sys; print(len(json.load(open(sys.argv[1]))["sources"]))' \
        "${SELECTION}"
)

run_source() {
    local rank=$1
    local clip_index clip_name expected status
    read -r clip_index clip_name < <(
        "${PYTHON}" -c \
            'import json,sys; r=json.load(open(sys.argv[1]))["sources"][int(sys.argv[2])]; print(r["clip_index"],r["clip_name"])' \
            "${SELECTION}" "${rank}"
    )
    expected=$(printf '%s/clip_%03d_%s/clip_summary.json' \
        "${OUTPUT}" "${clip_index}" "${clip_name}")
    if [[ -f "${expected}" ]]; then
        echo "rank ${rank}: source ${clip_index} already complete; skipping"
        return 0
    fi
    set +e
    "${PYTHON}" -u -B -m mm_sonic.build_c490_terrain_maneuver_bank \
        --archive "${ARCHIVE}" \
        --clip-index "${clip_index}" \
        --output-root "${OUTPUT}" \
        --allow-family stairs500_stair_p1 \
        --allow-family stairs500_stair_p2 \
        --ramp-frames 16 \
        --hold-frames 15 \
        --reverse-ramp-frames 16 \
        --reverse-hold-frames 4 \
        --pivot-speed-mps 0.20
    status=$?
    set -e
    if [[ ${status} -eq 2 ]]; then
        echo "rank ${rank}: source ${clip_index} produced no admitted maneuver"
        return 0
    fi
    return "${status}"
}

if (( SHARD_COUNT > 0 )); then
    for ((rank=SLURM_ARRAY_TASK_ID; rank<SOURCE_COUNT; rank+=SHARD_COUNT)); do
        run_source "${rank}"
    done
else
    run_source "${SLURM_ARRAY_TASK_ID}"
fi
