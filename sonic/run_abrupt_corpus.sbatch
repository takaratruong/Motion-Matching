#!/bin/bash
# Build and visualize the mirrored abrupt-command kinematic bank.  This is a
# CPU preprocessing job; SONIC noise collection is intentionally a later gate.

#SBATCH --job-name=takara-mm-abrupt-src
#SBATCH --account=move
#SBATCH --partition=humanoid
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-takara-mm-abrupt-src-%j.out

set -euo pipefail

MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
RUN_ROOT=${TAKARA_RUN_ROOT:-/move/data/terrain-aware/sonic-rollouts/takara_bones_mm_abrupt_v1_task12}
MOTIONS_DIR=${TAKARA_MOTIONS_DIR:-/move/data/terrain-aware/motion-matching/takara_bones_walk_support_v2_startstop}
CORPUS=${RUN_ROOT}/_source/takara_bones_mm_abrupt.zarr

mkdir -p "${RUN_ROOT}/_source" "${RUN_ROOT}/_figures" "${RUN_ROOT}/_pipeline"
if [[ -e "${CORPUS}" ]]; then
    echo "refusing duplicate abrupt-corpus ownership key: ${CORPUS}" >&2
    exit 9
fi

export PYTHONPATH="${MM_ROOT}/sonic/python:${GRAIL_ROOT}:${PYTHONPATH:-}"
CLOC_PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
"${CLOC_PYTHON}" -u -m mm_sonic.offline_corpus \
    --motions-dir "${MOTIONS_DIR}" \
    --output "${CORPUS}" \
    --device cpu \
    --frames 600 \
    --curriculum abrupt \
    --source-quaternion-convention wxyz \
    --pelvis-velocity-weight 1.0 \
    --trajectory-position-weight 1.0 \
    --trajectory-facing-weight 1.5 \
    --trajectory-model takara_ball \
    --max-source-joint-step-rad 0.35

"${CLOC_PYTHON}" -u -m mm_sonic.plot_corpus \
    --corpus "${CORPUS}" \
    --output "${RUN_ROOT}/_figures/abrupt_stick_ball_root_first24.png" \
    --max-clips 24

"${CLOC_PYTHON}" -u -m mm_sonic.plot_corpus \
    --corpus "${CORPUS}" \
    --output "${RUN_ROOT}/_figures/all102_abrupt_base_routes.png" \
    --max-clips 204 \
    --base-only \
    --columns 6 \
    --cell-width 3.1 \
    --cell-height 2.8 \
    --dpi 150

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
for shard in $(seq 0 11); do
    python -u -m mm_sonic.prepare_sonic_bundle \
        --corpus "${CORPUS}" \
        --output "${RUN_ROOT}/_pipeline/prepared_shard${shard}" \
        --grail-stairs-root "${GRAIL_ROOT}" \
        --shard-index "${shard}" \
        --shard-count 12
done
