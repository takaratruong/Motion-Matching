#!/bin/bash
# Attach exact Task4/Task12 commands and abrupt-event masks to finalized noisy
# shards.  This is CPU-only and intentionally lightweight.

#SBATCH --job-name=takara-abrupt-label
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --qos=normal
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-takara-abrupt-label-%A_%a.out

set -euo pipefail

MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=${TAKARA_RUN_ROOT:-/move/data/terrain-aware/sonic-rollouts/takara_bones_mm_abrupt_v1_task12}
SHARD=${TAKARA_SHARD:-${SLURM_ARRAY_TASK_ID}}
CORPUS=${TAKARA_CORPUS_PATH:-${RUN_ROOT}/_source/takara_bones_mm_abrupt.zarr}
ROLLOUT=${TAKARA_ROLLOUT:-${RUN_ROOT}/noisy_shard${SHARD}}
OUTPUT=${TAKARA_OUTPUT:-${RUN_ROOT}/noisy_shard${SHARD}_commands.zarr}

if [[ ! -f "${ROLLOUT}/collect_summary.json" ]]; then
    echo "missing finalized noisy rollout shard: ${ROLLOUT}" >&2
    exit 8
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing duplicate command-sidecar ownership key: ${OUTPUT}" >&2
    exit 9
fi

export PYTHONPATH="${MM_ROOT}/sonic/python:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="" /move/u/bodow/miniconda3/envs/cloc3/bin/python -u \
    -m mm_sonic.annotate_rollout_commands \
    --corpus "${CORPUS}" \
    --rollout "${ROLLOUT}" \
    --output "${OUTPUT}"
