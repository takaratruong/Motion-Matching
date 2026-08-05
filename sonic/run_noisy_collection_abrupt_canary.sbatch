#!/bin/bash
# One-shard physics/noise canary for the abrupt command bank.  Scale only if
# this produces enough full windows without a pathological fall rate.

#SBATCH --job-name=takara-abrupt-canary
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-takara-abrupt-canary-%j.out

set -euo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=${TAKARA_RUN_ROOT:-/move/data/terrain-aware/sonic-rollouts/takara_bones_mm_abrupt_v1_task12}
SHARD=${TAKARA_SHARD:-0}
PREPARED=${RUN_ROOT}/_pipeline/prepared_shard${SHARD}
OUTPUT=${RUN_ROOT}/noisy_canary_shard${SHARD}
CHECKPOINT=${TAKARA_TRACKER_CHECKPOINT_DIR:-/move/u/justingu/Projects/grail-stairs/sonic_port/takara/ft_runs/poc}
DATASET_PREFIX=${TAKARA_DATASET_PREFIX:-takara_bones_mm_abrupt_v1_task12_canary}
NUM_ENVS=${TAKARA_NUM_ENVS:-384}

if [[ ! -f "${PREPARED}/motion_lib_merged.pkl" ]]; then
    echo "missing prepared shard: ${PREPARED}" >&2
    exit 8
fi
if [[ ! -r "${CHECKPOINT}/last.pt" || ! -r "${CHECKPOINT}/config.yaml" ]]; then
    echo "missing readable checkpoint/config bundle: ${CHECKPOINT}" >&2
    exit 8
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing duplicate noisy-canary ownership key: ${OUTPUT}" >&2
    exit 9
fi

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=Yes
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${GRAIL_ROOT}:${PYTHONPATH:-}"
source "${MM_ROOT}/sonic/setup_writable_isaac_runtime.sh"
cd "${GRAIL_ROOT}"

python -u -m sonic_port.collect.collect_sonic_windowed \
    --ckpt_dir "${CHECKPOINT}" \
    --terrain_motion_dir "${PREPARED}" \
    --motion_file "${PREPARED}/motion_lib_merged.pkl" \
    --out_dir "${OUTPUT}" \
    --num_envs "${NUM_ENVS}" \
    --samples_per_step 3 \
    --end_multiplier 30 \
    --per_start_max_fails 400 \
    --start_stride 1 \
    --collect_steps 150 \
    --episode_length_s 5 \
    --seed "$((700 + SHARD))" \
    --store_dtype float32 \
    --flat \
    --no-edge_gate \
    --assert_no_short \
    --flush_every 2000 \
    --max_hours 1.35 \
    --dataset_name "${DATASET_PREFIX}_shard${SHARD}"

test -f "${OUTPUT}/collect_summary.json"
