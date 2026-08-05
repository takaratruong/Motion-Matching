#!/bin/bash
# Physics/noise collection for the complete mirrored abrupt-command bank.
# Shards are independent ownership keys; keep only two L40s active so this
# robustness leg does not crowd the main training/evaluation queue.

#SBATCH --job-name=takara-abrupt-noise
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --qos=normal
#SBATCH --array=0-11%2
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-takara-abrupt-noise-%A_%a.out

set -euo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=${TAKARA_RUN_ROOT:-/move/data/terrain-aware/sonic-rollouts/takara_bones_mm_abrupt_v1_task12}
SHARD=${TAKARA_SHARD:-${SLURM_ARRAY_TASK_ID}}
PREPARED=${RUN_ROOT}/_pipeline/prepared_shard${SHARD}
OUTPUT=${TAKARA_OUTPUT:-${RUN_ROOT}/noisy_shard${SHARD}}
CHECKPOINT=${TAKARA_TRACKER_CHECKPOINT_DIR:-/move/u/justingu/Projects/grail-stairs/sonic_port/takara/ft_runs/poc}
DATASET_PREFIX=${TAKARA_DATASET_PREFIX:-takara_bones_mm_abrupt_v1_task12}
NUM_ENVS=${TAKARA_NUM_ENVS:-256}

if [[ ! -f "${PREPARED}/motion_lib_merged.pkl" ]]; then
    echo "missing prepared shard: ${PREPARED}" >&2
    exit 8
fi
if [[ ! -r "${CHECKPOINT}/last.pt" || ! -r "${CHECKPOINT}/config.yaml" ]]; then
    echo "missing readable checkpoint/config bundle: ${CHECKPOINT}" >&2
    exit 8
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing duplicate noisy-shard ownership key: ${OUTPUT}" >&2
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

set +e
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
    --seed "$((800 + SHARD))" \
    --store_dtype float32 \
    --flat \
    --no-edge_gate \
    --assert_no_short \
    --flush_every 2000 \
    --max_hours 1.35 \
    --dataset_name "${DATASET_PREFIX}_shard${SHARD}"
COLLECT_STATUS=$?
set -e

test -f "${OUTPUT}/collect_summary.json"
if [[ ${COLLECT_STATUS} -ne 0 ]]; then
    # A start can be intrinsically untrackable even after the bounded retry
    # budget.  Keep an otherwise complete shard instead of discarding roughly
    # forty minutes of valid FP32 rollouts for a handful of missing windows.
    /move/u/bodow/miniconda3/envs/cloc3/bin/python - \
        "${OUTPUT}/collect_summary.json" "${COLLECT_STATUS}" <<'PY'
import json
import sys

summary_path, collector_status = sys.argv[1], int(sys.argv[2])
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)
saved = int(summary.get("saved", 0))
target = int(summary.get("target", 0))
coverage = saved / target if target > 0 else 0.0
if coverage < 0.9998:
    raise SystemExit(collector_status)
print(
    f"accepting bounded-retry shard at {saved}/{target} windows "
    f"({coverage:.6%} coverage)"
)
PY
fi
