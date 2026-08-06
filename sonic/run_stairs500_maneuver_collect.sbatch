#!/bin/bash
# Noised full-rollout pilot for a physically qualified stairs500 maneuver tracker.

#SBATCH --job-name=st500-man-collect
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodelist=move2
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-maneuver-collect-%j.out

set -euo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
CHECKPOINT=${STAIRS500_TRACKER_CHECKPOINT:?set STAIRS500_TRACKER_CHECKPOINT}
BUNDLE=${STAIRS500_TRACKER_BUNDLE:?set STAIRS500_TRACKER_BUNDLE}
OUTPUT=${STAIRS500_TRACKER_OUTPUT:?set STAIRS500_TRACKER_OUTPUT}
PER_CLIP=${STAIRS500_TRACKER_PER_CLIP:-8}
NUM_ENVS=${STAIRS500_TRACKER_NUM_ENVS:-200}
MAX_ROUNDS=${STAIRS500_TRACKER_MAX_ROUNDS:-40}
DATASET_NAME=${STAIRS500_TRACKER_DATASET_NAME:-stairs500_maneuver_noised_pilot}

test -r "${CHECKPOINT}/config.yaml"
test -r "${CHECKPOINT}/last.pt"
test -r "${BUNDLE}/motion_lib_merged.pkl"
test -r "${BUNDLE}/objects.pkl"
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing to overwrite ${OUTPUT}" >&2
    exit 9
fi

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=Yes
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GRAIL_ROOT}:${PYTHONPATH:-}"
source "${MM_ROOT}/sonic/setup_writable_isaac_runtime.sh"
cd "${GRAIL_ROOT}"

python -u -m sonic_port.collect.collect_sonic_multiclip \
    --ckpt_dir "${CHECKPOINT}" \
    --terrain_motion_dir "${BUNDLE}" \
    --motion_file "${BUNDLE}/motion_lib_merged.pkl" \
    --out_dir "${OUTPUT}" \
    --per_clip "${PER_CLIP}" \
    --num_envs "${NUM_ENVS}" \
    --max_rounds "${MAX_ROUNDS}" \
    --episode_length_s 20 \
    --max_mpjpe_mm 300 \
    --store_dtype float32 \
    --dataset_name "${DATASET_NAME}"

test -f "${OUTPUT}/collect_summary.json"
