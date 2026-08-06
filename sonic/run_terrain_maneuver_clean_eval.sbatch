#!/bin/bash
# Deterministic, one-rollout-per-clip diagnostic for the 28-motion terrain bank.
# This deliberately saves failed episodes so every failure can be rendered; its
# output is evaluation evidence only and must never be used as training data.

#SBATCH --job-name=terr-man-eval
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-maneuver-eval-%j.out

set -euo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1
BUNDLE=${RUN_ROOT}/bundle
TRAIN_OUTPUT=${RUN_ROOT}/sonic_finetune_v1
CHECKPOINT=${RUN_ROOT}/sonic_finetune_v1_eval_checkpoint
OUTPUT=${RUN_ROOT}/clean_eval_v1

if [[ ! -r "${BUNDLE}/motion_lib_merged.pkl" || ! -r "${BUNDLE}/objects.pkl" ]]; then
    echo "missing terrain maneuver bundle" >&2
    exit 8
fi
if [[ ! -r "${TRAIN_OUTPUT}/config.yaml" ]]; then
    echo "missing completed fine-tune config" >&2
    exit 8
fi
mapfile -t MODELS < <(find "${TRAIN_OUTPUT}" -maxdepth 1 -type f -name 'model_step_*.pt' | sort -V)
if (( ${#MODELS[@]} == 0 )); then
    echo "fine-tune wrote no model checkpoint" >&2
    exit 8
fi
LATEST=${MODELS[$((${#MODELS[@]} - 1))]}
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing to overwrite ${OUTPUT}" >&2
    exit 9
fi
if [[ -e "${CHECKPOINT}" ]]; then
    echo "refusing to overwrite ${CHECKPOINT}" >&2
    exit 9
fi
mkdir -p "${CHECKPOINT}"
ln -s "${TRAIN_OUTPUT}/config.yaml" "${CHECKPOINT}/config.yaml"
ln -s "${LATEST}" "${CHECKPOINT}/last.pt"
echo "evaluating ${LATEST}"

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=Yes
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${GRAIL_ROOT}:${PYTHONPATH:-}"
source "${MM_ROOT}/sonic/setup_writable_isaac_runtime.sh"
cd "${GRAIL_ROOT}"

python -u -m sonic_port.collect.collect_sonic_multiclip \
    --ckpt_dir "${CHECKPOINT}" \
    --terrain_motion_dir "${BUNDLE}" \
    --motion_file "${BUNDLE}/motion_lib_merged.pkl" \
    --out_dir "${OUTPUT}" \
    --per_clip 1 \
    --num_envs 28 \
    --max_rounds 1 \
    --episode_length_s 20 \
    --clean_collect \
    --accept_failed \
    --max_mpjpe_mm 10000 \
    --store_dtype float32 \
    --dataset_name terrain_maneuver_bank28_finetune_clean_diagnostic

test -f "${OUTPUT}/collect_summary.json"
