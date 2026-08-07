#!/bin/bash
# Train SONIC on the full exact-audited stairs500 directional corpus.
#
# Paired terrain training assigns env k to motion/terrain k.  Keep NUM_ENVS
# equal to the bundle clip count so every clip is present in every iteration;
# using fewer environments would silently expose only the leading subset.

#SBATCH --job-name=st500-omni-ft
#SBATCH --account=move
#SBATCH --partition=humanoid
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-st500-omni-ft-%j.out

set -uo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1
PARENT=${STAIRS500_OMNI_PARENT:-${RUN_ROOT}/sonic_selected200_finetune_v1}
BUNDLE=${STAIRS500_OMNI_BUNDLE:-${RUN_ROOT}/omnidirectional_sweep28_post_v1/sonic_bundle_replay200_all_transitions32_v1}
OUTPUT=${STAIRS500_OMNI_OUTPUT:-${RUN_ROOT}/sonic_omnidirectional_all1118_finetune_v1}
ITERATIONS=${STAIRS500_OMNI_ITERS:-400}
SAVE_INTERVAL=${STAIRS500_OMNI_SAVE_INTERVAL:-50}
LOG=/tmp/stairs500_omnidirectional_finetune_${SLURM_JOB_ID}.log

if [[ ! -r "${PARENT}/last.pt" || ! -r "${PARENT}/config.yaml" ]]; then
    echo "missing stable parent checkpoint: ${PARENT}" >&2
    exit 8
fi
if [[ ! -r "${BUNDLE}/motion_lib_merged.pkl" || ! -r "${BUNDLE}/objects.pkl" || ! -r "${BUNDLE}/clips.json" ]]; then
    echo "missing complete directional bundle: ${BUNDLE}" >&2
    exit 8
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing to overwrite ${OUTPUT}" >&2
    exit 9
fi

CLIP_COUNT=$(
    /move/u/bodow/miniconda3/envs/cloc3/bin/python -c \
        'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
        "${BUNDLE}/clips.json"
)
if (( CLIP_COUNT < 1 )); then
    echo "empty bundle: ${BUNDLE}" >&2
    exit 8
fi
# PPO splits one environment batch across four minibatches.  Round upward so
# every paired clip is represented while keeping that split exact; any extra
# environments wrap to the first clips rather than omitting the tail.
NUM_MINI_BATCHES=${STAIRS500_OMNI_NUM_MINI_BATCHES:-4}
NUM_ENVS=$(( (CLIP_COUNT + NUM_MINI_BATCHES - 1) / NUM_MINI_BATCHES * NUM_MINI_BATCHES ))
echo "paired coverage: ${CLIP_COUNT} clips -> ${NUM_ENVS} environments (${NUM_ENVS}-${CLIP_COUNT} wrapped repeats)"

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=Yes
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
source "${MM_ROOT}/sonic/setup_writable_isaac_runtime.sh"

cd "${GRAIL_ROOT}/GRAIL/imports/SONIC"
python -u -m gear_sonic.train_agent_trl \
    --config-dir="${PARENT}" --config-name=config hydra.job.chdir=false \
    hydra.run.dir="${OUTPUT}/hydra" \
    exp_base=stairs500_omnidirectional exp_var=all1118 timestamp=all1118 \
    env=terrain terrain=flat \
    checkpoint="${PARENT}/last.pt" \
    experiment_dir="${OUTPUT}" \
    num_envs="${NUM_ENVS}" multi_gpu=false num_gpus=1 headless=true use_wandb=false \
    "++algo.config.num_learning_iterations=${ITERATIONS}" \
    "++algo.config.save_interval=${SAVE_INTERVAL}" \
    "callbacks.model_save.save_frequency=${SAVE_INTERVAL}" \
    manager_env.config.terrain_motion_dir="${BUNDLE}" \
    manager_env.config.flat_usd_path="${BUNDLE}/flat_placeholder.usd" \
    '++manager_env.config.flat_to_terrain_ratio=0' \
    manager_env.commands.motion.motion_lib_cfg.motion_file="${BUNDLE}/robot" \
    manager_env.commands.motion.motion_lib_cfg.object_motion_file="${BUNDLE}/objects.pkl" \
    2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}

if grep -qE 'Traceback \(most recent call last\)|CUDA out of memory' "${LOG}"; then
    rc=1
fi
if [[ "${rc}" -eq 0 ]] && ! find "${OUTPUT}" -maxdepth 1 -type f -name 'model_step_*.pt' -print -quit | grep -q .; then
    echo "full directional fine-tune wrote no checkpoint" >&2
    rc=1
fi
exit "${rc}"
