#!/bin/bash
# Short hard-example pass over the nine failures from the first deterministic
# full-bank evaluation.  Always re-evaluate the complete 28-clip bank afterward.

#SBATCH --job-name=terr-man-hard9
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-maneuver-hard9-%j.out

set -uo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1
PARENT=${TERRAIN_MANEUVER_PARENT:-${RUN_ROOT}/sonic_finetune_v1}
BUNDLE=${TERRAIN_MANEUVER_HARD_BUNDLE:-${RUN_ROOT}/hard9_bundle_v1}
OUTPUT=${TERRAIN_MANEUVER_HARD_OUTPUT:-${RUN_ROOT}/sonic_hard9_finetune_v1}
LOG=/tmp/terrain_maneuver_hard9_${SLURM_JOB_ID}.log
ITERATIONS=${TERRAIN_MANEUVER_HARD9_ITERS:-200}
NUM_ENVS=${TERRAIN_MANEUVER_HARD9_ENVS:-192}

if [[ ! -r "${PARENT}/last.pt" || ! -r "${PARENT}/config.yaml" ]]; then
    echo "missing completed 28-clip parent checkpoint" >&2
    exit 8
fi
if [[ ! -r "${BUNDLE}/motion_lib_merged.pkl" || ! -r "${BUNDLE}/objects.pkl" ]]; then
    echo "missing nine-failure hard-example bundle" >&2
    exit 8
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "refusing to overwrite ${OUTPUT}" >&2
    exit 9
fi

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
    exp_base=terrain_maneuver_hard9 exp_var=bank28_hard9 timestamp=hard9 \
    env=terrain terrain=flat \
    checkpoint="${PARENT}/last.pt" \
    experiment_dir="${OUTPUT}" \
    num_envs="${NUM_ENVS}" multi_gpu=false num_gpus=1 headless=true use_wandb=false \
    "++algo.config.num_learning_iterations=${ITERATIONS}" \
    '++algo.config.save_interval=50' \
    callbacks.model_save.save_frequency=50 \
    manager_env.config.terrain_motion_dir="${BUNDLE}" \
    manager_env.config.flat_usd_path="${BUNDLE}/flat_placeholder.usd" \
    '++manager_env.config.flat_to_terrain_ratio=0' \
    manager_env.commands.motion.motion_lib_cfg.motion_file="${BUNDLE}/robot" \
    manager_env.commands.motion.motion_lib_cfg.object_motion_file="${BUNDLE}/objects.pkl" \
    2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}

if grep -qE 'Traceback \(most recent call last\)' "${LOG}"; then
    echo "hard-example fine-tune emitted a traceback" >&2
    rc=1
fi
if [[ "${rc}" -eq 0 ]] && ! find "${OUTPUT}" -type f -name 'model_*.pt' -print -quit | grep -q .; then
    echo "hard-example fine-tune wrote no checkpoint" >&2
    rc=1
fi
exit "${rc}"
