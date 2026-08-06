#!/bin/bash
# Render selected episodes from a deterministic terrain-maneuver tracker gate.

#SBATCH --job-name=terr-man-video
#SBATCH --account=move
#SBATCH --partition=move,humanoid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-maneuver-video-%j.out

set -euo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROLLOUT=${TERRAIN_MANEUVER_REVIEW_ROLLOUT:?set TERRAIN_MANEUVER_REVIEW_ROLLOUT}
VIDEOS=${TERRAIN_MANEUVER_REVIEW_VIDEOS:?set TERRAIN_MANEUVER_REVIEW_VIDEOS}
EPISODES=${TERRAIN_MANEUVER_REVIEW_EPISODES:-}
EPISODES=${EPISODES//:/,}

if [[ ! -r "${ROLLOUT}/collect_summary.json" ]]; then
    echo "missing completed terrain-maneuver rollout: ${ROLLOUT}" >&2
    exit 8
fi
if [[ -z "${EPISODES}" ]]; then
    EPISODE_COUNT=$(
        /move/u/bodow/miniconda3/envs/cloc3/bin/python -c \
            'import json, sys; print(int(json.load(open(sys.argv[1]))["episodes"]))' \
            "${ROLLOUT}/collect_summary.json"
    )
    EPISODES=$(seq -s, 0 "$((EPISODE_COUNT - 1))")
fi
mkdir -p "${VIDEOS}"

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate isaac6_test
export OMNI_KIT_ACCEPT_EULA=Yes
export ENABLE_CAMERAS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GRAIL_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export ISAAC_SHARED_ROOT_OVERRIDE=/move/u/justingu/miniconda3/envs/isaac6_test/lib/python3.12/site-packages/isaacsim
source "${MM_ROOT}/sonic/setup_writable_isaac_runtime.sh"
cd "${GRAIL_ROOT}"

python -u -m sonic_port.collect.render_thirdperson_sonic \
    --dataset "${ROLLOUT}" \
    --episodes "${EPISODES}" \
    --out_dir "${VIDEOS}" \
    --max_frames 0 \
    --fps 25
