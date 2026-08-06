#!/bin/bash
# Render a deliberately stratified subset of the physical clean-bank diagnostic.

#SBATCH --job-name=terr-man-video
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --array=0-7%2
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-terrain-maneuver-review-%A_%a.out

set -euo pipefail

GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
MM_ROOT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1
ROLLOUT=${RUN_ROOT}/clean_eval_v1
VIDEOS=${RUN_ROOT}/clean_eval_v1_videos
TARGET_STEMS=(
    terrain_maneuver_pilot_backward_complete_stair_event2_v2_event_02_stairs_stop_restart
    terrain_maneuver_pilot_grounded_cached_stair_event2_v2_event_02_stairs_reverse
    terrain_maneuver_pilot_slope_registered_down118_v2_event_00_direct_terrain_reverse
    terrain_maneuver_pilot_slope_registered_down147_v2_event_00_direct_terrain_reverse
    clip_079_terrain_slopes__slope_019__006_pilots_event_00_direct_terrain_stop_restart
    clip_083_terrain_slopes__slope_030__009_pilots_event_00_direct_terrain_reverse
    clip_121_terrain_slopes__slope_127__006_pilots_event_00_direct_terrain_stop_restart
    clip_136_terrain_slopes__slope_176__009_pilots_event_00_direct_terrain_reverse
)
TARGET=${TARGET_STEMS[$SLURM_ARRAY_TASK_ID]}

if [[ ! -f "${ROLLOUT}/collect_summary.json" ]]; then
    echo "missing clean diagnostic: ${ROLLOUT}" >&2
    exit 8
fi
mkdir -p "${VIDEOS}"

# Completion order is not clip order, so resolve the episode from its stored
# clip identity instead of assuming an array index.
EPISODE=$(
    /move/u/bodow/miniconda3/envs/cloc3/bin/python - "${ROLLOUT}" "${TARGET}" <<'PY'
import glob
import sys

import numpy as np
import zarr

rollout, target = sys.argv[1:]
stores = sorted(glob.glob(f"{rollout}/*.zarr"))
if len(stores) != 1:
    raise SystemExit(f"expected one rollout zarr, found {stores}")
root = zarr.open(stores[0], mode="r")
clips = [str(value) for value in np.asarray(root["meta/episode_clip"])]
matches = [index for index, clip in enumerate(clips) if clip == target]
if len(matches) != 1:
    raise SystemExit(f"expected one episode for {target}, found {matches}")
print(matches[0])
PY
)

if compgen -G "${VIDEOS}/tp_ep${EPISODE}_*.mp4" >/dev/null; then
    echo "refusing duplicate render for episode ${EPISODE}" >&2
    exit 9
fi

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate isaac6_test
export OMNI_KIT_ACCEPT_EULA=Yes
export ENABLE_CAMERAS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GRAIL_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export ISAAC_SHARED_ROOT_OVERRIDE="${CONDA_PREFIX}/lib/python3.12/site-packages/isaacsim"
source "${MM_ROOT}/sonic/setup_writable_isaac_runtime.sh"
cd "${GRAIL_ROOT}"

python -u -m sonic_port.collect.render_thirdperson_sonic \
    --dataset "${ROLLOUT}" \
    --episodes "${EPISODE}" \
    --out_dir "${VIDEOS}" \
    --max_frames 0 \
    --fps 25

compgen -G "${VIDEOS}/tp_ep${EPISODE}_*.mp4" >/dev/null
