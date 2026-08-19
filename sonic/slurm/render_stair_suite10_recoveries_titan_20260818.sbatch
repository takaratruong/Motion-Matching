#!/bin/bash
# Render one dense baseline and every completed recovery mode for each case.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --array=0-24%4
#SBATCH --job-name=stair10-render
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%A_%a.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected

mapfile -t PROPOSALS < <(find "$ROOT/scenes" -path '*/proposals_refcmd/*.json' -type f | sort)
INDEX=${SLURM_ARRAY_TASK_ID}
if (( INDEX >= ${#PROPOSALS[@]} )); then
  echo "No proposal at case index $INDEX; intentional unused array slot."
  exit 0
fi
PROPOSAL=${PROPOSALS[$INDEX]}
SCENE_ROOT=$(dirname "$(dirname "$PROPOSAL")")
SCENE_ID=$(basename "$SCENE_ROOT")
CASE_ID=$(basename "$PROPOSAL" .json)
VIDEO_ROOT=$ROOT/recovery_videos/$SCENE_ID/$CASE_ID
mkdir -p "$VIDEO_ROOT"

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl

RENDERED_BEFORE=0
RENDERED_MODES=0
for MODE in exact blend25 blend50; do
  ATTEMPT=$ROOT/recovery_attempts/$SCENE_ID/${CASE_ID}_${MODE}
  if [[ ! -s "$ATTEMPT/receipt.json" || ! -s "$ATTEMPT/tracker_rollout.npz" ]]; then
    echo "Skipping incomplete attempt $ATTEMPT"
    continue
  fi
  BEFORE_ARGS=()
  if (( RENDERED_BEFORE == 0 )); then
    BEFORE_ARGS=(--before-out "$VIDEO_ROOT/before_dense.mp4")
  fi
  "$PYTHON" -u -m mm_sonic.render_justin_recovery_comparison \
    --proposal "$PROPOSAL" --attempt "$ATTEMPT" \
    "${BEFORE_ARGS[@]}" \
    --after-out "$VIDEO_ROOT/after_${MODE}.mp4" \
    --metadata-out "$VIDEO_ROOT/comparison_${MODE}.json" \
    --width 960 --height 540 --frame-stride 2
  RENDERED_BEFORE=1
  RENDERED_MODES=$((RENDERED_MODES + 1))
done

if (( RENDERED_MODES == 0 )); then
  echo "No completed recovery attempt for $CASE_ID"
  exit 0
fi
echo "Rendered $RENDERED_MODES recovery modes for $CASE_ID into $VIDEO_ROOT"
