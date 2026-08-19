#!/bin/bash
# Render the full dense-only and dense-to-tracker Karen10 rollouts offline.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=k10-render
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/render-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/render-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
PROPOSAL=${PROPOSAL:-$ROOT/karen10_d150_seed1_successref_proposal.json}
ATTEMPT=${ATTEMPT:?set ATTEMPT to a completed tracker attempt directory}

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
[[ -s "$PROPOSAL" && -s "$ATTEMPT/receipt.json" && -s "$ATTEMPT/tracker_rollout.npz" ]]

exec "$PYTHON" -u -m mm_sonic.render_justin_recovery_comparison \
  --proposal "$PROPOSAL" --attempt "$ATTEMPT" \
  --before-out "$ATTEMPT/whole_before_dense.mp4" \
  --after-out "$ATTEMPT/whole_after_recovery.mp4" \
  --metadata-out "$ATTEMPT/whole_comparison.json" \
  --width 960 --height 540 --frame-stride 1
