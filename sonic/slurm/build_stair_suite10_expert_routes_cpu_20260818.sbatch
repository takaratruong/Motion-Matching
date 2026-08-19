#!/bin/bash
# Generate collision-audited tracker references on ten exact stair geometries.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%4
#SBATCH --job-name=stair10-expert
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%A_%a.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected
CATALOG=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/stairs500_motion_route_profiles_v1.npz
SCENES=(
  stairset_01_karen3
  stairset_02_shallow4
  stairset_03_compact5
  stairset_04_karen5
  stairset_05_broad5
  stairset_06_steep4
  stairset_07_steepdeep5
  stairset_08_narrow5
  stairset_09_low7
  stairset_10_irregular6
)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
SCENE_ROOT=$ROOT/scenes/$SCENE
CONTRACT=$SCENE_ROOT/scene_contract.json
START_X=$(jq -er '.route.start_xy[0]' "$CONTRACT")
END_X=$(jq -er '.route.end_xy[0]' "$CONTRACT")

[[ -s "$SCENE_ROOT/terrain.usda" && -s "$CATALOG" ]]
mkdir -p "$ROOT/slurm_logs"
cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

exec "$PYTHON" -u -m mm_sonic.generate_generic_stair_route \
  --terrain-usd "$SCENE_ROOT/terrain.usda" \
  --start-xy "$START_X" 0.0 \
  --end-xy "$END_X" 0.0 \
  --route-catalog "$CATALOG" \
  --maximum-block-transitions 7 \
  --output-dir "$SCENE_ROOT/route"
