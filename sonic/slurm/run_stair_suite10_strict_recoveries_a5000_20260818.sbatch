#!/bin/bash
# Test exact and partially reference-aligned tracker handoffs on failed cases.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-74%4
#SBATCH --job-name=stair10-recover
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%A_%a.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
TRACKER=/move/u/justingu/Projects/grail-stairs
PYTHON=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected

mapfile -t PROPOSALS < <(find "$ROOT/scenes" -path '*/proposals_refcmd/*.json' -type f | sort)
CASE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODE_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
if (( CASE_INDEX >= ${#PROPOSALS[@]} )); then
  echo "No proposal at case index $CASE_INDEX; intentional unused array slot."
  exit 0
fi
PROPOSAL=${PROPOSALS[$CASE_INDEX]}
SCENE_ROOT=$(dirname "$(dirname "$PROPOSAL")")
OBJECT_USD=$SCENE_ROOT/terrain_rigid.usda
CASE_ID=$(basename "$PROPOSAL" .json)
SCENE_ID=$(basename "$SCENE_ROOT")
read -r QUERY CANDIDATE < <(
  jq -er '
    [.queries | to_entries[] as $query |
      $query.value.candidates | to_entries[] |
      {query: $query.key, candidate: .key, cost: .value.total_cost}]
    | min_by(.cost) | [.query, .candidate] | @tsv
  ' "$PROPOSAL"
)
case "$MODE_INDEX" in
  0) MODE=exact; BLEND=0; MODE_ARGS=(--exact-learner-handoff) ;;
  1) MODE=blend25; BLEND=0.25; MODE_ARGS=() ;;
  2) MODE=blend50; BLEND=0.50; MODE_ARGS=() ;;
esac
OUTPUT=$ROOT/recovery_attempts/${SCENE_ID}/${CASE_ID}_${MODE}
if [[ -s "$OUTPUT/receipt.json" && -s "$OUTPUT/tracker_rollout.npz" ]]; then
  echo "Skipping completed attempt $OUTPUT"
  exit 0
fi
[[ ! -e "$OUTPUT" && -s "$OBJECT_USD" && -x "$PYTHON" ]]

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/sonic/python:$TRACKER:$TRACKER/GRAIL/imports/SONIC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
KIT_CACHE=/tmp/bodow-stair10-recover-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$KIT_CACHE" "$(dirname "$OUTPUT")"
cd "$REPO"

exec "$PYTHON" -u -m mm_sonic.justin_tracker_recovery \
  --proposal "$PROPOSAL" --query-index "$QUERY" --candidate-index "$CANDIDATE" \
  --out "$OUTPUT" --object-usd "$OBJECT_USD" \
  --tracker-repo "$TRACKER" \
  --checkpoint-dir "$TRACKER/data/sonic_checkpoints/terrain_release" \
  --device cuda:0 --seed "$MODE_INDEX" --episode-length-s 20 \
  --reference-blend "$BLEND" --kit-cache "$KIT_CACHE" "${MODE_ARGS[@]}"
