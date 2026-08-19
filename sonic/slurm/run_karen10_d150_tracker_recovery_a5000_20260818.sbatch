#!/bin/bash
# Compare fixed-rate and adaptive-clock tracker recovery on the step-8 failure.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a5000:1
# Isaac startup uses a shared shader/cache path and can exhaust an A5000 when
# two instances initialize together on the same node.  Run the modes serially.
#SBATCH --array=0-1%1
#SBATCH --job-name=karen10-track
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/tracker-%A_%a.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/tracker-%A_%a.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
TRACKER=/move/u/justingu/Projects/grail-stairs
PYTHON=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
PROPOSAL=${PROPOSAL:-$ROOT/karen10_d150_seed1_successref_proposal.json}
USD=$ROOT/karen10_rigid.usda
OUTPUT_PREFIX=${OUTPUT_PREFIX:-attempt_d150_seed1_successref}

QUERY=$(jq -er '[.queries | to_entries[] | select(.value.candidates | length > 0)][0].key' "$PROPOSAL")
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  MODE=fixed
  MODE_ARGS=()
else
  MODE=adaptive
  MODE_ARGS=(--adaptive-reference-clock --clock-freeze-root-error-m 0.20 --clock-resume-root-error-m 0.12 --max-consecutive-clock-hold-steps 1)
fi
OUTPUT=$ROOT/${OUTPUT_PREFIX}_${MODE}
[[ ! -e "$OUTPUT" && -s "$USD" && -x "$PYTHON" ]]

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/sonic/python:$TRACKER:$TRACKER/GRAIL/imports/SONIC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
KIT_CACHE=/tmp/bodow-karen10-track-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$KIT_CACHE"
cd "$REPO"

exec "$PYTHON" -u -m mm_sonic.justin_tracker_recovery \
  --proposal "$PROPOSAL" --query-index "$QUERY" --candidate-index 0 \
  --out "$OUTPUT" --object-usd "$USD" \
  --tracker-repo "$TRACKER" \
  --checkpoint-dir "$TRACKER/data/sonic_checkpoints/terrain_release" \
  --device cuda:0 --seed "$SLURM_ARRAY_TASK_ID" --episode-length-s 20 \
  --reference-blend 0 --exact-learner-handoff \
  --kit-cache "$KIT_CACHE" "${MODE_ARGS[@]}"
