#!/bin/bash
# TASK_FILE columns: proposal, query_index, candidate_index, output_dir, seed.
# Each output directory is an exclusive ownership key.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-0%1
#SBATCH --job-name=justin-rematch
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/slurm_logs/%x-%A_%a.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/slurm_logs/%x-%A_%a.err

set -euo pipefail
: "${TASK_FILE:?export TASK_FILE with one tab-separated task per array index}"

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
TRACKER=/move/u/justingu/Projects/grail-stairs
PYTHON=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
[[ -s "$TASK_FILE" && -x "$PYTHON" ]]
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
IFS=$'\t' read -r proposal query_index candidate_index output seed <<< "$line"
[[ -n "$proposal" && -n "$query_index" && -n "$candidate_index" && -n "$output" && -n "$seed" ]]
[[ -s "$proposal" ]]
if [[ -e "$output" ]]; then
  echo "ownership collision: $output already exists" >&2
  exit 73
fi

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/sonic/python:$TRACKER:$TRACKER/GRAIL/imports/SONIC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
kit_cache=/tmp/bodow-justin-rematch-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$kit_cache"
cd "$REPO"

exec "$PYTHON" -u -m mm_sonic.justin_tracker_recovery \
  --proposal "$proposal" \
  --query-index "$query_index" \
  --candidate-index "$candidate_index" \
  --out "$output" \
  --tracker-repo "$TRACKER" \
  --checkpoint-dir "$TRACKER/data/sonic_checkpoints/terrain_release" \
  --device cuda:0 --seed "$seed" --episode-length-s 14 \
  --reference-blend 0 --exact-learner-handoff \
  --second-stage-rematch \
  --second-stage-stable-steps 5 \
  --second-stage-reference-start-frame 230 \
  --second-stage-reference-end-frame 270 \
  --second-stage-joint-weight-m-per-rad 0.10 \
  --second-stage-extra-steps 150 \
  --kit-cache "$kit_cache"
