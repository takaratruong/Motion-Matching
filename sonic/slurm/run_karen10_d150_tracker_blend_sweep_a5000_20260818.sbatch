#!/bin/bash
# Measure how far the d150 handoff must move toward its matched reference state.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-3%1
#SBATCH --job-name=k10-blend
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/blend-%A_%a.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/blend-%A_%a.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
TRACKER=/move/u/justingu/Projects/grail-stairs
PYTHON=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
PROPOSAL=${PROPOSAL:-$ROOT/karen10_d150_seed1_successref_proposal.json}
USD=$ROOT/karen10_rigid.usda
OUTPUT_PREFIX=${OUTPUT_PREFIX:-attempt_d150_seed1_successref_blend}
BLENDS=(0.25 0.50 0.75 1.00)
BLEND=${BLENDS[$SLURM_ARRAY_TASK_ID]}
LABEL=${BLEND/./p}
OUTPUT=$ROOT/${OUTPUT_PREFIX}_${LABEL}

QUERY=$(jq -er '[.queries | to_entries[] | select(.value.candidates | length > 0)][0].key' "$PROPOSAL")
[[ ! -e "$OUTPUT" && -s "$USD" && -x "$PYTHON" ]]

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/sonic/python:$TRACKER:$TRACKER/GRAIL/imports/SONIC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
KIT_CACHE=/tmp/bodow-karen10-blend-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$KIT_CACHE"
cd "$REPO"

exec "$PYTHON" -u -m mm_sonic.justin_tracker_recovery \
  --proposal "$PROPOSAL" --query-index "$QUERY" --candidate-index 0 \
  --out "$OUTPUT" --object-usd "$USD" \
  --tracker-repo "$TRACKER" \
  --checkpoint-dir "$TRACKER/data/sonic_checkpoints/terrain_release" \
  --device cuda:0 --seed 0 --episode-length-s 20 \
  --reference-blend "$BLEND" --kit-cache "$KIT_CACHE"
