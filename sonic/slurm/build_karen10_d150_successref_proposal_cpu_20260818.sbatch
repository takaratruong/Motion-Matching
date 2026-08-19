#!/bin/bash
# Use a successful d100 learner ascent as a control reference for d150 recovery.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=k10-successref
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/successref-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/successref-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
CAMPAIGN=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_campaign_v1/baseline_traces
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
SUCCESS_TRACE=$CAMPAIGN/karen10_d100_seed1/recovery_trace.npz
FAILED_TRACE=$CAMPAIGN/karen10_d150_seed1/recovery_trace.npz
REFERENCE=$ROOT/karen10_success_d100_seed1_reference.npz
PROPOSAL=$ROOT/karen10_d150_seed1_successref_proposal.json
SCENE=$REPO/sonic/configs/karen10_d150_recovery_scene.json

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
[[ -s "$SUCCESS_TRACE" && -s "$FAILED_TRACE" && -s "$SCENE" ]]

"$PYTHON" -m mm_sonic.build_trace_recovery_reference \
  --trace "$SUCCESS_TRACE" --output "$REFERENCE" \
  --learner-front-xy 1.0 0.0 --reference-front-xy 0.0 0.0 \
  --stop-physics-step 3200
exec "$PYTHON" -u -m mm_sonic.justin_recovery_cli \
  --trace "$FAILED_TRACE" --first-fall-physics-step 2491 \
  --reference "$REFERENCE" --allow-single-reference \
  --scene-contract "$SCENE" --top-k 16 --out "$PROPOSAL"
