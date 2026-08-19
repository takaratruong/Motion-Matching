#!/bin/bash
# Match the d150 failure against every successful varied-approach rollout.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=k10-multiref
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/multiref-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/multiref-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
CAMPAIGN=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_campaign_v1/baseline_traces
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
FAILED_TRACE=$CAMPAIGN/karen10_d150_seed1/recovery_trace.npz
PROPOSAL=$ROOT/karen10_d150_seed1_multisuccess_proposal.json
SCENE=$REPO/sonic/configs/karen10_d150_recovery_scene.json
SUCCESS_CASES=(d075_seed2 d100_seed1 d100_seed2 d300_seed1 d300_seed2)
SUCCESS_FRONTS=(0.75 1.0 1.0 3.0 3.0)

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
[[ -s "$FAILED_TRACE" && -s "$SCENE" ]]

REFERENCE_ARGS=()
for index in "${!SUCCESS_CASES[@]}"; do
  case_name=${SUCCESS_CASES[$index]}
  reference=$ROOT/karen10_success_${case_name}_reference_full.npz
  "$PYTHON" -m mm_sonic.build_trace_recovery_reference \
    --trace "$CAMPAIGN/karen10_${case_name}/recovery_trace.npz" \
    --output "$reference" \
    --learner-front-xy "${SUCCESS_FRONTS[$index]}" 0.0 \
    --reference-front-xy 0.0 0.0 --stop-physics-step 5900
  REFERENCE_ARGS+=(--reference "$reference")
done

exec "$PYTHON" -u -m mm_sonic.justin_recovery_cli \
  --trace "$FAILED_TRACE" --first-fall-physics-step 2491 \
  "${REFERENCE_ARGS[@]}" --scene-contract "$SCENE" --top-k 16 \
  --out "$PROPOSAL"
