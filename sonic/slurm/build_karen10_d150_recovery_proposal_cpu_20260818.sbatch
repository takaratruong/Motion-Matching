#!/bin/bash
# Convert the accepted Karen10 route and match the d150/seed1 pre-fall states.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=karen10-proposal
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/proposal-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/proposal-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
TRACE=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_campaign_v1/baseline_traces/karen10_d150_seed1/recovery_trace.npz
REFERENCE=$ROOT/karen10_reference.npz
PROPOSAL=$ROOT/karen10_d150_seed1_proposal.json
SCENE=$REPO/sonic/configs/karen10_d150_recovery_scene.json

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
[[ -s "$ROOT/route/summary.json" && -s "$TRACE" && -s "$SCENE" ]]
MOTION=$(jq -er '.motion' "$ROOT/route/summary.json")
[[ -s "$MOTION" ]]

"$PYTHON" -m mm_sonic.build_synthesized_recovery_reference \
  --motion "$MOTION" --output "$REFERENCE"
exec "$PYTHON" -u -m mm_sonic.justin_recovery_cli \
  --trace "$TRACE" --first-fall-physics-step 2491 \
  --reference "$REFERENCE" --allow-single-reference \
  --scene-contract "$SCENE" --top-k 16 --out "$PROPOSAL"
