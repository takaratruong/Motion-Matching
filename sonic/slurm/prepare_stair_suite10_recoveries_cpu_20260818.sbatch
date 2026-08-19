#!/bin/bash
# Build strict references/proposals after all geometry route audits finish.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=stair10-prep
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected
BASELINE=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v1/baseline_traces

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$PYTHON" -u -m mm_sonic.prepare_stair_suite_recoveries \
  --suite-root "$ROOT" --baseline-root "$BASELINE"
exec "$PYTHON" -u -m mm_sonic.build_reference_command_stair_suite_manifest \
  --base-manifest "$REPO/../TML-BeyondMimic-justin-recovery-test/config/sonic_stair_suite10_campaign_v1.json" \
  --baseline-root "$BASELINE" --suite-root "$ROOT" \
  --output "$ROOT/reference_command_eval_manifest.json"
