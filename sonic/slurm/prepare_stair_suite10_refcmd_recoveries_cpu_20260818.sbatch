#!/bin/bash
# Match new command-parity failures to the exact expert references.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=stair10-refprep
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected/slurm_logs/%x-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
ROOT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/stair_suite10_v2_corrected
BASELINE=$ROOT/reference_command_baseline_traces

cd "$REPO"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

exec "$PYTHON" -u -m mm_sonic.prepare_stair_suite_recoveries \
  --suite-root "$ROOT" --baseline-root "$BASELINE" \
  --proposal-dirname proposals_refcmd \
  --summary-name preparation_refcmd_summary.json
