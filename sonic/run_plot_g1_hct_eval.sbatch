#!/bin/bash
#SBATCH --job-name=g1-hct-plot
#SBATCH --partition=sc-freecpu
#SBATCH --account=move
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/move/data/terrain-aware/g1-hct-omni/slurm/plot-%j.out

set -euo pipefail

: "${SOURCE:?set SOURCE to an evaluation NPZ}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the figure directory}"

REPO=/move/u/bodow/Projects/Motion-Matching-takara-corpus
PY=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$REPO/sonic/python"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -u -m mm_sonic.plot_g1_hct_eval \
  --source "$SOURCE" \
  --output-dir "$OUTPUT_DIR"

test -s "$OUTPUT_DIR/intended_vs_actual_routes.png"
test -s "$OUTPUT_DIR/terrain_performance.png"
