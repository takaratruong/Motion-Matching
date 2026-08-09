#!/bin/bash
#SBATCH --job-name=g1-hct-review
#SBATCH --partition=move
#SBATCH --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/move/data/terrain-aware/g1-hct-omni/slurm/review-%j.out

set -euo pipefail

: "${SOURCE:?set SOURCE to an evaluation NPZ}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the review directory}"

REPO=/move/u/bodow/Projects/Motion-Matching-takara-corpus
PY=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$REPO/sonic/python"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -u -m mm_sonic.render_g1_hct_review_suite \
  --source "$SOURCE" \
  --output-dir "$OUTPUT_DIR" \
  --stride 2

test -s "$OUTPUT_DIR/review_suite.json"
