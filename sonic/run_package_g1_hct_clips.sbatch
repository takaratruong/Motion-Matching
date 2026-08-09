#!/bin/bash
#SBATCH --job-name=g1-hct-package
#SBATCH --partition=sc-freecpu
#SBATCH --account=move
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/move/data/terrain-aware/g1-hct-omni/slurm/package-%j.out

set -euo pipefail

: "${SOURCE:?set SOURCE to an exact-terrain evaluation NPZ}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the collection directory}"

REPO=/move/u/bodow/Projects/Motion-Matching-takara-corpus
PY=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$REPO/sonic/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -u -m mm_sonic.package_g1_hct_clips \
  --source "$SOURCE" \
  --output-dir "$OUTPUT_DIR" \
  --drop-seconds 1 \
  --expected-clips 200 \
  --workers 4

test -s "$OUTPUT_DIR/manifest.json"
test "$(find "$OUTPUT_DIR/clips" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 200
