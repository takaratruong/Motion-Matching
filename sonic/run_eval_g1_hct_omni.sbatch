#!/bin/bash
#SBATCH --job-name=g1-hct-eval
#SBATCH --partition=move
#SBATCH --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/move/data/terrain-aware/g1-hct-omni/slurm/eval-%j.out

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT to a model checkpoint}"
: "${OUTPUT:?set OUTPUT to the rollout NPZ path}"

REPO=/move/u/bodow/Projects/Motion-Matching-takara-corpus
PY=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
CACHE_ROOT=${TMPDIR:-/tmp}/g1-hct-eval-${SLURM_JOB_ID}
NUM_ENVS=${NUM_ENVS:-200}
STEPS=${STEPS:-600}
HCT_TERRAIN_SEED=${HCT_TERRAIN_SEED:-44}
HCT_TERRAIN_LEVEL=${HCT_TERRAIN_LEVEL:-9}

export PYTHONPATH="$REPO/sonic/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMNI_CACHE_DIR="$CACHE_ROOT/omni"
export OMNI_KIT_CACHE_DIR="$CACHE_ROOT/kit"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
mkdir -p "$OMNI_CACHE_DIR" "$OMNI_KIT_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$CACHE_ROOT/portable"

cd /move/data/terrain-aware/g1-hct-omni
"$PY" -u -m mm_sonic.evaluate_g1_hct_omni \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --num_envs "$NUM_ENVS" \
  --steps "$STEPS" \
  --terrain_seed "$HCT_TERRAIN_SEED" \
  --terrain_level "$HCT_TERRAIN_LEVEL" \
  --headless \
  --kit_args "--portable-root $CACHE_ROOT/portable --reset-user"

test -s "$OUTPUT"
test -s "${OUTPUT%.npz}.json"
