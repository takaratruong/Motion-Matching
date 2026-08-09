#!/bin/bash
#SBATCH --job-name=g1-hct-omni
#SBATCH --partition=move
#SBATCH --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/move/data/terrain-aware/g1-hct-omni/slurm/train-%j.out

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-takara-corpus
RUN_ROOT=/move/data/terrain-aware/g1-hct-omni/train
PY=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
CACHE_ROOT=${TMPDIR:-/tmp}/g1-hct-omni-${SLURM_JOB_ID}
LOAD_RUN=${LOAD_RUN:-seed_upright}
CHECKPOINT=${CHECKPOINT:-model_2999.pt}
MAX_ITERATIONS=${MAX_ITERATIONS:-1000}
RUN_NAME=${RUN_NAME:-omni_hct_v1}
SEED=${SEED:-42}
TASK=${TASK:-Terrain-G1-HCT-Omni-v0}

cd "$RUN_ROOT"
export PYTHONPATH="$REPO/sonic/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMNI_CACHE_DIR="$CACHE_ROOT/omni"
export OMNI_KIT_CACHE_DIR="$CACHE_ROOT/kit"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
mkdir -p "$OMNI_CACHE_DIR" "$OMNI_KIT_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$CACHE_ROOT/portable"

"$PY" -u -m mm_sonic.train_g1_hct_omni \
  --task "$TASK" \
  --headless \
  --num_envs 4096 \
  --max_iterations "$MAX_ITERATIONS" \
  --run_name "$RUN_NAME" \
  --seed "$SEED" \
  --resume \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT" \
  --kit_args "--portable-root $CACHE_ROOT/portable --reset-user"
