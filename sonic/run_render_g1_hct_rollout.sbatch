#!/bin/bash
#SBATCH --job-name=g1-hct-render
#SBATCH --partition=move
#SBATCH --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/move/data/terrain-aware/g1-hct-omni/slurm/render-%j.out

set -euo pipefail

: "${SOURCE:?set SOURCE to an evaluation NPZ}"
: "${OUTPUT:?set OUTPUT to an MP4 path}"

REPO=/move/u/bodow/Projects/Motion-Matching-takara-corpus
PY=/move/u/justingu/miniconda3/envs/env_isaaclab/bin/python
ARGS=(--source "$SOURCE" --output "$OUTPUT" --stride "${STRIDE:-2}")
if [[ -n "${ENV_INDEX:-}" ]]; then ARGS+=(--env "$ENV_INDEX"); fi
if [[ -n "${TERRAIN:-}" ]]; then ARGS+=(--terrain "$TERRAIN"); fi
if [[ -n "${PROGRAM:-}" ]]; then ARGS+=(--program "$PROGRAM"); fi

export PYTHONPATH="$REPO/sonic/python"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -u -m mm_sonic.render_g1_hct_rollout "${ARGS[@]}"
test -s "$OUTPUT"
