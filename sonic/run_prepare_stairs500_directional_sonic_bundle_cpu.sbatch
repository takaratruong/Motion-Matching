#!/bin/bash
# Package exact-audited directional stairs and admitted transitions for SONIC.

#SBATCH --job-name=stairs500-dir-bundle
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-dir-bundle-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
GRAIL_ROOT=/move/u/justingu/Projects/grail-stairs
SELECTION=${STAIRS500_DIRECTIONAL_BUNDLE_SELECTION:?set STAIRS500_DIRECTIONAL_BUNDLE_SELECTION}
REPLAY_SELECTION=${STAIRS500_DIRECTIONAL_BUNDLE_REPLAY_SELECTION:-}
TRANSITIONS=${STAIRS500_DIRECTIONAL_BUNDLE_TRANSITIONS:?set STAIRS500_DIRECTIONAL_BUNDLE_TRANSITIONS}
OUTPUT=${STAIRS500_DIRECTIONAL_BUNDLE_OUTPUT:?set STAIRS500_DIRECTIONAL_BUNDLE_OUTPUT}
PYTHON=/move/u/bodow/Projects/reference-nvidia-groot-wbc/.venv_motionbricks/bin/python

export PYTHONPATH="${PROJECT}/sonic/python:${GRAIL_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SELECTION_ARGUMENTS=(--selection-file "${SELECTION}")
if [[ -n "${REPLAY_SELECTION}" ]]; then
    SELECTION_ARGUMENTS+=(--selection-file "${REPLAY_SELECTION}")
fi

"${PYTHON}" -u -B -m mm_sonic.prepare_terrain_maneuver_sonic_bundle \
    "${SELECTION_ARGUMENTS[@]}" \
    --manifest-root "${TRANSITIONS}" \
    --output "${OUTPUT}"
