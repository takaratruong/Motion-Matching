#!/bin/bash
# Build one paired motion+terrain directional family per C490 archive clip.

#SBATCH --job-name=c490-cowarp
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-c490-cowarp-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ARCHIVE=${C490_COWARP_ARCHIVE:-/move/data/terrain-aware/motion-matching/grail-c490-curb-slope-clean-50hz-v1.zarr}
MODEL=${C490_COWARP_MODEL:-/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml}
OUTPUT=${C490_COWARP_OUTPUT:-/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_general_v1/c490_cowarped_directional_v1}
MODE_LIST=${C490_COWARP_MODES:-cowarp_diagonal_left:cowarp_diagonal_right:cowarp_lane_left:cowarp_lane_right:cowarp_slalom_left_right:cowarp_slalom_right_left}
AMPLITUDE=${C490_COWARP_AMPLITUDE_M:-0.65}
RENDER=${C490_COWARP_RENDER:-0}
RENDER_STRIDE=${C490_COWARP_RENDER_STRIDE:-2}
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MODE_ARGUMENTS=()
IFS=: read -r -a MODES <<< "${MODE_LIST}"
for MODE in "${MODES[@]}"; do
    MODE_ARGUMENTS+=(--mode "${MODE}")
done
if [[ "${RENDER}" == "1" ]]; then
    MODE_ARGUMENTS+=(--render --render-frame-stride "${RENDER_STRIDE}")
fi

"${PYTHON}" -u -B -m mm_sonic.build_c490_cowarped_directional \
    --archive "${ARCHIVE}" \
    --model "${MODEL}" \
    --clip-index "${SLURM_ARRAY_TASK_ID}" \
    --output-root "${OUTPUT}" \
    --maximum-amplitude-m "${AMPLITUDE}" \
    "${MODE_ARGUMENTS[@]}"
