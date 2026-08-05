#!/bin/bash
# Compile the accepted mixed-scene primitives into browser-runtime portals.

#SBATCH --job-name=generic-mixed-portals
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-generic-mixed-portals-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROUTE_SUMMARY=${ROUTE_SUMMARY:-${PROJECT}/artifacts/generic_terrain/unified_mixed_smoke_v1/generation/summary.json}
PORTAL_OUTPUT=${PORTAL_OUTPUT:-${PROJECT}/artifacts/generic_terrain/unified_mixed_smoke_v1/portals}
APPROACH_RAW=${APPROACH_RAW:-${PROJECT}/artifacts/global_scene_terrain/motionbricks_course_target26_angle_neg45_v1/generation/approach_raw.npz}
EXIT_RAW=${EXIT_RAW:-${PROJECT}/artifacts/global_scene_terrain/motionbricks_course_target26_pos45_v1/reused_generation/exit_raw.npz}
export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa
export XDG_CACHE_HOME=/tmp/generic-mixed-portals-${SLURM_JOB_ID}

/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.compose_generic_terrain_portals \
    --route-summary "${ROUTE_SUMMARY}" \
    --output-dir "${PORTAL_OUTPUT}" \
    --approach-raw "${APPROACH_RAW}" \
    --exit-raw "${EXIT_RAW}"
