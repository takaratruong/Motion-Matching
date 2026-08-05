#!/bin/bash
# Compile the reverse traversal of every obstacle in the mixed gauntlet.

#SBATCH --job-name=generic-reverse-portals
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-generic-reverse-portals-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
FORWARD=${PROJECT}/artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_v1
ROOT=${PROJECT}/artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_reverse_v1
export PYTHONPATH="${PROJECT}/sonic/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa
export XDG_CACHE_HOME=/tmp/generic-reverse-portals-${SLURM_JOB_ID}

cd "${PROJECT}"
/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.generate_generic_terrain_waypoint_route \
    --terrain-usd "${FORWARD}/terrain.usda" \
    --waypoint-xy 18.500  0.300 \
    --waypoint-xy 17.200 -0.500 \
    --waypoint-xy 14.800 -0.500 \
    --waypoint-xy 13.250  0.700 \
    --waypoint-xy 11.800  0.700 \
    --waypoint-xy 10.750 -0.600 \
    --waypoint-xy  9.300 -0.600 \
    --waypoint-xy  7.005  0.800 \
    --waypoint-xy  5.645  0.800 \
    --waypoint-xy  1.615 -0.800 \
    --waypoint-xy -0.615 -0.800 \
    --waypoint-xy -1.500 -0.800 \
    --primitive-cache-summary "${FORWARD}/generation/summary.json" \
    --output-dir "${ROOT}/generation"

/move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.compose_generic_terrain_portals \
    --route-summary "${ROOT}/generation/summary.json" \
    --output-dir "${ROOT}/portals_entry_v1"
