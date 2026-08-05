#!/bin/bash
# Five-event exact-mesh route: up-ramp, curb, stair ascent, stair descent,
# down-ramp, with live curved MotionBricks travel between portals.

#SBATCH --job-name=generic-gauntlet
#SBATCH --account=move
#SBATCH --partition=move,humanoid,sc-freecpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:07:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-generic-gauntlet-%j.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
ROOT=${TAKARA_GAUNTLET_ROOT:-${PROJECT}/artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_v1}
TERRAIN=${ROOT}/terrain.usda
OUTPUT=${ROOT}/generation
export PYTHONPATH="${PROJECT}/sonic/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa
export XDG_CACHE_HOME=/tmp/generic-gauntlet-${SLURM_JOB_ID}

cd "${PROJECT}"
python -B -m mm_sonic.build_generic_terrain_smoke_scene \
    --gauntlet \
    --output "${TERRAIN}"

timeout 5m /move/u/bodow/miniconda3/envs/cloc3/bin/python -u -B \
    -m mm_sonic.generate_generic_terrain_waypoint_route \
    --terrain-usd "${TERRAIN}" \
    --waypoint-xy -1.500 -0.800 \
    --waypoint-xy -0.615 -0.800 \
    --waypoint-xy  1.615 -0.800 \
    --waypoint-xy  5.645  0.800 \
    --waypoint-xy  7.005  0.800 \
    --waypoint-xy  9.300 -0.600 \
    --waypoint-xy 10.750 -0.600 \
    --waypoint-xy 11.800  0.700 \
    --waypoint-xy 13.250  0.700 \
    --waypoint-xy 14.800 -0.500 \
    --waypoint-xy 17.200 -0.500 \
    --waypoint-xy 18.500  0.300 \
    --primitive-cache-summary \
      "${PROJECT}/artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/summary.json" \
    --output-dir "${OUTPUT}"
