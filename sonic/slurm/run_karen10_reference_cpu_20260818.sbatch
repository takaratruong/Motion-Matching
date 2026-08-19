#!/bin/bash
# Synthesize and collision-audit a tracker reference on the exact Karen10 mesh.
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=karen10-ref
#SBATCH --output=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/slurm-%j.out
#SBATCH --error=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference/slurm-%j.err

set -euo pipefail

REPO=/move/u/bodow/Projects/Motion-Matching-justin-recovery-test
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
OUTPUT=/move/u/bodow/outputs/justin_recovery_test/dense_epoch2_state41/karen10_tracker_reference
USD=$OUTPUT/karen10.usda
RIGID_USD=$OUTPUT/karen10_rigid.usda
CATALOG=$OUTPUT/stairs500_motion_route_profiles_v1.npz
ROUTE_DIR=${ROUTE_DIR:-$OUTPUT/route}
GENERATOR_ARGS=()
if [[ "${FAST_FALLBACK:-0}" == 1 ]]; then
  GENERATOR_ARGS=(
    --coverage-candidates-per-span 4
    --composition-candidates-per-span 1
    --accepted-candidates-per-span 1
    --maximum-candidate-chains 8
  )
fi

cd "$REPO"
mkdir -p "$OUTPUT"
export PYTHONPATH="$REPO/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

"$PYTHON" -m mm_sonic.build_karen10_recovery_scene \
  --output "$USD" --rigid-output "$RIGID_USD"
if [[ ! -s "$CATALOG" ]]; then
  "$PYTHON" -c 'import json,sys; from mm_sonic.terrain_oracle.stair_geometry_warp import build_motion_route_profile_catalog; print(json.dumps(build_motion_route_profile_catalog(sys.argv[1], sys.argv[2], worker_count=int(sys.argv[3])), indent=2))' \
    /move/data/terrain-aware/motion-matching/grail-stairs500-clean-50hz-v1.zarr \
    "$CATALOG" "${SLURM_CPUS_PER_TASK:-16}"
fi
exec "$PYTHON" -u -m mm_sonic.generate_generic_stair_route \
  --terrain-usd "$USD" \
  --start-xy -1.5 0.0 \
  --end-xy 3.65 0.0 \
  --route-catalog "$CATALOG" \
  --output-dir "$ROUTE_DIR" "${GENERATOR_ARGS[@]}"
