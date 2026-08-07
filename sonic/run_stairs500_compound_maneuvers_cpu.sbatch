#!/bin/bash
# Add progress-distributed stops, reversals, and forward/backward bounces to
# exact-audited directional stair traversals.

#SBATCH --job-name=stairs500-compound
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-compound-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SELECTION=${STAIRS500_COMPOUND_SELECTION:?set STAIRS500_COMPOUND_SELECTION}
OUTPUT=${STAIRS500_COMPOUND_OUTPUT:?set STAIRS500_COMPOUND_OUTPUT}
ARCHIVE=/move/data/terrain-aware/motion-matching/grail-stairs500-clean-50hz-v1.zarr
MODEL=/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python
PIVOT_COUNT=${STAIRS500_COMPOUND_PIVOT_COUNT:-3}
MODE_LIST=${STAIRS500_COMPOUND_MODES:-stop_restart:reverse:bounce}

export PYTHONPATH="${PROJECT}/sonic/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

readarray -t VALUES < <(
    "${PYTHON}" - "${SELECTION}" "${SLURM_ARRAY_TASK_ID}" "${ARCHIVE}" <<'PY'
import json
from pathlib import Path
import sys

import zarr

selection = json.load(open(sys.argv[1]))
entry = selection[int(sys.argv[2])]
report_path = Path(entry["report"]).expanduser().resolve()
report = json.load(open(report_path))
motion = Path(str(report.get("motion", report_path.parent / "motion.npz")))
archive = zarr.open_group(sys.argv[3], mode="r")
clip_index = int(report["clip_index"])
print(motion.expanduser().resolve())
print(Path(str(archive["terrain_usd_path"][clip_index])).resolve())
print(str(report["label"]))
PY
)

MOTION=${VALUES[0]}
TERRAIN=${VALUES[1]}
LABEL=${VALUES[2]}
RANK=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
DESTINATION="${OUTPUT}/clip_${RANK}_${LABEL}/pilots"

MODE_ARGUMENTS=()
IFS=: read -r -a MODES <<< "${MODE_LIST}"
for MODE in "${MODES[@]}"; do
    MODE_ARGUMENTS+=(--mode "${MODE}")
done

"${PYTHON}" -u -B -m mm_sonic.build_terrain_maneuver_pilots \
    --source-motion "${MOTION}" \
    --terrain-usd "${TERRAIN}" \
    --output "${DESTINATION}" \
    --motion-archive "${ARCHIVE}" \
    --model-path "${MODEL}" \
    --pivot-count "${PIVOT_COUNT}" \
    "${MODE_ARGUMENTS[@]}" \
    --ramp-frames 16 \
    --hold-frames 4 \
    --reverse-ramp-frames 16 \
    --reverse-hold-frames 4 \
    --pivot-speed-mps 0.20 \
    --maximum-stance-run-drift-m 0.020
