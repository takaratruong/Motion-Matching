#!/bin/bash
# Add contact-preserving stop/restart and reversal to reviewed stair motions.

#SBATCH --job-name=stairs500-dir-transition
#SBATCH --account=move
#SBATCH --partition=sc-freecpu,move
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/move/u/bodow/Projects/Motion-Matching-takara-corpus/slurm-stairs500-dir-transition-%A_%a.out

set -euo pipefail

PROJECT=/move/u/bodow/Projects/Motion-Matching-takara-corpus
SELECTION=${STAIRS500_DIRECTIONAL_TRANSITION_SELECTION:?set STAIRS500_DIRECTIONAL_TRANSITION_SELECTION}
OUTPUT=${STAIRS500_DIRECTIONAL_TRANSITION_OUTPUT:?set STAIRS500_DIRECTIONAL_TRANSITION_OUTPUT}
ARCHIVE=/move/data/terrain-aware/motion-matching/grail-stairs500-clean-50hz-v1.zarr
MODEL=/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml
PYTHON=/move/u/bodow/miniconda3/envs/cloc3/bin/python

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

"${PYTHON}" -u -B -m mm_sonic.build_terrain_maneuver_pilots \
    --source-motion "${MOTION}" \
    --terrain-usd "${TERRAIN}" \
    --output "${DESTINATION}" \
    --motion-archive "${ARCHIVE}" \
    --model-path "${MODEL}" \
    --mode stop_restart \
    --mode reverse \
    --ramp-frames 24 \
    --hold-frames 20 \
    --reverse-ramp-frames 20 \
    --reverse-hold-frames 20 \
    --pivot-speed-mps 0.20 \
    --maximum-stance-run-drift-m 0.020
