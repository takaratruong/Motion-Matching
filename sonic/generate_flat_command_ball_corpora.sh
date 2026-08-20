#!/usr/bin/env bash

# Recreate the clean Takara/BONES command-ball corpora without SONIC.
#
# Required environment variables:
#   BONES_SOURCE_ZARR  clean retargeted BONES 50 Hz G1 zarr
#   TAKARA_WALK_NPZ    legacy TakaraWalk 50 Hz NPZ (body_quat_w is XYZW)
#   MOTION_BANK_DIR    new output directory for the native WXYZ support bank
#   CORPUS_ROOT        parent directory for the two clean corpus run roots
#
# Optional:
#   PYTHON_BIN         Python with numpy, torch, and zarr 2.x (default: python3)
#   CORPUS_DEVICE      Torch device used by motion matching (default: cpu)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

: "${BONES_SOURCE_ZARR:?set BONES_SOURCE_ZARR to locomotion_50hz.zarr}"
: "${TAKARA_WALK_NPZ:?set TAKARA_WALK_NPZ to takara_walk_50hz.npz}"
: "${MOTION_BANK_DIR:?set MOTION_BANK_DIR to a new support-bank directory}"
: "${CORPUS_ROOT:?set CORPUS_ROOT to the clean corpus output parent}"

PYTHON_BIN=${PYTHON_BIN:-python3}
CORPUS_DEVICE=${CORPUS_DEVICE:-cpu}

HUMAN_RUN_ROOT="$CORPUS_ROOT/takara_bones_mm_human_full_v1_task12"
OMNI_RUN_ROOT="$CORPUS_ROOT/takara_bones_mm_omnidirectional_v1_task12"
HUMAN_CORPUS="$HUMAN_RUN_ROOT/_source/takara_bones_mm_human_full.zarr"
OMNI_CORPUS="$OMNI_RUN_ROOT/_source/takara_bones_mm_omnidirectional.zarr"

if [[ ! -d "$BONES_SOURCE_ZARR" ]]; then
    echo "missing BONES source zarr: $BONES_SOURCE_ZARR" >&2
    exit 8
fi
if [[ ! -f "$TAKARA_WALK_NPZ" ]]; then
    echo "missing TakaraWalk source NPZ: $TAKARA_WALK_NPZ" >&2
    exit 8
fi
for output in "$MOTION_BANK_DIR" "$HUMAN_CORPUS" "$OMNI_CORPUS"; do
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite reproduction output: $output" >&2
        exit 9
    fi
done

export PYTHONPATH="$MM_ROOT/sonic/python${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import numpy
import torch
import zarr

major = int(zarr.__version__.split(".", maxsplit=1)[0])
if major != 2:
    raise SystemExit(f"this corpus writer requires zarr 2.x, found {zarr.__version__}")
print(f"numpy={numpy.__version__} torch={torch.__version__} zarr={zarr.__version__}")
PY

mkdir -p "$(dirname "$MOTION_BANK_DIR")" \
    "$HUMAN_RUN_ROOT/_source" \
    "$OMNI_RUN_ROOT/_source"

"$PYTHON_BIN" -u -m mm_sonic.export_bones_motion_bank \
    --source-zarr "$BONES_SOURCE_ZARR" \
    --include-legacy-takara-npz "$TAKARA_WALK_NPZ" \
    --output-dir "$MOTION_BANK_DIR"

COMMON_CORPUS_ARGS=(
    --motions-dir "$MOTION_BANK_DIR"
    --device "$CORPUS_DEVICE"
    --frames 600
    --source-quaternion-convention wxyz
    --pelvis-velocity-weight 1.0
    --trajectory-position-weight 1.0
    --trajectory-facing-weight 1.5
    --trajectory-model takara_ball
    --max-source-joint-step-rad 0.35
)

"$PYTHON_BIN" -u -m mm_sonic.offline_corpus \
    "${COMMON_CORPUS_ARGS[@]}" \
    --output "$HUMAN_CORPUS"

"$PYTHON_BIN" -u -m mm_sonic.offline_corpus \
    "${COMMON_CORPUS_ARGS[@]}" \
    --curriculum omnidirectional \
    --output "$OMNI_CORPUS"

"$PYTHON_BIN" -u -m mm_sonic.analyze_corpus \
    --corpus "$HUMAN_CORPUS" \
    --output "$HUMAN_RUN_ROOT/clean_corpus_analysis.json"
"$PYTHON_BIN" -u -m mm_sonic.analyze_corpus \
    --corpus "$OMNI_CORPUS" \
    --output "$OMNI_RUN_ROOT/clean_corpus_analysis.json"

echo "clean human corpus: $HUMAN_CORPUS"
echo "clean omnidirectional corpus: $OMNI_CORPUS"
