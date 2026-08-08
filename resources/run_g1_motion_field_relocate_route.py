#!/usr/bin/env python3
"""Rigidly relocate a terrain-valid authentic route footprint in XY."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).with_name("run_g1_motion_field_transitions.py")
    spec = importlib.util.spec_from_file_location("motion_field_transitions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--translation-x-m", type=float, required=True)
    parser.add_argument("--translation-y-m", type=float, required=True)
    parser.add_argument(
        "--during-first-support-transfer", action="store_true"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        window = slice(args.start_frame, args.stop_frame)
        route = {
            name: np.asarray(archive[name][window]).copy()
            for name in (
                "joint_position",
                "root_position_world",
                "root_orientation_world_wxyz",
                "source_support_mask",
            )
        }
    translation = (args.translation_x_m, args.translation_y_m)
    transition = _module()
    relocate = (
        transition.translate_route_xy_through_first_support_transfer
        if args.during_first_support_transfer
        else transition.translate_route_xy
    )
    relocated = relocate(route=route, translation_xy_m=translation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **relocated)
    record = {
        "schema": "g1-motion-field-rigid-route-relocation/v1",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "translation_xy_m": list(translation),
        "source_frames": [args.start_frame, args.stop_frame],
        "during_first_support_transfer": args.during_first_support_transfer,
        "pose_edited": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
