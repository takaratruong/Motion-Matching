#!/usr/bin/env python3
"""Build an exact same-stance connector between two route boundaries."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics


def _module():
    path = Path(__file__).with_name("run_g1_motion_field_transitions.py")
    spec = importlib.util.spec_from_file_location("motion_field_transitions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(path: Path, frame: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "joint": np.asarray(archive["joint_position"][frame]).copy(),
            "root": np.asarray(archive["root_position_world"][frame]).copy(),
            "orientation": np.asarray(
                archive["root_orientation_world_wxyz"][frame]
            ).copy(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--incoming-frame", type=int, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-frame", type=int, default=0)
    parser.add_argument("--stance-foot", choices=("left", "right"), required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    incoming = _frame(args.incoming, args.incoming_frame)
    target = _frame(args.target, args.target_frame)
    connector = _module().stance_anchored_boundary_connector(
        incoming_joint_position=incoming["joint"],
        incoming_root_position_world=incoming["root"],
        incoming_root_orientation_world_wxyz=incoming["orientation"],
        target_joint_position=target["joint"],
        target_root_position_world=target["root"],
        target_root_orientation_world_wxyz=target["orientation"],
        stance_foot=0 if args.stance_foot == "left" else 1,
        frame_count=args.frame_count,
        foot_kinematics=MujocoG1FootKinematics(args.g1_xml),
        sole_kinematics=MujocoG1SoleKinematics(args.g1_xml),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **connector)
    print(f"wrote {args.output} ({args.frame_count} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
