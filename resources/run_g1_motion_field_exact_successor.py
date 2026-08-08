#!/usr/bin/env python3
"""Append an exact connector and successor to an existing motion-field edge."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).with_name("run_g1_motion_field_transitions.py")
    spec = importlib.util.spec_from_file_location("motion_field_transitions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy()
            for name in (
                "joint_position",
                "root_position_world",
                "root_orientation_world_wxyz",
                "source_support_mask",
            )
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--connector", type=Path, required=True)
    parser.add_argument("--successor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = _module().assemble_exact_successor_edge(
        incoming_route=_route(args.incoming),
        connector=_route(args.connector),
        successor=_route(args.successor),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **route)
    print(f"wrote {args.output} ({len(route['joint_position'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
