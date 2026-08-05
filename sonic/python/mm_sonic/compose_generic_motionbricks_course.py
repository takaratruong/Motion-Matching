"""Attach MotionBricks flat approach/exit motion to a mesh-only stair route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compose_motionbricks_terrain_course import compose
from .generate_generic_stair_route import load_target_mesh
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-usd", type=Path, required=True)
    parser.add_argument("--terrain-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument(
        "--terrain-quaternion-wxyz",
        type=float,
        nargs=4,
        default=(1, 0, 0, 0),
    )
    parser.add_argument("--stair-motion", type=Path, required=True)
    parser.add_argument("--approach-raw", type=Path, required=True)
    parser.add_argument("--exit-raw", type=Path, required=True)
    parser.add_argument("--stairs-archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/move/u/justingu/Projects/TWIST2/assets/g1/"
            "g1_29dof_rev_1_0.xml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-candidate-limit", type=int, default=4)
    parser.add_argument("--maximum-candidate-combinations", type=int, default=8)
    parser.add_argument(
        "--maximum-seam-foot-error-m", type=float, default=0.018
    )
    parser.add_argument("--forced-approach-phase-index", type=int)
    parser.add_argument("--forced-exit-phase-index", type=int)
    parser.add_argument(
        "--omit-exit",
        action="store_true",
        help="end at the authored traversal and let the runtime synthesize the exit",
    )
    parser.add_argument("--render", action="store_true")
    arguments = parser.parse_args(argv)
    mesh = load_target_mesh(
        arguments.terrain_usd,
        position_world=tuple(arguments.terrain_position),
        quaternion_world_from_usd_wxyz=tuple(
            arguments.terrain_quaternion_wxyz
        ),
    )
    result = compose(
        approach_raw=arguments.approach_raw.expanduser().resolve(),
        exit_raw=arguments.exit_raw.expanduser().resolve(),
        stair_motion=arguments.stair_motion.expanduser().resolve(),
        stairs_archive=arguments.stairs_archive.expanduser().resolve(),
        target_clip_index=None,
        target_mesh=mesh,
        model_path=arguments.model_path.expanduser().resolve(),
        output_dir=arguments.output_dir.expanduser().resolve(),
        phase_candidate_limit=arguments.phase_candidate_limit,
        maximum_candidate_combinations=(
            arguments.maximum_candidate_combinations
        ),
        maximum_seam_foot_error_m=arguments.maximum_seam_foot_error_m,
        forced_approach_phase_index=(
            arguments.forced_approach_phase_index
        ),
        forced_exit_phase_index=arguments.forced_exit_phase_index,
        include_exit=not arguments.omit_exit,
        render=arguments.render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
