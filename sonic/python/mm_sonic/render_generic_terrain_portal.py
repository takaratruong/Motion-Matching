"""Render one course from a generic terrain portal manifest on its exact mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .generate_generic_stair_route import load_target_mesh
from .render_stitched_motion import load_stitched_motion_npz, render_stitched_motion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal-manifest", type=Path, required=True)
    parser.add_argument("--event-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-stride", type=int, default=1)
    arguments = parser.parse_args(argv)
    manifest = json.loads(arguments.portal_manifest.expanduser().resolve().read_text())
    index = int(arguments.event_index)
    motion = Path(str(manifest["course_motions"][index])).expanduser().resolve()
    mesh = load_target_mesh(
        Path(str(manifest["terrain_usd"])),
        position_world=manifest["terrain_position_world"],
        quaternion_world_from_usd_wxyz=(
            manifest["terrain_quaternion_world_from_usd_wxyz"]
        ),
    )
    result = render_stitched_motion(
        load_stitched_motion_npz(motion),
        archive_path=arguments.motion_archive,
        target_mesh=mesh,
        model_path=arguments.model_path,
        output_path=arguments.output,
        width=arguments.width,
        height=arguments.height,
        frame_stride=arguments.frame_stride,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
