"""Render one accepted directional-stair report on its exact archive mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render_stitched_motion import (
    load_stitched_motion_npz,
    render_stitched_motion,
)


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    arguments = parser.parse_args()
    report_path = arguments.report.expanduser().resolve()
    report = json.loads(report_path.read_text())
    if report.get("status") != "accepted":
        raise ValueError("refusing to render a rejected directional pilot")
    result = render_stitched_motion(
        load_stitched_motion_npz(report_path.parent / "motion.npz"),
        model_path=arguments.model,
        output_path=arguments.output,
        archive_path=arguments.archive,
        target_clip_index=int(report["clip_index"]),
        frame_stride=arguments.frame_stride,
        width=arguments.width,
        height=arguments.height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
