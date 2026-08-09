"""Render the fixed high-difficulty visual review matrix for a G1 HCT eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render_g1_hct_rollout import DEFAULT_G1_SCENE, render


REVIEW_CASES = (
    ("rolling_hills", "abrupt_omni"),
    ("rolling_hills", "zigzag_yaw"),
    ("discrete_obstacles", "zigzag_yaw"),
    ("discrete_obstacles", "diagonal_arc_left"),
    ("rough_slope_up", "steady_forward"),
    ("rough_slope_down", "forward_stop_backward"),
    ("stairs_up", "left_strafe"),
    ("stairs_down", "right_strafe"),
    ("stairs_up", "diagonal_arc_left"),
    ("stairs_down", "diagonal_arc_right"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_G1_SCENE)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for terrain, program in REVIEW_CASES:
        output = args.output_dir / f"{terrain}__{program}.mp4"
        try:
            render(
                args.source,
                output,
                env_index=None,
                terrain=terrain,
                program=program,
                scene=args.scene,
                stride=args.stride,
            )
            results.append({
                "terrain": terrain,
                "program": program,
                "status": "rendered",
                "video": str(output.resolve()),
            })
        except Exception as error:
            results.append({
                "terrain": terrain,
                "program": program,
                "status": "rejected_or_failed",
                "reason": f"{type(error).__name__}: {error}",
            })
    report = {
        "source": str(args.source.resolve()),
        "rendered": sum(row["status"] == "rendered" for row in results),
        "requested": len(results),
        "cases": results,
    }
    path = args.output_dir / "review_suite.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["rendered"]:
        raise SystemExit("no clean review case could be rendered")


if __name__ == "__main__":
    main()
