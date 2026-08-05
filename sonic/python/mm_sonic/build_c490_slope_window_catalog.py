"""Extract reusable uphill and downhill windows from clean C490 slope clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .terrain_oracle.continuous_slope import (
    extract_monotonic_slope_window,
    surface_height_along_trajectory,
)
from .terrain_oracle.stair_geometry_warp import _archive_terrain_index
from .terrain_oracle.terrain_route_profile import sample_terrain_route_profile


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-c490-curb-slope-clean-50hz-v1.zarr"
)
DEFAULT_OUTPUT = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-c490-slope-source-windows-v1.json"
)


def build_catalog(archive_path: Path, output_path: Path) -> dict[str, object]:
    import zarr

    source = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(source), mode="r")
    rows: list[dict[str, object]] = []
    for clip_index in range(len(archive["clip_names"])):
        if str(archive["clip_family"][clip_index]) != "c490_slope":
            continue
        start = int(archive["clip_start_idx"][clip_index])
        stop = int(archive["clip_end_idx"][clip_index])
        root = np.asarray(
            archive["body_pos_w"][start:stop, 0], dtype=np.float64
        )
        mesh = _archive_terrain_index(archive, clip_index)
        height = surface_height_along_trajectory(mesh, root)
        for traversal in ("up", "down"):
            window = extract_monotonic_slope_window(
                root, height, traversal=traversal
            )
            if window is None:
                continue
            first = window.start_frame
            last = window.stop_frame - 1
            profile = sample_terrain_route_profile(
                mesh,
                root[first, :2],
                root[last, :2],
                sample_spacing_m=0.01,
            )
            signed_delta = (
                profile.endpoint_height_delta_m
                if traversal == "up"
                else -profile.endpoint_height_delta_m
            )
            if (
                profile.jump_count != 0
                or profile.continuous_slope_fraction < 0.10
                or signed_delta < 0.025
            ):
                continue
            rows.append(
                {
                    "clip_index": int(clip_index),
                    "clip_name": str(archive["clip_names"][clip_index]),
                    "traversal": traversal,
                    "start_frame": int(window.start_frame),
                    "stop_frame": int(window.stop_frame),
                    "route_start_xy": [float(v) for v in root[first, :2]],
                    "route_end_xy": [float(v) for v in root[last, :2]],
                    "route_length_m": float(
                        np.linalg.norm(root[last, :2] - root[first, :2])
                    ),
                    "height_delta_m": float(signed_delta),
                    "height_range_m": float(profile.height_range_m),
                    "continuous_slope_fraction": float(
                        profile.continuous_slope_fraction
                    ),
                    "maximum_continuous_slope_rad": float(
                        profile.maximum_continuous_slope_rad
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["traversal"]),
            float(row["height_delta_m"]),
            int(row["clip_index"]),
        )
    )
    payload = {
        "schema": "grail-c490-slope-source-windows/v1",
        "archive_path": str(source),
        "window_count": len(rows),
        "traversal_counts": {
            traversal: sum(row["traversal"] == traversal for row in rows)
            for traversal in ("up", "down")
        },
        "windows": rows,
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    result = build_catalog(arguments.archive, arguments.output)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "window_count", "traversal_counts")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
