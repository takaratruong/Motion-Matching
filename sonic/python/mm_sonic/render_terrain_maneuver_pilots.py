"""Render terrain maneuver pilot bundles on their exact target mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .canonical_terrain_matcher import RegularGridHeightField
from .generate_generic_stair_route import load_target_mesh
from .render_stitched_motion import (
    load_stitched_motion_npz,
    render_stitched_motion,
)
from .terrain_maneuver_composer import HOLD
from .terrain_oracle.stitch import StitchedMotion


def _hold_window(
    motion: StitchedMotion,
    motion_path: Path,
    *,
    context_frames: int,
) -> tuple[StitchedMotion, int, int]:
    with np.load(motion_path, allow_pickle=False) as data:
        phase = np.asarray(data["maneuver_phase"], dtype=np.uint8)
    hold = np.flatnonzero(phase == HOLD)
    if len(phase) != len(motion.root_position_world) or not len(hold):
        raise ValueError(f"motion has no valid maneuver hold: {motion_path}")
    context = int(context_frames)
    if context < 1:
        raise ValueError("hold-window context must be positive")
    start = max(0, int(hold[0]) - context)
    stop = min(len(phase), int(hold[-1]) + 1 + context)
    seams = tuple(
        int(value) - start
        for value in motion.seam_indices
        if start <= int(value) < stop
    )
    return (
        StitchedMotion(
            fps=motion.fps,
            root_position_world=motion.root_position_world[start:stop],
            root_quaternion_world_wxyz=(
                motion.root_quaternion_world_wxyz[start:stop]
            ),
            joint_position=motion.joint_position[start:stop],
            provenance=motion.provenance[start:stop],
            seam_indices=seams,
        ),
        start,
        stop,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--motion-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--view",
        choices=("default", "side", "overhead"),
        default="default",
        help="camera framing; side/overhead expose lateral terrain motion",
    )
    parser.add_argument(
        "--hold-window-context-frames",
        type=int,
        help="render only this many full-rate frames before and after the hold",
    )
    arguments = parser.parse_args(argv)

    manifest_path = arguments.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    shared_terrain = None
    if "terrain_usd" in manifest:
        shared_terrain = load_target_mesh(
            Path(str(manifest["terrain_usd"])),
            position_world=manifest["terrain_position_world"],
            quaternion_world_from_usd_wxyz=(
                manifest["terrain_quaternion_world_from_usd_wxyz"]
            ),
        )
    archive = zarr.open_group(
        str(arguments.motion_archive.expanduser().resolve()), mode="r"
    )
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    selected = set(str(value) for value in arguments.label)
    rows = [
        row
        for row in manifest["pilots"]
        if not selected or str(row["label"]) in selected
    ]
    if selected and selected != {str(row["label"]) for row in rows}:
        missing = sorted(selected - {str(row["label"]) for row in rows})
        raise ValueError(f"unknown pilot labels: {missing}")
    output_dir = (
        manifest_path.parent / "renders"
        if arguments.output_dir is None
        else arguments.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        label = str(row["label"])
        motion_path = Path(str(row["motion"]))
        motion = load_stitched_motion_npz(motion_path)
        source_start = 0
        source_stop = len(motion.root_position_world)
        suffix = ""
        if arguments.hold_window_context_frames is not None:
            motion, source_start, source_stop = _hold_window(
                motion,
                motion_path,
                context_frames=arguments.hold_window_context_frames,
            )
            suffix = ".hold_window"
        output = output_dir / f"{label}{suffix}.mp4"
        terrain = shared_terrain
        if terrain is None:
            with np.load(Path(str(row["terrain_npz"])), allow_pickle=False) as data:
                spacing = tuple(float(value) for value in data["spacing_m"])
                origin = tuple(float(value) for value in data["origin_xy"])
                terrain = RegularGridHeightField(
                    np.asarray(data["height"], dtype=np.float64),
                    valid=np.asarray(data["valid"], dtype=np.bool_),
                    spacing_m=spacing,
                    origin_xy=origin,
                ).index
        path_extent = float(
            np.max(np.ptp(np.asarray(motion.root_position_world)[:, :2], axis=0))
        )
        camera: dict[str, object] = {}
        if arguments.view == "side":
            camera = {
                "camera_azimuth_offset_deg": 90.0,
                "camera_elevation_deg": -16.0,
                "camera_distance": 2.8,
                "camera_follow_root": True,
            }
        elif arguments.view == "overhead":
            camera = {
                "camera_azimuth_offset_deg": 90.0,
                "camera_elevation_deg": -68.0,
                "camera_distance": float(
                    np.clip(1.10 * path_extent + 1.6, 2.8, 5.2)
                ),
                "camera_follow_root": False,
            }
        receipt = render_stitched_motion(
            motion,
            target_mesh=terrain,
            joint_names=joint_names,
            model_path=arguments.model_path,
            output_path=output,
            width=arguments.width,
            height=arguments.height,
            frame_stride=arguments.frame_stride,
            **camera,
        )
        results.append(
            {
                "label": label,
                "video": str(output),
                "source_frame_start": source_start,
                "source_frame_stop": source_stop,
                "view": arguments.view,
                **receipt,
            }
        )
        print(output, flush=True)
    (output_dir / "render_manifest.json").write_text(
        json.dumps({"videos": results}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
