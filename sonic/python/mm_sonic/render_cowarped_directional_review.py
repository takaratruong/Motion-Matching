"""Render one selected paired co-warp motion without regenerating it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .render_stitched_motion import (
    _load_usd_mesh,
    load_stitched_motion_npz,
    render_stitched_motion,
)
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


def render_selected(
    *,
    selection_path: Path,
    selection_index: int,
    archive_path: Path,
    model_path: Path,
    output_root: Path,
    frame_stride: int,
    camera_azimuth_offset_deg: float,
) -> dict[str, object]:
    payload = json.loads(selection_path.read_text())
    rows = list(payload["rows"])
    index = int(selection_index)
    if not 0 <= index < len(rows):
        return {"status": "skipped", "selection_index": index, "count": len(rows)}
    row = dict(rows[index])
    motion_path = Path(str(row["motion"])).resolve()
    terrain_path = Path(str(row["terrain_usd"])).resolve()
    mesh = _load_usd_mesh(
        terrain_path,
        source_asset_sha256=hashlib.sha256(terrain_path.read_bytes()).hexdigest(),
    )
    target_mesh = TerrainMeshIndex(
        mesh,
        RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        ),
    )
    archive = zarr.open_group(str(archive_path), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    destination = (
        output_root
        / f"rank_{index:03d}_clip_{int(row['clip_index']):03d}"
        / f"{row['mode']}.mp4"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = render_stitched_motion(
        load_stitched_motion_npz(motion_path),
        model_path=model_path,
        output_path=destination,
        target_mesh=target_mesh,
        joint_names=joint_names,
        width=640,
        height=360,
        frame_stride=max(1, int(frame_stride)),
        camera_azimuth_offset_deg=float(camera_azimuth_offset_deg),
        camera_follow_root=True,
    )
    report = {
        "status": "rendered",
        "selection_index": index,
        "clip_index": int(row["clip_index"]),
        "mode": str(row["mode"]),
        "motion": str(motion_path),
        "terrain_usd": str(terrain_path),
        **result,
    }
    (destination.parent / "render.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-index", type=int, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--camera-azimuth-offset-deg", type=float, default=90.0)
    arguments = parser.parse_args()
    print(
        json.dumps(
            render_selected(
                selection_path=arguments.selection.expanduser().resolve(),
                selection_index=arguments.selection_index,
                archive_path=arguments.archive.expanduser().resolve(),
                model_path=arguments.model.expanduser().resolve(),
                output_root=arguments.output_root.expanduser().resolve(),
                frame_stride=arguments.frame_stride,
                camera_azimuth_offset_deg=arguments.camera_azimuth_offset_deg,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
