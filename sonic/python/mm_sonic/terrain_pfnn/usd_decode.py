"""Decode one hash-bound USD terrain to a numeric-only NPZ helper artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from mm_sonic.build_terrain_pfnn_dataset import _height_at
from mm_sonic.grail_terrain_source import load_grail_motion, resample_grail_motion
from mm_sonic.terrain_oracle.contact import CanonicalMeshQuery
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.source_grail import _load_usd_mesh


SCHEMA = "mm-sonic-usd-mesh/v1"
PAIR_SCHEMA = "mm-sonic-grail-pair/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(
    usd_path: str | Path,
    output_path: str | Path,
    expected_sha256: str,
    *,
    robot_path: str | Path | None = None,
    expected_robot_sha256: str | None = None,
    center_frame: int | None = None,
) -> None:
    source = Path(usd_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise ValueError("USD helper source hash mismatch")
    mesh = _load_usd_mesh(source, source_asset_sha256=expected_sha256)
    extra: dict[str, np.ndarray] = {}
    schema = SCHEMA
    if robot_path is not None or expected_robot_sha256 is not None or center_frame is not None:
        if (
            robot_path is None
            or expected_robot_sha256 is None
            or center_frame is None
        ):
            raise ValueError("GRAIL helper robot/hash/frame contract is invalid")
        robot = Path(robot_path).expanduser().resolve()
        if (
            not robot.is_file()
            or type(expected_robot_sha256) is not str
            or _sha256(robot) != expected_robot_sha256
            or type(center_frame) is not int
            or center_frame < 0
        ):
            raise ValueError("GRAIL helper robot/hash/frame contract is invalid")
        raw = load_grail_motion(robot, expected_frames=250)
        motion = resample_grail_motion(
            raw.root_position,
            raw.root_quaternion_xyzw,
            raw.dof_mujoco,
            source_fps=25.0,
            target_fps=30.0,
        )
        if center_frame >= len(motion.root_position):
            raise ValueError("GRAIL helper center frame is outside resampled motion")
        root_xy = np.asarray(motion.root_position[center_frame, :2], dtype=np.float64)
        x, y, z, w = np.asarray(
            motion.root_quaternion_xyzw[center_frame], dtype=np.float64
        )
        yaw = np.arctan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        )
        world_from_usd = RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.asarray(
                (np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)),
                dtype=np.float32,
            ),
        )
        query = CanonicalMeshQuery(mesh, world_from_usd)
        support = float(_height_at(query, root_xy.reshape(1, 2))[0])
        if not np.isfinite(support):
            raise ValueError("GRAIL helper source support anchor is invalid")
        schema = PAIR_SCHEMA
        extra = {
            "robot_sha256": np.asarray(expected_robot_sha256),
            "anchor_root_xy": np.asarray(root_xy, dtype=np.float32),
            "anchor_yaw": np.asarray(yaw, dtype=np.float32),
            "anchor_support": np.asarray(support, dtype=np.float32),
            "anchor_center_frame": np.asarray(center_frame, dtype=np.int32),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez(
            stream,
            schema=np.asarray(schema),
            source_sha256=np.asarray(expected_sha256),
            vertices=np.asarray(mesh.vertices_local, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            valid_faces=np.asarray(mesh.valid_faces, dtype=np.bool_),
            **extra,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot")
    parser.add_argument("--expected-robot-sha256")
    parser.add_argument("--center-frame", type=int)
    arguments = parser.parse_args(argv)
    decode(
        arguments.usd,
        arguments.output,
        arguments.expected_sha256,
        robot_path=arguments.robot,
        expected_robot_sha256=arguments.expected_robot_sha256,
        center_frame=arguments.center_frame,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PAIR_SCHEMA", "SCHEMA", "decode", "main"]
