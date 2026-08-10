"""Build a compact canonical corpus from the natural-terrain gold manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .terrain_oracle import storage
from .terrain_oracle.canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
    TerrainBinding,
    derive_clip_kinematics,
)
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.source_grail import _load_usd_mesh


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = q.T
    return np.unwrap(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def _commands(
    arrays: dict[str, np.ndarray], root_quaternion: np.ndarray, fps: float
) -> CommandTrack:
    frame_count = len(root_quaternion)
    velocity = np.asarray(
        arrays["command_velocity_robot_local_xy"], dtype=np.float32
    )
    desired_yaw = np.asarray(
        arrays["command_facing_yaw_world_rad"], dtype=np.float64
    )
    root_yaw = _yaw_from_wxyz(root_quaternion)
    local_facing_yaw = np.arctan2(
        np.sin(desired_yaw - root_yaw), np.cos(desired_yaw - root_yaw)
    )
    facing = np.stack(
        (np.cos(local_facing_yaw), np.sin(local_facing_yaw)), axis=1
    ).astype(np.float32)
    return CommandTrack(
        observed_travel_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_facing_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_mask=np.zeros(frame_count, dtype=np.bool_),
        inferred_velocity_local_xy=velocity,
        inferred_facing_local_xy=facing,
        inferred_yaw_rate_rad_s=np.asarray(
            np.gradient(root_yaw) * float(fps), dtype=np.float32
        ),
    )


def _canonicalize(
    row: dict[str, object],
    *,
    adapter: _G1FootfallAdapter,
    body_ids: np.ndarray,
    terrain_binding: TerrainBinding,
) -> CanonicalClip:
    motion_path = Path(str(row["motion"])).expanduser().resolve()
    with np.load(motion_path, allow_pickle=False) as bundle:
        arrays = {name: np.asarray(bundle[name]).copy() for name in bundle.files}
    root = np.asarray(arrays["root_position_world"], dtype=np.float32)
    quaternion = np.asarray(
        arrays["root_quaternion_world_wxyz"], dtype=np.float32
    )
    joints = np.asarray(arrays["joint_position"], dtype=np.float32)
    fps = float(np.asarray(arrays["fps"]).reshape(()))
    if abs(fps - 50.0) > 1.0e-4:
        raise ValueError(f"gold motion is not 50 Hz: {motion_path}")
    frame_count = len(root)
    body_position = np.empty((frame_count, 30, 3), dtype=np.float32)
    body_quaternion = np.empty((frame_count, 30, 4), dtype=np.float32)
    sole_position = np.empty((frame_count, 2, 3), dtype=np.float32)
    radii = adapter.sole_sphere_radii()
    for frame in range(frame_count):
        adapter._set_pose(
            adapter._adapted_data, root[frame], quaternion[frame], joints[frame]
        )
        body_position[frame] = adapter._adapted_data.xpos[body_ids]
        body_quaternion[frame] = adapter._adapted_data.xquat[body_ids]
        for foot in range(2):
            support = adapter._sole_positions(adapter._adapted_data, foot).copy()
            support[:, 2] -= radii[foot]
            sole_position[frame, foot] = np.mean(support, axis=0)

    stance = np.asarray(
        arrays.get(
            "authored_stance_mask",
            np.zeros((frame_count, 2), dtype=np.bool_),
        ),
        dtype=np.bool_,
    )
    support_count = np.asarray(
        arrays.get("target_stance_support_point_count", stance.astype(np.int16)),
        dtype=np.float32,
    )
    confidence = np.clip(support_count / 4.0, 0.0, 1.0)
    payload = motion_path.read_bytes()
    source = SourceIdentity(
        source_format="natural-terrain-gold-motion-npz-v1",
        source_path=str(motion_path),
        source_size_bytes=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_license_id="UNRECORDED",
        coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
        quaternion_convention="wxyz",
        pose_origin=CLEAN_POSE_ORIGIN,
    )
    construction = str(row["construction"])
    family = str(row.get("clip_family") or "terrain")
    clip_index = int(row["clip_index"])
    clip_id = f"natural_gold/{construction}/{family}/clip_{clip_index:04d}"
    mirror_of = None
    if construction == "exact_scene_mirror":
        mirror_of = (
            f"natural_gold/densely_reviewed_natural_source/{family}/"
            f"clip_{clip_index:04d}"
        )
    elif construction == "exact_mirror_time_reverse":
        mirror_of = (
            f"natural_gold/exact_time_reverse/{family}/clip_{clip_index:04d}"
        )
    zeros3 = np.zeros((frame_count, 3), dtype=np.float32)
    zeros_joint = np.zeros((frame_count, 29), dtype=np.float32)
    zeros_body = np.zeros((frame_count, 30, 3), dtype=np.float32)
    clip = CanonicalClip(
        clip_id=clip_id,
        fps=50.0,
        source=source,
        joint_names=ISAACLAB_JOINT_NAMES,
        body_names=ISAACLAB_BODY_NAMES,
        root_position_world=root,
        root_quaternion_world_wxyz=quaternion,
        joint_position=joints,
        root_linear_velocity_world=zeros3,
        root_angular_velocity_world=zeros3.copy(),
        joint_velocity=zeros_joint,
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
        body_linear_velocity_world=zeros_body,
        body_angular_velocity_world=zeros_body.copy(),
        sole_position_world=sole_position,
        sole_quaternion_world_wxyz=body_quaternion[:, (18, 19)],
        heel_position_world=sole_position.copy(),
        toe_position_world=sole_position.copy(),
        contact=stance.astype(np.float32),
        contact_confidence=confidence.astype(np.float32),
        commands=_commands(arrays, quaternion, fps),
        terrain=terrain_binding,
        action_tags=(
            "natural-terrain-gold",
            construction,
            family,
            "quiet-upper-limb",
            "exact-contact-audited",
        ),
        mirror_of=mirror_of,
    )
    clip = derive_clip_kinematics(clip)
    clip.validate()
    return clip


def build(
    *,
    gold_manifest: Path,
    model_path: Path,
    output: Path,
    limit: int | None = None,
) -> dict[str, object]:
    manifest_path = gold_manifest.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output = output.expanduser().resolve()
    rows = json.loads(manifest_path.read_text())["rows"]
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError("natural terrain gold selection is empty")
    if output.exists():
        raise FileExistsError(f"canonical overlay output exists: {output}")

    adapter = _G1FootfallAdapter(
        model_path, ISAACLAB_JOINT_NAMES, maximum_joint_correction_rad=0.0
    )
    mujoco = adapter._mujoco
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ISAACLAB_BODY_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        raise ValueError("G1 model is missing canonical body names")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (stage / "clips").mkdir()
        (stage / "meshes").mkdir()
        mesh_records: dict[str, storage.MeshRecord] = {}
        clip_records = []
        construction_counts: Counter[str] = Counter()
        for index, row in enumerate(rows, start=1):
            terrain_path = Path(str(row["terrain_usd"])).expanduser().resolve()
            terrain_payload = terrain_path.read_bytes()
            terrain_sha = hashlib.sha256(terrain_payload).hexdigest()
            mesh = _load_usd_mesh(
                terrain_path, source_asset_sha256=terrain_sha
            )
            mesh_sha = storage.mesh_digest(mesh)
            if mesh_sha not in mesh_records:
                mesh_records[mesh_sha] = storage.write_mesh(stage / "meshes", mesh)
            binding = TerrainBinding(
                asset_path=str(terrain_path),
                asset_size_bytes=len(terrain_payload),
                asset_sha256=terrain_sha,
                asset_license_id="UNRECORDED",
                mesh_sha256=mesh_sha,
                world_from_terrain=RigidTransform(
                    np.zeros(3, dtype=np.float32),
                    np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                ),
                validity_mask_path=None,
            )
            clip = _canonicalize(
                row,
                adapter=adapter,
                body_ids=body_ids,
                terrain_binding=binding,
            )
            clip_records.append(storage.write_clip(stage / "clips", clip))
            construction_counts[str(row["construction"])] += 1
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(f"canonicalized {index}/{len(rows)}: {clip.clip_id}")

        records = tuple(
            sorted(
                clip_records,
                key=lambda record: (record.clip_id, record.sha256),
            )
        )
        metadata = {
            "natural_terrain_gold": {
                "schema": "natural-terrain-gold-canonical-corpus/v1",
                "manifest": str(manifest_path),
                "clip_count": len(clip_records),
                "construction_counts": dict(sorted(construction_counts.items())),
            },
            "mesh_records": [
                record.to_dict() for record in mesh_records.values()
            ],
        }
        manifest_directory = storage.publish_corpus(
            stage / "_manifest", records, metadata
        )
        os.replace(manifest_directory / "manifest.json", stage / "manifest.json")
        (manifest_directory / storage.COMPLETION_MARKER).unlink()
        manifest_directory.rmdir()
        storage._seal_directory(stage)
        storage._publish_directory_no_replace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "schema": "natural-terrain-gold-canonical-corpus/v1",
        "clip_count": len(clip_records),
        "output": str(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build(
        gold_manifest=arguments.gold_manifest,
        model_path=arguments.model,
        output=arguments.output,
        limit=arguments.limit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
