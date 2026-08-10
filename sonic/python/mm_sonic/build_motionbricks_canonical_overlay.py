"""Add the validated MotionBricks G1 bank to a canonical terrain corpus."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .compose_motionbricks_terrain_course import _resample_motionbricks
from .build_bones_side_on_stair_pilots import _attenuate_upper_limb_motion
from .render_continuous_terrain_matching import DEFAULT_MODEL
from .terrain_oracle import storage
from .terrain_oracle.canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
    derive_clip_kinematics,
)
from .terrain_oracle.contact import (
    CanonicalMeshQuery,
    ContactConfig,
    SoleGeometry,
    reconstruct_contacts,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.source_flat import _infer_commands


def _resampled_observed_commands(
    path: Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        source_fps = float(np.asarray(data["fps"]).item())
        raw = np.asarray(data["raw_two_stick_command"], dtype=np.float64)
    source_time = np.arange(len(raw), dtype=np.float64) / source_fps
    target_time = np.minimum(
        np.arange(frame_count, dtype=np.float64) / 50.0,
        source_time[-1],
    )
    resampled = np.stack(
        [
            np.interp(target_time, source_time, raw[:, column])
            for column in range(raw.shape[1])
        ],
        axis=1,
    )
    return (
        np.asarray(resampled[:, :2], dtype=np.float32),
        np.asarray(resampled[:, 2:4], dtype=np.float32),
        np.asarray(resampled[:, 4] >= 0.5, dtype=np.bool_),
    )


def _canonicalize_recording(
    path: Path,
    *,
    adapter: _G1FootfallAdapter,
    body_ids: np.ndarray,
    terrain_binding: object,
    surface_query: CanonicalMeshQuery,
    contact_config: ContactConfig,
    arm_swing_scale: float,
    wrist_swing_scale: float,
    arm_smoothing_sigma_frames: float,
) -> CanonicalClip:
    segment = _resample_motionbricks(path, label=f"motionbricks/{path.stem}")
    quiet_joints, _attenuation = _attenuate_upper_limb_motion(
        segment.joint_position,
        ISAACLAB_JOINT_NAMES,
        fps=50.0,
        arm_swing_scale=arm_swing_scale,
        wrist_swing_scale=wrist_swing_scale,
        smoothing_sigma_frames=arm_smoothing_sigma_frames,
    )
    segment = replace(
        segment, joint_position=np.asarray(quiet_joints, dtype=np.float32)
    )
    frame_count = len(segment.root_position_world)
    body_position = np.empty((frame_count, 30, 3), dtype=np.float32)
    body_quaternion = np.empty((frame_count, 30, 4), dtype=np.float32)
    for frame in range(frame_count):
        adapter._set_pose(
            adapter._adapted_data,
            segment.root_position_world[frame],
            segment.root_quaternion_world_wxyz[frame],
            segment.joint_position[frame],
        )
        body_position[frame] = adapter._adapted_data.xpos[body_ids]
        body_quaternion[frame] = adapter._adapted_data.xquat[body_ids]

    payload = path.read_bytes()
    source = SourceIdentity(
        source_format="motionbricks-g1-recording-v1",
        source_path=str(path.resolve()),
        source_size_bytes=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_license_id="UNRECORDED",
        coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
        quaternion_convention="wxyz",
        pose_origin=CLEAN_POSE_ORIGIN,
    )
    zeros3 = np.zeros((frame_count, 3), dtype=np.float32)
    zeros_joint = np.zeros((frame_count, 29), dtype=np.float32)
    zeros_body = np.zeros((frame_count, 30, 3), dtype=np.float32)
    observed_travel, observed_facing, observed_mask = (
        _resampled_observed_commands(path, frame_count)
    )
    placeholder_commands = CommandTrack(
        observed_travel_stick_xy=observed_travel,
        observed_facing_stick_xy=observed_facing,
        observed_mask=observed_mask,
        inferred_velocity_local_xy=np.zeros((frame_count, 2), dtype=np.float32),
        inferred_facing_local_xy=np.tile(
            np.asarray((1.0, 0.0), dtype=np.float32), (frame_count, 1)
        ),
        inferred_yaw_rate_rad_s=np.zeros(frame_count, dtype=np.float32),
    )
    clip_id = f"motionbricks/{path.stem}"
    mirror_of = None
    if path.stem.endswith("__mirror"):
        mirror_of = f"motionbricks/{path.stem.removesuffix('__mirror')}"
    clip = CanonicalClip(
        clip_id=clip_id,
        fps=50.0,
        source=source,
        joint_names=ISAACLAB_JOINT_NAMES,
        body_names=ISAACLAB_BODY_NAMES,
        root_position_world=np.asarray(
            segment.root_position_world, dtype=np.float32
        ),
        root_quaternion_world_wxyz=np.asarray(
            segment.root_quaternion_world_wxyz, dtype=np.float32
        ),
        joint_position=np.asarray(segment.joint_position, dtype=np.float32),
        root_linear_velocity_world=zeros3,
        root_angular_velocity_world=zeros3.copy(),
        joint_velocity=zeros_joint,
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
        body_linear_velocity_world=zeros_body,
        body_angular_velocity_world=zeros_body.copy(),
        sole_position_world=body_position[:, (18, 19)],
        sole_quaternion_world_wxyz=body_quaternion[:, (18, 19)],
        heel_position_world=body_position[:, (18, 19)],
        toe_position_world=body_position[:, (18, 19)],
        contact=np.zeros((frame_count, 2), dtype=np.float32),
        contact_confidence=np.zeros((frame_count, 2), dtype=np.float32),
        commands=placeholder_commands,
        terrain=terrain_binding,
        action_tags=(
            "flat",
            "motionbricks",
            "omnidirectional",
            "quiet-upper-limb",
            "contacts-reconstructed",
        ),
        mirror_of=mirror_of,
    )
    clip = derive_clip_kinematics(clip)
    inferred = _infer_commands(
        clip.root_quaternion_world_wxyz,
        clip.root_linear_velocity_world,
        clip.root_angular_velocity_world,
    )
    clip = replace(
        clip,
        commands=CommandTrack(
            observed_travel_stick_xy=observed_travel,
            observed_facing_stick_xy=observed_facing,
            observed_mask=observed_mask,
            inferred_velocity_local_xy=inferred.inferred_velocity_local_xy,
            inferred_facing_local_xy=inferred.inferred_facing_local_xy,
            inferred_yaw_rate_rad_s=inferred.inferred_yaw_rate_rad_s,
        ),
    )
    clip = reconstruct_contacts(
        clip, surface_query, contact_config
    ).apply(clip)
    clip.validate()
    return clip


def build_overlay(
    *,
    base_corpus: Path,
    recordings: Path,
    model_path: Path,
    output: Path,
    limit: int | None = None,
    arm_swing_scale: float = 0.35,
    wrist_swing_scale: float = 0.10,
    arm_smoothing_sigma_frames: float = 2.0,
) -> dict[str, object]:
    base = storage.load_corpus(base_corpus)
    paths = sorted(recordings.glob("*.npz"))
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise ValueError("MotionBricks recording selection is empty")
    if output.exists():
        raise FileExistsError(f"overlay output already exists: {output}")

    flat_record = next(
        record for record in base.clips if record.clip_id == "flat/takara_walk"
    )
    flat_clip = storage.read_clip(base_corpus / flat_record.relative_path)
    if flat_clip.terrain is None:
        raise ValueError("base flat clip has no terrain binding")
    flat_mesh_record = next(
        record
        for record in base.meshes
        if record.sha256 == flat_clip.terrain.mesh_sha256
    )
    flat_mesh = storage.read_mesh(base_corpus / flat_mesh_record.relative_path)
    surface_query = CanonicalMeshQuery(
        flat_mesh, flat_clip.terrain.world_from_terrain
    )
    adapter = _G1FootfallAdapter(
        model_path,
        ISAACLAB_JOINT_NAMES,
        maximum_joint_correction_rad=0.35,
    )
    mujoco = adapter._mujoco
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(
                adapter.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            for name in ISAACLAB_BODY_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        raise ValueError("G1 model is missing canonical body names")
    contact_config = ContactConfig(
        geometry=SoleGeometry.from_model(adapter.model)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        (stage / "clips").mkdir()
        (stage / "meshes").mkdir()
        for record in base.clips:
            os.link(
                base_corpus / record.relative_path,
                stage / record.relative_path,
            )
        for record in base.meshes:
            os.link(
                base_corpus / record.relative_path,
                stage / record.relative_path,
            )

        new_records: list[storage.ClipRecord] = []
        contact_frames = 0
        for index, path in enumerate(paths, start=1):
            clip = _canonicalize_recording(
                path,
                adapter=adapter,
                body_ids=body_ids,
                terrain_binding=flat_clip.terrain,
                surface_query=surface_query,
                contact_config=contact_config,
                arm_swing_scale=arm_swing_scale,
                wrist_swing_scale=wrist_swing_scale,
                arm_smoothing_sigma_frames=arm_smoothing_sigma_frames,
            )
            contact_frames += int(np.count_nonzero(clip.contact))
            new_records.append(storage.write_clip(stage / "clips", clip))
            if index == 1 or index % 20 == 0 or index == len(paths):
                print(f"canonicalized {index}/{len(paths)}: {clip.clip_id}")

        records = tuple(
            sorted(
                (*base.clips, *new_records),
                key=lambda record: (record.clip_id, record.sha256),
            )
        )
        manifest_directory = storage.publish_corpus(
            stage / "_manifest",
            records,
            {
                **base.metadata,
                "motionbricks_overlay": {
                    "schema": "motionbricks-canonical-overlay/v1",
                    "source_recordings": str(recordings.resolve()),
                    "recording_count": len(paths),
                    "contact_frame_foot_count": contact_frames,
                    "arm_swing_scale": float(arm_swing_scale),
                    "wrist_swing_scale": float(wrist_swing_scale),
                    "arm_smoothing_sigma_frames": float(
                        arm_smoothing_sigma_frames
                    ),
                },
                "mesh_records": [record.to_dict() for record in base.meshes],
            },
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
        "schema": "motionbricks-canonical-overlay/v1",
        "base_clip_count": len(base.clips),
        "added_clip_count": len(paths),
        "total_clip_count": len(base.clips) + len(paths),
        "contact_frame_foot_count": contact_frames,
        "arm_swing_scale": float(arm_swing_scale),
        "wrist_swing_scale": float(wrist_swing_scale),
        "arm_smoothing_sigma_frames": float(arm_smoothing_sigma_frames),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-corpus", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--arm-swing-scale", type=float, default=0.35)
    parser.add_argument("--wrist-swing-scale", type=float, default=0.10)
    parser.add_argument(
        "--arm-smoothing-sigma-frames", type=float, default=2.0
    )
    args = parser.parse_args()
    if not 0.0 <= args.arm_swing_scale <= 1.0:
        parser.error("--arm-swing-scale must lie in [0,1]")
    if not 0.0 <= args.wrist_swing_scale <= 1.0:
        parser.error("--wrist-swing-scale must lie in [0,1]")
    if args.arm_smoothing_sigma_frames < 0.0:
        parser.error("--arm-smoothing-sigma-frames must be nonnegative")
    report = build_overlay(
        base_corpus=args.base_corpus.expanduser().resolve(),
        recordings=args.recordings.expanduser().resolve(),
        model_path=args.model.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        limit=args.limit,
        arm_swing_scale=args.arm_swing_scale,
        wrist_swing_scale=args.wrist_swing_scale,
        arm_smoothing_sigma_frames=args.arm_smoothing_sigma_frames,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
