from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.conversion import (
    finite_difference_quaternions,
    finite_difference_vectors,
)
from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_reach_builder.mirror import (
    MIRROR_BONES,
    _reflect_rotations,
)
from resources.g1_reach_builder.motions import _contacts
from resources.g1_reach_builder.sources import load_gmr_archive
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
    forward_local_hierarchy,
    world_to_local,
)
from resources.g1_terrain_builder.schema import SourceClip
from resources.retime_flat_database import (
    FlatDatabase,
    read_database,
    write_database,
)


TARGET_FPS = 25.0
DEFAULT_SOURCE_FPS = 100.0
DEFAULT_G1_XML = Path(
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)
_MOTION_FILES = {
    "walking": "walking_database.bin",
    "carry_left": "carry_left_database.bin",
    "carry_right": "carry_right_database.bin",
}


@dataclass
class EpisodeMotion:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    source_frames: np.ndarray


@dataclass(frozen=True)
class EpisodePackPaths:
    walking: Path
    carry_left: Path
    carry_right: Path
    manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_episode_motion(motion: EpisodeMotion) -> None:
    frames = len(motion.positions)
    expected = (
        (motion.positions, (frames, 31, 3), np.float32, "positions"),
        (motion.velocities, (frames, 31, 3), np.float32, "velocities"),
        (motion.rotations, (frames, 31, 4), np.float32, "rotations"),
        (
            motion.angular_velocities,
            (frames, 31, 3),
            np.float32,
            "angular velocities",
        ),
        (
            motion.foot_contacts,
            (frames, 2),
            np.uint8,
            "foot contacts",
        ),
        (
            motion.source_frames,
            (frames,),
            np.int32,
            "source frames",
        ),
    )
    if frames < 2:
        raise ValueError("episode motion requires at least two frames")
    for values, shape, dtype, label in expected:
        array = np.asarray(values)
        if array.shape != shape:
            if label == "positions":
                raise ValueError(
                    f"episode motion requires 31 bones, got {array.shape}"
                )
            raise ValueError(
                f"episode motion {label} shape must be {shape}, got "
                f"{array.shape}"
            )
        if array.dtype != np.dtype(dtype):
            raise ValueError(
                f"episode motion {label} dtype must be {np.dtype(dtype)}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(
            array
        ).all():
            raise ValueError(f"episode motion {label} contains non-finite values")
    if np.any(
        (motion.foot_contacts != 0) & (motion.foot_contacts != 1)
    ):
        raise ValueError("episode motion foot contacts must be binary")
    rotation_error = np.max(
        np.abs(np.linalg.norm(motion.rotations, axis=-1) - 1.0)
    )
    if rotation_error > 1.0e-4:
        raise ValueError(
            f"episode motion rotation norm error {rotation_error}"
        )
    if np.any(np.diff(motion.source_frames) < 0):
        raise ValueError("episode motion source frames must be monotonic")


def _episode_motion(
    positions: np.ndarray,
    rotations: np.ndarray,
    source_frames: np.ndarray,
) -> EpisodeMotion:
    positions = np.asarray(positions, np.float32)
    rotations = holden_quat.unroll(
        holden_quat.normalize(np.asarray(rotations, np.float64))
    ).astype(np.float32)
    world_positions, _ = forward_local_hierarchy(
        positions.astype(np.float64),
        rotations.astype(np.float64),
        G1_SKELETON.parents,
    )
    result = EpisodeMotion(
        positions=positions,
        velocities=finite_difference_vectors(positions, TARGET_FPS),
        rotations=rotations,
        angular_velocities=finite_difference_quaternions(
            rotations, TARGET_FPS
        ),
        foot_contacts=_contacts(world_positions, TARGET_FPS),
        source_frames=np.asarray(source_frames, np.int32),
    )
    validate_episode_motion(result)
    return result


def convert_named_source(
    archive: Path,
    sequence_id: str,
    kinematics: G1Kinematics,
    source_fps: float = DEFAULT_SOURCE_FPS,
) -> EpisodeMotion:
    sources = load_gmr_archive(
        Path(archive),
        fps_override=source_fps,
        include_excluded=True,
    )
    source = next(
        (value for value in sources if value.sequence_id == sequence_id),
        None,
    )
    if source is None:
        raise ValueError(f"episode source {sequence_id!r} is missing")
    motion, skeleton, _ = convert_source_clip(
        SourceClip(
            source.sequence_id,
            source.fps,
            source.qpos,
            source.source_frames,
            "episode",
        ),
        kinematics,
        target_fps=TARGET_FPS,
    )
    if skeleton.signature() != G1_SKELETON.signature():
        raise ValueError("episode source skeleton signature mismatch")
    return _episode_motion(
        motion.positions,
        motion.rotations,
        motion.source_frames,
    )


def mirror_episode_motion(source: EpisodeMotion) -> EpisodeMotion:
    validate_episode_motion(source)
    world_positions, world_rotations = forward_local_hierarchy(
        source.positions.astype(np.float64),
        source.rotations.astype(np.float64),
        G1_SKELETON.parents,
    )
    mirrored_positions = world_positions[:, MIRROR_BONES].copy()
    mirrored_positions[..., 2] *= -1.0
    mirrored_rotations = _reflect_rotations(
        world_rotations[:, MIRROR_BONES]
    )
    local_positions, local_rotations = world_to_local(
        mirrored_positions,
        mirrored_rotations,
        G1_SKELETON.parents,
    )
    result = _episode_motion(
        local_positions,
        local_rotations,
        source.source_frames.copy(),
    )
    result.foot_contacts = source.foot_contacts[:, ::-1].copy()
    validate_episode_motion(result)
    return result


def _flat_database(motion: EpisodeMotion) -> FlatDatabase:
    validate_episode_motion(motion)
    return FlatDatabase(
        bone_positions=motion.positions,
        bone_velocities=motion.velocities,
        bone_rotations=motion.rotations,
        bone_angular_velocities=motion.angular_velocities,
        bone_parents=np.asarray(G1_SKELETON.parents, np.int32),
        range_starts=np.array([0], np.int32),
        range_stops=np.array([len(motion.positions)], np.int32),
        contact_states=motion.foot_contacts,
    )


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"episode pack path is not a directory: {path}")
    shutil.rmtree(path)


def write_episode_pack(
    output: Path,
    walking: EpisodeMotion,
    carry_left: EpisodeMotion,
    carry_right: EpisodeMotion,
    *,
    source_archive: Path | None = None,
) -> EpisodePackPaths:
    motions = {
        "walking": walking,
        "carry_left": carry_left,
        "carry_right": carry_right,
    }
    for motion in motions.values():
        validate_episode_motion(motion)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    previous = output.parent / f".{output.name}.previous-{os.getpid()}"
    _remove_directory(temporary)
    _remove_directory(previous)
    temporary.mkdir()
    try:
        manifest_motions: dict[str, dict[str, object]] = {}
        for name, motion in motions.items():
            path = temporary / _MOTION_FILES[name]
            write_database(path, _flat_database(motion))
            loaded = read_database(path)
            if loaded.bone_positions.shape != motion.positions.shape:
                raise ValueError(f"episode motion {name} round-trip mismatch")
            manifest_motions[name] = {
                "path": path.name,
                "frame_count": len(motion.positions),
                "sha256": _sha256(path),
            }
        archive = None if source_archive is None else Path(source_archive)
        manifest = {
            "schema": "g1-episode-motion-pack",
            "version": 1,
            "target_fps": TARGET_FPS,
            "bone_count": 31,
            "skeleton_signature": G1_SKELETON.signature(),
            "source_archive": None if archive is None else str(archive.resolve()),
            "source_archive_sha256": (
                None if archive is None else _sha256(archive)
            ),
            "motions": manifest_motions,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            if not output.is_dir():
                raise ValueError(
                    f"episode output is not a directory: {output}"
                )
            os.replace(output, previous)
        try:
            os.replace(temporary, output)
        except BaseException:
            if previous.exists() and not output.exists():
                os.replace(previous, output)
            raise
        _remove_directory(previous)
    except BaseException:
        _remove_directory(temporary)
        raise
    return EpisodePackPaths(
        walking=output / _MOTION_FILES["walking"],
        carry_left=output / _MOTION_FILES["carry_left"],
        carry_right=output / _MOTION_FILES["carry_right"],
        manifest=output / "manifest.json",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact walking and bilateral carry motion packs"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--g1-xml",
        type=Path,
        default=Path(os.environ.get("G1_XML", DEFAULT_G1_XML)),
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=DEFAULT_SOURCE_FPS,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    kinematics = G1Kinematics(str(args.g1_xml))
    walking = convert_named_source(
        args.archive,
        "walking",
        kinematics,
        args.source_fps,
    )
    carry_left = convert_named_source(
        args.archive,
        "carry_walking",
        kinematics,
        args.source_fps,
    )
    carry_right = mirror_episode_motion(carry_left)
    paths = write_episode_pack(
        args.output,
        walking,
        carry_left,
        carry_right,
        source_archive=args.archive,
    )
    print(
        "BUILT g1-episode-motion-pack "
        f"walking={len(walking.positions)} "
        f"carry_left={len(carry_left.positions)} "
        f"carry_right={len(carry_right.positions)} "
        f"target_fps={TARGET_FPS:g} output={paths.manifest.parent}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
