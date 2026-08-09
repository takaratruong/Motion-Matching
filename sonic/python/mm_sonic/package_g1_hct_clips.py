"""Split a mechanically clean HCT evaluation archive into 11-second clips."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np


FRAME_ARRAYS = (
    "root_pos",
    "root_quat",
    "root_lin_vel_b",
    "root_ang_vel_b",
    "joint_pos",
    "joint_vel",
    "actions",
    "foot_pos",
    "foot_contact",
    "scan_xyz",
    "commands",
    "episode_length",
    "dones",
)


def package(
    source: Path,
    output_dir: Path,
    *,
    drop_seconds: float,
    expected_clips: int,
    workers: int,
) -> Path:
    archive = np.load(source, allow_pickle=False)
    dt = float(archive["dt"])
    start = int(round(drop_seconds / dt))
    frame_count, environment_count = archive["root_pos"].shape[:2]
    clean = np.asarray(archive["clean_envs"], dtype=bool)
    selected = np.flatnonzero(clean)
    if len(selected) != expected_clips:
        raise ValueError(
            f"expected {expected_clips} clean environments, found {len(selected)}; "
            "do not silently package failed rollouts"
        )
    if start < 0 or start >= frame_count - 1:
        raise ValueError(f"drop_seconds={drop_seconds} removes the whole {frame_count}-frame archive")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    terrain_path = output_dir / "terrain.npz"
    np.savez_compressed(
        terrain_path,
        vertices_world=np.asarray(archive["terrain_vertices_world"], dtype=np.float32),
        faces=np.asarray(archive["terrain_faces"], dtype=np.int32),
    )

    shared = {
        "joint_names": archive["joint_names"],
        "body_names": archive["body_names"],
        "foot_body_indices": archive["foot_body_indices"],
        "contact_body_names": archive["contact_body_names"],
        "contact_foot_body_indices": archive["contact_foot_body_indices"],
        "contact_foot_body_names": archive["contact_foot_body_names"],
        "default_joint_position": archive["default_joint_position"],
        "arm_joint_indices": archive["arm_joint_indices"],
    }
    # NpzFile indexing decompresses an entire member before slicing it.  Load
    # each time-major member once; otherwise splitting 200 environments repeats
    # the same decompression 200 times.
    frame_data = {name: np.asarray(archive[name]) for name in FRAME_ARRAYS}
    environment_data = {
        name: np.asarray(archive[name])
        for name in (
            "terrain_name",
            "program_name",
            "terrain_type",
            "terrain_level_initial",
            "terrain_env_origin",
            "command_rmse",
            "mean_stance_ankle_speed",
            "arm_velocity_rms",
            "arm_deviation_rms",
            "maximum_joint_step",
        )
    }
    adjacent_report = {}
    report_path = source.with_suffix(".json")
    if report_path.is_file():
        adjacent_report = json.loads(report_path.read_text())
    terrain_seed = (
        int(archive["terrain_seed"])
        if "terrain_seed" in archive.files
        else adjacent_report.get("terrain_seed")
    )
    terrain_level = (
        int(archive["terrain_level"])
        if "terrain_level" in archive.files
        else adjacent_report.get("terrain_level")
    )
    checkpoint = (
        str(archive["checkpoint"])
        if "checkpoint" in archive.files
        else adjacent_report.get("checkpoint")
    )
    def write_clip(index: int) -> dict:
        terrain = str(environment_data["terrain_name"][index])
        program = str(environment_data["program_name"][index])
        clip_id = f"{index:03d}__{terrain}__{program}"
        path = clips_dir / f"{clip_id}.npz"
        payload = {
            name: frame_data[name][start:, index]
            for name in FRAME_ARRAYS
        }
        payload.update(shared)
        payload.update({
            "dt": np.float32(dt),
            "source_env_index": np.int64(index),
            "terrain_name": np.asarray(terrain),
            "program_name": np.asarray(program),
            "terrain_type": np.int64(environment_data["terrain_type"][index]),
            "terrain_level": np.int64(environment_data["terrain_level_initial"][index]),
            "terrain_env_origin": np.asarray(environment_data["terrain_env_origin"][index], dtype=np.float32),
            "terrain_mesh_relative_path": np.asarray("../terrain.npz"),
            "command_rmse": np.float32(environment_data["command_rmse"][index]),
            "mean_stance_ankle_speed": np.float32(environment_data["mean_stance_ankle_speed"][index]),
            "arm_velocity_rms": np.float32(environment_data["arm_velocity_rms"][index]),
            "arm_deviation_rms": np.float32(environment_data["arm_deviation_rms"][index]),
            "maximum_joint_step": np.float32(environment_data["maximum_joint_step"][index]),
        })
        np.savez_compressed(path, **payload)
        return {
            "clip_id": clip_id,
            "path": str(path.relative_to(output_dir)),
            "source_env_index": int(index),
            "terrain": terrain,
            "program": program,
            "frames": frame_count - start,
            "duration_s": (frame_count - start) * dt,
            "command_rmse": float(environment_data["command_rmse"][index]),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(write_clip, selected.tolist()))

    manifest = {
        "source": str(source.resolve()),
        "checkpoint": checkpoint,
        "terrain_seed": terrain_seed,
        "terrain_level": terrain_level,
        "dt": dt,
        "drop_seconds": drop_seconds,
        "clip_count": len(rows),
        "frames_per_clip": frame_count - start,
        "seconds_per_clip": (frame_count - start) * dt,
        "terrain_mesh": terrain_path.name,
        "clips": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "clips"}, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-seconds", type=float, default=1.0)
    parser.add_argument("--expected-clips", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    package(
        args.source.resolve(),
        args.output_dir.resolve(),
        drop_seconds=args.drop_seconds,
        expected_clips=args.expected_clips,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
