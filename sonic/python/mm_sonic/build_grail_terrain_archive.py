"""Build the deterministic stage-one GRAIL stair archive.

The archive is flat and directly compatible with ``TerrainKinematicArchive``:
clip ends are exclusive, joints and bodies use the IsaacLab G1 order, and
``body_quat_w`` is explicitly declared XYZW despite its historical field name.
Source terrain poses remain WXYZ in a separately named metadata array.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

import numpy as np

from .grail_terrain_source import (
    DEFAULT_GRAIL_FAMILIES,
    EXPECTED_GRAIL_FAMILY_COUNTS,
    GrailArchiveClip,
    GrailClipRecord,
    GrailGeometry,
    GrailSelectionCandidate,
    G1MujocoFK,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    StagedSelectionConfig,
    build_archive_clip,
    candidate_from_motion,
    discover_grail_clips,
    load_grail_geometry,
    load_grail_motion,
    select_staged_subset,
    stage_one_selection_config,
)


DEFAULT_SHARD_ROOT = Path("/move/data/terrain-aware/grail-sweep/shards")
DEFAULT_GEOMETRY_CSV = Path(
    "/move/u/justingu/Projects/grail-stairs/data/corpus_geometry.csv"
)
DEFAULT_G1_MJCF = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)
ARCHIVE_SCHEMA = "mm-sonic-grail-terrain-archive-v1"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _unicode(values: Iterable[object]) -> np.ndarray:
    strings = tuple(str(value) for value in values)
    width = max(1, max((len(value) for value in strings), default=0))
    return np.asarray(strings, dtype=f"<U{width}")


def _chunks(array: np.ndarray) -> tuple[int, ...]:
    if array.ndim == 0:
        return ()
    first = max(1, min(int(array.shape[0]), 1024))
    return (first,) + tuple(int(value) for value in array.shape[1:])


def _publish_directory_no_replace(
    source: str | Path,
    destination: str | Path,
) -> None:
    """Atomically publish a directory without replacing any existing path."""

    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise RuntimeError(
            "atomic no-replace publication requires Linux renameat2"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def write_grail_terrain_archive(
    output: str | Path,
    clips: Iterable[GrailArchiveClip],
    *,
    provenance: dict[str, object],
) -> None:
    """Atomically concatenate validated clips into a Justin-compatible zarr."""

    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("zarr is required to write the GRAIL archive") from error

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(clips)
    if not values:
        raise ValueError("cannot write an empty GRAIL archive")
    names = tuple(clip.candidate.record.stem for clip in values)
    if len(set(names)) != len(names):
        raise ValueError("archive clip names must be unique")
    if any(clip.fps != 50.0 for clip in values):
        raise ValueError("all archive clips must be exactly 50 Hz")

    frame_counts = np.asarray(
        [len(clip.joint_position) for clip in values], dtype=np.int64
    )
    ends = np.cumsum(frame_counts, dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    arrays: dict[str, np.ndarray] = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "clip_names": _unicode(names),
        "clip_start_idx": starts,
        "clip_end_idx": ends,
        "joint_names": _unicode(ISAACLAB_JOINT_NAMES),
        "body_names": _unicode(ISAACLAB_BODY_NAMES),
        "joint_pos": np.concatenate(
            [clip.joint_position for clip in values], axis=0
        ),
        "joint_vel": np.concatenate(
            [clip.joint_velocity for clip in values], axis=0
        ),
        "body_pos_w": np.concatenate(
            [clip.body_position_world for clip in values], axis=0
        ),
        "body_quat_w": np.concatenate(
            [clip.body_quaternion_world_xyzw for clip in values], axis=0
        ),
        "body_lin_vel_w": np.concatenate(
            [clip.body_linear_velocity_world for clip in values], axis=0
        ),
        "clip_family": _unicode(
            clip.candidate.record.family for clip in values
        ),
        "clip_traversal": _unicode(
            clip.candidate.traversal for clip in values
        ),
        "clip_approach_angle_bin": _unicode(
            clip.candidate.approach_angle_bin for clip in values
        ),
        "clip_approach_heading_delta_deg": np.asarray(
            [
                clip.candidate.approach_heading_delta_deg
                for clip in values
            ],
            dtype=np.float32,
        ),
        "clip_source_frames": np.asarray(
            [clip.candidate.record.n_frames for clip in values],
            dtype=np.int32,
        ),
        "clip_output_frames": frame_counts.astype(np.int32),
        "terrain_position_env": np.stack(
            [
                clip.candidate.record.terrain_position_env
                for clip in values
            ]
        ).astype(np.float32),
        "terrain_rotation_env_wxyz": np.stack(
            [
                clip.candidate.record.terrain_rotation_env_wxyz
                for clip in values
            ]
        ).astype(np.float32),
        "terrain_pose_source": _unicode(
            clip.candidate.record.pose_source for clip in values
        ),
        "terrain_usd_path": _unicode(
            clip.candidate.record.usd_path for clip in values
        ),
        "source_robot_path": _unicode(
            clip.candidate.record.robot_path for clip in values
        ),
        "source_shard_path": _unicode(
            clip.candidate.record.shard_path for clip in values
        ),
        "stair_n_steps": np.asarray(
            [clip.candidate.geometry.n_steps for clip in values],
            dtype=np.int16,
        ),
        "stair_rise_m": np.asarray(
            [clip.candidate.geometry.rise_m for clip in values],
            dtype=np.float32,
        ),
        "stair_tread_m": np.asarray(
            [clip.candidate.geometry.tread_m for clip in values],
            dtype=np.float32,
        ),
        "travel_yaw_rad": np.asarray(
            [clip.candidate.geometry.travel_yaw_rad for clip in values],
            dtype=np.float32,
        ),
        "geometry_ascending": np.asarray(
            [clip.candidate.geometry.ascending for clip in values],
            dtype=np.bool_,
        ),
        "root_delta_z_m": np.asarray(
            [clip.candidate.geometry.root_delta_z_m for clip in values],
            dtype=np.float32,
        ),
    }
    for name, array in arrays.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"archive array {name} contains non-finite values")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        compressor = zarr.Blosc(
            cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE
        )
        root = zarr.open_group(str(temporary), mode="w")
        for name, array in arrays.items():
            root.array(
                name,
                np.ascontiguousarray(array),
                chunks=_chunks(array),
                compressor=compressor,
                overwrite=False,
            )
        root.attrs.update(
            {
                "schema": ARCHIVE_SCHEMA,
                "quaternion_convention": "xyzw",
                "source_root_quaternion_convention": "xyzw",
                "terrain_rotation_convention": "wxyz",
                "clip_end_convention": "exclusive",
                "joint_order": "g1-29dof-isaaclab-v1",
                "body_order": "g1-30body-isaaclab-v1",
                "source_clip_count": len(values),
                "frame_count": int(ends[-1]),
                "provenance_json": json.dumps(
                    provenance, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        del root
        _publish_directory_no_replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def scan_grail_candidates(
    records: Iterable[GrailClipRecord],
    geometry_by_stem: dict[str, GrailGeometry],
    *,
    progress_every: int = 0,
) -> tuple[GrailSelectionCandidate, ...]:
    """Measure traversal/facing for all joined sources without retaining motion."""

    values = tuple(records)
    missing = sorted(
        record.stem for record in values if record.stem not in geometry_by_stem
    )
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"{len(missing)} shard clips are missing geometry; first: {preview}"
        )
    candidates: list[GrailSelectionCandidate] = []
    for index, record in enumerate(values, start=1):
        geometry = geometry_by_stem[record.stem]
        motion = load_grail_motion(
            record.robot_path, expected_frames=record.n_frames
        )
        candidates.append(candidate_from_motion(record, geometry, motion))
        if progress_every > 0 and index % progress_every == 0:
            print(
                f"measured {index}/{len(values)} GRAIL clips",
                file=sys.stderr,
                flush=True,
            )
    return tuple(candidates)


def _counter_json(counter: Counter[object]) -> dict[str, int]:
    return {
        (
            "/".join(str(part) for part in key)
            if isinstance(key, tuple)
            else str(key)
        ): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def build_grail_terrain_archive(
    output: str | Path,
    *,
    shard_root: str | Path = DEFAULT_SHARD_ROOT,
    geometry_csv: str | Path = DEFAULT_GEOMETRY_CSV,
    g1_mjcf: str | Path = DEFAULT_G1_MJCF,
    config: StagedSelectionConfig | None = None,
    progress_every: int = 500,
) -> dict[str, object]:
    """Discover, select, convert, and atomically write stage one."""

    selection_config = config or stage_one_selection_config()
    resolved_shards = Path(shard_root).expanduser().resolve()
    resolved_geometry = Path(geometry_csv).expanduser().resolve()
    resolved_model = Path(g1_mjcf).expanduser().resolve()
    records = discover_grail_clips(
        resolved_shards,
        families=DEFAULT_GRAIL_FAMILIES,
        expected_family_counts=dict(EXPECTED_GRAIL_FAMILY_COUNTS),
    )
    geometry_by_stem = load_grail_geometry(resolved_geometry)
    candidates = scan_grail_candidates(
        records,
        geometry_by_stem,
        progress_every=progress_every,
    )
    selected = select_staged_subset(candidates, config=selection_config)

    fk = G1MujocoFK(resolved_model)
    converted: list[GrailArchiveClip] = []
    for index, candidate in enumerate(selected, start=1):
        motion = load_grail_motion(
            candidate.record.robot_path,
            expected_frames=candidate.record.n_frames,
        )
        converted.append(build_archive_clip(candidate, motion, fk=fk))
        if progress_every > 0 and (
            index % progress_every == 0 or index == len(selected)
        ):
            print(
                f"converted {index}/{len(selected)} selected clips",
                file=sys.stderr,
                flush=True,
            )

    provenance: dict[str, object] = {
        "schema": ARCHIVE_SCHEMA,
        "shard_root": str(resolved_shards),
        "geometry_csv": str(resolved_geometry),
        "g1_mjcf": str(resolved_model),
        "selection_seed": selection_config.seed,
        "selection_family_targets": [
            list(value) for value in selection_config.family_targets
        ],
        "selection_angle_targets": [
            list(value) for value in selection_config.angle_targets
        ],
        "discovered_source_count": len(records),
        "selected_source_count": len(selected),
        "candidate_traversal_counts": _counter_json(
            Counter(candidate.traversal for candidate in candidates)
        ),
        "candidate_family_traversal_counts": _counter_json(
            Counter(
                (candidate.traversal, candidate.record.family)
                for candidate in candidates
            )
        ),
        "candidate_approach_angle_counts": _counter_json(
            Counter(
                (candidate.traversal, candidate.approach_angle_bin)
                for candidate in candidates
            )
        ),
        "selected_stems": [
            candidate.record.stem for candidate in selected
        ],
    }
    write_grail_terrain_archive(output, converted, provenance=provenance)
    return {
        "output": str(Path(output).expanduser().resolve()),
        "source_clips": len(selected),
        "frames": sum(len(clip.joint_position) for clip in converted),
        "traversals": _counter_json(
            Counter(candidate.traversal for candidate in selected)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-zarr",
        "--output",
        dest="output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--shard-root", type=Path, default=DEFAULT_SHARD_ROOT
    )
    parser.add_argument(
        "--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV
    )
    parser.add_argument(
        "--g1-mjcf",
        "--model",
        dest="g1_mjcf",
        type=Path,
        default=DEFAULT_G1_MJCF,
    )
    parser.add_argument(
        "--seed", default="grail-stage-one-v1"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="write progress to stderr every N clips; zero disables it",
    )
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    summary = build_grail_terrain_archive(
        args.output,
        shard_root=args.shard_root,
        geometry_csv=args.geometry_csv,
        g1_mjcf=args.g1_mjcf,
        config=stage_one_selection_config(seed=args.seed),
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
