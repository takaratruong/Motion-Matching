"""Convert Justin's clean 500-clip GRAIL stair selection to a 50 Hz archive.

The selection manifest is authoritative for clip membership and order.  The
four bundle directories provide the paired clean robot pickle, terrain USD,
and the full per-clip terrain pose.  In particular, terrain yaw is copied from
``clips.json`` without canonicalizing it to a single stair orientation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Iterable

import numpy as np

from .build_grail_terrain_archive import (
    DEFAULT_G1_MJCF,
    DEFAULT_GEOMETRY_CSV,
    write_grail_terrain_archive,
)
from .grail_terrain_source import (
    GrailArchiveClip,
    GrailClipRecord,
    G1MujocoFK,
    build_archive_clip,
    candidate_from_motion,
    load_grail_geometry,
    load_grail_motion,
)


DEFAULT_STEMS_MANIFEST = Path(
    "/move/data/terrain-aware/grail-sweep/stairs500_v1/stems_final.txt"
)
DEFAULT_BUNDLE_ROOT = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "grail_stairs500_sonic_v1/_group_bundles"
)
ARCHIVE_SCHEMA = "mm-sonic-grail-stairs500-archive-v1"
_BUNDLE_NAME = re.compile(
    r"^group(?P<group>\d+)_(?P<partition>stair_p[12])_(?P<shard>\d+)$"
)


def read_stem_manifest(
    path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Read unique, non-empty stems in their manifest order."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stems = tuple(
        line.strip()
        for line in source.read_text().splitlines()
        if line.strip()
    )
    if not stems:
        raise ValueError(f"stem manifest is empty: {source}")
    if len(set(stems)) != len(stems):
        raise ValueError(f"stem manifest contains duplicate entries: {source}")
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        stems = stems[:limit]
    return stems


def _read_bundle_records(bundle: Path) -> tuple[GrailClipRecord, ...]:
    match = _BUNDLE_NAME.fullmatch(bundle.name)
    if match is None:
        raise ValueError(f"unrecognized stairs500 bundle name: {bundle.name}")
    try:
        raw_records = json.loads((bundle / "clips.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {bundle / 'clips.json'}") from error
    if not isinstance(raw_records, list):
        raise ValueError(f"{bundle / 'clips.json'} must contain a list")

    family = f"stairs500_{match.group('partition')}"
    records: list[GrailClipRecord] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError(f"{bundle / 'clips.json'} contains a non-object")
        stem = raw.get("stem")
        if not isinstance(stem, str) or not stem:
            raise ValueError(f"{bundle / 'clips.json'} contains an invalid stem")
        robot_path = bundle / "robot" / f"{stem}.pkl"
        usd_path = bundle / "object_usd" / f"{stem}.usd"
        if not robot_path.is_file() or not usd_path.is_file():
            raise ValueError(f"{stem}: bundle is missing robot pickle or USD")
        n_frames = raw.get("n_frames")
        terrain = raw.get("terrain")
        if type(n_frames) is not int or n_frames < 2:
            raise ValueError(f"{stem}: n_frames must be an integer >= 2")
        if not isinstance(terrain, dict):
            raise ValueError(f"{stem}: terrain pose is missing")
        position = np.asarray(terrain.get("position_env"), dtype=np.float64)
        rotation = np.asarray(
            terrain.get("rotation_env_wxyz"), dtype=np.float64
        )
        if (
            position.shape != (3,)
            or rotation.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
        ):
            raise ValueError(f"{stem}: terrain pose is invalid")
        if not np.isclose(np.linalg.norm(rotation), 1.0, atol=1.0e-4):
            raise ValueError(f"{stem}: terrain WXYZ quaternion is not normalized")
        records.append(
            GrailClipRecord(
                family=family,
                shard_path=bundle,
                stem=stem,
                robot_path=robot_path,
                usd_path=usd_path,
                n_frames=n_frames,
                terrain_position_env=np.ascontiguousarray(
                    position, dtype=np.float32
                ),
                terrain_rotation_env_wxyz=np.ascontiguousarray(
                    rotation, dtype=np.float32
                ),
                pose_source=str(raw.get("pose_source") or "bundle"),
            )
        )
    return tuple(records)


def discover_stairs500_records(
    bundle_root: str | Path,
    stems: Iterable[str],
) -> tuple[GrailClipRecord, ...]:
    """Join manifest stems to the four readable bundles, in manifest order."""

    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    bundles = tuple(
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / "clips.json").is_file()
        and (path / "robot").is_dir()
        and (path / "object_usd").is_dir()
        and _BUNDLE_NAME.fullmatch(path.name) is not None
    )
    if len(bundles) != 4:
        raise ValueError(
            f"expected four readable stairs500 bundles under {root}, "
            f"found {len(bundles)}"
        )

    by_stem: dict[str, GrailClipRecord] = {}
    for bundle in bundles:
        for record in _read_bundle_records(bundle):
            if record.stem in by_stem:
                raise ValueError(
                    f"duplicate stairs500 stem across bundles: {record.stem}"
                )
            by_stem[record.stem] = record

    ordered_stems = tuple(str(stem) for stem in stems)
    missing = tuple(stem for stem in ordered_stems if stem not in by_stem)
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"{len(missing)} manifest stems are missing from bundles; "
            f"first: {preview}"
        )
    return tuple(by_stem[stem] for stem in ordered_stems)


def _counter_json(counter: Counter[object]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _write_archive_via_local_stage(
    output: str | Path,
    clips: Iterable[GrailArchiveClip],
    *,
    provenance: dict[str, object],
    staging_root: str | Path | None = None,
) -> None:
    """Use the existing writer locally, then copy its zarr onto shared storage."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = (
        None
        if staging_root is None
        else str(Path(staging_root).expanduser().resolve())
    )
    with tempfile.TemporaryDirectory(
        prefix="mm-sonic-stairs500-",
        dir=root,
    ) as directory:
        staged = Path(directory) / destination.name
        write_grail_terrain_archive(staged, clips, provenance=provenance)
        shutil.move(str(staged), str(destination))


def build_stairs500_archive(
    output: str | Path,
    *,
    stems_manifest: str | Path = DEFAULT_STEMS_MANIFEST,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    geometry_csv: str | Path = DEFAULT_GEOMETRY_CSV,
    g1_mjcf: str | Path = DEFAULT_G1_MJCF,
    limit: int | None = None,
    progress_every: int = 25,
) -> dict[str, object]:
    """Convert the selected clean stair clips and write one 50 Hz zarr."""

    manifest_path = Path(stems_manifest).expanduser().resolve()
    bundle_path = Path(bundle_root).expanduser().resolve()
    geometry_path = Path(geometry_csv).expanduser().resolve()
    model_path = Path(g1_mjcf).expanduser().resolve()
    stems = read_stem_manifest(manifest_path, limit=limit)
    records = discover_stairs500_records(bundle_path, stems)
    geometry_by_stem = load_grail_geometry(geometry_path)
    missing_geometry = tuple(
        record.stem
        for record in records
        if record.stem not in geometry_by_stem
    )
    if missing_geometry:
        preview = ", ".join(missing_geometry[:3])
        raise ValueError(
            f"{len(missing_geometry)} selected stems lack geometry; "
            f"first: {preview}"
        )

    fk = G1MujocoFK(model_path)
    converted: list[GrailArchiveClip] = []
    for index, record in enumerate(records, start=1):
        geometry = geometry_by_stem[record.stem]
        motion = load_grail_motion(
            record.robot_path,
            expected_frames=record.n_frames,
        )
        candidate = candidate_from_motion(record, geometry, motion)
        converted.append(build_archive_clip(candidate, motion, fk=fk))
        if progress_every > 0 and (
            index % progress_every == 0 or index == len(records)
        ):
            print(
                f"converted {index}/{len(records)} stairs500 clips",
                file=sys.stderr,
                flush=True,
            )

    candidates = tuple(clip.candidate for clip in converted)
    provenance: dict[str, object] = {
        "schema": ARCHIVE_SCHEMA,
        "stems_manifest": str(manifest_path),
        "bundle_root": str(bundle_path),
        "geometry_csv": str(geometry_path),
        "g1_mjcf": str(model_path),
        "manifest_clip_count": len(
            read_stem_manifest(manifest_path)
        ),
        "selected_clip_count": len(records),
        "limit": limit,
        "selected_stems": list(stems),
        "terrain_pose_policy": "preserve_clips_json_position_and_wxyz_rotation",
    }
    _write_archive_via_local_stage(
        output,
        converted,
        provenance=provenance,
    )
    return {
        "output": str(Path(output).expanduser().resolve()),
        "source_clips": len(converted),
        "frames": sum(len(clip.joint_position) for clip in converted),
        "families": _counter_json(
            Counter(candidate.record.family for candidate in candidates)
        ),
        "traversals": _counter_json(
            Counter(candidate.traversal for candidate in candidates)
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
        "--stems-manifest",
        type=Path,
        default=DEFAULT_STEMS_MANIFEST,
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
    )
    parser.add_argument(
        "--geometry-csv",
        type=Path,
        default=DEFAULT_GEOMETRY_CSV,
    )
    parser.add_argument(
        "--g1-mjcf",
        "--model",
        dest="g1_mjcf",
        type=Path,
        default=DEFAULT_G1_MJCF,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="write progress to stderr every N clips; zero disables it",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    summary = build_stairs500_archive(
        args.output,
        stems_manifest=args.stems_manifest,
        bundle_root=args.bundle_root,
        geometry_csv=args.geometry_csv,
        g1_mjcf=args.g1_mjcf,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
