"""Build a clean 50 Hz G1 archive from the C490 curb and slope sources.

The released C490 shards predate the enriched per-clip terrain-pose metadata,
but all of these assets share the documented GRAIL placement: the USD is at
the world origin with a -90 degree yaw.  Robot motion and terrain therefore
remain paired without consulting any noisy SONIC rollout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_stairs500_archive import _write_archive_via_local_stage
from .grail_terrain_source import (
    GrailClipRecord,
    GrailGeometry,
    GrailSelectionCandidate,
    G1MujocoFK,
    build_archive_clip,
    load_grail_motion,
)
from .terrain_oracle.source_grail import _load_usd_mesh


DEFAULT_SHARD_ROOT = Path("/move/data/terrain-aware/grail-sweep/shards")
DEFAULT_OUTPUT = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-c490-curb-slope-clean-50hz-v1.zarr"
)
FAMILIES = ("c490_curb_0000", "c490_slope_0000")
TERRAIN_YAW_RAD = math.radians(-89.99999237060547)
TERRAIN_QUATERNION_WXYZ = np.asarray(
    (
        math.cos(0.5 * TERRAIN_YAW_RAD),
        0.0,
        0.0,
        math.sin(0.5 * TERRAIN_YAW_RAD),
    ),
    dtype=np.float32,
)


def _travel_yaw(root_position: np.ndarray) -> float:
    xy = np.asarray(root_position, dtype=np.float64)[:, :2]
    displacement = xy - xy[0]
    farthest = int(np.argmax(np.linalg.norm(displacement, axis=1)))
    direction = displacement[farthest]
    if float(np.linalg.norm(direction)) < 0.05:
        centred = xy - np.mean(xy, axis=0)
        _, _, right = np.linalg.svd(centred, full_matrices=False)
        direction = right[0]
        early = np.mean(np.diff(xy[: min(30, len(xy))], axis=0), axis=0)
        if float(direction @ early) < 0.0:
            direction = -direction
    return math.atan2(float(direction[1]), float(direction[0]))


def _traversal(root_height: np.ndarray) -> tuple[str, float]:
    height = np.asarray(root_height, dtype=np.float64)
    count = min(10, max(3, len(height) // 20))
    delta = float(np.median(height[-count:]) - np.median(height[:count]))
    low, high = np.quantile(height, (0.02, 0.98))
    threshold = max(0.04, 0.20 * float(high - low))
    if delta > threshold:
        return "up", delta
    if delta < -threshold:
        return "down", delta
    return "roundtrip", delta


def _records(shard_root: Path) -> tuple[tuple[str, Path, dict[str, object]], ...]:
    rows: list[tuple[str, Path, dict[str, object]]] = []
    seen: set[str] = set()
    for family in FAMILIES:
        shard = shard_root / family
        raw = json.loads((shard / "clips.json").read_text())
        if not isinstance(raw, list):
            raise ValueError(f"{shard / 'clips.json'} must contain a list")
        for row in raw:
            stem = str(row["stem"])
            if stem in seen:
                raise ValueError(f"duplicate C490 stem: {stem}")
            seen.add(stem)
            rows.append((family.removesuffix("_0000"), shard, row))
    return tuple(rows)


def _bundle_records(
    bundle_root: Path,
) -> tuple[tuple[str, Path, dict[str, object]], ...]:
    """Load all clean curb/slope pairs from a merged GRAIL bundle."""

    raw = json.loads((bundle_root / "clips.json").read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{bundle_root / 'clips.json'} must contain a list")
    rows: list[tuple[str, Path, dict[str, object]]] = []
    seen: set[str] = set()
    for value in raw:
        row = dict(value)
        category = str(row.get("category", ""))
        if category not in ("curb", "slope"):
            continue
        stem = str(row["stem"])
        if stem in seen:
            raise ValueError(f"duplicate bundle stem: {stem}")
        seen.add(stem)
        rows.append((f"bundle_{category}", bundle_root, row))
    return tuple(rows)


def build_archive(
    output: Path,
    *,
    shard_root: Path = DEFAULT_SHARD_ROOT,
    bundle_root: Path | None = None,
    model_path: Path = DEFAULT_G1_MJCF,
    limit: int | None = None,
    progress_every: int = 10,
) -> dict[str, object]:
    source = (
        shard_root.expanduser().resolve()
        if bundle_root is None
        else bundle_root.expanduser().resolve()
    )
    rows = _records(source) if bundle_root is None else _bundle_records(source)
    if limit is not None:
        rows = rows[: int(limit)]
    fk = G1MujocoFK(model_path.expanduser().resolve())
    converted = []
    traversals: Counter[str] = Counter()
    terrain_types: Counter[str] = Counter()
    for index, (family, shard, raw) in enumerate(rows, start=1):
        stem = str(raw["stem"])
        frame_count = int(raw["n_frames"])
        robot = shard / "robot" / f"{stem}.pkl"
        usd = shard / "object_usd" / f"{stem}.usd"
        motion = load_grail_motion(robot, expected_frames=frame_count)
        terrain = _load_usd_mesh(usd, source_asset_sha256="0" * 64)
        traversal, root_delta = _traversal(motion.root_position[:, 2])
        travel_yaw = _travel_yaw(motion.root_position)
        horizontal_range = float(
            np.max(
                np.linalg.norm(
                    motion.root_position[:, :2]
                    - motion.root_position[0, :2],
                    axis=1,
                )
            )
        )
        height_range = float(
            np.max(terrain.vertices_local[:, 2])
            - np.min(terrain.vertices_local[:, 2])
        )
        record = GrailClipRecord(
            family=family,
            shard_path=shard,
            stem=stem,
            robot_path=robot,
            usd_path=usd,
            n_frames=frame_count,
            terrain_position_env=np.zeros(3, dtype=np.float32),
            terrain_rotation_env_wxyz=TERRAIN_QUATERNION_WXYZ.copy(),
            pose_source=(
                "c490_documented_fixed_minus90_yaw"
                if bundle_root is None
                else str(raw.get("pose_source", "legacy_default_yaw"))
            ),
        )
        geometry = GrailGeometry(
            stem=stem,
            fps=25.0,
            n_frames=frame_count,
            n_steps=1,
            rise_m=max(0.01, height_range),
            tread_m=max(0.05, horizontal_range),
            travel_yaw_rad=travel_yaw,
            ascending=traversal == "up",
            root_delta_z_m=root_delta,
        )
        candidate = GrailSelectionCandidate(
            record=record,
            geometry=geometry,
            traversal=traversal,
            approach_heading_delta_deg=0.0,
        )
        converted.append(build_archive_clip(candidate, motion, fk=fk))
        traversals[traversal] += 1
        terrain_types[family] += 1
        if progress_every > 0 and (
            index % progress_every == 0 or index == len(rows)
        ):
            print(f"converted {index}/{len(rows)} curb/slope clips", flush=True)

    destination = output.expanduser().resolve()
    provenance = {
        "schema": "mm-sonic-grail-curb-slope-archive-v2",
        "source_root": str(source),
        "source_layout": "c490_shards" if bundle_root is None else "merged_bundle",
        "g1_mjcf": str(model_path.expanduser().resolve()),
        "source_families": (
            list(FAMILIES)
            if bundle_root is None
            else sorted({str(family) for family, _root, _row in rows})
        ),
        "terrain_transform": {
            "position_world": [0.0, 0.0, 0.0],
            "yaw_deg": -89.99999237060547,
            "quaternion_world_from_usd_wxyz": [
                float(value) for value in TERRAIN_QUATERNION_WXYZ
            ],
        },
        "terrain_type_counts": dict(sorted(terrain_types.items())),
        "traversal_counts": dict(sorted(traversals.items())),
    }
    _write_archive_via_local_stage(
        destination,
        converted,
        provenance=provenance,
    )
    return {
        "output": str(destination),
        "source_clips": len(converted),
        "frames": sum(len(clip.joint_position) for clip in converted),
        "terrain_types": dict(sorted(terrain_types.items())),
        "traversals": dict(sorted(traversals.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    arguments = parser.parse_args(argv)
    result = build_archive(
        arguments.output,
        shard_root=arguments.shard_root,
        bundle_root=arguments.bundle_root,
        model_path=arguments.model,
        limit=arguments.limit,
        progress_every=arguments.progress_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
