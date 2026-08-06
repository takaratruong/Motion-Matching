"""Select a compact, diverse stairs500 source set for terrain maneuvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE


DEFAULT_FAMILIES = ("stairs500_stair_p1", "stairs500_stair_p2")
DEFAULT_TRAVERSALS = ("up", "down")


def _farthest_point_order(features: np.ndarray) -> tuple[int, ...]:
    """Return a deterministic k-center ordering over standardized features."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("features must be a finite, non-empty matrix")
    span = np.ptp(values, axis=0)
    scaled = (values - np.min(values, axis=0)) / np.where(span > 0.0, span, 1.0)
    centre = np.mean(scaled, axis=0)
    first = int(np.argmin(np.linalg.norm(scaled - centre, axis=1)))
    selected = [first]
    distance = np.linalg.norm(scaled - scaled[first], axis=1)
    distance[first] = -1.0
    while len(selected) < len(scaled):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(
            distance,
            np.linalg.norm(scaled - scaled[index], axis=1),
        )
        distance[np.asarray(selected, dtype=np.int64)] = -1.0
    return tuple(selected)


def select_sources(
    archive_path: str | Path,
    *,
    per_traversal: int = 60,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    traversals: tuple[str, ...] = DEFAULT_TRAVERSALS,
) -> dict[str, object]:
    """Choose a diverse, direction-balanced set from the requested families."""

    import zarr

    archive_path = Path(archive_path).expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    names = np.asarray(archive["clip_names"][:]).astype(str)
    clip_family = np.asarray(archive["clip_family"][:]).astype(str)
    traversal = np.asarray(archive["clip_traversal"][:]).astype(str)
    family_code = np.asarray(
        [families.index(value) if value in families else -1 for value in clip_family],
        dtype=np.float64,
    )
    feature = np.stack(
        (
            family_code,
            np.asarray(archive["stair_n_steps"][:], dtype=np.float64),
            np.asarray(archive["stair_rise_m"][:], dtype=np.float64),
            np.asarray(archive["stair_tread_m"][:], dtype=np.float64),
            np.asarray(
                archive["clip_approach_heading_delta_deg"][:],
                dtype=np.float64,
            ),
            np.abs(np.asarray(archive["root_delta_z_m"][:], dtype=np.float64)),
        ),
        axis=1,
    )
    rows: list[dict[str, object]] = []
    rank = 0
    family_mask = np.isin(clip_family, np.asarray(families))
    for direction in traversals:
        candidates = np.flatnonzero(family_mask & (traversal == direction))
        if len(candidates) < int(per_traversal):
            raise ValueError(
                f"{direction} has {len(candidates)} eligible clips, fewer "
                f"than requested {per_traversal}"
            )
        ordering = _farthest_point_order(feature[candidates])
        for local_index in ordering[: int(per_traversal)]:
            index = int(candidates[local_index])
            rows.append(
                {
                    "selection_rank": rank,
                    "clip_index": index,
                    "clip_name": names[index],
                    "clip_family": clip_family[index],
                    "clip_traversal": traversal[index],
                    "stair_n_steps": int(archive["stair_n_steps"][index]),
                    "stair_rise_m": float(archive["stair_rise_m"][index]),
                    "stair_tread_m": float(archive["stair_tread_m"][index]),
                    "approach_heading_delta_deg": float(
                        archive["clip_approach_heading_delta_deg"][index]
                    ),
                }
            )
            rank += 1
    if len({int(row["clip_index"]) for row in rows}) != len(rows):
        raise RuntimeError("selection contains duplicate clip indices")
    return {
        "schema": "stairs500-terrain-maneuver-source-selection/v1",
        "archive": str(archive_path),
        "per_traversal": int(per_traversal),
        "families": list(families),
        "traversals": list(traversals),
        "source_count": len(rows),
        "expected_candidate_count": 2 * len(rows),
        "sources": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-traversal", type=int, default=60)
    arguments = parser.parse_args(argv)
    if arguments.per_traversal < 1:
        parser.error("--per-traversal must be positive")
    result = select_sources(
        arguments.archive,
        per_traversal=arguments.per_traversal,
    )
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "source_count": result["source_count"],
                "expected_candidate_count": result["expected_candidate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
