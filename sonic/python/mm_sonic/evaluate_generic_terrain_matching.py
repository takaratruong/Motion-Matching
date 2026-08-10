"""Evaluate generic terrain-conditioned retrieval on canonical G1 clips.

The self mode is a wiring check: an authored query should retrieve its exact
row without any stair labels.  Leave-one-out is the useful generalisation
check: the target clip and terrain never enter the source database, so success
must come from another motion with a compatible path and local height profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .canonical_terrain_matcher import (
    GenericTerrainSearchConfig,
    build_terrain_row_features,
    exact_trajectory_from_clip,
    generated_state_from_clip,
    load_canonical_terrain_library,
    select_terrain_candidate,
)
from .terrain_oracle.storage import load_corpus
from .torch_motion_features import FEATURE_HORIZON_FRAMES, TorchMotionDatabase


DEFAULT_CORPUS = Path(
    "/move/data/terrain-aware/motion-matching/terrain-oracle-task9.wFKkIV/raw"
)
DEFAULT_FAMILIES = (
    "grail/c490_curb/",
    "grail/c490_slope/",
    "grail/c490_stair_p1/",
    "grail/c490_stair_p2/",
)


def _balanced_source_ids(
    corpus: Path,
    target_clip_id: str,
    *,
    mode: str,
    per_family: int,
    include_flat: bool,
) -> list[str]:
    manifest = load_corpus(corpus)
    all_ids = [record.clip_id for record in manifest.clips]
    selected: list[str] = []
    if include_flat and "flat/takara_walk" in all_ids:
        selected.append("flat/takara_walk")
    for prefix in DEFAULT_FAMILIES:
        family = [clip_id for clip_id in all_ids if clip_id.startswith(prefix)]
        if mode == "leave-one-out":
            family = [clip_id for clip_id in family if clip_id != target_clip_id]
        selected.extend(family[:per_family])
    if mode == "self" and target_clip_id not in selected:
        selected.append(target_clip_id)
    return selected


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    source_ids = _balanced_source_ids(
        args.corpus,
        args.target_clip_id,
        mode=args.mode,
        per_family=args.source_per_family,
        include_flat=not args.no_flat,
    )
    source_library = load_canonical_terrain_library(
        args.corpus, clip_ids=source_ids
    )
    target_library = load_canonical_terrain_library(
        args.corpus, clip_ids=[args.target_clip_id]
    )
    target = target_library.canonical_clips[0]
    target_field = target_library.height_fields[0]
    device = torch.device(args.device)
    database = TorchMotionDatabase.from_folder(
        source_library.folder, device=device
    )
    terrain_rows = build_terrain_row_features(source_library, database)

    source_target_index = (
        source_library.clip_ids.index(args.target_clip_id)
        if args.target_clip_id in source_library.clip_ids
        else None
    )
    stop = target.frame_count - FEATURE_HORIZON_FRAMES[-1]
    frames = np.arange(args.start_frame, stop, args.frame_stride, dtype=np.int64)
    if args.maximum_queries is not None:
        frames = frames[: args.maximum_queries]
    if len(frames) == 0:
        raise ValueError("evaluation frame selection is empty")

    rows = []
    selected_counts: Counter[str] = Counter()
    for frame in frames:
        selection = select_terrain_candidate(
            database,
            terrain_rows,
            generated_state_from_clip(target, int(frame), device=device),
            exact_trajectory_from_clip(target, int(frame), device=device),
            target_field,
            config=GenericTerrainSearchConfig(
                preselection_count=args.preselection_count
            ),
        )
        selected_id = source_library.clip_ids[selection.selected_clip_index]
        selected_counts[selected_id] += 1
        exact = bool(
            source_target_index is not None
            and selection.selected_clip_index == source_target_index
            and selection.selected_frame_index == int(frame)
        )
        same_family = selected_id.split("/")[:2] == args.target_clip_id.split("/")[:2]
        rows.append(
            {
                "query_frame": int(frame),
                "selected_clip_id": selected_id,
                "selected_frame": selection.selected_frame_index,
                "exact_source_row": exact,
                "same_terrain_family": same_family,
                "feature_cost": selection.feature_cost,
                "terrain_cost": selection.terrain_cost,
                "terrain_rms_m": selection.terrain_rms_m,
                "minimum_prospective_foot_clearance_m": (
                    selection.minimum_prospective_foot_clearance_m
                ),
                "total_cost": selection.total_cost,
            }
        )

    terrain_rms = np.asarray([row["terrain_rms_m"] for row in rows], dtype=np.float64)
    report = {
        "schema": "generic-terrain-retrieval-eval/v1",
        "mode": args.mode,
        "corpus": str(args.corpus.resolve()),
        "target_clip_id": args.target_clip_id,
        "source_clip_count": len(source_library.clip_ids),
        "source_row_count": database.feature_shape[0],
        "query_count": len(rows),
        "exact_retrieval_fraction": float(np.mean([row["exact_source_row"] for row in rows])),
        "same_family_fraction": float(np.mean([row["same_terrain_family"] for row in rows])),
        "terrain_rms_mean_m": float(np.mean(terrain_rms)),
        "terrain_rms_p95_m": float(np.quantile(terrain_rms, 0.95)),
        "selected_clip_counts": dict(selected_counts.most_common()),
        "source_clip_ids": list(source_library.clip_ids),
        "queries": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--target-clip-id", required=True)
    parser.add_argument("--mode", choices=("self", "leave-one-out"), default="self")
    parser.add_argument("--source-per-family", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--maximum-queries", type=int)
    parser.add_argument("--preselection-count", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-flat", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "queries"}, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
