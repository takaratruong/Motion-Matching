"""Build a JSONL bank of composable clean stairs500 support fragments."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from typing import Iterable

from .terrain_oracle.fragments import SegmentationConfig
from .terrain_oracle.stairs500_fragments import (
    StairFragmentRecord,
    process_archive_clip,
)


DEFAULT_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
DEFAULT_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def write_fragment_bank(
    output: str | Path,
    *,
    archive_path: str | Path,
    clip_indices: Iterable[int],
    fragments: Iterable[StairFragmentRecord],
) -> dict[str, object]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    indices = tuple(int(index) for index in clip_indices)
    records = tuple(fragments)
    with (destination / "fragments.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    summary: dict[str, object] = {
        "archive_path": str(Path(archive_path)),
        "clip_indices": list(indices),
        "clip_count": len(indices),
        "fragment_count": len(records),
        "roles": dict(
            sorted(Counter(record.role.value for record in records).items())
        ),
        "directions": dict(
            sorted(
                Counter(record.source_direction for record in records).items()
            )
        ),
        "fragments_file": "fragments.jsonl",
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _process_index(
    index: int,
    *,
    archive_path: str,
    model_path: str,
    config: SegmentationConfig,
) -> tuple[StairFragmentRecord, ...]:
    return process_archive_clip(archive_path, index, model_path, config)


def build_stairs500_fragment_bank(
    archive_path: str | Path,
    output: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    clip_indices: Iterable[int],
    workers: int = 1,
    config: SegmentationConfig = SegmentationConfig(),
) -> dict[str, object]:
    indices = tuple(int(index) for index in clip_indices)
    worker = partial(
        _process_index,
        archive_path=str(Path(archive_path)),
        model_path=str(Path(model_path)),
        config=config,
    )
    if workers <= 1:
        chunks = tuple(worker(index) for index in indices)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunks = tuple(executor.map(worker, indices))
    fragments = tuple(record for chunk in chunks for record in chunk)
    return write_fragment_bank(
        output,
        archive_path=archive_path,
        clip_indices=indices,
        fragments=fragments,
    )


def _indices(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=_indices, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--context-frames", type=int, default=8)
    args = parser.parse_args(argv)
    summary = build_stairs500_fragment_bank(
        args.archive,
        args.output,
        model_path=args.model,
        clip_indices=args.indices,
        workers=args.workers,
        config=SegmentationConfig(
            stable_frames=args.stable_frames,
            context_before=args.context_frames,
            context_after=args.context_frames,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
