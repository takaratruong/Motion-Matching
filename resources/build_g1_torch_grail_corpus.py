#!/usr/bin/env python3
"""Build a deterministic balanced or full GRAIL terrain-motion corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "sonic" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from resources.g1_torch_terrain_builder.bulk_grail import (
    discover_bulk_grail_candidates,
    resolve_bulk_grail_candidates,
    select_bulk_grail_candidates,
)
from resources.g1_torch_terrain_builder.publish import publish_expanded_corpus


_PARTITIONS = ("curb", "stair_p1", "stair_p2")
_INVENTORY_SCHEMA = "g1-grail-terrain-inventory/v1"
_INVENTORY_REVISION = "943946a972d5de2eb0d2ff214b236d0e43575fd7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, manifest: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _selection_metadata(
    *,
    dataset_root: Path,
    partitions: tuple[str, ...],
    candidates,
    selected,
    limit_per_partition: int | None,
    selection_seed: int,
) -> dict:
    available = Counter(candidate.partition for candidate in candidates)
    selected_count = Counter(candidate.partition for candidate in selected)
    return {
        "inventory_schema": _INVENTORY_SCHEMA,
        "inventory_revision": _INVENTORY_REVISION,
        "inventory_sha256": _sha256(
            dataset_root / "g1_mm_inventory.json"
        ),
        "partitions": list(partitions),
        "available_by_partition": {
            partition: available[partition] for partition in partitions
        },
        "selected_by_partition": {
            partition: selected_count[partition] for partition in partitions
        },
        "limit_per_partition": limit_per_partition,
        "selection_seed": selection_seed,
        "selected_logical_names": [
            candidate.logical_name for candidate in selected
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default="/home/ubuntu/datasets/GRAIL"
    )
    parser.add_argument(
        "--partition",
        dest="partitions",
        action="append",
        choices=_PARTITIONS,
    )
    parser.add_argument("--limit-per-partition", type=int)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--source-root", default="/home/ubuntu/Downloads/artifacts"
    )
    parser.add_argument(
        "--g1-xml",
        default=(
            "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/"
            "g1_29dof.xml"
        ),
    )
    parser.add_argument(
        "--flat-motion",
        default=(
            "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz"
        ),
    )
    parser.add_argument(
        "--output", default="build/torch-grail-terrain-representative"
    )
    parser.add_argument(
        "--report",
        default="build/torch-grail-terrain-representative-report.json",
    )
    args = parser.parse_args(argv)
    partitions = (
        tuple(args.partitions) if args.partitions is not None else _PARTITIONS
    )
    if len(partitions) != len(set(partitions)):
        parser.error("--partition values must be unique")
    if (
        args.limit_per_partition is not None
        and args.limit_per_partition <= 0
    ):
        parser.error("--limit-per-partition must be positive")

    dataset_root = Path(args.dataset_root).resolve()
    candidates = discover_bulk_grail_candidates(dataset_root, partitions)
    selected = select_bulk_grail_candidates(
        candidates,
        limit_per_partition=args.limit_per_partition,
        seed=args.selection_seed,
    )
    resolved = resolve_bulk_grail_candidates(dataset_root, selected)
    metadata = _selection_metadata(
        dataset_root=dataset_root,
        partitions=partitions,
        candidates=candidates,
        selected=selected,
        limit_per_partition=args.limit_per_partition,
        selection_seed=args.selection_seed,
    )
    manifest = publish_expanded_corpus(
        output=args.output,
        source_root=args.source_root,
        grail_root=dataset_root,
        g1_xml=args.g1_xml,
        flat_motion=args.flat_motion,
        resolved_sources=resolved,
        corpus_metadata=metadata,
    )
    report = Path(args.report).resolve()
    _write_report(report, manifest)
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "schema": manifest["schema"],
                "selected": len(selected),
                "accepted": len(manifest["accepted_clips"]),
                "rejected": len(manifest["rejected_candidates"]),
                "output": str(Path(args.output).resolve()),
                "report": str(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
