"""Prepare an offline motion-matching zarr for Justin's SONIC collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def prepare(
    corpus: Path,
    output: Path,
    *,
    grail_stairs_root: Path,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    import zarr

    from sonic_port.pipeline.manifest import (
        ClipBinding,
        PipelineManifest,
        RigidTransform,
    )
    from sonic_port.pipeline.prepare import prepare_motion_bundle

    corpus = corpus.expanduser().resolve()
    output = output.expanduser().resolve()
    grail_stairs_root = grail_stairs_root.expanduser().resolve()
    if not corpus.is_dir():
        raise FileNotFoundError(corpus)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prepared bundle: {output}")
    root = zarr.open(str(corpus), mode="r")
    clip_ids = [str(value) for value in np.asarray(root["meta/episode_clip"])]
    categories = [
        str(value) for value in np.asarray(root["meta/episode_category"])
    ]
    if len(clip_ids) != len(categories) or len(clip_ids) != len(set(clip_ids)):
        raise ValueError("invalid corpus clip metadata")
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard index and count must be provided together")
    if shard_count is not None:
        if not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("invalid clip shard")
        selected = [
            index
            for index in range(len(clip_ids))
            if (index // 2) % int(shard_count) == int(shard_index)
        ]
        clip_ids = [clip_ids[index] for index in selected]
        categories = [categories[index] for index in selected]
    flat = (
        grail_stairs_root
        / "data"
        / "sonic_takara"
        / "flat_ground.usd"
    )
    if not flat.is_file():
        raise FileNotFoundError(flat)
    identity = (1.0, 0.0, 0.0, 0.0)
    zero = (0.0, 0.0, 0.0)
    clips = tuple(
        ClipBinding(
            clip_id=clip_id,
            source_id=clip_id,
            category=category,
            terrain=RigidTransform(
                asset=str(flat),
                position=zero,
                rotation_wxyz=identity,
            ),
            reference=RigidTransform(
                asset=None,
                position=zero,
                rotation_wxyz=identity,
            ),
        )
        for clip_id, category in zip(clip_ids, categories, strict=True)
    )
    manifest = PipelineManifest(
        schema_version=1,
        dataset_name=output.parent.name,
        output=output.parent,
        source_type="motion_zarr",
        source_path=corpus,
        quaternion_convention="wxyz",
        clips=clips,
    )
    bundle = prepare_motion_bundle(manifest, output)
    object_usd = output / "object_usd"
    object_usd.mkdir(parents=True, exist_ok=True)
    for clip_id in clip_ids:
        shutil.copy2(flat, object_usd / f"{clip_id}.usd")
    shutil.copy2(flat, output / "flat_placeholder.usd")
    provenance = {
        "corpus": str(corpus),
        "prepared_dir": str(bundle.root),
        "motion_file": str(bundle.motion_file),
        "clip_count": len(clip_ids),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "grail_stairs_root": str(grail_stairs_root),
        "flat_placeholder": str(flat),
    }
    (output / "bridge_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grail-stairs-root",
        type=Path,
        default=Path("/move/u/justingu/Projects/grail-stairs"),
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    prepare(
        args.corpus,
        args.output,
        grail_stairs_root=args.grail_stairs_root,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(f"prepared {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
