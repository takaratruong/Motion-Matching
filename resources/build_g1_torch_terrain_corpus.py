#!/usr/bin/env python3
"""Build the authenticated expanded native 50 Hz terrain corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "sonic" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from resources.g1_torch_terrain_builder.publish import (
    publish_expanded_corpus,
)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", default="/home/ubuntu/Downloads/artifacts"
    )
    parser.add_argument(
        "--grail-root",
        default="/home/ubuntu/datasets/GRAIL/data/stair_p1",
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
    parser.add_argument("--output", default="build/torch-terrain-expanded")
    parser.add_argument(
        "--report",
        default="build/torch-terrain-expanded-admission.json",
    )
    args = parser.parse_args(argv)
    manifest = publish_expanded_corpus(
        output=args.output,
        source_root=args.source_root,
        grail_root=args.grail_root,
        g1_xml=args.g1_xml,
        flat_motion=args.flat_motion,
    )
    _write_report(Path(args.report), manifest)
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "schema": manifest["schema"],
                "accepted": len(manifest["accepted_clips"]),
                "rejected": len(manifest["rejected_candidates"]),
                "output": str(Path(args.output).resolve()),
                "report": str(Path(args.report).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
