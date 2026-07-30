#!/usr/bin/env python3
"""Build the small native 50 Hz Torch stair-motion dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "sonic" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from resources.g1_torch_stair_builder.publish import publish_stair_slice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output", default="build/torch-stair-small")
    args = parser.parse_args(argv)
    manifest = publish_stair_slice(
        output=args.output,
        grail_root=args.grail_root,
        g1_xml=args.g1_xml,
        flat_motion=args.flat_motion,
    )
    print(
        json.dumps(
            {
                "status": "BUILT",
                "schema": manifest["schema"],
                "clips": len(manifest["clips"]),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
