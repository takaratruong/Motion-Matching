import argparse
from collections.abc import Sequence
import os
from pathlib import Path

from .g1_interaction_builder.metadata import GRAIL_DATASET_ID


DATASET_ID = GRAIL_DATASET_ID
ALLOW_PATTERNS = (
    "data/pickup_table/robot/*.pkl",
    "data/pickup_table/objects/*.pkl",
    "data/pickup_table/meta/*.pkl",
    "data/pickup_table/object_usd/*.usd",
    "data/pickup_table/object_usd/textures/*",
)


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the GRAIL pickup-table source subset"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(f"dataset={DATASET_ID}")
        for pattern in ALLOW_PATTERNS:
            print(f"allow_pattern={pattern}")
        return 0
    _snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=args.output,
        allow_patterns=list(ALLOW_PATTERNS),
        token=args.token,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
