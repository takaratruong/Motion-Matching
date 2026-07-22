import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .g1_reach_builder.annotations import load_annotations
from .g1_reach_builder.artifacts import write_reach_pack
from .g1_reach_builder.build import prepare_reach_pack
from .g1_reach_builder.review import read_review_corpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bilateral G1 reach pack")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    corpus = read_review_corpus(args.review)
    annotations = load_annotations(args.annotations, args.review)
    pending = sum(
        annotation.status == "pending" for annotation in annotations.annotations
    )
    if pending and not args.allow_pending:
        print(
            f"ERROR pending_annotations={pending}; review all proposals or use --allow-pending",
            file=sys.stderr,
        )
        return 1
    try:
        reaches, artifact, features, manifest = prepare_reach_pack(
            corpus, annotations
        )
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    write_reach_pack(args.output, artifact, features, manifest)
    print(
        f"BUILT reach-pack schema=1 captured={manifest['captured_reaches']} "
        f"mirrored={manifest['mirrored_reaches']} total={len(reaches)} "
        f"frames={len(artifact.positions)} pending={pending} output={args.output}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
