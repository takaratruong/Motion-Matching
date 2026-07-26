import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .g1_reach_builder.annotations import load_annotations
from .g1_reach_builder.artifacts import write_reach_pack
from .g1_reach_builder.build import ReachCorpusInput, prepare_combined_reach_pack
from .g1_reach_builder.review import read_review_corpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bilateral G1 reach pack")
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    return parser.parse_args(argv)


def _as_list(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def run(args: argparse.Namespace) -> int:
    reviews = _as_list(args.review)
    annotation_paths = _as_list(args.annotations)
    if not reviews or len(reviews) != len(annotation_paths):
        print(
            f"ERROR review/annotation count mismatch: "
            f"{len(reviews)} reviews, {len(annotation_paths)} annotations",
            file=sys.stderr,
        )
        return 1
    inputs: list[ReachCorpusInput] = []
    pending = 0
    for review_path, annotation_path in zip(reviews, annotation_paths):
        corpus = read_review_corpus(review_path)
        annotations = load_annotations(annotation_path, review_path)
        pending += sum(
            annotation.status == "pending"
            for annotation in annotations.annotations
        )
        inputs.append(ReachCorpusInput(
            label=str(review_path),
            corpus=corpus,
            annotations=annotations,
        ))
    if pending and not args.allow_pending:
        print(
            f"ERROR pending_annotations={pending}; review all proposals or use --allow-pending",
            file=sys.stderr,
        )
        return 1
    try:
        reaches, artifact, features, manifest = prepare_combined_reach_pack(inputs)
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    write_reach_pack(args.output, artifact, features, manifest)
    print(
        f"BUILT reach-pack schema={manifest['version']} captured={manifest['captured_reaches']} "
        f"mirrored={manifest['mirrored_reaches']} total={len(reaches)} "
        f"frames={len(artifact.positions)} paired_returns={manifest['paired_returns']} "
        f"unavailable_returns={manifest['unavailable_returns']} "
        f"return_frames={manifest['return_frame_count']} pending={pending} "
        f"corpora={len(manifest['corpora'])} output={args.output}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
