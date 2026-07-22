import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .g1_reach_builder.annotations import (
    create_annotation_document,
    load_annotations,
    update_annotation,
    write_annotations_atomic,
)
from .g1_reach_builder.motions import build_captured_reach
from .g1_reach_builder.review import read_review_corpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm structurally valid default G1 reach proposals"
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    corpus = read_review_corpus(args.review)
    pending = create_annotation_document(args.review)
    completed = []
    for current in pending.annotations:
        try:
            accepted = update_annotation(
                corpus,
                current,
                departure_frame=current.departure_frame,
                grab_frame=current.grab_frame,
                status="accepted",
                note="default proposal structurally confirmed",
            )
            build_captured_reach(corpus, accepted)
        except ValueError as error:
            completed.append(update_annotation(
                corpus,
                current,
                departure_frame=current.departure_frame,
                grab_frame=current.grab_frame,
                status="rejected",
                note=f"structural: {error}",
            ))
        else:
            completed.append(accepted)

    document = replace(pending, annotations=tuple(completed))
    write_annotations_atomic(args.output, document)
    verified = load_annotations(args.output, args.review)
    if any(value.status == "pending" for value in verified.annotations):
        raise RuntimeError("default confirmation left pending annotations")

    totals = Counter(value.status for value in verified.annotations)
    by_source = Counter(
        (value.sequence_id, value.status) for value in verified.annotations
    )
    print(
        f"CONFIRMED total={len(verified.annotations)} "
        f"accepted={totals['accepted']} rejected={totals['rejected']} "
        f"output={args.output}"
    )
    for sequence_id in sorted({value.sequence_id for value in verified.annotations}):
        print(
            f"SOURCE {sequence_id} "
            f"accepted={by_source[(sequence_id, 'accepted')]} "
            f"rejected={by_source[(sequence_id, 'rejected')]}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
