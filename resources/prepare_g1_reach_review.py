import argparse
from collections.abc import Sequence
from pathlib import Path

from resources.g1_terrain_builder.kinematics import G1Kinematics

from .g1_reach_builder.review import (
    convert_review_sources,
    write_review_corpus,
)
from .g1_reach_builder.sources import (
    load_gmr_archive,
    load_soma_csv_directory,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare trusted GMR recordings for reach annotation"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--soma-csv-dir", type=Path)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.soma_csv_dir is not None:
        all_sources = load_soma_csv_directory(
            args.soma_csv_dir,
            fps=100.0 if args.source_fps is None else args.source_fps,
        )
    else:
        all_sources = load_gmr_archive(
            args.archive,
            fps_override=args.source_fps,
            include_excluded=True,
        )
    included = [source for source in all_sources if source.disposition == "included"]
    excluded = tuple(
        source.sequence_id
        for source in all_sources
        if source.disposition == "excluded"
    )
    corpus = convert_review_sources(
        included,
        G1Kinematics(str(args.g1_xml)),
        excluded_sequence_ids=excluded,
    )
    write_review_corpus(args.output, corpus)
    rates = sorted(set(float(value) for value in corpus.source_fps))
    print(
        f"BUILT reach-review schema=1 included={len(included)} "
        f"excluded={len(excluded)} frames={len(corpus.positions)} "
        f"source_fps={','.join(f'{rate:g}' for rate in rates)} "
        f"target_fps={corpus.fps:g} output={args.output}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
