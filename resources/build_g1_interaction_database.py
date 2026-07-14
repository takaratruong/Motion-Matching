import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from resources.g1_terrain_builder.kinematics import G1Kinematics

from .g1_interaction_builder.artifacts import write_artifact_set
from .g1_interaction_builder.build import (
    build_labeled_clips,
    build_manifest,
    build_object_dimensions,
    build_validation_report,
    prepare_database,
)
from .g1_interaction_builder.schema import G1_SKELETON
from .g1_interaction_builder.sources import discover_source_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build schema-v1 G1 tabletop interaction artifacts"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/g1_interaction"),
    )
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--heldout-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-rejections", action="store_true")
    args = parser.parse_args(argv)
    if args.target_fps != 25.0:
        parser.error("schema v1 requires --target-fps 25")
    if args.heldout_count < 1:
        parser.error("--heldout-count must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def run(args: argparse.Namespace) -> int:
    sources = sorted(
        discover_source_paths(args.source_root),
        key=lambda source: source.sequence_id,
    )
    if args.limit is not None:
        sources = sources[: args.limit]

    dimensions = build_object_dimensions(sources)
    kinematics = G1Kinematics(str(args.g1_xml))
    included, rejections, numeric_reports = build_labeled_clips(
        sources, dimensions, kinematics
    )

    if rejections and not args.allow_rejections:
        for rejection in rejections:
            print(
                "REJECTED "
                f"sequence_id={rejection.sequence_id} "
                f"stage={rejection.stage} code={rejection.code} "
                f"message={rejection.message}",
                file=sys.stderr,
            )
        print(
            f"ERROR rejected_clips={len(rejections)}; "
            "rerun with --allow-rejections to publish reviewed exclusions",
            file=sys.stderr,
        )
        return 1

    object_count = len({clip.motion.object_id for clip in included})
    if object_count <= args.heldout_count:
        print(
            "ERROR valid clips cannot form nonempty database and held-out "
            f"partitions: objects={object_count} "
            f"heldout_count={args.heldout_count}",
            file=sys.stderr,
        )
        return 1

    database, artifact, features, split = prepare_database(
        included,
        heldout_count=args.heldout_count,
        seed=args.seed,
        skeleton=G1_SKELETON,
    )
    report = build_validation_report(
        len(sources),
        included,
        rejections,
        numeric_reports,
        len(artifact.positions),
    )
    manifest = build_manifest(
        source_root=args.source_root,
        source_count=len(sources),
        included=included,
        rejections=rejections,
        database=database,
        artifact=artifact,
        split=split,
        target_fps=args.target_fps,
        diagnostic_limit=args.limit,
    )
    write_artifact_set(
        args.output,
        artifact,
        features,
        split,
        manifest,
        report,
    )
    print(
        f"BUILT schema=1 fps=25 clips={len(database)} "
        f"frames={len(artifact.positions)} "
        f"heldout_objects={len(split.heldout_objects)} "
        f"rejected={len(rejections)} output={args.output}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
