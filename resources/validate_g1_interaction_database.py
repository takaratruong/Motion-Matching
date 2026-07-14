import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path

from .g1_interaction_builder.artifacts import read_artifact_set


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate schema-v1 G1 tabletop interaction artifacts"
    )
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(input_path: Path) -> str:
    input_path = Path(input_path)
    artifact, features, manifest, split, _ = read_artifact_set(input_path)
    database_hash = _sha256(input_path / "interaction_database.bin")
    feature_hash = _sha256(input_path / "interaction_features.bin")
    return (
        f"VALID schema={manifest['schema_version']} fps={artifact.fps} "
        f"bones={artifact.positions.shape[1]} "
        f"features={features.values.shape[1]} "
        f"clips={len(artifact.range_starts)} "
        f"frames={len(artifact.positions)} "
        f"heldout_objects={len(split['heldout_objects'])} "
        f"db_sha256={database_hash} feature_sha256={feature_hash}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(validate(args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
