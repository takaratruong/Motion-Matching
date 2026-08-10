"""Read-only source inventory for the complete walking-only terrain corpus."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from resources.g1_terrain_builder.artifacts import (
    canonical_json_bytes,
    sha256_file,
)

from .full_walking_terrain_lmm_contracts import (
    INVENTORY_SCHEMA,
    FullWalkingInventory,
    SourceRecord,
    build_split_ledger,
    inventory_manifest_bytes,
    source_record_json,
    split_ledger_bytes,
)

STRICT_BANK_MANIFEST_SHA256 = (
    "997d9cb31da2ad611a181456f0d119723902163d1463b2b055f655490bb4783d"
)
PFNN_SOURCE_TABLE_SHA256 = (
    "41ddd2a84dc2fc4a39226bd18463e8704d2915335996cd832b51e14223167db6"
)
EXPECTED_GRAIL_COUNTS = {
    "curb": 1_769,
    "slope": 1_880,
    "stair_p1": 6_094,
    "stair_p2": 6_094,
}
EXPECTED_PFNN_IDENTITIES = 80
EXPECTED_IDENTITIES = 15_918
_GRAIL_INPUTS = {
    "curb": (("robot", ".pkl"), ("object_usd", ".usd"), ("objects", ".pkl"),
             ("recon", ".pkl"), ("meta", ".pkl")),
    "slope": (("robot", ".pkl"), ("object_usd", ".usd"), ("objects", ".pkl"),
              ("recon", ".pkl"), ("meta", ".pkl")),
    "stair_p1": (("robot", ".pkl"), ("object_usd", ".usd"),
                 ("objects", ".pkl"), ("meta", ".pkl")),
    "stair_p2": (("robot", ".pkl"), ("object_usd", ".usd"),
                 ("objects", ".pkl")),
}
_TRIAL_SUFFIX = re.compile(r"__\d+$")


def _regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"missing {label}: {path}") from error
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file: {path}")
    return resolved


def _descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    path = _regular_file(path, "source descriptor")
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source descriptor escapes its authority root: {path}") from error
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _strict_bank_descriptor(bank: Path) -> dict[str, Any]:
    bank = Path(bank)
    manifest_path = bank / "manifest.json" if bank.is_dir() else bank
    manifest_path = _regular_file(manifest_path, "strict bank manifest")
    digest = sha256_file(manifest_path)
    decoded = json.loads(manifest_path.read_bytes())
    if decoded.get("schema") == "g1-hybrid-terrain-lmm-corpus/v2-strict":
        receipt = decoded.get("source_manifest")
        if type(receipt) is not dict or type(receipt.get("path")) is not str:
            raise ValueError("strict cache does not contain a source manifest receipt")
        source_manifest = _regular_file(
            Path(receipt["path"]), "strict bank source manifest"
        )
        if sha256_file(source_manifest) != receipt.get("sha256"):
            raise ValueError("strict cache source manifest SHA-256 mismatch")
        manifest_path = source_manifest
        digest = receipt["sha256"]
        decoded = json.loads(manifest_path.read_bytes())
    if digest != STRICT_BANK_MANIFEST_SHA256:
        raise ValueError("strict bank manifest SHA-256 mismatch")
    if decoded.get("schema") != "g1-terrain-artifacts/v3" \
            or decoded.get("database_frames") != 3_970_932 \
            or decoded.get("total_clips") != 15_815 \
            or decoded.get("grail_clips") != 15_814 \
            or decoded.get("skipped_clips") != 23:
        raise ValueError("strict bank manifest cardinality contract changed")
    return {
        "path": "manifest.json",
        "size_bytes": manifest_path.stat().st_size,
        "sha256": digest,
        "schema": decoded["schema"],
    }


def _pfnn_source_paths(source_table: Path) -> tuple[str, ...]:
    source = source_table.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_table))
    values: object | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "data_terrain"
            for target in statement.targets
        ):
            values = ast.literal_eval(statement.value)
            break
    if type(values) is not list or not all(type(value) is str for value in values):
        raise ValueError("PFNN source table has no literal data_terrain list")
    normalized = tuple(Path(value).as_posix().removeprefix("./") for value in values)
    if len(normalized) != EXPECTED_PFNN_IDENTITIES \
            or len(set(normalized)) != EXPECTED_PFNN_IDENTITIES:
        raise ValueError("PFNN source table must name exactly 80 unique identities")
    return normalized


def _pfnn_semantics(stem: str) -> str:
    if "LocomotionFlat12_000" in stem:
        return "jumpy"
    if any(name in stem for name in ("NewCaptures01_000", "NewCaptures02_000")):
        return "flat"
    if any(name in stem for name in (
        "NewCaptures03_000", "NewCaptures03_001", "NewCaptures03_002",
        "NewCaptures04_000",
    )):
        return "jumpy"
    if "WalkingUpSteps06_000" in stem:
        return "beam"
    if any(name in stem for name in (
        "WalkingUpSteps09_000", "WalkingUpSteps10_000", "WalkingUpSteps11_000",
    )):
        return "flat"
    if "Flat" in stem:
        return "flat"
    return "rocky"


def _pfnn_records(pfnn_root: Path) -> list[SourceRecord]:
    root = Path(pfnn_root).resolve(strict=True)
    source_table = _regular_file(root / "generate_database.py", "PFNN source table")
    table_descriptor = _descriptor(source_table, root=root)
    if table_descriptor["sha256"] != PFNN_SOURCE_TABLE_SHA256 \
            or table_descriptor["size_bytes"] != 21_707:
        raise ValueError("PFNN source table authority changed")
    records: list[SourceRecord] = []
    source_paths = _pfnn_source_paths(source_table)
    source_ids = {
        Path(relative).stem: f"pfnn:{Path(relative).stem}" for relative in source_paths
    }
    for relative in source_paths:
        motion = _regular_file(root / relative, "PFNN BVH")
        stem = motion.stem
        canonical_stem = stem.removesuffix("_mirror")
        inputs = {
            "motion": _descriptor(motion, root=root),
            "gait": _descriptor(motion.with_suffix(".gait"), root=root),
            "phase": _descriptor(motion.with_suffix(".phase"), root=root),
            "footsteps": _descriptor(
                motion.with_name(f"{stem}_footsteps.txt"), root=root
            ),
        }
        source_id = source_ids[stem]
        records.append(
            SourceRecord(
                source_id=source_id,
                canonical_source_id=f"pfnn:{canonical_stem}",
                terrain_id=f"pfnn-terrain:{canonical_stem}",
                mirror_of=(
                    source_ids[canonical_stem] if stem.endswith("_mirror") else None
                ),
                family="flat",
                authority={
                    "kind": "pfnn",
                    "source_table": table_descriptor,
                    "terrain_semantics": _pfnn_semantics(stem),
                    "inputs": inputs,
                },
            )
        )
    if sum(record.mirror_of is not None for record in records) != 40:
        raise ValueError("PFNN inventory must contain 40 canonical/mirror pairs")
    return records


def _grail_records(grail_root: Path, bank: dict[str, Any]) -> list[SourceRecord]:
    root = Path(grail_root).resolve(strict=True)
    records: list[SourceRecord] = []
    counts: Counter[str] = Counter()
    for category in ("curb", "slope", "stair_p1", "stair_p2"):
        category_root = root / category
        robot_root = category_root / "robot"
        robot_paths = sorted(robot_root.glob("*.pkl"), key=lambda path: path.name)
        if len(robot_paths) != EXPECTED_GRAIL_COUNTS[category]:
            raise ValueError(
                f"GRAIL {category} identity count must be "
                f"{EXPECTED_GRAIL_COUNTS[category]}"
            )
        for robot in robot_paths:
            stem = robot.stem
            inputs = {
                role: _descriptor(category_root / role / f"{stem}{suffix}", root=root)
                for role, suffix in _GRAIL_INPUTS[category]
            }
            canonical = _TRIAL_SUFFIX.sub("", stem)
            family = category if category in ("curb", "slope") else "stair"
            records.append(
                SourceRecord(
                    source_id=f"grail:{category}:{stem}",
                    canonical_source_id=f"grail:{canonical}",
                    terrain_id=f"grail-terrain:{canonical}",
                    mirror_of=None,
                    family=family,
                    authority={
                        "kind": "grail",
                        "category": category,
                        "bank_manifest_sha256": bank["sha256"],
                        "inputs": inputs,
                    },
                )
            )
            counts[category] += 1
    if dict(counts) != EXPECTED_GRAIL_COUNTS:
        raise ValueError("GRAIL family counts changed during inventory")
    return records


def _takara_record(takara: Path) -> SourceRecord:
    path = _regular_file(Path(takara), "Takara source")
    descriptor = _descriptor(path, root=path.parent)
    return SourceRecord(
        source_id="takara:takara_walk_50hz",
        canonical_source_id="takara:takara_walk_50hz",
        terrain_id="flat:takara",
        mirror_of=None,
        family="flat",
        authority={"kind": "takara", "inputs": {"motion": descriptor}},
    )


def build_inventory(
    *, bank: Path, pfnn_root: Path, grail_root: Path, takara: Path
) -> FullWalkingInventory:
    """Hash source descriptors and freeze the exact 15,918-identity ledger."""

    bank_descriptor = _strict_bank_descriptor(bank)
    sources = tuple(sorted(
        [
            *_pfnn_records(pfnn_root),
            *_grail_records(grail_root, bank_descriptor),
            _takara_record(takara),
        ],
        key=lambda record: record.source_id,
    ))
    kinds = Counter(record.authority["kind"] for record in sources)
    if kinds != {"pfnn": 80, "grail": 15_837, "takara": 1} \
            or len(sources) != EXPECTED_IDENTITIES:
        raise ValueError("full walking inventory cardinality contract changed")
    identity_payload = _inventory_identity_bytes(sources)
    build_id = hashlib.sha256(identity_payload).hexdigest()
    provisional = FullWalkingInventory(build_id, sources, "")
    manifest_sha256 = hashlib.sha256(inventory_manifest_bytes(provisional)).hexdigest()
    return FullWalkingInventory(build_id, sources, manifest_sha256)


def _inventory_identity_bytes(sources: tuple[SourceRecord, ...]) -> bytes:
    return canonical_json_bytes(
        {
            "schema": INVENTORY_SCHEMA,
            "sources": [source_record_json(record) for record in sources],
        }
    )


def _require_root_stable_authority(value: object, label: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key == "path" and (
                type(child) is not str
                or Path(child).is_absolute()
                or ".." in Path(child).parts
            ):
                raise ValueError(f"{label} contains a host-dependent path")
            _require_root_stable_authority(child, label)
    elif type(value) is list:
        for child in value:
            _require_root_stable_authority(child, label)


def load_inventory(
    path: Path, *, expected_manifest_sha256: str | None = None
) -> FullWalkingInventory:
    path = _regular_file(Path(path), "walking inventory")
    payload = path.read_bytes()
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_manifest_sha256 is not None and (
        type(expected_manifest_sha256) is not str
        or len(expected_manifest_sha256) != 64
        or manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("walking inventory manifest SHA-256 mismatch")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("walking inventory is not UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError("walking inventory is not canonical JSON")
    if type(value) is not dict or set(value) != {"schema", "build_id", "sources"} \
            or value["schema"] != INVENTORY_SCHEMA \
            or type(value["sources"]) is not list:
        raise ValueError("walking inventory has invalid schema or keys")
    records = []
    expected_keys = {
        "source_id", "canonical_source_id", "terrain_id", "mirror_of",
        "family", "authority",
    }
    for item in value["sources"]:
        if type(item) is not dict or set(item) != expected_keys:
            raise ValueError("walking inventory source keys are invalid")
        _require_root_stable_authority(item["authority"], "source authority")
        records.append(SourceRecord(**item))
    sources = tuple(records)
    kinds = Counter(record.authority.get("kind") for record in sources)
    families = Counter(record.family for record in sources)
    if len(sources) != EXPECTED_IDENTITIES \
            or kinds != {"pfnn": 80, "grail": 15_837, "takara": 1} \
            or families != {"flat": 81, "curb": 1_769, "slope": 1_880,
                            "stair": 12_188} \
            or sum(record.mirror_of is not None for record in sources) != 40:
        raise ValueError("walking inventory exact cardinality/family contract changed")
    expected_build_id = hashlib.sha256(_inventory_identity_bytes(sources)).hexdigest()
    if value["build_id"] != expected_build_id:
        raise ValueError("walking inventory build_id does not match canonical content")
    inventory = FullWalkingInventory(
        build_id=value["build_id"],
        sources=sources,
        manifest_sha256=manifest_sha256,
    )
    if inventory_manifest_bytes(inventory) != payload:
        raise ValueError("walking inventory canonical content changed")
    return inventory


def _publish_file_exclusive(payload: bytes, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, output)
        directory = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--bank", type=Path, required=True)
    build.add_argument("--pfnn-root", type=Path, required=True)
    build.add_argument("--grail-root", type=Path, required=True)
    build.add_argument("--takara", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--split-output", type=Path, required=True)
    build.add_argument("--seed", type=int, required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--bank", type=Path, required=True)
    inventory.add_argument("--pfnn-root", type=Path, required=True)
    inventory.add_argument("--grail-root", type=Path, required=True)
    inventory.add_argument("--takara", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("split")
    split.add_argument("--inventory", type=Path, required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command in ("build", "inventory"):
        inventory = build_inventory(
            bank=arguments.bank,
            pfnn_root=arguments.pfnn_root,
            grail_root=arguments.grail_root,
            takara=arguments.takara,
        )
        _publish_file_exclusive(inventory_manifest_bytes(inventory), arguments.output)
        result = {
            "path": str(arguments.output.resolve()),
            "build_id": inventory.build_id,
            "manifest_sha256": inventory.manifest_sha256,
            "sources": len(inventory.sources),
        }
        if arguments.command == "build":
            ledger = build_split_ledger(
                inventory, build_id=inventory.build_id, seed=arguments.seed
            )
            split_payload = split_ledger_bytes(ledger)
            try:
                _publish_file_exclusive(split_payload, arguments.split_output)
            except BaseException:
                arguments.output.unlink(missing_ok=True)
                raise
            result.update({
                "split_path": str(arguments.split_output.resolve()),
                "split_manifest_sha256": hashlib.sha256(split_payload).hexdigest(),
                "assignments": len(ledger.assignments),
            })
    else:
        inventory = load_inventory(arguments.inventory)
        ledger = build_split_ledger(
            inventory, build_id=inventory.build_id, seed=arguments.seed
        )
        payload = split_ledger_bytes(ledger)
        _publish_file_exclusive(payload, arguments.output)
        result = {
            "path": str(arguments.output.resolve()),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "assignments": len(ledger.assignments),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_inventory", "load_inventory", "main"]
