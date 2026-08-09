"""Audit and offline-retarget the minimal released-PFNN controller slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

from mm_sonic.retarget_pfnn_bvh_g1 import retarget_sample
from mm_sonic.terrain_pfnn.source_pfnn_released import (
    PFNNSliceRole,
    audit_released_pfnn_metrics,
    discover_released_pfnn_records,
    select_vertical_slice,
    vertical_slice_receipt,
)


_SCHEMA = "g1-pfnn-vertical-slice-retarget/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {"schema", "status", "selection_sha256", "items"}
_ITEM_FIELDS = {
    "stem",
    "role",
    "start_frame_120hz",
    "stop_frame_120hz",
    "coverage",
    "output",
    "output_sha256",
    "receipt",
    "receipt_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_file(root: Path, value: object) -> Path:
    if type(value) is not str or not value or Path(value).is_absolute():
        raise ValueError("retarget artifact path is invalid")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("retarget artifact path escapes output")
    return path


def _validate_item(root: Path, item: object) -> None:
    if type(item) is not dict or set(item) != _ITEM_FIELDS:
        raise ValueError("retarget manifest item fields are invalid")
    if (
        type(item["stem"]) is not str
        or not item["stem"]
        or item["role"] not in ("train", "validation")
        or type(item["start_frame_120hz"]) is not int
        or type(item["stop_frame_120hz"]) is not int
        or item["stop_frame_120hz"] <= item["start_frame_120hz"]
        or not isinstance(item["coverage"], list)
    ):
        raise ValueError("retarget manifest item metadata is invalid")
    output = _relative_file(root, item["output"])
    receipt = _relative_file(root, item["receipt"])
    for label, path in (("output", output), ("receipt", receipt)):
        expected = item[f"{label}_sha256"]
        if (
            type(expected) is not str
            or _SHA256_RE.fullmatch(expected) is None
            or not path.is_file()
            or _sha256(path) != expected
        ):
            raise ValueError(f"retarget {label} digest mismatch")
    try:
        receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("retarget receipt is invalid") from error
    if (
        type(receipt_document) is not dict
        or receipt_document.get("output_sha256") != item["output_sha256"]
    ):
        raise ValueError("retarget receipt output digest mismatch")


def _load_manifest(path: Path, selection_sha256: str) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("retarget resume manifest is invalid") from error
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_FIELDS
        or manifest["schema"] != _SCHEMA
        or manifest["status"] not in ("building", "accepted")
        or manifest["selection_sha256"] != selection_sha256
        or not isinstance(manifest["items"], list)
    ):
        raise ValueError("retarget resume manifest contract is invalid")
    return manifest


def retarget_vertical_slice(
    selection: tuple[PFNNSliceRole, ...],
    *,
    output: Path,
    gmr_root: Path,
    retarget_project_root: Path,
    resume: bool = False,
    retarget_one: Callable[..., dict[str, object]] = retarget_sample,
) -> dict[str, object]:
    """Retarget exactly four sealed PFNN intervals with verified resume."""

    receipt = vertical_slice_receipt(selection)
    selection_sha256 = str(receipt["sha256"])
    root = Path(output).expanduser().resolve()
    manifest_path = root / "retarget-manifest.json"
    selection_path = root / "source-selection.json"
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing to overwrite nonempty output: {root}")
        manifest = _load_manifest(manifest_path, selection_sha256)
        try:
            existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("retarget source selection is invalid") from error
        if existing_selection != receipt:
            raise ValueError("retarget source selection digest mismatch")
        for item in manifest["items"]:
            _validate_item(root, item)
        if manifest["status"] == "accepted":
            if len(manifest["items"]) != len(selection):
                raise ValueError("accepted retarget manifest is incomplete")
            return manifest
    elif resume:
        raise ValueError("retarget resume requires an existing output")
    else:
        root.mkdir(parents=True, exist_ok=False)
        _atomic_json(selection_path, receipt)
        manifest = {
            "schema": _SCHEMA,
            "status": "building",
            "selection_sha256": selection_sha256,
            "items": [],
        }
        _atomic_json(manifest_path, manifest)

    completed = {str(item["stem"]): item for item in manifest["items"]}
    if len(completed) != len(manifest["items"]):
        raise ValueError("retarget resume manifest contains duplicate stems")
    for selected in selection:
        if selected.record.stem in completed:
            item = completed[selected.record.stem]
            expected_identity = (
                selected.role,
                selected.start_frame_120hz,
                selected.stop_frame_120hz,
                list(selected.coverage),
            )
            actual_identity = (
                item["role"],
                item["start_frame_120hz"],
                item["stop_frame_120hz"],
                item["coverage"],
            )
            if actual_identity != expected_identity:
                raise ValueError("retarget resume item conflicts with selection")
            continue
        relative = (
            Path("retargets")
            / f"{selected.record.stem}__{selected.start_frame_120hz:05d}_"
            f"{selected.stop_frame_120hz:05d}.npz"
        )
        destination = root / relative
        result = retarget_one(
            source=selected.record.bvh_path,
            gmr_root=Path(gmr_root),
            retarget_project_root=Path(retarget_project_root),
            output=destination,
            start_frame=selected.start_frame_120hz,
            frame_count=selected.stop_frame_120hz - selected.start_frame_120hz,
            warmup_frames=120,
            grounding="source",
        )
        receipt_artifact = destination.with_suffix(".receipt.json")
        if (
            type(result) is not dict
            or result.get("status") != "accepted"
            or Path(str(result.get("output"))).resolve() != destination.resolve()
            or Path(str(result.get("receipt"))).resolve() != receipt_artifact.resolve()
            or result.get("output_sha256") != _sha256(destination)
            or not receipt_artifact.is_file()
        ):
            raise ValueError("retarget result does not match the requested item")
        item = {
            "stem": selected.record.stem,
            "role": selected.role,
            "start_frame_120hz": selected.start_frame_120hz,
            "stop_frame_120hz": selected.stop_frame_120hz,
            "coverage": list(selected.coverage),
            "output": relative.as_posix(),
            "output_sha256": _sha256(destination),
            "receipt": receipt_artifact.relative_to(root).as_posix(),
            "receipt_sha256": _sha256(receipt_artifact),
        }
        _validate_item(root, item)
        manifest["items"].append(item)
        _atomic_json(manifest_path, manifest)
    manifest["status"] = "accepted"
    _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfnn-root", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--retarget-project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    pfnn_root = arguments.pfnn_root.expanduser().resolve(strict=True)
    records = discover_released_pfnn_records(pfnn_root / "data" / "animations")
    metrics = audit_released_pfnn_metrics(records, pfnn_root)
    selection = select_vertical_slice(records, metrics)
    manifest = retarget_vertical_slice(
        selection,
        output=arguments.output,
        gmr_root=arguments.gmr_root,
        retarget_project_root=arguments.retarget_project_root,
        resume=arguments.resume,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
