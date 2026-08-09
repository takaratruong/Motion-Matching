"""Combine the retargeted released-PFNN corpus with native-G1 GRAIL rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .build_g1_pfnn_vertical_dataset import (
    VerticalDataset,
    VerticalSplitArrays,
    _dataset_digest,
    load_vertical_dataset,
    save_vertical_dataset,
)
from .terrain_pfnn.dataset import PFNNShardDataset
from .terrain_pfnn.dataset import denormalize_pfnn_input
from .terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from .train_classic_g1_pfnn import (
    _grail_train_validation_masks,
    _mirror_normalized_examples,
)


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _grail_physical_rows(dataset: object) -> dict[str, np.ndarray]:
    required = ("x_mean", "x_std", "y_mean", "y_std")
    if getattr(dataset, "split", None) != "train" or not all(
        hasattr(dataset, name) for name in required
    ):
        raise ValueError("GRAIL source must be a normalized train dataset")
    rows = [dataset[index] for index in range(len(dataset))]
    rows = [row for row in rows if str(row.get("clip_id", "")).startswith("terrain_slopes__")]
    if not rows:
        raise ValueError("GRAIL source contains no terrain_slope rows")
    x_normalized = np.stack([np.asarray(row["x"], dtype=np.float32) for row in rows])
    y_normalized = np.stack([np.asarray(row["y"], dtype=np.float32) for row in rows])
    phase = np.asarray([row["phase"] for row in rows], dtype=np.float32)
    clip = np.asarray([str(row["clip_id"]) for row in rows], dtype="<U128")
    lane = np.asarray([str(row["sequence_lane"]) for row in rows], dtype="<U16")
    terrain_class = np.asarray(
        [str(row["terrain_class"]) for row in rows], dtype="<U10"
    )
    center = np.asarray([int(row["center_frame"]) for row in rows], dtype=np.int64)
    terrain = np.asarray([str(row["terrain_sha256"]) for row in rows], dtype="<U64")
    x_mean = np.asarray(dataset.x_mean, dtype=np.float32)
    x_std = np.asarray(dataset.x_std, dtype=np.float32)
    y_mean = np.asarray(dataset.y_mean, dtype=np.float32)
    y_std = np.asarray(dataset.y_std, dtype=np.float32)
    if (
        x_normalized.shape[1:] != (INPUT_LAYOUT.size,)
        or y_normalized.shape[1:] != (OUTPUT_LAYOUT.size,)
        or not all(
            np.isfinite(value).all()
            for value in (x_normalized, y_normalized, phase)
        )
    ):
        raise ValueError("GRAIL normalized rows are invalid")
    return {
        "x_normalized": x_normalized,
        "y_normalized": y_normalized,
        "x": denormalize_pfnn_input(x_normalized, x_mean, x_std),
        "y": y_normalized * y_std + y_mean,
        "phase": phase,
        "clip_id": clip,
        "sequence_lane": lane,
        "terrain_class": terrain_class,
        "center_frame": center,
        "terrain_sha256": terrain,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _grail_split(values: Mapping[str, np.ndarray], mask: np.ndarray) -> VerticalSplitArrays:
    indices = np.flatnonzero(mask)
    normal = {name: values[name] for name in ("x_mean", "x_std", "y_mean", "y_std")}
    mirror_x, mirror_y, mirror_phase = _mirror_normalized_examples(
        values["x_normalized"][indices],
        values["y_normalized"][indices],
        values["phase"][indices],
        normal,
    )
    physical_mirror_x = mirror_x * values["x_std"] + values["x_mean"]
    physical_mirror_y = mirror_y * values["y_std"] + values["y_mean"]
    count = len(indices)
    return VerticalSplitArrays(
        x=np.concatenate((values["x"][indices], physical_mirror_x), axis=0).astype(np.float32),
        y=np.concatenate((values["y"][indices], physical_mirror_y), axis=0).astype(np.float32),
        phase=np.concatenate((values["phase"][indices], mirror_phase)).astype(np.float32),
        clip_id=np.concatenate(
            (
                values["clip_id"][indices],
                np.char.add(values["clip_id"][indices], "__mirror"),
            )
        ).astype("<U128"),
        sequence_lane=np.tile(values["sequence_lane"][indices], 2).astype("<U16"),
        center_frame_120hz=np.tile(values["center_frame"][indices] * 4, 2).astype(np.int64),
        root_world_xy=np.zeros((2 * count, 2), dtype=np.float32),
        root_world_yaw=np.zeros(2 * count, dtype=np.float32),
        terrain_class=np.tile(values["terrain_class"][indices], 2).astype("<U10"),
        terrain_sha256=np.tile(values["terrain_sha256"][indices], 2).astype("<U64"),
        mirrored=np.concatenate(
            (np.zeros(count, dtype=np.bool_), np.ones(count, dtype=np.bool_))
        ),
    )


def _concatenate(left: VerticalSplitArrays, right: VerticalSplitArrays) -> VerticalSplitArrays:
    return VerticalSplitArrays(
        **{
            name: np.concatenate((getattr(left, name), getattr(right, name)), axis=0)
            for name in (
                "x",
                "y",
                "phase",
                "clip_id",
                "sequence_lane",
                "center_frame_120hz",
                "root_world_xy",
                "root_world_yaw",
                "terrain_class",
                "terrain_sha256",
                "mirrored",
            )
        }
    )


def _filter_infeasible_joint_transitions(
    arrays: VerticalSplitArrays,
    *,
    maximum_joint_step_rad: float = 0.225,
) -> VerticalSplitArrays:
    """Drop rows whose target pose follows an unsafe same-lane target pose."""

    rows: dict[tuple[str, str, int], int] = {}
    keep = np.ones(len(arrays.phase), dtype=np.bool_)
    for index in range(len(arrays.phase)):
        key = (
            str(arrays.clip_id[index]),
            str(arrays.sequence_lane[index]),
            int(arrays.center_frame_120hz[index]),
        )
        if key in rows:
            raise ValueError("mixed PFNN corpus contains duplicate sequence rows")
        rows[key] = index
    joint = OUTPUT_LAYOUT["joint_position"]
    for (clip, lane, center), index in rows.items():
        predecessor = rows.get((clip, lane, center - 4))
        if predecessor is None:
            continue
        step = float(
            np.max(np.abs(arrays.y[index, joint] - arrays.y[predecessor, joint]))
        )
        if step > float(maximum_joint_step_rad) + 1.0e-6:
            keep[index] = False
    if not np.any(keep):
        raise ValueError("mixed PFNN transition filter removed every row")
    return VerticalSplitArrays(
        **{
            name: np.asarray(getattr(arrays, name))[keep]
            for name in (
                "x",
                "y",
                "phase",
                "clip_id",
                "sequence_lane",
                "center_frame_120hz",
                "root_world_xy",
                "root_world_yaw",
                "terrain_class",
                "terrain_sha256",
                "mirrored",
            )
        }
    )


def combine_vertical_and_grail(
    vertical: VerticalDataset,
    grail_train: object,
    *,
    grail_dataset_sha256: str,
) -> VerticalDataset:
    """Return a deterministic physical corpus with family-level GRAIL holdouts."""

    if not isinstance(vertical, VerticalDataset):
        raise TypeError("vertical must be a VerticalDataset")
    grail_digest = _digest(grail_dataset_sha256, "grail_dataset_sha256")
    values = _grail_physical_rows(grail_train)
    optimized, held_out = _grail_train_validation_masks(
        np.full(len(values["clip_id"]), "grail", dtype="<U6"),
        values["clip_id"],
    )
    splits = {
        "train": _filter_infeasible_joint_transitions(
            _concatenate(vertical.splits["train"], _grail_split(values, optimized))
        ),
        "validation": _filter_infeasible_joint_transitions(
            _concatenate(vertical.splits["validation"], _grail_split(values, held_out))
        ),
    }
    train = splits["train"]
    x_mean = np.mean(train.x, axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.std(train.x, axis=0, dtype=np.float64).astype(np.float32)
    y_mean = np.mean(train.y, axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.std(train.y, axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < np.float32(1.0e-6)] = np.float32(1.0)
    y_std[y_std < np.float32(1.0e-6)] = np.float32(1.0)
    contact = OUTPUT_LAYOUT["contact_logit"]
    y_mean[contact] = np.float32(0.0)
    y_std[contact] = np.float32(1.0)
    normalization = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    terrain_receipt = hashlib.sha256(
        json.dumps(
            {
                "grail_dataset_sha256": grail_digest,
                "vertical_terrain_receipt_set_sha256": vertical.terrain_receipt_set_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    roles = dict(vertical.source_roles)
    for split, arrays in splits.items():
        for clip in arrays.clip_id:
            name = str(clip).removesuffix("__mirror")
            if name.startswith("terrain_slopes__"):
                roles[name] = split
    dataset_sha = _dataset_digest(
        splits,
        normalization,
        selection_sha256=vertical.selection_sha256,
        retarget_manifest_sha256=vertical.retarget_manifest_sha256,
        terrain_receipt_set_sha256=terrain_receipt,
        source_roles=roles,
    )
    return VerticalDataset(
        splits=splits,
        **normalization,
        selection_sha256=vertical.selection_sha256,
        retarget_manifest_sha256=vertical.retarget_manifest_sha256,
        terrain_receipt_set_sha256=terrain_receipt,
        source_roles=roles,
        dataset_sha256=dataset_sha,
    )


def build_mixed_dataset(
    *, vertical_root: Path, grail_root: Path, output: Path
) -> VerticalDataset:
    vertical = load_vertical_dataset(Path(vertical_root))
    grail_path = Path(grail_root).expanduser().resolve(strict=True)
    manifest = json.loads((grail_path / "manifest.json").read_text(encoding="utf-8"))
    grail_digest = _digest(manifest.get("dataset_digest_sha256"), "grail dataset digest")
    mixed = combine_vertical_and_grail(
        vertical,
        PFNNShardDataset(grail_path, "train"),
        grail_dataset_sha256=grail_digest,
    )
    save_vertical_dataset(Path(output), mixed)
    return mixed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--grail-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dataset = build_mixed_dataset(
        vertical_root=arguments.vertical_root,
        grail_root=arguments.grail_root,
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "status": "accepted",
                "dataset_sha256": dataset.dataset_sha256,
                "train_rows": len(dataset.splits["train"].phase),
                "validation_rows": len(dataset.splits["validation"].phase),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = ["build_mixed_dataset", "combine_vertical_and_grail"]


if __name__ == "__main__":
    raise SystemExit(main())
