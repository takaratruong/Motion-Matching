"""Join stored joystick commands onto noisy SONIC rollout rows.

The emitted Task4 is expressed in the *physical rollout pelvis yaw frame*.
It uses no global root position or translation odometry: only the stored
world/controller command and the current pelvis orientation, matching the
runtime two-stick interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .offline_corpus import (
    CORPUS_SCHEMA,
    TASK4_SCHEMA,
    TASK12_KNOT_OFFSETS,
    TASK12_SCHEMA,
    _task4,
    _yaw_from_wxyz,
)


SIDECAR_SCHEMA = "takara-rollout-command-sidecar/v2"
COMMAND_KEYS = (
    "command_raw_sticks",
    "command_buttons",
    "command_requested_velocity_local_xy",
    "command_requested_velocity_world_xy",
    "command_requested_heading_world_yaw",
    "command_requested_path_world_xy",
    "command_shaped_velocity_world_xy",
    "command_shaped_heading_world_yaw",
    "command_ball_position_world_xy",
    "command_ball_velocity_world_xy",
    "command_ball_heading_world_yaw",
    "command_ball_heading_velocity_rad_s",
)
OPTIONAL_COMMAND_KEYS = ("command_event_mask",)


def _single_zarr(path: Path) -> Path:
    value = path.expanduser().resolve()
    if (value / ".zgroup").is_file():
        return value
    found = sorted(value.glob("*.zarr"))
    if len(found) != 1:
        raise ValueError(f"expected one rollout zarr in {value}, found {found}")
    return found[0]


def annotate(corpus: Path, rollout: Path, output: Path) -> None:
    import zarr

    source_path = corpus.expanduser().resolve()
    rollout_path = _single_zarr(rollout)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite command sidecar: {output}")
    source = zarr.open(str(source_path), mode="r")
    physical = zarr.open(str(rollout_path), mode="r")
    if source.attrs.get("schema") != CORPUS_SCHEMA:
        raise ValueError("source is not a Takara motion-matching corpus")

    source_ends = np.asarray(source["meta/episode_ends"], dtype=np.int64)
    source_starts = np.concatenate(([0], source_ends[:-1]))
    source_names = [
        str(value) for value in np.asarray(source["meta/episode_clip"])
    ]
    source_slices = {
        name: (int(start), int(stop))
        for name, start, stop in zip(
            source_names, source_starts, source_ends, strict=True
        )
    }
    source_seeds = (
        {
            name: int(seed)
            for name, seed in zip(
                source_names,
                np.asarray(source["meta/episode_command_seed"], dtype=np.int64),
                strict=True,
            )
        }
        if "episode_command_seed" in source["meta"]
        else {name: -1 for name in source_names}
    )
    episode_ends = np.asarray(physical["meta/episode_ends"], dtype=np.int64)
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    episode_names = [
        str(value) for value in np.asarray(physical["meta/episode_clip"])
    ]
    if len(episode_names) != len(episode_ends):
        raise ValueError("rollout episode metadata is inconsistent")
    if "episode_start_frame" in physical["meta"]:
        reference_starts = np.asarray(
            physical["meta/episode_start_frame"], dtype=np.int64
        )
    else:
        reference_starts = np.zeros(len(episode_names), dtype=np.int64)
    if reference_starts.shape != (len(episode_names),):
        raise ValueError("rollout episode_start_frame has wrong shape")

    source_rows = np.empty(int(episode_ends[-1]), dtype=np.int64)
    for name, row_start, row_stop, reference_start in zip(
        episode_names,
        episode_starts,
        episode_ends,
        reference_starts,
        strict=True,
    ):
        if name not in source_slices:
            raise ValueError(f"rollout clip {name!r} is absent from command corpus")
        source_start, source_stop = source_slices[name]
        length = int(row_stop - row_start)
        first = source_start + int(reference_start)
        last = first + length
        if first < source_start or last > source_stop:
            raise ValueError(
                f"{name}: rollout rows [{reference_start},{reference_start + length}) "
                f"exceed source length {source_stop - source_start}"
            )
        source_rows[int(row_start) : int(row_stop)] = np.arange(
            first, last, dtype=np.int64
        )

    body_rotation = np.asarray(physical["data/body_rot"], dtype=np.float32)
    if body_rotation.ndim == 2 and body_rotation.shape[1] == 120:
        root_quaternion = body_rotation.reshape(-1, 30, 4)[:, 0]
    elif body_rotation.ndim == 3 and body_rotation.shape[1:] == (30, 4):
        root_quaternion = body_rotation[:, 0]
    else:
        raise ValueError(f"unexpected rollout body_rot shape {body_rotation.shape}")
    root_yaw = _yaw_from_wxyz(root_quaternion)
    shaped_velocity = np.asarray(
        source["data/command_shaped_velocity_world_xy"], dtype=np.float32
    )[source_rows]
    shaped_heading = np.asarray(
        source["data/command_shaped_heading_world_yaw"], dtype=np.float32
    )[source_rows]
    task4 = _task4(shaped_velocity, shaped_heading, root_yaw)

    compressor = zarr.Blosc(
        cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE
    )
    sidecar = zarr.open_group(str(output), mode="w")
    data = sidecar.create_group("data")
    meta = sidecar.create_group("meta")
    emitted_command_keys = list(COMMAND_KEYS)
    emitted_command_keys.extend(
        key for key in OPTIONAL_COMMAND_KEYS if f"data/{key}" in source
    )
    for key in emitted_command_keys:
        values = np.asarray(source[f"data/{key}"])[source_rows]
        data.array(
            key,
            values,
            chunks=(min(4096, len(values)),) + values.shape[1:],
            compressor=compressor,
        )
    data.array(
        "task4_intended",
        task4,
        chunks=(min(4096, len(task4)), 4),
        compressor=compressor,
    )
    task12 = np.asarray(source["data/task12_intended"], dtype=np.float32)[
        source_rows
    ]
    data.array(
        "task12_intended",
        task12,
        chunks=(min(4096, len(task12)), len(TASK12_SCHEMA)),
        compressor=compressor,
    )
    data.array(
        "source_frame",
        source_rows,
        chunks=(min(4096, len(source_rows)),),
        compressor=compressor,
    )
    name_width = max(1, max(map(len, episode_names)))
    meta.array("episode_ends", episode_ends)
    meta.array(
        "episode_clip", np.asarray(episode_names, dtype=f"<U{name_width}")
    )
    meta.array("episode_start_frame", reference_starts)
    meta.array(
        "episode_command_seed",
        np.asarray([source_seeds[name] for name in episode_names], dtype=np.int64),
    )
    provenance = {
        "schema": SIDECAR_SCHEMA,
        "task4_schema": TASK4_SCHEMA,
        "task12_schema": TASK12_SCHEMA,
        "task12_knot_offsets": TASK12_KNOT_OFFSETS,
        "task12_localization_input": (
            "future filtered-ball heading at each knot; no robot position"
        ),
        "corpus": str(source_path),
        "rollout": str(rollout_path),
        "uses_global_root_position": False,
        "localization_input": "current physical pelvis WXYZ orientation only",
        "row_count": len(source_rows),
        "episode_count": len(episode_names),
        "command_keys": emitted_command_keys,
    }
    sidecar.attrs.update(
        {
            "schema": SIDECAR_SCHEMA,
            "provenance_json": json.dumps(provenance, sort_keys=True),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotate(args.corpus, args.rollout, args.output)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
