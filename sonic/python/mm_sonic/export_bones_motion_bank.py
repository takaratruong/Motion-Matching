"""Export a curated BONES zarr subset into the native motion-matcher layout.

The raw BONES G1 stores already contain clean 50 Hz kinematics in the same
29-joint / 30-body IsaacLab order used by the Takara matcher. This tool keeps
the clean reference motions separate from the short noisy BONES rollout
datasets: it exports one ``motion.npz`` per selected clip so motion matching
can compose long, joystick-driven trajectories while preserving the exact
joystick stream as the command label.

The exporter never mutates the source zarr and refuses to overwrite an
existing output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .joints import ContractError

DEFAULT_BONES_ZARR = Path("/move/data/bones/g1/zarr/locomotion_50hz.zarr")

# Justin's already-used walk/turn/arc/backward/sideways/start/stop support set,
# augmented with clean in-place turns. The source contains an explicit mirror
# for every entry; the exporter requires and includes it by default.
_BONES_CORE_BASE_CLIPS = (
    "walk_forward_loop_002__A023",
    "walk_forward_start_001__A022",
    "walk_forward_stop_001__A022",
    "walk_forward_start_001__A025",
    "turn_walk_360_002__A046",
    "turn_walk_360_003__A046",
    "turn_start_walk_090_001__A030",
    "turn_start_walk_045_001__A030",
    "walk_arc_cw_loop_003__A045",
    "arc_walk_left_loop_001__A030",
    "arc_walk_left_start_001__A029",
    "walk_arc_cw_start_001__A048",
    "walk_arc_cw_stop_001__A046",
    "walk_backward_stop_004__A022",
    "walk_backward_stop_001__A021",
    "walk_backward_start_002__A034",
    "walk_sideway_090_stop_001__A021",
    "walk_sideway_135_stop_001__A021",
    "walk_sideway_045_stop_001__A021",
    "walk_ff_stop_180_R_001__A046",
    "walk_forward_stop_002__A041",
    "walk_forward_normal_001__A005",
    "walk_forward_start_002__A032",
    "walk_forward_start_002__A037",
    "walk_forward_start_002__A040",
    "walk_backward_start_001__A029",
    "walk_backward_start_002__A041",
    "mohak_turn_090_001__A030",
    "idle_turn_270_001__A046",
    "idle_turn_360_001__A046",
)


def _transition_matrix(
    stem: str,
    variants: Sequence[int],
    *,
    subjects: Sequence[str] = ("A021", "A022"),
) -> tuple[str, ...]:
    return tuple(
        f"{stem}_{variant:03d}__{subject}"
        for variant in variants
        for subject in subjects
    )


# Starts and stops need enough density that nearest-neighbour search does not
# have to synthesize every change from loop frames. These cover multiple takes
# and actors for forward, backward, lateral, diagonal, and turn-then-walk
# transitions. Exact source mirrors are added below by the exporter.
_BONES_START_STOP_BASE_CLIPS = (
    _transition_matrix("walk_forward_start", (1, 2, 4))
    + _transition_matrix("walk_forward_stop", (1, 2, 3))
    + _transition_matrix("walk_backward_start", (1, 2, 3))
    + _transition_matrix("walk_backward_stop", (1, 2, 3))
    + _transition_matrix("walk_sideway_045_start", (1, 2))
    + _transition_matrix("walk_sideway_045_stop", (1, 2))
    + _transition_matrix("walk_sideway_090_start", (1, 2, 3))
    + _transition_matrix("walk_sideway_090_stop", (1, 2, 3))
    + _transition_matrix("walk_sideway_135_start", (1, 2))
    + _transition_matrix("walk_sideway_135_stop", (1, 2))
    + tuple(
        f"turn_start_walk_{heading:04d}_001__{subject}"
        for heading in (0, 45, 90, 135)
        for subject in ("A021", "A022")
    )
    + (
        "Turn_Start_Walk_0180_001__A017",
        "Turn_Start_Walk_0180_001__A018",
    )
)

DEFAULT_BONES_BASE_CLIPS = tuple(
    dict.fromkeys(_BONES_CORE_BASE_CLIPS + _BONES_START_STOP_BASE_CLIPS)
)

BONES_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

_FRAME_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_SAFE_CLIP_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_MIN_FRAMES = 46


def mirror_clip_name(name: str) -> str:
    """Return the explicit BONES mirror name for one unmirrored clip."""
    if name.endswith("_M"):
        raise ContractError(f"base clip must not already be mirrored: {name}")
    return f"{name}_M"


def resolve_clip_selection(
    available_names: Sequence[str],
    base_names: Sequence[str],
    *,
    include_mirrors: bool = True,
) -> tuple[str, ...]:
    """Resolve exact clips in deterministic base/mirror order."""
    available = tuple(str(name) for name in available_names)
    if len(set(available)) != len(available):
        raise ContractError("BONES clip_names must be unique")
    present = set(available)

    selected: list[str] = []
    seen: set[str] = set()
    for raw_name in base_names:
        name = str(raw_name)
        if not name or not _SAFE_CLIP_NAME.fullmatch(name):
            raise ContractError(f"unsafe or empty BONES clip name: {name!r}")
        if name in seen:
            raise ContractError(f"duplicate requested BONES clip: {name}")
        if name not in present:
            raise ContractError(f"requested BONES clip is missing: {name}")
        selected.append(name)
        seen.add(name)

        if include_mirrors:
            mirrored = mirror_clip_name(name)
            if mirrored not in present:
                raise ContractError(
                    f"requested BONES mirror is missing: {mirrored}"
                )
            selected.append(mirrored)
            seen.add(mirrored)
    return tuple(selected)


def _array(store: Mapping[str, object], name: str) -> object:
    try:
        return store[name]
    except Exception as error:
        raise ContractError(f"BONES zarr is missing array {name}") from error


def _validate_store(
    store: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fps = np.asarray(_array(store, "fps"))
    if fps.size != 1 or float(fps.reshape(-1)[0]) != 50.0:
        raise ContractError("BONES source fps must equal 50")

    names = np.asarray(_array(store, "clip_names"), dtype=str)
    starts = np.asarray(_array(store, "clip_start_idx"), dtype=np.int64)
    ends = np.asarray(_array(store, "clip_end_idx"), dtype=np.int64)
    if names.ndim != 1 or starts.shape != names.shape or ends.shape != names.shape:
        raise ContractError(
            "BONES clip_names/start/end arrays must be aligned rank-one arrays"
        )
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ContractError("BONES clip ranges must be positive exclusive intervals")

    body_names = tuple(str(name) for name in np.asarray(_array(store, "body_names")))
    if body_names != BONES_BODY_NAMES:
        raise ContractError("BONES body_names/order does not match the G1 matcher layout")

    expected_frames = int(ends.max(initial=0))
    trailing_shapes = {
        "joint_pos": (29,),
        "joint_vel": (29,),
        "body_pos_w": (30, 3),
        "body_quat_w": (30, 4),
        "body_lin_vel_w": (30, 3),
        "body_ang_vel_w": (30, 3),
    }
    for field, trailing in trailing_shapes.items():
        array = _array(store, field)
        shape = tuple(int(value) for value in array.shape)  # type: ignore[attr-defined]
        if len(shape) != len(trailing) + 1 or shape[1:] != trailing:
            raise ContractError(
                f"BONES {field} shape {shape} does not end in {trailing}"
            )
        if shape[0] < expected_frames:
            raise ContractError(
                f"BONES {field} has {shape[0]} rows, needs {expected_frames}"
            )
    return names, starts, ends


def export_motion_bank(
    store: Mapping[str, object],
    output_dir: str | Path,
    *,
    base_names: Sequence[str] = DEFAULT_BONES_BASE_CLIPS,
    include_mirrors: bool = True,
    source_label: str,
    legacy_takara: Mapping[str, object] | None = None,
    legacy_takara_label: str | None = None,
) -> dict[str, object]:
    """Export selected clips from an open zarr-like store.

    ``clip_end_idx`` is treated as exclusive, matching Justin's BONES loader.
    The temporary directory is renamed only after every clip and the manifest
    have been written successfully.
    """
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    names, starts, ends = _validate_store(store)
    selected = resolve_clip_selection(
        names.tolist(),
        base_names,
        include_mirrors=include_mirrors,
    )
    index_by_name = {str(name): index for index, name in enumerate(names)}

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    manifest_clips: list[dict[str, object]] = []
    try:
        for name in selected:
            index = index_by_name[name]
            start = int(starts[index])
            end = int(ends[index])
            frames = end - start
            if frames < _MIN_FRAMES:
                raise ContractError(
                    f"BONES clip {name} has {frames} frames, needs {_MIN_FRAMES}"
                )

            arrays: dict[str, np.ndarray] = {
                "fps": np.asarray([50], dtype=np.float32)
            }
            for field in _FRAME_FIELDS:
                values = np.asarray(_array(store, field)[start:end])  # type: ignore[index]
                values = np.ascontiguousarray(values, dtype=np.float32)
                if not np.all(np.isfinite(values)):
                    raise ContractError(f"BONES clip {name} field {field} is non-finite")
                arrays[field] = values

            quaternion_norm = np.linalg.norm(arrays["body_quat_w"], axis=-1)
            if not np.allclose(quaternion_norm, 1.0, atol=1e-4, rtol=0.0):
                raise ContractError(f"BONES clip {name} has non-unit WXYZ quaternions")

            clip_dir = temporary / name
            clip_dir.mkdir()
            np.savez(clip_dir / "motion.npz", **arrays)
            manifest_clips.append(
                {
                    "name": name,
                    "source": "bones",
                    "source_index": index,
                    "source_start": start,
                    "source_end_exclusive": end,
                    "frames": frames,
                    "mirrored": name.endswith("_M"),
                }
            )

        if legacy_takara is not None:
            required = ("fps",) + _FRAME_FIELDS
            missing = [field for field in required if field not in legacy_takara]
            if missing:
                raise ContractError(
                    f"legacy Takara NPZ is missing fields: {', '.join(missing)}"
                )
            fps = np.asarray(legacy_takara["fps"])
            if fps.size != 1 or float(fps.reshape(-1)[0]) != 50.0:
                raise ContractError("legacy Takara source fps must equal 50")

            takara_arrays: dict[str, np.ndarray] = {
                "fps": np.asarray([50], dtype=np.float32)
            }
            trailing_shapes = {
                "joint_pos": (29,),
                "joint_vel": (29,),
                "body_pos_w": (30, 3),
                "body_quat_w": (30, 4),
                "body_lin_vel_w": (30, 3),
                "body_ang_vel_w": (30, 3),
            }
            frames: int | None = None
            for field in _FRAME_FIELDS:
                values = np.ascontiguousarray(
                    np.asarray(legacy_takara[field]), dtype=np.float32
                )
                expected_trailing = trailing_shapes[field]
                if (
                    values.ndim != len(expected_trailing) + 1
                    or values.shape[1:] != expected_trailing
                ):
                    raise ContractError(
                        f"legacy Takara field {field} shape {values.shape} "
                        f"does not end in {expected_trailing}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ContractError(
                        f"legacy Takara field {field} is non-finite"
                    )
                if frames is None:
                    frames = int(values.shape[0])
                elif values.shape[0] != frames:
                    raise ContractError(
                        f"legacy Takara field {field} has inconsistent frames"
                    )
                takara_arrays[field] = values
            assert frames is not None
            if frames < _MIN_FRAMES:
                raise ContractError(
                    f"legacy Takara clip has {frames} frames, needs {_MIN_FRAMES}"
                )

            # Takara's historical body_quat_w field is actually XYZW. Native
            # matcher folders are strictly WXYZ.
            takara_arrays["body_quat_w"] = np.ascontiguousarray(
                takara_arrays["body_quat_w"][..., (3, 0, 1, 2)]
            )
            quaternion_norm = np.linalg.norm(
                takara_arrays["body_quat_w"], axis=-1
            )
            if not np.allclose(
                quaternion_norm, 1.0, atol=1e-4, rtol=0.0
            ):
                raise ContractError(
                    "legacy Takara clip has non-unit quaternions"
                )

            clip_dir = temporary / "takara_walk"
            clip_dir.mkdir()
            np.savez(clip_dir / "motion.npz", **takara_arrays)
            manifest_clips.append(
                {
                    "name": "takara_walk",
                    "source": "legacy_takara_npz",
                    "frames": frames,
                    "mirrored": False,
                }
            )

        manifest: dict[str, object] = {
            "schema": "mm-sonic-bones-motion-bank-v2",
            "source": source_label,
            "legacy_takara_source": legacy_takara_label,
            "fps": 50,
            "quaternion_convention": "wxyz",
            "clip_end_convention": "exclusive",
            "layout": "g1-29dof-isaaclab-v1",
            "base_clip_count": len(base_names),
            "include_mirrors": include_mirrors,
            "bones_clip_count": len(selected),
            "clip_count": len(manifest_clips),
            "clips": manifest_clips,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary)
        raise


def _load_clip_list(path: Path | None) -> Sequence[str]:
    if path is None:
        return DEFAULT_BONES_BASE_CLIPS
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ContractError("--clip-list must be a JSON list of strings")
    return tuple(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, default=DEFAULT_BONES_ZARR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clip-list",
        type=Path,
        default=None,
        help="optional JSON base-clip list; defaults to the curated support bank",
    )
    parser.add_argument(
        "--no-mirrors",
        action="store_true",
        help="export base clips only (not recommended for joystick coverage)",
    )
    parser.add_argument(
        "--include-legacy-takara-npz",
        type=Path,
        help=(
            "also include the legacy XYZW TakaraWalk NPZ, converted to native "
            "WXYZ; this builds an augmented rather than BONES-only search bank"
        ),
    )
    args = parser.parse_args()

    try:
        import zarr
    except ImportError as error:
        raise SystemExit(
            "zarr is required only for export; use Justin's env_isaaclab environment"
        ) from error

    source = args.source_zarr.expanduser().resolve(strict=True)
    store = zarr.open(str(source), mode="r")
    takara_path = None
    takara_values = None
    if args.include_legacy_takara_npz is not None:
        takara_path = args.include_legacy_takara_npz.expanduser().resolve(
            strict=True
        )
        with np.load(takara_path, allow_pickle=False) as takara:
            takara_values = {
                field: np.asarray(takara[field]) for field in takara.files
            }
    manifest = export_motion_bank(
        store,
        args.output_dir,
        base_names=_load_clip_list(args.clip_list),
        include_mirrors=not args.no_mirrors,
        source_label=str(source),
        legacy_takara=takara_values,
        legacy_takara_label=None if takara_path is None else str(takara_path),
    )
    print(
        f"exported {manifest['clip_count']} motion clips "
        f"({manifest['bones_clip_count']} BONES) to "
        f"{args.output_dir.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
