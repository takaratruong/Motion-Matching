"""Pure rigid placement and blending of three path-motion phases."""

from __future__ import annotations

import math

import numpy as np

from .joints import ContractError
from .torch_heading_footprint_realizer import blend_action_boundary


_CONNECTOR_KEYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
)


def _connector(value: object, label: str) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or set(value) != set(_CONNECTOR_KEYS):
        raise ContractError(f"path phase {label} connector is invalid")
    arrays = {
        name: np.asarray(value[name], dtype=np.float64)
        for name in _CONNECTOR_KEYS
    }
    frames = len(arrays["joint_position"])
    if (
        frames < 2
        or arrays["joint_position"].shape != (frames, 29)
        or arrays["root_position_world"].shape != (frames, 3)
        or arrays["root_orientation_world_wxyz"].shape != (frames, 4)
        or not all(np.isfinite(array).all() for array in arrays.values())
        or np.any(
            np.abs(
                np.linalg.norm(
                    arrays["root_orientation_world_wxyz"], axis=1
                )
                - 1.0
            )
            > 1.0e-4
        )
    ):
        raise ContractError(f"path phase {label} arrays are invalid")
    return {name: np.array(array, copy=True) for name, array in arrays.items()}


def _support(value: object, frames: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (frames, 2)
        or array.dtype != np.bool_
    ):
        raise ContractError(f"path phase {label} support is invalid")
    return np.array(array, copy=True)


def _joint_velocity(joints: np.ndarray) -> np.ndarray:
    return np.gradient(joints, axis=0)


def _select_overlap(
    mount: dict[str, np.ndarray],
    interior: dict[str, np.ndarray],
    *,
    blend_frames: int,
    maximum_root_gap_m: float,
) -> tuple[int, int, float]:
    mount_velocity = _joint_velocity(mount["joint_position"])
    interior_velocity = _joint_velocity(interior["joint_position"])
    candidates = []
    maximum_interior_frame = len(interior["joint_position"]) - blend_frames
    for mount_frame in range(len(mount["joint_position"])):
        for interior_frame in range(maximum_interior_frame + 1):
            root_gap = float(
                np.linalg.norm(
                    mount["root_position_world"][mount_frame]
                    - interior["root_position_world"][interior_frame]
                )
            )
            if root_gap > maximum_root_gap_m:
                continue
            joint_gap = float(
                np.linalg.norm(
                    mount["joint_position"][mount_frame]
                    - interior["joint_position"][interior_frame]
                )
            )
            velocity_gap = float(
                np.linalg.norm(
                    mount_velocity[mount_frame]
                    - interior_velocity[interior_frame]
                )
            )
            score = (
                root_gap
                + 0.04 * joint_gap
                + 0.02 * velocity_gap
                + 1.0e-6
                * (
                    len(mount["joint_position"])
                    - 1
                    - mount_frame
                    + interior_frame
                )
            )
            candidates.append(
                (score, root_gap, mount_frame, interior_frame)
            )
    if not candidates:
        raise ContractError("path phase mount/interior overlap is unavailable")
    _, root_gap, mount_frame, interior_frame = min(candidates)
    return int(mount_frame), int(interior_frame), float(root_gap)


def _blend(
    outgoing: dict[str, np.ndarray],
    *,
    previous: dict[str, np.ndarray],
    blend_frames: int,
) -> dict[str, np.ndarray]:
    joints, roots, quaternion = blend_action_boundary(
        joint_position=outgoing["joint_position"],
        root_position_world=outgoing["root_position_world"],
        root_orientation_world_wxyz=outgoing[
            "root_orientation_world_wxyz"
        ],
        previous_joint_position=previous["joint_position"][-1],
        previous_root_position_world=previous["root_position_world"][-1],
        previous_root_orientation_world_wxyz=previous[
            "root_orientation_world_wxyz"
        ][-1],
        blend_frames=blend_frames,
    )
    return {
        "joint_position": joints,
        "root_position_world": roots,
        "root_orientation_world_wxyz": quaternion,
    }


def compose_path_phases(
    *,
    mount: dict[str, np.ndarray],
    interior: dict[str, np.ndarray],
    dismount: dict[str, np.ndarray],
    mount_support: np.ndarray,
    interior_support: np.ndarray,
    dismount_support: np.ndarray,
    blend_frames: int = 10,
    maximum_root_gap_m: float = 0.12,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """Rigidly place, trim, and blend mount/interior/dismount phases."""

    if (
        type(blend_frames) is not int
        or blend_frames < 2
        or not math.isfinite(float(maximum_root_gap_m))
        or maximum_root_gap_m <= 0.0
    ):
        raise ContractError("path phase composite options are invalid")
    placed_mount = _connector(mount, "mount")
    placed_interior = _connector(interior, "interior")
    placed_dismount = _connector(dismount, "dismount")
    supports = (
        _support(
            mount_support,
            len(placed_mount["joint_position"]),
            "mount",
        ),
        _support(
            interior_support,
            len(placed_interior["joint_position"]),
            "interior",
        ),
        _support(
            dismount_support,
            len(placed_dismount["joint_position"]),
            "dismount",
        ),
    )
    if (
        blend_frames > len(placed_interior["joint_position"])
        or blend_frames > len(placed_dismount["joint_position"])
    ):
        raise ContractError("path phase blend exceeds a source phase")

    interior_shift = (
        placed_dismount["root_position_world"][0]
        - placed_interior["root_position_world"][-1]
    )
    placed_interior["root_position_world"] += interior_shift
    mount_frame, interior_frame, first_root_gap = _select_overlap(
        placed_mount,
        placed_interior,
        blend_frames=blend_frames,
        maximum_root_gap_m=float(maximum_root_gap_m),
    )

    mount_prefix = {
        name: array[: mount_frame + 1]
        for name, array in placed_mount.items()
    }
    interior_suffix = {
        name: array[interior_frame:]
        for name, array in placed_interior.items()
    }
    blended_interior = _blend(
        interior_suffix,
        previous=mount_prefix,
        blend_frames=blend_frames,
    )
    blended_dismount = _blend(
        placed_dismount,
        previous=blended_interior,
        blend_frames=blend_frames,
    )
    arrays = {
        name: np.ascontiguousarray(
            np.concatenate(
                (
                    mount_prefix[name],
                    blended_interior[name],
                    blended_dismount[name],
                ),
                axis=0,
            )
        )
        for name in _CONNECTOR_KEYS
    }
    support = np.ascontiguousarray(
        np.concatenate(
            (
                supports[0][: mount_frame + 1],
                supports[1][interior_frame:],
                supports[2],
            ),
            axis=0,
        )
    )
    first_boundary = len(mount_prefix["joint_position"])
    second_boundary = first_boundary + len(
        blended_interior["joint_position"]
    )
    metrics = {
        "schema": "g1-horizontal-phase-composite/v1",
        "frame_count": len(arrays["joint_position"]),
        "blend_frames": blend_frames,
        "mount_source_stop_frame": mount_frame + 1,
        "interior_source_start_frame": interior_frame,
        "interior_translation_matcher_xyz": interior_shift.tolist(),
        "mount_interior_root_gap_m": first_root_gap,
        "interior_dismount_root_gap_m": float(
            np.linalg.norm(
                placed_interior["root_position_world"][-1]
                - placed_dismount["root_position_world"][0]
            )
        ),
        "splice_output_frames": [first_boundary, second_boundary],
        "maximum_joint_step_rad": float(
            np.abs(np.diff(arrays["joint_position"], axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(
                np.diff(arrays["root_position_world"], axis=0), axis=1
            ).max()
        ),
    }
    return arrays, support, metrics
