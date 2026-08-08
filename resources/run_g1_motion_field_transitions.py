#!/usr/bin/env python3
"""Build local source-motion turns between object-field traversal lines."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from mm_sonic.joints import ContractError
from mm_sonic.resample import shortest_path_slerp
from mm_sonic.torch_contact_oracle_actions import mirror_g1_joint_state

_ROUTE_ARRAYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
    "source_support_mask",
)


def _validated_route(value: object, *, minimum_frames: int = 2) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or any(name not in value for name in _ROUTE_ARRAYS):
        raise ContractError("motion-field transition route is incomplete")
    route = {name: np.asarray(value[name]) for name in _ROUTE_ARRAYS}
    frames = len(route["joint_position"])
    if (
        type(minimum_frames) is not int
        or minimum_frames < 2
        or frames < minimum_frames
        or route["joint_position"].shape != (frames, 29)
        or route["root_position_world"].shape != (frames, 3)
        or route["root_orientation_world_wxyz"].shape != (frames, 4)
        or route["source_support_mask"].shape != (frames, 2)
        or route["source_support_mask"].dtype != np.bool_
        or not all(np.isfinite(route[name]).all() for name in _ROUTE_ARRAYS[:-1])
        or np.any(
            np.abs(
                np.linalg.norm(route["root_orientation_world_wxyz"], axis=1)
                - 1.0
            )
            > 1.0e-4
        )
    ):
        raise ContractError("motion-field transition route is invalid")
    return route


def translate_route_xy(
    *, route: object, translation_xy_m: object
) -> dict[str, np.ndarray]:
    """Rigidly relocate an authentic route footprint without editing motion."""

    source = _validated_route(route)
    translation = np.asarray(translation_xy_m, dtype=np.float64)
    if (
        translation.shape != (2,)
        or not np.isfinite(translation).all()
        or np.linalg.norm(translation) > 0.05 + 1.0e-12
    ):
        raise ContractError("motion-field route translation is invalid")
    roots = np.asarray(source["root_position_world"], dtype=np.float64).copy()
    roots[:, :2] += translation
    return {
        "joint_position": np.ascontiguousarray(source["joint_position"]),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            source["root_orientation_world_wxyz"]
        ),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"]
        ),
    }


def translate_route_xy_through_first_support_transfer(
    *, route: object, translation_xy_m: object
) -> dict[str, np.ndarray]:
    """Relocate a footprint during flight, preserving both planted stances."""

    source = _validated_route(route, minimum_frames=5)
    translation = np.asarray(translation_xy_m, dtype=np.float64)
    if (
        translation.shape != (2,)
        or not np.isfinite(translation).all()
        or np.linalg.norm(translation) > 0.05 + 1.0e-12
        or int(source["source_support_mask"][0].sum()) != 1
    ):
        raise ContractError("motion-field support-transfer translation is invalid")
    support = source["source_support_mask"]
    initial_foot = int(np.flatnonzero(support[0])[0])
    opposite_foot = 1 - initial_foot
    initial_stop = 0
    while initial_stop + 1 < len(support) and bool(
        support[initial_stop + 1, initial_foot]
    ):
        initial_stop += 1
    opposite = np.flatnonzero(support[initial_stop + 1 :, opposite_foot])
    if len(opposite) == 0:
        raise ContractError("motion-field support-transfer translation has no landing")
    landing = initial_stop + 1 + int(opposite[0])
    if landing - initial_stop < 2:
        raise ContractError("motion-field support-transfer translation has no flight")
    weight = np.ones(len(support), dtype=np.float64)
    weight[: initial_stop + 1] = 0.0
    progress = np.linspace(0.0, 1.0, landing - initial_stop + 1)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    weight[initial_stop : landing + 1] = smooth
    roots = np.asarray(source["root_position_world"], dtype=np.float64).copy()
    roots[:, :2] += weight[:, None] * translation[None, :]
    return {
        "joint_position": np.ascontiguousarray(source["joint_position"]),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            source["root_orientation_world_wxyz"]
        ),
        "source_support_mask": np.ascontiguousarray(support),
    }


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _quaternion_wxyz_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ContractError("motion-field rigid rotation is invalid")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        output = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            output = np.asarray(((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale))
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            output = np.asarray(((rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale))
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            output = np.asarray(((rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale))
    output /= np.linalg.norm(output)
    return output


def _rigid_point_alignment(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[0] < 3
        or source.shape[1] != 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ContractError("motion-field rigid point alignment is invalid")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _endpoint(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.shape != shape or not np.isfinite(output).all():
        raise ContractError(f"motion-field transition {name} is invalid")
    return output


def aligned_terrain_clearance(
    *,
    sole_position_matcher: object,
    matcher_to_scene_xy,
    sample_scene_surface,
) -> np.ndarray:
    """Measure sole clearance after the authoritative matcher-to-scene map."""

    sole = np.asarray(sole_position_matcher, dtype=np.float64)
    if (
        sole.ndim != 4
        or sole.shape[1] != 2
        or sole.shape[2] < 3
        or sole.shape[3] != 3
        or not np.isfinite(sole).all()
        or not callable(matcher_to_scene_xy)
        or not callable(sample_scene_surface)
    ):
        raise ContractError("motion-field transition sole geometry is invalid")
    scene_xy = np.asarray(matcher_to_scene_xy(sole[..., :2]), dtype=np.float64)
    if scene_xy.shape != sole.shape[:-1] + (2,) or not np.isfinite(scene_xy).all():
        raise ContractError("motion-field transition terrain alignment is invalid")
    surface = np.asarray(sample_scene_surface(scene_xy), dtype=np.float64)
    if surface.shape != sole.shape[:-1] or not np.isfinite(surface).all():
        raise ContractError("motion-field transition terrain samples are invalid")
    return np.ascontiguousarray(sole[..., 2] - surface)


def minimum_root_xy_contact_correction_schedule(
    *,
    sole_position_matcher: object,
    support_mask: object,
    matcher_to_scene_xy,
    sample_scene_surface,
    search_radius_m: float = 0.02,
    search_step_m: float = 0.002,
) -> np.ndarray:
    """Find the smallest per-frame root translation that restores sole support.

    This is deliberately a rigid XY correction.  It preserves the generated
    articulated pose and is intended for quantization-scale terrain-boundary
    misses, not for manufacturing a missing step.
    """

    sole = np.asarray(sole_position_matcher, dtype=np.float64)
    support = np.asarray(support_mask)
    numeric = (search_radius_m, search_step_m)
    if (
        sole.ndim != 4
        or sole.shape[0] < 2
        or sole.shape[1] != 2
        or sole.shape[2] < 3
        or sole.shape[3] != 3
        or support.shape != sole.shape[:2]
        or support.dtype != np.bool_
        or not support.any()
        or not np.isfinite(sole).all()
        or not callable(matcher_to_scene_xy)
        or not callable(sample_scene_surface)
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in numeric
        )
        or float(search_step_m) > float(search_radius_m)
    ):
        raise ContractError("motion-field local contact search is invalid")

    count = math.floor(float(search_radius_m) / float(search_step_m))
    values = np.arange(-count, count + 1, dtype=np.float64) * float(
        search_step_m
    )
    offsets = [(float(x), float(y)) for x in values for y in values]
    offsets.sort(
        key=lambda xy: (
            xy[0] * xy[0] + xy[1] * xy[1],
            abs(xy[1]),
            abs(xy[0]),
            xy,
        )
    )
    correction = np.zeros((len(sole), 2), dtype=np.float64)
    for frame in range(len(sole)):
        selected = None
        for offset in offsets:
            shifted = sole[frame].copy()
            shifted[..., :2] += np.asarray(offset)
            scene_xy = np.asarray(
                matcher_to_scene_xy(shifted[..., :2]), dtype=np.float64
            )
            surface = np.asarray(sample_scene_surface(scene_xy), dtype=np.float64)
            if surface.shape != shifted.shape[:-1] or not np.isfinite(surface).all():
                raise ContractError("motion-field local terrain sample is invalid")
            clearance = shifted[..., 2] - surface
            points = ((clearance >= -0.025) & (clearance <= 0.035)).sum(axis=1)
            if clearance.min() >= -0.025 and bool(
                np.all(points[support[frame]] >= 3)
            ):
                selected = offset
                break
        if selected is None:
            raise ContractError(
                f"motion-field frame {frame} has no local contact correction"
            )
        correction[frame] = selected
    return np.ascontiguousarray(correction)


def apply_root_xy_correction_schedule(
    *, route: object, correction_xy_m: object
) -> dict[str, np.ndarray]:
    """Apply a bounded XY terrain correction without editing pose or heading."""

    source = _validated_route(route)
    correction = np.asarray(correction_xy_m, dtype=np.float64)
    if (
        correction.shape != (len(source["joint_position"]), 2)
        or not np.isfinite(correction).all()
        or np.linalg.norm(correction, axis=1).max() > 0.025 + 1.0e-12
        or np.linalg.norm(np.diff(correction, axis=0), axis=1).max()
        > 0.010 + 1.0e-12
        or np.linalg.norm(correction[0]) > 1.0e-12
        or np.linalg.norm(correction[-1]) > 1.0e-12
    ):
        raise ContractError("motion-field root correction schedule is invalid")
    roots = np.asarray(source["root_position_world"], dtype=np.float64).copy()
    roots[:, :2] += correction
    return {
        "joint_position": np.ascontiguousarray(source["joint_position"]),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            source["root_orientation_world_wxyz"]
        ),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"]
        ),
    }


def minimum_root_z_clearance_correction_schedule(
    *,
    sole_clearance_m: object,
    support_mask: object,
    maximum_lift_m: float = 0.02,
    search_step_m: float = 0.001,
) -> np.ndarray:
    """Find the least whole-body lift that clears each terrain collision."""

    clearance = np.asarray(sole_clearance_m, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        clearance.ndim != 3
        or clearance.shape[0] < 2
        or clearance.shape[1] != 2
        or clearance.shape[2] < 3
        or support.shape != clearance.shape[:2]
        or support.dtype != np.bool_
        or not support.any()
        or not np.isfinite(clearance).all()
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in (maximum_lift_m, search_step_m)
        )
        or float(search_step_m) > float(maximum_lift_m)
    ):
        raise ContractError("motion-field root clearance lift is invalid")
    count = math.floor(float(maximum_lift_m) / float(search_step_m))
    lifts = np.arange(count + 1, dtype=np.float64) * float(search_step_m)
    correction = np.zeros(len(clearance), dtype=np.float64)
    for frame in range(len(clearance)):
        selected = None
        for lift in lifts:
            shifted = clearance[frame] + lift
            points = ((shifted >= -0.025) & (shifted <= 0.035)).sum(axis=1)
            nearest = np.abs(shifted).min(axis=1)
            if (
                shifted.min() >= -0.025 - 1.0e-12
                and np.all(points[support[frame]] >= 3)
                and np.all(nearest[support[frame]] <= 0.035 + 1.0e-12)
            ):
                selected = float(lift)
                break
        if selected is None:
            raise ContractError(
                f"motion-field frame {frame} has no bounded clearance lift"
            )
        correction[frame] = selected
    return np.ascontiguousarray(correction)


def apply_root_z_correction_schedule(
    *, route: object, correction_z_m: object
) -> dict[str, np.ndarray]:
    """Apply a bounded vertical clearance lift without editing pose or XY."""

    source = _validated_route(route)
    correction = np.asarray(correction_z_m, dtype=np.float64)
    if (
        correction.shape != (len(source["joint_position"]),)
        or not np.isfinite(correction).all()
        or correction.min() < -1.0e-12
        or correction.max() > 0.020 + 1.0e-12
        or np.abs(np.diff(correction)).max() > 0.010 + 1.0e-12
        or abs(float(correction[0])) > 1.0e-12
        or abs(float(correction[-1])) > 1.0e-12
    ):
        raise ContractError("motion-field root height correction is invalid")
    roots = np.asarray(source["root_position_world"], dtype=np.float64).copy()
    roots[:, 2] += correction
    return {
        "joint_position": np.ascontiguousarray(source["joint_position"]),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            source["root_orientation_world_wxyz"]
        ),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"]
        ),
    }


def reflect_route_across_heading_axis(
    *, route: object, axis_y_m: float
) -> dict[str, np.ndarray]:
    """Reflect a target-ordered G1 route across a constant-heading XY axis."""

    return reflect_route_across_planar_axis(
        route=route,
        axis_origin_xy=(0.0, axis_y_m),
        axis_heading_rad=0.0,
    )


def reflect_route_across_planar_axis(
    *, route: object, axis_origin_xy: object, axis_heading_rad: float
) -> dict[str, np.ndarray]:
    """Reflect a G1 route across an arbitrary oriented planar symmetry axis."""

    source = _validated_route(route)
    origin = np.asarray(axis_origin_xy, dtype=np.float64)
    if (
        origin.shape != (2,)
        or not np.isfinite(origin).all()
        or isinstance(axis_heading_rad, bool)
        or not isinstance(axis_heading_rad, (int, float))
        or not math.isfinite(float(axis_heading_rad))
    ):
        raise ContractError("motion-field reflection axis is invalid")
    joints = (
        mirror_g1_joint_state(
            torch.tensor(source["joint_position"], dtype=torch.float64)
        )
        .cpu()
        .numpy()
    )
    roots = np.asarray(source["root_position_world"], dtype=np.float64).copy()
    heading = float(axis_heading_rad)
    cosine, sine = math.cos(heading), math.sin(heading)
    relative = roots[:, :2] - origin
    normal_projection = -sine * relative[:, 0] + cosine * relative[:, 1]
    roots[:, 0] = origin[0] + relative[:, 0] + 2.0 * sine * normal_projection
    roots[:, 1] = origin[1] + relative[:, 1] - 2.0 * cosine * normal_projection
    orientations = np.asarray(
        source["root_orientation_world_wxyz"], dtype=np.float64
    ).copy()
    inverse_axis = np.asarray(
        (math.cos(-0.5 * heading), 0.0, 0.0, math.sin(-0.5 * heading))
    )
    axis = np.asarray(
        (math.cos(0.5 * heading), 0.0, 0.0, math.sin(0.5 * heading))
    )
    orientations = _quaternion_multiply(inverse_axis, orientations)
    orientations[:, 1] *= -1.0
    orientations[:, 3] *= -1.0
    orientations = _quaternion_multiply(axis, orientations)
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    return {
        "joint_position": np.ascontiguousarray(joints),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(orientations),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"][:, ::-1]
        ),
    }


def stance_anchored_boundary_connector(
    *,
    incoming_joint_position: object,
    incoming_root_position_world: object,
    incoming_root_orientation_world_wxyz: object,
    target_joint_position: object,
    target_root_position_world: object,
    target_root_orientation_world_wxyz: object,
    stance_foot: int,
    frame_count: int,
    foot_kinematics: object,
    sole_kinematics: object | None = None,
) -> dict[str, np.ndarray]:
    """Interpolate exact same-stance boundaries while locking the stance foot."""

    incoming_joint = _endpoint(incoming_joint_position, (29,), "incoming joint")
    incoming_root = _endpoint(incoming_root_position_world, (3,), "incoming root")
    incoming_orientation = _endpoint(
        incoming_root_orientation_world_wxyz, (4,), "incoming orientation"
    )
    target_joint = _endpoint(target_joint_position, (29,), "target joint")
    target_root = _endpoint(target_root_position_world, (3,), "target root")
    target_orientation = _endpoint(
        target_root_orientation_world_wxyz, (4,), "target orientation"
    )
    if (
        stance_foot not in (0, 1)
        or type(frame_count) is not int
        or frame_count < 6
        or not callable(getattr(foot_kinematics, "foot_positions", None))
    ):
        raise ContractError("motion-field stance connector input is invalid")
    progress = np.linspace(0.0, 1.0, frame_count)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    joints = (
        (1.0 - smooth[:, None]) * incoming_joint[None]
        + smooth[:, None] * target_joint[None]
    )
    roots = (
        (1.0 - smooth[:, None]) * incoming_root[None]
        + smooth[:, None] * target_root[None]
    )
    orientations = np.stack(
        [
            shortest_path_slerp(
                incoming_orientation, target_orientation, float(weight)
            )
            for weight in smooth
        ]
    )
    feet = np.asarray(
        foot_kinematics.foot_positions(joints, roots, orientations),
        dtype=np.float64,
    )
    if feet.shape != (frame_count, 2, 3) or not np.isfinite(feet).all():
        raise ContractError("motion-field stance connector kinematics is invalid")
    if sole_kinematics is None:
        correction = feet[0, stance_foot] - feet[:, stance_foot]
        if (
            np.linalg.norm(correction[-1]) > 1.0e-4
            or np.linalg.norm(correction, axis=1).max() > 0.12
        ):
            raise ContractError("motion-field stance connector endpoints are incompatible")
        roots[1:-1] += correction[1:-1]
    else:
        if not callable(getattr(sole_kinematics, "sole_points", None)):
            raise ContractError("motion-field stance sole kinematics is invalid")
        soles = np.asarray(
            sole_kinematics.sole_points(joints, roots, orientations),
            dtype=np.float64,
        )
        if (
            soles.ndim != 4
            or soles.shape[:2] != (frame_count, 2)
            or soles.shape[2] < 3
            or soles.shape[3] != 3
            or not np.isfinite(soles).all()
        ):
            raise ContractError("motion-field stance sole geometry is invalid")
        anchor = soles[0, stance_foot]
        target_sole = soles[-1, stance_foot]
        endpoint_rotation, _ = _rigid_point_alignment(anchor, target_sole)
        endpoint_quaternion = _quaternion_wxyz_from_rotation_matrix(
            endpoint_rotation
        )
        anchor_center = anchor.mean(axis=0)
        target_center = target_sole.mean(axis=0)
        for frame in range(1, frame_count - 1):
            desired_rotation = _rotation_matrix_wxyz(
                shortest_path_slerp(
                    np.asarray((1.0, 0.0, 0.0, 0.0)),
                    endpoint_quaternion,
                    float(smooth[frame]),
                )
            )
            desired_center = (
                (1.0 - smooth[frame]) * anchor_center
                + smooth[frame] * target_center
            )
            desired = (
                desired_rotation @ (anchor - anchor_center).T
            ).T + desired_center
            correction_rotation, correction_translation = _rigid_point_alignment(
                soles[frame, stance_foot], desired
            )
            roots[frame] = (
                correction_rotation @ roots[frame] + correction_translation
            )
            correction_quaternion = _quaternion_wxyz_from_rotation_matrix(
                correction_rotation
            )
            orientations[frame] = _quaternion_multiply(
                correction_quaternion, orientations[frame]
            )
            orientations[frame] /= np.linalg.norm(orientations[frame])
    support = np.zeros((frame_count, 2), dtype=np.bool_)
    support[:, stance_foot] = True
    return {
        "joint_position": np.ascontiguousarray(joints),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(orientations),
        "source_support_mask": support,
    }


def stance_anchored_prefix_blend(
    *,
    route: object,
    stance_foot: int,
    blend_stop_frame: int,
    foot_kinematics: object,
) -> dict[str, np.ndarray]:
    """Rebuild a short connector prefix while rigidly locking its stance foot."""

    source = _validated_route(route, minimum_frames=6)
    if (
        stance_foot not in (0, 1)
        or type(blend_stop_frame) is not int
        or not 2 <= blend_stop_frame < len(source["joint_position"]) - 1
        or not callable(getattr(foot_kinematics, "foot_positions", None))
    ):
        raise ContractError("motion-field stance prefix input is invalid")
    joints = np.asarray(source["joint_position"], dtype=np.float64).copy()
    roots = np.asarray(source["root_position_world"], dtype=np.float64).copy()
    orientations = np.asarray(
        source["root_orientation_world_wxyz"], dtype=np.float64
    ).copy()
    for frame in range(1, blend_stop_frame):
        progress = float(frame) / float(blend_stop_frame)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        joints[frame] = (
            (1.0 - smooth) * joints[0] + smooth * joints[blend_stop_frame]
        )
        roots[frame] = (
            (1.0 - smooth) * roots[0] + smooth * roots[blend_stop_frame]
        )
        orientations[frame] = shortest_path_slerp(
            orientations[0], orientations[blend_stop_frame], smooth
        )
    feet = np.asarray(
        foot_kinematics.foot_positions(
            joints[: blend_stop_frame + 1],
            roots[: blend_stop_frame + 1],
            orientations[: blend_stop_frame + 1],
        ),
        dtype=np.float64,
    )
    if feet.shape != (blend_stop_frame + 1, 2, 3) or not np.isfinite(feet).all():
        raise ContractError("motion-field stance prefix kinematics is invalid")
    anchor = feet[0, stance_foot]
    correction = anchor - feet[:, stance_foot]
    if (
        # Terrain projection is allowed a millimetre-scale stance residual;
        # the route hard gate remains an order of magnitude looser (10 mm).
        np.linalg.norm(correction[-1]) > 0.003
        or np.linalg.norm(correction, axis=1).max() > 0.08
    ):
        raise ContractError("motion-field stance prefix endpoint is incompatible")
    roots[1:blend_stop_frame] += correction[1:blend_stop_frame]
    return {
        "joint_position": np.ascontiguousarray(joints),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(orientations),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"]
        ),
    }


def stance_anchored_terminal_blend(
    *,
    route: object,
    target_joint_position: object,
    target_root_position_world: object,
    target_root_orientation_world_wxyz: object,
    stance_foot: int,
    blend_start_frame: int,
    foot_kinematics: object,
) -> dict[str, np.ndarray]:
    """End exactly on a target pose while keeping the terminal stance planted."""

    source = _validated_route(route, minimum_frames=6)
    target_joint = _endpoint(target_joint_position, (29,), "target joint")
    target_root = _endpoint(target_root_position_world, (3,), "target root")
    target_orientation = _endpoint(
        target_root_orientation_world_wxyz, (4,), "target orientation"
    )
    if (
        type(blend_start_frame) is not int
        or not 1 <= blend_start_frame < len(source["joint_position"]) - 2
        or abs(float(np.linalg.norm(target_orientation)) - 1.0) > 1.0e-4
    ):
        raise ContractError("motion-field terminal blend input is invalid")
    endpoint = {name: np.asarray(value).copy() for name, value in source.items()}
    endpoint["joint_position"][-1] = target_joint
    endpoint["root_position_world"][-1] = target_root
    endpoint["root_orientation_world_wxyz"][-1] = target_orientation
    reversed_route = {
        name: np.ascontiguousarray(value[::-1]) for name, value in endpoint.items()
    }
    reversed_blend = stance_anchored_prefix_blend(
        route=reversed_route,
        stance_foot=stance_foot,
        blend_stop_frame=len(source["joint_position"]) - 1 - blend_start_frame,
        foot_kinematics=foot_kinematics,
    )
    return {
        name: np.ascontiguousarray(value[::-1])
        for name, value in reversed_blend.items()
    }


def place_authentic_transition(
    *,
    source_route: object,
    incoming_root_position_world: object,
    incoming_root_orientation_world_wxyz: object,
    outgoing_root_position_world: object,
) -> dict[str, np.ndarray]:
    """Place an authentic source turn without editing any articulated pose."""

    source = _validated_route(source_route, minimum_frames=5)
    incoming_root = _endpoint(
        incoming_root_position_world, (3,), "incoming root"
    )
    incoming_quaternion = _endpoint(
        incoming_root_orientation_world_wxyz, (4,), "incoming orientation"
    )
    outgoing_root = _endpoint(
        outgoing_root_position_world, (3,), "outgoing root"
    )
    if abs(float(np.linalg.norm(incoming_quaternion)) - 1.0) > 1.0e-4:
        raise ContractError("motion-field endpoint orientation is invalid")

    source_root = source["root_position_world"]
    source_quaternion = source["root_orientation_world_wxyz"]
    yaw_delta = _yaw_wxyz(incoming_quaternion) - _yaw_wxyz(source_quaternion[0])
    cosine, sine = math.cos(yaw_delta), math.sin(yaw_delta)
    relative = source_root - source_root[0]
    placed_root = np.empty_like(source_root, dtype=np.float64)
    placed_root[:, 0] = (
        incoming_root[0] + cosine * relative[:, 0] - sine * relative[:, 1]
    )
    placed_root[:, 1] = (
        incoming_root[1] + sine * relative[:, 0] + cosine * relative[:, 1]
    )
    placed_root[:, 2] = incoming_root[2] + relative[:, 2]
    progress = np.linspace(0.0, 1.0, len(placed_root))
    smooth = progress * progress * (3.0 - 2.0 * progress)
    placed_root += smooth[:, None] * (outgoing_root - placed_root[-1])[None, :]
    placed_root[0] = incoming_root
    placed_root[-1] = outgoing_root

    yaw_quaternion = np.array(
        (math.cos(0.5 * yaw_delta), 0.0, 0.0, math.sin(0.5 * yaw_delta)),
        dtype=np.float64,
    )
    placed_quaternion = _quaternion_multiply(yaw_quaternion, source_quaternion)
    placed_quaternion /= np.linalg.norm(placed_quaternion, axis=1, keepdims=True)
    return {
        "joint_position": np.ascontiguousarray(
            source["joint_position"], dtype=np.float64
        ),
        "root_position_world": np.ascontiguousarray(placed_root),
        "root_orientation_world_wxyz": np.ascontiguousarray(placed_quaternion),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"], dtype=np.bool_
        ),
    }


def align_authentic_transition_contact(
    *,
    placed_route: object,
    placed_start_foot_position_world: object,
    target_start_foot_position_world: object,
) -> dict[str, np.ndarray]:
    """Translate a placed source turn so its initial stance contact is exact."""

    placed = _validated_route(placed_route, minimum_frames=4)
    source_foot = _endpoint(
        placed_start_foot_position_world, (3,), "placed start foot"
    )
    target_foot = _endpoint(
        target_start_foot_position_world, (3,), "target start foot"
    )
    roots = np.asarray(placed["root_position_world"], dtype=np.float64).copy()
    roots += (target_foot - source_foot)[None, :]
    return {
        "joint_position": np.ascontiguousarray(
            placed["joint_position"], dtype=np.float64
        ),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            placed["root_orientation_world_wxyz"], dtype=np.float64
        ),
        "source_support_mask": np.ascontiguousarray(
            placed["source_support_mask"], dtype=np.bool_
        ),
    }


def align_authentic_transition_through_support_transfer(
    *,
    placed_route: object,
    placed_start_foot_position_world: object,
    target_start_foot_position_world: object,
) -> dict[str, np.ndarray]:
    """Hold an exact stance anchor, then release it during authentic flight."""

    placed = _validated_route(placed_route, minimum_frames=5)
    source_foot = _endpoint(
        placed_start_foot_position_world, (3,), "placed start foot"
    )
    target_foot = _endpoint(
        target_start_foot_position_world, (3,), "target start foot"
    )
    support = placed["source_support_mask"]
    if int(support[0].sum()) != 1:
        raise ContractError("motion-field transition initial support is invalid")
    initial_foot = int(np.flatnonzero(support[0])[0])
    opposite_foot = 1 - initial_foot
    initial_stop = 0
    while initial_stop + 1 < len(support) and bool(
        support[initial_stop + 1, initial_foot]
    ):
        initial_stop += 1
    opposite = np.flatnonzero(
        support[initial_stop + 1 :, opposite_foot]
    )
    if len(opposite) == 0:
        raise ContractError("motion-field transition has no opposite landing")
    landing = initial_stop + 1 + int(opposite[0])
    if landing - initial_stop < 2:
        raise ContractError("motion-field transition has no flight correction window")

    weight = np.zeros(len(support), dtype=np.float64)
    weight[: initial_stop + 1] = 1.0
    progress = np.linspace(0.0, 1.0, landing - initial_stop + 1)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    weight[initial_stop : landing + 1] = 1.0 - smooth
    roots = np.asarray(placed["root_position_world"], dtype=np.float64).copy()
    roots += weight[:, None] * (target_foot - source_foot)[None, :]
    return {
        "joint_position": np.ascontiguousarray(
            placed["joint_position"], dtype=np.float64
        ),
        "root_position_world": np.ascontiguousarray(roots),
        "root_orientation_world_wxyz": np.ascontiguousarray(
            placed["root_orientation_world_wxyz"], dtype=np.float64
        ),
        "source_support_mask": np.ascontiguousarray(support, dtype=np.bool_),
    }


def support_transfer_root_height_schedule(
    *,
    root_position_world: object,
    support_mask: object,
    target_root_height_world: float,
) -> np.ndarray:
    """Reach a new terrain-relative pelvis height at the opposite landing."""

    roots = np.asarray(root_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        roots.ndim != 2
        or roots.shape[1] != 3
        or len(roots) < 5
        or support.shape != (len(roots), 2)
        or support.dtype != np.bool_
        or not np.isfinite(roots).all()
        or isinstance(target_root_height_world, bool)
        or not isinstance(target_root_height_world, (int, float))
        or not math.isfinite(float(target_root_height_world))
        or int(support[0].sum()) != 1
    ):
        raise ContractError("motion-field root height schedule is invalid")
    initial_foot = int(np.flatnonzero(support[0])[0])
    opposite_foot = 1 - initial_foot
    landing_frames = np.flatnonzero(support[1:, opposite_foot])
    if len(landing_frames) == 0:
        raise ContractError("motion-field root height schedule has no landing")
    landing = int(landing_frames[0] + 1)
    if landing < 2:
        raise ContractError("motion-field root height schedule is too short")

    output = roots.copy()
    progress = np.linspace(0.0, 1.0, landing + 1)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    output[: landing + 1, 2] = (
        (1.0 - smooth) * roots[0, 2]
        + smooth * float(target_root_height_world)
    )
    output[landing:, 2] = float(target_root_height_world)
    return np.ascontiguousarray(output)


def return_support_root_height_schedule(
    *,
    root_position_world: object,
    support_mask: object,
    target_root_height_world: float,
) -> np.ndarray:
    """Hold the first landing level and change height on the return step."""

    roots = np.asarray(root_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        roots.ndim != 2
        or roots.shape[1] != 3
        or len(roots) < 8
        or support.shape != (len(roots), 2)
        or support.dtype != np.bool_
        or not np.isfinite(roots).all()
        or isinstance(target_root_height_world, bool)
        or not isinstance(target_root_height_world, (int, float))
        or not math.isfinite(float(target_root_height_world))
        or int(support[0].sum()) != 1
    ):
        raise ContractError("motion-field return height schedule is invalid")
    initial_foot = int(np.flatnonzero(support[0])[0])
    opposite_foot = 1 - initial_foot
    opposite_frames = np.flatnonzero(
        support[1:, opposite_foot] & ~support[1:, initial_foot]
    )
    if len(opposite_frames) == 0:
        raise ContractError("motion-field return height schedule has no opposite landing")
    opposite_landing = int(opposite_frames[0] + 1)
    return_frames = np.flatnonzero(
        support[opposite_landing + 1 :, initial_foot]
    )
    if len(return_frames) == 0:
        raise ContractError("motion-field return height schedule has no return landing")
    return_landing = int(opposite_landing + 1 + return_frames[0])
    if return_landing - opposite_landing < 2:
        raise ContractError("motion-field return height transfer is too short")

    output = roots.copy()
    output[: opposite_landing + 1, 2] = roots[0, 2]
    progress = np.linspace(
        0.0, 1.0, return_landing - opposite_landing + 1
    )
    smooth = progress * progress * (3.0 - 2.0 * progress)
    output[opposite_landing : return_landing + 1, 2] = (
        (1.0 - smooth) * roots[0, 2]
        + smooth * float(target_root_height_world)
    )
    output[return_landing:, 2] = float(target_root_height_world)
    return np.ascontiguousarray(output)


def landing_root_correction_schedule(
    *,
    support_mask: object,
    opposite_landing_correction_m: float,
) -> np.ndarray:
    """Interpolate a measured landing-height correction between exact ends."""

    support = np.asarray(support_mask)
    if (
        support.ndim != 2
        or support.shape[1] != 2
        or len(support) < 8
        or support.dtype != np.bool_
        or int(support[0].sum()) != 1
        or isinstance(opposite_landing_correction_m, bool)
        or not isinstance(opposite_landing_correction_m, (int, float))
        or not math.isfinite(float(opposite_landing_correction_m))
    ):
        raise ContractError("motion-field landing correction is invalid")
    initial_foot = int(np.flatnonzero(support[0])[0])
    opposite_foot = 1 - initial_foot
    opposite_onsets = np.flatnonzero(support[1:, opposite_foot])
    opposite_exclusive = np.flatnonzero(
        support[1:, opposite_foot] & ~support[1:, initial_foot]
    )
    if len(opposite_onsets) == 0 or len(opposite_exclusive) == 0:
        raise ContractError("motion-field landing correction has no opposite landing")
    opposite_landing = int(opposite_onsets[0] + 1)
    opposite_stop = int(opposite_exclusive[0] + 1)
    while opposite_stop + 1 < len(support) and bool(
        support[opposite_stop + 1, opposite_foot]
        and not support[opposite_stop + 1, initial_foot]
    ):
        opposite_stop += 1
    return_frames = np.flatnonzero(
        support[opposite_stop + 1 :, initial_foot]
        & ~support[opposite_stop + 1 :, opposite_foot]
    )
    if len(return_frames) == 0:
        raise ContractError("motion-field landing correction has no return landing")
    return_landing = int(opposite_stop + 1 + return_frames[0])
    if opposite_landing < 2 or return_landing - opposite_stop < 2:
        raise ContractError("motion-field landing correction windows are too short")

    correction = np.zeros(len(support), dtype=np.float64)
    rise = np.linspace(0.0, 1.0, opposite_landing + 1)
    rise = rise * rise * (3.0 - 2.0 * rise)
    correction[: opposite_landing + 1] = (
        rise * float(opposite_landing_correction_m)
    )
    correction[opposite_landing : opposite_stop + 1] = float(
        opposite_landing_correction_m
    )
    fall = np.linspace(0.0, 1.0, return_landing - opposite_stop + 1)
    fall = fall * fall * (3.0 - 2.0 * fall)
    correction[opposite_stop : return_landing + 1] = (
        (1.0 - fall) * float(opposite_landing_correction_m)
    )
    return np.ascontiguousarray(correction)


def endpoint_corrected_transition(
    *,
    source_route: object,
    incoming_joint_position: object,
    incoming_root_position_world: object,
    incoming_root_orientation_world_wxyz: object,
    outgoing_joint_position: object,
    outgoing_root_position_world: object,
    outgoing_root_orientation_world_wxyz: object,
    endpoint_blend_frames: int,
) -> dict[str, np.ndarray]:
    """Rigidly place one authentic turn and correct only its endpoint bands."""

    source = _validated_route(source_route, minimum_frames=5)
    frames = len(source["joint_position"])
    if (
        type(endpoint_blend_frames) is not int
        or endpoint_blend_frames < 1
        or 2 * endpoint_blend_frames + 1 >= frames
    ):
        raise ContractError("motion-field endpoint blend is invalid")
    incoming_joint = _endpoint(incoming_joint_position, (29,), "incoming joint")
    incoming_root = _endpoint(incoming_root_position_world, (3,), "incoming root")
    incoming_quaternion = _endpoint(
        incoming_root_orientation_world_wxyz, (4,), "incoming orientation"
    )
    outgoing_joint = _endpoint(outgoing_joint_position, (29,), "outgoing joint")
    outgoing_root = _endpoint(outgoing_root_position_world, (3,), "outgoing root")
    outgoing_quaternion = _endpoint(
        outgoing_root_orientation_world_wxyz, (4,), "outgoing orientation"
    )
    if any(
        abs(float(np.linalg.norm(value)) - 1.0) > 1.0e-4
        for value in (incoming_quaternion, outgoing_quaternion)
    ):
        raise ContractError("motion-field endpoint orientation is invalid")

    source_root = source["root_position_world"]
    source_quaternion = source["root_orientation_world_wxyz"]
    yaw_delta = _yaw_wxyz(incoming_quaternion) - _yaw_wxyz(source_quaternion[0])
    cosine, sine = math.cos(yaw_delta), math.sin(yaw_delta)
    relative = source_root - source_root[0]
    placed_root = np.empty_like(source_root, dtype=np.float64)
    placed_root[:, 0] = incoming_root[0] + cosine * relative[:, 0] - sine * relative[:, 1]
    placed_root[:, 1] = incoming_root[1] + sine * relative[:, 0] + cosine * relative[:, 1]
    placed_root[:, 2] = incoming_root[2] + relative[:, 2]
    endpoint_delta = outgoing_root - placed_root[-1]
    progress = np.linspace(0.0, 1.0, frames)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    placed_root += smooth[:, None] * endpoint_delta[None, :]
    placed_root[0] = incoming_root
    placed_root[-1] = outgoing_root

    yaw_quaternion = np.array(
        (math.cos(0.5 * yaw_delta), 0.0, 0.0, math.sin(0.5 * yaw_delta)),
        dtype=np.float64,
    )
    placed_quaternion = _quaternion_multiply(yaw_quaternion, source_quaternion)
    placed_quaternion /= np.linalg.norm(placed_quaternion, axis=1, keepdims=True)
    blend = endpoint_blend_frames
    for frame in range(blend + 1):
        placed_quaternion[frame] = shortest_path_slerp(
            incoming_quaternion,
            placed_quaternion[blend],
            frame / blend,
        )
    last_anchor = frames - blend - 1
    for local_frame in range(blend + 1):
        frame = last_anchor + local_frame
        placed_quaternion[frame] = shortest_path_slerp(
            placed_quaternion[last_anchor],
            outgoing_quaternion,
            local_frame / blend,
        )
    placed_quaternion[0] = incoming_quaternion
    placed_quaternion[-1] = outgoing_quaternion

    joints = np.asarray(source["joint_position"], dtype=np.float64).copy()
    source_start_anchor = joints[blend].copy()
    source_stop_anchor = joints[-blend - 1].copy()
    for frame in range(blend + 1):
        fraction = frame / blend
        joints[frame] = (1.0 - fraction) * incoming_joint + fraction * source_start_anchor
        reverse_frame = frames - blend - 1 + frame
        joints[reverse_frame] = (
            (1.0 - fraction) * source_stop_anchor + fraction * outgoing_joint
        )
    joints[0] = incoming_joint
    joints[-1] = outgoing_joint
    return {
        "joint_position": np.ascontiguousarray(joints),
        "root_position_world": np.ascontiguousarray(placed_root),
        "root_orientation_world_wxyz": np.ascontiguousarray(placed_quaternion),
        "source_support_mask": np.ascontiguousarray(
            source["source_support_mask"], dtype=np.bool_
        ),
    }


def assemble_transition_routes(
    *,
    incoming_route: object,
    outgoing_route: object,
    incoming_frame: int,
    outgoing_frame: int,
    connector: object,
) -> dict[str, np.ndarray]:
    """Splice one exact-endpoint connector without editing route interiors."""

    incoming = _validated_route(incoming_route)
    outgoing = _validated_route(outgoing_route)
    turn = _validated_route(connector)
    if (
        type(incoming_frame) is not int
        or type(outgoing_frame) is not int
        or not 1 <= incoming_frame < len(incoming["joint_position"]) - 1
        or not 1 <= outgoing_frame < len(outgoing["joint_position"]) - 1
        or not np.array_equal(
            incoming["source_support_mask"][incoming_frame],
            turn["source_support_mask"][0],
        )
        or not np.array_equal(
            outgoing["source_support_mask"][outgoing_frame],
            turn["source_support_mask"][-1],
        )
    ):
        raise ContractError("motion-field transition support boundary is invalid")
    boundaries = (
        (incoming, incoming_frame, turn, 0),
        (turn, -1, outgoing, outgoing_frame),
    )
    for first, first_frame, second, second_frame in boundaries:
        if (
            np.abs(
                first["joint_position"][first_frame]
                - second["joint_position"][second_frame]
            ).max()
            > 1.0e-8
            or np.linalg.norm(
                first["root_position_world"][first_frame]
                - second["root_position_world"][second_frame]
            )
            > 1.0e-8
            or abs(
                float(
                    np.dot(
                        first["root_orientation_world_wxyz"][first_frame],
                        second["root_orientation_world_wxyz"][second_frame],
                    )
                    / (
                        np.linalg.norm(
                            first["root_orientation_world_wxyz"][first_frame]
                        )
                        * np.linalg.norm(
                            second["root_orientation_world_wxyz"][second_frame]
                        )
                    )
                )
            )
            < 1.0 - 1.0e-8
        ):
            raise ContractError("motion-field transition endpoint is inexact")
    output = {}
    for name in _ROUTE_ARRAYS:
        output[name] = np.ascontiguousarray(
            np.concatenate(
                (
                    incoming[name][:incoming_frame],
                    turn[name],
                    outgoing[name][outgoing_frame + 1 :],
                ),
                axis=0,
            )
        )
    return output


def assemble_finite_radius_transition_edge(
    *,
    incoming_route: object,
    incoming_frame: int,
    connector: object,
    authentic_turn: object,
    preview_start_frame: int = 0,
    maximum_connector_turn_joint_step_rad: float = 0.25,
    maximum_connector_turn_root_step_m: float = 0.04,
    maximum_connector_turn_orientation_step_rad: float = 0.15,
) -> dict[str, np.ndarray]:
    """Assemble a turn that exits naturally onto a nearby outgoing lane.

    The generated connector owns the incoming endpoint.  Its terminal pose may
    differ slightly from the first immutable source-turn frame, but only within
    the same hard continuity bounds used to validate the complete edge.
    """

    incoming = _validated_route(incoming_route, minimum_frames=4)
    bridge = _validated_route(connector, minimum_frames=4)
    turn = _validated_route(authentic_turn, minimum_frames=5)
    numeric_limits = (
        maximum_connector_turn_joint_step_rad,
        maximum_connector_turn_root_step_m,
        maximum_connector_turn_orientation_step_rad,
    )
    if (
        type(incoming_frame) is not int
        or not 1 <= incoming_frame < len(incoming["joint_position"])
        or type(preview_start_frame) is not int
        or not 0 <= preview_start_frame < incoming_frame
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in numeric_limits
        )
        or not np.array_equal(
            incoming["source_support_mask"][incoming_frame],
            bridge["source_support_mask"][0],
        )
        or not np.array_equal(
            bridge["source_support_mask"][-1],
            turn["source_support_mask"][0],
        )
    ):
        raise ContractError("finite-radius transition boundary is invalid")

    incoming_joint_gap = float(
        np.abs(
            incoming["joint_position"][incoming_frame]
            - bridge["joint_position"][0]
        ).max()
    )
    incoming_root_gap = float(
        np.linalg.norm(
            incoming["root_position_world"][incoming_frame]
            - bridge["root_position_world"][0]
        )
    )
    incoming_quaternion = incoming["root_orientation_world_wxyz"][incoming_frame]
    bridge_start_quaternion = bridge["root_orientation_world_wxyz"][0]
    incoming_dot = abs(
        float(np.dot(incoming_quaternion, bridge_start_quaternion))
        / float(
            np.linalg.norm(incoming_quaternion)
            * np.linalg.norm(bridge_start_quaternion)
        )
    )
    if (
        incoming_joint_gap > 1.0e-6
        or incoming_root_gap > 1.0e-6
        or incoming_dot < 1.0 - 1.0e-8
    ):
        raise ContractError("finite-radius transition incoming endpoint is inexact")

    turn_joint_gap = float(
        np.abs(
            bridge["joint_position"][-1] - turn["joint_position"][0]
        ).max()
    )
    turn_root_gap = float(
        np.linalg.norm(
            bridge["root_position_world"][-1]
            - turn["root_position_world"][0]
        )
    )
    bridge_stop_quaternion = bridge["root_orientation_world_wxyz"][-1]
    turn_start_quaternion = turn["root_orientation_world_wxyz"][0]
    turn_dot = float(
        np.clip(
            abs(
                np.dot(bridge_stop_quaternion, turn_start_quaternion)
            ),
            0.0,
            1.0,
        )
    )
    turn_dot /= float(
        np.linalg.norm(bridge_stop_quaternion)
        * np.linalg.norm(turn_start_quaternion)
    )
    turn_dot = float(np.clip(turn_dot, 0.0, 1.0))
    turn_orientation_gap = 2.0 * math.acos(turn_dot)
    if (
        turn_joint_gap > float(maximum_connector_turn_joint_step_rad)
        or turn_root_gap > float(maximum_connector_turn_root_step_m)
        or turn_orientation_gap
        > float(maximum_connector_turn_orientation_step_rad)
    ):
        raise ContractError("finite-radius connector-to-turn join is discontinuous")

    output = {}
    for name in _ROUTE_ARRAYS:
        output[name] = np.ascontiguousarray(
            np.concatenate(
                (
                    incoming[name][preview_start_frame:incoming_frame],
                    bridge[name],
                    turn[name],
                ),
                axis=0,
            )
        )
    return output


def assemble_exact_successor_edge(
    *, incoming_route: object, connector: object, successor: object
) -> dict[str, np.ndarray]:
    """Append an exact connector and authentic successor without dwell frames."""

    incoming = _validated_route(incoming_route, minimum_frames=2)
    bridge = _validated_route(connector, minimum_frames=2)
    outgoing = _validated_route(successor, minimum_frames=2)
    for first, second in ((incoming, bridge), (bridge, outgoing)):
        if (
            not np.array_equal(
                first["source_support_mask"][-1],
                second["source_support_mask"][0],
            )
            or np.abs(
                first["joint_position"][-1] - second["joint_position"][0]
            ).max()
            > 1.0e-6
            or np.linalg.norm(
                first["root_position_world"][-1]
                - second["root_position_world"][0]
            )
            > 1.0e-6
            or abs(
                float(
                    np.dot(
                        first["root_orientation_world_wxyz"][-1],
                        second["root_orientation_world_wxyz"][0],
                    )
                    / (
                        np.linalg.norm(
                            first["root_orientation_world_wxyz"][-1]
                        )
                        * np.linalg.norm(
                            second["root_orientation_world_wxyz"][0]
                        )
                    )
                )
            )
            < 1.0 - 1.0e-8
        ):
            raise ContractError("motion-field exact successor boundary is invalid")
    return {
        name: np.ascontiguousarray(
            np.concatenate(
                (incoming[name], bridge[name][1:], outgoing[name][1:]), axis=0
            )
        )
        for name in _ROUTE_ARRAYS
    }


def transition_entry_candidate_rejections(
    metrics: object,
) -> tuple[str, ...]:
    """Reject authentic turns that are valid but infeasible to enter."""

    if not isinstance(metrics, Mapping):
        raise ContractError("motion-field transition entry metrics are invalid")
    try:
        root_translation = float(metrics["required_contact_root_translation_m"])
        pose_delta = float(metrics["starting_pose_max_joint_delta_rad"])
        support_compatible = metrics["support_compatible"]
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("motion-field transition entry metrics are invalid") from error
    if (
        not math.isfinite(root_translation)
        or root_translation < 0.0
        or not math.isfinite(pose_delta)
        or pose_delta < 0.0
        or type(support_compatible) is not bool
    ):
        raise ContractError("motion-field transition entry metrics are invalid")
    output = []
    if not support_compatible:
        output.append("entry_support_mismatch")
    if root_translation > 0.12:
        output.append("entry_root_relocation")
    if pose_delta > 0.85:
        output.append("entry_pose_mismatch")
    return tuple(output)


def terrain_transition_contact_metrics(
    *,
    sole_clearance_m: object,
    support_mask: object,
) -> dict[str, float | int]:
    """Measure terrain contact without anchoring a walk to its final pose."""

    clearance = np.asarray(sole_clearance_m, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        clearance.ndim != 3
        or clearance.shape[0] < 2
        or clearance.shape[1] != 2
        or clearance.shape[2] < 3
        or support.shape != clearance.shape[:2]
        or support.dtype != np.bool_
        or not support.any()
        or not np.isfinite(clearance).all()
    ):
        raise ContractError("motion-field terrain contact evidence is invalid")
    supported_points = (
        (clearance >= -0.025) & (clearance <= 0.035)
    ).sum(axis=2)
    nearest_surface = np.abs(clearance).min(axis=2)
    return {
        "minimum_sole_clearance_m": float(clearance.min()),
        "minimum_supported_sole_points": int(supported_points[support].min()),
        "maximum_stance_contact_error_m": float(
            nearest_surface[support].max()
        ),
    }


def transition_candidate_rejections(metrics: object) -> tuple[str, ...]:
    """Return deterministic hard-gate failures for one assembled transition."""

    if not isinstance(metrics, Mapping):
        raise ContractError("motion-field transition metrics are invalid")
    limits = (
        ("maximum_joint_step_rad", 0.25, "joint_discontinuity", "maximum"),
        ("maximum_root_step_m", 0.04, "root_discontinuity", "maximum"),
        ("minimum_sole_clearance_m", -0.025, "terrain_penetration", "minimum"),
        ("maximum_stance_contact_error_m", 0.035, "stance_contact_failure", "maximum"),
        ("maximum_stance_horizontal_step_m", 0.010, "stance_foot_slide", "maximum"),
        ("minimum_supported_sole_points", 3.0, "incomplete_sole_support", "minimum"),
        ("heading_change_error_degrees", 12.0, "heading_change_mismatch", "maximum"),
    )
    output = []
    for name, threshold, reason, mode in limits:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("motion-field transition metrics are invalid") from error
        if not math.isfinite(value):
            raise ContractError("motion-field transition metrics are invalid")
        if (mode == "maximum" and value > threshold) or (
            mode == "minimum" and value < threshold
        ):
            output.append(reason)
    return tuple(output)
