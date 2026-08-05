"""Bounded realization of constant-heading footprint action plans."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .joints import ContractError
from .torch_flat_gait_contact_overlay import (
    anticipate_touchdown_positions,
    phase_contact_weights,
    source_relative_com_targets,
)
from .torch_heading_footprint_search import HeadingFootprintPlan


@dataclass(frozen=True)
class RealizedHeadingTraversal:
    joint_position: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    source_support_mask: np.ndarray
    source_frame_provenance: np.ndarray
    maximum_stance_error_m: float
    minimum_sole_clearance_m: float

    def __post_init__(self) -> None:
        joints = np.asarray(self.joint_position, dtype=np.float64)
        root = np.asarray(self.root_position_world, dtype=np.float64)
        quaternion = np.asarray(
            self.root_orientation_world_wxyz, dtype=np.float64
        )
        support = np.asarray(self.source_support_mask)
        provenance = np.asarray(self.source_frame_provenance)
        frames = len(joints)
        metrics = (
            self.maximum_stance_error_m,
            self.minimum_sole_clearance_m,
        )
        if (
            joints.shape != (frames, 29)
            or root.shape != (frames, 3)
            or quaternion.shape != (frames, 4)
            or support.shape != (frames, 2)
            or support.dtype != np.bool_
            or provenance.shape != (frames, 2)
            or provenance.dtype.kind not in "iu"
            or frames < 1
            or not bool(support.any(axis=1).all())
            or not all(
                np.isfinite(value).all()
                for value in (joints, root, quaternion)
            )
            or any(not math.isfinite(float(value)) for value in metrics)
        ):
            raise ContractError("realized heading traversal is invalid")
        object.__setattr__(
            self, "joint_position", np.ascontiguousarray(joints)
        )
        object.__setattr__(
            self, "root_position_world", np.ascontiguousarray(root)
        )
        object.__setattr__(
            self,
            "root_orientation_world_wxyz",
            np.ascontiguousarray(quaternion),
        )
        object.__setattr__(
            self, "source_support_mask", np.ascontiguousarray(support)
        )
        object.__setattr__(
            self,
            "source_frame_provenance",
            np.ascontiguousarray(provenance, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "maximum_stance_error_m",
            float(self.maximum_stance_error_m),
        )
        object.__setattr__(
            self,
            "minimum_sole_clearance_m",
            float(self.minimum_sole_clearance_m),
        )


class RealizationFailure(ContractError):
    def __init__(
        self,
        *,
        code: str,
        edge_index: int,
        source_action_key: tuple[int, int],
        reasons: tuple[str, ...],
    ) -> None:
        if (
            not isinstance(code, str)
            or not code
            or type(edge_index) is not int
            or edge_index < 0
            or not isinstance(source_action_key, tuple)
            or len(source_action_key) != 2
            or any(
                type(value) is not int or value < 0
                for value in source_action_key
            )
            or not isinstance(reasons, tuple)
            or not reasons
            or any(not isinstance(item, str) or not item for item in reasons)
        ):
            raise ContractError("heading realization failure is invalid")
        self.code = code
        self.edge_index = edge_index
        self.source_action_key = source_action_key
        self.reasons = reasons
        super().__init__(
            f"{code} at edge {edge_index} {source_action_key}: "
            f"{'; '.join(reasons)}"
        )


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _quaternion_multiply(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    return np.asarray(values, dtype=np.float64) @ rotation.T


def _source_inventory(source: object):
    try:
        clips = source.clips
        root_index = int(source.root_body_index)
        foot_indices = tuple(int(value) for value in source.foot_body_indices)
        action_index = source.action_index
        support_for_clip = source.support_mask
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError(
            "heading realization source is invalid"
        ) from error
    if (
        len(foot_indices) != 2
        or not callable(support_for_clip)
        or not hasattr(action_index, "entry")
    ):
        raise ContractError("heading realization source is invalid")
    return (
        clips,
        root_index,
        foot_indices,
        action_index,
        support_for_clip,
    )


def _surface_sampler(terrain: object):
    sampler = getattr(terrain, "sample_surface", None)
    if sampler is None and callable(terrain):
        sampler = terrain
    if not callable(sampler):
        raise ContractError("heading realization terrain is invalid")
    return sampler


def _scene_to_world_xy(terrain: object, points: np.ndarray) -> np.ndarray:
    transform = getattr(terrain, "scene_to_world_xy", None)
    if transform is None:
        return np.asarray(points, dtype=np.float64)
    output = np.asarray(transform(points), dtype=np.float64)
    if output.shape != np.asarray(points).shape or not np.isfinite(
        output
    ).all():
        raise ContractError("heading realization scene transform is invalid")
    return output


def _scene_heading_to_world(
    terrain: object, heading: np.ndarray
) -> np.ndarray:
    transform = getattr(terrain, "scene_heading_to_world", None)
    output = (
        np.asarray(heading, dtype=np.float64)
        if transform is None
        else np.asarray(transform(heading), dtype=np.float64)
    )
    if (
        output.shape != (2,)
        or not np.isfinite(output).all()
        or np.linalg.norm(output) <= 1.0e-6
    ):
        raise ContractError("heading realization heading transform is invalid")
    return output / np.linalg.norm(output)


def _support_anchors(
    *,
    support: np.ndarray,
    transformed_source_feet: np.ndarray,
    touchdown_targets: dict[tuple[int, int], np.ndarray],
    initial_anchor: np.ndarray,
) -> np.ndarray:
    anchors = np.full_like(transformed_source_feet, np.nan)
    for foot in range(2):
        active_start: int | None = None
        for frame in range(len(support)):
            if support[frame, foot] and active_start is None:
                active_start = frame
                if frame == 0:
                    anchor = initial_anchor[foot]
                else:
                    key = (frame, foot)
                    if key not in touchdown_targets:
                        raise ContractError(
                            "source touchdown has no planned footprint"
                        )
                    anchor = touchdown_targets[key]
            if support[frame, foot]:
                anchors[frame, foot] = anchor
            else:
                active_start = None
    return anchors


def realize_heading_footprint_plan(
    *,
    plan: HeadingFootprintPlan,
    source: object,
    terrain: object,
    kinematics: object,
    retargeter: object,
    ankle_origin_sole_m: float = 0.035,
    touchdown_blend_frames: int = 8,
    maximum_joint_step_rad: float = 0.30,
    maximum_root_step_m: float = 0.05,
    maximum_stance_error_m: float = 0.02,
) -> RealizedHeadingTraversal:
    """Rotate, contact-retarget, and concatenate selected source actions."""

    if (
        not isinstance(plan, HeadingFootprintPlan)
        or not callable(getattr(kinematics, "foot_positions", None))
        or not callable(
            getattr(kinematics, "center_of_mass_positions", None)
        )
        or not callable(getattr(retargeter, "solve_frame", None))
        or type(touchdown_blend_frames) is not int
        or touchdown_blend_frames < 1
    ):
        raise ContractError("heading realization input is invalid")
    bounds = (
        ankle_origin_sole_m,
        maximum_joint_step_rad,
        maximum_root_step_m,
        maximum_stance_error_m,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in bounds
    ):
        raise ContractError("heading realization bounds are invalid")
    sampler = _surface_sampler(terrain)
    (
        clips,
        root_body_index,
        foot_body_indices,
        action_index,
        support_for_clip,
    ) = _source_inventory(source)
    target_heading_world = _scene_heading_to_world(
        terrain, plan.heading_scene_xy.detach().cpu().numpy()
    )
    target_yaw = math.atan2(
        float(target_heading_world[1]),
        float(target_heading_world[0]),
    )

    all_joints: list[np.ndarray] = []
    all_roots: list[np.ndarray] = []
    all_quaternions: list[np.ndarray] = []
    all_support: list[np.ndarray] = []
    all_provenance: list[np.ndarray] = []
    previous_joint: np.ndarray | None = None
    previous_root: np.ndarray | None = None
    previous_support: np.ndarray | None = None
    previous_feet: np.ndarray | None = None
    maximum_stance = 0.0
    minimum_clearance = math.inf

    for edge_index, edge in enumerate(plan.edges):
        action = action_index.entry(*edge.action_key)
        if action is None:
            raise RealizationFailure(
                code="missing_source_action",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=("selected action is absent from the source index",),
            )
        try:
            clip = clips[action.clip_index]
            frames = np.arange(
                action.start_frame, action.end_frame, dtype=np.int64
            )
            joints = np.asarray(
                clip.joint_position[frames], dtype=np.float64
            ).copy()
            body_position = np.asarray(
                clip.body_position_world[frames], dtype=np.float64
            )
            root = body_position[:, root_body_index].copy()
            source_feet = body_position[:, foot_body_indices].copy()
            quaternion = np.asarray(
                clip.body_quaternion_world_wxyz[
                    frames, root_body_index
                ],
                dtype=np.float64,
            ).copy()
            raw_support = support_for_clip(action.clip_index)
            if isinstance(raw_support, torch.Tensor):
                raw_support = raw_support.detach().cpu().numpy()
            support = np.asarray(raw_support)[frames].astype(
                np.bool_, copy=True
            )
        except (IndexError, TypeError, ValueError, AttributeError) as error:
            raise RealizationFailure(
                code="invalid_source_window",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=("source arrays do not cover the selected action",),
            ) from error
        if (
            joints.shape != (len(frames), 29)
            or root.shape != (len(frames), 3)
            or source_feet.shape != (len(frames), 2, 3)
            or quaternion.shape != (len(frames), 4)
            or support.shape != (len(frames), 2)
            or len(frames) < 2
            or not bool(support.any(axis=1).all())
        ):
            raise RealizationFailure(
                code="invalid_source_window",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=("source support or kinematics are malformed",),
            )
        if not bool(support[-1].all()):
            raise RealizationFailure(
                code="terminal_mid_swing",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "selected source window does not end in complete support",
                ),
            )
        if previous_support is not None and not np.array_equal(
            support[0], previous_support
        ):
            raise RealizationFailure(
                code="support_phase_mismatch",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "source entry support differs from prior terminal support",
                ),
            )

        source_yaw = _yaw_from_wxyz(quaternion[0])
        yaw_delta = target_yaw - source_yaw
        root_start_xy = root[0, :2].copy()
        root[:, :2] = _rotate_xy(
            root[:, :2] - root_start_xy, yaw_delta
        )
        source_feet[:, :, :2] = _rotate_xy(
            source_feet[:, :, :2] - root_start_xy, yaw_delta
        )
        yaw_quaternion = np.array(
            (
                math.cos(yaw_delta / 2.0),
                0.0,
                0.0,
                math.sin(yaw_delta / 2.0),
            ),
            dtype=np.float64,
        )
        quaternion = _quaternion_multiply(
            np.broadcast_to(yaw_quaternion, quaternion.shape),
            quaternion,
        )

        if previous_feet is None:
            requested_root = getattr(
                source, "start_root_position_world", None
            )
            translation = (
                np.zeros(3, dtype=np.float64)
                if requested_root is None
                else np.asarray(requested_root, dtype=np.float64) - root[0]
            )
        else:
            translation = (
                previous_feet[support[0]]
                - source_feet[0, support[0]]
            ).mean(axis=0)
        root += translation
        source_feet += translation
        initial_anchor = (
            source_feet[0].copy()
            if previous_feet is None
            else previous_feet.copy()
        )

        onset = (~support[:-1]) & support[1:]
        events = tuple(
            (int(frame) + 1, int(foot))
            for frame, foot in np.argwhere(onset)
        )
        expected_events = tuple(
            (
                action.landing_frame_offsets[index],
                action.landing_feet[index],
            )
            for index in range(2)
        )
        if events[:2] != expected_events:
            raise RealizationFailure(
                code="source_contact_mismatch",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "authenticated support events differ from action metadata",
                ),
            )
        edge_footprints = plan.footprints[
            2 * edge_index : 2 * edge_index + 2
        ]
        touchdown_targets = {}
        for event, footprint in zip(expected_events, edge_footprints):
            target_world_xy = _scene_to_world_xy(
                terrain,
                footprint.center_scene_xy.detach().cpu().numpy(),
            )
            touchdown_targets[event] = np.array(
                (
                    float(target_world_xy[0]),
                    float(target_world_xy[1]),
                    footprint.surface_height_m
                    + float(ankle_origin_sole_m),
                ),
                dtype=np.float64,
            )
        try:
            stance_anchors = _support_anchors(
                support=support,
                transformed_source_feet=source_feet,
                touchdown_targets=touchdown_targets,
                initial_anchor=initial_anchor,
            )
            targets = anticipate_touchdown_positions(
                foot_position_target_world=source_feet,
                stance_anchor_world=stance_anchors,
                support_mask=support,
                blend_frames=touchdown_blend_frames,
            )
            targets[support] = stance_anchors[support]
            weights = 10.0 * phase_contact_weights(
                support_mask=support,
                minimum_swing_weight=0.1,
                blend_frames=touchdown_blend_frames,
            )
            source_com = kinematics.center_of_mass_positions(
                joints, root, quaternion
            )
            com_targets = source_relative_com_targets(
                source_center_of_mass_world=source_com,
                source_foot_position_world=source_feet,
                support_mask=support,
                stance_anchor_world=stance_anchors,
            )
        except ContractError as error:
            raise RealizationFailure(
                code="invalid_contact_targets",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(str(error),),
            ) from error

        solved_joints = []
        solved_roots = []
        window_feet = []
        for local_frame in range(len(frames)):
            kwargs = dict(
                joint_position=joints[local_frame],
                root_position_world=root[local_frame],
                root_orientation_world_wxyz=quaternion[local_frame],
                solve_feet=np.ones(2, dtype=np.bool_),
                enforce_target_error_feet=support[local_frame],
                target_foot_position_world=targets[local_frame],
                target_foot_position_weights=weights[local_frame],
                target_center_of_mass_world_xy=com_targets[local_frame],
            )
            if previous_joint is not None:
                kwargs["initial_joint_position"] = previous_joint
                kwargs["initial_root_position_world"] = previous_root
            try:
                solved_joint, solved_root = retargeter.solve_frame(
                    **kwargs
                )
            except (ContractError, ValueError, RuntimeError) as error:
                raise RealizationFailure(
                    code="retarget_unreachable",
                    edge_index=edge_index,
                    source_action_key=edge.action_key,
                    reasons=(str(error),),
                ) from error
            solved_joint = np.asarray(
                solved_joint, dtype=np.float64
            )
            solved_root = np.asarray(solved_root, dtype=np.float64)
            actual_feet = np.asarray(
                kinematics.foot_positions(
                    solved_joint[None],
                    solved_root[None],
                    quaternion[local_frame : local_frame + 1],
                )[0],
                dtype=np.float64,
            )
            if previous_joint is not None:
                joint_step = float(
                    np.max(np.abs(solved_joint - previous_joint))
                )
                root_step = float(
                    np.linalg.norm(solved_root - previous_root)
                )
                if (
                    joint_step > maximum_joint_step_rad
                    or root_step > maximum_root_step_m
                ):
                    raise RealizationFailure(
                        code="boundary_discontinuity",
                        edge_index=edge_index,
                        source_action_key=edge.action_key,
                        reasons=(
                            f"joint step {joint_step:.6f} rad, "
                            f"root step {root_step:.6f} m",
                        ),
                    )
            stance_error = np.linalg.norm(
                actual_feet[support[local_frame]]
                - stance_anchors[local_frame, support[local_frame]],
                axis=1,
            )
            maximum_stance = max(
                maximum_stance, float(stance_error.max())
            )
            surface_height = np.asarray(
                sampler(actual_feet[:, :2]), dtype=np.float64
            )
            if surface_height.shape != (2,) or not np.isfinite(
                surface_height
            ).all():
                raise ContractError(
                    "heading realization terrain samples are invalid"
                )
            clearance = (
                actual_feet[:, 2]
                - float(ankle_origin_sole_m)
                - surface_height
            )
            minimum_clearance = min(
                minimum_clearance, float(clearance.min())
            )
            solved_joints.append(solved_joint.copy())
            solved_roots.append(solved_root.copy())
            window_feet.append(actual_feet.copy())
            previous_joint = solved_joint
            previous_root = solved_root
        if maximum_stance > maximum_stance_error_m:
            raise RealizationFailure(
                code="stance_error",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    f"maximum stance error {maximum_stance:.6f} m",
                ),
            )

        keep = slice(None) if edge_index == 0 else slice(1, None)
        all_joints.extend(solved_joints[keep])
        all_roots.extend(solved_roots[keep])
        all_quaternions.extend(quaternion[keep])
        all_support.extend(support[keep])
        all_provenance.extend(
            np.stack(
                (
                    np.full(len(frames), action.clip_index),
                    frames,
                ),
                axis=1,
            )[keep]
        )
        previous_support = support[-1].copy()
        previous_feet = window_feet[-1].copy()

    return RealizedHeadingTraversal(
        joint_position=np.asarray(all_joints),
        root_position_world=np.asarray(all_roots),
        root_orientation_world_wxyz=np.asarray(all_quaternions),
        source_support_mask=np.asarray(all_support),
        source_frame_provenance=np.asarray(all_provenance),
        maximum_stance_error_m=maximum_stance,
        minimum_sole_clearance_m=minimum_clearance,
    )
