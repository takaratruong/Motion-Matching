"""Adapt retrieved motion fragments onto a privileged reference footfall path.

The reference path supplies terrain-compatible root poses and executable foot
placements.  Retrieved fragments still supply the authored joint motion and
timing.  A bounded leg-only IK solve reconciles their foot geometry, allowing
short fragments from different stair clips to be composed without assuming
that every source staircase has identical rises and treads.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .math3d import slerp_wxyz, unroll_quaternions_wxyz
from .stitch import (
    FragmentSelection,
    FrameProvenance,
    StitchedMotion,
)


_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_LEG_JOINT_NAMES = (
    (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
)


@dataclass(frozen=True)
class ReferenceFragmentSelection:
    """A retrieved source fragment paired with a terrain-valid reference span."""

    source: FragmentSelection
    reference: FragmentSelection


@dataclass(frozen=True)
class ReferenceStitchResult:
    """A reference-adapted motion and the mechanical cost of that adaptation."""

    motion: StitchedMotion
    source_switch_indices: tuple[int, ...]
    maximum_joint_correction_rad: float
    maximum_foot_target_error_m: float
    maximum_source_switch_joint_step_rad: float
    per_frame_joint_correction_rad: np.ndarray
    per_frame_foot_target_error_m: np.ndarray


@dataclass(frozen=True)
class AutomaticReferenceSelection:
    """The lowest-cost feasible cross-source substitution found in a bank."""

    result: ReferenceStitchResult
    selections: tuple[ReferenceFragmentSelection, ...]
    reference_fragment_id: str
    source_fragment_id: str
    reference_clip_index: int
    source_clip_index: int
    score: float
    evaluated_candidate_count: int
    rejected_candidate_count: int


@dataclass(frozen=True)
class _ArchivePoseClip:
    clip_id: str
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray


def _descendants(model: object, root_body: int) -> frozenset[int]:
    result: set[int] = set()
    for body_id in range(1, int(model.nbody)):
        cursor = body_id
        while cursor > 0:
            if cursor == root_body:
                result.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return frozenset(result)


def _interpolate(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    fraction = coordinates - lower
    return (
        np.asarray(values[lower], dtype=np.float64) * (1.0 - fraction[:, None])
        + np.asarray(values[upper], dtype=np.float64) * fraction[:, None]
    )


def _interpolate_quaternions(
    values: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    return np.asarray(
        slerp_wxyz(values[lower], values[upper], coordinates - lower),
        dtype=np.float64,
    )


def _reference_coordinates(
    selection: FragmentSelection,
    count: int,
    *,
    omit_entry: bool,
) -> np.ndarray:
    start = int(selection.source_start_frame) + int(omit_entry)
    stop = int(selection.source_stop_frame) - 1
    if count <= 0 or stop < start:
        raise ValueError("reference fragment is too short for emitted source frames")
    if count == 1:
        return np.asarray((float(stop),), dtype=np.float64)
    return np.linspace(float(start), float(stop), count, dtype=np.float64)


def _load_archive_clips(
    archive_path: Path,
    selections: Sequence[ReferenceFragmentSelection],
) -> tuple[dict[int, _ArchivePoseClip], tuple[str, ...], float]:
    import zarr

    archive = zarr.open_group(str(archive_path), mode="r")
    indices = {
        item.source.archive_clip_index
        for item in selections
    } | {
        item.reference.archive_clip_index
        for item in selections
    }
    clips: dict[int, _ArchivePoseClip] = {}
    for index in indices:
        start = int(archive["clip_start_idx"][index])
        stop = int(archive["clip_end_idx"][index])
        xyzw = np.asarray(
            archive["body_quat_w"][start:stop, 0], dtype=np.float64
        )
        clips[index] = _ArchivePoseClip(
            clip_id=str(archive["clip_names"][index]),
            root_position_world=np.asarray(
                archive["body_pos_w"][start:stop, 0], dtype=np.float64
            ),
            root_quaternion_world_wxyz=np.asarray(
                unroll_quaternions_wxyz(xyzw[..., (3, 0, 1, 2)]),
                dtype=np.float64,
            ),
            joint_position=np.asarray(
                archive["joint_pos"][start:stop], dtype=np.float64
            ),
        )
    return (
        clips,
        tuple(str(value) for value in archive["joint_names"][:]),
        float(archive["fps"][0]),
    )


class _G1FootfallAdapter:
    def __init__(
        self,
        model_path: Path,
        joint_names: Sequence[str],
        *,
        maximum_joint_correction_rad: float,
        target_tolerance_m: float = 2.5e-4,
        maximum_iterations: int = 32,
        damping: float = 0.012,
        posture_weight: float = 0.00005,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self._reference_data = mujoco.MjData(self.model)
        self._adapted_data = mujoco.MjData(self.model)
        self.maximum_joint_correction_rad = float(
            maximum_joint_correction_rad
        )
        self.target_tolerance_m = float(target_tolerance_m)
        self.maximum_iterations = int(maximum_iterations)
        self.damping = float(damping)
        self.posture_weight = float(posture_weight)

        free = np.flatnonzero(
            np.asarray(self.model.jnt_type)
            == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if free.size != 1:
            raise ValueError("G1 model must contain exactly one free joint")
        self._root_address = int(
            self.model.jnt_qposadr[int(free[0])]
        )

        joint_addresses: list[int] = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"G1 model is missing archive joint {name}")
            joint_addresses.append(
                int(self.model.jnt_qposadr[joint_id])
            )
        self._joint_addresses = np.asarray(
            joint_addresses, dtype=np.int32
        )

        sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        sphere_geoms: list[tuple[int, ...]] = []
        foot_body_ids: list[int] = []
        for body_name in _FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"G1 model is missing body {body_name}")
            foot_body_ids.append(int(body_id))
            descendants = _descendants(self.model, int(body_id))
            spheres = tuple(
                geom_id
                for geom_id in range(int(self.model.ngeom))
                if (
                    int(self.model.geom_bodyid[geom_id]) in descendants
                    and int(self.model.geom_type[geom_id]) == sphere_type
                )
            )
            if not spheres:
                raise ValueError(f"G1 model has no sole spheres for {body_name}")
            sphere_geoms.append(spheres)
        self._sphere_geoms = (sphere_geoms[0], sphere_geoms[1])
        self._foot_body_ids = (foot_body_ids[0], foot_body_ids[1])

        mujoco.mj_forward(self.model, self._reference_data)
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        envelope_points: list[np.ndarray] = []
        directions = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 0.0, 1.0)
                for y in (-1.0, 0.0, 1.0)
                for z in (-1.0, 0.0, 1.0)
                if (x, y, z) != (0.0, 0.0, 0.0)
            ],
            dtype=np.float64,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for body_id in self._foot_body_ids:
            descendants = _descendants(self.model, body_id)
            body_position = np.asarray(
                self._reference_data.xpos[body_id], dtype=np.float64
            )
            body_rotation = np.asarray(
                self._reference_data.xmat[body_id], dtype=np.float64
            ).reshape(3, 3)
            body_vertices = []
            for geom_id in range(int(self.model.ngeom)):
                if (
                    int(self.model.geom_bodyid[geom_id]) not in descendants
                    or int(self.model.geom_type[geom_id]) != mesh_type
                ):
                    continue
                mesh_id = int(self.model.geom_dataid[geom_id])
                start = int(self.model.mesh_vertadr[mesh_id])
                stop = start + int(self.model.mesh_vertnum[mesh_id])
                vertices = np.asarray(
                    self.model.mesh_vert[start:stop], dtype=np.float64
                )
                geom_position = np.asarray(
                    self._reference_data.geom_xpos[geom_id],
                    dtype=np.float64,
                )
                geom_rotation = np.asarray(
                    self._reference_data.geom_xmat[geom_id],
                    dtype=np.float64,
                ).reshape(3, 3)
                vertices_world = vertices @ geom_rotation.T + geom_position
                body_vertices.append(
                    (vertices_world - body_position) @ body_rotation
                )
            if not body_vertices:
                raise ValueError("G1 foot has no mesh collision envelope")
            vertices = np.concatenate(body_vertices, axis=0)
            selected = np.unique(
                np.argmax(vertices @ directions.T, axis=0)
            )
            envelope_points.append(vertices[selected])
        self._foot_collision_envelope_points_body = (
            envelope_points[0],
            envelope_points[1],
        )

        leg_qpos: list[np.ndarray] = []
        leg_dofs: list[np.ndarray] = []
        leg_limits: list[np.ndarray] = []
        for names in _LEG_JOINT_NAMES:
            ids = np.asarray(
                [
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                    for name in names
                ],
                dtype=np.int32,
            )
            if np.any(ids < 0):
                raise ValueError("G1 model is missing a leg joint")
            leg_qpos.append(
                np.asarray(self.model.jnt_qposadr[ids], dtype=np.int32)
            )
            leg_dofs.append(
                np.asarray(self.model.jnt_dofadr[ids], dtype=np.int32)
            )
            leg_limits.append(
                np.asarray(self.model.jnt_range[ids], dtype=np.float64)
            )
        self._leg_qpos = (leg_qpos[0], leg_qpos[1])
        self._leg_dofs = (leg_dofs[0], leg_dofs[1])
        self._leg_limits = (leg_limits[0], leg_limits[1])

    def _set_pose(
        self,
        data: object,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joints: np.ndarray,
    ) -> None:
        data.qpos[:] = self.model.qpos0
        root = self._root_address
        data.qpos[root : root + 3] = root_position
        data.qpos[root + 3 : root + 7] = (
            root_quaternion_wxyz
            / np.linalg.norm(root_quaternion_wxyz)
        )
        data.qpos[self._joint_addresses] = joints
        self._mujoco.mj_forward(self.model, data)

    def _sole_positions(
        self, data: object, foot: int
    ) -> np.ndarray:
        return np.asarray(
            [data.geom_xpos[index] for index in self._sphere_geoms[foot]],
            dtype=np.float64,
        )

    def sole_positions_for_pose(
        self,
        *,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact MuJoCo sole-sphere centres for an arbitrary pose."""

        self._set_pose(
            self._reference_data,
            root_position,
            root_quaternion_wxyz,
            joints,
        )
        return (
            self._sole_positions(self._reference_data, 0).copy(),
            self._sole_positions(self._reference_data, 1).copy(),
        )

    def sole_support_points_for_pose(
        self,
        *,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the bottoms of the exact sole collision spheres."""

        centres = self.sole_positions_for_pose(
            root_position=root_position,
            root_quaternion_wxyz=root_quaternion_wxyz,
            joints=joints,
        )
        result = []
        for foot in range(2):
            points = centres[foot].copy()
            points[:, 2] -= np.asarray(
                [
                    self.model.geom_size[index, 0]
                    for index in self._sphere_geoms[foot]
                ],
                dtype=np.float64,
            )
            result.append(points)
        return result[0], result[1]

    def sole_sphere_radii(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact collision-sphere radii in left/right foot order."""

        return tuple(
            np.asarray(
                [
                    self.model.geom_size[index, 0]
                    for index in self._sphere_geoms[foot]
                ],
                dtype=np.float64,
            )
            for foot in range(2)
        )  # type: ignore[return-value]

    def foot_collision_envelope_points_for_pose(
        self,
        *,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return compact extremal samples of each authored foot mesh hull."""

        self._set_pose(
            self._reference_data,
            root_position,
            root_quaternion_wxyz,
            joints,
        )
        output = []
        for body_id, points_body in zip(
            self._foot_body_ids,
            self._foot_collision_envelope_points_body,
            strict=True,
        ):
            rotation = np.asarray(
                self._reference_data.xmat[body_id], dtype=np.float64
            ).reshape(3, 3)
            position = np.asarray(
                self._reference_data.xpos[body_id], dtype=np.float64
            )
            output.append(points_body @ rotation.T + position)
        return output[0], output[1]

    def _solve_foot(
        self,
        foot: int,
        target: np.ndarray,
        reference_qpos: np.ndarray,
    ) -> None:
        data = self._adapted_data
        dofs = self._leg_dofs[foot]
        addresses = self._leg_qpos[foot]
        limits = self._leg_limits[foot]
        reference = reference_qpos[addresses]
        identity = np.eye(len(dofs), dtype=np.float64)
        jacobian_position = np.zeros(
            (3, int(self.model.nv)), dtype=np.float64
        )
        for _ in range(self.maximum_iterations):
            self._mujoco.mj_forward(self.model, data)
            current = self._sole_positions(data, foot)
            error = (target - current).reshape(-1)
            if float(
                np.max(np.linalg.norm(target - current, axis=1))
            ) <= self.target_tolerance_m:
                return
            rows: list[np.ndarray] = []
            for geom_id in self._sphere_geoms[foot]:
                point = np.asarray(
                    data.geom_xpos[geom_id], dtype=np.float64
                )
                body_id = int(self.model.geom_bodyid[geom_id])
                jacobian_position.fill(0.0)
                self._mujoco.mj_jac(
                    self.model,
                    data,
                    jacobian_position,
                    None,
                    point,
                    body_id,
                )
                rows.append(jacobian_position[:, dofs].copy())
            jacobian = np.vstack(rows)
            transpose = jacobian.T
            lhs = (
                transpose @ jacobian
                + (self.damping**2 + self.posture_weight) * identity
            )
            rhs = (
                transpose @ error
                + self.posture_weight
                * (reference - data.qpos[addresses])
            )
            delta = np.linalg.solve(lhs, rhs)
            delta = np.clip(delta, -0.04, 0.04)
            correction = np.clip(
                data.qpos[addresses] + delta - reference,
                -self.maximum_joint_correction_rad,
                self.maximum_joint_correction_rad,
            )
            data.qpos[addresses] = np.clip(
                reference + correction,
                limits[:, 0] + 1.0e-6,
                limits[:, 1] - 1.0e-6,
            )

    def adapt(
        self,
        *,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        authored_joints: np.ndarray,
        reference_joints: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        targets = self.sole_positions_for_pose(
            root_position=root_position,
            root_quaternion_wxyz=root_quaternion_wxyz,
            joints=reference_joints,
        )
        return self.adapt_to_targets(
            root_position=root_position,
            root_quaternion_wxyz=root_quaternion_wxyz,
            authored_joints=authored_joints,
            sole_targets_world=targets,
        )

    def adapt_to_targets(
        self,
        *,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        authored_joints: np.ndarray,
        sole_targets_world: Sequence[np.ndarray],
        initial_joints: np.ndarray | None = None,
        continuity_joints: np.ndarray | None = None,
        continuity_feet: Sequence[bool] = (False, False),
        maximum_continuity_joint_step_rad: float | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Fit authored leg joints to explicit world-space sole-sphere targets.

        ``initial_joints`` is an optional IK warm start.  The authored pose
        remains the posture regularizer, so a previous-frame warm start adds
        temporal continuity without turning the solve into pose copying.
        Only leg joints are taken from the warm start; the rest of the
        character stays exactly on the authored pose.

        ``continuity_joints`` optionally imposes a hard per-joint step bound
        on the selected feet after IK.  This is intended for unplanted legs,
        where a small target residual is preferable to switching IK branches
        near a singular configuration.
        """

        if len(sole_targets_world) != 2:
            raise ValueError("sole targets must contain left and right feet")
        targets = (
            np.asarray(sole_targets_world[0], dtype=np.float64),
            np.asarray(sole_targets_world[1], dtype=np.float64),
        )
        self._set_pose(
            self._adapted_data,
            root_position,
            root_quaternion_wxyz,
            authored_joints,
        )
        reference_qpos = self._adapted_data.qpos.copy()
        if initial_joints is not None:
            initial = np.asarray(initial_joints, dtype=np.float64)
            authored = np.asarray(authored_joints, dtype=np.float64)
            if initial.shape != authored.shape:
                raise ValueError(
                    "IK warm start must have the authored joint shape"
                )
            leg_addresses = np.concatenate(self._leg_qpos)
            leg_mask = np.isin(self._joint_addresses, leg_addresses)
            self._adapted_data.qpos[
                self._joint_addresses[leg_mask]
            ] = initial[leg_mask]
            self._mujoco.mj_forward(self.model, self._adapted_data)
        self._solve_foot(0, targets[0], reference_qpos)
        self._solve_foot(1, targets[1], reference_qpos)
        if maximum_continuity_joint_step_rad is not None:
            maximum_step = float(maximum_continuity_joint_step_rad)
            if not math.isfinite(maximum_step) or maximum_step <= 0.0:
                raise ValueError(
                    "maximum IK continuity joint step must be positive"
                )
            if continuity_joints is None:
                raise ValueError(
                    "IK continuity joints are required with a step bound"
                )
            continuity = np.asarray(continuity_joints, dtype=np.float64)
            authored = np.asarray(authored_joints, dtype=np.float64)
            if continuity.shape != authored.shape:
                raise ValueError(
                    "IK continuity joints must have the authored joint shape"
                )
            selected_feet = tuple(bool(value) for value in continuity_feet)
            if len(selected_feet) != 2:
                raise ValueError("IK continuity feet must contain two flags")
            continuity_qpos = reference_qpos.copy()
            continuity_qpos[self._joint_addresses] = continuity
            for foot, selected in enumerate(selected_feet):
                if not selected:
                    continue
                addresses = self._leg_qpos[foot]
                lower = continuity_qpos[addresses] - maximum_step
                upper = continuity_qpos[addresses] + maximum_step
                self._adapted_data.qpos[addresses] = np.clip(
                    self._adapted_data.qpos[addresses], lower, upper
                )
        self._mujoco.mj_forward(self.model, self._adapted_data)

        result = np.asarray(
            self._adapted_data.qpos[self._joint_addresses],
            dtype=np.float64,
        ).copy()
        correction = float(np.max(np.abs(result - authored_joints)))
        error = max(
            float(
                np.max(
                    np.linalg.norm(
                        targets[foot]
                        - self._sole_positions(self._adapted_data, foot),
                        axis=1,
                    )
                )
            )
            for foot in range(2)
        )
        return result, correction, error


def stitch_archive_to_reference(
    archive_path: str | Path,
    selections: Sequence[ReferenceFragmentSelection],
    *,
    model_path: str | Path,
    decay_frames: float = 8.0,
    maximum_joint_correction_rad: float = 0.35,
    maximum_foot_target_error_m: float = 0.001,
    maximum_source_switch_joint_step_rad: float = 0.20,
) -> ReferenceStitchResult:
    """Fit retrieved fragments to an ordered terrain-valid reference route."""

    selection_tuple = tuple(selections)
    if not selection_tuple:
        raise ValueError("reference stitch needs at least one fragment")
    if (
        not math.isfinite(float(decay_frames))
        or float(decay_frames) <= 0.0
        or not math.isfinite(float(maximum_joint_correction_rad))
        or float(maximum_joint_correction_rad) <= 0.0
        or not math.isfinite(float(maximum_foot_target_error_m))
        or float(maximum_foot_target_error_m) <= 0.0
        or not math.isfinite(
            float(maximum_source_switch_joint_step_rad)
        )
        or float(maximum_source_switch_joint_step_rad) <= 0.0
    ):
        raise ValueError("reference stitch bounds must be positive and finite")

    clips, joint_names, fps = _load_archive_clips(
        Path(archive_path), selection_tuple
    )
    adapter = _G1FootfallAdapter(
        Path(model_path),
        joint_names,
        maximum_joint_correction_rad=maximum_joint_correction_rad,
    )

    root_positions: list[np.ndarray] = []
    root_quaternions: list[np.ndarray] = []
    joint_positions: list[np.ndarray] = []
    joint_corrections: list[np.ndarray] = []
    foot_target_errors: list[np.ndarray] = []
    provenance: list[FrameProvenance] = []
    seam_indices: list[int] = []

    for selection_index, item in enumerate(selection_tuple):
        source = item.source
        reference = item.reference
        source_clip = clips[source.archive_clip_index]
        reference_clip = clips[reference.archive_clip_index]
        omit_entry = selection_index > 0
        source_start = int(source.source_start_frame) + int(omit_entry)
        source_stop = int(source.source_stop_frame)
        source_frames = np.arange(source_start, source_stop)
        if not len(source_frames):
            raise ValueError("source fragment is too short")
        if (
            int(source.source_start_frame) < 0
            or source_stop > len(source_clip.joint_position)
            or int(reference.source_start_frame) < 0
            or int(reference.source_stop_frame)
            > len(reference_clip.joint_position)
        ):
            raise ValueError("fragment selection exceeds its archive clip")

        coordinates = _reference_coordinates(
            reference, len(source_frames), omit_entry=omit_entry
        )
        reference_root = _interpolate(
            reference_clip.root_position_world, coordinates
        )
        reference_quaternion = _interpolate_quaternions(
            reference_clip.root_quaternion_world_wxyz, coordinates
        )
        reference_joints = _interpolate(
            reference_clip.joint_position, coordinates
        )
        authored_joints = np.asarray(
            source_clip.joint_position[source_frames],
            dtype=np.float64,
        ).copy()

        if omit_entry:
            entry = int(source.source_start_frame)
            offsets = source_frames - entry
            decay = np.exp(
                -np.asarray(offsets, dtype=np.float64)
                / float(decay_frames)
            )
            joint_offset = (
                joint_positions[-1][-1]
                - source_clip.joint_position[entry]
            )
            authored_joints += decay[:, None] * joint_offset
            seam_indices.append(sum(len(value) for value in root_positions))

        adapted_joints = np.empty_like(authored_joints)
        corrections = np.empty(len(authored_joints), dtype=np.float64)
        errors = np.empty(len(authored_joints), dtype=np.float64)
        for frame in range(len(authored_joints)):
            (
                adapted_joints[frame],
                corrections[frame],
                errors[frame],
            ) = adapter.adapt(
                root_position=reference_root[frame],
                root_quaternion_wxyz=reference_quaternion[frame],
                authored_joints=authored_joints[frame],
                reference_joints=reference_joints[frame],
            )

        root_positions.append(reference_root)
        root_quaternions.append(reference_quaternion)
        joint_positions.append(adapted_joints)
        joint_corrections.append(corrections)
        foot_target_errors.append(errors)
        provenance.extend(
            FrameProvenance(
                source.archive_clip_index,
                int(frame),
                source_clip.clip_id,
            )
            for frame in source_frames
        )

    roots = np.concatenate(root_positions)
    quaternions = np.concatenate(root_quaternions)
    adapted = np.concatenate(joint_positions)
    corrections = np.concatenate(joint_corrections)
    errors = np.concatenate(foot_target_errors)

    maximum_correction = float(np.max(corrections))
    maximum_error = float(np.max(errors))
    if maximum_correction > float(maximum_joint_correction_rad) + 1.0e-6:
        raise ValueError(
            "reference footfall fit exceeds joint-correction bound: "
            f"{maximum_correction:.6f} rad"
        )
    if maximum_error > float(maximum_foot_target_error_m):
        raise ValueError(
            "reference footfall fit misses target: "
            f"{maximum_error:.6f} m"
        )

    source_switches = tuple(
        index
        for index in range(1, len(provenance))
        if (
            provenance[index - 1].archive_clip_index
            != provenance[index].archive_clip_index
        )
    )
    maximum_source_switch_step = max(
        (
            float(
                np.max(
                    np.abs(
                        adapted[index] - adapted[index - 1]
                    )
                )
            )
            for index in source_switches
        ),
        default=0.0,
    )
    if maximum_source_switch_step > float(
        maximum_source_switch_joint_step_rad
    ):
        raise ValueError(
            "reference stitch exceeds source-switch joint-step bound: "
            f"{maximum_source_switch_step:.6f} rad"
        )
    motion = StitchedMotion(
        fps=fps,
        root_position_world=np.asarray(roots, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            quaternions, dtype=np.float32
        ),
        joint_position=np.asarray(adapted, dtype=np.float32),
        provenance=tuple(provenance),
        seam_indices=tuple(seam_indices),
    )
    return ReferenceStitchResult(
        motion=motion,
        source_switch_indices=source_switches,
        maximum_joint_correction_rad=maximum_correction,
        maximum_foot_target_error_m=maximum_error,
        maximum_source_switch_joint_step_rad=(
            maximum_source_switch_step
        ),
        per_frame_joint_correction_rad=np.asarray(
            corrections, dtype=np.float32
        ),
        per_frame_foot_target_error_m=np.asarray(
            errors, dtype=np.float32
        ),
    )


def _fragment_selection(record: dict[str, object]) -> FragmentSelection:
    source_range = record["source_range"]
    if not isinstance(source_range, list) or len(source_range) != 2:
        raise ValueError("fragment record has no two-frame source_range")
    return FragmentSelection(
        int(record["archive_clip_index"]),
        int(source_range[0]),
        int(source_range[1]),
    )


def _cheap_substitution_cost(
    source: dict[str, object], reference: dict[str, object]
) -> float:
    duration_scale = max(float(reference["duration_s"]), 0.25)
    rise_scale = max(abs(float(reference["source_rise_m"])), 0.08)
    tread_scale = max(abs(float(reference["source_tread_m"])), 0.12)
    return (
        abs(float(source["duration_s"]) - float(reference["duration_s"]))
        / duration_scale
        + abs(
            float(source["vertical_delta_m"])
            - float(reference["vertical_delta_m"])
        )
        / rise_scale
        + abs(
            float(source["approximate_run_m"])
            - float(reference["approximate_run_m"])
        )
        / tread_scale
    )


def _compatible_substitution(
    source: dict[str, object], reference: dict[str, object]
) -> bool:
    return bool(
        source["archive_clip_index"] != reference["archive_clip_index"]
        and source["role"] == "middle"
        and reference["role"] == "middle"
        and source["source_direction"] == reference["source_direction"]
        and source["entry_phase"] == reference["entry_phase"]
        and source["exit_phase"] == reference["exit_phase"]
        and int(source["riser_transition_count"])
        == int(reference["riser_transition_count"])
    )


def _source_switch_step(motion: StitchedMotion) -> float:
    switches = [
        index
        for index in range(1, len(motion.provenance))
        if (
            motion.provenance[index - 1].archive_clip_index
            != motion.provenance[index].archive_clip_index
        )
    ]
    maximum = 0.0
    for switch in switches:
        start = max(0, switch - 5)
        stop = min(len(motion.joint_position), switch + 6)
        if stop - start > 1:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.diff(
                                motion.joint_position[start:stop],
                                axis=0,
                            )
                        )
                    )
                ),
            )
    return maximum


def select_best_single_substitution(
    archive_path: str | Path,
    fragment_bank_path: str | Path,
    *,
    reference_clip_index: int,
    model_path: str | Path,
    candidate_limit: int = 24,
    decay_frames: float = 8.0,
    maximum_joint_correction_rad: float = 0.35,
    maximum_foot_target_error_m: float = 0.001,
    maximum_source_switch_joint_step_rad: float = 0.20,
) -> AutomaticReferenceSelection:
    """Choose one feasible borrowed stair step using metadata and measured fit.

    The semantic bank only proposes candidates.  Final selection is based on
    actually reconstructing each shortlisted route and measuring its bounded
    joint correction, target-foot residual, and source-switch continuity.
    """

    limit = int(candidate_limit)
    if limit <= 0:
        raise ValueError("candidate_limit must be positive")
    with Path(fragment_bank_path).open() as stream:
        records = [
            json.loads(line)
            for line in stream
            if line.strip()
        ]
    target_records = sorted(
        (
            record
            for record in records
            if int(record["archive_clip_index"])
            == int(reference_clip_index)
        ),
        key=lambda record: int(record["source_range"][0]),
    )
    if not target_records:
        raise ValueError("reference clip has no fragments in the bank")

    proposals: list[
        tuple[float, int, dict[str, object], dict[str, object]]
    ] = []
    for slot, reference in enumerate(target_records):
        for source in records:
            if _compatible_substitution(source, reference):
                proposals.append(
                    (
                        _cheap_substitution_cost(source, reference),
                        slot,
                        source,
                        reference,
                    )
                )
    proposals.sort(
        key=lambda item: (
            item[0],
            str(item[2]["fragment_id"]),
            str(item[3]["fragment_id"]),
        )
    )
    proposals = proposals[:limit]
    if not proposals:
        raise ValueError("fragment bank has no semantic cross-source candidate")

    best: AutomaticReferenceSelection | None = None
    rejected = 0
    evaluated = 0
    for cheap_cost, slot, source_record, reference_record in proposals:
        selections: list[ReferenceFragmentSelection] = []
        for target_slot, target_record in enumerate(target_records):
            reference_selection = _fragment_selection(target_record)
            source_selection = (
                _fragment_selection(source_record)
                if target_slot == slot
                else reference_selection
            )
            selections.append(
                ReferenceFragmentSelection(
                    source=source_selection,
                    reference=reference_selection,
                )
            )
        evaluated += 1
        try:
            result = stitch_archive_to_reference(
                archive_path,
                selections,
                model_path=model_path,
                decay_frames=decay_frames,
                maximum_joint_correction_rad=(
                    maximum_joint_correction_rad
                ),
                maximum_foot_target_error_m=(
                    maximum_foot_target_error_m
                ),
                maximum_source_switch_joint_step_rad=(
                    maximum_source_switch_joint_step_rad
                ),
            )
        except ValueError:
            rejected += 1
            continue
        mean_correction = float(
            np.mean(result.per_frame_joint_correction_rad)
        )
        switch_step = _source_switch_step(result.motion)
        score = (
            result.maximum_joint_correction_rad
            + mean_correction
            + 0.25 * switch_step
            + 0.02 * cheap_cost
        )
        candidate = AutomaticReferenceSelection(
            result=result,
            selections=tuple(selections),
            reference_fragment_id=str(
                reference_record["fragment_id"]
            ),
            source_fragment_id=str(source_record["fragment_id"]),
            reference_clip_index=int(reference_clip_index),
            source_clip_index=int(
                source_record["archive_clip_index"]
            ),
            score=score,
            evaluated_candidate_count=evaluated,
            rejected_candidate_count=rejected,
        )
        if best is None or candidate.score < best.score:
            best = candidate

    if best is None:
        raise ValueError(
            "no cross-source substitution satisfied mechanical bounds"
        )
    return AutomaticReferenceSelection(
        result=best.result,
        selections=best.selections,
        reference_fragment_id=best.reference_fragment_id,
        source_fragment_id=best.source_fragment_id,
        reference_clip_index=best.reference_clip_index,
        source_clip_index=best.source_clip_index,
        score=best.score,
        evaluated_candidate_count=evaluated,
        rejected_candidate_count=rejected,
    )


__all__ = (
    "AutomaticReferenceSelection",
    "ReferenceFragmentSelection",
    "ReferenceStitchResult",
    "select_best_single_substitution",
    "stitch_archive_to_reference",
)
