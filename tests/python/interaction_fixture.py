import copy
import dataclasses
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.conversion import (
    finite_difference_quaternions,
    finite_difference_vectors,
)
from resources.g1_interaction_builder.phases import derive_interaction_labels
from resources.g1_interaction_builder.schema import (
    CanonicalInteractionClip,
    InteractionHand,
    LabeledInteractionClip,
    SourcePaths,
)


def write_source_fixture(
    root: Path,
    sequence_id: str = "pickup_table__cup_2__001",
    object_id: str = "cup_2",
    frames: int = 25,
) -> SourcePaths:
    robot_dir = root / "robot"
    objects_dir = root / "objects"
    meta_dir = root / "meta"
    object_usd_dir = root / "object_usd"
    for directory in (robot_dir, objects_dir, meta_dir, object_usd_dir):
        directory.mkdir(parents=True, exist_ok=True)

    robot_record = {
        "dof": np.zeros((frames, 29), np.float32),
        "root_trans_offset": np.zeros((frames, 3), np.float32),
        "root_rot": np.tile(
            np.array([0, 0, 0, 1], np.float32), (frames, 1)
        ),
        "fps": 25.0,
        "hand_dof_pos": np.zeros((frames, 14), np.float32),
    }
    object_record = {
        "root_pos": np.zeros((frames, 1, 3), np.float32),
        "root_quat": np.tile(
            np.array([0, 0, 0, 1], np.float32), (frames, 1, 1)
        ),
        "fps": 25.0,
        "contact_points_left_hand": {
            i: (
                np.array([[0.0, 0.0, 0.0]], np.float32)
                if i >= 10
                else np.empty((0, 3), np.float32)
            )
            for i in range(frames)
        },
        "contact_points_right_hand": {
            i: np.empty((0, 3), np.float32) for i in range(frames)
        },
    }
    meta_record = {
        "object_name": object_id,
        "table_pos": np.array([0.0, 0.0, 0.75], np.float32),
        "table_quat": np.array([0, 0, 0, 1], np.float32),
        "table_size": np.array([1.2, 0.8, 0.05], np.float32),
    }

    robot = robot_dir / f"{sequence_id}.pkl"
    objects = objects_dir / f"{sequence_id}.pkl"
    meta = meta_dir / f"{sequence_id}.pkl"
    object_usd = object_usd_dir / f"{sequence_id}.usd"
    joblib.dump({sequence_id: robot_record}, robot)
    joblib.dump({sequence_id: object_record}, objects)
    joblib.dump({sequence_id: meta_record}, meta)
    object_usd.touch()
    return SourcePaths(
        sequence_id=sequence_id,
        object_id=object_id,
        robot=robot,
        objects=objects,
        meta=meta,
        object_usd=object_usd,
    )


def canonical_pickup_fixture(
    active_hand: InteractionHand = InteractionHand.RIGHT,
) -> CanonicalInteractionClip:
    frames = 75
    fps = 25.0
    frame = np.arange(frames)

    positions = np.zeros((frames, 31, 3), np.float32)
    positions[:, 0, 1] = 1.0
    rotations = holden_quat.eye((frames, 31), dtype=np.float32)

    object_positions = np.tile(
        np.array([0.55, 0.75, -0.10], np.float64), (frames, 1)
    )
    object_positions[37:50, 1] = np.linspace(0.75, 0.90, 13)
    object_positions[50:, 1] = 0.90

    rest_yaw = np.deg2rad(35.0)
    rest_rotation = np.array(
        [np.cos(rest_yaw / 2.0), 0.0, np.sin(rest_yaw / 2.0), 0.0]
    )
    lift_turn = np.zeros(frames, np.float64)
    lift_turn[37:50] = np.linspace(0.0, np.deg2rad(20.0), 13)
    lift_turn[50:] = lift_turn[49]
    world_turn = np.column_stack(
        (
            np.cos(lift_turn / 2.0),
            np.sin(lift_turn / 2.0),
            np.zeros((frames, 2), np.float64),
        )
    )
    object_rotations = holden_quat.normalize(
        holden_quat.mul(world_turn, rest_rotation)
    )

    hand_positions = np.empty((frames, 2, 3), np.float64)
    hand_positions[:, 0] = object_positions[0] + [-0.30, 0.15, 0.20]
    hand_positions[:, 1] = object_positions[0] + [0.30, 0.15, 0.20]
    hand_rotations = holden_quat.eye((frames, 2), dtype=np.float64)

    active = int(active_hand)
    grasp_position_object = np.array([0.02, 0.00, -0.03])
    approach_position_object = np.tile(
        grasp_position_object, (frames, 1)
    )
    approach_position_object[:38, 0] = np.linspace(0.32, 0.03, 38)
    hand_positions[:, active] = object_positions + holden_quat.mul_vec(
        object_rotations, approach_position_object
    )
    hand_rotations[:, active] = object_rotations

    hand_contacts = np.zeros((frames, 2), np.uint8)
    hand_contacts[38:, active] = 1
    hand_dof = np.zeros((frames, 14), np.float32)

    clip = CanonicalInteractionClip(
        sequence_id="pickup_table__cup_2__fixture",
        object_id="cup_2",
        fps=fps,
        positions=positions,
        velocities=finite_difference_vectors(positions, fps),
        rotations=rotations,
        angular_velocities=finite_difference_quaternions(rotations, fps),
        foot_contacts=np.ones((frames, 2), np.uint8),
        hand_contacts=hand_contacts,
        hand_dof=hand_dof,
        hand_dof_velocities=np.zeros_like(hand_dof),
        hand_positions=hand_positions.astype(np.float32),
        hand_rotations=hand_rotations.astype(np.float32),
        object_positions=object_positions.astype(np.float32),
        object_rotations=object_rotations.astype(np.float32),
        object_velocities=finite_difference_vectors(object_positions, fps),
        object_angular_velocities=finite_difference_quaternions(
            object_rotations, fps
        ),
        table_position=np.array([0.0, 0.70, 0.0], np.float32),
        table_rotation=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        table_size=np.array([1.2, 0.05, 0.8], np.float32),
        object_dimensions=np.array([0.08, 0.20, 0.12], np.float32),
        source_frames=frame.astype(np.int32),
    )
    clip.validate()
    return clip


def transform_fixture_world(
    clip: CanonicalInteractionClip,
    translation: Sequence[float],
    yaw_degrees: float,
) -> CanonicalInteractionClip:
    out = copy.deepcopy(clip)
    translation = np.asarray(translation, np.float64)
    yaw = np.deg2rad(yaw_degrees)
    heading = np.array([np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0])

    def move_positions(values: np.ndarray) -> np.ndarray:
        return holden_quat.mul_vec(heading, values) + translation

    def move_rotations(values: np.ndarray) -> np.ndarray:
        q = np.broadcast_to(heading, values.shape)
        return holden_quat.normalize(holden_quat.mul(q, values))

    out.positions[:, 0] = move_positions(out.positions[:, 0])
    out.rotations[:, 0] = move_rotations(out.rotations[:, 0])
    out.velocities[:, 0] = holden_quat.mul_vec(
        heading, out.velocities[:, 0]
    )
    out.angular_velocities[:, 0] = holden_quat.mul_vec(
        heading, out.angular_velocities[:, 0]
    )
    out.hand_positions = move_positions(out.hand_positions)
    out.hand_rotations = move_rotations(out.hand_rotations)
    out.object_positions = move_positions(out.object_positions)
    out.object_rotations = move_rotations(out.object_rotations)
    out.object_velocities = holden_quat.mul_vec(
        heading, out.object_velocities
    )
    out.object_angular_velocities = holden_quat.mul_vec(
        heading, out.object_angular_velocities
    )
    out.table_position = move_positions(out.table_position)
    out.table_rotation = move_rotations(out.table_rotation)
    out.validate()
    return out


def labeled_clips_for_objects(
    object_ids: Sequence[str],
) -> list[LabeledInteractionClip]:
    base = derive_interaction_labels(canonical_pickup_fixture())
    clips = []
    for index, object_id in enumerate(object_ids):
        motion = copy.deepcopy(base.motion)
        motion.sequence_id = f"pickup_table__{object_id}__{index:03d}"
        motion.object_id = object_id
        clips.append(dataclasses.replace(base, motion=motion))
    return clips
