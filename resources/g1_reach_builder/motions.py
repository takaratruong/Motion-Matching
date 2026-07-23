from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.conversion import (
    finite_difference_quaternions,
    finite_difference_vectors,
)
from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_terrain_builder.kinematics import (
    forward_local_hierarchy,
    world_to_local,
)

from .annotations import ReachAnnotation
from .returns import find_return_stop
from .schema import ReviewCorpus
from .segmentation import outbound_approach_delta


_RETURN_UNAVAILABLE_PROPOSAL_ID = "pickup_north_1:cfdcdaa4db704be8"


class ReachHand(IntEnum):
    LEFT = 0
    RIGHT = 1


class ReachAugmentation(IntEnum):
    CAPTURED = 0
    MIRRORED = 1


@dataclass
class CanonicalReach:
    reach_id: str
    original_reach_id: str | None
    proposal_id: str
    sequence_id: str
    active_hand: ReachHand
    augmentation: ReachAugmentation
    fps: float
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    source_frames: np.ndarray
    contact_index: int
    return_available: bool
    endpoint_position_root: np.ndarray
    endpoint_rotation_root_wxyz: np.ndarray
    approach_direction_root: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.positions)

    def validate(self) -> None:
        frames = len(self.positions)
        if self.fps != 25.0:
            raise ValueError(f"canonical reach requires 25 Hz, got {self.fps}")
        arrays = (
            (self.positions, (frames, 31, 3), "positions"),
            (self.velocities, (frames, 31, 3), "velocities"),
            (self.rotations, (frames, 31, 4), "rotations"),
            (
                self.angular_velocities,
                (frames, 31, 3),
                "angular velocities",
            ),
            (self.foot_contacts, (frames, 2), "foot contacts"),
            (self.source_frames, (frames,), "source frames"),
            (self.endpoint_position_root, (3,), "endpoint position"),
            (
                self.endpoint_rotation_root_wxyz,
                (4,),
                "endpoint rotation",
            ),
            (self.approach_direction_root, (3,), "approach direction"),
        )
        if frames < 2:
            raise ValueError("canonical reach requires at least two frames")
        if not isinstance(self.return_available, (bool, np.bool_)):
            raise ValueError("canonical reach return availability must be boolean")
        if not 0 < self.contact_index < frames:
            raise ValueError("canonical reach contact index is outside clip")
        if (self.contact_index < frames - 1) != self.return_available:
            raise ValueError(
                "canonical reach return availability does not match contact index"
            )
        for value, shape, label in arrays:
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"reach {label} shape must be {shape}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"reach {label} contains non-finite values")
        if not np.array_equal(
            self.source_frames[1:] - self.source_frames[:-1],
            np.ones(frames - 1, np.int32),
        ):
            raise ValueError("canonical reach source frames must be contiguous")
        norm_error = np.max(np.abs(
            np.linalg.norm(self.rotations, axis=-1) - 1.0
        ))
        if norm_error > 1e-4:
            raise ValueError(f"reach rotation norm error {norm_error}")
        if abs(float(np.linalg.norm(
            self.endpoint_rotation_root_wxyz
        )) - 1.0) > 1e-4:
            raise ValueError("reach endpoint rotation is not unit length")
        if abs(float(np.linalg.norm(
            self.approach_direction_root
        )) - 1.0) > 1e-4:
            raise ValueError("reach approach direction is not unit length")
        if self.augmentation == ReachAugmentation.MIRRORED:
            if self.original_reach_id is None:
                raise ValueError("mirrored reach requires original reach id")
        elif self.original_reach_id is not None:
            raise ValueError("captured reach cannot identify a synthetic original")


def _contacts(world_positions: np.ndarray, fps: float) -> np.ndarray:
    toes = world_positions[:, [7, 13]]
    speeds = np.linalg.norm(
        finite_difference_vectors(toes, fps), axis=-1
    )
    floor = np.min(toes[:, :, 1], axis=0, keepdims=True)
    return np.logical_and(
        speeds < 0.15,
        toes[:, :, 1] - floor < 0.06,
    ).astype(np.uint8)


def _reach_from_local(
    *,
    reach_id: str,
    original_reach_id: str | None,
    proposal_id: str,
    sequence_id: str,
    active_hand: ReachHand,
    augmentation: ReachAugmentation,
    positions: np.ndarray,
    rotations: np.ndarray,
    world_positions: np.ndarray,
    source_frames: np.ndarray,
    contact_index: int,
    return_available: bool,
) -> CanonicalReach:
    wrist = (
        G1_SKELETON.names.index("LeftWrist")
        if active_hand == ReachHand.LEFT
        else G1_SKELETON.names.index("RightWrist")
    )
    world_rotations = forward_local_hierarchy(
        positions.astype(np.float64),
        rotations.astype(np.float64),
        G1_SKELETON.parents,
    )[1]
    approach_delta = outbound_approach_delta(
        world_positions[:, wrist],
        0,
        contact_index,
    )
    approach_length = float(np.linalg.norm(approach_delta))
    if approach_length < 0.01:
        raise ValueError(
            f"{sequence_id}: built reach approach displacement is below 0.01 m"
        )
    reach = CanonicalReach(
        reach_id=reach_id,
        original_reach_id=original_reach_id,
        proposal_id=proposal_id,
        sequence_id=sequence_id,
        active_hand=active_hand,
        augmentation=augmentation,
        fps=25.0,
        positions=positions.astype(np.float32),
        velocities=finite_difference_vectors(positions, 25.0),
        rotations=holden_quat.unroll(
            holden_quat.normalize(rotations.astype(np.float64))
        ).astype(np.float32),
        angular_velocities=finite_difference_quaternions(rotations, 25.0),
        foot_contacts=_contacts(world_positions, 25.0),
        source_frames=np.asarray(source_frames, np.int32),
        contact_index=contact_index,
        return_available=return_available,
        endpoint_position_root=world_positions[contact_index, wrist].astype(np.float32),
        endpoint_rotation_root_wxyz=holden_quat.normalize(
            world_rotations[contact_index, wrist]
        ).astype(np.float32),
        approach_direction_root=(approach_delta / approach_length).astype(
            np.float32
        ),
    )
    reach.validate()
    return reach


def build_captured_reach(
    corpus: ReviewCorpus,
    annotation: ReachAnnotation,
) -> CanonicalReach:
    corpus.validate()
    if annotation.status != "accepted":
        raise ValueError("captured reach requires accepted annotation")
    if annotation.active_hand != "left":
        raise ValueError("captured corpus annotations must use the left hand")
    try:
        source_index = corpus.sequence_ids.index(annotation.sequence_id)
    except ValueError as error:
        raise ValueError(
            f"unknown annotation sequence {annotation.sequence_id}"
        ) from error
    source_start = int(corpus.range_starts[source_index])
    source_stop = int(corpus.range_stops[source_index])
    if not (
        0 <= annotation.departure_frame < annotation.grab_frame
        < source_stop - source_start
    ):
        raise ValueError("accepted annotation frames are outside source")
    retained_start = max(
        annotation.departure_frame,
        annotation.grab_frame - 90,
    )
    first = source_start + retained_start
    source_positions = corpus.positions[source_start:source_stop].astype(np.float64)
    source_rotations = corpus.rotations[source_start:source_stop].astype(np.float64)
    source_world_positions, source_world_rotations = forward_local_hierarchy(
        source_positions,
        source_rotations,
        G1_SKELETON.parents,
    )
    wrist = G1_SKELETON.names.index("LeftWrist")
    source_root_inverse = holden_quat.inv(source_world_rotations[:, 0])
    source_wrist_trace = holden_quat.mul_vec(
        source_root_inverse,
        source_world_positions[:, wrist] - source_world_positions[:, 0],
    )
    try:
        retained_stop = find_return_stop(
            source_wrist_trace,
            annotation.departure_frame,
            annotation.grab_frame,
        )
        return_available = True
    except ValueError as error:
        if (
            annotation.proposal_id != _RETURN_UNAVAILABLE_PROPOSAL_ID
            or str(error) != "no paired return stable window exists"
        ):
            raise
        retained_stop = annotation.grab_frame + 1
        return_available = False
    stop = source_start + retained_stop
    if int(corpus.source_frames[source_start + annotation.grab_frame]) != (
        annotation.source_grab_frame
    ):
        raise ValueError("annotation source grab frame no longer matches review")
    local_positions = corpus.positions[first:stop].astype(np.float64)
    local_rotations = corpus.rotations[first:stop].astype(np.float64)
    world_positions, world_rotations = forward_local_hierarchy(
        local_positions, local_rotations, G1_SKELETON.parents
    )
    root_position = world_positions[0, 0].copy()
    root_rotation_inverse = holden_quat.inv(world_rotations[0, 0])
    normalized_positions = holden_quat.mul_vec(
        root_rotation_inverse,
        world_positions - root_position,
    )
    normalized_rotations = holden_quat.mul(
        np.broadcast_to(root_rotation_inverse, world_rotations.shape),
        world_rotations,
    )
    positions, rotations = world_to_local(
        normalized_positions,
        normalized_rotations,
        G1_SKELETON.parents,
    )
    return _reach_from_local(
        reach_id=f"{annotation.proposal_id}:captured",
        original_reach_id=None,
        proposal_id=annotation.proposal_id,
        sequence_id=annotation.sequence_id,
        active_hand=ReachHand.LEFT,
        augmentation=ReachAugmentation.CAPTURED,
        positions=positions,
        rotations=rotations,
        world_positions=normalized_positions,
        source_frames=np.arange(retained_start, retained_stop, dtype=np.int32),
        contact_index=annotation.grab_frame - retained_start,
        return_available=return_available,
    )
