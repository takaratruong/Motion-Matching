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

from .motions import (
    CanonicalReach,
    ReachAugmentation,
    ReachHand,
    _contacts,
)


MIRROR_BONES = (
    0, 1,
    8, 9, 10, 11, 12, 13,
    2, 3, 4, 5, 6, 7,
    14, 15, 16,
    24, 25, 26, 27, 28, 29, 30,
    17, 18, 19, 20, 21, 22, 23,
)
_REFLECTION = np.diag([1.0, 1.0, -1.0])


def _validate_map() -> None:
    if len(MIRROR_BONES) != len(G1_SKELETON.names):
        raise ValueError("G1 mirror map has wrong bone count")
    for destination, source in enumerate(MIRROR_BONES):
        destination_name = G1_SKELETON.names[destination]
        expected = (
            destination_name.replace("Left", "Right")
            if destination_name.startswith("Left")
            else destination_name.replace("Right", "Left")
            if destination_name.startswith("Right")
            else destination_name
        )
        if G1_SKELETON.names[source] != expected:
            raise ValueError(
                f"invalid G1 mirror pair {destination_name} -> "
                f"{G1_SKELETON.names[source]}"
            )


_validate_map()


def _reflect_rotations(rotations: np.ndarray) -> np.ndarray:
    matrices = holden_quat.to_xform(rotations)
    reflected = np.einsum(
        "ij,...jk,kl->...il",
        _REFLECTION,
        matrices,
        _REFLECTION,
    )
    return holden_quat.from_xform(reflected)


def mirror_reach(
    source: CanonicalReach,
    *,
    allow_mirrored: bool = False,
) -> CanonicalReach:
    source.validate()
    if (
        source.augmentation == ReachAugmentation.MIRRORED
        and not allow_mirrored
    ):
        raise ValueError("reach is already mirrored")
    world_positions, world_rotations = forward_local_hierarchy(
        source.positions.astype(np.float64),
        source.rotations.astype(np.float64),
        G1_SKELETON.parents,
    )
    mirrored_positions = world_positions[:, MIRROR_BONES].copy()
    mirrored_positions[..., 2] *= -1.0
    mirrored_rotations = _reflect_rotations(
        world_rotations[:, MIRROR_BONES]
    )
    local_positions, local_rotations = world_to_local(
        mirrored_positions,
        mirrored_rotations,
        G1_SKELETON.parents,
    )
    local_rotations = holden_quat.unroll(
        holden_quat.normalize(local_rotations)
    )
    active_hand = (
        ReachHand.RIGHT
        if source.active_hand == ReachHand.LEFT
        else ReachHand.LEFT
    )
    augmentation = (
        ReachAugmentation.MIRRORED
        if source.augmentation == ReachAugmentation.CAPTURED
        else ReachAugmentation.CAPTURED
    )
    original = (
        source.reach_id
        if augmentation == ReachAugmentation.MIRRORED
        else None
    )
    endpoint_position = np.asarray(source.endpoint_position_root).copy()
    endpoint_position[2] *= -1.0
    endpoint_rotation = _reflect_rotations(
        np.asarray(source.endpoint_rotation_root_wxyz)
    )
    approach = np.asarray(source.approach_direction_root).copy()
    approach[2] *= -1.0
    result = CanonicalReach(
        reach_id=f"{source.reach_id}:mirror",
        original_reach_id=original,
        proposal_id=source.proposal_id,
        sequence_id=source.sequence_id,
        active_hand=active_hand,
        augmentation=augmentation,
        fps=source.fps,
        positions=local_positions.astype(np.float32),
        velocities=finite_difference_vectors(local_positions, source.fps),
        rotations=local_rotations.astype(np.float32),
        angular_velocities=finite_difference_quaternions(
            local_rotations, source.fps
        ),
        foot_contacts=source.foot_contacts[:, ::-1].copy(),
        source_frames=source.source_frames.copy(),
        contact_index=source.contact_index,
        return_available=source.return_available,
        endpoint_position_root=endpoint_position.astype(np.float32),
        endpoint_rotation_root_wxyz=holden_quat.normalize(
            endpoint_rotation
        ).astype(np.float32),
        approach_direction_root=approach.astype(np.float32),
    )
    expected_contacts = _contacts(mirrored_positions, source.fps)
    if not np.array_equal(result.foot_contacts, expected_contacts):
        result.foot_contacts = expected_contacts
    result.validate()
    return result


def build_bilateral_reaches(
    captured: list[CanonicalReach],
) -> list[CanonicalReach]:
    result: list[CanonicalReach] = []
    for reach in captured:
        if reach.augmentation != ReachAugmentation.CAPTURED:
            raise ValueError("bilateral builder accepts captured reaches only")
        result.extend((reach, mirror_reach(reach)))
    return result
