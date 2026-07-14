import numpy as np

from resources import quat as holden_quat

from .schema import (
    CanonicalInteractionClip,
    InteractionHand,
    InteractionPhase,
    InteractionValidationError,
    LabeledInteractionClip,
    PhaseConfig,
)


def hand_in_object(
    hand_position: np.ndarray,
    hand_rotation: np.ndarray,
    object_position: np.ndarray,
    object_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inv_object = holden_quat.inv(object_rotation)
    return (
        holden_quat.mul_vec(
            inv_object, hand_position - object_position
        ),
        holden_quat.abs(
            holden_quat.normalize(
                holden_quat.mul(inv_object, hand_rotation)
            )
        ),
    )


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _stable_contact_frame(
    clip: CanonicalInteractionClip,
    hand: InteractionHand,
    config: PhaseConfig,
) -> int | None:
    position, rotation = hand_in_object(
        clip.hand_positions[:, int(hand)],
        clip.hand_rotations[:, int(hand)],
        clip.object_positions,
        clip.object_rotations,
    )
    distinct = np.flatnonzero(
        np.r_[True, clip.source_frames[1:] != clip.source_frames[:-1]]
    )
    for index in range(
        len(distinct) - config.stable_source_samples + 1
    ):
        sample = distinct[index:index + config.stable_source_samples]
        if not np.all(clip.hand_contacts[sample, int(hand)]):
            continue
        window_positions = position[sample]
        pairwise_positions = np.linalg.norm(
            window_positions[:, None] - window_positions[None, :],
            axis=-1,
        )
        if (
            float(np.max(pairwise_positions))
            > config.max_relative_position_m
        ):
            continue
        window_rotations = rotation[sample]
        pairwise_angles = [
            quaternion_angle(window_rotations[a], window_rotations[b])
            for a in range(len(window_rotations))
            for b in range(a + 1, len(window_rotations))
        ]
        if max(pairwise_angles) > config.max_relative_angle_radians:
            continue
        return int(sample[0])
    return None


def derive_interaction_labels(
    clip: CanonicalInteractionClip,
    config: PhaseConfig = PhaseConfig(),
) -> LabeledInteractionClip:
    clip.validate()
    contacts = {
        hand: _stable_contact_frame(clip, hand, config)
        for hand in InteractionHand
    }
    valid = [
        (hand, frame) for hand, frame in contacts.items() if frame is not None
    ]
    if len(valid) != 1:
        code = "no_stable_contact" if not valid else "ambiguous_active_hand"
        raise InteractionValidationError(code, clip.sequence_id)
    active_hand, contact_frame = valid[0]
    frames = len(clip.positions)
    active_contact = clip.hand_contacts[:, int(active_hand)].astype(bool)
    reach_frame = max(
        0, contact_frame - round(config.reach_seconds * clip.fps)
    )
    time_to_contact = np.maximum(
        0.0, (contact_frame - np.arange(frames)) / clip.fps
    ).astype(np.float32)
    support_height = float(clip.object_positions[contact_frame, 1])

    lifted = np.flatnonzero(
        active_contact[contact_frame:]
        & (
            clip.object_positions[contact_frame:, 1]
            >= support_height + config.lift_height_m
        )
    )
    if not len(lifted):
        code = (
            "contact_lost_before_hold"
            if not np.all(active_contact[contact_frame:])
            else "no_five_centimeter_lift"
        )
        raise InteractionValidationError(code, clip.sequence_id)
    lift_frame = contact_frame + int(lifted[0])
    if not np.all(active_contact[contact_frame:lift_frame + 1]):
        raise InteractionValidationError(
            "contact_lost_before_hold", clip.sequence_id
        )

    object_velocity = clip.object_velocities[:, 1]
    stable = active_contact & (
        np.abs(object_velocity) < config.hold_speed_mps
    )
    hold_samples = round(config.hold_seconds * clip.fps)
    hold_frame = None
    for start in range(lift_frame, frames - hold_samples + 1):
        if np.all(stable[start:start + hold_samples]):
            hold_frame = start
            break
    if hold_frame is None:
        code = (
            "contact_lost_before_hold"
            if not np.all(active_contact[contact_frame:])
            else "no_stable_hold"
        )
        raise InteractionValidationError(code, clip.sequence_id)
    if hold_frame == lift_frame:
        raise InteractionValidationError(
            "no_distinct_lift_phase", clip.sequence_id
        )
    if not np.all(
        active_contact[contact_frame:hold_frame + hold_samples]
    ):
        raise InteractionValidationError(
            "contact_lost_before_hold", clip.sequence_id
        )

    grasp_stop = min(
        frames,
        contact_frame + round(config.grasp_average_seconds * clip.fps),
    )
    grasp_indices = (
        np.flatnonzero(active_contact[contact_frame:grasp_stop])
        + contact_frame
    )
    grasp_positions, grasp_rotations = hand_in_object(
        clip.hand_positions[grasp_indices, int(active_hand)],
        clip.hand_rotations[grasp_indices, int(active_hand)],
        clip.object_positions[grasp_indices],
        clip.object_rotations[grasp_indices],
    )
    grasp_position = np.median(grasp_positions, axis=0)
    aligned = grasp_rotations.copy()
    aligned[np.sum(aligned * aligned[0], axis=1) < 0] *= -1
    grasp_rotation = holden_quat.normalize(np.mean(aligned, axis=0))
    if not np.isfinite(grasp_position).all() or not np.isfinite(
        grasp_rotation
    ).all():
        raise InteractionValidationError("invalid_grasp", clip.sequence_id)

    approach_world = (
        clip.hand_positions[contact_frame, int(active_hand)]
        - clip.hand_positions[reach_frame, int(active_hand)]
    )
    approach_object = holden_quat.mul_vec(
        holden_quat.inv(clip.object_rotations[contact_frame]),
        approach_world,
    )
    approach_object[1] = 0.0
    approach_norm = float(np.linalg.norm(approach_object))
    if approach_norm < 1e-4:
        raise InteractionValidationError("invalid_approach", clip.sequence_id)
    approach_object /= approach_norm

    phases = np.full(frames, InteractionPhase.HOLD, np.uint8)
    phases[:reach_frame] = InteractionPhase.APPROACH
    phases[reach_frame:contact_frame] = InteractionPhase.REACH
    phases[contact_frame:lift_frame] = InteractionPhase.CONTACT
    phases[lift_frame:hold_frame] = InteractionPhase.LIFT
    return LabeledInteractionClip(
        motion=clip,
        active_hand=active_hand,
        phases=phases,
        time_to_contact=time_to_contact,
        contact_frame=contact_frame,
        lift_frame=lift_frame,
        hold_frame=hold_frame,
        support_height=support_height,
        grasp_position_object=grasp_position.astype(np.float32),
        grasp_rotation_object=grasp_rotation.astype(np.float32),
        approach_direction_object=approach_object.astype(np.float32),
    )
