"""Contact contracts for native split-height G1 stair entries."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .joints import ContractError


@dataclass(frozen=True)
class CompleteStepUpSequence:
    start_frame: int
    first_contact_frame: int
    trailing_contact_frame: int
    end_exclusive: int
    landing_foot: int
    source_first_rise_m: float
    source_final_height_delta_m: float


@dataclass(frozen=True)
class PlacedStepUpValidation:
    unsupported_frame_count: int
    final_split_height_contact_transfer: bool
    contact_height_pattern_error_m: float
    maximum_stance_contact_error_m: float
    minimum_sole_clearance_m: float


def swing_clearance_targets(
    *,
    foot_position_world: object,
    support_mask: object,
    minimum_sole_clearance_m: object,
    minimum_allowed_clearance_m: float = -0.025,
    clearance_margin_m: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ankle targets that lift only colliding unsupported feet."""

    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    clearance = np.asarray(minimum_sole_clearance_m, dtype=np.float64)
    if (
        feet.shape != (2, 3)
        or support.dtype != np.bool_
        or support.shape != (2,)
        or clearance.shape != (2,)
        or not np.isfinite(feet).all()
        or not np.isfinite(clearance).all()
        or not math.isfinite(float(minimum_allowed_clearance_m))
        or not math.isfinite(float(clearance_margin_m))
        or minimum_allowed_clearance_m >= clearance_margin_m
        or clearance_margin_m < 0.0
    ):
        raise ContractError("step-up swing-clearance input is invalid")
    mask = (~support) & (
        clearance < float(minimum_allowed_clearance_m)
    )
    targets = feet.copy()
    targets[mask, 2] += float(clearance_margin_m) - clearance[mask]
    return np.ascontiguousarray(mask), np.ascontiguousarray(targets)


def _contact_arrays(
    support_mask: object, surface_height_m: object
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support_mask)
    surface = np.asarray(surface_height_m, dtype=np.float64)
    if (
        support.dtype != np.bool_
        or support.ndim != 2
        or support.shape[1:] != (2,)
        or surface.shape != support.shape
        or len(support) < 1
        or not np.isfinite(surface).all()
    ):
        raise ContractError(
            "step-up support and surface arrays must be finite (T,2) arrays"
        )
    return support, surface


def find_complete_step_up_sequence(
    *,
    support_mask: object,
    surface_height_m: object,
    start_frame: int,
    first_contact_frame: int,
    landing_foot: int,
    stable_contact_frames: int = 3,
    search_horizon_frames: int = 120,
) -> CompleteStepUpSequence:
    """Find the trailing-foot contact that completes one native stair entry."""

    support, surface = _contact_arrays(support_mask, surface_height_m)
    frames = len(support)
    if (
        type(start_frame) is not int
        or type(first_contact_frame) is not int
        or type(landing_foot) is not int
        or type(stable_contact_frames) is not int
        or type(search_horizon_frames) is not int
        or not 0 <= start_frame < first_contact_frame < frames
        or landing_foot not in (0, 1)
        or stable_contact_frames < 1
        or search_horizon_frames < stable_contact_frames
        or first_contact_frame + stable_contact_frames > frames
    ):
        raise ContractError("step-up sequence bounds are invalid")
    trailing_foot = 1 - landing_foot
    first_stop = first_contact_frame + stable_contact_frames
    if (
        not bool(support[first_contact_frame:first_stop, landing_foot].all())
        or not bool(support[first_contact_frame, trailing_foot])
    ):
        raise ContractError("step-up first contact is not stably supported")
    first_rise = float(
        surface[first_contact_frame, landing_foot]
        - surface[first_contact_frame, trailing_foot]
    )
    if not math.isfinite(first_rise) or first_rise < 0.08:
        raise ContractError("step-up first contact is not elevated")

    search_stop = min(
        frames - stable_contact_frames + 1,
        first_contact_frame + search_horizon_frames + 1,
    )
    trailing_contact = None
    for frame in range(first_stop, search_stop):
        stop = frame + stable_contact_frames
        previous = support[
            max(first_contact_frame, frame - stable_contact_frames) : frame,
            trailing_foot,
        ]
        if (
            not bool(previous.any())
            and bool(support[frame:stop, trailing_foot].all())
        ):
            trailing_contact = frame
            break
    if trailing_contact is None:
        raise ContractError("step-up trailing contact was not found")
    end_exclusive = trailing_contact + stable_contact_frames
    if not bool(support[start_frame:end_exclusive].any(axis=1).all()):
        raise ContractError("step-up sequence contains an unsupported frame")
    final_delta = float(
        surface[trailing_contact, trailing_foot]
        - surface[first_contact_frame, landing_foot]
    )
    return CompleteStepUpSequence(
        start_frame=start_frame,
        first_contact_frame=first_contact_frame,
        trailing_contact_frame=trailing_contact,
        end_exclusive=end_exclusive,
        landing_foot=landing_foot,
        source_first_rise_m=first_rise,
        source_final_height_delta_m=final_delta,
    )


def validate_placed_step_up(
    *,
    sequence: CompleteStepUpSequence,
    source_support_mask: object,
    ankle_clearance_m: object,
    sole_clearance_m: object,
    target_final_height_delta_m: float,
    stance_contact_tolerance_m: float = 0.020,
    contact_height_tolerance_m: float = 0.025,
    minimum_split_height_m: float = 0.080,
    maximum_penetration_m: float = 0.025,
    maximum_contact_sole_clearance_m: float = 0.035,
) -> PlacedStepUpValidation:
    """Validate support continuity and both target stair contacts."""

    if not isinstance(sequence, CompleteStepUpSequence):
        raise ContractError("step-up validation sequence is invalid")
    support = np.asarray(source_support_mask)
    ankle = np.asarray(ankle_clearance_m, dtype=np.float64)
    sole = np.asarray(sole_clearance_m, dtype=np.float64)
    frames = len(support) if support.ndim >= 1 else 0
    scalar_values = (
        target_final_height_delta_m,
        stance_contact_tolerance_m,
        contact_height_tolerance_m,
        minimum_split_height_m,
        maximum_penetration_m,
        maximum_contact_sole_clearance_m,
    )
    if (
        support.dtype != np.bool_
        or support.shape != (frames, 2)
        or ankle.shape != (frames, 2)
        or sole.shape != (frames, 2, 7)
        or not np.isfinite(ankle).all()
        or not np.isfinite(sole).all()
        or not all(math.isfinite(float(value)) for value in scalar_values)
        or min(scalar_values[1:]) <= 0.0
        or not 0 <= sequence.start_frame < sequence.end_exclusive <= frames
    ):
        raise ContractError("placed step-up validation input is invalid")

    start = sequence.start_frame
    stop = sequence.end_exclusive
    first = sequence.first_contact_frame
    trailing = sequence.trailing_contact_frame
    landing_foot = sequence.landing_foot
    mapped = support & (np.abs(ankle) <= stance_contact_tolerance_m)
    unsupported = int((~mapped[start:stop].any(axis=1)).sum())
    if unsupported:
        raise ContractError("placed step-up contains an unsupported frame")
    if not bool(mapped[first, landing_foot]):
        raise ContractError("placed step-up leading contact is unsupported")
    trailing_foot = 1 - landing_foot
    if not bool(mapped[trailing:stop, trailing_foot].all()):
        raise ContractError(
            "placed step-up does not finish with a stable contact transfer"
        )

    height_error = abs(
        float(target_final_height_delta_m)
        - sequence.source_final_height_delta_m
    )
    if (
        abs(float(target_final_height_delta_m)) < minimum_split_height_m
        or abs(sequence.source_final_height_delta_m)
        < minimum_split_height_m
    ):
        raise ContractError("placed step-up does not finish at split height")
    if height_error > contact_height_tolerance_m:
        raise ContractError("placed step-up contact height pattern is invalid")

    span_sole = sole[start:stop]
    minimum_clearance = float(span_sole.min())
    if minimum_clearance < -maximum_penetration_m:
        raise ContractError("placed step-up sole penetration is excessive")
    contact_sole = np.concatenate(
        (
            sole[first, landing_foot].reshape(-1),
            sole[trailing:stop, trailing_foot].reshape(-1),
        )
    )
    if (
        float(contact_sole.min()) < -maximum_penetration_m
        or float(contact_sole.max()) > maximum_contact_sole_clearance_m
    ):
        raise ContractError("placed step-up sole contact is incomplete")
    stance_error = float(np.abs(ankle[start:stop][support[start:stop]]).max())
    return PlacedStepUpValidation(
        unsupported_frame_count=unsupported,
        final_split_height_contact_transfer=True,
        contact_height_pattern_error_m=height_error,
        maximum_stance_contact_error_m=stance_error,
        minimum_sole_clearance_m=minimum_clearance,
    )
