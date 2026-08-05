"""Practical contact-fragment extraction for canonical motion clips.

The fragment boundary is a stable support event, not an arbitrary fixed-size
window.  Source ranges include both boundary samples; the corresponding
half-open slice is therefore ``[entry_frame, exit_frame + 1)``.  Context ranges
are kept separately so later matching/inertialization can inspect neighboring
poses without changing the semantic fragment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .canonical import CanonicalClip
from .math3d import quaternion_multiply_wxyz


class SupportPhase(str, Enum):
    FLIGHT = "flight"
    LEFT = "left"
    RIGHT = "right"
    DOUBLE = "double"


@dataclass(frozen=True)
class SegmentationConfig:
    """Small set of knobs that affect semantic fragment boundaries."""

    stable_frames: int = 3
    context_before: int = 8
    context_after: int = 8
    contact_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        if self.context_before < 0 or self.context_after < 0:
            raise ValueError("fragment context must be nonnegative")
        if not 0.0 <= self.contact_threshold <= 1.0:
            raise ValueError("contact_threshold must be in [0, 1]")


@dataclass(frozen=True)
class SupportEvent:
    """Beginning of a support run that lasts for ``stable_frames`` samples."""

    frame: int
    phase: SupportPhase
    stable_frames: int

    @property
    def stop_frame(self) -> int:
        return self.frame + self.stable_frames


@dataclass(frozen=True)
class FragmentRef:
    """Half-open semantic and context ranges into one canonical source clip."""

    fragment_id: str
    clip_id: str
    source_start_frame: int
    source_stop_frame: int
    context_start_frame: int
    context_stop_frame: int
    entry_phase: SupportPhase
    exit_phase: SupportPhase

    @property
    def start_frame(self) -> int:
        """Compatibility alias for the semantic source start."""

        return self.source_start_frame

    @property
    def stop_frame(self) -> int:
        """Compatibility alias for the semantic source stop."""

        return self.source_stop_frame

    @property
    def frame_count(self) -> int:
        return self.source_stop_frame - self.source_start_frame


@dataclass(frozen=True)
class ContactPhaseSpan:
    """One run in fragment-relative, half-open frame coordinates."""

    phase: SupportPhase
    start_offset: int
    stop_offset: int


@dataclass(frozen=True)
class EndpointSummary:
    """Compact pose/velocity state used to compare fragment transitions."""

    phase: SupportPhase
    root_position_local_xyz: tuple[float, float, float]
    root_quaternion_local_wxyz: tuple[float, float, float, float]
    root_height_m: float
    root_linear_velocity_local_xyz: tuple[float, float, float]
    root_angular_velocity_local_xyz: tuple[float, float, float]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    sole_position_local_xyz: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]


@dataclass(frozen=True)
class FragmentDescriptor:
    """Lightweight route-matching information for one :class:`FragmentRef`."""

    fragment_id: str
    root_displacement_local_xyz: tuple[float, float, float]
    root_yaw_delta_rad: float
    duration_s: float
    entry: EndpointSummary
    exit: EndpointSummary
    contact_phases: tuple[ContactPhaseSpan, ...]
    terrain_tags: tuple[str, ...]
    action_tags: tuple[str, ...]


def _phase(left_contact: bool, right_contact: bool) -> SupportPhase:
    if left_contact:
        return SupportPhase.DOUBLE if right_contact else SupportPhase.LEFT
    return SupportPhase.RIGHT if right_contact else SupportPhase.FLIGHT


def _frame_phases(
    clip: CanonicalClip,
    contact_threshold: float,
    start_frame: int = 0,
    stop_frame: int | None = None,
) -> tuple[SupportPhase, ...]:
    active = (
        np.asarray(clip.contact)[start_frame:stop_frame] >= contact_threshold
    )
    return tuple(_phase(bool(row[0]), bool(row[1])) for row in active)


def _phase_runs(
    phases: tuple[SupportPhase, ...],
) -> tuple[tuple[SupportPhase, int, int], ...]:
    if not phases:
        return ()
    runs: list[tuple[SupportPhase, int, int]] = []
    start = 0
    current = phases[0]
    for frame in range(1, len(phases)):
        if phases[frame] == current:
            continue
        runs.append((current, start, frame))
        current = phases[frame]
        start = frame
    runs.append((current, start, len(phases)))
    return tuple(runs)


def stable_support_events(
    clip: CanonicalClip,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[SupportEvent, ...]:
    """Return support changes whose observed run survives the stability filter.

    If a short flicker separates two stable runs with the same phase, the later
    run is not emitted as a new semantic event.  This avoids creating D->D or
    L->L fragments from contact relabeling noise.
    """

    phases = _frame_phases(clip, config.contact_threshold)
    events: list[SupportEvent] = []
    for phase, start, stop in _phase_runs(phases):
        run_frames = stop - start
        if run_frames < config.stable_frames:
            continue
        if (
            events
            and events[-1].phase == phase
            and start - events[-1].stop_frame < config.stable_frames
        ):
            continue
        events.append(SupportEvent(start, phase, run_frames))
    return tuple(events)


def segment_clip(
    clip: CanonicalClip,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[FragmentRef, ...]:
    """Split ``clip`` into fragments between adjacent stable support events."""

    events = stable_support_events(clip, config)
    fragments: list[FragmentRef] = []
    for entry, exit in zip(events, events[1:]):
        source_start = entry.frame
        # Include the exit event sample while retaining a half-open source range.
        source_stop = min(clip.frame_count, exit.frame + 1)
        if source_stop - source_start < 2:
            continue
        context_start = max(0, source_start - config.context_before)
        context_stop = min(clip.frame_count, source_stop + config.context_after)
        fragment_id = (
            f"{clip.clip_id}[{source_start}:{source_stop})/"
            f"{entry.phase.value}>{exit.phase.value}"
        )
        fragments.append(
            FragmentRef(
                fragment_id=fragment_id,
                clip_id=clip.clip_id,
                source_start_frame=source_start,
                source_stop_frame=source_stop,
                context_start_frame=context_start,
                context_stop_frame=context_stop,
                entry_phase=entry.phase,
                exit_phase=exit.phase,
            )
        )
    return tuple(fragments)


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _world_to_entry_local(vectors: np.ndarray, entry_yaw: float) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float64)
    output = np.array(value, copy=True)
    cosine = math.cos(entry_yaw)
    sine = math.sin(entry_yaw)
    output[..., 0] = cosine * value[..., 0] + sine * value[..., 1]
    output[..., 1] = -sine * value[..., 0] + cosine * value[..., 1]
    return output


def _tuple3(value: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(item) for item in np.asarray(value))  # type: ignore[return-value]


def _tuple4(value: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(item) for item in np.asarray(value))  # type: ignore[return-value]


def _heading_local_quaternion(
    quaternion: np.ndarray, entry_yaw: float
) -> tuple[float, float, float, float]:
    heading_inverse = np.asarray(
        (
            math.cos(-entry_yaw / 2.0),
            0.0,
            0.0,
            math.sin(-entry_yaw / 2.0),
        ),
        dtype=np.float64,
    )
    relative = np.asarray(
        quaternion_multiply_wxyz(
            heading_inverse, quaternion
        ),
        dtype=np.float64,
    )
    relative /= np.linalg.norm(relative)
    if relative[0] < 0.0:
        relative *= -1.0
    return _tuple4(relative)


def _endpoint_summary(
    clip: CanonicalClip,
    frame: int,
    phase: SupportPhase,
    entry_position: np.ndarray,
    entry_yaw: float,
) -> EndpointSummary:
    root_offset = _world_to_entry_local(
        np.asarray(clip.root_position_world[frame], dtype=np.float64) - entry_position,
        entry_yaw,
    )
    sole_offset = _world_to_entry_local(
        np.asarray(clip.sole_position_world[frame], dtype=np.float64)
        - entry_position[None, :],
        entry_yaw,
    )
    linear_velocity = _world_to_entry_local(
        clip.root_linear_velocity_world[frame], entry_yaw
    )
    angular_velocity = _world_to_entry_local(
        clip.root_angular_velocity_world[frame], entry_yaw
    )
    return EndpointSummary(
        phase=phase,
        root_position_local_xyz=_tuple3(root_offset),
        root_quaternion_local_wxyz=_heading_local_quaternion(
            clip.root_quaternion_world_wxyz[frame], entry_yaw
        ),
        root_height_m=float(clip.root_position_world[frame, 2]),
        root_linear_velocity_local_xyz=_tuple3(linear_velocity),
        root_angular_velocity_local_xyz=_tuple3(angular_velocity),
        joint_position=tuple(float(value) for value in clip.joint_position[frame]),
        joint_velocity=tuple(float(value) for value in clip.joint_velocity[frame]),
        sole_position_local_xyz=(
            _tuple3(sole_offset[0]),
            _tuple3(sole_offset[1]),
        ),
    )


def _terrain_tags(clip: CanonicalClip) -> tuple[str, ...]:
    terrain_keywords = (
        "terrain",
        "stair",
        "step",
        "curb",
        "slope",
        "ramp",
        "platform",
    )
    tags = tuple(
        tag
        for tag in clip.action_tags
        if any(keyword in tag.lower() for keyword in terrain_keywords)
    )
    if tags:
        return tags
    flat_sources = {"flat", "lafan", "bones", "takara", "takarawalk", "takara-walk"}
    if any(tag.lower() in flat_sources for tag in clip.action_tags):
        return ("flat",)
    return ("terrain",) if clip.terrain is not None else ("flat",)


def extract_fragment_descriptor(
    clip: CanonicalClip,
    ref: FragmentRef,
    *,
    contact_threshold: float = 0.5,
) -> FragmentDescriptor:
    """Extract an entry-root-local descriptor suitable for coarse route search."""

    if ref.clip_id != clip.clip_id:
        raise ValueError("fragment belongs to a different canonical clip")
    if not (
        0
        <= ref.source_start_frame
        < ref.source_stop_frame
        <= clip.frame_count
    ):
        raise ValueError("fragment source range is outside the canonical clip")
    start = ref.source_start_frame
    exit_frame = ref.source_stop_frame - 1
    entry_position = np.asarray(clip.root_position_world[start], dtype=np.float64)
    entry_quaternion = np.asarray(
        clip.root_quaternion_world_wxyz[start], dtype=np.float64
    )
    entry_yaw = _yaw_wxyz(entry_quaternion)
    exit_yaw = _yaw_wxyz(clip.root_quaternion_world_wxyz[exit_frame])
    displacement = _world_to_entry_local(
        np.asarray(clip.root_position_world[exit_frame], dtype=np.float64)
        - entry_position,
        entry_yaw,
    )
    phase_spans = tuple(
        ContactPhaseSpan(phase, span_start, span_stop)
        for phase, span_start, span_stop in _phase_runs(
            _frame_phases(
                clip,
                contact_threshold,
                ref.source_start_frame,
                ref.source_stop_frame,
            )
        )
    )
    return FragmentDescriptor(
        fragment_id=ref.fragment_id,
        root_displacement_local_xyz=_tuple3(displacement),
        root_yaw_delta_rad=_wrap_angle(exit_yaw - entry_yaw),
        duration_s=(exit_frame - start) / float(clip.fps),
        entry=_endpoint_summary(
            clip,
            start,
            ref.entry_phase,
            entry_position,
            entry_yaw,
        ),
        exit=_endpoint_summary(
            clip,
            exit_frame,
            ref.exit_phase,
            entry_position,
            entry_yaw,
        ),
        contact_phases=phase_spans,
        terrain_tags=_terrain_tags(clip),
        action_tags=tuple(clip.action_tags),
    )


__all__ = (
    "ContactPhaseSpan",
    "EndpointSummary",
    "FragmentDescriptor",
    "FragmentRef",
    "SegmentationConfig",
    "SupportEvent",
    "SupportPhase",
    "extract_fragment_descriptor",
    "segment_clip",
    "stable_support_events",
)
