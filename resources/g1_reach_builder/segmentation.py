from dataclasses import dataclass
import hashlib

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_terrain_builder.kinematics import forward_local_hierarchy

from .schema import ReviewCorpus


@dataclass(frozen=True)
class SegmentationConfig:
    fps: float = 25.0
    neutral_radius_m: float = 0.08
    return_radius_m: float = 0.10
    minimum_excursion_m: float = 0.18
    minimum_separation_s: float = 0.60
    maximum_outbound_s: float = 3.60
    approach_window_s: float = 0.20
    minimum_approach_displacement_m: float = 0.01
    minimum_peak_prominence_m: float = 0.08

    def validate(self) -> None:
        values = (
            self.fps,
            self.neutral_radius_m,
            self.return_radius_m,
            self.minimum_excursion_m,
            self.minimum_separation_s,
            self.maximum_outbound_s,
            self.approach_window_s,
            self.minimum_approach_displacement_m,
            self.minimum_peak_prominence_m,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("segmentation values must be finite and positive")
        if self.neutral_radius_m > self.return_radius_m:
            raise ValueError("neutral radius cannot exceed return radius")


@dataclass(frozen=True)
class PauseSegmentationConfig:
    fps: float = 25.0
    neutral_radius_m: float = 0.10
    minimum_excursion_m: float = 0.18
    maximum_contact_speed_mps: float = 0.10
    minimum_contact_separation_s: float = 1.20
    contact_prominence_window_s: float = 0.40
    minimum_speed_prominence_mps: float = 0.03
    minimum_approach_frames: int = 10
    minimum_approach_travel_m: float = 0.18

    def validate(self) -> None:
        positive = (
            self.fps,
            self.neutral_radius_m,
            self.minimum_excursion_m,
            self.maximum_contact_speed_mps,
            self.minimum_contact_separation_s,
            self.contact_prominence_window_s,
            self.minimum_speed_prominence_mps,
            self.minimum_approach_travel_m,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError(
                "pause segmentation values must be finite and positive"
            )
        if self.minimum_approach_frames < 2:
            raise ValueError(
                "pause segmentation requires at least two approach frames"
            )


@dataclass(frozen=True)
class ReachProposal:
    proposal_id: str
    sequence_id: str
    departure_frame: int
    grab_frame: int
    source_departure_frame: int
    source_grab_frame: int
    excursion_m: float
    approach_displacement_m: float
    confidence: float
    status: str = "pending"


def outbound_approach_delta(
    trace: np.ndarray,
    departure_frame: int,
    grab_frame: int,
    *,
    minimum_displacement_m: float = 0.01,
    preferred_window_frames: int = 5,
    maximum_window_frames: int = 25,
) -> np.ndarray:
    trace = np.asarray(trace, np.float64)
    if (
        trace.ndim != 2
        or trace.shape[1] != 3
        or not 0 <= departure_frame <= grab_frame < len(trace)
        or preferred_window_frames < 1
        or maximum_window_frames < preferred_window_frames
        or not np.isfinite(minimum_displacement_m)
        or minimum_displacement_m <= 0.0
    ):
        raise ValueError("invalid outbound approach query")
    preferred_start = max(
        departure_frame, grab_frame - preferred_window_frames
    )
    delta = trace[grab_frame] - trace[preferred_start]
    if float(np.linalg.norm(delta)) >= minimum_displacement_m:
        return delta
    earliest = max(departure_frame, grab_frame - maximum_window_frames)
    for start in range(preferred_start - 1, earliest - 1, -1):
        candidate = trace[grab_frame] - trace[start]
        if float(np.linalg.norm(candidate)) >= minimum_displacement_m:
            return candidate
    return delta


def _neutral_center(trace: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.gradient(trace, 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    threshold = float(np.percentile(speed, 20.0))
    slow = trace[speed <= threshold + 1e-12]
    if len(slow) == 0:
        slow = trace
    cells = np.rint(slow / 0.04).astype(np.int64)
    unique, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    largest = int(np.argmax(counts))
    del unique
    return np.median(slow[inverse == largest], axis=0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def prominent_endpoint_frames(
    distance: np.ndarray,
    config: SegmentationConfig,
) -> list[int]:
    distance = np.asarray(distance, np.float64)
    if distance.ndim != 1 or len(distance) < 3:
        return []
    window = int(round(config.maximum_outbound_s * config.fps))
    candidates = np.flatnonzero(
        (distance[1:-1] >= distance[:-2])
        & (distance[1:-1] > distance[2:])
        & (distance[1:-1] >= config.minimum_excursion_m)
    ) + 1
    prominent = []
    for peak in candidates.tolist():
        left = float(np.min(distance[max(0, peak - window) : peak + 1]))
        right = float(
            np.min(distance[peak : min(len(distance), peak + window + 1)])
        )
        if (
            float(distance[peak]) - max(left, right)
            >= config.minimum_peak_prominence_m
        ):
            prominent.append(peak)

    for start, stop in _runs(distance > config.return_radius_m):
        peak = start + int(np.argmax(distance[start:stop]))
        if distance[peak] >= config.minimum_excursion_m:
            prominent.append(peak)

    minimum_separation = int(round(
        config.minimum_separation_s * config.fps
    ))
    selected: list[int] = []
    for peak in sorted(set(prominent)):
        if selected and peak - selected[-1] <= minimum_separation:
            if distance[peak] > distance[selected[-1]]:
                selected[-1] = peak
        else:
            selected.append(peak)
    return selected


def propose_wrist_trace(
    trace: np.ndarray,
    source_frames: np.ndarray,
    sequence_id: str,
    config: SegmentationConfig = SegmentationConfig(),
) -> list[ReachProposal]:
    config.validate()
    trace = np.asarray(trace, np.float64)
    source_frames = np.asarray(source_frames)
    if trace.ndim != 2 or trace.shape[1] != 3 or len(trace) < 2:
        raise ValueError("wrist trace shape must be (T, 3) with T >= 2")
    if source_frames.shape != (len(trace),):
        raise ValueError("source frame count does not match wrist trace")
    if not np.isfinite(trace).all():
        raise ValueError("wrist trace contains non-finite values")

    center = _neutral_center(trace, config.fps)
    distance = np.linalg.norm(trace - center, axis=1)
    approach_frames = max(1, int(round(
        config.approach_window_s * config.fps
    )))
    maximum_frames = int(round(config.maximum_outbound_s * config.fps))
    proposals: list[ReachProposal] = []
    for peak in prominent_endpoint_frames(distance, config):
        grab = peak
        excursion = float(distance[grab])
        prior_neutral = np.flatnonzero(
            distance[: grab + 1] <= config.neutral_radius_m
        )
        if len(prior_neutral):
            departure = int(prior_neutral[-1])
        else:
            local_start = max(0, grab - maximum_frames)
            departure = local_start + int(np.argmin(
                distance[local_start : grab + 1]
            ))
        departure = max(departure, grab - maximum_frames)
        for candidate in range(
            grab,
            max(departure, grab - int(round(config.fps))) - 1,
            -1,
        ):
            approach_start = max(departure, candidate - approach_frames)
            if np.linalg.norm(trace[candidate] - trace[approach_start]) >= (
                config.minimum_approach_displacement_m
            ):
                grab = candidate
                break
        approach_delta = outbound_approach_delta(
            trace,
            departure,
            grab,
            minimum_displacement_m=config.minimum_approach_displacement_m,
            preferred_window_frames=approach_frames,
            maximum_window_frames=int(round(config.fps)),
        )
        approach_displacement = float(np.linalg.norm(approach_delta))
        confidence = min(1.0, excursion / 0.45) * min(
            1.0, approach_displacement / 0.10
        )
        if peak + approach_frames < len(trace):
            confidence = min(1.0, confidence + 0.10)
        source_grab = int(source_frames[grab])
        identity = hashlib.sha256(
            f"{sequence_id}\0{source_grab}".encode("utf-8")
        ).hexdigest()[:16]
        proposals.append(ReachProposal(
            proposal_id=f"{sequence_id}:{identity}",
            sequence_id=sequence_id,
            departure_frame=departure,
            grab_frame=grab,
            source_departure_frame=int(source_frames[departure]),
            source_grab_frame=source_grab,
            excursion_m=excursion,
            approach_displacement_m=approach_displacement,
            confidence=float(confidence),
        ))

    return proposals


def _smoothed_wrist_speed(trace: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.gradient(trace, 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return np.convolve(
        speed,
        np.asarray([0.25, 0.50, 0.25], np.float64),
        mode="same",
    )


def _contact_pause_frames(
    distance: np.ndarray,
    speed: np.ndarray,
    config: PauseSegmentationConfig,
) -> list[int]:
    minima = np.flatnonzero(
        (speed[1:-1] <= speed[:-2])
        & (speed[1:-1] < speed[2:])
    ) + 1
    minima = minima[
        (distance[minima] >= config.minimum_excursion_m)
        & (speed[minima] <= config.maximum_contact_speed_mps)
    ]
    prominence_window = max(
        1, int(round(config.contact_prominence_window_s * config.fps))
    )
    prominent: list[int] = []
    for frame in minima.tolist():
        left = speed[max(0, frame - prominence_window) : frame + 1]
        right = speed[frame : min(len(speed), frame + prominence_window + 1)]
        prominence = min(float(np.max(left)), float(np.max(right))) - float(
            speed[frame]
        )
        if prominence >= config.minimum_speed_prominence_mps:
            prominent.append(frame)

    separation = max(
        1, int(round(config.minimum_contact_separation_s * config.fps))
    )
    selected: list[int] = []
    for frame in sorted(prominent, key=lambda value: (speed[value], value)):
        if all(abs(frame - current) >= separation for current in selected):
            selected.append(frame)
    return sorted(selected)


def propose_pause_bounded_wrist_trace(
    trace: np.ndarray,
    source_frames: np.ndarray,
    sequence_id: str,
    config: PauseSegmentationConfig = PauseSegmentationConfig(),
) -> list[ReachProposal]:
    config.validate()
    trace = np.asarray(trace, np.float64)
    source_frames = np.asarray(source_frames)
    if trace.ndim != 2 or trace.shape[1] != 3 or len(trace) < 3:
        raise ValueError("wrist trace shape must be (T, 3) with T >= 3")
    if source_frames.shape != (len(trace),):
        raise ValueError("source frame count does not match wrist trace")
    if not np.isfinite(trace).all():
        raise ValueError("wrist trace contains non-finite values")

    center = _neutral_center(trace, config.fps)
    distance = np.linalg.norm(trace - center, axis=1)
    speed = _smoothed_wrist_speed(trace, config.fps)
    proposals: list[ReachProposal] = []
    previous_contact = -1
    for contact in _contact_pause_frames(distance, speed, config):
        interval_start = previous_contact + 1
        interval = np.arange(interval_start, contact + 1)
        neutral = interval[
            distance[interval] <= config.neutral_radius_m
        ]
        if len(neutral):
            departure = int(neutral[np.argmin(speed[neutral])])
        else:
            departure = interval_start + int(
                np.argmin(distance[interval_start : contact + 1])
            )
        approach_frames = contact - departure + 1
        approach_delta = trace[contact] - trace[departure]
        approach_travel = float(np.linalg.norm(approach_delta))
        if (
            approach_frames < config.minimum_approach_frames
            or approach_travel < config.minimum_approach_travel_m
        ):
            continue

        source_grab = int(source_frames[contact])
        identity = hashlib.sha256(
            f"{sequence_id}\0pause\0{source_grab}".encode("utf-8")
        ).hexdigest()[:16]
        proposals.append(ReachProposal(
            proposal_id=f"{sequence_id}:{identity}",
            sequence_id=sequence_id,
            departure_frame=departure,
            grab_frame=contact,
            source_departure_frame=int(source_frames[departure]),
            source_grab_frame=source_grab,
            excursion_m=float(distance[contact]),
            approach_displacement_m=approach_travel,
            confidence=min(1.0, approach_travel / 0.45),
        ))
        previous_contact = contact
    return proposals


def _root_relative_left_wrist(
    positions: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    world_positions, world_rotations = forward_local_hierarchy(
        positions.astype(np.float64),
        rotations.astype(np.float64),
        G1_SKELETON.parents,
    )
    simulation = 0
    left_wrist = G1_SKELETON.names.index("LeftWrist")
    inverse_root = holden_quat.inv(world_rotations[:, simulation])
    return holden_quat.mul_vec(
        inverse_root,
        world_positions[:, left_wrist] - world_positions[:, simulation],
    )


def propose_reaches(
    corpus: ReviewCorpus,
    config: SegmentationConfig = SegmentationConfig(),
) -> list[ReachProposal]:
    corpus.validate()
    if config.fps != corpus.fps:
        raise ValueError(
            f"segmentation fps {config.fps} does not match corpus {corpus.fps}"
        )
    result: list[ReachProposal] = []
    for clip, sequence_id in enumerate(corpus.sequence_ids):
        start = int(corpus.range_starts[clip])
        stop = int(corpus.range_stops[clip])
        trace = _root_relative_left_wrist(
            corpus.positions[start:stop], corpus.rotations[start:stop]
        )
        result.extend(propose_wrist_trace(
            trace,
            corpus.source_frames[start:stop],
            sequence_id,
            config,
        ))
    return result


def propose_pause_bounded_reaches(
    corpus: ReviewCorpus,
    config: PauseSegmentationConfig = PauseSegmentationConfig(),
) -> list[ReachProposal]:
    corpus.validate()
    if config.fps != corpus.fps:
        raise ValueError(
            f"segmentation fps {config.fps} does not match corpus {corpus.fps}"
        )
    result: list[ReachProposal] = []
    for clip, sequence_id in enumerate(corpus.sequence_ids):
        start = int(corpus.range_starts[clip])
        stop = int(corpus.range_stops[clip])
        trace = _root_relative_left_wrist(
            corpus.positions[start:stop], corpus.rotations[start:stop]
        )
        result.extend(propose_pause_bounded_wrist_trace(
            trace,
            corpus.source_frames[start:stop],
            sequence_id,
            config,
        ))
    return result
