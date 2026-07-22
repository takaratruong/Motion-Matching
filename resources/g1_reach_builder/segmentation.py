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
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("segmentation values must be finite and positive")
        if self.neutral_radius_m > self.return_radius_m:
            raise ValueError("neutral radius cannot exceed return radius")


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
    for start, stop in _runs(distance > config.return_radius_m):
        grab = start + int(np.argmax(distance[start:stop]))
        excursion = float(distance[grab])
        if excursion < config.minimum_excursion_m:
            continue
        prior_neutral = np.flatnonzero(
            distance[: start + 1] <= config.neutral_radius_m
        )
        departure = int(prior_neutral[-1]) if len(prior_neutral) else start
        departure = max(departure, grab - maximum_frames)
        approach_start = max(departure, grab - approach_frames)
        approach_displacement = float(np.linalg.norm(
            trace[grab] - trace[approach_start]
        ))
        returned = stop < len(trace)
        confidence = min(1.0, excursion / 0.45) * min(
            1.0, approach_displacement / 0.10
        )
        if returned:
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

    minimum_separation = int(round(
        config.minimum_separation_s * config.fps
    ))
    merged: list[ReachProposal] = []
    for proposal in proposals:
        if merged and proposal.grab_frame - merged[-1].grab_frame < minimum_separation:
            if proposal.excursion_m > merged[-1].excursion_m:
                merged[-1] = proposal
            continue
        merged.append(proposal)
    return merged


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

