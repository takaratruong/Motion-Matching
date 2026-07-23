from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReturnSegmentationConfig:
    fps: float = 25.0
    maximum_return_s: float = 8.0
    stable_window_s: float = 0.20
    maximum_stable_speed_mps: float = 0.40
    minimum_retraction_m: float = 0.08

    def validate(self) -> None:
        values = (
            self.fps,
            self.maximum_return_s,
            self.stable_window_s,
            self.maximum_stable_speed_mps,
            self.minimum_retraction_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("return segmentation values must be finite and positive")


def find_return_stop(
    trace: np.ndarray,
    departure_frame: int,
    grab_frame: int,
    config: ReturnSegmentationConfig = ReturnSegmentationConfig(),
) -> int:
    config.validate()
    trace = np.asarray(trace, np.float64)
    if trace.ndim != 2 or trace.shape[1] != 3 or len(trace) < 2:
        raise ValueError("return trace shape must be (T, 3) with T >= 2")
    if not np.isfinite(trace).all():
        raise ValueError("return trace contains non-finite values")
    if not 0 <= departure_frame < grab_frame < len(trace):
        raise ValueError("invalid return frame ordering")

    stable_frames = int(round(config.stable_window_s * config.fps))
    maximum_return_frames = int(round(config.maximum_return_s * config.fps))
    if stable_frames < 1 or maximum_return_frames < stable_frames:
        raise ValueError("return segmentation window is invalid")
    search_stop = min(len(trace), grab_frame + maximum_return_frames + 1)
    if search_stop - (grab_frame + 1) < stable_frames:
        raise ValueError("no paired return stable window exists")

    speeds = np.empty(len(trace), np.float64)
    speeds[0] = np.inf
    speeds[1:] = np.linalg.norm(np.diff(trace, axis=0), axis=1) * config.fps
    departure_wrist = trace[departure_frame]
    grab_wrist = trace[grab_frame]
    best_stop: int | None = None
    best_distance = np.inf
    for start in range(grab_frame + 1, search_stop - stable_frames + 1):
        stop = start + stable_frames
        window = trace[start:stop]
        if np.any(speeds[start:stop] > config.maximum_stable_speed_mps):
            continue
        if np.any(
            np.linalg.norm(window - grab_wrist, axis=1)
            < config.minimum_retraction_m
        ):
            continue
        distance = float(np.linalg.norm(np.mean(window, axis=0) - departure_wrist))
        if distance < best_distance:
            best_distance = distance
            best_stop = stop
    if best_stop is None:
        raise ValueError("no paired return stable window exists")
    return best_stop
