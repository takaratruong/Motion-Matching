"""Scripted closed-loop gate for the released-PFNN native-G1 slice."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from mm_sonic.terrain_pfnn.pfnn_surface import load_placed_pfnn_surface
from mm_sonic.terrain_pfnn.runtime import TerrainSample
from mm_sonic.terrain_pfnn_viewer import _PFNNTerrainCallback, _load_runtime, _parser as viewer_parser


@dataclass(frozen=True)
class VerticalCommandSegment:
    name: str
    ticks: int
    command: tuple[float, float]

    def __post_init__(self) -> None:
        if self.name not in {
            "start",
            "straight",
            "left_turn",
            "right_turn",
            "ascent",
            "descent",
            "stop",
        }:
            raise ValueError("vertical command segment name is invalid")
        if type(self.ticks) is not int or self.ticks < 1:
            raise ValueError("vertical command segment ticks are invalid")
        command = np.asarray(self.command, dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError("vertical command segment command is invalid")


def vertical_slice_command_script() -> tuple[VerticalCommandSegment, ...]:
    """Return the fixed twenty-second start/turn/grade/stop route."""

    return (
        VerticalCommandSegment("start", 15, (0.0, 0.0)),
        VerticalCommandSegment("straight", 75, (0.45, 0.0)),
        VerticalCommandSegment("left_turn", 75, (0.35, 0.25)),
        VerticalCommandSegment("right_turn", 75, (0.35, -0.25)),
        VerticalCommandSegment("ascent", 150, (0.45, 0.0)),
        VerticalCommandSegment("descent", 150, (-0.45, 0.0)),
        VerticalCommandSegment("stop", 60, (0.0, 0.0)),
    )


def evaluate_script(
    runtime: object,
    terrain: object,
    script: tuple[VerticalCommandSegment, ...],
    *,
    minimum_grade_degrees: float = 3.0,
) -> dict[str, object]:
    """Run the fixed route and reject the first invalid controller tick."""

    if not callable(getattr(runtime, "step", None)) or not callable(terrain):
        raise TypeError("runtime and terrain callbacks are required")
    if (
        not script
        or not all(isinstance(segment, VerticalCommandSegment) for segment in script)
        or not math.isfinite(minimum_grade_degrees)
        or minimum_grade_degrees <= 0.0
    ):
        raise ValueError("vertical evaluation contract is invalid")
    previous_xy: np.ndarray | None = None
    maximum_grade = -math.inf
    minimum_grade = math.inf
    segment_displacement = {segment.name: 0.0 for segment in script}
    stop_speeds: list[float] = []
    tick = 0
    for segment in script:
        command = np.asarray(segment.command, dtype=np.float64)
        start_xy: np.ndarray | None = None
        final_xy: np.ndarray | None = None
        for _ in range(segment.ticks):
            frame = runtime.step(command, camera_yaw=0.0)
            diagnostics = getattr(frame, "diagnostics", None)
            if not isinstance(diagnostics, dict):
                raise ValueError(f"runtime diagnostics missing at tick {tick}")
            if "hold_reason" in diagnostics:
                raise ValueError(
                    f"hold_reason at tick {tick}: {diagnostics['hold_reason']}"
                )
            values = (
                np.asarray(getattr(frame, "root_position_world", ())),
                np.asarray(getattr(frame, "joint_position_isaaclab", ())),
                np.asarray(getattr(frame, "contact_probability", ())),
                np.asarray(getattr(frame, "phase", math.nan)),
            )
            if (
                values[0].shape != (3,)
                or values[1].shape != (29,)
                or values[2].shape != (4,)
                or not all(np.isfinite(value).all() for value in values)
            ):
                raise ValueError(f"nonfinite runtime frame at tick {tick}")
            xy = np.asarray(values[0][:2], dtype=np.float64)
            sample = terrain(xy)
            if not isinstance(sample, TerrainSample):
                raise ValueError(f"unsupported terrain at tick {tick}")
            if start_xy is None:
                start_xy = xy.copy()
            final_xy = xy.copy()
            if previous_xy is not None:
                delta = xy - previous_xy
                distance = float(np.linalg.norm(delta))
                if distance > 1.0e-8:
                    signed = sample.signed_grade_degrees(delta / distance)
                    maximum_grade = max(maximum_grade, signed)
                    minimum_grade = min(minimum_grade, signed)
            previous_xy = xy
            if segment.name == "stop":
                stop_speeds.append(
                    abs(float(diagnostics.get("realized_speed_m_s", math.inf)))
                )
            tick += 1
        assert start_xy is not None and final_xy is not None
        segment_displacement[segment.name] = float(np.linalg.norm(final_xy - start_xy))
    if maximum_grade < minimum_grade_degrees:
        raise ValueError("script did not realize the required ascent grade")
    if minimum_grade > -minimum_grade_degrees:
        raise ValueError("script did not realize the required descent grade")
    if not stop_speeds or max(stop_speeds[-30:]) > 0.05:
        raise ValueError("script did not realize the stop segment")
    return {
        "accepted": True,
        "ticks": tick,
        "maximum_signed_grade_degrees": maximum_grade,
        "minimum_signed_grade_degrees": minimum_grade,
        "segment_displacement_m": segment_displacement,
        "maximum_final_stop_speed_m_s": max(stop_speeds[-30:]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    defaults = viewer_parser().parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--terrain-fit", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    checkpoint = arguments.checkpoint.expanduser().resolve(strict=True)
    dataset = arguments.dataset.expanduser().resolve(strict=True)
    terrain_fit = arguments.terrain_fit.expanduser().resolve(strict=True)
    model_path = arguments.model_path.expanduser().resolve(strict=True)
    surface = load_placed_pfnn_surface(terrain_fit)
    terrain = _PFNNTerrainCallback(
        surface,
        x_samples=np.linspace(-6.0, 6.0, 321),
        y_samples=np.linspace(-4.0, 4.0, 161),
    )
    viewer_arguments = viewer_parser().parse_args(
        ["--device", arguments.device]
    )
    runtime = _load_runtime(
        viewer_arguments,
        checkpoint,
        dataset,
        model_path,
        terrain,
        strict=True,
    )
    result = evaluate_script(runtime, terrain, vertical_slice_command_script())
    receipt = {
        **result,
        "schema": "g1-pfnn-vertical-evaluation/v1",
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_manifest_sha256": _sha256(dataset),
        "terrain_fit_sha256": _sha256(terrain_fit),
    }
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    _atomic_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
