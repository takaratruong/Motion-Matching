"""Deterministic, controller-shaped flat-locomotion command curriculum.

The curriculum is intentionally kinematic: it records what a human's two
sticks requested before the motion matcher or physical tracker sees it.  The
left stick controls planar travel and the right stick controls facing.
Centering the right stick holds the last requested facing direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .joints import ContractError


CURRICULUM_SCHEMA = "takara-human-command-curriculum/v2"
RAW_STICK_CHANNELS = ("left_x", "left_y", "right_x", "right_y")
BUTTON_CHANNELS = ("walk", "independent_facing", "stand")
COMMAND_EVENT_ABRUPT = 1
COMMAND_EVENT_STOP_OR_RESTART = 2
COMMAND_EVENT_VELOCITY_REVERSE = 4
COMMAND_EVENT_HEADING_JUMP = 8
COMMAND_EVENT_DUAL_STICK = 16


@dataclass(frozen=True)
class CommandTrace:
    trace_id: str
    category: str
    raw_sticks: np.ndarray
    buttons: np.ndarray
    requested_velocity_local_xy: np.ndarray
    requested_heading_world_yaw: np.ndarray
    seed: int | None

    def __post_init__(self) -> None:
        frames = int(self.raw_sticks.shape[0])
        expected = {
            "raw_sticks": (frames, 4),
            "buttons": (frames, 3),
            "requested_velocity_local_xy": (frames, 2),
            "requested_heading_world_yaw": (frames,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ContractError(
                    f"{self.trace_id}: {name} shape {value.shape}, expected {shape}"
                )
            if not np.isfinite(value).all():
                raise ContractError(f"{self.trace_id}: {name} contains NaN/Inf")
        if np.max(np.abs(self.raw_sticks), initial=0.0) > 1.0 + 1e-6:
            raise ContractError(f"{self.trace_id}: raw stick outside [-1, 1]")

    @property
    def frame_count(self) -> int:
        return int(self.raw_sticks.shape[0])


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _piecewise(
    frames: int,
    knots: list[tuple[int, tuple[float, float, float, float]]],
) -> np.ndarray:
    if frames <= 0 or len(knots) < 2 or knots[0][0] != 0 or knots[-1][0] != frames - 1:
        raise ContractError("piecewise stick trace requires endpoint knots")
    out = np.zeros((frames, 4), dtype=np.float32)
    for (start, left), (stop, right) in zip(knots[:-1], knots[1:], strict=True):
        if stop <= start:
            raise ContractError("piecewise stick knot frames must increase")
        alpha = _smoothstep(
            np.linspace(0.0, 1.0, stop - start + 1, dtype=np.float32)
        )
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        out[start : stop + 1] = a + alpha[:, None] * (b - a)
    return np.clip(out, -1.0, 1.0)


def _direction_sticks(
    travel_angle: float,
    travel_magnitude: float,
    facing_angle: float | None,
) -> tuple[float, float, float, float]:
    """Invert the controller mapping for world-frame travel/facing directions."""
    forward = float(travel_magnitude) * math.cos(travel_angle)
    lateral = float(travel_magnitude) * math.sin(travel_angle)
    left_x = -lateral
    left_y = -forward
    if facing_angle is None:
        return left_x, left_y, 0.0, 0.0
    return (
        left_x,
        left_y,
        -math.sin(facing_angle),
        -math.cos(facing_angle),
    )


def _radial_stick(x: float, y: float, deadzone: float = 0.2) -> tuple[float, float]:
    magnitude = math.hypot(x, y)
    if magnitude <= deadzone:
        return 0.0, 0.0
    shaped = min(1.0, magnitude * magnitude)
    return x * shaped / magnitude, y * shaped / magnitude


def map_two_stick_commands(
    raw_sticks: np.ndarray,
    buttons: np.ndarray,
    *,
    initial_heading_yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map sticks to robot-local velocity and command-frame facing.

    The heading frame is anchored to the robot at reset. The velocity is
    intentionally *not* world-frame: the corpus generator rotates it by the
    matcher's current root yaw at every frame.
    """
    raw = np.asarray(raw_sticks, dtype=np.float32)
    flags = np.asarray(buttons, dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ContractError("raw_sticks must have shape (T,4)")
    if flags.shape != (len(raw), 3):
        raise ContractError("buttons must have shape (T,3)")
    velocities = np.zeros((len(raw), 2), dtype=np.float32)
    headings = np.zeros(len(raw), dtype=np.float32)
    heading = float(initial_heading_yaw)
    for frame, (left_x, left_y, right_x, right_y) in enumerate(raw):
        lx, ly = _radial_stick(float(left_x), float(left_y))
        rx, ry = _radial_stick(float(right_x), float(right_y))
        forward = -ly
        lateral = -lx
        forward_scale = 0.72 if forward >= 0.0 else 0.55
        velocities[frame] = (
            forward * forward_scale,
            lateral * 0.55,
        )
        if math.hypot(rx, ry) > 0.01:
            heading = math.atan2(-rx, -ry)
        if flags[frame, 2]:
            velocities[frame] = 0.0
        headings[frame] = heading
    return velocities, headings


def command_event_mask(raw_sticks: np.ndarray) -> np.ndarray:
    """Label large one-frame joystick events for balanced distillation.

    The bit mask survives exact sagittal mirroring.  It lets a downstream
    sampler deliberately retain windows around rare harsh commands instead of
    hoping uniform frame sampling sees enough of them.
    """

    raw = np.asarray(raw_sticks, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ContractError("raw_sticks must be finite with shape (T,4)")
    mask = np.zeros(len(raw), dtype=np.uint8)
    if len(raw) < 2:
        return mask
    previous = raw[:-1]
    current = raw[1:]
    left_delta = np.linalg.norm(current[:, :2] - previous[:, :2], axis=1)
    right_delta = np.linalg.norm(current[:, 2:] - previous[:, 2:], axis=1)
    abrupt = np.hypot(left_delta, right_delta) > 0.25
    row = np.zeros(len(raw) - 1, dtype=np.uint8)
    row[abrupt] |= COMMAND_EVENT_ABRUPT

    previous_left_norm = np.linalg.norm(previous[:, :2], axis=1)
    current_left_norm = np.linalg.norm(current[:, :2], axis=1)
    stopped_before = previous_left_norm <= 0.20
    stopped_now = current_left_norm <= 0.20
    row[abrupt & (stopped_before != stopped_now)] |= COMMAND_EVENT_STOP_OR_RESTART

    moving = (previous_left_norm > 0.20) & (current_left_norm > 0.20)
    left_cosine = np.ones(len(row), dtype=np.float32)
    left_cosine[moving] = np.sum(
        previous[moving, :2] * current[moving, :2], axis=1
    ) / (previous_left_norm[moving] * current_left_norm[moving])
    row[abrupt & moving & (left_cosine <= -0.70)] |= (
        COMMAND_EVENT_VELOCITY_REVERSE
    )

    previous_right_norm = np.linalg.norm(previous[:, 2:], axis=1)
    current_right_norm = np.linalg.norm(current[:, 2:], axis=1)
    facing = (previous_right_norm > 0.20) & (current_right_norm > 0.20)
    right_cosine = np.ones(len(row), dtype=np.float32)
    right_cosine[facing] = np.sum(
        previous[facing, 2:] * current[facing, 2:], axis=1
    ) / (previous_right_norm[facing] * current_right_norm[facing])
    row[abrupt & facing & (right_cosine <= 1.0e-6)] |= (
        COMMAND_EVENT_HEADING_JUMP
    )
    row[abrupt & (left_delta > 0.25) & (right_delta > 0.25)] |= (
        COMMAND_EVENT_DUAL_STICK
    )
    mask[1:] = row
    return mask


def _trace(
    trace_id: str,
    category: str,
    raw: np.ndarray,
    *,
    seed: int | None = None,
    stand: np.ndarray | None = None,
) -> CommandTrace:
    frames = len(raw)
    buttons = np.ones((frames, 3), dtype=np.uint8)
    buttons[:, 2] = 0
    if stand is not None:
        buttons[:, 2] = np.asarray(stand, dtype=np.uint8)
    velocity, heading = map_two_stick_commands(raw, buttons)
    return CommandTrace(
        trace_id=trace_id,
        category=category,
        raw_sticks=np.ascontiguousarray(raw, dtype=np.float32),
        buttons=buttons,
        requested_velocity_local_xy=velocity,
        requested_heading_world_yaw=heading,
        seed=seed,
    )


def _steady_traces(frames: int) -> list[CommandTrace]:
    definitions = (
        ("forward_slow", 0.0, 0.55, 0.0),
        ("forward_fast", 0.0, 0.95, 0.0),
        ("backward_slow", math.pi, 0.55, 0.0),
        ("backward_fast", math.pi, 0.90, 0.0),
        ("left_strafe", math.pi / 2, 0.75, 0.0),
        ("right_strafe", -math.pi / 2, 0.75, 0.0),
        ("diagonal_left", math.pi / 4, 0.80, math.pi / 4),
        ("diagonal_right", -math.pi / 4, 0.80, -math.pi / 4),
    )
    traces = []
    ramp = max(10, frames // 20)
    for name, travel, magnitude, facing in definitions:
        command = _direction_sticks(travel, magnitude, facing)
        raw = _piecewise(
            frames,
            [(0, (0.0, 0.0, 0.0, -1.0)), (ramp, command), (frames - 1, command)],
        )
        traces.append(_trace(name, "steady", raw))
    return traces


def _transition_traces(frames: int) -> list[CommandTrace]:
    definitions = (
        ("start_stop_forward", (0.0, 0.8, 0.0), (0.0, 0.0, 0.0)),
        ("start_stop_backward", (math.pi, 0.75, 0.0), (0.0, 0.0, 0.0)),
        ("forward_to_backward", (0.0, 0.8, 0.0), (math.pi, 0.8, 0.0)),
        ("backward_to_forward", (math.pi, 0.8, 0.0), (0.0, 0.8, 0.0)),
        ("left_to_right", (math.pi / 2, 0.8, 0.0), (-math.pi / 2, 0.8, 0.0)),
        ("right_to_left", (-math.pi / 2, 0.8, 0.0), (math.pi / 2, 0.8, 0.0)),
        ("forward_to_left", (0.0, 0.8, 0.0), (math.pi / 2, 0.8, math.pi / 2)),
        ("forward_to_right", (0.0, 0.8, 0.0), (-math.pi / 2, 0.8, -math.pi / 2)),
        ("diagonal_toggle_left", (math.pi / 4, 0.8, math.pi / 4), (-math.pi / 4, 0.8, -math.pi / 4)),
        ("speed_toggle", (0.0, 0.35, 0.0), (0.0, 1.0, 0.0)),
        ("walk_stop_walk", (0.0, 0.8, 0.0), (0.0, 0.0, 0.0)),
        ("four_way_toggle", (0.0, 0.8, 0.0), (math.pi, 0.8, math.pi)),
    )
    traces = []
    q1, q2, q3 = frames // 4, frames // 2, 3 * frames // 4
    for name, first, second in definitions:
        a = _direction_sticks(*first)
        b = _direction_sticks(*second)
        if name == "walk_stop_walk":
            knots = [(0, a), (q1, a), (q2, b), (q3, a), (frames - 1, a)]
        elif name == "four_way_toggle":
            left = _direction_sticks(math.pi / 2, 0.8, math.pi / 2)
            right = _direction_sticks(-math.pi / 2, 0.8, -math.pi / 2)
            knots = [(0, a), (q1, left), (q2, b), (q3, right), (frames - 1, a)]
        else:
            knots = [(0, a), (q2 - 15, a), (q2 + 15, b), (frames - 1, b)]
        traces.append(_trace(name, "transition", _piecewise(frames, knots)))
    return traces


def _curve_traces(frames: int) -> list[CommandTrace]:
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    definitions = (
        ("arc_left_gentle", 0.25, 0.65),
        ("arc_right_gentle", -0.25, 0.65),
        ("arc_left_tight", 0.50, 0.80),
        ("arc_right_tight", -0.50, 0.80),
        ("circle_left", 1.0, 1.0),
        ("circle_right", -1.0, 1.0),
        ("slalom_slow", 0.30, 2.0),
        ("slalom_fast", 0.45, 4.0),
        ("s_curve_left", 0.50, 1.0),
        ("s_curve_right", -0.50, 1.0),
        ("zigzag_smooth", 0.65, 3.0),
        ("zigzag_quick", 0.75, 6.0),
    )
    traces = []
    for name, amplitude, cycles in definitions:
        if "circle" in name:
            angle = np.sign(amplitude) * 2.0 * np.pi * t
        elif "arc" in name:
            angle = amplitude * np.pi * t
        else:
            angle = amplitude * np.sin(2.0 * np.pi * cycles * t)
        raw = np.stack(
            [
                -0.80 * np.sin(angle),
                -0.80 * np.cos(angle),
                -np.sin(angle),
                -np.cos(angle),
            ],
            axis=1,
        ).astype(np.float32)
        traces.append(_trace(name, "curve", raw))
    return traces


def _independent_facing_traces(frames: int) -> list[CommandTrace]:
    definitions = (
        ("crab_left_face_forward", math.pi / 2, 0.75, 0.0),
        ("crab_right_face_forward", -math.pi / 2, 0.75, 0.0),
        ("backward_face_forward", math.pi, 0.75, 0.0),
        ("forward_face_left", 0.0, 0.75, math.pi / 2),
        ("forward_face_right", 0.0, 0.75, -math.pi / 2),
        ("diagonal_face_forward", math.pi / 4, 0.75, 0.0),
    )
    traces = []
    for name, travel, magnitude, facing in definitions:
        raw = np.broadcast_to(
            np.asarray(_direction_sticks(travel, magnitude, facing), dtype=np.float32),
            (frames, 4),
        ).copy()
        traces.append(_trace(name, "independent_facing", raw))

    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for name, direction in (("spin_left", 1.0), ("spin_right", -1.0)):
        yaw = direction * 2.0 * np.pi * t
        raw = np.stack(
            [
                np.zeros(frames),
                np.zeros(frames),
                -np.sin(yaw),
                -np.cos(yaw),
            ],
            axis=1,
        ).astype(np.float32)
        traces.append(_trace(name, "independent_facing", raw))
    for name, direction in (
        ("turn_then_walk_left", math.pi / 2),
        ("turn_then_walk_right", -math.pi / 2),
    ):
        q = frames // 2
        a = _direction_sticks(0.0, 0.0, 0.0)
        b = _direction_sticks(0.0, 0.0, direction)
        c = _direction_sticks(direction, 0.75, direction)
        raw = _piecewise(frames, [(0, a), (q - 20, b), (q + 20, c), (frames - 1, c)])
        traces.append(_trace(name, "independent_facing", raw))
    return traces


def _random_trace(frames: int, index: int) -> CommandTrace:
    seed = 10_000 + index
    rng = np.random.default_rng(seed)
    knot_frames = [0]
    while knot_frames[-1] < frames - 1:
        knot_frames.append(min(frames - 1, knot_frames[-1] + int(rng.integers(35, 91))))
    knots: list[tuple[int, tuple[float, float, float, float]]] = []
    held_facing = 0.0
    for frame in knot_frames:
        if rng.random() < 0.16:
            magnitude = 0.0
        else:
            magnitude = float(rng.uniform(0.30, 0.95))
        travel = float(rng.uniform(-math.pi, math.pi))
        if rng.random() < 0.28:
            facing = None
        elif rng.random() < 0.62:
            facing = travel
            held_facing = facing
        else:
            held_facing = float(
                np.clip(
                    held_facing + rng.normal(0.0, 0.75),
                    -math.pi,
                    math.pi,
                )
            )
            facing = held_facing
        knots.append((frame, _direction_sticks(travel, magnitude, facing)))
    return _trace(
        f"human_random_{index:02d}",
        "human_random",
        _piecewise(frames, knots),
        seed=seed,
    )


def build_base_curriculum(*, frames: int = 600) -> tuple[CommandTrace, ...]:
    """Return 60 deterministic 50 Hz traces (12 seconds by default)."""
    if frames < 100:
        raise ContractError("curriculum clips need at least 100 frames")
    traces = (
        _steady_traces(frames)
        + _transition_traces(frames)
        + _curve_traces(frames)
        + _independent_facing_traces(frames)
        + [_random_trace(frames, index) for index in range(18)]
    )
    if len(traces) != 60:
        raise AssertionError(f"curriculum count changed: {len(traces)}")
    ids = [trace.trace_id for trace in traces]
    if len(ids) != len(set(ids)):
        raise AssertionError("curriculum trace IDs are not unique")
    return tuple(traces)


def _canonical_omnidirectional_states() -> tuple[tuple[int, int], ...]:
    """Return one representative from each sagittal-mirror state orbit."""
    states = []
    for travel_bin in range(8):
        for heading_bin in range(8):
            state = (travel_bin, heading_bin)
            mirror = ((-travel_bin) % 8, (-heading_bin) % 8)
            if state <= mirror:
                states.append(state)
    if len(states) != 34:
        raise AssertionError(f"unexpected omnidirectional orbit count: {len(states)}")
    return tuple(states)


def _omni_state(
    travel_bin: int,
    heading_bin: int,
    *,
    magnitude: float = 0.70,
) -> tuple[float, float, float, float]:
    spacing = math.pi / 4.0
    return _direction_sticks(
        (travel_bin % 8) * spacing,
        magnitude,
        (heading_bin % 8) * spacing,
    )


def _scaled_knots(
    frames: int,
    states: tuple[tuple[float, float, float, float], ...],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Give six command states human-like holds and smooth 0.6 s transitions."""
    # Keep the final state long enough to observe the post-restart gait.  The
    # previous schedule spent only 0.72 s there in a 12 s clip, which made a
    # successful restart difficult to distinguish from a transient.
    fractions = (0.0, 0.08, 0.13, 0.23, 0.28, 0.38, 0.43, 0.53, 0.58, 0.72, 0.79, 1.0)
    if len(states) != 6:
        raise ContractError("omnidirectional transition trace needs six states")
    indices = [int(round(value * (frames - 1))) for value in fractions]
    if any(right <= left for left, right in zip(indices[:-1], indices[1:])):
        raise ContractError("curriculum clip is too short for transition schedule")
    knots = []
    for index, state in enumerate(states):
        knots.extend(((indices[2 * index], state), (indices[2 * index + 1], state)))
    return knots


def build_omnidirectional_curriculum(
    *,
    frames: int = 600,
    steady_magnitudes: tuple[float, ...] = (0.70,),
) -> tuple[CommandTrace, ...]:
    """Balanced two-stick state grid plus independent, smoothed transitions.

    With the default steady magnitude, the returned 102 base traces become
    204 physical clips after the corpus writer adds exact sagittal mirrors.
    Each requested steady magnitude adds a complete 34-orbit steady grid; the
    grids and their mirrors cover every one of the 8 travel directions crossed
    with every one of the 8 absolute/reset-frame headings.  Two transition
    variants per canonical state independently change travel, heading, both
    sticks, and stop/resume without introducing discontinuous raw-stick jumps.
    """
    if frames < 200:
        raise ContractError("omnidirectional curriculum needs at least 200 frames")
    if not steady_magnitudes or any(
        not math.isfinite(value) or value <= 0.06 or value > 1.0
        for value in steady_magnitudes
    ):
        raise ContractError("steady magnitudes must be finite walking inputs in (0.06, 1]")
    if len(set(steady_magnitudes)) != len(steady_magnitudes):
        raise ContractError("steady magnitudes must be unique")
    traces: list[CommandTrace] = []
    start = _omni_state(0, 0, magnitude=0.0)
    ramp = max(15, frames // 10)
    canonical = _canonical_omnidirectional_states()
    for magnitude in steady_magnitudes:
        magnitude_tag = int(round(100.0 * magnitude))
        for travel_bin, heading_bin in canonical:
            target = _omni_state(travel_bin, heading_bin, magnitude=magnitude)
            raw = _piecewise(
                frames,
                [(0, start), (ramp, target), (frames - 1, target)],
            )
            trace_id = f"omni_steady_t{travel_bin}_h{heading_bin}"
            if steady_magnitudes != (0.70,):
                trace_id = f"omni_steady_s{magnitude_tag:02d}_t{travel_bin}_h{heading_bin}"
            traces.append(_trace(trace_id, "omni_steady", raw))

    for travel_bin, heading_bin in canonical:
        source = _omni_state(travel_bin, heading_bin)
        variants = (
            (
                _omni_state(travel_bin + 2, heading_bin),
                _omni_state(travel_bin + 2, heading_bin + 2),
                _omni_state(travel_bin - 1, heading_bin - 2),
                _omni_state(travel_bin - 1, heading_bin - 2, magnitude=0.0),
                _omni_state(travel_bin, heading_bin),
            ),
            (
                _omni_state(travel_bin + 4, heading_bin),
                _omni_state(travel_bin + 4, heading_bin + 4),
                _omni_state(travel_bin + 1, heading_bin - 1),
                _omni_state(travel_bin + 1, heading_bin - 1, magnitude=0.0),
                _omni_state(travel_bin - 2, heading_bin + 2),
            ),
        )
        for variant, destinations in enumerate(variants):
            raw = _piecewise(
                frames,
                _scaled_knots(frames, (source, *destinations)),
            )
            traces.append(
                _trace(
                    f"omni_transition_v{variant}_t{travel_bin}_h{heading_bin}",
                    "omni_transition",
                    raw,
                )
            )
    expected = 68 + 34 * len(steady_magnitudes)
    if len(traces) != expected:
        raise AssertionError(f"omnidirectional trace count changed: {len(traces)}")
    ids = [trace.trace_id for trace in traces]
    if len(ids) != len(set(ids)):
        raise AssertionError("omnidirectional trace IDs are not unique")
    return tuple(traces)


def _held_step_trace(
    frames: int,
    states: tuple[tuple[float, float, float, float], ...],
    *,
    phase_offset: int,
) -> np.ndarray:
    """Hold commands and change them in one frame at varied gait phases."""

    if len(states) != 5:
        raise ContractError("abrupt command trace needs five held states")
    # Co-prime-ish holds avoid repeatedly landing command changes on the same
    # side of a nominal walking cycle.  The small deterministic offset varies
    # the first event across the 34 canonical command states.
    fractions = np.asarray((0.0, 0.18, 0.39, 0.61, 0.80, 1.0))
    boundaries = np.rint(fractions * frames).astype(np.int64)
    shift = int(phase_offset) % max(1, frames // 30)
    boundaries[1:-1] += shift
    boundaries = np.clip(boundaries, 0, frames)
    if np.any(np.diff(boundaries) <= 0):
        raise ContractError("abrupt curriculum clip is too short")
    raw = np.empty((frames, 4), dtype=np.float32)
    for state, start, stop in zip(
        states, boundaries[:-1], boundaries[1:], strict=True
    ):
        raw[int(start) : int(stop)] = np.asarray(state, dtype=np.float32)
    return raw


def build_abrupt_curriculum(*, frames: int = 600) -> tuple[CommandTrace, ...]:
    """Balanced one-frame two-stick changes with safe realized transitions.

    The command deliberately jumps; the motion matcher/controller remains
    responsible for producing a smooth body response.  Three traces per
    canonical sagittal orbit cover translation reversal, simultaneous travel
    and facing changes, stopping/restarting, speed toggles, and yaw reversals.
    The corpus writer adds exact mirrors, yielding 204 physical clips by
    default without relying on left/right sampling luck.
    """

    if frames < 200:
        raise ContractError("abrupt curriculum needs at least 200 frames")
    traces: list[CommandTrace] = []
    for index, (travel_bin, heading_bin) in enumerate(
        _canonical_omnidirectional_states()
    ):
        source = _omni_state(travel_bin, heading_bin, magnitude=0.70)
        reverse = _omni_state(travel_bin + 4, heading_bin, magnitude=0.70)
        left = _omni_state(travel_bin + 2, heading_bin, magnitude=0.70)
        right = _omni_state(travel_bin - 2, heading_bin, magnitude=0.70)
        stopped = _omni_state(travel_bin, heading_bin, magnitude=0.0)
        dual_left = _omni_state(
            travel_bin + 2, heading_bin + 2, magnitude=0.80
        )
        dual_reverse = _omni_state(
            travel_bin + 4, heading_bin + 4, magnitude=0.80
        )
        dual_right = _omni_state(
            travel_bin - 2, heading_bin - 2, magnitude=0.80
        )
        slow = _omni_state(travel_bin, heading_bin, magnitude=0.32)
        fast = _omni_state(travel_bin, heading_bin, magnitude=1.0)
        yaw_reverse = _omni_state(
            travel_bin, heading_bin + 4, magnitude=0.70
        )
        definitions = (
            (
                "velocity",
                (source, reverse, stopped, left, right),
            ),
            (
                "dual_stick",
                (source, dual_left, dual_reverse, stopped, dual_right),
            ),
            (
                "stop_speed_yaw",
                (slow, fast, stopped, yaw_reverse, source),
            ),
        )
        for variant, states in definitions:
            raw = _held_step_trace(
                frames,
                states,
                phase_offset=7 * index + 3 * len(traces),
            )
            traces.append(
                _trace(
                    f"abrupt_{variant}_t{travel_bin}_h{heading_bin}",
                    f"abrupt_{variant}",
                    raw,
                )
            )
    if len(traces) != 102:
        raise AssertionError(f"abrupt curriculum count changed: {len(traces)}")
    if len({trace.trace_id for trace in traces}) != len(traces):
        raise AssertionError("abrupt trace IDs are not unique")
    return tuple(traces)


def mirror_command_trace(trace: CommandTrace) -> CommandTrace:
    """Return the exact sagittal command mirror of one trace."""
    raw = trace.raw_sticks.copy()
    raw[:, 0] *= -1.0
    raw[:, 2] *= -1.0
    velocity = trace.requested_velocity_local_xy.copy()
    velocity[:, 1] *= -1.0
    heading = -trace.requested_heading_world_yaw
    return CommandTrace(
        trace_id=f"{trace.trace_id}__mirror",
        category=trace.category,
        raw_sticks=raw,
        buttons=trace.buttons.copy(),
        requested_velocity_local_xy=velocity,
        requested_heading_world_yaw=heading.astype(np.float32),
        seed=trace.seed,
    )


def build_paired_curriculum(*, frames: int = 600) -> tuple[CommandTrace, ...]:
    """Return each base trace followed immediately by its exact command mirror."""
    paired: list[CommandTrace] = []
    for trace in build_base_curriculum(frames=frames):
        paired.extend((trace, mirror_command_trace(trace)))
    return tuple(paired)


__all__ = [
    "BUTTON_CHANNELS",
    "COMMAND_EVENT_ABRUPT",
    "COMMAND_EVENT_DUAL_STICK",
    "COMMAND_EVENT_HEADING_JUMP",
    "COMMAND_EVENT_STOP_OR_RESTART",
    "COMMAND_EVENT_VELOCITY_REVERSE",
    "CURRICULUM_SCHEMA",
    "CommandTrace",
    "RAW_STICK_CHANNELS",
    "build_abrupt_curriculum",
    "build_base_curriculum",
    "build_omnidirectional_curriculum",
    "build_paired_curriculum",
    "command_event_mask",
    "map_two_stick_commands",
    "mirror_command_trace",
]
