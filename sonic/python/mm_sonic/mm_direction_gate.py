"""Live Motion-Matching-only four-reset direction gate (Task 2/P1).

The pure scoring core scores one committed/prepared :class:`TargetChunk`
against a registered MuJoCo command axis using :func:`signed_root_projection`
from :mod:`mm_sonic.direction_probe`. It reuses the registered direction axes
and per-direction minimum projections; nothing here relaxes those floors.

The synthetic tests that exercise this module validate the gate/oracle only.
They are NOT evidence that live Motion Matching follows a command; the
controller runs the real four-reset live probe separately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .commands import CommandSample
from .direction_probe import (
    REGISTERED_DIRECTION_AXES_MUJOCO,
    REGISTERED_MINIMUM_PROJECTION_M,
    signed_root_projection,
)
from .joints import ContractError
from .transform import holden_to_mujoco_vectors


GATE_SCHEMA = "mm-direction-gate/v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Exactly these four directions must each be present once for a passing gate.
_REQUIRED_DIRECTIONS = frozenset(REGISTERED_DIRECTION_AXES_MUJOCO)
_DIRECTION_ORDER = tuple(REGISTERED_DIRECTION_AXES_MUJOCO)


def _registered_direction(direction: object) -> str:
    if type(direction) is not str or direction not in REGISTERED_DIRECTION_AXES_MUJOCO:
        raise ContractError(
            "direction must be one of forward/backward/left/right"
        )
    return direction


@dataclass(frozen=True)
class DirectionScore:
    """Immutable per-direction score of one generated target chunk."""

    direction: str
    requested_velocity_mujoco: tuple[float, float, float]
    root_displacement_mujoco: tuple[float, float, float]
    signed_projection_m: float
    orthogonal_projection_m: float
    minimum_projection_m: float
    passed: bool


@dataclass(frozen=True)
class DirectionResult:
    """One direction's score plus the live evidence that produced it."""

    score: DirectionScore
    reset_sha256: str
    candidate_id: str
    hello: Mapping[str, object]
    scene: Mapping[str, object]
    stdout_path: str
    stderr_path: str


def score_target_direction(direction: object, target: object) -> DirectionScore:
    """Score ``target`` against the registered axis for ``direction``.

    The signed projection uses :func:`signed_root_projection`; the orthogonal
    projection uses the in-plane axis rotated 90 degrees. A score passes when
    the signed projection reaches the registered minimum for the direction.
    """

    name = _registered_direction(direction)
    axis = REGISTERED_DIRECTION_AXES_MUJOCO[name]
    minimum = REGISTERED_MINIMUM_PROJECTION_M[name]

    signed = signed_root_projection(target, axis)
    orthogonal_axis = (-axis[1], axis[0], 0.0)
    orthogonal = signed_root_projection(target, orthogonal_axis)

    path = np.asarray(target.virtual_root_position, np.float64)
    displacement = path[-1] - path[0]
    displacement[2] = 0.0

    # The requested direction is already an explicit scorer input, so command
    # metadata is optional at this pure boundary. Live TargetChunks carry the
    # exact requested speed; controller-owned synthetic targets intentionally
    # contain only root kinematics.
    velocity = np.asarray(axis, np.float64)
    command = getattr(target, "command", None)
    if isinstance(command, Mapping) and "requested_velocity_holden" in command:
        velocity = np.asarray(
            holden_to_mujoco_vectors(command["requested_velocity_holden"]),
            np.float64,
        )

    return DirectionScore(
        direction=name,
        requested_velocity_mujoco=(
            float(velocity[0]),
            float(velocity[1]),
            float(velocity[2]),
        ),
        root_displacement_mujoco=(
            float(displacement[0]),
            float(displacement[1]),
            float(displacement[2]),
        ),
        signed_projection_m=float(signed),
        orthogonal_projection_m=float(orthogonal),
        minimum_projection_m=float(minimum),
        passed=bool(signed >= minimum),
    )


def _json_safe_mapping(value: object) -> dict[str, object]:
    """Recursively coerce a mapping to a JSON-serializable dict."""

    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"value of type {type(value)!r} is not JSON-safe")


def _score_to_json(entry: object, reset_sha256: str) -> dict[str, object]:
    if isinstance(entry, DirectionResult):
        score = entry.score
        evidence = {
            "candidate_id": entry.candidate_id,
            "hello": _json_safe_mapping(entry.hello),
            "scene": _json_safe_mapping(entry.scene),
            "stdout_path": entry.stdout_path,
            "stderr_path": entry.stderr_path,
        }
    else:
        score = entry
        evidence = {}
    return {
        "direction": score.direction,
        "requested_velocity_mujoco": list(score.requested_velocity_mujoco),
        "root_displacement_mujoco": list(score.root_displacement_mujoco),
        "signed_projection_m": score.signed_projection_m,
        "orthogonal_projection_m": score.orthogonal_projection_m,
        "minimum_projection_m": score.minimum_projection_m,
        "reset_sha256": reset_sha256,
        "passed": score.passed,
        **evidence,
    }


def _entry_score(entry: object) -> DirectionScore:
    return entry.score if isinstance(entry, DirectionResult) else entry


def direction_gate_summary(
    scores: object, reset_hashes: object
) -> dict[str, object]:
    """Assemble the JSON-safe ``mm-direction-gate/v1`` report.

    ``scores`` maps each direction to a :class:`DirectionScore` or the richer
    :class:`DirectionResult`. The gate passes only when exactly
    forward/backward/left/right are present once, every per-direction score
    passes, and all four lowercase SHA-256 reset hashes are identical.
    """

    if isinstance(scores, dict):
        if set(scores) != _REQUIRED_DIRECTIONS:
            raise ContractError(
                "direction gate requires exactly forward/backward/left/right scores"
            )
        score_slots = {direction: scores[direction] for direction in _DIRECTION_ORDER}
    elif isinstance(scores, (list, tuple)) and len(scores) == len(_DIRECTION_ORDER):
        score_slots = dict(zip(_DIRECTION_ORDER, scores, strict=True))
    else:
        raise ContractError(
            "direction gate requires exactly forward/backward/left/right scores"
        )

    if isinstance(reset_hashes, dict):
        if set(reset_hashes) != _REQUIRED_DIRECTIONS:
            raise ContractError(
                "direction gate requires one reset hash per registered direction"
            )
        reset_slots = {
            direction: reset_hashes[direction] for direction in _DIRECTION_ORDER
        }
    elif isinstance(reset_hashes, (list, tuple)) and len(reset_hashes) == len(
        _DIRECTION_ORDER
    ):
        reset_slots = dict(zip(_DIRECTION_ORDER, reset_hashes, strict=True))
    else:
        raise ContractError(
            "direction gate requires one reset hash per registered direction"
        )

    directions_match = True
    for direction, entry in score_slots.items():
        score = _entry_score(entry)
        if not isinstance(score, DirectionScore):
            raise ContractError(
                f"direction gate score for {direction} is malformed"
            )
        directions_match = directions_match and score.direction == direction
    digests = []
    for direction in _DIRECTION_ORDER:
        digest = reset_slots[direction]
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ContractError(
                "direction gate reset hashes must be lowercase SHA-256 digests"
            )
        digests.append(digest)

    same_reset = len(set(digests)) == 1
    all_passed = directions_match and all(
        _entry_score(entry).passed for entry in score_slots.values()
    )

    return {
        "schema": GATE_SCHEMA,
        "same_reset": same_reset,
        "passed": bool(same_reset and all_passed),
        "scores": {
            direction: _score_to_json(score_slots[direction], reset_slots[direction])
            for direction in _DIRECTION_ORDER
        },
    }


def direction_command(
    direction: object, speed_mps: object, *, chunk_index: int
) -> CommandSample:
    """A MuJoCo command that requests ``direction`` at ``speed_mps``.

    The velocity is the registered horizontal axis scaled by ``speed_mps`` with
    the identity heading (``wxyz = (1, 0, 0, 0)``).
    """

    name = _registered_direction(direction)
    if type(speed_mps) not in (int, float):
        raise ContractError("direction speed must be numeric")
    axis = REGISTERED_DIRECTION_AXES_MUJOCO[name]
    speed = float(speed_mps)
    return CommandSample(
        chunk_index=chunk_index,
        requested_velocity_mujoco=(
            axis[0] * speed,
            axis[1] * speed,
            axis[2] * speed,
        ),
        desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


# The exact ordered InitialBoundary arrays hashed to identify a reset. Names
# alone are excluded; the numeric state is what must be identical across the
# four resets.
_BOUNDARY_ARRAYS = (
    "joint_position_source",
    "joint_velocity_source",
    "physical_pelvis_position_holden",
    "physical_pelvis_orientation_holden",
    "virtual_root_position_holden",
    "virtual_root_orientation_holden",
)


def hash_initial_boundary(initial: object) -> str:
    """Deterministic lowercase SHA-256 over the complete validated boundary."""

    digest = hashlib.sha256()
    names = getattr(initial, "source_joint_names", None)
    if names is None:
        raise ContractError("initial boundary must expose source_joint_names")
    digest.update("\x00".join(names).encode("utf-8"))
    digest.update(b"\x00")
    for field in _BOUNDARY_ARRAYS:
        value = getattr(initial, field, None)
        if value is None:
            raise ContractError(f"initial boundary is missing {field}")
        array = np.ascontiguousarray(np.asarray(value, np.float64))
        digest.update(field.encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def run_live_direction(
    direction: object,
    *,
    client: object,
    validator: object,
    timeline_factory: object,
    config: object,
    session_id: str,
    speed_mps: float,
) -> DirectionResult:
    """Score one direction from a fresh reset, never committing anything.

    Ordering: hello, reset, validate the initial boundary, hash it, generate
    exactly one ``source_intervals=10`` candidate at chunk 0, validate and
    prepare it, score its target, then abort the uncommitted candidate (both
    timeline and MM) and close the client. The client is closed even on error.
    """

    name = _registered_direction(direction)
    candidate_id = f"{session_id}:candidate:000000"
    try:
        hello = client.hello()
        reset = client.reset(config, session_id=session_id)
        initial = validator.validate_initial(reset)
        reset_hash = hash_initial_boundary(initial)
        scene = reset["scene"] if isinstance(reset, Mapping) else {}
        timeline = timeline_factory(initial)
        command = direction_command(name, speed_mps, chunk_index=0)
        raw = client.generate(
            command,
            session_id=session_id,
            candidate_id=candidate_id,
            predecessor_id=None,
            source_intervals=10,
        )
        source = validator.validate_source(raw)
        prepared = timeline.prepare(source)
        score = score_target_direction(name, prepared.target)
        # Never reuse accepted state across directions: abort the uncommitted
        # candidate on both the timeline and MM before closing.
        timeline.abort(prepared.source_candidate_id)
        client.abort(candidate_id)
    finally:
        client.close()
    return DirectionResult(
        score=score,
        reset_sha256=reset_hash,
        candidate_id=candidate_id,
        hello=dict(hello) if isinstance(hello, Mapping) else {},
        scene=dict(scene) if isinstance(scene, Mapping) else {},
        stdout_path=str(getattr(client, "stdout_archive", "")),
        stderr_path=str(getattr(client, "stderr_archive", "")),
    )


def run_live_direction_gate(
    *,
    client_factory: object,
    validator_factory: object,
    timeline_factory: object,
    config: object,
    session_prefix: str,
    speed_mps: float,
) -> dict[str, object]:
    """Run the four-reset MM-only direction gate and assemble its report.

    Each direction gets a fresh isolated client and session; no accepted state
    is ever reused across directions.
    """

    results: dict[str, DirectionResult] = {}
    reset_hashes: dict[str, str] = {}
    for direction in sorted(_REQUIRED_DIRECTIONS):
        session_id = f"{session_prefix}:{direction}"
        client = client_factory(direction, session_id)
        result = run_live_direction(
            direction,
            client=client,
            validator=validator_factory(),
            timeline_factory=timeline_factory,
            config=config,
            session_id=session_id,
            speed_mps=speed_mps,
        )
        results[direction] = result
        reset_hashes[direction] = result.reset_sha256
    return direction_gate_summary(results, reset_hashes)


def write_gate_artifact(summary: Mapping[str, object], output_dir: str | Path) -> str:
    """Atomically write ``summary`` as JSON beneath ``output_dir``.

    Returns the absolute path of the written artifact.
    """

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "mm-direction-gate.json"
    payload = json.dumps(summary, indent=2, sort_keys=True).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=str(directory), prefix=".mm-direction-gate-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return str(final_path)


_DEFAULT_TERRAIN_DIR = "/home/ubuntu/projects/motion-matching/resources/g1_terrain"


def build_gate_parser() -> argparse.ArgumentParser:
    """The module CLI parser with flat-diagnostic production defaults."""

    parser = argparse.ArgumentParser(
        prog="python -m mm_sonic.mm_direction_gate",
        description=(
            "Run the live Motion-Matching-only four-reset direction gate."
        ),
    )
    parser.add_argument("--mm-server", required=True)
    parser.add_argument("--terrain-dir", default=_DEFAULT_TERRAIN_DIR)
    parser.add_argument("--output-dir", required=True)
    # Production defaults match manual_demo's current flat diagnostic path.
    parser.add_argument("--scene-id", default="sonic-flat-baseline")
    parser.add_argument("--route-id", default="flat-12s")
    parser.add_argument("--terrain-weight", type=float, default=0.0)
    parser.add_argument("--speed-mps", type=float, default=0.5)
    return parser


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SONIC_ROOT = _REPOSITORY_ROOT / "sonic"


def _production_run_gate(args: argparse.Namespace) -> dict[str, object]:
    """Wire the real MM client/validator/timeline factories and run the gate.

    The controller executes this path separately; the managed candidate never
    launches the live MM server here.
    """

    from .coordinator import SessionConfig, SourceValidator
    from .joints import load_joint_contract
    from .process import MMChunkClient
    from .timeline import TargetTimeline

    contract = load_joint_contract(_SONIC_ROOT / "configs/g1_joint_contract.json")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    python_root = str(_SONIC_ROOT / "python")
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_root if not prior else f"{python_root}{os.pathsep}{prior}"
    )
    environment["SONIC_TERRAIN_DIR"] = str(
        Path(args.terrain_dir).expanduser().resolve()
    )
    config = SessionConfig(args.scene_id, args.route_id, args.terrain_weight)

    def client_factory(direction: str, session_id: str) -> MMChunkClient:
        run_root = output_dir / f"mm-{direction}"
        run_root.mkdir(parents=True, exist_ok=True)
        return MMChunkClient(
            run_root=run_root,
            command=(str(Path(args.mm_server).expanduser()),),
            stdout_archive=run_root / "mm.stdout",
            stderr_archive=run_root / "mm.stderr",
            env=environment,
            cwd=_REPOSITORY_ROOT,
        )

    return run_live_direction_gate(
        client_factory=client_factory,
        validator_factory=lambda: SourceValidator(contract),
        timeline_factory=lambda initial: TargetTimeline(initial, contract),
        config=config,
        session_prefix="mm-direction-gate",
        speed_mps=args.speed_mps,
    )


def main(argv: list[str] | None = None, *, run_gate=None) -> int:
    """CLI entrypoint: run the gate, write the artifact atomically, print path.

    Exit code is 0 only when all four directions pass from the identical reset;
    evidence is preserved (written) regardless of the verdict.
    """

    args = build_gate_parser().parse_args(argv)
    runner = run_gate if run_gate is not None else _production_run_gate
    if run_gate is not None:
        summary = runner(
            config=_summary_config(args),
            speed_mps=args.speed_mps,
        )
    else:
        summary = runner(args)
    path = write_gate_artifact(summary, args.output_dir)
    print(path, flush=True)
    return 0 if summary.get("passed") is True else 1


def _summary_config(args: argparse.Namespace):
    from .coordinator import SessionConfig

    return SessionConfig(args.scene_id, args.route_id, args.terrain_weight)


if __name__ == "__main__":  # pragma: no cover - exercised by the controller
    raise SystemExit(main())
