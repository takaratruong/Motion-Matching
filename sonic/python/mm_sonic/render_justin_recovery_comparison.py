"""Render full Justin dense-only and dense-to-tracker recovery rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .justin_recovery import (
    DEFAULT_S13_SCENE,
    RecoveryQuery,
    RecoverySceneContract,
    learner_state_in_reference_frame,
    load_recovery_trace,
)
from .render_justin_tracker_rollout import (
    ARTICULATION_JOINT_NAMES,
    DEFAULT_MODEL_PATH,
    _combined_usd_mesh,
)
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(proposal_path: Path, attempt: Path) -> tuple[dict, dict, dict]:
    proposal = json.loads(proposal_path.read_text())
    receipt_path = attempt / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("handoff_mode") != "exact_learner_state_and_history":
        raise ValueError("comparison requires an exact learner-history handoff")
    if receipt.get("proposal_sha256") != _sha256(proposal_path):
        raise ValueError("attempt receipt does not belong to this proposal")
    query_index = int(receipt["query_index"])
    candidate_index = int(receipt["candidate_index"])
    query = proposal["queries"][query_index]
    candidate = query["candidates"][candidate_index]
    if candidate != receipt["candidate"]:
        raise ValueError("attempt candidate differs from the proposal")
    return proposal, query, receipt


def _scene_contract(proposal: Mapping[str, object]) -> RecoverySceneContract:
    payload = proposal.get("scene_transform")
    if not isinstance(payload, Mapping):
        return DEFAULT_S13_SCENE
    return RecoverySceneContract(
        scene_id=str(payload["scene_id"]),
        learner_front_xy=tuple(map(float, payload["learner_front_xy"])),
        reference_front_xy=tuple(map(float, payload["reference_front_xy"])),
        learner_heading_yaw_rad=float(payload["learner_heading_yaw_rad"]),
        reference_heading_yaw_rad=float(payload["reference_heading_yaw_rad"]),
        tread_m=float(payload["tread_m"]),
        num_steps=int(payload["num_steps"]),
    )


def _learner_motion(
    proposal: Mapping[str, object], scene: RecoverySceneContract
) -> StitchedMotion:
    trace_path = Path(str(proposal["trace_path"])).resolve()
    trace = load_recovery_trace(trace_path)
    if trace.sha256 != proposal["trace_sha256"]:
        raise ValueError("learner trace changed after proposal generation")
    roots: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    for index, physics_step in enumerate(trace.arrays["physics_step"]):
        query = RecoveryQuery(
            rewind_s=0.0,
            trace_index=index,
            physics_step=int(physics_step),
            history_start=max(0, index - 19),
        )
        state = learner_state_in_reference_frame(trace, query, scene)
        roots.append(state["root_pose_wxyz"][:3])
        quaternions.append(state["root_pose_wxyz"][3:])
    frame_count = len(roots)
    return StitchedMotion(
        fps=float(trace.arrays["control_hz"]),
        root_position_world=np.asarray(roots, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternions, dtype=np.float32),
        joint_position=np.asarray(trace.arrays["joint_pos_isaac"], dtype=np.float32),
        provenance=tuple(
            FrameProvenance(0, frame, "dense") for frame in range(frame_count)
        ),
        seam_indices=(),
    )


def _recovered_motion(
    learner: StitchedMotion, query: Mapping[str, object], attempt: Path
) -> tuple[StitchedMotion, dict[str, float | int]]:
    handoff_frame = int(query["trace_index"])
    rollout_path = attempt / "tracker_rollout.npz"
    with np.load(rollout_path, allow_pickle=False) as data:
        tracker_root = np.asarray(data["root_pos"], dtype=np.float32)
        tracker_quat = np.asarray(data["root_quat_wxyz"], dtype=np.float32)
        tracker_joint = np.asarray(data["joint_pos"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if fps != learner.fps:
        raise ValueError("learner and tracker frame rates differ")
    if tracker_root.shape[0] == 0:
        raise ValueError("tracker rollout is empty")
    prefix_count = handoff_frame + 1
    root = np.concatenate((learner.root_position_world[:prefix_count], tracker_root), axis=0)
    quat = np.concatenate(
        (learner.root_quaternion_world_wxyz[:prefix_count], tracker_quat), axis=0
    )
    joints = np.concatenate(
        (learner.joint_position[:prefix_count], tracker_joint), axis=0
    )
    tracker_count = len(tracker_root)
    provenance = tuple(
        FrameProvenance(0, frame, "dense") for frame in range(prefix_count)
    ) + tuple(
        FrameProvenance(1, frame, "tracker") for frame in range(tracker_count)
    )
    position_jump = float(np.linalg.norm(tracker_root[0] - root[prefix_count - 1]))
    quaternion_dot = float(
        abs(np.dot(tracker_quat[0], quat[prefix_count - 1]))
    )
    quaternion_dot = min(max(quaternion_dot, 0.0), 1.0)
    return (
        StitchedMotion(
            fps=learner.fps,
            root_position_world=root,
            root_quaternion_world_wxyz=quat,
            joint_position=joints,
            provenance=provenance,
            seam_indices=(prefix_count,),
        ),
        {
            "handoff_frame": handoff_frame,
            "handoff_time_s": handoff_frame / learner.fps,
            "tracker_first_frame": prefix_count,
            "tracker_frames": tracker_count,
            "handoff_position_step_m": position_jump,
            "handoff_rotation_step_rad": 2.0 * math.acos(quaternion_dot),
        },
    )


def _camera_offset_for_absolute_azimuth(
    motion: StitchedMotion, absolute_azimuth_deg: float
) -> float:
    displacement = (
        motion.root_position_world[-1, :2] - motion.root_position_world[0, :2]
    )
    travel_yaw = math.degrees(math.atan2(float(displacement[1]), float(displacement[0])))
    return float(absolute_azimuth_deg) - travel_yaw


def render_comparison(
    *,
    proposal_path: Path,
    attempt: Path,
    before_output: Path,
    after_output: Path,
    metadata_output: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    width: int = 960,
    height: int = 540,
    frame_stride: int = 1,
) -> Path:
    proposal_path = proposal_path.resolve()
    attempt = attempt.resolve()
    proposal, query, receipt = _load_contract(proposal_path, attempt)
    scene = _scene_contract(proposal)
    learner = _learner_motion(proposal, scene)
    recovered, seam = _recovered_motion(learner, query, attempt)
    usd_paths = tuple((attempt / "tracker_bundle" / "object_usd").glob("*.usd"))
    if len(usd_paths) != 1:
        raise ValueError("attempt does not contain exactly one terrain USD")
    identity = RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )
    terrain = TerrainMeshIndex(_combined_usd_mesh(usd_paths[0]), identity)
    render_common = {
        "model_path": model_path,
        "target_mesh": terrain,
        "joint_names": ARTICULATION_JOINT_NAMES,
        "width": int(width),
        "height": int(height),
        "frame_stride": int(frame_stride),
        "camera_elevation_deg": -14.0,
        "camera_distance": 2.8,
        "camera_follow_root": True,
    }
    before_render = render_stitched_motion(
        learner,
        output_path=before_output,
        camera_azimuth_offset_deg=_camera_offset_for_absolute_azimuth(learner, 270.0),
        **render_common,
    )
    after_render = render_stitched_motion(
        recovered,
        output_path=after_output,
        camera_azimuth_offset_deg=_camera_offset_for_absolute_azimuth(recovered, 270.0),
        **render_common,
    )
    metadata = {
        "schema": "justin-recovery-whole-rollout-comparison/v1",
        "proposal": str(proposal_path),
        "attempt": str(attempt),
        "trace": str(proposal["trace_path"]),
        "scene": scene.to_json(),
        "query_index": int(receipt["query_index"]),
        "candidate_index": int(receipt["candidate_index"]),
        "rewind_s": float(receipt["rewind_s"]),
        "before": before_render,
        "after": after_render,
        "handoff": seam,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_output.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--before-out", type=Path, required=True)
    parser.add_argument("--after-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        render_comparison(
            proposal_path=args.proposal,
            attempt=args.attempt,
            before_output=args.before_out,
            after_output=args.after_out,
            metadata_output=args.metadata_out,
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            frame_stride=args.frame_stride,
        )
    )


if __name__ == "__main__":
    main()
