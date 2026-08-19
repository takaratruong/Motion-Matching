"""Select the safest accepted expert block with the intended traversal speed.

Geometry composition historically ranked penetration and warp cost only.  For
recovery this can prefer a technically safe clip that stalls on a tread.  All
inputs to this selector already passed the exact full-body audit; this stage
adds route completion, backward-motion, lateral, and commanded-speed quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .compose_coherent_block_plan import (
    _mechanically_accepted,
    _maximum_steps,
    _save_motion,
)
from .generate_generic_stair_route import load_target_mesh
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_oracle.math3d import quaternion_multiply_wxyz
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = map(float, quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _extend_source_prefix(
    *,
    selected_path: Path,
    terrain_usd: Path,
    output_path: Path,
    archive_path: Path = DEFAULT_STAIRS_ARCHIVE,
    model_path: Path = DEFAULT_G1_MJCF,
) -> dict[str, object]:
    """Recover unused source-contiguous context around the stair block."""

    import zarr

    motion = load_stitched_motion_npz(selected_path)
    first = motion.provenance[0]
    available = int(first.source_frame)
    receipt: dict[str, object] = {
        "available_source_prefix_frames": available,
        "selected_motion": str(selected_path),
        "extended_motion": None,
        "accepted_prefix_frames": 0,
        "accepted_suffix_frames": 0,
    }
    archive = zarr.open_group(str(archive_path), mode="r")
    clip = int(first.archive_clip_index)
    global_start = int(archive["clip_start_idx"][clip])
    target_mesh = load_target_mesh(terrain_usd)
    attempts = []
    context_motion = motion
    for count in sorted(set((available, min(60, available), min(40, available), min(20, available))), reverse=True):
        if count <= 0:
            continue
        anchor_global = global_start + available
        source_anchor_position = np.asarray(
            archive["body_pos_w"][anchor_global, 0], dtype=np.float64
        )
        source_anchor_quaternion = np.asarray(
            archive["body_quat_w"][anchor_global, 0], dtype=np.float64
        )[[3, 0, 1, 2]]
        target_anchor_position = np.asarray(
            motion.root_position_world[0], dtype=np.float64
        )
        target_anchor_quaternion = np.asarray(
            motion.root_quaternion_world_wxyz[0], dtype=np.float64
        )
        yaw_delta = _yaw(target_anchor_quaternion) - _yaw(
            source_anchor_quaternion
        )
        cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
        rotation = np.asarray(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        half = 0.5 * yaw_delta
        yaw_quaternion = np.asarray(
            (np.cos(half), 0.0, 0.0, np.sin(half)), dtype=np.float64
        )
        local_start = available - count
        source_slice = slice(global_start + local_start, anchor_global)
        root = np.asarray(archive["body_pos_w"][source_slice, 0], dtype=np.float64)
        relative_xy = root[:, :2] - source_anchor_position[:2]
        root[:, :2] = target_anchor_position[:2] + relative_xy @ rotation.T
        root[:, 2] += target_anchor_position[2] - source_anchor_position[2]
        source_quaternion = np.asarray(
            archive["body_quat_w"][source_slice, 0], dtype=np.float64
        )[..., (3, 0, 1, 2)]
        quaternion = quaternion_multiply_wxyz(
            yaw_quaternion[None], source_quaternion
        )
        joints = np.asarray(archive["joint_pos"][source_slice], dtype=np.float32)
        prefix_provenance = tuple(
            FrameProvenance(clip, frame, str(first.clip_id))
            for frame in range(local_start, available)
        )
        extended = StitchedMotion(
            fps=motion.fps,
            root_position_world=np.concatenate(
                (root.astype(np.float32), motion.root_position_world), axis=0
            ),
            root_quaternion_world_wxyz=np.concatenate(
                (np.asarray(quaternion, dtype=np.float32), motion.root_quaternion_world_wxyz),
                axis=0,
            ),
            joint_position=np.concatenate((joints, motion.joint_position), axis=0),
            provenance=prefix_provenance + motion.provenance,
            seam_indices=(count,) + tuple(count + value for value in motion.seam_indices),
        )
        mechanics = _maximum_steps(extended)
        audit = audit_stair_motion_collisions(
            extended,
            archive_path=archive_path,
            target_mesh=target_mesh,
            model_path=model_path,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=1.0e-6,
        )
        attempt = {
            "prefix_frames": count,
            "mechanics": mechanics,
            "mechanics_accepted": _mechanically_accepted(mechanics),
            "full_body_collision_audit": {
                "accepted": bool(audit.accepted),
                "maximum_foot_penetration_m": float(
                    audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    audit.maximum_forbidden_body_penetration_m
                ),
            },
        }
        attempts.append(attempt)
        if attempt["mechanics_accepted"] and audit.accepted:
            _save_motion(output_path, extended)
            receipt.update(
                extended_motion=str(output_path.resolve()),
                extended_motion_sha256=_sha256(output_path),
                accepted_prefix_frames=count,
                extension_audit=attempt,
            )
            context_motion = extended
            break
    receipt["attempts"] = attempts

    last = motion.provenance[-1]
    suffix_clip = int(last.archive_clip_index)
    suffix_global_start = int(archive["clip_start_idx"][suffix_clip])
    suffix_clip_frames = int(
        archive["clip_end_idx"][suffix_clip]
        - archive["clip_start_idx"][suffix_clip]
    )
    suffix_available = max(0, suffix_clip_frames - 1 - int(last.source_frame))
    receipt["available_source_suffix_frames"] = suffix_available
    suffix_attempts = []
    for count in sorted(
        set(
            (
                suffix_available,
                min(150, suffix_available),
                min(100, suffix_available),
                min(60, suffix_available),
                min(40, suffix_available),
                min(20, suffix_available),
            )
        ),
        reverse=True,
    ):
        if count <= 0:
            continue
        anchor_global = suffix_global_start + int(last.source_frame)
        source_anchor_position = np.asarray(
            archive["body_pos_w"][anchor_global, 0], dtype=np.float64
        )
        source_anchor_quaternion = np.asarray(
            archive["body_quat_w"][anchor_global, 0], dtype=np.float64
        )[[3, 0, 1, 2]]
        target_anchor_position = np.asarray(
            context_motion.root_position_world[-1], dtype=np.float64
        )
        target_anchor_quaternion = np.asarray(
            context_motion.root_quaternion_world_wxyz[-1], dtype=np.float64
        )
        yaw_delta = _yaw(target_anchor_quaternion) - _yaw(
            source_anchor_quaternion
        )
        cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
        rotation = np.asarray(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        half = 0.5 * yaw_delta
        yaw_quaternion = np.asarray(
            (np.cos(half), 0.0, 0.0, np.sin(half)), dtype=np.float64
        )
        source_slice = slice(anchor_global + 1, anchor_global + 1 + count)
        root = np.asarray(
            archive["body_pos_w"][source_slice, 0], dtype=np.float64
        )
        relative_xy = root[:, :2] - source_anchor_position[:2]
        root[:, :2] = target_anchor_position[:2] + relative_xy @ rotation.T
        root[:, 2] += target_anchor_position[2] - source_anchor_position[2]
        source_quaternion = np.asarray(
            archive["body_quat_w"][source_slice, 0], dtype=np.float64
        )[..., (3, 0, 1, 2)]
        quaternion = quaternion_multiply_wxyz(
            yaw_quaternion[None], source_quaternion
        )
        joints = np.asarray(archive["joint_pos"][source_slice], dtype=np.float32)
        suffix_provenance = tuple(
            FrameProvenance(suffix_clip, frame, str(last.clip_id))
            for frame in range(
                int(last.source_frame) + 1,
                int(last.source_frame) + 1 + count,
            )
        )
        suffix_start = len(context_motion.root_position_world)
        extended = StitchedMotion(
            fps=context_motion.fps,
            root_position_world=np.concatenate(
                (context_motion.root_position_world, root.astype(np.float32)), axis=0
            ),
            root_quaternion_world_wxyz=np.concatenate(
                (
                    context_motion.root_quaternion_world_wxyz,
                    np.asarray(quaternion, dtype=np.float32),
                ),
                axis=0,
            ),
            joint_position=np.concatenate(
                (context_motion.joint_position, joints), axis=0
            ),
            provenance=context_motion.provenance + suffix_provenance,
            seam_indices=tuple(context_motion.seam_indices) + (suffix_start,),
        )
        mechanics = _maximum_steps(extended)
        audit = audit_stair_motion_collisions(
            extended,
            archive_path=archive_path,
            target_mesh=target_mesh,
            model_path=model_path,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=1.0e-6,
        )
        attempt = {
            "suffix_frames": count,
            "mechanics": mechanics,
            "mechanics_accepted": _mechanically_accepted(mechanics),
            "full_body_collision_audit": {
                "accepted": bool(audit.accepted),
                "maximum_foot_penetration_m": float(
                    audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    audit.maximum_forbidden_body_penetration_m
                ),
            },
        }
        suffix_attempts.append(attempt)
        if attempt["mechanics_accepted"] and audit.accepted:
            _save_motion(output_path, extended)
            receipt.update(
                extended_motion=str(output_path.resolve()),
                extended_motion_sha256=_sha256(output_path),
                accepted_suffix_frames=count,
                suffix_extension_audit=attempt,
            )
            context_motion = extended
            break
    receipt["suffix_attempts"] = suffix_attempts
    return receipt


def select(
    *, route_dir: Path, contract_path: Path, target_forward_speed_mps: float
) -> dict[str, object]:
    route = route_dir.expanduser().resolve()
    contract = json.loads(contract_path.expanduser().resolve().read_text())
    summary = json.loads((route / "summary.json").read_text())
    if summary.get("status") != "accepted":
        raise ValueError("route has no accepted expert motion")
    candidates = sorted(
        (route / "composition").glob("accepted_span_00_candidate_*.npz")
    )
    # A multi-block route can only be judged from the completed, exact-audited
    # composition.  Single-block alternatives remain useful for selecting a
    # faster coherent traversal when they cover the entire route.
    summary_motion = Path(str(summary["motion"])).resolve()
    if summary_motion not in {value.resolve() for value in candidates}:
        candidates.append(summary_motion)
    top_progress = float(contract["evaluation_geometry"]["top_progress_min_m"])
    half_width = float(contract["evaluation_geometry"]["lateral_half_width_m"])
    rows = []
    for path in candidates:
        with np.load(path, allow_pickle=False) as data:
            root = np.asarray(data["root_position_world"], dtype=np.float64)
            source_clips = np.asarray(
                data.get("source_archive_clip_index", np.asarray([-1])),
                dtype=np.int64,
            )
        progress = root[:, 0]
        delta = np.diff(progress)
        duration = (len(root) - 1) / 50.0
        net = float(progress[-1] - progress[0])
        forward_speed = net / duration
        backward_distance = float(-np.minimum(delta, 0.0).sum())
        backward_ratio = backward_distance / max(net, 1.0e-8)
        completed = bool(progress[-1] >= top_progress and net > 0.5)
        lateral_max = float(np.max(np.abs(root[:, 1])))
        lateral_ok = bool(lateral_max <= half_width)
        score = float(
            abs(forward_speed - target_forward_speed_mps)
            + 2.0 * backward_ratio
            + max(0.0, lateral_max - half_width) * 10.0
            + (0.0 if completed else 100.0)
        )
        rows.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "source_archive_clip_indices": np.unique(source_clips).tolist(),
                "frames": len(root),
                "net_progress_m": net,
                "mean_forward_speed_mps": forward_speed,
                "backward_distance_ratio": backward_ratio,
                "maximum_lateral_offset_m": lateral_max,
                "completed_top_progress": completed,
                "within_lateral_contract": lateral_ok,
                "score": score,
            }
        )
    accepted = [
        value for value in rows
        if value["completed_top_progress"] and value["within_lateral_contract"]
    ]
    if not accepted:
        raise ValueError("no exact-audited expert also completes the route envelope")
    selected = min(accepted, key=lambda value: (value["score"], value["path"]))
    extension = _extend_source_prefix(
        selected_path=Path(str(selected["path"])),
        terrain_usd=Path(str(contract["assets"]["terrain_usd"])),
        output_path=route / "command_selected_extended_motion.npz",
    )
    selected["recovery_motion_path"] = (
        extension["extended_motion"]
        if extension["extended_motion"] is not None
        else selected["path"]
    )
    receipt = {
        "schema": "command-compatible-stair-expert-selection/v1",
        "scene_id": contract["scene_id"],
        "geometry_sha256": contract["geometry_sha256"],
        "target_forward_speed_mps": float(target_forward_speed_mps),
        "all_candidates_previously_exact_audited": True,
        "selected": selected,
        "source_prefix_extension": extension,
        "candidates": rows,
    }
    destination = route / "command_selection.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--target-forward-speed-mps", type=float, default=0.55)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            select(
                route_dir=arguments.route_dir,
                contract_path=arguments.scene_contract,
                target_forward_speed_mps=arguments.target_forward_speed_mps,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
