"""Render a saved Justin tracker recovery rollout without rerunning physics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.canonical import CanonicalTerrainMesh
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_MODEL_PATH = Path(
    "/move/u/justingu/Projects/grail-stairs/GRAIL/imports/SONIC/gear_sonic/"
    "data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
)

# ``tracker_rollout.npz`` stores IsaacLab articulation order, not MJCF document
# order.  Address the named MJCF joints in exactly this saved-array order.
ARTICULATION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def _combined_usd_mesh(path: Path) -> CanonicalTerrainMesh:
    """Load every mesh prim because Justin's USD has three steps plus a platform."""

    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as error:
        raise RuntimeError("USD Python bindings are required") from error

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"could not open USD: {path}")
    if (
        UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
    ):
        raise ValueError("Justin USD must declare Z up and metersPerUnit=1")

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points = np.asarray(
            [
                tuple(transform.Transform(Gf.Vec3d(*map(float, point))))
                for point in points
            ],
            dtype=np.float64,
        )
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        left_handed = (
            mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
        )
        cursor = 0
        for count in counts:
            polygon = indices[cursor : cursor + int(count)]
            cursor += int(count)
            for offset in range(1, len(polygon) - 1):
                triangle = (
                    vertex_offset + int(polygon[0]),
                    vertex_offset + int(polygon[offset]),
                    vertex_offset + int(polygon[offset + 1]),
                )
                if left_handed:
                    triangle = (triangle[0], triangle[2], triangle[1])
                faces.append(triangle)
        vertices.extend(points)
        vertex_offset += len(points)

    if not vertices or not faces:
        raise ValueError(f"Justin USD contains no renderable mesh: {path}")
    face_array = np.asarray(faces, dtype=np.int32)
    return CanonicalTerrainMesh(
        vertices_local=np.asarray(vertices, dtype=np.float32),
        faces=face_array,
        valid_faces=np.ones(len(face_array), dtype=np.bool_),
        source_asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def render_attempt(
    attempt: Path,
    output: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    width: int = 960,
    height: int = 540,
    frame_stride: int = 1,
) -> Path:
    attempt = attempt.expanduser().resolve()
    rollout_path = attempt / "tracker_rollout.npz"
    usd_paths = tuple((attempt / "tracker_bundle" / "object_usd").glob("*.usd"))
    if not rollout_path.is_file() or len(usd_paths) != 1:
        raise ValueError(f"invalid Justin tracker attempt directory: {attempt}")

    with np.load(rollout_path, allow_pickle=False) as data:
        root_position = np.asarray(data["root_pos"], dtype=np.float32)
        root_quaternion = np.asarray(data["root_quat_wxyz"], dtype=np.float32)
        joint_position = np.asarray(data["joint_pos"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frame_count = len(root_position)
    if (
        frame_count == 0
        or root_position.shape != (frame_count, 3)
        or root_quaternion.shape != (frame_count, 4)
        or joint_position.shape != (frame_count, 29)
        or not np.isfinite(root_position).all()
        or not np.isfinite(root_quaternion).all()
        or not np.isfinite(joint_position).all()
    ):
        raise ValueError(f"invalid tracker rollout arrays: {rollout_path}")

    motion = StitchedMotion(
        fps=fps,
        root_position_world=root_position,
        root_quaternion_world_wxyz=root_quaternion,
        joint_position=joint_position,
        provenance=tuple(
            FrameProvenance(0, frame, attempt.name)
            for frame in range(frame_count)
        ),
        seam_indices=(),
    )
    identity = RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )
    terrain = TerrainMeshIndex(_combined_usd_mesh(usd_paths[0]), identity)
    render_stitched_motion(
        motion,
        model_path=model_path,
        output_path=output,
        target_mesh=terrain,
        joint_names=ARTICULATION_JOINT_NAMES,
        width=width,
        height=height,
        frame_stride=frame_stride,
        camera_distance=2.8,
        camera_elevation_deg=-14.0,
    )
    return output.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        render_attempt(
            args.attempt,
            args.out,
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            frame_stride=args.frame_stride,
        )
    )


if __name__ == "__main__":
    main()
