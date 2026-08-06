"""Prepare selected terrain-maneuver pilots for a small SONIC replay.

Each ``--pilot`` is written as ``MANIFEST#LABEL``.  The selected kinematics
are converted directly into SONIC's native motion-library representation and
paired with the exact terrain asset and pose recorded by the pilot manifest.
No clean motion is re-matched and no rollout state is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from .offline_corpus import JOINT_NAMES


_BARE_MESH_DECLARATION = 'def Mesh "Terrain"\n{'


def _usd_crate_has_physics(path: Path) -> bool | None:
    """Inspect a binary USD crate when the optional USD bindings exist."""

    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return None
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f"failed to open USD crate: {path}")
    has_rigid_body = False
    has_collision = False
    for prim in stage.Traverse():
        has_rigid_body = has_rigid_body or bool(UsdPhysics.RigidBodyAPI(prim))
        has_collision = has_collision or bool(UsdPhysics.CollisionAPI(prim))
    return has_rigid_body and has_collision


def _copy_physics_ready_terrain(source: Path, destination: Path) -> str:
    """Copy a terrain, adding static collision schemas to our bare USDA meshes.

    Original GRAIL terrains are already physics-authored USD crates.  The composed
    course terrains used by the kinematic oracle are deliberately minimal USDA
    meshes, so Isaac Lab otherwise sees no rigid body and refuses to attach its
    contact sensors.  Keeping the points/faces byte-for-byte and adding schemas to
    the same mesh is enough for the collector's kinematic rigid-object wrapper.
    """
    payload = source.read_bytes()
    if payload.startswith(b"PXR-USDC"):
        physics_ready = _usd_crate_has_physics(source)
        if physics_ready is False:
            raise ValueError(f"binary USD terrain lacks physics schemas: {source}")
        destination.write_bytes(payload)
        return (
            "source_physics"
            if physics_ready is True
            else "source_usd_crate_uninspected"
        )
    if b"PhysicsRigidBody" in payload and b"PhysicsCollision" in payload:
        destination.write_bytes(payload)
        return "source_physics"
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"terrain has no discoverable physics schemas and is not USDA: {source}"
        ) from error
    if _BARE_MESH_DECLARATION not in text:
        raise ValueError(
            f"unsupported bare USDA terrain root (expected Mesh Terrain): {source}"
        )
    physics_declaration = '''def Xform "Terrain" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1

    def Mesh "Geometry" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        bool physics:collisionEnabled = 1
        token physics:approximation = "none"'''
    destination.write_text(
        text.replace(_BARE_MESH_DECLARATION, physics_declaration, 1).rstrip()
        + "\n}\n"
    )
    return "authored_static_mesh_physics"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection(value: str) -> tuple[Path, str]:
    if "#" not in value:
        raise ValueError("pilot must have the form MANIFEST#LABEL")
    path, label = value.rsplit("#", 1)
    if not path or not label:
        raise ValueError("pilot must have the form MANIFEST#LABEL")
    return Path(path).expanduser().resolve(), label


def _accepted_manifest_selections(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Expand manifests into deterministic, automatically admitted selections."""

    selections: list[str] = []
    seen: set[tuple[Path, str]] = set()
    for value in paths:
        path = value.expanduser().resolve()
        manifest = json.loads(path.read_text())
        for row in manifest.get("pilots", ()):
            if not bool(row.get("automatic_gate_accepted", False)):
                continue
            label = str(row["label"])
            key = (path, label)
            if key in seen:
                continue
            seen.add(key)
            selections.append(f"{path}#{label}")
    return tuple(selections)


def _clip_stem(manifest_path: Path, label: str) -> str:
    """Return a stable ID that remains unique for nested bank manifests."""

    source = f"{manifest_path.parent.parent.name}_{manifest_path.parent.name}_{label}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", source).strip("_")


def prepare(pilots: tuple[str, ...], output: Path) -> dict[str, object]:
    import joblib

    from sonic_port.takara.csv_to_motion_lib import (
        build_entry,
        csv_joint_permutation,
    )
    from sonic_port.takara.mjcf_spec import load_spec

    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite SONIC bundle: {destination}")
    if not pilots:
        raise ValueError("at least one --pilot is required")
    robot_dir = destination / "robot"
    terrain_dir = destination / "object_usd"
    robot_dir.mkdir(parents=True)
    terrain_dir.mkdir(parents=True)

    spec = load_spec()
    permutation = csv_joint_permutation(
        list(JOINT_NAMES), list(spec.joint_names)
    )
    merged: dict[str, dict[str, object]] = {}
    object_motions: dict[str, dict[str, object]] = {}
    clip_records: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    used_ids: set[str] = set()
    flat_placeholder: Path | None = None

    for value in pilots:
        manifest_path, label = _selection(value)
        manifest = json.loads(manifest_path.read_text())
        matches = [row for row in manifest["pilots"] if row["label"] == label]
        if len(matches) != 1:
            raise ValueError(
                f"{manifest_path}: expected exactly one pilot labelled {label!r}"
            )
        row = matches[0]
        if not bool(row.get("automatic_gate_accepted", False)):
            raise ValueError(f"refusing automatically rejected pilot: {label}")
        terrain = Path(str(manifest["terrain_usd"])).expanduser().resolve()
        motion = Path(str(row["motion"])).expanduser().resolve()
        if not terrain.is_file() or not motion.is_file():
            raise FileNotFoundError(f"missing motion or terrain for {label}")

        stem = _clip_stem(manifest_path, label)
        if stem in used_ids:
            raise ValueError(f"duplicate generated clip id {stem!r}")
        used_ids.add(stem)

        with np.load(motion, allow_pickle=False) as data:
            root = np.asarray(data["root_position_world"], dtype=np.float64)
            quaternion_wxyz = np.asarray(
                data["root_quaternion_world_wxyz"], dtype=np.float64
            )
            joints = np.asarray(data["joint_position"], dtype=np.float64)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if (
            root.ndim != 2
            or root.shape[1] != 3
            or quaternion_wxyz.shape != (len(root), 4)
            or joints.shape != (len(root), 29)
            or not np.isfinite(root).all()
            or not np.isfinite(quaternion_wxyz).all()
            or not np.isfinite(joints).all()
            or not np.isfinite(fps)
            or fps <= 0.0
        ):
            raise ValueError(f"invalid maneuver arrays in {motion}")
        quaternion_wxyz /= np.linalg.norm(
            quaternion_wxyz, axis=1, keepdims=True
        )
        entry = build_entry(
            root,
            quaternion_wxyz[:, (1, 2, 3, 0)],
            joints[:, permutation],
            spec.dof_axis,
            fps,
        )
        merged[stem] = entry
        joblib.dump({stem: entry}, robot_dir / f"{stem}.pkl")
        terrain_copy = terrain_dir / f"{stem}.usd"
        terrain_physics = _copy_physics_ready_terrain(terrain, terrain_copy)
        if flat_placeholder is None:
            flat_placeholder = terrain_copy

        terrain_pose = {
            "position_env": list(manifest["terrain_position_world"]),
            "rotation_env_wxyz": list(
                manifest["terrain_quaternion_world_from_usd_wxyz"]
            ),
        }
        object_position = np.asarray(terrain_pose["position_env"], dtype=np.float32)
        object_quaternion = np.asarray(
            terrain_pose["rotation_env_wxyz"], dtype=np.float32
        )
        object_quaternion /= np.linalg.norm(object_quaternion)
        object_motions[stem] = {
            "root_pos": np.broadcast_to(
                object_position, (len(root), 1, 3)
            ).copy(),
            "root_quat": np.broadcast_to(
                object_quaternion, (len(root), 1, 4)
            ).copy(),
            "fps": fps,
        }
        clip_records.append(
            {
                "stem": stem,
                "terrain": terrain_pose,
                "reference_transform": {
                    "position_env": [0.0, 0.0, 0.0],
                    "rotation_env_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "category": str(row.get("kind", "terrain_maneuver")),
                "n_frames": int(len(root)),
                "fps": fps,
            }
        )
        provenance.append(
            {
                "clip_id": stem,
                "pilot_manifest": str(manifest_path),
                "pilot_label": label,
                "motion": str(motion),
                "motion_sha256": _sha256(motion),
                "terrain": str(terrain),
                "terrain_sha256": _sha256(terrain),
                "bundled_terrain_sha256": _sha256(terrain_copy),
                "terrain_physics": terrain_physics,
                "automatic_gate_accepted": True,
                "visual_review_required_before_collection_scale_up": True,
            }
        )

    assert flat_placeholder is not None
    shutil.copy2(flat_placeholder, destination / "flat_placeholder.usd")
    joblib.dump(merged, destination / "motion_lib_merged.pkl")
    joblib.dump(object_motions, destination / "objects.pkl")
    (destination / "clips.json").write_text(
        json.dumps(clip_records, indent=2, sort_keys=True) + "\n"
    )
    payload: dict[str, object] = {
        "schema": "terrain-maneuver-sonic-bundle/v1",
        "clip_count": len(clip_records),
        "joint_source_order": list(JOINT_NAMES),
        "joint_sonic_order": list(spec.joint_names),
        "joint_permutation": list(permutation),
        "clips": provenance,
    }
    (destination / "bundle_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="append", default=[])
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="include every automatically admitted pilot in this manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    expanded = _accepted_manifest_selections(tuple(arguments.manifest))
    selections = tuple(dict.fromkeys((*arguments.pilot, *expanded)))
    result = prepare(selections, arguments.output)
    print(
        json.dumps(
            {
                "clip_count": result["clip_count"],
                "output": str(arguments.output.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
