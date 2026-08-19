"""Build tracker USDs and a strict recovery contract for one stair-suite scene.

The source of truth is the frozen TML campaign manifest.  Its local boxes use
the Karen convention (ascent along local -X); the tracker assets below rotate
those boxes into a simple reference frame with the first riser at X=0 and
ascent along world +X.  No learner motion is used to construct the geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


DEFAULT_MANIFEST = Path(
    "/move/u/bodow/Projects/TML-BeyondMimic-justin-recovery-test/"
    "config/sonic_stair_suite10_campaign_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_geometry(manifest_path: Path, scene_id: str) -> dict[str, object]:
    source = manifest_path.expanduser().resolve()
    payload = json.loads(source.read_text())
    if payload.get("schema") != "sonic-fused-depth-recovery-campaign/v1":
        raise ValueError("unsupported stair-suite campaign manifest")
    matches = [
        value for value in payload.get("geometries", [])
        if value.get("scene_id") == scene_id
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one {scene_id!r} geometry")
    geometry = dict(matches[0])
    bounds = geometry.get("step_bounds")
    if not isinstance(bounds, list) or len(bounds) != int(geometry["step_count"]):
        raise ValueError("stair-suite geometry has invalid step bounds")
    geometry["source_manifest"] = str(source)
    geometry["source_manifest_sha256"] = _sha256(source)
    return geometry


def _reference_bounds(geometry: Mapping[str, object]):
    """Yield exact box bounds after the TML yaw-pi placement transform."""

    rows = geometry["step_bounds"]
    first_min = rows[0][1]
    first_max = rows[0][2]
    local_center_y = 0.5 * (float(first_min[1]) + float(first_max[1]))
    for name, low, high in rows:
        # TML places the local stair body at yaw=pi.  Its Y origin equals the
        # local tread centre, so this is the same physical centered staircase.
        yield (
            str(name),
            (
                -float(high[0]),
                local_center_y - float(high[1]),
                float(low[2]),
            ),
            (
                -float(low[0]),
                local_center_y - float(low[1]),
                float(high[2]),
            ),
        )


def _write_usd(path: Path, geometry: Mapping[str, object], *, rigid: bool) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    model = UsdGeom.Xform.Define(stage, "/model")
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(model.GetPrim())
        UsdPhysics.MassAPI.Apply(model.GetPrim()).CreateMassAttr(1.0)
    UsdGeom.Xform.Define(stage, "/model/geometry")
    combined_points = []
    combined_faces = []
    for index, (name, low, high) in enumerate(_reference_bounds(geometry)):
        lx, ly, lz = low
        hx, hy, hz = high
        points = (
            (lx, ly, lz), (hx, ly, lz), (hx, hy, lz), (lx, hy, lz),
            (lx, ly, hz), (hx, ly, hz), (hx, hy, hz), (lx, hy, hz),
        )
        faces = (
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
        )
        if rigid:
            mesh = UsdGeom.Mesh.Define(
                stage, f"/model/geometry/box_{index + 1:02d}_{name}"
            )
            mesh.CreatePointsAttr([Gf.Vec3f(*value) for value in points])
            mesh.CreateFaceVertexCountsAttr([3] * 12)
            mesh.CreateFaceVertexIndicesAttr(faces)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(
                mesh.GetPrim()
            ).CreateApproximationAttr("convexHull")
        else:
            offset = len(combined_points)
            combined_points.extend(points)
            combined_faces.extend(offset + value for value in faces)
    if not rigid:
        # The route oracle intentionally consumes one terrain mesh.  Combining
        # the triangles changes no coordinates or surfaces; Isaac retains the
        # per-box convex collision representation in the separate rigid USD.
        mesh = UsdGeom.Mesh.Define(stage, "/model/geometry/terrain")
        mesh.CreatePointsAttr([Gf.Vec3f(*value) for value in combined_points])
        mesh.CreateFaceVertexCountsAttr([3] * (len(combined_faces) // 3))
        mesh.CreateFaceVertexIndicesAttr(combined_faces)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.SetDefaultPrim(model.GetPrim())
    stage.GetRootLayer().Save()
    return destination


def build(
    *,
    manifest_path: Path,
    scene_id: str,
    output: Path,
    rigid_output: Path,
    contract_output: Path,
    learner_front_x_m: float = 1.5,
) -> dict[str, object]:
    geometry = load_geometry(manifest_path, scene_id)
    terrain = _write_usd(output, geometry, rigid=False)
    rigid = _write_usd(rigid_output, geometry, rigid=True)
    reference_rows = tuple(_reference_bounds(geometry))
    top_progress = float(reference_rows[-1][1][0])
    top_back_edge = float(reference_rows[-1][2][0])
    # Finish safely inside the platform.  Sampling exactly on its back edge
    # makes the mesh ray query ambiguous between the top face and flat-ground
    # fallback, which can falsely turn a monotonic ascent into a final drop.
    route_end = top_progress + min(0.70, 0.65 * (top_back_edge - top_progress))
    riser_progress = tuple(float(value) for value in geometry["riser_progress_m"])
    rises = tuple(float(value) for value in geometry["riser_heights_m"])
    contract = {
        "schema": "stair-suite-tracker-recovery-scene/v1",
        "scene_id": scene_id,
        "geometry_sha256": str(geometry["geometry_sha256"]),
        "learner_front_xy": [float(learner_front_x_m), 0.0],
        "reference_front_xy": [0.0, 0.0],
        "learner_heading_yaw_rad": 0.0,
        "reference_heading_yaw_rad": 0.0,
        "tread_m": (
            float(riser_progress[1] - riser_progress[0])
            if len(riser_progress) > 1 else 0.3
        ),
        "riser_progress_m": list(riser_progress),
        "num_steps": int(geometry["step_count"]),
        "exact_command_required": True,
        "command_contract": {
            "task12_source": "exact learner recovery_trace.npz task12",
            "allowed_max_abs_error": 0.0,
        },
        "route": {
            # The expert route only needs enough pre-riser context to cover
            # recovery rewinds.  The learner's 0.75/1.5/3.0 m initial spawn
            # distance is not a geometry property and is preserved separately
            # in each failed trace.
            "start_xy": [-1.5, 0.0],
            "end_xy": [route_end, 0.0],
        },
        "evaluation_geometry": {
            "front_xy": [0.0, 0.0],
            "heading_yaw_rad": 0.0,
            "top_progress_min_m": top_progress,
            "lateral_half_width_m": 0.45 * float(geometry["width_m"]),
            "root_z_min_m": float(sum(rises) + 0.48),
            "scene_id": scene_id,
            "geometry_sha256": str(geometry["geometry_sha256"]),
            "object_usd_sha256": _sha256(rigid),
        },
        "assets": {
            "terrain_usd": str(terrain),
            "terrain_usd_sha256": _sha256(terrain),
            "rigid_usd": str(rigid),
            "rigid_usd_sha256": _sha256(rigid),
        },
        "source_manifest": str(geometry["source_manifest"]),
        "source_manifest_sha256": str(geometry["source_manifest_sha256"]),
        "reference_step_bounds": [
            [name, list(low), list(high)] for name, low, high in reference_rows
        ],
    }
    destination = contract_output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--learner-front-x-m", type=float, default=1.5)
    arguments = parser.parse_args(argv)
    result = build(
        manifest_path=arguments.manifest,
        scene_id=arguments.scene_id,
        output=arguments.output,
        rigid_output=arguments.rigid_output,
        contract_output=arguments.contract_output,
        learner_front_x_m=arguments.learner_front_x_m,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
