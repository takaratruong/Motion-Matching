"""Build the exact centerline profile for the ten-step Karen recovery test."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_generic_terrain_smoke_scene import _build_profile


KAREN_TREAD_PITCH_M = 0.33074625
KAREN_TOP_PLATFORM_DEPTH_M = 1.04252227554
KAREN_STEP_TOPS_M = (
    0.19558225,
    0.3511645,
    0.52674675,
    0.702329,
    0.87791125,
    1.0534935,
    1.22907575,
    1.404658,
    1.58024025,
    1.7558225,
)
KAREN_COMMON_HALF_WIDTH_M = 0.6473596816
KAREN_TREAD_BOX_DEPTH_M = 0.38102977554
# The source Karen asset's first two boxes have slightly different lateral
# footprints.  Coordinates here are relative to the tread-center convention
# used by the recovery scene; steps 3--10 share the source step-3 footprint.
KAREN_LATERAL_BOUNDS_M = (
    (-0.6911061816, 0.6911061816),
    (-0.6692329316, 0.7129794316),
    *((-0.6473596816, 0.7785991816),) * 8,
)


def karen10_profile() -> tuple[tuple[float, float], ...]:
    """Return the effective highest walkable surface along the centerline."""

    profile: list[tuple[float, float]] = [(-2.0, 0.0), (0.0, 0.0)]
    for step_index, top_z_m in enumerate(KAREN_STEP_TOPS_M):
        front_x_m = step_index * KAREN_TREAD_PITCH_M
        profile.append((front_x_m, top_z_m))
        depth_m = (
            KAREN_TOP_PLATFORM_DEPTH_M
            if step_index == len(KAREN_STEP_TOPS_M) - 1
            else KAREN_TREAD_PITCH_M
        )
        profile.append((front_x_m + depth_m, top_z_m))
    return tuple(profile)


def build(output: Path) -> Path:
    return _build_profile(
        output,
        karen10_profile(),
        half_width=KAREN_COMMON_HALF_WIDTH_M,
    )


def build_rigid(output: Path) -> Path:
    """Write ten closed convex step boxes under one Isaac rigid-body root."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    model = UsdGeom.Xform.Define(stage, "/model")
    UsdPhysics.RigidBodyAPI.Apply(model.GetPrim())
    # The tracker spawner makes terrain objects kinematic.  Keep a valid
    # positive authored mass, matching its accepted generated terrain assets.
    UsdPhysics.MassAPI.Apply(model.GetPrim()).CreateMassAttr(1.0)
    UsdGeom.Xform.Define(stage, "/model/geometry")
    for step_index, top_z_m in enumerate(KAREN_STEP_TOPS_M):
        front_x_m = step_index * KAREN_TREAD_PITCH_M
        depth_m = (
            KAREN_TOP_PLATFORM_DEPTH_M
            if step_index == len(KAREN_STEP_TOPS_M) - 1
            else KAREN_TREAD_BOX_DEPTH_M
        )
        left, right = front_x_m, front_x_m + depth_m
        low_y, high_y = KAREN_LATERAL_BOUNDS_M[step_index]
        points = (
            (left, low_y, 0.0),
            (right, low_y, 0.0),
            (right, high_y, 0.0),
            (left, high_y, 0.0),
            (left, low_y, top_z_m),
            (right, low_y, top_z_m),
            (right, high_y, top_z_m),
            (left, high_y, top_z_m),
        )
        faces = (
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        )
        mesh = UsdGeom.Mesh.Define(
            stage, f"/model/geometry/step_{step_index + 1:02d}"
        )
        mesh.CreatePointsAttr([Gf.Vec3f(*value) for value in points])
        mesh.CreateFaceVertexCountsAttr([3] * 12)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(
            mesh.GetPrim()
        ).CreateApproximationAttr("convexHull")
    stage.SetDefaultPrim(model.GetPrim())
    stage.GetRootLayer().Save()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rigid-output", type=Path)
    arguments = parser.parse_args(argv)
    print(build(arguments.output))
    if arguments.rigid_output is not None:
        print(build_rigid(arguments.rigid_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
