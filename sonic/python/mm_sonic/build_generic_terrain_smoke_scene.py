"""Build a small exact-mesh scene for the unified terrain dispatcher smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_profile(
    output: Path, profile: tuple[tuple[float, float], ...], *, half_width: float
) -> Path:
    """Write one extruded exact-height profile as a triangle USD mesh."""

    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = tuple(profile)
    if len(profile) < 2:
        raise ValueError("terrain profile needs at least two vertices")
    if any(right[0] < left[0] for left, right in zip(profile, profile[1:])):
        raise ValueError("terrain profile x coordinates must be nondecreasing")
    if half_width <= 0.0:
        raise ValueError("terrain half width must be positive")
    # Consecutive duplicate X positions are intentional vertical risers.
    points: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for (left_x, left_z), (right_x, right_z) in zip(profile, profile[1:]):
        base = len(points)
        points.extend(
            (
                (left_x, -half_width, left_z),
                (right_x, -half_width, right_z),
                (right_x, half_width, right_z),
                (left_x, half_width, left_z),
            )
        )
        # Counter-clockwise from the free-space side: upward for walkable
        # strips and outward along vertical risers.
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError:
        # USD is deliberately a tiny ASCII format here.  Keeping a no-pxr
        # fallback makes procedural evaluation scenes reproducible on the
        # CPU/login environment; the resulting file is byte-equivalent in
        # geometry to the pxr-authored mesh consumed by the terrain loader.
        counts = ", ".join("3" for _ in range(len(indices) // 3))
        index_text = ", ".join(str(value) for value in indices)
        point_text = ", ".join(
            f"({x:g}, {y:g}, {z:g})" for x, y, z in points
        )
        destination.write_text(
            "#usda 1.0\n"
            "(\n"
            '    defaultPrim = "Terrain"\n'
            "    metersPerUnit = 1\n"
            '    upAxis = "Z"\n'
            ")\n\n"
            'def Mesh "Terrain"\n'
            "{\n"
            f"    int[] faceVertexCounts = [{counts}]\n"
            f"    int[] faceVertexIndices = [{index_text}]\n"
            f"    point3f[] points = [{point_text}]\n"
            '    uniform token subdivisionScheme = "none"\n'
            "}\n"
        )
        return destination

    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Terrain")
    gf_points = [Gf.Vec3f(*value) for value in points]
    mesh.CreatePointsAttr(gf_points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return destination


def build(output: Path) -> Path:
    """Write one mesh containing flat, ramp, and curb route geometry."""

    # Along +X this yields: flat, ramp up, an elevated flat route, one curb
    # up/down, and elevated flat again.  Down-ramp synthesis is covered by
    # the separate held-out exact-mesh suite rather than duplicated here.
    profile = (
        (-2.0, 0.00),
        (-0.5, 0.00),
        (1.5, 0.25),
        (2.2, 0.25),
        (5.5, 0.25),
        (6.0, 0.25),
        (6.0, 0.41),
        (6.65, 0.41),
        (6.65, 0.25),
        (9.0, 0.25),
    )
    return _build_profile(output, profile, half_width=2.0)


def build_gauntlet(output: Path) -> Path:
    """Write a mixed ramp/curb/up-stair/down-stair/ramp test strip."""

    profile = (
        (-2.0, 0.00),
        (-0.5, 0.00),
        (1.5, 0.25),
        (5.5, 0.25),
        (6.0, 0.25),
        (6.0, 0.41),
        (6.65, 0.41),
        (6.65, 0.25),
        (9.50, 0.25),
        (9.50, 0.40),
        (9.85, 0.40),
        (9.85, 0.55),
        (10.20, 0.55),
        (10.20, 0.70),
        (10.55, 0.70),
        (10.55, 0.85),
        (12.00, 0.85),
        (12.00, 0.70),
        (12.35, 0.70),
        (12.35, 0.55),
        (12.70, 0.55),
        (12.70, 0.40),
        (13.05, 0.40),
        (13.05, 0.25),
        (15.00, 0.25),
        (17.00, 0.00),
        (19.00, 0.00),
    )
    return _build_profile(output, profile, half_width=2.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gauntlet",
        action="store_true",
        help="include stair ascent/descent and a final downhill ramp",
    )
    arguments = parser.parse_args(argv)
    print(
        build_gauntlet(arguments.output)
        if arguments.gauntlet
        else build(arguments.output)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
