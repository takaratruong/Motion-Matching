"""Deterministically export the authenticated G1 FBX for Raylib."""

import argparse
import hashlib
import json
import math
import pathlib
import sys

import bpy


SOURCE_SHA256 = "1546cb574d0c9296f8200c8df8f75d1618e89bf43a6b46b8414fdfc852a4815b"
MAX_VERTICES = 65535
TARGET_VERTICES = 52000
RIGID_GROUP_COUNT = 35
RIGID_WEIGHT_TOLERANCE = 1e-6
NONZERO_WEIGHT_TOLERANCE = 1e-8


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_rigid_groups(mesh):
    groups = list(mesh.vertex_groups)
    if len(groups) != RIGID_GROUP_COUNT:
        raise RuntimeError("decimated G1 mesh must contain 35 vertex groups")

    group_indices = {group.index for group in groups}
    vertex_counts = {group.index: 0 for group in groups}
    assignments = []
    for vertex in mesh.data.vertices:
        if any(not math.isfinite(value.weight) for value in vertex.groups):
            raise RuntimeError("decimated G1 mesh contains non-finite weights")
        weighted = [
            value
            for value in vertex.groups
            if value.weight > NONZERO_WEIGHT_TOLERANCE
        ]
        if (
            len(weighted) != 1
            or weighted[0].group not in group_indices
            or abs(weighted[0].weight - 1.0) > RIGID_WEIGHT_TOLERANCE
        ):
            raise RuntimeError("decimated G1 mesh contains non-rigid weights")
        group_index = weighted[0].group
        assignments.append(group_index)
        vertex_counts[group_index] += 1
    if any(count == 0 for count in vertex_counts.values()):
        raise RuntimeError("decimated G1 mesh contains an empty vertex group")

    corner_counts = {group.index: 0 for group in groups}
    for polygon in mesh.data.polygons:
        polygon_groups = {assignments[index] for index in polygon.vertices}
        if len(polygon_groups) != 1:
            raise RuntimeError("decimated G1 polygon spans rigid vertex groups")
        group_index = next(iter(polygon_groups))
        corner_counts[group_index] += polygon.loop_total
    if any(count == 0 for count in corner_counts.values()):
        raise RuntimeError("decimated G1 vertex group contains no polygons")
    if any(count > MAX_VERTICES for count in corner_counts.values()):
        raise RuntimeError("G1 rigid group exceeds Raylib u16 vertex limit")

    original_data = mesh.data
    collections = tuple(mesh.users_collection)
    if not collections:
        raise RuntimeError("decimated G1 mesh is not linked to a collection")
    outputs = []
    for group in sorted(groups, key=lambda value: value.name):
        value = mesh.copy()
        value.data = mesh.data.copy()
        value.name = f"G1_{group.name}"
        value.data.name = f"G1_{group.name}"
        for collection in collections:
            collection.objects.link(value)
        bpy.ops.object.select_all(action="DESELECT")
        value.select_set(True)
        bpy.context.view_layer.objects.active = value
        value.vertex_groups.active_index = group.index
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.vertex_group_select()
        bpy.ops.mesh.select_all(action="INVERT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")
        if not value.data.vertices or not value.data.polygons:
            raise RuntimeError("G1 rigid group split produced an empty mesh")
        outputs.append(value)
    bpy.data.objects.remove(mesh, do_unlink=True)
    bpy.data.meshes.remove(original_data)
    return outputs


def export(source, output):
    if source.stat().st_size != 11348604 or sha256(source) != SOURCE_SHA256:
        raise RuntimeError("G1 FBX source identity mismatch")
    if (bpy.context.scene.render.threads_mode != "FIXED" or
            bpy.context.scene.render.threads != 1):
        raise RuntimeError("G1 export requires Blender --threads 1")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    armatures = [value for value in bpy.data.objects if value.type == "ARMATURE"]
    meshes = [value for value in bpy.data.objects if value.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("G1 FBX must contain one armature and one mesh")
    mesh = meshes[0]
    original_vertices = len(mesh.data.vertices)
    if original_vertices != 196692:
        raise RuntimeError("G1 FBX vertex count mismatch")
    modifier = mesh.modifiers.new("raylib_u16", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = TARGET_VERTICES / original_vertices
    mesh.modifiers.move(len(mesh.modifiers) - 1, 0)
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    split_rigid_groups(mesh)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_yup=True,
        export_skins=True,
        export_animations=False,
        export_apply=False,
    )


def _certify(source, output):
    repository_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from resources.g1_mesh.validate_g1_raylib_glb import (
        EVIDENCE_FIELDS,
        REQUIRED_ARTICULATED_BONES,
        inspect_g1_glb,
        validate_g1_glb,
    )

    evidence = inspect_g1_glb(output)
    payload = {field: evidence[field] for field in EVIDENCE_FIELDS}
    payload.update(
        {
            "blender_version": bpy.app.version_string,
            "glb_size": output.stat().st_size,
            "skin_names": evidence["skin_names"],
            "required_articulated_bones": list(REQUIRED_ARTICULATED_BONES),
            "source_size": source.stat().st_size,
            "source_sha256": sha256(source),
        }
    )
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return validate_g1_glb(output, manifest_path, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = output.with_name("manifest.json")
    try:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        export(source, output)
        report = _certify(source, output)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    except Exception:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
