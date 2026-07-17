# G1 Mesh Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the real Unitree G1 mesh and render it from the exact accepted motion-matching pose, with optional bone and physical-sole diagnostics.

**Architecture:** Task 1 converts the existing validated rigid-skinned FBX into one Raylib-compatible, indexed GLB and validates its skin, bounds, weights, and hashes. Task 2 adds a focused renderer that maps 30 articulated skin bones to `G1Bone`, propagates nine fixed attachments from bind-local transforms, and integrates mesh/bone/sole toggles without changing simulation state.

**Tech Stack:** C++17, Raylib 5.0, raymath, Blender 4.0.2 Python, Python 3.12 `unittest`, binary glTF 2.0, Git.

## Global Constraints

- Exact float32 `0.04` second simulation cadence remains unchanged.
- Terrain assets and motion data remain separate.
- Heading and travel direction remain independent.
- No IK or motion-matching behavior is changed in this checkpoint.
- Use `/home/ubuntu/projects/g1_mm/g1.fbx` only as conversion input; its byte size is `11348604` and SHA-256 is `1546cb574d0c9296f8200c8df8f75d1618e89bf43a6b46b8414fdfc852a4815b`.
- Do not add empirical root, scale, axis, ankle, or foot offsets.
- Do not inspect, discover, query, signal, replace, restart, or terminate the old visualizer. Never use `ps`, `pgrep`, `pidof`, `pkill`, `systemctl`, or a visualizer helper.
- Finite processes and the final candidate launch use direct `DISPLAY=:1` commands and unique output paths.
- Every coherent test-backed commit is pushed immediately to `checkpoint/g1-footprint-task6`.
- Stage only task-owned paths and preserve unrelated work.

## File Map

- Create `resources/g1_mesh/export_g1_raylib_glb.py`: Blender-only deterministic FBX-to-GLB exporter with a sub-65,536-vertex mesh and canonical manifest.
- Create `resources/g1_mesh/validate_g1_raylib_glb.py`: dependency-free GLB/manifest validator used by tests and the exporter.
- Create `resources/g1_mesh/__init__.py`: explicit Python package marker for focused tests.
- Create `resources/g1_mesh/g1_raylib.glb`: committed runtime asset.
- Create `resources/g1_mesh/manifest.json`: source/output hashes and structural evidence.
- Create `tests/python/test_g1_mesh_asset.py`: asset corruption, hierarchy, rigid-weight, bounds, and hash tests.
- Create `g1_mesh_renderer.h`: exact bone mapping, bind-local propagation, Raylib model/pose ownership, drawing, and cleanup.
- Create `tests/cpp/test_g1_mesh_renderer.cpp`: pure mapping/pose tests plus controller source-wiring guards.
- Modify `controller.cpp`: load/update/draw/unload the mesh and add `M`, `B`, and `P` diagnostic toggles.

---

### Task 1: Export and Certify the Raylib GLB

**Files:**

- Create: `tests/python/test_g1_mesh_asset.py`
- Create: `resources/g1_mesh/__init__.py`
- Create: `resources/g1_mesh/export_g1_raylib_glb.py`
- Create: `resources/g1_mesh/validate_g1_raylib_glb.py`
- Create: `resources/g1_mesh/g1_raylib.glb`
- Create: `resources/g1_mesh/manifest.json`

**Interfaces:**

- Consumes `/home/ubuntu/projects/g1_mm/g1.fbx` with the exact authenticated hash in Global Constraints.
- Produces:

```python
def inspect_g1_glb(glb_path: pathlib.Path) -> dict[str, object]:
    """Return structural evidence or raise ValueError."""


def validate_g1_glb(
    glb_path: pathlib.Path,
    manifest_path: pathlib.Path,
    source_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Return certified counts/bounds or raise ValueError."""
```

- The returned dictionary contains `bone_count`, `skin_count`, `vertex_count`,
  `maximum_primitive_vertices`, `rigid_vertex_count`, `height_m`, and
  `glb_sha256`.

- [ ] **Step 1: Write the asset validator tests before creating production files**

Create `tests/python/test_g1_mesh_asset.py` with the following contract:

```python
import copy
import json
import pathlib
import tempfile
import unittest

from resources.g1_mesh.validate_g1_raylib_glb import validate_g1_glb


ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSET = ROOT / "resources/g1_mesh/g1_raylib.glb"
MANIFEST = ROOT / "resources/g1_mesh/manifest.json"
SOURCE = pathlib.Path("/home/ubuntu/projects/g1_mm/g1.fbx")


class G1MeshAssetTests(unittest.TestCase):
    def test_committed_asset_is_raylib_compatible(self):
        report = validate_g1_glb(ASSET, MANIFEST, SOURCE)
        self.assertEqual(report["skin_count"], 1)
        self.assertEqual(report["bone_count"], 39)
        self.assertLessEqual(report["maximum_primitive_vertices"], 65535)
        self.assertEqual(report["rigid_vertex_count"], report["vertex_count"])
        self.assertGreaterEqual(report["height_m"], 0.8)
        self.assertLessEqual(report["height_m"], 1.6)

    def test_manifest_names_all_required_articulated_bones_once(self):
        manifest = json.loads(MANIFEST.read_text())
        required = manifest["required_articulated_bones"]
        self.assertEqual(len(required), 30)
        self.assertEqual(len(set(required)), 30)
        self.assertNotIn("Simulation", required)
        self.assertIn("left_ankle_pitch_link", required)
        self.assertIn("left_ankle_roll_link", required)
        self.assertIn("right_ankle_pitch_link", required)
        self.assertIn("right_ankle_roll_link", required)

    def test_corrupt_glb_and_manifest_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            glb = root / "g1.glb"
            manifest = root / "manifest.json"
            glb.write_bytes(ASSET.read_bytes()[:-1] + b"x")
            payload = copy.deepcopy(json.loads(MANIFEST.read_text()))
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "GLB SHA-256"):
                validate_g1_glb(glb, manifest)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and record the expected RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_g1_mesh_asset -v
```

Expected: import failure for
`resources.g1_mesh.validate_g1_raylib_glb`; neither validator nor asset exists.

- [ ] **Step 3: Implement the deterministic exporter**

Create `resources/g1_mesh/export_g1_raylib_glb.py`. Its executable core is:

```python
import argparse
import hashlib
import json
import pathlib
import sys

import bpy


SOURCE_SHA256 = "1546cb574d0c9296f8200c8df8f75d1618e89bf43a6b46b8414fdfc852a4815b"
MAX_VERTICES = 65535
TARGET_VERTICES = 52000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, output):
    if source.stat().st_size != 11348604 or sha256(source) != SOURCE_SHA256:
        raise RuntimeError("G1 FBX source identity mismatch")
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
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    if len(mesh.data.vertices) > MAX_VERTICES:
        raise RuntimeError("decimated G1 mesh exceeds Raylib u16 limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_yup=True,
        export_skins=True,
        export_animations=False,
        export_apply=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    export(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
```

After export, the script adds the repository root to `sys.path`, imports
`inspect_g1_glb` and `validate_g1_glb`, and calls `inspect_g1_glb(output)`.
It combines that returned evidence with Blender version, source size/hash,
output size/hash, the ordered 39 skin names, and the exact ordered 30 required
articulated names from the design. It writes the adjacent `manifest.json`
using `json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"`,
then calls `validate_g1_glb(output, manifest_path, source)`. Any failed
inspection or final validation deletes both incomplete outputs and exits
nonzero.

- [ ] **Step 4: Implement dependency-free GLB validation**

Create `resources/g1_mesh/validate_g1_raylib_glb.py` with these exact checks:

```python
GLTF_JSON_CHUNK = 0x4E4F534A
GLTF_BIN_CHUNK = 0x004E4942
UNSIGNED_BYTE = 5121
UNSIGNED_SHORT = 5123
FLOAT = 5126


def _chunks(data):
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("invalid GLB header")
    offset = 12
    result = {}
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        result[kind] = data[offset:offset + length]
        offset += length
    if offset != len(data) or GLTF_JSON_CHUNK not in result:
        raise ValueError("invalid GLB chunks")
    return result


def _accessor_bytes(document, binary, index):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride")
    return accessor, memoryview(binary)[start:], stride
```

`inspect_g1_glb` decodes the JSON chunk, requires exactly one skin and 39
unique joint node names, validates the exact parent relation of every required
articulated name, then walks every primitive. `POSITION` must be float `VEC3`;
`JOINTS_0` must be unsigned-byte or unsigned-short `VEC4`; `WEIGHTS_0` must be
float `VEC4`; indices must be unsigned-short; and each position accessor count
must be at most `65535`. Decode weights using the accessor/view byte offsets
and stride. For every vertex require exactly one weight within `1e-6` of
`1.0` and the other three within `1e-6` of zero. Union the accessor `min` and
`max` triples and require the largest bounding-box extent in `[0.8, 1.6]`.
`validate_g1_glb` calls `inspect_g1_glb`, then requires every recorded
structural field, the output hash/size, and, when supplied, the exact source
hash/size to match the manifest. Both functions return the evidence dictionary
documented in Interfaces.

- [ ] **Step 5: Generate the committed asset and obtain GREEN**

Run:

```bash
blender --background --python resources/g1_mesh/export_g1_raylib_glb.py -- \
  --source /home/ubuntu/projects/g1_mm/g1.fbx \
  --output resources/g1_mesh/g1_raylib.glb
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_g1_mesh_asset -v
```

Expected: exporter exits zero; validator reports one 39-bone skin, rigid
weights, no primitive over 65,535 vertices, and a 0.8–1.6 m extent; all three
tests pass.

- [ ] **Step 6: Run asset self-review and commit**

```bash
git diff --check
git add resources/g1_mesh tests/python/test_g1_mesh_asset.py
git diff --cached --check
git commit -m "feat: package certified G1 render mesh"
```

Expected: exactly the exporter, validator, GLB, manifest, and focused tests are
committed. The external FBX is not staged.

---

### Task 2: Transfer Accepted Poses and Integrate Diagnostic Rendering

**Files:**

- Create: `tests/cpp/test_g1_mesh_renderer.cpp`
- Create: `g1_mesh_renderer.h`
- Modify: `controller.cpp`
- Verify only: `g1_frame_transaction.h`
- Verify only: `g1_controller_state.h`
- Verify only: `g1_ik_runtime.h`
- Verify only: `resources/check_g1_runtime_log.py`

**Interfaces:**

- Produces:

```cpp
enum { G1MeshBoneCount = 39 };

struct G1MeshBinding {
    int model_to_g1[G1MeshBoneCount];
    Transform bind_local[G1MeshBoneCount];
    int parents[G1MeshBoneCount];
};

struct G1MeshRenderer {
    Model model;
    ModelAnimation animation;
    G1MeshBinding binding;
    bool loaded;
};

int g1_mesh_g1_bone_for_name(const char* name);

bool g1_mesh_binding_build(
    G1MeshBinding& output,
    const BoneInfo* bones,
    const Transform* global_bind_pose,
    int bone_count,
    char* error,
    int error_capacity);

bool g1_mesh_pose_build(
    Transform* output,
    int output_count,
    const G1MeshBinding& binding,
    const slice1d<vec3> accepted_positions,
    const slice1d<quat> accepted_rotations,
    char* error,
    int error_capacity);

bool g1_mesh_renderer_load(
    G1MeshRenderer& renderer,
    const char* path,
    char* error,
    int error_capacity);

bool g1_mesh_renderer_update(
    G1MeshRenderer& renderer,
    const slice1d<vec3> accepted_positions,
    const slice1d<quat> accepted_rotations,
    char* error,
    int error_capacity);

void g1_mesh_renderer_draw(const G1MeshRenderer& renderer);
void g1_mesh_renderer_unload(G1MeshRenderer& renderer);
```

- [ ] **Step 1: Write mapping, pose, failure, and controller-wiring tests first**

Create `tests/cpp/test_g1_mesh_renderer.cpp`. Include
`g1_mesh_renderer.h`. Define `Fixture` with owned 39-element `BoneInfo`,
global-bind, output, and binding arrays; 31-element accepted-position and
accepted-rotation arrays exposed through `slice1d`; cached torso/head indices;
and a 256-byte error buffer. `valid_fixture()` fills the exact ordered manifest
names and production parent tree, identity rotations/scales, and finite
positions. Define `compose_transform()` with parent-global times child-local
quaternion/translation composition and `transform_near()` with componentwise
`5e-6` position/quaternion tolerance. Add these tests:

```cpp
static void test_exact_articulated_mapping()
{
    check(g1_mesh_g1_bone_for_name("pelvis") == G1_Hips,
          "pelvis maps to hips");
    check(g1_mesh_g1_bone_for_name("left_ankle_pitch_link") ==
              G1_LeftAnkle,
          "left ankle pitch maps to LeftAnkle");
    check(g1_mesh_g1_bone_for_name("left_ankle_roll_link") == G1_LeftToe,
          "left ankle roll maps to LeftToe");
    check(g1_mesh_g1_bone_for_name("right_ankle_pitch_link") ==
              G1_RightAnkle,
          "right ankle pitch maps to RightAnkle");
    check(g1_mesh_g1_bone_for_name("right_ankle_roll_link") == G1_RightToe,
          "right ankle roll maps to RightToe");
    check(g1_mesh_g1_bone_for_name("imu_in_pelvis") == -1,
          "fixed attachment has no motion bone");
}

static void test_fixed_attachment_follows_parent_bind_local()
{
    Fixture value = valid_fixture();
    check(g1_mesh_binding_build(
              value.binding, value.bones, value.bind_pose,
              G1MeshBoneCount, value.error, sizeof(value.error)),
          "valid binding builds");
    value.accepted_positions(G1_Spine2) = vec3(1.0f, 2.0f, 3.0f);
    value.accepted_rotations(G1_Spine2) =
        quat_from_angle_axis(0.5f, vec3(0.0f, 1.0f, 0.0f));
    check(g1_mesh_pose_build(
              value.output, G1MeshBoneCount, value.binding,
              value.accepted_positions, value.accepted_rotations,
              value.error, sizeof(value.error)),
          "finite accepted pose builds");
    check(transform_near(
              value.output[value.head_index],
              compose_transform(
                  value.output[value.torso_index],
                  value.binding.bind_local[value.head_index]),
              5.0e-6f),
          "head preserves authored torso-local bind transform");
}
```

Also test all 30 names exactly once, mapped-parent mismatch, missing bone,
duplicate bone, wrong count, NaN position, non-unit quaternion, and output
alias refusal. Add a rest-pose test that feeds each mapped bone its synthetic
global bind transform and requires every mapped and fixed output to reproduce
the full 39-bone global bind pose within `0.0005` m and quaternion
sign-equivalent tolerance. Verify a second unload of a canonical unloaded
renderer is a no-op. Read `controller.cpp` and require: the renderer uses only
`accepted_state.ik_global_bone_positions/rotations`; `M`, `B`, and `P` use
`IsKeyPressed`; mesh update occurs before `BeginDrawing`; mesh draw occurs
inside `BeginMode3D`; mesh unload occurs before `CloseWindow`; and no call
passes `working_state` or either candidate state. Require exactly one
controller cleanup call to `g1_mesh_renderer_unload`, and require both terrain
and mesh load failures to route through `normal_cleanup`.

- [ ] **Step 2: Compile and record RED before creating the renderer**

```bash
root=/tmp/g1-mesh-renderer
rm -rf "$root"
mkdir -p "$root"
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    -I. -I/home/ubuntu/apps/raylib/src \
    tests/cpp/test_g1_mesh_renderer.cpp \
    -L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 \
    -o "$root/test-g1-mesh-renderer" \
    >"$root/red.stdout" 2>"$root/red.stderr"; then
  echo "renderer RED unexpectedly compiled" >&2
  exit 1
fi
rg 'g1_mesh_renderer.h|g1_mesh_' "$root/red.stderr"
```

Expected: compilation fails because `g1_mesh_renderer.h` does not exist.

- [ ] **Step 3: Implement exact mapping and pure pose transfer**

Create `g1_mesh_renderer.h`. Define a 31-entry name array whose element zero
is `nullptr` and elements 1–30 are the exact GLB names in the design table.
`g1_mesh_binding_build` must:

```cpp
for (int model_bone = 0; model_bone < G1MeshBoneCount; ++model_bone) {
    output.model_to_g1[model_bone] =
        g1_mesh_g1_bone_for_name(bones[model_bone].name);
    output.parents[model_bone] = bones[model_bone].parent;
}
```

Require exactly one model bone for every `G1Bone` from `G1_Hips` through
`G1_RightWrist`, `pelvis` parent `-1`, and every other mapped model parent to
map to its exact production G1 parent. Require each fixed parent to be an
earlier valid bone. Derive fixed bind-local transforms from global bind pose:

```cpp
local.rotation = quat_to_raylib(quat_mul(
    quat_inv(raylib_to_quat(parent.rotation)),
    raylib_to_quat(child.rotation)));
local.translation = to_Vector3(quat_mul_vec3(
    quat_inv(raylib_to_quat(parent.rotation)),
    from_Vector3(child.translation) - from_Vector3(parent.translation)));
local.scale = Vector3{1.0f, 1.0f, 1.0f};
```

`g1_mesh_pose_build` validates exact input/output sizes, finite positions,
unit quaternions, and non-overlapping storage. Mapped bones copy accepted
global positions/quaternions. Fixed bones compose their already-built parent
with `bind_local`; every output scale is exactly one. It returns false without
partially publishing output by building into a local 39-transform array and
copying only after all validation succeeds.

- [ ] **Step 4: Implement Raylib resource ownership**

`g1_mesh_renderer_load` starts from a canonical unloaded value, calls
`LoadModel`, requires one or more meshes and exactly 39 bones, builds the
binding, allocates one `ModelAnimation` frame with `MemAlloc`, copies the model
bone array, verifies `IsModelAnimationValid`, and sets `loaded=true` only at
the end. A failure unloads any allocated animation/model before returning.

`g1_mesh_renderer_update` calls `g1_mesh_pose_build` into
`animation.framePoses[0]`, then calls `UpdateModelAnimation(model, animation,
0)`. Draw uses `DrawModel(model, Vector3{0,0,0}, 1.0f, WHITE)`. Unload calls
`UnloadModelAnimation` and `UnloadModel` exactly once when loaded, then resets
the entire renderer to its canonical default.

- [ ] **Step 5: Integrate mesh and diagnostic layers into `controller.cpp`**

Include `g1_mesh_renderer.h`. After the window is ready, declare a renderer
and the exact defaults:

```cpp
G1MeshRenderer g1_mesh_renderer = {};
bool show_g1_mesh = true;
bool show_g1_bones = true;
bool show_g1_sole_proxies = true;
```

Use `MM_G1_MESH_PATH` when nonempty; otherwise use
`resources/g1_mesh/g1_raylib.glb`. Load after the terrain model succeeds.
Add `g1_mesh_renderer_unload(g1_mesh_renderer);` inside `normal_cleanup`
before the terrain model unload and `CloseWindow`. Do not increment or change
the terrain `model_load_count` or `model_unload_count`.

In each frame, toggle only local booleans:

```cpp
if (::IsKeyPressed(KEY_M)) show_g1_mesh = !show_g1_mesh;
if (::IsKeyPressed(KEY_B)) show_g1_bones = !show_g1_bones;
if (::IsKeyPressed(KEY_P)) show_g1_sole_proxies = !show_g1_sole_proxies;
```

Before drawing, update from only the accepted pose. Inside `BeginMode3D`, draw
terrain, mesh when enabled, skeleton when enabled, and sole proxies when
enabled. The sole helper uses `g1_left_leg_config()` and
`g1_right_leg_config()`, transforms each of the four `sole_points_local` by
the accepted `config.contact` bone, and draws perimeter order `{0,1,3,2}`.
Use `GREEN` for `accepted_state.curr_bone_contacts(foot)` and `ORANGE`
otherwise. Add three 2D on/off labels and `M mesh  B bones  P sole proxies`.

- [ ] **Step 6: Run focused GREEN and strict/sanitized tests**

```bash
root=/tmp/g1-mesh-renderer
rayinc=(-I/home/ubuntu/apps/raylib/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. "${rayinc[@]}" \
  tests/cpp/test_g1_mesh_renderer.cpp -o "$root/test-strict" "${raylib[@]}"
"$root/test-strict"
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined -I. "${rayinc[@]}" \
  tests/cpp/test_g1_mesh_renderer.cpp -o "$root/test-sanitized" "${raylib[@]}"
ASAN_OPTIONS=detect_leaks=1 "$root/test-sanitized"
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_g1_mesh_asset -v
```

Expected: mapping, fixed propagation, negative cases, controller wiring,
sanitizers, and asset checks all pass without warnings.

- [ ] **Step 7: Build the production controller and run a finite terrain gate**

```bash
set -euo pipefail
root=/tmp/g1-mesh-renderer
strict=(-std=c++17 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off \
  -frounding-math -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread \
  -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp -o "$root/recovery.o"
g++ "${fast[@]}" "${rayinc[@]}" -c controller.cpp -o "$root/controller.o"
g++ "$root/controller.o" "$root/clearance.o" "$root/root-reach.o" \
  "$root/recovery.o" "${raylib[@]}" -o "$root/controller-g1-mesh"
DISPLAY=:1 MM_IK=0 "$root/controller-g1-mesh" \
  --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
  --terrain-scene mixed-multilevel --test-mode route \
  --test-route full-course --test-frames 32 \
  --test-heading forward --terrain-weight 4 \
  --log "$root/runtime.csv" --cleanup-log "$root/cleanup.json"
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py "$root/runtime.csv"
test "$(wc -l < "$root/runtime.csv")" -eq 33
```

Expected: GLB loads without a Raylib warning, 32 exact-25-Hz frames complete,
the unchanged runtime checker passes, and cleanup reports balanced terrain
model ownership plus a closed window. No persistent window is launched yet.

- [ ] **Step 8: Verify behavior scope and commit**

```bash
git diff --check
git diff --exit-code HEAD -- g1_frame_transaction.h g1_controller_state.h \
  g1_ik_runtime.h resources/check_g1_runtime_log.py
git add g1_mesh_renderer.h tests/cpp/test_g1_mesh_renderer.cpp controller.cpp
git diff --cached --check
git commit -m "feat: render accepted G1 mesh pose"
```

Expected: only renderer, focused tests, and controller integration are in the
commit; IK, transaction, checker, terrain, and motion assets are unchanged.

---

## Final Review and Candidate Launch

After both task reviews are clean, run the focused C++ strict/sanitized tests,
the mesh asset tests, the full Python discovery suite, and the 32-frame gate
again from a clean build. Generate a whole-range review package from
`2524cc1` through `HEAD` and obtain an independent final spec/quality review.

Then launch one separate persistent candidate from the reviewed binary:

```bash
cd /home/ubuntu/projects/motion-matching/.worktrees/g1-footprint-task6
DISPLAY=:1 MM_IK=0 /tmp/g1-mesh-renderer/controller-g1-mesh \
  --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
  --terrain-scene mixed-multilevel --test-mode live --terrain-weight 4
```

The old window remains untouched. The new window starts with mesh, bones, and
sole proxies visible; `M`, `B`, and `P` independently toggle those layers.
