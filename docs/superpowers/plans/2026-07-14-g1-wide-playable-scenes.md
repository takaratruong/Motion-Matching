# G1 Wide Playable Scenes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand every G1 terrain scene to at least six metres of lateral playable floor without changing its central terrain, routes, spawn, or shared motion pack.

**Architecture:** Keep the existing `0.60 m` feature half-width and add a separate `3.0 m` playable half-width in the scene-definition layer. Rasterize the same surfaces over wider X bounds, classify only proven-flat added side aprons as certified, and rebuild/promote only the ignored scene catalog while preserving every authenticated motion payload byte.

**Tech Stack:** Python 3, NumPy, `unittest`, G1HF/v2 heightfields, G1WM/v1 walkability, canonical JSON/OBJ scene artifacts, the existing C++17/Raylib runtime.

## Global Constraints

- Cover exactly the fourteen IDs in `REQUIRED_SCENE_IDS`; do not add duplicate sandbox scenes.
- `PLAYABLE_HALF_WIDTH == 3.0`, `COURSE_HALF_WIDTH == 0.60`, `SCENE_CELL_SIZE == 0.02`, `LOOKAHEAD_MARGIN == 1.0`, and `WALKABILITY_CLASSIFICATION_HALO == 0.25` remain exact source constants.
- Preserve every existing route waypoint/outcome/hold, spawn transform, procedural feature parameter, GRAIL source transform, fixed diagonal, and Z bound.
- Known-flat new side cells are class `1`; central class `1`/`2` and direct blocked routes retain their existing meanings.
- Never modify or stage `resources/database.bin`, `resources/features.bin`, or a shared motion payload under `resources/g1_terrain`.
- The authoritative v2 database must remain SHA-256 `1849ecbc3775fda0a7cb2fb1bfe9ed15d0f457bd86d977d60bbe6de8b0a0fed6`.
- Preserve the live old visualizer until a replacement build has passed scene loading; do not add the optional G1 mesh in this plan.
- Follow TDD: observe every focused RED before implementation, then obtain focused and full GREEN before each commit.

---

### Task 1: Separate Feature Width from Playable Width

**Files:**
- Modify: `resources/g1_terrain_builder/scenes.py:235-375,797-866`
- Modify: `tests/python/test_scenes.py:690-914,914-1040`

**Interfaces:**
- Consumes: existing `SceneDefinition`, `_runtime_f32`, `_runtime_f32_upper_ceiling`, `COURSE_HALF_WIDTH`, and `LOOKAHEAD_MARGIN`.
- Produces: `PLAYABLE_HALF_WIDTH = 3.0`, `_wide_playable_x_bounds(core_bounds, spawn_x) -> tuple[float, float]`, and `_wide_heightfield_bounds(core_bounds, playable_bounds) -> tuple[float, float, float, float]`.

- [ ] **Step 1: Add failing width and central-geometry tests**

Import `PLAYABLE_HALF_WIDTH` in `tests/python/test_scenes.py`. Add these assertions to `GrailSceneTests` and `ProceduralSceneTests`:

```python
def assert_wide_x_contract(test, scene):
    px0, px1, _, _ = scene.playable_bounds_xz
    hx0, hx1, _, _ = scene.heightfield_bounds_xz
    test.assertGreaterEqual(px1 - px0, 2.0 * PLAYABLE_HALF_WIDTH)
    test.assertLessEqual(px0, scene.spawn_position[0] - PLAYABLE_HALF_WIDTH)
    test.assertGreaterEqual(px1, scene.spawn_position[0] + PLAYABLE_HALF_WIDTH)
    test.assertGreaterEqual(px0 - hx0, LOOKAHEAD_MARGIN)
    test.assertGreaterEqual(hx1 - px1, LOOKAHEAD_MARGIN)


def test_all_procedural_scenes_have_six_metre_playable_floor(self):
    for scene_id, scene in self.definitions.items():
        with self.subTest(scene=scene.scene_id):
            assert_wide_x_contract(self, scene)
            self.assertEqual(scene.playable_bounds_xz[:2], (-3.0, 3.0))
            width_key = "lane_width_m" if scene_id == "blocked-course" \
                else "width_m"
            self.assertEqual(
                scene.provenance["parameters"][width_key], 1.2)
            z = 0.5 * (
                scene.playable_bounds_xz[2] +
                scene.playable_bounds_xz[3])
            self.assertEqual(scene.surface.height(-2.0, z), 0.0)
            self.assertEqual(scene.surface.height(+2.0, z), 0.0)


def test_grail_scenes_union_old_route_envelope_with_six_metre_floor(self):
    clip = fake_grail_clip(GRAIL_DEFAULT_BASE)
    scene = grail_scene_definition(
        "grail-curb-default", GRAIL_DEFAULT_BASE, clip, None)
    assert_wide_x_contract(self, scene)
    route_x = [point[0] for point in scene.routes[0].waypoints_xz]
    self.assertLessEqual(scene.playable_bounds_xz[0], min(route_x) - 0.6)
    self.assertGreaterEqual(scene.playable_bounds_xz[1], max(route_x) + 0.6)
```

Also add a procedural runtime-reference test. Rasterize the same surface over the prior narrow X bounds (`[-1.6,1.6]`, or `[-2.4,2.4]` for `blocked-course`) and over the new bounds. At all route points and the existing stair/ramp/join probes, require `abs(wide.height(x,z)-narrow.height(x,z)) <= 1e-6`.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes.GrailSceneTests \
  tests.python.test_scenes.ProceduralSceneTests -v
```

Expected: the new width assertions fail because corridor scenes still publish `[-0.6,+0.6]`, blocked course publishes `[-1.4,+1.4]`, and GRAIL uses only its route envelope.

- [ ] **Step 3: Implement the checked bound helpers**

Add next to the existing constants in `scenes.py`:

```python
COURSE_HALF_WIDTH = 0.60
PLAYABLE_HALF_WIDTH = 3.0
FLAT_SPAWN_LENGTH = 2.0
LOOKAHEAD_MARGIN = 1.0


def _runtime_f32_lower_floor(value, label):
    target = _finite_real(value, label)
    result = _runtime_f32(target, label)
    if result > target:
        result = _runtime_f32(
            np.nextafter(np.float32(result), np.float32(-np.inf)),
            f"{label} lower floor")
    return result


def _wide_playable_x_bounds(core_bounds, spawn_x):
    core = _strict_bounds(core_bounds, "core playable")
    spawn = _runtime_f32(spawn_x, "wide playable spawn x")
    lower = _runtime_f32_lower_floor(
        min(core[0], spawn - PLAYABLE_HALF_WIDTH),
        "wide playable xmin")
    upper = _runtime_f32_upper_ceiling(
        max(core[1], spawn + PLAYABLE_HALF_WIDTH),
        "wide playable xmax")
    return lower, upper


def _wide_heightfield_bounds(core_bounds, playable_bounds):
    core = _strict_bounds(core_bounds, "core heightfield")
    playable = _strict_bounds(playable_bounds, "wide playable")
    xmin = _runtime_f32_lower_floor(
        min(core[0], playable[0] - LOOKAHEAD_MARGIN),
        "wide heightfield xmin")
    xmax = _runtime_f32_upper_ceiling(
        max(core[1], playable[1] + LOOKAHEAD_MARGIN),
        "wide heightfield xmax")
    return xmin, xmax, core[2], core[3]
```

In `_corridor_definition`, keep `core_playable = (-COURSE_HALF_WIDTH, COURSE_HALF_WIDTH, 0.0, course_end_z)`, derive wide X around spawn zero, then derive expanded heightfield bounds from the old `±(COURSE_HALF_WIDTH + LOOKAHEAD_MARGIN)` core bounds. Apply the same pattern to `_blocked_definition`, retaining its old `[-2.4,+2.4]` core heightfield. In `grail_scene_definition`, first retain the old path-derived `core_playable`, then publish the union with `spawn[0] ± 3.0`; pass the old mesh/core bounds through `_wide_heightfield_bounds`.

Do not change `LongitudinalProfileSurface.half_width`, `CrossSlopeSurface.half_width`, `BlockedCourseSurface`, route construction, or provenance feature widths.

- [ ] **Step 4: Run focused and schema tests GREEN**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes.GrailSceneTests \
  tests.python.test_scenes.ProceduralSceneTests \
  tests.python.test_scenes.SceneSchemaTests -v
```

Expected: all tests pass; every published playable X interval is at least six metres, central runtime probes remain within `1e-6 m`, and feature widths/routes/spawns are unchanged.

- [ ] **Step 5: Review and commit the width separation**

```bash
git diff --check
git add resources/g1_terrain_builder/scenes.py tests/python/test_scenes.py
git diff --cached --check
git commit -m "feat: widen G1 scene playable bounds"
```

Expected: the commit contains only the scene builder and scene tests.

---

### Task 2: Certify the Flat Side Aprons Without Weakening Central Routes

**Files:**
- Modify: `resources/g1_terrain_builder/scenes.py:355-690,797-866`
- Modify: `tests/python/test_scenes.py:812-914,1020-1185`
- Modify: `resources/validate_g1_terrain_database.py`
- Modify: `tests/python/test_validator.py`

**Interfaces:**
- Consumes: Task 1 wide bounds, `_walkability_classification_bounds`, `_region`, and each scene's existing core/source X envelope.
- Produces: class-1 flat aprons, central stress retention, class-1 blocked-course bypass lanes, and an exact validator oracle for the expanded deterministic region model while keeping both direct safe-stop routes class `0` at their endpoints.

- [ ] **Step 1: Add failing class and region tests**

Add table-driven probes:

```python
def test_procedural_side_aprons_are_certified(self):
    for scene_id, scene in self.definitions.items():
        z = min(2.25, scene.playable_bounds_xz[3] - 0.25)
        for x in (-2.5, 2.5):
            with self.subTest(scene=scene_id, x=x):
                self.assertEqual(scene.surface.height(x, z), 0.0)
                self.assertEqual(scene.walkability(x, z), 1)
    stress = self.definitions["ramp-15-stress"]
    self.assertEqual(stress.walkability(0.0, 2.25), 2)
    self.assertEqual(stress.walkability(-2.5, 2.25), 1)
    self.assertEqual(stress.walkability(+2.5, 2.25), 1)


def test_blocked_course_has_two_certified_outer_bypasses(self):
    scene = self.definitions["blocked-course"]
    for z in (1.0, 2.25, scene.playable_bounds_xz[3] - 0.25):
        self.assertEqual(scene.walkability(-2.5, z), 1)
        self.assertEqual(scene.walkability(+2.5, z), 1)
    self.assertEqual(scene.walkability(-0.8, 2.25), 0)
    self.assertEqual(scene.walkability(+0.8, 2.25), 0)
    self.assertEqual(scene.walkability(0.0, 2.25), 0)
```

For each GRAIL definition, compute `mesh_xmin, mesh_xmax` from `scene.surface.xz_bounds()`. Require old route points to retain `route.walkability_class`; require probes at `playable_xmin + 0.25` and `playable_xmax - 0.25` to be class `1` only when strictly outside the mesh X extrema. Build every scene and reuse `_region_cell_indices`, `_route_cell_covers`, and `decoded_footprint_classes` to prove every published region is class-pure and every route retains its prior class sequence.

- [ ] **Step 2: Run focused tests and observe RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes.GrailSceneTests.test_grail_g1wm_regions_routes_and_endpoint_footprints_are_class_pure \
  tests.python.test_scenes.ProceduralSceneTests.test_procedural_side_aprons_are_certified \
  tests.python.test_scenes.ProceduralSceneTests.test_blocked_course_has_two_certified_outer_bypasses -v
```

Expected: stress-scene aprons still return `2`, and the blocked course still marks every cell past the obstacle threshold as `0`.

- [ ] **Step 3: Implement explicit core/apron classifiers**

For `_corridor_definition`, classify the outer halo first. Class-1 scenes return `1` throughout it. For the 15-degree stress scene, return `2` only when `-COURSE_HALF_WIDTH <= x <= COURSE_HALF_WIDTH`; return `1` on the proven-flat sides. Publish a central stress rectangle and two certified side rectangles.

Region bounds are interpreted by `_region_cell_indices` through nearest-node mapping. A mere binary32 predecessor/successor of a class threshold can still map to the same G1WM node, so derive the last apron node, first core node, last core node, and first opposite-apron node from the authoritative heightfield origin and `SCENE_CELL_SIZE`. Publish those four runtime binary32 node coordinates and test that the three region covers are class-pure, disjoint, and adjacent by cell index.

For `_blocked_definition`, retain the exact obstacle Z threshold and use:

```python
wall_min_x = surface.wall_center_x - surface.lane_half_width
ramp_max_x = surface.ramp_center_x + surface.lane_half_width

def walkability(x, z):
    if not _bounds_contains(classification, x, z):
        return 0
    if z <= obstacle_start - SCENE_CELL_SIZE:
        return 1
    if x < wall_min_x or x > ramp_max_x:
        return 1
    return 0
```

Publish the full-width approach plus left/right certified bypass regions. Keep `wall`, `ramp`, and the narrow central gap represented as blocked regions. Do not change either safe-stop route.

For `grail_scene_definition`, retain `core_playable`, `mesh_xmin`, and `mesh_xmax`. The old playable rectangle and the source-mesh X envelope use the old scene class; newly added X cells strictly outside both use class `1`. Certified GRAIL scenes may publish one full-width certified region. Stress GRAIL scenes publish a central stress rectangle and class-1 left/right exterior rectangles.

Update the independent validator's deterministic region oracle at the same time. It must require the exact approved IDs/classes/adjacent grid partitions; accepting an arbitrary region count is not allowed. This compatibility update is required before candidate packs using the new scene model can validate.

- [ ] **Step 4: Run all scene tests GREEN**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes -v
```

Expected: all scene schema, route supercover, region purity, halo, deterministic-byte, GRAIL, and procedural tests pass.

- [ ] **Step 5: Run adjacent Python suites GREEN**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain \
  tests.python.test_artifacts \
  tests.python.test_validator \
  tests.python.test_build_cli -v
```

Expected: all tests pass; the authenticated schemas and atomic artifact behavior remain unchanged.

- [ ] **Step 6: Review and commit apron classification**

```bash
git diff --check
git add resources/g1_terrain_builder/scenes.py tests/python/test_scenes.py \
  resources/validate_g1_terrain_database.py tests/python/test_validator.py \
  docs/superpowers/plans/2026-07-14-g1-wide-playable-scenes.md
git diff --cached --check
git commit -m "feat: certify wide G1 scene aprons"
```

Expected: only the scene builder/tests, exact validator oracle/tests, and this execution-plan correction are committed.

---

### Task 3: Rebuild and Atomically Promote Only the Scene Pack

**Files:**
- Generate, ignored: `/tmp/g1-terrain-wide-candidate-v1/`
- Replace atomically, ignored: `resources/g1_terrain/scenes/`
- Update inside the atomic candidate, ignored: `resources/g1_terrain/manifest.json`
- Preserve: every other file in `resources/g1_terrain/`

**Interfaces:**
- Consumes: Task 2 `build_scene_pack(all_scene_definitions(...))` output and the existing authenticated v2 pack.
- Produces: a validated authoritative scene catalog with new hashes while every motion payload hash remains unchanged.

- [ ] **Step 1: Record protected hashes and current scene bounds**

```bash
sha256sum \
  resources/database.bin \
  resources/features.bin \
  resources/g1_terrain/database.bin \
  resources/g1_terrain/terrain_features.bin \
  resources/g1_terrain/terrain_support.bin
for file in resources/g1_terrain/scenes/*/scene.json
do
  jq -r '[.id,.bounds.playable_min_xz[0],.bounds.playable_max_xz[0]]|@tsv' "$file"
done
```

Expected: the authoritative database hash is `1849ecbc...0a0fed6`; procedural scenes still show the narrow baseline before promotion. Save the complete command output in the task review notes.

- [ ] **Step 2: Build a diagnostic candidate to a separate directory**

Require `/tmp/g1-terrain-wide-candidate-v1` not to exist, then run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py \
  --grail-limit 0 \
  --output /tmp/g1-terrain-wide-candidate-v1
```

Expected: `BUILT g1-terrain-artifacts/v2 ... scenes=14`; only Takara motion is built in this disposable diagnostic pack, while all four selected GRAIL scene definitions are still constructed from their route sources. Do not point this command at `resources/g1_terrain`.

- [ ] **Step 3: Validate candidate scene widths and deterministic hashes**

```bash
for file in /tmp/g1-terrain-wide-candidate-v1/scenes/*/scene.json
do
  jq -e '(.bounds.playable_max_xz[0]-.bounds.playable_min_xz[0]) >= 6.0' "$file" >/dev/null
done
sha256sum /tmp/g1-terrain-wide-candidate-v1/scenes/index.json
```

The byte-exact repeated-build test in `ProceduralSceneTests` and the canonical scene-pack tests from Task 2 own repeat determinism, so do not perform a second corpus scan here.

- [ ] **Step 4: Stage a hard-linked authoritative pack with only scenes replaced**

Run this repository-root script exactly. It refuses pre-existing scratch state, hard-links immutable payloads, copies only candidate scenes, updates only the authenticated scene-index digest, validates the complete 708 MB candidate, and atomically exchanges directories:

```python
import hashlib
import json
import os
import shutil
import subprocess

from resources.g1_terrain_builder.artifacts import (
    _fsync_parent, _fsync_tree, _locked_parent, _rename_exchange,
    canonical_json_bytes,
)

root = os.path.abspath("resources/g1_terrain")
candidate = "/tmp/g1-terrain-wide-candidate-v1/scenes"
backup = os.path.abspath("resources/.g1_terrain-wide-backup")
if os.path.lexists(backup):
    raise RuntimeError(f"refusing pre-existing backup path: {backup}")
shutil.copytree(root, backup, copy_function=os.link)
shutil.rmtree(os.path.join(backup, "scenes"))
shutil.copytree(candidate, os.path.join(backup, "scenes"))

manifest_path = os.path.join(backup, "manifest.json")
with open(manifest_path, "rb") as stream:
    manifest = json.load(stream)
with open(os.path.join(backup, "scenes", "index.json"), "rb") as stream:
    index_bytes = stream.read()
manifest["scene_index"]["sha256"] = hashlib.sha256(index_bytes).hexdigest()
temporary_manifest = manifest_path + ".new"
with open(temporary_manifest, "xb") as stream:
    stream.write(canonical_json_bytes(manifest))
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary_manifest, manifest_path)

expected = {
    "database.bin": "1849ecbc3775fda0a7cb2fb1bfe9ed15d0f457bd86d977d60bbe6de8b0a0fed6",
}
for name, digest in expected.items():
    with open(os.path.join(backup, name), "rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != digest:
        raise RuntimeError(f"protected hash changed for {name}: {observed}")

subprocess.run([
    "/home/ubuntu/miniconda3/envs/diffsim/bin/python",
    "resources/validate_g1_terrain_database.py", backup,
], check=True)
_fsync_tree(backup)
parent = os.path.dirname(root)
with _locked_parent(parent) as parent_descriptor:
    _rename_exchange(backup, root)
    _fsync_parent(parent_descriptor)
print(f"PROMOTED scenes; rollback pack retained at {backup}")
```

Execute it through the Python interpreter without saving a source file. Expected: validator reports `VALID ... scenes=14`; atomic exchange succeeds; `resources/.g1_terrain-wide-backup` now contains the complete old pack for rollback.

- [ ] **Step 5: Revalidate protected hashes and published metadata**

```bash
sha256sum \
  resources/database.bin \
  resources/features.bin \
  resources/g1_terrain/database.bin \
  resources/g1_terrain/terrain_features.bin \
  resources/g1_terrain/terrain_support.bin
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
for file in resources/g1_terrain/scenes/*/scene.json
do
  jq -e '(.bounds.playable_max_xz[0]-.bounds.playable_min_xz[0]) >= 6.0' "$file" >/dev/null
done
```

Expected: all five protected hashes exactly match Step 1; validator succeeds; all fourteen width checks succeed.

---

### Task 4: Load Every Wide Scene and Hand Back to the IK Plan

**Files:**
- Build only: `/tmp/controller_g1_wide`
- Generate only: `/tmp/g1-wide-scene-cycle.csv`, `/tmp/g1-wide-scene-cycle-cleanup.json`
- Preserve running old executable: `controller_g1_terrain_task10`

**Interfaces:**
- Consumes: the promoted wide scene catalog.
- Produces: native load/switch evidence, an interactive render smoke, and a clean checkpoint before resuming Task 3/4 of `2026-07-13-g1-terrain-ik-clearance.md`.

- [ ] **Step 1: Run native scene/runtime tests**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_wide
/tmp/test_scene_runtime_wide
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_switch.cpp -o /tmp/test_scene_switch_wide
/tmp/test_scene_switch_wide
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime_wide
/tmp/test_terrain_runtime_wide
```

Expected: every executable exits `0` without output.

- [ ] **Step 2: Build the production controller**

```bash
g++ -O3 -ffast-math -march=native -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -I /home/ubuntu/apps/raylib/src \
  -I /home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_g1_wide -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected: exit `0` and executable `/tmp/controller_g1_wide`.

- [ ] **Step 3: Load all fourteen scenes through two ordered cycles**

```bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=700 \
  MM_TERRAIN_WEIGHT=4 \
  MM_LOG=/tmp/g1-wide-scene-cycle.csv \
  MM_CLEANUP_LOG=/tmp/g1-wide-scene-cycle-cleanup.json \
  /tmp/controller_g1_wide
```

Run `resources/check_g1_runtime_log.py --gate-f` with the exact comma-separated `REQUIRED_SCENE_IDS` order. Expected: `complete_cycles=2`, 28 generations, one motion-pack load, finite rows, successful cleanup, and every scene loaded/unloaded exactly twice. Gate F intentionally requires at least two complete ordered cycles; a 350-frame single-cycle run is only a loader smoke and does not satisfy the gate.

- [ ] **Step 4: Perform the 60-second rendered smoke without killing the old visualizer**

Launch `/tmp/controller_g1_wide` on `DISPLAY=:1` using the default scene and observe the built-in FPS counter for sixty seconds. Require median visible FPS at least `25`, a six-metre floor visible on both sides, unchanged central curb/stair/ramp proportions, and no bounds-induced safe stop while turning and walking on both aprons. Do not terminate PID `2203848` until a replacement visualizer is explicitly requested and the new process is confirmed alive.

- [ ] **Step 5: Accept or roll back the scene pack**

If any load, performance, geometry, or manual-navigation check fails, atomically exchange `resources/.g1_terrain-wide-backup` with `resources/g1_terrain` using `_rename_exchange`, retain the failed candidate for diagnosis, and keep the old visualizer.

If all checks pass, remove only the owned `resources/.g1_terrain-wide-backup`, retain the generated wide scenes, and record final artifact sizes/hashes. The source commits are already complete; ignored scene artifacts are not staged.

- [ ] **Step 6: Resume the terrain-IK/clearance plan**

Update the working task list: mark the wide-map insertion complete, return to the in-progress bounded named-leg Task 3 review/fix cycle, then continue Tasks 4-9. Keep optional G1 meshes deferred until Task 10 at the absolute end.

---

## Final Verification Matrix

Before declaring the wide-map insertion complete, require all of:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes \
  tests.python.test_terrain \
  tests.python.test_artifacts \
  tests.python.test_validator \
  tests.python.test_build_cli -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
git diff --check
git status --short
```

Expected: all tests and validator pass; source worktree changes add no new protected-payload modification beyond the user's pre-existing `resources/database.bin` and `resources/features.bin` state; ignored scene assets are wider; all recorded protected hashes remain exact; the old visualizer remains available until replacement.
