# Grounded Reach Mesh Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a certified articulated G1 mesh, a `0.65 m` tabletop, and mouse orbit/pan/zoom controls to the existing grounded exhaustive reach viewer.

**Architecture:** The selected animated `interaction::WorldPose` is the only pose source for both the certified mesh renderer and optional blue skeleton. Scene scale and camera controls remain visual/query-front-end concerns; exhaustive search, grounding, IK, and collision semantics stay unchanged.

**Tech Stack:** C++17, Raylib 6, existing `g1_mesh_renderer.h`, certified `resources/g1_mesh/g1_raylib.glb`, Python `unittest`, Make.

## Global Constraints

- Mesh and skeleton consume the exact same 31-bone world pose with no adapter, scale, root offset, or alternate reference.
- Mesh is visible by default; `M` toggles mesh and `B` toggles skeleton overlay.
- Mesh failures fall back to the skeleton with an HUD diagnostic and do not crash the viewer.
- Main tabletop top is exactly `0.65 m`; rendered and collision geometry remain identical.
- Camera input is visual-only: left-drag orbit, middle-drag pan, wheel zoom.
- Preserve grounded root Y, existing IK files, collision gates, all 4,608 candidate identities, and explicit Enter search.
- No controller, terrain model, physics, diffusion, screenshot, or second viewer process.

---

## File Structure

- Modify `g1_reach_coverage_viewer.cpp`: scene constants, orbit camera state/input, mesh lifecycle, same-pose update/draw, toggles, and HUD fallback.
- Modify `g1_reach_coverage_probe.cpp`: lower-table fixtures derived from shared generated geometry.
- Modify `tests/python/test_g1_reach_coverage_viewer.py`: source contracts for table height, camera, mesh lifecycle, pose identity, and fallback.
- Modify `Makefile`: declare renderer/asset dependencies and retain the existing Raylib link path.
- Reuse `g1_mesh_renderer.h`, `g1_kinematic_contract.h`, `resources/g1_mesh/g1_raylib.glb`, and `tests/cpp/test_g1_mesh_renderer.cpp` unchanged.

### Task 1: Lowered Shared Scene and Inspectable Camera

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Modify: `g1_reach_coverage_probe.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: `interaction::make_coverage_environment`, Raylib mouse input, existing object keyboard controls.
- Produces: `kTableTop=0.65F`, shared table center Y `0.62F`, and `update_orbit_camera(Camera3D&, OrbitCameraState&)`.

- [ ] **Step 1: Add failing scene/camera source-contract tests**

Add a test requiring these exact viewer elements:

```python
def test_lowered_scene_and_mouse_camera_contract(self):
    source = self.source(VIEWER)
    for required in (
        "kTableTop = 0.65F", "kTableCenterY = 0.62F",
        "OrbitCameraState", "update_orbit_camera(",
        "GetMouseDelta()", "GetMouseWheelMove()",
        "MOUSE_BUTTON_LEFT", "MOUSE_BUTTON_MIDDLE",
        "std::clamp", "camera.target",
    ):
        self.assertIn(required, source)
    self.assertNotIn("constexpr float table_top = 0.77F", source)
```

Extend the probe contract to require `kTableTop = 0.65F` and fixture positions derived from `coverage.boxes` rather than the old hard-coded table/shelf/lower Y values.

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
```

Expected: FAIL because the viewer still uses the fixed camera and `0.77 m` tabletop.

- [ ] **Step 3: Introduce shared scene constants**

At file scope in viewer and probe define:

```cpp
constexpr float kTableTop = 0.65F;
constexpr float kTableThickness = 0.06F;
constexpr float kTableCenterY = kTableTop - 0.5F * kTableThickness;
constexpr vec3 kTableDimensions(1.20F, kTableThickness, 0.75F);
```

Use `{vec3(0.0F, kTableCenterY, 0.0F), quat()}` everywhere the main table is created. Make `reset_object` place the object at `kTableTop + 0.5F * dimensions.y`.

In the probe, derive supported object centers from generated boxes:

```cpp
const auto supported_center = [](const interaction::OrientedBox& support) {
    return support.world.position +
        vec3(0.0F, 0.5F * support.dimensions.y + 0.05F, 0.0F);
};
```

Use box `0` for the main table, box `5` for the shelf board, and box `8` for the lower tabletop. Keep open-space and below-table fixtures explicit but relative to `kTableTop`.

- [ ] **Step 4: Implement bounded mouse camera control**

Add:

```cpp
struct OrbitCameraState {
    vec3 target{0.0F, 0.70F, 0.0F};
    float azimuth = -0.80F;
    float altitude = 0.32F;
    float distance = 3.50F;
};
```

`update_orbit_camera` reads one `GetMouseDelta()` and one
`GetMouseWheelMove()` per frame. Left drag changes azimuth/altitude with
altitude clamped to `[-1.30, 1.30]`. Middle drag pans along normalized camera
right/up vectors. Wheel scales distance and clamps it to `[0.60, 8.0]`.
Finally derive `camera.position`, `camera.target`, and world-up without touching
object/query state.

- [ ] **Step 5: Run source tests and warning-clean syntax**

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -pthread -I. \
  -I.deps/raylib/src -fsyntax-only g1_reach_coverage_viewer.cpp
```

Expected: PASS with no compiler diagnostics.

- [ ] **Step 6: Commit**

```bash
git add g1_reach_coverage_viewer.cpp g1_reach_coverage_probe.cpp \
  tests/python/test_g1_reach_coverage_viewer.py
git commit -m "feat: lower and navigate reach coverage scene"
```

### Task 2: Certified Mesh on the Exact Selected Pose

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`
- Modify: `Makefile`
- Test: `tests/cpp/test_g1_mesh_renderer.cpp`

**Interfaces:**
- Consumes: `interaction::WorldPose`, `g1_mesh_renderer_load/update/draw/unload`.
- Produces: mesh-default rendering, `M`/`B` toggles, skeleton fallback, and renderer diagnostics.

- [ ] **Step 1: Replace the obsolete no-mesh test with failing integration contracts**

Remove `"g1_mesh_renderer"` and `"LoadModel("` from the forbidden list. Require:

```python
for required in (
    '#include "g1_mesh_renderer.h"',
    '"resources/g1_mesh/g1_raylib.glb"',
    "bool show_g1_mesh = true", "bool show_g1_bones = false",
    "IsKeyPressed(KEY_M)", "IsKeyPressed(KEY_B)",
    "::g1_mesh_renderer_load(", "::g1_mesh_renderer_update(",
    "::g1_mesh_renderer_draw(", "::g1_mesh_renderer_unload(",
    "mesh_world_pose.positions", "mesh_world_pose.rotations",
    "G1 MESH DISABLED", "show_g1_bones = true",
):
    self.assertIn(required, source)
```

Assert update occurs before `BeginDrawing`, draw occurs inside `BeginMode3D`,
and unload precedes `CloseWindow`.

- [ ] **Step 2: Run the test to verify RED**

```bash
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests.test_flat_viewer_contract -v
```

Expected: FAIL because the mesh renderer is not integrated.

- [ ] **Step 3: Add safe renderer lifecycle and toggles**

After `InitWindow`, create `G1MeshRenderer g1_mesh_renderer{}`, load the GLB,
and retain a `std::array<char, 256> g1_mesh_error`. On load failure set
`show_g1_mesh=false`, `show_g1_bones=true`, and retain the message rather than
throwing.

Handle exactly one `KEY_M` and one `KEY_B` toggle. Wrap the render loop so both
normal exit and exceptions call:

```cpp
::g1_mesh_renderer_unload(g1_mesh_renderer);
CloseWindow();
```

before the exception reaches the existing outer diagnostic.

- [ ] **Step 4: Update and draw from one world pose**

For the selected animation sample compute:

```cpp
const interaction::WorldPose mesh_world_pose =
    interaction::world_pose(evaluation.poses[sample]);
```

Before `BeginDrawing`, if mesh is enabled call update with:

```cpp
slice1d<vec3>(
    static_cast<int>(g1_skeleton::BoneCount),
    mesh_world_pose.positions.data())
slice1d<quat>(
    static_cast<int>(g1_skeleton::BoneCount),
    mesh_world_pose.rotations.data())
```

On update failure disable mesh, enable bones, and retain the error. Inside the
3D pass draw the mesh when enabled and call the existing skeleton drawing only
when `show_g1_bones` is true. Do not recompute or alter the pose for either
renderer.

- [ ] **Step 5: Add HUD state and build dependencies**

Show `M mesh ON/OFF | B bones ON/OFF`. When disabled by a renderer error show
`G1 MESH DISABLED: <message>` in maroon.

Add `g1_mesh_renderer.h g1_kinematic_contract.h array.h
resources/g1_mesh/g1_raylib.glb` to the viewer target dependencies without
adding any new link library.

- [ ] **Step 6: Run renderer and integration gates**

```bash
make build/tests/test_g1_mesh_renderer g1_reach_coverage_viewer
build/tests/test_g1_mesh_renderer resources/g1_mesh/g1_raylib.glb
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
```

Expected: renderer test, source contracts, and release build PASS.

- [ ] **Step 7: Commit**

```bash
git add g1_reach_coverage_viewer.cpp \
  tests/python/test_g1_reach_coverage_viewer.py Makefile
git commit -m "feat: render grounded reach motion on G1 mesh"
```

### Task 3: Real Evidence, Review, and One-Viewer Handoff

**Files:**
- Modify: `.superpowers/sdd/progress.md`
- Generate: `build/g1-reaches/contact-anchored-coverage-report-v3.json` (do not commit)

**Interfaces:**
- Consumes: Tasks 1-2 and the real v2 reach pack.
- Produces: reviewed final commits and exactly one running mesh viewer.

- [ ] **Step 1: Run the complete automated gate**

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_g1_mesh_renderer build/tests/test_reach_search \
  build/tests/test_reach_coverage build/tests/test_reach_database \
  build/tests/test_interaction_hand_trajectories \
  build/tests/test_interaction_ik g1_reach_coverage_probe \
  g1_reach_coverage_viewer
build/tests/test_g1_mesh_renderer resources/g1_mesh/g1_raylib.glb
build/tests/test_reach_search
build/tests/test_reach_coverage
build/tests/test_reach_database
build/tests/test_interaction_hand_trajectories
build/tests/test_interaction_ik
git diff --exit-code 50f5542 -- \
  interaction_ik.h interaction_ik.cpp g1_arm_joint_metadata.h
```

Expected: all tests/builds pass and unchanged-IK diff emits no output.

- [ ] **Step 2: Regenerate lowered-table evidence**

```bash
./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 \
  --json build/g1-reaches/contact-anchored-coverage-report-v3.json
jq '{search_integrity_passed, fixtures: (.shared_grasps | \
  map_values({coverage_demonstrated, accepted, elapsed_seconds}))}' \
  build/g1-reaches/contact-anchored-coverage-report-v3.json
```

Expected: all fixtures process 4,608/4,608 within 30 seconds. Coverage counts
may change because the table is lower; do not relax IK or collision gates.

- [ ] **Step 3: Record ledger and commit**

Append mesh/camera/table commits, exact test commands, report counts, and
unchanged-IK evidence to `.superpowers/sdd/progress.md`, then:

```bash
git add -f .superpowers/sdd/progress.md
git commit -m "docs: record grounded mesh viewer evidence"
```

- [ ] **Step 4: Request focused whole-feature review**

Review from `fc0b435` through `HEAD` for same-pose mesh correctness, lifecycle,
fallback, camera isolation, table/render/collision consistency, grounded-root
preservation, memory, and tests. Resolve every Critical/Important issue with a
failing test first.

- [ ] **Step 5: Replace only the current flat viewer**

Resolve `pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'`, terminate only that
exact PID, verify none remains, then launch exactly one:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v2
```

Verify one matching PID and bounded RSS with `ps -o pid,rss,etime,cmd`.
