# Diffusion Pickup G1 Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the certified articulated G1 robot mesh from the terrain-aware visualizer in the diffusion pickup runtime.

**Architecture:** Stage the already-certified GLB renderer as a self-contained component, then connect it to the diffusion controller's final `global_bone_positions` and `global_bone_rotations`. Keep the diffusion, interaction, attachment, and locomotion paths unchanged; this is a presentation-only consumer of the accepted pose.

**Tech Stack:** C++17, Raylib `Model`/`ModelAnimation`, Python `unittest`, existing 31-bone G1 runtime.

## Global Constraints

- Use `resources/g1_mesh/g1_raylib.glb` and `g1_mesh_renderer.h` from certified commit `6de5f87`.
- Show the robot mesh by default and hide the diagnostic bone overlay by default.
- `M` toggles robot mesh visibility and `B` toggles diagnostic bones.
- Update only from the final global pose; do not use adjusted, predicted, or candidate poses.
- Do not modify `resources/features.bin` or generated diffusion artifacts.
- Do not alter diffusion generation, route following, interaction state, attachment, or placement behavior.

---

### Task 1: Stage and prove the certified renderer package

**Files:**
- Create: `g1_mesh_renderer.h`
- Create: `g1_kinematic_contract.h`
- Create: `resources/g1_mesh/__init__.py`
- Create: `resources/g1_mesh/export_g1_raylib_glb.py`
- Create: `resources/g1_mesh/g1_raylib.glb`
- Create: `resources/g1_mesh/manifest.json`
- Create: `resources/g1_mesh/validate_g1_raylib_glb.py`
- Test: `tests/cpp/test_g1_mesh_renderer.cpp`
- Test: `tests/python/test_g1_mesh_asset.py`

**Interfaces:**
- Consumes: Raylib `Model`/`ModelAnimation`, `slice1d<vec3>`, `slice1d<quat>`, and the 31-bone G1 hierarchy.
- Produces: `G1MeshRenderer`, `g1_mesh_renderer_load`, `g1_mesh_renderer_update`, `g1_mesh_renderer_draw`, and `g1_mesh_renderer_unload`.

- [ ] **Step 1: Restore only the focused tests first**

  ```bash
  git restore --source 6de5f87 -- \
    tests/cpp/test_g1_mesh_renderer.cpp \
    tests/python/test_g1_mesh_asset.py
  ```

- [ ] **Step 2: Verify RED because the renderer and asset are absent**

  ```bash
  mkdir -p /tmp/diffusion-pickup-g1-mesh-build
  g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
    -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
    tests/cpp/test_g1_mesh_renderer.cpp \
    -o /tmp/diffusion-pickup-g1-mesh-build/test-g1-mesh-renderer \
    -L/home/ubuntu/apps/raylib/src \
    -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
  ```

  Expected: compilation fails with `g1_mesh_renderer.h: No such file or directory`.

- [ ] **Step 3: Restore the minimal certified renderer and complete asset package**

  ```bash
  git restore --source 6de5f87 -- \
    g1_mesh_renderer.h g1_kinematic_contract.h resources/g1_mesh
  ```

- [ ] **Step 4: Verify GREEN with component and asset tests**

  ```bash
  g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
    -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
    tests/cpp/test_g1_mesh_renderer.cpp \
    -o /tmp/diffusion-pickup-g1-mesh-build/test-g1-mesh-renderer \
    -L/home/ubuntu/apps/raylib/src \
    -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
  /tmp/diffusion-pickup-g1-mesh-build/test-g1-mesh-renderer
  python3 -m unittest tests.python.test_g1_mesh_asset -v
  ```

  Expected: renderer exits 0 and all asset tests pass.

- [ ] **Step 5: Commit the renderer package**

  ```bash
  git add g1_mesh_renderer.h g1_kinematic_contract.h resources/g1_mesh \
    tests/cpp/test_g1_mesh_renderer.cpp tests/python/test_g1_mesh_asset.py
  git commit -m "feat: stage certified G1 render mesh"
  ```

### Task 2: Drive the mesh from the diffusion controller pose

**Files:**
- Modify: `controller.cpp`
- Create: `tests/python/test_diffusion_pickup_g1_mesh_integration.py`

**Interfaces:**
- Consumes: `global_bone_positions`, `global_bone_rotations`, and Task 1's renderer API.
- Produces: mesh startup/load, per-frame pose update, conditional draw, `M`/`B` toggles, and normal cleanup.

- [ ] **Step 1: Write a focused source-contract test**

  Create `tests/python/test_diffusion_pickup_g1_mesh_integration.py` with:

  ```python
  import pathlib
  import unittest

  ROOT = pathlib.Path(__file__).resolve().parents[2]
  SOURCE = (ROOT / "controller.cpp").read_text(encoding="utf-8")


  class DiffusionPickupG1MeshIntegrationTests(unittest.TestCase):
      def test_mesh_consumes_only_final_global_pose(self):
          self.assertIn('#include "g1_mesh_renderer.h"', SOURCE)
          self.assertIn("bool show_g1_mesh = true;", SOURCE)
          self.assertIn("bool show_g1_bones = false;", SOURCE)
          self.assertEqual(SOURCE.count("IsKeyPressed(KEY_M)"), 1)
          self.assertEqual(SOURCE.count("IsKeyPressed(KEY_B)"), 1)
          update = SOURCE.index("::g1_mesh_renderer_update(")
          update_end = SOURCE.index(");", update)
          call = SOURCE[update:update_end]
          self.assertIn("global_bone_positions", call)
          self.assertIn("global_bone_rotations", call)
          self.assertNotIn("adjusted_bone", call)
          begin = SOURCE.index("BeginMode3D(camera);", update)
          draw = SOURCE.index(
              "::g1_mesh_renderer_draw(g1_mesh_renderer);", begin
          )
          end = SOURCE.index("EndMode3D();", draw)
          self.assertLess(update, begin)
          self.assertLess(begin, draw)
          self.assertLess(draw, end)

      def test_mesh_has_explicit_lifecycle(self):
          load = SOURCE.index("::g1_mesh_renderer_load(")
          loop = SOURCE.index("auto update_func = [&]()")
          unload = SOURCE.index("::g1_mesh_renderer_unload(g1_mesh_renderer);")
          close = SOURCE.index("CloseWindow();", unload)
          self.assertLess(load, loop)
          self.assertLess(loop, unload)
          self.assertLess(unload, close)
          self.assertEqual(
              SOURCE.count("::g1_mesh_renderer_unload(g1_mesh_renderer);"), 1
          )


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Verify RED against the skeleton-only controller**

  ```bash
  python3 -m unittest \
    tests.python.test_diffusion_pickup_g1_mesh_integration -v
  ```

  Expected: both tests fail because the renderer include and lifecycle calls are absent.

- [ ] **Step 3: Add startup, update, draw, toggles, and cleanup**

  In `controller.cpp`:

  - Include `g1_mesh_renderer.h` beside the G1 runtime headers.
  - After all ordinary/interactions assets have loaded successfully, create `G1MeshRenderer g1_mesh_renderer = {};`, `bool show_g1_mesh = true;`, `bool show_g1_bones = false;`, and `char g1_mesh_error_message[256] = {};`; load `resources/g1_mesh/g1_raylib.glb` and fail with `G1 mesh load error:` if validation fails.
  - In `update_func`, toggle `show_g1_mesh` with `KEY_M` and `show_g1_bones` with `KEY_B`.
  - After the final global transforms are computed and before `BeginDrawing`, call:

    ```cpp
    if (!::g1_mesh_renderer_update(
            g1_mesh_renderer,
            global_bone_positions,
            global_bone_rotations,
            g1_mesh_error_message,
            static_cast<int>(sizeof(g1_mesh_error_message))))
    {
        throw std::runtime_error(
            std::string("G1 mesh update error: ") + g1_mesh_error_message);
    }
    ```

  - Immediately after `BeginMode3D(camera)`, draw only when `show_g1_mesh` is true:

    ```cpp
    if (show_g1_mesh)
    {
        ::g1_mesh_renderer_draw(g1_mesh_renderer);
    }
    ```

  - Wrap the existing joint/cylinder skeleton loop in `if (show_g1_bones)`.
  - Before the normal `CloseWindow()`, call `::g1_mesh_renderer_unload(g1_mesh_renderer);` exactly once.

- [ ] **Step 4: Verify GREEN and regression safety**

  ```bash
  python3 -m unittest \
    tests.python.test_diffusion_pickup_g1_mesh_integration \
    tests.python.test_g1_mesh_asset -v
  /tmp/diffusion-pickup-g1-mesh-build/test-g1-mesh-renderer
  ```

  Expected: all tests pass and the component executable exits 0.

- [ ] **Step 5: Commit controller integration**

  ```bash
  git add controller.cpp tests/python/test_diffusion_pickup_g1_mesh_integration.py
  git commit -m "feat: render G1 mesh in diffusion pickup runtime"
  ```

### Task 3: Build and launch the mesh-enabled pickup visualizer

**Files:**
- Build output: `controller`
- Runtime log: `/tmp/diffusion-pickup-g1-mesh.log`

**Interfaces:**
- Consumes: the current diffusion model/runtime environment and Task 2's controller.
- Produces: a live mesh-enabled visualizer for user evaluation.

- [ ] **Step 1: Build the current controller**

  ```bash
  make -j2 controller
  ```

  Expected: exit 0 with no compiler errors.

- [ ] **Step 2: Run the existing focused interaction regressions**

  ```bash
  python3 -m unittest \
    tests.python.test_diffusion_pickup_g1_mesh_integration \
    tests.python.test_g1_mesh_asset -v
  ```

  Expected: all tests pass.

- [ ] **Step 3: Launch with the current learned-pickup artifacts**

  ```bash
  DISPLAY=:1 \
  MM_INTERACTION_PACK="$PWD/build/smart-pickup/full-pack" \
  MM_FEATURES_OUTPUT="$PWD/build/g1-funnels/runtime-object-anchor-live/locomotion-features.bin" \
  ./controller > /tmp/diffusion-pickup-g1-mesh.log 2>&1
  ```

  Expected log: `G1 mesh loaded:` and no `G1 mesh load error:` or `G1 mesh update error:`.

- [ ] **Step 4: Smoke-test visual behavior**

  Confirm the mesh is visible at startup, `M` hides/restores it, `B` shows/hides the diagnostic skeleton, and the mesh follows walking, approach, pickup, carry, placement, and post-placement poses without returning to bind pose.
