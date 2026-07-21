# Flat Interaction Terrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict live-launch option that makes the native G1 diffusion pickup environment physically and visually flat.

**Architecture:** Parse one boolean environment option before loading controller state. Keep the validated scene's coordinate grid and dimensions, but replace its height and walkability payload after scene validation; render a plane instead of the terrain model when enabled.

**Tech Stack:** C++17, Raylib 6, Python `unittest`, Make

## Global Constraints

- Preserve the native 31-bone G1 controller and interaction pose path.
- Do not modify the certified terrain pack or dirty terrain workspace.
- `MM_INTERACTION_FLAT_TERRAIN` accepts only absent, `0`, or `1`.
- The flat override must affect physics and rendering together.
- Run only one live controller and do not use screen-capture tooling.

---

### Task 1: Strict Flat Terrain Override and Live Relaunch

**Files:**
- Modify: `controller.cpp`
- Modify: `tests/python/test_native_g1_diffusion_controller.py`

**Interfaces:**
- Consumes: `scene_pack active_scene`, `heightfield.heights`, `walkability_grid.cells`, and Raylib drawing.
- Produces: `bool flat_interaction_terrain` controlled by `MM_INTERACTION_FLAT_TERRAIN`.

- [ ] **Step 1: Write the failing source gate**

Add a test requiring strict option parsing, flattening of all height and walkability cells, and a `DrawPlane` branch that excludes `DrawModel` when flat mode is active:

```python
def test_flat_interaction_terrain_changes_physics_and_rendering(self):
    self.assertIn("MM_INTERACTION_FLAT_TERRAIN", SOURCE)
    self.assertIn("g1_parse_flat_interaction_terrain(", SOURCE)
    self.assertIn("active_scene.terrain.heights", SOURCE)
    self.assertIn("active_scene.walkability.cells", SOURCE)
    self.assertIn("if (flat_interaction_terrain)", SOURCE)
    self.assertIn("DrawPlane(", SOURCE)
```

- [ ] **Step 2: Run the gate and verify RED**

Run:

```bash
python3 -m unittest \
  tests.python.test_native_g1_diffusion_controller.NativeG1ControllerTests.test_flat_interaction_terrain_changes_physics_and_rendering -v
```

Expected: FAIL because the option and rendering branch do not exist.

- [ ] **Step 3: Implement strict parsing**

Add:

```cpp
static bool g1_parse_flat_interaction_terrain(
    bool& enabled, char* error, int error_capacity)
{
    const char* value = getenv("MM_INTERACTION_FLAT_TERRAIN");
    if (value == NULL || strcmp(value, "0") == 0) {
        enabled = false;
        return true;
    }
    if (strcmp(value, "1") == 0) {
        enabled = true;
        return true;
    }
    return g1_error(error, error_capacity,
        "MM_INTERACTION_FLAT_TERRAIN must be 0 or 1, got '%s'", value);
}
```

Parse it during startup and fail with exit code 2 on invalid input.

- [ ] **Step 4: Flatten validated runtime data**

Immediately after `scene_pack_load` succeeds, preserve dimensions/origins and replace payload values:

```cpp
if (flat_interaction_terrain) {
    active_scene.terrain.exterior_height = 0.0F;
    for (int index = 0; index < active_scene.terrain.heights.size; ++index) {
        active_scene.terrain.heights(index) = 0.0F;
    }
    for (int index = 0; index < active_scene.walkability.cells.size; ++index) {
        active_scene.walkability.cells(index) = 1U;
    }
}
```

Apply the same override after every successful scene switch so UI navigation cannot restore uneven physics.

- [ ] **Step 5: Render only a flat plane in override mode**

Replace the unconditional terrain drawing with:

```cpp
if (flat_interaction_terrain) {
    const bounds3d& bounds = active_scene.metadata.heightfield_bounds;
    const float width = static_cast<float>(bounds.maximum.x - bounds.minimum.x);
    const float depth = static_cast<float>(bounds.maximum.z - bounds.minimum.z);
    const Vector3 center{
        static_cast<float>(0.5 * (bounds.minimum.x + bounds.maximum.x)),
        0.0F,
        static_cast<float>(0.5 * (bounds.minimum.z + bounds.maximum.z))};
    DrawPlane(center, Vector2{width, depth}, Color{205, 199, 184, 255});
} else {
    DrawModel(terrain_model, Vector3{}, 1.0F, Color{205, 199, 184, 255});
    DrawModelWires(terrain_model, Vector3{}, 1.0F, DARKGRAY);
}
```

- [ ] **Step 6: Verify behavior and regressions**

Run:

```bash
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
make controller -j2
MM_INTERACTION_FLAT_TERRAIN=invalid ./controller
```

Expected: source tests pass, build exits 0, invalid option exits 2 before opening a window and names the invalid option.

Then run the existing focused native bridge/runtime checks and a bounded 10-frame flat controller launch with the validated pack and learned-worker environment.

- [ ] **Step 7: Commit and relaunch exactly one controller**

```bash
git add controller.cpp tests/python/test_native_g1_diffusion_controller.py
git commit -m "feat: run native G1 interaction on flat terrain"
```

Stop only the exact prior native-G1 controller PID after checking its working directory. Relaunch with `MM_INTERACTION_FLAT_TERRAIN=1` plus the existing absolute terrain pack, interaction pack, checkpoint, worker, and Torch-enabled Python paths. Verify one process and its window metadata with `ps` and `wmctrl`.
