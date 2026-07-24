# Trackpad Right-Drag Camera Pan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add right-button drag as a trackpad-friendly camera-pan gesture without removing the existing orbit, middle-pan, or wheel-zoom controls.

**Architecture:** Extend the existing `update_orbit_camera` input state with one `pan_down` boolean shared by right and middle buttons. Give pan precedence over left-button orbit and update the existing source-contract test and HUD hint.

**Tech Stack:** C++17, raylib mouse input, Python `unittest`, Make, X11/xdotool live verification.

## Global Constraints

- Left-button drag continues to orbit.
- Right-button drag pans in the current view plane.
- Middle-button drag remains a pan fallback.
- Wheel zoom remains unchanged.
- Pan wins if left and right are simultaneously down.
- Preserve all unrelated user changes and generated reach-pack artifacts.

---

### Task 1: Add Right-Drag Pan

**Files:**
- Modify: `tests/python/test_g1_reach_coverage_viewer.py:131`
- Modify: `g1_reach_coverage_viewer.cpp:77`
- Modify: `g1_reach_coverage_viewer.cpp:569`

**Interfaces:**
- Consumes: raylib `IsMouseButtonDown`, `MOUSE_BUTTON_LEFT`, `MOUSE_BUTTON_RIGHT`, and `MOUSE_BUTTON_MIDDLE`.
- Produces: a `pan_down` input state used by orbit and pan branches.

- [ ] **Step 1: Write the failing source-contract test**

Add right-button, pan precedence, and HUD assertions:

```python
self.assertIn("MOUSE_BUTTON_RIGHT", source)
self.assertIn(
    "const bool pan_down =",
    source,
)
self.assertIn(
    "IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !pan_down",
    source,
)
self.assertIn("right/middle pan", source)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests.test_lowered_scene_and_mouse_camera_contract \
  -v
```

Expected: failure because `MOUSE_BUTTON_RIGHT` and `right/middle pan` are absent.

- [ ] **Step 3: Implement the minimal camera-input change**

In `update_orbit_camera`, derive pan state before orbit handling:

```cpp
const bool pan_down =
    IsMouseButtonDown(MOUSE_BUTTON_RIGHT) ||
    IsMouseButtonDown(MOUSE_BUTTON_MIDDLE);
if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !pan_down) {
    // Existing orbit body remains unchanged.
}
```

Replace the middle-only pan condition:

```cpp
if (pan_down) {
    const float pan_scale = 0.0015F * state.distance;
    state.target = state.target +
        pan_scale * (mouse_delta.x * right + mouse_delta.y * view_up);
}
```

Update the HUD hint:

```cpp
"M mesh %s | B bones %s | left orbit | right/middle pan | wheel zoom"
```

- [ ] **Step 4: Run focused and complete viewer tests**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer -v
make g1_reach_coverage_viewer
```

Expected: 14 viewer tests pass and the release viewer links successfully.

- [ ] **Step 5: Verify the live gesture**

Launch the replacement viewer on display `:1`, focus its X11 window, and send a
held button-3 drag with `xdotool`. Capture screenshots before and after. The
rendered camera view must change, the process must remain alive, and only one
viewer process should remain afterward.

- [ ] **Step 6: Commit**

```bash
git add g1_reach_coverage_viewer.cpp \
  tests/python/test_g1_reach_coverage_viewer.py
git commit -m "feat: pan camera with right drag"
```
