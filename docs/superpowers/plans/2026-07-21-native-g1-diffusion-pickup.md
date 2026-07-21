# Native G1 Diffusion Pickup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one playable controller whose locomotion, diffusion approach, pickup, carry, placement, skeleton debug drawing, and mesh rendering all consume the same native 31-bone G1 pose.

**Architecture:** Create an isolated integration worktree from the diffusion branch so all interaction modules remain available, then transplant the proven native-G1 terrain controller and supporting runtime under non-conflicting filenames. Add a small native-G1 snapshot/handoff bridge, wire the existing smart-pickup and interaction runtimes directly to it, and publish one final G1 pose to FK, grasp attachment, debug drawing, and the mesh.

**Tech Stack:** C++17, Raylib 6, existing G1 terrain runtime, existing interaction runtime, existing diffusion funnel worker/checkpoint, Python `unittest`, Make.

## Global Constraints

- Do not modify, reset, clean, stage, or commit the dirty `/home/ubuntu/projects/motion-matching` terrain workspace.
- Perform implementation in a new isolated worktree.
- The playable controller must not use `FlatControllerPose`, `collapse_interaction_pose`, or `expand_flat_controller_pose`.
- The final mesh, blue skeleton, grasp validation, and object attachment must consume the same final 31-bone G1 pose.
- Preserve the existing `F` immediate-stop and object-relative diffusion route behavior.
- Preserve pickup, carry, placement, and pickup-after-placement state semantics.
- Do not use screen capture, ImageMagick `import`, or X11 image acquisition.
- Run only one live controller process during manual validation.
- Do not stage generated packs, checkpoints, evidence, binaries, or `resources/features.bin`.

---

## File Structure

- `controller.cpp`: native G1 playable loop and integration wiring.
- `g1_terrain_skeleton.h`: terrain database validation copied from the native branch; renamed to avoid colliding with the interaction `g1_skeleton` namespace.
- `g1_controller_state.h`, `g1_clearance.*`, `g1_command_runtime.h`, `g1_footprint_runtime.h`, `g1_ik.h`, `g1_ik_runtime.h`, `g1_runtime_diagnostics.h`, `g1_surface_query.h`, `motion_match_log.h`, `route_runtime.h`, `scene_runtime.h`, `support_runtime.h`, `terrain_runtime.h`: native terrain runtime units.
- `interaction_native_g1_bridge.h/.cpp`: exact array-to-`interaction::Pose` conversion and G1-to-G1 ownership handoff.
- `tests/cpp/test_interaction_native_g1_bridge.cpp`: bridge and handoff behavior.
- `tests/python/test_native_g1_diffusion_controller.py`: source-level architecture and final-pose fan-out gate.
- Existing `interaction_*.{h,cpp}`: interaction, diffusion, carry, and placement logic; the flat adapter is not linked into the playable controller.

---

### Task 1: Isolated Native-G1 Controller Baseline

**Files:**
- Modify: `controller.cpp`
- Create: `g1_terrain_skeleton.h`
- Restore from `g1-terrain-motion-matching`: `g1_clearance.cpp`, `g1_clearance.h`, `g1_command_runtime.h`, `g1_controller_state.h`, `g1_footprint_runtime.h`, `g1_ik.h`, `g1_ik_runtime.h`, `g1_runtime_diagnostics.h`, `g1_surface_query.h`, `motion_match_log.h`, `route_runtime.h`, `scene_runtime.h`, `support_runtime.h`, `terrain_runtime.h`
- Modify: `Makefile`
- Create: `tests/python/test_native_g1_diffusion_controller.py`

**Interfaces:**
- Consumes: committed native terrain source at `g1-terrain-motion-matching` (`dd31c81` or the branch head recorded when the worktree is created).
- Produces: a compiling native 31-bone G1 controller in the isolated integration worktree.

- [ ] **Step 1: Create the isolated worktree using the worktree skill**

Create branch `native-g1-diffusion-pickup-20260721` from commit `4b624b3` in a new sibling worktree. Record both the diffusion source commit and native terrain source commit in the plan execution notes. Do not touch the dirty terrain checkout.

- [ ] **Step 2: Write the failing architecture test**

```python
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "controller.cpp").read_text(encoding="utf-8")

class NativeG1ControllerTests(unittest.TestCase):
    def test_controller_is_native_g1(self):
        self.assertIn("g1_controller_state state", SOURCE)
        self.assertIn("G1_BoneCount", SOURCE)
        self.assertIn("g1_skeleton_validate", SOURCE)
        self.assertNotIn("FlatControllerPose", SOURCE)
        self.assertNotIn("collapse_interaction_pose", SOURCE)
        self.assertNotIn("expand_flat_controller_pose", SOURCE)
```

- [ ] **Step 3: Run the architecture test and verify RED**

Run: `python3 -m unittest tests.python.test_native_g1_diffusion_controller -v`

Expected: FAIL because the current controller still contains `FlatControllerPose` and expansion calls.

- [ ] **Step 4: Import the native controller without overwriting interaction modules**

Mechanically restore the listed native runtime files from the recorded native terrain commit. Restore its `controller.cpp` as the controller baseline. Copy native `g1_skeleton.h` as `g1_terrain_skeleton.h`, update the controller include to that filename, and retain the compact interaction `g1_skeleton.h` and shared `g1_kinematic_contract.h` already present in the diffusion branch.

- [ ] **Step 5: Make native terrain sources explicit in the controller build**

Add to `Makefile`:

```make
NATIVE_G1_SOURCES := g1_clearance.cpp
NATIVE_G1_HEADERS := g1_terrain_skeleton.h g1_command_runtime.h \
  g1_controller_state.h g1_footprint_runtime.h g1_ik.h \
  g1_ik_runtime.h g1_runtime_diagnostics.h g1_surface_query.h \
  motion_match_log.h route_runtime.h scene_runtime.h support_runtime.h \
  terrain_runtime.h

SOURCE := controller.cpp $(NATIVE_G1_SOURCES)
```

Keep the existing Linux Raylib bootstrap and C++17 flags.

- [ ] **Step 6: Verify native architecture and build GREEN**

Run:

```bash
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
make controller
```

Expected: architecture test passes; controller builds with exit 0; warnings from vendored Raygui are allowed.

- [ ] **Step 7: Commit the native baseline**

```bash
git add controller.cpp Makefile g1_terrain_skeleton.h \
  g1_clearance.cpp g1_clearance.h g1_command_runtime.h \
  g1_controller_state.h g1_footprint_runtime.h g1_ik.h \
  g1_ik_runtime.h g1_runtime_diagnostics.h g1_surface_query.h \
  motion_match_log.h route_runtime.h scene_runtime.h support_runtime.h \
  terrain_runtime.h tests/python/test_native_g1_diffusion_controller.py
git commit -m "feat: establish native G1 playable baseline"
```

---

### Task 2: Exact Native-G1 Snapshot and Pose Handoff

**Files:**
- Create: `interaction_native_g1_bridge.h`
- Create: `interaction_native_g1_bridge.cpp`
- Create: `tests/cpp/test_interaction_native_g1_bridge.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: 31 local G1 positions, velocities, rotations, angular velocities, contacts, and three future root transforms.
- Produces:
  - `interaction::LocomotionSnapshot capture_native_g1_snapshot(...)`
  - `void write_native_g1_pose(const interaction::Pose&, slice1d<vec3>, slice1d<vec3>, slice1d<quat>, slice1d<vec3>, slice1d<bool>)`
  - `interaction::NativeG1FrameState NativeG1PoseHandoff::apply(const interaction::Pose&, const interaction::RuntimeOutput&, float)`

- [ ] **Step 1: Write the failing bridge test**

The test constructs 31 unique local channels, captures them, and requires bit-equal values at every bone. It then writes the pose into fresh arrays and requires another exact round trip. Add an ownership test:

```cpp
interaction::NativeG1PoseHandoff handoff;
interaction::RuntimeOutput runtime{};
runtime.owns_pose = true;
runtime.pose = interaction_pose;
const auto owned = handoff.apply(locomotion_pose, runtime, 0.04F);
require(owned.runtime_owns_pose, "runtime ownership was lost");
require_pose_near(owned.pose, interaction_pose, 1.0e-6F);
```

Add a release test that requires the first unowned frame to equal the previously displayed pose and monotonically converge to locomotion over 0.25 seconds.

- [ ] **Step 2: Run the bridge test and verify RED**

Run: `make build/tests/test_interaction_native_g1_bridge`

Expected: FAIL because `interaction_native_g1_bridge.h` does not exist.

- [ ] **Step 3: Define the bridge API**

```cpp
namespace interaction {

struct NativeG1FrameState {
    Pose pose{};
    bool runtime_owns_pose = false;
    bool synchronize_simulation_root = false;
};

LocomotionSnapshot capture_native_g1_snapshot(
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts,
    slice1d<vec3> trajectory_positions,
    slice1d<quat> trajectory_rotations);

void write_native_g1_pose(
    const Pose& pose,
    slice1d<vec3> positions,
    slice1d<vec3> velocities,
    slice1d<quat> rotations,
    slice1d<vec3> angular_velocities,
    slice1d<bool> contacts);

class NativeG1PoseHandoff {
public:
    NativeG1FrameState apply(
        const Pose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt);
    void reset();
private:
    bool owned_last_tick_ = false;
    float release_seconds_ = 0.0F;
    Pose release_source_{};
    Pose displayed_{};
};

}
```

- [ ] **Step 4: Implement exact conversion and G1-to-G1 release blending**

Validate every slice has `g1_skeleton::BoneCount` elements, map contact indices to `LeftToe` and `RightToe`, copy the first three future trajectory entries, and use `interpolate_pose(release_source_, locomotion_pose, alpha)` with `alpha = clamp(release_seconds_ / 0.25F, 0, 1)`.

- [ ] **Step 5: Add the focused Make target**

```make
$(CPP_TEST_DIR)/test_interaction_native_g1_bridge: \
  tests/cpp/test_interaction_native_g1_bridge.cpp \
  interaction_native_g1_bridge.cpp interaction_native_g1_bridge.h \
  interaction_pose.cpp interaction_pose.h interaction_runtime.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_native_g1_bridge.cpp \
	  interaction_native_g1_bridge.cpp interaction_pose.cpp -o $@
```

Include additional runtime sources only when unresolved symbols prove they are required.

- [ ] **Step 6: Run focused tests GREEN**

Run:

```bash
make build/tests/test_interaction_native_g1_bridge
./build/tests/test_interaction_native_g1_bridge
```

Expected: exit 0.

- [ ] **Step 7: Commit the bridge**

```bash
git add interaction_native_g1_bridge.h interaction_native_g1_bridge.cpp \
  tests/cpp/test_interaction_native_g1_bridge.cpp Makefile
git commit -m "feat: bridge native G1 controller poses"
```

---

### Task 3: Link the Interaction and Diffusion Runtime into Native G1

**Files:**
- Modify: `Makefile`
- Modify: `controller.cpp`
- Modify: `tests/python/test_native_g1_diffusion_controller.py`

**Interfaces:**
- Consumes: existing interaction database/features pack, `interaction::SmartPickupController`, and Task 2 snapshot bridge.
- Produces: a native G1 controller that loads interaction assets and advances the interaction runtime at 25 Hz without changing ordinary locomotion when inactive.

- [ ] **Step 1: Extend the source gate with failing integration assertions**

```python
def test_native_controller_links_diffusion_without_flat_adapter(self):
    self.assertIn('#include "interaction_native_g1_bridge.h"', SOURCE)
    self.assertIn("interaction::SmartPickupController", SOURCE)
    self.assertIn("interaction::RuntimeInput", SOURCE)
    self.assertIn("capture_native_g1_snapshot(", SOURCE)
    self.assertNotIn("interaction_controller_adapter.cpp", MAKEFILE)
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m unittest tests.python.test_native_g1_diffusion_controller -v`

Expected: FAIL because the native controller does not yet include the bridge or interaction runtime.

- [ ] **Step 3: Link only native-compatible interaction sources**

Define `INTERACTION_NATIVE_G1_SOURCES` in `Makefile` from the existing interaction, smart-pickup, learned-backend, funnel, arrival, carry, placement, and SHA sources. Explicitly omit:

```text
interaction_controller_adapter.cpp
interaction_target_rig_ik.cpp
locomotion_controller_update.cpp
```

Add `interaction_native_g1_bridge.cpp` and `$(INTERACTION_NATIVE_G1_SOURCES)` to `SOURCE` after the native terrain sources.

- [ ] **Step 4: Load the interaction pack and initialize runtime state**

Port the validated pack-loading, target registry, surface registry, object/table/rack authored scene, `SmartPickupProductionConfig`, `SmartPickupController`, and `Runtime` initialization blocks from the diffusion controller. Keep errors non-fatal to native locomotion: publish `interaction_pack_diagnostic` and leave runtime disabled when pack validation fails.

- [ ] **Step 5: Capture native locomotion after terrain IK**

Call `capture_native_g1_snapshot` from `state.adjusted_bone_*`, contact state, and `state.trajectory_*` arrays after native locomotion and terrain contact adjustment but before final FK publication. Assert at runtime that all input arrays contain exactly 31 bones.

- [ ] **Step 6: Advance inactive runtime and prove locomotion identity**

Feed the snapshot to `RuntimeInput` each 25 Hz tick. When `RuntimeOutput::owns_pose` is false and no release blend is active, require `NativeG1PoseHandoff` to publish the exact locomotion pose. Write it back with `write_native_g1_pose`, then run native FK.

- [ ] **Step 7: Run source, bridge, interaction-runtime, and build gates**

Run:

```bash
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
./build/tests/test_interaction_native_g1_bridge
make build/tests/test_interaction_runtime
./build/tests/test_interaction_runtime
make controller
```

Expected: all exit 0.

- [ ] **Step 8: Commit runtime linkage**

```bash
git add Makefile controller.cpp tests/python/test_native_g1_diffusion_controller.py
git commit -m "feat: link interaction runtime into native G1 controller"
```

---

### Task 4: Immediate `F` Diffusion Approach on Native G1 Root

**Files:**
- Modify: `controller.cpp`
- Modify: `tests/python/test_native_g1_diffusion_controller.py`
- Test: `tests/cpp/test_interaction_smart_pickup_controller.cpp`
- Test: `tests/cpp/test_interaction_funnel_follower.cpp`

**Interfaces:**
- Consumes: native G1 snapshot, selected target/affordance, camera-relative input, and existing learned funnel backend.
- Produces: immediate `F` capture and object-relative root targets applied through the native G1 command path.

- [ ] **Step 1: Add failing source assertions for update order**

Require this order in `controller.cpp`:

```text
IsKeyPressed(KEY_F)
SmartPickupController::pre_step
native G1 command construction
native locomotion update
capture_native_g1_snapshot
SmartPickupController::post_step
Runtime::update
```

Also require `MM_INTERACTION_FUNNEL_CHECKPOINT` and `MM_INTERACTION_FUNNEL_WORKER` configuration to remain supported.

- [ ] **Step 2: Run source gate RED**

Run: `python3 -m unittest tests.python.test_native_g1_diffusion_controller -v`

Expected: FAIL at the first missing order assertion.

- [ ] **Step 3: Wire pre-step before native command generation**

On the `F` edge, pass the selected object and affordance to `SmartPickupController::pre_step`. Use returned left/right sticks and steering suppression to build the same native `G1CommandSnapshot` used by terrain locomotion. Do not wait for the character to reach a separate radius before beginning planning.

- [ ] **Step 4: Wire post-step after the fresh native snapshot**

Pass the current 31-bone snapshot, native simulation velocity, displayed planar speed, camera azimuth, obstacles, and monotonically increasing request ID to `post_step`. Preserve the certified preview callback and request identity checks.

- [ ] **Step 5: Apply learned route targets through native root commands**

Convert each object-relative follower sample to world root position/yaw using the selected target transform. Publish it as the native command's immediate trajectory/root target while the follower is active. On completion, submit the certified `PickRequest` in the same tick. On cancellation or worker failure, clear suppression and resume native locomotion without moving the object.

- [ ] **Step 6: Run focused route gates**

Run:

```bash
make build/tests/test_interaction_funnel_follower \
  build/tests/test_interaction_smart_pickup_controller
./build/tests/test_interaction_funnel_follower
./build/tests/test_interaction_smart_pickup_controller
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
make controller
```

Expected: all exit 0.

- [ ] **Step 7: Commit native approach control**

```bash
git add controller.cpp tests/python/test_native_g1_diffusion_controller.py
git commit -m "feat: drive native G1 with diffusion approach"
```

---

### Task 5: One Final G1 Pose for Grasp, Mesh, and Skeleton

**Files:**
- Modify: `controller.cpp`
- Modify: `interaction_native_g1_bridge.cpp`
- Modify: `tests/cpp/test_interaction_native_g1_bridge.cpp`
- Modify: `tests/python/test_native_g1_diffusion_controller.py`

**Interfaces:**
- Consumes: `RuntimeOutput::pose`, native locomotion pose, and active grasp constraint.
- Produces: `interaction::Pose final_g1_pose` and `interaction::WorldPose final_g1_world_pose`, used by every visual and grasp consumer.

- [ ] **Step 1: Add failing final-pose fan-out tests**

The Python gate must isolate the final-pose block and require all of these consumers:

```text
write_native_g1_pose(final_g1_pose, ...)
g1_mesh_renderer_update(... final_g1_world_pose.positions ...)
debug skeleton loop over final_g1_world_pose.positions
grasp validation against final_g1_world_pose
```

It must reject any mesh or skeleton call using pre-handoff `state.global_bone_positions`.

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m unittest tests.python.test_native_g1_diffusion_controller -v`

Expected: FAIL because one named final pose does not yet fan out to all consumers.

- [ ] **Step 3: Publish runtime ownership directly**

Use `NativeG1PoseHandoff::apply(native_snapshot.pose, runtime_output, 0.04F)`. Name the result `final_g1_pose`, write it to native local arrays, run FK once, and construct `final_g1_world_pose` from those final arrays. Do not collapse or retarget it.

- [ ] **Step 4: Use the final wrist for grasp validation and attachment**

At full hand-constraint weight, validate the selected G1 wrist world position against `grasp_world.position` with the existing certified tolerance. Object attachment and carried-object transforms must use the same final wrist transform.

- [ ] **Step 5: Feed mesh and blue skeleton from final G1 world data**

Call `g1_mesh_renderer_update` with final world arrays. Draw the optional skeleton using `g1_skeleton::kParents` and those same arrays. Keep `M` and `B` as visual-only toggles.

- [ ] **Step 6: Run bridge, IK, attachment, source, and build tests**

Run:

```bash
./build/tests/test_interaction_native_g1_bridge
make build/tests/test_interaction_ik build/tests/test_interaction_attachment
./build/tests/test_interaction_ik
./build/tests/test_interaction_attachment
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
make controller
```

Expected: all exit 0.

- [ ] **Step 7: Commit truthful final-pose rendering**

```bash
git add controller.cpp interaction_native_g1_bridge.cpp \
  tests/cpp/test_interaction_native_g1_bridge.cpp \
  tests/python/test_native_g1_diffusion_controller.py
git commit -m "feat: publish one final native G1 pose"
```

---

### Task 6: Carry, Placement, and Re-Pick Lifecycle

**Files:**
- Modify: `controller.cpp`
- Modify: `tests/python/test_native_g1_diffusion_controller.py`
- Test: `tests/cpp/test_interaction_carry.cpp`
- Test: `tests/cpp/test_interaction_place_controller.cpp`
- Test: `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`

**Interfaces:**
- Consumes: native final G1 pose, interaction object state, placement surfaces, and `F` edge.
- Produces: pickup → carry → place → free → pickup lifecycle in the native playable controller.

- [ ] **Step 1: Add failing lifecycle source assertions**

Require one native controller path for `PickRequest`, `PlaceRequest`, `ObjectState::Held`, `ObjectState::Free`, placement preview, and incrementing request IDs. Require reset/cancel to clear both funnel and interaction ownership.

- [ ] **Step 2: Run source gate RED**

Run: `python3 -m unittest tests.python.test_native_g1_diffusion_controller -v`

Expected: FAIL on missing place or re-pick wiring.

- [ ] **Step 3: Port scene and placement resolvers**

Port the validated object registry, table/rack surface registry, single-affordance resolver, support-fit preview, and scene handoff logic. Use the native G1 root and final wrist; do not introduce a flat snapshot.

- [ ] **Step 4: Preserve object generation and request identity**

After placement release, update the registry's free object transform and generation exactly once. A subsequent `F` must resolve the current generation and allocate a strictly newer request ID before planning.

- [ ] **Step 5: Run lifecycle gates**

Run:

```bash
make build/tests/test_interaction_carry \
  build/tests/test_interaction_place_controller \
  build/tests/test_interaction_smart_pickup_scenarios
./build/tests/test_interaction_carry
./build/tests/test_interaction_place_controller
./build/tests/test_interaction_smart_pickup_scenarios
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
make controller
```

Expected: all exit 0.

- [ ] **Step 6: Commit lifecycle integration**

```bash
git add controller.cpp tests/python/test_native_g1_diffusion_controller.py
git commit -m "feat: complete native G1 object lifecycle"
```

---

### Task 7: Focused Verification and Safe Live Handoff

**Files:**
- Modify only if a focused gate exposes a defect.

**Interfaces:**
- Consumes: completed native-G1 controller and existing interaction pack/checkpoint.
- Produces: one live native-G1 visualizer for user acceptance.

- [ ] **Step 1: Run fresh focused verification**

```bash
git diff --check
python3 -m unittest \
  tests.python.test_native_g1_diffusion_controller \
  tests.python.test_g1_mesh_asset
./build/tests/test_interaction_native_g1_bridge
./build/tests/test_interaction_funnel_follower
./build/tests/test_interaction_smart_pickup_controller
./build/tests/test_interaction_runtime
./build/tests/test_interaction_ik
./build/tests/test_interaction_attachment
./build/tests/test_interaction_carry
./build/tests/test_interaction_place_controller
./build/tests/test_interaction_smart_pickup_scenarios
make controller
```

Expected: all commands exit 0. Vendored Raygui compiler warnings are permitted; test failures are not.

- [ ] **Step 2: Verify source architecture explicitly**

Run:

```bash
rg -n "FlatControllerPose|collapse_interaction_pose|expand_flat_controller_pose" controller.cpp
```

Expected: no matches and exit 1 from `rg`.

- [ ] **Step 3: Stop only the previous known controller PID**

Resolve the PID using `pgrep -n -x controller`, inspect it with `ps`, and send `SIGTERM` only after its command and working directory match the old visualizer. Do not use broad `pkill` patterns.

- [ ] **Step 4: Launch one native-G1 controller without screen capture**

```bash
DISPLAY=:1 \
MM_INTERACTION_PACK=build/smart-pickup/full-pack \
MM_INTERACTION_FUNNEL_CHECKPOINT=build/g1-funnels/checkpoint-schema3-20k.pt \
./controller
```

Verify only process and window metadata with `ps` and `wmctrl`.

- [ ] **Step 5: Manual acceptance checkpoint**

Ask the user to verify walking, immediate `F`, continuous approach, pickup, carry, placement, and pickup after placement. The mesh and blue G1 skeleton must agree at the wrist and elbow. Do not claim visual success without the user's observation.

- [ ] **Step 6: Commit any verification-only fix, then report status**

If no fix was required, leave the last implementation commit as HEAD. If a focused defect was fixed test-first, commit only its source and test files with a specific `fix:` message. Report the live PID, exact passing gates, and any deferred item.

