# Industry Smart Pickup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Each behavior change follows superpowers:test-driven-development and every
> task receives an independent specification-and-quality review. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the player press interact from varied clear starts, navigate with
ordinary flat locomotion to the best authored Smart Object-style pickup slot,
motion-match a compatible GRAIL pickup globally, carry the object, and place it
on the second table without a root or joint teleport.

**Architecture:** A right-hand `GraspAffordance` owns ordered object-planar root
slots. A raylib-free selector maps, continuously clearance-checks, ranks, and
freezes one slot; the manual assist drives ordinary locomotion to that exact
transform and requests one unrestricted runtime preview only after settling.
The existing matcher, bounded root alignment, IK, contact attachment, Carry,
and placement pipeline remain execution authority.

**Tech Stack:** C++17, the existing Holden-style flat motion matcher, G1/GRAIL
schema-v1 data at 25 Hz, Python 3 artifact builders and evidence validators,
Make, and raylib for the final native visualization only.

## Global Constraints

- Work only in
  `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement` on branch
  `g1-tabletop-placement`. Do not merge or modify the terrain-aware worktree.
- Preserve all pre-existing dirty changes. After every independently reviewed
  task, create a scoped checkpoint commit and push this branch to `checkpoint`.
  Do not merge, switch branches, or clean the worktree. Stage new files only by
  explicit path and never stage the protected probe.
- Never access, stat, hash, execute, modify, stage, or delete the repository-root
  untracked `interaction_query_probe`. Never enumerate repository-root untracked
  files. If status is needed, use `git status --untracked-files=no`.
- Use `apply_patch` for source, test, and documentation edits.
- Control, simulation, flat locomotion, GRAIL data, interaction playback,
  evidence, and video remain exactly 25 Hz (`dt = 0.04 s`). Do not add 60 Hz or
  render-rate interpolation.
- Checkpoint 1 is flat-only, one rigid right-hand tabletop target, one carried
  object, and one destination table. Do not add terrain, shelves, articulated
  objects, several-target UI, two-hand grasps, learned proposals, or global
  pathfinding.
- Use authored slots owned by the grasp affordance and expressed in the target
  object's XZ/yaw frame. Do not generate runtime arcs, splines, funnels, clip-0
  waypoints, or motion-linked slot restrictions.
- Build and validate the complete GRAIL pickup-table pack. Manual and acceptance
  runs must not use the one-clip diagnostic pack.
- Slot selection may use only the post-step live root, exact frozen target and
  affordance metadata, the table proxy, and the same immutable controller
  obstacle arrays used by ordinary locomotion. Motion-match cost never chooses
  or changes the navigation slot.
- Direct travel is at most `1.00 m + 2e-5 m`. The table is a closed yaw-only XZ
  rectangle expanded by `0.24 m`. Generic blockers use exact closed-segment
  distance to their world-axis-aligned 3D AABBs and require clearance strictly
  greater than `0.60002 m` at fixed live-root Y. The tabletop target has no invented
  root collider.
- Manual Smart Pickup uses frozen-slot pose tolerances, not the legacy
  `0.35-0.45 m` object-origin standoff. Preserve `ArrivalConfig` standoff and
  legacy `arrival_ready` behavior for placement.
- Approach latch is inclusive at `0.03 m` root error, `0.05 m/s` simulation
  speed, and `20 degrees` yaw error. Settling is inclusive at `0.15 m` root
  error, `0.10 m/s` displayed speed, and `20 degrees` yaw error for five
  consecutive 25 Hz observations.
- Runtime preview and Preflight search the complete interaction database with
  unchanged limits: `0.25 m` root correction, `25 degrees` yaw correction,
  `0.12 m` hand correction, `25 degrees` hand orientation, and total normalized
  cost `9.0`.
- A sole `Reason::PoorMatch` may retry stationary within the existing arrival
  deadline. Hard path/correction/metadata failures fail without slot hopping,
  snapping, threshold expansion, or a second request.
- `PickRequest` remains target handle, affordance ID, and request ID. Slot ID is
  diagnostic only. TargetRegistry reservation remains owned by runtime
  Preflight.
- Attachment uses the frozen authored target affordance's `hand_in_object`.
  Preserve existing contact gates, contiguous playback, Carry, placement, and
  release authority.
- Change only the manual Smart Pickup path. Preserve the placement auto-demo's
  legacy `make_pick_reach_waypoint`, Plus/Minus slots, preview order, evidence,
  and tests.
- Every production behavior is preceded by a focused failing test that fails for
  the intended missing behavior, followed by the minimal implementation and a
  green focused plus regression run.

---

## File Structure

- `interaction_target.h/.cpp`: authored slot record, slot validation, and exact
  ordered affordance metadata identity.
- `interaction_pick_slots.h/.cpp`: object-planar mapping, deterministic
  quantization/ranking, exact direct-segment clearance, and frozen-slot
  revalidation; no runtime or raylib dependency.
- `interaction_pick_assist.h/.cpp`: one-slot manual state machine, ordinary
  steering, settling, preview retry, and one request handoff.
- `interaction_smart_pickup_controller.h/.cpp`: raylib-free production
  coordinator for pre-step intent capture and post-step activation,
  preview/observation, and one-shot request extraction; both the native
  controller and headless oracle call this unit.
- `interaction_pick_approach.h/.cpp`: unchanged legacy Plus/Minus implementation
  retained only for placement.
- `interaction_smart_pickup_scene.h/.cpp`: baked data-derived manual target,
  destination scene, and stable slot-provenance constants.
- `interaction_smart_pickup_preview_probe.cpp` and
  `interaction_smart_pickup_scene_probe.cpp`: unrestricted full-pack
  certification and exact compiled-scene comparison.
- `interaction_smart_pickup_scenarios.h/.cpp`: deterministic native acceptance
  starts, target transforms, and immutable blockers; no behavior algorithms.
- `interaction_controller_adapter.h/.cpp`: existing controller/runtime pose and
  scene adapter boundaries; shared exact identity remains owned by
  `interaction_target.*`.
- `controller.cpp`: post-step activation ordering, one-slot preview call,
  immutable obstacle handoff, diagnostics, and debug drawing.
- `interaction_matcher.cpp`, `interaction_runtime.cpp`,
  `interaction_attachment.cpp`, and `interaction_place.cpp`: delegate or extend
  existing metadata equality/hash seams so ordered slot mutation invalidates an
  attempt; selection/alignment/playback algorithms remain unchanged.
- `tests/cpp/test_interaction_target.cpp`: slot validation and legacy-empty
  coverage.
- `tests/cpp/test_interaction_pick_slots.cpp`: pure mapping, route, ranking, and
  revalidation coverage.
- `tests/cpp/test_interaction_pick_assist.cpp`: simplified state-machine and
  one-shot handoff coverage.
- `tests/cpp/test_interaction_controller_adapter.cpp`: exact authored demo scene
  and slot provenance coverage.
- `tests/cpp/test_live_flat_smart_pickup_oracle.cpp`: full-pack unrestricted
  preview and deterministic varied-start oracle.
- `resources/extract_g1_pick_slots.py`: offline manifest-aware first-Reach slot
  extraction and static compatibility/path report; never linked or called by
  runtime.
- `tests/python/test_extract_g1_pick_slots.py`: synthetic extraction geometry,
  stable identity, and deterministic-output tests.
- `tests/python/test_smart_pickup_full_pack_gate.py`: manifest-aware wrapper that
  joins C++ candidate ordinals back to stable source provenance.
- `tests/python/test_playable_interaction_evidence.py`: manual wiring policy,
  exact 25 Hz, obstacle identity, evidence schema, and no-pose-write coverage.
- `resources/validate_smart_pickup_graphical_evidence.py` and its Python test:
  fail-closed multi-scenario native evidence validation and canonical summary.
- `Makefile`: explicit new source/test/oracle inputs and normal/fast-math gates.
- `README.md`: full-pack manual command, controls, slot diagnostics, limits, and
  explicit generalization contract.
- `docs/superpowers/specs/2026-07-17-smart-pickup-authored-data.md`: full-pack
  hashes and the stable source provenance for every baked target/slot constant.

### Task 0: Publish the Full 25 Hz Tabletop Pack

**Files:**
- Generate: `build/smart-pickup/full-pack/interaction_database.bin`
- Generate: `build/smart-pickup/full-pack/interaction_features.bin`
- Generate: `build/smart-pickup/full-pack/manifest.json`
- Generate: `build/smart-pickup/full-pack/evaluation_split.json`
- Generate: `build/smart-pickup/full-pack/validation_report.json`

**Interfaces:**
- Consumes: raw pickup-table corpus at
  `/home/ubuntu/datasets/GRAIL/data/pickup_table`, G1 XML at
  `/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml`, schema-v1
  builder defaults `target_fps=25`, `heldout_count=20`, `seed=20260714`.
- Produces: one validated unrestricted runtime pack at
  `build/smart-pickup/full-pack`; later tasks pass this path through
  `MM_INTERACTION_PACK` or their explicit pack argument.

- [x] **Step 1: Prove that the current diagnostic pack cannot satisfy this checkpoint**

  Run:

  ```bash
  python - <<'PY'
  import json
  from pathlib import Path

  manifest = json.loads(
      Path("resources/g1_interaction/manifest.json").read_text()
  )
  assert manifest["diagnostic_limit"] is None
  assert len(manifest["clips"]) == 2045
  PY
  ```

  Expected: `AssertionError`, because the current pack records
  `diagnostic_limit == 5` and contains one database clip. This is the intended
  red precondition, not a builder failure.

- [x] **Step 2: Build the complete deterministic pack without overwriting the diagnostic pack**

  Run:

  ```bash
  python -m resources.build_g1_interaction_database \
    --source-root /home/ubuntu/datasets/GRAIL/data/pickup_table \
    --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
    --output build/smart-pickup/full-pack \
    --target-fps 25 \
    --heldout-count 20 \
    --seed 20260714 \
    --allow-rejections
  ```

  Expected terminal summary:

  ```text
  BUILT schema=1 fps=25 clips=2045 frames=511250 heldout_objects=20 rejected=883 output=build/smart-pickup/full-pack
  ```

- [x] **Step 3: Validate the published artifact set and exact corpus contract**

  Run:

  ```bash
  python -m resources.validate_g1_interaction_database \
    --input build/smart-pickup/full-pack
  G1_INTERACTION_DIR=build/smart-pickup/full-pack \
    python -m unittest tests.python.test_interaction_gate1 -v
  ```

  Expected: validator reports schema `1`, fps `25`, bones `31`, features `71`,
  clips `2045`, frames `511250`, held-out objects `20`; the full-corpus Gate 1
  test passes.

- [x] **Step 4: Verify the exact manifest/report counts and retained source clips**

  Run:

  ```bash
  python - <<'PY'
  import json
  from pathlib import Path

  root = Path("build/smart-pickup/full-pack")
  manifest = json.loads((root / "manifest.json").read_text())
  report = json.loads((root / "validation_report.json").read_text())
  ids = {clip["sequence_id"] for clip in manifest["clips"]}
  assert manifest["target_fps"] == 25.0
  assert manifest["diagnostic_limit"] is None
  assert len(manifest["clips"]) == 2045
  assert report["source_clips"] == 2991
  assert report["included_clips"] == 2108
  assert report["rejected_clips"] == 883
  assert report["included_frames"] == 511250
  assert any(value.startswith("pickup_table__beer_10__") for value in ids)
  print("full Smart Pickup pack contract passed")
  PY
  ```

  Expected: `full Smart Pickup pack contract passed`.

### Task 1: Add Authored Slot Data and Exact Metadata Identity

**Files:**
- Modify: `interaction_target.h`
- Modify: `interaction_target.cpp`
- Modify: `interaction_matcher.cpp`
- Modify: `interaction_runtime.cpp`
- Modify: `interaction_attachment.cpp`
- Modify: `interaction_place.cpp`
- Modify: `interaction_place_probe.cpp`
- Modify: `tests/cpp/test_interaction_target.cpp`
- Modify: `tests/cpp/test_interaction_matcher.cpp`
- Modify: `tests/cpp/test_interaction_runtime.cpp`
- Modify: `tests/cpp/test_interaction_attachment.cpp`
- Modify: `tests/cpp/test_interaction_place.cpp`
- Modify: `tests/python/test_place_probe.py`

**Interfaces:**
- Consumes: existing `InteractionTarget`, `GraspAffordance`, target validation,
  exact runtime target snapshots, and canonical placement fingerprints.
- Produces:

  ```cpp
  struct GraspInteractionSlot {
      uint32_t id = 0U;
      float root_x_object_m = 0.0F;
      float root_z_object_m = 0.0F;
      float root_yaw_object_radians = 0.0F;
  };

  bool same_authored_grasp_affordance(
      const GraspAffordance& left,
      const GraspAffordance& right);
  bool same_interaction_target_snapshot(
      const InteractionTarget& left,
      const InteractionTarget& right);
  ```

  `GraspAffordance` gains
  `std::vector<GraspInteractionSlot> interaction_slots;`. Empty vectors remain
  valid for legacy fixtures and placement.

- [x] **Step 1: Write failing target-validation tests**

  Add cases to `tests/cpp/test_interaction_target.cpp` that construct the
  following ordered slot vector and successfully `upsert` it:

  ```cpp
  target.affordances.front().interaction_slots = {
      {3U, -0.41F, -0.22F, 1.10F},
      {9U,  0.18F, -0.39F, 0.20F},
  };
  ```

  Copy the target and independently assert that `upsert` throws
  `std::invalid_argument` for:

  ```cpp
  slots[0].id = 0U;
  slots[1].id = slots[0].id;
  slots[0].root_x_object_m = quiet_nan;
  slots[0].root_z_object_m = positive_infinity;
  slots[0].root_yaw_object_radians = quiet_nan;
  ```

  Add a separate target with `interaction_slots.clear()` and assert it still
  registers successfully.

- [x] **Step 2: Run the target test and verify the red failure**

  Run:

  ```bash
  make build/tests/test_interaction_target
  ```

  Expected: compilation fails because `GraspInteractionSlot` and
  `interaction_slots` do not exist.

- [x] **Step 3: Add the minimal slot record, validation, and shared exact comparison**

  Add this data shape before `GraspAffordance` in `interaction_target.h`:

  ```cpp
  struct GraspInteractionSlot {
      uint32_t id = 0U;
      float root_x_object_m = 0.0F;
      float root_z_object_m = 0.0F;
      float root_yaw_object_radians = 0.0F;
  };
  ```

  Add `interaction_slots` as the last field of `GraspAffordance`. In
  `validate_affordance`, reject non-finite geometry, zero IDs, and duplicate IDs
  using the same ordered nested-loop style already used for affordance IDs.
  Implement `same_authored_grasp_affordance` with exact scalar comparison and
  exact ordered vector size/content comparison:

  ```cpp
  if (left.interaction_slots.size() != right.interaction_slots.size()) {
      return false;
  }
  for (size_t i = 0; i < left.interaction_slots.size(); ++i) {
      const GraspInteractionSlot& a = left.interaction_slots[i];
      const GraspInteractionSlot& b = right.interaction_slots[i];
      if (a.id != b.id ||
          a.root_x_object_m != b.root_x_object_m ||
          a.root_z_object_m != b.root_z_object_m ||
          a.root_yaw_object_radians != b.root_yaw_object_radians) {
          return false;
      }
  }
  return true;
  ```

  Implement `same_interaction_target_snapshot` with exact handle, transforms,
  profile, bounds, dimensions, table, dynamic state/owner, affordance order, and
  `same_authored_grasp_affordance` for each affordance.

- [x] **Step 4: Run target validation green**

  Run:

  ```bash
  make build/tests/test_interaction_target && \
    build/tests/test_interaction_target
  ```

  Expected: target tests pass with empty legacy vectors and valid ordered slots;
  every malformed slot fails closed.

- [x] **Step 5: Write failing metadata-mutation tests at execution boundaries**

  Add one focused case at each existing seam:

  ```cpp
  copied.affordances.front().interaction_slots[0].root_x_object_m += 0.001F;
  ```

  - Matcher: copied request affordance versus target affordance must reject with
    `Reason::TargetUnavailable`.
  - Runtime: replacing target metadata between preview/build must invalidate the
    exact target snapshot. Replacing the registered target's ordered slot vector
    while the attempt is already in Carry must also invalidate the frozen
    runtime target rather than silently continuing with changed authored data.
  - Attachment: a changed slot vector must fail exact target/affordance identity
    instead of attaching.
  - Placement canonical fingerprint: changing slot order, then changing one
    slot ID, must change the fingerprint.
  - Placement probe scene identity: changing slot order or one local value must
    fail its exact target/affordance invariant.

  Each test begins from a fixture with two nonempty slots so order mutation is
  observable.

- [x] **Step 6: Run the boundary tests and verify red behavior**

  Run:

  ```bash
  make \
    build/tests/test_interaction_matcher \
    build/tests/test_interaction_runtime \
    build/tests/test_interaction_attachment \
    build/tests/test_interaction_place \
    interaction_place_probe && \
    build/tests/test_interaction_matcher && \
    build/tests/test_interaction_runtime && \
    build/tests/test_interaction_attachment && \
    build/tests/test_interaction_place
  python -m unittest tests.python.test_place_probe -v
  ```

  Expected: the new mutation assertions fail because current local equality and
  hash functions ignore slots.

- [x] **Step 7: Route every exact affordance seam through the shared comparison**

  Make the current local helpers in matcher, runtime, and attachment
  delegate to `same_authored_grasp_affordance` and, where they compare the whole
  snapshot, `same_interaction_target_snapshot`. Do not alter their surrounding
  target/state checks. Extend `hash_grasp` in `interaction_place.cpp` in ordered
  form:

  ```cpp
  hash.u64(static_cast<uint64_t>(value.interaction_slots.size()));
  for (const GraspInteractionSlot& slot : value.interaction_slots) {
      hash.u32(slot.id);
      hash.scalar(slot.root_x_object_m);
      hash.scalar(slot.root_z_object_m);
      hash.scalar(slot.root_yaw_object_radians);
  }
  ```

  Keep slot order authoritative; do not sort at runtime.

- [x] **Step 8: Run focused and safe regressions**

  Run:

  ```bash
  make \
    build/tests/test_interaction_target \
    build/tests/test_interaction_matcher \
    build/tests/test_interaction_runtime \
    build/tests/test_interaction_attachment \
    build/tests/test_interaction_place \
    interaction_place_probe && \
    build/tests/test_interaction_target && \
    build/tests/test_interaction_matcher && \
    build/tests/test_interaction_runtime && \
    build/tests/test_interaction_attachment && \
    build/tests/test_interaction_place
  python -m unittest tests.python.test_place_probe -v
  make test-interaction-safe
  ```

  Expected: all focused binaries and the safe interaction suite pass; placement
  fixtures with empty slot vectors remain unchanged.

### Task 2: Map, Clearance-Check, Rank, and Freeze One Slot

**Files:**
- Create: `interaction_pick_slots.h`
- Create: `interaction_pick_slots.cpp`
- Create: `tests/cpp/test_interaction_pick_slots.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `GraspInteractionSlot`, an exact target/affordance snapshot, the
  post-step live root, and a copied ordered view of the controller's immutable
  obstacle centers/sizes.
- Produces:

  ```cpp
  enum class PickSlotReason : uint8_t {
      None,
      NoAuthoredSlot,
      InvalidGeometry,
      OutsideTravelEnvelope,
      TableBlocked,
      ObstacleBlocked,
      AllSlotsBlocked,
  };

  struct PickNavigationObstacle {
      vec3 center_world{};
      vec3 size_world{};
  };

  struct MappedPickSlot {
      uint32_t id = 0U;
      Transform root_world{};
      float route_length_m = 0.0F;
      float heading_change_radians = 0.0F;
      float object_origin_distance_m = 0.0F;
      float object_bounds_center_distance_m = 0.0F;
      uint64_t route_millimetres = 0U;
      uint64_t heading_milliradians = 0U;
      PickSlotReason reason = PickSlotReason::InvalidGeometry;
      int32_t obstacle_index = -1;
  };

  struct PickSlotSelection {
      std::vector<MappedPickSlot> ordered{};
      std::optional<size_t> selected_index{};
      PickSlotReason reason = PickSlotReason::NoAuthoredSlot;
  };

  struct PickSlotConfig {
      float maximum_direct_travel_m = 1.00F;
      float travel_tolerance_m = 2.0e-5F;
      float table_root_expansion_m = 0.24F;
      float obstacle_root_radius_m = 0.60F;
      float obstacle_safety_margin_m = 2.0e-5F;
  };

  PickSlotSelection select_pick_slot(
      Transform live_root,
      const InteractionTarget& target,
      const GraspAffordance& affordance,
      const std::vector<PickNavigationObstacle>& obstacles,
      const PickSlotConfig& config = {});

  PickSlotReason revalidate_frozen_pick_slot(
      Transform live_root,
      Transform frozen_root,
      const InteractionTarget& target,
      const std::vector<PickNavigationObstacle>& obstacles,
      const PickSlotConfig& config = {});
  ```

- [x] **Step 1: Write failing object-planar mapping tests**

  In `tests/cpp/test_interaction_pick_slots.cpp`, create one target with a slot
  `{7U, 0.20F, -0.40F, 0.25F}`. Assert:

  ```cpp
  // Identity object at (1, 0.8, 3), live root Y = 0.
  mapped.position == vec3(1.20F, 0.0F, 2.60F);
  mapped_yaw == 0.25F;

  // Target translated to (4, 0.8, -2) and yawed +90 degrees.
  translated_rotated.position == vec3(3.60F, 0.0F, -2.20F);
  translated_rotated_yaw == 0.25F + 0.5F * PIf;
  ```

  Add cases proving object pitch/roll do not tilt root Y, a live root Y of
  `1.25F` is preserved exactly, and a quaternion whose object-forward projects
  to zero in XZ yields `PickSlotReason::InvalidGeometry`.

- [x] **Step 2: Write failing direct-clearance tests**

  Cover all exact geometry contracts with real pure-function calls:

  - A rotated table whose expanded rectangle intersects only the middle of the
    segment rejects with `TableBlocked` although both endpoints are outside.
  - Touching the table's `half_extent + 0.24F` boundary is blocked; one
    representable float farther out is clear.
  - A world-axis-aligned blocker centered on the route rejects at exact 3D
    distances `0.60000F` and `0.60002F`, while vertical separation producing
    distance `0.60003F` is clear. The next representable distance below/above
    the configured `radius + margin` must also reject/clear respectively.
  - A corner case with segment/AABB minimum distance `0.60003F`, but overlap
    with the AABB expanded independently on every axis, is clear. This prevents
    replacing sphere/AABB distance with a conservative expanded-box shortcut.
  - A blocker between endpoints is rejected even when both endpoints clear it.
  - Zero/negative size, NaN center, and mismatched malformed values fail closed
    as `InvalidGeometry`, retaining the first malformed obstacle's input index;
    if no candidate can be valid, selection reason is `InvalidGeometry`.
  - Moving the tabletop target across the route does not change eligibility;
    the two derived target-distance diagnostics do change.
  - Direct length exactly `1.00002F` is eligible and `1.00003F` is not.

- [x] **Step 3: Write failing deterministic-ranking tests**

  Provide at least four clear authored slots and assert ordering by the exact
  tuple:

  ```cpp
  std::tuple{
      candidate.route_millimetres,
      candidate.heading_milliradians,
      candidate.id,
  }
  ```

  Cases must prove:

  - a one-millimetre shorter route beats a smaller heading change;
  - routes in the same millimetre bucket use heading;
  - an exact route/heading tie uses the lower nonzero slot ID, independent of
    authored vector order;
  - a route no longer than `1e-5F` uses mapped slot yaw for heading;
  - half-millimetre and half-milliradian values round half-up;
  - NaN, infinity, and a finite value whose scaled double exceeds
    `uint64_t::max()` reject rather than wrapping;
  - one blocked nearest slot permits the next clear slot;
  - no authored slots reports `NoAuthoredSlot`, all over-distance reports
    `OutsideTravelEnvelope`, and a nonempty all-blocked set reports
    `AllSlotsBlocked`.

  Add malformed `PickSlotConfig` cases for zero, negative, NaN, infinity, and
  overflow-prone travel/tolerance, table expansion, obstacle radius, and safety
  margin. Every constructor/call must fail closed with the same result in normal
  and fast-math builds.

- [x] **Step 4: Add the explicit Make target and verify the red failure**

  Add `build/tests/test_interaction_pick_slots` to `CPP_TEST_BINS`, with only
  `interaction_pick_slots.cpp`, `interaction_target.cpp`, and
  `interaction_pose.cpp` plus their headers as production prerequisites.

  Run:

  ```bash
  make build/tests/test_interaction_pick_slots
  ```

  Expected: compilation fails because the new interface has no implementation.

- [x] **Step 5: Implement finite object-planar mapping and half-up keys**

  Derive target planar forward from object rotation applied to world +Z,
  normalize only XZ, and use
  `right = cross(world_up, forward)`. Map with:

  ```cpp
  const vec3 position = vec3(
      target.object_world.position.x,
      live_root.position.y,
      target.object_world.position.z) +
      slot.root_x_object_m * right +
      slot.root_z_object_m * forward;
  const float yaw = std::atan2(forward.x, forward.z) +
      slot.root_yaw_object_radians;
  ```

  Normalize yaw with `atan2(sin(yaw), cos(yaw))`. Compute nonnegative keys in
  double with `floor(value * scale + 0.5)` after a pre-cast overflow check;
  scales are `1000.0` for metres and radians. Detect finite IEEE-754 floats by
  inspecting exponent bits with `memcpy`, as existing interaction code does;
  do not use `std::isfinite` in this fast-math-sensitive seam. Validate every
  `PickSlotConfig` scalar before mapping any candidate.

- [x] **Step 6: Implement exact closed-segment table and sphere/AABB clearance**

  For the table, transform segment endpoints by inverse table yaw and apply a
  two-axis slab intersection against half extents plus `0.24F`; use slab epsilon
  `1e-7F` and treat boundary contact as intersection.

  For each controller obstacle, compute exact squared distance from the 3D
  segment to the AABB. Split parameter `[0,1]` at every finite crossing of each
  coordinate with the AABB's min/max plane. On each interval, determine active
  outside axes at its midpoint; their squared distance is one quadratic in
  `t`. Evaluate both interval endpoints and the clamped quadratic minimum. The
  smallest value across intervals is exact for the piecewise-quadratic distance
  function. Require strict separation:

  ```cpp
  minimum_distance >
      config.obstacle_root_radius_m +
      config.obstacle_safety_margin_m;
  ```

  Iterate blockers in input order and retain the first failing index.

- [x] **Step 7: Implement selection and frozen-route revalidation**

  Map/evaluate every authored slot in authored order for diagnostics. Select
  among `reason == PickSlotReason::None` by the quantized tuple without sorting
  or mutating the authored vector. `revalidate_frozen_pick_slot` evaluates the
  closed remaining XZ segment from the current root to the exact frozen root;
  for generic 3D AABB clearance it evaluates both endpoints at the current
  observed root Y, even if the stored frozen root has a different Y. It does not
  mutate the frozen transform, map, or choose another slot. Add a test with
  differing current/frozen Y that distinguishes this fixed-current-Y result from
  a sloped-segment result.

- [x] **Step 8: Run focused normal and fast-math tests**

  Run:

  ```bash
  make build/tests/test_interaction_pick_slots && \
    build/tests/test_interaction_pick_slots
  g++ -std=c++17 -Wall -Wextra -Werror -pedantic -ffast-math -I. \
    tests/cpp/test_interaction_pick_slots.cpp \
    interaction_pick_slots.cpp interaction_target.cpp interaction_pose.cpp \
    -o build/tests/test_interaction_pick_slots_release_fast_math && \
    build/tests/test_interaction_pick_slots_release_fast_math --fast-math-canary
  ```

  Expected: both modes pass and print identical selected slot IDs and rejection
  reasons for the shared canary matrix.

### Task 3: Replace the Manual Funnel State Machine with One Frozen Slot

**Files:**
- Modify: `interaction_pick_assist.h`
- Modify: `interaction_pick_assist.cpp`
- Rewrite manual cases in: `tests/cpp/test_interaction_pick_assist.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: a pre-step latched target snapshot plus the post-step current target
  and live root, copied immutable obstacles, current displayed/simulation state,
  and at most one runtime `PickEntryPreview` for the frozen prospective root.
- Produces:

  ```cpp
  enum class PickAssistState : uint8_t {
      Idle,
      SlotApproach,
      Settling,
      FinalPreview,
      ReadyToSubmit,
      Submitted,
      Failed,
  };

  enum class PickAssistReason : uint8_t {
      None,
      Cancelled,
      TargetUnavailable,
      TargetChanged,
      SlotChanged,
      RuntimeChanged,
      NoAuthoredSlot,
      InvalidGeometry,
      OutsideTravelEnvelope,
      TableBlocked,
      ObstacleBlocked,
      AllSlotsBlocked,
      ArrivalDeadline,
      PoorMatch,
      FinalPreviewRejected,
  };

  struct PickAssistStart {
      InteractionTarget target_snapshot{};
      uint32_t affordance_id = 0U;
      Transform root_world{};
      std::vector<PickNavigationObstacle> obstacles{};
  };

  struct PickAssistObservation {
      RuntimeState runtime_state = RuntimeState::Locomotion;
      const InteractionTarget* target = nullptr;
      Transform displayed_root{};
      vec3 simulation_velocity{};
      float displayed_planar_speed_mps = 0.0F;
      float camera_azimuth = 0.0F;
      uint64_t snapshot_fingerprint = 0U;
      uint64_t preview_snapshot_fingerprint = 0U;
      std::optional<PickEntryPreview> preview{};
  };
  ```

  `PickAssistDiagnostics` stores the complete mapped slot diagnostics, selected
  authored slot ID, accumulated assisted travel, derived object distances,
  arrival metrics, final preview details, and one stable reason. It stores no
  Plus/Minus identity, common-entry point, or hand score.

  Begin signature is:

  ```cpp
  bool begin(
      const PickAssistStart& start,
      const InteractionTarget* post_step_target);
  ```

  It validates the pre-step latch against the post-step target, then calls
  `select_pick_slot` exactly once from `start.root_world` before freezing.
  `pick_assist_reason_from_slot_reason` exhaustively maps every `PickSlotReason`
  value one-to-one (`None` through `AllSlotsBlocked`); target/runtime/arrival
  reasons are created only by their owning assist checks.

- [x] **Step 1: Replace old fixture expectations with a failing one-slot begin test**

  Build a target with three slots arranged so deterministic selection picks slot
  ID `9`. Pass the same target as the post-step observation and assert `begin`:

  ```cpp
  assist.diagnostics().state == PickAssistState::SlotApproach;
  assist.diagnostics().selected_slot_id == 9U;
  assist.diagnostics().route_length_m ==
      assist.diagnostics().slot_selection.ordered[
          *assist.diagnostics().slot_selection.selected_index].route_length_m;
  ```

  Add begin failures for the selector's aggregate no-winner results:
  `NoAuthoredSlot`, `InvalidGeometry`, `OutsideTravelEnvelope`, and
  `AllSlotsBlocked`. Separately assert the exhaustive mapping function returns
  the same-named assist reason for every enum value, including the
  revalidation-only `TableBlocked` and `ObstacleBlocked`, and `None` for
  `None`. Assert a failed assist owns no input and can begin a new valid attempt.

- [x] **Step 2: Write failing approach and travel-accounting tests**

  Starting `0.80 m` from the frozen slot, assert the first observation emits an
  ordinary camera-relative left stick, does not request preview, and does not
  change any input target/pose value. Move the observed displayed root along the
  route and assert:

  - travel accumulates from consecutive observed XZ roots;
  - exactly `1.00002 m` remains legal;
  - the next positive representable step beyond the cap fails
    `OutsideTravelEnvelope`;
  - remaining table or immutable-obstacle blockage maps to `TableBlocked` or
    `ObstacleBlocked` respectively and fails without choosing a different slot;
  - outside the slow radius, camera input changes only camera-relative command
    coordinates, not the frozen world slot;
  - inside the slow radius, output uses `arrival_navigation_stick`,
    `arrival_facing_stick`, and `force_strafe=true`.

- [x] **Step 3: Write failing exact identity, cancellation, and arrival tests**

  Independently mutate target generation, target object pose, ownership state,
  affordance grasp, selected slot geometry, selected slot ID, and slot order.
  Expect target changes to report `TargetChanged`, selected-slot/order changes to
  report `SlotChanged`, and no replacement slot to be chosen.

  Prove pre-submit `cancel()` clears steering/preview/submission in the same tick
  and leaves reason `Cancelled`. Prove cancellation after `Submitted` cannot
  create or revoke a request.

  For arrival, use the frozen slot and assert inclusive approach latch at:

  ```text
  root error = 0.03 m
  simulation planar speed = 0.05 m/s
  yaw error = 20 degrees
  ```

  One next representable value above each boundary must not latch. No assertion
  may use object-origin standoff.

- [x] **Step 4: Write failing settle, preview retry, and one-shot tests**

  After approach latch, feed five observations at root error `0.15 m`, displayed
  speed `0.10 m/s`, and yaw error `20 degrees`. Assert exactly the fifth stable
  tick enters `FinalPreview` and requests one preview root equal to the frozen
  mapped root. An unstable observation resets the consecutive count.

  Then exercise these one-preview outcomes:

  ```cpp
  // Retryable only.
  preview.path_feasible = true;
  preview.match_ready = false;
  preview.match_reason = Reason::PoorMatch;

  // Certified.
  preview.path_feasible = true;
  preview.match_ready = true;
  preview.path_reason = Reason::None;
  preview.match_reason = Reason::None;
  preview.prospective_root = frozen_prospective_root;
  ```

  A PoorMatch stays stationary and retries without changing slot until the
  existing arrival deadline, then fails `PoorMatch`. BlockedPath,
  CorrectionLimit, TargetUnavailable, a non-finite root, a mismatched
  fingerprint, or a different prospective root must never certify; hard
  runtime rejection fails `FinalPreviewRejected` immediately, while a missing
  same-tick preview remains stationary and requests it again.

  The first certified preview sets `submit_interact=true`. Two calls to
  `take_submission(71U)` return exactly one request equal to:

  ```cpp
  PickRequest{target.handle, affordance.id, 71U}
  ```

  and state becomes `Submitted`.

- [x] **Step 5: Run the rewritten tests and verify red failure**

  Run:

  ```bash
  make build/tests/test_interaction_pick_assist
  ```

  Expected: compilation fails on the new states/start/observation interfaces.

- [x] **Step 6: Implement the minimal frozen-slot state machine**

  Remove `reach_entry_distance_m`, `reach_entry_tolerance_m`,
  `maximum_preview_ticks`, `entry_point_`, two-slot preview arrays, and every
  call to `choose_pick_entry_slot`. Keep the existing arrival steering helpers.

  At every active observation, apply checks in this order:

  ```text
  runtime still Locomotion
  -> target exists with frozen handle/generation and remains Free/unowned
  -> selected affordance and ordered slots exactly match
  -> displayed metrics are finite
  -> add consecutive planar travel and enforce 1.00002 m cap
  -> revalidate only the remaining segment to the frozen root
  -> advance SlotApproach / Settling / FinalPreview
  ```

  Compare selected slot identity before the general affordance snapshot so a
  slot mutation receives `SlotChanged`. Do not write displayed root,
  simulation root, pose, joints, or prospective runtime snapshot.

  Implement the selector/revalidation reason conversion as an exhaustive switch
  with no permissive default and compile it under the repository's warnings-as-
  errors flags so a new enum value cannot silently fall through to a different
  failure reason.

- [x] **Step 7: Implement stationary PoorMatch retry and final diagnostics**

  Capture all preview conjuncts before branching. In `FinalPreview`, certify
  only exact fingerprint, prospective-root identity, `path_feasible`, and
  `match_ready`. Keep the output braked with `stationary_constraint=true` for a
  missing preview or sole `Reason::PoorMatch`. Track whether a PoorMatch was
  observed so deadline failure reports `PoorMatch`; other elapsed arrival
  failures report `ArrivalDeadline`.

- [x] **Step 8: Run focused and safe regressions**

  Run:

  ```bash
  make build/tests/test_interaction_pick_assist && \
    build/tests/test_interaction_pick_assist && \
    make test-interaction-safe
  ```

  Expected: the new manual assist tests and all preserved placement/interaction
  tests pass. The placement auto-demo still compiles against
  `interaction_pick_approach.*` unchanged.

### Task 4: Produce a Reproducible Manifest-Aware Slot Authoring Report

**Files:**
- Create: `resources/extract_g1_pick_slots.py`
- Create: `tests/python/test_extract_g1_pick_slots.py`
- Generate: `build/smart-pickup/beer10-slot-candidates.json`

**Interfaces:**
- Consumes: the exact Task 0 artifact set and target stable provenance
  `pickup_table__beer_10__001`.
- Produces:

  ```python
  def extract_slot_candidates(
      artifact: InteractionArtifact,
      manifest: Mapping[str, object],
      target_sequence_id: str,
  ) -> dict:
      """Return deterministic target metadata and static-feasible slots."""
  ```

  CLI:

  ```text
  python -m resources.extract_g1_pick_slots \
    --pack build/smart-pickup/full-pack \
    --target-sequence-id pickup_table__beer_10__001 \
    --output build/smart-pickup/beer10-slot-candidates.json
  ```

  This tool is offline authoring support. No C++ runtime target, controller, or
  Make runtime source list may import or execute it.

- [x] **Step 1: Write failing synthetic stable-identity and phase tests**

  Build a two-clip `InteractionArtifact` fixture with manifest clips deliberately
  supplied out of caller order. Assert extraction:

  - resolves the exact target by `sequence_id`, never clip ordinal;
  - derives first Reach, Contact, Lift, and Hold local frames;
  - uses object frame `contact - 1` for alignment;
  - reports stable key
    `(dataset_id, schema_version, sequence_id, entry_frame - range_start,
    active_hand)`;
  - rejects missing/duplicate sequence IDs, range mismatch, nonmonotonic phases,
    missing events, and a target in the held-out partition; and
  - preserves `source_frames[entry]` only as a separate provenance field.

- [x] **Step 2: Write failing transform, compatibility, and path tests**

  Use small canonical 31-bone poses with a nonidentity object yaw. Assert exact
  extraction of:

  ```text
  root_x_object_m
  root_z_object_m
  root_yaw_object_radians
  entry/contact/lift/hold/stop local frames
  contact hand position/orientation error to target affordance
  root/table and hand/table/object static path status
  entry root planar speed
  ```

  Add boundary cases for hand error `0.12 m`, hand orientation `25 degrees`,
  table proxy `0.24 m`, clearance radius `0.04 m`, and Contact-minus-one object
  exemption. Equality at correction thresholds is accepted; equality at a
  collision boundary is blocked. A candidate whose entry clears but later Hold
  root enters the expanded table must be rejected.

- [x] **Step 3: Write failing deterministic dedupe/output tests**

  Feed candidates in reverse order and assert byte-identical JSON output sorted
  by stable provenance. Greedily retain a candidate unless an already-retained
  same-hand candidate is within both `0.10 m` planar root distance and
  `10 degrees` root yaw. Assert the report includes rejected counts by exact
  static gate and never serializes a runtime clip allowlist.

- [x] **Step 4: Run the focused tests and verify red failure**

  Run:

  ```bash
  python -m unittest tests.python.test_extract_g1_pick_slots -v
  ```

  Expected: import failure because `resources.extract_g1_pick_slots` does not
  exist.

- [x] **Step 5: Implement the minimal offline extractor**

  Reuse `read_artifact_set` for schema/manifest validation. Reconstruct canonical
  world transforms in parent order using the manifest skeleton parents. Derive
  source-to-target alignment exactly as the matcher does:

  ```python
  yaw_delta = wrap(target_object_yaw - source_object_yaw_at_contact_minus_one)
  mapped_xz = target_object_xz + rotate_yaw(
      source_value_xz - source_object_xz_at_contact_minus_one,
      yaw_delta,
  )
  ```

  Evaluate every integer root frame from Reach through `stop - 1`, every hand
  segment against the table, and pre-contact hand segments against the object.
  Use float64 for geometry but serialize reviewed float32 source values with
  nine decimal digits. Reject non-finite input before arithmetic. Mark the
  resulting records `static_path_feasible`; do not call them runtime-certified.

- [x] **Step 6: Run focused tests green**

  Run:

  ```bash
  python -m unittest tests.python.test_extract_g1_pick_slots -v
  ```

  Expected: every stable identity, transform, boundary, path, dedupe, and
  deterministic serialization test passes.

- [x] **Step 7: Generate and inspect the full-pack beer target report**

  Run:

  ```bash
  python -m resources.extract_g1_pick_slots \
    --pack build/smart-pickup/full-pack \
    --target-sequence-id pickup_table__beer_10__001 \
    --output build/smart-pickup/beer10-slot-candidates.json
  python - <<'PY'
  import json
  from pathlib import Path

  report = json.loads(
      Path("build/smart-pickup/beer10-slot-candidates.json").read_text()
  )
  assert report["target"]["sequence_id"] == "pickup_table__beer_10__001"
  assert report["target"]["active_hand"] == 1
  assert len(report["retained_candidates"]) >= 3
  assert all(item["static_path_feasible"] for item in report["retained_candidates"])
  assert all(item["stable_key"][0] == report["dataset_id"] for item in report["retained_candidates"])
  print(len(report["retained_candidates"]), "reviewable static slot candidates")
  PY
  ```

  Expected: at least three deterministically ordered reviewable candidates. This
  step is not sufficient to claim `match_ready`; Task 5 applies unrestricted
  runtime preview and realized-transition clearance.

### Task 5: Certify Slots with Unrestricted Runtime Preview and Bake the Scene

**Files:**
- Create: `interaction_smart_pickup_scene.h`
- Create: `interaction_smart_pickup_scene.cpp`
- Create: `interaction_smart_pickup_preview_probe.cpp`
- Create: `interaction_smart_pickup_scene_probe.cpp`
- Create: `tests/cpp/test_interaction_smart_pickup_scene.cpp`
- Create: `tests/python/test_smart_pickup_full_pack_gate.py`
- Create: `docs/superpowers/specs/2026-07-17-smart-pickup-authored-data.md`
- Modify: `Makefile`

**Interfaces:**
- Consumes: full pack, ordinary flat `resources/database.bin`, Task 4 static
  report, runtime `preview_pick`, and stationary flat frames derived by
  `stationary_motion_matching::derive_candidates`.
- Produces:

  ```cpp
  struct SmartPickupSlotProvenance {
      uint32_t slot_id = 0U;
      const char* sequence_id = nullptr;
      int32_t entry_local_frame = -1;
      int32_t contact_local_frame = -1;
  };

  InteractionTarget make_smart_pickup_demo_target();
  PlacementSurface make_smart_pickup_demo_destination_surface(
      const InteractionTarget& source_target);
  const std::array<SmartPickupSlotProvenance, 3>&
      smart_pickup_demo_slot_provenance();
  ```

  Safe authoring probe CLI:

  ```text
  interaction_smart_pickup_preview_probe \
    resources/database.bin \
    build/smart-pickup/full-pack \
    build/smart-pickup/beer10-slot-candidates.json
  ```

  The probe reads candidate provenance/root data but never restricts the runtime
  matcher to candidate clips. It emits one compact JSON object per candidate and
  one final `selected_slots` object.

  Baked-scene probe CLI:

  ```text
  interaction_smart_pickup_scene_probe --json
  ```

  It emits the compiled target constants and every ordered slot ID/local root
  plus provenance using float32 hexadecimal bit patterns as comparison
  authority. The Python gate requires this output to match the preview probe's
  final `selected_slots` object byte-for-byte after canonical JSON encoding.

- [x] **Step 1: Write failing baked-scene unit tests**

  Assert the authored target has handle `{1,1}`, profile `1`, Free/unowned
  state, one right-hand affordance ID `1`, exactly three nonzero ordered slot
  IDs, zero-centered bounds, and these frozen beer target values:

  ```text
  table position = (0, 0.360757500, 3)
  table rotation = (1, 0, 0, 0)
  table size = (2, 0.0399999991, 0.600000024)
  object position = (0.00394439697, 0.503655553, 2.77000808716)
  object rotation = (-0.0669774629, 0.670088462,
                     0.0799996098, -0.734911910)
  object dimensions = (0.0645366386, 0.0645366609, 0.240097240)
  grasp position object = (0.0930671170, -0.119263843, 0.0375832170)
  grasp rotation object = (0.308746904, 0.0609171167,
                           -0.155041456, 0.936443567)
  approach object = (-0.997760296, 0, 0.0668911785)
  clearance radius = 0.04
  ```

  Assert the destination is the same support translated exactly `+1.20 m` in
  world Z and passes `evaluate_placement_fit` for the baked object bounds.

- [x] **Step 2: Add the safe probe targets and verify red failures**

  Add explicit Make targets for `test_interaction_smart_pickup_scene`,
  `interaction_smart_pickup_preview_probe`, and
  `interaction_smart_pickup_scene_probe`. The preview probe links existing
  flat-pose bridge, runtime, matcher, IK, attachment, Carry, and target sources;
  the scene probe links only the baked scene/target/placement dependencies.
  Neither target links or invokes the protected repository-root query probe.

  Run:

  ```bash
  make \
    build/tests/test_interaction_smart_pickup_scene \
    interaction_smart_pickup_preview_probe \
    interaction_smart_pickup_scene_probe
  ```

  Expected: compilation fails because the scene/probe sources do not exist.

- [x] **Step 3: Write the failing manifest-aware preview-and-bake gate**

  First create `tests/python/test_smart_pickup_full_pack_gate.py`. The gate
  receives `SMART_PICKUP_FULL_PACK`, `SMART_PICKUP_PREVIEW_PROBE`, and
  `SMART_PICKUP_SCENE_PROBE`. Before launching C++, assert:

  ```python
  manifest["diagnostic_limit"] is None
  len(manifest["clips"]) == 2045
  report["included_frames"] == 511250
  sha256(database_bin) == "4d3b65f73e9a207988aaaebded36b988f811ec068988c7e829732701d9d2da1b"
  sha256(features_bin) == "3b492ca7e5ed12aade5750ff925c689f4acf56f28edc31a4a5f341e434adf145"
  ```

  Join every probe candidate ordinal and selected matcher ordinal back to the
  manifest. Select the first three runtime-ready candidates in stable provenance
  order after the Task 4 `0.10 m AND 10 degrees` dedupe. Fail if fewer than
  three remain, if any source ID/local entry differs from the Task 4 report, or
  if the probe used zero/multiple target registrations.

  Add synthetic parser tests before production code. They require the final
  `selected_slots` record and compiled scene record to have identical ordered
  slot IDs, stable sequence/local entry/contact provenance, and float32 hex bits
  for all three local root scalars. Independently mutate each field, reorder two
  rows, add/drop a row, and change a target scalar; every mutation must fail.
  This is the mechanical guard against copying a certified initializer
  incorrectly into the baked scene.

  Run once before implementation and expect failure because the new gate module
  and executable are unavailable:

  ```bash
  SMART_PICKUP_FULL_PACK=build/smart-pickup/full-pack \
  SMART_PICKUP_PREVIEW_PROBE=./interaction_smart_pickup_preview_probe \
  SMART_PICKUP_SCENE_PROBE=./interaction_smart_pickup_scene_probe \
    python -m unittest tests.python.test_smart_pickup_full_pack_gate -v
  ```

  Expected: failure because the probe executables and gate implementation do
  not yet satisfy the new behavioral contract. A compile-failure-only check is
  not accepted as probe TDD evidence.

- [x] **Step 4: Implement the unrestricted stationary preview probe**

  Reuse the proven flat-pose bridge logic from
  `tests/cpp/test_live_flat_pick_entry_oracle.cpp`: validate the 23-bone flat
  parent tree, expand each stationary flat frame against one fixed G1 reference
  pose, and populate future roots without crossing a flat clip range.

  For every Task 4 retained candidate, call:

  ```cpp
  runtime.preview_pick(
      stationary_snapshot,
      candidate.prospective_root,
      target_handle,
      affordance_id);
  ```

  across all validated stationary flat frames until at least one result has
  `path_feasible && match_ready`. Report the lowest total cost, then lowest flat
  frame on an exact cost tie. Include candidate stable source ID/local entry,
  prospective root and its float32 hex bits, path/match reasons,
  feasible/contact frames, total cost, selected runtime clip/entry, snapshot
  fingerprint, and preview count.

  Build target metadata from the report's beer target record and insert it
  through `TargetRegistry::upsert`. The candidate source ordinal may define only
  the prospective root; it must never be passed as a matcher allowlist. Emit one
  final canonically ordered `selected_slots` record for the Python gate.

- [x] **Step 5: Run the probe and freeze the first three certified constants**

  Run:

  ```bash
  make interaction_smart_pickup_preview_probe
  ./interaction_smart_pickup_preview_probe \
    resources/database.bin \
    build/smart-pickup/full-pack \
    build/smart-pickup/beer10-slot-candidates.json \
    > build/smart-pickup/beer10-runtime-preview.jsonl
  ```

  The unrestricted runtime probe certified all 19 retained Task 4 candidates.
  The first three runtime-ready candidates in actual stable Task 4 order are:

  ```text
  pickup_table__alcohol_10__005 entry 125 contact 150 (-0.396769345,-0.0414382927,1.38969707) id 1
  pickup_table__alcohol_13__005 entry 118 contact 143 (-0.402728528,-0.244846597,1.71573567) id 2
  pickup_table__apple_1__000    entry 90  contact 115 (-0.467868507,-0.194983453,1.29209125) id 3
  ```

  Output was byte-identical across repeated full runs (20 JSONL records,
  214,595 bytes, SHA-256
  `3b8b1375a587c9f5b2e04db4578e8e855e0f6099c656d21d2ecbab5aaab68110`).
  The Python gate independently recomputes this choice and verifies the final
  `selected_slots` record. No gate was loosened and no motion was hand-selected
  by appearance.

- [x] **Step 6: Bake only the certified literal scene values**

  Before entering the data, extend the scene test with the exact three ordered
  IDs, local-root float32 bit patterns, and provenance rows emitted by Step 5;
  run it and record the expected red mismatch against the empty/unfinalized
  scene. Then implement `interaction_smart_pickup_scene.cpp` with the exact
  target values above and those three initializer rows. The runtime file may
  contain stable source strings for diagnostics, but no clip ordinals, global
  frames, manifest parser, candidate subset, file I/O, or runtime generation.

  `make_smart_pickup_demo_destination_surface` must derive its geometry from the
  passed authored target, preserve object-in-support transform, translate the
  table `+1.20 m` in Z, and call existing placement-fit certification.

  Implement `interaction_smart_pickup_scene_probe.cpp` to serialize the compiled
  target and ordered `smart_pickup_demo_slot_provenance()` rows with the same
  canonical keys and float32 hex representation as `selected_slots`. Build and
  run it, then run the Python gate. Any slot value, order, ID, provenance, or
  target mismatch must fail before controller integration:

  ```bash
  make interaction_smart_pickup_scene_probe
  SMART_PICKUP_FULL_PACK=build/smart-pickup/full-pack \
  SMART_PICKUP_PREVIEW_PROBE=./interaction_smart_pickup_preview_probe \
  SMART_PICKUP_SCENE_PROBE=./interaction_smart_pickup_scene_probe \
    python -m unittest tests.python.test_smart_pickup_full_pack_gate -v
  ```

- [x] **Step 7: Write the authored-data record**

  Record the exact Task 0 hashes/counts, target source key and local
  entry/contact frames, every selected slot's stable key/local event frames,
  object-planar constants, mapped demo-world root, preview stationary frame,
  selected unrestricted matcher source ID/frame/cost, and the statement:

  ```text
  Source provenance is authoring evidence only. Runtime slot IDs do not restrict
  the global motion matcher and are not PickRequest authority.
  ```

- [x] **Step 8: Run scene, full-pack, and safe regressions**

  Run:

  ```bash
  make build/tests/test_interaction_smart_pickup_scene && \
    build/tests/test_interaction_smart_pickup_scene
  SMART_PICKUP_FULL_PACK=build/smart-pickup/full-pack \
  SMART_PICKUP_PREVIEW_PROBE=./interaction_smart_pickup_preview_probe \
  SMART_PICKUP_SCENE_PROBE=./interaction_smart_pickup_scene_probe \
    python -m unittest tests.python.test_smart_pickup_full_pack_gate -v
  make test-interaction-safe
  ```

  Expected: the scene test, all three unrestricted full-pack preview
  certifications, manifest joins, and the existing safe suite pass.

### Task 6: Wire Two-Phase Manual Activation into the 25 Hz Controller

**Files:**
- Create: `interaction_smart_pickup_controller.h`
- Create: `interaction_smart_pickup_controller.cpp`
- Create: `tests/cpp/test_interaction_smart_pickup_controller.cpp`
- Modify: `controller.cpp`
- Modify: `interaction_controller_adapter.h`
- Modify: `interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_live_flat_pick_entry_oracle.cpp`
- Modify: `tests/python/test_playable_interaction_evidence.py`
- Modify: `tests/python/test_playable_placement_evidence.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: one pre-step interact edge, the exact pre-step Smart Pickup target
  snapshot, one authoritative post-step `live_flat_snapshot`, copied controller
  obstacle arrays, the Task 5 scene, and `InteractionPickAssist`.
- Produces: a pending activation that owns no locomotion before the post-step
  snapshot, one manual preview call for the frozen authored slot, ordinary
  assisted locomotion beginning on the following tick, and at most one
  `PickRequest`.

  `SmartPickupController` is a raylib-free production class, not a test harness.
  Its private pending record is conceptually:

  ```cpp
  struct PendingManualPickActivation {
      InteractionTarget target_snapshot{};
      uint32_t affordance_id = 0U;
  };
  ```

  It deliberately contains no root, slot, motion clip, preview, or generated
  waypoint. `PickAssistStart::root_world` is populated only from the single
  authoritative post-step live-flat snapshot. Its public boundary is:

  ```cpp
  using SmartPickupPreviewCallback = std::function<std::optional<PickEntryPreview>(
      const LocomotionSnapshot&,
      PickEntryRoot,
      TargetHandle,
      uint32_t)>;

  SmartPickupPreStepResult pre_step(const SmartPickupPreStepInput& input);

  SmartPickupPostStepResult post_step(
      const SmartPickupPostStepInput& input,
      const SmartPickupPreviewCallback& preview_pick);
  ```

  `pre_step` captures/cancels intent and reports input consumption. The caller
  performs its one ordinary locomotion step. `post_step` receives that step's
  sole live-flat snapshot, current target, immutable copied obstacles, displayed
  metrics, camera azimuth, and next request ID. It begins a pending assist,
  calls unrestricted preview at most once when the already-active assist asks
  for it, observes exactly once, and returns at most one extracted
  `PickRequest`; it does not step locomotion or submit to the scheduler.
  Native/headless callers supply a narrow lambda that delegates directly to
  `InteractionRuntime::preview_pick`; tests supply a counting callback.

- [x] **Step 1: Write failing source-contract tests for the manual/placement split**

  Update `tests/python/test_playable_interaction_evidence.py` to assert that:

  - manual interact uses `make_smart_pickup_demo_target` and the one-slot
    `InteractionPickAssist` path;
  - no manual branch calls `choose_pick_entry_slot`,
    `make_pick_reach_waypoint`, or writes root/pose/joint arrays;
  - ordinary manual Smart Pickup and acceptance mode require an explicit
    `MM_INTERACTION_PACK` whose loaded database is 25/1 Hz, 2,045 clips, and
    511,250 frames with matching feature-frame count; missing/mismatched packs
    fail startup rather than falling back to `resources/g1_interaction`; and
  - every controller update involved in activation remains `1.0F / 25.0F`.

  Update `tests/python/test_playable_placement_evidence.py` so it no longer
  counts every textual `preview_pick` occurrence in `controller.cpp`. Instead,
  assert that the legacy placement helper still performs exactly the original
  ordered Plus-then-Minus pair and the new manual helper performs exactly one
  preview for its frozen prospective root. Preserve every existing placement
  scheduler and auto-demo assertion.

  Run:

  ```bash
  python -m unittest \
    tests.python.test_playable_interaction_evidence \
    tests.python.test_playable_placement_evidence -v
  ```

  Expected: the new manual assertions fail because the controller still uses
  the legacy fixed-two pickup path.

- [x] **Step 2: Write a failing two-phase production-coordinator test**

  Add `tests/cpp/test_interaction_smart_pickup_controller.cpp` around the new
  production class and the existing adapter seams.
  Count locomotion provider calls, live-flat bridge calls, assist `begin` calls,
  assist observations, runtime previews, and submissions. On a tick with an F
  edge assert this order:

  ```text
  capture exact target intent
  -> consume interact and zero manual sticks
  -> perform exactly one ordinary 25 Hz locomotion step
  -> create exactly one live_flat_snapshot
  -> re-read and exactly compare the current target
  -> copy the same ordered obstacle centers/sizes supplied to locomotion
  -> select/freeze one slot with the post-step root
  -> observe the assist once with that same snapshot/fingerprint
  -> publish scheduler/debug state
  ```

  The activation tick must report zero assisted displacement, zero preview
  calls, zero submissions, and identical observation/scheduler snapshot
  fingerprints. Assisted steering may affect only the next 25 Hz tick.

- [x] **Step 3: Write failing precedence, mutation, and failure tests**

  Cover these exact controller-level cases:

  - simultaneous X and F applies cancel first; no pending activation, selection,
    preview, or request is created;
  - target generation, pose, state/owner, affordance, slot order, or slot value
    mutation during the one locomotion step consumes the F edge and fails
    `TargetChanged` because no slot existed before freezing; the same slot
    mutations after successful `begin` fail `SlotChanged`;
  - a missing target or selector failure consumes the edge, emits the stable
    failure diagnostic, and produces no `PickRequest`;
  - an active manual assist owns the sticks, but camera look and cancellation
    remain responsive;
  - controller obstacle centers/sizes passed to locomotion and converted into
    `PickNavigationObstacle` are element-for-element identical and remain
    frozen for the attempt; and
  - every activation case has exactly one locomotion provider call, one bridge
    snapshot, and one assist observation—never an extra hidden step.

- [ ] **Step 4: Migrate the old live-flat oracle before implementation**

  In `tests/cpp/test_live_flat_pick_entry_oracle.cpp`, retain the proven
  flat-pose bridge, stationary-snapshot, and legacy placement Plus/Minus tests.
  Remove or rewrite only manual assertions that require the old fixed-two
  pickup chooser. Route any retained exact target comparison through
  `same_interaction_target_snapshot`, including ordered slots. Do not weaken the
  placement oracle or delete shared bridge coverage.

  Run the focused targets and confirm red failures are attributable to missing
  two-phase wiring, not stale fixed-two expectations:

  ```bash
  make \
    build/tests/test_interaction_controller_adapter \
    build/tests/test_live_flat_pick_entry_oracle
  ```

- [ ] **Step 5: Implement the shared pre-step/post-step production coordinator**

  Implement `SmartPickupController` first, then replace only the native manual
  pickup branch with calls to it. On the F edge while runtime is in
  Locomotion and the known Smart Pickup demo target is Free/unowned:

  1. copy the exact target and affordance ID into
     `PendingManualPickActivation`;
  2. consume F and force movement/facing sticks to zero for that tick;
  3. perform the already scheduled single 25 Hz locomotion update;
  4. construct one authoritative live-flat snapshot;
  5. re-fetch the target and require exact equality with the captured snapshot;
  6. copy the controller's exact obstacle arrays into `PickAssistStart`;
  7. set `start.root_world` from the post-step snapshot and call `begin` once;
  8. call `observe` once with the same snapshot and publish its fingerprint.

  Clear the pending record on every success or failure. Do not map a slot,
  preview a motion, or begin steering before step 4. Do not use object-origin
  distance or the legacy `1.45 m` interaction hint as an activation gate.

  `controller.cpp` must contain no second copy of the pending/activation state
  machine: it gathers raylib/controller inputs, calls `pre_step`, advances its
  existing locomotion provider once, constructs the snapshot, then calls
  `post_step`. The Task 7 oracle invokes the same class.

- [ ] **Step 6: Add one isolated manual preview and one-shot submission path**

  Inside the shared coordinator, add a narrowly named helper that accepts the
  already-frozen prospective root and invokes `preview_pick` exactly once.
  Keep the
  legacy placement helper and its Plus/Minus call order byte-for-byte equivalent
  in behavior. Feed the manual preview and snapshot fingerprint into the assist;
  when `take_submission(next_request_id)` returns a request, return it in
  `SmartPickupPostStepResult`. The native controller and headless oracle each
  hand that result to the same existing scheduler submission seam once and
  increment the request ID once; neither reimplements assist extraction.

  A missing or sole PoorMatch preview keeps the assist stationary as specified
  by Task 3. No controller code may inspect the selected runtime clip to alter
  the authored slot or route.

- [ ] **Step 7: Wire the baked scene without changing auto-demo fixtures**

  Manual mode registers `make_smart_pickup_demo_target()` and uses
  `make_smart_pickup_demo_destination_surface()` for the eventual place. Keep
  existing clip-0-derived target creation for legacy auto-demo/probe fixtures so
  their hashes and Plus/Minus evidence remain stable. The manual scene contains
  one known target by design; multi-target choice is a later checkpoint.

  Enforce the explicit full-pack shape before registering the manual scene.
  This runtime guard rejects the one-clip diagnostic pack; Tasks 5/7/9 retain
  exact binary hash authority. Only the separately detected legacy interaction
  and placement auto-demo fixture modes may load their diagnostic pack.

- [ ] **Step 8: Run focused, structural, and safe regressions**

  Run:

  ```bash
  make \
    build/tests/test_interaction_smart_pickup_controller \
    build/tests/test_interaction_controller_adapter \
    build/tests/test_live_flat_pick_entry_oracle && \
    build/tests/test_interaction_smart_pickup_controller && \
    build/tests/test_interaction_controller_adapter && \
    build/tests/test_live_flat_pick_entry_oracle \
      resources/database.bin resources/g1_interaction --exhaustive
  python -m unittest \
    tests.python.test_playable_interaction_evidence \
    tests.python.test_playable_placement_evidence -v
  make test-interaction-safe
  ```

  Expected: two-phase counts/order pass, old manual fixed-two behavior is gone,
  placement Plus/Minus behavior is unchanged, and the complete safe suite is
  green. The retained placement-only oracle intentionally uses its legacy
  diagnostic pack so clip-0 fixture hashes remain stable; new Smart Pickup
  preview/oracle/acceptance targets use only the full pack.

### Task 7: Prove Varied Starts, Target Transforms, and Blocker Behavior Headlessly

**Files:**
- Create: `interaction_smart_pickup_scenarios.h`
- Create: `interaction_smart_pickup_scenarios.cpp`
- Create: `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`
- Create: `tests/cpp/test_live_flat_smart_pickup_oracle.cpp`
- Modify: `tests/python/test_smart_pickup_full_pack_gate.py`
- Modify: `tests/python/test_playable_interaction_evidence.py`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: the explicit full pack, the ordinary flat database, the baked
  Smart Pickup target/slots, real controller obstacle arrays, unrestricted
  runtime preview, actual Preflight/Playback/Carry state transitions, and the
  existing placement scheduler.
- Produces: deterministic JSONL evidence for walk -> pick -> carry -> place,
  including selected authored slot, route/blocker diagnostics, unrestricted
  selected motion provenance, state sequence, contact/attachment/release, and
  continuity metrics.

  Oracle CLI:

  ```text
  test_live_flat_smart_pickup_oracle \
    resources/database.bin \
    build/smart-pickup/full-pack \
    --jsonl build/smart-pickup/headless-acceptance.jsonl
  ```

- [ ] **Step 1: Write the failing full-pack identity preflight**

  Before constructing runtime, require the manifest/report counts and the exact
  Task 0 database/features SHA-256 values. Reject a pack with non-null
  `diagnostic_limit`, one missing selected provenance sequence, wrong FPS, or a
  substituted diagnostic pack. Join every runtime-selected clip ordinal back to
  its manifest sequence ID in evidence.

  Run once with the existing one-clip pack and expect a clean preflight failure;
  run with the full pack and proceed to currently missing oracle cases.

- [ ] **Step 2: Write failing production scenario and transformed-target cases**

  Define `SmartPickupDemoScenario` in the new raylib-free production unit with
  stable ID, initial root, target/destination planar transform, ordered
  immutable blockers, and expected class (clear, alternate-slot, all-blocked).
  Write its unit tests first, then implement the minimal shared table containing
  at least this deterministic matrix, all at root Y `0`:

  - three clear starts separated by at least `0.30 m` and spanning at least
    `45 degrees` of initial yaw;
  - one translated target/destination pair;
  - one target/destination pair yawed in the world while preserving every slot
    in object-local coordinates; and
  - ten exact repeats of each case.

  Each clear case must select a valid slot, and the matrix must exercise at
  least two distinct slot IDs. Repeated cases must produce identical selected
  slot ID, mapped root, request count, terminal state, and manifest-joined
  runtime clip/frame. The transformed cases must prove slot mapping changes in
  world space without modifying object-local constants.

  Add a selector-only compatible-object contract case with a distinct target
  handle/profile/dimensions but an explicit valid right-hand grasp and copied
  ordered slot template. It must map/select normally because Smart Pickup does
  not inspect semantic category. Add missing-slot and incompatible-grasp cases
  that fail explicitly. This does not claim full motion execution for an unseen
  mesh and does not add a second selectable object to the manual scene.

  Both this task's headless oracle and Task 8's native mode must consume the
  production scenario records directly; neither may duplicate their roots,
  target transforms, blockers, or IDs in test/controller-local arrays.

- [ ] **Step 3: Write failing blocker and frozen-slot cases**

  Add one controller AABB that blocks the nominal best route while leaving a
  different authored slot clear. Assert initial selection chooses the alternate
  slot and that the exact blocker arrays reach both ordinary locomotion and
  `InteractionPickAssist` unchanged. Add an all-routes-blocked case that fails
  before motion with no preview and no request.

  In a separate raylib-free revalidation case, keep the copied blocker arrays
  immutable and feed a finite observed root that has deviated so its remaining
  segment now intersects the same frozen blocker. Assert failure on that 25 Hz
  observation without hopping to another slot. Return the observed root to the
  earlier clear location and prove the failed attempt remains failed until a new
  F edge begins a new attempt. This is a state-machine observation test, not a
  production root write or a moving-obstacle claim.

- [ ] **Step 4: Write failing end-to-end state and continuity assertions**

  Drive the real headless controller/runtime path with ordinary assisted flat
  locomotion until the slot settles, then allow exactly one request. Assert the
  ordered state subsequence:

  ```text
  Locomotion
  -> Preflight
  -> Align
  -> PickupReplay (Reach -> Contact -> Lift/Hold phase progression)
  -> Hold
  -> Carry
  -> PlacePreflight
  -> PlaceAlign
  -> PlaceReplay
  -> PlaceRelease
  -> Locomotion
  ```

  Require one attachment edge, one release edge, and object ownership consistent
  with the runtime state. Measure root position, root yaw, each joint angle,
  head/neck position, active elbow, inactive arm, and object transform across
  every adjacent 25 Hz frame. Assert the already approved ceilings on the first
  run: non-owned root step `<= 0.20 m`, ordinary joint translation
  `<= 0.20 m`, distal foot/toe translation speed `<= 12 m/s`, and every joint
  rotation step `<= 60 degrees`. Also reuse the existing evidence bounds:
  quaternion norm error `<= 1e-3`, grasp composition error `<= 1e-5`, active
  hand position/orientation error `<= 0.01 m / 2 degrees` through
  PickupReplay/Hold/Carry, layered inactive-hand elevation `<= 0.10 m`, place
  root/yaw error `<= 0.25 m / 25 degrees`, final object error
  `<= 0.02 m / 10 degrees`, and support gap in `[-0.005, 0.020] m`. Preserve
  the existing placement IK request/acceptance bounds (`0.12 m / 25 degrees`
  and `0.04 m / 15 degrees`). Measured maxima are diagnostics only and may
  tighten a later reviewed contract; they never define this run's pass limit.
  Never hide a discontinuity with interpolation.

- [ ] **Step 5: Implement the raylib-free oracle using production seams**

  Iterate the production scenario table. Reuse the retained live-flat bridge
  helpers but invoke the production slot
  selector/assist through the exact Task 6 `SmartPickupController`, plus the
  controller obstacle conversion, unrestricted runtime, existing scheduler
  submission, attachment, Carry, placement scheduler, and release. Do not
  reproduce activation/observation/preview logic or any other algorithm in the
  test. Advance all components exactly once per `0.04 s` tick and cap each
  scenario with explicit approach, interaction, Carry, and placement deadlines.

  For deterministic headless steering, feed the assist's ordinary stick output
  into the same flat locomotion provider. Do not teleport the live root to the
  slot or directly force runtime phases.

- [ ] **Step 6: Add stable diagnostics and source-policy evidence**

  Emit one JSON object per tick/case with at least:

  ```text
  case_id, repeat, tick, fps, controller_state, runtime_state,
  selected_slot_id, mapped_slot_root, route_length, obstacle_index,
  snapshot_fingerprint, preview_count, request_count,
  matcher_sequence_id, matcher_entry_local_frame,
  root_delta, yaw_delta, maximum_joint_delta,
  head_delta, neck_delta, active_elbow_delta, inactive_arm_delta,
  object_delta, attached, released, failure_reason
  ```

  Extend Python evidence tests to require 25 Hz on every row, at least two slot
  IDs across clear cases, one alternate-slot blocker case, one all-blocked case,
  one complete Carry and placement release, one request per successful attempt,
  and no root/pose write seam in the manual controller path. Native overlay and
  graphical evidence wiring are implemented test-first in Task 8, not in this
  raylib-free oracle.

- [ ] **Step 7: Add normal/fast-math parity and repeat checks**

  Build the oracle normally and with `-ffast-math`. Run the selection/blocker
  matrix in both modes and compare stable case results byte-for-byte after
  excluding wall-clock timing fields. The full Playback trace need run only in
  normal mode, but every malformed and boundary geometry case remains in both.

  Run:

  ```bash
  make \
    build/tests/test_interaction_smart_pickup_scenarios \
    build/tests/test_live_flat_smart_pickup_oracle && \
    build/tests/test_interaction_smart_pickup_scenarios && \
    build/tests/test_live_flat_smart_pickup_oracle \
      resources/database.bin build/smart-pickup/full-pack \
      --jsonl build/smart-pickup/headless-acceptance.jsonl
  make test-interaction-pick-slots-fast-math
  SMART_PICKUP_FULL_PACK=build/smart-pickup/full-pack \
  SMART_PICKUP_PREVIEW_PROBE=./interaction_smart_pickup_preview_probe \
  SMART_PICKUP_SCENE_PROBE=./interaction_smart_pickup_scene_probe \
    python -m unittest \
      tests.python.test_smart_pickup_full_pack_gate \
      tests.python.test_playable_interaction_evidence -v
  ```

- [ ] **Step 8: Document the exact supported checkpoint**

  In `README.md`, add the full-pack build/selection command, controls (`F`
  interact, `X` cancel), debug colors, 25 Hz contract, and these explicit limits:
  one known right-hand rigid target, authored slots, direct clear route, flat
  locomotion, one carried object, and one destination. State that object-local
  slots generalize to translated/yawed targets and varied clear starts, not to
  unseen affordance geometry, shelves, doors, global obstacle planning, or
  learned proposals.

- [ ] **Step 9: Run the complete safe regression gate**

  Run:

  ```bash
  make test-smart-pickup-full-pack
  make test-interaction-safe
  ```

  `test-smart-pickup-full-pack` must depend on the explicit preview/scene probes
  and export all three gate variables to the Python test; it must fail rather
  than fall back when the generated full pack is absent. Expected: all Python
  tests, focused C++ tests, full-pack gates, and normal plus fast-math
  interaction checks pass without touching terrain-aware artifacts.

### Task 8: Add a Deterministic Native Scenario and Evidence Harness

**Files:**
- Modify: `interaction_smart_pickup_scenarios.h`
- Modify: `interaction_smart_pickup_scenarios.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`
- Create: `resources/validate_smart_pickup_graphical_evidence.py`
- Create: `tests/python/test_smart_pickup_graphical_evidence.py`
- Modify: `controller.cpp`
- Modify: `tests/python/test_playable_interaction_evidence.py`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: the exact Task 7 scenario matrix, the shared production
  `SmartPickupController`, existing flat-locomotion initialization and placement
  workflow, immutable controller blockers, and the normal raylib render loop.
- Produces: a fail-closed deterministic native mode selected only by:

  ```text
  MM_SMART_PICKUP_AUTODEMO=1
  MM_SMART_PICKUP_SCENARIO=<stable-case-id>
  MM_SMART_PICKUP_REPEAT=<0-through-9>
  MM_SMART_PICKUP_LOG=<new-jsonl-path>
  MM_SMART_PICKUP_SCREENSHOT=<new-png-path>
  MM_SMART_PICKUP_CAPTURE_DIR=<new-frame-directory>
  ```

  Manual play remains F/X controlled when the mode is absent. Existing
  `MM_INTERACTION_AUTODEMO` and placement auto-demo variables/behavior remain
  unchanged.

- [ ] **Step 1: Write failing raylib-free native-option tests**

  Extend the production scenario tests with a raylib-free native option parser.
  Require the exact Task 7 table, unique IDs, finite transforms, root Y `0`,
  starts separated by `>= 0.30 m`, a `>= 45 degree` yaw span, ten legal repeat
  indices, and blocker arrays identical to those handed to locomotion/assist.
  Parse scenario ID and repeat without reading global environment inside the
  pure unit; reject unknown IDs, duplicate IDs, malformed geometry, and repeat
  indices outside `[0,9]`.

  Run and record the expected missing-interface failure:

  ```bash
  make build/tests/test_interaction_smart_pickup_scenarios
  ```

- [ ] **Step 2: Write failing graphical evidence-validator tests**

  Create synthetic complete JSONL traces and assert the validator requires one
  row per 25 Hz simulation tick, the actual production state order and phase
  progression, exact scenario/reset metadata, one request/attachment/release,
  full-pack hashes, selected slot/matcher provenance, and every Task 7
  continuity/grasp/placement ceiling. Independently corrupt/drop/duplicate each
  required edge and exceed each hard bound by the smallest serialized amount;
  all must fail. A measured maximum below a ceiling is output summary only and
  cannot replace the ceiling.

  Require a nonblank PNG with existing pixel-variation rules. Paths must be
  nonempty, distinct, have existing parents, and publish atomically from
  per-process temporary files. Require one capture PNG named by simulation tick
  after every 25 Hz update so transition windows and video are tied to evidence
  rows rather than render interpolation. Build the capture directory under a
  per-process temporary sibling and atomically rename it only on success. The
  production validator CLI accepts an evidence
  directory plus Task 7 headless JSONL, verifies all scenario/repeat pairs, and
  atomically writes one canonical merged JSONL and summary JSON.

  Synthetic filesystem cases must also reject a missing/extra/duplicate tick
  frame, a filename/tick mismatch, blank frame, and a capture count different
  from the JSONL row count.

  Run and record the expected red failure before implementation:

  ```bash
  python -m unittest tests.python.test_smart_pickup_graphical_evidence -v
  ```

  Expected: failures for the missing validator/schema behavior.

- [ ] **Step 3: Write failing environment and source-contract tests**

  Extend the structural evidence test to require:

  - Smart Pickup auto-demo refuses the diagnostic pack, missing variables,
    unknown scenario/repeat, reused paths, and existing output destinations,
    including a pre-existing capture directory;
  - scenario initialization occurs only during reset before the first 25 Hz
    step, never as a live root/pose/joint write;
  - translated/yawed target, destination, and immutable blockers are registered
    before runtime/controller construction;
  - the scripted driver emits one F edge, then delegates approach/pick to
    `SmartPickupController` and Carry/place to the existing placement workflow;
  - JSONL is written once per simulation tick, not once per arbitrary render;
  - the red/green/cyan slot overlay reads diagnostics only; and
  - legacy interaction/placement auto-demo variable names and fixed-two
    placement behavior remain present and separate.

  Run and record the expected structural red failure:

  ```bash
  python -m unittest tests.python.test_playable_interaction_evidence -v
  ```

  Expected: failures for the missing fail-closed native wiring and tick capture
  contract.

- [ ] **Step 4: Implement scenario lookup and deterministic reset**

  Implement the raylib-free scenario table and parser. In the guarded native
  mode, initialize flat locomotion at the selected start before the first step,
  transform the baked target/destination in object-planar world space, and copy
  the selected blocker array into the same controller arrays consumed by
  locomotion and Smart Pickup. The repeat index is evidence metadata only; it
  must not perturb state, seed, slot order, or matcher inputs.

- [ ] **Step 5: Implement the shared-path native driver and overlay**

  Emit one interact edge after reset, then allow the production assist to own
  ordinary locomotion through pickup. On Carry, invoke the existing placement
  approach/preview/submission workflow to reach the scenario destination. Do
  not teleport, directly force phases, cache a preview candidate, or add a
  scenario-only match restriction.

  Wire the already-tested read-only overlay: object planar frame, all mapped
  slots/routes, red blocked, green eligible, cyan frozen, IDs/cost keys/reasons,
  request count, and runtime phase. Manual mode uses the same overlay.

- [ ] **Step 6: Implement atomic per-simulation-tick evidence**

  Serialize the Task 7 fields plus scenario ID/repeat, full data hashes,
  corrections, complete joint/world transforms, grasp/object/placement metrics,
  and overlay decision state after each authoritative 25 Hz simulation update.
  After rendering that state, capture exactly one tick-numbered PNG into the
  temporary capture directory. Capture a nonblank terminal screenshot after
  release (or stable failure for all-blocked), validate internally required
  terminal conditions, flush/close, then atomically rename the log, screenshot,
  and capture directory. Any timeout, window close, I/O error, validation
  failure, or unexpected state exits nonzero and leaves no final artifact.

- [ ] **Step 7: Run focused, structural, and safe regressions**

  Run:

  ```bash
  make build/tests/test_interaction_smart_pickup_scenarios && \
    build/tests/test_interaction_smart_pickup_scenarios
  python -m unittest \
    tests.python.test_smart_pickup_graphical_evidence \
    tests.python.test_playable_interaction_evidence \
    tests.python.test_playable_placement_evidence -v
  make test-interaction-safe
  ```

  Expected: scenario/parser/evidence boundary tests pass and every legacy
  interaction/placement gate remains green.

### Task 9: Run Isolated Native Walk-Pick-Carry-Place Acceptance

**Files:**
- Generate: `build/smart-pickup/graphical-acceptance.jsonl`
- Generate: `build/smart-pickup/graphical-acceptance-summary.json`
- Generate: `build/smart-pickup/evidence/` screenshots and short 25 Hz captures
- Generate: `build/smart-pickup/graphical-acceptance-25fps.mp4`
- Modify only if a reproduced defect requires a tested fix: files from Tasks
  1-8, with a new failing regression first

**Interfaces:**
- Consumes: the production `controller`, explicit full pack through
  `MM_INTERACTION_PACK`, the Task 8 deterministic scenario mode, an agent-owned
  isolated X display/window, and the Task 7 acceptance matrix.
- Produces: human-visible evidence that the same tested path walks, picks,
  carries, and places without the previously observed root/head/elbow/inactive-
  arm/object discontinuities.

- [ ] **Step 1: Build the production controller and establish isolation**

  Build with the repository's normal controller target. Before launching a
  window, generate the exact final headless comparison trace from the same
  binaries/data that the graphical runs will use:

  ```bash
  build/tests/test_live_flat_smart_pickup_oracle \
    resources/database.bin build/smart-pickup/full-pack \
    --jsonl build/smart-pickup/headless-final.jsonl
  ```

  Then start a new isolated
  X server/display owned by this task; record its PID, display number, and
  controller child PID. Never use or terminate `DISPLAY=:1`, never send global
  input, never use `pkill`, and never interact with a terrain-aware window.

  Launch each case/repeat only with the scoped Task 8 variables:

  ```bash
  MM_INTERACTION_PACK=build/smart-pickup/full-pack \
  MM_SMART_PICKUP_AUTODEMO=1 \
  MM_SMART_PICKUP_SCENARIO=${case_id} \
  MM_SMART_PICKUP_REPEAT=${repeat} \
  MM_SMART_PICKUP_LOG=build/smart-pickup/evidence/${case_id}-${repeat}.jsonl \
  MM_SMART_PICKUP_SCREENSHOT=build/smart-pickup/evidence/${case_id}-${repeat}.png \
  MM_SMART_PICKUP_CAPTURE_DIR=build/smart-pickup/evidence/${case_id}-${repeat}-frames \
  DISPLAY=${SMART_PICKUP_DISPLAY} ./controller
  ```

  Assert the startup diagnostics report schema 1, 25 Hz, 2,045 clips, and the
  exact full-pack hash before the deterministic driver begins.

- [ ] **Step 2: Verify the authored decision overlay without changing behavior**

  Confirm the Task 8 interaction debug overlay draws the target object planar
  frame, each mapped slot, and the direct segment from live root to slot. Its
  stable colors are red for rejected/blocked candidates, green for eligible
  unselected candidates, and cyan for the frozen selected slot. Confirm it
  displays slot ID, route length, heading key, object-origin distance
  (diagnostic only), current assist state, preview reason, request count, and
  runtime phase. Drawing remains read-only and absent from headless sources.

- [ ] **Step 3: Run the clear-start visual matrix at exactly 25 Hz**

  Run each of the three clear-start scenario IDs through the deterministic
  native mode and allow the tested shared assist/runtime path to reach Carry,
  then the existing place workflow to release on the destination. Run repeat
  indices `0` through `9`; the index changes evidence metadata only. Do not send
  XTEST/global input or alter the initialized pose after simulation begins.

  Capture one screenshot during approach, Reach, Contact, Carry, and after
  placement for each distinct selected slot. Capture short frame sequences
  around activation, pickup transition/contact, Carry entry, place transition,
  and release at one image per simulation tick; do not resample to 60 Hz or
  render back at 6 Hz. Assemble the representative sequences into
  `graphical-acceptance-25fps.mp4` at exactly 25 fps with no interpolated or
  duplicated timing frames.

- [ ] **Step 4: Run transformed-target and blocker visual cases**

  Run all ten repeats of the translated target, yawed target, and nominal-route
  blocker scenario IDs. Visually and numerically confirm the cyan slot follows the
  object's planar frame, the blocked slot remains red, and the alternate green
  slot is selected without passing through the table/blocker. Run the all-
  blocked case and confirm no movement, preview, or request begins.

- [ ] **Step 5: Validate graphical evidence against headless invariants**

  Parse the graphical JSONL and require:

  - exactly 25 Hz timestamps and one state row per simulation tick;
  - one request, attachment, and release per successful run;
  - the same selected slot IDs and terminal outcomes as the Task 7 cases;
  - no root correction above `0.25 m`, hand correction above `0.12 m`, or yaw
    correction above `25 degrees`;
  - continuity maxima no greater than the frozen Task 7 thresholds; and
  - no head/neck detachment, elbow flip, inactive-arm discontinuity, object
    teleport, table penetration, or controller collision.

  Run:

  ```bash
  python -m resources.validate_smart_pickup_graphical_evidence \
    --input-dir build/smart-pickup/evidence \
    --headless build/smart-pickup/headless-final.jsonl \
    --merged-jsonl build/smart-pickup/graphical-acceptance.jsonl \
    --output build/smart-pickup/graphical-acceptance-summary.json
  ```

  Generate the summary from this tested parser, not from manual judgment alone.
  Screenshots/video are supporting evidence; per-tick JSONL is the acceptance
  authority. Verify the MP4 has a declared `25/1` frame rate, finite positive
  duration/frame count, and nonblank sampled frames.

- [ ] **Step 6: Debug any reproduced defect systematically**

  If a case fails, preserve the first failing trace, identify the earliest bad
  tick and owning component, add the smallest failing unit/headless regression,
  and only then patch production code. Re-run that focused test, Task 7 oracle,
  and the failing graphical case. Do not tune thresholds, bake a new slot, or
  add smoothing merely to hide the symptom.

- [ ] **Step 7: Run final verification and shut down only owned processes**

  Run:

  ```bash
  make test-interaction-safe
  build/tests/test_live_flat_smart_pickup_oracle \
    resources/database.bin build/smart-pickup/full-pack \
    --jsonl build/smart-pickup/headless-final-rerun.jsonl
  cmp build/smart-pickup/headless-final.jsonl \
    build/smart-pickup/headless-final-rerun.jsonl
  SMART_PICKUP_FULL_PACK=build/smart-pickup/full-pack \
  SMART_PICKUP_PREVIEW_PROBE=./interaction_smart_pickup_preview_probe \
  SMART_PICKUP_SCENE_PROBE=./interaction_smart_pickup_scene_probe \
    python -m unittest \
      tests.python.test_smart_pickup_full_pack_gate \
      tests.python.test_smart_pickup_graphical_evidence \
      tests.python.test_playable_interaction_evidence \
      tests.python.test_playable_placement_evidence -v
  ```

  If any production code or data changed after Step 1, discard the graphical
  artifacts and restart Task 9 from Step 1; do not compare graphical results to
  a stale headless trace.

  Stop only the recorded controller child and isolated X-server PIDs. Verify
  they exited and that `DISPLAY=:1` and terrain-aware processes were untouched.
  The checkpoint is complete only when the safe suite, headless matrix, and
  isolated graphical summary all pass.
