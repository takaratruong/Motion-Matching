# Truthful Table/Rack Repetition Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task-by-task. Every task
> uses a fresh implementation subagent followed by spec-compliance and code-
> quality review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a native-25-Hz flat-ground scene where one object repeatedly
moves between a table and varied rack slots through real pickup, attachment,
Carry, truthful height-compatible placement, release, and re-pick.

**Architecture:** Preserve `InteractionRuntime` as the sole authority for
reservation, attachment, Carry, placement, release, and target generation. Add
an explicitly typed precomputed-reversed-pickup placement tier, construct it
from full-pack pickup provenance, expose table/rack destinations through a
semantic catalog, and drive a frozen route only after exact runtime state
transitions. Headless gates prove the entire lifecycle before the graphical
controller exposes it.

**Tech Stack:** C++17, existing G1 interaction database/matcher/IK/runtime,
Python 3.10 with NumPy, canonical JSON/binary evidence, GNU Make, Raylib, exact
25 Hz (`0.04 s`) updates.

## Global Constraints

- Work only in branch/worktree `g1-tabletop-placement` at
  `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement`.
- Do not merge, modify, signal, focus, or stop terrain work or its process.
- Simulation, controller updates, evidence rows, and video are exactly 25 Hz;
  do not add 60 Hz interpolation or a lower-rate playback path.
- The processed full pack is
  `build/smart-pickup/full-pack`: 2,045 clips, 511,250 frames, 633 packed object
  IDs, 25/1 fps. Do not parse the raw 50 GB pickup PKLs.
- `InteractionRuntime` and `TargetRegistry` remain the only attachment,
  ownership, placement, release, and generation authorities.
- A pickup succeeds only after actual `Carry`, `Held`, `attached=true`, exact
  request ownership, exactly one unattached-to-attached edge, and a subsequent
  ordinary Carry tick preserving the grasp.
- Only genuine authored put-down data may use `RecordedPlace`.
  Reversed pickup data uses `PrecomputedReversedPickup`; the immediately
  preceding pickup uses `ReversedPickup`.
- Placement priority is exactly genuine `RecordedPlace`, then compatible
  `PrecomputedReversedPickup`, then same-pickup `ReversedPickup`.
- Applied vertical correction never exceeds `0.12 m`.
- The positive rack set has three data-backed support heights whose lowest-to-
  highest span is strictly greater than `0.24002 m`; all three select distinct
  source IDs and each serves as a placement destination and later pickup source.
- The five surface-local offsets are fixed as left `(-0.30, 0.00)`, centre
  `(0.00, 0.00)`, right `(0.30, 0.00)`, shallow `(0.00, -0.10)`, and deep
  `(0.00, 0.10)` metres. The declared positive matrix is their cross product
  with low/middle/high: 15 rows.
- Every positive row performs two table-to-rack-to-table round trips in one
  unchanged runtime/controller/registry: four attachments, four releases, four
  re-picks, stable object ID, and final generation `g+4`.
- Rejected pickup leaves `Free`, unattached, owner zero, and generation
  unchanged, then a corrected request completes a real pickup and placement in
  that same runtime/registry. Rejected placement retains `Held/attached`, owner,
  and generation, then recovers through a valid placement in that same runtime.
- Stacked rack tiers are selected by exact semantic surface/affordance identity;
  never call `resolve_single_surface` for the routed destination.
- Temporary videos and reports live outside the repository under
  `/home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-a/`.
- Do not access, stat, hash, execute, modify, stage, delete, or enumerate the
  protected repository-root untracked `interaction_query_probe`.
- Inspect status only with `git status --short --untracked-files=no`. Stage only
  explicit paths; never use `git add .` or `git add -A`.
- At plan start, `Makefile` and
  `tests/python/test_smart_pickup_full_pack_gate.py` contain an unfinished
  agent-owned attachment-oracle checkpoint. Preserve and reconcile those edits;
  do not overwrite or stage them blindly.
- Every task follows RED -> GREEN -> focused regression -> independent review ->
  explicit-path commit -> `git push checkpoint HEAD:g1-tabletop-placement`.
- Plan B may not begin until Task 7's executable
  `gate-smart-pickup-plan-a` passes both Task 6's normal/release headless matrix
  and the environment-driven graphical JSONL/PNG gate.

## File Map

- `interaction_pickup_provenance.{h,cpp}`: leaf raw/global pickup provenance,
  checked local/global conversion, immutable certified identities, and exact
  registry lookup shared by runtime and scenarios without a header cycle.
- `interaction_smart_pickup_scenarios.{h,cpp}`: headless Smart Pickup lifecycle
  cases and actual attachment witnesses.
- `interaction_place.{h,cpp}`: truthful placement source kinds, selection,
  fingerprinting, and playback.
- `interaction_smart_pickup_place_library.{h,cpp}`: verify and own the exact
  pack, hydrate the height-indexed precomputed reversal library, and construct
  the immutable certified pickup registry as one lifetime-stable bundle.
- `resources/g1_interaction_builder/precomputed_place.py`: deterministic source
  extraction and canonical artifact writing.
- `resources/build_g1_precomputed_place_library.py`: command-line builder for
  the reviewed full pack.
- `interaction_table_rack_scene.{h,cpp}`: semantic table/rack catalog, surfaces,
  fixed 3-by-5 matrix, and exact support identity.
- `interaction_table_rack_route.{h,cpp}`: route latch/advance/reset state machine.
- `interaction_table_rack_probe.cpp`: repeated positive/negative full-chain
  evidence in one unchanged runtime per row.
- `controller.cpp`: exact highlighted target, keyboard integration, rack render,
  HUD, and native-25-Hz evidence.

---

### Task 1: Replace the preview-only oracle with a required attachment witness

**Files:**

- Create: `interaction_pickup_provenance.h`
- Create: `interaction_pickup_provenance.cpp`
- Reconcile/Create: `interaction_smart_pickup_scenarios.h`
- Reconcile/Create: `interaction_smart_pickup_scenarios.cpp`
- Reconcile/Create: `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`
- Modify: `interaction_runtime.h`
- Modify: `interaction_runtime.cpp`
- Modify: `tests/cpp/test_interaction_runtime.cpp`
- Modify: `tests/python/test_smart_pickup_full_pack_gate.py`
- Modify: `Makefile`

**Interfaces:**

- Consumes: `SmartPickupController`, `InteractionRuntime`, `TargetRegistry`, one
  certified full-pack target, and exact `0.04F` observations.
- Produces: `SmartPickupLifecycleWitness` and JSONL `case_result` rows that prove
  request, attachment, ownership, Carry, post-Carry grasp preservation, and
  the actual runtime-selected clip/event/grasp provenance. `MatchCandidate`
  currently has no `source_id`; this task does not invent one there. A certified
  provenance registry performs the exact raw-provenance-to-source join.

- [ ] **Step 1: Preserve and inspect the unfinished checkpoint**

  ```bash
  git status --short --untracked-files=no
  git diff -- Makefile tests/python/test_smart_pickup_full_pack_gate.py
  git diff --cached --check
  ```

  Expected: only the known tracked files are dirty and no unrelated path is
  staged. Do not enumerate untracked paths.

- [ ] **Step 2: Write/complete the failing C++ lifecycle assertions**

  Make `interaction_pickup_provenance.h` a leaf interface that includes only
  `interaction_target.h` plus its own standard-library dependencies. It must
  not include `interaction_runtime.h` or
  `interaction_smart_pickup_scenarios.h`; both of those headers include this
  leaf directly. Put these public declarations in the leaf:

  ```cpp
  struct PickupSourceProvenance {
      int32_t clip_ordinal = -1;
      int32_t range_start = -1;
      int32_t range_stop = -1;
      int32_t entry_global_frame = -1;
      int32_t contact_global_frame = -1;
      int32_t lift_global_frame = -1;
      int32_t hold_global_frame = -1;
      Hand active_hand = Hand::Right;
      uint64_t object_profile_id = 0U;
      ObjectLocalBounds object_bounds{};
      Transform hand_in_object{};
      float source_support_height_m = 0.0F;
  };

  struct CertifiedPickupSourceIdentity {
      PickupSourceProvenance provenance{};
      std::string sequence_id{};
      std::string object_id{};
      int32_t reverse_start_global_frame = -1;
      std::string join_key_sha256{};
      uint64_t source_id = 0U;
  };

  struct CertifiedPickupSourceLocalRow {
      int32_t clip_ordinal = -1;
      int32_t range_start = -1;
      int32_t range_stop = -1;
      int32_t entry_local_frame = -1;
      int32_t contact_local_frame = -1;
      int32_t lift_local_frame = -1;
      int32_t hold_local_frame = -1;
      int32_t reverse_start_local_frame = -1;
      Hand active_hand = Hand::Right;
      uint64_t object_profile_id = 0U;
      ObjectLocalBounds object_bounds{};
      Transform hand_in_object{};
      float source_support_height_m = 0.0F;
      std::string sequence_id{};
      std::string object_id{};
      std::string join_key_sha256{};
      uint64_t source_id = 0U;
  };

  int32_t checked_pickup_global_to_local_frame(
      int32_t global_frame,
      int32_t range_start,
      int32_t range_stop);
  int32_t checked_pickup_local_to_global_frame(
      int32_t local_frame,
      int32_t range_start,
      int32_t range_stop);
  CertifiedPickupSourceIdentity make_certified_pickup_source_identity(
      const CertifiedPickupSourceLocalRow& row);

  class CertifiedPickupSourceRegistry {
  public:
      // Rows are already-global identities returned by the factory above.
      explicit CertifiedPickupSourceRegistry(
          std::vector<CertifiedPickupSourceIdentity> rows);
      std::optional<CertifiedPickupSourceIdentity> resolve_exact(
          const PickupSourceProvenance& actual) const;
  };
  ```

  The leaf includes `<cstdint>`, `<optional>`, `<stdexcept>`, `<string>`, and
  `<vector>` itself, so every declaration above is callable without relying on
  transitive includes.

  `interaction_smart_pickup_scenarios.h` includes the leaf and separately
  declares:

  ```cpp

  struct SmartPickupLifecycleWitness {
      uint64_t request_id = 0U;
      TargetHandle target_before{};
      TargetHandle target_after{};
      uint64_t owner_request = 0U;
      RuntimeState terminal_state = RuntimeState::Disabled;
      ObjectState terminal_object_state = ObjectState::Free;
      uint32_t request_count = 0U;
      uint32_t attachment_edges = 0U;
      uint32_t release_edges = 0U;
      bool attached = false;
      bool post_carry_tick_observed = false;
      bool grasp_preserved = false;
      float carry_root_displacement_m = 0.0F;
      float carry_object_displacement_m = 0.0F;
      CertifiedPickupSourceIdentity actual_source{};
  };
  ```

  Append
  `const CertifiedPickupSourceRegistry* pickup_source_registry = nullptr` to
  both existing public `InteractionRuntime` constructors (after
  `RuntimeConfig`) and store that non-owning pointer for the runtime lifetime;
  preserve existing call sites through the default. The source certifier and
  table/rack fixture pass a registry explicitly. An ordinary caller may omit
  it, in which case raw provenance remains available but join key/source ID are
  empty/zero and cannot satisfy a certified gate.

  Add `CertifiedPickupSourceIdentity pickup_source{}` to `RuntimeDiagnostics` and the
  same field to `PickEntryPreview`. Derive its raw fields from the actual
  accepted `MatchCandidate` plus `Database::range_starts/range_stops`,
  `active_hands`, `object_dimensions`, `grasp_positions_object`, and
  `grasp_rotations_object`; derive `source_support_height_m` from the selected
  clip's table transform/size support plane using the same float32 operations as
  the structural extractor. Never copy a declared interaction slot. Convert
  object dimensions to the exact source bounds representation used by
  `interaction_place`. If a `CertifiedPickupSourceRegistry` was supplied to the
  runtime, resolve all raw fields exactly and populate sequence/object/reverse
  metadata, join key, and source ID from that one resolved registry row;
  otherwise leave every non-`provenance` identity field at its empty/default
  value. Both preview and committed preflight must derive byte-identical raw
  provenance.

  All fields named `*_global_frame` use the database's concatenated frame
  domain and must satisfy `range_start <= frame < range_stop`. The one and only
  conversion used by C++ and Python is:

  ```text
  local_frame  = global_frame - range_start
  global_frame = checked_add(range_start, local_frame)
  clip_length  = range_stop - range_start
  0 <= local_frame < clip_length
  ```

  The public checked functions implement that text literally. They require
  `0 <= range_start < range_stop <= INT32_MAX`;
  `checked_pickup_global_to_local_frame` throws `std::invalid_argument` for an
  invalid range and `std::out_of_range` unless
  `range_start <= global_frame < range_stop`;
  `checked_pickup_local_to_global_frame` throws `std::invalid_argument` for an
  invalid range, `std::out_of_range` unless
  `0 <= local_frame < range_stop-range_start`, and `std::overflow_error` if its
  checked addition cannot fit `int32_t`.

  There is no implicit local conversion in the registry constructor.
  `make_certified_pickup_source_identity` is the sole local-row factory: it
  validates
  `entry < contact < lift < hold <= reverse_start < clip_length`, calls the
  public local-to-global function for all five fields, copies every other field
  exactly, and returns an already-global `CertifiedPickupSourceIdentity`.
  `CertifiedPickupSourceRegistry(std::vector<CertifiedPickupSourceIdentity>)`
  accepts only those already-global values, independently validates all five
  against the stored range plus metadata/join/source invariants, and never
  adds `range_start` itself. Task 1 synthetic rows and every Task 3 generated
  local row must call the factory before constructing the registry. Runtime
  matching supplies the first four global values; after their exact raw lookup,
  the registry supplies the fifth global reverse-start value. Serialization
  back to Task 3 calls the public global-to-local function for all five. Tests
  call both public functions and the factory from translation units other than
  `interaction_pickup_provenance.cpp`; they cover local `0`, local
  `clip_length - 1`, every event, nonzero `range_start`, exact round trips, and
  reject negative local/global values, `global == range_stop`, local
  `== clip_length`, reversed/empty ranges, and checked-add overflow.

  In `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`, reject the old
  Locomotion-only result and require:

  ```cpp
  TEST_CHECK(witness.request_count == 1U);
  TEST_CHECK(witness.attachment_edges == 1U);
  TEST_CHECK(witness.release_edges == 0U);
  TEST_CHECK(witness.terminal_state == RuntimeState::Carry);
  TEST_CHECK(witness.terminal_object_state == ObjectState::Held);
  TEST_CHECK(witness.attached);
  TEST_CHECK(witness.owner_request == witness.request_id);
  TEST_CHECK(witness.target_after == witness.target_before);
  TEST_CHECK(witness.post_carry_tick_observed);
  TEST_CHECK(witness.grasp_preserved);
  TEST_CHECK(witness.carry_root_displacement_m > 0.0F);
  TEST_CHECK(witness.carry_object_displacement_m > 0.0F);
  TEST_CHECK(witness.actual_source.provenance.clip_ordinal >= 0);
  TEST_CHECK(witness.actual_source.provenance.range_start <=
             witness.actual_source.provenance.entry_global_frame);
  TEST_CHECK(witness.actual_source.provenance.entry_global_frame <
             witness.actual_source.provenance.contact_global_frame);
  TEST_CHECK(witness.actual_source.provenance.contact_global_frame <
             witness.actual_source.provenance.lift_global_frame);
  TEST_CHECK(witness.actual_source.provenance.lift_global_frame <
             witness.actual_source.provenance.hold_global_frame);
  TEST_CHECK(witness.actual_source.provenance.hold_global_frame <
             witness.actual_source.provenance.range_stop);
  TEST_CHECK(std::isfinite(
      witness.actual_source.provenance.source_support_height_m));
  TEST_CHECK(witness.actual_source.sequence_id.empty());
  TEST_CHECK(witness.actual_source.object_id.empty());
  TEST_CHECK(witness.actual_source.reverse_start_global_frame == -1);
  TEST_CHECK(witness.actual_source.source_id == 0U);
  TEST_CHECK(witness.actual_source.join_key_sha256.empty());
  ```

  That full-pack lifecycle fixture deliberately constructs the runtime without
  a certified registry: Task 1 proves actual attachment and raw matcher
  provenance, not a certification artifact that does not exist until Task 3.
  In a separate synthetic unit fixture, construct a
  `CertifiedPickupSourceLocalRow` from known local events, convert it through
  `make_certified_pickup_source_identity`, pass the resulting global row to the
  registry before runtime construction, and require the
  same pickup to resolve a nonzero source ID/canonical join key plus exact
  sequence/object/global reverse-start metadata. Mutating any raw field must
  leave the identity sentinel zero/empty while preserving the actual raw
  provenance.

- [ ] **Step 3: Run RED**

  ```bash
  make build/tests/test_interaction_smart_pickup_scenarios
  build/tests/test_interaction_smart_pickup_scenarios
  ```

  Expected: FAIL because the current oracle either lacks the witness fields or
  terminates in Locomotion without actual attachment.

- [ ] **Step 4: Implement the actual runtime witness**

  Track the attachment edge around every runtime update and stop only after a
  normal Carry update:

  ```cpp
  Transform pose_hand_world(const Pose& pose, Hand hand) {
      const WorldPose world = world_pose(pose);
      const size_t bone = hand == Hand::Left
          ? static_cast<size_t>(g1_skeleton::LeftWrist)
          : static_cast<size_t>(g1_skeleton::RightWrist);
      return {world.positions[bone], world.rotations[bone]};
  }

  const bool was_attached = output.diagnostics.attached;
  output = runtime.update(input);
  if (!was_attached && output.diagnostics.attached) {
      ++witness.attachment_edges;
  }
  if (was_attached && !output.diagnostics.attached) {
      ++witness.release_edges;
  }

  if (output.diagnostics.state == RuntimeState::Carry &&
      output.diagnostics.attached) {
      RuntimeInput carry_tick{};
      carry_tick.dt = 0.04F;
      carry_tick.locomotion = displaced_flat_snapshot(
          output.pose, vec3(0.02F, 0.0F, 0.0F));
      const RuntimeOutput after = runtime.update(carry_tick);
      const Transform expected_hand_after = compose(
          after.object_world, affordance.hand_in_object);
      const Transform actual_hand_after = pose_hand_world(
          after.pose, affordance.hand);
      witness.post_carry_tick_observed = true;
      witness.grasp_preserved = transform_near(
          actual_hand_after, expected_hand_after, 0.00002F, 0.00002F);
      output = after;
  }
  ```

  Populate the target state and owner from `registry.find(target_handle)`; do not
  infer them from animation state. Copy `witness.actual_source` only from the
  runtime's frozen `pickup_source`; do not use `fixture.source_slot`, catalog
  provenance, or the requested slot as evidence. Define `displaced_flat_snapshot` in the
  scenario fixture by copying the published flat snapshot and advancing only its
  Simulation-root X position by `0.02F`; define `transform_near` using planar
  position and quaternion-angle errors with the two explicit `0.00002F` limits.

  Put registry implementation plus the public checked conversion/factory
  definitions in
  `interaction_pickup_provenance.cpp`. Wire the leaf exactly once into every
  runtime link path:

  ```make
  INTERACTION_PICKUP_PROVENANCE_SOURCES := interaction_pickup_provenance.cpp
  INTERACTION_PICKUP_PROVENANCE_HEADERS := interaction_pickup_provenance.h

  # Append the source to both independently composed production/runtime lists.
  INTERACTION_SOURCES += $(INTERACTION_PICKUP_PROVENANCE_SOURCES)
  INTERACTION_RUNTIME_SOURCES += $(INTERACTION_PICKUP_PROVENANCE_SOURCES)
  INTERACTION_RUNTIME_HEADERS += $(INTERACTION_PICKUP_PROVENANCE_HEADERS)
  INTERACTION_SMART_PICKUP_CONTROLLER_HEADERS += \
    $(INTERACTION_PICKUP_PROVENANCE_HEADERS)
  INTERACTION_SMART_PICKUP_PREVIEW_HEADERS += \
    $(INTERACTION_PICKUP_PROVENANCE_HEADERS)
  ```

  Add the leaf source/header explicitly to the scenario-test prerequisites and
  add `$(INTERACTION_RUNTIME_HEADERS)` to every runtime-test/probe prerequisite
  list that compiles `$(INTERACTION_RUNTIME_SOURCES)`. Also
  keep the compiler command supplied through
  `INTERACTION_SMART_PICKUP_CONTROLLER_LINK_SOURCES`, which already contains
  `INTERACTION_RUNTIME_SOURCES`; do not list the `.cpp` twice on one link line.
  Task 6 repeats the header dependency and explicitly lists the source while
  filtering its inherited copy, so its final link command also contains it
  exactly once.

- [ ] **Step 5: Make the Python gate require every lifecycle field**

  Add these keys to the oracle record contract and validate them exactly:

  ```python
  _ATTACHMENT_KEYS = {
      "attached",
      "attachment_edges",
      "carry_object_displacement_m",
      "carry_root_displacement_m",
      "grasp_preserved",
      "owner_request",
      "post_carry_tick_observed",
      "release_edges",
      "target_generation_after",
      "target_generation_before",
      "target_state",
      "source_id",
      "source_join_key_sha256",
      "source_sequence_id",
      "source_object_id",
      "source_clip_ordinal",
      "source_range_start",
      "source_range_stop",
      "source_entry_global_frame",
      "source_contact_global_frame",
      "source_lift_global_frame",
      "source_hold_global_frame",
      "source_reverse_start_global_frame",
      "source_active_hand",
      "source_object_profile_id",
      "source_object_bounds_f32_hex",
      "source_hand_in_object_f32_hex",
      "source_support_height_f32_hex",
  }

  _require(record["terminal_state"] == "Carry", "pickup must end in Carry")
  _require(record["target_state"] == "Held", "pickup target must be Held")
  _require(record["attached"] is True, "pickup must be attached")
  _require(record["attachment_edges"] == 1, "pickup needs one attach edge")
  _require(record["release_edges"] == 0, "pickup must not release")
  _require(record["owner_request"] == record["request_id"], "owner differs")
  _require(record["target_generation_after"] ==
           record["target_generation_before"], "pickup changed generation")
  _require(record["post_carry_tick_observed"] is True, "missing Carry tick")
  _require(record["grasp_preserved"] is True, "Carry lost the grasp")
  _require(record["source_id"] == 0, "Task 1 must not fake certification")
  _require(record["source_join_key_sha256"] == "",
           "Task 1 must leave the certification join empty")
  _require(record["source_sequence_id"] == "", "unexpected sequence identity")
  _require(record["source_object_id"] == "", "unexpected object identity")
  _require(record["source_reverse_start_global_frame"] == -1,
           "unexpected reverse-start identity")
  _require(record["source_range_start"] <=
           record["source_entry_global_frame"] <
           record["source_contact_global_frame"] <
           record["source_lift_global_frame"] <
           record["source_hold_global_frame"] <
           record["source_range_stop"], "invalid global event provenance")
  ```

  Join-key/source-ID validity is tested with a synthetic
  `CertifiedPickupSourceRegistry`: independently mutate clip, range, every
  event, hand, one bounds component, one grasp translation component, and grasp
  quaternion sign/canonicalization or the exact float32 source-support height
  and require the exact lookup to fail. The registry constructor also rejects
  duplicate raw-provenance keys and invalid sequence/object/reverse-start
  metadata, a noncanonical join SHA, or a source ID unequal to the unsigned
  big-endian first eight bytes of that SHA. `sequence_id`, `object_id`, and
  `reverse_start_global_frame` are returned
  only by the registry row selected through that exact raw-provenance lookup;
  they are never copied from a requested fixture or catalog slot. The
  Task 1 Python full-pack oracle accepts only the zero/empty certified identity
  sentinel and validates raw global provenance. The synthetic C++ registry
  tests alone prove exact resolution here; Task 3 creates the first production
  frozen registry using the identical key and frame-conversion algorithms.

- [ ] **Step 6: Run GREEN and focused regression**

  ```bash
  make build/tests/test_interaction_smart_pickup_scenarios
  build/tests/test_interaction_smart_pickup_scenarios
  make build/tests/test_interaction_runtime \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  build/tests/test_interaction_runtime
  SMART_PICKUP_FULL_PACK="$PWD/build/smart-pickup/full-pack" \
    SMART_PICKUP_HEADLESS_ORACLE="$PWD/build/tests/test_interaction_smart_pickup_scenarios" \
    python -m unittest \
      tests.python.test_smart_pickup_full_pack_gate.SmartPickupFullPackGateTests \
      -v
  ```

  Expected: all pass, and every positive oracle row ends in actual Carry.

- [ ] **Step 7: Review, commit, and push only this checkpoint**

  ```bash
  git add Makefile \
    interaction_pickup_provenance.h interaction_pickup_provenance.cpp \
    interaction_smart_pickup_scenarios.h \
    interaction_smart_pickup_scenarios.cpp \
    interaction_runtime.h interaction_runtime.cpp \
    tests/cpp/test_interaction_runtime.cpp \
    tests/cpp/test_interaction_smart_pickup_scenarios.cpp \
    tests/python/test_smart_pickup_full_pack_gate.py
  git diff --cached --check
  git commit -m "test: require actual smart pickup attachment"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 2: Add truthful precomputed-reversal placement semantics

**Files:**

- Modify: `interaction_place.h`
- Modify: `interaction_place.cpp`
- Modify: `interaction_place_controller.h`
- Modify: `interaction_place_controller.cpp`
- Modify: `interaction_runtime.h`
- Modify: `interaction_runtime.cpp`
- Modify: `interaction_debug_draw.h`
- Test: `tests/cpp/test_interaction_place.cpp`
- Test: `tests/cpp/test_interaction_place_controller.cpp`
- Test: `tests/cpp/test_interaction_runtime.cpp`

**Interfaces:**

- Consumes: current `RecordedPlaceClip`, pickup `Database`, held grasp, target
  bounds, destination surface, IK/collision gates.
- Produces: appended `PlaceMotionMode::PrecomputedReversedPickup`, explicit
  `PrecomputedReversedPickupClip`, priority-preserving selection, and source
  provenance in `PlaceCandidate`/runtime diagnostics.

- [ ] **Step 1: Write failing source-kind and priority tests**

  Add tests that construct all three tiers and assert:

  ```cpp
  TEST_CHECK(static_cast<uint8_t>(PlaceMotionMode::ReversedPickup) == 2U);
  TEST_CHECK(
      static_cast<uint8_t>(PlaceMotionMode::PrecomputedReversedPickup) == 3U);

  fixture.library.recorded = {make_recorded_clip(1001U)};
  fixture.library.precomputed_reversed = {
      make_precomputed_reverse(2001U, 0.72F)};
  TEST_CHECK(select_place_motion(fixture.input).candidate.mode ==
             PlaceMotionMode::RecordedPlace);

  fixture.library.recorded.clear();
  TEST_CHECK(select_place_motion(fixture.input).candidate.mode ==
             PlaceMotionMode::PrecomputedReversedPickup);

  fixture.library.precomputed_reversed.clear();
  TEST_CHECK(select_place_motion(fixture.input).candidate.mode ==
             PlaceMotionMode::ReversedPickup);
  ```

  Also cover duplicate/zero source IDs, wrong 25/1 rate, bad frame order,
  discontinuous phase/contact evidence, wrong hand/grasp/bounds, source support
  mismatch, correction exactly `0.12F` accepted and `0.120001F` rejected,
  collision failure, deterministic tie ordering, selection fingerprint changes,
  and revalidation in `PlaceController::begin`.

  In `tests/cpp/test_interaction_runtime.cpp`, add a real place preflight/replay
  test whose selected candidate uses intentionally distinct finite sentinel
  values and prove they survive the `PlaceController` boundary into
  `RuntimePlaceDiagnostics`:

  ```cpp
  TEST_CHECK(output.diagnostics.place.source_id == 2001U);
  TEST_CHECK(output.diagnostics.place.requested_support_height_m == 0.71875F);
  TEST_CHECK(output.diagnostics.place.source_support_height_m == 0.625F);
  TEST_CHECK(output.diagnostics.place.target_support_height_m == 0.71875F);
  TEST_CHECK(
      output.diagnostics.place.requested_vertical_correction_m == 0.09375F);
  TEST_CHECK(
      output.diagnostics.place.applied_vertical_correction_m == 0.09375F);
  ```

  Check the first five fields immediately after accepted Place preflight and
  again during `PlaceReplay`; check the applied field on the first replay step
  where the controller publishes the correction. Mutate each sentinel
  independently and prove runtime diagnostics change with it rather than being
  recomputed from a default or the immediately preceding pickup.

  Add separate RED assertions at all three public boundaries. For the same
  selected candidate, `preview_place_motion`,
  `InteractionRuntime::preview_place`, and accepted `PlacePreflight` must each
  expose identical values for the five immutable provenance fields
  `source_id`, `requested_support_height_m`, `source_support_height_m`,
  `target_support_height_m`, and `requested_vertical_correction_m`. Replay
  updates must retain those five unchanged while only
  `applied_vertical_correction_m` advances from zero to its measured value.

- [ ] **Step 2: Run RED**

  ```bash
  make build/tests/test_interaction_place \
       build/tests/test_interaction_place_controller \
       build/tests/test_interaction_runtime
  ```

  Expected: compile failure because the new enum/vector/type does not exist.

- [ ] **Step 3: Add the public payload without changing existing enum values**

  Append the new value and add this database-backed row:

  ```cpp
  enum class PlaceMotionMode : uint8_t {
      None = 0,
      RecordedPlace = 1,
      ReversedPickup = 2,
      PrecomputedReversedPickup = 3,
  };

  struct PrecomputedReversedPickupClip {
      uint64_t id = 0U;
      uint64_t object_profile_id = 0U;
      std::string sequence_id{};
      int32_t clip = -1;
      int32_t entry_frame = -1;
      int32_t contact_frame = -1;
      int32_t lift_frame = -1;
      int32_t hold_frame = -1;
      int32_t reverse_start_frame = -1;
      Hand hand = Hand::Right;
      Transform source_hand_in_object{};
      ObjectLocalBounds source_object_bounds{};
      PlacementSurface source_surface{};
      uint32_t source_affordance_id = 0U;
      float source_support_height_m = 0.0F;
      float source_grasp_height_above_support_m = 0.0F;
  };

  struct PlaceMotionLibrary {
      std::vector<RecordedPlaceClip> recorded;
      std::vector<PrecomputedReversedPickupClip> precomputed_reversed;
  };

  // Add these fields to the existing PlaceCandidate definition:
  float source_support_height_m = 0.0F;
  float requested_support_height_m = 0.0F;
  float target_support_height_m = 0.0F;
  float requested_vertical_correction_m = 0.0F;
  ```

  Add `<string>` directly to `interaction_place.h`.

  Define the height meanings once and use them in every tier:

  - `requested_support_height_m` is
    `PlaceMatchInput::surface.surface_world.position.y`, frozen at selection;
  - `source_support_height_m` is the source clip/provenance support plane Y;
  - `target_support_height_m` is the selected candidate's corrected release
    support plane Y and must equal the requested support height for an accepted
    candidate within the existing serialized comparison tolerance; and
  - `requested_vertical_correction_m` is
    `target_support_height_m - source_support_height_m`, including its sign.

  Add the same four immutable values plus `source_id` to `PlaceStep` in
  `interaction_place_controller.h`. Add `applied_vertical_correction_m` to
  `PlaceStep`; it is zero before the correction is applied and thereafter is
  the signed world-Y component actually applied by `PlaceController::update`.

- [ ] **Step 4: Implement validation, selection, hashing, and playback**

  Factor the current reverse construction so it accepts either the immediately
  preceding pickup or a validated precomputed row:

  ```cpp
  struct ReverseSource {
      PlaceMotionMode mode = PlaceMotionMode::ReversedPickup;
      uint64_t source_id = 0U;
      int32_t clip = -1;
      int32_t entry_frame = -1;
      int32_t contact_frame = -1;
      int32_t lift_frame = -1;
      int32_t hold_frame = -1;
      int32_t reverse_start_frame = -1;
  };

  std::optional<PlaceCandidate> reverse_candidate(
      const PlaceMatchInput& input,
      const ValidatedInput& validated,
      const ReverseSource& source,
      SelectionFailures& failures);
  ```

  `select_place_motion` must remain visibly ordered:

  ```cpp
  if (auto recorded = select_recorded_tier(input, validated, failures)) {
      return accept_with_fingerprint(input, *recorded);
  }
  if (auto precomputed =
          select_precomputed_reverse_tier(input, validated, failures)) {
      return accept_with_fingerprint(input, *precomputed);
  }
  if (auto immediate = select_reverse_tier(
          input, validated, certified_prefix_stop, failures)) {
      return accept_with_fingerprint(
          input, *immediate, certified_prefix_stop);
  }
  return reject(failures.strongest());
  ```

  Hash every precomputed provenance field in library order. Database-backed
  interpolation is used for both reverse modes:

  ```cpp
  const bool database_backed =
      candidate.mode == PlaceMotionMode::ReversedPickup ||
      candidate.mode == PlaceMotionMode::PrecomputedReversedPickup;
  if (database_backed) {
      return interpolate_pose(
          pose_at_frame(*input.pickup_database, left),
          pose_at_frame(*input.pickup_database, right), alpha);
  }
  ```

  Add `source_id`, `requested_support_height_m`,
  `source_support_height_m`, `target_support_height_m`,
  `requested_vertical_correction_m`, and
  `applied_vertical_correction_m` to `RuntimePlaceDiagnostics`. In
  `InteractionRuntime::update_place_diagnostics`, copy them from the frozen
  candidate/`PlaceStep`; do not infer them from the live surface after
  preflight. Compute the applied vertical component from the signed world-Y
  component of the accepted root/hand correction, not from total 3D correction
  magnitude. Include every new candidate field in `selection_fingerprint`,
  both `same_candidate` implementations, runtime exact-comparison helpers, and
  cancellation/recovery snapshots.

  Populate the five immutable fields inside every successful tier constructor
  before calculating `selection_id`. `preview_place_motion` returns that exact
  candidate; `InteractionRuntime::preview_place` must not rebuild or erase the
  fields. On accepted Place preflight, freeze them from
  `frozen_place_preview_.candidate` into `RuntimePlaceDiagnostics`, and in
  `update_place_diagnostics` preserve those frozen values rather than replacing
  them from a later live preview. Tests must cover genuine recorded,
  precomputed-reverse, and immediate-reverse paths.

- [ ] **Step 5: Run GREEN and release/fast-math regression**

  ```bash
  make build/tests/test_interaction_place \
       build/tests/test_interaction_place_controller \
       build/tests/test_interaction_runtime
  build/tests/test_interaction_place
  build/tests/test_interaction_place_controller
  build/tests/test_interaction_runtime
  make test-interaction-place-selection-fast-math
  build/tests/test_interaction_place_selection_fast_math --fast-math-canary
  ```

  Expected: all pass; current `ReversedPickup == 2` tests remain unchanged.

- [ ] **Step 6: Review, commit, and push**

  ```bash
  git add interaction_place.h interaction_place.cpp \
    interaction_place_controller.h interaction_place_controller.cpp \
    interaction_runtime.h interaction_runtime.cpp interaction_debug_draw.h \
    tests/cpp/test_interaction_place.cpp \
    tests/cpp/test_interaction_place_controller.cpp \
    tests/cpp/test_interaction_runtime.cpp
  git diff --cached --check
  git commit -m "feat: distinguish precomputed pickup reversals"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 3: Build and freeze the height-indexed precomputed library

**Files:**

- Create: `resources/g1_interaction_builder/precomputed_place.py`
- Create: `resources/build_g1_precomputed_place_library.py`
- Create: `tests/python/test_g1_precomputed_place_library.py`
- Create: `interaction_smart_pickup_place_library.h`
- Create: `interaction_smart_pickup_place_library.cpp`
- Create: `tests/cpp/test_interaction_smart_pickup_place_library.cpp`
- Create: `interaction_smart_pickup_place_source_probe.cpp`
- Modify: `interaction_smart_pickup_scene_probe.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_scene.cpp`
- Generate/Track: `resources/g1_smart_pickup_place_sources.json`
- Generate/Track: `interaction_smart_pickup_place_sources.inc`
- Modify: `Makefile`

**Interfaces:**

- Consumes: `read_artifact_set`, full-pack manifest/hash, phase/contact arrays,
  source table/object/grasp metadata, and actual `InteractionRuntime` pickup
  witnesses at candidate support heights.
- Produces: canonical schema-v1 structural and runtime-certification JSON,
  deterministic one-to-one structural/runtime joins, an exactly specified
  certified-height triple, generated constexpr placement and pickup-registry
  tables, and one verified, owning `CertifiedPickupSourceBundle` factory.

- [ ] **Step 1: Write failing deterministic extraction tests**

  The Python API has these exact dataclasses/signatures:

  ```python
  @dataclass(frozen=True)
  class PrecomputedPlaceSource:
      source_id: int
      sequence_id: str
      object_id: str
      object_profile_id: int
      clip_ordinal: int
      range_start: int
      range_stop: int
      entry_local_frame: int
      contact_local_frame: int
      lift_local_frame: int
      hold_local_frame: int
      reverse_start_local_frame: int
      active_hand: int
      support_height_m: float
      grasp_height_above_support_m: float
      bounds_center_object: tuple[float, float, float]
      bounds_half_extents_object: tuple[float, float, float]
      hand_in_object_position: tuple[float, float, float]
      hand_in_object_rotation: tuple[float, float, float, float]

  @dataclass(frozen=True)
  class PlayableObjectSpec:
      object_id: str
      object_profile_id: int
      bounds_center_object: tuple[float, float, float]
      bounds_half_extents_object: tuple[float, float, float]
      active_hand: int
      hand_in_object_position: tuple[float, float, float]
      hand_in_object_rotation: tuple[float, float, float, float]
      table_support_height_m: float

  @dataclass(frozen=True)
  class PackIdentity:
      database_sha256: str
      features_sha256: str
      manifest_sha256: str
      fps_numerator: int
      fps_denominator: int
      clip_count: int
      frame_count: int

  @dataclass(frozen=True)
  class RuntimePickupCertification:
      join_key: str
      source_id: int
      sequence_id: str
      clip_ordinal: int
      range_start: int
      range_stop: int
      entry_local_frame: int
      contact_local_frame: int
      lift_local_frame: int
      hold_local_frame: int
      reverse_start_local_frame: int
      active_hand: int
      support_height_f32_hex: str
      object_id: str
      object_profile_id: int
      bounds_center_object_f32_hex: tuple[str, str, str]
      bounds_half_extents_object_f32_hex: tuple[str, str, str]
      hand_in_object_position_f32_hex: tuple[str, str, str]
      hand_in_object_rotation_f32_hex: tuple[str, str, str, str]
      matched_sequence_id: str
      matched_clip_ordinal: int
      matched_range_start: int
      matched_range_stop: int
      matched_entry_local_frame: int
      matched_contact_local_frame: int
      matched_lift_local_frame: int
      matched_hold_local_frame: int
      request_id: int
      target_id: int
      target_generation_before: int
      target_generation_after: int
      owner_request: int
      terminal_state: str
      terminal_object_state: str
      attached: bool
      attachment_edges: int
      release_edges: int
      post_carry_tick_observed: bool
      grasp_preserved: bool
      carry_root_displacement_m_f32_hex: str
      carry_object_displacement_m_f32_hex: str
      rejection_reason: str

  @dataclass(frozen=True)
  class PrecomputedPlaceCertification:
      schema_version: int
      pack: PackIdentity
      playable_object: PlayableObjectSpec
      rows: tuple[RuntimePickupCertification, ...]

  from typing import Literal

  FrozenPlacementRole = Literal[
      "tier_low", "tier_middle", "tier_high", "table_return"]

  @dataclass(frozen=True)
  class FrozenPlacementSource:
      source: PrecomputedPlaceSource
      roles: tuple[FrozenPlacementRole, ...]

  @dataclass(frozen=True)
  class FrozenSmartPickupPlaceSources:
      schema_version: int
      pack: PackIdentity
      playable_object: PlayableObjectSpec
      placement_sources: tuple[FrozenPlacementSource, ...]
      certified_pickup_sources: tuple[PrecomputedPlaceSource, ...]

  extract_precomputed_place_candidates(
      pack: Path
  ) -> tuple[PrecomputedPlaceSource, ...]

  write_precomputed_place_sources(
      output_json: Path,
      output_include: Path,
      tier_sources: tuple[
          PrecomputedPlaceSource,
          PrecomputedPlaceSource,
          PrecomputedPlaceSource,
      ],
      table_return_source: PrecomputedPlaceSource,
      certified_pickup_sources: Sequence[PrecomputedPlaceSource],
      pack_identity: PackIdentity,
      playable_object: PlayableObjectSpec,
  ) -> None

  write_structural_place_sources(
      output: Path,
      pack_identity: PackIdentity,
      playable_object: PlayableObjectSpec,
      sources: Sequence[PrecomputedPlaceSource],
  ) -> None

  read_structural_place_sources(
      path: Path,
      expected_pack: PackIdentity,
  ) -> tuple[PlayableObjectSpec, tuple[PrecomputedPlaceSource, ...]]

  read_runtime_pickup_certification(
      path: Path,
      expected_pack: PackIdentity,
  ) -> PrecomputedPlaceCertification

  read_frozen_smart_pickup_place_sources(
      path: Path,
      expected_pack: PackIdentity,
  ) -> FrozenSmartPickupPlaceSources

  resolve_frozen_pickup_source(
      artifact: FrozenSmartPickupPlaceSources,
      *,
      source_id: int,
      join_key_sha256: str,
  ) -> PrecomputedPlaceSource

  resolve_frozen_placement_source(
      artifact: FrozenSmartPickupPlaceSources,
      *,
      source_id: int,
      required_role: FrozenPlacementRole,
  ) -> FrozenPlacementSource

  select_certified_height_triple(
      structural: Sequence[PrecomputedPlaceSource],
      certification: PrecomputedPlaceCertification,
  ) -> tuple[
      tuple[PrecomputedPlaceSource,
            PrecomputedPlaceSource,
            PrecomputedPlaceSource],
      PrecomputedPlaceSource,
  ]
  ```

  `PrecomputedPlaceSource` and both persisted Task 3 schemas use only local
  event fields. Define and use these helpers for entry, contact, lift, hold,
  and reverse-start without exception:

  ```python
  _INT32_MAX = (1 << 31) - 1

  def global_to_local(global_frame: int, start: int, stop: int) -> int:
      if not (0 <= start < stop <= _INT32_MAX):
          raise CertificationError("invalid global range")
      if not (start <= global_frame < stop):
          raise CertificationError("global frame outside range")
      return global_frame - start

  def local_to_global(local_frame: int, start: int, stop: int) -> int:
      if not (0 <= start < stop <= _INT32_MAX):
          raise CertificationError("invalid global range")
      clip_length = stop - start
      if not (0 <= local_frame < clip_length):
          raise CertificationError("local frame outside clip")
      global_frame = start + local_frame
      if global_frame > _INT32_MAX:
          raise CertificationError("global frame overflow")
      return global_frame
  ```

  C++ implements the identical checked operations in the provenance leaf.
  Tests round-trip all five events at nonzero starts and both legal boundaries,
  cross-check C++ registry identities against serialized Python locals, and
  reject each invalid range/bound/overflow case. No JSON key named merely
  `entry_frame`, `contact_frame`, `lift_frame`, `hold_frame`, or
  `reverse_start_frame` is permitted: Task 1 raw records say `*_global_frame`;
  Task 3 structural/certification records say `*_local_frame`.

  Tests require byte-identical repeat output, stable `(height, sequence_id)`
  order, exact 25 Hz, complete event ordering, five contact samples ending at
  `reverse_start`, finite metadata, unique nonzero IDs, and pack-hash mismatch
  rejection. They also require strict top-level/row key sets, deterministic
  join-key recomputation, rejection of missing/extra/duplicate certification
  rows, and permutation-invariant triple selection.

  Extend `interaction_smart_pickup_scene_probe --json` so its existing target
  record publishes the compiled target's `object_profile_id`, exact
  `ObjectLocalBounds`, active hand, grasp transform, and support-plane Y as
  float32 hex. `PlayableObjectSpec.object_id` is the manifest object ID joined
  through that target record's existing sequence ID; its deterministic profile
  ID is the compiled target's nonzero `InteractionTarget::object_profile_id`
  (currently `1U`), not a per-source remap. Structural extraction retains only
  rows with that exact manifest object ID, assigns that same deterministic
  profile ID, and preserves each row's exact database bounds, active hand, and
  grasp transform. It never overwrites source geometry with the playable
  object's geometry; Step 4 applies the real placement compatibility gates.

- [ ] **Step 2: Run Python RED**

  ```bash
  python -m unittest tests.python.test_g1_precomputed_place_library -v
  make build/tests/test_interaction_smart_pickup_place_library
  ```

  Expected: Python import failure for the new module and no successful C++
  library test build before the source schema/hydrator exists.

- [ ] **Step 3: Implement structural extraction and canonical output**

  Use the first phase occurrence and stable Hold window; never infer an event
  from nominal clip length:

  ```python
  reach = int(np.flatnonzero(phases == int(InteractionPhase.REACH))[0])
  contact = int(np.flatnonzero(phases == int(InteractionPhase.CONTACT))[0])
  lift = int(np.flatnonzero(phases == int(InteractionPhase.LIFT))[0])
  hold = int(np.flatnonzero(phases == int(InteractionPhase.HOLD))[0])
  reverse_start = hold
  while reverse_start + 4 < len(phases):
      window = slice(reverse_start, reverse_start + 5)
      if np.all(phases[window] == int(InteractionPhase.HOLD)) and \
              np.all(hand_contacts[window, active_hand] == 1):
          reverse_start += 4
          break
      reverse_start += 1
  else:
      raise CandidateRejection("stable_hold")
  ```

  Define `canonical_json_bytes(value)` as
  `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
  allow_nan=False).encode("ascii") + b"\n"`. All persisted floats use their
  exact IEEE-754 float32 hex string (`0x` plus eight lowercase digits); decimal
  values are display-only and are never join authority.

  The structural/runtime join payload has this exact key set:

  ```python
  def structural_runtime_join_payload(
      source: PrecomputedPlaceSource,
      pack: PackIdentity,
  ) -> dict[str, object]:
      return {
          "active_hand": source.active_hand,
          "clip_ordinal": source.clip_ordinal,
          "contact_local_frame": source.contact_local_frame,
          "database_sha256": pack.database_sha256,
          "entry_local_frame": source.entry_local_frame,
          "features_sha256": pack.features_sha256,
          "hold_local_frame": source.hold_local_frame,
          "lift_local_frame": source.lift_local_frame,
          "manifest_sha256": pack.manifest_sha256,
          "object_id": source.object_id,
          "object_profile_id": source.object_profile_id,
          "bounds_center_object_f32_hex": [
              f32_hex(value) for value in source.bounds_center_object],
          "bounds_half_extents_object_f32_hex": [
              f32_hex(value) for value in source.bounds_half_extents_object],
          "hand_in_object_position_f32_hex": [
              f32_hex(value) for value in source.hand_in_object_position],
          "hand_in_object_rotation_f32_hex": [
              f32_hex(value) for value in source.hand_in_object_rotation],
          "range_start": source.range_start,
          "range_stop": source.range_stop,
          "reverse_start_local_frame": source.reverse_start_local_frame,
          "sequence_id": source.sequence_id,
          "support_height_f32_hex": f32_hex(source.support_height_m),
      }

  def structural_runtime_join_key(
      source: PrecomputedPlaceSource,
      pack: PackIdentity,
  ) -> str:
      payload = canonical_json_bytes(
          structural_runtime_join_payload(source, pack))
      return hashlib.sha256(payload).hexdigest()
  ```

  Derive `source_id` as the unsigned big-endian integer represented by the
  first eight bytes of that SHA-256. Reject the astronomically unlikely zero
  result rather than remapping it. The full 64-character lowercase join key,
  not `source_id`, is the one-to-one join authority.

  The runtime-certification document uses this executable schema:

  ```json
  {
    "pack": {
      "clip_count": 2045,
      "database_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "features_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "fps_denominator": 1,
      "fps_numerator": 25,
      "frame_count": 511250,
      "manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    },
    "record_type": "precomputed_place_runtime_certification",
    "playable_object": {
      "active_hand": 1,
      "bounds_center_object_f32_hex": ["0x00000000", "0x00000000", "0x00000000"],
      "bounds_half_extents_object_f32_hex": ["0x3dcccccd", "0x3dcccccd", "0x3dcccccd"],
      "hand_in_object_position_f32_hex": ["0x00000000", "0x00000000", "0x00000000"],
      "hand_in_object_rotation_f32_hex": ["0x3f800000", "0x00000000", "0x00000000", "0x00000000"],
      "object_id": "beer_10",
      "object_profile_id": 1,
      "table_support_height_f32_hex": "0x3f000000"
    },
    "rows": [],
    "schema_version": 1
  }
  ```

  The parser requires the exact top-level key set
  `{pack,playable_object,record_type,rows,schema_version}`, the exact seven `pack` keys shown,
  and the exact `RuntimePickupCertification` fields declared in Step 1. It
  rejects unknown keys as well as missing keys, non-canonical hex strings,
  non-ASCII sequence IDs, any non-25/1 pack, and any pack identity mismatch.
  Serialize structural JSON and certification JSON with canonical JSON bytes,
  UTF-8/ASCII, compact separators, sorted keys, and exactly one trailing LF.

  `write_structural_place_sources` is the sole structural writer consumed by
  the C++ probe. Its executable mapping is exactly:

  ```python
  _STRUCTURAL_TOP_KEYS = frozenset({
      "pack", "playable_object", "record_type", "rows", "schema_version",
  })
  _STRUCTURAL_ROW_KEYS = frozenset({
      "active_hand", "bounds_center_object_f32_hex",
      "bounds_half_extents_object_f32_hex", "clip_ordinal",
      "contact_local_frame", "entry_local_frame",
      "grasp_height_above_support_f32_hex", "hand_in_object_position_f32_hex",
      "hand_in_object_rotation_f32_hex", "hold_local_frame", "join_key",
      "lift_local_frame", "object_id", "object_profile_id", "range_start",
      "range_stop", "reverse_start_local_frame", "sequence_id", "source_id",
      "support_height_f32_hex",
  })

  def structural_row_to_json(
      source: PrecomputedPlaceSource, pack: PackIdentity
  ) -> dict[str, object]:
      row = {
          "active_hand": source.active_hand,
          "bounds_center_object_f32_hex": [
              f32_hex(x) for x in source.bounds_center_object],
          "bounds_half_extents_object_f32_hex": [
              f32_hex(x) for x in source.bounds_half_extents_object],
          "clip_ordinal": source.clip_ordinal,
          "contact_local_frame": source.contact_local_frame,
          "entry_local_frame": source.entry_local_frame,
          "grasp_height_above_support_f32_hex":
              f32_hex(source.grasp_height_above_support_m),
          "hand_in_object_position_f32_hex": [
              f32_hex(x) for x in source.hand_in_object_position],
          "hand_in_object_rotation_f32_hex": [
              f32_hex(x) for x in source.hand_in_object_rotation],
          "hold_local_frame": source.hold_local_frame,
          "join_key": structural_runtime_join_key(source, pack),
          "lift_local_frame": source.lift_local_frame,
          "object_id": source.object_id,
          "object_profile_id": source.object_profile_id,
          "range_start": source.range_start,
          "range_stop": source.range_stop,
          "reverse_start_local_frame": source.reverse_start_local_frame,
          "sequence_id": source.sequence_id,
          "source_id": source.source_id,
          "support_height_f32_hex": f32_hex(source.support_height_m),
      }
      assert frozenset(row) == _STRUCTURAL_ROW_KEYS
      return row

  document = {
      "pack": pack_identity_to_json(pack_identity),
      "playable_object": playable_object_to_json(playable_object),
      "record_type": "precomputed_place_structural_sources",
      "rows": sorted(
          (structural_row_to_json(row, pack_identity) for row in sources),
          key=lambda row: row["join_key"],
      ),
      "schema_version": 1,
  }
  assert frozenset(document) == _STRUCTURAL_TOP_KEYS
  output.write_bytes(canonical_json_bytes(document))
  ```

  `pack_identity_to_json` emits exactly the seven `PackIdentity` fields shown;
  `playable_object_to_json` emits exactly the eight playable-object keys shown.
  Scalar IDs/frames/hands are JSON integers, IDs must be in their stated
  unsigned ranges, all named `*_f32_hex` values are strings matching
  `0x[0-9a-f]{8}`, vector lengths are exactly 3/3/3/4, and string IDs are
  nonempty ASCII. `read_structural_place_sources` rejects unknown/missing keys,
  noncanonical raw bytes, bad order, duplicate key/ID, any row whose manifest
  object/profile differ from the one playable object, and pack mismatch. It
  does not reject merely because bounds/grasp are outside compatibility
  tolerance; the selector must filter those rows. This is the exact structural
  file consumed by the C++ probe; the probe does not consume the final generated
  include or an informal Python object dump.

  `resources/g1_smart_pickup_place_sources.json` is the final frozen source
  authority consumed by Python gates. It is not the structural or runtime-
  certification intermediate. Its exact executable schema is:

  ```python
  _FROZEN_TOP_KEYS = frozenset({
      "certified_pickup_sources",
      "pack",
      "placement_sources",
      "playable_object",
      "record_type",
      "schema_version",
  })
  _FROZEN_SOURCE_ROW_KEYS = frozenset({
      "active_hand",
      "bounds_center_object_f32_hex",
      "bounds_half_extents_object_f32_hex",
      "clip_ordinal",
      "contact_local_frame",
      "entry_local_frame",
      "grasp_height_above_support_f32_hex",
      "hand_in_object_position_f32_hex",
      "hand_in_object_rotation_f32_hex",
      "hold_local_frame",
      "join_key",
      "lift_local_frame",
      "object_id",
      "object_profile_id",
      "range_start",
      "range_stop",
      "reverse_start_local_frame",
      "sequence_id",
      "source_id",
      "support_height_f32_hex",
  })
  _FROZEN_PLACEMENT_ROW_KEYS = (
      _FROZEN_SOURCE_ROW_KEYS | frozenset({"roles"}))
  _FROZEN_ROLE_ORDER = (
      "tier_low", "tier_middle", "tier_high", "table_return")
  ```

  Top-level values map exactly as follows: `schema_version` is JSON integer `1`;
  `record_type` is string `"g1_smart_pickup_place_sources"`; `pack` has exactly
  the seven `PackIdentity` keys/types already frozen above; `playable_object`
  has exactly the eight `PlayableObjectSpec` JSON keys/types already frozen
  above; and both source fields are JSON arrays. Every certified-pickup element
  has exactly `_FROZEN_SOURCE_ROW_KEYS`. Every placement element has exactly
  `_FROZEN_PLACEMENT_ROW_KEYS`; removing `roles` must leave a byte-for-byte equal
  canonical source-row object in `certified_pickup_sources`.

  Source scalar mappings are exact: `active_hand` is integer `0` or `1`;
  `clip_ordinal`, ranges, and five local frames are nonnegative JSON integers no
  greater than `INT32_MAX`; `object_profile_id` and `source_id` are nonzero
  uint64 JSON integers; `sequence_id` and `object_id` are nonempty ASCII strings;
  `join_key` is exactly 64 lowercase hexadecimal characters and its first eight
  raw bytes interpreted big-endian equal `source_id`; scalar float fields are
  canonical `0x[0-9a-f]{8}` float32 strings; bounds/position/rotation fields are
  JSON arrays of exactly 3/3/3/4 canonical float32 strings. Ranges satisfy
  `0 <= range_start < range_stop <= INT32_MAX`; all five locals are within the
  clip and satisfy `entry < contact < lift < hold <= reverse_start`; decoded
  floats/quaternion are finite and retain the existing normalization and
  compatibility requirements. Recompute every join key from the decoded source
  plus exact `pack` and reject any difference.

  `certified_pickup_sources` is every one-to-one joined, runtime-eligible,
  playable-compatible source, sorted strictly by ASCII `join_key`, with unique
  join keys and source IDs. `write_precomputed_place_sources` maps the three
  selected tier inputs to roles `tier_low`, `tier_middle`, and `tier_high`, and
  the selected table-return input to `table_return`. It coalesces equal source
  identities into one placement row whose `roles` array is the applicable
  subsequence of `_FROZEN_ROLE_ORDER`; therefore table return may alias a tier
  without duplicating a source row. Placement rows are sorted strictly by
  `join_key`, have unique join keys/source IDs, are an exact subset of certified
  rows, contain every role exactly once across the array, and map the three tier
  roles to three distinct source IDs. No other role string/order is accepted.

  `read_frozen_smart_pickup_place_sources` first requires the input bytes to be
  exactly `canonical_json_bytes(parsed_value)`, then enforces every key, type,
  range, ordering, role, subset, join, source-ID, playable-object, and expected-
  pack rule above before returning immutable dataclasses. It never imports the
  generated C++ include and never shells out to a C++ probe. The two lookup
  functions build no fallback: `resolve_frozen_pickup_source` requires one row
  where both the supplied nonzero source ID and full join key match that same
  certified row; `resolve_frozen_placement_source` requires one placement row
  with the exact source ID and requested role. Zero, multiple, ID-only,
  join-only, or wrong-role matches raise `CertificationError`. Unit tests mutate
  every top/row key and type, canonical bytes, role mapping/order, array order,
  source/join pair, subset relation, local bound, and float-vector length.
  `write_precomputed_place_sources` writes JSON/include temporaries, reopens the
  JSON through this public reader with the exact pack, verifies every requested
  tier/table/certified mapping against the returned dataclasses, and only then
  atomically replaces both final outputs; a failure publishes neither output.

- [ ] **Step 4: Add an actual-pick source certifier and freeze three heights**

  `interaction_smart_pickup_place_source_probe` accepts exactly:

  ```text
  interaction_smart_pickup_place_source_probe
      <full-pack-directory> <structural-sources.json> --json
  ```

  It validates the structural document and pack identity first, constructs one
  `InteractionTarget` profile from the document's single `playable_object`, and
  reuses that exact object profile, bounds, hand, and grasp for every trial;
  only the authored support plane/object pose changes to the structural row's
  exact float32 height. It must not construct unrelated per-row object targets.
  For every row it runs a real pickup through `InteractionRuntime` with a
  `CertifiedPickupSourceRegistry` built by mapping every structural row into an
  exact `CertifiedPickupSourceLocalRow`, calling
  `make_certified_pickup_source_identity`, and passing the returned already-
  global identities to the registry constructor. Construct that registry in final
  local storage before each trial runtime, pass its address to the runtime
  constructor, and keep it unmoved/alive through destruction of that runtime.
  The runtime observes one ordinary
  post-Carry tick, and emits exactly one `RuntimePickupCertification`, including
  failed rows with a non-`None` `rejection_reason`. Output rows are sorted by
  `join_key`; the probe never drops, retries, substitutes, or reorders a failed
  source. The emitted matched clip/range/events and bounds/hand/grasp/support
  values are copied from the runtime's frozen
  `CertifiedPickupSourceIdentity::provenance`; matched sequence/object/reverse-start,
  join key, and source ID are copied from the remaining fields of that same
  registry-resolved identity. None are copied from the currently iterated
  structural row. Before emitting, the probe converts actual global entry,
  contact, lift, hold, and resolved reverse-start through
  `global_to_local(frame, range_start, range_stop)`; both the ordinary and
  `matched_*` certification fields are local. The parser applies
  Python `local_to_global` to all five, proves exact equality with the registry
  identity, then performs the independent structural join and recomputes the
  key before acceptance.

  `read_runtime_pickup_certification` rejects the whole certification document
  for any of these structural failures:

  - a missing, extra, or duplicate join key/source ID;
  - a row whose copied provenance does not exactly recompute its join key;
  - a row whose matcher clip/range/events do not equal its structural source;
  - pack hash/count/rate mismatch, noncanonical JSON values, or unsorted rows;
  - target ID/generation mutation during pickup or owner/request mismatch; or
  - nonfinite or nonpositive Carry root/object displacement evidence.

  A joined row is eligible for the height triple only when all of these hold:

  ```python
  row.rejection_reason == "None"
  row.terminal_state == "Carry"
  row.terminal_object_state == "Held"
  row.attached is True
  row.attachment_edges == 1
  row.release_edges == 0
  row.owner_request == row.request_id
  row.target_generation_after == row.target_generation_before
  row.post_carry_tick_observed is True
  row.grasp_preserved is True
  f32_from_hex(row.carry_root_displacement_m_f32_hex) > 0.0
  f32_from_hex(row.carry_object_displacement_m_f32_hex) > 0.0
  ```

  Implement the exact triple selector as follows:

  ```python
  def runtime_is_eligible(row: RuntimePickupCertification) -> bool:
      root_displacement = f32_from_hex(
          row.carry_root_displacement_m_f32_hex)
      object_displacement = f32_from_hex(
          row.carry_object_displacement_m_f32_hex)
      return (
          row.rejection_reason == "None"
          and row.terminal_state == "Carry"
          and row.terminal_object_state == "Held"
          and row.attached is True
          and row.attachment_edges == 1
          and row.release_edges == 0
          and row.owner_request == row.request_id
          and row.target_generation_after == row.target_generation_before
          and row.post_carry_tick_observed is True
          and row.grasp_preserved is True
          and math.isfinite(root_displacement)
          and math.isfinite(object_displacement)
          and root_displacement > 0.0
          and object_displacement > 0.0
      )

  def join_one_to_one_or_raise(structural, certification):
      structural_by_key = {}
      for source in structural:
          key = structural_runtime_join_key(source, certification.pack)
          if key in structural_by_key or source.source_id == 0:
              raise CertificationError("duplicate structural key/source")
          structural_by_key[key] = source
      runtime_by_key = {}
      runtime_source_ids = set()
      for row in certification.rows:
          if row.join_key in runtime_by_key or row.source_id in runtime_source_ids:
              raise CertificationError("duplicate runtime key/source")
          runtime_by_key[row.join_key] = row
          runtime_source_ids.add(row.source_id)
      if runtime_by_key.keys() != structural_by_key.keys():
          raise CertificationError("missing or extra runtime certification")
      joined = []
      for key in sorted(structural_by_key):
          source = structural_by_key[key]
          runtime = runtime_by_key[key]
          expected_identity = (
              source.source_id, source.sequence_id, source.clip_ordinal,
              source.range_start, source.range_stop,
              source.entry_local_frame, source.contact_local_frame,
              source.lift_local_frame, source.hold_local_frame,
              source.reverse_start_local_frame, source.active_hand,
              source.object_id, source.object_profile_id,
              tuple(f32_hex(x) for x in source.bounds_center_object),
              tuple(f32_hex(x) for x in source.bounds_half_extents_object),
              tuple(f32_hex(x) for x in source.hand_in_object_position),
              tuple(f32_hex(x) for x in source.hand_in_object_rotation),
              f32_hex(source.support_height_m),
          )
          actual_identity = (
              runtime.source_id, runtime.sequence_id, runtime.clip_ordinal,
              runtime.range_start, runtime.range_stop,
              runtime.entry_local_frame, runtime.contact_local_frame,
              runtime.lift_local_frame, runtime.hold_local_frame,
              runtime.reverse_start_local_frame, runtime.active_hand,
              runtime.object_id, runtime.object_profile_id,
              runtime.bounds_center_object_f32_hex,
              runtime.bounds_half_extents_object_f32_hex,
              runtime.hand_in_object_position_f32_hex,
              runtime.hand_in_object_rotation_f32_hex,
              runtime.support_height_f32_hex,
          )
          if actual_identity != expected_identity:
              raise CertificationError("runtime provenance differs")
          matched_identity = (
              runtime.matched_sequence_id, runtime.matched_clip_ordinal,
              runtime.matched_range_start, runtime.matched_range_stop,
              runtime.matched_entry_local_frame,
              runtime.matched_contact_local_frame,
              runtime.matched_lift_local_frame,
              runtime.matched_hold_local_frame,
          )
          expected_match = (
              source.sequence_id, source.clip_ordinal,
              source.range_start, source.range_stop,
              source.entry_local_frame, source.contact_local_frame,
              source.lift_local_frame, source.hold_local_frame,
          )
          if matched_identity != expected_match:
              raise CertificationError("runtime matcher differs")
          joined.append((source, runtime))
      return tuple(joined)

  _BOUNDS_TOLERANCE_M = 0.001
  _GRASP_POSITION_TOLERANCE_M = 0.02
  _GRASP_ORIENTATION_TOLERANCE_RADIANS = 0.174532925
  _MAXIMUM_VERTICAL_CORRECTION_M = 0.12

  def _rotation_error_radians(left, right):
      # Mirror rotation_gate::measure: normalize in double, treat q/-q as the
      # same rotation, then measure the shortest arc.
      left_norm = math.sqrt(sum(float(x) * float(x) for x in left))
      right_norm = math.sqrt(sum(float(x) * float(x) for x in right))
      if not (left_norm > 1.0e-12 and right_norm > 1.0e-12):
          return math.inf
      dot = sum((float(a) / left_norm) * (float(b) / right_norm)
                for a, b in zip(left, right, strict=True))
      return 2.0 * math.acos(min(1.0, max(0.0, abs(dot))))

  def place_compatible_with_playable(source, playable):
      bounds_pairs = (
          tuple(zip(source.bounds_center_object,
                    playable.bounds_center_object, strict=True)),
          tuple(zip(source.bounds_half_extents_object,
                    playable.bounds_half_extents_object, strict=True)),
      )
      grasp_position_error = math.sqrt(sum(
          (float(a) - float(b)) ** 2
          for a, b in zip(source.hand_in_object_position,
                          playable.hand_in_object_position, strict=True)))
      return (
          source.object_id == playable.object_id
          and source.object_profile_id == playable.object_profile_id
          and source.active_hand == playable.active_hand
          and all(abs(float(a) - float(b)) <= _BOUNDS_TOLERANCE_M
                  for group in bounds_pairs for a, b in group)
          and grasp_position_error <= _GRASP_POSITION_TOLERANCE_M
          and _rotation_error_radians(
              source.hand_in_object_rotation,
              playable.hand_in_object_rotation,
          ) <= _GRASP_ORIENTATION_TOLERANCE_RADIANS
      )

  def select_certified_height_triple(structural, certification):
      joined = join_one_to_one_or_raise(structural, certification)
      eligible = tuple(sorted(
          (source for source, runtime in joined
           if runtime_is_eligible(runtime)
           and place_compatible_with_playable(
               source, certification.playable_object)),
          key=lambda row: (
              f32_from_hex(f32_hex(row.support_height_m)),
              row.sequence_id.encode("ascii"),
              row.source_id,
          ),
      ))
      table_height = f32_from_hex(f32_hex(
          certification.playable_object.table_support_height_m))
      table_return = min(
          (row for row in eligible
           if abs(f32_from_hex(f32_hex(row.support_height_m)) - table_height)
              <= _MAXIMUM_VERTICAL_CORRECTION_M),
          key=lambda row: (
              abs(f32_from_hex(f32_hex(row.support_height_m)) - table_height),
              f32_from_hex(f32_hex(row.support_height_m)),
              row.sequence_id.encode("ascii"),
              row.source_id,
          ),
          default=None,
      )
      if table_return is None:
          raise CertificationError("no compatible certified table return")
      for low_index in range(len(eligible)):
          for middle_index in range(low_index + 1, len(eligible)):
              for high_index in range(middle_index + 1, len(eligible)):
                  low = eligible[low_index]
                  middle = eligible[middle_index]
                  high = eligible[high_index]
                  heights = tuple(f32_from_hex(f32_hex(row.support_height_m))
                                  for row in (low, middle, high))
                  if not (heights[0] < heights[1] < heights[2]):
                      continue
                  if not (heights[2] - heights[0] > 0.24002):
                      continue
                  if len({low.source_id, middle.source_id,
                          high.source_id}) != 3:
                      continue
                  return ((low, middle, high), table_return)
      supported = tuple(f32_from_hex(f32_hex(row.support_height_m))
                        for row in eligible)
      measured = None if not supported else (min(supported), max(supported))
      raise CertificationError(f"no certified triple; range={measured!r}")
  ```

  Because `eligible` has a total deterministic order and enumeration is
  increasing-index order, the first returned tuple is the exact
  lexicographically first certified triple; candidates are streamed and no
  `O(n^3)` list is materialized. The table-return row is the compatible,
  runtime-certified row within `0.12 m` of the one authored table height with
  minimum `(absolute height correction, source height, ASCII sequence ID,
  source ID)`. It may equal a tier row; otherwise it is the fourth generated
  row. All three tier rows and the table-return row are compatible with the
  same single `PlayableObjectSpec` under the exact `interaction_place.cpp`
  profile/hand, componentwise bounds, grasp-position, and quaternion-angle
  tolerances. Cross-language boundary tests run the Python helper and C++ place
  selector exactly at and one float32 step across all three tolerances to
  prevent drift. Tests also cover an exact serialized `0.24002` span rejection,
  acceptance at the next representable float32 span via
  `np.nextafter(np.float32(0.24002), np.float32(np.inf))`, competing triples,
  competing table-return rows, input permutations, duplicated heights/IDs,
  profile/hand/bounds/grasp incompatibility, every runtime eligibility
  conjunct, the table-return error, and the measured-range error. Generate
  `interaction_smart_pickup_place_sources.inc` with two distinct constexpr
  arrays: the returned tier/table placement rows with explicit roles, and every
  one-to-one joined row that is runtime-eligible and compatible with the single
  playable object, sorted by join key, as certified pickup-registry rows. A row
  may appear in both arrays, but each array independently rejects duplicate
  source IDs/join keys. The generated JSON carries the same two named arrays;
  C++ never reconstructs registry rows from placement rows alone.

- [ ] **Step 5: Implement audited C++ hydration**

  Expose:

  ```cpp
  struct SmartPickupPackIdentity {
      std::string database_sha256{};
      std::string features_sha256{};
      std::string manifest_sha256{};
      uint32_t fps_numerator = 0U;
      uint32_t fps_denominator = 0U;
      uint32_t clip_count = 0U;
      uint64_t frame_count = 0U;
  };

  struct SmartPickupPlayableObjectDiagnostics {
      std::string object_id{};
      uint64_t object_profile_id = 0U;
      ObjectLocalBounds object_bounds{};
      Hand active_hand = Hand::Right;
      Transform hand_in_object{};
      float table_support_height_m = 0.0F;
  };

  struct SmartPickupCertifiedSourceDiagnostics {
      uint64_t source_id = 0U;
      std::string join_key_sha256{};
      float support_height_m = 0.0F;
  };

  struct SmartPickupPlaceLibraryDiagnostics {
      SmartPickupPackIdentity pack{};
      SmartPickupPlayableObjectDiagnostics playable_object{};
      std::array<float, 3> tier_support_heights_m{};
      std::array<uint64_t, 3> tier_source_ids{};
      float table_return_support_height_m = 0.0F;
      uint64_t table_return_source_id = 0U;
      std::vector<SmartPickupCertifiedSourceDiagnostics> certified_sources{};
      float certified_source_min_support_height_m = 0.0F;
      float certified_source_max_support_height_m = 0.0F;
  };

  struct CertifiedPickupSourceBundle {
      Database database{};
      Features features{};
      PlaceMotionLibrary place_library{};
      CertifiedPickupSourceRegistry pickup_sources;
      SmartPickupPlaceLibraryDiagnostics diagnostics{};
  };

  CertifiedPickupSourceBundle load_certified_smart_pickup_source_bundle(
      const std::filesystem::path& pack_directory);
  ```

  The factory is the only production hydration API; delete/do not expose the
  former database-only library-hydration form. It
  reads each of `interaction_database.bin`, `interaction_features.bin`, and
  `manifest.json` once into immutable byte buffers, hashes those exact buffers,
  loads/parses from those same buffers, derives the manifest's 25/1 rate and
  clip/frame counts, and
  constructs an observed `SmartPickupPackIdentity`. Before returning anything,
  compare all seven fields for exact equality with the generated frozen
  identity. Cross-check the returned `database` and `features` header
  counts/rate/feature rows
  against the observed identity. Hash mismatch, missing/extra identity field,
  malformed manifest, path/file substitution, count/rate mismatch, or a
  database object not loaded by this factory throws `FormatError`.

  For every constexpr placement and pickup-registry row, verify clip
  range, exact local-to-global-to-local joins for all five frames, phase order, active hand,
  contact-through-reverse-start, finite source table/object/grasp values,
  playable object/profile/bounds/grasp compatibility under the same C++
  tolerances, exact 25/1 fps, and unique IDs before returning the library. The
  generated rows include explicit tier/table-return roles; hydration rejects a
  missing or duplicate role and does not rediscover the return row. Build
  `pickup_sources` from the separate generated registry array by constructing
  one `CertifiedPickupSourceLocalRow` per generated row and calling the leaf's
  public `make_certified_pickup_source_identity`; pass only the returned global
  identities to the registry constructor. Hydration catches the leaf's exact
  conversion/factory exceptions and rethrows `FormatError` with the row index.
  Populate `diagnostics.certified_sources` with exactly one entry per generated
  registry row in the generated join-key order; source ID, full join key, and
  float32 support height must equal the generated include row. The generator's
  reopen check and a cross-language unit test separately require those same
  three fields to equal the corresponding final frozen JSON row; production
  C++ hydration never opens or treats that JSON as a second runtime input.
  Compute diagnostics min/max from the complete vector and verify the vector is
  nonempty with unique ID/join pairs. Placement rows are a subset, so this
  vector is also the unique union of both generated arrays.
  Tests independently mutate each of the
  three hashes, fps numerator/denominator, clip count, frame count, database
  header, feature row count, all five local/global frame joins, and registry
  row identity in isolated temporary fixture copies and require rejection; tests
  never mutate the reviewed full-pack directory. Throw `FormatError` on any mismatch.

  The bundle owns every object referenced by a runtime. Callers construct it in
  its final storage before constructing `InteractionRuntime`, pass references
  to `bundle.database`, `bundle.features`, and `bundle.place_library` plus
  `&bundle.pickup_sources`, and never move/destroy the bundle until all such
  runtimes are destroyed.

- [ ] **Step 6: Run GREEN and generate the frozen reviewed rows**

  ```bash
  python -m unittest tests.python.test_g1_precomputed_place_library -v
  build_dir=build/table-rack/place-library
  mkdir -p "$build_dir"
  make interaction_smart_pickup_scene_probe
  ./interaction_smart_pickup_scene_probe --json > "$build_dir/scene.json"
  python resources/build_g1_precomputed_place_library.py \
    --pack build/smart-pickup/full-pack \
    --scene "$build_dir/scene.json" \
    --extract-only \
    --output-structural "$build_dir/structural.json"
  make interaction_smart_pickup_place_source_probe
  ./interaction_smart_pickup_place_source_probe \
    build/smart-pickup/full-pack "$build_dir/structural.json" --json \
    > "$build_dir/certified.json"
  python resources/build_g1_precomputed_place_library.py \
    --pack build/smart-pickup/full-pack \
    --scene "$build_dir/scene.json" \
    --structural "$build_dir/structural.json" \
    --certification "$build_dir/certified.json" \
    --output-json resources/g1_smart_pickup_place_sources.json \
    --output-include interaction_smart_pickup_place_sources.inc
  python resources/build_g1_precomputed_place_library.py \
    --pack build/smart-pickup/full-pack \
    --scene "$build_dir/scene.json" \
    --structural "$build_dir/structural.json" \
    --certification "$build_dir/certified.json" \
    --output-json "$build_dir/repeat.json" \
    --output-include "$build_dir/repeat.inc"
  cmp resources/g1_smart_pickup_place_sources.json "$build_dir/repeat.json"
  cmp interaction_smart_pickup_place_sources.inc "$build_dir/repeat.inc"
  make build/tests/test_interaction_smart_pickup_place_library
  build/tests/test_interaction_smart_pickup_place_library \
    build/smart-pickup/full-pack
  ```

  Expected: exactly three tier IDs, strict span above `0.24002 m`, every source
  real-attached, schema/join corruption tests pass, repeated output is
  byte-identical, and the path-owning C++ bundle factory accepts the exact
  seven-field full pack and rejects every independently corrupted identity.

- [ ] **Step 7: Review, commit, and push**

  ```bash
  git add resources/g1_interaction_builder/precomputed_place.py \
    resources/build_g1_precomputed_place_library.py \
    resources/g1_smart_pickup_place_sources.json \
    interaction_smart_pickup_place_sources.inc \
    interaction_smart_pickup_place_library.h \
    interaction_smart_pickup_place_library.cpp \
    interaction_smart_pickup_place_source_probe.cpp \
    interaction_smart_pickup_scene_probe.cpp \
    tests/python/test_g1_precomputed_place_library.py \
    tests/cpp/test_interaction_smart_pickup_scene.cpp \
    tests/cpp/test_interaction_smart_pickup_place_library.cpp Makefile
  git diff --cached --check
  git commit -m "feat: build certified height-indexed place library"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 4: Add the semantic table/rack catalog and fixed matrix

**Files:**

- Create: `interaction_surface_handle.h`
- Create: `interaction_table_rack_scene.h`
- Create: `interaction_table_rack_scene.cpp`
- Create: `tests/cpp/test_interaction_table_rack_scene.cpp`
- Modify: `interaction_place_target.h`
- Modify: `interaction_target.h`
- Modify: `interaction_target.cpp`
- Modify: `interaction_runtime.cpp`
- Modify: `interaction_smart_pickup_scene.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_scene.cpp`
- Modify: `tests/cpp/test_interaction_place_target.cpp`
- Modify: `tests/cpp/test_interaction_target.cpp`
- Modify: `tests/cpp/interaction_runtime_fixture.h`
- Modify: `Makefile`

**Interfaces:**

- Produces: exact support identity on every free target, semantic slots, three
  rack tiers from certified source heights, 15 positive rows, and mandatory
  below/above negative rows.

- [ ] **Step 1: Write failing support-identity and matrix tests**

  Require placement to persist exact support identity:

  ```cpp
  struct PlacedSupportContext {
      Transform table_world{};
      vec3 table_size{};
      SurfaceHandle surface{};
      uint32_t affordance_id = 0U;
  };

  // Add these fields to InteractionTarget:
  SurfaceHandle support_surface{};
  uint32_t support_affordance_id = 0U;

  // Preserve the existing reset overload and add:
  TargetHandle reset(
      uint64_t id,
      Transform object_world,
      PlacedSupportContext authored_support);
  ```

  Move the existing `SurfaceHandle` definition unchanged from
  `interaction_place_target.h` into `interaction_surface_handle.h`, include the
  new leaf header from both `interaction_target.h` and
  `interaction_place_target.h`, and retain equality over both `id` and
  `generation`. This avoids an include cycle while allowing the target registry
  to persist the complete handle.

  Tests assert `place_held` copies the complete `SurfaceHandle` including a
  deliberately non-default generation, changing only the generation makes
  `same_interaction_target_snapshot` false, target generation increments exactly
  once, and `reset` restores the authored table handle/affordance. Matrix tests
  assert 15 stable rows, five exact offsets per tier, three distinct source IDs,
  height span, unique semantic IDs, placement fit, and no ambiguous resolver
  use.

- [ ] **Step 2: Run RED**

  ```bash
  make build/tests/test_interaction_place_target
  make build/tests/test_interaction_target
  make build/tests/test_interaction_table_rack_scene
  ```

  Expected: compile failure for missing support identity and scene API.

- [ ] **Step 3: Implement the catalog contracts**

  ```cpp
  enum class TableRackTier : uint8_t { Low, Middle, High };
  enum class TableRackOffset : uint8_t {
      Left, Centre, Right, Shallow, Deep
  };

  struct SemanticPlacementSlot {
      uint32_t semantic_slot_id = 0U;
      SurfaceHandle surface{};
      uint32_t affordance_id = 0U;
      std::string label{};
      TableRackTier tier = TableRackTier::Low;
      TableRackOffset offset = TableRackOffset::Centre;
      vec3 offset_surface{};
      uint64_t place_source_id = 0U;
  };

  struct TableRackMatrixRow {
      uint32_t row_id = 0U;
      uint32_t table_slot_id = 0U;
      uint32_t rack_slot_id = 0U;
      bool expected_success = true;
  };

  enum class TableRackNegativeKind : uint8_t {
      PickupStaleTarget = 0U,
      PlaceBelowLow = 1U,
      PlaceAboveHigh = 2U,
  };

  struct TableRackNegativeRow {
      uint32_t case_id = 0U;
      TableRackNegativeKind kind = TableRackNegativeKind::PickupStaleTarget;
      uint32_t source_slot_id = 0U;
      SemanticPlacementSlot request_slot{};
      uint32_t recovery_slot_id = 0U;
      Reason expected_reason = Reason::None;
  };

  struct TableRackSceneDefinition {
      InteractionTarget initial_target{};
      std::vector<PlacementSurface> surfaces{};
      std::vector<SemanticPlacementSlot> slots{};
      std::vector<TableRackMatrixRow> positive_rows{};
      SemanticPlacementSlot below_low_place_slot{};
      SemanticPlacementSlot above_high_place_slot{};
      std::array<TableRackNegativeRow, 3> negative_rows{};
  };

  TableRackSceneDefinition make_table_rack_scene(
      const SmartPickupPlaceLibraryDiagnostics& library);
  ```

  Use fixed rack X/Z world placement `(1.80, 3.00)`, shelf usable size
  `(1.20, 0.50)`, and the certified low/middle/high support Y values. Build all
  five affordances per tier using the exact offsets in Global Constraints.
  Freeze row IDs as
  `1 + 5 * static_cast<uint32_t>(tier) + static_cast<uint32_t>(offset)` using
  the enum declaration orders above: low rows `1..5`, middle `6..10`, high
  `11..15`. Thus graphical row `8` is deterministically Middle/Right.

  Freeze both negative slots as real catalog entries and real surfaces, not
  probe-local transforms. Form the ordered `all_generated_source_heights` by
  transforming `library.certified_sources` in its frozen join-key order and
  preserving each exact authored float32 support height. Hydration has already
  proved this is the nonempty unique union of pickup-registry and placement
  rows; scene construction rejects an empty vector or a min/max diagnostic that
  does not bit-equal recomputation from it.
  Use the bundle diagnostics' `certified_source_min_support_height_m` and
  `certified_source_max_support_height_m` only as starting boundaries. Construct
  each final authored float height with the same float subtraction used by
  `interaction_place`:

  ```cpp
  float first_strictly_rejected_height(
      float boundary,
      float direction,
      const std::vector<float>& all_generated_source_heights) {
      const float rejected_delta = std::nextafter(
          0.12F, std::numeric_limits<float>::infinity());
      float authored = boundary + direction * rejected_delta;
      const float outward = direction < 0.0F
          ? -std::numeric_limits<float>::infinity()
          : std::numeric_limits<float>::infinity();
      auto strictly_rejects_every_source = [&](float value) {
          return std::all_of(
              all_generated_source_heights.begin(),
              all_generated_source_heights.end(),
              [&](float source_height) {
                  const float selector_requested_correction =
                      value - source_height;
                  return std::isfinite(selector_requested_correction) &&
                      std::fabs(selector_requested_correction) > 0.12F;
              });
      };
      while (!strictly_rejects_every_source(authored)) {
          const float next = std::nextafter(authored, outward);
          if (!std::isfinite(next) || next == authored) {
              throw FormatError("cannot author strict correction rejection");
          }
          authored = next;
      }
      return authored;
  }
  ```

  `interaction_table_rack_scene.cpp` includes `<algorithm>`, `<cmath>`,
  `<limits>`, and `<vector>` directly for this construction.

  Require a nonempty finite source-height vector and `direction` exactly `-1.0F`
  or `+1.0F`. The first rounded candidate may or may not already pass; the loop
  steps the **final authored float** outward until the selector-side float
  subtraction is strictly outside the inclusive `0.12F` limit for every
  generated source. Never compare a double-precision ideal delta or assume the
  initial min/max arithmetic survived float32 rounding. The below-low surface is
  `SurfaceHandle{9001U, 1U}`, affordance/semantic slot `9001U`, Centre offset,
  and support Y exactly `first_strictly_rejected_height(
  certified_source_min_support_height_m, -1.0F, all_generated_source_heights)`.
  The
  above-high surface is `SurfaceHandle{9002U, 1U}`, affordance/semantic slot
  `9002U`, Centre offset, and support Y exactly
  `first_strictly_rejected_height(
  certified_source_max_support_height_m, +1.0F, all_generated_source_heights)`.
  Both use the rack's exact
  support orientation/usable size, appear once in `surfaces` and `slots`, and
  have IDs disjoint from every positive table/rack entry. Freeze negative case
  `9000U` as `PickupStaleTarget`: `source_slot_id` and `request_slot` name the
  one valid authored table slot, its submitted target handle is copied with
  generation incremented by one, expected reason is `Reason::TargetChanged`,
  and recovery is Low/Centre. Freeze case `9001U` as `PlaceBelowLow`, sourced
  from the valid table slot, expected `Reason::CorrectionLimit`, and recovered
  to Low/Centre. Freeze case `9002U` as `PlaceAboveHigh`, also sourced from the
  valid table slot, expected `Reason::CorrectionLimit`, and recovered to
  High/Centre. Tests assert all of those exact handles,
  affordances, heights, meanings, and uniqueness, including that the invalid
  place request resolves by complete handle plus affordance rather than a
  synthesized pose. Tests run the exact selector-side
  `authored_height - source_height` float subtraction against every generated
  source for both negative surfaces and require finite
  `fabs(correction) > 0.12F`; they include boundary values whose initial
  min/max-plus-delta expression rounds to `<=0.12F`, proving the outward loop is
  necessary.

- [ ] **Step 4: Persist exact surface/affordance identity on release**

  In `InteractionRuntime` construct the destination support from the accepted
  request, not a resolver:

  ```cpp
  const PlacedSupportContext support{
      surface->support_volume_world,
      surface->support_volume_size,
      place_request_->surface,
      place_request_->affordance_id,
  };
  const std::optional<TargetHandle> placed = registry_->place_held(
      held, owner, placed_world, support);
  ```

  Add the complete `support_surface` handle and affordance ID to
  `same_interaction_target_snapshot` and every fixture author. Never reconstruct
  a handle as `{surface_id, 1U}` and never compare only `surface.id`. The Smart
  Pickup target authors the table's exact surface handle/affordance ID. The new
  reset overload restores pose plus that authored full handle/table context and
  is used by scenario R; the legacy pose-only overload retains its current
  behavior.

- [ ] **Step 5: Run GREEN and focused regression**

  ```bash
  make build/tests/test_interaction_place_target \
       build/tests/test_interaction_target \
       build/tests/test_interaction_table_rack_scene \
       build/tests/test_interaction_runtime
  build/tests/test_interaction_place_target
  build/tests/test_interaction_target
  build/tests/test_interaction_table_rack_scene
  build/tests/test_interaction_runtime
  ```

- [ ] **Step 6: Review, commit, and push**

  ```bash
  git add interaction_surface_handle.h interaction_place_target.h \
    interaction_target.h interaction_target.cpp interaction_runtime.cpp \
    interaction_smart_pickup_scene.cpp \
    interaction_table_rack_scene.h interaction_table_rack_scene.cpp \
    tests/cpp/test_interaction_target.cpp \
    tests/cpp/test_interaction_place_target.cpp \
    tests/cpp/test_interaction_table_rack_scene.cpp \
    tests/cpp/test_interaction_smart_pickup_scene.cpp \
    tests/cpp/interaction_runtime_fixture.h Makefile
  git diff --cached --check
  git commit -m "feat: add semantic table rack matrix"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 5: Advance only after exact release and support commit

**Files:**

- Create: `interaction_table_rack_route.h`
- Create: `interaction_table_rack_route.cpp`
- Create: `tests/cpp/test_interaction_table_rack_route.cpp`
- Modify: `Makefile`

**Interfaces:**

- Consumes: one frozen matrix row, registered semantic slots, current target,
  and runtime diagnostics.
- Produces: exact highlighted destination, cycle/leg counters, reset/cancel
  semantics, and no advancement on animation-only or failed attempts.

- [ ] **Step 1: Write failing route-state tests**

  ```cpp
  const TargetHandle initial_target{77U, 9U};
  TableRackRoute route(row, catalog, initial_target);
  TEST_CHECK(route.leg() == TableRackLeg::TableToRack);
  TEST_CHECK(route.highlighted().semantic_slot_id == row.rack_slot_id);

  route.observe(runtime_carry, held_target);
  TEST_CHECK(route.completed_legs() == 0U);

  route.observe(runtime_locomotion, free_target_on_wrong_support);
  TEST_CHECK(route.completed_legs() == 0U);

  route.observe(runtime_locomotion, free_target_on_exact_rack_support);
  TEST_CHECK(route.completed_legs() == 1U);
  TEST_CHECK(route.leg() == TableRackLeg::RackToTable);
  ```

  Cover failure/cancel retaining route, four exact releases completing two round
  trips, generation sequence `g+1..g+4`, and reset restoring row/leg/counters.

- [ ] **Step 2: Run RED**

  ```bash
  make build/tests/test_interaction_table_rack_route
  ```

- [ ] **Step 3: Implement the pure route state machine**

  ```cpp
  enum class TableRackLeg : uint8_t {
      TableToRack = 0U,
      RackToTable = 1U,
  };

  class TableRackRoute {
  public:
      TableRackRoute(
          TableRackMatrixRow row,
          std::vector<SemanticPlacementSlot> catalog,
          TargetHandle initial_target);
      const SemanticPlacementSlot& highlighted() const;
      void observe(
          const RuntimeDiagnostics& runtime,
          const InteractionTarget& target);
      void reset(TargetHandle initial_target);
      TableRackLeg leg() const;
      uint32_t completed_legs() const;
      uint32_t completed_round_trips() const;
  private:
      bool exact_destination_committed(
          const InteractionTarget& target) const;
  };
  ```

  `exact_destination_committed` requires Locomotion, `Free`, owner zero,
  unattached diagnostics, the exact complete `SurfaceHandle` (ID and
  generation), exact affordance ID, expected target ID, and exactly one target
  generation increment since the prior leg. The constructor used by production
  and tests is exactly the three-argument signature shown above; do not add a
  two-argument convenience constructor with different initial-generation
  behavior.

- [ ] **Step 4: Run GREEN**

  ```bash
  make build/tests/test_interaction_table_rack_route
  build/tests/test_interaction_table_rack_route
  ```

- [ ] **Step 5: Review, commit, and push**

  ```bash
  git add interaction_table_rack_route.h interaction_table_rack_route.cpp \
    tests/cpp/test_interaction_table_rack_route.cpp Makefile
  git diff --cached --check
  git commit -m "feat: add exact table rack route state"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 6: Prove the repeated positive and negative full-chain matrix

**Files:**

- Create: `interaction_table_rack_probe.cpp`
- Create: `tests/python/test_table_rack_full_chain_gate.py`
- Modify: `Makefile`

**Interfaces:**

- Consumes: verified full-pack `CertifiedPickupSourceBundle`, final frozen
  `resources/g1_smart_pickup_place_sources.json`, scene catalog, route state
  machine, `SmartPickupController`, flat locomotion snapshots,
  `InteractionRuntime`.
- Produces: canonical JSON with every declared row and event-level lifecycle
  evidence; this is Plan B's hard prerequisite.

- [ ] **Step 1: Write the failing Python evidence contract**

  The top-level canonical JSON document has the exact key set
  `{schema_version,record_type,pack_identity,rows,negative_rows}` with
  `schema_version == 1` and
  `record_type == "table_rack_full_chain_matrix"`. Handles are always encoded
  as `{"generation": <uint32>, "id": <uint64>}`; an ID without generation is
  invalid. `pack_identity` is the exact seven-field `PackIdentity` object from
  Task 3 and must match the frozen source library and input pack.

  Before validating positive or negative rows, require a nonempty
  `TABLE_RACK_SOURCES` path and call
  `read_frozen_smart_pickup_place_sources(path, expected_pack)` with that exact
  evidence `pack_identity`. Retain the immutable returned artifact for the
  complete document. Import the reader and both lookup functions directly from
  `resources.g1_interaction_builder.precomputed_place`; the gate must not copy
  their parsing/join logic. Python has no access to `CertifiedPickupSourceBundle` or
  its C++ registry; fixture/catalog declarations are never a lookup substitute.

  Every positive row contains exactly four `pickups` and four `placements`, in
  `leg_index` order `0,1,2,3`. Both record kinds share this exact lifecycle
  identity payload:

  ```python
  _LEG_LIFECYCLE_KEYS = {
      "affordance_id",
      "applied_vertical_correction_m",
      "attachment_edges",
      "leg_index",
      "release_edges",
      "requested_vertical_correction_m",
      "source_id",
      "source_kind",
      "support_handle",
      "target_generation_after",
      "target_generation_before",
  }
  ```

  Pickup records additionally publish `request_id`, `owner_request`,
  `terminal_state`, `terminal_object_state`, `attached`,
  `post_carry_tick_observed`, `grasp_preserved`, `source_join_key_sha256`,
  `source_sequence_id`, `source_object_id`, `source_clip_ordinal`,
  `source_range_start`, `source_range_stop`, `source_entry_global_frame`,
  `source_contact_global_frame`, `source_lift_global_frame`,
  `source_hold_global_frame`, `source_reverse_start_global_frame`, `source_active_hand`,
  `source_object_profile_id`, `source_object_bounds_f32_hex`,
  `source_hand_in_object_f32_hex`, and `source_support_height_f32_hex`. Their
  `support_handle` and `affordance_id` are the exact support stored on the Free
  target before that pickup; `source_kind == "pickup_match"`; `source_id` and
  all provenance fields come from the runtime's frozen
  `CertifiedPickupSourceIdentity`; both correction fields are exactly `0.0`;
  attachment edges are one, release edges are zero, and target generation is
  unchanged. The gate recomputes the canonical join key, verifies its first
  eight bytes produce `source_id`, converts all five recorded global event
  frames to local with the checked Task 3 helper, calls
  `resolve_frozen_pickup_source` with the recorded source ID and full join key,
  and compares every decoded local identity field to that returned frozen row.
  A fixture/catalog source ID is never accepted as pickup evidence.

  Placement records additionally publish `destination_slot_id`,
  `destination_tier`, `requested_support_height_m`,
  `source_support_height_m`, `target_support_height_m`, `terminal_state`,
  `terminal_object_state`, and `attached`. Their `support_handle` and
  `affordance_id` are the exact routed destination; `source_kind` and
  `source_id` come from the actually selected frozen `PlaceCandidate`; attach
  edges are zero, release edges are one, target generation advances exactly
  once, and the terminal state is Locomotion/Free/unattached on that same full
  support handle.

  Map each placement destination to its required frozen role exactly:
  `low->tier_low`, `middle->tier_middle`, `high->tier_high`, and
  `table->table_return`. Call `resolve_frozen_placement_source` with the recorded
  selected source ID and that role, require
  `source_kind == "precomputed_reversed_pickup"`, and compare recorded source,
  requested, target, and correction heights to the resolved source and routed
  destination. An ID present only in `certified_pickup_sources`, a placement ID
  with the wrong role, or a C++ bundle-only claim fails.

  Require every positive row to expose and satisfy:

  ```python
  assert row["updates_hz"] == 25
  assert len(row["pickups"]) == 4
  assert len(row["placements"]) == 4
  assert [item["leg_index"] for item in row["pickups"]] == [0, 1, 2, 3]
  assert [item["leg_index"] for item in row["placements"]] == [0, 1, 2, 3]
  assert row["round_trips"] == 2
  assert row["attachments"] == 4
  assert row["releases"] == 4
  assert row["repicks"] == 4
  assert row["initial_target_id"] == row["final_target_id"]
  assert row["final_generation"] == row["initial_generation"] + 4
  assert row["reset_count"] == 0
  assert row["registry_rebuild_count"] == 0
  assert all(place["source_kind"] == "precomputed_reversed_pickup"
             for place in row["placements"])
  assert all(abs(place["applied_vertical_correction_m"]) <= 0.12
             for place in row["placements"])
  ```

  Across the complete positive set require 15 unique row IDs, all five offsets
  at every tier, and strict height span. Compute the actually selected rack-tier
  IDs from placement records, not library declarations:

  ```python
  selected_by_tier = {
      tier: {
          place["source_id"]
          for row in rows
          for place in row["placements"]
          if place["destination_tier"] == tier
      }
      for tier in ("low", "middle", "high")
  }
  assert all(len(ids) == 1 for ids in selected_by_tier.values())
  assert len({next(iter(ids)) for ids in selected_by_tier.values()}) == 3
  ```

  Each singleton must equal that tier's certified frozen source ID. Require all
  declared negatives and same-runtime recovery. Corruption tests independently
  remove/reorder/duplicate each leg, mutate either handle generation, swap an
  affordance/source kind/source ID, alter either correction, alter edge counts,
  or change either generation, and require rejection.

- [ ] **Step 2: Run RED**

  ```bash
  python -m unittest tests.python.test_table_rack_full_chain_gate -v
  ```

  Expected: missing probe/gate configuration.

- [ ] **Step 3: Implement one unchanged-runtime row runner**

  The probe owns one fixture for the entire row:

  ```cpp
  CertifiedPickupSourceBundle bundle =
      load_certified_smart_pickup_source_bundle(pack_directory);
  const TableRackSceneDefinition scene =
      make_table_rack_scene(bundle.diagnostics);
  for (const TableRackMatrixRow& row : scene.positive_rows) {
    RowFixture fixture(bundle, scene, row);
    const TargetHandle initial = fixture.target_handle();
    for (uint32_t leg = 0U; leg < 4U; ++leg) {
      const TargetHandle generation_before_pick = fixture.target_handle();
      fixture.drive_to_pickup_source();
      const SmartPickupLifecycleWitness pickup = fixture.pick_to_carry();
      require_actual_attachment(pickup);
      require(pickup.actual_source.source_id != 0U,
              "runtime pickup source was not certified");
      require(pickup.actual_source.join_key_sha256.size() == 64U,
              "runtime pickup join key was not frozen");
      evidence.pickups.push_back(make_pickup_record(
          leg, pickup.actual_source, fixture.target_support_before_pick(),
          pickup, generation_before_pick));
      fixture.carry_to(route.highlighted());
      const PlaceLegWitness placement =
          fixture.place_to_locomotion(route.highlighted());
      evidence.placements.push_back(make_placement_record(
          leg, route.highlighted(), placement));
      route.observe(fixture.runtime().diagnostics(), fixture.target());
    }
    require(fixture.target().handle.id == initial.id, "target ID changed");
    require(fixture.target().handle.generation == initial.generation + 4U,
            "generation did not advance once per release");
  }
  ```

  Construct `bundle` once in final process storage before any row runtime and
  keep it alive until every row fixture is destroyed. `RowFixture` constructs
  its `InteractionRuntime` with `bundle.database`, `bundle.features`,
  `bundle.place_library`, and `&bundle.pickup_sources`; it may not load the pack
  again or create a second certified registry. Thus Task 6, unlike Task 1,
  requires every successful pickup's actual source ID/join key to be nonzero
  and resolvable in `bundle.pickup_sources`.

  `make_pickup_record` copies source metadata only from
  `pickup.actual_source`; `target_support_before_pick()` supplies only the
  independently observed support handle/affordance and may not supply source
  provenance. Never construct a second runtime/registry inside the loop. Emit failed rows
  rather than omitting them. `PlaceLegWitness` freezes the exact destination
  `SurfaceHandle`, affordance, selected source kind/ID, all three support
  heights, requested/applied signed correction, attach/release edge counts, and
  before/after target generations before route advancement.

- [ ] **Step 4: Add mandatory negative and recovery execution**

  Execute exactly the three `TableRackNegativeRow` objects authored by Task 4;
  do not recompute heights or create probe-local surfaces. The evidence parser
  requires exactly these nested key sets:

  ```python
  _STATE_SNAPSHOT_KEYS = frozenset({
      "attached", "object_state", "owner_request", "runtime_state",
      "support_affordance_id", "support_handle", "target_generation",
      "target_id",
  })
  _PICKUP_NEGATIVE_KEYS = frozenset({
      "after_rejection", "attachment_edges", "before", "case_id",
      "kind", "reason", "recovery", "registry_instance_id", "release_edges",
      "request_count", "requested_target_handle", "result",
      "runtime_instance_id", "source_affordance_id", "source_slot_id",
      "source_support_handle",
  })
  _PICKUP_RECOVERY_KEYS = frozenset({
      "after_pickup", "after_place", "attachment_edges", "before",
      "destination_affordance_id", "destination_slot_id",
      "destination_support_handle", "pickup_source_id",
      "pickup_source_join_key_sha256", "placement_source_id",
      "placement_source_kind", "registry_instance_id", "release_edges",
      "requested_vertical_correction_f32_hex",
      "applied_vertical_correction_f32_hex", "result", "runtime_instance_id",
  })
  _PICKUP_PRECONDITION_KEYS = frozenset({
      "attached", "attachment_edges", "carry_object_displacement_f32_hex",
      "carry_root_displacement_f32_hex", "grasp_preserved", "owner_request",
      "post_carry_tick_observed", "registry_instance_id", "release_edges",
      "request_count", "request_id", "reason", "result",
      "runtime_instance_id", "source_id",
      "source_join_key_sha256", "target_generation_after",
      "target_generation_before", "target_id", "terminal_object_state",
      "terminal_state",
  })
  _PLACE_NEGATIVE_KEYS = frozenset({
      "after_rejection", "attachment_edges", "before", "case_id",
      "invalid_affordance_id", "invalid_slot_id", "invalid_support_handle",
      "kind", "pickup_precondition", "reason", "recovery",
      "registry_instance_id", "release_edges", "request_count", "result",
      "runtime_instance_id",
  })
  _PLACE_RECOVERY_KEYS = frozenset({
      "after", "applied_vertical_correction_f32_hex", "attachment_edges",
      "before", "destination_affordance_id", "destination_slot_id",
      "destination_support_handle", "registry_instance_id", "release_edges",
      "requested_vertical_correction_f32_hex", "result",
      "runtime_instance_id", "source_id", "source_kind",
  })
  ```

  `support_handle`, `requested_target_handle`, `source_support_handle`,
  `invalid_support_handle`, and
  `destination_support_handle` always use the exact two-key handle schema from
  Step 1. Every set is equality-checked; unknown and missing keys fail.
  `runtime_instance_id` and `registry_instance_id` are nonzero process-local
  monotonically assigned integers, stable for the entire case, used only to
  prove identity and never as pointers. `request_count` is the observed delta
  across the invalid request, not a fixture declaration. `negative_rows` is
  exactly three records sorted by `case_id` (`9000`, `9001`, `9002`); the
  validator selects `_PICKUP_NEGATIVE_KEYS` only for `pickup_stale_target` and
  `_PLACE_NEGATIVE_KEYS` only for the two place kinds. Each place row's
  `pickup_precondition` must equal `_PICKUP_PRECONDITION_KEYS` exactly.

  Case `9000` has `kind == "pickup_stale_target"`. The object begins on the one
  valid certified table support. `requested_target_handle.id` equals the live
  target ID and its generation is exactly live generation plus one, while the
  registry itself is not mutated. The invalid request delta is one,
  `result == "Rejected"`, and `reason == "TargetChanged"`. Before and
  `after_rejection` are byte-identical `Locomotion/Free`, owner `0`, unattached
  snapshots on the valid table handle/affordance with unchanged generation;
  the invalid attempt has zero attachment/release edges. Recovery changes only
  the submitted handle to the current live handle, then performs a real
  certified pickup and a real placement to Low/Centre. Its `before` equals
  `after_rejection`, `after_pickup` is `Carry/Held`, attached with the pickup
  request as owner and unchanged generation, and `after_place` is
  `Locomotion/Free`, owner `0`, unattached with generation plus one on the exact
  Low/Centre support. The nested recovery has the same runtime/registry IDs,
  one attachment and one release edge, a nonzero actual pickup source ID and
  canonical join key, a compatible precomputed placement source, and
  finite requested/applied placement corrections whose absolute values do not
  exceed `0.12`, and `result == "Succeeded"`. No reset, target mutation, teleport, second runtime,
  or second registry is permitted. The validator resolves the recovery's
  `(pickup_source_id,pickup_source_join_key_sha256)` through
  `resolve_frozen_pickup_source` and its placement source through
  `resolve_frozen_placement_source(...,required_role="tier_low")`.

  Cases `9001` and `9002` first obtain the same playable object from their
  declared valid table source through a real certified pickup in one runtime/
  registry. Their nested `pickup_precondition` records the observed request ID
  and count, `result == "Succeeded"`, `reason == "None"`, one attachment and
  zero release edges, `Carry/Held`, attached,
  owner equal to request, unchanged target ID/generation, a normal post-Carry
  tick, preserved grasp, finite positive root/object Carry displacement, the
  actual nonzero source ID/canonical join key, and the same nonzero outer
  runtime/registry instance IDs. The validator resolves that exact join/source
  with `resolve_frozen_pickup_source` in the parsed frozen JSON artifact and
  independently checks profile, hand, bounds, and grasp compatibility with the
  artifact's `playable_object` under the
  Task 3 tolerances. Any compatible certified source is valid; the fixture must
  not claim or require the table-return placement source unless an exact frozen
  matcher snapshot separately guarantees that result.

  `before` must equal the terminal target/runtime state described by
  `pickup_precondition`, including owner, attachment, target ID, and generation.
  They use respectively `kind == "place_below_low"` with Task 4's
  exact below-low handle and Low/Centre recovery, and
  `kind == "place_above_high"` with its exact above-high handle and High/Centre
  recovery. Because Task 4 places these surfaces more than `0.12F` outside the
  minimum/maximum support of every certified registry row, both precomputed and
  same-pickup reverse tiers reject regardless of which compatible certified
  pickup won. Immediately before and after each invalid place request, snapshots
  have the same target ID/generation, `Carry/Held`, the same nonzero owner
  request, and `attached == true`; the invalid attempt has request delta one,
  zero attachment/release edges, `result == "Rejected"`, and
  `reason == "CorrectionLimit"`. Required recovery `before` equals
  `after_rejection`; recovery IDs equal the outer instance IDs; it selects a
  compatible `precomputed_reversed_pickup` source, applies a finite correction
  of absolute value at most `0.12`, emits zero attachment and exactly one
  release edge, increments generation exactly once, and ends
  `Locomotion/Free`, owner `0`, unattached on the exact recovery handle/
  affordance. Tests mutate every before/after state, owner, attachment,
  generation, edge, instance ID, requested target/handle generation, recovery
  source, every pickup-precondition request/state/edge/displacement/join/source
  field, and every recovery field and require rejection. These are the required
  same-runtime recovery witnesses.

  Resolve case `9001` recovery placement with required role `tier_low` and case
  `9002` with `tier_high`. Positive and negative validation tests independently
  mutate a source ID while retaining its join, mutate a join while retaining its
  ID, substitute a certified-only row, and use a valid placement source under
  the wrong role; every mutation must fail through the public artifact reader/
  lookup APIs.

- [ ] **Step 5: Run executable normal/release full-chain parity**

  Add a release build of the same probe and a headless umbrella target. Both
  binaries consume the same frozen pack and must emit byte-identical canonical
  JSON before either result is accepted:

  ```make
  RELEASE_INTERACTION_TABLE_RACK_PROBE := \
    $(CPP_TEST_DIR)/interaction_table_rack_probe_release

  INTERACTION_TABLE_RACK_PROBE_SOURCES := \
    interaction_table_rack_probe.cpp \
    interaction_table_rack_scene.cpp interaction_table_rack_route.cpp \
    interaction_smart_pickup_scenarios.cpp \
    interaction_smart_pickup_place_library.cpp \
    interaction_pickup_provenance.cpp \
    $(filter-out interaction_pickup_provenance.cpp,\
      $(INTERACTION_SMART_PICKUP_CONTROLLER_LINK_SOURCES))
  INTERACTION_TABLE_RACK_PROBE_HEADERS := \
    interaction_table_rack_scene.h interaction_table_rack_route.h \
    interaction_smart_pickup_scenarios.h \
    interaction_smart_pickup_place_library.h \
    interaction_smart_pickup_place_sources.inc \
    interaction_pickup_provenance.h \
    $(filter-out interaction_pickup_provenance.h,\
      $(INTERACTION_SMART_PICKUP_CONTROLLER_HEADERS))

  interaction_table_rack_probe: $(INTERACTION_TABLE_RACK_PROBE_SOURCES) \
      $(INTERACTION_TABLE_RACK_PROBE_HEADERS)
	$(CXX) $(CPP_TEST_FLAGS) $(INTERACTION_TABLE_RACK_PROBE_SOURCES) -o $@

  $(RELEASE_INTERACTION_TABLE_RACK_PROBE): \
      $(INTERACTION_TABLE_RACK_PROBE_SOURCES) \
      $(INTERACTION_TABLE_RACK_PROBE_HEADERS) | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $(CONTROLLER_RELEASE_PARITY_FLAGS) \
	  $(INTERACTION_TABLE_RACK_PROBE_SOURCES) -o $@

  .PHONY: gate-table-rack-full-chain-parity
  gate-table-rack-full-chain-parity: interaction_table_rack_probe \
      $(RELEASE_INTERACTION_TABLE_RACK_PROBE)
	@mkdir -p build/table-rack/evidence
	@./interaction_table_rack_probe \
	  "$(INTERACTION_DEMO_PACK)" --json \
	  > build/table-rack/evidence/full-chain-normal.json
	@$(RELEASE_INTERACTION_TABLE_RACK_PROBE) \
	  "$(INTERACTION_DEMO_PACK)" --json \
	  > build/table-rack/evidence/full-chain-release.json
	@cmp build/table-rack/evidence/full-chain-normal.json \
	     build/table-rack/evidence/full-chain-release.json
	@TABLE_RACK_EVIDENCE="$(CURDIR)/build/table-rack/evidence/full-chain-normal.json" \
	  TABLE_RACK_PACK="$(CURDIR)/$(INTERACTION_DEMO_PACK)" \
	  TABLE_RACK_SOURCES="$(CURDIR)/resources/g1_smart_pickup_place_sources.json" \
	  python -m unittest tests.python.test_table_rack_full_chain_gate -v
	@TABLE_RACK_EVIDENCE="$(CURDIR)/build/table-rack/evidence/full-chain-release.json" \
	  TABLE_RACK_PACK="$(CURDIR)/$(INTERACTION_DEMO_PACK)" \
	  TABLE_RACK_SOURCES="$(CURDIR)/resources/g1_smart_pickup_place_sources.json" \
	  python -m unittest tests.python.test_table_rack_full_chain_gate -v

  .PHONY: gate-smart-pickup-plan-a-headless
  gate-smart-pickup-plan-a-headless: gate-table-rack-full-chain-parity
  ```

  ```bash
  make gate-table-rack-full-chain-parity \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  make gate-smart-pickup-plan-a-headless \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  make build/tests/test_interaction_runtime \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  build/tests/test_interaction_runtime
  ```

  Expected: every declared row present and passing. If the measured data cannot
  satisfy the strict height span/full chain, stop and report the supported range;
  do not proceed to Plan B or relabel unreachable shelves.

- [ ] **Step 6: Review, commit, and push**

  ```bash
  git add interaction_table_rack_probe.cpp \
    tests/python/test_table_rack_full_chain_gate.py Makefile
  git diff --cached --check
  git commit -m "test: prove repeated table rack full chain"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 7: Expose keyboard control, rack rendering, HUD, and 25-fps evidence

**Files:**

- Modify: `controller.cpp`
- Modify: `interaction_debug_draw.h`
- Create: `resources/validate_table_rack_evidence.py`
- Create: `tests/python/test_playable_table_rack_evidence.py`
- Modify: `Makefile`

**Interfaces:**

- Consumes: registered table/rack scene, `TableRackRoute::highlighted`, existing
  WASD/F/X/R controls, current runtime output, and exact support identity.
- Produces: controllable table/rack play, highlighted destination, source-kind
  HUD, route counters, and atomically published external native-25-fps JSONL/PNG
  evidence through the controller's existing environment-driven autodemo path.

- [ ] **Step 1: Write failing environment, HUD, and validator tests**

  `controller.cpp` has `int main(void)` and `parse_autodemo_environment()`;
  therefore this task must not add command-line flags. In
  `tests/python/test_playable_table_rack_evidence.py`, add source-contract tests
  that require the existing selectors plus this exact third selector:

  ```text
  MM_INTERACTION_AUTODEMO
  MM_INTERACTION_PLACE_AUTODEMO
  MM_INTERACTION_TABLE_RACK_AUTODEMO
  ```

  Require exactly one selector, with the selected value exactly `1`. Table/rack
  mode additionally requires
  `MM_INTERACTION_TABLE_RACK_ROW_ID=<canonical nonzero uint32 decimal>`; reject
  missing, empty, signed, whitespace-padded, leading-zero, overflow, or unknown
  row IDs, and reject this variable when another mode is selected. Reuse the
  existing required `MM_INTERACTION_LOG` and `MM_INTERACTION_SCREENSHOT`
  variables unchanged. Continue loading the pack from the existing
  `MM_INTERACTION_PACK` lookup and the locomotion-feature output from the
  existing `MM_FEATURES_OUTPUT` lookup; table/rack mode requires both to be
  explicitly nonempty even though ordinary/manual mode retains its current
  feature-output default.

  Source-contract tests also require controller setup to construct and retain
  `CertifiedPickupSourceBundle` before `InteractionRuntime`, pass its exact
  database/features/place-library/registry members, register every
  catalog surface, bypass the nearby resolver for routed placement, retain
  `int main(void)`, call `SetTargetFPS(25)`, and draw these HUD labels:

  ```text
  source=<semantic label>
  destination=<semantic label>
  place_source=precomputed_reversed_pickup:<source id>
  correction=<metres>
  target=<id>:<generation> owner=<request> state=<state> attached=<bool>
  route=<row> leg=<leg>/4 round_trip=<count>/2
  ```

  Define the validator's public API before implementation:

  ```python
  class TableRackEvidenceValidationError(ValueError):
      pass

  @dataclass(frozen=True)
  class TableRackPlayableSummary:
      row_id: int
      runtime_ticks: int
      completed_legs: int
      attachment_edges: int
      release_edges: int
      initial_target_id: int
      initial_target_generation: int
      final_target_generation: int

  def validate_table_rack_evidence(
      jsonl_path: Path,
      png_path: Path,
      headless_path: Path,
      pack_path: Path,
  ) -> TableRackPlayableSummary:
      records = _read_canonical_jsonl(jsonl_path)
      headless = _read_canonical_headless_document(headless_path)
      actual_pack = _read_and_hash_pack_identity(pack_path)
      summary = _validate_table_rack_records(
          records, headless["pack_identity"], actual_pack)
      _validate_table_rack_png(png_path)
      return summary
  ```

  Corruption tests must independently reject a missing/extra/reordered JSONL
  record, missing/extra key, duplicate/skipped runtime tick or render frame,
  nonzero scheduler phase, wrong 25 Hz/dt identity, early route advancement,
  wrong surface ID or generation, affordance/source kind/source ID mutation,
  requested/applied correction mutation, attachment/release edge mutation,
  target ID/generation/owner mutation, false success summary, premature/max-tick
  exit, truncated/wrong-size/blank PNG, JSONL/PNG/headless path aliasing, any of
  the seven graphical/headless pack fields differing, and any pack artifact
  hash/count/rate differing from both evidence documents.

- [ ] **Step 2: Run RED**

  ```bash
  python -m unittest tests.python.test_playable_table_rack_evidence -v
  ```

  Expected: import/source-contract failure because the third mode, validator,
  and evidence schema do not exist.

- [ ] **Step 3: Extend the existing environment parser and auto-exit state**

  Extend the current types, without adding another `main` or argument parser:

  ```cpp
  enum class AutodemoMode {
      Pickup,
      Placement,
      TableRack,
  };

  struct AutodemoConfiguration {
      AutodemoMode mode = AutodemoMode::Pickup;
      uint32_t table_rack_row_id = 0U;
      // Preserve the existing log/screenshot final, temporary, and backup
      // paths and preserve_backups_after_rollback_failure field unchanged.
  };

  constexpr uint32_t kTableRackAutodemoMaximumTicks = 2400U;
  constexpr uint32_t kTableRackAutodemoSettledTicks = 3U;
  static_assert(kTableRackAutodemoMaximumTicks ==
                locomotion_timing::ticks_for_milliseconds(96000U));
  static_assert(kTableRackAutodemoSettledTicks ==
                locomotion_timing::ticks_for_milliseconds(120U));
  ```

  `parse_autodemo_environment()` reads
  `MM_INTERACTION_TABLE_RACK_AUTODEMO` beside the two existing selectors,
  counts non-null selectors, and requires exactly one when autodemo is enabled.
  Parse the row with `std::from_chars`, require complete consumption and the
  canonical decimal spelling `std::to_string(parsed) == input`, then validate
  membership in `TableRackSceneDefinition::positive_rows` after the scene is
  built. Continue using existing parent-directory, path-distinctness,
  temporary/backup, SIGTERM, and atomic publication code.

  Add `table_rack_autodemo_enabled` and a
  `ControllerTableRackAutodemoState` that contains the selected row/route,
  `runtime_tick`, `render_frame`, prior attachment state, cumulative attach and
  release edges, settled ticks, completion/failure flags, and the frozen initial
  target/support identities. It drives exactly one visible
  table→rack→table round trip: two real pickups and two real placements.

  Success is recognized only after the second exact support commit, Locomotion,
  Free/unattached owner zero, target ID unchanged, generation `g+2`, two attach
  edges, two release edges, and three consecutive frames with no interaction
  pose override. Then write the summary, flush/close JSONL, take and validate the
  PNG, call existing `publish_autodemo_evidence`, set `complete=true` and
  `exit_requested=true`, and return exit code `0` through the existing main-loop
  epilogue.

  Any runtime Rejected/Cancelled/Failed/Reset result after the first action,
  identity mismatch, invalid route transition, or attempt to execute tick 2401
  sets failure and exits `1`. Window close or SIGTERM before publication also
  exits `1`. On failure, close the temporary log and use existing cleanup so no
  partial final JSONL/PNG replaces prior evidence. The external timeout remains
  a hang backstop only; it is not the success/failure oracle.

- [ ] **Step 4: Integrate exact highlighted placement and manual controls**

  Load and retain the verified owning bundle before runtime construction:

  ```cpp
  std::optional<CertifiedPickupSourceBundle> table_rack_bundle;
  table_rack_bundle.emplace(load_certified_smart_pickup_source_bundle(
      resolved_interaction_pack_directory));
  const TableRackSceneDefinition table_rack =
      make_table_rack_scene(table_rack_bundle->diagnostics);
  for (const PlacementSurface& surface : table_rack.surfaces) {
      interaction_surface_registry.upsert(surface);
  }
  InteractionRuntime interaction_runtime(
      table_rack_bundle->database,
      table_rack_bundle->features,
      interaction_target_registry,
      interaction_surface_registry,
      table_rack_bundle->place_library,
      interaction_runtime_config,
      &table_rack_bundle->pickup_sources);
  ```

  Declare the optional bundle before the optional/runtime storage so C++
  destruction order destroys the runtime first. Once the registry pointer is
  passed, never move/re-emplace/reset the bundle. Table/rack autodemo fails
  before opening the window if seven-field verification fails. Manual
  table/rack mode uses the same bundle; no caller constructs a place library or
  certified registry independently.

  Add `interaction_smart_pickup_place_library.cpp` to
  `CONTROLLER_SMART_PICKUP_SOURCES` and add its header, the generated include,
  and `interaction_pickup_provenance.h` to the controller target's explicit
  prerequisites. `interaction_pickup_provenance.cpp` arrives exactly once via
  `INTERACTION_SOURCES` from Task 1.

  When F is pressed in Carry, latch exactly:

  ```cpp
  const SemanticPlacementSlot& goal = table_rack_route.highlighted();
  ControllerPlaceTarget place_target{
      goal.surface,
      goal.affordance_id,
      interaction_next_request_id++,
  };
  ```

  X cancels without route advancement. R resets the target, runtime attempt, and
  route. WASD remains available in Locomotion/Carry. Draw the rack as three open
  shelves at the exact support heights and visibly highlight only `goal`.

- [ ] **Step 5: Implement canonical JSONL/PNG evidence and its validator**

  The successful JSONL contains exactly one header, contiguous tick records,
  and exactly one final summary. Every line is one ASCII JSON object with keys
  sorted lexicographically, compact separators, no NaN/Infinity, and exactly one
  trailing LF. Float authority is a lowercase eight-digit float32 hex string;
  exact `0.04F` is `"0x3d23d70a"`.

  Header exact keys are:

  ```python
  _HEADER_KEYS = {
      "dt_f32_hex", "initial_target_generation", "initial_target_id",
      "max_ticks", "pack_identity", "record_type", "row_id",
      "schema_version", "updates_hz",
  }
  ```

  with `record_type == "table_rack_header"`, `schema_version == 1`,
  `updates_hz == 25`, `dt_f32_hex == "0x3d23d70a"`, and
  `max_ticks == 2400`. `pack_identity` is an object with exactly these seven
  keys and types, copied from `SmartPickupPlaceLibraryDiagnostics::pack`:

  ```python
  _PACK_IDENTITY_KEYS = {
      "clip_count", "database_sha256", "features_sha256",
      "fps_denominator", "fps_numerator", "frame_count", "manifest_sha256",
  }
  ```

  The three hashes are 64-character lowercase hex strings; the other fields
  are nonnegative JSON integers and rate is exactly 25/1. The validator requires
  object equality with Task 6's headless `pack_identity` and independently
  hashes `interaction_database.bin`, `interaction_features.bin`, and
  `manifest.json` in the supplied pack directory, then derives and compares
  manifest clip/frame/rate counts. A compact surrogate hash is not accepted.

  Tick exact keys are:

  ```python
  _TICK_KEYS = {
      "action", "applied_vertical_correction_f32_hex", "attached",
      "attachment_edges", "completed_legs", "destination_affordance_id",
      "destination_slot_id", "destination_surface_generation",
      "destination_surface_id", "dt_f32_hex", "leg_index", "object_state",
      "owner_request", "owns_pose", "place_source_id", "place_source_kind",
      "reason", "record_type", "release_edges", "render_frame", "result",
      "round_trips", "row_id", "runtime_state", "runtime_tick",
      "scheduler_phase", "schema_version", "source_affordance_id",
      "source_slot_id", "source_support_height_f32_hex",
      "source_surface_generation", "source_surface_id", "target_generation",
      "target_id", "target_support_height_f32_hex",
      "requested_support_height_f32_hex",
      "requested_vertical_correction_f32_hex", "updates_hz",
  }
  ```

  Before a Place candidate exists, place source IDs and all correction/height
  hex strings use zero/`"none"` sentinels; once frozen they may never change for
  that leg. Tick records require `record_type == "table_rack_tick"`, contiguous
  `runtime_tick` and `render_frame` increments of one, `scheduler_phase == 0`,
  exact 25/`0.04F`, monotonic binary edge counters, and route advancement only
  on an exact Locomotion/Free support commit.

  Summary exact keys are:

  ```python
  _SUMMARY_KEYS = {
      "attachment_edges", "completed_legs", "exit_reason",
      "final_target_generation", "final_target_id",
      "initial_target_generation", "initial_target_id", "record_type",
      "release_edges", "render_frames", "round_trips", "row_id",
      "runtime_ticks", "schema_version", "success",
  }
  ```

  It requires `record_type == "table_rack_summary"`, `success is True`,
  `exit_reason == "completed"`, `completed_legs == 2`, `round_trips == 1`,
  two attachment/release edges, stable target ID, generation `g+2`, and
  `runtime_ticks == render_frames <= 2400`.

  `validate_table_rack_evidence` rejects noncanonical raw JSONL bytes, validates
  every lifecycle/identity transition, then validates PNG signature/IHDR,
  exactly `1280x720`, file size above 10,000 bytes, and nonblank decoded RGB
  content using the existing placement-evidence image rule. Its CLI is exactly:

  ```bash
  python resources/validate_table_rack_evidence.py \
    --jsonl /absolute/path/table-rack.jsonl \
    --png /absolute/path/table-rack.png \
    --headless /absolute/path/full-chain-normal.json \
    --pack /absolute/path/full-pack
  ```

- [ ] **Step 6: Run graphical/headless regression and record evidence**

  ```bash
  make gate-smart-pickup-plan-a-headless \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  make controller bootstrap-raylib \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  evidence=/home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-a
  mkdir -p "$evidence"
  DISPLAY="${DISPLAY:-:1}" \
  MM_INTERACTION_TABLE_RACK_AUTODEMO=1 \
  MM_INTERACTION_TABLE_RACK_ROW_ID=8 \
  MM_INTERACTION_PACK="$PWD/build/smart-pickup/full-pack" \
  MM_INTERACTION_LOG="$evidence/table-rack.jsonl" \
  MM_INTERACTION_SCREENSHOT="$evidence/table-rack.png" \
  MM_FEATURES_OUTPUT="$evidence/locomotion-features.bin" \
    timeout --signal=TERM --kill-after=5s 110s ./controller
  python resources/validate_table_rack_evidence.py \
    --jsonl "$evidence/table-rack.jsonl" \
    --png "$evidence/table-rack.png" \
    --headless "$PWD/build/table-rack/evidence/full-chain-normal.json" \
    --pack "$PWD/build/smart-pickup/full-pack"
  TABLE_RACK_PLAYABLE_LOG="$evidence/table-rack.jsonl" \
  TABLE_RACK_PLAYABLE_PNG="$evidence/table-rack.png" \
  TABLE_RACK_HEADLESS_EVIDENCE="$PWD/build/table-rack/evidence/full-chain-normal.json" \
  TABLE_RACK_PACK="$PWD/build/smart-pickup/full-pack" \
    python -m unittest tests.python.test_playable_table_rack_evidence -v
  ```

  This uses environment variables because `controller` is `main(void)`. Expected
  exit is `0`; timeout `124`, SIGTERM/window close, internal max tick, or any
  evidence/interaction failure is nonzero.

- [ ] **Step 7: Add the executable graphical and completion gates**

  Add these exact Make variables/targets:

  ```make
  TABLE_RACK_PLAYABLE_EVIDENCE_DIR ?= \
    /home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-a
  TABLE_RACK_PLAYABLE_LOG ?= \
    $(TABLE_RACK_PLAYABLE_EVIDENCE_DIR)/table-rack.jsonl
  TABLE_RACK_PLAYABLE_PNG ?= \
    $(TABLE_RACK_PLAYABLE_EVIDENCE_DIR)/table-rack.png
  TABLE_RACK_PLAYABLE_FEATURES ?= \
    $(TABLE_RACK_PLAYABLE_EVIDENCE_DIR)/locomotion-features.bin
  TABLE_RACK_HEADLESS_EVIDENCE ?= \
    build/table-rack/evidence/full-chain-normal.json
  TABLE_RACK_PLAYABLE_ROW_ID ?= 8

  .PHONY: gate-playable-table-rack
  gate-playable-table-rack:
	$(MAKE) gate-smart-pickup-plan-a-headless \
	  INTERACTION_DEMO_PACK="$(INTERACTION_DEMO_PACK)"
	$(MAKE) bootstrap-raylib \
	  INTERACTION_DEMO_PACK="$(INTERACTION_DEMO_PACK)"
	$(MAKE) controller INTERACTION_DEMO_PACK="$(INTERACTION_DEMO_PACK)"
	mkdir -p "$(TABLE_RACK_PLAYABLE_EVIDENCE_DIR)"
	@display="$${DISPLAY:-:1}"; \
	  timeout --signal=TERM --kill-after=1s 5s \
	    xdpyinfo -display "$$display" >/dev/null
	@set -eu; \
	  features_before="$$(sha256sum resources/features.bin)"; \
	  database_before="$$(sha256sum \
	    "$(INTERACTION_DEMO_PACK)/interaction_database.bin")"; \
	  interaction_features_before="$$(sha256sum \
	    "$(INTERACTION_DEMO_PACK)/interaction_features.bin")"; \
	  manifest_before="$$(sha256sum \
	    "$(INTERACTION_DEMO_PACK)/manifest.json")"; \
	  display="$${DISPLAY:-:1}"; \
	  DISPLAY="$$display" \
	  MM_INTERACTION_TABLE_RACK_AUTODEMO=1 \
	  MM_INTERACTION_TABLE_RACK_ROW_ID="$(TABLE_RACK_PLAYABLE_ROW_ID)" \
	  MM_INTERACTION_PACK="$(INTERACTION_DEMO_PACK)" \
	  MM_INTERACTION_LOG="$(TABLE_RACK_PLAYABLE_LOG)" \
	  MM_INTERACTION_SCREENSHOT="$(TABLE_RACK_PLAYABLE_PNG)" \
	  MM_FEATURES_OUTPUT="$(TABLE_RACK_PLAYABLE_FEATURES)" \
	    timeout --signal=TERM --kill-after=5s 110s ./controller; \
	  features_after="$$(sha256sum resources/features.bin)"; \
	  database_after="$$(sha256sum \
	    "$(INTERACTION_DEMO_PACK)/interaction_database.bin")"; \
	  interaction_features_after="$$(sha256sum \
	    "$(INTERACTION_DEMO_PACK)/interaction_features.bin")"; \
	  manifest_after="$$(sha256sum \
	    "$(INTERACTION_DEMO_PACK)/manifest.json")"; \
	  test "$$features_before" = "$$features_after" || { \
	    echo "ERROR resources/features.bin changed during table/rack gate" >&2; \
	    exit 1; \
	  }; \
	  test "$$database_before" = "$$database_after" || { \
	    echo "ERROR interaction_database.bin changed during table/rack gate" >&2; \
	    exit 1; \
	  }; \
	  test "$$interaction_features_before" = \
	       "$$interaction_features_after" || { \
	    echo "ERROR interaction_features.bin changed during table/rack gate" >&2; \
	    exit 1; \
	  }; \
	  test "$$manifest_before" = "$$manifest_after" || { \
	    echo "ERROR manifest.json changed during table/rack gate" >&2; \
	    exit 1; \
	  }
	python resources/validate_table_rack_evidence.py \
	  --jsonl "$(TABLE_RACK_PLAYABLE_LOG)" \
	  --png "$(TABLE_RACK_PLAYABLE_PNG)" \
	  --headless "$(TABLE_RACK_HEADLESS_EVIDENCE)" \
	  --pack "$(INTERACTION_DEMO_PACK)"
	TABLE_RACK_PLAYABLE_LOG="$(TABLE_RACK_PLAYABLE_LOG)" \
	TABLE_RACK_PLAYABLE_PNG="$(TABLE_RACK_PLAYABLE_PNG)" \
	TABLE_RACK_HEADLESS_EVIDENCE="$(TABLE_RACK_HEADLESS_EVIDENCE)" \
	TABLE_RACK_PACK="$(INTERACTION_DEMO_PACK)" \
	  python -m unittest tests.python.test_playable_table_rack_evidence -v

  .PHONY: gate-smart-pickup-plan-a
  gate-smart-pickup-plan-a: gate-playable-table-rack
  ```

  The graphical target consumes the supplied full pack read-only. It must not
  depend on or invoke any pack-building/rebuilding target, and must not invoke
  or mention the protected
  `interaction_query_probe`.

- [ ] **Step 8: Run final Plan A verification**

  ```bash
  make gate-smart-pickup-plan-a \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  python -m unittest tests.python.test_playable_table_rack_evidence -v
  make build/tests/test_interaction_place \
       build/tests/test_interaction_place_controller \
       build/tests/test_interaction_runtime \
       build/tests/test_interaction_table_rack_scene \
       build/tests/test_interaction_table_rack_route \
       INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  build/tests/test_interaction_place
  build/tests/test_interaction_place_controller
  build/tests/test_interaction_runtime
  build/tests/test_interaction_table_rack_scene
  build/tests/test_interaction_table_rack_route
  ```

- [ ] **Step 9: Review, commit, and push**

  ```bash
  git add controller.cpp interaction_debug_draw.h \
    resources/validate_table_rack_evidence.py \
    tests/python/test_playable_table_rack_evidence.py Makefile
  git diff --cached --check
  git commit -m "feat: expose repeated table rack interactions"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

## Plan A Completion Gate

Plan A is complete only when
`make gate-smart-pickup-plan-a INTERACTION_DEMO_PACK=build/smart-pickup/full-pack`
returns zero. That target depends on Task 6's byte-identical normal/release
full-chain matrix and Task 7's environment-driven graphical JSONL/PNG validator.
The graphical evidence must show real attachment, Carry, truthful placement
source, release, re-pick, exact support/generation route advancement, and exact
25 Hz ticks. Animation-only Reach, request submission, Align, returning to
Locomotion, a timeout, or a partial temporary artifact is not completion. Only
then may the learned-funnel implementation plan begin.
