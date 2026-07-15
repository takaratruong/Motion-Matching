# Ownership-Reference Retarget Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the visible Align and layered-Carry skeleton discontinuities by retargeting G1 motion as calibrated world-rotation deltas onto one captured flat skeleton and by limiting layered Carry to upper-body authority.

**Architecture:** On the first runtime-owned render, capture the raw 31-joint G1 pose and the pose currently displayed by the 23-joint controller. The 31→23 retarget then applies mapped world-rotation and explicit root deltas relative to those references while retaining the captured flat rig's non-root local translations, including stable Neck and Head locals. Full-body interaction states publish this continuous target directly with quaternion hemisphere continuity; layered Carry composes only flat upper-body channels over the fresh 60 Hz locomotion root/lower body.

**Tech Stack:** C++17 controller/interaction runtime, existing `vec3`/`quat` FK utilities, Make-based C++ tests, Python `unittest` graphical-evidence validator, Raylib/X11 autoplay gate.

## Global Constraints

- Begin from clean branch head `f42268579cc3fc8fc94fcdf16a8887c0f6fef59b` in `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching`.
- Never access, stat, hash, execute, modify, stage, or delete the protected repo-root untracked `interaction_query_probe` binary. Only `build/task12/interaction_query_probe_safe` and the tracked `interaction_query_probe.cpp` source are permitted.
- Never work in or modify the terrain checkout `/home/ubuntu/projects/motion-matching`; all edits and commands use the isolated worktree.
- Write each production correction only after its focused test has been observed failing for the expected behavioral reason.
- Keep runtime decisions at exactly 25 Hz and controller presentation at exactly 60 Hz.
- Do not change final foot IK, the unified renderer skeleton, interaction database formats, or the strict adjacent-joint limits of `0.20 m` and `60.0 degrees`.
- Do not add Head/Neck-only suppression or any continuity exemption.
- Full-body ownership applies to Align, PickupReplay, Hold, and recorded Carry. Layered Carry means exactly `RuntimeState::Carry && !diagnostics.recorded_carry`.
- In layered Carry, bones `0..9` and both foot contacts are bit-equal to the fresh 60 Hz locomotion pose; interaction owns bones `10..22`, with Neck and Head retaining the captured target-reference locals.
- Raw-reference publication is exactly the captured displayed flat reference, and each fresh ownership epoch captures new raw and flat references.
- The strict graphical gate must validate final post-foot-IK/FK rendered joints; focused C++ retarget tests isolate the pre-foot-IK adapter output.

---

### Task 1: Replace Dynamic Anchor Collapse with Ownership-Reference Delta Retargeting

**Files:**
- Modify: `interaction_controller_adapter.h`
- Modify: `interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`

**Interfaces:**
- Consumes: `Pose`, `FlatControllerPose`, `kFlatControllerAnchors`, `world_pose`, and the existing flat FK helpers.
- Produces: `FlatControllerPose collapse_interaction_pose(const Pose& interaction_pose, const Pose& interaction_reference, const FlatControllerPose& flat_reference)`.
- Invariant: when `interaction_pose` equals `interaction_reference`, the result is exactly `flat_reference`; otherwise every non-root `positions[bone]` remains bit-equal to `flat_reference.positions[bone]`.

- [ ] **Step 1: Replace the old exact-anchor tests with RED reference-retarget tests**

  Remove the collapse round-trip assertion that requires mapped G1 world positions to overwrite flat local translations. Retain input-validation and 23-element canary coverage after updating calls to the three-argument API. Add focused tests with these bodies:

  ```cpp
  void test_reference_retarget_raw_reference_is_exact_flat_reference() {
      const Pose raw_reference = make_pose(0.5F);
      const FlatControllerPose flat_reference = make_flat_pose();
      const FlatControllerPose result = interaction::collapse_interaction_pose(
          raw_reference, raw_reference, flat_reference);
      require(
          flat_pose_bits_equal(result, flat_reference),
          "raw ownership reference did not reproduce displayed flat reference");
  }

  void test_reference_retarget_fixes_nonroot_translations_under_mismatched_source() {
      const Pose raw_reference = make_pose(0.25F);
      Pose current = raw_reference;
      for (size_t bone = 1; bone < g1_skeleton::BoneCount; ++bone) {
          const float value = static_cast<float>(bone + 1U);
          current.positions[bone] = vec3(
              10.0F * value, -7.0F * value, 13.0F * value);
      }
      const FlatControllerPose flat_reference = make_flat_pose();
      const FlatControllerPose result = interaction::collapse_interaction_pose(
          current, raw_reference, flat_reference);
      for (size_t bone = 1; bone < interaction::kFlatControllerBoneCount;
           ++bone) {
          require(
              vec_bits_equal(
                  result.positions[bone], flat_reference.positions[bone]),
              "source morphology changed a target non-root translation");
      }
  }

  void test_reference_retarget_transfers_mapped_world_rotation_delta() {
      constexpr size_t kFlatSpineUpper = 12U;
      constexpr size_t kG1SpineUpper = 16U;
      const Pose raw_reference = make_pose(0.75F);
      Pose current = raw_reference;
      current.rotations[kG1SpineUpper] = quat_mul(
          quat_from_angle_axis(
              0.35F, normalize(vec3(0.2F, 0.9F, 0.3F))),
          current.rotations[kG1SpineUpper]);
      const FlatControllerPose flat_reference = make_flat_pose();
      const FlatControllerPose result = interaction::collapse_interaction_pose(
          current, raw_reference, flat_reference);
      const interaction::WorldPose source_reference_world =
          interaction::world_pose(raw_reference);
      const interaction::WorldPose source_current_world =
          interaction::world_pose(current);
      const FlatWorldPose target_reference_world =
          flat_world_pose(flat_reference);
      const FlatWorldPose target_current_world = flat_world_pose(result);
      const quat source_world_delta = quat_mul(
          source_current_world.rotations[kG1SpineUpper],
          quat_inv(source_reference_world.rotations[kG1SpineUpper]));
      const quat expected = quat_mul(
          source_world_delta,
          target_reference_world.rotations[kFlatSpineUpper]);
      require_same_rotation(
          target_current_world.rotations[kFlatSpineUpper],
          expected,
          "mapped world-rotation delta was not calibrated onto flat reference");
  }

  void test_reference_retarget_keeps_neck_and_head_reference_locals() {
      const Pose raw_reference = make_pose(1.0F);
      Pose current = raw_reference;
      current.rotations[16] = quat_mul(
          quat_from_angle_axis(0.08F, vec3(0.0F, 1.0F, 0.0F)),
          current.rotations[16]);
      const FlatControllerPose flat_reference = make_flat_pose();
      const FlatControllerPose result = interaction::collapse_interaction_pose(
          current, raw_reference, flat_reference);
      for (size_t bone : {13U, 14U}) {
          require(vec_bits_equal(result.positions[bone],
                                 flat_reference.positions[bone]),
                  "Neck/Head local translation left target reference");
          require(quat_bits_equal(result.rotations[bone],
                                  flat_reference.rotations[bone]),
                  "Neck/Head local rotation left target reference");
      }
      const FlatWorldPose before = flat_world_pose(flat_reference);
      const FlatWorldPose after = flat_world_pose(result);
      require(rotation_distance(before.rotations[13], after.rotations[13]) <
                  0.10F,
              "Neck world motion exceeded its continuous parent delta");
      require(rotation_distance(before.rotations[14], after.rotations[14]) <
                  0.10F,
              "Head world motion exceeded its continuous parent delta");
  }
  ```

- [ ] **Step 2: Run the adapter test and record the expected RED**

  Run:

  ```bash
  make -B build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_controller_adapter
  ```

  Expected: compilation fails because `collapse_interaction_pose` still accepts two arguments, proving the tests require the new reference contract. After declaring the signature without implementing behavior, the fixed-translation and raw-identity assertions must fail against the old exact-anchor collapse.

- [ ] **Step 3: Declare and implement the reference-delta collapse**

  Change the header declaration to:

  ```cpp
  FlatControllerPose collapse_interaction_pose(
      const Pose& interaction_pose,
      const Pose& interaction_reference,
      const FlatControllerPose& flat_reference);
  ```

  In `interaction_controller_adapter.cpp`, validate all three poses, return `flat_reference` when every raw skeletal channel/contact is equal to its captured raw reference, and otherwise construct the result topologically:

  ```cpp
  const WorldPose source_world = world_pose(interaction_pose);
  const WorldPose source_reference_world = world_pose(interaction_reference);
  const FlatWorldPose flat_reference_world = flat_world_pose(flat_reference);
  FlatControllerPose retargeted = flat_reference;
  FlatWorldPose retargeted_world{};

  for (size_t flat_bone = 0;
       flat_bone < kFlatControllerBoneCount;
       ++flat_bone) {
      const int32_t source_value = kFlatToG1Bone[flat_bone];
      if (source_value >= 0) {
          const size_t source_bone = static_cast<size_t>(source_value);
          const quat world_delta = normalized_rotation(quat_mul(
              source_world.rotations[source_bone],
              quat_inv(source_reference_world.rotations[source_bone])));
          const quat desired_world_rotation = normalized_rotation(quat_mul(
              world_delta,
              flat_reference_world.rotations[flat_bone]));
          const vec3 desired_world_angular_velocity =
              flat_reference_world.angular_velocities[flat_bone] +
              source_world.angular_velocities[source_bone] -
              source_reference_world.angular_velocities[source_bone];

          const int32_t parent = kFlatControllerParents[flat_bone];
          if (parent < 0) {
              retargeted.positions[flat_bone] =
                  flat_reference.positions[flat_bone] +
                  source_world.positions[source_bone] -
                  source_reference_world.positions[source_bone];
              retargeted.velocities[flat_bone] =
                  flat_reference.velocities[flat_bone] +
                  source_world.velocities[source_bone] -
                  source_reference_world.velocities[source_bone];
              retargeted.rotations[flat_bone] = desired_world_rotation;
              retargeted.angular_velocities[flat_bone] =
                  desired_world_angular_velocity;
          } else {
              const size_t parent_bone = static_cast<size_t>(parent);
              retargeted.positions[flat_bone] =
                  flat_reference.positions[flat_bone];
              retargeted.velocities[flat_bone] =
                  flat_reference.velocities[flat_bone];
              retargeted.rotations[flat_bone] = normalized_rotation(
                  quat_inv_mul(
                      retargeted_world.rotations[parent_bone],
                      desired_world_rotation));
              retargeted.angular_velocities[flat_bone] = quat_inv_mul_vec3(
                  retargeted_world.rotations[parent_bone],
                  desired_world_angular_velocity -
                      retargeted_world.angular_velocities[parent_bone]);
          }
      }
      update_flat_world_bone(retargeted_world, retargeted, flat_bone);
  }
  retargeted.foot_contacts = interaction_pose.foot_contacts;
  ```

  Leave flat bones `13` and `14` untouched from `flat_reference`; their world transforms change only through flat bone `12`. Do not solve any non-root local position from a source world anchor.

- [ ] **Step 4: Run the focused test GREEN and inspect warnings**

  Run:

  ```bash
  make -B build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_controller_adapter
  ```

  Expected: exit `0` with no compiler warnings. The new raw identity, fixed-translation, mapped-delta, Neck/Head, invalid-input, antipodal-equivalence, and canary tests pass.

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add interaction_controller_adapter.h interaction_controller_adapter.cpp tests/cpp/test_interaction_controller_adapter.cpp
  git commit -m "fix: retarget interaction deltas onto flat reference"
  ```

---

### Task 2: Publish Reference-Continuous Ownership and Mask Layered Carry

**Files:**
- Modify: `interaction_controller_adapter.h`
- Modify: `interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`

**Interfaces:**
- Consumes: the Task 1 three-reference `collapse_interaction_pose` API.
- Produces: an ownership epoch containing `ownership_interaction_reference_`, `ownership_flat_reference_`, and `last_rendered_pose_`; release continues using `blend_source_` and `blend_seconds_`.
- Authority: full-body owned output for pre-Carry/recorded Carry; bones `10..22` only for layered Carry.

- [ ] **Step 1: Write RED handoff tests for entry identity, branch continuity, re-entry, and layered authority**

  Replace `test_frame_handoff_retargets_before_blending_in_flat_topology` and the fixed-source interpolation expectation. Update `make_owned_output` so callers provide a raw `Pose` directly. Add:

  ```cpp
  void test_frame_handoff_first_owned_frame_is_exact_displayed_reference() {
      const FlatControllerPose entry = make_flat_pose();
      RuntimeOutput owned;
      owned.owns_pose = true;
      owned.pose = make_pose(0.375F);
      owned.diagnostics.state = RuntimeState::Align;
      ControllerInteractionFrameHandoff handoff;
      const ControllerInteractionFrameState frame = handoff.apply(
          entry, owned, interaction::kControllerStepSeconds);
      require(flat_pose_bits_equal(frame.pose, entry),
              "first ownership frame did not preserve displayed entry pose");
  }

  void test_frame_handoff_crosses_179_9_to_180_1_incrementally() {
      FlatControllerPose entry = make_flat_pose();
      entry.rotations[0] = quat();
      RuntimeOutput owned;
      owned.owns_pose = true;
      owned.pose = make_pose(0.0F);
      owned.pose.rotations[0] = quat();
      owned.diagnostics.state = RuntimeState::Align;
      ControllerInteractionFrameHandoff handoff;
      ControllerInteractionFrameState previous = handoff.apply(
          entry, owned, interaction::kControllerStepSeconds);
      for (float degrees :
           {30.0F, 60.0F, 90.0F, 120.0F, 150.0F, 179.9F, 180.1F}) {
          owned.pose.rotations[0] = quat_from_angle_axis(
              degrees * 3.14159265358979323846F / 180.0F,
              vec3(0.0F, 1.0F, 0.0F));
          const ControllerInteractionFrameState frame = handoff.apply(
              entry, owned, interaction::kControllerStepSeconds);
          require(rotation_distance(
                      previous.pose.rotations[0], frame.pose.rotations[0]) <
                      31.0F * 3.14159265358979323846F / 180.0F,
                  "moving target switched the old fixed-source branch");
          previous = frame;
      }
  }

  void test_layered_carry_keeps_fresh_lower_body_and_contacts_bit_exact() {
      FlatControllerPose entry = make_flat_pose();
      RuntimeOutput output;
      output.owns_pose = true;
      output.pose = make_pose(0.0F);
      output.diagnostics.state = RuntimeState::Align;
      ControllerInteractionFrameHandoff handoff;
      (void)handoff.apply(entry, output, interaction::kControllerStepSeconds);

      FlatControllerPose locomotion = entry;
      for (size_t bone = 0; bone <= 9U; ++bone) {
          const float value = static_cast<float>(bone + 1U);
          locomotion.positions[bone].x += 0.03F * value;
          locomotion.velocities[bone].z -= 0.04F * value;
          locomotion.rotations[bone] = quat_from_angle_axis(
              0.01F * value, vec3(0.0F, 1.0F, 0.0F));
      }
      locomotion.foot_contacts = {0U, 1U};
      output.diagnostics.state = RuntimeState::Carry;
      output.diagnostics.recorded_carry = false;
      output.pose.positions[8] = vec3(900.0F, -400.0F, 700.0F);
      output.pose.rotations[8] = quat_from_angle_axis(
          3.08F, vec3(1.0F, 0.0F, 0.0F));
      output.pose.rotations[16] = quat_from_angle_axis(
          0.45F, vec3(0.0F, 0.0F, 1.0F));

      const ControllerInteractionFrameState frame = handoff.apply(
          locomotion, output, interaction::kControllerStepSeconds);
      for (size_t bone = 0; bone <= 9U; ++bone) {
          require(vec_bits_equal(frame.pose.positions[bone],
                                 locomotion.positions[bone]) &&
                      vec_bits_equal(frame.pose.velocities[bone],
                                     locomotion.velocities[bone]) &&
                      quat_bits_equal(frame.pose.rotations[bone],
                                      locomotion.rotations[bone]) &&
                      vec_bits_equal(frame.pose.angular_velocities[bone],
                                     locomotion.angular_velocities[bone]),
                  "layered Carry changed fresh locomotion lower body");
      }
      require(frame.pose.foot_contacts == locomotion.foot_contacts,
              "layered Carry changed fresh locomotion contacts");
      require(!quat_bits_equal(frame.pose.rotations[12],
                               locomotion.rotations[12]),
              "layered Carry failed to own the upper body");
  }
  ```

  Extend the existing reset/re-entry test to change the entire new flat entry and raw reference, then require the first reacquired frame to be bit-equal to that fresh entry. Extend the root-sync test to prove Align and recorded Carry continue publishing retargeted full-body root/legs rather than the live layered base.

- [ ] **Step 2: Run the focused test and capture the expected RED**

  Run:

  ```bash
  make -B build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_controller_adapter
  ```

  Expected: abort at first-entry identity because the existing handoff advances a `0.25 s` fixed-source blend; after that assertion is isolated, the layered lower-body test fails because the cached raw Carry pose still owns bones `0..9` and contacts.

- [ ] **Step 3: Replace entry blending with ownership-reference publication**

  Replace `ownership_fallback_` in the class with:

  ```cpp
  Pose ownership_interaction_reference_{};
  FlatControllerPose ownership_flat_reference_{};
  ```

  On `!runtime_owned_last_update_`, capture the raw runtime pose and the actually displayed pose:

  ```cpp
  ownership_interaction_reference_ = runtime_output.pose;
  ownership_flat_reference_ =
      release_active_ ? last_rendered_pose_ : locomotion_pose;
  release_active_ = false;
  runtime_owned_last_update_ = true;
  state.pose = ownership_flat_reference_;
  ```

  On subsequent owned frames, call:

  ```cpp
  FlatControllerPose target = collapse_interaction_pose(
      runtime_output.pose,
      ownership_interaction_reference_,
      ownership_flat_reference_);
  for (size_t bone = 0; bone < target.rotations.size(); ++bone) {
      if (quat_dot(last_rendered_pose_.rotations[bone],
                   target.rotations[bone]) < 0.0F) {
          target.rotations[bone] = -target.rotations[bone];
      }
  }
  ```

  Publish `target` directly for full-body states. Do not compute an ownership-entry `alpha`, do not interpolate each moving target against a fixed `blend_source_`, and do not add a completion snap. Keep `blend_source_`, `blend_seconds_`, and `blend_flat_pose` only for the existing release override.

- [ ] **Step 4: Compose layered Carry over fresh 60 Hz locomotion**

  Detect layered authority exactly once:

  ```cpp
  const bool layered_carry =
      runtime_output.diagnostics.state == RuntimeState::Carry &&
      !runtime_output.diagnostics.recorded_carry;
  if (layered_carry) {
      state.pose = locomotion_pose;
      for (size_t bone = 10U;
           bone < kFlatControllerBoneCount;
           ++bone) {
          state.pose.positions[bone] = target.positions[bone];
          state.pose.velocities[bone] = target.velocities[bone];
          state.pose.rotations[bone] = target.rotations[bone];
          state.pose.angular_velocities[bone] =
              target.angular_velocities[bone];
      }
      state.pose.foot_contacts = locomotion_pose.foot_contacts;
  } else {
      state.pose = target;
  }
  ```

  Keep `runtime_owns_pose=true` and `overrides_locomotion_pose=true` in layered Carry, but keep `synchronize_simulation_root=false`. Align, PickupReplay, Hold, and recorded Carry retain `synchronize_simulation_root=true` and full-body retarget authority.

- [ ] **Step 5: Preserve release and fresh-epoch reset semantics**

  Continue to capture `last_rendered_pose_` as the release source and return it on the first non-owned frame. Clear both ownership references in `reset()`. Ensure a fresh owned epoch overwrites them after reset or completed release.

- [ ] **Step 6: Run the focused adapter test GREEN**

  Run:

  ```bash
  make -B build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_controller_adapter
  ```

  Expected: exit `0`. The `179.9→180.1` target moves by approximately `0.2 degrees`, every layered lower-body channel/contact remains bit-equal across both held and changed raw 25 Hz samples, upper-body channels remain interaction-owned, and pre-Carry/recorded Carry remain full-body/root-synchronized.

- [ ] **Step 7: Commit Task 2**

  ```bash
  git add interaction_controller_adapter.h interaction_controller_adapter.cpp tests/cpp/test_interaction_controller_adapter.cpp
  git commit -m "fix: compose layered carry over live locomotion"
  ```

---

### Task 3: Correct Rendered-Joint Diagnostics and Preserve Trace-Layer Isolation

**Files:**
- Modify: `controller.cpp`
- Modify: `tests/python/test_playable_interaction_evidence.py`

**Interfaces:**
- Consumes: `character.h`'s exact 23-joint order and the existing post-final-FK `AutodemoRenderedJoints` capture.
- Produces: identical joint labels in C++ evidence writing and Python validation diagnostics.

- [ ] **Step 1: Write the failing exact-name policy test**

  Add:

  ```python
  def test_flat_joint_names_match_character_h_order(self):
      self.assertEqual(
          FLAT_JOINT_NAMES,
          (
              "Entity", "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot",
              "LeftToe", "RightUpLeg", "RightLeg", "RightFoot",
              "RightToe", "Spine", "Spine1", "Spine2", "Neck", "Head",
              "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
              "RightShoulder", "RightArm", "RightForeArm", "RightHand",
          ),
      )
  ```

  Extend `test_autodemo_samples_all_rendered_joints_after_final_fk_before_draw` to require one final rendered-joint capture call after `forward_kinematics_full` and before `BeginDrawing`, and require the JSON writer to serialize only that `AutodemoRenderedJoints` value. This keeps the graphical threshold explicitly post-foot-IK/FK while Task 1's C++ tests remain pre-foot-IK retarget isolation.

- [ ] **Step 2: Run the Python module and record the expected RED**

  Run:

  ```bash
  python -m unittest tests.python.test_playable_interaction_evidence -v
  ```

  Expected: only the new exact-name test fails first, reporting `Root` versus `Entity` at index `0`; the existing post-FK ordering policy remains green.

- [ ] **Step 3: Correct both exact joint-name tables**

  Replace the 23 entries of `kAutodemoFlatJointNames` in `controller.cpp` and `FLAT_JOINT_NAMES` in the Python validator with this exact order:

  ```text
  Entity, Hips, LeftUpLeg, LeftLeg, LeftFoot, LeftToe,
  RightUpLeg, RightLeg, RightFoot, RightToe,
  Spine, Spine1, Spine2, Neck, Head,
  LeftShoulder, LeftArm, LeftForeArm, LeftHand,
  RightShoulder, RightArm, RightForeArm, RightHand
  ```

  Do not alter array order, joint indices, evidence JSON keys, thresholds, or capture timing.

- [ ] **Step 4: Run the Python evidence module GREEN**

  Run:

  ```bash
  python -m unittest tests.python.test_playable_interaction_evidence -v
  ```

  Expected: all tests pass with one existing graphical-environment skip when no live evidence paths are configured.

- [ ] **Step 5: Commit Task 3**

  ```bash
  git add controller.cpp tests/python/test_playable_interaction_evidence.py
  git commit -m "test: correct rendered joint diagnostic names"
  ```

---

### Task 4: Verify Safe Suites and the Strict Graphical Continuity Gate

**Files:**
- Verify only: `interaction_controller_adapter.cpp`
- Verify only: `controller.cpp`
- Verify only: `tests/cpp/test_interaction_controller_adapter.cpp`
- Verify only: `tests/python/test_playable_interaction_evidence.py`
- Generate: `build/task12/demo-ownership-retarget-20260715-v1/`
- Generate: `playable-evidence/ownership-retarget-20260715-v1/pickup.jsonl`
- Generate: `playable-evidence/ownership-retarget-20260715-v1/pickup.png`

**Interfaces:**
- Consumes: all three implementation commits.
- Produces: fresh compiler/test output, strict post-FK evidence, and phase-by-phase maximum adjacent translation/rotation measurements.

- [ ] **Step 1: Run focused and release-fast-math builds**

  Run:

  ```bash
  make -B build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_controller_adapter
  make -B controller BUILD_MODE=RELEASE
  ```

  Expected: every command exits `0`; the controller compiles with `-O3 -ffast-math` and no new warnings.

- [ ] **Step 2: Run the complete safe interaction suite**

  Run only the safe target:

  ```bash
  make test-interaction-safe
  ```

  Expected: all Python tests, all C++ binaries, and both release-fast-math binaries pass. The build uses `build/task12/interaction_query_probe_safe` and never invokes the protected root binary.

- [ ] **Step 3: Run a unique strict graphical gate**

  Run:

  ```bash
  make gate-playable-interaction \
    INTERACTION_DEMO_PACK=build/task12/demo-ownership-retarget-20260715-v1 \
    PLAYABLE_EVIDENCE_DIR=playable-evidence/ownership-retarget-20260715-v1 \
    PLAYABLE_LOG_PATH=playable-evidence/ownership-retarget-20260715-v1/pickup.jsonl \
    PLAYABLE_SCREENSHOT_PATH=playable-evidence/ownership-retarget-20260715-v1/pickup.png \
    PLAYABLE_FEATURES_OUTPUT=playable-evidence/ownership-retarget-20260715-v1/locomotion-features.bin
  ```

  Expected: exit `0`; the unchanged validator accepts every adjacent final rendered joint at `<= 0.20 m` and `<= 60.0 degrees`, with no state, joint, or frame exemptions.

- [ ] **Step 4: Parse and report phase maxima**

  Parse the fresh JSONL into collapsed state phases and report, for each phase, the maximum adjacent world translation and shortest-arc rotation with exact frame pair, joint index, and corrected joint name. Explicitly report Align and Carry maxima. If the strict validator fails, identify whether the first violation already exists in the pre-foot-IK adapter output or is introduced after final foot IK/FK; make only an evidence-backed correction inside Tasks 1–2 and rerun this entire task.

- [ ] **Step 5: Run final status and diff checks**

  Run:

  ```bash
  git status --short --untracked-files=no
  git diff --check f42268579cc3fc8fc94fcdf16a8887c0f6fef59b..HEAD
  git log --oneline f42268579cc3fc8fc94fcdf16a8887c0f6fef59b..HEAD
  ```

  Expected: tracked status is empty, `git diff --check` exits `0`, and the plan plus three scoped implementation commits are present. Do not merge this branch.
