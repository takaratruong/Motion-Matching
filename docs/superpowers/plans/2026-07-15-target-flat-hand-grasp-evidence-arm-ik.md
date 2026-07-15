# Target-Flat Hand/Grasp Evidence and Arm IK Implementation Plan

> **Execution:** Use subagent-driven development, TDD for every behavior change, an independent exact-diff review per commit, and verification-before-completion.

**Goal:** Make the final rendered selected hand follow the scene-authoritative grasp through contact, Hold, and Carry without losing the now-proven skeleton continuity or release continuity.

**Worktree:** `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching`

**Protected artifact:** Never access, stat, hash, execute, rebuild, modify, stage, or delete the repo-root untracked `interaction_query_probe`. Only tracked source and `build/task12/interaction_query_probe_safe` through safe Make targets are permitted. Use `git status --short --untracked-files=no`.

## Task 1: Persist Authoritative Hand/Grasp Evidence

**Files:** `controller.cpp`, `tests/python/test_playable_interaction_evidence.py`

1. Add RED Python tests for the exact new field order, valid/invalid active-hand indices, bool-as-int rejection, finite vectors, unit/sign-equivalent quaternions, exact composition of `object_world * hand_in_object`, selected left/right hand lookup, and deterministic PickupReplay/Hold/Carry count/mean/max summaries.
2. Extend the source-policy test so all grasp inputs and the rendered hand are captured after final FK and before draw, and the writer serializes only the immutable capture.
3. Run `python -m unittest tests.python.test_playable_interaction_evidence -v`; record the schema/policy RED before production edits.
4. Add `grasp_evidence_valid`, `active_hand_joint`, object rotation, `hand_in_object`, and derived grasp transforms to the immutable capture and fixed-six JSON writer. Require exact runtime target/affordance/hand agreement in interaction-owned states.
5. Run the focused Python module GREEN, then `make test-interaction-safe`.
6. Commit as `test: trace rendered hand against selected grasp`; obtain independent review.
7. Run a unique fresh graphical gate in `playable-evidence/grasp-baseline-20260715-v1/`. Report per-state raw position/orientation count, mean, max, frame, and corrected joint name. Do not add IK or alignment thresholds in this task.

## Task 2: Publish the Runtime Reach/Contact Weight

**Files:** `interaction_runtime.h`, `interaction_runtime.cpp`, `tests/cpp/test_interaction_runtime.cpp`, and fast-math validation if required.

1. Add RED tests proving weight is finite/in-range, exactly zero at entry, monotonic across Reach, exactly one by Contact and throughout attached PickupReplay/Hold/Carry, stable on cached outputs, and reset outside ownership.
2. Publish `RuntimeDiagnostics::hand_constraint_weight` from the same candidate entry/contact progress used by reach correction. Do not derive it from 60 Hz render time.
3. Run strict runtime tests plus the relevant release/fast-math binaries GREEN.
4. Commit as `feat: publish authored hand constraint weight`; obtain independent review.

## Task 3: Add the Flat Target-Rig Arm Solver

**Files:** add `interaction_target_rig_ik.h/.cpp` and `tests/cpp/test_interaction_target_rig_ik.cpp`; update `Makefile`; minimally refactor/share flat FK definitions from `interaction_controller_adapter.*` without changing retarget behavior.

1. Add RED tests for both hands:
   - weight-zero bit identity;
   - every local translation and inactive channel bit identity;
   - reachable full-weight position `<= 0.001 m` and calibrated orientation `<= 0.5 deg`;
   - fixed segment lengths;
   - mirrored, straight, folded, unreachable, antiparallel, and degenerate-pole finite behavior;
   - previous-pole branch continuity;
   - quaternion hemisphere continuity and finite angular velocities.
2. Implement rotation-only epoch calibration, weighted semantic-grasp target, robust analytic two-bone solve, wrist orientation, reach clamp, and diagnostics. Do not copy G1 hinge limits or modify object state.
3. Build/run the focused solver test with strict warnings, then the adapter/runtime focused suites.
4. Commit as `feat: solve selected flat arm to semantic grasp`; obtain independent review.

## Task 4: Integrate IK with Handoff and Release

**Files:** `interaction_controller_adapter.h/.cpp`, `controller.cpp`, focused C++ tests, and Python policy tests.

1. Add RED tests proving:
   - first ownership frame remains exact at weight zero;
   - full-body and layered Carry both correct only the selected arm;
   - layered bones `0..9` and contacts remain bit-equal to fresh locomotion;
   - exact selected scene target/affordance supplies the constraint;
   - stale/missing affordance disables IK safely;
   - the handoff records the final corrected pose without changing ownership references;
   - the first release frame equals the last corrected pose and the normal `0.25 s` release completes.
2. Resolve scene object/affordance before final pose publication, apply target-rig IK after handoff/layer composition and before foot IK/final FK, and commit the corrected pose as the handoff release origin.
3. Add calibrated grip-orientation evidence (raw orientation remains diagnostic). Add attached-state alignment validation at `<= 0.01 m` and `<= 2 deg` only for reachable, full-weight frames; unreachable frames must be explicit failures, not exemptions.
4. Run focused C++/Python tests GREEN.
5. Commit as `fix: constrain rendered hand to active grasp`; obtain independent review.

## Task 5: Fresh Closure Gate and Visual Inspection

1. Run `make -B controller BUILD_MODE=RELEASE` and `make test-interaction-safe`.
2. Run a unique `gate-playable-interaction` pack/evidence path such as `grasp-ik-20260715-v1`; retain unchanged `0.20 m` / `60 deg` whole-skeleton limits and the new attached-grasp limits.
3. Report state-phase continuity maxima and hand/grasp count/mean/max for PickupReplay pre-contact, attached PickupReplay, Hold, and Carry.
4. Generate a fresh lossless/stage capture and inspect entry, contact, Hold, moving Carry, and Reset/release. Verify that the object sits in the selected hand and that elbow/wrist motion does not branch or pop.
5. Run `git status --short --untracked-files=no`, `git diff --check a144d35..HEAD`, and an independent whole-branch review. Do not merge terrain-aware work.

If the current target is physically unreachable for the flat arm, stop after reporting exact reach shortfall and design an explicit clavicle/root-warp or affordance-offset correction; do not stretch local translations.
