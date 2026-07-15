# Carry seam and fast-math validation design

## Goal

Make every Hold-to-Carry and Carry-mode selection boundary continuous, regardless of whether the destination pose is already inside the hand-IK correction limits, while preserving immediate live-root motion and the existing grasp tolerances. Harden Carry's invalid-input checks under the controller's optimized `-ffast-math` build.

## Review findings

The current Carry warm path is entered only after the complete layered pose fails IK. An IK-feasible mismatch outside the active arm chain, such as a 0.9 radian `LeftHipPitch` change during right-hand Carry, therefore publishes in one frame. A recorded candidate returns even earlier, so initial recorded selection and recorded retargets bypass smoothing as well.

Carry also uses `std::isfinite`. Under `-ffast-math`, the compiler may assume values are finite and optimize those checks away. The existing target registry avoids this by inspecting IEEE-754 exponent bits, but Carry does not have an equivalent optimized test target.

## Publication-stage Carry transition

Recorded and layered evaluation will produce a candidate instead of publishing immediately. One publication stage will handle both modes.

Each candidate has a seam key:

- layered: `(layered, 0)`;
- recorded: `(recorded, recorded_selection_epoch)`.

The epoch increments when search explicitly selects or retargets a recorded source, including a terminal restart or a different recorded range. Ordinary frame-by-frame progression within the current recorded range does not change the epoch.

`CarryController::start` leaves the published key unset. The first candidate therefore starts a transition. A later candidate whose key differs from the active key also starts a transition. Starting a transition captures the last published safe pose as a fixed non-root source and resets explicit progress to 0.

On each 25 Hz update:

1. Advance requested progress toward 1 by `dt / 0.50 s`.
2. Set the root position, rotation, and root velocities directly from live locomotion; the root is never interpolated or held.
3. Interpolate every continuous non-root pose, velocity, and hand-DOF channel from the captured source toward the current candidate at cumulative progress. Copy the candidate's discrete foot-contact labels.
4. Solve the active hand against the candidate object. Publish only if the solved hand/object remains within the existing Carry and IK tolerances.
5. If the full-channel step is not solvable, retry that same non-root progress from the last accepted active-arm rotations, then back off only the unaccepted progress increment. Progress never moves backward.

Both the pose and object publication stay on the last accepted state if all bounded attempts fail, except that the last-safe pose is remapped to the current live root. This retains the existing rejection guarantee without freezing locomotion.

At progress 1 the destination is published through the same IK/bounds path used during transition. Direct publication resumes only on a later update that produces the same mode/epoch candidate without a discontinuity. Thus there is no special completion-frame snap. If a full layered target continually needs the accepted active-arm correction, direct layered evaluation starts from that accepted arm state.

`recorded()` reflects the candidate mode during its transition. Transaction rollback covers the selection epoch, seam key, transition source, and transition progress as well as the existing search/object state.

## Fast-math-safe finite validation

Carry will use local bit-level `finite(float)` and `finite(double)` overloads. They copy the value into `uint32_t` or `uint64_t` and reject an all-ones exponent, matching the hardened controller adapter and target registry pattern. Every Carry `std::isfinite` use, including accumulated search time and matching costs, will use these overloads.

A dedicated `test_interaction_carry_release_fast_math` binary will compile with `-O3 -DNDEBUG -ffast-math`. Its source uses always-active exception checks and compilation guards requiring both `NDEBUG` and fast-math. It proves that a NaN final Hold pose and a NaN update `dt` throw `std::invalid_argument` in the actual optimized configuration.

The dedicated binary intentionally does not run the assertion-based normal Carry suite under `NDEBUG`; that would erase test expressions and mix invalid-input safety with floating threshold sensitivity. The normal suite continues to own inclusive-threshold semantics. The safe suite will build and execute both the existing target and new Carry optimized binaries.

## Tests and acceptance

- Default-config layered RED: right-hand Carry with only live `LeftHipPitch` changed by 0.9 radians must move only a bounded fraction on the first 0.04 s tick and converge on the 0.50 s schedule.
- Recorded initial RED: a selected recorded pose with a deliberate non-arm mismatch must not publish in one frame and must converge without losing the grasp.
- Recorded switch RED: switching between two certified ranges with different non-arm poses must reset transition progress; normal progression within one range must not.
- Existing last-safe rejection, live-root, object continuity, `--allow-rejections`, and greater-than-0.20 m graphical displacement contracts remain green.
- The optimized Carry target must reject NaN Hold and NaN `dt` with assertions disabled.
- Final verification uses the safe suite and a fresh isolated graphical pack/evidence path. The validated full pack and protected untracked repo-root probe remain untouched.
