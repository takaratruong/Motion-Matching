# Target-Flat Hand/Grasp Evidence and Arm IK Design

**Status:** Approved for implementation on `g1-manipulation-motion-matching` after `a144d35`.

## Problem

The ownership-reference retarget now keeps the authoritative 23-joint rendered skeleton continuous. The fresh strict trace passes the unchanged `0.20 m` / `60 deg` adjacent-joint limits, but continuity alone does not prove that the rendered hand remains on the selected grasp. The existing trace contains the final rendered hand transforms and only the object position, so it cannot reconstruct the selected grasp transform. Earlier diagnostics measured about `0.29 m` hand separation in Hold and up to `0.79 m` in Carry.

The correction must remain animation-led. It must not pin the hand to the final grasp throughout PickupReplay, stretch local translations, move the object to hide an error, or reintroduce a release snap.

## Evidence Contract

Autoplay evidence remains a fixed-order, fixed-six-decimal JSONL schema. Add:

- `grasp_evidence_valid`;
- `active_hand_joint` (`18`, `22`, or `-1` when invalid);
- the scene-authoritative `object_world_rotation` (object position already exists);
- `hand_in_object_position` and `hand_in_object_rotation` from the exact selected affordance;
- the derived `grasp_world_position` and `grasp_world_rotation`.

The final rendered hand remains authoritative in `joint_world_positions` and `joint_world_rotations`. The writer receives one immutable post-final-FK capture; it must not read mutable globals or a debug skeleton. For PickupReplay, Hold, and Carry, the target handle, affordance ID, and selected hand must agree before evidence is marked valid.

The Python validator recomputes `grasp_world = object_world * hand_in_object`, validates finite/unit transforms and exact hand indices, and reports per-state count/mean/max raw wrist-to-grasp position and sign-invariant raw orientation error. This first evidence change is diagnostic: it does not weaken the global continuity gate and does not yet reject large grasp errors.

Raw wrist orientation is not a visual correctness gate. G1 and the flat rig can use different hand-bone axes; the previously observed roughly `123 deg` difference may be frame convention rather than visible separation.

## Runtime Contact Weight

The interaction runtime publishes a finite `hand_constraint_weight` in `[0, 1]` from the same playback/contact progress used by reach correction:

- exactly `0` at replay entry;
- monotonic during the authored reach;
- exactly `1` by Contact and while attached in PickupReplay, Hold, and Carry;
- reset only when runtime ownership ends; release does not apply a new solve.

The 60 Hz controller consumes the cached 25 Hz value. It does not infer a curve from render time. A held cached value must produce deterministic non-due frames.

## Target-Rig Calibration

At each fresh ownership epoch, capture the selected G1 reference-hand world rotation and the displayed flat reference-hand world rotation. The axis calibration is:

```text
C_rot = inverse(R_g1_reference_hand) * R_flat_reference_hand
R_flat_goal = R_grasp * C_rot
```

Only orientation uses this epoch calibration. Full-weight wrist position targets `grasp_world.position` directly. No ownership-entry translation is preserved as calibration because it would conflate initial pose/morphology mismatch with a real grip-socket offset. If a future target rig needs a palm/socket positional offset, it must be explicit affordance or target-rig data; the current default is zero.

The semantic calibrated grip orientation is `R_flat_hand * inverse(C_rot)` and can be compared with `R_grasp` after the solver is integrated.

## Flat Arm IK

Use a dedicated, testable analytic solver on the selected flat chain:

- left: keep `LeftShoulder(15)` fixed, rotate `LeftArm(16)` and `LeftForeArm(17)`, then orient `LeftHand(18)`;
- right: keep `RightShoulder(19)` fixed, rotate `RightArm(20)` and `RightForeArm(21)`, then orient `RightHand(22)`.

Every local translation, the inactive arm, and bones `0..14` remain bit-identical. Segment lengths therefore cannot change.

For the current base hand transform `F_base`, form the weighted end-effector target:

```text
p_target = lerp(p_base, p_grasp, weight)
R_target = nlerp_shortest(R_base, R_grasp * C_rot, weight)
```

This is an IK correction weight over the current animation, not a repeated feedback lerp of the previous rendered hand.

Solve the two segment position analytically with law of cosines. Clamp target distance to the reachable annulus using a small numerical epsilon. Select the elbow branch from the current animation elbow-plane projection; retain the previous valid pole near a singularity; use a deterministic Spine2-derived fallback only when both are degenerate. Do not invent G1 hinge limits for the flat rig. Hemisphere-normalize every quaternion and return finite diagnostics for reachable, clamped, and rejected solves.

Run the solver at 60 Hz after full-body/layered handoff composition and exact scene-grasp resolution, before foot IK and the final FK. Foot IK remains unchanged. The object remains runtime-authoritative.

## Ownership and Release

The handoff must store the final IK-corrected flat pose as its last rendered pose without mutating the immutable ownership references. On the first non-owned frame, no new arm solve runs; the existing `0.25 s` release blend starts from the last corrected pose. This prevents a corrected Hold/Carry hand from snapping back to the pre-IK arm at Reset.

## Acceptance

- `weight == 0` returns the input pose bit-for-bit.
- Only the selected arm rotations/angular velocities may change; all local translations remain bit-identical.
- A reachable, full-weight synthetic target is reached within `1 mm`; calibrated orientation within `0.5 deg`.
- Straight, folded, mirrored, unreachable, and degenerate-pole cases remain finite and preserve the chosen elbow branch.
- The first release frame preserves the last corrected hand within `1 mm` / `0.5 deg`, then the normal release completes.
- Fresh attached Contact/Hold/Carry evidence reaches raw wrist position error `<= 0.01 m` and calibrated orientation error `<= 2 deg` when the target is reachable.
- The unchanged whole-skeleton `0.20 m` / `60 deg` continuity gate still passes with no exemptions.

If the full-weight target is unreachable, report the clamped reach and keep the object authoritative. Do not stretch the arm or silently pass the alignment gate.
