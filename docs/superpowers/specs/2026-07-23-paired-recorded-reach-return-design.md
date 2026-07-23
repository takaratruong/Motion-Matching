# Paired Recorded Reach Return

**Date:** 2026-07-23

## Goal

After a successful warped pickup, replay the selected recording's authored
post-grab return to its nominal posture. Preserve the working outbound pickup
and reversible placement behavior, keep the object rigidly attached, and
remove the procedural carry-IK transition that currently rejects feasible
returns.

Walking-quality changes are outside this repair.

## Confirmed Cause

The trusted `g1_retargeted_motions.zip` recordings contain repeated
neutral-to-grab-to-neutral actions. The review corpus preserves each complete
source sequence, but reach-pack version 1 intentionally stores only the
outbound interval ending at the confirmed grab frame.

At runtime, `LayeredCarry` therefore has no recorded return. It retains the
final pickup arm pose and the contact-time object transform relative to the
root while the walking matcher changes the torso. The solver eventually tries
to maintain an infeasible contact reach and reports carry IK rejection. The
visible carry blend interpolates displayed poses but does not define a
returning wrist target.

## Chosen Architecture

Build a versioned reach pack that pairs every accepted outbound reach with its
authored post-grab return. Mirrored right-hand reaches receive the mirrored
return from the same captured source interval.

The endpoint search features and outbound candidate ranking remain unchanged.
The selected plan additionally carries a shaped return pose sequence.

```text
walking approach
      |
warped recorded outbound reach
      |
certified contact + attach
      |
warped recorded return
  correction weight: 1 -> 0
      |
recorded nominal arm over walking locomotion
      |
placement uses the existing search and reversed reach playback
```

## Return Extraction

For each accepted annotation, use its source sequence and grab frame from the
full 25 Hz review corpus. Find the first stable post-grab return to the
sequence's neutral wrist basin, bounded before the next distinct outward
excursion and by a maximum return duration. Stability requires a short
low-velocity interval, not a single proximity sample.

The builder records:

- the outbound range;
- the contact/grab frame;
- the paired return range;
- the return endpoint;
- captured or mirrored provenance.

If an accepted outbound reach ends before a valid, nonempty return was
recorded, keep its outbound motion and mark its return unavailable. Apply the
same marker to its mirror. Such clips remain valid coverage/search data but
are excluded from playable pickup planning, which requires a complete return.
Do not silently substitute a reversed outbound clip, a donor return, or a
generic carry pose. Aggregate return durations and unavailable-return counts
are reported in the manifest.

## Continuous Return Warp

The outbound planner already produces a corrected contact hand transform for
the requested grasp. Let:

- `H_source_contact` be the aligned source wrist at contact;
- `H_target_contact` be the solved wrist at contact; and
- `D = H_target_contact * inverse(H_source_contact)` be the contact correction.

For each recorded return frame, evaluate a smooth correction weight `w` that
starts at one at contact and reaches zero at the recorded return endpoint.
Interpolate an SE(3) correction from identity to `D`, then apply it to that
frame's aligned source wrist:

`H_target(t) = interpolate(identity, D, w(t)) * H_source(t)`.

Use the existing posture-aware bounded IK to shape each return frame against
`H_target(t)`. The recorded return pose is the posture reference, and the
previous accepted frame is the temporal seed. The first published return pose
must equal the outbound contact pose within the existing seam tolerances. The
last target is the unmodified recorded nominal wrist.

The frozen `hand_in_object` transform remains authoritative. Every attached
object transform is derived from the displayed wrist, so the object cannot
drift independently of the hand while the warp decays.

## Search and Collision Validity

Return shaping is part of candidate feasibility, not a post-attachment best
effort. Before committing a pickup candidate:

- every return IK frame must be finite and accepted;
- the active hand's grasp volume may contact the held object;
- all other body parts must remain outside the object;
- body and held-object motion must remain collision-free against the table,
  shelf, lower table, and floor under the existing collision conventions; and
- the contact-to-return seam must satisfy bounded root and joint changes.

If a candidate's return fails, search continues to the next outbound candidate.
Failure diagnostics distinguish return IK, object collision, environment
collision, and seam rejection.

## Episode State and Neutral Hold (narrowed baseline)

**User-approved narrowed scope (2026-07-23):** this milestone ends the playable
baseline at the recorded return endpoint. There is no carry motion matching,
walking-while-carrying, or placement locomotion after neutral. The pickup ->
paired recorded return -> stable neutral hold path is the entire behavior.

Add an attached `Return` state after pickup contact. It owns the complete shaped
return playback and ignores locomotion commands until the recorded endpoint is
reached. Reset and cancellation retain the existing attachment safety rules.

Return playback validity and ordering:

1. `commit` rejects an empty or non-finite return trajectory before mutating any
   episode state (`finite_return_trajectory` guards `InteractionEpisode::commit`,
   so a NaN return leaves the runtime in `FreeLocomotion` with no active attempt);
2. the first paired return pose is published at contact by `update_reach`;
3. `update_return` advances at most one authored sample per update, so a large
   `dt` can never skip past intermediate recorded return samples; and
4. the object stays attached and wrist-synchronized through every return sample
   via the frozen `hand_in_object` transform and `update_attached_object`.

At the final recorded return sample, the episode enters the `Neutral` state:

1. capture the displayed pose and attached object transform as the frozen
   neutral hold (`neutral_pose_`, `neutral_object_`);
2. stop all motion matching — no walking matcher advance, no carry IK, no
   layered carry start; and
3. on every later update, `update_neutral_hold` re-publishes the exact recorded
   neutral pose and holds the object transform unchanged, ignoring all
   locomotion commands.

This makes the data's reachable return endpoint a stable neutral hold and
removes the failure mode where changing walking torso motion invalidates a fixed
world-space IK target.

Placement, carry, and layered-carry code remain in the tree only so the project
keeps compiling; they are unreachable in the narrowed baseline because `Return`
now terminates in `Neutral` rather than handing off to `Carry`.

## Compatibility

Reach-pack loading is versioned and rejects ambiguous layouts. A contact frame
equal to the final stored clip frame explicitly means that no paired return is
available; it is not inferred from missing metadata. The new pack is built
beside existing generated artifacts and becomes the explicit viewer input only
after validation. Source archives and review annotations are not modified.

Generated reach packs and viewer binaries remain untracked.

## Verification

Automated tests must prove:

1. all accepted outbound reaches remain present;
2. every available captured return is source-contiguous and its mirror contains
   the exact bilateral mirrored return;
3. return-unavailable clips and their mirrors are excluded from playable
   pickup planning without disappearing from coverage;
4. the outbound endpoint and search features are unchanged;
5. the shaped return begins at the solved target grasp;
6. warp weight reaches zero at the recorded nominal endpoint;
7. every accepted frame preserves the frozen hand/object transform;
8. return IK and collision failures reject the candidate before attachment;
9. `commit` rejects an empty or non-finite return trajectory before mutating
   episode state;
10. the episode transitions `Reach -> Return -> Neutral`, publishing the first
    return pose at contact and every later authored sample in order under a
    large `dt`;
11. the final recorded neutral pose and attached object remain unchanged under
    later locomotion input, with no motion matching after neutral; and
12. left- and right-hand pickup retain one finite attached 31-bone G1 pose
    through the neutral hold.

Manual acceptance picks up from multiple approach angles and object heights,
watches the complete authored return, and confirms the character then holds the
recorded neutral pose with the object attached while later locomotion commands
are ignored. The viewer labels this state HOLD. Pickup quality must not regress,
the object must not jump at contact, and no carry IK rejection may appear.
