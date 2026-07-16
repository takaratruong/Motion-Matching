# Pickup entry feasibility preview

Date: 2026-07-16
Status: approved design checkpoint

## Context and measured failure

The placement auto-demo must approach the pickup through ordinary live flat
locomotion and submit one ordinary Interact request. It may use the mapped
clip-0 Reach root as navigation geometry, but it may not install a canonical
pose, relocate either displayed or simulation root, fabricate an external
`MatchInput`, reserve a target during staging, or retry a rejected interaction.

The current affordance arc constructs two positions at the required clearance
chord and preserved object standoff, then chooses the greater active-hand score
without asking whether that root is collision-feasible. For the current pack:

- table center is `(0.000000, 0.679574, 3.000000)` and size is
  `(2.000000, 0.040000, 0.600000)`;
- the unchanged root/table proxy has planar half extents `(1.24, 0.54)`;
- the pickup object center is
  `O = (0.142708, 0.797915, 2.816170)`;
- the stable Reach root is `R = (0.251025, 0, 2.425281)`;
- preserved standoff is `D = 0.405618 m`, clearance chord is
  `C = 0.138841 m`, and the corresponding arc angle is `19.709 degrees`;
- the greater right-hand-score candidate is
  `P_minus = (0.376504, 0, 2.484710)`. Its table-local Z is `-0.515290`,
  which penetrates the root/table proxy by `0.024710 m`;
- the live root at rejection was `(0.343549, 0, 2.482831)`. Its table-local Z
  was `-0.517169`, inside the proxy by `0.022831 m`; and
- `P_plus = (0.112855, 0, 2.411651)` has table-local Z `-0.588349`, outside
  the proxy by `0.048349 m`, while preserving the same `C` and `D`.

The pickup matcher therefore rejects the preferred slot during Preflight with
`BlockedPath`. The primary Reach entry is frame 114 and the first failing
root/table test is its entry root, before Align or entry-blend playback begins.
Contact starts at frame 139. This is neither terminal contact nor a reason to
change contact exemptions, collision dimensions, IK, or any threshold.

A direct read-only probe against the current binary database and feature pack
used the same canonical-equivalent Reach snapshot shape, rigidly mapped a copy
to each prospective root, and called the real query normalization and whole-clip
matcher. Its oracle results were:

| Prospective root | Path/match result | Entry | Contact | Cost |
| --- | --- | ---: | ---: | ---: |
| `R` | accepted | 114 | 139 | `0.666622` |
| `P_minus` | `BlockedPath` | - | 139 | - |
| `P_plus` | accepted | 114 | 139 | `0.718336` |

The canonical-equivalent probe is diagnostic evidence only. Production preview
must consume the actual live-flat snapshot supplied on a native 25 Hz controller
tick. It must never silently substitute the diagnostic snapshot when a walking
or transient live pose reports `PoorMatch`.

## Goals

The controller will treat `P_plus` and `P_minus` as two entry slots. A new
runtime-owned, read-only pickup preview will evaluate each slot using one common
live-flat snapshot, a prospective planar XZ/world-yaw root, the exact registered
target and affordance, the runtime's database/features/configuration, and the
same matcher pipeline used by pickup Preflight. Collision/path feasibility is an
eligibility filter. Active-hand score ranks only eligible, match-ready slots.

The design must:

- preserve both exact arc invariants, `|P - R| = C` and `|P - O| = D`;
- preserve the `0.15 m` Reach neighborhood, `[0.35 m, 0.45 m]` standoff,
  `0.25 m` matcher root-correction limit, `25 degree` matcher yaw limit, and all
  existing collision/IK/cost limits;
- distinguish a geometrically feasible slot from a live snapshot that is not
  yet match-ready because of transient feature cost;
- keep registry lookup, query construction, matcher configuration, and
  prospective snapshot mapping inside `InteractionRuntime`;
- leave runtime state, diagnostics, both registries, target ownership, request
  IDs, pose ownership, and attachment unchanged;
- use only the ordinary live-flat provider and ordinary left-stick navigation;
- choose once before the slot-specific approach, then keep one-way braking; and
- record enough evidence to independently reconstruct both slots, their order,
  their preview outcomes, and the deterministic selection.

## Non-goals and fixed boundaries

This work does not change the interaction pack, selectable entry frames, matcher
costs or priorities, path predicates, collision geometry, terminal-contact
exemption, IK, playback, attachment, placement, scheduler rate, or controller
movement thresholds. It does not add a canonical placement snapshot, arbitrary
six-DoF pickup correction, direct root write, external `MatchInput`, public free
selector, reservation preview, failed-Interact fallback, or post-brake reversal.

The existing earlier-Approach fallback remains owned by the matcher. Broader
multi-clip or multi-entry search is outside this amendment.

## Two-stage entry-slot selection

### Stage 1: construct both exact arc slots

The existing validated inputs remain `R`, `O`, object rotation and dimensions,
object-local approach, affordance clearance, and active hand. The oriented
support and clearance chord remain:

```text
A_world = normalize(planar(Q * A_object))
L_world = cross(world_up, A_world)
L_object = inverse(Q) * L_world
support = 0.5 * dot(abs(L_object), object_dimensions)
C = support + clearance + 0.001 m
r = planar(R - O)
D = length(r)
theta = 2 * asin(C / (2D))
P_plus  = O + rotate_y(+theta, r)
P_minus = O + rotate_y(-theta, r)
```

Both candidates copy `R.y` and `R.rotation`. Construction validates both rather
than selecting one immediately. Each must independently satisfy the existing
finite/unit/positive-input checks and, within `2e-5 m`:

```text
planar_length(P_slot - R) == C
planar_length(P_slot - O) == D
C <= 0.15 m
0.35 m <= D <= 0.45 m
P_slot.y == R.y
P_slot.rotation == R.rotation
```

Slots have stable identities `Plus` and `Minus`. Evaluation order is always
`Plus`, then `Minus`, regardless of world orientation or active hand.

The existing hand score is retained only as a ranking preference:

```text
H_world = (+1 for Right, -1 for Left) * L_world
score(slot) = dot(P_slot - R, H_world)
```

It is not a collision certificate. The current right-hand sign is also not
semantically reversed: at Reach 114, `(R - O) dot L_world = +0.295034 m`, while
the right wrist relative to the root projects `-0.255378 m`; at contact 139 the
values are `+0.296782 m` and `-0.275342 m`. Positive `L_world` places the root
opposite the anatomical right hand, consistently with the recorded clip.

### Stage 2: runtime preview and selection

The controller first follows the existing common pre-entry route derived from
`R`. At a normal 25 Hz tick, it supplies the same actual live-flat snapshot to
two read-only previews, one for each prospective slot. No runtime update is
inserted and no alternate-rate callback is introduced.

Preview exposes two separate facts:

1. `path_feasible`: at least one normal pickup entry survives request-independent
   target/context validation, correction limits, contact-hand compatibility, and
   the exact existing root/table, hand/table, and pre-contact hand/object path
   predicates for the prospective root.
2. `match_ready`: the exact normalized query and configured whole-clip selection
   accept a candidate for that prospective snapshot now.

`match_ready` implies `path_feasible`. A path-feasible result may be not ready
with `match_reason = PoorMatch` while the live base pose is still walking. The
controller does not replace that pose with canonical data. It holds at the
common pre-entry point with ordinary zero input and evaluates both slots again
from the same next live-flat provider sample on a subsequent 25 Hz tick. This
settling phase is bounded by the existing approach deadline. Exhausting the
deadline fails closed and publishes both last outcomes.

Once at least one slot is match-ready, the eligible set contains exactly the
slots for which both booleans are true. A higher-scoring but blocked or otherwise
not-ready slot cannot win. The eligible slot with the greater hand score wins.
If both scores are bit-equal, Right chooses `Plus` and Left chooses `Minus`,
matching the existing deterministic tie rule. If only one slot is eligible, it
wins regardless of score. If neither path is feasible, selection fails
immediately. If paths are feasible but no slot becomes match-ready before the
deadline, selection fails without Interact.

The selected identity and transform are frozen before slot-specific navigation.
The controller then uses ordinary left-stick input and existing one-way braking
toward that slot. It never changes slots after braking. The eventual Interact
still uses the ordinary live request resolver and actual live-flat snapshot;
Preflight reserves, rebuilds, and selects normally. Preview is advisory and no
preview candidate or identity is submitted as execution authority. An actual
Preflight rejection fails the gate; it does not trigger a move to the other slot.

## Runtime-owned read-only API

The public API is deliberately narrower than `MatchInput`:

```cpp
struct PickEntryRoot {
    float world_x = 0.0F;
    float world_z = 0.0F;
    float world_yaw_radians = 0.0F;
};

struct PickEntryPreview {
    bool path_feasible = false;
    bool match_ready = false;
    Reason path_reason = Reason::None;
    Reason match_reason = Reason::None;
    PickEntryRoot prospective_root{};
    int32_t feasible_entry_frame = -1;
    int32_t contact_frame = -1;
    float total_cost = 0.0F;
    MatchCandidate match_candidate{};  // valid only when match_ready
};

PickEntryPreview InteractionRuntime::preview_pick(
    const LocomotionSnapshot& live_flat_snapshot,
    PickEntryRoot prospective_root,
    TargetHandle target,
    uint32_t affordance_id) const;
```

The method is valid only in `Locomotion` with available pickup database,
features, registry, and matcher configuration. It performs exact-generation
lookup, requires a `Free` target, resolves exactly the requested authored
affordance, and rejects missing, stale, targeted, attached, or held objects.
It does not accept a caller-supplied target snapshot, affordance snapshot,
features, database, query, `MatchInput`, config, candidate, or request ID.

The strong `PickEntryRoot` type is intentionally planar. Slot X/Z come from
`P_plus` or `P_minus`, and slot yaw is the world yaw of `R.rotation`. The API
cannot request root Y, pitch, or roll changes because pickup matching and entry
correction support only planar translation and world yaw.

The matcher gains one internal structured evaluation path shared by preview and
normal selection. It reports whether any entry passed all hard feasibility
filters before cost and retains the existing `select_whole_clip` accepted/reason
behavior unchanged. Preview does not achieve path evidence by raising
`maximum_cost`, disabling a feature group, swallowing `BlockedPath`, or calling a
different collision helper. Normal Preflight and preview share target/query input
assembly and the same configured evaluator; request ownership and reservation
validation remain the Preflight-only outer layer.

`path_reason` is `None` when `path_feasible` is true. Otherwise it is the same
deterministically aggregated hard-filter reason that prevented feasibility.
`match_reason` is `None` only when `match_ready` is true; a feasible but
cost-rejected preview reports `PoorMatch`. Invalid feature storage or another
non-transient match failure remains visible and is never relabeled `PoorMatch`.

When no entry is path-feasible, hard failures retain the current matcher order:
`OutOfRange`, `CorrectionLimit`, `BlockedPath`, then `NoCandidate`. Reaching the
cost test proves path feasibility even when the cost is over its unchanged
limit. `feasible_entry_frame` and `contact_frame` identify the first feasible
entry in the matcher's existing deterministic evaluation order; the nested
`match_candidate` identifies the cost-ranked winner only when `match_ready`.

## Prospective rigid snapshot mapping

Preview copies the caller's snapshot before any transformation. Let `p_live`
and `q_live` be its root position and rotation. The runtime derives:

```text
live_yaw = world_yaw(q_live)
delta_yaw = shortest_angle(slot.world_yaw - live_yaw)
q_delta = world_yaw_rotation(delta_yaw)
p_slot = (slot.world_x, p_live.y, slot.world_z)
t_delta = p_slot - q_delta * p_live
```

The runtime applies the embedded planar rigid map
`map(p) = t_delta + q_delta * p` consistently:

- copied root X/Z and world yaw become exactly the requested values;
- copied root Y is unchanged, and its pitch/roll content is preserved by
  `q_mapped = q_delta * q_live` rather than replaced by a slot quaternion;
- copied root linear and angular velocities rotate by `q_delta`;
- every future root position maps through `map`;
- every future root rotation is left-multiplied by `q_delta`;
- child-local positions, rotations, velocities, and angular velocities remain
  unchanged, which rigidly maps every reconstructed world bone transform and
  velocity through the root; and
- hand DOF, hand velocities, and foot contacts remain unchanged.

Mapping must prove, within the existing pose tolerances, that every mapped world
bone position equals `map(original)`, every world bone rotation is
`q_delta * original`, world linear/angular velocities are rotated by `q_delta`,
live root Y and pitch/roll are preserved, relative bone geometry is unchanged,
future-root deltas are rigidly equivalent, and the input snapshot is exactly
field-for-field unchanged. Both slot previews in one epoch use the same original
snapshot and differ only in the three `PickEntryRoot` scalars.

Non-finite snapshot data, non-finite prospective X/Z/yaw, a non-finite or
non-unit live root rotation, an undefined live world yaw, a failed mapping
invariant, or an out-of-contract planar root returns a rejected preview. Preview
never normalizes bad caller data into acceptance.

## Mutation and ownership contract

Calling `preview_pick` any number of times must not change:

- `InteractionRuntime::state()`, diagnostics, pose, target/object snapshots,
  request/candidate/player/attachment/carry/place optionals, or event clocks;
- target or surface registry contents, generation, state, owner, or reservation;
- the caller's live snapshot or prospective planar root;
- controller/scheduler phase, pending edges, request sequence, pose authority,
  or steering suppression; or
- simulation/displayed roots and their initialization/inertialization state.

Preview has no `dt`, emits no playback event, and cannot reserve or attach. It
may be called only from an ordinary controller tick after the live-flat provider
has produced the snapshot. Preview call counters are observational evidence, not
runtime state or timing input.

## Rejected alternatives

### Direct controller matcher call

Having `controller.cpp` construct `QueryInput`/`MatchInput` and call
`select_whole_clip` is locally shorter but violates the established ownership
boundary. It exposes database/features/config and registry snapshots to the
controller, duplicates Preflight validation, and is exactly an external match
input. It would require `external_match_input_calls > 0` and invalidate the
placement evidence contract. Using the clip Reach pose to make that call would
also be a forbidden canonical-snapshot substitution. The runtime preview is the
only supported route.

### Sequential failed-Interact retry

Navigating to the preferred slot, submitting Interact, accepting a
`BlockedPath`, and then walking to the other slot mutates target reservation and
runtime diagnostics, adds a failed state transition to evidence, reverses the
one-way approach after braking, and treats an execution failure as a selector.
Only one Interact may be submitted; both slots are previewed before slot-specific
navigation.

### Outward-radius replacement

The fixed-`C`, fixed-`D` construction is the intersection of the circle around
`R` with radius `C` and the circle around `O` with radius `D`, so it has exactly
the two existing solutions. The current table-outward point on the object radius
would be approximately `(0.142708, 0, 2.410551)`. It preserves `D`, but its Reach
chord is only `0.109314 m`, short of `C` by `0.029527 m`, and therefore discards
the oriented clearance proof. If both `C` and `D` are preserved, `P_plus` is
already the outward solution. Geometry is not replaced; feasibility selects
between the two valid intersections.

## Failure behavior

Candidate construction or preview fails closed on malformed geometry, stale or
non-Free registry state, missing affordance/dependencies, invalid live snapshot,
invalid prospective planar root, rigid-map failure, hard correction/path failure,
non-deterministic repeated results, preview mutation, no feasible slot, or no
match-ready slot before the approach deadline. A runtime preview exception is a
visible controller error after verifying rollback/no mutation.

No failure may clamp the arc, choose an unevaluated world-axis fallback, loosen
cost/collision/IK limits, use canonical data, reserve a target, pulse Interact,
switch slots after freezing, or continue to evidence publication as success.

## Evidence contract

Each placement evidence row from the first preview epoch onward records immutable
arc inputs plus a fixed ordered pair of slot records. Each record includes:

- identity (`Plus` or `Minus`) and evaluation order;
- navigation transform, prospective preview X/Z/world-yaw, chord, preserved
  standoff, and hand score;
- preview epoch and one common live-flat snapshot fingerprint/source marker;
- `path_feasible`, `match_ready`, `path_reason`, and `match_reason`;
- feasible entry/contact frames and total cost when available; and
- whether the slot was eligible and selected.

Run-level instrumentation records runtime pickup-preview call count, preview
epochs, preview mutation count, selected slot, freeze tick, actual Interact count,
live-flat/canonical provider counts, external-match-input count, reservation
transitions, root-relocation/direct-write counts, and the existing approach,
settle, state, collision, attachment, placement, and handoff evidence.

The snapshot fingerprint is a canonical field-by-field digest of the live
provider sample, with quaternion signs and negative zero canonicalized. It does
not hash object addresses or struct padding.

The Python validator independently reconstructs `C`, `D`, `theta`, both slots,
stable identities/order, both scores, and the deterministic winner from logged
outcomes. It requires:

- both previews in an epoch share one actual live-flat snapshot fingerprint;
- `canonical_snapshot_used == false`, `external_match_input_calls == 0`,
  preview mutation count zero, and all root-write/initialization counts zero;
- no blocked or not-ready slot is ranked as eligible;
- the selected slot is the hand-score winner among eligible slots, with the
  exact tie rule;
- selection freezes once, before slot-specific navigation and braking;
- exactly one normal pick resolver call and one Interact submission; and
- actual Preflight reaches Align rather than triggering any retry.

For the current pack's named acceptance evidence, `Minus` must report
`BlockedPath`, `Plus` must be path-feasible and match-ready after any bounded
live settling, the selected identity must be `Plus`, and previewed entry/contact
must be 114/139. If an initial walking epoch reports `PoorMatch` for both, that
epoch remains in evidence and a later live-flat epoch must become ready; no
canonical result may replace it.

## TDD and verification

Implementation begins with failing tests for:

1. the exact public const API and separate path/match result fields;
2. planar rigid mapping of all world bone transforms/velocities and future roots,
   preserved live root Y/pitch/roll, unchanged relative pose channels, and an
   unchanged input snapshot;
3. exact boundary behavior for root/table, hand/table, and pre-contact
   hand/object path checks;
4. `BlockedPath` as path-infeasible, accepted as feasible/ready, and an
   over-cost otherwise-valid query as feasible/not-ready/`PoorMatch`;
5. parity between preview's shared internal evaluation and ordinary
   `select_whole_clip`/Preflight on the same realized snapshot;
6. rejection of disabled/non-Locomotion runtime, missing dependencies, stale
   generation, wrong affordance, non-Free target, invalid snapshot/planar root,
   and every attempted non-finite X/Z/yaw input;
7. repeated preview determinism and exactly field-equal runtime diagnostics and
   registry snapshots before and after success, rejection, and exception paths;
8. a blocked higher-hand-score slot losing to a feasible lower-score slot,
   two-ready score ranking, one-ready selection, exact ties, no-ready deferral,
   deadline failure, and immutable selection after freeze;
9. common-snapshot/evaluation-order behavior at native 25 Hz without extra
   runtime updates, scheduler edges, request IDs, or reservations;
10. static policy that the placement pickup-entry selection scope calls only
    `interaction_runtime.preview_pick`, never constructs `QueryInput` or
    `MatchInput`, never calls `select_whole_clip`, and never uses a canonical
    snapshot or direct root write;
11. evidence rejection for candidate/order/geometry/score/outcome/selection,
    provider, mutation, external-input, retry, or timing corruption; and
12. the current-pack direct oracle: `R` accepted at `0.666622`, `P_minus`
    `BlockedPath`, and `P_plus` accepted at `0.718336`, with Reach 114 and
    Contact 139.

Focused C++ tests run in normal and release-fast-math builds. Existing matcher,
runtime, collision, scheduler, placement, Python validator, and policy suites
must remain green with unchanged thresholds. Temporary predicate diagnostics are
removed after the oracle is captured.

## Real graphical acceptance

The native 25 Hz placement gate must show:

1. the default-spawn live-flat walk to the common pre-entry region with the
   existing minimum displacement and tick count;
2. paired runtime-owned previews from the same real live-flat snapshot, including
   any honestly recorded transient `PoorMatch` epochs;
3. current-pack `Minus = BlockedPath`, `Plus = feasible/ready`, and deterministic
   `Plus` selection without a canonical snapshot or external match input;
4. ordinary slot-specific left-stick navigation, one-way braking, unchanged
   Reach/yaw/standoff/speed settle bounds, and exactly one Interact;
5. collapsed pickup and placement states exactly
   `Locomotion, Preflight, Align, PickupReplay, Hold, Carry, PlacePreflight,
   PlaceAlign, PlaceReplay, PlaceRelease, Locomotion`;
6. no preview/runtime/registry mutation, retry, reset, root relocation, direct
   root write, or changed collision/IK/matcher threshold; and
7. the existing successful carry, placement, release, seven-frame handoff,
   atomic validated JSONL/PNG publication, and visual continuity checks.

If the real live-flat preview never becomes match-ready, or actual Preflight
disagrees and rejects, the gate exits nonzero with both candidate outcomes and
the live snapshot provenance. It does not fall back to the diagnostic canonical
snapshot or submit a second interaction.
