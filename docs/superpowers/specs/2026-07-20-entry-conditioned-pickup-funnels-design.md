# Entry-Conditioned Hierarchical Pickup Funnels

## Goal

Replace Smart Pickup's fixed local entry motions with diverse diffusion-generated
final approaches that are conditioned on how the character actually reaches the
object. Long-range travel remains ordinary motion matching. The existing pickup
matcher, contact playback, attachment, Hold, Carry, and placement systems remain
authoritative after the learned approach.

The first playable version targets the known grasp condition and generates 32
candidate root funnels for each attempt. It does not generate full-body poses or
long-range navigation.

## Why Entry Conditioning Is Required

An object/grasp-only model answers, "Which approaches can end at this grasp?"
It can produce diverse routes, but those routes need not agree with the
character's arrival direction or velocity. Connecting the live character to an
unrelated generated first sample adds avoidable motion and can invalidate the
rest of the approach.

The revised model answers, "Given this object, grasp, and observed local entry
state, which final approaches can reach the grasp?" Noise still produces route
diversity, while the entry condition makes that diversity relevant to the live
attempt.

## Hierarchical Runtime

Smart Pickup has two movement stages.

1. **Coarse capture:** Ordinary motion matching steers toward a collision-free
   point in a capture annulus around the object. The initial annulus is
   `0.45–1.00 m` in planar object-relative distance. A character already inside
   the valid annulus skips coarse travel. This stage does not consult authored
   pickup entry slots.
2. **Learned final approach:** After position, facing, and speed settle inside
   the capture annulus, the controller freezes the actual entry state, requests
   32 funnels, certifies them, freezes one winner, and follows it through
   ordinary locomotion controls.

The coarse target follows the character-to-object bearing and is projected into
the annulus, with yaw facing the object. The existing table and obstacle sweep
rejects an unsafe connector. If the radial target is unsafe, deterministic
angular alternatives are checked in increasing absolute angular offset. The
entry state is frozen only after the root reaches this target and settles below
`0.10 m/s`, so braking while the proposal worker runs does not invalidate the
condition. No diffusion resampling occurs within an attempt.

## Model Condition

The existing 18-value object/grasp condition is retained:

- active-hand one-hot;
- object-local grasp translation;
- object-local grasp rotation 6D;
- object-local horizontal approach direction;
- object dimensions;
- support height; and
- grasp height above the support.

Six float32 entry values are appended, producing a 24-value condition:

```text
entry_x_object
entry_z_object
entry_sin_yaw_object
entry_cos_yaw_object
entry_velocity_x_object
entry_velocity_z_object
```

The yaw pair is unit length. Velocity is the observed planar simulation-root
velocity rotated into the object's frame. No absolute world position, world
yaw, object ID, clip ID, frame ID, or full skeletal pose enters the model.

Full-body pose is deliberately excluded because the model predicts only root
motion. The existing matcher and final preview remain responsible for
full-body feasibility.

## Proposal Generation Boundary

A continuously varying live entry condition cannot use one proposal artifact
generated before the application starts. The first playable implementation
therefore uses an asynchronous one-shot proposal worker:

1. At capture, C++ writes one strict request containing the frozen 24-value
   condition, checkpoint identity, and deterministic batch seed.
2. A pinned Python/Torch worker samples exactly one 32-proposal batch and
   atomically publishes a strict binary response artifact.
3. The 25 Hz controller remains in `ProposalPending`, publishes braking
   controls, and polls only for the completed response. It never imports or
   executes Python/Torch in a controller tick.
4. C++ verifies request identity, condition bytes, checkpoint identity, batch
   seed, schema, and proposal certification before entering selection.

The worker is replaceable behind a `FunnelProposalProvider` boundary. A future
ONNX or native C++ inference provider can implement the same request/response
contract without changing controller behavior. Worker launch failure, timeout,
partial output, stale output, or identity mismatch fails the attempt closed.
The first implementation uses a 250-tick, ten-second timeout. A cancelled or
superseded attempt may leave a worker result on disk, but its attempt identity
cannot match a later request.

## Trajectory Representation and Timing

Training examples cover exactly 75 native 25 Hz frames, or three seconds, ending
at the first Reach frame. Examples without the complete history are rejected.
Each example is represented by knots at native offsets
`[0,5,10,15,20,25,30,35,39,44,49,54,59,64,69,74]`. Runtime must not execute
those 16 knots as 16 consecutive ticks; that would compress three seconds into
`0.64 s`.

The diffusion target remains in outward, grasp-to-entry order and is expressed
in the frozen entry root's planar frame:

- position is relative to the entry root;
- yaw is relative to entry yaw as `(sin(relative_yaw), cos(relative_yaw))`;
- outward knots 0–14 are generated by the model; and
- outward knot 15, the entry endpoint, is exactly `(0, 0, 0, 1)`.

Outward knot 15 is excluded from the learned loss and restored exactly after
sampling. The projected trajectory is reversed exactly once into execution
order, guaranteeing that execution sample zero is the frozen live entry state
without requiring a second connector. The 24-value condition still contains
the entry state in the object's frame so the model can reason about the entry-
to-grasp geometry.

For execution, the 16 knots are reconstructed into the object/world frames and
expanded to 75 native targets. Position uses linear interpolation between fixed
knot times. Yaw uses shortest-arc interpolation. The follower publishes one
target per native 25 Hz tick for the original three-second duration.

## Sampling, Certification, and Selection

One stable batch contains exactly 32 proposals generated from 32 deterministic
noise samples under the same frozen 24-value condition. The controller freezes
the batch before movement starts.

Each proposal is rejected if any of the following holds:

- non-finite values or a non-unit yaw pair;
- an execution start knot other than the exact entry identity after the one
  reversal and reconstruction;
- a translation step above `0.08 m` or yaw step above `15 degrees` in the
  expanded 75-tick trajectory;
- table, object, or obstacle intersection along the live-root connector or any
  expanded trajectory segment;
- terminal root incompatible with the selected grasp; or
- failed existing matcher preview at the terminal root.

Survivors are ranked deterministically by route length, heading change,
minimum clearance, matcher cost, and proposal index. Once the controller starts
following a proposal, its identity cannot change. The remaining route is
revalidated against the frozen target and live obstacle geometry each tick.

## Controller and Authority Boundaries

The learned backend owns coarse capture, proposal request lifecycle, route
selection, and final-approach steering. It publishes stick/facing targets only;
it never writes simulation root, displayed root, joints, target registry state,
or attachment state.

After the final target is observed, the backend requests the existing final
pickup preview. A successful preview produces the unchanged `PickRequest`.
The existing runtime then owns motion matching, Reach/Contact playback,
attachment, Hold, Carry, and placement exactly as before.

Authored Smart Pickup remains an explicit fallback provider. Learned evidence
must identify the learned provider and may not silently consult authored entry
slots.

## Failure and Cancellation

The attempt fails closed when:

- no safe coarse capture point exists;
- the learned artifact is missing, malformed, or condition-incompatible;
- all 32 proposals fail certification or preview;
- the target, selected grasp, or frozen scene authority changes;
- a remaining segment becomes blocked;
- a native controller tick is missed;
- tracking exceeds `0.18 m` or `25 degrees`; or
- the user cancels or applies manual override.

Failure never submits a pickup request. Diagnostics preserve the provider,
proposal seed/index, frozen entry condition, rejected-proposal counts, selected
route timing, follower progress, and exact failure reason.

## Training Migration

The current 18-condition checkpoint is not silently reused. Dataset export will
add the six entry values and convert each three-second trajectory to the
entry-relative representation. The compact model input changes from 18 to 24
condition values and is retrained into a versioned checkpoint.

The proposal artifact schema is versioned again to include:

- the 24-value condition;
- fixed knot timing or its frozen schema identifier;
- 32 entry-relative 16-knot proposals;
- seeds and certification flags; and
- enough identity metadata to reject an incompatible checkpoint/runtime pair.

The C++ loader and Python writer share golden byte-layout tests.

## Verification

Focused tests must prove:

- object-frame entry position, yaw, and velocity encoding;
- invariance to a common world translation and yaw rotation;
- exact outward entry identity at knot 15 and execution identity at sample zero
  for every proposal;
- deterministic diversity across the 32 proposals;
- 16-knot to 75-tick interpolation with exact endpoints and shortest yaw arcs;
- coarse capture without authored slot reads;
- certification and deterministic ranking;
- cancellation on tracking, authority, obstacle, and tick violations;
- one final matcher preview followed by one unchanged `PickRequest`;
- actual registry `Held`, `attached=true`, Hold/Carry progression in a headless
  integration test; and
- unchanged authored-provider regressions.

Playable evidence must show at least two different entry bearings reaching the
same grasp with different certified learned funnels.

## Explicit Non-Goals

- Generating long-range navigation with diffusion.
- Generating full-body joint poses.
- Learning obstacle geometry inside the diffusion condition.
- Replacing the existing matcher, contact playback, or attachment authority.
- RGB or GraspMolmo integration.
- Online training or Python/Torch execution inside the 25 Hz controller loop.
