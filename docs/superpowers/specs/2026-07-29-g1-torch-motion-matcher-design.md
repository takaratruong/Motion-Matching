# G1 In-Process Torch Motion Matcher Design

## Status

Approved conversational design, awaiting written-spec review.

## Purpose

The existing terrain-aware G1 Motion Matching to SONIC path has proved useful
for research, but its C++ matcher server, generated artifacts, coordinate
conversions, and process boundaries make flat-locomotion experiments slower to
change and harder to diagnose than necessary.

The first replacement is deliberately narrow:

> Load Takara-format G1 motion folders directly into PyTorch, perform exact
> flat-locomotion Motion Matching on the GPU, and call the matcher as a normal
> synchronous Python function from the existing SONIC integration.

The first version must respond to forward, backward, lateral, turning, stop,
and direction-reversal commands. It keeps the released GEAR SONIC
policy/controller and MuJoCo backend. It removes only the C++ Motion Matching
server from this execution path.

This is the flat baseline for later terrain, depth, latency, action-chunking,
and learned-kinematics experiments. Those later hypotheses must not complicate
this implementation.

## Constraints established by the current systems

### Takara motion contract

The frozen Takara Walk clip contains 34,863 frames at 50 Hz:

| Field | Required shape | Convention |
|---|---:|---|
| `fps` | `[1]` | exactly `50` |
| `joint_pos` | `[T, 29]` | G1 IsaacLab joint order |
| `joint_vel` | `[T, 29]` | G1 IsaacLab joint order |
| `body_pos_w` | `[T, 30, 3]` | world, Z-up |
| `body_quat_w` | `[T, 30, 4]` | world, wxyz |
| `body_lin_vel_w` | `[T, 30, 3]` | world, Z-up |
| `body_ang_vel_w` | `[T, 30, 3]` | world, Z-up |

The matcher keeps this native representation. It does not remap joints into
MuJoCo order, create Holden Y-up data, or route the database through the
existing terrain-builder artifacts.

V1 freezes one explicit layout identity, `g1-29dof-isaaclab-v1`, derived from
the GEAR G1 body tuple:

- pelvis/root body: index `0`;
- left ankle-roll foot body: index `18`;
- right ankle-roll foot body: index `19`; and
- joint arrays: the 29 non-root G1 joints in IsaacLab order.

Takara NPZ files do not carry body names, so semantic order cannot be inferred
from shape alone. Startup diagnostics must print the frozen layout identity and
indices. Additional files are accepted only under this same export contract;
data in another order must be converted before it is placed in the motion
folder.

### Released SONIC contract

The released G1 encoder consumes ten future reference samples for:

- 29 joint positions;
- 29 joint velocities; and
- root/anchor orientation in wxyz form.

The source offsets are:

```
0, 5, 10, 15, 20, 25, 30, 35, 40, 45
```

At 50 Hz these samples span 45 intervals, or 0.9 seconds. They do not span
1.0 second.

The released encoder does not consume root translation. The matcher still
preserves root position and orientation in its result and future window so a
later root-conditioned encoder can use them without changing the matcher
boundary.

### Runtime state ownership

The matcher owns its previous generated reference state. Its query does not use
measured robot pose, estimated global root position, or measured robot
velocity. Proprioception remains an input to SONIC, not to this first matcher.

This separation is intentional. It tests whether vanilla generated-reference
Motion Matching can steer the released SONIC controller before depth or
measured-state correction is introduced.

## Alternatives considered

### 1. Exact brute-force PyTorch GPU search — selected

Store the normalized feature matrix as one dense tensor and evaluate squared
L2 distance plus `argmin` on the GPU. A local L40S measurement for 34,863 rows
and 31 dummy features was approximately 0.033 ms per exact query. Search is
therefore not the difficult part at the current database size.

This approach has the smallest implementation and diagnostic surface. It
returns the true best row and leaves feature design and transition behavior as
the only important quality variables.

### 2. FAISS, another approximate index, or custom CUDA

An index may become useful when the database grows by orders of magnitude. At
the current scale it introduces dependency, tuning, and nearest-neighbor recall
questions without solving a measured bottleneck. Custom CUDA also restores a
compiled extension boundary that this design is intended to remove.

This option is deferred. The public matcher interface does not expose the
search implementation, so it can be revisited without changing callers.

### 3. Close semantic port of the existing C++ matcher

A direct port would preserve more of the old acceleration, trajectory,
inertialization, and artifact behavior. It would also preserve much of the
complexity being removed, including assumptions tied to the prior database and
coordinate pipeline.

The C++ implementation remains an oracle for specific semantics such as feature
weights, continuation scoring, exclusion ranges, and inertialization. It is not
ported wholesale.

## Architecture

The implementation adds four focused Python modules beneath
`sonic/python/mm_sonic/`:

### `torch_motion_data.py`

`MotionFolder` discovers `motions_dir/**/motion.npz` recursively and sorts paths
by their normalized relative path. It validates and owns each clip.

`TorchMotionDatabase` concatenates valid clips, records clip/frame provenance,
builds validity masks, stores owned native-convention motion tensors, and stores
normalized search features. Runtime code cannot mutate these tensors.

### `torch_motion_features.py`

This module contains the shared feature extractor and native Z-up quaternion
helpers. The database builder and runtime query call the same feature logic.
There is no separate approximate online implementation.

### `torch_motion_matcher.py`

`TorchMotionMatcher` owns:

- selected clip and frame;
- generated planar root transform;
- generated root height and orientation;
- generated joint and feature-pose state;
- shaped command velocity and heading;
- search cadence;
- inertialization offsets; and
- diagnostics from the last committed step.

It advances at 50 Hz and performs an exact search normally at 10 Hz.

### `torch_motion_sonic.py`

`SonicReferenceAdapter` converts an immutable match result into the existing
Python SONIC reference input. It verifies IsaacLab joint order, wxyz
orientation, expected future offsets, and complete publication.

The integration remains synchronous:

```
operator command
    -> TorchMotionMatcher.step(...)
    -> immutable MotionMatchResult
    -> SonicReferenceAdapter
    -> existing GEAR SONIC encoder/controller
    -> existing MuJoCo step
```

Physics advances only after the matcher and adapter have committed one complete
valid reference.

## Public interface

The primary API is stateful and intentionally small:

```python
matcher = TorchMotionMatcher.from_folder(
    motions_dir,
    device="cuda",
)
matcher.reset()

result = matcher.step(
    velocity_world_xy=(vx, vy),
    heading_world_yaw=yaw,
    dt=0.02,
)
```

The existing `CommandSample` adapter supplies its MuJoCo-world XY velocity and
extracts a Z-up yaw from its unit wxyz heading quaternion. The matcher never
receives the robot's measured state.

`MotionMatchResult` is immutable and contains:

- current `joint_position [29]`;
- current `joint_velocity [29]`;
- current `root_position_world [3]`;
- current `root_orientation_world_wxyz [4]`;
- `joint_position_window [10, 29]`;
- `joint_velocity_window [10, 29]`;
- `root_position_window [10, 3]`;
- `root_orientation_window_wxyz [10, 4]`;
- selected clip path and source frame;
- raw incumbent and selected costs;
- whether search and transition occurred; and
- CPU wall time plus CUDA-synchronized search time when diagnostics are
  enabled.

Adding valid NPZ clips changes only database rows and provenance. It does not
change this API.

`reset()` is deterministic. It selects, during database construction, the valid
frame with the smallest joint-velocity squared norm, breaking ties by global
database row. It places that source root at generated world XY `(0, 0)` with
generated yaw `0`, retains the source height/roll/pitch, clears command and
inertialization state, and forces the first non-idle command to search.

`device="auto"` may select CUDA or CPU at startup. The selected device is fixed
for the matcher lifetime; runtime failure never causes a silent device switch.

## Database construction

Every NPZ file must satisfy all of these conditions:

1. every required field exists;
2. `fps` contains exactly one finite value equal to 50;
3. every motion array has the required rank and trailing dimensions;
4. every motion array has the same frame count;
5. the clip has at least 46 frames;
6. all values are finite;
7. every body quaternion has norm within `1e-4` of one; and
8. data is representable as contiguous float32 without non-finite conversion.

Quaternions that pass the tolerance are normalized on the owned copy.
Quaternion signs are canonicalized temporally within each clip by flipping a
sample when its dot product with the previous sample is negative.

The database does not resample clips. A bad file fails the entire startup with
its relative path and exact failed field. Valid files are never silently
skipped.

For each clip, only frames whose complete feature horizon and complete SONIC
window remain inside that clip are searchable. With the final offset at 45,
the searchable source-frame range is `0..T-46`, inclusive. Windows never clamp
or cross from one clip into another.

## Search features

The flat feature vector has 27 values:

| Group | Size | Initial weight |
|---|---:|---:|
| left foot position in current root-heading frame | 3 | 0.75 |
| right foot position in current root-heading frame | 3 | 0.75 |
| left foot world velocity rotated into root-heading frame | 3 | 1.0 |
| right foot world velocity rotated into root-heading frame | 3 | 1.0 |
| pelvis world velocity rotated into root-heading frame | 3 | 1.0 |
| future root XY positions at 0.3, 0.6, 0.9 seconds | 6 | 1.0 |
| future facing XY directions at 0.3, 0.6, 0.9 seconds | 6 | 1.5 |

Positions are translated by the current root position and rotated by inverse
current root yaw. Velocities are rotated but not translated. Future root
positions are expressed relative to the current root in the same heading
frame. Future facing directions use the G1/MuJoCo positive-X forward axis.
All temporary calculations remain Z-up.

Normalization follows the existing matcher semantics per group:

1. compute the mean of each component over searchable database rows;
2. compute each component standard deviation;
3. take the arithmetic mean of those component standard deviations for the
   group;
4. set the group's scale to `group_std / weight`; and
5. normalize both database and query as `(value - component_mean) / scale`.

A required group with zero, non-finite, or non-positive group standard
deviation fails database construction. V1 has no disabled feature groups and
no terrain dimensions.

The online pose and velocity groups come from the previous generated reference,
including any active inertialization offsets. The trajectory groups come from
the shaped command prediction. This is the defining split between current
generated kinematics and desired future motion.

## Command shaping and trajectory prediction

Commands are sampled every 20 ms. Before building the query:

- planar velocity approaches the requested vector with a maximum acceleration
  of `1.5 m/s²` while increasing speed and `2.0 m/s²` while decreasing speed;
- desired yaw follows the shortest wrapped arc with a maximum rate of
  `120 degrees/s`; and
- the same bounded update is simulated forward at 20 ms increments to predict
  root XY and facing at 0.3, 0.6, and 0.9 seconds.

For one bounded velocity update, choose `2.0 m/s²` when requested speed is less
than current speed or the current/requested planar dot product is negative;
otherwise choose `1.5 m/s²`. Move the current velocity toward the requested
vector by at most `acceleration * dt` in Euclidean norm. This same rule handles
deceleration, lateral changes, and reversals without component-wise ambiguity.

The prediction integrates velocity in the world frame and converts the sampled
future points/directions into the generated root-heading frame for the query.
It does not use the robot's physical root estimate.

Normal exact search runs every five calls. Search is forced on the next call
when:

- the command changes from moving to stop;
- the command changes from stop to moving;
- both old and requested speeds are at least `0.15 m/s` and their planar dot
  product is negative; or
- the current source frame has no valid successor.

Speeds at or below `0.05 m/s` are stop commands. These thresholds are frozen v1
defaults and are surfaced in diagnostics/configuration for controlled sweeps.

## Candidate selection

On every call, the ordinary incumbent is the next source frame in the current
clip. If it remains valid, its exact normalized feature cost is computed first.

During search:

- invalid clip-tail rows are masked;
- rows within 20 source frames of the current frame in the same clip are
  excluded;
- every eligible row receives exact squared-L2 feature cost;
- a non-continuation row receives a `0.1` transition penalty in normalized-cost
  units; and
- ties resolve to the smallest global database row.

A transition occurs only when:

```
candidate_feature_cost + transition_penalty < incumbent_feature_cost
```

At a clip end where no incumbent exists, the lowest-cost valid candidate is
required even if it is worse. If no valid candidate exists, the step fails
without modifying matcher state.

The search implementation uses a dense PyTorch reduction. It may process a
query as shape `[1, 27]` internally so later batching does not require rewriting
feature distance code, but v1 exposes only the single-state `step()` API.

## Transition alignment and inertialization

When a new frame wins:

1. align its root yaw to the current generated yaw;
2. translate its root XY to the current generated XY;
3. retain the source frame's native height, roll, and pitch;
4. compute offsets between the previous generated state and the aligned
   candidate state; and
5. decay those offsets with critically damped inertialization.

The initial half-life is `0.10 seconds`, matching the current C++ default.
Offsets cover:

- current and future joint position/velocity;
- full root translation and orientation;
- root linear/angular velocity needed by generated feature state; and
- the pelvis/foot pose and velocity values used by the next query.

Quaternion signs are made continuous before logarithmic-map offset calculation
or interpolation. Every emitted quaternion is renormalized and checked.

The selected source clip is never modified. Alignment and inertialization apply
only to the generated state/result. The future window applies the same
analytically decayed transition offsets at each offset time. Smoothing only the
current sample while exposing an unsmoothed future window is forbidden.

On ordinary continuation, the matcher advances one source frame and transports
that source frame's root delta through the committed planar alignment. It does
not integrate the requested command directly into the emitted reference root.
The command trajectory influences source selection; selected source kinematics
advance the generated reference. Active offsets are then evaluated at the new
time. The generated root remains continuous even when the source clip changes.

## SONIC reference-window assembly

After selection, the matcher gathers the chosen clip at offsets
`0,5,...,45`, applies the committed planar alignment, applies transition
offsets at the corresponding future times, and creates one immutable result.

The adapter verifies:

- window shapes are exact;
- joint order is `g1-29dof-isaaclab-v1`;
- offsets are exactly `0,5,...,45`;
- quaternions are finite unit wxyz values;
- all tensor values are finite; and
- the result sequence is exactly one greater than the last published sequence.

Only then does it publish the reference to the existing SONIC encoder path.
The GEAR checkpoint, controller, observation schema, and MuJoCo dynamics remain
unchanged in this experiment.

## Failure and transaction semantics

`step()` constructs a private next state, validates the complete result, and
only then swaps it into the matcher. Any exception leaves clip/frame, command
filters, inertialization, sequence, and last valid result unchanged.

Invalid commands are rejected before state construction. Velocity must contain
two finite float values, yaw and `dt` must be finite, and v1 requires
`dt == 0.02` within `1e-9`.

The integration treats one matcher result, one SONIC reference publication, and
one physics step as a boundary:

1. compute the match result;
2. validate and publish the complete SONIC reference;
3. request/receive the SONIC action; and
4. advance MuJoCo once.

Failure before the fourth operation does not advance physics. The viewer holds
the last valid rendered state and reports the failed stage. No partial window,
mixed old/new reference, or implicit retry is published.

## Verification

### Loader and layout

Tests must cover deterministic recursive ordering, multiple clips, every
missing field, wrong FPS, wrong shapes, mismatched lengths, short clips,
NaN/Inf conversion, quaternion norms, temporal quaternion signs, fixed G1 body
indices, and exact clip-tail masks.

### Feature equivalence

One shared fixture must prove that offline row extraction and runtime query
extraction produce identical raw and normalized features for the same generated
state. Additional metamorphic tests apply a common world XY translation and
common Z yaw and require identical heading-relative features.

The PyTorch result must match a small independent NumPy oracle for means,
group scales, feature rows, query normalization, and weighted squared-L2 cost.

### Exact search and state

CPU and GPU search must select the same row and cost as the NumPy oracle,
including masks, ties, continuation penalties, local exclusions, forced
reversals, and clip ends.

State tests must cover deterministic reset, ordinary continuation, immediate
start/stop/reversal search, SE(2) alignment, quaternion sign continuity,
bounded inertialization, future-window smoothing, immutable database tensors,
and unchanged state/result after every injected failure.

At a transition, applying the newly created offsets at time zero must
reconstruct the pre-transition generated joint/root state within `1e-5`.
Without another transition, every offset norm must decrease monotonically until
it is below `1e-5`. These are the exact meanings of continuity and bounded
inertialization in this design.

### SONIC boundary

Adapter tests must prove the exact IsaacLab joint order, wxyz convention,
ten-sample offsets, 0.9-second span, sequence monotonicity, and atomic
publication. A process-level integration test must prove that the Python path
does not launch or contact the C++ Motion Matching server.

### Performance

Performance measurements require warm-up and CUDA synchronization around the
measured region. On the L40S with the 34,863-frame Takara database:

- exact-search p99 must be below `1 ms`; and
- complete `TorchMotionMatcher.step()` p99 must be below `2 ms`.

CPU is a deterministic correctness fallback and has no v1 real-time gate.
Database construction is a startup operation and is reported separately from
per-step latency.

### Behavioral canaries

An offline scripted canary runs:

- stand to forward;
- forward to backward;
- backward to left;
- left to right;
- translation while changing heading;
- turn in place;
- moving to stop; and
- stop to moving.

Each command must select source motion with matching signed planar motion or
heading trend, produce finite windows, and keep configured joint/root
discontinuity metrics within the inertialization envelope. Stop and strong
reversal searches occur on the next 20 ms call; other changes occur no later
than the normal 100 ms search boundary.

The final live gate is one 60-second mixed-command flat MuJoCo/SONIC session,
with every movement class held for at least four seconds, and with:

- no fall;
- no crash or frozen viewer;
- no stale-command or sequence error;
- recorded matcher/search latency;
- successful Backspace reset; and
- two clean launch/exit/relaunch cycles.

The same released SONIC policy identity must be recorded before and after the
test.

## Success criteria

V1 is complete when:

1. the motion folder alone is sufficient to build the matcher;
2. the matcher is one normal in-process Python object/function boundary;
3. exact GPU search and complete steps meet their latency gates;
4. flat forward/backward/lateral/turn/stop/reversal commands satisfy the
   offline and live canaries;
5. reference transitions are window-consistent and bounded;
6. the C++ Motion Matching server is absent from this path;
7. released GEAR SONIC policy/controller and MuJoCo remain unchanged; and
8. adding compatible NPZ clips requires no public API change.

## Non-goals

This design does not add:

- terrain or obstacle features;
- depth, lidar, or learned perception;
- measured-state feedback into Motion Matching;
- root-conditioned SONIC training;
- action chunking;
- batched training-environment state;
- approximate nearest neighbors;
- a custom CUDA/C++ extension;
- a new simulator; or
- removal of the existing GEAR SONIC or MuJoCo process boundaries.

Those experiments begin only after this flat reference generator is correct,
fast, and stable.
