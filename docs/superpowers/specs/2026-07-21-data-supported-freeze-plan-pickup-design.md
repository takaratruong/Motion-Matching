# Data-Supported Freeze, Plan, and Pickup

Date: 2026-07-21

Status: approved design; pending written-spec review

## Goal

Pressing `F` freezes the exact current character state while the runtime builds
and certifies a complete current-root-to-pickup plan. Once ready, the runtime
resumes once and executes locomotion, the learned near-object funnel, and the
recorded pickup transition continuously, with no planning or handoff stop.

## Evidence and Scope

The current training set contains 1,548 right-handed funnels sampled over the
74 frames (2.96 seconds at 25 Hz) preceding the `REACH` phase. It does not
contain room-scale approach locomotion:

- Median funnel-entry radius is 0.554 m; the 95th percentile is 0.779 m.
- Only one row reaches 1.0 m, and the maximum is 1.030 m.
- The complete source clips also remain within 1.06 m of the object; there is
  no hidden 2-to-3-metre walking coverage to extract.
- Median entry speed is 0.120 m/s and the maximum is 0.514 m/s.
- Median speed at the `REACH` boundary is 0.040 m/s.
- All rows use the right hand, and their circular-mean object-local entry
  bearing is approximately +86 degrees.

Consequently, the existing diffusion model remains responsible only for the
slow near-object setup represented by its data. It will not be queried from a
2-to-3-metre root or at ordinary walking speed. Room-scale navigation remains
the responsibility of collision planning and motion matching. Left-hand
generation and whole-approach diffusion require new data and are outside this
runtime change.

Production sampling remains at 50 DDIM steps.

## User-Visible Sequence

1. On the rising edge of `F`, capture the displayed root, simulation root,
   current full-body pose, velocities, gait phase, motion-matching frame, and
   relevant interaction snapshot.
2. Freeze simulation and animation on that exact frame. Do not search for or
   transition to an idle animation. Rendering, proposal-worker polling,
   cancellation input, and planning diagnostics continue.
3. Build and certify the complete plan while the captured simulation state is
   immutable.
4. When a complete plan is ready, resume physics and animation once.
5. Execute the locked plan continuously through locomotion, learned funnel,
   and pickup. There is no `AwaitEntry`, terminal settling pause, or late
   matcher preview.
6. If planning is cancelled, times out, or produces no complete valid chain,
   cancel Smart Pickup and restore manual locomotion from the captured state.

The computation freeze is intentional and visible. No simulation time passes
during it, so the character resumes from exactly the pose on which the plan was
conditioned.

## Complete Plan

The frozen planning transaction produces one immutable chain with three
segments.

### 1. Collision-Checked Locomotion

Generate right-handed interaction-entry candidates in the data-supported
region. The initial runtime envelope is 0.45-to-0.75 m object radius and
0.05-to-0.30 m/s planned entry speed. Candidate yaw faces the object, and
candidate bearing is sampled from the existing collision-safe capture
directions, with data-support and path cost included in scoring.

For each entry candidate, compute a collision-free planar path from the exact
frozen root. Use the existing table and obstacle bounds inflated by the route
clearance. Prefer a direct segment when valid; otherwise use a deterministic
visibility graph around inflated obstacle corners. Smooth corners only when
the replacement segments remain collision-free.

Convert the selected polyline into a root trajectory for the existing motion
matching locomotion controller. Preserve the frozen initial velocity and taper
continuously to the candidate's nonzero, data-supported entry velocity. The
planner may slow the character naturally but must not insert a zero-speed
waypoint before the interaction.

### 2. Learned Near-Object Funnel

Condition the current diffusion checkpoint on the selected entry transform,
its planned object-local velocity, the target object, and the selected grasp.
Generate the usual proposal batch and retain only artifact-certified,
collision-free proposals.

The proposal's first execution sample must agree with the planned entry pose.
The locomotion trajectory and learned funnel are concatenated before the
simulation resumes. They are not executed by two independently armed
controllers.

### 3. Recorded Pickup Transition

During the same frozen planning transaction, run matcher preview for each
surviving funnel terminal and select the pickup replay together with the
locomotion and funnel segments. A chain is not ready unless its final pickup
preview is feasible and match-ready.

Retain the preview's `MatchCandidate` and `CertifiedPickupSourceIdentity` in a
learned-only certified submission token. Extend the learned submission seam so
the interaction runtime can consume that token after checking target,
affordance, object transform, terminal-root tolerance, and pickup-source
identity. It must not discard the frozen selection and run a new matcher search
that could introduce a late planning state. Existing authored `PickRequest`
behavior remains unchanged.

During execution, transition directly into that selected pickup at the funnel
terminal. Remove the current three-tick terminal-settle requirement and final
late preview from this path. The playback inertializer blends the live terminal
pose into the certified pickup clip; if the live root has escaped the previewed
position or yaw bounds, fail before submission instead of stopping to replan.
The terminal can retain the slow motion present in the demonstrations, but the
runtime must not impose an additional stop.

## Selection

Score complete chains, never isolated segments. The deterministic score order
is:

1. complete collision and matcher feasibility;
2. distance from the entry condition to the measured data-support envelope;
3. locomotion path length and corner severity;
4. locomotion-to-funnel position, yaw, and velocity seam error;
5. funnel route length and terminal heading error;
6. pickup matcher preview cost;
7. proposal index as the final stable tie-break.

The selected chain freezes its target identity, object transform, obstacle
snapshot, entry transform, proposal identity, route, and pickup preview. Any
material authority or geometry change before execution begins invalidates the
transaction and returns control to the user.

## Runtime State and Timing

Replace moving prefetch with these learned-backend states:

- `FrozenPlanning`: immutable simulation state; launch and poll generation.
- `FrozenSelectionPreview`: evaluate complete-chain matcher previews while the
  simulation remains frozen.
- `PlannedChainFollow`: resume once and follow the unified root route.
- `PickupTransition`: submit the preselected pickup without terminal settling.
- Existing submitted, failed, and cancelled terminal states remain.

Simulation ticks, motion matching, physics, interaction playback, and
animation clocks do not advance in frozen states. A separate render/planning
frame counter drives worker timeout and polling. The initial timeout remains
10 seconds (250 render frames at the 25 Hz presentation target).

`WASD`, `X`, reset, target deletion, or target mutation during planning aborts
the worker, clears the transaction, and resumes the captured manual state.

## Unified Route Follower

Generalize the spatial funnel follower to consume an immutable variable-length
execution route. The route begins at the exact frozen root, includes the
planned locomotion segment and the learned funnel, and ends at the preselected
pickup seam.

Progress is determined by monotonic projection onto the remaining route, with
the existing lookahead steering behavior. Collision validation covers the
remaining complete route. Debug rendering shows:

- white: frozen/current tracked root;
- blue: complete locomotion-plus-funnel route;
- cyan: locomotion-to-diffusion seam;
- magenta: pickup transition root;
- green/orange: progress and lookahead.

## Failure Handling

- No collision-free supported entry: unfreeze and report no safe approach.
- Proposal launch, worker, identity, or timeout failure: cancel the worker,
  unfreeze, and return manual control.
- No proposal with a feasible pickup preview: unfreeze and report no complete
  pickup plan.
- Route invalidation before resume: discard the plan and unfreeze.
- Tracking failure after resume: fail closed through the existing cancellation
  path; do not teleport the character or object.

No failure leaves physics frozen after the learned pickup transaction ends.

## Verification

- A planning-frame test proves that root, pose, velocities, animation frame,
  physics state, and simulation tick remain bit-identical while the worker is
  pending.
- A request made from 3 m is conditioned on a selected 0.45-to-0.75-m entry,
  not the frozen 3-m root.
- Planned entry speed remains within 0.05-to-0.30 m/s and is nonzero.
- The unified route starts at the exact frozen root and contains both the
  collision-planned approach and selected learned funnel.
- No locomotion update occurs until proposal generation and pickup preview are
  both complete.
- Resume occurs once; subsequent outputs never request a stationary constraint
  before pickup.
- Pickup submission occurs at the certified terminal without three settling
  ticks or a late preview.
- Cancellation, timeout, target mutation, and no-valid-chain cases all
  unfreeze and restore manual control.
- Existing artifact identity, route-collision, Carry, attachment, placement,
  and object-authority tests remain green.
- A live 2-to-3-metre test visibly freezes on `F`, draws the complete route,
  then resumes through approach and pickup without an imposed intermediate
  stop.

## Deferred Work

- Collect paired room-scale locomotion-to-pickup data and train a unified or
  hierarchical long-horizon model.
- Add left-hand funnels and rebalance object-local approach bearings.
- Replace process-per-request inference with a resident model worker to reduce
  the intentional computation freeze.
