# Full-Walking Existing Pose Utilities Design

## Goal

Make the full-walking terrain viewer move smoothly without the custom per-joint
leg/arm thresholds. Reuse the repository's established transition and terrain
utilities while leaving corpus search, formal evaluation, and published model
artifacts unchanged.

## Scope

This is an interactive diagnostic display change only. The full 9,758,524-row
matcher continues to select and advance canonical source rows exactly as it does
today. Formal smoke/evaluation remains CPU-exact and receives no display filter.

Remove the custom search-jump articulated/leg rejection and arm-slew behavior.
Retain the existing diagnostic mechanical-row filter, terrain-domain hold, and
single-GPU search path.

## Architecture

Add one focused display postprocessor between `HybridMatcher.step()` and the
MuJoCo viewer's `data.qpos` assignment. The postprocessor owns no search state.
It accepts the selected runtime state, source contact pair, and 60 Hz step, then:

1. Converts the selected native qpos to the existing `KinematicPose` contract.
2. Detects a motion-match jump as any row change that is not the next row in the
   same range. At a jump it creates the existing `PoseInertializer` from the
   previously displayed pose to the new source pose, using the repository's
   retained 0.10-second half-life. Successor and repeated neutral frames advance
   the same inertializer; no hand-written joint clamps or slew rules remain.
3. Runs the existing `G1TerrainPoseRepair` on the inertialized pose.
4. Runs the existing `G1TerrainFootLock` with the corpus's left/right source
   contact labels. The foot-lock update is transactional through its existing
   snapshot/restore API.
5. Converts the accepted `KinematicPose` back with the existing `build_qpos`.

The adapter derives body transforms through MuJoCo forward kinematics. It derives
joint/body velocities from consecutive source poses at the authenticated corpus
rate; the first pose uses zeros. These velocity fields serve the existing
inertializer contract and do not alter matching features.

## Failure Behavior

Display corrections must never stall corpus playback:

- If the inertialized pose cannot be repaired, retry the raw selected source
  pose through pose repair, dropping only that transition blend.
- If foot locking rejects, restore its prior snapshot and display the repaired
  source pose without the lock for that frame.
- If the raw source pose itself cannot be repaired, retain the previous display
  pose and expose a diagnostic counter/reason. Matcher source state still
  advances, so the next valid source pose can recover.
- Reset clears inertialization and foot-lock state atomically.

There is no learned-pose fallback, joint clamping, or custom transition gate in
this display path.

## Terrain Model

The render model remains unchanged. Interactive mode creates a separate,
diagnostic collision-oracle model from the same authoritative mesh, with a
`terrain_`-prefixed geom name and enabled collision masks so the existing repair
and foot-lock utilities recognize it. The same authenticated height authority is
exposed by a small `height_at_world_xy` adapter. Formal/evaluation models retain
their current non-colliding geometry and bytes/semantics.

## Alternatives Considered

- Using only `PoseInertializer` is the closest small change to normal Takara, but
  it cannot prevent a planted sole from moving while the blend decays.
- Rebuilding the sharded corpus for `TorchMotionMatcher` would reproduce the
  complete Takara matcher path, but its `MotionFolder`/dense-clip ABI is not the
  9.7-million-row full-corpus ABI and is outside this bounded fix.
- `G1TerrainTransitionGuard` is intentionally a short stair-handoff lock and its
  retained viewer configuration disables continuous source-contact tracking.
  Continuous full-walking display therefore composes the underlying repair and
  foot-lock utilities directly, with the explicit non-freezing fallbacks above.

## Viewer Integration and Evidence

`run_interactive` accepts an optional postprocessor factory. It constructs the
filter only after the interactive model exists and calls it inside each fixed-rate
simulation callback, not merely once per rendered frame. This ensures blends and
contact release continue on neutral input and during render catch-up.

The overlay and interactive receipt identify the exact existing utilities,
half-life, repair/lock acceptance counts, lock-bypass counts, raw-repair failures,
and last failure reason. The title remains diagnostic and explicitly not
acceptance evidence.

## Verification

Test-first coverage will prove:

- a search jump is continuous at the switch and decays on successors and neutral
  frames through `PoseInertializer`;
- the custom leg/arm thresholds are absent;
- source contacts reach `G1TerrainFootLock` in left/right order;
- repair and lock rejection follow the non-freezing fallback rules and restore
  temporal state;
- reset clears all display-filter state;
- the interactive terrain is collision-enabled and utility-recognizable, while
  formal terrain remains unchanged;
- formal evaluator construction never creates or calls the display postprocessor;
- existing runtime/viewer suites remain green.

After automated verification, run one bounded 600-frame ramp route on physical
GPU 5. Require visible row/root advancement, zero learned fallback/clamp, no
display-filter deadlock, and report joint discontinuity plus planted-foot slip.
This route is diagnostic evidence; it does not change formal acceptance.
