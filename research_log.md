# Terrain-aware motion-matching research log

## 2026-08-01 baseline diagnosis

- The earlier route canary completed flat → stair → platform → stair → flat,
  but its reported entry support-foot mismatches were 18.34 mm and 29.90 mm.
- The prior runtime guard measured root support and foot-centre height, not
  collision of the articulated G1 foot/body geometry with the staircase.
- A user-observed failure allowed a selected sideways flat clip to enter the
  first tread even though the sparse command preview was safe.  Flat matching
  is now prepare/validate/commit transactional and validates the selected
  dense pose preview; focused tests and straight/adversarial canaries pass that
  guard.
- Remaining visible stair penetration is therefore being split into:
  (A) direct source/reference versus its canonical staircase and
  (B) transition, placement, inertialization, and matching errors.

## 2026-08-01 clean c490/GRAIL expansion

### Source convention and importer

- The c490 corpus was audited from paired clean PKL and USD sources. No
  SONIC-tracked/noised rollout row is used as an authored terrain pose.
- G1 sole spheres resolved the source/world convention. Root orientation and
  body angular velocities are quaternion-derived; the importer does not reuse
  a mislabeled Euler or quaternion component.
- Stair translation is fitted without scale or pose warping. A shadowed
  traversal variable had made ascent translation check the beginning rather
  than the final stable endpoint; a regression test reproduced and fixed it.
- Descent is not synthesized by reversing ascent. Five genuine descending
  sources survived the complete G1 audit:
  `1086`, `1719`, `1814`, `1846`, and `1911`.
- Long descents produce two different clean crops:
  - `entry`: the first three drops, preserving the top approach;
  - `landing`: the last three drops, preserving the real bottom landing.
- Every retained clip has an analytic left/right mirror. Mirroring transforms
  root/body pose, history, velocity, contacts, trajectory, and terrain
  placement together.

### Mechanical publication

- Raw expanded export: 38 c490 clips from 14 source stems.
- Accepted after complete-frame repair/audit: 28.
- Rejected and excluded: 10 (`2220`, `2599`, `3487`, `3574`, and `3724`, plus
  mirrors) because they exceeded the 5 mm sole or zero-forbidden-collision
  contract after bounded repair.
- Published archive:
  `artifacts/terrain_catalog/justin_grail_c490_repaired_v9.zarr`
  (57 clips, 18,217 frames).
- Published catalog:
  `artifacts/terrain_catalog/justin_grail_c490_repaired_v9_entry45.npz`
  (16,849 searchable rows).
- A zero-trim re-audit of the final merged archive accepted all 28 c490 clips
  without another repair: maximum complete-frame sole penetration 4.993 mm,
  zero forbidden penetration. Report:
  `artifacts/terrain_catalog/c490_final_reaudit_v9/repair_report.json`.
- The first 9,631 feature/source/pose/contact/tread/future-command rows and the
  feature scale are bit-identical to v5. Exactly 720 derived `entry` booleans
  differ because the catalog was deliberately rebuilt with a 45-frame
  reviewed entry window; v8-entry45 and v9 have identical baseline labels.
  The v9 result therefore measures appended coverage, not a normalization
  change.

### Runtime lifecycle defects found

1. Complete c490 route clips often begin inside the reviewed entry window.
   `StairHandoffIndex` subtracted another lookback and silently contributed no
   entry rows. The index now retains reviewed frame-zero entry rows.
2. The landing preview used a nominal 35 mm ankle-to-sole proxy after exact
   G1 pose repair was already available. On the platform this reported a false
   7.5--9.1 mm penetration and rejected every flat bridge forever. Landing
   mode now retains the dense support-edge path gate and delegates sole
   clearance to the immediately following exact articulated repair.
3. The second fix changed `up_pos_30` from a permanent 470-frame landing freeze
   to a 55-frame bridge and a successful flat handoff.

Neither lifecycle fix relaxes the forbidden-body collision gate or lets flat
motion cross an unsupported edge.

### Canonical comparison

Like-for-like route matrix:

| Catalog | Entered | Exited | Forbidden collision frames | Vertical riser-contact frames |
|---|---:|---:|---:|---:|
| v8 / iteration 90 | 10/14 | 8/14 | 0 | 864 |
| v9 / iteration 97 | 13/14 | 13/14 | 0 | 3 |

The new matcher selected genuine c490 descents on four downhill routes:

- `down_neg_30` → `1086__landing__mirror`;
- `down_neg_60` → `1086__landing`;
- `down_pos_30` → `1814__mirror`;
- the wider ±90/positive-60 cases retained safer older candidates when they
  scored better.

The remaining failure is `up_pos_90`. It reaches a collision-free staging pose
about 0.51 m before the first riser, but every pose/yaw-compatible entry has an
incompatible outgoing foot phase. Loosening only the scripted lateral corridor
did not create a valid handoff and was discarded. The next fix is more
phase-compatible pre-entry data or bounded contact-preserving warping, not a
larger collision/contact tolerance.

### Visual audit

- Actual G1-mesh videos and dense 50 fps foot windows were inspected for the
  c490 `down_pos_30` and `down_neg_30` routes.
- Both clear all three treads and return to flat ground without a fall,
  scissoring collapse, multi-centimetre penetration, or body collision.
- `down_neg_30` is cleaner. Both remain conservative, high-kneed, and
  pause-heavy; they are useful clean data but not yet polished game animation.
- Full-route source-contact IK was tested on `up_neg_90`, `up_pos_30`, and
  `down_neg_30`. It reduced some authored-contact displacement but did not
  solve the entry blend/contact-label artifact and was not promoted.
- A contact-only “exact stance slide” metric was also rejected: moving swing
  toes can touch a tread and are not planted stance. Dense video plus
  penetration/contact-normal diagnostics remain the current gait audit.

### Retained evidence

- Route map:
  `artifacts/terrain_autoresearch/iteration97_v9_full14_canonical/routes_intent_ball_actual.png`
- True-descent G1 videos:
  - `artifacts/terrain_autoresearch/iteration95_v9_rendered/down_neg30/waypoint_down_neg_30.mp4`
  - `artifacts/terrain_autoresearch/iteration95_v9_rendered/down_pos30/waypoint_down_pos_30.mp4`
- Dense foot sheets live beside those videos.
- The Python viewer and `launch_terrain_mm_kinematic_interactive.sh` now both
  default to the v9 archive/catalog. A final integration test caught and
  removed the launcher's stale v5/v3 override; the default headless canary
  passes.
- The same review caught a browser/evaluator parity bug: only the evaluator
  deferred the nominal sole proxy during an exactly repaired landing. Both
  paths now use one shared mode predicate. A fresh `up_pos_30` rollout entered
  and exited, completed a 55-frame landing bridge, had zero commanded frozen
  frames and zero forbidden-collision frames:
  `artifacts/terrain_verification/v9_postreview_up_pos30/aggregate.json`.
- Final focused verification: 119 tests plus 5 unittest subtests pass.

## 2026-08-02 privileged exact-geometry bootstrap

### Whole-route geometry transfer

- Added an offline, clean-kinematic stair transfer that consumes an authored
  source motion, an exact target mesh, and a requested world-XY route. It does
  not read target root, body, or joint samples after the evaluator supplies
  the route endpoints.
- Corresponding support plateaus define a piecewise route warp. Full-body
  timing and non-leg motion remain authored; bounded leg-only IK reconciles
  exact G1 sole-sphere targets.
- Candidate retrieval is automatic. Archive rise/tread/step metadata is only a
  prefilter; candidates must pass final IK residual, root-lift, vertical sole,
  nearest-triangle/riser, and stance-support gates.
- The post-lift residual bug was removed: any sub-5 mm clearance repair is
  applied to both the root and target scaffold before final IK and audit.
- A collision-free foot hanging beside the mesh no longer counts as supported.
  Source kinematic stance frames require target sole-bottom support from either
  the exact mesh or the explicit ground plane at contact height.

### Verified clean canaries

- Automatic ascent: source clip 13 onto target mesh 37, 499 frames at 50 Hz.
  No target motion samples are used. Maximum leg correction is 0.35458 rad,
  final foot residual 0.276 mm, triangle/sphere penetration 1.500 mm, and root
  clearance lift 3.325 mm.
  Artifact:
  `/move/data/terrain-aware/motion-matching/stairs500-geometry-warp-auto-target037-v2`.
- Automatic descent: source clip 215 onto target mesh 261, 499 frames at 50 Hz.
  Maximum leg correction is 0.13421 rad, final foot residual 0.249 mm,
  triangle/sphere penetration 1.428 mm, and root lift 0.494 mm.
  Artifact:
  `/move/data/terrain-aware/motion-matching/stairs500-geometry-warp-215-to-261-v1`.
- Both full G1 mesh videos were reviewed at route scale and in dense transition
  windows. Neither falls, freezes, crosses its feet, or visibly intersects the
  staircase. These remain kinematic canaries, not tracker qualification.
- The ascent has eight frames (409--416) where an authored right stance foot
  retains only one of four conservative sole support samples on the short top
  tread. It passes the one-point non-hanging gate but correctly fails the
  optional two-point strict gate. The descent satisfies the strict check.

### Automatic one-riser composition

- The representative bank contains 58 automatically segmented semantic units:
  38 ascent and 20 descent fragments, with entry/middle/exit roles and exactly
  one adjacent tread transition per unit.
- A layered dynamic program now selects exactly one fragment per target riser.
  It filters direction, role, signed rise/run, and existing pose/velocity/
  support-phase seam mechanics. Fragment IDs may repeat; there is no rule such
  as “stair step 3 uses clip X.”
- On the exact target-37 route it selects six units across clips 8 and 37. On
  target 261 it reuses three compatible middle-step units from clip 261 to
  cover all six drops. Both routes are supported under the default geometry
  tolerances.
- The remaining implementation step is reconstruction: build one continuous
  authored trace from the selected units, warp that synthetic support sequence
  onto target footholds, rerun bounded IK/collision/support audits, then expose
  only the accepted result in the interactive viewer.

### Continuous one-riser reconstruction

- The previous per-target source choices in this log are obsolete. The final
  reusable path loads the precomputed exact fragment bank, excludes the target
  clip automatically, ranks complete entry/middle/exit plans, reconstructs each
  plan, and retries the next rank after a mechanical rejection. Fragment roles
  and source ranges are derived from contacts and exact source geometry; stair
  numbers are not mapped to clips by hand.
- Reconstruction now uses spatial route intervals to assign repeated support
  heights, separate support classification from strict collision clearance,
  smooth root-anchor corrections through time, release stance guidance around
  liftoff, and fit bracketed swing paths. Hard rejection covers stance and
  swing residuals, sole and exact triangle/sphere penetration, minimum stance
  support, root translation/rotation/acceleration, inter-frame joint motion,
  fragment seams, and source-switch seams.
- Held-out target-37 ascent passes the original strict 50 mm swing limit at
  ranked plan 3. It contains 489 frames at 50 Hz and selects clips
  `407, 76, 76, 36, 3, 26`; target 37 is absent. Maximum stance/swing residuals
  are 0.342/19.88 mm, sole-sphere penetration is 2.112 mm, minimum stance support is
  two points, root translation/rotation steps are 19.47 mm/2.12 degrees, root
  acceleration is 18.87 m/s^2, maximum joint step is 0.227 rad, and maximum
  fragment/source-switch seam step is 0.127 rad.
- Held-out target-261 descent passes all hard mechanics at ranked plan 0. It
  contains 442 frames at 50 Hz and selects six independent ranges from clip
  `171`; target 261 is absent and no target pose data is used. Maximum stance
  residual is 0.249 mm, sole-sphere penetration is 0.380 mm, root
  translation/rotation steps are 25.20 mm/3.03 degrees, root acceleration is
  20.20 m/s^2, and maximum joint/seam steps are 0.205/0.092 rad. The one relaxed
  quantity is swing guidance: 83.85 mm under a 110 mm swing-only soft limit;
  stance, collision, support, and continuity limits are unchanged.
- The original four-sphere audit missed toe/heel and pitched-foot visual-mesh
  intersections. The retained audit now collides the complete G1 geometry
  against one thin convex prism per exact terrain triangle; a single MuJoCo
  terrain mesh is invalid here because its convex hull bridges the stair
  treads. The ascent/descent full-foot maxima are 3.112/1.276 mm, respectively,
  with zero forbidden-body contacts and no threshold-exceeding frames.
- Swing-foot clearance uses a complete oriented foot envelope and phase-bounded
  temporal ramps. Planted-foot search carries compact extrema from the actual
  authored mesh, rejects same-tread penetration, and can align a pitched heel
  to the tread while retaining sole-probe support. Swing scaffolds still
  reserve worst-case IK error; planted scaffolds use the declared final
  penetration bound and are checked again after IK.
- G1-mesh videos and dense sheets show both routes clearing all six risers and
  ending upright with no freeze, fall, foot crossing, shift-hop, pose teleport,
  or visible stair penetration. The ascent is brisk/high-kneed and has a mild
  upper-step knee/leg snap; the descent is conservative and high-kneed. These
  are mechanically valid kinematic canaries, not polished gait or SONIC
  tracking qualification.
- Retained evidence:
  - ascent:
    `artifacts/terrain_fragment_reconstruction/target037-fullfoot-v7`;
  - descent:
    `artifacts/terrain_fragment_reconstruction/target261-fullfoot-v6`.
- Each retained directory includes `full_body_collision_audit.json`, a
  `route_and_mechanics.png` diagnostic, seam/dense contact sheets, and a
  full-rate `g1_kinematic_50fps.mp4` render.
- Final focused verification: 52 stair-oracle tests, 7 reconstruction-CLI
  tests, and 2 reference-stitch tests pass. The changed modules compile,
  scoped `git diff --check` is clean, both MP4s decode at 50 Hz with the exact
  source frame count, and both retained full-body audit JSON files pass.

## 2026-08-03 final held-out riser composition

- Compact one-riser exact-terrain kernels cover 59/60 isolated transitions
  under the unchanged 5 mm complete-foot and zero forbidden-body gates.
- Deeper phase pools repair target 8. Mid-landing kernel boundaries repair the
  metre-wide landings on target 4 without inserting a flat bridge.
- The lone unsupported clip-39 transition becomes feasible inside a coherent
  three-riser block selected from non-boundary fragment roles. The block audit
  is 0.986 mm foot / 0 body penetration; its full seven-riser composition is
  3.965 mm / 0.
- The final deterministic suite is 12/12 accepted. Across 3,717 frames, the
  worst route has 4.743 mm complete-foot penetration, zero forbidden-body
  penetration, 42.13 mm root translation step, 0.224 rad root rotation step,
  0.232 rad joint step, and 37.82 m/s^2 root acceleration.
- All twelve G1 MP4s were sampled from start through finish. The robot clears
  every staircase and ends upright; no sampled route freezes, falls, or visibly
  enters a tread. Motion remains conservative/high-kneed and is not yet a
  SONIC tracking result.
- Evidence and review figures:
  `artifacts/terrain_fragment_composition/heldout12_phasebeam_final_v14`.
  The directory also contains one labelled 74.34-second, 50 Hz review reel of
  all twelve routes.

## 2026-08-03 coherent-warp profile-ranking A/B

- Hypothesis: replacing stale scalar rise/tread/step metadata with ray-cast
  physical stair profiles would improve leave-one-out coherent traversal
  coverage.
- The fixed 12-geometry suite requested two independently accepted whole-route
  warps per target from the first 12 ranked candidates. Nominal and exact
  ranking both accepted ten total warps and passed 3/12 targets.
- Exact ranking averaged 10.25 attempted candidates versus 9.67 for nominal;
  first-warp joint correction was 0.235 versus 0.228 rad. It therefore was not
  enabled as the production default.
- The rejection traces expose a more specific representation mismatch: a
  source and target can have equal physical mesh level counts while their
  actual root-endpoint routes sample different support-level counts. The next
  iteration ranks the exact motion-conditioned support routes that the warper
  consumes, rather than the entire mesh profile.
- Evidence: `artifacts/coherent_warp_coverage/heldout12_ab_v1`.

## 2026-08-03 motion-conditioned route ranking

- The prefilter now samples the exact support route between every motion's
  actual root endpoints, using the same terrain query and sole footprint as
  the geometry warper. It compares per-route level count, tread widths, and
  rises rather than whole-mesh or scalar metadata.
- On the fixed two-warp/12-candidate suite this raises strict coverage from
  3/12 to 4/12 and accepted coherent warps from 10 to 12. Mean attempted
  candidates improves slightly from 9.67 to 9.58. Maximum accepted
  sphere/triangle penetration remains 3.775 mm.
- The route ranking becomes the default for future environment-specific
  coherent asset caches. It does not restart or alter the active viewer.
- Eight targets still lack two coherent whole-route styles. This is evidence
  to proceed to source-contiguous multi-riser blocks, not to relax mechanical
  bounds or increase the candidate limit indefinitely.
- Evidence: `artifacts/coherent_warp_coverage/heldout12_route_v2`.

## 2026-08-03 deeper coherent-route search

- The eight two-style failures were rerun with up to 50 route-ranked whole
  traversals. Targets with fewer than 50 compatible routes exhausted all of
  them. No target gained a second mechanically accepted warp and the four
  zero-warp targets remained at zero.
- Rejections are dominated by excessive root-clearance lift and foot-target
  error. The 4/12 whole-route ceiling is therefore not a shallow retrieval
  problem; increasing the search budget is discarded.
- Evidence: `artifacts/coherent_warp_coverage/heldout12_route50_v3`.

## 2026-08-03 source-contiguous multi-riser coverage

- Sequential stable-step fragments from each original source are collapsed
  into longer chronological blocks while retaining every authored frame
  between their endpoints.  The matcher can therefore reuse a one-, two-,
  three-, or four-transition excerpt without pretending each riser is an
  independently named animation.
- On the fixed 12-route suite, longest-first block coverage succeeds on 11/12
  targets.  Six routes use a single four-transition-or-shorter block and have
  zero internal seams; five longer routes use two blocks and one seam.  The
  maximum accepted triangle/sphere proxy penetration is 4.360 mm and no
  mechanical gate was relaxed.
- Clip 39 remains the only miss under the general block warper.  Its known
  accepted fallback is the previously audited four short kernels followed by
  one coherent three-riser block; the new general result is not allowed to
  erase that working case.
- These are independent block gates, not yet complete route acceptance.  The
  next stage searches multiple exact-mesh-safe gait phases per span, composes
  them at the shared landing, and reruns temporal plus full-G1 collision gates.
- Evidence: `artifacts/coherent_block_coverage/heldout12_blocks_v2`.

## 2026-08-03 exact phase-pool block composition

- Every selected target span now searches multiple proxy-feasible source
  blocks and retains up to four that also pass complete-G1 collision after a
  generic smooth visual-foot clearance envelope.  Candidate chains are ranked
  by raw landing pose/velocity compatibility, then tested across critically
  damped blend times under the full temporal and exact-mesh gates.
- Ten of twelve development targets pass directly.  Six use one uninterrupted
  block and four use two blocks with one seam.  The maximum exact foot
  penetration is 4.998 mm, forbidden-body penetration is zero, the largest
  smooth clearance correction is 11.09 mm, maximum root acceleration is
  23.71 m/s^2, and maximum seam root acceleration is 16.55 m/s^2.
- Dense 25 fps full-body and foot sheets cover all four new seams.  None shows
  a source-switch teleport, one-foot hop-hold, or planted-foot double-tap.
  Target 436 is borderline rather than polished: its 4.998 mm margin and
  0.06-second join are the sharpest in the retained set.
- Clip 369 rejects the long block because the visible foot meets a vertical
  riser; clip 39 has no general proxy plan.  Both remain accepted through the
  previously validated short-kernel/multi-riser fallback, so the hierarchy is
  still 12/12 without relaxing a collision bound.
- Evidence:
  `artifacts/coherent_block_composition/heldout12_phasepool_clearance_v3`.

## 2026-08-03 disjoint geometry50 validation launch

- Fifty new targets, balanced 25 ascent / 25 descent, were selected by the
  same deterministic farthest-point geometry rule after excluding every
  development target and the two original canaries.
- CPU-only coverage job 16465281 is chained to exact composition/render job
  16465284 and aggregate job 16465286.  The final report will be written to
  `artifacts/coherent_block_validation/geometry50_aggregate_v1.json`.

## 2026-08-03 disjoint geometry50 coherent-block result

- Proxy retrieval covers 48/50 fresh stair geometries.  Exact composition
  accepts 38 in the base search and two more in the failure-only depth-128
  search, yielding 40/50 overall and 40/48 of proxy-supported targets.
- The exact accepted set has at most 4.978 mm complete-foot penetration, zero
  forbidden-body penetration, 39.37 m/s^2 maximum root acceleration, and
  19.00 m/s^2 maximum seam root acceleration.  No bound was relaxed.
- Dense 25 fps review covers the worst accepted ascent seam, descent seam,
  overall root acceleration, the depth-search recovery, and both joins of the
  sole accepted three-block route.  The joins are visually continuous with no
  teleport, one-foot hop-hold, or planted-foot double-tap.  Motion remains
  conservative/high-kneed rather than polished.
- The two proxy misses and eight exact misses are retained as failures.  A
  filtered 53-kernel compact-riser fallback is running only for those ten
  geometries; it reuses the already accepted hierarchy rather than increasing
  the collision or mechanics envelope.
- Evidence:
  `artifacts/coherent_block_validation/geometry50_aggregate_deep_v2.json`.

## 2026-08-03 complete geometry50 terrain hierarchy

- The compact-riser fallback was applied only to the ten coherent-block
  failures.  A reporting inconsistency was fixed: compact routes now receive
  the same bounded smooth exact-mesh clearance pass as coherent blocks, with
  mechanics rechecked afterward rather than relaxing the 5 mm foot gate.
- Seven compact routes pass.  Together with 40 coherent-block routes, strict
  leave-one-out geometry coverage is 47/50.  Maximum complete-foot
  penetration is 4.978 mm, forbidden-body penetration is zero, maximum root
  acceleration is 39.37 m/s^2, and maximum seam root acceleration is
  39.22 m/s^2.
- Auditing every clean target clip on its own mesh shows only 10/50 pass
  unchanged, but 36/40 failures are foot-only clearance mismatch.  The same
  bounded smooth repair raises exact-source validity to 42/50.  Combining
  those with five coherent substitutes and two compact routes gives 49/50
  for the intended globally privileged/source-covered setting.
- Clip 431 is the sole source-covered miss.  Its clean 499-frame animation is
  temporally smooth, but exact collision finds 45.9 mm foot and 43.1 mm
  forbidden-body penetration on its own 24.9 cm narrow staircase.  Neither
  source-inclusive nor leave-one-out compact search finds a safe replacement.
- Dense 25 fps review of the worst source lifts and the highest-acceleration
  compact seams shows no teleport, hop-hold, or visible terrain hovering.  The
  retained kinematics remain conservative/high-kneed rather than game-ready
  polished.
- Evidence:
  `artifacts/coherent_block_validation/geometry50_hierarchy_aggregate_v1.json`,
  `artifacts/coherent_block_validation/geometry50_source_covered_hierarchy_v1.json`,
  `artifacts/archive_identity_traversal/geometry50_repaired_v1/aggregate.json`,
  and `artifacts/coherent_block_validation/geometry50_hierarchy_coverage_v1.png`.

## 2026-08-05 held-out waypoint steering and reusable portals

- A fully oblique three-leg S route is a genuine current coverage failure: all
  twelve C490 ramp candidates reject on foot-target or root-clearance gates.
  The route is retained as a negative benchmark rather than weakening those
  gates.
- The global game-style hierarchy now permits arbitrary waypoint curves on
  flat support while aligning non-flat portal legs to the terrain gradient.
  Accepted legs resume from disk, and already-qualified primitives can be
  rigidly instantiated when a coarse exact-mesh support-height signature
  matches within 3 mm.  The final portal composer remains the sole expensive
  whole-course collision authority.
- The first held-out route spans 10.76 m, 1.50 m laterally, and four heading
  changes.  Its shifted ramp and shifted up/down curb both compile to accepted
  MotionBricks portal courses.  Maximum complete-foot penetration is 2.03 mm
  on the ramp and 2.78 mm on the curb; forbidden-body penetration is zero.
- Whole-clip 48-frame sheets and dense 25 fps entry/exit windows show upright,
  continuous alternating motion without a source-switch teleport or freeze.
  The curb remains the visually riskier event because its selected trial uses
  a 0.637 rad contact-retarget correction and a high step.
- A separate abrupt-command curriculum adds 102 canonical traces and 204
  physical clips after exact sagittal mirroring.  It covers velocity, yaw, and
  simultaneous two-stick reversals, stop/restart, and speed jumps without
  changing the retained smooth omnidirectional curriculum.  The source corpus
  now marks each hard event with a mirrored-invariant bit mask so those windows
  can be explicitly weighted during distillation.
- The retained live route captures both portals and the terminal waypoint.  It
  travels 14.56 m with 2.95 cm RMS and 6.15 cm p95 deviation from the intended
  route; all flat travel is generated live.  Evidence:
  `artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/live_rollout_seed17_v4_forward_nudge`.
- Dense exit review exposes a brief MotionBricks reacquisition crouch after
  each committed course.  Continuing the measured exit tangent and forcing the
  first continuation into `walk` both preserve completion but do not remove
  it.  Those variants are rejected as visual fixes; the remaining issue is a
  pose/state-distribution handoff, not waypoint steering.
- CPU job 16514194 materialized the separate abrupt bank in 3m36s using two
  CPUs and under 1 GB resident memory.  It contains 204 mirrored clips and
  122,400 frames: 816 large one-frame events, including 408 stop/restarts, 136
  near-180-degree travel reversals, 408 heading jumps of at least 90 degrees,
  and 340 simultaneous two-stick jumps.  All twelve SONIC input shards are
  prepared; GPU noising remains gated on visual/trajectory acceptance.
  Evidence:
  `/move/data/terrain-aware/sonic-rollouts/takara_bones_mm_abrupt_v1_task12`.

## 2026-08-05 raised-support MotionBricks continuation

- The post-terrain crouch was not primarily a waypoint or gait-phase problem.
  Both authored courses end on a support surface 0.25 m above MotionBricks'
  training floor, and the unnormalized continuation lowered its pelvis by
  approximately 0.33 m while returning toward floor-relative training poses.
- The retained interface subtracts the current support height from the four
  context root-z values before MotionBricks generation, then restores that
  support height to each newly generated batch exactly once.  An initial
  implementation restored it on cached batches too and accumulated 0.25 m per
  controller tick; that failed rollout was rejected and the ownership test is
  now based on whether a new qpos batch was actually emitted.
- The corrected v9 rollout captures both portals and reaches the final
  waypoint.  Its two selected continuations lose only 15.64 and 10.43 mm of
  pelvis height, versus roughly 0.33 m before support normalization.  Dense
  25 fps review around both exits shows continuous alternating steps and only
  a mild ordinary knee bend, with no deep squat, fall, freeze, or pose snap.
- Route tracking remains accepted but is slightly less precise than v4:
  5.43 cm RMS and 16.48 cm p95 deviation, with 13 terrain-guarded frames.
  This trade is retained because the route completes and the visible
  distribution seam is materially removed; waypoint tuning did not solve it.
- Evidence:
  `artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/live_rollout_seed17_v9_support_once_move`.
