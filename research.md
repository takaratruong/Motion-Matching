# Terrain-aware G1 motion matching

## Objective

Produce clean G1 kinematics that can walk freely on flat ground, approach the
Justin staircase from useful joystick headings, ascend and descend, and leave
the staircase without freezing, foot sliding, body/tread intersection, or
visible pose discontinuities.  Runtime matching must consume only the current
G1 state, two-stick intent, and a robot-centred causal terrain observation.
Known global scene coordinates may be used only by the offline evaluator and
the exact-geometry bootstrap safety oracle.

The current milestone deliberately relaxes that final information boundary:
first reproduce the polished globally informed terrain-matching behavior on
G1, then replace privileged scene pose with causal robot-centred perception.
This staged bootstrap prevents perception uncertainty from masking animation,
search, blending, or contact defects.

## Current verified baseline (2026-08-01)

- Flat bank:
  `/move/data/terrain-aware/motion-matching/takara_bones_walk_support_v2_startstop`
- Terrain archive:
  `artifacts/terrain_catalog/justin_grail_c490_repaired_v9.zarr`
- Runtime catalog:
  `artifacts/terrain_catalog/justin_grail_c490_repaired_v9_entry45.npz`
- Reference feature scale:
  `artifacts/terrain_catalog/justin_grail_repaired_v5_intended_contact12mm_core11.npz`
- Scene source:
  `/move/u/justingu/Projects/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/up_pos_30/g1_29dof_spherehand_w_multi_boxes.xml`
- Runtime controller: `mm_sonic.hybrid_terrain_interactive`
- Browser/Switch viewer: `mm_sonic.terrain_interactive_viewer`
- Canonical 14-route evaluation:
  `artifacts/terrain_autoresearch/iteration97_v9_full14_canonical`

The archive contains the original 29 accepted clips plus 28 mechanically
accepted c490 clips. Those additions include five genuine descending GRAIL
sources, their analytic mirrors, and separate top-entry and bottom-landing
crops. They are not reversed ascents. Ten other proposed additions were
rejected by the complete-frame G1 geometry gate and are not published.

The first 9,631 baseline feature, source, pose, contact, tread, future
trajectory, and command rows—and the 60-dimensional feature scale—are
bit-for-bit unchanged from v5. The one deliberate metadata change is 720
boolean `entry` labels from rebuilding the reviewed pre-contact window with
the 45-frame entry contract; v8-entry45 and v9 agree on those labels.
Appending c490 data therefore does not silently renormalize or reweight the
baseline matcher.

## Runtime information boundary

The runtime command is still an ordinary two-stick command:

- robot-local planar travel velocity;
- independent world-facing yaw.

The matcher consumes the current kinematic G1 state and a 48x32 causal
robot-centred height map. It does not consume global root position, odometry, a
global map, or future user input. The scripted evaluator does use a global
waypoint to act like a human operator steering back toward the test route; that
privileged evaluator state is converted into the same two-stick command before
reaching the matcher.

## Mechanical evaluator

Every retained iteration is evaluated on clean kinematics before SONIC:

1. Direct source audit against its canonical stair geometry.
2. Scripted flat-to-stair scenarios at straight, ±30, ±60, and ±90 degree
   approach headings for both ascent and descent.
3. Flat escape and stop scenarios near each stair boundary.
4. Exact MuJoCo geometry contact distances and dense sole-point clearances,
   not only foot-centre heights.
5. Stance-foot sliding, root/joint discontinuity, stair entry, stair
   completion, top/bottom exit, safe-stop duration, and unintended freezing.
6. Full G1 mesh video, a 48-frame contact sheet, and dense touchdown windows
   for every candidate that beats the baseline mechanically.

The canonical v9 result is 13/14 routes entering and exiting, compared with
10/14 entering and 8/14 exiting for v8. It has zero forbidden body-collision
frames. Vertical riser contact falls from 864 frames in v8 to 3 in v9. The
remaining failure is the +90-degree uphill route: the flat approach reaches a
safe pre-riser pose, but no entry clip has a compatible outgoing contact phase.

## Hard gates

- No robot collision with a tread deeper than 5 mm.
- No foot-sole penetration deeper than 5 mm.
- No planted-foot drift above 10 mm per accepted cross-clip transition.
- Offline composition rejects any one-frame root translation above 60 mm,
  root rotation above 0.35 rad, joint step above 0.25 rad, or root
  acceleration above 40 m/s^2.
- No unsafe flat motion committed across a riser or unsupported ledge.
- No fall/freeze claim may be inferred from route completion alone.

Offline publication uses the 5 mm gate. Runtime pose repair permits up to 6 mm
because the MuJoCo collision solver otherwise permanently rolls back on
sub-millimetre residuals; the canonical maximum was 5.995 mm. This distinction
must remain explicit.

## Score and keep policy

The scalar score is lexicographic in practice:

1. Hard safety violations and penetration depth.
2. Number of successful stair completions and exits across the angle grid.
3. Sustained-intent freeze time and unsafe-candidate rejection rate.
4. Foot sliding and transition discontinuity.
5. Command/path error and motion quality.

An iteration is kept only if it improves this ordering without regressing a
hard gate.  More motion clips are not accepted merely because coverage grows;
their terrain/contact annotations and direct-source collision audit must pass.

## Verified one-riser reconstruction (2026-08-02)

The privileged exact-geometry bootstrap now composes a complete route from
automatically segmented entry, single-riser, and exit fragments. It does not
hard-code a source clip for a particular stair number. A layered beam search
selects one fragment per target riser, then continuous reconstruction applies
bounded inertialization, spatial target-level assignment, smoothed root
anchors, stance-foot locks, bracketed swing trajectories, and exact target-mesh
rejection. The target clip is excluded from retrieval automatically.

Two held-out six-riser canaries now pass:

- Target 37 ascent: 489 frames at 50 Hz, composed from clips
  `407, 76, 76, 36, 3, 26`. Maximum stance error is 0.342 mm, swing error
  19.88 mm, exact triangle/sphere penetration 2.112 mm, root step 19.47 mm,
  root rotation step 2.12 degrees, root acceleration 18.87 m/s^2, and joint
  step 0.227 rad. Every stance has at least two conservative sole supports.
- Target 261 descent: 442 frames at 50 Hz, composed from six separately
  selected ranges of source clip `171`. Maximum stance error is 0.249 mm,
  exact triangle/sphere penetration 0.380 mm, root step 25.20 mm, root
  rotation step 3.03 degrees, root acceleration 20.20 m/s^2, and joint step
  0.205 rad. Its maximum swing guidance residual is 83.85 mm, accepted under
  the 110 mm swing-only soft limit; all stance, collision, and continuity gates
  remain unchanged.

An independent full-G1 audit now forces each robot geom to collide against one
thin convex prism per exact terrain triangle. This avoids MuJoCo's
single-mesh convex hull bridging across the staircase while checking the
complete authored foot mesh, not only four sole spheres. The ascent/descent
maximum full-foot penetrations are 3.112/1.276 mm; both have zero forbidden
body contact and zero threshold-exceeding frames.

Both full-route 50 Hz G1-mesh videos were reviewed. They ascend/descend every
riser and finish upright without freezing, falling, a pose teleport, or
visible foot-through-tread motion. The ascent has a mild upper-step knee/leg
snap and both styles remain conservative and high-kneed, especially the
descent; this is a mechanics proof rather than a polished animation claim.

## Held-out geometry sweep baseline (2026-08-03)

A deterministic farthest-point sweep now tests six ascents and six descents
spanning rise, tread, stair count, and approach angle while excluding the two
development canaries. With eight ranked plans per target, 3/12 routes pass
both reconstruction and the independent full-G1 audit: two ascents and one
descent. The accepted routes include a +34.7-degree four-step ascent and a
+26.9-degree five-step descent, so the method does generalize beyond straight
canaries. Their maximum complete-foot penetrations are 0--3.194 mm and all
three have zero forbidden-body penetration.

Eight targets reject during mechanical fitting and one descent chooses a
mechanically valid candidate whose complete-foot penetration is 6.199 mm,
1.199 mm beyond the publication gate. That failure motivates audit-aware
ranked fallback: an unsafe candidate must be discarded and the next plan tried,
without relaxing the 5 mm complete-foot or forbidden-body gates. Several other
failures are small reconstruction near-misses (3--6 mm stance residual or
0.253--0.257 rad joint step); large residual and unsupported-foothold failures
remain true coverage or fitting gaps.

Retained evidence:
`artifacts/terrain_fragment_sweep/heldout12_fullfoot_v1`.

Audit-aware ranked fallback was then enabled. On clip 143 it rejects the
rank-2 candidate at 6.199 mm complete-foot penetration, continues searching,
and also rejects rank 4 at 5.107 mm rather than publishing either sequence.
No later candidate passes, so strict coverage remains 3/12. This is retained
as a safety/correctness fix, not claimed as a coverage improvement. Evidence:
`artifacts/terrain_fragment_sweep/heldout12_auditfallback_v2`.

A follow-up aligned the internal stance-fit and joint-step cutoffs to 6 mm and
0.27 rad/frame while leaving collision gates unchanged. Coverage remains
3/12: apparent stance near-misses then expose 147--156 mm swing errors or a
0.286 rad/frame jolt, and the two newly mechanically eligible clip-128 plans
fail the exact foot audit at 7.631 and 9.030 mm. The relaxed configuration is
discarded. Reconstruction rejection reporting now evaluates and reports all
mechanical violations together so future experiments do not discover masked
limits one rerun at a time. Evidence:
`artifacts/terrain_fragment_sweep/heldout12_quality_envelope_v3`.

## Riser-kernel decomposition and phase composition (2026-08-03)

The 3/12 complete-route result was primarily a representation failure. A whole
stair animation was being stretched across every metre of the requested route,
including long treads and landings. The replacement representation cuts each
held-out route at tread midpoints and reconstructs one compact, independently
audited motion kernel per height transition. Narrow treads become direct
motion-graph seams; only a genuinely wide landing is assigned to the ordinary
Takara/BONES flat matcher.

With the collision and mechanics gates unchanged, 57/60 held-out risers pass
independently (95%), and all risers pass on 10/12 complete target staircases.
The three failures comprise one terminal route that sampled beyond a finite top
platform and two difficult 22--23 cm risers on clip 39. This sharply separates
motion coverage from sequence composition. Evidence:
`artifacts/terrain_fragment_sweep/heldout12_riser_kernels_midpoint_v8`.

Each accepted riser can retain several independently audited source phases. A
global beam search scores endpoint root pose, joint pose, and finite-difference
velocity compatibility, then inertializes the selected chain and audits the
entire result again. On held-out target 11, the selected three-source chain
passes at 4.315 mm maximum complete-foot penetration and zero forbidden-body
penetration. Maximum inter-frame root translation, root rotation, and joint
steps are 24.99 mm, 0.298 rad, and 0.198 rad. Both source switches were visually
reviewed at full rate and show continuous stair ascent without a freeze or pose
teleport. Evidence:
`artifacts/terrain_fragment_composition/target00_phasebeam_v3`.

The isolated-kernel sweep was then repaired without changing collision gates.
A finite-top-platform terminal window and a smoothed high-riser reconstruction
raise strict isolated coverage from 57/60 to 59/60. The only unsupported
isolated unit is clip-39 riser 5; it is supported when planned jointly with its
neighbours, which is the correct representation for that contact sequence.

Three composition failures required distinct, retained fixes:

- Target 8 uses deeper phase pools on its final two risers and passes at
  4.443 mm complete-foot penetration with zero forbidden-body penetration.
- Target 4 has metre-wide landings. Splitting each landing between its adjacent
  stair kernels, rather than inserting an unrelated flat animation, gives a
  continuous accepted route at 4.743 mm and zero forbidden-body penetration.
- Target 10's unsupported high riser is covered by one coherent three-riser
  exact-terrain block. The block independently passes at 0.986 mm; the full
  seven-riser route passes at 3.965 mm and zero forbidden-body penetration.

The final geometry-spanning suite is now 12/12 strictly accepted: six ascents
and six descents over 3--8 steps, 10.2--23.8 cm nominal rises, 23.0--100.7 cm
treads/landings, and -25.2 to +34.7 degree approach headings. Across all 3,717
frames (74.1 seconds), maximum complete-foot penetration is 4.743 mm and
maximum forbidden-body penetration is zero. Maximum one-frame root
translation, root rotation, joint motion, and root acceleration are 42.13 mm,
0.224 rad, 0.232 rad, and 37.82 m/s^2. Every final MP4 decodes at 50 Hz and the
robot-mesh contact sheet shows all routes progress across the complete terrain
and finish upright, without a freeze, fall, or visible tread intersection.
The style is still deliberately conservative and high-kneed.

Final evidence:
`artifacts/terrain_fragment_composition/heldout12_phasebeam_final_v14`.
The directory contains the per-route motions, exact audits, G1 videos,
`all_routes_review_reel_50fps.mp4`, `all_routes_robot_contact_sheet.png`, and
`all_routes_intended_vs_actual.png`.

## Privileged online motion matching (2026-08-03)

The fragment composer is no longer the active path toward the game-quality
bootstrap.  The replacement follows the rolling architecture in Daniel
Holden's reference motion matcher and the environment-conditioned extension in
Environment-aware Motion Matching: a persistent simulation character predicts
the joystick trajectory, dense frame search compares current pose and future
trajectory, the incumbent successor is preferred, and a transition is
inertialized only when another state is materially better.  Terrain is an
additional query dimension rather than a separate route player.

The first implementation is intentionally privileged/global.  It compiles the
173-clip Takara/BONES flat bank and clean G1 stair motion into one online
database.  Stair rows include traversal, target compatibility, terrain-local
root stage/lateral/height, relative facing, geometry, and contact phase.  A
selected stair frame is placed by mapping its recorded terrain transform onto
the target terrain transform; it is never root-aligned independently of the
stairs.  Search runs every five 50-Hz frames, preserves successor motion when
safe, forbids cross-phase flight transitions, and hands back to ordinary flat
motion after a geometry/height completion test.  The persistent command
character is synchronized to terrain-locked output so a speed mismatch cannot
accumulate into a landing snap.

The clean stair source is now the full G1 stairs500 archive:
`/move/data/terrain-aware/motion-matching/grail-stairs500-clean-50hz-v1.zarr`
(500 clips, 249,500 frames, about 83 minutes).  Combined with the flat bank,
the available clean corpus is about 125 minutes.  The implementation lives in
`mm_sonic.privileged_terrain_matcher`; the scripted evaluator and exact-mesh
renderer live in `mm_sonic.evaluate_privileged_online_matcher`.

Verified bootstrap results:

- Exact ascent clip 0 enters from flat, traverses seven steps, returns to flat,
  and keeps maximum exact-mesh sole penetration below 0.4 mm.  A loose
  nominal-geometry search was discarded after video showed feet hovering over
  mismatched treads.  Tight compatibility plus terrain locking removes that
  defect and reduces the largest root step to 18.3 mm.  Evidence:
  `artifacts/privileged_online_matcher/smoke_up_clip0_tightgeom_ballfix_v3`.
- Exact descent clip 100 now remains flat until it reaches the authored top
  entry instead of decaying a half-metre inertial offset.  It descends six
  steps and exits to flat with 0.14 mm maximum exact-mesh penetration, 1.80 mm
  95th-percentile support-height residual, 0.023 m/s 95th-percentile support
  slip, and no backward frames.  Evidence:
  `artifacts/privileged_online_matcher/smoke_down_clip100_entryfix_v2`.
- Target-specific geometry compilation can add other G1 styles safely.  For
  target 15, five candidates were mechanically rejected; source 42 passed
  route warping and bounded stance-foot IK and was admitted.  Online search
  selected that warped source, followed it coherently, then returned to flat.
  Exact target-mesh audit gives 0.79 mm maximum penetration and 6.46 mm
  95th-percentile low-speed support residual.  Evidence:
  `artifacts/privileged_online_matcher/warped_up_clip15_v1`.

The Learned Motion Matching terrain query is now implemented directly for G1:
future terrain heights beneath both feet are compared at 0/15/30/45 frames in
the current character frame.  A 324,255-row database combines all 500 clean
stair clips with the 173-clip flat bank.  This experiment exposed an important
runtime bug: supplying terrain registration only on the transition frame let
the ordinary command-ball root correction drag an otherwise exact stair clip
almost 0.8 m out of registration by the landing.  Supplying the terrain
transform on every terrain successor reduced exact-mesh penetration from
23.8 cm to 0.39 mm.  Evidence:
`artifacts/privileged_online_matcher/carpet_full500_target0_persistentregistration_v5`.

The stronger geometry-generalization check removes the target motion from the
database.  Another G1 traverse is route-warped onto the globally known target
stairs, stance feet are corrected with bounded IK, and the result must pass the
exact target-mesh audit before entering online search:

- Held-out ascent target 15 uses warped source 42.  A +20-degree approach
  traverses and exits with 0.358 mm maximum penetration, 0.173 rad maximum
  joint step, 20.6 mm maximum root step, and one stair-source seam.  The
  0.25-rad source-window filter removes a 0.378-rad discontinuity already
  present inside the flat Takara clip.  Evidence:
  `artifacts/privileged_online_matcher/warped_loo_target15_source42_sourcefilter_v4`.
- Held-out descent target 242 uses warped source 190.  A -25-degree approach
  traverses and exits with 0.081 mm maximum penetration, 0.145 rad maximum
  joint step, 22.2 mm maximum root step, and one stair-source seam.  Evidence:
  `artifacts/privileged_online_matcher/warped_loo_down242_oblique25_v1`.
- A true diagonal entry starts 0.45 m off centre with -30-degree body yaw.
  Forward/riser phase remains exact while the complete warped animation is
  placed in a bounded lateral stair lane.  It traverses and exits with 0.361 mm
  penetration, 0.173 rad joint step, and 20.6 mm root step.  The rejected
  predecessor either shifted the clip through a riser (loose 0.22 m phase
  gate) or never entered (a strict 3-D root gate that incorrectly included the
  freely placeable lateral coordinate).  Evidence:
  `artifacts/privileged_online_matcher/warped_loo_up15_lateral045_laneplacement_v3`.
- A second, independently held-out four-step geometry (target 473) uses warped
  source 487 and starts 0.30 m off centre at +25 degrees.  It traverses and
  exits with 0.114 mm penetration, 0.141 rad maximum joint step, and 17.7 mm
  maximum root step.  This rules out a target-15-only result.  Evidence:
  `artifacts/privileged_online_matcher/warped_loo_up473_short4_lateral030_yaw25_v1`.
- A -0.35 m lateral, +30-degree held-out descent of target 242 now uses warped
  source 190 and traverses/exits with 0.081 mm penetration, 0.144 rad maximum
  joint step, and 20.9 mm maximum root step.  The earlier safe non-entry was
  not random: down-stair clips deliberately begin in a 7--10 cm lower pelvis
  preparation pose than the ordinary flat gait.  Separating the strict 6 cm
  forward-phase gate from an explicit 11 cm vertical inertialization allowance
  enables that preparation without weakening lateral, facing, or exact-mesh
  checks.  Evidence:
  `artifacts/privileged_online_matcher/warped_loo_down242_lateralminus035_yaw30_v3`.

The accepted ascent is live in a dedicated Switch-controller browser viewer.
Its receipt proves Torch search, the Takara command ball, robot-local travel,
independent facing, no command-side root-translation odometry, wxyz source
quaternions, the 0.25-rad source filter, target exclusion, and warped source 42.
This viewer is still globally privileged, clean kinematics, and a one-way
target scene; it is the easy-information bootstrap, not yet the final causal or
physics-tracked controller.

The one-way bootstrap has now been generalized into a finite four-way raised
platform.  A paired four-step ascent/descent geometry is placed on every side;
global root location plus requested travel direction selects a nearby physical
portal, and the common matcher uses the same audited G1 sources at all four
world orientations.  Both target motions are removed from retrieval.  The live
course receipt reports warped sources 488→410 and 146→220, 94,482 searchable
rows, robot-local travel, independent facing, and no command-side root
translation odometry.  This is the first explorable multi-object scene; a
scripted west-up/platform/east-down mechanical qualification is run separately
so interactive impressions cannot hide a failed terrain-to-terrain transfer.

This is a real online matcher but not yet the final learned-demo equivalent.
On a target with one compatible traverse, search correctly stays on its
successor sequence; broader steering requires several target-compatible
warped styles and reusable support cycles.  The current work therefore expands
the warped bank, tests oblique/recovery commands, and then exposes the same
controller through the Switch viewer.  Causal ego terrain perception remains a
later information-removal step after the privileged controller is visually and
mechanically strong.

### Terrain-height controller checkpoint (2026-08-03, later pass)

The online terrain-height path now uses the exact ray-cast USD stair profile;
the archived nominal rise/tread fields were stale for several clips.  Three
runtime faults were also removed: the temporal exclusion mask had been
discarding the authored successor, flat motion could continue into the first
riser, and the landing test waited for the nominal full run even when the last
physical riser ended earlier.  Stair rows now retain valid successors, enter
by 0.60 m lookahead, register each selected row into a smoothly centred lane,
and exit just beyond the measured last riser.

The resulting controller completes all six evaluated ascent/descent geometry
pairs, spanning four to six levels, 11.0--19.7 cm rises, and 27.5--48.0 cm
treads.  The formerly trapped 465/323 long-tread case now completes with a
0.157 rad maximum joint step, 21.1 mm maximum root step, and 3.25 mm
ankle-proxy penetration.  Four oblique starts at lateral offsets -0.9, -0.6,
+0.6, and +0.9 m all complete the nominal bidirectional course.  Evidence:
`artifacts/privileged_online_matcher/varied_carpet_v5_physical_landing` and
`artifacts/privileged_online_matcher/oblique_carpet_v3_safe_splice`.

The exact 451 clean clip exposed a second issue hidden by the old ankle proxy:
the executable MuJoCo G1 foot collision geometry penetrates a tread by several
centimetres even when the ankle centre appears clear.  Keeping that clip out
and using two independently audited geometry warps reduces rejected collision
frames from 114 to 3 before contact cleanup.  The paper-style stateful support
contact/IK pass is now enabled after root clearance.  It latches slow support
feet, releases them with a spring, raises swing feet by a bounded amount, and
limits a failed release to 0.18 rad/frame.  On nominal 449/197 it gives zero
ankle-proxy penetration and reduces support slip from 0.094 to 0.082 m/s with
a 0.170 rad maximum joint step.  On held-out 451/212 it gives 1.16 mm maximum
proxy penetration, 0.082 m/s support slip, and a 0.180 rad maximum joint step.
Evidence:
`artifacts/privileged_online_matcher/contact_filter_v1/nominal449` and
`artifacts/privileged_online_matcher/contact_filter_v2_continuous/hard451_loo`.

### Coherent traversal hierarchy (2026-08-03)

Visual review showed that collision-valid one-riser composition was still the
wrong primary representation: repeated source changes made climbs look
teleported and foot-slid even when every mechanical threshold passed.  A
preprocessing fault had made this unnecessarily severe.  Stable feet were
assigned to treads by dividing their height by the archive's nominal rise,
but several source meshes are strongly non-uniform.  Target 49 has six exact
rises ranging from 11.3 to 24.9 cm; the old labels were `0,2,3,4,6,7` and only
two adjacent fragments survived.  Exact ray-cast tread heights now recover all
six consecutive support transitions and retain the authored approach and
landing.  Focused extraction/reconstruction tests pass.

The runtime hierarchy is consequently:

1. Use a complete, target-compatible G1 traversal as one coherent asset.
2. Rigidly place an exact asset, or apply one bounded whole-route geometry
   warp to a close asset, and mechanically audit it before runtime.
3. Motion-match the Takara/BONES flat approach and landing, but play the chosen
   stair traversal sequentially until its audited landing state.
4. Use support-fragment composition only when no long compatible asset covers
   the requested terrain, and prefer source-contiguous multi-step blocks.

The target-49 identity ceiling retains all 499 authored frames with no seams,
zero joint correction, effectively zero foot-target error, and 2.83 mm maximum
source sole/mesh overlap.  Evidence:
`artifacts/global_target49_routes/coherent_asset_identity_v1`.

The first complete course using this hierarchy enters ascent clip 449 at frame
2 and advances uninterrupted through frame 438; it enters descent clip 197 at
frame 3 and advances uninterrupted through frame 371.  It has 5 commanded
near-frozen frames in 1,200, 20.3 mm maximum root step, 0.158 rad maximum joint
step, and 0.075 m/s support-slip p95.  Its only stair-source discontinuities
are the two intended flat/terrain handoffs; 19 other source changes are normal
flat motion matching.  Evidence:
`artifacts/privileged_online_matcher/four_way_course_coherent_449_197_v9`.

### Balanced two-stick flat bank (2026-08-03)

The old 173-clip Takara/BONES bank did not actually cover the requested
two-stick control space.  It contained lateral starts and stops but no
sustained sideways loops, and only two large idle turns.  Consequently a
continuous left command recycled one start clip for 335/400 frames, diagonal
travel fell back to TakaraWalk for 370/400 frames, and a stationary +90-degree
yaw command substituted backward/start motions.  This explains the live
symptoms: no diagonal translation without first turning the torso and a
root-swivel rather than a planted turning step.

`select_bones_locomotion_bank.py` now mines the 39,466-clip retargeted BONES
locomotion store with a mechanical gate and balanced semantic quotas.  It
scanned 872 neutral locomotion candidates and selected 175 base clips spanning
steady omnidirectional loops, multiple speeds, arcs, short 45/90/135-degree
turns, starts, and stops.  True mirrored variants are exported for every BONES
clip, producing a 351-clip Takara/BONES bank and 400,380 searchable live-viewer
rows under the stricter 0.25-rad source-step gate.  The selection is recorded
in `sonic/configs/bones_balanced_two_stick_v1.json`; mechanical and coverage
statistics are in
`artifacts/bones_balanced_two_stick_v1/selection_report.json`.

Scripted probes now use sustained sideways/diagonal loops for the whole
command, and the expanded +90-degree stationary-yaw result lifts a swing foot
by 10.2 cm rather than swivelling both planted feet.  The remaining pivot still
has visible stance slip and commanded arcs switch sources too frequently, so
these are improvements rather than a finished flat controller.  A public
MotionBricks reference checkout is staged separately at
`/move/u/bodow/Projects/reference-nvidia-groot-wbc` as a possible generator for
missing flat transitions; it has not been admitted to the bank without the
same G1 mechanical gate.

The course viewer also had two coplanar ground planes: the canonical G1 MJCF
already defines `floor`, while the course builder added `course_ground` at the
same height.  The builder now reuses an existing plane, eliminating the floor
texture flashing without changing terrain geometry.

### Coherent-warp geometry ranking (2026-08-03)

The first broader post-viewer experiment compared the original nominal
rise/tread/step prefilter against a ray-cast physical-mesh profile prefilter on
the fixed 12-geometry held-out suite. Each target required two mechanically
accepted leave-one-out whole-route warps among 12 candidates. Both methods
pass 3/12 targets and admit ten total warps. Exact-profile ranking is slightly
worse in candidate attempts and joint correction, so it is not enabled.

This negative result localizes the next mismatch: the whole physical stair
profile is not necessarily the profile traversed between a motion clip's
actual root endpoints. Several exact-profile candidates still fail because
the geometry warper samples a different support-level count along that route.
The next prefilter therefore compares the same motion-conditioned support
routes used by the warper. Evidence:
`artifacts/coherent_warp_coverage/heldout12_ab_v1`.

Motion-conditioned route ranking then compares the exact support levels,
tread widths, and rises sampled between each clip's actual root endpoints.
This increases two-style leave-one-out coherent coverage from 3/12 to 4/12
and accepted warps from 10 to 12 without changing the mechanical gates. It is
retained as the default for future coherent asset-cache builds. The remaining
8/12 gap demonstrates that complete traversals alone are not sufficient;
coverage must fall back to the longest source-contiguous multi-riser blocks
before considering one-riser composition. Evidence:
`artifacts/coherent_warp_coverage/heldout12_route_v2`.

Increasing the route-ranked search from 12 to as many as 50 candidates on all
eight failures adds no second accepted style and leaves all zero-warp targets
at zero. Several targets exhaust every compatible whole route. Remaining
rejections are mechanical clearance or foot-target failures, so deeper search
is discarded and contiguous multi-riser sub-traversals are now required.
Evidence: `artifacts/coherent_warp_coverage/heldout12_route50_v3`.

Source-contiguous blocks recover almost all of the coverage hidden by the
whole-clip representation.  Stable transition fragments are grouped by their
original source clip and chronological frame ranges; all authored frames
between the first and last fragment are retained, so this is not per-step
hard-coding or frame substitution.  A longest-block dynamic program covers
11/12 held-out routes: six use one coherent block with no internal source
switch, and five use two blocks with one shared-landing switch.  Those plans
span 4--7 stair transitions and keep the existing 0.50-rad warp, 3-mm target,
6-mm proxy-penetration, 25-mm root-lift, and stance-support gates.  The sole
failure is clip 39, for which the already-audited short-kernel plus coherent
three-riser fallback remains available.  Independent block feasibility is not
yet a full-route result: exact complete-G1 collision and phase-compatible seam
composition are being evaluated separately.  Evidence:
`artifacts/coherent_block_coverage/heldout12_blocks_v2`.

The exact phase-pool compositor converts ten of those eleven proxy plans into
complete accepted G1 routes.  It searches up to four exact-mesh-safe phases
per span, ranks shared-landing pose and velocity compatibility, and chooses a
critically damped seam only after full-route collision and acceleration
checks.  A generic temporally smoothed pelvis-clearance envelope closes the
gap between the four tiny sole probes and the rendered foot mesh; the largest
correction is 11.09 mm and the resulting maximum root acceleration is
23.71 m/s^2.  Across the ten new routes, complete-foot penetration is at most
4.998 mm and forbidden-body penetration is zero.  Dense 25 fps review of all
four new two-block joins shows no source-switch teleport, hop-hold, or
double-tap; clip 436 has the tightest collision margin and a mildly sharper
join, so it is accepted but not called polished.  Clips 369 and 39 retain the
older exact short-kernel/multi-riser fallback, keeping the full hierarchy at
12/12 while direct long-block acceptance is 10/12.  Evidence:
`artifacts/coherent_block_composition/heldout12_phasepool_clearance_v3`.

A disjoint, balanced 50-target validation set was selected by farthest-point
coverage over rise, tread, step count, and approach angle.  None is in the
12-route development set.  Long-block retrieval supports 48/50 geometries;
the base exact search accepts 38, and deeper retrieval only on exact failures
recovers two more (clips 373 and 473), for 40/50 overall and 40/48 among
retrievable routes.  Every accepted route remains below 4.979 mm complete-foot
penetration with zero forbidden-body penetration.  The deepest accepted route
uses three blocks; dense 25 fps review of both joins shows continuous motion,
not the earlier source-switch teleport or hop-hold.  The remaining ten
geometries are now being passed to the previously validated compact-riser plus
flat-bridge fallback without changing any acceptance threshold.  Evidence:
`artifacts/coherent_block_validation/geometry50_aggregate_deep_v2.json`.

The complete terrain hierarchy now accepts 47/50 fresh geometries without
using their own source clip: 40 use source-contiguous coherent blocks and
seven use compact reusable riser fragments.  All accepted routes remain below
4.979 mm complete-foot penetration, have zero forbidden-body penetration, and
stay below 40 m/s^2 root acceleration.  Applying the globally privileged
runtime assumption--use an exact source traversal when one exists, after the
same bounded smooth clearance correction--raises coverage to 49/50: 42 exact
source traversals, five coherent substitutes, and two compact-riser routes.
The only remaining source-covered failure is clip 431, a 24.9 cm narrow-tread
ascent whose clean retarget already penetrates its own mesh by 43.1 mm with a
forbidden body link.  Dense review of the worst repaired source motions and
the highest-acceleration compact routes shows continuous motion without the
old teleport or freeze, although the gait remains conservative.  Evidence:
`artifacts/coherent_block_validation/geometry50_hierarchy_aggregate_v1.json`,
`artifacts/coherent_block_validation/geometry50_source_covered_hierarchy_v1.json`,
and `artifacts/coherent_block_validation/geometry50_hierarchy_coverage_v1.png`.

### Visual correction: zero-seam registered sources (2026-08-03)

Full-rate review invalidated the earlier implication that the compact-riser
fallbacks were polished merely because they passed collision and mechanics
gates. In particular, targets 409 and 434 contained respectively 4 and 5
stair fragments, 2 and 3 flat bridges, and 2.92 and 3.72 seconds of filler.
Their stance-slip p95 values were 0.861 and 0.426 m/s. Those artifacts remain
useful collision/coverage experiments, but they are not the retained animation
path and must not be described as game-quality motion.

The retained globally privileged representation now keeps all 499 authored
frames of an exact traversal, with zero internal seams and zero flat fillers.
Before playback, the whole root trajectory is registered to its known target
mesh by the smallest constant travel-axis offset that passes the exact audit
(0 to -8 cm in the 50-geometry suite). Per-frame cleanup then resolves only
actual G1 contacts along their exact three-dimensional MuJoCo contact normals.
It does not independently lift each foot or replace stair steps. A causal
inertialization is applied only to a repair residual when independent contact
cleanup would turn an otherwise valid authored joint transition into a
greater-than-0.25-rad discontinuity.

On the disjoint 50-geometry suite this exact-source stage is now 50/50 under
all publication gates. Across all routes it has at most 4.997 mm complete-foot
penetration, zero forbidden-body penetration, 0.240 rad one-frame joint motion,
24.55 m/s^2 root acceleration, and 0.124 m/s stance-slip p95. The one final
near miss, target 373, came from contact repair increasing an authored
0.216-rad transition to 0.257 rad; carrying 0.017 rad of the repair residual
into the next frame reduces the transition to 0.240 rad without changing its
collision or slip result. Evidence:
`artifacts/coherent_contact_repair/geometry50_registered_v1/aggregate.json`.

Targets 409, 434, 63, 398, 431, and 452 were rendered at 50 Hz and reviewed
over whole-route and dense touchdown sheets. The difficult tall/narrow target
431, previously rejected with 43.1 mm source collision, passes after one
constant -6 cm route registration and six small contact repairs; it has
0.203-rad maximum joint motion and 0.086 m/s stance-slip p95. Evidence:
`artifacts/coherent_contact_repair/registered_review_v1`.

The same registered ascent/descent sources are now consumed by the rolling
flat/stair matcher. A 24-second flat-to-ascent-to-platform-to-descent-to-flat
course completes bidirectionally with no rejected pose repairs or continuity
clips, 0.162-rad maximum joint motion, and 0.076 m/s stance-slip p95. The
whole-route and dense ascent/descent sheets show continuous traversal without
a fall or stair-fragment teleport. A more aggressive online 3-D contact-
normal A/B was discarded: it raised sole penetration to 15.1 mm, produced 23
frozen frames, and rejected seven repairs. The online path therefore retains
bounded root clearance while the preprocessed source asset uses exact 3-D
contact repair. Evidence:
`artifacts/privileged_online_matcher/four_way_course_registered_449_197_v11`.

### Corrected global MotionBricks terrain bank (2026-08-04)

The earlier `50/50` wording was too broad.  It described exact-source audit
acceptance in the development suite, but nine of those motions do not actually
span both physical terminal surfaces of their stairs.  The corrected global
terrain index contains 41 complete traversals: 19 ascents and 22 descents,
covering 3--7 physical levels, 10.0--24.9 cm mean risers, 22.4--73.9 cm mean
treads, 0.40--1.33 m total height, and 0.74--3.75 m total run.  All subsequent
catalog counts use this 41-route denominator.

A reusable MotionBricks flat transition now composes successfully onto all 41
routes.  Three initial template banks each passed 41/41, with at most 4.992 mm
complete-foot penetration, zero forbidden-body penetration, 0.223 rad joint
step, 33.86 m/s^2 root acceleration, and 2.78 cm root translation step.  Eight
hard descent exits revealed a real phase bug: an exit beginning mid-swing has
no preceding stance history, so the source-contact-release detector could hold
one foot indefinitely.  Retrying only that seam with the deterministic
staggered release recovers the route under the same collision and mechanics
gates.  Two zero/negative entry cases additionally use the already-qualified
positive-family exit; entry and safe landing phase are therefore treated as
independent assets.

Route plotting then exposed that those nominal `-45/0/+45` banks often selected
the first 40--43 samples of MotionBricks startup, only 4--5 cm of travel.  They
are valid contact-phase stubs, not meaningful multi-angle approaches, and are
not the final interactive bank.  Approach phase selection now requires both an
active planar gait and at least 35 cm of authored displacement.  On the hard
target409 stair this produces three exact-safe run-ins: a 2.09 m centred path,
a 2.37 m left path from 0.57 m lateral offset, and a 1.41 m right path from
0.73 m lateral offset.  All three retain zero forbidden-body contact, at most
2.184 mm complete-foot penetration, a 0.173 rad maximum joint step, and a
32.73 m/s^2 maximum root acceleration.  The left path is an analytic sagittal
mirror of the clean right MotionBricks transition about its authored route
axis; pose, MuJoCo/IsaacLab joint order, planned path, facing, and command
vectors are all mirrored, and the two-pass involution error is at most
3.6e-7.  The full rebuild accepts 41/41 centre, 41/41 left, and 41/41 right
courses.  Across all 123 courses, maximum complete-foot penetration is 4.991
mm, forbidden-body penetration remains zero, maximum one-frame joint motion
is 0.216 rad, maximum root acceleration is 32.73 m/s^2, and maximum root
translation is 2.771 cm.  Evidence:
`artifacts/global_scene_terrain/motionbricks_global_terrain_catalog_active_41x3_v2.json`.

The same exact target mesh can now be traversed in the direction missing from
its authored clip.  For target452, the geometry-ranked target220 ascent is
warped coherently into a target452 descent rather than being split into
per-step fragments.  It retains 2.946 mm maximum complete-foot penetration,
zero forbidden-body penetration, a 0.139 rad maximum joint step, 16.59 m/s^2
root acceleration, and a 1.71 cm maximum root step.  Centre and right active
MotionBricks approaches both compose onto that generated descent under the
same final 5 mm/zero-body collision gates.  The left approach is correctly
rejected because its oblique run-in intersects the narrow top platform.  A
geometry-ranked opposite-direction pass over the other 40 physical routes is
in progress; this one accepted route is therefore a bidirectional proof, not
yet a 41-route bidirectional coverage claim.  Whole-route 2 fps and terrain
10 fps review of both accepted target452 courses shows continuous descent,
clean bottom-step clearance, and flat exit without a freeze, teleport,
visible foot-through-step event, or landing hover.  Evidence:
`artifacts/global_scene_terrain/bidirectional_target452_v3/courses`.

The bank-wide reverse pass now ranks against all 500 clean archive clips using
motion-conditioned route geometry instead of the 41-route catalog's coarse
mean-rise/mean-tread proxies.  Thirteen opposite-direction motions currently pass
the unchanged final physical gates (targets 29, 107, 125, 156, 205, 242, 254,
264, 302, 308, 395, 435, and 452); candidate failures remain rejected and are retried at
the next geometry rank.  Three-step targets are a real source-coverage corner case:
the archive has no complete three-step descents.  Rather than stitching
unrelated steps, target29 uses one contiguous three-step subspan from a clean
four-step descent.  Its generated motion has 2.202 mm complete-foot
penetration, zero forbidden-body penetration, a 0.156 rad maximum joint step,
and a 1.81 cm maximum root step.  The full MotionBricks-to-descent-to-
MotionBricks course also passes, with 2.000 mm complete-foot penetration and
zero forbidden-body penetration.  Whole-route, dense 10 fps, and 25 fps seam
review show three continuous contacts and a clean flat exit without a freeze,
teleport, hover, or visible step penetration.  Evidence:
`artifacts/global_scene_terrain/opposite_level3_target29_course_center_v1/composition`.

Re-warping that accepted target29 descent is not a valid shortcut for the two
other three-step meshes.  Although the step count matches, target10 requires a
33% longer run and target47 substantially taller rises.  A geometry-only
second warp was honestly rejected before qualification: maximum foot-target
errors were 176.5 mm on target10 and 422.6 mm on target47.  The experimental
code path was removed; the negative artifacts are retained at
`artifacts/global_scene_terrain/opposite_level3_derived_target29_v1`.

Target47 is now the first accepted boundary-context fragment fallback.  A
boundary-diverse search selected one true entry fragment followed by two
source-compatible fragments (clips 279, 339, and 339).  The isolated motion
retains 0.117 mm sole penetration, 0.254 mm maximum stance-foot residual,
15.735 mm swing guidance residual, a 0.223 rad maximum joint step, 14.19
m/s^2 root acceleration, and a 1.98 cm root step.  More importantly, the full
MotionBricks-to-descent-to-MotionBricks course also passes: 2.357 mm complete-
foot penetration, zero forbidden-body penetration, a 0.175 rad maximum joint
step, and only 0.016/0.023 rad joint steps at the two flat/stair seams.  Whole-
route, 5 fps terrain, 10 fps handoff, and 20 fps internal-seam review show a
continuous three-step descent and flat walk-away without a freeze, teleport,
hover, visible penetration, or pose snap.  Evidence:
`artifacts/global_scene_terrain/opposite_level3_fragment_boundary_diverse_courses_center_v2/target_47/composition`.

Target10 now also has an accepted boundary-context fallback, after isolating a
different failure from target47.  Its requested sole path was smooth, but the
independent per-frame IK crossed a near-straight right-knee singularity and
changed 0.325 rad in one frame.  Seeding the next solve from the preceding pose
did not fix it and worsened stance residual, so that hypothesis was rejected.
A swing-only 0.15 rad per-joint continuity bound instead spreads the same knee
motion over three frames; stance legs remain exact and all original target,
collision, and mechanics audits remain authoritative.  The isolated motion
retains 3.545 mm maximum stance-foot residual, 1.363 mm mesh penetration, a
0.150 rad maximum joint step, and 20.05 m/s^2 root acceleration.  Its full
MotionBricks-to-descent-to-MotionBricks course also passes with 2.459 mm
complete-foot penetration, zero forbidden-body penetration, a 0.150 rad
maximum joint step, and a 2.42 cm maximum root step.  Evidence:
`artifacts/global_scene_terrain/opposite_level3_fragment_continuity_courses_center_v7/target_10/composition`.
Whole-route 2 fps, terrain 5 fps, singularity-region 20 fps, and exit-seam
20 fps review show a continuous three-step descent and flat walk-away without
a knee snap, freeze, teleport, hover, visible penetration, or landing hitch.

The same boundary-fragment fallback now scales beyond the two hand-debugged
three-step geometries.  Target113 is an accepted four-level reverse course:
its full MotionBricks-to-stair-to-MotionBricks composition has 3.763 mm
complete-foot penetration, zero forbidden-body penetration, a 0.145 rad
maximum joint step, and 13.89 m/s^2 root acceleration.  Whole-course and
terrain-dense review show a continuous approach, climb, top clearance, and
flat walk-away without a freeze, pose snap, hover, or visible penetration.
Evidence:
`artifacts/global_scene_terrain/opposite_boundary_fragment_courses_center_v2/target_113/composition`.

Scaling also exposed a real limitation of the cheap sole-sphere collision
proxy.  Target135's first isolated candidate passed that proxy at 1.263 mm but
the complete foot mesh penetrated 8.782 mm in the composed course.  Complete
foot and forbidden-body collision audits are therefore now part of ranked
candidate selection itself, so a bad candidate is rejected and search
continues instead of relying on the final course gate.  On target135 this
rejects rank 0 and accepts rank 16, whose isolated motion has 2.884 mm
complete-foot penetration, zero forbidden-body penetration, a 0.150 rad
maximum joint step, 0.480 mm stance residual, and 10.16 m/s^2 root
acceleration.  Its corrected full course passes at 3.860 mm complete-foot
penetration and zero forbidden-body penetration.  Visual review confirms a
continuous four-step ascent with stable flat entry and exit and no teleport,
freeze, hover, or tread intersection.  Evidence:
`artifacts/global_scene_terrain/opposite_boundary_fragment_courses_fullbody_retries_v3/target_135/composition`.

Across the thirteen coherent opposite-direction motions, twelve now compose
into accepted full flat-to-stair-to-flat courses under the unchanged final
5 mm complete-foot and zero forbidden-body gates.  Target435 remains a valid
isolated coherent traversal but is not published as a course because its flat
handoff seam does not fit.  Together with the two accepted boundary-context
fallbacks above, the current reverse course set contains fourteen physically
qualified courses.

Those courses are now merged into a single directional runtime catalog rather
than living as disconnected experiments.  It contains 41 physical geometries,
55 directed routes, and 137 accepted courses: the original 123 three-family
courses plus one opposite-centre course on 14 geometries.  The catalog-aware
viewer automatically loads both directions when they exist.  Its runtime
audit accepts all 137 own-start captures, selects the intended family in all
137 cases, and validates all 137 MotionBricks landing contexts with a maximum
endpoint error of 4.0e-15.  Evidence:
`artifacts/global_scene_terrain/motionbricks_global_terrain_catalog_bidirectional_v3.runtime_audit.json`.

The multi-entry catalog and runtime portal are purely kinematic and globally
privileged.  The official MotionBricks controller owns ordinary flat motion;
near a known stair, the selector chooses an exact-safe course by global portal
position, travel alignment, yaw, and lower-body phase.  After landing, the
last four poses are resampled at 30 Hz to seed MotionBricks again.  An audit of
the final active 123-course catalog accepts all 123 own-start portal captures,
selects the intended family in all 123 cases, and validates all 123
end-context handoffs (maximum endpoint error 4.0e-15).  Dense 10 fps visual
review of the hardest
ascent and two difficult descents shows no old freeze, teleport, landing
hover, or visible foot-through-step event, although the underlying authored
stair gaits remain conservative.  Evidence:
`artifacts/global_scene_terrain/motionbricks_global_terrain_catalog_active_41x3_v2.runtime_audit.json`.

The separate MotionBricks-to-SONIC collection leg successfully stores native
G1 clean kinematics and exact two-stick commands.  It has not yet produced a
qualified broad noised physics corpus: generic SONIC reaches 181.2 mm mean
MPJPE on straight motion and 1157.7 mm on the omnidirectional trace, while the
TakaraWalk-finetuned tracker reaches 304.2 mm on straight and terminates after
76/699 omnidirectional frames.  The blocker is tracker adaptation to the
MotionBricks style, not clean command generation.

### Scene-agnostic ramps, curbs, and unified dispatch (2026-08-04)

The arbitrary-mesh stair API is no longer only an exact-source ceiling.  Four
leave-one-source-out routes all pass while explicitly excluding the clean
motion associated with the target mesh: two descents, one ordinary ascent,
and one oblique ascent.  Each result uses a coherent source span, retains zero
forbidden-body collision, and stays below 2.34 mm complete-foot penetration.
A representative 765-frame MotionBricks-flat-to-stair-to-flat course also
passes at 2.34 mm foot penetration, zero body collision, 0.149 rad maximum
joint step, and 17.04 m/s^2 root acceleration.  Evidence:
`artifacts/generic_stair_route/heldout4_mesh_only_v1` and
`artifacts/generic_stair_route/canary_target223_mesh_only_v1/motionbricks_course`.

A separate clean C490 archive adds 149 authored G1 clips--72 curb and 77
slope, 74,351 frames total--without using the SONIC rollout states.  The slope
index contains 89 monotonic windows (50 uphill and 39 downhill).  Four
leave-one-source-out ramp routes all pass, including both ascent and descent,
with 2.07--4.45 mm complete-foot penetration and zero forbidden-body
collision.  Two representative full MotionBricks-flat/ramp/flat courses pass
at 2.03 and 3.95 mm foot penetration with 0.119 and 0.137 rad maximum joint
steps.  Evidence:
`artifacts/generic_terrain/slope_heldout4_mesh_only_v1`.

Alternating curbs cannot be treated as one monotonic staircase or forced onto
an unrelated whole source sequence.  The retained path decomposes a requested
support route into coherent adjacent transfers, retrieves from 65 clean
source windows (38 up and 27 down), warps each transfer to the exact target
mesh, and then performs phase-aware, contact-retargeted composition.  The two
short held-out curb routes pass at 2.62 and 3.10 mm foot penetration, zero
body collision, and 0.111/0.127 rad maximum joint step.  The first complete
MotionBricks-flat/curb/flat course passes at 2.34 mm, zero body collision, and
0.146 rad maximum joint step.  The two long alternating routes remain stress
tests for tall-transfer coverage and internal seam compatibility rather than
being counted as accepted coverage.  Evidence:
`artifacts/generic_terrain/curb_heldout4_mesh_only_v1`.

The unified route API now reads one exact USD mesh plus world-space route and
dispatches flat intervals to live MotionBricks, continuous ramps to the C490
slope warp, monotonic steps to the stair backend, and alternating steps to the
curb transfer composer.  A procedural 10 m mixed scene is accepted end to end
from geometry alone: it discovers one ramp portal and one separate up/down
curb portal.  The ramp passes at 2.47 mm foot penetration and 0.082 rad joint
step; the two-transfer curb passes at 4.38 mm and 0.134 rad, with zero body
collision for both.  A direct arbitrary flat-bank splice was not sufficient
for the curb exit.  The accepted boundary instead keeps the consecutive
source gait after the descent and normalizes its following 8.285 cm source
curb to flat, preserving the shared pose while smoothly removing only the
terrain-height component.  Its landing overlap is 0.13 mm at the root with
zero joint gap.

Runtime portal compilation now accepts both events in one manifest after
jointly ranking exact-safe terrain primitives and MotionBricks entry/exit
phases.  The complete ramp course passes at 2.03 mm foot penetration, zero
body collision, and 0.118 rad maximum joint step.  The complete curb course
passes at 2.78 mm, zero body collision, 0.230 rad maximum joint step, and
22.83 m/s^2 root acceleration.  L40 renders and dense seam sheets show no
visible teleport at either curb boundary.  Evidence:
`artifacts/generic_terrain/unified_mixed_smoke_v1/portals_landing_bridge`.

## Current limitations

- This result is clean kinematics, not a SONIC physics/tracking qualification.
- The accepted mixed runtime is one ramp plus one up/down curb.  It validates
  unified dispatch and both live-flat handoffs, not exhaustive combinations of
  every terrain event or arbitrary steering while committed to a portal.
- The 50/50 registered-source result remains an exact-source ceiling, but it is
  no longer the only generalization evidence: four unseen stair meshes and
  four unseen ramp meshes now pass leave-one-source-out generation.  Those are
  still compact geometry-spanning suites, not exhaustive coverage of every
  possible terrain profile.
- The rolling course completes, but it still has ordinary flat-bank source
  changes and only two registered stair styles. Broader steering on a stair
  requires more compatible coherent traversals and production-quality
  transition selection; the current result should not be equated with a
  shipped game locomotion stack.
- The 12-route held-out suite now passes completely, but it is still a compact
  geometry-spanning benchmark rather than evidence over every terrain mesh.
- The bootstrap uses a requested global route and exact global target mesh. It
  proves that fragment composition and terrain fitting can work when the
  geometry problem is made easy; it is not yet the desired causal,
  robot-centred runtime matcher.
- Some authored G1 stair styles remain conservative and high-kneed, with
  visible lateral pelvis weave. Exact-source playback removes our artificial
  fragment jolts but cannot make the underlying retarget more natural.
- Raw authored-contact slide metrics overstate entry slide during the
  inertialized handoff, while contact-only metrics mistake moving toe scuffs
  for stance. Dense foot video remains authoritative until a velocity-aware
  support metric is implemented.
- Stateful support-contact IK is now the default for the privileged course and
  improves both penetration and stance slip.  Four hard-geometry frames still
  fall back to root-only clearance; their visual release is bounded rather
  than treated as proof that the underlying source motion is physically valid.
- +90-degree uphill entry remains a real contact-phase/data-coverage gap.
- The visible G1 foot mesh extends beyond its four tiny MuJoCo sole probes,
  especially in toe-down poses.  A live platform-edge reproduction measured
  2.53 cm of rendered-mesh penetration despite the proxy check; the generic
  final mesh/terrain clearance gate removed it exactly.  This is the last
  course-specific debugging pass: turn quality and remaining motion polish
  are data/matching coverage work, not reasons to tune the four-way viewer.

## Branches from here

1. Expand the exact-geometry benchmark beyond the 12 geometry-spanning routes,
   rank passing plans by motion quality, and publish only accepted sequences to
   the privileged interactive viewer.
2. Add phase-compatible curved pre-entry clips, especially for the missing
   +90-degree uphill side, rather than relaxing the planted-foot handoff.
3. Expand clean terrain coverage with the next mechanically accepted
   GRAIL/Karen sources and correct analytic mirroring of pose, contacts,
   trajectory, and robot-centred height observations.
4. Add bounded contact-preserving timing/pelvis/swing-foot warping with planted
   contacts as equality constraints and exact collision rejection.
5. Keep the Takara/BONES bank as the flat locomotion prior; LAFAN can fill
   start/stop/turn gaps only after G1 retargeting and the same mechanical gate.
6. Replay accepted kinematics through the correct tuned SONIC trackers. Keep
   clean kinematic failure separate from tracking/physics failure.
7. Store the exact joystick stream during future collection so the diffusion
   policy need not infer commands retroactively from root displacement.

## Autonomy

Run unattended.  Time-box individual probes where practical, log every
hypothesis and result, revert failed experimental changes locally, and leave
the best verified artifacts ready for review.

### Curved coherent global-terrain showcase (2026-08-05)

The frame-by-frame canonical matcher is not the retained path for the global
ceiling.  Its adaptive-transition run reduced source switches from 52 to 30,
but still reached 10.98 cm foot penetration, 13.48 cm hover, and only 7.83 m
progress.  Delaying a bad transition merely let flat clips continue into an
obstacle.  This branch is rejected.

The retained architecture plans with global scene information and commits to
an already-qualified coherent terrain traversal for each obstacle.  Exact
terrain and motion receive the same rigid transform, while grounded official
MotionBricks curves connect the obstacle courses.  One 44.52 s course now
contains an oblique four-step ascent, an S-curve, a turned four-step descent,
a right arc, and an oblique ramp ascent.  It covers 17.11 m with 3.32 m lateral
span in the first accepted rendering; a phase-selected connector variant
increases lateral span to 4.95 m.

A useful failure was found during video review.  The first composition passed
the complete-body penetration audit, yet neither sole had terrain beneath it
for 48 consecutive descent-exit frames and 18 ramp-entry frames.  A
penetration-only metric therefore cannot certify support.  The composer now
casts beneath every exact G1 sole-sphere bottom and rejects any frame without
nearby support.  Directional landing context extends only the lower approach
under an ascent and the lower landing under a descent; extending the upper
platform through the obstacle was tested and rejected because it intersected
the moving robot.

The corrected supported course passes all 2,227 frames with zero unsupported
frames, a 35.6 mm worst nearest-support gap, 3.69 mm maximum complete-foot
penetration, zero forbidden-body penetration, a 0.170 rad maximum joint step,
and 19.37 m/s^2 maximum root acceleration.  Evidence:
`artifacts/global_terrain_showcase/coherent_curved_v4_supported`.

The remaining external flat/course foot-lock attempt is not being counted as
successful.  With the original arbitrary flat phases, its first unreachable
sole target is 38.4 mm away.  Endpoint phase selection lowers the first miss
to 24.9 mm and increases the route's lateral span, but does not make the
contact solve valid at the 20 mm threshold.  Simply relaxing the threshold
causes the error to grow as a locked foot is dragged by the moving pelvis, so
that experiment is rejected.  The rendered fallback uses bounded
critical-damped pose inertialization; dense seam video, not the failed IK
receipt, decides whether it is visually acceptable.  Evidence for the
phase-selected candidate:
`artifacts/global_terrain_showcase/coherent_curved_v8_phase_selected`.
Rigidly aligning the connector to the outgoing support sole was also rejected:
it required a 9.42 cm whole-clip translation and still missed the first sole
target by 22.6 mm, while slightly worsening the root step.  That is not a
substitute for selecting a genuinely compatible two-sided gait boundary.

This is a purely kinematic, globally privileged planned-course ceiling.  It
demonstrates curved flat travel and multiple terrain events without replaying
one whole scene clip, but it is not yet arbitrary live steering while a
terrain portal is committed and it is not a SONIC qualification.

The retained extended gauntlet adds a right S-curve, a turned ramp descent, a
diagonal-left approach, and an alternating up/down curb traverse.  The final
70.8 s sequence contains five terrain events and four independent
MotionBricks connectors, travels 24.47 m, and spans 9.73 m laterally.  It
passes all 3,541 frames with zero unsupported frames, 4.33 mm maximum
complete-foot penetration, zero forbidden-body penetration, 0.170 rad maximum
joint step, and 19.37 m/s^2 maximum root acceleration.  Exact-sole clearance
is 2.68 mm at p50 and 14.20 mm at p95; 112 frames exceed 20 mm and the maximum
is 35.85 mm.  Those outliers are concentrated in the first ascent, so the
numeric support pass was followed by a dense rendered-mesh review rather than
being treated as sufficient.

Five section sheets, four close foot sheets, and 10 fps sheets around all
eight external joins were reviewed.  They show continuous poses through the
joins, terrain beneath the robot, and no visible foot mesh embedded through a
tread.  The extended result is therefore retained as the current visual
showcase.  The naïve near-ground-foot-speed statistic was explicitly rejected:
it includes moving toe/scuff samples and is not a valid planted-foot slide
metric.  Evidence:
`artifacts/global_terrain_showcase/coherent_curved_v13_extended_gauntlet`.

### Held-out live waypoint route (2026-08-05)

The next branch removes curated flat connectors.  A polyline compiler now
leaves arbitrary flat legs under official live MotionBricks control and emits
only non-flat terrain events as committed portals.  Accepted route legs resume
without recomputation.  Route-equivalent accepted primitives may be rigidly
instanced after their relative support-height signatures agree within 3 mm;
the composed course still has to pass the existing exact complete-G1 audit.

The first portal-aligned S route is 10.76 m long with 1.50 m lateral span and
four heading changes.  Both shifted events compile: the ramp course has 2.03
mm maximum complete-foot penetration and the curb course 2.78 mm, with zero
forbidden-body penetration.  Dense entry/exit video windows show no teleport
or freeze.  A fully oblique ramp approach remains rejected by every retained
C490 candidate, correctly exposing data coverage rather than being hidden by
threshold relaxation.

An additive abrupt two-stick curriculum now provides 102 canonical traces
(204 after exact mirroring): translation reversals, yaw reversals, simultaneous
travel/facing changes, stop/restart, and slow/fast jumps at varied phases.  The
smooth omnidirectional curriculum is unchanged so robustness clips can be
mixed at an explicit training weight.  Each one-frame event is also stored as
a mirrored-invariant bit mask (abrupt, stop/restart, velocity reversal,
heading jump, and dual-stick jump), so distillation can oversample the hard
windows without oversampling every held frame of the clip.  The materialized
bank contains 204 clips / 122,400 frames and 816 labelled abrupt events (408
stop/restarts, 136 travel reversals, 408 heading jumps, and 340 dual-stick
jumps); all twelve SONIC input shards are prepared at
`/move/data/terrain-aware/sonic-rollouts/takara_bones_mm_abrupt_v1_task12`.

The retained headless live-control check starts the official MotionBricks
controller off-axis, steers to each globally known portal using evaluator-only
waypoint feedback, commits during terrain, and returns control after each
landing.  It captures both portals and the final waypoint over 14.56 m at
2.95 cm RMS / 6.15 cm p95 path deviation.  Every flat frame is generated live;
only the exact-audited ramp and curb traversals are committed.  It records raw
commands, actual root trajectory, intended-versus-actual map, and a rendered
exact-mesh video.  Evidence:
`artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/live_rollout_seed17_v4_forward_nudge`.

Dense review found a short exaggerated crouch when live MotionBricks
reacquires its internal gait after each authored terrain exit.  Preserving the
measured 0.5 s exit tangent and changing the slow/walk boundary both retained
2/2 completion but did not remove the crouch; those variants are rejected as
visual fixes.  This local state-distribution seam is not being hidden by the
route metrics and requires pose-compatible exit retrieval or learned
inertialization rather than another waypoint-controller tweak.

The crouch was subsequently localized to the vertical coordinate convention,
not the waypoint controller.  Both exits sit on support 0.25 m above the flat
MotionBricks training floor.  Canonicalizing the four-frame context to that
local support height before generation, then restoring the support offset once
per newly generated batch, reduces the two early continuation root drops from
about 0.33 m to 15.64 and 10.43 mm.  The corrected rollout still captures both
portals and the final waypoint.  Dense 25 fps exit review shows continuous
alternating steps with no deep squat, freeze, fall, or pose snap.  Path error
is 5.43 cm RMS / 16.48 cm p95, worse than v4 but accepted because the visible
state-distribution failure is removed without losing completion.  Evidence:
`artifacts/generic_terrain/waypoint_routes/canary_portal_aligned_s_v2/live_rollout_seed17_v9_support_once_move`.

### Five-event live terrain gauntlet (2026-08-05)

The retained runtime now uses entry-only terrain portals.  Each portal owns
the flat approach and exact non-flat traversal, then hands control immediately
back to live MotionBricks at the final authored pose.  The first four decoded
MotionBricks frames overlap the conditioning history and are discarded rather
than replayed.  A deterministic 16-phase search selects the least disruptive
future gait at each exit.  This removes the delayed steering and finite-landing
overshoot caused by prerecorded exits.

The mixed route contains five independently compiled events: ramp ascent,
curb up/down, four-step stair ascent, four-step stair descent, and ramp
descent.  Live MotionBricks generates all curved flat travel and turns between
them.  The accepted rollout captures all five portals, reaches the terminal
waypoint without a terrain-guard stop, travels 28.69 m over a 28.06 m planned
route, and has 4.24 cm RMS / 7.99 cm p95 route deviation.  It runs for 62.39 s.

Because clean MotionBricks poses and the G1 collision-sphere floor differ by
a few centimetres at some gait phases, the globally privileged ceiling now
applies the standard game-animation root-height projection: exact sole support
points are lifted only enough to retain 3 mm terrain clearance.  The lift is
2.96 cm maximum and 1.08 cm p95.  It does not change joint motion and is not
fed back into MotionBricks.  The same implementation is shared by the
headless evaluator and browser/Switch viewer.

An exact complete-G1 audit over all 3,120 resampled frames accepts the full
motion: 4.209 mm maximum foot penetration, zero forbidden-body penetration,
and zero threshold-exceedance frames.  Dense full-route, stair, and final-exit
video review shows continuous locomotion without a freeze, fall, body-through-
terrain event, or obvious final handoff snap.  The largest joint step is
0.357 rad at the final downhill-to-flat handoff, which remains the visually
riskier boundary.  Evidence:
`artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_v1/live_rollout_seed17_v8_exact_clearance`.

This is still a globally privileged, purely kinematic ceiling.  It proves that
live arbitrary flat steering and multiple exact-safe terrain traversals can be
combined without replaying whole prerecorded scenes; it is not yet a SONIC
tracking qualification or a causal onboard terrain selector.

### Interactive portal correction (2026-08-05)

The first browser version did not reproduce the headless route result.  It
committed at frame zero of a long authored approach, allowed the flat gait's
root to follow non-flat support, exposed only one traversal direction, and
derived lateral registration from small sideways wiggles in the authored root
trajectory.  In practice this locked out the joystick for seconds, floated or
carried the robot onto ramps, rejected off-centre stair captures, and made
reverse traversal impossible.

The corrected runtime commits 0.8 s before the terrain seam, keeps live flat
root height on its current support plane, exposes forward and reverse portals,
and shifts each portal only along the exact obstacle-lateral axis stored in the
route manifest.  It verifies the shifted course against the exact terrain
support profile before committing.  All ten directional courses retain the
same support profile at lateral shifts of -1.0, -0.5, +0.5, and +1.0 m.

An off-centre stair-ascent probe starting from an official MotionBricks flat
gait captures 16 cm before the late portal with 0.087 rad lower-body phase
error.  The complete ascent passes the exact G1 audit at 2.000 mm maximum foot
penetration and zero forbidden-body collision.  The corresponding reverse
ramp probe also captures from live flat gait and passes with zero foot or body
penetration.  Dense video review shows neither the reported freeze nor a
root-height teleport.  Evidence:
`artifacts/generic_terrain/interactive_probes/stairs_up_jit080_lane_flat_mb_v2`
and
`artifacts/generic_terrain/interactive_probes/ramp_down_jit_lane_flat_mb_v2`.

### Verified command invalidation and grounded backward-facing portals (2026-08-05)

The interactive failure in which a backward command worked but a subsequent
forward command kept following the old track was reproduced headlessly.  The
official MotionBricks agent deliberately retains a generated future for about
half a second before replanning, so an abrupt joystick reversal could continue
playing the retreat trajectory.  The shared headless/browser runtime now
compares the operator command with the command that actually generated the
live buffer.  Stop/start, large direction or speed changes, and large facing
changes invalidate that future; small joystick jitter does not.  A forced
replan is conditioned on the last four poses that were actually published and
accepted by the exact terrain checks, and the four decoded conditioning poses
are consumed rather than replayed.  This prevents both stale-track playback
and a four-frame rewind.

The no-match safety probe deliberately faces the G1 ninety degrees away from
every compatible portal, holds 36 commands into the obstacle, retreats, and
then reverses forward.  The corrected runtime blocks all 36 unsafe commands,
accepts zero portals, retreats 1.0146 m, and then advances 0.9059 m after the
reversal.  It records zero flat-support recoveries, zero foot/body penetration,
5.58 mm maximum nearest-sole clearance, 0.208 rad maximum 50 Hz joint step,
2.22 cm maximum root step, and 0.027 rad maximum root angular step.  A 48-frame
whole-video sheet and dense 25 fps windows around both command changes show a
continuous gait with no rewind, freeze, or flight.  Evidence:
`artifacts/generic_terrain/interactive_probes/no_match_retreat_v3_command_invalidation`
and the exact no-render acceptance replay
`artifacts/generic_terrain/interactive_probes/no_match_retreat_v4_command_invalidation`.

Course-exit contact transfer is now an atomic safety phase.  The selected
phase-matched future plays until the anchored support foot has transferred to
the landing; the newest joystick command is then replanned from verified
post-handoff poses.  The two-leg IK uses its hard per-joint temporal constraint
instead of merely checking the result afterward.  This retains the 0.22 rad
30 Hz handoff bound without loosening it.

The final five-event replay captures all five portals and the terminal
waypoint over 48.06 s / 24.43 m.  It has zero terrain-guard blocks, zero flat
support violations or recoveries, 2.779 mm maximum complete-foot penetration,
zero forbidden-body penetration, 4.97 cm maximum / 1.79 cm p95 nearest-sole
clearance, and 0.230 rad maximum 50 Hz joint step.  All five support handoffs
release; their maximum step is 0.219999 rad and maximum sole-target residual is
5.64 mm.  Route deviation is 6.65 cm RMS / 17.58 cm p95.  The whole 48-frame
route sheet, dense terrain windows, the long course-2 handoff window, and the
largest entry-blend window were reviewed without a visible teleport, freeze,
or body/foot-through-terrain event.  Evidence:
`artifacts/generic_terrain/interactive_probes/backward_facing_full_v13_bounded_atomic_handoff`.
The current source tree is replayed without rendering at
`artifacts/generic_terrain/interactive_probes/backward_facing_full_v15_final_code`.

Opposite-direction portal synthesis is now complete without admitting the
unsupported source approaches.  Two old entry-only source courses were
collision-free but hovered by 16--26 cm over their global approach lanes.
Their terrain traversals and landing phases were still useful: a genuine
phase-matched MotionBricks exit was generated at each grounded endpoint, the
complete motion was time-reversed, and only the grounded near-seam landing
tail was retained.  One generated flat exit required the same exact-support
root projection used by the live runtime (12.8 mm maximum) before reversal.
The final five-course complementary bundle has grounded endpoints on every
course and no support-clearance threshold exceedance.  Evidence:
`artifacts/generic_terrain/waypoint_routes/mixed_gauntlet_reverse_backward_facing_complete_v2`.

Its independent 48.63 s / 24.46 m replay captures all five portals and the
terminal waypoint.  It records zero terrain-guard blocks, zero flat-support
violations or recoveries, 4.088 mm maximum foot penetration, zero forbidden
body penetration, 4.969 cm maximum / 1.771 cm p95 nearest-sole clearance, and
0.240 rad maximum 50 Hz joint step.  All five exit handoffs release within the
0.22 rad hard handoff bound; path deviation is 6.87 cm RMS / 17.73 cm p95.
The full video plus dense windows around both repaired courses, the maximum
joint-step frame, and the maximum-clearance handoff show no visible rewind,
freeze, teleport, or flight.  Evidence:
`artifacts/generic_terrain/interactive_probes/backward_facing_complement_full_v1`.

A direct slope direction-reversal regression traverses one direction, changes
direction at the landing, and traverses back.  It captures both portals,
reaches the final waypoint, has zero guard blocks and support recoveries,
2.116 mm maximum foot penetration, 2.471 cm maximum support clearance, and no
repeated-pose run.  The dense reversal window is visually continuous.
Evidence:
`artifacts/generic_terrain/interactive_probes/slope_direction_reversal_roundtrip_v1`.

### Terrain-maneuver artifact audit and registered slope scale-up (2026-08-05)

The visible hover, foot glide, and cadence hitches in the first maneuver video
are treated as failures.  In particular, procedurally moving only the root of a
flat walk over `rolling_bumps`, `smooth_hill`, or `cross_slope_bumps` is not a
valid terrain adaptation.  The strict audit measures 13.99--30.33 mm p95 and
34.26--47.54 mm maximum intended-stance hover on those candidates.  They remain
quarantined until a support-aware leg/contact solve or genuine grounded source
motion passes the same admission gates.

A separate concrete bug was found in the prior varied-slope evidence.  C490
slope motions are expressed against the archive's approximately -90 degree
terrain yaw, but `slope_varied_up99_mid_v1` and
`slope_varied_down147_mid_v1` loaded the correct USD at identity.  The mesh then
extended along +X while the root travelled along -Y.  Those renders and the
associated SONIC slope probe are invalidated.  Direct-source construction now
looks up the exact terrain position and quaternion by USD path in the clean
archive, and a regression test covers the non-identity transform.

Four source motions were rebuilt against their correctly registered, distinct
meshes: target 88 uphill, target 99 uphill, target 118 downhill, and target 147
downhill.  Their eight stop/restart and reversal variants all pass the exact
complete-G1 collision, planted-foot drift, double-support hold, and mechanics
gates.  Maximum foot penetration is 4.447 mm, maximum accepted stance-run drift
is 5.743 mm, and forbidden-body penetration is zero.  Full-rate 50 Hz windows
around every hold boundary show no terrain teleport or new foot penetration.
The two uphill stop poses are visibly wide and are retained only as robustness
data; the downhill pairs and uphill reversals are the cleaner visual examples.
Evidence:
`artifacts/terrain_maneuver_pilot/slope_registered_up88_v1`,
`artifacts/terrain_maneuver_pilot/slope_registered_up99_v2`,
`artifacts/terrain_maneuver_pilot/slope_registered_down118_v2`, and
`artifacts/terrain_maneuver_pilot/slope_registered_down147_v2`.

The CPU-only twelve-source C490 scale-up completed as job `16532200`.  Seven
sources produced fourteen automatically admitted pilots; three had no central
supported pivot and two failed source mechanics, so they emitted no motion.
The admitted set has 4.918 mm maximum foot penetration, 3.590 mm maximum
stance-run drift, 0.141 rad maximum 50 Hz joint step, and zero forbidden-body
collision.  Dense 50 Hz review of all fourteen hold boundaries shows continuous
motion without the earlier terrain-relative hover or mesh mismatch.  Four
source profiles have a pivot-foot separation below 40 cm and supply eight
cleaner examples; three harder profiles supply six wide-support robustness
examples.  Registered terrain profiles span shallow roughness, a double hill,
crest/valley and plateau traversals, a single mound, and a sustained 46 cm
grade.  Evidence:
`artifacts/terrain_maneuver_bank/c490_slope_stratified12_v1` and its
`terrain_profiles.png` overview.

The current exact-grounded pilot set therefore contains twenty-eight
mechanically admitted candidates across two stair courses and twelve real
slope/hill geometries.  The valid stair SONIC probe still diverges, so noisy
rollout collection remains blocked on a clean-bank fine-tune; the old
misregistered slope tracker conclusion is intentionally not reused.

The twenty-eight pilots are packaged without rematching at
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/bundle`.
The bundle has 28 unique motion IDs, 14 unique source manifests, 22 clips with
the non-identity C490 terrain rotation, and a static object-motion file carrying
the same per-clip poses into SONIC observations and physics.  The first launch
correctly stopped on duplicate nested-manifest clip names; clip IDs now include
the source directory and a regression test covers that collision.

A five-iteration A5000 fine-tune smoke test completed as job `16532442` in 57
seconds.  It discovered all 28 terrains, instantiated 32 exact-terrain
environments, resumed the `terrain_release` checkpoint at step 20,000, completed
training, and wrote `model_step_000005.pt`.  Peak host RSS was 2.8 GB.  The
follow-up clean-bank run is job `16532461`, deliberately using one A5000, two
CPUs, 12 GB host RAM, and 384 environments rather than the inherited 64--96 GB
requests.  A one-rollout-per-clip clean diagnostic is chained as job
`16532493`; it retains failed episodes only so all 28 physical outcomes can be
measured and rendered, and its output is explicitly excluded from training.
Eight stratified third-person reviews (two stair courses, two registered
downhill sources, and four distinct rough/hill/slope profiles) were initially
chained as array job `16532504` with at most two A5000 tasks active.  That array
was cancelled before it started after the `move3` driver wedge; the review will
be resubmitted on L40 after the hard-example result.  No noisy collection is
scheduled before those clean results are audited.

The 400-iteration fine-tune completed successfully in 20m37s and wrote
`model_step_000400.pt` plus `last.pt`.  Its final randomized training timeout
rate was 90.47%, but the deterministic frame-zero gate is materially harder:
19/28 clips completed and 9/28 terminated.  Twelve of the fourteen newly
scaled C490 rough/hill/slope pilots completed.  The failures are all four old
stair pilots, both old composed-slope pilots, the registered target-118
downhill reversal, and the target-76 and target-136 reversals.  Median peak
MPJPE across all diagnostic episodes is 219.0 mm; diagnostic failures are
retained for review and remain barred from training data.

A nine-clip hard-example bundle was materialized from those exact failures.
The current v2 copy is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/hard9_bundle_v2`;
it replaces the target-76 and target-136 reversals with their shorter versions
while preserving all nine clip IDs.
The first continuation attempt was aborted before initialization when the
evaluation process wedged the NVIDIA driver during Isaac shutdown on `move3`;
the partial directory was preserved with suffix
`failed_driver_lock_16529858`.  L40 job `16532692` then failed before Isaac
launch because its assigned `move4` GPU could not initialize CUDA; that partial
directory is preserved with suffix `failed_cuda_init_16532692`.  The same
modest 192-environment continuation was rerun as Titan RTX job `16532758` on
explicit node `move2`.  Full-bank evaluation job `16532759` used separate
output and evaluated
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/bundle_v2`.
Its result is reported below.  The two earlier pending v1 jobs were cancelled
before allocation, so they consumed no GPU time.

Dense 25 fps review of the original five-window varied-terrain montage
confirmed that its apparent teleports occur at hard montage cuts, but also
confirmed a genuine source-side roughness: each 24/30/24-frame maneuver freezes
a supported mid-stride pose long enough to look unnatural.  A v2 builder now
permits reversal-specific timing.  The seven admitted varied-terrain sources
were rebuilt with a 16-frame deceleration, four-frame reversal pivot, and
16-frame acceleration; stop/restart uses 16/15/16.  All fourteen v2 motions
pass the exact collision/mechanics gates.  Across the reversal set, maximum
stance-run drift is 2.941 mm and maximum 50 Hz joint step is 0.123 rad.  Four
dense synchronized renders show continuous foot lock without the long frozen
pose.  Evidence: `artifacts/terrain_maneuver_bank/c490_slope_quick7_v2` and
`quick_reverse4_grid.mp4`.

The full v2 bundle contains all 28 candidates and keeps the same clip IDs as
v1.  Only the seven newly scaled source pairs use the revised transition
timing.  This makes the chained full-bank evaluation a per-clip physical A/B
comparison rather than a differently sampled benchmark.

A second CPU-only varied-terrain sweep ran as job `16532796`.  Twelve new C490
sources were selected from archive pelvis/terrain profiles rather than by
inventing heightfields: large hills, multi-crest/valley paths, repeated shallow
bumps, rough crests, and sustained grades.  Eight sources yielded fifteen
automatically accepted maneuvers; three lacked a central supported pivot, and
one failed the pilot mechanics gate.  Across the accepted set, maximum foot
penetration is 2.637 mm, maximum stance-run drift is 11.003 mm, and maximum
50 Hz joint step is 0.127 rad.  Seven representative reversals were rendered
dense at 50 Hz as job `16532832`; the contact sheets show continuous approach,
the short supported pivot, and continuous departure without the old long
mid-stride freeze.  This expansion remains a clean kinematic candidate bank,
not tracker-training data, until the current 28-clip physical gate finishes.
Evidence: `artifacts/terrain_maneuver_bank/c490_slope_varied12_v2`,
`terrain_profiles.png`, and `varied_reverse7_grid.mp4`.

The fifteen accepted motions are packaged separately at
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/varied15_bundle_v1`.
The first packaging attempt exposed a false negative in the binary-USD physics
check: compressed USD crates do not reliably expose schema names as raw byte
strings.  The copier now inspects crate stages with USD bindings when
available, confirms both rigid-body and collision APIs, and leaves the source
asset unchanged.  All fifteen bundled terrains were inspected as
`source_physics`; the interrupted partial is preserved with suffix
`failed_usdc_detection`.  A 15-environment clean physical diagnostic is job
`16533077`, chained after the v2 all-28 gate so it consumes no extra concurrent
GPU lane and cannot run against an unqualified checkpoint.

The 200-iteration hard-nine continuation completed as job `16532758` in
17m31s.  It reached 87.91% timeout completion in its final randomized training
window and learned six of the nine parent failures.  The independent full-bank
gate, job `16532759`, nevertheless passed only 6/28 clips, with median peak
MPJPE 587.6 mm and maximum 1744.6 mm.  It forgot nearly the entire easy set and
is rejected.  The varied-15 child `16533077` was cancelled before simulator
startup and wrote no output.  The fair parent-on-v2 evaluation, L40S job
`16533290`, passes 20/28 clips (median peak MPJPE 232.9 mm, maximum 1261.2 mm),
so the shorter v2 transition timing gains one pass over v1.  The step-50 hard
checkpoint, job `16533291`, passes only 2/28 (median 816.1 mm, maximum
2575.5 mm).  Hard-only training is therefore destructive before as well as
after convergence; step-100 job `16533292` was cancelled during startup.

A balanced replay bundle was built directly from those fair v2 outcomes at
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/replay44_bundle_v1`.
It contains all 28 clips once and two additional stable aliases of each of the
eight failures: 44 clips total, 24 hard instances and 20 retained passes.  All
44 motion, object-motion, scene-binding, and provenance IDs were checked.  The
100-iteration continuation, job `16533346`, passes 24/28 in the independent
full-bank diagnostic `16533347`, with median peak MPJPE 164.7 mm and maximum
1395.8 mm.  It retains every one of the parent's twenty passes and fixes five
of the eight parent failures.  Balanced anti-forgetting replay is therefore a
real improvement over hard-only continuation.

A second bounded replay attempted to focus the remaining four failures while
keeping all 28 clips.  Two L40 starts (`16533397` and `16533482`) stalled during
Isaac initialization on the same move4 device and were preserved as
infrastructure failures.  The healthy Titan retry `16533556` completed, but its
full-bank diagnostic `16533557` passes only 22/28.  It regresses two stair
motions and is rejected; the 24/28 replay-44 checkpoint remains the qualified
core parent.

Bundle preparation now writes `objects.pkl` directly.  Every constant object
trajectory uses the same per-clip terrain position and quaternion as
`clips.json`, so SONIC reset state and scene registration share one source of
truth.  The varied-15 bundle was rebuilt and all fifteen object-motion entries
were checked against their scene bindings.

The replay-44 parent does not generalize physically to the new terrain bank:
clean diagnostic `16533876` terminates on all 15 motions within 33--82 policy
steps.  This is tracker distribution shift, not a kinematic admission result.
A joint bank therefore keeps the 28 core motions once and the 15 varied motions
twice (58 replay entries, 43 unique evaluation motions).  Its 100-iteration
continuation is job `16533939`.  The step-50 full gate, job `16534018`, passes
35/43: 25/28 core and 10/15 varied, with median peak MPJPE 179.4 mm.  The
step-100 gate, job `16534019`, passes 36/43: 24/28 core and 12/15 varied, with
median peak MPJPE 254.1 mm.  Step 50 favors core fidelity; step 100 covers two
additional varied reversals.  Several surviving reversals still exceed 0.9 m
peak MPJPE, and three varied motions still terminate at step 100, so neither
checkpoint is approved for noisy collection.  The saved-state visual audit
below confirms that these are genuine tracking limitations rather than a
metric-only false alarm.

The cleaner varied-terrain kinematic showcase excludes the two clip-82 motions
and the clip-142 stop/restart from its presentation set because their stance-run
drift exceeds 5 mm, while retaining them as stress tests.  Evidence is
`artifacts/terrain_maneuver_bank/c490_slope_varied12_v2/varied_reverse6_showcase_grid.mp4`;
full-length renders are generated separately so short pivot windows cannot hide
cadence or contact artifacts.

All six full-length clean showcase renders completed as CPU-only job `16533955`
and are synchronized in
`artifacts/terrain_maneuver_bank/c490_slope_varied12_v2/varied_reverse6_full_grid.mp4`.
They cover real registered hill/grade, crest/valley, repeated-bump,
multi-crest, and sustained-slope geometry.  Dense review confirms that the
16/4/16 reversal removes the old long mid-stride freeze.  A small supported-foot
shuffle remains visible on some pivots, so these examples are cleaner rather
than artifact-free.

The step-100 saved-state physical audit was reconstructed on the exact bundled
terrain as CPU jobs `16534237` and `16534313`; this avoids spending a GPU merely
to replay saved states.  The labeled comparison is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/tracker_review_core43_replay58_step100_varied6_cpu_v1/varied_physical6_audit_grid.mp4`.
The three deterministic failures remain upright initially but gradually drift
or stumble and terminate; they are genuine physical failures, not montage cuts.
The reviewed survivors complete their stop/reversal without teleporting or a
long freeze, although occasional stance-foot shuffle and large path error remain.
This confirms useful varied-terrain coverage but does not clear either replay-58
checkpoint for noisy collection.

A final bounded repair keeps the same 43 unique motions, retains the second
copy of every varied motion, and gives the three remaining varied failures four
total replay copies (64 entries).  It resumes the replay-58 step-100 model at a
four-times-smaller actor learning rate (`5e-6`) with adaptive bounds
`1e-6`--`2e-5`.  Healthy Titan job `16534785` completed 50 iterations after an
unsupported V100 start and a stalled move4 start were isolated without writing
checkpoints.  The step-25 full gate, job `16534786`, is the best survival result:
38/43, split 25/28 core and 13/15 varied, with 246.6 mm median and 1180.7 mm
maximum peak MPJPE.  Continued training is destructive: step-50 job `16534787`
passes 35/43, split 21/28 core and 14/15 varied.  Step 50 is rejected.  Step 25
is retained as the broadest diagnostic tracker, but its high-error reversal
survivors still prevent noisy collection.

The final step-25 physical audit was replayed without another GPU allocation as
CPU job `16534984`.  Its labeled full-length grid is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/tracker_review_core43_replay64_step25_varied6_cpu_v1/step25_physical6_audit_grid.mp4`.
The ordinary stop and reversal survivors are temporally continuous through the
pivot, with no teleport or old long hold, but a small supported-foot shuffle is
still visible.  The orange clip-94 survivor is the key counterexample to using
binary survival alone: it remains upright and looks locally continuous while
departing from the reference by 1181 mm.  The three labeled failures likewise
drift or terminate rather than exhibiting a hidden montage cut.  Visual review
therefore agrees with the numeric gate: step 25 is useful for diagnosis, not for
noised data collection.

### Stairs500 stop/reversal scale-up (2026-08-06)

The clean stairs500 archive contains 500 full G1 traversals and 249,500 frames
(about 83 minutes at 50 Hz), split evenly between ascent and descent.  It spans
3--8 steps, 0.100--0.249 m risers, 0.218--1.007 m treads, and multiple registered
approach angles.  The existing `grail_stairs500_sonic_v1` corpus already supplies
large quantities of ordinary noisy stair tracking, so this scale-up targets the
missing joystick events rather than recollecting unchanged traversals.

A deterministic source selector chose 120 ascents and 120 descents.  CPU-only
job `16539799` completed the exact-mesh sweep for all 240 sources.  Sources with
centimetre-scale penetration in the nominally clean archive were rejected rather
than repaired beyond the bounded clearance allowance.  The remaining 174 source
manifests contain 332 automatically admitted maneuvers: 97 descent stops, 101
descent reversals, 66 ascent stops, and 68 ascent reversals.  Every admitted
motion preserves its original registered terrain and passes the complete-G1
collision, mechanics, hold-support, and planted-foot gates.

The training bank is a deliberately balanced subset rather than all 332 motions.
`selected200_v1.json` contains 50 examples in each ascent/descent x
stop/reversal cell, selected by farthest-point coverage of family, step count,
riser, tread, approach angle, and elevation change.  It contains 200 motions
from 115 distinct source traversals, with 3--8 steps, 0.100--0.238 m risers,
0.230--1.007 m treads, and -40.15 to +34.72 degree approach offsets.  Maximum
foot penetration is 4.954 mm, maximum per-frame planted-foot step is 2.174 mm,
and maximum contiguous stance drift is 11.117 mm.  Stop/restart clips are 10.62
seconds; reversal medians are 9.84 seconds uphill and 10.54 seconds downhill.

The exact artifacts are:

- generated bank:
  `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/full_bank_v1`;
- balanced selection:
  `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_v1.json`;
- SONIC-native bundle:
  `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_bundle_v1`;
- labeled eight-motion clean review:
  `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_kinematic_review_v1/selected200_kinematic_review8_grid.mp4`.

The eight full-trajectory reviews cover two examples from every balance cell.
They show grounded approach, stair traversal, supported stopping or reversal,
and departure without the old terrain-transform mismatch, root-only hover, or
pose teleport.  The short reversal hold can still contain a small supported-foot
shuffle, so these are robustness motions rather than claims of perfect human
style.

Per-motion specialist tracking also closes two of the four remaining legacy
hard cases.  The clip-136 reversal completes at 291.6 mm peak MPJPE, and the
low-rate clip-94 reversal completes at 246.5 mm.  Clip-127 survives its entire
trajectory but remains excluded at 444.7 mm; the old grounded stair stop still
terminates early.  Together with previously qualified routes, the legacy
43-motion set is therefore 41/43 under the strict full-survival and at-most-300
mm gate.  The two misses should be replaced by cleaner stairs500 candidates,
not used to weaken the gate.

The first 200-motion SONIC launch used 256 environments and was stopped at
iteration 75 after its scene receipt exposed a 2:1 weight for 56 clips.  Its
directory is preserved as `sonic_selected200_finetune_v1.biased_env256_16542678`
and is not a candidate checkpoint.  Equal-weight job `16542899` uses exactly
200 environments, 200 motions, and 200 unique terrain USDs from the original
`terrain_release` tracker.  Checkpoints at 50, 100, 200, 300, and 400 iterations
receive separate deterministic all-200 gates; no noised rollout or depth job is
released merely because training completed.

All five deterministic gates are complete.  A pass requires both full-clip
survival and peak MPJPE at most 300 mm.  The checkpoint progression is:

| iteration | survived | strict passes | median MPJPE | p90 MPJPE |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 116/200 | 94/200 | 227.0 mm | 485.4 mm |
| 100 | 126/200 | 105/200 | 192.7 mm | 523.8 mm |
| 200 | 150/200 | 116/200 | 199.7 mm | 501.6 mm |
| 300 | 148/200 | 115/200 | 214.6 mm | 490.4 mm |
| 400 | 157/200 | **120/200** | **185.4 mm** | **485.5 mm** |

Iteration 400 is the best single universal checkpoint.  Its strict passes are
48/50 ascent stops, 46/50 descent stops, 7/50 ascent reversals, and 19/50
descent reversals.  The union over all five checkpoints reaches 137/200, but
the residual failures remain concentrated in reversal rather than ordinary
stair locomotion.  This supports routing offline collection by maneuver family
instead of forcing one tracker to cover incompatible events.

The first labeled physical-audit video in
`selected200_step400_physical_review8_v1` is invalid.  Its offline renderer
treated SONIC's stored joint offsets as absolute joint angles instead of adding
`meta/default_joint_pos`, which placed the rendered feet 5--8 cm inside the
stairs even though those were not the saved simulator poses.  The corrected
review is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_step400_physical_review8_fixed_v2/step400_physical_review8_fixed_grid.mp4`.
Offline forward kinematics after the fix agrees with the stored SONIC body
positions to within 0.01 mm.  Exact-mesh reconstruction of all 200
deterministic rollouts has 4.823 mm median and 8.154 mm maximum foot contact;
199/200 have zero forbidden-body penetration.  The sole exception is the
clip-191 reversal (3.383 mm forbidden-body contact), which subsequently
produced no accepted noisy rollout and is absent from the retained batch.
The deterministic receipt is `clean_eval_selected200_v1/geometry_audit_v1/summary.json`.

The 120 iteration-400 strict passes have been repackaged as
`selected120_step400_qualified_bundle_v1`.  A separate equal-weight 100-motion
reversal tracker was fine-tuned from universal iteration 200, because further
universal training already showed diminishing returns on the hard ascent
reversals.  Its deterministic progression is:

| specialist iteration | survived | strict passes | up strict | down strict |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 59/100 | 28/100 | 10/50 | 18/50 |
| 100 | 63/100 | 36/100 | 14/50 | 22/50 |
| 200 | **70/100** | 48/100 | **20/50** | 28/50 |
| 300 | 66/100 | **49/100** | 19/50 | **30/50** |

Iteration 300 is the best single reversal checkpoint, while iteration 200 is
retained for its slightly better uphill coverage.  The union of the specialist
checkpoints qualifies 66/100 reversals; routing across both universal and
specialist checkpoint families qualifies 168/200 motions overall.  For the
first noised pilot, iteration 400 supplies 120 motions and specialist
iterations 300, 200, and 100 supply 28, 11, and 3 additional non-overlapping
reversals.  No rejected motion is admitted to those bundles.

The iteration-400 noised pilot produced 236 accepted full rollouts from 118 of
the 120 qualified motions: two rollouts per retained motion, 116,874 frames, or
38.96 minutes at 50 Hz.  All stored episodes have `episode_failed=false`; peak
MPJPE has median 195.5 mm, p90 267.7 mm, and maximum 298.7 mm.  The retained
coverage is 48 ascent stops, 46 descent stops, 7 ascent reversals, and 17
descent reversals.  Only the clip-135 and clip-191 descent reversals remained at
zero successes after 32--34 noised attempts each, so they are excluded rather
than causing the other 236 successful rollouts to be repeated.

The original noisy-review video in `noised_selected120_step400_review4_v1`
has the same relative-versus-absolute joint-angle rendering bug and must not be
used.  The corrected stratified review is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/noised_selected120_step400_review4_fixed_v2/noised_selected120_step400_review4_fixed_grid.mp4`.
An exact-mesh post-audit of all 236 retained noisy episodes gives 5.356 mm
median, 6.733 mm p90, and 9.132 mm maximum foot contact, with zero
forbidden-body penetration in every episode.  The 5 mm source-kinematics gate
is intentionally not reused as a rigid-contact rollout gate: deterministic
physical tracking itself reaches 8.154 mm.  A separate 10 mm physical foot
compliance limit plus zero forbidden-body contact admits all 236 retained
rollouts without concealing the distinction between the two gates.
The complete receipt is
`noised_selected120_step400_perclip2_v1/geometry_audit_physical10_v2/summary.json`.

### Stairs500 stair-local two-stick expansion (2026-08-06)

Straight stair traversals are insufficient for joystick distillation: they do
not show diagonal travel, steering while already on the staircase, backward
ascent/descent, or velocity and body facing as independent controls.  The
first directional sweep attempted 80 distinct quality-vetted clean stairs500
traversals, but only 46 source traversals contributed at least one accepted
motion to the 856-motion bank.  It maps each source into the registered
stair-local frame, fits a
bounded lateral route, and exports per-frame world/local planar command
velocity plus an independent facing yaw.  No global root position is part of
the policy command interface.

The warp is contact-aware rather than a root-only transform.  Source stance
runs are detected from full-foot support and speed, successive sole targets are
anchored in world space, the pelvis is corrected toward the active support
polygon, and G1 leg IK fits the authored pose to those targets.  Every candidate
must then pass joint-step, root-acceleration, planted-foot drift, full-G1 exact
mesh, and forbidden-body collision gates.  Mechanical rejects skip the costly
mesh pass.  Clean temporal reversal supplies exact backward ascent/descent;
exact reflection about each registered stair centerline mirrors root, command,
quaternion, joints, left/right contacts, and terrain-relative fields.

The important command families are:

- diagonal and lane-changing paths;
- slalom and zigzag paths, including variants that rotate the pelvis while
  steering on the stairs;
- independent facing and counter-facing weaves;
- fixed-facing oblique travel, where the route changes direction but the torso
  keeps the authored stair heading;
- exact backward-time versions of all mechanically supported families.

The final pre-mirror gate admitted 428 motions.  All 428 exact reflections pass,
giving 856 balanced motions: 800 directional/non-straight examples and 56
backward-straight controls.  The split is 248 up/native, 166 up/reverse-time,
166 down/native, and 276 down/reverse-time.  Every left/right mode count is
identical after reflection.  The realized lateral envelope reaches 0.926 m.
Notable paired counts are 52 soft fixed-facing, 40 medium fixed-facing, 22 hard
32-degree fixed-facing, 6 lane changes, 74 turning slaloms, 88 turning zigzags,
and 62 micro facing weaves.  The full map is
`omnidirectional_sweep28_post_v1/all_routes.png`; solid lines are realized G1
roots, dashed lines are intended routes, colored arrows are travel, and black
arrows are torso facing.

The stronger fixed-facing construction matters.  A representative 24-degree
ascent realizes 0.417 m lateral travel with 4.13 mm maximum stance target error
and 2.33 mm stance drift.  A representative 32-degree descent realizes 0.569 m
lateral travel with 1.04 mm maximum stance target error and 0.20 mm stance
drift.  Both pass the exact mesh gate and dense side/overhead review without a
teleport, hover, or body/stair intersection.  Their videos are under
`travel_fixed_heading_visual1_v1/{side,overhead}`.  In contrast, blindly
turning the pelvis through 32 degrees was overwhelmingly outside IK/stance
bounds; its unstarted low-yield sweep was stopped and replaced by the
data-supported fixed-facing construction.

The visual gate is deliberately non-basic.  `review32.json` contains paired
fixed-facing oblique ascents/descents, extreme lane changes, independent-facing
motions, turning weaves, native diagonals, backward diagonals, and two real
stair cases.  It alternates original and exact-mirrored examples within paired
strata.  Side and overhead render arrays plus stop/restart/reversal transition
generation consume this same manifest.

A six-motion early SONIC canary established learnability before the final bank:
the stable 200-motion parent is 0/6 zero-shot; a 50-iteration equal-weight
fine-tune gives 4/6 strict passes and 5/6 full survivors.  All six saved physical
rollouts pass the exact geometry audit with zero forbidden-body penetration and
at most 6.02 mm foot contact.  Continuing to iteration 100 falls to 3/6, so the
universal run must evaluate early checkpoints rather than assume longer is
better.  A separate four-motion fixed-facing canary covers oblique ascent,
backward oblique descent, and both lateral polarities.  Its 50-iteration
checkpoint survives only 2/4 motions.  At iteration 100 all 4/4 survive 499
frames and all four pass the exact geometry audit with zero forbidden-body
penetration.  Maximum foot contact is 5.46 mm.  The backward clip still has a
906.6 mm peak MPJPE spike, so survival alone is not being mistaken for faithful
tracking.

The four-motion canary is intentionally not the visual proof for the expanded
command space.  A separate 32-motion qualification bundle is materialized at
`omnidirectional_sweep28_post_v1/sonic_bundle_exotic_review32_v1`.  It contains
paired fixed-facing obliques, lane changes, independent-facing motions, turning
slaloms and zigzags, native diagonals, backward diagonals, ascents, descents,
and exact left/right mirrors.  A 100-iteration one-A5000 physical qualification
and its deterministic evaluation/render/audit chain are jobs
`16554664`--`16554667`.  The deterministic result is 21/32 full survivors.
All 32 saved episodes pass the exact geometry audit with zero forbidden-body
contact; maximum foot contact is 7.82 mm.  Turning slaloms/zigzags, native
diagonals, forward lane change, fixed-facing travel, and a facing weave survive.
The 11 failures cluster in backward obliques and counter-facing cases.  The
intended-versus-physical route sheet is
`exotic_review32_a5000_ft_step0100_routes_v1.png`.

A diagnostic 100-iteration continuation on only those 11 failures is rejected:
although its hard-subset training timeout rate rises to 72%, the complete
32-motion reevaluation retains only 4/32 no-fall classes.  This is catastrophic
forgetting, not an acceptable tracker.  Therefore hard-only fine-tuning is not
the expansion recipe; hard motions must remain mixed with the stable replay and
the other successful directional families.

The universal SONIC bundle intentionally uses all 856 exact-accepted motions,
not the 382-motion gold-only subset: the visually clean largest-angle examples
are often silver because they require larger bounded joint correction.  It also
includes the stable selected200 replay bank and 62 accepted stop/restart or
mid-stair reversal transitions derived from `review32.json`, for 1,118 motions
total.  The target is
`omnidirectional_sweep28_post_v1/sonic_bundle_replay200_all_transitions32_v1`.
Training will start from `sonic_selected200_finetune_v1/model_step_000400.pt`,
save early checkpoints, and qualify motions only after deterministic survival,
MPJPE, exact-mesh, and dense visual gates.  No noised collection is released
from training completion alone.

SONIC paired-terrain training does not rotate a smaller environment pool
through a larger motion library: environment `k` remains bound to motion and
terrain `k`.  Consequently, a nominal 192-environment run over this bundle
would only train the leading 192 examples, regardless of iteration count.  The
universal launcher derives the coverage count from `clips.json`.  PPO requires
the environment batch to divide evenly over four minibatches, so the 1,118
clips use 1,120 paired environments: every clip occurs once and the first two
wrap once, with no omitted tail.  The initial exact-1,118 preflight reached all
terrain construction but correctly stopped at the indivisible minibatch.  Job
`16554807` is the corrected five-iteration A5000 preflight; the queued L40
launcher uses the same corrected rule.  Once that preflight passes, job
`16554900` trains the complete mixed bank for 400 iterations on one A5000 and
saves every 50 iterations for early-checkpoint qualification.  If it does not
fit, the fallback is explicit balanced bundles no larger than the environment
count, not a partial run mislabeled as full-bank training.

### Universal directional tracker and 240-source expansion (2026-08-07)

Job `16554900` completed all 400 iterations over the 1,118-motion mixed bank.
The deterministic 342-motion exotic stress gate improved from 200/342 full
rollouts at iteration 50, to 247/342 at iteration 200, and **277/342** at
iteration 400.  Iteration 400 is therefore the retained checkpoint.  It
completes 221/280 direct directional motions and 56/62 stop/restart or reversal
transitions.  The direct split is 144/146 native-time motions and 77/134 exact
backward-time motions, making backward locomotion the remaining failure
concentration rather than ordinary or forward directional stairs.

The surviving direct families at iteration 400 are 59/74 turning slaloms,
68/88 turning zigzags, 34/42 counter-facing traversals, 4/6 crab traversals,
5/6 lane changes, 18/22 hard fixed-facing obliques, 31/40 medium fixed-facing
obliques, and 2/2 medium diagonals.  Median peak MPJPE among all full rollouts
is 155.7 mm.  The selected route audit is
`exotic_stress342_all1118_a5000_step0400_routes18_v1`; it shows intended and
physically tracked routes for left/right diagonals, hard obliques, lane
changes, slaloms, zigzags, counter-facing motion, and crab motion.

Exact-mesh auditing of all 342 saved iteration-400 rollouts passes 342/342.
There are zero forbidden-body intersections and zero threshold-exceeding foot
frames.  Maximum foot penetration is 9.14 mm, median is 4.73 mm, and p95 is
6.66 mm under the separate 10 mm rigid-contact rollout threshold.

The same checkpoint also improves the original 200-motion stable replay bank:
193/200 motions complete, and 182/200 both complete and remain at or below
300 mm peak MPJPE.  The stable-only parent achieved 157/200 completion and
120/200 strict passes.  The new family breakdown is 49/50 uphill stops, 49/50
uphill reversals, 49/50 downhill stops, and 46/50 downhill reversals.  Thus the
directional mixture strengthened rather than erased ordinary stair tracking.
The exact-mesh audit passes all 200 saved replay rollouts with zero forbidden
body contact and zero foot-threshold violations; maximum foot penetration is
7.47 mm.

The 856-motion bank nevertheless repeats too few physical sources for the
intended collection scale.  The next expansion consumes all 240 already
prepared clean stairs500 source traversals.  A dense tier tries ten-degree
fixed-facing travel and eight-degree simultaneous turning in both directions;
the hard tier tries medium/hard fixed-facing travel, lane and crab motion,
counter-facing motion, turning slaloms and zigzags, and medium diagonals, in
native and exact backward time.  Each accepted motion is exact-mesh gated and
then reflected with the same full-state symmetry transform.  Generation,
mirroring, combined export, route plotting, and paired side/overhead 32-motion
review are chained as jobs `16554955`, `16555019`, `16555045`, `16555265`,
`16555343`--`16555346`, `16555359`, and `16555377`--`16555378`.  The combined
output is `omnidirectional_240source_post_v1`.  Counts from this expansion are
not recorded until all mechanical rejects and exact mirrors have finished.

### Compound on-terrain maneuvers and behavior-preserving scale-up (2026-08-07)

The first stop/reversal generator chose one globally best pivot from the full
motion.  That was insufficient for joystick data: directional source files do
not always carry seams, so the chosen pose could lie on a long flat approach or
exit, and one pivot per traversal concentrated every command change at the same
height.  The composer now localizes monotonic stair motion from the endpoint
height plateaus, divides the true terrain interval into progress bands, and
selects only measured double-support poses within those bands.  Ordinary
stop/restart and reverse schedules are emitted at every supported band.  A new
`bounce` schedule advances to an upper supported pose, decelerates, retreats to
a lower supported pose, and resumes forward travel.  It changes only time along
an already exact-grounded source path and never interpolates between unrelated
animations.

The corrected eight-source canary places every pivot between 20.8% and 79.5%
of the detected stair interval.  It admits 25/25 generated pilots, with no
forbidden-body penetration, 3.77 mm maximum foot penetration, 1.90 mm maximum
per-frame stance step, and 0.197 rad maximum joint step.  Only one source has
two sufficiently separated double-support windows for a full bounce; this is a
real source-coverage limitation rather than a reason to shorten the smoothing
ramp or relax contact gates.

A broader 64-motion directional selection then produced 308 compound pilots.
294 pass the automatic exact/mechanics gates and 263 pass the stricter showcase
gate of at most 5 mm foot penetration, 10 mm stance-run drift, 3 mm stance step,
0.20 rad joint step, and 30 m/s² root acceleration.  The admitted set contains
36 bounces, 129 reversals, and 129 stop/restarts.  Its source paths reach 24
degrees across the stair axis and 0.650 m lateral travel; source facing offsets
reach 12 degrees.  Dense review of the strongest 24-degree example shows no
foot-through-step event.  Side and overhead cameras are now explicit renderer
options because the old front-follow camera visually hid lateral motion.

The source selector itself was also too coarse: it grouped crab,
counter-facing, and facing-weave motion together, and grouped lanes with
diagonals.  The corrected selector preserves twelve distinct joystick
families: diagonal, lane, crab, counter-face, facing weave, fixed-facing
oblique, facing-only, path slalom, path zigzag, turning, turning slalom, and
turning zigzag.  The new 64-source compound bank is
`compound_exotic64_behavior_split_v3`; its strict 32-motion side review and
12-motion overhead review are jobs `16558233` and `16558234`.  The completed
240-source expansion will feed the same behavior-preserving selector into a
96-source compound bank automatically (jobs `16557409`--`16557412`).

The same short 16/4/16 timing was scaled over every registered C490 continuous
terrain source rather than invented heightfields.  Of 77 real hill, grade,
crest/valley, repeated-bump, and rough-slope sources, 65 admit motions.  The
bank contains 376 pilots, 364 automatic admissions, and 354 strict admissions:
52 bounces, 159 reversals, and 153 stop/restarts.  Maximum automatically
accepted foot penetration is 4.99 mm and forbidden-body penetration is zero.
The strict terrain-diverse 24-motion review uses 17 physical sources and is
balanced as 12 bounces, six reversals, and six stop/restarts.  Its bank and
profile overview are `c490_slope_compound77_onterrain_v2` and
`c490_slope_compound77_onterrain_post_v2/terrain_profiles.png`.

The registered curb bank supplies another 72 source geometries.  Fifty-one
sources admit 278 exact/mechanical pilots, of which 203 pass the strict
showcase gate: 37 bounces, 134 reversals, and 107 stop/restarts before the
strict subset is applied.  The terrain-diverse 24-motion visual review is under
`c490_curb_compound72_post_v1`.

A separate experiment attempted to bend complete C490 hill traversals laterally
and re-anchor rigid stance soles directly to the continuous mesh.  Only 7/144
micro variants passed across three of twelve tested geometries; 31 left the
finite terrain mesh and 106 violated the unchanged mechanics/contact gate.
That method is retained as a diagnostic but is **not** being scaled or counted
as general omnidirectional rough-terrain coverage.  The reliable current
coverage is exact source traversal plus compound timing on real terrain, and
exact support-aware path/facing warps on stairs.

### Paired terrain-motion co-warps (2026-08-07)

The failed fixed-mesh lateral warp identified the wrong invariant: it bent the
robot route while leaving unrelated terrain under the stance feet.  The paired
construction instead applies one smooth path map to both the registered mesh
and the motion.  Swing feet retain their pelvis-local transform, while every
stance sole is attached to the exact source face by barycentric coordinates and
mapped to the corresponding target face.  This preserves authored contact
without assuming an infinite heightfield or inserting a root-only clearance
offset.  Every output is still rejected on excessive joint/root steps, IK
correction, stance error/drift, fewer than two stance support probes, more than
5 mm foot penetration, or any forbidden-body penetration.

The completed 32-source stair sweep attempted 400 motions and accepted 100:
48 ascents and 52 descents across all sixteen requested command families.  They
include left/right diagonals, 0.25 m lane shifts, gentle and stronger slaloms,
zigzags, independent facing offsets, and path/facing combinations in opposite
directions.  The admitted envelope reaches 28 degrees of path steering, 0.50 m
of lateral-offset range, and 10 degrees of independent facing.  A separate
0.40 m lane tier accepts another 15 motions, split as eight ascents and seven
descents, with path angles up to 24 degrees.  Maximum foot penetration is
4.985 mm in the main bank and 4.924 mm in the stronger lane tier; forbidden
body penetration is zero throughout.  The main artifacts are
`cowarped_directional32_amp025_v1/{summary.json,all_routes.png,routes_by_mode.png}`.
The mode-separated route plot uses the unwarped source heading as its reference
so diagonal motion is not falsely rotated back onto a straight axis; realized
G1 roots are solid blue and intended roots are dashed red.

Timing composition is now applied to these paired assets rather than looking
up the original straight archive terrain.  Only frames marked as authored
double stance with at least two target support probes under each sole may be a
stop or reversal pivot.  Out-and-back rough paths restrict pivots to the first
monotonic passage.  Retimed files preserve the intended path, contact masks,
support counts, and audit traces, and they recompute both world and robot-local
velocity labels plus facing and exact zero-stick holds.  The first 24-motion
stair subset produces 120 pilots, of which 112 pass the strict gate: 17
bounces, 49 reversals, and 46 stop/restarts across all sixteen spatial command
families.  They use 13 physical stair sources, have zero forbidden-body
penetration, and reach 4.923 mm maximum foot penetration.  The completed
100-source expansion attempts 422 event motions and strictly admits 370: 47
bounces, 178 reversals, and 145 stop/restarts.  Those span all sixteen source
command families, 14 physical stairs, 196 ascents and 174 descents; maximum
foot penetration is 4.985 mm and forbidden-body penetration is zero.

A deliberately stronger stair canary adds 38-degree hard diagonals, held lane
changes, two-stage left/right lane changes, and slaloms whose travel and facing
commands move in opposite directions.  All eight new command families are
represented among 13 admitted motions from four source clips.  The realized
envelope reaches 0.857 m of lateral range and 12 degrees of independent facing;
maximum foot penetration is 4.662 mm and forbidden-body penetration remains
zero.  Dense side and overhead renders of the hard diagonal and held-lane cases
show continuous, grounded traversal rather than the earlier root-only slide.
The 32-source scale and its stop/reverse/bounce compound pass are complete.

That 32-source stronger scale accepts 18 motions from eight physical stair
sources: all eight hard-diagonal, held-lane, step-change, and counter-facing
families are present, with six ascents and twelve descents.  It retains the
38-degree/0.857 m envelope, reaches 12 degrees of independent facing, and has
4.783 mm maximum foot penetration with zero forbidden-body penetration.
Its compound pass attempts 63 motions and strictly accepts 50: four bounces,
26 reversals, and 20 stop/restarts from six physical stairs, split as 19
ascents and 31 descents.  All eight stronger source command families remain
represented after the exact retiming audit.

A complementary hard-crab tier explicitly separates the two sticks.  Its
diagonal path reaches 24 degrees while a 24-degree opposing facing command
keeps the pelvis closer to stair-forward at peak obliquity; straight-path
facing weaves reach +/-20 degrees.  Scaling over the fifteen previously useful
stair sources accepts twelve motions from four physical stairs, evenly split
between ascent and descent and covering all four crab/facing families.  The
maximum lateral range is 0.489 m, maximum foot penetration is 4.746 mm, and
forbidden-body penetration is zero.  Side and overhead renders of the crab
descent and its stop/restart retime show no visible pop or terrain crossing.
The strict crab compound pass retains 13/16 event motions: seven reversals and
six stop/restarts across all four crab/facing families.  Twelve are descents
and one is an ascent; rejected ascent retimes are not forced into the bank.

The rough-terrain 0.40 m lane tier accepted 33 direct paired motions from 21
physical curb/slope sources.  Its compound pass attempted 167 event motions
and strictly admitted 143: 19 bounces, 69 reversals, and 55 stop/restarts from
20 physical sources.  The bank contains 24 uphill and 119 round-trip examples,
with 4.794 mm maximum foot penetration and zero forbidden-body penetration.

The broader 0.25 m curb/slope directional pass admits 194 motions from 33
physical sources and covers every one of the sixteen velocity/facing command
families.  Its envelope reaches 26.7 degrees of path angle, 0.50 m lateral
range, and 10 degrees independent facing; maximum foot penetration is 4.991
mm with zero forbidden-body penetration.  Rather than recomputing its already
completed first compound half, review rows 32--63 were retimed separately.
That novel half strictly admits 141/160 events from 21 physical sources: 19
bounces, 67 reversals, and 55 stop/restarts, including 14 uphill and 127
round-trip motions.  Its maximum foot penetration is 4.909 mm and forbidden
body penetration is zero.

Finally, an unchanged-stair extreme-entry probe distinguishes a true oblique
approach from a paired bent-stair co-warp.  Full-stair 50-degree traversals and
75-degree side-on pelvis turns were mechanically or collision rejected and
remain negative controls.  A more realistic profile enters a neighbouring
lane at 50 degrees and then continues straight up the original staircase.
Clip 473 admits the native left ascent and its exact right mirror: both have
0.337 m lateral range, 2.21 mm maximum stance-run drift, 4.78 mm maximum
stance target error, 4.822 mm maximum foot penetration, and zero forbidden-body
penetration.  These are retained only after side and overhead visual review.

The resulting balanced collection is
`terrain_maneuver_collection_v1/balanced_exotic200_v1`.  It contains exactly
200 clips from 66 physical terrain sources with a five-clip-per-source cap and
a seven-second minimum duration (median 10.4 s).  Traversal coverage is 65
ascents, 55 descents, and 80 curb/slope round trips.  Timing coverage is 58
bounces, 70 reversals, 70 stop/restarts, plus the symmetric pair of direct
50-degree fixed-stair entries.  Spatial coverage also explicitly retains hard
crab/facing, hard diagonal, lane-hold, step-change, counter-facing slalom, and
zigzag families instead of filling the quota with basic straight climbs.
`selection.routes.png` plots every realized and intended route by source bank,
and `sonic_bundle` packages all 200 exact motion/terrain pairs for tracker
fine-tuning and noisy collection.
