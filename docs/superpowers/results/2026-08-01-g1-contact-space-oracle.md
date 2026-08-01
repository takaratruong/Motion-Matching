# G1 contact-space terrain oracle

Date: 2026-08-01

## Result

**Classification: `action-representation-or-corpus-insufficient`.**

The corrected rigid contact-space oracle is deterministic and terrain-safe
within its configured tolerances, but it does not qualify as the
omnidirectional terrain matcher. The frozen v7 configuration completes 5 of 21
route contracts and 19 of 21 routes without transition exhaustion. It solves
both diagonal ascents, both diagonal descents, and the middle-right 90-degree
stair turn. It does not solve the lateral cross-tread, side-exit, reversal, or
mixed behavioral contracts; `side-mount-left` and `turn-45-lower-left` have no
feasible one-contact continuation after the exact-entry set is exhausted.

The failure is not explained by a dense height grid or missing global lateral
motion alone. The indexed corpus contains hundreds of useful low-yaw lateral
actions, but broad-bound search exhausts all feasible one-contact successors
on two routes. The fixed root/joint trajectories cannot connect into those
actions while simultaneously satisfying boundary pose, foot, and terrain
constraints. This is therefore evidence against the current discrete action
representation (or its effective corpus connectivity), not evidence that the
search formulation has qualified. Continuous contact warping remains the most
promising next representation, but is a proposed fix rather than this result's
classification.

This remains a privileged-heightmap, native-50-Hz, kinematic result. Sonic,
tracking, depth, physics, and real-time latency are deliberately excluded.

## Frozen implementation

- Branch: `research/g1-torch-terrain-kinematics`
- Selected commit: `e267b10` (`fix: canonicalize oracle failure evidence`)
- Dataset identity: `979158922c1551bbbf655b21a03020b3976fb6af4d05ddda4ec3ed5790ddf498`
- Config identity: `375bd77c1eb12568b26e86b0caca80ed02e72009af7359d387bb86d5cad6579c`
- Implementation identity: `g1-contact-oracle/all-flight-clearance-v7`
- Actions: 9,004, consisting of 4,502 source actions plus exact sagittal mirrors
- Source rejections: 1,518 unstable terminal support and 164 inconsistent contact order
- Search: four landings, beam 32, foothold beam 50, transition shortlist 32
- Selected weights: path 100, facing 10, foothold 50, stance/entry-contact continuity 1,000

Mirroring was checked against MuJoCo FK on 64 random G1 poses. Maximum, mean,
and p99 ankle-position reflection error were all exactly 0.0 m. In addition,
all 7,856 emitted v7 frames were replayed through independent MuJoCo FK using
their saved joints, root positions, and full root quaternions. Maximum ankle
error against the planned foot arrays was 1.9834e-7 m; qpos root and quaternion
errors were exactly zero. This corrects the earlier yaw-only qpos evidence bug,
which invalidated the original v4 visualization and safety interpretation.

The 21 routes were split deterministically over seven GPUs. Batches 1 and 2
were regenerated after canonicalizing structured-failure serialization; every
non-timing rollout array matched the preceding v7 artifact byte-for-byte, and
all successful-route hashes remained unchanged. The report-only SHA over the
concatenated seven ordered batch hashes is
`626626dba229e04d706b86402c30484296ea90bb9430571ba4c493405221446b`.
The authoritative batch hashes are:

| Batch | Artifact root | SHA-256 |
|---|---|---|
| 1 | `build/contact-oracle/full-v7q-b1` | `66bb64f5ae1962329ceed86b734c4c18e0344b9ff7cce96f033566c37d51a863` |
| 2 | `build/contact-oracle/full-v7q-b2` | `fe464d2662122cfebd333c73c6a777888c22aecb48b5e00928e3cbe76892226c` |
| 3 | `build/contact-oracle/full-v7-b3` | `b0daf0f9031c4a6a05d3cee08cd2ac33bbdfededec429738d318895898f77f45` |
| 4 | `build/contact-oracle/full-v7-b4` | `d3424038c94dde76289c7486ae9a59bdb3eab48882b9f7fa28d7c791f8d86f2c` |
| 5 | `build/contact-oracle/full-v7-b5` | `42790eb80daf632faf595947dc6ee138307c300d7f56519da9e5b9273564ee7f` |
| 6 | `build/contact-oracle/full-v7-b6` | `f171c1c6e7463f988c5b1ee390f51dd3d9f4ede7fc928e875085b521c5730233` |
| 7 | `build/contact-oracle/full-v7-b7` | `be9463c4fedd98c30ac42f3d9734bd35f7c18014186e0c56167bab620997c775` |

## Full route matrix

`Exec` means the route reached its scripted frame count without exhausting all
valid transitions. `Pass` means the complete behavioral outcome contract also
passed. Slide is aggregate stance-foot travel for the route.

| Route | Exec | Pass | Slide (m) | Max stall | Elevated samples | Failure |
|---|---:|---:|---:|---:|---:|---|
| side-mount-left | no | no | 0.229 | 2 | 2 | incomplete; terrain not engaged; final surface |
| side-mount-right | yes | no | 0.167 | 16 | 0 | continue-up; terrain not engaged; final surface; stall |
| cross-tread-left-to-right | yes | no | 0.356 | 2 | 2 | cross progress; terrain not engaged |
| cross-tread-right-to-left | yes | no | 0.445 | 12 | 0 | cross progress; terrain not engaged; stall |
| turn-45-lower-left | no | no | 0.142 | 6 | 205 | incomplete |
| turn-45-lower-right | yes | no | 0.049 | 0 | 248 | final heading and lateral drift |
| turn-90-middle-left | yes | no | 0.094 | 0 | 357 | final heading and lateral drift |
| turn-90-middle-right | yes | **yes** | 0.264 | 0 | 342 | — |
| turn-180-upper-left | yes | no | 0.215 | 0 | 478 | final lateral drift |
| turn-180-upper-right | yes | no | 0.215 | 0 | 478 | final lateral drift |
| diagonal-up-left | yes | **yes** | 0.045 | 0 | 297 | — |
| diagonal-up-right | yes | **yes** | 0.074 | 0 | 259 | — |
| diagonal-down-left | yes | **yes** | 0.173 | 0 | 566 | — |
| diagonal-down-right | yes | **yes** | 0.146 | 2 | 587 | — |
| side-exit-lower-left | yes | no | 0.198 | 0 | 257 | final surface not flat |
| side-exit-lower-right | yes | no | 0.051 | 0 | 258 | final surface not flat |
| side-exit-upper-left | yes | no | 0.497 | 2 | 557 | final surface not flat |
| side-exit-upper-right | yes | no | 0.168 | 16 | 557 | final surface; stall |
| riser-stop-restart | yes | no | 0.113 | 26 | 0 | terrain not engaged; final surface; stall |
| riser-reversal | yes | no | 0.550 | 2 | 449 | reverse-down progress |
| mixed-adversarial | yes | no | 0.226 | 6 | 667 | turn/reverse progress; final surface; final heading |

Aggregate v7 values:

- behavioral passes: 5/21;
- exception-free routes: 19/21;
- stance slide: 4.417654 m;
- maximum measured penetration: 0.005171 m;
- maximum moving-command stall: 26 frames; and
- elevated-foot samples: 6,566.

Every emitted edge remained inside the configured validation limits. Across
the saved rollouts, maximum stance-height error was 0.047089 m, maximum landing
error was 0.029544 m, and minimum unsupported-foot clearance was -0.028467 m.
The limits are 0.05 m, 0.03 m, and -0.03 m respectively. The reported
penetration metric measures ankle-origin penetration, not sole clearance. The maximum
direct sole penetration was 0.040171 m on a supported foot, within the broad
0.05 m stance-height tolerance. All 285 planning events serialize per-edge and
cumulative cost components, planned
and emitted landing coordinates, beam/rejection counts, and joint boundary
errors. Maximum entry joint-position and velocity errors were 2.4841 rad and
7.6344 rad/s. Every completed first edge had exactly zero planned-versus-emitted
landing error. All 30 zero-speed frames in `riser-stop-restart` were explicit
holds with zero root displacement and no selected action.

The route hashes authenticate these planning events as well as the frame arrays
and canonical structured failures; `plan_time_ns` is the only deliberately
unhashed field. An independent artifact-only recomputation reproduced all 21
route hashes and all seven matrix hashes. Aggregated diagnostics from the
authoritative v7 planning events were:

| Gate | Rejections |
|---|---:|
| support order | 58,354,170 |
| entry-foot error | 55,811,467 |
| transition shortlist | 6,932,898 |
| joint velocity | 1,186,410 |
| landing height | 227,989 |
| stance height | 47,934 |
| landing edge margin | 24,374 |
| joint position | 18,381 |
| no successor | 15,756 |
| swing penetration | 11,607 |
| height deformation | 11,112 |

## Visual review

Eight-frame offscreen MuJoCo contact sheets were rendered from the exact saved
qpos for the six frozen review routes under
`build/contact-oracle/visual-review-v7`. The review agrees with the behavioral
metrics rather than revealing a hidden successful traversal:

- `cross-tread-left-to-right` approaches and moves alongside/behind the stair
  silhouette but never establishes the requested lateral tread crossing. It
  records only two elevated-foot samples and 0.356 m stance slide.
- `diagonal-down-left` is the clean positive case: it reaches the elevated
  surface, traverses diagonally, and descends without a prolonged stall. Its
  0.173 m stance slide is still visible as imperfect contact preservation.
- `side-exit-upper-left` climbs and remains on the upper platform instead of
  stepping off its side. This matches the failed final-flat-surface contract
  and the route's high 0.497 m stance slide.
- `turn-90-middle-left` reaches the stair and changes pose while elevated, but
  ends with the wrong heading and lateral position rather than a controlled
  in-place 90-degree turn.
- `riser-reversal` climbs and begins repositioning, but does not complete the
  required reverse descent. It has the largest reviewed stance slide (0.550 m).
- `mixed-adversarial` now executes all 500 frames and reaches the stair, but it
  misses reverse-down progress, the final flat surface, and final heading.

The sparse visual review also exposes abrupt pose changes at source boundaries.
Across these routes the largest boundary joint-vector jump ranges from 0.965 to
1.575 rad and the largest boundary root jump from 1.49 to 1.89 cm. The feet are
geometrically consistent with each shown qpos after the full-orientation fix;
the remaining sliding, missed task geometry, and discontinuities are properties
of rigid clip concatenation.

## Comparisons

| System | Passes | Exception-free | Slide (m) | Max penetration (m) | Max stall |
|---|---:|---:|---:|---:|---:|
| Retained no-latency layered baseline | 9/21 | 18/21 | 6.625 | 0.136 | 25 |
| Initial rigid oracle v1 | 5/21 | 21/21 | 7.039 | 0.000 | 8 |
| Exhaustive-failure all-feet oracle v7 | 5/21 | 19/21 | 4.418 | 0.005 | 26 |

The v7 oracle reduces aggregate slide by 33% relative to the retained baseline
and removes the baseline's 13.6 cm penetration outlier, but it misses the
required 50% slide reduction and loses behavioral coverage. It is evidence,
not a replacement.

## Decisive ablations

### Planned foothold shortlist

Ranking the 32-action transition shortlist by the planned landing foot, XY,
and time converted `cross-tread-right-to-left` from zero elevated samples to
331 and converted `diagonal-up-right` from zero to 50. This proved that
foothold intent must participate before shortlist pruning.

### Exhaustive one-step failure audit

The top-32 shortlist was not sufficient evidence of corpus failure. Exhausting
all exact-entry candidates whenever the final horizon-one shortlist was empty
recovered `diagonal-up-right` as a behavioral pass and removed transition
exhaustion from `side-exit-upper-right`, `riser-stop-restart`, and
`mixed-adversarial`. Exception-free coverage rose from 14 to 19 routes. The two
remaining failure states have no feasible one-contact action under the broad
entry and terrain bounds.

### Exact sagittal mirrors

The source inventory had one-to-two-order-of-magnitude handedness gaps in some
up/down and turning bins. Exact mirroring doubled the inventory to 9,004 and
fixed `diagonal-up-left` in the targeted matrix: 61 elevated samples, 0.072 m
slide, and no stalls. It did not solve side exits.

### One-shot support-foot root lock

Rigidly translating each selected phase to its current support foot was
rejected. On `side-exit-lower-left`, slide increased from 0.085 to 0.482 m and
maximum root boundary jumps increased from 1.24 to 6.92 cm. On the 90-degree
turn, pivot progress fell from 0.676 to 0.434 and stalls increased from 2 to 8.
Locking only the entry sample moves the error into the rest of the phase and
the pelvis trajectory.

### Strict entry-foot gates

Reducing the 8 cm entry-foot ceiling to 3 cm or 2 cm left no feasible action at
the reset state. Strict filtering cannot replace transition synthesis.

### Entry-contact continuity cost

Adding 3D entry-foot mismatch before shortlist pruning and in the beam cost
reduced boundary jumps, but could trade path progress for continuity. A weight
of 1,000 was the useful compromise; 5,000 overconstrained motion. This term is
retained, but it cannot create a missing bridge.

### Route-anchored future trajectory

The original cost compared only each action's local displacement with the
command over that action. It reset at every landing, so lag accumulated without
penalty. V7 retains the v4 absolute target path integrated from the route
reset. This produced the first pass on `turn-90-middle-right`: 342 elevated
samples, zero stalls, and 0.264 m slide. It also produced clean diagonal passes. This matches
the principle of tracking a future target to prevent lag in Daniel Holden's
[New Movement Model](https://theorangeduck.com/page/new-movement-model).

### Lateral coverage and facing

The mirrored action inventory contains:

| Minimum low-yaw lateral speed | Actions | Up >0.1 m | Down <-0.1 m | Flat |
|---:|---:|---:|---:|---:|
| 0.10 m/s | 1,748 | 484 | 470 | 756 |
| 0.20 m/s | 328 | 68 | 48 | 200 |
| 0.30 m/s | 94 | 18 | 6 | 68 |
| 0.38 m/s | 54 | 6 | 4 | 44 |

Thus global lateral data exists. The failed cross route instead entered
turning clips with 60–100 degree yaw changes, then exhausted the low-yaw rigid
transition chain. Raising facing weight from 10 to 100 reduced one exit's
slide but broke the previously passing 90-degree route and made cross-tread
exhaust earlier. The useful actions are not sufficiently connected under the
rigid boundary representation.

## Interpretation against prior work

[Terrain-Adaptive Bipedal Locomotion Control](https://grail.cs.washington.edu/projects/loco/)
separates per-footstep end-effector planning from a per-timestep solver that
honors ground contacts, and demonstrates sharp turns plus forward, backward,
and sideways transitions on uneven terrain. Our planner now makes contact
decisions, but rigid playback has no equivalent per-timestep synthesis layer.

[PFNN](https://www.pure.ed.ac.uk/ws/portalfiles/portal/35467734/phasefunction.pdf)
fits motion to terrain using contact terms, mirrors captures, and generates the
whole-body pose conditioned on future trajectory and terrain. It does not
concatenate immutable one-landing clips at their raw boundaries.

[Multi-Contact Locomotion Using a Contact Graph with Feasibility Predictors](https://doi.org/10.1145/3072959.2983619)
similarly plans feasible contact points and then generates/deforms the
whole-body motion. These systems support the measured conclusion: contact
search is necessary, but contact-aware synthesis or deformation is the missing
layer.

## Next experiment

Retain the deterministic four-contact planner and replace rigid execution with
a separately testable contact warp:

1. inertialize root and joint boundary offsets over a bounded transition;
2. lock the stance sole throughout its support interval, not only at entry;
3. solve lower-body IK to the planned swing landing while preserving the
   source pelvis/facing trajectory;
4. re-run the identical 21-route matrix and require at least 18 passes, all
   cross/exit/reversal cases, no hard contact violation, at most five stall
   frames, and at least 50% slide reduction; and
5. only after this kinematic oracle qualifies, distill it into a PFNN-style or
   DAgger student and then introduce depth/proprioception and Sonic tracking.

Do not spend more time on dense-grid resolution, scalar path/facing weights,
or Sonic integration before the contact warp passes this oracle matrix.
