# G1 contact-space terrain oracle

Date: 2026-08-01

## Result

**Classification: `continuous-contact-warp-required`.**

The rigid contact-space oracle is deterministic and terrain-safe within its
configured tolerances, but it does not qualify as the omnidirectional terrain
matcher. The frozen v4 configuration completes 4 of 21 route contracts and 14
of 21 routes without transition exhaustion. It solves both diagonal descents,
diagonal-up-left, and the middle-right 90-degree stair turn. It does not solve
the lateral mount, cross-tread, side-exit, reversal, or mixed families.

The failure is not explained by a dense height grid or missing global lateral
motion alone. The indexed corpus contains hundreds of useful low-yaw lateral
actions, but the fixed root/joint trajectories cannot connect into those
actions while simultaneously satisfying boundary pose, foot, and terrain
constraints. A one-shot stance-foot root lock made the result worse. The next
representation must preserve the contact planner while continuously deforming
root/foot trajectories and solving stance-foot IK across transitions.

This remains a privileged-heightmap, native-50-Hz, kinematic result. Sonic,
tracking, depth, physics, and real-time latency are deliberately excluded.

## Frozen implementation

- Branch: `research/g1-torch-terrain-kinematics`
- Selected commit: `7696884` (`fix: anchor oracle search to commanded route`)
- Dataset identity: `979158922c1551bbbf655b21a03020b3976fb6af4d05ddda4ec3ed5790ddf498`
- Config identity: `a090d5a789fd62022768a9e119b843e9b442d6db908dfa3f7874fbe468c51410`
- Implementation identity: `g1-contact-oracle/route-anchored-continuity-v4`
- Actions: 9,004, consisting of 4,502 source actions plus exact sagittal mirrors
- Source rejections: 1,518 unstable terminal support and 164 inconsistent contact order
- Search: four landings, beam 32, foothold beam 50, transition shortlist 32
- Selected weights: path 100, facing 10, foothold 50, stance/entry-contact continuity 1,000

Mirroring was checked against MuJoCo FK on 64 random G1 poses. Maximum, mean,
and p99 ankle-position reflection error were all exactly 0.0 m.

The 21 routes were split deterministically over seven GPUs. The report-only
SHA over the seven ordered batch hashes is
`b037937652374050cb039557d43c8355afff65a5d5bf6fd78867b2f4b4780884`.
The authoritative batch hashes are:

| Batch | SHA-256 |
|---|---|
| 1 | `42bf6f12c80bbe70f4c97a042e2bdd0f7f2fcb51252dbf18c3a0840313e398e5` |
| 2 | `aee6d57bc19828de2a14d11ad4491ce87e62d1db54773de4f4d6dac9a522b77c` |
| 3 | `509ce6807e67b7faf1b30740ed37d3fc952187d5e52828ffc4c50844f722d093` |
| 4 | `f37dbbb38d955f2d055ade134aa41c2852859e2eca4988003aabe800a2cf0e7d` |
| 5 | `9821bc41d5997074f2fd79846f18b734685debdaa6d59fbb2a15fe8082e95828` |
| 6 | `7c304b3d031ba17d89210b4858e43817e43065828b4a054f481cbded7d7d7007` |
| 7 | `78bad074a9d859cf36c2716148b594fc6d69dd4a58957ab8c714715c5219ead8` |

## Full route matrix

`Exec` means the route reached its scripted frame count without exhausting all
valid transitions. `Pass` means the complete behavioral outcome contract also
passed. Slide is aggregate stance-foot travel for the route.

| Route | Exec | Pass | Slide (m) | Max stall | Elevated samples | Failure |
|---|---:|---:|---:|---:|---:|---|
| side-mount-left | no | no | 0.229 | 2 | 2 | incomplete; terrain not engaged; final surface |
| side-mount-right | no | no | 0.152 | 0 | 0 | incomplete; continue-up; terrain not engaged; final surface |
| cross-tread-left-to-right | yes | no | 0.356 | 2 | 2 | cross progress; terrain not engaged |
| cross-tread-right-to-left | yes | no | 0.445 | 12 | 0 | cross progress; terrain not engaged; stall |
| turn-45-lower-left | no | no | 0.052 | 0 | 51 | incomplete; final heading |
| turn-45-lower-right | yes | no | 0.049 | 0 | 248 | final heading and lateral drift |
| turn-90-middle-left | yes | no | 0.094 | 0 | 357 | final heading and lateral drift |
| turn-90-middle-right | yes | **yes** | 0.264 | 0 | 342 | — |
| turn-180-upper-left | yes | no | 0.215 | 0 | 478 | final lateral drift |
| turn-180-upper-right | yes | no | 0.215 | 0 | 478 | final lateral drift |
| diagonal-up-left | yes | **yes** | 0.045 | 0 | 297 | — |
| diagonal-up-right | no | no | 0.119 | 0 | 121 | incomplete |
| diagonal-down-left | yes | **yes** | 0.173 | 0 | 566 | — |
| diagonal-down-right | yes | **yes** | 0.146 | 2 | 587 | — |
| side-exit-lower-left | yes | no | 0.198 | 0 | 257 | final surface not flat |
| side-exit-lower-right | yes | no | 0.051 | 0 | 258 | final surface not flat |
| side-exit-upper-left | yes | no | 0.497 | 2 | 557 | final surface not flat |
| side-exit-upper-right | no | no | 0.165 | 16 | 535 | incomplete; final surface; stall |
| riser-stop-restart | no | no | 0.167 | 0 | 178 | incomplete |
| riser-reversal | yes | no | 0.550 | 2 | 449 | reverse-down progress |
| mixed-adversarial | no | no | 0.052 | 0 | 51 | incomplete; turn/reverse/exit progress; final surface |

Aggregate v4 values:

- behavioral passes: 4/21;
- exception-free routes: 14/21;
- stance slide: 4.233945 m;
- maximum measured penetration: 0.005171 m;
- maximum moving-command stall: 16 frames; and
- elevated-foot samples: 5,814.

Every emitted edge remained inside the configured validation limits. Across
the saved rollouts, maximum stance-height error was 0.047089 m, maximum landing
error was 0.029334 m, and minimum swing clearance was -0.028467 m. The limits
are 0.05 m, 0.03 m, and -0.03 m respectively. The nonzero penetration metric is
therefore permitted swing-toe clearance, not an unchecked fallback.

## Comparisons

| System | Passes | Exception-free | Slide (m) | Max penetration (m) | Max stall |
|---|---:|---:|---:|---:|---:|
| Retained no-latency layered baseline | 9/21 | 18/21 | 6.625 | 0.136 | 25 |
| Initial rigid oracle v1 | 5/21 | 21/21 | 7.039 | 0.000 | 8 |
| Route-anchored mirrored oracle v4 | 4/21 | 14/21 | 4.234 | 0.005 | 16 |

The v4 oracle reduces aggregate slide by 36% relative to the retained baseline
and removes the baseline's 13.6 cm penetration outlier, but it misses the
required 50% slide reduction and loses behavioral coverage. It is evidence,
not a replacement.

## Decisive ablations

### Planned foothold shortlist

Ranking the 32-action transition shortlist by the planned landing foot, XY,
and time converted `cross-tread-right-to-left` from zero elevated samples to
331 and converted `diagonal-up-right` from zero to 50. This proved that
foothold intent must participate before shortlist pruning.

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
penalty. V4 integrates one absolute target path from the route reset. This
produced the first pass on `turn-90-middle-right`: 342 elevated samples, zero
stalls, and 0.264 m slide. It also produced clean diagonal passes. This matches
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
