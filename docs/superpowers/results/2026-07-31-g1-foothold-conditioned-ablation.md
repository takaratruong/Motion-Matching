# G1 foothold-conditioned arm comparison

Date: 2026-07-31

## Scope and method

This is a privileged-heightmap, 50 Hz, kinematic comparison. It excludes
Sonic, tracking, physics, depth inference, and real-time latency. Every arm
uses the same 760-clip corpus, centered GRAIL staircase, independent route
resets, 21-route command matrix, MuJoCo FK, and 7 cm contact-height tolerance.

The comparison now measures route outcomes and visible motion quality. In
addition to penetration, slide, stalls, and progress, it reports source
discontinuities, cross-clip switches, emitted full-joint jerk, and joint
velocity jumps at transitions. This catches the failure where an arm reaches
the requested terrain but visibly splices incompatible clips.

## Implementation defect found

The original `two-contact` and `hybrid` arms were byte-identical; both added
the foothold residual to the ordinary motion cost. That violated the approved
design, in which hybrid first applies the two-contact feasibility gate and then
uses ordinary pose/velocity/trajectory cost among feasible clips.

The corrected arms are:

- **first-contact:** hard first-contact gate, ordinary MM ranking;
- **two-contact:** hard two-contact gate plus foothold-descriptor ranking;
- **hybrid:** hard two-contact gate, ordinary MM ranking;
- **continuous:** no hard contact gate, continuous descriptor residual;
- **layered:** hard two-contact height/order gate, descriptor ranking, deep
  safe fallback; and
- **layered-hybrid:** layered gate/fallback with ordinary MM ranking.

The old two-contact and hybrid shared matrix SHA was
`e53c11126068ea1539a46e88624d95179dcc6ab2260dbc0d807982c4292f8a6b`.
The corrected hybrid SHA is
`e3bedae3abface72fc2900ec1050b31f0fa5a815f0111b683f5b6af6d7d8d1f2`.

## Complete primary-arm result

`Switches` is the total cross-clip count over all 21 routes. `Jerk p95` is the
aggregate emitted full-joint jerk in rad/s³. `Velocity jump` is the worst
route's p95 emitted joint-velocity change at a source discontinuity in rad/s.
A failed-closed candidate-exhaustion exception is reported separately from a
terrain safety violation.

| Arm | Routes | Complete classes | Safety violations | Exceptions | Switches | Jerk p95 | Velocity jump | Slide (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla dense MM | 5/21 | 0 | 2 | 17 | 1,189 | 242,020 | 91.11 | 9.399 |
| Contact segments only | 12/21 | 2 | 0 | 3 | 206 | 27,996 | 55.23 | 7.444 |
| Continuous feature control | 10/21 | 1 | 0 | 5 | 140 | 24,627 | 54.71 | 8.472 |
| First-contact | 9/21 | 1 | 0 | 6 | 120 | 23,836 | 60.75 | 9.499 |
| Two-contact | 6/21 | 1 | 0 | 13 | 94 | 17,462 | 53.50 | 5.706 |
| Corrected hybrid | 10/21 | 1 | 0 | 16 | 69 | 17,481 | 53.50 | 4.532 |
| Layered | 14/21 | 3 | 0 | 0 | 160 | 29,103 | 59.84 | 9.389 |
| Layered-hybrid, continuity 0.5 | **15/21** | **4** | **0** | **0** | **157** | **26,690** | **54.71** | 9.633 |

The corrected strict hybrid is the cleanest retrieval arm—69 switches and
17,481 jerk—but its two-contact gate exhausts coverage on 16 routes. Layered
is the first arm with no exceptions, but its descriptor-first ranking explains
the user's observation that many selected clips do not visually match. The
merged arm retains layered coverage while allowing full-pose continuity and
ordinary motion compatibility to influence the choice.

## Full-pose continuity comparison

The merged arm was evaluated at symmetric position/velocity weights 0.1,
0.25, 0.5, 1.0, and 2.0, plus asymmetric 0.5/2.0 and 2.0/0.5 controls.

| Position / velocity | Routes | Exceptions | Switches | Jerk p95 | Stall fraction | Slide (m) |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 / 0.1 | 15 | 0 | 164 | 33,954 | 1.213% | 9.590 |
| 0.25 / 0.25 | 15 | 0 | 166 | 33,954 | 1.072% | 9.475 |
| **0.5 / 0.5** | **15** | **0** | **157** | **26,690** | 1.142% | 9.633 |
| 1.0 / 1.0 | 12 | 2 | 145 | 24,112 | 0.284% | 8.822 |
| 2.0 / 2.0 | 14 | 0 | 157 | 23,505 | 0.663% | 9.982 |
| 0.5 / 2.0 | 14 | 0 | 161 | 28,003 | 0.494% | 9.585 |
| 2.0 / 0.5 | 14 | 0 | 158 | 30,020 | 0.691% | 9.457 |

The original strict lexicographic score selects 0.25 because its stall fraction
is 0.070 percentage points lower than 0.5 before pose coherence is considered.
For the explicit visual clip-mismatch feedback, 0.5 is the retained viewer
candidate: it keeps all 15 routes, removes nine cross-clip switches relative
to 0.25, and lowers aggregate jerk by 21%. Weights 1.0 and 2.0 buy additional
smoothness by losing behavior, and both asymmetric hypotheses lose a route
without improving jerk over 0.5.

The retained matrix SHA is
`af8a500bc799454cd806bc317fc302ef7ff77f899722bad8ac4351c8ac7ada6f`.
It passes both cross-tread routes, all diagonals, both side mounts, reversal,
stop/restart, and five of six turns. It still fails all four side exits, the
mixed route's final-flat requirement, and the middle-left 90-degree final
heading.

## Interpretation against prior work

The result supports a planner/retriever split rather than vanilla height
features alone. Short-horizon foothold planning is consistent with terrain
adaptive systems that plan end-effector trajectories at footstep boundaries,
while the retrieval layer preserves captured whole-body style. See
[Terrain-Adaptive Bipedal Locomotion Control](https://grail.cs.washington.edu/projects/loco/).

[PFNN](https://www.research.ed.ac.uk/en/publications/phase-functioned-neural-networks-for-character-control/)
and [Learned Motion Matching](https://static-wordpress.ubisoft.com/montreal.ubisoft.com/wp-content/uploads/2020/07/09154101/Learned_Motion_Matching.pdf)
remain reasonable later distillation models. They do not repair the present
oracle's missing lateral drop/exit actions: training on these traces would
learn the same coverage hole. The four side-exit failures therefore require
new compatible lateral-down motion or a separately qualified foothold/IK
bridge, not another continuous height-feature weight.

## Retained configuration

Use `sonic/configs/experiments/torch_grail_layered_hybrid.json` with
`--contact-segments --foothold-arm layered-hybrid`. This checkpoint is better
than the prior viewer but is not a completed omnidirectional terrain matcher.
