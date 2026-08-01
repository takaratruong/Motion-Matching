# G1 foothold-conditioned motion matching: experiment results

Date: 2026-07-31

## Scope

This qualification covers privileged-heightmap, 50 Hz, kinematic G1 motion
matching only. It does not include Sonic, tracking, dynamics, or a depth model.
The corpus is `build/torch-grail-terrain-feedback-v2` (760 clips, manifest SHA
`979158922c1551bbbf655b21a03020b3976fb6af4d05ddda4ec3ed5790ddf498`). The stress scene is one recorded GRAIL stair height grid, and all
reported foot positions come from authoritative MuJoCo FK.

## Selected algorithm

The selected matcher uses:

1. a two-contact action inventory built from source support masks;
2. a short foothold plan sampled directly on the query height grid;
3. hard landing-foot order and landing-height gates;
4. soft landing XY, timing, and ordinary motion-feature costs;
5. persistent source-travel direction filtering for real lateral commands;
6. a deep, offline-only fallback when the primary layered candidate set is
   empty;
7. validated action chunks, latched to the operator command that selected them;
8. whole-chunk direction checks for locomotion, with slow pivots classified by
   the planned 15--45-frame speed rather than the first shaped frame.

The matcher never fabricates foot heights or accepts an unvalidated extended
chunk. Full emitted windows are checked against the query terrain using MuJoCo
FK before commitment.

## Rejected ablations

| Ablation | Useful result | Rejection reason |
|---|---|---|
| Unconditional long action latch | 180-degree turn reached 0.19 m slide | Could preserve an old-direction action through reversal or strafe commands |
| Strict two-contact gate | Strong semantic constraint | Candidate dead ends and freezing under sparse command coverage |
| Continuous-control arm | Reversal completed | Failed side/cross/exit cases and accumulated 0.74 m slide |
| Persistent direction gate for every moving command | Cross slide improved from 0.68 to 0.60 m | Reverse slide regressed from 0.39 to 0.68 m and penetration increased |
| Direction-relaxed final rescue | Let cross-right finish without an exception | Required progress collapsed to 0.03, slide rose to 0.86 m, penetration remained 0.043 m, and side-mount-right still exhausted candidates |
| 0.08 m landing-height tolerance | Stayed below half a riser | Cross-right was metric-identical and side-mount-right exceeded four minutes of enlarged search without completing |
| Hard lateral speed floor | Cross progress rose to 0.46 | Cross slide doubled to 1.25 m; exit jerk rose to 953 m/s^3 and still missed flat ground |
| Soft lateral speed cost | Preserved candidate coverage | Produced byte-identical selections and metrics at a safe weight |
| Extra inertialization / output smoothing | Reduced some local discontinuities | Regressed command response and aggregate route quality |
| Jerk reranking | Reduced one isolated jerk measurement | Regressed progress, slide, or penetration in the route matrix |

## Representative final metrics

These are deterministic, independently reset routes. Progress is the achieved
fraction of commanded displacement for each required segment. Slide is summed
stance-foot travel, so it grows with route duration and is used comparatively.

| Route | Outcome | Required progress | Slide (m) | Max penetration (m) | Jerk p95 (m/s^3) | Transitions | Longest stall (frames) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Side mount left | pass | 0.35 / 0.47 | 0.622 | 0.000 | 682 | 11 | 0 |
| Cross tread left-to-right | pass | 0.23 | 0.601 | 0.000 | 622 | 11 | 6 |
| Diagonal down left | pass | 0.47 | 0.433 | 0.000 | 365 | 9 | 2 |
| Riser reversal | pass | 0.13 | 0.394 | 0.0115 | 587 | 6 | 3 |
| Stop/restart | pass | 0.66 | 0.145 | 0.000 | 228 | 4 | 0 |
| Turn 180 degrees upper left | pass | 0.81 | 0.477 | 0.000 | 341 | 6 | 0 |
| Side exit upper left | fail: final surface | 0.23 | 0.468 | 0.0028 | 490 | 10 | 8 |
| Mixed adversarial | fail: final surface | 0.82 / 0.58 / 0.10 / 0.27 | 0.439 | 0.000 | 507 | 12 | 12 |

The two failed fixed-duration routes do respond to every command; they fail
because the sparse true-lateral clips move too slowly to carry both feet beyond
the physical stair edge before the command segment ends. In a separate
700-frame sustained exit, the required outcome did complete: both feet reached
flat ground, lateral progress was 0.34 of the requested speed, and the longest
stall was 16 frames. The diagnostic process then failed seven frames into its
final forward segment because it left the finite authoritative height-grid
domain. This distinguishes slow/limited lateral coverage from a matcher freeze,
but its 2.25 m accumulated slide and 35 transitions are not production quality.

## Complete 21-route matrix

The exact committed matcher completed 13 of 21 routes. The matrix SHA is
`0f8993cd3980d88fdb3d637919e177755ccf09a342b77c98dc61b49e5870fad3`.
All routes were reset independently; no route result was reused from a targeted
ablation.

Passes:

- cross-tread left-to-right;
- diagonal up and down in both directions;
- riser reversal and stop/restart;
- upper-right side exit;
- left side mount;
- 45-degree lower-right, both 180-degree upper turns, and 90-degree
  middle-left.

Failures:

| Route | Failure |
|---|---|
| Cross-tread right-to-left | Safe-candidate exhaustion at frame 308; terrain was not engaged |
| Side-mount right | Safe-candidate exhaustion at frame 353; final surface was not elevated |
| Side-exit lower left/right | Fixed command duration ended before both feet reached flat ground |
| Side-exit upper left | Fixed command duration ended before both feet reached flat ground |
| Mixed adversarial | All required segment progress thresholds passed, but the final surface remained elevated |
| Turn 45 degrees lower left | Final heading exceeded the 0.35 rad tolerance |
| Turn 90 degrees middle right | Final heading exceeded the 0.35 rad tolerance |

The directional asymmetry and two candidate-exhaustion failures mean this is a
feedback-ready research checkpoint, not a completed omnidirectional terrain
matcher.

The recorded query stair is also not centered on matcher y=0: its sampled
cross-section is approximately -0.24 to +0.62 m. The route generator mirrors
commands around y=0, so these original left/right routes are not geometric
mirrors of the actual obstacle. The follow-up below corrects that evaluator
bias before interpreting left/right deltas.

## Centered-evaluator follow-up

The evaluator now measures the widest elevated lateral interval at 70% of the
reference ascent and resets the character on that interval's midline. For this
scene, the measured matcher-world reset is approximately
`(-0.0533, +0.1667)` m. The reset is part of the deterministic configuration
identity, and the same placement is used by Backspace in the interactive
viewer.

The centered 21-route matrix completed 14 routes, one more than the original
matrix. Its SHA is
`875e90850d4af7d890a111591621af91ba18dec3dd5096e35ec23489ff4741e6`.
Most importantly, both former safe-candidate-exhaustion routes now complete:

| Route | Original | Centered | Centered progress | Centered slide (m) |
|---|---:|---:|---:|---:|
| Cross-tread right-to-left | fail: candidate exhaustion | pass | 0.56 | 0.786 |
| Side-mount right | fail: candidate exhaustion | pass | 0.29 / 0.46 | 0.632 |
| Cross-tread left-to-right | pass | pass | 0.18 | 0.761 |
| Side-mount left | pass | fail: continue-up progress | 0.41 / 0.08 | 0.524 |

The centered pass set contains both cross-tread routes, all four diagonals,
both 90-degree turns, both 180-degree turns, the lower-left 45-degree turn,
the right side mount, reversal, and stop/restart. The seven remaining failures
are all four fixed-duration side exits, the left side mount, the lower-right
45-degree turn, and the mixed route's final-surface requirement. No centered
route raised an exception or exhausted the safe candidate set.

Centering therefore removes a real evaluator defect and exposes a more useful
algorithm result: bidirectional cross-tread traversal is available, but true
lateral travel remains too slow and inconsistent to clear either stair edge
reliably. It also increases slide on several routes (for example, centered
cross-tread slide is 0.761/0.786 m), so 14/21 is still a research checkpoint,
not a production-quality omnidirectional matcher.

## Corpus coverage finding

Contact-segment travel relative to source facing is strongly imbalanced:

| Travel class | Segments | Median speed (m/s) | 75th percentile (m/s) |
|---|---:|---:|---:|
| Forward | 5,676 | 0.339 | 0.466 |
| Backward | 241 | 0.124 | 0.185 |
| Lateral | 267 | 0.208 | 0.323 |

This explains why hard speed filtering selects discontinuous clips and why a
fixed-heading lateral exit is much slower than forward ascent. More true
lateral/backward terrain data, or a later stride-warping/IK layer, is the clean
way to close this gap. It is not evidence for weakening terrain contact gates.

## Interactive viewer

Run the exact qualified configuration with:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=sonic/python \
sonic/.torch-mm-venv/bin/python -B -m mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_contact_segment.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda --contact-segments --foothold-arm layered \
  --foothold-height-tolerance-m 0.07
```

Controls are WASD, Space to stop, Backspace to reset, and X to close.
