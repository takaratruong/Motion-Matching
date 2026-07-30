# G1 Native Torch Stair Kinematics Results

Date: 2026-07-29  
Branch: `research/g1-torch-terrain-kinematics`  
Qualified implementation commit: `f7d8630dae24af885bb5ff9ace50872060846470`

## Conclusion

**Not promising under the frozen acceptance rule.**

The dense privileged-height condition proves that terrain features can make the
native matcher select stair kinematics: it selected the recorded ascent at the
first step, reached the upper landing, achieved 108.3% of reference horizontal
progress and 96.0% of reference root-height gain, and reduced integrated
penetration by 91.8% versus flat. It nevertheless violated the required
`-0.03 m` minimum foot-clearance contract with a `-0.482 m` transient.

The failure occurs after the ascent succeeds. At 8.2–8.4 s, while stopped on the
landing, matching transitions from the ascent into ordinary descent clips. The
worst saved frame is frame 419 (time 8.38 s), selected source
`stair/0002/motion.npz:364`; its left foot is blended through the stair while
the terrain residual is extremely large. This is a source-selection/transition
failure, not an invalid surface alignment.

The latency hypothesis is more encouraging: dense matcher step p50/p95/p99 is
13.58/16.88/18.52 ms on the L40S, and exact-search p50/p95/p99 is
1.95/5.24/6.39 ms. The matcher therefore fits 50 Hz at p99 for this small
corpus, although its 25.93 ms maximum misses one deadline. These numbers exclude
depth inference, SONIC policy inference, transport, and physics.

## Frozen Inputs and Identities

- Dataset schema: `g1-torch-stair-slice/v1`
- Experiment schema: `g1-torch-stair-small-experiment/v1`
- Dataset manifest SHA-256:
  `7169520eefcf657231ad3df19121a985ede6edd801fb53d03562653ef9e5aefb`
- Motion inventory SHA-256:
  `06e358833a3b6836916aa5bde5964b49a650764bb12fe8c646d1b7e27bac0e6b`
- Flat Takara output: 34,863 frames at 50 Hz
- Four pinned GRAIL outputs: 499 frames each at 50 Hz
- Query scene: `stair/updown-0000/motion.npz`
- Recorded ascent: frames 0–266 (5.32 s)
- Reference progress: 2.012919 m
- Reference root-height gain: 0.631209 m
- First riser: 1.272165 m
- Command speed: 0.378368 m/s
- Terrain weight: 4.0 for legacy and dense
- Torch: 2.13.0+cu130
- GPU: NVIDIA L40S

GRAIL object `root_quat` was verified to be source wxyz, unlike robot
`root_rot`, which is source xyzw. The protected real-data oracle checks both
recorded ankle-roll bodies against the transformed USD surface. Its maximum
contact error was `1.5670626e-05 m`, below the frozen `1e-4 m` limit.

The old `g1_terrain_takara_slopes_stairs` four-height sidecar was rejected as an
alignment oracle because its rows for this source behaved as a single roughly
0.70 m curb from a different reconstruction surface.

## Condition Results

| Value | Flat | Legacy 4 | Dense 91 |
|---|---:|---:|---:|
| First stair selection (m) | 1.9375 | 0.2892 | 0.0000 |
| Maximum progress (m) | 1.9442 | 2.4142 | 2.1796 |
| Reference progress ratio | 96.6% | 119.9% | 108.3% |
| Maximum root-height gain (m) | 0.0153 | 0.4552 | 0.6058 |
| Reference height ratio | 2.4% | 72.1% | 96.0% |
| Minimum foot clearance (m) | -0.5965 | -0.3566 | -0.4820 |
| Penetration integral (m·s) | 3.8814 | 0.2252 | 0.3169 |
| Penetration ratio vs flat | 100.0% | 5.8% | 8.2% |
| Reached upper landing | No | No | **Yes** |

The stair-selection deadline is 1.072165 m
(`first_riser - 0.20 m`). Earlier selection passes.

### Dense frozen acceptance

| Criterion | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| Stair selected by pre-riser deadline | ≤ 1.0722 m | 0.0000 m | Yes |
| Horizontal reference progress | ≥ 90% | 108.3% | Yes |
| Root-height gain | ≥ 80% | 96.0% | Yes |
| Minimum foot clearance | ≥ -0.030 m | -0.482 m | **No** |
| Landing or ≤ 50% flat penetration | either | landing and 8.2% | Yes |

Overall: **4/5 criteria; failed minimum foot clearance.**

## Latency

All values are milliseconds. Step time is the normal transactional matcher
call. Search time is the exact-search portion on searched frames.

| Condition | DB build | Step p50 | Step p95 | Step p99 | Search p50 | Search p95 | Search p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 690.10 | 15.34 | 26.88 | 32.77 | 3.35 | 11.14 | 13.64 |
| Legacy 4 | 698.35 | 18.75 | 40.49 | 46.69 | 2.07 | 6.60 | 7.34 |
| Dense 91 | 636.46 | **13.58** | **16.88** | **18.52** | **1.95** | **5.24** | **6.39** |

Dense maximum step/search times were 25.93/7.41 ms. Database construction
includes strict motion loading, feature extraction, normalization, and device
transfer. The vectorized legacy terrain rows alone build in 0.23 s; the original
scalar-synchronizing implementation was rejected during qualification.

## Representative Dense Events

| Time (s) | Selected source | Frame | Transition | Motion cost | Terrain cost |
|---:|---|---:|:---:|---:|---:|
| 0.00 | `updown-0000` | 3 | Yes | 24.08 | 9.07 |
| 2.20 | `updown-0000` | 125 | No | 2.24 | 249.02 |
| 2.30 | `updown-0000` | 130 | No | 1.65 | 576.05 |
| 8.20 | `0001` | 402 | Yes | 50.61 | 5,543.64 |
| 8.30 | `0002` | 360 | Yes | 36.89 | 5,598.76 |
| 8.40 | `updown-0000` | 260 | Yes | 176.59 | 7,869.97 |

The first-riser crossing occurs around saved frame 114 at progress 1.2702 m.
At the worst-clearance frame 419, continuation has advanced `0002` to frame 364
and left-foot clearance is `-0.48198 m`.

## Reproducible Artifacts

Generated artifacts are intentionally ignored by Git:

- Results: `build/torch-stair-small-results/`
- Acceptance: `build/torch-stair-small-results/acceptance.json`
- Worst-frame PNG:
  `build/torch-stair-small-results/dense-worst-clearance.png`
- 9.0 s, 450-frame, 50 Hz H.264 recording:
  `build/torch-stair-small-results/dense.mp4`

The video was verified as H.264, 1320×960, 50 fps, 450 frames, and 9.0 s.

Deterministic rollout identities:

- Flat:
  `7fbebb8cdeb01a449bb6a1279aec5f52c85c16277ecea17363bf2d0f19129ccb`
- Legacy:
  `0932f9eff4da5ebd3d429b3214a8dfe6a98b9a322711aa270739fa354c80a20c`
- Dense:
  `3fdd87c469a185b856c9848338020fb109f591e20b32fecd8db3523a50dd55ca`

Authenticated rollout NPZ identities:

- Flat:
  `15f425c3a1a5ab3271dbc9144262f623244101cde8d42272f797f91559fd8e10`
- Legacy:
  `424997906aef9184fae05df0e1fbae750c591c486488bfb3b2d22ff70214401c`
- Dense:
  `0a9bb775f4488c02193ea6e4a929269509169da728c2d4bcb015c0477923f7b8`

## Verification

- Exact five-clip inventory and frame counts: pass
- Real recorded-contact alignment oracle: pass (`1.567e-05 m`)
- Full focused regression: 65 tests pass
- Flat/legacy/dense CUDA CLI: pass
- Saved artifact authentication: pass
- Headless worst-frame PNG: pass
- Full H.264 MP4 encode and ffprobe validation: pass

## Next Experiment

Keep the dense grid and weight-4 result frozen. Add a tested source-phase or
direction constraint that prevents a stopped character on the upper landing
from entering descent-only clips, and evaluate transition clearance before any
depth encoder or SONIC tracking run. The current small-corpus matcher latency
does not justify action chunking by itself; depth plus controller latency remains
unmeasured.
