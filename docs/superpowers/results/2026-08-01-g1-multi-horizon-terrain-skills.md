# G1 Multi-Horizon Terrain Skill Qualification

## Result

**Classification: continuation gate passed; ready for viewer/full-matrix feedback.**

The dimension-normalized multi-horizon matcher executed all six frozen routes
without exception and passed every behavior gate on five routes. The sole
remaining route failure was `side-exit-upper-left`, which completed its lateral
exit but ended on an elevated surface instead of flat terrain.

This is a substantial improvement over whole-remaining-skill matching, which
passed 0/6 routes and produced stalls as long as 158 frames. The qualified
matcher has a longest moving-command stall of 2--3 frames on every route.

It is not yet a claim of visually final motion. Accumulated stance slide remains
high on cross-tread, diagonal descent, and the 90-degree turn. The next feedback
round should therefore evaluate contact locking/visual quality as well as the
remaining side-exit terminal-surface error.

## Qualified algorithm

- Every owned terrain-skill entry has 25/50/100-frame outcome records ending at
  the first stable double-support state no more than 25 frames late.
- Records contain local root displacement, accumulated yaw, root/surface height
  change, duration, and time-local stall risk.
- The query simulates bounded velocity and yaw for 100 steps and samples desired
  terrain height along the commanded root path.
- Wrong turn sign, zero moving progress, wrong vertical sign, non-finite rows,
  and the current source neighborhood are hard-gated.
- Remaining candidates receive one GPU cost combining the unchanged 27-value
  MM entry feature and outcome terms, followed by full supported-foot terrain
  trace validation.
- Playback commits consecutive source frames only through the selected stable
  endpoint. Command changes latch until the next double-support boundary.
- Failure is structured; the adapter never silently substitutes a static hold.

The qualified entry weight is `1/27`. The entry term is a sum over 27 normalized
features, so this converts it to a per-dimension average and makes its natural
scale comparable to the low-dimensional outcome groups.

## Corpus and preprocessing

- Dataset: `build/torch-grail-terrain-feedback-v2`
- Clips: 760
- Search rows: 379,404
- Coherent terrain skills: 676
- Owned entry rows: 48,446
- Unique finite horizon records: 48,327
- Every recorded endpoint: exact double support
- Horizon preprocessing on `cuda:0`: 3.357 s
- Representative 48,327-record GPU rank: 49.665 ms

The initial bulk builder exposed and fixed a redundant GPU synchronization:
support masks were copied once per entry. The qualified builder copies stable
support indices once per skill and uses binary search per target horizon.

## Frozen six-route result

Artifacts: `build/multi-horizon-terrain-skills/qualified-v2-final`

| Route | Pass | Failure | Progress | Longest stall | Support p95 | Chunks | Transitions | Stance slide |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cross-tread-left-to-right | yes | -- | 1.510 m | 2 | 0.0198 m | 14 | 13 | 0.508 m |
| turn-90-middle-left | yes | -- | 1.382 m | 2 | 0.0198 m | 8 | 7 | 0.627 m |
| diagonal-down-left | yes | -- | 1.028 m | 3 | 0.0197 m | 12 | 11 | 0.948 m |
| side-exit-upper-left | no | final surface not flat | 1.745 m | 2 | 0.0198 m | 13 | 12 | 0.320 m |
| riser-reversal | yes | -- | 0.373 m | 2 | 0.0198 m | 8 | 7 | 0.453 m |
| mixed-adversarial | yes | -- | 0.595 m | 2 | 0.0199 m | 12 | 11 | 0.027 m |

The 90-degree route retained and improved the required sub-gates: ascent ratio
`0.730`, pivot ratio `0.471`, and final heading error `0.0366 rad`. Mixed
adversarial passed all four local progress segments. Side exit achieved its
lateral progress ratio (`0.258`) but did not descend to the final flat surface.

## Weight ablation

The legacy combined setting (`entry_weight=1`) passed 2/6. On the four failed
routes, entry-only and outcome-only proved that the required behaviors were
present in the corpus but selected under different cost scales:

| Variant | Evaluated | Full passes | Key result |
|---|---:|---:|---|
| legacy combined | 6 | 2 | cross-tread and diagonal passed |
| entry-only | 4 legacy failures | 2 | turn-90 and reversal passed |
| outcome-only | 4 legacy failures | 2 | side exit and reversal passed |
| dimension-normalized combined | 6 | 5 | only side final surface failed |

This supports a cost-scale diagnosis rather than a data-coverage diagnosis.
The normalized combined setting is now the checked-in default; every ablation
is still reproducible through the qualification CLI.

## Deterministic identities

| Route | Matrix SHA-256 | Chunk-event SHA-256 |
|---|---|---|
| cross-tread-left-to-right | `fd2d8d155dca1f1b76ef6382db7f6506c7428492adbb924c2f5972da37fdff9d` | `c11a53083aab44524b36fe25926c677f46478e5bea8537fc630acbdc40bfe91f` |
| turn-90-middle-left | `f100db91c345ec0a8e66ef52ef82adcda6233186b7c0f21e0bd28d7b67b440d2` | `8edb3f37377151d2ec7f758cd69d6bb34d85547176064d3cb46ea29e9cffa50d` |
| diagonal-down-left | `627bdec88f259f1eee2f19fc76114314edece7578c438ced83468e27194fc19d` | `948a4a1811e69d2b917d5c2ff97b0e84a8cb24150a02e113e6ac6b4e0b2b4250` |
| side-exit-upper-left | `3582566e5e65b9c666e06b8cf5b07578dac7720c929c2ea4acb0f5de0ba9cf0a` | `2f9e4d531c1f7810b9a35705dce04f9571d7928f3781f1efa1788769104f729e` |
| riser-reversal | `ae773bf05cee2d3dec3ab2c9f8e94eb1f7f9e5d689d94fab3ff8e87f30f49199` | `f71fd99fd03ffe6eef45ade11305bdbddff0c30176a5c0554e4118b18cba3d00` |
| mixed-adversarial | `2ceae7fdcbf37b86fc17b9c9cacb0c41b565e612e970005de5da8fcb138da8be` | `8ab0522dc7ee8921e303bab8659aa75e5f0f1c73f8989ad8eb66d95de5b1b093` |

## Verification

Fresh focused and neighboring verification after the final implementation:

```text
Ran 366 tests in 21.779s
OK (skipped=2)

Ran 4 CLI contract tests in 0.002s
OK
```

The two skips are the existing opt-in authenticated contact-segment and stair
alignment oracles. The final six route artifacts were then regenerated with the
committed serializer; their behavior hashes exactly matched the prior qualified
v2 run.

## Decision and next work

The continuation requirements are satisfied:

- 6/6 routes executed without exception;
- 5/6 passed all frozen behavior gates (required: at least 2/6);
- maximum support-height p95 was `0.0199 m` (required: at most `0.03 m`);
- all chunk endpoints were stable double support with consecutive source frames;
- the 90-degree pivot retained progress and final-heading passes.

Proceed to the interactive viewer/full route matrix. The next algorithmic work
should target (1) terminal-surface planning for side exit and (2) stance/contact
locking or a slide-aware transition cost. Sonic/tracking remains intentionally
out of scope until the kinematic visual-quality pass is complete.
