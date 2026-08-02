# G1 PHP-Style Terrain Skill Qualification Slice

## Result

**Classification: useful partial falsifier; not ready for viewer feedback.**

The implementation reproduced the useful PHP structure: ordinary 27-value
motion matching ranks stable entry states, whole-skill command and terrain
filters reject incompatible candidates, and accepted skills replay sequentially
with inertialization until a stable double-support switch.

All six hard routes executed for their complete frame count without an
exception, but none passed every frozen behavior gate. The continuation gate
of at least two behavior passes was therefore not met.

## Corpus

- Dataset: `build/torch-grail-terrain-feedback-v2`
- Clips: 760
- Search rows: 379,404
- Coherent terrain skills: 676
- Stable internal/pre-skill entry rows: 49,109 before the one-second remaining
  horizon and command filters
- Playback frequency: 50 Hz

## Best measured slice

| Route | Frames | Failure reasons | Root progress | Longest stall | Support error p95 | Transitions |
|---|---:|---|---:|---:|---:|---:|
| cross-tread-left-to-right | 410 | moving-command stall | 1.724 m | 23 | 0.016 m | 4 |
| turn-90-middle-left | 345 | final-command lateral drift | 1.542 m | 3 | 0.016 m | 4 |
| diagonal-down-left | 460 | moving-command stall | 1.296 m | 39 | 0.018 m | 5 |
| side-exit-upper-left | 445 | final surface; moving-command stall | 1.225 m | 47 | 0.019 m | 6 |
| riser-reversal | 390 | reverse-down progress; moving-command stall | 1.053 m | 158 | 0.018 m | 6 |
| mixed-adversarial | 500 | four segment-progress gates; final surface; stall | 1.333 m | 18 | 0.019 m | 7 |

Cross-tread and side-exit use the zero-travel-filter rerun under
`build/php-terrain-skills/slice-v2`; the remaining rows use the six-GPU slice
under `build/php-terrain-skills/slice-v1`.

## What changed the result

1. Whole clips with only a single start entry could not turn on the staircase.
2. Adding stable double-support entries inside coherent clips made the 90-degree
   pivot reachable: its pivot-progress and final-heading gates changed from fail
   to pass, leaving only lateral drift.
3. Vectorized whole-remaining-skill yaw/travel filtering prevented a locally
   cheap but globally wrong skill from winning.
4. Rejecting zero-travel skills improved cross-tread progress from 1.204 m to
   1.724 m and reduced its longest stall from 39 to 23 frames.
5. An exact 10-frame source-stall hard gate was tested and rejected because it
   exhausted the useful candidate set and did not finish in a reasonable search
   window.

## Falsifier

Entry-only MM plus one coherent remaining-clip trajectory is not sufficient for
the full omnidirectional problem. Terrain support alignment is already accurate
(1.6--1.9 cm p95 on all routes), but the system still needs a skill-level
progress model that scores *time-local* displacement/contact outcomes, not just
net remaining travel and yaw. The next experiment should add a short sequence
descriptor over fixed 0.5/1.0/2.0-second horizons and retain sequential playback
only through the selected horizon or the next stable double support.

## Verification

Focused and neighboring suite:

```text
Ran 107 tests in 3.229s
OK (skipped=1)
```

The skip is the existing opt-in authenticated contact-segment oracle.
