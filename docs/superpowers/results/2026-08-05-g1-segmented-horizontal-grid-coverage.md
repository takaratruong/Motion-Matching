# G1 Segmented Horizontal Grid Coverage Result

## Question

Does the fixed horizontal staircase grid really lack interior motions, or did
the original whole-window evaluator hide reusable walking segments when a
mount or dismount failed?

## Root cause

The original evaluator required one unedited raw window to cover ground
approach, mount, elevated interior, dismount, and ground exit. Any boundary
failure rejected the complete lane. Its corrected whole-window result was
therefore only `3/11` full lanes, despite containing no independent
measurement of interior gait coverage.

The segmented evaluator derives mount, stable interior, and dismount intervals
from the two nominal foot-height tracks and certifies each phase separately
with the existing rigid-placement gates. Interior contact signatures remove a
common height offset but preserve the left-right split.

## Exact experiment

- Target: `grail-stair_p1-db7949fce1b2e48d39f4`
- Paths: `+X`, from `-1.2795985755 m` to `+1.2590216406 m`
- Lanes: `y=-1.0..+1.0 m` at `0.20 m`
- Nominal step width: `0.20 m`
- Corpus: all `12,645` height-grid GRAIL source clips
- Coarse pool: `2,000`
- Rigid-placement shortlist: `800`
- Joint edits: none

## Result

Lane-level coverage:

- full: `2`
- partial: `9`
- infeasible: `0`

Phase-level coverage:

- full: `15`
- partial: `8`
- infeasible: `10`

Interior-only coverage:

- full: `9`
- partial: `1`
- infeasible: `1`

| Lane y (m) | Mean height (m) | Left/right split (m) | Interior | Selected raw window |
|---:|---:|---:|---|---|
| -1.0 | 0.169 | ±0.169 | full | `grail-stair_p2-aed0cd704840558acdd1`, `[13,128)` |
| -0.8 | 0.663 | ±0.324 | infeasible | — |
| -0.6 | 0.987 | ±0.000 | full | `grail-curb-8e4fd24e39e2a3f8639b`, `[335,432)` |
| -0.4 | 0.903 | ±0.084 | full | `grail-curb-2b0d8a15c9b72d9327bc`, `[153,253)` |
| -0.2 | 0.789 | ±0.029 | full | `grail-curb-faa7b1c207f92ca7d8d4`, `[400,484)` |
| 0.0 | 0.708 | ±0.052 | full | `grail-curb-2b0d8a15c9b72d9327bc`, `[153,253)` |
| +0.2 | 0.581 | ±0.074 | full | `grail-curb-941f5d49acf12dcc42e5`, `[125,204)` |
| +0.4 | 0.428 | ±0.079 | full | `grail-curb-2b0d8a15c9b72d9327bc`, `[153,253)` |
| +0.6 | 0.268 | ±0.081 | full | `grail-curb-2b0d8a15c9b72d9327bc`, `[153,253)` |
| +0.8 | 0.153 | ±0.034 | partial, 0.683/1.189 m | `grail-stair_p1-06e79e3085dcaf62a1aa`, `[316,376)` |
| +1.0 | 0.060 | ±0.060 | full | `grail-curb-fe5ccf4d29d80932719b`, `[245,332)` |

The same unedited raw window,
`grail-curb-2b0d8a15c9b72d9327bc [153,253)`, certifies four interiors at
different elevations (`y=-0.4, 0.0, +0.4, +0.6`). This is direct evidence
that a plank gait can be reused by rigid placement.

The one infeasible interior, `y=-0.8`, is not an ordinary even plank. Its two
nominal foot tracks differ by approximately `0.648 m`. That is substantially
larger than the approximately `0.16 m` split already covered by other lanes.

## Boundary interpretation

The remaining lane-level partial results are predominantly boundary failures,
not missing interior walking:

- high side mounts such as the approximately `0.987 m` platform at `y=-0.6`
  are correctly allowed to remain infeasible;
- valid interior segments remain visible and reusable;
- partial dismounts are retained rather than discarded with the lane.

This supports the intended architecture: retrieve/place mount, repeat or reuse
the interior gait, retrieve/place dismount, then solve only the two transition
splices.

## Runtime correction

The first phase-aware implementation used scalar Python calls for every
window/query pair and was stopped after more than 30 minutes. A batched NumPy
kernel was added and regression-tested against the scalar
`contact_signature_cost` formula at `1e-12` tolerance. A 20-clip benchmark
projected approximately `78 s` for corpus scoring on 32 workers; complete
wall time is longer because rigid sole/contact certification is still scalar.

## Determinism

Two complete full-corpus runs (`v2` and `v3`) produced:

- byte-identical `phase-grid-summary.json`;
- byte-identical 23 phase traversal connectors;
- summary SHA-256
  `9485694458f82ac4f4a1ccb230ac65ca201b6edd126e3373e64c7b982e1a2b6a`.

## Artifacts

- `build/g1-horizontal-grid-phase-20cm-v2/phase-grid-summary.json`
- `build/g1-horizontal-grid-phase-20cm-v2/lanes/`
- `build/g1-horizontal-grid-phase-20cm-v2/interior-grid-playlist.npz`
- `build/g1-horizontal-grid-phase-20cm-v2/interior-grid-playlist.json`
- deterministic repeat: `build/g1-horizontal-grid-phase-20cm-v3/`

The live MuJoCo viewer plays the ten accepted interior segments at 50 Hz.
