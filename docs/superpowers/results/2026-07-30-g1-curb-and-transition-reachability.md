# G1 Curb Admission and Transition-Reachability Result

Date: 2026-07-30

Branch: `research/g1-torch-terrain-kinematics`

## Result

One authenticated curb recording was added to the expanded terrain corpus, but
the frozen omnidirectional matrix remained at 15/21 passing routes. A subsequent
strict, read-only bounded transition search found no complete safe side-exit
landing for any of the four failing exits.

The result does not justify a runtime planner. It finds no safe frontier among
the top four retained transitions for either lower exit and no complete
upper-exit landing within the bounded search. An earlier relaxed exploratory
probe did expose useful upper-exit bridge kinematics, but forcing them improved
progress without making either route finish flat. Under this frozen allocation,
the evidence points toward missing lateral drop/landing coverage; it does not
rule out a wider offline connectivity search.

The reachability search is diagnostic only. It did not alter any selected clip,
source frame, or dense output window in paired route/control replays.

## Authenticated curb registration

`chair-step-climbing-final` embeds a constant source object pose:

```text
object_pos_w XY = (-0.07415730506181717, 0.3856179714202881)
```

The checked chair configuration places the same box at:

```text
BOX_POSITION XY = (-0.1, 0.5)
```

The registry therefore applies their exact difference:

```text
motion_to_terrain_xy_yaw =
(-0.025842694938182836, 0.11438202857971191, 0.0)
```

This is source-derived registration, not a fit against the admission threshold.
The source motion hash and checked chair configuration hash remain pinned.

With identity registration the clip was rejected:

- contact-like samples: 296;
- elevated contact samples: 58; and
- p95 contact-height error: 0.050826 m.

With the recorded-object registration it was accepted:

- contact-like samples: 413;
- elevated contact samples: 175; and
- p95 contact-height error: 0.022378 m.

The rebuilt corpus is:

```text
build/torch-terrain-expanded-combined-v42
build/torch-terrain-expanded-combined-v42-admission.json
```

It contains nine clips including flat locomotion. Malformed, layout-invalid,
FK-inconsistent, and unauthenticated chair clips remain rejected.
The final manifest identity is:

```text
48543eae034bdac979ebcf0c7db062daac4873480de4aac2ca979d77eb1c5702
```

Chair archives with no embedded object pose now produce the structured
`missing_authoritative_terrain` rejection rather than borrowing external
geometry without source registration.

## Frozen 21-route comparison

The v42 corpus was evaluated with the retained expanded matcher and the frozen
small-corpus normalization on five GPUs:

```text
build/omni-expanded-curb-smallnorm-full-g{1..5}-v43
```

The later fail-closed chair change altered only rejected-candidate metadata;
the nine accepted motion and terrain artifacts used by the matcher are
unchanged.

| Metric | Previous v41 | Curb v43 |
|---|---:|---:|
| Passing routes | 15/21 | 15/21 |
| Terrain rescues | 168 | 189 |
| Rescue cycles | 0 | 0 |
| Mean stalled-moving fraction | 0.016903 | 0.016929 |
| Worst stall | 4 frames | 4 frames |
| Total stance slide | 15.695 m | 15.616 m |

The curb clip was selected for only 15 frames across three already-passing
routes. It was never selected during a failing side mount or side exit.
Nevertheless, it increased diagonal-up progress:

- right: 0.173 to 0.353; and
- left: 0.465 to 0.521.

The clip is useful and retained, but it does not contain the missing lower
lateral-exit bridge.

## Read-only reachability contract

The diagnostic uses:

- maximum transition depth 2;
- beam width 4 for the frozen evidence run;
- maximum 64 composed and validated states;
- every possible source advance from 1 through 15 frames between the first and
  second transition;
- only the currently issued command;
- the retained continuity and loop-revisit ranking costs, with hypothetical
  per-frame selection history propagated through the first-hop motion;
- an explicit 46-frame composed emitted-window clearance validator, distinct
  from the unchanged runtime matcher's retained 10-frame preview; and
- a terminal requiring full 46-frame terrain safety, at least 0.05 m of
  command-aligned root progress, at least 0.05 m of surface drop under each
  foot, and simultaneous contact-like clearance for both feet at least once in
  the final 10 frames. Progress, both-foot drop, and support must hold at that
  same tail frame, and at least one foot must remain support-like at the final
  frame.

Each result reports command-change frame, compute nanoseconds, expanded count,
depth, reachability, first and terminal source identities, and budget
exhaustion. A result at exactly 64 compositions conservatively records that the
search cap was reached; it does not inspect a 65th state to distinguish natural
frontier exhaustion.

The frozen 64-state allocation composes the top four ranked first hops, then
tests the top-ranked second hop at each of the 15 permitted advance timings for
every safe first-hop parent: `4 + 4 × 15 = 64`. This is a bounded beam
experiment, not an exhaustive search over every database row at every timing.

## Side-exit replay

| Route | Reachable | Depth | States | First phase | Terminal phase | Compute |
|---|---:|---:|---:|---|---|---:|
| lower-left | no | 1 | 4 | — | — | 0.526 s |
| lower-right | no | 1 | 4 | — | — | 0.528 s |
| upper-left | no | 2 | 64 | — | — | 6.843 s |
| upper-right | no | 2 | 64 | — | — | 6.882 s |

The exact current replay used base config
`4b213eac374982979f60174120ad27381100cc6f36b87e52244c38273f2f484c`
and retained normalization
`7240411fe82683b1ce9ac66c500a3b0dbebbbf60235ad50fd8f10515fb41f27a`.
Every result tensor and every stable matcher diagnostic was compared against a
separate untouched control matcher. The paired output hashes were identical:

| Route | Diagnostic output SHA-256 | Control output SHA-256 |
|---|---|---|
| lower-left | `b5304f576ff3f2bfaab8e6333f000f77ba56730f7e121d965146dc2f56355fd2` | `b5304f576ff3f2bfaab8e6333f000f77ba56730f7e121d965146dc2f56355fd2` |
| lower-right | `92e58a4647b6b3ac46c445c5cd65c0cc8e1fc203df0fa635eaebf9d8aee0efcf` | `92e58a4647b6b3ac46c445c5cd65c0cc8e1fc203df0fa635eaebf9d8aee0efcf` |
| upper-left | `4ab4e95a2a40f32609fd729bf6b03b634a01371f1ed9e0ef09c8ec2441d5e97d` | `4ab4e95a2a40f32609fd729bf6b03b634a01371f1ed9e0ef09c8ec2441d5e97d` |
| upper-right | `e41d09973edd2b589955600fd99df071f08c5c17c372c68717381ae4ca869379` | `e41d09973edd2b589955600fd99df071f08c5c17c372c68717381ae4ca869379` |

The lower exits had no safe depth-one frontier among their top four retained
ranked transitions. Both upper exits reached the 64-composition cap without
finding a strict terminal, so the bounded result does not claim that a wider
or longer offline search is impossible.

The measured upper-route search time is about 6.9 seconds, roughly 340 times
the 20 ms control period. It is suitable as an offline causal diagnostic, not
as the current runtime implementation. Reported planner time begins inside the
generic bounded search and excludes public-adapter argument validation and
command tensor construction.

The canonical captured-state evidence is tracked at:

```text
docs/superpowers/results/
2026-07-30-g1-curb-and-transition-reachability-evidence.json
```

The captured artifact's SHA-256 is
`8576fc84b1e1408b0601f6c0af43535c582a5f68e9547058879baa5417e7366d`.
Reruns reproduce the stable identities and paired output hashes; measured
`planner_compute_ns` values and therefore the whole-file SHA are expected to
vary. Run it with:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_torch_terrain_reachability.py \
  --dataset build/torch-terrain-expanded-combined-v42 \
  --config sonic/configs/experiments/torch_terrain_expanded.json \
  --normalization-dataset build/torch-stair-small \
  --normalization-config \
    sonic/configs/experiments/torch_stair_small_terrain_weight3.json \
  --output docs/superpowers/results/\
2026-07-30-g1-curb-and-transition-reachability-evidence.json \
  --device cuda
```

## Relaxed bridge probe and forced-phase falsification

Before the strict both-foot landing terminal was introduced, a deliberately
relaxed mean-support-drop probe found `side-step` frame 418 for the upper-left
exit and frames 410 then 446 for the upper-right exit. Those candidates met
command progress and clearance bounds, but could leave one foot high or
unsupported, so they are not accepted terminals.

As a falsification check, the relaxed probe's first phase was forced at each
upper-exit command boundary. For the two-hop right exit, the first phase was
held for 15 control intervals (0.30 s) before forcing the second phase. Normal
matching resumed afterward.

| Route | Baseline exit progress | Forced exit progress | Final result |
|---|---:|---:|---|
| upper-left | 0.141 | 0.150 | fail: final surface not flat |
| upper-right | -0.079 | 0.359 | fail: final surface not flat |

Thus the short descending phases can improve commanded progress, but selecting
them is insufficient to solve the route. This evidence motivated the stricter
terminal contract above. No behavior-changing planner is retained.

## Verification

Focused regression command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_terrain_features \
  tests.python.test_sonic_torch_transition_reachability \
  tests.python.test_sonic_torch_terrain_reachability_rollout \
  tests.python.test_run_g1_torch_terrain_reachability \
  tests.python.test_torch_terrain_registry \
  tests.python.test_torch_terrain_adapters \
  tests.python.test_torch_terrain_admission
```

Result: 64 tests ran; 62 passed and two explicitly opt-in real-data oracles
were skipped.

The captured-state replay comparison is also opt-in because it takes about two
minutes and requires the local GPU datasets:

```bash
RUN_G1_TERRAIN_REACHABILITY_ORACLE=1 CUDA_VISIBLE_DEVICES=6 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_run_g1_torch_terrain_reachability
```

It regenerates all four routes in a temporary file, removes only
`planner_compute_ns`, and requires exact equality with the tracked evidence.

## Next experiment

Do not put the current beam search in the 50 Hz loop. Seek or collect explicit
lateral drop and landing motions for lower and upper tread exits. A later
connectivity experiment may extend the offline horizon only after the new
coverage baseline is frozen; it must not be confused with a real-time 50 Hz
planner.
