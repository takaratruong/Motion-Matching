# G1 Terrain Normalization and Phase-Selection Investigation

**Date:** 2026-07-30  
**Scope:** deterministic 50 Hz kinematic MuJoCo motion matching on the frozen
21-route same-stair omnidirectional suite.

## Result

Restoring the small-corpus feature normalization while retaining the combined
expanded motion corpus and ranked safe-rescue matcher improved the measured matrix from
14/21 to 15/21 routes. It restored `cross-tread-left-to-right`, retained both
180-degree turns and `riser-reversal`, reduced mean stalled-moving fraction
from 0.0261 to 0.0171, and kept rescue-cycle count at zero.

This result does **not** solve the system. The six remaining failures are:

- `side-mount-left`;
- all four `side-exit-*` routes; and
- `mixed-adversarial` final-surface completion.

The failures now share one lateral terrain-exit mechanism rather than the
previous mixture of reversal, turn, cross-tread, and exit failures.

## Reproducible configurations

The final improved 15/21 matrix used:

- motion dataset:
  `build/torch-terrain-expanded-combined-v39`, rebuilt from the current branch
  with both `down-continuous-33` and `staircase-final-v3` admitted;
- experiment:
  `sonic/configs/experiments/torch_terrain_expanded.json`;
- normalization dataset:
  `build/torch-stair-small`;
- normalization experiment:
  `sonic/configs/experiments/torch_stair_small_terrain_weight3.json`;
- normalization digest:
  `7240411fe82683b1ce9ac66c500a3b0dbebbbf60235ad50fd8f10515fb41f27a`;
- output shards:
  `build/omni-expanded-combined-smallnorm-full-g{1..5}-v41`.

The earlier v3-only `v35` matrix produced the same 15/21 pass set. The exact
combined corpus slightly improved aggregate rescues, stall, and sliding.

The immediately preceding 14/21 matrix used the same matcher family and v3
corpus but normalization digest
`d0a69e43c291fa9b1fc2d23455d87d3a053ebaf17d6d1309a622685e52eb6e4b`
from the expanded corpus. Its output shards are
`build/omni-expanded-crossclip25-full-split-g{1..5}-v23`.

## Aggregate comparison

| Arm | Passes | Rescues | Rescue cycles | Mean stalled fraction | Worst stall | Stance slide |
|---|---:|---:|---:|---:|---:|---:|
| Original retained (`v10`) | 13/21 | 12 | 0 | 0.02835 | 5 frames | 10.763 m |
| Expanded normalization (`v23`) | 14/21 | 161 | 0 | 0.02615 | 6 frames | 15.757 m |
| Small normalization, combined corpus (`v41`) | **15/21** | 168 | 0 | **0.01690** | **4 frames** | 15.695 m |

The small normalization is retained as the best behavioral arm, but the
increased rescue count and sliding versus `v10` remain unresolved.

## Cross-tread root cause

At the lateral-command boundary (`cross-tread-left-to-right`, frame 259):

- the working `v10` trajectory was at matcher root
  `[1.394, -0.877]` in `staircase-side-stepto` source frame 338;
- the expanded-normalization trajectory was at
  `[1.492, -0.928]` in `grail-stair-updown-0000` source frame 208.

The failing trajectory had advanced approximately 0.11 m farther along the
stair and one support level higher before the same command arrived. At that
state the useful side-step phase around source frame 509 was:

- rank 316 at the first measured lateral selection boundary;
- rejected by the unchanged emitted-window terrain validator;
- terrain-feature cost about 74.1, versus about 50 for the available safe
  but low-progress side-step phases near 413--443.

The matcher therefore did not merely overlook a good candidate: the upstream
normalization-induced path put the robot in a state from which that candidate
was not terrain-safe. Restoring the small normalization recovered both
cross-tread directions (`0.444` and `0.210` segment progress ratios).

## v3 phase-selection tradeoff

`staircase-final-v3` contains a full ascent and descent:

- useful ascending motion begins near source frame 60;
- the root is still ascending through roughly frame 260;
- turnaround occurs around frames 280--300;
- the useful descending phase continues afterward.

With v3 admitted, `side-exit-upper-right` entered v3 at route frame 80,
source frame 82, while still on flat terrain. That preparatory choice ultimately
put the robot near `[1.478, -0.739]` at the exit-command boundary and the route
failed its final-flat condition.

Without v3, the same route stayed in the GRAIL/side-step path, reached
approximately `[1.287, -0.869]`, followed side-step frames 564 onward, and
passed with exit progress ratio `0.572`. However, removing v3 regressed:

- `riser-reversal` from pass (`0.260`) to fail (`0.015`);
- `diagonal-down-left` from pass (`0.378`) to fail (`0.003`); and
- the mixed route's turn/reversal behavior.

The focused no-v3 output is
`build/omni-expanded-nov3-smallnorm-focused-v36`.

Searchable-v3 start-frame experiments at 100, 160, 220, 240, and 280 all
restored the side exit but lost the reversal and diagonal-down benefit. Outputs
are `build/omni-expanded-v3-start{100,160,220,240}-focused-v38` and
`build/omni-expanded-v3-descent-only-focused-v37`.

## Causal limitation

For the upper-exit and reversal routes, all commands before the late command
change are identical forward ascent. A causal interactive matcher cannot know
during early ascent whether the operator will later request a side exit or a
reversal. The current v3 clip requires an early multi-second commitment to reach
its useful descent phase, while the side-step path is required for the best
lateral exit.

Consequently, another static clip weight, registration, or unconditional phase
cut cannot select the correct early path for both future commands. The next
experiment must change one of these architectural assumptions:

1. add short-horizon transition planning that can reach a safe descending or
   lateral phase after the command changes;
2. add explicit command commitment/action chunks that expose intended future
   commands, accepting their latency tradeoff; or
3. collect bridge motions that connect a shared ascent state to both immediate
   reversal and lateral-exit phases.

## Next experiment acceptance

The next candidate must be evaluated on the exact 21-route matrix and must:

- retain at least 15/21 existing passes;
- make at least one currently failing side-exit route finish flat;
- keep `riser-reversal`, both cross-tread routes, and both 180-degree turns
  passing;
- keep rescue cycles at zero;
- keep mean stalled-moving fraction at or below 0.025; and
- avoid increasing total stance slide above the current 15.695 m.

Any proposal requiring knowledge of a command before the operator supplies it
must state the added command latency explicitly.
