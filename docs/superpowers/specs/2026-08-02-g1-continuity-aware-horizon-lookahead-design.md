# G1 Continuity-Aware Horizon Lookahead Design

## Problem and Evidence

The first same-skill continuation build did not visibly change the operator experience. On the deterministic `riser-stop-restart` route it committed only one continuation in 290 frames and reduced global selections only from eight to seven.

The failure occurs before playback. Global horizon search ranks only the immediate 25-, 50-, or 100-frame outcome. It does not consider whether the selected terrain skill contains a later stable endpoint that remains compatible with the upcoming terrain. Inventory inspection found that only 35–44 percent of horizon records have any later stable endpoint. Among the seven selected chunks in the corrected smoke run, two had no later endpoint and two more had terrain-incompatible suffixes. Continuation therefore cannot repair most choices after selection.

## Scope

This change remains privileged-heightmap, 50 Hz, and kinematic-only. It changes candidate selection for the experimental `--continuous-skill` mode. It does not add Sonic tracking, physics, depth inference, endpoint warp, foot IK, or strict collision preview.

The strict raw-foot feasibility arm remains separate. It must not be silently enabled by continuity lookahead because the current foot-lock filter changes the emitted pose after selection.

## Selection Architecture

For a nonzero moving command with `--continuous-skill` enabled, global search performs a continuity-preferred pass before the existing pass.

For each normally ranked candidate:

1. Validate its immediate endpoint with the existing terrain and optional contact-phase gates.
2. Find the first later stable double-support endpoint in the same placed skill, using the existing 25-frame target and lateness window.
3. Require at least 5 cm of suffix root progress and at most five suffix stall frames.
4. Validate the combined rigid placement from the candidate entry through that later endpoint against the query terrain using the continuation-only relative surface profile and its explicit 8 cm tolerance.
5. Accept the first normally ranked candidate that passes all four layers.

The lookahead does not add a new soft cost. It is a preferred feasibility layer so a marginally cheaper dead-end candidate cannot beat a candidate with coherent runway.

## Fallback and Command Semantics

If the continuity-preferred pass exhausts candidates, search repeats with the current immediate-horizon validation. This preserves coverage and prevents a new freeze mode.

Zero commands never request runway. A changed nonzero command may prefer runway for the new command but must not continue the old skill. Once a coherent skill is playing, unchanged-command continuation retains the existing no-restart behavior. Release-to-current-double-support and backspace reset remain unchanged.

The implementation records whether selection used `continuity-preferred` or `immediate-fallback`, plus rejection counts for `no-later-endpoint`, `suffix-progress`, `suffix-stall`, and `suffix-terrain`.

## Failure Handling

No unchecked candidate may play. If both preferred and immediate passes fail, the matcher holds the last completed safe endpoint. Exceptions from a terrain-domain miss are converted to a typed rejection rather than terminating the viewer.

Default multi-horizon behavior remains unchanged when `--continuous-skill` is absent.

## Testing and Acceptance

Unit tests must prove:

- a slightly more expensive candidate with valid runway is selected over a cheaper dead end;
- immediate fallback selects the dead-end candidate when no runway candidate exists;
- zero commands skip runway preference;
- changed commands do not continue the old skill;
- default-off behavior and stable ordering remain unchanged;
- failed preferred and immediate passes hold the last safe endpoint.

The deterministic A/B uses the exact retained viewer configuration and at least these routes: held forward ascent, `riser-stop-restart`, diagonal ascent/descent, and a turn on the stairs. A build is ready for operator feedback only if:

- all routes complete without exception;
- held-forward motion commits at least two same-skill continuations or reduces global selections by at least 40 percent;
- no route loses its previous behavioral pass;
- maximum moving-command stall remains at most five frames; and
- visual inspection shows continuous stepping rather than repeated settling at chunk boundaries.

If these gates fail, the result is evidence against one-horizon lookahead; do not compensate by forcing 100-frame chunks or weakening terrain compatibility in the same experiment.
