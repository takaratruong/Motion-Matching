# G1 Interactive Terrain Coverage Design

## Status

Approved for autonomous execution under the operator's standing instruction to
continue quality-first terrain improvement without waiting for feedback.

## Goal

Prevent valid interactive commands from appearing frozen when an immediate
mid-step replan has no feasible exact-contact replacement. Preserve the strict
terrain/contact validator and the qualified v58 route behavior.

## Evidence

The live viewer remained mapped, focused, and sampled keyboard input. At the
reported frozen pose, the matcher rejected all 64 validated candidates:

- 45 for landing-height mismatch;
- 19 for stance-height mismatch; and
- 9,637 lower-ranked candidates were never exactly validated.

The viewer then correctly latched that failed command to avoid repeating an
expensive identical search. Offline stress routes reproduced the same failure
at command boundaries. Both diagonal descents exhausted their complete
prefiltered inventories, proving that a larger shortlist alone cannot solve the
general failure.

At each reproduced boundary the phase-gated horizon adapter forcibly truncated
the current chunk even when the current source support was single stance. This
bypassed the base matcher's safe-interrupt rule. A failed replacement therefore
aborted instead of preserving the already-valid step until double support.

## Considered Approaches

### Repeat the failed 64-candidate search

Rejected. A deterministic search with unchanged state and command produces the
same failure while consuming compute and blocking the viewer.

### Relax exact contact constraints

Rejected. Responsiveness obtained by accepting invalid landing or stance
geometry would undo the sole/contact qualification and reintroduce the stair
collisions that earlier variants exhibited.

### Adaptive exact-search expansion

Retained as an ablation for failures with an unvalidated shortlist. It cannot
solve failures whose complete prefiltered inventory is exact-invalid.

### Transactional immediate replan

Selected. Keep the current immediate command-change search for responsiveness.
If that search has no exact-valid replacement while the current skill is not at
a safe interrupt frame, discard the attempted interruption, advance the
already-valid chunk, retain the new command as pending, and retry through the
base matcher's existing double-support gate. Do not catch failures at a safe
interrupt point or endpoint; those remain true coverage failures.

## Architecture

Make forced mid-step interruption transactional in the horizon matcher. The
temporary truncated state is used only for the immediate attempt. On
`HorizonSearchFailure`, restore the original state and invoke the existing base
prepare path, which advances the skill and marks the command for replanning.
The next source-labelled double-support frame performs the normal exact search.
No fallback pose is synthesized and no terrain failure is accepted.

The offline route runner and live viewer use the same matcher implementation.
No viewer-only retry or fallback motion is permitted.

## Evaluation

Use the existing deterministic same-stair route corpus, prioritizing:

- `riser-stop-restart`;
- `riser-reversal`;
- both 180-degree turns;
- upper and lower side exits;
- side mounts; and
- `mixed-adversarial`.

For every retained version:

- count completed routes and exact coverage failures;
- measure command-change-to-root-motion delay and longest moving stall;
- retain exact emitted-contact validation and sole clearance;
- compare stance slide and root jerk with v58; and
- render failed and improved routes for visual inspection.

## Acceptance

A change is retained only if it reduces deterministic coverage failures or
moving-command stalls on the adversarial routes, keeps immediate preemption
when an exact-valid replacement exists, keeps the v58
`turn-90-middle-right` route complete, never accepts an exact terrain-invalid
candidate, and does not increase that route's `0.411985839 m` stance-slide
baseline.

## Scope

This work remains privileged-height, kinematic, and independent of Sonic,
physics stepping, depth learning, and real-time latency optimization. Existing
dirty contact-oracle and landing-bridge work is not modified.

## Execution Findings

The transactional single-support correction is retained. It preserved the
qualified v58 route bit-for-bit and moved `cross-tread-left-to-right` past its
frame-60 command boundary.

That longer route exposed a missing qualification check: the matrix's ankle
metric passed while exact sole geometry reached `-0.051865 m` clearance on 19
frames. The unsafe samples came from sequential source-skill continuations,
which used the coarse surface-profile validator instead of the exact emitted
sole preview. Sequential continuations must therefore pass the same filtered
pose and sole validation as newly selected chunks. A route that stops earlier
because this check rejects an extension is safer, but is not counted as a
locomotion-quality improvement.

The following recovery ideas were tested and rejected as defaults:

- ten stationary command-shaping retries only delayed diagonal descent from
  frame 282 to frame 292 and preserved the same failure;
- validating 512 instead of 64 candidates took more than 30 minutes per
  two-route slice, still failed every crossing/mount route, and often selected
  flat clips that never engaged the stair;
- disabling the terrain prefilter failed earlier because the first 64 ranked
  candidates were all exact-invalid; and
- increasing the future-surface sign gate allowed the left diagonal route to
  reach frame 360, but it moved opposite the requested descent and therefore
  regressed command outcome despite greater frame coverage.

These results rule out treating frame count alone as progress. Retained search
changes must improve commanded segment progress and exact sole safety together.

Exact continuation preview changes the old v58 turn route after output frame
328. This is intentional: the old route stops at source frame 445, one frame
before its selected continuation begins a landing that would reach
`-0.174130 m` sole clearance. The replacement route completes safely (minimum
sole clearance `-0.024206 m`) but has worse final heading and slide, so the old
frame-count contract is not sufficient evidence of a safe indefinite hold.

A latched stationary command now preserves a completed exact-safe pose when no
stationary replacement exists. Repeating the same stop does not repeat the
expensive deterministic search. This moved `riser-stop-restart` from a failure
at frame 179 to its restart boundary at frame 283; it did not by itself solve
the subsequent moving transition.

A two-tier monotonic exact search is retained. The normal 64-candidate search
and runway preference remain unchanged. Only after that search fails, a larger
shortlist is evaluated with runway preference disabled, so increasing rescue
coverage cannot replace an earlier successful choice. With a 256-candidate
rescue, `riser-stop-restart` completed all 290 frames, achieved a `0.544725`
restart progress ratio and `0.254820 m` total stance slide, and had zero sole
samples below `-0.025 m` (minimum `-0.024206 m`). The same rescue exhausted the
complete 167-candidate prefiltered crossing inventory without finding a safe
cross-tread transition, proving that rescue depth is useful but not the general
on-stair turning solution.

The accepted corpus does contain turn outcomes at split-height support: 7,870
horizons start with more than `0.10 m` foot-height split, and 278 horizons pair
that split with more than 20 degrees yaw and less than `0.20 m` root travel.
The remaining failure is therefore transition reachability/ranking rather than
simple absence of turning data. A hard support-foot entry-continuity prefilter
was also rejected: although it reduced exact stance-height rejections from 64
to 21 at one boundary, it removed useful candidates and made the staged turn
fail at frame 202 instead of frame 300. Transition reachability should be used
as a soft ranking feature or a rescue-only ordering signal, not a hard gate.

## Literature and Reference-Code Check

The official Perceptive BFM/TCRS implementation was inspected at commit
`eef3b35268bac19e762e49c519c8e8141e084944`
(`https://github.com/Mondo-Robotics/PMT`). Its G1 kinematic synthesizer does
substantially more than height-conditioned clip selection: it projects each
touchdown so the ankle and multiple sole points share a safe surface, locks the
result through stance, optimizes the mid-foot swing path against terrain,
reconstructs pelvis height from weighted support, repairs lower-leg collisions,
and runs multi-point Jacobian IK. This independently supports the diagnosis that
exact search is a safety oracle and motion-intent selector, but is not by itself
a complete terrain-conforming synthesizer.

The comparison also exposed a concrete bug in our horizon targets. During a
turn, the target generator rigidly rotated the *currently planted* foot offsets
with the future root heading and sampled those rotated points on the height map.
On a riser this fabricates large per-foot height changes before either foot has
stepped, and the surface-direction hard gate then removes tens of thousands of
otherwise relevant turning horizons. Stationary turn targets now keep current
footholds fixed. Broadly disabling the surface-direction gate for turns was
tested and rejected: it made the established 90-degree regression route fail at
frame 251 and left the 180-degree route failing at the same frame 285 after all
256 exact candidates were rejected. The sign gate is therefore retained; the
stationary-target fix removes the fabricated height signal at its source.

Two terrain-conformal synthesis ablations follow from the reference-code
comparison. Both are optional so route evidence can be compared against the
same exact-search baseline:

- Support-aware root height uses the mean locked-support ankle correction to
  adjust pelvis height before leg IK. It starts at zero (no reset pop), changes
  by at most 2 cm per frame, and resets at a newly rebased skill chunk so an
  already-filtered root offset is not applied twice. Exact candidate preview
  carries the same chunk-start marker and snapshots the root-filter state.
- Full-sole touchdown projection is a sparse repair: contacts whose exact sole
  samples already share one surface take the legacy foot-lock path bit-for-bit.
  Only an edge-straddling onset searches a 1 cm two-dimensional grid within a
  configurable radius. It lexicographically favors a footprint whose exact sole
  samples share one surface, then the original support height, then minimum
  shift. A successful projected XY is latched through stance and ankle height is
  reconstructed from all sole samples, rather than from one ankle height query.
  The projected support leg tracks that latched target directly for its entire
  stance; the other leg and non-leg joints keep their normal smoothing and speed
  limits. The 2D
  search is intentional: an omni-directional foot can be oriented across,
  rather than along, the stair height gradient.

The exact emitted-contact validator remains authoritative for both ablations;
neither correction may turn a rejected candidate into an unchecked output.

Standalone support-root reconstruction is rejected by route evidence. With a
0.05 s halflife, both the hard 180-degree route and the established 90-degree
control fail at frame 94 (240 of 256 exact candidates penetrate); with a 0.10 s
halflife the 180-degree route fails at frame 266, still earlier than the
fixed-root baseline at frame 285. The option remains isolated while the combined
touchdown experiment finishes, but it must not become a default.

Sparse touchdown projection preserves the 90-degree control for all 345 frames
and the 180-degree baseline through its existing frame-285 boundary. At that
boundary normal search rejects 39,100 records by the surface-direction sign and
reaches zero exact candidates, so touchdown placement is not yet exercised.
The next ablation keeps that sign gate for normal search but disables it only in
the expanded exact rescue when sparse projection is active. This preserves normal
search ordering and exposes the disputed clips only to the full safety oracle.

The terrain feature in Learned Motion Matching provides a more specific
follow-up than indiscriminately widening that rescue. Holden et al. query the
runtime terrain below each candidate's predicted future toe locations at 0,
15, 30, and 45 frames and compare those heights with the terrain fitted to the
source motion. Our normal horizon cost instead compares each source outcome
with one command-projected pair of future footholds. Candidate-specific foot
traces exist, but until now they were used only by a late binary prefilter.
That can discard a nearly compatible footstep that sparse projection could
repair, while removing the prefilter ranks candidates without using where they
actually step.

The next layered rescue therefore uses the placed source support trace as a
soft height cost. For every candidate it samples the query height below that
candidate's foot path, aligns the first supported surface, and averages the
squared relative-height error using the existing horizon height scale and
weight. The ordinary search retains its existing hard gates and exact preview.
Only a projection-enabled rescue removes the rigid profile gate, adds this
candidate-conditioned cost, and then applies exact filtered sole replay. This
tests the literature's candidate-toe principle without weakening safety or
changing the established route unless normal search has already failed.

Initial layered-rescue results separate coverage from ranking. Removing both
the surface sign gate and rigid profile prefilter but retaining the ordinary
ranking did not advance either hard boundary: `cross-tread-left-to-right`
still failed at frame 227 after 1,024 exact rejections, and
`turn-180-upper-left` still failed at frame 285 after 1,024. The staged turn
still failed at frame 300 after 1,024. At 512 candidates, all 512 mixed-route
rejections were stance-height failures. Larger unguided search is therefore
rejected as the explanation.

Candidate-specific support-trace cost changed which failures appeared but did
not find a safe transition at 512. For the crossing route, the first 256
changed from 343 stance-height rejections under unguided ranking to 128; at
512 the soft ranking produced 298 stance-height, 94 sole, 76 edge, and 44
landing-height rejections. The 180-degree route remained dominated by stance
height (487 of 512), and the mixed route had 498 stance-height rejections.
This is evidence that candidate terrain placement is informative, but the
remaining boundary is the root-anchored transition composition itself.

The next ablation places rescue candidates from the current support contacts.
The source clip keeps the current root heading, but its world translation is
the mean translation that aligns source entry ankles with the currently
supported ankle or ankles. Root and joint inertialization still make the first
emitted pose exactly continuous; they then decay toward the contact-anchored
clip placement while foot lock holds stance. Candidate acceptance and live
playback use the identical placement. This follows the support-anchor stage of
the inspected TCRS implementation and directly targets the measured
stance-height failures without relaxing their tolerance.

Anticipating the sparse touchdown shift during swing is rejected as a default.
Although it removed the direct onset correction's joint-speed bypass, it changed
the live state before later searches and made both primary adversarial routes
fail earlier: the cross-tread route moved from frame 227 to 187 (466 of 512
rescue candidates failed stance height), and the 180-degree turn moved from
frame 285 to 237 (331 stance-height failures). The mechanism remains an
explicit ablation, but contact anchoring is next evaluated with the prior onset
projection so its effect is not conflated with this known regression.

Static contact anchoring is useful but incomplete. In the anticipatory ablation
it found an exact-safe rescue after 213 exact rejections and played 76 frames
onto double support spanning adjacent stair levels. At that endpoint the two
ankles were approximately `0.410 m` and `0.245 m` high over surfaces at
`0.357 m` and `0.193 m`, respectively. The following search nevertheless
rejected all 512 candidates, 511 for stance height. Inspection showed that the
soft candidate trace was still sampled in the old pelvis-rooted frame even
though exact rescue composition used the support-contact frame. The next score
places every candidate from its overlapping entry contact(s) before querying
terrain, so split-height candidates are ranked at the locations where exact
playback will actually put them. Missing contact overlap remains invalid and
the exact per-frame sole replay remains the final gate.

Separating anticipatory projection confirms that the regression belongs to the
anticipatory mechanism, not static contact anchoring. With onset-only
projection, the established 90-degree control completes all 345 frames and all
non-timing output arrays are bit-for-bit identical to the v79 exact-safe
baseline. An independent MuJoCo sole audit measures `-0.024206 m` minimum
clearance and zero samples below `-0.025 m`. Its existing outcome contract still
reports the known `0.434182 rad` final-heading miss; that is not newly caused by
the rescue changes.

Onset-only projection also restores the staged-turn ablation from the
anticipatory failure at frame 198 to its prior frame-300 boundary. It completes
the entire staged in-place turn with `0.251281 m` total stance slide before the
requested diagonal descent; all 512 rescue candidates at that descent boundary
then fail stance height. This is the split-height re-entry case used to evaluate
the contact-frame cost.

The combined anticipatory/contact-anchor crossing ablation validates contact
anchoring as a useful mechanism despite the anticipatory regression elsewhere.
It advances `cross-tread-left-to-right` from the old frame-227 boundary to frame
371, selects three exact-safe contact-anchored rescues, and reaches a `0.385666`
commanded lateral progress ratio. It is not a retained quality result: the
route remains incomplete and accumulates `0.799548 m` stance slide. Subsequent
tests keep contact anchoring, disable anticipatory projection, and use the
contact-frame score.

The clean onset-only/contact-anchor crossing control advances the old
frame-227 boundary to frame 253 and selects one exact-safe anchored rescue, but
has not yet begun commanded lateral progress and then exhausts another 512
candidates. It also retains the direct projection's `19.625183 rad/s` peak
joint-speed defect. This isolates two later requirements: contact-frame ranking
must produce sustained transitions, and any retained touchdown projection must
respect the output speed bound without reproducing the rejected anticipatory
state drift.

Contact-frame terrain height alone is rejected as the split-height solution.
It preserves the 345-frame control bit-for-bit, but the 180-degree and staged
boundaries remain at frames 285 and 300; 491 and 511 of their respective 512
rescues fail stance height. The next soft ranking term measures double-support
geometry after the best shared contact translation. It penalizes stance
width/orientation residual continuously rather than reinstating the earlier
hard entry-continuity filter that removed useful candidates.

The crossing result agrees with that rejection: contact-frame height changes
the terminal boundary only from frame 253 to 254, selects the same rescue clip,
still has zero commanded lateral progress, and preserves the `19.625183 rad/s`
peak. Height placement is not the missing transition-continuity variable.

Soft entry stance geometry is also rejected. It promotes a contact-anchored
clip earlier, but both the 180-degree and staged routes regress to frame 156,
never engage the stair, and the next rescue changes from stance-dominated to
315 sole-penetration rejections. Entry alignability alone favors a locally easy
but globally wrong clip. The term is removed rather than accumulated.

The next synthesis ablation leaves ranking at the non-regressing contact-frame
baseline and increases only the pre-IK stance unlock radius from `0.25 m` to
`0.50 m`. Joint correction remains capped at `0.35 rad`, and every candidate
still requires exact emitted sole replay. This tests whether the heuristic
distance unlock is discarding transitions that the bounded IK can actually
repair.

The crossing route shows that entry geometry is not uniformly useless: the
ungated term reaches frame 288 with `0.538131` lateral progress and
`0.421949 m` stance slide, versus frame 254, zero lateral progress, and
`0.267697 m` slide for contact-frame height alone. The regression occurs because
the same term is applied while both supports are still on flat ground. A second
ranking ablation therefore activates stance geometry only when current support
is already elevated above the query grid's base or spans more than `0.05 m` of
height. This should preserve the flat approach while retaining its useful
on-terrain ordering; it is evaluated separately from the widened-lock test.

The widened pre-IK unlock is rejected. Its 90-degree control is bit-for-bit
unchanged, but the staged boundary is also exactly unchanged at frame 300 with
511 stance-height and one edge rejection. Candidates are not being lost to the
`0.25 m` distance heuristic; the bounded transition IK/composition itself
cannot realize them. The experimental CLI plumbing is removed.

Support-root reconstruction is revisited only with the missing terrain gate.
The earlier global versions regressed on flat approach frames because they
altered pelvis height before terrain engagement. The real runner now treats the
query grid minimum as its base surface and enables the existing bounded,
smoothed support-root offset only when current support is elevated by more than
`0.05 m` or spans more than `0.05 m`. Direct filter use remains ungated unless a
base is supplied. This is evaluated with a `0.025 s` halflife against the same
control and hard routes.

The terrain gate prevents the original flat-approach failure, but support-root
reconstruction is still rejected as a retained default. At a `0.025 s`
halflife it lets `turn-180-upper-left` emit all 420 requested frames and reaches
`0.036633 rad` final heading error, versus the fixed-root failure at frame 285.
That apparent coverage comes with `1.301439 m` total stance slide, only
`0.017043` commanded pivot progress, and a `14.894147 rad/s` peak joint speed.
More importantly, the established `turn-90-middle-right` control regresses from
345 to 254 frames and reaches a `29.401133 rad/s` peak, while staged descent
still fails at exactly frame 300. Pelvis-height freedom can expose transitions,
but the present mean support-height controller is not a quality-preserving
solution.

The lateral crossing confirms the rejection: support-root reconstruction moves
the terminal boundary only from frame 288 to 289, reduces commanded lateral
progress from `0.538131` to `0.525681`, increases stance slide from
`0.421949 m` to `0.430193 m`, and retains the `19.625183 rad/s` touchdown
spike.

The next transition test fits the rescue placement yaw as well as translation.
When both current and candidate entry supports overlap, the angle between the
source feet is aligned toward the live support-foot angle, clamped to 15
degrees. Single-support rescue keeps the root-heading placement unchanged.
Candidate-specific terrain/stance ranking, exact emitted preview, and committed
playback use the identical candidate yaw. This directly tests whether the
remaining double-support failures are caused by an avoidable stance-axis
residual rather than by missing motion data.

The 15-degree contact-yaw fit preserves the established 90-degree control for
all 345 frames. Every non-timing output array is bit-for-bit identical to the
fixed-yaw contact-frame baseline; stance slide remains `0.468954 m` and peak
joint speed remains `10.649070 rad/s`. This establishes that rescue-only yaw
fitting has no effect before ordinary search has failed.

Contact-yaw fitting solves one measured reachability case rather than all hard
routes. `cross-tread-left-to-right` completes all 410 frames, passes its outcome
contract with `0.504132` lateral progress, reaches `0.021205 rad` final heading
error, and has zero exact sole samples below `-0.025 m` (minimum
`-0.024954 m`). The fixed-yaw baseline stopped at frame 288. The same 15-degree
fit leaves the 180-degree, staged-descent, and mixed-route boundaries exactly at
frames 285, 300, and 180 respectively. It is retained as a real rescue
capability, not claimed as the general on-stair turn solution.

Coverage alone is insufficient for the crossing result. It accumulates
`1.021976 m` inferred stance slide over 410 frames and retains the
`19.625183 rad/s` projected-touchdown spike. Re-deriving the source support mask
for every emitted frame shows that every inferred sliding stance frame is also
marked source support; missing contact labels are not the cause. The foot-lock
filter was softly decaying supported-leg corrections. The next ablation treats
supported-leg IK as a hard task before the existing global `12 rad/s` speed
limit, while retaining soft decay for swing clearance and all non-support
joints. Direct touchdown no longer bypasses that speed limiter.

Hard stance is rejected globally and the terrain-gated form is evaluated only
as an ablation. The
ungated filter bounds peak joint speed at exactly `12 rad/s`, yet changes the
flat crossing approach and dead-ends at frame 63 after only two chunks. The
ablation uses the query-grid minimum as its base and makes support hard
only when a current support surface is more than `0.05 m` above that base or
the two support surfaces span more than `0.05 m`. Direct/unit use without a
base remains ungated. This mirrors the successful terrain gate for ranking and
keeps the established flat approach on its original soft correction path.

The terrain gate does not rescue hard stance. Five 15-degree routes and two
30-degree controls all fail: the ordinary turn, 180-degree turn, staged route,
and mixed route converge on the same frame-147 boundary, while lateral crossing
ends at frame 195. Crossing slide falls from `1.021976 m` to `0.184646 m` and
peak joint speed falls to `10.436138 rad/s`, which confirms that soft supported-
leg decay causes much of the visible sliding. But the exact emitted pose is no
longer reachable by the following transition, and 474 of 512 final crossing
candidates fail stance height. The implementation is removed; the next sweep
shortens the soft correction half-life instead of making all terrain support
an immediate hard task.

The direct onset touchdown repair remains a separate quality defect: accepted
route ablations have reached `19.625183 rad/s`, and the support-root regression
reaches `29.401133 rad/s`, because the projected support leg bypasses the normal
speed limiter. The bounded follow-up plans the same exact sole target but emits
no correction before the final three swing frames and never bypasses the output
speed bound. Exact emitted replay remains authoritative, so a target that
cannot be reached within that late window is rejected rather than snapped at
contact.

Late-swing anticipation is also rejected. Restricting its effect to the final
three swing frames does bound the observed peak at `10.170445 rad/s`, but the
90-degree control still changes state during its ascent and fails at frame 193
instead of completing 345. The retained correction direction leaves swing
playback untouched, makes only a safely projected support leg immediate before
the global output speed limiter, and leaves ordinary supported-leg correction
soft. This removes the direct-touchdown speed-limit bypass without imposing the
rejected hard task on every elevated contact.

Shortening ordinary foot-lock decay is also rejected. With the projected leg
still bounded at `12 rad/s`, `0.020 s` fails the crossing at frame 63 and
`0.015 s` fails at frame 185; the retained `0.025 s` reaches frame 254. Faster
ordinary correction changes transition reachability without solving the
projected touchdown, so the half-life remains `0.025 s`.

The `12 rad/s` projected-touchdown bound is itself too restrictive for the
lateral rescue. It removes the former `19.625183 rad/s` snap, but also removes
the exact-valid transition that completed the route. The bound is therefore an
explicit offline and live-viewer parameter included in the configuration
identity and result summary, rather than a hidden constant. The default remains
conservative until a route-qualified recommendation is selected.

At `16 rad/s`, the 15-degree contact-yaw crossing completes all 410 frames and
passes the outcome contract. Total stance slide drops from `1.021976 m` in the
unbounded v105 result to `0.779092 m`, lateral progress remains passing at
`0.494908`, and peak joint speed is exactly `16 rad/s`. An independent full
MuJoCo sole audit measures `-0.024954 m` minimum clearance and zero samples
below `-0.025 m`. The 18 and 20 rad/s variants choose the same route outcome and
improve slide by only `0.000130 m` and `0.000236 m`, respectively, while raising
the peak to 18 and `19.625183 rad/s`.

Refining the lower threshold finds a better sequence. Both 14 and 15 rad/s
complete, but 13 rad/s reduces slide to `0.741733 m`, p95 root jerk to
`865.941196 m/s^3`, and retains passing lateral progress at `0.490994`. The
12.5 rad/s route also completes and reduces slide further to `0.726178 m`, but
raises p95 jerk to `875.710489 m/s^3`; 12 rad/s fails. The retained default is
therefore 13 rad/s: it has a 0.5 rad/s margin over the measured transition
threshold and the best combined slide/jerk result. Its independent full-sole
audit again has `-0.024954 m` minimum clearance and zero samples below
`-0.025 m`.

Allowing 30, 60, or 90 degrees of contact-frame yaw does not solve the on-stair
180-degree turn. All three variants retain the exact frame-285 trajectory and
failure boundary, with stance-height accounting for the majority of rejected
candidates. The useful yaw fit remains capped at 15 degrees; the unresolved
turn is not a rigid-placement-angle problem.

Expanding the exact rescue shortlist from 512 to 2,048 also fails to solve that
turn and regresses its boundary from frame 285 to frame 202 by selecting a
different locally valid rescue. All 2,048 terminal candidates are still
invalid: 853 fail stance height, 544 sole penetration, 453 landing edge margin,
and 198 landing height. Search depth alone is therefore rejected. The local
choice is non-monotonic and needs either entry-state reachability or successor
feasibility, not a larger one-step shortlist.

The lateral crossing remains asymmetric. The mirrored right-to-left route is
bit-for-bit identical with 16 and 20 rad/s output bounds, stops at frame 266,
and reaches only 0.555 commanded lateral progress. Its natural peak joint speed
is 10.513 rad/s, proving that this boundary is not caused by the output-speed
cap. `side-exit-upper-left` and `diagonal-down-left` both stop at frame 282;
the former reaches only 0.252 exit progress and the latter moves against the
command at -0.199 progress. The riser reversal emits all 390 frames but reaches
only 0.051 reverse progress. These are distinct command/contact reachability
failures rather than variants of the solved projected-touchdown snap.

The wider representative slice adds one genuine success. At the 16 rad/s
experimental bound, `side-mount-left` emits all 370 frames, passes both
mount/continue-up segments, and has zero ankle penetration. Total stance slide
is still 0.552 m and the montage remains crouched, so this is a reachability
baseline rather than finished visual quality. An independent full-sole audit
finds a -0.024954 m minimum and zero samples below -0.025 m.

The same slice exposes two ranking defects. `diagonal-up-left` reaches frame
179 and `diagonal-down-right` reaches frame 282 before all 512 rescue candidates
fail; stance height accounts for 396 and 469 rejections respectively. The
terrain-profile score averaged support-height fit across the full chunk, so a
candidate with the wrong entry foot-height split could outrank an entry-valid
candidate. An explicit contact-relative entry height-split score is rejected:
it leaves both diagonal failures and the frame-303 90-degree-turn dead end
unchanged, including the final 445 stance-height rejections. Separately,
`turn-45-lower-left` emits all
290 frames but finishes with 0.510 rad heading error: its 30-degree
sign-consistency gate switches off before the 0.35 rad route contract is met.
The 15-degree follow-up is bit-for-bit identical and is rejected. The gate is
applied to bounded yaw change within each search horizon, not to total live
heading error, so even a 0.510 rad residual can produce a sub-threshold local
target. A 0.10 rad diagnostic improves final heading error from 0.510 to 0.432
rad and slide from 0.528 to 0.416 m, but still fails both heading and lateral
drift while reducing pivot progress from 0.469 to 0.366. It remains an
unqualified diagnostic pending a clean lower-threshold comparison.

At the retained 13 rad/s bound, `turn-90-middle-left` reaches frame 303 with
passing 0.495 pivot progress and only 0.128 rad heading error, then exhausts its
successors; 445 of the final 512 exact candidates fail stance height. This is
not a turning-direction failure like the 45-degree threshold issue. It is a
post-turn contact-state dead end. The rejected entry-height term does not alter
its path or failure boundary.

The original `riser-stop-restart` contract produces a false positive. It emits
all 290 frames, safely restarts with 0.526 progress, and has zero full-sole
samples below -0.025 m, but visual inspection shows the matcher finishing a
high-knee chunk throughout most of the 30-frame stop window. Root drift during
the nominal stop is 0.202 m. The benchmark now measures maximum displacement
from the start of every zero-velocity segment and caps it at 0.05 m; under that
corrected contract the route fails `segment:stop-at-riser:stopped-drift`. This
is evidence for a short stop-specific horizon or safe touchdown-and-hold path,
not a reason to report the old matrix flag as success.

A transactional double-support hold does not solve that benchmark. The stop
arrives 13 frames into a 70-frame source chunk, and its remaining source frames
contain no double-support boundary during the entire 30-frame stop command.
The experiment is metric-identical at 0.202 m stopped drift and is removed.
Stopping therefore requires an interruptible landing retarget or a shorter
contact chunk; freezing at an already-safe boundary is insufficient.

Relaxing the existing safe-runway preference is also rejected. Increasing its
total/outcome cost budgets from 1.0/0.05 to 8.0/2.0 and then 100/100 leaves
`diagonal-down-right` bit-for-bit identical: both variants stop at frame 282
with the same negative -0.507 progress and 469 stance-height rejections. A
same-clip safe runway was already preferred wherever one existed; the missing
post-command contact transition is not hidden below the preference budget.
The temporary CLI controls are removed.

Validating the emitted same-source successor before assigning runway
preference is also rejected. On `side-mount-right` it moves the failure from
frame 266 to frame 230, reduces mount progress from 0.085 to zero, and never
engages the stair. Requiring a safe same-source continuation is not equivalent
to general successor reachability because a valid next chunk may switch clips.
The approximation cannot be promoted to a hard preference condition.

The retained corpus has 760 clips, while the authenticated GRAIL inventory
contains 13,957 records. A full corpus is being built in a separate artifact
for a coverage comparison. It does not replace the qualified corpus unless it
improves the same adversarial route contracts and visual checks.

This diagnosis is consistent with the newer Terrain Consistent
Reference-Guided RL result: its terrain adaptation projects desired footsteps
onto valid footholds and adjusts swing-foot and center-of-mass trajectories as
one synthesis problem, rather than repairing only a root path or only an ankle
after selection: https://arxiv.org/abs/2605.15517. Together with TCRS, this
supports retaining exact contact-aware ranking now while treating coupled
foothold/swing/root synthesis as the next representation-level step.

The clean `0.05 rad` local turn-gate comparison is rejected. Like the earlier
`0.10 rad` diagnostic it completes all 290 frames, but it still fails final
heading at `0.431863 rad` and lateral drift while reducing pivot progress from
`0.46888` to `0.365713`. The experimental CLI override is removed; a smaller
per-horizon sign gate does not control the accumulated live heading error.

Softly ranking candidates against a stable two-foothold query plan is also
rejected. It changes `side-mount-right` from the retained frame-266 failure
with `0.084705` mount progress to a frame-232 failure with zero progress. More
decisively, it changes the passing 370-frame `side-mount-left` control into a
frame-229 failure with zero progress. The selected early foothold is locally
stable but leaves no feasible successor. Both handedness results reproduce the
local-foothold-versus-reachability trap; the implementation is removed.

The next structural ablation exposes supported single-support source entries
while retaining exact support-pattern gating and the complete emitted sole,
landing, edge, and contact checks. The default inventory remains restricted to
stable entries. This tests whether the present double-support-only inventory,
rather than another scalar cost, causes delayed stops and freezes by making an
active half-step impossible to replace with a phase-compatible remainder.

Unrestricted single-support entry ranking is rejected. It moves
`side-mount-right` from frame 266 to 300, increases mount-direction progress
from `0.084705` to `0.416032`, and reduces slide from `0.376412 m` to
`0.150287 m`, but never establishes an elevated contact and stops at the first
riser. The montage confirms a cleaner approach followed by an exact-contact
dead end. More importantly, it regresses both established controls:
`side-mount-left` and `cross-tread-left-to-right` stop at the identical frame
153 with no terrain engagement, instead of completing 370 and 410 frames.
Immediate validity of an arbitrary mid-step replacement is therefore not
enough.

The narrowed follow-up exposes those entries only for a zero-speed command.
Every moving search prefilters back to the original double-support records
before the exact shortlist, preserving their relative order; the broader set
is visible only when attempting to stop mid-step. This directly isolates the
`0.202 m` stopped-drift failure without changing moving command selection.

The first narrowed build still regressed both moving controls at frame 227.
The selection mask was correct, but inventory construction assigned newly
added single-support rows before later stable skills claimed their canonical
rows; overlapping row ownership therefore changed the original stable
inventory underneath the mask. The corrected builder now constructs and
freezes the complete default inventory first, asserts its owners through unit
coverage, and only then assigns previously unowned stop rows. The stop and
controls are rerun from this behavior-preserving inventory.

That owner-preserving pass revealed that canonical pre-skill entries are not
all double support; filtering by contact label still removed intentional
baseline records. The final isolation therefore carries explicit canonical
row provenance from inventory construction into every horizon record. Moving
commands accept that exact original set, while stopped commands may also use
the newly added rows. Contact state is no longer used as a proxy for record
provenance.

The explicit-provenance version is also rejected and removed. The stop route
regresses from 290 emitted frames to a frame-190 exact-contact failure, and the
passing cross-tread control regresses from 410 to frame 253. Side-mount-left
still completes its 370-frame control, so this is not a loader or general
execution failure. Enlarging the inventory changes the globally ranked
shortlist even when non-canonical moving records are filtered afterward; more
importantly, the added stop records still do not provide a valid landing-and-
hold successor. This closes the mid-step-entry experiment rather than leaving
an unqualified mode in the runner.

The full authenticated corpus build completed with 12,646 accepted clips and
1,312 rejected records, compared with 760 clips in the curated corpus. The
next coverage ablation searches that complete horizon inventory with
same-skill continuation preference disabled. The source-neighborhood
exclusion remains only a ±20-frame anti-self-match rule; candidate ranking is
otherwise global across all clips. Outcomes remain expressed in the
character-heading frame so clips are reusable at arbitrary world poses.

The first literal full-corpus query exhausts a 46 GB GPU before emitting frame
one. Ranking itself fits, but the emitted-terrain prefilter expands every
globally ranked record's support, sole, and landing traces before applying the
64-candidate exact-validation budget; one attempted allocation alone is
2.185 GB after the device is nearly full. This is a layered-retrieval defect,
not evidence against global coverage. The prefilter is changed to consume
stable ranked batches and stop as soon as the exact-validation shortlist is
full. Unit coverage requires that a one-record shortlist never evaluates a
second ranked prefilter batch. Full-corpus qualification is rerun only through
that bounded retrieval path.

Here “global sequence search” means more than a global candidate inventory.
The retained matcher still commits the cheapest one-step exact-valid chunk,
which can end at a state with no valid next chunk. The proposed next layer
keeps a bounded beam of exact-valid first chunks, rolls each to its emitted
terminal pose, and ranks the first edge by its own cost plus the best valid
successor cost. This directly tests the repeated locally-valid/global-dead-end
diagnosis without changing the terrain coordinate convention.

Disabling coherent continuation is not a valid proxy for global search. On the
760-clip corpus, that ablation regresses `turn-90-middle-left` from the retained
frame-303 boundary to frame 122 with only three chunks; 473 of the final 512
exact candidates fail stance height. The global mode therefore preserves a
safe same-skill continuation under an unchanged command. At a required
boundary search it evaluates a bounded set of exact-valid first edges, rolls
each through the retained output filter to obtain its terminal pose, performs
a general cross-clip successor search from that pose, and minimizes first-edge
plus successor cost. The second edge uses the exact source terrain/contact
contract as a feasibility approximation; the committed first edge still uses
the full emitted foot/sole validator.

Bounded layered retrieval makes the 12,646-clip corpus executable, but the
coverage result is negative. With continuation disabled, the full corpus fails
`turn-90-middle-left` at frame 121 with four chunks, compared with frame 122
and three chunks for the 760-clip corpus. At the terminal search, 235 of the
512 exact candidates fail landing edge margin, 145 stance height, 91 landing
height, and 41 sole penetration. The extra corpus therefore changes the local
choice but does not create a useful route; the qualified dataset is not
replaced. The layered prefilter remains a general scalability correction.

Four-wide two-edge sequence scoring with turn-boundary replanning is also not
sufficient. It reaches the same frame-303 boundary as the retained greedy
turn, with seven committed chunks, then finds zero emitted-valid first edges
among the 512 rescue records: 440 fail stance height, 55 sole penetration, 14
landing height, and 3 landing edge margin. The approximate second edge judged
earlier endpoints to have source-compatible successors, but those successors
did not preserve emitted foot/sole feasibility when reached. A useful global
planner must therefore carry the output-filter/contact state across both
edges, not only roll the first edge exactly and validate the second against
its source trace.

The final global ablation does carry that state. For each exact-valid first
edge, it generates the second edge from the first edge's filtered terminal
pose, previews both raw edge sequences through one isolated foot-lock/IK state
machine, discards the filtered prefix, and applies the full emitted
foot/sole/contact validator to the second edge. Beam width is fixed at two for
the first qualification; second-edge exact shortlist widths 8 and 16 run in
parallel. The prefix contract has direct unit coverage proving that prefix
frames advance filter state without entering the second-edge validation
window.

That exact two-edge version is rejected as well. With beam width two and eight
exact emitted successor candidates per branch, it still reaches frame 303 and
then has zero emitted-valid outgoing edges: 443 fail stance height, 55 sole
penetration, 13 landing height, and 1 landing edge margin. A 16-successor run
is stopped after the 8-successor result because deeper successor evaluation
cannot help at a node whose current-edge rescue contains no valid action. The
global CLI, chained-preview API, and sequence-selection surface are removed;
only the independently qualified batched prefilter remains. The experiment
shows that lookahead over the current double-support chunk graph cannot repair
this turn. The next representation must make a different action available
before frame 303: shorter interruptible contact actions or coupled
foothold/swing/root synthesis.
