# G1 IK Root-Reach Preconditioner Design

## Objective

Allow an exact recorded-contact sole target to remain physically reachable by
moving only the reversible IK candidate root in world Y by the smallest
dual-certified amount: the nearest binary32 delta which is both admitted by
the baseline analytic reach interval and authenticated against the adjusted
production target. Preserve recorded contact timing, exact terrain locks,
independent travel and heading, IK-off byte identity, and the existing atomic
safe-stop transaction.

## Observed Failure

The first IK-on frame of `grail-curb-low:curb-forward` records both feet in
contact. The exact flat-ground targets are valid, but the unadjusted Takara
pose has both legs almost fully extended:

- left target requires approximately `-0.0087 m` of root Y correction;
- right target requires approximately `-0.0157 m`;
- without correction, the left solver requests `0.502262 m` from a current
  effective maximum of `0.493778 m` and emits `target-unreachable`.

This is systemic rather than curb-specific. Across the flat Takara range,
11,430 of 17,136 recorded-contact foot rows (66.7 percent) place the exact
flat 5 mm sole target outside the current reach shell before root
preconditioning.

A read-only debugger probe translated only the IK candidate root by
`-0.016 m`. The formerly failing live frame then accepted with IK applied,
both feet reachable, left/right contact residuals of approximately
`0.000174 m` and `0.000134 m`, route progress, and balanced cleanup. No
production source or running visualizer was changed by that probe.

Those residuals are historical feasibility evidence from a disposable binary
which predates the Task 1 strict projection kernel; they are not the Task 2
acceptance oracle. Replaying the embedded frame through the current strict
kernel selects binary32 delta `0xbc80e8f0`
(`-0.01573607325553894 m`). The named production solve then reports precise
physical-sole residuals of approximately `0.0020950164 m` left and
`0.0001104668 m` right. Both are reachable, correction-unlimited, and satisfy
the unchanged exact promoted `<= 0.005f` production convergence rule.

## Architectural Boundary

The correction belongs to a frame-level, two-foot preconditioner at the start
of the reversible IK transaction. It does not mutate `support_frame_state`.
Persistent terrain support must remain identical between IK-on and IK-off so
Gate L can continue comparing matching, query, terrain, support, simulation
XZ, and command intent bit for bit.

The production order is:

1. Build the ordinary terrain-support baseline exactly as today.
2. Run checked FK and materialize both exact `G1FootTarget` values.
3. Complete the existing blocked-footprint and unavailable-landing
   classification. If either requests a begin-time stop, retain the canonical
   inactive plan and do not invoke root planning.
4. For recorded-contact feet, derive the same surface-aligned physical-sole
   position target that the named solver will consume.
5. Build and intersect their feasible root-Y reach intervals.
6. Select the baseline-analytic-admitted binary32 delta closest to positive
   zero, bounded to `[-0.05 m, +0.05 m]`, and authenticate it with the
   adjusted production target and reach projection.
7. Before publishing either scratch array, use the strict checked apply helper
   to compute the adjusted local root Y into a scalar candidate. Copy the
   untouched baseline arrays only after that succeeds, then assign the scalar
   once to only `scratch_positions(G1_Simulation).y`.
8. Run the existing fixed left-then-right bounded position/orientation IK,
   defensive clearance, final FK, pose certificate, and outer atomic commit.
   If both foot stages succeed but the retained recorded-contact plan is the
   active/non-common/canonical-zero bounded-rejection form, terminate after
   foot 1 with the existing frame-level `target-unreachable` safe stop. Do
   not enter final FK and do not mark either successful foot as failed.

The preconditioner receives no command, trajectory, travel-direction, or
heading owner. Root X/Z, every local rotation, the accepted support baseline,
and all matcher inputs are outside its mutable output.

## Interfaces and Provenance

Add a small value-only reach-plan result with these semantics:

- whether recorded-contact planning was active;
- whether a common bounded interval was found;
- whether a nonzero root adjustment was applied; and
- the exact binary32 root-Y delta.

The plan is computed before its destination pose is changed. On success, the
transaction result retains the root-Y delta so validators can authenticate
the adjustment against the candidate pose. It is not persistent controller
state and is recomputed from each accepted support-retargeted baseline.

Accepted provenance is authenticated at the operations which actually own
the bits. The strict checked apply helper recomputes expected local
`G1_Simulation.y` from the untouched support-retargeted local root and the
retained plan. The accepted IK local pose must match that value, retain root
X/Z bit for bit, and retain every non-root local position bit for bit. The
support-retargeted and rendered Hips diagnostics are then independently bound
to `global_bone_positions(G1_Hips).y` and
`ik_global_bone_positions(G1_Hips).y`, respectively, after both global arrays
have passed their existing checked-FK ownership gates.

Do not authenticate the plan by requiring exact binary32 equality between
rendered Hips Y and support-retargeted Hips Y plus the delta. Checked FK forms
the rendered value as the adjusted local root followed by the parent/child
addition, while that diagnostic expression adds the delta after the baseline
parent/child addition. Those differently parenthesized binary32 operations
need not have identical bits.

Refactor, rather than duplicate, the two pieces of math shared with production
IK:

- physical-sole target rotation/contact-origin/initial-ankle derivation; and
- the checked effective two-bone reach shell used by target projection.

Each recorded foot contributes the root-Y values for which its exact ankle
target remains inside that effective shell. The fixed-size interval
intersection handles both feet without order dependence. The selected float
is the dual-certified member nearest zero: baseline analytic admission is a
required proof domain, then adjusted production-target authentication is the
second proof. Boundary rounding is moved toward the analytic interval
interior and then checked again through the real projection. Revalidation
must apply the candidate to a fixed 31-bone copy of the local pose and rerun
the same checked strict FK. It must then rederive the physical ankle target
from the adjusted contact origin/rotation and ankle origin, using the original
desired sole center/normal, before projecting the adjusted hip/knee/ankle
points. Neither translating previously computed global points nor projecting
the baseline-derived ankle target is numerically equivalent to the named
production solver, so neither may authenticate publication. At most 32
adjacent boundary floats are tested in total across all intervals, chosen by
one deterministic nearest-first frontier; this is a global budget, not a
per-interval allowance. A result is publishable only when every contributing
foot reports its rederived exact production target as reachable.

A finite candidate-local quantization case is distinct from malformed
arithmetic. A nonzero candidate delta can round back to the unchanged local
root Y even though at least one signed 5 cm cap remains representable. Task 2
classifies that candidate as rejected and publishes the ordinary valid
active/non-common/unapplied positive-zero plan. The two named foot stages
still execute in fixed order. If neither independently rejects, the frame
converts the retained planner disposition to `target-unreachable` only after
foot 1; this is a planner-level terminal form with two independently
successful foot transcripts, not a forged foot failure.

The search never probes outside the baseline analytic intersection merely
because adjusted binary32 recomputation makes a nearer float production-
reachable. The exact translated-level regression demonstrates this boundary:
the adjusted production target accepts `0xbc80e8e7` through `0xbc80e8f5`, but
the baseline analytic interval first admits `0xbc80e8f5`; therefore
`0xbc80e8f5` is the nearest dual-certified result. The intentionally excluded
production-valid gap is about 26 nm in root Y and does not justify changing
the proof domain or the named solver.

No runtime option and no log column is added. The difference between existing
`support_retargeted_hips_y` and `ik_adjusted_hips_y` remains approximate live
quality evidence for the candidate-only correction; it is not the exact plan
oracle. The retained result, strict local-root relation, authoritative FK
bindings, equality, and logical state digests provide exact internal
provenance.

## Failure and Rollback

- With IK disabled, no recorded contact, or an already reachable set of
  targets, the preconditioner is an exact no-op; IK-off pose and support bytes
  remain unchanged.
- If the two feet have no common interval, the nearest interval lies outside
  the 5 cm bound, or the final production projection does not authenticate
  the selected float, no heuristic target or contact change is made. The
  ordinary per-foot stages run left then right. A rejecting foot emits the
  existing finite `target-unreachable` diagnostic. If both feet succeed while
  the retained plan is active/non-common/unapplied positive zero, the frame
  emits the same diagnostic after foot 1 without changing either successful
  foot result.
- Angular correction limits, sole residuals, orientation alignment, swing
  clearance, and whole-pose clearance remain authoritative after root
  planning. The preconditioner cannot waive them.
- Nonfinite input, malformed topology, arithmetic failure, or aliasing is a
  checked global error with no output assignment.
- A later finite or global failure discards the entire working pose, root
  delta, lock/history state, timers, and route progress through the existing
  frame transaction.
- An outer footprint rejection occurs before IK begin and therefore publishes
  no attempted IK plan. A direct blocked or unavailable-landing rejection at
  the begin checkpoint retains the canonical inactive positive-zero plan.
  Rejections after foot 0 or foot 1 retain a valid plan whose `active` bit is
  exactly the OR of the recorded-contact bits; active/non-common/unapplied
  positive zero is the allowed infeasible, out-of-cap, or finite
  candidate-quantization form. The after-foot-1 planner terminal authenticates
  `target-unreachable`, that exact plan, two successful foot transcripts,
  `next_foot == 2`, and a frame safe stop. An accepted contact-active frame
  additionally requires a common interval. A later finish or pose-certificate
  failure cannot publish the working plan into the accepted controller state.

## Tests

Write failures before production changes and cover:

1. The authentic two-contact geometry corresponding to the measured live
   frame, including the nearest dual-certified common delta and exact
   production convergence for both final physical-sole residuals; plus an
   exact translated-level case distinguishing baseline analytic admission
   from adjusted production-target reachability.
2. One recorded contact plus one swing foot, and two contacts whose interval
   order is reversed, proving order independence.
3. Flat, ramp, and cross-slope normals using the shared physical-sole target
   derivation.
4. An established world-space lock across a root level change.
5. Already-reachable, no-contact, and IK-off exact byte no-ops, including
   positive and negative zero inputs where applicable.
6. Empty intersections, corrections just inside/at/outside 5 cm, minimum-shell
   conflicts, one-ULP boundary cases, and production-projection
   revalidation.
7. Invalid input, aliasing, and arithmetic rollback.
8. Full frame rejection/acceptance ownership, including immutable command and
   heading snapshots, strict local-root application, root X/Z and non-root
   position identity, and checkpoint-specific plan forms. Include the
   candidate-quantization terminal whose two foot solves succeed but whose
   active/non-common/canonical-zero plan finite-rejects atomically at the
   second-foot stage; mutate common/applied/delta, stop reason, either foot
   success transcript, cursor, and frame safe-stop evidence independently.
9. Authoritative support/rendered Hips FK bindings, including the fixed
   multilevel binary32 parenthesization fixture local root
   `a=0x3f000000`, local Hips Y `b=0x3f000001`, and Task 2 delta
   `d=0xbc80e8f0`: checked FK `(a+d)+b` is `0x3f7bf8ba`, while the
   reassociated `(a+b)+d` is `0x3f7bf8b8`. This proves recomputing
   `support Hips + delta` is not the accepted exact oracle.
10. All independent logical hashes, controller-state validation, controller
   logging digest parity, dirty-copy ownership, and plan-field forgeries.
11. Strict caller, optimized `-ffast-math` caller linked to the strict kernel,
   strict/fast parity records, ASan/UBSan/float sanitizers, and production
   no-seam/no-main controller builds and negative runner compilation.
12. The injected pose-certificate global rollback uses a genuinely
    planner-common all-contact source. Starting from the symmetric source
    support word `0xbf08efbb`, move exactly one binary32 ULP toward positive
    infinity to `0xbf08efba`; freeze the active/common/unapplied positive-zero
    plan and both successful foot transcripts before proving the working plan
    cannot replace the prior accepted plan.

Task 2 also exposes a fixed-size planner audit only when
`G1_IK_ENABLE_TEST_SEAMS` is defined. The audited wrapper calls the same
private planner implementation as the ordinary wrapper and transactionally
records the sorted cursors' initial materialized binary32 bits plus every
global attempt's cursor index, candidate bits, and rejected/accepted status.
The trace holds at most four cursors and exactly the planner-wide maximum of
32 attempts. It is used to prove the authentic inward sequence, a real
multi-cursor switch, one-cursor exhaustion, and a symmetric two-cursor trace
which alternates cursors for exactly 32 total rejections. The combination
rejects both a sequential/per-interval scheduler and a 33/64/128-attempt
budget regression; audit capacity is checked before every write so a
hypothetical 33rd attempt fails transactionally rather than truncating the
trace. Invalid execution publishes neither plan nor trace. A root-reach
object compiled without the macro exposes no audit declaration or symbol and
still runs the complete ordinary non-seam test surface.

After focused tests, rebuild a disposable controller and run:

- the exact low-curb regression first for 1, then 32, then 800 frames;
- inherited curb, stair, multilevel, ramp, slope, lateral, diagonal, and
  safe-stop Gate E cases; and
- the complete 30-cell Gate L IK-off/IK-on matrix.

IK-off logs must retain every immutable Task-3/Gate-L oracle field. IK-on runs
must complete without footprint/IK safe stops, respect the 5 cm rendered-Hips
step gate, retain exact heading bits, and meet all existing physical-clearance
and heading-quality thresholds.

## Scope

This change does not alter motion data, contact derivation, terrain artifacts,
surface locks, reach/correction thresholds, the 41-entry swing ladder,
support smoothing, the 31-dimensional matcher, route logic, logging schema,
or visualizer process. It does not merge terrain packs. The existing
visualizer remains untouched until Task 8 certifies and atomically replaces
the complete runtime. G1 mesh integration remains the final optional task.
