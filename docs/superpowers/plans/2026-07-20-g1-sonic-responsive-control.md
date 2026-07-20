# G1 SONIC Responsive Motion-Matching Control Implementation Plan

**Authors:** Claude Plan A, selected and tightened by the Codex controller
**Status:** Selected implementation plan
**Approved contract:** `docs/superpowers/specs/2026-07-20-g1-sonic-responsive-control-design.md`
**Design base:** `7113ebf1673638ef84dc4a783af5fb7e138f2a63`

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> or `superpowers:executing-plans` to implement this plan task-by-task once the
> controller unfreezes source changes. Steps use checkbox (`- [ ]`) syntax for
> tracking. All formal verification gates below (`gate-001` and the RED/GREEN
> commands) are **controller-owned**; this plan states the commands and expected
> outcomes but does not run them.

**Interpreter idiom (as used by sibling plans):**
`/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.<module> -v`,
run from the repository root.

> Scope discipline for this plan. It optimizes for the **smallest safe transport
> and protocol change** that could deliver responsive control. It proves or
> falsifies the design's proposed short committed prefix strictly from the code
> that exists at HEAD, and turns every genuine uncertainty into a named,
> executable probe. It does not modify SONIC policy code or weights, and it
> keeps physics paused whenever a safe target prefix is unavailable.

---

## 1. Root-cause analysis from exact code

### 1.1 Where the 0.8–1.2 s latency comes from (observed facts)

The interactive driver is `sonic/python/mm_sonic/manual_demo.py`. The registered
timing constants are exact:

- `manual_demo.py:84` `_PRELOAD_CHUNKS = 4`
- `manual_demo.py:85` `_CHUNK_DURATION_S = 0.4`
- `manual_demo.py:811-812` — interactive default is `preload_chunks = 2` (`namespace.preload_chunks = 2 if interactive else _PRELOAD_CHUNKS`).
- `manual_demo.py:92-103` `_validated_preload_chunks` clamps the preload depth to the integer range `1..4`.

The preload loop commits `preload_chunks` full chunks *before* any operator
command is consumed:

- `manual_demo.py:560-561` — `for index in range(preload_chunks): generate_and_publish(_stand_command(index), wait=True)`.

Each committed chunk is exactly **20 target rows at 50 Hz = 0.4 s**:

- `timeline.py:31` `_TARGET_ROWS = 20`; `timeline.py:32` `_TARGET_RATE_HZ = 50.0`.
- `timeline.py:487-489` `frame_index = np.arange(first_index, first_index + _TARGET_ROWS, ...)`.
- `manual_demo.py:581` `steps_per_chunk = round(_CHUNK_DURATION_S / simulator.sim_dt)` — one operator boundary releases exactly one 0.4 s chunk of physics via `gate.release_steps(steps_per_chunk)` (`manual_demo.py:373`, `595`, `683`).

**Consequence (observed → derived).** At an operator boundary the loop samples
the latest command, generates one new 0.4 s chunk, appends it to the tail of the
committed timeline, and releases exactly the *oldest* not-yet-consumed 0.4 s of
physics. A freshly sampled command therefore first appears in simulation only
after the already-queued `preload_chunks` chunks drain: `preload_chunks × 0.4 s`.
At the interactive default of 2 that is **0.8 s**; at 4 it is 1.6 s. The
demo prints this exact figure as `presents_in=... s` / `lookahead latency`
(`manual_demo.py:381`, `621`). This matches the design's observed 0.8–1.2 s
window and confirms the design's claim that the latency is **intentional
queueing, not X11 input latency** (`operator_x11.py`/`ContinuousControlLoop`
mailbox sampling in `manual_demo.py:338` samples the *latest* mailbox state at
the boundary, so input freshness is not the bottleneck).

### 1.2 The append-only, fixed-width transport (observed facts)

The transport commits fixed-size, append-only chunks and cannot revoke a
published chunk:

- **Fixed 20-frame target chunk.** `timeline.py:67-168` `TargetChunk.__post_init__`
  hard-validates every array to shape `(_TARGET_ROWS, …)` = `(20, …)`. There is
  no code path that produces a target chunk of any other length.
- **Fixed `source_intervals = 10`.** Enforced independently in three places:
  `coordinator.py:596-597` (`source_intervals must equal the registered value 10`),
  `process.py:1113-1114` (`MM source_intervals must equal 10`),
  `sonic/cpp/mm_chunk_server.cpp:1458` (`"supported_source_intervals":[10]`) and
  `:1681` (`"source_intervals":10`). The source grid is 11 boundaries at 25 Hz
  (`resample.py:21-24` `_SOURCE_RATE_HZ=25`, `_SOURCE_INTERVALS=10`,
  `_SOURCE_BOUNDARIES=11`) resampled to exactly 20 target rows
  (`resample.py:282` `_TARGET_ROWS = 20`; docstring `resample.py:288`
  "Map and resample 11 source boundaries into exactly 20 new rows").
- **Append-only commit.** `timeline.py:536-569` `TargetTimeline.commit`
  concatenates the new 20 rows onto the canonical buffer and advances
  `_last_frame_index`; there is no truncation or overwrite API. The only
  pre-commit reversal is `TargetTimeline.abort` (`timeline.py:571-580`), which
  discards a *pending, not-yet-committed* candidate only.
- **Exactly one pending candidate.** `timeline.py:458-463` — `prepare` raises if
  a candidate is already pending. The coordinator preserves this: it owns one
  latched command and one in-flight candidate (`coordinator.py:963-972`,
  `1038-1051`).
- **Merged, monotone stream on the SONIC side.** `process.py:2554-2623`
  `wait_for_stream_processing` authenticates the pinned
  `[StreamedMotionMerger] Processing … / Merged motion:` transcript with
  `merged_count == global_start + frame_count` and `frame_step=1`
  (`process.py:2576`, `2585-2591`). The reference SONIC consumer
  (`g1_deploy_onnx_ref`) appends incoming frames to a growing motion; there is
  no "replace tail" message in the pinned transcript vocabulary
  (`process.py:74-75`). Published frames are therefore irrevocable once merged.
- **Reference-output invariant.** `reference.py:177-180` requires
  `(canonical.count - 1) % 20 == 0` ("frame zero plus complete 20-row chunks").
  A sub-20-frame committed unit would violate this official-reference invariant.

### 1.3 The one-chunk stop failure (design fact, mechanism from code)

The design records that a prior one-chunk configuration (`preload_chunks = 1`,
0.4 s lookahead) failed the existing stopping-displacement threshold. The
mechanism is visible in the transport: because a committed 20-frame chunk is
irrevocable (§1.2), a "stop" command sampled at a boundary cannot cancel the
0.4 s of forward motion already committed in the single queued chunk; the robot
continues to translate for one full chunk after the operator releases the key.
Reducing to one chunk lowers latency to 0.4 s but does not remove the last
irrevocable chunk, which is what the stop-displacement gate measures.
**Falsification consequence:** any design that keeps the 20-frame atomic chunk
has a hard floor of one irrevocable 0.4 s chunk of committed future motion.

### 1.4 SONIC future-reference requirement (observed evidence + one probe)

Two independent pieces of code evidence show SONIC consumes reference frames
*ahead of* the physics it drives, so physics must never advance past the last
appended reference frame:

- `manual_demo.py:438-444` (verbatim comment): the interactive producer keeps
  MuJoCo at wall-clock pace *"so the policy/control threads can consume LowState
  and the newly appended reference before physics advances past them,"* and
  explicitly contrasts this with the "fully preloaded evidence runner" that "can
  safely use unpaced physics; this live producer cannot."
- `zmq_v1.py:52` and `:281-283` — the pose header carries a `catch_up` `u8`
  flag; baseline ZMQ v1 requires `catch_up == 0` (`"baseline ZMQ v1 catch_up
  must remain zero"`). A non-zero `catch_up` is the transport's own signal that
  the consumer is behind the appended reference. This is the observable
  handle for "reference lookahead consumed."

**Inference (marked).** The exact number of future reference frames the
unmodified `g1_deploy_onnx_ref` policy requires between the current physics
frame and the reference tail is **not derivable from the Python side**; it lives
in the pinned GEAR binary. This is a genuine uncertainty and becomes
**Probe P0** (§4). Until P0 measures it, no sub-200 ms claim is admissible,
because the committed prefix must always contain at least that many
future-reference frames ahead of the frame currently being presented.

---

## 2. Verdict on the proposed short committed prefix

The design's recommended approach exposes "only the shortest prefix that the
existing target transport and SONIC consumer can safely accept." Its
illustrative **100 ms** prefix is tightened here to the shortest unit that is
both source-grid aligned and capable of meeting the under-200 ms contract:
**80 ms** (two 25 Hz source intervals, four 50 Hz target frames).

**Falsified as literally stated, against HEAD.** The smallest atomic unit the
*existing* transport can carry is one 20-frame target chunk = **0.4 s**
(§1.2). 100 ms = 5 target frames is not representable without editing the
frozen protocol: `timeline.py` `_TARGET_ROWS = 20` and the `TargetChunk` shape
validators, the `(count-1) % 20` reference invariant, and the resampler's
"11→20" contract would all have to change. The design itself anticipates this
outcome and instructs: *"If the existing SONIC stream requires a longer atomic
unit, use the shortest proven unit and report the resulting lower bound rather
than silently claiming 200 milliseconds."* This plan honours that instruction.

**Therefore the honest lower bound for presentation latency on the unmodified
transport is one irrevocable 0.4 s chunk plus MM generation time**, i.e.
`presentation_latency ≥ 0.4 s + t_generate` when preload is reduced to a single
committed chunk. That is above the 200 ms target. Claiming ≤ 200 ms requires a
**protocol change** to the 20-frame target chunk *and* proof (Probe P0) that
SONIC's future-reference requirement fits inside the smaller prefix.

**Smallest-safe-transport recommendation (this plan A).** Deliver responsiveness
in two staged, independently gated commits, largest safety margin first:

- **Stage R1 (transport-preserving).** Reduce irrevocable lookahead to exactly
  one committed 0.4 s chunk, add the boundary trace and physics-pause-on-miss
  behaviour, and prove direction correctness and SONIC tracking. This is the
  smallest change that measurably improves responsiveness (0.8 s → 0.4 s at the
  interactive default) **without touching the wire protocol**. It cannot meet
  200 ms and does not claim to; it reports the measured 0.4 s + generation
  floor and re-checks the stop-displacement gate that the prior one-chunk
  attempt failed.
- **Stage R2 (minimal protocol change), gated behind Probe P0 and P4.** The
  shortest source-grid-aligned unit is exactly **2 source intervals = 80 ms =
  4 target frames**. Proceed only if P0 shows unmodified SONIC needs no more
  than 4 future frames and the measured generation-plus-publication p95 is no
  more than 120 ms. Introduce registered `source_intervals in {2, 10}` and an
  additive four-frame `target-chunk/v2`; leave `target-chunk/v1` untouched as
  rollback. If either bound fails, abandon Stage R2 and record
  "sub-200 ms infeasible on this policy/transport" rather than selecting an
  arbitrary width that cannot align with the 25 Hz MM source grid.

This ordering guarantees the first commit is safe and useful even if the sub-200
ms goal proves infeasible, and it never claims a latency the transport cannot
honour.

---

## 3. Separation of concerns: direction vs. tracking (test gap analysis)

The design requires proving **Motion-Matching direction correctness** and
**SONIC target-following correctness** independently, for backward and lateral
commands. Current coverage is asymmetric:

- **Command/axis mapping is covered.** `tests/python/test_sonic_holden_control.py:43`
  `test_forward_and_left_match_mujoco_axes` proves the *operator→MuJoCo*
  velocity signs (forward `vx=+0.9`, left `vy=+0.6`). `test_sonic_transform.py`
  covers Holden↔MuJoCo axis/quaternion conversions
  (`test_all_three_holden_unit_axes_map_to_the_mujoco_basis:48`, etc.).
- **Gap 1 — generated MM root direction is unproven.** No test asserts that the
  MM *result* (the generated virtual-root displacement in a committed
  `TargetChunk`) actually moves **backward** or **laterally** with the expected
  signed projection. The timeline stores `virtual_root_position`
  (`timeline.py:79`, `131-136`, `516-518`) but no test projects a
  backward/lateral command's generated displacement onto the requested axis.
- **Gap 2 — SONIC tracking of those targets is unproven per direction.** No
  test separates "MM moved backward" from "SONIC followed the backward target."

These two gaps are exactly the design's **MM-only direction gate** and **flat
dynamic gate**. Plan A closes Gap 1 with a pure, deterministic direction probe
(Probe P1, a fast unit-style gate) and Gap 2 with the controller-owned dynamic
gate (§6), keeping the two error sources distinguishable in the boundary trace.

---

## 4. Executable probes (every uncertainty is a probe)

Each probe is a small, named, executable artifact. Probes P1–P3 are pure and
run under `unittest`; P0 and P4 require the pinned GEAR binary and are executed
by the controller in the dynamic gate environment. All are RED-first.

| Probe | Question it settles | Kind | Blocking for |
|-------|--------------------|------|--------------|
| **P0** | Can unmodified `g1_deploy_onnx_ref` produce healthy control with exactly 4 future frames, measured using sealed 4/8/12/16/20-frame prefixes and `catch_up`/process/control evidence? | Dynamic (GEAR) | Any sub-200 ms claim; Stage R2 |
| **P1** | Does a backward / left / right command produce a generated virtual-root displacement with the correct signed projection and magnitude ≥ registered minimum? | Pure unit | MM-only direction gate; buffer tuning per direction |
| **P2** | With preload reduced to one chunk, does a stale (superseded) candidate get aborted before publication and never merged? | Pure unit (coordinator) | Stage R1 responsiveness |
| **P3** | Does the boundary trace bind exactly one input transition to exactly one presented prefix with monotonic timestamps? | Pure unit | Latency measurement validity |
| **P4** | With one-chunk lookahead, does the stop command keep displacement within the existing stop threshold, and does SONIC stay upright? | Dynamic (GEAR) | Stage R1 acceptance |

**P0 is the falsification hinge.** The controller feeds sealed prefixes of
4, 8, 12, 16, and 20 frames from the same authenticated reference while
physics is gated, then records merge acknowledgement, `catch_up`, first valid
control output, process survival, and upright release. Four frames must pass;
otherwise the source-grid-aligned 80 ms design cannot meet the contract and the
plan records sub-200 ms as infeasible for this policy/transport.

---

## 5. Test-first task breakdown

Task numbering is the intended commit order. Each task states the new test name,
the file, the RED command, the expected failure, the minimal change, and the
GREEN command, followed by checkbox steps for tracking. New production code
lives beside the existing modules; **no existing SONIC policy code or weights
are touched, and no wire message is changed in Stage R1.**

### Task 1: Boundary trace record (Stage R1)

- **New module:** `sonic/python/mm_sonic/boundary_trace.py` (pure dataclass +
  validators, mirroring the immutability idiom of `commands.py:60` and
  `timeline.py:67`).
- **New test file:** `tests/python/test_sonic_boundary_trace.py`.
  - `test_trace_binds_one_transition_to_one_prefix_with_monotonic_stamps`
    (**Probe P3**). Asserts the seven design timestamps
    (`X11 transition → sampled → MM start → MM complete → committed →
    published+ack → physics released → first simulated frame`, design §
    "Boundary trace") are monotone non-decreasing and that
    `key_transition_id → presented_prefix_id` is 1:1.
  - `test_trace_rejects_nonmonotonic_or_unbound_records`.
- **RED command:**
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_boundary_trace -v`
- **Expected RED:** `ModuleNotFoundError: mm_sonic.boundary_trace` (module absent).
- **Minimal change:** add the frozen `BoundaryTrace` dataclass with monotonic
  validation and stable `input_transition_id` / `presented_prefix_id` fields,
  plus the requested numeric fields (requested velocity, requested heading,
  generated virtual-root displacement, published physical-root displacement,
  observed MuJoCo-root displacement).
- **GREEN command:** same as RED; all trace tests pass.
- **Commit boundary 1:** "feat: add responsive boundary trace record."

Checklist:

- [ ] Step 1.1 — Write the two RED tests in `tests/python/test_sonic_boundary_trace.py` and run the RED command; confirm `ModuleNotFoundError: mm_sonic.boundary_trace`.
- [ ] Step 1.2 — Add the frozen `BoundaryTrace` dataclass with the seven-timestamp ordering validator and the five numeric fields; keep it pure with no I/O.
- [ ] Step 1.3 — Run the GREEN command and confirm both trace tests pass with monotonic and 1:1 binding assertions green.
- [ ] Step 1.4 — Commit as commit boundary 1 ("feat: add responsive boundary trace record").

### Task 2: MM direction probe (Stage R1, closes Gap 1)

- **New module:** `sonic/python/mm_sonic/direction_probe.py`. Pure function
  `signed_root_projection(target: TargetChunk, axis_mujoco) -> float` operating
  on `TargetChunk.virtual_root_position` (`timeline.py:79`) and a registered
  minimum-magnitude table for `{forward, backward, left, right}`.
- **New test:** `tests/python/test_sonic_direction_probe.py`
  - `test_backward_left_right_forward_have_expected_signed_root_projection`
    (**Probe P1**). Builds four deterministic `TargetChunk`s from fixed synthetic
    resampled sources (reusing the `test_sonic_timeline.py:96` construction
    idiom), and asserts the signed projection onto the requested axis is
    positive and ≥ the registered minimum, and that the *orthogonal* projection
    is within tolerance.
  - `test_neutral_after_direction_reduces_generated_travel` — a neutral command
    following each direction yields a smaller signed projection than the active
    command (design "MM-only direction gate", neutral clause).
- **RED command:**
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_direction_probe -v`
- **Expected RED:** `ModuleNotFoundError: mm_sonic.direction_probe`.
- **Minimal change:** implement the pure projection + registered-minimum table.
  No dependency on the live MM server; the probe operates on any committed
  `TargetChunk`, so the same function is reused by the dynamic gate to score
  live generations.
- **GREEN command:** same as RED.
- **Commit boundary 2:** "feat: prove MM-generated direction signs from targets."

The synthetic tests validate the scoring oracle only; they are not evidence
that Motion Matching follows a command. Add a controller-owned MM-only probe
that resets the real `mm_chunk_server` to the same authenticated initial state
for forward, backward, left, and right, generates one candidate per reset, and
scores each returned `TargetChunk` with `signed_root_projection`. Require the
expected signed projection and minimum magnitude before any SONIC tracking run.
This live probe is the actual P1 scientific gate.

> Honest note: P1 proves *MM direction* only. It intentionally does **not** prove
> SONIC follows the target; that is the dynamic flat gate (§6). Keeping them in
> separate artifacts is what makes a passed-MM / failed-physics result diagnosable.

Checklist:

- [ ] Step 2.1 — Write the two RED tests in `tests/python/test_sonic_direction_probe.py` using the `test_sonic_timeline.py:96` chunk-construction idiom; run RED and confirm `ModuleNotFoundError: mm_sonic.direction_probe`.
- [ ] Step 2.2 — Implement `signed_root_projection` plus the registered per-direction minimum-magnitude table; keep it pure over `TargetChunk.virtual_root_position`.
- [ ] Step 2.3 — Run GREEN; confirm signed projection and orthogonal-tolerance assertions pass for forward/backward/left/right and the neutral-reduces-travel case.
- [ ] Step 2.4 — Run the real MM-only four-reset probe and preserve each candidate ID, command, root displacement, projection, and pass/fail result; stop if any sign fails.
- [ ] Step 2.5 — Commit as commit boundary 2 ("feat: prove MM-generated direction signs from targets").

### Task 3: Responsive prefix scheduler (Stage R1)

- **New module:** `sonic/python/mm_sonic/responsive_scheduler.py`. A thin
  orchestrator that drives the existing `Coordinator.run_one_chunk`
  (`coordinator.py:950`) one chunk at a time, latches an immutable mailbox
  `IntentSnapshot(revision, observed_ns, command)`, and emits a `BoundaryTrace`.
- **Modify:** `sonic/python/mm_sonic/operator_x11.py` to increment `revision`
  only when effective velocity, heading, stand, or terminate changes; camera
  motion alone must not supersede a locomotion candidate.
- **Modify:** `sonic/python/mm_sonic/coordinator.py` to accept an optional
  `command_is_current: Callable[[CommandSample], bool]`. Evaluate it after MM
  generation/validation and immediately before publication. A false result
  raises a typed pre-commit supersession outcome that uses the coordinator's
  existing `TargetTimeline.abort` plus `mm.abort` cleanup and releases no
  physics. The scheduler retries the same chunk index with the newest revision.
- **New test file:** `tests/python/test_sonic_responsive_scheduler.py`, using
  the injected-dependency test doubles already established in
  `tests/python/test_sonic_coordinator.py` (which constructs `Coordinator` with
  fake `mm`, `gate`, `publisher`, `timeline_factory`, `run`).
  - `test_scheduler_never_holds_more_than_one_unconsumed_prefix` — asserts at
    every boundary the timeline has at most one committed-but-unreleased chunk
    (design "Acceptance Tests / Pure and protocol tests").
  - `test_stale_candidate_from_newer_latched_command_is_aborted_before_publish`
    (**Probe P2**). The fake MM callback publishes a newer mailbox revision
    during generation; the coordinator predicate then rejects the old command,
    aborts via `TargetTimeline.abort` and `mm.abort`, and never calls
    `publisher.send_prepared` for it. This drives the real pre-commit cleanup.
  - `test_generation_failure_or_timeout_leaves_physics_paused` — a raising
    `mm.generate` must terminalize with `simulation_paused=True`
    (mirrors `test_sonic_coordinator.py:1346`
    `test_dead_server_is_detected_without_abort_and_terminalizes_paused`) and
    must **not** release any physics steps (`gate.release_steps` uncalled).
  - `test_physics_cannot_advance_before_target_acknowledgement` — asserts
    `gate.release_steps` is invoked strictly after the timeline commit and
    publication (ordering assertion on the fake gate/publisher call log),
    reproducing the real ordering at `coordinator.py:1104` (publish) →
    `:1120` (mm commit) → `:1131` (timeline commit) → `:1146` (release).
- **RED command:**
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_responsive_scheduler -v`
- **Expected RED:** `ModuleNotFoundError: mm_sonic.responsive_scheduler`, then,
  after a stub module exists, `AssertionError` on the ordering/one-prefix
  assertions until the scheduler enforces single-prefix + pause-on-miss.
- **Minimal change:** implement the scheduler as a boundary loop like
  `operator_runtime.py:run_operator_boundary_loop`. It passes the latched
  revision predicate into the coordinator, retries only typed supersession,
  records a trace for committed prefixes, and leaves all generation,
  validation, publication, and timeout failures paused and terminal.
- **GREEN command:** same as RED.
- **Commit boundary 3:** "feat: single-prefix responsive scheduler with pause-on-miss."

Checklist:

- [ ] Step 3.1 — Write RED mailbox tests proving camera-only updates retain the revision while backward, lateral, stop, and heading changes increment it.
- [ ] Step 3.2 — Write the four RED scheduler/coordinator tests using the `test_sonic_coordinator.py` injected fakes; confirm the supersession case currently publishes the stale candidate.
- [ ] Step 3.3 — Implement `IntentSnapshot`, the optional coordinator `command_is_current` pre-publication check, and typed pre-commit supersession cleanup.
- [ ] Step 3.4 — Implement the boundary loop, retrying the same chunk index only after typed supersession and never after another failure.
- [ ] Step 3.5 — Force a raising `mm.generate` and acknowledgement failure; assert terminal `simulation_paused=True` with `gate.release_steps` uncalled, run GREEN, and commit as commit boundary 3.

### Task 4: Reduce interactive lookahead to one chunk (Stage R1 wiring)

- **Change:** In `manual_demo.py`, add an explicit `--responsive` selector that
  routes the interactive path through the Task 3 scheduler with a one-chunk
  committed prefix, leaving the existing two-chunk launcher path
  (`manual_demo.py:811-812` default) **untouched** as the rollback/comparison
  path (design "Evidence and Rollback"). The responsive path emits per-boundary
  `BoundaryTrace` records and the design's published latency fields.
- **New test:** `tests/python/test_sonic_manual_demo.py`
  - `test_responsive_selector_uses_single_prefix_and_emits_boundary_trace`
    — argument parsing + summary schema assertions only (no live process),
    reusing the summary-schema assertion pattern already in
    `test_sonic_manual_demo.py`.
- **RED command:**
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_manual_demo -v`
- **Expected RED:** `AttributeError`/argparse error — `--responsive` unknown.
- **Minimal change:** add the flag, the scheduler wiring, and the trace/latency
  fields to the summary; do not alter the two-chunk default.
- **GREEN command:** same as RED.
- **Commit boundary 4:** "feat: opt-in responsive one-prefix interactive path."

Checklist:

- [ ] Step 4.1 — Add the RED `test_responsive_selector_uses_single_prefix_and_emits_boundary_trace` to `tests/python/test_sonic_manual_demo.py`; run RED and confirm the argparse/`AttributeError` failure on `--responsive`.
- [ ] Step 4.2 — Add the `--responsive` flag and route it through the Task 3 scheduler with one committed prefix, leaving the two-chunk default path untouched.
- [ ] Step 4.3 — Extend the run summary with the boundary-trace and latency fields; run GREEN and commit as commit boundary 4.

### Task 5: SONIC future-reference measurement (Probe P0, Stage R2 precondition)

- **New test:** `tests/python/test_sonic_stage_b.py` (or a sibling
  dynamic-gate module the controller runs)
  - `test_measures_sonic_future_reference_frame_requirement` (**Probe P0**).
    Drives pinned `g1_deploy_onnx_ref` with sealed prefixes of 4, 8, 12, 16,
    and 20 frames from the same authenticated target. For each prefix it records
    `wait_for_stream_processing` acknowledgement, `catch_up`, first valid
    control output, process survival, and an exactly gated release. This is a
    protected feasibility experiment; it does not relax the production v1
    20-frame validator to obtain a pass.
- **RED/GREEN commands:** controller-owned dynamic-gate invocation. Expected
  RED: the registered assertion that 4 frames suffice fails with the smallest
  healthy prefix in its diagnostic. GREEN exists only when the 4-frame case
  produces healthy control and upright release.
- **Decision rule (recorded in evidence):**
  - If 4 frames are healthy and MM-generation-plus-publication p95 is at most
    120 ms: proceed to Stage R2.
  - Otherwise: **stop at R1**, report sub-200 ms infeasible for the unmodified
    policy/transport, and publish the measured lower bound.
- **Commit boundary 5:** "test: measure SONIC future-reference frame requirement."

Checklist:

- [ ] Step 5.1 — Add the protected 4/8/12/16/20-frame prefix experiment and prove the initial four-frame sufficiency assertion is RED or GREEN from the real pinned binary.
- [ ] Step 5.2 — Record every prefix result plus MM generation/publication p50, p95, and max in run evidence.
- [ ] Step 5.3 — Apply the exact four-frame/120 ms decision rule and commit the measured gate as commit boundary 5.

### Task 6: Registered 80 ms protocol variant (Stage R2, only if P0 permits)

- **Modify:** `sonic/cpp/mm_chunk_protocol.h` and
  `sonic/cpp/mm_chunk_server.cpp` so requests accept exactly
  `source_intervals in {2, 10}`, candidate vectors contain `N+1` boundaries and
  `N` diagnostics, hello advertises `[2,10]`, and serialization echoes `N`.
- **Modify:** `sonic/python/mm_sonic/process.py` and
  `sonic/python/mm_sonic/schema.py` to accept only 2 or 10 and validate every
  boundary/diagnostic array from the authenticated interval count.
- **Modify:** `sonic/python/mm_sonic/resample.py` to map `N` 25 Hz source
  intervals into `2N` 50 Hz target frames, and
  `sonic/python/mm_sonic/timeline.py` to add `target-chunk/v2` fixed at four
  frames while leaving v1 fixed at twenty.
- **Modify:** `sonic/python/mm_sonic/manual_demo.py` responsive mode to publish
  one four-frame prefix and release exactly
  `round((2 / 25) / simulator.sim_dt)` physics steps. At the authenticated
  `0.005 s` timestep this is 16 physics steps, not four.
- **Tests:** `tests/cpp/test_mm_chunk_protocol.cpp`,
  `tests/python/test_sonic_process.py`, `test_sonic_schema.py`,
  `test_sonic_resample.py`, `test_sonic_timeline.py`, and
  `test_sonic_manual_demo.py` cover 2→3 boundaries→4 target frames, the legacy
  10→11→20 path, rejection of all other widths, contiguous frame indices,
  exact acknowledgement-before-16-step-release ordering, and v1 rollback.
- **RED command:**
  `ctest --test-dir build -R mm_chunk_protocol --output-on-failure` followed by
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_process tests.python.test_sonic_schema tests.python.test_sonic_resample tests.python.test_sonic_timeline tests.python.test_sonic_manual_demo -v`.
- **Expected RED:** interval 2 is rejected by the C++/Python protocol and the
  fixed 11/10/20 shapes.
- **GREEN:** both registered sizes pass; byte/content expectations for the v1
  10/20 path remain unchanged except the explicit capability advertisement.
- **Commit boundary 6:** "feat: add registered 80 ms SONIC target prefixes."

Checklist:

- [ ] Step 6.1 — Only after Task 5 is GREEN, write C++ and Python RED tests for the exact registered sizes `{2,10}` and rejection of every other size.
- [ ] Step 6.2 — Parameterize the C++ transaction and JSON serialization by the authenticated request size; run the focused C++ GREEN gate.
- [ ] Step 6.3 — Parameterize Python client/schema shapes and cross-field validation; run process/schema GREEN tests.
- [ ] Step 6.4 — Parameterize resampling and timeline frame allocation, add four-frame v2 beside twenty-frame v1, and run resample/timeline GREEN tests.
- [ ] Step 6.5 — Wire responsive publication and exact 16-step gated physics release, prove no release precedes acknowledgement, and run manual-demo GREEN tests.
- [ ] Step 6.6 — Run the complete CPU regression suite and commit as boundary 6 only when v1 and v2 are both green.

---

## 6. Controller-owned integration gates

These are executed by the controller (`gate-001` plus the dynamic gates); this
plan does not run them.

- **gate-001** — the controller-owned plan-quality verifier over this selected
  plan. Claude Plan A first passed the protected format/quality gate and an
  independent review; Codex's protocol amendments must pass the same protected
  gate before implementation begins.
- **MM-only direction gate** — Probe P1 (Task 2) plus a live variant that scores
  four live generations from an identical authenticated reset
  (`scene_runtime.py:initial_physics_state`) with the same
  `signed_root_projection`.
- **Flat dynamic gate** — unmodified SONIC stays upright across direction and
  stop transitions; the run reports command→target and target→root tracking
  **separately** (design §"Latency Contract" published fields), and p95
  presentation latency ≤ 200 ms **only for on-time generations** — admissible
  solely if Task 5 (P0) has established that the four-frame sealed prefix is
  healthy, generation plus publication p95 is at most 120 ms, and Stage R2 is
  in effect.
  Under Stage R1 this gate reports the measured 0.4 s + generation floor and is
  not expected to meet 200 ms.
- **Terrain dynamic gate** — repeat the responsive forward/stop sequence on
  `grail-curb-default` / `curb-forward` with terrain weight `4.0`
  (design §"Terrain dynamic gate"); SONIC stays upright and advances the
  registered route. `terrain_weight` binary32-exactness is already enforced by
  `SessionConfig.__post_init__` (`coordinator.py:80-94`).
- **Stop-displacement gate (P4)** — the specific gate the prior one-chunk attempt
  failed; re-run under Stage R1 and record pass/fail honestly.

Physics-pause safety is asserted at both layers: the coordinator terminalizes
paused on any failure (`coordinator.py:876-916` `_terminalize_failure` /
`_paused_where_possible`), and the scheduler (Task 3) never releases steps on a
miss.

---

## 7. Latency contract and honest bounds

Primary metric (design §"Latency Contract"):
`presentation_latency = first_released_frame_for_new_command − key_transition_timestamp`,
measured from the Task 1 boundary trace.

Published per run (all already producible from existing timing +
the new trace): input-to-sample latency, MM generation duration
(`coordinator.py` `_TIMING_STAGES` `mm_generation` at `:40`), validation +
publication duration (`projection_validation`, `resampling`, `publication`),
queued irrevocable lookahead at the transition (chunks × 0.4 s), presentation
latency, physics pause duration (generation stall, reported separately as
non-real-time), and direction + tracking errors.

**Honest bounds this plan commits to:**

1. Stage R1 lowers presentation latency from `0.8 s` (interactive default of 2
   chunks) to a floor of **`0.4 s + t_generate`** (one irrevocable chunk). It
   does **not** meet 200 ms and does not claim to.
2. A ≤ 200 ms p95 claim is admissible **only** after Probe P0 establishes that
   the unmodified SONIC remains healthy with a four-frame sealed prefix and
   generation plus publication p95 is at most 120 ms. Stage R2 then uses
   exactly two 25 Hz MM source intervals: 80 ms, four 50 Hz target frames, and
   16 MuJoCo steps at `sim_dt = 0.005`. If either P0 gate fails, the deliverable
   is the R1 floor plus a recorded protocol-blocker verdict.
3. Paused wall-clock time is reported as generation stall, never counted as
   simulated real-time success (design §"Latency Contract").

---

## 8. Risks and mitigations

Each risk below is tied to the probe or gate that retires it. Ordering
(R1 before R2) is itself the primary risk mitigation: it lands a safe, useful
improvement before any protocol change is attempted.

- **Risk A — SONIC cannot run from a four-frame sealed prefix.** If Probe P0
  finds any process, catch-up, control-validity, or upright-health failure at
  four frames, the 20-frame floor cannot shrink to the required 80 ms safely.
  *Mitigation:* the plan fails closed at the R1 floor and reports the measured
  minimum healthy prefix rather than claiming 200 ms; no protocol edit is
  attempted.
- **Risk B — the one-chunk stop-displacement regression recurs.** The prior
  one-chunk attempt failed the stop gate (§1.3). *Mitigation:* Probe P4 re-runs
  the exact stop-displacement gate under Stage R1 and records pass/fail
  honestly; a failure blocks Stage R1 acceptance, not just a warning.
- **Risk C — a superseded candidate leaks into the irrevocable merged stream.**
  A latched command change mid-generation could publish stale motion.
  *Mitigation:* Probe P2 drives the real coordinator abort branch and asserts
  `publisher.send_prepared` is never called for the stale candidate.
- **Risk D — physics advances past the last appended reference frame.** SONIC
  requires future-reference lookahead (§1.4). *Mitigation:* the scheduler holds
  the gate closed until acknowledgement and never releases steps on a miss
  (Task 3), asserted by `test_physics_cannot_advance_before_target_acknowledgement`
  and `test_generation_failure_or_timeout_leaves_physics_paused`.
- **Risk E — a protocol edit corrupts the append-only reference invariant.** The
  additive `target-chunk/v2` could break the `(count-1) % 20` contract for `v1`
  consumers. *Mitigation:* v2 is additive and versioned behind a new schema tag;
  `v1` and the two-chunk launcher remain the untouched rollback path, and Stage
  R2 is entered only after P0 permits.
- **Risk F — MM direction correctness is conflated with SONIC tracking.**
  *Mitigation:* Probe P1 (pure) and the flat dynamic gate (physics) are separate
  artifacts, so a passed-MM / failed-physics result is diagnosable at the exact
  failing boundary.

---

## 9. Evidence, rollback, and non-goals

- Every run stores the command artifact (`manual-commands.json`), the boundary
  trace, MM candidate identities (`coordinator.py:784-786`
  `session:candidate:NNNNNN`), scene and reset hashes
  (`scene_runtime.py` `initial_boundary_sha256`, `qpos_sha256`), generated and
  published root trajectories (`ReferenceDiagnostics` in `reference.py:60`,
  `mm_root_diagnostic.csv` at `reference.py:232`), the simulator snapshot
  (`gated_sim.py:527` `snapshot`), and the exact configuration.
- **Rollback:** the existing two-chunk launcher path and `target-chunk/v1`
  remain the untouched comparison/rollback route; the responsive scheduler and
  any v2 unit are selected explicitly and never silently alter formal Stage B
  evidence.
- **Non-goals honoured:** no SONIC training, no depth conditioning, no obstacle
  avoidance, no manipulation, no MuJoCo-root feedback into Motion Matching, and
  no claim of real-time performance while physics is paused.

---

## 10. Summary of what is proven vs. inferred

- **Observed from code:** the 0.8–1.6 s lookahead arithmetic (§1.1); the fixed
  20-frame / `source_intervals=10` append-only transport and its irrevocable
  commit (§1.2); the existing direction-mapping coverage and the two missing
  MM-direction / SONIC-tracking proofs (§3); the pause-on-failure machinery
  (§6); the `catch_up`/merged-tail evidence that SONIC consumes future reference
  frames (§1.4).
- **Inferred / made into probes:** whether the pinned SONIC binary remains
  healthy at the required four-frame prefix and its minimum healthy prefix
  (Probe P0); the per-direction generated-root magnitudes (Probe P1); the
  one-chunk stop-displacement outcome (Probe P4).
- **Falsified:** the literal 100 ms committed prefix on the *unmodified*
  transport (§2); the smallest atomic unit that exists at HEAD is 0.4 s, so
  sub-200 ms requires an additive protocol change gated behind P0.
