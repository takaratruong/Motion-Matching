# G1 SONIC Stage-R1 Responsive Wiring Implementation Plan

**Author:** Claude (independent wiring plan)
**Status:** Implementation-ready (Stage R1 wiring only)
**Milestone:** `responsive-wiring-plan`
**HEAD:** `ab5e4b01aa9a031032cdf2a6e922df3a480e9c06`
**Design base:** `docs/superpowers/plans/2026-07-20-g1-sonic-responsive-control.md`
(selected Stage-R1/R2 plan; this plan implements its **Task 4 wiring** only)

> **Scope of this plan.** This is the *smallest correct wiring* that lets
> `manual_demo` opt into **one-chunk responsive SONIC control** using the
> already-implemented `ResponsiveScheduler` (`responsive_scheduler.py`),
> `BoundaryControlMailbox` / `IntentSnapshot` (`operator_x11.py`), the
> coordinator supersession path (`coordinator.py` `command_is_current` /
> `CandidateSuperseded`), and `BoundaryTrace` (`boundary_trace.py`). It does
> **not** re-implement those; they exist at HEAD. It changes no default path.

> **For agentic workers.** All formal verification (`gate-001`, RED/GREEN
> commands) is controller-owned; this plan states commands and expected
> outcomes but does not run live MM/GEAR/SONIC/MuJoCo/GPU/network. Interpreter
> idiom used by sibling plans:
> `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.<module> -v`,
> run from the repository root.

---

## 1. Live-evidence starting point and the next fault boundary

Per the frozen request and the controller's forwarded live evidence: **vanilla
Motion Matching already passed forward, backward, left, and right** direction
checks. The MM-only direction proof is implemented and its oracle is pure:

- `sonic/python/mm_sonic/direction_probe.py:63` `signed_root_projection(target, axis_mujoco)`
  projects the final-minus-initial `virtual_root_position` displacement onto a
  horizontal MuJoCo x-y axis (`direction_probe.py:66-80`).
- `sonic/python/mm_sonic/mm_direction_gate.py:80` `score_target_direction`,
  `:309` `run_live_direction`, `:364` `run_live_direction_gate` score four live
  resets against registered per-direction minimums.
- Commit `ab5e4b0 feat: add live MM direction gate` is the most recent commit.

**Therefore the next open fault boundary is not MM direction.** It is SONIC
tracking/publication/lookahead under a *responsive one-chunk* prefix: whether a
freshly re-sampled command is committed and released without stale motion, and
whether the boundary trace can bind the input transition to the presented
prefix with honest tracking evidence. This plan wires exactly that boundary and
specifies its tests; the dynamic tracking/stop gates remain controller-owned
(the selected plan's §6 flat/stop/terrain dynamic gates).

---

## 2. The architecture mismatch to reconcile (observed facts)

### 2.1 Two different "commit one chunk" owners exist

`ResponsiveScheduler` and `manual_demo` implement the *same* generate → prepare
→ publish → commit transaction through **two different owners**:

- **Scheduler path (`responsive_scheduler.py:33` `run_one_prefix`).** Samples
  `mailbox.sample_intent(chunk_index)` → `(IntentSnapshot, mapped)`
  (`responsive_scheduler.py:45`), binds `snapshot.revision` into a
  `command_is_current` predicate (`:50-54`), calls
  `coordinator.run_one_chunk(snapshot.command, command_is_current=...)`
  (`:57-60`), and retries the *same* `chunk_index` only on typed
  `CandidateSuperseded` (`:61-63`). A terminate snapshot (`snapshot.command is
  None`) returns `None` without touching the coordinator (`:46-47`).
- **`Coordinator.run_one_chunk` (`coordinator.py:1006`).** The full state
  machine: `READY_PAUSED` guard (`:1015`), latch, `gate.pause()` (`:1060`), MM
  generate (`:1071-1077`), source validation (`:1087-1091`), timeline prepare +
  optional kinematic validation (`:1101-1126`), `write_prepared` enqueue
  (`:1127-1140`), **the last pre-publication supersession check**
  (`:1146-1163`), publish (`:1165-1199`), `mm.commit` (`:1206`),
  `timeline.commit` (`:1217`), `gate.release_steps(steps_per_chunk)`
  (`:1229-1234`), acceptance assembly and evidence writes (`:1246-1293`), then
  back to `READY_PAUSED` (`:1294-1296`).
- **`manual_demo.generate_and_publish` (`manual_demo.py:473-498`).** A parallel,
  *coordinator-free* transaction: `mm.generate` (`:481`), `timeline.prepare`
  (`:488`), `publish(...)` (`:490`), on failure `timeline.abort` + `mm.abort`
  (`:492-494`), then `mm.commit` + `timeline.commit` + `recorder.record`
  (`:495-497`), and physics is released *separately* by the caller via
  `gate.release_steps(steps_per_chunk)` (`manual_demo.py:373`, `595`, `683`).

`manual_demo` owns its own nested loop and a **custom GEAR/simulator bootstrap**
(`manual_demo.py:501-581`): startup transaction, `_drive_simulator_until`,
`prepare_stream` (preload loop `:560-561`), `gear.stop_group()` →
`_reset_and_prime_scored_epoch` → `gear.continue_group()` →
`gear.activate_control()` (`:569-578`), and only then constructs
`SimulationPolicyGate(gear, simulator)` and `gate.pause()` (`:579-580`).

**Consequence.** The `Coordinator` is *not currently instantiated anywhere in
`manual_demo`*. `manual_demo` never builds a `Coordinator`; it drives `mm`,
`timeline`, `publisher`, and `gate` directly. Verified:
`grep -n "Coordinator" sonic/python/mm_sonic/manual_demo.py` returns nothing
(only `SourceValidator` and `SessionConfig` are imported from `coordinator`,
`manual_demo.py:28`). This is the crux the wiring must reconcile: the scheduler
speaks `Coordinator.run_one_chunk`, but `manual_demo` has no coordinator.

### 2.2 Why an adapter — not a coordinator retrofit — is the smallest change

Retrofitting a real `Coordinator` into `manual_demo` would require rebuilding
its dependency graph (`mm`, `validator`, `timeline_factory`, `run`,
`publisher`, `stream`, `gate`, `target_motion_logfile`, `steps_per_chunk`,
`delivery_auditor`, readiness/preflight ownership — `coordinator.py:587-648`).
The coordinator owns `preflight`/readiness and the official-log identity
(`coordinator.py:710-796`), which `manual_demo` performs through an entirely
different bootstrap (`prepare_stream` + `_reset_and_prime_scored_epoch`). That
is a large, risky refactor and would touch the default path.

**The scheduler only requires the `run_one_chunk(command, command_is_current=…)`
duck type** (`responsive_scheduler.py:26` checks `hasattr(coordinator,
"run_one_chunk")`). The smallest correct change is therefore a **thin adapter**
that presents `manual_demo`'s existing committed-transaction as
`run_one_chunk`, plus a `sample_intent` mailbox that `manual_demo` already has
(`BoundaryControlMailbox`, `operator_x11.py:122`, exposes `sample_intent`
`:216` and `current_revision` `:145`). No coordinator, no bootstrap change.

---

## 3. The exact adapter (smallest correct symbol set)

Introduce one new module, `sonic/python/mm_sonic/responsive_wiring.py`, holding
a single adapter class and a wiring helper. **No production code outside this
module and `manual_demo.py` is edited; no default path changes.**

### 3.1 `ManualChunkCommitter` — the `run_one_chunk` adapter

A class that wraps `manual_demo`'s existing per-boundary transaction and exposes
the exact scheduler contract, preserving the coordinator's failure ordering:

```
class ManualChunkCommitter:
    """Adapt manual_demo's committed transaction to the scheduler's
    run_one_chunk(command, *, command_is_current) contract.

    Ordering mirrors Coordinator.run_one_chunk exactly for the responsive
    path: generate -> validate -> prepare -> (LAST supersession check) ->
    publish -> mm.commit -> timeline.commit -> release physics -> record.
    Physics is released ONLY after publication and both commits.
    """

    def __init__(self, *, mm, validator, timeline, publisher, gate,
                 session_id, steps_per_chunk, recorder, publish, trace_sink):
        ...

    def run_one_chunk(self, command, *, command_is_current=None):
        # 1. next_chunk / chunk_index guard (mirrors manual_demo.py:476-479)
        # 2. candidate_id = f"{session_id}:candidate:{chunk:06d}"
        # 3. raw = mm.generate(...);  mm_started_ns / mm_completed_ns captured
        # 4. checked = validator.validate_source(raw)
        # 5. prepared = timeline.prepare(checked)
        # 6. LAST supersession check: if command_is_current is not None and
        #    not command_is_current(command):
        #        timeline.abort(candidate_id); mm.abort(candidate_id)
        #        raise CandidateSuperseded(candidate_id, command)
        # 7. publish(prepared.target.buffer, phase="timeline", wait=False)
        #        on BaseException: timeline.abort + mm.abort; re-raise
        # 8. committed_ns; mm.commit(candidate_id); timeline.commit(prepared)
        # 9. advance = gate.release_steps(steps_per_chunk)  # physics released
        #10. recorder.record(command); emit BoundaryTrace via trace_sink
        #11. return an AcceptedChunk-like object (target, advance, timings)
```

Key correctness properties the adapter must hold, each mapped to existing code:

- **Exactly one preloaded chunk.** The wiring calls `prepare_stream` with a
  single preload iteration (see §4); `steps_per_chunk` stays
  `round(_CHUNK_DURATION_S / simulator.sim_dt)` (`manual_demo.py:581`), so one
  boundary releases exactly one 0.4 s chunk. `_CHUNK_DURATION_S = 0.4`
  (`manual_demo.py:85`).
- **Re-sample revised input before publication.** The scheduler re-samples
  `sample_intent` at the top of `run_one_prefix` (`responsive_scheduler.py:45`)
  and re-binds the revision; on supersession it loops and re-samples the newest
  revision (`:61-63`). The adapter's step 6 is the *last* pre-publication check,
  identical in position to `coordinator.py:1146-1163`.
- **Abort stale candidates.** Step 6 calls `timeline.abort(candidate_id)` +
  `mm.abort(candidate_id)` (the exact cleanup `manual_demo.py:492-494` already
  uses on failure) and raises `CandidateSuperseded` (`coordinator.py:74`), which
  the scheduler catches and retries (`responsive_scheduler.py:61`).
- **Release physics only after publication and commit.** Step 9's
  `gate.release_steps` runs strictly after step 7 (publish) and step 8 (`mm`
  + `timeline` commit) — mirroring `coordinator.py:1190` (publish) → `:1206`
  (`mm.commit`) → `:1217` (`timeline.commit`) → `:1229` (release). No physics is
  released on generation/validation/publication failure or on supersession.

### 3.2 Why reuse `CandidateSuperseded` verbatim

`responsive_scheduler.py:18` imports `CandidateSuperseded` from `coordinator`
and catches exactly that type (`:61`). The adapter must raise the same class
(`coordinator.py:74`, constructor `(candidate_id, command)` at `:82-85`) so the
scheduler's retry contract is satisfied without any scheduler edit.

---

## 4. `manual_demo` opt-in wiring (default path untouched)

### 4.1 New CLI flag and default preservation

Add to `_parser` (`manual_demo.py:824-841`):

```
parser.add_argument("--responsive", action="store_true")
```

In `_resolve_mode_defaults` (`manual_demo.py:790-812`) **leave every existing
default unchanged**. The existing interactive default remains
`preload_chunks = 2` (`manual_demo.py:812`) and two-chunk behavior. `--responsive`
is opt-in and orthogonal; it is only honored on the interactive/x11 path.

**Non-negotiable default invariant.** When `namespace.responsive` is false,
`run_demo` executes exactly the code at HEAD. The responsive branch is a *new*
`if namespace.responsive:` arm inside the `input_source == "x11"` block
(`manual_demo.py:602-655`), guarded so the two-chunk path is byte-for-byte the
current one. The `test_sonic_manual_demo.py` default-preserving tests
(`:50 test_preload_chunks_defaults_to_four`, `:53`, `:69`) must remain GREEN.

### 4.2 One preloaded chunk for responsive mode

Responsive mode must use **exactly one** preloaded 0.4 s chunk. The current
preload loop is `for index in range(preload_chunks): generate_and_publish(...)`
(`manual_demo.py:560-561`). For responsive mode, force `preload_chunks = 1`
locally (validated by `_validated_preload_chunks`, `manual_demo.py:92-103`,
which accepts 1) so `prepare_stream` commits a single stand chunk before control
begins. This is the "one committed prefix" floor from the selected plan
(§2 "Stage R1"): lookahead drops from 0.8 s (two chunks) to one irrevocable
0.4 s chunk. The default two-chunk path never sets this.

### 4.3 Responsive boundary loop

Replace, **only inside the `if namespace.responsive:` arm**, the direct
`_consume_x11_boundary` loop with a scheduler-driven loop:

```
committer = ManualChunkCommitter(mm=mm, validator=validator, timeline=timeline,
    publisher=publisher, gate=gate, session_id=session_id,
    steps_per_chunk=steps_per_chunk, recorder=recorder, publish=publish,
    trace_sink=trace_records.append)
scheduler = ResponsiveScheduler(committer, control_loop.mailbox)
for _consumed_chunk in range(namespace.chunks):
    # Deliver camera synchronously first (see §7), then commit one prefix.
    accepted = scheduler.run_one_prefix(next_chunk)
    if accepted is None:      # terminate snapshot: X pressed
        break
    next_chunk += 1
```

`control_loop.mailbox` is the live `BoundaryControlMailbox`
(`operator_x11.py:259` `self.mailbox = BoundaryControlMailbox()`), which already
exposes `sample_intent` (`:216`) and `current_revision` (`:145`) — the two
attributes `ResponsiveScheduler.__init__` requires
(`responsive_scheduler.py:26-29`). No mailbox change is needed.

**Ordering note (crucial).** `manual_demo`'s `next_chunk` counter
(`manual_demo.py:455`, incremented at `:498`) must stay the source of truth for
`chunk_index`. The adapter's step-1 guard must reject
`command.chunk_index != next_chunk` exactly as `manual_demo.py:476-479` and
`coordinator.py:1027-1033` do. On supersession, the scheduler re-samples the
*same* `chunk_index` (`responsive_scheduler.py:62`), and because no commit
occurred, `next_chunk` is unchanged — the invariant holds.

---

## 5. Honest BoundaryTrace evidence semantics (the hard constraint)

`BoundaryTrace` (`boundary_trace.py:68`) requires eight monotone timestamps
(`_TIMESTAMP_FIELDS`, `:21-30`) and five finite numeric vectors
(`_VECTOR_FIELDS`, `:33-39`), including `observed_mujoco_root_displacement`
(width 3, `:38`). The blunt fact: **the simulator snapshot and `AdvanceResult`
expose no root pose.**

### 5.1 What each trace field can honestly be sourced from

| Trace field | Honest source | Evidence |
|---|---|---|
| `input_transition_id` | `f"{session_id}:rev:{snapshot.revision:06d}"` | `IntentSnapshot.revision`, `operator_x11.py:109` |
| `presented_prefix_id` | `candidate_id` = `f"{session_id}:candidate:{chunk:06d}"` | `coordinator.py:800`, `manual_demo.py:480` |
| `input_observed_ns` | `snapshot.observed_ns` (mailbox monotonic clock) | `operator_x11.py:110,184` |
| `sampled_ns` | monotonic read at scheduler sample return | wrap `sample_intent` call site |
| `mm_started_ns` / `mm_completed_ns` | monotonic around adapter step 3 (`mm.generate`) | mirrors `_TIMING_STAGES` `mm_generation`, `coordinator.py:40` |
| `committed_ns` | monotonic after `mm.commit`+`timeline.commit` (adapter step 8) | `coordinator.py:1206,1217` |
| `published_ack_ns` | monotonic after `publish(...)` returns (adapter step 7) | `manual_demo.py:490`; wait-mode ack `:462-471` |
| `physics_released_ns` | monotonic after `gate.release_steps` returns (step 9) | `manual_demo.py:595`; gate `process.py:2940` |
| `first_simulated_frame_ns` | monotonic at first post-release step, or `= physics_released_ns` if not separately observable | see §5.3 |
| `requested_velocity_mujoco` | `command.requested_velocity_mujoco` | `commands.py` `CommandSample`; used `manual_demo.py:374,598` |
| `requested_heading_mujoco_wxyz` | `command.desired_heading_mujoco_wxyz` | `manual_demo.py:396-398` |
| `generated_virtual_root_displacement_mujoco` | last-minus-first row of `prepared.target.virtual_root_position` | `timeline.py:79,516-518`; `direction_probe.py:66-80` |
| `published_physical_root_displacement_mujoco` | **see §5.2 — not on the wire** | `zmq_v1.py:69,71` |
| `observed_mujoco_root_displacement` | **see §5.3 — not in snapshot** | `gated_sim.py:527-536`, `process.py:1727-1744` |

### 5.2 `published_physical_root_displacement_mujoco`: the wire carries no root

The published buffer is a `CanonicalTargetBuffer` carrying **only**
`joint_position` and `body_quat_w` (`zmq_v1.py:69,71`; buffer fields
`timeline.py:181-185`). The wire (`_pose_arrays`, `zmq_v1.py:66-72`) contains no
root translation at all. `virtual_root_position` lives on the *prepared
`TargetChunk`* (`timeline.py:79`) but is **not** part of `.buffer`
(`timeline.py:171-177` includes only `joint_position`, `body_quat_w`,
`frame_index`).

**Honest resolution.** The "published physical-root displacement" is the
producer-side root trajectory that is actually published/recorded, which is the
prepared target's root converted to the physical/MuJoCo basis — the same
`virtual_root_position` used for the generated field, recorded as the
`ReferenceDiagnostics` / `mm_root_diagnostic.csv` artifact
(`reference.py:232`, header validated `:354-365`). Since Stage R1 does not add a
root channel to the wire, the plan must state one of two honest options and
**not invent zero evidence**:

- **Option A (recommended for Stage R1):** set
  `published_physical_root_displacement_mujoco` equal to the
  `generated_virtual_root_displacement_mujoco` *only* when the published buffer
  is provably the prepared target's buffer (identity check:
  `prepared.target.buffer is published_buffer`), and document in the trace that
  Stage R1 publishes joint/quaternion targets whose root channel is the
  generated target root (no separate physical-root wire exists). This is
  truthful: the two are the same object at HEAD.
- **Option B:** omit the responsive trace's physical-root field from Stage R1 by
  gating trace emission behind an explicit `--emit-root-trace` that is only set
  when a root-diagnostic sink is wired, and record R1 without it. This defers to
  the diagnostic CSV rather than fabricating a wire value.

Because `BoundaryTrace.__post_init__` requires a finite width-3 vector for that
field (`boundary_trace.py:106-109`), Option A is the smallest correct choice:
it supplies a *real, cited* value (the generated target root) and states plainly
that Stage R1 has no independent physical-root wire channel. **Do not** fill it
with zeros — that would be inventing evidence the request forbids.

### 5.3 `observed_mujoco_root_displacement`: snapshot has no root, state.jsonl does

`GatedSimulatorClient.snapshot()` returns only
`{steps, sim_time_s, state_rows, contact_rows}` (`process.py:1727-1744`,
runner `gated_sim.py:527-536`), and `AdvanceResult` is
`{steps, sim_time_start_s, sim_time_end_s, state_rows, contact_rows}`
(`process.py:581-586`). **Neither exposes root pose.** Confirmed by inspection.

However, the per-step **state log does** carry root pose. `MujocoBackend.sample`
returns `qpos` and `pelvis_position_m`/`pelvis_quaternion_wxyz`
(`gated_sim.py:1170-1188`), and `advance()` writes each strided sample as a
`state.jsonl` row (`gated_sim.py:497-509`; file `logs/state.jsonl`,
`gated_sim.py:448`). So the observed MuJoCo root displacement for a chunk is
**recoverable** as `pelvis_position_m` (or `qpos[0:3]`) at the last released
step minus the value at the boundary start, read from `state.jsonl` for the
`log_dir` the responsive run passed to `simulator.reset(...)` (`manual_demo.py:534-541`
passes `log_dir=bundle.path / "scored-sim-logs"` via `_reset_and_prime_scored_epoch`,
`manual_demo.py:570-576`).

**Honest resolution (the only truthful one).** The `AdvanceResult` returned by
`gate.release_steps` does **not** contain root pose, so the trace cannot be
emitted synchronously from `run_one_chunk` alone with a real observed
displacement. Two disciplined options, both cited, neither inventing zeros:

- **Option A (recommended):** Emit the `BoundaryTrace` **after** reading the
  two `state.jsonl` rows bounding the just-released chunk. The committer records
  `state_rows` before/after `release_steps` (both available on `AdvanceResult`,
  `process.py:585`), then reads the last matching `state.jsonl` row's
  `pelvis_position_m` for the observed end position and the boundary-start row's
  for the start, and sets `observed_mujoco_root_displacement = end - start` in
  the MuJoCo world basis. This is a real measurement from an existing log; it
  adds only a narrow file read, not a protocol change.
- **Option B (fail-honest):** If the responsive run is configured without a
  state log stride that captures the boundary frames, do **not** synthesize a
  displacement. Instead emit a *partial* evidence record to a separate
  `responsive-latency.jsonl` artifact (timestamps + generated/requested vectors
  only) and record in the plan/run summary that `observed_mujoco_root_displacement`
  was unavailable for that run — never a zero vector inside a `BoundaryTrace`.

**Recommendation:** Option A. It is the smallest change that produces a real
observed root displacement, and it keeps `BoundaryTrace` fully populated with
cited evidence. The state stride must be set so at least the boundary-start and
chunk-end frames are logged (stride divides `steps_per_chunk`).

### 5.4 Timestamp monotonicity

All eight timestamps use a single monotonic clock (`time.monotonic_ns`, matching
`operator_x11.py:132` and `coordinator.py:601`). Because the adapter performs
generate → commit → publish → release strictly in order, and
`input_observed_ns ≤ sampled_ns` by construction (mailbox samples observed_ns
before returning, `operator_x11.py:184`), the non-decreasing constraint
(`boundary_trace.py:102-104`) is satisfied by real ordering, not by clamping.

---

## 6. Test-first steps (RED → GREEN), Stage R1 wiring only

Task numbering is commit order. Every test is CPU-only and uses the injected
test-double idiom already established in `tests/python/test_sonic_coordinator.py`
(fake `mm`, `gate`, `publisher`, `timeline`, `run`) and
`tests/python/test_sonic_manual_demo.py` (`_BoundaryControlLoop`,
`_BoundarySimulator`, `_BoundaryGate`, `_consume_x11_boundary`).

### Task W1: `ManualChunkCommitter` adapter satisfies the scheduler contract

- **New module:** `sonic/python/mm_sonic/responsive_wiring.py`.
- **New test file:** `tests/python/test_sonic_responsive_wiring.py`.
  - `test_committer_commits_one_prefix_in_coordinator_order` — with fake `mm`,
    `timeline`, `publisher`, `gate`, drive `committer.run_one_chunk(command)`
    and assert the exact call order: `mm.generate` → `validator.validate_source`
    → `timeline.prepare` → `publisher.send`/`publish` → `mm.commit` →
    `timeline.commit` → `gate.release_steps`. Mirrors the ordering proven for
    the real coordinator (`coordinator.py:1190→1206→1217→1229`) and for
    `manual_demo.generate_and_publish` (`manual_demo.py:481→488→490→495→496`)
    plus the caller's release (`manual_demo.py:595`).
  - `test_stale_candidate_is_aborted_before_publish_and_releases_no_physics`
    (drives the **real** supersession branch). A `command_is_current` that
    returns `False` must cause `timeline.abort` + `mm.abort` and raise
    `CandidateSuperseded`, with `publisher.send`/`publish` **never called** and
    `gate.release_steps` **never called**. Asserts on the fake call logs.
  - `test_generation_failure_releases_no_physics` — a raising `mm.generate`
    propagates; `gate.release_steps` uncalled. (Physics-pause safety at the
    wiring layer.)
- **RED command:**
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_responsive_wiring -v`
- **Expected RED:** `ModuleNotFoundError: mm_sonic.responsive_wiring`, then
  `AssertionError` on ordering/abort until the adapter enforces §3.1.
- **GREEN command:** same as RED.
- **Commit boundary W1:** "feat: responsive chunk-committer adapter for scheduler."

Checklist:
- [ ] W1.1 Write the three RED tests using the coordinator/manual-demo fakes; confirm `ModuleNotFoundError`.
- [ ] W1.2 Implement `ManualChunkCommitter.run_one_chunk` with the §3.1 ordering and the `CandidateSuperseded` raise (imported from `coordinator`, `coordinator.py:74`).
- [ ] W1.3 Run GREEN; confirm order, abort-before-publish, and no-release-on-failure.
- [ ] W1.4 Commit.

### Task W2: Scheduler ↔ committer integration retries supersession only

- **Same test file.**
  - `test_scheduler_retries_same_chunk_index_on_supersession` — inject a fake
    committer whose first `run_one_chunk` raises `CandidateSuperseded` and whose
    second succeeds; assert the mailbox is re-sampled at the *same* `chunk_index`
    (spy on `sample_intent`) and that exactly one prefix is accepted. This
    exercises `responsive_scheduler.py:44-63` against the adapter.
  - `test_terminate_snapshot_returns_none_without_committing` — a mailbox whose
    `sample_intent` yields `IntentSnapshot(command=None)` makes
    `run_one_prefix` return `None` and never calls `committer.run_one_chunk`
    (`responsive_scheduler.py:46-47`).
- **RED/GREEN command:** same module command.
- **Expected RED:** `AssertionError` until the adapter raises the exact typed
  supersession the scheduler catches.
- **Commit boundary W2:** "test: scheduler+committer supersession/terminate integration."

Checklist:
- [ ] W2.1 Write the two integration RED tests.
- [ ] W2.2 Confirm they pass against the W1 adapter (no scheduler edit needed).
- [ ] W2.3 Commit.

### Task W3: BoundaryTrace emission with honest root evidence

- **Same test file.**
  - `test_committer_emits_boundary_trace_with_monotone_stamps_and_real_vectors`
    — assert the emitted `BoundaryTrace` has non-decreasing timestamps
    (`boundary_trace.py:102-104`) and that `requested_velocity_mujoco`,
    `requested_heading_mujoco_wxyz`, and
    `generated_virtual_root_displacement_mujoco` equal the command / prepared
    target values (§5.1). Uses a fake state-log reader returning known
    `pelvis_position_m` rows so `observed_mujoco_root_displacement` equals the
    injected end-minus-start (§5.3 Option A).
  - `test_trace_uses_generated_root_for_published_field_and_documents_it`
    — asserts `published_physical_root_displacement_mujoco` equals the generated
    target root (§5.2 Option A) when `prepared.target.buffer` is the published
    buffer, proving no zero vector is fabricated.
  - `test_trace_omitted_when_observed_root_unavailable` — when the state-log
    reader yields no bounding rows, the committer emits a partial
    `responsive-latency.jsonl` record and **does not** construct a
    `BoundaryTrace` with a zero observed vector (§5.3 Option B fallback).
- **RED/GREEN command:** same module command.
- **Commit boundary W3:** "feat: emit responsive boundary trace with cited root evidence."

Checklist:
- [ ] W3.1 Write the three RED trace tests with a fake state-log reader.
- [ ] W3.2 Implement trace assembly (§5) reading `state.jsonl` rows for observed root; wire `trace_sink`.
- [ ] W3.3 Run GREEN; confirm no zero-vector fabrication and monotone stamps.
- [ ] W3.4 Commit.

### Task W4: `manual_demo` `--responsive` opt-in wiring (default untouched)

- **Modify:** `sonic/python/mm_sonic/manual_demo.py` — add `--responsive`
  (`_parser`, `manual_demo.py:824-841`), a responsive arm inside the x11 block
  (`manual_demo.py:602-655`) that forces one preload chunk (§4.2), builds
  `ManualChunkCommitter` + `ResponsiveScheduler`, runs the §4.3 loop, and adds
  responsive trace/latency fields to the summary. Leave `_resolve_mode_defaults`
  defaults and the two-chunk path unchanged.
- **New tests in `tests/python/test_sonic_manual_demo.py`:**
  - `test_responsive_flag_defaults_false_and_is_opt_in` —
    `_parser().parse_args([]).responsive is False`;
    `_parser().parse_args(["--responsive"]).responsive is True`. Mirrors
    `test_onscreen_is_explicit_and_opt_in` (`test_sonic_manual_demo.py:46`).
  - `test_responsive_does_not_change_interactive_two_chunk_default` — assert
    `_parser().parse_args(["--mode","interactive"]).preload_chunks == 2`
    remains (guards `manual_demo.py:812`) and `responsive is False`.
  - `test_default_preload_chunks_unchanged` — the existing `:50`/`:69` tests
    still pass (regression guard for the untouched default).
- **RED command:**
  `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sonic_manual_demo -v`
- **Expected RED:** argparse/`AttributeError` — `--responsive` unknown.
- **GREEN command:** same as RED.
- **Commit boundary W4:** "feat: opt-in --responsive one-prefix interactive path."

Checklist:
- [ ] W4.1 Add the RED flag/default tests; confirm the argparse failure.
- [ ] W4.2 Add `--responsive`, the responsive x11 arm, and one-preload-chunk wiring; leave defaults untouched.
- [ ] W4.3 Extend the summary with trace/latency fields; run GREEN including the default-preservation regression tests.
- [ ] W4.4 Commit.

---

## 7. Required behavior coverage (each mapped to code + test)

- **Camera delivery.** The responsive loop must still deliver the synchronized
  camera before committing a prefix, using the exact consumed-ack semantics in
  `_consume_x11_boundary` (`manual_demo.py:342-370`): only an error string
  starting with `"camera data"` disables the camera once; other
  `ProcessProtocolError`s are fatal (`manual_demo.py:356-357`). Wiring: extract
  the camera block into a helper (or call `_consume_x11_boundary` with a no-op
  `generate_and_publish` for camera only) so the responsive loop reuses it
  verbatim. Covered by the existing camera tests
  (`test_sonic_manual_demo.py:368,416,467`), which must remain GREEN.
- **Terminate (X).** `sample_intent` yields `IntentSnapshot(command=None)` when
  terminate is latched (`operator_x11.py:196-202`); the scheduler returns `None`
  (`responsive_scheduler.py:46-47`); the responsive loop breaks (§4.3). Test:
  `test_terminate_snapshot_returns_none_without_committing` (W2).
- **Backward / lateral / stand.** These are command *values*, not new code
  paths: backward/lateral set signed `requested_velocity_mujoco`
  (`operator_x11.py:204-210`); stand forces `(0,0,0)` velocity and retains
  heading (`operator_x11.py:195,207`). MM direction correctness is already
  proven live (§1). The committer emits the generated-root displacement
  (§5.1), enabling the controller's dynamic tracking gate to score each.
- **Stand vs. neutral.** `_stand_command` (`manual_demo.py:207-213`) is used for
  the single preload chunk; the mailbox's stand latch
  (`operator_x11.py:174-177,195-197`) yields zero-velocity commands during
  control. No extra path.
- **Generation failure.** A raising `mm.generate` propagates through the adapter
  without releasing physics (Task W1
  `test_generation_failure_releases_no_physics`), and — because the scheduler
  only catches `CandidateSuperseded` (`responsive_scheduler.py:61`) — every
  other exception propagates unchanged (`responsive_scheduler.py:39`), so the
  boundary loop stops with physics paused (`gate.pause()` remains in effect from
  `manual_demo.py:580`; no `release_steps` was called).
- **Stale supersession.** The `command_is_current` predicate the scheduler binds
  (`responsive_scheduler.py:50-54`) uses
  `self._mailbox.current_revision <= bound_revision` — a newer locomotion
  revision (`operator_x11.py:170-172`) fails the predicate, and the adapter
  aborts the candidate before publish and raises `CandidateSuperseded` (Task W1).
  Camera-only updates retain the revision (`operator_x11.py:150-172`), so they
  never supersede a candidate. Test: W1 abort test + W2 retry test.
- **Trace/latency evidence.** §5; Task W3. Latency published per boundary:
  input→sample, MM generation (`mm_started_ns`..`mm_completed_ns`),
  commit, publish-ack, physics-release, first-frame; plus generated/observed
  root displacement. The Stage-R1 honest floor is one irrevocable 0.4 s chunk
  plus generation time (selected plan §7.1); this plan does not claim ≤200 ms.

---

## 8. Failure ordering and physics-pause invariant

The wiring preserves the coordinator's fail-closed ordering at the adapter
boundary:

1. **Pre-publication failures** (generate, validate, prepare, supersession):
   candidate aborted via `timeline.abort` + `mm.abort`
   (`manual_demo.py:492-494` cleanup; `coordinator.py:1146-1163` semantics),
   **no physics released**, gate stays paused (`manual_demo.py:580`).
2. **Publication failure**: same cleanup then re-raise (`manual_demo.py:491-494`);
   no commit, no release.
3. **Post-publication** (`mm.commit`, `timeline.commit`, `release_steps`): this
   is the irreversible boundary (`coordinator.py:1201-1243`). A failure here
   propagates; the adapter must not attempt a rollback (matching
   `coordinator.py:1201-1202` "no abort or timeline rollback at this point").
4. **Supersession** is the *only* retried outcome (`responsive_scheduler.py:61`);
   all others propagate and stop the loop with physics paused.

**Physics-pause safety is asserted by Task W1's
`test_generation_failure_releases_no_physics` and W1's supersession test
(`gate.release_steps` uncalled on both).** This is the wiring-layer analogue of
the coordinator's `_terminalize_failure` paused invariant
(`coordinator.py:890-930`, `_paused_where_possible` `:880-888`).

---

## 9. Rollback

- **Opt-in only.** `--responsive` defaults to `False` (§4.1). Without it,
  `run_demo` runs the exact HEAD code; the two-chunk interactive default
  (`manual_demo.py:812`) and the `_consume_x11_boundary` loop
  (`manual_demo.py:636-653`) are untouched. Reverting is "do not pass the flag."
- **Isolated new module.** `responsive_wiring.py` is additive; deleting it plus
  the `--responsive` arm restores HEAD behavior. No SONIC/GEAR policy code or
  weights are touched (`gear_sonic_deploy` binary and `runtime/*.onnx` unchanged,
  `manual_demo.py:411-423`).
- **No protocol change.** Stage R1 uses the existing 20-frame `target-chunk/v1`
  wire (`timeline.py` `_TARGET_ROWS`, `zmq_v1.py:66-72`); the 80 ms `v2`
  protocol variant is explicitly **out of scope** for this wiring milestone
  (selected plan §"Stage R2", gated behind Probe P0).

---

## 10. Observed vs. inferred (honesty ledger)

- **Observed from code:**
  - Scheduler contract and retry-only-on-supersession
    (`responsive_scheduler.py:26-63`).
  - `manual_demo` owns a coordinator-free transaction and custom bootstrap; no
    `Coordinator` is instantiated there (`manual_demo.py:473-580`;
    `grep Coordinator manual_demo.py` → only `SessionConfig`/`SourceValidator`).
  - Mailbox already exposes `sample_intent` + `current_revision`
    (`operator_x11.py:145,216`).
  - `snapshot()` and `AdvanceResult` expose **no** root pose
    (`process.py:581-586,1727-1744`; `gated_sim.py:527-536`).
  - The wire buffer carries only `joint_position` + `body_quat_w`, no root
    translation (`zmq_v1.py:66-72`; `timeline.py:171-185`).
  - `state.jsonl` per-step rows **do** carry `pelvis_position_m`/`qpos`
    (`gated_sim.py:1170-1188,497-509,448`) — the honest source for observed root
    displacement.
  - `virtual_root_position` is on the prepared `TargetChunk`, not on `.buffer`
    (`timeline.py:79,171-177`).
- **Inferred / decision points (marked, not fabricated):**
  - `published_physical_root_displacement_mujoco` has no independent wire
    channel at HEAD; §5.2 chooses the *cited* generated-root value (Option A)
    and forbids a zero vector.
  - `first_simulated_frame_ns` may equal `physics_released_ns` if the
    post-release step boundary is not separately observable; §5.3 sources the
    observed displacement from `state.jsonl`, not from `AdvanceResult`.
- **Deferred to controller-owned gates (not this plan):** the flat/stop/terrain
  dynamic SONIC tracking gates and Probe P0/P4 (selected plan §6), which require
  live GEAR/SONIC/MuJoCo this plan is forbidden to run.

---

## 11. Controller-owned verification

- **gate-001** — the controller-owned plan-quality verifier over this report at
  `docs/superpowers/plans/2026-07-20-g1-sonic-responsive-wiring-claude.md`.
  Pre-change proof was `FileNotFoundError` on this exact path (RED); creating
  this report is the first-progress artifact. The controller executes gate-001;
  this plan does not reconstruct or invoke it.
- **CPU test gates (worker-runnable during implementation):** the RED/GREEN
  `unittest` commands in §6 (`test_sonic_responsive_wiring`,
  `test_sonic_manual_demo`), plus regression of `test_sonic_coordinator`,
  `test_sonic_responsive_scheduler`, and `test_sonic_boundary_trace` to prove no
  existing behavior changed.
