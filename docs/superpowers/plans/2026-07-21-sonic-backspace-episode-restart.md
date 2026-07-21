# Sonic Backspace Episode Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a rising Backspace edge cleanly tear down the current visible Sonic episode and automatically launch a fresh, fully synchronized episode at the registered terrain start.

**Architecture:** Add Backspace as an X11-only control edge backed by a dedicated `threading.Event`, separate from the locomotion mailbox and `X` cancellation. Boundary loops convert that edge into an internal `OperatorRestartRequested` outcome; `run_demo` records the interrupted episode before its existing fail-closed cleanup, and `main` supervises sequential `run_demo` calls with fresh child processes and run bundles.

**Tech Stack:** Python 3.10, `ctypes`/libX11, `threading.Event`, JSON evidence through `RunBundle`, `unittest`, MuJoCo/GEAR/SONIC integration canary.

## Global Constraints

- Backspace is active only after `LIVE X11` readiness and is rising-edge triggered.
- Backspace restarts MM, streamed reference, GEAR, simulator, and input state together; pose-only reset is forbidden.
- The current MuJoCo window closes and a fresh one opens after the normal 20–30 second startup.
- `X` exits the complete demonstration and never aliases Backspace.
- Every episode uses a unique existing run directory; replacement output never appends to interrupted output.
- Startup, protocol, validation, teardown, and child-process failures are fatal and never become automatic retries.
- Backspace during pre-live startup, hot GEAR reset, automatic fall recovery, and policy-quality changes are out of scope.

---

### Task 1: Add a Separate Backspace Edge to Continuous X11 Input

**Files:**
- Modify: `sonic/python/mm_sonic/operator_x11.py:20-90,328-460`
- Test: `tests/python/test_sonic_operator_x11.py:70-205,400-560`

**Interfaces:**
- Consumes: existing `KEYSYMS`, `_ACTIONS`, `_X11_GRABBED_CONTROL_KEYS`, `KeyLevels`, and `ContinuousControlLoop(..., cancel_event: threading.Event | None)`.
- Produces: exact registry entry `"BACKSPACE": 0xFF08`; `ContinuousControlLoop(..., restart_event: threading.Event | None = None)`; rising-edge restart behavior independent of `cancel_event` and the locomotion mailbox.

- [ ] **Step 1: Write failing registry, grab, and key-state tests**

Add assertions that freeze the exact keysym and require Backspace in the passive-grab set, plus a `KeyLevels` acceptance test:

```python
def test_registry_has_exact_backspace_keysym(self) -> None:
    self.assertEqual(KEYSYMS["BACKSPACE"], 0xFF08)
    self.assertIn("BACKSPACE", _X11_GRABBED_CONTROL_KEYS)

def test_backspace_is_a_valid_non_locomotion_key(self) -> None:
    levels = KeyLevels(focused=True, pressed=frozenset({"BACKSPACE"}))
    state = normalized_state_from_pressed(levels.pressed)
    self.assertEqual(state, NormalizedControlState())
```

- [ ] **Step 2: Write failing edge-semantics tests**

Use the existing `FakeProvider` and `_mapper()` fixtures to prove one held press sets only the restart event and logs one DOWN transition; release followed by a new press must log a second DOWN transition:

```python
def test_backspace_rising_edge_sets_only_restart_event(self) -> None:
    cancel = threading.Event()
    restart = threading.Event()
    events: list[str] = []
    provider = FakeProvider([
        KeyLevels(focused=True, pressed=frozenset()),
        KeyLevels(focused=True, pressed=frozenset({"BACKSPACE"})),
        KeyLevels(focused=True, pressed=frozenset({"BACKSPACE"})),
    ])
    loop = ContinuousControlLoop(
        provider,
        _mapper(),
        event_sink=events.append,
        period_s=0.001,
        cancel_event=cancel,
        restart_event=restart,
    )
    with loop:
        self.assertTrue(loop.wait_for_sequence(3, timeout_s=1.0))
    self.assertTrue(restart.is_set())
    self.assertFalse(cancel.is_set())
    self.assertEqual(events.count("KEY BACKSPACE DOWN -> restart"), 1)
```

Add constructor validation requiring `restart_event` to be `None` or exactly a `threading.Event` instance.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest \
  tests.python.test_sonic_operator_x11.KeysymRegistryTests \
  tests.python.test_sonic_operator_x11.KeyLevelsTests \
  tests.python.test_sonic_operator_x11.ContinuousControlLoopTests \
  tests.python.test_sonic_operator_x11.X11GrabLifecycleTests -v
```

Expected: failures because `BACKSPACE` and the `restart_event` constructor parameter do not exist.

- [ ] **Step 4: Implement the minimal Backspace input contract**

Add the exact registry/action/grab entries and the independent event:

```python
KEYSYMS = {
    "X": 0x0078,
    "BACKSPACE": 0xFF08,
    "SPACE": 0x0020,
}

_ACTIONS = {
    "X": "terminate",
    "BACKSPACE": "restart",
    "SPACE": "stand",
}

_X11_GRABBED_CONTROL_KEYS = (
    "X",
    "BACKSPACE",
    "SPACE",
)
```

Extend `ContinuousControlLoop.__init__`:

```python
restart_event: threading.Event | None = None,
```

Validate it alongside `cancel_event`, save `self._restart_event`, and extend the rising-edge branch in `_process_transitions`:

```python
if key == "X" and self._cancel_event is not None:
    self._cancel_event.set()
if key == "BACKSPACE" and self._restart_event is not None:
    self._restart_event.set()
```

Do not add Backspace handling to `normalized_state_from_pressed` or `HoldenControlMapper`.

- [ ] **Step 5: Run focused and complete X11 tests and confirm GREEN**

Run the Step 3 command, then:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest \
  tests.python.test_sonic_operator_x11 -v
```

Expected: all X11 tests pass, including exact passive-grab cardinality updates.

- [ ] **Step 6: Commit the input edge**

```bash
git add sonic/python/mm_sonic/operator_x11.py \
  tests/python/test_sonic_operator_x11.py
git commit -m "feat: add Sonic Backspace restart edge"
```

---

### Task 2: Stop Both X11 Schedulers at a Restart Boundary

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py:90-130,516-675,930-1040`
- Test: `tests/python/test_sonic_manual_demo.py:45-140,1021-1210`

**Interfaces:**
- Consumes: Task 1's `ContinuousControlLoop(..., restart_event=event)` and a per-episode `threading.Event`.
- Produces: internal `class OperatorRestartRequested(Exception)`; `_raise_if_restart_requested(restart_event: threading.Event) -> None`; `_consume_x11_boundary(..., restart_event: threading.Event) -> X11BoundaryResult`; `_run_responsive_x11_loop(..., restart_event: threading.Event) -> CameraDeliveryState`.

- [ ] **Step 1: Write the failing pure restart-check test**

Import the new symbol from `manual_demo` and require no-op/raise behavior:

```python
def test_restart_check_raises_only_when_requested(self) -> None:
    event = threading.Event()
    _raise_if_restart_requested(event)
    event.set()
    with self.assertRaises(OperatorRestartRequested):
        _raise_if_restart_requested(event)
```

- [ ] **Step 2: Write the failing responsive no-commit test**

Extend `ResponsiveX11LoopTests` with an already-set event and a committer whose `run_one_chunk` raises if called:

```python
def test_responsive_restart_stops_before_sampling_or_commit(self) -> None:
    restart = threading.Event()
    restart.set()

    class _NeverCommitter:
        next_chunk = 7

        def run_one_chunk(self, command, *, command_is_current=None):
            raise AssertionError("restart must stop before commit")

    with self.assertRaises(OperatorRestartRequested):
        _run_responsive_x11_loop(
            control_loop=object(),
            committer=_NeverCommitter(),
            simulator=_BoundarySimulator(),
            chunks=3,
            camera_state=CameraDeliveryState(),
            event_sink=lambda _event: None,
            trace_sink=lambda _trace: None,
            session_id="session",
            camera_disabled_prefix="CAMERA DISABLED",
            restart_event=restart,
        )
```

Add a sibling default-boundary test that proves an already-set event prevents
mailbox access, camera delivery, MM generation, and physics release:

```python
def test_default_boundary_restart_stops_before_mailbox_sample(self) -> None:
    restart = threading.Event()
    restart.set()

    class _NeverControlLoop:
        @property
        def mailbox(self):
            raise AssertionError("restart must stop before mailbox access")

    with self.assertRaises(OperatorRestartRequested):
        _consume_x11_boundary(
            control_loop=_NeverControlLoop(),
            simulator=object(),
            gate=object(),
            generate_and_publish=lambda *_args, **_kwargs: self.fail(
                "restart must not generate"
            ),
            chunk_index=3,
            steps_per_chunk=40,
            preload_chunks=1,
            camera_state=CameraDeliveryState(),
            event_sink=lambda _event: None,
            control_prefix="CONTROL chunk=",
            camera_disabled_prefix="CAMERA DISABLED",
            restart_event=restart,
        )
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest \
  tests.python.test_sonic_manual_demo.ResponsiveX11LoopTests \
  tests.python.test_sonic_manual_demo.X11TargetReadinessTests -v
```

Expected: import/signature failures for the missing restart types and parameter.

- [ ] **Step 4: Implement the restart boundary**

Add the internal outcome and check near the other operator helpers:

```python
class OperatorRestartRequested(Exception):
    """Normal operator request to replace the complete live episode."""


def _raise_if_restart_requested(restart_event: threading.Event) -> None:
    if restart_event.is_set():
        raise OperatorRestartRequested
```

Create `restart = threading.Event()` next to the existing per-episode
`cancellation` and pass it into `ContinuousControlLoop`. Add
`restart_event: threading.Event` to `_consume_x11_boundary` and call the helper
as its first statement. Add the same parameter to `_run_responsive_x11_loop`
and call the helper as the first statement of every loop iteration, before
`scheduler.run_one_prefix`. Pass the event explicitly at every production and
test call site.

Update the live help text to include `Backspace restart` without changing the
existing WASD, camera, stand, or exit text.

- [ ] **Step 5: Run manual X11 and responsive suites and confirm GREEN**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_responsive_scheduler \
  tests.python.test_sonic_responsive_wiring -v
```

Expected: all tests pass; restart tests prove zero new candidate work after the checked boundary.

- [ ] **Step 6: Commit boundary handling**

```bash
git add sonic/python/mm_sonic/manual_demo.py \
  tests/python/test_sonic_manual_demo.py
git commit -m "feat: stop Sonic episodes on Backspace"
```

---

### Task 3: Record Interrupted Evidence and Supervise Fresh Episodes

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py:625-720,1040-1210,1274-1285`
- Test: `tests/python/test_sonic_manual_demo.py:1210-1455`

**Interfaces:**
- Consumes: Task 2's `OperatorRestartRequested`; existing `RunBundle.write_text`, `GatedSimulatorClient.snapshot() -> dict[str, object]`, `run_demo(namespace: argparse.Namespace) -> Path`, and `main(argv: list[str] | None = None) -> int`.
- Produces: `_write_episode_restart_outcome(bundle: object, *, episode_ordinal: int, generated_chunks: int, snapshot: Mapping[str, object]) -> dict[str, object]`; `run_demo(namespace, *, episode_ordinal: int = 1) -> Path`; sequential relaunch in `main`.

- [ ] **Step 1: Write failing evidence-schema tests**

Use `_FakeBundle` and an exact snapshot to freeze the record:

```python
def test_restart_outcome_is_exact_and_episode_local(self) -> None:
    with TemporaryDirectory() as tmp:
        bundle = _FakeBundle(Path(tmp))
        record = _write_episode_restart_outcome(
            bundle,
            episode_ordinal=2,
            generated_chunks=13,
            snapshot={
                "steps": 80,
                "sim_time_s": 0.4,
                "state_rows": 20,
                "contact_rows": 80,
            },
        )
        self.assertEqual(record, {
            "schema": "mm-sonic-episode-outcome/v1",
            "episode_ordinal": 2,
            "outcome": "operator_restart",
            "trigger_key": "BACKSPACE",
            "generated_chunks": 13,
            "committed_chunks": 13,
            "sim_time_s": 0.4,
        })
        self.assertIn("episode-outcome.json", bundle.written)
```

Add strict validation cases for Boolean/nonpositive ordinals, Boolean/negative
chunk counts, and missing/nonfinite `sim_time_s`.

- [ ] **Step 2: Write failing supervisor tests**

Patch `run_demo` with a deterministic two-call fake. The first call raises
`OperatorRestartRequested`; the second returns a path. Assert ordinals `[1, 2]`,
the same namespace object across calls, and `main(...) == 0`:

```python
def test_main_relaunches_exactly_after_operator_restart(self) -> None:
    calls = []

    def fake_run(namespace, *, episode_ordinal):
        calls.append((namespace, episode_ordinal))
        if episode_ordinal == 1:
            raise OperatorRestartRequested
        return Path("/fresh")

    with patch.object(manual_demo, "run_demo", side_effect=fake_run):
        self.assertEqual(main(["--chunks", "1"]), 0)
    self.assertEqual([ordinal for _, ordinal in calls], [1, 2])
    self.assertIs(calls[0][0], calls[1][0])
```

Add tests showing any other `ContractError`/`ProcessError` escapes after one
call, and normal completion performs one call:

```python
def test_main_does_not_retry_process_failure(self) -> None:
    with patch.object(
        manual_demo,
        "run_demo",
        side_effect=ProcessError("gear failed"),
    ) as run:
        with self.assertRaisesRegex(ProcessError, "gear failed"):
            main(["--chunks", "1"])
    run.assert_called_once()

def test_main_runs_one_episode_without_restart(self) -> None:
    with patch.object(
        manual_demo,
        "run_demo",
        return_value=Path("/complete"),
    ) as run:
        self.assertEqual(main(["--chunks", "1"]), 0)
    self.assertEqual(run.call_args.kwargs["episode_ordinal"], 1)
```

Add a run-level cleanup test using the existing fake constructor harness. Each
fake appends its name in `close()`. Force `OperatorRestartRequested` from the
first X11 boundary and assert, after `run_demo` re-raises, that the close log
contains `provider`, `simulator`/`gate`, `gear`, `publisher`, and `mm`, and that
`pgrep`-style replacement construction is not part of `run_demo`:

```python
with self.assertRaises(OperatorRestartRequested):
    manual_demo.run_demo(namespace, episode_ordinal=1)
self.assertIn("provider", close_log)
self.assertIn("publisher", close_log)
self.assertIn("mm", close_log)
self.assertTrue("gate" in close_log or "simulator" in close_log)
self.assertEqual(created_episode_ordinals, [1])
```

- [ ] **Step 3: Run focused evidence/supervisor tests and confirm RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest \
  tests.python.test_sonic_manual_demo.WriteEpisodeRestartOutcomeTests \
  tests.python.test_sonic_manual_demo.EpisodeSupervisorTests -v
```

Expected: failures because the writer, ordinal argument, and supervisor loop do not exist.

- [ ] **Step 4: Implement the exact episode record**

Add a pure validated writer:

```python
from collections.abc import Mapping

def _write_episode_restart_outcome(
    bundle: object,
    *,
    episode_ordinal: int,
    generated_chunks: int,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    if type(episode_ordinal) is not int or episode_ordinal <= 0:
        raise ContractError("episode ordinal must be a positive integer")
    if type(generated_chunks) is not int or generated_chunks < 0:
        raise ContractError("generated chunks must be a nonnegative integer")
    if type(snapshot) is not dict or "sim_time_s" not in snapshot:
        raise ContractError("restart snapshot must contain sim_time_s")
    sim_time_s = snapshot["sim_time_s"]
    if (
        type(sim_time_s) not in (int, float)
        or not math.isfinite(float(sim_time_s))
        or float(sim_time_s) < 0.0
    ):
        raise ContractError("restart snapshot sim_time_s must be finite and nonnegative")
    record = {
        "schema": "mm-sonic-episode-outcome/v1",
        "episode_ordinal": episode_ordinal,
        "outcome": "operator_restart",
        "trigger_key": "BACKSPACE",
        "generated_chunks": generated_chunks,
        "committed_chunks": generated_chunks,
        "sim_time_s": float(sim_time_s),
    }
    bundle.write_text(
        "episode-outcome.json",
        json.dumps(record, sort_keys=True, indent=2) + "\n",
    )
    return record
```

Use the repository's existing strict numeric validation style; do not accept
coercible strings or Booleans.

- [ ] **Step 5: Record before teardown and implement the supervisor loop**

Validate `episode_ordinal` at the start of `run_demo`. Initialize
`committer: ManualChunkCommitter | None = None` before the main transaction so
an interrupted responsive loop reports the committer's current chunk rather
than the stale pre-loop local. Then add an outer handler
before its existing `finally`:

```python
except OperatorRestartRequested:
    generated_chunks = (
        committer.next_chunk if committer is not None else next_chunk
    )
    _write_episode_restart_outcome(
        bundle,
        episode_ordinal=episode_ordinal,
        generated_chunks=generated_chunks,
        snapshot=simulator.snapshot(),
    )
    raise
finally:
    # retain the existing fail-closed cleanup verbatim
```

Change `main` to supervise only the explicit restart outcome:

```python
episode_ordinal = 1
while True:
    try:
        run_demo(namespace, episode_ordinal=episode_ordinal)
    except OperatorRestartRequested:
        print(
            f"RESTART BACKSPACE: episode {episode_ordinal} -> "
            f"{episode_ordinal + 1}",
            flush=True,
        )
        episode_ordinal += 1
        continue
    return 0
```

Do not catch `BaseException`, `ContractError`, `ProcessError`, or cleanup errors.

- [ ] **Step 6: Run the full manual-demo suite and confirm GREEN**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest \
  tests.python.test_sonic_manual_demo -v
```

Expected: all tests pass; supervisor tests prove strict sequential teardown and fresh ordinals.

- [ ] **Step 7: Commit episode supervision**

```bash
git add sonic/python/mm_sonic/manual_demo.py \
  tests/python/test_sonic_manual_demo.py
git commit -m "feat: restart complete Sonic episodes"
```

---

### Task 4: Protected Regression Gates and Visible Backspace Canary

**Files:**
- Modify only if a test exposes a defect: files from Tasks 1–3
- Verify: `tests/python/test_sonic_operator_x11.py`
- Verify: `tests/python/test_sonic_manual_demo.py`
- Verify: `tests/python/test_sonic_responsive_scheduler.py`
- Verify: `tests/python/test_sonic_responsive_wiring.py`
- Verify: `tests/python/test_sonic_gated_sim.py`

**Interfaces:**
- Consumes: complete Backspace restart implementation from Tasks 1–3 and the existing qualified verifier environment.
- Produces: protected-suite evidence plus a live process/window replacement canary on `DISPLAY=:1`.

- [ ] **Step 1: Run all affected unit suites**

Run:

```bash
VERIFIER_PY=/home/ubuntu/reliable-claude-projects/motion-matching/verifiers/stage-a-integration-env-py310-20260717/venv/bin/python
PYTHONPATH=sonic/python "$VERIFIER_PY" -B -m unittest \
  tests.python.test_sonic_operator_x11 \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_responsive_scheduler \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_gated_sim -v
```

Expected: all affected tests pass.

- [ ] **Step 2: Run the frozen protected viewer-isolation suites**

Run:

```bash
VERIFIER_PY=/home/ubuntu/reliable-claude-projects/motion-matching/verifiers/stage-a-integration-env-py310-20260717/venv/bin/python
EVALUATOR=/home/ubuntu/reliable-claude-projects/motion-matching/verifiers/sonic-viewer-key-isolation-20260721/evaluate.py
"$VERIFIER_PY" -B -I "$EVALUATOR" --repo . --suite x11
"$VERIFIER_PY" -B -I "$EVALUATOR" --repo . --suite gated
```

Expected: both evaluators print `pass` and exit zero. If the frozen X11
evaluator intentionally asserts the former exact key set, update a copied
qualified evaluator only after preserving its existing assertions and adding
the approved Backspace contract; do not weaken viewer-isolation checks.

- [ ] **Step 3: Run a visible full-restart canary**

Launch the existing terrain command with `--chunks 100000`, `--onscreen`,
`--responsive`, `--responsive-source-intervals 5`, scene
`grail-curb-default`, route `curb-forward`, terrain weight `4`, movement model
`holden-turn-v1`, and X11 input. Focus the new MuJoCo window and wait for:

```text
LIVE X11
SONIC first action ready
SONIC policy command received
```

Record the first `manual_demo`, `mm_chunk_server`, `gated_sim`, and GEAR PIDs and
the first MuJoCo window ID. Inject one real rising edge:

```bash
DISPLAY=:1 xdotool keydown BackSpace
sleep 0.2
DISPLAY=:1 xdotool keyup BackSpace
```

Expected terminal sequence:

```text
KEY BACKSPACE DOWN -> restart
RESTART BACKSPACE: episode 1 -> 2
```

The UP transition may be logged if the 50 Hz sampler observes release before
teardown; it is not required for the rising-edge restart contract.

Expected lifecycle: all first-episode child PIDs exit before replacement PIDs
appear; the first window closes; a new MuJoCo window opens; no overlapping GEAR,
MM, or simulator children exist.

- [ ] **Step 4: Verify replacement evidence, start pose, controls, and exit**

Require the interrupted directory's `episode-outcome.json` to match schema
`mm-sonic-episode-outcome/v1`, ordinal `1`, outcome `operator_restart`, and key
`BACKSPACE`. Require a distinct replacement run directory and fresh episode-2
state/contact files. Capture the replacement viewer and verify the pelvis/start
pose is upright on the same visible terrain.

Focus the replacement window, inject `W` down/up, and require exact transition
events plus increasing state rows. Inject `X` and require clean termination with
no third episode and no leaked project processes.

- [ ] **Step 5: Run final repository checks and commit any test-driven repair**

Run:

```bash
git diff --check
git status --short
```

Expected: clean worktree. If Steps 1–4 required a minimal repair, rerun every
command above, then commit only that repair and its regression test:

```bash
git add sonic/python/mm_sonic tests/python
git commit -m "fix: harden Sonic episode restart"
```
