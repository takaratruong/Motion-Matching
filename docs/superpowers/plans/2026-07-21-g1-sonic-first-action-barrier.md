# G1 SONIC First-Action Barrier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent live SONIC physics from starting until GEAR has produced and published the first policy action for the primed scored state.

**Architecture:** Add one fail-closed readiness method to `GearProcess` that reads the owned GEAR `action.csv` through the retained directory descriptor and validates the first policy-produced row. Add a small manual-demo startup transaction that resumes and activates GEAR, crosses that barrier while MuJoCo is paused, then creates and pauses the existing `SimulationPolicyGate` before any scored release.

**Tech Stack:** Python 3.10, `unittest`, POSIX process groups and directory file descriptors, official GEAR CSV logging, and the existing gated MuJoCo protocol.

## Global Constraints

- Do not edit or fork the official GEAR checkout.
- Do not enable or retain the elastic band in the scored epoch.
- Do not change MM chunks, target rows, the 0.2 s horizon, or command scheduling.
- The barrier must observe actual policy output; it must not use a fixed startup sleep.
- Missing, partial, malformed, non-finite, timed-out, or child-aborted action evidence must fail before physics release.
- Read only `action.csv` in the wrapper-owned GEAR logs directory without following a replacement symlink.
- Preserve the existing full-process-group SIGSTOP/SIGCONT lifecycle.
- Use test-first implementation and warning-strict verification.

---

### Task 1: Authenticate the first policy action

**Files:**
- Modify: `tests/python/test_sonic_process.py`
- Modify: `sonic/python/mm_sonic/process.py`

**Interfaces:**
- Consumes: an alive, resumed `GearProcess` whose `activate_control()` call has completed.
- Produces: `GearProcess.wait_for_first_policy_action() -> Mapping[str, object]` with immutable `index`, `time_ms`, `time_monotonic_ms`, and `action` values.

- [ ] **Step 1: Write the failing success-path test**

Add a PTY child fixture to `GearProcessTests` that completes the real WAIT and
stream-preload handshakes. After receiving the CONTROL key, it prints the
authenticated control marker and writes this file beneath the `--logs-dir`
argument:

```python
header = [
    "index", "time_ms", "time_realtime_ms", "time_monotonic_ms",
    "ros_timestamp", *[f"act_{index}" for index in range(29)],
]
initial = ["0", "0.000", "1.000", "2.000", "0.000000000", *(["0.0"] * 29)]
policy = ["1", "20.000", "21.000", "22.000", "0.000000000", *(["0.25"] * 29)]
with (logs_dir / "action.csv").open("w", encoding="ascii", newline="") as sink:
    sink.write(",".join(header) + "\n")
    sink.write(",".join(initial) + "\n")
    sink.flush()
    time.sleep(0.02)
    sink.write(",".join(policy) + "\n")
    sink.flush()
```

Drive the real lifecycle through `start_to_wait_for_control()`,
`enable_stream_for_preload()`, `stop_group()`, `continue_group()`, and
`activate_control()`. Assert:

```python
ready = gear.wait_for_first_policy_action()
self.assertEqual(ready["index"], 1)
self.assertEqual(ready["time_ms"], 20.0)
self.assertEqual(ready["time_monotonic_ms"], 22.0)
self.assertEqual(ready["action"], (0.25,) * 29)
with self.assertRaises(TypeError):
    ready["index"] = 2
```

- [ ] **Step 2: Run RED for the missing readiness API**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_process.GearProcessTests.test_wait_for_first_policy_action_returns_authenticated_row
```

Expected: FAIL with `AttributeError` because
`wait_for_first_policy_action` does not exist.

- [ ] **Step 3: Implement the minimum owned-log parser and wait loop**

In `process.py`, define the exact header once:

```python
_GEAR_ACTION_HEADER = (
    "index", "time_ms", "time_realtime_ms", "time_monotonic_ms",
    "ros_timestamp", *(f"act_{index}" for index in range(29)),
)
```

Add private parsing that accepts only complete ASCII lines, requires the exact
header, exact row widths, indices `0` then `1`, and finite numeric fields. Open
`action.csv` with `os.open("action.csv", os.O_RDONLY | os.O_NOFOLLOW,
dir_fd=self._owned_logs_directory.leaf_fd)`, require `stat.S_ISREG` from
`os.fstat`, read a bounded snapshot, and close the file descriptor in `finally`.
Return `None` while the file, initialization row, or complete policy row does
not yet exist; raise `ProcessError` immediately for a complete malformed row.

Implement:

```python
def wait_for_first_policy_action(self) -> Mapping[str, object]:
    if not self._control_active or not self.group_is_resumed():
        raise ProcessError(
            "first policy action requires resumed active GEAR control"
        )
    deadline = (
        None if self._readiness_timeout_s is None
        else time.monotonic() + self._readiness_timeout_s
    )
    while True:
        self.require_alive()
        evidence = self._read_first_policy_action()
        if evidence is not None:
            return evidence
        if _cancelled(self._cancelled):
            raise OperatorCancelled("operator cancelled first policy action wait")
        if deadline is not None and time.monotonic() >= deadline:
            raise ProcessError("timed out waiting for first GEAR policy action")
        time.sleep(
            self._readiness_poll_s
            if deadline is None
            else min(
                self._readiness_poll_s,
                max(0.0, deadline - time.monotonic()),
            )
        )
```

The private parser returns:

```python
MappingProxyType({
    "index": 1,
    "time_ms": numeric_prefix[1],
    "time_monotonic_ms": numeric_prefix[3],
    "action": tuple(action_values),
})
```

- [ ] **Step 4: Run GREEN for the success path**

Re-run the exact Step 2 command. Expected: PASS without warnings.

- [ ] **Step 5: Add fail-closed tests one behavior at a time**

Add focused tests using the same real lifecycle fixture. Name them
`test_first_policy_action_initial_row_alone_times_out`,
`test_first_policy_action_rejects_wrong_header`,
`test_first_policy_action_rejects_wrong_policy_index`,
`test_first_policy_action_rejects_nonfinite_action`,
`test_first_policy_action_child_death_wins_over_timeout`,
`test_first_policy_action_requires_resumed_active_control`, and
`test_first_policy_action_rejects_replaced_log_symlink`. Each fixture writes
the exact success CSV from Step 1 with only its named property changed, then
asserts the exact `ProcessError`, `ChildProcessDied`, or timeout category.

For each test, run it before changing production code and confirm the failure
names the missing validation. Then make only the parser/wait-loop change needed
for that case.

- [ ] **Step 6: Run the complete process suite**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_process
```

Expected: all process tests pass without warnings.

- [ ] **Step 7: Commit the authenticated barrier**

```bash
git add sonic/python/mm_sonic/process.py tests/python/test_sonic_process.py
git commit -m "feat: wait for SONIC first policy action"
```

### Task 2: Gate manual scored startup on action readiness

**Files:**
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_demo.py`

**Interfaces:**
- Consumes: `GearProcess.wait_for_first_policy_action()` from Task 1 and the existing `SimulationPolicyGate`.
- Produces: `_activate_scored_control(gear: object, simulator: object) -> SimulationPolicyGate`.

- [ ] **Step 1: Write the failing startup-order test**

Import `_activate_scored_control` in `test_sonic_manual_demo.py`. Add fake gear
and simulator objects that append each operation to one list. The fake gear
must expose `continue_group`, `activate_control`,
`wait_for_first_policy_action`, `stop_group`, `group_is_stopped`,
`group_is_resumed`, and `require_alive`; the simulator must expose
`sim_dt = 0.005` and `require_alive`.

Assert:

```python
gate = _activate_scored_control(gear, simulator)
self.assertIsInstance(gate, SimulationPolicyGate)
self.assertEqual(
    calls,
    [
        "gear.continue_group",
        "gear.activate_control",
        "gear.wait_for_first_policy_action",
        "gear.stop_group",
        "simulator.require_alive",
    ],
)
self.assertTrue(gate.is_paused)
```

Also make the fake barrier raise `ProcessError` and assert that no
`gear.stop_group` through a gate and no simulator advance occurs after the
barrier failure.

- [ ] **Step 2: Run RED for missing startup transaction**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_manual_demo.ScoredControlStartupTests
```

Expected: FAIL because `_activate_scored_control` does not exist.

- [ ] **Step 3: Implement the minimum startup transaction**

Add:

```python
def _activate_scored_control(
    gear: object, simulator: object
) -> SimulationPolicyGate:
    """Prepare one policy action before any scored physics release."""

    gear.continue_group()
    gear.activate_control()
    ready = gear.wait_for_first_policy_action()
    print(
        "SONIC first action ready: "
        f"index={ready['index']} policy_time={ready['time_ms']:.3f}ms",
        flush=True,
    )
    gate = SimulationPolicyGate(gear, simulator)
    gate.pause()
    return gate
```

Replace the current four statements in `run_demo()`:

```python
gear.continue_group()
gear.activate_control()
gate = SimulationPolicyGate(gear, simulator)
gate.pause()
```

with:

```python
gate = _activate_scored_control(gear, simulator)
```

- [ ] **Step 4: Run GREEN and focused regression suites**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_manual_demo tests.python.test_sonic_process \
  tests.python.test_sonic_responsive_wiring
```

Expected: all tests pass without warnings. Existing reset/prime tests must
continue proving `elastic_band_enabled=False` for the scored epoch.

- [ ] **Step 5: Commit the live wiring**

```bash
git add sonic/python/mm_sonic/manual_demo.py \
  tests/python/test_sonic_manual_demo.py
git commit -m "fix: synchronize SONIC scored startup"
```

### Task 3: Verify determinism and retain live evidence

**Files:**
- Verify only: repository and new run artifacts beneath `/home/ubuntu/mm-sonic-holden-turn-live/`

**Interfaces:**
- Consumes: Tasks 1-2 and the pinned `holden-turn-v1` MM server.
- Produces: warning-strict suite evidence, repeated no-input stability results, and one visible interactive terrain run.

- [ ] **Step 1: Run static and full-suite gates**

Run:

```bash
git diff --check
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  SONIC_PROJECT_CLI=/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver/sonic/build/g1_project_pose_cli \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest discover -v \
  -s tests/python -p 'test*.py'
```

Expected: all tests pass; optional unavailable-artifact tests remain explicit
skips. Confirm `git status --short` contains no unrelated changes.

- [ ] **Step 2: Run three bounded no-input trials**

Launch the existing 5-interval responsive terrain command headlessly or with
an isolated display for at least 10 simulated seconds per trial. For each run,
require the `SONIC first action ready` line before the first scored `advance`,
no simulator reset/freeze marker, pelvis height at least 0.2 m, and pelvis-up z
above the existing fall threshold for the full window.

- [ ] **Step 3: Compare startup timing and MM timing**

Record first policy-action timing for all trials. Confirm the barrier crosses
before scored time advances. Preserve the existing responsive trace fields and
verify MM generation remains within the previously observed range (median
about 1.27 ms, p95 about 4.21 ms); this patch must not widen MM computation or
the 0.2 s horizon.

- [ ] **Step 4: Launch the visible terrain demo**

Launch `holden-turn-v1` with `--onscreen --responsive
--responsive-source-intervals 5`. Confirm terrain remains visible, the robot
does not fall while standing through the bounded observation window, and W/A/S/D
commands still reach the matcher. Retain any freeze if a genuine later fall
occurs so its state can be diagnosed rather than reset.

- [ ] **Step 5: Request independent review**

Review the final diff against the design, with special attention to owned-path
handling, child-death precedence, no physics release before readiness, and the
absence of elastic-band or MM scheduling changes. Resolve every finding and
re-run the affected focused and full suites before reporting completion.
