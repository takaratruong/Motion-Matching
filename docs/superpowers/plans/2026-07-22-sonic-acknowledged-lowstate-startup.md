# Sonic Acknowledged LowState Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the visible Sonic demo's post-focus LowState race by proving GEAR received a no-step reset-state observation before CONTROL activation.

**Architecture:** Add a simulation-only GEAR protocol-v5 `RESUME` acknowledgement after the existing `PAUSE`/`SYNC` fence. Consume that acknowledgement only in Motion Matching's X11 startup path, pin the exact resulting GEAR commit, then require automated cross-repository gates and a three-episode visible restart canary.

**Tech Stack:** C++20, GoogleTest, UNIX `SOCK_SEQPACKET`, Python 3.10, standard-library `unittest`, MuJoCo/CycloneDDS, git worktrees, Reliable Claude with Codex-owned protected gates.

## Global Constraints

- Base the GEAR work on exact clean commit `4a63412b034f1fef3e8e087adbb9be6cbeaaebdc` in a new branch/worktree; do not modify `/home/ubuntu/projects/gear-sonic-worktrees/simulation-control-gate`.
- Keep Motion Matching work on `/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver`, branch `research/g1-low-latency-driver`.
- Change the simulation-control capability exactly from `READY 4` to `READY 5`; Motion Matching must reject every other capability marker.
- The new wire request is exactly `RESUME <epoch>\n`; the success response is exactly `RESUMED <epoch> <uint32-tick>\n`.
- A third request argument fails with `malformed-RESUME`; a wrong epoch/state, missing LowState, pre-sync request, or duplicate request fails with `expected-SYNC` and makes the gate fatal.
- The successful X11 sequence is exactly focus, OS continue, PAUSE, begin SYNC, no-step LowState refresh, finish SYNC, RESUME, CONTROL, first-action evidence, received-command evidence, policy-gate pause.
- Do not add a sleep, retry, automatic crash restart, watchdog relaxation, elastic-band change, or pre-CONTROL MuJoCo step.
- Keep script-mode startup and later `PAUSE`/`SYNC`/`ARM` chunk releases behaviorally unchanged.
- Reliable Claude performs bulk implementation in two sequential terminal jobs. Codex owns job contracts, protected verifiers, diff review, integration, exact-SHA checks, and live acceptance.
- Do not run commands that create caches or build outputs inside a nonterminal Reliable Claude managed worktree. Run protected commands through its foreman, or after terminal handoff in a disposable projection.
- An automated `passed` job is a candidate only. Completion requires the visible three-episode canary in Task 5.

---

### Task 1: Add the acknowledged GEAR `RESUME` transition

**Files:**
- Modify in the new GEAR worktree: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/simulation_control_gate.hpp`
- Modify in the new GEAR worktree: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/simulation_control_gate.cpp`
- Modify in the new GEAR worktree: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_simulation_control_gate.cpp`

**Interfaces:**
- Consumes: protocol-v4 `PAUSE`, `SYNC`, and `ARM`; `ObserveLowState(uint32_t)`; `WorkerLease` admission.
- Produces: protocol-v5 `RESUME <epoch>` and `RESUMED <epoch> <tick>`; new internal state `State::kResuming`.

- [ ] **Step 1: Create and verify the isolated GEAR branch/worktree**

Use the worktree workflow to create branch `research/simulation-control-resume-v5` from `4a63412b034f1fef3e8e087adbb9be6cbeaaebdc` under `/home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5`. Then run:

```bash
git -C /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5 rev-parse HEAD
git -C /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5 status --short
git -C /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-gate status --short
```

Expected: the first command prints the exact base SHA and both status commands are empty.

- [ ] **Step 2: Write RED protocol tests**

Change every existing expected capability marker in `test_simulation_control_gate.cpp` to `READY 5`, then add these focused cases. The admitted-input case is the important concurrency proof:

```cpp
TEST(SimulationControlGate, AcknowledgedResumeDrainsInputAndOpensSameTick) {
  SocketPair sockets;
  sonic::SimulationControlGate gate(sockets.child, false);
  sockets.child = -1;
  ASSERT_EQ(ReceivePacket(sockets.parent), "READY 5\n");
  gate.ObserveLowState(123);
  SendPacket(sockets.parent, "PAUSE 1\n");
  ASSERT_EQ(ReceivePacket(sockets.parent), "PAUSED 1 123\n");
  SynchronizeGate(sockets.parent, gate, 1, 123);

  ASSERT_TRUE(gate.TryEnter(sonic::SimulationControlGate::Worker::kInput));
  SendPacket(sockets.parent, "RESUME 1\n");
  EXPECT_EQ(ReceivePacket(sockets.parent, 30), "");
  EXPECT_FALSE(gate.TryEnter(sonic::SimulationControlGate::Worker::kInput));
  EXPECT_FALSE(gate.TryEnter(sonic::SimulationControlGate::Worker::kControl));
  gate.Exit(sonic::SimulationControlGate::Worker::kInput);

  EXPECT_EQ(ReceivePacket(sockets.parent), "RESUMED 1 123\n");
  EXPECT_TRUE(gate.TryEnter(sonic::SimulationControlGate::Worker::kInput));
  gate.Exit(sonic::SimulationControlGate::Worker::kInput);
  EXPECT_TRUE(gate.TryEnter(sonic::SimulationControlGate::Worker::kControl));
  gate.Exit(sonic::SimulationControlGate::Worker::kControl);
}

TEST(SimulationControlGate, ResumeBeforeSyncFailsClosed) {
  SocketPair sockets;
  sonic::SimulationControlGate gate(sockets.child, false);
  sockets.child = -1;
  ASSERT_EQ(ReceivePacket(sockets.parent), "READY 5\n");
  gate.ObserveLowState(9);
  SendPacket(sockets.parent, "PAUSE 1\n");
  ASSERT_EQ(ReceivePacket(sockets.parent), "PAUSED 1 9\n");
  SendPacket(sockets.parent, "RESUME 1\n");
  EXPECT_EQ(ReceivePacket(sockets.parent), "ERROR 1 expected-SYNC\n");
  EXPECT_FALSE(gate.TryEnter(sonic::SimulationControlGate::Worker::kInput));
  EXPECT_FALSE(gate.TryEnter(sonic::SimulationControlGate::Worker::kControl));
}

TEST(SimulationControlGate, ResumeWithMarkerFailsClosed) {
  SocketPair sockets;
  sonic::SimulationControlGate gate(sockets.child, false);
  sockets.child = -1;
  ASSERT_EQ(ReceivePacket(sockets.parent), "READY 5\n");
  gate.ObserveLowState(10);
  SendPacket(sockets.parent, "PAUSE 1\n");
  ASSERT_EQ(ReceivePacket(sockets.parent), "PAUSED 1 10\n");
  SynchronizeGate(sockets.parent, gate, 1, 10);
  SendPacket(sockets.parent, "RESUME 1 20\n");
  EXPECT_EQ(ReceivePacket(sockets.parent), "ERROR 1 malformed-RESUME\n");
}

TEST(SimulationControlGate, DuplicateResumeFailsClosed) {
  SocketPair sockets;
  sonic::SimulationControlGate gate(sockets.child, false);
  sockets.child = -1;
  ASSERT_EQ(ReceivePacket(sockets.parent), "READY 5\n");
  gate.ObserveLowState(11);
  SendPacket(sockets.parent, "PAUSE 1\n");
  ASSERT_EQ(ReceivePacket(sockets.parent), "PAUSED 1 11\n");
  SynchronizeGate(sockets.parent, gate, 1, 11);
  SendPacket(sockets.parent, "RESUME 1\n");
  ASSERT_EQ(ReceivePacket(sockets.parent), "RESUMED 1 11\n");
  SendPacket(sockets.parent, "RESUME 1\n");
  EXPECT_EQ(ReceivePacket(sockets.parent), "ERROR 1 expected-SYNC\n");
}

TEST(SimulationControlGate, StaleResumeEpochFailsClosed) {
  SocketPair sockets;
  sonic::SimulationControlGate gate(sockets.child, false);
  sockets.child = -1;
  ASSERT_EQ(ReceivePacket(sockets.parent), "READY 5\n");
  gate.ObserveLowState(12);
  SendPacket(sockets.parent, "PAUSE 1\n");
  ASSERT_EQ(ReceivePacket(sockets.parent), "PAUSED 1 12\n");
  SynchronizeGate(sockets.parent, gate, 1, 12);
  SendPacket(sockets.parent, "RESUME 2\n");
  EXPECT_EQ(ReceivePacket(sockets.parent), "ERROR 2 expected-SYNC\n");
}
```

- [ ] **Step 3: Build and run RED**

Configure the isolated worktree if its build directory does not exist, then build and filter the unit binary:

```bash
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTensorRT_ROOT=/usr \
  -Donnxruntime_ROOT=/home/ubuntu/.local/share/motion-matching-deps/onnxruntime-linux-x64-1.16.3 \
  '-DTensorRT_FIND_COMPONENTS=nvinfer;nvinfer_plugin;nvonnxparser'
cmake --build gear_sonic_deploy/build --target run_tests --parallel "$(nproc)"
gear_sonic_deploy/target/release/run_tests --gtest_filter='SimulationControlGate.*'
```

Expected: compilation or tests fail because `READY 5` and `RESUME` are not implemented; preserve the first causal failure in the Reliable Claude evidence.

- [ ] **Step 4: Implement the minimal gate state and transition**

Add `kResuming` to `SimulationControlGate::State`. Emit `READY 5`. Add this request branch after `SYNC` and before `ARM`:

```cpp
if (verb == "RESUME") {
  if (stream_frame_end.has_value()) {
    FailLocked(epoch, "malformed-RESUME");
    return;
  }
  if (state_ != State::kPaused || epoch != current_epoch_ ||
      !has_lowstate_tick_ || !epoch_synchronized_) {
    FailLocked(epoch, "expected-SYNC");
    return;
  }
  state_ = State::kResuming;
  condition_.wait(lock, [this] {
    return state_ == State::kStopped || state_ == State::kFatal ||
           WorkersIdleLocked();
  });
  if (state_ != State::kResuming) return;
  const uint32_t tick = lowstate_tick_;
  epoch_synchronized_ = false;
  state_ = State::kRunning;
  SendLocked("RESUMED " + std::to_string(epoch) + " " +
             std::to_string(tick) + "\n");
  return;
}
```

Block new input admission during the drain:

```cpp
if (worker == Worker::kInput) {
  if (input_active_ || state_ == State::kArmed ||
      state_ == State::kResuming || state_ == State::kFatal ||
      state_ == State::kStopped) {
    return false;
  }
  input_active_ = true;
  return true;
}
```

Do not alter the `ARM`, changed-tick, or input-commit semantics.

- [ ] **Step 5: Run GREEN and the full GEAR unit binary**

```bash
cmake --build gear_sonic_deploy/build --target run_tests --parallel "$(nproc)"
gear_sonic_deploy/target/release/run_tests --gtest_filter='SimulationControlGate.*'
gear_sonic_deploy/target/release/run_tests
git diff --check
```

Expected: both test commands pass, `git diff --check` is silent, and only the three declared files are modified.

- [ ] **Step 6: Commit the independently reviewed GEAR change**

```bash
git add \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/simulation_control_gate.hpp \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/simulation_control_gate.cpp \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_simulation_control_gate.cpp
git commit -m "feat: acknowledge simulation control resume"
git rev-parse HEAD
```

Expected: one clean commit whose 40-character lowercase SHA becomes the only GEAR identity consumed by Tasks 2–5.

---

### Task 2: Consume protocol v5 in `GearProcess` and pin the exact GEAR commit

**Files:**
- Modify: `sonic/python/mm_sonic/process.py:93,2581-2675,3162-3270`
- Modify: `tests/python/test_sonic_process.py:2590-2760`
- Modify: `sonic/configs/gear_sonic.lock.json:4`
- Modify: `sonic/python/mm_sonic/gated_sim.py:38`
- Modify: `sonic/python/mm_sonic/metrics.py:34`
- Modify exact-SHA test fixtures that still name the old commit: `tests/python/test_sonic_scene.py`, `tests/python/test_sonic_metrics.py`

**Interfaces:**
- Consumes: Task 1's exact GEAR commit and wire response `RESUMED <epoch> <tick>`.
- Produces: `GearProcess.resume_simulation_control() -> None`; local state transition `paused+synchronized -> resuming -> running`.

- [ ] **Step 1: Write RED client protocol tests**

Update simulation-control fake children in `test_sonic_process.py` to send `READY 5`. Extend the main fake child with:

```python
elif packet == b"RESUME 1\n":
    channel.send(b"RESUMED 1 100\n")
```

After `finish_simulation_control_sync()`, assert:

```python
gear.resume_simulation_control()
self.assertFalse(gear.simulation_control_is_paused)
self.assertEqual(gear._simulation_control_state, "running")
self.assertFalse(gear._simulation_control_synchronized)
self.assertEqual(gear._simulation_control_tick, 100)
```

Add a malformed-response case whose fake child returns `RESUMED 2 100\n`; assert `ProcessProtocolError`, local state `unknown`, and synchronized flag false. Add a local precondition case that calls the method from `running` and from unsynchronized `paused`; each must raise `ProcessError` without sending a packet.

Add a prepared-WAIT pause case by starting the fake GEAR process, setting its already-tested preparation flags to `_wait_for_control_ready = True`, `_input_prepared = True`, `_control_active = False`, and asserting `pause_simulation_control()` sends `PAUSE 1\n` successfully. A separate case with `_input_prepared = False` must raise before sending.

- [ ] **Step 2: Run RED**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_process.GearProcessTests
```

Expected: failure because `READY 5` and `resume_simulation_control` are absent. Record the causal failing test, not unrelated later failures.

- [ ] **Step 3: Implement protocol-v5 client state handling**

Set:

```python
_SIMULATION_CONTROL_READY = ("READY", "5")
```

Change the mismatch text to `GEAR simulation control capability is not READY/v5`. Permit pause only when either active CONTROL or an authenticated prepared WAIT epoch exists:

```python
if not self._control_active and not (
    self._wait_for_control_ready and self._input_prepared
):
    raise ProcessError(
        "GEAR simulation control gate requires active policy control "
        "or prepared WAIT_FOR_CONTROL"
    )
```

Add immediately after `finish_simulation_control_sync`:

```python
def resume_simulation_control(self) -> None:
    """Resume fenced policy workers without requiring a changed physics tick."""

    if not self.simulation_control_gate:
        raise ProcessError("GEAR simulation control gate is not enabled")
    if self._simulation_control_state != "paused":
        raise ProcessError("GEAR simulation control must be paused before resume")
    if not self._simulation_control_synchronized:
        raise ProcessError(
            "GEAR simulation control must be synchronized before resume"
        )
    epoch = self._simulation_control_epoch
    self._simulation_control_state = "resuming"
    self._simulation_control_synchronized = False
    try:
        self._send_simulation_control_packet("RESUME", epoch)
        tick = self._wait_simulation_control_packet("RESUMED", epoch)
    except BaseException:
        self._simulation_control_state = "unknown"
        raise
    assert tick is not None
    self._simulation_control_tick = tick
    self._simulation_control_state = "running"
```

Do not special-case `RESUMED` in `_wait_simulation_control_packet`; its existing exact three-field epoch/uint32 parser is the intended implementation.

- [ ] **Step 4: Replace every pinned old GEAR identity with Task 1's exact commit**

First capture and validate the source identity:

```bash
GEAR_COMMIT="$(git -C /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5 rev-parse HEAD)"
case "$GEAR_COMMIT" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) exit 1 ;;
esac
```

Use that exact literal in `gear_sonic.lock.json`, `gated_sim.py`, `metrics.py`, and the two exact-SHA fixtures. Leave `current_frame_advancement_sha256` unchanged because Task 1 does not modify `g1_deploy_onnx_ref.cpp`. Then require:

```bash
rg -n '4a63412b034f1fef3e8e087adbb9be6cbeaaebdc' sonic tests/python
```

Expected: no matches.

- [ ] **Step 5: Run GREEN and identity regression tests**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_process \
  tests.python.test_sonic_external \
  tests.python.test_sonic_gated_sim \
  tests.python.test_sonic_metrics \
  tests.python.test_sonic_scene
git diff --check
```

Expected: all selected tests pass and the diff check is silent.

- [ ] **Step 6: Commit the client protocol and pin update**

```bash
git add sonic/python/mm_sonic/process.py \
  tests/python/test_sonic_process.py \
  sonic/configs/gear_sonic.lock.json \
  sonic/python/mm_sonic/gated_sim.py \
  sonic/python/mm_sonic/metrics.py \
  tests/python/test_sonic_scene.py \
  tests/python/test_sonic_metrics.py
git commit -m "feat: consume GEAR resume acknowledgement"
```

Expected: one focused Motion Matching commit and a clean worktree.

---

### Task 3: Synchronize post-focus LowState before X11 CONTROL activation

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py:450-485`
- Modify: `tests/python/test_sonic_manual_demo.py:590-790`

**Interfaces:**
- Consumes: `GearProcess.pause_simulation_control()`, `begin_simulation_control_sync()`, `finish_simulation_control_sync()`, `resume_simulation_control()`, and `GatedSimulatorClient.refresh_low_state()`.
- Produces: exact X11 startup ordering while preserving the no-barrier script path.

- [ ] **Step 1: Extend startup fakes and write the RED order test**

Add these methods to `ScoredControlStartupTests._Gear`:

```python
def begin_simulation_control_sync(self):
    self.calls.append("gear.begin_simulation_control_sync")
    self.control_paused = False

def finish_simulation_control_sync(self):
    self.calls.append("gear.finish_simulation_control_sync")
    self.control_paused = True

def resume_simulation_control(self):
    self.calls.append("gear.resume_simulation_control")
    self.control_paused = False
```

Add this method to `ScoredControlStartupTests._Simulator`:

```python
def refresh_low_state(self):
    self.calls.append("simulator.refresh_low_state")
    return {"published": True, "steps": 0}
```

Replace the existing input-readiness assertion with the complete expected order:

```python
self.assertEqual(
    calls,
    [
        "input.ready",
        "gear.continue_group",
        "gear.pause_simulation_control",
        "gear.begin_simulation_control_sync",
        "simulator.refresh_low_state",
        "gear.finish_simulation_control_sync",
        "gear.resume_simulation_control",
        "gear.activate_control",
        "gear.wait_for_first_policy_action",
        "gear.wait_for_received_policy_command",
        "gear.pause_simulation_control",
        "simulator.require_alive",
    ],
)
self.assertEqual(simulator.advance_calls, 0)
```

Keep `test_action_readiness_precedes_control_channel_gate_pause` as the script-path regression; its original call order must remain unchanged.

- [ ] **Step 2: Add RED fail-closed boundary tests**

For each of these methods—pause, begin sync, refresh, finish sync, resume, and activate—configure the fake to raise `ProcessError` and assert that every later method is absent from `calls`, both readiness strings are absent from captured stdout, and `simulator.advance_calls == 0`:

```python
with self.assertRaisesRegex(ProcessError, "synthetic startup boundary"):
    with redirect_stdout(output):
        _activate_scored_control(
            gear,
            simulator,
            before_control=lambda: calls.append("input.ready"),
        )
self.assertNotIn("SONIC first action ready", output.getvalue())
self.assertNotIn("SONIC policy command received", output.getvalue())
self.assertEqual(simulator.advance_calls, 0)
```

Use `subTest(method=method_name)` so all six boundaries are independently reported.

- [ ] **Step 3: Run RED**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_manual_demo.ScoredControlStartupTests
```

Expected: the X11 order test fails because the synchronization calls are absent.

- [ ] **Step 4: Implement the X11-only transaction**

Change `_activate_scored_control` to:

```python
if before_control is not None:
    before_control()
gear.continue_group()
if before_control is not None:
    gear.pause_simulation_control()
    gear.begin_simulation_control_sync()
    simulator.refresh_low_state()
    gear.finish_simulation_control_sync()
    gear.resume_simulation_control()
gear.activate_control()
```

Leave the existing action evidence, received-command evidence, policy-gate construction, and final `gate.pause()` after this block byte-for-byte unchanged.

- [ ] **Step 5: Run GREEN and affected lifecycle suites**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_process \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_operator_x11
git diff --check
```

Expected: all selected tests pass; script-mode assertions remain unchanged; no bytecode/cache files appear.

- [ ] **Step 6: Commit the startup integration**

```bash
git add sonic/python/mm_sonic/manual_demo.py tests/python/test_sonic_manual_demo.py
git commit -m "fix: acknowledge fresh LowState before Sonic control"
```

Expected: one focused commit and a clean worktree.

---

### Task 4: Build exact binaries and run protected cross-repository qualification

**Files:**
- Create ignored GEAR build outputs under the Task 1 worktree: `gear_sonic_deploy/build/**`, `gear_sonic_deploy/target/release/**`
- Create ignored Motion Matching build output in a directory whose suffix is the exact 12-character Motion Matching HEAD printed in Step 3.
- Create controller-owned evidence outside candidate worktrees: `/home/ubuntu/reliable-claude-projects/motion-matching/evidence/sonic-ack-lowstate-20260722/**`

**Interfaces:**
- Consumes: terminal reviewed commits from Tasks 1–3.
- Produces: exact GEAR and Motion Matching binaries plus protected automated evidence authorizing a live canary.

- [ ] **Step 1: Verify both repository identities and clean tracked state**

```bash
git -C /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5 status --short
git -C /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5 rev-parse HEAD
git -C /home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver status --short
git -C /home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver rev-parse HEAD
```

Expected: both status outputs are empty. The GEAR SHA exactly matches all Motion Matching pins.

- [ ] **Step 2: Build and authenticate the exact GEAR executable**

```bash
cmake --build gear_sonic_deploy/build --target run_tests g1_deploy_onnx_ref --parallel "$(nproc)"
gear_sonic_deploy/target/release/run_tests
file gear_sonic_deploy/target/release/g1_deploy_onnx_ref
sha256sum gear_sonic_deploy/target/release/g1_deploy_onnx_ref
ldd gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```

Expected: tests pass; the executable is an x86-64 ELF; `ldd` contains no `not found`. Save the SHA-256 and build ID in controller-owned evidence.

- [ ] **Step 3: Build the exact Motion Matching server into a new immutable path**

From the Motion Matching worktree:

```bash
MOTION_SHORT_SHA="$(git rev-parse --short=12 HEAD)"
cmake -S . -B "/home/ubuntu/mm-sonic-ack-lowstate-build-${MOTION_SHORT_SHA}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "/home/ubuntu/mm-sonic-ack-lowstate-build-${MOTION_SHORT_SHA}" \
  --target mm_chunk_server --parallel "$(nproc)"
sha256sum "/home/ubuntu/mm-sonic-ack-lowstate-build-${MOTION_SHORT_SHA}/mm_chunk_server"
```

Expected: a regular executable built from the exact Motion Matching HEAD; do not repoint or trust `sonic/build`.

- [ ] **Step 4: Run the full affected Python catalog and protected X11 verifier**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_process \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_gated_sim \
  tests.python.test_sonic_metrics \
  tests.python.test_sonic_external \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_operator_x11
python /home/ubuntu/reliable-claude-projects/motion-matching/verifiers/sonic-backspace-final-fix-20260722/evaluate_x11.py --repo .
```

Expected: zero failures. The protected gate confirms X11 key isolation/restart behavior and the affected catalog confirms protocol/startup order. Do not edit the protected verifier.

- [ ] **Step 5: Perform Codex diff review and anti-workaround scan**

```bash
git diff 4a63412b034f1fef3e8e087adbb9be6cbeaaebdc..HEAD -- \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/simulation_control_gate.hpp \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/simulation_control_gate.cpp \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_simulation_control_gate.cpp
git -C /home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver \
  diff 9a49872..HEAD
rg -n "sleep|retry|watchdog|elastic.band|advance\(" \
  sonic/python/mm_sonic/manual_demo.py sonic/python/mm_sonic/process.py
```

Expected: every relevant new line implements the approved fence; there is no new sleep, retry, watchdog change, elastic-band change, or startup physics advance. Existing unrelated matches are classified by unchanged blame, not deleted.

---

### Task 5: Pass the visible three-episode terrain canary

**Files:**
- Create runtime artifacts under a new root: `/home/ubuntu/mm-sonic-ack-lowstate-live-20260722/**`
- Create terminal transcript: `/tmp/g1-sonic-ack-lowstate-live.log`
- Preserve each episode's `episode-outcome.json`, GEAR stdout/stderr, simulator transcript, and scored logs.

**Interfaces:**
- Consumes: exact authenticated binaries from Task 4 and the existing terrain-aware X11 command.
- Produces: human-visible and artifact-backed proof that initial startup plus two Backspace restarts cross the acknowledged LowState fence.

- [ ] **Step 1: Prove launch quiescence without touching unrelated viewers**

List processes owned by the new run root and exact new binary paths. Require none. Record but do not kill the pre-existing unrelated Holden viewer PID/window identified before this plan.

- [ ] **Step 2: Launch episode 1 with explicit exact paths**

Use the qualified interactive defaults (`grail-curb-default`, `curb-forward`, terrain weight `4.0`) with responsive five-interval control and the capped Holden movement model. Launch from the Motion Matching worktree with exact paths computed from repository identity:

```bash
MOTION_SHORT_SHA="$(git rev-parse --short=12 HEAD)"
MM_SERVER="/home/ubuntu/mm-sonic-ack-lowstate-build-${MOTION_SHORT_SHA}/mm_chunk_server"
test -x "$MM_SERVER"
env DISPLAY=:1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m mm_sonic.manual_demo \
  --mode interactive \
  --input-source x11 \
  --onscreen \
  --responsive \
  --responsive-source-intervals 5 \
  --movement-model holden-turn-v1 \
  --chunks 100000 \
  --mm-server "$MM_SERVER" \
  --gear-checkout /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-resume-v5 \
  --output-root /home/ubuntu/mm-sonic-ack-lowstate-live-20260722 \
  2>&1 | tee /tmp/g1-sonic-ack-lowstate-live.log
```

The binary suffix is computed from the exact Task 4 Motion Matching HEAD, not a symbolic symlink.

- [ ] **Step 3: Qualify episode 1 and request restart**

Focus the new terrain viewer. Require, in order, `LIVE X11`, `SONIC first action ready`, and `SONIC policy command received`. Exercise W/A/S/D briefly and confirm the process remains alive. Press and release Backspace once. Require `RESTART BACKSPACE: episode 1 -> 2` and complete teardown of episode 1 before the next GEAR/simulator pair starts.

- [ ] **Step 4: Qualify episode 2 and request a second restart**

Focus the replacement terrain viewer at the registered start. Again require both readiness markers and a healthy process. Press and release Backspace once. Require `RESTART BACKSPACE: episode 2 -> 3`, a distinct run/session directory, and no overlapping episode-2 processes.

- [ ] **Step 5: Qualify episode 3 and exit normally**

Focus the third terrain viewer at the same registered start. Require both readiness markers, exercise one locomotion command, then press X. Require a normal episode-supervisor exit with no crash conversion or automatic retry.

- [ ] **Step 6: Verify artifacts, forbidden errors, and survivor count**

```bash
rg -n "Lost LowState|Safety check failed|ChildProcessDied|Traceback" \
  /tmp/g1-sonic-ack-lowstate-live.log \
  /home/ubuntu/mm-sonic-ack-lowstate-live-20260722
rg -n "RESTART BACKSPACE: episode 1 -> 2|RESTART BACKSPACE: episode 2 -> 3|SONIC first action ready|SONIC policy command received" \
  /tmp/g1-sonic-ack-lowstate-live.log
find /home/ubuntu/mm-sonic-ack-lowstate-live-20260722 -name episode-outcome.json -print
```

Expected: the forbidden-error search has no matches; exactly two restart transitions exist; each of three episodes has both readiness markers; both interrupted episodes preserve `operator_restart` outcomes; episode 3 exits normally. Confirm zero surviving processes whose command lines reference the new run root, GEAR worktree, or Motion Matching binary.

- [ ] **Step 7: Record final evidence without claiming locomotion quality**

Record both repository SHAs, both executable SHA-256 values, test transcripts, three episode directories, restart outcomes, readiness-marker positions, forbidden-error result, and survivor check. Report only startup/restart reliability. Do not interpret terrain clearance, walking quality, or depth readiness from this canary.
