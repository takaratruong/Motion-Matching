# SONIC Elastic-Band Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the virtual lifting band on only during unscored GEAR cold-start and prove it is off for every scored physics epoch.

**Architecture:** Add one required `elastic_band_enabled: bool` value to the existing gated-simulator reset path. Apply it to the pinned external band's `enable` field immediately before each reset, echo it in reset evidence, and make the CLI select `True` for bootstrap and `False` for scored reset.

**Tech Stack:** Python 3.10, `unittest`, NumPy, strict JSONL protocol, pinned external GEAR/MuJoCo adapter.

## Global Constraints

- Do not alter MM references, SONIC model inputs, gains, scene geometry, timing, or metric thresholds.
- Do not change the pinned external checkout.
- Treat missing or non-boolean band state as a protocol error.
- Preserve exact request/response object validation and run-local path confinement.
- Write and run each focused RED before production code.

---

### Task 1: Backend and server reset contract

**Files:**
- Modify: `sonic/python/mm_sonic/gated_sim.py`
- Test: `tests/python/test_sonic_gated_sim.py`

**Interfaces:**
- Consumes: `elastic_band_enabled: bool` on `GatedSimulatorRunner.reset`.
- Produces: `SimulatorBackend.reset_from_qpos(..., elastic_band_enabled: bool)` and reset response field `elastic_band_enabled: bool`.

- [ ] **Step 1: Write failing backend and protocol tests**

Add focused cases that require two consecutive resets to pass `True` then
`False` to a fake backend, reject `0`, `1`, `None`, and missing JSON fields,
and verify `ExternalGearBackend` changes an existing fake band's `enable`
attribute before completing reset.

- [ ] **Step 2: Run the focused RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python tests/python/test_sonic_gated_sim.py
```

Expected: the new cases fail because reset has no explicit band argument or
response evidence.

- [ ] **Step 3: Implement the minimal server/backend change**

Thread the required boolean through the abstract backend, runner, and exact
JSONL reset request. In `ExternalGearBackend.reset_from_qpos`, require the
pinned `sim_env.elastic_band` object, assign its `enable` field, then perform
the existing MuJoCo reset. Echo the applied boolean in the runner response.

- [ ] **Step 4: Run the focused GREEN**

Run the Task 1 command again. Expected: all gated-simulator tests pass without
warnings.

### Task 2: Strict client transport

**Files:**
- Modify: `sonic/python/mm_sonic/process.py`
- Test: `tests/python/test_sonic_process.py`

**Interfaces:**
- Consumes: `GatedSimulatorClient.reset(..., elastic_band_enabled: bool)`.
- Produces: an exact reset request containing the boolean and a validated reset
  result containing the same boolean.

- [ ] **Step 1: Write the failing client test**

Extend the tiny JSONL child fixture to assert the request contains a real JSON
boolean, echo it in reset data, and add a response-mismatch case that the client
rejects.

- [ ] **Step 2: Run the focused RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python tests/python/test_sonic_process.py \
  GatedSimulatorClientTests
```

Expected: failure because the client neither sends nor validates band state.

- [ ] **Step 3: Implement exact transport and echo validation**

Require a strict Python `bool`, include it in `_request("reset", ...)`, include
the field in the exact reset-data schema, and raise `ProcessProtocolError` if
the response value is not the requested boolean.

- [ ] **Step 4: Run the focused GREEN**

Run the Task 2 command again. Expected: all client tests pass.

### Task 3: Bootstrap/scored lifecycle and evidence

**Files:**
- Modify: `sonic/python/mm_sonic/cli.py`
- Test: `tests/python/test_sonic_timing.py`
- Test: `tests/python/test_sonic_cli.py`

**Interfaces:**
- Consumes: the strict client reset argument and echoed response.
- Produces: bootstrap evidence with `elastic_band_enabled: true` and scored
  prime evidence with `elastic_band_enabled: false`.

- [ ] **Step 1: Write failing lifecycle tests**

Require `_execute_known_good_scoring_epoch` to make its first reset with
`elastic_band_enabled=True`. Require `_reset_and_prime_scored_epoch` to reset
with `elastic_band_enabled=False`, retain the echoed response, and advance only
after that reset. Assert generated gate evidence retains both values.

- [ ] **Step 2: Run the focused RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python tests/python/test_sonic_timing.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python tests/python/test_sonic_cli.py
```

Expected: new assertions fail because existing reset calls do not select or
record band state.

- [ ] **Step 3: Implement the two-state lifecycle**

Store the bootstrap reset response in the existing bootstrap evidence mapping,
pass `True` on that call, and pass `False` in
`_reset_and_prime_scored_epoch`. Preserve all existing stop/prime/control
ordering and counters.

- [ ] **Step 4: Run the focused GREEN**

Run both Task 3 commands again. Expected: both modules pass.

### Task 4: Protected verification and handoff

**Files:**
- Modify only files listed in Tasks 1-3.

**Interfaces:**
- Consumes: the complete lifecycle diff.
- Produces: a candidate that passes the external evaluator and is ready for a
  real known-good smoke run outside the managed worktree.

- [ ] **Step 1: Run the controller-owned evaluator**

The Reliable Claude controller runs:

```bash
/home/ubuntu/reliable-claude-projects/motion-matching/verifiers/stage-a-integration-env-py310-20260717/venv/bin/python \
  -B -I \
  /home/ubuntu/reliable-claude-projects/motion-matching/evaluators/evaluate_sonic_elastic_band_lifecycle.py \
  --repo .
```

Expected: `status=pass` with bootstrap enabled and scored disabled.

- [ ] **Step 2: Run the affected regression set**

Run the four focused modules above plus `git diff --check`. Expected: all pass
with no warning or unexpected changed path.

- [ ] **Step 3: Stop at verified candidate**

Do not run the external GEAR binary inside a nonterminal managed worktree and
do not claim locomotion success. Codex will integrate the terminal candidate,
run the real known-good smoke, render it, and inspect the montage.
