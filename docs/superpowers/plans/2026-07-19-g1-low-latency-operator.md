# G1 Low-Latency Operator Preview Implementation Plan

**Goal:** Reduce interactive command latency by selecting the smallest safe
manual-demo preload depth while preserving the qualified four-chunk default.

**Architecture:** Parameterize only the non-scored manual demo's initial and
rolling queue depth. Keep the formal auditor fixed at four chunks, run real
physics at candidate depths, and choose from visual evidence before broader
verification.

---

### Task 1: Parameterize the manual queue depth

**Files:**

- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Test: `tests/python/test_sonic_manual_demo.py`

- [ ] Add RED parser tests for default `4`, explicit `1` and `2`, zero,
      negative, non-integer, and values above `4`.
- [ ] Add a small validated preload helper or parser validation that accepts
      only exact integers in `[1, 4]`.
- [ ] Replace every runtime use of `_PRELOAD_CHUNKS` with the validated selected
      value while retaining `_PRELOAD_CHUNKS = 4` as the default.
- [ ] Assert command artifacts, console latency, and summary latency derive
      from the selected value.
- [ ] Run the focused manual-demo and evidence parser tests.

### Task 2: Prove the qualified evidence boundary is unchanged

**Files:**

- Verify without changing: `sonic/python/mm_sonic/manual_evidence.py`
- Test: `tests/python/test_sonic_manual_evidence.py`

- [ ] Prove the formal auditor still rejects non-four-chunk evidence.
- [ ] Prove the default parser path still records four chunks and `1.6` seconds.
- [ ] Run manual demo/evidence, operator, terminal, and replay tests warning
      strict.
- [ ] Commit the repository change before real trials.

### Task 3: Run the two-chunk visual trial

- [ ] Ensure no other GEAR/Sonic trial is active.
- [ ] Run a fresh 30-chunk scripted physics trial with
      `--preload-chunks 2`.
- [ ] Preserve its run root and capture a replay/video or dense contact sheet.
- [ ] Inspect the actual trajectory before formal metrics.
- [ ] Compute state/contact/hand sanity metrics and record any underrun or
      controller diagnostic exactly.

### Task 4: Run the one-chunk visual trial

- [ ] Repeat Task 3 with `--preload-chunks 1` only if the two-chunk run passes.
- [ ] Preserve a failure unchanged if the controller underruns or motion looks
      wrong.
- [ ] Select one chunk only if visual and numeric checks both pass.

### Task 5: Expose and qualify the selected driver

- [ ] Update `/home/ubuntu/drive-g1-sonic.sh` to pass the selected depth
      explicitly and print the correct latency.
- [ ] Launch the interactive viewer, focus the terminal, and visually confirm
      the robot before handing controls to the user.
- [ ] Run the focused 107-test manual/viewer surface and the protected closed-
      hand evaluator.
- [ ] Record the selected depth, run/video paths, metrics, commit, and rollback
      point in a result note and fresh-session handoff.
