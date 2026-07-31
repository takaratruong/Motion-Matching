# G1 Local Contact-Segment Retargeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make recorded terrain contact segments reusable on local query geometry for forward, downhill, lateral, and diagonal commands without unbounded rescue stalls.

**Architecture:** Preserve the existing local 95-value terrain search feature and authoritative whole-segment FK validation. Remove absolute scene-coordinate placement from the runtime path, align candidate body yaw to commanded facing, compare rotated source velocity with commanded travel, and bound ranked rescue. Inject the same policy into the existing 21-route evaluation harness.

**Tech Stack:** Python 3, PyTorch/CUDA, NumPy, MuJoCo FK, unittest.

## Global Constraints

- Work only in `/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver` on `research/g1-torch-terrain-kinematics`.
- Preserve untracked `build/` and `sonic/configs/experiments/torch_grail_representative.json`.
- Do not connect Sonic, tracking, physics, or learned depth.
- Write and observe each failing unit test before changing production code.
- Do not launch the viewer until real-data automated gates pass.

---

### Task 1: Local placement and independent travel/facing

**Files:**
- Modify: `sonic/python/mm_sonic/torch_contact_segments.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_contact_segments.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Consumes: `TerrainContactSegmentPolicy.resolve_entry(...)`, `ShapedCommand.trajectory`, `_ComposedCandidate`.
- Produces: local support-foot placement, candidate yaw from commanded facing, and travel-direction compatibility.

- [ ] **Step 1: Write failing placement and command tests**

Add a contact-policy test that uses identical local source/query surfaces with different absolute alignment translations and asserts `resolve_entry` returns a placement when `maximum_scene_xy_mismatch_m=None`. Add matcher tests that assert a terrain candidate yaw equals `shaped.heading_world_yaw - source_root_yaw`, and that a lateral velocity with forward facing tests compatibility against lateral travel rather than facing.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_contact_segments \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: yaw and lateral-compatibility assertions fail against registered-yaw/facing behavior.

- [ ] **Step 3: Implement local yaw and travel helpers**

Add `_command_travel_direction(shaped)` that normalizes the final trajectory displacement and falls back to normalized shaped velocity. In `_compose_candidate`, use `_wrapped_angle(shaped.heading_world_yaw - source_yaw)` for terrain and flat clips. Rotate source root velocity by that yaw and compare it with travel direction in `_segment_command_compatible`. Build terrain entry eligibility from source velocities rotated by per-row candidate yaw rather than static registered scene yaw. Construct the current runtime policy with `maximum_scene_xy_mismatch_m=None` so `resolve_entry` performs support-foot vertical retargeting without absolute XY rejection.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_contact_segments.py \
  sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_contact_segments.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "fix: retarget terrain segments in local command frame"
```

---

### Task 2: Bound failed rescue latency

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Consumes: `MatcherConfig.transition_window_candidate_count`.
- Produces: a hard composition bound for `_ranked_terrain_rescue` and terrain-entry override.

- [ ] **Step 1: Write a failing composition-count test**

Use a synthetic matcher with 100 ranked invalid terrain entries, set `transition_window_candidate_count=4`, count `_compose_candidate` calls, and assert no more than four ranked entries are composed in each rescue path.

- [ ] **Step 2: Run the single test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: composition count exceeds four.

- [ ] **Step 3: Apply the candidate bound**

Wrap both ranked terrain loops with `itertools.islice(ranked, self.config.transition_window_candidate_count)`. Preserve rank numbering, incumbent fallback, and diagnostics.

- [ ] **Step 4: Run matcher tests and verify GREEN**

Run the Step 2 command. Expected: all matcher tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "fix: bound terrain rescue candidate validation"
```

---

### Task 3: Contact-policy adversarial evaluator

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_omni_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_contact_segment_rollout.py`
- Modify: `resources/run_g1_torch_terrain_omni.py`
- Modify: `tests/python/test_sonic_torch_terrain_omni_rollout.py`

**Interfaces:**
- Consumes: `run_resolved_omni_matrix`, `TerrainContactSegmentPolicy`, contact experiment schema.
- Produces: optional contact policy injection and `maximum_step_time_ns` fail-fast evaluation.

- [ ] **Step 1: Write failing evaluator tests**

Add a fake matcher result with `step_time_ns=1_000_000_001`, call `run_omni_matrix(..., maximum_step_time_ns=1_000_000_000)`, and assert the route stops after one recorded frame with failure stage `latency`. Add a factory-injection test proving a supplied contact policy reaches `TorchMotionMatcher.from_folder`.

- [ ] **Step 2: Run evaluator tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_omni_rollout
```

Expected: unexpected keyword arguments or missing latency failure.

- [ ] **Step 3: Implement injection and timing gate**

Thread `contact_segment_policy=None` through `run_resolved_omni_matrix` into `TorchMotionMatcher.from_folder`. Thread `maximum_step_time_ns=None` through `run_omni_matrix` and `_run_route`; after recording a committed frame, raise `RuntimeError("matcher step exceeded ...")` with stage `latency` when the diagnostic exceeds the limit. Update the resource runner with `--contact-segments` to load/resolve the contact schema and build the exact MuJoCo-FK policy.

- [ ] **Step 4: Run evaluator tests and verify GREEN**

Run the Step 2 command. Expected: all evaluator tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_omni_rollout.py \
  sonic/python/mm_sonic/torch_contact_segment_rollout.py \
  resources/run_g1_torch_terrain_omni.py \
  tests/python/test_sonic_torch_terrain_omni_rollout.py
git commit -m "feat: evaluate contact terrain matcher adversarially"
```

---

### Task 4: Real-data qualification

**Files:**
- Create generated artifacts only under: `build/contact-omni-local-*`
- Modify if evidence warrants: the Task 1 or Task 2 production files and their tests.

**Interfaces:**
- Consumes: representative corpus, contact config, G1 MuJoCo XML, same-stair routes.
- Produces: deterministic route metrics and a go/no-go viewer decision.

- [ ] **Step 1: Run constant-forward and focused adversarial routes**

Run `riser-reversal`, both cross-tread routes, both diagonal-down routes, both upper side exits, and `mixed-adversarial` across available GPUs with the 1-second step gate.

- [ ] **Step 2: Inspect candidate reachability and failures**

Require nonzero safe reverse and lateral entry, no step above one second, and no backward progress during a constant-forward command. If a gate fails, add one minimal failing regression test before altering production behavior.

- [ ] **Step 3: Run the complete unit suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover -s tests/python -v
```

Expected: all non-opt-in tests pass; real-data opt-in tests remain explicitly skipped unless selected.

- [ ] **Step 4: Viewer decision**

Launch one fresh viewer only when focused real-data gates pass. Otherwise publish the measured failing routes and continue the test/fix loop without asking the user to diagnose visually.
