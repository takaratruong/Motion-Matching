# G1 Blend-Aware Transition Hysteresis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce visible 50 Hz terrain-motion stutter by discouraging repeated transitions while the current inertial blend is still settling.

**Architecture:** Add a pure linear-decay helper and pass its result into exact candidate selection as an extra non-incumbent cost. Reconstruct the two new parameters from the frozen experiment configuration, then sweep only the penalty magnitude against deterministic stutter and terrain-safety evidence.

**Tech Stack:** Python 3.10, PyTorch, NumPy, `unittest`, MuJoCo, JSON experiment configuration.

## Global Constraints

- Continue exact search every `0.02 s`; do not add a hard transition lockout.
- Preserve candidate eligibility, feature normalization, terrain preview, source exclusion, and emitted output.
- The extra cost applies only to non-incumbent candidates and decays linearly to zero.
- Zero magnitude or zero duration disables hysteresis.
- A retained candidate must preserve at least 90% reference progress, at least 80% reference root-height gain, the upper landing, and at least `-0.03 m` MuJoCo-FK ankle-origin clearance.
- Do not modify SONIC tracking, physics integration, depth inference, or learned control.

---

### Task 1: Pure Blend-Aware Candidate Cost

**Files:**
- Modify: `tests/python/test_sonic_torch_motion_search.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`

**Interfaces:**
- Consumes: `MatcherConfig`, `select_exact_candidate(...)`.
- Produces: `active_transition_penalty(blend_age_s: float, config: MatcherConfig) -> float` and `select_exact_candidate(..., additional_transition_penalty: float = 0.0)`.

- [x] **Step 1: Write failing decay and selection tests**

Add imports for `ContractError` and `active_transition_penalty`, then add tests that assert:

```python
def test_active_transition_penalty_decays_linearly_and_can_be_disabled(self):
    cfg = MatcherConfig(
        transition_settle_duration_s=0.20,
        transition_settle_penalty=10.0,
    )
    self.assertAlmostEqual(active_transition_penalty(0.0, cfg), 10.0)
    self.assertAlmostEqual(active_transition_penalty(0.05, cfg), 7.5)
    self.assertAlmostEqual(active_transition_penalty(0.20, cfg), 0.0)
    self.assertAlmostEqual(active_transition_penalty(1.0, cfg), 0.0)
    self.assertEqual(
        active_transition_penalty(
            0.0, MatcherConfig(transition_settle_penalty=0.0)
        ),
        0.0,
    )

def test_additional_penalty_never_applies_to_incumbent(self):
    db = _database(
        [[math.sqrt(1.0)], [math.sqrt(0.5)]],
        [0, 1],
        [1, 1],
        "cpu",
    )
    held = select_exact_candidate(
        db,
        torch.zeros(1),
        current_clip_index=0,
        current_frame_index=0,
        incumbent_row=0,
        search=True,
        config=MatcherConfig(),
        additional_transition_penalty=0.5,
    )
    self.assertEqual(held.selected_row, 0)
    switched = select_exact_candidate(
        db,
        torch.zeros(1),
        current_clip_index=0,
        current_frame_index=0,
        incumbent_row=0,
        search=True,
        config=MatcherConfig(),
        additional_transition_penalty=0.3,
    )
    self.assertEqual(switched.selected_row, 1)
```

Also add subtests proving negative/non-finite ages, magnitudes, durations, and explicit additional penalties raise `ContractError`.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_search -v
```

Expected: import or constructor failures because the helper and configuration fields do not exist.

- [x] **Step 3: Add the minimal helper and selection input**

Extend `MatcherConfig`:

```python
transition_settle_duration_s: float = 0.0
transition_settle_penalty: float = 0.0
```

Add:

```python
def active_transition_penalty(
    blend_age_s: float,
    config: MatcherConfig = MatcherConfig(),
) -> float:
    age = float(blend_age_s)
    duration = float(config.transition_settle_duration_s)
    magnitude = float(config.transition_settle_penalty)
    if not math.isfinite(age) or age < 0.0:
        raise ContractError("blend_age_s must be finite and non-negative")
    if not math.isfinite(duration) or duration < 0.0:
        raise ContractError(
            "transition_settle_duration_s must be finite and non-negative"
        )
    if not math.isfinite(magnitude) or magnitude < 0.0:
        raise ContractError(
            "transition_settle_penalty must be finite and non-negative"
        )
    if duration == 0.0 or magnitude == 0.0 or age >= duration:
        return 0.0
    return magnitude * (1.0 - age / duration)
```

Add `additional_transition_penalty: float = 0.0` as a keyword-only argument to
`select_exact_candidate`. Validate it as finite and non-negative, then replace:

```python
total_costs = feature_costs + config.transition_penalty
```

with:

```python
total_costs = (
    feature_costs
    + config.transition_penalty
    + additional_transition_penalty
)
```

Continue overwriting the incumbent row with its raw feature cost.

- [x] **Step 4: Run focused and existing flat search tests**

Run the command from Step 2.

Expected: all tests pass on CPU and CUDA when CUDA is available.

- [x] **Step 5: Commit the pure algorithm**

```bash
git add tests/python/test_sonic_torch_motion_search.py \
  sonic/python/mm_sonic/torch_motion_matcher.py
git commit -m "feat: add blend-aware transition cost"
```

### Task 2: Matcher-State and Experiment Wiring

**Files:**
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_terrain_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_rollout.py`
- Modify: `sonic/configs/experiments/torch_stair_small.json`

**Interfaces:**
- Consumes: `active_transition_penalty(...)` and `select_exact_candidate(..., additional_transition_penalty=...)`.
- Produces: exact matcher configuration reconstruction and runtime use of `_Offsets.elapsed_s`.

- [x] **Step 1: Write failing wiring tests**

In the rollout configuration test, assert:

```python
self.assertAlmostEqual(matcher.transition_settle_duration_s, 0.20)
self.assertGreater(matcher.transition_settle_penalty, 0.0)
```

Add invalid JSON cases for negative and non-finite settle duration/penalty, while
explicit zero values remain valid.

In the matcher test, patch `select_exact_candidate`, execute a continuation
after reset and an accepted transition, and assert the
`additional_transition_penalty` passed after reset is zero while the value
passed immediately after the accepted transition is positive.

- [x] **Step 2: Run wiring tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_terrain_rollout -v
```

Expected: failures because the new JSON keys and matcher runtime input are not wired.

- [x] **Step 3: Wire validated configuration and runtime state**

Add `transition_settle_duration_s` and `transition_settle_penalty` to the
rollout's exact matcher float-field set. Validate these two as finite and
non-negative; retain positive validation for the existing fields. Reconstruct
both fields in `matcher_config_from_resolved`.

Set the initial `_Offsets.elapsed_s` in `reset()` to
`config.transition_settle_duration_s`, because reset is not an accepted
transition and zero inertial offsets must not trigger hysteresis.

Before `select_exact_candidate` in `prepare_step`, calculate:

```python
settle_penalty = active_transition_penalty(
    state.offsets.elapsed_s, self.config
)
```

and pass it as `additional_transition_penalty=settle_penalty`.

Add the initial experiment values:

```json
"transition_settle_duration_s": 0.2,
"transition_settle_penalty": 10.0
```

- [x] **Step 4: Run focused wiring tests**

Run the command from Step 2.

Expected: all focused matcher and rollout tests pass.

- [x] **Step 5: Run the full Torch matcher suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch_*.py' -v
```

Expected: zero failures; the existing optional MuJoCo-dependent skip is allowed.

- [x] **Step 6: Commit runtime wiring**

```bash
git add tests/python/test_sonic_torch_motion_matcher.py \
  tests/python/test_sonic_torch_terrain_rollout.py \
  sonic/python/mm_sonic/torch_motion_matcher.py \
  sonic/python/mm_sonic/torch_terrain_rollout.py \
  sonic/configs/experiments/torch_stair_small.json
git commit -m "feat: settle terrain matcher transitions"
```

### Task 3: Terrain-Safety Override

**Files:**
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_terrain_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_rollout.py`

**Interfaces:**
- Consumes: `active_transition_penalty(...)`, transactional
  `prepare_step(...)`, and `EmittedWindowValidator`.
- Produces: one bounded zero-hysteresis retry and
  `MotionMatchDiagnostics.hysteresis_overridden: bool`.

- [x] **Step 1: Write failing transactional safety-override tests**

Build a matcher with active hysteresis and a scripted validator. Patch
`select_exact_candidate` so its first decision retains the incumbent and its
second decision transitions. Assert:

```python
result = matcher.step((0.5, 0.0), 0.0)
self.assertTrue(result.diagnostics.transitioned)
self.assertTrue(result.diagnostics.hysteresis_overridden)
self.assertEqual(selector.call_count, 2)
self.assertGreater(
    selector.call_args_list[0].kwargs["additional_transition_penalty"],
    0.0,
)
self.assertEqual(
    selector.call_args_list[1].kwargs["additional_transition_penalty"],
    0.0,
)
```

Add separate tests proving an unsafe retry, a retry that retains the incumbent,
and an unsafe incumbent with zero active penalty raise `ContractError` without
advancing the committed sequence.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_matcher -v
```

Expected: the safe-override test raises `emitted-window incumbent is unsafe` and
the diagnostic field does not exist.

- [x] **Step 3: Implement one bounded retry**

Add `hysteresis_overridden: bool` to `MotionMatchDiagnostics` and thread it
through `_make_result`, reset, normal steps, and rollout serialization.

When the first composed candidate is an unsafe incumbent and
`settle_penalty > 0.0`, call `select_exact_candidate` a second time with the
same arguments except `additional_transition_penalty=0.0`. Require the retry
decision to transition, compose it from the unchanged state, and validate its
emitted window. If it is safe, replace the original candidate and decision and
set `hysteresis_overridden=True`. Otherwise raise `ContractError`. Do not loop
or mutate `_state` before the existing commit boundary.

- [x] **Step 4: Save and validate the override diagnostic**

Add a boolean `hysteresis_overridden` row to deterministic rollout artifacts,
event JSON, saved-array validation, and tests. Timing remains excluded from the
deterministic digest; the new boolean is included.

- [x] **Step 5: Run focused and full suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_terrain_rollout -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch_*.py' -v
```

Expected: zero failures; the existing optional skip is allowed.

- [x] **Step 6: Commit the safety override**

```bash
git add tests/python/test_sonic_torch_motion_matcher.py \
  tests/python/test_sonic_torch_terrain_rollout.py \
  sonic/python/mm_sonic/torch_motion_matcher.py \
  sonic/python/mm_sonic/torch_terrain_rollout.py
git commit -m "fix: let terrain safety override transition settling"
```

### Task 4: Deterministic Magnitude Sweep and Viewer Qualification

**Files:**
- Modify: `sonic/configs/experiments/torch_stair_small.json`
- Modify: `docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md`

**Interfaces:**
- Consumes: frozen dataset `build/torch-stair-small` and the deterministic stair-rollout CLI.
- Produces: one retained magnitude and a reproducible evidence directory.

- [x] **Step 1: Evaluate the initial magnitude**

Run the dense condition on `cuda:1` into a new iteration directory using the
existing `mm_sonic.torch_terrain_rollout` CLI and the frozen config. Save the
rollout, events, resolved config, and metrics.

Expected: all prior stair gates pass; otherwise the candidate is rejected.

- [x] **Step 2: Calculate the stutter report**

From `rollout.npz`, report transition count, transition-interval quantiles,
intervals at most `0.20 s`, source jumps, transition-conditioned joint/root
acceleration and jerk, contact-foot planar velocity, progress, height, and
minimum clearance.

Expected: no accepted transitions on consecutive `0.02 s` frames and lower
transition-conditioned acceleration or jerk than quality-loop-02.

- [x] **Step 3: Sweep only hysteresis magnitude**

If the initial value fails either terrain gates or stutter diagnostics, evaluate
the bounded set `5.0`, `20.0`, and `40.0`, holding duration at `0.20 s` and every
other parameter fixed. Reject candidates that fail terrain gates. Among passing
candidates, retain the smallest magnitude that removes consecutive-frame
transition bursts and minimizes transition-conditioned jerk.

- [x] **Step 4: Cross-check retained clearance with MuJoCo FK**

Replay every saved `joint_position`, `root_position_world`, and
`root_orientation_world_wxyz` through the pinned G1 XML with `mj_forward`.

Expected: ankle-origin clearance is at least `-0.03 m`.

- [x] **Step 5: Update results and run final verification**

Record baseline and sweep values, the retained configuration, deterministic
hash, exact commands, and any rejected candidates in the results document.
Then rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch_*.py' -v
```

Expected: zero failures; the existing optional skip is allowed.

- [x] **Step 6: Commit evidence**

```bash
git add sonic/configs/experiments/torch_stair_small.json \
  docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md
git commit -m "docs: qualify blend-aware terrain transitions"
```

- [x] **Step 7: Launch the retained interactive viewer**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:1
```

Expected: the user can control the kinematic robot over the visible stairs
without the prior rapid transition stutter.

### Task 5: General Unsafe-Incumbent Terrain Rescue

**Files:**
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_terrain_rollout.py`
- Modify: `tests/python/test_sonic_torch_terrain_viewer.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_viewer.py`

**Interfaces:**
- Consumes: exact candidate selection, current successor row, emitted-window
  validator, and transactional prepare/commit.
- Produces: a single forced non-incumbent rescue search and
  `MotionMatchDiagnostics.terrain_safety_override: bool`.

- [ ] **Step 1: Reproduce the settled-blend live crash**

Write a test in which ordinary selection retains an unsafe incumbent with
`additional_transition_penalty == 0.0`. Assert a second selector call uses:

```python
self.assertIsNone(rescue["incumbent_row"])
self.assertTrue(rescue["search"])
self.assertEqual(rescue["additional_transition_penalty"], 0.0)
```

Return a safe non-local transition and assert it commits with
`terrain_safety_override=True`.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_matcher -v
```

Expected: `emitted-window incumbent is unsafe`.

- [ ] **Step 3: Implement the bounded rescue**

When any selected incumbent fails its emitted-window validator, call
`select_exact_candidate` once with `incumbent_row=None`, `search=True`, and
`additional_transition_penalty=0.0`. Compose the returned row as a transition
against the real successor, validate it, and commit only if safe. Preserve the
original incumbent cost in diagnostics and include rescue search time.

Rename `hysteresis_overridden` to `terrain_safety_override` throughout runtime,
rollout arrays, events, metrics, tests, and documentation.

- [ ] **Step 4: Preserve historical rollout loading**

When `terrain_safety_override` is absent from a saved rollout, synthesize an
all-false boolean array after archive authentication. Add a loader regression
test using an archive without the new field.

- [ ] **Step 5: Verify scripted and live behavior**

Run the full Torch suite, flat/legacy/dense CUDA qualification, authoritative
MuJoCo-FK sweep, and the same user-controlled command sequence that produced the
settled-blend crash.

- [ ] **Step 6: Commit the retained rescue and evidence**

Commit production code/tests separately from the updated result evidence. Do
not call the viewer stable until it survives user steering and the process
remains alive.
