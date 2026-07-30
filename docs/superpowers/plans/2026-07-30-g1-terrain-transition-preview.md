# G1 Terrain Transition Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject a lower-cost terrain transition when its actual first 0.20 seconds of inertialized output would put an ankle more than 3 cm below the query surface, while preserving a safe incumbent transactionally.

**Architecture:** `TorchMotionMatcher` gains an optional generic emitted-window validator and one shared candidate-composition function. Search still produces its normal exact lowest-cost decision. The matcher composes and validates that decision; an unsafe transition falls back to the already valid incumbent, while an unsafe incumbent or missing incumbent fails closed. `TerrainFootClearanceValidator` implements the injected policy from the authenticated query grid, and only the dense stair condition enables it.

**Tech Stack:** Python 3.10, PyTorch, NumPy, unittest, native MuJoCo forward kinematics.

## Global Constraints

- Work only on `research/g1-torch-terrain-kinematics`.
- Keep output at exactly 50 Hz (`dt=0.02`) and preview exactly 10 emitted frames (`0.20 s`).
- Use the existing acceptance threshold `minimum_foot_clearance_m=-0.03`; do not create a second clearance threshold.
- Validate the inertialized emitted body window, not the raw database clip.
- Search cost, eligibility, transition penalty, tie order, and the 20-frame local exclusion remain unchanged.
- The validator may reject only a prospective transition. If the incumbent is unsafe or absent, fail explicitly.
- No validator means bitwise-identical flat matcher behavior.
- Do not add SONIC, physics, depth inference, global root height, or wall-clock acceptance.
- Follow RED-GREEN-REFACTOR and commit each independently testable task.

---

### Task 1: Add generic transactional emitted-window validation

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Produces: `EmittedWindowValidator`, a callable protocol taking a cloned `(46, 3, 3)` float32 body-position window and returning exact `bool`.
- Extends: `TorchMotionMatcher.from_folder(..., emitted_window_validator: EmittedWindowValidator | None = None)`.
- Extends: `MotionMatchDiagnostics.transition_rejected: bool`.

- [ ] **Step 1: Write failing transactional fallback tests**

Import `mock`, `MatcherConfig`, `SearchDecision`, and
`EmittedWindowValidator`. Add:

```python
class _ScriptedWindowValidator:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.windows = []

    def __call__(self, window):
        self.windows.append(window.clone())
        return self.decisions.pop(0)
```

Create a 120-frame synthetic matcher with
`MatcherConfig(search_interval_steps=1)`, reset it, identify the exact successor
row, and patch `select_exact_candidate` to return a different valid row with
`transitioned=True`. Supply decisions `[False, True]`. Assert after one step:

```python
self.assertEqual(result.diagnostics.selected_frame, reset_frame + 1)
self.assertFalse(result.diagnostics.transitioned)
self.assertTrue(result.diagnostics.transition_rejected)
self.assertTrue(result.diagnostics.searched)
self.assertEqual(len(validator.windows), 2)
self.assertEqual(validator.windows[0].shape, (46, 3, 3))
```

Mutate the first captured clone and assert the committed result does not change.

Add a second test whose selected decision is the incumbent and whose validator
returns `False`; assert `ContractError` containing `incumbent` and verify a
subsequent valid prepare still starts from the unchanged sequence. Add a third
test with a validator returning `torch.tensor(True)` and assert a contract error
because validator results must be exact Python `bool`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: import or constructor failure because the validator seam does not
exist.

- [ ] **Step 3: Factor one shared candidate-composition function**

Add a private frozen `_ComposedCandidate` containing:

```python
clip_index: int
frame_index: int
transitioned: bool
yaw_offset: torch.Tensor
translation_xy: torch.Tensor
offsets: _Offsets
dense_joint_position: torch.Tensor
dense_joint_velocity: torch.Tensor
dense_root_position: torch.Tensor
dense_root_quaternion: torch.Tensor
dense_root_linear_velocity: torch.Tensor
dense_root_angular_velocity: torch.Tensor
dense_body_position: torch.Tensor
dense_body_velocity: torch.Tensor
```

Move the existing selected-row placement, target alignment, offset creation,
spring decay, and dense-window composition from `prepare_step` into:

```python
def _compose_candidate(
    self,
    state: _MatcherState,
    shaped: ShapedCommand,
    selected_row: int,
    incumbent_row: int | None,
) -> _ComposedCandidate:
```

Define `transitioned` as
`incumbent_row is None or selected_row != incumbent_row`. Preserve every
existing equation and tensor operation exactly. The normal selected path must
consume this helper before validation, and the fallback path must call the same
helper for the incumbent. No candidate-composition formula may remain duplicated
inside `prepare_step`.

- [ ] **Step 4: Implement optional validation and exact fallback**

Define:

```python
class EmittedWindowValidator(Protocol):
    def __call__(
        self, feature_body_position_window: torch.Tensor
    ) -> bool:
        ...
```

Store the optional validator on the matcher. After composing the search
decision, call it with `dense_body_position.clone()`. Reject non-`bool` output.

If validation fails:

- when the candidate is the incumbent, raise
  `ContractError("emitted-window incumbent is unsafe")`;
- when no incumbent row exists, raise
  `ContractError("unsafe transition has no incumbent")`;
- otherwise compose and validate the incumbent;
- if that validation fails, raise the incumbent error; and
- if it passes, replace the selected decision with the incumbent while
  preserving `searched`, `search_time_ns`, and `incumbent_cost`, and set
  `transition_rejected=True`.

Compute final split motion/extension costs from the retained row after fallback.
Set `transition_rejected=False` on reset and ordinary accepted decisions.

- [ ] **Step 5: Run matcher and flat-equivalence GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_search
```

Expected: PASS, including the existing 100-command no-extension bitwise
equivalence test.

- [ ] **Step 6: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: validate emitted transition windows transactionally"
```

---

### Task 2: Implement authenticated terrain foot-clearance validation

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_features.py`
- Modify: `tests/python/test_sonic_torch_terrain_features.py`

**Interfaces:**
- Produces: `TerrainFootClearanceValidator(extension, preview_steps, minimum_clearance_m)`.
- Callable input: inertialized `(46, 3, 3)` root/left/right body positions in matcher world.
- Callable output: `True` only when both ankle origins stay at or above the threshold for all preview frames.

- [ ] **Step 1: Write failing geometry and contract tests**

Using the synthetic loaded terrain dataset from `TerrainFeatureTests`, construct
the dense query extension and validator with 10 steps and `-0.03 m`.
Build a `(46, 3, 3)` window whose XY points lie inside the query grid. Set both
foot Z values to sampled height plus `0.035 m`; assert accepted. Lower one foot
at preview frame 9 to height minus `0.031 m`; assert rejected. Lower only frame
10; assert accepted, proving the window is exactly frames 0 through 9.

Assert rejection of:

- preview steps 0 or 47;
- non-finite or non-numeric clearance threshold;
- wrong body-window shape;
- float64, CPU/device mismatch, or non-finite body values; and
- an out-of-domain foot query.

- [ ] **Step 2: Run focused test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_features
```

Expected: import failure for `TerrainFootClearanceValidator`.

- [ ] **Step 3: Implement the vectorized validator**

Add the frozen dataclass:

```python
@dataclass(frozen=True)
class TerrainFootClearanceValidator:
    extension: TerrainFeatureExtension
    preview_steps: int
    minimum_clearance_m: float

    def __call__(self, body_window: torch.Tensor) -> bool:
```

Validate constructor fields in `__post_init__`. In `__call__`, require exact
shape `(46, 3, 3)`, float32, finite values, and the extension dataset device.
For `body_window[:preview_steps]`, map all three bodies' XY coordinates through
`extension.alignment.matcher_to_scene_xy`, sample the authoritative query grid,
and compute:

```python
clearance = body_window[:preview_steps, 1:, 2] - surface[:, 1:]
return bool((clearance.min() >= self.minimum_clearance_m).item())
```

Do not inspect root Z and do not clamp out-of-domain samples.

- [ ] **Step 4: Run terrain-feature GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_features
```

Expected: PASS.

Commit:

```bash
git add sonic/python/mm_sonic/torch_terrain_features.py \
  tests/python/test_sonic_torch_terrain_features.py
git commit -m "feat: validate terrain clearance over emitted windows"
```

---

### Task 3: Wire dense rollouts and the live viewer to the same preview

**Files:**
- Modify: `sonic/configs/experiments/torch_stair_small.json`
- Modify: `sonic/python/mm_sonic/torch_terrain_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_rollout.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Consumes: top-level `terrain_transition_preview_steps: 10`.
- Produces: `terrain_transition_validator_from_resolved(resolved)`.
- Adds rollout array/event field `transition_rejected` and metric `rejected_transition_count`.

- [ ] **Step 1: Write failing configuration, propagation, and evidence tests**

Extend the config test to require
`terrain_transition_preview_steps` as an exact positive integer in `[1, 46]`
and reject booleans, zero, 47, and extra nested alternatives.

Assert:

```python
validator = terrain_transition_validator_from_resolved(self.resolved)
self.assertEqual(validator.preview_steps, 10)
self.assertEqual(
    validator.minimum_clearance_m,
    self.resolved.resolved_config["acceptance"]["minimum_foot_clearance_m"],
)
```

Patch `TorchMotionMatcher.from_folder` during short synthetic flat and dense
rollouts. Assert flat receives `emitted_window_validator=None`, dense receives a
`TerrainFootClearanceValidator`, and both still receive the same pinned
`MatcherConfig`.

Assert every rollout contains Boolean `transition_rejected` with shape
`(step_count,)`, and:

```python
metrics["rejected_transition_count"] == int(
    arrays["transition_rejected"].sum()
)
```

Extend the viewer source test to prove it constructs the same helper and passes
`emitted_window_validator`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer
```

Expected: failures for missing configuration, helper, and evidence fields.

- [ ] **Step 3: Validate config and build the shared validator**

Add `"terrain_transition_preview_steps": 10` to the JSON. Validate exact integer
type and range 1 through 46 in `_validate_base_config`.

Implement:

```python
def terrain_transition_validator_from_resolved(
    resolved: ResolvedStairConfig,
) -> TerrainFootClearanceValidator:
    return TerrainFootClearanceValidator(
        extension=resolved.measurement_extension,
        preview_steps=int(
            resolved.resolved_config["terrain_transition_preview_steps"]
        ),
        minimum_clearance_m=float(
            resolved.resolved_config["acceptance"][
                "minimum_foot_clearance_m"
            ]
        ),
    )
```

- [ ] **Step 4: Enable validation only for dense matching**

In `run_stair_rollout`, pass the helper as `emitted_window_validator` only when
`condition == "dense"`; pass `None` for flat and legacy controls. In
`run_live_viewer`, always pass the helper because that viewer requires the dense
condition.

Record `diagnostics.transition_rejected` in each rollout row and event. Add
`rejected_transition_count` to metrics. Include rejected transitions in the
event stream even though the retained result itself did not transition.

- [ ] **Step 5: Run focused GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_search
```

Expected: PASS.

Commit:

```bash
git add sonic/configs/experiments/torch_stair_small.json \
  sonic/python/mm_sonic/torch_terrain_rollout.py \
  sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_terrain_rollout.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py
git commit -m "feat: reject unsafe dense stair transitions"
```

---

### Task 4: Qualify the retained preview and update evidence

**Files:**
- Modify: `docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md`
- Generated only: `build/torch-stair-small-results/quality-loop-02/`

**Interfaces:**
- Consumes: committed Tasks 1–3.
- Produces: full frozen acceptance, MuJoCo-FK cross-check, and a live viewer candidate.

- [ ] **Step 1: Run all frozen conditions on a free GPU**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_rollout \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --conditions flat legacy dense \
  --device cuda:1 \
  --output build/torch-stair-small-results/quality-loop-02
```

Expected dense result from the diagnostic prototype: minimum clearance at least
`-0.03 m`, zero or near-zero penetration, at least 90% progress, at least 80%
height gain, upper landing reached, approximately 36 transitions and 15 rejected
transitions. Exact deterministic values come from the retained implementation.

- [ ] **Step 2: Verify acceptance and MuJoCo forward kinematics**

Call `evaluate_dense_acceptance` on saved metrics and require `passed=True`.
At the saved minimum-clearance frame, apply root and joints to the pinned G1
MuJoCo model with `mj_forward`; require both ankle clearances to agree with the
rollout within `0.005 m` and to remain above `-0.03 m`.

- [ ] **Step 3: Run complete Torch regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch_*.py' -v
git diff --check
git status --short
```

Expected: PASS, with only the results document modified.

- [ ] **Step 4: Record qualification and commit**

Append implementation commits, deterministic hashes, full gate table,
rejection count, worst-frame cross-check, and latency diagnostics to the
results document.

Commit:

```bash
git add docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md
git commit -m "docs: qualify terrain transition preview"
```

- [ ] **Step 5: Launch the retained live viewer**

On the same free GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:1
```

Verify the process remains alive and the MuJoCo window appears before handing
control to the user.
