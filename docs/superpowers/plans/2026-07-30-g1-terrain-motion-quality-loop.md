# G1 Terrain Motion-Quality Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain the first two measured stair-quality improvements by making terrain-extension weights dimension invariant and pinning the stair experiment to exact 50 Hz search.

**Architecture:** The established seven motion groups keep their existing normalization exactly. Only an optional feature extension receives group-total weighting by multiplying its shared scale by the square root of its dimension. The experiment config explicitly records every `MatcherConfig` field, and both the deterministic rollout and live viewer construct the matcher from that same record.

**Tech Stack:** Python 3.10, PyTorch, NumPy, unittest, native MuJoCo kinematic viewer.

## Global Constraints

- Work only on `research/g1-torch-terrain-kinematics`.
- Keep matcher output at exactly 50 Hz with `dt=0.02`.
- Do not add SONIC, physics integration, depth inference, or global-root-height features.
- Do not change normalization or search behavior when no extension is present.
- Treat wall-clock search latency as diagnostic only.
- Keep the frozen dataset, query scene, command stream, and quality thresholds unchanged.
- Follow RED-GREEN-REFACTOR and commit each independently testable task.

---

### Task 1: Make optional extension weight group-total and dimension invariant

**Files:**
- Modify: `tests/python/test_sonic_torch_motion_features.py`
- Modify: `sonic/python/mm_sonic/torch_motion_features.py`

**Interfaces:**
- Consumes: `SearchFeatureExtension.dimension`, `.weight`, and database rows.
- Produces: unchanged `TorchMotionDatabase.from_folder(...)`; its extension scale is `mean_component_std * sqrt(extension.dimension) / extension.weight`.

- [ ] **Step 1: Write the failing dimension-invariance test**

Add a synthetic repeated-component extension:

```python
class _RepeatedValueExtension:
    name = "repeated_extension"
    weight = 2.0

    def __init__(self, dimension):
        self.dimension = dimension

    def database_rows(self, folder, device):
        rows = []
        for clip in folder.clips:
            value = torch.arange(
                clip.valid_frame_stop, dtype=torch.float32, device=device
            )
            rows.append(value[:, None].repeat(1, self.dimension))
        return tuple(rows)

    def query_row(self, state, trajectory):
        return state.root_position_world[0].repeat(self.dimension)
```

Build databases with dimensions 2 and 8 from the same motion folder. Assert
that every row's summed squared normalized extension cost is equal between the
two databases, and assert:

```python
raw = _RepeatedValueExtension(8).database_rows(folder, _CPU)[0]
expected = raw.std(dim=0, unbiased=False).mean() * math.sqrt(8) / 2.0
_, scale = eight.normalization.parameters_copy()
torch.testing.assert_close(
    scale[27:], expected.expand(8), rtol=1e-6, atol=1e-6
)
```

Also build a no-extension database and compare its mean, scale, and normalized
features bitwise with a separately constructed no-extension database so the
flat contract remains explicit.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_features.NormalizationAndDatabaseTests.test_extension_weight_is_dimension_invariant
```

Expected: FAIL because the eight-dimensional extension cost is four times the
two-dimensional cost and the scale omits `sqrt(dimension)`.

- [ ] **Step 3: Implement the minimum extension-only scale correction**

In `_build_database`, retain `FEATURE_GROUPS` unchanged and mark only the
appended extension as dimension invariant:

```python
groups = [
    (name, group_slice, weight, False)
    for name, group_slice, weight in FEATURE_GROUPS
]
if extension_contract is not None:
    name, dimension, weight = extension_contract
    groups.append(
        (
            name,
            slice(FEATURE_DIM, FEATURE_DIM + dimension),
            weight,
            True,
        )
    )
for name, group_slice, weight, group_total_weight in groups:
    group_std = component_std[group_slice].mean()
    group_scale = group_std / weight
    if group_total_weight:
        group_scale = group_scale * math.sqrt(
            group_slice.stop - group_slice.start
        )
```

Preserve the existing finite/positive scale check and assignment.

- [ ] **Step 4: Run focused and flat-equivalence tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_matcher.TorchMotionMatcherTests.test_explicit_none_preserves_flat_matcher_for_100_commands
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_motion_features.py \
  tests/python/test_sonic_torch_motion_features.py
git commit -m "fix: make terrain extension weight dimension invariant"
```

---

### Task 2: Pin the stair experiment's matcher configuration

**Files:**
- Modify: `sonic/configs/experiments/torch_stair_small.json`
- Modify: `sonic/python/mm_sonic/torch_terrain_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_rollout.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Consumes: top-level experiment JSON field `matcher`.
- Produces: `matcher_config_from_resolved(config: Mapping) -> MatcherConfig`, shared by rollout and viewer.

- [ ] **Step 1: Write failing config and propagation tests**

Require this exact matcher record:

```json
{
  "search_interval_steps": 1,
  "acceleration_mps2": 1.5,
  "deceleration_mps2": 2.0,
  "yaw_rate_rad_s": 2.0943951023931953,
  "stop_speed_mps": 0.05,
  "reversal_speed_mps": 0.15,
  "exclusion_frames": 20,
  "transition_penalty": 0.1,
  "inertialization_halflife_s": 0.1
}
```

In the rollout tests, assert that `load_experiment_config` rejects a missing,
extra, non-finite, non-positive interval, negative exclusion, or non-positive
halflife field. Assert:

```python
matcher = matcher_config_from_resolved(self.resolved.resolved_config)
self.assertEqual(matcher.search_interval_steps, 1)
self.assertEqual(matcher.dt, 0.02)
self.assertAlmostEqual(matcher.inertialization_halflife_s, 0.1)
```

Patch `TorchMotionMatcher.from_folder` with `wraps` during a short synthetic
rollout and assert its `config` keyword equals the helper result. In the viewer
test, inspect the module source and assert `matcher_config_from_resolved`
appears in `run_live_viewer`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer
```

Expected: FAIL because the matcher record and helper do not yet exist.

- [ ] **Step 3: Validate and construct the pinned matcher**

Import `MatcherConfig` into `torch_terrain_rollout.py`. Define an exact tuple of
matcher keys and extend `_validate_base_config` to require the exact record.
Integer fields reject booleans and require:

```python
search_interval_steps >= 1
exclusion_frames >= 0
```

Floating fields must be finite; acceleration, deceleration, yaw rate, stop
speed, reversal speed, transition penalty, and halflife must be positive.
Implement:

```python
def matcher_config_from_resolved(config: Mapping) -> MatcherConfig:
    values = config["matcher"]
    return MatcherConfig(
        dt=float(config["dt"]),
        search_interval_steps=int(values["search_interval_steps"]),
        acceleration_mps2=float(values["acceleration_mps2"]),
        deceleration_mps2=float(values["deceleration_mps2"]),
        yaw_rate_rad_s=float(values["yaw_rate_rad_s"]),
        stop_speed_mps=float(values["stop_speed_mps"]),
        reversal_speed_mps=float(values["reversal_speed_mps"]),
        exclusion_frames=int(values["exclusion_frames"]),
        transition_penalty=float(values["transition_penalty"]),
        inertialization_halflife_s=float(
            values["inertialization_halflife_s"]
        ),
    )
```

Add the exact matcher record to `torch_stair_small.json`.

- [ ] **Step 4: Route rollout and viewer through the same config**

In `run_stair_rollout`, pass:

```python
config=matcher_config_from_resolved(config.resolved_config)
```

to `TorchMotionMatcher.from_folder`. Import the helper in
`torch_terrain_live_viewer.py` and pass the identical value when constructing
its matcher. Do not change the generic `MatcherConfig` defaults.

- [ ] **Step 5: Run focused tests and commit Task 2**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: PASS.

Commit:

```bash
git add sonic/configs/experiments/torch_stair_small.json \
  sonic/python/mm_sonic/torch_terrain_rollout.py \
  sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_terrain_rollout.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py
git commit -m "feat: pin 50 Hz stair search configuration"
```

---

### Task 3: Publish deterministic evidence for the retained candidate

**Files:**
- Modify: `docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md`
- Generated only: `build/torch-stair-small-results/quality-loop-01/`

**Interfaces:**
- Consumes: committed Tasks 1 and 2.
- Produces: reproducible flat/legacy/dense rollouts, gate evaluation, and a recorded rejected-ablation table.

- [ ] **Step 1: Run all three frozen conditions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_rollout \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --conditions flat legacy dense \
  --device cuda \
  --output build/torch-stair-small-results/quality-loop-01
```

If unrelated GPU jobs prevent allocation, use `--device cpu` and record the
device; deterministic kinematics, rather than latency, are the acceptance
subject.

- [ ] **Step 2: Run the acceptance evaluator and inspect the worst frame**

Load each `metrics.json`, call `evaluate_dense_acceptance`, and report:

```text
first stair selection, maximum progress, maximum root-height gain,
minimum foot clearance, integrated penetration, upper-landing result,
transition count, worst selected clip/frame, p50/p95 search time
```

Cross-check the worst frame by applying its saved root and joint state to the
native G1 MuJoCo model with `mj_forward`; query the authenticated stair grid at
the left/right ankle-roll body XY positions. The matcher and MuJoCo values must
agree within `0.005 m`.

- [ ] **Step 3: Record retained and rejected ablations**

Append a dated quality-loop section to the results document containing:

- original dense baseline;
- extension dimension correction with five-frame search;
- transition penalty `10.0` rejection;
- search intervals two and one;
- halflives `0.05` and `0.02` rejection; and
- the committed candidate's acceptance criteria.

State explicitly that the next hypothesis is terrain-aware emitted-window
candidate preview if the `-0.03 m` clearance gate still fails.

- [ ] **Step 4: Run regression verification and commit evidence**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch_*.py' -v
git diff --check
```

Expected: PASS and no whitespace errors.

Commit:

```bash
git add docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md
git commit -m "docs: record first terrain motion quality iteration"
```
