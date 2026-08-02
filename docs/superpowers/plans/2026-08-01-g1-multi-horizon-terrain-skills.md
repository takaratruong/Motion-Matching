# G1 Multi-Horizon Terrain Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match GRAIL terrain motion by 0.5/1.0/2.0-second outcomes and replay only to a stable selected endpoint, earning at least two full passes on the frozen six-route slice.

**Architecture:** A new immutable horizon inventory flattens every valid `(entry row, target horizon)` pair into GPU tensors. A layered search combines the unchanged normalized 27-value entry cost with local displacement/yaw/height/duration/stall outcome costs, then applies the existing full trace terrain validator. The existing sequential composer receives an explicit stable endpoint, while an isolated horizon rollout adapter and CLI preserve the current whole-skill baseline.

**Tech Stack:** Python 3.10, PyTorch, NumPy, MuJoCo FK, `unittest`, authenticated GRAIL 50 Hz motion and height-grid artifacts.

## Global Constraints

- Work only on `research/g1-torch-terrain-kinematics` in the existing linked worktree.
- Do not stage or modify the dirty landing-bridge, FK, or contact-segment files already present.
- Keep the standard motion-matching feature exactly 27 values.
- Target horizons are exactly 25, 50, and 100 frames; an endpoint may be at most 25 frames late.
- Never end a chunk in flight or single support.
- Do not use the rejected ten-frame stall hard gate; stall is a ranked cost.
- No Sonic, tracking, physics, depth, learned model, viewer, or full 21-route matrix in this plan.
- Continue only if all six routes execute, at least two pass behavior, support-height p95 is at most 0.03 m, and the 90-degree pivot retains progress and heading passes.

---

### Task 1: Immutable multi-horizon descriptor inventory

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skill_horizons.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_horizons.py`

**Interfaces:**
- Consumes: `TerrainDataset`, `TorchMotionDatabase`, `TerrainSkillInventory`.
- Produces: `TerrainSkillHorizonInventory`, `stable_horizon_endpoints`, `remaining_stall_profile`, `build_horizon_inventory`.

- [ ] **Step 1: Write failing endpoint and descriptor tests**

```python
def test_endpoint_is_first_double_support_within_lateness_cap(self):
    support = torch.zeros((140, 2), dtype=torch.bool)
    support[29] = True; support[52] = True; support[90] = True
    self.assertEqual(
        stable_horizon_endpoints(support, entry_frame=2, playback_stop=120),
        ((25, 30), (50, 53)),
    )

def test_remaining_stall_profile_is_time_local(self):
    root_xy = torch.tensor(
        [[0., 0.], [0., 0.], [0., 0.], [.1, 0.], [.2, 0.]]
    )
    self.assertEqual(remaining_stall_profile(root_xy).tolist(), [2, 1, 0, 0, 1])

def test_descriptor_uses_entry_heading_frame_and_unwrapped_yaw(self):
    record = describe_horizon(synthetic_clip, entry_frame=2, endpoint_frame=53)
    self.assertTrue(torch.allclose(record.root_displacement_local_xy, torch.tensor([1., .5])))
    self.assertAlmostEqual(record.yaw_delta_rad, 1.75)
```

- [ ] **Step 2: Verify RED**

Run:
`PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_terrain_skill_horizons`

Expected: missing `mm_sonic.torch_terrain_skill_horizons` import.

- [ ] **Step 3: Implement pure endpoint and descriptor primitives**

```python
HORIZON_TARGET_FRAMES = (25, 50, 100)
MAXIMUM_ENDPOINT_LATENESS_FRAMES = 25

@dataclass(frozen=True)
class TerrainSkillHorizonInventory:
    entry_row: torch.Tensor
    skill_index: torch.Tensor
    clip_index: torch.Tensor
    entry_frame: torch.Tensor
    target_frames: torch.Tensor
    endpoint_frame_exclusive: torch.Tensor
    root_displacement_local_xy: torch.Tensor
    yaw_delta_rad: torch.Tensor
    root_height_delta_m: torch.Tensor
    surface_height_delta_m: torch.Tensor
    maximum_stall_frames: torch.Tensor
    duration_frames: torch.Tensor
```

`stable_horizon_endpoints` searches `[entry + target, entry + target + 25]`,
requires exact double support, stores the stable endpoint as exclusive
`frame + 1`, and omits unavailable targets. `remaining_stall_profile` uses
forward root speed at 50 Hz with the frozen 0.03 m/s threshold. Accumulated yaw
is the sum of wrapped per-frame yaw differences, never endpoint subtraction.

- [ ] **Step 4: Build the flattened inventory and assert exact ownership**

`build_horizon_inventory` iterates the existing owned entry rows, maps source
frames through `database._search_frame_index`, builds all valid records, stacks
once per field, validates device/dtype/shape/finite invariants, and returns
owned tensors. It rejects duplicate `(entry row, target frames)` pairs and
records counts for `no_endpoint`, `too_short`, and `invalid_source`.

- [ ] **Step 5: Verify GREEN and corpus construction**

Run the new tests, current terrain-skill tests, then build the authenticated
inventory on `cuda:0`. Expected: tests pass, every endpoint is double support,
and the inventory contains at least 10,000 records without per-frame warnings.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_horizons.py \
  tests/python/test_sonic_torch_terrain_skill_horizons.py
git commit -m "feat: index multi-horizon terrain outcomes"
```

### Task 2: Layered GPU horizon search

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_horizon_search.py`

**Interfaces:**
- Consumes: horizon inventory, normalized 27-value query, current root/command state, terrain validator callback.
- Produces: `HorizonSearchConfig`, `HorizonCost`, `TerrainSkillHorizonResult`, `predict_horizon_targets`, `rank_horizon_candidates`, `select_horizon_candidate`.

- [ ] **Step 1: Write failing target and cost tests**

```python
def test_targets_sample_bounded_command_at_all_three_horizons(self):
    target = predict_horizon_targets(current_state, requested_command, matcher_config)
    self.assertEqual(target.frames.tolist(), [25, 50, 100])

def test_combined_cost_matches_independent_components(self):
    ranked = rank_horizon_candidates(database, inventory, query, target, config)
    self.assertTrue(torch.allclose(ranked.entry_cost, expected_entry))
    self.assertTrue(torch.allclose(ranked.outcome_cost, expected_outcome))
    self.assertTrue(torch.allclose(ranked.total_cost, expected_entry + expected_outcome))

def test_wrong_turn_and_zero_progress_are_hard_gated(self):
    ranked = rank_horizon_candidates(database, inventory, query, left_turn_target, config)
    self.assertEqual(ranked.candidate_indices.tolist(), [2])
```

- [ ] **Step 2: Verify RED**

Run the new module; expect missing-module failure.

- [ ] **Step 3: Implement exact 100-frame command prediction**

Reuse `bounded_velocity_step`, `bounded_yaw_step`, and `MatcherConfig.dt` for
100 sequential Torch steps. Return local desired XY displacement and accumulated
yaw at indices 24, 49, and 99. Do not extrapolate the existing 45-frame query.

- [ ] **Step 4: Implement vectorized layered ranking**

```python
@dataclass(frozen=True)
class HorizonSearchConfig:
    entry_weight: float = 1.0
    displacement_weight: float = 8.0
    yaw_weight: float = 3.0
    height_weight: float = 4.0
    duration_weight: float = 0.25
    stall_weight: float = 0.05
    moving_speed_mps: float = 0.10
```

Index `database._search_features` by `inventory.entry_row` for the entry cost.
Normalize displacement by `0.5 m`, yaw by `pi/2`, height by `0.18 m`, duration
by `100 frames`, and stall by `25 frames`. Hard-gate wrong turn sign when the
desired turn exceeds 30 degrees, zero displacement under moving commands,
wrong surface-height sign beyond 0.05 m, and non-finite rows. Rank with stable
Torch argsort on total cost.

- [ ] **Step 5: Implement ranked terrain validation and diagnostics**

Call the validator with `(record_index, entry_row, endpoint_exclusive)` in rank
order. Return the first accepted record with all component costs and counts for
`turn`, `progress`, `surface`, `nonfinite`, and `terrain`. Raise a structured
`HorizonSearchFailure` if none survives; never return a hold pose.

- [ ] **Step 6: Verify GREEN and parity**

Run the new tests plus current skill-search and motion-matcher suites. Expected:
all pass and equal-cost candidates preserve flattened inventory order.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py \
  tests/python/test_sonic_torch_terrain_skill_horizon_search.py
git commit -m "feat: rank multi-horizon terrain skills"
```

### Task 3: Stable-endpoint chunk execution

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_composer.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_composer.py`
- Create: `sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py`

**Interfaces:**
- Extends: `start_skill(..., playback_stop: int | None = None)` and `TerrainSkillState.playback_stop`.
- Produces: `TerrainSkillHorizonMatcher`, transactional `prepare_step`/`commit`, `HorizonChunkEvent`, `run_resolved_horizon_matrix`.

- [ ] **Step 1: Write failing explicit-endpoint composer tests**

```python
def test_explicit_endpoint_stops_before_whole_skill(self):
    state = start_skill(folder, skill, selected_entry_frame=3, current=pose, playback_stop=8)
    emitted = []
    while True:
        step = advance_skill(state); emitted.append(step.frame.source_frame)
        state = step.state
        if step.completed: break
    self.assertEqual(emitted, [3, 4, 5, 6, 7])
```

- [ ] **Step 2: Verify RED, implement endpoint validation, verify GREEN**

Require `selected_entry_frame < playback_stop <= skill.interval.playback_stop`
and exact double support at `playback_stop - 1`. Store the endpoint in every
new state and use it in `advance_skill` and `completed`.

- [ ] **Step 3: Write failing transactional rollout tests**

Tests require: selected horizon endpoint is honored; source frames are
consecutive; a changed command latches until double support; no search happens
inside a chunk; no candidate raises `HorizonSearchFailure`; and every
`HorizonChunkEvent` records row, target, endpoint, component costs, rejection
counts, and release reason.

- [ ] **Step 4: Wire the isolated transactional horizon matcher adapter**

Reuse current reset/FK/query state logic from `TerrainSkillMatcher`, but replace
whole-skill intent tensors and `select_terrain_skill` with the Task 1 inventory
and Task 2 selection. Terrain validation samples only from selected entry
through selected endpoint. A command change releases at the next double-support
frame. At normal completion, replan immediately; on failure, propagate the
structured error to the route harness.

- [ ] **Step 5: Verify GREEN and neighboring rollout tests**

Run composer, new adapter, current whole-skill rollout, omni-route, and FK tests.
Expected: all pass; the pre-existing authenticated opt-in test may skip.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_composer.py \
  tests/python/test_sonic_torch_terrain_skill_composer.py \
  sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py \
  tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py
git commit -m "feat: execute stable terrain skill horizons"
```

### Task 4: Six-GPU qualification and falsifiable evidence

**Files:**
- Create: `resources/run_g1_torch_multi_horizon_skills.py`
- Create: `tests/python/test_run_g1_torch_multi_horizon_skills.py`
- Create: `sonic/configs/experiments/torch_grail_multi_horizon_skills.json`
- Create: `docs/superpowers/results/2026-08-01-g1-multi-horizon-terrain-skills.md`

**Interfaces:**
- CLI accepts dataset/config/G1 XML/output/device/repeated route/qualification slice.
- Saves the existing pickle-free route matrix plus `chunk-events.json` per route.

- [ ] **Step 1: Write failing CLI and serialization contract tests**

Require exact six-route order, duplicate/unknown-route rejection, explicit
device selection, atomic non-overwriting output, canonical chunk event JSON,
and timing-independent deterministic identity.

- [ ] **Step 2: Verify RED and implement the CLI/config**

Use the current PHP config values plus a `multi_horizon` object containing the
exact Task 2 weights and horizons. Resolve the same query stair and reset clip.
The CLI returns zero when every route executes, regardless of behavior pass,
and nonzero on any route exception.

- [ ] **Step 3: Execute the enumerated focused and neighboring test modules**

Run the horizon, composer, whole-skill, matcher, contact, FK, CLI, and existing
omnidirectional route modules. Expected: all pass with only existing opt-in
authenticated skips.

- [ ] **Step 4: Run the frozen six-route slice concurrently**

Launch one route per `cuda:0` through `cuda:5`, each with a unique output under
`build/multi-horizon-terrain-skills/slice-v1`. Wait for all jobs and parse every
route's metrics and chunk events.

- [ ] **Step 5: Run the component-cost ablation**

For any failed route, rerun its shortest representative with entry-only,
outcome-only, and combined weights on separate GPUs. Do not tune the frozen
combined weights after viewing route outcomes; use the ablation only to classify
the failure.

- [ ] **Step 6: Record and commit evidence**

The report includes execution/pass counts, every failure reason, support p95,
progress, stall, heading, transition/chunk counts, deterministic hashes,
per-layer candidate counts, and the explicit continuation/falsifier decision.

```bash
git add resources/run_g1_torch_multi_horizon_skills.py \
  tests/python/test_run_g1_torch_multi_horizon_skills.py \
  sonic/configs/experiments/torch_grail_multi_horizon_skills.json
git commit -m "feat: qualify multi-horizon terrain skills"
git add docs/superpowers/results/2026-08-01-g1-multi-horizon-terrain-skills.md
git commit -m "docs: record multi-horizon terrain results"
```
