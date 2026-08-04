# G1 MotionBricks Terrain Task Actor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and certify one same-heading horizontal staircase traversal by supplying terrain-valid proxy keyframes to the released MotionBricks backbone.

**Architecture:** A pure task-actor module extracts sparse four-frame proxy windows and converts generated G1 trajectories into terrain contact evidence. A focused runner converts current and target G1 qpos into MotionBricks' native context/target constraints, generates each transition at 30 Hz, and retains only automatically certified output. The existing MuJoCo connector viewer plays the native-rate result.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo, released MotionBricks checkpoints, unittest.

## Global Constraints

- The first route is fixed: flat approach, supported step-up, unequal-height horizontal walking, supported step-down, flat departure.
- Preserve one same global heading throughout.
- MotionBricks generates every transition.
- Do not use ARDY, motion matching, route warping, or interpolated pose stitching in the generated route.
- Do not add Sonic or dynamic tracking.
- Do not modify the pinned MotionBricks checkout.
- Reuse the authoritative staircase sampler, G1 FK, sole geometry, and traversal validator.
- Work in `research/g1-torch-terrain-kinematics`; preserve all unrelated dirty changes.

---

### Task 1: Terrain task-actor proxy contract

**Files:**
- Create: `sonic/python/mm_sonic/torch_motionbricks_task_actor.py`
- Create: `tests/python/test_sonic_torch_motionbricks_task_actor.py`

**Interfaces:**
- Consumes: a route mapping containing `joint_position`, `root_position_world`, `root_orientation_world_wxyz`, and `source_support_mask`; ordered proxy endpoint frames; generated foot positions and sole clearances.
- Produces: `ProxyKeyframeSequence`, `extract_proxy_keyframes(...)`, `infer_generated_support(...)`, and `assemble_generated_route(...)`.

- [ ] **Step 1: Write failing proxy extraction tests**

```python
def test_extracts_ordered_four_frame_proxy_windows():
    route = valid_route(frame_count=40)
    sequence = extract_proxy_keyframes(route, endpoint_frames=(11, 23, 35))
    assert sequence.qpos.shape == (3, 4, 36)
    np.testing.assert_array_equal(
        sequence.support_mask[:, -1],
        route["source_support_mask"][[11, 23, 35]],
    )

def test_rejects_overlapping_or_airborne_proxy_endpoints():
    route = valid_route(frame_count=40)
    route["source_support_mask"][23] = False
    with self.assertRaises(ContractError):
        extract_proxy_keyframes(route, endpoint_frames=(11, 23, 35))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_motionbricks_task_actor -v
```

Expected: `ModuleNotFoundError` for `mm_sonic.torch_motionbricks_task_actor`.

- [ ] **Step 3: Implement validated proxy extraction**

Implement an immutable `ProxyKeyframeSequence` carrying contiguous
`qpos (proxies, 4, 36)`, `support_mask (proxies, 4, 2)`, and
`endpoint_frames`. Convert target-order route joints into native MuJoCo joint
order with `PINNED_TARGET_TO_SOURCE_PERMUTATION`. Reject malformed arrays,
non-unit quaternions, unordered/overlapping windows, and unsupported endpoint
frames.

- [ ] **Step 4: Write and verify support-inference tests**

```python
def test_infers_support_only_from_low_clearance_and_low_speed():
    support = infer_generated_support(
        foot_speed_mps=np.array([[0.03, 0.30], [0.04, 0.02]]),
        minimum_sole_clearance_m=np.array([[0.01, 0.01], [0.08, 0.00]]),
    )
    np.testing.assert_array_equal(
        support,
        np.array([[True, False], [False, True]]),
    )
```

Use `0.12 m/s` maximum stance speed and the existing `[-0.025, 0.035] m`
sole-contact band. Require at least three valid sole points before declaring
support.

- [ ] **Step 5: Implement route assembly and run GREEN**

`assemble_generated_route(...)` concatenates the first four context frames and
each generated transition after removing its four overlapping context frames.
It emits only route arrays and segment boundaries; it performs no pose blend or
root warp.

Run the Task 1 test command and require `OK`.

### Task 2: Direct MotionBricks target-pose conditioning

**Files:**
- Create: `resources/run_g1_motionbricks_terrain_task_actor.py`
- Create: `tests/python/test_run_g1_motionbricks_terrain_task_actor.py`

**Interfaces:**
- Consumes: a pinned MotionBricks checkout, a proxy traversal archive, endpoint frames, the staircase dataset/config, and the G1 XML.
- Produces: one native-30-Hz `traversal.npz`, `proxy-plan.json`, `candidate-metrics.json`, and `metrics.json`.

- [ ] **Step 1: Write a failing transform-decomposition test**

Use a fake converter returning known global joint positions and rotations.
Verify:

```python
constraints = motionbricks_constraints(fake_agent, context_qpos, target_qpos)
np.testing.assert_allclose(
    constraints["target_global_root_positions"],
    target_positions[:, :, 0] * np.array([1.0, 0.0, 1.0]),
)
np.testing.assert_allclose(
    constraints["target_global_joint_positions"]
    + constraints["target_global_root_positions"][:, :, None],
    target_positions,
)
```

Also verify all context and target tensors have four frames and the target
heading is derived from the target root rotation.

- [ ] **Step 2: Run the focused runner test and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_motionbricks_terrain_task_actor -v
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement direct keyframe constraints**

Create `motionbricks_constraints(agent, context_qpos, target_qpos)` using
`agent._converter.convert_mujoco_qpos_to_motion_transforms`. Decompose projected
root translation exactly as `clip_holder` does:

```python
root = positions[:, :, :1] * torch.tensor(
    (1.0, 0.0, 1.0), device=positions.device
)
relative_positions = positions - root
```

Return the keys consumed by `full_navigation_agent._generate_inbetween_frames`,
including four context transforms, four target transforms,
`target_global_root_positions`, and `target_root_headings`.

- [ ] **Step 4: Implement isolated native generation**

Initialize the released `navigation_demo` with
`force_canonicalization=False`, `source_root_realignment=False`,
`target_root_realignment=True`, and `pre_filter_qpos=False`. For every proxy
target, call `_generate_inbetween_frames` from the accepted prior four qpos
frames. Read the checkpoint's `min_tokens` and `max_tokens` values and search
every supported duration by passing one allowed token count at a time (the
released G1 checkpoint currently exposes 6 through 16 tokens, or 24 through 64
frames). Never patch the MotionBricks checkout.

- [ ] **Step 5: Run the runner tests and require GREEN**

The tests must cover malformed target shapes, exact projected-root
decomposition, and overlap-free segment assembly. Run the Task 2 test command
and require `OK`.

### Task 3: Terrain certification and candidate selection

**Files:**
- Modify: `resources/run_g1_motionbricks_terrain_task_actor.py`
- Modify: `tests/python/test_run_g1_motionbricks_terrain_task_actor.py`

**Interfaces:**
- Consumes: generated native qpos candidates plus authoritative foot/sole FK and target terrain sampling.
- Produces: deterministic candidate rejection reasons, scores, the accepted route, and validation artifacts.

- [ ] **Step 1: Write failing candidate-ranking tests**

```python
def test_rank_rejects_penetration_before_smoothness():
    bad = metrics(minimum_sole_clearance_m=-0.026, score=0.0)
    good = metrics(minimum_sole_clearance_m=-0.010, score=10.0)
    assert select_candidate((bad, good)).candidate_id == good.candidate_id

def test_rank_rejects_heading_change_and_unsupported_stance():
    with self.assertRaises(ContractError):
        select_candidate((
            metrics(maximum_heading_error_rad=0.31),
            metrics(unsupported_frame_count=1),
        ))
```

- [ ] **Step 2: Verify RED**

Run the Task 2 test command and require failure because candidate selection is
missing.

- [ ] **Step 3: Implement authoritative candidate metrics**

For every candidate:

1. compute feet with `MujocoG1FootKinematics`;
2. compute eight sole points per foot with `MujocoG1SoleKinematics`;
3. sample the target staircase under every foot and sole point;
4. infer support from foot speed, clearance, and at least three sole points;
5. reject minimum clearance below `-0.025 m`, stance contact error above
   `0.020 m`, stance slide above `0.010 m/frame`, heading error above
   `0.30 rad`, endpoint root error above `0.10 m`, or any unsupported frame
   outside the explicit step-down flight;
6. rank survivors lexicographically by endpoint error, stance slide,
   penetration margin, joint acceleration, and segment discontinuity.

- [ ] **Step 4: Generate the fixed route**

Use the certified proxy artifact
`build/g1-traversal-library/horizontal-full/motionbricks-contact-exit-v4/traversal.npz`
with default stable endpoints `29,73,132,201,232,257`. Write output under
`build/g1-traversal-library/horizontal-full/motionbricks-task-actor-v1/`.
Record every tried token length and rejection reason.

- [ ] **Step 5: Run full automatic validation**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  resources/run_g1_validate_traversal.py \
  --input build/g1-traversal-library/horizontal-full/motionbricks-task-actor-v1/traversal.npz \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-traversal-library/horizontal-full/motionbricks-task-actor-v1/validation.json
```

Expected: exit `0`; no hidden correction or unsupported frames.

### Task 4: Native-rate visual qualification

**Files:**
- Modify: `resources/run_g1_stair_pivot_viewer.py`
- Modify: `tests/python/test_run_g1_stair_pivot_viewer.py`

**Interfaces:**
- Consumes: `--frames-per-second` as a finite positive float.
- Produces: native-rate MotionBricks playback with unchanged kinematics.

- [ ] **Step 1: Write a failing parser test**

```python
args = MODULE._parser().parse_args((*required, "--frames-per-second", "30"))
self.assertEqual(args.frames_per_second, 30.0)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_stair_pivot_viewer -v
```

Expected: parser rejects `--frames-per-second`.

- [ ] **Step 3: Implement and verify native-rate playback**

Default to `50.0`; reject non-finite or non-positive values; replace the fixed
`0.02 s` clock with `1.0 / args.frames_per_second`. Run the focused test and
require `OK`.

- [ ] **Step 4: Run regression tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_motionbricks_task_actor \
  tests.python.test_run_g1_motionbricks_terrain_task_actor \
  tests.python.test_run_g1_stair_pivot_viewer \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: all tests pass.

- [ ] **Step 5: Launch the qualified route**

Close the native flat MotionBricks probe, then run:

```bash
DISPLAY=:1 PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  resources/run_g1_stair_pivot_viewer.py \
  --connector build/g1-traversal-library/horizontal-full/motionbricks-task-actor-v1/traversal.npz \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --frames-per-second 30 \
  --loop
```

Confirm the process and visible MuJoCo window remain live before requesting
visual feedback.
