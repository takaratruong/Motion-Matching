# G1 Terrain Landing Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that a deterministic contact-IK action can fill the measured lateral stair-landing hole without perturbing ordinary motion matching.

**Architecture:** Extend the existing MuJoCo G1 FK adapter with a bounded position-only leg IK solve, then build an isolated landing-bridge generator that retargets the authenticated multi-support curb reference to query-terrain foot contacts. Qualify the bridge offline at the lower side-exit boundary before adding any behavior-changing matcher hook.

**Tech Stack:** Python 3.10, NumPy, PyTorch, native MuJoCo Python bindings, `unittest`.

## Global Constraints

- Remain deterministic 50 Hz kinematics only; do not add Sonic, tracking, physics stepping, depth, or learned models.
- Preserve ordinary `TorchMotionMatcher` output exactly until a real lower-exit bridge passes all validators.
- Change only the 12 G1 leg joints; preserve the floating root, root orientation, and 17 non-leg joints during each IK solve.
- Use the existing 3 cm penetration, 10-frame unsupported-run, and 5% lost-source-support gates without relaxation.
- Every failure rejects the complete bridge and leaves the ordinary matcher state unchanged.

---

### Task 1: Bounded G1 foot-position IK

**Files:**
- Modify: `sonic/python/mm_sonic/torch_g1_fk.py`
- Modify: `tests/python/test_sonic_torch_g1_fk.py`

**Interfaces:**
- Consumes: target-ordered 29-joint pose, root position/quaternion, shape-`(2,)` solve mask, shape-`(2,3)` ankle targets.
- Produces: `MujocoG1FootKinematics.solve_leg_positions(...) -> np.ndarray` with shape `(29,)`.

- [ ] **Step 1: Write failing reachable-target and preservation tests**

Construct the real G1 helper at the zero joint pose and root `[0, 0, 0.8]`.
Move only the left ankle target by `+0.01 m` in world X and assert the returned
pose reaches it within `0.005 m`. Assert target joint indices 6--28 and all six
right-leg indices remain bitwise equal to the input.

```python
target = feet.copy()
target[0, 0] += 0.01
solved = helper.solve_leg_positions(
    joints,
    root,
    quaternion,
    np.array([True, False]),
    target,
)
actual = helper.foot_positions(solved[None], root[None], quaternion[None])[0]
self.assertLessEqual(np.linalg.norm(actual[0] - target[0]), 0.005)
left_leg = {0, 3, 6, 9, 13, 17}
preserved = [index for index in range(29) if index not in left_leg]
np.testing.assert_array_equal(solved[preserved], joints[preserved])
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_g1_fk
```

Expected: `AttributeError` because `solve_leg_positions` does not exist.

- [ ] **Step 3: Implement the bounded damped-least-squares solver**

During `_initialize`, resolve the six named hinge joints per leg and retain
their `jnt_qposadr`, `jnt_dofadr`, and limits. For each of at most 32
iterations, run `mj_forward`, stack `mj_jacBody` position rows for selected
ankles, restrict columns to the selected leg DOFs, and apply:

```python
delta = jacobian.T @ np.linalg.solve(
    jacobian @ jacobian.T + 1e-4 * np.eye(error.size),
    error,
)
delta = np.clip(delta, -0.10, 0.10)
```

Clamp updated hinge qpos to `model.jnt_range`. Return the target-ordered pose
when every selected ankle is within 5 mm; otherwise raise
`ContractError("G1 foot IK target is unreachable")`. Validate all shapes,
finite values, nonempty solve mask, and unit quaternion before mutating the
private MuJoCo data.

- [ ] **Step 4: Add unreachable and malformed-input tests**

Use a target two metres below the ankle and require deterministic
`ContractError("unreachable")`. Cover a false/false mask, wrong target shape,
NaN, and nonunit root quaternion.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all G1 FK/IK tests pass.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_g1_fk.py \
  tests/python/test_sonic_torch_g1_fk.py
git commit -m "feat: solve bounded G1 foot IK"
```

---

### Task 2: Immutable terrain landing bridge generator

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_landing_bridge.py`
- Create: `tests/python/test_sonic_torch_terrain_landing_bridge.py`

**Interfaces:**
- Consumes: `TerrainDataset`, `TerrainFeatureExtension`, `MujocoG1FootKinematics`, source `ContactSegment`, start joint/root state, and command direction.
- Produces: immutable `TerrainLandingBridge` arrays for joints, joint velocities, roots, root quaternions, and feet, or `ContractError`.

- [ ] **Step 1: Write failing immutable-output and source-window tests**

Define `TerrainLandingBridge` with owned read-only float64 arrays and exact
shapes `(frames,29)`, `(frames,29)`, `(frames,3)`, `(frames,4)`, and
`(frames,2,3)`. Define `LandingBridgeReference(clip_index=14,
start_frame=361,end_frame=421)` and require exactly 60 frames and the recorded
support mask slice.

- [ ] **Step 2: Run the new test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_landing_bridge
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement deterministic reference alignment and contact targets**

Normalize the command direction, align source root yaw to the supplied start
yaw, and rigidly align source frame 361 root XY to the supplied start root XY.
Use authenticated source support transitions to lock a supported ankle at its
current target. At each new support onset, search query terrain along the
command direction in 2 cm increments from 0.10 m through 0.80 m and choose the
first sample at least 5 cm below the current support surface. Set target ankle
Z to sampled surface plus `ANKLE_ORIGIN_SOLE_M`. Interpolate swing XY with
cubic smoothstep and add a `4*h*t*(1-t)` clearance arc where `h=0.10 m`.

- [ ] **Step 4: Implement sequential per-frame IK and velocity derivation**

Seed frame zero from the supplied start joints. For each later frame, start
from the preceding solved joints, solve every source-supported foot to its
locked target, and solve a swing foot to its interpolated target only after
the source foot has cleared support. Reject an IK discontinuity above 0.35 rad
per joint per frame. Derive velocities by forward difference at 0.02 s, copy
frame-one velocity into frame zero, and compute final feet through authoritative
FK.

- [ ] **Step 5: Add fail-closed unit tests**

With small fake grids and a scripted IK adapter, cover no lower foothold,
out-of-domain sampling, an IK exception, excessive joint discontinuity, and
mutation of returned arrays. Require every case to reject before returning a
partial bridge.

- [ ] **Step 6: Run bridge tests and verify GREEN**

Run the Step 2 command. Expected: all bridge tests pass.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_landing_bridge.py \
  tests/python/test_sonic_torch_terrain_landing_bridge.py
git commit -m "feat: generate terrain landing bridges"
```

---

### Task 3: Real lower-exit feasibility oracle

**Files:**
- Create: `resources/run_g1_torch_landing_bridge.py`
- Create: `tests/python/test_run_g1_torch_landing_bridge.py`

**Interfaces:**
- Consumes: representative corpus, contact config, G1 XML, route name, capture frame.
- Produces: deterministic JSON with source identity, capture state, IK residuals, contact validation, terrain validation, and accepted/rejected result.

- [ ] **Step 1: Write failing CLI/source-contract tests**

Require `--dataset`, `--config`, `--g1-xml`, `--route`, `--capture-frame`, and
`--output`. Reject routes other than the four frozen side exits. Patch the
bridge generator and assert that the CLI passes the committed matcher state at
the exact capture frame without modifying that matcher.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_run_g1_torch_landing_bridge
```

Expected: import failure because the resource runner does not exist.

- [ ] **Step 3: Implement read-only capture and validation**

Replay the named route through the real contact matcher to the requested frame,
copy the committed state, generate the bridge, construct a matching
`SegmentPlacement` over frames 361--421, and call both
`TerrainContactSegmentPolicy.validate_emitted` and the configured emitted
window validator. Save JSON atomically. Exclude wall-clock timing from the
deterministic digest.

- [ ] **Step 4: Run the lower-left command-boundary oracle**

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_torch_landing_bridge.py \
  --dataset build/torch-grail-terrain-representative \
  --config sonic/configs/experiments/torch_grail_contact_segment.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --route side-exit-lower-left --capture-frame 195 \
  --output build/g1-lower-left-landing-bridge-v1.json
```

Expected Phase-A gate: accepted contact validation, accepted terrain preview,
minimum clearance at least `-0.03 m`, unsupported run at most 10 frames, and
final two ankle surfaces on flat terrain. If this gate fails, preserve the
evidence and stop before any matcher admission change.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_g1_fk \
  tests.python.test_sonic_torch_terrain_landing_bridge \
  tests.python.test_run_g1_torch_landing_bridge
git add resources/run_g1_torch_landing_bridge.py \
  tests/python/test_run_g1_torch_landing_bridge.py
git commit -m "feat: evaluate G1 landing bridge feasibility"
```

---

### Task 4: Admit only a qualified bridge and re-run the matrix

**Files:**
- Modify only after Task 3 passes: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify only after Task 3 passes: `tests/python/test_sonic_torch_motion_matcher.py`
- Modify only after Task 3 passes: `sonic/python/mm_sonic/torch_terrain_omni_rollout.py`
- Modify only after Task 3 passes: `tests/python/test_sonic_torch_terrain_omni_rollout.py`

**Interfaces:**
- Consumes: a provider returning a fully validated `TerrainLandingBridge` or `None`.
- Produces: transactional sequential bridge playback with ordinary matcher fallback.

- [ ] **Step 1: Write failing transactional playback tests**

Inject a scripted provider. Require bridge playback to emit frames 0--59
sequentially, reject prepare/commit token misuse through the existing
transaction contract, and return the ordinary candidate unchanged when the
provider returns `None` or raises `ContractError`.

- [ ] **Step 2: Verify RED, implement the minimal optional provider hook, and verify GREEN**

The hook is consulted only on a forced lateral/down search after ordinary
terrain rescue finds no safe substantial-down action. Store bridge playback in
an immutable commitment distinct from source-clip commitments. Do not alter
the no-provider path. Run both matcher and omni-rollout suites.

- [ ] **Step 3: Run all 21 routes across eight GPUs**

Use a 30-second diagnostic ceiling rather than the runner's one-second default.
Require at least 15 clean passes, one lower side exit flat, all prior 14 clean
routes retained, and zero rescue cycles. If any retained route regresses,
remove the admission hook while preserving the Phase-A generator and evidence.

- [ ] **Step 4: Run the complete Python suite and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover -s tests/python -v
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  sonic/python/mm_sonic/torch_terrain_omni_rollout.py \
  tests/python/test_sonic_torch_motion_matcher.py \
  tests/python/test_sonic_torch_terrain_omni_rollout.py
git commit -m "feat: admit qualified terrain landing bridges"
```
