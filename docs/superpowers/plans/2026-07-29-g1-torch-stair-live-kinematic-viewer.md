# G1 Torch Stair Live Kinematic Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native MuJoCo window that runs the dense small-stair Torch
matcher from live keyboard commands while using MuJoCo only for forward
kinematics and rendering.

**Architecture:** A new `torch_terrain_live_viewer` module owns three small
boundaries: pure keyboard-to-command mapping, authenticated G1/heightfield
scene construction plus `qpos` application, and a paced 50 Hz live loop. It
reuses the existing resolved stair configuration, matcher, height encoder, X11
provider, and joint permutation.

**Tech Stack:** Python 3.10, PyTorch/CUDA, MuJoCo 3.11 passive viewer, NumPy,
libX11, `unittest`.

## Global Constraints

- The module must never launch SONIC, GEAR, a policy, a simulator coordinator,
  or a transport.
- The module must contain no call to `mj_step`; only `mj_forward` may update
  MuJoCo kinematics.
- The matcher and command loop run at a requested 0.02-second interval.
- Inputs are the authenticated five-clip dataset, experiment config, and pinned
  29-DoF G1 XML.
- `W/S/A/D`, `Space`, `Backspace`, and `X` implement the approved operator
  contract.

---

### Task 1: Pure Live Command and State Conversion

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Create: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Consumes: `operator_x11.KeyLevels`,
  `joints.PINNED_TARGET_TO_SOURCE_PERMUTATION`.
- Produces:
  `LiveControlCommand`,
  `command_from_keys(pressed, reference_direction_xy, speed)`,
  `matcher_result_qpos(result)`.

- [ ] **Step 1: Write failing command tests**

Test zero input, `W/S/A/D`, diagonal normalization, `Space` precedence, and
finite shape validation. The reference direction `[0, 1]` must map `W` to
`[0, speed]`, `S` to its negative, `A` to the left perpendicular, and `D` to
the right perpendicular.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_live_viewer -v
```

Expected: import failure for `mm_sonic.torch_terrain_live_viewer`.

- [ ] **Step 3: Implement the minimal pure functions**

Normalize the combined planar key vector before multiplying by the positive
reference speed. Preserve the previous heading when stopped; use
`atan2(vy, vx)` when moving. Convert a matcher result to a finite `(36,)`
float64 `qpos`, with wxyz root orientation and the existing target-to-source
joint permutation.

- [ ] **Step 4: Re-run the focused test and confirm GREEN**

Run the command from Step 2. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py
git commit -m "feat: map live stair matcher controls"
```

### Task 2: Native MuJoCo Heightfield Renderer

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Consumes: `ResolvedStairConfig.measurement_extension`, a real G1 XML.
- Produces:
  `build_kinematic_scene(g1_xml, resolved) -> (MjModel, MjData)`,
  `apply_kinematic_state(mujoco, model, data, qpos) -> None`,
  `dense_patch_positions(result, measurement) -> np.ndarray`.

- [ ] **Step 1: Write failing scene tests**

Build a minimal 36-qpos G1 XML and synthetic resolved stair corpus. Require one
MuJoCo heightfield, finite terrain data, `nq == 36`, exact qpos application,
changed body positions after `mj_forward`, and source inspection proving the
new module contains no `mj_step(`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run the Task 1 command. Expected: missing scene functions.

- [ ] **Step 3: Implement scene construction**

Parse a copy of the pinned XML, make its `meshdir` absolute, remove the flat
floor, add an `hfield` asset and static geom, and compile it from an owned
temporary XML. Fill `model.hfield_data` from the authenticated query grid,
normalized by the declared Z extent. Place and rotate the heightfield by the
inverse matcher-to-scene alignment. `apply_kinematic_state` writes qpos and
calls only `mujoco.mj_forward`.

- [ ] **Step 4: Implement dense marker coordinates**

Reproduce the existing 13-by-7 dense sample locations using the matched root
yaw, map them to scene coordinates for height sampling, and return 91 points
in matcher coordinates with authenticated heights.

- [ ] **Step 5: Re-run focused tests and commit**

Expected: all live-viewer tests pass.

```bash
git add sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py
git commit -m "feat: render stair matches kinematically in MuJoCo"
```

### Task 3: 50 Hz Interactive Loop and CLI

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`
- Modify: `TORCH_TERRAIN_QUICKSTART.md`

**Interfaces:**
- Consumes:
  `resolve_stair_config`,
  `TerrainFeatureExtension.for_condition`,
  `TorchMotionMatcher.from_folder`,
  `X11KeyStateProvider`,
  Task 1/2 helpers.
- Produces:
  `run_live_viewer(...) -> None`,
  `build_live_viewer_argument_parser()`,
  module CLI.

- [ ] **Step 1: Write failing lifecycle and CLI tests**

Require flags `--dataset`, `--config`, `--g1-xml`, and `--device`; test edge
latching so held Backspace resets only once; test that `X` exits and cleanup
closes both viewer and X11 provider.

- [ ] **Step 2: Run the focused test and confirm RED**

Run the Task 1 command. Expected: missing lifecycle/CLI interfaces.

- [ ] **Step 3: Implement the live loop**

Resolve the dense condition, reset the matcher, build and open the passive
MuJoCo viewer, then sample focused X11 levels at each tick. Reset on the
Backspace rising edge, exit on `X`, otherwise call
`matcher.prepare_step(..., dt=0.02)` and `matcher.commit(...)`. Apply qpos,
update 91 sphere markers, follow the root with the camera, set the diagnostic
text overlay, call `viewer.sync()`, and sleep only for the remaining tick
budget. Release viewer and X11 resources in `finally`.

- [ ] **Step 4: Document the exact launch command**

Add:

```bash
DISPLAY=:1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -B -m \
  mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda
```

State explicitly that this calls `mj_forward`, never `mj_step`, and never
launches SONIC.

- [ ] **Step 5: Run focused and regression tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_live_viewer \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_viewer -v
```

Expected: all tests pass.

- [ ] **Step 6: Run a bounded CUDA smoke**

Start the CLI on `DISPLAY=:1`, require a live MuJoCo window and process, then
close it through the operator contract. Confirm no SONIC, GEAR, or gated
simulator process belongs to the run.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py \
  TORCH_TERRAIN_QUICKSTART.md
git commit -m "feat: add live MuJoCo stair kinematics viewer"
```

