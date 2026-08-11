# Native-G1 Terrain PFNN Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the trained terrain-aware native-G1 PFNN directly on authenticated ramp and stair scenes with no custom playback or pose correction.

**Architecture:** `TerrainPFNNRuntime` remains byte-for-byte unchanged and owns every motion output. A small viewer adapter supplies the authenticated scene height/gradient in PFNN course coordinates and rigidly maps the emitted root frame into MuJoCo; joints and the full quaternion are copied exactly.

**Tech Stack:** Python 3.11, PyTorch, NumPy, MuJoCo, pytest.

## Global Constraints

- Do not modify `sonic/python/mm_sonic/terrain_pfnn/runtime.py` or any checkpoint/dataset artifact.
- Do not instantiate or call `HybridMatcher`, `ExistingUtilityPosePostprocessor`, pose repair, or foot locking in the PFNN viewer path.
- Keep `command_driven_root=False`; PFNN owns planar root motion and learned support-relative root height.
- Keep `hold_idle_pose=False`; neutral input must pass through PFNN rather than a viewer hold branch.
- Preserve the complete PFNN root quaternion and all 29 emitted joints exactly after the existing IsaacLab-to-MuJoCo permutation.
- Terrain model input and rendered mesh must come from the same loaded `SceneTerrainAdapter`.
- Preserve the existing terrain-fit demo when `--scene` is absent; reject ambiguous simultaneous terrain authorities.
- Preserve the user's existing modified report files without editing or staging them.

---

### Task 1: Exact authenticated-scene viewer adapter

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn_viewer.py`
- Modify: `tests/python/test_terrain_pfnn_viewer.py`

**Interfaces:**
- Consumes: `load_scene_terrain(scene, terrain_root=...) -> SceneTerrainAdapter`, `TerrainSample`, and `PFNNRuntimeFrame`.
- Produces: `_ScenePFNNTerrainCallback`, exact `_apply_frame`, and CLI options `--scene` / `--terrain-root`.

- [ ] **Step 1: Write failing tests for the rigid scene callback**

  Add a two-by-two sloped `SceneTerrainAdapter` fixture. Assert local PFNN
  `(0, 0)` maps to its scene spawn, local +X maps along the scene heading,
  returned height equals `authority.height`, returned gradient is expressed in
  local axes, and the callback's vertices/faces are exactly `native_mesh()`.

- [ ] **Step 2: Run the callback tests and verify RED**

  Run:
  `PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/python/test_terrain_pfnn_viewer.py -k 'scene_terrain_callback'`

  Expected: failure because `_ScenePFNNTerrainCallback` is absent.

- [ ] **Step 3: Write failing tests for exact PFNN frame rendering**

  Construct a PFNN frame with nonzero roll, pitch, and yaw plus 29 distinct
  joints. Assert `_apply_frame` writes the rigidly placed XYZ, the normalized
  premultiplied full quaternion, and the exact existing joint permutation.
  Assert no `display_joints_mujoco` override or `_blend_display_joints` call is
  used by `advance`.
  Add a construction test proving `_load_runtime` passes
  `command_driven_root=False` and `hold_idle_pose=False`.

- [ ] **Step 4: Run exact-frame tests and verify RED**

  Run:
  `PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/python/test_terrain_pfnn_viewer.py -k 'exact_frame or full_quaternion'`

  Expected: the current yaw-only quaternion/display-blend behavior fails.

- [ ] **Step 5: Implement the minimal adapter and direct renderer**

  Add `_ScenePFNNTerrainCallback(adapter)` with a 2-D rotation determined only
  by `adapter.spawn_heading`, scene height sampling through
  `adapter.authority`, centered finite differences no larger than one scene
  cell for the local gradient, `collision_heights_at`, and the unchanged native
  mesh. Add root-position and quaternion placement helpers. Change
  `_apply_frame` to use the complete frame quaternion and exact joint vector.
  Remove `_blend_display_joints` from the interactive advance path. Add optional
  CLI `--scene` and `--terrain-root`; reject `--scene` together with an explicit
  `--terrain-fit`. Change only `_load_runtime`'s viewer-owned idle configuration
  from `hold_idle_pose=True` to `hold_idle_pose=False`; otherwise leave PFNN
  construction unchanged.

- [ ] **Step 6: Run focused and full viewer tests**

  Run:
  `PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/python/test_terrain_pfnn_viewer.py`

  Expected: all tests pass.

- [ ] **Step 7: Run static checks and commit**

  Run Ruff check/format-check and `python -m py_compile` on the two files, then
  `git diff --check`. Commit only the two scoped code/test files with message
  `fix: render terrain PFNN output directly`.

### Task 2: Frozen ramp/stair verification and visible launch

**Files:**
- Modify only tests if a verification defect requires a regression; otherwise no repository edits.

**Interfaces:**
- Consumes: Task 1 CLI, frozen checkpoint/dataset, `ramp-10-up-down`, and `stairs-standard`.
- Produces: bounded trace evidence and a live MuJoCo ramp viewer.

- [ ] **Step 1: Authenticate artifacts and run CPU/static preflight**

  Pin HEAD plus checkpoint, dataset, scene, source, and test hashes. Confirm the
  selected manifest includes `terrain_slopes__*` and `WalkingUpSteps*` training
  sources. Confirm no hybrid/postprocessor symbol is reachable from the PFNN
  scene execution path.

- [ ] **Step 2: Run bounded terrain-PFNN smoke routes**

  Use the existing trained checkpoint and dataset with `--scene
  ramp-10-up-down` and then `stairs-standard`. Drive the model forward and
  reverse through rise/drop sections. Require finite frames, positive phase
  advancement under motion, model-owned nonzero planar root deltas, terrain
  rise/drop, and no hold reason. Do not add corrective runtime code to make a
  failed route pass.

- [ ] **Step 3: Independent code review**

  Review the complete Task 1 diff for runtime isolation, exact output mapping,
  terrain/mesh identity, and test discrimination. Fix every Critical or
  Important finding with a focused regression and re-review.

- [ ] **Step 4: Launch the visible ramp viewer**

  After an idle-GPU check, launch the exact committed viewer on the ramp, verify
  its MuJoCo window and expected GPU identity, and leave it running for the
  user. Do not inject controls or modify the process after readiness.
