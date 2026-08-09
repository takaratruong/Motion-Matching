# G1 PFNN Sample Retarget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget the released `LocomotionFlat01_000.bvh` clip to native 29-DoF G1 motion at 120 Hz and open an interactive MuJoCo viewer for user approval before any corpus or PFNN work.

**Architecture:** Reuse pinned GMR for BVH parsing, proportional scaling, Mink/MuJoCo IK, and joint limits. Add only a PFNN-source adapter (`Spine1 -> Spine2`), use the pinned `retargeting_project` grounding implementation, save a validated neutral artifact, and play it through GMR's existing `RobotMotionViewer` with pause/seek controls.

**Tech Stack:** Python 3.10, NumPy, MuJoCo, Mink, GMR, `takaratruong/retargeting_project`, unittest.

**Execution mode:** Execute inline in this worktree using the `executing-plans` and `test-driven-development` skills. Do not delegate or modify the existing PFNN training path.

## Global Constraints

- Source motion: `/home/ubuntu/datasets/pfnn/pfnn/data/animations/LocomotionFlat01_000.bvh`.
- Preserve every source frame and record `fps=120`; do not decimate the approval artifact.
- Pin `takaratruong/retargeting_project` to `fb3433a6310ab4198102d3905e74b73944fc1f6b`.
- Pin `YanjieZe/GMR` to `bb1bbe40774794fceb2a7c579a3464a28e68c844`.
- Use GMR's `bvh_nokov -> unitree_g1` mapping and the retargeting project's grounding function.
- The only source-skeleton alias is exact `Spine1 -> Spine2`.
- Do not modify either external repository.
- Do not begin batch retargeting, terrain fitting, dataset generation, or PFNN training.
- Output goes under ignored `sonic/runs/native-g1-pfnn/sample-retarget/`.
- User approval of the displayed motion is the terminal gate.

---

### Task 1: Add the native PFNN-BVH retarget adapter

**Files:**
- Create: `sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py`
- Create: `tests/python/test_retarget_pfnn_bvh_g1.py`

**Interfaces:**
- Produces: `prepare_pfnn_bvh(source: Path, destination: Path) -> SourceReceipt`.
- Produces: `validate_g1_motion(motion: object, *, expected_frames: int, expected_fps: float, joint_limits: np.ndarray) -> None`.
- Produces: CLI `python -m mm_sonic.retarget_pfnn_bvh_g1 --source ... --gmr-root ... --retarget-project-root ... --output ...`.

- [ ] **Step 1: Write the source-adapter RED test**

Create a two-frame synthetic BVH containing `Spine1`, `LeftToeBase`, and `RightToeBase`. Assert that `prepare_pfnn_bvh`:

```python
receipt = prepare_pfnn_bvh(source, prepared)
self.assertEqual(receipt.frame_count, 2)
self.assertEqual(receipt.fps, 120.0)
self.assertIn("JOINT Spine2", prepared.read_text())
self.assertNotIn("JOINT Spine1", prepared.read_text())
self.assertEqual(receipt.aliases, (("Spine1", "Spine2"),))
```

Also assert rejection for a missing `Spine1`, an existing `Spine2`, a frame-count mismatch, a non-120-Hz frame time, or either missing toe-base joint.

- [ ] **Step 2: Run the adapter test and observe RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_retarget_pfnn_bvh_g1.PFNNBVHRetargetTests.test_prepares_native_120_hz_source
```

Expected: error importing `mm_sonic.retarget_pfnn_bvh_g1`.

- [ ] **Step 3: Implement exact source preparation**

Implement `SourceReceipt` as a frozen dataclass containing source SHA-256, prepared SHA-256, frame count, FPS, and the exact alias tuple. Parse `Frames:` and `Frame Time:` without rewriting motion rows. Require `round(1 / frame_time) == 120`, replace exactly one hierarchy declaration matching `JOINT Spine1`, and write the prepared file atomically.

- [ ] **Step 4: Write the artifact-validator RED test**

Use a synthetic motion object and G1 joint limits. Require shapes `[T,3]`, `[T,4]`, and `[T,29]`; finite values; unit XYZW quaternions within `1e-5`; all joints within native limits plus `1e-6`; exactly 120 Hz; and exact frame count. Independently corrupt each field and assert rejection.

- [ ] **Step 5: Run the validator test and observe RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_retarget_pfnn_bvh_g1.PFNNBVHRetargetTests.test_validates_native_g1_artifact
```

Expected: failure because `validate_g1_motion` is missing.

- [ ] **Step 6: Implement the minimal validator and GMR call**

The CLI must:

1. prepare a temporary aliased BVH;
2. import pinned GMR from `--gmr-root`;
3. call `load_bvh_file(prepared, format="nokov")`;
4. construct `GeneralMotionRetargeting(src_human="bvh_nokov", tgt_robot="unitree_g1", actual_human_height=height)`;
5. call `retargeter.retarget(frame)` sequentially for every native frame;
6. convert GMR root quaternion WXYZ to XYZW;
7. instantiate the pinned project's `CommonMotion`;
8. set the pinned project's `retarget.GMR` path and call its existing `postprocess` grounding function;
9. validate against the GMR MuJoCo model's joint limits; and
10. atomically save `motion.npz` plus `receipt.json` containing source/prepared/output hashes, pinned commits, FPS, frame count, joint order, and grounding offset.

No pose filter, smoothing, contact lock, or downstream correction is permitted.

- [ ] **Step 7: Run the complete adapter suite**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_retarget_pfnn_bvh_g1
```

Expected: all tests pass.

- [ ] **Step 8: Commit the adapter**

```bash
git add sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py \
  tests/python/test_retarget_pfnn_bvh_g1.py
git commit -m "feat: retarget native PFNN BVH to G1"
```

---

### Task 2: Add the interactive sample viewer

**Files:**
- Create: `sonic/python/mm_sonic/view_g1_retarget.py`
- Modify: `tests/python/test_retarget_pfnn_bvh_g1.py`

**Interfaces:**
- Consumes: validated neutral `motion.npz` from Task 1.
- Produces: CLI `python -m mm_sonic.view_g1_retarget --motion ... --gmr-root ...`.

- [ ] **Step 1: Write viewer-state RED tests**

Define a pure `PlaybackState(frame_count: int)` and assert:

```python
state.toggle_pause()
self.assertTrue(state.paused)
state.seek(+1)
self.assertEqual(state.frame_index, 1)
state.seek(-2)
self.assertEqual(state.frame_index, state.frame_count - 1)
state.home()
self.assertEqual(state.frame_index, 0)
```

Assert `advance()` wraps only when not paused.

- [ ] **Step 2: Run the viewer-state test and observe RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_retarget_pfnn_bvh_g1.PFNNBVHRetargetTests.test_viewer_playback_controls
```

Expected: import error for `mm_sonic.view_g1_retarget`.

- [ ] **Step 3: Implement the viewer**

Load the neutral artifact with `allow_pickle=False`, validate it, construct GMR's existing `RobotMotionViewer(robot_type="unitree_g1", motion_fps=120, camera_follow=True)`, and install these controls:

- Space: pause/resume.
- Right arrow: seek one frame forward.
- Left arrow: seek one frame backward.
- Home: return to frame zero.
- Escape: close.

Each rendered qpos is exactly the stored root position, XYZW-to-WXYZ quaternion, and 29 joint coordinates. Display the current frame, time, FPS, and paused state on stdout once per second.

- [ ] **Step 4: Run focused tests and static checks**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_retarget_pfnn_bvh_g1
sonic/.torch-mm-venv/bin/python -m py_compile \
  sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py \
  sonic/python/mm_sonic/view_g1_retarget.py \
  tests/python/test_retarget_pfnn_bvh_g1.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the viewer**

```bash
git add sonic/python/mm_sonic/view_g1_retarget.py \
  tests/python/test_retarget_pfnn_bvh_g1.py
git commit -m "feat: inspect native G1 retargets"
```

---

### Task 3: Produce and review the real 120 Hz sample

**Files:**
- Generate: `sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.npz`
- Generate: `sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.receipt.json`

**Interfaces:**
- Consumes: exact pinned external repositories and released PFNN BVH.
- Produces: the user-visible approval artifact; no later task may start without approval.

- [ ] **Step 1: Prepare pinned dependencies in an isolated cache**

Clone or verify exact commits under `/home/ubuntu/.cache/native-g1-pfnn/`, create `/home/ubuntu/.cache/native-g1-pfnn/venv`, and install pinned GMR plus the linked project's runtime dependencies. Reject dirty repositories or wrong commits.

- [ ] **Step 2: Run the real native-rate retarget**

```bash
/home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m mm_sonic.retarget_pfnn_bvh_g1 \
  --source /home/ubuntu/datasets/pfnn/pfnn/data/animations/LocomotionFlat01_000.bvh \
  --gmr-root /home/ubuntu/.cache/native-g1-pfnn/GMR \
  --retarget-project-root /home/ubuntu/.cache/native-g1-pfnn/retargeting_project \
  --output sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.npz
```

Expected: 8,171 finite frames, 120 Hz, 29 joint coordinates, normalized quaternions, native-limit compliance, and an accepted receipt.

- [ ] **Step 3: Perform a headless artifact inspection**

Report exact root-height range, root displacement, per-joint range, minimum heel/toe height, maximum joint-limit margin, quaternion norm error, and output hashes. Reject the sample before opening the viewer if any invariant fails.

- [ ] **Step 4: Open the viewer for user approval**

```bash
PYTHONPATH=sonic/python \
  /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m mm_sonic.view_g1_retarget \
  --motion sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.npz \
  --gmr-root /home/ubuntu/.cache/native-g1-pfnn/GMR
```

Keep the viewer open. Tell the user the controls and request approval or the first visibly bad frame. Do not begin any later PFNN work in this session.

## Completion Definition

This plan is complete only when the 120 Hz G1 motion is visible in MuJoCo and the user can pause and seek it. User rejection is a successful diagnostic outcome but blocks every batch-retargeting, terrain, dataset, training, and PFNN-runtime task.
