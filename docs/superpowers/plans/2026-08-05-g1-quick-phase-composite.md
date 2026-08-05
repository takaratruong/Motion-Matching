# G1 Quick Horizontal Phase Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the certified `y=+0.8 m` mount, partial interior, and dismount into one validated kinematic traversal.

**Architecture:** A pure array composer rigidly shifts the interior, selects the mount/interior overlap, blends two boundaries, and reports splice metrics. A thin experiment runner loads the existing artifacts and source support masks, performs sole/contact certification on the target scene, and writes a viewer-ready connector.

**Tech Stack:** Python 3, NumPy, MuJoCo G1 forward kinematics, existing terrain sampler, `unittest`.

## Global Constraints

- Do not perform another corpus search.
- Do not edit source motions outside the two 10-frame blend horizons.
- Do not relax `-0.03 m` sole-clearance or `0.03 m` stance-error gates.
- Reject unsupported frames.
- Preserve normalized quaternions.

---

### Task 1: Pure three-phase composer

**Files:**
- Create: `sonic/python/mm_sonic/torch_path_phase_composite.py`
- Create: `tests/python/test_sonic_torch_path_phase_composite.py`

**Interfaces:**
- Consumes: three connector dictionaries and three Boolean support masks.
- Produces: `compose_path_phases(...) -> (arrays, support, metrics)`.

- [ ] **Step 1: Write failing behavior tests**

```python
arrays, support, metrics = compose_path_phases(
    mount=mount,
    interior=interior,
    dismount=dismount,
    mount_support=mount_support,
    interior_support=interior_support,
    dismount_support=dismount_support,
    blend_frames=10,
)
self.assertLess(metrics["mount_interior_root_gap_m"], 0.12)
self.assertEqual(arrays["joint_position"].shape[1], 29)
self.assertTrue(support.any(axis=1).all())
np.testing.assert_allclose(
    np.linalg.norm(arrays["root_orientation_world_wxyz"], axis=1), 1.0
)
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_path_phase_composite -v
```

Expected: fail because the composer module does not exist.

- [ ] **Step 3: Implement minimal composition**

Implement:

```python
def compose_path_phases(
    *,
    mount: dict[str, np.ndarray],
    interior: dict[str, np.ndarray],
    dismount: dict[str, np.ndarray],
    mount_support: np.ndarray,
    interior_support: np.ndarray,
    dismount_support: np.ndarray,
    blend_frames: int = 10,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    ...
```

Rigidly translate the interior endpoint onto the dismount entrance, search the
mount/interior overlap under the `0.12 m` root gate, blend outgoing prefixes
with cubic smoothstep and shortest-arc quaternion interpolation, concatenate,
and calculate maximum joint/root steps.

- [ ] **Step 4: Verify GREEN and commit**

```bash
git add sonic/python/mm_sonic/torch_path_phase_composite.py \
  tests/python/test_sonic_torch_path_phase_composite.py
git commit -m "feat: compose rigid horizontal path phases"
```

### Task 2: Terrain certification and artifact

**Files:**
- Create: `resources/run_g1_horizontal_phase_composite.py`
- Create: `tests/python/test_run_g1_horizontal_phase_composite.py`

**Interfaces:**
- Consumes: phase root, full source dataset, target dataset/config, and G1 XML.
- Produces: `composite.npz` and `metrics.json`.

- [ ] **Step 1: Write failing loader and gate tests**

```python
self.assertEqual(_phase_source_slice(summary, "mount"), (clip, start, stop))
self.assertFalse(_accepted_metrics({
    "minimum_sole_clearance_m": -0.031,
    "maximum_stance_error_m": 0.01,
    "unsupported_frame_count": 0,
}))
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_horizontal_phase_composite -v
```

Expected: fail because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Load each phase connector and its exact source support slice. Compute G1 ankle
and sole positions on the composed motion, sample the target height grid in
scene coordinates, calculate stance and sole metrics, set `accepted` only
under the frozen gates, and write the artifact.

- [ ] **Step 4: Run, inspect, and commit**

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python \
  resources/run_g1_horizontal_phase_composite.py \
  --phase-root build/g1-horizontal-grid-phase-20cm-v2 \
  --lane-id lane-pos-0p8 \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --config sonic/configs/experiments/torch_grail_raw_horizontal_preview.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-horizontal-phase-composite-pos-0p8-v1
```

Commit the runner and its tests after the artifact passes or records an
explicit rejection.

### Task 3: Visual and regression verification

- [ ] Run the new tests plus path placement, phase coverage, and viewer tests.
- [ ] Compile both new Python files and run `git diff --check`.
- [ ] Close the prior interior-playlist viewer and launch `composite.npz` at
  50 Hz only if `metrics.json` reports `"accepted": true`.
