# PFNN Terrain Transfer Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the approved G1 stair retarget on the exact rank-zero PFNN terrain fit and report heel/toe gaps before any corrective IK.

**Architecture:** A new extraction module reproduces PFNN's released patch fitting and saves a safe numeric terrain artifact. A comparison module evaluates native G1 sole probes against that same continuous fit. The existing viewer gains an exact PFNN mesh mode and contact overlay while the inferred `auto-steps` mode remains diagnostic-only.

**Tech Stack:** Python 3.10, NumPy, SciPy, released PFNN BVH utilities, pinned GMR, MuJoCo, `unittest`.

## Global Constraints

- Use `WalkingUpSteps01_000.bvh`, display source frames 8160 through 8279 at exactly 120 Hz.
- Use the PFNN footstep cycle 8135 through 8229 and rank-zero fitted patch.
- Reproduce PFNN position scale 5.6444, patch seed 2, patch size 128, horizontal scale 3.937007874, vertical scale 3.0, and linear RBF smooth value 0.1.
- Preserve the released terrain; never infer terrain from G1 feet or rescale it to G1 morphology.
- Do not add GRAIL, train a model, change runtime control, or add corrective IK in this plan.
- All production changes use test-first RED/GREEN cycles.

---

## File Structure

- Create `sonic/python/mm_sonic/pfnn_terrain_fit.py`: safe terrain-fit types, PFNN math, extraction CLI, artifact validation, and continuous queries.
- Create `sonic/python/mm_sonic/compare_pfnn_terrain_g1.py`: MuJoCo sole probes, stance-gap classification, report CLI.
- Modify `sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py`: allow source-world height instead of flat grounding.
- Modify `sonic/python/mm_sonic/view_g1_retarget.py`: load the exact terrain, render its mesh, and display sole markers.
- Create `tests/python/test_pfnn_terrain_transfer.py`: focused terrain and comparison tests.
- Modify `tests/python/test_retarget_pfnn_bvh_g1.py`: source-grounding regression.

### Task 1: Extract the exact PFNN fitted terrain

**Files:**
- Create: `sonic/python/mm_sonic/pfnn_terrain_fit.py`
- Create: `tests/python/test_pfnn_terrain_transfer.py`

**Interfaces:**
- Produces: `PFNNTerrainFit`, `patch_height`, `terrain_height_pfnn`, `terrain_height_g1`, `fit_terrain_cycle`, `save_terrain_fit`, `load_terrain_fit`.
- Consumes: released `patches.npz`, PFNN BVH/motion modules, BVH/footstep paths, source frame interval.

- [ ] **Step 1: Write failing math and artifact tests**

```python
def test_patch_height_matches_reference_bilinear_sampling():
    patch = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    query = np.array([[0.0, 0.0], [1.968503937, 1.968503937]])
    actual = patch_height(patch, query, hscale=3.937007874, vscale=3.0)
    np.testing.assert_allclose(actual, reference_patchfunc(patch, query), atol=0, rtol=0)

def test_terrain_artifact_rejects_tampered_rbf_weights(tmp_path):
    path = write_valid_fit(tmp_path)
    tamper_npz_field(path, "rbf_weights", np.array([[np.nan]]))
    with self.assertRaisesRegex(ValueError, "rbf_weights"):
        load_terrain_fit(path)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_pfnn_terrain_transfer
```

Expected: import failure for `mm_sonic.pfnn_terrain_fit`.

- [ ] **Step 3: Implement exact query and safe artifact contracts**

```python
@dataclass(frozen=True)
class PFNNTerrainFit:
    patch: np.ndarray
    patch_coord: np.ndarray
    contact_center_xz: np.ndarray
    patch_height_mean: float
    stance_height_mean: float
    rbf_centers_xz: np.ndarray
    rbf_epsilon: np.ndarray
    rbf_weights: np.ndarray
    source_contacts: np.ndarray
    source_start_frame: int
    source_frame_count: int

def terrain_height_pfnn(fit: PFNNTerrainFit, xz: np.ndarray) -> np.ndarray:
    base = patch_height(fit.patch[None], xz - fit.contact_center_xz)[0]
    base = base - fit.patch_height_mean + fit.stance_height_mean
    distance = scipy.spatial.distance.cdist(xz, fit.rbf_centers_xz)
    return base + (fit.rbf_weights @ (fit.rbf_epsilon * distance).T).T[:, 0]

def terrain_height_g1(fit: PFNNTerrainFit, xy_m: np.ndarray) -> np.ndarray:
    source_xz = np.column_stack((100.0 * xy_m[:, 0], -100.0 * xy_m[:, 1]))
    return terrain_height_pfnn(fit, source_xz) / 100.0
```

- [ ] **Step 4: Add fitting tests for cycle selection, rank, contacts, and determinism**

Use a bounded synthetic patch bank to prove exact `np.argsort(error)[0]`, source contact mapping from 60 Hz to 120 Hz, RBF serialization, and identical semantic receipts across two fresh writes.

- [ ] **Step 5: Implement `fit_terrain_cycle` and CLI**

The CLI takes exact paths rather than discovering inputs:

```bash
python -m mm_sonic.pfnn_terrain_fit \
  --pfnn-root /home/ubuntu/datasets/pfnn/pfnn \
  --patches /home/ubuntu/datasets/pfnn/pfnn/patches.npz \
  --source data/animations/WalkingUpSteps01_000.bvh \
  --cycle-start 8135 --cycle-stop 8229 \
  --display-start 8160 --display-count 120 \
  --output sonic/runs/native-g1-pfnn/pfnn-transfer/terrain-fit.npz
```

Copy the formulas and operation order from the released scripts, validate input hashes, and emit a JSON receipt naming selected patch index/coordinate, fitting error, contact count, and artifact hash.

- [ ] **Step 6: Verify Task 1 GREEN and commit**

Run the focused suite twice, `py_compile`, and `git diff --check`. Commit only the extraction module and focused test.

### Task 2: Compare the unflattened G1 retarget with the terrain

**Files:**
- Modify: `sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py`
- Create: `sonic/python/mm_sonic/compare_pfnn_terrain_g1.py`
- Modify: `tests/python/test_retarget_pfnn_bvh_g1.py`
- Modify: `tests/python/test_pfnn_terrain_transfer.py`

**Interfaces:**
- Consumes: `PFNNTerrainFit`, validated G1 motion artifact, pinned GMR G1 model.
- Produces: `SoleGapReport`, `sole_probes`, `compare_motion_to_terrain`, rejected/accepted JSON report.

- [ ] **Step 1: Write RED tests for source grounding and sole classification**

```python
def test_source_grounding_does_not_apply_flat_floor_shift():
    motion, offset = postprocess_motion(fake_motion(), grounding="source")
    np.testing.assert_array_equal(motion.root_pos, fake_motion().root_pos)
    self.assertEqual(offset, 0.0)

def test_stance_probe_classification_uses_continuous_fit():
    report = compare_probe_gaps(np.array([-0.011, 0.0, 0.021]), stance=True)
    self.assertEqual(report.labels, ("penetrating", "contact", "floating"))
```

- [ ] **Step 2: Run targeted tests and verify RED**

Expected: missing `grounding="source"` behavior and comparison API.

- [ ] **Step 3: Implement source grounding and comparison**

Add `--grounding flat|source` with existing `flat` behavior as default. Source mode skips `retargeting_project.postprocess`, records zero grounding offset, and otherwise leaves the approved GMR output unchanged.

The comparison module uses MuJoCo FK on `left_ankle_roll_link` and `right_ankle_roll_link`, evaluating heel `[-0.066, 0, -0.034]` and toe `[0.12, 0, -0.034]` against `terrain_height_g1`. It validates the timeline, source interval, hashes, joint limits, maximum joint step, and stance support.

- [ ] **Step 4: Run focused suites and commit**

Run both focused modules, `py_compile`, and `git diff --check`. Commit the source-grounding and comparison changes separately from Task 1.

### Task 3: Render the exact terrain and run the real gate

**Files:**
- Modify: `sonic/python/mm_sonic/view_g1_retarget.py`
- Modify: `tests/python/test_pfnn_terrain_transfer.py`

**Interfaces:**
- Consumes: validated terrain artifact and G1 motion.
- Produces: `--terrain pfnn-fit --terrain-artifact PATH`, continuous mesh, four probe markers, live numeric status.

- [ ] **Step 1: Write RED tests for mesh construction and CLI binding**

```python
def test_mesh_vertices_are_exact_g1_coordinate_samples():
    vertices, faces = terrain_mesh(fit, x_samples, y_samples)
    np.testing.assert_allclose(vertices[:, 2], terrain_height_g1(fit, vertices[:, :2]))
    self.assertEqual(faces.shape, (2 * (rows - 1) * (cols - 1), 3))

def test_pfnn_mode_requires_the_exact_terrain_artifact():
    with self.assertRaisesRegex(ValueError, "terrain artifact"):
        validate_viewer_inputs(motion, terrain="pfnn-fit", terrain_artifact=None)
```

- [ ] **Step 2: Implement the exact terrain viewer**

Triangulate a dense grid covering the displayed root and sole probes, write a temporary MuJoCo mesh asset, and add four colored spherical marker bodies. Update marker colors from the same signed gaps used by the report. Print frame, foot, stance, and heel/toe gap every second and while paused.

- [ ] **Step 3: Run automated verification**

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_retarget_pfnn_bvh_g1 \
  tests.python.test_pfnn_terrain_transfer
python -m py_compile \
  sonic/python/mm_sonic/pfnn_terrain_fit.py \
  sonic/python/mm_sonic/compare_pfnn_terrain_g1.py \
  sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py \
  sonic/python/mm_sonic/view_g1_retarget.py
git diff --check
```

- [ ] **Step 4: Generate the released PFNN patch database once**

Run the unmodified released `generate_patches.py` in the PFNN root using the Miniconda interpreter. Record its SHA-256 and candidate count. Do not run `generate_database.py`.

- [ ] **Step 5: Extract twice and compare receipts**

Run Task 1 twice in fresh processes to distinct outputs and require identical selected patch identity, fit parameters, query grid, and semantic receipt fields.

- [ ] **Step 6: Retarget and compare**

Create a new 120-frame artifact with `--grounding source`, then run the comparison CLI. Preserve the report whether it accepts or rejects; do not add contact correction in this task.

- [ ] **Step 7: Launch the interactive viewer**

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m mm_sonic.view_g1_retarget \
  --motion sonic/runs/native-g1-pfnn/pfnn-transfer/motion-source-grounded.npz \
  --gmr-root /home/ubuntu/.cache/native-g1-pfnn/GMR \
  --terrain pfnn-fit \
  --terrain-artifact sonic/runs/native-g1-pfnn/pfnn-transfer/terrain-fit.npz
```

Keep the viewer open for user inspection. Report exact contact metrics and ask for visual approval. If the unmodified retarget fails, stop and use its first-failure evidence to design the smallest offline terrain-contact correction; do not hide the mismatch.

- [ ] **Step 8: Commit viewer changes**

Commit only after focused tests and real artifact/viewer inputs are verified. GRAIL remains out of scope until the user approves this gate.
