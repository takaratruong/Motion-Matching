# G1 PFNN Playable Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a joystick-controlled native-G1 PFNN trained on a small offline-retargeted subset of the released PFNN walking and terrain data.

**Architecture:** Reuse the existing `288 -> 268` G1 feature seam, four-bank cubic PFNN, classic one-step loss, closed-loop runtime, and MuJoCo viewer. Add an isolated released-PFNN adapter that selects bounded source intervals, consumes offline GMR retarget artifacts, transfers original phase/contact/terrain semantics, and writes a compact train/validation corpus. Train once on GPU, gate it in a scripted exact-terrain route, then personally drive the same viewer.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch/CUDA, MuJoCo, GMR/retargeting_project, `unittest`, released PFNN BVH/phase/gait/footstep/heightmap data.

## Global Constraints

- Keep the original PFNN architecture: four phase control points, cubic interpolation, two 512-unit ELU hidden layers, and 30 Hz one-step recurrence.
- Retarget source motion at 120 Hz and only then sample G1 motion at 30 Hz.
- Use morphology scale `0.875` and fixed world-Z terrain placement `0.05224985936713168 m`.
- Train only idle/walk, turning, ascent, and descent in this milestone.
- Do not use GRAIL, running, jumping, IK, foot locking, output clamping, pose repair, or teacher-forced runtime correction.
- Use only non-mirrored released PFNN BVHs; mirror training features exactly once afterward.
- Fail closed on missing annotations, hash mismatch, invalid phase/contact, nonfinite motion, joint-limit violation, terrain-query failure, or missing coverage.
- A one-step loss is not an acceptance gate; the model must pass the scripted closed loop and a live interactive drive.

---

### Task 1: Released PFNN source inventory and deterministic slice selection

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/source_pfnn_released.py`
- Create: `tests/python/test_terrain_pfnn_source_pfnn_released.py`

**Interfaces:**
- Consumes: released animation directory containing matching `.bvh`, `.phase`, `.gait`, and `_footsteps.txt` files.
- Produces: `ReleasedPFNNRecord`, `PFNNSliceRole`, `discover_released_pfnn_records(root)`, `select_vertical_slice(records, metrics)`, and `vertical_slice_receipt(selection)`.

- [ ] **Step 1: Write failing discovery and selection tests**

```python
def test_selects_train_and_validation_coverage_without_mirrored_sources(tmp_path):
    records = make_released_fixture(tmp_path, include_mirrors=True)
    metrics = {
        "flat_straight": metric(idle_transition=True, turn_left=0, turn_right=0),
        "flat_turn": metric(turn_left=1.2, turn_right=-1.1),
        "steps_a": metric(ascent=9.0, descent=-8.0),
        "steps_b": metric(ascent=12.0, descent=-10.0),
        "steps_validation": metric(ascent=7.0, descent=-6.0),
    }
    selection = select_vertical_slice(records, metrics)
    assert [item.role for item in selection].count("train") == 3
    assert [item.role for item in selection].count("validation") == 1
    assert all("_mirror" not in item.record.stem for item in selection)
    assert required_coverage(selection) == {
        "idle_transition", "straight", "left_turn", "right_turn", "ascent", "descent"
    }
```

Write these additional exact tests in the same class:

```python
def test_discovery_rejects_missing_or_duplicate_source_members(self):
    with self.assertRaisesRegex(ValueError, "missing PFNN sidecar"):
        discover_released_pfnn_records(missing_phase_root)
    with self.assertRaisesRegex(ValueError, "duplicate PFNN stem"):
        discover_released_pfnn_records(duplicate_stem_root)

def test_selection_rejects_missing_required_coverage(self):
    with self.assertRaisesRegex(ValueError, "vertical slice lacks right_turn"):
        select_vertical_slice(records, metrics_without_right_turn)

def test_receipt_binds_hashes_and_is_order_independent(self):
    self.assertEqual(
        vertical_slice_receipt(select_vertical_slice(records, metrics)),
        vertical_slice_receipt(select_vertical_slice(tuple(reversed(records)), metrics)),
    )
    changed = replace(records[0], bvh_sha256="f" * 64)
    self.assertNotEqual(
        vertical_slice_receipt(select_vertical_slice((changed, *records[1:]), metrics)),
        vertical_slice_receipt(select_vertical_slice(records, metrics)),
    )
```

- [ ] **Step 2: Run the source tests and record RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_source_pfnn_released
```

Expected: import error for `mm_sonic.terrain_pfnn.source_pfnn_released`.

- [ ] **Step 3: Implement strict records and selection**

```python
@dataclass(frozen=True)
class ReleasedPFNNRecord:
    stem: str
    bvh_path: Path
    phase_path: Path
    gait_path: Path
    footsteps_path: Path
    bvh_sha256: str
    phase_sha256: str
    gait_sha256: str
    footsteps_sha256: str

@dataclass(frozen=True)
class PFNNSliceRole:
    record: ReleasedPFNNRecord
    role: Literal["train", "validation"]
    start_frame_120hz: int
    stop_frame_120hz: int
    coverage: tuple[str, ...]
```

Discovery must sort by stem, reject `_mirror`, require all four files, hash
streamingly, and reject duplicate canonical stems. Selection must choose four
source intervals: three training intervals covering idle transition, straight,
both turn directions, ascent, and descent, plus one terrain validation interval.
Each
interval must include 120 frames of context on both sides for the one-second
trajectory window. Tie-break on `(record.stem, start_frame, stop_frame)`.

- [ ] **Step 4: Run source tests GREEN**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/source_pfnn_released.py \
  tests/python/test_terrain_pfnn_source_pfnn_released.py
git commit -m "feat: select released PFNN controller slice"
```

---

### Task 2: Convert approved 120 Hz G1 retargets into native PFNN sources

**Files:**
- Create: `sonic/python/mm_sonic/build_g1_pfnn_vertical_slice.py`
- Modify: `sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/sources.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/phase.py`
- Create: `tests/python/test_build_g1_pfnn_vertical_slice.py`
- Modify: `tests/python/test_terrain_pfnn_sources.py`
- Modify: `tests/python/test_terrain_pfnn_phase.py`

**Interfaces:**
- Consumes: `PFNNSliceRole`, the existing GMR driver, G1 MJCF/FK, source `.phase` and source contacts.
- Produces: `retarget_vertical_slice_item(...)`, `load_pfnn_retarget_source(...) -> PFNNSourceClip`, and `released_pfnn_phase_track(...) -> ContactPhaseTrack`.

- [ ] **Step 1: Write RED tests for temporal and kinematic conversion**

```python
def test_loads_120hz_retarget_as_exact_30hz_source():
    source = load_pfnn_retarget_source(
        motion_path=fixture_motion_120hz,
        clip_id="pfnn__flat_turn__00480_01920",
        terrain_id="flat",
        fk=fake_fk,
    )
    assert source.fps == 30.0
    np.testing.assert_array_equal(source.root_position_world, root_120hz[::4])
    np.testing.assert_array_equal(source.joint_position, joints_120hz[::4])
    assert source.body_position_world.shape == (len(root_120hz[::4]), 30, 3)
```

```python
def test_released_phase_track_preserves_phase_contacts_and_advance():
    track = released_pfnn_phase_track(phase_120hz, contacts_120hz)
    np.testing.assert_allclose(track.phase, np.remainder(phase_120hz[::4], 1.0) * (2*np.pi))
    assert track.contact.shape == (len(phase_120hz[::4]), 4)
    assert np.all(track.valid)
    assert np.all(track.phase_advance[:-1] >= 0.0)
```

Write one table-driven rejection test:

```python
def test_retarget_source_and_phase_reject_contract_violations(self):
    failures = (
        (motion_at_60hz, "exactly 120 Hz"),
        (motion_with_bad_quaternion, "normalized XYZW"),
        (motion_with_wrong_joint_order, "29-joint order"),
        (motion_outside_joint_limits, "joint-limit violation"),
        (motion_with_wrong_interval_receipt, "selected interval"),
    )
    for motion, message in failures:
        with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
            load_pfnn_retarget_source(motion, fk=fake_fk)
    for phase, contact, message in (
        (backward_phase, valid_contacts, "phase discontinuity"),
        (valid_phase, nonbinary_contacts, "binary contacts"),
    ):
        with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
            released_pfnn_phase_track(phase, contact)
```

- [ ] **Step 2: Run Task 2 RED**

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m unittest -v tests.python.test_build_g1_pfnn_vertical_slice \
  tests.python.test_terrain_pfnn_sources tests.python.test_terrain_pfnn_phase
```

Expected: missing conversion APIs.

- [ ] **Step 3: Export a bounded retarget entry point**

Add to `retarget_pfnn_bvh_g1.py`:

```python
def retarget_pfnn_interval(
    *, source: Path, output: Path, start_frame: int, stop_frame: int,
    warmup_frames: int, gmr_root: Path, retarget_project_root: Path,
    model_path: Path,
) -> Path:
    """Run the existing pinned GMR path and atomically save one 120 Hz interval."""
```

This function must reuse the current preparation, `scale_pfnn_frames`, GMR
solve, source grounding, joint-limit validation, and receipt writer. It must
not duplicate retarget math or call flat postprocessing.

- [ ] **Step 4: Implement exact 120-to-30 conversion and phase track**

`load_pfnn_retarget_source` samples indices `0,4,8,...`, runs canonical G1 FK,
computes body/root/joint velocities at 30 Hz with existing finite-difference
helpers, and binds the motion receipt SHA. `released_pfnn_phase_track` converts
normalized source phase to radians, unwraps successive samples to compute
nonnegative phase advance, copies four authored heel/toe contacts, and marks
only finite consecutive frames valid.

- [ ] **Step 5: Implement the resumable retarget orchestrator**

The CLI writes `vertical-slice-retarget/v1` JSON before work begins and updates
each item atomically after successful retarget validation. `--resume` must
rehash completed outputs and refuse changed inputs. It runs exactly the four
sealed selection items and never discovers additional clips during resume.

- [ ] **Step 6: Run Task 2 GREEN and compile**

Run the Step 2 command, followed by:

```bash
/home/ubuntu/.cache/native-g1-pfnn/venv/bin/python -m py_compile \
  sonic/python/mm_sonic/build_g1_pfnn_vertical_slice.py \
  sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py \
  sonic/python/mm_sonic/terrain_pfnn/sources.py \
  sonic/python/mm_sonic/terrain_pfnn/phase.py
```

- [ ] **Step 7: Commit Task 2**

```bash
git add sonic/python/mm_sonic/build_g1_pfnn_vertical_slice.py \
  sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py \
  sonic/python/mm_sonic/terrain_pfnn/sources.py \
  sonic/python/mm_sonic/terrain_pfnn/phase.py \
  tests/python/test_build_g1_pfnn_vertical_slice.py \
  tests/python/test_terrain_pfnn_sources.py tests/python/test_terrain_pfnn_phase.py
git commit -m "feat: retarget released PFNN controller slice"
```

---

### Task 3: Transfer rank-zero PFNN terrain cycles and build the compact corpus

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/pfnn_surface.py`
- Create: `sonic/python/mm_sonic/build_g1_pfnn_vertical_dataset.py`
- Modify: `sonic/python/mm_sonic/pfnn_terrain_fit.py`
- Create: `tests/python/test_terrain_pfnn_pfnn_surface.py`
- Create: `tests/python/test_build_g1_pfnn_vertical_dataset.py`

**Interfaces:**
- Consumes: retarget manifest, selected source sidecars, `fit_terrain_cycle`, `build_clip_windows_with_audit`.
- Produces: `PlacedPFNNSurface`, `load_placed_pfnn_surface(path)`, and a `g1-pfnn-vertical-dataset/v1` manifest with train/validation arrays and normalization.

- [ ] **Step 1: Write RED tests for placed surface and cycle-local windows**

```python
def test_placed_surface_uses_frozen_morphology_transform():
    surface = PlacedPFNNSurface(fit=fit, scale=0.875, z_offset=0.05224985936713168)
    xy_g1 = xy_source_m * 0.875
    np.testing.assert_allclose(
        surface.height_at(xy_g1),
        terrain_height_g1(fit, xy_g1, scale=0.875, z_offset=0.05224985936713168),
        atol=1e-12, rtol=0,
    )
```

```python
def test_dataset_rows_reconstruct_approved_transfer_sample():
    dataset = build_vertical_dataset(fixture_manifest, fixture_sources)
    row = dataset.row_for("WalkingUpSteps01_000", center_frame=8160 // 4)
    np.testing.assert_allclose(denormalize(row.x), expected_x, atol=3e-5, rtol=0)
    np.testing.assert_allclose(denormalize(row.y), expected_y, atol=3e-5, rtol=0)
```

Write exact rejection and determinism tests:

```python
def test_cycle_windows_stay_inside_their_fitted_surface(self):
    rows = build_cycle_windows(source, phase_track, surface, cycle=(100, 180))
    self.assertTrue(all(100 <= row.center_frame <= 180 for row in rows))

def test_dataset_is_deterministic_and_rejects_invalid_provenance(self):
    first = build_vertical_dataset(manifest, sources)
    second = build_vertical_dataset(manifest, tuple(reversed(sources)))
    self.assertEqual(first.dataset_sha256, second.dataset_sha256)
    for bad, message in (
        (duplicate_center_sources, "duplicate PFNN window"),
        (overlapping_split_sources, "train/validation source overlap"),
        (changed_fit_sources, "terrain fit digest mismatch"),
        (missing_query_sources, "terrain query unavailable"),
    ):
        with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
            build_vertical_dataset(manifest, bad)

def test_normalization_uses_training_rows_only(self):
    baseline = build_vertical_dataset(manifest, sources)
    changed_validation = replace_validation_values(sources, value=1e6)
    rebuilt = build_vertical_dataset(manifest, changed_validation)
    np.testing.assert_array_equal(baseline.x_mean, rebuilt.x_mean)
    np.testing.assert_array_equal(baseline.y_std, rebuilt.y_std)
```

- [ ] **Step 2: Run Task 3 RED**

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_pfnn_surface \
  tests.python.test_build_g1_pfnn_vertical_dataset
```

- [ ] **Step 3: Implement immutable placed PFNN surfaces**

```python
@dataclass(frozen=True)
class PlacedPFNNSurface:
    fit: PFNNTerrainFit
    scale: float = 0.875
    z_offset: float = 0.05224985936713168

    def height_at(self, xy: object) -> np.ndarray:
        return terrain_height_g1(
            self.fit, np.asarray(xy), scale=self.scale, z_offset=self.z_offset
        )

    def gradient_at(self, xy: object, epsilon: float = 1e-4) -> np.ndarray:
        # centered finite differences over the same height_at function
```

Serialization must embed or bind the numeric `PFNNTerrainFit` artifact and
reject a scale/offset other than the frozen values for this dataset schema.

- [ ] **Step 4: Build cycle-local examples using original phase and contacts**

For each valid footstep pair, fit the rank-zero surface, slice the matching 30
Hz G1 source and phase track with one-second context, and call
`build_clip_windows_with_audit`. Treat `(source stem, footstep cycle)` as one
terrain example. Flat examples use a zero-gradient surface at the frozen G1
support origin. Mirror train windows exactly once with `mirror_window`.

- [ ] **Step 5: Write and validate the compact dataset**

The manifest stores separate train and validation NPZ paths, hashes, row counts,
source identities, terrain coverage counts, rejection counts, normalization,
feature layouts, joint order, and the retarget/source/terrain receipt digests.
Compute `x_mean/x_std/y_mean/y_std` from unmirrored plus mirrored training rows
only. Force contact normalization to mean zero/std one. The loader rejects
unknown fields, object dtypes, nonfinite arrays, overlap, or digest mismatch.

- [ ] **Step 6: Run Task 3 GREEN and focused regression tests**

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_pfnn_surface \
  tests.python.test_build_g1_pfnn_vertical_dataset \
  tests.python.test_pfnn_terrain_transfer tests.python.test_terrain_pfnn_features
```

- [ ] **Step 7: Commit Task 3**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/pfnn_surface.py \
  sonic/python/mm_sonic/build_g1_pfnn_vertical_dataset.py \
  sonic/python/mm_sonic/pfnn_terrain_fit.py \
  tests/python/test_terrain_pfnn_pfnn_surface.py \
  tests/python/test_build_g1_pfnn_vertical_dataset.py
git commit -m "feat: build retargeted PFNN terrain corpus"
```

---

### Task 4: Train the classic PFNN on the released-PFNN vertical corpus

**Files:**
- Modify: `sonic/python/mm_sonic/train_classic_g1_pfnn.py`
- Modify: `tests/python/test_train_classic_g1_pfnn.py`

**Interfaces:**
- Consumes: `g1-pfnn-vertical-dataset/v1` manifest and arrays.
- Produces: `classic-g1-pfnn/v2` checkpoint bound to the vertical-slice receipts.

- [ ] **Step 1: Write RED tests for PFNN-only training and checkpoint binding**

```python
def test_vertical_training_uses_four_phase_banks_and_original_style_loss(tmp_path):
    result = train(vertical_arguments(dataset=fixture_dataset, epochs=2, output=tmp_path/"run"))
    checkpoint = load_classic_checkpoint(result)
    assert checkpoint.model_state["W0"].shape[0] == 4
    assert checkpoint.model_state["W0"].shape[1] == 512
    assert checkpoint.source_kind == "released_pfnn"
    assert checkpoint.vertical_slice_receipt_sha256 == fixture_receipt_sha
```

Write the following receipt and optimizer-boundary test:

```python
def test_checkpoint_tamper_and_validation_optimizer_isolation(self):
    trained = train_with_spy(vertical_fixture)
    self.assertTrue(set(trained.optimizer_clip_ids).isdisjoint(validation_clip_ids))
    np.testing.assert_array_equal(trained.before_reload_output, trained.after_reload_output)
    self.assertLess(trained.final_fixture_loss, 0.1 * trained.initial_fixture_loss)
    for field in ("dataset_digest", "vertical_slice_receipt", "terrain_receipt_set"):
        with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
            load_classic_checkpoint(tamper_checkpoint(trained.path, field))
```

- [ ] **Step 2: Run Task 4 RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_train_classic_g1_pfnn
```

- [ ] **Step 3: Add the vertical dataset loader without changing model math**

Add `--train-source released-pfnn`. Materialize only the compact train and
validation arrays. Use existing normalization, `PhaseFunctionedNetwork`,
`classic_pfnn_loss`, Adam `1e-4`, batch size 32, seed 23456, and validation
selection. Do not invoke the mixed GRAIL/LAFAN sampler for this mode.

- [ ] **Step 4: Bind the source receipts in checkpoint v2**

Add exact `source_kind`, `vertical_slice_receipt_sha256`, and
`terrain_receipt_set_sha256` fields. The safe loader validates all fields before
model construction and continues using `weights_only=True`.

- [ ] **Step 5: Run the deterministic CPU overfit gate**

Train a fixed 512-row fixture for 2,000 updates. Require finite loss, final loss
below 10% of initial, exact safe reload, and no joint-limit-invalid target. Stop
and diagnose if this gate fails; do not tune the runtime.

- [ ] **Step 6: Run Task 4 GREEN and model/runtime regression tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_train_classic_g1_pfnn \
  tests.python.test_terrain_pfnn_model tests.python.test_terrain_pfnn_runtime
```

- [ ] **Step 7: Commit Task 4**

```bash
git add sonic/python/mm_sonic/train_classic_g1_pfnn.py \
  tests/python/test_train_classic_g1_pfnn.py
git commit -m "feat: train classic G1 PFNN on released data"
```

---

### Task 5: Render the exact PFNN terrain and gate the closed-loop controller

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn_viewer.py`
- Create: `sonic/python/mm_sonic/evaluate_g1_pfnn_vertical_slice.py`
- Modify: `tests/python/test_terrain_pfnn_viewer.py`
- Create: `tests/python/test_evaluate_g1_pfnn_vertical_slice.py`

**Interfaces:**
- Consumes: checkpoint v2, compact dataset manifest, and an exact `PlacedPFNNSurface`.
- Produces: scripted route receipt and the final interactive viewer.

- [ ] **Step 1: Write viewer-terrain and scripted-route RED tests**

```python
def test_viewer_queries_and_renders_the_same_pfnn_surface():
    callback, vertices = build_pfnn_viewer_surface(fit_artifact)
    np.testing.assert_allclose(
        vertices[:, 2], callback.collision_heights_at(vertices[:, :2]), atol=1e-6, rtol=0
    )
```

```python
def test_script_contains_start_turns_grades_and_stop():
    script = vertical_slice_command_script()
    assert set(segment.name for segment in script) == {
        "start", "straight", "left_turn", "right_turn", "ascent", "descent", "stop"
    }
```

Write exact route failure tests:

```python
def test_evaluator_rejects_first_invalid_runtime_tick(self):
    failures = (
        (runtime_leaves_surface, "unsupported terrain"),
        (runtime_holds, "hold_reason"),
        (runtime_nonfinite, "nonfinite"),
        (runtime_breaches_envelope, "motion envelope"),
    )
    for runtime, message in failures:
        with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
            evaluate_script(runtime, valid_surface, vertical_slice_command_script())

def test_viewer_rejects_provenance_or_surface_mismatch(self):
    with self.assertRaisesRegex(ValueError, "source receipt"):
        load_vertical_runtime(checkpoint_without_source_receipt, dataset, surface)
    with self.assertRaisesRegex(ValueError, "renderer/query surface mismatch"):
        build_pfnn_viewer_surface(tampered_fit_artifact)
```

- [ ] **Step 2: Run Task 5 RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_viewer \
  tests.python.test_evaluate_g1_pfnn_vertical_slice
```

- [ ] **Step 3: Replace the synthetic hill callback with placed PFNN terrain**

Add viewer arguments `--terrain-fit`, `--terrain-scale 0.875`, and
`--terrain-z-offset 0.05224985936713168`. Build the MuJoCo mesh from the same
`PlacedPFNNSurface.height_at` callable used by runtime and collision queries.
Show the 12-by-three terrain probes and four contact markers in the viewer.

- [ ] **Step 4: Implement the deterministic scripted evaluator**

Run start, straight, left turn, right turn, ascent, descent, and stop at 30 Hz.
Every tick records root pose, command, phase, grade, contacts, and diagnostics.
Reject immediately on `hold_reason`, nonfinite state, unsupported terrain,
joint-limit violation, or failure to realize both signed grade directions.
Write JSON only after all steps pass, bound to checkpoint/dataset/terrain hashes.

- [ ] **Step 5: Run Task 5 GREEN and the complete focused suite**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_source_pfnn_released \
  tests.python.test_build_g1_pfnn_vertical_slice \
  tests.python.test_terrain_pfnn_pfnn_surface \
  tests.python.test_build_g1_pfnn_vertical_dataset \
  tests.python.test_train_classic_g1_pfnn \
  tests.python.test_terrain_pfnn_viewer \
  tests.python.test_evaluate_g1_pfnn_vertical_slice \
  tests.python.test_terrain_pfnn_model tests.python.test_terrain_pfnn_runtime
```

- [ ] **Step 6: Commit Task 5**

```bash
git add sonic/python/mm_sonic/terrain_pfnn_viewer.py \
  sonic/python/mm_sonic/evaluate_g1_pfnn_vertical_slice.py \
  tests/python/test_terrain_pfnn_viewer.py \
  tests/python/test_evaluate_g1_pfnn_vertical_slice.py
git commit -m "feat: drive G1 PFNN on released terrain"
```

---

### Task 6: Build real artifacts, train once, and personally drive the viewer

**Files:**
- Generate (ignored): `sonic/runs/native-g1-pfnn/vertical-slice/source-selection.json`
- Generate (ignored): `sonic/runs/native-g1-pfnn/vertical-slice/retarget-manifest.json`
- Generate (ignored): `sonic/runs/native-g1-pfnn/vertical-slice/dataset/manifest.json`
- Generate (ignored): `sonic/runs/native-g1-pfnn/vertical-slice/model/best.pt`
- Generate (ignored): `sonic/runs/native-g1-pfnn/vertical-slice/evaluation.json`

**Interfaces:**
- Consumes: Tasks 1-5 and the pinned released PFNN/GMR inputs.
- Produces: the playable viewer and exact artifact receipts.

- [ ] **Step 1: Audit and seal the real source selection**

Run the source audit CLI against `/home/ubuntu/datasets/pfnn/pfnn/data/animations`.
Inspect the selected filenames, intervals, idle transition, yaw coverage, and
positive/negative grade coverage before authorizing retargeting.

- [ ] **Step 2: Retarget all four bounded intervals at 120 Hz**

Use the pinned GMR and retargeting_project commits already validated by the
approved sample. Resume only from hash-verified completed items. Validate every
artifact against the G1 MJCF and visually inspect one flat/turn, one ascent, and
one descent interval before building data.

- [ ] **Step 3: Fit terrain cycles and build the compact 30 Hz dataset**

Require nonzero train and validation rows for flat, turning, ascent, and descent,
exact train-only normalization, deterministic repeat hashes, and approved-sample
row reconstruction before training.

- [ ] **Step 4: Run one GPU training job**

Use GPU 1, seed 23456, hidden size 512, batch size 32, Adam `1e-4`, and at most
80 epochs with best validation checkpoint selection. If training becomes
nonfinite or the overfit prerequisite failed, stop instead of changing the
controller or thresholds.

- [ ] **Step 5: Run the scripted closed-loop gate**

The evaluator must pass start, straight, both turns, ascent, descent, and stop
without holds or invalid frames. On failure, preserve the candidate and diagnose
the first failing input/output transition before any retraining decision.

- [ ] **Step 6: Open and personally operate the final viewer**

Use W/A/S/D through the same input path the user will use. Traverse the exact
PFNN terrain in both grade directions, turn left and right, stop, restart, and
inspect foot/terrain alignment from side and three-quarter camera views. Capture
a live screenshot and the runtime receipt. If anything is visibly broken,
continue debugging before presenting it.

- [ ] **Step 7: Final verification and handoff**

Run the complete focused suite from Task 5, `py_compile` every changed Python
file, and `git diff --check`. Report exact tests, source/dataset/checkpoint/
terrain/evaluation hashes, scripted route result, and live viewer command.
