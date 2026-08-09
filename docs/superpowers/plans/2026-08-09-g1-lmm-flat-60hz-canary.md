# G1 LMM Flat 60 Hz Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one visibly correct, interactive 60 Hz flat-walking G1 canary using Orange Duck's complete learned-motion-matching architecture and an offline-retargeted G1 clip.

**Architecture:** Build one immutable 60 Hz G1 data bundle shared by ordinary motion matching and LMM. Train Orange Duck-compatible compressor/decompressor, stepper, and projector networks on range-safe windows, load them through the existing C++ network ABI, and expose an explicit ordinary/LMM viewer switch. This plan intentionally stops at flat walking; slopes, stairs, and corpus expansion remain blocked until this vertical slice passes.

**Tech Stack:** Python 3, NumPy, PyTorch/CUDA, MuJoCo FK, C++17, Orange Duck `database.h`/`lmm.h`/`nnet.h`, unittest/C++ test binaries.

## Global Constraints

- G1 data, ordinary MM, LMM, controller, terrain query, and evaluation all run at exactly 60 Hz.
- Use the original highest-rate source when available; never upsample a convenience 25 Hz derivative when its original exists.
- The flat canary is walking-only and contains no IK, damping, pose smoothing, joint clamping, or display-only correction.
- Preserve the canonical 31-bone hierarchy, 31 matching features, 32 latent values, two contacts, and Orange Duck binary network ABI.
- Training windows remain inside one validated `[range_start, range_stop)` and never cross a rejected joint discontinuity.
- All projector math is in normalized feature space.
- Every feature/network/data artifact is fail-closed and receipt-bound before runtime state construction.
- Stop at the first failed mechanical or visual gate; do not conceal a failure with post-processing.

---

### Task 1: Build the one-clip 60 Hz G1 data and feature bundle

**Files:**
- Modify: `resources/g1_terrain_builder/resample.py`
- Modify: `resources/g1_terrain_builder/schema.py`
- Modify: `resources/g1_terrain_builder/kinematics.py`
- Modify: `resources/g1_terrain_builder/sources.py`
- Create: `resources/g1_terrain_builder/features.py`
- Modify: `resources/g1_terrain_builder/artifacts.py`
- Modify: `resources/build_g1_terrain_database.py`
- Modify: `resources/validate_g1_terrain_database.py`
- Test: `tests/python/test_resample.py`
- Test: `tests/python/test_database_builder.py`
- Test: `tests/python/test_build_cli.py`
- Test: `tests/python/test_artifacts.py`

**Interfaces:**
- Produces: `resample_map(frames: int, source_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning int32 left/right indices and float32 alpha.
- Produces: `build_matching_features(artifacts: ArtifactSet, fps: float, horizons: tuple[int, int, int]) -> FeatureSet` with normalized `[T,31]` features plus offset/scale.
- Produces: receipt-checked loader for `sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.npz`.
- Produces: CLI `resources/build_g1_terrain_database.py --output-fps 60 --flat-only --retarget-npz PATH --retarget-receipt PATH --output PATH`.
- Produces: immutable `database.bin`, `features.bin`, and manifest with exact rate, horizons, source interpolation map, continuity-safe ranges, skeleton, dimensions, and hashes.

- [ ] **Step 1: Write the failing 60 Hz resampling tests**

```python
def test_resample_map_50_to_60_has_exact_brackets_and_alpha():
    left, right, alpha = resample_map(6, 50.0, 60.0)
    np.testing.assert_array_equal(left[:4], [0, 0, 1, 2])
    np.testing.assert_array_equal(right[:4], [0, 1, 2, 3])
    np.testing.assert_allclose(alpha[:4], [0.0, 5/6, 2/3, 0.5])
    assert left.dtype == right.dtype == np.int32
    assert alpha.dtype == np.float32

def test_flat_cli_writes_60_hz_31d_feature_receipt():
    manifest = build_flat_fixture(output_fps=60.0)
    assert manifest["output_fps"] == 60.0
    assert manifest["trajectory_horizons"] == [20, 40, 60]
    assert manifest["feature_dimensions"] == 31
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_resample tests.python.test_database_builder tests.python.test_build_cli tests.python.test_artifacts`

Expected: fail because `resample_map`, `features.bin`, `--output-fps`, or `--flat-only` is absent.

- [ ] **Step 3: Implement exact interpolation provenance and the shared feature exporter**

```python
def resample_map(frames, source_fps, target_fps):
    count = output_frame_count(frames, source_fps, target_fps)
    source_u = np.arange(count, dtype=np.float64) * source_fps / target_fps
    left = np.floor(source_u).astype(np.int64)
    left = np.clip(left, 0, frames - 1)
    right = np.minimum(left + 1, frames - 1)
    alpha = (source_u - left).astype(np.float32)
    exact = (alpha == 0.0) | (left == right)
    right[exact] = left[exact]
    alpha[exact] = 0.0
    return left.astype(np.int32), right.astype(np.int32), alpha
```

Feature generation must reproduce the C++ ordering exactly: feet positions, feet velocities, hip velocity, root positions at 20/40/60, root facings at 20/40/60, then four zero flat-terrain deltas. Write normalized float32 rows and float32 offset/scale in the existing `features.bin` layout.
At 60 Hz, bind the time-derived 31/61-frame root filters, 7-frame contact
median, and 121-row forward terrain path. The real Takara 50 Hz source must
produce exactly 41,835 output rows; an existing PFNN 120 Hz fixture must map
every output row to the exact even source index.

Before combining the real retarget, split every source edge whose maximum
native/local joint step exceeds `0.25 rad`, discard fragments shorter than 61
frames, and bind the independently recomputable continuity receipt. For the
approved full retarget this rejects 31 edges and publishes exactly 3,853 rows
in 12 ranges; the longest range has 1,177 frames.

- [ ] **Step 4: Run GREEN and build the real flat bundle**

Run the RED command again; expect all tests pass.

Run:

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py \
  --output-fps 60 --flat-only \
  --retarget-npz sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.npz \
  --retarget-receipt sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.receipt.json \
  --output sonic/runs/g1-lmm-flat-60hz/data
```

Expected: accepted manifest, 3,853 frames in 12 continuity-safe flat ranges,
exact retained 120-to-60 even source indices, maximum admitted joint step
`<=0.25 rad`, 31 bones, 31 features, horizons 20/40/60, and all artifact hashes
present.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_terrain_builder resources/build_g1_terrain_database.py \
  resources/validate_g1_terrain_database.py tests/python
git commit -m "feat: build 60 Hz G1 LMM flat data"
```

### Task 2: Migrate the G1 ordinary matcher and query path to 60 Hz

**Files:**
- Modify: `database.h`
- Modify: `controller.cpp`
- Modify: `scene_runtime.h`
- Modify: `route_runtime.h`
- Modify: `g1_kinematic_contract.h`
- Modify: `g1_footprint_runtime.h`
- Modify: `g1_ik.h`
- Modify: `g1_ik_runtime.h`
- Modify: `g1_clearance.cpp`
- Modify: `motion_match_log.h`
- Modify: `sonic/cpp/g1_runtime.h`
- Modify: `resources/check_g1_runtime_log.py`
- Test: `tests/cpp/test_terrain_database.cpp`
- Test: `tests/cpp/test_g1_runtime.cpp`
- Test: `tests/cpp/test_support_matching.cpp`

**Interfaces:**
- Consumes: Task 1 manifest rate and `[20,40,60]` horizons.
- Produces: `database_trajectory_horizons(int out[3], float fps)` and a 60 Hz G1 controller tick.
- Produces: per-frame raw and normalized query logging shared by ordinary MM and LMM.

- [ ] **Step 1: Write failing horizon/rate/parity tests**

```cpp
int horizons[3];
database_trajectory_horizons(horizons, 60.0f);
check(horizons[0] == 20 && horizons[1] == 40 && horizons[2] == 60,
      "60 Hz horizons");
check(!g1_manifest_rate_compatible(25.0f), "stale G1 rate rejected");
```

Add a synthetic 61-frame clip whose trajectory is known analytically and assert ordinary runtime raw query, independently generated raw feature, and normalized `features.bin` row each differ by at most `1e-6`.

- [ ] **Step 2: Run RED**

Run: `cmake --build build -j2 --target test_terrain_database test_g1_runtime test_support_matching && ctest --test-dir build -R 'terrain_database|g1_runtime|support_matching' --output-on-failure`

Expected: horizon or stale-rate assertions fail.

- [ ] **Step 3: Implement the rate-parameterized G1 path**

```cpp
static inline void database_trajectory_horizons(int out[3], const float fps)
{
    assert(isfinite(fps) && fps > 0.0f);
    out[0] = int(roundf(fps / 3.0f));
    out[1] = int(roundf(2.0f * fps / 3.0f));
    out[2] = int(roundf(fps));
}
```

Use `1.0f/60.0f` for the G1 controller and pass the validated manifest rate to feature generation. Leave shipped 23-bone resources on their legacy path.
Use exact float32 `1/60` (`0x3c888889`) in every G1 rate gate. Preserve the
stock 20-frame Orange Duck search margins at 60 Hz. Convert the 15-second log
gate to 900 frames and reject a 25 Hz G1 scene manifest before controller state
or IK state is allocated.

- [ ] **Step 4: Run GREEN and replay the ordinary flat control**

Run the RED command again; expect all selected tests pass. Launch the ordinary viewer with the Task 1 bundle for 600 scripted frames and require zero holds/resets, finite pose, joint step `<=0.25 rad`, and feature parity `<=1e-6`.

- [ ] **Step 5: Commit**

```bash
git add database.h controller.cpp scene_runtime.h route_runtime.h \
  g1_kinematic_contract.h g1_footprint_runtime.h g1_ik.h g1_ik_runtime.h \
  g1_clearance.cpp motion_match_log.h sonic/cpp/g1_runtime.h \
  resources/check_g1_runtime_log.py tests/cpp
git commit -m "feat: run G1 motion matching at 60 Hz"
```

### Task 3: Add deterministic range-safe CUDA training for the G1 LMM networks

**Files:**
- Create: `resources/g1_lmm/__init__.py`
- Create: `resources/g1_lmm/models.py`
- Create: `resources/g1_lmm/dataset.py`
- Create: `resources/g1_lmm/training.py`
- Create: `resources/train_g1_lmm.py`
- Test: `tests/python/test_g1_lmm_training.py`

**Interfaces:**
- Consumes: Task 1 `database.bin` and normalized `features.bin`.
- Produces: Orange Duck ABI-compatible `latent.bin`, `decompressor.bin`, `stepper.bin`, `projector.bin` and `manifest.json`.
- Produces: CLI `resources/train_g1_lmm.py DATA_DIR OUTPUT_DIR --device cuda:1 --stage all`.

- [ ] **Step 1: Write failing architecture, range, and reload tests**

```python
def test_flat_dimensions_and_range_safe_windows():
    dims = G1LmmDimensions(features=31, latent=32, bones=31, contacts=2)
    assert dims.state == 63
    assert dims.compressor_input == 908
    assert dims.decompressor_output == 458
    windows = range_safe_windows(np.array([0, 100]), np.array([80, 180]), 20)
    assert not any(window[0] < 80 <= window[-1] for window in windows)

def test_tiny_training_exports_bitwise_reloadable_networks(tmp_path):
    receipt = train_tiny_fixture(tmp_path, seed=1234, device="cpu")
    assert receipt["finite"]
    assert evaluate_exported_networks(tmp_path) == receipt["reload_metrics"]
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=.:resources sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_g1_lmm_training`

Expected: import failure for `resources.g1_lmm`.

- [ ] **Step 3: Implement the exact Orange Duck architectures on CUDA**

```python
class Decompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(63, 512)
        self.linear1 = nn.Linear(512, 458)
    def forward(self, x):
        return self.linear1(F.relu(self.linear0(x)))
```

Implement the compressor `908->512->512->512->32`, stepper
`63->512->512->63`, and projector `31->512->512->512->512->63`. Preserve the
legacy normalization and binary matrix order, but use CUDA, deterministic
seeds, finite checks, range-safe samplers, immutable receipts, and configurable
budgets. Projector targets and costs remain normalized.

- [ ] **Step 4: Run GREEN, overfit 64 adjacent flat frames, then train the flat bundle**

Run the RED command again; expect all tests pass.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:resources sonic/.torch-mm-venv/bin/python \
  resources/train_g1_lmm.py sonic/runs/g1-lmm-flat-60hz/data \
  sonic/runs/g1-lmm-flat-60hz/model --device cuda:0 --stage all
```

The command must stop between stages unless the 64-frame overfit and withheld
decompressor articulation gates pass. It then requires finite 2/5/8-second
stepper rollouts and projector reconstruction before publishing the model
manifest.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_lmm resources/train_g1_lmm.py tests/python/test_g1_lmm_training.py
git commit -m "feat: train G1 learned motion matching"
```

### Task 4: Load and drive the G1 LMM flat canary

**Files:**
- Modify: `lmm.h`
- Modify: `controller.cpp`
- Modify: `sonic/cpp/g1_controller_state.h`
- Modify: `sonic/cpp/g1_runtime.h`
- Test: `tests/cpp/test_g1_controller_state.cpp`
- Test: `tests/cpp/test_g1_runtime.cpp`
- Create: `tests/cpp/test_g1_lmm.cpp`

**Interfaces:**
- Consumes: Tasks 1-3 data/model manifest pair.
- Produces: runtime mode `--locomotion-engine ordinary|lmm` and per-frame engine/pose/query diagnostics.
- Produces: transactional projector -> stepper -> provisional decode -> final decode -> single commit tick.

- [ ] **Step 1: Write failing loader, normalized-projector, and transaction tests**

```cpp
check(g1_lmm_dimensions_valid(31, 32, 31, 2), "G1 LMM dimensions");
check(projector_cost_normalized(query_norm, projected_norm) == expected,
      "projector cost uses one unit system");
check(state.commit_count == 1 && state.stepper_count == 1,
      "one accepted LMM tick commits and steps exactly once");
```

Tamper each data/model digest and assert rejection occurs before allocating
network evaluation state. Make a projector return NaN and assert all root,
pose, feature, latent, contact, and terrain state remains bitwise unchanged.

- [ ] **Step 2: Run RED**

Run: `cmake --build build -j2 --target test_g1_lmm test_g1_runtime test_g1_controller_state && ctest --test-dir build -R 'g1_lmm|g1_runtime|g1_controller_state' --output-on-failure`

Expected: missing G1 LMM state/loader or raw-vs-normalized projector failure.

- [ ] **Step 3: Implement explicit G1 LMM mode**

Normalize the query before projector distance, overwrite terrain features after
projection, step exactly once, decode provisionally without mutation, overwrite
candidate-root terrain, decode finally, validate, then commit once. Remove the
hard-coded dormant boolean in favor of the explicit operator mode; never
automatically commit ordinary MM during an LMM acceptance run.

- [ ] **Step 4: Run GREEN and the real 60 Hz interactive gate**

Run the RED command again; expect all tests pass. Launch the viewer, drive with
W/A/S/D for at least 20 seconds, start/stop/turn/reverse, and record a video plus
machine receipt. Require all frames owned by LMM, zero holds/resets/fallbacks,
joint step `<=0.25 rad/frame`, root translation velocity
`<=1.5 m/s` (`25 mm/frame`), root rotation velocity `<=8.75 rad/s`
(`0.145834 rad/frame`), and coherent full-body articulation on visual review.

- [ ] **Step 5: Commit**

```bash
git add lmm.h controller.cpp sonic/cpp/g1_controller_state.h \
  sonic/cpp/g1_runtime.h tests/cpp
git commit -m "feat: drive the 60 Hz G1 LMM canary"
```

### Task 5: Final evidence and decision

**Files:**
- Create: `.superpowers/sdd/g1-lmm-flat-60hz-canary-report.md` (ignored artifact)

**Interfaces:**
- Consumes: exact source commits and immutable Task 1/3 bundles.
- Produces: reproducible commands, hashes, numerical metrics, video path, and a pass/fail statement.

- [ ] **Step 1: Run the dependency-split Python and C++ suites**

Run all Task 1/3 Python tests in their owning environments and all Task 2/4 C++ tests. Record exact counts and output.

- [ ] **Step 2: Rebuild/reload without mutation**

Snapshot every artifact hash/size/mtime, rerun builder resume and safe model reload, and assert the snapshot is byte-identical.

- [ ] **Step 3: Personally drive and visually inspect**

Use the same W/A/S/D interaction the user will use. Inspect the entire body,
not only the root. If limbs twist, contacts drag, the model holds, or any gate
fails, continue debugging rather than publishing success.

- [ ] **Step 4: Publish the evidence report**

Record the source HEAD, data/model manifest hashes, frame rate, clip identity,
training receipts, test counts, mechanical maxima, engine-owner counters,
viewer command, and video path. State PASS only when the executable result and
visual inspection both pass.
