# G1 Torch Stair Kinematics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native 50 Hz Torch motion-matching experiment that compares flat, legacy four-height, and dense local-height features on one explicit four-recording GRAIL stair corpus.

**Architecture:** An offline builder converts four pinned 25 Hz GRAIL robot/object records into the existing native Z-up Takara motion-folder contract and publishes matching Z-up height grids transactionally. A generic optional feature-extension seam appends terrain rows to the qualified 27-dimensional Torch matcher without changing flat behavior. A deterministic renderer-free rollout runs all three feature conditions and a read-only viewer renders saved kinematics.

**Tech Stack:** Python 3.10, NumPy, joblib, MuJoCo, USD (`pxr`), PyTorch 2.13/CUDA, unittest, matplotlib.

## Global Constraints

- Work only on `research/g1-torch-terrain-kinematics`, based on qualified commit `dbeaccc7b6b603f515cebd57c9951b95b1c04c31`; never change `research/g1-low-latency-driver`.
- Runtime data is exactly 50 Hz (`dt=0.02`), Z-up, planar X/Y, positive-X forward, wxyz quaternion, 29 IsaacLab-order joints, and 30 IsaacLab-order bodies.
- Input GRAIL robot and object records must declare exactly `fps=25.0`; the converter emits 499 samples for each 250-sample input while preserving timestamps `0.00` through `9.96` seconds.
- Load only the four checked `04d99a9e43` recordings listed in the design; never discover the full stair corpus by glob.
- Flat mode must remain bitwise/numerically equivalent to the qualified 27-feature matcher.
- The legacy terrain block has 4 values at distances `0.25, 0.50, 0.75, 1.00` metres.
- The dense terrain block has 91 values: forward `[-0.15, 1.65]` and lateral `[-0.45, 0.45]`, both spaced by `0.15` metres, flattened forward-major.
- All terrain values are relative to terrain height at the current planar root location; global root height, global XY, and scene identity are not search features.
- Do not run SONIC, MuJoCo physics integration, or train a depth model in this milestone.
- Generated motion/height artifacts live under ignored `build/torch-stair-small/` and are never committed.
- Follow test-driven development: add a focused failing test, run and inspect RED, add the minimum production behavior, then run GREEN before each commit.

---

### Task 1: Pin and convert the four-recording stair corpus

**Files:**
- Create: `resources/g1_torch_stair_builder/__init__.py`
- Create: `resources/g1_torch_stair_builder/corpus.py`
- Create: `resources/g1_torch_stair_builder/conversion.py`
- Create: `tests/python/test_torch_stair_conversion.py`

**Interfaces:**
- Consumes: local GRAIL `robot/<base>.pkl`, the pinned G1 MJCF, and `resources.g1_terrain_builder.kinematics.G1Kinematics`.
- Produces: `STAIR_BASES: tuple[str, str, str, str]`, `load_pinned_sources(root: Path) -> Sequence[PinnedStairSource]`, and `convert_source_to_native_50hz(source: PinnedStairSource, kinematics: G1Kinematics) -> NativeMotionArrays`.

- [ ] **Step 1: Write the failing inventory and conversion tests**

Add tests that build synthetic joblib records in a temporary `robot/` directory
and assert:

```python
class PinnedCorpusTests(unittest.TestCase):
    def test_inventory_is_exactly_four_named_04d99a9e43_records(self):
        self.assertEqual(len(STAIR_BASES), 4)
        self.assertEqual(
            [name.rsplit("__", 1)[-1] for name in STAIR_BASES[:3]],
            ["0000", "0001", "0002"],
        )
        self.assertTrue(STAIR_BASES[3].endswith("_updown__0000"))

    def test_loader_rejects_missing_extra_identity_and_non_25hz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root, fps=25.0)
            (root / "robot" / f"{STAIR_BASES[0]}.pkl").unlink()
            with self.assertRaisesRegex(
                ValueError, rf"{re.escape(STAIR_BASES[0])}.*missing"
            ):
                load_pinned_sources(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(
                root, fps=25.0, wrong_outer_key_for=STAIR_BASES[1]
            )
            with self.assertRaisesRegex(ValueError, "outer record identity"):
                load_pinned_sources(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root, fps=50.0)
            with self.assertRaisesRegex(ValueError, "fps.*25"):
                load_pinned_sources(root)


class NativeConversionTests(unittest.TestCase):
    def test_250_samples_at_25hz_become_499_native_50hz_samples(self):
        converted = convert_source_to_native_50hz(source, fake_kinematics)
        self.assertEqual(converted.fps, 50)
        self.assertEqual(converted.joint_position.shape, (499, 29))
        self.assertEqual(converted.body_position_world.shape, (499, 30, 3))
        np.testing.assert_array_equal(
            converted.joint_position[[0, -1]],
            expected_isaac_order_endpoints,
        )
        np.testing.assert_allclose(
            np.linalg.norm(converted.body_quaternion_world_wxyz, axis=-1),
            1.0,
            rtol=0.0,
            atol=1e-5,
        )
```

Use analytic linear root/joint trajectories and a root quaternion that crosses a
sign boundary. The fake kinematics returns named MuJoCo-order body transforms so
the test independently proves the frozen Isaac body permutation, including feet
at indices 18 and 19.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  python3 -B -m unittest -v tests.python.test_torch_stair_conversion
```

Expected: import failure for `resources.g1_torch_stair_builder`.

- [ ] **Step 3: Implement the pinned source contract**

In `corpus.py`, define immutable records:

```python
@dataclass(frozen=True)
class PinnedStairSource:
    base: str
    robot_path: Path
    object_path: Path
    usd_path: Path
    robot_qpos_mujoco: np.ndarray
    object_position_world: np.ndarray
    object_quaternion_world_xyzw: np.ndarray
    object_scale: np.ndarray
    source_fps: float
    source_sha256: dict[str, str]
```

`load_pinned_sources()` must:

- resolve only `STAIR_BASES`;
- require one outer record in both robot and object PKLs;
- require robot fields `dof`, `root_trans_offset`, `root_rot`, and `fps`;
- require object fields `root_pos`, `root_quat`, `scale`, and `fps`;
- require 250 robot/object samples and constant object transforms;
- convert robot root quaternion xyzw to normalized wxyz;
- retain object quaternion as documented xyzw until surface construction; and
- own finite contiguous arrays.

- [ ] **Step 4: Implement native 50 Hz conversion**

In `conversion.py`, define:

```python
@dataclass(frozen=True)
class NativeMotionArrays:
    fps: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    def as_npz_fields(self) -> dict[str, np.ndarray]:
        return {
            "fps": np.array([self.fps], np.int64),
            "joint_pos": self.joint_position,
            "joint_vel": self.joint_velocity,
            "body_pos_w": self.body_position_world,
            "body_quat_w": self.body_quaternion_world_wxyz,
            "body_lin_vel_w": self.body_linear_velocity_world,
            "body_ang_vel_w": self.body_angular_velocity_world,
        }
```

Construct 50 Hz qpos by calling the existing timestamp vector and quaternion
resamplers. Reorder MuJoCo joint positions to IsaacLab order with
`PINNED_TARGET_TO_SOURCE_PERMUTATION`. Run `G1Kinematics.world_from_qpos()` at
the 499 output samples, then reorder named MuJoCo bodies into this exact
IsaacLab body tuple:

```python
ISAAC_BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link",
    "right_ankle_pitch_link", "left_shoulder_roll_link",
    "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
```

Derive joint and body velocities at 50 Hz and validate the output by passing its
NPZ fields through the existing strict `MotionFolder` loader in a temporary
directory.

- [ ] **Step 5: Run GREEN and commit**

Run the Task 1 test plus the existing native loader suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  python3 -B -m unittest -v \
  tests.python.test_torch_stair_conversion \
  tests.python.test_sonic_torch_motion_data
```

Expected: PASS.

Commit:

```bash
git add resources/g1_torch_stair_builder tests/python/test_torch_stair_conversion.py
git commit -m "feat: convert pinned GRAIL stairs to native 50 Hz"
```

---

### Task 2: Build aligned Z-up stair height grids and publish the small dataset

**Files:**
- Create: `resources/g1_torch_stair_builder/surface.py`
- Create: `resources/g1_torch_stair_builder/publish.py`
- Create: `resources/build_g1_torch_stair_slice.py`
- Create: `tests/python/test_torch_stair_surface.py`
- Create: `tests/python/test_torch_stair_publish.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `PinnedStairSource`, `NativeMotionArrays`, one flat Takara `motion.npz`, and the GRAIL object USD/pose records.
- Produces: `ZUpHeightGrid.sample_xy(points_xy)`, `build_source_height_grid(source)`, and `publish_stair_slice(output, inputs) -> dict`.

- [ ] **Step 1: Write failing surface tests**

Use a synthetic two-level triangle mesh with a known xyzw object transform.
Assert:

```python
grid = ZUpHeightGrid.from_world_mesh(vertices, triangles, cell_size_m=0.02)
np.testing.assert_allclose(
    grid.sample_xy(np.array([[0.0, 0.0], [0.4, 0.0]], np.float32)),
    [0.0, 0.2],
    atol=1e-5,
)
with self.assertRaisesRegex(ValueError, "outside"):
    grid.sample_xy(np.array([[100.0, 100.0]], np.float32))
```

Also assert object `root_pos`, xyzw `root_quat`, and scale transform USD mesh
vertices into the robot's Z-up world before rasterization.

- [ ] **Step 2: Run the surface test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  python3 -B -m unittest -v tests.python.test_torch_stair_surface
```

Expected: import failure for `surface.ZUpHeightGrid`.

- [ ] **Step 3: Implement the strict native surface**

Define:

```python
@dataclass(frozen=True)
class ZUpHeightGrid:
    origin_xy: np.ndarray
    cell_size_m: float
    height_z: np.ndarray

    def sample_xy(self, points_xy: np.ndarray) -> np.ndarray:
        return _sample_fixed_diagonal(self, points_xy)

    def as_npz_fields(self) -> dict[str, np.ndarray]:
        return {
            "origin_xy": self.origin_xy,
            "cell_size_m": np.array([self.cell_size_m], np.float32),
            "height_z": self.height_z,
        }

    @staticmethod
    def load(path: Path) -> "ZUpHeightGrid":
        with np.load(path, allow_pickle=False) as data:
            return ZUpHeightGrid(
                np.asarray(data["origin_xy"], np.float32),
                float(np.asarray(data["cell_size_m"]).reshape(-1)[0]),
                np.asarray(data["height_z"], np.float32),
            )
```

Use fixed-diagonal triangle interpolation, finite float32 storage, a 0.02-metre
cell, and an explicit domain. Add a 0.75-metre planar margin around the union of
the transformed object mesh and robot-root trajectory. Outside-domain queries
raise; they never silently return flat ground.

- [ ] **Step 4: Write failing transactional publication tests**

Tests must assert:

- exactly five `motion.npz` files are published (`flat` plus four stairs);
- every stair clip has one `terrain.npz`;
- the flat clip declares a flat provider without copying a terrain grid;
- `manifest.json` records all source hashes and output shapes;
- the resulting directory loads through `MotionFolder.load`;
- a forced failure before rename preserves the prior destination exactly; and
- no `.tmp-*` publication directory remains.

- [ ] **Step 5: Implement publisher and CLI**

Publish this ignored tree:

```text
build/torch-stair-small/
  manifest.json
  flat/motion.npz
  stair/0000/motion.npz
  stair/0000/terrain.npz
  stair/0001/motion.npz
  stair/0001/terrain.npz
  stair/0002/motion.npz
  stair/0002/terrain.npz
  stair/updown-0000/motion.npz
  stair/updown-0000/terrain.npz
```

The CLI accepts explicit `--grail-root`, `--g1-xml`, `--flat-motion`, and
`--output`. It writes a sibling staging directory, validates every generated
file and hash, fsyncs files/directories, and atomically replaces the destination.
Copy the 60 MB flat NPZ once; do not symlink runtime inputs.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  python3 -B -m unittest -v \
  tests.python.test_torch_stair_surface \
  tests.python.test_torch_stair_publish
```

Expected: PASS.

Commit:

```bash
git add .gitignore resources/build_g1_torch_stair_slice.py \
  resources/g1_torch_stair_builder tests/python/test_torch_stair_surface.py \
  tests/python/test_torch_stair_publish.py
git commit -m "feat: publish aligned Torch stair slice"
```

---

### Task 3: Add an optional search-feature extension without changing flat matching

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_features.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_motion_features.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Consumes: an optional `SearchFeatureExtension`.
- Produces: dynamic database/query concatenation and cost decomposition while `extension=None` preserves the existing API and results.

- [ ] **Step 1: Write the failing extension and flat-equivalence tests**

Define the desired protocol through a test extension:

```python
class SearchFeatureExtension(Protocol):
    name: str
    dimension: int
    weight: float

    def database_rows(
        self, folder: MotionFolder, device: torch.device
    ) -> Sequence[torch.Tensor]:
        raise NotImplementedError

    def query_row(
        self, state: GeneratedFeatureState, trajectory: CommandTrajectory
    ) -> torch.Tensor:
        raise NotImplementedError
```

Tests assert:

- two extension values produce a 29-column database;
- per-clip extension rows must equal each clip's `valid_frame_stop`;
- non-finite, wrong-dtype, wrong-device, zero-variance, or wrong-dimension rows
  are rejected;
- database and query both call the same extension object;
- the extension is frozen at construction; and
- `reset_clip_path="flat/motion.npz"` selects the minimum-joint-velocity row
  within that clip while the omitted argument preserves global reset behavior;
- two matchers built with omitted extension versus explicit `None` return
  identical selections, costs, joints, roots, and windows for 100 commands.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: `SearchFeatureExtension` is unavailable or constructor rejects the
new argument.

- [ ] **Step 3: Implement dynamic normalization and query concatenation**

Change signatures without changing default calls:

```python
TorchMotionDatabase.from_folder(
    folder, *, device, extension: SearchFeatureExtension | None = None,
    reset_clip_path: str | None = None
)

TorchMotionMatcher.from_folder(
    motions_dir, *, device="auto", config=MatcherConfig(),
    extension: SearchFeatureExtension | None = None,
    reset_clip_path: str | None = None
)
```

Build the original 27 raw columns exactly as today. When an extension exists,
append its rows and normalize its slice using one group scale
`mean(component_std) / extension.weight`. In `prepare_step`, append
`extension.query_row(feature_state, shaped.trajectory)` before applying the
shared normalization.

When `reset_clip_path` is present, resolve it against exact normalized
`MotionClip.relative_path` identity and select the lowest joint-velocity row
only within that clip. Missing or duplicate identity is a contract error.

Add `motion_feature_cost` and `extension_feature_cost` to
`MotionMatchDiagnostics`. Compute them from normalized squared residual slices;
their sum must equal `selected_feature_cost` within float32 tolerance.

- [ ] **Step 4: Expose dense root/foot kinematics for evaluation**

Add these immutable result tensors:

```python
dense_feature_body_position_window: torch.Tensor  # (46, 3, 3)
dense_feature_body_velocity_window: torch.Tensor  # (46, 3, 3)
```

The body order is root, left ankle-roll, right ankle-roll. Wire the already
computed `dense_bp` and `dense_bv` through `_make_result` and `_copy_result`.
Tests assert finite values, clone isolation, and row-zero equality with the
matcher's committed feature state.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_sonic
```

Expected: PASS with flat equivalence.

Commit:

```bash
git add sonic/python/mm_sonic/torch_motion_features.py \
  sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_features.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: extend Torch search features transactionally"
```

---

### Task 4: Implement legacy and dense terrain feature encoders

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_features.py`
- Create: `tests/python/test_sonic_torch_terrain_features.py`

**Interfaces:**
- Consumes: `build/torch-stair-small/manifest.json`, clip height grids, and one active query-scene alignment.
- Produces: `TerrainDataset.load(root)`, `TerrainFeatureExtension.for_condition(dataset, condition, query_scene, weight)`, and exact database/query feature rows.

- [ ] **Step 1: Write failing encoder tests with an independent oracle**

Create analytic flat, ramp, and two-step grids. Use an independent NumPy sampler
in the test and assert:

```python
legacy = extension_for("legacy", weight=4.0)
dense = extension_for("dense", weight=4.0)
self.assertEqual(legacy.dimension, 4)
self.assertEqual(dense.dimension, 91)
self.assertEqual(tuple(dense.database_rows(folder, cpu)[0].shape), (valid, 91))
np.testing.assert_allclose(
    dense.query_row(state, trajectory).numpy(),
    oracle_dense_patch(test_grid, state, trajectory),
    rtol=0.0,
    atol=1e-5,
)
```

The tests must prove:

- forward-major grid order and exact coordinates;
- common world translation and yaw do not change local observations when the
  terrain alignment changes with them;
- adding a constant to all surface heights does not change observations;
- legacy points follow the curved command centerline by arc distance;
- stopped motion extends from current heading;
- database rows use recorded root trajectory and each clip's own terrain;
- query rows use the selected test terrain/alignment;
- flat clips produce zeros; and
- any sample outside the grid is rejected before search.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_features
```

Expected: import failure for `mm_sonic.torch_terrain_features`.

- [ ] **Step 3: Implement terrain dataset and Torch sampler**

Define:

```python
@dataclass(frozen=True)
class TerrainSceneAlignment:
    translation_world_xy: torch.Tensor
    yaw_world_rad: torch.Tensor

@dataclass(frozen=True)
class TerrainDataset:
    motion_root: Path
    clip_grids: Sequence[ZUpTorchHeightGrid | None]
    manifest_sha256: str

    @staticmethod
    def load(root: str | Path, *, device: str | torch.device) -> "TerrainDataset":
        return _load_terrain_dataset(Path(root), torch.device(device))
```

Load float32 grids once onto the database device. Implement vectorized
fixed-diagonal sampling in Torch; do not perform per-frame CPU height calls.
`TerrainFeatureExtension` precomputes immutable database rows at construction and
uses the same local-point and sampler functions for live query rows.

- [ ] **Step 4: Validate real USD/object alignment against recorded contacts**

Add an opt-in real-data test guarded by
`MM_REAL_STAIR_ALIGNMENT_ORACLE=1`. At the original 25 Hz source frames,
forward-kinematics the recorded left/right ankle-roll bodies and sample the new
native USD/object surface beneath them. Require at least 100 near-contact frames
per foot and require each minimum clearance to match the G1 sole offset of
0.035 m within `1e-4` m.

Do not use the old `g1_terrain_takara_slopes_stairs` sidecar as an oracle.
Investigation during implementation showed that its rows for this source behave
as a single roughly 0.70 m curb and came from a different reconstruction
surface, not the four-step object pose stored with this recording. This check
also freezes the discovered GRAIL convention: object `root_quat` is source
wxyz, unlike robot `root_rot`, which is source xyzw.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_features
```

Expected: PASS.

Commit:

```bash
git add sonic/python/mm_sonic/torch_terrain_features.py \
  tests/python/test_sonic_torch_terrain_features.py
git commit -m "feat: add Torch stair terrain encoders"
```

---

### Task 5: Add deterministic A/B rollout and acceptance metrics

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_rollout.py`
- Create: `sonic/configs/experiments/torch_stair_small.json`
- Create: `tests/python/test_sonic_torch_terrain_rollout.py`

**Interfaces:**
- Consumes: one generated small dataset and condition `flat`, `legacy`, or `dense`.
- Produces: `run_stair_rollout(config, condition, device) -> StairRollout`, saved `rollout.npz`, `metrics.json`, and `events.jsonl`.

- [ ] **Step 1: Write failing deterministic rollout tests**

Use a synthetic matcher fixture and two-step grid. Assert:

- identical condition/config/inventory yields identical result hashes;
- all conditions consume the exact same command samples;
- no call to `mujoco.mj_step` or SONIC occurs;
- every 20 ms row records selection, split costs, transition, root, feet, and
  timing;
- penetration is `foot_z - terrain_height(foot_xy)`;
- progress is projection onto the frozen reference direction;
- integrated penetration uses `sum(max(0, -clearance) * 0.02)`; and
- acceptance implements the five exact design thresholds.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_rollout
```

Expected: import failure for `torch_terrain_rollout`.

- [ ] **Step 3: Implement the benchmark config**

Store:

```json
{
  "schema": "g1-torch-stair-small-experiment/v1",
  "dt": 0.02,
    "duration_s": 9.0,
    "query_scene": "stair/updown-0000/motion.npz",
    "reset_clip": "flat/motion.npz",
  "conditions": {
    "flat": {"encoder": null, "weight": 0.0},
    "legacy": {"encoder": "legacy", "weight": 4.0},
    "dense": {"encoder": "dense", "weight": 4.0}
  },
  "acceptance": {
    "latest_stair_selection_before_riser_m": 0.20,
    "minimum_reference_horizontal_progress_ratio": 0.90,
    "minimum_reference_root_height_gain_ratio": 0.80,
    "minimum_foot_clearance_m": -0.03,
    "maximum_penetration_integral_ratio_vs_flat": 0.50
  }
}
```

Derive the straight command speed, direction, first-riser location, reference
horizontal progress, and reference root-height gain from the ascent segment of
converted `_updown__0000` before constructing any matcher. The ordinary
`__0000`–`__0002` clips begin at the top and are descent recordings, so they
cannot define the uphill query start. Freeze the derived values into the
run-local resolved config and hash it.

- [ ] **Step 4: Implement rollout and structured results**

At each step, call the normal transactional matcher API, commit exactly once,
sample clearance from the dense feature-body window row zero, and append one
immutable row. Save arrays with `allow_pickle=False`; write sorted JSON and
line-oriented events. Include build manifest SHA, matcher inventory SHA,
condition, encoder dimensions/weight, device, Torch version, CUDA device name,
and latency percentiles. Save all 29 joint positions, root position, root wxyz
orientation, and root/left-foot/right-foot diagnostic positions for every step
so the viewer never needs to recreate matcher state.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_rollout
```

Expected: PASS.

Commit:

```bash
git add sonic/python/mm_sonic/torch_terrain_rollout.py \
  sonic/configs/experiments/torch_stair_small.json \
  tests/python/test_sonic_torch_terrain_rollout.py
git commit -m "feat: benchmark Torch stair kinematics"
```

---

### Task 6: Add a read-only rollout viewer and one-command experiment CLI

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_viewer.py`
- Create: `tests/python/test_sonic_torch_terrain_viewer.py`
- Create: `TORCH_TERRAIN_QUICKSTART.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: saved rollout and height grid.
- Produces: `python -m mm_sonic.torch_terrain_rollout` for building/running conditions and `python -m mm_sonic.torch_terrain_viewer` for interactive playback or MP4 recording.

- [ ] **Step 1: Write failing viewer-boundary tests**

Tests assert the viewer:

- validates rollout/grid hashes and shapes before creating a window;
- never imports or constructs `TorchMotionMatcher`;
- advances only through saved frames;
- maps Escape to close, Space to pause, Left/Right to step, and R to restart;
- calls `mujoco.mj_forward` (never `mujoco.mj_step`) to reconstruct and render
  the complete 30-body G1 skeleton from saved root/joint kinematics;
- renders terrain patch sample points, selected source, condition, costs,
  clearance, and latency; and
- produces a non-empty headless PNG for one synthetic frame.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_viewer
```

Expected: import failure for `torch_terrain_viewer`.

- [ ] **Step 3: Implement viewer and CLI help**

Use matplotlib's 3D axes and animation loop. Load the pinned G1 MJCF, reorder
saved IsaacLab joints into MuJoCo qpos order, call `mujoco.mj_forward`, and draw
the named body-parent segments as the full skeleton. Render the height grid as a
surface, the three feature bodies and root trajectory as additional diagnostic
geometry, and a text panel for the exact saved metrics. The viewer is
deliberately not a second matcher and never mutates rollout files.

Document exact commands:

```bash
python3 resources/build_g1_torch_stair_slice.py \
  --grail-root /home/ubuntu/datasets/GRAIL/data/stair_p1 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --flat-motion /home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz \
  --output build/torch-stair-small

PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m \
  mm_sonic.torch_terrain_rollout \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --conditions flat legacy dense \
  --device cuda \
  --output build/torch-stair-small-results

PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m \
  mm_sonic.torch_terrain_viewer \
  --run build/torch-stair-small-results/dense \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
```

- [ ] **Step 4: Run GREEN and commit**

Run:

```bash
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_viewer
```

Expected: PASS.

Commit:

```bash
git add README.md TORCH_TERRAIN_QUICKSTART.md \
  sonic/python/mm_sonic/torch_terrain_viewer.py \
  tests/python/test_sonic_torch_terrain_viewer.py
git commit -m "docs: add Torch stair experiment workflow"
```

---

### Task 7: Qualify real data, run all three conditions, and publish the branch

**Files:**
- Create: `docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md`
- Modify only if evidence requires a tested fix: files introduced by Tasks 1-6.

**Interfaces:**
- Consumes: the four local real GRAIL recordings, flat Takara clip, L40S, and completed test suites.
- Produces: a reproducible real-data result pack, written conclusion, pushed branch, and branch URL.

- [ ] **Step 1: Build and validate the real five-clip dataset**

Run the documented build command. Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -c \
  'from mm_sonic.torch_motion_data import MotionFolder; \
f=MotionFolder.load("build/torch-stair-small"); \
print(len(f.clips), [c.frame_count for c in f.clips], f.inventory_sha256)'
```

Expected: five clips; flat frame count 34,863 and four stair frame counts 499.

- [ ] **Step 2: Run the real alignment oracle**

Run:

```bash
MM_REAL_STAIR_ALIGNMENT_ORACLE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_features
```

Expected: PASS with recorded left/right minimum contact-clearance error no
greater than `1e-4 m`.
If it fails, stop the A/B run and debug alignment; do not tune matcher weights
against a misregistered surface.

- [ ] **Step 3: Run full focused regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_data \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_sonic \
  tests.python.test_torch_stair_conversion \
  tests.python.test_torch_stair_surface \
  tests.python.test_torch_stair_publish \
  tests.python.test_sonic_torch_terrain_features \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_viewer
```

Expected: PASS.

- [ ] **Step 4: Run flat, legacy, and dense on CUDA**

Run the documented rollout command. Verify each condition contains
`rollout.npz`, `metrics.json`, and `events.jsonl`, then render one headless PNG
and interactively inspect the dense rollout.

Do not tune the dense grid layout or acceptance thresholds after viewing
condition outcomes. Weight changes are allowed only as separately named runs
that preserve the original weight-4 result.

- [ ] **Step 5: Record evidence and conclusion**

Write the result document with:

- exact branch/commit and data hashes;
- source/output frame counts;
- alignment error;
- database build, search p50/p95/p99, and step p50/p95/p99 latency;
- the five acceptance values for each condition;
- representative transition/source events around the first riser;
- links to run-local metrics and screenshot paths; and
- one conclusion: `promising`, `not promising`, or `invalid`, with the failed
  contract named for `invalid`.

- [ ] **Step 6: Verify clean state, commit results, and push**

Run:

```bash
git diff --check
git status --short
git add docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md
git commit -m "docs: report Torch stair kinematics results"
git push -u checkpoint research/g1-torch-terrain-kinematics
git status --short --branch
```

Expected: clean branch tracking
`checkpoint/research/g1-torch-terrain-kinematics`.
