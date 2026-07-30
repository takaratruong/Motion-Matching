# G1 In-Process Torch Motion Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the C++ Motion Matching server in one opt-in flat-locomotion path with a transactional, exact-search PyTorch matcher that publishes a dense 46-frame reference horizon to the released GEAR SONIC controller.

**Architecture:** Load native Takara Z-up/wxyz NPZ clips into immutable tensors, compute the same 27-dimensional pose/trajectory features offline and online, and search them exactly on CUDA. A stateful matcher owns generated reference state and prepares window-consistent inertialized results; a two-phase SONIC adapter publishes and acknowledges overlapping dense horizons before committing matcher state or advancing MuJoCo.

**Tech Stack:** Python 3.10, NumPy, PyTorch 2.13 with CUDA 13.0, `unittest`, existing `mm_sonic` ZMQ v1 publisher, existing GEAR SONIC binary, existing gated MuJoCo backend.

## Global Constraints

- Work only in `/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver` on `research/g1-low-latency-driver`; preserve unrelated changes.
- Follow test-driven development: add one focused RED test, run it and inspect the expected failure, add the minimum production behavior, then run GREEN before each commit.
- Do not mutate the shared `sonic/.venv` symlink target. Create the ignored `sonic/.torch-mm-venv` from `/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python --system-site-packages`.
- The live Torch environment must report Python 3.10, PyTorch `2.13.0+cu130`, and `torch.cuda.is_available() == True`; CPU remains a correctness fallback only.
- Accept only 50 Hz Takara NPZ clips with 29 IsaacLab-order joints, 30 IsaacLab-order bodies, native Z-up positions, and wxyz quaternions.
- Freeze layout identity `g1-29dof-isaaclab-v1`: pelvis `0`, left ankle-roll `18`, right ankle-roll `19`.
- Do not create Holden Y-up motion data or remap the database into MuJoCo order.
- Search features are exactly 27 values with horizons 15, 30, and 45 frames and weights `0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.5` by group.
- Exact search normally runs every five 20 ms calls; start, stop, reversal, and clip-end conditions force search.
- The matcher owns previous generated reference state; measured robot state never enters its query.
- Every result carries dense rows `0..45`; encoder rows are exact dense rows
  `range(0, 46, 5)`, spanning 0.9 seconds.
- Live integration uses `prepare_step -> publish/acknowledge -> matcher.commit -> SONIC action -> physics advance`; no failure before physics release may mutate matcher state or advance MuJoCo.
- Keep released GEAR policy/controller, observation configuration, encoder, and MuJoCo dynamics unchanged.
- Do not add terrain, depth, lidar, measured-state matching, root-conditioned training, action chunking, ANN/FAISS, custom CUDA/C++, or batched environment state.
- Keep all Torch imports out of existing baseline module-import paths. In
  particular, `manual_demo.py` and `scene.py` use branch-local lazy imports, and
  every new Torch-only test raises `unittest.SkipTest` before importing
  production Torch modules when the optional dependency is absent.

---

## File Map

- Modify `.gitignore`: ignore the isolated Torch runtime.
- Modify `sonic/pyproject.toml`: add a `torch-mm` optional dependency without changing baseline or integration extras.
- Modify `sonic/README.md`: document the isolated environment and opt-in launch.
- Create `sonic/python/mm_sonic/torch_motion_data.py`: layout, NPZ
  discovery/validation, owned clips, and deterministic inventory provenance.
- Create `sonic/python/mm_sonic/torch_motion_features.py`: quaternion/heading math, shared raw features, group normalization, immutable Torch database.
- Create `sonic/python/mm_sonic/torch_motion_matcher.py`: command shaping, exact selection, inertialization, transactional matcher, immutable dense result.
- Create `sonic/python/mm_sonic/torch_motion_sonic.py`: dense canonical-buffer adapter, acknowledged publication, matcher/physics committer, initial qpos conversion.
- Create `sonic/python/mm_sonic/torch_motion_canary.py`: deterministic offline scripts and synchronized latency benchmark.
- Modify `sonic/python/mm_sonic/manual_demo.py`: opt-in Torch backend that bypasses `MMChunkClient`.
- Modify `sonic/python/mm_sonic/gear_action.py`: expose the existing IsaacLab-to-MuJoCo joint-vector conversion instead of copying the permutation.
- Modify `sonic/python/mm_sonic/process.py`: split the existing control-channel
  release into an authenticated no-physics `ARMED` preparation and a
  single-use physics commit while preserving `release_steps`.
- Create `tests/python/torch_motion_test_utils.py`: deterministic synthetic native Takara clips and NumPy oracles.
- Create `tests/python/test_sonic_torch_motion_data.py`: loader/layout contract.
- Create `tests/python/test_sonic_torch_motion_features.py`: feature and normalization oracle.
- Create `tests/python/test_sonic_torch_motion_search.py`: command and exact-search semantics.
- Create `tests/python/test_sonic_torch_motion_inertialization.py`: spring/quaternion transition contract.
- Create `tests/python/test_sonic_torch_motion_matcher.py`: matcher state, two-phase transaction, dense windows.
- Create `tests/python/test_sonic_torch_motion_sonic.py`: adapter, overlapping publication, committer, initial qpos.
- Create `tests/python/test_sonic_torch_motion_canary.py`: offline behavior and benchmark reporting.
- Modify `tests/python/test_sonic_manual_demo.py`: parser, dependency ownership, no-MM-server process proof, Backspace reset.
- Modify `tests/python/test_sonic_gear_action.py`: public joint conversion proof.
- Create `docs/superpowers/results/2026-07-29-g1-torch-motion-matcher.md`: final identities, commands, latency, behavior, and live evidence.

### Task 1: Isolated Torch runtime and native Takara loader

**Files:**
- Modify: `.gitignore`
- Modify: `sonic/pyproject.toml`
- Modify: `sonic/README.md`
- Create: `sonic/python/mm_sonic/torch_motion_data.py`
- Create: `tests/python/torch_motion_test_utils.py`
- Create: `tests/python/test_sonic_torch_motion_data.py`

**Interfaces:**
- Consumes: NPZ fields and layout from the approved design.
- Produces: `G1_TAKARA_LAYOUT`, `MotionClip`, `MotionFolder.load(root)`,
  `MotionFolder.inventory_sha256`, `discover_motion_paths(root)`.

- [ ] **Step 1: Create the isolated test/runtime environment without touching `sonic/.venv`**

Run:

```bash
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m venv \
  --system-site-packages sonic/.torch-mm-venv
sonic/.torch-mm-venv/bin/pip install -e 'sonic[integration]'
sonic/.torch-mm-venv/bin/python - <<'PY'
import sys
import mujoco
import numpy
import torch
import zmq
print(sys.version.split()[0])
print(torch.__version__)
print(torch.cuda.is_available())
print(numpy.__version__, mujoco.__version__, zmq.__version__)
PY
```

Expected: Python `3.10.x`, Torch `2.13.0+cu130`, `True`, and successful imports. Confirm `readlink -f sonic/.venv` is unchanged before and after.

- [ ] **Step 2: Write RED loader tests and the reusable synthetic fixture**

Create a fixture function with this exact interface:

```python
def write_takara_clip(
    path: Path,
    *,
    frames: int = 60,
    fps: int = 50,
    yaw_rate: float = 0.0,
    root_velocity_xy: tuple[float, float] = (0.4, 0.0),
) -> Path:
    """Write one finite native Z-up/wxyz G1 NPZ and return motion.npz."""
```

Use pelvis index `0`, feet `18/19`, unit wxyz quaternions, analytically
consistent body velocities, and distinct joint values per frame. Implement
these named tests with explicit temporary directories, `np.savez`,
`assertRaisesRegex`, and exact shape/value assertions:

- `test_discovers_nested_clips_in_relative_path_order`
- `test_loads_exact_layout_and_makes_owned_arrays_read_only`
- `test_rejects_each_missing_field_with_relative_path`
- `test_rejects_wrong_fps_shapes_lengths_short_clip_and_nonfinite`
- `test_rejects_bad_quaternion_norm_and_canonicalizes_sign`
- `test_valid_source_frames_end_at_t_minus_46_inclusive`
- `test_inventory_digest_changes_with_relative_path_or_file_bytes`

- [ ] **Step 3: Run RED and verify the missing module is the only failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_data
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mm_sonic.torch_motion_data'`.

- [ ] **Step 4: Implement strict discovery and loading**

Create these exact public types:

```python
@dataclass(frozen=True)
class MotionLayout:
    identity: str
    joint_count: int
    body_count: int
    root_body_index: int
    left_foot_body_index: int
    right_foot_body_index: int


G1_TAKARA_LAYOUT = MotionLayout(
    identity="g1-29dof-isaaclab-v1",
    joint_count=29,
    body_count=30,
    root_body_index=0,
    left_foot_body_index=18,
    right_foot_body_index=19,
)


@dataclass(frozen=True)
class MotionClip:
    relative_path: str
    fps: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])

    @property
    def valid_frame_stop(self) -> int:
        return self.frame_count - 45


@dataclass(frozen=True)
class MotionFolder:
    root: Path
    layout: MotionLayout
    clips: Sequence[MotionClip]
    inventory_sha256: str
```

Implement `MotionFolder.load(root: str | Path, *, layout: MotionLayout =
G1_TAKARA_LAYOUT) -> MotionFolder`:

1. resolve the root and require a directory;
2. sort `root.rglob("motion.npz")` by POSIX relative path;
3. require at least one file;
4. load with `allow_pickle=False`;
5. validate required keys, scalar FPS 50, exact ranks/trailing shapes, equal
   frame counts, at least 46 frames, finite float32 conversion, and quaternion
   norm error no greater than `1e-4`;
6. normalize quaternions, flip every negative consecutive dot product per body,
   copy to contiguous arrays, and set `writeable=False`;
7. compute `inventory_sha256` from the ordered sequence of each relative POSIX
   path, a NUL separator, the file byte length, another NUL separator, and the
   file SHA-256; and
8. raise `ContractError` containing the relative path and failed field on the
   first failure.

- [ ] **Step 5: Add environment metadata**

Add:

```toml
[project.optional-dependencies]
torch-mm = [
  "torch>=2.13",
]
```

Preserve the existing `test` and `integration` extras. Add
`/sonic/.torch-mm-venv/` to `.gitignore`. Document the exact environment
creation and verification command in `sonic/README.md`, including the warning
that `sonic/.venv` is a shared symlink and must not be modified.

Refresh the editable install after adding the extra:

```bash
sonic/.torch-mm-venv/bin/pip install -e 'sonic[integration,torch-mm]'
```

- [ ] **Step 6: Run GREEN and baseline non-regression**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_data
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_schema tests.python.test_sonic_zmq_v1
```

Expected: all selected tests PASS; baseline tests must not import Torch.

- [ ] **Step 7: Commit**

```bash
git add .gitignore sonic/pyproject.toml sonic/README.md \
  sonic/python/mm_sonic/torch_motion_data.py \
  tests/python/torch_motion_test_utils.py \
  tests/python/test_sonic_torch_motion_data.py
git commit -m "feat: load native Takara motion folders"
```

### Task 2: Shared Torch features and immutable database

**Files:**
- Create: `sonic/python/mm_sonic/torch_motion_features.py`
- Modify: `sonic/python/mm_sonic/torch_motion_data.py`
- Modify: `tests/python/torch_motion_test_utils.py`
- Create: `tests/python/test_sonic_torch_motion_features.py`

**Interfaces:**
- Consumes: `MotionFolder`, `MotionClip`, `G1_TAKARA_LAYOUT`.
- Produces: `GeneratedFeatureState`, `CommandTrajectory`, `FeatureNormalization`, `TorchMotionDatabase.from_folder`, `extract_query_features`.

- [ ] **Step 1: Write RED feature/oracle tests**

Define an independent NumPy oracle in `torch_motion_test_utils.py`, not by
calling production helpers. Implement these named tests:

- `test_feature_vector_is_exactly_27_values_in_frozen_group_order`
- `test_clip_row_and_generated_query_use_identical_extractor`
- `test_global_xy_translation_and_common_z_yaw_do_not_change_features`
- `test_group_means_scales_and_normalized_rows_match_numpy_oracle`
- `test_zero_variation_group_fails_database_build`
- `test_database_provenance_and_successor_lookup_do_not_cross_clips`
- `test_cpu_and_cuda_database_tensors_are_immutable_by_api`

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_features
```

Expected: FAIL importing `mm_sonic.torch_motion_features`.

- [ ] **Step 3: Implement quaternion and query feature primitives**

Create these frozen interfaces:

```python
FEATURE_HORIZON_FRAMES = (15, 30, 45)
FEATURE_GROUPS = (
    ("left_foot_position", slice(0, 3), 0.75),
    ("right_foot_position", slice(3, 6), 0.75),
    ("left_foot_velocity", slice(6, 9), 1.0),
    ("right_foot_velocity", slice(9, 12), 1.0),
    ("pelvis_velocity", slice(12, 15), 1.0),
    ("trajectory_position", slice(15, 21), 1.0),
    ("trajectory_facing", slice(21, 27), 1.5),
)


@dataclass(frozen=True)
class GeneratedFeatureState:
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    root_linear_velocity_world: torch.Tensor
    left_foot_position_world: torch.Tensor
    right_foot_position_world: torch.Tensor
    left_foot_velocity_world: torch.Tensor
    right_foot_velocity_world: torch.Tensor


@dataclass(frozen=True)
class CommandTrajectory:
    position_world_xy: torch.Tensor
    facing_world_xy: torch.Tensor
```

Implement `extract_query_features(state: GeneratedFeatureState, trajectory:
CommandTrajectory) -> torch.Tensor` with Z-up wxyz yaw extraction, inverse-yaw
rotation, root-relative foot positions, rotated world velocities, and flattened
three-horizon XY position/facing values. Use positive X as forward. Reject
wrong shapes, dtype/device disagreement, and non-finite tensors.

- [ ] **Step 4: Implement group normalization and database construction**

Use:

```python
@dataclass(frozen=True)
class FeatureNormalization:
    _component_mean: torch.Tensor
    _component_scale: torch.Tensor

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self._component_mean) / self._component_scale

    def parameters_copy(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._component_mean.clone(), self._component_scale.clone()


@dataclass(frozen=True)
class TorchMotionDatabase:
    folder: MotionFolder
    device: torch.device
    normalization: FeatureNormalization
    reset_row: int
    _search_features: torch.Tensor
    _search_clip_index: torch.Tensor
    _search_frame_index: torch.Tensor

    @property
    def feature_shape(self) -> tuple[int, int]:
        return tuple(self._search_features.shape)

    def normalized_features_copy(self) -> torch.Tensor:
        return self._search_features.clone()
```

Implement `TorchMotionDatabase.from_folder(folder: MotionFolder, *, device:
str | torch.device) -> TorchMotionDatabase` and
`row_for_source(clip_index: int, frame_index: int) -> int | None`. Build every
valid row by constructing `GeneratedFeatureState` from the clip and using the
shared extractor. For each group compute per-component mean, component
standard deviation, arithmetic mean group standard deviation, and scale
`group_std / weight`. Fail on non-positive/non-finite scale. Keep search and
provenance tensors private, return clones from diagnostic accessors, expose no
mutator, and keep a private `(clip, frame) -> row` mapping. The search helper
may consume the private matrix only inside the matcher module. Choose
`reset_row` by minimum joint-velocity squared norm with smallest-row tie break.

- [ ] **Step 5: Run GREEN plus the frozen real Takara construction smoke test**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_features
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B - <<'PY'
from mm_sonic.torch_motion_data import MotionFolder
from mm_sonic.torch_motion_features import TorchMotionDatabase
folder = MotionFolder.load("/home/ubuntu/Downloads/takara_walk_50hz.npz_v0")
db = TorchMotionDatabase.from_folder(folder, device="cuda")
print(len(folder.clips), db.feature_shape, db.device)
assert db.feature_shape == (34818, 27)
assert str(db.device) == "cuda"
PY
```

Expected: all tests PASS and smoke output includes `(34818, 27) cuda`.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_data.py \
  sonic/python/mm_sonic/torch_motion_features.py \
  tests/python/torch_motion_test_utils.py \
  tests/python/test_sonic_torch_motion_features.py
git commit -m "feat: build exact Torch motion features"
```

### Task 3: Command shaping and exact candidate selection

**Files:**
- Create: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Create: `tests/python/test_sonic_torch_motion_search.py`

**Interfaces:**
- Consumes: `TorchMotionDatabase`, `GeneratedFeatureState`, `CommandTrajectory`.
- Produces: `MatcherConfig`, `ShapedCommand`, `SearchDecision`, `bounded_velocity_step`, `bounded_yaw_step`, `predict_command_trajectory`, `select_exact_candidate`.

- [ ] **Step 1: Write RED command/search tests**

Add exact tests for acceleration, deceleration, yaw wrap, horizons, search
cadence triggers, masks, exclusion, transition penalty, incumbent retention,
clip-end forced transition, and lowest-row ties:

- `test_acceleration_deceleration_reversal_and_yaw_caps_are_exact`
- `test_prediction_samples_steps_15_30_45`
- `test_start_stop_reversal_and_clip_end_force_search`
- `test_cpu_and_cuda_match_float32_numpy_oracle`
- `test_local_20_frame_exclusion_applies_only_to_current_clip`
- `test_candidate_must_beat_incumbent_after_point_one_penalty`
- `test_missing_incumbent_requires_lowest_cost_valid_candidate`
- `test_equal_cost_chooses_smallest_global_row`

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_search
```

Expected: FAIL importing matcher symbols.

- [ ] **Step 3: Implement frozen configuration and bounded commands**

Use:

```python
@dataclass(frozen=True)
class MatcherConfig:
    dt: float = 0.02
    search_interval_steps: int = 5
    acceleration_mps2: float = 1.5
    deceleration_mps2: float = 2.0
    yaw_rate_rad_s: float = math.radians(120.0)
    stop_speed_mps: float = 0.05
    reversal_speed_mps: float = 0.15
    exclusion_frames: int = 20
    transition_penalty: float = 0.1
    inertialization_halflife_s: float = 0.10


@dataclass(frozen=True)
class ShapedCommand:
    velocity_world_xy: torch.Tensor
    heading_world_yaw: torch.Tensor
    trajectory: CommandTrajectory
    force_search: bool
```

For velocity, choose deceleration when target norm is smaller or dot product is
negative; move toward target by at most `limit * dt` in Euclidean norm. For
yaw, use `atan2(sin(delta), cos(delta))` and clamp to
`yaw_rate_rad_s * dt`. Predict 45 internal 20 ms steps and gather steps
15/30/45.

- [ ] **Step 4: Implement exact selection**

Use:

```python
@dataclass(frozen=True)
class SearchDecision:
    selected_row: int
    incumbent_row: int | None
    incumbent_cost: float
    selected_feature_cost: float
    selected_total_cost: float
    searched: bool
    transitioned: bool


```

Implement `select_exact_candidate(database, normalized_query, *,
current_clip_index, current_frame_index, incumbent_row, search, config) ->
SearchDecision` with one dense float32 squared-L2 reduction. Mask the
20-frame same-clip neighborhood, add `0.1` only to non-incumbents, retain a
valid incumbent unless a candidate total cost is strictly smaller, and use the
first Torch `argmin` row for ties. When `search=False`, require the incumbent.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_search
```

Expected: all tests PASS on CPU and CUDA.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_search.py
git commit -m "feat: add exact Torch candidate search"
```

### Task 4: Critically damped transition primitives

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Create: `tests/python/test_sonic_torch_motion_inertialization.py`

**Interfaces:**
- Consumes: Torch tensors representing aligned source/target position, velocity, wxyz rotation, and angular velocity.
- Produces: `TransitionOffsets`, `inertialize_vector_transition`, `inertialize_quaternion_transition`, `sample_transition_offsets`.

- [ ] **Step 1: Write RED inertialization tests against the C++ spring equations**

Add an independent NumPy implementation of:

```python
y = (4.0 * math.log(2.0) / (halflife + 1.0e-5)) / 2.0
j1 = velocity_offset + position_offset * y
decay = math.exp(-y * time_s)
position = decay * (position_offset + j1 * time_s)
velocity = decay * (velocity_offset - j1 * y * time_s)
```

Test vector parity, quaternion shortest hemisphere, unit output, transition
time-zero reconstruction within `1e-5`, monotonically decreasing norms, and
dense samples at all 46 times.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_inertialization
```

Expected: FAIL because transition helpers are absent.

- [ ] **Step 3: Implement exact vector and quaternion spring helpers**

Implement wxyz multiply/inverse, shortest-hemisphere sign, scaled-angle-axis
log/exp, and:

```python
def decay_spring_offsets(
    position_offset: torch.Tensor,
    velocity_offset: torch.Tensor,
    *,
    halflife_s: float,
    time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = (4.0 * math.log(2.0) / (halflife_s + 1.0e-5)) / 2.0
    j1 = velocity_offset + position_offset * y
    decay = torch.exp(-y * time_s)
    return (
        decay * (position_offset + j1 * time_s),
        decay * (velocity_offset - j1 * y * time_s),
    )
```

Broadcast `time_s` over arbitrary state shapes without CPU round trips.
Quaternion offsets decay in scaled-angle-axis space and are reconstructed as
unit wxyz quaternions.

- [ ] **Step 4: Implement immutable transition offsets**

Create:

```python
@dataclass(frozen=True)
class TransitionOffsets:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_rotation_wxyz: torch.Tensor
    root_angular_velocity: torch.Tensor
    feature_body_position: torch.Tensor
    feature_body_velocity: torch.Tensor
    elapsed_s: float
```

Add constructors that accumulate an active old offset before subtracting the
new aligned target, matching `spring.h::inertialize_transition`. Add a sampler
that evaluates current plus dense future times `0.00..0.90` seconds.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_inertialization
```

Expected: all parity and continuity tests PASS.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_inertialization.py
git commit -m "feat: add window-consistent inertialization"
```

### Task 5: Transactional stateful matcher and dense results

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Create: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Consumes: database, command/search helpers, transition helpers.
- Produces: `MotionMatchDiagnostics`, `MotionMatchResult`, `PreparedMotionMatch`, `TorchMotionMatcher.from_folder`, `.reset`, `.prepare_step`, `.commit`, `.step`.

- [ ] **Step 1: Write RED matcher-state tests**

Implement:

- `test_reset_uses_lowest_velocity_row_and_origin_heading`
- `test_prepare_does_not_mutate_and_commit_mutates_exactly_once`
- `test_foreign_stale_and_double_commit_are_rejected`
- `test_step_is_prepare_then_commit_convenience`
- `test_ordinary_continuation_uses_selected_source_root_delta`
- `test_transition_aligns_xy_yaw_and_preserves_source_height_roll_pitch`
- `test_generated_query_uses_inertialized_previous_output`
- `test_dense_46_rows_and_sampled_10_rows_have_exact_index_parity`
- `test_clip_windows_never_clamp_or_cross`
- `test_invalid_command_dt_and_injected_failure_leave_state_unchanged`
- `test_device_is_fixed_and_database_tensors_do_not_change`

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: FAIL importing stateful matcher classes.

- [ ] **Step 3: Define immutable public results and private state token**

Use exact fields:

```python
@dataclass(frozen=True)
class MotionMatchDiagnostics:
    sequence: int
    selected_clip_path: str
    selected_frame: int
    incumbent_cost: float
    selected_feature_cost: float
    selected_total_cost: float
    searched: bool
    transitioned: bool
    force_search_reason: str | None
    search_time_ns: int | None
    step_time_ns: int


@dataclass(frozen=True)
class MotionMatchResult:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    dense_joint_position_window: torch.Tensor
    dense_joint_velocity_window: torch.Tensor
    dense_root_position_window: torch.Tensor
    dense_root_orientation_window_wxyz: torch.Tensor
    joint_position_window: torch.Tensor
    joint_velocity_window: torch.Tensor
    root_position_window: torch.Tensor
    root_orientation_window_wxyz: torch.Tensor
    diagnostics: MotionMatchDiagnostics


@dataclass(frozen=True)
class PreparedMotionMatch:
    result: MotionMatchResult
    _owner_token: object
    _base_sequence: int
    _next_state: object
```

Validate shapes, finite values, unit quaternions, dense/sample equality, and
clone every emitted tensor so callers cannot mutate matcher state.

- [ ] **Step 4: Implement reset and prepare/commit**

Implement these exact public methods on `TorchMotionMatcher`:

- `from_folder(motions_dir: str | Path, *, device: str = "auto", config:
  MatcherConfig = MatcherConfig()) -> TorchMotionMatcher`
- `reset() -> MotionMatchResult`
- `prepare_step(velocity_world_xy: tuple[float, float],
  heading_world_yaw: float, *, dt: float = 0.02) -> PreparedMotionMatch`
- `commit(prepared: PreparedMotionMatch) -> MotionMatchResult`
- `step(velocity_world_xy: tuple[float, float], heading_world_yaw: float, *,
  dt: float = 0.02) -> MotionMatchResult`
- read-only property `motion_inventory_sha256 -> str`

`step` calls `prepare_step` with the unchanged public arguments followed by
`return commit(prepared)`. `prepare_step` must build a separate next-state
object, select/align/inertialize one source frame, gather dense source rows
0..45, derive sampled rows with `torch.arange(0, 46, 5)`, validate the
result, and return without assigning any matcher field. `commit` checks owner,
base sequence, single use, and exact prepared identity before one state swap.

- [ ] **Step 5: Run GREEN and real-data 200-step smoke**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_matcher
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B - <<'PY'
from mm_sonic.torch_motion_matcher import TorchMotionMatcher
matcher = TorchMotionMatcher.from_folder(
    "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0",
    device="cuda",
)
for index in range(200):
    velocity = (0.5, 0.0) if index < 100 else (-0.5, 0.0)
    result = matcher.step(velocity, 0.0)
assert result.diagnostics.sequence == 200
print(result.diagnostics)
PY
```

Expected: unit tests PASS and the smoke test finishes with sequence 200 and
finite diagnostics.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: add transactional Torch motion matcher"
```

### Task 6: Dense SONIC publication and physics transaction

**Files:**
- Create: `sonic/python/mm_sonic/torch_motion_sonic.py`
- Modify: `sonic/python/mm_sonic/gear_action.py`
- Modify: `sonic/python/mm_sonic/process.py`
- Modify: `tests/python/test_sonic_gear_action.py`
- Modify: `tests/python/test_sonic_process.py`
- Create: `tests/python/test_sonic_torch_motion_sonic.py`

**Interfaces:**
- Consumes: `PreparedMotionMatch`, `MotionMatchResult`,
  `CanonicalTargetBuffer`, `PosePublisher`, and the released GEAR
  simulation-control channel.
- Produces: `isaaclab_to_mujoco_joint_vector`,
  `PreparedSimulationRelease`, `PreparedSonicReference`,
  `AcknowledgedSonicReference`, `SonicReferenceAdapter`,
  `TorchMotionCommitter`, `initial_qpos_from_match`.

- [ ] **Step 1: Write RED conversion, adapter, and failure-order tests**

Implement:

- `test_initial_qpos_uses_root_and_existing_joint_permutation`
- `test_prepare_copies_dense_cuda_rows_to_exact_little_endian_buffer`
- `test_sampled_rows_equal_dense_offsets_zero_to_45`
- `test_consecutive_ranges_are_n_to_n45_then_n1_to_n46`
- `test_prepare_rejects_wrong_sequence_shape_unit_or_layout`
- `test_success_order_is_prepare_send_ack_commit_release_record`
- `test_supersession_happens_before_publication`
- `test_prepare_send_and_ack_failures_do_not_commit_or_release`
- `test_commit_failure_after_ack_is_terminal_and_does_not_release`
- `test_release_failure_does_not_claim_successful_record`
- `test_prepare_release_waits_for_exact_stream_end_without_advancing_physics`
- `test_prepared_release_rejects_foreign_stale_and_double_commit`
- `test_release_steps_control_channel_remains_prepare_plus_commit_compatible`

Use injected fakes that append exact method names to an event list.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_gear_action \
  tests.python.test_sonic_process \
  tests.python.test_sonic_torch_motion_sonic
```

Expected: FAIL on the missing public conversion, missing gate split, and missing
adapter module.

- [ ] **Step 3: Promote the existing joint permutation**

In `gear_action.py`, keep the frozen tuple bytes unchanged and add:

```python
def isaaclab_to_mujoco_joint_vector(values: object) -> np.ndarray:
    source = np.asarray(values)
    if source.shape != (29,) or not np.all(np.isfinite(source)):
        raise ValueError("IsaacLab joint vector must contain 29 finite values")
    return np.ascontiguousarray(source[_ISAACLAB_TO_MUJOCO], dtype=np.float64)
```

Export it in `__all__` and prove exact permutation parity in the existing test.

- [ ] **Step 4: Split the control-channel physics gate transactionally**

In `process.py`, add an opaque frozen `PreparedSimulationRelease` carrying the
owning gate token, base release sequence, expected stream-frame end, and
validated pre-release state. Add:

- `SimulationPolicyGate.prepare_release(*, expected_stream_frame_end: int) ->
  PreparedSimulationRelease`
- `SimulationPolicyGate.commit_release(prepared: PreparedSimulationRelease,
  steps: int) -> AdvanceResult`

Torch mode requires `pause_strategy="control-channel"`.
`prepare_release` requires the gate paused, checks bound simulator identity and
process liveness, begins simulation-control synchronization, refreshes LowState
without stepping, finishes synchronization, and sends `ARM` with the published
final frame. It returns only after released GEAR replies `ARMED`, proving that
the input worker committed at least that stream frame while Control, Planner,
and physics remain fenced. It performs no simulator advance.

`commit_release` accepts only the exact current single-use object, advances the
simulator by the requested steps so GEAR produces the action, re-pauses the
control channel, validates exact step/time deltas, and increments the release
sequence. A failure after `ARMED` is terminal and is never retried.

Keep `release_steps` compatible: for control-channel mode it calls
`prepare_release` then `commit_release`; its process-pause branch remains
unchanged. Add fake-channel event-order tests and preserve all existing gate
tests.

- [ ] **Step 5: Implement the dense adapter**

Use:

```python
@dataclass(frozen=True)
class PreparedSonicReference:
    buffer: CanonicalTargetBuffer
    result_sequence: int
    global_frame_start: int
    global_frame_end: int
    publisher_prepared: object
    _adapter_token: object
    _base_acked_sequence: int


@dataclass(frozen=True)
class AcknowledgedSonicReference:
    publication: PreparedSonicReference
    prepared_release: PreparedSimulationRelease
    send_result: Mapping[str, object]
```

Implement `SonicReferenceAdapter.__init__(publisher: object,
gate: SimulationPolicyGate, *, initial_acked_sequence: int = 0,
initial_global_frame_end: int = 45)`,
`prepare(result: MotionMatchResult, *, global_frame_start: int) ->
PreparedSonicReference`, and `send_and_wait(prepared: PreparedSonicReference)
-> AcknowledgedSonicReference`. `prepare` copies dense result tensors once to
CPU little-endian float32 arrays and constructs contiguous int64 indices
`global_start..global_start+45`, validates sampled parity and that the result
sequence is exactly one after the last acknowledged sequence, builds
`CanonicalTargetBuffer`, and calls `publisher.prepare` without sending.
`send_and_wait` calls `publisher.send_prepared`, then
`gate.prepare_release(expected_stream_frame_end=global_frame_end)`; only after
GEAR returns `ARMED` does it advance the adapter's acknowledged sequence/global
cursor and return the prepared physics-release token. It rejects foreign,
stale, and double-send objects.

- [ ] **Step 6: Implement the transaction committer**

Implement `TorchMotionCommitter.__init__` with keyword-only `matcher`,
`adapter`, `gate`, `recorder`, positive `steps_per_policy`, and zero-defaulted
`next_command_index` plus one-defaulted `next_global_frame`. Implement
`run_one_step(command: CommandSample, *, command_is_current:
Callable[[CommandSample], bool] | None = None) -> TorchCommitRecord`.
Also implement `run_one_chunk` as a thin call to `run_one_step` so the existing
`ResponsiveScheduler` can drive one 20 ms Torch boundary without changing its
tested C++-backend contract.
Define `TorchCommitRecord` as a frozen dataclass carrying command/global frame
indices, matcher sequence, selected source, generated root pose, and timing.
Convert heading quaternion to Z-up yaw, call
`matcher.prepare_step`, perform the last supersession check, adapter prepare,
send/receive `ARMED`, matcher commit, then call
`gate.commit_release(acknowledged.prepared_release, steps_per_policy)`. Record
only after the release returns and increment command/global indices by one.
Return timing and generated root diagnostics through `TorchCommitRecord`; add
one explicit conversion from that record to the existing responsive evidence
sink instead of relying on duck typing. Do not catch and retry terminal
failures. When the last check is stale, raise the existing
`CandidateSuperseded` before `publisher.prepare` so the scheduler may safely
resample the same boundary.

- [ ] **Step 7: Run GREEN**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_gear_action \
  tests.python.test_sonic_process \
  tests.python.test_sonic_torch_motion_sonic \
  tests.python.test_sonic_zmq_v1
```

Expected: all tests PASS, including exact fake event order.

- [ ] **Step 8: Commit**

```bash
git add sonic/python/mm_sonic/gear_action.py \
  sonic/python/mm_sonic/process.py \
  sonic/python/mm_sonic/torch_motion_sonic.py \
  tests/python/test_sonic_gear_action.py \
  tests/python/test_sonic_process.py \
  tests/python/test_sonic_torch_motion_sonic.py
git commit -m "feat: publish transactional Torch SONIC windows"
```

### Task 7: Opt-in manual demo without the C++ matcher server

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/scene.py`
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `tests/python/test_sonic_scene.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Consumes: matcher, adapter/committer, existing X11 mapper, flat scene registry, GEAR process, gated simulator.
- Produces: CLI options `--motion-backend torch`, `--motions-dir`, `--torch-device`; flat Torch startup and Backspace restart.

- [ ] **Step 1: Write RED CLI and dependency-ownership tests**

Add parser tests requiring:

```text
--motion-backend torch
--motions-dir /absolute/path
--torch-device auto|cpu|cuda
```

Reject Torch mode unless scene is `sonic-flat-baseline`, terrain weight is
zero, and motions directory resolves. Add injected startup tests proving:

1. Torch mode constructs `TorchMotionMatcher` and never resolves or constructs
   `MMChunkClient`;
2. the reset result creates simulator qpos in MuJoCo joint order;
3. initial 46 rows are acknowledged before CONTROL;
4. each loop releases one 20 ms policy interval;
5. Backspace closes all owned resources, creates a fresh matcher, and restarts
   at global frame zero; and
6. C++ mode retains its existing behavior unchanged.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_scene
```

Expected: FAIL because the parser does not know `--motion-backend`.

- [ ] **Step 3: Add a flat scene registration path that has no MM identity**

Factor the existing authenticated flat branch into
`register_flat_reference_scene(*, registry_path: Path, official_scene_xml:
Path, output_dir: Path, expected_official_scene_sha256: str,
expected_robot_sha256: str, source_kind: str, source_sha256: str) ->
RegisteredScene`. Reuse the existing flat XML authentication and output logic.
Require `source_kind == "torch-motion-folder"` and a lowercase SHA-256 equal to
`matcher.motion_inventory_sha256`. Do not accept terrain artifacts or
MM catalog identity.

- [ ] **Step 4: Add the Torch startup/loop branch**

In `run_demo`, validate the new options before creating a `RunBundle`. Branch
before `mm_server` resolution:

```python
if namespace.motion_backend == "torch":
    return _run_torch_demo(namespace, episode_ordinal=episode_ordinal)
```

Implement `_run_torch_demo` using the existing resource-ownership pattern:

Import `TorchMotionMatcher`, `SonicReferenceAdapter`, and
`TorchMotionCommitter` inside `_run_torch_demo`, never at module scope, so the
baseline environment can still import and test `manual_demo.py` without Torch.

1. load matcher and reset;
2. register the authenticated flat scene without MM;
3. construct initial qpos from the reset result;
4. construct `PosePublisher`, `GearProcess`, and `GatedSimulatorClient`;
5. start GEAR in WAIT, enable stream, use the existing
   `publication_boundary`/`wait_for_stream_processing` proof to publish and
   acknowledge reset sequence zero at global rows `0..45`;
6. activate CONTROL only after the first reference and LowState barriers, bind
   `SimulationPolicyGate` with `pause_strategy="control-channel"`, and create
   the live adapter with initial acknowledged sequence zero/global end 45;
7. use existing X11/terminal command mapping and `TorchMotionCommitter` with
   `next_global_frame=1`; derive `steps_per_policy = round(0.02 / sim_dt)` and
   reject a non-integral ratio, then drive the committer through the existing
   `ResponsiveScheduler` at one committed 20 ms boundary per loop;
8. route Backspace through the existing episode supervisor; and
9. close resources in reverse order on every exit.

Keep the existing C++ path byte-for-byte except for parser defaults and the
early opt-in branch.

- [ ] **Step 5: Run GREEN and the complete existing manual-demo regression**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_operator_x11 \
  tests.python.test_sonic_responsive_scheduler \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_gated_sim
```

Expected: all tests PASS; the no-MM-server assertion must inspect constructor
calls rather than process names after cleanup.

- [ ] **Step 6: Document the opt-in launch**

Add one command to `sonic/README.md` using `sonic/.torch-mm-venv/bin/python`,
`--motion-backend torch`, the Takara folder, flat scene, responsive X11,
onscreen mode, and existing explicit GEAR/runtime/source-run paths. State that
physics remains fenced until each overlapping 46-row window is acknowledged.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/manual_demo.py \
  sonic/python/mm_sonic/scene.py sonic/README.md \
  tests/python/test_sonic_manual_demo.py tests/python/test_sonic_scene.py
git commit -m "feat: integrate Torch matcher into manual SONIC"
```

### Task 8: Offline behavior and synchronized performance canaries

**Files:**
- Create: `sonic/python/mm_sonic/torch_motion_canary.py`
- Create: `tests/python/test_sonic_torch_motion_canary.py`

**Interfaces:**
- Consumes: `TorchMotionMatcher`, frozen command classes.
- Produces: `CanarySegment`, `CanaryReport`, `run_offline_canary`, `benchmark_matcher`, module CLI.

- [ ] **Step 1: Write RED report and threshold tests**

Define a synthetic database with explicit forward, backward, left, right,
turning, and idle segments. Test that the command script contains:

```python
(
    "stand_to_forward",
    "forward_to_backward",
    "backward_to_left",
    "left_to_right",
    "translate_change_heading",
    "turn_in_place",
    "moving_to_stop",
    "stop_to_moving",
)
```

Assert immediate forced-search steps, signed selected root velocity/heading,
finite dense windows, time-zero transition reconstruction, offset monotonicity,
JSON schema, and failure when p99 gates are exceeded.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_canary
```

Expected: FAIL importing canary symbols.

- [ ] **Step 3: Implement offline behavior reporting**

Use:

```python
@dataclass(frozen=True)
class CanarySegment:
    name: str
    velocity_world_xy: tuple[float, float]
    heading_world_yaw: float
    steps: int


@dataclass(frozen=True)
class CanaryReport:
    schema: str
    passed: bool
    segment_results: Sequence[Mapping[str, object]]
    search_p99_ms: float | None
    step_p99_ms: float | None
```

Run each segment for at least 200 steps in real-data mode. Record selected
source root signed velocity, heading delta, search triggers, maximum transition
reconstruction error, and minimum offset-norm decrease. A failed required
predicate sets `passed=False` and names the exact segment/predicate.

- [ ] **Step 4: Implement synchronized benchmarking**

Warm up 200 steps. For 2,000 measured steps, use CUDA events around exact
search and `time.perf_counter_ns` around complete step. Record start, execute
the step, call `torch.cuda.synchronize()` on every measured iteration, then
record the end so the complete-step sample cannot measure enqueue time only.
Synchronize the search events before reading them and compute NumPy percentile
99. Fail the CLI unless:

```python
search_p99_ms < 1.0
step_p99_ms < 2.0
```

CPU reports timings but does not apply these thresholds.

- [ ] **Step 5: Run GREEN and real Takara gates**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_canary
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_motion_canary \
  --motions-dir /home/ubuntu/Downloads/takara_walk_50hz.npz_v0 \
  --device cuda \
  --output sonic/runs/torch-mm-offline-canary.json
```

Expected: tests PASS, command exits 0, report says `passed: true`, search p99
below 1 ms, complete step p99 below 2 ms.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_canary.py \
  tests/python/test_sonic_torch_motion_canary.py
git commit -m "test: add Torch matcher behavior and latency gates"
```

### Task 9: Full regression, live flat canary, and retained evidence

**Files:**
- Create: `docs/superpowers/results/2026-07-29-g1-torch-motion-matcher.md`
- Modify only if a verified defect is found: files introduced in Tasks 1–8 and their direct tests.

**Interfaces:**
- Consumes: all completed implementation tasks and explicit external runtime paths.
- Produces: one retained evidence note and launchable verified branch.

- [ ] **Step 1: Run all Torch unit/integration tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_data \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_inertialization \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_sonic \
  tests.python.test_sonic_torch_motion_canary \
  tests.python.test_sonic_gear_action \
  tests.python.test_sonic_process \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_operator_x11 \
  tests.python.test_sonic_responsive_scheduler \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_gated_sim
```

Expected: all tests PASS.

- [ ] **Step 2: Run the full baseline Python suite in the existing environment**

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_*.py' -v
```

Expected: all baseline tests PASS. Torch-only modules may be skipped only when
their test module explicitly detects the missing optional Torch dependency;
no existing test may be skipped newly.

- [ ] **Step 3: Run the real offline behavior/performance canary twice**

```bash
for run in 1 2; do
  PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
    -m mm_sonic.torch_motion_canary \
    --motions-dir /home/ubuntu/Downloads/takara_walk_50hz.npz_v0 \
    --device cuda \
    --output "sonic/runs/torch-mm-offline-canary-${run}.json"
done
```

Expected: both exit 0, select identical frame sequences for the deterministic
script, and pass both p99 gates.

- [ ] **Step 4: Launch the opt-in visible flat demo**

Use the exact explicit paths resolved by the current preflight and the README
command:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.manual_demo \
  --mode interactive \
  --input-source x11 \
  --responsive \
  --onscreen \
  --motion-backend torch \
  --motions-dir /home/ubuntu/Downloads/takara_walk_50hz.npz_v0 \
  --torch-device cuda \
  --scene-id sonic-flat-baseline \
  --route-id flat-12s \
  --terrain-weight 0 \
  --chunks 3000 \
  --gear-checkout /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-gate \
  --source-run /home/ubuntu/mm-flat-walk-stage-b-r16-hold-resume.tym9Fc/stage-b/stage-b-20260718T005901613783Z-8b63d477 \
  --runtime /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
```

Before execution, require all three external paths above to resolve, authenticate
the source-run manifest and runtime file hashes through the existing preflight,
and stop if any identity differs. Do not edit external checkouts.

- [ ] **Step 5: Execute the 60-second operator script**

Hold each class for at least four seconds: forward, backward, left, right,
translation plus heading change, turn in place, stop, and reversal. Press
Backspace once, repeat forward/reversal/stop, then exit cleanly. Require:

- no fall;
- no crash or frozen viewer;
- no stale command or sequence diagnostic;
- every publication contains 46 contiguous rows;
- matcher/search p99 gates remain satisfied;
- Backspace creates a fresh matcher sequence at zero; and
- no C++ MM server process was launched.

If a requirement fails, retain logs, reproduce with the shortest command
segment, use systematic debugging, add a RED regression, and change only the
smallest responsible Task 1–8 file.

- [ ] **Step 6: Perform two clean launch/exit/relaunch cycles**

Launch, hold forward for four seconds, reverse for four seconds, stop, and exit.
Repeat from a clean process table. Confirm no stale ZMQ port, DDS participant,
GEAR process, simulator process, or viewer survives either exit.

- [ ] **Step 7: Write retained evidence**

Record:

- branch and commit SHA;
- Takara path, clip count/frame count, and inventory digest;
- Python/Torch/CUDA/GPU identities;
- released policy, encoder, observation config, GEAR checkout, and MuJoCo
  identities;
- exact test commands and counts;
- both offline deterministic hashes and p99 values;
- live run directory and command-segment outcomes;
- reset and two relaunch outcomes;
- explicit statement that the C++ MM server was absent; and
- remaining non-goals.

- [ ] **Step 8: Run final diff and evidence checks**

```bash
git diff --check
git status --short
rg -n 'TBD|TODO|FIXME|PLACEHOLDER' \
  sonic/python/mm_sonic/torch_motion_*.py \
  docs/superpowers/results/2026-07-29-g1-torch-motion-matcher.md
```

Expected: `git diff --check` is silent, status contains only the evidence file
and any directly justified final fixes, and the placeholder scan has no output.

- [ ] **Step 9: Commit final evidence**

```bash
git add docs/superpowers/results/2026-07-29-g1-torch-motion-matcher.md
git commit -m "docs: qualify Torch motion matcher"
```
