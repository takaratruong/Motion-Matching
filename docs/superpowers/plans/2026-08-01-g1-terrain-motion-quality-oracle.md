# G1 Terrain Motion Quality Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic offline oracle that identifies whether terrain-motion quality is limited by source data, retrieval, placement, composition, corpus coverage, or rigid action representation before changing the matcher.

**Architecture:** Preserve the multi-horizon matcher and its authenticated v0 artifacts. Add a pure contact-quality evaluator, reconstruct source contacts from saved clip/frame identities, capture deterministic difficult states, and reuse the existing mirrored `ContactPhaseActionIndex`, foothold planner, rigid placement, and terrain validator for exhaustive ranking. Compare native, rigidly placed, and current-style inertialized previews without IK or warping, then publish a classification report and visual contact sheets.

**Tech Stack:** Python 3.10, NumPy, PyTorch/CUDA, MuJoCo Python bindings, `unittest`, existing Torch motion/contact-oracle modules.

## Global Constraints

- Work only in `/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver` on `research/g1-torch-terrain-kinematics`.
- Preserve the v0 multi-horizon matcher and all existing baseline artifacts.
- Do not modify the pre-existing dirty landing-bridge, G1 FK, contact-segment, or associated test files.
- Do not connect Sonic, tracking, physics stepping, depth inference, or learned models.
- Do not use IK or contact warping in oracle output.
- Reuse `ContactPhaseActionIndex`, exact sagittal mirroring, `plan_footholds`, `place_action`, and `validate_placement`.
- Search the complete action inventory without online shortlist limits.
- Exclude timing fields from deterministic hashes.
- Write and observe each failing unit test before production changes.

---

### Task 1: Truthful source-conditioned contact metrics

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_contact_quality.py`
- Test: `tests/python/test_sonic_torch_terrain_contact_quality.py`

**Interfaces:**
- Consumes: emitted feet/surfaces, source support masks, root/command trajectories, and a transition-start mask.
- Produces: `ContactQualityMetrics` and `evaluate_contact_quality(...)` with no dataset or renderer dependency.

- [ ] **Step 1: Write failing synthetic metric tests**

Define tests for a clean alternating step, a source-supported floating foot, a 12-frame no-contact run, a moving chunk with no unload, and stance drift concentrated after a transition. Use this import and contract:

```python
from mm_sonic.torch_terrain_contact_quality import evaluate_contact_quality

metrics = evaluate_contact_quality(
    foot_position_world=feet,
    foot_surface_height_m=surface,
    source_support_mask=source_support,
    root_position_world=root,
    command_velocity_world_xy=command,
    transition_start_mask=transitions,
    dt_s=0.02,
)
self.assertEqual(metrics.complete_step_count, 1)
self.assertEqual(metrics.maximum_no_contact_frames, 12)
self.assertGreater(metrics.expected_stance_floating_fraction[0], 0.0)
self.assertGreater(metrics.transition_source_stance_drift_m, 0.0)
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_contact_quality
```

Expected: import failure because `torch_terrain_contact_quality` does not exist.

- [ ] **Step 3: Implement immutable metrics and validation**

Create:

```python
@dataclass(frozen=True)
class ContactQualityMetrics:
    emitted_contact_mask: np.ndarray
    expected_stance_floating_mask: np.ndarray
    contact_agreement_fraction: float
    expected_stance_floating_fraction: tuple[float, float]
    maximum_no_contact_frames: int
    unload_count: tuple[int, int]
    touchdown_count: tuple[int, int]
    complete_step_count: int
    complete_steps_per_m: float
    source_stance_drift_m: tuple[float, float]
    transition_source_stance_drift_m: float
    steady_source_stance_drift_m: float
    command_to_unload_frames: tuple[int, ...]
    command_to_touchdown_frames: tuple[int, ...]

def evaluate_contact_quality(
    *,
    foot_position_world: np.ndarray,
    foot_surface_height_m: np.ndarray,
    source_support_mask: np.ndarray,
    root_position_world: np.ndarray,
    command_velocity_world_xy: np.ndarray,
    transition_start_mask: np.ndarray,
    dt_s: float = 0.02,
) -> ContactQualityMetrics:
    arrays = _validated_contact_arrays(
        foot_position_world,
        foot_surface_height_m,
        source_support_mask,
        root_position_world,
        command_velocity_world_xy,
        transition_start_mask,
    )
    return _measure_validated_contact_quality(*arrays, dt_s=dt_s)
```

Infer emitted contact with the existing 3.5 cm ankle-origin sole offset, 2 cm clearance tolerance, and 0.12 m/s vertical-speed limit. Count source contact transitions directly. Attribute horizontal drift whenever the same source foot is supported on adjacent frames, regardless of emitted contact, so floating cannot remove drift from the denominator. Mark the first 15 frames from every transition as its neighborhood.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all contact-quality tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_contact_quality.py \
  tests/python/test_sonic_torch_terrain_contact_quality.py
git commit -m "feat: measure source-conditioned terrain contact quality"
```

---

### Task 2: Reconstruct source contacts from authenticated rollout identities

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_quality_artifacts.py`
- Test: `tests/python/test_sonic_torch_terrain_quality_artifacts.py`

**Interfaces:**
- Consumes: `TerrainDataset` and saved omni `arrays.npz` mappings.
- Produces: `reconstruct_source_support(...)`, `transition_start_mask(...)`, `analyze_saved_route(...)`, and canonical JSON.

- [ ] **Step 1: Write failing reconstruction and hash tests**

Use a fake dataset with two clips and assert consecutive source frames retain their support rows, a clip switch marks one transition, unknown clip identities fail closed, and timing-array changes do not alter the report hash:

```python
support = reconstruct_source_support(dataset, clip_paths, source_frames)
np.testing.assert_array_equal(support, expected_support)
np.testing.assert_array_equal(
    transition_start_mask(clip_paths, source_frames),
    np.array([True, False, True]),
)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_artifacts
```

Expected: import failure for the new artifact module.

- [ ] **Step 3: Implement read-only artifact analysis**

Implement:

```python
@dataclass(frozen=True)
class SavedRouteQuality:
    route_name: str
    metrics: ContactQualityMetrics
    source_support_mask: np.ndarray
    transition_start_mask: np.ndarray
    deterministic_sha256: str

def reconstruct_source_support(
    dataset: TerrainDataset,
    selected_clip_path: np.ndarray,
    selected_source_frame: np.ndarray,
) -> np.ndarray:
    clip_lookup = _unique_clip_lookup(dataset)
    return _source_support_rows(
        dataset, clip_lookup, selected_clip_path, selected_source_frame
    )

def analyze_saved_route(
    dataset: TerrainDataset,
    route_name: str,
    arrays: Mapping[str, np.ndarray],
) -> SavedRouteQuality:
    support = reconstruct_source_support(
        dataset, arrays["selected_clip_path"], arrays["selected_source_frame"]
    )
    transitions = transition_start_mask(
        arrays["selected_clip_path"], arrays["selected_source_frame"]
    )
    return _saved_route_quality(route_name, arrays, support, transitions)
```

Resolve each relative clip path exactly once. Use the existing authenticated source-support function without modifying its file. Hash route name, non-timing arrays, reconstructed support, transitions, and serialized metrics.

- [ ] **Step 4: Run artifact and neighboring metric tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_artifacts \
  tests.python.test_sonic_torch_terrain_contact_quality \
  tests.python.test_sonic_torch_terrain_omni_metrics
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_quality_artifacts.py \
  tests/python/test_sonic_torch_terrain_quality_artifacts.py
git commit -m "feat: authenticate terrain motion quality evidence"
```

---

### Task 3: Frozen difficult-state records

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_quality_states.py`
- Test: `tests/python/test_sonic_torch_terrain_quality_states.py`

**Interfaces:**
- Consumes: one saved route, reconstructed source contacts, query terrain sampling, and deterministic capture specifications.
- Produces: immutable `FrozenQualityState`, capture selection, NPZ/JSON serialization, and authentication.

- [ ] **Step 1: Write failing state-selection and round-trip tests**

Cover command-segment boundaries, the first split-height stance, the first elevated reversal boundary, duplicate capture suppression, terrain patch shape, mutation rejection, and byte-stable save/load:

```python
states = capture_quality_states(
    route_name="turn-90-middle-left",
    arrays=arrays,
    source_support_mask=support,
    terrain_patch_sampler=sample_patch,
)
self.assertTrue(any(state.reason == "command-boundary" for state in states))
self.assertTrue(any(state.reason == "split-height-stance" for state in states))
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_states
```

Expected: import failure for `torch_terrain_quality_states`.

- [ ] **Step 3: Implement state records and capture policy**

Create:

```python
@dataclass(frozen=True)
class FrozenQualityState:
    state_id: str
    route_name: str
    route_frame: int
    reason: str
    qpos: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_yaw_world: float
    foot_position_world: np.ndarray
    source_support_mask: np.ndarray
    command_velocity_world_xy: np.ndarray
    command_heading_world_yaw: float
    terrain_patch_world_xyh: np.ndarray
    selected_clip_path: str
    selected_source_frame: int

def capture_quality_states(
    *,
    route_name: str,
    arrays: Mapping[str, np.ndarray],
    source_support_mask: np.ndarray,
    terrain_patch_sampler: Callable[[np.ndarray], np.ndarray],
) -> tuple[FrozenQualityState, ...]:
    capture_frames = _quality_capture_frames(arrays, source_support_mask)
    return _frozen_states_for_frames(
        route_name, arrays, source_support_mask, terrain_patch_sampler,
        capture_frames,
    )
```

Use a 21-by-21 grid covering 1.0 m square around the root. Capture frame zero, every command-segment boundary, every source discontinuity, the first frame with foot-surface height difference at least 8 cm, and the first elevated frame preceding each reversal or turn command. Deduplicate by `(route_name, route_frame, reason)` and hash canonical content into `state_id`.

- [ ] **Step 4: Implement atomic state corpus save/load**

Write `save_quality_state_corpus(states, output)` and `load_quality_state_corpus(output)`. Store numeric arrays in compressed NPZ and identities/offsets in canonical JSON through a staging directory renamed atomically.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all state tests pass.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_quality_states.py \
  tests/python/test_sonic_torch_terrain_quality_states.py
git commit -m "feat: freeze difficult terrain motion states"
```

---

### Task 4: Native action quality descriptors

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_action_quality.py`
- Test: `tests/python/test_sonic_torch_terrain_action_quality.py`

**Interfaces:**
- Consumes: existing immutable `ContactPhaseAction` instances.
- Produces: `NativeActionQuality`, natural landing/contact descriptors, and two-step successor descriptors.

- [ ] **Step 1: Write failing native-quality tests**

Construct synthetic actions with a stationary stance foot, a sliding stance foot, clean/penetrating swing arcs, and an exact alternating successor. Assert natural landing, landing time, native drift, and two-step availability:

```python
quality = describe_native_action_quality(action)
np.testing.assert_allclose(quality.natural_landing_local_xyz, expected)
self.assertEqual(quality.landing_frame_offset, action.frame_count - 1)
self.assertGreater(bad_quality.source_stance_drift_m, quality.source_stance_drift_m)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_action_quality
```

Expected: import failure for the new action-quality module.

- [ ] **Step 3: Implement descriptors without changing the action index**

Create:

```python
@dataclass(frozen=True)
class NativeActionQuality:
    action_index: int
    source_key: tuple[int, int, int, bool]
    entry_support: tuple[bool, bool]
    swing_foot: int
    landing_frame_offset: int
    natural_landing_local_xyz: np.ndarray
    root_displacement_local_xy: np.ndarray
    root_yaw_delta_rad: float
    source_stance_drift_m: float
    minimum_swing_clearance_m: float
    entry_joint_speed_norm: float
    terminal_joint_speed_norm: float
    exact_successor_index: int | None

def build_native_quality_index(
    index: ContactPhaseActionIndex,
) -> tuple[NativeActionQuality, ...]:
    return tuple(
        describe_native_action_quality(action_index, action, index)
        for action_index, action in enumerate(index.actions)
    )
```

Compute source stance drift directly from action foot positions and support masks. Use `index.exact_successor_indices` to expose coherent two-step windows; do not invent graph edges.

- [ ] **Step 4: Run action-quality and existing oracle-action tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_action_quality \
  tests.python.test_sonic_torch_contact_oracle_actions
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_action_quality.py \
  tests/python/test_sonic_torch_terrain_action_quality.py
git commit -m "feat: describe native terrain action quality"
```

---

### Task 5: Exhaustive multi-objective quality ranking

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_quality_oracle.py`
- Test: `tests/python/test_sonic_torch_terrain_quality_oracle.py`

**Interfaces:**
- Consumes: `FrozenQualityState`, mirrored `ContactPhaseActionIndex`, native qualities, one planned foothold, query-surface callback, and existing placement constraints.
- Produces: `QualityOracleResult` with four independently ranked top-five sets and exhaustive rejection counts.

- [ ] **Step 1: Write failing ranking tests**

Use synthetic actions that independently win landing, continuity, combined, and exact-successor categories. Assert every inventory row is evaluated, an airborne/contact-mismatched entry is rejected, and stable source-key order breaks ties:

```python
result = rank_quality_actions(
    state=state,
    index=index,
    native_quality=qualities,
    desired_landing_world_xyz=landing,
    command_target_world_xy=target,
    sample_surface=sample_surface,
    constraints=OracleConstraints(),
    top_k=5,
)
self.assertEqual(result.evaluated_action_count, len(index.actions))
self.assertEqual(result.best_landing[0].action_index, landing_winner)
self.assertEqual(result.best_continuity[0].action_index, continuity_winner)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_oracle
```

Expected: import failure for `torch_terrain_quality_oracle`.

- [ ] **Step 3: Implement explicit cost components**

Create immutable `QualityCandidateScore` with action identity, feasibility, landing error, support mismatch, joint position/velocity boundary error, command displacement/facing error, native stance drift, clearance margin, and total. Implement four stable orderings:

```python
@dataclass(frozen=True)
class QualityOracleResult:
    state_id: str
    evaluated_action_count: int
    rejected_by_reason: Mapping[str, int]
    best_landing: tuple[QualityCandidateScore, ...]
    best_continuity: tuple[QualityCandidateScore, ...]
    best_combined: tuple[QualityCandidateScore, ...]
    best_two_step: tuple[QualityCandidateScore, ...]
    deterministic_sha256: str
```

Require exact entry support, call existing `place_action` and `validate_placement`, and compare the terminal swing foot to the desired landing. Keep cost components unweighted in the record. Use dimension-normalized squared residuals only for the combined ordering; do not reuse the online shortlist.

- [ ] **Step 4: Add coverage-failure tests**

Assert zero feasible candidates returns a valid result with empty ranked sets and structured rejection counts rather than an exception. Assert a missing desired foothold fails input validation before enumeration.

- [ ] **Step 5: Run ranking and neighboring oracle tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_oracle \
  tests.python.test_sonic_torch_contact_oracle_search \
  tests.python.test_sonic_torch_terrain_action_quality
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_quality_oracle.py \
  tests/python/test_sonic_torch_terrain_quality_oracle.py
git commit -m "feat: rank exhaustive terrain motion quality"
```

---

### Task 6: Native, placed, and inertialized boundary previews

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_quality_preview.py`
- Test: `tests/python/test_sonic_torch_terrain_quality_preview.py`

**Interfaces:**
- Consumes: a frozen state, selected contact action, rigid placement, G1 foot-kinematics callback, and inertialization halflife.
- Produces: three immutable preview stages with qpos, feet, support, and boundary/contact metrics.

- [ ] **Step 1: Write failing stage-isolation tests**

Use a synthetic action whose native stance is stationary, rigid placement preserves pairwise foot displacement, and an initial joint/root offset creates drift only in the inertialized stage:

```python
preview = build_quality_preview(
    state=state,
    action=action,
    foot_kinematics=fk,
    inertialization_halflife_s=0.10,
)
self.assertAlmostEqual(preview.native.source_stance_drift_m, 0.0)
self.assertAlmostEqual(preview.placed.source_stance_drift_m, 0.0)
self.assertGreater(preview.composed.source_stance_drift_m, 0.0)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_preview
```

Expected: import failure for the preview module.

- [ ] **Step 3: Implement the three stages**

Create:

```python
@dataclass(frozen=True)
class QualityPreviewStage:
    qpos: np.ndarray
    foot_position_world: np.ndarray
    source_support_mask: np.ndarray
    joint_boundary_jump_rad: float
    joint_velocity_boundary_jump_rad_s: float
    root_boundary_jump_m: float
    source_stance_drift_m: float

@dataclass(frozen=True)
class QualityPreview:
    native: QualityPreviewStage
    placed: QualityPreviewStage
    composed: QualityPreviewStage
```

Native uses source-local root/joints. Placed uses existing `place_action` and unchanged source joints. Composed computes current-minus-placed root, quaternion scaled-axis, joint-position, joint-velocity, and root-velocity offsets, then calls the existing `decay_spring_offsets` math for every 50 Hz action frame. Run authoritative FK for all three stages. Do not solve IK or alter source support.

- [ ] **Step 4: Run preview and composer regression tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_quality_preview \
  tests.python.test_sonic_torch_terrain_skill_composer
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_quality_preview.py \
  tests/python/test_sonic_torch_terrain_quality_preview.py
git commit -m "feat: decompose terrain transition quality"
```

---

### Task 7: Oracle CLI, contact sheets, and classification report

**Files:**
- Create: `resources/run_g1_torch_motion_quality_oracle.py`
- Create: `tests/python/test_run_g1_torch_motion_quality_oracle.py`
- Create: `sonic/python/mm_sonic/torch_terrain_quality_report.py`
- Test: `tests/python/test_sonic_torch_terrain_quality_report.py`

**Interfaces:**
- Consumes: dataset/config/G1 XML, frozen v0 artifact root, contact-oracle config, and output path.
- Produces: authenticated contact-quality summaries, frozen states, exhaustive rankings, stage previews, PNG contact sheets, and a per-state classification.

- [ ] **Step 1: Write failing CLI and classification tests**

Require these arguments:

```text
--dataset
--terrain-config
--oracle-config
--g1-xml
--baseline-artifacts
--output
--device
```

Patch each boundary and assert the CLI passes resolved identities unchanged. Classification tests require:

```python
self.assertEqual(classify_state(no_native), "corpus")
self.assertEqual(classify_state(native_only), "placement")
self.assertEqual(classify_state(placed_only), "composition")
self.assertEqual(classify_state(good_missed), "search")
self.assertEqual(classify_state(individual_only), "representation")
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_run_g1_torch_motion_quality_oracle \
  tests.python.test_sonic_torch_terrain_quality_report
```

Expected: import failures for the CLI and report module.

- [ ] **Step 3: Implement orchestration and atomic artifacts**

Resolve the terrain dataset once, build and mirror the existing contact-phase index once, reconstruct route support, analyze baseline contact quality, capture states, call the existing foothold planner for each state, rank every action, and build previews for the union of all ranked top-five sets. Save each state under `states/<state_id>/` with canonical JSON and compressed arrays. Exclude wall-clock and GPU timing from hashes.

- [ ] **Step 4: Implement deterministic classifications**

Define explicit thresholds from native-corpus quantiles saved in the report: native contact acceptance uses the corpus median plus two median absolute deviations for stance drift and boundary motion; terrain acceptance retains existing oracle limits. Classify in this order: `source`, `corpus`, `placement`, `composition`, `search`, `representation`, or `qualified`. Serialize the metrics that trigger every label.

- [ ] **Step 5: Render contact sheets from saved qpos**

For each ranked candidate, render native, placed, and composed stages at entry, quarter, half, three-quarter, and terminal frames using the G1 MuJoCo model and query stair geometry. Save one PNG with stage labels, action source identity, support mask, landing error, stance drift, and boundary jumps. Rendering reads saved qpos only and cannot alter oracle results.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all CLI/report tests pass.

- [ ] **Step 7: Commit**

```bash
git add resources/run_g1_torch_motion_quality_oracle.py \
  sonic/python/mm_sonic/torch_terrain_quality_report.py \
  tests/python/test_run_g1_torch_motion_quality_oracle.py \
  tests/python/test_sonic_torch_terrain_quality_report.py
git commit -m "feat: qualify terrain motion quality offline"
```

---

### Task 8: Real-data qualification and correction decision

**Files:**
- Create generated artifacts only: `build/terrain-motion-quality-oracle-v1/`
- Create: `docs/superpowers/results/2026-08-01-g1-terrain-motion-quality-oracle.md`

**Interfaces:**
- Consumes: the full 760-clip dataset, frozen six-route v0 artifacts, query stair, and G1 model.
- Produces: deterministic oracle evidence and exactly one recommended correction category.

- [ ] **Step 1: Run the oracle on one GPU without shortlist limits**

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_torch_motion_quality_oracle.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --terrain-config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --oracle-config sonic/configs/experiments/torch_grail_contact_oracle.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --baseline-artifacts build/multi-horizon-terrain-skills/qualified-v2-final \
  --output build/terrain-motion-quality-oracle-v1 \
  --device cuda:0
```

Expected: every captured state produces exhaustive counts, ranked sets or a structured coverage result, three-stage previews for ranked candidates, and a deterministic classification.

- [ ] **Step 2: Re-run and authenticate determinism**

Run the same command with output `build/terrain-motion-quality-oracle-v1-repeat`. Compare the top-level non-timing SHA-256 and every state SHA-256; all must match.

- [ ] **Step 3: Review contact sheets against metrics**

Reject the evaluator if a visibly floating, non-stepping, or grossly discontinuous candidate receives `qualified`. Add the smallest synthetic regression test for the missed condition before changing any threshold or classification rule, then regenerate both deterministic runs.

- [ ] **Step 4: Run the complete non-opt-in Python suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover -s tests/python -v
```

Expected: all non-opt-in tests pass; authenticated real-data tests remain skipped unless explicitly enabled.

- [ ] **Step 5: Publish the evidence report**

Record corpus/action counts, baseline truthful-contact metrics, per-state top candidates, native/placed/composed deltas, classifications, deterministic hashes, and one correction decision. The decision must follow the design rules: search, composition, targeted corpus, or representation. Do not implement that correction in this plan.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/results/2026-08-01-g1-terrain-motion-quality-oracle.md
git commit -m "docs: classify terrain motion quality failures"
```
