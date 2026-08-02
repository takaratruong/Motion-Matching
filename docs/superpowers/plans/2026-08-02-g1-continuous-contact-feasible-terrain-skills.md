# G1 Continuous Contact-Feasible Terrain Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make unchanged held commands continue coherent GRAIL terrain skills while rejecting landing, swing, and whole-body terrain collisions before playback.

**Architecture:** Add an every-frame heightmap feasibility module and a separate transactional MuJoCo collision preview. Extend an active zero-warp terrain skill sequentially when its command is unchanged and its next contact horizon passes both validators; otherwise run the existing global search through the same validation pipeline.

**Tech Stack:** Python 3, PyTorch, NumPy, MuJoCo Python bindings, `unittest`, existing `mm_sonic` terrain-skill and artifact infrastructure.

**Implementation note:** Interface sketches below define required behavior and diagnostics. During execution, adapt parameter names and tensor/result types to the repository's current public contracts; do not introduce duplicate dataset or matcher abstractions merely to match a sketch.

## Global Constraints

- Remain privileged-heightmap, 50 Hz, kinematic-only; do not add physics, Sonic tracking, or depth inference.
- Preserve initial-zero freeze and release-to-current-double-support action chunking.
- Continue only an exactly identical normalized nonzero velocity/heading tuple.
- Do not use a wall-clock cutoff for candidate feasibility; report latency instead.
- Hard-reject invalid candidates before soft ranking and never emit an unchecked fallback.
- Require at least 5 cm continuation progress and at most five moving-command stall frames.
- Fast bounds: 5 cm stance error, 3 cm landing error, 4 cm cross-footprint offsets, 2.5 cm footprint height range, -5 mm unsupported-sole clearance, and 6 cm landing-height deformation.
- Exact preview bound: no supported sole, unsupported foot, or forbidden robot-link terrain penetration deeper than 5 mm.
- Same-skill continuation requires zero endpoint translation/yaw warp; nonzero-warp experiments retain global search behavior.
- Preserve current multi-horizon defaults until the combined ablation qualifies.
- Do not modify or stage the existing uncommitted landing-bridge, contact-segment, or G1 FK work.

---

## File Structure

- Create `sonic/python/mm_sonic/torch_terrain_contact_feasibility.py`: immutable fast-feasibility types and every-frame heightmap validation.
- Create `sonic/python/mm_sonic/torch_terrain_collision_preview.py`: exact, transactional native-MuJoCo contact validation.
- Modify `sonic/python/mm_sonic/torch_terrain_foot_lock.py`: snapshot/restore seam for side-effect-free candidate preview.
- Modify `sonic/python/mm_sonic/torch_terrain_skill_horizons.py`: next sequential stable endpoint lookup.
- Modify `sonic/python/mm_sonic/torch_terrain_skill_composer.py`: extend an existing placed skill without restarting or re-inertializing it.
- Modify `sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py`: typed candidate-validation diagnostics and new feasible-candidate cost fields.
- Modify `sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py`: continuation-first selection and shared validation pipeline.
- Modify `sonic/python/mm_sonic/torch_terrain_live_viewer.py`: experimental flags, collision-preview construction, and overlay diagnostics.
- Modify `resources/run_g1_torch_multi_horizon_skills.py`: deterministic experiment flags and artifact diagnostics.
- Create `resources/run_g1_torch_continuation_ablation.py`: four-arm ablation runner.
- Create `sonic/configs/experiments/torch_grail_continuous_contact_feasible.json`: explicit experimental thresholds.
- Add focused tests under `tests/python/` matching each production module.
- Create `docs/superpowers/results/2026-08-02-g1-continuous-contact-feasible-terrain-skills.md` only after measured qualification.

---

### Task 1: Every-frame contact and landing feasibility

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_contact_feasibility.py`
- Create: `tests/python/test_sonic_torch_terrain_contact_feasibility.py`

**Interfaces:**
- Consumes: `TerrainSkill`, `TerrainDataset`, `TerrainFeatureExtension`, current root pose, entry frame, and exclusive endpoint.
- Produces: `TerrainContactFeasibilityConfig`, `TerrainContactFeasibilityResult`, and `validate_terrain_skill_contact_trace(...)`.

- [ ] **Step 1: Write failing configuration and result-contract tests**

```python
from mm_sonic.torch_terrain_contact_feasibility import (
    TerrainContactFeasibilityConfig,
    TerrainContactFeasibilityResult,
)

def test_contact_feasibility_defaults_are_frozen(self):
    value = TerrainContactFeasibilityConfig()
    self.assertEqual(value.sample_stride, 1)
    self.assertEqual(value.edge_margin_m, 0.04)
    self.assertEqual(value.maximum_edge_height_range_m, 0.025)
    self.assertEqual(value.minimum_swing_clearance_m, -0.005)

def test_result_requires_reason_exactly_when_rejected(self):
    with self.assertRaisesRegex(ContractError, "result"):
        TerrainContactFeasibilityResult(
            accepted=False,
            reason=None,
            maximum_stance_error_m=0.0,
            landing_error_m=0.0,
            minimum_swing_clearance_m=0.0,
            maximum_footprint_height_range_m=0.0,
        )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_contact_feasibility -v`

Expected: import failure because `torch_terrain_contact_feasibility` does not exist.

- [ ] **Step 3: Implement immutable configuration and result types**

```python
@dataclass(frozen=True)
class TerrainContactFeasibilityConfig:
    sample_stride: int = 1
    stance_height_tolerance_m: float = 0.05
    landing_height_tolerance_m: float = 0.03
    edge_margin_m: float = 0.04
    maximum_edge_height_range_m: float = 0.025
    minimum_swing_clearance_m: float = -0.005
    maximum_height_deformation_m: float = 0.06

@dataclass(frozen=True)
class TerrainContactFeasibilityResult:
    accepted: bool
    reason: str | None
    maximum_stance_error_m: float
    landing_error_m: float
    minimum_swing_clearance_m: float
    maximum_footprint_height_range_m: float
```

Validate exact types, finite values, positive tolerances, `sample_stride == 1`, and `accepted == (reason is None)` in `__post_init__`.

- [ ] **Step 4: Add synthetic failures for every feasibility layer**

Create a three-step height-grid fixture and a placed two-foot trace. Add one focused test per rejection reason in this order:

```python
EXPECTED_REASONS = (
    "terrain-domain",
    "stance-height",
    "landing-height",
    "landing-edge-margin",
    "swing-penetration",
    "height-deformation",
)
```

The edge fixture must put the landing center on the high tread while one `+0.04 m` footprint probe samples the lower surface. The swing fixture must keep the source foot unsupported while its placed sole is 6 mm below the sampled riser.

- [ ] **Step 5: Run the layer tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_contact_feasibility -v`

Expected: failures because `validate_terrain_skill_contact_trace` is missing.

- [ ] **Step 6: Implement every-frame placed-trace validation**

```python
def validate_terrain_skill_contact_trace(
    *,
    skill: TerrainSkill,
    canonical_entry_row: int,
    endpoint_frame_exclusive: int,
    dataset: TerrainDataset,
    database: TorchMotionDatabase,
    query_terrain: TerrainFeatureExtension,
    current_root_position_world: torch.Tensor,
    current_root_orientation_world_wxyz: torch.Tensor,
    config: TerrainContactFeasibilityConfig,
) -> TerrainContactFeasibilityResult:
    ...
```

Reuse the rigid yaw/translation placement from `terrain_skill_compatible`, but create `torch.arange(entry_frame, endpoint)` with no stride. Sample both ankle centers every frame. Compute sole clearance with `ANKLE_ORIGIN_SOLE_M`. Detect each landing as `~support[t-1] & support[t]`; sample center plus `(±margin, 0)` and `(0, ±margin)` for every landing. Evaluate constraints in `EXPECTED_REASONS` order and return measured extrema even on rejection.

- [ ] **Step 7: Verify the fast validator and neighboring terrain tests**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_contact_feasibility \
  tests.python.test_sonic_torch_terrain_skill_rollout \
  tests.python.test_sonic_torch_terrain_skill_horizons -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_terrain_contact_feasibility.py \
  tests/python/test_sonic_torch_terrain_contact_feasibility.py
git commit -m "feat: validate terrain skill contact traces"
```

---

### Task 2: Transactional foot cleanup and exact MuJoCo collision preview

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_foot_lock.py`
- Create: `sonic/python/mm_sonic/torch_terrain_collision_preview.py`
- Modify: `tests/python/test_sonic_torch_terrain_foot_lock.py`
- Create: `tests/python/test_sonic_torch_terrain_collision_preview.py`

**Interfaces:**
- Consumes: a sequence of uncommitted matcher-style results, per-frame support masks, the existing stateful foot filter, and a kinematic MuJoCo model/data pair.
- Produces: `TerrainFootLockSnapshot`, `TerrainCollisionPreviewConfig`, `TerrainCollisionPreviewResult`, and `MujocoTerrainCollisionPreview.validate(...)`.

- [ ] **Step 1: Write snapshot round-trip tests before changing the filter**

```python
def test_snapshot_restore_makes_preview_transactional(self):
    foot_lock = self.build_filter()
    baseline = foot_lock.snapshot()
    foot_lock.apply(self.moving_result(frame=10))
    foot_lock.restore(baseline)
    self.assertEqual(foot_lock.snapshot(), baseline)

def test_restore_rejects_snapshot_from_another_filter(self):
    with self.assertRaisesRegex(ValueError, "snapshot"):
        self.first.restore(self.second.snapshot())
```

- [ ] **Step 2: Run snapshot tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_foot_lock.TerrainFootLockSnapshotTests -v`

Expected: `AttributeError` for missing `snapshot`.

- [ ] **Step 3: Add exact snapshot/restore state**

```python
@dataclass(frozen=True)
class TerrainFootLockSnapshot:
    owner_token: object
    lock_position: torch.Tensor
    previous_support: torch.Tensor
    locked: torch.Tensor
    joint_offset: torch.Tensor
    previous_joint_position: torch.Tensor | None
    failure_count: int
```

Give every filter a private `self._snapshot_owner = object()`. `snapshot()` clones every mutable tensor. `restore()` verifies identity with `owner_token`, clones values back, and restores `None` exactly. Do not expose constructor configuration or use `deepcopy`.

- [ ] **Step 4: Write collision-preview tests with a minimal heightfield model**

Cover four exact cases:

```python
def test_supported_sole_contact_within_five_mm_passes(self): ...
def test_unsupported_sole_penetration_rejects(self): ...
def test_forbidden_shin_penetration_rejects_with_geom_name(self): ...
def test_preview_restores_filter_after_rejection(self): ...
```

The fixture must name allowed left/right sole geoms explicitly and include a named shin geom. Assert rejection reason, frame index, geom names, and penetration depth.

- [ ] **Step 5: Run collision tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_collision_preview -v`

Expected: import failure because the preview module does not exist.

- [ ] **Step 6: Implement exact preview without physics integration**

```python
@dataclass(frozen=True)
class TerrainCollisionPreviewConfig:
    maximum_forbidden_penetration_m: float = 0.005
    left_sole_geom_names: tuple[str, ...] = ("left_foot",)
    right_sole_geom_names: tuple[str, ...] = ("right_foot",)

class MujocoTerrainCollisionPreview:
    def validate(
        self,
        *,
        results: Sequence[object],
        support_mask: torch.Tensor,
        result_filter: TerrainFootLockFilter,
    ) -> TerrainCollisionPreviewResult:
        ...
```

For each frame, call `result_filter.apply`, map the corrected target state through `target_state_qpos`, assign `data.qpos`, and call `mujoco.mj_forward` only. Inspect `data.contact[:data.ncon]`; identify the terrain geom and robot geom by model geom IDs/names. Permit configured sole contact only when that foot is supported. Reject contact distance `< -0.005`. Wrap the entire sequence in `snapshot = result_filter.snapshot()` and `finally: result_filter.restore(snapshot)`.

- [ ] **Step 7: Verify preview and foot-lock suites**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_collision_preview \
  tests.python.test_sonic_torch_terrain_foot_lock -v
```

Expected: all tests pass and no MuJoCo warnings.

- [ ] **Step 8: Commit Task 2**

```bash
git add sonic/python/mm_sonic/torch_terrain_foot_lock.py \
  sonic/python/mm_sonic/torch_terrain_collision_preview.py \
  tests/python/test_sonic_torch_terrain_foot_lock.py \
  tests/python/test_sonic_torch_terrain_collision_preview.py
git commit -m "feat: preview terrain collisions transactionally"
```

---

### Task 3: Extend a coherent skill without a boundary transition

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizons.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_composer.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_horizons.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_composer.py`

**Interfaces:**
- Consumes: the completed `TerrainSkillState`, its support mask, and its skill interval.
- Produces: `next_sequential_horizon_endpoint(...)` and `extend_skill_state(...)`.

- [ ] **Step 1: Write endpoint-selection tests**

```python
def test_next_endpoint_is_after_current_exclusive_endpoint(self):
    support = self.support_with_double_support_at(24, 49, 74)
    self.assertEqual(
        next_sequential_horizon_endpoint(
            support,
            current_endpoint_frame_exclusive=25,
            playback_stop=75,
        ),
        50,
    )

def test_no_later_stable_endpoint_returns_none(self): ...
```

Require the next endpoint to be the first double-support frame at least 25 frames after the current endpoint, with the existing 25-frame lateness window, and never beyond `playback_stop`.

- [ ] **Step 2: Run endpoint tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_skill_horizons.SequentialEndpointTests -v`

Expected: import/name failure for `next_sequential_horizon_endpoint`.

- [ ] **Step 3: Implement sequential endpoint lookup**

```python
def next_sequential_horizon_endpoint(
    support_mask: torch.Tensor,
    *,
    current_endpoint_frame_exclusive: int,
    playback_stop: int,
    target_frames: int = 25,
) -> int | None:
    first = current_endpoint_frame_exclusive + target_frames - 1
    final = min(playback_stop - 1, first + MAXIMUM_ENDPOINT_LATENESS_FRAMES)
    for frame in range(first, final + 1):
        if bool(support_mask[frame].all().item()):
            return frame + 1
    return None
```

Apply the same strict tensor and bound validation used by `stable_horizon_endpoints`.

- [ ] **Step 4: Write state-extension tests**

```python
def test_extension_preserves_placement_and_offsets(self):
    completed = replace(self.state, next_source_frame=50, playback_stop=50)
    extended = extend_skill_state(completed, playback_stop=75)
    self.assertEqual(extended.next_source_frame, 50)
    self.assertEqual(extended.playback_stop, 75)
    self.assertIs(extended.offsets, completed.offsets)
    self.assertIs(extended.translation_world, completed.translation_world)

def test_extension_rejects_nonzero_endpoint_warp(self): ...
def test_extended_frames_are_exactly_50_through_74(self): ...
```

- [ ] **Step 5: Run composer tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_skill_composer.TerrainSkillExtensionTests -v`

Expected: import/name failure for `extend_skill_state`.

- [ ] **Step 6: Implement zero-warp state extension**

```python
def extend_skill_state(
    state: TerrainSkillState, *, playback_stop: int
) -> TerrainSkillState:
    if state.next_source_frame != state.playback_stop:
        raise ContractError("terrain skill extension requires a completed chunk")
    if bool(state.endpoint_translation_warp_world_xy.abs().max().item()) or bool(
        state.endpoint_yaw_warp_rad.abs().item()
    ):
        raise ContractError("terrain skill extension requires zero endpoint warp")
    if not state.playback_stop < playback_stop <= state.skill.interval.playback_stop:
        raise ContractError("terrain skill extension endpoint is invalid")
    if not bool(state.skill.support_mask[playback_stop - 1].all().item()):
        raise ContractError("terrain skill extension endpoint must be double support")
    return replace(state, playback_stop=playback_stop)
```

- [ ] **Step 7: Verify horizon/composer suites**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_skill_horizons \
  tests.python.test_sonic_torch_terrain_skill_composer -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_horizons.py \
  sonic/python/mm_sonic/torch_terrain_skill_composer.py \
  tests/python/test_sonic_torch_terrain_skill_horizons.py \
  tests/python/test_sonic_torch_terrain_skill_composer.py
git commit -m "feat: extend coherent terrain skill playback"
```

---

### Task 4: Continuation-first matcher and typed candidate validation

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_horizon_search.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py`

**Interfaces:**
- Consumes: Tasks 1–3 interfaces plus a collision-preview callable.
- Produces: `TerrainCandidateValidation`, `TerrainContinuationEvent`, and continuation-first behavior in `TerrainSkillHorizonMatcher`.

- [ ] **Step 1: Write typed validation and rejection-diagnostic tests**

```python
@dataclass(frozen=True)
class TerrainCandidateValidation:
    accepted: bool
    reason: str | None
    contact: TerrainContactFeasibilityResult
    collision: TerrainCollisionPreviewResult | None
    footprint_cost: float
    clearance_cost: float
    double_support_cost: float

def test_selector_counts_each_typed_rejection_reason(self):
    validator = SequenceValidator(("landing-edge-margin", "forbidden-link", None))
    selected = select_horizon_candidate(..., candidate_validator=validator)
    self.assertEqual(selected.rejected_by_reason["landing-edge-margin"], 1)
    self.assertEqual(selected.rejected_by_reason["forbidden-link"], 1)
```

- [ ] **Step 2: Run search tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_skill_horizon_search -v`

Expected: failure because the selector accepts only a boolean `terrain_validator`.

- [ ] **Step 3: Replace boolean acceptance with typed validation**

Change `select_horizon_candidate` to accept:

```python
CandidateValidator = Callable[[int, int, int], TerrainCandidateValidation]
```

Keep ranking device-resident. Iterate ranked records on CPU exactly as today, call the validator, add its three soft costs to the candidate's reported total, and continue on rejection. Preserve deterministic source-row order when totals tie. Serialize reason counts in insertion order after the existing `turn/progress/surface/local/nonfinite` counts.

- [ ] **Step 4: Write continuation behavior tests with a fake validator**

Cover these cases separately:

```python
def test_same_nonzero_command_extends_current_skill_before_global_search(self): ...
def test_changed_nonzero_command_uses_global_search(self): ...
def test_zero_command_does_not_extend(self): ...
def test_invalid_continuation_falls_back_to_global_search(self): ...
def test_no_feasible_fallback_holds_completed_safe_endpoint(self): ...
```

The first test must assert identical skill object, placement tensors, and sequential source frame. It must also assert that the mocked global selector was never called.

- [ ] **Step 5: Run rollout tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_skill_horizon_rollout.ContinuationFirstMatcherTests -v`

Expected: failures because completed chunks always call global selection.

- [ ] **Step 6: Implement continuation-first selection**

Add constructor arguments:

```python
continuous_skill_enabled: bool = False
candidate_validator: Callable[..., TerrainCandidateValidation] | None = None
maximum_continuation_stall_frames: int = 5
minimum_continuation_progress_m: float = 0.05
```

At a completed skill boundary, compare the exact incoming command with `_last_command`. When enabled, unchanged, nonzero, and zero-warp, build the next endpoint, compute source progress/stall from the horizon inventory helpers, validate the placed continuation, and call `extend_skill_state`. Do not call `start_skill`, reset the result filter, or append a normal transition event.

On success append:

```python
@dataclass(frozen=True)
class TerrainContinuationEvent:
    skill_index: int
    start_frame: int
    endpoint_frame_exclusive: int
    command: tuple[tuple[float, float], float]
    validation: TerrainCandidateValidation
```

On failure record the reason and proceed through global typed validation. If global validation exhausts candidates, retain the completed state and emit the existing hold result plus structured failure; never advance beyond the endpoint.

- [ ] **Step 7: Run search, rollout, and existing frozen unit suites**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_skill_horizon_search \
  tests.python.test_sonic_torch_terrain_skill_horizon_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer -v
```

Expected: all tests pass with continuation disabled by default.

- [ ] **Step 8: Commit Task 4**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py \
  sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py \
  tests/python/test_sonic_torch_terrain_skill_horizon_search.py \
  tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py
git commit -m "feat: continue feasible terrain skills"
```

---

### Task 5: Experimental configuration, viewer wiring, and artifacts

**Files:**
- Create: `sonic/configs/experiments/torch_grail_continuous_contact_feasible.json`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `resources/run_g1_torch_multi_horizon_skills.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`
- Modify: `tests/python/test_run_g1_torch_multi_horizon_skills.py`

**Interfaces:**
- Consumes: combined validator and continuation matcher from Task 4.
- Produces: `--continuous-skill`, `--contact-feasibility`, and `--exact-collision-preview` options plus deterministic saved diagnostics.

- [ ] **Step 1: Write parser and invalid-combination tests**

```python
def test_combined_experiment_flags_are_explicit(self):
    args = build_live_viewer_argument_parser().parse_args([
        "--dataset", "motions", "--config", "experiment.json",
        "--g1-xml", "g1.xml", "--multi-horizon", "--foot-lock",
        "--continuous-skill", "--contact-feasibility",
        "--exact-collision-preview",
    ])
    self.assertTrue(args.continuous_skill)
    self.assertTrue(args.contact_feasibility)
    self.assertTrue(args.exact_collision_preview)

def test_exact_preview_requires_contact_feasibility_and_foot_lock(self): ...
def test_continuation_rejects_nonzero_endpoint_warp_flags(self): ...
```

- [ ] **Step 2: Run parser tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_sonic_torch_terrain_live_viewer.LiveMujocoSceneTests -v`

Expected: unrecognized argument failures.

- [ ] **Step 3: Add the explicit experimental configuration**

Copy the retained `torch_grail_multi_horizon_skills.json` values and add:

```json
"continuous_contact_feasibility": {
  "minimum_continuation_progress_m": 0.05,
  "maximum_continuation_stall_frames": 5,
  "stance_height_tolerance_m": 0.05,
  "landing_height_tolerance_m": 0.03,
  "edge_margin_m": 0.04,
  "maximum_edge_height_range_m": 0.025,
  "minimum_swing_clearance_m": -0.005,
  "maximum_height_deformation_m": 0.06,
  "maximum_forbidden_penetration_m": 0.005
}
```

Add a strict parser that requires exactly these keys and returns the Task 1/2 configs. Do not change the retained baseline JSON.

- [ ] **Step 4: Wire the viewer and runner**

Construct the fast validator only with `--contact-feasibility`. Construct `MujocoTerrainCollisionPreview` only with `--exact-collision-preview`, using the same kinematic scene XML as the viewer/runner. Pass the validator and `continuous_skill_enabled` to `TerrainSkillHorizonMatcher`. Keep all new flags false by default.

Add overlay rows:

```text
continuation=<count> transition=<count> validated=<count>
reject=<reason-or-none> edge=<m> swing=<m> collision=<m>
```

- [ ] **Step 5: Save deterministic validation and continuation events**

Extend `chunk-events.json` or add `candidate-events.json` beside each route array. Hash all command identities, source bounds, accepted/rejected reason, cost components, contact metrics, collision geom names, frame index, and penetration. Continue excluding timing values from deterministic hashes.

- [ ] **Step 6: Run viewer/runner tests and verify GREEN**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_live_viewer \
  tests.python.test_run_g1_torch_multi_horizon_skills -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add sonic/configs/experiments/torch_grail_continuous_contact_feasible.json \
  sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  resources/run_g1_torch_multi_horizon_skills.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py \
  tests/python/test_run_g1_torch_multi_horizon_skills.py
git commit -m "feat: expose contact-feasible terrain continuation"
```

---

### Task 6: Four-arm deterministic ablation runner

**Files:**
- Create: `resources/run_g1_torch_continuation_ablation.py`
- Create: `tests/python/test_run_g1_torch_continuation_ablation.py`

**Interfaces:**
- Consumes: the existing multi-horizon runner entry point and Task 5 flags.
- Produces: four isolated artifact roots and one canonical comparison JSON.

- [ ] **Step 1: Write arm-construction and aggregation tests**

```python
EXPECTED_ARMS = {
    "baseline": (False, False, False),
    "continuation-only": (True, False, False),
    "feasibility-only": (False, True, True),
    "combined": (True, True, True),
}

def test_ablation_arms_are_exact_and_ordered(self):
    self.assertEqual(dict(ablation_arms()), EXPECTED_ARMS)

def test_report_rejects_mixed_dataset_or_route_identity(self): ...
def test_report_excludes_timing_from_deterministic_hash(self): ...
```

- [ ] **Step 2: Run ablation tests and verify RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_run_g1_torch_continuation_ablation -v`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement the runner and canonical report**

The CLI accepts the normal dataset/config/XML/device/output arguments and repeatable `--route`. Run each arm into `<output>/<arm>`, refuse existing destinations, and write `<output>/comparison.json` containing:

```json
{
  "schema": "g1-continuous-contact-feasible-ablation/v1",
  "arms": {},
  "route_order": [],
  "metrics": {
    "behavioral_passes": {},
    "clip_transitions": {},
    "continuations": {},
    "longest_moving_stall_frames": {},
    "maximum_forbidden_penetration_m": {},
    "maximum_landing_footprint_range_m": {},
    "stance_slide_m": {}
  },
  "deterministic_sha256": "..."
}
```

Use atomic staging/rename and canonical JSON encoding already used by the multi-horizon runner.

- [ ] **Step 4: Verify runner tests**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_run_g1_torch_continuation_ablation -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add resources/run_g1_torch_continuation_ablation.py \
  tests/python/test_run_g1_torch_continuation_ablation.py
git commit -m "feat: compare terrain continuation ablations"
```

---

### Task 7: Qualification, visual feedback build, and result report

**Files:**
- Create after measurement: `docs/superpowers/results/2026-08-02-g1-continuous-contact-feasible-terrain-skills.md`
- Modify only if a measured defect requires it: files from Tasks 1–6 with a new RED/GREEN cycle.

**Interfaces:**
- Consumes: four-arm runner and live viewer.
- Produces: qualification artifacts, contact sheets, a reset live viewer, and a measured recommendation.

- [ ] **Step 1: Run the held-W ascent smoke ablation first**

Add a route fixture containing one uninterrupted 300-frame forward command at the resolved viewer speed. Run all four arms on separate GPUs. Reject the combined build immediately if it cannot reach the upper landing, exceeds five stall frames, or reports any forbidden penetration deeper than 5 mm.

Run pattern:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_torch_continuation_ablation.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_continuous_contact_feasible.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:0 --route held-forward-ascent \
  --output build/continuous-contact-feasible/held-forward-v1
```

- [ ] **Step 2: Run the frozen six-route ablation**

Run the same command with `--qualification-slice` into a new immutable output. Require combined 6/6, no upper-side-exit regression, maximum forbidden penetration ≤ 0.005 m, landing footprint range ≤ 0.025 m, and moving stall ≤ five frames.

- [ ] **Step 3: Run the broad 21-route ablation**

Run without a route filter, distributed over available GPUs if needed. Merge only deterministic per-route outputs. Require at least 16/21 combined behavioral passes and report every exchanged pass/failure against baseline; do not summarize only the aggregate.

- [ ] **Step 4: Render and inspect mandatory contact sheets**

Render held ascent, release/restart, diagonal up/down, cross-tread, both upper side exits, riser reversal, 90-degree turn, side approach, and edge-parallel walking. Inspect exact qpos at approach, each landing, every continuation boundary, maximum penetration frame, and final state. Record visual failures even when numerical gates pass.

- [ ] **Step 5: Launch the combined live viewer and reproduce operator controls**

Launch with all retained foot cleanup flags plus:

```text
--continuous-skill --contact-feasibility --exact-collision-preview
```

Verify manually and by screenshots:

1. Backspace reset stays at the flat start.
2. One short W tap finishes one safe action chunk.
3. Held W crosses all risers without stopping at every double-support boundary.
4. Release on a tread settles once and remains stable.
5. A/D side mount and upper side exit do not collide with the stair edge.

- [ ] **Step 6: Run the full neighboring regression suite**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest discover \
  -s tests/python -p 'test_sonic_torch_terrain*.py'
```

Expected: zero failures. Record the exact run/pass/skip counts.

- [ ] **Step 7: Write the measured result report**

The report must include exact commands, commit, dataset/config identities, all four arm hashes, per-route metrics, rejection counts, collision links/depths, transition/continuation counts, screenshots/contact sheets, latency distributions, regressions, and an explicit classification:

- `qualified-default-candidate` only if every acceptance gate and visual review passes;
- `continuation-only-useful` if cadence improves but feasibility loses coverage;
- `feasibility-only-useful` if collisions improve but cadence does not;
- `contact-warp-required` if feasible coherent continuation cannot retain 16/21; or
- `rejected` otherwise.

- [ ] **Step 8: Verify, commit, and push only owned files**

Run `git diff --check`, the focused tests for every modified module, and the full terrain discovery command. Stage only files named in this plan and the result document. Preserve all unrelated dirty files.

```bash
git add docs/superpowers/results/2026-08-02-g1-continuous-contact-feasible-terrain-skills.md
git commit -m "docs: qualify contact-feasible terrain continuation"
git push checkpoint research/g1-torch-terrain-kinematics
```

Confirm the remote branch SHA equals local `HEAD` before reporting completion.
