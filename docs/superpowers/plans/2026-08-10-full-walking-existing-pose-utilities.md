# Full-Walking Existing Pose Utilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-walking viewer's custom leg/arm transition thresholds with the repository's existing full-pose inertializer, terrain pose repair, and G1 foot lock.

**Architecture:** Keep `HybridMatcher` responsible only for row selection, canonical source placement, and safety. Add a display-only postprocessor that converts native qpos to `KinematicPose`, applies `PoseInertializer`, `G1TerrainPoseRepair`, and `G1TerrainFootLock`, then converts back to qpos. Wire it only into interactive fixed-rate callbacks using a separate collision-oracle model; formal evaluation remains unchanged.

**Tech Stack:** Python 3.11, NumPy, MuJoCo, pytest, existing `mm_sonic.hybrid_terrain_interactive`, `mm_sonic.terrain_pose_repair`, and `mm_sonic.terrain_foot_lock` utilities.

## Global Constraints

- Interactive diagnostic display only; formal smoke/evaluation stays CPU-exact and unfiltered.
- Retain all 9,758,524 full-corpus rows, diagnostic mechanical filtering, support-domain holds, and single-GPU search.
- Use the repository's `PoseInertializer` with exact half-life `0.10` seconds.
- Use corpus contact labels in exact left/right order for `G1TerrainFootLock`.
- No learned-pose fallback, joint clamp, custom leg gate, or custom arm slew in the new display path.
- A repair/lock rejection must not stop source row/root advancement.
- Do not stage or modify `.superpowers/sdd/task-1-report.md` or `.superpowers/sdd/task-2-report.md`.

---

## File Structure

- Create `sonic/python/mm_sonic/hybrid_terrain_lmm_postprocess.py`: qpos/KinematicPose adapter, existing-utility display pipeline, telemetry, reset.
- Create `tests/python/test_hybrid_terrain_lmm_postprocess.py`: focused adapter, inertialization, repair/lock, rollback, and reset tests.
- Modify `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py`: remove only the custom articulated/leg transition gate while retaining canonical diagnostic source poses.
- Modify `tests/python/test_hybrid_terrain_lmm_runtime.py`: replace custom-threshold assertions with raw canonical source-jump behavior.
- Modify `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py`: optional fixed-rate display postprocessor and separate collision-oracle model.
- Modify `sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py`: construct the existing-utility postprocessor only for `view`, expose honest identity/label fields.
- Modify `tests/python/test_hybrid_terrain_lmm_viewer.py` and `tests/python/test_full_walking_terrain_lmm_viewer.py`: wiring, formal isolation, collision model, receipt telemetry.

### Task 1: Remove the Custom Matcher Transition Thresholds

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py`
- Modify: `tests/python/test_hybrid_terrain_lmm_runtime.py`

**Interfaces:**
- Consumes: existing `HybridMatcher(diagnostic_stability=True, diagnostic_canonical_source_pose=True)` behavior.
- Produces: raw native-valid canonical qpos for every selected candidate; no `DIAGNOSTIC_MAX_SEARCH_JUMP_*` constant or `diagnostic_search_jump` commit argument.

- [ ] **Step 1: Replace the custom-gate test with a failing raw-source test**

Create a test whose nearest row converts to a canonical joint value above the old `0.5` threshold and assert that the exact nearest row commits with zero diagnostic pose rejections:

```python
def test_canonical_diagnostic_search_jump_preserves_raw_source_pose(self):
    values = np.full((8, 31), 10.0, dtype=np.float32)
    values[4] = 0.0
    values[:, 27:31] = 0.0
    corpus = _mechanically_safe(_corpus(values))
    corpus.artifacts.positions[:, 1, 0] = 0.0
    corpus.artifacts.positions[4, 1, 0] = 0.8
    matcher = HybridMatcher(
        corpus,
        _Generator(),
        TerrainAuthority.flat(),
        pose_converter=_pose_converter,
        diagnostic_stability=True,
        diagnostic_canonical_source_pose=True,
    )

    state = matcher.select_query(np.zeros(31, dtype=np.float64))

    assert state.row == 4
    assert state.qpos[7] == pytest.approx(0.8)
    assert state.diagnostic_pose_rejection_count == 0
    assert state.pose_source == "canonical-diagnostic"
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_runtime.py -k 'canonical_diagnostic_search_jump_preserves_raw_source_pose'
```

Expected: FAIL because the old search-jump threshold rejects row 4.

- [ ] **Step 3: Remove the custom shaping code**

Delete the custom search-jump constant, the `diagnostic_search_jump` parameter
from `_commit_row`, the articulated/leg qpos delta block, and its call-site
propagation. Keep canonical decoding as:

```python
decoded = canonical if self.diagnostic_canonical_source_pose else self._decode(
    row, seeded_features
)
```

Remove the uncommitted custom arm-slew code and restore pose source to:

```python
pose_source = (
    "canonical-diagnostic" if self.diagnostic_canonical_source_pose else "learned"
)
```

- [ ] **Step 4: Run GREEN and runtime regression**

Run:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_runtime.py
```

Expected: all runtime tests PASS.

- [ ] **Step 5: Commit Task 1 only**

```bash
git add sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py
git commit -m "fix: leave full walking display transitions to pose utilities"
```

### Task 2: Build the Existing-Utility Display Postprocessor

**Files:**
- Create: `sonic/python/mm_sonic/hybrid_terrain_lmm_postprocess.py`
- Create: `tests/python/test_hybrid_terrain_lmm_postprocess.py`

**Interfaces:**
- Consumes: native G1 `model`, scene object with `height_at_world_xy`, native qpos `[36]`, row/range IDs, source contacts `[2]`, and positive `dt_s`.
- Produces: `ExistingUtilityPosePostprocessor.step(qpos, *, row, range_index, source_contact, dt_s) -> np.ndarray`, `reset() -> None`, and `identity() -> dict[str, object]`.

- [ ] **Step 1: Write qpos/KinematicPose RED tests**

Add real-G1 tests that create `NativeQposPoseAdapter(model)`, round-trip one
finite native qpos through `to_pose(qpos, dt_s=1/60)` and `build_qpos`, and
assert exact joint order/root quaternion conventions. A second call asserts
finite joint/body velocities with shapes `(29,)`, `(30,3)`, and `(30,3)`.

```python
pose = adapter.to_pose(qpos, dt_s=1.0 / 60.0)
roundtrip = build_qpos(
    pose.root_position_world,
    pose.root_orientation_world_xyzw,
    pose.joint_position,
    model,
)
np.testing.assert_allclose(roundtrip, qpos, atol=1.0e-6, rtol=0.0)
assert pose.joint_velocity.shape == (29,)
assert pose.body_linear_velocity_world.shape == (30, 3)
assert pose.body_angular_velocity_world.shape == (30, 3)
```

- [ ] **Step 2: Write postprocessor RED tests**

Use injected recording repair/lock fakes while using the real `PoseInertializer`. Prove:

```python
# Search switch frame equals the previously displayed pose.
switched = processor.step(target_b, row=20, range_index=2,
                          source_contact=np.array([True, False]), dt_s=1/60)
np.testing.assert_allclose(switched, displayed_a, atol=1.0e-6)

# Successor and repeated-row neutral ticks decay toward target_b.
successor = processor.step(target_c, row=21, range_index=2,
                           source_contact=np.array([True, False]), dt_s=1/60)
neutral = processor.step(target_c, row=21, range_index=2,
                         source_contact=np.array([True, False]), dt_s=1/60)
assert np.linalg.norm(neutral - target_c) < np.linalg.norm(switched - target_b)

# Lock rejection restores its snapshot and publishes repaired source motion.
assert foot_locker.restore_calls == 1
assert not np.array_equal(rejected_lock_output, previous_display)
```

Also assert exact left/right contact forwarding, raw-repair fallback, raw-repair failure hold/recovery, and reset clearing the foot lock/inertializer.

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_postprocess.py
```

Expected: collection FAIL because the new module is absent.

- [ ] **Step 4: Implement `NativeQposPoseAdapter`**

Implement this exact public surface and the conversion algorithm below it:

```python
class NativeQposPoseAdapter:
    def __init__(self, model: object) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._previous_qpos = None
        self._previous_body_position = None
        self._previous_body_quaternion_wxyz = None

    def to_pose(self, qpos: object, *, dt_s: float) -> KinematicPose:
        """Validate qpos, run FK, derive velocities, and update prior samples."""
```

Use named `MUJOCO_JOINT_NAMES`, `BODY_NAMES`, and
`mujoco_to_isaaclab_joint_vector`; use `mj_differentiatePos` for joint velocity,
MuJoCo FK for body transforms, finite differences for body linear velocity, and
`terrain_oracle.math3d.angular_velocity_world_wxyz` on consecutive wxyz body
quaternions. First/reset velocities are zero.

- [ ] **Step 5: Implement `ExistingUtilityPosePostprocessor`**

Implement:

```python
class ExistingUtilityPosePostprocessor:
    def __init__(self, model, scene, *, fps=60.0,
                 inertialization_halflife_s=0.10,
                 pose_adapter=None, pose_repairer=None, foot_locker=None):
        """Build or accept the existing pose utilities and reset state."""

    def reset(self) -> None:
        """Clear adapter, inertializer, displayed pose, and foot-lock state."""

    def step(self, qpos, *, row, range_index, source_contact, dt_s) -> np.ndarray:
        """Return the non-freezing, utility-filtered display qpos."""

    def identity(self) -> dict[str, object]:
        """Return immutable policy values and current diagnostic counters."""
```

Transition detection is `row != previous_row`, excluding same-range
`row == previous_row + 1`. On a transition create
`PoseInertializer(previous_display_pose, raw_target, halflife_s=0.10)` and apply
at elapsed zero. On subsequent calls—including repeated-row neutral calls—advance
elapsed by `dt_s`. Apply repair, snapshot/apply lock, restore and bypass lock on
lock rejection, retry raw target on inertialized-repair rejection, and hold only
when raw repair also rejects. Track exact counters and last reason.

- [ ] **Step 6: Run GREEN and static checks**

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_postprocess.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  sonic/python/mm_sonic/hybrid_terrain_lmm_postprocess.py \
  tests/python/test_hybrid_terrain_lmm_postprocess.py
```

Expected: all postprocessor tests PASS; Ruff says `All checks passed!`.

- [ ] **Step 7: Commit Task 2 only**

```bash
git add sonic/python/mm_sonic/hybrid_terrain_lmm_postprocess.py \
  tests/python/test_hybrid_terrain_lmm_postprocess.py
git commit -m "feat: reuse existing terrain pose utilities"
```

### Task 3: Wire the Interactive Viewer and Collision Oracle

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py`
- Modify: `sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py`
- Modify: `tests/python/test_hybrid_terrain_lmm_viewer.py`
- Modify: `tests/python/test_full_walking_terrain_lmm_viewer.py`

**Interfaces:**
- Consumes: `ExistingUtilityPosePostprocessor` from Task 2 and raw canonical qpos from Task 1.
- Produces: optional `display_postprocessor_factory(model, matcher, terrain)` on `run_interactive`; full `view` supplies it, formal/smoke does not.

- [ ] **Step 1: Write viewer wiring RED tests**

Add tests that assert:

```python
# Callback order is matcher step -> display postprocessor -> viewer qpos.
assert events == ["matcher.step", "postprocessor.step", "mj_forward"]

# Full interactive factory is supplied; formal evaluator never supplies/builds it.
assert full_view_kwargs["display_postprocessor_factory"] is not None
assert formal_postprocessor_calls == []

# Reset calls matcher.reset then postprocessor.reset before the next display tick.
assert reset_events == ["matcher.reset", "postprocessor.reset"]
```

Add a real-model test that the diagnostic collision oracle has a
`terrain_hybrid_lmm_authoritative` geom with nonzero collision masks, while
`build_viewer_model(g1_xml, terrain, interactive_visuals=False)` retains the existing
non-colliding geom.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_full_walking_terrain_lmm_viewer.py \
  -k 'postprocessor or collision_oracle or existing_pose_utilities'
```

Expected: FAIL because viewer factory/collision-oracle APIs are absent.

- [ ] **Step 3: Add the collision-oracle model**

Extract mesh insertion into a helper and add:

```python
def build_diagnostic_collision_model(g1_xml: Path, terrain: SceneTerrainAdapter):
    spec = mujoco.MjSpec.from_file(str(g1_xml))
    vertices, faces = terrain.native_mesh()
    spec.add_mesh(name="terrain_hybrid_lmm_authoritative_mesh",
                  uservert=vertices.ravel(), userface=faces.ravel(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    spec.worldbody.add_geom(name="terrain_hybrid_lmm_authoritative",
                            type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname="terrain_hybrid_lmm_authoritative_mesh",
                            contype=1, conaffinity=1)
    return spec.compile()
```

Do not modify `build_viewer_model` default/formal geometry or masks.

- [ ] **Step 4: Add optional fixed-rate display filtering**

Extend `run_interactive` with keyword argument
`display_postprocessor_factory=None`. Construct the
postprocessor after the render model exists. In the accumulator callback:

```python
state = matcher.step(command, dt=dt)
display_qpos = (
    state.qpos
    if postprocessor is None
    else postprocessor.step(
        state.qpos,
        row=state.row,
        range_index=state.range_index,
        source_contact=np.asarray(matcher.artifacts.contacts[state.row], dtype=bool),
        dt_s=dt,
    )
)
```

Assign `data.qpos[:] = display_qpos`. On reset call `postprocessor.reset()` and
seed it on the next fixed-rate callback.

- [ ] **Step 5: Wire full interactive only and disclose identity**

In the full `view` branch, supply a factory that builds the separate collision
model, terrain-height shim, and `ExistingUtilityPosePostprocessor`. Remove the
uncommitted custom leg/arm constant imports, labels, and identity fields. Add:

```python
"diagnostic_display_postprocessor": "existing-pose-inertializer-repair-foot-lock/v1",
"inertialization_halflife_s": 0.10,
"pose_repair_count": postprocessor.pose_repair_count,
"foot_lock_accept_count": postprocessor.foot_lock_accept_count,
"foot_lock_bypass_count": postprocessor.foot_lock_bypass_count,
"raw_repair_failure_count": postprocessor.raw_repair_failure_count,
```

Keep the title diagnostic and include `EXISTING INERTIALIZER + TERRAIN FOOT LOCK`.

- [ ] **Step 6: Run GREEN and aggregate suites**

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_postprocess.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_full_walking_terrain_lmm_viewer.py
```

Expected: all tests PASS.

- [ ] **Step 7: Run static verification and commit**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
  sonic/python/mm_sonic/hybrid_terrain_lmm_postprocess.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
  sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py \
  tests/python/test_hybrid_terrain_lmm_postprocess.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_full_walking_terrain_lmm_viewer.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff format --check <same files>
git diff --check
```

Then commit exactly the viewer and viewer-test files; do not stage the two SDD
reports.

### Task 4: Independent Review and Bounded Live Route

**Files:**
- No production changes unless a test-first review fix is required.

**Interfaces:**
- Consumes: committed Task 1–3 integration.
- Produces: reviewed code plus one bounded diagnostic route result.

- [ ] **Step 1: Request independent code review**

Review specifically for: use of the real existing utilities, display-only/formal
isolation, contact ordering, collision model recognition, non-freezing fallbacks,
reset behavior, and absence of custom joint thresholds.

- [ ] **Step 2: Re-run affected aggregate after review fixes**

Run the exact four-suite command from Task 3 and all static checks. Expected:
all PASS with a clean scoped diff.

- [ ] **Step 3: Run one bounded 600-frame GPU5 ramp route**

Use the existing authenticated corpus/model and physical GPU 5 only. Record:
row/range/root advancement; source switches; learned fallback/clamp; display
repair/lock/bypass/failure counters; candidate exhaustion; max/p95 joint delta;
turning planted-foot slip; wall/search latency. Abort at 20 minutes.

Required readiness conditions:

```text
frames == 600
distinct_rows > 1
root_distance_m > 0
fallback_count == 0
joint_clamp_count == 0
raw_repair_failure_count == 0
no repeated display-filter deadlock
```

- [ ] **Step 4: Hand off the live viewer**

If the bounded route meets readiness conditions, launch one interactive viewer on
GPU5, verify the process owns only GPU5, and report its PID/log. Keep the title
diagnostic. If slip remains high, report the measured value without adding custom
smoothing.
