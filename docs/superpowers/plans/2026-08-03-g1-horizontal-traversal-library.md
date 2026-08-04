# G1 Horizontal Traversal Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an offline library of complete G1 flat–mount–cross–descend–flat trajectories for every 10 cm horizontal line across one authenticated staircase in both directions.

**Architecture:** Treat complete accepted GRAIL curb and stair clips as whole-route anchors and use them unchanged whenever possible. When a complete route requires a splice, use the pinned ARDY G1 model only for the short constrained in-between window, then apply bounded contact repair and independent full-sole validation. Save every non-dominated valid route and explicit failures; do not build an online motion matcher.

**Tech Stack:** Python 3, NumPy, PyTorch, MuJoCo G1 FK/sole kinematics, existing `TerrainDataset`, `ContactSegmentIndex`, stair-frame utilities, and ARDY G1.

## Global Constraints

- The global stair frame uses `u` for ascent and `v` for horizontal travel.
- Grid spacing is exactly `0.10 m`; evaluate both `+v` and `-v`.
- Every accepted route contains flat approach, terrain-action mount, tread crossing, terrain-action descent, and flat departure.
- Mount/descent seeds may come from accepted GRAIL curb or staircase clips.
- Online motion matching, Sonic tracking, physics control, depth, and realtime latency are out of scope.
- Pin ARDY upstream to `693f74d13b3d04a0a22ce127ee79c929dd89756b`; do not modify its source.
- ARDY may generate splice windows only. GRAIL determines footholds, contact order, and route phases.
- Full-sole penetration must be at least `-0.025 m`; planted terminal sole error must be at most `0.025 m`.
- Failed lines remain explicit failures; constraints are never relaxed to manufacture coverage.
- Preserve unrelated working-tree changes.

---

### Task 1: Horizontal grid and library artifact schema

**Files:**
- Create: `sonic/python/mm_sonic/torch_horizontal_traversal_library.py`
- Create: `tests/python/test_sonic_torch_horizontal_traversal_library.py`

**Interfaces:**
- Consumes: `StairFrame` from `mm_sonic.torch_terrain_omni_routes`.
- Produces:
  - `HorizontalGridLine(line_index: int, stair_u_m: float, direction: int)`
  - `TraversalArtifact(line_id: str, route_sha256: str, route_path: str, source_family: str, source_clip: str, phase_boundaries: tuple[int, ...], minimum_sole_clearance_m: float, maximum_planted_error_m: float, maximum_joint_speed_rad_s: float, maximum_joint_acceleration_rad_s2: float)`
  - `TraversalFailure(line_id: str, failed_phase: str, rejection_histogram: tuple[tuple[str, int], ...])`
  - `HorizontalTraversalLibrary(dataset_manifest_sha256: str, stair_frame: StairFrame, spacing_m: float, artifacts: tuple[TraversalArtifact, ...], failures: tuple[TraversalFailure, ...])`
  - `horizontal_grid_lines(frame: StairFrame, spacing_m: float) -> tuple[HorizontalGridLine, ...]`
  - `horizontal_library_from_dict(payload: object) -> HorizontalTraversalLibrary`

- [ ] **Step 1: Write failing immutable-schema tests**

```python
def test_horizontal_grid_uses_ten_centimeter_offsets_in_both_directions():
    frame = StairFrame((0.0, 0.0), 0.0, 0.6223, 0.3302, 0.1778, 3)
    lines = horizontal_grid_lines(frame, spacing_m=0.10)
    assert {line.direction for line in lines} == {-1, 1}
    assert [line.stair_u_m for line in lines if line.direction == 1] == [
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
    ]

def test_library_rejects_success_and_failure_for_same_line():
    with pytest.raises(ContractError, match="duplicate outcome"):
        HorizontalTraversalLibrary(
            dataset_manifest_sha256="a" * 64,
            stair_frame=frame,
            spacing_m=0.10,
            artifacts=(artifact,),
            failures=(failure_for_same_line,),
        )
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_sonic_torch_horizontal_traversal_library
```

Expected: FAIL because `torch_horizontal_traversal_library` does not exist.

- [ ] **Step 3: Implement the schema and deterministic identities**

```python
@dataclass(frozen=True, order=True)
class HorizontalGridLine:
    line_index: int
    stair_u_m: float
    direction: int

    def __post_init__(self) -> None:
        if (
            type(self.line_index) is not int
            or self.line_index < 0
            or not math.isfinite(float(self.stair_u_m))
            or self.direction not in (-1, 1)
        ):
            raise ContractError("horizontal grid line is invalid")

    @property
    def line_id(self) -> str:
        return _canonical_sha256(
            {"schema": "g1-horizontal-grid-line/v1", **asdict(self)}
        )


def horizontal_grid_lines(
    frame: StairFrame, *, spacing_m: float = 0.10
) -> tuple[HorizontalGridLine, ...]:
    length = frame.tread_depth_m * frame.tread_count
    centers = np.arange(spacing_m / 2.0, length, spacing_m)
    return tuple(
        HorizontalGridLine(index, float(u), direction)
        for index, u in enumerate(centers)
        for direction in (-1, 1)
    )
```

`TraversalArtifact` must record `line_id`, direction, NPZ path, deterministic
motion SHA, source clip/family, five phase boundaries, minimum sole clearance,
maximum planted error, maximum joint speed, and maximum joint acceleration.
`TraversalFailure` must record the failed phase and rejection histogram.
Round-trip JSON parsing must recompute and authenticate every identity.

- [ ] **Step 4: Run schema tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_horizontal_traversal_library.py \
  tests/python/test_sonic_torch_horizontal_traversal_library.py
git commit -m "feat: add horizontal traversal library schema"
```

---

### Task 2: Whole-route curb and stair warm-start inventory

**Files:**
- Create: `sonic/python/mm_sonic/torch_horizontal_traversal_sources.py`
- Create: `tests/python/test_sonic_torch_horizontal_traversal_sources.py`

**Interfaces:**
- Consumes:
  - `TerrainDataset`
  - `source_support_mask(dataset, clip_index) -> torch.Tensor`
- Produces:
  - `HorizontalRouteSeed`
  - `horizontal_route_seeds(dataset: TerrainDataset, minimum_elevation_m: float = 0.08) -> tuple[HorizontalRouteSeed, ...]`

- [ ] **Step 1: Write failing phase-classification tests**

```python
def test_complete_height_profile_has_five_ordered_phases():
    support_height = np.array([
        0.0, 0.0, 0.0, 0.18, 0.18, 0.18, 0.18, 0.0, 0.0, 0.0
    ])
    phases = complete_route_phases(
        support_height,
        minimum_elevation_m=0.08,
        minimum_flat_frames=2,
    )
    assert phases == RoutePhases(0, 3, 4, 7, 8, 10)

def test_incomplete_clip_without_descent_is_not_a_route_seed():
    assert complete_route_phases(
        np.array([0.0, 0.0, 0.18, 0.18, 0.18]),
        minimum_elevation_m=0.08,
        minimum_flat_frames=2,
    ) is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_sonic_torch_horizontal_traversal_sources
```

Expected: FAIL because the source inventory module is missing.

- [ ] **Step 3: Implement whole-clip seed characterization**

```python
@dataclass(frozen=True)
class HorizontalRouteSeed:
    clip_index: int
    relative_path: str
    family: Literal["curb", "stair"]
    phases: RoutePhases
    source_elevation_m: float
    root_progress_m: float
    support_mask: np.ndarray


def _family(descriptor: dict) -> str | None:
    name = str(descriptor.get("logical_name", ""))
    if name.startswith("grail-curb-"):
        return "curb"
    if name.startswith(("grail-stair-", "grail-stair_p2-")):
        return "stair"
    return None
```

For every accepted terrain clip, sample its paired authenticated height grid
under both feet, derive support with `source_support_mask`, reduce supported
foot heights to a robust per-frame surface level, and retain only clips with:

- at least two flat frames before mount and after descent;
- one elevated interval of at least `0.08 m`;
- a return to the initial surface within `0.025 m`;
- at least `0.40 m` root travel;
- finite 50 Hz motion arrays.

Sort seeds by family, elevation, root progress, and source identity. Emit a
small JSON summary so the first real run reports how many complete curb and
stair clips actually exist before optimization begins.

- [ ] **Step 4: Run source tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Run the real inventory once**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B - <<'PY'
from mm_sonic.torch_terrain_features import TerrainDataset
from mm_sonic.torch_horizontal_traversal_sources import horizontal_route_seeds
d = TerrainDataset.load("build/torch-grail-terrain-full-v1", device="cpu")
s = horizontal_route_seeds(d)
print({"total": len(s), "curb": sum(x.family == "curb" for x in s),
       "stair": sum(x.family == "stair" for x in s)})
assert s
PY
```

Expected: a nonzero deterministic seed count. If zero, stop with the measured
phase rejection histogram; do not proceed with invented geometric routes.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_horizontal_traversal_sources.py \
  tests/python/test_sonic_torch_horizontal_traversal_sources.py
git commit -m "feat: inventory complete terrain traversal seeds"
```

---

### Task 3: Constrained ARDY in-betweening and contact repair

**Files:**
- Create: `sonic/python/mm_sonic/torch_ardy_inbetween.py`
- Create: `resources/run_g1_ardy_inbetween_worker.py`
- Create: `tests/python/test_sonic_torch_ardy_inbetween.py`
- Create: `tests/python/test_run_g1_ardy_inbetween_worker.py`

**Interfaces:**
- Consumes `HorizontalRouteSeed` plus adjacent GRAIL full-body, root, feet,
  and contact-boundary anchor states.
- Produces:
  - `ArdyInbetweenRequest`
  - `ArdyInbetweenResult`
  - `generate_ardy_inbetween(request, worker_python, worker_script)`
  - `repair_and_validate_inbetween(result, support_mask, sole_targets, kinematics, sample_surface_height_m)`

- [ ] **Step 1: Write failing bridge and anchor-preservation tests**

```python
def test_request_downsamples_only_transition_to_ardy_25hz():
    request = ArdyInbetweenRequest.from_grail_50hz(
        leading_anchor=leading_50hz,
        trailing_anchor=trailing_50hz,
        transition_frame_count_50hz=40,
        root_waypoints_world=root_waypoints,
        foot_constraints_world=foot_constraints,
    )
    assert request.ardy_frame_count_25hz == 20

def test_contact_repair_preserves_grail_boundary_frames_exactly():
    repaired = repair_and_validate_inbetween(
        generated_result,
        prescribed_support_mask=support,
        prescribed_sole_pose_world=sole_targets,
        kinematics=fake_bounded_kinematics,
        sample_surface_height_m=fake_surface_sampler,
    )
    np.testing.assert_array_equal(repaired.joint_position[0], leading_joint)
    np.testing.assert_array_equal(repaired.joint_position[-1], trailing_joint)
```

- [ ] **Step 2: Run tests and verify the missing adapter failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_sonic_torch_ardy_inbetween \
tests.python.test_run_g1_ardy_inbetween_worker
```

Expected: FAIL because the ARDY adapter and worker are missing.

- [ ] **Step 3: Install and verify official pinned ARDY**

```bash
git clone https://github.com/nv-tlabs/ardy.git /home/ubuntu/projects/ardy
git -C /home/ubuntu/projects/ardy checkout \
  693f74d13b3d04a0a22ce127ee79c929dd89756b
python3.11 -m venv /home/ubuntu/projects/ardy/.venv
/home/ubuntu/projects/ardy/.venv/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu126
/home/ubuntu/projects/ardy/.venv/bin/pip install -e \
  "/home/ubuntu/projects/ardy[demo]"
test "$(git -C /home/ubuntu/projects/ardy rev-parse HEAD)" = \
  693f74d13b3d04a0a22ce127ee79c929dd89756b
```

Expected: `import ardy` succeeds from the pinned unmodified checkout. Missing
checkpoint credentials are reported as an external blocker; do not substitute
an unconstrained interpolator while calling it ARDY.

- [ ] **Step 4: Implement the isolated ARDY worker**

The worker accepts one authenticated NPZ request and emits one NPZ result. It
loads the released G1 Horizon-52 checkpoint and supplies full-body leading and
trailing keyframes, dense root waypoints, and constrained feet at prescribed
contact frames. It uses a deterministic seed and makes no text-driven route
decisions.

- [ ] **Step 5: Implement 50 Hz reconstruction and contact repair**

Resample generated frames from 25 Hz to 50 Hz, restore both exact GRAIL
boundary frames, then hold every prescribed planted seven-point sole pose with
bounded leg IK. Reject unreachable projection, changed contact order, a
boundary mismatch, or any full-sole/speed/acceleration violation.

- [ ] **Step 6: Run tests and one real canary**

Run the Step 2 command, then generate one constrained one-second G1 in-between
between compatible GRAIL flat boundaries.

Expected: tests pass, both anchors are byte-identical, and the real canary
passes independent G1 full-sole validation.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/torch_ardy_inbetween.py \
  resources/run_g1_ardy_inbetween_worker.py \
  tests/python/test_sonic_torch_ardy_inbetween.py \
  tests/python/test_run_g1_ardy_inbetween_worker.py
git commit -m "feat: add constrained ARDY G1 in-betweening"
```

---

### Deferred fallback: Full-route projected trajectory optimization

Do not execute this fallback unless the pinned ARDY G1 checkpoint is genuinely
blocked or its constrained canary fails terrain validation. The hybrid path
above is the selected implementation.

**Files:**
- Create: `sonic/python/mm_sonic/torch_horizontal_trajectory_optimizer.py`
- Create: `tests/python/test_sonic_torch_horizontal_trajectory_optimizer.py`

**Interfaces:**
- Consumes:
  - `HorizontalGridLine`
  - `HorizontalRouteSeed`
  - `BoundedFootKinematics`
  - query staircase height sampler
- Produces:
  - `OptimizedHorizontalTrajectory`
  - `optimize_horizontal_trajectory(seed: HorizontalRouteSeed, line: HorizontalGridLine, target_elevation_m: float, entry_v_m: float, exit_v_m: float, ascent_world_xy: np.ndarray, kinematics: BoundedFootKinematics, sample_surface_height_m: Callable[[np.ndarray], np.ndarray], joint_position_lower: np.ndarray, joint_position_upper: np.ndarray, config: HorizontalOptimizerConfig) -> OptimizedHorizontalTrajectory`
  - `validate_horizontal_trajectory(trajectory: OptimizedHorizontalTrajectory, support_mask: np.ndarray, sample_surface_height_m: Callable[[np.ndarray], np.ndarray], sole_kinematics: MujocoG1SoleKinematics, line: HorizontalGridLine, config: HorizontalOptimizerConfig) -> HorizontalTrajectoryValidation`

- [ ] **Step 1: Write failing alignment, seam, and rejection tests**

```python
def test_alignment_maps_source_progress_to_requested_horizontal_direction():
    aligned = align_seed_root(
        source_root_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
        line=HorizontalGridLine(0, 0.05, -1),
        ascent_world_xy=np.array([1.0, 0.0]),
        entry_v_m=0.50,
        exit_v_m=-0.50,
    )
    np.testing.assert_allclose(aligned[:, 0], 0.05)
    assert aligned[-1, 1] < aligned[0, 1]

def test_optimizer_rejects_target_curb_above_seed_and_leg_envelope():
    with pytest.raises(ContractError, match="height envelope"):
        optimize_horizontal_trajectory(
            seed=seed_for_018_m_curb,
            line=HorizontalGridLine(0, 0.05, 1),
            target_elevation_m=0.54,
            entry_v_m=-0.50,
            exit_v_m=0.50,
            ascent_world_xy=np.array([1.0, 0.0]),
            kinematics=fake_bounded_kinematics,
            sample_surface_height_m=fake_surface_sampler,
            joint_position_lower=np.full(29, -3.0),
            joint_position_upper=np.full(29, 3.0),
            config=HorizontalOptimizerConfig(maximum_elevation_scale=1.25),
        )
```

Also test that the output begins and ends on flat ground, preserves the source
contact order, and retains the source frame count.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_sonic_torch_horizontal_trajectory_optimizer
```

Expected: FAIL because the optimizer module is missing.

- [ ] **Step 3: Implement rigid whole-route alignment**

Rotate the source root displacement onto the requested `±v` direction,
translate its median elevated interval onto the chosen `u` line, and scale
only the source elevation profile:

```python
height_scale = target_elevation_m / seed.source_elevation_m
if not 0.75 <= height_scale <= maximum_elevation_scale:
    raise ContractError("horizontal target exceeds source height envelope")
root_z = flat_z + height_scale * (source_root_z - source_flat_z)
```

Apply the same rigid yaw/translation to body and foot trajectories. Preserve
all source joint positions as the initial motion prior.

- [ ] **Step 4: Implement projected trajectory optimization**

Use at most 12 alternating iterations:

1. optimize root and all joints with PyTorch Adam for 40 steps against source
   deviation, joint velocity, joint acceleration, root smoothness, and
   endpoint progress;
2. project every planted frame back to its fixed seven-point sole pose with
   `BoundedFootKinematics.solve_leg_sole_positions_bounded`;
3. clamp joint positions to the authenticated G1 joint contract;
4. reject immediately if projection is unreachable or contact order changes.

The optimizer configuration is immutable:

```python
@dataclass(frozen=True)
class HorizontalOptimizerConfig:
    outer_iterations: int = 12
    gradient_steps: int = 40
    learning_rate: float = 0.01
    source_weight: float = 1.0
    velocity_weight: float = 0.05
    acceleration_weight: float = 0.01
    endpoint_weight: float = 20.0
    maximum_elevation_scale: float = 1.25
    maximum_joint_speed_rad_s: float = 13.0
    maximum_joint_acceleration_rad_s2: float = 100.0
```

Do not differentiate through MuJoCo. The differentiable step improves temporal
quality; the bounded MuJoCo sole IK projection enforces contact feasibility.

- [ ] **Step 5: Implement independent final validation**

Recompute all seven sole points from emitted root/joint states. Query the
staircase height field at every point. Measure:

- minimum full-sole clearance;
- planted full-sole absolute error and spread;
- flat start/end error;
- lateral progress and line drift;
- joint speed and acceleration;
- five phase completion.

Reject if any Global Constraint fails. Hash the exact emitted arrays only
after validation.

- [ ] **Step 6: Run optimizer tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/torch_horizontal_trajectory_optimizer.py \
  tests/python/test_sonic_torch_horizontal_trajectory_optimizer.py
git commit -m "feat: optimize complete horizontal terrain traversals"
```

---

### Task 4: Batch library builder and first horizontal qualification

**Files:**
- Create: `resources/run_g1_horizontal_traversal_library.py`
- Create: `tests/python/test_run_g1_horizontal_traversal_library.py`
- Modify: `docs/superpowers/plans/2026-08-03-g1-horizontal-traversal-library.md`

**Interfaces:**
- Consumes Tasks 1–3 plus the existing authenticated stair configuration.
- Produces:
  - `<output>/library.json`
  - `<output>/routes/<line-id>/<trajectory-sha>.npz`
  - `<output>/routes/<line-id>/<trajectory-sha>.json`
  - `<output>/inventory.json`

- [ ] **Step 1: Write failing CLI and failure-accounting tests**

```python
def test_parser_requires_dataset_stair_and_g1_inputs():
    args = module._parser().parse_args([
        "--dataset", "corpus",
        "--config", "experiment.json",
        "--g1-xml", "g1.xml",
        "--output", "library",
    ])
    assert args.spacing_m == 0.10
    assert args.maximum_seed_count_per_line == 32

def test_builder_records_unreachable_line_instead_of_partial_success():
    result = build_line(fake_high_line, seeds=(low_curb_seed,))
    assert result.artifacts == ()
    assert result.failure.failed_phase == "mount"
    assert result.failure.rejection_histogram["height_envelope"] == 1
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_run_g1_horizontal_traversal_library
```

Expected: FAIL because the runner is missing.

- [ ] **Step 3: Implement deterministic batch generation**

For each 10 cm line and both directions:

1. compute target tread elevation from the authenticated staircase query;
2. rank whole-route seeds by elevation ratio, travel distance, source contact
   error, and family (`curb` wins ties);
3. try at most 32 anchor sequences, invoking ARDY only when a splice is
   required;
4. retain all valid non-dominated trajectories;
5. otherwise record the failed phase and complete rejection histogram.

Write to a temporary output directory and atomically rename it only after the
library manifest round-trips through `horizontal_library_from_dict`.

- [ ] **Step 4: Run the focused unit suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_sonic_torch_horizontal_traversal_library \
tests.python.test_sonic_torch_horizontal_traversal_sources \
tests.python.test_sonic_torch_ardy_inbetween \
tests.python.test_run_g1_ardy_inbetween_worker \
tests.python.test_run_g1_horizontal_traversal_library
```

Expected: PASS.

- [ ] **Step 5: Build the real horizontal library**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B \
resources/run_g1_horizontal_traversal_library.py \
  --dataset build/torch-grail-terrain-full-v1 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-horizontal-traversal-library-v1 \
  --spacing-m 0.10 \
  --maximum-seed-count-per-line 32
```

Expected: every line/direction has at least one artifact or one explicit
failure, and at least one complete route is accepted.

- [ ] **Step 6: Record measured coverage in this plan**

Add the exact seed counts, successful line count, failed line count, rejection
histogram, best route metrics, and deterministic library identity. Do not
describe a route as successful until its saved NPZ has been independently
revalidated.

- [ ] **Step 7: Commit**

```bash
git add resources/run_g1_horizontal_traversal_library.py \
  tests/python/test_run_g1_horizontal_traversal_library.py \
  docs/superpowers/plans/2026-08-03-g1-horizontal-traversal-library.md
git commit -m "feat: build horizontal staircase traversal library"
```

---

### Task 5: Coverage and route visualizer

**Files:**
- Create: `resources/run_g1_horizontal_traversal_viewer.py`
- Create: `tests/python/test_run_g1_horizontal_traversal_viewer.py`

**Interfaces:**
- Consumes Task 4 `library.json` and saved route NPZ artifacts.
- Produces interactive MuJoCo playback and headless coverage/contact sheets.

- [ ] **Step 1: Write failing parser and artifact-authentication tests**

```python
def test_viewer_requires_library_dataset_and_robot():
    args = module._parser().parse_args([
        "--library", "library.json",
        "--dataset", "corpus",
        "--config", "experiment.json",
        "--g1-xml", "g1.xml",
    ])
    assert args.loop

def test_viewer_rejects_tampered_route_npz():
    with pytest.raises(ContractError, match="identity"):
        module._load_route(tampered_path, expected_sha256="a" * 64)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_run_g1_horizontal_traversal_viewer
```

Expected: FAIL because the viewer is missing.

- [ ] **Step 3: Implement coverage and playback**

Render the staircase with:

- green lines for at least one qualified route;
- red lines for explicit failures;
- a highlighted selected line and direction;
- source family, curb height, and validation metrics in the overlay.

Keys: left/right cycle library entries, space pauses, backspace rewinds, `X`
exits. Headless flags write one coverage image and one six-frame contact sheet
per accepted route.

- [ ] **Step 4: Run viewer tests and render all contact sheets**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B -m unittest -q \
tests.python.test_run_g1_horizontal_traversal_viewer

MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
sonic/.torch-mm-venv/bin/python -B \
resources/run_g1_horizontal_traversal_viewer.py \
  --library build/g1-horizontal-traversal-library-v1/library.json \
  --dataset build/torch-grail-terrain-full-v1 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --render-directory build/g1-horizontal-traversal-library-v1/renders
```

Expected: tests pass and every accepted route has a contact sheet.

- [ ] **Step 5: Visually inspect every accepted contact sheet**

Reject and regenerate any route with visible root sliding, missing approach or
departure, tilted planted soles, early mounting, edge floating, or a missing
descent—even when scalar metrics pass.

- [ ] **Step 6: Run final verification and commit**

Run the Task 4 focused suite plus the viewer test, then:

```bash
git diff --check
git add resources/run_g1_horizontal_traversal_viewer.py \
  tests/python/test_run_g1_horizontal_traversal_viewer.py
git commit -m "feat: visualize horizontal traversal coverage"
```

Expected: all focused tests pass, every library entry authenticates, and the
worktree contains no staged unrelated files.
