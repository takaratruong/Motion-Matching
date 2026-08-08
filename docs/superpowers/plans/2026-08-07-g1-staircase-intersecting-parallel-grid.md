# G1 Staircase-Intersecting Parallel Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terrain-derived 20 cm parallel grid in which every playable route mounts, traverses, and dismounts the staircase, while rejecting flat substitutions and visibly unnatural chains.

**Architecture:** Add a pure geometry/profile layer that derives the lateral sampling band from elevated terrain cells and classifies every centerline before search. Keep the existing root-path search as the motion proposer, add contact/cadence-aware candidate ranking, and place an independent terrain-event and naturalness audit between search success and playlist admission. Preserve the accepted horizontal artifacts and use a new staircase-only playlist contract rather than weakening the generic playlist API.

**Tech Stack:** Python 3, NumPy, PyTorch, MuJoCo G1 forward kinematics, `unittest`, existing GRAIL height-grid and root-path search infrastructure.

## Global Constraints

- Grid spacing is exactly 0.20 m for the feedback build.
- Every playable route must contain `ground -> elevated staircase -> opposite ground`.
- Flat-only routes are reported as `flat_only_excluded` and never enter the viewer.
- Failed staircase lanes remain in the grid report and are never replaced with easier paths.
- Heading, spacing, terrain footprint, and endpoints remain runtime inputs.
- No heading-specific source frames, offsets, tread coordinates, or heights may be hardcoded.
- Existing accepted horizontal artifacts must not be overwritten.
- MotionBricks or interpolation may bridge only short contact-compatible boundaries; it may not invent missing terrain-changing motion.

---

### Task 1: Terrain-Derived Staircase Grid Geometry

**Files:**
- Modify: `sonic/python/mm_sonic/torch_path_motion_grid.py`
- Modify: `tests/python/test_sonic_torch_path_motion_grid.py`

**Interfaces:**
- Consumes: elevated height-grid cell centers in scene XY, a runtime center start, heading, path length, spacing, and a callable scene-height sampler.
- Produces: `StaircasePathContract`, `staircase_parallel_path_grid(...)`, and `classify_staircase_path(...)` for the runner and playlist admission.

- [ ] **Step 1: Write failing tests for terrain-centered sampling and route classification**

Add imports and these cases to `tests/python/test_sonic_torch_path_motion_grid.py`:

```python
from mm_sonic.torch_path_motion_grid import (
    StaircasePathContract,
    classify_staircase_path,
    staircase_parallel_path_grid,
)

def test_staircase_grid_centers_twenty_centimeter_lattice_on_elevated_band(self):
    points = np.array(
        [(x, y) for x in np.linspace(0.0, 1.0, 6)
         for y in np.linspace(-0.51, 0.49, 6)],
        dtype=np.float64,
    )
    paths = staircase_parallel_path_grid(
        center_start_scene_xy=(-0.5, 0.8),
        heading_scene_xy=(1.0, 0.0),
        path_length_m=2.0,
        elevated_scene_xy=points,
        spacing_m=0.20,
    )
    np.testing.assert_allclose(
        [path.start_scene_xy[1] for path in paths],
        (-0.41, -0.21, -0.01, 0.19, 0.39),
        atol=1.0e-12,
    )
    self.assertTrue(all(-0.51 < p.start_scene_xy[1] < 0.49 for p in paths))

def test_staircase_profile_requires_approach_elevation_and_opposite_exit(self):
    path = parallel_path_grid(
        center_start_scene_xy=(0.0, 0.0),
        heading_scene_xy=(1.0, 0.0),
        path_length_m=2.0,
        minimum_lateral_offset_m=0.0,
        maximum_lateral_offset_m=0.0,
        spacing_m=0.2,
    )[0]

    def staircase(points):
        x = np.asarray(points)[:, 0]
        return np.where((x >= 0.5) & (x <= 1.5), 0.2, 0.0)

    contract = classify_staircase_path(path=path, sample_surface=staircase)
    self.assertEqual(contract.classification, "staircase_intersecting")
    self.assertEqual(contract.ordered_surface_heights_m, (0.0, 0.2, 0.0))

def test_flat_path_is_explicitly_excluded(self):
    path = parallel_path_grid(
        center_start_scene_xy=(0.0, 0.0),
        heading_scene_xy=(1.0, 0.0),
        path_length_m=2.0,
        minimum_lateral_offset_m=0.0,
        maximum_lateral_offset_m=0.0,
        spacing_m=0.2,
    )[0]
    contract = classify_staircase_path(
        path=path,
        sample_surface=lambda points: np.zeros(len(points)),
    )
    self.assertEqual(contract.classification, "flat_only_excluded")
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_sonic_torch_path_motion_grid -v
```

Expected: import failures for the three new staircase-grid symbols.

- [ ] **Step 3: Implement centered geometry and terrain profile contracts**

Add this public contract and implementations to `torch_path_motion_grid.py`:

```python
@dataclass(frozen=True)
class StaircasePathContract:
    path: ParallelPath
    classification: str
    ordered_surface_heights_m: tuple[float, ...]
    elevated_intervals_m: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, ParallelPath)
            or self.classification not in (
                "staircase_intersecting",
                "flat_only_excluded",
                "missing_approach",
                "missing_opposite_exit",
            )
            or not self.ordered_surface_heights_m
        ):
            raise ContractError("staircase path contract is invalid")


def staircase_parallel_path_grid(
    *,
    center_start_scene_xy: tuple[float, float],
    heading_scene_xy: tuple[float, float],
    path_length_m: float,
    elevated_scene_xy: object,
    spacing_m: float,
) -> tuple[ParallelPath, ...]:
    points = np.asarray(elevated_scene_xy, dtype=np.float64)
    heading = np.asarray(heading_scene_xy, dtype=np.float64)
    origin = np.asarray(center_start_scene_xy, dtype=np.float64)
    if (
        points.ndim != 2 or points.shape[1:] != (2,) or not len(points)
        or not np.isfinite(points).all() or not np.isfinite(heading).all()
        or float(np.linalg.norm(heading)) <= 0.0 or spacing_m <= 0.0
    ):
        raise ContractError("staircase terrain footprint is invalid")
    forward = heading / np.linalg.norm(heading)
    lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
    projected = (points - origin) @ lateral
    lower, upper = float(projected.min()), float(projected.max())
    count = max(1, int(math.floor((upper - lower) / spacing_m + 1.0e-9)))
    first = 0.5 * (lower + upper) - 0.5 * (count - 1) * spacing_m
    paths = parallel_path_grid(
        center_start_scene_xy=tuple(origin),
        heading_scene_xy=tuple(forward),
        path_length_m=path_length_m,
        minimum_lateral_offset_m=first,
        maximum_lateral_offset_m=first + (count - 1) * spacing_m,
        spacing_m=spacing_m,
    )
    return paths


def classify_staircase_path(
    *,
    path: ParallelPath,
    sample_surface,
    sample_spacing_m: float = 0.02,
    elevated_threshold_m: float = 0.05,
    minimum_ground_run_m: float = 0.15,
    minimum_elevated_run_m: float = 0.08,
) -> StaircasePathContract:
    start = np.asarray(path.start_scene_xy, dtype=np.float64)
    stop = np.asarray(path.stop_scene_xy, dtype=np.float64)
    length = float(np.linalg.norm(stop - start))
    count = int(math.ceil(length / sample_spacing_m)) + 1
    distance = np.linspace(0.0, length, count)
    points = start + distance[:, None] * (stop - start)[None, :] / length
    height = np.asarray(sample_surface(points), dtype=np.float64)
    ground = float(min(height[0], height[-1]))
    elevated = height > ground + elevated_threshold_m
    ordered = [float(height[0])]
    for value in height[1:]:
        if abs(float(value) - ordered[-1]) > elevated_threshold_m:
            ordered.append(float(value))
    intervals = []
    for begin in np.flatnonzero(elevated & ~np.r_[False, elevated[:-1]]):
        stops = np.flatnonzero(~elevated[begin:])
        end = len(elevated) if not len(stops) else begin + int(stops[0])
        intervals.append((float(distance[begin]), float(distance[end - 1])))
    classification = "staircase_intersecting"
    if not intervals or max(b - a for a, b in intervals) < minimum_elevated_run_m:
        classification = "flat_only_excluded"
    elif intervals[0][0] < minimum_ground_run_m:
        classification = "missing_approach"
    elif length - intervals[-1][1] < minimum_ground_run_m:
        classification = "missing_opposite_exit"
    return StaircasePathContract(
        path=path,
        classification=classification,
        ordered_surface_heights_m=tuple(ordered),
        elevated_intervals_m=tuple(intervals),
    )
```

During implementation, retain the exact validation style already used in this
module and normalize clustered height representatives so tests do not depend
on floating interpolation noise.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Task 1 unittest command. Expected: all grid tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_path_motion_grid.py \
  tests/python/test_sonic_torch_path_motion_grid.py
git commit -m "feat: derive parallel grid from staircase footprint"
```

---

### Task 2: Staircase-Only Grid Orchestration and Complete Reporting

**Files:**
- Modify: `resources/run_g1_parallel_path_grid.py`
- Modify: `tests/python/test_run_g1_parallel_path_grid.py`

**Interfaces:**
- Consumes: `staircase_parallel_path_grid(...)`, `classify_staircase_path(...)`, `_target_grid(...)`, and `_sample_height_grid(...)`.
- Produces: `terrain_grid_contracts(...)`, a v2 grid summary with excluded/failed/validated lanes, and root searches only for staircase-intersecting lanes.

- [ ] **Step 1: Replace the flat-fallback test with staircase-admission tests**

Add tests that assert:

```python
def test_parser_uses_automatic_staircase_band_by_default(self):
    args = _MODULE.parser().parse_args([
        "--source-dataset", "source", "--target-scene", "scene",
        "--g1-xml", "g1.xml", "--center-start", "0.1", "0.2",
        "--heading-degrees", "-45", "--path-length-m", "2.4",
        "--spacing-m", "0.2", "--output", "output",
    ])
    self.assertIsNone(args.minimum_lateral_offset_m)
    self.assertIsNone(args.maximum_lateral_offset_m)

def test_summary_counts_excluded_paths_without_treating_them_as_failures(self):
    summary = _MODULE.grid_summary(
        geometry={"spacing_m": 0.2},
        results=(
            {"path_id": "flat", "status": "excluded",
             "failure_code": "flat_only_excluded"},
            {"path_id": "bad", "status": "failed",
             "failure_code": "no_mount_motion"},
            {"path_id": "good", "status": "validated"},
        ),
    )
    self.assertEqual(summary["excluded_path_count"], 1)
    self.assertEqual(summary["failed_path_count"], 1)
    self.assertEqual(summary["validated_path_count"], 1)
    self.assertEqual(len(summary["results"]), 3)

def test_flat_contract_never_dispatches_root_search(self):
    dispatch = _MODULE.searchable_contracts((
        SimpleNamespace(classification="flat_only_excluded"),
        SimpleNamespace(classification="staircase_intersecting"),
    ))
    self.assertEqual(len(dispatch), 1)
    self.assertEqual(dispatch[0].classification, "staircase_intersecting")

def test_search_failure_is_reported_as_missing_required_phase(self):
    self.assertEqual(
        _MODULE.classify_search_failure(
            log_text="ContractError: no complete ordered-level chain",
            diagnostic={"first_missing_phase": "dismount"},
        ),
        "no_dismount_motion",
    )
```

Delete the old expectation that ordered-level failure retries as ordinary flat
root search. A staircase lane must fail rather than silently change tasks.

- [ ] **Step 2: Run runner tests and confirm RED**

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_run_g1_parallel_path_grid -v
```

Expected: parser and summary assertions fail because the runner still requires
manual lateral bounds and supports ordinary-flat fallback.

- [ ] **Step 3: Implement terrain-derived orchestration**

Change the manual lateral flags to optional paired overrides and add these
helpers:

```python
def searchable_contracts(contracts):
    return tuple(
        item for item in contracts
        if item.classification == "staircase_intersecting"
    )


def elevated_grid_cell_centers(origin, cell, height, *, threshold_m=0.05):
    ground = float(np.nanmin(height))
    rows, columns = np.nonzero(height > ground + threshold_m)
    if not len(rows):
        raise ContractError("target terrain has no elevated staircase cells")
    return np.column_stack((
        origin[0] + (columns + 0.5) * cell,
        origin[1] + (rows + 0.5) * cell,
    ))


def classify_search_failure(*, log_text, diagnostic):
    phase = diagnostic.get("first_missing_phase") if isinstance(diagnostic, dict) else None
    if phase in ("mount", "interior", "dismount"):
        return {
            "mount": "no_mount_motion",
            "interior": "no_elevated_traversal_motion",
            "dismount": "no_dismount_motion",
        }[phase]
    if "height" in log_text and "corpus" in log_text:
        return "height_out_of_corpus_range"
    if "footprint" in log_text:
        return "no_terrain_footprint"
    return "no_contact_compatible_chain"
```

In `main()`, resolve the target descriptor and height grid before generating
paths. Generate the centered terrain band, classify every path profile, write
all profiles to `grid-contract.json`, dispatch `_run_path` only for
`staircase_intersecting` contracts, and synthesize `status="excluded"` records
for all others. Remove `should_retry_without_ordered_levels`,
`flat_fallback_stale_artifacts`, and the ordinary-flat retry block. Every
dispatched staircase route always passes `--ordered-contact-levels`.

Update `grid_summary(...)` to accept only `excluded`, `failed`, or `validated`
and return:

```python
{
    "schema": "g1-staircase-parallel-path-grid/v2",
    "geometry": dict(geometry),
    "requested_path_count": len(results),
    "staircase_path_count": statuses.count("failed") + statuses.count("validated"),
    "excluded_path_count": statuses.count("excluded"),
    "validated_path_count": statuses.count("validated"),
    "failed_path_count": statuses.count("failed"),
    "results": list(results),
}
```

- [ ] **Step 4: Run runner and grid tests**

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_sonic_torch_path_motion_grid \
  tests.python.test_run_g1_parallel_path_grid -v
```

Expected: all tests pass and no test describes a flat fallback.

- [ ] **Step 5: Commit Task 2**

```bash
git add resources/run_g1_parallel_path_grid.py \
  tests/python/test_run_g1_parallel_path_grid.py
git commit -m "fix: exclude flat routes from staircase grid"
```

---

### Task 3: Contact and Naturalness Quality Oracle

**Files:**
- Create: `sonic/python/mm_sonic/torch_staircase_traversal_quality.py`
- Create: `tests/python/test_sonic_torch_staircase_traversal_quality.py`
- Modify: `resources/run_g1_validate_traversal.py`
- Modify: `tests/python/test_run_g1_validate_traversal.py`

**Interfaces:**
- Consumes: scene-space feet and complete sole points, support mask, route heading, sampled terrain, and ordered route profile.
- Produces: `measure_staircase_traversal_quality(...)`, `admit_staircase_traversal(...)`, explicit terrain-event/naturalness metrics, and stable failure reasons.

- [ ] **Step 1: Write failing oracle tests**

Create tests for these exact behaviors:

```python
def test_rejects_motion_that_never_reaches_elevated_terrain():
    metrics = synthetic_metrics(terrain_events=("ground", "ground"))
    with self.assertRaisesRegex(ContractError, "missing elevated traversal"):
        admit_staircase_traversal(metrics)

def test_rejects_stance_slide_even_when_vertical_contact_is_valid():
    metrics = synthetic_metrics(maximum_stance_horizontal_step_m=0.018)
    with self.assertRaisesRegex(ContractError, "stance foot slides"):
        admit_staircase_traversal(metrics)

def test_height_change_is_exempt_but_same_height_cadence_is_bounded():
    metrics = measure_cadence(
        touchdown_frames=(0, 30, 62, 92, 150),
        touchdown_heights_m=(0.0, 0.0, 0.0, 0.2, 0.0),
    )
    self.assertEqual(metrics["same_height_half_step_frames"], [30, 32])
    self.assertEqual(metrics["cadence_violation_count"], 0)

def test_rejects_crossed_feet_in_heading_local_frame():
    metrics = synthetic_metrics(minimum_lateral_foot_separation_m=-0.03)
    with self.assertRaisesRegex(ContractError, "feet cross"):
        admit_staircase_traversal(metrics)

def test_rejects_excessive_same_height_double_support():
    metrics = synthetic_metrics(maximum_same_height_double_support_frames=26)
    with self.assertRaisesRegex(ContractError, "double support"):
        admit_staircase_traversal(metrics)
```

Add validator tests ensuring the independent validator calls the new admission
function after its existing contact checks.

- [ ] **Step 2: Run oracle tests and confirm RED**

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_sonic_torch_staircase_traversal_quality \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: module import failure.

- [ ] **Step 3: Implement the standalone quality oracle**

Create `torch_staircase_traversal_quality.py` with immutable metric inputs and
these public functions:

```python
def measure_cadence(*, touchdown_frames, touchdown_heights_m,
                    height_change_threshold_m=0.05,
                    allowed_relative_error=0.20):
    frames = np.asarray(touchdown_frames, dtype=np.int64)
    heights = np.asarray(touchdown_heights_m, dtype=np.float64)
    intervals = np.diff(frames)
    same_height = np.abs(np.diff(heights)) <= height_change_threshold_m
    trusted = intervals[same_height]
    target = float(np.median(trusted)) if len(trusted) else 0.0
    violations = (
        np.abs(trusted - target) > allowed_relative_error * target
        if target > 0.0 else np.zeros(len(trusted), dtype=np.bool_)
    )
    return {
        "target_half_step_frames": target,
        "same_height_half_step_frames": trusted.astype(int).tolist(),
        "cadence_violation_count": int(violations.sum()),
    }


def admit_staircase_traversal(metrics):
    events = tuple(metrics["terrain_events"])
    if events[0] != "ground" or "elevated" not in events[1:-1]:
        raise ContractError("traversal is missing elevated traversal")
    if events[-1] != "opposite_ground":
        raise ContractError("traversal is missing opposite-ground exit")
    if metrics["maximum_stance_horizontal_step_m"] > 0.010:
        raise ContractError("traversal stance foot slides")
    if metrics["maximum_complete_sole_contact_error_m"] > 0.025:
        raise ContractError("traversal loses complete-sole contact")
    if metrics["minimum_lateral_foot_separation_m"] < 0.0:
        raise ContractError("traversal feet cross")
    if metrics["cadence_violation_count"]:
        raise ContractError("traversal same-height cadence is inconsistent")
    if metrics["maximum_same_height_double_support_frames"] > 20:
        raise ContractError("traversal has excessive same-height double support")
    if metrics["minimum_supported_sole_points"] < 3:
        raise ContractError("traversal has incomplete sole support")
    if not metrics["terminal_complete_support"]:
        raise ContractError("traversal terminal phase is mid-swing")
```

`measure_staircase_traversal_quality(...)` must derive actual touchdown events
from false-to-true support edges, sample the complete sole against terrain,
measure heading-local left/right separation, measure continuous double-support
runs, and report worst frames. It must not trust source filenames or declared
terrain phases.

- [ ] **Step 4: Wire the oracle into independent validation**

Extend `run_g1_validate_traversal.py` with optional route-contract JSON input.
When supplied, map feet/soles to scene coordinates, call
`measure_staircase_traversal_quality(...)`, merge its metrics into the report,
then call `admit_staircase_traversal(...)`. Preserve the existing validator for
non-grid callers that omit this argument.

- [ ] **Step 5: Run oracle and validator tests**

Run the Task 3 unittest command. Expected: all pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add sonic/python/mm_sonic/torch_staircase_traversal_quality.py \
  tests/python/test_sonic_torch_staircase_traversal_quality.py \
  resources/run_g1_validate_traversal.py \
  tests/python/test_run_g1_validate_traversal.py
git commit -m "feat: independently audit staircase traversal quality"
```

---

### Task 4: Naturalness-Aware Motion Search

**Files:**
- Modify: `resources/run_g1_root_path_motion_search.py`
- Modify: `tests/python/test_run_g1_root_path_motion_search.py`
- Modify: `sonic/python/mm_sonic/torch_root_path_motion_graph.py`
- Modify: `tests/python/test_sonic_torch_root_path_motion_graph.py`

**Interfaces:**
- Consumes: candidate support/contact timing, actual feet, heading-local path progress, and certified splice frames.
- Produces: `candidate_naturalness_penalty(...)`, coherent-source preference, and transition costs that penalize foot unlock, phase mismatch, cadence mismatch, and fragmentation before chain selection.

- [ ] **Step 1: Write failing ranking tests**

Add a graph regression demonstrating that a coherent two-window chain beats a
four-fragment chain with equal geometric cost:

```python
def test_search_prefers_coherent_chain_over_many_equal_cost_fragments(self):
    candidates = (
        self.candidate("long-a", 0.0, 1.1, 0.1),
        self.candidate("long-b", 1.0, 2.0, 0.1),
        self.candidate("s1", 0.0, 0.6, 0.05),
        self.candidate("s2", 0.5, 1.1, 0.05),
        self.candidate("s3", 1.0, 1.6, 0.05),
        self.candidate("s4", 1.5, 2.0, 0.05),
    )
    transitions = (
        RootPathMotionTransition("long-a", "long-b", 0.05),
        RootPathMotionTransition("s1", "s2", 0.05),
        RootPathMotionTransition("s2", "s3", 0.05),
        RootPathMotionTransition("s3", "s4", 0.05),
    )
    result = select_root_path_motion_chain(
        candidates=candidates, transitions=transitions,
        path_length_m=2.0, fragmentation_cost=0.05,
    )
    self.assertEqual(result.candidate_ids, ("long-a", "long-b"))
```

Add root-search tests that a candidate with crossed feet or an excessively
long same-height double-support dwell receives a larger penalty than a clean
alternating gait, and that a transition retaining the same source window has
lower cost than an equally matched arbitrary splice.

- [ ] **Step 2: Run search tests and confirm RED**

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_sonic_torch_root_path_motion_graph \
  tests.python.test_run_g1_root_path_motion_search -v
```

Expected: unknown `fragmentation_cost` and missing naturalness helper.

- [ ] **Step 3: Add fragmentation-aware graph cost**

Add `fragmentation_cost: float = 0.0` to both chain selectors, validate it as a
finite nonnegative value, and add it exactly once for every transition added
to a chain:

```python
previous_best[0]
+ float(transition.transition_cost)
+ float(fragmentation_cost)
+ float(candidate.placement_cost)
```

Keep the default zero so existing callers remain byte-for-byte compatible.

- [ ] **Step 4: Add candidate and transition naturalness costs**

Implement:

```python
def _candidate_naturalness_penalty(*, support, feet_scene, heading_scene_xy):
    support = np.asarray(support, dtype=np.bool_)
    feet = np.asarray(feet_scene, dtype=np.float64)
    heading = np.asarray(heading_scene_xy, dtype=np.float64)
    lateral = np.array((-heading[1], heading[0])) / np.linalg.norm(heading)
    separation = (feet[:, 0, :2] - feet[:, 1, :2]) @ lateral
    double = support.all(axis=1)
    longest_double = max(
        (stop - start for start, stop in _true_runs(double)), default=0
    )
    crossing = max(0.0, -float(np.min(separation)))
    dwell = max(0, longest_double - 20)
    return 25.0 * crossing + 0.01 * dwell
```

Apply the penalty to candidate placement cost after terrain certification.
Extend `_transition_candidates(...)` with a same-source bonus implemented as a
reduction bounded at zero, plus explicit stance-foot position gap and support
phase mismatch penalties. Pass `fragmentation_cost=0.05` from both ordered and
ordinary root-path searches. Record each selected candidate's naturalness
components and total fragmentation cost in `report.json`.

- [ ] **Step 5: Run search tests and focused regression suite**

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_sonic_torch_root_path_motion_graph \
  tests.python.test_run_g1_root_path_motion_search \
  tests.python.test_sonic_torch_path_motion_grid \
  tests.python.test_run_g1_parallel_path_grid -v
```

Expected: all pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add resources/run_g1_root_path_motion_search.py \
  tests/python/test_run_g1_root_path_motion_search.py \
  sonic/python/mm_sonic/torch_root_path_motion_graph.py \
  tests/python/test_sonic_torch_root_path_motion_graph.py
git commit -m "fix: rank coherent terrain motion chains"
```

---

### Task 5: Independent Admission, Staircase-Only Playlist, and Feedback Build

**Files:**
- Modify: `sonic/python/mm_sonic/torch_path_motion_grid.py`
- Modify: `tests/python/test_sonic_torch_path_motion_grid.py`
- Modify: `resources/run_g1_parallel_path_grid.py`
- Modify: `tests/python/test_run_g1_parallel_path_grid.py`
- Create: `docs/superpowers/results/2026-08-07-g1-staircase-intersecting-parallel-grid.md`

**Interfaces:**
- Consumes: complete grid summary, staircase contracts, root-search artifacts, and independent validation reports.
- Produces: `build_staircase_grid_playlist(...)`, a corrected 45-degree build directory, coverage report, playlist, and 50 Hz MuJoCo launch command.

- [ ] **Step 1: Write failing playlist-admission tests**

Add tests proving the builder rejects flat and unaudited paths:

```python
def test_staircase_playlist_rejects_flat_or_unvalidated_route(self):
    path = parallel_path_grid(
        center_start_scene_xy=(0.0, 0.0), heading_scene_xy=(1.0, 0.0),
        path_length_m=1.0, minimum_lateral_offset_m=0.0,
        maximum_lateral_offset_m=0.0, spacing_m=0.2,
    )[0]
    connector = self.connector(0.0)
    with self.assertRaisesRegex(ContractError, "not staircase-admitted"):
        build_staircase_grid_playlist((
            (path, connector, {"classification": "flat_only_excluded",
                               "independently_validated": True}),
        ))
    with self.assertRaisesRegex(ContractError, "not staircase-admitted"):
        build_staircase_grid_playlist((
            (path, connector, {"classification": "staircase_intersecting",
                               "independently_validated": False}),
        ))
```

- [ ] **Step 2: Run grid tests and confirm RED**

Run the Task 1 unittest command. Expected: missing staircase playlist builder.

- [ ] **Step 3: Implement staircase-only playlist admission**

Add `build_staircase_grid_playlist(...)` beside the generic builder. It accepts
only ordered `ParallelPath` entries whose admission dictionary has both
`classification == "staircase_intersecting"` and
`independently_validated is True`. Its metadata schema is
`g1-staircase-parallel-path-playlist/v1`, and every segment embeds ordered
terrain levels and the independent quality summary.

In the runner, invoke the independent validator after each root-search
success. A validator failure changes the route to
`status="failed", failure_code="validation_failed"`; only independent passes
are written to the playlist. Do not create an empty playlist when every route
fails.

- [ ] **Step 4: Run the complete focused test suite**

```bash
PYTHONPATH=sonic/python:. python -m unittest \
  tests.python.test_sonic_torch_path_motion_grid \
  tests.python.test_run_g1_parallel_path_grid \
  tests.python.test_sonic_torch_staircase_traversal_quality \
  tests.python.test_run_g1_validate_traversal \
  tests.python.test_sonic_torch_root_path_motion_graph \
  tests.python.test_run_g1_root_path_motion_search -v
```

Expected: all tests pass.

- [ ] **Step 5: Preserve the horizontal regression before running experiments**

Record SHA-256 hashes of the accepted horizontal manifest and traversal:

```bash
sha256sum \
  build/g1-horizontal-grid-*/grid-playlist.json \
  build/g1-horizontal-grid-*/grid-playlist.npz
```

If more than one directory matches, use the accepted path recorded in the
current horizontal result document. Save the exact paths and hashes in the new
result document; do not write into those directories.

- [ ] **Step 6: Run the corrected 45-degree 20 cm grid**

Run the approved -45-degree geometry without manual lateral bounds:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_parallel_path_grid.py \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --target-scene grail-stair_p1-db7949fce1b2e48d39f4 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --center-start -0.4835621836 1.4914988213 \
  --heading-degrees -45 \
  --path-length-m 2.5000000000475615 \
  --spacing-m 0.20 \
  --segment-length-m 0.90 --search-stride-m 0.20 \
  --step-width-m 0.20 --workers-per-path 16 --concurrent-paths 4 \
  --coarse-results 2400 --placement-shortlist 600 \
  --candidates-per-anchor 24 --blend-frames 4 \
  --ordered-contact-levels \
  --output build/g1-staircase-grid-neg45-20cm-v7-admitted
```

The runner must derive the lateral band and must not use an ordinary flat
fallback.

Expected artifacts:

```text
grid-contract.json
grid-summary.json
grid-playlist.npz            # only when at least one route passes
grid-playlist.json           # only when at least one route passes
path-*/report.json or path-*/diagnostic.json
path-*/independent-validation.json for every search success
```

- [ ] **Step 7: Audit the build before launching**

Verify with a small read-only inspection command that:

- every summary entry is present;
- no playlist source has `search_mode == "ordinary-flat-fallback"`;
- every playlist route has nonempty elevated intervals;
- every playlist route has `independently_validated == true`;
- every motion starts and ends on ground and crosses elevated terrain;
- the horizontal hashes from Step 5 are unchanged.

Then replay each admitted route at 50 Hz and inspect contact, sliding, cadence,
mount, elevated gait, and dismount. If an admitted route is visibly unnatural,
record the exact route/frame and treat it as a missing oracle regression before
showing the playlist.

- [ ] **Step 8: Document results and launch the corrected viewer**

Write the exact grid offsets, terrain profiles, passes, classified failures,
quality metrics, preserved horizontal hashes, viewer PID, and launch command to
`docs/superpowers/results/2026-08-07-g1-staircase-intersecting-parallel-grid.md`.
Close only obsolete grid viewers after the new viewer is confirmed alive; keep
unrelated user processes untouched.

- [ ] **Step 9: Commit Task 5 source and result document**

```bash
git add sonic/python/mm_sonic/torch_path_motion_grid.py \
  tests/python/test_sonic_torch_path_motion_grid.py \
  resources/run_g1_parallel_path_grid.py \
  tests/python/test_run_g1_parallel_path_grid.py \
  docs/superpowers/results/2026-08-07-g1-staircase-intersecting-parallel-grid.md
git commit -m "feat: publish admitted staircase traversal grid"
```
