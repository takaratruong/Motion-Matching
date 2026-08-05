# G1 Heading-General Footprint Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline 50 Hz G1 terrain traversal planner that follows a constant arbitrary heading by generating terrain-valid footprint candidates and retrieving and retargeting contact-to-contact GRAIL motions.

**Architecture:** A heading-local nominal path produces alternating footprint targets. A full-sole terrain search expands each target into a candidate layer, then a deterministic beam search reuses the existing `FootholdActionIndex` and `FootholdTransitionGraph` to select compatible two-contact motion windows. A separate realization pass aligns and retargets the selected source windows, and the existing independent traversal validator accepts or rejects the final kinematics.

**Tech Stack:** Python 3.10, NumPy, PyTorch, SciPy least-squares, MuJoCo kinematics, existing `mm_sonic` terrain/contact infrastructure, `unittest`.

## Global Constraints

- The runtime input is a constant normalized heading, start state, distance, and speed.
- No fixed scene lane, heading-specific code path, or runtime source-frame selection is permitted.
- All contact and retrieval features use heading-local forward/lateral coordinates.
- Every declared contact must validate the complete sampled sole footprint.
- XY footprint adjustment is bounded and scored; Z-only terrain projection is insufficient.
- Retrieved motions are proposals and must survive bounded retargeting and independent validation.
- The traversal must end at a complete contact phase, never a frozen mid-swing.
- Sonic tracking, physics, time-varying headings, and real-time optimization remain out of scope.

---

### Task 1: Heading-local nominal footprint path

**Files:**
- Create: `sonic/python/mm_sonic/torch_heading_footprint_path.py`
- Test: `tests/python/test_sonic_torch_heading_footprint_path.py`

**Interfaces:**
- Consumes: starting foot centers `(2, 2)`, support mask `(2,)`, heading `(2,)`, distance, speed, stride, width, and 50 Hz frame rate.
- Produces: `ConstantHeadingRequest`, `NominalFootprint`, `NominalFootprintPath`, `heading_basis()`, `scene_to_heading_local()`, `heading_local_to_scene()`, and `nominal_footprint_path()`.

Use these exact fields:

```python
@dataclass(frozen=True)
class ConstantHeadingRequest:
    start_foot_scene_xy: torch.Tensor
    start_support: torch.Tensor
    heading_scene_xy: torch.Tensor
    distance_m: float
    speed_mps: float
    stride_m: float
    step_width_m: float
    frames_per_second: float


@dataclass(frozen=True)
class NominalFootprint:
    step_index: int
    foot: int
    center_scene_xy: torch.Tensor
    center_heading_xy: torch.Tensor
    yaw_scene_rad: float
    contact_frame: int


@dataclass(frozen=True)
class NominalFootprintPath:
    origin_scene_xy: torch.Tensor
    heading_scene_xy: torch.Tensor
    footprints: tuple[NominalFootprint, ...]
    progress_m: float
```

- [ ] **Step 1: Write failing rotation and path tests**

```python
import math
import unittest

import torch

from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    heading_local_to_scene,
    nominal_footprint_path,
    scene_to_heading_local,
)


class HeadingFootprintPathTests(unittest.TestCase):
    def test_heading_coordinates_round_trip_at_arbitrary_yaw(self):
        heading = torch.tensor(
            (math.sqrt(0.5), math.sqrt(0.5)), dtype=torch.float32
        )
        origin = torch.tensor((1.0, -0.3), dtype=torch.float32)
        local = torch.tensor(((0.4, -0.1), (0.9, 0.2)))
        scene = heading_local_to_scene(local, origin, heading)
        torch.testing.assert_close(
            scene_to_heading_local(scene, origin, heading), local
        )

    def test_path_alternates_and_finishes_on_contact_boundary(self):
        request = ConstantHeadingRequest(
            start_foot_scene_xy=torch.tensor(
                ((0.0, 0.12), (0.0, -0.12))
            ),
            start_support=torch.tensor((True, True)),
            heading_scene_xy=torch.tensor((0.6, 0.8)),
            distance_m=1.0,
            speed_mps=0.4,
            stride_m=0.25,
            step_width_m=0.24,
            frames_per_second=50.0,
        )
        path = nominal_footprint_path(request)
        self.assertGreaterEqual(len(path.footprints), 4)
        self.assertTrue(all(
            left.foot != right.foot
            for left, right in zip(path.footprints, path.footprints[1:])
        ))
        self.assertLessEqual(path.progress_m, request.distance_m)
        self.assertGreater(
            path.progress_m, request.distance_m - request.stride_m
        )
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_path -v
```

Expected: `ModuleNotFoundError` for
`mm_sonic.torch_heading_footprint_path`.

- [ ] **Step 3: Implement immutable request/path contracts and transforms**

Implement tensor-owning frozen dataclasses with strict shape, dtype, device,
finite-value, normalization, and positive-scalar validation. Use:

```python
def heading_basis(heading_scene_xy: torch.Tensor) -> torch.Tensor:
    forward = heading_scene_xy / torch.linalg.vector_norm(heading_scene_xy)
    lateral = torch.stack((-forward[1], forward[0]))
    return torch.stack((forward, lateral), dim=1)


def heading_local_to_scene(local, origin, heading):
    return origin + local @ heading_basis(heading).T


def scene_to_heading_local(scene, origin, heading):
    return (scene - origin) @ heading_basis(heading)
```

`nominal_footprint_path()` must choose the first moving foot from the actual
support mask, alternate feet, place centers at successive stride progress
with lateral offsets `+width/2` for foot 0 and `-width/2` for foot 1, compute
contact frame indices from `progress/speed*fps`, and stop at the last complete
step not beyond `distance_m`.

- [ ] **Step 4: Run the focused test and existing foothold tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_path \
  tests.python.test_sonic_torch_foothold_actions -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  sonic/python/mm_sonic/torch_heading_footprint_path.py \
  tests/python/test_sonic_torch_heading_footprint_path.py
git commit -m "feat: generate heading-local footprint paths"
```

---

### Task 2: Full-sole terrain candidate layers

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_footprint_candidates.py`
- Test: `tests/python/test_sonic_torch_terrain_footprint_candidates.py`
- Reference: `sonic/python/mm_sonic/torch_g1_sole_kinematics.py`

**Interfaces:**
- Consumes: `NominalFootprintPath`, per-foot sole XY samples in the sole frame, terrain `sample_surface(points_scene_xy)`, and bounded forward/lateral search offsets.
- Produces: `TerrainFootprintCandidate`, `TerrainFootprintLayer`, `FootprintCandidateFailure`, and `terrain_footprint_layers()`.

Use these exact fields:

```python
@dataclass(frozen=True)
class TerrainFootprintCandidate:
    step_index: int
    foot: int
    center_scene_xy: torch.Tensor
    center_heading_xy: torch.Tensor
    offset_heading_xy: torch.Tensor
    yaw_scene_rad: float
    surface_height_m: float
    placement_cost: float


@dataclass(frozen=True)
class TerrainFootprintLayer:
    step_index: int
    nominal: NominalFootprint
    candidates: tuple[TerrainFootprintCandidate, ...]


@dataclass(frozen=True)
class FootprintCandidateFailure(Exception):
    code: str
    step_index: int
    attempted_offsets: int
    reasons: tuple[str, ...]
```

- [ ] **Step 1: Write failing full-sole and rotation tests**

```python
import unittest

import torch

from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    nominal_footprint_path,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    terrain_footprint_layers,
)


class TerrainFootprintCandidateTests(unittest.TestCase):
    def test_center_valid_but_toe_over_edge_is_rejected(self):
        request = ConstantHeadingRequest(
            start_foot_scene_xy=torch.tensor(
                ((0.0, 0.12), (0.0, -0.12))
            ),
            start_support=torch.tensor((True, True)),
            heading_scene_xy=torch.tensor((1.0, 0.0)),
            distance_m=0.5,
            speed_mps=0.4,
            stride_m=0.25,
            step_width_m=0.24,
            frames_per_second=50.0,
        )
        path = nominal_footprint_path(request)
        sole = torch.tensor(
            ((-0.10, -0.04), (-0.10, 0.04), (0.12, -0.04), (0.12, 0.04))
        )

        def edge(points):
            return torch.where(
                points[..., 0] <= 0.30,
                torch.zeros(points.shape[:-1]),
                torch.full(points.shape[:-1], -0.20),
            )

        layers = terrain_footprint_layers(
            path=path,
            sole_offsets_by_foot=(sole, sole),
            sample_surface=edge,
            forward_offsets_m=(0.0, -0.05, -0.10),
            lateral_offsets_m=(0.0,),
            maximum_surface_variation_m=0.025,
            edge_safety_margin_m=0.01,
        )
        self.assertTrue(layers[0].candidates)
        self.assertLess(
            layers[0].candidates[0].center_scene_xy[0],
            path.footprints[0].center_scene_xy[0],
        )

    def test_global_rotation_preserves_local_candidate_offsets(self):
        sole = torch.tensor(
            ((-0.10, -0.04), (-0.10, 0.04), (0.12, -0.04), (0.12, 0.04))
        )
        paths = []
        for heading in (
            torch.tensor((1.0, 0.0)),
            torch.tensor((2.0 ** -0.5, 2.0 ** -0.5)),
        ):
            request = ConstantHeadingRequest(
                start_foot_scene_xy=torch.tensor(
                    ((0.0, 0.12), (0.0, -0.12))
                ),
                start_support=torch.tensor((True, True)),
                heading_scene_xy=heading,
                distance_m=0.5,
                speed_mps=0.4,
                stride_m=0.25,
                step_width_m=0.24,
                frames_per_second=50.0,
            )
            path = nominal_footprint_path(request)
            paths.append(terrain_footprint_layers(
                path=path,
                sole_offsets_by_foot=(sole, sole),
                sample_surface=lambda points: torch.zeros(
                    points.shape[:-1], dtype=points.dtype
                ),
                forward_offsets_m=(-0.05, 0.0, 0.05),
                lateral_offsets_m=(-0.05, 0.0, 0.05),
                maximum_surface_variation_m=0.025,
                edge_safety_margin_m=0.01,
            ))
        first = torch.stack([
            item.offset_heading_xy for item in paths[0][0].candidates
        ])
        second = torch.stack([
            item.offset_heading_xy for item in paths[1][0].candidates
        ])
        torch.testing.assert_close(first, second)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_footprint_candidates -v
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement vectorized oriented-sole enumeration**

For each nominal footprint and each `(forward_offset, lateral_offset)`:

```python
candidate_local = nominal.center_heading_xy + torch.tensor(
    (forward_offset, lateral_offset),
    dtype=nominal.center_scene_xy.dtype,
    device=nominal.center_scene_xy.device,
)
candidate_scene = heading_local_to_scene(
    candidate_local[None],
    path.origin_scene_xy,
    path.heading_scene_xy,
)[0]
sole_scene = candidate_scene + (
    sole_offsets_by_foot[nominal.foot]
    @ heading_basis(path.heading_scene_xy).T
)
height = sample_surface(sole_scene)
valid = bool(
    (height.max() - height.min() <= maximum_surface_variation_m).item()
)
```

Also query sole samples expanded outward by `edge_safety_margin_m`; reject the
candidate when any expanded sample differs from the supporting height by more
than the surface threshold. Sort accepted candidates by squared forward and
lateral offset, assign the median supporting height, and raise
`FootprintCandidateFailure(
code="no_terrain_footprint", step_index=nominal.step_index,
attempted_offsets=len(forward_offsets) * len(lateral_offsets),
reasons=("no complete sole footprint has one supporting surface",))` when a
layer is empty.

- [ ] **Step 4: Run candidate, sole, and terrain feature tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_footprint_candidates \
  tests.python.test_sonic_torch_g1_sole_kinematics \
  tests.python.test_sonic_torch_terrain_features -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  sonic/python/mm_sonic/torch_terrain_footprint_candidates.py \
  tests/python/test_sonic_torch_terrain_footprint_candidates.py
git commit -m "feat: search full-sole terrain footprints"
```

---

### Task 3: Contact-action beam search

**Files:**
- Create: `sonic/python/mm_sonic/torch_heading_footprint_search.py`
- Test: `tests/python/test_sonic_torch_heading_footprint_search.py`
- Modify: `sonic/python/mm_sonic/torch_foothold_actions.py`
- Modify: `tests/python/test_sonic_torch_foothold_actions.py`

**Interfaces:**
- Consumes: candidate layers, `FootholdActionIndex`, optional `FootholdTransitionGraph`, current joint/velocity state, requested speed, and beam thresholds.
- Produces: `FootprintActionEdge`, `HeadingFootprintPlan`, `HeadingPlanningFailure`, `action_edge_cost()`, and `search_heading_footprint_plan()`.

Use these exact fields:

```python
@dataclass(frozen=True)
class FootprintActionEdge:
    action_key: tuple[int, int]
    candidate_indices: tuple[int, int]
    descriptor_cost: float
    transition_cost: float


@dataclass(frozen=True)
class HeadingFootprintPlan:
    heading_scene_xy: torch.Tensor
    footprints: tuple[TerrainFootprintCandidate, ...]
    edges: tuple[FootprintActionEdge, ...]
    action_keys: tuple[tuple[int, int], ...]
    total_cost: float


@dataclass(frozen=True)
class HeadingPlanningFailure(Exception):
    code: str
    step_index: int
    candidate_count: int
    action_count: int
    reasons: tuple[str, ...]
```

- [ ] **Step 1: Write failing tests for non-greedy selection and heading invariance**

Create synthetic alternating layers and three `FootholdAction` objects:

```python
def test_beam_chooses_displaced_contact_with_compatible_successor(self):
    greedy_dead_end = action(
        key=(0, 10),
        landing_xy=((0.25, -0.12), (0.50, 0.12)),
    )
    compatible_first = action(
        key=(0, 20),
        landing_xy=((0.23, -0.10), (0.48, 0.10)),
    )
    successor = action(
        key=(0, 30),
        landing_xy=((0.25, -0.12), (0.50, 0.12)),
    )
    graph = graph_with_only_edges(((0, 20), (0, 30)))
    result = search_heading_footprint_plan(
        layers=three_candidate_layers(),
        action_index=index(greedy_dead_end, compatible_first, successor),
        transition_graph=graph,
        beam_width=4,
        xy_tolerance_m=0.10,
        height_tolerance_m=0.04,
        timing_tolerance_frames=8,
    )
    self.assertEqual(
        result.action_keys, ((0, 20), (0, 30))
    )
```

Add a second test that rotates every footprint and heading by 45 degrees while
leaving action descriptors in their local coordinates, then asserts identical
selected action keys and total cost.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_search -v
```

Expected: import failure for the new module.

- [ ] **Step 3: Add a public local-descriptor matcher**

Move the existing per-action contact comparison inside
`rank_foothold_actions()` into:

```python
def foothold_action_descriptor_cost(
    *,
    landing_feet: torch.Tensor,
    landing_xy_heading_m: torch.Tensor,
    landing_height_delta_m: torch.Tensor,
    landing_frame_offsets: torch.Tensor,
    action: FootholdAction,
    xy_tolerance_m: float,
    height_tolerance_m: float,
    timing_tolerance_frames: int,
) -> torch.Tensor:
    action_feet = torch.tensor(
        action.landing_feet,
        dtype=torch.int64,
        device=landing_feet.device,
    )
    if not bool((landing_feet == action_feet).all().item()):
        return torch.tensor(
            torch.inf, dtype=landing_xy_heading_m.dtype,
            device=landing_xy_heading_m.device,
        )
    xy = torch.linalg.vector_norm(
        landing_xy_heading_m - action.landing_xy_start_frame_m, dim=-1
    )
    height = torch.abs(
        landing_height_delta_m - action.landing_height_delta_m
    )
    action_time = torch.tensor(
        action.landing_frame_offsets,
        dtype=torch.int64,
        device=landing_frame_offsets.device,
    )
    timing = torch.abs(landing_frame_offsets - action_time)
    if bool(
        (xy > xy_tolerance_m).any().item()
        or (height > height_tolerance_m).any().item()
        or (timing > timing_tolerance_frames).any().item()
    ):
        return torch.tensor(
            torch.inf, dtype=landing_xy_heading_m.dtype,
            device=landing_xy_heading_m.device,
        )
    return (
        torch.square(xy / xy_tolerance_m).sum()
        + torch.square(height / height_tolerance_m).sum()
        + torch.square(
            timing.to(landing_xy_heading_m.dtype)
            / max(1, timing_tolerance_frames)
        ).sum()
    )
```

Return `torch.inf` when foot identity or any hard tolerance fails; otherwise
return the normalized squared XY, height, and timing error. Refactor
`rank_foothold_actions()` to call this function without changing its behavior.

- [ ] **Step 4: Implement deterministic layered beam search**

Advance the beam by the two alternating contacts represented by each
`FootholdAction`. For every beam state:

1. Enumerate candidate pairs in the next two layers.
2. Convert their displacement from the current stance into heading-local
   coordinates.
3. Evaluate actions with `foothold_action_descriptor_cost()`.
4. Apply `FootholdTransitionGraph.pair_eligibility()` between consecutive
   source windows.
5. Add footprint displacement, action descriptor, and boundary graph costs.
6. Retain the best `beam_width` states using `(cost, action_keys,
   candidate_indices)` as the deterministic ordering key.

Return `HeadingPlanningFailure(
code="no_motion_coverage", step_index=layer.step_index,
candidate_count=len(layer.candidates), action_count=len(action_index.actions),
reasons=("no contact action satisfies descriptor tolerances",))` when a layer
has terrain candidates but no action matches. Return
`HeadingPlanningFailure(
code="sequence_dead_end", step_index=layer.step_index,
candidate_count=len(layer.candidates), action_count=len(action_index.actions),
reasons=("transition graph removed every beam state",))` when later graph
compatibility removes every state.

- [ ] **Step 5: Run search and existing foothold tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_search \
  tests.python.test_sonic_torch_foothold_actions -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  sonic/python/mm_sonic/torch_foothold_actions.py \
  sonic/python/mm_sonic/torch_heading_footprint_search.py \
  tests/python/test_sonic_torch_foothold_actions.py \
  tests/python/test_sonic_torch_heading_footprint_search.py
git commit -m "feat: search contact actions along footprint paths"
```

---

### Task 4: Bounded motion realization

**Files:**
- Create: `sonic/python/mm_sonic/torch_heading_footprint_realizer.py`
- Test: `tests/python/test_sonic_torch_heading_footprint_realizer.py`
- Reuse: `sonic/python/mm_sonic/torch_horizontal_terrain_retarget.py`
- Reuse: `sonic/python/mm_sonic/torch_g1_fk.py`
- Reuse: `sonic/python/mm_sonic/torch_g1_sole_kinematics.py`

**Interfaces:**
- Consumes: `HeadingFootprintPlan`, source dataset clips, terrain extension,
  G1 XML, and retarget bounds.
- Produces: `RealizedHeadingTraversal`, `RealizationFailure`, and
  `realize_heading_footprint_plan()`.

Use these exact fields:

```python
@dataclass(frozen=True)
class RealizedHeadingTraversal:
    joint_position: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    source_support_mask: np.ndarray
    source_frame_provenance: np.ndarray
    maximum_stance_error_m: float
    minimum_sole_clearance_m: float


@dataclass(frozen=True)
class RealizationFailure(Exception):
    code: str
    edge_index: int
    source_action_key: tuple[int, int]
    reasons: tuple[str, ...]
```

- [ ] **Step 1: Write failing SE(2), stance-lock, and terminal-phase tests**

```python
def test_realizer_rotates_source_window_and_locks_stance_sole(self):
    plan = synthetic_plan(heading=(0.0, 1.0))
    result = realize_heading_footprint_plan(
        plan=plan,
        source=synthetic_two_window_source(),
        terrain=flat_terrain(),
        kinematics=synthetic_kinematics(),
        retargeter=recording_retargeter(),
    )
    self.assertLess(result.maximum_stance_error_m, 0.005)
    self.assertGreater(result.root_position_world[-1, 1], 0.4)
    self.assertAlmostEqual(result.root_position_world[-1, 0], 0.0, places=2)
    self.assertTrue(result.source_support_mask[-1].any())


def test_realizer_rejects_terminal_mid_swing(self):
    with self.assertRaisesRegex(RealizationFailure, "terminal_mid_swing"):
        realize_heading_footprint_plan(
            plan=synthetic_plan(),
            source=source_ending_before_touchdown(),
            terrain=flat_terrain(),
            kinematics=synthetic_kinematics(),
            retargeter=recording_retargeter(),
        )
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_realizer -v
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement source extraction and stance-aligned transforms**

For each selected `FootholdAction`, extract
`[start_frame:end_frame]`. Compute source start yaw, rotate root, feet, and
root orientation by `requested_yaw-source_yaw`, and translate so the active
source stance foot equals the selected predecessor footprint. Preserve source
frame count and authenticated support mask.

Use an explicit `source_frame_provenance` array of
`(clip_index, source_frame)` pairs. Reject a source window whose support start
does not match the accumulated support phase.

- [ ] **Step 4: Implement contact retargeting and concatenation**

Build fixed stance anchors for every source support interval. Map touchdown
events in order to the selected footprint centers and supporting heights.
Across each frame call `WideBoundG1TerrainRetargeter.solve_frame()` with:

```python
solve_feet = np.ones(2, dtype=np.bool_)
enforce_target_error_feet = support_mask[frame]
target_foot_position_weights = phase_contact_weights(
    support_mask=window_support,
    minimum_swing_weight=0.1,
    blend_frames=8,
)[local_frame] * 10.0
target_center_of_mass_world_xy = source_relative_com_targets(
    source_center_of_mass_world=source_center_of_mass,
    source_foot_position_world=source_feet,
    support_mask=window_support,
    stance_anchor_world=stance_anchors,
)[local_frame]
```

Use `anticipate_touchdown_positions()` for the swing target and the
rate-limited clearance envelope for terrain collision. Concatenate only at
complete contact boundaries, remove duplicate boundary frames, and reject
maximum joint step above `0.30 rad`, root step above `0.05 m`, stance origin
error above `0.02 m`, or unsupported frames outside source-authenticated
flight.

- [ ] **Step 5: Run realizer and retargeting tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_realizer \
  tests.python.test_sonic_torch_horizontal_terrain_retarget \
  tests.python.test_sonic_torch_flat_gait_contact_overlay -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  sonic/python/mm_sonic/torch_heading_footprint_realizer.py \
  tests/python/test_sonic_torch_heading_footprint_realizer.py
git commit -m "feat: realize footprint plans with bounded retargeting"
```

---

### Task 5: Offline planner runner and classified failures

**Files:**
- Create: `resources/run_g1_heading_footprint_planner.py`
- Test: `tests/python/test_run_g1_heading_footprint_planner.py`
- Modify: `resources/run_g1_validate_traversal.py`
- Modify: `tests/python/test_run_g1_validate_traversal.py`

**Interfaces:**
- Consumes: dataset/config/G1 XML, `--heading-degrees`, `--distance-m`,
  `--speed-mps`, start-state artifact, search offsets, and beam width.
- Produces: `traversal.npz`, `footprints.json`, `metrics.json`,
  `failure.json`, and process status `0` on validated success or `2` on a
  classified planning failure.

- [ ] **Step 1: Write parser and failure-schema tests**

```python
def test_parser_requires_heading_distance_and_speed(self):
    args = module.parser().parse_args([
        "--dataset", "dataset",
        "--config", "config.json",
        "--g1-xml", "g1.xml",
        "--start-state", "start.npz",
        "--heading-degrees", "45",
        "--distance-m", "1.0",
        "--speed-mps", "0.4",
        "--output", "out",
    ])
    self.assertEqual(args.heading_degrees, 45.0)
    self.assertEqual(args.distance_m, 1.0)
    self.assertFalse(hasattr(args, "lane_scene_y"))
    self.assertFalse(hasattr(args, "source_start"))


def test_failure_json_distinguishes_terrain_from_motion_coverage(self):
    record = module.failure_record(
        code="no_motion_coverage",
        heading_degrees=45.0,
        step_index=3,
        attempted_footprints=8,
        attempted_actions=512,
        reasons=("height tolerance",),
    )
    self.assertEqual(record["code"], "no_motion_coverage")
    self.assertEqual(record["step_index"], 3)
```

- [ ] **Step 2: Run runner tests and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_heading_footprint_planner -v
```

Expected: file import failure.

- [ ] **Step 3: Implement orchestration without fixed lanes or frames**

The runner must:

1. load the start state and measure actual foot centers/support;
2. convert `--heading-degrees` to a scene vector;
3. derive stride and width from median `FootholdAction` landing descriptors;
4. call `nominal_footprint_path()`;
5. obtain the actual G1 sole XY template from
   `MujocoG1SoleKinematics` at the start pose;
6. call `terrain_footprint_layers()`;
7. build/reuse `FootholdActionIndex` and `FootholdTransitionGraph`;
8. call `search_heading_footprint_plan()`;
9. call `realize_heading_footprint_plan()`;
10. save the artifact and invoke independent validation.

Expose only generic bounds:

```python
parser.add_argument("--heading-degrees", type=float, required=True)
parser.add_argument("--distance-m", type=float, required=True)
parser.add_argument("--speed-mps", type=float, required=True)
parser.add_argument("--forward-search-m", type=float, default=0.10)
parser.add_argument("--lateral-search-m", type=float, default=0.10)
parser.add_argument("--search-resolution-m", type=float, default=0.025)
parser.add_argument("--beam-width", type=int, default=16)
```

- [ ] **Step 4: Extend independent validation metrics**

Add optional expected heading and planned footprint inputs to
`run_g1_validate_traversal.py`. Report maximum heading error, progress error,
complete-sole contact error, worst frame, worst footprint, terminal support,
and provenance coverage. Reject a terminal unsupported/mid-swing artifact and
any declared support with fewer than the configured sole sample count.

- [ ] **Step 5: Run runner and validator tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_heading_footprint_planner \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add \
  resources/run_g1_heading_footprint_planner.py \
  resources/run_g1_validate_traversal.py \
  tests/python/test_run_g1_heading_footprint_planner.py \
  tests/python/test_run_g1_validate_traversal.py
git commit -m "feat: run heading-general footprint planning"
```

---

### Task 6: Multi-heading acceptance and visualization

**Files:**
- Create: `resources/run_g1_heading_footprint_acceptance.py`
- Test: `tests/python/test_run_g1_heading_footprint_acceptance.py`
- Modify: `docs/g1-horizontal-contact-traversal.md`

**Interfaces:**
- Consumes: the planner runner configuration and headings `0`, `45`, and
  `90` degrees relative to the scene X axis.
- Produces: one acceptance summary JSON, per-heading artifacts, and commands
  compatible with `run_g1_stair_pivot_viewer.py`.

- [ ] **Step 1: Write failing acceptance aggregation test**

```python
def test_acceptance_requires_three_validated_distinct_headings(self):
    summary = module.acceptance_summary((
        result(heading=0.0, validated=True),
        result(heading=45.0, validated=True),
        result(heading=90.0, validated=True),
    ))
    self.assertTrue(summary["accepted"])
    self.assertEqual(summary["validated_heading_count"], 3)


def test_acceptance_rejects_heading_specific_runtime_fields(self):
    with self.assertRaisesRegex(ValueError, "forbidden runtime field"):
        module.reject_heading_specific_fields({
            "lane_scene_y": (0.32, 0.12)
        })
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_heading_footprint_acceptance -v
```

Expected: file import failure.

- [ ] **Step 3: Implement acceptance matrix**

Run the same planner arguments at headings `0`, `45`, and `90`, changing only
`--heading-degrees` and output directory. Also run one shifted start pose and
one perturbed synthetic stair grid. Aggregate:

```json
{
  "schema": "g1-heading-footprint-acceptance/v1",
  "accepted": true,
  "validated_heading_count": 3,
  "results": [],
  "forbidden_runtime_fields": []
}
```

Acceptance requires three validated distinct headings, zero forbidden fields,
complete terminal support, complete provenance, and no contact or continuity
threshold violation. A classified `no_motion_coverage` result remains a valid
planner failure record but does not count as a successful heading.

- [ ] **Step 4: Run the full focused suite**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_path \
  tests.python.test_sonic_torch_terrain_footprint_candidates \
  tests.python.test_sonic_torch_heading_footprint_search \
  tests.python.test_sonic_torch_heading_footprint_realizer \
  tests.python.test_run_g1_heading_footprint_planner \
  tests.python.test_run_g1_heading_footprint_acceptance \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the real acceptance matrix**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_heading_footprint_acceptance.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --start-state build/g1-step-up-search/horizontal-current-check/step-up.npz \
  --distance-m 1.0 \
  --speed-mps 0.4 \
  --output build/g1-heading-footprint-acceptance-v1
```

Expected: `summary.json` records at least three independently validated
headings or identifies exactly which heading lacks terrain or corpus coverage.
Do not mark the feature complete when fewer than three headings validate.

- [ ] **Step 6: Document reproduction and viewer commands**

Replace the fixed-lane rebuild section in
`docs/g1-horizontal-contact-traversal.md` with the general runner and show
examples for headings `0`, `45`, and `90`. State that the old v47 artifact is
only a regression reference.

- [ ] **Step 7: Commit Task 6**

```bash
git add \
  resources/run_g1_heading_footprint_acceptance.py \
  tests/python/test_run_g1_heading_footprint_acceptance.py \
  docs/g1-horizontal-contact-traversal.md
git commit -m "test: qualify heading-general footprint planning"
```

---

## Final verification

- [ ] Run `git diff --check`.
- [ ] Run every focused test command from Tasks 1–6.
- [ ] Run the real three-heading acceptance matrix.
- [ ] Inspect the three generated traversals in the MuJoCo kinematic viewer.
- [ ] Confirm the planner CLI contains none of:

```text
lane_scene_y
source_start
source_stop
touchdown_frames
drop_frame
gait_frame
```

- [ ] Confirm every success artifact contains planned footprints, complete
source provenance, independent validation metrics, and terminal support.
- [ ] Request code review before merging or pushing the implementation branch.
