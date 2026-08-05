# G1 Contact-Path Terrain Motion Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate natural, terrain-valid G1 kinematics from an angle-general terrain-projected contact path using GRAIL-native global retargeting, with ARDY as an optional comparison backend.

**Architecture:** A pure geometry module projects a normal alternating gait onto two command-local terrain lanes and inserts contacts at missed support-height transitions. A separate root-path module derives smooth pelvis waypoints. The primary backend globally retargets an authenticated GRAIL gait to the complete root/contact schedule; an isolated ARDY adapter receives the same constraints as an A/B candidate. Existing MuJoCo validation and visual-quality review decide acceptance.

**Tech Stack:** Python 3.11, NumPy, PyTorch, official ARDY G1 at pinned commit `693f74d13b3d04a0a22ce127ee79c929dd89756b`, MuJoCo G1 FK/sole kinematics, `unittest`.

## Global Constraints

- Kinematics only; do not integrate Sonic or a tracking controller.
- ARDY runs at 25 Hz and accepted output is resampled to 50 Hz.
- The released ARDY G1 model was trained on Bones Rigplay 1, not GRAIL; ARDY is optional and must not block the GRAIL-native backend.
- Do not modify the official ARDY repository.
- Do not classify terrain as stairs, curbs, ramps, blocks, beams, or edges.
- Do not hard-code a terrain coordinate or approach heading.
- Every declared stance must have full-sole support and bounded planted-foot drift.
- Contact planning must be independent of GRAIL action retrieval and ARDY generation.
- ARDY candidates are proposals; independent MuJoCo validation decides acceptance.
- Preserve all existing worktree changes and stage only files named by the active task.

---

### Task 1: Angle-General Contact-Path Projection

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_contact_path.py`
- Create: `tests/python/test_sonic_torch_terrain_contact_path.py`

**Interfaces:**
- Consumes: a callable full-sole terrain sampler, start center, unit heading, gait stride/width, distance, and search limits.
- Produces: `TerrainContact`, `TerrainContactSchedule`, and `project_terrain_contact_path(...)`.

- [ ] **Step 1: Write failing tests for nominal flat contacts and rotation equivariance**

```python
import math
import unittest
import torch

from mm_sonic.torch_terrain_contact_path import project_terrain_contact_path


class FlatTerrain:
    def full_sole_support(self, centers_xy, yaw_rad):
        count = len(centers_xy)
        return (
            torch.zeros(count),
            torch.ones(count, dtype=torch.bool),
            torch.full((count,), 0.05),
        )


class TerrainContactPathTests(unittest.TestCase):
    def test_flat_path_preserves_alternating_nominal_gait(self):
        result = project_terrain_contact_path(
            terrain=FlatTerrain(),
            start_center_scene_xy=torch.tensor((0.0, 0.0)),
            heading_scene_xy=torch.tensor((1.0, 0.0)),
            distance_m=0.9,
            nominal_stride_m=0.30,
            nominal_step_width_m=0.22,
            nominal_step_frames=12,
            progress_search_m=0.12,
            lateral_search_m=0.04,
            search_resolution_m=0.02,
            transition_height_m=0.04,
        )
        self.assertEqual(tuple(c.foot for c in result.contacts), (0, 1, 0, 1, 0, 1))
        torch.testing.assert_close(
            torch.stack([c.center_heading_sl for c in result.contacts])[:, 0],
            torch.tensor((0.15, 0.30, 0.45, 0.60, 0.75, 0.90)),
        )

    def test_rotating_command_rotates_contacts_without_changing_local_plan(self):
        forward = project_terrain_contact_path(
            terrain=FlatTerrain(),
            start_center_scene_xy=torch.zeros(2),
            heading_scene_xy=torch.tensor((1.0, 0.0)),
            distance_m=0.6,
            nominal_stride_m=0.30,
            nominal_step_width_m=0.22,
            nominal_step_frames=12,
            progress_search_m=0.12,
            lateral_search_m=0.04,
            search_resolution_m=0.02,
            transition_height_m=0.04,
        )
        diagonal = project_terrain_contact_path(
            terrain=FlatTerrain(),
            start_center_scene_xy=torch.zeros(2),
            heading_scene_xy=torch.tensor((math.sqrt(0.5), math.sqrt(0.5))),
            distance_m=0.6,
            nominal_stride_m=0.30,
            nominal_step_width_m=0.22,
            nominal_step_frames=12,
            progress_search_m=0.12,
            lateral_search_m=0.04,
            search_resolution_m=0.02,
            transition_height_m=0.04,
        )
        torch.testing.assert_close(
            torch.stack([c.center_heading_sl for c in forward.contacts]),
            torch.stack([c.center_heading_sl for c in diagonal.contacts]),
        )
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_contact_path -v
```

Expected: `ModuleNotFoundError: No module named 'mm_sonic.torch_terrain_contact_path'`.

- [ ] **Step 3: Implement immutable contact types and flat-path projection**

```python
"""Project a normal alternating gait onto full-sole terrain support."""

from dataclasses import dataclass
import math
import torch
from .joints import ContractError


@dataclass(frozen=True)
class TerrainContact:
    foot: int
    frame: int
    center_scene_xy: torch.Tensor
    center_heading_sl: torch.Tensor
    yaw_scene_rad: float
    surface_height_m: float
    support_margin_m: float
    inserted_for_transition: bool = False


@dataclass(frozen=True)
class TerrainContactSchedule:
    origin_scene_xy: torch.Tensor
    heading_scene_xy: torch.Tensor
    contacts: tuple[TerrainContact, ...]
    frame_count: int


def _basis(heading):
    unit = heading / torch.linalg.vector_norm(heading)
    lateral = torch.stack((-unit[1], unit[0]))
    return unit, lateral


def project_terrain_contact_path(
    *,
    terrain,
    start_center_scene_xy,
    heading_scene_xy,
    distance_m,
    nominal_stride_m,
    nominal_step_width_m,
    nominal_step_frames,
    progress_search_m,
    lateral_search_m,
    search_resolution_m,
    transition_height_m,
):
    origin = torch.as_tensor(start_center_scene_xy).clone()
    heading, lateral = _basis(torch.as_tensor(heading_scene_xy))
    step_advance = nominal_stride_m / 2.0
    count = int(math.ceil(distance_m / step_advance))
    contacts = []
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    for index in range(count):
        foot = index % 2
        progress = min(distance_m, (index + 1) * step_advance)
        side = nominal_step_width_m * (0.5 if foot == 0 else -0.5)
        local = torch.tensor((progress, side), dtype=origin.dtype, device=origin.device)
        center = origin + local[0] * heading + local[1] * lateral
        height, valid, margin = terrain.full_sole_support(center[None], yaw)
        if not bool(valid[0]):
            raise ContractError("no full-sole placement near nominal contact")
        contacts.append(TerrainContact(
            foot=foot,
            frame=(index + 1) * nominal_step_frames,
            center_scene_xy=center,
            center_heading_sl=local,
            yaw_scene_rad=yaw,
            surface_height_m=float(height[0]),
            support_margin_m=float(margin[0]),
        ))
    return TerrainContactSchedule(
        origin_scene_xy=origin,
        heading_scene_xy=heading,
        contacts=tuple(contacts),
        frame_count=contacts[-1].frame + nominal_step_frames,
    )
```

- [ ] **Step 4: Run the flat-path tests**

Run the command from Step 2.

Expected: both tests pass.

- [ ] **Step 5: Add failing tests for full-sole snapping and transition insertion**

```python
class StepTerrain:
    def full_sole_support(self, centers_xy, yaw_rad):
        x = centers_xy[:, 0]
        near_edge = (x > 0.44) & (x < 0.56)
        height = torch.where(x >= 0.50, 0.18, 0.0)
        return height, ~near_edge, torch.where(near_edge, 0.0, torch.full_like(x, 0.04))


def test_contact_snaps_away_from_invalid_full_sole_edge(self):
    result = project_terrain_contact_path(
        terrain=StepTerrain(),
        start_center_scene_xy=torch.zeros(2),
        heading_scene_xy=torch.tensor((1.0, 0.0)),
        distance_m=0.9,
        nominal_stride_m=0.30,
        nominal_step_width_m=0.22,
        nominal_step_frames=12,
        progress_search_m=0.12,
        lateral_search_m=0.04,
        search_resolution_m=0.02,
        transition_height_m=0.04,
    )
    self.assertTrue(all(not 0.44 < float(c.center_scene_xy[0]) < 0.56 for c in result.contacts))


def test_skipped_height_change_inserts_transition_contact(self):
    result = project_terrain_contact_path(
        terrain=StepTerrain(),
        start_center_scene_xy=torch.zeros(2),
        heading_scene_xy=torch.tensor((1.0, 0.0)),
        distance_m=0.9,
        nominal_stride_m=0.60,
        nominal_step_width_m=0.22,
        nominal_step_frames=12,
        progress_search_m=0.12,
        lateral_search_m=0.04,
        search_resolution_m=0.02,
        transition_height_m=0.04,
    )
    self.assertTrue(any(c.inserted_for_transition for c in result.contacts))
    self.assertTrue(any(c.surface_height_m < 0.04 for c in result.contacts))
    self.assertTrue(any(c.surface_height_m > 0.14 for c in result.contacts))
```

- [ ] **Step 6: Run the new tests and verify behavioral failures**

Expected: snapping fails because nominal invalid contacts raise, and insertion fails because the output has only nominal contacts.

- [ ] **Step 7: Implement deterministic local search and transition insertion**

Add helpers that:

```python
def _offset_grid(radius, resolution, *, device, dtype):
    count = int(math.floor(radius / resolution))
    return torch.arange(-count, count + 1, device=device, dtype=dtype) * resolution


def _best_full_sole_candidate(terrain, nominal_scene, heading, lateral, yaw, progress_radius, lateral_radius, resolution):
    ds = _offset_grid(progress_radius, resolution, device=nominal_scene.device, dtype=nominal_scene.dtype)
    dl = _offset_grid(lateral_radius, resolution, device=nominal_scene.device, dtype=nominal_scene.dtype)
    pairs = torch.cartesian_prod(ds, dl)
    centers = nominal_scene + pairs[:, :1] * heading + pairs[:, 1:] * lateral
    height, valid, margin = terrain.full_sole_support(centers, yaw)
    score = torch.square(pairs[:, 0] / progress_radius) + torch.square(
        pairs[:, 1] / max(lateral_radius, resolution)
    ) - 0.05 * margin
    score = torch.where(valid, score, torch.full_like(score, torch.inf))
    best = int(torch.argmin(score))
    if not torch.isfinite(score[best]):
        raise ContractError("no full-sole placement near nominal contact")
    return centers[best], pairs[best], height[best], margin[best]
```

Sample each foot lane at `search_resolution_m`, identify adjacent valid samples
whose support heights differ by at least `transition_height_m`, and ensure the
contact sequence contains valid placements on both sides of every crossed
event. Insert the missing placement at the closest safe sample, preserve
alternation when possible, renumber frames at `nominal_step_frames`, and mark
`inserted_for_transition=True`.

- [ ] **Step 8: Run all Task 1 tests and commit**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_contact_path -v
```

Expected: all four tests pass.

Commit:

```bash
git add sonic/python/mm_sonic/torch_terrain_contact_path.py \
  tests/python/test_sonic_torch_terrain_contact_path.py
git commit -m "feat: project gait contacts onto terrain"
```

---

### Task 2: Contact Intervals and Smooth Root Path

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_root_path.py`
- Create: `tests/python/test_sonic_torch_terrain_root_path.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_contact_path.py`
- Modify: `tests/python/test_sonic_torch_terrain_contact_path.py`

**Interfaces:**
- Consumes: `TerrainContactSchedule`, stance/double-support timing, nominal pelvis clearance.
- Produces: `support_mask_from_contacts(...)` and `terrain_root_path(...) -> TerrainRootPath`.

- [ ] **Step 1: Write failing tests**

```python
import unittest
import torch
from mm_sonic.torch_terrain_root_path import (
    support_mask_from_contacts,
    terrain_root_path,
)


class TerrainRootPathTests(unittest.TestCase):
    def test_touchdown_expands_to_fixed_stance_interval(self):
        schedule = make_schedule_for_test(
            frames=(8, 16, 24, 32),
            feet=(0, 1, 0, 1),
            heights=(0.0, 0.18, 0.18, 0.0),
        )
        support = support_mask_from_contacts(schedule, stance_frames=14)
        self.assertEqual(tuple(support.shape), (schedule.frame_count, 2))
        self.assertTrue(support.any(dim=1).all())
        self.assertTrue(bool((support.sum(dim=1) == 2).any()))

    def test_root_height_smooths_a_support_discontinuity(self):
        schedule = make_schedule_for_test(
            frames=(8, 16, 24, 32),
            feet=(0, 1, 0, 1),
            heights=(0.0, 0.18, 0.18, 0.0),
        )
        result = terrain_root_path(
            schedule,
            nominal_pelvis_clearance_m=0.72,
            stance_frames=14,
            smoothing_passes=3,
        )
        self.assertLess(float(torch.diff(result.position_scene_xyz[:, 2]).abs().max()), 0.06)
        torch.testing.assert_close(
            torch.linalg.vector_norm(result.heading_scene_xy),
            torch.tensor(1.0),
        )
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Use `python -m unittest tests.python.test_sonic_torch_terrain_root_path -v`.

- [ ] **Step 3: Implement support intervals and root path**

Create `TerrainRootPath(position_scene_xyz, heading_scene_xy, support_mask)`.
Reconstruct each foot's held stance position from consecutive contacts. Use the
support-foot midpoint for planar root targets, linear interpolation between
contact frames, and support-height plus pelvis clearance for height targets.
Apply the normalized five-tap kernel `(1, 4, 6, 4, 1) / 16` for the configured
number of smoothing passes while preserving the first and last root samples.

- [ ] **Step 4: Run both contact and root-path tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_contact_path \
  tests.python.test_sonic_torch_terrain_root_path -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_contact_path.py \
  sonic/python/mm_sonic/torch_terrain_root_path.py \
  tests/python/test_sonic_torch_terrain_contact_path.py \
  tests/python/test_sonic_torch_terrain_root_path.py
git commit -m "feat: derive terrain contact and root timing"
```

---

### Task 3: GRAIL-Native Globally Regularized Retargeting

**Files:**
- Create: `sonic/python/mm_sonic/torch_grail_contact_trajectory.py`
- Create: `tests/python/test_sonic_torch_grail_contact_trajectory.py`

**Interfaces:**
- Consumes: a 50 Hz `TerrainContactSchedule`, `TerrainRootPath`, authenticated
  GRAIL gait frames, G1 XML, and sole kinematics.
- Produces: `GrailContactTrajectory` with root/joints/support plus objective and
  constraint metrics.

- [ ] **Step 1: Write failing tests for reference preservation and stance lock**

Create a short synthetic two-contact gait and a fake linear foot-kinematics
adapter. Assert that `retarget_grail_gait_to_contact_path(...)`:

```python
result = retarget_grail_gait_to_contact_path(
    schedule=schedule,
    root_path=root_path,
    reference=reference,
    kinematics=fake_kinematics,
    maximum_iterations=40,
)
self.assertEqual(result.joint_position.shape, reference.joint_position.shape)
self.assertLess(result.maximum_stance_position_error_m, 1.0e-3)
self.assertLess(result.maximum_stance_yaw_error_rad, 1.0e-3)
self.assertLess(
    torch.linalg.vector_norm(
        torch.diff(result.joint_position, dim=0)
        - torch.diff(reference.joint_position, dim=0)
    ),
    0.25,
)
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_grail_contact_trajectory -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement reference tiling and explicit residual blocks**

Create immutable `GrailGaitReference` and `GrailContactTrajectory` types.
Phase-align and tile the source gait to `schedule.frame_count`. Build residual
blocks for reference pose, reference joint velocity, root path, stance heel/toe
position, stance sole yaw, swing clearance, and joint acceleration. Normalize
every block by its declared tolerance so no residual silently dominates due to
units.

- [ ] **Step 4: Implement overlapping-window least-squares solve**

Optimize root XYZ and 29 joint values over contact-to-contact windows with one
contact interval of overlap. Use the existing MuJoCo G1 FK/Jacobian adapter,
box bounds from the G1 XML, and the previous accepted window as initialization.
Keep stance heel/toe and yaw constraints hard through rejection thresholds.
Blend only duplicated overlap variables inside the optimizer; do not post-hoc
blend planted feet.

- [ ] **Step 5: Add failing tests for swing clearance and whole-sole orientation**

Use terrain with a narrow raised obstacle and assert the optimized swing heel,
ankle, and toe all clear it. Add a regression where ankle position is valid but
toe yaw is wrong and assert the candidate is rejected.

- [ ] **Step 6: Implement clearance waypoints and hard acceptance**

Sample terrain beneath heel, ankle, toe, and lateral sole points throughout
swing. Add clearance residuals only where samples violate the scheduled
clearance envelope. Reject after optimization if stance position/yaw, swing
clearance, joint limits, or continuity exceed their hard thresholds.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_grail_contact_trajectory -v
```

Commit:

```bash
git add sonic/python/mm_sonic/torch_grail_contact_trajectory.py \
  tests/python/test_sonic_torch_grail_contact_trajectory.py
git commit -m "feat: globally retarget GRAIL gait to contact paths"
```

---

### Task 4: Optional Official ARDY G1 Constraint Adapter

**Files:**
- Create: `sonic/python/mm_sonic/torch_ardy_contact_schedule.py`
- Create: `tests/python/test_sonic_torch_ardy_contact_schedule.py`
- Create: `resources/run_g1_ardy_contact_schedule.py`
- Create: `tests/python/test_run_g1_ardy_contact_schedule.py`

**Interfaces:**
- Consumes: a 50 Hz `TerrainContactSchedule`, `TerrainRootPath`, a clean G1 flat-gait seed, and the G1 XML.
- Produces: an ARDY JSON constraint file at 25 Hz and a generated candidate directory containing `.npz` and `.csv`.

- [ ] **Step 1: Write failing coordinate and downsampling tests**

```python
import unittest
import numpy as np
import torch
from mm_sonic.torch_ardy_contact_schedule import (
    mujoco_xyz_to_ardy_xyz,
    schedule_frames_50hz_to_ardy_25hz,
)


class ArdyContactScheduleTests(unittest.TestCase):
    def test_axis_conversion_maps_mujoco_z_up_to_ardy_y_up(self):
        result = mujoco_xyz_to_ardy_xyz(torch.tensor(((1.0, 2.0, 3.0),)))
        torch.testing.assert_close(result[:, 1], torch.tensor((3.0,)))
        torch.testing.assert_close(torch.linalg.vector_norm(result), torch.tensor(math.sqrt(14.0)))

    def test_contact_frames_round_to_unique_25hz_indices(self):
        result = schedule_frames_50hz_to_ardy_25hz((0, 11, 12, 25))
        self.assertEqual(result, (0, 6, 12))
```

- [ ] **Step 2: Verify both tests fail because the adapter is missing**

Run `python -m unittest tests.python.test_sonic_torch_ardy_contact_schedule -v`.

- [ ] **Step 3: Implement coordinate conversion and frame mapping**

Use the same fixed frame rotation already authenticated by
`resources/export_g1_ardy_transition_constraints.py`. Keep the conversion in a
small pure module and test round trips. Map 50 Hz frame `f` to
`round(f * 25 / 50)`, deduplicate while preserving order, and reject schedules
whose distinct contact events collapse onto one ARDY frame.

- [ ] **Step 4: Write failing constraint-content tests**

Create a fake schedule and seed pose and assert:

```python
constraints = build_ardy_contact_constraints(...)
self.assertEqual(type(constraints[0]).__name__, "Root2DConstraintSet")
self.assertEqual(type(constraints[-1]).__name__, "FullBodyConstraintSet")
self.assertTrue(any(type(c).__name__ == "LeftFootConstraintSet" for c in constraints))
self.assertTrue(any(type(c).__name__ == "RightFootConstraintSet" for c in constraints))
self.assertGreater(len(constraints[0].frame_indices), 2)
```

Also assert each stance foot constraint contains both ankle and toe positions,
constant world foot yaw, coherent root XY, and the root height from
`TerrainRootPath`.

- [ ] **Step 5: Implement coherent sparse key-pose construction**

Reuse `_qpos_to_ardy(...)` logic from
`resources/export_g1_ardy_transition_constraints.py` in the adapter. For every
stance sample:

1. select the nearest phase-matched flat-gait pose;
2. transform its root to the planned root waypoint;
3. use existing bounded G1 IK to place the constrained stance foot;
4. convert the complete solved pose to ARDY global joints;
5. create a foot constraint without hard-coding terrain coordinates;
6. create exact full-body start/end constraints.

Use `Root2DConstraintSet` for planar root/heading samples. Save with
`save_constraints_lst`. Do not patch ARDY.

- [ ] **Step 6: Implement and test the isolated ARDY worker**

`resources/run_g1_ardy_contact_schedule.py` must:

- validate the pinned ARDY checkout commit;
- write the constraints file;
- call `/home/ubuntu/projects/ardy/.venv/bin/python scripts/generate.py`;
- request `g152`, the supplied duration, prompt `"a person walks forward"`,
  multiple seeds, and one chosen GPU;
- save stdout, stderr, command, commit, model, seed, and return code;
- fail if no candidate `.npz` and `.csv` are produced.

Test command construction with `unittest.mock`; do not invoke the model in the
unit test.

- [ ] **Step 7: Run adapter and worker unit tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_ardy_contact_schedule \
  tests.python.test_run_g1_ardy_contact_schedule -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add sonic/python/mm_sonic/torch_ardy_contact_schedule.py \
  resources/run_g1_ardy_contact_schedule.py \
  tests/python/test_sonic_torch_ardy_contact_schedule.py \
  tests/python/test_run_g1_ardy_contact_schedule.py
git commit -m "feat: condition ARDY on terrain contact paths"
```

---

### Task 5: End-to-End Candidate Generation and Honest Validation

**Files:**
- Create: `resources/run_g1_contact_path_terrain_generation.py`
- Create: `tests/python/test_run_g1_contact_path_terrain_generation.py`
- Modify: `resources/run_g1_validate_traversal.py`
- Modify: `tests/python/test_run_g1_validate_traversal.py`
- Create: `docs/superpowers/results/2026-08-05-g1-contact-path-ardy-terrain-generation.md`

**Interfaces:**
- Consumes: dataset/config, start state, heading, distance, GRAIL gait seed, G1 XML, and optional ARDY sample count/GPU.
- Produces: `contact-schedule.npz`, `root-path.npz`, the GRAIL-native candidate, optional raw ARDY candidates, validation JSON per candidate, ranked accepted candidates, and `best/traversal.npz`.

- [ ] **Step 1: Write failing CLI-contract and validation tests**

Test that the runner parser requires no terrain-action key and accepts
`--heading-degrees`, `--distance-m`, `--grail-gait-seed`,
`--ardy-samples`, and `--device`. Assert `--ardy-samples 0` is valid and
still produces the primary GRAIL-native job.
Extend traversal validation with hard checks:

```python
self.assertEqual(report["unsupported_declared_stance_frames"], 0)
self.assertLessEqual(report["maximum_planted_foot_drift_m"], 0.010)
self.assertGreaterEqual(report["minimum_swing_sole_clearance_m"], -0.005)
self.assertLessEqual(report["maximum_joint_step_rad"], 0.30)
self.assertLessEqual(report["maximum_root_step_m"], 0.04)
self.assertEqual(report["frozen_command_windows"], 0)
```

- [ ] **Step 2: Run tests and verify missing parser/metrics failures**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_contact_path_terrain_generation \
  tests.python.test_run_g1_validate_traversal -v
```

- [ ] **Step 3: Implement the orchestration runner**

The runner must:

1. resolve the existing height grid and alignment;
2. create the angle-general full-sole contact schedule;
3. derive the root path;
4. globally retarget the authenticated GRAIL gait;
5. optionally export ARDY constraints and generate seeds on assigned GPUs;
6. convert optional ARDY candidates to project G1 qpos;
7. run bounded repair only for candidates within repair thresholds;
8. validate every candidate with the identical contract;
9. rank only hard-valid candidates;
10. write a failure summary when none pass.

The manifest must include every input, seed, command, source commit, constraint
density, repair magnitude, validation metrics, and rejection reason.

- [ ] **Step 4: Run the focused unit suite**

Use the command from Step 2 plus the Task 1–3 tests.

Expected: all tests pass.

- [ ] **Step 5: Generate horizontal candidates**

Run the runner with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_contact_path_terrain_generation.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --start-state build/g1-heading-footprint-route-v30/heading-0/traversal.npz \
  --grail-gait-seed build/torch-grail-flat-current \
  --heading-degrees 0 --distance-m 1.20 \
  --ardy-samples 16 --device cuda \
  --output build/g1-contact-path-ardy/horizontal
```

Expected: `best/traversal.npz` exists only if all hard validation thresholds
pass.

- [ ] **Step 6: Generate 45-degree candidates**

Repeat Step 5 with `--heading-degrees -45` and output
`build/g1-contact-path-ardy/neg45`.

Expected: the same algorithm and configuration run without a heading-specific
branch.

- [ ] **Step 7: Render both candidates and record visual-quality decisions**

Render fixed-camera videos and contact sheets with the existing MuJoCo viewer.
Record objective metrics plus explicit review flags for foot floating, crouch
shuffle, inward ankle, lunge, sliding dismount, and heading drift. Reject a
candidate with any severe visual flag even when contact metrics pass.

- [ ] **Step 8: Commit orchestration and results**

```bash
git add resources/run_g1_contact_path_terrain_generation.py \
  resources/run_g1_validate_traversal.py \
  tests/python/test_run_g1_contact_path_terrain_generation.py \
  tests/python/test_run_g1_validate_traversal.py \
  docs/superpowers/results/2026-08-05-g1-contact-path-ardy-terrain-generation.md
git commit -m "feat: generate and validate contact-path terrain motion"
```

---

### Task 6: Heading and Terrain-Transform Generalization Sweep

**Files:**
- Create: `resources/run_g1_contact_path_generalization_sweep.py`
- Create: `tests/python/test_run_g1_contact_path_generalization_sweep.py`
- Modify: `docs/superpowers/results/2026-08-05-g1-contact-path-ardy-terrain-generation.md`

**Interfaces:**
- Consumes: the Task 5 runner and a list of headings/transforms.
- Produces: a single sweep manifest and comparison table without changing planner parameters between headings.

- [ ] **Step 1: Write a failing dry-run matrix test**

Assert that headings `range(-90, 91, 15)` and two rigid terrain transforms
produce 26 unique commands, all sharing one configuration hash, and that no
command contains a clip ID, staircase label, or heading-specific parameter.

- [ ] **Step 2: Run and verify the missing-runner failure**

Run `python -m unittest tests.python.test_run_g1_contact_path_generalization_sweep -v`.

- [ ] **Step 3: Implement the sweep runner**

Generate deterministic subprocess commands, assign jobs round-robin over GPUs
0–7, cap one ARDY process per GPU, and aggregate accepted/rejected counts and
failure classes. Refuse to aggregate outputs whose configuration hashes differ.

- [ ] **Step 4: Run the dry-run test and a contact-planning-only sweep**

First run the unit test. Then run the sweep with `--contact-plan-only` to verify
all headings preserve full-sole geometry before spending ARDY compute.

- [ ] **Step 5: Run selected full synthesis jobs and update results**

Run full synthesis for `-90, -45, 0, 45, 90` degrees and both terrain
transforms. Record hard metrics, visual flags, generation time, and classified
failures. Do not report generality from the horizontal and diagonal examples
alone.

- [ ] **Step 6: Commit**

```bash
git add resources/run_g1_contact_path_generalization_sweep.py \
  tests/python/test_run_g1_contact_path_generalization_sweep.py \
  docs/superpowers/results/2026-08-05-g1-contact-path-ardy-terrain-generation.md
git commit -m "test: sweep terrain contact generation headings"
```
