# G1 Action-Conditioned Terrain Contact Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline G1 terrain planner whose graph edges are complete, terrain-feasible GRAIL contact actions rather than independently selected footprint endpoints.

**Architecture:** Canonical action sweeps are transformed from each graph node into an arbitrary commanded frame. Bounded landing corrections are sampled against the raw height field, a batched full-sweep filter rejects contact and collision defects, and exact one-edge realization supplies successor nodes to a deterministic best-first search. A separate validator accepts only complete, natural, provenance-covered routes.

**Tech Stack:** Python 3.10, PyTorch, NumPy, MuJoCo kinematics, existing `FootholdActionIndex`, existing bounded G1 terrain retargeter, `unittest`.

## Global Constraints

- The runtime may not name an angle, lane, staircase, curb, or object class.
- Constant heading, offline kinematic planning, and 50 Hz output are the current scope.
- All planar action features use a command-local SE(2) frame.
- Every declared contact must support the complete sampled sole.
- Every graph edge must pass swept swing-sole feasibility before search may retain it.
- Exact realization failures are cached and may not be repeatedly selected.
- Retargeting remains bounded; contact labels may not be changed to obtain a pass.
- Unreachable terrain returns a classified failure with rejection counts.
- Existing uncommitted project changes and build artifacts must be preserved.

---

## File Structure

- Create `sonic/python/mm_sonic/torch_terrain_action_sweeps.py`: immutable canonical action-sweep metadata.
- Create `sonic/python/mm_sonic/torch_action_terrain_candidates.py`: bounded landing corrections and full-sole landing tests.
- Create `sonic/python/mm_sonic/torch_action_terrain_feasibility.py`: batched whole-action contact and swept-clearance filter.
- Create `sonic/python/mm_sonic/torch_action_terrain_graph.py`: node/edge contracts, deterministic search, failure cache, and result contract.
- Create `resources/run_g1_action_terrain_graph.py`: real GRAIL/MuJoCo adapter, exact edge realization, archive/report writer.
- Create `resources/run_g1_action_terrain_acceptance.py`: angle/object metamorphic acceptance matrix.
- Create one matching test module per new Python module or resource.
- Modify `docs/g1-horizontal-contact-traversal.md`: reproduction and interpretation of the new planner.

---

### Task 1: Canonical Terrain Action Sweeps

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_action_sweeps.py`
- Create: `tests/python/test_sonic_torch_terrain_action_sweeps.py`

**Interfaces:**
- Consumes: one `FootholdAction`, its source root/sole trajectories, source support mask, and source root yaw.
- Produces: `TerrainActionSweep` and `canonicalize_action_sweep(...)`.

```python
@dataclass(frozen=True)
class TerrainActionSweep:
    action_key: tuple[int, int]
    source_frames: torch.Tensor               # int64 [F]
    support_mask: torch.Tensor                # bool [F, 2]
    root_position_local_m: torch.Tensor       # float [F, 3]
    root_yaw_local_rad: torch.Tensor          # float [F]
    sole_center_local_m: torch.Tensor         # float [F, 2, 3]
    sole_points_local_m: torch.Tensor         # float [F, 2, S, 3]
    body_points_local_m: torch.Tensor         # float [F, B, 3]
    landing_frame_offsets: tuple[int, int]
    start_support: tuple[bool, bool]
```

- [ ] **Step 1: Write failing ownership and SE(2)-invariance tests**

```python
class TerrainActionSweepTests(unittest.TestCase):
    def test_global_rigid_transform_preserves_canonical_sweep(self):
        base = _synthetic_action_world(yaw=0.0, translation=(0.0, 0.0))
        moved = _synthetic_action_world(yaw=0.73, translation=(1.2, -0.4))

        first = canonicalize_action_sweep(**base)
        second = canonicalize_action_sweep(**moved)

        torch.testing.assert_close(
            first.root_position_local_m,
            second.root_position_local_m,
            atol=1e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            first.sole_points_local_m,
            second.sole_points_local_m,
            atol=1e-5,
            rtol=0.0,
        )

    def test_sweep_owns_input_tensors(self):
        values = _synthetic_action_world(yaw=0.0, translation=(0.0, 0.0))
        result = canonicalize_action_sweep(**values)
        values["root_position_world"].zero_()
        self.assertGreater(float(result.root_position_local_m[-1, 0]), 0.1)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_action_sweeps -v
```

Expected: import failure for `mm_sonic.torch_terrain_action_sweeps`.

- [ ] **Step 3: Implement strict contracts and canonicalization**

Use the first root XY and first root yaw as the planar origin. Rotate all XY
vectors by `-source_root_yaw_rad`; subtract the first root Z from root and sole
Z. Validate exact shapes, shared floating dtype/device, finite values,
monotonic source frames, at least one support per source frame, and landing
offsets inside the window.

```python
def _rotate_xy(points: torch.Tensor, yaw: float) -> torch.Tensor:
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = points.new_tensor(((c, -s), (s, c)))
    return points @ rotation.T

def canonicalize_action_sweep(
    *,
    action: FootholdAction,
    source_frames: torch.Tensor,
    root_position_world: torch.Tensor,
    root_yaw_world_rad: torch.Tensor,
    sole_center_world: torch.Tensor,
    sole_points_world: torch.Tensor,
    body_points_world: torch.Tensor,
    support_mask: torch.Tensor,
) -> TerrainActionSweep:
    origin_xy = root_position_world[0, :2]
    origin_z = root_position_world[0, 2]
    yaw = float(root_yaw_world_rad[0].item())
    root = root_position_world.clone()
    root[:, :2] = _rotate_xy(root[:, :2] - origin_xy, -yaw)
    root[:, 2] -= origin_z
    centers = sole_center_world.clone()
    centers[..., :2] = _rotate_xy(
        centers[..., :2] - origin_xy, -yaw
    )
    centers[..., 2] -= origin_z
    points = sole_points_world.clone()
    points[..., :2] = _rotate_xy(points[..., :2] - origin_xy, -yaw)
    points[..., 2] -= origin_z
    bodies = body_points_world.clone()
    bodies[..., :2] = _rotate_xy(bodies[..., :2] - origin_xy, -yaw)
    bodies[..., 2] -= origin_z
    return TerrainActionSweep(
        action_key=(action.clip_index, action.start_frame),
        source_frames=source_frames,
        support_mask=support_mask,
        root_position_local_m=root,
        root_yaw_local_rad=root_yaw_world_rad - yaw,
        sole_center_local_m=centers,
        sole_points_local_m=points,
        body_points_local_m=bodies,
        landing_frame_offsets=action.landing_frame_offsets,
        start_support=action.start_support,
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_action_sweeps \
  tests.python.test_sonic_torch_foothold_actions -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_action_sweeps.py \
  tests/python/test_sonic_torch_terrain_action_sweeps.py
git commit -m "feat: canonicalize terrain action sweeps"
```

---

### Task 2: Action-Conditioned Landing Candidates

**Files:**
- Create: `sonic/python/mm_sonic/torch_action_terrain_candidates.py`
- Create: `tests/python/test_sonic_torch_action_terrain_candidates.py`

**Interfaces:**
- Consumes: `TerrainActionSweep`, a node root transform, heading, correction
  samples, and `sample_surface(points_scene_xy)`.
- Produces: `ActionLandingCandidate` and
  `action_conditioned_landing_candidates(...)`.

```python
@dataclass(frozen=True)
class ActionLandingCandidate:
    action_key: tuple[int, int]
    correction_heading_xyz_m: torch.Tensor  # [2, 3]
    landing_center_scene_xyz_m: torch.Tensor # [2, 3]
    landing_sole_scene_xyz_m: torch.Tensor   # [2, S, 3]
    placement_cost: float
    minimum_edge_margin_m: float
```

- [ ] **Step 1: Write failing arbitrary-rotation and edge tests**

```python
class ActionTerrainCandidateTests(unittest.TestCase):
    def test_rotating_action_and_terrain_preserves_candidate_cost(self):
        first = _run_candidate_case(yaw=0.0)
        second = _run_candidate_case(yaw=math.pi / 3.0)
        self.assertEqual(len(first), len(second))
        self.assertAlmostEqual(
            first[0].placement_cost, second[0].placement_cost, places=6
        )

    def test_full_sole_rejects_center_valid_edge_straddle(self):
        candidates = action_conditioned_landing_candidates(
            **_edge_case_inputs(center_height=0.20, toe_height=0.0)
        )
        self.assertEqual(candidates, ())
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_action_terrain_candidates -v
```

Expected: missing-module import failure.

- [ ] **Step 3: Implement batched correction enumeration**

Generate deterministic Cartesian products of forward/lateral corrections at
2.5 cm resolution inside configured bounds. Apply corrections only to feet
that swing in the action. Transform source landing centers and sole points by
the node SE(2), query all sole points plus the uncertainty halo, and reject
surface variation above 2.5 cm.

Use normalized correction cost:

```python
cost = (
    torch.square(forward / maximum_forward_correction_m)
    + torch.square(lateral / maximum_lateral_correction_m)
    + torch.square(
        vertical / maximum_vertical_correction_m
    )
).sum(dim=1)
```

Sort candidates by `(placement_cost, correction L1 norm, original index)`.
Do not inspect terrain type or command angle.

- [ ] **Step 4: Run candidate and existing sole tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_action_terrain_candidates \
  tests.python.test_sonic_torch_terrain_footprint_candidates \
  tests.python.test_sonic_torch_g1_sole_kinematics -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_action_terrain_candidates.py \
  tests/python/test_sonic_torch_action_terrain_candidates.py
git commit -m "feat: generate action-conditioned terrain landings"
```

---

### Task 3: Swept Terrain Feasibility

**Files:**
- Create: `sonic/python/mm_sonic/torch_action_terrain_feasibility.py`
- Create: `tests/python/test_sonic_torch_action_terrain_feasibility.py`

**Interfaces:**
- Consumes: `TerrainActionSweep`, `ActionLandingCandidate`, node transform,
  terrain sampler, contact, swing, and conservative body tolerances.
- Produces: `ActionTerrainFeasibility` and
  `evaluate_action_terrain_feasibility(...)`.

```python
@dataclass(frozen=True)
class ActionTerrainFeasibility:
    feasible: bool
    code: str
    minimum_swing_clearance_m: float
    maximum_contact_error_m: float
    minimum_body_clearance_m: float
    worst_frame: int
    worst_foot: int
```

- [ ] **Step 1: Write the regression that matches the diagonal defect**

```python
class ActionTerrainFeasibilityTests(unittest.TestCase):
    def test_endpoint_valid_action_with_colliding_swing_is_rejected(self):
        result = evaluate_action_terrain_feasibility(
            **_diagonal_riser_case(
                source_peak_clearance_m=0.07,
                required_clearance_m=0.23,
            )
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.code, "swing_collision")
        self.assertEqual(result.worst_foot, 0)

    def test_joint_rotation_preserves_feasibility_and_metrics(self):
        first = evaluate_action_terrain_feasibility(
            **_valid_case(yaw=0.0)
        )
        second = evaluate_action_terrain_feasibility(
            **_valid_case(yaw=1.17)
        )
        self.assertEqual(first.feasible, second.feasible)
        self.assertAlmostEqual(
            first.minimum_swing_clearance_m,
            second.minimum_swing_clearance_m,
            places=5,
        )
```

- [ ] **Step 2: Run and verify missing implementation**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_action_terrain_feasibility -v
```

Expected: missing-module import failure.

- [ ] **Step 3: Implement the batched sweep**

For each swing interval, interpolate the landing correction from zero at
liftoff to its full value at touchdown using smoothstep
`p*p*(3 - 2*p)`. Keep supported sole points locked to their node/landing
anchors. Transform every sole point into scene space and query terrain in one
batch.

Measured sole clearance is:

```python
clearance = sole_points_scene[..., 2] - sample_surface(
    sole_points_scene[..., :2]
)
```

Rules:

- declared contact: `abs(clearance) <= contact_tolerance_m` for at least the
  configured minimum number of sole samples;
- swing: `clearance >= minimum_swing_clearance_m`;
- conservative knees, pelvis, wrists, and torso samples:
  `body_z - terrain_z >= minimum_body_clearance_m`;
- any violation returns the first deterministic worst frame/foot;
- no support labels are edited.

- [ ] **Step 4: Run focused feasibility tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_action_terrain_feasibility \
  tests.python.test_sonic_torch_heading_footprint_realizer -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_action_terrain_feasibility.py \
  tests/python/test_sonic_torch_action_terrain_feasibility.py
git commit -m "feat: reject terrain-colliding action sweeps"
```

---

### Task 4: Deterministic Action Contact Graph Search

**Files:**
- Create: `sonic/python/mm_sonic/torch_action_terrain_graph.py`
- Create: `tests/python/test_sonic_torch_action_terrain_graph.py`

**Interfaces:**
- Consumes: an initial `ActionTerrainNode`, goal distance, bounded search
  values, and an `expand(node)` callback.
- Produces: `ActionTerrainEdge`, `ActionTerrainRoute`,
  `ActionTerrainPlanningFailure`, and
  `search_action_terrain_route(...)`.

```python
@dataclass(frozen=True)
class ActionTerrainNode:
    node_id: int
    root_position_world: torch.Tensor       # [3]
    root_orientation_world_wxyz: torch.Tensor # [4]
    joint_position: torch.Tensor            # [29]
    joint_velocity: torch.Tensor            # [29]
    root_linear_velocity_world: torch.Tensor # [3]
    root_angular_velocity_world: torch.Tensor # [3]
    sole_center_world: torch.Tensor         # [2, 3]
    support_mask: torch.Tensor              # bool [2]
    progress_m: float
    lateral_error_m: float

@dataclass(frozen=True)
class ActionTerrainEdge:
    source_node_id: int
    target_node: ActionTerrainNode
    action_key: tuple[int, int]
    candidate_index: int
    frame_count: int
    cost: float
    minimum_clearance_m: float
    maximum_contact_error_m: float

@dataclass(frozen=True)
class ActionTerrainRoute:
    nodes: tuple[ActionTerrainNode, ...]
    edges: tuple[ActionTerrainEdge, ...]
    total_cost: float
    rejection_counts: dict[str, int]
```

- [ ] **Step 1: Write cycle, failure-cache, and preparatory-step tests**

```python
class ActionTerrainGraphTests(unittest.TestCase):
    def test_search_chooses_short_preparatory_action_before_step_up(self):
        route = search_action_terrain_route(
            initial_node=_node(progress=0.0),
            goal_distance_m=0.8,
            expand=_synthetic_expander_with_lunge_and_microstep(),
            maximum_expansions=32,
            beam_width=8,
        )
        self.assertEqual(
            tuple(edge.action_key for edge in route.edges),
            ((0, 10), (0, 30), (0, 50)),
        )

    def test_failed_exact_edge_is_evaluated_only_once(self):
        expander = _counting_expander_with_repeated_failure()
        with self.assertRaises(ActionTerrainPlanningFailure):
            search_action_terrain_route(
                initial_node=_node(progress=0.0),
                goal_distance_m=1.0,
                expand=expander,
                maximum_expansions=16,
                beam_width=4,
            )
        self.assertEqual(expander.calls_for((0, 90), 3), 1)
```

- [ ] **Step 2: Run and verify missing implementation**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_action_terrain_graph -v
```

Expected: missing-module import failure.

- [ ] **Step 3: Implement bounded best-first search**

Use a heap ordered by:

```python
priority = (
    accumulated_cost
    + heuristic_weight
    * max(0.0, goal_distance_m - node.progress_m)
    / maximum_progress_per_action_m,
    -node.progress_m,
    action_key_path,
    candidate_index_path,
)
```

Reject edges with progress below `-maximum_backtrack_m`. Quantize state keys
from both sole centers, support mask, progress, joint pose, and root velocity;
do not merge states using root XY alone. Cache expansion failures by
`(state_key, action_key, candidate_index)`. Return only when goal progress is
met and terminal support is complete.

- [ ] **Step 4: Run graph and legacy search tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_action_terrain_graph \
  tests.python.test_sonic_torch_heading_footprint_search -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_action_terrain_graph.py \
  tests/python/test_sonic_torch_action_terrain_graph.py
git commit -m "feat: search realizable terrain action graph"
```

---

### Task 5: Real GRAIL Edge Expansion and Route Runner

**Files:**
- Create: `resources/run_g1_action_terrain_graph.py`
- Create: `tests/python/test_run_g1_action_terrain_graph.py`
- Modify: `sonic/python/mm_sonic/torch_heading_footprint_realizer.py`
- Modify: `tests/python/test_sonic_torch_heading_footprint_realizer.py`

**Interfaces:**
- Consumes: dataset/config/G1 XML/start state/heading/distance/speed and
  search thresholds.
- Produces: `route.npz`, `graph.json`, `metrics.json`, `validation.json`, or
  `failure.json`.
- Extracts:

```python
def realize_single_action_edge(
    *,
    action: FootholdAction,
    source: object,
    start_joint_position: np.ndarray,
    start_root_position_world: np.ndarray,
    start_root_orientation_world_wxyz: np.ndarray,
    landing_centers_world: np.ndarray,
    terrain: object,
    kinematics: object,
    retargeter: object,
    action_time_scale: float,
) -> RealizedHeadingTraversal:
    ...
```

- [ ] **Step 1: Write failing single-edge start-state and CLI tests**

```python
class SingleActionEdgeTests(unittest.TestCase):
    def test_successor_edge_begins_from_predecessor_terminal_state(self):
        first = realize_single_action_edge(**_edge_case(start_x=0.0))
        second = realize_single_action_edge(
            **_edge_case(
                start_x=float(first.root_position_world[-1, 0]),
                start_joint=first.joint_position[-1],
            )
        )
        self.assertLess(
            float(np.max(np.abs(
                second.joint_position[0] - first.joint_position[-1]
            ))),
            0.30,
        )

class ActionTerrainRunnerTests(unittest.TestCase):
    def test_parser_has_no_angle_or_object_specific_switches(self):
        help_text = _parser().format_help()
        self.assertIn("--heading-degrees", help_text)
        self.assertNotIn("--stair", help_text)
        self.assertNotIn("--curb", help_text)
        self.assertNotIn("--lane", help_text)
```

- [ ] **Step 2: Run and verify failures**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_heading_footprint_realizer.SingleActionEdgeTests \
  tests.python.test_run_g1_action_terrain_graph -v
```

Expected: missing function and resource import failures.

- [ ] **Step 3: Extract exact one-edge realization**

Move the existing per-action solve loop body into
`realize_single_action_edge`. Preserve all current stance, swing-height,
orientation, COM, transition, and continuity checks. Make the existing
`realize_heading_footprint_plan` call this function so the horizontal
regression remains behaviorally identical.

- [ ] **Step 4: Implement the real expander and runner**

The runner must:

1. load the GRAIL dataset, contact index, and exact-contact actions;
2. build canonical sweeps once;
3. construct the actual starting graph node;
4. query actions compatible with support and transition bounds;
5. generate terrain landing candidates;
6. run the swept feasibility filter;
7. run exact single-edge realization;
8. create successor nodes from exact terminal kinematics;
9. search to complete support beyond requested progress;
10. concatenate edge archives without dropping boundary frames;
11. run `resources/run_g1_validate_traversal.py`;
12. write deterministic rejection counts and provenance.

Use one generic command:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_action_terrain_graph.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --start-state START.npz \
  --heading-degrees HEADING \
  --distance-m DISTANCE \
  --speed-mps 0.4 \
  --output OUTPUT
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_action_sweeps \
  tests.python.test_sonic_torch_action_terrain_candidates \
  tests.python.test_sonic_torch_action_terrain_feasibility \
  tests.python.test_sonic_torch_action_terrain_graph \
  tests.python.test_sonic_torch_heading_footprint_realizer \
  tests.python.test_run_g1_action_terrain_graph \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: all tests pass.

- [ ] **Step 6: Run horizontal and diagonal real cases**

Run the generic command first with the validated horizontal start/heading and
then with the `-45` degree start. Expected for both:

- `validation.json` has `"validated": true`;
- no `--forbid-action` option is used;
- every retained edge passed the sweep filter;
- terminal support is `[true, true]`;
- provenance coverage is `1.0`.

- [ ] **Step 7: Commit**

```bash
git add resources/run_g1_action_terrain_graph.py \
  tests/python/test_run_g1_action_terrain_graph.py \
  sonic/python/mm_sonic/torch_heading_footprint_realizer.py \
  tests/python/test_sonic_torch_heading_footprint_realizer.py
git commit -m "feat: run action-conditioned terrain planning"
```

---

### Task 6: Rotation and Object Metamorphic Acceptance

**Files:**
- Create: `resources/run_g1_action_terrain_acceptance.py`
- Create: `tests/python/test_run_g1_action_terrain_acceptance.py`
- Modify: `docs/g1-horizontal-contact-traversal.md`

**Interfaces:**
- Consumes: the generic runner inputs plus repeatable headings and terrain
  transform/perturbation seeds.
- Produces: per-case artifacts and `summary.json`.

- [ ] **Step 1: Write failing matrix and forbidden-field tests**

```python
class ActionTerrainAcceptanceTests(unittest.TestCase):
    def test_matrix_sweeps_angles_without_heading_specific_overrides(self):
        matrix = build_acceptance_matrix(
            headings_degrees=tuple(range(-180, 180, 15)),
            translations=((0.0, 0.0), (0.10, -0.05)),
            height_scales=(0.9, 1.0, 1.1),
        )
        self.assertEqual(len(matrix), 24 * 2 * 3)
        self.assertTrue(all(
            not set(case) & {
                "lane", "stair", "curb", "forbid_action", "source_frames"
            }
            for case in matrix
        ))
```

- [ ] **Step 2: Run and verify missing resource**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_action_terrain_acceptance -v
```

Expected: resource import failure.

- [ ] **Step 3: Implement deterministic acceptance reporting**

For each case, invoke the same generic runner arguments. Record:

- success or classified failure;
- action keys and costs;
- rejection counts;
- contact, clearance, continuity, progress, and naturalness metrics;
- rotation/translation-equivalence deltas;
- forbidden runtime fields, which must be empty.

The summary distinguishes `validated`, `unreachable`, `no_motion_coverage`,
and planner defects. It must not count classified unreachable terrain as a
planner success.

- [ ] **Step 4: Document reproduction and interpretation**

Replace the old heading-grid acceptance section with the new generic runner,
the matrix runner, and the distinction between geometric generality and
physical/library reachability. Include the optional later DSMS refinement as
a post-processing stage, not a prerequisite.

- [ ] **Step 5: Run the complete focused suite**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_action_sweeps \
  tests.python.test_sonic_torch_action_terrain_candidates \
  tests.python.test_sonic_torch_action_terrain_feasibility \
  tests.python.test_sonic_torch_action_terrain_graph \
  tests.python.test_sonic_torch_heading_footprint_realizer \
  tests.python.test_run_g1_action_terrain_graph \
  tests.python.test_run_g1_action_terrain_acceptance \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: all tests pass.

- [ ] **Step 6: Run the real acceptance matrix**

Use all headings at 15-degree increments and at least two translated starts.
Acceptance requires:

- horizontal baseline still validated;
- at least three non-collinear feasible headings validated;
- no retained swept collision;
- no runtime blacklist or object/angle field;
- every failure classified;
- all successful routes end in complete support with full provenance.

- [ ] **Step 7: Commit**

```bash
git add resources/run_g1_action_terrain_acceptance.py \
  tests/python/test_run_g1_action_terrain_acceptance.py \
  docs/g1-horizontal-contact-traversal.md
git commit -m "test: qualify general terrain action planning"
```

---

## Final Verification

- [ ] Run `git diff --check`.
- [ ] Run the complete focused suite from Task 6.
- [ ] Inspect contact sheets or videos for horizontal, diagonal, and one
  head-on route.
- [ ] Confirm no new runtime field or branch names an angle, lane, stair,
  curb, block, or object class.
- [ ] Confirm exact realization failures appear in rejection counts and are
  not retried under the same state/action/candidate signature.
- [ ] Confirm all successful routes have complete provenance and terminal
  double support.
