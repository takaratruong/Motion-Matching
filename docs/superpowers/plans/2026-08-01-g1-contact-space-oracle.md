# G1 Contact-Space Terrain Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, non-real-time contact-phase beam-search oracle that proves whether the existing GRAIL corpus can synthesize high-quality omnidirectional G1 stair kinematics.

**Architecture:** Extract one-landing contact-phase actions with authoritative MuJoCo FK, place them rigidly in matcher world, reject actions using exact terrain/contact constraints, and search four landings ahead. Execute only the first planned edge, replan at the next stable contact, save route diagnostics, and compare the same 21-route stress matrix with the retained layered graph hybrid.

**Tech Stack:** Python 3.11, PyTorch, NumPy, MuJoCo, unittest, existing `mm_sonic` terrain datasets and omnidirectional route contracts.

## Global Constraints

- Work only in `/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver` on `research/g1-torch-terrain-kinematics`.
- Preserve all pre-existing modified and untracked files; stage only files named by the active task.
- Keep Sonic, tracking, depth inference, physics simulation, continuous IK, and real-time latency out of scope.
- Do not change retained `TorchMotionMatcher` or `FootholdActionPolicy` behavior.
- Use authoritative MuJoCo FK for feet and native 50 Hz source motion.
- Use deterministic stable ordering by `(clip_index, start_frame, end_frame)`.
- Default search horizon is four landings and beam width is 256; there is no wall-clock cutoff.
- Never emit an unchecked fallback action.
- Commit after every task with that task's exact file set.

---

## File Structure

- `sonic/python/mm_sonic/torch_contact_oracle_actions.py`: immutable one-landing action extraction and source inventory diagnostics.
- `sonic/python/mm_sonic/torch_contact_oracle_search.py`: rigid placement, terrain feasibility, cost decomposition, reachability, and beam search.
- `sonic/python/mm_sonic/torch_contact_oracle_rollout.py`: receding-horizon execution over existing route schedules and deterministic artifact construction.
- `sonic/python/mm_sonic/torch_contact_oracle_viewer.py`: saved-rollout MuJoCo kinematic viewer.
- `sonic/configs/experiments/torch_grail_contact_oracle.json`: frozen oracle constraints and cost weights.
- `resources/run_g1_torch_contact_oracle.py`: command-line matrix runner.
- `tests/python/test_sonic_torch_contact_oracle_actions.py`: action extraction contracts.
- `tests/python/test_sonic_torch_contact_oracle_search.py`: placement, feasibility, costs, and greedy-dead-end proof.
- `tests/python/test_sonic_torch_contact_oracle_rollout.py`: route scheduling, horizon fallback, arrays, and hashing.
- `tests/python/test_sonic_torch_contact_oracle_viewer.py`: viewer loading and qpos application.
- `tests/python/test_run_g1_torch_contact_oracle.py`: CLI contract.
- `docs/superpowers/results/2026-08-01-g1-contact-space-oracle.md`: measured 21-route comparison and interpretation.

### Task 1: Extract immutable one-landing contact-phase actions

**Files:**
- Create: `sonic/python/mm_sonic/torch_contact_oracle_actions.py`
- Create: `tests/python/test_sonic_torch_contact_oracle_actions.py`

**Interfaces:**
- Consumes: `ContactSegmentIndex`, `TerrainDataset`, and `MujocoG1FootKinematics.foot_positions(...)`.
- Produces: `ContactPhaseAction`, `action_from_profiles(...)`, `ContactPhaseActionIndex.from_dataset(dataset, segment_index, foot_kinematics)`, `ContactPhaseInventory`, `ContactPhaseActionIndex.action(index)`, and `ContactPhaseActionIndex.exact_successor(index)`.

- [ ] **Step 1: Write failing extraction and rejection-accounting tests**

Add tests that construct two alternating contact onsets and assert an edge begins at the first onset and includes the next onset as its final frame:

```python
def test_extracts_one_landing_phase_with_inclusive_terminal_contact(self):
    support = torch.tensor([
        [True, False], [True, False], [True, True],
        [False, True], [False, True], [True, True],
    ])
    profiles = synthetic_profiles(frames=6)
    action = action_from_profiles(
        clip_index=3,
        start_frame=2,
        landing_frame=5,
        support_mask=support,
        joint_position=profiles.joint_position,
        joint_velocity=profiles.joint_velocity,
        root_position_world=profiles.root_position,
        root_orientation_world_wxyz=profiles.root_quaternion,
        foot_position_world=profiles.foot_position,
        foot_surface_height_m=profiles.foot_surface,
    )
    self.assertEqual((action.start_frame, action.end_frame), (2, 6))
    self.assertEqual(action.swing_foot, 0)
    self.assertEqual(tuple(action.exit_support.tolist()), (True, True))
    self.assertEqual(action.frame_count, 4)
```

Also assert malformed/no-next-landing candidates increment exact rejection keys: `no-next-opposite-landing`, `invalid-fk`, `unstable-terminal-support`, and `inconsistent-contact-order`.

- [ ] **Step 2: Run the focused test and verify the module is missing**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_sonic_torch_contact_oracle_actions
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mm_sonic.torch_contact_oracle_actions'`.

- [ ] **Step 3: Implement the immutable action and index**

Use these public records and half-open frame convention:

```python
@dataclass(frozen=True)
class ContactPhaseAction:
    clip_index: int
    start_frame: int
    end_frame: int
    swing_foot: int
    entry_support: torch.Tensor          # [2] bool
    exit_support: torch.Tensor           # [2] bool
    support_mask: torch.Tensor            # [T, 2] bool
    joint_position: torch.Tensor           # [T, 29]
    joint_velocity: torch.Tensor           # [T, 29]
    root_position_local: torch.Tensor      # [T, 3]
    root_yaw_local: torch.Tensor           # [T]
    foot_position_local: torch.Tensor      # [T, 2, 3]
    foot_surface_delta_m: torch.Tensor     # [T, 2]
    minimum_swing_clearance_m: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def source_key(self) -> tuple[int, int, int]:
        return self.clip_index, self.start_frame, self.end_frame


@dataclass(frozen=True)
class ContactPhaseInventory:
    retained_count: int
    rejected_by_reason: Mapping[str, int]


@dataclass(frozen=True)
class ContactPhaseActionIndex:
    actions: tuple[ContactPhaseAction, ...]
    inventory: ContactPhaseInventory
    exact_successor_indices: tuple[int | None, ...]

    @classmethod
    def from_dataset(cls, dataset, segment_index, foot_kinematics):
        actions, rejected = [], Counter()
        for segment in sorted(segment_index.segments):
            landing = segment.end_frame
            clip = dataset.folder.clips[segment.clip_index]
            if landing >= clip.joint_position.shape[0]:
                rejected["no-next-opposite-landing"] += 1
                continue
            action = _action_from_dataset_segment(
                dataset, segment_index, foot_kinematics, segment, landing
            )
            if isinstance(action, str):
                rejected[action] += 1
            else:
                actions.append(action)
        ordered = tuple(sorted(actions, key=lambda value: value.source_key))
        by_entry = {
            (action.clip_index, action.start_frame): index
            for index, action in enumerate(ordered)
        }
        successors = tuple(
            by_entry.get((action.clip_index, action.end_frame - 1))
            for action in ordered
        )
        return cls(
            ordered,
            ContactPhaseInventory(len(ordered), MappingProxyType(dict(rejected))),
            successors,
        )
```

Convert all source positions to the start-root SE(2) frame, subtract the start root Z from root/foot Z, unwrap yaw relative to the start yaw, clone tensors in `__post_init__`, and reject non-finite or incorrectly shaped values with `ContractError`.
Compute exact successors from the next source contact phase in the same clip;
`exact_successor(index)` returns the mapped action or `None` without guessing a
cross-clip continuation.

- [ ] **Step 4: Run action tests**

Run the Task 1 command. Expected: all tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_contact_oracle_actions.py tests/python/test_sonic_torch_contact_oracle_actions.py
git commit -m "feat: index one-landing terrain actions"
```

### Task 2: Place actions rigidly and enforce exact terrain feasibility

**Files:**
- Create: `sonic/python/mm_sonic/torch_contact_oracle_search.py`
- Create: `tests/python/test_sonic_torch_contact_oracle_search.py`

**Interfaces:**
- Consumes: `ContactPhaseAction` and a callable `sample_surface(points_xy: torch.Tensor) -> torch.Tensor`.
- Produces: `OracleState`, `PlacedContactPhase`, `OracleConstraints`, `FeasibilityResult`, `place_action(...)`, and `validate_placement(...)`.

- [ ] **Step 1: Write failing rigid-transform and terrain-constraint tests**

Cover a 90-degree placement, stable landing, edge rejection, stance mismatch, swing penetration, source/query height-change mismatch, support-order mismatch, and transition-bound rejection:

```python
def test_rigid_placement_rotates_source_trajectory_about_entry_root(self):
    action = one_meter_forward_action()
    state = oracle_state(root_xy=(2.0, 3.0), root_yaw=math.pi / 2.0)
    placed = place_action(action, state)
    np.testing.assert_allclose(
        placed.root_position_world[-1, :2].cpu(), [2.0, 4.0], atol=1e-6
    )
    self.assertAlmostEqual(float(placed.root_yaw_world[-1]), math.pi / 2.0)


def test_rejects_landing_on_height_discontinuity_inside_edge_margin(self):
    result = validate_placement(
        placed=edge_straddling_action(),
        state=oracle_state(),
        sample_surface=step_surface,
        constraints=OracleConstraints(),
    )
    self.assertFalse(result.accepted)
    self.assertEqual(result.reason, "landing-edge-margin")
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_sonic_torch_contact_oracle_search
```

Expected: FAIL because the search module does not exist.

- [ ] **Step 3: Implement state, placement, and ordered hard constraints**

Freeze the initial constraints in one dataclass:

```python
@dataclass(frozen=True)
class OracleConstraints:
    stance_height_tolerance_m: float = 0.05
    landing_height_tolerance_m: float = 0.03
    minimum_swing_clearance_m: float = -0.03
    maximum_height_deformation_m: float = 0.06
    edge_margin_m: float = 0.04
    maximum_edge_height_range_m: float = 0.025
    maximum_entry_foot_error_m: float = 0.08
    maximum_joint_position_error_rad: float = 1.50
    maximum_joint_velocity_error_rad_s: float = 8.0


@dataclass(frozen=True)
class OracleState:
    root_position_world: torch.Tensor      # [3]
    root_yaw_world: torch.Tensor            # scalar
    foot_position_world: torch.Tensor       # [2, 3]
    support_mask: torch.Tensor              # [2] bool
    joint_position: torch.Tensor            # [29]
    joint_velocity: torch.Tensor            # [29]
    route_frame: int
    source_history: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True)
class FeasibilityResult:
    accepted: bool
    reason: str | None
    stance_error_m: float
    landing_error_m: float
    minimum_swing_clearance_m: float
```

`validate_placement` must evaluate constraints in this fixed order so diagnostics are deterministic: `support-order`, `entry-foot-error`, `joint-position`, `joint-velocity`, `stance-height`, `landing-height`, `landing-edge-margin`, `swing-penetration`, `height-deformation`. Surface edge checks sample center and ±margin on X/Y. No exception may be converted into acceptance.
All stance, landing, and swing clearances are sole clearances computed as
`placed_ankle_z - sampled_surface_z - ANKLE_ORIGIN_SOLE_M`; do not compare the
ankle body origin directly with terrain height.

- [ ] **Step 4: Run Task 2 tests**

Run the Task 2 command. Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add sonic/python/mm_sonic/torch_contact_oracle_search.py tests/python/test_sonic_torch_contact_oracle_search.py
git commit -m "feat: validate rigid terrain action placement"
```

### Task 3: Add deterministic four-contact beam search and the greedy-dead-end proof

**Files:**
- Modify: `sonic/python/mm_sonic/torch_contact_oracle_search.py`
- Modify: `tests/python/test_sonic_torch_contact_oracle_search.py`
- Create: `sonic/configs/experiments/torch_grail_contact_oracle.json`

**Interfaces:**
- Consumes: Task 2 state/placement/feasibility and `CommandSchedule.sample(frame)`.
- Produces: `OracleCost`, `OracleSearchConfig`, `OraclePlan`, `OracleSearchFailure`, `search_contact_plan(...)`, and `advance_state(...)`.

- [ ] **Step 1: Write a failing graph where greedy search enters a dead end**

Construct actions `cheap_dead_end`, `costlier_bridge`, and `goal`. The first has lower immediate path cost but no feasible successor; the bridge reaches the goal:

```python
def test_four_contact_search_rejects_cheapest_greedy_dead_end(self):
    index = synthetic_index(
        actions=(cheap_dead_end(), costlier_bridge(), bridge_successor(), goal())
    )
    plan = search_contact_plan(
        initial_state=oracle_state(),
        actions=index.actions,
        command_schedule=constant_schedule(velocity=(0.3, 0.0), heading=0.0),
        sample_surface=flat_surface,
        config=OracleSearchConfig(horizon_landings=3, beam_width=8),
    )
    self.assertEqual(plan.action_indices, (1, 2, 3))
    self.assertGreater(plan.expansion.rejected_by_reason["no-successor"], 0)
```

Also test stable tie-breaking, repeated-action cost, facing independent of velocity, shortening `4 -> 3 -> 2 -> 1`, and structured failure when no one-step action exists.

- [ ] **Step 2: Run the single greedy-dead-end test and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_sonic_torch_contact_oracle_search.ContactOracleSearchTests.test_four_contact_search_rejects_cheapest_greedy_dead_end
```

Expected: FAIL because `search_contact_plan` is missing.

- [ ] **Step 3: Implement cost decomposition and stable beam search**

Use this exact cost schema and defaults:

```python
@dataclass(frozen=True)
class OracleCost:
    path: float = 0.0
    facing: float = 0.0
    foothold: float = 0.0
    timing: float = 0.0
    joint_position: float = 0.0
    joint_velocity: float = 0.0
    stance_motion: float = 0.0
    clearance_margin: float = 0.0
    reachability: float = 0.0
    repetition: float = 0.0

    @property
    def total(self) -> float:
        return sum(astuple(self))


@dataclass(frozen=True)
class OracleSearchConfig:
    horizon_landings: int = 4
    beam_width: int = 256
    foothold_beam_width: int = 50
    constraints: OracleConstraints = OracleConstraints()
    path_weight: float = 25.0
    facing_weight: float = 10.0
    foothold_weight: float = 50.0
    timing_weight: float = 0.10
    joint_position_weight: float = 0.50
    joint_velocity_weight: float = 0.05
    stance_motion_weight: float = 100.0
    clearance_margin_weight: float = 1.0
    reachability_weight: float = 10.0
    repetition_weight: float = 2.0
```

For each node, call existing `plan_footholds(...)` and compare the candidate's first landing to same-foot first targets. Integrate the route command schedule over the candidate frame count for desired terminal root and facing. Compute reachability as `weight / (1 + feasible_successor_count)` and reject nonterminal nodes with zero feasible successors. Sort beam entries by `(cumulative_cost.total, source_key_path)` and retain exactly the first `beam_width`.

Implement `search_contact_plan` by attempting the configured horizon down to one. Return the first complete horizon. Raise `OracleSearchFailure` with rejection counts only when the one-contact search is empty.

- [ ] **Step 4: Freeze the experiment configuration**

Create `torch_grail_contact_oracle.json` with schema `g1-contact-space-oracle/v1`, all Task 2 constraints, all Task 3 weights, `horizon_landings: 4`, `beam_width: 256`, and `foothold_beam_width: 50`. The loader must reject missing/extra keys and non-finite values.

- [ ] **Step 5: Run all search tests**

Run the Task 2 command. Expected: all tests PASS, including the greedy-dead-end fixture.

- [ ] **Step 6: Commit Task 3**

```bash
git add sonic/python/mm_sonic/torch_contact_oracle_search.py tests/python/test_sonic_torch_contact_oracle_search.py sonic/configs/experiments/torch_grail_contact_oracle.json
git commit -m "feat: plan terrain contacts with deterministic lookahead"
```

### Task 4: Execute receding-horizon plans on existing stair routes

**Files:**
- Create: `sonic/python/mm_sonic/torch_contact_oracle_rollout.py`
- Create: `tests/python/test_sonic_torch_contact_oracle_rollout.py`

**Interfaces:**
- Consumes: Tasks 1–3, `OmniRoute`, `world_commands`, `evaluate_route_outcome`, and `evaluate_omni_route`.
- Produces: `OracleRouteRun`, `OracleMatrix`, `run_oracle_route(...)`, `run_resolved_oracle_matrix(...)`, and `save_oracle_matrix(...)`.

- [ ] **Step 1: Write failing route execution tests**

Test that the first edge only is executed, the planner is called again at its terminal contact, commands remain on the original per-frame route schedule, a final edge is trimmed to exact route length, zero-speed frames hold the last pose, and no-plan errors are recorded rather than replaced:

```python
def test_executes_one_edge_then_replans_at_terminal_contact(self):
    planner = RecordingPlanner(plans=(plan(0, 1), plan(2, 3)))
    run = run_oracle_route(
        route=fourteen_frame_route(),
        stair_frame=self.stair,
        initial_state=oracle_state(),
        planner=planner,
        action_index=synthetic_index(),
        terrain_sampler=flat_surface,
    )
    self.assertEqual(planner.requested_route_frames, [0, 6, 12])
    self.assertEqual(run.completed_frames, 14)
    self.assertEqual(run.arrays["planned_horizon"].tolist(), [2] * 6 + [2] * 6 + [1] * 2)
```

- [ ] **Step 2: Run rollout tests and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_sonic_torch_contact_oracle_rollout
```

Expected: FAIL because the rollout module does not exist.

- [ ] **Step 3: Implement route arrays and transactional execution**

Emit the existing outcome/metric arrays plus oracle diagnostics:

```python
ORACLE_ARRAY_SHAPES = {
    "command_velocity_world_xy": (2,),
    "command_heading_world_yaw": (),
    "command_segment_index": (),
    "qpos": (36,),
    "joint_position": (29,),
    "joint_velocity": (29,),
    "root_position_world": (3,),
    "root_yaw_world": (),
    "foot_position_world": (2, 3),
    "foot_surface_height_m": (2,),
    "selected_clip_path": (),
    "selected_source_frame": (),
    "selected_action_index": (),
    "planned_horizon": (),
    "plan_total_cost": (),
    "plan_time_ns": (),
    "stance_error_m": (),
    "landing_error_m": (),
    "minimum_swing_clearance_m": (),
}
```

Build all per-edge rows in temporary lists, validate every shape/value, then append atomically. A planning exception produces `RouteFailure(stage="oracle-search", ...)` and ends the route. Saved hashes exclude `plan_time_ns`, matching the existing timing-array policy.

- [ ] **Step 4: Build the resolved GRAIL adapter**

`run_resolved_oracle_matrix` must create `ContactSegmentIndex.from_dataset`, `ContactPhaseActionIndex.from_dataset`, the matcher-world surface closure from `measurement_extension`, the same `StairFrame` and reset position as `run_resolved_omni_matrix`, and one initial reset state from the configured reset clip. It must not instantiate or step `TorchMotionMatcher` after reset-state construction.

- [ ] **Step 5: Run rollout and neighboring route/metric tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_sonic_torch_contact_oracle_rollout \
  tests.python.test_sonic_torch_terrain_omni_routes \
  tests.python.test_sonic_torch_terrain_omni_metrics \
  tests.python.test_sonic_torch_terrain_omni_rollout
```

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add sonic/python/mm_sonic/torch_contact_oracle_rollout.py tests/python/test_sonic_torch_contact_oracle_rollout.py
git commit -m "feat: execute contact oracle stair routes"
```

### Task 5: Add the matrix CLI and deterministic saved artifacts

**Files:**
- Create: `resources/run_g1_torch_contact_oracle.py`
- Create: `tests/python/test_run_g1_torch_contact_oracle.py`
- Modify: `sonic/python/mm_sonic/torch_contact_oracle_rollout.py`
- Modify: `tests/python/test_sonic_torch_contact_oracle_rollout.py`

**Interfaces:**
- Consumes: `load_contact_oracle_config`, `same_stair_routes`, `run_resolved_oracle_matrix`, and `save_oracle_matrix`.
- Produces: a CLI accepting `--dataset`, `--config`, `--g1-xml`, `--output`, `--device`, and repeatable `--route`.

- [ ] **Step 1: Write failing CLI and round-trip tests**

Assert exact options, unknown-route rejection, config-schema rejection, selected route ordering, atomic output replacement, per-route `rollout.npz` plus `diagnostics.json`, matrix `summary.json`, and deterministic SHA round-trip.

```python
def test_parser_exposes_only_oracle_inputs(self):
    parser = build_parser()
    args = parser.parse_args([
        "--dataset", "data", "--config", "oracle.json",
        "--g1-xml", "g1.xml", "--output", "out",
        "--device", "cuda:7", "--route", "diagonal-down-left",
    ])
    self.assertEqual(args.route, ["diagonal-down-left"])
    self.assertFalse(hasattr(args, "contact_segments"))
    self.assertFalse(hasattr(args, "foothold_arm"))
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_run_g1_torch_contact_oracle
```

Expected: FAIL because the resource module is missing.

- [ ] **Step 3: Implement CLI and atomic save**

`main()` must load exactly schema `g1-contact-space-oracle/v1`, resolve requested routes in inventory order, run the matrix, atomically save through a sibling temporary directory plus `os.replace`, print one JSON summary line, and return `0` only when execution completed. Behavioral acceptance remains in the saved summary and must not prevent saving failed research evidence.

- [ ] **Step 4: Run CLI and rollout tests**

Run both Task 4 and Task 5 test commands. Expected: all tests PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add resources/run_g1_torch_contact_oracle.py tests/python/test_run_g1_torch_contact_oracle.py sonic/python/mm_sonic/torch_contact_oracle_rollout.py tests/python/test_sonic_torch_contact_oracle_rollout.py
git commit -m "feat: run and save contact oracle matrix"
```

### Task 6: Add an inspectable MuJoCo kinematic viewer

**Files:**
- Create: `sonic/python/mm_sonic/torch_contact_oracle_viewer.py`
- Create: `tests/python/test_sonic_torch_contact_oracle_viewer.py`

**Interfaces:**
- Consumes: one saved route directory, `target_state_qpos`, and the terrain scene construction helpers from `torch_terrain_live_viewer`.
- Produces: `load_oracle_route(path)`, `apply_oracle_frame(...)`, and module CLI `python -m mm_sonic.torch_contact_oracle_viewer`.

- [ ] **Step 1: Write failing strict-load and frame-application tests**

```python
def test_applies_saved_qpos_and_exposes_oracle_overlay(self):
    saved = load_oracle_route(self.fixture)
    overlay = apply_oracle_frame(self.mujoco, self.model, self.data, saved, 2)
    np.testing.assert_array_equal(self.data.qpos, saved.arrays["qpos"][2])
    self.assertIn("action=", overlay)
    self.assertIn("horizon=", overlay)
    self.assertIn("landing_error=", overlay)
```

Reject missing arrays, hash mismatch, non-finite qpos, mismatched route identity, and out-of-range frames.

- [ ] **Step 2: Run viewer tests and verify failure**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v tests.python.test_sonic_torch_contact_oracle_viewer
```

Expected: FAIL because the viewer module does not exist.

- [ ] **Step 3: Implement passive playback**

Use the exact saved qpos without physics stepping. Support `Space` pause/resume, left/right frame stepping while paused, `Backspace` rewind, and `X` exit. Overlay route/frame, source clip/frame, action, horizon, total plan cost, stance/landing error, and swing clearance. Do not add live WASD in this oracle viewer.

- [ ] **Step 4: Run viewer tests**

Run the Task 6 command. Expected: all tests PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add sonic/python/mm_sonic/torch_contact_oracle_viewer.py tests/python/test_sonic_torch_contact_oracle_viewer.py
git commit -m "feat: visualize saved contact oracle routes"
```

### Task 7: Run the full matrix, compare the retained baseline, and record the decision

**Files:**
- Create: `docs/superpowers/results/2026-08-01-g1-contact-space-oracle.md`
- Modify implementation/test files only if a measured defect is first reproduced by a focused failing test.

**Interfaces:**
- Consumes: the completed CLI and the retained layered graph hybrid matrix runner.
- Produces: saved oracle matrix, retained baseline matrix, visual review route set, and a committed evidence report.

- [ ] **Step 1: Run all focused oracle tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_sonic_torch_contact_oracle_actions \
  tests.python.test_sonic_torch_contact_oracle_search \
  tests.python.test_sonic_torch_contact_oracle_rollout \
  tests.python.test_sonic_torch_contact_oracle_viewer \
  tests.python.test_run_g1_torch_contact_oracle
```

Expected: all tests PASS.

- [ ] **Step 2: Run the neighboring terrain regression suite**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_sonic_torch_contact_segments \
  tests.python.test_sonic_torch_foothold_actions \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_terrain_omni_routes \
  tests.python.test_sonic_torch_terrain_omni_metrics \
  tests.python.test_sonic_torch_terrain_omni_rollout
```

Expected: all tests PASS.

- [ ] **Step 3: Run the 21-route oracle matrix on an available GPU**

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python \
  resources/run_g1_torch_contact_oracle.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_contact_oracle.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:7 \
  --output build/contact-oracle/full-matrix
```

Expected: every route produces a saved artifact or a structured `oracle-search` failure; the process does not crash and does not emit unchecked fallbacks.

- [ ] **Step 4: Run the retained comparison with identical routes**

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python \
  resources/run_g1_torch_terrain_omni.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_layered_graph_hybrid.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:7 \
  --contact-segments \
  --foothold-arm layered-graph-hybrid \
  --output build/contact-oracle/retained-baseline
```

Expected: a complete retained comparison artifact on the same route inventory.

- [ ] **Step 5: Visually inspect the decisive failures and successes**

Launch the saved oracle viewer for `cross-tread-left-to-right`, `diagonal-down-left`, `side-exit-upper-left`, `turn-90-middle-left`, `riser-reversal`, and `mixed-adversarial`. Record foot placement, sliding, premature stepping, stalls, and source discontinuities in the result document.

- [ ] **Step 6: Write the evidence report and classify the result**

The report must include inventory counts/rejections, route table, contact violations, aggregate slide, maximum stall, cost/rejection diagnostics, deterministic hashes, and one of these exact conclusions:

- `search-formulation-qualified` when all design acceptance gates pass;
- `continuous-contact-warp-required` when sequences complete but placement/slide fails;
- `action-representation-or-corpus-insufficient` when broad-bound oracle search has no feasible sequence; or
- `oracle-implementation-unqualified` when tests, determinism, or safety contracts fail.

- [ ] **Step 7: Commit measured evidence**

```bash
git add docs/superpowers/results/2026-08-01-g1-contact-space-oracle.md
git commit -m "docs: evaluate contact-space terrain oracle"
```

Do not commit `build/contact-oracle`; record its deterministic hashes and paths in the report.
