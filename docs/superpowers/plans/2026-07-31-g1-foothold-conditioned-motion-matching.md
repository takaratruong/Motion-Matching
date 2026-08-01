# G1 Foothold-Conditioned Motion Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare first-contact filtering, two-contact action search, and their hybrid on the frozen omnidirectional stair benchmark, retaining only a feedback-ready kinematic winner.

**Architecture:** Extract immutable two-contact descriptors from authenticated source FK and terrain, build short query foothold plans from emitted contacts and the heightmap, and expose deterministic eligibility/cost tensors to the existing Torch matcher. Exact FK/contact validation remains the final authority and accepted motion plays only to the next contact boundary.

**Tech Stack:** Python 3, PyTorch, NumPy, MuJoCo FK, `unittest`, existing `mm_sonic` Torch matcher and terrain route harness.

## Global Constraints

- Privileged-heightmap, 50 Hz, kinematic G1 locomotion only.
- Sonic, tracking, physics, depth inference, and real-time latency are excluded.
- WASD specifies world travel direction and body heading follows that direction.
- Existing `-0.03 m` penetration and source support-profile gates may not be weakened.
- All alternatives use the same corpus, normalization, staircase, route commands, and evaluator.
- A viewer launches only after automated qualification.

---

### Task 1: Authenticated two-contact action inventory

**Files:**
- Create: `sonic/python/mm_sonic/torch_foothold_actions.py`
- Create: `tests/python/test_sonic_torch_foothold_actions.py`

**Interfaces:**
- Consumes: `TerrainDataset`, `ContactSegmentIndex`, and source FK body positions.
- Produces: `FootholdAction`, `FootholdActionIndex.from_dataset(dataset, segments)`, and `rows_for_database(database) -> Sequence[FootholdAction | None]`.

- [ ] **Step 1: Write failing descriptor tests**

```python
def test_two_contact_action_records_alternating_landings():
    action = action_from_profiles(
        clip_index=2,
        start_frame=1,
        support_mask=torch.tensor(
            [[1, 1], [1, 0], [1, 0], [1, 1], [0, 1], [1, 1]],
            dtype=torch.bool,
        ),
        foot_xy_m=torch.tensor(
            [[[0., 0.], [0., .2]], [[0., 0.], [0., .2]],
             [[0., 0.], [.2, .2]], [[0., 0.], [.3, .2]],
             [[0., 0.], [.3, .2]], [[.3, 0.], [.3, .2]]],
            dtype=torch.float32,
        ),
        foot_surface_height_m=torch.tensor(
            [[0., 0.], [0., 0.], [0., 0.], [0., .18],
             [0., .18], [.18, .18]], dtype=torch.float32,
        ),
        root_xy_m=torch.zeros((6, 2), dtype=torch.float32),
        root_yaw_rad=torch.zeros(6, dtype=torch.float32),
    )
    assert action.landing_feet == (1, 0)
    assert action.landing_frame_offsets == (2, 4)
    torch.testing.assert_close(
        action.landing_height_delta_m, torch.tensor((.18, .18))
    )

def test_action_rejects_repeated_same_foot_landing():
    with self.assertRaisesRegex(ContractError, "alternating"):
        FootholdAction(
            clip_index=0, start_frame=0, end_frame=5,
            start_support=(True, False), landing_feet=(0, 0),
            landing_frame_offsets=(2, 4),
            landing_xy_start_frame_m=torch.zeros((2, 2)),
            landing_height_delta_m=torch.zeros(2),
            root_displacement_m=torch.zeros((2, 2)),
            root_yaw_delta_rad=torch.zeros(2),
            minimum_swing_clearance_m=0.04,
            maximum_unsupported_frames=0,
        )
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_foothold_actions`

Expected: import failure for `mm_sonic.torch_foothold_actions`.

- [ ] **Step 3: Implement immutable descriptors and extraction**

```python
@dataclass(frozen=True)
class FootholdAction:
    clip_index: int
    start_frame: int
    end_frame: int
    start_support: tuple[bool, bool]
    landing_feet: tuple[int, int]
    landing_frame_offsets: tuple[int, int]
    landing_xy_start_frame_m: torch.Tensor  # (2, 2)
    landing_height_delta_m: torch.Tensor  # (2,)
    root_displacement_m: torch.Tensor  # (2, 2)
    root_yaw_delta_rad: torch.Tensor  # (2,)
    minimum_swing_clearance_m: float
    maximum_unsupported_frames: int

def action_from_profiles(
    clip_index: int,
    start_frame: int,
    support_mask: torch.Tensor,
    foot_xy_m: torch.Tensor,
    foot_surface_height_m: torch.Tensor,
    root_xy_m: torch.Tensor,
    root_yaw_rad: torch.Tensor,
) -> FootholdAction | None:
    onsets = [
        (frame, foot)
        for frame in range(start_frame + 1, len(support_mask))
        for foot in (0, 1)
        if support_mask[frame, foot] and not support_mask[frame - 1, foot]
    ]
    pair = next(
        ((a, b) for i, a in enumerate(onsets) for b in onsets[i + 1:]
         if a[1] != b[1]),
        None,
    )
    if pair is None:
        return None
    first, second = pair
    if first[1] == second[1]:
        raise ContractError("foothold action landings must alternate")
    # The implementation rotates the two landing/root displacements into the
    # start-root frame, measures swing clearance against each source surface,
    # counts the longest no-support run, and clones tensors before construction.
```

- [ ] **Step 4: Run descriptor tests and existing contact tests**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_foothold_actions tests.python.test_sonic_torch_contact_segments`

Expected: all tests pass.

- [ ] **Step 5: Commit the inventory**

```bash
git add sonic/python/mm_sonic/torch_foothold_actions.py tests/python/test_sonic_torch_foothold_actions.py
git commit -m "feat: index two-contact terrain actions"
```

### Task 2: Heightmap foothold plan and first-contact filter

**Files:**
- Modify: `sonic/python/mm_sonic/torch_foothold_actions.py`
- Modify: `tests/python/test_sonic_torch_foothold_actions.py`

**Interfaces:**
- Consumes: current FK feet, query `_TorchHeightGrid`, terrain alignment, command velocity, and `FootholdActionIndex`.
- Produces: `FootholdPlan`, `plan_footholds` and `first_contact_eligibility`, both with the exact signatures shown below.

- [ ] **Step 1: Write failing edge/timing tests**

```python
def test_flat_approach_cannot_select_raised_first_landing():
    plan = plan_footholds(
        foot_xy_m=torch.tensor(((0., -.1), (0., .1))),
        support_mask=torch.tensor((True, True)),
        command_xy=torch.tensor((1., 0.)),
        sample_surface=step_surface(riser_x=.50, height=.18),
        reachable_forward_m=(.20, .40),
        lateral_samples_m=(-.10, 0., .10),
        edge_margin_m=.04,
    )
    assert torch.all(plan.landing_height_delta_m[:, 0] == 0.0)

def test_first_contact_filter_is_hard_not_weighted():
    eligible = first_contact_eligibility(
        plan=flat_plan,
        actions=(raised_action_with_zero_pose_cost, flat_action),
        height_tolerance_m=.04,
        xy_tolerance_m=.12,
        timing_tolerance_frames=8,
    )
    assert eligible.tolist() == [False, True]
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_foothold_actions`

Expected: missing `plan_footholds` and `first_contact_eligibility`.

- [ ] **Step 3: Implement deterministic foothold enumeration**

```python
@dataclass(frozen=True)
class FootholdPlan:
    moving_foot: int
    landing_xy_world_m: torch.Tensor  # (beam, 2, 2)
    landing_height_delta_m: torch.Tensor  # (beam, 2)
    score: torch.Tensor  # (beam,)

def plan_footholds(
    foot_xy_m: torch.Tensor,
    support_mask: torch.Tensor,
    command_xy: torch.Tensor,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    reachable_forward_m: Sequence[float],
    lateral_samples_m: Sequence[float],
    edge_margin_m: float,
    beam_width: int = 8,
) -> FootholdPlan:
    # Enumerate source-observed forward/lateral reach samples, query center and
    # four sole-corner heights, reject edge/height disagreement, alternate the
    # moving foot, then retain stable lexicographically sorted two-step beams.
```

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_foothold_actions`

Expected: all tests pass, including the premature-step negative control.

- [ ] **Step 5: Commit the planner/filter**

```bash
git add sonic/python/mm_sonic/torch_foothold_actions.py tests/python/test_sonic_torch_foothold_actions.py
git commit -m "feat: plan and filter terrain footholds"
```

### Task 3: Two-contact and hybrid ranking arms

**Files:**
- Modify: `sonic/python/mm_sonic/torch_foothold_actions.py`
- Modify: `tests/python/test_sonic_torch_foothold_actions.py`

**Interfaces:**
- Consumes: `FootholdPlan`, action descriptors, database-row motion costs, and transition costs.
- Produces: `FootholdSelectionArm` enum and `rank_foothold_actions` returning `FootholdRanking` with row eligibility and additive costs.

- [ ] **Step 1: Write failing arm-ablation tests**

```python
def test_two_contact_arm_rejects_wrong_second_tread():
    ranking = rank_foothold_actions(
        arm=FootholdSelectionArm.TWO_CONTACT,
        plan=two_step_up_plan,
        actions=(right_first_wrong_second, right_both),
        motion_cost=torch.tensor((0., 100.)),
    )
    assert ranking.eligible.tolist() == [False, True]

def test_hybrid_breaks_feasible_tie_with_motion_cost():
    ranking = rank_foothold_actions(
        arm=FootholdSelectionArm.HYBRID,
        plan=two_step_flat_plan,
        actions=(rough_pose, continuous_pose),
        motion_cost=torch.tensor((8., 1.)),
    )
    assert ranking.selected_action == 1
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_foothold_actions`

Expected: missing ranking types/functions.

- [ ] **Step 3: Implement four explicit arms**

```python
class FootholdSelectionArm(Enum):
    FIRST_CONTACT = "first-contact"
    TWO_CONTACT = "two-contact"
    HYBRID = "hybrid"
    CONTINUOUS = "continuous-control"

@dataclass(frozen=True)
class FootholdRanking:
    eligible: torch.Tensor
    additional_cost: torch.Tensor
    selected_action: int | None

# FIRST_CONTACT gates contact 0; TWO_CONTACT gates contacts 0 and 1; HYBRID
# uses the TWO_CONTACT gate and adds existing motion/transition costs;
# CONTINUOUS makes all topology-compatible rows eligible and adds descriptor
# residuals, providing the negative-control arm.
```

- [ ] **Step 4: Run all inventory/planner/ranking tests**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_foothold_actions`

Expected: all tests pass with deterministic ties.

- [ ] **Step 5: Commit ranking arms**

```bash
git add sonic/python/mm_sonic/torch_foothold_actions.py tests/python/test_sonic_torch_foothold_actions.py
git commit -m "feat: compare foothold selection arms"
```

### Task 4: Transactional matcher integration

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `sonic/python/mm_sonic/torch_contact_segments.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Consumes: `FootholdActionPolicy.prepare(state, shaped, database)` returning row eligibility, row costs, and diagnostics.
- Produces: optional `foothold_action_policy` constructor argument and diagnostics for planned/selected contacts, coverage, and rejection reason.

- [ ] **Step 1: Write failing transactional matcher tests**

```python
def test_infeasible_low_cost_terrain_row_cannot_win(self):
    matcher = make_matcher(
        foothold_policy=fake_policy(eligible_rows=[False, True])
    )
    result = matcher.step((1.0, 0.0), 0.0, dt=.02)
    self.assertEqual(result.diagnostics.selected_frame, feasible_high_cost_frame)

def test_no_foothold_candidate_retains_state_transactionally(self):
    matcher = make_matcher(foothold_policy=fake_policy(eligible_rows=[False]*4))
    before = matcher.state_snapshot()
    prepared = matcher.prepare_step((0.0, -1.0), -math.pi / 2, dt=.02)
    self.assertEqual(matcher.state_snapshot(), before)
    with self.assertRaisesRegex(ContractError, "no feasible foothold action"):
        matcher.commit(prepared)
```

- [ ] **Step 2: Run focused matcher tests and confirm RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_motion_matcher`

Expected: unknown `foothold_action_policy` and missing diagnostics.

- [ ] **Step 3: Integrate eligibility before exact ranking**

```python
foothold = (
    None if self._foothold_action_policy is None
    else self._foothold_action_policy.prepare(state, shaped, self.database)
)
eligible = self._command_transition_eligibility(state, shaped)
costs = transition_costs
if foothold is not None:
    eligible = eligible & foothold.row_eligibility
    costs = costs + foothold.additional_row_cost
```

Apply the same eligibility and costs to ordinary selection, ranked terrain
rescue, and any transition reranking. Never apply them to exact sequential
playback inside a committed source action.

- [ ] **Step 4: Run matcher, contact, and terrain regression tests**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_motion_matcher tests.python.test_sonic_torch_contact_segments tests.python.test_sonic_torch_terrain_features`

Expected: all tests pass and legacy behavior is identical with no policy.

- [ ] **Step 5: Commit integration**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py sonic/python/mm_sonic/torch_contact_segments.py tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: condition terrain matching on foothold plans"
```

### Task 5: Coverage report and four-arm adversarial evaluation

**Files:**
- Create: `resources/run_g1_torch_foothold_ablation.py`
- Create: `tests/python/test_run_g1_torch_foothold_ablation.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_omni_metrics.py`
- Modify: `tests/python/test_sonic_torch_terrain_omni_metrics.py`
- Create: `docs/superpowers/results/2026-07-31-g1-foothold-conditioned-ablation.md`

**Interfaces:**
- Consumes: frozen route definitions, feedback corpus, four selection arms, and existing exact metrics.
- Produces: one JSON result per arm plus a deterministic comparison report and winning config.

- [ ] **Step 1: Write failing metric/report tests**

```python
def test_premature_changed_height_contact_fails_route():
    metrics = evaluate_route(fake_route(first_riser_x=.5, first_up_contact_x=.3))
    assert not metrics.first_contact_timing_pass

def test_selector_orders_safety_before_route_count():
    winner = select_winner((unsafe_20_of_21, safe_16_of_21))
    assert winner.name == "safe-16"
```

- [ ] **Step 2: Run report tests and confirm RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_run_g1_torch_foothold_ablation tests.python.test_sonic_torch_terrain_omni_metrics`

Expected: missing timing metric and selector.

- [ ] **Step 3: Implement coverage-first ablation runner**

```python
ARMS = tuple(FootholdSelectionArm)
for arm in ARMS:
    report = run_matrix(dataset=args.dataset, arm=arm, routes=args.routes)
    write_json(args.output / arm.value / "matrix.json", report)
winner = select_winner(reports)  # safety, classes, freeze, timing, slide, pose
write_json(args.output / "comparison.json", winner.as_dict())
```

Before behavioral routes, emit counts keyed by start topology, next-foot pair,
quantized landing direction, and quantized two-contact height deltas. Abort an
arm with `coverage_failure` when no source descriptor matches a required route
signature.

- [ ] **Step 4: Run synthetic/report tests, then the real four-arm matrix**

Run tests:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_run_g1_torch_foothold_ablation \
  tests.python.test_sonic_torch_terrain_omni_metrics
```

Run real ablations on separate GPUs:

```bash
for spec in '0 first-contact' '1 two-contact' '2 hybrid' '3 continuous-control'; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=sonic/python:. \
    sonic/.torch-mm-venv/bin/python -B \
    resources/run_g1_torch_foothold_ablation.py \
    --dataset build/torch-grail-terrain-feedback-v2 \
    --config sonic/configs/experiments/torch_grail_representative.json \
    --arm $2 --output build/foothold-ablation/$2 &
done
wait
```

Expected: four complete machine-readable reports, no safety regression in the
retained arm, and an explicit winner or explicit descriptor coverage failure.

- [ ] **Step 5: Inspect adversarial traces and write the result**

The report must include each route class, premature-contact timing, exact first
rejection, freeze frames, support/penetration/slide, selected action identities,
coverage holes, and the reason the winning arm beat every alternative.

- [ ] **Step 6: Run the full focused suite and commit evidence**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_foothold_actions \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_contact_segments \
  tests.python.test_sonic_torch_terrain_features \
  tests.python.test_sonic_torch_terrain_omni_metrics \
  tests.python.test_run_g1_torch_foothold_ablation
```

Expected: all focused tests pass.

```bash
git add resources/run_g1_torch_foothold_ablation.py \
  tests/python/test_run_g1_torch_foothold_ablation.py \
  sonic/python/mm_sonic/torch_terrain_omni_metrics.py \
  tests/python/test_sonic_torch_terrain_omni_metrics.py \
  docs/superpowers/results/2026-07-31-g1-foothold-conditioned-ablation.md
git commit -m "research: compare foothold-conditioned terrain matching"
```

### Task 6: Feedback-ready viewer qualification

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`
- Create: `sonic/configs/experiments/torch_grail_foothold_winner.json`

**Interfaces:**
- Consumes: the winning arm/config and policy.
- Produces: viewer overlay showing planned contacts and exact rejection state.

- [ ] **Step 1: Write a failing overlay/config test**

```python
def test_overlay_names_foothold_arm_and_planned_landings():
    left, right = overlay(result_with_foothold_plan, command)[2:]
    assert "foothold=hybrid" in right
    assert "next=R:+0.18m,L:+0.18m" in right
```

- [ ] **Step 2: Run the viewer test and confirm RED**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_terrain_live_viewer`

Expected: overlay lacks foothold diagnostics.

- [ ] **Step 3: Add only the winning policy and diagnostics to the viewer**

```python
right += (
    f"\nfoothold={diagnostic.foothold_arm} "
    f"coverage={diagnostic.foothold_candidate_count}"
    f"\nnext={diagnostic.foothold_landing_summary}"
)
```

- [ ] **Step 4: Re-run qualification and launch one viewer**

Run: `PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_terrain_live_viewer`

Then launch exactly one viewer with the winning config. Expected: terrain and
robot visible, WASD responsive, and no old viewers running.

- [ ] **Step 5: Commit the qualified viewer**

```bash
git add sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py \
  sonic/configs/experiments/torch_grail_foothold_winner.json
git commit -m "feat: visualize qualified foothold matcher"
```
