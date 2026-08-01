# G1 PHP-Style Terrain Skill Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the existing PyTorch motion matcher to enter coherent GRAIL terrain skills and replay each selected skill sequentially with inertialization.

**Architecture:** Extract immutable whole-terrain episodes and PHP-style entry windows from authenticated support/height profiles. Search only compatible entry rows with the existing 27-value matcher, then commit sequential source playback until a stable exit. Qualify a six-route falsification slice before running the full 21-route matrix.

**Tech Stack:** Python 3.10, PyTorch, NumPy, MuJoCo FK, `unittest`, authenticated GRAIL 50 Hz motion/height-grid artifacts.

## Global Constraints

- Work only on `research/g1-torch-terrain-kinematics` in the existing linked worktree.
- Do not stage or modify the dirty landing-bridge, FK, or contact-segment files already present.
- Do not change retained matcher defaults, Sonic, tracking, depth, or physics.
- Use the unchanged 21 same-stair routes and existing terrain metrics.
- No unchecked fallback, mid-flight skill boundary, or timing-dependent hash.

---

### Task 1: Deterministic terrain-skill inventory

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skills.py`
- Create: `tests/python/test_sonic_torch_terrain_skills.py`

**Interfaces:**
- Consumes: `TerrainDataset`, per-clip boolean support masks, paired height grids.
- Produces: `TerrainSkill`, `TerrainSkillInventory`, `extract_skill_intervals`, and `build_terrain_skill_inventory`.

- [ ] **Step 1: Write failing pure interval tests**

```python
def test_extracts_one_complete_elevated_episode_with_entry_window(self):
    support = torch.ones((30, 2), dtype=torch.bool)
    heights = torch.zeros((30, 2))
    heights[10:22] = 0.18
    self.assertEqual(
        extract_skill_intervals(
            support, heights, valid_frame_stop=25,
            entry_window_frames=5, minimum_surface_change_m=0.08,
        ),
        (SkillInterval(4, 9, 23),),
    )

def test_never_uses_flight_as_a_skill_boundary(self):
    support = torch.ones((30, 2), dtype=torch.bool)
    support[9:12, 1] = False
    heights = torch.zeros((30, 2)); heights[12:22] = 0.18
    interval = extract_skill_intervals(
        support, heights, valid_frame_stop=25,
        entry_window_frames=5, minimum_surface_change_m=0.08,
    )[0]
    self.assertTrue(bool(support[interval.playback_start].all()))
    self.assertTrue(bool(support[interval.playback_stop - 1].all()))
```

- [ ] **Step 2: Verify RED**

Run:
`PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v tests.python.test_sonic_torch_terrain_skills`

Expected: import failure for `mm_sonic.torch_terrain_skills`.

- [ ] **Step 3: Implement the immutable inventory**

```python
@dataclass(frozen=True, order=True)
class SkillInterval:
    entry_start: int
    playback_start: int
    playback_stop: int

@dataclass(frozen=True)
class TerrainSkill:
    skill_index: int
    clip_index: int
    interval: SkillInterval
    entry_rows: tuple[int, ...]
    support_mask: torch.Tensor
    foot_surface_height_m: torch.Tensor

@dataclass(frozen=True)
class TerrainSkillInventory:
    skills: tuple[TerrainSkill, ...]
    rejected_by_reason: Mapping[str, int]
    row_to_skill: Mapping[int, int]
```

`extract_skill_intervals` finds changed-height supported frames, groups them
until a 50-frame stable baseline gap, expands each group to stable
double-support boundaries, and rejects boundaries outside searchable rows.
`build_terrain_skill_inventory` samples each source foot through its paired
grid/alignment, maps entry frames through `database.row_for_source`, owns all
tensors, and records deterministic rejection counts.

- [ ] **Step 4: Verify GREEN and regress neighboring extraction**

Run the new module plus `tests.python.test_sonic_torch_contact_segments`.
Expected: all tests pass; the authenticated opt-in test may skip.

- [ ] **Step 5: Commit only the two new files**

```bash
git add sonic/python/mm_sonic/torch_terrain_skills.py tests/python/test_sonic_torch_terrain_skills.py
git commit -m "feat: index coherent GRAIL terrain skills"
```

### Task 2: PHP-style entry selection

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skill_search.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_search.py`

**Interfaces:**
- Consumes: normalized 27-value query, inventory, current joint state, target-terrain validator callback.
- Produces: `TerrainSkillCandidate`, `TerrainSkillSearchResult`, `skill_entry_eligibility`, `select_terrain_skill`.

- [ ] **Step 1: Write failing tests for eligibility and gate order**

```python
def test_only_php_entry_windows_are_searchable(self):
    eligible = skill_entry_eligibility(database, inventory)
    self.assertEqual(torch.nonzero(eligible).flatten().tolist(), [2, 3, 7])

def test_terrain_gate_runs_before_feature_ranking(self):
    result = select_terrain_skill(
        database, inventory, normalized_query,
        current_clip_index=0, current_frame_index=0,
        terrain_validator=lambda skill, row: skill.skill_index == 1,
    )
    self.assertEqual(result.skill.skill_index, 1)
    self.assertEqual(result.rejected_by_reason, {"terrain": 1})
```

- [ ] **Step 2: Verify RED**

Run the new test module; expect missing-module failure.

- [ ] **Step 3: Implement selection using the existing matcher primitive**

Build one eligibility mask per feasible skill. Call
`rank_exact_transition_candidates` with the union entry mask, validate ranked
rows in deterministic order, and return the first terrain-valid row. Map the
row back through `inventory.row_to_skill`. Do not copy or rewrite nearest-neighbor
math.

```python
@dataclass(frozen=True)
class TerrainSkillSearchResult:
    skill: TerrainSkill
    selected_row: int
    selected_feature_cost: float
    rejected_by_reason: Mapping[str, int]
```

- [ ] **Step 4: Verify GREEN and matcher parity**

Run the new tests and `tests.python.test_sonic_torch_motion_matcher`.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_search.py tests/python/test_sonic_torch_terrain_skill_search.py
git commit -m "feat: select PHP-style terrain skill entries"
```

### Task 3: Sequential skill executor with inertialization

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skill_composer.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_composer.py`

**Interfaces:**
- Produces: `TerrainSkillState`, `TerrainSkillFrame`, `start_skill`, `advance_skill`, `can_interrupt_skill`.

- [ ] **Step 1: Write failing transactional playback tests**

```python
def test_committed_skill_advances_every_source_frame_once(self):
    state = start_skill(skill, selected_entry_frame=8, current=pose)
    frames = []
    for _ in range(5):
        step = advance_skill(state)
        state = step.state
        frames.append(step.frame)
    self.assertEqual([frame.source_frame for frame in frames], [10, 11, 12, 13, 14])

def test_interrupt_is_allowed_only_at_stable_double_support(self):
    self.assertFalse(can_interrupt_skill(skill, 12))
    self.assertTrue(can_interrupt_skill(skill, 20))
```

- [ ] **Step 2: Verify RED**

Run the new module; expect missing-module failure.

- [ ] **Step 3: Implement placement and ten-frame decay**

Rigidly place the source root at the selected entry root. Compute joint/root
position and velocity offsets from the current pose, then evaluate the existing
`decay_spring_offsets` at `frame_age / 50`. Apply the offsets to sequential
source frames only. Preserve the source full root quaternion with a yaw offset.
Return immutable frames containing qpos, joints, root, feet, support, source
provenance, and inertialization residual.

- [ ] **Step 4: Verify GREEN, continuity, and FK**

Tests must assert source indices are consecutive, the first emitted pose equals
the current pose within `1e-6`, residual magnitude decreases monotonically, and
MuJoCo FK feet match the emitted feet within `1e-5 m`.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_composer.py tests/python/test_sonic_torch_terrain_skill_composer.py
git commit -m "feat: replay terrain skills sequentially"
```

### Task 4: Rapid six-route qualification slice

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skill_rollout.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_rollout.py`
- Create: `resources/run_g1_torch_terrain_skills.py`
- Create: `tests/python/test_run_g1_torch_terrain_skills.py`
- Create: `sonic/configs/experiments/torch_grail_php_skills.json`

**Interfaces:**
- Runs `cross-tread-left-to-right`, `turn-90-middle-left`, `diagonal-down-left`, `side-exit-upper-left`, `riser-reversal`, and `mixed-adversarial` first.

- [ ] **Step 1: Write failing rollout and CLI contract tests**

Require exact route ordering, structured no-skill failure, no mid-flight stop,
pickle-free arrays, timing-excluded deterministic hashes, and route selection.

- [ ] **Step 2: Verify RED**

Run both new test modules; expect missing imports.

- [ ] **Step 3: Implement the minimal offline rollout**

At reset, ordinary matcher state supplies the query. Before terrain entry, rank
skill entries against the full scripted route trajectory, choose one feasible
skill, and replay it sequentially. At stable double support, re-search only when
the command segment changed. Save qpos plus source/skill/command/contact and cost
diagnostics. Reuse `evaluate_omnidirectional_route` for outcomes.

- [ ] **Step 4: Run unit tests and the six-route GPU slice**

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_torch_terrain_skills.py \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_php_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:0 --qualification-slice \
  --output build/php-terrain-skills/slice-v1
```

Continue only if at least four of six routes execute without exception, at
least two pass behavior, all FK/contact gates pass, and visual boundary jumps
are below the rigid oracle. Otherwise record the falsifier before adding viewer
or full-matrix machinery.

- [ ] **Step 5: Commit code/config/tests, never build artifacts**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_rollout.py \
  tests/python/test_sonic_torch_terrain_skill_rollout.py \
  resources/run_g1_torch_terrain_skills.py \
  tests/python/test_run_g1_torch_terrain_skills.py \
  sonic/configs/experiments/torch_grail_php_skills.json
git commit -m "feat: evaluate PHP-style terrain skills"
```

### Task 5: Full matrix, viewer, and evidence

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_skill_viewer.py`
- Create: `tests/python/test_sonic_torch_terrain_skill_viewer.py`
- Create: `docs/superpowers/results/2026-08-01-g1-php-terrain-skills.md`

- [ ] **Step 1: Add strict saved-route loader and passive MuJoCo viewer tests**
- [ ] **Step 2: Implement exact-qpos playback with pause, stepping, reset, and exit**
- [ ] **Step 3: Run all focused and neighboring tests**
- [ ] **Step 4: Run all 21 routes across available GPUs and authenticate artifacts**
- [ ] **Step 5: Visually review cross, turn, descent, side-exit, reversal, and mixed routes**
- [ ] **Step 6: Record pass counts, safety, slide, stalls, hashes, and the explicit result classification**
- [ ] **Step 7: Commit viewer/tests and then the evidence report in separate commits**

The result qualifies only at the thresholds in the design. If the six-route
slice fails its continuation gate, Task 5 becomes a failure report rather than
an expanded implementation.
