# G1 Ordered-Contact Diagonal Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a raw-GRAIL 45-degree route that touches every crossed stair level in order and exits with both feet supported on opposite-side ground.

**Architecture:** Add a pure terrain-level contract that derives the ordered height sequence and matches actual placed touchdown events against it. Annotate each certified raw placement with its contiguous level progress, admit graph transitions only when level progress overlaps or advances by one, and accept only a chain that completes every required level and the final double-support ground exit.

**Tech Stack:** Python 3.10, NumPy, existing GRAIL placement/sole-kinematics modules, MuJoCo, `unittest`.

## Global Constraints

- Kinematic playback only; do not connect Sonic or physics tracking.
- Preserve raw GRAIL joint positions; allow only the existing contact-compatible boundary blend.
- Do not use IK, ARDY, MotionBricks, foot projection, spatial scaling, or heading-specific source identities.
- Derive required terrain levels from the supplied height grid.
- Reject skipped levels and missing exits instead of shortening the route.
- Retain the existing 0.03 m stance-error and -0.03 m sole-clearance limits.

---

### Task 1: Pure ordered terrain-level contract

**Files:**
- Create: `sonic/python/mm_sonic/torch_ordered_terrain_levels.py`
- Create: `tests/python/test_sonic_torch_ordered_terrain_levels.py`

**Interfaces:**
- Consumes: path endpoints, a dense terrain sampler, target touchdown XY positions, and target touchdown heights.
- Produces: `RequiredTerrainLevel`, `OrderedTerrainLevelContract`, `derive_ordered_terrain_levels(...)`, and `match_ordered_touchdown_levels(...)`.

- [ ] **Step 1: Write failing derivation tests**

Create a synthetic -45-degree path whose centerline surfaces are
`0.0, 0.187, 0.349, 0.0`. Assert:

```python
contract = derive_ordered_terrain_levels(
    path_start_scene_xy=(-0.5, 1.4),
    path_stop_scene_xy=(0.7, 0.2),
    sample_surface=synthetic_stair_surface,
    sample_spacing_m=0.01,
    height_tolerance_m=0.03,
)
self.assertEqual(
    tuple(round(level.height_m, 3) for level in contract.levels),
    (0.000, 0.187, 0.349, 0.000),
)
self.assertTrue(contract.levels[-1].requires_double_support)
```

- [ ] **Step 2: Run derivation tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_ordered_terrain_levels -v
```

Expected: import failure because `torch_ordered_terrain_levels` does not exist.

- [ ] **Step 3: Implement immutable terrain-level records and derivation**

Implement frozen dataclasses:

```python
@dataclass(frozen=True)
class RequiredTerrainLevel:
    index: int
    height_m: float
    start_m: float
    stop_m: float
    requires_double_support: bool = False

@dataclass(frozen=True)
class OrderedTerrainLevelContract:
    path_start_scene_xy: tuple[float, float]
    path_heading_scene_xy: tuple[float, float]
    path_length_m: float
    levels: tuple[RequiredTerrainLevel, ...]
```

Dense-sample the centerline, run-length merge adjacent heights within
`height_tolerance_m`, discard components shorter than two samples, preserve a
return to an earlier height as a new level, and mark the final level as
requiring double support.

- [ ] **Step 4: Write failing actual-touchdown matching tests**

Prove that:

```python
self.assertEqual(
    match_ordered_touchdown_levels(
        contract,
        touchdown_progress_m=(0.10, 0.55, 0.90, 1.35, 1.55),
        touchdown_height_m=(0.0, 0.187, 0.349, 0.0, 0.0),
        touchdown_foot=(0, 1, 0, 1, 0),
    ),
    (0, 1, 2, 3, 3),
)
```

and that `(0.0, 0.349, 0.0, 0.0)` raises `OrderedLevelRejected` with
reason `skipped-required-level`, while a final repeated touchdown from the
same foot raises reason `missing-ground-exit`.

- [ ] **Step 5: Implement minimal actual-touchdown matching**

Assign each event to the nearest level whose height and progress interval
match. Require nondecreasing indices, increments of at most one, and two
alternating touchdowns on the final required level.

- [ ] **Step 6: Run Task 1 tests**

Run the Task 1 test module and expect all tests to pass.

---

### Task 2: Level-aware raw-placement chain

**Files:**
- Modify: `resources/run_g1_root_path_motion_search.py`
- Modify: `tests/python/test_run_g1_root_path_motion_search.py`
- Modify: `sonic/python/mm_sonic/torch_root_path_motion_graph.py`
- Modify: `tests/python/test_sonic_torch_root_path_motion_graph.py`

**Interfaces:**
- Consumes: certified placed raw windows and an `OrderedTerrainLevelContract`.
- Produces: `OrderedLevelMotionCandidate`, `select_ordered_level_motion_chain(...)`, `required-levels.json`, and a complete traversal only when all levels plus the exit are satisfied.

- [ ] **Step 1: Write failing ordered-graph tests**

Define:

```python
@dataclass(frozen=True)
class OrderedLevelMotionCandidate:
    candidate: RootPathMotionCandidate
    level_indices: tuple[int, ...]
    touchdown_feet: tuple[int, ...]
```

Test a graph where a cheaper edge reports levels `(0, 2)` and a valid pair
reports `(0, 1)` then `(1, 2, 3, 3)`. Assert the selector rejects the skipped
edge and selects the valid pair. Assert that `(0, 1, 2)` cannot terminate
because it lacks the final ground level.

- [ ] **Step 2: Run graph tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_root_path_motion_graph -v
```

Expected: import failures for the ordered candidate and selector.

- [ ] **Step 3: Implement ordered graph selection**

Reuse the existing transition and coverage checks. Carry the highest
contiguously satisfied required-level index in dynamic-programming state.
Reject an edge whose first unseen index is greater than the previous index
plus one. Return only a chain whose state reaches the final required index and
whose final level has two alternating touchdown feet.

- [ ] **Step 4: Write failing runner integration tests**

Add tests for helpers that:

- transform raw event XY positions with the certified placement yaw and
  translation;
- sample target heights at transformed event positions;
- compute event progress on the requested path;
- reject the observed `0.000 -> 0.349 m` candidate as
  `skipped-required-level`;
- require `--ordered-contact-levels` to emit `required-levels.json`;
- omit `traversal.npz` when the exit contract is incomplete.

- [ ] **Step 5: Run integration tests and verify RED**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_root_path_motion_search -v
```

Expected: failures for missing ordered-contact helpers and CLI behavior.

- [ ] **Step 6: Implement runner integration**

Add `--ordered-contact-levels`. Derive the contract before scanning, annotate
each certified candidate from its actual transformed touchdown events, record
stable ordered-level rejection reasons, call
`select_ordered_level_motion_chain`, and serialize the required and satisfied
level sequences. The legacy unflagged search remains unchanged.

- [ ] **Step 7: Run focused regression suites**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_ordered_terrain_levels \
  tests.python.test_sonic_torch_root_path_motion_graph \
  tests.python.test_run_g1_root_path_motion_search \
  tests.python.test_run_g1_path_motion_placement -v
```

Expected: all tests pass.

- [ ] **Step 8: Run the decisive 45-degree route**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_root_path_motion_search.py \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --target-scene grail-stair_p1-db7949fce1b2e48d39f4 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --path-start -0.4835621836 1.4414988213 \
  --path-stop 0.7000000000 0.2579366377 \
  --ordered-contact-levels \
  --segment-length-m 0.90 --search-stride-m 0.20 \
  --step-width-m 0.20 --workers 32 \
  --coarse-results 2400 --placement-shortlist 600 \
  --candidates-per-anchor 24 --blend-frames 4 \
  --output build/g1-root-path-motion-search-angle-neg45-ordered-exit-v1
```

Accept only an exit-zero result whose report records required and satisfied
levels `(0.000, 0.187, 0.349, 0.000)` and alternating final ground feet.

- [ ] **Step 9: Verify and visualize**

Run the four focused suites again, `git diff --check`, render a contact sheet,
inspect every required touchdown, close stale viewers, and launch the 50 Hz
MuJoCo viewer only if the numerical and visual contracts pass.
