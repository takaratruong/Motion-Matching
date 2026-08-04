# Uneven Supported G1 Step-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search authenticated GRAIL motions for a native two-contact entry whose unequal support heights match the target horizontal staircase, validate every stance and sole contact, and emit a complete kinematic playback.

**Architecture:** A focused pure module extracts complete two-contact sequences and validates placed contacts. A command-line searcher performs cheap source/target contact-pattern filtering before MuJoCo FK, rigidly places surviving sequences, ranks only fully valid results, and writes the best artifact and metrics for the existing viewer.

**Tech Stack:** Python 3.10, NumPy, PyTorch CPU height-grid sampling, MuJoCo G1 ankle and sole FK, `unittest`.

## Global Constraints

- Preserve all pre-existing dirty worktree changes and stage only files named by this plan.
- Use authenticated source motion and paired source terrain from `build/torch-grail-terrain-full-v1`.
- Require zero unsupported frames.
- Require the source and target two-contact height differences to agree within 25 mm.
- Require both final feet to contact their respective target stair surfaces for at least three consecutive frames.
- Reject any sole penetration deeper than 25 mm.
- Keep facing-to-travel error at or below 10 degrees.
- Do not use tracking, physics, IK retargeting, or cross-clip splicing in this native-motion test.

---

### Task 1: Complete Contact-Sequence Contract

**Files:**
- Create: `sonic/python/mm_sonic/torch_supported_step_up.py`
- Create: `tests/python/test_sonic_torch_supported_step_up.py`

**Interfaces:**
- Consumes: boolean `(T,2)` support masks and float `(T,2)` source terrain heights.
- Produces: `CompleteStepUpSequence`, `find_complete_step_up_sequence(...)`, and `validate_placed_step_up(...)`.

- [ ] **Step 1: Write failing extraction tests**

Add tests with synthetic support and surface arrays that require:

```python
sequence = find_complete_step_up_sequence(
    support_mask=support,
    surface_height_m=surface,
    start_frame=0,
    first_contact_frame=20,
    landing_foot=0,
    stable_contact_frames=3,
)
self.assertEqual(sequence.trailing_contact_frame, 50)
self.assertAlmostEqual(sequence.source_final_height_delta_m, 0.18)
```

Also assert rejection when the clip ends after only the first contact and when any frame between entry and completion has no support.

- [ ] **Step 2: Run extraction tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_supported_step_up
```

Expected: import failure because `mm_sonic.torch_supported_step_up` does not exist.

- [ ] **Step 3: Implement the minimal extraction API**

Create immutable result types and implement:

```python
def find_complete_step_up_sequence(
    *,
    support_mask: object,
    surface_height_m: object,
    start_frame: int,
    first_contact_frame: int,
    landing_foot: int,
    stable_contact_frames: int = 3,
    search_horizon_frames: int = 120,
) -> CompleteStepUpSequence:
    ...
```

The function validates shapes and finite values, verifies a raised leading-foot contact with lower trailing-foot support, searches for the trailing foot's next stable contact, and rejects any unsupported frame through that stable contact.

- [ ] **Step 4: Write failing placed-contact tests**

Construct synthetic ankle-clearance `(T,2)` and sole-clearance `(T,2,7)` arrays. Require acceptance for zero-flight split-height support, then separately reject:

```python
validate_placed_step_up(
    sequence=sequence,
    source_support_mask=support,
    ankle_clearance_m=ankle_clearance,
    sole_clearance_m=sole_clearance,
    target_final_height_delta_m=0.18,
)
```

- [ ] **Step 5: Run placed-contact tests and verify RED**

Run the Task 1 test command. Expected: extraction tests pass and placement tests fail because `validate_placed_step_up` is missing.

- [ ] **Step 6: Implement minimal placed-contact validation**

Return `PlacedStepUpValidation` containing unsupported-frame count, final split-support status, height-pattern error, maximum stance-contact error, and minimum sole clearance. Raise `ContractError` unless every global contact constraint passes.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run the Task 1 test command. Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_supported_step_up.py \
  tests/python/test_sonic_torch_supported_step_up.py
git commit -m "feat: validate complete uneven step-up contacts"
```

### Task 2: Native Uneven-Motion Search and Artifact

**Files:**
- Create: `resources/run_g1_supported_step_up_search.py`
- Create: `tests/python/test_run_g1_supported_step_up_search.py`

**Interfaces:**
- Consumes: authenticated source dataset, target dataset/config, G1 XML, and target landing search bounds.
- Produces: `step-up.npz`, `metrics.json`, and `ranked.json`.

- [ ] **Step 1: Write failing cheap-filter tests**

Test a pure `_contact_pattern_matches(...)` helper with source landing/trailing foot offsets and a tiny fake target sampler. Verify that a level curb continuation is rejected against a 0.17 m target split while a native 0.18 m split is retained.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_run_g1_supported_step_up_search
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement source event inventory and cheap filtering**

The runner must:

1. load every authenticated GRAIL stair/curb descriptor and paired terrain;
2. derive support from ankle height and vertical speed;
3. detect a raised first contact with the other foot still supported below;
4. call `find_complete_step_up_sequence`;
5. align source travel to target scene `+x`;
6. sample predicted target surfaces for both first and trailing contacts;
7. discard candidates whose source and target contact-height patterns differ by more than 25 mm before invoking MuJoCo FK.

- [ ] **Step 4: Run cheap-filter tests and verify GREEN**

Run the Task 2 test command. Expected: all tests pass.

- [ ] **Step 5: Implement FK placement, ranking, and output**

For each cheap-filter survivor, rotate and translate the complete source segment, match first-contact height, compute ankle and seven-point sole FK, and call `validate_placed_step_up`. Rank accepted candidates by height-pattern error, stance error, penetration, and facing error. Write the best NPZ plus auditable metrics and all accepted rankings.

- [ ] **Step 6: Run focused test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_supported_step_up \
  tests.python.test_run_g1_supported_step_up_search
```

Expected: all tests pass.

- [ ] **Step 7: Run the real native-motion search**

Run:

```bash
PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_supported_step_up_search.py \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-step-up-search/uneven-complete
```

Expected: exit 0, at least one accepted native sequence, zero unsupported
frames, and a final stable split-height contact transfer. The landing foot may
already be lifting as the trailing foot establishes its new contact; dynamic
stair walking must not be forced into double support.

- [ ] **Step 8: Commit Task 2**

```bash
git add resources/run_g1_supported_step_up_search.py \
  tests/python/test_run_g1_supported_step_up_search.py
git commit -m "feat: search native uneven stair entries"
```

### Task 3: Visual Verification and Relaunch

**Files:**
- Modify only generated files under: `build/g1-step-up-search/uneven-complete/`

**Interfaces:**
- Consumes: accepted `step-up.npz`.
- Produces: inspected contact sheet and live passive MuJoCo playback.

- [ ] **Step 1: Render the contact sheet**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_stair_pivot_viewer.py \
  --connector build/g1-step-up-search/uneven-complete/step-up.npz \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --render-contact-sheet \
  build/g1-step-up-search/uneven-complete/contact-sheet.png
```

Expected: exit 0 and a six-panel image.

- [ ] **Step 2: Inspect the rendered contact sequence**

Verify the leading foot lands on its elevated tread, the trailing foot lands on its different valid stair surface, the pelvis transfers over the staircase, and neither foot floats or penetrates visibly.

- [ ] **Step 3: Run final verification**

Run the two focused test modules again and parse `metrics.json` to assert:

```python
assert metrics["unsupported_frame_count"] == 0
assert metrics["final_split_height_contact_transfer"]
assert metrics["minimum_sole_clearance_m"] >= -0.025
assert metrics["contact_height_pattern_error_m"] <= 0.025
```

- [ ] **Step 4: Replace the old viewer**

Terminate only exact `resources/run_g1_stair_pivot_viewer.py` processes, launch the accepted uneven-complete artifact with `--loop`, and activate the new MuJoCo window.
