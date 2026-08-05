# G1 Segmented Horizontal Grid Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report and retrieve mount, reusable interior walking, and dismount coverage independently for every 20 cm horizontal staircase lane.

**Architecture:** A pure terrain-phase module converts the two nominal foot-height tracks into three phase intervals and normalized signatures. A separate experiment runner scores the existing unedited raw GRAIL windows for every phase, certifies each candidate with the existing rigid placement gates, and emits phase-level and lane-level coverage without changing the original whole-window baseline.

**Tech Stack:** Python 3, NumPy, PyTorch, MuJoCo G1 sole kinematics, `unittest`, existing GRAIL dataset and rigid path-placement primitives.

## Global Constraints

- Derive boundary positions from sampled terrain rather than fixed staircase coordinates.
- Remove only a common vertical offset from interior signatures.
- Never edit source joint positions or move feet independently.
- Keep the existing heading, path-tube, stance-height, and sole-clearance gates.
- Report mount, interior, and dismount independently.

---

### Task 1: Pure terrain phase contract

**Files:**
- Create: `sonic/python/mm_sonic/torch_path_terrain_phases.py`
- Create: `tests/python/test_sonic_torch_path_terrain_phases.py`

**Interfaces:**
- Produces: `TerrainPathPhase`, `segment_path_surface(...)`, `classify_phase_coverage(...)`
- Consumes: a straight path, nominal step width, and NumPy terrain sampler.

- [ ] **Step 1: Write failing segmentation and normalization tests**

Test a synthetic ground/elevated/ground profile, two vertically translated
profiles, a split-height interior, and partial lane classification.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_path_terrain_phases -v
```

Expected: fail because `mm_sonic.torch_path_terrain_phases` does not exist.

- [ ] **Step 3: Implement the smallest pure phase module**

Implement immutable validated phase records. Sample the two foot tracks at
`0.01 m`, find the elevated connected component, identify the maximal stable
interior, and include terrain-derived context around both boundaries.
Normalize each signature by subtracting its first two-foot mean while
preserving the left-right split.

- [ ] **Step 4: Verify GREEN and commit**

Run the targeted tests, then:

```bash
git add sonic/python/mm_sonic/torch_path_terrain_phases.py \
  tests/python/test_sonic_torch_path_terrain_phases.py
git commit -m "feat: segment horizontal terrain paths"
```

### Task 2: Phase-aware raw-window evaluator

**Files:**
- Create: `resources/run_g1_horizontal_grid_phase_coverage.py`
- Create: `tests/python/test_run_g1_horizontal_grid_phase_coverage.py`
- Reuse: `resources/run_g1_horizontal_grid_coverage.py`
- Reuse: `sonic/python/mm_sonic/torch_path_motion_placement.py`

**Interfaces:**
- Consumes: `TerrainPathPhase` records and the existing raw motion corpus.
- Produces: `phase-grid-summary.json` plus one placement report and traversal
  connector per certified phase.

- [ ] **Step 1: Write failing runner contract tests**

Test phase query construction for two-contact interiors, height-offset
invariance, independent phase classification, and deterministic report
ordering.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_horizontal_grid_phase_coverage -v
```

Expected: fail because the phase runner does not exist.

- [ ] **Step 3: Implement shared scan and certification**

Extract each raw source once, score it against all lane-phase query sets, keep
bounded diverse shortlists, and rigidly certify each phase at its actual path
position. Preserve independent rejection counts and raw source identity.

- [ ] **Step 4: Verify GREEN and commit**

Run the new runner tests plus the existing placement/grid tests, then:

```bash
git add resources/run_g1_horizontal_grid_phase_coverage.py \
  tests/python/test_run_g1_horizontal_grid_phase_coverage.py
git commit -m "feat: evaluate segmented horizontal grid coverage"
```

### Task 3: Fixed-grid experiment and evidence

**Files:**
- Create: `docs/superpowers/results/2026-08-05-g1-segmented-horizontal-grid-coverage.md`
- Generate: `build/g1-horizontal-grid-phase-20cm-v1/phase-grid-summary.json`

**Interfaces:**
- Consumes: the fixed target scene and full height-grid GRAIL corpus.
- Produces: reproducible phase coverage counts and selected source identities.

- [ ] **Step 1: Run the exact 11-lane experiment**

Use the same `x`, `y`, spacing, target scene, thresholds, and corpus as the
whole-window baseline.

- [ ] **Step 2: Repeat for determinism**

Run into a second output directory and compare the JSON and connector hashes.

- [ ] **Step 3: Inspect the expected correction**

Confirm that interior coverage is reported independently from mount/dismount
and that vertically translated equivalent interiors can reuse the same source
window when certification allows it.

- [ ] **Step 4: Record evidence and commit**

Document exact counts, clips, frames, rejection modes, hashes, and limitations.

### Task 4: Full verification

**Files:**
- Verify only.

- [ ] **Step 1: Run focused and regression tests**

Run phase, grid, raw-placement, contact-window, and viewer test modules.

- [ ] **Step 2: Run syntax and diff checks**

Compile the new Python modules and run `git diff --check`.

- [ ] **Step 3: Inspect generated artifacts**

Validate every report schema, referenced connector, and lane/phase ordering.
