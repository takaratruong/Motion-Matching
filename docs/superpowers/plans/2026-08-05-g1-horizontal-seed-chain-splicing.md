# G1 Horizontal Seed-Chain Splicing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one continuous, terrain-certified horizontal traversal from
the approved mount, uneven-travel, and dismount source motions.

**Architecture:** A pure boundary module selects cut frames and assembles
immutable source segments. A focused runner uses released MotionBricks only
when direct source concatenation fails, corrects only generated frames to the
exact endpoints, and accepts output only through the existing MuJoCo
foot/sole terrain oracle.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MotionBricks, MuJoCo, unittest.

## Global Constraints

- Qualify `lane-pos-0p8` first.
- Preserve approved source motion outside selected boundary windows exactly.
- Keep one constant global heading.
- Use MotionBricks as a proposal generator, never as contact truth.
- Do not add Sonic, physics tracking, or depth input.
- Do not modify the MotionBricks checkout.
- Do not emit a route when no candidate passes the strict contract.

---

### Task 1: Boundary cut-pair and immutable assembly contract

**Files:**
- Create: `sonic/python/mm_sonic/torch_seed_chain_splicing.py`
- Create: `tests/python/test_sonic_torch_seed_chain_splicing.py`

**Interfaces:**
- Produces: `SeedMotion`, `BoundaryCut`, `rank_boundary_cuts(...)`, and
  `assemble_seed_chain(...)`.
- Consumes: validated connector arrays, inferred support masks, and generated
  transition arrays.

- [ ] **Step 1: Write failing cut-ranking tests**

Create fixtures with two supported gait phases. Prove the compatible
same-support, low-pose-error frame pair ranks before a closer world-root pair
with the wrong support identity. Prove malformed quaternions and empty support
windows fail closed.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_seed_chain_splicing -v
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement validated cut ranking**

Validate `(F,29)` joint positions, `(F,3)` roots, `(F,4)` unit quaternions,
`(F,2)` boolean support, and at least four frames on both sides of a cut.
Rank only equal non-empty support masks using weighted root distance, shortest
heading difference, joint RMS error, and joint-velocity RMS error.

- [ ] **Step 4: Write failing immutable-assembly tests**

Prove assembly removes no source frame outside the chosen cut ranges, excludes
the four context/target frames duplicated by a generated transition, records
exact source/output ranges, and rejects a transition whose endpoints differ
from the selected source poses.

- [ ] **Step 5: Implement assembly and verify GREEN**

Concatenate source prefix, transition interior, next source suffix for both
boundaries. Emit route arrays and segment metadata. Re-run the focused suite
and require `OK`.

### Task 2: MotionBricks boundary proposal runner

**Files:**
- Create: `resources/run_g1_motionbricks_seed_chain.py`
- Create: `tests/python/test_run_g1_motionbricks_seed_chain.py`

**Interfaces:**
- Consumes: one chain from `seed-chains.json`, local MotionBricks weights,
  staircase sampling configuration, and the G1 XML.
- Produces: generated transition candidates plus deterministic metrics.
- Reuses:
  `resources.run_g1_motionbricks_contact_exit.generate_route_conditioned_qpos`
  and `endpoint_warp_and_resample`.

- [ ] **Step 1: Write a failing source-preservation test**

Use a fake generator and two synthetic boundaries. Assert all generated
endpoint correction remains inside the generated transition and every retained
source frame is bit-identical.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_motionbricks_seed_chain -v
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement local model loading and duration sweep**

Load the complete local MotionBricks checkout at
`/home/ubuntu/projects/gear-sonic-worktrees/motionbricks-contact-transition/motionbricks`.
For each ranked cut pair, supply four outgoing and four incoming qpos frames.
Sweep the checkpoint's inclusive `min_tokens..max_tokens` range and resample
each proposal to 50 Hz.

- [ ] **Step 4: Implement authoritative boundary metrics**

Compute G1 foot positions and eight-point sole samples; sample the staircase
under every sole; infer support from clearance, contact sample count, and foot
velocity; calculate root/joint steps, heading, stance slide, and unsupported
runs. Record all rejection reasons before ranking candidates.

- [ ] **Step 5: Verify GREEN**

Run the Task 2 tests and Task 1 regression tests. Require `OK`.

### Task 3: Qualify the strict `lane-pos-0p8` traversal

**Files:**
- Modify: `resources/run_g1_motionbricks_seed_chain.py`
- Modify: `tests/python/test_run_g1_motionbricks_seed_chain.py`

**Interfaces:**
- Consumes:
  `build/g1-horizontal-seed-chains-v1/seed-chains.json`.
- Produces:
  `build/g1-horizontal-continuous-v1/lane-pos-0p8/`.

- [ ] **Step 1: Run the cut-pair diagnostic**

Evaluate the last/first 60 frames of each adjacent source and retain the top
eight support-compatible cut pairs per boundary. Write all raw pair metrics to
`boundary-candidates.json`.

- [ ] **Step 2: Generate one variable at a time**

Try direct concatenation first. If it fails, generate the inclusive MotionBricks
token sweep for that cut pair. Stop only after a fully accepted boundary is
found or every pair/duration is exhausted.

- [ ] **Step 3: Assemble and run strict route validation**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_validate_traversal.py \
  --input build/g1-horizontal-continuous-v1/lane-pos-0p8/traversal.npz \
  --target-dataset build/g1-horizontal-grid-phase-20cm-stable-dismount-v1/dataset \
  --config sonic/configs/experiments/torch_grail_raw_horizontal_preview.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-horizontal-continuous-v1/lane-pos-0p8/validation.json
```

Expected: exit `0` and all design thresholds pass.

- [ ] **Step 4: Render and inspect**

Render a contact sheet, inspect it for mid-air support, foot crossing,
penetration, sliding, and discontinuity, then launch the 50 Hz MuJoCo viewer.

- [ ] **Step 5: Commit the qualified implementation**

Stage only the new module, runner, tests, spec, and plan. Do not stage unrelated
dirty research files or generated build artifacts.

### Task 4: Apply the unchanged algorithm to remaining lanes

**Files:**
- Modify: `resources/run_g1_motionbricks_seed_chain.py`
- Modify: `tests/python/test_run_g1_motionbricks_seed_chain.py`

**Interfaces:**
- Consumes: `lane-pos-1p0` and `lane-pos-0p6`.
- Produces their continuous route artifacts and aggregate qualification report.

- [ ] **Step 1: Add a failing multi-lane metadata test**

Prove strict source status propagates to route metadata and the `lane-pos-0p6`
dismount remains marked as a visual exception.

- [ ] **Step 2: Implement multi-lane iteration**

Apply the exact Task 3 search and acceptance thresholds without lane-specific
frame IDs, transforms, or threshold changes.

- [ ] **Step 3: Validate and render every accepted lane**

Run the strict validator and contact-sheet renderer independently per lane.
Do not hide rejected lanes from the aggregate report.

- [ ] **Step 4: Run the complete focused regression suite**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_seed_chain_splicing \
  tests.python.test_run_g1_motionbricks_seed_chain \
  tests.python.test_run_g1_horizontal_seed_chains \
  tests.python.test_run_g1_validate_traversal -v
```

Expected: all tests pass with no warnings or errors.
