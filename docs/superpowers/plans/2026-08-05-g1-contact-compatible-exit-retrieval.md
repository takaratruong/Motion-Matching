# G1 Contact-Compatible Exit Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieve and splice a natural raw GRAIL exit whose opening contact
state matches the accepted horizontal walk.

**Architecture:** Extend the existing seed-boundary ranker to accept placed
sole trajectories and enforce a hard contact-geometry gate. Use that gate while
selecting an already terrain-certified exit candidate, then assemble and
visually audit the route.

**Tech Stack:** Python 3, NumPy, MuJoCo G1 forward kinematics, unittest.

## Global Constraints

- Kinematic-only; do not integrate Sonic or physics.
- Preserve raw GRAIL joint trajectories.
- Do not use IK, MotionBricks generation, or ARDY generation.
- Do not claim success until the rendered complete route is inspected.

---

### Task 1: Contact-geometry boundary gate

**Files:**
- Modify: `sonic/python/mm_sonic/torch_seed_chain_splicing.py`
- Test: `tests/python/test_sonic_torch_seed_chain_splicing.py`

**Interfaces:**
- Consumes: `SeedMotion` and `(frames, 2, 3)` placed sole centers.
- Produces: `select_boundary_compatible_motion(..., outgoing_foot_position_world, candidate_foot_positions_world, maximum_support_foot_gap_m)`.

- [ ] Add a failing test where pose/support-only ranking accepts a candidate
  whose planted foot jumps vertically.
- [ ] Run the focused test and confirm it fails for the missing footprint gate.
- [ ] Add footprint inputs and reject cuts whose common support foot moves more
  than the configured tolerance.
- [ ] Run the focused splicing tests and confirm they pass.

### Task 2: Use placed soles during terrain exit selection

**Files:**
- Modify: `resources/run_g1_horizontal_grid_coverage.py`
- Test: `tests/python/test_run_g1_horizontal_grid_coverage.py`

**Interfaces:**
- Consumes: placed candidate roots/joints/quaternions and G1 sole kinematics.
- Produces: only contact-compatible selected exit candidates.

- [ ] Add a failing selection test with one low-cost incompatible exit and one
  higher-cost contact-compatible exit.
- [ ] Run the focused test and confirm the incompatible exit is selected.
- [ ] Compute matcher-frame sole centers for the outgoing seed and each placed
  candidate, and pass them to the boundary selector.
- [ ] Run the focused coverage tests and confirm the compatible exit is selected.

### Task 3: Build and audit the single route

**Files:**
- Create: `resources/run_g1_contact_compatible_exit_search.py`
- Test: `tests/python/test_run_g1_contact_compatible_exit_search.py`

**Interfaces:**
- Consumes: accepted prefix, cached raw candidate rows, target scene, and G1 XML.
- Produces: `traversal.npz`, `report.json`, and a rendered contact sheet.

- [ ] Add a failing test for selecting an exit from a split-height terminal
  stance and preserving the raw candidate joints.
- [ ] Run the focused test and confirm the search entry point is absent.
- [ ] Implement the minimal cached-candidate placement, contact-compatible
  selection, short blend, and provenance report.
- [ ] Run the focused test and all seed/grid search tests.
- [ ] Run the real lane-pos-0p6 search, validate the artifact, and render the
  full route.
- [ ] Inspect the render; if it fails visually, record the observed phase and
  return to retrieval rather than adding IK.
