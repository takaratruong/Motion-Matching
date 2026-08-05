# G1 Horizontal Contact-Overlay Traversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one complete same-heading G1 traversal parallel to
the stair risers from a normal-walking contact overlay, GRAIL window retrieval,
and constrained generation only when retrieval lacks coverage.

**Architecture:** A contact planner converts a normal gait into sole-aware
terrain contacts. A corpus search ranks complete GRAIL windows and solves their
sequence globally. A composer applies bounded alignment/contact correction,
invokes the existing ARDY bridge only for uncovered intervals, validates the
whole route, and exports one passive MuJoCo artifact.

**Tech Stack:** Python 3, NumPy, PyTorch, MuJoCo, SciPy, authenticated GRAIL
NPZ/height grids, optional installed G1 ARDY checkpoint, unittest.

## Global Constraints

- Implement only the fixed horizontal route parallel to stair risers.
- Preserve same heading and travel direction for the complete route.
- Produce 50 Hz kinematic qpos; do not add Sonic or physics tracking.
- Contact intervals and sole transforms are authoritative.
- Reject correction above 0.35 rad per joint or 0.06 m at the root.
- Do not relax the accepted contact, clearance, sliding, or continuity limits.
- Preserve all pre-existing dirty-worktree changes.

---

### Task 1: Nominal sole-aware contact overlay

**Files:**
- Create: `sonic/python/mm_sonic/torch_horizontal_contact_overlay.py`
- Create: `tests/python/test_sonic_torch_horizontal_contact_overlay.py`

**Interfaces:**
- Consumes: flat-gait sole positions, support masks, target height sampler, and
  fixed scene-X route endpoints.
- Produces: `ContactInterval`, `HorizontalContactPlan`, and
  `horizontal_contact_plan(...)`.

- [ ] Write failing tests proving that horizontal means scene-X travel on the
  representative grid, stance soles never straddle a riser, phase/lead-foot
  variants are deterministic, and the plan contains approach, entry, two
  alternating split-height steps, exit, and departure.
- [ ] Run the focused unittest and verify failure because the module is absent.
- [ ] Implement immutable contact-plan types and bounded gait variants.
- [ ] Run the focused unittest and require `OK`.
- [ ] Commit the contact overlay and its tests.

### Task 2: Contact-window inventory and sequence search

**Files:**
- Create: `sonic/python/mm_sonic/torch_grail_contact_window_search.py`
- Create: `tests/python/test_sonic_torch_grail_contact_window_search.py`
- Create: `resources/run_g1_grail_contact_inventory.py`
- Create: `tests/python/test_run_g1_grail_contact_inventory.py`

**Interfaces:**
- Consumes: `HorizontalContactPlan` and authenticated GRAIL motion/grid pairs.
- Produces: `ContactWindow`, `WindowEdge`, `best_contact_window_path(...)`, and
  `contact-inventory.json`.

- [ ] Write failing tests for height/contact mismatch rejection, boundary
  velocity cost, deterministic dynamic programming, and explicit no-path
  failure.
- [ ] Run the focused tests and verify their expected missing-interface
  failures.
- [ ] Implement window signatures, deterministic costs, and global path search.
- [ ] Implement bounded parallel corpus scanning and authenticated inventory
  output.
- [ ] Run the full 12,646-clip inventory and retain source provenance for every
  selected interval.
- [ ] Run focused tests and require `OK`.
- [ ] Commit source search, runner, and tests.

### Task 3: Bounded composition and optional constrained fallback

**Files:**
- Create: `resources/run_g1_horizontal_contact_traversal.py`
- Create: `tests/python/test_run_g1_horizontal_contact_traversal.py`
- Modify: `resources/run_g1_ardy_terrain_bridge.py`
- Modify: `tests/python/test_run_g1_ardy_terrain_bridge.py`

**Interfaces:**
- Consumes: contact plan, selected GRAIL path, target staircase, G1 XML, and
  optional ARDY G1 proposal.
- Produces: `traversal.npz`, `contact-plan.json`, `sources.json`, and
  `metrics.json`.

- [ ] Write failing tests proving exact authored support preservation, bounded
  correction rejection, exact splice endpoints, and fallback only for an
  uncovered interval.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement rigid placement, bounded time warp, terrain-contact correction,
  and contact-preserving composition.
- [ ] Wire the installed ARDY G1 model as the missing-coverage proposal path.
- [ ] Generate the fixed horizontal route; iterate plan variants, selected
  windows, and fallback seeds until one artifact passes without threshold
  relaxation.
- [ ] Run focused tests and require `OK`.
- [ ] Commit composer, bridge changes, and tests.

### Task 4: Whole-route certification and operator playback

**Files:**
- Modify: `resources/run_g1_validate_traversal.py`
- Modify: `tests/python/test_run_g1_validate_traversal.py`
- Modify: `resources/run_g1_stair_pivot_viewer.py`
- Modify: `TORCH_TERRAIN_QUICKSTART.md`

**Interfaces:**
- Consumes: the Task 3 traversal artifact and authored contact plan.
- Produces: `validation.json`, `contact-sheet.png`, and a looped passive MuJoCo
  viewer.

- [ ] Write failing tests for required semantic phases, scene-X route
  classification, two alternating split-height steps, monotonic progress,
  heading, support, sole, sliding, correction, and derivative gates.
- [ ] Run focused tests and verify expected failures.
- [ ] Extend validation to enforce the complete horizontal-route contract.
- [ ] Render the accepted contact sheet and inspect every phase.
- [ ] Launch passive kinematic playback from the exact validated artifact.
- [ ] Run all relevant Python tests and the fresh end-to-end rebuild.
- [ ] Commit certification, documentation, and final reproducible metadata.
