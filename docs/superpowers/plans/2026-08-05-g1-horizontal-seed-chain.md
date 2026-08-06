# G1 Horizontal Seed Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reusable mount/travel/dismount seed chains for MotionBricks.

**Architecture:** A small packager computes interval boundaries, validates maximum gaps, and concatenates unchanged connectors into a held viewer playlist. Source motions remain immutable.

**Tech Stack:** Python 3.10, NumPy, unittest, JSON, existing G1 NPZ connectors.

## Global Constraints

- Seed order is exactly `mount`, `interior`, `dismount`.
- Maximum uncovered boundary is 0.40 m.
- Seed interiors are not blended or edited.
- The `+0.6 m` dismount remains labeled as a visual reference exception.

---

### Task 1: Boundary contract

**Files:**
- Create: `resources/run_g1_horizontal_seed_chains.py`
- Create: `tests/python/test_run_g1_horizontal_seed_chains.py`

**Interfaces:**
- Produces: `_chain_boundaries(seeds, maximum_gap_m) -> tuple[dict, ...]`.

- [ ] Write failing tests for overlap, short gap, excessive gap, and invalid seed order.
- [ ] Run the focused unittest and verify RED.
- [ ] Implement the minimum boundary contract.
- [ ] Run the focused unittest and verify GREEN.

### Task 2: Immutable playlist packager

**Files:**
- Modify: `resources/run_g1_horizontal_seed_chains.py`
- Test: `tests/python/test_run_g1_horizontal_seed_chains.py`

**Interfaces:**
- Produces NPZ playlist arrays and JSON chain metadata.

- [ ] Write failing tests for immutable segment concatenation and teleport boundaries.
- [ ] Implement connector validation, holds, playlist generation, and metadata output.
- [ ] Run focused and related phase-grid tests.

### Task 3: Three-lane chain generation

**Files:**
- Generate: `build/g1-horizontal-seed-chains-v1/`

- [ ] Generate chains for `+0.6`, `+0.8`, and `+1.0`.
- [ ] Verify every boundary gap is at most 0.40 m.
- [ ] Render and inspect contact sheets, then launch the complete playlist.

