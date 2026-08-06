# G1 Semantic Dismount Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select only controlled, terrain-descending G1 dismount seeds.

**Architecture:** Add a source-contact semantic predicate and route dismount tasks through it before existing signature scoring and terrain placement. Preserve stable mount routing and unchanged interior routing.

**Tech Stack:** Python 3.10, NumPy, unittest, existing GRAIL phase-grid search.

## Global Constraints

- Minimum initial supported run: 8 frames.
- Minimum downward contact-height change: 0.12 m.
- Maximum landing-surface difference: 0.04 m.
- Maximum consecutive unsupported run: 20 frames.
- Minimum terminal supported run: 8 frames.
- Both feet must contact the lower surface.

---

### Task 1: Dismount semantic predicate

**Files:**
- Modify: `resources/run_g1_horizontal_grid_phase_coverage.py`
- Test: `tests/python/test_run_g1_horizontal_grid_phase_coverage.py`

**Interfaces:**
- Consumes: `RawMotionWindow`, source support `(F, 2)`, and source surface heights `(F, 2)`.
- Produces: `_semantic_dismount_window(window, source_support, source_surface) -> bool`.

- [ ] Add failing tests for the approved descent profile and four rejection cases: flat motion, excessive flight, one-foot landing, and unsupported ending.
- [ ] Run the focused unittest and verify the missing predicate fails.
- [ ] Implement event-height, support-run, flight-run, and two-foot landing checks.
- [ ] Run the focused unittest and verify all cases pass.

### Task 2: Dismount-only routing

**Files:**
- Modify: `resources/run_g1_horizontal_grid_phase_coverage.py`
- Test: `tests/python/test_run_g1_horizontal_grid_phase_coverage.py`

**Interfaces:**
- Consumes: phase kind plus source windows, support, and surface profiles.
- Produces: stable windows for mount, semantic descents for dismount, and all windows for interior.

- [ ] Extend the phase-routing test to prove each phase receives the correct window set.
- [ ] Run the focused unittest and verify RED.
- [ ] Route dismount scoring through the semantic subset and pass source surfaces from `_scan_one`.
- [ ] Run the phase-grid and feature-seed focused tests and verify GREEN.

### Task 3: Full-grid dismount qualification

**Files:**
- Generate: `build/g1-horizontal-grid-phase-20cm-stable-dismount-v1/`

**Interfaces:**
- Consumes: full GRAIL terrain corpus and the target staircase.
- Produces: regenerated phase grid, direct dismount audit, and full viewer playlist.

- [ ] Rerun the complete 20 cm phase grid with 32 workers and the existing placement settings.
- [ ] Audit every selected dismount against all semantic thresholds directly from source profiles.
- [ ] Concatenate every admitted dismount with start/end holds and teleport boundaries.
- [ ] Launch the complete playlist at 50 Hz for visual feedback.

