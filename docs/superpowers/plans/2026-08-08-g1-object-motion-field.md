# G1 Object-Local Rasterized Motion Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-heading object-local traversal graph with validated motion lines and contact-compatible transition opportunities.

**Architecture:** Reuse the terrain-derived parallel-grid geometry for each heading, combine the resulting lines in a single immutable field manifest, compute exact cross-heading segment intersections, and attach admitted route/transition artifacts without hiding missing coverage.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo G1 kinematics, GRAIL motion corpus, `unittest`.

## Global Constraints

- Initial spacing is 0.20 m and headings are runtime inputs.
- Geometry and lookup are object-local and equivariant to global translation/yaw.
- Failed line motions and transitions remain explicit; no flat substitutes.
- MotionBricks may bridge only short contact-compatible boundaries.
- Existing horizontal and diagonal artifacts are immutable inputs.

---

### Task 1: Multi-Heading Field Geometry

**Files:**
- Create: `sonic/python/mm_sonic/torch_object_motion_field.py`
- Create: `tests/python/test_sonic_torch_object_motion_field.py`

**Interfaces:**
- Consumes: elevated terrain XY, center/path length, heading degrees, spacing.
- Produces: `MotionFieldLine`, `MotionFieldIntersection`, `rasterized_motion_field(...)`, and `motion_field_intersections(...)`.

- [ ] Add failing tests for four heading families, 20 cm parallel spacing, exact cross-heading intersections, same-family exclusion, and translation/yaw equivariance.
- [ ] Run the focused test and confirm RED because the module is absent.
- [ ] Implement validated immutable records using the existing staircase grid geometry.
- [ ] Run the focused geometry suite and confirm GREEN.

### Task 2: Route Admission and Lookup Manifest

**Files:**
- Create: `resources/run_g1_object_motion_field.py`
- Create: `tests/python/test_run_g1_object_motion_field.py`

**Interfaces:**
- Consumes: geometry, route contracts, route NPZs, validation reports.
- Produces: complete field manifest, admitted-motion index, and explicit missing-line records.

- [ ] Add failing tests that reject flat, invalid, or missing routes while retaining their field cells and that admit only matching heading/terrain contracts.
- [ ] Run RED, implement manifest construction, then run GREEN.
- [ ] Ingest the retained horizontal routes and the source-preserving negative-45-degree baseline without modifying either artifact.

### Task 3: Phase-Compatible Transition Opportunities

**Files:**
- Modify: `sonic/python/mm_sonic/torch_object_motion_field.py`
- Modify: `tests/python/test_sonic_torch_object_motion_field.py`

**Interfaces:**
- Consumes: admitted line root paths, contact events, geometric intersections, search radius.
- Produces: `TransitionOpportunity` records with nearby frame pairs and hard compatibility reasons.

- [ ] Add failing tests for nearest intersection frames, support-phase compatibility, terrain-height compatibility, and explicit rejection reasons.
- [ ] Implement deterministic opportunity extraction and verify translation/yaw invariance.
- [ ] Emit opportunities for every admitted cross-heading pair.

### Task 4: Retrieve, Validate, and Publish Transition Edges

**Files:**
- Create: `resources/run_g1_motion_field_transitions.py`
- Create: `tests/python/test_run_g1_motion_field_transitions.py`
- Create: `build/g1-object-motion-field-v1/` artifacts.

**Interfaces:**
- Consumes: opportunities, GRAIL turn windows, route artifacts, target terrain, G1 XML.
- Produces: accepted transition NPZs, rejection reports, directed graph index, and visual audit playlist.

- [ ] Add failing tests proving transition splices preserve source outside local windows and reject phase, contact, heading, and continuity failures.
- [ ] Reuse contact-compatible boundary search to retrieve turns and generate only short connectors when necessary.
- [ ] Independently validate every assembled route-change sequence and render dense visual sheets.
- [ ] Publish only accepted continuation/transition edges; launch a playlist that demonstrates at least one horizontal-to-diagonal transition near an intersection.
