# G1 Diagonal Contact-Window Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve valid GRAIL diagonal gait spans and use MotionBricks only for localized contact-invalid windows.

**Architecture:** Detect bracketed unsupported runs in the 50 Hz proxy, generate endpoint-exact MotionBricks candidates per window, splice only accepted candidates, and validate the complete assembled route. Reuse the existing exact-keyframe generator, 30-to-50 Hz endpoint resampler, sole kinematics, cadence audit, and independent traversal validator.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo, GRAIL motion archives, released MotionBricks, `unittest`.

## Global Constraints

- Do not modify frames outside a selected contact window.
- Do not classify a whole segment as flight because it contains one unsupported frame.
- Do not hardcode the diagonal angle or staircase lane.
- Reject a candidate before it can replace the source route when global naturalness or contact validation fails.

---

### Task 1: Local Contact-Window Contract

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motionbricks_task_actor.py`
- Test: `tests/python/test_sonic_torch_motionbricks_task_actor.py`

**Interfaces:**
- Consumes: source support mask `(F, 2)` and four-frame context size.
- Produces: `unsupported_contact_windows(...) -> tuple[ContactRepairWindow, ...]`.

- [ ] Add a failing test where two unsupported frames inside a long supported route produce one window bracketed by the immediately adjacent supported frames.
- [ ] Run the focused test and confirm RED because the API is absent.
- [ ] Implement immutable validated window records and run detection without changing the source support mask.
- [ ] Add and pass cases for multiple runs, edge rejection, and overlapping context rejection.

### Task 2: Source-Preserving Window Assembly

**Files:**
- Modify: `resources/run_g1_motionbricks_stutter_repair.py`
- Test: `tests/python/test_run_g1_motionbricks_stutter_repair.py`

**Interfaces:**
- Consumes: a route, `ContactRepairWindow`, generated native qpos, and generated support.
- Produces: `splice_contact_window(...) -> dict[str, np.ndarray]` plus source provenance.

- [ ] Add a failing test proving every prefix/suffix frame is byte-identical after a shorter generated window is inserted.
- [ ] Run the focused test and confirm RED because provenance and explicit support are absent.
- [ ] Reuse endpoint-exact 30-to-50 Hz resampling and implement the minimal splice/provenance output.
- [ ] Run both focused modules and all existing MotionBricks splice regressions.

### Task 3: Generate and Globally Gate the Diagonal

**Files:**
- Modify: `resources/run_g1_motionbricks_terrain_task_actor.py`
- Modify: `tests/python/test_run_g1_motionbricks_terrain_task_actor.py`
- Create: `build/g1-diagonal-contact-window-v1/` artifacts.

**Interfaces:**
- Consumes: `build/g1-diagonal-cadence-v1/traversal.npz`, detected repair windows, terrain dataset/config, G1 XML, and the pinned MotionBricks checkpoint.
- Produces: candidate records, source-preserving traversal, provenance, cadence report, and independent validation report.

- [ ] Add a failing selection test that rejects a duration-matched candidate with repeated same-foot touchdowns or a 0.58-radian joint jump.
- [ ] Run the focused test and confirm the old selector accepts the bad candidate.
- [ ] Add a contact-window runner mode that loads MotionBricks once, sweeps supported token counts per window, and accepts only whole-route-valid candidates.
- [ ] Generate the negative-45-degree center route and retain the source route if any window has no passing candidate.
- [ ] Run the independent validator, render a dense event sheet, close the old viewer, and launch only the accepted route at 50 Hz.
