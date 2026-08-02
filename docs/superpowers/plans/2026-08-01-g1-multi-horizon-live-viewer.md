# G1 Multi-Horizon Live Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the qualified multi-horizon matcher in the existing controllable MuJoCo kinematic viewer.

**Architecture:** Add an opt-in viewer mode and a testable matcher factory. Reuse the existing render/input loop while adapting diagnostics and structured search-failure handling for horizon chunks.

**Tech Stack:** Python 3.10, PyTorch, MuJoCo viewer, X11 key polling, `unittest`.

## Global Constraints

- Preserve the current vanilla/contact/foothold viewer behavior.
- Multi-horizon mode uses the plain 27-value database and checked-in qualified weights.
- Do not modify the dirty contact-segment, FK, or landing-bridge files.
- Do not launch Sonic or integrate physics.

### Task 1: Viewer mode contract

**Files:**
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`

**Interfaces:**
- Produces: `--multi-horizon`, `_validate_live_mode`, horizon-aware `_diagnostic_overlay`.

- [ ] Write parser, exclusivity, and horizon-overlay tests.
- [ ] Run them and confirm missing flag/helper behavior fails.
- [ ] Implement the minimal parser, validation, and overlay behavior.
- [ ] Run focused viewer tests.

### Task 2: Qualified matcher construction

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py`
- Modify: `resources/run_g1_torch_multi_horizon_skills.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Produces: `horizon_search_config_from_experiment`, `_build_live_matcher`.

- [ ] Write a failing construction contract test.
- [ ] Move strict search-config parsing into the library and retain the CLI wrapper.
- [ ] Build the 27-value base matcher, skill inventory, horizon inventory, and horizon adapter.
- [ ] Verify the focused and neighboring suites.

### Task 3: Launch and qualify

**Files:**
- Modify: `TORCH_TERRAIN_QUICKSTART.md`

**Interfaces:**
- Launches: `python -m mm_sonic.torch_terrain_live_viewer --multi-horizon ...`.

- [ ] Add the exact launch command and controls.
- [ ] Run fresh focused tests and compile checks.
- [ ] Launch on `DISPLAY=:1`, verify the PID remains live, and inspect its log for exceptions.
- [ ] Commit and push the isolated viewer changes.
