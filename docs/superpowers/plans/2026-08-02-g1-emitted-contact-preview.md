# G1 Emitted-Contact Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject terrain motion candidates whose actual inertialized kinematics would float, penetrate terrain, or land across a stair edge.

**Architecture:** Add a pure preview helper that advances an isolated `TerrainSkillState`, batches the emitted states through the existing G1 foot FK, and returns the existing every-frame contact-feasibility result. Layer it after inexpensive terrain checks in global horizon selection and expose one explicit A/B flag in the route runner and live viewer.

**Tech Stack:** Python 3, PyTorch, NumPy, MuJoCo-backed batch FK, `unittest`, existing terrain-skill/contact-feasibility modules.

## Global Constraints

- Preview must never mutate committed matcher, skill, or foot-lock state.
- Reject candidate entry frames with no supporting foot.
- Validate emitted support, landing footprint, and swing samples at every frame.
- Preserve current behavior when emitted preview is disabled.
- Search latency is not a qualification constraint for this kinematics stage.
- Do not launch the interactive viewer until rendered output is inspected directly.

---

### Task 1: Pure emitted-contact preview

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_emitted_contact_preview.py`
- Create: `tests/python/test_sonic_torch_terrain_emitted_contact_preview.py`

**Interfaces:**
- Consumes: `TerrainSkillPose`, `TerrainSkill`, `start_skill`, `advance_skill`, batch `foot_kinematics.foot_positions`, a terrain sampler, and `TerrainContactFeasibilityConfig`.
- Produces: `preview_emitted_contact_trace(...) -> TerrainContactFeasibilityResult`.

- [ ] **Step 1: Write failing tests**

Create a four-frame synthetic skill and batch FK fixture.  Assert that a
supported flat candidate with emitted ankle height equal to
`ANKLE_ORIGIN_SOLE_M` is accepted.  Set the current joint offset so the exact
first emitted pose raises the supported feet by 20 cm and assert rejection with
`reason == "stance-height"`.  Set the entry support mask to false for both feet
and assert `reason == "unsupported-entry"`.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_terrain_emitted_contact_preview -v
```

Expected: import failure because `torch_terrain_emitted_contact_preview` does
not exist.

- [ ] **Step 3: Implement the pure preview**

Implement this public signature:

```python
def preview_emitted_contact_trace(
    *,
    folder: Any,
    skill: TerrainSkill,
    selected_entry_frame: int,
    endpoint_frame_exclusive: int,
    current: TerrainSkillPose,
    halflife_s: float,
    foot_kinematics: Any,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: TerrainContactFeasibilityConfig,
) -> TerrainContactFeasibilityResult:
```

Reject an unsupported entry with a structured result.  Otherwise call
`start_skill`, advance through the endpoint into local lists, stack emitted
joint/root/orientation tensors, call batch FK once, and pass the resulting foot
trace plus the exact emitted source-frame support slice to
`validate_placed_contact_trace(..., align_initial_support=False)`.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command and require all three tests to pass without warnings.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_terrain_emitted_contact_preview.py \
  tests/python/test_sonic_torch_terrain_emitted_contact_preview.py
git commit -m "feat: preview emitted terrain contacts"
```

---

### Task 2: Layer preview into global horizon selection

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py`

**Interfaces:**
- Consumes: Task 1 `preview_emitted_contact_trace`.
- Produces: `emitted_contact_preview_enabled: bool = False` on `TerrainSkillHorizonMatcher` and structured `emitted-*` candidate rejections through the existing terrain validator path.

- [ ] **Step 1: Write the failing integration test**

Use the transactional horizon fixture with a batch FK whose emitted supported
feet are 20 cm above flat terrain.  With preview disabled, assert the existing
candidate is selected.  With preview enabled, provide a second emitted-valid
candidate and assert the invalid cheaper candidate is rejected in favor of the
valid one.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_terrain_skill_horizon_rollout -v
```

Expected: constructor rejection for the unknown preview flag.

- [ ] **Step 3: Implement layered selection**

In `compatible`, retain the existing cheap phase and terrain-profile checks.
When preview is enabled, build a `sample_query(points_xy)` closure, preview the
candidate from `self._current_pose()` through the proposed endpoint using
`self.config.inertialization_halflife_s`, and return only
`validation.accepted`.  Do not touch `self._skill_state`, `result_filter`, or
committed diagnostics during preview.

- [ ] **Step 4: Run GREEN and neighboring tests**

Run Task 1 tests plus horizon search, horizon rollout, and live-viewer tests.
Require zero failures.

- [ ] **Step 5: Commit Task 2**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py \
  tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py
git commit -m "feat: gate horizons on emitted contacts"
```

---

### Task 3: A/B flags, emitted audit, and direct render review

**Files:**
- Modify: `resources/run_g1_torch_multi_horizon_skills.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_run_g1_torch_multi_horizon_skills.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Consumes: Task 2 matcher flag.
- Produces: `--emitted-contact-preview` in both CLIs and deterministic runner identity/output fields.

- [ ] **Step 1: Write failing CLI propagation tests**

Assert both parsers expose `--emitted-contact-preview`, the route runner passes
`emitted_contact_preview_enabled=True`, and the live matcher receives the same
flag only in multi-horizon mode.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_run_g1_torch_multi_horizon_skills \
  tests.python.test_sonic_torch_terrain_live_viewer -v
```

Expected: parser failures for the missing flag.

- [ ] **Step 3: Implement flag propagation**

Add the boolean flag to both parsers and function signatures, require
multi-horizon mode in live validation, pass it into
`TerrainSkillHorizonMatcher`, and include it in deterministic runner identity
and printed summary.

- [ ] **Step 4: Run the four-route emitted-preview matrix**

Run diagonal up/down, turn-90-middle-left, and side-exit-upper-left with the
existing qualified foot-lock flags plus `--emitted-contact-preview`.

- [ ] **Step 5: Audit emitted FK and render output**

For each saved route, derive support from selected clip/frame, then measure
emitted stance error, landing error, five-point 4 cm landing-footprint height
range, and swing clearance.  Require the limits in the design.  Render side
contact sheets and transition-dense strips with MuJoCo EGL, inspect every image
directly, and record any visible failure before changing code again.

- [ ] **Step 6: Verify and commit**

Run the focused suite, `git diff --check`, stage only Task 3 files, and commit:

```bash
git commit -m "feat: qualify emitted terrain contacts"
```

- [ ] **Step 7: Launch only after visual qualification**

If and only if both numeric emitted-contact limits and direct render inspection
pass, launch the interactive viewer with `--emitted-contact-preview`.  Otherwise
keep the viewer closed and use the measured failure to design stance/landing
placement optimization.
