# G1 Continuity-Aware Horizon Lookahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prefer terrain-skill horizons with one compatible same-skill suffix so held commands do not repeatedly settle and globally re-search at double-support boundaries.

**Architecture:** Extend ranked horizon selection with an optional preferred validator while retaining the best immediate-valid candidate as fallback. The horizon matcher supplies a runway validator that checks a later double-support endpoint, suffix progress/stall, and the combined relative terrain profile.

**Tech Stack:** Python 3, PyTorch, NumPy, `unittest`, existing Torch terrain-skill matcher and deterministic route harness.

## Global Constraints

- Apply only when `--continuous-skill` is enabled and requested planar speed exceeds 0.05 m/s.
- Preserve normal ranked order inside both the preferred and fallback sets.
- Fall back to the best immediate-valid candidate when no preferred candidate exists.
- Preserve current behavior exactly when the optional preferred validator is absent.
- Require a later endpoint, at least 5 cm suffix progress, at most five suffix stall frames, and at most 8 cm combined relative surface-profile error.
- Do not enable strict raw-foot feasibility or endpoint warp.
- Preserve release, command-change, reset, foot-lock, and unrelated dirty landing/FK changes.

---

### Task 1: Preferred ranked selection with deterministic fallback

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_horizon_search.py`

**Interfaces:**
- Consumes: existing ranked candidates and immediate `terrain_validator`.
- Produces: optional `preferred_validator: Callable[[int, int, int], bool]` and `TerrainSkillHorizonResult.selection_mode` equal to `preferred`, `immediate`, or `immediate-fallback`.

- [ ] **Step 1: Write failing preferred/fallback tests**

Add tests where record 2 is cheapest and immediate-valid but not preferred, while record 3 is immediate-valid and preferred. Assert record 3 and `selection_mode == "preferred"`. Add a second test with no preferred record and assert record 2 with `selection_mode == "immediate-fallback"`.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_skill_horizon_search.HorizonRankingTest -v
```

Expected: `select_horizon_candidate` rejects the unknown `preferred_validator` argument.

- [ ] **Step 3: Implement one-pass preferred selection**

Validate immediate terrain in ranked order. Retain the first immediate-valid result. Return the first immediate-valid result whose preferred validator passes. If none passes, return the retained fallback. Count preferred rejections separately and preserve the existing failure when no immediate candidate passes.

- [ ] **Step 4: Run GREEN and neighboring search tests**

Run the Task 1 command. Expected: all search tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py \
  tests/python/test_sonic_torch_terrain_skill_horizon_search.py
git commit -m "feat: prefer continuable terrain horizons"
```

---

### Task 2: Same-skill runway validator and measured A/B

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py`
- Modify: `resources/run_g1_torch_multi_horizon_skills.py`
- Modify: `tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py`
- Modify: `tests/python/test_run_g1_torch_multi_horizon_skills.py`

**Interfaces:**
- Consumes: Task 1 `preferred_validator` and existing horizon/skill inventories.
- Produces: continuity-preferred global selection, `HorizonChunkEvent.selection_mode`, and deterministic artifact fields.

- [ ] **Step 1: Write failing matcher tests**

Build a two-record fixture whose cheaper record has no later stable endpoint and whose second record has one. With continuous mode and a moving command, assert selection chooses the second record. With both records lacking runway, assert the cheaper record is selected as immediate fallback. With continuous mode disabled, assert the cheaper record remains selected.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_skill_horizon_rollout -v
```

Expected: the cheaper dead-end record is still selected.

- [ ] **Step 3: Implement runway validation**

For each candidate, find `next_sequential_horizon_endpoint`. Reject missing endpoints. Measure source-root suffix displacement and `remaining_stall_profile`. Then call `terrain_skill_compatible` from the candidate entry through the later endpoint with `tolerance_m=0.08`. Supply this callable as Task 1's preferred validator only for continuous moving commands.

- [ ] **Step 4: Serialize selection mode**

Add `selection_mode` to `HorizonChunkEvent` and canonical chunk JSON. Exclude timing as before. Add parser/artifact tests proving default-off hashes retain deterministic ordering.

- [ ] **Step 5: Run focused regression**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_terrain_skill_horizon_search \
  tests.python.test_sonic_torch_terrain_skill_horizon_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer \
  tests.python.test_run_g1_torch_multi_horizon_skills -v
```

Expected: zero failures.

- [ ] **Step 6: Run deterministic A/B**

Run `riser-stop-restart` once without and once with `--continuous-skill` using the retained foot-lock command. Then run held-forward ascent. Require no lost route pass and either at least two same-skill continuations or at least 40 percent fewer global selections on held-forward motion.

- [ ] **Step 7: Launch only if the gate passes**

Close the current viewer and launch the exact passing configuration with `--continuous-skill`. If the gate fails, keep the current viewer closed and report the measured rejection rather than requesting operator feedback on another ineffective build.

- [ ] **Step 8: Verify, commit, and push**

Run the focused suite, `git diff --check`, stage only the four owned files, commit `feat: select terrain skills with coherent runway`, and push `checkpoint research/g1-torch-terrain-kinematics`.
