# G1 Interactive Terrain Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make immediate terrain command changes transactional so a failed mid-step replan continues to a safe support boundary instead of freezing or aborting.

**Architecture:** Preserve the immediate exact replan when it succeeds. When a forced single-support interruption raises `HorizonSearchFailure`, restore the original valid skill state, advance it through the base safe-interrupt path, and retain the new command as pending for retry at double support. Evaluate the shared matcher through offline adversarial routes before launching the viewer.

**Tech Stack:** Python 3.10, PyTorch/CUDA, NumPy, MuJoCo FK, `unittest`.

## Global Constraints

- Work only in the existing `research/g1-torch-terrain-kinematics` worktree.
- Do not modify pre-existing dirty contact-segment, G1 FK, or landing-bridge files.
- Do not relax terrain/contact/sole validation thresholds.
- Preserve the qualified v58 route and its `0.411985839 m` stance-slide ceiling.
- Latency is measured but is not a rejection criterion.
- Write and observe every focused test fail before production changes.

---

### Task 1: Freeze the adversarial baseline

**Files:**
- No source changes.
- Output: `build/interactive-coverage-v58-*`

**Interfaces:**
- Consumes: existing `resources/run_g1_torch_multi_horizon_skills.py`.
- Produces: route matrices, arrays, metrics, and chunk ledgers.

- [ ] Run reversal, stop/restart, turns, exits, mounts, and mixed routes with the exact v58 configuration on separate GPUs.
- [ ] Record completion, failure frame/reason, stalls, slide, and deterministic hashes.
- [ ] Select the smallest deterministic route that fails by exact shortlist exhaustion as the regression case.

### Task 2: Make unsafe immediate replans transactional

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_skill_horizon_search.py`
- Test: `tests/python/test_sonic_torch_terrain_skill_horizon_rollout.py`
- Test: `tests/python/test_sonic_torch_terrain_skill_horizon_search.py`

**Interfaces:**
- Consumes: ranked candidates and structured `HorizonSearchFailure` ledgers.
- Produces: immediate success when feasible and deferred safe-boundary retry when infeasible.

- [ ] Write a failing synthetic test where a single-support command change has no exact-valid replacement.
- [ ] Verify the test fails because `prepare_step` raises `HorizonSearchFailure` instead of advancing the current valid chunk with `replan_pending=True`.
- [ ] Catch only the unsafe forced-attempt failure, restore the exact original state, and delegate to the existing safe-interrupt path.
- [ ] Preserve the existing test proving that a feasible immediate replacement still preempts single support.
- [ ] Verify the focused search/rollout tests pass.
- [ ] Commit the isolated behavior.

### Task 3: Qualify route coverage and contact quality

**Files:**
- Modify only if required: `resources/run_g1_torch_multi_horizon_skills.py`
- Test only if required: `tests/python/test_run_g1_torch_multi_horizon_skills.py`
- Output: `build/interactive-coverage-adaptive-*`

**Interfaces:**
- Consumes: the adaptive matcher and Task 1 route set.
- Produces: directly comparable matrices and metrics.

- [ ] Re-run the exact Task 1 routes with transactional replanning.
- [ ] Reject the change if it does not reduce coverage failures or stalls.
- [ ] Re-run `turn-90-middle-right` and enforce completion, exact contacts, and stance slide no greater than `0.411985839 m`.
- [ ] Run the full focused unit suite and `git diff --check`.

### Task 4: Render and inspect retained sequences

**Files:**
- No source changes unless a renderer defect is independently reproduced.
- Output: `build/interactive-coverage-visual-audit-*`

**Interfaces:**
- Consumes: saved route arrays from Tasks 1 and 3.
- Produces: MP4/contact sheets and an inspection record.

- [ ] Render the baseline failure and adaptive result from identical cameras.
- [ ] Inspect command boundaries, foot placement, penetration, sliding, stalls, and route completion.
- [ ] Retain and commit only a version whose render agrees with its metrics.
