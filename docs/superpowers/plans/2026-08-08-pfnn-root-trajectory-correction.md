# PFNN Root-Trajectory Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore PFNN's unscaled root trajectory and apply one deterministic stance-median vertical anchor without altering the approved G1 joint motion.

**Architecture:** The retargeter captures the displayed source `Hips` positions before GMR scaling and replaces only output root translation. The comparison module computes one scalar anchor from exact stance gaps and writes a second validated motion artifact. Existing terrain, gates, and viewer remain authoritative.

**Tech Stack:** Python 3.10, NumPy, pinned GMR, MuJoCo, `unittest`.

## Global Constraints

- Source frames 8160 through 8279, fitted patch 8967, and 120 Hz timeline remain unchanged.
- G1 joints and root orientations must be bit-identical before and after correction.
- Terrain scale, shape, placement, thresholds, and contact annotations remain unchanged.
- Exactly one vertical scalar is allowed; no per-frame IK or foot locking.

---

### Task 1: Restore source root translation

**Files:**
- Modify: `sonic/python/mm_sonic/retarget_pfnn_bvh_g1.py`
- Modify: `tests/python/test_retarget_pfnn_bvh_g1.py`

**Interfaces:**
- Produces: `restore_source_root_translation(qpos, source_root) -> np.ndarray` and CLI `--root-translation gmr|source`.

- [ ] **Step 1: Write RED tests** proving source XY is copied, Z uses source displacement anchored at GMR frame zero, input arrays remain unchanged, and non-root qpos is bit-identical.
- [ ] **Step 2: Run the targeted test** and require missing-function RED.
- [ ] **Step 3: Implement the pure function and CLI binding** using the already-scaled displayed `Hips` positions.
- [ ] **Step 4: Run both focused suites, `py_compile`, and `git diff --check`.**

### Task 2: Apply one stance-median vertical anchor

**Files:**
- Modify: `sonic/python/mm_sonic/compare_pfnn_terrain_g1.py`
- Modify: `tests/python/test_pfnn_terrain_transfer.py`

**Interfaces:**
- Produces: `stance_median_anchor(report, contacts) -> float` and `apply_root_z_anchor(motion, anchor) -> dict[str, object]`.

- [ ] **Step 1: Write RED tests** proving only authored stance gaps contribute, the returned value is the negative median, only root Z changes, and joints/quaternions are bit-identical.
- [ ] **Step 2: Implement both pure functions** and a CLI option to save the anchored motion with a provenance receipt.
- [ ] **Step 3: Generate one source-root motion**, compare with anchor zero, compute one anchor, save one anchored motion, and rerun the exact gate once.
- [ ] **Step 4: If accepted, launch the exact terrain viewer for user approval. If rejected, preserve evidence and stop without IK.**
- [ ] **Step 5: Run 19+ focused tests, compile/diff checks, and commit the scoped correction.**
