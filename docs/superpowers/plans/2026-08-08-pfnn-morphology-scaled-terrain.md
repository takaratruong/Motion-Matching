# PFNN Morphology-Scaled Terrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare and display the approved G1 retarget on PFNN fitted terrain uniformly scaled by exactly 0.875.

**Architecture:** Add one validated scale parameter to the continuous terrain query, comparison report, mesh builder, and viewer. Native scale 1.0 remains the default and bit-identical.

**Tech Stack:** Python 3.10, NumPy, MuJoCo, `unittest`.

## Global Constraints

- Approval scale is exactly 0.875 and applies uniformly in XYZ about world origin.
- Do not modify motion, contacts, fit parameters, thresholds, or GRAIL assets.
- Preserve the native scale-1 comparison and artifact.

---

### Task 1: Implement and verify uniform terrain scaling

**Files:**
- Modify: `sonic/python/mm_sonic/pfnn_terrain_fit.py`
- Modify: `sonic/python/mm_sonic/compare_pfnn_terrain_g1.py`
- Modify: `sonic/python/mm_sonic/view_g1_retarget.py`
- Modify: `tests/python/test_pfnn_terrain_transfer.py`

**Interfaces:**
- `terrain_height_g1(fit, query_xy_m, *, scale=1.0)`
- `compare_motion_to_terrain(motion, fit, model, *, terrain_scale=1.0)`
- `terrain_mesh(fit, x_samples, y_samples, *, scale=1.0)`
- CLI `--terrain-scale`, default 1.0, approval value exactly 0.875.

- [ ] Write a RED test proving `h_scaled(s * xy) == s * h_native(xy)` and scale 1 is unchanged.
- [ ] Implement finite positive uniform scaling in the continuous query.
- [ ] Thread the same scalar through comparison, mesh generation, live markers, JSON output, and viewer status.
- [ ] Run the focused suite, compile checks, and `git diff --check`.
- [ ] Rerun the existing comparison once with `--terrain-scale 0.875`.
- [ ] Reopen the interactive viewer with `--terrain-scale 0.875` and preserve the result.
- [ ] Commit the scoped implementation after verification.
