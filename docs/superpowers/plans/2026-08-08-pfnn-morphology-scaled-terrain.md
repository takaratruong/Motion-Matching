# PFNN Morphology-Scaled Terrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare and display the approved G1 retarget on PFNN fitted terrain uniformly scaled by exactly 0.875.

**Architecture:** Add validated uniform scale and constant vertical-placement parameters to the continuous terrain query, comparison report, mesh builder, and viewer. Native scale 1.0 and zero offset remain the defaults and bit-identical.

**Tech Stack:** Python 3.10, NumPy, MuJoCo, `unittest`.

## Global Constraints

- Approval scale is exactly 0.875 and applies uniformly in XYZ about world origin.
- Approval vertical offset is exactly 0.05224985936713168 m, derived once from the scaled comparison's 80 authored stance probes.
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
- `terrain_height_g1(fit, query_xy_m, *, scale=1.0, z_offset=0.0)`
- `compare_motion_to_terrain(motion, fit, model, *, terrain_scale=1.0, terrain_z_offset=0.0)`
- `terrain_mesh(fit, x_samples, y_samples, *, scale=1.0, z_offset=0.0)`
- CLI `--terrain-scale` and `--terrain-z-offset`, defaults 1.0 and 0.0.

- [ ] Write a RED test proving `h_scaled(s * xy) == s * h_native(xy)` and scale 1 is unchanged.
- [ ] Implement finite positive uniform scaling in the continuous query.
- [ ] Thread the same scalar through comparison, mesh generation, live markers, JSON output, and viewer status.
- [ ] Prove and thread the single constant vertical offset without changing XY or surface shape.
- [ ] Run the focused suite, compile checks, and `git diff --check`.
- [ ] Rerun the existing comparison once with `--terrain-scale 0.875 --terrain-z-offset 0.05224985936713168`.
- [ ] Reopen the interactive viewer with the same scale and offset and preserve the result.
- [ ] Commit the scoped implementation after verification.
