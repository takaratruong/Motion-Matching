# Reach Vertical Endpoint Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude reach candidates with more than 0.10 m recorded-to-requested endpoint height mismatch before IK and visualization.

**Architecture:** Store the limit in `CoverageConfig`, centralize the compatibility predicate in reach coverage, use it while selecting/enumerating search candidates, and enforce it again in direct shaping.

**Tech Stack:** C++17, existing reach coverage/search pipeline, assert-based C++ tests.

## Global Constraints

- Default maximum vertical endpoint delta is exactly `0.10F`.
- Exactly 0.10 m is accepted; greater values are excluded.
- Filtering occurs before search workers.
- Direct shaping rejects bypasses as `OutsideEnvelope`.

---

### Task 1: Vertical Compatibility Contract

**Files:**
- Modify: `reach_coverage.h`
- Modify: `reach_coverage.cpp`
- Modify: `tests/cpp/test_reach_coverage.cpp`

- [ ] Add failing tests for inclusive 0.10 m compatibility, rejection above 0.10 m, and invalid config values.
- [ ] Add `maximum_endpoint_vertical_delta_m` and the shared compatibility predicate.
- [ ] Enforce it before pose generation in `shape_candidate`.
- [ ] Build and run `build/tests/cpp/test_reach_coverage`.

### Task 2: Exhaustive Search Prefilter

**Files:**
- Modify: `reach_search.cpp`
- Modify: `tests/cpp/test_reach_search.cpp`

- [ ] Add a failing test with one vertically compatible clip and one incompatible clip; assert search total contains only the compatible clip's yaw placements.
- [ ] Filter enumerated candidates before allocating worker slots.
- [ ] Verify candidate selection applies the same predicate.
- [ ] Build and run `build/tests/cpp/test_reach_search`.

### Task 3: Runtime Verification

- [ ] Run all reach C++ and Python viewer tests.
- [ ] Rebuild the viewer.
- [ ] Launch combined `reach-pack-v6`, search at multiple object heights, and confirm displayed candidates stay within 0.10 m source endpoint height.
