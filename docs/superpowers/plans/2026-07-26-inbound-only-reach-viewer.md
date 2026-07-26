# Inbound-Only Reach Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the coverage viewer animate and draw only the inbound approach through contact.

**Architecture:** Add one clip-local contact-boundary helper in the viewer and use it for both animation modulo and wrist-path truncation. The reach pack remains unchanged.

**Tech Stack:** C++17, raylib, Python `unittest`.

## Global Constraints

- Do not modify generated reach-pack motion data.
- Do not alter search, IK, collision, ranking, camera, or object controls.
- The contact frame is inclusive.

---

### Task 1: Enforce inbound-only visualization

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Test: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: `reach::Database::range_starts`, `contact_frames`, and selected `Candidate::clip`.
- Produces: a contact-inclusive inbound pose count used by animation and path drawing.

- [ ] **Step 1: Write the failing source-contract test**

Require the viewer source to derive a contact-inclusive pose count and use it
for animation sampling and wrist-path construction. Forbid modulo by the full
`animation_poses->size()`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests.test_viewer_plays_only_inbound_approach -v
```

Expected: failure because playback currently uses the complete pose vector.

- [ ] **Step 3: Implement the contact-inclusive boundary**

Add a helper that validates the selected clip and returns
`contact_frame - range_start + 1`. Apply it to selected animation, raw wrist
paths, and retargeted wrist paths.

- [ ] **Step 4: Run tests and build**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer -v
make g1_reach_coverage_viewer
```

Expected: all viewer tests pass and the binary builds.

- [ ] **Step 5: Relaunch and verify**

Launch `g1_reach_coverage_viewer build/g1-reaches/reach-pack-v5`, run one
search, and confirm the selected character loops from approach start through
contact without playing the return.

- [ ] **Step 6: Commit**

```bash
git add g1_reach_coverage_viewer.cpp \
  tests/python/test_g1_reach_coverage_viewer.py \
  docs/superpowers/specs/2026-07-26-inbound-only-reach-viewer-design.md \
  docs/superpowers/plans/2026-07-26-inbound-only-reach-viewer.md
git commit -m "fix: show inbound reach approach only"
```
