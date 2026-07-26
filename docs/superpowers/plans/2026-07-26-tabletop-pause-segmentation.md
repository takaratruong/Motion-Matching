# Tabletop Pause Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the tabletop reach pack with inbound clips bounded by neutral and contact wrist-speed minima.

**Architecture:** Add an opt-in pause-bounded proposal function while preserving the existing radial segmenter. Thread the mode through review publication and the preparation CLI, auto-confirm structurally valid proposals, then rebuild and validate a new tabletop-only pack.

**Tech Stack:** Python, NumPy, unittest, existing reach review/annotation/pack pipeline.

## Global Constraints

- Do not change existing radial segmentation defaults.
- End every tabletop proposal at the contact pause before withdrawal.
- Reject segments shorter than 10 frames or with less than 0.18 m travel.
- Preserve all existing generated packs.

---

### Task 1: Pause-Bounded Proposal Extraction

**Files:**
- Modify: `resources/g1_reach_builder/segmentation.py`
- Test: `tests/python/test_g1_reach_segmentation.py`

**Interfaces:**
- Produces: `propose_pause_bounded_wrist_trace(trace, source_frames, sequence_id, config) -> list[ReachProposal]`
- Produces: `propose_pause_bounded_reaches(corpus, config) -> list[ReachProposal]`

- [ ] Add synthetic reach/contact/withdraw tests asserting the grab frame is the speed-minimum contact pause and the next proposal begins at the intervening neutral pause.
- [ ] Run the focused tests and observe failure because the APIs do not exist.
- [ ] Implement smoothed speed minima, temporal nonmaximum suppression, neutral/local-minimum starts, and structural rejection.
- [ ] Run `python3 -m unittest tests.python.test_g1_reach_segmentation -v`.

### Task 2: Opt-In Review Publication

**Files:**
- Modify: `resources/g1_reach_builder/review.py`
- Modify: `resources/prepare_g1_reach_review.py`
- Test: `tests/python/test_g1_reach_segmentation.py`
- Test: `tests/python/test_g1_reach_review.py`

**Interfaces:**
- `write_review_corpus(output, corpus, *, pause_bounded=False) -> None`
- CLI flag: `--pause-bounded`

- [ ] Add failing tests proving opt-in publication uses pause proposals while default publication remains unchanged.
- [ ] Implement the keyword and CLI flag.
- [ ] Run both segmentation and review test modules.

### Task 3: Rebuild and Verify the Corrected Data

**Files:**
- Create: `build/g1-reaches/tabletop-review-v3/`
- Create: `build/g1-reaches/tabletop-annotations-v3.json`
- Create: `build/g1-reaches/tabletop-reach-pack-v3/`

- [ ] Rebuild the corrected review from the extracted SOMA CSV directory with `--pause-bounded`.
- [ ] Auto-confirm structurally valid proposals.
- [ ] Build the isolated reach pack.
- [ ] Validate 203 captured and 203 mirrored reaches, duration statistics, source provenance, and absence of post-contact playback.
- [ ] Launch the viewer on `tabletop-reach-pack-v3`, search, and inspect several candidates.
