# Downhill Stair Registration Analysis (G1 Terrain Motion Matching)

**Date:** 2026-07-30 · **HEAD:** 5f031213
**Scope:** Bounded research/implementation-design audit. Changes no project
behavior; recommends test-first seams and one focused experiment.
**Discipline:** each claim is tagged **[Observed]** (cited artifact) or
**[Inference]** (reasoning over observed facts).

---

## 1. Summary

Admitting `down-continuous-33` and using its transform requires fixing **two
distinct transform defects** and running **one focused experiment**.

- **Defect A — registration never records a non-identity transform.**
  `build_source_terrain` hardcodes `motion_to_terrain_xy_yaw=(0.0, 0.0, 0.0)`
  (`resources/g1_torch_terrain_builder/terrain.py:353`). Under identity the clip
  has **115 contact samples, 0 elevated contacts**, and is rejected
  `insufficient_elevated_contact_samples`
  (`admission.py:279-292`). With the inferred transform `(-0.29, 3.99, -pi/2)`
  it has **269 contacts, 154 elevated, contact-error p95 0.0315826738 m**, and
  passes every *unchanged* threshold.
- **Defect B — the published transform is ignored downstream.** The publisher
  writes the field into the manifest terrain descriptor
  (`publish.py:101-104`), but `TerrainDataset.load` never reads it
  (`sonic/python/mm_sonic/torch_terrain_features.py:292-368`) and both
  `database_rows` (`:582-634`) and query-scene alignment
  (`TerrainSceneAlignment` / `for_condition`, `:68-114,524-580`) assume the
  motion and terrain frames coincide.
- **Focused outcome.** The retained matcher passes 13/21 hard outcomes; the
  target is **riser-reversal backward progress** (`_RISER_REVERSAL`,
  `torch_terrain_omni_routes.py:222-230`), whose prior reversal-history
  exemption was falsified. One single-variable experiment gates whether the
  admitted transformed clip helps it.

No protected threshold is weakened by any recommendation below.

---

## 2. Component walk-through (observed)

### 2.1 Source registration — `registry.py`
**[Observed]** `down-continuous-33` is registered as a `stair-local`
`native-npz` clip on the `fixed-staircase` adapter (`registry.py:198`).
`SourceSpec` carries no transform field; placement is produced later by the
terrain adapter.

### 2.2 Terrain adapter — `terrain.py`
**[Observed]** `TerrainEvidence` owns `motion_to_terrain_xy_yaw` and validates it
as a finite 3-tuple (`terrain.py:45,52-58`). `_fixed_staircase` places three
boxes at fixed staircase coordinates (`terrain.py:111-119`); `_boxes_grid`
expands the grid to also cover the clip's own root XY (`terrain.py:88-95`).
`build_source_terrain` **always** returns identity (`terrain.py:353`).
**[Inference]** For a downhill clip whose motion frame sits far from the fixed
box coordinates, identity leaves the feet off the boxes → 0 elevated contacts.
The transform is the rigid map from the clip's motion frame onto the box frame.

### 2.3 Admission — `admission.py`
**[Observed]** Admission **already consumes** the transform: `_terrain_xy`
applies rotation+translation (`admission.py:147-161`) and `admit_candidate`
maps foot XY into the scene before sampling the grid
(`admission.py:232-236`). Gates in order: contact-sample floor
`_MINIMUM_CONTACT_SAMPLES=50` (`:257`), contact p95 `_CONTACT_P95_ERROR_M=0.035`
(`:268`), elevated floor `_MINIMUM_ELEVATED_CONTACT_SAMPLES=25` on elevated
terrain (`:279-282`).
**[Inference]** Because admission is transform-aware, the frozen 154-elevated /
p95 0.0316 result reproduces **without any admission edit** once registration
supplies the non-identity transform. Defect A is upstream of admission.

### 2.4 Publisher / manifest — `publish.py`
**[Observed]** `_terrain_descriptor` emits `motion_to_terrain_xy_yaw` into every
heightgrid clip descriptor (`publish.py:101-104`). `_validate_candidate` loads
`TerrainDataset` and rechecks motion/terrain hashes (`publish.py:135-151`) but
does not assert the transform round-trips into any consumer.

### 2.5 Database features & query alignment — `torch_terrain_features.py`
**[Observed]** `TerrainDataset.load` validates schema, hashes, and grid
metadata (origin/shape/cell) but reads **no** `motion_to_terrain_xy_yaw`
(`:292-368`). `_TorchHeightGrid` and `clip_grids` carry no transform.
`database_rows` samples `clip.body_position_world` directly against the grid
(`:610-634`). `TerrainSceneAlignment` is built purely from the query clip's
own root pose (`:557-572`); `matcher_to_scene_xy` composes only that rigid map
(`:98-114`). `TerrainFootClearanceValidator` reuses the same alignment
(`:748-751`).
**[Inference]** The offline grid is materialized in the clip's motion frame,
while admission's elevated-contact proof is computed in the box frame via the
transform. A clip admitted only because of a non-identity transform would then
have its database rows and live queries evaluated in an unaligned frame —
train/serve skew. The transform must be threaded into both `database_rows` and
`matcher_to_scene_xy`.

### 2.6 Focused route — `torch_terrain_omni_routes.py`
**[Observed]** `_ROUTES` enumerates exactly 21 routes (`:251-273`);
`_RISER_REVERSAL` ascends then reverses down with `required_segments=
("reverse-down",)` (`:222-230`).

---

## 3. Challenging the proposed seam

**[Inference]** A single-site fix ("apply the transform in `database_rows`
only") is **insufficient and wrong**. The transform must be consumed at three
sites, and it originates at a fourth:

1. **Registration/adapter (producer):** `build_source_terrain` must emit the
   registered non-identity transform for `down-continuous-33` instead of the
   hardcoded identity (`terrain.py:353`).
2. **Offline database rows:** `database_rows` must map motion XY → terrain frame
   before `grid.sample_xy` (`torch_terrain_features.py:610-634`).
3. **Online query alignment:** the same transform must compose into
   `TerrainSceneAlignment` so `matcher_to_scene_xy` and the clearance validator
   agree with the offline frame (`:98-114,557-572,748-751`).
4. **Manifest validation:** `TerrainDataset.load` must read + range/finiteness-
   check the field and fail closed on a malformed value *before* building any
   grid or alignment.

Fixing only site 2 leaves online skew; fixing only site 3 leaves the database
misaligned. The minimal correct seam is **one shared transform, defaulted to
identity, carried on the clip grid and consumed by both frames**, so every
existing identity clip is byte-for-byte unchanged.

---

## 4. Test-first seams (recommended; not implemented here)

Mirror the existing `unittest` style in `tests/python/test_torch_terrain_*`.

1. **Identity-compatibility regression** — a clip whose manifest transform is
   `[0,0,0]` yields byte-identical `database_rows` and `matcher_to_scene_xy`
   output to today. This is the RED→GREEN anchor: it must pass before and after
   the change. Reuse the shape-(2,)/scalar tensors already asserted in
   `TerrainSceneAlignment.__post_init__`.
2. **Transform-consumption end-to-end** — inject a **distinct non-default
   sentinel** transform (e.g. `(-0.29, 3.99, -pi/2)`) at the manifest source,
   use a call-counting spy to assert the transform accessor is invoked exactly
   once per frame, and assert the **exact** transformed sample coordinates at
   both the offline `database_rows` sink and the online `query_row` sink.
   Default-equal / identity fixtures do not prove continuity.
3. **Validation-ordering** — a malformed / non-finite transform in the manifest
   raises `ContractError` in `TerrainDataset.load` **before** any grid load or
   alignment construction; assert no downstream grid was built.
4. **Admission reproduction (real entrypoint)** — drive `admit_candidate` with a
   `down-continuous-33`-shaped fixture and the inferred transform through the
   real gate ordering, asserting `elevated_contact_sample_count == 154` and
   `contact_height_error_m["p95"] <= 0.0315826738` with `accepted is True`;
   and with identity, assert `reason == "insufficient_elevated_contact_samples"`
   and `elevated_contact_sample_count == 0`. This exercises the rejection branch
   at the real entrypoint, not a helper.

---

## 5. Minimum source / manifest / runtime changes

- **Registry/adapter:** register the inferred `(-0.29, 3.99, -pi/2)` for
  `down-continuous-33` and have `build_source_terrain` emit it (replace the
  single hardcoded identity at `terrain.py:353` with the registered value;
  identity remains the default for all other clips).
- **Manifest:** no schema shape change — the field is already published
  (`publish.py:101-104`). Add a read + finiteness/length check in
  `TerrainDataset.load`; preserve the legacy `g1-torch-stair-slice/v1` five-clip
  path unchanged (`torch_terrain_features.py:254-257`).
- **Runtime:** thread one transform onto `_TorchHeightGrid`/`clip_grids` (or a
  parallel per-clip tuple) and compose it in `database_rows` and
  `TerrainSceneAlignment`; default identity ⇒ no behavior change for existing
  clips. No new abstraction beyond this one accessor.

Reuse note: `_terrain_xy` (`admission.py:147-161`) and `_rotate_xy`
(`torch_terrain_features.py:57-65`) already implement the exact rigid map; the
runtime change reuses `_rotate_xy` + translation rather than adding new math.

---

## 6. Focused experiment (single variable)

**Variable:** admission + downstream consumption of `down-continuous-33` with
transform `(-0.29, 3.99, -pi/2)` (treatment) vs. clip absent / identity
(baseline). Only authenticated motion coverage and its required transform
consumption change.

**Corpus leg (offline, project test):**
- **Accept** iff the clip is admitted with `elevated_contact_sample_count == 154`
  and `contact_height_error_m["p95"] <= 0.0315826738 m`, with **no** admission,
  contact, or terrain-quality threshold relaxed, and every previously accepted
  clip still accepted.
- **Reject** if admission would require weakening any threshold, if the elevated
  count falls below the strict `>=25` gate, or if the identity/legacy regression
  (§4.1) changes any existing clip's rows.

**Focused-route leg (matcher, focused GPU rollout of `riser-reversal`):**
- **Accept** iff riser-reversal backward progress flips to **pass** *and* the
  other 12 currently-passing hard outcomes remain passing (≥13/21, zero
  regression).
- **Reject** if any previously-passing outcome regresses, or if riser-reversal
  backward progress does not improve.

**Residual risks:** the inferred source transform stays provisional until the
project test and the focused GPU rollout pass; full 21-route matrix behavior
remains unqualified until the focused route accepts the added coverage.

---

## 7. Scope guarantees (negative proofs)
- No implementation, tests, configs, existing docs, Git metadata, or protected
  thresholds are modified by this report.
- No contact-admission, outcome, rescue, or terrain-quality threshold is
  weakened; every acceptance criterion tightens or preserves existing gates.
- No network, remote, credential, GPU device, ignored benchmark artifact, or
  other worktree was accessed; evidence is from repository source only.

---

## 8. Post-audit foreman adjudication

The implementation seam in this audit was accepted, but its proposed
`riser-reversal` experiment was rejected after controller-side data inspection.
`down-continuous-33` contributes 133 forward-local frames and **zero**
backward-local frames, so it is a valid downhill-kinematics addition but not a
valid single-variable test of backward stair locomotion.

The controller then evaluated the independently registered
`staircase-final-v3` clip, which contributes 220 backward-local frames, 181 of
them with elevated foot support. With frozen baseline normalization, the
three-route A/B changed diagonal-down-left from pass to fail, changed
diagonal-down-right from fail to pass, and worsened riser-reversal progress
from `-0.0117` to `-0.2172`. The new clip was selected only once during the
reversal route and never during its reverse-down segment. Therefore the
coverage candidate is rejected as a retained matcher improvement.

The next hypothesis is selection compatibility: determine why authenticated
backward/elevated rows do not enter the reverse-down top candidates before
adding more corpus clips or changing transition penalties. The next experiment
must inspect decomposed search costs at the first reverse-down search and reject
the hypothesis if the registered rows are already competitive but are excluded
later by transition or clearance gating.
