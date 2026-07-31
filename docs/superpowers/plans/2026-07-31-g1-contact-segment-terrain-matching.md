# G1 Contact-Segment Terrain Motion Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and visually qualify a kinematic terrain matcher that preserves one authenticated GRAIL stair motion's support-to-support contact segments instead of replacing source frames every 20 ms.

**Architecture:** Add a small immutable contact-segment index and an optional terrain segment policy beside the existing Torch matcher. The policy limits terrain transitions to authenticated contact onsets, applies one support-surface vertical offset to the whole selected segment, and commits sequentially until the next opposite-foot onset. A MuJoCo kinematics helper validates the same emitted root/joint state shown by the viewer; the existing matcher remains bitwise unchanged when no policy is supplied.

**Tech Stack:** Python 3.10, PyTorch 2.13, NumPy, MuJoCo 3.11, `unittest`, native 50 Hz Takara/GRAIL motion data.

## Global Constraints

- This project is kinematics-only: no tracking, physics, depth estimation, inverse kinematics, foot locking, learned model, or external controller integration.
- Use only `flat/motion.npz` and `terrain/grail/grail-stair_p1-d01be55953dba78b90e9/motion.npz` for the first qualification.
- Output remains native Z-up, wxyz, 29 target-ordered joints, and exactly 50 Hz.
- Terrain contact is `0.035 ± 0.020 m` ankle-origin clearance with absolute vertical ankle speed at most `0.12 m/s`.
- Valid source segments contain 5–60 frames; the selected real clip's ordinary intervals are 33–54 frames.
- A committed segment may not make a non-sequential source-frame change.
- No automated metric can override failed visual inspection.
- Preserve existing `build/` artifacts and the untracked `sonic/configs/experiments/torch_grail_representative.json`.

---

## File structure

- `resources/g1_torch_terrain_builder/bulk_grail.py`: exact logical-name selection from the pinned GRAIL inventory.
- `resources/build_g1_torch_grail_corpus.py`: CLI option for publishing a flat-plus-named-source experiment corpus.
- `sonic/python/mm_sonic/torch_contact_segments.py`: contact derivation, segment indexing, support-foot placement, and terrain policy contracts.
- `sonic/python/mm_sonic/torch_g1_fk.py`: one focused MuJoCo FK adapter for batches of emitted root/joint states.
- `sonic/python/mm_sonic/torch_motion_matcher.py`: optional policy hook, vertical placement, transactional segment commitment, and diagnostics.
- `sonic/python/mm_sonic/torch_contact_segment_rollout.py`: authoritative rollout arrays, metrics, and acceptance gates.
- `resources/run_g1_torch_contact_segment.py`: reproducible experiment CLI.
- `sonic/python/mm_sonic/torch_terrain_live_viewer.py`: opt-in policy construction for the existing controllable MuJoCo viewer.
- `sonic/configs/experiments/torch_grail_contact_segment.json`: the frozen two-clip experiment settings and gates.
- Focused tests live beside the existing bulk-GRAIL, matcher, terrain, rollout, and viewer tests.

---

### Task 1: Publish the exact two-clip experiment corpus

**Files:**
- Modify: `resources/g1_torch_terrain_builder/bulk_grail.py`
- Modify: `resources/build_g1_torch_grail_corpus.py`
- Modify: `tests/python/test_torch_bulk_grail.py`
- Modify: `tests/python/test_build_torch_grail_corpus_cli.py`

**Interfaces:**
- Produces: `select_named_bulk_grail_candidates(candidates, logical_names) -> tuple[BulkGrailCandidate, ...]`.
- Produces: repeatable CLI flag `--logical-name NAME`, mutually exclusive with `--limit-per-partition`.
- The publisher still adds the existing flat control, so one named source yields exactly two accepted clips.

- [ ] **Step 1: Write failing exact-selection tests**

Add tests that freeze requested order, duplicate rejection, missing-name rejection, and CLI selection:

```python
def test_named_selection_is_exact_and_requested_ordered(self):
    selected = select_named_bulk_grail_candidates(
        self.candidates,
        (self.candidates[2].logical_name, self.candidates[0].logical_name),
    )
    self.assertEqual(
        tuple(item.logical_name for item in selected),
        (self.candidates[2].logical_name, self.candidates[0].logical_name),
    )

def test_named_selection_rejects_duplicate_and_missing_names(self):
    name = self.candidates[0].logical_name
    with self.assertRaisesRegex(ValueError, "duplicate"):
        select_named_bulk_grail_candidates(self.candidates, (name, name))
    with self.assertRaisesRegex(ValueError, "not found"):
        select_named_bulk_grail_candidates(self.candidates, ("missing",))
```

In the CLI test, patch `select_named_bulk_grail_candidates` and assert the published source list contains only the resolved named source.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_torch_bulk_grail \
  tests.python.test_build_torch_grail_corpus_cli -v
```

Expected: import or attribute failure for `select_named_bulk_grail_candidates`.

- [ ] **Step 3: Implement exact selection and CLI validation**

Add this bounded selector in `bulk_grail.py`:

```python
def select_named_bulk_grail_candidates(
    candidates: Sequence[BulkGrailCandidate],
    logical_names: Sequence[str],
) -> tuple[BulkGrailCandidate, ...]:
    requested = tuple(logical_names)
    if not requested:
        raise ValueError("named GRAIL selection must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("named GRAIL selection contains a duplicate")
    by_name = {candidate.logical_name: candidate for candidate in candidates}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"named GRAIL candidate not found: {missing[0]}")
    return tuple(by_name[name] for name in requested)
```

Add `--logical-name` with `action="append"`. Reject its use with `--limit-per-partition`; use exact selection when supplied and retain the current seeded selection otherwise. Record `selection_mode: "named"` and the selected names in corpus metadata.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command.

Expected: all focused bulk/CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_torch_terrain_builder/bulk_grail.py \
  resources/build_g1_torch_grail_corpus.py \
  tests/python/test_torch_bulk_grail.py \
  tests/python/test_build_torch_grail_corpus_cli.py
git commit -m "feat: select exact GRAIL terrain sources"
```

---

### Task 2: Derive immutable support-to-support segments

**Files:**
- Create: `sonic/python/mm_sonic/torch_contact_segments.py`
- Create: `tests/python/test_sonic_torch_contact_segments.py`

**Interfaces:**
- Produces `ContactSegment(clip_index, start_frame, end_frame, entering_foot)` where `end_frame` is exclusive and `entering_foot` is `0` for left or `1` for right.
- Produces `ContactSegmentIndex.from_dataset(dataset, minimum_frames=5, maximum_frames=60)`.
- Produces `support_mask(clip_index) -> torch.Tensor[frames, 2]` and `entry(clip_index, frame_index) -> ContactSegment | None`.
- Produces `terrain_entry_eligibility(database) -> torch.Tensor[rows]` with boolean dtype: flat rows and valid terrain entry rows are true; terrain non-entry rows are false.

- [ ] **Step 1: Write failing synthetic segmentation tests**

Construct explicit support masks and assert opposite-foot onset behavior:

```python
def test_segments_run_from_onset_to_next_opposite_onset(self):
    support = torch.zeros((80, 2), dtype=torch.bool)
    support[10:28, 0] = True
    support[30:49, 1] = True
    support[51:70, 0] = True
    segments = segments_from_support_mask(
        clip_index=3,
        support_mask=support,
        minimum_frames=5,
        maximum_frames=60,
    )
    self.assertEqual(
        segments,
        (
            ContactSegment(3, 10, 30, 0),
            ContactSegment(3, 30, 51, 1),
        ),
    )

def test_segment_bounds_reject_startup_and_anomalous_gaps(self):
    # Onsets at 2R, 4L, 54L, 87R, and 168L produce lengths 2, 83, and 33.
    # Only the 33-frame segment is retained under the 5..60 contract.
    support = torch.zeros((180, 2), dtype=torch.bool)
    support[2:3, 1] = True
    support[4:5, 0] = True
    support[54:86, 0] = True
    support[87:167, 1] = True
    support[168:, 0] = True
    self.assertEqual(
        segments_from_support_mask(2, support, 5, 60),
        (ContactSegment(2, 54, 87, 0),),
    )
```

Add malformed dtype/shape/non-finite threshold tests and a row-eligibility test proving flat rows remain eligible while only terrain entry rows are true.

- [ ] **Step 2: Run the new test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segments -v
```

Expected: `ModuleNotFoundError` for `mm_sonic.torch_contact_segments`.

- [ ] **Step 3: Implement contact derivation and indexing**

Use the existing constants exactly:

```python
ANKLE_ORIGIN_SOLE_M = 0.035
STANCE_CLEARANCE_TOLERANCE_M = 0.020
STANCE_VERTICAL_SPEED_MAX_MPS = 0.12

@dataclass(frozen=True, order=True)
class ContactSegment:
    clip_index: int
    start_frame: int
    end_frame: int
    entering_foot: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

def source_support_mask(dataset: TerrainDataset, clip_index: int) -> torch.Tensor:
    clip = dataset.folder.clips[clip_index]
    feet = torch.tensor(
        clip.body_position_world[:, [18, 19]],
        dtype=torch.float32,
        device=dataset.device,
    )
    vertical_speed = torch.tensor(
        clip.body_linear_velocity_world[:, [18, 19], 2],
        dtype=torch.float32,
        device=dataset.device,
    )
    grid = dataset.clip_grids[clip_index]
    if grid is None:
        surface = torch.zeros_like(feet[..., 2])
    else:
        alignment = dataset.clip_alignments[clip_index]
        surface = grid.sample_xy(alignment.matcher_to_scene_xy(feet[..., :2]))
    clearance = feet[..., 2] - surface
    return (
        (torch.abs(clearance - ANKLE_ORIGIN_SOLE_M)
         <= STANCE_CLEARANCE_TOLERANCE_M)
        & (torch.abs(vertical_speed) <= STANCE_VERTICAL_SPEED_MAX_MPS)
    )
```

`segments_from_support_mask` detects `(~support[:-1] & support[1:])`, finds the next later onset of the opposite foot, and retains only half-open segments within the frozen bounds. `ContactSegmentIndex` owns read-only tuples and a dictionary keyed by `(clip_index, start_frame)`.

- [ ] **Step 4: Add the opt-in real-data oracle**

Guard it with `MM_GRAIL_CONTACT_TEST=1`. Load the exact two-clip artifact and assert the selected stair source contains segment lengths `33, 40, 41, 54, 35, 36, 36, 47, 19, 22` after bound filtering, with no retained length outside 5–60.

- [ ] **Step 5: Run synthetic and guarded real-data tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segments -v

MM_GRAIL_CONTACT_TEST=1 \
MM_GRAIL_CONTACT_DATASET=build/torch-grail-contact-segment \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segments.ContactSegmentRealDataTests -v
```

Expected: synthetic tests pass; the guarded test passes after the Task 8 artifact build.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_contact_segments.py \
  tests/python/test_sonic_torch_contact_segments.py
git commit -m "feat: index terrain contact segments"
```

---

### Task 3: Add authoritative batched MuJoCo foot kinematics

**Files:**
- Create: `sonic/python/mm_sonic/torch_g1_fk.py`
- Create: `tests/python/test_sonic_torch_g1_fk.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`

**Interfaces:**
- Produces `MujocoG1FootKinematics(g1_xml)`.
- Produces `foot_positions(joint_position, root_position, root_orientation_wxyz) -> np.ndarray[frames, 2, 3]`.
- The helper owns one model/data pair and resolves `left_ankle_roll_link` and `right_ankle_roll_link` by name.
- `matcher_result_qpos` delegates its target-to-source joint conversion to the same module so viewer and validator cannot diverge.

- [ ] **Step 1: Write failing FK parity and validation tests**

Use the real G1 XML when `mujoco` is installed. Build a two-frame state from one known Takara clip and compare the helper with direct `mj_forward` calls at `1e-10 m` absolute tolerance. Add shape, quaternion-norm, non-finite, missing-body, and `nq != 36` failures.

```python
actual = fk.foot_positions(joints, roots, quaternions)
self.assertEqual(actual.shape, (2, 2, 3))
np.testing.assert_allclose(actual, direct, rtol=0.0, atol=1e-10)
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_g1_fk -v
```

Expected: import failure for `torch_g1_fk`.

- [ ] **Step 3: Implement one canonical qpos/FK path**

Implement strict conversion and batch evaluation:

```python
def target_state_qpos(joints, root, quaternion) -> np.ndarray:
    target = np.asarray(joints, np.float64)
    source = np.empty(29, np.float64)
    source[np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)] = target
    qpos = np.concatenate((np.asarray(root), np.asarray(quaternion), source))
    if qpos.shape != (36,) or not np.isfinite(qpos).all():
        raise ContractError("G1 emitted state is invalid")
    return qpos

class MujocoG1FootKinematics:
    def foot_positions(self, joint_position, root_position,
                       root_orientation_wxyz):
        output = np.empty((len(joint_position), 2, 3), np.float64)
        for index in range(len(output)):
            self.data.qpos[:] = target_state_qpos(
                joint_position[index], root_position[index],
                root_orientation_wxyz[index],
            )
            self.mujoco.mj_forward(self.model, self.data)
            output[index] = self.data.xpos[self.foot_body_ids]
        return output
```

Replace the duplicated conversion inside `matcher_result_qpos` with `target_state_qpos` while retaining its public API.

- [ ] **Step 4: Run FK and viewer regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_g1_fk \
  tests.python.test_sonic_torch_terrain_live_viewer -v
```

Expected: all tests pass with unchanged viewer qpos.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_g1_fk.py \
  sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_g1_fk.py
git commit -m "feat: share authoritative G1 foot kinematics"
```

---

### Task 4: Apply support-foot vertical placement to terrain entries

**Files:**
- Modify: `sonic/python/mm_sonic/torch_contact_segments.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_contact_segments.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Produces `SegmentPlacement(segment, vertical_offset_m, source_support_mask)`.
- Produces `TerrainContactSegmentPolicy.entry_eligibility(database) -> torch.Tensor` and `resolve_entry(clip_index, frame_index, yaw_offset, translation_xy, current_support_mask) -> SegmentPlacement | None`.
- `TorchMotionMatcher.from_folder` gains keyword argument `contact_segment_policy=None` and remains exactly backward-compatible.
- `_aligned_targets` gains required internal arguments `translation_z` and `horizon`; it applies one rigid Z translation to root and all diagnostic bodies.
- `select_exact_candidate` and `rank_exact_transition_candidates` gain an optional boolean `transition_eligible_rows`; the incumbent successor remains eligible regardless of this transition-only mask.

- [ ] **Step 1: Write failing support-foot placement tests**

Use source and query grids with different base heights. Assert the entering ankle reaches the same sole clearance and both root/feet receive one identical offset:

```python
placement = policy.resolve_entry(
    clip_index=1,
    frame_index=10,
    yaw_offset=torch.tensor(0.0),
    translation_xy=torch.tensor([2.0, 0.0]),
    current_support_mask=torch.tensor([False, False]),
)
self.assertEqual(placement.segment.entering_foot, 0)
self.assertAlmostEqual(placement.vertical_offset_m, 0.40, places=6)
```

Add tests proving terrain non-entry rows are ineligible, flat rows remain eligible, an opposite-only current support rejects instantaneous support switching, and an airborne current state accepts either entering foot.

- [ ] **Step 2: Write a failing matcher no-policy equivalence test**

Construct two matchers with identical inputs, one using the default and one explicitly passing `contact_segment_policy=None`. For 100 commands assert every result tensor and diagnostic is bitwise equal.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segments \
  tests.python.test_sonic_torch_motion_matcher -v
```

Expected: missing `SegmentPlacement`/policy API and unexpected matcher keyword.

- [ ] **Step 4: Implement entry costs and rigid vertical placement**

`resolve_entry` transforms the source entering ankle with the proposed yaw/XY placement, samples source and query surfaces, and returns:

```python
vertical_offset = float(target_surface - source_surface)
return SegmentPlacement(
    segment=segment,
    vertical_offset_m=vertical_offset,
    source_support_mask=self.index.support_mask(clip_index)[
        segment.start_frame:segment.end_frame
    ].clone(),
)
```

Before resolving an entry, call `policy.query_support_mask(state.feature_body_position, state.feature_body_velocity)` to sample the current emitted feet against the query grid with the same clearance/vertical-speed thresholds. Pass that exact two-value mask to `resolve_entry`; a rejected support-side transition proceeds through the same ranked-entry rescue path as any other invalid terrain entry.

In `prepare_step`, pass `policy.entry_eligibility(database)` into exact search. Add a strict helper that requires a shape-`(rows,)` boolean tensor on the database device. Apply it only to transition candidates after the existing local exclusion; never mask the incumbent successor. Ranked rescue receives the same mask. In `_compose_candidate`, resolve a placement only for a transitioned terrain entry, set `translation_z`, and call:

```python
targets = self._aligned_targets(
    clip_index,
    frame_index,
    yaw_offset,
    translation_xy,
    translation_z,
    horizon=max(46, placement.segment.frame_count if placement else 46),
)
```

Slice candidate arrays to the first 46 frames in `_make_result` so the public output contract remains unchanged. Reject a selected terrain row without a valid placement and let ranked rescue examine the next finite entry candidate.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 3 command.

Expected: all existing matcher tests and new policy tests pass; default behavior remains bitwise equal.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_contact_segments.py \
  sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_contact_segments.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: place terrain segments through support feet"
```

---

### Task 5: Commit terrain entries through the complete contact segment

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Produces `SegmentCommitment(clip_index, start_frame, end_frame, entering_foot, vertical_offset_m)`.
- Extends `MotionMatchDiagnostics` with `segment_committed`, `segment_start_frame`, `segment_end_frame`, `segment_entering_foot`, `segment_vertical_offset_m`, and `segment_rejection_reason`.
- Commitment is stored only in `_MatcherState`/`PreparedMotionMatch`; `prepare_step` is read-only and `commit` is the only mutation point.

- [ ] **Step 1: Write failing commitment behavior tests**

Freeze sequential playback and release:

```python
first = matcher.step((0.4, 0.0), 0.0)
self.assertTrue(first.diagnostics.segment_committed)
start = first.diagnostics.selected_frame
end = first.diagnostics.segment_end_frame
frames = [first.diagnostics.selected_frame]
while frames[-1] + 1 < end:
    frames.append(matcher.step((-0.4, 0.0), math.pi).diagnostics.selected_frame)
self.assertEqual(frames, list(range(start, end)))
released = matcher.step((-0.4, 0.0), math.pi)
self.assertFalse(released.diagnostics.segment_committed)
```

Add tests for stale/foreign prepared values, rejected entry leaving commitment/history unchanged, clip-end failure, and no search-selected transition while committed even when the command reverses.

- [ ] **Step 2: Run the commitment tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_motion_matcher.TorchMotionMatcherTests.test_contact_segment_commitment_is_sequential_until_release \
  tests.python.test_sonic_torch_motion_matcher.TorchMotionMatcherTests.test_rejected_contact_entry_is_transactional -v
```

Expected: missing commitment diagnostics/state.

- [ ] **Step 3: Implement transactional commitment**

When `state.commitment` is active and `state.frame_index + 1 < end_frame`, bypass candidate search and select the exact successor with the stored yaw, XY translation, and Z translation. Preserve shaped command evolution for diagnostics but set `searched=False`, `transitioned=False`, and `force_search_reason="segment_commitment"`.

On an accepted terrain entry, store:

```python
commitment = SegmentCommitment(
    clip_index=candidate.clip_index,
    start_frame=placement.segment.start_frame,
    end_frame=placement.segment.end_frame,
    entering_foot=placement.segment.entering_foot,
    vertical_offset_m=placement.vertical_offset_m,
)
```

Release by writing `None` into the prepared next state when its selected frame equals `end_frame - 1`. Never mutate `_state`, `_selection_history`, or commitment during preparation.

- [ ] **Step 4: Run all matcher/contact tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_contact_segments -v
```

Expected: all tests pass; no-policy equivalence remains bitwise.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: commit terrain contact segments transactionally"
```

---

### Task 6: Validate complete emitted segments with authoritative FK

**Files:**
- Modify: `sonic/python/mm_sonic/torch_contact_segments.py`
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_contact_segments.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`

**Interfaces:**
- Produces `SegmentValidation(accepted, reason, foot_position_world, foot_clearance_m, emitted_support_mask, lost_source_support_fraction, longest_unsupported_frames)`.
- Produces `TerrainContactSegmentPolicy.validate_emitted(placement, joints, roots, quaternions) -> SegmentValidation`.
- Candidate validation covers the entire 5–60-frame segment; the public result still exposes the first 46-frame window.
- For a placed terrain segment, matcher feature-body root/feet are replaced by the authoritative emitted root plus MuJoCo-FK ankles before result/state construction; no-policy matching retains the old diagnostic path.

- [ ] **Step 1: Write failing validator tests**

Inject a fake FK adapter returning controlled feet. Cover: valid alternating support, penetration below `-0.03 m`, entering support outside `0.035 ± 0.020 m`, 11 consecutive unsupported frames, source support lost on more than 5%, and out-of-domain terrain queries.

```python
validation = policy.validate_emitted(
    placement,
    joint_position=torch.zeros((40, 29)),
    root_position=torch.zeros((40, 3)),
    root_orientation_wxyz=unit_quaternions(40),
)
self.assertFalse(validation.accepted)
self.assertEqual(validation.reason, "unsupported_run")
self.assertEqual(validation.longest_unsupported_frames, 11)
```

Add a matcher test proving a failed full-segment validation rejects the entry transactionally and ranked rescue cannot accept another invalid entry.
Add a parity test asserting `dense_feature_body_position_window[:, 1:]` equals a fresh FK reconstruction of `dense_joint_position_window`, `dense_root_position_window`, and `dense_root_orientation_window_wxyz` for every committed output frame.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segments \
  tests.python.test_sonic_torch_motion_matcher -v
```

Expected: missing `SegmentValidation` and `validate_emitted`.

- [ ] **Step 3: Implement full-segment FK validation**

Compute feet once with `MujocoG1FootKinematics`, sample the query grid at emitted foot XY, derive clearance and vertical speed at 50 Hz, then compute emitted support with the frozen thresholds. Use this exact decision order:

```python
if float(clearance.min()) < -0.03:
    reason = "penetration"
elif not bool(emitted_support[0, segment.entering_foot]):
    reason = "entering_support"
elif longest_false_run(emitted_support.any(dim=1)) > 10:
    reason = "unsupported_run"
elif lost_source_support_fraction > 0.05:
    reason = "source_support_lost"
else:
    reason = None
```

Return owned immutable NumPy arrays in `SegmentValidation`. Call validation after composing a placed terrain entry and before existing terrain-clearance validation. Every rescue candidate must pass both validators. For an accepted placement, replace the candidate's diagnostic body window with `torch.cat((root[:, None, :], fk_feet), dim=1)` and derive body velocity by finite difference at `0.02 s`, retaining the source first-frame velocity only where no previous emitted FK sample exists. This makes subsequent query features, validators, saved diagnostics, and the viewer consume the same feet.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass with exact rejection reasons and transactional state.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/torch_contact_segments.py \
  sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_contact_segments.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: validate emitted terrain contact segments"
```

---

### Task 7: Add the deterministic rollout, metrics, and viewer wiring

**Files:**
- Create: `sonic/python/mm_sonic/torch_contact_segment_rollout.py`
- Create: `resources/run_g1_torch_contact_segment.py`
- Create: `sonic/configs/experiments/torch_grail_contact_segment.json`
- Create: `tests/python/test_sonic_torch_contact_segment_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`

**Interfaces:**
- Produces `run_contact_segment_rollout(resolved, g1_xml) -> ContactSegmentRollout`.
- Produces `evaluate_contact_segment_acceptance(metrics) -> tuple[AcceptanceGate, ...]`.
- CLI writes `rollout.npz`, `metrics.json`, `events.jsonl`, and `resolved-config.json` transactionally.
- Viewer flag `--contact-segments` constructs the exact same policy/FK path as the rollout.

- [ ] **Step 1: Write failing metric and wiring tests**

Use synthetic FK arrays to freeze these gates:

```python
self.assertEqual(
    [(gate.name, gate.passed) for gate in gates],
    [
        ("unsupported_fraction", True),
        ("longest_unsupported", True),
        ("stance_slide", True),
        ("minimum_clearance", True),
        ("lost_source_support", True),
        ("committed_sequence", True),
        ("upper_landing", True),
    ],
)
```

Add one failing boundary test per threshold: 15% inclusive, 10 frames inclusive, 0.35 m inclusive, `-0.03 m` inclusive, 5% inclusive, any non-sequential committed frame rejected, and landing required. Add a viewer source-contract test that `--contact-segments` supplies the policy to `TorchMotionMatcher.from_folder`.

- [ ] **Step 2: Run rollout/viewer tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segment_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer -v
```

Expected: missing rollout module and viewer flag.

- [ ] **Step 3: Implement authoritative rollout and acceptance**

The rollout records per frame:

```python
rows = {
    "joint_position": [],
    "root_position_world": [],
    "root_orientation_world_wxyz": [],
    "foot_position_world": [],
    "foot_surface_height_m": [],
    "foot_clearance_m": [],
    "source_support_mask": [],
    "emitted_support_mask": [],
    "segment_committed": [],
    "segment_start_frame": [],
    "segment_end_frame": [],
    "segment_entering_foot": [],
    "segment_vertical_offset_m": [],
    "selected_clip_index": [],
    "selected_frame": [],
    "step_time_ns": [],
    "search_time_ns": [],
}
```

Compute stance slide only over consecutive frames where the same foot is supported in both. Compute unsupported fraction, longest unsupported run, source-support loss, sequential-commit violations, penetration, progress, and landing from the saved authoritative FK arrays.

- [ ] **Step 4: Add frozen configuration**

Copy command shaping and continuity values from `torch_grail_representative.json`, but set:

```json
{
  "schema": "g1-torch-contact-segment-experiment/v1",
  "query_scene": "terrain/grail/grail-stair_p1-d01be55953dba78b90e9/motion.npz",
  "reset_clip": "flat/motion.npz",
  "contact_segments": {
    "minimum_frames": 5,
    "maximum_frames": 60,
    "maximum_unsupported_frames": 10,
    "maximum_lost_source_support_fraction": 0.05
  },
  "acceptance": {
    "maximum_unsupported_fraction": 0.15,
    "maximum_longest_unsupported_frames": 10,
    "maximum_stance_slide_m": 0.35,
    "minimum_foot_clearance_m": -0.03,
    "maximum_lost_source_support_fraction": 0.05,
    "require_sequential_commitment": true,
    "require_upper_landing": true
  }
}
```

Retain the complete existing matcher block rather than replacing omitted values with defaults.

- [ ] **Step 5: Run rollout/viewer tests and verify GREEN**

Run the Step 2 command.

Expected: all new acceptance boundary tests and existing viewer tests pass.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/torch_contact_segment_rollout.py \
  resources/run_g1_torch_contact_segment.py \
  sonic/configs/experiments/torch_grail_contact_segment.json \
  sonic/python/mm_sonic/torch_terrain_live_viewer.py \
  tests/python/test_sonic_torch_contact_segment_rollout.py \
  tests/python/test_sonic_torch_terrain_live_viewer.py
git commit -m "feat: evaluate contact-segment terrain matching"
```

---

### Task 8: Build, qualify, and visually inspect the experiment

**Files:**
- Create: `docs/superpowers/results/2026-07-31-g1-contact-segment-terrain-results.md`
- Generated only: `build/torch-grail-contact-segment/`
- Generated only: `build/torch-grail-contact-segment-report.json`
- Generated only: `build/torch-grail-contact-segment-rollout/`

**Interfaces:**
- Input logical name: `grail-stair_p1-d01be55953dba78b90e9`.
- Automated result: seven named gates with observed values and pass/fail.
- Manual result: explicit `PASS` or `FAIL` with the first visible failure; never infer a visual pass from metrics.

- [ ] **Step 1: Run the complete focused regression group**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_torch_bulk_grail \
  tests.python.test_build_torch_grail_corpus_cli \
  tests.python.test_sonic_torch_contact_segments \
  tests.python.test_sonic_torch_g1_fk \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_terrain_features \
  tests.python.test_sonic_torch_contact_segment_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer -v
```

Expected: all tests pass with no unexpected skip except the explicitly gated real-data oracle.

- [ ] **Step 2: Publish the exact two-clip corpus**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/build_g1_torch_grail_corpus.py \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --partition stair_p1 \
  --logical-name grail-stair_p1-d01be55953dba78b90e9 \
  --output build/torch-grail-contact-segment \
  --report build/torch-grail-contact-segment-report.json
```

Expected: exactly two accepted clips (`flat/motion.npz` and the named stair), zero unreported exceptions, and an authenticated manifest.

- [ ] **Step 3: Run the guarded real-data contact test**

```bash
MM_GRAIL_CONTACT_TEST=1 \
MM_GRAIL_CONTACT_DATASET=build/torch-grail-contact-segment \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_sonic_torch_contact_segments.ContactSegmentRealDataTests -v
```

Expected: pass with the frozen 5–60-frame real segment inventory.

- [ ] **Step 4: Run the deterministic CUDA/MuJoCo rollout**

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_torch_contact_segment.py \
  --dataset build/torch-grail-contact-segment \
  --config sonic/configs/experiments/torch_grail_contact_segment.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:0 \
  --output build/torch-grail-contact-segment-rollout
```

Expected: the process exits zero only if all seven automated gates pass. A failed gate is an experiment result, not permission to weaken a threshold.

- [ ] **Step 5: Compare against both controls**

Record the untouched source control and the current representative matcher control beside the new result:

| Metric | Untouched source | Current matcher | Contact segments |
|---|---:|---:|---:|
| Unsupported fraction | 0.072 | 0.742 | measured |
| Longest unsupported run | 14 | 81 | measured |
| Stance slide | 0.313 m / 499 fr | 0.621 m / 450 fr | measured |
| Source-sequence changes | 0 | 62 | measured |

If any automated gate fails, classify the first failure using the spec's five failure reasons and stop before opening the viewer.

- [ ] **Step 6: Launch the controllable viewer only after automated PASS**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m \
  mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-grail-contact-segment \
  --config sonic/configs/experiments/torch_grail_contact_segment.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:0 \
  --contact-segments
```

Expected: visible authenticated stair, working WASD/Backspace controls, and no physics process. Manually inspect tread landings, hovering, missed steps, and continuous sliding.

- [ ] **Step 7: Write the truthful result and commit**

Document exact revisions, artifact identities, test commands, automated metrics, and the user's visual verdict. If visual inspection fails, mark the experiment failed and name the first observed failure.

```bash
git add docs/superpowers/results/2026-07-31-g1-contact-segment-terrain-results.md
git commit -m "docs: report contact-segment terrain results"
```

Do not add generated `build/` artifacts or the previously untracked representative config.
