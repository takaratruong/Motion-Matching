# G1 Terrain PFNN Joint-State Implementation Plan

> **For agentic workers:** Use test-driven development, verification before
> completion, and independent review. Work only in the files assigned to each
> task; the repository is shared with parallel workers.

**Goal:** Replace the rejected 288-input diagnostic checkpoint with the
approved 346-input terrain-aware G1 PFNN (`288 existing inputs + 29 q + 29
qdot`), then promote a checkpoint only after fixed limb, ramp, and stair gates
pass.

**Architecture:** Keep the released PFNN phase banks, recurrence, trajectory
conditioning, terrain samples, direct 268-value G1 output, and viewer. Add only
the missing observable joint state. Existing 288-input artifacts remain
explicit v2 artifacts; all new datasets/checkpoints are fail-closed v3
artifacts. No IK, smoothing, blending, clipping, foot lock, or viewer-side pose
correction is authorized.

**Canonical names:**

- `INPUT_LAYOUT` remains the existing 288-value compatibility layout.
- `CLASSIC_G1_INPUT_LAYOUT_V3` is the 346-value layout and appends
  `joint_position` then `joint_velocity`, each width 29.
- Dataset schema: `g1-pfnn-vertical-dataset/v3`.
- Checkpoint schema: `classic-g1-pfnn/v3`.
- Transfer evaluation schema: `classic-g1-pfnn-transfer-evaluation/v2`.

**Protected files:** Never edit or stage `.superpowers/sdd/task-1-report.md` or
`.superpowers/sdd/task-2-report.md`.

## Task 1: 346-input model, checkpoint, and runtime contract

**Owner scope:**

- `sonic/python/mm_sonic/terrain_pfnn/layout.py`
- `sonic/python/mm_sonic/terrain_pfnn/model.py`
- `sonic/python/mm_sonic/terrain_pfnn/recurrence.py`
- `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- `sonic/python/mm_sonic/train_classic_g1_pfnn.py`
- directly corresponding tests only

- [ ] Add RED tests that assert the v3 field slices are `[288:317]` and
  `[317:346]`, v2 stays 288, and `PhaseFunctionedNetwork` defaults to 288 but
  accepts an explicit 346 input width without changing its 268 output.
- [ ] Add RED checkpoint tests for exact v3 schema/layout/input-size binding,
  round trip, tamper rejection, and v2-as-v3 rejection.
- [ ] Add RED recurrence/runtime tests proving bootstrap uses receipt-bound
  q/qdot, a committed frame appends raw predicted q and
  `(q_next-q_current)*30`, and a held/invalid frame rolls back q and qdot with
  all other recurrent state.
- [ ] Implement the smallest compatible model/checkpoint/runtime changes. Do
  not modify phase interpolation, hidden layers, output decoding, terrain
  reconstruction, command ownership, or authorization gates.
- [ ] Run focused tests, the complete PFNN runtime/model/checkpoint suites,
  Ruff, `py_compile`, and `git diff --check`; commit only owned files.

## Task 2: v3 terrain dataset and exact q/qdot provenance

**Owner scope:**

- `sonic/python/mm_sonic/terrain_pfnn/dataset.py`
- `sonic/python/mm_sonic/terrain_pfnn/provenance.py`
- `sonic/python/mm_sonic/terrain_pfnn/sources.py` only if required
- released/vertical/mixed PFNN dataset builder scripts
- directly corresponding tests only

- [ ] Add RED tests for v3 rows shaped `(346,)`, exact `q(t)` and qdot(t)
  alignment in canonical IsaacLab order, train-only normalization, and schema
  or layout tamper rejection.
- [ ] Add RED migration tests that accept only a unique consecutive
  same-clip/same-lane predecessor, reject boundaries and transitions over
  `0.225 rad`, and record rejection counts by source and joint.
- [ ] Add RED mirror tests using the existing 29-joint permutation/sign vector
  for both q and qdot; applying the mirror twice must reproduce the original
  physical row bit-for-bit.
- [ ] Implement v3 construction without changing the physical meaning of the
  original 288 fields. Use source q/qdot directly where present; otherwise use
  only the exact predecessor adapter above. Do not unwrap, repair, interpolate,
  or synthesize missing joint state.
- [ ] Run focused and full dataset/builder suites, Ruff, `py_compile`, receipt
  integrity checks, and `git diff --check`; commit only owned files.

## Task 3: deterministic 288-vs-346 falsification experiment

**Owner scope:**

- a new bounded experiment/evaluation module and tests
- no edits to Task 1 or Task 2 owned files without coordination

- [ ] Build fixed fit and immediately-following held-out blocks from
  `LocomotionFlat02_000`, `WalkingUpSteps02_000`,
  `WalkingUpSteps09_000`, and `WalkingUpSteps12_000` after the 0.225-rad
  boundary filter.
- [ ] Train three otherwise identical small deterministic treatments: 288
  baseline, raw 346 q/qdot, and diagnostic 375 sin(q)/cos(q)/qdot.
- [ ] Persist row identities, source provenance, seeds, update budget, metrics,
  and artifact hashes. Diagnostic checkpoints are never promotable.
- [ ] Accept raw 346 only if it passes every fixed released-PFNN limb gate on
  fitted and held-out blocks while 288 fails at least one held-out gate. Stop
  if raw fails or periodic alone passes; do not add weighting or smoothing.

Fixed limb gates:

- joint MAE `<= 0.100 rad`
- joint RMSE `<= 0.150 rad`
- frame-max p95 `<= 0.500 rad`
- maximum joint error `<= 1.000 rad`

## Task 4: rebuild, train, and promote terrain artifacts

- [ ] Rebuild the broader released-PFNN corpus under v3; do not reuse the
  rejected v2/v3 run directories. Preserve clip-disjoint train/validation
  identities and immutable receipts.
- [ ] Train a released-only v3 checkpoint and require all fixed limb gates.
  A failed released-only model blocks mixing.
- [ ] Add corrected GRAIL slope rows under the same 346-input contract, build
  train-only normalization once, and train/fine-tune the mixed model without
  allowing GRAIL aggregates to dilute released-source metrics.
- [ ] Run exact-input, deterministic undamped flat/stair, joint-limit,
  transition, finite-output, and closed-loop terrain gates. Preserve every
  failed candidate; never relabel it as accepted.
- [ ] Independently review the final checkpoint receipt, source metrics, model
  contract, and runtime isolation before promotion.

## Task 5: user-facing validation

- [ ] Point the existing direct terrain PFNN viewer at the promoted v3
  checkpoint; make no viewer motion changes.
- [ ] Run bounded ascent, crest, descent, reverse approach, and stairs routes
  with no holds, terrain loss, implausible twisting, or foot dragging.
- [ ] Only after offline and closed-loop gates pass, launch the exact committed
  viewer for a final visual check with the user's normal controls.

## Completion rule

The work is not complete when code compiles or a model trains. It is complete
only when the raw 346 representation passes the A/B falsification, the
released-only and mixed checkpoints pass the fixed source-specific gates, the
closed-loop ramp/stair routes pass, and the same promoted artifact is shown in
the uncorrected direct viewer.
