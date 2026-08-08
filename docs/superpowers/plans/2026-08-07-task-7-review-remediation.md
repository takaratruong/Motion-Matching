# Task 7 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the provisional pipeline checkpoint promotable only after exact fixed-sample reproduction and a safe, nonstationary, manifest-bound known-training-terrain rollout, while correcting Task 7 provenance, terrain, collision, metrics, seed, and resume contracts.

**Architecture:** Checkpoint v4 cryptographically binds fitted row receipts, the deterministic lowest-speed valid flat seed, and a verifier receipt. A provenance-bearing scenario object validates all sealed inputs before opening its callback. The raw runtime remains transactional; the recorder derives traversal from realized root motion and audits deterministic collision surfaces. Pipeline training writes a provisional candidate first and a separate verifier atomically promotes an immutable best only after both receipts pass.

**Tech Stack:** Python 3.11, NumPy, PyTorch, MuJoCo, unittest, reviewed GRAIL USD/PKL decoders.

## Global Constraints

- Do not change root/joint/phase/penetration/stance safety thresholds.
- Do not add IK, qpos correction, foot locks, root projection, MotionBricks, portal, or teacher forcing.
- Use only the sealed remediation manifest and training rows for the provisional experiment.
- Outside exact paired-mesh support returns `None`; no boundary extrapolation or synthetic apron.
- Run exactly one controlled expanded-coverage experiment, preserve earlier artifacts, and stop after its first new failure if it does not pass.
- Pipeline promotion does not require the Task 8 18.9-degree gate, but must report it false/not-applicable and must satisfy every other listed native runtime gate plus realized crossing/return.

---

### Task 1: Lowest-speed seed, fitted row receipts, and resume sealing

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Produces: canonical fitted-row SHA-256 receipts; lowest-speed valid-flat seed selection; seed-to-subset binding; pre-restore subset comparison.

- [ ] Add RED regressions for a slower flat candidate without a long suffix, a seed row absent from the receipt, nonadjacent predecessor/first keys, changed resume subset, and row content tampering.
- [ ] Run the named tests and confirm failures are caused by current suffix preference/missing receipt validation.
- [ ] Implement canonical row hashing over normalized x/y, float32 phase, and sealed row provenance; include row hashes in `fitted_subset` and seed provenance.
- [ ] Define valid seed as a unique accepted train row with a unique same-clip `t-1`, `terrain_class=flat`, finite normalized/physical values, predecessor joints inside canonical limits, and exact recurrent trajectory/body/phase reconstruction. Sort only by `(target_speed, clip_id, center_frame)`.
- [ ] Bind seed predecessor/first keys and row hashes to the exact fitted receipt; require same clip, adjacency, train split, and receipt digest at checkpoint save/load.
- [ ] Compare the newly materialized fitted receipt with the resume checkpoint before `restore_training_state` mutates model/optimizer.
- [ ] Arrange the expanded deterministic subset as the complete unique run containing the global seed, followed by deterministic unique runs from the sealed known-terrain identity, then remaining GRAIL runs; require flat/slope/transition coverage.
- [ ] Run focused GREEN tests.

### Task 2: Runtime world seed and exact supported terrain

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Modify: `sonic/python/mm_sonic/evaluate_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Produces: world seed initialization; exact-only `SourceAlignedTerrain`; deterministic supported-flat scenario alignment.

- [ ] Add RED tests proving nonzero seed world xy/yaw affect root/trajectory/quaternion, outside-mesh queries return `None`, and a chosen start supports every seed probe on a real flat surface before crossing a native slope.
- [ ] Run RED and record the hardcoded-zero and boundary-extrapolation failures.
- [ ] Initialize root/history/trajectory from stored finite world xy/yaw and rotate local trajectory/directions accordingly.
- [ ] Remove nearest-boundary fallback from `SourceAlignedTerrain`; return `None` unless an upward triangle contains the query.
- [ ] Deterministically select sealed `slope_001` training terrain and an exact top-surface flat start whose 36 seed probes match the flat seed and whose +X axis crosses the supported native hill before returning to supported continuation.
- [ ] Run focused GREEN tests.

### Task 3: Collision surfaces and realized metrics

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Produces: deterministic collision surface samples; realized traversal/crossing metrics; fail-closed stance evidence.

- [ ] Add RED slope regressions for sphere, capsule, box, cylinder/ellipsoid where present, and mesh samples that catch the deepest uphill/side penetration rather than only global-lowest-Z points; assert exact sole/forbidden collision sets.
- [ ] Add RED recorder tests where caller tags claim traversal but root is stationary, where realized motion crosses/returns without tags, and where zero predicted-stance samples fail.
- [ ] Run RED and record missed slope penetration, tag-only traversal, and empty-stance pass.
- [ ] Sample deterministic full primitive surfaces with fixed angular grids and all transformed collision-mesh vertices; keep the exact eight sole-sphere bottoms for allowed support geometry while using sufficient surface samples for forbidden collision geometry.
- [ ] Derive forward/backward grade reach, supported crossing, displacement, and return from realized consecutive world positions and terrain samples; retain tags only as scenario annotations.
- [ ] Record stance sample count and require it to be positive globally and for applicable walking segments.
- [ ] Run focused GREEN tests.

### Task 4: Sealed validation scenario provenance

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Produces: `ClosedLoopScenarioProvenance`; factory-bound validation scenarios; promotion-safe aggregate result.

- [ ] Add RED callback-spy tests for train/test split, forged identity receipt, dataset digest mismatch, kinematic mismatch, and terrain asset/version hash mismatch; prove rejection occurs before callback execution.
- [ ] Run RED and record that arbitrary outer-relabeled callables currently execute.
- [ ] Implement immutable scenario provenance bound to manifest schema/digest, exact validation identity-set receipt, kinematic signature, and procedural terrain asset/source hash/version.
- [ ] Require the factory to authenticate provenance before exposing/evaluating the callback; aggregate only matching validation scenario objects.
- [ ] Make normal checkpoint promotion verify the aggregate provenance receipt.
- [ ] Run focused GREEN tests.

### Task 5: Pipeline verifier receipt and guarded best promotion

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Modify: `sonic/python/mm_sonic/evaluate_terrain_pfnn.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Produces: exact pipeline verification receipt and atomic candidate-to-best promotion.

- [ ] Add RED tests proving one-step acceptance writes only a candidate, no best exists before a receipt, fixed-sample mismatch rejects, stationary phase-advancing rollout rejects, noncrossing rollout rejects, and any checkpoint/dataset/kinematic/scenario mismatch rejects.
- [ ] Run RED and record current one-step-only best publication.
- [ ] Save pipeline candidates with provisional selection only. Compute a fixed-sample reproduction receipt bound to exact fitted rows and checkpoint bytes.
- [ ] Run the 600-tick known-terrain verifier and require finite, zero holds, no reversal/freeze, all root/joint/limit/sole/forbidden/stance gates, positive realized displacement, supported hill crossing in both realized directions, and return to supported continuation. Report native max grade and leave the 18.9-degree Task 8 gate not applicable.
- [ ] Bind verifier receipt to checkpoint SHA, dataset digest, kinematic signature, fitted receipt, scenario receipt, and metrics digest; atomically copy/promote immutable `best.pt` only after full verification.
- [ ] Run focused GREEN tests.

### Task 6: One controlled expanded-coverage experiment and final evidence

**Files:**
- Modify: `.superpowers/sdd/task-7-report.md`

**Interfaces:**
- Consumes: final candidate/verifier contract.
- Produces: one preserved experimental artifact, evaluator JSON, report, and separate fix commit.

- [ ] Record hypothesis: the 256-row failure is caused by short recurrent coverage and sparse command/terrain coverage; a 2,048-row train-only subset containing the seed's complete 102-row unique-center flat run, 1,649 deterministic `slope_001` rows, and 297 `slope_000` rows, followed by 16-frame rollout fine-tuning, should prevent immediate out-of-coverage raw-output failures while fitting the exact supported known terrain.
- [ ] Run the focused runtime/training/evaluator tests and broad Task 1-7 suite before training.
- [ ] Run one fresh fixed-seed 2,048-row, 4,000-step, rollout16/256-update experiment; preserve all earlier run directories.
- [ ] If training or the exact 600-tick verifier fails, stop after recording the first failure and do not attempt another architecture.
- [ ] If it passes, reload the promoted immutable best and rerun the exact 20-second evaluator.
- [ ] Append every RED/GREEN command/output, subset receipt/classes/commands/terrain coverage, training losses, verifier receipt, all failed metrics, and self-review to `.superpowers/sdd/task-7-report.md`.
- [ ] Run final compile, `git diff --check`, focused tests, broad tests, and real evaluator; inspect the diff and commit the review fixes separately.

## Self-Review

- Spec coverage: every Critical/Important finding maps to Tasks 1-5; the one-experiment restriction and reporting are Task 6.
- Placeholder scan: no deferred implementation placeholders; Task 8 remains explicitly outside pipeline promotion.
- Type consistency: pipeline and validation receipts are distinct; both bind checkpoint/dataset/kinematic/scenario provenance, while only validation enforces the full held-out 18.9-degree gate.
