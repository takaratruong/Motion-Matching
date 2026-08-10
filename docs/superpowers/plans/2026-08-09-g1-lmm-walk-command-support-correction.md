# G1 LMM Walk Command-Support Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and personally drive one honest 60 Hz G1 learned-motion-matching flat-walking canary whose projector is trained and gated on the same supported cardinal command population used at runtime.

**Architecture:** Retarget four bounded walking-only PFNN intervals into an immutable multi-clip data-v4 bundle. Generate a separate immutable command-support bundle with the ordinary 60 Hz runtime, train the unchanged Orange Duck networks against those bound queries, and hard-reject unsupported queries transactionally in the LMM runtime.

**Tech Stack:** Python 3, NumPy, PyTorch/CUDA, MuJoCo/GMR retargeting, C++17, Orange Duck binary network ABI, Raylib controller.

## Global Constraints

- Exact output rate is 60 Hz with horizons 20/40/60.
- Admit only stand/walk/start/stop/gentle-turn rows; reject run, jog, jump, crouch, strafe, and backward-walk rows. Keep per-row labels inside source/continuity ranges so transition windows remain learnable.
- Keep projector architecture, objective, optimizer, learning rate, batch, 50k budget, and `.05/.10/.50` thresholds unchanged.
- Decoder and stepper budgets are 100k each for the sole full run.
- Fix `desired_gait=1` (walk endpoint) and disable the gait/strafe toggles in LMM-canary mode.
- No automatic fallback, nearest-row substitution, query clamp, pose hold, or hidden reset is allowed.
- All artifacts are immutable, hash/size bound, little-endian, and fail closed before model/evaluation allocation.
- The sole full GPU run starts only after data and command-support preflight passes.
- One full run only; do not tune seed, thresholds, budgets, or corpus after seeing its result.

---

### Task 1: Retarget and publish the walking-only data-v4 bundle

**Files:**
- Modify: `resources/build_g1_terrain_database.py`
- Modify: `resources/validate_g1_terrain_database.py`
- Modify: `resources/g1_terrain_builder/artifacts.py`
- Modify: `resources/g1_terrain_builder/sources.py`
- Modify: `resources/g1_terrain_builder/schema.py`
- Test: `tests/python/test_build_cli.py`
- Test: `tests/python/test_g1_terrain_builder.py`

**Interfaces:**
- Consumes four exact 120 Hz BVH intervals and their exact 18-field G1 retarget receipts.
- Produces immutable `g1-lmm-flat-data/v4` with `database.bin`, `features.bin`, `motion_labels.bin`, and `manifest.json`.

- [ ] **Step 1: Generate the four retarget candidates at new paths**

Run the pinned `mm_sonic.retarget_pfnn_bvh_g1` CLI for Flat08 `[0,840)`, `[11160,12000)`, `[12090,12930)`, and Flat11 `[17062,17790)`, with `--grounding flat` and no output overwrite.

- [ ] **Step 2: Audit the candidates without publishing**

Require exact receipt/source/prepared/output hashes, pinned commits, canonical joints, finite values, joint limits, flat grounding, bilateral contacts, native/local continuity `<=0.25 rad/frame`, root-speed/turn distributions, collision-proxy bounds, and exact gait exclusion. Stop if any candidate fails.

- [ ] **Step 3: Write RED multi-source/schema tests**

Add focused tests for exact four-source identities/intervals, duration-preserving gait resampling, the exact `0.8`/`1e-6` gait rules, start/stop derivative labels, `[0.25,0.85] m` and `-35/-10/+10/+35` degree trajectory bins, the fixed minimum label counts, gait contamination, source-map/range digests, data-v3 rejection by the new loader, data-v4 tamper rejection, and a clean-checkout self-contained fixture.

- [ ] **Step 4: Run RED**

Run the focused Task-1 tests and require failure only for missing data-v4/multi-source behavior.

- [ ] **Step 5: Implement minimal data-v4 publication**

Reuse the current authenticated-byte G1 XML path, exact resampling, fragment-local filtering, continuity splitting, OD bilateral contact derivation, feature generation, and transactional publisher. Add exact motion-label encoding and four-source receipt binding without weakening data-v3 validation.

- [ ] **Step 6: Run focused and full modified-area GREEN**

Run Task-1 focused tests, builder/validator suites, `py_compile`, and `git diff --check`.

- [ ] **Step 7: Publish twice and compare**

Build data-v4 transactionally twice, run the standalone validator, and require byte-identical database/features/labels/manifest hashes.

- [ ] **Step 8: Commit and review**

Stage only Task-1 files, commit, and obtain independent review before Task 2.

---

### Task 2: Generate the immutable command-support bundle

**Files:**
- Create: `resources/g1_lmm_command_support.cpp`
- Create: `resources/build_g1_lmm_command_support.py`
- Modify: `resources/g1_lmm/dataset.py`
- Test: `tests/cpp/test_g1_lmm_command_support.cpp`
- Test: `tests/python/test_g1_lmm_command_support.py`

**Interfaces:**
- Consumes exact data-v4 and unchanged ordinary 60 Hz controller/runtime code.
- Produces `g1-lmm-command-support/v1` containing `queries.bin`, `targets.bin`, `command_matrix.json`, and `manifest.json`.

- [ ] **Step 1: Write RED query-generation and tamper tests**

Cover exact `desired_gait=1`, disabled gait/strafe toggles, the seven exact 120-tick key-mask tapes and eight-per-split odd/even seventeenths seed-rank algorithm, deterministic applied-command/query bits, numeric-zero then positive-zero-canonicalized terrain, exact nearest row/class/terrain, separately named L2/RMSE support, 30-tick destination deadlines, disjoint seed rows/query bytes, code/compiler/flags/binary digest binding, and every artifact/manifest tamper.

- [ ] **Step 2: Run RED**

Compile with C++17 warnings-as-errors and run the focused Python tests; require missing-tool/schema failures.

- [ ] **Step 3: Implement deterministic ordinary-runtime recording**

Load data-v4, select the preregistered seed ranks, execute the exact command tapes at binary32 60 Hz through the same command/query implementation used by the viewer, serialize raw/normalized queries and oracle receipts, and fail at the first unsupported/wrong-class/deadline tick.

- [ ] **Step 4: Implement atomic support publication and independent validation**

Use exclusive staging, canonical JSON, little-endian matrices, exact hashes/sizes, and a second parser/recomputer that does not trust producer summaries.

- [ ] **Step 5: Run focused/full GREEN and static checks**

Run strict C++ tests, Python support tests, `py_compile`, and `git diff --check`.

- [ ] **Step 6: Publish twice and freeze**

Generate two support bundles from the frozen data-v4/code/config, require byte-identical outputs, then record all hashes.

- [ ] **Step 7: Commit and review**

Stage only Task-2 files, commit, and obtain independent review before GPU training code changes.

---

### Task 3: Train and publish model-v2 against supported queries

**Files:**
- Modify: `resources/g1_lmm/dataset.py`
- Modify: `resources/g1_lmm/training.py`
- Modify: `resources/train_g1_lmm.py`
- Test: `tests/python/test_g1_lmm_training.py`

**Interfaces:**
- Consumes accepted data-v4 and command-support-v1 bundles.
- Produces `g1-lmm-model/v2`, `projector_support.json`, and exact training/evaluation receipts.

- [ ] **Step 1: Write RED loader/population/metric/publication tests**

Cover exact support manifest binding, no-replacement/disjoint-byte 4,096 command plus 4,096 supported-noise training queries, disjoint command plus 512 supported-noise gate queries, the exact 65,536-candidate caps, zero terrain noise, support rejection, honestly named raw-latent error, mixed-unit maximum parity, per-command-tick gates, accepted schema/scope, and every tamper.

- [ ] **Step 2: Run RED**

Run only the new focused tests and require failures for missing support/model-v2 behavior.

- [ ] **Step 3: Implement supported projector populations**

Keep the existing network/objective/optimizer/budget. Replace only the fixed unsupported pool/gate with receipt-bound command queries plus deterministic rejection-sampled, flat-zero, in-support noise queries.

- [ ] **Step 4: Correct projector diagnostics units**

Keep the existing raw latent units and `.10` threshold. Rename the misleading latent and mixed maximum fields without changing arithmetic.

- [ ] **Step 5: Bind model-v2 and projector support**

Hash data/support/query/oracle receipts, exact scopes, `generalization_claim=none`, thresholds, runtime policy, and all network artifacts. Reject old schemas before model construction.

- [ ] **Step 6: Run full trainer verification**

Run focused/full Task-3 tests, isolated deterministic CUDA export tests, real bundle preflight, `py_compile`, and `git diff --check`.

- [ ] **Step 7: Commit and review**

Commit only Task-3 files and obtain independent review before the sole GPU run.

---

### Task 4: Enforce per-tick support in the C++ runtime

**Files:**
- Modify: `lmm.h`
- Modify: `sonic/cpp/g1_runtime.h`
- Modify: `g1_controller_state.h`
- Modify: `controller.cpp`
- Test: `tests/cpp/test_g1_lmm.cpp`
- Test: `tests/cpp/test_g1_runtime.cpp`
- Test: `tests/cpp/test_g1_controller_state.cpp`

**Interfaces:**
- Consumes accepted data-v4, support-v1, and model-v2.
- Produces one transactional LMM tick with exact support/projector diagnostics and no fallback.

- [ ] **Step 1: Write RED schema/support/transaction/UI tests**

Cover model/support exact-key parsing before allocation, every hash/type/value tamper, exact NN oracle parity with separately named L2/RMSE, per-tick `.50/.05/.10/.50` rejections before terrain overwrite, late-failure bitwise rollback, zero ordinary commits, disabled gait toggle with `desired_gait=1`, rejected strafe mode, and ordinary-mode overlay byte/layout preservation.

- [ ] **Step 2: Run RED strict builds**

Compile and run focused tests with C++17 warnings-as-errors; require missing model-v2/support-guard failures.

- [ ] **Step 3: Implement exact support and projector diagnostics**

Compute exact nearest feature/latent from the small database in normalized space, evaluate fixed thresholds, stage all diagnostics/state on the clone, and commit once only after every gate passes.

- [ ] **Step 4: Implement explicit canary controls and overlay**

Fix `desired_gait=1`, disable the gait toggle, reject strafe mode, show scope/support metrics, and preserve ordinary controller UI/layout exactly.

- [ ] **Step 5: Run strict/full relevant GREEN**

Run C++ unit/runtime/controller tests, O3/fast-math rollback tests, controller syntax/full link, Python receipt tests, and `git diff --check`.

- [ ] **Step 6: Commit and review**

Commit only Task-4 files and obtain independent review before training.

---

### Task 5: Sole training run, scripted drive, and personal visual acceptance

**Files:**
- Write ignored receipts/reports only under `sonic/runs/g1-lmm-flat-60hz/` and `.superpowers/sdd/`.

**Interfaces:**
- Consumes frozen reviewed source/data/support/runtime/trainer hashes.
- Produces either one immutable accepted model-v2 plus drive evidence, or one immutable rejection and terminal stop.

- [ ] **Step 1: Freeze and verify**

Require clean tracked state, exact commits/hashes, idle named GPU, accepted data/support, absent output, and no live trainer/viewer.

- [ ] **Step 2: Launch exactly one full CUDA run**

Use decoder 100k, stepper 100k, projector 50k, exact seed/config, data-v4, support-v1, and a new output directory. Stop at the first gate failure with no retry.

- [ ] **Step 3: Verify accepted publication or terminal rejection**

On success, require all gates, exact artifact closure, safe reload, model-v2 scopes, and no staging residue. On failure, preserve only sanitized receipts and stop.

- [ ] **Step 4: Run the preregistered scripted LMM matrix**

Require every tick to be LMM-owned, supported, finite, within pose/collision/contact bounds, and free of holds/resets/rejections/fallbacks.

- [ ] **Step 5: Personally drive and visually inspect**

Open the real viewer, drive W/A/S/D within the declared canary envelope, inspect still screenshots and live motion for twisting, sliding, pose discontinuity, contact error, and control response. Debug runtime/code defects with the unchanged model; do not retrain.

- [ ] **Step 6: Open the unchanged interactive viewer for the user**

Leave the accepted viewer process running only after scripted, numerical, and personal visual gates pass.

- [ ] **Step 7: Final verification and whole-branch review**

Run dependency-split Python/C++/static suites, verify clean tracked state and immutable artifact hashes, obtain broad independent review, and record the final evidence.
