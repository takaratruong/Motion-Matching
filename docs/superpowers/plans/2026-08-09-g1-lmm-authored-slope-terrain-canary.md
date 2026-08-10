# G1 LMM Authored-Slope Terrain Canary Implementation Plan

Status: approved for test-first execution

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `subagent-driven-development` for independent tasks and follow
> `test-driven-development`, `systematic-debugging`, and
> `verification-before-completion` at every commit boundary.

**Goal:** Publish, train, reload, and personally inspect one honest 60 Hz G1
Learned Motion Matching authored-route canary on an authenticated paired GRAIL
ramp with a frozen explicit support-datum calibration.

**Architecture:** Preserve the accepted flat-v3 and projector runtime
unchanged. Add a parallel, authenticated `g1-lmm-terrain-data/v1` bundle that
combines the accepted 256-row flat clip with a 595-row admitted native-G1
GRAIL slope clip (598 provisional rows minus the mechanically rejected first
three). Train only the Orange Duck compressor/decompressor and recurrent stepper.
At runtime, initialize an exact feature/latent seed and use a separate
projector-free transaction with authoritative terrain resampling. `W` advances
the authored route and release pauses it.

**Claim:** Authenticated, support-calibrated paired flat+slope overfit and
runtime-plumbing canary only; `generalization_claim=none`.

**Tech stack:** Python 3, NumPy, SciPy, MuJoCo, PyTorch/CUDA, C++17, Raylib,
Orange Duck binary codecs.

## Frozen constraints

- Data: flat-v3 manifest SHA `5b5c48cc...5382db1`; GRAIL robot/USD hashes
  `b77480d5...561ee1` / `8d1e696f...eb5a5`; canonical G1 XML
  `749209c0...e4376`.
- Rate: exact 60 Hz, database horizons 20/40/60, ranges `[0,256)` and
  `[256,851)`.
- Registration: authenticated GRAIL reconstruction plus the shared
  Z-up-to-Holden basis. The explicit 12 mm terrain-only support calibration is
  separate from the rigid transform, exterior source height is exactly zero,
  and provisional rows `[0,3)` are rejected before fitting.
- Root representation: `Simulation` is planar XZ/yaw with Y=0; terrain/root
  recurrence uses it, while `Hips` local translation/rotation reconstructs the
  full global pelvis for rendering, FK, and geometry evaluation.
- No projector artifact, allocation, query, fallback, nearest-row
  substitution, clamp, IK, smoothing, or hidden root snap in the terrain lane.
- All source, terrain, scene, data, model, and evaluation bytes are size/SHA
  bound and atomically published.
- Flat-v3 bytes and behavior must remain unchanged.
- One GPU fit only after all non-GPU preflight is green: decompressor 100k,
  stepper 100k, existing seed/optimizer/LR/objectives and unchanged gates.
- Sole fit config: physical GPU3 UUID
  `GPU-87fb0777-4169-d4e9-12cc-382bdf7c0730` as `cuda:0`; base seed1234,
  stage seeds1234/1235/1236, batch32, constant LR1e-3, AdamW
  betas(.9,.999)/eps1e-8/weight_decay1e-3/AMSGrad with foreach/fused/
  capturable/differentiable disabled, dt1/60, overfit1000, stepper window20,
  decompressor100000, stepper100000, and no scheduler/projector.
- Any preflight or fit failure stops; no threshold, source, seed, budget, or
  registration changes after model output is observed.

## Task 1: Authenticate and register the authored GRAIL pair

**Files**

- Modify: `resources/build_g1_terrain_database.py`
- Modify: `resources/g1_terrain_builder/sources.py`
- Modify: `resources/g1_terrain_builder/scenes.py`
- Modify only if a shared exact adapter is required:
  `resources/g1_terrain_builder/terrain.py`
- Test: `tests/python/test_sources.py`
- Test: `tests/python/test_grail_terrain_source.py`
- Test: `tests/python/test_scenes.py`
- Test: `tests/python/test_build_cli.py`

**RED**

- Add tests for the exact frozen source/XML hashes, 250@25 -> 598@60
  left/right/alpha provenance, authenticated reconstruction plus shared
  Z-up-to-Holden basis, exterior-flat policy, explicit terrain-only 12 mm
  support calibration, exact 9.96 s -> 9.95 s span/0.01 s shortfall, float64
  inverse round-trip, `[0,3)` rejection, and no double application of terrain
  height.
- Assert that `Simulation` remains planar/Y=0 while FK reconstructs the native
  global `Hips` height and orientation.
- Add exact-mesh feature regression tests for the four frozen min/max/std
  columns and a synthetic tilted-plane sign/axis smoke test.

Run:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest -v tests.python.test_sources \
  tests.python.test_grail_terrain_source tests.python.test_scenes \
  tests.python.test_build_cli
```

**GREEN**

- Implement a narrow authenticated slope-source/terrain loader and one exact
  `authored-slope` scene constructor using source bytes, the reconstruction/
  common-basis paired transform, explicit exterior-flat continuation, and the
  separately named support calibration. Return a validated provisional source,
  terrain, scene, and provenance receipt; Task 2 owns combination/publication.
- Keep `_assemble_flat_candidate` and the general 25 Hz builder unchanged.

**Stop:** any source/hash/count/transform/support mismatch.

**Commit:** `feat(data): authenticate authored GRAIL slope pair`

## Task 2: Build and atomically publish the two-range data bundle

**Files**

- Modify: `resources/g1_terrain_builder/database.py`
- Modify: `resources/g1_terrain_builder/features.py`
- Modify: `resources/g1_terrain_builder/artifacts.py`
- Modify: `resources/build_g1_terrain_database.py`
- Test: `tests/python/test_database_builder.py`
- Test: `tests/python/test_artifacts.py`
- Test: `tests/python/test_build_cli.py`
- Test: `tests/python/test_terrain.py`

**RED**

- Require schema `g1-lmm-terrain-data/v1`, exactly 851 rows, two ranges,
  31 bones/features, 32 latent dimensions, two contacts, active finite terrain
  scales, exact scene/support/feature/database descriptors, and no temporal
  operation across global row 256.
- Require a mutually exclusive `--authored-slope-terrain` build mode. Its
  exact `--flat-data`, `--slope-robot`, `--slope-usd`, `--slope-recon`,
  `--slope-metadata`, and `--g1-xml` paths are mandatory and hash checked;
  flat-only/general modes reject those options before loading payloads.
- Add post-validation mutation, unexpected-file, locked-parent, exchange, and
  rollback tests for the exact hybrid tree.

**GREEN**

- Implement `_assemble_lmm_terrain_candidate`: independently authenticate/
  rebuild flat-v3, consume Task 1's authenticated slope source, discard
  provisional `[0,3)`, append its 595 rows with `combine_clips`, derive Orange
  Duck contacts, and jointly build normalized 31D features.
- Add `publish_lmm_terrain_artifacts` by reusing existing fsync, locked-parent,
  validation-under-lock, atomic-exchange, and rollback machinery. Do not weaken
  either existing publisher.
- Publish `database.bin`, `features.bin`, terrain/support sidecars, one-scene
  pack, `validation.json`, and `manifest.json`.

Run:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest -v tests.python.test_database_builder \
  tests.python.test_artifacts tests.python.test_build_cli \
  tests.python.test_terrain
```

**Stop:** flat-v3 drift, boundary leakage, inactive terrain scale, or tree/hash
failure.

**Commit:** `feat(data): publish authored-slope LMM bundle`

## Task 3: Independently validate geometry, parity, and mechanics

**Files**

- Modify: `resources/validate_g1_terrain_database.py`
- Reuse/factor only when necessary:
  `sonic/python/mm_sonic/terrain_oracle/audit.py`
- Test: `tests/python/test_validator.py`
- Test: `tests/python/test_terrain.py`
- Test: `tests/python/test_build_cli.py`
- Test: `tests/python/test_oracle_audit.py`

**RED**

- Add a separate validator dispatch that reconstructs both sources without
  trusting builder arrays, replays the source map, checks the rigid transform,
  recomputes FK/continuity/contacts/terrain features/normalization, and verifies
  raw-mesh vs raster vs walkability parity at every bound sample.
- Extend the validator CLI with the same exact flat-data/slope/XML input set as
  the builder; terrain-v1 full-source validation requires every input, while
  flat/general schemas reject irrelevant terrain-v1 source options.
- Encode an offline decoded-transform mechanics evaluator. Forward-kinematics
  the 31 Holden bones, attach every authenticated XML geom directly to its
  mapped decoded bone (no hinge-qpos projection), evaluate exact terrain depth,
  and place the same geom transforms into temporary MuJoCo collision data for
  enabled self-contact. On source rows require direct-geom/MuJoCo-forward
  transform parity <=1e-5 m/rad.
- Use the authenticated XML's four ankle-roll sphere bottoms per foot for
  calibrated penetration, clamped minimum-probe support gap, seven-tick-
  confirmed planted tangent drift, forbidden robot-terrain contacts, and
  enabled self-contact. Record unaided/calibrated float64 penetration
  `0.017864829147878552` / `0.0058648290435704235` m, binary32 calibrated
  parity `0.0058648294757275045` m, and float64/binary32 support gap
  `0.00021573805692624848` / `0.0002157502790678112` m.
- Require exact rollout seeds 0, 421, and 530 with 60/120/240 successors.

**GREEN**

- Implement `_validate_lmm_terrain_artifact_directory` without changing the
  flat-v3 validator.
- Run the real builder twice, require byte-identical trees, and run the
  standalone validator from a fresh process.

Run:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest -v tests.python.test_validator tests.python.test_terrain \
  tests.python.test_build_cli tests.python.test_oracle_audit
```

**Hard stop before GPU:** any spec data/mechanical/parity gate fails.

**Commit:** `feat(validation): gate authored-slope LMM data`

## Task 4: Add the projector-free Python training contract

**Files**

- Modify: `resources/g1_lmm/dataset.py`
- Modify: `resources/g1_lmm/training.py`
- Modify: `resources/train_g1_lmm.py`
- Test: `tests/python/test_g1_lmm_training.py`

**RED**

- Add a sibling terrain fixture and tests for exact schema/source/range/scene
  binding, all-row compressor/decompressor windows, range-safe stepper windows,
  exact seed populations, and rejection of flat-v3 in the terrain branch.
- Add a mutually exclusive `--authored-slope-terrain-canary` training selector
  and a separate exact terrain config/receipt containing every frozen optimizer,
  seed, window, dt, budget, deterministic-CUDA, and physical-GPU field. Its
  `--stage all` path means overfit -> decompressor -> stepper and cannot dispatch
  or fall through to projector training.
- Require model schema `g1-lmm-terrain-model/v1`, scopes from the design, and
  the exact artifact set `latent.bin`, `decompressor.bin`, `stepper.bin`,
  `training.json`, `evaluation.json`; any `projector.bin` is an error.
- A successful training manifest has `status=accepted` only for
  `acceptance_scope=numerical-training-only`,
  `artifact_role=canary_candidate`, and `viewer_authorized=false`. It cannot
  authorize interactive use without the separate final runtime receipt.
- Exercise every decompressor and nine stepper rollout reducer with deterministic
  tiny fixtures. Assert configured budgets 100k/100k.

**GREEN**

- Add separate `load_terrain_training_bundle` and `train_terrain_bundle`
  branches while factoring only behavior that is truly identical to flat-v3.
- Make every AdamW argument explicit and bind the unchanged constant-LR/no-
  scheduler behavior in the receipt.
- Export the latent table plus decompressor/stepper; do not construct projector
  queries or a projector network.

Run:

```bash
PYTHONPATH=.:resources sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_g1_lmm_training
```

**Stop:** cross-range samples, projector construction, schema drift, or a
flat-v3 regression.

**Commit:** `feat(training): define projector-free terrain LMM canary`

## Task 5: Authenticate the terrain data/model in C++

**Files**

- Modify: `scene_runtime.h`
- Modify: `lmm.h`
- Test: `tests/cpp/test_scene_runtime.cpp`
- Test: `tests/cpp/test_g1_runtime.cpp`
- Test: `tests/cpp/test_g1_lmm.cpp`

**RED**

- Add positive terrain data/model fixtures and missing/extra/type/size/SHA/
  schema/scope/seed tamper matrices that reject before evaluation allocation.
- Require the exact numerical-only candidate lifecycle fields; a model manifest
  cannot claim final viewer authorization.
- Require exactly two allocated network evaluations and no projector bytes or
  evaluation for terrain models. Retain all existing flat/projector tests.

**GREEN**

- Add a distinct `G1_LMMTerrainDataSchema` discriminator and exact two-range
  parser; do not reinterpret `flat_lmm_bundle`.
- Extend the existing model bundle/loader transaction with a terrain-model
  branch for latent/decompressor/stepper only.

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime && \
  /tmp/test_scene_runtime
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime && \
  /tmp/test_g1_runtime
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm && /tmp/test_g1_lmm
```

**Commit:** `feat(runtime): load projector-free terrain LMM bundles`

## Task 6: Implement the authored recurrent transaction and viewer controls

**Files**

- Modify: `sonic/cpp/g1_runtime.h`
- Modify: `g1_controller_state.h`
- Modify: `motion_match_log.h`
- Modify: `controller.cpp`
- Create: `resources/check_g1_lmm_terrain_canary.py`
- Modify: `sonic/python/mm_sonic/terrain_oracle/audit.py`
- Test: `tests/cpp/test_g1_lmm.cpp`
- Test: `tests/cpp/test_g1_runtime.cpp`
- Test: `tests/cpp/test_g1_controller_state.cpp`
- Test: `tests/cpp/test_route_runtime.cpp`
- Test: `tests/cpp/test_terrain_runtime.cpp`
- Test: `tests/python/test_runtime_log.py`
- Test: `tests/python/test_oracle_audit.py`

**RED**

- Verify exact-seed initialization and the order: current authoritative terrain
  overwrite -> one recurrent step -> provisional decode -> predicted-root
  terrain resample -> final decode -> finite/fixed-point/terrain/continuity
  gates -> atomic commit. Exact MuJoCo mechanics are an offline acceptance gate,
  not falsely claimed as an in-transaction viewer check.
- Verify no projector call/allocation, complete rollback on every late failure,
  `W` advance, key-release pause, A/S/D rejection, and bitwise no mutation on
  pause/reject.
- Bind/log authored row, source/local and registered pose, terrain prediction,
  fixed-point/resample values, complete mechanics inputs, network/commit
  counts, and engine ownership.
- Freeze reset at global row256 with exact pose/features/latent, 594 committed
  successors257..850, release/reject no mutation, and terminal W at row850
  returning `AUTHORED ROUTE COMPLETE` before any recurrent call.
- Add an authenticated little-endian evidence trace containing all 31 final
  global bone positions/quaternions, contacts, Simulation XZ/yaw, terrain,
  cursor, and ownership counters for reset plus every committed successor.
- Add schema `g1-lmm-terrain-canary-evaluation/v1`. The checker atomically
  publishes accepted/rejected receipts binding data/model/XML/trace/log/code
  hashes and every numeric reduction. Only `status=accepted` and
  `viewer_authorized=true` in this separate receipt authorize interactive
  mode; deterministic evidence mode may load the numerical candidate solely
  under `MM_TEST_MODE=authored-slope`.
- Add missing/extra/type/value/hash tamper tests for that receipt, rejected-
  receipt publication tests, and a fail-before-window interactive-startup test
  when authorization is absent or stale.

**GREEN**

- Implement `g1_runtime_step_lmm_authored` as a separate transaction; preserve
  `g1_runtime_step_lmm_normalized` for flat projector models.
- Add the one-scene authored temporal route and controller gate. Increment its
  cursor only after a committed LMM tick.
- Extend the Task 3 offline mechanics evaluator to consume the runtime evidence
  trace and apply the identical direct-geom collision, self-contact, sole,
  drift, and support-gap reductions before viewer acceptance.

Run the five strict C++ tests directly with the same flags above, then:

```bash
PYTHONPATH=.:resources:sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_runtime_log tests.python.test_oracle_audit
```

Syntax/full-link the exact acceptance binary:

```bash
g++ -std=c++17 -O2 -I. \
  -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp /home/ubuntu/apps/raylib/src/libraylib.a \
  -lGL -lm -lpthread -ldl -lrt -lX11 \
  -o /tmp/controller_g1_lmm_terrain
```

**Stop:** projector/ordinary fallback, mutation on pause/reject/end, wrong
terrain order, incomplete evidence trace, or source/runtime metric mismatch.

**Commit:** `feat(viewer): replay authored slope through recurrent LMM`

## Task 7: Freeze the non-GPU integration checkpoint

No production edits or commit.

- Rebuild and independently validate the real data bundle twice.
- Run the synthetic-plane smoke, exact source mechanical evaluator, C++ data
  reload, and synthetic accepted terrain-model reload.
- Run the full relevant Python/C++ suites and prove flat-v3 hashes unchanged.
- Record immutable data/tree hashes and the sole training command/config.

Use the exact builder inputs below for both temporary builds and the final
canonical publication at `sonic/runs/g1-lmm-terrain-60hz/data-v1`:

```bash
PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py \
  --authored-slope-terrain --output-fps 60 \
  --flat-data sonic/runs/g1-lmm-flat-60hz/data-v3 \
  --slope-robot /home/ubuntu/datasets/GRAIL/data/slope/robot/terrain_slopes__slope_000__000.pkl \
  --slope-usd /home/ubuntu/datasets/GRAIL/data/slope/object_usd/terrain_slopes__slope_000__000.usd \
  --slope-recon /home/ubuntu/datasets/GRAIL/data/slope/recon/terrain_slopes__slope_000__000.pkl \
  --slope-metadata /home/ubuntu/datasets/GRAIL/data/slope/meta/terrain_slopes__slope_000__000.pkl \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output sonic/runs/g1-lmm-terrain-60hz/data-v1

PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py \
  sonic/runs/g1-lmm-terrain-60hz/data-v1 --full-source-validation \
  --flat-data sonic/runs/g1-lmm-flat-60hz/data-v3 \
  --slope-robot /home/ubuntu/datasets/GRAIL/data/slope/robot/terrain_slopes__slope_000__000.pkl \
  --slope-usd /home/ubuntu/datasets/GRAIL/data/slope/object_usd/terrain_slopes__slope_000__000.usd \
  --slope-recon /home/ubuntu/datasets/GRAIL/data/slope/recon/terrain_slopes__slope_000__000.pkl \
  --slope-metadata /home/ubuntu/datasets/GRAIL/data/slope/meta/terrain_slopes__slope_000__000.pkl \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
```

The two temporary outputs use fresh `mktemp -d` parents with the same arguments
except `--output`; require byte-identical relative trees before the canonical
publication.

**Hard stop:** any failure or dirty/unreviewed code/config. Do not launch CUDA.

## Task 8: Run the sole GPU fit and numerical gates

No implementation commit.

- Train once on an idle named physical GPU: decompressor 100k, then stepper
  100k. Preserve first-failure semantics and atomic rejected receipts.
- Require all-row decompressor gates and each flat/ascent/descent 60/120/240
  recurrent gate.
- Safely reload the numerically accepted immutable `canary_candidate` in a
  fresh Python and C++ evidence-mode process. This does not authorize the
  interactive viewer; only Task 9's separate accepted runtime receipt does.

The sole command is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=.:resources sonic/.torch-mm-venv/bin/python \
  resources/train_g1_lmm.py \
  sonic/runs/g1-lmm-terrain-60hz/data-v1 \
  sonic/runs/g1-lmm-terrain-60hz/model-v1 \
  --authored-slope-terrain-canary --stage all --device cuda:0 \
  --seed 1234 --batch-size 32 --learning-rate 0.001 \
  --overfit-steps 1000 --decompressor-steps 100000 \
  --stepper-steps 100000 --stepper-window 20
```

**Hard stop:** any nonfinite, gate, export, receipt, or reload failure. No retry
or tuning.

## Task 9: Personally run the terrain viewer and hand it off

No implementation commit unless runtime debugging uncovers a reproducible code
defect; any such fix returns through RED/GREEN review before rerun.

- Drive the exact authored route with `W`, pause/restart with release, and
  inspect both ascent and descent against source A/B.
- Require every runtime, collision, contact, drift, support-gap, terrain,
  ownership, and continuity gate.
- Keep zeroed/shuffled terrain report-only; never use it to rescue acceptance.
- Leave the accepted viewer in `authored-slope` mode for the user.

The deterministic evidence run first emits exactly 595 trace records (reset
row256 plus 594 commits through row850), probes terminal no-mutation on its
final controller frame, and exits:

```bash
mkdir -p sonic/runs/g1-lmm-terrain-60hz/evidence-v1

DISPLAY=:1 \
G1_TERRAIN_DIR=sonic/runs/g1-lmm-terrain-60hz/data-v1 \
G1_LMM_MODEL_DIR=sonic/runs/g1-lmm-terrain-60hz/model-v1 \
MM_TERRAIN_SCENE=authored-slope \
MM_TEST_MODE=authored-slope MM_TEST_FRAMES=595 \
MM_LMM_TRACE=sonic/runs/g1-lmm-terrain-60hz/evidence-v1/runtime.trace.bin \
MM_LOG=sonic/runs/g1-lmm-terrain-60hz/evidence-v1/runtime.csv \
/tmp/controller_g1_lmm_terrain --locomotion-engine lmm

PYTHONPATH=.:resources:sonic/python sonic/.torch-mm-venv/bin/python \
  resources/check_g1_lmm_terrain_canary.py \
  --data sonic/runs/g1-lmm-terrain-60hz/data-v1 \
  --model sonic/runs/g1-lmm-terrain-60hz/model-v1 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --trace sonic/runs/g1-lmm-terrain-60hz/evidence-v1/runtime.trace.bin \
  --log sonic/runs/g1-lmm-terrain-60hz/evidence-v1/runtime.csv \
  --output sonic/runs/g1-lmm-terrain-60hz/evidence-v1/evaluation.json
```

The checker always atomically writes a valid accepted/rejected receipt and
returns nonzero on rejection. After an accepted receipt, the personal
interactive run omits test mode and supplies that authorization:

```bash
DISPLAY=:1 \
G1_TERRAIN_DIR=sonic/runs/g1-lmm-terrain-60hz/data-v1 \
G1_LMM_MODEL_DIR=sonic/runs/g1-lmm-terrain-60hz/model-v1 \
G1_LMM_CANARY_EVALUATION=sonic/runs/g1-lmm-terrain-60hz/evidence-v1/evaluation.json \
MM_TERRAIN_SCENE=authored-slope \
/tmp/controller_g1_lmm_terrain --locomotion-engine lmm
```

Passing establishes only calibrated paired-route G1 LMM terrain replay, not
command coverage or unseen-terrain generalization.
