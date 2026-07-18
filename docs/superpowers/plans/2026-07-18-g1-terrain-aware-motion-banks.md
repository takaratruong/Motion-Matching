# G1 Terrain-Aware Motion Banks Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to execute this plan task-by-task. Every task
> receives a fresh implementer, a task-scoped review, fixes for all Critical or
> Important findings, a clean re-review, and a pushed checkpoint.

**Goal:** Build and launch a classic-UI corrected-mesh G1 visualizer whose
motion matcher uses real flat, curb, slope, and stair data; searches by terrain
family and heading-relative travel direction; and exposes physical sole slip
and penetration while IK remains disabled.

**Architecture:** Keep one authenticated Holden pose database but add explicit
source terrain banks and a compact per-frame direction/speed index. Replace the
four-height terrain feature with a twelve-value longitudinal/lateral profile,
classify runtime terrain as flat/slope/curb/stair, and search only compatible
source ranges and direction masks. Build against the complete GRAIL slope and
stair robot corpora. Record read-only sole diagnostics before changing
selection, freeze a quality baseline, and require the new matcher to beat it
before replacing the running visualizer.

**Tech Stack:** Python 3.10, NumPy/SciPy, MuJoCo, OpenUSD, joblib,
`huggingface_hub`, C++17, GCC, Raylib/Raygui, X11, ffmpeg, Git.

## Global Constraints

- Work only in
  `/home/ubuntu/projects/motion-matching/.worktrees/g1-playable-mesh` on branch
  `g1-playable-mesh`.
- Push every coherent verified checkpoint to
  `checkpoint/g1-playable-mesh`; generated datasets and multi-gigabyte packs
  remain untracked.
- Preserve the classic Raygui UI, keyboard/gamepad controls, orbit camera,
  corrected 35-part mesh, exact 25 Hz cadence, and independent desired travel
  versus desired heading contract.
- Terrain classification, search, fallback, and safe-stop logic may never
  rotate heading toward travel.
- Keep scene terrain, motion data, and mesh assets separate. Do not create
  per-scene precomputed animation.
- Use the real GRAIL flat/curb/slope/stair sources. Do not relabel turns as
  strafes or infer staircase geometry from root height.
- The complete `stair_p1/robot` and `stair_p2/robot` corpora are acquired and
  audited. Runtime work is bounded by source-bank and direction indexes, not a
  global four-million-frame scan.
- Phase A is strictly pre-IK: every deterministic log and the launched
  visualizer must report `ik_enabled=0`. Do not change `g1_ik_runtime.h`, IK
  solver math, IK UI, or the unfinished transactional branch.
- Version-3 live, route, and terrain runs default to an effective terrain
  feature weight of 4.0 unless an explicit valid override is supplied. The UI
  may still expose the weight, but the playable terrain build may not silently
  start with terrain matching disabled.
- Keep the current user visualizer, PID `1680537`, running and untouched until
  Task 13 has a pushed candidate that passes every finite and live-input gate.
- New artifact builds go to a separate candidate directory. Never overwrite
  `/home/ubuntu/projects/motion-matching/resources/g1_terrain` in place.
- TDD is mandatory: install a focused failing test, observe the expected RED
  failure, implement the smallest production change, observe GREEN, then
  refactor if necessary.
- The starting branch has one known unrelated baseline failure:
  `test_contract_definitions_have_one_production_owner` detects the same
  `0x3d23d70a` literal in `g1_ik_runtime.h` and
  `g1_kinematic_contract.h`. Do not broaden Phase A to fix it. All other tests
  must pass and this exact exception must be reported separately.
- Use `/home/ubuntu/projects/motion-matching-verifications` for disposable
  videos. Purge rejected/accepted clips only after the user finishes reviewing
  them.

## Execution and Review Protocol

Before Task 1, initialize `.superpowers/sdd/progress.md` with the branch,
approved spec commit `3cac0e3`, known baseline exception, and Tasks 1-13 as
pending. For each task:

1. record the task base commit;
2. generate a task brief with the subagent-driven-development helper;
3. dispatch one fresh implementer;
4. require focused RED/GREEN evidence, self-review, commit, push, and a durable
   task report;
5. generate a review package from the recorded base through task HEAD;
6. dispatch a fresh task reviewer;
7. fix and re-review every Critical or Important finding; and
8. append the clean commit range to the progress ledger.

Do not pause between tasks unless an external-data ambiguity, design conflict,
or repeated blocker genuinely prevents progress. After Task 13, request one
broad whole-branch review. Do not merge this branch as part of Phase A.

---

### Task 1: Pin and Acquire the Complete GRAIL Stair Robot Corpus

**Files:**

- Create: `resources/grail_terrain_acquisition.py`
- Create: `resources/grail_terrain_inputs.json`
- Create: `tests/python/test_grail_terrain_acquisition.py`
- External create/update:
  `/home/ubuntu/datasets/GRAIL/g1_mm_inventory.json`

**Interfaces:**

- Pinned dataset:
  `nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL` at revision
  `943946a972d5de2eb0d2ff214b236d0e43575fd7`.
- Required robot paths:
  `data/stair_p1/robot/*.pkl` and `data/stair_p2/robot/*.pkl`.
- Locked robot inventory: 6,094 files and 1,210,648,890 bytes for `stair_p1`;
  6,094 files and 1,210,377,370 bytes for `stair_p2`; 12,188 files and
  2,421,026,260 bytes total.
- Production command supports `inventory`, `download`, and `verify` without
  importing or executing pickle payloads.

- [ ] **Step 1: Install acquisition-contract tests**

Write tests that require:

- exact repository ID, revision, allowed robot globs, counts, byte totals, and
  canonical inventory SHA-256 in `resources/grail_terrain_inputs.json`;
- rejection of path traversal, symlinks, directories, duplicates, unexpected
  modalities, changed byte sizes, missing files, and non-regular files;
- canonical lexical path ordering and streaming SHA-256 hashing;
- atomic external inventory publication; and
- no `joblib.load`, `pickle.load`, or source execution during acquisition.

Run:

```bash
python3 -m unittest tests.python.test_grail_terrain_acquisition -v
```

Expected RED: import/file failure because the acquisition module and pinned
manifest do not exist.

- [ ] **Step 2: Implement inventory, download, and offline verification**

Implement a small CLI around `huggingface_hub` that enumerates only the pinned
paths, compares the remote tree to the locked manifest, downloads into the
existing `/home/ubuntu/datasets/GRAIL` root with resumable semantics, hashes
regular files by streaming reads, and atomically writes the external inventory.
The verifier must operate offline after download.

Run the focused tests again. Expected GREEN: all acquisition tests pass without
network access by using their temporary fake inventory.

- [ ] **Step 3: Acquire and verify both robot partitions**

```bash
python3 resources/grail_terrain_acquisition.py download \
  --manifest resources/grail_terrain_inputs.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --modalities robot
python3 resources/grail_terrain_acquisition.py verify \
  --manifest resources/grail_terrain_inputs.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --modalities robot \
  --inventory /home/ubuntu/datasets/GRAIL/g1_mm_inventory.json
```

Expected: exact locked counts and byte totals, no missing/unexpected robot
files, and a canonical inventory hash equal to the checked-in manifest.

- [ ] **Step 4: Commit and push**

```bash
git add resources/grail_terrain_acquisition.py \
  resources/grail_terrain_inputs.json \
  tests/python/test_grail_terrain_acquisition.py
git commit -m "feat: pin complete GRAIL stair motion inputs"
git push checkpoint g1-playable-mesh
```

Do not add the external inventory or downloaded files.

---

### Task 2: Reconstruct Source Terrain from Post-RL Object Data

**Files:**

- Modify: `resources/g1_terrain_builder/sources.py`
- Modify: `resources/g1_terrain_builder/terrain.py`
- Modify: `resources/grail_terrain_inputs.json`
- Modify: `resources/grail_terrain_acquisition.py`
- Modify: `tests/python/test_sources.py`
- Modify: `tests/python/test_terrain.py`
- Modify: `tests/python/test_grail_terrain_acquisition.py`

**Interfaces:**

- Add a strict `load_grail_object_pose(path)` loader for released post-RL
  `objects/*.pkl` files.
- Add `GrailTerrain.from_release(usd_path, objects_path)` using the static
  post-RL object transform in the same simulation frame as `robot/*.pkl`.
- Require `root_pos`, `root_quat`, `fps`, and `scale` with exact shapes,
  finite values, normalized quaternions, static-transform tolerance, and one
  object record.
- Preserve `GrailTerrain.from_base` as the accepted curb oracle until parity is
  proven.

- [ ] **Step 1: Install post-RL object and surface parity tests**

Tests must reject malformed or moving terrain-object records and compare the
new release path against the existing reconstruction path for the four locked
curb fixtures. Compare transformed vertex bounds, top height, and a grid of
height queries using strict tolerances chosen before production code changes.
Also prove caller arrays are copied/frozen and USD topology validation remains
unchanged.

```bash
python3 -m unittest \
  tests.python.test_sources \
  tests.python.test_terrain -v
```

Expected RED: the post-RL loader and `from_release` do not exist.

- [ ] **Step 2: Implement the loader and transform**

Use explicit xyzw-to-wxyz conversion and the existing Z-up-to-Holden basis
contract. Do not compensate with empirical offsets, scaling, or root-motion
alignment. If curb parity fails, diagnose the released coordinate convention;
do not weaken the oracle.

Run the focused tests. Expected GREEN: all source/terrain tests pass and all
four release surfaces match the accepted curb surfaces.

- [ ] **Step 3: Extend and acquire required terrain modalities**

Extend the pinned acquisition manifest to the matching slope/stair
`objects/*.pkl` and geometry-bearing `object_usd/**/*.usd` files only. Do not
download videos, reconstructions, textures unless a USD cannot open without a
specific referenced geometry file; any such exception must be explicit in the
allowlist and tests.

```bash
python3 resources/grail_terrain_acquisition.py download \
  --manifest resources/grail_terrain_inputs.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --modalities objects object_usd
python3 resources/grail_terrain_acquisition.py verify \
  --manifest resources/grail_terrain_inputs.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --modalities robot objects object_usd \
  --inventory /home/ubuntu/datasets/GRAIL/g1_mm_inventory.json
```

Expected: every required slope/stair robot basename resolves to exactly one
object-pose file and one openable geometry asset.

- [ ] **Step 4: Commit and push**

```bash
git add resources/g1_terrain_builder/sources.py \
  resources/g1_terrain_builder/terrain.py \
  resources/grail_terrain_inputs.json \
  resources/grail_terrain_acquisition.py \
  tests/python/test_sources.py tests/python/test_terrain.py \
  tests/python/test_grail_terrain_acquisition.py
git commit -m "feat: reconstruct GRAIL post-RL terrain surfaces"
git push checkpoint g1-playable-mesh
```

---

### Task 3: Add Read-Only Sole Diagnostics and Freeze the Pre-Matcher Baseline

**Files:**

- Create: `g1_sole_diagnostics.h`
- Create: `tests/cpp/test_g1_sole_diagnostics.cpp`
- Create: `resources/summarize_g1_motion_quality.py`
- Create: `tests/python/test_motion_quality_summary.py`
- Create: `tests/fixtures/g1_motion_quality_baseline.json`
- Modify: `motion_match_log.h`
- Modify: `g1_runtime_diagnostics.h`
- Modify: `controller.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`

**Interfaces:**

- Transform the four authoritative `sole_points_local` values from each
  `G1LegConfig` through the final rendered ankle pose.
- Query G1HF/v2 at all eight sole points and report per-foot/minimum vertical
  clearance without changing any pose.
- Track stance slip as horizontal displacement of a contacting sole centroid
  between consecutive accepted frames, reset on contact release, scene reset,
  or controlled error.
- Append bank-neutral columns to the CSV; preserve the existing immutable
  prefix and keep `ik_enabled=0`.

- [ ] **Step 1: Install sole diagnostic component tests**

Cover exact local-to-world transforms, sloped surfaces, mixed-height sole
corners, minimum selection, stance-slip accumulation/reset, non-finite input,
out-of-domain queries, and transactional output preservation on failure.

```bash
g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_sole_diagnostics.cpp \
  -o /tmp/test-g1-sole-diagnostics \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected RED: compilation fails because `g1_sole_diagnostics.h` is missing.

- [ ] **Step 2: Install log and summary tests**

Add fixtures that require eight sole clearances, per-foot minima, global sole
minimum, left/right stance slip, and contact-reset flags. The summary tool must
produce deterministic JSON metrics for source-family counts, searches,
transitions, selected-cost percentiles, rendered joint clearance, sole
clearance, and stance slip.

```bash
python3 -m unittest \
  tests.python.test_runtime_log \
  tests.python.test_motion_quality_summary -v
```

Expected RED: required columns and summarizer do not exist.

- [ ] **Step 3: Implement diagnostics only**

Integrate after the stable controller has produced final global pose arrays and
before CSV publication. Do not call any IK or mutate global/local pose,
support, matching, command, or terrain state. Keep the mesh pose path
unchanged.

Run the C++ and Python focused tests. Expected GREEN: all pass.

- [ ] **Step 4: Build and record the old-pack baseline**

Build `/tmp/g1-motion-banks-baseline/controller-g1-baseline` with the standard
Raylib command. Run exact 100-frame `stairs-shallow/flat-positive-z` forward
and `stairs-shallow/flat-positive-x` heading-preserving lateral routes. Run
800-frame `stairs-standard/ascent-landing-descent`,
`ramp-10-up-down/up-landing-down`, the same stairs route with a
`positive-x` heading override, and
`mixed-multilevel/tangent-level-boundary` using the current external v2 pack.
Store raw CSV only in `/tmp/g1-motion-banks-baseline/`.

Use `resources/summarize_g1_motion_quality.py` to create the compact checked-in
baseline JSON. Freeze comparative Phase-A thresholds before matcher tuning:
hard zero wrong-bank frames on flat; hard category correctness after terrain
activation; and predeclared improvements for lateral cost/churn, stance slip,
and sole penetration.

- [ ] **Step 5: Verify and commit**

```bash
python3 -m unittest \
  tests.python.test_runtime_log \
  tests.python.test_motion_quality_summary -v
/tmp/test-g1-sole-diagnostics
git diff --check
git add g1_sole_diagnostics.h motion_match_log.h \
  g1_runtime_diagnostics.h controller.cpp \
  resources/check_g1_runtime_log.py \
  resources/summarize_g1_motion_quality.py \
  tests/cpp/test_g1_sole_diagnostics.cpp \
  tests/python/test_runtime_log.py \
  tests/python/test_motion_quality_summary.py \
  tests/fixtures/g1_motion_quality_baseline.json
git commit -m "test: freeze pre-IK G1 motion quality baseline"
git push checkpoint g1-playable-mesh
```

---

### Task 4: Define the Twelve-Dimensional Terrain Descriptor and Classifier

**Files:**

- Modify: `resources/g1_terrain_builder/terrain.py`
- Modify: `resources/g1_terrain_builder/schema.py`
- Modify: `resources/g1_terrain_builder/database.py`
- Modify: `tests/python/test_terrain.py`
- Modify: `tests/python/test_schema.py`
- Modify: `tests/python/test_database_builder.py`

**Interfaces:**

- `TERRAIN_LONGITUDINAL_DISTANCES = (0.125, ..., 1.0)`.
- Four cross-track samples at 0.25, 0.50, 0.75, and 1.00 m using fixed
  left/right sole-corridor offsets around the travel path while retaining the
  independent heading.
- `sample_terrain_descriptor(...) -> float32[12]`.
- `classify_terrain_profile(...)` returns family, signed elevation mode,
  confidence, and diagnostic step/grade values from dense height/normal
  samples.

- [ ] **Step 1: Install analytic descriptor tests**

Use exact synthetic flat, 10-degree longitudinal slope, 10-degree cross-slope,
single 0.12 m curb, three 0.12-by-0.32 m stairs, and mirrored descent surfaces.
Require exact shape/dtype, finite values, translation invariance, correct
family/elevation mode, repeatability at float32 boundaries, and no heading
write. Add alias cases where the old four samples match but slope and stairs
must classify differently.

```bash
python3 -m unittest \
  tests.python.test_terrain \
  tests.python.test_schema \
  tests.python.test_database_builder -v
```

Expected RED: the twelve-dimensional API and schema do not exist.

- [ ] **Step 2: Implement descriptor and classifier**

Share interpolation and surface-normal semantics with G1HF/v2. Classification
thresholds are named constants with unit-bearing comments and are tested on
both signs. Reject non-finite, degenerate, or out-of-domain inputs rather than
returning `flat`.

Update `HoldenClip` and `ArtifactSet` terrain shape contracts from four to
twelve. Do not change artifact file schemas yet.

- [ ] **Step 3: Run focused GREEN tests and commit**

```bash
python3 -m unittest \
  tests.python.test_terrain \
  tests.python.test_schema \
  tests.python.test_database_builder -v
git diff --check
git add resources/g1_terrain_builder/terrain.py \
  resources/g1_terrain_builder/schema.py \
  resources/g1_terrain_builder/database.py \
  tests/python/test_terrain.py tests/python/test_schema.py \
  tests/python/test_database_builder.py
git commit -m "feat: define terrain profile and family classifier"
git push checkpoint g1-playable-mesh
```

---

### Task 5: Build Deterministic Terrain-Bank and Direction Indexes

**Files:**

- Create: `resources/g1_terrain_builder/motion_index.py`
- Create: `tests/python/test_motion_index.py`
- Modify: `resources/g1_terrain_builder/schema.py`
- Modify: `tests/python/test_schema.py`

**Interfaces:**

- Terrain families are exactly `flat`, `curb`, `slope`, `stair`.
- Direction-mask bits are exactly: idle, forward, forward-right, right,
  back-right, backward, back-left, left, forward-left.
- Moving bins overlap at their angular boundaries. Lateral and diagonal masks
  require at most 15 degrees of one-second heading change; a turn is not a
  strafe.
- Speed masks contain overlapping low-speed and moving classes using the
  one-second root displacement contract.
- Each per-frame packed index row is four bytes:
  little-endian `uint16 direction_mask`, `uint8 speed_mask`, and signed
  `int8 elevation_mode`.
- Source terrain-bank ranges are explicit manifest records; per-frame masks do
  not duplicate pose data.

- [ ] **Step 1: Install motion-index tests**

Cover all compass sectors, overlap boundaries, mirrored lateral motion,
stationary/start/stop frames, clip-end clamping, turning rejection, ascent and
descent, source-boundary isolation, exact packed bytes, invalid masks, and
complete frame coverage.

```bash
python3 -m unittest tests.python.test_motion_index -v
```

Expected RED: module missing.

- [ ] **Step 2: Implement index derivation and codec**

Create pure derivation functions plus `G1MI/v1` reader/writer helpers. Reader
size arithmetic must be overflow-safe and reject truncation, trailing bytes,
invalid bits, and elevation values outside -1/0/+1. Writers validate before
opening output paths.

- [ ] **Step 3: Run GREEN tests and commit**

```bash
python3 -m unittest \
  tests.python.test_motion_index tests.python.test_schema -v
git diff --check
git add resources/g1_terrain_builder/motion_index.py \
  resources/g1_terrain_builder/schema.py \
  tests/python/test_motion_index.py tests/python/test_schema.py
git commit -m "feat: index G1 motion by terrain and travel direction"
git push checkpoint g1-playable-mesh
```

---

### Task 6: Publish and Validate the Version-3 Motion Pack Schema

**Files:**

- Modify: `resources/g1_terrain_builder/artifacts.py`
- Modify: `resources/validate_g1_terrain_database.py`
- Modify: `tests/python/test_artifacts.py`
- Modify: `tests/python/test_validator.py`
- Modify: `tests/python/test_build_cli.py`

**Interfaces:**

- Manifest schema: `g1-terrain-artifacts/v3`.
- Feature dimensions: 39; terrain dimensions: 12; support dimensions: 3.
- Terrain sidecar: `G1TF/v2`, version 2, twelve float32 columns.
- Motion index sidecar: `G1MI/v1`, version 1, one four-byte row per database
  frame.
- Source records add exact `terrain_family`.
- Manifest adds authenticated `motion_index` and `motion_banks` descriptors.
- Expected staged tree includes `motion_index.bin` and rejects any extra node.

- [ ] **Step 1: Install schema/publication RED tests**

Extend byte-codec, manifest-key, staged-tree, transactional-publication, hash,
source-range, bank-coverage, and validator tests. Require every source exactly
once in one bank, every frame exactly once in one source, and every frame at
least one valid speed/direction mask.

```bash
python3 -m unittest \
  tests.python.test_artifacts \
  tests.python.test_validator \
  tests.python.test_build_cli -v
```

Expected RED: v3 fields/sidecar are missing and old four-dimensional contracts
reject the candidate.

- [ ] **Step 2: Implement v3 publication and validation**

Keep atomic staging/rollback guarantees unchanged. Stream hashes and large
files; never call `read_bytes()` on the multi-gigabyte database. Preserve
G1SP/v1 and scene schemas. Validation must accept 15,838 sources and
3,976,682 frames without fixed old-corpus literals.

- [ ] **Step 3: Run focused GREEN tests and commit**

```bash
python3 -m unittest \
  tests.python.test_artifacts \
  tests.python.test_validator \
  tests.python.test_build_cli -v
git diff --check
git add resources/g1_terrain_builder/artifacts.py \
  resources/validate_g1_terrain_database.py \
  tests/python/test_artifacts.py tests/python/test_validator.py \
  tests/python/test_build_cli.py
git commit -m "feat: publish terrain-aware G1 motion pack v3"
git push checkpoint g1-playable-mesh
```

---

### Task 7: Generalize the Builder Across Flat, Curb, Slope, and Stair Sources

**Files:**

- Modify: `resources/build_g1_terrain_database.py`
- Modify: `resources/g1_terrain_builder/sources.py`
- Modify: `resources/g1_terrain_builder/database.py`
- Modify: `resources/g1_terrain_builder/motion_index.py`
- Modify: `resources/validate_g1_terrain_database.py`
- Modify: `tests/python/test_build_cli.py`
- Modify: `tests/python/test_validator.py`
- Modify: `tests/python/test_sources.py`

**Interfaces:**

- Builder consumes the pinned acquisition manifest/inventory and explicit
  family roots instead of one hard-coded curb glob.
- Ordering is deterministic: Takara flat first, then curb, slope, stair_p1,
  stair_p2, with lexical paths inside each family.
- Expected complete totals are 15,838 sources and 3,976,682 25 Hz frames.
- `--source-limit-per-family` exists only for smoke fixtures and marks the
  resulting candidate diagnostic/nonpublishable as a production pack.
- Every real source uses its authenticated surface, twelve-dimensional
  descriptor, support rows, contacts, and four-byte motion index.

- [ ] **Step 1: Install multi-family builder tests**

Use tiny fake flat/curb/slope/stair sources to prove exact ordering, identity,
terrain family, surface pairing, source-boundary derivatives/contacts,
descriptor rows, index rows, duplicate rejection, missing modality failure,
and streaming progress. Prove a stair robot cannot use a curb or synthetic
surface.

```bash
python3 -m unittest \
  tests.python.test_build_cli \
  tests.python.test_validator \
  tests.python.test_sources -v
```

Expected RED: builder accepts only one curb glob and emits v2 data.

- [ ] **Step 2: Implement the multi-family build path**

Keep one `G1Kinematics` instance. Release each source/object/terrain after its
converted clip and metadata are retained. Validate per-source conversion
reports during build. Full candidate validation checks all aggregate bytes and
deterministically rebuilds a lexical/coverage sample from every family rather
than redundantly loading every pickle a second time.

- [ ] **Step 3: Build a bounded smoke pack**

```bash
python3 resources/build_g1_terrain_database.py \
  --acquisition-manifest resources/grail_terrain_inputs.json \
  --acquisition-inventory /home/ubuntu/datasets/GRAIL/g1_mm_inventory.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --source-limit-per-family 2 \
  --output /tmp/g1-terrain-banks-smoke
python3 resources/validate_g1_terrain_database.py \
  /tmp/g1-terrain-banks-smoke
```

Expected: one flat source plus two sources from each terrain family, v3
manifest, 39/12/3 dimensions, valid motion index, and no publication residue.

- [ ] **Step 4: Run GREEN tests and commit**

```bash
python3 -m unittest \
  tests.python.test_build_cli \
  tests.python.test_validator \
  tests.python.test_sources -v
git diff --check
git add resources/build_g1_terrain_database.py \
  resources/g1_terrain_builder/sources.py \
  resources/g1_terrain_builder/database.py \
  resources/g1_terrain_builder/motion_index.py \
  resources/validate_g1_terrain_database.py \
  tests/python/test_build_cli.py tests/python/test_validator.py \
  tests/python/test_sources.py
git commit -m "feat: build complete G1 terrain motion banks"
git push checkpoint g1-playable-mesh
```

---

### Task 8: Load Version-3 Banks and Motion Indexes in C++

**Files:**

- Create: `motion_index_runtime.h`
- Create: `tests/cpp/test_motion_index_runtime.cpp`
- Modify: `scene_runtime.h`
- Modify: `tests/cpp/test_scene_runtime.cpp`

**Interfaces:**

- Strictly load `g1-terrain-artifacts/v3`, `G1TF/v2`/12, and
  `G1MI/v1`/four-byte rows.
- Parse dynamic source counts and exact `terrain_family` values.
- Validate bank source ranges against database range arrays and index row count
  against database frames.
- Build immutable runtime source-bank lists and per-bound direction/speed
  presence masks without duplicating poses.

- [ ] **Step 1: Install C++ loader tests**

Test exact valid bytes plus truncation, trailing data, integer overflow,
unknown family, duplicate/omitted source bank, bad direction bits, bad speed
bits, invalid elevation, row-count mismatch, and transactional preservation of
the previous loaded pack on failure.

```bash
g++ -std=c++17 -O2 -I. tests/cpp/test_motion_index_runtime.cpp \
  -o /tmp/test-motion-index-runtime
g++ -std=c++17 -O2 -I. tests/cpp/test_scene_runtime.cpp \
  -o /tmp/test-scene-runtime
```

Expected RED: missing runtime index type and v3 manifest support.

- [ ] **Step 2: Implement strict loaders and derived masks**

Use bounded arithmetic and owned vectors. No pointer into temporary JSON or
file buffers may survive loading. Preserve the prior output object on every
failure.

- [ ] **Step 3: Run GREEN tests and commit**

```bash
/tmp/test-motion-index-runtime
/tmp/test-scene-runtime
git diff --check
git add motion_index_runtime.h scene_runtime.h \
  tests/cpp/test_motion_index_runtime.cpp tests/cpp/test_scene_runtime.cpp
git commit -m "feat: load G1 motion banks and frame indexes"
git push checkpoint g1-playable-mesh
```

---

### Task 9: Implement Runtime Terrain Descriptor and Bank Hysteresis

**Files:**

- Create: `motion_bank_runtime.h`
- Create: `tests/cpp/test_motion_bank_runtime.cpp`
- Modify: `terrain_runtime.h`
- Modify: `tests/cpp/test_terrain_runtime.cpp`

**Interfaces:**

- Runtime descriptor produces the same twelve float32 values as the Python
  builder for locked flat/slope/cross-slope/curb/stair fixtures.
- Runtime classifier produces exact family/elevation/confidence parity with
  Python.
- `motion_bank_state` owns current/pending family, pending-frame count, and
  transition reason. Fixed hysteresis retains the current family through
  ambiguous edge samples but enters a confident new family before its first
  required motion event.
- Invalid/out-of-domain input has an explicit failure status and never returns
  flat.

- [ ] **Step 1: Export locked Python oracle fixtures**

Add byte-exact descriptor/classification cases to the C++ test. The test data
is generated once from Task 4 APIs and committed as literals; production C++
does not invoke Python.

- [ ] **Step 2: Observe RED and implement parity**

```bash
g++ -std=c++17 -O2 -I. tests/cpp/test_motion_bank_runtime.cpp \
  -o /tmp/test-motion-bank-runtime
```

Expected RED: header/API missing. Implement the descriptor, classifier, and
hysteresis with no heap allocation per frame and no command-heading writes.

- [ ] **Step 3: Run GREEN tests and commit**

```bash
/tmp/test-motion-bank-runtime
g++ -std=c++17 -O2 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test-terrain-runtime
/tmp/test-terrain-runtime
git diff --check
git add motion_bank_runtime.h terrain_runtime.h \
  tests/cpp/test_motion_bank_runtime.cpp tests/cpp/test_terrain_runtime.cpp
git commit -m "feat: classify runtime terrain motion banks"
git push checkpoint g1-playable-mesh
```

---

### Task 10: Add Indexed Bank- and Direction-Compatible Database Search

**Files:**

- Modify: `database.h`
- Modify: `motion_index_runtime.h`
- Create: `tests/cpp/test_indexed_database_search.cpp`
- Modify: `tests/cpp/test_terrain_database.cpp`

**Interfaces:**

- `database_search_indexed(...)` accepts compatible source ranges,
  direction mask, speed mask, elevation mode, current frame, and transition
  cost.
- It skips large/small bounds with no compatible mask and filters every frame
  before evaluating its feature distance.
- If the incumbent source/frame is incompatible, it cannot win by retention;
  the search either returns a compatible candidate or an explicit empty-set
  status.
- Clip-end exclusion and `ignore_surrounding` use original animation ranges,
  not direction-mask runs.
- Search results match brute-force eligible-frame oracles exactly.

- [ ] **Step 1: Install indexed-search tests**

Cover flat versus curb zero-feature ties, slope versus stair ties, left/right
lateral masks, speed overlap, elevation sign, incompatible incumbent,
transition cost, clip ends, empty set, bound-mask skipping, and randomized
indexed-versus-brute-force parity.

```bash
g++ -std=c++17 -O2 -I. tests/cpp/test_indexed_database_search.cpp \
  -o /tmp/test-indexed-database-search
```

Expected RED: indexed API missing.

- [ ] **Step 2: Implement the minimum indexed search path**

Reuse existing normalized feature distance and bounds. Do not fork pose cost
math or change legacy search behavior outside the new API. Record search
eligibility/rejection diagnostics for later logging.

- [ ] **Step 3: Run GREEN and performance micro-gates**

```bash
/tmp/test-indexed-database-search
g++ -std=c++17 -O2 -I. tests/cpp/test_terrain_database.cpp \
  -o /tmp/test-terrain-database
/tmp/test-terrain-database
```

The synthetic four-million-row eligibility benchmark must remain inside the
25 Hz search budget on this host and must examine fewer candidates than global
search for fixed flat/lateral/stair queries.

- [ ] **Step 4: Commit and push**

```bash
git add database.h motion_index_runtime.h \
  tests/cpp/test_indexed_database_search.cpp \
  tests/cpp/test_terrain_database.cpp
git commit -m "feat: search compatible G1 motion bank frames"
git push checkpoint g1-playable-mesh
```

---

### Task 11: Integrate the 39-D Query and Banked Search into the Stable Controller

**Files:**

- Modify: `controller.cpp`
- Modify: `database.h`
- Modify: `scene_runtime.h`
- Modify: `motion_match_log.h`
- Modify: `g1_runtime_diagnostics.h`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `resources/summarize_g1_motion_quality.py`
- Modify: `tests/python/test_runtime_log.py`
- Create: `tests/python/test_g1_banked_controller_integration.py`

**Interfaces:**

- Build exactly 39 query values: existing 27 pose/trajectory values plus the
  twelve-value runtime terrain descriptor.
- Expand finite-query and bit-snapshot contracts from 31 to 39 values. CSV
  query snapshots contain exactly 312 lowercase hexadecimal characters, and
  terrain diagnostic columns contain all twelve descriptor values with their
  longitudinal/cross-track sample provenance.
- Choose terrain family through `motion_bank_state`, direction/speed masks from
  predicted travel relative to independently predicted heading, and elevation
  sign from the classifier.
- Search only through `database_search_indexed`.
- Log requested/active bank, direction mask, speed mask, elevation mode,
  confidence, bank transition/reason, eligible frames/bounds, and empty-set
  status.
- Default effective terrain weight is 4.0 for v3 live/route/terrain runs and
  is recorded exactly; an explicit valid environment/UI override remains
  supported.
- Flat lookahead cannot select curb/slope/stair. Repeated steps cannot select
  slope/curb after stair-bank activation. IK stays off.

- [ ] **Step 1: Install source-contract and log RED tests**

Require one 39-D query construction, one indexed-search call, no legacy global
search in the G1 path, preserved classic UI/mesh/control strings, no
heading-from-velocity assignment, and no IK enable. Extend log schema and
checker fixtures for the 312-character query snapshot, twelve terrain values,
sample provenance, exact new bank columns, and state transitions.

```bash
python3 -m unittest \
  tests.python.test_g1_banked_controller_integration \
  tests.python.test_runtime_log \
  tests.python.test_motion_quality_summary -v
```

Expected RED: controller remains 31-D/global and columns are absent.

- [ ] **Step 2: Implement controller integration**

Keep all new bank decisions local until the query and compatible search set are
valid. On empty compatible set, request the existing safe-stop and preserve
heading/last accepted pose; never fall back to global search. Rebuild matching
features at 39 dimensions with the requested terrain weight.

- [ ] **Step 3: Build and run smoke-pack finite tests**

```bash
mkdir -p /tmp/g1-motion-banks-controller
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/g1-motion-banks-controller/controller-g1-banks \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
G1_TERRAIN_DIR=/tmp/g1-terrain-banks-smoke \
MM_TERRAIN_WEIGHT=4 MM_TERRAIN_SCENE=stairs-shallow \
MM_TEST_MODE=route MM_TEST_ROUTE=flat-positive-z \
MM_TEST_FRAMES=100 MM_LOG=/tmp/g1-motion-banks-controller/flat.csv \
/tmp/g1-motion-banks-controller/controller-g1-banks
```

Expected: valid v3 pack, 39-D query snapshots, flat bank only, finite pose,
`ik_enabled=0`, and no controlled error.

- [ ] **Step 4: Run focused GREEN tests, commit, and push**

```bash
python3 -m unittest \
  tests.python.test_g1_banked_controller_integration \
  tests.python.test_runtime_log \
  tests.python.test_motion_quality_summary -v
git diff --check
git add controller.cpp database.h scene_runtime.h motion_match_log.h \
  g1_runtime_diagnostics.h resources/check_g1_runtime_log.py \
  resources/summarize_g1_motion_quality.py \
  tests/python/test_runtime_log.py \
  tests/python/test_g1_banked_controller_integration.py
git commit -m "feat: drive stable G1 controller with motion banks"
git push checkpoint g1-playable-mesh
```

---

### Task 12: Build the Full Pack and Pass Pre-IK Motion-Quality Gates

**Files:**

- Create: `resources/check_g1_motion_quality.py`
- Create: `tests/python/test_g1_motion_quality.py`
- Modify: `tests/fixtures/g1_motion_quality_baseline.json` only if Task 3
  recorded values were serialized incorrectly; do not tune thresholds here.
- External create:
  `/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate`
- External create: `/tmp/g1-motion-banks-phase-a/*`

**Interfaces:**

- Full candidate totals: 15,838 sources and 3,976,682 frames.
- Candidate pack is published separately and fully authenticated.
- Quality checker consumes the Task 3 baseline and candidate logs; it cannot
  change thresholds based on candidate results.
- Hard gates: zero wrong-family selections on flat; category-correct slope and
  stair selection after activation; exact independent-heading oracle;
  `ik_enabled=0`; complete routes; finite 39-D queries; no unnoticed sole
  penetration.
- Comparative gates: lateral selected cost/transition churn, stance slip, and
  sole penetration meet the predeclared Task 3 improvements.

- [ ] **Step 1: Install quality-checker tests**

Use compact synthetic log fixtures to prove each hard and comparative failure
is rejected independently, thresholds come only from the baseline file, and a
candidate cannot pass merely by route completion.

```bash
python3 -m unittest tests.python.test_g1_motion_quality -v
```

Expected RED: checker missing.

- [ ] **Step 2: Implement checker and run GREEN unit tests**

The report is sorted JSON with per-route bank correctness, cost/churn, joint
and sole penetration, stance slip, route completion, heading invariance, and
IK state. A nonpassing route exits nonzero.

- [ ] **Step 3: Build and validate the complete candidate pack**

```bash
python3 resources/grail_terrain_acquisition.py verify \
  --manifest resources/grail_terrain_inputs.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --modalities robot objects object_usd \
  --inventory /home/ubuntu/datasets/GRAIL/g1_mm_inventory.json
python3 resources/build_g1_terrain_database.py \
  --acquisition-manifest resources/grail_terrain_inputs.json \
  --acquisition-inventory /home/ubuntu/datasets/GRAIL/g1_mm_inventory.json \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --output /home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate
python3 resources/validate_g1_terrain_database.py \
  /home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate
```

Do not run this in a shell that overwrites the accepted v2 pack. Record elapsed
time, peak RSS, exact hashes, counts, and validation report in the task report.

- [ ] **Step 4: Run the deterministic route matrix**

Build the release candidate binary under `/tmp/g1-motion-banks-phase-a/`.
Generate logs for:

- flat forward acceleration;
- mirrored flat lateral and both diagonals;
- `ramp-10-up-down/up-landing-down` forward/lateral/diagonal;
- `cross-slope-10` forward/lateral;
- `stairs-standard/ascent-landing-descent` forward and mirrored lateral;
- `stairs-unseen-variable/ascent-landing-descent` forward;
- mixed tangential/partial-support edges; and
- scene switching and cleanup.

Use 800 frames for terrain routes unless their declared route needs more. Run:

```bash
python3 resources/check_g1_motion_quality.py \
  --baseline tests/fixtures/g1_motion_quality_baseline.json \
  --logs /tmp/g1-motion-banks-phase-a/logs \
  --output /tmp/g1-motion-banks-phase-a/quality-report.json
```

Expected: every hard and comparative gate passes. If a gate fails, diagnose
data coverage, classifier, or search; do not add IK, heading rotation, or a
global-search fallback.

- [ ] **Step 5: Run performance and live-input gates**

Prove 25 Hz route execution without missed fixed steps and exercise real
forward, backward, left/right lateral, both diagonals, strafe, camera, scene
switch, mesh toggle, and bone toggle input on a disposable process. The
process must remain alive, own one X11 client, keep the classic UI, and unload
each model exactly once on normal exit.

- [ ] **Step 6: Commit and push quality infrastructure**

```bash
python3 -m unittest tests.python.test_g1_motion_quality -v
git diff --check
git add resources/check_g1_motion_quality.py \
  tests/python/test_g1_motion_quality.py
git commit -m "test: certify pre-IK G1 terrain motion quality"
git push checkpoint g1-playable-mesh
```

Generated pack/log/report files remain external. Record their authenticated
paths and hashes in the progress ledger/task report.

---

### Task 13: Record Visual Evidence and Launch the Pre-IK Mesh Visualizer

**Files:**

- Create: `resources/record_g1_motion_verification.sh`
- Create: `tests/python/test_g1_verification_recipe.py`
- Modify:
  `docs/superpowers/plans/2026-07-18-g1-terrain-aware-motion-banks.md`
  only to check completed boxes and append exact final evidence.
- External create/update:
  `/home/ubuntu/projects/motion-matching-verifications/*`

**Interfaces:**

- Produce short synchronized old/new clips for flat acceleration, lateral flat
  travel, stairs, slopes, and tangential/partial support.
- Each new clip visibly overlays active source family, direction mask, terrain
  classification, sole clearance/slip, and `IK OFF`.
- Keep the classic UI and corrected mesh.
- Replace PID `1680537` only after all Task 12 evidence is rechecked from the
  exact pushed commit and binary.

- [ ] **Step 1: Install verification-recipe tests**

Statically require strict shell mode, bounded recording duration, exact binary
and pack paths, separate old/new outputs, no process wildcard kills, no write
inside the Git worktree, ffmpeg error propagation, and explicit `IK OFF`
overlay evidence.

```bash
python3 -m unittest tests.python.test_g1_verification_recipe -v
```

Expected RED: script missing.

- [ ] **Step 2: Implement and test the recorder**

Record on a disposable X11 process without touching PID `1680537`. Verify each
MP4 with `ffprobe`, enforce a bounded directory size, and leave only the five
review clips plus a small manifest containing commit/binary/pack hashes.

- [ ] **Step 3: Re-run final finite and live gates from pushed HEAD**

Rebuild the exact final binary, rerun the quality checker, verify Git
worktree/remote equality, confirm the candidate pack hashes, and confirm the
old PID still names the old exact executable with empty stderr.

- [ ] **Step 4: Commit and push the reproducible launch checkpoint**

```bash
git add resources/record_g1_motion_verification.sh \
  tests/python/test_g1_verification_recipe.py \
  docs/superpowers/plans/2026-07-18-g1-terrain-aware-motion-banks.md
git commit -m "docs: certify pre-IK terrain motion visualizer"
git push checkpoint g1-playable-mesh
```

- [ ] **Step 5: Replace only the accepted running visualizer**

Verify PID `1680537` again, terminate that exact process cleanly, and wait for
its X11 client to disappear. Launch exactly one new detached process with:

```bash
DISPLAY=:1 \
G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate \
setsid -f stdbuf -oL \
  /tmp/g1-motion-banks-phase-a/controller-g1-banks \
  </dev/null \
  >/tmp/g1-motion-banks-phase-a/visualizer.stdout \
  2>/tmp/g1-motion-banks-phase-a/visualizer.stderr
```

From a separate shell, prove exactly one matching process, exact `/proc/PID/exe`,
one real X11 client with matching `_NET_WM_PID`, classic UI text, mesh-load
evidence, empty stderr, and continued life after real movement/camera input.

- [ ] **Step 6: Stop Phase A for user visual inspection**

Report the new PID, commit, binary, candidate pack, quality report, and
verification folder. Explicitly state that IK is off and no IK task begins
until the user has inspected this visualizer and requested Phase C.

---

## Final Phase-A Verification

Before claiming Phase A ready:

1. every Task 1-13 report exists and every task review is clean;
2. `.superpowers/sdd/progress.md` names exact commit ranges;
3. local HEAD equals `checkpoint/g1-playable-mesh` and the worktree is clean;
4. all focused Python/C++ tests pass;
5. the 306-test baseline passes except for the one exact pre-existing IK
   ownership failure;
6. full candidate pack and quality-report hashes are recorded;
7. the new visualizer is the sole matching process and survives real input;
8. the five disposable verification clips are available to the user; and
9. every runtime log and overlay reports `IK OFF` / `ik_enabled=0`.

Then dispatch the broad whole-branch reviewer using the merge-base-to-HEAD
review package. Fix and re-review all Critical/Important findings. Do not merge
or begin IK. Hand the running pre-IK visualizer to the user for inspection.
