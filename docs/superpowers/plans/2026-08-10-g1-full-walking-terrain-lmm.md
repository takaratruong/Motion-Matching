# Full Walking-Only Terrain LMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and visualize one steerable 60 Hz Unitree G1
terrain motion matcher from all admitted walking-only PFNN, GRAIL, and Takara
data.

**Architecture:** A new full-walking artifact family authenticates the frozen
25 Hz broad bank, derives its GRAIL ranges at 60 Hz, and appends source-native
missing slopes, Takara, and full PFNN walking/mirror lanes. A deterministic
split ledger and train-only normalization produce an immutable full corpus for
the existing compressor/decompressor trainer. The runtime replaces
command-owned translation with one motion-derived `SE(2)` transform and adds
contact-aware exact scoring, while a separate reducer compares coverage and
foot slip against the frozen overnight baseline.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch/CUDA, MuJoCo, existing G1
terrain-builder codecs, content-addressed canonical JSON/SHA-256 publications,
`pytest`, Ruff, and memory-mapped uncompressed NumPy arrays.

## Global Constraints

- The binding design is
  `docs/superpowers/specs/2026-08-10-g1-full-walking-terrain-lmm-design.md`.
- Admit only stand, walk, stand/walk blends, and terrain transitions. Exclude
  jog, run, crouch, crawl, jump-only, sitting, pickup, and manipulation.
- Include all official PFNN mirrors, all 1,769 GRAIL curb clips, all 1,880
  GRAIL slope clips, all 6,094 stair-p1 clips, all 6,094 stair-p2 clips, and
  the complete Takara walk.
- The canonical rate is exactly 60 Hz with horizons `(20, 40, 60)` and the
  canonical 31-bone G1 hierarchy/signature from
  `resources/g1_terrain_builder/schema.py`.
- Resample positions/vectors by timestamp and rotations by sign-continuous
  WXYZ quaternion SLERP. Recompute velocities, angular velocities, contacts,
  support-relative placement, quality, features, and successor validity inside
  each derived range.
- No derivative, contact filter, horizon, training window, successor, mirror,
  terrain-fit cycle, or split assignment crosses a range boundary.
- Split connected components are the transitive closure of canonical source,
  terrain, and mirror identities. Selection uses validation only; test is read
  once after checkpoint freeze; all-row refit binds both receipts.
- The new model ABI is 60 Hz: `dt=1/60`, 31-D matching features, 908-D
  compressor input, `31 + latent` decoder input, and 458-D target.
- Runtime planar travel comes only from the selected range's local root/yaw
  deltas composed into one world `SE(2)` transform. Command input is a desired
  supported speed and heading, never a separately integrated translation.
- Formal acceptance requires zero candidate exhaustion, source fallback, joint
  clamp, native-limit violation, nonfinite state, and out-of-range successor.
- Large artifacts live under `sonic/runs/g1-full-walking-terrain-lmm/` and are
  never committed to Git.
- Do not stage or overwrite the concurrent edits in
  `resources/validate_g1_terrain_database.py`,
  `sonic/python/mm_sonic/terrain_oracle/audit.py`,
  `tests/python/test_oracle_audit.py`, or
  `tests/python/test_validator.py`.
- All output directories are immutable and must be absent before publication.
  Use staging, file fsync, directory fsync, atomic rename, and canonical JSON.
- Python test prefix:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python
  ```

- CUDA training prefix:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /home/ubuntu/miniconda3/envs/foundation_stereo/bin/python
  ```

- Deadline control: the verified corpus/index must publish by 11:30 PDT to
  leave the measured low-bound window for selection, one-time test, all-row
  refit, latent export, and 10,000-frame evidence. If it has not published by
  11:30, report the exact blocking lane and revised estimate immediately, keep
  all unaffected work running, and do not weaken or relabel a gate.

## Dependency and Parallelism Graph

```text
Task 1: contracts, split ledger, lane publication
   |------------------|-------------------|
Task 2: GRAIL/Takara  Task 3: full PFNN  Task 6: runtime SE(2)
   |------------------|                   |
Task 4: full corpus/index                 Task 7: slip/evidence reducer
   |
Task 5: selection/test/refit
   |--------------------------------------|
Task 8: two builds + training + 10k-frame formal evaluation + viewer
```

Tasks 2, 3, and 6 run in isolated linked worktrees from the exact reviewed
Task 1 commit. Task 7 may start after Task 6 interfaces freeze. Integrate
reviewed commits serially into the main feature branch; never share Git index
operations among concurrent workers.

---

### Task 1: Freeze inventory, split ledger, contracts, and immutable lanes

**Files:**
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_contracts.py`
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_inventory.py`
- Create: `tests/python/test_full_walking_terrain_lmm_contracts.py`
- Create: `tests/python/test_full_walking_terrain_lmm_inventory.py`
- Modify: `sonic/python/mm_sonic/__init__.py`

**Interfaces:**
- Consumes: strict bank manifest, PFNN source table/sidecars, GRAIL category
  directories, Takara descriptors, `ArtifactSet`, canonical JSON/SHA helpers,
  and canonical G1 skeleton constants.
- Produces: `SourceRecord`, `FullWalkingInventory`, `RangeRecord`,
  `LaneArtifact`, `SplitAssignment`, `build_inventory`,
  `connected_split_groups`, `build_split_ledger`,
  `publish_lane_exclusive`, and `load_lane` for Tasks 2-4.

- [ ] **Step 1: Write contract and split REDs**

  Add tests that construct four tiny sources/ranges and assert transitive grouping,
  mirror co-location, deterministic stratified train/validation/test
  assignments, empty-stratum rejection, exact schema/key rejection, immutable
  output refusal, member tamper rejection, and byte-identical repeated fixture
  publication. Add a real metadata-only inventory test requiring exactly 80
  PFNN identities, 15,837 GRAIL identities, one Takara identity, and no sitting,
  pickup, jog, run, crouch, crawl, or jump-only trainable identity. The public
  records are:

  ```python
  JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

  @dataclass(frozen=True)
  class SourceRecord:
      source_id: str
      canonical_source_id: str
      terrain_id: str
      mirror_of: str | None
      family: Literal["flat", "curb", "slope", "stair"]
      authority: Mapping[str, JSONValue]

  @dataclass(frozen=True)
  class FullWalkingInventory:
      build_id: str
      sources: tuple[SourceRecord, ...]
      manifest_sha256: str

  @dataclass(frozen=True)
  class SplitAssignment:
      split_group_id: str
      split: Literal["train", "validation", "test"]
      source_ids: tuple[str, ...]

  @dataclass(frozen=True)
  class RangeRecord:
      range_id: str
      canonical_source_id: str
      terrain_id: str
      mirror_of: str | None
      family: Literal["flat", "curb", "slope", "stair"]
      split_group_id: str
      split: Literal["train", "validation", "test"]
      start: int
      stop: int
      quality: Literal["clean", "usable", "quarantined"]
      authority: Mapping[str, JSONValue]

  @dataclass(frozen=True)
  class LaneArtifact:
      root: Path
      manifest_sha256: str
      artifacts: ArtifactSet
      terrain_grid: np.ndarray       # float32 [rows, 36]
      ranges: tuple[RangeRecord, ...]

  def build_inventory(
      *, bank: Path, pfnn_root: Path, grail_root: Path, takara: Path
  ) -> FullWalkingInventory: ...

  def build_split_ledger(
      inventory: FullWalkingInventory, *, build_id: str, seed: int
  ) -> tuple[SplitAssignment, ...]: ...

  def publish_lane_exclusive(lane: LaneArtifact, output: Path) -> Path: ...

  def load_lane(
      root: Path, *, expected_manifest_sha256: str | None = None
  ) -> LaneArtifact: ...
  ```

- [ ] **Step 2: Run the focused tests and record the RED**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_contracts.py \
    tests/python/test_full_walking_terrain_lmm_inventory.py
  ```

  Expected: collection fails because
  `mm_sonic.full_walking_terrain_lmm_contracts` does not exist.

- [ ] **Step 3: Implement the exact contracts and split ledger**

  Publish canonical `g1-full-walking-terrain-lmm-inventory/v1` bytes before
  processing. The inventory authenticates every source descriptor but performs
  no motion/terrain conversion. Require exact family counts and exactly 15,918
  identities. Build and exclusively publish the connected-component split
  ledger from this inventory; every later lane binds both manifest digests.

  Use schema `g1-full-walking-terrain-lmm-lane/v1`. Validate exact field sets,
  little-endian C-contiguous arrays, finite values, canonical skeleton, full
  range coverage, and one record per range. Implement connected components
  with union-find over source, terrain, and mirror tokens. Sort components by
  canonical JSON identity and assign each terrain stratum by
  `sha256(build_id || seed || group_id)`, with an 80/10/10 target and at least
  one component in validation and test whenever the stratum has at least three
  components. Publish `database.bin`, `terrain_grid.npy`, `ranges.json`, and
  `manifest.json` through one staging directory and atomic rename.

- [ ] **Step 4: Make the focused tests GREEN and run common regressions**

  Run the focused command plus:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_database_builder.py \
    tests/python/test_hybrid_terrain_lmm_data.py \
    tests/python/test_full_walking_terrain_lmm_inventory.py
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit only Task 1 files**

  ```bash
  git add sonic/python/mm_sonic/__init__.py \
    sonic/python/mm_sonic/full_walking_terrain_lmm_contracts.py \
    sonic/python/mm_sonic/full_walking_terrain_lmm_inventory.py \
    tests/python/test_full_walking_terrain_lmm_contracts.py \
    tests/python/test_full_walking_terrain_lmm_inventory.py
  git commit -m "feat: add full walking corpus contracts"
  ```

---

### Task 2: Derive all GRAIL ranges and Takara at 60 Hz

**Files:**
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_grail.py`
- Create: `tests/python/test_full_walking_terrain_lmm_grail.py`

**Interfaces:**
- Consumes: Task 1 lane API; strict broad bank manifest SHA
  `997d9cb31da2ad611a181456f0d119723902163d1463b2b055f655490bb4783d`;
  `resample_map`, `resample_vectors`, `resample_quaternions_wxyz`,
  `refresh_lmm_clip_dynamics`, and authenticated source loaders.
- Produces: `authenticate_broad_bank`, `resample_inherited_range`,
  `build_inherited_grail_lane`, `build_missing_slope_lane`, and
  `build_takara_lane`.

  ```python
  def resample_inherited_range(
      parent: ArtifactSet, range_index: int, record: SourceRecord
  ) -> tuple[HoldenClip, RangeRecord]: ...

  def build_inherited_grail_lane(
      *, bank: Path, inventory: Path, split_ledger: Path, output: Path
  ) -> Path: ...

  def build_missing_slope_lane(
      *, grail_root: Path, inventory: Path, split_ledger: Path,
      g1_xml: Path, output: Path
  ) -> Path: ...

  def build_takara_lane(
      *, source: Path, remap: Path, inventory: Path, split_ledger: Path,
      g1_xml: Path, output: Path
  ) -> Path: ...
  ```

- [ ] **Step 1: Write resampling, authority, and inventory REDs**

  Tests must assert `250@25 -> 598@60`, `34_863@50 -> 41_835@60`, exact
  endpoints/source maps, antipodal-quaternion continuity, no cross-range
  velocity/contact leakage, exact inherited counts `(1769,1857,12188)`, exact
  set difference of 23 slope stems, and rejection of any changed bank member,
  Takara source, remap, raw slope robot, or USD digest.

- [ ] **Step 2: Run the focused RED**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_grail.py
  ```

  Expected: missing-module failure.

- [ ] **Step 3: Implement authenticated inherited range derivation**

  Require Task 1 inventory and split-ledger digests. Authenticate the broad
  bank's exact manifest and every listed payload before
  decoding. Exclude source range zero (Takara). For every remaining range,
  create one `HoldenClip`, resample positions/terrain/support linearly and
  local WXYZ rotations with sign-continuous SLERP, retain parent
  `(left,right,alpha)` rows in its receipt, then call
  `refresh_lmm_clip_dynamics(..., fps=60.0)`. Publish family lanes in sorted
  parent-range order. Do not treat the strict cache's 25 Hz normalized features
  as a 60 Hz authority.

- [ ] **Step 4: Implement the 23 missing slopes and Takara**

  Derive missing slope stems as the authenticated raw inventory minus parent
  bank slope identities. Preauthenticate robot, USD, object trajectory,
  reconstruction, and metadata bytes before existing decoders run. Use the
  existing slope placement/terrain adapter and canonical G1 conversion at
  60 Hz. Preauthenticate Takara
  `/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz` and
  `/home/ubuntu/projects/g1_mm/isaac_to_mj.npy`, then call the existing Takara
  loader and `convert_source_clip(..., target_fps=60.0)` with flat support.

- [ ] **Step 5: Make focused and source regressions GREEN**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_grail.py \
    tests/python/test_resample.py \
    tests/python/test_database_builder.py \
    tests/python/test_grail_terrain_source.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit only Task 2 files**

  ```bash
  git add sonic/python/mm_sonic/full_walking_terrain_lmm_grail.py \
    tests/python/test_full_walking_terrain_lmm_grail.py
  git commit -m "feat: derive full GRAIL and Takara walking lanes"
  ```

---

### Task 3: Process every PFNN walking/standing/transition source and mirror

**Files:**
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_pfnn.py`
- Create: `tests/python/test_full_walking_terrain_lmm_pfnn.py`

**Interfaces:**
- Consumes: Task 1 lane API; PFNN source table from pinned
  `/home/ubuntu/datasets/pfnn/pfnn/generate_database.py`; pinned GMR and
  retarget project; gait/phase/footstep sidecars; `patches.npz`; exact flat,
  rocky, beam, and jumpy terrain semantics.
- Produces: `discover_full_pfnn_inventory`, `admitted_pfnn_ranges`,
  `fit_pfnn_range_terrain`, `process_pfnn_identity`, and
  `build_full_pfnn_lane`.

  ```python
  def admitted_pfnn_ranges(
      gait: np.ndarray, continuity_breaks: np.ndarray,
      fit_intervals: tuple[tuple[int, int, str], ...]
  ) -> tuple[tuple[int, int, str], ...]: ...

  def process_pfnn_identity(
      record: SourceRecord, *, pfnn_root: Path, gmr_root: Path,
      retarget_root: Path, g1_xml: Path
  ) -> tuple[list[HoldenClip], list[RangeRecord], list[dict[str, object]]]: ...

  def build_full_pfnn_lane(
      *, inventory: Path, split_ledger: Path, pfnn_root: Path,
      gmr_root: Path, retarget_root: Path, g1_xml: Path,
      workers: int, output: Path
  ) -> Path: ...
  ```

- [ ] **Step 1: Write inventory and admission REDs**

  Freeze 80 processable identities (40 canonical plus 40 official mirrors),
  with the rest pose as metadata only. Test exact sidecar presence, canonical
  mirror links, the eight-column gait rule, 458 maximal admitted native runs,
  493,214 admitted 120 Hz rows, and exclusion counts:
  `jog=248368`, `run=77430`, `crouch=75726`, `jump=30598`,
  `crawl=6644`. Verify each excluded row terminates a range and no 60 Hz sample
  uses an interpolation bracket outside one admitted gait-plus-fit interval.

- [ ] **Step 2: Run the focused RED**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_pfnn.py
  ```

  Expected: missing-module failure.

- [ ] **Step 3: Implement authenticated inventory, gait ranges, and mirrors**

  Require the Task 1 inventory and split-ledger digests. Parse the reviewed
  upstream source table without importing its side-effecting
  module. Bind BVH/gait/phase/footstep hashes. Define admitted rows as finite
  sidecar rows whose allowed stand/walk mass is positive and at least the
  excluded gait mass; split on every excluded/malformed row and continuity
  break. Mirror identities retain their own bytes but share one canonical split
  group, and their three terrain-grid lanes are right/center/left reversed.

- [ ] **Step 4: Implement full retarget and exact family terrain semantics**

  Run `retarget_sample(..., grounding="source")` once per identity with pinned
  clean commits. Retain prepared BVH, native G1 NPZ, and exact receipt. Apply
  flat constant/no-contact-zero semantics to flat intervals; apply the upstream
  patch objective and residual fit separately for rocky, beam, and jumpy
  intervals. Reset/derive beam RNG deterministically from the frozen identity
  rather than worker order. Convert each final safe interval independently to
  60 Hz and publish explicit exclusion/fit/source receipts.

- [ ] **Step 5: Make PFNN-focused and legacy regressions GREEN**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_pfnn.py \
    tests/python/test_terrain_pfnn_sources.py \
    tests/python/test_terrain_pfnn_phase.py \
    tests/python/test_pfnn_terrain_transfer.py \
    tests/python/test_retarget_pfnn_bvh_g1.py \
    tests/python/test_hybrid_terrain_lmm_pfnn.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit only Task 3 files**

  ```bash
  git add sonic/python/mm_sonic/full_walking_terrain_lmm_pfnn.py \
    tests/python/test_full_walking_terrain_lmm_pfnn.py
  git commit -m "feat: add full PFNN walking terrain lane"
  ```

---

### Task 4: Assemble, verify, normalize, index, and reproduce the full corpus

**Files:**
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_corpus.py`
- Create: `tests/python/test_full_walking_terrain_lmm_corpus.py`

**Interfaces:**
- Consumes: Task 1 lane artifacts and split API; Task 2 GRAIL/Takara lanes;
  Task 3 PFNN lane.
- Produces: `FullWalkingCorpus`, `assemble_full_corpus`,
  `load_full_corpus`, `verify_full_corpus`, and the `build|verify|reproduce`
  CLI used by Tasks 5 and 8.

  ```python
  @dataclass(frozen=True)
  class FullWalkingCorpus:
      root: Path
      artifacts: ArtifactSet
      features: FeatureSet
      terrain_grid: np.ndarray
      family_ids: np.ndarray
      source_ids: np.ndarray          # int32 [rows]
      source_names: tuple[str, ...]
      eligible_mask: np.ndarray
      train_mask: np.ndarray
      validation_mask: np.ndarray
      test_mask: np.ndarray
      manifest_sha256: str
      fps: float = 60.0
      horizons: tuple[int, int, int] = (20, 40, 60)

  def assemble_full_corpus(lanes: Sequence[Path], output: Path) -> Path: ...
  def load_full_corpus(
      root: Path, *, expected_manifest_sha256: str | None = None
  ) -> FullWalkingCorpus: ...
  def verify_full_corpus(root: Path, receipt: Path) -> Path: ...
  ```

- [ ] **Step 1: Write assembly and fail-closed REDs**

  Build four synthetic lanes and assert deterministic range order, byte-exact
  inheritance of the prepublished connected split ledger, no empty required
  stratum, exact row/range/family/source
  counts, train-only normalization, horizon clamping, family/source-balanced
  views, and rejection of coherent manifest, split, feature, derivative,
  contact, source-map, or parent-authority mutations. Assert two fresh builds
  have identical payload and manifest digests.

- [ ] **Step 2: Run the focused RED**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_corpus.py
  ```

  Expected: missing-module failure.

- [ ] **Step 3: Implement virtual/memory-mapped full corpus assembly**

  Concatenate lane ranges in stable `(family, canonical_source_id, mirror,
  range_id)` order. Store the articulated database, 4-D terrain, 3-D support,
  36-D terrain grid, row/range/family/source IDs, split masks, and source maps
  as immutable lane-backed/memory-mapped members. Build raw 31-D features with
  60 Hz horizons, fit offset/scale only on `clean + usable` train rows, and
  transform every row. Publish and bind an authenticated scene pack containing
  `flat-standard`, `grail-curb-default`, `ramp-10-up-down`, and
  `stairs-standard`. The loader reopens and authenticates all lane manifests
  plus every combined member before returning ordinary `ArtifactSet` and
  `FeatureSet` interfaces to the existing trainer/runtime.

- [ ] **Step 4: Implement independent verification and reproduce command**

  Recompute range maps, dynamics, contacts, raw features, split components,
  ledger inheritance, normalization statistics, and all manifest counts without calling assembly
  helpers that produced them. `reproduce` builds a second temporary corpus,
  compares every relative path/size/SHA against the first, publishes
  `determinism.json`, then removes only its owned temporary duplicate after the
  receipt is durably synced.

- [ ] **Step 5: Make focused and aggregate data tests GREEN**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_contracts.py \
    tests/python/test_full_walking_terrain_lmm_grail.py \
    tests/python/test_full_walking_terrain_lmm_pfnn.py \
    tests/python/test_full_walking_terrain_lmm_corpus.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit only Task 4 files**

  ```bash
  git add sonic/python/mm_sonic/full_walking_terrain_lmm_corpus.py \
    tests/python/test_full_walking_terrain_lmm_corpus.py
  git commit -m "feat: publish verified full walking corpus"
  ```

---

### Task 5: Train selection, one-time test evaluation, and all-row refit

**Files:**
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_training.py`
- Create: `tests/python/test_full_walking_terrain_lmm_training.py`
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_training.py`

**Interfaces:**
- Consumes: Task 4 `FullWalkingCorpus`; existing compressor/decompressor and
  physical metric reducer.
- Produces: `train_selection`, `evaluate_frozen_test`, `train_all_rows`, and
  a CLI that publishes the new versioned 60 Hz model ABI.

  ```python
  def train_selection(
      corpus: FullWalkingCorpus, output: Path, *, config: HybridModelConfig
  ) -> Mapping[str, object]: ...

  def evaluate_frozen_test(
      corpus: FullWalkingCorpus, model: Path, output: Path, *, device: str
  ) -> Mapping[str, object]: ...

  def train_all_rows(
      corpus: FullWalkingCorpus, output: Path, *, selection_model: Path,
      test_receipt: Path, config: HybridModelConfig
  ) -> Mapping[str, object]: ...
  ```

- [ ] **Step 1: Write ABI, split, balancing, and provenance REDs**

  Assert rejection of 25 Hz corpora/models; exact 60 Hz dimensions; validation
  only model selection; one-time test receipt bound to the frozen selection
  manifest; no test metric in architecture choice; hierarchical batches
  balanced by terrain, canonical source, speed, turn, contact, and transition;
  and all-row refit refusal without exact selection/test hashes.

- [ ] **Step 2: Run the focused RED**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_training.py
  ```

  Expected: missing-module failure or absent 60 Hz ABI fields.

- [ ] **Step 3: Implement the 60 Hz wrapper and hierarchical sampler**

  Keep the accepted latent-32/width-512 visual-articulation network. Add exact
  corpus `fps`, horizons, feature/target dimensions, split-ledger SHA, and
  corpus-manifest SHA to training/model receipts. Extend the shared trainer
  without breaking legacy artifacts: `HybridModelConfig.dt` accepts only exact
  `0.04` or `1/60`, while this wrapper requires `1/60`; tri-split corpora expose
  explicit train/validation/test masks, and legacy corpora retain their
  train/evaluation contract. When `fit_all_rows=True`, fit only the full
  corpus's `eligible_mask`, not quarantined/rejected rows. Replace row-family-only
  post-coverage sampling with precomputed nested pools keyed by
  `(terrain_class, canonical_source, speed_bin, turn_bin, contact_bin,
  transition_bin)`, using deterministic empty-bin redistribution. Preserve one
  shuffled complete train-row coverage pass before post-coverage steps.

- [ ] **Step 4: Implement selection/test/refit commands and gates**

  `select` trains on train rows and evaluates validation rows. Freeze its model,
  configuration, and validation receipt. `test` loads that exact manifest and
  evaluates test rows once, publishing an exclusive receipt. `refit` requires
  both manifest digests and fits all clean/usable rows. Report validation and
  test metrics separately against all-joint MAE `0.03`, lower-body locomotion
  frame-max p95 `0.10`, FK p95 `0.08`, support p95 `0.05`, and bilateral
  contact F1 `0.85`.  Publish the all-30-joint frame-max p95 as a diagnostic
  under the immutable `lower-body-locomotion-v1` acceptance profile; do not
  quarantine valid terrain sources merely to reduce that diagnostic.

- [ ] **Step 5: Make focused and existing training tests GREEN**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/foundation_stereo/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_training.py \
    tests/python/test_hybrid_terrain_lmm_training.py
  ```

  Expected: all pass with CPU synthetic fixtures.

- [ ] **Step 6: Commit only Task 5 files**

  ```bash
  git add sonic/python/mm_sonic/full_walking_terrain_lmm_training.py \
    sonic/python/mm_sonic/hybrid_terrain_lmm_training.py \
    tests/python/test_full_walking_terrain_lmm_training.py
  git commit -m "feat: train full walking terrain generator"
  ```

---

### Task 6: Replace command-owned translation with motion-derived `SE(2)` root

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py`
- Modify: `tests/python/test_hybrid_terrain_lmm_runtime.py`

**Interfaces:**
- Consumes: existing `HybridMatcher`, range-safe exact matcher, learned decoder,
  native-limit retry.
- Produces: motion-derived root/yaw composition, contact-aware score,
  `candidate_exhausted` transaction, and new runtime counters for Task 7.

  ```python
  @dataclass(frozen=True)
  class SE2Transform:
      xy: np.ndarray
      yaw: float

  def compose_root_delta(
      world: SE2Transform, local_delta_xy: np.ndarray, local_delta_yaw: float
  ) -> SE2Transform: ...
  ```

- [ ] **Step 1: Write motion-root and retry-exhaustion REDs**

  Add synthetic ranges with known local root/yaw deltas. Assert contiguous
  successor composition exactly once, search re-anchor without teleport,
  common yaw alignment across articulation/root/terrain/probes, zero movement
  for unsupported speed, command speed clamped to corpus p95, hard contact-bit
  compatibility, and bounded retry exhaustion retaining the complete previous
  state with `candidate_exhausted` and no fallback/clamp.

- [ ] **Step 2: Run the focused REDs**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_hybrid_terrain_lmm_runtime.py
  ```

  Expected: failures proving current fixed-speed command integration and
  canonical fallback on retry exhaustion.

- [ ] **Step 3: Implement world-transform ownership and exact score terms**

  Store the active source-local root pose at search commit and one world
  `SE(2)` anchor. For successor `r -> r+1`, compute canonical local delta from
  `Simulation` positions/rotations and compose it once into world. On a search
  jump, update only the source anchor; do not apply the selected row's prior
  source delta. Exclude candidates whose left/right contact bits differ from
  the active contact state, then add the frozen `0.1` different-range cost to
  normalized feature distance. Never fall back to contact-incompatible rows.
  Preserve stable `(score,row)` ties and exact brute/tree parity.

  Derive runtime `fps` and `dt` from the authenticated corpus instead of the
  module's legacy 25 Hz constants. The full-walking wrapper requires exactly
  60 Hz; existing legacy wrappers continue to authenticate 25 Hz artifacts.

- [ ] **Step 4: Implement finite exhaustion without fallback**

  Record the retry budget in runtime identity. If all budgeted exact candidates
  are learned-native-invalid, roll back row/root/heading/query/decode/search
  state bitwise, increment an exhaustion counter, and expose
  `candidate_exhausted`; do not decode canonical source, clamp, or move root.

- [ ] **Step 5: Make runtime tests and 1,000-step synthetic smoke GREEN**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_hybrid_terrain_lmm_runtime.py \
    tests/python/test_hybrid_terrain_lmm_viewer.py
  ```

  Expected: all pass; synthetic smoke reports motion-root ownership.

- [ ] **Step 6: Commit only Task 6 files**

  ```bash
  git add sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
    tests/python/test_hybrid_terrain_lmm_runtime.py
  git commit -m "feat: drive terrain runtime from motion root"
  ```

---

### Task 7: Add coverage/slip comparison and full-corpus formal viewer evidence

**Files:**
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_evaluation.py`
- Create: `sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py`
- Create: `tests/python/test_full_walking_terrain_lmm_evaluation.py`
- Create: `tests/python/test_full_walking_terrain_lmm_viewer.py`
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py`

**Interfaces:**
- Consumes: Task 4 corpus, Task 5 model, Task 6 matcher, frozen overnight
  baseline, authenticated flat/curb/slope/stair scenes, canonical G1 XML.
- Produces: `SlipAccumulator`, `compare_baseline_candidate`, formal 10k-frame
  receipt, and `smoke|view` CLI.

  ```python
  @dataclass(frozen=True)
  class FormalRoute:
      scene_id: str
      duration_seconds: float
      command_times: np.ndarray
      command_speed: np.ndarray
      command_steering: np.ndarray

  @dataclass
  class SlipAccumulator:
      planted_speeds_mps: list[float]

      def update(
          self, previous_probes: np.ndarray, current_probes: np.ndarray,
          previous_contacts: np.ndarray, current_contacts: np.ndarray,
          *, dt: float
      ) -> None: ...

  def compare_baseline_candidate(
      *, baseline: HybridMatcher, candidate: HybridMatcher,
      routes: Sequence[FormalRoute], g1_xml: Path
  ) -> Mapping[str, object]: ...
  ```

- [ ] **Step 1: Write reducer and formal-identity REDs**

  Use analytic support-probe trajectories to assert contact-only slip median
  and p95, desired/realized speed MAE, p95 query distance, selected canonical
  identities, per-terrain aggregation, and exact baseline/candidate command
  reuse. Assert the formal loader rejects the overnight cache/model, partial
  PFNN, capped search, wrong XML/assets, absent split/test/refit receipts,
  non-motion root ownership, any exhaustion/fallback/clamp, or fewer than
  10,000 aggregate MuJoCo forwards.

- [ ] **Step 2: Run the focused REDs**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_evaluation.py \
    tests/python/test_full_walking_terrain_lmm_viewer.py
  ```

  Expected: missing-module failures.

- [ ] **Step 3: Implement the common baseline/candidate reducer**

  After every `mj_forward`, sample authenticated native ankle support probes.
  During each boolean contact run, accumulate planar displacement divided by
  that system's exact timestep; compute population median/p95 over all planted
  samples. Record query distance and world-root realized speed each frame. The
  route JSON is continuous-time: sample it at 25 Hz for the frozen baseline and
  at 60 Hz for the candidate over the same per-scene duration. Candidate formal
  evidence contains 2,500 frames per scene (10,000 total); the baseline uses
  `floor((2500-1)*25/60)+1 = 1042` frames per scene. Publish raw counts plus
  deterministic reducers; never infer improvement from different routes.

- [ ] **Step 4: Implement full-corpus viewer and formal acceptance**

  Reuse the hardened scene/XML/asset capture and exact full-search gates. Add
  full corpus/model/split/test/determinism identities, motion-root ownership,
  supported speed envelope, retry budget/exhaustion, coverage/slip metrics, and
  canonical-source selections. Require each route to resolve from the Task 4
  authenticated scene index; `flat-standard` is a manifest-bound zero-height
  G1HF/v2 grid rather than the diagnostic generated-flat shortcut. Accept only
  if query-distance p95 improves at
  least 30%, slip p95 improves at least 50%, slip median is `<=0.02 m/s`, slip
  p95 is `<=0.08 m/s`, speed MAE is `<=0.12 m/s`, every terrain selects at
  least two canonical identities, and 10,000 forwards have zero safety event.

- [ ] **Step 5: Make focused and existing viewer tests GREEN**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_evaluation.py \
    tests/python/test_full_walking_terrain_lmm_viewer.py \
    tests/python/test_hybrid_terrain_lmm_viewer.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit only Task 7 files**

  ```bash
  git add sonic/python/mm_sonic/full_walking_terrain_lmm_evaluation.py \
    sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py \
    sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
    tests/python/test_full_walking_terrain_lmm_evaluation.py \
    tests/python/test_full_walking_terrain_lmm_viewer.py
  git commit -m "feat: evaluate full walking terrain coverage"
  ```

---

### Task 8: Build twice, train, evaluate 10,000 frames, and launch the viewer

**Files:**
- Create: `docs/superpowers/results/2026-08-10-g1-full-walking-terrain-lmm-result.md`
- Create outside Git: `sonic/runs/g1-full-walking-terrain-lmm/**`

**Interfaces:**
- Consumes: Tasks 1-7 and the exact immutable authorities below.
- Produces: full lane/corpus/model/evaluation/viewer artifacts and the honest
  terminal result report.

- [ ] **Step 1: Run the complete preflight suite and static checks**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
    tests/python/test_full_walking_terrain_lmm_contracts.py \
    tests/python/test_full_walking_terrain_lmm_inventory.py \
    tests/python/test_full_walking_terrain_lmm_grail.py \
    tests/python/test_full_walking_terrain_lmm_pfnn.py \
    tests/python/test_full_walking_terrain_lmm_corpus.py \
    tests/python/test_full_walking_terrain_lmm_training.py \
    tests/python/test_hybrid_terrain_lmm_runtime.py \
    tests/python/test_full_walking_terrain_lmm_evaluation.py \
    tests/python/test_full_walking_terrain_lmm_viewer.py
  ruff check sonic/python/mm_sonic/full_walking_terrain_lmm_*.py \
    sonic/python/mm_sonic/hybrid_terrain_lmm_{training,runtime,viewer}.py \
    tests/python/test_full_walking_terrain_lmm_*.py \
    tests/python/test_hybrid_terrain_lmm_runtime.py
  git diff --check
  ```

  Expected: all pass and all output targets below are absent.

- [ ] **Step 2: Publish inventory/split authority, then launch source builds concurrently**

  ```bash
  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_inventory build \
    --bank /home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate \
    --pfnn-root /home/ubuntu/datasets/pfnn/pfnn \
    --grail-root /home/ubuntu/datasets/GRAIL/data \
    --takara /home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz \
    --output sonic/runs/g1-full-walking-terrain-lmm/inventory-v1.json \
    --split-output sonic/runs/g1-full-walking-terrain-lmm/split-ledger-v1.json \
    --seed 20260810
  ```

  Expected: exactly 15,918 identities and nonempty train/validation/test
  components for every gated terrain class. Only after these immutable files
  publish, launch the following three processes.

  ```bash
  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_grail inherited \
    --inventory sonic/runs/g1-full-walking-terrain-lmm/inventory-v1.json \
    --split-ledger sonic/runs/g1-full-walking-terrain-lmm/split-ledger-v1.json \
    --bank /home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate \
    --output sonic/runs/g1-full-walking-terrain-lmm/lanes/grail-inherited-60hz-v1

  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_grail supplement \
    --inventory sonic/runs/g1-full-walking-terrain-lmm/inventory-v1.json \
    --split-ledger sonic/runs/g1-full-walking-terrain-lmm/split-ledger-v1.json \
    --grail-root /home/ubuntu/datasets/GRAIL/data \
    --takara /home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz \
    --takara-remap /home/ubuntu/projects/g1_mm/isaac_to_mj.npy \
    --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
    --output sonic/runs/g1-full-walking-terrain-lmm/lanes/supplement-60hz-v1

  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_pfnn build \
    --inventory sonic/runs/g1-full-walking-terrain-lmm/inventory-v1.json \
    --split-ledger sonic/runs/g1-full-walking-terrain-lmm/split-ledger-v1.json \
    --pfnn-root /home/ubuntu/datasets/pfnn/pfnn \
    --gmr-root /home/ubuntu/.cache/native-g1-pfnn/GMR \
    --retarget-root /home/ubuntu/.cache/native-g1-pfnn/retargeting_project \
    --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
    --workers 8 \
    --output sonic/runs/g1-full-walking-terrain-lmm/lanes/pfnn-60hz-v1
  ```

  Expected terminal counts before PFNN fit/quality filtering: inherited GRAIL
  `15,814 * 598 = 9,456,772` rows, missing slopes `23 * 598 = 13,754`, Takara
  `41,835`, and PFNN gait-admitted at most `246,630` 60 Hz rows, for at most
  `9,758,991` rows before PFNN fit/continuity and all-family quality filtering.
  Every one of 15,918 source identities has a terminal
  included/excluded/rejected receipt.

- [ ] **Step 3: Assemble, verify, and start the deterministic rebuild**

  ```bash
  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_corpus build \
    --inventory sonic/runs/g1-full-walking-terrain-lmm/inventory-v1.json \
    --split-ledger sonic/runs/g1-full-walking-terrain-lmm/split-ledger-v1.json \
    --lane sonic/runs/g1-full-walking-terrain-lmm/lanes/grail-inherited-60hz-v1 \
    --lane sonic/runs/g1-full-walking-terrain-lmm/lanes/supplement-60hz-v1 \
    --lane sonic/runs/g1-full-walking-terrain-lmm/lanes/pfnn-60hz-v1 \
    --output sonic/runs/g1-full-walking-terrain-lmm/corpus-v1

  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_corpus verify \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --receipt sonic/runs/g1-full-walking-terrain-lmm/evidence/corpus-v1-verify.json
  ```

  After the first corpus publishes, launch this exact reproduction in parallel
  with model training; formal evidence waits for its accepted receipt:

  ```bash
  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_corpus reproduce \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --inventory sonic/runs/g1-full-walking-terrain-lmm/inventory-v1.json \
    --split-ledger sonic/runs/g1-full-walking-terrain-lmm/split-ledger-v1.json \
    --lane sonic/runs/g1-full-walking-terrain-lmm/lanes/grail-inherited-60hz-v1 \
    --lane sonic/runs/g1-full-walking-terrain-lmm/lanes/supplement-60hz-v1 \
    --lane sonic/runs/g1-full-walking-terrain-lmm/lanes/pfnn-60hz-v1 \
    --scratch sonic/runs/g1-full-walking-terrain-lmm/.corpus-v1-reproduction \
    --receipt sonic/runs/g1-full-walking-terrain-lmm/evidence/determinism-v1.json
  ```

  Expected: every payload digest matches and the command removes only its own
  scratch duplicate after the immutable accepted receipt is synced.

- [ ] **Step 4: Train selection on GPU 5, freeze test, and refit on GPU 6**

  ```bash
  PYTHONPATH=.:resources:sonic/python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /home/ubuntu/miniconda3/envs/foundation_stereo/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_training select \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --output sonic/runs/g1-full-walking-terrain-lmm/selection-latent32-v1 \
    --device cuda:5 --seed 1234 --batch-size 256 --post-coverage-steps 30000

  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/foundation_stereo/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_training test \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --model sonic/runs/g1-full-walking-terrain-lmm/selection-latent32-v1 \
    --output sonic/runs/g1-full-walking-terrain-lmm/evidence/selection-test-v1.json \
    --device cuda:5

  PYTHONPATH=.:resources:sonic/python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /home/ubuntu/miniconda3/envs/foundation_stereo/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_training refit \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --selection sonic/runs/g1-full-walking-terrain-lmm/selection-latent32-v1 \
    --test-receipt sonic/runs/g1-full-walking-terrain-lmm/evidence/selection-test-v1.json \
    --output sonic/runs/g1-full-walking-terrain-lmm/final-latent32-v1-allrows \
    --device cuda:6 --seed 1234 --batch-size 256 --post-coverage-steps 30000
  ```

  If and only if latent-32 fails a frozen validation/test gate, run latent-64
  with the same data, seed, loss, and schedule as the sole architecture change.
  Do not launch a VAE unless both deterministic variants fit train rows and
  fail held-out identities.

- [ ] **Step 5: Run identical baseline/full routes for at least 10,000 frames**

  ```bash
  PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_viewer smoke \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --model sonic/runs/g1-full-walking-terrain-lmm/final-latent32-v1-allrows \
    --baseline-corpus sonic/runs/g1-hybrid-terrain-lmm/corpus-primary-pfnn-v2-strict \
    --baseline-model sonic/runs/g1-hybrid-terrain-lmm/final-combined-v2-strict-latent32-visual-v1-allrows \
    --scene flat-standard --scene grail-curb-default --scene ramp-10-up-down \
    --scene stairs-standard --frames-per-scene 2500 \
    --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
    --receipt sonic/runs/g1-full-walking-terrain-lmm/evidence/formal-10000-v1.json
  ```

  Expected: accepted receipt with full search, all coverage/slip gates green,
  and exactly zero exhaustion/fallback/clamp/native/nonfinite/successor events.

- [ ] **Step 6: Launch the interactive viewer and inspect steering**

  ```bash
  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    PYTHONPATH=.:resources:sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m \
    mm_sonic.full_walking_terrain_lmm_viewer view \
    --corpus sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 \
    --model sonic/runs/g1-full-walking-terrain-lmm/final-latent32-v1-allrows \
    --scene ramp-10-up-down \
    --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
  ```

  Verify W/S speed, A/D steering, Space stop, R reset, motion-derived root
  travel, planted-foot behavior, and visible diagnostics.

- [ ] **Step 7: Publish the result document and commit it**

  Record every input/lane/corpus/model/evidence SHA, exact included/excluded
  counts, validation/test metrics, determinism result, baseline/candidate slip
  and coverage, safety counters, viewer command, screenshot path/hash, and any
  red gate without weakening or relabeling it.

  ```bash
  git add docs/superpowers/results/2026-08-10-g1-full-walking-terrain-lmm-result.md
  git commit -m "docs: publish full walking terrain LMM result"
  ```

## Plan Self-Review Checklist

- Every design requirement maps to Tasks 1-8.
- Tasks 2, 3, and 6 have disjoint production/test ownership and may run in
  parallel after Task 1.
- Validation selects; test evaluates once; refit binds both immutable receipts.
- Duration-preserving GRAIL conversion is 598 rows per 250-row clip, not 600.
- The second deterministic build overlaps training but gates formal evidence.
- The 2:00 PM target is aggressive and never permits a partial completion
  claim or weakened gate.
