# Terrain Oracle Phase 1: Audited Canonical Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, content-addressed, visually auditable 50 Hz G1
corpus from the clean Takara/BONES, Justin, GRAIL, and LAFAN sources.

**Architecture:** Add a new `mm_sonic.terrain_oracle` package without changing the
existing greedy terrain viewer. Source-specific adapters terminate at one strict
Z-up/wxyz G1 contract. Contact reconstruction, symmetry, audit, splitting, and
coverage operate only on that canonical contract and publish immutable local
artifacts plus small Git-tracked manifests.

**Tech Stack:** Python 3.10+, NumPy, SciPy 1.15.3, MuJoCo, Zarr 2.18, joblib,
OpenUSD only at the GRAIL import boundary, unittest/pytest.

## Global Constraints

- External motion and terrain assets are read-only; do not modify Takara's or
  Justin's repositories.
- Clean source motion defines the corpus. SONIC rollout windows may not supply
  canonical poses.
- The canonical representation is Z-up, right-handed, 50 Hz, root/body quaternion
  `wxyz`, 29 joints in `ISAACLAB_JOINT_NAMES`, and 30 bodies in
  `ISAACLAB_BODY_NAMES`.
- Every derivative is recomputed after resampling and never across clip boundaries.
- Source and mirrored variants remain in the same split.
- Byte-identical and near-duplicate sources remain in the same split and count once
  when estimating coverage.
- No command inferred from motion may overwrite an observed joystick command.
- Large arrays and videos remain outside Git. Git receives schemas, code, manifests,
  hashes, reports, and small synthetic fixtures only.
- Artifact publication is no-replace and deterministic.

---

## File map

- `sonic/python/mm_sonic/terrain_oracle/__init__.py`: public phase-1 exports.
- `sonic/python/mm_sonic/terrain_oracle/math3d.py`: checked quaternion and rigid
  transform operations.
- `sonic/python/mm_sonic/terrain_oracle/canonical.py`: canonical clip, command,
  terrain, and provenance contracts.
- `sonic/python/mm_sonic/terrain_oracle/storage.py`: immutable clip/corpus storage.
- `sonic/python/mm_sonic/terrain_oracle/source_flat.py`: Takara/BONES adapter.
- `sonic/python/mm_sonic/terrain_oracle/source_justin.py`: Justin Zarr/NPZ adapter.
- `sonic/python/mm_sonic/terrain_oracle/source_grail.py`: clean c490 PKL/USD adapter.
- `sonic/python/mm_sonic/terrain_oracle/source_lafan.py`: 30 Hz G1 CSV adapter.
- `sonic/python/mm_sonic/terrain_oracle/contact.py`: sole geometry and contact
  reconstruction.
- `sonic/python/mm_sonic/terrain_oracle/symmetry.py`: exact full-contract mirror.
- `sonic/python/mm_sonic/terrain_oracle/audit.py`: mechanical source audit.
- `sonic/python/mm_sonic/terrain_oracle/coverage.py`: deduplication, split grouping,
  and frozen coverage atlas.
- `sonic/python/mm_sonic/terrain_oracle/corpus_cli.py`: import/audit/freeze CLI.
- `sonic/schemas/terrain_oracle_corpus_v1.schema.json`: corpus manifest schema.
- `sonic/schemas/terrain_oracle_audit_v1.schema.json`: source audit schema.
- `sonic/schemas/terrain_oracle_coverage_v1.schema.json`: coverage manifest schema.
- `tests/python/terrain_oracle_test_utils.py`: independent synthetic fixtures.
- `tests/python/test_oracle_*.py`: focused phase-1 tests.

### Task 1: Canonical math and motion contracts

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/__init__.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/math3d.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/canonical.py`
- Create: `tests/python/terrain_oracle_test_utils.py`
- Create: `tests/python/test_oracle_canonical.py`

**Interfaces:**
- Produces: `RigidTransform`, `CommandTrack`, `TerrainBinding`, `SourceIdentity`,
  `CanonicalTerrainMesh`, `CanonicalClip`, `CanonicalClip.validate()`, and
  `derive_clip_kinematics()`.
- Canonical array shapes are `[T,3]` root position, `[T,4]` root wxyz,
  `[T,29]` joints, `[T,30,3]` body position/velocity, `[T,30,4]` body wxyz,
  `[T,2,3]` sole/heel/toe positions, `[T,2,4]` sole wxyz, and `[T,2]`
  contact/confidence.

- [ ] **Step 1: Write failing validation and derivative tests**

```python
def test_canonical_clip_rejects_xyzw_root_and_cross_clip_derivative():
    clip = synthetic_canonical_clip(frames=6)
    bad = replace(clip, root_quaternion_world_wxyz=clip.root_quaternion_world_wxyz[:, (1, 2, 3, 0)])
    with self.assertRaisesRegex(ContractError, "root quaternion"):
        bad.validate()
    np.testing.assert_allclose(
        clip.root_linear_velocity_world[:, 0],
        np.full(6, 0.4, np.float32),
        atol=1e-6,
    )

def test_observed_command_is_separate_from_inferred_command():
    clip = synthetic_canonical_clip(frames=6, observed_commands=False)
    self.assertFalse(np.any(clip.commands.observed_mask))
    self.assertTrue(np.isfinite(clip.commands.inferred_velocity_local_xy).all())
```

- [ ] **Step 2: Run the tests and verify contract failures**

Run:

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_canonical -v
```

Expected: import failure for `mm_sonic.terrain_oracle.canonical`.

- [ ] **Step 3: Implement checked immutable dataclasses**

```python
@dataclass(frozen=True)
class CommandTrack:
    observed_travel_stick_xy: np.ndarray
    observed_facing_stick_xy: np.ndarray
    observed_mask: np.ndarray
    inferred_velocity_local_xy: np.ndarray
    inferred_facing_local_xy: np.ndarray
    inferred_yaw_rate_rad_s: np.ndarray

@dataclass(frozen=True)
class TerrainBinding:
    asset_path: str
    asset_sha256: str
    mesh_sha256: str
    world_from_terrain: RigidTransform
    validity_mask_path: str | None

@dataclass(frozen=True)
class CanonicalTerrainMesh:
    vertices_local: np.ndarray
    faces: np.ndarray
    valid_faces: np.ndarray
    source_asset_sha256: str

@dataclass(frozen=True)
class CanonicalClip:
    clip_id: str
    fps: float
    source: SourceIdentity
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray
    root_linear_velocity_world: np.ndarray
    root_angular_velocity_world: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray
    sole_position_world: np.ndarray
    sole_quaternion_world_wxyz: np.ndarray
    heel_position_world: np.ndarray
    toe_position_world: np.ndarray
    contact: np.ndarray
    contact_confidence: np.ndarray
    commands: CommandTrack
    terrain: TerrainBinding | None
    action_tags: tuple[str, ...]
    mirror_of: str | None = None

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])

    def validate(self) -> None:
        if self.fps != 50.0 or self.frame_count < 2:
            raise ContractError(
                "canonical clip must contain at least two 50 Hz frames"
            )
        expected = {
            "root_position_world": (self.frame_count, 3),
            "root_quaternion_world_wxyz": (self.frame_count, 4),
            "joint_position": (self.frame_count, 29),
            "body_position_world": (self.frame_count, 30, 3),
            "body_quaternion_world_wxyz": (self.frame_count, 30, 4),
            "contact": (self.frame_count, 2),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ContractError(
                    f"{name} must be finite with shape {shape}"
                )
        root_norm = np.linalg.norm(
            self.root_quaternion_world_wxyz, axis=1
        )
        if not np.allclose(root_norm, 1.0, atol=1.0e-5):
            raise ContractError("root quaternion must be normalized wxyz")
```

Implement `RigidTransform.compose`, `inverse`, `apply_points`, shortest-path
`slerp_wxyz`, quaternion unrolling, and central/one-sided finite differences.
All constructors own C-contiguous read-only arrays.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle \
  tests/python/terrain_oracle_test_utils.py \
  tests/python/test_oracle_canonical.py
git commit -m "feat: define canonical terrain oracle motion contract"
```

### Task 2: Immutable clip and corpus storage

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/storage.py`
- Create: `sonic/schemas/terrain_oracle_corpus_v1.schema.json`
- Create: `tests/python/test_oracle_storage.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `CanonicalClip`, `CanonicalTerrainMesh`.
- Produces:
  `write_clip(output: Path, clip: CanonicalClip) -> ClipRecord`,
  `read_clip(path: Path) -> CanonicalClip`,
  `write_mesh(output: Path, mesh: CanonicalTerrainMesh) -> MeshRecord`,
  `read_mesh(path: Path) -> CanonicalTerrainMesh`,
  `publish_corpus(output: Path, records: Sequence[ClipRecord], metadata: dict) -> Path`,
  and `load_corpus(path: Path) -> CorpusManifest`.

- [ ] **Step 1: Write no-overwrite, hash, and round-trip tests**

```python
def test_clip_round_trip_is_allow_pickle_false_and_content_addressed():
    clip = synthetic_canonical_clip(frames=8)
    record = write_clip(root / "clips", clip)
    self.assertEqual(record.relative_path, f"clips/{record.sha256}.npz")
    restored = read_clip(root / record.relative_path)
    np.testing.assert_array_equal(restored.joint_position, clip.joint_position)
    with self.assertRaises(FileExistsError):
        publish_corpus(root / "corpus", [record], {"name": "again"})
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_storage -v
```

Expected: import failure for `terrain_oracle.storage`.

- [ ] **Step 3: Implement deterministic storage**

Use `np.savez` with `allow_pickle=False` compatible arrays, canonical sorted JSON,
SHA-256 over the exact NPZ bytes, a temporary sibling directory, `fsync`, and
Linux no-replace rename. The manifest builder emits the exact repository constants:

```python
manifest = {
    "schema": "terrain-oracle-corpus/v1",
    "coordinate_frame": "z-up-right-handed",
    "quaternion_convention": "wxyz",
    "fps": 50.0,
    "joint_order": list(ISAACLAB_JOINT_NAMES),
    "body_order": list(ISAACLAB_BODY_NAMES),
    "clips": [record.to_dict() for record in records],
    "meshes": [record.to_dict() for record in mesh_records],
}
```

Add `/sonic/.oracle-venv/` to `.gitignore`; phase 3 creates that isolated
environment without modifying an upstream Conda environment.

Do not serialize external assets into Git; store their path, size, hash, license,
and transform in metadata.

- [ ] **Step 4: Validate schema and round trip**

Run Step 2 and:

```bash
git diff --check
```

Expected: tests pass and no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add .gitignore sonic/python/mm_sonic/terrain_oracle/storage.py \
  sonic/schemas/terrain_oracle_corpus_v1.schema.json \
  tests/python/test_oracle_storage.py
git commit -m "feat: add immutable canonical corpus storage"
```

### Task 3: Flat and Justin clean-source adapters

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/source_flat.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/source_justin.py`
- Create: `tests/python/test_oracle_source_flat.py`
- Create: `tests/python/test_oracle_source_justin.py`

**Interfaces:**
- Consumes: `MotionFolder` from `mm_sonic.torch_motion_data` and Justin's Zarr
  arrays.
- Produces:
  `iter_flat_clips(root: Path, *, tags: tuple[str, ...]) -> Iterator[CanonicalClip]`
  and
  `iter_justin_clips(zarr_path: Path, terrain: TerrainBinding) -> Iterator[CanonicalClip]`.

- [ ] **Step 1: Write adapter boundary tests**

```python
def test_flat_adapter_preserves_native_wxyz_and_50_hz():
    source = write_takara_clip(root / "walk", frames=60)
    clip = next(iter_flat_clips(root, tags=("flat", "walk")))
    self.assertEqual(clip.fps, 50.0)
    self.assertEqual(clip.source.quaternion_convention, "wxyz")
    np.testing.assert_array_equal(clip.joint_position, load_npz(source)["joint_pos"])

def test_justin_adapter_reorders_explicit_xyzw_only_once():
    write_synthetic_justin_zarr(root / "justin.zarr")
    clip = next(iter_justin_clips(root / "justin.zarr", terrain_binding()))
    np.testing.assert_allclose(clip.root_quaternion_world_wxyz[0], (1, 0, 0, 0))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_source_flat \
  tests.python.test_oracle_source_justin -v
```

- [ ] **Step 3: Implement adapters**

Reuse `MotionFolder.load` for all 173 Takara/BONES folders. For Justin, validate
`quaternion_convention == "xyzw"`, `fps == 50`, exclusive clip boundaries, and
the 29/30 G1 ordering before converting root/body quaternions to wxyz. Infer
commands into the `inferred_*` fields and keep `observed_mask=False` unless a
source command sidecar is explicitly present.

- [ ] **Step 4: Run focused tests and a read-only inventory canary**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_source_flat \
  tests.python.test_oracle_source_justin -v
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m mm_sonic.terrain_oracle.corpus_cli inventory \
  --flat /move/data/terrain-aware/motion-matching/takara_bones_walk_support_v2_startstop \
  --justin /move/u/justingu/rmr_tracking/motions/isaac6/stairs_justin_mobu/stairs_justin_mobu_50hz.zarr
```

Expected inventory: 173 flat clips and the exact unique Justin clip list; no
external file changes.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/source_flat.py \
  sonic/python/mm_sonic/terrain_oracle/source_justin.py \
  tests/python/test_oracle_source_flat.py \
  tests/python/test_oracle_source_justin.py
git commit -m "feat: import clean flat and Justin motion sources"
```

### Task 4: Clean c490 GRAIL and G1 LAFAN adapters

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/source_grail.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/source_lafan.py`
- Create: `tests/python/test_oracle_source_grail.py`
- Create: `tests/python/test_oracle_source_lafan.py`

**Interfaces:**
- Consumes: existing `discover_grail_clips`, `load_grail_motion`,
  `resample_grail_motion`, `G1MujocoFK`; LAFAN G1 CSV rows with 36 columns
  `XYZ QX QY QZ QW + 29 joints`.
- Produces:
  `iter_grail_clips(shard_root: Path, families: Sequence[str], model_path: Path)`,
  `load_lafan_csv(path: Path, model_path: Path, *, terrain: TerrainBinding | None)`,
  and `classify_lafan_name(name: str) -> tuple[str, ...]`.

- [ ] **Step 1: Write strict source tests**

```python
def test_lafan_30_hz_csv_resamples_to_50_hz_without_duplicate_endpoint():
    write_lafan_csv(path, frames=4)
    clip = load_lafan_csv(path, model_path, terrain=None)
    self.assertEqual(clip.frame_count, 6)
    self.assertEqual(clip.fps, 50.0)
    self.assertEqual(clip.source.license_id, "CC-BY-NC-ND-4.0")

def test_c490_inventory_is_exact_and_never_reads_rollout_windows():
    records = discover_clean_c490_records(shard_root)
    counts = Counter(r.family for r in records)
    self.assertEqual(counts, {
        "c490_stair_p1": 166, "c490_stair_p2": 174,
        "c490_slope": 77, "c490_curb": 72,
    })
    self.assertEqual(counts["c490_stair_p1"] + counts["c490_stair_p2"], 340)
    self.assertEqual(counts["c490_slope"], 77)
    self.assertEqual(counts["c490_curb"], 72)
    self.assertEqual(sum(counts.values()), 489)
    self.assertTrue(all("/sonic-rollouts/" not in str(r.robot_path) for r in records))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_source_grail \
  tests.python.test_oracle_source_lafan -v
```

- [ ] **Step 3: Implement both adapters**

The GRAIL importer requests exactly:

```python
C490_FAMILIES = (
    "c490_stair_p1",
    "c490_stair_p2",
    "c490_slope",
    "c490_curb",
)
C490_EXPECTED_COUNTS = {
    "c490_stair_p1": 166,
    "c490_stair_p2": 174,
    "c490_slope": 77,
    "c490_curb": 72,
}
```

It preserves the paired USD hash and exact `terrain_position_env` /
`terrain_rotation_env_wxyz`; any additional environment origin must be applied
once at an explicit call boundary and tested. Extract the USD vertices, triangulated
faces, and validity into one content-addressed `CanonicalTerrainMesh`; the clip's
`TerrainBinding` references that mesh hash. The LAFAN adapter accepts only the
documented G1 CSV order, resamples translations/joints continuously and root XYZW
with shortest-path SLERP, then runs the repository's validated MuJoCo FK. LAFAN
obstacle names remain flat/unbound until a terrain pairing is explicitly supplied.

- [ ] **Step 4: Run focused tests and exact inventory**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_source_grail \
  tests.python.test_oracle_source_lafan -v
```

Expected: tests pass; GRAIL test asserts exactly 340 stair, 77 slope, and 72
curb sources, for 489 clean sources total.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/source_grail.py \
  sonic/python/mm_sonic/terrain_oracle/source_lafan.py \
  tests/python/test_oracle_source_grail.py \
  tests/python/test_oracle_source_lafan.py
git commit -m "feat: import clean GRAIL and retargeted LAFAN motion"
```

### Task 5: Sole geometry and contact reconstruction

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/contact.py`
- Create: `tests/python/test_oracle_contact.py`

**Interfaces:**
- Consumes: a canonical pose sequence, exact MuJoCo G1 model, and optional
  `CanonicalTerrainMesh` surface query.
- Produces:
  `SoleGeometry.from_model(model)`,
  `CanonicalMeshQuery(mesh, world_from_terrain)`,
  `reconstruct_contacts(clip, query, config) -> ContactReconstruction`,
  and `ContactReconstruction.apply(clip) -> CanonicalClip`.

- [ ] **Step 1: Write analytic plane/step/contact tests**

```python
def test_stationary_complete_sole_is_contact_but_moving_hovering_sole_is_not():
    grounded = reconstruct_contacts(plane_clip(z=0.0, foot_speed=0.0), plane_query, config)
    hovering = reconstruct_contacts(plane_clip(z=0.04, foot_speed=0.0), plane_query, config)
    moving = reconstruct_contacts(plane_clip(z=0.0, foot_speed=0.5), plane_query, config)
    self.assertTrue(np.all(grounded.contact[:, 0]))
    self.assertFalse(np.any(hovering.contact[:, 0]))
    self.assertFalse(np.any(moving.contact[:, 0]))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_contact -v
```

- [ ] **Step 3: Implement hysteretic full-sole contact**

Use named ankle-roll descendant geoms, sole corners plus heel/toe probes, exact
triangle ray/closest-point distance and normal, tangential speed, angular speed, and
enter/leave hysteresis. `CanonicalMeshQuery` validates triangle indices and
valid-face masks and transforms queries exactly once through `TerrainBinding`.
Confidence is a finite `[0,1]` score; support identity is never inferred from foot
height alone.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/contact.py \
  tests/python/test_oracle_contact.py
git commit -m "feat: reconstruct exact canonical foot contacts"
```

### Task 6: Full-contract symmetric augmentation

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/symmetry.py`
- Create: `tests/python/test_oracle_symmetry.py`

**Interfaces:**
- Consumes: `CanonicalClip`, `CanonicalTerrainMesh`.
- Produces:
  `mirror_clip(clip: CanonicalClip) -> CanonicalClip`,
  `mirror_terrain(binding, mesh)`,
  and `assert_mirror_involution(original, mirrored)`.

- [ ] **Step 1: Write an exhaustive involution test**

```python
def test_mirror_twice_restores_every_history_command_contact_and_terrain_field():
    original = asymmetric_canonical_clip_with_commands_and_stairs()
    mirrored = mirror_clip(original)
    restored = mirror_clip(mirrored)
    assert_canonical_arrays_equal(restored, original)
    np.testing.assert_array_equal(mirrored.contact[:, 0], original.contact[:, 1])
    np.testing.assert_allclose(
        mirrored.commands.observed_travel_stick_xy[:, 0],
        -original.commands.observed_travel_stick_xy[:, 0],
    )
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_symmetry -v
```

- [ ] **Step 3: Implement one canonical mirror**

Move no existing caller yet. Reuse the validated joint/body permutations and signs
from `offline_corpus`, and reflect root/body/sole/heel/toe translations,
orientations, linear/angular velocities, contacts, confidence, observed and
inferred commands, terrain vertices/normals, terrain transform, and provenance.
The mirror plane is canonical local sagittal `y=0`; global placement occurs later.

- [ ] **Step 4: Run the new and existing symmetry tests**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_symmetry \
  tests.python.test_sonic_offline_corpus \
  tests.python.test_sonic_warp_c490_stair_snippets -v
```

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/symmetry.py \
  tests/python/test_oracle_symmetry.py
git commit -m "feat: mirror every canonical motion and terrain field"
```

### Task 7: Mechanical and robot-mesh source audit

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/audit.py`
- Create: `sonic/schemas/terrain_oracle_audit_v1.schema.json`
- Create: `tests/python/test_oracle_audit.py`

**Interfaces:**
- Consumes: `CanonicalClip`, exact G1 model, and exact terrain query.
- Produces:
  `audit_clip(clip, model, terrain) -> ClipAudit`,
  `ClipAudit.status in {"accepted", "accepted_with_intervals_removed", "rejected"}`,
  `accepted_intervals`, metrics, and structured reasons.

- [ ] **Step 1: Write failure-classification tests**

```python
def test_audit_rejects_quaternion_jump_body_penetration_and_stance_skate():
    report = audit_clip(corrupted_clip(), model, terrain)
    self.assertEqual(report.status, "rejected")
    self.assertEqual(
        {r.code for r in report.reasons},
        {"quaternion_discontinuity", "body_penetration", "stance_skate"},
    )
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_audit -v
```

- [ ] **Step 3: Implement the independent source audit**

Compute joint limit/velocity, quaternion continuity, root plausibility, contact
consistency, complete-sole support, stance drift, foot/body collision, terrain
registration, and derivative discontinuity from disk-loaded arrays. Reuse
`evaluate_kinematic_sequence` only as a low-level contact calculation; do not trust
source labels. The report records source hash, model hash, terrain hash, thresholds,
accepted intervals, and metric maxima.

- [ ] **Step 4: Run focused tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/audit.py \
  sonic/schemas/terrain_oracle_audit_v1.schema.json \
  tests/python/test_oracle_audit.py
git commit -m "feat: audit canonical source motion mechanically"
```

### Task 8: Deduplication, split grouping, and frozen coverage atlas

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/coverage.py`
- Create: `sonic/schemas/terrain_oracle_coverage_v1.schema.json`
- Create: `tests/python/test_oracle_coverage.py`

**Interfaces:**
- Consumes: accepted clip records and audit reports.
- Produces:
  `deduplicate(records) -> Deduplication`,
  `assign_grouped_splits(records, seed) -> SplitManifest`,
  `build_coverage(records) -> CoverageManifest`,
  and `freeze_coverage(path, manifest)`.

- [ ] **Step 1: Write anti-leakage and anti-density tests**

```python
def test_duplicates_mirrors_and_same_source_never_cross_splits():
    split = assign_grouped_splits(records_with_duplicates_and_mirrors(), "oracle-v1")
    self.assertEqual(split["walk"], split["walk__mirror"])
    self.assertEqual(split["walk"], split["walk_byte_copy"])

def test_repeating_one_ascent_window_does_not_change_coverage():
    one = build_coverage([ascent, descent])
    many = build_coverage([ascent] * 1000 + [descent])
    self.assertEqual(one.connected_cells, many.connected_cells)

def test_one_authored_terrain_asset_never_crosses_splits():
    split = assign_grouped_splits(
        records_sharing_one_terrain_asset(), "oracle-v1"
    )
    self.assertEqual(len(set(split.values())), 1)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_coverage -v
```

- [ ] **Step 3: Implement deterministic joint coverage**

Coverage axes are movement/facing offset, speed, yaw rate, start/stop/reverse,
support phase/leg, step forward/lateral/vertical displacement, surface normal,
contact yaw, swing clearance, duration, and action class. Publish marginal counts,
joint occupied cells, connected components, source IDs, dedup groups, and the
frozen content hash. Group source clips, mirrors, duplicates, complete authored
terrain assets, and procedural parameter families before assigning splits. A frozen
manifest cannot be overwritten.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/coverage.py \
  sonic/schemas/terrain_oracle_coverage_v1.schema.json \
  tests/python/test_oracle_coverage.py
git commit -m "feat: freeze balanced terrain motion coverage"
```

### Task 9: Corpus CLI, exhaustive renders, and real-data gate

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/corpus_cli.py`
- Create: `tests/python/test_oracle_corpus_cli.py`
- Modify: `sonic/python/mm_sonic/terrain_oracle/__init__.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Produces CLI subcommands:
  `inventory`, `import`, `audit`, `render-audit`, `coverage`, and `freeze`.
- `freeze` requires every accepted interval to have a mechanical audit and a
  robot-mesh render record.

- [ ] **Step 1: Write end-to-end synthetic CLI test**

```python
def test_cli_freeze_requires_audits_and_renders(tmp_path):
    self.assertEqual(main(["import", "--fixture", str(src), "--output", str(raw)]), 0)
    self.assertEqual(main(["audit", "--corpus", str(raw), "--output", str(audited)]), 0)
    self.assertEqual(main(["freeze", "--corpus", str(audited), "--output", str(frozen)]), 2)
    create_render_receipts(audited)
    self.assertEqual(main(["freeze", "--corpus", str(audited), "--output", str(frozen)]), 0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_corpus_cli -v
```

- [ ] **Step 3: Implement CLI orchestration and render receipts**

Each robot-mesh render receipt records clip hash, exact model/terrain hashes,
renderer invocation, output video hash, contact-overlay hash, and completion. A
contact sheet includes every accepted interval; stratified full videos cover every
source/action/terrain/direction group.

- [ ] **Step 4: Run the complete phase-1 test suite**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest discover -s tests/python -p 'test_oracle_*.py' -v
git diff --check
```

Expected: all phase-1 tests pass.

- [ ] **Step 5: Run and inspect the real-data gate**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m mm_sonic.terrain_oracle.corpus_cli inventory \
  --flat /move/data/terrain-aware/motion-matching/takara_bones_walk_support_v2_startstop \
  --justin /move/u/justingu/rmr_tracking/motions/isaac6/stairs_justin_mobu/stairs_justin_mobu_50hz.zarr \
  --grail-root /move/data/terrain-aware/grail-sweep/shards \
  --grail-families c490_stair_p1,c490_stair_p2,c490_slope,c490_curb \
  --output sonic/runs/terrain-oracle-v1/inventory.json
```

Expected: 173 flat sources, all Justin sources, and exactly 489 clean GRAIL
sources. Download the selected G1 LAFAN CSV release at the audited immutable
revision, then rerun inventory:

```bash
/move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -m huggingface_hub.commands.huggingface_cli download \
  lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset \
  --revision ce1572906efe6157840e8474d5a0d7aa87481e74 \
  --include 'g1/*.csv' LICENSE README.md \
  --local-dir /move/data/terrain-aware/motion-matching/lafan1_g1_ce157290
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m mm_sonic.terrain_oracle.corpus_cli inventory \
  --lafan /move/data/terrain-aware/motion-matching/lafan1_g1_ce157290/g1 \
  --output sonic/runs/terrain-oracle-v1/lafan-inventory.json
```

Record the upstream revision and `CC-BY-NC-ND-4.0` license in the corpus manifest.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/corpus_cli.py \
  sonic/python/mm_sonic/terrain_oracle/__init__.py \
  tests/python/test_oracle_corpus_cli.py sonic/README.md
git commit -m "feat: publish audited canonical terrain corpus"
```

## Phase-1 exit gate

Do not start phase 2 until:

- every accepted clip round-trips through the canonical storage contract;
- the real inventory contains 173 flat sources and exactly 340 stair, 77 slope,
  and 72 curb clean c490 GRAIL sources (489 total), plus the audited Justin and
  selected LAFAN sources;
- mirrored involution covers every canonical field;
- train/evaluation groups have no source, mirror, or duplicate leakage;
- coverage is frozen from clean deduplicated sources;
- every accepted interval has mechanical audit evidence and robot-mesh render
  evidence; and
- rejected or trimmed intervals retain explicit reasons.
