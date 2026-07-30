# G1 Expanded Stair and Curb Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a deterministic native 50 Hz motion-matching corpus containing every locally available G1 stair or curb recording whose motion identity and terrain alignment pass strict admission.

**Architecture:** A checked registry pins candidate motion and geometry identities. Family-specific adapters canonicalize motion and construct source height grids, a common admission layer verifies layout/FK/contact alignment and duplicates, and a transactional publisher emits accepted clips plus structured rejection evidence without modifying the qualified small corpus.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo, USD (`pxr`), XML, unittest, existing G1 Torch motion/terrain builders.

## Global Constraints

- Work only on `research/g1-torch-terrain-kinematics`.
- Preserve `build/torch-stair-small` and its checked configuration as the control.
- Output is exactly 50 Hz, Z-up, planar X/Y, wxyz, 29 IsaacLab-order joints, and 30 IsaacLab-order bodies.
- Only explicitly registered sources may be considered.
- Motion shape alone never proves joint/body identity.
- Every accepted non-flat clip has authoritative terrain and a checked motion-to-terrain transform.
- Missing or ambiguous terrain rejects the candidate; it never becomes flat.
- Exact duplicate motion hashes publish once.
- Generated NPZ data remains ignored.
- Follow red/green TDD for every production behavior.

---

### Task 1: Checked candidate registry and native-motion admission

**Files:**
- Create: `resources/g1_torch_terrain_builder/__init__.py`
- Create: `resources/g1_torch_terrain_builder/registry.py`
- Create: `resources/g1_torch_terrain_builder/motion.py`
- Create: `tests/python/test_torch_terrain_registry.py`
- Create: `tests/python/test_torch_terrain_motion.py`

**Interfaces:**
- Consumes: local source root `/home/ubuntu/Downloads/artifacts` and GRAIL root `/home/ubuntu/datasets/GRAIL/data/stair_p1`.
- Produces: `SourceSpec`, `CANDIDATE_SPECS`, `resolve_registered_sources()`, and `load_native_motion_50hz()`.

- [ ] **Step 1: Write registry tests that pin names, families, and hashes**

Add tests that assert the registry contains only the approved families and the
exact motion identities already measured on this host:

```python
EXPECTED = {
    "staircase-side-stepto": (
        "stair-local",
        "e92ad315d8ed212a9fe9cd656199e5bf1389993530ddc2850a4dbe113e0ef18c",
    ),
    "staircase-final-v3": (
        "stair-local",
        "22e80e8787a13a9c946b745a58fba6a0a4ffba658bbd50d8ecd9ffc2159af1b6",
    ),
    "up-continuous-33": (
        "stair-local",
        "03da1a99a161327fd6650e8b304934e182ab34b3ff9ec8cc0978c69700a6696c",
    ),
    "down-continuous-33": (
        "stair-local",
        "ca05856c8fe06f8d01e8a91bd300e118c28852b641a1c1154415c06a99c5eb3b",
    ),
    "chair-step-climbing-final": (
        "curb-chair",
        "69c545d85349033d1dbdc41041ea2baec730e3700f16dd1075c53fd08c99dc76",
    ),
}
self.assertTrue(EXPECTED.items() <= {
    spec.logical_name: (spec.family, spec.motion_sha256)
    for spec in CANDIDATE_SPECS
}.items())
self.assertNotIn("crane", {spec.logical_name for spec in CANDIDATE_SPECS})
```

Also test that a changed hash, symlink, missing file, duplicate logical name, or
path outside the declared source root fails before any archive is loaded.

- [ ] **Step 2: Run registry tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_registry
```

Expected: FAIL because `resources.g1_torch_terrain_builder.registry` is absent.

- [ ] **Step 3: Implement the immutable checked registry**

Define:

```python
@dataclass(frozen=True)
class SourceSpec:
    logical_name: str
    family: Literal["grail", "stair-local", "stair-karen", "curb-chair"]
    motion_relative_path: str
    motion_sha256: str
    terrain_adapter: Literal[
        "grail-usd", "fixed-staircase", "scaled-staircase-084",
        "karen-metadata", "chair-object",
    ]
    geometry_relative_paths: tuple[str, ...]
    expected_layout: str = "g1-29dof-isaaclab-v1"

@dataclass(frozen=True)
class ResolvedSource:
    spec: SourceSpec
    motion_path: Path
    geometry_paths: tuple[Path, ...]
    source_sha256: Mapping[str, str]
```

Populate `CANDIDATE_SPECS` with this exact local inventory in this order:

```python
PINNED_LOCAL_MOTIONS = (
    ("staircase-v0", "staircase:v0/motion.npz",
     "265133ea0b1e460f40a7e15921cb0ae0ca270cadbe73ad22709761c35198b925"),
    ("staircase-final", "staircase_final:v0/motion.npz",
     "846fc49ef5a6a628353f4644fb989b5341ee0a8f73f26aa72f1c09346a69065b"),
    ("staircase-final-2", "staircase_final_2:v0/motion.npz",
     "72ed7871b8ef1feffc650bd36107707bfcc293b79741edda1455894d0d6ffb67"),
    ("staircase-final-v3", "staircase_final_v3:v0/motion.npz",
     "22e80e8787a13a9c946b745a58fba6a0a4ffba658bbd50d8ecd9ffc2159af1b6"),
    ("staircase-side-stepto", "staircase_side_stepto:v1/motion.npz",
     "e92ad315d8ed212a9fe9cd656199e5bf1389993530ddc2850a4dbe113e0ef18c"),
    ("up-continuous-33", "up_continuous_33.npz:v0/motion.npz",
     "03da1a99a161327fd6650e8b304934e182ab34b3ff9ec8cc0978c69700a6696c"),
    ("down-continuous-33", "down_continuous_33.npz:v0/motion.npz",
     "ca05856c8fe06f8d01e8a91bd300e118c28852b641a1c1154415c06a99c5eb3b"),
    ("walk-up-33", "walk_up_33.npz:v0/motion.npz",
     "1158fb751c8d6ee1be19e878e12ab35042b7127559e44cb941279e0dbf0b378f"),
    ("walk-down-33", "walk_down_33.npz:v0/motion.npz",
     "62bd560bd2d00bf6cae97e33bc8d1dea299f8a57a147acf0a08c7f1b79814704"),
    ("up-continuous-karen", "up_continuous_v2_karen_stairs:v0/motion.npz",
     "fc6df7b83c41923675d64933267ec9c46ec9c6d8330f58159f9e04740376299b"),
    ("down-continuous-karen", "down_continuous_v2_karen_stairs:v0/motion.npz",
     "693bd8f1469eab388404e824d3079cde7533a42540fb9d47c03d3bee01c12068"),
    ("walk-up-karen", "walk_up_karen_stairs:v0/motion.npz",
     "4cee3698372634ab00352e112d121523f5db0ea412074fdc7e70279b295e33e5"),
    ("walk-down-karen", "walk_down_karen_stairs:v0/motion.npz",
     "cb4b352fc4595cb5014859e40a18bdbb6db27ca7f1157e44b0fa69f0197c97da"),
    ("chair-step-v0", "chair_step:v0/motion.npz",
     "25dd0116ae03fc0d6b5d6f9759e452a841f7c9c105035b3850864560101f3a85"),
    ("chair-step-v2", "chair_step:v2/motion.npz",
     "e145bb8e6ed9a82fb4194b5f0dae805c2d6762caf9b2c725ae72a10b0debfdc5"),
    ("chair-step-v3", "chair_step:v3/motion.npz",
     "ce3398dd6038766181ab10390d2f12a5f5b0f3fca7c05856f4b39f02afc18659"),
    ("chair-step-climbing-final", "chair_step_climbing_final:v1/motion.npz",
     "69c545d85349033d1dbdc41041ea2baec730e3700f16dd1075c53fd08c99dc76"),
    ("chair-step-tracking-2", "chair_step_tracking_2:v0/motion.npz",
     "5d5f78a6da8552db5b1297b8bd5ed566d00335e8b0d5db3d634e33069d118a3d"),
    ("chair-step-tracking-final", "chair_step_tracking_final:v6/motion.npz",
     "80cb05c3a94503084d24152cbd8f7367f0f5b8c36c2a0eb19d4ebb5ce5b6b283"),
    ("chair-step-truncated", "chair_step_truncated_converted.npz:v0/motion.npz",
     "f9a858ae4d9e0b097255bbb1dffcc6c214e3c8a49ae8701645a87a91c65b5256"),
)
```

Prepend the existing four `STAIR_BASES` as `grail` entries by adapting
`load_pinned_sources(grail_root)`. `resolve_registered_sources(source_root,
grail_root)` must require those four identities plus the exact 20 local entries
above.

Pin local fixed-stair geometry to:

```python
FIXED_STAIR_GEOMETRY_SHA256 = {
    "staircase/box_environment.xml":
        "d521bf511bce8511652af961d624579a312ec4350fbb2e5b5fab717e9b7965ca",
    "staircase/box_models/box1.obj":
        "412cab2eb06f0501d20009823c7127c5293257d87b1e86ffdb3da2c7697ec5de",
    "staircase/box_models/box2.obj":
        "8d3e9e204fa2646dbac9726ffe0da163f0b5811259981a76ee49c420079ff22c",
    "staircase/box_models/box3.obj":
        "dc8679af4fa022512613f42e340567fe9ec27697e71613f9d1b9695a767b1ca4",
}
```

The finalized `staircase-side-stepto` artifact was produced against the
0.8380952380952381-scale staircase at scene Y offset -0.025 m, not the
unscaled geometry. Pin
`staircase/multi_boxes_scaled_0.84_0.84_0.84.urdf` at
`78b560b01880e7202f5d36cf55cb66ab8ab98189fdb3f152c4406bf17557ac2f`,
pin the same three OBJ identities, and use `scaled-staircase-084` for this
candidate. Parse the checked collision mesh scales and OBJ bounds rather than
relaxing the contact oracle.

The chair adapter also pins
`rmr_tracking/.../chair_step/chair_step_env_cfg.py` to
`c6187a72e9ab3126faa33b516d4f9f99e78539e764105236fed669007547da74`
and reads `BOX_POSITION`/`BOX_SIZE` through a non-importing AST parser. The
contact-height oracle remains authoritative because the file documents that
the position was checked in simulation.

- [ ] **Step 4: Run registry tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write native-motion conversion tests**

Use temporary NPZ fixtures at 25, 33, and 50 Hz. Assert:

```python
motion = load_native_motion_50hz(path, source_fps=33.0)
self.assertEqual(motion.fps, 50)
self.assertEqual(motion.joint_position.shape[1:], (29,))
self.assertEqual(motion.body_position_world.shape[1:], (30, 3))
np.testing.assert_allclose(motion.body_position_world[[0, -1]],
                           expected_endpoints, atol=1e-6)
np.testing.assert_allclose(
    np.linalg.norm(motion.body_quaternion_world_wxyz, axis=-1),
    1.0, rtol=0.0, atol=1e-5,
)
```

Add failures for absent joint provenance, unknown FPS, non-unit quaternions,
wrong body count, fewer than 46 output frames, and endpoints lost during
resampling.

- [ ] **Step 6: Run motion tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_motion
```

Expected: FAIL because `load_native_motion_50hz` is absent.

- [ ] **Step 7: Implement strict native conversion**

Load only the exact fields:

```python
REQUIRED = {
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
}
```

Reuse `NativeMotionArrays`, `resample_vectors`,
`resample_quaternions_wxyz`, and `derive_velocities`. Preserve source endpoints,
derive output velocities at 50 Hz, and require either checked registry
provenance or exact `joint_names` matching the pinned G1 layout.

- [ ] **Step 8: Run focused and existing conversion tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_registry \
  tests.python.test_torch_terrain_motion \
  tests.python.test_torch_stair_conversion \
  tests.python.test_sonic_torch_motion_data
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add resources/g1_torch_terrain_builder \
  tests/python/test_torch_terrain_registry.py \
  tests/python/test_torch_terrain_motion.py
git commit -m "feat: register expanded terrain motion sources"
```

---

### Task 2: Family-specific authoritative terrain adapters

**Files:**
- Create: `resources/g1_torch_terrain_builder/terrain.py`
- Create: `tests/python/test_torch_terrain_adapters.py`
- Modify: `resources/g1_torch_stair_builder/surface.py`

**Interfaces:**
- Consumes: `ResolvedSource`, `NativeMotionArrays`, checked XML/OBJ/USD/metadata.
- Produces: `TerrainEvidence` and `build_source_terrain()`.

- [ ] **Step 1: Write fixed-staircase geometry tests**

Build the local staircase adapter from the checked geometry and assert the
documented surface:

```python
terrain = build_source_terrain(source, motion)
sample = terrain.grid.sample_xy(np.array([
    [0.10, 0.00], [-0.20, 0.00], [-0.55, 0.00], [0.50, 0.00],
], np.float32))
np.testing.assert_allclose(
    sample, [0.1778, 0.3556, 0.5334, 0.0], atol=0.011,
)
```

Assert Y outside `[-0.192, 0.430]` is floor height and that swapping X/Y fails
the oracle.

- [ ] **Step 2: Write Karen and chair adapter tests**

For Karen fixtures, require metadata to identify exact box geometry and scale.
For chair fixtures, require `object_pos_w`, `object_quat_w`, and checked object
geometry. Assert a motion-only chair archive returns structured rejection
`missing_authoritative_terrain`.

- [ ] **Step 3: Run adapter tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_adapters
```

Expected: FAIL because `TerrainEvidence` is absent.

- [ ] **Step 4: Extract shared height-grid operations without behavior change**

Keep `ZUpHeightGrid`, `ZUpTriangleSurface`, and rasterization semantics
unchanged. Move only generally reusable mesh/height helpers if needed, and
retain import compatibility from `g1_torch_stair_builder.surface`.

- [ ] **Step 5: Implement terrain adapters**

Define:

```python
@dataclass(frozen=True)
class TerrainEvidence:
    adapter: str
    grid: ZUpHeightGrid
    geometry_sha256: Mapping[str, str]
    motion_to_terrain_xy_yaw: tuple[float, float, float]

def build_source_terrain(
    source: ResolvedSource,
    motion: NativeMotionArrays,
) -> TerrainEvidence:
    ...
```

Dispatch only on the checked `terrain_adapter`. Build the fixed three-step
surface from exact documented dimensions, use GRAIL's existing USD path for
GRAIL entries, parse Karen metadata fail-closed, and reject chair entries
without exact geometry.

- [ ] **Step 6: Run adapter and existing surface tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_adapters \
  tests.python.test_torch_stair_surface \
  tests.python.test_sonic_torch_terrain_features
```

Expected: PASS with existing grid hashes unchanged.

- [ ] **Step 7: Commit Task 2**

```bash
git add resources/g1_torch_terrain_builder/terrain.py \
  resources/g1_torch_stair_builder/surface.py \
  tests/python/test_torch_terrain_adapters.py
git commit -m "feat: build authenticated terrain source grids"
```

---

### Task 3: Common FK, alignment, and duplicate admission

**Files:**
- Create: `resources/g1_torch_terrain_builder/admission.py`
- Create: `tests/python/test_torch_terrain_admission.py`

**Interfaces:**
- Consumes: `ResolvedSource`, `NativeMotionArrays`, `TerrainEvidence`, pinned G1 XML.
- Produces: `AdmissionReport` and `admit_candidate()`.

- [ ] **Step 1: Write admission tests**

Use analytic feet on two tread heights and assert:

```python
report = admit_candidate(source, motion, terrain, fk=fake_fk)
self.assertTrue(report.accepted)
self.assertGreaterEqual(report.contact_sample_count, 50)
self.assertLessEqual(report.contact_height_error_m["p95"], 0.035)
```

Add one test per rejection code:
`layout_unproven`, `fk_mismatch`, `insufficient_contact_samples`,
`insufficient_elevated_contact_samples`, `contact_alignment_failed`,
`exact_duplicate`, and
`missing_authoritative_terrain`.

- [ ] **Step 2: Run admission tests and verify RED**

Expected: FAIL because `admit_candidate` is absent.

- [ ] **Step 3: Implement immutable admission evidence**

Define:

```python
@dataclass(frozen=True)
class AdmissionReport:
    logical_name: str
    accepted: bool
    reason: str | None
    output_frames: int
    contact_sample_count: int
    elevated_contact_sample_count: int
    contact_height_error_m: Mapping[str, float]
    fk_position_error_m: Mapping[str, float]
    duplicate_of: str | None

def admit_candidate(
    source: ResolvedSource,
    motion: NativeMotionArrays,
    terrain: TerrainEvidence | None,
    *,
    fk: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    accepted_motion_hashes: Mapping[str, str],
) -> AdmissionReport:
    ...
```

Contact-like samples require ankle-origin clearance near 0.035 m and bounded
vertical foot speed. FK compares root/joint reconstructed bodies with published
body positions. Process candidates in registry order so exact-duplicate
ownership is deterministic. If the grid contains elevated terrain, require at
least 25 aligned elevated contact samples so flat approach/exit frames alone
cannot authenticate a stair or curb transform.

- [ ] **Step 4: Run admission tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_admission
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add resources/g1_torch_terrain_builder/admission.py \
  tests/python/test_torch_terrain_admission.py
git commit -m "feat: qualify terrain motion candidates"
```

---

### Task 4: Transactional expanded publisher and CLI

**Files:**
- Create: `resources/g1_torch_terrain_builder/publish.py`
- Create: `resources/build_g1_torch_terrain_corpus.py`
- Create: `tests/python/test_torch_terrain_publish.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: registered/admitted candidates, flat Takara motion, pinned G1 XML.
- Produces: `publish_expanded_corpus()` and `build/torch-terrain-expanded`.

- [ ] **Step 1: Write publisher tests**

Test a registry with accepted, rejected, and duplicate fixtures. Assert:

```python
manifest = publish_expanded_corpus(...)
self.assertEqual(manifest["schema"], "g1-torch-terrain-corpus/v1")
self.assertEqual(
    {entry["logical_name"] for entry in manifest["accepted_clips"]},
    {"flat-takara", "side-step"},
)
self.assertEqual(
    manifest["rejected_candidates"][0]["reason"],
    "missing_authoritative_terrain",
)
MotionFolder.load(output)
```

Force failure immediately before atomic rename and prove the prior output and
hashes remain unchanged with no staging directories.

- [ ] **Step 2: Run publisher tests and verify RED**

Expected: FAIL because `publish_expanded_corpus` is absent.

- [ ] **Step 3: Implement deterministic transactional publication**

Reuse `_deterministic_npz_bytes`, `_fsync_tree`, hash verification, and atomic
exchange from the small publisher. Write accepted clips under
`terrain/<family>/<logical-name>/`. Store complete admission evidence for both
accepted and rejected candidates. Validate every accepted terrain sidecar and
load the final tree with `MotionFolder.load` twice around the injectable
pre-commit validator.

- [ ] **Step 4: Implement CLI**

Support exact arguments:

```text
--source-root /home/ubuntu/Downloads/artifacts
--grail-root /home/ubuntu/datasets/GRAIL/data/stair_p1
--g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
--flat-motion /home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz
--output build/torch-terrain-expanded
--report build/torch-terrain-expanded-admission.json
```

Exit zero when publication succeeds even if individual candidates are
explicitly rejected; exit nonzero on publisher/integrity failure.

- [ ] **Step 5: Run publisher and existing publication tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_torch_terrain_publish \
  tests.python.test_torch_stair_publish \
  tests.python.test_sonic_torch_motion_data
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add .gitignore resources/build_g1_torch_terrain_corpus.py \
  resources/g1_torch_terrain_builder/publish.py \
  tests/python/test_torch_terrain_publish.py
git commit -m "feat: publish expanded terrain motion corpus"
```

---

### Task 5: Build and qualify the real local corpus

**Files:**
- Create: `docs/superpowers/results/2026-07-30-g1-expanded-terrain-corpus.md`
- Generated: `build/torch-terrain-expanded/`
- Generated: `build/torch-terrain-expanded-admission.json`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: authenticated local dataset and result record for the benchmark plan.

- [ ] **Step 1: Run the real publisher**

Run the CLI with the exact defaults from Task 4. Expected: JSON status
`PUBLISHED`, at least the four GRAIL clips and
`staircase-side-stepto` accepted, and every other candidate either accepted or
listed with one explicit rejection reason.

- [ ] **Step 2: Verify artifact integrity**

Load with `MotionFolder`, validate every manifest hash, print accepted frame
counts by family, and rerun publication. Expected: identical manifest and
output hashes on the second build.

- [ ] **Step 3: Run the full builder and Torch matcher tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_torch_*terrain*.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch*.py' -v
```

Expected: PASS; optional real-data oracles may skip only when explicitly marked.

- [ ] **Step 4: Record accepted and rejected evidence**

Write the result document with exact hashes, frame counts, adapter identities,
alignment error distributions, rejections, build duration, and any provenance
limitations. Do not relabel rejected data as usable.

- [ ] **Step 5: Commit Task 5**

```bash
git add docs/superpowers/results/2026-07-30-g1-expanded-terrain-corpus.md
git commit -m "docs: qualify expanded terrain motion corpus"
```
