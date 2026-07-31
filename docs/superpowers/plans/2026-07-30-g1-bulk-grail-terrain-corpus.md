# G1 Bulk GRAIL Terrain Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic representative and full native 50 Hz
terrain-motion corpora from all paired local GRAIL curb and stair recordings,
then run the existing dense kinematic matcher on them.

**Architecture:** A pinned-inventory adapter discovers candidates without
hardcoded clip names, deterministically selects balanced subsets, and resolves
only selected paired files into the existing authenticated publisher. The GRAIL
record loader accepts bounded static-terrain jitter, and a triangle-bounded
rasterizer removes the serial grid-point/triangle bottleneck without changing
height-grid semantics.

**Tech Stack:** Python 3, NumPy, joblib, OpenUSD `pxr`, MuJoCo kinematics,
PyTorch/CUDA, `unittest`, deterministic NPZ/JSON artifact publication.

## Global Constraints

- Work only on branch `research/g1-torch-terrain-kinematics` in the existing
  linked worktree.
- Preserve the qualified small and expanded publishers and their artifacts.
- Kinematics only: no SONIC, tracking, physics, obstacle avoidance, or learned
  depth in this plan.
- Source revision must equal
  `943946a972d5de2eb0d2ff214b236d0e43575fd7`.
- Requested partitions are exactly `curb`, `stair_p1`, and `stair_p2`.
- Output rate remains 50 Hz and layout remains
  `g1-29dof-isaaclab-v1`.
- Static object-pose excursion must not exceed 0.02 m translation or 2 degrees
  shortest-arc rotation.
- The faster rasterizer must produce exactly the same float32 node heights as
  the reference implementation on qualification fixtures and real sources.
- Use test-first red/green cycles and commit each independently reviewable task.

---

### Task 1: Generalize the strict GRAIL source loader

**Files:**
- Modify: `resources/g1_torch_stair_builder/corpus.py`
- Create: `tests/python/test_torch_stair_corpus.py`

**Interfaces:**
- Produces:
  `load_source(root: str | Path, base: str) -> PinnedStairSource`.
- Preserves:
  `load_pinned_sources(root: str | Path) -> tuple[PinnedStairSource, ...]`.
- Adds immutable audit fields
  `object_translation_excursion_m: float` and
  `object_rotation_excursion_rad: float` to `PinnedStairSource`.

- [ ] **Step 1: Write failing tests for bounded recorded-pose jitter**

Add fixtures that create a 250-frame object trajectory with 1 cm translation
and 1 degree quaternion motion, plus separate 2.1 cm and 2.1 degree failures.
Exercise the public single-source loader:

```python
source = load_source(root, "terrain_curbs__curb_000__000")
self.assertAlmostEqual(source.object_translation_excursion_m, 0.01)
self.assertAlmostEqual(
    source.object_rotation_excursion_rad, np.deg2rad(1.0)
)
np.testing.assert_array_equal(
    source.object_position_world, object_positions[0, 0]
)
```

Assert the two out-of-contract fixtures fail with messages containing
`translation excursion` and `rotation excursion`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_stair_corpus
```

Expected: failure because `load_source` and the excursion fields do not exist,
or because `_constant_row` rejects the bounded trajectory.

- [ ] **Step 3: Implement the minimal static-pose contract**

Add constants and helpers:

```python
_MAX_OBJECT_TRANSLATION_EXCURSION_M = 0.02
_MAX_OBJECT_ROTATION_EXCURSION_RAD = np.deg2rad(2.0)

def _static_object_pose(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    base: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    normalized_wxyz = _normalized_unrolled_wxyz(
        quaternions_wxyz[:, (1, 2, 3, 0)], base
    )
    translation = float(
        np.linalg.norm(positions - positions[:1], axis=1).max()
    )
    dots = np.clip(
        np.abs(np.sum(normalized_wxyz * normalized_wxyz[:1], axis=1)),
        0.0,
        1.0,
    )
    rotation = float((2.0 * np.arccos(dots)).max())
    if translation > _MAX_OBJECT_TRANSLATION_EXCURSION_M:
        raise ValueError(f"{base}: object translation excursion exceeds 0.02 m")
    if rotation > _MAX_OBJECT_ROTATION_EXCURSION_RAD:
        raise ValueError(f"{base}: object rotation excursion exceeds 2 degrees")
    quaternion_xyzw = normalized_wxyz[0, (1, 2, 3, 0)]
    return positions[0].copy(), quaternion_xyzw, translation, rotation
```

Keep the object's public quaternion field in xyzw, use frame zero as the
static transform, rename `_load_source` to `load_source`, and make
`load_pinned_sources` call it for the original four names.

- [ ] **Step 4: Run focused and dependent tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_stair_corpus \
  tests.python.test_torch_stair_conversion \
  tests.python.test_torch_stair_surface \
  tests.python.test_torch_stair_publish
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_torch_stair_builder/corpus.py \
  tests/python/test_torch_stair_corpus.py
git commit -m "feat: load bounded-static GRAIL terrain records"
```

---

### Task 2: Add pinned bulk inventory discovery and balanced selection

**Files:**
- Create: `resources/g1_torch_terrain_builder/bulk_grail.py`
- Create: `tests/python/test_torch_bulk_grail.py`

**Interfaces:**
- Produces immutable `BulkGrailCandidate`.
- Produces:
  `discover_bulk_grail_candidates(dataset_root, partitions)`.
- Produces:
  `select_bulk_grail_candidates(candidates, limit_per_partition, seed)`.
- Produces:
  `resolve_bulk_grail_candidates(dataset_root, candidates)`.
- The final resolver returns `tuple[ResolvedSource, ...]` compatible with
  `publish_expanded_corpus(..., resolved_sources=...)`.

- [ ] **Step 1: Write failing inventory contract tests**

Create a temporary inventory with one record in each requested partition and
real tiny robot/object/USD files. Assert:

```python
candidates = discover_bulk_grail_candidates(
    root, ("curb", "stair_p1", "stair_p2")
)
self.assertEqual([item.partition for item in candidates],
                 ["curb", "stair_p1", "stair_p2"])
self.assertTrue(all(item.stem for item in candidates))
```

Add individual tests for wrong schema, wrong repository revision, duplicate
manifest path, missing paired object, symlink escape, byte-size mismatch, hash
mismatch, unsupported partition, and duplicate logical identity. Include the
real inventory defect: curb object hashes are absent but are computed from the
required local paired file.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_bulk_grail
```

Expected: import failure because `bulk_grail.py` does not exist.

- [ ] **Step 3: Implement fail-closed discovery and resolution**

Define:

```python
_SCHEMA = "g1-grail-terrain-inventory/v1"
_REPOSITORY_ID = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
_REVISION = "943946a972d5de2eb0d2ff214b236d0e43575fd7"
_PARTITIONS = frozenset(("curb", "stair_p1", "stair_p2"))

@dataclass(frozen=True)
class BulkGrailCandidate:
    partition: str
    stem: str
    robot_relative_path: str
    object_relative_path: str
    usd_relative_path: str
    robot_sha256: str
    usd_sha256: str

    @property
    def logical_name(self) -> str:
        identity = hashlib.sha256(
            f"{self.partition}\0{self.stem}".encode("utf-8")
        ).hexdigest()[:20]
        return f"grail-{self.partition}-{identity}"
```

Parse inventory entries into a unique path map. Use robot entries as the
authoritative clip set and derive exact paired paths by replacing the modality
directory while preserving the stem. Validate selected files with resolved
path containment, no symlinks, expected bytes, and SHA-256. Create
`SourceSpec(family="grail", source_adapter="grail-record",
terrain_adapter="grail-usd", ...)` relative to the dataset root.

- [ ] **Step 4: Write failing deterministic selection tests**

Create 17 candidates in each partition. Assert:

```python
first = select_bulk_grail_candidates(
    reversed(candidates), limit_per_partition=5, seed=7
)
second = select_bulk_grail_candidates(
    candidates, limit_per_partition=5, seed=7
)
self.assertEqual(first, second)
self.assertEqual(Counter(x.partition for x in first),
                 {"curb": 5, "stair_p1": 5, "stair_p2": 5})
self.assertNotEqual(
    first,
    select_bulk_grail_candidates(
        candidates, limit_per_partition=5, seed=8
    ),
)
```

Assert `None` selects all, zero/negative limits fail, duplicate candidates
fail, and a limit larger than a partition returns the complete partition.

- [ ] **Step 5: Implement interval-balanced selection**

Sort each partition by stem, divide its indices into `N` integer intervals,
and choose the minimum seeded SHA-256 rank inside every interval:

```python
rank = hashlib.sha256(
    f"{seed}\0{candidate.partition}\0{candidate.stem}".encode("utf-8")
).digest()
```

Return the final tuple sorted by `(partition, stem)` so filesystem enumeration
order never affects the result.

- [ ] **Step 6: Run focused tests and the real read-only inventory smoke**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_bulk_grail

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -c \
  'from resources.g1_torch_terrain_builder.bulk_grail import *; from collections import Counter; r="/home/ubuntu/datasets/GRAIL"; c=discover_bulk_grail_candidates(r, ("curb","stair_p1","stair_p2")); print(len(c), Counter(x.partition for x in c))'
```

Expected: tests pass and the smoke prints 13,957 total with counts 1,769,
6,094, and 6,094.

- [ ] **Step 7: Commit**

```bash
git add resources/g1_torch_terrain_builder/bulk_grail.py \
  tests/python/test_torch_bulk_grail.py
git commit -m "feat: index pinned bulk GRAIL terrain sources"
```

---

### Task 3: Replace the terrain rasterization bottleneck

**Files:**
- Modify: `resources/g1_torch_stair_builder/surface.py`
- Modify: `tests/python/test_torch_stair_surface.py`

**Interfaces:**
- Preserves:
  `rasterize_zup_surface(surface, bounds_xy, cell_size_m) -> ZUpHeightGrid`.
- Adds private `_rasterize_zup_surface_reference(...)` solely for
  qualification tests and benchmarking.

- [ ] **Step 1: Write failing parity tests**

Add helpers for overlapping, reversed-winding, vertical, boundary-touching,
and seeded-random triangle surfaces. For each fixture, assert:

```python
expected = _rasterize_zup_surface_reference(
    surface, bounds_xy=bounds, cell_size_m=cell
)
actual = rasterize_zup_surface(
    surface, bounds_xy=bounds, cell_size_m=cell
)
np.testing.assert_array_equal(actual.origin_xy, expected.origin_xy)
self.assertEqual(actual.cell_size_m, expected.cell_size_m)
np.testing.assert_array_equal(actual.height_z, expected.height_z)
```

The first red test should monkeypatch `ZUpTriangleSurface.height_xy` to fail.
The optimized implementation must not invoke the row-wise query method.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_stair_surface
```

Expected: the monkeypatched test fails because the current rasterizer calls
`height_xy` once per row.

- [ ] **Step 3: Preserve the reference and implement triangle-bounded updates**

Move the current body unchanged to
`_rasterize_zup_surface_reference`. Share grid geometry validation through a
small `_grid_geometry` helper. The optimized body initializes a float64 height
array to `surface.exterior_height_z`, then for every triangle:

```python
projected = triangle[:, :2]
v0 = projected[1] - projected[0]
v1 = projected[2] - projected[0]
determinant = v0[0] * v1[1] - v0[1] * v1[0]
if abs(determinant) <= _PROJECTED_EPSILON:
    continue

# Convert projected bounds plus the existing tolerance to clipped grid indices.
# Build only that rectangle, evaluate the same u/v/w equations, and update:
window = heights[iy0:iy1 + 1, ix0:ix1 + 1]
window[inside] = np.maximum(window[inside], z[inside])
```

Convert to float32 only after every triangle has been accumulated, matching the
reference's rounding point.

- [ ] **Step 4: Verify parity on tests and real sources**

Run unit tests, then compare reference and optimized grids for one curb, one
procedural stair, and the largest local stair USD. Print timings and require
bitwise `height_z` equality.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_stair_surface \
  tests.python.test_torch_stair_publish \
  tests.python.test_torch_terrain_publish
```

Expected: all pass; real-source optimized construction is materially faster
than the 2--15 second reference measurements.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_torch_stair_builder/surface.py \
  tests/python/test_torch_stair_surface.py
git commit -m "perf: rasterize GRAIL terrain by triangle bounds"
```

---

### Task 4: Connect bulk sources to the authenticated publisher

**Files:**
- Modify: `resources/g1_torch_terrain_builder/publish.py`
- Modify: `resources/g1_torch_terrain_builder/terrain.py`
- Create: `resources/build_g1_torch_grail_corpus.py`
- Modify: `tests/python/test_torch_terrain_publish.py`
- Create: `tests/python/test_build_torch_grail_corpus_cli.py`

**Interfaces:**
- `publish_expanded_corpus` gains optional
  `corpus_metadata: Mapping[str, object] | None = None`.
- The bulk CLI accepts `--dataset-root`, repeated `--partition`,
  `--limit-per-partition`, `--selection-seed`, `--g1-xml`, `--flat-motion`,
  `--output`, and `--report`.

- [ ] **Step 1: Write failing generalized-loader integration tests**

Construct a `ResolvedSource` whose GRAIL motion is not one of `STAIR_BASES`.
Assert default motion loading calls `load_source(source.motion_path.parents[1],
source.motion_path.stem)`. Assert `build_source_terrain` does the same for the
USD adapter. The test must fail against the old four-record lookup.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_terrain_publish
```

Expected: failure containing `registered GRAIL source is not pinned`.

- [ ] **Step 3: Use the public single-source loader**

Replace both `load_pinned_sources(...)/next(...)` lookups with:

```python
source_record = load_source(
    source.motion_path.parents[1], source.motion_path.stem
)
```

Keep all existing publisher and terrain-adapter type checks.

- [ ] **Step 4: Write failing publisher metadata and CLI tests**

Assert a supplied metadata mapping appears exactly as `bulk_grail` in the
transactionally validated manifest. For the CLI, mock discovery, selection,
resolution, and publication and assert:

```python
self.assertEqual(call["partitions"],
                 ("curb", "stair_p1", "stair_p2"))
self.assertEqual(call["limit_per_partition"], 2)
self.assertEqual(call["seed"], 11)
self.assertEqual(published_metadata["available_by_partition"],
                 {"curb": 3, "stair_p1": 3, "stair_p2": 3})
```

Also assert repeated `--partition` values are rejected, an omitted limit means
all, and the report is written atomically.

- [ ] **Step 5: Implement manifest metadata and the bulk CLI**

The CLI flow is:

```python
candidates = discover_bulk_grail_candidates(
    args.dataset_root, tuple(args.partition)
)
selected = select_bulk_grail_candidates(
    candidates,
    limit_per_partition=args.limit_per_partition,
    seed=args.selection_seed,
)
resolved = resolve_bulk_grail_candidates(args.dataset_root, selected)
manifest = publish_expanded_corpus(
    output=args.output,
    source_root=args.source_root,
    grail_root=args.dataset_root,
    g1_xml=args.g1_xml,
    flat_motion=args.flat_motion,
    resolved_sources=resolved,
    corpus_metadata=selection_descriptor(...),
)
```

`selection_descriptor` records inventory schema/revision/SHA, seed, limit,
available counts, selected counts, and selected logical identities.

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_torch_bulk_grail \
  tests.python.test_torch_terrain_publish \
  tests.python.test_build_torch_grail_corpus_cli \
  tests.python.test_sonic_torch_terrain_features \
  tests.python.test_sonic_torch_motion_matcher
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add resources/g1_torch_terrain_builder/publish.py \
  resources/g1_torch_terrain_builder/terrain.py \
  resources/build_g1_torch_grail_corpus.py \
  tests/python/test_torch_terrain_publish.py \
  tests/python/test_build_torch_grail_corpus_cli.py
git commit -m "feat: publish balanced bulk GRAIL terrain corpora"
```

---

### Task 5: Build, qualify, and interactively evaluate staged corpora

**Files:**
- Create: `sonic/configs/experiments/torch_grail_representative.json`
- Create: `docs/superpowers/results/2026-07-30-g1-bulk-grail-terrain-results.md`

**Interfaces:**
- Smoke artifact:
  `build/torch-grail-terrain-smoke`.
- Representative artifact:
  `build/torch-grail-terrain-representative`.
- Full artifact:
  `build/torch-grail-terrain-full`.

- [ ] **Step 1: Run the complete Python regression baseline**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest discover -s tests/python -v
```

Expected: all tests pass. Record exact count and duration.

- [ ] **Step 2: Build a two-per-partition smoke corpus**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/build_g1_torch_grail_corpus.py \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --limit-per-partition 2 \
  --output build/torch-grail-terrain-smoke \
  --report build/torch-grail-terrain-smoke-report.json
```

Expected: seven accepted clips including flat, zero unreported exceptions, and
a loadable `TerrainDataset` on CPU and `cuda:1`.

- [ ] **Step 3: Measure smoke publication and runtime costs**

Record wall time, accepted/rejected counts, output bytes, CPU load time,
`cuda:1` load time, peak GPU allocation, database build time, and 1,000 matcher
steps with p50/p95/p99 latency. If the rasterizer is still dominant, stop and
optimize it before increasing the selection.

- [ ] **Step 4: Build a balanced representative corpus**

Start at 64 clips per partition:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/build_g1_torch_grail_corpus.py \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --limit-per-partition 64 \
  --output build/torch-grail-terrain-representative \
  --report build/torch-grail-terrain-representative-report.json
```

Increase to 128 per partition only when build/load/search measurements leave a
comfortable interactive margin. Record selection and rejection reasons.

- [ ] **Step 5: Create the representative viewer configuration**

Copy the checked dense matcher settings from
`sonic/configs/experiments/torch_terrain_expanded.json`. Set `query_scene` to
one accepted stair clip and `reset_clip` to `flat/motion.npz`. Do not change
weights, cadence, continuity, or command dynamics during the first comparison.

- [ ] **Step 6: Launch the kinematic viewer**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m \
  mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-grail-terrain-representative \
  --config sonic/configs/experiments/torch_grail_representative.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:1
```

Expected: visible terrain, responsive WASD control, Backspace reset, no physics
or tracking process, and no out-of-domain exception.

- [ ] **Step 7: Run adversarial routes and document evidence**

Exercise side approaches, cross-tread traversal, turns, diagonals, side exits,
reversals, and stop/restart. Record freezes, source/frame loops, terrain cost,
selection families, and matcher latency. Compare qualitatively and
quantitatively with `build/torch-terrain-expanded-combined-v42`.

- [ ] **Step 8: Build the full corpus if staged measurements pass**

Omit `--limit-per-partition`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  resources/build_g1_torch_grail_corpus.py \
  --dataset-root /home/ubuntu/datasets/GRAIL \
  --output build/torch-grail-terrain-full \
  --report build/torch-grail-terrain-full-report.json
```

Expected source selection count: 13,957. Document admitted count, every
structured rejection, build/load/search measurements, and whether the full
corpus changes the representative behavior.

- [ ] **Step 9: Run completion verification**

Run:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -m unittest discover -s tests/python -v
git status --short --branch
```

Expected: no diff errors, all tests pass, and only intentional source changes
plus ignored build artifacts.

- [ ] **Step 10: Commit results and configuration**

```bash
git add sonic/configs/experiments/torch_grail_representative.json \
  docs/superpowers/results/2026-07-30-g1-bulk-grail-terrain-results.md
git commit -m "docs: qualify bulk GRAIL terrain corpus"
```
