# Mixed Table/Ground Grasp Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and visualize one GRAIL tabletop-plus-ground pickup corpus whose motions are selected continuously from the requested world-grasp height and pose.

**Architecture:** Extend the schema-v1 importer to ingest both pickup categories while encoding ground support as virtual floor metadata, then publish a separate combined artifact. Refactor grasp search to score a single Contact pose before expanding complete trajectories, and load only viewer-required binary fields so the larger corpus stays within the current DCV memory envelope.

**Tech Stack:** Python 3, NumPy, joblib, OpenUSD, MuJoCo G1 FK, C++17, raylib, `unittest`, Make, schema-v1 interaction binaries.

## Global Constraints

- Retrieval has no Table/Ground mode switch, category gate, category penalty, category quota, or fixed height threshold.
- Keep the existing 0.12 m grasp-position compatibility envelope.
- Preserve yaw/XZ body alignment and recorded world Y.
- Extract clip start through final contiguous Lift; omit Hold.
- Ground virtual support is source Z-up position `[0, 0, -0.02]`, size `[20, 20, 0.04]`, and identity rotation, producing canonical floor top Y=0.
- Keep the schema-v1 binary layout unchanged.
- Preserve existing single-root tabletop schema and manifest structure.
- Build the combined artifact at `build/smart-pickup/table-ground-pack`; never overwrite `build/smart-pickup/full-pack`.
- Keep Enter-only recomputation, selected-only full-pose caching, and background path stride.
- Do not add mesh, terrain, diffusion, controller, or screenshot code to the standalone viewer.
- The live combined viewer must use no more than 1.1 GB steady-state RSS.

---

### Task 1: Category-Aware Source Discovery and Ground Support

**Files:**
- Modify: `resources/g1_interaction_builder/sources.py`
- Modify: `tests/python/interaction_fixture.py`
- Modify: `tests/python/test_interaction_sources.py`

**Interfaces:**
- Consumes: existing `SourcePaths`, native GRAIL modality layout, and `load_raw_interaction`.
- Produces: `sequence_parts(sequence_id: str) -> tuple[str, str]`, extended `object_id_from_sequence`, `discover_source_paths_many(roots: Sequence[Path]) -> list[SourcePaths]`, and category-specific strict metadata loading.

- [ ] **Step 1: Extend the fixture and write failing parser/discovery/support tests**

Add `ground: bool = False` to `write_source_fixture`. When true, emit a `pickup_ground` sequence and native metadata containing only `object_name`. Add tests equivalent to:

```python
def test_sequence_parser_accepts_exact_table_and_ground_ids(self):
    self.assertEqual(
        sequence_parts("pickup_table__cup_2__001"),
        ("pickup_table", "cup_2"),
    )
    self.assertEqual(
        sequence_parts("pickup_ground__cup_2__001"),
        ("pickup_ground", "cup_2"),
    )
    for invalid in (
        "pickup_ground__cup_2__01",
        "pickup_ground__cup_2__0001",
        "putdown_ground__cup_2__001",
    ):
        with self.assertRaisesRegex(ValueError, "invalid pickup sequence id"):
            sequence_parts(invalid)

def test_discovers_multiple_roots_and_rejects_duplicate_sequences(self):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        table = write_source_fixture(
            base / "table", sequence_id="pickup_table__cup_2__001"
        )
        ground = write_source_fixture(
            base / "ground", sequence_id="pickup_ground__cup_2__001",
            ground=True,
        )
        found = discover_source_paths_many([base / "table", base / "ground"])
        self.assertEqual(
            [item.sequence_id for item in found],
            [ground.sequence_id, table.sequence_id],
        )

def test_ground_meta_synthesizes_exact_virtual_floor(self):
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_source_fixture(Path(tmp), ground=True)
        clip = load_raw_interaction(paths, np.ones(3, np.float32))
        np.testing.assert_array_equal(
            clip.table_position, np.array([0.0, 0.0, -0.02], np.float32)
        )
        np.testing.assert_array_equal(
            clip.table_size, np.array([20.0, 20.0, 0.04], np.float32)
        )
        np.testing.assert_array_equal(
            clip.table_rotation, np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        )
```

Also test that ground metadata containing any subset of tabletop fields raises `SourceValidationError` with `invalid_source_record`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m unittest tests.python.test_interaction_sources -v
```

Expected: failures for missing `sequence_parts`, missing `discover_source_paths_many`, rejection of `pickup_ground`, and missing virtual-floor synthesis.

- [ ] **Step 3: Implement exact category parsing, multi-root discovery, and strict metadata**

Implement these contracts in `sources.py`:

```python
SEQUENCE_RE = re.compile(
    r"^(pickup_table|pickup_ground)__(.+)__([0-9]{3})$"
)
GROUND_SUPPORT_POSITION = np.array([0.0, 0.0, -0.02], np.float32)
GROUND_SUPPORT_ROTATION_XYZW = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
GROUND_SUPPORT_SIZE = np.array([20.0, 20.0, 0.04], np.float32)

def sequence_parts(sequence_id: str) -> tuple[str, str]:
    match = SEQUENCE_RE.fullmatch(sequence_id)
    if match is None:
        raise ValueError(f"invalid pickup sequence id: {sequence_id}")
    return match.group(1), match.group(2)

def object_id_from_sequence(sequence_id: str) -> str:
    return sequence_parts(sequence_id)[1]

def discover_source_paths_many(roots: Sequence[Path]) -> list[SourcePaths]:
    discovered = [item for root in roots for item in discover_source_paths(root)]
    sequence_ids = [item.sequence_id for item in discovered]
    duplicates = sorted(
        sequence for sequence, count in Counter(sequence_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate sequence ids across roots: {duplicates}")
    return sorted(discovered, key=lambda item: item.sequence_id)
```

Make `_meta_record` branch on `sequence_parts(sequence_id)[0]`. Keep the existing exact tabletop key set. For ground, accept only `{"object_name"}` in either direct or one-record wrapped form and return a normalized dictionary with the three virtual support arrays. Do not accept partial tabletop keys.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.python.test_interaction_sources -v
git diff --check
```

Expected: all interaction-source tests pass and `git diff --check` exits 0.

- [ ] **Step 5: Commit the source-ingestion unit**

```bash
git add resources/g1_interaction_builder/sources.py \
  tests/python/interaction_fixture.py \
  tests/python/test_interaction_sources.py
git commit -m "feat: ingest GRAIL ground pickups"
```

### Task 2: Multi-Root Combined Artifact Publication

**Files:**
- Modify: `resources/build_g1_interaction_database.py`
- Modify: `resources/g1_interaction_builder/build.py`
- Modify: `resources/g1_interaction_builder/artifacts.py`
- Modify: `Makefile`
- Modify: `tests/python/test_interaction_build_cli.py`
- Modify: `tests/python/test_interaction_artifacts.py`

**Interfaces:**
- Consumes: `discover_source_paths_many`, existing object split, artifact writer, and schema-v1 validator.
- Produces: repeated `--source-root`, conditional additive manifest provenance, cross-category dimension validation, and `mixed-interaction-pack`.

- [ ] **Step 1: Write failing CLI, manifest, and dimension tests**

Add tests asserting:

```python
args = build_cli.parse_args([
    "--source-root", "data/pickup_table",
    "--source-root", "data/pickup_ground",
    "--g1-xml", "g1.xml",
])
self.assertEqual(
    args.source_root,
    [Path("data/pickup_table"), Path("data/pickup_ground")],
)
```

Build a fast mixed fixture with the same `cup_2` object in both roots and another object so a nonempty held-out split is possible. Assert the manifest has sorted absolute `source_roots`, `source_category` values for both categories, and `source_root` equal to their absolute common parent. Assert a one-root fixture omits both additive fields.

Run `split_objects` on labeled table and ground clips sharing `cup_2` and assert
both clips resolve to the same partition: either both are present in the
database artifact or both are absent as held-out clips. Assert
`set(database_objects).isdisjoint(heldout_objects)`.

Patch representative USD sizes so matching table/ground bounds pass and a difference greater than `1e-3` raises `ValueError("inconsistent cross-category USD bounds")`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python3 -m unittest \
  tests.python.test_interaction_build_cli \
  tests.python.test_interaction_artifacts -v
```

Expected: failures because `--source-root` is singular and mixed provenance/dimension checks do not exist.

- [ ] **Step 3: Implement multi-root build and conditional provenance**

Change argument parsing to:

```python
parser.add_argument(
    "--source-root", type=Path, action="append", required=True
)
```

In `run`, resolve and sort unique roots, call `discover_source_paths_many`, and pass the root tuple to `build_manifest`. For one root, preserve the old `source_root` and clip dictionaries. For multiple roots, use `Path(os.path.commonpath(...))`, add sorted `source_roots`, and add `source_category = sequence_parts(sequence_id)[0]` per clip.

Read one deterministic USD representative per `(object_id, source_category)`. Require cross-category values to satisfy `np.allclose(left, right, rtol=0.0, atol=1e-3)` and return one dimension per object ID. Validate additive provenance only when present and continue accepting old manifests.

Add:

```make
GRAIL_PICKUP_GROUND_ROOT ?= /home/ubuntu/datasets/GRAIL/data/pickup_ground
MIXED_INTERACTION_PACK ?= build/smart-pickup/table-ground-pack

.PHONY: mixed-interaction-pack
mixed-interaction-pack:
	python -m resources.build_g1_interaction_database \
	  --source-root "$(GRAIL_PICKUP_ROOT)" \
	  --source-root "$(GRAIL_PICKUP_GROUND_ROOT)" \
	  --g1-xml "$(G1_XML)" \
	  --output "$(MIXED_INTERACTION_PACK)" \
	  --target-fps 25 --allow-rejections
	python -m resources.validate_g1_interaction_database \
	  --input "$(MIXED_INTERACTION_PACK)"
```

- [ ] **Step 4: Run focused and safe regression tests**

```bash
python3 -m unittest \
  tests.python.test_interaction_sources \
  tests.python.test_interaction_build_cli \
  tests.python.test_interaction_artifacts \
  tests.python.test_hand_trajectory_viewer -v
git diff --check
```

Expected: all tests pass and existing single-root fixtures remain green.

- [ ] **Step 5: Commit combined publication support**

```bash
git add resources/build_g1_interaction_database.py \
  resources/g1_interaction_builder/build.py \
  resources/g1_interaction_builder/artifacts.py \
  Makefile tests/python/test_interaction_build_cli.py \
  tests/python/test_interaction_artifacts.py
git commit -m "feat: publish mixed pickup interaction packs"
```

### Task 3: Continuous Height Retrieval with Early Contact Scoring

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Modify: `tests/cpp/test_interaction_hand_trajectories.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`

**Interfaces:**
- Consumes: combined schema-v1 database and existing upright grasp alignment.
- Produces: `SupportKind`, `support_kind(const Database&, size_t)`, and Contact scoring before full trajectory expansion.

- [ ] **Step 1: Write failing support-label and mixed-height tests**

Extend the synthetic database fixture so each clip can set support position and size. Add tests equivalent to:

```cpp
require(interaction::support_kind(database, table_clip) ==
            interaction::SupportKind::Table,
        "table clip mislabeled");
require(interaction::support_kind(database, ground_clip) ==
            interaction::SupportKind::Ground,
        "exact virtual floor was not labeled ground");
```

Create table and ground clips with the same Contact wrist pose. Assert both are selected and sorted only by `(cost, clip)`. Move `query.grasp_world_position.y` more than 0.12 m from one clip and assert that clip is absent without a category condition.

Add a ground-labeled synthetic clip with at least twelve Approach samples and
assert `start_frame`, `reach_point`, `contact_point`, final `lift_frame`, and
path size still span clip start through final contiguous Lift.

Add this source-contract assertion:

```python
trajectory_source = TRAJECTORY_SOURCE_PATH.read_text(encoding="utf-8")
self.assertLess(
    trajectory_source.index("if (position_error >"),
    trajectory_source.index(
        "for (int32_t frame = trajectory.start_frame;"
    ),
)
```

- [ ] **Step 2: Build the focused test and verify RED**

```bash
make -B build/tests/test_interaction_hand_trajectories
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
```

Expected: compile failure for missing `SupportKind`/`support_kind` or assertion failure because path expansion still precedes Contact rejection.

- [ ] **Step 3: Implement support diagnostics and two-stage construction**

Add:

```cpp
enum class SupportKind : uint8_t { Table = 0U, Ground = 1U };
SupportKind support_kind(const Database& database, size_t clip);
```

Add `SupportKind support = SupportKind::Table;` to `HandTrajectory`.

Classify Ground only when canonical support metadata matches within `1e-4F`: position `(0,-0.02,0)`, size `(20,0.04,20)`, identity rotation, and top Y=0. Everything else is Table. Store the diagnostic on `HandTrajectory`.

Refactor `select_hand_trajectories` in this order:

```cpp
const ClipPhases phases = clip_phases(database, clip);
const Transform source_object = object_transform(database, phases.contact - 1);
const WorldPose contact_world = world_pose(
    pose_at_frame(database, phases.contact));
const Transform source_contact = hand_transform(contact_world, query.hand);
const Transform alignment = upright_grasp_alignment(source_contact, query);
const Transform mapped_contact = compose(alignment, source_contact);
const float position_error = length(
    mapped_contact.position - query.grasp_world_position);
const float orientation_error = query.constrain_grasp_orientation
    ? quaternion_angle(mapped_contact.rotation, query.grasp_world_rotation)
    : 0.0F;
if (position_error > config.maximum_grasp_position_error_m ||
    orientation_error > config.maximum_grasp_orientation_error_radians) {
    continue;
}
HandTrajectory trajectory{};
trajectory.clip = static_cast<int32_t>(clip);
trajectory.start_frame = database.range_starts.at(clip);
trajectory.reach_frame = phases.reach;
trajectory.contact_frame = phases.contact;
trajectory.lift_frame = phases.last_lift;
```

Do not include `SupportKind` in membership, alignment, cost, or sort keys.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
git diff --check
```

Expected: build and test exit 0, with mixed support clips coexisting at overlap height.

- [ ] **Step 5: Commit retrieval changes**

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp \
  tests/cpp/test_interaction_hand_trajectories.cpp \
  tests/python/test_hand_trajectory_viewer.py
git commit -m "perf: prefilter pickup trajectories by grasp height"
```

### Task 4: Focused Low-Memory Trajectory Database Loader

**Files:**
- Create: `interaction_trajectory_database.h`
- Create: `tests/cpp/test_interaction_trajectory_database.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: unchanged `G1INTDB1` binary layout and `interaction::Database`.
- Produces: `Database load_trajectory_database(const std::filesystem::path&)` retaining only viewer-required fields.

- [ ] **Step 1: Write a failing compact-loader equivalence test**

Create a minimal valid schema-v1 binary fixture using the same field order as `load_database`. Load it both ways and assert equality for:

```text
parents, range_starts, range_stops, positions, rotations, phases,
active_hands, object_positions, object_rotations, table_positions,
table_rotations, table_sizes, object_dimensions,
grasp_positions_object, grasp_rotations_object,
approach_directions_object
```

Assert these compact fields are empty:

```text
velocities, angular_velocities, foot_contacts, hand_contacts,
hand_dof, hand_dof_velocities, time_to_contact,
object_velocities, object_angular_velocities, source_frames
```

Add malformed fixtures for truncation inside a skipped array, extra trailing bytes, invalid ranges, nonmonotonic/missing phases, and invalid retained quaternions.

- [ ] **Step 2: Build the test and verify RED**

```bash
make -B build/tests/test_interaction_trajectory_database
```

Expected: failure because `interaction_trajectory_database.h` and its loader do not exist.

- [ ] **Step 3: Implement checked field skipping and focused validation**

Implement a header-only loader beside `interaction_database.h`. Reuse its magic/header/count helpers. Add:

```cpp
template<class T>
void skip_vector(
    std::istream& input,
    size_t input_size,
    size_t count,
    std::string_view label);

Database load_trajectory_database(const std::filesystem::path& path);
```

Before reading arrays, compute the exact schema-v1 expected byte count with checked multiplication and require it equals `file_size(path)`. Read only retained fields and `seekg` over the rest. Require exact EOF position and validate retained finiteness, unit quaternions, support/object dimensions, clip ranges, active-hand values, and monotonic required phases. Do not call full `validate_database`, because skipped vectors are intentionally empty.

Add the test target to `CPP_TEST_BINS` and a direct rule using `CPP_TEST_FLAGS`.

- [ ] **Step 4: Run compact-loader and existing database tests**

```bash
make -B build/tests/test_interaction_trajectory_database \
  build/tests/test_interaction_database
./build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_database
git diff --check
```

Expected: both binaries exit 0.

- [ ] **Step 5: Commit the focused loader**

```bash
git add interaction_trajectory_database.h \
  tests/cpp/test_interaction_trajectory_database.cpp Makefile
git commit -m "perf: load compact trajectory viewer data"
```

### Task 5: Mixed-Support Viewer Integration

**Files:**
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `load_trajectory_database`, `HandTrajectory::support`, and the combined pack target.
- Produces: tabletop-preferred startup, `TABLE`/`GROUND` labels, and unchanged staged controls/collision behavior.

- [ ] **Step 1: Write failing viewer source-contract tests**

Require:

```python
self.assertIn('#include "interaction_trajectory_database.h"', source)
self.assertIn("load_trajectory_database(", source)
self.assertNotIn("interaction::load_database(", source)
self.assertIn("SupportKind::Table", source)
self.assertIn("SupportKind::Ground", source)
self.assertIn('"GROUND"', source)
self.assertIn('"TABLE"', source)
self.assertIn("MIXED_INTERACTION_PACK", MAKEFILE)
self.assertIn("mixed-interaction-pack", MAKEFILE)
```

Retain every existing assertion for Enter-only search, selected-only poses, path stride, collision APIs, and forbidden mesh/terrain/diffusion/controller dependencies.

- [ ] **Step 2: Run viewer tests and verify RED**

```bash
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
```

Expected: failures for the old full loader and absent support label.

- [ ] **Step 3: Implement focused loading and support-aware presentation**

Replace the database load with:

```cpp
const interaction::Database database =
    interaction::load_trajectory_database(
        pack / "interaction_database.bin");
```

Change canonical grasp discovery to retain the first valid Contact clip as a fallback but return the first clip whose `support_kind(database, clip)` is `Table`. Do not change retrieval membership.

Add:

```cpp
const char* support_name(interaction::SupportKind support) {
    return support == interaction::SupportKind::Ground
        ? "GROUND" : "TABLE";
}
```

Display support next to `SAFE` and phase. Keep the visible table, ground grid, target-object collision, table collision, and no floor slab collision. Keep the environment-based pack path so launch selects the combined pack without hardcoding it.

Add `interaction_trajectory_database.h` to the viewer target dependencies.

- [ ] **Step 4: Run viewer, C++, and release-build checks**

```bash
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_trajectory_database \
  build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit viewer integration**

```bash
git add hand_trajectory_viewer.cpp \
  tests/python/test_hand_trajectory_viewer.py Makefile
git commit -m "feat: visualize table and ground pickup searches"
```

### Task 6: Build, Validate, Review, and Launch the Combined Corpus

**Files:**
- Generated only: `build/smart-pickup/table-ground-pack/*`
- No tracked source changes expected unless verification exposes a defect.

**Interfaces:**
- Consumes: Tasks 1--5 and local GRAIL table/ground roots.
- Produces: one validated combined pack and exactly one matching live viewer process.

- [ ] **Step 1: Run the complete pre-build verification suite**

```bash
python3 -m unittest \
  tests.python.test_interaction_sources \
  tests.python.test_interaction_build_cli \
  tests.python.test_interaction_artifacts \
  tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_trajectory_database \
  build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
```

Expected: all tests/builds exit 0 and the worktree is clean.

- [ ] **Step 2: Request code review and resolve Critical/Important findings**

Review the full range from `4402f3f` through current `HEAD` against the approved spec. The reviewer must inspect category-independent ranking, virtual-floor semantics, skipped-field byte validation, single-root compatibility, and viewer memory behavior. Apply every valid Critical or Important correction using a failing regression first, rerun Step 1, and commit fixes.

- [ ] **Step 3: Build the full combined artifact without replacing the tabletop pack**

```bash
make mixed-interaction-pack \
  GRAIL_PICKUP_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
  GRAIL_PICKUP_GROUND_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_ground \
  MIXED_INTERACTION_PACK=build/smart-pickup/table-ground-pack
```

Expected: builder exits 0 with `source_clips=4604`; validator exits 0. Rejections are recorded, but database clips must remain from both categories.

- [ ] **Step 4: Verify combined provenance and category representation**

```bash
jq '{source_clips,included_clips,rejected_clips,source_roots,
     categories:([.clips[].source_category]|unique),
     database_objects:.split.database_object_count}' \
  build/smart-pickup/table-ground-pack/manifest.json
python3 -m resources.validate_g1_interaction_database \
  --input build/smart-pickup/table-ground-pack
```

Expected: `source_clips` is 4604, categories are exactly `pickup_ground` and `pickup_table`, counts are consistent, and validation exits 0.

- [ ] **Step 5: Replace only the exact old viewer process**

Resolve exactly one process whose command is `^./hand_trajectory_viewer$`. Stop that exact PID normally. Launch from this worktree:

```bash
DISPLAY=:1 \
MM_INTERACTION_PACK="$PWD/build/smart-pickup/table-ground-pack" \
./hand_trajectory_viewer
```

Do not stop or launch controller, mesh, terrain, diffusion, or capture processes.

- [ ] **Step 6: Verify live runtime state and memory bound**

Confirm exactly one viewer process. Compare `sha256sum hand_trajectory_viewer` with `/proc/<pid>/exe`, inspect `DISPLAY` and `MM_INTERACTION_PACK` in `/proc/<pid>/environ`, and run:

```bash
ps -o pid,%cpu,%mem,rss,vsz,etime,cmd -p <pid>
git status --short
```

Expected: matching hashes, `DISPLAY=:1`, combined pack path, no runtime errors, clean worktree, and steady-state RSS at or below 1,126,400 KiB (1.1 GiB).

- [ ] **Step 7: Hand off visual validation**

Tell the user that high grasps should return TABLE options, low grasps moved clear of the visible table should return GROUND options, and overlap heights may show both. Ask them to move the object with existing controls, press Enter, and cycle `/` or `]`; do not claim motion quality until they visually confirm it.
