# Retargeted Reach Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the trusted left-hand GMR recordings into manually confirmed outbound G1 reaches, add traceable right-hand mirrors, and visualize/measurably evaluate arbitrary grasp coverage with the current arm IK.

**Architecture:** A Python pipeline loads the archive, converts full recordings to the canonical 31-bone skeleton, proposes reach endpoints, supports manual annotation, and atomically publishes a separate reach pack. A new C++ loader and evaluator consume that pack, keep each reach rooted at its normalized departure pose, shape only the active arm toward a movable grasp, and expose exact IK and collision outcomes in a flat raylib viewer and deterministic probe.

**Tech Stack:** Python 3, NumPy, SciPy, MuJoCo, Matplotlib, `unittest`, C++17, raylib, existing G1 kinematics/pose/IK/collision code, Make.

## Global Constraints

- Treat `/home/ubuntu/Downloads/g1_retargeted_motions.zip` as trusted user-authored pickle input; never load arbitrary third-party pickle files.
- Read the embedded 30 Hz source rate and resample to exactly 25 Hz.
- Process only `pickup_*`, `drawers`, `left_to_right_*`, and `right_to_left_*`; preserve but exclude `walking` and `carry_walking`.
- Publish only manually accepted outbound reaches ending at the confirmed grab frame, with at most 3.6 seconds of context.
- Derive approach direction over the final 0.20 seconds and reject displacement below 1 cm.
- Mirror canonical Y-up motion across `z = 0` with `M = diag(1, 1, -1)` and retain captured-versus-mirrored provenance.
- Keep the current seven-joint one-arm IK, fixed root/waist/legs, 45 cm request envelope, 4 cm endpoint position gate, 15 degree approach-axis gate, and 60 degree full-orientation gate.
- Never filter retrieval by recorded wrist quaternion; rank spatial and approach compatibility, then impose requested wrist orientation during shaping.
- Use only the flat skeleton viewers. Do not launch mesh or terrain renderers, take screenshots, signal the controller, or run more than one viewer process.
- Preserve the existing GRAIL interaction pack, viewer behavior, and controller paths.

---

### Task 1: Trusted GMR Loading and Canonical Review Corpus

**Files:**
- Create: `resources/g1_reach_builder/__init__.py`
- Create: `resources/g1_reach_builder/schema.py`
- Create: `resources/g1_reach_builder/sources.py`
- Create: `resources/g1_reach_builder/review.py`
- Create: `resources/prepare_g1_reach_review.py`
- Create: `tests/python/test_g1_reach_sources.py`
- Create: `tests/python/test_g1_reach_review.py`

**Interfaces:**
- Produces: `GMRSource`, `ReviewCorpus`, `load_gmr_archive(path, fps_override=None)`, `convert_review_sources(sources, kinematics)`, `write_review_corpus(output, corpus)`, and `read_review_corpus(output)`.
- Consumes: existing `G1Kinematics`, `SourceClip`, `convert_source_clip`, `G1_SKELETON`, and finite-difference helpers.

- [ ] **Step 1: Write failing archive-loader tests**

Create a tiny ZIP fixture containing one valid pickup member, excluded walking/carrying members, and malformed variants. Assert xyzw-to-wxyz root conversion, embedded 30 Hz use, exact 36-column qpos construction, in-scope filtering, explicit missing-FPS handling, finite-array validation, and archive SHA-256 provenance:

```python
sources = load_gmr_archive(archive)
self.assertEqual([s.sequence_id for s in sources], ["pickup_north_0"])
self.assertEqual(sources[0].fps, 30.0)
np.testing.assert_array_equal(sources[0].qpos[:, :3], root_pos)
np.testing.assert_array_equal(sources[0].qpos[:, 3:7], root_rot[:, [3, 0, 1, 2]])
np.testing.assert_array_equal(sources[0].qpos[:, 7:], dof_pos)
self.assertEqual(sources[0].archive_sha256, hashlib.sha256(archive.read_bytes()).hexdigest())
all_sources = {s.sequence_id: s for s in load_gmr_archive(archive, include_excluded=True)}
self.assertEqual(all_sources["walking"].disposition, "excluded")
```

- [ ] **Step 2: Run the source tests to verify RED**

Run: `python3 -m unittest tests.python.test_g1_reach_sources -v`

Expected: FAIL because `resources.g1_reach_builder.sources` does not exist.

- [ ] **Step 3: Implement validated source loading**

Define focused schema records and construct MuJoCo qpos without modifying the ZIP:

```python
@dataclass(frozen=True)
class GMRSource:
    sequence_id: str
    archive_member: str
    archive_sha256: str
    fps: float
    qpos: np.ndarray
    source_frames: np.ndarray
    disposition: str = "included"

INCLUDED = ("pickup_", "drawers", "left_to_right_", "right_to_left_")
EXCLUDED = ("walking", "carry_walking")

def gmr_qpos(record: Mapping[str, object], member: str) -> tuple[float, np.ndarray]:
    root_pos = require_array(record, "root_pos", (-1, 3), member)
    root_xyzw = require_array(record, "root_rot", (len(root_pos), 4), member)
    dof = require_array(record, "dof_pos", (len(root_pos), 29), member)
    fps = require_positive_fps(record.get("fps"), member)
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    return fps, np.concatenate((root_pos, root_wxyz, dof), axis=1)
```

Only deserialize members under `g1_retargeted_motions/gmr_pkl/`. Reject duplicate sequence IDs, malformed quaternion norms, wrong shapes, non-finite values, and absent FPS unless `fps_override` is explicitly supplied.

- [ ] **Step 4: Write failing canonical-review tests**

Mock `G1Kinematics` with the established fixture converter and assert a deterministic, safe review directory:

```python
corpus = convert_review_sources([source], fake_kinematics)
self.assertEqual(corpus.fps, 25.0)
self.assertEqual(corpus.skeleton_signature, G1_SKELETON.signature())
self.assertEqual(corpus.range_starts.tolist(), [0])
self.assertEqual(corpus.range_stops.tolist(), [len(corpus.positions)])
write_review_corpus(output, corpus)
round_trip = read_review_corpus(output)
np.testing.assert_array_equal(round_trip.source_frames, corpus.source_frames)
self.assertEqual(set(p.name for p in output.iterdir()), {"review_motions.npz", "manifest.json"})
```

- [ ] **Step 5: Implement conversion and atomic review publication**

Use one `SourceClip` per included recording, call `convert_source_clip(..., target_fps=25.0)`, verify `G1_SKELETON.signature()`, concatenate local positions/rotations/source frames, and write `review_motions.npz` without object arrays plus canonical JSON. Publish through a sibling temporary directory followed by `os.replace`, using the recovery pattern in `resources/g1_interaction_builder/artifacts.py`.

The CLI contract is:

```python
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--g1-xml", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-fps", type=float)
```

- [ ] **Step 6: Run focused tests and the real archive conversion**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_sources tests.python.test_g1_reach_review -v
python3 -m resources.prepare_g1_reach_review \
  --archive /home/ubuntu/Downloads/g1_retargeted_motions.zip \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-reaches/review
```

Expected: tests PASS; CLI reports 8 included sources, 2 excluded sources, `source_fps=30`, and `target_fps=25`.

- [ ] **Step 7: Commit**

```bash
git add resources/g1_reach_builder resources/prepare_g1_reach_review.py tests/python/test_g1_reach_sources.py tests/python/test_g1_reach_review.py
git commit -m "feat: import retargeted G1 reach recordings"
```

### Task 2: Reach Candidate Proposals

**Files:**
- Create: `resources/g1_reach_builder/segmentation.py`
- Create: `tests/python/test_g1_reach_segmentation.py`
- Modify: `resources/prepare_g1_reach_review.py`
- Modify: `resources/g1_reach_builder/review.py`

**Interfaces:**
- Consumes: `ReviewCorpus` from Task 1.
- Produces: `ReachProposal`, `SegmentationConfig`, `propose_reaches(corpus, config)`, and `proposals.json` inside the review directory.

- [ ] **Step 1: Write synthetic segmentation tests**

Construct root-relative wrist traces for two neutral→reach→neutral cycles plus walking-like oscillation and assert deterministic maxima, preceding departures, source-frame mapping, 3.6-second cap, and no automatic acceptance:

```python
config = SegmentationConfig(fps=25.0)
proposals = propose_wrist_trace(trace, source_frames, "pickup_north_0", config)
self.assertEqual([p.grab_frame for p in proposals], [50, 125])
self.assertTrue(all(p.grab_frame - p.departure_frame <= 90 for p in proposals))
self.assertTrue(all(p.status == "pending" for p in proposals))
self.assertGreaterEqual(proposals[0].excursion_m, 0.18)
```

- [ ] **Step 2: Run segmentation tests to verify RED**

Run: `python3 -m unittest tests.python.test_g1_reach_segmentation -v`

Expected: FAIL because `segmentation.py` does not exist.

- [ ] **Step 3: Implement deterministic proposal detection**

Compute canonical world transforms with `forward_local_hierarchy`, then express `LeftWrist` in the `Simulation` frame. Use these frozen defaults:

```python
@dataclass(frozen=True)
class SegmentationConfig:
    fps: float = 25.0
    neutral_radius_m: float = 0.08
    return_radius_m: float = 0.10
    minimum_excursion_m: float = 0.18
    minimum_separation_s: float = 0.60
    maximum_outbound_s: float = 3.60
    approach_window_s: float = 0.20
    minimum_approach_displacement_m: float = 0.01
```

Estimate the neutral center as the component-wise median of the slowest 20 percent of root-relative wrist samples. Split cycles at returns within `return_radius_m`, choose the largest excursion in each cycle, select the last prior frame within `neutral_radius_m`, merge peaks closer than `minimum_separation_s`, and retain every proposal as `pending`. Walking suppression is advisory only; manual review remains authoritative.

- [ ] **Step 4: Persist proposals beside the review corpus**

Write canonical JSON containing `proposal_id`, `sequence_id`, canonical and source departure/grab frames, excursion, final 0.20-second displacement, confidence score, and `status: pending`. Include its SHA-256 in `manifest.json`; make `read_review_corpus` verify both files.

- [ ] **Step 5: Run tests and regenerate the real review directory**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_segmentation tests.python.test_g1_reach_review -v
python3 -m resources.prepare_g1_reach_review \
  --archive /home/ubuntu/Downloads/g1_retargeted_motions.zip \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-reaches/review
```

Expected: tests PASS and every proposal remains `pending`.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_reach_builder/segmentation.py resources/g1_reach_builder/review.py resources/prepare_g1_reach_review.py tests/python/test_g1_reach_segmentation.py tests/python/test_g1_reach_review.py
git commit -m "feat: propose outbound reach segments"
```

### Task 3: Manual Grab-Frame Annotation Tool

**Files:**
- Create: `resources/g1_reach_builder/annotations.py`
- Create: `tools/annotate_g1_reaches.py`
- Create: `tests/python/test_g1_reach_annotations.py`

**Interfaces:**
- Consumes: `review_motions.npz`, `proposals.json`, and review `manifest.json`.
- Produces: `annotations.json` through `load_annotations`, `update_annotation`, and `write_annotations_atomic`.

- [ ] **Step 1: Write failing annotation-model tests**

Test deterministic IDs, frame-bound validation, accepted/rejected/pending states, 1 cm approach rejection, archive/proposal checksum binding, and interrupted-write recovery:

```python
accepted = update_annotation(
    proposal, departure_frame=20, grab_frame=70, status="accepted", note=""
)
self.assertEqual(accepted.active_hand, "left")
self.assertEqual(accepted.frame_count, 51)
write_annotations_atomic(path, document)
self.assertEqual(load_annotations(path, review_manifest).annotations[0], accepted)
with self.assertRaisesRegex(ValueError, "approach displacement"):
    update_annotation(still_trace, departure_frame=20, grab_frame=21, status="accepted", note="")
```

- [ ] **Step 2: Run annotation tests to verify RED**

Run: `python3 -m unittest tests.python.test_g1_reach_annotations -v`

Expected: FAIL because the annotation module does not exist.

- [ ] **Step 3: Implement the annotation document and atomic writes**

Use a versioned document with this exact per-reach record:

```python
@dataclass(frozen=True)
class ReachAnnotation:
    proposal_id: str
    sequence_id: str
    active_hand: str
    departure_frame: int
    grab_frame: int
    source_departure_frame: int
    source_grab_frame: int
    status: str
    note: str
    endpoint_position_root: tuple[float, float, float]
    endpoint_rotation_root_wxyz: tuple[float, float, float, float]
    approach_direction_root: tuple[float, float, float]
```

Compute endpoint fields from the operator-confirmed frames, never from the original proposal. Save to `annotations.json.tmp`, `fsync` the file, and replace `annotations.json`; keep a valid existing file if serialization or replacement fails.

- [ ] **Step 4: Write source-contract tests for the Matplotlib tool**

Assert that the tool imports Matplotlib lazily, draws the canonical 31-bone hierarchy, and exposes these controls: Space play/pause, J/L one-frame scrub, Shift+J/L ten-frame scrub, `[`/`]` previous/next proposal, `1` set departure, `2` set grab, A accept, R reject, S save, Escape close.

- [ ] **Step 5: Implement the flat annotation viewer**

The event handler must update state without recomputing FK for the full source:

```python
def on_key(event: KeyEvent) -> None:
    actions = {
        " ": toggle_play,
        "j": lambda: scrub(-1),
        "l": lambda: scrub(1),
        "J": lambda: scrub(-10),
        "L": lambda: scrub(10),
        "[": previous_proposal,
        "]": next_proposal,
        "1": set_departure,
        "2": set_grab,
        "a": accept_current,
        "r": reject_current,
        "s": save,
        "escape": close,
    }
    action = actions.get(event.key)
    if action is not None:
        action()
```

Show source/proposal index, canonical and source frames, departure/grab markers, pending/accepted/rejected status, and approach displacement. Draw only the G1 skeleton, wrist trace, and timeline; do not load terrain or mesh assets.

- [ ] **Step 6: Run annotation tests and a noninteractive smoke test**

Run:

```bash
MPLBACKEND=Agg python3 -m unittest tests.python.test_g1_reach_annotations -v
MPLBACKEND=Agg python3 tools/annotate_g1_reaches.py \
  --review build/g1-reaches/review --annotations build/g1-reaches/annotations.json \
  --smoke-test
```

Expected: tests PASS; smoke test loads the first proposal, renders one frame, and exits without opening a window.

- [ ] **Step 7: Commit**

```bash
git add resources/g1_reach_builder/annotations.py tools/annotate_g1_reaches.py tests/python/test_g1_reach_annotations.py
git commit -m "feat: add reach grab-frame annotation tool"
```

### Task 4: Outbound Canonicalization and Bilateral Mirroring

**Files:**
- Create: `resources/g1_reach_builder/motions.py`
- Create: `resources/g1_reach_builder/mirror.py`
- Create: `tests/python/test_g1_reach_motions.py`
- Create: `tests/python/test_g1_reach_mirror.py`

**Interfaces:**
- Consumes: accepted `ReachAnnotation` records and `ReviewCorpus`.
- Produces: `CanonicalReach`, `build_captured_reach`, `mirror_reach`, and `build_bilateral_reaches`.

- [ ] **Step 1: Write failing outbound canonicalization tests**

Assert inclusive grab-frame clipping, 90-frame maximum, departure-root identity, recomputed derivatives, left active hand, and exact annotation provenance:

```python
reach = build_captured_reach(corpus, annotation)
self.assertLessEqual(reach.frame_count, 91)
self.assertEqual(reach.source_frames[-1], annotation.source_grab_frame)
np.testing.assert_allclose(reach.positions[0, 0], [0, 0, 0], atol=1e-6)
np.testing.assert_allclose(reach.rotations[0, 0], [1, 0, 0, 0], atol=1e-6)
self.assertEqual(reach.active_hand, ReachHand.LEFT)
self.assertEqual(reach.augmentation, ReachAugmentation.CAPTURED)
```

- [ ] **Step 2: Implement root-normalized outbound extraction**

Slice `max(departure, grab - 90)` through `grab` inclusive. Transform all world-space bones by the inverse departure `Simulation` transform, convert back to local transforms with `world_to_local`, and recompute linear/angular velocities at 25 Hz. Keep the root fixed only by this rigid normalization; do not modify waist, legs, or arm pose.

- [ ] **Step 3: Write failing mirror tests**

Use an asymmetric full-body fixture and assert explicit bone swaps, canonical `z` reflection, right active hand, quaternion properness, endpoint/approach symmetry, provenance, and double-mirror identity:

```python
right = mirror_reach(left)
self.assertEqual(right.active_hand, ReachHand.RIGHT)
self.assertEqual(right.augmentation, ReachAugmentation.MIRRORED)
self.assertEqual(right.original_reach_id, left.reach_id)
np.testing.assert_allclose(right.endpoint_position_root, left.endpoint_position_root * [1, 1, -1], atol=1e-5)
np.testing.assert_allclose(mirror_reach(right, allow_mirrored=True).positions, left.positions, atol=1e-5)
self.assertTrue(np.all(np.linalg.det(quaternion_matrices(right.rotations)) > 0.9999))
```

- [ ] **Step 4: Implement world-space reflection and left/right exchange**

Freeze the complete 31-entry map, including center bones mapping to themselves:

```python
MIRROR_NAMES = {
    name: (name.replace("Left", "Right") if name.startswith("Left")
           else name.replace("Right", "Left") if name.startswith("Right")
           else name)
    for name in G1_SKELETON.names
}
M = np.diag([1.0, 1.0, -1.0])
mirrored_gp[:, dst] = gp[:, src] @ M
mirrored_R[:, dst] = M @ R[:, src] @ M
```

Resolve names to a validated integer map once. Reflect world transforms, swap bones, convert back to local transforms, normalize/unroll quaternions, swap foot contacts, recompute velocities, and refuse to mirror an already mirrored record in normal builds.

- [ ] **Step 5: Run motion and mirror tests**

Run: `python3 -m unittest tests.python.test_g1_reach_motions tests.python.test_g1_reach_mirror -v`

Expected: PASS, including `1e-5` m and `1e-4` rad symmetry limits.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_reach_builder/motions.py resources/g1_reach_builder/mirror.py tests/python/test_g1_reach_motions.py tests/python/test_g1_reach_mirror.py
git commit -m "feat: build bilateral outbound reach motions"
```

### Task 5: Reach Pack Serialization and Build CLI

**Files:**
- Create: `resources/g1_reach_builder/artifacts.py`
- Create: `resources/g1_reach_builder/build.py`
- Create: `resources/build_g1_reach_database.py`
- Create: `tests/python/test_g1_reach_artifacts.py`
- Create: `tests/python/test_g1_reach_build_cli.py`

**Interfaces:**
- Consumes: bilateral `CanonicalReach` records.
- Produces: `ReachArtifact`, `ReachFeatures`, `write_reach_pack`, `read_reach_pack`, and the three-file reach pack.

- [ ] **Step 1: Write failing binary round-trip and corruption tests**

Freeze the wire contracts `G1RCHD1` and `G1RCHF1`, little-endian version 1, 25/1 FPS, 31 bones, and ten endpoint features `[position xyz, approach xyz, wrist wxyz]`. Test truncated headers/vectors, overflow counts, trailing bytes, non-unit quaternions, bad ranges, invalid hand/augmentation values, source-table bounds, and manifest checksum mismatches.

```python
write_reach_pack(output, artifact, features, manifest)
loaded, loaded_features, loaded_manifest = read_reach_pack(output)
np.testing.assert_array_equal(loaded.positions, artifact.positions)
np.testing.assert_array_equal(loaded_features.values, features.values)
self.assertEqual(loaded_manifest["captured_reaches"], 1)
self.assertEqual(loaded_manifest["mirrored_reaches"], 1)
```

- [ ] **Step 2: Run artifact tests to verify RED**

Run: `python3 -m unittest tests.python.test_g1_reach_artifacts -v`

Expected: FAIL because the artifact module does not exist.

- [ ] **Step 3: Implement the independent reach schema and atomic pack writer**

Define arrays for ranges, local pose channels, foot contacts, active hands, augmentation kinds, source indices/names, original reach indices, endpoint position/rotation, approach direction, and source frames. Use these exact files:

```text
reach_database.bin
reach_features.bin
manifest.json
```

Validate all arrays before writing, write through a private sibling directory, read the temporary pack back fully, then atomically replace the published directory. Manifest counts must distinguish source recordings, confirmed annotations, captured reaches, mirrored reaches, and total reaches.

- [ ] **Step 4: Write and implement build-CLI tests**

Test that pending annotations block publication, rejected annotations are omitted, accepted annotations emit exactly captured+mirrored pairs, source exclusions remain in the manifest, and output is byte-deterministic:

```python
result = run(build_args(review, annotations, output))
self.assertEqual(result, 0)
artifact, _, manifest = read_reach_pack(output)
self.assertEqual(artifact.active_hands.tolist(), [0, 1])
self.assertEqual(artifact.augmentations.tolist(), [0, 1])
self.assertEqual(manifest["pending_annotations"], 0)
```

The CLI accepts `--review`, `--annotations`, `--output`, and `--allow-pending`; the default refuses pending records, while `--allow-pending` skips them and records their exact count without treating them as accepted.

- [ ] **Step 5: Run focused Python tests**

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_artifacts \
  tests.python.test_g1_reach_build_cli -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_reach_builder/artifacts.py resources/g1_reach_builder/build.py resources/build_g1_reach_database.py tests/python/test_g1_reach_artifacts.py tests/python/test_g1_reach_build_cli.py
git commit -m "feat: publish reach-only motion packs"
```

### Task 6: C++ Reach Pack Loader and Pose Access

**Files:**
- Create: `reach_database.h`
- Create: `reach_database.cpp`
- Create: `reach_motion.h`
- Create: `reach_motion.cpp`
- Create: `tests/cpp/test_reach_database.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Task 5 wire formats.
- Produces: `reach::Database`, `reach::Features`, `reach::Pack`, `reach::load_pack(path)`, `reach::pose_at_frame(database, frame)`, and `reach::endpoint_transform(database, clip)`.

- [ ] **Step 1: Write the failing cross-language loader test**

Generate a Python fixture pack, then assert all headers, channels, provenance fields, source names, endpoint transforms, and exact `interaction::Pose` reconstruction in C++:

```cpp
const reach::Pack pack = reach::load_pack(fixture_path("reach-pack"));
assert(pack.database.clip_count == 2U);
assert(pack.database.source_names.at(0) == "pickup_north_0");
assert(pack.database.active_hands == std::vector<uint8_t>({0U, 1U}));
const interaction::Pose pose = reach::pose_at_frame(pack.database, 0);
assert(near(pose.positions[g1_skeleton::Simulation], vec3(0, 0, 0)));
```

- [ ] **Step 2: Add the test target and verify RED**

Run: `make build/tests/test_reach_database && build/tests/test_reach_database`

Expected: compilation FAIL because `reach_database.h` is absent.

- [ ] **Step 3: Implement bounded little-endian loaders**

Use the existing `interaction::FormatError`, guarded count multiplication, exact file-size checks, quaternion validation, skeleton signature/count checks, feature/database clip parity, range coverage, enum bounds, source-index bounds, and EOF rejection. Do not parse JSON in C++; source names and provenance needed by the HUD come from the binary string table.

Expose this stable contract:

```cpp
namespace reach {
enum class Hand : uint8_t { Left = 0U, Right = 1U };
enum class Augmentation : uint8_t { Captured = 0U, Mirrored = 1U };
struct Pack { Database database; Features features; };
Pack load_pack(const std::filesystem::path& directory);
interaction::Pose pose_at_frame(const Database&, int32_t frame);
interaction::Transform endpoint_transform(const Database&, size_t clip);
}
```

- [ ] **Step 4: Run loader and existing interaction database tests**

Run:

```bash
make build/tests/test_reach_database build/tests/test_interaction_database
build/tests/test_reach_database
build/tests/test_interaction_database
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add reach_database.h reach_database.cpp reach_motion.h reach_motion.cpp tests/cpp/test_reach_database.cpp Makefile
git commit -m "feat: load reach-only motion packs"
```

### Task 7: Retrieval, Arm Shaping, and Exact Diagnostics

**Files:**
- Create: `reach_coverage.h`
- Create: `reach_coverage.cpp`
- Create: `tests/cpp/test_reach_coverage.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `reach::Pack`, existing `interaction::solve_hand_ik`, `interaction::world_pose`, and trajectory collision helpers.
- Produces: `reach::Query`, `reach::Candidate`, `reach::Evaluation`, `reach::Diagnostics`, `select_candidates`, `shape_candidate`, and `evaluate_candidate`.

- [ ] **Step 1: Write failing retrieval and zero-retarget tests**

Assert same-hand filtering, no recorded-wrist-orientation prefilter, 45 cm position envelope, spatial/approach/clip deterministic ranking, and exact source endpoint reproduction for captured and mirrored fixtures:

```cpp
reach::Query query{};
query.hand = reach::Hand::Left;
query.target = reach::endpoint_transform(pack.database, 0U);
query.approach_world = reach::approach_direction(pack.database, 0U);
const auto candidates = reach::select_candidates(pack, query);
assert(candidates.front().clip == 0U);
const reach::Evaluation zero = reach::shape_candidate(pack, candidates.front(), query);
assert(zero.rejection == reach::Rejection::None);
assert(zero.position_error_m <= 0.001F);
assert(zero.approach_error_radians <= 0.008726646F);
```

- [ ] **Step 2: Run the coverage test to verify RED**

Run: `make build/tests/test_reach_coverage && build/tests/test_reach_coverage`

Expected: compilation FAIL because `reach_coverage.h` is absent.

- [ ] **Step 3: Implement retrieval and smooth endpoint shaping**

Keep each canonical root fixed. For each frame, ramp the endpoint correction with smoothstep and solve the current active arm:

```cpp
const float u = frame_count == 1U ? 1.0F
    : static_cast<float>(sample) / static_cast<float>(frame_count - 1U);
const float alpha = u * u * (3.0F - 2.0F * u);
const vec3 desired_position = source_hand.position +
    alpha * (query.target.position - source_endpoint.position);
const quat correction = quat_mul(query.target.rotation,
    quat_inv(source_endpoint.rotation));
const quat desired_rotation = quat_mul(
    quat_nlerp_shortest(quat(1.0F, 0.0F, 0.0F, 0.0F), correction, alpha),
    source_hand.rotation);
```

Call `solve_hand_ik` with maximum request position 0.45 m, maximum orientation pi, accepted position 0.04 m, accepted orientation pi/3, orientation scale 0.10, and at least 16 iterations. Retain the best bounded pose returned by the existing solver; never move the root, waist, legs, or inactive arm.

- [ ] **Step 4: Implement exclusive rejection stages and collision checks**

Use this exact enum order and count the first rejection only:

```cpp
enum class Rejection : uint8_t {
    None, OutsideEnvelope, InvalidSolver, PositionError,
    ApproachAxisError, FullOrientationError,
    ObjectCollision, EnvironmentCollision,
};
```

Record `joint_limit_saturated` separately when `IKResult::reason == Reason::JointLimit`. At the final frame gate position at 4 cm, derive achieved approach from the displacement between the final wrist sample and the sample five frames earlier (a 0.20-second interval at 25 Hz) and gate at 15 degrees, then gate full wrist orientation at 60 degrees. Pass accepted pose trajectories through existing object/environment collision code with endpoint as the contact point and the active wrist as designated contact.

- [ ] **Step 5: Add perturbation and collision tests**

Test ±5/10/20/30/45 cm offsets, ±15/30/60/90 degree rotations about all three local wrist axes, 180-degree twist, joint-limit saturation without causal overclaim, active-hand object contact exemption, non-hand object collision, and body/table collision. Assert captured and mirrored results are counted separately and their union equals total evaluations.

- [ ] **Step 6: Run coverage, IK, and hand-trajectory tests**

Run:

```bash
make build/tests/test_reach_coverage build/tests/test_interaction_ik build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_interaction_ik
build/tests/test_interaction_hand_trajectories
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add reach_coverage.h reach_coverage.cpp tests/cpp/test_reach_coverage.cpp Makefile
git commit -m "feat: evaluate retargeted reach coverage"
```

### Task 8: Flat Coverage Viewer and Deterministic Probe

**Files:**
- Create: `g1_reach_coverage_viewer.cpp`
- Create: `g1_reach_coverage_probe.cpp`
- Create: `tests/python/test_g1_reach_coverage_viewer.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Task 7 selection/evaluation API and Task 5 reach pack.
- Produces: `g1_reach_coverage_viewer`, `g1_reach_coverage_probe`, interactive visualization, and JSON coverage summaries.

- [ ] **Step 1: Write failing viewer/probe source-contract tests**

Assert a standalone reach-pack argument, flat raylib skeleton, no mesh/terrain/screenshot dependencies, explicit Enter search, slash/bracket cycling, active-hand toggle, open/coverage-environment toggle, target translation/rotation controls, desired/achieved wrist axes, captured/mirrored colors, and all rejection labels.

```python
for required in ("KEY_ENTER", "KEY_SLASH", "CAPTURED LEFT", "MIRRORED RIGHT",
                 "POSITION ERROR", "APPROACH AXIS", "FULL ORIENTATION",
                 "OBJECT COLLISION", "ENVIRONMENT COLLISION"):
    self.assertIn(required, source)
for forbidden in ("g1_mesh_renderer", "terrain_runtime", "TakeScreenshot("):
    self.assertNotIn(forbidden, source)
```

- [ ] **Step 2: Add Make targets and verify RED**

Run: `python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v`

Expected: FAIL because the viewer source does not exist.

- [ ] **Step 3: Implement the flat viewer**

Use the current hand-trajectory viewer's raylib camera and cylinder/sphere skeleton style without changing that executable. Controls are:

```text
Arrow keys: target X/Z     W/S: target Y
Q/E: yaw                   R/F: pitch             Z/C: roll
M: left/right hand         G: open/coverage environment
Enter: rerun search        [/], /: cycle accepted options
V: show/hide rejected      Backspace: reset target
```

Default to the coverage environment from `make_coverage_environment`, with `G` switching to open space. Represent the generic object as a configurable oriented box (`--object-size X Y Z`, default `0.10 0.10 0.10` m). Place the desired grasp at the center of its local +X face, set the desired approach toward that face along object-local -X, and orient the default wrist frame so its +X axis agrees with that approach. Moving or rotating the object then changes a concrete world grasp while retrieval remains object-identity independent. Draw every candidate wrist path root-relative, animate the selected complete outbound skeleton, and display source name, captured/mirrored provenance, hand, target/achieved frames, stale state, accepted count, exclusive rejection counts, and joint-limit saturation.

- [ ] **Step 4: Implement deterministic zero-retarget and perturbation probe**

The probe accepts `PACK [--json OUTPUT]`, runs every clip's own endpoint first, exits nonzero if any zero-retarget result exceeds 1 mm or 0.5 degrees, then runs the frozen position/orientation perturbation matrix. Serialize sorted JSON with per-hand, per-source, height-band, direction-band, augmentation, and union counts.

- [ ] **Step 5: Run source tests and compile release tools**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make g1_reach_coverage_viewer g1_reach_coverage_probe
```

Expected: tests PASS and both binaries compile.

- [ ] **Step 6: Commit**

```bash
git add g1_reach_coverage_viewer.cpp g1_reach_coverage_probe.cpp tests/python/test_g1_reach_coverage_viewer.py Makefile
git commit -m "feat: visualize and probe G1 reach coverage"
```

### Task 9: Review the Real Corpus, Publish It, and Verify the Baseline

**Files:**
- Generate: `build/g1-reaches/annotations.json`
- Generate: `build/g1-reaches/reach-pack/`
- Generate: `build/g1-reaches/coverage-report.json`
- Modify only if evidence is wrong: the focused implementation/test file responsible for that failure.

**Interfaces:**
- Consumes: all prior tasks and the trusted archive.
- Produces: reviewed bilateral reach pack, coverage report, one live flat viewer, and evidence deciding whether IK—not data—is the next bottleneck.

- [ ] **Step 1: Open the annotation viewer for manual confirmation**

Run only after checking that no other annotation or coverage viewer is active:

```bash
DISPLAY=:1 python3 tools/annotate_g1_reaches.py \
  --review build/g1-reaches/review \
  --annotations build/g1-reaches/annotations.json
```

Expected: the operator confirms outbound left-hand reaches and rejects walking/carrying/non-reach proposals; all records end accepted or rejected.

- [ ] **Step 2: Validate annotations and build the bilateral pack**

Run:

```bash
python3 -m resources.build_g1_reach_database \
  --review build/g1-reaches/review \
  --annotations build/g1-reaches/annotations.json \
  --output build/g1-reaches/reach-pack
```

Expected: `pending_annotations=0`, `mirrored_reaches=captured_reaches`, and the pack reads back successfully.

- [ ] **Step 3: Run the zero-retarget and perturbation evidence probe**

Run:

```bash
./g1_reach_coverage_probe build/g1-reaches/reach-pack \
  --json build/g1-reaches/coverage-report.json
```

Expected: exit 0, every captured and mirrored zero-retarget endpoint is within 1 mm/0.5 degrees, and all perturbation outcomes sum exactly to the attempted count.

- [ ] **Step 4: Run the focused and aggregate regression gates**

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_sources \
  tests.python.test_g1_reach_review \
  tests.python.test_g1_reach_segmentation \
  tests.python.test_g1_reach_annotations \
  tests.python.test_g1_reach_motions \
  tests.python.test_g1_reach_mirror \
  tests.python.test_g1_reach_artifacts \
  tests.python.test_g1_reach_build_cli \
  tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_reach_database build/tests/test_reach_coverage
build/tests/test_reach_database
build/tests/test_reach_coverage
python3 -m unittest discover -s tests/python -t . -v
make test-cpp
git diff --check
```

Expected: all tests PASS and `git diff --check` emits no output.

- [ ] **Step 5: Run one bounded-memory flat-viewer smoke session**

Start exactly one process:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer build/g1-reaches/reach-pack
```

Verify target movement marks results stale, Enter recomputes, `/` cycles complete reaches, both hands are selectable, `G` distinguishes open-space from furniture coverage, skeleton animation remains responsive, and RSS remains below 1.1 GiB for ten samples. Do not start mesh or terrain executables.

- [ ] **Step 6: Record the evidence conclusion and commit source changes**

Do not commit generated build artifacts. Summarize captured/mirrored counts, endpoint bands, zero-retarget status, perturbation acceptance, and the dominant rejection stages in the final handoff. If the zero-retarget gate passes but perturbations fail primarily at position/orientation gates, recommend the hierarchical waist+arm IK follow-up; if source endpoint bands are absent, recommend more capture/annotation instead.

```bash
git status --short
git log --oneline --max-count=12
```

Expected: only intentional source/test changes are committed; `build/g1-reaches` remains untracked or ignored.
