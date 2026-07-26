# SOMA Tabletop Reach Corpus Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the 12 continuous SOMA tabletop recordings through the existing reach segmentation and return pipeline, then publish them together with the original accepted reaches as a deterministic bilateral v4 pack.

**Architecture:** A strict SOMA CSV adapter converts the delivery's centimeter/degree representation into the existing validated 36-value G1 qpos source record. The existing canonical converter, proposal detector, annotation model, outbound/contact/return builder, and mirror stage remain unchanged. The pack builder gains ordered multi-corpus input so the new tabletop review can be combined without invalidating the original review-v2 annotations.

**Tech Stack:** Python 3, NumPy, SciPy `Rotation`, existing G1 MuJoCo kinematics, Python `unittest`, C++17 coverage probe, existing Tk/raylib viewers.

## Global Constraints

- Consume `/home/ubuntu/Downloads/takara_tabletop_delivery/03_retargeted_soma_csv/` directly; do not substitute the sibling GMR retargeting.
- Treat every CSV as one continuous 100 Hz source containing many reaches, not one reach clip.
- Reuse the current proposal, annotation, 3.6-second outbound cap, contact-index, paired-return, and bilateral-mirroring code unchanged.
- Parse root XYZ as centimeters, root Euler as intrinsic ZYX degrees stored in XYZ columns, and all 29 G1 joint DOFs as degrees.
- Namespace sequence IDs as `tabletop_soma/<filename-stem>`.
- Preserve every existing review, annotation document, and reach pack; publish the combined result as `build/g1-reaches/reach-pack-v4`.
- Preserve unrelated user changes and generated artifacts.
- Do not alter IK, collision policy, grasp-orientation behavior, locomotion, or diffusion in this plan.

---

### Task 1: Strict SOMA CSV Source Adapter

**Files:**
- Modify: `resources/g1_reach_builder/sources.py`
- Modify: `resources/prepare_g1_reach_review.py`
- Modify: `tests/python/test_g1_reach_sources.py`
- Modify: `tests/python/test_g1_reach_review.py`

**Interfaces:**
- Consumes: the existing `GMRSource` validated qpos carrier and `convert_review_sources`.
- Produces: `SOMA_COLUMNS`, `load_soma_csv_directory(path: Path, fps: float = 100.0) -> list[GMRSource]`, and the `--soma-csv-dir` preparation mode.

- [ ] **Step 1: Write failing SOMA source tests**

Add a fixture that writes exact SOMA headers and numeric rows, then require units, rotations, ordering, provenance, and namespacing:

```python
from scipy.spatial.transform import Rotation as ScipyRotation
from resources.g1_reach_builder.sources import (
    SOMA_COLUMNS,
    load_soma_csv_directory,
)

def write_soma_csv(path: Path, frames: int = 4) -> np.ndarray:
    values = np.zeros((frames, len(SOMA_COLUMNS)), np.float64)
    values[:, 0] = np.arange(frames)
    values[:, 1:4] = np.array([100.0, -200.0, 75.0])
    values[:, 4:7] = np.array([10.0, 20.0, 30.0])
    values[:, 7:] = np.arange(29) + np.arange(frames)[:, None]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(SOMA_COLUMNS)
        writer.writerows(values)
    return values

source = load_soma_csv_directory(root)[0]
self.assertEqual(source.sequence_id, "tabletop_soma/height0_top")
self.assertEqual(source.fps, 100.0)
np.testing.assert_allclose(source.qpos[:, :3], values[:, 1:4] / 100.0)
expected_xyzw = ScipyRotation.from_euler(
    "ZYX", values[:, 4:7][:, [2, 1, 0]], degrees=True
).as_quat()
np.testing.assert_allclose(source.qpos[:, 3:7], expected_xyzw[:, [3, 0, 1, 2]])
np.testing.assert_allclose(source.qpos[:, 7:], np.radians(values[:, 7:]))
self.assertEqual(source.archive_sha256, hashlib.sha256(csv_path.read_bytes()).hexdigest())
```

Add subtests that reject:

```python
("header", mutate_header, "SOMA header"),
("frame gap", mutate_frame_to_two, "contiguous from zero"),
("nonfinite", mutate_joint_to_nan, "non-finite"),
("fps", lambda _: None, "invalid fps"),
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_sources.GMRArchiveTests.test_loads_soma_csv_directory_with_exact_units_and_rotation \
  tests.python.test_g1_reach_sources.GMRArchiveTests.test_rejects_malformed_soma_csv_sources \
  -v
```

Expected: import failure because `SOMA_COLUMNS` and `load_soma_csv_directory` do not exist.

- [ ] **Step 3: Implement strict SOMA parsing**

Define the exact columns from the delivery header and parse without object arrays:

```python
SOMA_G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
SOMA_COLUMNS = (
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
    *tuple(f"{name}_dof" for name in SOMA_G1_JOINT_NAMES),
)

def load_soma_csv_directory(
    path: Path,
    fps: float = 100.0,
) -> list[GMRSource]:
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid fps {fps}")
    result = []
    for csv_path in sorted(Path(path).glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = tuple(next(reader))
            if header != SOMA_COLUMNS:
                raise ValueError(f"{csv_path.name}: SOMA header mismatch")
            values = np.asarray([[float(value) for value in row] for row in reader])
        if values.ndim != 2 or values.shape[1] != len(SOMA_COLUMNS):
            raise ValueError(f"{csv_path.name}: SOMA row width mismatch")
        if not np.isfinite(values).all():
            raise ValueError(f"{csv_path.name}: non-finite value")
        frames = values[:, 0]
        if not np.array_equal(frames, np.arange(len(values))):
            raise ValueError(f"{csv_path.name}: source frames must be contiguous from zero")
        root_xyzw = ScipyRotation.from_euler(
            "ZYX", values[:, 4:7][:, [2, 1, 0]], degrees=True
        ).as_quat()
        qpos = np.concatenate((
            values[:, 1:4] / 100.0,
            root_xyzw[:, [3, 0, 1, 2]],
            np.radians(values[:, 7:]),
        ), axis=1)
        result.append(GMRSource(
            sequence_id=f"tabletop_soma/{csv_path.stem}",
            archive_member=csv_path.name,
            archive_path=csv_path.resolve(),
            archive_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            fps=float(fps),
            fps_overridden=True,
            qpos=qpos,
            source_frames=np.arange(len(qpos), dtype=np.int32),
            disposition="included",
        ))
    if not result:
        raise ValueError(f"no SOMA CSV files in {path}")
    return result
```

Use the established G1 29-DOF order explicitly; do not derive it from the input header.

- [ ] **Step 4: Add the mutually exclusive preparation input**

Update the CLI parser and run path:

```python
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--archive", type=Path)
source.add_argument("--soma-csv-dir", type=Path)
parser.add_argument("--source-fps", type=float)

if args.soma_csv_dir is not None:
    all_sources = load_soma_csv_directory(
        args.soma_csv_dir,
        fps=100.0 if args.source_fps is None else args.source_fps,
    )
else:
    all_sources = load_gmr_archive(
        args.archive,
        fps_override=args.source_fps,
        include_excluded=True,
    )
```

Add parser/run tests proving the old archive path remains valid and SOMA mode passes all parsed sources into the existing converter.

```python
soma = prepare_cli.parse_args([
    "--soma-csv-dir", str(root),
    "--source-fps", "100",
    "--g1-xml", str(xml),
    "--output", str(output),
])
self.assertEqual(soma.soma_csv_dir, root)
self.assertIsNone(soma.archive)
with self.assertRaises(SystemExit):
    prepare_cli.parse_args([
        "--archive", str(archive),
        "--soma-csv-dir", str(root),
        "--g1-xml", str(xml),
        "--output", str(output),
    ])
```

- [ ] **Step 5: Run focused and complete builder tests**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_sources \
  tests.python.test_g1_reach_review \
  -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_reach_builder/sources.py \
  resources/prepare_g1_reach_review.py \
  tests/python/test_g1_reach_sources.py \
  tests/python/test_g1_reach_review.py
git commit -m "feat: import SOMA tabletop reach recordings"
```

---

### Task 2: Multi-Corpus Reach-Pack Publication

**Files:**
- Modify: `resources/g1_reach_builder/build.py`
- Modify: `resources/build_g1_reach_database.py`
- Modify: `tests/python/test_g1_reach_build_cli.py`

**Interfaces:**
- Consumes: ordered validated `(label, ReviewCorpus, AnnotationDocument)` inputs.
- Produces: `ReachCorpusInput`, `prepare_combined_reach_pack(inputs)`, repeatable `--review` and `--annotations` CLI arguments, and per-corpus manifest counts.

- [ ] **Step 1: Write failing multi-corpus tests**

Create two review fixtures with distinct sequence IDs, accept one proposal in each, and assert:

```python
result = build_cli.run(Namespace(
    review=[review_a, review_b],
    annotations=[annotations_a, annotations_b],
    output=output,
    allow_pending=False,
))
artifact, _, manifest = read_reach_pack(output)
self.assertEqual(result, 0)
self.assertEqual(manifest["captured_reaches"], 2)
self.assertEqual(manifest["mirrored_reaches"], 2)
self.assertEqual(len(manifest["corpora"]), 2)
self.assertEqual(
    [record["captured_reaches"] for record in manifest["corpora"]],
    [1, 1],
)
self.assertEqual(artifact.active_hands.tolist(), [0, 1, 0, 1])
```

Also assert mismatched review/annotation counts return an actionable CLI error and duplicate proposal/reach IDs are rejected before publication.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_build_cli.ReachBuildCLITests.test_combines_two_review_corpora_before_mirroring \
  tests.python.test_g1_reach_build_cli.ReachBuildCLITests.test_rejects_mismatched_or_duplicate_corpus_inputs \
  -v
```

Expected: failure because the builder accepts only one corpus.

- [ ] **Step 3: Implement combined canonical publication**

Add:

```python
@dataclass(frozen=True)
class ReachCorpusInput:
    label: str
    corpus: ReviewCorpus
    annotations: AnnotationDocument

def prepare_combined_reach_pack(
    inputs: Sequence[ReachCorpusInput],
) -> tuple[list[CanonicalReach], ReachArtifact, ReachFeatures, dict]:
    if not inputs:
        raise ValueError("combined reach pack requires at least one corpus")
    captured = []
    corpus_records = []
    for item in inputs:
        current = [
            build_captured_reach(item.corpus, annotation)
            for annotation in item.annotations.annotations
            if annotation.status == "accepted"
        ]
        captured.extend(current)
        corpus_records.append({
            "label": item.label,
            "captured_reaches": len(current),
            "pending_annotations": sum(
                value.status == "pending"
                for value in item.annotations.annotations
            ),
            "rejected_annotations": sum(
                value.status == "rejected"
                for value in item.annotations.annotations
            ),
            "sequence_ids": list(item.corpus.sequence_ids),
        })
    ids = [reach.reach_id for reach in captured]
    proposals = [reach.proposal_id for reach in captured]
    if len(ids) != len(set(ids)) or len(proposals) != len(set(proposals)):
        raise ValueError("duplicate reach or proposal id across corpora")
    reaches = build_bilateral_reaches(captured)
    artifact, features = assemble_reach_pack(reaches)
```

Preserve the existing manifest fields and add `"corpora": corpus_records`.
Keep `prepare_reach_pack(corpus, annotations)` as a one-input compatibility
wrapper around `prepare_combined_reach_pack`.

- [ ] **Step 4: Support repeatable ordered CLI pairs**

Use repeatable arguments:

```python
parser.add_argument("--review", type=Path, action="append", required=True)
parser.add_argument("--annotations", type=Path, action="append", required=True)
```

Normalize direct-test `Path` values into one-element lists, require equal
nonzero counts, load each pair, sum pending annotations, and pass ordered
`ReachCorpusInput` values to `prepare_combined_reach_pack`.

- [ ] **Step 5: Run focused and complete pack-builder tests**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_build_cli \
  tests.python.test_g1_reach_motions \
  tests.python.test_g1_reach_mirror \
  tests.python.test_g1_reach_returns \
  tests.python.test_g1_reach_artifacts \
  -v
```

Expected: all tests pass, including the existing single-corpus behavior.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_reach_builder/build.py \
  resources/build_g1_reach_database.py \
  tests/python/test_g1_reach_build_cli.py
git commit -m "feat: combine independent reach corpora"
```

---

### Task 3: Make Coverage Verification Pack-Size Agnostic

**Files:**
- Modify: `g1_reach_coverage_probe.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: `pack.database.clip_count` and `reach::kYawPlacementCount`.
- Produces: dynamic expected-instance verification and JSON evidence for any valid pack size.

- [ ] **Step 1: Write the failing source-contract test**

Add:

```python
source = self.source(PROBE)
self.assertNotIn("kExpectedInstances = 4608U", source)
self.assertIn("pack.database.clip_count * reach::kYawPlacementCount", source)
self.assertIn("size_t expected_instances", source)
self.assertIn('"expected_instances"', source)
```

In `test_probe_contract`, remove the obsolete required literal `"4608"` and
replace it with `"expected_instances"`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests.test_probe_counts_diversity_across_every_accepted_candidate \
  -v
```

Expected: failure because the probe still freezes 4,608 instances.

- [ ] **Step 3: Thread dynamic expected instances through the probe**

Remove `kExpectedInstances`, compute:

```cpp
const size_t expected_instances =
    static_cast<size_t>(pack.database.clip_count) *
    reach::kYawPlacementCount;
```

Pass `expected_instances` into `valid_report` and `to_json`; compare every
fixture's raw and processed counts against that value and write it to JSON.
Do not weaken the 30-second deadline, position gate, open-space acceptance,
per-hand, or azimuth-sector gates.

- [ ] **Step 4: Run tests and rebuild the probe**

Run:

```bash
/usr/bin/python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make g1_reach_coverage_probe
```

Expected: 14 viewer/probe contract tests pass and the release probe links.

- [ ] **Step 5: Commit**

```bash
git add g1_reach_coverage_probe.cpp \
  tests/python/test_g1_reach_coverage_viewer.py
git commit -m "fix: derive reach coverage instance count from pack"
```

---

### Task 4: Build and Validate the Real Tabletop and Combined Corpora

**Files:**
- Generate: `build/g1-reaches/tabletop-review-v1/`
- Generate: `build/g1-reaches/tabletop-annotations-v1.json`
- Generate: `build/g1-reaches/reach-pack-v4/`
- Generate: `build/g1-reaches/tabletop-proposal-audit-v1.json`
- Generate: `build/g1-reaches/combined-coverage-v4.json`

**Interfaces:**
- Consumes: the real SOMA directory, G1 XML, existing review-v2/annotations-v2, and Tasks 1–3.
- Produces: a reviewed tabletop corpus and combined v4 pack; generated artifacts remain untracked.

- [ ] **Step 1: Convert the 12 real continuous sources**

Run:

```bash
/usr/bin/python3 -m resources.prepare_g1_reach_review \
  --soma-csv-dir /home/ubuntu/Downloads/takara_tabletop_delivery/03_retargeted_soma_csv \
  --source-fps 100 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-reaches/tabletop-review-v1
```

Expected: `included=12`, `excluded=0`, `source_fps=100`, and `target_fps=25`.

- [ ] **Step 2: Audit proposal counts per continuous source**

Read the validated review/proposals and write deterministic JSON containing
each source's frame and proposal counts:

```bash
/usr/bin/python3 - <<'PY'
from collections import Counter
import json
from pathlib import Path
from resources.g1_reach_builder.review import (
    read_reach_proposals,
    read_review_corpus,
)

review = Path("build/g1-reaches/tabletop-review-v1")
output = Path("build/g1-reaches/tabletop-proposal-audit-v1.json")
corpus = read_review_corpus(review)
proposals = read_reach_proposals(review)
counts = Counter(value.sequence_id for value in proposals)
records = []
for index, sequence_id in enumerate(corpus.sequence_ids):
    frames = int(corpus.range_stops[index] - corpus.range_starts[index])
    records.append({
        "sequence_id": sequence_id,
        "frames": frames,
        "proposals": counts[sequence_id],
    })
document = {
    "source_count": len(corpus.sequence_ids),
    "proposal_count": len(proposals),
    "sources": records,
}
if document["source_count"] != 12:
    raise SystemExit(f"expected 12 sources, got {document['source_count']}")
if not all(value["sequence_id"].startswith("tabletop_soma/") for value in records):
    raise SystemExit("tabletop sequence namespace mismatch")
if not all(value["proposals"] > 0 for value in records):
    raise SystemExit("one or more tabletop sources has zero proposals")
if not 120 <= document["proposal_count"] <= 360:
    raise SystemExit(
        f"proposal total {document['proposal_count']} is outside [120, 360]"
    )
output.write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(document, sort_keys=True))
PY
```

Require:

```text
source_count == 12
all sequence IDs start with tabletop_soma/
all per-source proposal counts are greater than zero
total proposal count is between 120 and 360
```

The expected center is approximately 240. Any source outside 10–30 proposals
must be inspected in the existing annotation viewer; never replace detection
with equal-duration splitting.

- [ ] **Step 3: Structurally confirm the default boundaries**

Run:

```bash
/usr/bin/python3 -m resources.confirm_g1_reach_defaults \
  --review build/g1-reaches/tabletop-review-v1 \
  --output build/g1-reaches/tabletop-annotations-v1.json
/usr/bin/python3 tools/annotate_g1_reaches.py \
  --review build/g1-reaches/tabletop-review-v1 \
  --annotations build/g1-reaches/tabletop-annotations-v1.json \
  --smoke-test
```

Expected: no pending annotations; structurally invalid boundaries are rejected
with reasons; the flat annotation viewer smoke test succeeds.

- [ ] **Step 4: Build the combined v4 pack**

Run:

```bash
/usr/bin/python3 -m resources.build_g1_reach_database \
  --review build/g1-reaches/review-v2 \
  --annotations build/g1-reaches/annotations-v2.json \
  --review build/g1-reaches/tabletop-review-v1 \
  --annotations build/g1-reaches/tabletop-annotations-v1.json \
  --output build/g1-reaches/reach-pack-v4
```

Verify the manifest equations:

```text
captured_reaches == 192 + tabletop accepted annotations
mirrored_reaches == captured_reaches
total reach records == 2 * captured_reaches
corpora[0].captured_reaches == 192
corpora[1].captured_reaches == tabletop accepted annotations
paired_returns + unavailable_returns == total reach records
```

- [ ] **Step 5: Run the combined coverage probe**

Run:

```bash
make g1_reach_coverage_probe g1_reach_coverage_viewer
./g1_reach_coverage_probe \
  build/g1-reaches/reach-pack-v4 \
  --json build/g1-reaches/combined-coverage-v4.json
```

Expected: exit zero, complete processing of the dynamically reported instance
count, nonzero bilateral open-space coverage, and completion within 30 seconds.
If the complete combined search exceeds 30 seconds, record the measured
result as a performance blocker; do not silently raise the deadline or drop
tabletop clips in this ingestion task.

- [ ] **Step 6: Run the complete protected verification set**

Run:

```bash
/usr/bin/python3 -m unittest \
  tests.python.test_g1_reach_sources \
  tests.python.test_g1_reach_review \
  tests.python.test_g1_reach_segmentation \
  tests.python.test_g1_reach_annotations \
  tests.python.test_g1_reach_motions \
  tests.python.test_g1_reach_mirror \
  tests.python.test_g1_reach_returns \
  tests.python.test_g1_reach_artifacts \
  tests.python.test_g1_reach_build_cli \
  tests.python.test_g1_reach_coverage_viewer \
  -v
git diff --check
```

Expected: every test passes and no whitespace errors are reported.

- [ ] **Step 7: Launch exactly one viewer on v4**

Stop only the existing reach-coverage viewer, launch the rebuilt viewer with:

```bash
./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v4
```

Run one open-space search, select RETARGETED mode, cycle both old and
`tabletop_soma/` candidates, and verify complete outbound animation. Verify
stored paired-return counts from the v4 manifest and builder tests. Leave one
healthy viewer running.

- [ ] **Step 8: Commit tracked implementation only**

Generated reviews, annotations, packs, reports, and binaries remain untracked.
If no additional tracked changes were required by the real-data run, do not
create an empty commit.
