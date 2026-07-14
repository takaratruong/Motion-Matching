# G1 Tabletop Data Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Validation Gate 1: reproducibly convert GRAIL tabletop pickup recordings into versioned 25 Hz G1 interaction artifacts and prove that Python and C++ replay exactly the same body, hand, object, table, contact, phase, grasp, and feature data.

**Architecture:** A Python offline builder discovers matched GRAIL source triplets, converts native G1 motion and scene state into one canonical coordinate/skeleton contract, derives single-hand interaction semantics, constructs target-relative features and object-disjoint splits, then atomically publishes two binary files plus JSON metadata. A small, Raylib-free C++ loader validates the same schema and produces a deterministic probe digest; no locomotion or interaction-controller behavior changes in this gate. Later plans consume these immutable artifacts behind a game-facing `PickRequest -> InteractionResult` boundary.

**Tech Stack:** C++17, Python 3.10+, NumPy 2.x, SciPy 1.x, joblib 1.x, MuJoCo 3.x, OpenUSD Python bindings, Hugging Face Hub, `unittest`, GNU Make, GRAIL native G1 recordings.

## Global Constraints

- Work only in `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching` on branch `g1-manipulation-motion-matching`.
- Do not write to, merge from, or rebase onto the moving checkout at `/home/ubuntu/projects/motion-matching`; the only terrain-derived dependency is immutable commit `cbe90b7`.
- Preserve the existing flat-ground G1 locomotion behavior and ordinary `resources/database.bin` / `resources/features.bin`; this gate does not modify `controller.cpp`.
- Generated interaction artifacts live only under `resources/g1_interaction/` and are ignored by Git.
- Convert source state from GRAIL Z-up to the Holden runtime's Y-up coordinates and convert source quaternion ordering from `xyzw` to normalized `wxyz`.
- Keep native GRAIL clips at the locomotion matcher's fixed 25 Hz rate. Future non-25-Hz inputs use time-based vector interpolation and normalized shortest-arc quaternion interpolation; derivatives never cross clip ranges.
- Admit only coherent single-hand rigid tabletop pickups. Every excluded clip is named with one machine-readable reason in `validation_report.json`.
- Stable contact requires three consecutive distinct source samples with non-empty hand contact and at most `0.02 m` / `10 degrees` of hand-in-object transform change across that window.
- Phase constants are `approach=0`, `reach=1`, `contact=2`, `lift=3`, and `hold=4`; reach begins `1.0 s` before stable contact, lift begins at `0.05 m`, and hold requires object vertical speed below `0.1 m/s` for `0.2 s` while contact persists.
- The grasp transform is hand-in-object, averaged over the first `0.2 s` of stable contact. Runtime object attachment will later use its inverse; do not store a second attachment transform.
- Feature dimension is exactly 71 with fixed group ranges: pose `[0,33)`, trajectory `[33,45)`, grasp `[45,57)`, root target `[57,65)`, context `[65,71)`.
- Object identity, source sequence identity, raw mesh data, phase, active hand, grip tag, contact, and range validity never enter the continuous feature vector.
- The held-out split is deterministic and object-disjoint. No sequence from a held-out object may appear in the database partition.
- Binary files are little-endian, schema version 1, and start with exact eight-byte magics `G1INTDB1` and `G1INTFT1`.
- Publishing is atomic: validate a sibling temporary directory completely, then rename it to the requested output; a failed build leaves any prior output untouched.
- Unit tests require no network. Full-corpus validation uses the already-downloaded root `/home/ubuntu/datasets/GRAIL/data/pickup_table` and may be overridden by CLI arguments.
- Level 1's high-level caller remains out of scope. The later runtime API receives an explicit target/affordance command and returns accepted, succeeded, rejected, cancelled, or failed with a reason; it does not decide which household task to perform.

---

## Scope Boundary and Follow-on Plans

This is the first independently testable plan for the approved Level 1 design. It ends when canonical data replay and cross-language artifact parity pass. It does not add the `Interact` input, a pickup state machine, motion selection, warping, IK, attachment, carrying, or placement.

The next plans consume this gate in order:

1. whole-clip `Pick` primitive and `PickRequest -> InteractionResult` coordinator;
2. staged approach/reach matcher and pre-contact commit rule;
3. bounded root warp, hand IK, foot locking, and object attachment;
4. held-out evaluation harness and playable locomotion-to-pickup demo.

The future laundry-basket example belongs above this layer: a player, script, or VLM sequences `pick`, `carry`, `open`, and `place`; each motion primitive owns only its embodied execution and result.

## Frozen Repository Foundation

At execution time, first replay this branch's documentation commits onto the clean G1 foundation already produced at immutable commit `cbe90b7`:

```bash
cd /home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching
test "$(git branch --show-current)" = "g1-manipulation-motion-matching"
test -z "$(git status --porcelain)"
git rebase --onto cbe90b7 0cbb4e0
git merge-base --is-ancestor cbe90b7 HEAD
```

Expected: all commands exit 0, the branch remains `g1-manipulation-motion-matching`, and the design and this plan appear after `cbe90b7`. This imports the known-good flat G1 controller plus the tested generic `resources/g1_terrain_builder/{schema,sources,resample,kinematics}.py` conversion foundation without consuming any later terrain work.

## File and Responsibility Map

| Path | Responsibility |
| --- | --- |
| `requirements-interaction.txt` | Reproducible offline-builder dependencies. |
| `g1_skeleton.h` | Runtime G1 bone indices, parents, and skeleton signature. |
| `resources/__init__.py` | Stable Python package root for module-form CLI execution. |
| `resources/g1_interaction_builder/schema.py` | Frozen raw, canonical, labeled, feature, split, and artifact dataclasses/enums. |
| `resources/g1_interaction_builder/sources.py` | Match GRAIL robot/object/meta/USD files and load native records. |
| `resources/g1_interaction_builder/object_geometry.py` | Compute object-local USD bounds and cache dimensions by object identity. |
| `resources/g1_interaction_builder/conversion.py` | Coordinate conversion, canonical 25 Hz sampling, derivatives, hands, feet, and canonical clips. |
| `resources/g1_interaction_builder/phases.py` | Stable-contact detection, active-hand choice, phase labels, grasp estimate, and approach axis. |
| `resources/g1_interaction_builder/features.py` | Exact 71-D target-relative features and Holden-style group normalization. |
| `resources/g1_interaction_builder/splits.py` | Seeded object-disjoint database/evaluation partition. |
| `resources/g1_interaction_builder/artifacts.py` | Assemble, validate, write, read, and atomically publish schema-v1 artifacts. |
| `resources/g1_interaction_builder/build.py` | Per-clip build orchestration and rejection reporting. |
| `resources/build_g1_interaction_database.py` | User-facing build CLI. |
| `resources/validate_g1_interaction_database.py` | Standalone semantic and binary validator. |
| `resources/fetch_grail_pickup_table.py` | Optional authenticated source download with constrained allow-patterns. |
| `interaction_database.h` | Strict, dependency-free C++ schema-v1 artifact loader. |
| `interaction_probe.cpp` | Headless cross-language replay/digest executable. |
| `tests/python/interaction_fixture.py` | Deterministic synthetic source and canonical fixtures. |
| `tests/python/test_interaction_*.py` | Focused source, conversion, phase, feature, split, artifact, and CLI tests. |
| `tests/cpp/test_g1_skeleton.cpp` | Compile-time/runtime skeleton contract test. |
| `tests/cpp/test_interaction_database.cpp` | C++ corruption and array-shape loader tests. |
| `Makefile` | Portable Python, C++, probe, and Gate 1 verification targets in addition to the existing demo build. |
| `.gitignore` | Ignore the virtual environment, test build outputs, and generated interaction pack. |
| `README.md` | Exact setup, source, build, validation, and replay commands. |

## Frozen Python Interfaces

The following names and types are shared across tasks and must not drift:

```python
class InteractionHand(IntEnum):
    LEFT = 0
    RIGHT = 1

class InteractionPhase(IntEnum):
    APPROACH = 0
    REACH = 1
    CONTACT = 2
    LIFT = 3
    HOLD = 4

@dataclass(frozen=True)
class SourcePaths:
    sequence_id: str
    object_id: str
    robot: Path
    objects: Path
    meta: Path
    object_usd: Path

@dataclass
class RawInteractionClip:
    sequence_id: str
    object_id: str
    fps: float
    qpos: np.ndarray                 # (T, 36), root quaternion wxyz
    hand_dof: np.ndarray             # (T, 14)
    object_positions: np.ndarray     # (T, 3), source Z-up
    object_rotations: np.ndarray     # (T, 4), wxyz
    hand_contacts: np.ndarray        # (T, 2), uint8 [left, right]
    table_position: np.ndarray       # (3,), source Z-up
    table_rotation: np.ndarray       # (4,), wxyz
    table_size: np.ndarray           # (3,), source xyz dimensions
    object_dimensions: np.ndarray    # (3,), object-local xyz dimensions
    source_frames: np.ndarray        # (T,), int32

@dataclass
class CanonicalInteractionClip:
    sequence_id: str
    object_id: str
    fps: float
    positions: np.ndarray            # (T, 31, 3), local Holden hierarchy
    velocities: np.ndarray           # (T, 31, 3), local m/s
    rotations: np.ndarray            # (T, 31, 4), local wxyz
    angular_velocities: np.ndarray   # (T, 31, 3), local rad/s
    foot_contacts: np.ndarray        # (T, 2), uint8
    hand_contacts: np.ndarray        # (T, 2), uint8
    hand_dof: np.ndarray             # (T, 14)
    hand_dof_velocities: np.ndarray  # (T, 14), rad/s
    hand_positions: np.ndarray       # (T, 2, 3), world Y-up
    hand_rotations: np.ndarray       # (T, 2, 4), world wxyz
    object_positions: np.ndarray     # (T, 3), world Y-up
    object_rotations: np.ndarray     # (T, 4), world wxyz
    object_velocities: np.ndarray    # (T, 3), world m/s
    object_angular_velocities: np.ndarray # (T, 3), world rad/s
    table_position: np.ndarray       # (3,), world Y-up
    table_rotation: np.ndarray       # (4,), world wxyz
    table_size: np.ndarray           # (3,), Y-up axis order
    object_dimensions: np.ndarray    # (3,), object-local axis order
    source_frames: np.ndarray        # (T,), int32

@dataclass
class LabeledInteractionClip:
    motion: CanonicalInteractionClip
    active_hand: InteractionHand
    phases: np.ndarray               # (T,), uint8
    time_to_contact: np.ndarray      # (T,), float32 seconds
    contact_frame: int
    lift_frame: int
    hold_frame: int
    support_height: float
    grasp_position_object: np.ndarray     # (3,), hand in object
    grasp_rotation_object: np.ndarray     # (4,), hand in object, wxyz
    approach_direction_object: np.ndarray # (3,), normalized and horizontal

@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    stop: int

@dataclass
class FeatureSet:
    values: np.ndarray               # (N, 71), normalized float32
    offsets: np.ndarray              # (71,), float32
    scales: np.ndarray               # (71,), positive float32
    groups: tuple[FeatureGroup, ...]

@dataclass(frozen=True)
class EvaluationSplit:
    seed: int
    database_objects: tuple[str, ...]
    heldout_objects: tuple[str, ...]

@dataclass(frozen=True)
class PhaseConfig:
    stable_source_samples: int = 3
    max_relative_position_m: float = 0.02
    max_relative_angle_radians: float = 0.17453292519943295
    reach_seconds: float = 1.0
    lift_height_m: float = 0.05
    hold_speed_mps: float = 0.1
    hold_seconds: float = 0.2
    grasp_average_seconds: float = 0.2

class InteractionBuildError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code

class SourceValidationError(InteractionBuildError):
    pass

class ConversionValidationError(InteractionBuildError):
    pass

class InteractionValidationError(InteractionBuildError):
    pass

@dataclass
class InteractionArtifact:
    fps: int
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    hand_contacts: np.ndarray
    hand_dof: np.ndarray
    hand_dof_velocities: np.ndarray
    phases: np.ndarray
    active_hands: np.ndarray
    time_to_contact: np.ndarray
    object_positions: np.ndarray
    object_rotations: np.ndarray
    object_velocities: np.ndarray
    object_angular_velocities: np.ndarray
    table_positions: np.ndarray
    table_rotations: np.ndarray
    table_sizes: np.ndarray
    object_dimensions: np.ndarray
    grasp_positions_object: np.ndarray
    grasp_rotations_object: np.ndarray
    approach_directions_object: np.ndarray
    source_frames: np.ndarray

G1_SKELETON = SkeletonSpec(
    names=(
        "Simulation", "Hips",
        "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee", "RightAnkle", "RightToe",
        "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
        "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
        "RightWristRoll", "RightWristPitch", "RightWrist",
    ),
    parents=np.array(
        [-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
         15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29],
        np.int32,
    ),
)
```

At module import, assert `G1_SKELETON.signature()` equals
`6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7`.

## Frozen Binary Layout

All scalar fields and arrays are little-endian and tightly packed without C++ struct padding.

`interaction_database.bin`:

```text
char[8] magic = G1INTDB1
u32 version = 1
u32 endian_marker = 0x01020304
u32 fps_numerator = 25
u32 fps_denominator = 1
u32 frame_count
u32 bone_count = 31
u32 clip_count
u32 hand_dof_count = 14
i32 parents[bone_count]
i32 range_starts[clip_count]
i32 range_stops[clip_count]
f32 positions[frame_count][bone_count][3]
f32 velocities[frame_count][bone_count][3]
f32 rotations[frame_count][bone_count][4]
f32 angular_velocities[frame_count][bone_count][3]
u8  foot_contacts[frame_count][2]
u8  hand_contacts[frame_count][2]
f32 hand_dof[frame_count][hand_dof_count]
f32 hand_dof_velocities[frame_count][hand_dof_count]
u8  phases[frame_count]
u8  active_hands[clip_count]
f32 time_to_contact[frame_count]
f32 object_positions[frame_count][3]
f32 object_rotations[frame_count][4]
f32 object_velocities[frame_count][3]
f32 object_angular_velocities[frame_count][3]
f32 table_positions[clip_count][3]
f32 table_rotations[clip_count][4]
f32 table_sizes[clip_count][3]
f32 object_dimensions[clip_count][3]
f32 grasp_positions_object[clip_count][3]
f32 grasp_rotations_object[clip_count][4]
f32 approach_directions_object[clip_count][3]
i32 source_frames[frame_count]
```

`interaction_features.bin`:

```text
char[8] magic = G1INTFT1
u32 version = 1
u32 endian_marker = 0x01020304
u32 frame_count
u32 dimension = 71
u32 group_count = 5
u32 group_starts[group_count] = 0,33,45,57,65
u32 group_stops[group_count] = 33,45,57,65,71
f32 offsets[dimension]
f32 scales[dimension]
f32 values[frame_count][dimension]
```

The writer rejects any non-C-contiguous array, then explicitly converts accepted arrays to the declared little-endian dtype before writing. The loader rejects trailing bytes as well as truncated files.

### Task 1: Establish the Frozen G1 Foundation and Portable Test Harness

**Files:**
- Create: `requirements-interaction.txt`
- Create: `g1_skeleton.h`
- Create: `resources/__init__.py`
- Create: `tests/cpp/test_g1_skeleton.cpp`
- Modify: `resources/g1_terrain_builder/kinematics.py`
- Modify: `tests/python/test_kinematics.py`
- Modify: `Makefile`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: immutable commit `cbe90b7`, `G1Kinematics(xml_path: str)`, and `convert_source_clip(source, kinematics, target_fps)`.
- Produces: `g1_skeleton::{Bone, kBoneNames, kParents, kSkeletonSignature}` and portable `make test-python` / `make test-cpp` targets.

- [ ] **Step 1: Rebase onto and verify the immutable foundation**

Run the commands in **Frozen Repository Foundation** exactly.

Expected: exit 0 and `git log --oneline --decorate -4` shows this branch's documentation commits immediately above `cbe90b7`.

- [ ] **Step 2: Write the failing skeleton/import tests**

Create `tests/cpp/test_g1_skeleton.cpp`:

```cpp
#include "g1_skeleton.h"
#include <cassert>
#include <string_view>

int main() {
    using namespace g1_skeleton;
    static_assert(BoneCount == 31);
    static_assert(LeftToe == 7 && RightToe == 13);
    static_assert(LeftWrist == 23 && RightWrist == 30);
    assert(kParents[Simulation] == -1);
    assert(kParents[Hips] == Simulation);
    assert(kParents[LeftWrist] == LeftWristPitch);
    assert(kParents[RightWrist] == RightWristPitch);
    assert(kBoneNames[Spine2] == std::string_view("Spine2"));
    assert(kSkeletonSignature == std::string_view(
        "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7"));
}
```

In `tests/python/test_kinematics.py`, replace the hard-coded XML constant with:

```python
import os

G1_XML = os.environ.get(
    "G1_XML",
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
)
```

Run:

```bash
g++ -std=c++17 -I. tests/cpp/test_g1_skeleton.cpp -o /tmp/test_g1_skeleton
python -m unittest tests.python.test_kinematics -v
```

Expected: C++ compilation fails because `g1_skeleton.h` does not exist; Python fails because `kinematics.py` imports `quat` through the other checkout's absolute path when run outside that layout.

- [ ] **Step 3: Add the exact skeleton and worktree-relative import**

Create `g1_skeleton.h` with this public contract:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace g1_skeleton {
enum Bone : int32_t {
    Simulation, Hips,
    LeftHipPitch, LeftHipRoll, LeftHipYaw, LeftKnee, LeftAnkle, LeftToe,
    RightHipPitch, RightHipRoll, RightHipYaw, RightKnee, RightAnkle, RightToe,
    Spine, Spine1, Spine2,
    LeftShoulderPitch, LeftShoulderRoll, LeftShoulderYaw, LeftElbow,
    LeftWristRoll, LeftWristPitch, LeftWrist,
    RightShoulderPitch, RightShoulderRoll, RightShoulderYaw, RightElbow,
    RightWristRoll, RightWristPitch, RightWrist,
    BoneCount
};

inline constexpr std::array<std::string_view, BoneCount> kBoneNames = {
    "Simulation", "Hips",
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
    "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee", "RightAnkle", "RightToe",
    "Spine", "Spine1", "Spine2",
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
    "LeftWristRoll", "LeftWristPitch", "LeftWrist",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWrist"
};

inline constexpr std::array<int32_t, BoneCount> kParents = {
    -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
};

inline constexpr std::string_view kSkeletonSignature =
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7";
static_assert(kBoneNames.size() == BoneCount);
static_assert(kParents.size() == BoneCount);
}
```

In `resources/g1_terrain_builder/kinematics.py`, remove the absolute `sys.path.insert` call and use:

```python
from resources import quat as holden_quat
```

Create `requirements-interaction.txt`:

```text
numpy>=2.0,<3
scipy>=1.13,<2
joblib>=1.4,<2
mujoco>=3.2,<4
usd-core>=24.5,<26
huggingface-hub>=0.24,<2
```

Create `resources/__init__.py`:

```python
"""Offline builders and runtime resources for motion matching."""
```

Append to `.gitignore`:

```text
.venv/
build/tests/
interaction_probe
resources/g1_interaction/
```

- [ ] **Step 4: Add portable test targets and run the baseline**

Append to `Makefile` using tab-indented recipes:

```make
CXX ?= g++
CPP_TEST_FLAGS ?= -std=c++17 -Wall -Wextra -Werror -pedantic -I.
CPP_TEST_DIR := build/tests
CPP_TEST_BINS := $(CPP_TEST_DIR)/test_g1_skeleton

.PHONY: test-python test-cpp test-interaction

$(CPP_TEST_DIR):
	mkdir -p $@

$(CPP_TEST_DIR)/test_g1_skeleton: tests/cpp/test_g1_skeleton.cpp g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

test-python:
	python -m unittest discover -s tests/python -t . -v

test-cpp: $(CPP_TEST_BINS)
	@for test_bin in $(CPP_TEST_BINS); do $$test_bin || exit 1; done

test-interaction: test-python test-cpp
```

Run:

```bash
python -m pip install -r requirements-interaction.txt
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml make test-interaction
```

Expected: every inherited Python test reports `ok`, `test_g1_skeleton` exits 0, and the existing controller resources remain unmodified.

- [ ] **Step 5: Commit the foundation**

```bash
git add .gitignore Makefile requirements-interaction.txt g1_skeleton.h resources/__init__.py \
  resources/g1_terrain_builder/kinematics.py tests/python/test_kinematics.py \
  tests/cpp/test_g1_skeleton.cpp
git commit -m "build: establish isolated G1 interaction foundation"
```

### Task 2: Load Matched GRAIL Tabletop Sources and Object Bounds

**Files:**
- Create: `resources/g1_interaction_builder/__init__.py`
- Create: `resources/g1_interaction_builder/schema.py`
- Create: `resources/g1_interaction_builder/sources.py`
- Create: `resources/g1_interaction_builder/object_geometry.py`
- Create: `tests/python/interaction_fixture.py`
- Create: `tests/python/test_interaction_sources.py`

**Interfaces:**
- Consumes: GRAIL directories `robot/`, `objects/`, `meta/`, and `object_usd/`.
- Produces: `discover_source_paths(root: Path) -> list[SourcePaths]`, `load_raw_interaction(paths: SourcePaths, object_dimensions: np.ndarray) -> RawInteractionClip`, `read_usd_dimensions(path: Path) -> np.ndarray`, and `build_dimension_catalog(paths: Sequence[SourcePaths]) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write a deterministic GRAIL fixture and failing loader tests**

In `tests/python/interaction_fixture.py`, define `write_source_fixture(root: Path, sequence_id="pickup_table__cup_2__001", object_id="cup_2", frames=25) -> SourcePaths`. It must create four subdirectories and write three one-record joblib files with these exact keys:

```python
robot_record = {
    "dof": np.zeros((frames, 29), np.float32),
    "root_trans_offset": np.zeros((frames, 3), np.float32),
    "root_rot": np.tile(np.array([0, 0, 0, 1], np.float32), (frames, 1)),
    "fps": 25.0,
    "hand_dof_pos": np.zeros((frames, 14), np.float32),
}
object_record = {
    "root_pos": np.zeros((frames, 1, 3), np.float32),
    "root_quat": np.tile(np.array([0, 0, 0, 1], np.float32), (frames, 1, 1)),
    "fps": 25.0,
    "contact_points_left_hand": {
        i: (np.array([[0.0, 0.0, 0.0]], np.float32) if i >= 10 else np.empty((0, 3), np.float32))
        for i in range(frames)
    },
    "contact_points_right_hand": {i: np.empty((0, 3), np.float32) for i in range(frames)},
}
meta_record = {
    "object_name": object_id,
    "table_pos": np.array([0.0, 0.0, 0.75], np.float32),
    "table_quat": np.array([0, 0, 0, 1], np.float32),
    "table_size": np.array([1.2, 0.8, 0.05], np.float32),
}
```

Use `joblib.dump({sequence_id: record}, path)` for each record and create an empty same-basename `.usd` path because USD parsing is tested through an injected fake in this unit.

In `tests/python/test_interaction_sources.py`, add:

```python
class InteractionSourceTests(unittest.TestCase):
    def test_discovery_requires_complete_sorted_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = write_source_fixture(root)
            found = discover_source_paths(root)
            self.assertEqual(found, [expected])
            expected.meta.unlink()
            with self.assertRaisesRegex(ValueError, "missing meta"):
                discover_source_paths(root)

    def test_loads_native_g1_hands_contacts_and_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            clip = load_raw_interaction(
                paths, np.array([0.08, 0.12, 0.20], np.float32))
            self.assertEqual(clip.qpos.shape, (25, 36))
            self.assertEqual(clip.hand_dof.shape, (25, 14))
            np.testing.assert_array_equal(clip.hand_contacts[:10], 0)
            np.testing.assert_array_equal(clip.hand_contacts[10:, 0], 1)
            np.testing.assert_allclose(clip.qpos[:, 3], 1.0)
            np.testing.assert_allclose(clip.table_size, [1.2, 0.8, 0.05])
            np.testing.assert_allclose(clip.object_dimensions, [0.08, 0.12, 0.20])

    def test_nonempty_contact_points_not_dictionary_membership_define_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_fixture(Path(tmp))
            clip = load_raw_interaction(paths, np.ones(3, np.float32))
            self.assertEqual(int(clip.hand_contacts[:10].sum()), 0)
```

Run: `python -m unittest tests.python.test_interaction_sources -v`

Expected: import failure for `resources.g1_interaction_builder`.

- [ ] **Step 2: Implement and validate the frozen source schema**

Add the enum/dataclass declarations from **Frozen Python Interfaces** to `schema.py`. Each `validate()` method must enforce its documented shape, finite values, positive FPS, unit quaternions within `1e-4`, consistent frame counts, `uint8` contacts in `{0,1}`, strictly nondecreasing source frames, positive table/object dimensions, and `qpos.shape[1] == 36`. Error messages start with `sequence_id`.

Implement these source helpers in `sources.py`:

```python
SEQUENCE_RE = re.compile(r"^pickup_table__(.+)__([0-9]{3})$")

def object_id_from_sequence(sequence_id: str) -> str:
    match = SEQUENCE_RE.fullmatch(sequence_id)
    if match is None:
        raise ValueError(f"invalid pickup-table sequence id: {sequence_id}")
    return match.group(1)

def _single_record(path: Path) -> dict:
    records = joblib.load(path)
    if not isinstance(records, dict) or len(records) != 1:
        raise ValueError(f"{path}: expected exactly one record")
    return next(iter(records.values()))

def _contacts(record: dict, key: str, frames: int) -> np.ndarray:
    points = record[key]
    return np.array([
        int(len(np.asarray(points.get(i, points.get(str(i), np.empty((0, 3)))))) > 0)
        for i in range(frames)
    ], dtype=np.uint8)
```

`discover_source_paths` sorts by `sequence_id`, derives IDs through `object_id_from_sequence`, and raises one error listing every missing modality. `load_raw_interaction` normalizes quaternions, converts both root/object/table `xyzw` values to `wxyz`, unwraps singleton object axes, verifies all source FPS values are equal, builds qpos `[root xyz, root wxyz, 29 dof]`, and calls `validate()` before returning.

- [ ] **Step 3: Add USD-bound extraction with identity consistency checks**

In `object_geometry.py`, implement:

```python
def read_usd_dimensions(path: Path) -> np.ndarray:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open object USD: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    dimensions = np.asarray(world.GetSize(), np.float32)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError(f"invalid object bounds in {path}: {dimensions}")
    return dimensions

def build_dimension_catalog(paths: Sequence[SourcePaths]) -> dict[str, np.ndarray]:
    catalog: dict[str, np.ndarray] = {}
    for source in paths:
        size = read_usd_dimensions(source.object_usd)
        if source.object_id in catalog and not np.allclose(
            catalog[source.object_id], size, rtol=0.0, atol=1e-3
        ):
            raise ValueError(f"inconsistent USD bounds for object {source.object_id}")
        catalog[source.object_id] = size
    return catalog
```

Add tests that monkeypatch `read_usd_dimensions` to return equal dimensions for two sequences of one object and unequal dimensions for the same object. Expected: equal values produce one catalog row; unequal values raise `inconsistent USD bounds`.

- [ ] **Step 4: Run source tests and inspect one real clip**

```bash
python -m unittest tests.python.test_interaction_sources -v
python - <<'PY'
from pathlib import Path
from resources.g1_interaction_builder.object_geometry import read_usd_dimensions
from resources.g1_interaction_builder.sources import discover_source_paths, load_raw_interaction
root = Path('/home/ubuntu/datasets/GRAIL/data/pickup_table')
source = discover_source_paths(root)[0]
size = read_usd_dimensions(source.object_usd)
clip = load_raw_interaction(source, size)
print(source.sequence_id, clip.qpos.shape, clip.hand_dof.shape, size.tolist())
PY
```

Expected: all unit tests report `ok`; the real inspection prints one sequence, `(250, 36)`, `(250, 14)`, and three positive finite dimensions.

- [ ] **Step 5: Commit the source boundary**

```bash
git add resources/g1_interaction_builder tests/python/interaction_fixture.py \
  tests/python/test_interaction_sources.py
git commit -m "feat: load GRAIL tabletop interaction sources"
```

### Task 3: Convert Native G1 and Scene State to Canonical 25 Hz Motion

**Files:**
- Create: `resources/g1_interaction_builder/conversion.py`
- Create: `tests/python/test_interaction_conversion.py`
- Modify: `resources/g1_terrain_builder/kinematics.py`

**Interfaces:**
- Consumes: `RawInteractionClip`, `G1Kinematics`, generic resampling and FK helpers.
- Produces: `convert_interaction(raw: RawInteractionClip, kinematics: G1Kinematics, target_fps: float = 25.0) -> tuple[CanonicalInteractionClip, SkeletonSpec, dict[str, float]]`.

- [ ] **Step 1: Write failing duration, basis, derivative, and clip-boundary tests**

Create tests that assert:

```python
class InteractionConversionTests(unittest.TestCase):
    def test_derivatives_do_not_wrap_or_cross_a_clip(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], np.float32)
        v = finite_difference_vectors(x, 2.0)
        np.testing.assert_allclose(v[:, 0], [2.0, 3.0, 4.0])

    def test_contact_resampling_uses_nearest_source_frame(self):
        source_frames = np.array([0, 0, 1, 1, 2], np.int32)
        contacts = np.array([[0, 0], [1, 0], [1, 0]], np.uint8)
        np.testing.assert_array_equal(
            resample_discrete_by_source_frame(contacts, source_frames),
            [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]],
        )

    def test_zup_size_axis_order_becomes_yup(self):
        np.testing.assert_allclose(
            size_zup_to_yup(np.array([1.0, 2.0, 3.0])), [1.0, 3.0, 2.0])
```

Add an integration test using `G1_XML` and the synthetic source fixture. It calls `convert_interaction(raw, kinematics, target_fps=25.0)` and asserts 25 frames for a 25-frame 25 Hz clip, 31 bones, 25 FPS, normalized rotations, the exact skeleton signature, no non-finite arrays, and zero duration error for the native-rate fixture.

Run: `python -m unittest tests.python.test_interaction_conversion -v`

Expected: import failure for `conversion.py`.

- [ ] **Step 2: Expose reusable basis helpers and implement within-clip derivatives**

In `kinematics.py`, expose these thin wrappers around the existing basis quaternion:

```python
def vectors_zup_to_yup(values: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(Q_ZUP_TO_YUP, np.asarray(values).shape[:-1] + (4,))
    return holden_quat.mul_vec(q, np.asarray(values))

def quaternions_zup_to_yup(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    q = np.broadcast_to(Q_ZUP_TO_YUP, values.shape)
    qi = np.broadcast_to(holden_quat.inv(Q_ZUP_TO_YUP), values.shape)
    return holden_quat.normalize(holden_quat.mul(holden_quat.mul(q, values), qi))
```

In `convert_source_clip`, retain the reconstructed world rotations and enforce
the design's rotational FK tolerance next to the existing position check:

```python
exported_gp, exported_gq = forward_local_hierarchy(
    positions.astype(np.float64), rotations.astype(np.float64),
    skeleton.parents)
fk_error = float(np.max(np.linalg.norm(
    exported_gp[:, 1:] - gp, axis=-1)))
dots = np.clip(np.abs(np.sum(exported_gq[:, 1:] * gq, axis=-1)), 0.0, 1.0)
fk_rotation_error_degrees = float(np.degrees(np.max(2.0 * np.arccos(dots))))
if fk_error > 0.001:
    raise ValueError(f"{source.name}: exported FK error {fk_error} m")
if fk_rotation_error_degrees > 0.1:
    raise ValueError(
        f"{source.name}: exported rotational FK error "
        f"{fk_rotation_error_degrees} degrees")
```

Add `"fk_rotation_max_error_degrees": fk_rotation_error_degrees` to the returned
numeric report and assert it is at most `0.1` in `test_convert_source_clip_prepends_simulation_bone`.

In `conversion.py`, implement endpoint-safe derivatives:

```python
def finite_difference_vectors(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, np.float64)
    out = np.empty_like(values)
    if len(values) < 2:
        out.fill(0.0)
    else:
        out[0] = (values[1] - values[0]) * fps
        out[-1] = (values[-1] - values[-2]) * fps
        if len(values) > 2:
            out[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return out.astype(np.float32)

def finite_difference_quaternions(values: np.ndarray, fps: float) -> np.ndarray:
    """Differentiate rotations in their containing (spatial) frame."""
    q = holden_quat.unroll(holden_quat.normalize(np.asarray(values, np.float64)))
    out = np.zeros(q.shape[:-1] + (3,), np.float64)
    if len(q) > 1:
        out[0] = holden_quat.to_scaled_angle_axis(holden_quat.mul(q[1], holden_quat.inv(q[0]))) * fps
        out[-1] = holden_quat.to_scaled_angle_axis(holden_quat.mul(q[-1], holden_quat.inv(q[-2]))) * fps
    if len(q) > 2:
        delta = holden_quat.mul(q[2:], holden_quat.inv(q[:-2]))
        out[1:-1] = holden_quat.to_scaled_angle_axis(delta) * (0.5 * fps)
    return out.astype(np.float32)

def resample_discrete_by_source_frame(
    values: np.ndarray, source_frames: np.ndarray,
) -> np.ndarray:
    source_frames = np.asarray(source_frames, np.int64)
    if np.any(source_frames < 0) or np.any(source_frames >= len(values)):
        raise ConversionValidationError(
            "frame_count_mismatch", "resampled source frame is out of range")
    return np.asarray(values)[source_frames].copy()

def size_zup_to_yup(size: np.ndarray) -> np.ndarray:
    size = np.asarray(size, np.float32)
    if size.shape != (3,):
        raise ConversionValidationError(
            "invalid_dimensions", f"expected three dimensions, got {size.shape}")
    return size[[0, 2, 1]].copy()
```

- [ ] **Step 3: Implement canonical conversion**

`convert_interaction` must perform this exact sequence:

1. Wrap `RawInteractionClip.qpos` in `SourceClip` and call `convert_source_clip(source, kinematics, target_fps=25.0)`.
2. Reconstruct world transforms with `forward_local_hierarchy` and select wrists `[23, 30]` and toes `[7, 13]` from the 31-bone canonical skeleton.
3. Convert object/table position and rotation from Z-up to Y-up before resampling.
4. Resample `hand_dof`, object position, and object rotation by time; sample discrete contacts with `motion.source_frames`.
5. Convert dimension axis order with `size_zup_to_yup([x,y,z]) -> [x,z,y]` for table dimensions. Keep object-local dimensions in USD local axis order.
6. Compute pose derivatives with only this clip's frames. Quaternion derivatives
   are expressed in each rotation's containing frame: the parent frame for local
   bones and world space for world object orientations.
7. Mark each foot in contact when its world speed is below `0.15 m/s` and world Y height is below `0.06 m` above the minimum toe height in this clip.
8. Store world hand transforms, call `CanonicalInteractionClip.validate()`, and return the inherited FK/duration report plus `target_fps=25.0` and contact counts.

Use `resample_quaternions_wxyz` for every quaternion stream. Do not interpolate contacts or source frame numbers.

Implement that sequence with this function body; `SourceClip` and the conversion
helpers are imported from `resources.g1_terrain_builder`:

```python
def convert_interaction(
    raw: RawInteractionClip,
    kinematics: G1Kinematics,
    target_fps: float = 25.0,
) -> tuple[CanonicalInteractionClip, SkeletonSpec, dict[str, float]]:
    if target_fps != 25.0:
        raise ConversionValidationError(
            "fps_mismatch", f"schema v1 requires 25 Hz, got {target_fps}")
    raw.validate()
    model = kinematics.model
    joint_ids = np.flatnonzero(np.asarray(model.jnt_qposadr) >= 7)
    if model.nq != 36 or len(joint_ids) != 29:
        raise ConversionValidationError(
            "skeleton_mismatch",
            f"expected nq=36 and 29 body joints, got nq={model.nq}, joints={len(joint_ids)}")
    for joint_id in joint_ids:
        if not bool(model.jnt_limited[joint_id]):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        values = raw.qpos[:, qpos_index]
        if np.any(values < lower - 1e-4) or np.any(values > upper + 1e-4):
            raise ConversionValidationError(
                "joint_limit_violation",
                f"joint {model.joint(int(joint_id)).name} outside [{lower}, {upper}]")
    source = SourceClip(
        raw.sequence_id, raw.fps, raw.qpos, raw.source_frames, raw.object_id)
    try:
        motion, skeleton, report = convert_source_clip(
            source, kinematics, target_fps=target_fps)
    except ValueError as error:
        message = str(error)
        known = (
            ("rotational FK error", "fk_rotation_error"),
            ("exported FK error", "fk_error"),
            ("duration error", "duration_error"),
            ("quaternion", "invalid_quaternion"),
            ("non-finite", "non_finite"),
        )
        for fragment, code in known:
            if fragment in message:
                raise ConversionValidationError(code, message) from error
        raise
    if skeleton.signature() != G1_SKELETON.signature():
        raise ConversionValidationError(
            "skeleton_mismatch", skeleton.signature())

    world_positions, world_rotations = forward_local_hierarchy(
        motion.positions.astype(np.float64),
        motion.rotations.astype(np.float64),
        skeleton.parents,
    )
    object_positions = resample_vectors(
        vectors_zup_to_yup(raw.object_positions), raw.fps, target_fps)
    object_rotations = resample_quaternions_wxyz(
        quaternions_zup_to_yup(raw.object_rotations), raw.fps, target_fps)
    hand_dof = resample_vectors(raw.hand_dof, raw.fps, target_fps)
    hand_contacts = resample_discrete_by_source_frame(
        raw.hand_contacts, motion.source_frames)

    velocities = finite_difference_vectors(motion.positions, target_fps)
    angular_velocities = finite_difference_quaternions(
        motion.rotations, target_fps)
    hand_dof_velocities = finite_difference_vectors(hand_dof, target_fps)
    object_velocities = finite_difference_vectors(
        object_positions, target_fps)
    object_angular_velocities = finite_difference_quaternions(
        object_rotations, target_fps)
    toe_positions = world_positions[:, [7, 13]]
    toe_speeds = np.linalg.norm(
        finite_difference_vectors(toe_positions, target_fps), axis=-1)
    toe_floor = np.min(toe_positions[:, :, 1], axis=0, keepdims=True)
    foot_contacts = np.logical_and(
        toe_speeds < 0.15,
        toe_positions[:, :, 1] - toe_floor < 0.06,
    ).astype(np.uint8)

    clip = CanonicalInteractionClip(
        sequence_id=raw.sequence_id,
        object_id=raw.object_id,
        fps=target_fps,
        positions=motion.positions.astype(np.float32),
        velocities=velocities,
        rotations=motion.rotations.astype(np.float32),
        angular_velocities=angular_velocities,
        foot_contacts=foot_contacts,
        hand_contacts=hand_contacts,
        hand_dof=hand_dof.astype(np.float32),
        hand_dof_velocities=hand_dof_velocities,
        hand_positions=world_positions[:, [23, 30]].astype(np.float32),
        hand_rotations=world_rotations[:, [23, 30]].astype(np.float32),
        object_positions=object_positions.astype(np.float32),
        object_rotations=object_rotations.astype(np.float32),
        object_velocities=object_velocities,
        object_angular_velocities=object_angular_velocities,
        table_position=vectors_zup_to_yup(raw.table_position).astype(np.float32),
        table_rotation=quaternions_zup_to_yup(raw.table_rotation).astype(np.float32),
        table_size=size_zup_to_yup(raw.table_size),
        object_dimensions=raw.object_dimensions.astype(np.float32),
        source_frames=motion.source_frames.astype(np.int32),
    )
    clip.validate()
    report = dict(report)
    report.update({
        "target_fps": target_fps,
        "left_hand_contact_frames": int(hand_contacts[:, 0].sum()),
        "right_hand_contact_frames": int(hand_contacts[:, 1].sum()),
    })
    return clip, skeleton, report
```

- [ ] **Step 4: Run focused and inherited conversion tests**

```bash
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  python -m unittest tests.python.test_resample tests.python.test_kinematics \
  tests.python.test_interaction_conversion -v
```

Expected: all tests report `ok`; the integration test reports no FK, duration, quaternion, or shape failure.

- [ ] **Step 5: Commit canonical conversion**

```bash
git add resources/g1_terrain_builder/kinematics.py \
  resources/g1_interaction_builder/conversion.py \
  tests/python/test_interaction_conversion.py
git commit -m "feat: convert tabletop motion to canonical 25 Hz G1"
```

### Task 4: Derive Stable Contact, Monotonic Phases, Grasp Frame, and Approach Axis

**Files:**
- Create: `resources/g1_interaction_builder/phases.py`
- Create: `tests/python/test_interaction_phases.py`
- Modify: `tests/python/interaction_fixture.py`

**Interfaces:**
- Consumes: `CanonicalInteractionClip`.
- Produces: `derive_interaction_labels(clip: CanonicalInteractionClip, config: PhaseConfig = PhaseConfig()) -> LabeledInteractionClip`, where `PhaseConfig` freezes all thresholds from Global Constraints.

- [ ] **Step 1: Write failing synthetic semantic tests**

Extend the fixture with `canonical_pickup_fixture(active_hand=InteractionHand.RIGHT)`. It produces 75 frames at 25 Hz, holds the object at Y=`0.75` through frame 37, raises it linearly to Y=`0.90` by frame 49, then holds it; right-hand contact is true from frame 38 onward, and source frame numbers are `arange(75)`. The active wrist stays at a fixed `[0.02, 0.00, -0.03]` transform in object space during contact. Populate hand-DOF velocity with zeros and object linear/angular velocity with the same `finite_difference_vectors` / containing-frame `finite_difference_quaternions` functions used by conversion. Because object orientations are world transforms, store that quaternion derivative directly without rotating it again.

Add these reusable fixture transformations so later tests exercise exact world
invariance and object identity rather than hand-built feature rows:

```python
def transform_fixture_world(
    clip: CanonicalInteractionClip,
    translation: Sequence[float],
    yaw_degrees: float,
) -> CanonicalInteractionClip:
    out = copy.deepcopy(clip)
    translation = np.asarray(translation, np.float64)
    yaw = np.deg2rad(yaw_degrees)
    heading = np.array([np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0])

    def move_positions(values: np.ndarray) -> np.ndarray:
        return holden_quat.mul_vec(heading, values) + translation

    def move_rotations(values: np.ndarray) -> np.ndarray:
        q = np.broadcast_to(heading, values.shape)
        return holden_quat.normalize(holden_quat.mul(q, values))

    out.positions[:, 0] = move_positions(out.positions[:, 0])
    out.rotations[:, 0] = move_rotations(out.rotations[:, 0])
    out.velocities[:, 0] = holden_quat.mul_vec(heading, out.velocities[:, 0])
    out.angular_velocities[:, 0] = holden_quat.mul_vec(
        heading, out.angular_velocities[:, 0])
    out.hand_positions = move_positions(out.hand_positions)
    out.hand_rotations = move_rotations(out.hand_rotations)
    out.object_positions = move_positions(out.object_positions)
    out.object_rotations = move_rotations(out.object_rotations)
    out.table_position = move_positions(out.table_position)
    out.table_rotation = move_rotations(out.table_rotation)
    out.validate()
    return out

def labeled_clips_for_objects(object_ids: Sequence[str]) -> list[LabeledInteractionClip]:
    base = derive_interaction_labels(canonical_pickup_fixture())
    clips = []
    for index, object_id in enumerate(object_ids):
        motion = copy.deepcopy(base.motion)
        motion.sequence_id = f"pickup_table__{object_id}__{index:03d}"
        motion.object_id = object_id
        clips.append(dataclasses.replace(base, motion=motion))
    return clips
```

Test these exact outcomes:

```python
class InteractionPhaseTests(unittest.TestCase):
    def test_derives_one_active_hand_and_monotonic_phases(self):
        labeled = derive_interaction_labels(canonical_pickup_fixture())
        self.assertEqual(labeled.active_hand, InteractionHand.RIGHT)
        self.assertEqual(labeled.contact_frame, 38)
        self.assertEqual(labeled.phases[12], InteractionPhase.APPROACH)
        self.assertEqual(labeled.phases[13], InteractionPhase.REACH)
        self.assertEqual(labeled.phases[38], InteractionPhase.CONTACT)
        self.assertGreaterEqual(labeled.lift_frame, 38)
        self.assertGreater(labeled.hold_frame, labeled.lift_frame)
        self.assertTrue(np.all(np.diff(labeled.phases.astype(np.int16)) >= 0))
        self.assertAlmostEqual(labeled.time_to_contact[13], 1.0, places=5)
        self.assertAlmostEqual(labeled.time_to_contact[38], 0.0, places=5)

    def test_grasp_is_object_local_and_world_transform_invariant(self):
        a = derive_interaction_labels(canonical_pickup_fixture())
        b = derive_interaction_labels(transform_fixture_world(
            canonical_pickup_fixture(), translation=[3.0, 0.0, -2.0], yaw_degrees=73.0))
        np.testing.assert_allclose(a.grasp_position_object, b.grasp_position_object, atol=1e-5)
        self.assertLess(quaternion_angle(a.grasp_rotation_object, b.grasp_rotation_object), 1e-5)

    def test_rejects_two_hands_and_unstable_contact(self):
        both = canonical_pickup_fixture()
        both.hand_contacts[38:, :] = 1
        both.hand_positions[:, 0] = both.hand_positions[:, 1]
        both.hand_rotations[:, 0] = both.hand_rotations[:, 1]
        with self.assertRaisesRegex(ValueError, "ambiguous_active_hand"):
            derive_interaction_labels(both)
        unstable = canonical_pickup_fixture()
        unstable.hand_positions[38::2, 1, 0] += 0.05
        with self.assertRaisesRegex(ValueError, "no_stable_contact"):
            derive_interaction_labels(unstable)
```

Run: `python -m unittest tests.python.test_interaction_phases -v`

Expected: import failure for `phases.py`.

- [ ] **Step 2: Implement object-relative transforms and stable-contact detection**

Use the exact transform direction:

```python
def hand_in_object(
    hand_position: np.ndarray,
    hand_rotation: np.ndarray,
    object_position: np.ndarray,
    object_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inv_object = holden_quat.inv(object_rotation)
    return (
        holden_quat.mul_vec(inv_object, hand_position - object_position),
        holden_quat.abs(holden_quat.normalize(
            holden_quat.mul(inv_object, hand_rotation))),
    )

def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))
```

Stable-contact detection scans triples of consecutive 25 Hz source samples. A triple passes only when all three samples have non-empty contact, endpoint hand-in-object position changes by at most `0.02 m`, and endpoint rotation changes by at most `radians(10)`. The contact frame is the first sample in the first passing triple. Exactly one hand must pass. Keep the distinct-`source_frames` collapse in the helper so future resampled inputs obey the same source-sample rule.

Implement the scan exactly:

```python
def _stable_contact_frame(
    clip: CanonicalInteractionClip,
    hand: InteractionHand,
    config: PhaseConfig,
) -> int | None:
    position, rotation = hand_in_object(
        clip.hand_positions[:, int(hand)],
        clip.hand_rotations[:, int(hand)],
        clip.object_positions,
        clip.object_rotations,
    )
    distinct = np.flatnonzero(np.r_[
        True, clip.source_frames[1:] != clip.source_frames[:-1]])
    for index in range(len(distinct) - config.stable_source_samples + 1):
        sample = distinct[index:index + config.stable_source_samples]
        if not np.all(clip.hand_contacts[sample, int(hand)]):
            continue
        window_positions = position[sample]
        pairwise_positions = np.linalg.norm(
            window_positions[:, None] - window_positions[None, :], axis=-1)
        if float(np.max(pairwise_positions)) > config.max_relative_position_m:
            continue
        window_rotations = rotation[sample]
        pairwise_angles = [
            quaternion_angle(window_rotations[a], window_rotations[b])
            for a in range(len(window_rotations))
            for b in range(a + 1, len(window_rotations))
        ]
        if max(pairwise_angles) > config.max_relative_angle_radians:
            continue
        return int(sample[0])
    return None
```

- [ ] **Step 3: Implement phase and grasp derivation**

Implement the complete labeling path:

```python
def derive_interaction_labels(
    clip: CanonicalInteractionClip,
    config: PhaseConfig = PhaseConfig(),
) -> LabeledInteractionClip:
    clip.validate()
    contacts = {
        hand: _stable_contact_frame(clip, hand, config)
        for hand in InteractionHand
    }
    valid = [(hand, frame) for hand, frame in contacts.items() if frame is not None]
    if len(valid) != 1:
        code = "no_stable_contact" if not valid else "ambiguous_active_hand"
        raise InteractionValidationError(code, clip.sequence_id)
    active_hand, contact_frame = valid[0]
    frames = len(clip.positions)
    active_contact = clip.hand_contacts[:, int(active_hand)].astype(bool)
    reach_frame = max(0, contact_frame - round(config.reach_seconds * clip.fps))
    time_to_contact = np.maximum(
        0.0, (contact_frame - np.arange(frames)) / clip.fps).astype(np.float32)
    support_height = float(clip.object_positions[contact_frame, 1])

    lifted = np.flatnonzero(
        active_contact[contact_frame:]
        & (clip.object_positions[contact_frame:, 1]
           >= support_height + config.lift_height_m))
    if not len(lifted):
        code = (
            "contact_lost_before_hold"
            if not np.all(active_contact[contact_frame:])
            else "no_five_centimeter_lift")
        raise InteractionValidationError(code, clip.sequence_id)
    lift_frame = contact_frame + int(lifted[0])
    if not np.all(active_contact[contact_frame:lift_frame + 1]):
        raise InteractionValidationError(
            "contact_lost_before_hold", clip.sequence_id)

    object_velocity = clip.object_velocities[:, 1]
    stable = active_contact & (np.abs(object_velocity) < config.hold_speed_mps)
    hold_samples = round(config.hold_seconds * clip.fps)
    hold_frame = None
    for start in range(lift_frame, frames - hold_samples + 1):
        if np.all(stable[start:start + hold_samples]):
            hold_frame = start
            break
    if hold_frame is None:
        code = (
            "contact_lost_before_hold"
            if not np.all(active_contact[contact_frame:])
            else "no_stable_hold")
        raise InteractionValidationError(code, clip.sequence_id)
    if not np.all(active_contact[contact_frame:hold_frame + hold_samples]):
        raise InteractionValidationError(
            "contact_lost_before_hold", clip.sequence_id)

    grasp_stop = min(
        frames, contact_frame + round(config.grasp_average_seconds * clip.fps))
    grasp_indices = np.flatnonzero(active_contact[contact_frame:grasp_stop]) + contact_frame
    grasp_positions, grasp_rotations = hand_in_object(
        clip.hand_positions[grasp_indices, int(active_hand)],
        clip.hand_rotations[grasp_indices, int(active_hand)],
        clip.object_positions[grasp_indices],
        clip.object_rotations[grasp_indices],
    )
    grasp_position = np.median(grasp_positions, axis=0)
    aligned = grasp_rotations.copy()
    aligned[np.sum(aligned * aligned[0], axis=1) < 0] *= -1
    grasp_rotation = holden_quat.normalize(np.mean(aligned, axis=0))
    if not np.isfinite(grasp_position).all() or not np.isfinite(grasp_rotation).all():
        raise InteractionValidationError("invalid_grasp", clip.sequence_id)

    approach_world = (
        clip.hand_positions[contact_frame, int(active_hand)]
        - clip.hand_positions[reach_frame, int(active_hand)])
    approach_object = holden_quat.mul_vec(
        holden_quat.inv(clip.object_rotations[contact_frame]), approach_world)
    approach_object[1] = 0.0
    approach_norm = float(np.linalg.norm(approach_object))
    if approach_norm < 1e-4:
        raise InteractionValidationError("invalid_approach", clip.sequence_id)
    approach_object /= approach_norm

    phases = np.full(frames, InteractionPhase.HOLD, np.uint8)
    phases[:reach_frame] = InteractionPhase.APPROACH
    phases[reach_frame:contact_frame] = InteractionPhase.REACH
    phases[contact_frame:lift_frame] = InteractionPhase.CONTACT
    phases[lift_frame:hold_frame] = InteractionPhase.LIFT
    return LabeledInteractionClip(
        motion=clip,
        active_hand=active_hand,
        phases=phases,
        time_to_contact=time_to_contact,
        contact_frame=contact_frame,
        lift_frame=lift_frame,
        hold_frame=hold_frame,
        support_height=support_height,
        grasp_position_object=grasp_position.astype(np.float32),
        grasp_rotation_object=grasp_rotation.astype(np.float32),
        approach_direction_object=approach_object.astype(np.float32),
    )
```

The half-open phase ranges are `[0, reach)`, `[reach, contact)`,
`[contact, lift)`, `[lift, hold)`, and `[hold, T)`, as encoded above.

- [ ] **Step 4: Run phase tests**

Run: `python -m unittest tests.python.test_interaction_phases -v`

Expected: all tests report `ok`, including exact frame labels, world invariance, and actionable rejection codes.

- [ ] **Step 5: Commit interaction semantics**

```bash
git add resources/g1_interaction_builder/phases.py \
  tests/python/interaction_fixture.py tests/python/test_interaction_phases.py
git commit -m "feat: derive pickup contact phases and grasp frames"
```

### Task 5: Build the Exact Target-relative Feature Matrix and Object Split

**Files:**
- Create: `resources/g1_interaction_builder/features.py`
- Create: `resources/g1_interaction_builder/splits.py`
- Create: `tests/python/test_interaction_features.py`
- Create: `tests/python/test_interaction_splits.py`

**Interfaces:**
- Consumes: ordered `Sequence[LabeledInteractionClip]`, a fully validated
  `EvaluationSplit`, and `SkeletonSpec`.
- Produces: production
  `build_database_features(all_clips, split, skeleton) -> FeatureSet`,
  `partition_clips(all_clips, split) -> (database_clips, heldout_clips)`,
  low-level database-only `build_features(clips, skeleton) -> FeatureSet`,
  `normalize_feature_groups(raw, groups) -> FeatureSet`, and
  `split_objects(clips, heldout_count: int = 20, seed: int = 20260714) -> EvaluationSplit`.

- [ ] **Step 1: Write failing dimension, sign-invariance, normalization,
  split-validation, and production-boundary leakage tests**

```python
class InteractionFeatureTests(unittest.TestCase):
    def test_feature_layout_is_exactly_seventy_one(self):
        feature_set = build_features([derive_interaction_labels(canonical_pickup_fixture())], G1_SKELETON)
        self.assertEqual(feature_set.values.shape, (75, 71))
        self.assertEqual(
            [(g.name, g.start, g.stop) for g in feature_set.groups],
            [("pose", 0, 33), ("trajectory", 33, 45),
             ("grasp", 45, 57), ("root_target", 57, 65),
             ("context", 65, 71)],
        )
        self.assertTrue(np.isfinite(feature_set.values).all())
        self.assertTrue(np.all(feature_set.scales > 0))

    def test_common_world_translation_and_yaw_leave_features_unchanged(self):
        clip = canonical_pickup_fixture()
        a = build_features([derive_interaction_labels(clip)], G1_SKELETON)
        moved = transform_fixture_world(clip, [4.0, 0.0, -3.0], 121.0)
        b = build_features([derive_interaction_labels(moved)], G1_SKELETON)
        np.testing.assert_allclose(a.values, b.values, atol=2e-4)

class InteractionSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_object_disjoint(self):
        clips = labeled_clips_for_objects(["a", "a", "b", "c", "d"])
        a = split_objects(clips, heldout_count=2, seed=7)
        b = split_objects(clips, heldout_count=2, seed=7)
        self.assertEqual(a, b)
        self.assertFalse(set(a.database_objects) & set(a.heldout_objects))
        self.assertEqual(len(a.heldout_objects), 2)

    def test_split_rejects_too_few_objects(self):
        with self.assertRaisesRegex(ValueError, "need at least 3 unique objects"):
            split_objects(labeled_clips_for_objects(["a", "b"]), heldout_count=2)
```

Run:

```bash
python -m unittest tests.python.test_interaction_features \
  tests.python.test_interaction_splits -v
```

Expected: import failures for `features.py` and `splits.py`.

- [ ] **Step 2: Implement the fixed 71-D layout**

Define:

```python
FEATURE_GROUPS = (
    FeatureGroup("pose", 0, 33),
    FeatureGroup("trajectory", 33, 45),
    FeatureGroup("grasp", 45, 57),
    FeatureGroup("root_target", 57, 65),
    FeatureGroup("context", 65, 71),
)
POSE_BONES = (7, 13, 1, 16)  # left toe, right toe, hips, chest
FUTURE_OFFSETS = (8, 17, 25)
```

For each frame, append values in this exact order:

| Range | Values |
| --- | --- |
| `0:15` | Root-heading-relative positions of left toe, right toe, hips, chest, active wrist. |
| `15:30` | Root-heading-relative velocities of those same five bones. |
| `30:33` | Root-heading-relative root linear X/Z velocity, then root yaw velocity. |
| `33:39` | Future root-heading-relative X/Z positions at +8, +17, +25 frames (approximately one-third, two-thirds, and one second), clamped to this clip's final frame. |
| `39:45` | Future facing X/Z at the same horizons in the current root heading frame. |
| `45:48` | Active-hand position relative to the demonstrated world grasp frame. |
| `48:51` | Active-hand orientation error as scaled-angle-axis relative to the world grasp frame. |
| `51:54` | Active-hand linear velocity in the grasp frame. |
| `54:57` | Active-hand angular velocity in the grasp frame. |
| `57:60` | Root position relative to the world grasp frame. |
| `60:62` | Root facing X/Z in the grasp frame. |
| `62:65` | Root linear velocity in the grasp frame. |
| `65` | World grasp Y minus table-top Y. |
| `66:68` | Object-local demonstrated approach X/Z. |
| `68:71` | Object-local dimensions X/Y/Z. |

The world grasp is `object_world * grasp_object`. Root heading is canonical bone 0. World pose and velocity come from `forward_local_hierarchy` / `quat.fk_vel`. Angular velocity for hands is taken from canonical world FK velocity, not recomputed across clips. Validate the final write cursor equals 71 for every row.

Use this exact row builder; `normalize_feature_groups` is applied once after rows
from all database clips are concatenated:

```python
def _raw_clip_features(labeled: LabeledInteractionClip) -> np.ndarray:
    clip = labeled.motion
    world_rot, world_pos, world_vel, world_ang = holden_quat.fk_vel(
        clip.rotations, clip.positions, clip.velocities,
        clip.angular_velocities, G1_SKELETON.parents)
    object_vel = clip.object_velocities
    object_ang = clip.object_angular_velocities
    hand_bone = 23 if labeled.active_hand == InteractionHand.LEFT else 30
    pose_bones = POSE_BONES + (hand_bone,)
    rows = np.empty((len(clip.positions), 71), np.float32)

    for frame in range(len(clip.positions)):
        root_position = world_pos[frame, 0]
        root_rotation = world_rot[frame, 0]
        inverse_root = holden_quat.inv(root_rotation)
        grasp_offset_world = holden_quat.mul_vec(
            clip.object_rotations[frame], labeled.grasp_position_object)
        grasp_position = clip.object_positions[frame] + grasp_offset_world
        grasp_rotation = holden_quat.mul(
            clip.object_rotations[frame], labeled.grasp_rotation_object)
        inverse_grasp = holden_quat.inv(grasp_rotation)
        grasp_velocity = object_vel[frame] + np.cross(
            object_ang[frame], grasp_offset_world)

        values: list[float] = []
        for bone in pose_bones:
            values.extend(holden_quat.mul_vec(
                inverse_root, world_pos[frame, bone] - root_position))
        for bone in pose_bones:
            values.extend(holden_quat.mul_vec(
                inverse_root, world_vel[frame, bone]))
        root_velocity = holden_quat.mul_vec(
            inverse_root, world_vel[frame, 0])
        values.extend((root_velocity[0], root_velocity[2], world_ang[frame, 0, 1]))

        for offset in FUTURE_OFFSETS:
            future = min(frame + offset, len(clip.positions) - 1)
            delta = holden_quat.mul_vec(
                inverse_root, world_pos[future, 0] - root_position)
            values.extend((delta[0], delta[2]))
        for offset in FUTURE_OFFSETS:
            future = min(frame + offset, len(clip.positions) - 1)
            facing_world = holden_quat.mul_vec(
                world_rot[future, 0], np.array([0.0, 0.0, 1.0]))
            facing_local = holden_quat.mul_vec(inverse_root, facing_world)
            values.extend((facing_local[0], facing_local[2]))

        hand_position = world_pos[frame, hand_bone]
        hand_rotation = world_rot[frame, hand_bone]
        values.extend(holden_quat.mul_vec(
            inverse_grasp, hand_position - grasp_position))
        values.extend(holden_quat.to_scaled_angle_axis(
            holden_quat.abs(
                holden_quat.mul(inverse_grasp, hand_rotation))))
        values.extend(holden_quat.mul_vec(
            inverse_grasp, world_vel[frame, hand_bone] - grasp_velocity))
        values.extend(holden_quat.mul_vec(
            inverse_grasp, world_ang[frame, hand_bone] - object_ang[frame]))

        values.extend(holden_quat.mul_vec(
            inverse_grasp, root_position - grasp_position))
        root_facing_world = holden_quat.mul_vec(
            root_rotation, np.array([0.0, 0.0, 1.0]))
        root_facing_grasp = holden_quat.mul_vec(
            inverse_grasp, root_facing_world)
        values.extend((root_facing_grasp[0], root_facing_grasp[2]))
        values.extend(holden_quat.mul_vec(
            inverse_grasp, world_vel[frame, 0] - grasp_velocity))

        table_top = clip.table_position[1] + 0.5 * clip.table_size[1]
        values.append(grasp_position[1] - table_top)
        values.extend((
            labeled.approach_direction_object[0],
            labeled.approach_direction_object[2],
        ))
        values.extend(clip.object_dimensions)
        if len(values) != 71:
            raise InteractionValidationError(
                "feature_dimension", f"built {len(values)} values")
        rows[frame] = values
    return rows

def build_features(
    clips: Sequence[LabeledInteractionClip],
    skeleton: SkeletonSpec,
) -> FeatureSet:
    if skeleton.signature() != G1_SKELETON.signature():
        raise InteractionValidationError(
            "skeleton_mismatch", skeleton.signature())
    if not clips:
        raise InteractionValidationError("empty_database", "no database clips")
    raw = np.concatenate([_raw_clip_features(clip) for clip in clips], axis=0)
    return normalize_feature_groups(raw, FEATURE_GROUPS)

def build_database_features(
    clips: Sequence[LabeledInteractionClip],
    split: EvaluationSplit,
    skeleton: SkeletonSpec,
) -> FeatureSet:
    database, _ = partition_clips(clips, split)
    return build_features(database, skeleton)
```

`build_features` remains a low-level stable-order primitive for
already-partitioned unit tests and artifact internals.
`build_database_features` retains its Task 5 filtering API and direct leakage
tests. Task 6 introduces `prepare_artifacts` as the corpus-builder/validation-gate
boundary so artifact rows and feature rows share one ordered database tuple.

- [ ] **Step 3: Implement Holden-style group normalization**

For each group independently:

```python
offsets[start:stop] = np.mean(raw[:, start:stop], axis=0)
component_std = np.std(raw[:, start:stop], axis=0)
group_scale = max(float(np.mean(component_std)), 1e-5)
scales[start:stop] = group_scale
values[:, start:stop] = (raw[:, start:stop] - offsets[start:stop]) / group_scale
```

This stores unit group weights. Dynamic runtime weights belong to the later selector and must not be baked into artifacts. Reject a group containing non-finite values or a final non-finite normalized matrix.

Before slicing, require unique group names and an exact, ordered partition of
every input column. Reject an empty group list, zero-width or reversed ranges,
negative or out-of-range endpoints, overlaps, leading/interior/trailing gaps,
and any other non-contiguous layout with the offending group/range in the
error.

- [ ] **Step 4: Implement deterministic object-held-out splitting**

`split_objects` sorts unique object IDs, checks `0 < heldout_count < unique_count`, uses `np.random.default_rng(seed).permutation`, sorts the chosen held-out IDs for serialization, and defines database IDs as the sorted complement. Add `validate_split(clips, split)` that preserves and validates the exact serialized tuples before set comparison: both partitions must be nonempty and sorted, contain no duplicate IDs, have no overlap, and name every source object exactly once with no missing or unknown ID. Every error names the category and offending IDs. Add `partition_clips(clips, split)` as the only production partition helper; it validates the full split and returns database and held-out tuples in stable input order.

```python
def split_objects(
    clips: Sequence[LabeledInteractionClip],
    heldout_count: int = 20,
    seed: int = 20260714,
) -> EvaluationSplit:
    objects = sorted({clip.motion.object_id for clip in clips})
    if heldout_count <= 0:
        raise ValueError(
            f"heldout_count must be positive, got {heldout_count}")
    if heldout_count >= len(objects):
        raise ValueError(
            f"heldout_count {heldout_count} must leave at least one "
            f"database object; got {len(objects)} unique objects")
    permutation = np.random.default_rng(seed).permutation(len(objects))
    heldout = tuple(sorted(objects[index] for index in permutation[:heldout_count]))
    database = tuple(sorted(set(objects) - set(heldout)))
    split = EvaluationSplit(seed, database, heldout)
    validate_split(clips, split)
    return split

def validate_split(
    clips: Sequence[LabeledInteractionClip], split: EvaluationSplit,
) -> None:
    for category, object_ids in (
        ("database", split.database_objects),
        ("heldout", split.heldout_objects),
    ):
        if not object_ids:
            raise ValueError(f"{category} partition is empty")
        duplicates = sorted({item for item in object_ids
                             if object_ids.count(item) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate {category} object IDs: {duplicates!r}")
        if tuple(object_ids) != tuple(sorted(object_ids)):
            raise ValueError(
                f"{category} object IDs must be sorted; "
                f"got {tuple(object_ids)!r}")
    database = set(split.database_objects)
    heldout = set(split.heldout_objects)
    overlap = sorted(database & heldout)
    if overlap:
        raise ValueError(
            "overlapping object IDs in database and heldout partitions: "
            f"{overlap!r}")
    expected = {clip.motion.object_id for clip in clips}
    actual = database | heldout
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing source object IDs from split: {missing!r}")
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(
            f"unknown split object IDs not present in clips: {unknown!r}")

def partition_clips(
    clips: Sequence[LabeledInteractionClip], split: EvaluationSplit,
) -> tuple[tuple[LabeledInteractionClip, ...],
           tuple[LabeledInteractionClip, ...]]:
    validate_split(clips, split)
    database_ids = set(split.database_objects)
    database = tuple(
        clip for clip in clips if clip.motion.object_id in database_ids)
    heldout = tuple(
        clip for clip in clips if clip.motion.object_id not in database_ids)
    return database, heldout
```

- [ ] **Step 5: Run feature and split tests**

```bash
python -m unittest tests.python.test_interaction_features \
  tests.python.test_interaction_splits -v
```

Expected: all tests report `ok`; translation/yaw and quaternion-sign invariance
are within tolerance; malformed splits/groups fail actionably; the production
boundary has exact database-row coverage; and no held-out object affects raw
rows, offsets, scales, or normalized values.

- [ ] **Step 6: Commit features and split**

```bash
git add resources/g1_interaction_builder/features.py \
  resources/g1_interaction_builder/splits.py \
  tests/python/test_interaction_features.py tests/python/test_interaction_splits.py
git commit -m "feat: build target-relative pickup features and heldout split"
```

### Task 6: Serialize and Atomically Publish Versioned Artifacts

**Files:**
- Create: `resources/g1_interaction_builder/artifacts.py`
- Create: `tests/python/test_interaction_artifacts.py`

**Interfaces:**
- Consumes: labeled database clips, `FeatureSet`, `EvaluationSplit`, `SkeletonSpec`, build configuration, per-clip validation records.
- Produces: `assemble_database(clips, skeleton) -> InteractionArtifact`,
  `prepare_artifacts(all_clips, split, skeleton) -> (ordered_database_clips,
  InteractionArtifact, FeatureSet)`, `write_artifact_set(output, artifact,
  features, split, manifest, report) -> None`, and `read_artifact_set(output) ->
  tuple[InteractionArtifact, FeatureSet, dict, dict, dict]`.

- [ ] **Step 1: Write failing round-trip, corruption, and atomicity tests**

```python
class InteractionArtifactTests(unittest.TestCase):
    def test_round_trip_preserves_every_array_and_header(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(output, artifact, features, split, manifest, report)
            loaded = read_artifact_set(output)
            assert_artifact_equal(self, artifact, loaded[0])
            np.testing.assert_array_equal(features.values, loaded[1].values)
            self.assertEqual(loaded[2]["schema_version"], 1)
            self.assertEqual(loaded[2]["skeleton_signature"], G1_SKELETON.signature())

    def test_rejects_truncation_trailing_bytes_and_bad_magic(self):
        artifact, features, split, manifest, report = artifact_fixture()
        for mutation, message in binary_mutations():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(output, artifact, features, split, manifest, report)
                mutation(output / "interaction_database.bin")
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_failed_publish_preserves_previous_output(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(output, artifact, features, split, manifest, report)
            before = (output / "manifest.json").read_bytes()
            features.values[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_artifact_set(output, artifact, features, split, manifest, report)
            self.assertEqual(before, (output / "manifest.json").read_bytes())
```

Define the referenced fixture and comparison helpers immediately above the test
class:

```python
def artifact_fixture():
    clips = labeled_clips_for_objects(
        ["database_fixture_object", "heldout_fixture_object"])
    split = EvaluationSplit(
        seed=20260714,
        database_objects=("database_fixture_object",),
        heldout_objects=("heldout_fixture_object",),
    )
    ordered, artifact, features = prepare_artifacts(
        clips, split, G1_SKELETON)
    labeled = ordered[0]
    manifest = {
        "schema_version": 1,
        "source_clips": 2,
        "included_clips": 2,
        "rejected_clips": 0,
        "target_fps": 25.0,
        "skeleton_signature": G1_SKELETON.signature(),
        "clips": [{
            "sequence_id": labeled.motion.sequence_id,
            "object_id": labeled.motion.object_id,
            "range_start": 0,
            "range_stop": len(labeled.motion.positions),
        }],
    }
    report = {
        "schema_version": 1,
        "source_clips": 2,
        "included_clips": 2,
        "rejected_clips": 0,
        "included_frames": len(labeled.motion.positions),
        "rejections_by_code": {},
        "rejections": [],
        "numeric_bounds": {
            "fk_max_error_m": 0.0,
            "fk_rotation_max_error_degrees": 0.0,
            "duration_max_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        },
    }
    return artifact, features, split, manifest, report

def assert_artifact_equal(test: unittest.TestCase, expected, actual):
    for field in dataclasses.fields(InteractionArtifact):
        left = getattr(expected, field.name)
        right = getattr(actual, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            test.assertEqual(left, right)

def mutate_bad_magic(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[:8] = b"BADMAGIC"
    path.write_bytes(data)

def mutate_truncated(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:-1])

def mutate_trailing(path: Path) -> None:
    with path.open("ab") as stream:
        stream.write(b"x")

def binary_mutations():
    return (
        (mutate_bad_magic, "database magic"),
        (mutate_truncated, "truncated source_frames"),
        (mutate_trailing, "trailing bytes"),
    )
```

Run: `python -m unittest tests.python.test_interaction_artifacts -v`

Expected: import failure for `artifacts.py`.

- [ ] **Step 2: Implement artifact assembly and validation**

Concatenate clips in lexicographic `sequence_id` order. `range_starts` is the cumulative frame count before each clip and `range_stops` after it. All frame arrays concatenate on axis 0; table, dimensions, grasp, approach direction, and active hand stack once per clip. Preserve exact source frames without adding global offsets.

```python
def assemble_database(
    clips: Sequence[LabeledInteractionClip],
    skeleton: SkeletonSpec,
) -> InteractionArtifact:
    ordered = sorted(clips, key=lambda clip: clip.motion.sequence_id)
    if not ordered:
        raise ValueError("cannot assemble an empty interaction database")
    lengths = np.array([len(clip.motion.positions) for clip in ordered], np.int32)
    stops = np.cumsum(lengths, dtype=np.int32)
    starts = np.r_[np.int32(0), stops[:-1]].astype(np.int32)

    def frames(name: str) -> np.ndarray:
        return np.concatenate([getattr(clip.motion, name) for clip in ordered], axis=0)

    artifact = InteractionArtifact(
        fps=25,
        parents=skeleton.parents.astype(np.int32),
        range_starts=starts,
        range_stops=stops,
        positions=frames("positions"),
        velocities=frames("velocities"),
        rotations=frames("rotations"),
        angular_velocities=frames("angular_velocities"),
        foot_contacts=frames("foot_contacts"),
        hand_contacts=frames("hand_contacts"),
        hand_dof=frames("hand_dof"),
        hand_dof_velocities=frames("hand_dof_velocities"),
        phases=np.concatenate([clip.phases for clip in ordered]),
        active_hands=np.array([int(clip.active_hand) for clip in ordered], np.uint8),
        time_to_contact=np.concatenate(
            [clip.time_to_contact for clip in ordered]).astype(np.float32),
        object_positions=frames("object_positions"),
        object_rotations=frames("object_rotations"),
        object_velocities=frames("object_velocities"),
        object_angular_velocities=frames("object_angular_velocities"),
        table_positions=np.stack([clip.motion.table_position for clip in ordered]),
        table_rotations=np.stack([clip.motion.table_rotation for clip in ordered]),
        table_sizes=np.stack([clip.motion.table_size for clip in ordered]),
        object_dimensions=np.stack(
            [clip.motion.object_dimensions for clip in ordered]),
        grasp_positions_object=np.stack(
            [clip.grasp_position_object for clip in ordered]),
        grasp_rotations_object=np.stack(
            [clip.grasp_rotation_object for clip in ordered]),
        approach_directions_object=np.stack(
            [clip.approach_direction_object for clip in ordered]),
        source_frames=frames("source_frames"),
    )
    artifact.validate()
    return artifact
```

`InteractionArtifact.validate()` enforces the Frozen Binary Layout shapes; exact
31-bone parent array; range coverage `[0, frame_count)` with non-overlap and no
empty range; phase monotonicity within each range; contact/lift/hold presence in
every range; nonnegative nonincreasing time-to-contact that is zero from CONTACT
onward within `1e-4`; finite unit horizontal approach directions within `1e-4`;
active-hand contact throughout CONTACT and LIFT and for the first five HOLD
samples; normalized quaternions; positive dimensions; no non-finite numeric
value; and feature/database frame equality. Contact after the first five HOLD
samples is not required.

`prepare_artifacts` validates and partitions the complete labeled clip sequence,
rejects duplicate database `sequence_id` values, sorts the database clips exactly
once by `sequence_id`, and passes that same ordered tuple to an order-preserving
artifact assembler and low-level `build_features`. `assemble_database` retains
its public sorting behavior; Task 5's stable-order feature APIs remain unchanged.

- [ ] **Step 3: Implement exact little-endian writers and readers**

Use explicit pack formats and dtype conversion:

```python
DB_MAGIC = b"G1INTDB1"
FEATURE_MAGIC = b"G1INTFT1"
VERSION = 1
ENDIAN_MARKER = 0x01020304

def _write_array(stream: BinaryIO, value: np.ndarray, dtype: str) -> None:
    array = np.asarray(value, dtype=np.dtype(dtype).newbyteorder("<"), order="C")
    if not np.isfinite(array).all() and array.dtype.kind == "f":
        raise ValueError("cannot write non-finite array")
    stream.write(array.tobytes(order="C"))

def _read_exact(stream: BinaryIO, count: int, label: str) -> bytes:
    position = stream.tell()
    end = stream.seek(0, os.SEEK_END)
    stream.seek(position, os.SEEK_SET)
    if count > end - position:
        raise ValueError(f"truncated {label}")
    value = stream.read(count)
    if len(value) != count:
        raise ValueError(f"truncated {label}")
    return value
```

Write/read arrays in the exact Frozen Binary Layout order. After the last declared array, require `stream.read(1) == b""`; otherwise raise `trailing bytes`. JSON uses UTF-8, `sort_keys=True`, `indent=2`, and a terminating newline.

The database writer is a direct transcription of that layout:

```python
def _write_database(path: Path, value: InteractionArtifact) -> None:
    frame_count, bone_count = value.positions.shape[:2]
    clip_count = len(value.range_starts)
    with path.open("wb") as stream:
        stream.write(struct.pack(
            "<8s8I", DB_MAGIC, VERSION, ENDIAN_MARKER, 25, 1,
            frame_count, bone_count, clip_count, 14))
        for array, dtype in (
            (value.parents, "i4"),
            (value.range_starts, "i4"),
            (value.range_stops, "i4"),
            (value.positions, "f4"),
            (value.velocities, "f4"),
            (value.rotations, "f4"),
            (value.angular_velocities, "f4"),
            (value.foot_contacts, "u1"),
            (value.hand_contacts, "u1"),
            (value.hand_dof, "f4"),
            (value.hand_dof_velocities, "f4"),
            (value.phases, "u1"),
            (value.active_hands, "u1"),
            (value.time_to_contact, "f4"),
            (value.object_positions, "f4"),
            (value.object_rotations, "f4"),
            (value.object_velocities, "f4"),
            (value.object_angular_velocities, "f4"),
            (value.table_positions, "f4"),
            (value.table_rotations, "f4"),
            (value.table_sizes, "f4"),
            (value.object_dimensions, "f4"),
            (value.grasp_positions_object, "f4"),
            (value.grasp_rotations_object, "f4"),
            (value.approach_directions_object, "f4"),
            (value.source_frames, "i4"),
        ):
            _write_array(stream, array, dtype)

def _write_features(path: Path, value: FeatureSet) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack(
            "<8s5I", FEATURE_MAGIC, VERSION, ENDIAN_MARKER,
            len(value.values), value.values.shape[1], len(value.groups)))
        _write_array(stream, [group.start for group in value.groups], "u4")
        _write_array(stream, [group.stop for group in value.groups], "u4")
        _write_array(stream, value.offsets, "f4")
        _write_array(stream, value.scales, "f4")
        _write_array(stream, value.values, "f4")
```

- [ ] **Step 4: Implement atomic directory publication**

Write to `output.parent / f".{output.name}.tmp-{os.getpid()}"`, remove only that
process-specific temp directory if it exists, reread and validate the entire
temporary artifact set, rename an existing output to `.{name}.previous-{pid}`,
rename temp to output, then remove the previous directory. Before validating new
inputs, scan stale `.{name}.previous-*` paths: restore a valid previous pack when
the public output is absent, preserve a valid public pack when both exist, or
replace an invalid public pack with a validated previous pack. Never delete an
unvalidated sole backup. On any synchronous exception, restore the previous
directory if the final rename did not complete and remove the process-specific
temp. Never remove or modify source data.

```python
def write_artifact_set(
    output: Path,
    artifact: InteractionArtifact,
    features: FeatureSet,
    split: EvaluationSplit,
    manifest: dict,
    report: dict,
) -> None:
    output = Path(output)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    previous = output.parent / f".{output.name}.previous-{os.getpid()}"
    _recover_interrupted_publish(output, temporary)
    artifact.validate()
    _validate_features(features, len(artifact.positions))
    manifest_text, manifest_value = _canonical_json(manifest, "manifest")
    split_text, split_value = _canonical_json(
        dataclasses.asdict(split), "evaluation split")
    report_text, report_value = _canonical_json(
        report, "validation report")
    _validate_manifest(manifest_value, artifact)
    _validate_split(split_value, manifest_value)
    _validate_report(report_value, len(artifact.positions), manifest_value)
    _remove_private_path(temporary)
    temporary.mkdir(parents=True)
    moved_previous = False
    published = False
    try:
        _write_database(temporary / "interaction_database.bin", artifact)
        _write_features(temporary / "interaction_features.bin", features)
        _write_text(temporary / "manifest.json", manifest_text)
        _write_text(temporary / "evaluation_split.json", split_text)
        _write_text(temporary / "validation_report.json", report_text)
        read_artifact_set(temporary)
        if output.exists():
            os.replace(output, previous)
            moved_previous = True
        os.replace(temporary, output)
        published = True
        if moved_previous:
            shutil.rmtree(previous)
    except BaseException:
        if moved_previous and previous.exists():
            if published and output.exists():
                os.replace(output, temporary)
                published = False
            if not output.exists():
                os.replace(previous, output)
                moved_previous = False
        _remove_private_path(temporary)
        if not moved_previous:
            _remove_private_path(previous)
        raise
```

- [ ] **Step 5: Run artifact tests**

Run: `python -m unittest tests.python.test_interaction_artifacts -v`

Expected: all round-trip/corruption/atomicity tests report `ok`.

- [ ] **Step 6: Commit artifact publication**

```bash
git add resources/g1_interaction_builder/artifacts.py \
  tests/python/test_interaction_artifacts.py
git commit -m "feat: publish versioned interaction artifacts atomically"
```

### Task 7: Add the Corpus Builder, Validator, and Optional Fetch Utility

**Files:**
- Create: `resources/g1_interaction_builder/build.py`
- Create: `resources/build_g1_interaction_database.py`
- Create: `resources/validate_g1_interaction_database.py`
- Create: `resources/fetch_grail_pickup_table.py`
- Create: `tests/python/test_interaction_build_cli.py`

**Interfaces:**
- Consumes: Tasks 2-6 and CLI source paths.
- Produces: reproducible complete packs, `validation_report.json`, a standalone `VALID` summary line, and constrained source download.

- [ ] **Step 1: Write failing CLI and rejection-report tests**

Test `parse_args` and a two-object synthetic build. Use a fake kinematics converter and fake USD dimension reader so unit tests remain fast. Assert the following CLI defaults exactly:

```python
self.assertEqual(args.target_fps, 25.0)
self.assertEqual(args.heldout_count, 20)
self.assertEqual(args.seed, 20260714)
self.assertIsNone(args.limit)
self.assertFalse(args.allow_rejections)
self.assertEqual(args.output, Path("resources/g1_interaction"))
```

Add tests that:

- a source shape error creates rejection code `frame_count_mismatch`;
- normal mode exits nonzero and publishes nothing when any clip is rejected;
- `--allow-rejections` publishes only when at least one database and one held-out object remain;
- `--limit 2` selects the first two lexicographic source sequences and records `diagnostic_limit=2` in the manifest;
- the standalone validator prints exactly one line beginning `VALID schema=1 fps=25 bones=31 features=71`.

Run: `python -m unittest tests.python.test_interaction_build_cli -v`

Expected: imports fail for the three new scripts and `build.py`.

- [ ] **Step 2: Implement per-clip orchestration and rejection records**

Define one rejection schema:

```python
@dataclass(frozen=True)
class Rejection:
    sequence_id: str
    object_id: str
    stage: str
    code: str
    message: str
```

For each sorted source, run `load -> convert -> derive labels`. Catch only `SourceValidationError`, `ConversionValidationError`, and `InteractionValidationError`; map their explicit `.code` and message into `Rejection`. Programming errors propagate and abort. Build dimensions once per object identity. Do not catch `KeyboardInterrupt`, `MemoryError`, or broad `Exception`.

Implement the per-source loop and partition order as follows:

```python
EXPECTED_CLIP_ERRORS = (
    SourceValidationError,
    ConversionValidationError,
    InteractionValidationError,
)

def build_labeled_clips(
    sources: Sequence[SourcePaths],
    dimensions: Mapping[str, np.ndarray],
    kinematics: G1Kinematics,
) -> tuple[list[LabeledInteractionClip], list[Rejection], list[dict]]:
    included: list[LabeledInteractionClip] = []
    rejected: list[Rejection] = []
    numeric_reports: list[dict] = []
    for source in sources:
        try:
            raw = load_raw_interaction(source, dimensions[source.object_id])
            canonical, skeleton, numeric = convert_interaction(raw, kinematics, 25.0)
            if skeleton.signature() != G1_SKELETON.signature():
                raise ConversionValidationError(
                    "skeleton_mismatch", skeleton.signature())
            included.append(derive_interaction_labels(canonical))
            numeric_reports.append(numeric)
        except EXPECTED_CLIP_ERRORS as error:
            stage = (
                "source" if isinstance(error, SourceValidationError)
                else "conversion" if isinstance(error, ConversionValidationError)
                else "interaction")
            rejected.append(Rejection(
                source.sequence_id, source.object_id, stage,
                error.code, str(error)))
    rejected.sort(key=lambda item: (item.sequence_id, item.stage, item.code))
    return included, rejected, numeric_reports

def prepare_database(
    clips: Sequence[LabeledInteractionClip],
    heldout_count: int,
    seed: int,
    skeleton: SkeletonSpec,
) -> tuple[
    tuple[LabeledInteractionClip, ...],
    InteractionArtifact,
    FeatureSet,
    EvaluationSplit,
]:
    split = split_objects(clips, heldout_count=heldout_count, seed=seed)
    database, artifact, features = prepare_artifacts(
        clips, split, skeleton)
    return database, artifact, features, split
```

The caller derives labels for every valid source first, creates the object split,
then passes the complete labeled sequence and validated split to
`prepare_artifacts`. That boundary filters and sorts the database clips once and
builds both the artifact and features from the same tuple. Direct calls to
`partition_clips`, `assemble_database`, low-level `build_features`, or
`build_database_features` are forbidden in this corpus path. Held-out clips
contribute identities and later evaluation inputs but never feature
normalization statistics or database frames.

The report contains exact keys:

```json
{
  "schema_version": 1,
  "source_clips": 0,
  "included_clips": 0,
  "rejected_clips": 0,
  "included_frames": 0,
  "rejections_by_code": {},
  "rejections": [],
  "numeric_bounds": {
    "fk_max_error_m": 0.0,
    "fk_rotation_max_error_degrees": 0.0,
    "duration_max_error_s": 0.0,
    "quaternion_norm_max_error": 0.0
  }
}
```

Values are populated from the actual build; rejections sort by `(sequence_id, stage, code)`.
The manifest repeats `source_clips`, `included_clips`, and `rejected_clips` with
the exact same values. In both files, `source_clips == included_clips +
rejected_clips`; `included_clips` counts every valid labeled clip, including
held-out clips, while manifest `clips` contains only database clips in unique
lexicographic `sequence_id` order. The database artifact/manifest clip count may
therefore be smaller than `included_clips` but never larger. The rejection array
length equals `rejected_clips`, every record has exactly the five fields above,
and `rejections_by_code` is the exact histogram of their codes.

- [ ] **Step 3: Implement the build and validation CLIs**

`build_g1_interaction_database.py` accepts:

```text
--source-root PATH             required
--g1-xml PATH                 required
--output PATH                 default resources/g1_interaction
--target-fps FLOAT            default 25.0; reject every other value for schema v1
--heldout-count INTEGER       default 20
--seed INTEGER                default 20260714
--limit INTEGER              optional positive diagnostic prefix
--allow-rejections            default false
```

Build the parser with exact `pathlib.Path` conversion and positive integer
checks:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build schema-v1 G1 tabletop interaction artifacts")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("resources/g1_interaction"))
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--heldout-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-rejections", action="store_true")
    args = parser.parse_args(argv)
    if args.target_fps != 25.0:
        parser.error("schema v1 requires --target-fps 25")
    if args.heldout_count < 1:
        parser.error("--heldout-count must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args
```

The manifest records schema/magics, exact skeleton names/parents/signature, source root, GRAIL dataset ID, source/included/rejected counts, target FPS, clip order and ranges, per-clip object ID and active hand, phase/contact thresholds, all feature names/groups, split seed/counts, dependency versions, Git commit, diagnostic limit, and `source_date_epoch` as an integer only when that environment variable is set (JSON `null` otherwise). It must not record wall-clock time or authentication tokens, so identical inputs and commit produce byte-identical metadata.

`validate_g1_interaction_database.py --input PATH` rereads all five files, reruns semantic validation, verifies the manifest clip ranges and object split, computes SHA-256 for both binaries, and prints:

```text
VALID schema=1 fps=25 bones=31 features=71 clips=<C> frames=<N> heldout_objects=<H> db_sha256=<HEX> feature_sha256=<HEX>
```

- [ ] **Step 4: Implement constrained optional download**

`fetch_grail_pickup_table.py` calls:

```python
snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",
    repo_type="dataset",
    local_dir=args.output,
    allow_patterns=[
        "data/pickup_table/robot/*.pkl",
        "data/pickup_table/objects/*.pkl",
        "data/pickup_table/meta/*.pkl",
        "data/pickup_table/object_usd/*.usd",
        "data/pickup_table/object_usd/textures/*",
    ],
    token=args.token,
)
```

The token defaults to `HF_TOKEN` and is never printed. `--dry-run` prints the dataset ID and allow-patterns without network access. Unit-test only dry-run and mocked `snapshot_download` behavior.

- [ ] **Step 5: Run CLI tests and a representative real build**

```bash
python -m unittest tests.python.test_interaction_build_cli -v
rm -rf /tmp/g1_interaction_3
python -m resources.build_g1_interaction_database \
  --source-root /home/ubuntu/datasets/GRAIL/data/pickup_table \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output /tmp/g1_interaction_3 --limit 3 --allow-rejections \
  --heldout-count 1
python -m resources.validate_g1_interaction_database \
  --input /tmp/g1_interaction_3
```

Expected: unit tests report `ok`; the real build publishes five files; validator prints a `VALID schema=1 fps=25 bones=31 features=71` line. If one of the first three real clips is legitimately rejected, increase only `--limit` until at least two object identities remain and retain every rejection in the report.

- [ ] **Step 6: Commit the build tools**

```bash
git add resources/g1_interaction_builder/build.py \
  resources/build_g1_interaction_database.py \
  resources/validate_g1_interaction_database.py \
  resources/fetch_grail_pickup_table.py \
  tests/python/test_interaction_build_cli.py
git commit -m "feat: build and validate GRAIL interaction corpus"
```

### Task 8: Load and Probe the Same Artifacts in Headless C++

**Files:**
- Create: `interaction_database.h`
- Create: `interaction_probe.cpp`
- Create: `tests/cpp/test_interaction_database.cpp`
- Modify: `tests/python/test_interaction_artifacts.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Frozen Binary Layout files.
- Produces: `interaction::Database load_database(const std::filesystem::path&)`, `interaction::Features load_features(const std::filesystem::path&)`, strict corruption errors, and deterministic probe JSON.

- [ ] **Step 1: Write failing C++ loader and cross-language probe tests**

Create `tests/cpp/test_interaction_database.cpp` to generate a minimal valid schema-v1 binary fixture through byte-wise helper writes, load it, and assert all header/count/first/last values. Create mutations for bad magic, wrong endian marker, wrong bone count, invalid range, non-unit quaternion, truncation, and one trailing byte; each must throw `interaction::FormatError` containing the field name.

Extend the Python artifact test to:

1. publish `artifact_fixture()` to a temporary directory;
2. run `./interaction_probe <dir> --json`;
3. parse stdout JSON;
4. assert frame/clip/bone/feature counts, first/last source frames, all phase counts, maximum quaternion norm error, and binary SHA-256 values equal Python-computed values.

Run:

```bash
make build/tests/test_interaction_database interaction_probe
python -m unittest tests.python.test_interaction_artifacts.CrossLanguageProbeTests -v
```

Expected: compilation fails because loader/probe files do not exist.

- [ ] **Step 2: Implement a strict dependency-free C++ loader**

`interaction_database.h` owns vectors for every declared array and uses these fixed-width helpers:

```cpp
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template<class T>
T read_scalar(std::istream& input, std::string_view label) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw FormatError("truncated " + std::string(label));
    return value;
}

template<class T>
std::vector<T> read_vector(
    std::istream& input, size_t count, std::string_view label) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
        throw FormatError("overflow " + std::string(label));
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw FormatError("truncated " + std::string(label));
    return values;
}
```

Require a little-endian host with `std::endian::native` when C++20 is available, otherwise validate the marker's in-memory first byte. Before multiplying dimensions, use checked multiplication. Validate all Frozen Binary Layout invariants, finite floats, unit quaternion error at most `1e-4`, exact G1 parents, phase enum range and monotonicity per clip, and EOF after the final byte.

- [ ] **Step 3: Implement deterministic probe output**

`interaction_probe <pack> --json` loads both binaries plus the three JSON files only to hash/confirm their presence; binary semantics come from `interaction_database.h`. Output one compact JSON object with sorted keys:

```json
{"bone_count":31,"clip_count":1,"database_sha256":"0000000000000000000000000000000000000000000000000000000000000000","feature_count":71,"feature_sha256":"0000000000000000000000000000000000000000000000000000000000000000","first_source_frame":0,"frame_count":75,"last_source_frame":74,"max_quaternion_norm_error":0.0,"phase_counts":[13,25,4,8,25]}
```

The all-zero digests above illustrate the fixed 64-character JSON field width;
the executable emits the files' computed lowercase digests.

Implement SHA-256 as a small private implementation in `interaction_probe.cpp` or invoke no external process/library. Floating output uses `std::setprecision(9)`.

- [ ] **Step 4: Add Make targets and run cross-language tests**

Add:

```make
SOURCE = controller.cpp
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_database

$(CPP_TEST_DIR)/test_interaction_database: tests/cpp/test_interaction_database.cpp interaction_database.h g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

interaction_probe: interaction_probe.cpp interaction_database.h g1_skeleton.h
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@
```

Replace the original `SOURCE = $(wildcard *.cpp)` assignment with the explicit
`SOURCE = controller.cpp` line above. This prevents `interaction_probe.cpp` from
being linked into the Raylib controller and preserves the controller's one-main
build.

Run:

```bash
make test-cpp interaction_probe
python -m unittest tests.python.test_interaction_artifacts -v
```

Expected: both C++ tests exit 0 and Python/C++ probe fields agree exactly.

- [ ] **Step 5: Commit cross-language replay**

```bash
git add interaction_database.h interaction_probe.cpp Makefile \
  tests/cpp/test_interaction_database.cpp \
  tests/python/test_interaction_artifacts.py
git commit -m "feat: replay interaction artifacts in headless C++"
```

### Task 9: Document and Pass Validation Gate 1

**Files:**
- Modify: `README.md`
- Modify: `Makefile`
- Modify: `resources/g1_interaction_builder/build.py`
- Modify: `tests/python/test_interaction_build_cli.py`

**Interfaces:**
- Consumes: complete Tasks 1-8 and all 2,991 local pickup-table sequences.
- Produces: reproducible `make gate1-interaction`, a complete validation report, a representative C++ replay digest, and operator documentation.

Gate 1 must exercise Task 7's `prepare_database` path, which delegates artifact
and feature construction to `prepare_artifacts(all_labeled_clips, split,
G1_SKELETON)`. It must not call `partition_clips`, `assemble_database`,
low-level `build_features`, or `build_database_features`, and it must not
recreate database filtering with a list comprehension.

- [ ] **Step 1: Add a failing Gate 1 smoke assertion**

Add `tests/python/test_interaction_gate1.py` with a `Gate1PackTests` class that accepts `G1_INTERACTION_DIR`. When set, it reads the pack and asserts:

```python
self.assertEqual(manifest["schema_version"], 1)
self.assertEqual(manifest["source_clips"], 2991)
self.assertEqual(manifest["diagnostic_limit"], None)
self.assertEqual(manifest["target_fps"], 25.0)
self.assertEqual(manifest["skeleton_signature"], G1_SKELETON.signature())
self.assertEqual(features.values.shape[1], 71)
self.assertEqual(features.values.shape[0], artifact.positions.shape[0])
self.assertEqual(len(split["heldout_objects"]), 20)
self.assertFalse(set(split["heldout_objects"]) & set(split["database_objects"]))
self.assertEqual(report["included_clips"] + report["rejected_clips"], 2991)
self.assertEqual(manifest["source_clips"], report["source_clips"])
self.assertEqual(manifest["included_clips"], report["included_clips"])
self.assertEqual(manifest["rejected_clips"], report["rejected_clips"])
self.assertGreater(report["included_clips"], 0)
```

When the environment variable is absent, skip only this full-pack class with the message `G1_INTERACTION_DIR not set`; all unit and representative tests still run.

Run: `G1_INTERACTION_DIR=/tmp/missing python -m unittest tests.python.test_interaction_gate1 -v`

Expected: failure because the pack does not exist.

- [ ] **Step 2: Add a reproducible Gate 1 Make target**

Add configurable paths and target:

```make
GRAIL_PICKUP_ROOT ?= /home/ubuntu/datasets/GRAIL/data/pickup_table
G1_XML ?= /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
G1_INTERACTION_DIR ?= resources/g1_interaction

.PHONY: gate1-interaction
gate1-interaction: test-interaction interaction_probe
	python -m resources.build_g1_interaction_database --source-root $(GRAIL_PICKUP_ROOT) --g1-xml $(G1_XML) --output $(G1_INTERACTION_DIR) --allow-rejections
	python -m resources.validate_g1_interaction_database --input $(G1_INTERACTION_DIR)
	G1_INTERACTION_DIR=$(G1_INTERACTION_DIR) python -m unittest tests.python.test_interaction_gate1 -v
	./interaction_probe $(G1_INTERACTION_DIR) --json
```

Before enabling the target, add a table-driven unit test that asserts every emitted
rejection code belongs to this closed schema-v1 set:

```python
KNOWN_REJECTION_CODES = {
    "missing_field", "record_count", "frame_count_mismatch", "fps_mismatch",
    "non_finite", "invalid_quaternion", "invalid_dimensions",
    "joint_limit_violation", "skeleton_mismatch", "fk_error",
    "fk_rotation_error", "duration_error",
    "ambiguous_active_hand", "no_stable_contact", "contact_lost_before_hold",
    "no_five_centimeter_lift", "no_stable_hold", "invalid_grasp",
    "invalid_approach",
}

def test_rejection_code_schema_is_closed(self):
    self.assertEqual(set(all_schema_v1_rejection_codes()), KNOWN_REJECTION_CODES)
```

`--allow-rejections` means only those reviewed semantic exclusions may be
reported. A missing source modality, unknown code, programming error, or artifact
invariant still aborts publication.

- [ ] **Step 3: Run a 25-clip audit before the full corpus**

```bash
rm -rf /tmp/g1_interaction_25
python -m resources.build_g1_interaction_database \
  --source-root /home/ubuntu/datasets/GRAIL/data/pickup_table \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output /tmp/g1_interaction_25 --limit 25 --allow-rejections \
  --heldout-count 3
python -m resources.validate_g1_interaction_database --input /tmp/g1_interaction_25
./interaction_probe /tmp/g1_interaction_25 --json
```

Expected: exit 0; every rejection has a nonempty exact code/message; both validators agree on counts and SHA-256; no NaN, range, split, or quaternion error.

- [ ] **Step 4: Run the complete corpus and classify observed exclusions**

```bash
rm -rf resources/g1_interaction
make gate1-interaction
```

Expected: a complete pack with every excluded clip assigned one of the closed,
unit-tested schema-v1 rejection codes. Gate 1 passes only when:

- all 2,991 source paths were examined;
- included plus rejected equals 2,991;
- every rejection is deterministic and represented in a regression test;
- at least 20 valid object identities remain held out and object-disjoint;
- Python validation and the C++ probe agree;
- feature/database frame counts agree;
- maximum FK error is at most `0.001 m`, rotation error at most `0.1 degree`, and duration error at most one 25 Hz frame;
- no generated artifact overwrote ordinary locomotion resources.

The full-corpus build must retain a regression proving that mutating any
held-out clip leaves serialized feature values, offsets, and scales unchanged;
this is enforced through `prepare_artifacts`, not caller-side filtering.

- [ ] **Step 5: Document exact setup and results**

Add a `G1 tabletop interaction data replay` README section containing:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-interaction.txt
python -m resources.fetch_grail_pickup_table --output /path/to/GRAIL
make gate1-interaction \
  GRAIL_PICKUP_ROOT=/path/to/GRAIL/data/pickup_table \
  G1_XML=/path/to/g1_29dof.xml \
  G1_INTERACTION_DIR=resources/g1_interaction
```

Document the five generated files, their ignored/reproducible status, exact final full-corpus included/rejected/object/frame counts copied from `validation_report.json`, how to inspect rejections, and this explicit scope statement: `Gate 1 replays data only; it does not yet make the character pick up an object.`

- [ ] **Step 6: Run final clean verification**

```bash
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml make test-interaction interaction_probe
python -m resources.validate_g1_interaction_database --input resources/g1_interaction
G1_INTERACTION_DIR=resources/g1_interaction \
  python -m unittest tests.python.test_interaction_gate1 -v
./interaction_probe resources/g1_interaction --json
git status --short
```

Expected: all tests pass; both validators return exit 0; `git status --short` lists only intended source/document changes and no generated interaction binary.

- [ ] **Step 7: Commit Gate 1**

```bash
git add README.md Makefile resources/g1_interaction_builder/build.py \
  tests/python/test_interaction_build_cli.py
git commit -m "docs: validate GRAIL tabletop data replay gate"
```

## Gate 1 Handoff Contract

After this plan passes, later runtime work may depend only on:

- strict schema-v1 artifacts and their manifest;
- exact G1 skeleton signature and 25 Hz motion-sampling contract;
- clip ranges, active hand, phase, time-to-contact, contacts, object/table context, demonstrated grasp, and 71-D features;
- deterministic object split and rejection report;
- a Python/C++ parity test that detects schema drift.

The next plan must introduce the game-facing types without changing these files:

```cpp
struct PickRequest {
    uint64_t target_object_id;
    Transform object_world;
    Transform grasp_object;
    InteractionHand allowed_hand;
    Vec3 approach_axis_object;
    float approach_cone_radians;
    float clearance_radius_m;
    Vec3 object_dimensions_m;
};

enum class InteractionStatus {
    Accepted, Succeeded, Rejected, Cancelled, Failed
};

struct InteractionResult {
    InteractionStatus status;
    InteractionReason reason;
    uint64_t target_object_id;
};
```

That boundary is what lets a player script or future VLM say “pick this,” “place here,” or “open this” while the motion system answers only whether and how the embodied primitive can be executed.
