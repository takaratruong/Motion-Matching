# G1 SONIC Root Stream and Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry Motion Matching's physical pelvis trajectory bit-exactly into GEAR and expose the canonical 10-frame root trajectory as a new 30-value G1 encoder observation without changing the protocol-v1 baseline.

**Architecture:** Motion Matching creates an immutable `RootTargetBuffer` beside the existing `CanonicalTargetBuffer` and publishes it with packed-pose protocol v5; v1 stays byte-for-byte unchanged. GEAR v5 decodes and atomically merges root body positions, then a pure C++ canonicalizer produces `[local delta x, local delta y, absolute pelvis z]` for 10 frames at step 5. The file path and stream path consume the same physical-pelvis values and are checked against one shared numeric fixture.

**Tech Stack:** Python 3.11, NumPy, ZeroMQ packed messages, C++20, GoogleTest, GEAR `MotionSequence`, ONNX Runtime configuration.

## Global Constraints

- Motion Matching implementation starts in a new worktree and branch from the plan commit, which must descend directly from approved-design commit `1900933` on `research/g1-low-latency-driver`; do not implement on the terrain-aware branch in place.
- GEAR implementation starts in a new clean worktree from commit `294110cedba01ad764f1e268d57ddf7c1bbf9523`.
- Preserve packed-pose protocol v1 exactly; protocol v4 is token-only, so the root-capable motion protocol is v5.
- The transmitted root is `TargetChunk.physical_pelvis_position`, never `virtual_root_position` and never measured global robot XY.
- Protocol v5 field order is `joint_pos`, `joint_vel`, `body_quat_w`, `body_pos`, `frame_index`, optional hand fields, `catch_up`.
- `body_pos` is little-endian `f32[N,3]`, finite, C-contiguous, and has the same row count as every other motion field.
- Canonical root position for frame `i` is `inverse(heading(q0)) * (p_i - [p0.x,p0.y,0])`; therefore frame zero is `[0,0,p0.z]`.
- The deployed encoder observation superset appends the new 30 values after the existing 1,762 values; old offsets and ordering do not move.
- The observation name is `motion_root_position_refheading_10frame_step5` and its logical training name is `motion_root_position_refheading_mf_nonflat`.
- Cross-language canonicalization tolerance is `1e-6`; packed stream and reference-file root values must be float32 bit-equal.

---

## File Map

### Motion Matching repository

- Create `sonic/python/mm_sonic/root_target.py`: immutable owned root-capable target buffer.
- Create `sonic/python/mm_sonic/zmq_v5.py`: strict protocol-v5 header, codec, and publisher wrapper.
- Modify `sonic/python/mm_sonic/reference.py`: explicit physical-pelvis reference export mode while keeping the default zero-root behavior.
- Create `tests/python/test_sonic_root_target.py`: ownership and validation tests.
- Create `tests/python/test_sonic_zmq_v5.py`: exact header, payload, rejection, and v1 non-regression tests.
- Modify `tests/python/test_sonic_reference.py`: physical-pelvis file export and file/stream parity tests.

### GEAR repository

- Modify `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp`: v5 body-position ownership, validation, overlap copy, and incoming copy.
- Modify `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`: v5 field discovery, shape validation, decoding, and mode assignment.
- Create `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/root_trajectory.hpp`: pure canonicalization helper.
- Modify `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`: gatherer and observation registry entry.
- Create `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp`: merger and canonicalization tests.
- Create `gear_sonic_deploy/policy/release/observation_config_root_conditioned.yaml`: 1,792-value encoder superset with the new G1-only range appended.

### Cross-repository fixture

- Create `sonic/fixtures/root_trajectory_v1.json`: shared positions, quaternions, and expected canonical result.
- Create `sonic/python/mm_sonic/root_trajectory.py`: Python reference implementation and fixture validator.
- Create `tests/python/test_sonic_root_trajectory.py`: invariance and fixture tests.

---

### Task 1: Own the Physical Pelvis as an Explicit Target Type

**Files:**
- Create: `sonic/python/mm_sonic/root_target.py`
- Create: `tests/python/test_sonic_root_target.py`

**Interfaces:**
- Consumes: `CanonicalTargetBuffer` and `TargetChunk.physical_pelvis_position` from `mm_sonic.timeline`.
- Produces: `RootTargetBuffer(canonical: CanonicalTargetBuffer, body_position: np.ndarray)`, `RootTargetBuffer.from_target_chunk(chunk)`, and `count: int`.

- [ ] **Step 1: Write failing ownership and validation tests**

```python
def test_from_target_chunk_owns_physical_pelvis_not_virtual_root():
    chunk = make_target_chunk()
    root = RootTargetBuffer.from_target_chunk(chunk)
    np.testing.assert_array_equal(root.body_position.view(np.uint32), chunk.physical_pelvis_position.view(np.uint32))
    assert not np.array_equal(root.body_position, chunk.virtual_root_position)
    assert root.body_position.dtype == np.dtype("<f4")
    assert root.body_position.flags.c_contiguous
    assert not root.body_position.flags.writeable

def test_rejects_mismatched_or_nonfinite_body_position(self):
    canonical = make_canonical(3)
    with self.assertRaisesRegex(ContractError, r"body_position must have shape \[3,3\]"):
        RootTargetBuffer(canonical, np.zeros((2, 3), np.float32))
    bad = np.zeros((3, 3), np.float32)
    bad[1, 2] = np.nan
    with self.assertRaisesRegex(ContractError, "finite"):
        RootTargetBuffer(canonical, bad)
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_root_target.py' -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'mm_sonic.root_target'`.

- [ ] **Step 3: Implement the immutable type**

```python
@dataclass(frozen=True)
class RootTargetBuffer:
    canonical: CanonicalTargetBuffer
    body_position: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.canonical, CanonicalTargetBuffer):
            raise ContractError("root target requires a CanonicalTargetBuffer")
        value = np.asarray(self.body_position)
        shape = (self.canonical.count, 3)
        if value.shape != shape:
            raise ContractError(f"body_position must have shape [{shape[0]},3]")
        owned = np.ascontiguousarray(value, dtype="<f4").copy(order="C")
        if not np.all(np.isfinite(owned)):
            raise ContractError("body_position must contain finite values")
        owned.flags.writeable = False
        object.__setattr__(self, "body_position", owned)

    @classmethod
    def from_target_chunk(cls, chunk: TargetChunk) -> "RootTargetBuffer":
        if not isinstance(chunk, TargetChunk):
            raise ContractError("root target source must be a TargetChunk")
        return cls(chunk.buffer, chunk.physical_pelvis_position)

    @property
    def count(self) -> int:
        return self.canonical.count
```

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_*.py' -v`

Expected: all tests pass and the existing timeline tests remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/root_target.py tests/python/test_sonic_root_target.py
git commit -m "feat: own physical pelvis target trajectory"
```

### Task 2: Add Packed-Pose Protocol v5 Without Touching v1

**Files:**
- Create: `sonic/python/mm_sonic/zmq_v5.py`
- Create: `tests/python/test_sonic_zmq_v5.py`
- Test: `tests/python/test_sonic_zmq_v1.py`

**Interfaces:**
- Consumes: `RootTargetBuffer` from Task 1 and `Dex3HandTargets` from `mm_sonic.hands`.
- Produces: `pose_header_v5(count, include_hands=False)`, `encode_pose_v5(root, hand_targets=None)`, `decode_pose_v5(message)`, `DecodedPoseV5.root`, and `RootPosePublisher` with the same `prepare/send/close` surface as `PosePublisher`.

- [ ] **Step 1: Write exact wire-contract tests**

```python
def test_v5_appends_body_position_without_changing_v1():
    root = make_root_buffer(3)
    message = encode_pose_v5(root)
    decoded = decode_pose_v5(message)
    assert decoded.header["v"] == 5
    assert [field["name"] for field in decoded.header["fields"]] == [
        "joint_pos", "joint_vel", "body_quat_w", "body_pos", "frame_index", "catch_up"
    ]
    np.testing.assert_array_equal(decoded.body_position.view(np.uint32), root.body_position.view(np.uint32))
    assert encode_pose_v1(root.canonical) == manual_v1_message(root.canonical)

def test_v5_rejects_wrong_version_shape_and_payload_length(self):
    message = bytearray(encode_pose_v5(make_root_buffer(3)))
    for mutated, pattern in malformed_v5_messages(message):
        with self.assertRaisesRegex(ContractError, pattern):
            decode_pose_v5(mutated)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_zmq_v5.py' -v`

Expected: collection fails because `mm_sonic.zmq_v5` does not exist.

- [ ] **Step 3: Implement the v5 header and codec**

```python
def pose_header_v5(count: int, include_hands: bool = False) -> dict[str, object]:
    fields = [
        {"name": "joint_pos", "dtype": "f32", "shape": [count, 29]},
        {"name": "joint_vel", "dtype": "f32", "shape": [count, 29]},
        {"name": "body_quat_w", "dtype": "f32", "shape": [count, 4]},
        {"name": "body_pos", "dtype": "f32", "shape": [count, 3]},
        {"name": "frame_index", "dtype": "i64", "shape": [count]},
    ]
    if include_hands:
        fields += [
            {"name": "left_hand_joints", "dtype": "f32", "shape": [7]},
            {"name": "right_hand_joints", "dtype": "f32", "shape": [7]},
        ]
    fields.append({"name": "catch_up", "dtype": "u8", "shape": [1]})
    return {"v": 5, "endian": "le", "count": count, "fields": fields}

def encode_pose_v5(root: RootTargetBuffer, *, hand_targets: Dex3HandTargets | None = None) -> bytes:
    if not isinstance(root, RootTargetBuffer):
        raise ContractError("ZMQ v5 encoding requires a RootTargetBuffer")
    include_hands = hand_targets is not None
    if include_hands:
        hand_targets = validate_hand_targets(hand_targets)
    encoded = json.dumps(pose_header_v5(root.count, include_hands), separators=(",", ":")).encode("ascii")
    if len(encoded) > HEADER_BYTES:
        raise ContractError("ZMQ v5 header exceeds 1280 bytes")
    parts = [
        root.canonical.joint_position.tobytes(order="C"),
        root.canonical.joint_velocity.tobytes(order="C"),
        root.canonical.body_quat_w.tobytes(order="C"),
        root.body_position.tobytes(order="C"),
        root.canonical.frame_index.astype("<i8", copy=False).tobytes(order="C"),
    ]
    if include_hands:
        parts += [hand_targets.left_f32.tobytes(), hand_targets.right_f32.tobytes()]
    return TOPIC + encoded + bytes(HEADER_BYTES - len(encoded)) + b"".join(parts) + b"\0"
```

Implement `decode_pose_v5` with the same fixed topic/header split as v1. Decode JSON with a duplicate-key rejecting `object_pairs_hook`, require `set(header) == {"v","endian","count","fields"}`, require `header == pose_header_v5(count, include_hands)`, compute every payload slice from the declared dtype and shape, and require the sum of field byte lengths to equal the payload length. Copy each NumPy view into an owned read-only array, build `CanonicalTargetBuffer` from joints/quaternion/index so its unit-quaternion and contiguous-index validation runs, then build `RootTargetBuffer` from that canonical buffer and `body_pos`. Require `catch_up` to be exactly one byte in `{0,1}` and require both or neither hand fields. `verify_pose_v5_parity` compares little-endian bytes for all canonical fields, `body_pos`, and optional hands.

Subclass the existing publisher so socket ownership and archive/send semantics remain one implementation. Override only `prepare`; inherited `send()` dispatches to the override:

```python
class RootPosePublisher(PosePublisher):
    def prepare(
        self,
        buffer: RootTargetBuffer,
        *,
        phase: str = "timeline",
        attempt: int | None = None,
        left_hand_joints: object | None = None,
        right_hand_joints: object | None = None,
    ) -> PreparedPosePublication:
        if self._closed:
            raise ContractError("RootPosePublisher is closed")
        try:
            hands = resolve_hand_targets(
                left_hand_joints=left_hand_joints,
                right_hand_joints=right_hand_joints,
                default=self._default_hand_targets,
            )
            message = encode_pose_v5(buffer, hand_targets=hands)
            decoded = verify_pose_v5_parity(message, buffer, hand_targets=hands)
        except BaseException as error:
            _tag_failure(error, "zmq_encoding")
            raise
        try:
            archived = self._bundle.archive_transmission(
                message,
                first_frame_index=int(decoded.frame_index[0]),
                last_frame_index=int(decoded.frame_index[-1]),
                phase=phase,
                attempt=attempt,
            )
        except BaseException as error:
            _tag_failure(error, "artifact_enqueue")
            raise
        sequence = self._next_publication
        self._next_publication += 1
        publication = PreparedPosePublication(
            message=message,
            archive=archived,
            phase=phase,
            attempt=attempt,
            _publisher_token=self._publication_token,
            _sequence=sequence,
        )
        self._prepared_publications[sequence] = publication
        return publication
```

- [ ] **Step 4: Run v5 and protected-v1 tests**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_zmq_v*.py' -v`

Expected: both suites pass; the v1 golden bytes are unchanged.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/zmq_v5.py tests/python/test_sonic_zmq_v5.py
git commit -m "feat: add root-capable packed pose v5"
```

### Task 3: Export Physical Pelvis Through the Reference Directory

**Files:**
- Modify: `sonic/python/mm_sonic/reference.py`
- Modify: `tests/python/test_sonic_reference.py`

**Interfaces:**
- Consumes: existing `ReferenceDiagnostics.physical_pelvis_position`.
- Produces: `write_reference_bundle(..., body_position_source: Literal["untracked-zero", "physical-pelvis"] = "untracked-zero")`; the default remains byte-compatible, while physical mode writes exact pelvis bits and declares the mode in `info.txt`.

- [ ] **Step 1: Add failing mode and parity tests**

```python
def test_physical_pelvis_mode_writes_exact_body_position_bits(self):
    result = write_reference_bundle(
        self.bundle, self.canonical, self.diagnostics,
        source_sha256=self.source_hash, scene_id="grail-curb-low",
        route_id="curb-forward", body_position_source="physical-pelvis",
    )
    body = np.loadtxt(result.reference_directory / "body_pos.csv", delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    np.testing.assert_array_equal(body.view(np.uint32), self.diagnostics.physical_pelvis_position.view(np.uint32))
    self.assertIn("body_position: physical pelvis", (result.reference_directory / "info.txt").read_text())

def test_default_reference_remains_zero_root(self):
    result = self._write()
    body = np.loadtxt(result.reference_directory / "body_pos.csv", delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    np.testing.assert_array_equal(body.view(np.uint32), np.zeros_like(body).view(np.uint32))
```

- [ ] **Step 2: Verify the new keyword fails**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_reference.py' -v`

Expected: the physical-mode test fails with `unexpected keyword argument 'body_position_source'`.

- [ ] **Step 3: Implement the explicit mode**

```python
BodyPositionSource = Literal["untracked-zero", "physical-pelvis"]

def _reference_body_position(
    canonical: CanonicalTargetBuffer,
    diagnostics: ReferenceDiagnostics,
    source: BodyPositionSource,
) -> tuple[np.ndarray, str]:
    if source == "untracked-zero":
        return np.zeros((canonical.count, 3), dtype=np.float32), "zero and untracked"
    if source == "physical-pelvis":
        return diagnostics.physical_pelvis_position, "physical pelvis"
    raise ContractError("body_position_source must be untracked-zero or physical-pelvis")
```

Use the returned array for `body_pos.csv`, include the returned label in `info.txt`, and update `validate_reference_bundle` to compare the CSV against the selected expected array. Keep `canonical_target.npz` and its hash unchanged because root is an explicit companion channel.

- [ ] **Step 4: Run reference and codec tests**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_*.py' -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/reference.py tests/python/test_sonic_reference.py
git commit -m "feat: export physical pelvis references explicitly"
```

### Task 4: Decode and Atomically Merge v5 Root Positions in GEAR

**Files:**
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp`
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`
- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp`

**Interfaces:**
- Consumes: protocol-v5 `body_pos` bytes shaped `[N,3]`.
- Produces: `IncomingData::body_pos`, v5 validation, and `MotionSequence::BodyPositions(frame)[0]` containing the transmitted root after every successful merge.

- [ ] **Step 1: Write failing merger tests**

```cpp
TEST(StreamedRootMotion, CopiesIncomingAndOverlapBodyPositions) {
  StreamedMotionMerger merger;
  auto first = MakeV5Incoming({0, 5, 10}, {{{0,0,0.80}}, {{1,0,0.82}}, {{2,0,0.84}}});
  auto a = merger.MergeIncomingData(first, 0);
  ASSERT_NE(a.motion, nullptr);
  EXPECT_DOUBLE_EQ(a.motion->BodyPositions(2)[0][0], 2.0);
  auto second = MakeV5Incoming({10, 15}, {{{2,0,0.84}}, {{3,0,0.86}}});
  auto b = merger.MergeIncomingData(second, 1);
  ASSERT_NE(b.motion, nullptr);
  EXPECT_DOUBLE_EQ(b.motion->BodyPositions(b.motion->timesteps - 1)[0][2], 0.86);
}

TEST(StreamedRootMotion, RejectsV5WithoutOneRootPerFrame) {
  auto data = MakeV5Incoming({0, 5}, {{{0,0,0.8}}});
  StreamedMotionMerger merger;
  EXPECT_EQ(merger.MergeIncomingData(data, 0).motion, nullptr);
}
```

- [ ] **Step 2: Build and observe the missing member failure**

Run: `cmake -S gear_sonic_deploy -B gear_sonic_deploy/build`

Run: `cmake --build gear_sonic_deploy/build --target run_tests -j2`

Expected: compilation fails because `IncomingData` has no `body_pos` member.

- [ ] **Step 3: Implement v5 merger ownership**

```cpp
std::vector<std::vector<std::array<double, 3>>> body_pos;  // [frame][body][xyz]
```

For v5, require nonempty joint data, one body quaternion, one body position, and identical frame counts. In `CreateNewMotion`, reserve one position body. In `CopyIncomingDataToMotion`, copy `data.body_pos[frame][0]` into `BodyPositions(dst)[0]`. In `CopyOldDataToNewMotion`, copy old body positions alongside joints and quaternions. Allocate and populate the complete replacement `MotionSequence` before assigning `streamed_motion_`, retaining the existing atomic commit behavior.

- [ ] **Step 4: Implement endpoint v5 decoding**

```cpp
int body_pos_idx = -1;
// field scan
else if (f.name == "body_pos" || f.name == "body_pos_w") body_pos_idx = static_cast<int>(i);

if (protocol_version == 5 && body_pos_idx < 0) {
  std::cerr << "[ZMQEndpointInterface] Version 5 missing required field 'body_pos'" << std::endl;
  return result;
}
```

Accept v5 in the same joint-motion branch as v1, require shape `[N,3]`, decode `f32` and `f64` through the existing endian-safe helpers, set `incoming.protocol_version = 5`, and assign `incoming.body_pos`. Keep v1/v2/v3/v4 branches unchanged and reject a mid-session version switch as before.

- [ ] **Step 5: Build and run the C++ unit tests**

Run: `cmake --build gear_sonic_deploy/build --target run_tests -j2`

Run: `gear_sonic_deploy/target/release/run_tests --gtest_filter='StreamedRootMotion.*'`

Expected: build succeeds and both `StreamedRootMotion` tests pass.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp
git commit -m "feat: merge root positions from packed pose v5"
```

### Task 5: Canonicalize Root Trajectories Identically in Python and C++

**Files:**
- Create: `sonic/fixtures/root_trajectory_v1.json`
- Create: `sonic/python/mm_sonic/root_trajectory.py`
- Create: `tests/python/test_sonic_root_trajectory.py`
- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/root_trajectory.hpp`
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp`

**Interfaces:**
- Produces Python `canonical_root_trajectory(position_f32, quat_wxyz_f32) -> np.ndarray` and C++ `CanonicalRootTrajectory(std::span<const std::array<double,3>>, std::span<const std::array<double,4>>) -> std::vector<std::array<double,3>>`.

- [ ] **Step 1: Add the shared fixture and failing invariance tests**

The fixture contains 10 explicit float32 positions, 10 unit WXYZ quaternions, and 10 expected triples calculated in float64 then rounded once to float32. Tests load it and assert `atol=1e-6`, frame zero XY equals zero, frame zero Z equals physical pelvis Z, adding a constant XY translation changes no output, and premultiplying every input by one yaw rotation changes no output.

```python
actual = canonical_root_trajectory(fixture.position, fixture.quaternion)
np.testing.assert_allclose(actual, fixture.expected, rtol=0.0, atol=1.0e-6)
np.testing.assert_allclose(actual[0], [0.0, 0.0, fixture.position[0, 2]], atol=1.0e-6)
np.testing.assert_allclose(canonical_root_trajectory(translated, fixture.quaternion), actual, atol=1.0e-6)
np.testing.assert_allclose(canonical_root_trajectory(rotated_p, rotated_q), actual, atol=1.0e-6)
```

- [ ] **Step 2: Run and confirm the missing implementation**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_root_trajectory.py' -v`

Expected: import fails for `mm_sonic.root_trajectory`.

- [ ] **Step 3: Implement the Python oracle**

```python
def canonical_root_trajectory(position: object, quaternion_wxyz: object) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != (p.shape[0], 4) or p.shape[0] == 0:
        raise ContractError("root trajectory requires position [N,3] and quaternion [N,4]")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ContractError("root trajectory inputs must be finite")
    yaw = math.atan2(2.0 * (q[0, 0] * q[0, 3] + q[0, 1] * q[0, 2]), 1.0 - 2.0 * (q[0, 2] ** 2 + q[0, 3] ** 2))
    c, s = math.cos(yaw), math.sin(yaw)
    delta = p - np.array([p[0, 0], p[0, 1], 0.0])
    out = np.empty_like(delta)
    out[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    out[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    out[:, 2] = delta[:, 2]
    return np.ascontiguousarray(out, dtype=np.float32)
```

- [ ] **Step 4: Implement the same double-precision arithmetic in C++**

```cpp
#pragma once

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

inline std::vector<std::array<double, 3>> CanonicalRootTrajectory(
    std::span<const std::array<double, 3>> positions,
    std::span<const std::array<double, 4>> quaternions) {
  if (positions.empty() || positions.size() != quaternions.size())
    throw std::invalid_argument("root trajectory shape mismatch");
  const auto& q = quaternions.front();
  const double yaw = std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
  const double c = std::cos(yaw), s = std::sin(yaw);
  std::vector<std::array<double, 3>> out;
  out.reserve(positions.size());
  for (const auto& p : positions) {
    const double dx = p[0] - positions.front()[0];
    const double dy = p[1] - positions.front()[1];
    out.push_back({c * dx + s * dy, -s * dx + c * dy, p[2]});
  }
  return out;
}
```

Add a GoogleTest fixture with the same literals and assert every component is within `1e-6` of the JSON expected values.

- [ ] **Step 5: Run both language gates**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_root_trajectory.py' -v`

Run: `cmake --build gear_sonic_deploy/build --target run_tests -j2`

Run: `gear_sonic_deploy/target/release/run_tests --gtest_filter='RootTrajectory.*'`

Expected: all Python and C++ canonicalization/invariance tests pass at `1e-6`.

- [ ] **Step 6: Commit in each repository**

```bash
git add sonic/fixtures/root_trajectory_v1.json sonic/python/mm_sonic/root_trajectory.py tests/python/test_sonic_root_trajectory.py
git commit -m "test: pin root trajectory canonicalization"
```

```bash
git add gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/root_trajectory.hpp gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp
git commit -m "feat: canonicalize streamed root trajectory"
```

### Task 6: Register the 30-Value Runtime Observation and Config

**Files:**
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- Create: `gear_sonic_deploy/policy/release/observation_config_root_conditioned.yaml`
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp`

**Interfaces:**
- Consumes: `MotionSequence::BodyPositions`, `MotionSequence::BodyQuaternions`, current frame, play state, and `CanonicalRootTrajectory`.
- Produces: `GatherMotionRootPositionRefHeadingMultiFrame(buf, offset, 10, 5)` and registry entry `motion_root_position_refheading_10frame_step5` with dimension 30.

- [ ] **Step 1: Add failing gatherer-layout tests**

```cpp
TEST(RootTrajectory, GathererClampsFramesAndWritesFrameMajorXYZ) {
  auto motion = MakeRootMotion(7);
  std::vector<double> output(35, -99.0);
  ASSERT_TRUE(GatherRootFixture(motion, 5, 10, 5, output, 2));
  EXPECT_EQ(output[0], -99.0);
  EXPECT_NEAR(output[2], 0.0, 1e-6);
  EXPECT_NEAR(output[3], 0.0, 1e-6);
  EXPECT_NEAR(output[4], motion.BodyPositions(5)[0][2], 1e-6);
  EXPECT_NEAR(output[29], output[32], 1e-6);  // clamped final frame
}
```

- [ ] **Step 2: Implement and register the gatherer**

```cpp
{"motion_root_position_refheading_10frame_step5", 30,
 [this](std::vector<double>& buf, size_t offset) {
   return GatherMotionRootPositionRefHeadingMultiFrame(buf, offset, 10, 5);
 }},
```

The gatherer chooses frames with the same `operator_state.play`, step, and final-frame clamp behavior as `GatherMotionAnchorOrientationMutiFrame`, reads body index zero, canonicalizes all selected frames using the first selected quaternion heading, and copies XYZ frame-major into the requested offset. It fails closed if either body positions or body quaternions are absent.

- [ ] **Step 3: Add the root-conditioned observation config**

Copy the deployed 1,762-value configuration verbatim, append this enabled encoder observation after every existing entry, and require it only for G1:

```yaml
    - name: "motion_root_position_refheading_10frame_step5"
      enabled: true
  encoder_modes:
    - name: "g1"
      mode_id: 0
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_10frame_step5
        - motion_joint_velocities_10frame_step5
        - motion_anchor_orientation_10frame_step5
        - motion_root_position_refheading_10frame_step5
```

Keep teleop and SMPL required lists byte-for-byte equivalent to the deployed config. Document `Encoder input dimension: 1792` and `new range: [1762,1792)` in the header.

- [ ] **Step 4: Build and validate dimensions**

Run: `cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref run_tests -j2`

Expected: both targets build.

Run: `gear_sonic_deploy/target/release/run_tests --gtest_filter='RootTrajectory.*:StreamedRootMotion.*'`

Expected: all root tests pass.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_streamed_root_motion.cpp gear_sonic_deploy/policy/release/observation_config_root_conditioned.yaml
git commit -m "feat: expose canonical root trajectory to SONIC"
```

### Task 7: Prove File/Stream and Baseline Non-Regression

**Files:**
- Modify: `tests/python/test_sonic_reference.py`
- Modify: `tests/python/test_sonic_zmq_v5.py`
- Modify: `docs/superpowers/results/2026-07-22-g1-sonic-root-stream-runtime.md`

**Interfaces:**
- Consumes: one `RootTargetBuffer` encoded through v5 and exported through physical-pelvis reference mode.
- Produces: a result record containing repository SHAs, test commands, protocol version, field layout, canonicalization tolerance, and hashes for the fixture and config.

- [ ] **Step 1: Add the end-to-end bit-parity test**

```python
def test_reference_and_stream_body_positions_are_bit_equal(self):
    root_path = Path(self._temporary.name)
    chunk = make_target_chunk()
    root = RootTargetBuffer.from_target_chunk(chunk)
    decoded = decode_pose_v5(encode_pose_v5(root))
    bundle = RunBundle.create(root_path, "root-parity", "trial")
    written = write_reference_bundle(
        bundle, root.canonical, diagnostics_from_chunk(chunk), source_sha256="a" * 64,
        scene_id="grail-curb-low", route_id="curb-forward",
        body_position_source="physical-pelvis",
    )
    file_root = np.loadtxt(written.reference_directory / "body_pos.csv", delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    np.testing.assert_array_equal(file_root.view(np.uint32), decoded.body_position.view(np.uint32))
```

- [ ] **Step 2: Run the full affected Motion Matching suite**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_*.py' -v`

Expected: all tests pass.

- [ ] **Step 3: Run the complete GEAR C++ unit-test target**

Run: `cmake --build gear_sonic_deploy/build --target run_tests -j2`

Run: `gear_sonic_deploy/target/release/run_tests`

Expected: all existing and new C++ tests pass.

- [ ] **Step 4: Write the result record**

Record exact SHAs and command outputs, plus:

```json
{
  "protocol_v1_unchanged": true,
  "root_protocol_version": 5,
  "body_position_source": "physical-pelvis",
  "root_observation": "motion_root_position_refheading_10frame_step5",
  "root_observation_range": [1762, 1792],
  "canonicalization_atol": 1e-6,
  "file_stream_bit_equal": true
}
```

- [ ] **Step 5: Commit the evidence**

```bash
git add tests/python/test_sonic_reference.py tests/python/test_sonic_zmq_v5.py docs/superpowers/results/2026-07-22-g1-sonic-root-stream-runtime.md
git commit -m "test: qualify SONIC root stream runtime"
```
