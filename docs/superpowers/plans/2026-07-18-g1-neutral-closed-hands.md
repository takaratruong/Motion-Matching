# G1 Neutral Closed Hands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive both G1 Dex3 hands to an explicit relaxed-fist target through the real MM → GEAR → DDS → MuJoCo physics path, correct the discovered actuator-order mismatch, and produce close-up visual and machine-readable proof.

**Architecture:** A focused `hands.py` module owns mirrored Dex3 motor order, limits, targets, validation, and stable identity. The pose-v1 codec remains able to decode legacy body-only archives, while every production `PosePublisher` send carries two explicit seven-value hand fields and supports nonpersistent per-call overrides. Run-local robot XML motors are reordered to MuJoCo joint traversal and verified from the loaded model before simulation; the manual runner then binds scene and hand identities into its evidence and renders an upper-body close-up.

**Tech Stack:** Python 3.10+, NumPy, MuJoCo Python bindings, pyzmq, `unittest`, ffmpeg/ffprobe, existing GEAR/SONIC binaries.

## Global Constraints

- Keep `/tmp/groot-wbc-plan-inspect` and every official GEAR/SONIC source asset byte-identical.
- Modify only generated or copied run-local XML; preserve the previous run and release artifacts for rollback.
- Left Dex3 command order is `thumb0, thumb1, thumb2, middle0, middle1, index0, index1`.
- Right Dex3 command order is `thumb0, thumb1, thumb2, index0, index1, middle0, middle1`.
- Relaxed-fist radians are left `(0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875)` and right `(0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875)`, derived from the pinned Dex3 midpoint limits.
- The 29-joint MM body payload and unmodified SONIC policy output must not be reordered or numerically changed.
- Visual inspection of a close-up physics replay occurs before the full formal qualification.
- Formal acceptance requires per-joint settled median absolute hand error `<= 0.20 rad`, minimum root height `>= 0.65 m`, minimum pelvis-up dot `>= 0.90`, stopped-command drift `<= 0.25 m`, no fall marker, and exact 29-joint command-stream parity.
- Actuator mapping ambiguity, invalid hand targets, missing provenance, or a one-hand-only enriched message must fail before simulation or transport.

## File Structure

- Create `sonic/python/mm_sonic/hands.py`: immutable Dex3 orders, limits, neutral profile, override validation, identity hashing, and settled tracking metrics.
- Modify `sonic/python/mm_sonic/zmq_v1.py`: optional pose-v1 hand fields, legacy decode compatibility, and explicit production publisher defaults/overrides.
- Modify `sonic/python/mm_sonic/scene.py`: deterministic motor normalization plus loaded-model routing verification and provenance.
- Modify `sonic/python/mm_sonic/manual_demo.py`: materialize a genuinely run-local corrected scene and record explicit hand/scene identities.
- Modify `sonic/python/mm_sonic/manual_evidence.py`: authenticate hand fields, actuator identity, and settled hand tracking.
- Modify `sonic/python/mm_sonic/manual_replay.py`: make the second video panel an upper-body/hand close-up and emit a close-up still.
- Create `tests/python/test_sonic_hands.py` and `tests/python/test_sonic_manual_replay.py`; modify `tests/python/test_sonic_zmq_v1.py`, `tests/python/test_sonic_scene.py`, `tests/python/test_sonic_manual_demo.py`, and `tests/python/test_sonic_manual_evidence.py`.
- Controller-owned gate: `/home/ubuntu/reliable-claude-projects/motion-matching/evaluators/evaluate_g1_neutral_closed_hands.py`.

---

### Task 1: Freeze the mirrored Dex3 hand contract

**Files:**
- Create: `sonic/python/mm_sonic/hands.py`
- Create: `tests/python/test_sonic_hands.py`

**Interfaces:**
- Consumes: `mm_sonic.joints.ContractError` and state objects exposing `sim_time_s` and `qpos`.
- Produces: `Dex3HandTargets`, `NEUTRAL_HAND_TARGETS`, `resolve_hand_targets(...)`, `hand_targets_record(...)`, `parse_hand_targets_record(...)`, `hand_qpos_addresses(...)`, `hand_joint_ranges(...)`, and `measure_hand_tracking(...)`.

- [ ] **Step 1: Write the failing profile and validation tests**

Create `tests/python/test_sonic_hands.py` with tests that assert exact mirrored order, exact float32 targets, stable hash, one-side override behavior, and rejection of wrong width, booleans, nonfinite values, and values outside the side-specific limits:

```python
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.hands import (
    LEFT_HAND_JOINT_ORDER,
    NEUTRAL_HAND_TARGETS,
    RIGHT_HAND_JOINT_ORDER,
    hand_targets_record,
    measure_hand_tracking,
    resolve_hand_targets,
)
from mm_sonic.joints import ContractError


class Dex3HandContractTests(unittest.TestCase):
    def test_neutral_profile_has_exact_mirrored_motor_order_and_targets(self) -> None:
        self.assertEqual(
            LEFT_HAND_JOINT_ORDER,
            ("left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
             "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
             "left_hand_middle_1_joint", "left_hand_index_0_joint",
             "left_hand_index_1_joint"),
        )
        self.assertEqual(
            RIGHT_HAND_JOINT_ORDER,
            ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
             "right_hand_thumb_2_joint", "right_hand_index_0_joint",
             "right_hand_index_1_joint", "right_hand_middle_0_joint",
             "right_hand_middle_1_joint"),
        )
        np.testing.assert_array_equal(
            NEUTRAL_HAND_TARGETS.left_f32,
            np.asarray((0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875), dtype="<f4"),
        )
        np.testing.assert_array_equal(
            NEUTRAL_HAND_TARGETS.right_f32,
            np.asarray((0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875), dtype="<f4"),
        )
        record = hand_targets_record(NEUTRAL_HAND_TARGETS)
        self.assertEqual(record["profile"], "dex3-relaxed-fist-v1")
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

    def test_one_side_override_is_nonpersistent_data(self) -> None:
        left = (0.1, 0.2, 0.8, -0.7, -0.8, -0.6, -0.7)
        overridden = resolve_hand_targets(left_hand_joints=left)
        self.assertEqual(overridden.left, left)
        self.assertEqual(overridden.right, NEUTRAL_HAND_TARGETS.right)
        self.assertEqual(resolve_hand_targets(), NEUTRAL_HAND_TARGETS)

    def test_invalid_override_fails_closed(self) -> None:
        for value in ((0.0,) * 6, (0.0,) * 6 + (float("nan"),),
                      (0.0,) * 6 + (True,), (0.0,) * 6 + (9.0,)):
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    resolve_hand_targets(left_hand_joints=value)

    def test_tracking_uses_final_second_and_all_fourteen_joints(self) -> None:
        target = np.asarray(NEUTRAL_HAND_TARGETS.left + NEUTRAL_HAND_TARGETS.right)
        states = []
        for index in range(101):
            qpos = np.zeros(21)
            qpos[7:21] = target + (0.5 if index < 50 else 0.05)
            states.append(SimpleNamespace(sim_time_s=0.02 * index, qpos=qpos))
        report = measure_hand_tracking(
            states,
            qpos_addresses=tuple(range(7, 21)),
            joint_ranges=tuple((-2.0, 2.0) for _ in range(14)),
            targets=NEUTRAL_HAND_TARGETS,
            final_seconds=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.median_absolute_error_rad), 14)
        self.assertLessEqual(max(report.median_absolute_error_rad), 0.20)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.python.test_sonic_hands -v
```

Expected: import failure for `mm_sonic.hands`.

- [ ] **Step 3: Implement the immutable contract and metric**

Create `hands.py` with frozen tuple data, exact pinned limit arrays, midpoint-derived neutral values, and these signatures:

```python
@dataclass(frozen=True)
class Dex3HandTargets:
    profile: str
    left: tuple[float, ...]
    right: tuple[float, ...]

    @property
    def left_f32(self) -> np.ndarray:
        return np.asarray(self.left, dtype="<f4")

    @property
    def right_f32(self) -> np.ndarray:
        return np.asarray(self.right, dtype="<f4")


@dataclass(frozen=True)
class HandTrackingReport:
    median_absolute_error_rad: tuple[float, ...]
    median_position_rad: tuple[float, ...]
    unexpected_limit_joints: tuple[str, ...]
    error_pass: bool
    limit_pass: bool

    @property
    def passed(self) -> bool:
        return self.error_pass and self.limit_pass


def resolve_hand_targets(
    *,
    left_hand_joints: object | None = None,
    right_hand_joints: object | None = None,
    default: Dex3HandTargets = NEUTRAL_HAND_TARGETS,
) -> Dex3HandTargets:
    """Resolve each supplied side independently; omission restores its default."""


def hand_targets_record(targets: Dex3HandTargets) -> dict[str, object]:
    """Return profile, mirrored joint orders, f32 vectors, and a domain-separated SHA-256."""


def parse_hand_targets_record(value: object) -> Dex3HandTargets:
    """Require the exact record keys/orders and recompute its vector SHA-256."""


def hand_qpos_addresses(model: object) -> tuple[int, ...]:
    """Resolve the 14 exact named joints through model.jnt_qposadr."""


def hand_joint_ranges(model: object) -> tuple[tuple[float, float], ...]:
    """Resolve the same 14 exact named joints through model.jnt_range."""


def measure_hand_tracking(
    states: Sequence[object],
    *,
    qpos_addresses: Sequence[int],
    joint_ranges: Sequence[tuple[float, float]],
    targets: Dex3HandTargets,
    final_seconds: float = 1.0,
) -> HandTrackingReport:
    """Measure final-window medians, <=0.20 rad errors, and unexpected limit pinning."""
```

The target SHA-256 input is `b"mm-sonic-dex3-hand-targets/v1\0"`, then each side label, NUL-separated joint names, and the side's contiguous little-endian float32 bytes. Reject values before converting them to float32. A non-limit target is unexpectedly pinned when its final-window median is within `0.01 rad` of a registered limit while the target itself is more than `0.01 rad` from that limit.

- [ ] **Step 4: Run the hand contract tests and confirm GREEN**

Run the Step 2 command. Expected: four tests pass.

- [ ] **Step 5: Commit the contract slice**

```bash
git add sonic/python/mm_sonic/hands.py tests/python/test_sonic_hands.py
git commit -m "feat: define mirrored Dex3 hand targets"
```

---

### Task 2: Carry explicit hands in pose-v1 without breaking legacy evidence

**Files:**
- Modify: `sonic/python/mm_sonic/zmq_v1.py`
- Modify: `tests/python/test_sonic_zmq_v1.py`

**Interfaces:**
- Consumes: `Dex3HandTargets`, `NEUTRAL_HAND_TARGETS`, and `resolve_hand_targets(...)` from Task 1.
- Produces: `pose_header(count, include_hands=False)`, `encode_pose_v1(..., hand_targets=None)`, decoded optional hand arrays, and `PosePublisher` per-call side overrides.

- [ ] **Step 1: Add failing enriched-codec and publisher tests**

Add tests that build one enriched message and assert exact field order and payload bits:

```python
def test_enriched_message_preserves_both_hand_vectors_and_legacy_still_decodes(self) -> None:
    buffer = make_buffer(2, 40)
    enriched = encode_pose_v1(buffer, hand_targets=NEUTRAL_HAND_TARGETS)
    decoded = decode_pose_v1(enriched)
    self.assertEqual(
        [field["name"] for field in decoded.header["fields"]],
        ["joint_pos", "joint_vel", "body_quat_w", "frame_index",
         "left_hand_joints", "right_hand_joints", "catch_up"],
    )
    _assert_array_bits_equal(
        self, decoded.left_hand_joints, NEUTRAL_HAND_TARGETS.left_f32, "<f4"
    )
    _assert_array_bits_equal(
        self, decoded.right_hand_joints, NEUTRAL_HAND_TARGETS.right_f32, "<f4"
    )
    legacy = decode_pose_v1(encode_pose_v1(buffer))
    self.assertIsNone(legacy.left_hand_joints)
    self.assertIsNone(legacy.right_hand_joints)


def test_enriched_decoder_rejects_one_hand_only_and_invalid_hand_payload(self) -> None:
    buffer = make_buffer(1, 0)
    message = encode_pose_v1(buffer, hand_targets=NEUTRAL_HAND_TARGETS)
    header = pose_header(1, include_hands=True)
    header["fields"] = [
        field for field in header["fields"] if field["name"] != "right_hand_joints"
    ]
    with self.assertRaisesRegex(ContractError, "both hand"):
        decode_pose_v1(replace_header(message, header))


def test_publisher_defaults_to_neutral_and_override_reverts_next_call(self) -> None:
    first = publisher.prepare(make_buffer(1, 0), phase="readiness", attempt=1)
    second = publisher.prepare(
        make_buffer(1, 0), phase="readiness", attempt=2,
        left_hand_joints=(0.1, 0.2, 0.8, -0.7, -0.8, -0.6, -0.7),
    )
    third = publisher.prepare(make_buffer(1, 0), phase="readiness", attempt=3)
    self.assertEqual(decode_pose_v1(first.message).left_hand_joints.tolist(),
                     NEUTRAL_HAND_TARGETS.left_f32.tolist())
    self.assertNotEqual(decode_pose_v1(second.message).left_hand_joints.tolist(),
                        NEUTRAL_HAND_TARGETS.left_f32.tolist())
    self.assertEqual(decode_pose_v1(third.message).left_hand_joints.tolist(),
                     NEUTRAL_HAND_TARGETS.left_f32.tolist())
```

Use the existing fake publisher context inside `PosePublisherPhaseTests` for the third test, creating one fresh `RunBundle` and closing the publisher in `finally`.

- [ ] **Step 2: Run the focused codec tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.python.test_sonic_zmq_v1 -v
```

Expected: failures for the new `include_hands`, `hand_targets`, decoded fields, and override arguments.

- [ ] **Step 3: Implement the dual-layout codec and explicit publisher default**

Keep the legacy layout byte-identical when `hand_targets is None`. For enriched messages, insert these two descriptors immediately before `catch_up`:

```python
{"name": "left_hand_joints", "dtype": "f32", "shape": [7]}
{"name": "right_hand_joints", "dtype": "f32", "shape": [7]}
```

Extend `DecodedPoseV1` with:

```python
left_hand_joints: np.ndarray | None
right_hand_joints: np.ndarray | None
```

`_decode_header` must accept exactly `pose_header(count, include_hands=False)` or `pose_header(count, include_hands=True)`. `encode_pose_v1` appends both f32 vectors together or neither. `verify_pose_v1_parity` compares exact hand bits when an expected `Dex3HandTargets` is supplied.

Give `PosePublisher.__init__` a validated `default_hand_targets` argument defaulting to `NEUTRAL_HAND_TARGETS`. Give `prepare` and `send` keyword-only `left_hand_joints` and `right_hand_joints` arguments. Resolve each call independently and pass the resolved target to both encode and parity verification; never store an override as the next call's default.

- [ ] **Step 4: Run the codec suite and confirm GREEN**

Run the Step 2 command. Expected: all codec, artifact, publisher, dependency, and loopback tests pass or retain their existing guarded skip.

- [ ] **Step 5: Commit the transport slice**

```bash
git add sonic/python/mm_sonic/zmq_v1.py tests/python/test_sonic_zmq_v1.py
git commit -m "feat: publish explicit Dex3 hand targets"
```

---

### Task 3: Normalize and prove run-local actuator routing

**Files:**
- Modify: `sonic/python/mm_sonic/scene.py`
- Modify: `tests/python/test_sonic_scene.py`

**Interfaces:**
- Consumes: official/copy robot XML bytes.
- Produces: `normalize_run_local_actuators(robot_bytes, label=...) -> tuple[bytes, tuple[str, ...], str]` and `verify_loaded_actuator_routing(model) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing XML and loaded-model boundary tests**

Extend the synthetic robot fixture with one motor per regular joint, intentionally in reverse order. Add tests for deterministic normalization, duplicate/missing/unknown motor rejection, and loaded routing:

```python
def test_run_local_actuators_match_loaded_joint_traversal_with_sentinels(self):
    registered = self._register_flat_scene_with_synthetic_actuators()
    model = mujoco.MjModel.from_xml_path(str(registered.gear_scene_xml))
    order = verify_loaded_actuator_routing(model)
    self.assertEqual(order, tuple(model.joint(i).name for i in range(1, model.njnt)))
    sentinels = np.arange(1, model.nu + 1, dtype=np.float64)
    routed = np.zeros(model.nu, dtype=np.float64)
    for joint_id, sentinel in enumerate(sentinels, start=1):
        routed[joint_id - 1] = sentinel
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        self.assertEqual(routed[actuator_id], sentinels[joint_id - 1])


def test_official_boundary_places_left_hand_before_right_arm(self):
    model = mujoco.MjModel.from_xml_path(str(registered.gear_scene_xml))
    self.assertEqual(model.joint(int(model.actuator_trnid[22, 0])).name,
                     "left_hand_thumb_0_joint")
    self.assertEqual(model.joint(int(model.actuator_trnid[29, 0])).name,
                     "right_shoulder_pitch_joint")
```

The official integration test also asserts the complete 43-name tuple and that `output_hashes["official_robot"]` remains the pinned official SHA while `output_hashes["robot_include"]` names only the normalized run-local bytes.

- [ ] **Step 2: Run scene tests and confirm RED**

```bash
SONIC_GEAR_CHECKOUT=/tmp/groot-wbc-plan-inspect \
SONIC_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.python.test_sonic_scene -v
```

Expected: normalization/routing symbols are missing or the official boundary shows actuator 22 routed to `right_shoulder_pitch_joint`.

- [ ] **Step 3: Implement document-order normalization and loaded verification**

`normalize_run_local_actuators` must parse with `_parse_xml_root`, find exactly one `worldbody` and one `actuator`, collect named `joint` elements only below `worldbody` in document traversal, require one `<motor joint="...">` child per collected joint and no other actuator child type, reject duplicate joint or motor names, reorder the existing element objects, indent, and serialize deterministic UTF-8 XML.

The mapping digest is SHA-256 over a canonical ASCII JSON object with exactly two keys: `joint_order`, whose value is the complete ordered-name array, and `schema`, whose value is `mm-sonic-actuator-joint-order/v1`. Encode it with a trailing newline, `ensure_ascii=True`, `allow_nan=False`, compact separators, and sorted keys. `verify_loaded_actuator_routing` requires one free root at joint 0, `model.nu == model.njnt - 1`, and for every zero-based actuator slot `int(model.actuator_trnid[slot, 0]) == slot + 1`; its error names the first actuator, actual joint, and expected joint.

Change `_run_local_robot_bytes` to return `(robot_bytes, actuator_joint_order, actuator_joint_order_sha256)` from the normalizer. Call loaded verification immediately after `MjModel.from_xml_path`, require its order to equal the returned order, and add `actuator_joint_order` plus `actuator_joint_order_sha256` to `scene_registration.json`. Re-run the same verifier and digest comparison during registered replay authentication.

- [ ] **Step 4: Run scene tests and confirm GREEN**

Run the Step 2 command. Expected: all synthetic and official GEAR scene tests pass, including the left-hand/right-arm sentinel boundary.

- [ ] **Step 5: Commit the routing slice**

```bash
git add sonic/python/mm_sonic/scene.py tests/python/test_sonic_scene.py
git commit -m "fix: align run-local actuators with MuJoCo joints"
```

---

### Task 4: Wire the corrected scene and neutral profile into the manual runner

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_evidence.py`
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `tests/python/test_sonic_manual_evidence.py`

**Interfaces:**
- Consumes: `normalize_run_local_actuators`, `verify_loaded_actuator_routing`, `NEUTRAL_HAND_TARGETS`, and the default enriched `PosePublisher`.
- Produces: a local scene whose include points to the same run's normalized robot plus command-artifact and summary `scene_control`/`hand_control` records.

- [ ] **Step 1: Write failing run-local copy and summary tests**

Add a temporary source-run fixture whose `gear_scene.xml` contains an absolute include to a source robot with the old body-first actuator order. Assert `_copy_scene` writes a new local robot, rewrites the local overlay include to that exact new path, leaves source bytes unchanged, and returns exactly these three digest keys, each matching the SHA-256 syntax:

```python
self.assertEqual(
    set(scene_control),
    {"gear_scene_sha256", "gear_robot_sha256",
     "actuator_joint_order_sha256"},
)
for digest in scene_control.values():
    self.assertRegex(digest, r"^[0-9a-f]{64}$")
```

Also assert `manual-commands.json` uses schema `mm-sonic-manual-command/v3` with `hand_control == hand_targets_record(NEUTRAL_HAND_TARGETS)`, the manual summary uses schema `mm-sonic-manual-demo/v3` with the same record, and decoding every fake publisher message yields both neutral vectors.

- [ ] **Step 2: Run the manual-demo tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.python.test_sonic_manual_demo -v
```

Expected: `_copy_scene` still retains the source absolute include and the v3 identity fields are absent.

- [ ] **Step 3: Materialize and bind the corrected local scene**

Change `_copy_scene` to return `(scene_path, scene_control_record)`. Read `source_run/scene/gear_scene.xml` and `source_run/scene/gear_robot.xml`, normalize only the copied robot's actuator elements, write it as `bundle.path/scene/gear_robot.xml`, rewrite the copied overlay's sole include to that absolute local path, load the result, and call `verify_loaded_actuator_routing` before returning. Hash the actual bytes written; do not reuse source hashes.

Extend `ManualCommandArtifact` with `hand_targets: Dex3HandTargets`. Make `manual_command_artifact_bytes` accept a required `hand_targets` keyword and emit its canonical record under `hand_control`; make the strict parser call `parse_hand_targets_record` so any vector, order, profile, or digest tampering fails. Construct `CommandRecorder` and the publisher with `NEUTRAL_HAND_TARGETS`, then add these exact summary keys:

```python
"schema": "mm-sonic-manual-demo/v3",
"hand_control": hand_targets_record(NEUTRAL_HAND_TARGETS),
"scene_control": scene_control,
```

Keep the 29-body-joint generation and publication call sites otherwise unchanged.

- [ ] **Step 4: Run focused manual-demo and publisher tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_manual_evidence \
  tests.python.test_sonic_zmq_v1 -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the runner slice**

```bash
git add sonic/python/mm_sonic/manual_demo.py sonic/python/mm_sonic/manual_evidence.py \
  tests/python/test_sonic_manual_demo.py tests/python/test_sonic_manual_evidence.py
git commit -m "feat: run manual SONIC with neutral closed hands"
```

---

### Task 5: Authenticate hand transport and settled physics tracking

**Files:**
- Modify: `sonic/python/mm_sonic/manual_evidence.py`
- Modify: `tests/python/test_sonic_manual_evidence.py`

**Interfaces:**
- Consumes: enriched archived pose messages, v3 manual summary, run-local MuJoCo model, and Task 1 tracking metric.
- Produces: evidence schema `mm-sonic-manual-evidence/v3` with hand transport, actuator routing, and settled tracking checks.

- [ ] **Step 1: Add failing evidence tests for the exact former false-pass modes**

Update valid fixtures to encode neutral hands and place the 14 final-window qpos values at target plus `0.05 rad`. Add tests that fail on a missing hand field, one changed hand bit, old body-first actuator ordering, error above `0.20 rad`, and a non-limit target pinned at a joint limit. The passing document must include:

```python
self.assertTrue(document["checks"]["hand_transport_exact"])
self.assertTrue(document["checks"]["actuator_routing"])
self.assertTrue(document["checks"]["hand_tracking"])
self.assertLessEqual(
    max(document["hand_tracking"]["median_absolute_error_rad"]), 0.20
)
self.assertEqual(document["hand_control"]["sha256"],
                 hand_targets_record(NEUTRAL_HAND_TARGETS)["sha256"])
```

- [ ] **Step 2: Run manual-evidence tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.python.test_sonic_manual_evidence -v
```

Expected: v3 keys and hand checks are absent.

- [ ] **Step 3: Implement fail-closed v3 auditing**

Make `_validate_summary` return the validated `Dex3HandTargets` and scene-control mapping. Authenticate that its vector identity recomputes to the recorded hash. While loading every archived message, require both hand fields and exact f32 parity with that target; continue concatenating only the unchanged body fields for MM replay parity.

Load `scene/gear_scene.xml`, verify its SHA, its local include and robot SHA, and `verify_loaded_actuator_routing(model)`. Resolve `hand_qpos_addresses(model)` and `hand_joint_ranges(model)`, call `measure_hand_tracking` over the final simulated second, and add these `ManualEvidence` fields:

```python
hand_transport_exact: bool
actuator_routing_pass: bool
hand_tracking: HandTrackingReport
hand_targets: Dex3HandTargets
```

Include all three booleans in `passed`. Emit `mm-sonic-manual-evidence/v3`, the hand profile record, per-joint median errors/positions, unexpected-limit names, and the three new checks. Retain a clear version error for v2 summaries rather than silently claiming old runs satisfy the new full-robot evidence contract.

- [ ] **Step 4: Run evidence and regression tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.python.test_sonic_hands \
  tests.python.test_sonic_zmq_v1 \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_manual_evidence -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the evidence slice**

```bash
git add sonic/python/mm_sonic/manual_evidence.py tests/python/test_sonic_manual_evidence.py
git commit -m "test: qualify closed-hand physics tracking"
```

---

### Task 6: Produce the visual-first close-up and final qualification

**Files:**
- Modify: `sonic/python/mm_sonic/manual_replay.py`
- Create: `tests/python/test_sonic_manual_replay.py`
- Controller-owned: `/home/ubuntu/reliable-claude-projects/motion-matching/evaluators/evaluate_g1_neutral_closed_hands.py`

**Interfaces:**
- Consumes: one completed v3 manual physics run.
- Produces: full replay video, montage, `g1-sonic-hand-closeup.png`, v3 evidence JSON, and accessible copies under `/home/ubuntu`.

- [ ] **Step 1: Change the replay's second panel to a hand-readable camera**

Keep the left panel as the world trajectory. Set the right camera each frame to look at `(root_x, root_y, root_z + 0.45)`, distance `1.45`, azimuth `180`, elevation `-8`. Count and validate the available state rows so the renderer accepts either the five-chunk smoke stream or the 600-row final stream. Build the overlay frame total and montage sample indices from that count. Save the last right-panel render as `g1-sonic-hand-closeup.png`, return `(video, montage, closeup)` from `render_manual_replay`, and print all three paths from the CLI. Add a renderer test with a short valid state stream that asserts all three returned paths exist and the still has nonzero size.

- [ ] **Step 2: Run a short physical visual checkpoint before formal qualification**

Run five operator chunks through the real manual path to obtain a one-frame settled close-up without waiting for the final audit:

```bash
PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 python3 -m mm_sonic.manual_demo \
  --mode script --chunks 5 \
  --output-root /home/ubuntu/mm-sonic-hand-smoke-runs

SMOKE_ROOT=$(find /home/ubuntu/mm-sonic-hand-smoke-runs/manual-sonic \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | \
  sort -n | tail -n 1 | cut -d' ' -f2-)
test -n "$SMOKE_ROOT"
PYTHONPATH=sonic/python MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
python3 -m mm_sonic.manual_replay \
  --run-root "$SMOKE_ROOT" \
  --output-dir /home/ubuntu/g1-sonic-hand-smoke-visual
```

Inspect `/home/ubuntu/g1-sonic-hand-smoke-visual/g1-sonic-hand-closeup.png` directly. Stop here and repair mapping/transport if either hand is open, asymmetric, saturated, flailing, or if either forearm responds to a finger target.

- [ ] **Step 3: Run the complete 12-second physics demonstration**

```bash
PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 python3 -m mm_sonic.manual_demo \
  --mode script --chunks 30 \
  --output-root /home/ubuntu/mm-sonic-closed-hand-runs

RUN_ROOT=$(find /home/ubuntu/mm-sonic-closed-hand-runs/manual-sonic \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | \
  sort -n | tail -n 1 | cut -d' ' -f2-)
test -n "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > /home/ubuntu/g1-sonic-closed-hand-run-root.txt
```

Expected: 600 scored state rows over 12 seconds, 681 archived body frames, explicit hand fields in all 35 pose publications, no fall marker, and no surviving MM/GEAR/simulator process after exit.

- [ ] **Step 4: Render and inspect the full video before broad tests**

```bash
RUN_ROOT=$(head -n 1 /home/ubuntu/g1-sonic-closed-hand-run-root.txt)
test -d "$RUN_ROOT"
PYTHONPATH=sonic/python MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
python3 -m mm_sonic.manual_replay \
  --run-root "$RUN_ROOT" \
  --output-dir /home/ubuntu/g1-sonic-closed-hand-visual
```

Inspect `g1-sonic-hand-closeup.png` and representative montage frames. Required visual verdict: two relaxed substantially symmetric fists, correct arm motion, planted/upright walking, and no flailing or limit pinning.

- [ ] **Step 5: Run protected and project-local formal gates**

The controller-owned evaluator must load the actual run-local model, check all 43 `actuator_trnid` mappings including actuator slots 22 and 29, decode every archived pose, recompute the neutral profile hash, compute final-second 14-joint errors, verify official GEAR Git cleanliness at `60de0df7ffedeef415fe58d435e92cc5b01ba3d9`, and invoke the v3 bundle auditor.

Run:

```bash
RUN_ROOT=$(head -n 1 /home/ubuntu/g1-sonic-closed-hand-run-root.txt)
test -d "$RUN_ROOT"
PYTHONDONTWRITEBYTECODE=1 python3 \
  /home/ubuntu/reliable-claude-projects/motion-matching/evaluators/evaluate_g1_neutral_closed_hands.py \
  --repo /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline \
  --run-root "$RUN_ROOT"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.python.test_sonic_hands \
  tests.python.test_sonic_zmq_v1 \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_manual_evidence \
  tests.python.test_sonic_manual_replay -v

git diff --check
```

Expected: protected evaluator exits 0, focused suites pass, and `git diff --check` is silent.

- [ ] **Step 6: Publish accessible evidence and commit the renderer**

Copy the final artifacts to stable easy paths:

```bash
cp /home/ubuntu/g1-sonic-closed-hand-visual/g1-sonic-rolling-replay.mp4 \
  /home/ubuntu/g1-sonic-closed-hands.mp4
cp /home/ubuntu/g1-sonic-closed-hand-visual/g1-sonic-hand-closeup.png \
  /home/ubuntu/g1-sonic-closed-hands.png
```

Then commit only the reviewed renderer/test changes:

```bash
git add sonic/python/mm_sonic/manual_replay.py tests/python/test_sonic_manual_replay.py
git commit -m "feat: render G1 hand close-up evidence"
```
