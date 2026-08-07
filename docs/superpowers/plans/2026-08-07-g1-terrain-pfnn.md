# Native G1 Terrain PFNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and run a native 29-DoF Unitree G1 PFNN that accepts continuous W/A/S/D intent and terrain-height samples, then produces raw idle/walking motion over flat ground and 5--20 degree hills without MotionBricks, portals, IK, or foot locks.

**Architecture:** A four-bank cubic phase function selects the weights of a `288 -> 512 -> 512 -> 268` PyTorch regressor each frame. An offline NumPy/MuJoCo/USD pipeline converts paired GRAIL slopes and flat G1 LAFAN motion into provenance-preserving 30 Hz shards with exact contacts and gait phase; a closed-loop runtime blends user trajectory intent with the prior prediction, samples terrain, evaluates the PFNN, and renders native G1 qpos.

**Tech Stack:** Python 3.10+, NumPy, SciPy, PyTorch 2.13, MuJoCo 3.11, USD Python bindings, joblib, `unittest`, NVIDIA L40S GPUs.

## Global Constraints

- Runtime and dataset frame rate is exactly `30 Hz`.
- Input contains exactly `288` float values; output contains exactly `268` float values.
- Phase is separate from the input vector and selects all network weights through a cyclic four-bank cubic Catmull-Rom function.
- The network is `288 -> ELU(512) -> ELU(512) -> 268`.
- Canonical joint order is `ISAACLAB_JOINT_NAMES`; conversion to MuJoCo qpos occurs only at rendering and FK boundaries.
- Canonical root quaternions are WXYZ; GRAIL's XYZW convention is converted once at import.
- Runtime terrain samples are center and `+/-0.25 m` lateral at 12 trajectory knots.
- First-milestone semantics are exactly idle and walk.
- Training slopes are restricted to measured traversed grades in `[5, 20]` degrees.
- GRAIL split identity is `slope_NNN`, never the individual `__NNN` variant.
- Test identities do not contribute normalization, fitting, early stopping, or checkpoint selection.
- Validation may select a checkpoint but never updates model parameters or normalization.
- Runtime contains no MotionBricks calls, portal playback, IK, stance locks, or root projection.
- Unsupported terrain slows/stops the desired trajectory before inference rather than evaluating outside the 20-degree envelope.
- Generated datasets, checkpoints, reports, and videos live below `sonic/runs/terrain-pfnn-v1/` and remain untracked.
- The author PFNN package at `/home/ubuntu/datasets/pfnn/pfnn` is read-only and no source code is copied from it.

---

### Task 1: Freeze the 288/268 layout and cubic PFNN model

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/__init__.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn/layout.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn/model.py`
- Create: `tests/python/test_terrain_pfnn_model.py`

**Interfaces:**
- Consumes: batched tensors `x: Tensor[B,288]` and `phase: Tensor[B]` in radians.
- Produces: `INPUT_LAYOUT`, `OUTPUT_LAYOUT`, `catmull_rom_phase_banks(banks, phase)`, and `PhaseFunctionedNetwork.forward(x, phase) -> Tensor[B,268]`.

- [ ] **Step 1: Write the failing layout and interpolation tests**

Create `tests/python/test_terrain_pfnn_model.py` with:

```python
import math
import unittest

import torch

from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.model import (
    PhaseFunctionedNetwork,
    catmull_rom_phase_banks,
)


class TerrainPFNNModelTest(unittest.TestCase):
    def test_layout_sizes_are_frozen(self) -> None:
        self.assertEqual(INPUT_LAYOUT.size, 288)
        self.assertEqual(OUTPUT_LAYOUT.size, 268)
        self.assertEqual(INPUT_LAYOUT["terrain_height"].stop - INPUT_LAYOUT["terrain_height"].start, 36)
        self.assertEqual(OUTPUT_LAYOUT["contact_logit"].stop - OUTPUT_LAYOUT["contact_logit"].start, 4)

    def test_phase_banks_hit_control_points_and_wrap(self) -> None:
        banks = torch.arange(4, dtype=torch.float64)[:, None]
        phase = torch.tensor(
            [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
            dtype=torch.float64,
        )
        actual = catmull_rom_phase_banks(banks, phase)[:, 0]
        torch.testing.assert_close(actual, torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0], dtype=torch.float64))

    def test_phase_interpolation_matches_equation_seven_scalar_reference(self) -> None:
        banks = torch.tensor([[-0.4], [0.2], [0.9], [0.1]], dtype=torch.float64)
        amount = 0.37
        phase = torch.tensor([(1.0 + amount) * 0.5 * math.pi], dtype=torch.float64)
        p0, p1, p2, p3 = (value.item() for value in banks)
        expected = (
            p1
            + 0.5 * amount * (p2 - p0)
            + amount**2 * (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3)
            + amount**3 * (1.5 * p1 - 1.5 * p2 + 0.5 * p3 - 0.5 * p0)
        )
        torch.testing.assert_close(
            catmull_rom_phase_banks(banks, phase)[0, 0],
            torch.tensor(expected, dtype=torch.float64),
        )

    def test_phase_wrap_has_matching_value_and_derivative(self) -> None:
        banks = torch.tensor(
            [[0.0, 0.1, -0.1], [0.2, 0.0, 0.05], [0.3, -0.1, 0.1], [0.1, 0.05, 0.0]],
            dtype=torch.float64,
        )
        epsilon = 1.0e-5
        left = catmull_rom_phase_banks(banks, torch.tensor([2.0 * math.pi - epsilon], dtype=torch.float64))
        zero = catmull_rom_phase_banks(banks, torch.tensor([0.0], dtype=torch.float64))
        right = catmull_rom_phase_banks(banks, torch.tensor([epsilon], dtype=torch.float64))
        torch.testing.assert_close(left, zero, atol=2.0e-5, rtol=0.0)
        torch.testing.assert_close(right, zero, atol=2.0e-5, rtol=0.0)
        torch.testing.assert_close((zero - left) / epsilon, (right - zero) / epsilon, atol=2.0e-4, rtol=0.0)

    def test_network_contract_and_gradient(self) -> None:
        model = PhaseFunctionedNetwork()
        x = torch.randn(5, 288, requires_grad=True)
        phase = torch.linspace(0.0, 2.0 * math.pi, 5)
        output = model(x, phase)
        self.assertEqual(tuple(output.shape), (5, 268))
        output.square().mean().backward()
        self.assertIsNotNone(model.W1.grad)
        self.assertTrue(torch.isfinite(model.W1.grad).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_model
```

Expected: import failure because `mm_sonic.terrain_pfnn` does not exist.

- [ ] **Step 3: Implement the frozen layouts**

Create `layout.py` with this public contract:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VectorLayout:
    fields: tuple[tuple[str, int], ...]

    @property
    def size(self) -> int:
        return sum(width for _, width in self.fields)

    def __getitem__(self, name: str) -> slice:
        offset = 0
        for field, width in self.fields:
            if field == name:
                return slice(offset, offset + width)
            offset += width
        raise KeyError(name)


INPUT_LAYOUT = VectorLayout((
    ("trajectory_position", 24),
    ("trajectory_direction", 24),
    ("terrain_height", 36),
    ("semantic_intent", 24),
    ("previous_body_position", 90),
    ("previous_body_velocity", 90),
))

OUTPUT_LAYOUT = VectorLayout((
    ("trajectory_position", 24),
    ("trajectory_direction", 24),
    ("body_position", 90),
    ("body_velocity", 90),
    ("root_height", 1),
    ("root_tilt", 2),
    ("joint_position", 29),
    ("root_planar_velocity", 2),
    ("root_yaw_velocity", 1),
    ("phase_advance", 1),
    ("contact_logit", 4),
))

assert INPUT_LAYOUT.size == 288
assert OUTPUT_LAYOUT.size == 268

TRAJECTORY_TIMES_S = np.array(
    [-1.0, -5/6, -2/3, -1/2, -1/3, -1/6, 0.0, 1/6, 1/3, 1/2, 2/3, 5/6],
    dtype=np.float64,
)
CONTACT_ORDER = ("left_heel", "left_toe", "right_heel", "right_toe")
```

Import NumPy, validate the time vector shape, and export both layouts plus
`TRAJECTORY_TIMES_S` and `CONTACT_ORDER` from `terrain_pfnn/__init__.py`.

- [ ] **Step 4: Implement the cubic phase-functioned network**

Create `model.py` with:

```python
from __future__ import annotations

import math

import torch
from torch import nn

from .layout import INPUT_LAYOUT, OUTPUT_LAYOUT


def catmull_rom_phase_banks(banks: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    if banks.ndim < 2 or banks.shape[0] != 4:
        raise ValueError("phase banks must have shape [4,...]")
    if phase.ndim != 1:
        raise ValueError("phase must have shape [B]")
    coordinate = torch.remainder(phase, 2.0 * math.pi) * (4.0 / (2.0 * math.pi))
    k1 = torch.floor(coordinate).to(torch.long) % 4
    amount = coordinate - torch.floor(coordinate)
    k0, k2, k3 = (k1 - 1) % 4, (k1 + 1) % 4, (k1 + 2) % 4
    p0, p1, p2, p3 = banks[k0], banks[k1], banks[k2], banks[k3]
    shape = (len(phase),) + (1,) * (banks.ndim - 1)
    w = amount.reshape(shape)
    return (
        p1
        + 0.5 * w * (p2 - p0)
        + w.square() * (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3)
        + w.pow(3) * (1.5 * p1 - 1.5 * p2 + 0.5 * p3 - 0.5 * p0)
    )


def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias


class PhaseFunctionedNetwork(nn.Module):
    def __init__(self, hidden_size: int = 512, dropout_probability: float = 0.30) -> None:
        super().__init__()
        self.W0 = nn.Parameter(torch.empty(4, hidden_size, INPUT_LAYOUT.size))
        self.b0 = nn.Parameter(torch.zeros(4, hidden_size))
        self.W1 = nn.Parameter(torch.empty(4, hidden_size, hidden_size))
        self.b1 = nn.Parameter(torch.zeros(4, hidden_size))
        self.W2 = nn.Parameter(torch.empty(4, OUTPUT_LAYOUT.size, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(4, OUTPUT_LAYOUT.size))
        self.dropout = nn.Dropout(dropout_probability)
        self.activation = nn.ELU()
        for weight in (self.W0, self.W1, self.W2):
            for bank in weight:
                nn.init.xavier_uniform_(bank)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != INPUT_LAYOUT.size or phase.shape != (len(x),):
            raise ValueError("PFNN expects x[B,288] and phase[B]")
        h0 = self.activation(_batched_linear(x, catmull_rom_phase_banks(self.W0, phase), catmull_rom_phase_banks(self.b0, phase)))
        h1 = self.activation(_batched_linear(self.dropout(h0), catmull_rom_phase_banks(self.W1, phase), catmull_rom_phase_banks(self.b1, phase)))
        return _batched_linear(self.dropout(h1), catmull_rom_phase_banks(self.W2, phase), catmull_rom_phase_banks(self.b2, phase))
```

- [ ] **Step 5: Run the model tests and commit**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_model
```

Expected: five tests pass.

Commit:

```bash
git add sonic/python/mm_sonic/terrain_pfnn tests/python/test_terrain_pfnn_model.py
git commit -m "feat: add native G1 phase-functioned network"
```

---

### Task 2: Convert GRAIL and LAFAN into one 30 Hz G1 clip contract

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/sources.py`
- Create: `tests/python/test_terrain_pfnn_sources.py`

**Interfaces:**
- Consumes: released GRAIL robot PKL/USD pairs and 36-column LAFAN G1 CSVs.
- Produces: `PFNNSourceClip`, `discover_grail_slope_records(root)`, `load_grail_source(record, fk)`, and `load_lafan_source(path, fk)` in canonical WXYZ/IsaacLab order at 30 Hz.

- [ ] **Step 1: Write source-contract tests with synthetic files**

The test creates a four-frame GRAIL payload and a four-row LAFAN CSV, then asserts:

```python
class TerrainPFNNSourcesTest(unittest.TestCase):
    def test_grail_conversion_is_30hz_wxyz_and_canonical_joints(self) -> None:
        source = load_grail_source(self.grail_record, self.fake_fk)
        self.assertEqual(source.fps, 30.0)
        self.assertEqual(source.root_quaternion_world_wxyz.shape[1:], (4,))
        self.assertEqual(source.joint_position.shape[1:], (29,))
        self.assertEqual(source.body_position_world.shape[1:], (30, 3))
        np.testing.assert_allclose(np.linalg.norm(source.root_quaternion_world_wxyz, axis=1), 1.0, atol=1.0e-6)

    def test_lafan_30hz_rows_are_not_temporally_resampled(self) -> None:
        source = load_lafan_source(self.lafan_csv, self.fake_fk)
        self.assertEqual(source.frame_count, 4)
        np.testing.assert_allclose(source.root_position_world[:, 0], np.arange(4))

    def test_discovery_pairs_every_robot_with_same_stem_usd(self) -> None:
        records = discover_grail_slope_records(self.grail_root)
        self.assertEqual([record.stem for record in records], ["terrain_slopes__slope_113__000"])
```

Use a fake FK result with 30 named bodies so these tests do not require MuJoCo.

- [ ] **Step 2: Run the source tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_sources
```

Expected: import failure for `terrain_pfnn.sources`.

- [ ] **Step 3: Implement the source clip and strict discovery**

Create this public dataclass and validations:

```python
@dataclass(frozen=True)
class PFNNSourceClip:
    clip_id: str
    terrain_id: str
    fps: float
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray
    root_linear_velocity_world: np.ndarray
    root_angular_velocity_world: np.ndarray
    joint_velocity: np.ndarray
    terrain_path: Path | None
    terrain_position_world: np.ndarray
    terrain_quaternion_world_from_usd_wxyz: np.ndarray
    motion_sha256: str
    terrain_sha256: str | None
    source_license_id: str

    @property
    def frame_count(self) -> int:
        return len(self.root_position_world)
```

`discover_grail_slope_records()` must glob `data/slope/robot/*.pkl`, require a same-stem file in `data/slope/object_usd`, parse `slope_NNN` as `terrain_id`, reject duplicate stems, and return sorted records with the documented fixed `-90 degree` terrain yaw.

- [ ] **Step 4: Implement both converters**

Use existing reviewed primitives rather than duplicating convention math:

```python
raw = load_grail_motion(record.robot_path, expected_frames=250)
resampled = resample_grail_motion(
    raw.root_position,
    raw.root_quaternion_xyzw,
    raw.dof_mujoco,
    source_fps=25.0,
    target_fps=30.0,
)
forward = fk.forward(
    resampled.root_position,
    resampled.root_quaternion_xyzw,
    resampled.dof_mujoco,
)
root_wxyz = unroll_quaternions_wxyz(
    resampled.root_quaternion_xyzw[:, (3, 0, 1, 2)]
)
joints = mujoco_to_isaaclab_joints(resampled.dof_mujoco)
body_wxyz = unroll_quaternions_wxyz(
    forward.body_quaternion_world_xyzw[..., (3, 0, 1, 2)]
)
body_velocity = finite_difference(forward.body_position_world, 30.0)
```

Compute root/body angular velocity from consecutive unrolled WXYZ
quaternions using the shortest relative quaternion logarithm. Compute joint
velocity after joint-order conversion using one-sided endpoint and centered
interior differences. Validate every array against `frame_count`, including
`body_angular_velocity_world[T,30,3]`, before constructing the frozen clip.

For LAFAN, call the reviewed `_load_rows()`, retain its 30 Hz samples, convert rows `[:,(6,3,4,5)]` from XYZW to WXYZ, run FK using rows `[:,7:]` in MuJoCo order, and then convert joints to `ISAACLAB_JOINT_NAMES`.

- [ ] **Step 5: Run pure and one real-file canaries**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_sources
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.terrain_pfnn.sources \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1.xml \
  --inspect-one
```

Expected: tests pass; canary reports one finite 30 Hz GRAIL clip, its paired USD, and one finite LAFAN clip without writing an artifact.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/sources.py tests/python/test_terrain_pfnn_sources.py
git commit -m "feat: add 30 Hz PFNN G1 source adapters"
```

---

### Task 3: Reconstruct four contacts and continuous gait phase

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/phase.py`
- Create: `tests/python/test_terrain_pfnn_phase.py`

**Interfaces:**
- Consumes: 30 Hz ankle body transforms/velocities, exact G1 sole geometry, and a terrain surface query.
- Produces: `ContactPhaseTrack(contact: bool[T,4], confidence: float[T,4], phase: float[T], phase_advance: float[T], valid: bool[T])`, `reconstruct_heel_toe_contacts(...)`, and `phase_from_contacts(...)`.

- [ ] **Step 1: Write phase tests before geometry code**

Cover exact alternation, wrap, and rejection:

```python
def test_alternating_strikes_anchor_zero_pi_and_wrap(self) -> None:
    contact = np.zeros((91, 4), dtype=bool)
    contact[3:13, 0] = True
    contact[33:43, 2] = True
    contact[63:73, 0] = True
    track = phase_from_contacts(contact, fps=30.0)
    self.assertTrue(track.valid[3:64].all())
    self.assertAlmostEqual(track.phase[3], 0.0)
    self.assertAlmostEqual(track.phase[33], np.pi)
    self.assertAlmostEqual(track.phase[63], 0.0)
    self.assertTrue((track.phase_advance[3:63] >= 0.0).all())

def test_same_foot_twice_rejects_the_ambiguous_span(self) -> None:
    contact = contacts_with_strikes(("left", 3), ("left", 33), ("right", 63))
    track = phase_from_contacts(contact, fps=30.0)
    self.assertFalse(track.valid[3:33].any())
    self.assertTrue(track.valid[33:64].all())

def test_half_cycle_outside_point_two_to_one_second_is_rejected(self) -> None:
    contact = contacts_with_strikes(("left", 3), ("right", 8), ("left", 43))
    track = phase_from_contacts(contact, fps=30.0)
    self.assertFalse(track.valid[3:9].any())
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_phase
```

Expected: import failure for `terrain_pfnn.phase`.

- [ ] **Step 3: Implement deterministic strike-to-phase reconstruction**

Implement:

```python
@dataclass(frozen=True)
class ContactPhaseTrack:
    contact: np.ndarray
    confidence: np.ndarray
    phase: np.ndarray
    phase_advance: np.ndarray
    valid: np.ndarray


def phase_from_contacts(
    contact: object,
    *,
    confidence: object | None = None,
    fps: float,
) -> ContactPhaseTrack:
    channels = np.asarray(contact, dtype=bool)
    if channels.ndim != 2 or channels.shape[1] != 4 or fps <= 0.0:
        raise ValueError("phase reconstruction expects contact[T,4] and positive fps")
    foot = np.column_stack((channels[:, 0] | channels[:, 1], channels[:, 2] | channels[:, 3]))
    rising = foot & ~np.vstack((np.zeros((1, 2), dtype=bool), foot[:-1]))
    strikes = [(frame, side) for frame in range(len(foot)) for side in range(2) if rising[frame, side]]
    confidence_array = channels.astype(np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32)
    if confidence_array.shape != channels.shape or not np.isfinite(confidence_array).all():
        raise ValueError("confidence must be finite with shape [T,4]")
    unwrapped = np.zeros(len(foot), dtype=np.float64)
    valid = np.zeros(len(foot), dtype=bool)
    previous_stop_phase: float | None = None
    previous_stop_frame: int | None = None
    for (start, side), (stop, next_side) in zip(strikes[:-1], strikes[1:]):
        duration = (stop - start) / float(fps)
        if next_side == side or not 0.20 <= duration <= 1.00:
            previous_stop_phase = None
            previous_stop_frame = None
            continue
        base_phase = 0.0 if side == 0 else math.pi
        if previous_stop_frame == start and previous_stop_phase is not None:
            start_phase = previous_stop_phase
        else:
            start_phase = base_phase
        stop_phase = start_phase + math.pi
        unwrapped[start : stop + 1] = np.linspace(start_phase, stop_phase, stop - start + 1)
        valid[start : stop + 1] = True
        previous_stop_phase = stop_phase
        previous_stop_frame = stop
    wrapped = np.remainder(unwrapped, 2.0 * math.pi)
    advance = np.zeros(len(foot), dtype=np.float64)
    advance[:-1] = np.where(valid[:-1] & valid[1:], np.maximum(0.0, np.diff(unwrapped)), 0.0)
    return ContactPhaseTrack(channels, confidence_array, wrapped.astype(np.float32), advance.astype(np.float32), valid)
```

Preserve an unwrapped temporary per alternating run so the `2*pi -> 0` wrap
does not create a negative advance. Extend genuine stationary margins from the
nearest valid cycle with zero advance; do not bridge an internally rejected
walking span. Add tests for both behaviors. Idle-window augmentation in Task 4
assigns deterministic phase bins so an idle pose is not tied to one phase.

- [ ] **Step 4: Add heel/toe geometry reconstruction**

Use `SoleGeometry.from_model(model)` and its ordered heel/toe probe groups. For each channel apply the approved hysteresis thresholds (`0.012/0.025 m`, `0.12/0.20 m/s`, `0.75/1.25 rad/s`, normal Z `0.50`). Return separate heel and toe booleans/confidences; never duplicate one full-foot flag into both channels.

Require at least three consecutive non-contact frames before accepting a
rising edge as a strike. Reject simultaneous left/right rises as ambiguous and
record both conditions in the audit reason counts.

Add a native-model test that places one foot on a plane, lifts the toe probes by `0.04 m`, and asserts heel contact true/toe false.

- [ ] **Step 5: Run tests and a real slope phase audit**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_phase
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.terrain_pfnn.phase \
  --robot /home/ubuntu/datasets/GRAIL/data/slope/robot/terrain_slopes__slope_113__004.pkl \
  --terrain /home/ubuntu/datasets/GRAIL/data/slope/object_usd/terrain_slopes__slope_113__004.usd \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1.xml
```

Expected: tests pass and audit JSON reports finite alternating phase coverage plus every rejected span reason.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/phase.py tests/python/test_terrain_pfnn_phase.py
git commit -m "feat: reconstruct PFNN contacts and gait phase"
```

---

### Task 4: Build exact 288/268 training windows and sealed splits

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/features.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn/splits.py`
- Create: `tests/python/test_terrain_pfnn_features.py`

**Interfaces:**
- Consumes: `PFNNSourceClip`, `ContactPhaseTrack`, and `height_at(xy: ndarray[...,2]) -> ndarray[...]`.
- Produces: `PFNNTrainingWindow(x, y, phase, clip_id, split_identity, center_frame, motion_sha256, terrain_sha256)`, `build_clip_windows(...)`, `mirror_window(...)`, and deterministic `split_identity(name) -> Literal["train","validation","test"]`.

- [ ] **Step 1: Write transform, terrain-track, and split tests**

Use one synthetic straight clip over `z = 0.1*x` and assert:

```python
windows = build_clip_windows(clip, phase_track, height_at=slope.height_at)
sample = windows[len(windows) // 2]
self.assertEqual(sample.x.shape, (288,))
self.assertEqual(sample.y.shape, (268,))
terrain = sample.x[INPUT_LAYOUT["terrain_height"]].reshape(12, 3)
self.assertTrue(np.isfinite(terrain).all())
self.assertGreater(terrain[-1, 1], terrain[0, 1])
```

Also assert all names `terrain_slopes__slope_113__000` through `__009` return the same split, while every identity returns exactly one split.
Add a native 29-joint symmetry-map fixture and assert
`mirror_window(mirror_window(sample))` recovers every continuous value,
semantic bit, contact channel, and phase convention within `1e-6`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_features
```

Expected: imports fail for `features` and `splits`.

- [ ] **Step 3: Implement stable split identities**

Create `splits.py` with SHA-256 buckets:

```python
def terrain_identity(clip_id: str) -> str:
    match = re.search(r"slope_(\d+)", clip_id)
    if match is None:
        return clip_id.split("__", 1)[0]
    return f"slope_{int(match.group(1)):03d}"


def split_identity(name: str) -> str:
    identity = terrain_identity(name)
    bucket = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"
```

LAFAN uses its base CSV stem as the identity. Sort the 12 stems by their
SHA-256 digest once and commit the resulting explicit 8/2/2 assignment so
overlapping windows from a sequence can never cross splits.

- [ ] **Step 4: Implement local frames and trajectory sampling**

Use the exact time offsets from the spec and linear interpolation for root position, facing unit vectors, and body state. Define the local transform with root yaw only and center support height as origin. Terrain values are ordered `(left, center, right)` at each knot and relative to current center support.

Define `center_frame=t` as the current recurrent state: the input contains
body state and trajectory at `t`, while the target contains pose, body state,
trajectory, root motion, contacts, and phase advance for `t+1`. Reject when
either recurrent frame or any trajectory knot escapes the clip, required phase
is invalid, a terrain ray is missing, or a packed value is non-finite.

Accept a GRAIL terrain family only when its measured traversed non-flat flank
grade is in `[5,20]` degrees. Do not reject its individual flat run-up,
transition, summit, or landing windows merely because local grade is below 5
degrees. Classify every accepted window as flat, ascent, descent, or transition
for stratified sampling.

- [ ] **Step 5: Pack targets and verify FK/joint order**

Pack target fields exactly in `OUTPUT_LAYOUT` order. Root tilt is the X/Y
exponential-map component after factoring root yaw from WXYZ. Joint targets
remain `ISAACLAB_JOINT_NAMES`. Root planar and yaw velocities are local
per-second values; phase advance is radians per 30 Hz tick. The
`contact_logit` slice names the network interpretation, but supervised targets
in that slice are binary `0/1` contact labels for
`binary_cross_entropy_with_logits`; never logit-transform the labels.

Implement the committed G1 left/right body and joint permutation plus the
required lateral, roll, yaw, quaternion, contact, and phase transforms.
Deterministically augment stationary flat windows across eight equally spaced
phase bins with zero phase advance.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_features
```

Expected: all feature, transform, and split tests pass.

Commit:

```bash
git add sonic/python/mm_sonic/terrain_pfnn/features.py sonic/python/mm_sonic/terrain_pfnn/splits.py tests/python/test_terrain_pfnn_features.py
git commit -m "feat: build terrain PFNN training windows"
```

---

### Task 5: Write provenance-preserving NPZ shards and training-only normalization

**Files:**
- Create: `sonic/python/mm_sonic/build_terrain_pfnn_dataset.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn/dataset.py`
- Create: `tests/python/test_terrain_pfnn_dataset.py`

**Interfaces:**
- Consumes: source roots, G1 MJCF, optional terrain-family limit, and deterministic split policy.
- Produces: untracked `manifest.json`, `normalization.npz`, and
  `split/shard_NNNNN.npz` files containing `x[N,288]`, `y[N,268]`,
  `phase[N]`, `clip_id[N]`, `split_identity[N]`, `terrain_class[N]`,
  `center_frame[N]`, and provenance hashes.

- [ ] **Step 1: Write an artifact round-trip and leakage test**

The test builds three synthetic identities into a temporary directory, loads them with `PFNNShardDataset`, and asserts all shapes. It then changes test values by `1e6` and verifies `normalization.npz` remains byte-identical, proving test data cannot affect statistics.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_dataset
```

Expected: missing dataset modules.

- [ ] **Step 3: Implement bounded shard writing**

Accumulate no more than `50_000` windows before writing one compressed NPZ. Write to a sibling temporary filename, reopen and validate shapes/finiteness, then atomically rename. Refuse to overwrite a nonempty destination unless `--resume` finds an identical source hash in its manifest.

The manifest must serialize this exact schema, using the complete constants
from `layout.py` rather than incomplete handwritten arrays:

```python
manifest = {
    "schema": "mm-sonic-terrain-pfnn-dataset/v1",
    "fps": 30.0,
    "input_size": INPUT_LAYOUT.size,
    "output_size": OUTPUT_LAYOUT.size,
    "joint_order": list(ISAACLAB_JOINT_NAMES),
    "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
    "contact_order": list(CONTACT_ORDER),
    "source_roots": source_roots_with_license_and_manifest_hashes,
    "source_records": clip_id_to_motion_and_terrain_hashes,
    "split_identities": {"train": train_ids, "validation": validation_ids, "test": test_ids},
    "shards": shard_records_with_sha256_count_and_split,
    "rejections": rejection_counts_by_reason,
}
```

Each shard also stores `terrain_class[N]`, `split_identity[N]`,
`motion_sha256[N]`, and `terrain_sha256[N]`; LAFAN's absent terrain hash is the
empty string. Validate that `terrain_class` is one of `flat`, `ascent`,
`descent`, or `transition` so Task 6 can stratify without reopening source
files.

- [ ] **Step 4: Compute training-only normalization**

Use streaming float64 sums and squared sums from training shards only. Save
`x_mean`, `x_std`, `y_mean`, `y_std`; replace continuous standard deviations
below `1e-6` with `1.0`. Force `y_mean=0` and `y_std=1` for the four contact
slots so their binary targets remain binary and model outputs remain logits.
Apply the approved `0.1` scale to normalized previous-body
position/velocity input slices in both dataset loading and runtime packing.
Store training sample count, per-split counts, and split-identity digest beside
the arrays.

- [ ] **Step 5: Add the real dataset CLI and build a two-family canary**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1.xml \
  --output sonic/runs/terrain-pfnn-v1/canary-dataset \
  --terrain-family-limit 2
```

Expected: manifest status `accepted`, at least one shard, exact 288/268 shapes, no identity in multiple splits, and explicit rejection counts.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_dataset
git add sonic/python/mm_sonic/build_terrain_pfnn_dataset.py sonic/python/mm_sonic/terrain_pfnn/dataset.py tests/python/test_terrain_pfnn_dataset.py
git commit -m "feat: build sealed terrain PFNN datasets"
```

---

### Task 6: Train, checkpoint, and evaluate the PFNN

**Files:**
- Create: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Create: `sonic/python/mm_sonic/evaluate_terrain_pfnn.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn/kinematics.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Create: `tests/python/test_terrain_pfnn_kinematics.py`
- Create: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Consumes: dataset manifest/shards and normalization.
- Produces: checkpoint schema `mm-sonic-terrain-pfnn-checkpoint/v1`, per-loss JSONL, validation reports, and deterministic one-step/closed-loop evaluation results.

- [ ] **Step 1: Write loss and checkpoint tests**

Tests must verify:

```python
losses = pfnn_losses(prediction, target, model=model, normalization=normalization)
self.assertEqual(
    set(losses),
    {
        "trajectory_mse", "body_mse", "root_pose_mse", "joint_mse",
        "root_motion_mse", "phase_mse", "contact_bce",
        "trajectory_direction", "phase_nonnegative", "fk_consistency",
        "regularization", "total",
    },
)
self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

save_checkpoint(
    path, model, optimizer, normalization,
    dataset_digest="abc", kinematic_signature_sha256="def",
    runtime_seed=finite_runtime_seed(), step=17,
)
loaded = load_checkpoint(
    path, expected_dataset_digest="abc", expected_kinematic_signature_sha256="def",
)
self.assertEqual(loaded.step, 17)
with self.assertRaises(ValueError):
    load_checkpoint(
        path, expected_dataset_digest="different",
        expected_kinematic_signature_sha256="def",
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_training
```

Expected: missing training module.

- [ ] **Step 3: Implement and verify differentiable native G1 FK**

`TorchG1ForwardKinematics.from_mjcf(path)` extracts the fixed body transforms,
hinge axes, parent indices, canonical body names, joint-name permutation, and
joint limits from the native G1 MuJoCo model. Its differentiable forward pass
accepts local root height/roll/pitch and IsaacLab-order joint tensors and
returns all 30 local body positions.

Define `kinematic_signature_sha256` from canonical serialized body names,
parent indices, fixed transforms, joint names/axes/limits, and actuator-free
kinematic constants. This signature—not raw XML bytes—must match between the
standalone G1 model, the compiled viewer scene, and the checkpoint.

In `test_terrain_pfnn_kinematics.py`, generate eight deterministic in-limit
poses, evaluate the torch and existing MuJoCo FK paths from the same root
transform, reorder by body name, and require maximum position error below
`1e-5 m`. Backpropagate the mean squared body position and require finite
joint gradients. A body-name, joint-name, or model-hash mismatch is a hard
error rather than a guessed mapping.

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_kinematics
```

- [ ] **Step 4: Implement grouped losses and optimization**

Split continuous regression into separately reported trajectory, body,
root-pose, joint, root-motion, and phase MSE groups. The four contact outputs
are never denormalized for loss: they feed
`binary_cross_entropy_with_logits` against binary labels. Trajectory direction
penalizes `(|d|-1)^2`; phase penalizes `relu(-phase_advance)^2`; FK consistency
denormalizes predicted pose/body fields and compares predicted body positions
with `TorchG1ForwardKinematics`; regularization is
`0.01 * mean(abs(parameter))` across `W0,b0,W1,b1,W2,b2`.

Use Adam, batch size 32 per GPU, dropout probability 0.30, gradient norm cap `1.0`, fixed seeds, and stratified split sampling. Write each loss group, learning rate, samples, wall time, and validation score to JSONL.

After the one-step gate passes, implement the optional
`--rollout-finetune-frames N` stage. Group consecutive training windows by
`clip_id` and `center_frame`, unroll up to `N` recurrent steps, feed predicted
body/trajectory state back while retaining only training-split exogenous
intent and terrain samples, and backpropagate the mean grouped loss through
the unroll. A three-step synthetic test must prove the second input comes from
the first prediction and that gradients reach the first step. Full training
uses `N=16`; validation and test remain read-only.

- [ ] **Step 5: Implement immutable checkpoint selection**

Checkpoint payload contains model/optimizer state, layouts, normalization
arrays, code commit, dataset digest, train/validation identities, step, epoch,
seed, loss weights, G1 kinematic signature, canonical limits, and
`phase_advance_q99` computed only from training targets. It also contains a
finite `runtime_seed` chosen deterministically from the lowest-speed valid flat
training window: phase, qpos fields, local body state, and trajectory. This is
the sole initial state used by the standalone viewer.

`evaluate_terrain_pfnn.py` rejects a checkpoint whose dataset digest,
input/output size, joint/body order, G1 kinematic signature, normalization contract, or
30 Hz contract differs.

Validation score combines one-step normalized error and closed-loop failures. Test split evaluation requires `--sealed-test` and writes a receipt that prevents a second test selection pass into the same run directory.

- [ ] **Step 6: Pass a synthetic overfit test**

Train a 32-hidden-unit model on 64 deterministic samples for 300 steps. Expected: final total loss is below 10% of its initial value and saved/reloaded predictions match bitwise on CPU.

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_training
```

- [ ] **Step 7: Run the real pipeline-overfit gate**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.train_terrain_pfnn \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1.xml \
  --output sonic/runs/terrain-pfnn-v1/pipeline-overfit \
  --overfit-samples 256 \
  --steps 4000 \
  --seed 7
```

Expected: training loss decreases, checkpoint is finite, and one-step report marks the fixed subset accepted. Closed-loop acceptance is completed after Task 7 supplies the runtime.

- [ ] **Step 8: Commit**

```bash
git add sonic/python/mm_sonic/train_terrain_pfnn.py sonic/python/mm_sonic/evaluate_terrain_pfnn.py sonic/python/mm_sonic/terrain_pfnn/kinematics.py sonic/python/mm_sonic/terrain_pfnn/training.py tests/python/test_terrain_pfnn_kinematics.py tests/python/test_terrain_pfnn_training.py
git commit -m "feat: train and checkpoint terrain PFNN"
```

---

### Task 7: Add the closed-loop trajectory and G1 runtime

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Create: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Consumes: checkpoint, matching native G1 MJCF, `height_and_grade_at(xy)`, camera-relative desired velocity, prior runtime state.
- Produces: `TerrainPFNNRuntime.step(command, camera_yaw) -> PFNNRuntimeFrame` containing finite canonical qpos, phase, contacts, trajectory, envelope state, and diagnostics.

- [ ] **Step 1: Write feedback and envelope tests with a deterministic fake model**

Test that the second frame input contains the first prediction, not
teacher-forced state. Test that a 21-degree future sample yields
`supported=False`, replans the offending future knots to the last supported
point, and reaches model evaluation with zero desired speed and no unsupported
terrain feature. Test that a 19-degree sample calls the model and advances
phase modulo `2*pi`. Test that construction from a checkpoint uses its
training-only `runtime_seed` exactly.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_runtime
```

Expected: missing runtime module.

- [ ] **Step 3: Implement trajectory state and paper-style blending**

Define:

```python
@dataclass
class PFNNTrajectoryState:
    position_world_xy: np.ndarray  # [12,2]
    direction_world_xy: np.ndarray  # [12,2]
    semantic_intent: np.ndarray  # [12,2]


@dataclass(frozen=True)
class PFNNRuntimeFrame:
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position_isaaclab: np.ndarray
    phase: float
    contact_probability: np.ndarray
    supported: bool
    diagnostics: dict[str, object]
```

Blend future velocity with `tau_v=0.5` and facing with `tau_d=2.0` using `(1-t**tau)*prediction + t**tau*intent`, renormalizing directions. Shift recorded past knots from realized root motion each tick.

Load and verify the matching compiled kinematic signature before the first tick. Derive all
canonical joint limits by name, initialize exclusively from the checkpoint's
`runtime_seed`, and fail construction if the seed is non-finite or outside the
limits.

- [ ] **Step 4: Implement one closed-loop PFNN tick**

Pack 288 inputs, apply checkpoint normalization and the `0.1` body-input
scale, evaluate with `torch.inference_mode()`, denormalize only continuous
output slices, and apply sigmoid directly to the four raw contact logits.
Integrate local planar/yaw velocities over `1/30`, query center support,
reconstruct WXYZ from yaw and predicted roll/pitch, and validate the 29
IsaacLab-order joint limits. Advance phase by
`clip(phase_advance, 0, min(pi, 1.5 * phase_advance_q99))` modulo `2*pi`.

Convert the predicted next trajectory from its local output frame back to
world state and retain predicted next body position/velocity for the following
tick. This must match the `t -> t+1` feature convention in Task 4.

Do not apply qpos correction. Invalid output returns the previous finite frame
with `diagnostics["hold_reason"]`. Unsupported future samples are never packed:
truncate/replan future knots to supported ground first; if no forward
continuation exists, infer a stationary intent at the current supported point.

- [ ] **Step 5: Add closed-loop acceptance metrics**

Implement a recorder for finite duration, phase reversal/freeze, root/joint
steps, sole penetration, forbidden-body penetration, stance sole speed, grade
segments, traversal direction, landing recovery, and command path error. Emit
JSON with the exact gates: 20 seconds finite; no reversal/freeze; root
translation `<=0.060 m`; root rotation `<=0.35 rad`; joint step `<=0.25 rad`;
no limit violation; sole penetration `<=0.015 m`; zero forbidden-body
penetration; median stance sole speed `<=0.10 m/s`; 18.9-degree traversal in
both directions; and responsive flat walking after each landing. Report each
metric for flat, ascent, summit, descent, and landing as well as globally.

Define reversal as an unwrapped phase delta below `-1e-6`. Define walking
freeze as any supported one-second interval with desired speed at least
`0.20 m/s` but cumulative phase advance below `0.25 rad`. Define responsive
landing recovery as realized speed reaching at least 50% of desired speed
within two seconds on the flat landing.

- [ ] **Step 6: Run tests and the pipeline-overfit rollout**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_runtime
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.evaluate_terrain_pfnn \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit/best.pt \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset/manifest.json \
  --split train \
  --closed-loop-seconds 20
```

Expected: tests pass; the overfit checkpoint completes a finite known-slope rollout. If it fails, fix the representation/runtime mismatch before building the full dataset.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/runtime.py tests/python/test_terrain_pfnn_runtime.py sonic/python/mm_sonic/evaluate_terrain_pfnn.py
git commit -m "feat: run terrain PFNN in closed loop"
```

---

### Task 8: Build the controllable three-hill MuJoCo viewer

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/hill_map.py`
- Create: `sonic/python/mm_sonic/terrain_pfnn_viewer.py`
- Create: `tests/python/test_terrain_pfnn_hill_map.py`
- Create: `tests/python/test_terrain_pfnn_viewer.py`

**Interfaces:**
- Consumes: accepted checkpoint and G1 scene XML.
- Produces: local passive MuJoCo W/A/S/D viewer and bounded `--no-viewer --smoke-steps` canary over the identical queried/rendered mesh.

- [ ] **Step 1: Test hill grades and mesh/query identity**

Create a map with 10, 15, and 18.9-degree symmetric C1 mound profiles. Assert sampled maximum grades within `0.05 degrees`, flat run-ups/landings, finite triangle arrays, and `height_at()` equal to downward mesh raycasts within `1e-4 m`.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_hill_map tests.python.test_terrain_pfnn_viewer
```

Expected: missing map/viewer modules.

- [ ] **Step 3: Implement the shared map mesh/query**

`TerrainPFNNHillMap` owns one sampled height grid, vertices, faces, analytic derivative, and exact triangle query. Both the runtime callback and MuJoCo mesh use this object; the viewer must not maintain a second height formula.

Build each mound from flat run-up, C1 entry blend, constant-grade flank,
rounded C1 summit, mirrored descent, and flat landing segments. The mesh
triangle interpolation is authoritative for both `height_at()` and runtime
grade; the analytic derivative is used only to validate construction.

- [ ] **Step 4: Implement keyboard runtime and diagnostics**

Load the checkpoint, verify the compiled scene's G1 kinematic signature, create
`TerrainPFNNRuntime`, and maintain pressed-key state using a
`pynput.keyboard.Listener` that is stopped in `finally`. Map W/A/S/D to
camera-relative velocity, and render canonical joint output after
`isaaclab_to_mujoco_joint_vector()`. Overlay/trace fields are phase, advance,
desired/realized speed, grade, root height/tilt, contacts, supported flag, and
hold reason.

CLI defaults:

```text
--checkpoint sonic/runs/terrain-pfnn-v1/full-training/best.pt
--scene-xml /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/scene_29dof.xml
--max-steps 1000000
--trace-every 30
```

`--no-viewer` requires positive `--smoke-steps`; its automatic command walks forward through all three hills and returns nonzero on a held/non-finite frame.

- [ ] **Step 5: Run tests and a headless canary**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_hill_map tests.python.test_terrain_pfnn_viewer
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.terrain_pfnn_viewer \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit/best.pt \
  --no-viewer --smoke-steps 600 --trace-every 30
```

Expected: tests pass; canary remains finite and prints no IK/lock/portal state. The overfit model is not expected to pass unseen-hill quality gates.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/hill_map.py sonic/python/mm_sonic/terrain_pfnn_viewer.py tests/python/test_terrain_pfnn_hill_map.py tests/python/test_terrain_pfnn_viewer.py
git commit -m "feat: add controllable terrain PFNN hill viewer"
```

---

### Task 9: Build the full corpus, train, and run sealed verification

**Files:**
- Modify: `docs/superpowers/specs/2026-08-07-g1-terrain-pfnn-design.md`
- Create: `docs/research/terrain-pfnn-v1-results.md`

**Interfaces:**
- Consumes: all committed PFNN pipeline code and local external datasets.
- Produces: untracked full dataset/checkpoint/evaluation artifacts plus a committed provenance-and-results report with no binary model data.

- [ ] **Step 1: Run the complete focused test suite before data construction**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_terrain_pfnn_kinematics \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime \
  tests.python.test_terrain_pfnn_hill_map \
  tests.python.test_terrain_pfnn_viewer
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_sources \
  tests.python.test_terrain_pfnn_phase \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_dataset
```

Expected: all focused tests pass in their dependency-specific environments.

- [ ] **Step 2: Build the full sealed dataset**

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1.xml \
  --output sonic/runs/terrain-pfnn-v1/full-dataset
```

Expected: accepted manifest, 1,880 paired GRAIL sources inventoried before grade/phase rejection, 12 LAFAN sources inventoried, no split leakage, and training-only normalization receipt.

- [ ] **Step 3: Train and select only on validation**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/torchrun --standalone --nproc_per_node=8 \
  -m mm_sonic.train_terrain_pfnn \
  --dataset sonic/runs/terrain-pfnn-v1/full-dataset/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1.xml \
  --output sonic/runs/terrain-pfnn-v1/full-training \
  --epochs 20 \
  --batch-size-per-gpu 32 \
  --rollout-finetune-frames 16 \
  --seed 7
```

Expected: a selected `best.pt`, finite per-group losses, and validation report. Do not run sealed test if pipeline-overfit or validation closed-loop gates fail.

- [ ] **Step 4: Run one sealed test evaluation and the three-hill canary**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.evaluate_terrain_pfnn \
  --checkpoint sonic/runs/terrain-pfnn-v1/full-training/best.pt \
  --dataset sonic/runs/terrain-pfnn-v1/full-dataset/manifest.json \
  --split test --sealed-test --closed-loop-seconds 20 \
  --output sonic/runs/terrain-pfnn-v1/sealed-test
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.terrain_pfnn_viewer \
  --checkpoint sonic/runs/terrain-pfnn-v1/full-training/best.pt \
  --no-viewer --smoke-steps 1200 --trace-every 30
```

Expected: metrics explicitly report pass/fail for every design threshold and every flat/ascent/summit/descent/landing segment.

- [ ] **Step 5: Perform interactive acceptance**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.terrain_pfnn_viewer \
  --checkpoint sonic/runs/terrain-pfnn-v1/full-training/best.pt
```

Verify W/A/S/D on flat, uphill, summit, downhill, direction changes on a flank, both crossing directions, and resumed flat response. Record observations without enabling IK or modifying the sealed checkpoint.

- [ ] **Step 6: Write the results report and update design status**

`docs/research/terrain-pfnn-v1-results.md` records commit, dataset/checkpoint digests, source/license manifests, counts/rejections, training curve summary, every sealed metric, canary command, interactive observations, and artifact paths. Set the design status to `Implemented` only if all required gates pass; otherwise set it to `Prototype evaluated` and list failed thresholds verbatim.

- [ ] **Step 7: Run final verification and commit only text evidence**

```bash
git diff --check
git status --short
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m compileall -q sonic/python/mm_sonic/terrain_pfnn sonic/python/mm_sonic/terrain_pfnn_viewer.py
```

Confirm no files below `sonic/runs/` are staged.

Commit:

```bash
git add docs/superpowers/specs/2026-08-07-g1-terrain-pfnn-design.md docs/research/terrain-pfnn-v1-results.md
git commit -m "docs: report native G1 terrain PFNN results"
```
