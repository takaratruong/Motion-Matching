# MotionBricks Authored-Contact Terrain IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace geometric stance inference with MotionBricks-authored contacts and add bounded display-root height so the 18-degree hill controller does not drag released feet or saturate leg IK.

**Architecture:** A focused contact adapter samples the four contact channels from the exact MotionBricks feature frame paired with each qpos. The qpos-native terrain post-process consumes collapsed left/right stance, advances falling edges unconditionally, adjusts only display root Z, and then solves bounded stance/swing leg constraints. The viewer renders the detached result without writing it into MotionBricks buffers.

**Tech Stack:** Python 3.10, NumPy, PyTorch tensors supplied by MotionBricks, MuJoCo 3.11, `unittest`, pinned MotionBricks G1 checkpoints.

## Global Constraints

- Contact order is left heel, left toe, right heel, right toe.
- MotionBricks contact extraction uses `is_normalized=False` and `contact_thresh=0.5`.
- A foot is in stance when either authored channel for that foot is true.
- Authored contact false clears that foot's world lock in the same frame.
- Missing or malformed authored contacts produce `(False, False)` stance and clear locks.
- Display root X/Y and quaternion remain bit-exact raw MotionBricks values.
- Display root-Z correction stays within `[-0.20 m, +0.20 m]`.
- Supported root correction changes by at most `0.025 m` per frame; airborne decay changes by at most `0.010 m`.
- Stance leg correction stays within `0.30 rad` and changes by at most `0.08 rad` per frame.
- Released correction decays toward raw by at most `0.12 rad` per frame.
- Swing clearance is upward-only with a `0.015 m` minimum.
- Display corrections never mutate `full_agent.frames` or inference/controller context.

---

### Task 1: Sample authored contacts from the displayed MotionBricks frame

**Files:**
- Create: `sonic/python/mm_sonic/motionbricks_authored_contacts.py`
- Create: `tests/python/test_motionbricks_authored_contacts.py`

**Interfaces:**
- Consumes: a MotionBricks `full_agent` and the integer frame index recorded before `get_next_frame()`.
- Produces: `AuthoredFootContacts`, `collapse_authored_contacts(values)`, and `sample_authored_contacts(full_agent, frame_index)`.

- [ ] **Step 1: Write the failing pure and adapter tests**

Create:

```python
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.motionbricks_authored_contacts import (
    AuthoredFootContacts,
    collapse_authored_contacts,
    sample_authored_contacts,
)


class AuthoredContactContractTest(unittest.TestCase):
    def test_collapses_heel_and_toe_by_foot(self) -> None:
        self.assertEqual(
            collapse_authored_contacts((False, True, False, False)),
            (True, False),
        )
        self.assertEqual(
            collapse_authored_contacts((False, False, True, False)),
            (False, True),
        )

    def test_rejects_malformed_channels(self) -> None:
        for value in ((True, False), (True, False, True, np.nan)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collapse_authored_contacts(value)

    def test_samples_exact_unnormalized_feature_frame(self) -> None:
        features = np.arange(1 * 3 * 7, dtype=np.float32).reshape(1, 3, 7)

        class MotionRep:
            def __init__(self) -> None:
                self.calls: list[tuple[np.ndarray, bool, float]] = []

            def extract_foot_contacts(
                self,
                values: object,
                *,
                is_normalized: bool,
                contact_thresh: float,
            ) -> np.ndarray:
                array = np.asarray(values)
                self.calls.append((array.copy(), is_normalized, contact_thresh))
                return np.asarray([[[False, True, False, False]]])

        motion_rep = MotionRep()
        agent = SimpleNamespace(
            frames={"model_features": features},
            _motion_rep=motion_rep,
        )

        result = sample_authored_contacts(agent, 1)

        self.assertEqual(
            result,
            AuthoredFootContacts(
                channels=(False, True, False, False),
                stance=(True, False),
                valid=True,
                reason="ok",
            ),
        )
        np.testing.assert_array_equal(
            motion_rep.calls[0][0], features[:, 1:2]
        )
        self.assertFalse(motion_rep.calls[0][1])
        self.assertEqual(motion_rep.calls[0][2], 0.5)

    def test_unavailable_contacts_fail_open_to_swing(self) -> None:
        agent = SimpleNamespace(frames={}, _motion_rep=None)

        result = sample_authored_contacts(agent, 0)

        self.assertFalse(result.valid)
        self.assertEqual(result.channels, (False, False, False, False))
        self.assertEqual(result.stance, (False, False))
        self.assertIn("model_features", result.reason)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_motionbricks_authored_contacts
```

Expected: import failure because `motionbricks_authored_contacts` does not exist.

- [ ] **Step 3: Implement the contact adapter**

Create:

```python
"""Authored MotionBricks contacts paired with displayed qpos frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuthoredFootContacts:
    channels: tuple[bool, bool, bool, bool]
    stance: tuple[bool, bool]
    valid: bool
    reason: str


def collapse_authored_contacts(values: object) -> tuple[bool, bool]:
    channels = np.asarray(values)
    if (
        channels.shape != (4,)
        or channels.dtype.kind not in "biuf"
        or not np.isfinite(channels.astype(np.float64)).all()
    ):
        raise ValueError("authored contacts must contain four finite channels")
    active = channels.astype(np.float64) > 0.5
    return (
        bool(active[0] or active[1]),
        bool(active[2] or active[3]),
    )


def _unavailable(reason: str) -> AuthoredFootContacts:
    return AuthoredFootContacts(
        channels=(False, False, False, False),
        stance=(False, False),
        valid=False,
        reason=reason,
    )


def _as_numpy(values: object) -> np.ndarray:
    detached = values.detach() if callable(getattr(values, "detach", None)) else values
    cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    return np.asarray(cpu)


def sample_authored_contacts(
    full_agent: object,
    frame_index: int,
) -> AuthoredFootContacts:
    try:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame index must be a nonnegative integer")
        frames = getattr(full_agent, "frames", None)
        if not isinstance(frames, dict) or "model_features" not in frames:
            raise ValueError("model_features are unavailable")
        features = frames["model_features"]
        if len(features.shape) != 3 or frame_index >= int(features.shape[1]):
            raise ValueError("contact frame index is outside model_features")
        motion_rep = getattr(full_agent, "_motion_rep", None)
        extractor = getattr(motion_rep, "extract_foot_contacts", None)
        if not callable(extractor):
            raise ValueError("MotionBricks contact extractor is unavailable")
        extracted = extractor(
            features[:, frame_index : frame_index + 1],
            is_normalized=False,
            contact_thresh=0.5,
        )
        array = _as_numpy(extracted)
        if array.shape != (1, 1, 4):
            raise ValueError("contact extractor returned an unexpected shape")
        channels = tuple(bool(value) for value in array[0, 0])
        stance = collapse_authored_contacts(channels)
        return AuthoredFootContacts(channels, stance, True, "ok")
    except Exception as error:
        return _unavailable(str(error))
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: four tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_authored_contacts.py \
  tests/python/test_motionbricks_authored_contacts.py
git commit -m "feat: sample MotionBricks authored foot contacts"
```

### Task 2: Add bounded display-root height state

**Files:**
- Modify: `sonic/python/mm_sonic/motionbricks_hill_ik.py`
- Modify: `tests/python/test_motionbricks_hill_ik.py`

**Interfaces:**
- Consumes: current root-height correction, residual required support shift, and whether authored support exists.
- Produces: `_bounded_root_height_correction(current_m, required_support_shift_m, has_support) -> float`, root-height state in `MotionBricksHillFootIK`, and root diagnostics.

- [ ] **Step 1: Write failing root-bound tests**

Add:

```python
from mm_sonic.motionbricks_hill_ik import _bounded_root_height_correction

def test_supported_root_height_ramps_and_clamps(self) -> None:
    value = 0.0
    values = []
    for _ in range(20):
        value = _bounded_root_height_correction(
            value,
            required_support_shift_m=0.15,
            has_support=True,
        )
        values.append(value)
    self.assertAlmostEqual(values[0], 0.025)
    self.assertTrue(
        all(abs(right - left) <= 0.025 + 1.0e-12
            for left, right in zip(values, values[1:]))
    )
    self.assertLessEqual(max(values), 0.20)

def test_airborne_root_height_decays_toward_raw(self) -> None:
    self.assertAlmostEqual(
        _bounded_root_height_correction(
            0.08,
            required_support_shift_m=0.0,
            has_support=False,
        ),
        0.07,
    )
    self.assertAlmostEqual(
        _bounded_root_height_correction(
            -0.08,
            required_support_shift_m=0.0,
            has_support=False,
        ),
        -0.07,
    )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_motionbricks_hill_ik.HillFootIKContractTest
```

Expected: import failure because `_bounded_root_height_correction` is absent.

- [ ] **Step 3: Implement the bounded root helper**

Add:

```python
def _bounded_root_height_correction(
    current_m: float,
    *,
    required_support_shift_m: float,
    has_support: bool,
) -> float:
    current = float(current_m)
    required = float(required_support_shift_m)
    if (
        not np.isfinite(current)
        or not np.isfinite(required)
        or type(has_support) is not bool
    ):
        raise ValueError("root-height inputs must be finite")
    if has_support:
        desired = current + required
        step = 0.025
    else:
        desired = 0.0
        step = 0.010
    rate_bounded = float(np.clip(desired, current - step, current + step))
    return float(np.clip(rate_bounded, -0.20, 0.20))
```

Extend `HillFootIKDiagnostics` with:

```python
authored_stance: tuple[bool, bool]
locked: tuple[bool, bool]
root_height_correction_m: float
```

Extend `reset()` with:

```python
self._previous_contact = (False, False)
self._root_height_correction_m = 0.0
```

Replace `snapshot_state()` with the exact detached state used by the release
regression tests:

```python
def snapshot_state(
    self,
) -> tuple[
    tuple[bool, bool],
    tuple[np.ndarray | None, np.ndarray | None],
    float,
    tuple[np.ndarray, np.ndarray],
]:
    targets = tuple(
        None if target is None else target.copy()
        for target in self._targets
    )
    corrections = tuple(value.copy() for value in self._corrections)
    return (
        self._previous_contact,
        targets,
        float(self._root_height_correction_m),
        corrections,
    )
```

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run the command from Step 2. Expected: all contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_ik.py \
  tests/python/test_motionbricks_hill_ik.py
git commit -m "feat: bound MotionBricks display root height"
```

### Task 3: Replace geometric phases with authored-contact target ownership

**Files:**
- Modify: `sonic/python/mm_sonic/motionbricks_hill_ik.py`
- Modify: `tests/python/test_motionbricks_hill_ik.py`

**Interfaces:**
- Consumes: `MotionBricksHillFootIK.apply(raw_qpos, authored_stance, dt_s)`.
- Produces: a detached qpos with display root Z and leg corrections, same-frame lock release, and `HillFootIKResult`.

- [ ] **Step 1: Replace the failing model-backed behavior tests**

Retain pure stance-target and swing-lift tests. Replace geometric phase tests
with:

```python
def test_authored_toe_off_clears_lock_in_same_frame(self) -> None:
    raw = self.model.qpos0.copy()
    solver = MotionBricksHillFootIK(self.model, lambda xy: 0.0)
    planted = solver.apply(raw, (True, False), 1.0 / 30.0)
    self.assertEqual(planted.diagnostics.locked, (True, False))

    shifted = raw.copy()
    shifted[0] += 0.03
    released = solver.apply(shifted, (False, False), 1.0 / 30.0)

    self.assertEqual(released.diagnostics.authored_stance, (False, False))
    self.assertEqual(released.diagnostics.locked, (False, False))
    snapshot = solver.snapshot_state()
    self.assertIsNone(snapshot[1][0])

def test_invalid_terrain_cannot_resurrect_released_lock(self) -> None:
    finite = [True]
    def height_query(xy: object) -> float:
        del xy
        return 0.0 if finite[0] else math.nan

    raw = self.model.qpos0.copy()
    solver = MotionBricksHillFootIK(self.model, height_query)
    solver.apply(raw, (True, False), 1.0 / 30.0)
    finite[0] = False

    result = solver.apply(raw, (False, False), 1.0 / 30.0)

    self.assertFalse(result.diagnostics.accepted)
    self.assertEqual(result.diagnostics.locked, (False, False))
    self.assertIsNone(solver.snapshot_state()[1][0])

def test_display_root_reduces_eighteen_degree_penetration(self) -> None:
    hill = GentleHillProfile()
    raw = self.model.qpos0.copy()
    raw[0] = 3.25
    raw[2] += hill.height(raw[:2])
    solver = MotionBricksHillFootIK(self.model, hill.height)

    results = [
        solver.apply(raw, (True, True), 1.0 / 30.0)
        for _ in range(8)
    ]
    final = results[-1]

    np.testing.assert_array_equal(final.qpos[:2], raw[:2])
    np.testing.assert_array_equal(final.qpos[3:7], raw[3:7])
    self.assertLessEqual(
        abs(final.diagnostics.root_height_correction_m), 0.20
    )
    self.assertLess(final.diagnostics.corrected_penetration_m, 0.010)
    self.assertLessEqual(
        final.diagnostics.maximum_joint_correction_rad, 0.30 + 1.0e-9
    )

def test_authored_swing_never_creates_a_world_lock(self) -> None:
    raw = self.model.qpos0.copy()
    solver = MotionBricksHillFootIK(self.model, lambda xy: 0.0)

    for _ in range(4):
        result = solver.apply(raw, (False, False), 1.0 / 30.0)

    self.assertEqual(result.diagnostics.locked, (False, False))
    self.assertEqual(solver.snapshot_state()[1], (None, None))
```

Update invariance calls to pass authored stance:

```python
result = solver.apply(raw, (True, True), 1.0 / 30.0)
np.testing.assert_array_equal(result.qpos[:2], raw[:2])
np.testing.assert_array_equal(result.qpos[3:7], raw[3:7])
non_leg_except_root_z = solver.non_leg_qpos_addresses
non_leg_except_root_z = non_leg_except_root_z[
    non_leg_except_root_z != 2
]
np.testing.assert_array_equal(
    result.qpos[non_leg_except_root_z],
    raw[non_leg_except_root_z],
)
```

- [ ] **Step 2: Run model tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_motionbricks_hill_ik
```

Expected: `TypeError` because `apply` still accepts only `(raw_qpos, dt_s)`.

- [ ] **Step 3: Replace geometric state proposal**

Change the public signature:

```python
def apply(
    self,
    raw_qpos: object,
    authored_stance: object,
    dt_s: float,
) -> HillFootIKResult:
```

Validate contacts:

```python
contacts_array = np.asarray(authored_stance)
if contacts_array.shape != (2,) or contacts_array.dtype.kind != "b":
    raise ValueError("authored_stance must contain two booleans")
contact = (bool(contacts_array[0]), bool(contacts_array[1]))
```

At the beginning of `apply`, advance falling edges before any terrain query:

```python
proposed_targets = [
    self._targets[foot].copy()
    if contact[foot] and self._previous_contact[foot]
       and self._targets[foot] is not None
    else None
    for foot in range(2)
]
for foot in range(2):
    if not contact[foot]:
        proposed_targets[foot] = None
```

Compute the root correction from the pose containing the prior root offset:

```python
root_probe = raw.copy()
root_probe[2] += self._root_height_correction_m
self._set_pose(root_probe)
probe_centers = (
    self._sole_centers(0).copy(),
    self._sole_centers(1).copy(),
)
required_shifts = [
    float(self.height_query(center[:2]))
    + float(radius)
    - float(center[2])
    for foot in range(2)
    if contact[foot]
    for center, radius in zip(
        probe_centers[foot], self._sphere_radii[foot], strict=True
    )
]
next_root_correction = _bounded_root_height_correction(
    self._root_height_correction_m,
    required_support_shift_m=max(required_shifts, default=0.0),
    has_support=bool(required_shifts),
)
root_adjusted = raw.copy()
root_adjusted[2] += next_root_correction
self._set_pose(root_adjusted)
root_adjusted_centers = (
    self._sole_centers(0).copy(),
    self._sole_centers(1).copy(),
)
```

Construct only authored rising-edge targets:

```python
for foot in range(2):
    if contact[foot] and proposed_targets[foot] is None:
        proposed_targets[foot] = _project_stance_targets(
            root_adjusted_centers[foot],
            self._sphere_radii[foot],
            self.height_query,
        )
```

Change `_solve_foot` from phase ownership to an explicit constraint choice:

```python
def _solve_foot(
    self,
    qpos: np.ndarray,
    foot: int,
    target_centers: np.ndarray,
    *,
    lock_horizontal: bool,
) -> np.ndarray:
```

The solver always adds four vertical rows. It adds the mean sole X/Y rows only
when `lock_horizontal` is true. Call it with `lock_horizontal=True` for authored
stance and `False` for upward-only swing clearance.

Generalize `_bounded_leg_correction` so the caller supplies both safety bounds:

```python
def _bounded_leg_correction(
    raw_leg: np.ndarray,
    desired_leg: np.ndarray,
    previous_correction: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    *,
    maximum_correction_rad: float,
    maximum_step_rad: float,
) -> tuple[np.ndarray, bool]:
    desired_correction = np.clip(
        desired_leg - raw_leg,
        -maximum_correction_rad,
        maximum_correction_rad,
    )
    correction = np.clip(
        desired_correction,
        previous_correction - maximum_step_rad,
        previous_correction + maximum_step_rad,
    )
    bounded_leg = np.clip(
        raw_leg + correction,
        lower_limits,
        upper_limits,
    )
    forced_by_limits = not np.array_equal(
        bounded_leg,
        raw_leg + correction,
    )
    return bounded_leg - raw_leg, forced_by_limits
```

For authored stance, pass `maximum_correction_rad=0.30` and
`maximum_step_rad=0.08`. Outside stance, solve only an upward swing target when
needed; otherwise use raw as desired, then pass `maximum_correction_rad=0.30`
and `maximum_step_rad=0.12` so old correction decays without a horizontal
target.

The call site is:

```python
next_correction, forced_by_limits = _bounded_leg_correction(
    raw_leg,
    desired_leg,
    self._corrections[foot],
    self._joint_lower[foot],
    self._joint_upper[foot],
    maximum_correction_rad=0.30,
    maximum_step_rad=0.08 if contact[foot] else 0.12,
)
```

The candidate begins from `root_adjusted`, then overwrites every non-leg qpos
except address `2` from raw.

- [ ] **Step 4: Make release state unconditional on fallback**

Add:

```python
def _commit_releases(
    self,
    contact: tuple[bool, bool],
    targets: list[np.ndarray | None],
) -> None:
    retained = list(self._targets)
    for foot in range(2):
        if not contact[foot]:
            retained[foot] = None
    self._targets = (retained[0], retained[1])
    self._previous_contact = contact
```

Call it in the exception path before returning fallback. On successful frames,
commit contacts, targets, root correction, and corrections normally. Fallback
diagnostics must calculate `locked` from the post-release target state.

- [ ] **Step 5: Remove obsolete geometric phase ownership**

Delete:

- `FootPhase`;
- `_next_foot_phase`;
- `_previous_raw_centers`;
- clearance/speed stance entry and release thresholds; and
- `phases` from diagnostics.

No remaining code may create a stance target when `contact[foot]` is false.

- [ ] **Step 6: Run tests and verify GREEN**

Run the command from Step 2. Expected: all pure and native-model tests pass.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_ik.py \
  tests/python/test_motionbricks_hill_ik.py
git commit -m "feat: drive hill IK from authored contacts"
```

### Task 4: Pair contacts with qpos in the viewer and verify the full mound

**Files:**
- Modify: `sonic/python/mm_sonic/motionbricks_hill_viewer.py`
- Modify: `tests/python/test_motionbricks_hill.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Consumes: `sample_authored_contacts(full_agent, frame_index)` and `MotionBricksHillFootIK.apply(raw_qpos, contacts.stance, dt_s)`.
- Produces: authored contact/lock/root diagnostics in headless and interactive modes.

- [ ] **Step 1: Write the failing trace contract**

Update the trace test:

```python
from mm_sonic.motionbricks_authored_contacts import AuthoredFootContacts

contacts = AuthoredFootContacts(
    channels=(True, False, False, False),
    stance=(True, False),
    valid=True,
    reason="ok",
)
line = _trace_line(
    3,
    np.asarray((0.0, 0.0, 0.8)),
    GentleHillProfile(),
    SimpleNamespace(latest_trace=None),
    diagnostics,
    contacts,
)
self.assertIn("contacts=1000", line)
self.assertIn("stance=10", line)
self.assertIn("locked=10", line)
self.assertIn("root_dz=", line)
```

- [ ] **Step 2: Run the trace test and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_motionbricks_hill.HillViewerCliTest.test_trace_reports_ik_fallback_reason
```

Expected: `TypeError` because `_trace_line` does not accept contacts.

- [ ] **Step 3: Pair the frame index, contacts, and qpos**

Import:

```python
from .motionbricks_authored_contacts import (
    AuthoredFootContacts,
    sample_authored_contacts,
)
```

Change `_step_agent` to return both values:

```python
def _step_agent(
    demo: object,
    profile: GentleHillProfile,
    *,
    viewer: object,
    random_seed: int,
    automatic: bool,
) -> tuple[np.ndarray, AuthoredFootContacts]:
    import torch

    frame_index = int(demo.full_agent._current_frame_idx)
    contacts = sample_authored_contacts(
        demo.full_agent, frame_index
    )
    qpos = np.asarray(
        demo.full_agent.get_next_frame(), dtype=np.float64
    )
    # Existing raw controller and generation logic remains unchanged.
    return qpos, contacts
```

In both loops:

```python
raw_qpos, contacts = _step_agent(
    demo,
    profile,
    viewer=viewer,
    random_seed=arguments.random_seed,
    automatic=arguments.no_viewer,
)
ik_result = (
    None
    if foot_ik is None
    else foot_ik.apply(
        raw_qpos,
        contacts.stance,
        float(model.opt.timestep),
    )
)
```

- [ ] **Step 4: Extend trace diagnostics**

Change `_trace_line` to accept `contacts` and emit:

```python
channels = "".join("1" if value else "0" for value in contacts.channels)
stance = "".join("1" if value else "0" for value in contacts.stance)
locked = "".join(
    "1" if value else "0" for value in ik_diagnostics.locked
)
ik_text = (
    f"contacts={channels} stance={stance} locked={locked} "
    f"root_dz={ik_diagnostics.root_height_correction_m:.3f} "
    f"penetration={ik_diagnostics.raw_penetration_m:.3f}"
    f"->{ik_diagnostics.corrected_penetration_m:.3f} "
    f"residual={ik_diagnostics.maximum_target_residual_m:.3f} "
    f"correction={ik_diagnostics.maximum_joint_correction_rad:.3f} "
    f"accepted={int(ik_diagnostics.accepted)} "
    f"reason={ik_diagnostics.reason.replace(' ', '_')}"
)
```

Before printing any trace with IK enabled, assert the safety invariant:

```python
if any(
    locked and not stance
    for locked, stance in zip(
        ik_diagnostics.locked, contacts.stance, strict=True
    )
):
    raise RuntimeError("released authored foot retained a world lock")
```

- [ ] **Step 5: Update the README**

Replace geometric stance language with:

```text
The viewer reads MotionBricks' authored heel/toe contacts from the same feature
frame as each displayed qpos. World-space foot targets exist only while those
contacts are active and are cleared at toe-off. A bounded display-only root-Z
offset handles terrain-height error before leg IK. None of these display
corrections enter MotionBricks context.
```

- [ ] **Step 6: Run focused verification**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_motionbricks_authored_contacts \
  tests.python.test_motionbricks_hill \
  tests.python.test_motionbricks_hill_conditioning \
  tests.python.test_motionbricks_hill_ik
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m compileall -q sonic/python/mm_sonic
git diff --check
```

Expected: all tests pass, compilation exits zero, and the diff is clean.

- [ ] **Step 7: Run raw and corrected CUDA canaries**

Run:

```bash
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.motionbricks_hill_viewer \
  --no-viewer --no-ik --smoke-steps 360 --trace-every 60
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.motionbricks_hill_viewer \
  --no-viewer --smoke-steps 360 --trace-every 30
```

Expected corrected-arm invariants:

- every trace has finite values;
- no trace has a zero stance bit paired with a one lock bit;
- `abs(root_dz) <= 0.20`;
- `correction <= 0.30`;
- the root reaches and leaves the mound; and
- flank/crest penetration is lower than the first prototype's `0.151 m` peak.

- [ ] **Step 8: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_viewer.py \
  tests/python/test_motionbricks_hill.py sonic/README.md
git commit -m "feat: render authored-contact MotionBricks terrain IK"
```

- [ ] **Step 9: Launch the interactive acceptance run**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m \
  mm_sonic.motionbricks_hill_viewer \
  --max-steps 1000000 --trace-every 300
```

Expected: the viewer stays open; W/A/S/D remains controllable; released feet
never remain locked behind the moving root.
