# MotionBricks 18-Degree Hill Foot IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the controllable G1 MotionBricks prototype traverse an 18-degree radial mound with bounded stance-aware leg IK while keeping raw MotionBricks root and context unchanged.

**Architecture:** Raise the existing analytic mound to an exact 18-degree maximum grade. Add a qpos-native post-process that infers stance from raw sole clearance and speed, latches planted sole targets, solves only each leg's six joints with damped least squares, and applies upward-only swing clearance. Integrate the corrected copy only into MuJoCo rendering; generation and controller context continue to consume raw qpos.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo 3.x, pinned MotionBricks G1 checkpoints, `unittest`.

## Global Constraints

- Mound diameter remains exactly `7.0 m`; crest height becomes `0.7239760607 m` for an exact `18.0°` maximum grade.
- Root qpos `[0:7]` and every non-leg joint remain exactly equal to raw MotionBricks output.
- IK never mutates `full_agent.frames` and is never included in MotionBricks context.
- Stance enter thresholds are `0.035 m` clearance and `0.35 m/s` foot speed.
- Stance release thresholds are `0.075 m` clearance and `0.75 m/s` foot speed.
- Swing clearance is upward-only with a `0.015 m` minimum.
- Per-iteration joint motion is at most `0.04 rad`, absolute correction at most `0.35 rad`, and accepted inter-frame correction change at most `0.06 rad`.
- A failed IK frame falls back to raw qpos without stopping MotionBricks or committing temporal state.

---

### Task 1: Raise the radial mound to 18 degrees

**Files:**
- Modify: `sonic/python/mm_sonic/motionbricks_hill.py`
- Modify: `tests/python/test_motionbricks_hill.py`

**Interfaces:**
- Consumes: `GentleHillProfile.height(xy)` and `GentleHillProfile.mesh(sample_count)`.
- Produces: `DEFAULT_HILL_SLOPE_DEGREES`, `DEFAULT_HILL_HEIGHT_M`, and a default `GentleHillProfile` with an exact 18-degree maximum grade.

- [ ] **Step 1: Write the failing slope test**

Add:

```python
from mm_sonic.motionbricks_hill import (
    DEFAULT_HILL_HEIGHT_M,
    DEFAULT_HILL_SLOPE_DEGREES,
)

def test_default_mound_has_exact_eighteen_degree_grade(self) -> None:
    self.assertEqual(DEFAULT_HILL_SLOPE_DEGREES, 18.0)
    self.assertAlmostEqual(DEFAULT_HILL_HEIGHT_M, 0.7239760607, places=9)
    self.assertAlmostEqual(self.hill.height_m, DEFAULT_HILL_HEIGHT_M)
    self.assertAlmostEqual(self.hill.max_slope_degrees, 18.0, places=10)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python \
  -m unittest -v \
  tests.python.test_motionbricks_hill.GentleHillProfileTest.test_default_mound_has_exact_eighteen_degree_grade
```

Expected: import failure because the constants do not exist.

- [ ] **Step 3: Implement the exact slope constants**

Add above the dataclass and use the height constant as its default:

```python
DEFAULT_HILL_SLOPE_DEGREES = 18.0
DEFAULT_HILL_DIAMETER_M = 7.0
DEFAULT_HILL_HEIGHT_M = (
    math.tan(math.radians(DEFAULT_HILL_SLOPE_DEGREES))
    * DEFAULT_HILL_DIAMETER_M
    / math.pi
)

@dataclass(frozen=True)
class GentleHillProfile:
    domain_x: tuple[float, float] = (-5.0, 15.0)
    half_width: float = 8.0
    hill_start_x: float = 1.5
    hill_length: float = DEFAULT_HILL_DIAMETER_M
    height_m: float = DEFAULT_HILL_HEIGHT_M
```

- [ ] **Step 4: Run all hill tests and verify GREEN**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python \
  -m unittest -v tests.python.test_motionbricks_hill
```

Expected: all hill tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill.py \
  tests/python/test_motionbricks_hill.py
git commit -m "feat: raise MotionBricks mound to 18 degrees"
```

### Task 2: Define stance state and terrain target contracts

**Files:**
- Create: `sonic/python/mm_sonic/motionbricks_hill_ik.py`
- Create: `tests/python/test_motionbricks_hill_ik.py`

**Interfaces:**
- Consumes: a scalar finite `height_query(xy) -> float`.
- Produces: `FootPhase`, `HillFootIKDiagnostics`, `HillFootIKResult`, `_next_foot_phase`, `_project_stance_targets`, and `_required_swing_lift`.

- [ ] **Step 1: Write failing pure-contract tests**

Create tests that import the new module and require:

```python
def test_stance_hysteresis(self) -> None:
    self.assertEqual(
        _next_foot_phase(FootPhase.SWING, 0.02, 0.20),
        FootPhase.STANCE,
    )
    self.assertEqual(
        _next_foot_phase(FootPhase.STANCE, 0.06, 0.60),
        FootPhase.STANCE,
    )
    self.assertEqual(
        _next_foot_phase(FootPhase.STANCE, 0.08, 0.20),
        FootPhase.RELEASE,
    )
    self.assertEqual(
        _next_foot_phase(FootPhase.RELEASE, 0.08, 0.80),
        FootPhase.SWING,
    )

def test_stance_targets_follow_per_probe_terrain_height(self) -> None:
    centers = np.asarray(((1.0, 0.0, 0.2), (2.0, 0.0, 0.2)))
    radii = np.asarray((0.02, 0.03))
    targets = _project_stance_targets(
        centers, radii, lambda xy: 0.1 * float(xy[0])
    )
    np.testing.assert_allclose(
        targets,
        ((1.0, 0.0, 0.12), (2.0, 0.0, 0.23)),
    )

def test_swing_lift_is_upward_only(self) -> None:
    self.assertEqual(
        _required_swing_lift(
            np.asarray(((0.0, 0.0, 0.10),)),
            np.asarray((0.02,)),
            lambda xy: 0.0,
        ),
        0.0,
    )
    self.assertAlmostEqual(
        _required_swing_lift(
            np.asarray(((0.0, 0.0, 0.01),)),
            np.asarray((0.02,)),
            lambda xy: 0.0,
        ),
        0.025,
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python \
  -m unittest -v tests.python.test_motionbricks_hill_ik
```

Expected: import failure because `motionbricks_hill_ik` does not exist.

- [ ] **Step 3: Implement the pure contracts**

Create:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

HeightQuery = Callable[[object], float]

class FootPhase(str, Enum):
    SWING = "swing"
    STANCE = "stance"
    RELEASE = "release"

@dataclass(frozen=True)
class HillFootIKDiagnostics:
    phases: tuple[FootPhase, FootPhase]
    raw_penetration_m: float
    corrected_penetration_m: float
    maximum_target_residual_m: float
    maximum_joint_correction_rad: float
    iterations: int
    accepted: bool
    reason: str

@dataclass(frozen=True)
class HillFootIKResult:
    qpos: np.ndarray
    diagnostics: HillFootIKDiagnostics

def _next_foot_phase(
    previous: FootPhase,
    minimum_clearance_m: float,
    speed_mps: float,
) -> FootPhase:
    if previous is FootPhase.STANCE:
        if minimum_clearance_m > 0.075 or speed_mps > 0.75:
            return FootPhase.RELEASE
        return FootPhase.STANCE
    if previous is FootPhase.RELEASE:
        return FootPhase.SWING
    if minimum_clearance_m <= 0.035 and speed_mps <= 0.35:
        return FootPhase.STANCE
    return FootPhase.SWING

def _project_stance_targets(
    centers_world: np.ndarray,
    radii: np.ndarray,
    height_query: HeightQuery,
) -> np.ndarray:
    centers = np.asarray(centers_world, dtype=np.float64)
    sphere_radii = np.asarray(radii, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("sole centers must have shape (probes, 3)")
    if sphere_radii.shape != (len(centers),):
        raise ValueError("sole radii must match probe count")
    targets = centers.copy()
    targets[:, 2] = np.asarray(
        [float(height_query(point[:2])) for point in centers]
    ) + sphere_radii
    if not np.isfinite(targets).all():
        raise ValueError("stance targets must be finite")
    return targets

def _required_swing_lift(
    centers_world: np.ndarray,
    radii: np.ndarray,
    height_query: HeightQuery,
    minimum_clearance_m: float = 0.015,
) -> float:
    centers = np.asarray(centers_world, dtype=np.float64)
    sphere_radii = np.asarray(radii, dtype=np.float64)
    deficits = [
        float(height_query(point[:2]))
        + float(radius)
        + float(minimum_clearance_m)
        - float(point[2])
        for point, radius in zip(centers, sphere_radii, strict=True)
    ]
    return max(0.0, max(deficits, default=0.0))
```

- [ ] **Step 4: Run the pure tests and verify GREEN**

Run the command from Step 2. Expected: all pure tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_ik.py \
  tests/python/test_motionbricks_hill_ik.py
git commit -m "feat: define hill foot IK contact contracts"
```

### Task 3: Implement qpos-native bounded G1 leg IK

**Files:**
- Modify: `sonic/python/mm_sonic/motionbricks_hill_ik.py`
- Modify: `tests/python/test_motionbricks_hill_ik.py`

**Interfaces:**
- Consumes: native G1 `mujoco.MjModel`, `height_query`, raw qpos shape `(model.nq,)`, and positive `dt_s`.
- Produces: `MotionBricksHillFootIK(model, height_query).apply(raw_qpos, dt_s) -> HillFootIKResult` and `.reset()`.

- [ ] **Step 1: Add failing model-backed invariance tests**

Use the pinned model when available:

```python
MOTIONBRICKS_G1_SCENE = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/scene_29dof.xml"
)

@unittest.skipUnless(MOTIONBRICKS_G1_SCENE.is_file(), "G1 scene unavailable")
def test_apply_preserves_root_and_non_leg_qpos(self) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MOTIONBRICKS_G1_SCENE))
    raw = model.qpos0.copy()
    raw[:3] = (3.25, 0.0, 1.25)
    solver = MotionBricksHillFootIK(model, GentleHillProfile().height)
    solver.apply(raw, 1.0 / 30.0)
    result = solver.apply(raw, 1.0 / 30.0)
    np.testing.assert_array_equal(result.qpos[:7], raw[:7])
    np.testing.assert_array_equal(
        result.qpos[solver.non_leg_qpos_addresses],
        raw[solver.non_leg_qpos_addresses],
    )
    self.assertTrue(np.isfinite(result.qpos).all())
    self.assertLessEqual(
        result.diagnostics.maximum_joint_correction_rad,
        0.35 + 1.0e-9,
    )
```

Add a rollback test by injecting a height query that returns `NaN` after
initialization:

```python
@unittest.skipUnless(MOTIONBRICKS_G1_SCENE.is_file(), "G1 scene unavailable")
def test_failed_height_query_rolls_back_state(self) -> None:
    import mujoco

    finite = [True]

    def height_query(xy: object) -> float:
        del xy
        return 0.0 if finite[0] else math.nan

    model = mujoco.MjModel.from_xml_path(str(MOTIONBRICKS_G1_SCENE))
    raw = model.qpos0.copy()
    solver = MotionBricksHillFootIK(model, height_query)
    solver.apply(raw, 1.0 / 30.0)
    before = repr(solver.snapshot_state())
    finite[0] = False

    result = solver.apply(raw, 1.0 / 30.0)

    np.testing.assert_array_equal(result.qpos, raw)
    self.assertFalse(result.diagnostics.accepted)
    self.assertEqual(repr(solver.snapshot_state()), before)
```

- [ ] **Step 2: Run the model-backed tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_motionbricks_hill_ik
```

Expected: failure because `MotionBricksHillFootIK` is undefined.

- [ ] **Step 3: Implement model discovery and transactional state**

Implement these public methods and properties:

```python
class MotionBricksHillFootIK:
    def __init__(
        self,
        model: object,
        height_query: HeightQuery,
        *,
        maximum_iterations: int = 24,
        damping: float = 0.012,
        posture_weight: float = 0.00005,
    ) -> None:
        self.model = model
        self.height_query = height_query
        self.maximum_iterations = maximum_iterations
        self.damping = damping
        self.posture_weight = posture_weight
        self._discover_g1_layout()
        self._data = self._mujoco.MjData(model)
        self.reset()

    @property
    def non_leg_qpos_addresses(self) -> np.ndarray:
        return self._non_leg_qpos.copy()

    def reset(self) -> None:
        self._phases = (FootPhase.SWING, FootPhase.SWING)
        self._targets = (None, None)
        self._previous_raw_centers = None
        self._corrections = (
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )

    def snapshot_state(
        self,
    ) -> tuple[object, object, object, object]:
        return (
            self._phases,
            tuple(None if value is None else value.copy() for value in self._targets),
            None if self._previous_raw_centers is None else tuple(
                value.copy() for value in self._previous_raw_centers
            ),
            tuple(value.copy() for value in self._corrections),
        )
```

`_discover_g1_layout()` must resolve exactly one free joint, the named six
joints per leg from the design, four descendant sphere geoms per foot, their
radii, qpos/dof addresses, joint ranges, and a sorted complement of non-leg
qpos addresses.

- [ ] **Step 4: Implement the damped least-squares solve**

For each selected foot, iterate:

```python
current = self._sole_centers(foot)
jacobians = []
for geom_id in self._sphere_geoms[foot]:
    jacobian_position.fill(0.0)
    self._mujoco.mj_jac(
        self.model,
        self._data,
        jacobian_position,
        None,
        self._data.geom_xpos[geom_id],
        int(self.model.geom_bodyid[geom_id]),
    )
    jacobians.append(jacobian_position[:, dofs].copy())
vertical_error = target[:, 2] - current[:, 2]
rows = [value[2:3] for value in jacobians]
error = vertical_error
if phase is FootPhase.STANCE:
    centroid_jacobian = np.mean(np.asarray(jacobians), axis=0)
    centroid_error = (
        np.mean(target[:, :2], axis=0)
        - np.mean(current[:, :2], axis=0)
    )
    rows.extend((centroid_jacobian[0:1], centroid_jacobian[1:2]))
    error = np.concatenate((vertical_error, centroid_error))
jacobian = np.vstack(rows)
transpose = jacobian.T
lhs = (
    transpose @ jacobian
    + (self.damping**2 + self.posture_weight) * np.eye(6)
)
rhs = (
    transpose @ error
    + self.posture_weight * (reference - self._data.qpos[addresses])
)
delta = np.clip(np.linalg.solve(lhs, rhs), -0.04, 0.04)
self._data.qpos[addresses] = np.clip(
    self._data.qpos[addresses] + delta,
    limits[:, 0] + 1.0e-6,
    limits[:, 1] - 1.0e-6,
)
```

After both feet solve, clamp correction against raw joints to `±0.35 rad`,
then clamp its change against the last accepted correction to `±0.06 rad`.
Forward the bounded pose once more and calculate target residual and terrain
penetration from sole-sphere bottoms.

- [ ] **Step 5: Implement `apply` and rollback**

`apply` must:

- validate and copy raw qpos;
- measure raw sole centers, clearance, and speed;
- propose phases and stance targets without mutating owned state;
- use fixed stance targets, upward-only swing targets, or no target;
- solve into private `MjData`;
- overwrite root and non-leg qpos from raw after solving;
- keep the first frame measurement-only when velocity is unknown;
- accept only finite, bounded output with settled stance vertical residual at
  most `0.015 m`;
- during initial stance acquisition, commit a residual above `0.015 m` only
  when the rate-bounded correction strictly reduces terrain penetration;
- reject planted sole-centroid drift above `0.04 m`;
- commit proposed phases, targets, raw centres, and corrections on acceptance;
- return raw qpos and diagnostics with `accepted=False` on any exception or
  failed bound.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_motionbricks_hill_ik
```

Expected: pure and model-backed tests pass.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_ik.py \
  tests/python/test_motionbricks_hill_ik.py
git commit -m "feat: add bounded MotionBricks hill foot IK"
```

### Task 4: Integrate IK into the viewer without generation feedback

**Files:**
- Modify: `sonic/python/mm_sonic/motionbricks_hill_viewer.py`
- Modify: `tests/python/test_motionbricks_hill.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Consumes: `MotionBricksHillFootIK.apply(raw_qpos, dt_s)`.
- Produces: default IK-enabled viewer, `--no-ik`, and live IK diagnostics.

- [ ] **Step 1: Extend the failing CLI test**

Add:

```python
self.assertIn("--no-ik", help_text)
```

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python \
  -m unittest -v tests.python.test_motionbricks_hill.HillViewerCliTest
```

Expected: FAIL because the option is absent.

- [ ] **Step 2: Add the CLI switch and viewer integration**

Add:

```python
parser.add_argument(
    "--no-ik",
    action="store_true",
    help="render raw MotionBricks qpos without stance-aware foot IK",
)
```

After building `model, data`, construct the adapter unless disabled:

```python
foot_ik = (
    None
    if arguments.no_ik
    else MotionBricksHillFootIK(model, profile.height)
)
```

Keep `_step_agent` returning raw qpos. In both headless and interactive loops:

```python
raw_qpos = _step_agent(
    demo,
    profile,
    viewer=viewer,
    random_seed=arguments.random_seed,
    automatic=automatic,
)
ik_result = (
    None
    if foot_ik is None
    else foot_ik.apply(raw_qpos, float(model.opt.timestep))
)
display_qpos = raw_qpos if ik_result is None else ik_result.qpos
data.qpos[:] = display_qpos
mujoco.mj_forward(model, data)
```

Do not assign `display_qpos` to any `full_agent.frames` entry or context tensor.
Extend `_trace_line` to report phase, raw/corrected penetration, residual,
maximum correction, and acceptance from the latest result.

- [ ] **Step 3: Document IK and A/B launch commands**

Update `sonic/README.md` with:

```bash
# IK enabled (default)
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m \
  mm_sonic.motionbricks_hill_viewer

# Raw MotionBricks comparison
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m \
  mm_sonic.motionbricks_hill_viewer --no-ik
```

State explicitly that IK modifies only the display copy and is not fed back.

- [ ] **Step 4: Run focused verification**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_motionbricks_hill \
  tests.python.test_motionbricks_hill_conditioning \
  tests.python.test_motionbricks_hill_ik
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m compileall -q sonic/python/mm_sonic
git diff --check
```

Expected: all focused tests pass, compilation exits zero, and diff check is
clean.

- [ ] **Step 5: Run deterministic CUDA canaries**

Run raw and IK arms:

```bash
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.motionbricks_hill_viewer \
  --no-viewer --no-ik --smoke-steps 360 --trace-every 180
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.motionbricks_hill_viewer \
  --no-viewer --smoke-steps 360 --trace-every 180
```

Expected: both finish with finite qpos; IK reports bounded corrections and
accepted frames while the raw root reaches and leaves the 18-degree mound.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_viewer.py \
  tests/python/test_motionbricks_hill.py sonic/README.md
git commit -m "feat: enable foot IK in MotionBricks hill viewer"
```
