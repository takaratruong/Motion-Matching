# G1 Same-Stair Omnidirectional Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic renderer-independent benchmark that drives one authenticated staircase from every meaningful direction and measures terrain support, foot sliding, turns, side mounts/exits, and rescue cycles.

**Architecture:** A pure staircase-local route generator emits exact 50 Hz command episodes and transforms them into matcher world. A metrics module evaluates emitted MuJoCo-FK kinematics against the fixed query grid, and a runner compares the qualified small corpus with the expanded corpus under identical matcher configuration.

**Tech Stack:** Python 3.10, NumPy, PyTorch/CUDA, MuJoCo forward kinematics, unittest, existing Torch terrain matcher and artifact writers.

## Global Constraints

- Consume the expanded corpus from the preceding plan and preserve the small corpus as baseline.
- Use one fixed authenticated staircase for every route.
- Use exactly 50 Hz commands and kinematics.
- Do not call MuJoCo `mj_step`; FK only.
- Commands are defined in staircase-local coordinates, never source-clip coordinates.
- Evaluate each foot against its own local terrain height.
- Count sliding only in frozen stance-like intervals, never during swing.
- Freeze commands, route transforms, thresholds, and artifact identities before comparing algorithms.
- The first benchmark compares data only; matcher configuration is identical.

---

### Task 1: Pure same-stair route definitions

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_omni_routes.py`
- Create: `tests/python/test_sonic_torch_terrain_omni_routes.py`

**Interfaces:**
- Produces: `StairFrame`, `RouteCommand`, `OmniRoute`, and `same_stair_routes()`.

- [ ] **Step 1: Write route identity and transform tests**

Assert the frozen route names are:

```python
EXPECTED_NAMES = (
    "side-mount-left", "side-mount-right",
    "cross-tread-left-to-right", "cross-tread-right-to-left",
    "turn-45-lower-left", "turn-45-lower-right",
    "turn-90-middle-left", "turn-90-middle-right",
    "turn-180-upper-left", "turn-180-upper-right",
    "diagonal-up-left", "diagonal-up-right",
    "diagonal-down-left", "diagonal-down-right",
    "side-exit-lower-left", "side-exit-lower-right",
    "side-exit-upper-left", "side-exit-upper-right",
    "riser-stop-restart", "riser-reversal",
    "mixed-adversarial",
)
```

Rotate a synthetic stair frame by 90 degrees and prove every velocity and
heading rotates exactly while frame count, segment labels, and reset markers
remain unchanged.

- [ ] **Step 2: Run route tests and verify RED**

Expected: FAIL because the route module is absent.

- [ ] **Step 3: Implement immutable route contracts**

Define:

```python
@dataclass(frozen=True)
class StairFrame:
    origin_world_xy: tuple[float, float]
    ascent_world_yaw: float
    width_m: float
    tread_depth_m: float
    riser_height_m: float
    tread_count: int

@dataclass(frozen=True)
class RouteCommand:
    velocity_stair_xy: tuple[float, float]
    heading_stair_yaw: float
    frames: int
    segment: str
    reset_before: bool = False

@dataclass(frozen=True)
class OmniRoute:
    name: str
    commands: tuple[RouteCommand, ...]
    required_outcome: Literal["mount", "traverse", "turn", "exit", "mixed"]
```

Generate mirrored routes from one canonical definition, but materialize and
hash the complete expanded command list in artifacts.

- [ ] **Step 4: Run route tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_omni_routes
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_terrain_omni_routes.py \
  tests/python/test_sonic_torch_terrain_omni_routes.py
git commit -m "feat: define same-stair omnidirectional routes"
```

---

### Task 2: Per-foot support, sliding, and rescue-cycle metrics

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_omni_metrics.py`
- Create: `tests/python/test_sonic_torch_terrain_omni_metrics.py`

**Interfaces:**
- Consumes: 50 Hz root/foot kinematics, terrain samples, commands, and selection diagnostics.
- Produces: `evaluate_omni_route()` and structured route metrics.

- [ ] **Step 1: Write unequal-tread support tests**

Construct left/right ankle origins at surface heights `0.1778` and `0.3556`
plus the 0.035 m sole allowance. Assert both feet are supported with
`support_height_difference_m == 0.1778`; a common-ground implementation must
fail this test.

- [ ] **Step 2: Write stance-only sliding tests**

Create a 20-frame stance interval with clearance near 0.035 m, low vertical
speed, and 4 cm horizontal drift. Follow it with a swing interval moving 30 cm.
Assert only 4 cm is accumulated and the swing displacement is excluded.

- [ ] **Step 3: Write rescue-cycle detector tests**

Feed rescue events:

```python
[
    ("stair/0002", 321, (0.00, 0.00)),
    ("stair/0000", 336, (0.01, 0.00)),
    ("stair/0002", 323, (0.02, 0.00)),
    ("stair/0000", 338, (0.03, 0.00)),
]
```

Assert one `A -> B -> A -> B` cycle because frames share eight-frame
neighborhoods and net root displacement is below 0.05 m. Assert no cycle when
the fourth root is 0.08 m away or one event is not a terrain rescue.

- [ ] **Step 4: Run metric tests and verify RED**

Expected: FAIL because the metric module is absent.

- [ ] **Step 5: Implement pure metrics**

Define frozen thresholds:

```python
DT_S = 0.02
ANKLE_ORIGIN_SOLE_M = 0.035
STANCE_CLEARANCE_TOLERANCE_M = 0.02
STANCE_VERTICAL_SPEED_MAX_MPS = 0.12
RESCUE_NEIGHBORHOOD_FRAMES = 8
RESCUE_CYCLE_PROGRESS_MIN_M = 0.05
```

Return per-route distributions for clearance, penetration, stance slide,
support-height error, heading error, root progress, jerk, transition counts,
rescue cycles, and required-outcome completion.

- [ ] **Step 6: Run metric tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_omni_metrics
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add sonic/python/mm_sonic/torch_terrain_omni_metrics.py \
  tests/python/test_sonic_torch_terrain_omni_metrics.py
git commit -m "feat: measure same-stair terrain interaction"
```

---

### Task 3: Deterministic CUDA benchmark runner and artifacts

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_omni_rollout.py`
- Create: `tests/python/test_sonic_torch_terrain_omni_rollout.py`
- Create: `sonic/configs/experiments/torch_terrain_expanded.json`

**Interfaces:**
- Consumes: route generator, metrics, dataset, fixed query terrain, matcher config.
- Produces: one artifact directory per route and `matrix.json`.

- [ ] **Step 1: Write runner contract tests**

With a tiny fake matcher, assert every command commits exactly one frame,
Backspace-equivalent route resets happen only at declared boundaries, and saved
NPZ arrays include command, root, joints, MuJoCo-FK feet, local surface,
selection, rescue, and timing.

- [ ] **Step 2: Write exception-isolation and hash tests**

Make one route raise at frame 7. Assert other routes finish, the failed route
records stage/index/type/message, `matrix_pass` is false, and deterministic
hashes exclude timing arrays but include commands/config/dataset identities.

- [ ] **Step 3: Run runner tests and verify RED**

Expected: FAIL because the runner is absent.

- [ ] **Step 4: Implement renderer-independent runner**

Instantiate `TorchMotionMatcher` exactly as the qualified live viewer does.
Apply `matcher_result_qpos` and MuJoCo `mj_forward` for actual displayed feet.
Sample the fixed query grid independently for each foot. Save artifacts
transactionally using the existing challenge-matrix conventions.

- [ ] **Step 5: Add expanded experiment config**

Copy every retained matcher value from
`sonic/configs/experiments/torch_stair_small.json`; change only dataset/query
descriptors needed for `build/torch-terrain-expanded`. Pin:

```json
{
  "transition_joint_position_weight": 0.1,
  "transition_joint_velocity_weight": 0.1,
  "joint_reference_smoothing_weight": 0.2,
  "transition_window_jerk_weight": 0.0
}
```

- [ ] **Step 6: Run runner and existing challenge tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_terrain_omni_rollout \
  tests.python.test_sonic_torch_terrain_directional_rollout \
  tests.python.test_sonic_torch_terrain_live_viewer
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add sonic/python/mm_sonic/torch_terrain_omni_rollout.py \
  sonic/configs/experiments/torch_terrain_expanded.json \
  tests/python/test_sonic_torch_terrain_omni_rollout.py
git commit -m "feat: benchmark omnidirectional stair kinematics"
```

---

### Task 4: Small-versus-expanded data ablation and frozen gates

**Files:**
- Create: `docs/superpowers/results/2026-07-30-g1-same-stair-omnidirectional-benchmark.md`
- Generated: `build/torch-terrain-omni/small/`
- Generated: `build/torch-terrain-omni/expanded/`

**Interfaces:**
- Consumes: completed expanded corpus and Tasks 1–3.
- Produces: data-coverage verdict and frozen benchmark baseline for algorithm iteration.

- [ ] **Step 1: Run the complete matrix on the small corpus**

Use a free GPU and save all 21 routes under `small/`. Expected: completion even
when individual quality gates fail; the known lateral rescue cycle must be
detected if the recorded replay reaches it.

- [ ] **Step 2: Run the identical matrix on the expanded corpus**

Use the same GPU model, commands, query terrain, and matcher config. Expected:
same command hashes and route identities as the small run.

- [ ] **Step 3: Evaluate frozen comparison gates**

Require no regression on existing dense gates, clearance at least `-0.03 m`,
zero unhandled exceptions, no `A -> B -> A -> B` rescue cycle, outcome
completion for every route, and non-regressed stance sliding. Report failures
without relaxing thresholds.

- [ ] **Step 4: Run complete verification**

Run the full Torch suite, CUDA flat/legacy/dense rollout, directional benchmark,
six-scenario challenge matrix, and both omnidirectional matrices. Expected:
tests pass; the result document distinguishes software qualification from
research-quality failures.

- [ ] **Step 5: Record the data-only verdict**

Document per-route progress, clearance, support-height error, sliding, turn
settling, selected motion families, rescue cycles, and p50/p95 search time.
State whether additional coverage solved the observed cycle.

- [ ] **Step 6: Commit Task 4**

```bash
git add docs/superpowers/results/2026-07-30-g1-same-stair-omnidirectional-benchmark.md
git commit -m "docs: evaluate omnidirectional stair coverage"
```

- [ ] **Step 7: Launch qualified interactive viewer**

Launch the expanded dataset in the existing kinematic MuJoCo viewer only after
verification. Ask the user to repeat side mount, cross-tread walking, turns,
diagonals, and side exits. Capture any new deterministic failure as the next
algorithm hypothesis rather than tuning several mechanisms together.
