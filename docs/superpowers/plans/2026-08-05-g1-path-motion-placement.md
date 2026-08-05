# G1 Path Motion Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically retrieve and rigidly place raw GRAIL motion windows along one fixed-staircase path, then report raw path coverage and transition gaps.

**Architecture:** A path produces an approximate terrain-contact signature. A corpus scanner extracts variable-length raw contact signatures without using source-family labels, then a coarse matcher ranks them. The shortlist is rigidly aligned to the path, validated using actual feet and complete soles, and emitted as deterministic placed-window evidence.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo G1 FK/sole kinematics, existing `mm_sonic` terrain dataset and viewer utilities, `unittest`.

## Global Constraints

- Keep the target staircase fixed.
- The character faces and travels forward along each queried path.
- Search all accepted GRAIL terrain clips regardless of source-family label.
- Preserve all raw joint positions exactly.
- Allow only global yaw, XY translation, and one root-Z translation per window.
- Do not use Sonic, physics tracking, IK, foot projection, or spatial scaling.
- Nominal footprints are retrieval hints; actual raw contacts determine acceptance.
- Report uncovered path intervals rather than claiming arbitrary inbetweening works.
- Evaluate at 50 Hz.

---

### Task 1: Path contact signatures

**Files:**
- Create: `sonic/python/mm_sonic/torch_path_motion_placement.py`
- Create: `tests/python/test_sonic_torch_path_motion_placement.py`

**Interfaces:**
- Consumes: `NominalFootprintPath` and a callable terrain sampler.
- Produces: `PathContactSignature`, `RawContactEvent`, `RawMotionWindow`, `path_contact_signature(...)`, `extract_raw_motion_windows(...)`, and `contact_signature_cost(...)`.

- [ ] **Step 1: Write failing tests for path signatures**

Add tests proving that:

```python
signature = path_contact_signature(
    path=nominal_path,
    sample_surface=step_surface,
)
self.assertEqual(signature.foot_order, (0, 1, 0, 1))
self.assertEqual(signature.height_pattern_m, (0.0, 0.18, 0.0, 0.18))
```

The sampler must receive complete sole-center positions from the nominal path,
and the signature must store path-relative forward progress rather than scene
coordinates.

- [ ] **Step 2: Run the signature test and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_sonic_torch_path_motion_placement
```

Expected: import failure for `mm_sonic.torch_path_motion_placement`.

- [ ] **Step 3: Implement immutable signature records**

Implement validated frozen dataclasses:

```python
@dataclass(frozen=True)
class PathContactSignature:
    foot_order: tuple[int, ...]
    forward_m: tuple[float, ...]
    lateral_m: tuple[float, ...]
    contact_frame: tuple[int, ...]
    height_pattern_m: tuple[float, ...]

@dataclass(frozen=True)
class RawContactEvent:
    frame: int
    foot: int
    position_world_xy: tuple[float, float]
    surface_height_m: float

@dataclass(frozen=True)
class RawMotionWindow:
    source_clip: str
    start_frame: int
    stop_frame: int
    events: tuple[RawContactEvent, ...]
    forward_progress_m: float
    heading_error_rad: float
```

`path_contact_signature` samples the supplied terrain at each nominal
footprint and subtracts the first contact height so signatures are invariant
to global terrain elevation.

- [ ] **Step 4: Write failing extraction and matching tests**

Use synthetic support masks, feet, roots, quaternions, and surface heights to
prove:

- planted-foot flicker at the same XY/height is merged;
- raw windows may begin and end on level contacts;
- a level/split/level source sequence is extracted without source labels;
- forward-heading windows rank ahead of sideways windows;
- contact height, foot order, spacing, and timing contribute independently;
- mirrored starting phase is allowed only through an explicit foot swap.

- [ ] **Step 5: Run extraction tests and verify RED**

Run the same `unittest` command. Expected: failures for missing extraction and
cost behavior.

- [ ] **Step 6: Implement extraction and coarse matching**

`extract_raw_motion_windows` must:

- derive touchdown events from support onsets;
- merge same-foot detections within 0.07 m and 0.04 m height;
- enumerate alternating event windows with 4–12 contacts;
- require at least 0.40 m positive root progress;
- measure root-heading error against window travel;
- retain level, mount, split-height, dismount, and mixed patterns.

`contact_signature_cost` must align source and query height baselines and score
foot identity, normalized forward spacing, lateral spacing, relative height,
timing, progress, and heading. It must not inspect clip names.

- [ ] **Step 7: Run unit tests and commit**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_sonic_torch_path_motion_placement \
  tests.python.test_sonic_torch_heading_footprint_path \
  tests.python.test_sonic_torch_grail_contact_window_search
```

Expected: all tests pass.

Commit:

```bash
git add sonic/python/mm_sonic/torch_path_motion_placement.py \
  tests/python/test_sonic_torch_path_motion_placement.py
git commit -m "feat: index path contact motion windows"
```

---

### Task 2: Rigid path placement and certification

**Files:**
- Modify: `sonic/python/mm_sonic/torch_path_motion_placement.py`
- Modify: `tests/python/test_sonic_torch_path_motion_placement.py`

**Interfaces:**
- Consumes: a `RawMotionWindow`, raw joint/root/foot/sole trajectories, a scene path, and terrain sampler.
- Produces: `PlacedRawWindow`, `PlacementMetrics`, and `place_raw_window_on_path(...)`.

- [ ] **Step 1: Write failing rigid-placement tests**

Prove that placement:

- aligns source travel with an arbitrary path heading;
- maps the source start root to the path start;
- uses one median root-Z translation from supported contact residuals;
- preserves all joint positions bit-for-bit;
- rejects heading error above 15 degrees;
- rejects root lateral deviation above 0.15 m;
- rejects supported contact error above 0.03 m;
- rejects complete-sole penetration below -0.03 m.

- [ ] **Step 2: Run placement tests and verify RED**

Run the Task 1 test module. Expected: missing placement API failures.

- [ ] **Step 3: Implement placement and metrics**

Implement:

```python
@dataclass(frozen=True)
class PlacementMetrics:
    covered_start_m: float
    covered_stop_m: float
    heading_error_p95_rad: float
    maximum_lateral_error_m: float
    maximum_stance_error_m: float
    minimum_sole_clearance_m: float

@dataclass(frozen=True)
class PlacedRawWindow:
    window: RawMotionWindow
    yaw_scene_rad: float
    translation_scene_xyz: tuple[float, float, float]
    joint_position: np.ndarray
    root_position_scene: np.ndarray
    root_orientation_scene_wxyz: np.ndarray
    metrics: PlacementMetrics
```

The function returns either a certified placement or a stable rejection reason.
It performs no joint edits.

- [ ] **Step 4: Run placement tests and commit**

Run the Task 1 test module and expect all tests to pass.

Commit:

```bash
git add sonic/python/mm_sonic/torch_path_motion_placement.py \
  tests/python/test_sonic_torch_path_motion_placement.py
git commit -m "feat: certify rigid motion path placements"
```

---

### Task 3: Full-corpus automatic rediscovery

**Files:**
- Create: `resources/run_g1_path_motion_placement.py`
- Create: `tests/python/test_run_g1_path_motion_placement.py`
- Modify: `sonic/configs/experiments/torch_grail_raw_horizontal_preview.json`

**Interfaces:**
- Consumes: the GRAIL dataset, target scene, G1 XML, path start/stop, gait hypotheses, and shortlist size.
- Produces: `placements.json`, `coverage.json`, the best certified `traversal.npz`, and a reproducible viewer command.

- [ ] **Step 1: Write failing CLI contract tests**

Test strict argument validation, deterministic candidate ordering, absence of
source-family filtering, stable rejection counts, JSON schema fields, and that
the selected archive copies source joints unchanged.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_run_g1_path_motion_placement
```

Expected: import failure for the missing resource script.

- [ ] **Step 3: Implement parallel corpus scanning**

The CLI must:

- load the target terrain and construct left/right phase and stride hypotheses;
- scan every accepted terrain clip with `ProcessPoolExecutor`;
- compute support using existing GRAIL terrain sampling constants;
- extract variable raw windows;
- rank by the best path-signature cost;
- perform expensive FK/sole certification only on the deterministic shortlist;
- never accept `--source-clip` or another result-identity override.

- [ ] **Step 4: Emit deterministic evidence**

`placements.json` records query identity, corpus count, ranked coarse matches,
 certified placements, rejection reasons, and transforms. `coverage.json`
 records the best interval and uncovered prefix/suffix. `traversal.npz` uses
 matcher coordinates expected by the existing MuJoCo viewer.

- [ ] **Step 5: Run CLI unit tests and relevant regression tests**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_run_g1_path_motion_placement \
  tests.python.test_sonic_torch_path_motion_placement \
  tests.python.test_sonic_torch_contact_oracle_search
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add resources/run_g1_path_motion_placement.py \
  tests/python/test_run_g1_path_motion_placement.py \
  sonic/configs/experiments/torch_grail_raw_horizontal_preview.json
git commit -m "feat: search GRAIL motions along a terrain path"
```

---

### Task 4: Decisive fixed-staircase experiment

**Files:**
- Create: `docs/superpowers/results/2026-08-05-g1-path-motion-placement.md`
- Generated: `build/g1-path-motion-placement-horizontal/`

**Interfaces:**
- Consumes: the Task 3 CLI.
- Produces: the evidence required for the 2:30 PM checkpoint.

- [ ] **Step 1: Run automatic rediscovery**

Run the CLI using only the full dataset, target scene, G1 XML, and approved path
coordinates. Do not pass the known source clip or transform.

- [ ] **Step 2: Verify identity independence**

Inspect `placements.json` and the command line. Confirm no selected clip ID,
family name, frame range, or transform appears in input configuration.

- [ ] **Step 3: Verify numerical acceptance**

Require:

- heading p95 at or below 15 degrees;
- lateral root deviation at or below 0.15 m;
- maximum stance error at or below 0.03 m;
- minimum complete-sole clearance at or above -0.03 m;
- exact raw joint preservation.

- [ ] **Step 4: Render and inspect**

Render a contact sheet and launch the existing 50 Hz MuJoCo playback. Record
whether the motion visibly walks forward, mounts, traverses, and dismounts.

- [ ] **Step 5: Run a longer-path coverage probe**

Extend the path beyond the best single window. Report whether an exact repeat
or second placement is compatible. If not, record the uncovered interval and
boundary mismatch rather than creating an inbetween.

- [ ] **Step 6: Document and commit the evidence**

Write exact commands, selected identities, metrics, coverage, failures, and the
go/no-go conclusion.

Commit:

```bash
git add docs/superpowers/results/2026-08-05-g1-path-motion-placement.md
git commit -m "docs: report path motion placement evidence"
```
