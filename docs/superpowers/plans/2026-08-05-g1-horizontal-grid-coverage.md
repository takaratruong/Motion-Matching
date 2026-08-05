# G1 Horizontal Grid Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the unchanged directed-path rigid-motion search on 11 parallel horizontal staircase lanes spaced every 0.20 m and play all certified results in one colored-grid MuJoCo viewer.

**Architecture:** Add a small pure grid/playlist module, a grid runner that extracts raw windows once per source clip and scores them independently for every lane, and optional grid metadata support in the existing passive viewer. The single-path scoring, shortlist, rigid placement, and physical thresholds remain unchanged.

**Tech Stack:** Python 3.10, NumPy, PyTorch, MuJoCo, standard-library multiprocessing and JSON, existing GRAIL corpus helpers, `unittest`.

## Global Constraints

- Evaluate exactly 11 scene-Y lanes from `-1.0 m` through `+1.0 m` inclusive at `0.20 m` spacing.
- Use scene-X endpoints `-1.2795985755 m` and `1.2590216406 m` for every lane.
- Search all admitted height-grid GRAIL clips without source clip, family, frame, or transform overrides.
- Do not change the qualified single-path costs, placement thresholds, or joint trajectories per lane.
- Permit only global yaw, XY translation, and root-Z translation.
- Classify every lane as `full`, `partial`, or `infeasible`; never hide an uncovered interval.
- Playlist teleports are visualization boundaries, not motion transitions.

---

### Task 1: Pure grid and playlist contracts

**Files:**
- Create: `sonic/python/mm_sonic/torch_path_motion_grid.py`
- Create: `tests/python/test_sonic_torch_path_motion_grid.py`

**Interfaces:**
- Produces: `HorizontalGridLane`, `horizontal_grid_lanes(...)`, `classify_lane(...)`, and `build_grid_playlist(...)`.
- Consumes: connector arrays with `joint_position`, `root_position_world`, and `root_orientation_world_wxyz`.

- [ ] **Step 1: Write failing grid-construction tests**

```python
def test_horizontal_grid_has_exact_twenty_centimeter_lanes():
    lanes = horizontal_grid_lanes(
        start_x=-1.2795985755,
        stop_x=1.2590216406,
        minimum_y=-1.0,
        maximum_y=1.0,
        spacing_m=0.2,
    )
    self.assertEqual(len(lanes), 11)
    np.testing.assert_allclose([lane.center_y_m for lane in lanes],
                               np.linspace(-1.0, 1.0, 11))
```

Also test rejection of a spacing that does not exactly include both bounds,
stable lane identifiers, and `full`/`partial`/`infeasible` classification.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_sonic_torch_path_motion_grid
```

Expected: import failure because `torch_path_motion_grid` does not exist.

- [ ] **Step 3: Implement the minimal immutable grid contract**

Use frozen dataclasses, finite-value validation, integer-index lane generation,
and stable IDs such as `lane-neg-1p0`, `lane-zero-0p0`, and
`lane-pos-1p0`. Construct values as `minimum_y + index * spacing_m` to avoid
accumulated loop drift.

- [ ] **Step 4: Add failing playlist tests**

Test low-to-high lane ordering, configurable 20-frame start/end holds,
half-open frame ranges, exact connector inventory, unchanged motion frames,
and explicit teleport boundaries.

- [ ] **Step 5: Implement and verify playlist assembly**

`build_grid_playlist` must duplicate only endpoint poses for holds. It must not
interpolate between lanes. Return the concatenated arrays plus JSON-safe lane
frame ranges.

Run the Task 1 test module and the existing path-placement tests. Expected:
all pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_path_motion_grid.py \
  tests/python/test_sonic_torch_path_motion_grid.py
git commit -m "feat: define horizontal path grid playlists"
```

### Task 2: Shared-index grid evaluator

**Files:**
- Create: `resources/run_g1_horizontal_grid_coverage.py`
- Create: `tests/python/test_run_g1_horizontal_grid_coverage.py`
- Reuse: `resources/run_g1_path_motion_placement.py`

**Interfaces:**
- Consumes: the Task 1 grid API and existing private single-path helpers
  `_queries`, `_source_profiles`, `_row_to_window`,
  `_placement_shortlist_rows`, `_target_grid`, `_matcher_archive`, and
  `_minimal_dataset`.
- Produces: `_scan_grid_one`, `_trim_lane_pool`, `_classify_grid_result`, CLI
  `main`, per-lane artifacts, `grid-summary.json`, `grid-playlist.npz`, and
  `grid-playlist.json`.

- [ ] **Step 1: Write failing CLI and shared-extraction tests**

Assert that the CLI has no source identity option, defaults to `0.20 m`
spacing, and builds exactly 11 lanes. Use synthetic profiles to verify
`_scan_grid_one` calls raw-window extraction once while returning independent
costed rows for two distinct lane signatures.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_run_g1_horizontal_grid_coverage
```

Expected: import failure because the grid runner does not exist.

- [ ] **Step 3: Implement shared extraction and bounded candidate pools**

For one source descriptor, load profiles and extract windows once using the
union of event-count bounds. Score those windows independently for each
lane's query signatures and apply existing contact-count diversity retention.

The parent process must merge results into one bounded pool per lane. Every
256 source clips, trim each pool to:

```python
pool_limit = max(coarse_results, 4 * placement_shortlist)
```

using the existing cost, coverage, and blended shortlist orderings. Track
scanned clip and returned coarse-row counts separately.

- [ ] **Step 4: Write failing certification and output tests**

With tiny synthetic lane results, verify:

- complete coverage becomes `full`;
- explicit gaps become `partial`;
- no accepted placement becomes `infeasible`;
- every lane appears in `grid-summary.json`;
- partial and infeasible lanes are not silently counted as full;
- the playlist contains all full and partial connectors in lane order.

- [ ] **Step 5: Implement per-lane physical certification**

For each lane, use the same 800-row shortlist policy and
`place_raw_window_on_path`. Preserve the current threshold constants and sort
accepted placements by uncovered length, coarse cost, source identity, and
start frame. Cache source profiles and sole points within each lane
certification pass without changing the selected result.

Write the selected connector in the target matcher's coordinate frame. If no
candidate certifies, emit an infeasible lane report without a connector.

- [ ] **Step 6: Implement aggregate artifacts and verify GREEN**

Write deterministic sorted JSON, build the Task 1 playlist, hardlink the
minimal target dataset, and print the aggregate full/partial/infeasible counts.

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_run_g1_horizontal_grid_coverage \
  tests.python.test_sonic_torch_path_motion_grid \
  tests.python.test_run_g1_path_motion_placement \
  tests.python.test_sonic_torch_path_motion_placement
```

Expected: all pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add resources/run_g1_horizontal_grid_coverage.py \
  tests/python/test_run_g1_horizontal_grid_coverage.py
git commit -m "feat: evaluate rigid motions across horizontal grid"
```

### Task 3: Colored-grid playlist viewer

**Files:**
- Modify: `resources/run_g1_stair_pivot_viewer.py`
- Modify: `tests/python/test_run_g1_stair_pivot_viewer.py`

**Interfaces:**
- Consumes: optional `--grid-summary` and `--playlist-metadata` JSON paths.
- Produces: `_load_grid_overlay`, `_grid_line_colors`, and
  `_set_grid_markers`.

- [ ] **Step 1: Write failing parser and metadata tests**

Verify the parser accepts both grid files, rejects use of only one, validates
all 11 lanes and playlist frame ranges, and maps:

```text
full -> green
partial -> amber
infeasible -> red
current accepted lane -> white
```

- [ ] **Step 2: Run the viewer tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_run_g1_stair_pivot_viewer
```

Expected: parser or import failure for the missing grid interfaces.

- [ ] **Step 3: Implement optional grid loading and drawing**

Use `viewer.user_scn` and `mujoco.mjv_connector(... mjGEOM_CAPSULE ...)` to
draw each requested line slightly above its sampled terrain. Update colors
when the current playlist lane changes. Preserve existing behavior when grid
arguments are absent.

The overlay text must show lane ID, center Y, classification, and lane number.
Frames between accepted lane ranges are hold or teleport boundaries and must
not be interpolated.

- [ ] **Step 4: Verify viewer regression tests and commit**

Run the viewer tests plus the prior viewer/contact-sheet tests. Expected: all
pass.

```bash
git add resources/run_g1_stair_pivot_viewer.py \
  tests/python/test_run_g1_stair_pivot_viewer.py
git commit -m "feat: visualize horizontal grid coverage"
```

### Task 4: Full grid experiment and evidence

**Files:**
- Create: `docs/superpowers/results/2026-08-05-g1-horizontal-grid-coverage.md`
- Generate, do not commit: `build/g1-horizontal-grid-20cm-v1/`
- Generate, do not commit: `build/g1-horizontal-grid-20cm-v2/`

**Interfaces:**
- Consumes: the Task 2 runner and Task 3 viewer.
- Produces: deterministic grid evidence and a live playback command.

- [ ] **Step 1: Run the full 11-lane evaluation**

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_horizontal_grid_coverage.py \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --target-scene grail-stair_p1-db7949fce1b2e48d39f4 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --start-x -1.2795985755 \
  --stop-x 1.2590216406 \
  --minimum-y -1.0 \
  --maximum-y 1.0 \
  --spacing-m 0.20 \
  --workers 32 \
  --coarse-results 2000 \
  --placement-shortlist 800 \
  --output build/g1-horizontal-grid-20cm-v1
```

Inspect every lane classification, coverage interval, selected raw identity,
and physical metric. Do not tune the algorithm between lanes.

- [ ] **Step 2: Render and visually inspect the playlist**

Render a contact sheet and then launch the live viewer with the grid summary
and playlist metadata. Confirm all 11 lines are visible, colors match the
report, accepted lanes play low-to-high, and no interpolation occurs at
teleports.

- [ ] **Step 3: Run an independent deterministic repeat**

Repeat the identical command to `build/g1-horizontal-grid-20cm-v2`. Compare
the SHA-256 hashes of `grid-summary.json`, `grid-playlist.json`, and
`grid-playlist.npz`. Expected: byte-identical.

- [ ] **Step 4: Run final verification**

Run all new tests plus the 50 qualified path/contact tests, `py_compile`, and
`git diff --check`. Record exact commands and outputs.

- [ ] **Step 5: Write and commit the evidence report**

Report honest full/partial/infeasible counts, total arc coverage, per-lane
sources and gaps, deterministic hashes, runtime, and visual inspection. State
that this evaluates parallel horizontal lanes only.

```bash
git add docs/superpowers/results/2026-08-05-g1-horizontal-grid-coverage.md
git commit -m "docs: record horizontal grid coverage"
```
