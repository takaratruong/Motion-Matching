# Task 5 implementation report

Status: complete and verified.

Commit: `1cecd7a665e4087e5161d75c31d154ec285092a8`

Commit message: `feat: visualize table and ground pickup searches`

## RED evidence

Added the viewer source-contract assertions before changing production code:

- `interaction_trajectory_database.h` is included;
- `load_trajectory_database(` is used and `interaction::load_database(` is absent;
- the viewer references both `SupportKind::Table` and `SupportKind::Ground`;
- the exact `GROUND` and `TABLE` labels are present and selected support is
  rendered;
- the viewer target depends on the compact-loader header; and
- `MIXED_INTERACTION_PACK` and `mixed-interaction-pack` remain exposed by the
  Makefile.

The required RED command was run before production changes:

```text
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
```

It failed with the expected two source-contract failures:

1. `#include "interaction_trajectory_database.h"` was absent from
   `hand_trajectory_viewer.cpp` (the viewer was still using the legacy full
   loader).
2. `interaction_trajectory_database.h` was absent from the
   `hand_trajectory_viewer` Makefile dependency list.

The pre-existing assertions for Enter-only reruns, selected-only full-pose
caching, background path stride, object/table collision, visible table/grid,
and forbidden controller/mesh/terrain/diffusion/screenshot dependencies all
remained green during RED.

## Implementation

- The viewer now includes `interaction_trajectory_database.h` and loads
  `interaction_database.bin` with `interaction::load_trajectory_database`.
- Canonical startup records the first Contact clip as a fallback, but returns
  the first Contact clip classified as `SupportKind::Table`. Support kind is
  not used in `select_hand_trajectories` membership or its cost/clip ordering.
- `support_name` returns exactly `GROUND` for ground support and `TABLE`
  otherwise. The selected option HUD now shows support beside `SAFE` and phase.
- `pack_path` retains its argv and `MM_INTERACTION_PACK` selection; no combined
  pack path was hardcoded in C++. The existing full-pack default was left
  unchanged.
- The viewer Make target now depends on `interaction_trajectory_database.h`.
- The Task 2 `MIXED_INTERACTION_PACK` variable and `mixed-interaction-pack`
  target were retained without changing the full-pack configuration.

## Changed files

- `hand_trajectory_viewer.cpp`
- `tests/python/test_hand_trajectory_viewer.py`
- `Makefile`

## GREEN and final verification

After the focused implementation, the viewer contract suite passed all seven
tests:

```text
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
```

The complete required verification sequence then exited 0:

```text
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_trajectory_database \
  build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
```

The C++ test binaries built and ran successfully. The release-mode viewer built
successfully; its Make prerequisite rebuilt the ignored local raylib dependency.
`git diff --check` exited 0.

Before committing, the staged-file audit showed exactly the three owned files,
and `git diff --cached --check` exited 0.

## Caveats

- The viewer was compiled but not interactively launched; the task’s requested
  checks do not provide a display-backed runtime test.
- This report is intentionally uncommitted. The worktree also retains the
  pre-existing, intentionally uncommitted `.superpowers/sdd/task-4-report.md`
  change; it was neither staged, reverted, nor altered.

## Important-finding follow-up: ground-only support startup

Commit: `6ff59c8d346faa26f8892a8e034f8eae8238849b`

Commit message: `fix: support ground-only trajectory packs`

### Root cause

The first-Contact fallback correctly selected a Ground clip when no Table clip
was available, but startup unconditionally passed that clip's virtual-floor
metadata to `make_recorded_table_geometry`. The builder requires a tabletop
with a positive underside height, so the virtual floor was rejected and startup
terminated.

A direct `ShelfGeometry{}` substitute would not be valid either: the shared
type contains a fixed array of five boxes, and the feasibility evaluator rejects
default zero-dimension boxes. Within Task 5's owned-file boundary, the viewer
therefore represents absent support geometry as `std::nullopt` and supplies
valid far-away no-op boxes only at the fixed-size feasibility API boundary.
Those boxes are not rendered and do not create a floor/table obstacle in the
scene; the evaluator's unchanged object-collision pass remains active.

### Follow-up RED

Added `test_ground_support_uses_no_table_geometry` before changing production
code. The contract requires:

- a focused `support_geometry` helper returning optional geometry;
- an exact Ground branch returning `std::nullopt` before the recorded-table
  builder;
- a no-support collision bridge for the fixed-size evaluator API;
- use of the optional geometry in collision setup; and
- rendering guarded by `table_geometry.has_value()`.

The focused RED command was:

```text
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
```

It ran eight tests and failed exactly once because
`std::optional<ShelfGeometry> support_geometry(` was absent. The seven existing
viewer contracts remained green.

### Follow-up fix

- `support_geometry` preserves the exact recorded-table builder path for Table
  support and returns `std::nullopt` for Ground support.
- Ground startup no longer constructs recorded geometry from virtual-floor
  metadata and therefore cannot hit the table underside validation.
- Table collision and rendering continue to consume the same five recorded
  boxes. Ground rendering leaves `DrawGrid` visible and draws no support boxes.
- Ground feasibility uses valid no-op boxes one million metres from the current
  query anchor, preserving the unchanged object collision pass while applying
  no table/floor collision obstacle.
- Support kind is still absent from trajectory membership and cost/clip ranking.
  Compact loading, staged controls, and selected-pose caching are unchanged.

Follow-up changed files:

- `hand_trajectory_viewer.cpp`
- `tests/python/test_hand_trajectory_viewer.py`

`Makefile` required no follow-up change.

### Follow-up GREEN and verification

The focused suite passed all eight tests after the fix. The complete Task 5
verification sequence then exited 0:

```text
python3 -m unittest tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_trajectory_database \
  build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
```

Both C++ tests rebuilt and ran successfully, the release viewer rebuilt
successfully, and `git diff --check` reported no whitespace errors. Before the
follow-up commit, the staged-file audit contained exactly the two changed owned
files and `git diff --cached --check` exited 0.

The follow-up was compile- and contract-verified but not interactively launched
under a display. This report remains intentionally uncommitted, alongside the
untouched intentional Task 4 report modification.
