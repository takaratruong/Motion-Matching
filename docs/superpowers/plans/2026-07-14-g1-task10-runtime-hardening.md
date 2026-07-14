# G1 Task 10 Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement task-by-task, with a
> specification review and code-quality review at each commit boundary.

**Goal:** Make the remaining Gate C/D acceptance evidence describe a stable,
walkability-correct 25 Hz G1 runtime, then certify Task 10 without relaxing any
numerical threshold.

**Architecture:** Use Holden's existing transition-cost seam for settled-idle
hysteresis; reserve and stop only planar blocked dynamics; post-process the
four raw terrain samples through the walkability sweep; reject toe support on
class-0 surfaces; and correct checker predicates to compare coherent pipeline
stages and consecutive qualifying runs.

**Tech stack:** C++17 header-only helpers, Daniel Holden's C++ controller and
database search, standalone GCC test executables, Python `unittest`, Raylib
route playback, CSV acceptance evidence.

## Global constraints

- Runtime remains fixed at binary32 `0.04 s` (25 Hz).
- Preserve the exact 31D feature layout, database contents, terrain weights,
  IK/LMM-disabled gates, support halflives, and published acceptance numbers.
- Never stage or rewrite `resources/database.bin` or `resources/features.bin`.
  Required hashes are:
  - `28207d915847baac7a93e63545577b277f7c4300d839782eff48260b358b17d3`
  - `e6564503e214be36d3b7912efc19997a9222d85ca83f7227c0b1e7cedce8c73e`
- Preserve the existing dirty Task 10 Python files while C++ commits are made;
  do not stage them until final Task 10 acceptance.
- Keep visualizer PID `1697070` alive until all Task 10 gates and final review
  pass. Build and run candidates under `/tmp`.
- Robot meshes remain out of scope until the entire planned test program is
  complete.
- Treat the isolated probe worktree as diagnostic evidence only; implement
  production changes test-first in the main worktree.

---

### Task 1: Settled-idle matcher hysteresis

**Files:**

- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `tests/cpp/test_terrain_database.cpp`
- Modify: `g1_controller_state.h`
- Modify: `controller.cpp`

- [ ] Add `test_idle_match_transition_cost_policy` before implementation.
  Cover zero, exact `1e-4/0.05` boundaries, `nextafter` above either boundary,
  active-command/stopped-simulation, idle-command/moving-simulation, negative,
  NaN, and infinity. Only a valid settled pair returns exact `1.0f`; all other
  cases return exact `0.0f`.

- [ ] Add a controller source-order test proving the helper consumes
  `traversal.commanded_speed` and
  `walkability_xz_length(state.simulation_velocity)`, and its value is passed
  as the fifth `database_search` argument.

- [ ] Add database characterization coverage showing: explicit zero matches
  the default bit-for-bit; a weak improvement transitions at zero but not with
  cost one; an improvement greater than one still transitions. Include the
  end-of-animation `best_index=-1` case so the margin cannot strand the runtime.

- [ ] Compile/run the focused tests and record RED before production edits:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_state_idle_red
/tmp/test_g1_state_idle_red
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database_idle_red
/tmp/test_terrain_database_idle_red
```

- [ ] Implement stateless `g1_idle_match_transition_cost` in
  `g1_controller_state.h`. Wire the exact raw-command and planar-simulation
  inputs immediately before ordinary `database_search`. Do not add controller
  state, skip searches, change timers, or touch `database.h`.

- [ ] Run both focused tests in strict and `-O3 -ffast-math -DNDEBUG`
  configurations. Run the controller strict build listed under Task 6.

- [ ] Review the diff, then commit only these four files:

```bash
git add g1_controller_state.h controller.cpp \
  tests/cpp/test_g1_controller_state.cpp tests/cpp/test_terrain_database.cpp
git diff --cached --check
git commit -m "fix: stabilize settled motion matching"
```

---

### Task 2: Blocked safe-stop dynamics

**Files:**

- Modify: `tests/cpp/test_terrain_runtime.cpp`
- Modify: `terrain_runtime.h`
- Modify: `controller.cpp`

- [ ] Add a focused test for a pure planar-stop helper. A blocked diagnostic
  with `applied_speed <= 1e-4` must write positive zero only to velocity XZ and
  acceleration XZ. Exact Y bits, including signed zero/NaN payload fixtures,
  must remain unchanged. Clear diagnostics or speed just above the boundary
  must preserve every bit.

- [ ] Extend the controller data-flow regression to require the helper after
  `traversability_limit_command` and before trajectory prediction/matching.

- [ ] Add or update the command-limit fixture so the named blocked reserve is
  exactly `0.04f`, while class-1 recovery and invalid-input behavior remain
  unchanged. Prove RED before implementation.

- [ ] Implement the named `0.04f` reserve and planar-stop helper. Wire it at the
  approved command stage. Leave the existing post-integration
  preflight/application guard intact.

- [ ] Run `test_terrain_runtime` in debug, strict, release, and ASan/UBSan
  configurations. Review and commit only the three owned files:

```bash
git add terrain_runtime.h controller.cpp tests/cpp/test_terrain_runtime.cpp
git diff --cached --check
git commit -m "fix: stop blocked planar coasting"
```

---

### Task 3: Walkability-aware terrain query

**Files:**

- Modify: `tests/cpp/test_terrain_runtime.cpp`
- Modify: `terrain_runtime.h`
- Modify: `controller.cpp`

- [ ] Add RED tests for transactional
  `terrain_centerline_snapshot_apply_walkability_v2`:
  - a blocked raised wall preserves earlier clear entries and repeats the
    first last-safe point/height for it and all later entries;
  - a blocked rising ramp behaves the same;
  - identical class-1 and class-2 geometry remains bit-identical in all values
    and points;
  - a stationary forward snapshot remains filtered and a fresh away-facing
    snapshot restores raw values;
  - mismatched grid, nonfinite roots, invalid radius, and malformed snapshot
    return false without modifying the caller's snapshot.

- [ ] Extend the source-order regression to require raw v2 snapshot compute,
  walkability postprocessing, finite validation, and 31D query copy in that
  order.

- [ ] Implement the stateless postprocessor. Use animation root only as the
  height-feature base, simulation position as the footprint origin, sequential
  `walkability_sweep` segments, and the existing `0.20f` footprint. Latch on
  class 0/out-of-bounds; keep class 1/2 exact.

- [ ] Run the four `test_terrain_runtime` configurations, strict controller
  build, and a one-frame smoke for every scene. Review and commit:

```bash
git add terrain_runtime.h controller.cpp tests/cpp/test_terrain_runtime.cpp
git diff --cached --check
git commit -m "fix: mask inaccessible terrain queries"
```

---

### Task 4: Walkability-filtered support contacts

**Files:**

- Modify: `tests/cpp/test_support_runtime.cpp`
- Modify: `tests/cpp/test_support_matching.cpp` (only the existing controller
  call/order regression for the renamed walkability-aware builder)
- Modify: `support_runtime.h`
- Modify: `controller.cpp`

- [ ] Add RED tests for `support_observation_build_walkable`:
  - class-1/class-2 toe contacts and all height/delta bits match the ordinary
    builder;
  - a class-0 left or right toe has only that contact cleared;
  - both class-0 toes clear both contacts, leaving support update to its
    existing hold/root fallback;
  - an invalid/mismatched grid cannot create a valid toe contact;
  - failure remains transactional.

- [ ] Add a controller source regression proving the walkability wrapper uses
  `active_scene.walkability` and feeds `support_frame_update` before support is
  applied once to the final pose. Update the existing exact-call count in
  `test_support_matching.cpp` so it recognizes the approved wrapper without
  weakening its single-build and stage-order checks.

- [ ] Implement the wrapper without changing raw support heights/deltas or
  `support_frame_update`. Use the filtered observation in the controller so
  runtime-log contact bits describe the contacts that actually drive support.

- [ ] Run `test_support_runtime` in debug, strict, release, and ASan/UBSan
  configurations, plus strict controller build. Review and commit:

```bash
git add support_runtime.h controller.cpp tests/cpp/test_support_runtime.cpp \
  tests/cpp/test_support_matching.cpp
git diff --cached --check
git commit -m "fix: reject inaccessible toe support"
```

---

### Task 5: Correct Gate C and mixed-level acceptance stages

**Files:**

- Modify existing dirty file: `resources/check_g1_runtime_log.py`
- Modify existing dirty file: `tests/python/test_runtime_log.py`
- Do not commit until Task 6 runtime evidence passes

- [ ] Add/adjust synthetic tests before checker edits:
  - a raw-source Hips jump with smooth/equal final stages passes;
  - all three equal final stages stepping over `0.05 m` fails as rendered Hips;
  - landing alignment errors leaving exactly 50 consecutive combined
    height/alignment rows pass; leaving 49 or splitting the run fails;
  - mixed endpoint XZ and support span are independently mandatory;
  - `2.95 m` support span passes, below `2.95 m` fails;
  - 50 consecutive elevated matching/aligned rows pass, 49 or a splitting
    alignment error fails;
  - flat source metadata is accepted by standalone mixed checking while the
    existing global Gate C terrain-source activation test still rejects an
    entirely flat/no-activation route.

- [ ] Run the targeted tests and record RED. Then implement:
  - `abs(delta(rendered_hips_y)) <= 0.05` with stage agreement unchanged;
  - one longest consecutive block satisfying both landing-height and
    contact-aware alignment predicates;
  - ordered geometric endpoint crossings, explicit support span `>=2.95`, and
    an elevated matching/alignment run without a terrain-label predicate.

- [ ] Run the complete checker suite:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
```

Expected: all inherited and new tests pass. Do not stage the Python files yet.

---

### Task 6: Full verification and Task 10 acceptance

- [ ] Run focused C++ tests in strict and production configurations:

```bash
for test in test_g1_controller_state test_terrain_database \
  test_terrain_runtime test_support_runtime test_support_matching; do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/${test}.cpp" -o "/tmp/${test}_strict"
  "/tmp/${test}_strict"
  g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
    "tests/cpp/${test}.cpp" -o "/tmp/${test}_release"
  "/tmp/${test}_release"
done
```

- [ ] Build the fresh controller strictly, without replacing PID `1697070`:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -Wno-unused-result -Wno-class-memaccess -Wno-sign-compare \
  -Wno-pedantic -Wno-unused-parameter -Wno-missing-field-initializers \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_g1_task10_certified \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
kill -0 1697070
```

- [ ] Generate fresh weight-0 and weight-4, 800-frame logs for all six Gate C
  routes with the certified binary. Run `--gate-c --compare-control` on each.
  Generate fresh 600-frame logs for both blocked routes and run `--gate-d`.
  Every controller and checker call must exit zero.

- [ ] Quantitatively preserve the architecture:
  - active-command rows use transition cost zero and retain expected search;
  - every Gate C route has lower weight-four terrain error;
  - every landing has at least 50 consecutive aligned rows and rendered Hips
    steps at most `0.05 m`;
  - mixed spans at least `2.95 m`, crosses both endpoints, matches on its
    elevated level, traverses three plateaus/ramp, and returns to base;
  - both Gate D routes stop at least `0.02 m` short for at least 25 rows with
    support rise at most `0.02 m` and no class-0 footprint;
  - no short sub-stride loop remains.

- [ ] Recheck protected hashes, PID, `git diff --check`, and the full test
  suite. Request independent spec and quality review of all commits plus the
  uncommitted checker diff.

- [ ] If and only if every gate/review is clean, stage the complete Task 10
  checker/test diff and any reviewed final wiring, then commit:

```bash
git add resources/check_g1_runtime_log.py tests/python/test_runtime_log.py
git diff --cached --check
git commit -m "test: enforce G1 multilevel and safe-stop gates"
```

- [ ] Update `.superpowers/sdd/progress.md` and the Task 10 evidence report.
  Only then build the user-facing binary, terminate PID `1697070`, and launch
  the certified mixed-multilevel visualizer on `DISPLAY=:1`. Verify the new PID
  and window remain alive before reporting readiness.
