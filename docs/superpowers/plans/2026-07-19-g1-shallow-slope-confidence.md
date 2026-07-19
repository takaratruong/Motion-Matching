# G1 Shallow-Slope Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make certified 5-degree longitudinal and cross slopes select G1 slope motion banks without weakening curb/stair hysteresis or fabricating quality evidence.

**Architecture:** Keep the existing terrain-family classifier and shared two-frame `0.60` bank transition state machine. Recalibrate only the mirrored Python/C++ slope-confidence normalization constant from 10 degrees to the shallowest certified 5-degree slope, then require both unit parity and a real bounded controller trace.

**Tech Stack:** Python 3 `unittest`/NumPy, C++17 header-only runtime tests, Holden/raylib controller, deterministic 25 Hz CSV telemetry.

## Global Constraints

- Keep `MOTION_BANK_TRANSITION_MIN_CONFIDENCE == 0.60` and `MOTION_BANK_TRANSITION_REQUIRED_FRAMES == 2` unchanged.
- Keep flat classification limits at 2 degrees and 0.04 m unchanged.
- Keep curb/stair confidence rules, database/artifact schemas, scenes, headings, and IK unchanged.
- Python and C++ classifier behavior must remain mirrored.
- Remove the synthetic ramp-05 `.624` confidence override; tests must use classifier output.
- Push the accepted checkpoint to `checkpoint/g1-playable-mesh`.

---

### Task 1: Lock shallow-slope confidence parity

**Files:**
- Modify: `tests/python/test_terrain.py`
- Modify: `tests/cpp/test_motion_bank_runtime.cpp`
- Modify: `resources/g1_terrain_builder/terrain.py`
- Modify: `motion_bank_runtime.h`
- Modify: `tests/python/test_g1_motion_quality.py`

**Interfaces:**
- Consumes: `classify_terrain_profile(distances, heights, normals) -> TerrainClassification`; `motion_bank_classify_profile(...) -> bool`.
- Produces: mirrored `*_CONFIDENCE_REFERENCE_SLOPE_DEGREES == 5.0` behavior, with 3-degree confidence at the existing `0.60` transition boundary and 5-degree confidence near `1.0`.

- [ ] **Step 1: Add the failing Python boundary test and remove fabricated fixture confidence**

Add this method to `TerrainTests` in `tests/python/test_terrain.py`:

```python
def test_certified_five_degree_slope_owns_confident_bank_boundary(self):
    cases = ((2.0, "flat"), (3.0, "slope"), (5.0, "slope"))
    results = {}
    for degrees, family in cases:
        grade = math.tan(math.radians(degrees))
        result = classify_terrain_profile(*_dense_profile(
            lambda distance, grade=grade: grade * distance, grade))
        self.assertEqual(result.family, family)
        results[degrees] = result
    self.assertGreaterEqual(results[3.0].confidence, 0.60 - 1e-12)
    self.assertGreater(results[5.0].confidence, 0.99)
```

Also delete this entire block from `production_route_rows` in
`tests/python/test_g1_motion_quality.py`:

```python
if (spec["id"] == "gate-c-ramp-05-forward" and
        classification.family == "slope" and confidence >= .499):
    confidence = .624
```

- [ ] **Step 2: Run the Python test and verify RED**

Run:

```bash
python3 -m unittest -v tests.python.test_terrain.TerrainTests.test_certified_five_degree_slope_owns_confident_bank_boundary
python3 -m unittest -v \
  tests.python.test_g1_motion_quality.ProductionRouteOracleRedTests.test_gate_c_and_mixed_checks_are_composed
```

Expected: both fail because the current 5-degree confidence is approximately
`0.50`; the quality route cannot activate a nonflat source without the removed
override.

- [ ] **Step 3: Add the failing C++ behavioral parity test**

In `tests/cpp/test_motion_bank_runtime.cpp`, add a helper that fills a
`profile_fixture` with a planar slope at an arbitrary degree angle, then add:

```cpp
static void test_certified_shallow_slope_confidence_boundary()
{
    profile_fixture profile;
    initialize_degree_slope_profile(profile, 5.0);
    motion_bank_classification classification = {};
    motion_bank_classification_status status =
        motion_bank_classification_invalid_value;
    check(motion_bank_classify_profile(
              classification, status, profile.distances,
              profile.heights, profile.normals, 51),
          "certified five-degree slope classifies");
    check(classification.family == MOTION_BANK_FAMILY_SLOPE &&
              classification.confidence > 0.99,
          "certified five-degree slope is transition-confident");

    initialize_degree_slope_profile(profile, 3.0);
    check(motion_bank_classify_profile(
              classification, status, profile.distances,
              profile.heights, profile.normals, 51),
          "three-degree slope classifies");
    check(classification.family == MOTION_BANK_FAMILY_SLOPE &&
              classification.confidence >=
                  MOTION_BANK_TRANSITION_MIN_CONFIDENCE - 1e-12,
          "three-degree slope reaches the existing transition boundary");
}
```

Call it from `main()` immediately after
`test_locked_python_classifier_oracles()`.

- [ ] **Step 4: Compile/run the C++ test and verify RED**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. \
  tests/cpp/test_motion_bank_runtime.cpp \
  -o /tmp/g1-motion-banks-phase-a/test-motion-bank-shallow-red
/tmp/g1-motion-banks-phase-a/test-motion-bank-shallow-red
```

Expected: abort with `certified five-degree slope is transition-confident`.

- [ ] **Step 5: Implement the mirrored calibration**

In `resources/g1_terrain_builder/terrain.py` set:

```python
# The shallowest certified ramp/cross-slope is the full-confidence anchor.
TERRAIN_CONFIDENCE_REFERENCE_SLOPE_DEGREES = 5.0
```

In `motion_bank_runtime.h` set:

```cpp
// The shallowest certified ramp/cross-slope is the full-confidence anchor.
static const double MOTION_BANK_CONFIDENCE_REFERENCE_SLOPE_DEGREES = 5.0;
```

- [ ] **Step 6: Run focused GREEN verification**

Run both Python commands from Step 2 and the C++ compile/run from Step 4.

Expected: both pass; 2 degrees remains `flat`, 3 degrees reaches `0.60`, and
5 degrees exceeds `0.99`.

- [ ] **Step 7: Run adjacent classifier suites**

Run:

```bash
python3 -m unittest -v tests.python.test_terrain
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. \
  tests/cpp/test_motion_bank_runtime.cpp \
  -o /tmp/g1-motion-banks-phase-a/test-motion-bank-shallow-green
/tmp/g1-motion-banks-phase-a/test-motion-bank-shallow-green
```

Expected: all Python terrain tests pass and C++ prints
`motion bank runtime tests passed`.

### Task 2: Certify runtime activation

**Files:**
- Modify: `tests/python/test_g1_motion_quality.py`
- Verify: `resources/check_g1_motion_quality.py`
- Build from: `controller.cpp`

**Interfaces:**
- Consumes: the Task 1 mirrored classifier constants and
  `PRODUCTION_ROUTE_SPECS` ramp-05 route.
- Produces: an honest ramp-05 quality fixture plus bounded CSV evidence with
  nonzero active and selected slope-bank frames.

- [ ] **Step 1: Verify the honest fixture is GREEN with Task 1**

Run:

```bash
python3 -m unittest -v \
  tests.python.test_g1_motion_quality.ProductionRouteOracleRedTests.test_gate_c_and_mixed_checks_are_composed
```

Expected: PASS and the test's existing assertion observes at least one
`source_family == "slope"` row.

- [ ] **Step 2: Build an exact release controller atomically**

Run:

```bash
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp \
  -o /tmp/g1-motion-banks-phase-a/controller-g1-banks.new \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
mv /tmp/g1-motion-banks-phase-a/controller-g1-banks.new \
  /tmp/g1-motion-banks-phase-a/controller-g1-banks
```

Expected: exit 0 and a regular executable output.

- [ ] **Step 3: Run the bounded ramp-05 controller trace**

Use the completed full candidate path when available; the compatible smoke pack
`/tmp/g1-terrain-banks-smoke` is acceptable for the focused classifier check:

```bash
env -i \
  G1_TERRAIN_DIR=/tmp/g1-terrain-banks-smoke \
  MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_TERRAIN_SCENE=ramp-05-up-down \
  MM_TEST_MODE=route MM_TEST_ROUTE=up-landing-down \
  MM_TEST_FRAMES=800 MM_TEST_HEADING=forward \
  MM_LOG=/tmp/g1-motion-banks-phase-a/ramp05-shallow-green.csv \
  MM_CLEANUP_LOG=/tmp/g1-motion-banks-phase-a/ramp05-shallow-green-cleanup.json \
  /tmp/g1-motion-banks-phase-a/controller-g1-banks
```

Expected: exit 0, 800 data rows, nonzero `active_family=slope` and
`source_family=slope` counts, and normal cleanup `model_load_count=1`,
`model_unload_count=1`, `live_model_count=0`.

- [ ] **Step 4: Run motion-quality and adjacent verification**

Run:

```bash
python3 -m unittest -v tests.python.test_g1_motion_quality
python3 -m py_compile \
  resources/check_g1_motion_quality.py \
  tests/python/test_g1_motion_quality.py
git diff --check
```

Expected: all tests pass, compilation exits 0, and no whitespace errors.

- [ ] **Step 5: Review, commit, and push the runtime calibration checkpoint**

Stage only the five implementation/test files, review `git diff --cached`, then:

```bash
git commit -m "fix: activate motion banks on shallow slopes"
git push checkpoint HEAD:refs/heads/g1-playable-mesh
git rev-parse HEAD checkpoint/g1-playable-mesh
```

Expected: the two printed commit IDs are identical.  Commit the independent
motion-quality checker separately after its full rereview if it is not yet
accepted.
