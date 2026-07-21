# Trained-Object Entry Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend learned Smart Pickup's coarse route to 3.0 metres and evaluate the pinned checkpoint across 72 entry poses around its trained object.

**Architecture:** `FunnelCaptureConfig` owns a learned-only 3.0-metre object-radius gate and collision-checked route budget while preserving its 0.45-to-1.00-metre handoff annulus. A Python evaluation tool reuses the production sampler, projection, and certification functions, varying only the six entry-condition values and writing deterministic JSON evidence.

**Tech Stack:** C++17, Python 3.10, NumPy 2.2.6, PyTorch 2.5.1, GNU Make, existing learned-funnel checkpoint and training pack.

## Global Constraints

- Learned coarse pickup range is exactly 3.0 metres.
- Diffusion handoff remains inside the inclusive 0.45-to-1.00-metre annulus.
- Authored Smart Pickup and `MatchConfig::maximum_approach_m` remain exactly 1.0 metre.
- Evaluate only the object/grasp prefix from `build/smart-pickup/full-pack`; do not claim unseen-object generalization.
- Sweep eight bearings, three radii (0.55, 0.75, 0.95 metres), and three facing offsets (-15, 0, +15 degrees), for exactly 72 conditions.
- Sample exactly 32 proposals per condition using the pinned checkpoint and production projection/certification.
- Use `build/venvs/g1-funnels/bin/python` for checkpoint evaluation.
- Preserve unrelated changes, especially `resources/features.bin` and generated build directories.

---

### Task 1: Extend learned coarse capture to 3.0 metres

**Files:**

- Modify: `interaction_funnel_capture.h`
- Modify: `tests/cpp/test_interaction_funnel_capture.cpp`

**Interfaces:**

- Consumes: `select_funnel_capture(Transform, InteractionTarget, obstacles, FunnelCaptureConfig)`.
- Produces: a default `FunnelCaptureConfig` whose activation object radius and route budget are 3.0 metres while `minimum_object_radius_m` and `maximum_object_radius_m` remain 0.45 and 1.00.

- [ ] **Step 1: Write the failing range-boundary test**

Add a test that exercises the real default configuration:

```cpp
void test_default_route_reaches_three_metres_but_not_farther() {
    const auto at_limit = interaction::select_funnel_capture(
        root_at(0.0F, 3.0F), target_at_origin(), {});
    assert(at_limit.reason == interaction::PickSlotReason::None);
    assert(std::abs(at_limit.target_world.position.z - 1.0F) < 1.0e-6F);

    const auto beyond = interaction::select_funnel_capture(
        root_at(0.0F, 3.01F), target_at_origin(), {});
    assert(beyond.reason == interaction::PickSlotReason::OutsideTravelEnvelope);
}
```

Call it from `main()` and retain the existing annulus-clamping tests.

- [ ] **Step 2: Run RED**

Run:

```bash
make build/tests/test_interaction_funnel_capture
build/tests/test_interaction_funnel_capture
```

Expected: abort at the `at_limit` assertion because the current default route limit is 1.0 metre.

- [ ] **Step 3: Implement the learned-only default**

Give `FunnelCaptureConfig` an explicit constructor without modifying `PickSlotConfig`:

```cpp
struct FunnelCaptureConfig {
    float minimum_object_radius_m = 0.45F;
    float maximum_object_radius_m = 1.00F;
    float maximum_activation_object_radius_m = 3.00F;
    PickSlotConfig route{};

    FunnelCaptureConfig() {
        route.maximum_direct_travel_m = 3.00F;
    }
};
```

In `select_funnel_capture`, validate that the activation radius is finite and
at least `maximum_object_radius_m`, then return `OutsideTravelEnvelope` when
the live root's planar object radius exceeds it before enumerating candidates.

- [ ] **Step 4: Run GREEN and related learned-backend tests**

Run:

```bash
make build/tests/test_interaction_funnel_capture \
  build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_funnel_capture
build/tests/test_interaction_learned_pickup_backend
```

Expected: both binaries exit zero.

- [ ] **Step 5: Commit the range change**

Stage only the two listed files and commit:

```bash
git add interaction_funnel_capture.h tests/cpp/test_interaction_funnel_capture.cpp
git commit -m "feat: extend learned pickup capture range"
```

---

### Task 2: Add deterministic trained-object entry sweep

**Files:**

- Create: `tools/evaluate_g1_funnel_entry_sweep.py`
- Create: `tests/python/test_g1_funnel_entry_sweep.py`

**Interfaces:**

- Consumes: `load_dataset(pack)`, `sample_checkpoint(checkpoint, conditions, seed=...)`, `project_object_local_funnels(proposals)`, and `certify_proposals(proposals)`.
- Produces: `build/g1-funnels/evidence/trained-object-entry-sweep.json` with one summary and 72 per-condition records.

- [ ] **Step 1: Write failing condition-grid tests**

Import `build_entry_conditions` from the new tool and require the exact grid:

```python
base = np.arange(24, dtype=np.float32)
conditions, descriptors = build_entry_conditions(base)
self.assertEqual(conditions.shape, (72, 24))
self.assertEqual(len(descriptors), 72)
np.testing.assert_array_equal(
    conditions[:, :18], np.broadcast_to(base[:18], (72, 18)))
self.assertEqual({row["radius_m"] for row in descriptors}, {0.55, 0.75, 0.95})
self.assertEqual({row["bearing_degrees"] for row in descriptors}, set(range(0, 360, 45)))
self.assertEqual({row["yaw_offset_degrees"] for row in descriptors}, {-15, 0, 15})
self.assertTrue(np.array_equal(conditions[:, 22:24], np.zeros((72, 2), np.float32)))
```

Also verify every `(x,z)` radius and that every yaw faces the origin plus the declared offset.

- [ ] **Step 2: Write failing report tests with an injected sampler**

Use a deterministic fake sampler that returns `[72,32,16,4]`, ending every outward route at `condition[18:22]`. Require `evaluate_entry_sweep` to reverse once, project, preserve execution sample zero bit-exactly, certify each batch, and produce stable JSON-compatible dictionaries containing:

```python
{
    "condition_count": 72,
    "proposal_count_per_condition": 32,
    "passing_condition_count": 72,
    "minimum_accepted_proposals": 32,
}
```

Mutate one fake entry boundary and require an `entry_boundary_mismatch` failure rather than silently accepting it.

- [ ] **Step 3: Run RED**

Run:

```bash
build/venvs/g1-funnels/bin/python -m unittest -v \
  tests.python.test_g1_funnel_entry_sweep
```

Expected: import failure because `tools/evaluate_g1_funnel_entry_sweep.py` does not exist.

- [ ] **Step 4: Implement condition generation and evaluation**

Implement:

```python
def build_entry_conditions(base_condition: np.ndarray) \
        -> tuple[np.ndarray, list[dict[str, float]]]: ...

def evaluate_entry_sweep(
    checkpoint: Path,
    base_condition: np.ndarray,
    *,
    seed: int = 2026071901,
    device: str = "cpu",
    sampler=sample_checkpoint,
) -> dict: ...
```

Generate positions with `x = radius * sin(bearing)` and
`z = radius * cos(bearing)`. Compute facing yaw with
`atan2(-x, -z) + radians(yaw_offset)`, store sine/cosine in indices 20 and
21, and store zero velocity in 22 and 23. Sample all 72 conditions in one
checkpoint call, reverse outward routes once, project each 32-proposal batch,
and certify with production functions. Compare execution sample zero to the
broadcast condition entry with `np.array_equal` before computing metrics.

The CLI accepts `--pack`, `--checkpoint`, `--output`, `--seed`, and `--device`.
It loads `load_dataset(pack).conditions[0]`, writes sorted/indented JSON, and
prints passing conditions and minimum accepted count.

- [ ] **Step 5: Run GREEN and existing proposal tests**

Run:

```bash
build/venvs/g1-funnels/bin/python -m unittest -v \
  tests.python.test_g1_funnel_entry_sweep \
  tests.python.test_diffusion_model \
  tests.python.test_proposal_artifact
```

Expected: all tests pass.

- [ ] **Step 6: Commit the evaluator**

```bash
git add tools/evaluate_g1_funnel_entry_sweep.py \
  tests/python/test_g1_funnel_entry_sweep.py
git commit -m "test: evaluate trained-object funnel entries"
```

---

### Task 3: Run real checkpoint evidence and restore the live visualizer

**Files:**

- Generate: `build/g1-funnels/evidence/trained-object-entry-sweep.json`
- Do not commit generated checkpoint or evidence files.

**Interfaces:**

- Consumes: `build/g1-funnels/checkpoint-schema3-20k.pt` and `build/smart-pickup/full-pack`.
- Produces: quantitative proposal-stage evidence and a rebuilt learned controller ready for live trained-object testing.

- [ ] **Step 1: Run the real sweep**

Run:

```bash
build/venvs/g1-funnels/bin/python tools/evaluate_g1_funnel_entry_sweep.py \
  --pack build/smart-pickup/full-pack \
  --checkpoint build/g1-funnels/checkpoint-schema3-20k.pt \
  --output build/g1-funnels/evidence/trained-object-entry-sweep.json \
  --device cpu
```

Expected: 72 evaluated conditions and a report showing the actual pass count;
do not convert failures into a passing claim.

- [ ] **Step 2: Run focused regression verification**

Run:

```bash
make build/tests/test_interaction_funnel_capture \
  build/tests/test_interaction_learned_pickup_backend \
  build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_funnel_capture
build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_learned_pickup_end_to_end
build/venvs/g1-funnels/bin/python -m unittest -v \
  tests.python.test_g1_funnel_entry_sweep
```

Expected: all binaries and Python tests exit zero.

- [ ] **Step 3: Rebuild and restart learned mode**

Build `controller`, stop only the currently identified learned-controller PID,
and restart it on `DISPLAY=:1` with the same pinned Python worker, checkpoint,
and work directory environment. Focus the new learned window and confirm two
screenshots one second apart differ.

- [ ] **Step 4: Report evidence honestly**

Report the real sweep pass count, minimum/median accepted proposals, worst
entry poses, the exact checkpoint path, and whether the visualizer is advancing.
Explicitly state that this establishes entry-pose proposal generality for the
trained object only; live pickup trials still establish controller/contact
success.
