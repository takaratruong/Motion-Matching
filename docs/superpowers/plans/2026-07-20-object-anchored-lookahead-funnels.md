# Object-Anchored Lookahead Funnels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make learned Smart Pickup generate object-local approaches conditioned on the actual live entry and follow them monotonically without circling stale time-indexed waypoints.

**Architecture:** The worker samples 32 outward object-local trajectories and restores each outward entry endpoint exactly from the request condition. C++ maps the reversed execution route through the frozen object transform and uses a spatial lookahead follower whose progress can only move toward the terminal interaction root; the existing matcher and runtime retain contact and attachment authority.

**Tech Stack:** Python 3.10, NumPy 2.2.6, PyTorch 2.5.1, C++17, GNU Make, raylib controller, existing 25 Hz interaction runtime.

## Global Constraints

- Use `build/venvs/g1-funnels/bin/python` for model and artifact commands.
- Use only `build/smart-pickup/full-pack` for retraining.
- Keep the condition exactly 24 float32 values, with entry `(x,z,sin_yaw,cos_yaw,vx,vz)` in object coordinates at indices 18 through 23.
- Model and proposal samples are object-local `(x,z,sin_yaw,cos_yaw)`; no sample is transformed through the frozen entry root.
- Outward sample 15 is restored exactly from condition values 18 through 21 and reversed exactly once into execution sample 0.
- One attempt samples exactly 32 proposals once and never resamples while following.
- The target object transform and generation are frozen for the attempt; authority change cancels.
- The follower runs at 25 Hz but progress and completion are spatial, not tied to consuming one route sample per tick.
- Learned code publishes locomotion controls only. Existing matcher, Reach, Contact, attachment, Hold, Carry, placement, and release remain authoritative.
- Preserve unrelated changes, especially `resources/features.bin` and generated build directories.

---

### Task 1: Migrate training and proposal artifacts to object-local schema 3

**Files:**

- Modify: `resources/g1_interaction_builder/funnel_dataset.py`
- Modify: `resources/g1_interaction_builder/diffusion.py`
- Modify: `resources/g1_interaction_builder/proposal_artifact.py`
- Modify: `tools/run_g1_funnel_proposal_worker.py`
- Modify: `interaction_funnel_artifact.cpp`
- Modify: `tests/python/test_funnel_dataset.py`
- Modify: `tests/python/test_diffusion_model.py`
- Modify: `tests/python/test_proposal_artifact.py`
- Modify: `tests/python/test_proposal_request.py`
- Modify: `tests/cpp/test_interaction_funnel_artifact.cpp`
- Replace generated: `build/g1-funnels/checkpoint.pt`

**Interfaces:**

- Produces: schema-3 checkpoint and `G1FUNNL3` proposal artifact containing execution-order object-local trajectories.
- Consumes: the existing object-local entry condition at indices 18 through 23.

- [ ] **Step 1: Write failing object-local dataset and sampling tests**

Change the endpoint contract to require the outward endpoint to equal each row's object-local entry:

```python
dataset = funnel_dataset.load_dataset("build/smart-pickup/full-pack")
np.testing.assert_array_equal(
    dataset.funnels[:, 15], dataset.conditions[:, 18:22])

condition = torch.tensor(
    [[0.0] * 18 + [0.42, -0.17, 0.6, 0.8, 0.0, 0.0]],
    dtype=torch.float32,
)
samples = sample_checkpoint(checkpoint_path, condition, seed=11)
expected = condition[:, None, 18:22].expand(1, 32, 4)
self.assertTrue(torch.equal(samples[:, :, 15], expected))
```

Retain the existing common-world-transform invariance test for
`object_local_execution`.

- [ ] **Step 2: Write failing schema-3 artifact tests**

Set test conditions to a unit entry yaw and require every execution sample zero
to equal `condition[18:22]`. Require magic `G1FUNNL3`, schema version `3`, and
rejection of an artifact whose first sample differs by one float bit:

```python
condition[18:22] = (0.42, -0.17, 0.6, 0.8)
proposals[:, 0] = condition[18:22]
artifact = ProposalArtifact(
    3, 7, seed, digest, condition, proposals, seeds, accepted)
```

Mirror the exact contract in `test_interaction_funnel_artifact.cpp`.

- [ ] **Step 3: Run RED**

Run:

```bash
build/venvs/g1-funnels/bin/python -m unittest -v \
  tests.python.test_funnel_dataset \
  tests.python.test_diffusion_model \
  tests.python.test_proposal_artifact \
  tests.python.test_proposal_request
make build/tests/test_interaction_funnel_artifact
build/tests/test_interaction_funnel_artifact
```

Expected: endpoint identity/schema-2 assertions fail.

- [ ] **Step 4: Implement object-local targets and strict entry restoration**

In dataset extraction, remove `_entry_relative_execution` and use:

```python
funnel = np.ascontiguousarray(execution_object[::-1], dtype=np.float32)
funnel[15] = execution_object[0]
```

In `sample_checkpoint`, restore the per-batch endpoint after denormalization:

```python
entry = raw_condition[:, None, 18:22]
samples[:, :, 15] = entry.expand(-1, samples.shape[1], -1)
```

Train with knot 15 excluded from loss as before. Publish checkpoint schema 3
and reject older checkpoint schemas.

- [ ] **Step 5: Implement schema-3 request/response validation**

Change response magic/version to `G1FUNNL3`/`3`. In Python and C++, compare
each proposal's execution sample zero exactly against condition entries 18
through 21. The worker reverses sampled outward arrays once, certifies the
execution arrays, and writes schema 3 atomically.

- [ ] **Step 6: Run GREEN and retrain**

Run the Step 3 tests, then:

```bash
CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
  tools/train_g1_funnel.py \
  --pack build/smart-pickup/full-pack \
  --output build/g1-funnels/checkpoint.pt \
  --steps 20000
```

Sample spread-out retained conditions and require at least eight certifiable
proposals per condition. Expected: schema 3 checkpoint and all focused tests
pass.

- [ ] **Step 7: Commit**

Stage only the listed source/test files and commit:

```bash
git commit -m "fix: anchor learned funnels to objects"
```

---

### Task 2: Replace time-locked playback with monotonic geometric lookahead

**Files:**

- Modify: `interaction_funnel_follower.h`
- Modify: `interaction_funnel_follower.cpp`
- Modify: `tests/cpp/test_interaction_funnel_follower.cpp`

**Interfaces:**

- Consumes: 16 execution-order object-local knots and a tracked object-local root every consecutive controller tick.
- Produces: one object-local lookahead target, monotonic progress diagnostics, spatial completion, and fail-closed cancellation.

- [ ] **Step 1: Write failing monotonic follower tests**

Replace publication-clock expectations with tests that require:

```cpp
InteractionFunnelFollower follower(42U, smooth_samples(), 100U);
auto first = follower.tick({100U, sample_at(0.00F)});
auto lagged = follower.tick({101U, sample_at(0.01F)});
assert(first.published && lagged.published);
assert(lagged.sample.x >= first.sample.x);
assert(follower.diagnostics().progress_index >= 0);
```

Add separate tests for skipping a tracked pose ahead on the route, never
returning to an earlier target after lateral error, completing only after three
terminal observations, consecutive tick enforcement, cross-track cancellation
over `0.18 m`, yaw cancellation over `25 degrees`, and timeout at 250 ticks.

- [ ] **Step 2: Run RED**

Run:

```bash
make build/tests/test_interaction_funnel_follower
build/tests/test_interaction_funnel_follower
```

Expected: current follower publishes the next clock sample and lacks progress
diagnostics/spatial completion.

- [ ] **Step 3: Implement geometric progress and lookahead**

Expand knots to the existing 75-point polyline. Add constants:

```cpp
constexpr float kFunnelLookaheadDistance = 0.12F;
constexpr float kFunnelTerminalPositionTolerance = 0.04F;
constexpr float kFunnelTerminalYawTolerance = 0.34906585F;
constexpr uint32_t kFunnelRequiredTerminalTicks = 3U;
constexpr uint32_t kFunnelMaximumFollowTicks = 250U;
```

For each tracked pose, scan only segments at or after `progress_index_`, choose
the nearest projection, and update progress with `max(old_progress, nearest)`.
Walk accumulated arc length forward by `0.12 m` to select the published target.
If the terminal position/yaw is within tolerance, increment terminal settle;
otherwise reset it. Complete at three settled ticks. Cancel on missed ticks,
timeout, cross-track error, or yaw error. Expose `progress_index` and
`lookahead_index` in diagnostics.

- [ ] **Step 4: Run GREEN and sanitizers**

Run:

```bash
make build/tests/test_interaction_funnel_follower
build/tests/test_interaction_funnel_follower
g++ -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I. tests/cpp/test_interaction_funnel_follower.cpp \
  interaction_funnel_follower.cpp interaction_funnel_timing.cpp \
  -o build/tests/test_interaction_funnel_follower_san
build/tests/test_interaction_funnel_follower_san
```

Expected: all tests pass with no sanitizer output.

- [ ] **Step 5: Commit**

```bash
git commit -m "fix: follow learned funnels with spatial lookahead"
```

---

### Task 3: Integrate object anchoring and live-pose annulus handoff

**Files:**

- Modify: `interaction_learned_pickup_backend.h`
- Modify: `interaction_learned_pickup_backend.cpp`
- Modify: `interaction_learned_pickup_diagnostics.h`
- Modify: `tests/cpp/test_interaction_learned_pickup_backend.cpp`
- Modify: `tests/cpp/test_interaction_learned_pickup_end_to_end.cpp`

**Interfaces:**

- Consumes: schema-3 object-local proposals and follower outputs.
- Produces: world locomotion targets mapped through the frozen object transform, actual-live-pose handoff inside the annulus, and unchanged final `PickRequest` submission.

- [ ] **Step 1: Write failing frame and handoff tests**

Add a proposal whose first sample equals an object-local entry and assert that
`world_targets` begin at the frozen live root and end at the same object-relative
terminal after applying two different entry roots. Add a lifecycle test that
starts outside the annulus, observes a root crossing inside at a non-radial
position, and requires the frozen condition to encode that observed root rather
than the original `capture_.target_world`.

Add an end-to-end lag simulation where the tracked root advances slower than
the 75-point interpolation. Require nondecreasing follower progress, no target
behind progress, one final preview, one submission, one attachment edge, and
final `Carry`.

- [ ] **Step 2: Run RED**

Run:

```bash
make build/tests/test_interaction_learned_pickup_backend \
     build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_learned_pickup_end_to_end
```

Expected: entry-relative world mapping and time-locked lifecycle assertions
fail.

- [ ] **Step 3: Implement object/world transforms**

Replace entry transforms with exact object transforms:

```cpp
Transform object_local_to_world(Transform object, const FunnelSample& local);
FunnelSample world_to_object_local(Transform object, Transform world);
FunnelExecutionTargets world_targets(
    Transform object, const FunnelProposal& proposal);
```

Freeze `start_.target_snapshot.object_world` at begin. Build selection routes,
final roots, tracked follower input, and published world targets only through
that object transform.

- [ ] **Step 4: Implement live-pose annulus handoff**

While outside the maximum capture radius, keep existing automatic coarse
steering. As soon as the observed root is within `[0.45 m, 1.00 m]` and its
current-to-object connector is safe, set the capture position to the current
displayed root, require facing/speed settle for three ticks, and build the
condition from that exact observation. Do not steer back to the original radial
capture point after entering the annulus.

- [ ] **Step 5: Integrate follower diagnostics and final preview**

Pass `world_to_object_local(frozen_object_world_, displayed_root)` into the
follower and map its output back through `frozen_object_world_`. Continue
remaining-route obstacle validation from monotonic progress. On spatial
completion, issue the existing one-root final preview and submit the unchanged
request only after preview success.

- [ ] **Step 6: Run GREEN and controller regressions**

Run the Step 2 tests plus:

```bash
make build/tests/test_interaction_smart_pickup_controller controller
build/tests/test_interaction_smart_pickup_controller
```

Expected: all learned and authored regressions pass.

- [ ] **Step 7: Commit**

```bash
git commit -m "fix: hand learned pickup from live entry to object route"
```

---

### Task 4: Verify and visualize the corrected playable route

**Files:**

- Modify: `interaction_smart_pickup_controller.h`
- Modify: `interaction_smart_pickup_controller.cpp`
- Modify: `interaction_learned_pickup_backend.h`
- Modify: `interaction_learned_pickup_backend.cpp`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_controller.cpp`
- Modify: `tests/cpp/test_interaction_learned_pickup_end_to_end.cpp`
- Generate: `build/g1-funnels/evidence/object-anchored-live.png`
- Generate: `build/g1-funnels/evidence/object-anchored-live.json`

**Interfaces:**

- Consumes: frozen entry, selected world route, progress/lookahead indices, terminal root, and tracked root diagnostics.
- Produces: visible route overlay and machine-readable evidence from the actual learned provider.

- [ ] **Step 1: Add a failing overlay/evidence assertion**

Add `LearnedPickupDebugSnapshot` to the controller API with `active`,
`frozen_entry_world`, `tracked_root_world`, `terminal_root_world`,
`world_route[75]`, `progress_index`, and `lookahead_index`. Give
`SmartPickupAssistBackend` a default `learned_debug_snapshot()` returning
`std::nullopt`, override it in `LearnedSmartPickupBackend`, and forward it from
`SmartPickupController`. Tests require authored mode to return `nullopt` and an
armed learned lifecycle to return the exact selected route and monotonic indices.

- [ ] **Step 2: Implement the minimal overlay**

Populate `LearnedPickupDebugSnapshot` only from immutable route data and current
diagnostics. In `controller.cpp`, draw the selected route as an object-anchored
polyline, frozen entry as cyan, progress as green, current lookahead as orange,
terminal root as magenta, and tracked root as white. Keep drawing observational;
it must not change control state.

- [ ] **Step 3: Run the complete focused suite**

Run all Python funnel tests and these C++ tests:

```bash
build/tests/test_interaction_funnel_artifact
build/tests/test_interaction_funnel_worker
build/tests/test_interaction_funnel_capture
build/tests/test_interaction_funnel_timing
build/tests/test_interaction_funnel_follower
build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_smart_pickup_controller
```

Expected: every command exits zero.

- [ ] **Step 4: Run the actual learned controller and capture evidence**

Launch `controller` with `G1_SMART_PICKUP_PROVIDER=learned`, the schema-3
checkpoint, pinned Python, worker, runtime work directory, and full interaction
pack. Trigger one attempt from a noncanonical bearing. Require the overlay to
show monotonic progress without returning to an earlier point, then capture PNG
and JSON evidence after the runtime reaches `Carry` with exactly one attachment
edge.

- [ ] **Step 5: Commit source changes and report generated evidence separately**

Do not commit generated venvs, controller binaries, runtime files, screenshots,
or `resources/features.bin`. Commit only source/test overlay changes, if any:

```bash
git commit -m "feat: visualize learned funnel progress"
```
