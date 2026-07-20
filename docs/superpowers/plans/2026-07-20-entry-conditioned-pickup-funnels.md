# Entry-Conditioned Pickup Funnels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Smart Pickup's fixed local entry motion with 32 diffusion-generated three-second approaches conditioned on the character's settled object-relative entry state.

**Architecture:** Ordinary motion matching first reaches a deterministic capture point. A pinned asynchronous worker samples a strict 32-proposal artifact from a frozen 24-float object/grasp/entry condition; C++ validates and ranks that batch, expands 16 knots to 75 native targets, and drives ordinary controls before handing the terminal root to the unchanged pickup matcher and attachment runtime.

**Tech Stack:** Python 3.10, NumPy 2.2.6, PyTorch 2.5.1, C++17, POSIX process APIs, GNU Make, existing interaction runtime at exactly 25 Hz.

## Global Constraints

- Work only from the clean integration worktree based on checkpoint branch `g1-tabletop-placement`.
- Use `build/venvs/g1-funnels/bin/python` for every model or artifact command.
- Use only `build/smart-pickup/full-pack` for training data.
- Preserve the existing matcher, contact playback, attachment, Hold, Carry, and placement authority.
- Learned motion publishes controls only and never writes roots, joints, registry state, or attachment state.
- The model condition is exactly 24 float32 values: the existing 18 object/grasp values followed by object-relative entry `(x,z,sin_yaw,cos_yaw,vx,vz)`.
- Model targets are 16 entry-relative knots in outward order; outward knot 15 is exactly `(0,0,0,1)` and is reversed exactly once for execution.
- Knot native offsets are exactly `[0,5,10,15,20,25,30,35,39,44,49,54,59,64,69,74]`.
- Runtime expands the knots to exactly 75 consecutive 25 Hz targets.
- Sampling produces exactly 32 proposals in one deterministic batch and never resamples during an attempt.
- Proposal generation is asynchronous; the controller brakes in `ProposalPending` and times out after 250 ticks.
- Authored pickup slots may not be read by learned acceptance logic. Authored mode remains an explicit fallback.
- Every task uses RED, GREEN, focused regression, explicit-path staging, commit, and `git push checkpoint HEAD:g1-tabletop-placement`.
- Preserve unrelated user changes and generated build data. Inspect tracked status with `git status --short --untracked-files=no`.

---

### Task 1: Export and train the 24-value entry-conditioned model

**Files:**

- Modify: `resources/g1_interaction_builder/funnel_dataset.py`
- Modify: `resources/g1_interaction_builder/diffusion.py`
- Modify: `tools/train_g1_funnel.py`
- Modify: `tests/python/test_funnel_dataset.py`
- Modify: `tests/python/test_diffusion_model.py`
- Replace generated: `build/g1-funnels/checkpoint.pt`

**Interfaces:**

- Produces: `MODEL_CONDITION_DIM = 24`, `KNOT_FRAME_OFFSETS`, `FunnelDataset.conditions: float32[N,24]`, entry-relative outward `funnels: float32[N,16,4]`, and schema-2 checkpoints.
- Consumes: root/object positions, rotations, and root velocity from `InteractionArtifact`.

- [ ] **Step 1: Write failing dataset contract tests**

Add tests that load the full pack and require the new condition and exact entry endpoint:

```python
from resources.g1_interaction_builder.funnel_dataset import (
    KNOT_FRAME_OFFSETS,
    load_dataset,
)

dataset = load_dataset("build/smart-pickup/full-pack")
self.assertEqual(dataset.conditions.shape[1], 24)
self.assertEqual(tuple(KNOT_FRAME_OFFSETS.tolist()),
                 (0,5,10,15,20,25,30,35,39,44,49,54,59,64,69,74))
expected_entry = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
np.testing.assert_array_equal(
    dataset.funnels[:, 15],
    np.broadcast_to(expected_entry, dataset.funnels[:, 15].shape),
)
self.assertTrue(np.isfinite(dataset.conditions).all())
self.assertTrue(np.allclose(
    np.linalg.norm(dataset.conditions[:, 20:22], axis=1), 1.0,
    atol=2e-5,
))
```

Add a synthetic world-transform invariance test: apply the same planar translation and yaw to root, root velocity, and object transforms; require bitwise-equal 24-value conditions and entry-relative funnels after extraction.

- [ ] **Step 2: Write failing model and checkpoint tests**

Change test tensors to `[B,24]` and require knot 15 to be restored exactly by
`sample_checkpoint`, after denormalization:

```python
condition = torch.zeros((1, 24), dtype=torch.float32)
samples = sample_checkpoint(checkpoint_path, condition, seed=11)
self.assertEqual(tuple(samples.shape), (1, 32, 16, 4))
expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
self.assertTrue(torch.equal(
    samples[:, :, 15], expected.expand(1, 32, 4)))
```

Require a one-step training checkpoint to have `schema_version == 2`, condition normalization shape `(24,)`, and `knot_frame_offsets` equal the frozen tuple.

- [ ] **Step 3: Run RED**

Run:

```bash
build/venvs/g1-funnels/bin/python -m unittest -v \
  tests.python.test_funnel_dataset \
  tests.python.test_diffusion_model
```

Expected: failures showing the current 18-value condition, schema 1, and non-identity outward entry knot.

- [ ] **Step 4: Implement exact dataset transforms**

Set:

```python
KNOT_FRAME_OFFSETS = np.array(
    [0,5,10,15,20,25,30,35,39,44,49,54,59,64,69,74],
    dtype=np.int64,
)
APPROACH_HORIZON_FRAMES = 75
```

Reject rows when `reach - start < 74`. Let `entry = reach - 74`, use `entry + KNOT_FRAME_OFFSETS` in chronological order, and compute the existing object-local execution roots. Append this entry state to the existing condition:

```python
entry_x, entry_z = execution_object[0, :2]
entry_sin, entry_cos = execution_object[0, 2:4]
velocity_x, velocity_z = _object_local_delta(
    artifact.velocities[entry, 0], object_yaw)
condition = np.concatenate((
    _grasp_condition(artifact, clip, reach),
    np.array([entry_x, entry_z, entry_sin, entry_cos,
              velocity_x, velocity_z], np.float32),
)).astype(np.float32)
```

Convert execution roots to the entry frame. For positions rotate `p - entry_position` by inverse entry yaw; for yaw subtract entry yaw and store sine/cosine. Reverse once into outward order and assign `outward[15] = (0,0,0,1)` exactly.

- [ ] **Step 5: Implement the 24-value denoiser and masked endpoint**

Set `MODEL_CONDITION_DIM = 24`, change the condition projection input, and train only generated knots:

```python
prediction = model(noised, timestep, condition)
loss = torch.nn.functional.mse_loss(
    prediction[:, :, :15], clean[:, :, :15])
```

After denormalization and yaw projection, force the outward endpoint:

```python
samples[..., 15, 0] = 0.0
samples[..., 15, 1] = 0.0
samples[..., 15, 2] = 0.0
samples[..., 15, 3] = 1.0
```

Publish checkpoint schema 2 with `condition_dim=24` and the frozen knot offsets. Reject any other values in `load_funnel_checkpoint`.

- [ ] **Step 6: Run focused GREEN tests**

Run the Step 3 command. Expected: all dataset and model tests pass.

- [ ] **Step 7: Retrain and certify the checkpoint**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
  -m tools.train_g1_funnel \
  --pack build/smart-pickup/full-pack \
  --output build/g1-funnels/checkpoint.pt \
  --steps 20000
```

Then sample 64 spread-out retained conditions with seed `2026071901`.
Require every condition to retain at least eight certifiable proposals and the
median within-batch pairwise flattened-trajectory distance to be at least
`0.05`. Report total acceptance as a diagnostic, not as a gate: requiring nearly
every noisy proposal to pass rewards conditional-mean mode collapse rather than
useful diffusion diversity. Stop and diagnose if either survivor or diversity
gate fails.

- [ ] **Step 8: Commit and push**

Stage only the five source/test files and checkpoint. Commit `feat: condition pickup funnels on entry state`, then push the checkpoint branch.

---

### Task 2: Add strict asynchronous request/response artifacts

**Files:**

- Create: `resources/g1_interaction_builder/proposal_request.py`
- Create: `tools/run_g1_funnel_proposal_worker.py`
- Create: `interaction_funnel_worker.h`
- Create: `interaction_funnel_worker.cpp`
- Create: `tests/python/test_proposal_request.py`
- Create: `tests/cpp/test_interaction_funnel_worker.cpp`
- Modify: `resources/g1_interaction_builder/proposal_artifact.py`
- Modify: `interaction_funnel_artifact.h`
- Modify: `interaction_funnel_artifact.cpp`
- Modify: `tests/python/test_proposal_artifact.py`
- Modify: `tests/cpp/test_interaction_funnel_artifact.cpp`
- Modify: `Makefile`

**Interfaces:**

- Produces: `ProposalRequest`, response schema 2, `FunnelProposalProvider`, and `AsyncPythonFunnelProvider`.
- Response artifacts store execution-order entry-relative knots; the worker performs the model's one exact outward-to-execution reversal.

- [ ] **Step 1: Write Python golden-layout tests**

Freeze request layout as little-endian:

```text
magic G1FREQ02[8], schema u32=2, request_id u64, batch_seed u64,
checkpoint_sha256[32], condition float32[24]
```

Freeze response layout as:

```text
magic G1FUNNL2[8], schema u32=2, dimensions u32[4]=24,32,16,4,
request_id u64, batch_seed u64, checkpoint_sha256[32], condition float32[24],
execution proposals float32[32][16][4], seeds u64[32], accepted u8[32]
```

Tests require exact byte length, little-endian round trip, request/response identity equality, unique seeds, exact execution sample-zero identity, malformed/truncated/trailing rejection, and atomic output replacement.

- [ ] **Step 2: Write C++ provider and loader RED tests**

Define:

```cpp
enum class FunnelProposalPollState { Pending, Ready, Failed };

struct FunnelProposalRequest {
    uint64_t request_id = 0U;
    uint64_t batch_seed = 0U;
    std::array<uint8_t, 32> checkpoint_sha256{};
    std::array<float, 24> condition{};
};

struct FunnelProposalPoll {
    FunnelProposalPollState state = FunnelProposalPollState::Pending;
    std::optional<InteractionFunnelArtifact> artifact{};
    std::string error{};
};

class FunnelProposalProvider {
public:
    virtual ~FunnelProposalProvider() = default;
    virtual bool begin(const FunnelProposalRequest&) = 0;
    virtual FunnelProposalPoll poll() = 0;
    virtual void cancel() = 0;
};
```

Tests require one active request, pending polls, exact ready artifact, nonzero worker exit failure, timeout cancellation, and stale response rejection.

- [ ] **Step 3: Run RED**

Run the new Python tests and build both C++ tests. Expected: missing module/header failures.

- [ ] **Step 4: Implement Python schema-2 request, response, and worker**

Use `struct.Struct` and contiguous `<f4`/`<u8` arrays. The worker:

1. reads and validates one request;
2. hashes the checkpoint and requires exact identity;
3. samples one 32-proposal batch;
4. restores outward knot 15 exactly;
5. reverses once with `proposals[:, ::-1].copy()`;
6. certifies execution proposals;
7. writes a temporary response and atomically renames it.

The CLI accepts only `--request`, `--checkpoint`, and `--output` paths.

- [ ] **Step 5: Implement the strict C++ loader and asynchronous provider**

Upgrade the loader to schema 2 and expose request identity getters. Implement `AsyncPythonFunnelProvider` with `posix_spawnp`, fixed argv entries, `waitpid(..., WNOHANG)`, and exact response identity validation. `cancel()` sends `SIGTERM`, reaps the child, and invalidates its request ID. No shell command string or `system()` call is allowed.

- [ ] **Step 6: Run GREEN and existing artifact regressions**

Run both new test binaries plus `tests.python.test_proposal_artifact`. Expected: all pass with `-Wall -Wextra -Werror -pedantic`.

- [ ] **Step 7: Commit and push**

Commit `feat: sample entry-conditioned funnel artifacts asynchronously` and push.

---

### Task 3: Expand 16 knots into a native 75-tick follower

**Files:**

- Create: `interaction_funnel_timing.h`
- Create: `interaction_funnel_timing.cpp`
- Create: `tests/cpp/test_interaction_funnel_timing.cpp`
- Modify: `interaction_funnel_follower.h`
- Modify: `interaction_funnel_follower.cpp`
- Modify: `tests/cpp/test_interaction_funnel_follower.cpp`
- Modify: `Makefile`

**Interfaces:**

- Produces: `expand_funnel_execution` and a 75-target `InteractionFunnelFollower`.
- Consumes: schema-2 execution-order 16-knot proposals.

- [ ] **Step 1: Write exact interpolation RED tests**

Define:

```cpp
constexpr int kFunnelKnotCount = 16;
constexpr int kFunnelExecutionTickCount = 75;
using FunnelExecutionTargets =
    std::array<FunnelSample, kFunnelExecutionTickCount>;

FunnelExecutionTargets expand_funnel_execution(
    const std::array<FunnelSample, kFunnelKnotCount>& knots);
```

Require exact knot reproduction at all frozen offsets, exactly 75 outputs,
linear position interpolation, unit yaw vectors, shortest-arc yaw interpolation
across `+179°` to `-179°`, and exact first/last samples.

- [ ] **Step 2: Write follower RED tests**

Require publications `0..74` exactly once on consecutive tick indices, terminal completion only after target 74 is observed on tick 75, and cancellation on duplicate/skipped ticks, translation tracking over `0.18 m`, yaw tracking over `25°`, or explicit cancellation.

- [ ] **Step 3: Run RED**

Build and run timing/follower tests. Expected: missing timing API and current 16-publication behavior.

- [ ] **Step 4: Implement interpolation and follower schedule**

For each native tick, find enclosing frozen offsets, calculate float64 alpha, interpolate x/z, and interpolate yaw with:

```cpp
const double a = std::atan2(left.yaw_sin, left.yaw_cos);
const double b = std::atan2(right.yaw_sin, right.yaw_cos);
const double delta = std::atan2(std::sin(b - a), std::cos(b - a));
const double yaw = a + alpha * delta;
```

Cast final sine/cosine once to float. Update the follower to own 75 expanded targets and require a final observation after publication 74 before entering `Completed`.

- [ ] **Step 5: Run GREEN and sanitizers**

Run timing/follower tests normally and under ASan/UBSan. Expected: all pass and no sanitizer diagnostics.

- [ ] **Step 6: Commit and push**

Commit `fix: follow learned funnels over their native duration` and push.

---

### Task 4: Implement capture, proposal selection, and learned pickup lifecycle

**Files:**

- Create: `interaction_funnel_capture.h`
- Create: `interaction_funnel_capture.cpp`
- Create: `tests/cpp/test_interaction_funnel_capture.cpp`
- Modify: `interaction_learned_pickup_diagnostics.h`
- Modify: `interaction_learned_pickup_backend.h`
- Modify: `interaction_learned_pickup_backend.cpp`
- Modify: `interaction_pick_assist.h`
- Modify: `interaction_smart_pickup_controller.h`
- Modify: `interaction_smart_pickup_controller.cpp`
- Modify: `tests/cpp/test_interaction_learned_pickup_backend.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_controller.cpp`
- Modify: `Makefile`

**Interfaces:**

- Produces: the complete learned backend state machine behind `SmartPickupAssistBackend`.
- Consumes: `FunnelProposalProvider`, capture geometry, schema-2 artifacts, 75-tick follower, existing preview callback, and unchanged `PickRequest`.

- [ ] **Step 1: Extend the controller observation boundary in RED tests**

Add `uint64_t controller_tick` and `std::vector<PickNavigationObstacle> live_obstacles` to `PickAssistStart`, `PickAssistObservation`, and `SmartPickupPostStepInput`. Tests require the controller to forward the native tick and exact ordered live obstacles on activation and every active observation.

- [ ] **Step 2: Write deterministic capture RED tests**

Define `select_funnel_capture(Transform, InteractionTarget, obstacles, config)` returning a frozen target and failure reason. Require annulus bounds `0.45–1.00 m`, yaw facing the object, direct bearing first, alternating angular offsets, no authored affordance slot reads, and rejection through `revalidate_frozen_pick_slot` for unsafe connectors.

- [ ] **Step 3: Write backend lifecycle RED tests**

Use a fake provider and require this exact lifecycle:

```text
Idle -> CoarseCapture -> CaptureSettling -> ProposalPending
-> SelectionPreview -> FunnelFollow -> FinalPreview
-> ReadyToSubmit -> Submitted
```

Require three settled capture observations, one provider request, pending braking, one atomic 32-root preview batch, deterministic winner, 75 follower publications plus final tracking observation, one final preview, and one unchanged `PickRequest`. Add failures for provider timeout/error, no accepted proposals, malformed preview batches, target/affordance/obstacle mutation, follower failure, final preview rejection, and cancellation.

- [ ] **Step 4: Run RED**

Build capture/backend/controller tests. Expected: missing capture API and lifecycle states.

- [ ] **Step 5: Implement capture and frozen 24-value condition construction**

Capture candidates use the current object-relative bearing followed by angular offsets `+22.5,-22.5,+45,-45,...,+180` degrees. Project radius into `[0.45,1.00]`, face the object, and select the first connector accepted by `revalidate_frozen_pick_slot` with no affordance slot access.

At settled capture, encode the same 18 object/grasp values as Python and append root x/z/yaw and simulation velocity transformed into object coordinates. Golden tests must compare all 24 raw float bits between Python and C++.

- [ ] **Step 6: Implement backend lifecycle and selection**

Replace authored delegation in `LearnedSmartPickupBackend` with the tested lifecycle. Selection preview requests use every accepted proposal's reconstructed terminal root. Certify live connector/expanded segments before preview, then rank survivors by route millimetres, heading milliradians, larger clearance, preview cost, and proposal index. Freeze one proposal; never switch after follow starts.

Each follow observation revalidates frozen target identity, exact ordered obstacles, and all remaining route segments. Steering uses `arrival_navigation_stick` and `arrival_facing_stick`. Final preview certification uses existing path/match readiness and prospective-root equality before setting `submit_interact=true`.

- [ ] **Step 7: Run GREEN and authored regressions**

Run capture/backend/controller tests plus `test_interaction_pick_assist` and the existing Smart Pickup scene tests. Expected: all pass; authored diagnostics remain unchanged when authored mode is selected.

- [ ] **Step 8: Commit and push**

Commit `feat: drive smart pickup with entry-conditioned funnels` and push.

---

### Task 5: Wire playable mode and prove actual attachment

**Files:**

- Modify: `controller.cpp`
- Modify: `interaction_smart_pickup_controller.cpp`
- Modify: `interaction_smart_pickup_controller.h`
- Create: `tests/cpp/test_interaction_learned_pickup_end_to_end.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `Makefile`
- Create generated evidence under: `build/g1-funnels/evidence/`

**Interfaces:**

- Produces: explicit learned provider mode and a headless proof through actual Carry.
- Consumes: pinned Python path, checkpoint, worker script, and the completed learned backend.

- [ ] **Step 1: Write production-mode configuration RED tests**

Require:

```text
G1_SMART_PICKUP_PROVIDER=authored|learned
G1_FUNNEL_PYTHON=build/venvs/g1-funnels/bin/python
G1_FUNNEL_CHECKPOINT=build/g1-funnels/checkpoint.pt
G1_FUNNEL_WORKER=tools/run_g1_funnel_proposal_worker.py
G1_FUNNEL_WORK_DIR=build/g1-funnels/runtime
```

Unknown/missing learned configuration fails before Raylib initialization. Authored remains the default when the provider variable is absent.

- [ ] **Step 2: Write headless end-to-end RED test**

Drive a known table pickup from activation through capture, fake or real worker completion, preview callbacks, 75 learned targets, final preview, runtime submission, Reach/Contact, attachment, Hold, and one subsequent Carry tick. Require target registry `Held`, `attached=true`, exact owner request, and one attachment edge. Assert learned diagnostics identify the selected proposal and authored slot diagnostics remain unused.

- [ ] **Step 3: Run RED**

Build the adapter and end-to-end tests. Expected: missing environment mode and no learned attachment path.

- [ ] **Step 4: Implement production backend construction**

Parse configuration before graphical initialization. Construct `AsyncPythonFunnelProvider` and `LearnedSmartPickupBackend` only for explicit learned mode; otherwise construct the existing authored backend. Create the runtime work directory with attempt-specific request/response names and never reuse a response across attempts.

- [ ] **Step 5: Run full verification**

Run:

```bash
make build/tests/test_interaction_learned_pickup_end_to_end \
     build/tests/test_interaction_smart_pickup_controller \
     build/tests/test_interaction_pick_assist \
     build/tests/test_interaction_controller_adapter
build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_smart_pickup_controller
build/tests/test_interaction_pick_assist
build/tests/test_interaction_controller_adapter
```

Also rerun all Python funnel tests and the three protected funnel runtime verifier modes. Expected: all commands return zero.

- [ ] **Step 6: Record two-bearing evidence**

Run the headless scenario from two distinct initial object-relative bearings with the same object/grasp. Save condition, proposal seed/index, 75 target roots, final runtime state, and attachment audit as JSON under `build/g1-funnels/evidence/`. Require different frozen entry conditions, different selected trajectories, and successful Carry in both runs.

- [ ] **Step 7: Commit and push**

Commit source/tests as `feat: expose playable learned smart pickup`. Push the verified checkpoint branch. Keep generated evidence untracked unless the user explicitly requests publication.

---

## Completion Gate

The feature is complete only when:

- the committed schema-2 checkpoint consumes 24-value conditions;
- a live settled entry condition deterministically produces one atomic 32-proposal batch;
- learned mode reaches and follows one proposal over 75 native ticks without root writes;
- the unchanged runtime reaches actual attached Hold and a subsequent Carry tick;
- two entry bearings select different learned trajectories for the same grasp;
- authored fallback regressions pass; and
- the exact verified commit is pushed to `checkpoint/g1-tabletop-placement`.
