# Simultaneous Walk-to-Pickup Overlap Diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one full-body G1 walking window and one object-conditioned pickup window that denoise simultaneously through a shared 20-frame latent overlap and execute as one no-stop pickup timeline.

**Architecture:** Add a new full-body overlap pipeline beside the existing root-funnel implementation. Python owns motion encoding, dataset extraction, two denoisers, shared-latent DDIM sampling, checkpointing, and a resident worker; C++ owns strict IPC identity, candidate decoding/certification, frozen planning, generated timeline playback through Contact, preselection of the recorded Lift continuation, Contact/attachment/Lift authority, and visualization. The initial model uses two 50-frame windows at 25 Hz, global slices `[0, 50)` and `[30, 80)`, eight candidates, and 20 DDIM steps.

**Tech Stack:** Python 3, NumPy, PyTorch 2.5/CUDA 12.4, C++17, existing G1 interaction artifacts, existing raylib controller, Make-based C++ tests, `unittest` Python tests.

## Global Constraints

- Preserve the existing authored and root-funnel providers; add the overlap provider as an explicit production mode.
- Use exactly 25 Hz, 50 walk frames, 50 pickup frames, 20 shared frames, 80 unique frames, eight candidates, and a 3.20-second complete timeline.
- Use canonical 31-bone `g1_skeleton` / `FLAT_JOINT_NAMES` order.
- Use full G1 pose channels: Simulation-root translation, dynamic local Hips translation, 31 continuous 6D rotations, and three contact channels; derive velocities rather than diffusing them.
- Denoise both windows at the same timestep from one global latent and write overlap consensus back before the next step.
- Freeze simulation and animation while planning; keep rendering, cancellation, diagnostics, and worker polling responsive.
- Resume once and do not introduce a runtime state, stationary command, or late matcher search at global frames 30 or 50.
- Never silently fall back to authored Smart Pickup after learned-overlap activation.
- Keep the live scene on flat terrain and do not use X11 screenshot capture.
- Do not change or delete existing untracked controller/build artifacts.

---

## File Structure

### New Python files

- `resources/g1_interaction_builder/holden_database.py`: strict canonical 31-bone Holden reader/codec support.
- `resources/g1_interaction_builder/overlap_motion.py`: 195-channel motion codec, conditions, dataset records, and extraction.
- `resources/g1_interaction_builder/overlap_diffusion.py`: window denoiser, training losses, shared-overlap DDIM, and checkpoints.
- `resources/g1_interaction_builder/overlap_protocol.py`: strict resident-worker request/response framing.
- `tools/train_g1_overlap_diffusion.py`: deterministic two-expert training entry point.
- `tools/run_g1_overlap_worker.py`: resident prewarmed inference service.
- `tools/evaluate_g1_overlap_diffusion.py`: frozen sampler and seam-quality evaluation.

### New C++ files

- `interaction_overlap_protocol.h/.cpp`: request/response binary types and validation.
- `interaction_overlap_worker.h/.cpp`: resident process lifecycle and length-delimited IPC.
- `interaction_overlap_timeline.h/.cpp`: decoded 80-frame pose timeline, certification, continuation identity, scoring, and playback cursor.
- `interaction_overlap_pickup_backend.h/.cpp`: freeze, request, selection, generated execution, Contact, and preselected Lift-continuation state machine.

### Existing files to modify

- `interaction_pick_assist.h`: publish an optional generated full-body pose/contact event from a learned backend.
- `interaction_smart_pickup_controller.h/.cpp`: configure and construct the overlap provider/backend.
- `interaction_runtime.h/.cpp`: accept a certified generated Contact stream and its preselected recorded Lift continuation without late matching.
- `interaction_native_g1_bridge.h/.cpp`: give a generated overlap pose explicit authority before normal Hold/Carry release blending.
- `interaction_debug_draw.h`: draw the generated root timeline and shared overlap.
- `controller.cpp`: pass overlap environment variables and apply generated pose authority.
- `Makefile`: build the four new C++ units and tests.

---

### Task 1: Walking Database Reader and Full-Body Motion Codec

**Files:**
- Create: `resources/g1_interaction_builder/holden_database.py`
- Create: `resources/g1_interaction_builder/overlap_motion.py`
- Create: `tests/python/test_overlap_motion.py`

**Interfaces:**
- Produces: `read_holden_database(path: Path) -> HoldenDatabase`
- Produces: `encode_motion(positions, rotations, contacts, anchor) -> np.ndarray[N, 195]`
- Produces: `decode_motion(encoded, local_positions, anchor) -> DecodedMotion`
- Constants: `BONE_COUNT=31`, `FRAME_DIM=195`, `WALK_FRAMES=50`, `PICKUP_FRAMES=50`, `OVERLAP_FRAMES=20`, `TIMELINE_FRAMES=80`

- [ ] **Step 1: Write failing binary-reader and codec tests**

```python
def test_frame_schema_is_frozen():
    self.assertEqual(overlap_motion.FRAME_DIM, 3 + 3 + 31 * 6 + 3)
    self.assertEqual(overlap_motion.walk_slice(), slice(0, 50))
    self.assertEqual(overlap_motion.pickup_slice(), slice(30, 80))

def test_codec_round_trips_pose_and_canonicalizes_quaternion_sign():
    encoded = overlap_motion.encode_motion(positions, rotations, contacts, anchor)
    decoded = overlap_motion.decode_motion(encoded, positions[0], anchor)
    np.testing.assert_allclose(decoded.positions[:, 0], positions[:, 0], atol=2e-5)
    np.testing.assert_allclose(np.abs((decoded.rotations * rotations).sum(-1)), 1, atol=2e-5)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python3 -m unittest tests.python.test_overlap_motion -v`

Expected: FAIL with `ImportError: cannot import name 'overlap_motion'`.

- [ ] **Step 3: Implement strict little-endian Holden database loading**

Read the eight arrays written by `resources/generate_database_g1.py:227-243`; reject trailing bytes, non-finite floats, non-contiguous ranges, bone counts other than 31, and contact counts other than two. Return positions, velocities, rotations, angular velocities, parents, ranges, and contacts without loading `features.bin`.

- [ ] **Step 4: Implement the 195-channel codec**

Use channels `[simulation_root_xyz(3), hips_local_xyz(3), rotation6d(31*6), left_foot, right_foot, active_hand]`. Convert 6D rotations with Gram-Schmidt and reject a column norm below `1e-8`. Preserve the dynamic local Hips translation at bone index 1, reconstruct only bones 2–30 from frozen canonical local offsets, and derive linear/angular velocity with centered finite differences at 25 Hz. Add a regression whose Hips offset changes by at least 8 cm and round-trips exactly.

- [ ] **Step 5: Run focused and existing schema tests**

Run: `python3 -m unittest tests.python.test_overlap_motion tests.python.test_interaction_sources -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_interaction_builder/holden_database.py resources/g1_interaction_builder/overlap_motion.py tests/python/test_overlap_motion.py
git commit -m "feat: add full-body overlap motion codec"
```

### Task 2: Genuine Shared-Overlap Dataset Extraction

**Files:**
- Modify: `resources/g1_interaction_builder/overlap_motion.py`
- Create: `tests/python/test_overlap_dataset.py`
- Create: `tools/build_g1_overlap_dataset.py`

**Interfaces:**
- Produces: `OverlapDataset(walk_windows, pickup_windows, static_conditions, walk_temporal, pickup_temporal, sequence_indices, object_ids)`
- Produces: `extract_interaction_pairs(artifact: InteractionArtifact) -> OverlapDataset`
- Produces: `load_native_g1_walk(source_npz, g1_xml) -> HoldenClip`
- Produces: `extract_walking_windows(clip: HoldenClip, stride: int=10) -> np.ndarray[N,50,195]`

- [ ] **Step 1: Write failing source-index and overlap-identity tests**

```python
def test_interaction_pair_uses_real_contiguous_overlap():
    rows = extract_interaction_pairs(interaction_artifact_fixture(100))
    np.testing.assert_array_equal(rows.walk_windows[:, 30:50], rows.pickup_windows[:, 0:20])
    np.testing.assert_array_equal(rows.source_walk_ranges[0], [reach - 50, reach])
    np.testing.assert_array_equal(rows.source_pickup_ranges[0], [reach - 20, reach + 30])
```

Also assert rejection reasons for missing Reach, Contact after `R+29`, left hand, invalid grasp, missing continuation identity, and short clips. Do not reject a row merely because Lift occurs after `R+29`.

- [ ] **Step 2: Run tests and verify extraction symbols are missing**

Run: `python3 -m unittest tests.python.test_overlap_dataset -v`

Expected: FAIL with `ImportError` or missing `extract_interaction_pairs`.

- [ ] **Step 3: Implement exact 80-frame interaction extraction**

For each valid right-hand clip, use global source range `[R-50,R+30)`, walking target `[R-50,R)`, pickup target `[R-20,R+30)`, and object frame from Contact-minus-one. Build the 25-value static condition as the existing 18 grasp values plus initial root linear/angular velocity and `object_present=1`. Build nine temporal channels per global frame: object-local route `(x,z,sin(yaw),cos(yaw))` plus one-hot Walk/Approach/Reach/Contact/Lift. Record the source clip and post-window frame range that continues to the existing Lift witness.

- [ ] **Step 4: Implement balanced 50-frame walking extraction**

Load `/home/ubuntu/projects/mjx-diffphysics/env/g1/motions/lafan_walk_short.npz` and `/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml`, validate the 29 DoF/body names, convert through `G1Kinematics` and `convert_source_clip` into the canonical 31-bone Y-up `HoldenClip` at 25 Hz, and keep each 50-frame window inside that source clip. Construct each terminal-root goal frame, set object/grasp values to zero and `object_present=0`, and balance by planar-speed and yaw-rate bins before deterministic splitting. Do not use or retarget the tracked legacy 23-bone `resources/database.bin`.

- [ ] **Step 5: Add deterministic NPZ export with manifest JSON**

The CLI accepts `--walking-source`, `--g1-xml`, `--interaction-pack`, `--output`, and `--seed`; write training/validation/test arrays, source hashes, rejection counts, continuation identities, normalization partition, object IDs, and exact source ranges atomically.

- [ ] **Step 6: Run fixture tests and a real data audit**

Run:

```bash
python3 -m unittest tests.python.test_overlap_dataset -v
python3 tools/build_g1_overlap_dataset.py \
  --walking-source /home/ubuntu/projects/mjx-diffphysics/env/g1/motions/lafan_walk_short.npz \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --interaction-pack /home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/smart-pickup/full-pack \
  --output build/g1-overlap/dataset.npz \
  --seed 2026072101
```

Expected: tests PASS; CLI reports nonzero train/validation/test interaction rows, byte-identical 20-frame overlaps, valid continuation identities, and rejection counts without modifying source artifacts.

- [ ] **Step 7: Commit**

```bash
git add resources/g1_interaction_builder/overlap_motion.py tests/python/test_overlap_dataset.py tools/build_g1_overlap_dataset.py
git commit -m "feat: build genuine walk pickup overlap data"
```

### Task 3: Two Expert Denoisers and Global Consensus DDIM

**Files:**
- Create: `resources/g1_interaction_builder/overlap_diffusion.py`
- Create: `tests/python/test_overlap_diffusion.py`

**Interfaces:**
- Produces: `MotionWindowDenoiser(frame_dim=195, static_dim=25, temporal_dim=9)`
- Produces: `sample_coupled(walk_model, pickup_model, condition, *, seed, candidates=8, steps=20) -> Tensor[8,80,195]`
- Produces: `overlap_weight(index: int) -> float`

- [ ] **Step 1: Write failing simultaneous-sampling tests**

```python
def test_overlap_endpoints_and_shared_latent_contract():
    self.assertEqual(overlap_weight(0), 0.0)
    self.assertEqual(overlap_weight(19), 1.0)
    out, trace = sample_coupled(walk, pickup, condition, seed=9, steps=2, return_trace=True)
    self.assertEqual(tuple(out.shape), (8, 80, 195))
    for step in trace:
        self.assertTrue(torch.equal(step.walk_input[:, 30:50], step.pickup_input[:, 0:20]))
```

Add a directional test where constant walk/pickup epsilon predictors prove that global frames 30 and 49 select the walk and pickup endpoints exactly and an interior frame receives both predictions.

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_diffusion -v`

Expected: FAIL with missing `overlap_diffusion`.

- [ ] **Step 3: Implement the shared denoiser architecture**

Use a 256-wide temporal transformer with eight residual blocks and eight heads. Project frame, static-condition, temporal-condition, and sinusoidal timestep embeddings to width 256; emit epsilon in `[B,50,195]`. Keep walking and pickup weights independent but schemas identical.

- [ ] **Step 4: Implement one-global-latent DDIM**

Allocate CPU-seeded noise `[8,80,195]` once. At each timestep, slice `[0:50]` and `[30:80]`, call both models, assemble one epsilon tensor, blend overlap `j` with `w=j/19`, apply fixed-frame and differentiable task guidance, and perform one global DDIM update. Never independently update the two slices.

- [ ] **Step 5: Implement training losses**

Return named losses for epsilon, 6D pose, FK hand/feet, finite-difference velocity, acceleration, foot contact/sliding, grasp position/orientation, attachment, and overlap agreement. Unit tests use identity FK fixtures to verify every term is zero for exact predictions and positive for its isolated perturbation.

- [ ] **Step 6: Run deterministic sampler tests**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_diffusion -v`

Expected: PASS; repeated seeds are bit-identical on CPU.

- [ ] **Step 7: Commit**

```bash
git add resources/g1_interaction_builder/overlap_diffusion.py tests/python/test_overlap_diffusion.py
git commit -m "feat: add simultaneous overlap diffusion sampler"
```

### Task 3A: Preserve Dynamic Hips Translation Before Training

**Files:**
- Modify: `resources/g1_interaction_builder/overlap_motion.py`
- Modify: `resources/g1_interaction_builder/overlap_diffusion.py`
- Modify: `tests/python/test_overlap_motion.py`
- Modify: `tests/python/test_overlap_dataset.py`
- Modify: `tests/python/test_overlap_diffusion.py`
- Modify: `tools/build_g1_overlap_dataset.py`

- [ ] **Step 1: Add failing dynamic-Hips codec and 195-channel schema tests**

Prove an 8 cm local Hips displacement round-trips while all existing global translation/yaw invariance and byte-identical overlap tests remain true. Update sampler shape, loss-channel slices, normalization, manifest, and malformed-schema tests to 195.

- [ ] **Step 2: Implement the amended frame layout**

Encode Simulation-root XYZ, local Hips XYZ, 31 rotation-6D blocks, and three contacts. Decode dynamic Hips XYZ and restore frozen offsets only for bones 2–30. Keep all shared-latent, task-guidance, and loss semantics unchanged except for corrected channel offsets.

- [ ] **Step 3: Rebuild and audit the real dataset**

Run all overlap motion/dataset/diffusion tests, rebuild `build/g1-overlap/dataset.npz`, verify `frame_schema=195`, 2,020 interaction rows, byte-identical overlaps, object/source-disjoint splits, and nonzero Hips-channel variance.

- [ ] **Step 4: Commit**

```bash
git add resources/g1_interaction_builder/overlap_motion.py resources/g1_interaction_builder/overlap_diffusion.py tests/python/test_overlap_motion.py tests/python/test_overlap_dataset.py tests/python/test_overlap_diffusion.py tools/build_g1_overlap_dataset.py
git commit -m "fix: preserve dynamic hips in overlap motion"
```

### Task 4: Training, Checkpoints, and 20-vs-50-Step Quality Gate

**Files:**
- Create: `tools/train_g1_overlap_diffusion.py`
- Modify: `resources/g1_interaction_builder/overlap_diffusion.py`
- Create: `tests/python/test_overlap_checkpoint.py`
- Modify: `tools/build_g1_overlap_dataset.py`
- Modify: `tests/python/test_overlap_dataset.py`

**Interfaces:**
- Produces: schema-v1 checkpoint containing both model states, normalization, skeleton signature, dataset digest, losses, and sampler contract.
- Produces: `load_overlap_checkpoint(path, device) -> LoadedOverlapModels`
- Dataset stores canonical parents and local offsets for bones 2–30 so the real differentiable G1 FK losses never use an identity/proxy skeleton.

- [ ] **Step 1: Write failing tiny-training and identity tests**

Assert one training step writes both model states, `frame_dim=195`, windows `(50,50,20,80)`, prediction type `epsilon`, 25 Hz, normalization arrays, skeleton signature, dataset digest, and no test-partition statistics.

- [ ] **Step 2: Run the focused test and confirm schema failure**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_checkpoint -v`

Expected: FAIL because the trainer and checkpoint loader do not exist.

- [ ] **Step 3: Implement deterministic staged training**

Stage A pretrains the walking expert on walking plus interaction pre-Reach windows. Stage B trains the pickup expert on object-conditioned windows. Stage C jointly fine-tunes both experts on genuine overlap pairs with identical overlap noise and the full named loss set. Save `last.pt` every epoch and atomically replace `best.pt` only from validation `attach_proxy@8`, grasp error, and no-stop proxy.

- [ ] **Step 4: Add sampler quality-gate evaluation**

Evaluate the same frozen validation rows at 20 and 50 DDIM steps. Select 20 only when `attach_proxy@8` and no-stop completion are each within two percentage points, median grasp position is within 2 mm, and median grasp orientation is within 2 degrees of 50 steps; serialize the selected production step count.

- [ ] **Step 5: Run tiny training and checkpoint tests**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_checkpoint tests.python.test_overlap_diffusion -v`

Expected: PASS.

- [ ] **Step 6: Train the real first checkpoint**

Run:

```bash
/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python tools/train_g1_overlap_diffusion.py \
  --dataset build/g1-overlap/dataset.npz \
  --output build/g1-overlap/checkpoint.pt \
  --seed 2026072102 \
  --device cuda
```

Expected: one L40S is used; `checkpoint.pt`, training JSONL, and validation summary are written; no test rows influence checkpoint selection.

- [ ] **Step 7: Commit code and frozen training configuration, not large artifacts**

```bash
git add resources/g1_interaction_builder/overlap_diffusion.py tools/train_g1_overlap_diffusion.py tests/python/test_overlap_checkpoint.py
git commit -m "feat: train coupled walk pickup experts"
```

### Task 5: Strict Resident Python Worker

**Files:**
- Create: `resources/g1_interaction_builder/overlap_protocol.py`
- Create: `tools/run_g1_overlap_worker.py`
- Create: `tests/python/test_overlap_worker.py`

**Interfaces:**
- Produces: `OverlapRequest` and `OverlapResponse` schema-v1 length-delimited messages.
- Worker command: `run_g1_overlap_worker.py --checkpoint PATH --device cuda`

- [ ] **Step 1: Write failing framing, repeated-request, and stale-identity tests**

Use an in-memory `BytesIO` service loop to assert Ready is emitted only after checkpoint load/warmup, two requests reuse the same model object, request IDs round-trip, malformed lengths fail closed, and checkpoint hashes are computed once.

- [ ] **Step 2: Run tests and verify missing protocol**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_worker -v`

Expected: FAIL with missing protocol/worker modules.

- [ ] **Step 3: Implement strict binary messages**

Request payload contains magic/version, request ID, seed, checkpoint SHA-256, frozen target generation, encoded initial frame, static condition, `[80,9]` temporal condition, and obstacle/route identity. Response contains identity, sampler steps, elapsed microseconds, eight decoded `[80,31]` local positions/rotations, three contacts, phase events, and per-candidate finite diagnostics.

- [ ] **Step 4: Implement resident startup and warmup**

Load Torch/checkpoint/CUDA once, verify SHA once, allocate a warmup request, synchronize CUDA, emit Ready, then process length-prefixed stdin requests until EOF. Catch request-local errors and return Failed without exiting; exit only for checkpoint/CUDA corruption.

- [ ] **Step 5: Run unit and process-level worker tests**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_worker -v`

Expected: PASS; second request reports the same startup generation and no second checkpoint load.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_interaction_builder/overlap_protocol.py tools/run_g1_overlap_worker.py tests/python/test_overlap_worker.py
git commit -m "feat: add resident overlap inference worker"
```

### Task 6: C++ Resident Provider and Cross-Language Protocol

**Files:**
- Create: `interaction_overlap_protocol.h`
- Create: `interaction_overlap_protocol.cpp`
- Create: `interaction_overlap_worker.h`
- Create: `interaction_overlap_worker.cpp`
- Create: `tests/cpp/test_interaction_overlap_worker.cpp`
- Modify: `Makefile:149-184,394-431`

**Interfaces:**
- Produces: `OverlapMotionRequest`, `OverlapMotionResponse`, `serialize_overlap_request`, `deserialize_overlap_response`.
- Produces: `ResidentOverlapProvider::ready()`, `begin(request)`, `poll()`, `cancel()`, `restart()`.

- [ ] **Step 1: Write failing byte-layout and lifecycle tests**

Test exact Python-compatible little-endian bytes, non-finite rejection, eight-candidate/80-frame/31-bone shape rejection, Ready handshake, repeated requests through one child PID, stale request rejection, cancellation, EOF, and child crash.

- [ ] **Step 2: Build and confirm missing implementation failure**

Run: `make build/tests/test_interaction_overlap_worker`

Expected: FAIL because overlap protocol/provider files do not exist.

- [ ] **Step 3: Implement protocol validation**

Use fixed-width integer and float append/read helpers. Bound every length before allocation, cap payload at 8 MiB, reject trailing bytes and non-finite floats, and compare request ID, seed, checkpoint identity, target generation, and condition digest before returning Ready.

- [ ] **Step 4: Implement persistent child lifecycle**

Create stdin/stdout pipes, `posix_spawnp` the worker once, set read side nonblocking, consume Ready, and retain PID for repeated requests. `poll()` must never block the render/controller update. `cancel()` discards the active request but keeps a healthy worker; `restart()` terminates only the exact stored PID and starts a fresh child.

- [ ] **Step 5: Run C++ and Python protocol tests**

Run:

```bash
make build/tests/test_interaction_overlap_worker
build/tests/test_interaction_overlap_worker
/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_worker -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add interaction_overlap_protocol.h interaction_overlap_protocol.cpp interaction_overlap_worker.h interaction_overlap_worker.cpp tests/cpp/test_interaction_overlap_worker.cpp Makefile
git commit -m "feat: host persistent overlap worker"
```

### Task 7: C++ Timeline Decoding, Certification, and Playback

**Files:**
- Create: `interaction_overlap_timeline.h`
- Create: `interaction_overlap_timeline.cpp`
- Create: `tests/cpp/test_interaction_overlap_timeline.cpp`
- Modify: `Makefile`

**Interfaces:**
- Produces: `GeneratedPickupTimeline` containing 80 `Pose` frames, a Contact event, and a certified recorded-continuation identity.
- Produces: `certify_overlap_candidate(candidate, context) -> OverlapCertification`.
- Produces: `GeneratedTimelinePlayer::update(dt) -> GeneratedTimelineFrame`.

- [ ] **Step 1: Write failing certification tests**

Build one valid synthetic 80-frame candidate plus continuation and independently perturb frozen start, quaternion norm, joint speed, table collision, route corridor, foot penetration, three-frame overlap dwell, grasp transform, Contact geometry, continuation seam, and continuation Lift witness. Assert one stable rejection reason for each and stable scoring/candidate-index tie-breaks.

- [ ] **Step 2: Build and verify missing symbols**

Run: `make build/tests/test_interaction_overlap_timeline`

Expected: FAIL because timeline types are undefined.

- [ ] **Step 3: Implement response-to-Pose decoding and finite differences**

Convert 31 local positions/rotations into `interaction::Pose`, derive velocities/angular velocities at 25 Hz, threshold foot contacts at 0.5, retain learned hand contact only as a diagnostic, and require exact frozen frame-zero agreement within `2e-5` translation/quaternion-sign distance.

- [ ] **Step 4: Implement complete-chain certification and ranking**

Reuse existing G1 FK, joint, table, obstacle, grasp, attachment, and lift helpers rather than duplicating tolerances. Implement the no-stop rule exactly: when entering overlap above 0.10 m/s, reject three consecutive pre-Contact frames below 0.03 m/s in global frames `[30,55)`.

- [ ] **Step 5: Implement deterministic 25 Hz playback**

Advance monotonically by accumulated `dt`, interpolate only when render `dt` lies between generated frames, publish Contact exactly once, and enter the already certified recorded continuation directly after frame 79. Publish its Lift crossing exactly once and end in its Lift pose for the normal Hold/Carry inertializer.

- [ ] **Step 6: Run tests**

Run: `make build/tests/test_interaction_overlap_timeline && build/tests/test_interaction_overlap_timeline`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_overlap_timeline.h interaction_overlap_timeline.cpp tests/cpp/test_interaction_overlap_timeline.cpp Makefile
git commit -m "feat: certify generated pickup timelines"
```

### Task 8: Generated Pickup Runtime Authority

**Files:**
- Modify: `interaction_runtime.h:59-72,199-219,241-304`
- Modify: `interaction_runtime.cpp:1538-1570,1740-2075,2100-2470`
- Modify: `interaction_pick_assist.h:86-95`
- Create: `tests/cpp/test_interaction_generated_pickup.cpp`
- Modify: `Makefile`

**Interfaces:**
- Produces: `CertifiedGeneratedPickup` with target, affordance, condition digest, contact frame, selected timeline identity, and preselected recorded continuation identity.
- Adds: `InteractionRuntime::begin_generated_pickup(const CertifiedGeneratedPickup&)`.
- Adds: `RuntimeInput::generated_pose`, `generated_contact_crossing`, and continuation playback/Lift events.

- [ ] **Step 1: Write failing runtime authority tests**

Assert Locomotion accepts only a certified current target/generation; generated pre-Contact frames do not attach; Contact uses the existing geometric gate; attachment follows the generated hand; the preselected continuation reaches Lift/Held/Carry; stale target, bad grasp, missing Contact, bad continuation identity, early object motion, cancellation, and tracking failure reject without teleporting or authored fallback.

- [ ] **Step 2: Build and verify missing API failure**

Run: `make build/tests/test_interaction_generated_pickup`

Expected: FAIL because `begin_generated_pickup` is absent.

- [ ] **Step 3: Add the certified generated-pick seam**

Store one immutable token only after C++ certification. Revalidate target handle/generation, affordance, object transform, condition digest, frozen root, and preselected recorded continuation before accepting frame zero. Do not call the pickup matcher during execution; recorded continuation playback is instantiated and certified during frozen planning.

- [ ] **Step 4: Route Contact and Lift through existing attachment authority**

At the generated Contact crossing, call the existing geometric attachment checks with generated hand/object transforms. While attached, update object pose through `AttachmentController`; at frame 79 continue immediately into the preselected recorded post-Contact segment, and at its Lift crossing require the existing lift-height witness before entering Hold/Carry. Preserve all existing authored/recorded paths unchanged.

- [ ] **Step 5: Run generated and existing runtime tests**

Run:

```bash
make build/tests/test_interaction_generated_pickup build/tests/test_interaction_runtime build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_generated_pickup
build/tests/test_interaction_runtime
build/tests/test_interaction_learned_pickup_end_to_end
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add interaction_runtime.h interaction_runtime.cpp interaction_pick_assist.h tests/cpp/test_interaction_generated_pickup.cpp Makefile
git commit -m "feat: add generated pickup runtime authority"
```

### Task 9: Overlap Backend, Controller Integration, and Debug Drawing

**Files:**
- Create: `interaction_overlap_pickup_backend.h`
- Create: `interaction_overlap_pickup_backend.cpp`
- Create: `tests/cpp/test_interaction_overlap_pickup_backend.cpp`
- Modify: `interaction_smart_pickup_controller.h:20-32,120-153`
- Modify: `interaction_smart_pickup_controller.cpp:89-160,300-360`
- Modify: `interaction_native_g1_bridge.h/.cpp`
- Modify: `interaction_debug_draw.h`
- Modify: `controller.cpp:144-176,3000-3400,4300-4400`
- Modify: `Makefile`

**Interfaces:**
- Adds provider mode `overlap` and environment variables `G1_OVERLAP_CHECKPOINT`, `G1_OVERLAP_WORKER`, `G1_OVERLAP_PYTHON`, `G1_OVERLAP_WORK_DIR`.
- Backend states: `Idle`, `FrozenPlanning`, `FrozenSelection`, `Executing`, `Submitted`, `Failed`.

- [ ] **Step 1: Write failing freeze/resume/state tests**

Use a fake resident provider to prove `F` captures one immutable pose/velocity/target snapshot, planning polls without simulation ticks, eight responses and their continuations are certified once, execution owns all 80 generated poses and then the preselected continuation, no state change occurs at frames 30/50, resume occurs once, cancellation unfreezes, and all failure paths omit authored submission.

- [ ] **Step 2: Build and verify missing backend failure**

Run: `make build/tests/test_interaction_overlap_pickup_backend`

Expected: FAIL because the overlap backend does not exist.

- [ ] **Step 3: Implement frozen planning and selection**

Build the collision-free 80-frame route corridor from frozen root to object, serialize the full request, keep `planning_barrier=true` while polling, preselect and seam-certify each candidate's recorded Lift continuation, certify all complete chains in stable index order, choose the lowest score, and arm one `GeneratedTimelinePlayer` plus one `CertifiedGeneratedPickup` token.

- [ ] **Step 4: Implement generated pose ownership and runtime events**

Publish the generated `Pose` each tick through `PickAssistOutput`; let `NativeG1PoseHandoff` treat it as explicit learned authority. Forward generated Contact and continuation Lift crossings to `InteractionRuntime`. Frames 30 and 50 remain ordinary player indices and cannot trigger any controller transition; frame 79 enters only the continuation already chosen while frozen.

- [ ] **Step 5: Add explicit production configuration**

Parse `G1_SMART_PICKUP_PROVIDER=overlap` separately from `learned`; verify files and checkpoint SHA at startup; construct one `ResidentOverlapProvider`; leave `authored` and `learned` behavior byte-for-byte compatible in existing config tests.

- [ ] **Step 6: Draw diagnostics without screenshots**

Draw complete root route blue, shared global frames 30-49 cyan, Contact magenta, current progress green, and rejection/latency text. Keep flat terrain and existing G1 mesh rendering.

- [ ] **Step 7: Run focused and controller regression tests**

Run:

```bash
make build/tests/test_interaction_overlap_pickup_backend build/tests/test_interaction_smart_pickup_controller build/tests/test_interaction_native_g1_bridge
build/tests/test_interaction_overlap_pickup_backend
build/tests/test_interaction_smart_pickup_controller
build/tests/test_interaction_native_g1_bridge
python3 -m unittest tests.python.test_native_g1_diffusion_controller -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add interaction_overlap_pickup_backend.h interaction_overlap_pickup_backend.cpp tests/cpp/test_interaction_overlap_pickup_backend.cpp interaction_smart_pickup_controller.h interaction_smart_pickup_controller.cpp interaction_native_g1_bridge.h interaction_native_g1_bridge.cpp interaction_debug_draw.h controller.cpp Makefile
git commit -m "feat: execute simultaneous walk pickup motion"
```

### Task 10: Frozen Evaluation, Monolithic Baseline, and Live Proof

**Files:**
- Create: `tools/evaluate_g1_overlap_diffusion.py`
- Create: `tests/python/test_overlap_evaluation.py`
- Create: `resources/g1_interaction_builder/overlap_evaluation_manifest.json`
- Modify: `Makefile`

**Interfaces:**
- Produces deterministic JSON/JSONL metrics for current funnel, sequential ablation, coupled model, and monolithic baseline.
- Adds Make targets `g1-overlap-test`, `g1-overlap-evaluate`, and `g1-overlap-live`.

- [ ] **Step 1: Write failing manifest and no-stop metric tests**

Assert the manifest covers 1/2/3 m, three headings, stationary/slow/ordinary initial speeds, every certified height, fixed seeds, object-disjoint IDs, and all declared rows. Assert exactly three sub-0.03 m/s frames in `[30,55)` violate no-stop only when overlap entry exceeds 0.10 m/s.

- [ ] **Step 2: Run tests and verify missing evaluator failure**

Run: `/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/venvs/g1-funnels/bin/python -m unittest tests.python.test_overlap_evaluation -v`

Expected: FAIL because evaluation code/manifest are absent.

- [ ] **Step 3: Implement offline evaluation and baselines**

Evaluate `contact@8`, `attach@8`, `lift@8`, collision, grasp errors, seam velocity/acceleration, foot sliding, route deviation, deterministic regeneration, and warmed latency. Sequential ablation freezes the walking result before pickup inpainting. Monolithic baseline uses one 80-frame denoiser with the identical channels, conditions, split, candidate budget, steps, and certifier.

- [ ] **Step 4: Add complete test targets**

`g1-overlap-test` runs all new Python/C++ tests plus existing learned pickup, runtime, native bridge, mesh, placement, and controller tests. `g1-overlap-evaluate` writes immutable reports under `build/g1-overlap/evaluation`. `g1-overlap-live` launches one controller only, with the resident overlap worker and flat-terrain override.

- [ ] **Step 5: Run the complete automated gate**

Run: `make g1-overlap-test`

Expected: every new and existing selected test passes.

- [ ] **Step 6: Run frozen evaluation**

Run: `make g1-overlap-evaluate`

Expected: all manifest rows reported; coupled no-stop completion is strictly higher than sequential, collision-free `attach@8` is not lower, and warmed p95 latency is below 2.0 seconds. If not, preserve the report and return to the single failing layer rather than adjusting test rows.

- [ ] **Step 7: Launch one live flat-terrain controller**

Run: `make g1-overlap-live`

Expected: one controller and one resident worker; F freezes immediately, debug route/overlap appear, planning finishes, and G1 resumes once through continuous walking, pickup, attachment, Lift, Carry, placement, and re-pick from 1 m, 2 m, and 3 m.

- [ ] **Step 8: Commit evaluation code and frozen manifest**

```bash
git add tools/evaluate_g1_overlap_diffusion.py tests/python/test_overlap_evaluation.py resources/g1_interaction_builder/overlap_evaluation_manifest.json Makefile
git commit -m "test: gate simultaneous walk pickup diffusion"
```

## Final Verification

- [ ] Run `git status --short` and confirm only expected build artifacts remain untracked.
- [ ] Run `make g1-overlap-test` once more from the committed tree.
- [ ] Run `make g1-overlap-evaluate` and retain the complete immutable report.
- [ ] Confirm exactly one controller and one resident worker process are running.
- [ ] Test F from 1 m, 2 m, and 3 m without X11 capture; inspect the live visualization directly.
- [ ] Confirm placement and re-pick still work after generated pickup.
- [ ] Request code review before claiming completion.
