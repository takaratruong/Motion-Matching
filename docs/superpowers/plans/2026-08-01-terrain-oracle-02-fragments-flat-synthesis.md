# Terrain Oracle Phase 2: Contact Fragments and Flat Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the frozen canonical corpus into reusable contact-to-contact
fragments and prove smooth, grounded, globally planned flat locomotion before adding
terrain.

**Architecture:** Segment only at stable support events, retain context for
inertialization, and build a graph whose edges are mechanically compatible
transitions. A robot-centred `MotionRequest` isolates command generation from
matching. Deterministic full-route search selects an entire fragment sequence ending
in a valid stop, and a flat reconstruction stage applies bounded warping,
inertialization, and planted-foot correction.

**Tech Stack:** Python 3.10+, NumPy, SciPy 1.15.3, MuJoCo kinematics, unittest.

## Global Constraints

- Phase 1's frozen corpus and coverage hashes are immutable inputs.
- Fragment ranges are half-open `[start_frame, stop_frame)`.
- Whole clips are never treated as indivisible terrain skills.
- Search may publish a route only with a complete suffix or a valid safe stop.
- A missing route is `unsupported`; it is never represented by holding an arbitrary
  pose.
- Movement and body facing are independent robot-centred trajectories.
- Warps are accepted only near explicitly audited samples in a connected joint
  parameter neighborhood.
- A planted foot remains fixed in world coordinates throughout its stance interval.
- The old flat/terrain viewer remains unchanged as a comparison baseline.

---

## File map

- `terrain_oracle/fragments.py`: support events, fragment metadata, segmentation.
- `terrain_oracle/fragment_features.py`: entry/exit/contact/trajectory descriptors.
- `terrain_oracle/warp_envelope.py`: validated joint warp neighborhoods.
- `terrain_oracle/fragment_store.py`: immutable fragment bank.
- `terrain_oracle/request.py`: robot-centred request contract and providers.
- `terrain_oracle/motion_graph.py`: compatibility graph and deterministic search.
- `terrain_oracle/inertialization.py`: pose/velocity offset decay.
- `terrain_oracle/flat_reconstruction.py`: placement, warp, foot lock, output rows.
- `terrain_oracle/flat_oracle.py`: phase-2 orchestration and safe-stop semantics.
- `terrain_oracle/fragment_cli.py`: build/search/evaluate CLI.
- `tests/python/test_oracle_fragments.py` through
  `tests/python/test_oracle_flat_oracle.py`: phase-2 tests.

### Task 1: Stable support events and contact-to-contact segmentation

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/fragments.py`
- Create: `tests/python/test_oracle_fragments.py`

**Interfaces:**
- Consumes: `CanonicalClip`.
- Produces:
  `SupportPhase(str, Enum)`,
  `SupportEvent(frame: int, phase: SupportPhase, stable_frames: int)`,
  `FragmentRef(fragment_id, clip_id, start_frame, stop_frame, entry_phase, exit_phase)`,
  `stable_support_events(clip, config)`,
  and `segment_clip(clip, config)`.

- [ ] **Step 1: Write support-event and half-open-range tests**

```python
def test_segmenter_splits_only_after_stable_contact_and_keeps_context():
    clip = clip_with_contact_runs("DDDDLLLLRRRRDDDD")
    refs = segment_clip(
        clip,
        SegmentationConfig(stable_frames=3, context_before=2, context_after=2),
    )
    self.assertEqual(
        [(r.entry_phase, r.exit_phase) for r in refs],
        [
            (SupportPhase.DOUBLE, SupportPhase.LEFT),
            (SupportPhase.LEFT, SupportPhase.RIGHT),
            (SupportPhase.RIGHT, SupportPhase.DOUBLE),
        ],
    )
    self.assertTrue(all(r.start_frame < r.stop_frame for r in refs))
    self.assertTrue(all(r.stop_frame <= clip.frame_count for r in refs))

def test_one_frame_contact_flicker_does_not_create_fragment():
    clip = clip_with_contact_runs("DDDDLDRRRR")
    self.assertNotIn(SupportPhase.LEFT, [e.phase for e in stable_support_events(clip, config)])
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_fragments -v
```

- [ ] **Step 3: Implement deterministic segmentation**

Map `(False,False)`, `(True,False)`, `(False,True)`, and `(True,True)` to
`FLIGHT`, `LEFT`, `RIGHT`, and `DOUBLE`. Run-length encode phase labels, discard
runs shorter than `stable_frames`, and create one semantic range between adjacent
stable events. Store separate `source_start/source_stop` and padded
`context_start/context_stop`; do not change contact event frames.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/fragments.py \
  tests/python/test_oracle_fragments.py
git commit -m "feat: segment canonical motion by stable contacts"
```

### Task 2: Fragment descriptors and exact transition features

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/fragment_features.py`
- Create: `tests/python/test_oracle_fragment_features.py`

**Interfaces:**
- Consumes: `CanonicalClip`, `FragmentRef`.
- Produces:
  `FragmentFeatures`,
  `extract_fragment_features(clip, ref)`,
  and `transition_distance(left_exit, right_entry, scales)`.
- Descriptor fields include entry/exit local pose and velocity, support anchors,
  local root displacement/yaw/duration, movement/facing, swing path, pelvis path,
  terrain signature, swept bounds, and derivative maxima.

- [ ] **Step 1: Write invariance and asymmetry tests**

```python
def test_descriptor_is_global_translation_and_yaw_invariant():
    source = synthetic_walk_fragment()
    moved = rigidly_place_clip(source, translation=(8.0, -3.0, 0.0), yaw=1.2)
    np.testing.assert_allclose(
        extract_fragment_features(source.clip, source.ref).descriptor,
        extract_fragment_features(moved.clip, moved.ref).descriptor,
        atol=1e-5,
    )

def test_left_and_right_support_anchors_are_not_silently_merged():
    left = extract_fragment_features(left_support_clip(), left_ref())
    right = extract_fragment_features(right_support_clip(), right_ref())
    self.assertNotEqual(left.entry_phase, right.entry_phase)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_fragment_features -v
```

- [ ] **Step 3: Implement robot-local descriptors**

Use entry-root yaw to express all planar displacements and facing directions.
Store support transforms as position plus wxyz orientation. Compute swept volumes
from every frame, not just endpoints. Derive terrain signature from exact source
contact surfaces: support-height delta, normals, tread/run dimensions when
measurable, and swing clearance.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/fragment_features.py \
  tests/python/test_oracle_fragment_features.py
git commit -m "feat: extract contact fragment transition features"
```

### Task 3: Data-derived joint warp neighborhoods

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/warp_envelope.py`
- Create: `tests/python/test_oracle_warp_envelope.py`

**Interfaces:**
- Consumes: audited fragment feature rows and offline validation outcomes.
- Produces:
  `WarpParameters(dx_m, dy_m, dz_m, dyaw_rad, duration_scale)`,
  `ValidatedWarpSample(parameters, accepted, audit_hash)`,
  `WarpNeighborhood(fragment_id, samples, scale, interpolation_radius)`,
  `WarpNeighborhood.contains(parameters)`,
  and `build_warp_neighborhoods(features, validator, config)`.

- [ ] **Step 1: Write joint-distribution rejection tests**

```python
def test_marginally_valid_but_jointly_unseen_warp_is_rejected():
    samples = [
        accepted_warp(dx=0.10, dz=0.00),
        accepted_warp(dx=0.00, dz=0.10),
    ]
    neighborhood = WarpNeighborhood.from_samples(
        "f0", samples, scale=np.array([0.1, 0.1, 0.1, 0.2, 0.1]),
        interpolation_radius=0.25,
    )
    self.assertTrue(neighborhood.contains(params(dx=0.10, dz=0.00)))
    self.assertFalse(neighborhood.contains(params(dx=0.10, dz=0.10)))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_warp_envelope -v
```

- [ ] **Step 3: Implement sampled connected neighborhoods**

Generate candidate perturbations around observed same-semantics fragments, run the
offline mechanical warp validator, retain accepted samples, connect samples only
when their normalized distance is within `interpolation_radius` and the connecting
midpoint also passes validation, and keep the component containing the nominal
fragment. `contains()` requires proximity to that validated component; it must not
use an independent min/max box.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/warp_envelope.py \
  tests/python/test_oracle_warp_envelope.py
git commit -m "feat: validate joint motion warp neighborhoods"
```

### Task 4: Immutable fragment bank

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/fragment_store.py`
- Create: `tests/python/test_oracle_fragment_store.py`

**Interfaces:**
- Consumes: frozen corpus hash, `FragmentRef`, `FragmentFeatures`,
  `WarpNeighborhood`.
- Produces:
  `FragmentBank`,
  `build_fragment_bank(corpus, output, config) -> FragmentBankManifest`,
  and `load_fragment_bank(path, expected_corpus_hash) -> FragmentBank`.

- [ ] **Step 1: Write provenance and no-overwrite tests**

```python
def test_fragment_bank_binds_every_range_to_frozen_corpus_hash():
    manifest = build_fragment_bank(corpus, output, config)
    bank = load_fragment_bank(output, corpus.manifest_sha256)
    self.assertEqual(bank.corpus_sha256, corpus.manifest_sha256)
    self.assertTrue(all(f.clip_id in corpus.clip_ids for f in bank.fragments))
    with self.assertRaises(ContractError):
        load_fragment_bank(output, "0" * 64)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_fragment_store -v
```

- [ ] **Step 3: Implement array-backed bank**

Store compact numeric descriptor matrices in NPZ and canonical sorted metadata in
JSON. Pose frames remain in the canonical clip store and are referenced by
`clip_id/start/stop`; do not duplicate full trajectories per fragment. Bind the
bank to corpus, audit, coverage, segmentation-config, and warp-validation hashes.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/fragment_store.py \
  tests/python/test_oracle_fragment_store.py
git commit -m "feat: publish immutable contact fragment bank"
```

### Task 5: Robot-centred request and privileged flat providers

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/request.py`
- Create: `tests/python/test_oracle_request.py`

**Interfaces:**
- Produces:
  `MotionRequest(current, trajectory, terrain, footholds, constraints)`,
  `LocalTrajectory(time_s, position_local, velocity_local, facing_local,
  facing_yaw_rate_rad_s)`,
  `ScriptedGlobalProvider.build(global_state, global_route) -> MotionRequest`,
  and `CriticallyDampedTwoStickProvider.preview(state, input, horizon)`.

- [ ] **Step 1: Write recentering and no-global-leak tests**

```python
def test_same_world_route_becomes_robot_centred_request_after_pose_change():
    first = provider.build(global_state(x=1.0, yaw=0.0), route)
    second = provider.build(global_state(x=2.0, yaw=np.pi / 2), route)
    np.testing.assert_allclose(first.trajectory.position_local[0], (0.0, 0.0))
    np.testing.assert_allclose(second.trajectory.position_local[0], (0.0, 0.0))
    self.assertFalse(hasattr(second, "global_root_position"))

def test_movement_and_facing_are_independent():
    request = two_stick_preview(travel=(1.0, 0.0), facing=(0.0, 1.0))
    self.assertGreater(request.trajectory.velocity_local[1, 0], 0.0)
    self.assertGreater(request.trajectory.facing_local[1, 1], 0.0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_request -v
```

- [ ] **Step 3: Implement immutable requests**

The scripted provider may use exact global root and a complete route, but only the
robot-centred request crosses into matching. The interactive provider reuses the
validated Takara critically damped spring equations, emits local position/velocity
and facing samples plus the spring's exact instantaneous facing yaw-rate state,
supports a centred-stick stop, and stores no global root position. The yaw rate is
not reconstructed later with a finite difference.

- [ ] **Step 4: Run request and existing command-filter tests**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_request \
  tests.python.test_sonic_terrain_interactive -v
```

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/request.py \
  tests/python/test_oracle_request.py
git commit -m "feat: isolate robot-centred oracle requests"
```

### Task 6: Compatibility graph and deterministic full-route flat search

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/motion_graph.py`
- Create: `tests/python/test_oracle_motion_graph.py`

**Interfaces:**
- Consumes: `FragmentBank`, `MotionRequest`.
- Produces:
  `TransitionEdge(source_id, target_id, placement, warp, cost, margins)`,
  `MotionPlan(edges, terminal_fragment_id, total_cost)`,
  `UnsupportedReason(code, message, missing_transition, violated_margin)`,
  `build_compatibility_graph(bank, config)`,
  and `search_complete_route(graph, request, config) -> SearchResult`.

- [ ] **Step 1: Write known-optimum and no-prefix tests**

```python
def test_search_chooses_complete_lower_cost_path_and_valid_stop():
    graph = tiny_graph(
        edges=[("idle", "bad_prefix", 0.1), ("idle", "walk", 0.2),
               ("walk", "stop", 0.2)]
    )
    result = search_complete_route(graph, straight_request(), config)
    self.assertTrue(result.supported)
    self.assertEqual(result.plan.fragment_ids, ("idle", "walk", "stop"))

def test_dead_end_prefix_is_not_published():
    result = search_complete_route(dead_end_graph(), straight_request(), config)
    self.assertFalse(result.supported)
    self.assertIsNone(result.plan)
    self.assertEqual(result.reason.code, "no_safe_terminal_stop")
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/bodow/miniconda3/envs/cloc3/bin/python \
  -B -m unittest tests.python.test_oracle_motion_graph -v
```

- [ ] **Step 3: Implement graph construction and search**

Create an edge only when support phase, planted-foot transform, entry pose/velocity,
joint limits, and warp neighborhood are compatible. Search state includes route
sample index, root SE(2), support anchors, current fragment, elapsed time, and cost.
Use deterministic A* with a zero admissible heuristic first; tie-break by full
fragment ID. Require terminal double-support idle/stop and return structured
rejection counts.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass and repeated runs are identical.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/motion_graph.py \
  tests/python/test_oracle_motion_graph.py
git commit -m "feat: search complete contact motion routes"
```

### Task 7: Inertialization and flat contact reconstruction

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/inertialization.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/flat_reconstruction.py`
- Create: `tests/python/test_oracle_inertialization.py`
- Create: `tests/python/test_oracle_flat_reconstruction.py`

**Interfaces:**
- Consumes: `MotionPlan`, canonical clips, and exact flat support plane.
- Produces:
  `PoseOffsets.from_boundary(previous, incoming)`,
  `decay_offsets(offsets, time_s, half_life_s)`,
  and `reconstruct_flat(plan, corpus, config) -> ReconstructedMotion`.

- [ ] **Step 1: Write boundary continuity and stance-lock tests**

```python
def test_inertialized_transition_is_pose_and_velocity_continuous():
    output = reconstruct_flat(two_fragment_plan_with_offset(), corpus, config)
    seam = output.seams[0]
    self.assertLess(np.linalg.norm(output.root_position[seam] - output.root_position[seam - 1]), 0.03)
    self.assertLess(np.max(np.abs(output.joint_velocity[seam] - output.joint_velocity[seam - 1])), 0.5)

def test_stance_foot_remains_fixed_while_offsets_decay():
    output = reconstruct_flat(left_stance_transition_plan(), corpus, config)
    stance = output.sole_position_world[output.contact[:, 0], 0]
    self.assertLess(np.max(np.linalg.norm(stance - stance[0], axis=1)), 0.003)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_inertialization \
  tests.python.test_oracle_flat_reconstruction -v
```

- [ ] **Step 3: Implement reconstruction**

Place each fragment by its entry root/support anchor. Apply only a
`WarpNeighborhood.contains()` warp. Use critically damped positional, rotational,
joint, and velocity offset decay. During stance, solve named foot Jacobians with
damped least squares for root plus leg joints so the world contact transform is
fixed; enforce joint limits after each iteration and fail the reconstruction if
residual exceeds 3 mm or 1 degree.

- [ ] **Step 4: Run tests**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/inertialization.py \
  sonic/python/mm_sonic/terrain_oracle/flat_reconstruction.py \
  tests/python/test_oracle_inertialization.py \
  tests/python/test_oracle_flat_reconstruction.py
git commit -m "feat: reconstruct grounded inertialized flat motion"
```

### Task 8: Flat oracle orchestration and acceptance suite

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/flat_oracle.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/fragment_cli.py`
- Create: `tests/python/test_oracle_flat_oracle.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Produces:
  `FlatMotionOracle.synthesize(request) -> SynthesisResult`,
  `load_flat_audit(path: Path) -> FlatAudit`,
  CLI `build-fragments`, `plan-flat`, `render-flat`, and `evaluate-flat`.
- `SynthesisResult` is exactly one of a successful complete trajectory or an
  unsupported result with structured reason and safe-stop plan.

- [ ] **Step 1: Write supported behavior matrix test**

```python
def test_flat_oracle_supports_required_route_matrix_without_freeze():
    routes = (
        straight(), start_stop(), stop_restart(), left_turn(), right_turn(),
        forward_to_backward(), lateral_left(), lateral_right(),
        movement_facing_decoupled(), spin_in_place(),
    )
    for route in routes:
        with self.subTest(route=route.name):
            result = oracle.synthesize(route.request)
            self.assertTrue(result.supported)
            audit = load_flat_audit(result.audit_path)
            self.assertTrue(audit.complete)
            self.assertEqual(audit.repeated_nonidle_pose_frames, 0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_oracle_flat_oracle -v
```

- [ ] **Step 3: Implement orchestration and structured diagnostics**

Load and hash-check the corpus, coverage, and fragment bank; search; reconstruct;
run the independent kinematic evaluator from disk; and publish only a passing
trajectory. Record chosen source fragments, frame mapping, placement, warp,
inertialization, foot correction, route error, and search rejection histogram.

- [ ] **Step 4: Run all phase-2 tests and regression tests**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest discover -s tests/python -p 'test_oracle_*.py' -v
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m unittest tests.python.test_sonic_terrain_kinematic_evaluation \
  tests.python.test_sonic_offline_corpus -v
git diff --check
```

- [ ] **Step 5: Run the real flat matrix**

```bash
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m mm_sonic.terrain_oracle.fragment_cli build-fragments \
  --corpus sonic/runs/terrain-oracle-v1/corpus \
  --coverage sonic/runs/terrain-oracle-v1/coverage.json \
  --output sonic/runs/terrain-oracle-v1/fragments
PYTHONPATH=sonic/python:. /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -B -m mm_sonic.terrain_oracle.fragment_cli evaluate-flat \
  --bank sonic/runs/terrain-oracle-v1/fragments \
  --suite all-flat-v1 \
  --output sonic/runs/terrain-oracle-v1/flat-eval
```

Inspect the robot-mesh videos and route/contact plots for every route, not only the
aggregate metrics.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/flat_oracle.py \
  sonic/python/mm_sonic/terrain_oracle/fragment_cli.py \
  tests/python/test_oracle_flat_oracle.py sonic/README.md
git commit -m "feat: synthesize complete flat contact motion plans"
```

## Phase-2 exit gate

Do not add terrain until:

- every declared flat route completes with a planned stop;
- straight, turns, starts/stops, forward-to-backward, lateral, omnidirectional,
  spin, and movement/facing-decoupled routes are tested symmetrically;
- stance drift is at most 3 mm and stance orientation error at most 1 degree;
- there are no non-idle repeated-pose freezes, skating, hovering, or visible seams;
- robot-mesh videos are at least as smooth and responsive as the retained flat-only
  baseline; and
- unsupported requests produce a valid safe stop and structured reason.
