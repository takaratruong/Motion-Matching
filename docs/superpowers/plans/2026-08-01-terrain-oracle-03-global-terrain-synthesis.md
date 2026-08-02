# Terrain Oracle Phase 3: Privileged Global Terrain Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize a complete, smooth, collision-free G1 kinematic trajectory over
every frozen supported terrain route using exact global state, route, and terrain.

**Architecture:** Convert the exact global mesh into support surfaces and a graph of
alternative footholds. A deterministic full-horizon search jointly chooses each
foothold, contact fragment, placement, and validated warp. Reconstruction uses
contact-knot-preserving time warp, inertialization, and multi-frame whole-body IK.
An independent disk-level audit rejects the trajectory and feeds a forbidden edge
back to search until a complete audited solution or structured unsupported result
is obtained.

**Tech Stack:** Python 3.10+, NumPy, SciPy 1.15.3, MuJoCo, OpenUSD import boundary,
Shapely 2.1 for robust support polygons, unittest.

## Global Constraints

- The exact triangle mesh is the final support and collision authority.
- Height maps and SDFs are optional accelerators and must retain an explicit validity
  mask; zero height is not an invalid sample.
- Footholds require complete-sole containment inside a support polygon eroded by the
  sole footprint.
- Search state carries both planted-foot transforms and support identity.
- Terrain feasibility, swing/body clearance, joint limits, derivative limits, warp
  neighborhoods, and a safe continuation are hard constraints.
- Scripted evaluation plans the complete route before publishing frame zero.
- Offline runtime may be slow; correctness and deterministic diagnostics take
  priority over latency.
- A failed reconstruction or final audit causes global backtracking, not a local
  pose hold.
- SONIC and physics tracking remain out of scope.

---

## File map

- `terrain_oracle/terrain_mesh.py`: exact mesh import, transforms, ray/collision
  queries, support patches.
- `terrain_oracle/route.py`: global route/facing representation and sampling.
- `terrain_oracle/footholds.py`: left/right complete-sole candidate graph.
- `terrain_oracle/terrain_search.py`: coupled full-route foothold/fragment search.
- `terrain_oracle/spatial_warp.py`: validated task-space placement correction.
- `terrain_oracle/time_warp.py`: monotone contact-knot-preserving time maps.
- `terrain_oracle/terrain_warp_catalog.py`: full-IK-audited terrain warp samples.
- `terrain_oracle/contact_ik.py`: multi-frame whole-body contact IK.
- `terrain_oracle/reconstruct.py`: whole-plan reconstruction.
- `terrain_oracle/final_audit.py`: independent disk-level acceptance.
- `terrain_oracle/global_oracle.py`: search/reconstruct/audit/backtrack loop.
- `terrain_oracle/global_cli.py`: mesh/foothold/plan/synthesize CLI.
- `tests/python/test_oracle_terrain_*.py`: phase-3 unit and integration tests.

### Task 1: Exact mesh and support-surface representation

**Files:**
- Modify: `sonic/pyproject.toml`
- Create: `sonic/python/mm_sonic/terrain_oracle/terrain_mesh.py`
- Create: `tests/python/test_oracle_terrain_mesh.py`

**Interfaces:**
- Consumes: phase-1 `CanonicalTerrainMesh`, `TerrainBinding`, and
  `CanonicalMeshQuery`.
- Produces:
  `TerrainMeshIndex.from_canonical(mesh, binding)`,
  `SupportPatch(patch_id, world_from_patch, polygon_xy, normal_world, height_m)`,
  `TerrainMeshIndex.raycast(origin, direction) -> RayHit | None`,
  `TerrainMeshIndex.signed_distance(points) -> np.ndarray`,
  `TerrainMeshIndex.is_watertight_for_collision`,
  `mesh_from_heightfield(height, valid) -> CanonicalTerrainMesh`,
  `extract_support_patches(mesh, config)`,
  and `erode_support_for_sole(patch, sole_polygon_xy)`.

- [ ] **Step 1: Write analytic support and validity tests**

```python
def test_stair_mesh_extracts_distinct_treads_and_complete_sole_erosion():
    mesh = analytic_three_step_mesh(rise=0.18, run=0.33, width=0.60)
    patches = extract_support_patches(mesh, SupportConfig(max_slope_rad=0.2))
    self.assertEqual([round(p.height_m, 2) for p in patches], [0.0, 0.18, 0.36, 0.54])
    eroded = erode_support_for_sole(patches[1], g1_sole_polygon())
    self.assertTrue(eroded.contains(Point(0.16, 0.0)))
    self.assertFalse(eroded.contains(Point(0.01, 0.29)))

def test_invalid_heightfield_sample_is_not_interpreted_as_zero_ground():
    mesh = mesh_from_heightfield(np.zeros((3, 3)), valid=np.zeros((3, 3), bool))
    index = TerrainMeshIndex.from_canonical(mesh, identity_terrain_binding())
    self.assertIsNone(index.raycast((0, 0, 1), (0, 0, -1)))
```

- [ ] **Step 2: Add the isolated oracle dependency**

Add:

```toml
oracle = [
  "imageio[ffmpeg]==2.37.0",
  "joblib==1.5.3",
  "matplotlib==3.10.8",
  "mujoco==3.4.0",
  "scipy==1.15.3",
  "shapely==2.1.1",
  "zarr==2.18.3",
]
```

Create `sonic/.oracle-venv` from the read-only Isaac environment with
`--system-site-packages`, install `-e 'sonic[oracle]'` into that venv, and never
modify the upstream environment.

```bash
/move/u/justingu/miniconda3/envs/isaac6_test/bin/python -m venv \
  --system-site-packages sonic/.oracle-venv
sonic/.oracle-venv/bin/pip install -e 'sonic[oracle]'
sonic/.oracle-venv/bin/python -c \
  'import mujoco, scipy, shapely; assert scipy.__version__ == "1.15.3"'
```

- [ ] **Step 3: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_terrain_mesh -v
```

Expected: import failure for `terrain_oracle.terrain_mesh`.

- [ ] **Step 4: Implement exact mesh operations**

Validate manifold indices and finite vertices; transform meshes with
`RigidTransform`; compute ray/triangle intersection with deterministic tie-breaking;
cluster connected coplanar upward-facing triangles; union their projected polygons
with Shapely; preserve holes; and erode by the reflected G1 sole polygon. Signed
distance sign is confirmed with MuJoCo collision for final audits.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_terrain_mesh -v
git add sonic/pyproject.toml \
  sonic/python/mm_sonic/terrain_oracle/terrain_mesh.py \
  tests/python/test_oracle_terrain_mesh.py
git commit -m "feat: extract exact terrain support surfaces"
```

### Task 2: Complete world route and facing contract

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/route.py`
- Create: `tests/python/test_oracle_route.py`

**Interfaces:**
- Produces:
  `GlobalRoute(time_s, position_world, velocity_world, facing_world_xy)`,
  `GlobalRoute.sample_time(time_s)`,
  `GlobalRoute.sample_arc_length(distance_m)`,
  and `GlobalRoute.to_local(current_root: RigidTransform) -> LocalTrajectory`.

- [ ] **Step 1: Write route interpolation and facing-independence tests**

```python
def test_route_interpolation_preserves_stop_and_right_angle_arc():
    route = rounded_right_angle_route(radius_m=0.5, speed_mps=0.4)
    sampled = route.resample(dt_s=0.02)
    self.assertLess(np.max(np.linalg.norm(np.diff(sampled.velocity_world, axis=0), axis=1)), 0.08)
    np.testing.assert_allclose(sampled.velocity_world[-1], 0.0, atol=1e-7)

def test_facing_can_remain_forward_while_route_moves_laterally():
    route = lateral_route_with_fixed_facing()
    self.assertGreater(np.mean(route.velocity_world[:, 1]), 0.0)
    np.testing.assert_allclose(route.facing_world_xy, (1.0, 0.0), atol=1e-7)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_route -v
```

- [ ] **Step 3: Implement checked route sampling**

Use cubic Hermite interpolation for position/velocity, shortest-angle interpolation
for facing, exact zero velocity at declared stops, and a cumulative monotone arc
length table. `to_local()` uses exact global state inside the provider and emits no
global root field.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_route -v
git add sonic/python/mm_sonic/terrain_oracle/route.py \
  tests/python/test_oracle_route.py
git commit -m "feat: define complete privileged terrain routes"
```

### Task 3: Alternative foothold graph

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/footholds.py`
- Create: `tests/python/test_oracle_footholds.py`

**Interfaces:**
- Consumes: `GlobalRoute`, support patches, fragment-bank step envelopes.
- Produces:
  `Foothold(side, transform_world, support_patch_id, route_progress_m, margins)`,
  `FootholdEdge(source_id, target_id, compatible_fragment_ids)`,
  and `generate_foothold_graph(route, patches, bank, config)`.

- [ ] **Step 1: Write angle, containment, and alternative-path tests**

```python
def test_oblique_stair_approach_has_multiple_contained_alternatives():
    graph = generate_foothold_graph(route_at_yaw(0.45), stair_patches(), bank, config)
    first_step = graph.at_progress(1.0)
    self.assertGreaterEqual(len(first_step), 3)
    self.assertTrue(all(f.margins.sole_boundary_m >= config.minimum_boundary_m for f in first_step))

def test_candidate_with_one_sole_corner_over_riser_is_rejected():
    candidates = candidate_grid_for_patch(narrow_tread_patch(), side=FootSide.LEFT)
    self.assertFalse(any(c.transform_world.translation[0] > tread_back_edge for c in candidates))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_footholds -v
```

- [ ] **Step 3: Implement graph generation**

Sample route progress and yaw bands, intersect with eroded support polygons, and
generate several left/right transforms per region. Create an edge only if at least
one fragment has matching support semantics and its observed step displacement,
height, yaw, surface-normal, and swing-clearance parameters admit a validated warp.
Keep alternatives; do not greedily select one sequence.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_footholds -v
git add sonic/python/mm_sonic/terrain_oracle/footholds.py \
  tests/python/test_oracle_footholds.py
git commit -m "feat: generate complete-sole foothold alternatives"
```

### Task 4: Coupled full-horizon foothold and fragment search

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/terrain_search.py`
- Create: `tests/python/test_oracle_terrain_search.py`

**Interfaces:**
- Consumes: `FootholdGraph`, `FragmentBank`, `TerrainWarpOverlay`, `GlobalRoute`.
- Produces:
  `TerrainSearchState`,
  `PlannedFragment(fragment_id, entry_time_s, foothold_ids, placement, warp)`,
  `TerrainMotionPlan`,
  and `search_terrain_route(inputs, config, forbidden_edges=frozenset())`.

- [ ] **Step 1: Write global-backtracking and deterministic-cost tests**

```python
def test_search_avoids_locally_cheapest_foothold_with_no_descent_suffix():
    result = search_terrain_route(cube_stair_fixture(), config)
    self.assertTrue(result.supported)
    self.assertEqual(result.plan.foothold_ids, ("L0", "R1", "L_wide", "R_down", "stop"))

def test_identical_inputs_produce_byte_identical_plan_json():
    first = search_terrain_route(fixture, config).plan.to_json_bytes()
    second = search_terrain_route(fixture, config).plan.to_json_bytes()
    self.assertEqual(first, second)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_terrain_search -v
```

- [ ] **Step 3: Implement search state, hard gates, and costs**

Search state includes route progress, global root transform, entry pose/velocity
summary, support phase, both planted transforms, current fragment/source, elapsed
time, and audit margins. Expansion jointly selects next foothold, fragment,
placement, spatial warp, and time warp. Reject support mismatch, incomplete sole,
clearance/collision failure, joint/derivative limit, invalid terrain, unsafe suffix,
or warp outside the validated neighborhood. Rank accepted edges by seam,
path/facing, terrain, warp, smoothness, source quality, and repetition costs.

Use deterministic A* initially. The terminal node must be a valid double-support
stop on the final safe surface.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_terrain_search -v
git add sonic/python/mm_sonic/terrain_oracle/terrain_search.py \
  tests/python/test_oracle_terrain_search.py
git commit -m "feat: search complete terrain contact plans"
```

### Task 5: Contact-preserving spatial and temporal warp

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/spatial_warp.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/time_warp.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/terrain_warp_catalog.py`
- Create: `tests/python/test_oracle_spatial_warp.py`
- Create: `tests/python/test_oracle_time_warp.py`

**Interfaces:**
- Produces:
  `spatially_warp_fragment(clip, ref, targets, neighborhood) -> WarpedFragment`,
  `ContactTimeMap(source_time_s, output_time_s, contact_knots)`,
  `time_warp_fragment(fragment, time_map)`,
  and `TerrainWarpOverlay(fragment_id, validated_samples, source_bank_sha256)`.

- [ ] **Step 1: Write planted-anchor, monotonicity, and event-order tests**

```python
def test_spatial_warp_reaches_target_without_moving_existing_stance_anchor():
    warped = spatially_warp_fragment(fragment, ref, target_footholds, neighborhood)
    np.testing.assert_allclose(warped.sole_position_world[fragment.contact[:, 0], 0], left_anchor, atol=0.003)
    np.testing.assert_allclose(warped.touchdown_transform.translation, target.translation, atol=0.003)

def test_time_warp_is_monotone_and_preserves_contact_event_order():
    result = time_warp_fragment(fragment, ContactTimeMap.from_scale(fragment, 1.15))
    self.assertTrue(np.all(np.diff(result.source_time_s) > 0.0))
    self.assertEqual(result.contact_event_names, fragment.contact_event_names)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_spatial_warp tests.python.test_oracle_time_warp -v
```

- [ ] **Step 3: Implement bounded task-space warp**

Define the immutable terrain-warp overlay format without populating unvalidated
samples yet. Distribute root translation/yaw/height correction with cubic curves between contact
knots. Keep current stance anchors fixed and move the next touchdown target;
recompute joints later through IK rather than scaling angles. Build monotone
piecewise cubic source-time maps pinned at every contact event, resample
translations/joints continuously and quaternions with shortest-path SLERP, then
recompute derivatives.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_spatial_warp tests.python.test_oracle_time_warp -v
git add sonic/python/mm_sonic/terrain_oracle/spatial_warp.py \
  sonic/python/mm_sonic/terrain_oracle/time_warp.py \
  sonic/python/mm_sonic/terrain_oracle/terrain_warp_catalog.py \
  tests/python/test_oracle_spatial_warp.py \
  tests/python/test_oracle_time_warp.py
git commit -m "feat: warp motion around fixed contact events"
```

### Task 6: Multi-frame contact-constrained whole-body IK

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/contact_ik.py`
- Create: `tests/python/test_oracle_contact_ik.py`

**Interfaces:**
- Consumes: warped/inertialized seed trajectory, exact G1 model, terrain, contact
  schedule, target footholds.
- Produces:
  `ContactIKConfig`,
  `IKWindowResult`,
  and `solve_contact_ik(seed, constraints, model, terrain, config)`.

- [ ] **Step 1: Write synthetic stance, clearance, and limit tests**

```python
def test_windowed_ik_locks_both_stance_feet_and_clears_step():
    result = solve_contact_ik(seed_over_step(), constraints, model, terrain, config)
    self.assertTrue(result.converged)
    self.assertLess(result.maximum_stance_translation_error_m, 0.003)
    self.assertLess(result.maximum_stance_orientation_error_rad, np.deg2rad(1.0))
    self.assertGreaterEqual(result.minimum_swing_clearance_m, 0.0)
    self.assertEqual(result.joint_limit_violations, 0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_contact_ik -v
```

- [ ] **Step 3: Implement sliding-window least squares**

Optimize root translation/rotation increments and 29 joint increments for all frames
in an overlapping window. Residual blocks, in strict priority order, are stance
position/orientation, collision barrier, swing clearance/touchdown, joint
position/velocity limits, root/pelvis/torso smoothness, source deviation, and seam
correction smoothness. Use analytic MuJoCo body Jacobians where available and
SciPy `least_squares`; warm-start overlaps and blend only unconstrained corrections.
Return failure when exact post-solve residuals exceed acceptance thresholds.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_contact_ik -v
git add sonic/python/mm_sonic/terrain_oracle/contact_ik.py \
  tests/python/test_oracle_contact_ik.py
git commit -m "feat: solve multi-frame terrain contact IK"
```

### Task 7: Whole-plan reconstruction and provenance

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/reconstruct.py`
- Create: `tests/python/test_oracle_reconstruct.py`

**Interfaces:**
- Consumes: `TerrainMotionPlan`, corpus, fragment bank, exact terrain/model.
- Produces:
  `build_terrain_warp_overlay(bank, corpus, model, terrain_cases, config)`,
  and `ReconstructedMotion` with canonical output arrays, root/joint/body
  velocities and accelerations, plus per-frame
  `source_fragment_id`, `source_frame_float`, placement, spatial/time warp,
  inertialization, IK correction, foothold, and support IDs.

- [ ] **Step 1: Write seam and provenance tests**

```python
def test_reconstruction_has_complete_frame_provenance_and_no_pose_seam():
    motion = reconstruct(plan, corpus, bank, model, terrain, config)
    self.assertEqual(len(motion.provenance), motion.frame_count)
    self.assertTrue(all(row.source_fragment_id for row in motion.provenance))
    self.assertLess(motion.metrics.maximum_root_seam_velocity_jump_mps, 0.2)
    self.assertLess(motion.metrics.maximum_joint_seam_step_rad, 0.25)

def test_terrain_warp_overlay_contains_only_full_ik_audited_samples():
    overlay = build_terrain_warp_overlay(
        bank, corpus, model, terrain_cases, config
    )
    self.assertTrue(all(sample.audit_passed for sample in overlay.samples))
    self.assertFalse(overlay.contains(jointly_unseen_height_and_yaw_warp()))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_reconstruct -v
```

- [ ] **Step 3: Implement ordered reconstruction**

First build the terrain-warp overlay by running every candidate sample through
spatial/time warp, contact IK, and the independent fragment audit; retain only the
connected component containing the nominal sample. For each planned fragment: load
exact source rows, place, spatially warp, time warp,
inertialize against the prior output, solve overlapping contact IK, recompute all
FK, velocities, and accelerations, and append only after local constraints pass. Serialize the
trajectory and provenance atomically for the final auditor.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_reconstruct -v
git add sonic/python/mm_sonic/terrain_oracle/reconstruct.py \
  sonic/python/mm_sonic/terrain_oracle/terrain_warp_catalog.py \
  tests/python/test_oracle_reconstruct.py
git commit -m "feat: reconstruct complete terrain motion plans"
```

### Task 8: Independent final audit and backtracking oracle

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/final_audit.py`
- Create: `sonic/python/mm_sonic/terrain_oracle/global_oracle.py`
- Create: `tests/python/test_oracle_final_audit.py`
- Create: `tests/python/test_oracle_global_oracle.py`

**Interfaces:**
- Produces:
  `audit_saved_motion(path, model, terrain, route, thresholds) -> FinalAudit`,
  `FailedPlanEdge(fragment_id, foothold_ids, reason, interval)`,
  and `GlobalTerrainOracle.synthesize(request) -> SynthesisResult`.

- [ ] **Step 1: Write independent-recompute and backtracking tests**

```python
def test_auditor_detects_disk_tamper_not_visible_in_planner_cache():
    saved = write_passing_motion()
    tamper_joint_frame(saved, frame=20, joint=3, delta=0.5)
    report = audit_saved_motion(saved, model, terrain, route, thresholds)
    self.assertFalse(report.passed)
    self.assertEqual(report.failures[0].code, "joint_step")

def test_oracle_forbids_failed_edge_and_replans_complete_route():
    result = oracle.synthesize(fixture_request_with_bad_cheapest_edge())
    self.assertTrue(result.supported)
    self.assertEqual(result.attempt_count, 2)
    self.assertIn("bad_edge", result.forbidden_edge_ids)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_final_audit tests.python.test_oracle_global_oracle -v
```

- [ ] **Step 3: Implement independent audit**

Reload arrays from disk, run fresh MuJoCo FK/collision, reconstruct contact intervals,
and recompute route completion, stance translation/orientation drift, sole/body
penetration, support, swing/body clearance, joint/root derivatives, seam changes,
non-idle repeated poses, and terminal stop. Do not read planner pass/fail booleans.

- [ ] **Step 4: Implement bounded backtracking**

`GlobalTerrainOracle` iterates search → reconstruct → save → independent audit.
Translate each failure to the smallest responsible plan edge, add it to
`forbidden_edges`, and restart complete search. Stop with structured unsupported
reason only when search is exhausted or the configured deterministic attempt cap is
reached; preserve every failed diagnostic artifact.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_final_audit tests.python.test_oracle_global_oracle -v
git add sonic/python/mm_sonic/terrain_oracle/final_audit.py \
  sonic/python/mm_sonic/terrain_oracle/global_oracle.py \
  tests/python/test_oracle_final_audit.py \
  tests/python/test_oracle_global_oracle.py
git commit -m "feat: audit and backtrack global terrain synthesis"
```

### Task 9: Terrain CLI and covered-class integration matrix

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/global_cli.py`
- Create: `tests/python/test_oracle_global_cli.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Produces CLI subcommands:
  `extract-mesh`, `show-support`, `plan-footholds`, `build-terrain-warps`,
  `plan-route`, `synthesize`, and `audit`.

- [ ] **Step 1: Write synthetic CLI route test**

```python
def test_cli_synthesizes_flat_step_slope_stair_and_descent(tmp_path):
    exit_code = main([
        "synthesize", "--fixture", "mixed-supported-v1",
        "--output", str(tmp_path / "result"),
    ])
    self.assertEqual(exit_code, 0)
    report = json.loads((tmp_path / "result" / "final-audit.json").read_text())
    self.assertTrue(report["passed"])
    self.assertEqual(report["route_completion_fraction"], 1.0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_global_cli -v
```

- [ ] **Step 3: Implement CLI and immutable run bundles**

Every invocation records corpus/bank/coverage/model/terrain/route/config hashes,
search attempts, selected plan, reconstructed arrays, final audit, and failure
artifacts. Exit `0` only for audited completion, `2` for invalid input, and `3` for
valid but unsupported synthesis.

- [ ] **Step 4: Run all phase-3 tests**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_oracle_*.py' -v
git diff --check
```

- [ ] **Step 5: Run the real covered-class canaries**

Run one frozen representative route for each supported stratum: flat connector,
curb up/down, slope up/down/cross-slope when covered, stair ascent/descent at
straight and oblique approaches, landing, platform traversal, and flat continuation.
Generate robot-mesh videos before claiming the gate.

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B \
  -m mm_sonic.terrain_oracle.global_cli build-terrain-warps \
  --corpus sonic/runs/terrain-oracle-v1/corpus \
  --bank sonic/runs/terrain-oracle-v1/fragments \
  --coverage sonic/runs/terrain-oracle-v1/coverage.json \
  --output sonic/runs/terrain-oracle-v1/terrain-warps
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B \
  -m mm_sonic.terrain_oracle.global_cli synthesize \
  --suite covered-class-canary-v1 \
  --corpus sonic/runs/terrain-oracle-v1/corpus \
  --bank sonic/runs/terrain-oracle-v1/fragments \
  --terrain-warps sonic/runs/terrain-oracle-v1/terrain-warps \
  --output sonic/runs/terrain-oracle-v1/terrain-canaries
```

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/global_cli.py \
  tests/python/test_oracle_global_cli.py sonic/README.md
git commit -m "feat: expose privileged global terrain synthesis"
```

## Phase-3 exit gate

Do not build the interactive viewer until:

- every frozen supported terrain/action stratum has at least one complete audited
  route;
- ascent and descent, straight and oblique approach, landing, and return to flat are
  represented;
- stance drift is at most 3 mm and 1 degree, sole penetration at most 2 mm, and
  non-foot/body penetration is zero;
- the audit reports no freezes, hovering, skating, joint-limit violations, or
  unannotated loss of support;
- failed cheapest candidates demonstrably backtrack to complete alternatives;
- repeated runs are deterministic; and
- robot-mesh videos look natural enough to be plausible clean references, not merely
  numerically valid IK.
