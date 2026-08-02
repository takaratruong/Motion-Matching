# Terrain Oracle Phase 4: Benchmark, Interactive Viewer, and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the privileged oracle on frozen held-out terrain suites, expose it
through the existing Switch-controller browser workflow, and export clean,
provenance-complete references for SONIC and diffusion-policy experiments.

**Architecture:** Freeze deterministic supported and negative benchmark manifests
before tuning. Evaluate each synthesized trajectory mechanically from disk and
visually with robot-mesh videos and route/contact plots. The interactive frontend
keeps playing a complete valid plan while one background worker replans from exact
global state; plans switch only at compatible contact events. Exports separate
privileged planning data, deployable robot-centred command tensors, and target
kinematics so later privilege ablations cannot leak global state.

**Tech Stack:** Python 3.10+, NumPy, SciPy, MuJoCo offscreen rendering, Matplotlib,
FFmpeg/ImageIO, existing TML browser Gamepad frontend, unittest.

## Global Constraints

- Supported, negative, route, terrain, and seed manifests are frozen before tuning.
- The frozen coverage envelope cannot shrink in response to benchmark failures.
- Mechanical success and visual quality are both required.
- Interactive movement and facing commands are robot-centred two-stick commands.
- Exact global state and complete terrain may be used only inside the privileged
  provider/planner.
- The learned-policy command export contains no root world position, global route,
  global foothold coordinates, or odometry.
- The current complete valid interactive plan continues until a replacement is
  ready; stale partial results are discarded.
- Centred travel stick requests a planned stop.
- Large benchmark outputs, videos, and trajectories remain outside Git.
- SONIC tracking and policy training are downstream consumers, not acceptance
  evidence for the kinematic oracle.

---

## File map

- `terrain_oracle/procedural.py`: bounded procedural terrain and mixed-course
  generation.
- `terrain_oracle/benchmark_manifest.py`: frozen supported/negative suite contracts.
- `terrain_oracle/benchmark.py`: deterministic execution and aggregation.
- `terrain_oracle/visualize.py`: robot-mesh videos, maps, traces, contact sheets.
- `terrain_oracle/interactive_backend.py`: asynchronous plan lifecycle and handoff.
- `terrain_oracle/interactive_viewer.py`: existing browser/Gamepad integration.
- `sonic/launch_privileged_terrain_oracle.sh`: SSH/browser launcher.
- `terrain_oracle/export.py`: canonical, SONIC, task-4, task-12, and command exports.
- `terrain_oracle/ablation.py`: replaceable privilege providers and leak checks.
- `terrain_oracle/report_cli.py`: freeze/evaluate/render/export/ablate CLI.
- `sonic/schemas/terrain_oracle_benchmark_v1.schema.json`: benchmark manifest.
- `sonic/schemas/terrain_oracle_result_v1.schema.json`: result schema.
- `tests/python/test_oracle_*`: phase-4 tests.

### Task 1: Bounded procedural terrains and mixed routes

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/procedural.py`
- Create: `tests/python/test_oracle_procedural.py`

**Interfaces:**
- Consumes: frozen coverage ranges.
- Produces:
  `ProceduralCourse(course_id, mesh, route, parameter_record)`,
  `generate_course(spec, seed, coverage)`,
  `cube_stair_course(rise_m, run_m, steps, width_m, platform_size_m)`,
  and
  `mixed_course(sequence: tuple[str, ...], parameters: Mapping[str, float], seed: int)`.

- [ ] **Step 1: Write envelope and cube-course tests**

```python
def test_generator_samples_only_inside_frozen_joint_coverage():
    course = generate_course(spec, seed=20260801, coverage=coverage)
    self.assertTrue(coverage.contains_course_parameters(course.parameter_record))
    self.assertEqual(course.parameter_record["seed"], 20260801)

def test_cube_stairs_support_up_platform_turn_and_down_route():
    course = cube_stair_course(
        rise_m=0.18,
        run_m=0.33,
        steps=3,
        width_m=0.65,
        platform_size_m=1.5,
    )
    self.assertEqual(course.route.events, ("approach", "ascend", "platform", "turn", "descend", "stop"))
    self.assertTrue(course.mesh.is_watertight_for_collision)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_procedural -v
```

- [ ] **Step 3: Implement deterministic generation**

Generate flat, curb, slope, cross-slope, stair, platform, cube-stair, and mixed
courses only from jointly connected coverage cells. Derive every route from exact
geometry with approach, traversal, landing, exit, and stop margins. Store the full
parameter vector and mesh hash; identical seed/spec/coverage hashes produce
identical vertices, faces, and route samples.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_procedural -v
git add sonic/python/mm_sonic/terrain_oracle/procedural.py \
  tests/python/test_oracle_procedural.py
git commit -m "feat: generate covered procedural terrain courses"
```

### Task 2: Frozen supported and negative benchmark manifests

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/benchmark_manifest.py`
- Create: `sonic/schemas/terrain_oracle_benchmark_v1.schema.json`
- Create: `tests/python/test_oracle_benchmark_manifest.py`

**Interfaces:**
- Produces:
  `BenchmarkCase(case_id, expected, course, route, seed, stratum)`,
  `BenchmarkManifest`,
  `build_supported_manifest(coverage, seed)`,
  `build_negative_manifest(coverage, seed)`,
  and `freeze_benchmark(path, manifest)`.

- [ ] **Step 1: Write immutability, balance, and negative-boundary tests**

```python
def test_supported_manifest_is_stratified_not_source_count_weighted():
    manifest = build_supported_manifest(ascent_heavy_coverage(), seed=11)
    counts = Counter(case.stratum for case in manifest.cases)
    self.assertEqual(counts["stair_up"], counts["stair_down"])
    self.assertEqual(counts["turn_left"], counts["turn_right"])

def test_negative_cases_cross_exactly_one_frozen_boundary():
    manifest = build_negative_manifest(coverage, seed=12)
    self.assertTrue(all(case.expected == "unsupported" for case in manifest.cases))
    self.assertTrue(all(case.course.outside_dimensions == 1 for case in manifest.cases))
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_benchmark_manifest -v
```

- [ ] **Step 3: Implement manifests and freeze guards**

Supported strata include flat start/stop/restart, forward/back/lateral/diagonal,
left/right turns, in-place turns when covered, movement/facing decoupling,
forward-to-backward, curb up/down, slope up/down/cross-slope, stair up/down across
covered angles and dimensions, landings, flat transitions, cube stairs, and mixed
routes. Negative cases exceed exactly one warp/contact/support dimension or remove a
required support connection. Freeze binds the exact coverage/corpus/bank hashes and
refuses overwrite.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_benchmark_manifest -v
git add sonic/python/mm_sonic/terrain_oracle/benchmark_manifest.py \
  sonic/schemas/terrain_oracle_benchmark_v1.schema.json \
  tests/python/test_oracle_benchmark_manifest.py
git commit -m "feat: freeze balanced terrain oracle benchmarks"
```

### Task 3: Mechanical benchmark runner and acceptance aggregation

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/benchmark.py`
- Create: `sonic/schemas/terrain_oracle_result_v1.schema.json`
- Create: `tests/python/test_oracle_benchmark.py`

**Interfaces:**
- Consumes: frozen benchmark, `GlobalTerrainOracle`.
- Produces:
  `run_case(case, oracle, output) -> CaseResult`,
  `run_benchmark(manifest, oracle, output) -> BenchmarkResult`,
  and `assert_acceptance(result, thresholds)`.

- [ ] **Step 1: Write acceptance and unsupported-stop tests**

```python
def test_supported_suite_requires_every_case_to_pass():
    result = aggregate_results([passing_case(), failing_case("body_penetration")])
    with self.assertRaisesRegex(AcceptanceError, "body_penetration"):
        assert_acceptance(result, thresholds)

def test_negative_case_requires_structured_safe_stop():
    result = run_case(negative_high_step_case(), oracle, output)
    self.assertEqual(result.status, "unsupported")
    self.assertEqual(result.reason.code, "warp_height_outside_coverage")
    self.assertTrue(result.safe_stop_audit.passed)
    self.assertLess(result.completed_route_fraction, 1.0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_benchmark -v
```

- [ ] **Step 3: Implement deterministic execution and thresholds**

Supported acceptance requires 100% completion, zero non-idle repeated-pose frames,
stance drift ≤ 0.003 m, stance orientation error ≤ 1 degree, sole penetration
≤ 0.002 m, zero non-foot penetration, nonnegative configured clearance, no joint or
derivative limit violation, and deterministic rerun hashes. Negative cases require
unsupported classification, the expected boundary reason, and an audited stop on
the last safe support.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_benchmark -v
git add sonic/python/mm_sonic/terrain_oracle/benchmark.py \
  sonic/schemas/terrain_oracle_result_v1.schema.json \
  tests/python/test_oracle_benchmark.py
git commit -m "feat: evaluate frozen terrain oracle suites"
```

### Task 4: Robot-mesh videos and diagnostic figures

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/visualize.py`
- Create: `tests/python/test_oracle_visualize.py`

**Interfaces:**
- Produces:
  `render_robot_video(result_path, output_mp4)`,
  `plot_route_map(result_path, output_png)`,
  `plot_contact_side_view(result_path, output_png)`,
  `plot_fragment_timeline(result_path, output_png)`,
  and `build_contact_sheet(video_paths, output_jpg)`.

- [ ] **Step 1: Write artifact-content tests**

```python
def test_route_map_contains_intent_root_footholds_and_support_polygons():
    receipt = plot_route_map(result_path, output_png)
    self.assertEqual(
        set(receipt.layers),
        {"desired_route", "actual_root", "selected_footholds", "support_polygons"},
    )
    self.assertGreater(output_png.stat().st_size, 1000)

def test_video_receipt_binds_robot_model_terrain_and_motion_hashes():
    receipt = render_robot_video(result_path, output_mp4)
    self.assertEqual(receipt.motion_sha256, sha256(result_path / "motion.npz"))
    self.assertEqual(receipt.terrain_sha256, result.terrain_sha256)
```

- [ ] **Step 2: Run and verify failure**

```bash
MUJOCO_GL=egl PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B \
  -m unittest tests.python.test_oracle_visualize -v
```

- [ ] **Step 3: Implement visual evidence**

Render the articulated G1 mesh and exact terrain, not a stick figure. Overlay desired
route, planned footholds, support contact, and the current source fragment. Generate
top-down intended/actual maps, side-view sole/terrain/pelvis traces, fragment/time
strip, contact/clearance plots, source-versus-corrected comparison, and stratified
contact sheets.

- [ ] **Step 4: Run tests and commit**

```bash
MUJOCO_GL=egl PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B \
  -m unittest tests.python.test_oracle_visualize -v
git add sonic/python/mm_sonic/terrain_oracle/visualize.py \
  tests/python/test_oracle_visualize.py
git commit -m "feat: render terrain oracle visual evidence"
```

### Task 5: Privileged interactive plan lifecycle

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/interactive_backend.py`
- Create: `tests/python/test_oracle_interactive_backend.py`

**Interfaces:**
- Produces:
  `InteractiveOracleState`,
  `PlanGeneration(generation_id, request_hash, plan, status)`,
  `InteractiveOracleBackend.update(input, exact_state, terrain, dt_s)`,
  and `InteractiveStep(pose, active_plan, planning_status, unsupported_reason)`.

- [ ] **Step 1: Write stale-plan, handoff, and stop tests**

```python
def test_old_complete_plan_plays_until_new_generation_reaches_contact_handoff():
    backend = backend_with_slow_planner()
    first = backend.update(forward_input(), state, terrain, 0.02)
    second = backend.update(right_input(), advance(state), terrain, 0.02)
    self.assertEqual(second.active_plan.generation_id, first.active_plan.generation_id)
    finish_new_plan()
    at_handoff = advance_to_compatible_contact(backend)
    self.assertGreater(at_handoff.active_plan.generation_id, first.active_plan.generation_id)

def test_centered_stick_uses_planned_stop_and_never_holds_arbitrary_pose():
    steps = rollout_backend(forward_then_center_trace())
    self.assertEqual(steps[-1].active_plan.terminal_behavior, "double_support_idle")
    self.assertEqual(sum(s.repeated_nonidle_pose for s in steps), 0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_interactive_backend -v
```

- [ ] **Step 3: Implement one-worker privileged replanning**

Use one `ProcessPoolExecutor` worker and monotonically increasing generation IDs.
Integrate the critically damped local two-stick preview into a finite global route
using exact current global state and complete mesh. Continue the active complete plan
while planning. Ignore stale generations, switch only when support phase and planted
anchors match, and request a preplanned stop when commands leave coverage.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_interactive_backend -v
git add sonic/python/mm_sonic/terrain_oracle/interactive_backend.py \
  tests/python/test_oracle_interactive_backend.py
git commit -m "feat: replan privileged motion without freezing"
```

### Task 6: Browser/Switch interactive viewer

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/interactive_viewer.py`
- Create: `sonic/launch_privileged_terrain_oracle.sh`
- Create: `tests/python/test_oracle_interactive_viewer.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Reuses: TML `InteractiveControllerServer` and browser Gamepad API.
- Produces CLI arguments `--corpus`, `--bank`, `--coverage`, `--scene`,
  `--browser-host`, `--browser-port`, `--browser-token`, and `--frontend`.

- [ ] **Step 1: Write controller mapping and runtime identity tests**

```python
def test_switch_axes_map_to_local_travel_and_independent_facing():
    intent = browser_command_targets(
        axes=np.array([0.5, -1.0, -0.75, 0.0]),
        current_yaw_world=1.2,
    )
    np.testing.assert_allclose(intent.local_travel_stick, (0.5, 1.0))
    np.testing.assert_allclose(intent.local_facing_stick, (-0.75, 0.0))

def test_ready_record_names_exact_corpus_bank_and_model_hashes():
    record = build_ready_record(config)
    self.assertEqual(record["schema"], "terrain_oracle_interactive_ready/v1")
    self.assertEqual(record["backend"], "privileged-global-kinematic-oracle")
    self.assertFalse(record["physics"])
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_interactive_viewer -v
```

- [ ] **Step 3: Implement the viewer and launcher**

Render exact terrain, G1 mesh, desired route, selected/candidate footholds, actual
root trail, current support, fragment ID, planning generation, margins, and explicit
unsupported reason. Print localhost URL, SSH tunnel command, opaque token, runtime
hashes, and `CLEAN KINEMATICS, NOT PHYSICS`. Keep X11 keyboard fallback.

- [ ] **Step 4: Run browser smoke**

```bash
TERRAIN_ORACLE_PORT=8765 sonic/launch_privileged_terrain_oracle.sh
```

Expected: launcher prints a ready JSON record and an SSH tunnel command; Switch
controller appears through the browser Gamepad API; flat motion and terrain
transitions remain smooth during an extended session.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/interactive_viewer.py \
  sonic/launch_privileged_terrain_oracle.sh \
  tests/python/test_oracle_interactive_viewer.py sonic/README.md
git commit -m "feat: expose privileged terrain oracle interactively"
```

### Task 7: Clean SONIC and diffusion-policy export

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/export.py`
- Create: `tests/python/test_oracle_export.py`

**Interfaces:**
- Consumes: successful `ReconstructedMotion`, robot-centred `MotionRequest`,
  provenance, terrain.
- Produces:
  `export_canonical_motion`,
  `export_sonic_reference`,
  `export_task4(motions, requests, horizon_frames) -> np.ndarray`,
  `export_task12(requests, knot_offsets=(6, 12, 18, 24)) -> np.ndarray`,
  and `export_oracle_episode`.

- [ ] **Step 1: Write command semantics and privilege-leak tests**

```python
def test_task4_uses_local_velocity_and_facing_without_world_position():
    episode = export_oracle_episode(result, output)
    self.assertEqual(episode.task4.shape[1], 4)
    self.assertNotIn("root_position_world", episode.policy_command_fields)
    self.assertNotIn("global_route", episode.policy_command_fields)

def test_task12_preserves_existing_velocity_yaw_rate_knot_schema():
    request = constant_local_request(
        horizon_frames=25,
        velocity_local_xy=(0.4, -0.1),
        facing_local_xy=(1.0, 0.0),
        facing_yaw_rate_rad_s=0.25,
    )
    task12 = export_task12((request,), knot_offsets=(6, 12, 18, 24))
    self.assertEqual(task12.shape, (1, 12))
    np.testing.assert_allclose(
        task12[0],
        np.tile(np.array([0.4, -0.1, 0.25], dtype=np.float32), 4),
        atol=1e-6,
    )
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_export -v
```

- [ ] **Step 3: Implement separated exports**

The canonical target export includes root/joints/bodies/contacts and world
provenance for tracking. Preserve the existing Task 4 schema
`(velocity_local_forward, velocity_local_lateral, cos_facing_delta,
sin_facing_delta)` and Task 12's four
`(velocity_local_forward, velocity_local_lateral, yaw_rate)` knots at offsets
`(6, 12, 18, 24)`. Export the richer robot-local future position, velocity, and
facing trajectory under separate `oracle_local_trajectory_v1` fields. The deployable
command export includes only robot-local quantities, local terrain, masks, and
behavior flags. Store observed joystick commands separately from inferred commands.
Bind every episode to source fragment, terrain, route, planner, audit, and export
hashes.

- [ ] **Step 4: Run compatibility tests**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_export \
  tests.python.test_sonic_offline_corpus \
  tests.python.test_sonic_torch_motion_data -v
```

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/export.py \
  tests/python/test_oracle_export.py
git commit -m "feat: export oracle motion without policy privilege leaks"
```

### Task 8: Privilege-removal providers and comparable ablations

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/ablation.py`
- Create: `tests/python/test_oracle_ablation.py`

**Interfaces:**
- Produces provider stages:
  `FullGlobalProvider`,
  `RecedingGlobalProvider`,
  `ExactLocalTerrainProvider`,
  `PartialTerrainProvider`,
  `EstimatedRelativeStateProvider`,
  and `NoOdometryTwoStickProvider`.
- Produces `run_ablation(stage, benchmark, output)`.

The exact ordered stages are:

1. `FullGlobalProvider`: full route + complete mesh + true global state;
2. `RecedingGlobalProvider`: finite receding route + complete mesh + true global
   state;
3. `ExactLocalTerrainProvider`: finite route + exact robot-centred terrain + true
   global state;
4. `PartialTerrainProvider`: finite route + partial robot-centred terrain + true
   global state;
5. `EstimatedRelativeStateProvider`: finite route + partial terrain + estimated
   relative state;
6. `NoOdometryTwoStickProvider`: local critically damped two-stick intent +
   partial terrain + no global odometry;
7. the exported oracle data is tracked by SONIC and learned by the diffusion
   policy; and
8. runtime planner removal is tested only if stage 7 preserves benchmark quality.

- [ ] **Step 1: Write forbidden-input and common-request tests**

```python
def test_no_odometry_provider_cannot_access_global_root_or_route():
    provider = NoOdometryTwoStickProvider()
    with self.assertRaises(PrivilegeLeakError):
        provider.build(forbidden_global_context())

def test_every_provider_emits_same_motion_request_type():
    for provider in all_providers():
        self.assertIsInstance(provider.build(provider_fixture(provider)), MotionRequest)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_ablation -v
```

- [ ] **Step 3: Implement explicit capability objects**

Do not pass one broad context dictionary. Give each provider a typed capability
object containing only its allowed inputs. Serialize the capability schema and
provider name into each result. Run the same frozen benchmark and report completion,
contact, clearance, smoothness, route error, planning time, and quality loss versus
the full oracle.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B -m unittest \
  tests.python.test_oracle_ablation -v
git add sonic/python/mm_sonic/terrain_oracle/ablation.py \
  tests/python/test_oracle_ablation.py
git commit -m "feat: measure terrain oracle privilege removal"
```

### Task 9: Report CLI and final privileged-oracle gate

**Files:**
- Create: `sonic/python/mm_sonic/terrain_oracle/report_cli.py`
- Create: `tests/python/test_oracle_report_cli.py`
- Modify: `sonic/README.md`
- Modify: `research.md`
- Modify: `research_log.md`

**Interfaces:**
- Produces CLI subcommands:
  `freeze-benchmark`, `evaluate`, `render`, `contact-sheet`, `export`, and
  `ablate`.

- [ ] **Step 1: Write synthetic complete-report test**

```python
def test_report_cli_requires_metrics_visual_receipts_and_frozen_hashes(tmp_path):
    code = main(["evaluate", "--benchmark", str(bench), "--output", str(out)])
    self.assertEqual(code, 0)
    report = json.loads((out / "benchmark-result.json").read_text())
    self.assertTrue(report["accepted"])
    self.assertTrue(report["visual_evidence_complete"])
    self.assertEqual(report["benchmark_sha256"], sha256(bench))
```

- [ ] **Step 2: Run all phase-4 tests**

```bash
MUJOCO_GL=egl PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B \
  -m unittest discover -s tests/python -p 'test_oracle_*.py' -v
git diff --check
```

- [ ] **Step 3: Freeze the real benchmark before tuning**

```bash
PYTHONPATH=sonic/python:. sonic/.oracle-venv/bin/python -B \
  -m mm_sonic.terrain_oracle.report_cli freeze-benchmark \
  --coverage sonic/runs/terrain-oracle-v1/coverage.json \
  --bank sonic/runs/terrain-oracle-v1/fragments \
  --supported-output sonic/runs/terrain-oracle-v1/benchmarks/supported-v1.json \
  --negative-output sonic/runs/terrain-oracle-v1/benchmarks/negative-v1.json \
  --seed 20260801
```

- [ ] **Step 4: Run supported, negative, visual, and determinism gates**

Run the supported suite twice and require identical plan/result hashes. Run the
negative suite. Render every case, build stratified contact sheets, inspect all
failure videos and a representative full video from every stratum, and record the
inspection disposition.

- [ ] **Step 5: Document exact results and branch points**

Write corpus/bank/coverage/benchmark hashes, case counts, mechanical aggregates,
visual findings, unsupported gaps, timing, reproducible commands, and the exact
next privilege-ablation or SONIC export decision into `research.md` and
`research_log.md`. Do not describe the oracle as arbitrary-terrain capable beyond
the frozen coverage.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/terrain_oracle/report_cli.py \
  tests/python/test_oracle_report_cli.py sonic/README.md \
  research.md research_log.md
git commit -m "test: qualify privileged global terrain oracle"
```

## Phase-4 exit gate

The privileged oracle is ready for SONIC tracking or privilege removal only when:

- 100% of the frozen supported deterministic and seeded procedural suites pass;
- every negative case stops safely and names the missing transition/envelope;
- supported trajectories have no freeze, hover, skate, body penetration, visible
  pose seam, knee collapse, or unnatural pause;
- robot-mesh videos and intended/actual/foothold maps exist for every case;
- extended Switch-controller sessions remain smooth and responsive and never swap
  to partial/stale plans;
- all exports pass the no-global-policy-input test; and
- repeated runs reproduce identical hashes from saved manifests.
