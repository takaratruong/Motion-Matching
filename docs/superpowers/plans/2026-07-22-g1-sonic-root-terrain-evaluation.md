# G1 SONIC Root Terrain Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a paired, identity-pinned experiment that distinguishes root plumbing from learned root conditioning and determines whether the root-conditioned SONIC encoder can traverse the low curb three consecutive times without regressing flat locomotion.

**Architecture:** The evaluator runs three isolated variants: A0 is released SONIC over protocol v1, A1 is the identical released model over root-capable protocol v5, and B is the Stage-A root-conditioned encoder over v5. One deterministic straight-forward command artifact, resampled reference buffer, initial state, terrain scene, runtime binary, and physics schedule are shared across variants and trials; only the registered protocol/model/config identity changes. A1 is the causal negative control, while B must pass flat probes before its 3-of-3 low-curb gate; the default curb remains a separate stress result.

**Tech Stack:** Motion Matching Python orchestration, GEAR C++ deployment, MuJoCo simulation, ZMQ v1/v5, NumPy metrics, canonical JSON evidence, unittest.

## Global Constraints

- Motion Matching evaluation work uses the root implementation branch created from commit `1900933`; GEAR uses the root implementation branch created from `294110cedba01ad764f1e268d57ddf7c1bbf9523`.
- A0 uses packed-pose v1 and the exact released encoder/config hashes.
- A1 uses packed-pose v5 but the same exact released encoder/config hashes as A0; root data reaches `MotionSequence` but cannot enter the released encoder.
- B uses packed-pose v5, the trained Stage-A encoder, the released frozen control decoder, and `observation_config_root_conditioned.yaml`.
- All variants use scene `grail-curb-low`, route `curb-forward`, terrain weight `4.0`, movement model `holden-turn-v1`, and identical initial state and command bytes.
- The deterministic command is 5 chunks standing followed by 25 chunks at `[0.5,0,0]` m/s and identity heading; each chunk is 0.4 seconds.
- Physics remains paused while a new Motion Matching prefix is generated and delivered; the scored physics schedule is identical across variants.
- Primary terrain gate is three consecutive B trials with no reset inside a trial; all three must pass.
- A terrain pass requires exact command/frame/log coverage, target distance at most `0.25 m`, reached time at most `1.25 * nominal`, minimum local pelvis height at least `0.45 m`, minimum pelvis up-dot at least `0.5`, and no pelvis/knee/torso/hand terrain contact.
- Flat regression runs before terrain and must satisfy the existing `evaluate_flat_trial` gates against the released known-good thresholds.
- `grail-curb-default` is diagnostic stress evidence and cannot invalidate or rescue the low-curb primary result.
- Timing is diagnostic because physics pauses during generation, but it is complete: report Motion Matching generation, publication-to-consumption, input arrival age, controller inference, and paused-physics duration with count/p50/p95/p99/max, plus missed-frame and held-frame counts.
- If B fails any flat gate, or fewer than 3 low-curb trials pass, emit `stage_b_required: true`; do not start Stage B automatically.
- Every result includes exact repository, binary, model, config, scene, command, resampled-reference-buffer, and evidence hashes. A run with a changed identity is invalid rather than comparable.

---

## File Map

- Modify `sonic/python/mm_sonic/commands.py`: deterministic straight-curb command artifact.
- Modify `sonic/python/mm_sonic/manual_demo.py`: explicit `--pose-protocol`, v5 root publication, and script selection without changing interactive defaults.
- Modify `sonic/python/mm_sonic/manual_evidence.py`: protocol/model/config/root-stream identity in summaries.
- Create `sonic/python/mm_sonic/root_experiment.py`: experiment variants, identity validation, paired runner, scoring, and aggregate verdict.
- Modify `sonic/python/mm_sonic/metrics.py`: root-reference tracking metrics and three-trial aggregate gate.
- Modify `sonic/python/mm_sonic/cli.py`: `root-experiment` command.
- Create `sonic/configs/experiments/root_conditioning.json`: immutable variant registry and required hashes.
- Create `tests/python/test_sonic_root_experiment.py`: matrix, confound rejection, scoring, and aggregation tests.
- Modify `tests/python/test_sonic_manual_demo.py`: protocol selection and v5 physical-root publication tests.
- Modify `tests/python/test_sonic_metrics.py`: root response and three-trial gate tests.
- Create `docs/superpowers/results/2026-07-22-g1-sonic-root-terrain-evaluation.md`: final evidence and decision.

---

### Task 1: Freeze the Straight-Curb Command and Variant Registry

**Files:**
- Modify: `sonic/python/mm_sonic/commands.py`
- Create: `sonic/configs/experiments/root_conditioning.json`
- Create: `tests/python/test_sonic_root_experiment.py`

**Interfaces:**
- Produces: `straight_curb_command_script() -> tuple[CommandSample, ...]` and registry variants `a0_released_v1`, `a1_released_v5`, `b_root_stage_a_v5`.

- [ ] **Step 1: Write failing command and registry tests**

```python
def test_straight_curb_script_is_exactly_five_stand_then_twenty_five_forward():
    commands = straight_curb_command_script()
    assert len(commands) == 30
    assert [c.chunk_index for c in commands] == list(range(30))
    assert all(c.requested_velocity_mujoco == (0.0, 0.0, 0.0) for c in commands[:5])
    assert all(c.requested_velocity_mujoco == (0.5, 0.0, 0.0) for c in commands[5:])
    assert all(c.desired_heading_mujoco_wxyz == (1.0, 0.0, 0.0, 0.0) for c in commands)

def test_variant_registry_changes_only_registered_treatment_fields():
    registry = load_root_experiment_registry(FIXTURE_REGISTRY)
    a0, a1, b = registry.variants
    assert a0.model_sha256 == a1.model_sha256
    assert a0.config_sha256 == a1.config_sha256
    assert (a0.pose_protocol, a1.pose_protocol, b.pose_protocol) == (1, 5, 5)
    assert b.decoder_sha256 == a0.decoder_sha256
```

- [ ] **Step 2: Run and confirm missing interfaces**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_root_experiment.py' -v`

Expected: import fails for `straight_curb_command_script` or `root_experiment`.

- [ ] **Step 3: Implement the exact command**

```python
def straight_curb_command_script() -> tuple[CommandSample, ...]:
    return tuple(
        CommandSample(
            chunk_index=index,
            requested_velocity_mujoco=(0.0, 0.0, 0.0) if index < 5 else (0.5, 0.0, 0.0),
            desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        for index in range(30)
    )
```

- [ ] **Step 4: Add the registry schema and validator**

The JSON has schema `mm-sonic-root-experiment-registry/v1`, global GEAR/MM commits, scene/route/weight/movement-model, command SHA-256, and three variants. A variant has exact protocol integer, encoder path/hash, decoder path/hash, observation config path/hash, expected encoder input dimension, and expected root range (`null` for A0/A1; `[1762,1792]` for B). The validator requires only protocol to differ A0→A1 and only encoder/config/root range to differ A1→B; decoder hash must be common.

- [ ] **Step 5: Run tests and commit**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_root_experiment.py' -v`

Expected: tests pass.

```bash
git add sonic/python/mm_sonic/commands.py sonic/configs/experiments/root_conditioning.json tests/python/test_sonic_root_experiment.py
git commit -m "test: freeze SONIC root experiment matrix"
```

### Task 2: Select v1 or v5 Publication Explicitly in the Runtime

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_evidence.py`
- Modify: `tests/python/test_sonic_manual_demo.py`

**Interfaces:**
- Consumes: `namespace.pose_protocol` in `{1,5}`, `TargetChunk`, and runtime model/config paths.
- Produces: v1 `PosePublisher` or v5 `RootPosePublisher`; v5 sends `RootTargetBuffer.from_target_chunk(prepared.target)`.

- [ ] **Step 1: Write failing publisher-selection tests**

```python
def test_v5_publication_uses_physical_pelvis_target(self):
    sent = []
    namespace = manual_namespace(pose_protocol=5, script="straight-curb")
    with mock.patch.object(manual_demo, "RootPosePublisher", fake_root_publisher(sent)):
        run_one_prepared_publication(namespace, make_target_chunk())
    assert isinstance(sent[0], RootTargetBuffer)
    np.testing.assert_array_equal(sent[0].body_position, make_target_chunk().physical_pelvis_position)

def test_default_protocol_remains_v1():
    namespace = manual_demo._parser().parse_args([])
    assert namespace.pose_protocol == 1
```

- [ ] **Step 2: Add strict CLI choices and publisher factory**

```python
parser.add_argument("--pose-protocol", type=int, choices=(1, 5), default=1)
parser.add_argument("--script", choices=("legacy-flat", "straight-curb"), default="legacy-flat")

def _publisher_for_protocol(protocol: int, endpoint: str, *, bundle, hand_targets):
    if protocol == 1:
        return PosePublisher(endpoint, bundle=bundle, default_hand_targets=hand_targets)
    if protocol == 5:
        return RootPosePublisher(endpoint, bundle=bundle, default_hand_targets=hand_targets)
    raise ContractError("pose protocol must be 1 or 5")
```

Change the nested `publish` function to accept a `TargetChunk` for logical publications. It sends `target.buffer` for v1 and `RootTargetBuffer.from_target_chunk(target)` for v5. Readiness publication creates a `RootTargetBuffer` from the initial canonical buffer and a repeated initial physical pelvis row in v5. The v1 branch continues sending the same `CanonicalTargetBuffer` bytes.

- [ ] **Step 3: Record the treatment identity**

Add `pose_protocol`, encoder SHA-256, decoder SHA-256, observation-config SHA-256, and `body_position_source` (`untracked-zero` or `physical-pelvis`) to the canonical manual summary. Reject evidence when the declared protocol disagrees with archived message headers.

- [ ] **Step 4: Run runtime and v1 non-regression tests**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_*.py' -v`

Expected: all tests pass and v1 golden bytes remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/manual_demo.py sonic/python/mm_sonic/manual_evidence.py tests/python/test_sonic_manual_demo.py
git commit -m "feat: select root-capable SONIC publication"
```

### Task 3: Add Root-Response Metrics and the Three-Trial Gate

**Files:**
- Modify: `sonic/python/mm_sonic/metrics.py`
- Modify: `tests/python/test_sonic_metrics.py`

**Interfaces:**
- Produces: `RootResponseMetrics`, `root_response_metrics(reference_xyz, actual_xyz)`, and `evaluate_root_conditioning_trials(flat_pass, low_curb_verdicts)`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_root_response_reports_z_tracking_and_rise(self):
    reference = np.array([[0,0,.80], [.1,0,.82], [.2,0,.90]], np.float64)
    actual = np.array([[0,0,.79], [.1,0,.81], [.2,0,.88]], np.float64)
    metric = root_response_metrics(reference, actual)
    self.assertAlmostEqual(metric.reference_rise_m, 0.10)
    self.assertAlmostEqual(metric.actual_rise_m, 0.09)
    self.assertAlmostEqual(metric.z_rmse_m, np.sqrt(np.mean(np.array([.01,.01,.02]) ** 2)))

def test_primary_gate_requires_flat_and_three_consecutive_curb_passes():
    assert evaluate_root_conditioning_trials(True, [passed(), passed(), passed()]).primary_pass
    verdict = evaluate_root_conditioning_trials(True, [passed(), failed(), passed()])
    assert not verdict.primary_pass
    assert verdict.stage_b_required
```

- [ ] **Step 2: Implement immutable metrics and aggregate verdict**

```python
@dataclass(frozen=True)
class RootResponseMetrics:
    xy_rmse_m: float
    z_rmse_m: float
    reference_rise_m: float
    actual_rise_m: float
    rise_ratio: float

def root_response_metrics(reference_xyz: object, actual_xyz: object) -> RootResponseMetrics:
    reference = _finite_xyz(reference_xyz, "reference root")
    actual = _finite_xyz(actual_xyz, "actual root")
    if reference.shape != actual.shape:
        raise ContractError("root response rows must align")
    error = actual - reference
    reference_rise = float(np.max(reference[:, 2]) - reference[0, 2])
    actual_rise = float(np.max(actual[:, 2]) - actual[0, 2])
    return RootResponseMetrics(
        xy_rmse_m=float(np.sqrt(np.mean(np.square(error[:, :2])))),
        z_rmse_m=float(np.sqrt(np.mean(np.square(error[:, 2])))),
        reference_rise_m=reference_rise,
        actual_rise_m=actual_rise,
        rise_ratio=0.0 if reference_rise == 0.0 else actual_rise / reference_rise,
    )
```

The aggregate requires one passing flat verdict and exactly three consecutive low-curb verdicts, all `dynamic_pass`. It emits `stage_b_required = not primary_pass`.

- [ ] **Step 3: Run tests and commit**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_metrics.py' -v`

Expected: all tests pass.

```bash
git add sonic/python/mm_sonic/metrics.py tests/python/test_sonic_metrics.py
git commit -m "feat: score SONIC root terrain response"
```

### Task 4: Implement the Paired Root Experiment Runner

**Files:**
- Create: `sonic/python/mm_sonic/root_experiment.py`
- Modify: `sonic/python/mm_sonic/cli.py`
- Modify: `tests/python/test_sonic_root_experiment.py`

**Interfaces:**
- CLI: `python -m mm_sonic.cli root-experiment --registry PATH --variant NAME --trials N --output-root PATH [--scene-id ID --route-id ID]`.
- Produces per-trial bundles and aggregate `root-experiment-evidence.json` using schema `mm-sonic-root-experiment-evidence/v1`.

- [ ] **Step 1: Write failing orchestration tests with fake processes**

```python
def test_runner_reuses_command_initial_state_and_physics_schedule_across_variants(self):
    root = Path(self._temporary.name)
    evidence = run_fake_matrix(root)
    identities = [trial["paired_identity"] for trial in evidence["trials"]]
    assert len({item["command_sha256"] for item in identities}) == 1
    assert len({item["reference_buffer_sha256"] for item in identities}) == 1
    assert len({item["initial_qpos_sha256"] for item in identities}) == 1
    assert len({item["physics_schedule_sha256"] for item in identities}) == 1

def test_runner_rejects_a1_if_released_model_or_config_differs_from_a0(self):
    tmp_path = Path(self._temporary.name)
    registry = mutate_registry(tmp_path, variant="a1_released_v5", field="encoder_sha256")
    with self.assertRaisesRegex(ContractError, "A0/A1 model identity"):
        run_root_experiment(registry, "a1_released_v5", 1, tmp_path / "runs")
```

- [ ] **Step 2: Implement one isolated trial**

The runner resolves every path without following a changed file after hashing, creates a run-local scene, launches a fresh GEAR and MuJoCo process, streams the exact straight-curb script through the selected protocol, and closes all resources after the trial. It archives target rows, control rows, root reference rows, actual pelvis rows, contacts, timing events, stdout/stderr, command bytes, the exact resampled reference buffer and its SHA-256, initial qpos, physics schedule, model/config files or immutable hashes, and packed publications.

- [ ] **Step 3: Implement scoring and canonical evidence**

Use existing `evaluate_flat_trial` and `evaluate_terrain_trial`. Pair GEAR target-root rows and MuJoCo pelvis rows by scored control tick before calling `root_response_metrics`. Evidence rejects duplicate JSON keys, non-finite values, symlinks, changed files, partial row coverage, mixed process identities, a protocol/header mismatch, or any paired trial whose `reference_buffer_sha256` differs. Timing events include generation start/end, publication, consumption, source timestamp/arrival age, inference start/end, physics pause/release, requested frame, and consumed frame. Summaries use count/p50/p95/p99/max for Motion Matching generation, publication-to-consumption, arrival age, inference, and paused-physics duration, with explicit missed-frame and held-frame counts under `diagnostic_timing`.

- [ ] **Step 4: Register the CLI**

```python
root = subcommands.add_parser("root-experiment")
root.add_argument("--registry", required=True)
root.add_argument("--variant", choices=("a0_released_v1", "a1_released_v5", "b_root_stage_a_v5"), required=True)
root.add_argument("--trials", type=int, required=True)
root.add_argument("--output-root", required=True)
root.add_argument("--scene-id", default="grail-curb-low")
root.add_argument("--route-id", default="curb-forward")
```

Reject `trials <= 0`; require exactly three trials when variant B runs on `grail-curb-low`.

- [ ] **Step 5: Run fake-process tests and affected suites**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_sonic_*.py' -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add sonic/python/mm_sonic/root_experiment.py sonic/python/mm_sonic/cli.py tests/python/test_sonic_root_experiment.py
git commit -m "feat: run paired SONIC root experiments"
```

### Task 5: Run A0 and A1 as the Causal Negative Control

**Files:**
- Modify: `docs/superpowers/results/2026-07-22-g1-sonic-root-terrain-evaluation.md`

**Interfaces:**
- Produces one A0 and one A1 low-curb trial under identical identities except protocol/root plumbing.

- [ ] **Step 1: Run the released v1 baseline**

Run: `PYTHONPATH=sonic/python python -m mm_sonic.cli root-experiment --registry sonic/configs/experiments/root_conditioning.json --variant a0_released_v1 --trials 1 --output-root /home/ubuntu/mm-sonic-root-eval-a0`

Expected: evidence validates; outcome may pass or fail and is recorded without reinterpretation.

- [ ] **Step 2: Run released SONIC over root-capable v5**

Run: `PYTHONPATH=sonic/python python -m mm_sonic.cli root-experiment --registry sonic/configs/experiments/root_conditioning.json --variant a1_released_v5 --trials 1 --output-root /home/ubuntu/mm-sonic-root-eval-a1`

Expected: evidence validates, v5 body positions are nonzero and bit-equal to Motion Matching physical pelvis, but the released encoder input dimension/config remain exact.

- [ ] **Step 3: Verify the negative-control model response**

Run the released ONNX root perturbation probe from the training plan against A1's archived encoder input. Expected: maximum token delta is exactly `0.0` because `[1762,1792)` does not exist in the released input and existing root-z superset slots are excluded from G1 mode.

- [ ] **Step 4: Record paired evidence**

The result document records both run paths/hashes, root stream bit parity, released model/config identity equality, outcomes, root metrics, and diagnostic timing. It states only that root plumbing alone is or is not behaviorally sufficient; it does not treat a single terrain result as the learned-model verdict.

- [ ] **Step 5: Commit evidence**

```bash
git add docs/superpowers/results/2026-07-22-g1-sonic-root-terrain-evaluation.md
git commit -m "test: record SONIC root negative control"
```

### Task 6: Run Flat Regression and the Three-Trial B Gate

**Files:**
- Modify: `docs/superpowers/results/2026-07-22-g1-sonic-root-terrain-evaluation.md`

**Interfaces:**
- Consumes: Stage-A selected encoder/config and released decoder registered for B.
- Produces: one flat verdict, three consecutive low-curb verdicts, aggregate primary verdict, and `stage_b_required` decision.

- [ ] **Step 1: Run B on the existing deterministic flat gate**

Run: `PYTHONPATH=sonic/python python -m mm_sonic.cli root-experiment --registry sonic/configs/experiments/root_conditioning.json --variant b_root_stage_a_v5 --trials 1 --scene-id sonic-flat-baseline --route-id flat-12s --output-root /home/ubuntu/mm-sonic-root-eval-b-flat`

Expected: exact coverage and safety pass, and joint/pelvis tracking ratios remain within existing known-good thresholds. If flat fails, stop terrain execution and emit `stage_b_required: true`.

- [ ] **Step 2: Run three fresh consecutive low-curb trials**

Run: `PYTHONPATH=sonic/python python -m mm_sonic.cli root-experiment --registry sonic/configs/experiments/root_conditioning.json --variant b_root_stage_a_v5 --trials 3 --output-root /home/ubuntu/mm-sonic-root-eval-b-curb-low`

Expected for primary success: all three evidence bundles validate and all three terrain verdicts pass every registered gate.

- [ ] **Step 3: Compute the aggregate decision**

```json
{
  "flat_pass": true,
  "low_curb_successes": 3,
  "low_curb_trials": 3,
  "primary_pass": true,
  "stage_b_required": false
}
```

If observed values differ, record them exactly; `primary_pass` is true only for the object above.

- [ ] **Step 4: Run the default-curb stress diagnostic only after the primary decision**

Run: `PYTHONPATH=sonic/python python -m mm_sonic.cli root-experiment --registry sonic/configs/experiments/root_conditioning.json --variant b_root_stage_a_v5 --trials 1 --scene-id grail-curb-default --route-id curb-forward --output-root /home/ubuntu/mm-sonic-root-eval-b-curb-default`

Expected: valid separate evidence. Its pass/fail field is `stress_pass` and does not alter `primary_pass`.

- [ ] **Step 5: Complete and commit the result record**

Include exact commands, hashes, per-trial gates, root rise ratios, failure times if any, forbidden contacts, final target distances, and timing distributions. State one of two conclusions exactly:

- `Stage A supported: flat passed and 3/3 low-curb trials passed; proceed to depth-conditioned kinematics experiments.`
- `Stage A insufficient: flat or 3/3 low-curb gate failed; stage_b_required=true before depth experiments.`

```bash
git add docs/superpowers/results/2026-07-22-g1-sonic-root-terrain-evaluation.md
git commit -m "test: decide SONIC root terrain hypothesis"
```

### Task 7: Final Verification and Branch Handoff

**Files:**
- Test all files changed by the three root plans.

**Interfaces:**
- Produces clean Motion Matching and GEAR implementation branches with exact verification logs; does not merge into the terrain-aware branch.

- [ ] **Step 1: Run the full Motion Matching Python suite**

Run: `PYTHONPATH=sonic/python python -m unittest discover -s tests/python -p 'test_*.py' -v`

Expected: all tests pass.

- [ ] **Step 2: Run GEAR Python tests**

Run: `python -m pytest gear_sonic/tests -q`

Expected: all tests pass.

- [ ] **Step 3: Run GEAR C++ unit tests**

Run: `cmake --build gear_sonic_deploy/build --target run_tests g1_deploy_onnx_ref -j2`

Run: `gear_sonic_deploy/target/release/run_tests`

Expected: build succeeds and all unit tests pass.

- [ ] **Step 4: Verify worktree cleanliness and identities**

Run in each implementation worktree: `git status --short && git rev-parse HEAD && git log --oneline --decorate -10`

Expected: no uncommitted files, each branch descends from its pinned base, and commits are scoped by the task boundaries above.

- [ ] **Step 5: Request code review before any merge**

Use the `requesting-code-review` skill against the complete diff and both result records. Address findings through the `receiving-code-review` skill, rerun the affected gates, and retain the worktrees for user inspection. Do not merge or delete either branch without explicit user direction.
