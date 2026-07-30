# G1 Full-Pose Transition Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU full-joint transition-continuity cost that materially reduces stair-descent jerk while preserving ascent and terrain safety.

**Architecture:** A focused `torch_motion_continuity` module owns immutable row-aligned source joint states, database-derived active scales, and dimension-invariant position/velocity costs. Exact search accepts one optional per-row transition-cost vector and exempts the incumbent; the matcher computes that vector from its current emitted state and exposes component diagnostics. A deterministic 280-frame ascent, 100-frame reversal, and 260-frame descent benchmark selects weights before the existing rollout, MuJoCo-FK, and viewer gates.

**Tech Stack:** Python 3.10, PyTorch float32 CPU/CUDA tensors, NumPy, standard-library `unittest`, existing native Takara motion loader, existing terrain feature/validator stack, MuJoCo kinematic forward pass.

## Global Constraints

- Keep the experiment kinematic and privileged-height; do not add SONIC, physics integration, depth inference, terrain IK, or motion clips.
- Keep `dt = 0.02 s`, terrain feature weight `4.0`, inertialization half-life `0.10 s`, settle duration `0.20 s`, and settle magnitude `18.75` fixed during the continuity sweep.
- Apply continuity cost only to non-incumbent transitions. Incumbent continuation receives zero base, settle, and continuity transition cost.
- Both new weights default to zero and zero/omitted continuity must preserve existing selection exactly.
- Ranked terrain rescue uses continuity order but omits settle cost; it never bypasses or weakens the emitted-window terrain validator.
- Reject invalid cost tensors and all failed preparations transactionally.
- Acceptance requires all 640 directional frames, final root height at most `0.82 m`, clearance at least `-0.03 m`, descent transitions at most 11, descent p95 joint jerk at most `16,614.05 rad/s^3`, descent transition-neighborhood maximum jerk at most `58,035.6 rad/s^3`, ascent p95 joint jerk at most `14,427.81 rad/s^3`, all five existing dense terrain gates, MuJoCo-FK clearance at least `-0.03 m`, and the complete Torch suite.

---

### Task 1: Row-Aligned Full-Pose Continuity Cost

**Files:**
- Create: `sonic/python/mm_sonic/torch_motion_continuity.py`
- Create: `tests/python/test_sonic_torch_motion_continuity.py`

**Interfaces:**
- Consumes: `MotionFolder`, `MotionClip.valid_frame_stop`, database device, current emitted shape-`(29,)` joint position and velocity.
- Produces: `TransitionContinuityDatabase.from_folder(folder, device) -> TransitionContinuityDatabase`.
- Produces: `TransitionContinuityDatabase.costs(current_joint_position, current_joint_velocity, *, position_weight, velocity_weight) -> TransitionContinuityCosts`.
- Produces: `TransitionContinuityCosts.position`, `.velocity`, and `.total`, each float32 shape-`(row_count,)` on the database device.

- [ ] **Step 1: Write failing provenance and independent-oracle tests**

Create `tests/python/test_sonic_torch_motion_continuity.py` with a two-clip
fixture whose searchable row order is known. Assert row-aligned tensors
returned by `source_states_copy()` equal
`clip.joint_position[:clip.valid_frame_stop]` and velocity in relative-path
clip order. Use a small explicit tensor oracle:

```python
expected_position = 0.25 * torch.mean(
    torch.square(
        (source_position[:, active_position] - current_position[active_position])
        / position_scale[active_position]
    ),
    dim=1,
)
expected_velocity = 0.50 * torch.mean(
    torch.square(
        (source_velocity[:, active_velocity] - current_velocity[active_velocity])
        / velocity_scale[active_velocity]
    ),
    dim=1,
)
```

Assert component and total costs match the oracle on CPU and, when available,
CUDA. Make one joint component constant across every row and assert it is
excluded from the active position mask.

- [ ] **Step 2: Write failing validation and zero-weight tests**

Assert exact `ContractError` failures for:

```python
database.costs(torch.zeros(28), torch.zeros(29), position_weight=1, velocity_weight=1)
database.costs(torch.zeros(29, dtype=torch.float64), torch.zeros(29), position_weight=1, velocity_weight=1)
database.costs(torch.full((29,), float("nan")), torch.zeros(29), position_weight=1, velocity_weight=1)
database.costs(torch.zeros(29), torch.zeros(29), position_weight=-1, velocity_weight=1)
database.costs(torch.zeros(29), torch.zeros(29), position_weight=1, velocity_weight=float("inf"))
```

Assert zero/zero returns three exact all-zero float32 row vectors. Add fixtures
where all position or all velocity components are constant; enabling the
corresponding group must fail, while leaving its weight zero must succeed.

- [ ] **Step 3: Run Task 1 tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_continuity -v
```

Expected: import failure because `mm_sonic.torch_motion_continuity` does not
exist.

- [ ] **Step 4: Implement the immutable continuity database**

Create the module with these public types and validation:

```python
@dataclass(frozen=True)
class TransitionContinuityCosts:
    position: torch.Tensor
    velocity: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class TransitionContinuityDatabase:
    device: torch.device
    _joint_position: torch.Tensor
    _joint_velocity: torch.Tensor
    _position_scale: torch.Tensor
    _velocity_scale: torch.Tensor
    _position_active: torch.Tensor
    _velocity_active: torch.Tensor

    @classmethod
    def from_folder(
        cls, folder: MotionFolder, device: str | torch.device
    ) -> "TransitionContinuityDatabase":
        resolved = resolve_torch_device(device)
        position = torch.cat(
            tuple(
                torch.tensor(
                    clip.joint_position[: clip.valid_frame_stop],
                    dtype=torch.float32,
                    device=resolved,
                )
                for clip in folder.clips
            ),
            dim=0,
        )
        velocity = torch.cat(
            tuple(
                torch.tensor(
                    clip.joint_velocity[: clip.valid_frame_stop],
                    dtype=torch.float32,
                    device=resolved,
                )
                for clip in folder.clips
            ),
            dim=0,
        )
        position_scale = torch.std(position, dim=0, correction=0)
        velocity_scale = torch.std(velocity, dim=0, correction=0)
        position_active = torch.isfinite(position_scale) & (position_scale > 0)
        velocity_active = torch.isfinite(velocity_scale) & (velocity_scale > 0)
        return cls(
            resolved,
            position,
            velocity,
            position_scale,
            velocity_scale,
            position_active,
            velocity_active,
        )

    def source_states_copy(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._joint_position.clone(), self._joint_velocity.clone()

    def scales_copy(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._position_scale.clone(),
            self._velocity_scale.clone(),
            self._position_active.clone(),
            self._velocity_active.clone(),
        )
```

Implement `costs` with strict shape/dtype/device/finiteness checks. Validate
weights with `math.isfinite` and non-negativity. For each enabled group, require
at least one active component and compute the weighted mean squared normalized
residual. Disabled groups return `torch.zeros(row_count, ...)` without dividing
by their scales. Return owned tensors; do not expose mutation helpers.

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run the Task 1 command again.

Expected: all continuity tests pass on CPU and CUDA.

- [ ] **Step 6: Commit Task 1**

```bash
git add sonic/python/mm_sonic/torch_motion_continuity.py \
  tests/python/test_sonic_torch_motion_continuity.py
git commit -m "feat: compute full-pose transition continuity"
```

### Task 2: Exact Search Per-Row Transition Costs

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_motion_search.py`

**Interfaces:**
- Consumes: optional `additional_transition_costs: torch.Tensor | None`.
- Produces: `select_exact_candidate(..., additional_transition_costs=None)`.
- Produces: `rank_exact_transition_candidates(..., additional_transition_costs=None)`.
- Produces: `SearchDecision.selected_transition_cost: float`, defaulting to zero for compatibility with existing positional fixtures.

- [ ] **Step 1: Write failing selector ordering and incumbent-exemption tests**

Add synthetic three-row cases to `test_sonic_torch_motion_search.py`:

```python
row_cost = torch.tensor([100.0, 0.0, 2.0], dtype=torch.float32)
decision = select_exact_candidate(
    database,
    query,
    current_clip_index=0,
    current_frame_index=0,
    incumbent_row=0,
    search=True,
    config=MatcherConfig(),
    additional_transition_costs=row_cost,
)
```

Prove row 0 remains charged only its feature cost even though its supplied row
cost is 100. Prove a smoother feature-second-best row can beat the
feature-best transition. Assert:

```python
decision.selected_total_cost == (
    decision.selected_feature_cost
    + config.transition_penalty
    + decision.selected_transition_cost
)
```

Add a ranked-rescue case proving stable ordering includes the supplied row
cost and retains the smallest-global-row tie break.

- [ ] **Step 2: Write failing row-cost validation and zero equivalence tests**

For both selector functions, reject:

- Python lists;
- shape `(row_count, 1)`;
- wrong row count;
- float64;
- a different CUDA/CPU device;
- negative values; and
- NaN or infinity.

Run selection with `None` and with an exact zero vector and compare the complete
`SearchDecision` values. Run the existing NumPy oracle without the new argument
to prove compatibility.

- [ ] **Step 3: Run search tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_search -v
```

Expected: unexpected keyword argument
`additional_transition_costs`.

- [ ] **Step 4: Implement strict row-cost validation and exact addition**

Add:

```python
def _validated_transition_costs(
    features: torch.Tensor,
    additional_transition_costs: torch.Tensor | None,
) -> torch.Tensor:
    if additional_transition_costs is None:
        return torch.zeros(features.shape[0], dtype=torch.float32, device=features.device)
    costs = additional_transition_costs
    if (
        not isinstance(costs, torch.Tensor)
        or tuple(costs.shape) != (features.shape[0],)
        or costs.dtype != torch.float32
        or costs.device != features.device
        or not bool(torch.isfinite(costs).all().item())
        or bool((costs < 0).any().item())
    ):
        raise ContractError(
            "additional_transition_costs must be finite non-negative float32 "
            "with one row on the database device"
        )
    return costs
```

Extend both public signatures. In ordinary search compute:

```python
total_costs = (
    feature_costs
    + config.transition_penalty
    + additional_penalty
    + transition_costs
)
if incumbent_row is not None:
    total_costs[incumbent_row] = feature_costs[incumbent_row]
```

In ranking omit only the settle scalar:

```python
eligible_total_costs = (
    eligible_feature_costs
    + config.transition_penalty
    + transition_costs[eligible_rows]
)
```

Append `selected_transition_cost: float = 0.0` to `SearchDecision`. Transfer it
in the existing single selector synchronization and the existing single ranked
triple synchronization, extending ranked triples to quadruples.

- [ ] **Step 5: Run search tests and verify GREEN**

Run the Task 2 test command.

Expected: all search tests pass on CPU and CUDA.

- [ ] **Step 6: Run matcher compatibility tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_matcher -v
```

Expected: all existing positional `SearchDecision` fixtures remain valid and
all matcher tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_search.py
git commit -m "feat: price per-row transition continuity"
```

### Task 3: Matcher Runtime Wiring and Component Diagnostics

**Files:**
- Modify: `sonic/python/mm_sonic/torch_motion_matcher.py`
- Modify: `tests/python/test_sonic_torch_motion_matcher.py`
- Consume: `sonic/python/mm_sonic/torch_motion_continuity.py`

**Interfaces:**
- Consumes: `TransitionContinuityDatabase` and `TransitionContinuityCosts`.
- Produces: `MatcherConfig.transition_joint_position_weight: float = 0.0`.
- Produces: `MatcherConfig.transition_joint_velocity_weight: float = 0.0`.
- Produces diagnostics:
  - `selected_transition_position_cost: float`;
  - `selected_transition_velocity_cost: float`;
  - `selected_transition_continuity_cost: float`.

- [ ] **Step 1: Write failing zero-weight runtime equivalence test**

Extend the existing 100-command flat equivalence test. Construct one matcher
with the implicit defaults and one with both weights explicitly `0.0`. For
every command compare selected path/frame, every diagnostic except timing, and
all emitted dense tensors bitwise. Assert all three new diagnostics are zero.

- [ ] **Step 2: Write failing smoother-candidate runtime test**

Use the synthetic Takara fixture and patch
`TransitionContinuityDatabase.costs` to return:

```python
TransitionContinuityCosts(
    position=torch.tensor([0.0, 5.0, 0.1]),
    velocity=torch.tensor([0.0, 5.0, 0.1]),
    total=torch.tensor([0.0, 10.0, 0.2]),
)
```

Make the ordinary feature search prefer row 1 without continuity and row 2
with continuity. Assert the enabled matcher chooses row 2 and reports
position `0.1`, velocity `0.1`, total `0.2`. Assert an incumbent selection
reports all three as zero even if its input vector element is nonzero.

- [ ] **Step 3: Write failing ranked-rescue continuity test**

Script an unsafe incumbent and two non-incumbent candidates whose raw feature
order is opposite the supplied continuity order. Assert
`rank_exact_transition_candidates` receives the exact total row vector,
validates in continuity-adjusted order, skips an unsafe first candidate, and
commits the first safe one. Assert failed all-unsafe rescue still leaves
`matcher._state.sequence` unchanged.

- [ ] **Step 4: Run matcher tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_matcher -v
```

Expected: `MatcherConfig` rejects the new fields or the new diagnostics are
missing.

- [ ] **Step 5: Wire immutable continuity data into the matcher**

In `TorchMotionMatcher.__init__`, accept and retain one
`TransitionContinuityDatabase`. Verify its device and row count match
`TorchMotionDatabase`. In `from_folder`, construct it once from the already
loaded `MotionFolder`.

Before ordinary selection in `prepare_step`, compute:

```python
continuity = self._continuity.costs(
    state.joint_position,
    state.joint_velocity,
    position_weight=self.config.transition_joint_position_weight,
    velocity_weight=self.config.transition_joint_velocity_weight,
)
```

Pass `continuity.total` to ordinary selection and ranked rescue. Do not pass it
as or merge it with `additional_transition_penalty`; keep the settle scalar and
row vector distinct.

- [ ] **Step 6: Wire final-selection component diagnostics**

Add the three fields to `MotionMatchDiagnostics`. In `_make_result`, use the
final `decision.selected_row`:

```python
if decision.transitioned:
    position_cost, velocity_cost, total_cost = torch.stack(
        (
            continuity.position[decision.selected_row],
            continuity.velocity[decision.selected_row],
            continuity.total[decision.selected_row],
        )
    ).to(torch.float64).cpu().tolist()
else:
    position_cost = velocity_cost = total_cost = 0.0
```

Reset reports zero. Rejected transitions that retain the incumbent report
zero. Rescues report the accepted rescue row's costs. Assert the selected total
search cost already includes `selected_transition_continuity_cost`; do not add
it a second time.

- [ ] **Step 7: Run matcher tests and verify GREEN**

Run the Task 3 command.

Expected: all matcher tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add sonic/python/mm_sonic/torch_motion_matcher.py \
  tests/python/test_sonic_torch_motion_matcher.py
git commit -m "feat: use full-pose transition continuity"
```

### Task 4: Configuration and Backward-Compatible Rollout Evidence

**Files:**
- Modify: `sonic/configs/experiments/torch_stair_small.json`
- Modify: `sonic/python/mm_sonic/torch_terrain_rollout.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_viewer.py`
- Modify: `tests/python/test_sonic_torch_terrain_rollout.py`
- Modify: `tests/python/test_sonic_torch_terrain_viewer.py`

**Interfaces:**
- Consumes: the two new matcher configuration fields and three diagnostics.
- Produces rollout arrays:
  - `selected_transition_position_cost`;
  - `selected_transition_velocity_cost`;
  - `selected_transition_continuity_cost`.
- Historical authenticated rollouts without these fields load with float32
  all-zero arrays.

- [ ] **Step 1: Write failing config validation/reconstruction tests**

Update the exact config-field test to require both new names. Assert
`matcher_config_from_resolved` reconstructs their float values. Add negative,
NaN, infinity, boolean, missing-key, and extra-key cases. Both fields are
finite and non-negative, not strictly positive.

- [ ] **Step 2: Write failing artifact and legacy-loader tests**

Require all three arrays at shape `(frames,)`, float32, finite, and
non-negative. Require:

```python
np.testing.assert_allclose(
    arrays["selected_transition_position_cost"]
    + arrays["selected_transition_velocity_cost"],
    arrays["selected_transition_continuity_cost"],
    rtol=0,
    atol=2e-5,
)
```

Remove the three arrays from an authenticated saved archive, update its archive
hash, and assert `load_saved_rollout` synthesizes exact float32 zeros.

- [ ] **Step 3: Run rollout/viewer tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_viewer -v
```

Expected: exact matcher-field validation fails until production/config are
updated.

- [ ] **Step 4: Add strict configuration fields**

Add both names to `_MATCHER_FLOAT_FIELDS` and
`_MATCHER_NONNEGATIVE_FLOAT_FIELDS`, reconstruct them in
`matcher_config_from_resolved`, and initially set both JSON values to `0.0`:

```json
"transition_joint_position_weight": 0.0,
"transition_joint_velocity_weight": 0.0
```

The qualified nonzero values are written only after Task 6 selects them.

- [ ] **Step 5: Save and load component evidence**

Append each diagnostic as `np.float32` to rollout rows and include it in
transition events. In the viewer loader, synthesize missing fields after
archive authentication and add all three required shapes. Reject non-floating,
non-finite, or negative arrays.

- [ ] **Step 6: Run rollout/viewer tests and verify GREEN**

Run the Task 4 command.

Expected: all rollout and viewer tests pass, including historical archives.

- [ ] **Step 7: Commit Task 4**

```bash
git add sonic/configs/experiments/torch_stair_small.json \
  sonic/python/mm_sonic/torch_terrain_rollout.py \
  sonic/python/mm_sonic/torch_terrain_viewer.py \
  tests/python/test_sonic_torch_terrain_rollout.py \
  tests/python/test_sonic_torch_terrain_viewer.py
git commit -m "feat: record transition continuity evidence"
```

### Task 5: Deterministic Up-Turn-Down Benchmark

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_directional_rollout.py`
- Create: `tests/python/test_sonic_torch_terrain_directional_rollout.py`

**Interfaces:**
- Consumes: existing experiment resolver, dense terrain extension/validator,
  matcher configuration, and optional weight overrides.
- Produces: `run_directional_rollout(resolved, *, device, position_weight, velocity_weight) -> DirectionalRollout`.
- Produces: CLI module `mm_sonic.torch_terrain_directional_rollout`.
- Produces: saved `rollout.npz`, `metrics.json`, `events.jsonl`, and
  `resolved_config.json`.

- [ ] **Step 1: Write failing phase and metric-oracle tests**

Create pure tests for exact phase boundaries:

```python
self.assertEqual(directional_phase(0), "ascent")
self.assertEqual(directional_phase(279), "ascent")
self.assertEqual(directional_phase(280), "reversal")
self.assertEqual(directional_phase(379), "reversal")
self.assertEqual(directional_phase(380), "descent")
self.assertEqual(directional_phase(639), "descent")
```

Reject indices outside `[0, 639]`. Use small synthetic joint/root/clearance and
transition arrays with a hand-computed second and third finite difference.
Assert acceleration, jerk, transition-neighborhood `±4` frame membership,
accepted transition count, cross-clip count, minimum clearance, and final root
height exactly match the oracle.

- [ ] **Step 2: Write failing CLI/artifact contract tests**

Patch `run_directional_rollout` with a fixture result. Parse:

```bash
--dataset DATASET --config CONFIG --device cpu \
--position-weight 0.1 --velocity-weight 0.25 --output OUTPUT
```

Assert the CLI rejects negative/non-finite weights, saves pickle-free arrays,
canonical finite JSON, records the exact overrides, and refuses to silently
replace a non-directory output.

- [ ] **Step 3: Run directional tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_terrain_directional_rollout -v
```

Expected: import failure because the directional module does not exist.

- [ ] **Step 4: Implement the exact 640-frame command sequence**

Define:

```python
ASCENT_STOP = 280
REVERSAL_STOP = 380
STEP_COUNT = 640
TRANSITION_NEIGHBORHOOD_RADIUS = 4
```

Build a dense matcher with the existing terrain validator. For steps
`0..279`, command `+reference_direction * command_speed`; for `280..639`,
command the negative vector. Set command heading to `atan2(v_y, v_x)` every
frame. The matcher's existing bounded command shaper owns the turn; the
benchmark does not insert a hidden stop or teleport.

Save at least:

- joint position and velocity;
- root position;
- both emitted foot positions and clearance;
- selected clip index and frame;
- transitioned, rejected, and rescue diagnostics;
- continuity component costs; and
- motion/terrain/selected total costs.

Use the existing authenticated query grid and alignment for clearance. Do not
call `mj_step`, SONIC, or the renderer.

- [ ] **Step 5: Implement phase-separated metrics and gates**

For each phase, compute vector-norm joint acceleration and jerk from emitted
20 ms states, vector-norm root jerk, accepted/rejected transitions, cross-clip
transitions, contact-foot speed proxy, minimum clearance, and continuity-cost
percentiles. Mark jerk samples within four output frames of any accepted
transition as the transition neighborhood.

Return top-level gates:

```python
{
    "completed_640_frames": frame_count == 640,
    "returned_to_lower_height": final_root_height <= 0.82,
    "minimum_clearance": minimum_clearance >= -0.03,
    "descent_transition_count": descent_transitions <= 11,
    "descent_joint_jerk_p95": descent_p95 <= 16614.05,
    "descent_transition_max_jerk": descent_transition_max <= 58035.6,
    "ascent_joint_jerk_p95": ascent_p95 <= 14427.81,
}
```

The zero/zero baseline is allowed to fail the three improvement gates but must
still be saved. A nonzero candidate is qualified only when every gate is true.

- [ ] **Step 6: Implement transactional artifact save and CLI**

Follow `save_stair_rollout`: stage in a sibling temporary directory, write
pickle-free NPZ plus canonical JSON/JSONL, hash the archive, and atomically
replace an existing real output directory through a `.previous` backup.
Include a deterministic hash that excludes timing fields.

- [ ] **Step 7: Run directional tests and verify GREEN**

Run the Task 5 command.

Expected: all directional tests pass.

- [ ] **Step 8: Run focused matcher/terrain regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_motion_continuity \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_terrain_rollout \
  tests.python.test_sonic_torch_terrain_viewer \
  tests.python.test_sonic_torch_terrain_directional_rollout -v
```

Expected: zero failures.

- [ ] **Step 9: Commit Task 5**

```bash
git add sonic/python/mm_sonic/torch_terrain_directional_rollout.py \
  tests/python/test_sonic_torch_terrain_directional_rollout.py
git commit -m "feat: benchmark directional stair transitions"
```

### Task 6: Weight Sweep, Qualification, and Retained Evidence

**Files:**
- Modify: `sonic/configs/experiments/torch_stair_small.json`
- Modify: `docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md`
- Modify: `docs/superpowers/plans/2026-07-30-g1-full-pose-transition-continuity.md`

**Interfaces:**
- Consumes: the 25 fixed `(position_weight, velocity_weight)` cells.
- Produces: retained config weights, directional evidence, full terrain
  qualification, MuJoCo-FK result, and live viewer verdict.

- [ ] **Step 1: Run and save the zero-weight baseline**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_directional_rollout \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --device cuda:3 --position-weight 0 --velocity-weight 0 \
  --output build/torch-stair-small-results/quality-loop-07/directional/p0-v0
```

Expected reproduction: 640 frames, final root height at most `0.82 m`, no
penetration, 11 descent transitions, descent joint-jerk p95 approximately
`20,767.56`, and descent transition-neighborhood maximum jerk approximately
`96,726.0`. Diagnose rather than continue if the deterministic baseline differs
materially.

- [ ] **Step 2: Run the fixed 25-cell sweep**

For each Cartesian pair from:

```text
position = 0, 0.05, 0.10, 0.25, 0.50
velocity = 0, 0.05, 0.10, 0.25, 0.50
```

run the same CLI and save it under:

```text
build/torch-stair-small-results/quality-loop-07/directional/p<position>-v<velocity>
```

Use available `cuda:0` through `cuda:7` in bounded batches, with at most one
process per GPU. Capture every process exit status. Do not overwrite or infer a
missing cell.

- [ ] **Step 3: Select one retained cell by the frozen ordering**

Reject any candidate with a false gate. Sort the remaining cells by:

1. descent transition-neighborhood maximum joint jerk;
2. descent p95 joint jerk;
3. ascent p95 joint jerk;
4. `position_weight + velocity_weight`; and
5. position weight, then velocity weight.

Write the selected values explicitly into
`sonic/configs/experiments/torch_stair_small.json`. Re-run the selected cell
using only config values and require a byte-identical deterministic hash.

- [ ] **Step 4: Run the complete Torch suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B -m unittest discover \
  -s tests/python -p 'test_sonic_torch*.py' -v
```

Expected: zero failures; the existing protected real-data oracle may remain
skipped by its explicit opt-in gate.

- [ ] **Step 5: Run flat/legacy/dense CUDA qualification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_rollout \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --conditions flat legacy dense --device cuda:2 \
  --output build/torch-stair-small-results/quality-loop-07/full
```

Expected: dense passes all five frozen acceptance criteria and reports no
clearance below `-0.03 m`.

- [ ] **Step 6: Run authoritative MuJoCo-FK clearance**

Replay all 450 retained dense `joint_position`, `root_position_world`, and
`root_orientation_world_wxyz` rows through:

```text
/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
```

Map target joints through `PINNED_TARGET_TO_SOURCE_PERMUTATION`, call
`mujoco.mj_forward` only, sample both ankle-roll body XY positions against the
authenticated query grid, and report the minimum ankle-origin clearance.

Expected: minimum at least `-0.03 m`, zero penetrating samples.

- [ ] **Step 7: Request code review and address findings**

Use the requesting-code-review workflow on all commits from Task 1 through
Task 5. Review specifically:

- incumbent exemption;
- zero-weight equivalence;
- row-provenance alignment;
- rescue ordering and transactional failure;
- artifact backward compatibility; and
- benchmark phase/derivative indexing.

Apply accepted corrections test-first and rerun Steps 4 through 6.

- [ ] **Step 8: Launch prolonged interactive qualification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python:. \
  sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:2
```

Ask the user to traverse up, reverse on the landing, and descend repeatedly.
Do not call the candidate stable unless the process remains alive and the user
confirms descent is visibly smoother. Lateral and side-exit traversal remain
the next motion-inventory experiment and are not a hidden gate for this fix.

- [ ] **Step 9: Record honest retained evidence**

Update the results document with:

- the current Loop 06 ranked-rescue hashes and zero scripted rescue ranks;
- correction of the stale earlier live-stability claim;
- baseline directional metrics;
- all 25 sweep cells and rejection reasons;
- retained weights and deterministic identity;
- separate ascent/reversal/descent metrics;
- full five-gate rollout and MuJoCo-FK results;
- full test count;
- live verdict; and
- any remaining descent weakness.

Mark every completed checkbox in this plan. Do not hide a weaker descent result
inside an aggregate metric.

- [ ] **Step 10: Commit configuration and evidence**

```bash
git add sonic/configs/experiments/torch_stair_small.json \
  docs/superpowers/results/2026-07-29-g1-torch-stair-small-results.md \
  docs/superpowers/plans/2026-07-30-g1-full-pose-transition-continuity.md
git commit -m "docs: qualify full-pose transition continuity"
```
