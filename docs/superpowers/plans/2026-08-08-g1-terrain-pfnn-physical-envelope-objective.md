# G1 Terrain PFNN Physical-Envelope Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a train-only predecessor-aware physical-envelope objective that eliminates rare fitted joint-step, native-limit, and phase violations without changing PFNN inputs/outputs, runtime behavior, or authorization thresholds.

**Architecture:** Keep the existing dataset v2 and `288 -> 512 -> 512 -> 268` model. Derive an exact fitted predecessor-pair population, compute per-pair worst-joint risks with a linear lower-phase hinge, and reduce each family as lowest-ordinal global maximum plus the mean of strictly positive risks in the deterministic worst-10% tail while separately reporting standard CVaR. One-step training uses sealed predecessor targets; rollout step 0 uses its sealed predecessor and later steps use prior predictions without detaching; fixed-size DDP metadata validation precedes every equal-size value gather.

**Tech Stack:** Python 3.10, PyTorch, NumPy, `unittest`, CPU `gloo` for DDP regression tests, CUDA GPU 1 for one controlled artifact run.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-08-g1-terrain-pfnn-physical-envelope-objective-design.md`.
- Raw PFNN input/output layouts remain exactly `288 -> 512 -> 512 -> 268`.
- Runtime, recurrence, command semantics, controller, and raw-output interpretation remain unchanged.
- Runtime limits remain exactly: root translation `<= 0.060 m`, root rotation `<= 0.35 rad`, joint step `<= 0.25 rad`, native joint limits, positive root height, normalized finite root quaternion, trajectory-direction norm `[0.5,1.5]`, and phase `[0,min(pi,1.5*q99)]`.
- No output clamp, masking, correction, residual reconstruction, IK, stance lock, root projection, portal, MotionBricks call, or threshold relaxation.
- Objective constants are exact: joint-step onset `0.225`, joint-step scale `0.025`, native-limit inset `0.020`, upper-phase inset `0.020`, phase scale `0.020`, tail fraction `0.10`, squared joint-step/native/upper-phase hinges, linear lower-phase hinge, maximum coefficient `1.0`, tail-mass coefficient `1.0`, and weight `1.0` for each complete max-plus-tail family loss.
- Dataset schema remains `mm-sonic-terrain-pfnn-dataset/v2`; checkpoint advances to v6, one-step report to v2, and pipeline promotion receipt to v3.
- All objective data are train-only. Validation selection remains validation-only. The sealed test split stays closed until original Task 9.
- Resume validates active fitted rows, pair receipt, objective, target-feasibility receipt, pair sampler, sequence sampler, dataset, normalization, and kinematics before model or optimizer restore.
- The active pair population is mode-dependent but always train-only: exactly 2,011 pairs from the sealed 2,048-row fitted receipt for the controlled rerun, and the complete adjacent-pair population of the train split for later full-corpus training. The historical field name `fitted_pair_receipt` binds either active optimization population and records its row-scope digest.
- The observed `130/2011 = 6.4644%` failure rate is covered by a 10% tail only for complete-population reduction. Minibatches rely on their deterministic global maximum for nondilution and on the sampler for eventual population coverage.
- Exactly one new controlled `4,000 + 256` recurrence-v2 training run is allowed after all tests pass. Never resume or overwrite the failed Task 5 candidate.
- At the first real fitted or closed-loop failure, record exact provenance/value and stop. No second run or in-cycle hyperparameter change.
- The audit-only negative-phase tolerance is exactly `8 * np.finfo(np.float32).eps = 9.5367431640625e-7 rad`; record the unclamped minimum, reject strictly below its negative, and only then clamp audit values to zero. Predictions never use this tolerance.
- Before dataset resume, persist a six-file SHA-256/size/`mtime_ns` snapshot JSON; persist the same schema afterward and require exact entry and canonical-digest equality.
- The physical-envelope correction source/tests are frozen immediately before the sole corrected training command starts. After that boundary no correction source/test edit or commit is authorized; failures become evidence for a separately approved future cycle. A later blocker-free handoff may enter the already-approved original Task 8/9 viewer/full-corpus file scope, but it may not alter the frozen correction.

**Loss-key contract:** retain the existing base-loss names as
`BASE_LOSS_WEIGHT_KEYS`; add
`ENVELOPE_LOSS_WEIGHT_KEYS = ("joint_step_envelope", "joint_limit_envelope",
"phase_envelope")`; and define `LOSS_WEIGHT_KEYS` as their concatenation.
`pfnn_losses` computes and totals only `BASE_LOSS_WEIGHT_KEYS`.
`physical_envelope_loss` computes each family as lowest-ordinal global maximum
plus selected-positive tail mean, reports its standard CVaR separately, and the one-step and
rollout aggregators add all three family losses exactly once. The immutable
checkpoint `loss_weights` mapping contains the concatenated key set at float
`1.0`. This prevents expanding the frozen mapping from making the base-only
`pfnn_losses` path index absent envelope values.

**Test-fixture contract:** every private helper named in a test sketch below is
part of that RED step, not assumed infrastructure. `_single_parameter_output_model`
returns a module with one scalar parameter initialized to `0.30` and an
identity-normalized `[B,268]` output whose first joint equals that parameter.
`_unsafe_pair_batch` returns four zero `[288]` inputs, four zero phases,
predecessor outputs with first joint `0.0`, and current targets with first
joint `0.29`; this distinguishes predecessor-relative `0.30` from target error
`0.01`. `_single_rank_tail_reference` and
`_spawn_two_rank_gloo_tail_workers` use global ordinal risks
`{0: parameter, 1: 2*parameter, 2: 2*parameter, 3: 0*parameter}` at parameter
`1.0`, tail fraction `0.5`, rank-0 ordinals `[0,2]`, and rank-1 ordinals
`[1,3]`; lowest-ordinal maximum `2.0`, CVaR `2.0`, active count `2`, positive
tail mean `2.0`, and both combined loss and DDP-averaged parameter gradient are
exactly `4.0`.
The gloo worker must be a module-level spawn target and return both ranks'
values through a multiprocessing queue. `_scripted_three_step_model` and
`_unsafe_three_step_rollout` are extractions of the existing exact three-step
shared-recurrence fixture: use predecessor joint `0.0`, predictions `0.10`,
`0.40`, `0.40`, identity normalization, phase `0.10`, and native limits
`[-1,1]`, so step 1 alone has normalized-squared risk `9.0` and differentiates
to prediction 0. `_write_v6_checkpoint` extends the existing valid checkpoint
round-trip fixture with the exact v6 fields; `_write_v5_checkpoint` changes
only its schema; `_tamper_and_rehash_checkpoint` deep-copies one named nested
field and recomputes only the outer candidate SHA; and
`_resume_arguments_with_pair_digest` reuses the existing resume-before-restore
fixture while changing only the active pair digest. Do not substitute random
fixtures for these exact values.

---

### Task 1: Add exact physical-envelope risks and nondilutable global reduction

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Consumes: normalized `[N,268]` predictions and reached outputs, existing normalization, `[29,2]` native limits, phase cap, global ordinals, rank, and world size.
- Produces: `PhysicalEnvelopeObjective`, `physical_envelope_risks(...)`, `GlobalEnvelopeReduction`, `global_max_plus_tail(...)`, and `physical_envelope_loss(...)`.

- [ ] **Step 1: Write RED margin, reduction, and boundary-gradient tests**

Add tests with exact names:

```python
def test_physical_envelope_risks_use_exact_margins_and_phase_gradients(self):
    contract = training.PhysicalEnvelopeObjective()
    prediction = torch.zeros(
        (5, OUTPUT_LAYOUT.size), dtype=torch.float64, requires_grad=True
    )
    reached = torch.zeros_like(prediction)
    prediction.data[:, OUTPUT_LAYOUT["joint_position"]] = 0.0
    reached.data[:, OUTPUT_LAYOUT["joint_position"]] = 0.0
    prediction.data[0, OUTPUT_LAYOUT["joint_position"].start] = 0.225
    prediction.data[1, OUTPUT_LAYOUT["joint_position"].start] = 0.250
    prediction.data[2, OUTPUT_LAYOUT["phase_advance"]] = 0.0
    prediction.data[3, OUTPUT_LAYOUT["phase_advance"]] = 0.48
    prediction.data[4, OUTPUT_LAYOUT["phase_advance"]] = -1.0e-12
    normal = _normalization()
    limits = torch.tensor([[-1.0, 1.0]] * 29)
    risks = training.physical_envelope_risks(
        prediction,
        reached,
        normalization=normal,
        joint_limits=limits,
        phase_advance_cap=0.50,
        contract=contract,
    )
    self.assertEqual(float(risks["joint_step"][0]), 0.0)
    self.assertAlmostEqual(float(risks["joint_step"][1]), 1.0, places=6)
    self.assertEqual(float(risks["phase"][2]), 0.0)
    self.assertEqual(float(risks["phase"][3]), 0.0)
    self.assertAlmostEqual(float(risks["phase"][4]), 5.0e-11, places=20)
    sum(value.sum() for value in risks.values()).backward()
    self.assertEqual(float(prediction.grad[0, OUTPUT_LAYOUT["joint_position"].start]), 0.0)
    self.assertEqual(float(prediction.grad[2, OUTPUT_LAYOUT["phase_advance"].start]), 0.0)
    self.assertEqual(
        float(prediction.grad[4, OUTPUT_LAYOUT["phase_advance"].start]), -50.0
    )


def test_global_max_plus_tail_is_invariant_to_appended_safe_elements(self):
    for count in (1, 9, 10, 11, 32, 100, 1000):
        offender = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
        risk = torch.cat((offender.reshape(1), torch.zeros(count - 1)))
        reduction = training.global_max_plus_tail(
            risk,
            global_ordinals=torch.arange(count, dtype=torch.int64),
            tail_fraction=0.10,
            rank=0,
            world_size=1,
        )
        self.assertEqual(float(reduction.maximum), 9.0)
        self.assertEqual(reduction.active_count, 1)
        self.assertEqual(float(reduction.positive_tail_mean), 9.0)
        self.assertEqual(float(reduction.loss), 18.0)
        reduction.loss.backward()
        self.assertEqual(float(offender.grad), 2.0)


def test_global_max_plus_tail_ties_use_ascending_global_ordinal(self):
    risk = torch.ones(20, requires_grad=True)
    ordinals = torch.arange(20, dtype=torch.int64).flip(0)
    reduction = training.global_max_plus_tail(
        risk,
        global_ordinals=ordinals,
        tail_fraction=0.10,
        rank=0,
        world_size=1,
    )
    reduction.positive_tail_mean.backward()
    selected = set(ordinals[torch.nonzero(risk.grad, as_tuple=False).flatten()].tolist())
    self.assertEqual(selected, {0, 1})
```

Also cover shape/dtype failures, `phase_cap <= 0.020`, one bad joint among 28
safe joints, both native-limit sides, upper phase, and `tail_fraction` outside
`(0,1]`. Add one-step `[B]` and rollout `[T*B]` forms of the safe-append test;
both must keep the lone offender's family loss and gradient exactly unchanged.

Add a two-rank gloo table-driven test for these exact cases:

```text
case                    rank 0                     rank 1                 result
cross-rank max tie      risks [2,0], ord [0,2]    risks [2,0], ord [1,3] max=2, cvar=2, active_mean=2, loss=4 at f=.25
tail on rank 1 only     risks [0,0], ord [0,2]    risks [4,3], ord [1,3] max=4, cvar=3.5, active_mean=3.5, loss=7.5 at f=.50
unequal counts          risks [1,0], ord [0,2]    risks [1], ord [1]      collective ValueError before risk gather
rank mismatch           supplied rank 0           supplied rank 0         collective ValueError before risk gather
ordinal gap             risks [1,0], ord [0,3]    risks [1,0], ord [1,4] collective ValueError after equal gather
one-rank nonfinite      risks [1,0]                risks [inf,0]            collective ValueError before risk gather
dtype mismatch          float32 risks              float64 risks             identical ValueError before risk gather
device mismatch*        CPU risks                  CUDA risks                identical ValueError before risk gather
fraction mismatch       tail_fraction .10          tail_fraction .20         identical ValueError before risk gather
one invalid fraction    tail_fraction .10          tail_fraction NaN         identical ValueError before risk gather
safe rank appended      offender on rank 0        new all-zero rank(s)     loss and offender gradient unchanged
```

Expose the remaining Important cases as individually bounded REDs named
`test_two_rank_metadata_rejects_risk_dtype_mismatch_before_value_gather`,
`test_two_rank_metadata_rejects_device_type_mismatch_before_value_gather`,
`test_two_rank_metadata_rejects_fraction_bit_mismatch_before_value_gather`, and
`test_two_rank_metadata_rejects_one_invalid_fraction_before_value_gather`.
Every worker returns the same exception class/message and collective counters;
no worker is allowed to raise while constructing its sentinel metadata.

`*` Run the gloo CPU/CUDA mismatch only when CUDA is available; otherwise mark
that bounded case skipped with the explicit reason. Metadata remains on CPU for
gloo, so the malformed CUDA risk is never gathered.

The cross-rank tie test asserts max gradient goes only to lowest global ordinal
0; the all-selected-on-one-rank case asserts the DDP-averaged loss
and gradient equal a single-rank concatenated reference. Instrument the worker
to count metadata and risk/ordinal `all_gather` calls. Every rank must perform
exactly one metadata gather; assert zero risk/ordinal gathers for unequal
counts, rank mismatch, one-rank nonfinite, dtype/device mismatch, differing
valid fractions, and one invalid fraction.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_physical_envelope_risks_use_exact_margins_and_phase_gradients \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_global_max_plus_tail_is_invariant_to_appended_safe_elements \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_global_max_plus_tail_ties_use_ascending_global_ordinal \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_two_rank_metadata_rejects_risk_dtype_mismatch_before_value_gather \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_two_rank_metadata_rejects_device_type_mismatch_before_value_gather \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_two_rank_metadata_rejects_fraction_bit_mismatch_before_value_gather \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_two_rank_metadata_rejects_one_invalid_fraction_before_value_gather
```

Expected: errors naming missing `PhysicalEnvelopeObjective`,
`physical_envelope_risks`, and `global_max_plus_tail`.

- [ ] **Step 3: Implement the immutable contract and per-pair risks**

Add to `training.py`:

```python
@dataclass(frozen=True)
class PhysicalEnvelopeObjective:
    schema: str = "mm-sonic-physical-envelope-objective/v1"
    joint_step_onset_rad: float = 0.225
    joint_step_scale_rad: float = 0.025
    joint_limit_margin_rad: float = 0.020
    phase_upper_margin_rad: float = 0.020
    phase_scale_rad: float = 0.020
    tail_fraction: float = 0.10
    joint_step_hinge_power: int = 2
    joint_limit_hinge_power: int = 2
    phase_lower_hinge_power: int = 1
    phase_upper_hinge_power: int = 2
    maximum_coefficient: float = 1.0
    positive_tail_mean_coefficient: float = 1.0


DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE = PhysicalEnvelopeObjective()


def physical_envelope_risks(
    normalized_prediction: torch.Tensor,
    reached_output_normalized: torch.Tensor,
    *,
    normalization: object,
    joint_limits: torch.Tensor,
    phase_advance_cap: float,
    contract: PhysicalEnvelopeObjective,
) -> dict[str, torch.Tensor]:
    if (
        normalized_prediction.ndim != 2
        or normalized_prediction.shape[1] != OUTPUT_LAYOUT.size
        or reached_output_normalized.shape != normalized_prediction.shape
        or not normalized_prediction.is_floating_point()
        or reached_output_normalized.dtype != normalized_prediction.dtype
        or reached_output_normalized.device != normalized_prediction.device
    ):
        raise ValueError("physical envelope tensors are invalid")
    if not torch.isfinite(normalized_prediction).all() or not torch.isfinite(
        reached_output_normalized
    ).all():
        raise ValueError("physical envelope tensors must be finite")
    if contract != DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE:
        raise ValueError("physical envelope objective contract mismatch")
    if not math.isfinite(float(phase_advance_cap)) or (
        float(phase_advance_cap) <= contract.phase_upper_margin_rad
    ):
        raise ValueError("physical envelope phase cap is invalid")
    limits = torch.as_tensor(
        joint_limits,
        device=normalized_prediction.device,
        dtype=normalized_prediction.dtype,
    )
    if (
        limits.shape != (len(ISAACLAB_JOINT_NAMES), 2)
        or not torch.isfinite(limits).all()
        or torch.any(
            limits[:, 0] + contract.joint_limit_margin_rad
            > limits[:, 1] - contract.joint_limit_margin_rad
        )
    ):
        raise ValueError("physical envelope joint limits are invalid")
    normal = _normalization_tensors(
        normalization,
        device=normalized_prediction.device,
        dtype=normalized_prediction.dtype,
    )
    predicted = normalized_prediction * normal["y_std"] + normal["y_mean"]
    reached = reached_output_normalized * normal["y_std"] + normal["y_mean"]
    q_pred = predicted[:, OUTPUT_LAYOUT["joint_position"]]
    q_reached = reached[:, OUTPUT_LAYOUT["joint_position"]]
    step_excess = torch.relu(
        torch.abs(q_pred - q_reached) - contract.joint_step_onset_rad
    ) / contract.joint_step_scale_rad
    lower = torch.relu(
        limits[:, 0] + contract.joint_limit_margin_rad - q_pred
    ) / contract.joint_limit_margin_rad
    upper = torch.relu(
        q_pred - (limits[:, 1] - contract.joint_limit_margin_rad)
    ) / contract.joint_limit_margin_rad
    phase = predicted[:, OUTPUT_LAYOUT["phase_advance"]].reshape(-1)
    phase_low = torch.relu(-phase) / contract.phase_scale_rad
    phase_high = torch.relu(
        phase - (float(phase_advance_cap) - contract.phase_upper_margin_rad)
    ) / contract.phase_scale_rad
    return {
        "joint_step": torch.amax(step_excess.square(), dim=1),
        "joint_limit": torch.amax(torch.maximum(lower, upper).square(), dim=1),
        "phase": torch.maximum(phase_low, phase_high.square()),
    }
```

- [ ] **Step 4: Implement fixed-metadata DDP max-plus-tail reduction**

```python
@dataclass(frozen=True)
class GlobalEnvelopeReduction:
    maximum: torch.Tensor
    cvar: torch.Tensor
    tail_count: int
    active_count: int
    positive_tail_mean: torch.Tensor
    loss: torch.Tensor


def global_max_plus_tail(
    local_risk: torch.Tensor,
    *,
    global_ordinals: torch.Tensor,
    tail_fraction: float,
    rank: int,
    world_size: int,
) -> GlobalEnvelopeReduction:
    import struct
    import torch.distributed as dist

    initialized = dist.is_available() and dist.is_initialized()
    actual_rank = dist.get_rank() if initialized else 0
    actual_world_size = dist.get_world_size() if initialized else 1
    risk_is_tensor = isinstance(local_risk, torch.Tensor)
    ordinal_is_tensor = isinstance(global_ordinals, torch.Tensor)
    risk_dtype_code = (
        {torch.float32: 1, torch.float64: 2}.get(local_risk.dtype, -1)
        if risk_is_tensor else -1
    )
    risk_device_code = (
        {"cpu": 1, "cuda": 2}.get(local_risk.device.type, -1)
        if risk_is_tensor else -1
    )
    backend_name = (
        str(dist.get_backend())
        if initialized
        else ("nccl" if risk_device_code == 2 else "gloo")
    )
    backend_code = {"gloo": 1, "nccl": 2}.get(backend_name, -1)
    metadata_device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend_name == "nccl"
        else torch.device("cpu")
    )
    metadata_device_code = {"cpu": 1, "cuda": 2}[metadata_device.type]
    local_count = int(local_risk.numel()) if risk_is_tensor else -1
    risk_shape_valid = bool(
        risk_is_tensor and local_risk.ndim == 1 and local_count > 0
    )
    backend_device_valid = bool(
        (backend_name == "gloo" and risk_device_code == 1)
        or (backend_name == "nccl" and risk_device_code == 2)
    )
    finite = bool(
        risk_shape_valid
        and risk_dtype_code in (1, 2)
        and torch.isfinite(local_risk).all()
    )
    ordinal_shape_valid = bool(
        ordinal_is_tensor
        and risk_is_tensor
        and global_ordinals.shape == local_risk.shape
    )
    ordinal_dtype_valid = bool(
        ordinal_is_tensor and global_ordinals.dtype == torch.int64
    )
    ordinal_device_valid = bool(
        ordinal_is_tensor
        and risk_is_tensor
        and global_ordinals.device == local_risk.device
    )
    fraction_valid = bool(
        type(tail_fraction) is float
        and math.isfinite(tail_fraction)
        and 0.0 < tail_fraction <= 1.0
    )
    fraction_bits = (
        struct.unpack(">Q", struct.pack(">d", tail_fraction))[0]
        if fraction_valid else -1
    )
    metadata = torch.tensor(
        [
            1, local_count, int(risk_is_tensor), int(risk_shape_valid),
            risk_dtype_code, risk_device_code, int(backend_device_valid),
            int(finite), int(ordinal_is_tensor), int(ordinal_shape_valid),
            int(ordinal_dtype_valid), int(ordinal_device_valid),
            int(fraction_valid), fraction_bits,
            rank if type(rank) is int else -1,
            world_size if type(world_size) is int else -1,
            actual_rank, actual_world_size, backend_code, metadata_device_code,
        ],
        dtype=torch.int64,
        device=metadata_device,
    )
    gathered_metadata = [torch.empty_like(metadata) for _ in range(actual_world_size)]
    if initialized:
        dist.all_gather(gathered_metadata, metadata)
    else:
        gathered_metadata[0].copy_(metadata)
    meta = torch.stack(gathered_metadata).cpu()
    expected_ranks = torch.arange(actual_world_size, dtype=torch.int64)
    flags = meta[:, [2, 3, 6, 7, 8, 9, 10, 11, 12]]
    if (
        torch.any(meta[:, 0] != 1)
        or torch.any(meta[:, 1] <= 0)
        or torch.any(meta[:, 1] != meta[0, 1])
        or not torch.all(flags == 1)
        or meta[0, 4].item() not in (1, 2)
        or torch.any(meta[:, 4] != meta[0, 4])
        or meta[0, 5].item() not in (1, 2)
        or torch.any(meta[:, 5] != meta[0, 5])
        or torch.any(meta[:, 13] != meta[0, 13])
        or not torch.equal(meta[:, 14], expected_ranks)
        or torch.any(meta[:, 15] != actual_world_size)
        or not torch.equal(meta[:, 16], expected_ranks)
        or torch.any(meta[:, 17] != actual_world_size)
        or meta[0, 18].item() not in (1, 2)
        or torch.any(meta[:, 18] != meta[0, 18])
        or torch.any(meta[:, 19] != (1 if meta[0, 18] == 1 else 2))
    ):
        raise ValueError("global envelope metadata is invalid")

    gathered_risks = [torch.empty_like(local_risk) for _ in range(actual_world_size)]
    gathered_ordinals = [torch.empty_like(global_ordinals) for _ in range(actual_world_size)]
    if initialized:
        dist.all_gather(gathered_risks, local_risk.detach())
        dist.all_gather(gathered_ordinals, global_ordinals)
    else:
        gathered_risks[0].copy_(local_risk.detach())
        gathered_ordinals[0].copy_(global_ordinals)
    global_risk = torch.cat(gathered_risks)
    global_ordinal = torch.cat(gathered_ordinals)
    total_count = int(global_risk.numel())
    if not torch.equal(
        torch.sort(global_ordinal).values,
        torch.arange(total_count, dtype=torch.int64, device=global_ordinal.device),
    ):
        raise ValueError("global envelope ordinals must be exactly 0..N-1")

    risk_values = global_risk.cpu().tolist()
    ordinal_values = global_ordinal.cpu().tolist()
    order = sorted(
        range(total_count),
        key=lambda index: (-float(risk_values[index]), int(ordinal_values[index])),
    )
    tail_count = max(1, math.ceil(tail_fraction * total_count))
    tail_ordinals = {int(ordinal_values[index]) for index in order[:tail_count]}
    active_indices = [
        index for index in order[:tail_count] if float(risk_values[index]) > 0.0
    ]
    active_ordinals = {int(ordinal_values[index]) for index in active_indices}
    active_count = len(active_indices)
    maximum_value = float(risk_values[order[0]])
    maximum_ordinal = int(ordinal_values[order[0]])
    local_selected = local_risk[
        torch.isin(global_ordinals, torch.tensor(tuple(tail_ordinals), device=local_risk.device))
    ].sum()
    if active_count:
        local_active = local_risk[
            torch.isin(
                global_ordinals,
                torch.tensor(tuple(active_ordinals), device=local_risk.device),
            )
        ].sum()
        active_value = global_risk[active_indices].mean().to(local_risk.device)
    else:
        local_active = local_risk.sum() * 0.0
        active_value = torch.zeros((), dtype=local_risk.dtype, device=local_risk.device)
    local_maximum = local_risk[global_ordinals == maximum_ordinal].sum()
    selected_value = global_risk[order[:tail_count]].sum().to(local_risk.device)
    max_value = torch.as_tensor(maximum_value, dtype=local_risk.dtype, device=local_risk.device)
    cvar_surrogate = (actual_world_size / tail_count) * local_selected
    cvar = cvar_surrogate + (
        selected_value / tail_count - cvar_surrogate.detach()
    )
    active_surrogate = (
        actual_world_size / max(1, active_count)
    ) * local_active
    positive_tail_mean = active_surrogate + (
        active_value - active_surrogate.detach()
    )
    max_surrogate = actual_world_size * local_maximum
    maximum = max_surrogate + (max_value - max_surrogate.detach())
    return GlobalEnvelopeReduction(
        maximum=maximum,
        cvar=cvar,
        tail_count=tail_count,
        active_count=active_count,
        positive_tail_mean=positive_tail_mean,
        loss=(
            DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE.maximum_coefficient * maximum
            + DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE.positive_tail_mean_coefficient
            * positive_tail_mean
        ),
    )


def physical_envelope_loss(
    normalized_predictions: torch.Tensor,
    reached_outputs_normalized: torch.Tensor,
    *,
    normalization: object,
    joint_limits: torch.Tensor,
    phase_advance_cap: float,
    contract: PhysicalEnvelopeObjective,
    global_ordinals: torch.Tensor,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor]:
    risks = physical_envelope_risks(
        normalized_predictions,
        reached_outputs_normalized,
        normalization=normalization,
        joint_limits=joint_limits,
        phase_advance_cap=phase_advance_cap,
        contract=contract,
    )
    reductions = {
        name: global_max_plus_tail(
            value,
            global_ordinals=global_ordinals,
            tail_fraction=contract.tail_fraction,
            rank=rank,
            world_size=world_size,
        )
        for name, value in risks.items()
    }
    losses = {}
    for name, reduction in reductions.items():
        losses[name + "_maximum"] = reduction.maximum
        losses[name + "_cvar"] = reduction.cvar
        losses[name + "_positive_tail_mean"] = reduction.positive_tail_mean
        losses[name + "_envelope"] = reduction.loss
    losses["physical_envelope_total"] = sum(
        losses[name + "_envelope"] for name in risks
    )
    return losses
```

- [ ] **Step 5: Run focused and full training tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training
```

Expected: all tests pass with no warning or failure.

- [ ] **Step 6: Commit Task 1**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "feat: define PFNN physical envelope objective"
```

---

### Task 2: Materialize and seal the predecessor-pair population

**Files:**
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Consumes: the exact 2,048-row v2 fitted subset and existing lane-safe adjacency.
- Produces: `FittedTransitionPair`, `canonical_transition_pairs(...)`, compact `fitted_transition_pair_receipt(...)`, `materialize_transition_pairs(...)`, and predecessor-aware batch rows.

- [ ] **Step 1: Write RED adjacency, receipt, and target-feasibility tests**

Add
`test_predecessor_pair_receipt_is_complete_lane_safe_and_hash_bound`. Its
in-memory train dataset uses identity normalization and these deliberately
out-of-storage-order logical rows:

```text
storage  clip    lane          center  terrain
0        clip_b  motion        11      descent
1        clip_a  motion         3      flat
2        clip_a  motion         1      ascent
3        clip_a  idle_phase_0   2      transition
4        clip_b  motion        10      ascent
5        clip_a  motion         2      ascent
6        clip_a  idle_phase_0   1      flat
```

Every row has finite identity-normalized `x[288]`, `y[268]`, phase `0.25`,
split `train`, a nonempty split identity, and a valid v2 lane/class. Assert the
canonical pairs are exactly `clip_a/idle_phase_0/1->2`,
`clip_a/motion/1->2`, `clip_a/motion/2->3`, and
`clip_b/motion/10->11`; the receipt schema is
`mm-sonic-fitted-transition-pair-receipt/v1`; `sample_count == 4`; all four
class counts recompute from the current rows; the compact receipt has no
`pairs` field; and changing `pairs_sha256`, `sample_count`, an active-row field,
or a class count rejects with `pair receipt`. Reversing storage order must
produce byte-identical canonical receipt JSON.

Call the full-record validator with independently mutated sequences and require
rejection for: omitting pair 0, appending a nonadjacent extra pair, duplicating
pair 1, swapping pairs 1/2, and changing any source row split from `train` to
`validation`. Spy on receipt construction to prove it invokes both the primary
logical-key adjacency builder and the existing independent
`fitted_adjacent_indices`, and requires their complete ordered records to be
exactly equal.

Add `test_synthetic_target_envelope_audit_accepts_exact_boundaries_and_rejects_each_excess`.
Use identity normalization, native limits `[-1,1]` for every joint, phase cap
`0.50`, predecessor joint 0, and current target joint `0.225`, native joint
`0.98`, and phase `0.48`; the audit accepts equality. Set another target phase
to exactly `-9.5367431640625e-7`; require acceptance, an unclamped minimum equal
to that value, and audited minimum zero. In independent subtests, change only
one value to joint step `0.225001`, native joint `0.980001`, phase `-1e-6`, or
phase `0.480001` and require rejection naming that field. The real
recurrence-v2 values are intentionally verified read-only in Task 6, not loaded
by the hermetic unit suite.

Add separate regressions proving different lanes at the same center remain
distinct, gaps terminate pairs, duplicate `(clip,lane,center)` rows reject,
pair order is storage-independent, the current terrain class drives balancing,
and run starts never appear in the pair sampler.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_predecessor_pair_receipt_is_complete_lane_safe_and_hash_bound \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_synthetic_target_envelope_audit_accepts_exact_boundaries_and_rejects_each_excess
```

Expected: missing pair receipt and feasibility-audit APIs.

- [ ] **Step 3: Implement exact pair records and receipt validation**

```python
@dataclass(frozen=True)
class FittedTransitionPair:
    predecessor_index: int
    current_index: int
    clip_id: str
    sequence_lane: str
    predecessor_center_frame: int
    center_frame: int
    predecessor_row_sha256: str
    current_row_sha256: str
    terrain_class: str


def _pair_from_indices(
    dataset: object, predecessor_index: int, current_index: int
) -> FittedTransitionPair:
    predecessor = dataset[predecessor_index]
    current = dataset[current_index]
    if (
        predecessor["split"] != "train"
        or current["split"] != "train"
        or predecessor["clip_id"] != current["clip_id"]
        or predecessor["sequence_lane"] != current["sequence_lane"]
        or predecessor["center_frame"] + 1 != current["center_frame"]
    ):
        raise ValueError("transition pair is not train-only exact adjacency")
    return FittedTransitionPair(
        predecessor_index=predecessor_index,
        current_index=current_index,
        clip_id=str(current["clip_id"]),
        sequence_lane=str(current["sequence_lane"]),
        predecessor_center_frame=int(predecessor["center_frame"]),
        center_frame=int(current["center_frame"]),
        predecessor_row_sha256=fitted_row_sha256(predecessor),
        current_row_sha256=fitted_row_sha256(current),
        terrain_class=str(current["terrain_class"]),
    )


def _pair_receipt_record(pair: FittedTransitionPair) -> dict[str, object]:
    return {
        "clip_id": pair.clip_id,
        "sequence_lane": pair.sequence_lane,
        "predecessor_center_frame": pair.predecessor_center_frame,
        "center_frame": pair.center_frame,
        "predecessor_row_sha256": pair.predecessor_row_sha256,
        "current_row_sha256": pair.current_row_sha256,
        "terrain_class": pair.terrain_class,
    }


def canonical_transition_pairs(
    dataset: object,
) -> tuple[FittedTransitionPair, ...]:
    """Derive every canonical train-only same-lane adjacent pair internally."""
    rows: dict[tuple[str, str, int], int] = {}
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, Mapping) or row.get("split") != "train":
            raise ValueError("transition-pair source must be train-only")
        clip = row.get("clip_id")
        lane = row.get("sequence_lane")
        center = row.get("center_frame")
        if (
            type(clip) is not str or not clip
            or lane not in _SEQUENCE_LANES
            or type(center) is not int or center < 0
        ):
            raise ValueError("transition-pair provenance is invalid")
        fitted_row_sha256(row)
        key = (clip, lane, center)
        if key in rows:
            raise ValueError("duplicate transition-pair logical row")
        rows[key] = index
    pairs = [
        _pair_from_indices(dataset, predecessor_index, rows[(clip, lane, center + 1)])
        for (clip, lane, center), predecessor_index in rows.items()
        if (clip, lane, center + 1) in rows
    ]
    return tuple(sorted(
        pairs,
        key=lambda pair: (
            pair.clip_id, pair.sequence_lane, pair.predecessor_center_frame,
            pair.center_frame, pair.predecessor_row_sha256,
            pair.current_row_sha256,
        ),
    ))


def validate_canonical_transition_pairs(
    dataset: object, candidate: Sequence[FittedTransitionPair]
) -> tuple[FittedTransitionPair, ...]:
    primary = canonical_transition_pairs(dataset)
    independent = tuple(
        _pair_from_indices(dataset, predecessor, current)
        for predecessor, current in fitted_adjacent_indices(
            dataset, tuple(range(len(dataset)))
        )
    )
    checked = tuple(candidate)
    if checked != primary or checked != independent:
        raise ValueError("complete transition-pair recomputation mismatch")
    return checked


def fitted_transition_pair_receipt(dataset: object) -> dict[str, object]:
    active_row_digests = sorted(
        fitted_row_sha256(dataset[index]) for index in range(len(dataset))
    )
    pairs = validate_canonical_transition_pairs(
        dataset, canonical_transition_pairs(dataset)
    )
    class_counts = {name: 0 for name in _TERRAIN_CLASSES}
    for pair in pairs:
        class_counts[pair.terrain_class] += 1
    pair_payload = [_pair_receipt_record(pair) for pair in pairs]
    base = {
        "schema": "mm-sonic-fitted-transition-pair-receipt/v1",
        "active_row_count": len(active_row_digests),
        "active_rows_sha256": _canonical_sha256(
            {"row_sha256": active_row_digests}
        ),
        "sample_count": len(pairs),
        "class_counts": class_counts,
        "pairs_sha256": _canonical_sha256({"pairs": pair_payload}),
    }
    return {**base, "receipt_sha256": _canonical_sha256(base)}
```

`validate_fitted_transition_pair_receipt(receipt, dataset)` must require exact
compact keys/types, positive active-row count, all
four nonzero class counts, and exact equality with a freshly recomputed receipt.
It also recomputes full pair records through both adjacency paths before this
comparison. Active resume validation therefore detects omitted, extra,
duplicated, reordered, or split-invalid records and adding/removing a run-start
row, while checkpoint size remains independent of pair count.

- [ ] **Step 4: Implement pair materialization and exact feasibility audit**

`materialize_transition_pairs(dataset)` derives and independently verifies the
complete pair sequence itself; it accepts no caller-supplied adjacency list. It
must touch source indices once in sorted order
and return immutable copies containing `current`, `predecessor_y`, and the
canonical pair record. `fitted_target_envelope_audit` denormalizes targets once
and defines
`PHASE_AUDIT_NEGATIVE_TOLERANCE_RAD: float = 9.5367431640625e-7` and
asserts it equals `8 * np.finfo(np.float32).eps`.
It records `minimum_phase_advance_unclamped_rad`, rejects if that value is
strictly less than `-9.5367431640625e-7`, and only then uses
`np.maximum(phase, 0.0)` for the sealed audit. It rejects before training unless
all exact margins are feasible and emits both unclamped/clamped minima, maxima,
counts, and a canonical digest.

```python
def _predecessor_batch(dataset, indices, device):
    rows = [dataset[index] for index in indices]
    return (
        torch.as_tensor(np.stack([row["current"]["x"] for row in rows]), device=device),
        torch.as_tensor(np.asarray([row["current"]["phase"] for row in rows]), device=device),
        torch.as_tensor(np.stack([row["current"]["y"] for row in rows]), device=device),
        torch.as_tensor(np.stack([row["predecessor_y"] for row in rows]), device=device),
    )
```

- [ ] **Step 5: Run the full focused training suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/terrain_pfnn/training.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "feat: seal PFNN predecessor pairs"
```

---

### Task 3: Apply the envelope objective to one-step training and exact DDP batches

**Files:**
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Consumes: predecessor-pair dataset, pure envelope loss, balanced global epoch order.
- Produces: one-step updates with exact current targets plus predecessor-reached joint state and DDP-global ordinals.

- [ ] **Step 1: Write RED one-step and two-rank DDP equivalence tests**

```python
def test_one_step_envelope_uses_predecessor_target_and_reaches_model_gradient(self):
    model = self._single_parameter_output_model()
    current, phase, target, predecessor = self._unsafe_pair_batch()
    prediction = model(current, phase)
    envelope = physical_envelope_loss(
        prediction,
        predecessor,
        normalization=_normalization(),
        joint_limits=torch.tensor([[-1.0, 1.0]] * 29),
        phase_advance_cap=0.50,
        contract=DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
        global_ordinals=torch.arange(len(current)),
        rank=0,
        world_size=1,
    )
    envelope["physical_envelope_total"].backward()
    self.assertGreater(float(model.weight.grad.abs().sum()), 0.0)


def test_two_rank_global_max_plus_active_tail_matches_single_rank(self):
    expected_loss, expected_gradient = self._single_rank_tail_reference()
    results = self._spawn_two_rank_gloo_tail_workers()
    for loss, gradient in results:
        self.assertAlmostEqual(loss, expected_loss, places=7)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-7, rtol=0)
```

The gloo fixture uses two local processes, local batch 2, global ordinals
rank 0 `[0,2]`, rank 1 `[1,3]`, one tied risk, and a shared linear parameter.
Also assert the old isolated-row `_batch` path is unreachable in pipeline mode.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_one_step_envelope_uses_predecessor_target_and_reaches_model_gradient \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_two_rank_global_max_plus_active_tail_matches_single_rank
```

Expected: the trainer does not yet pass predecessor targets/global ordinals and
the DDP reference differs or the new helper is absent.

- [ ] **Step 3: Switch every one-step mode to active pair positions**

After `optimization_dataset` is selected—unchanged 2,048-row materialized
subset in pipeline mode and the full train dataset otherwise—derive pairs from
all of its rows:

```python
pair_records = validate_canonical_transition_pairs(
    optimization_dataset,
    canonical_transition_pairs(optimization_dataset),
)
pair_receipt = fitted_transition_pair_receipt(optimization_dataset)
pair_dataset = materialize_transition_pairs(optimization_dataset)
pair_classes = [pair_dataset[index]["current"]["terrain_class"] for index in range(len(pair_dataset))]
pair_indices = list(range(len(pair_dataset)))
```

Use `pair_indices/pair_classes` in `_balanced_epoch_indices`. Keep fixed-sample
evaluation on all 2,048 fitted rows in pipeline mode. Before training, assert
the active pair receipt equals the checkpoint-bound receipt on resume. In full
training, bind the full train-only pair receipt and never reuse the controlled
rerun's 2,011-pair digest.

- [ ] **Step 4: Add envelope losses to every one-step update**

For local batch item `i`, set its global ordinal to
`rank + world_size * i`. Compute base losses exactly as before, then:

```python
envelope = physical_envelope_loss(
    prediction,
    predecessor_y,
    normalization=train_dataset,
    joint_limits=kinematics.joint_limits,
    phase_advance_cap=phase_advance_cap,
    contract=DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
    global_ordinals=(
        torch.arange(len(batch_indices), device=device, dtype=torch.int64) * world_size + rank
    ),
    rank=rank,
    world_size=world_size,
)
losses.update(envelope)
losses["total"] = losses["total"] + envelope["physical_envelope_total"]
```

The padded DDP sampler must guarantee the same positive local batch count on
every rank; assert that the gathered ordinals sort to exact `0..N-1` before the
reduction. Metric JSON must include every family maximum, ordinary CVaR,
tail count, active-positive count/mean, and combined family loss. Every
one-step count is a predecessor/current pair count: a pair contributes at most
one to a family count regardless of how many joints or hinge sides are active.
Reject nonfinite envelope loss or gradient before `optimizer.step()`.

- [ ] **Step 5: Run focused training and DDP tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training
```

Expected: all tests, including the two-process gloo regression, pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/terrain_pfnn/training.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "feat: optimize PFNN fitted envelopes"
```

---

### Task 4: Apply predecessor-aware envelope loss through rollout recurrence

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`
- Modify: `tests/python/test_terrain_pfnn_recurrence.py`

**Interfaces:**
- Consumes: a lane-safe 16-frame sequence, exact predecessor target for its first row, and global batch ordinals.
- Produces: one lowest-ordinal global maximum plus selected-positive tail mean per envelope family over all `16 * global_batch` pair-time elements, with ordinary CVaR diagnostics and gradients through reached prior predictions.

- [ ] **Step 1: Write RED rollout reached-state and late-gradient tests**

```python
def test_rollout_envelope_uses_seed_predecessor_then_prior_prediction(self):
    model = self._scripted_three_step_model()
    rollout = autoregressive_unroll(
        model,
        self.inputs,
        self.initial_phase,
        self.targets,
        initial_predecessor_targets=self.predecessor_targets,
        normalization=self.normalization,
        phase_advance_cap=self.phase_cap,
        joint_limits=self.joint_limits,
        envelope_contract=DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
        global_batch_ordinals=torch.arange(self.batch_size),
        rank=0,
        world_size=1,
    )
    expected_step0 = (
        torch.relu(
            torch.abs(self.predictions[0][0, self.joint] - self.predecessor_joint)
            - 0.225
        ) / 0.025
    ).square()
    expected_step1 = (
        torch.relu(
            torch.abs(
                self.predictions[1][0, self.joint]
                - self.predictions[0][0, self.joint]
            ) - 0.225
        ) / 0.025
    ).square()
    self.assertEqual(rollout.envelope_risks["joint_step"][0, 0], expected_step0)
    self.assertEqual(rollout.envelope_risks["joint_step"][1, 0], expected_step1)


def test_late_rollout_envelope_gradient_reaches_prior_prediction(self):
    rollout = self._unsafe_three_step_rollout()
    gradient = torch.autograd.grad(
        rollout.losses["joint_step_envelope"],
        rollout.predictions[0],
        retain_graph=True,
    )[0]
    self.assertTrue(torch.isfinite(gradient).all())
    self.assertGreater(float(gradient.abs().sum()), 0.0)
```

Add a discovery test proving every 16-frame sequence has a fitted same-lane
predecessor and gaps/lane transitions eliminate the sequence. Add a rollout
safe-append regression with one positive pair-time risk and arbitrary additional
zero-risk times/batch items; family loss and the lone offender gradient must
remain exact. Assert rollout active/failure metrics count pair-time elements,
so one `[time,batch]` element contributes at most one regardless of joint count.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_rollout_envelope_uses_seed_predecessor_then_prior_prediction \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_late_rollout_envelope_gradient_reaches_prior_prediction
```

Expected: `autoregressive_unroll` rejects the required predecessor/objective
arguments or lacks envelope losses.

- [ ] **Step 3: Extend the unroll contract and collect pair-time risks**

Change the signature to require:

```python
def autoregressive_unroll(
    model: nn.Module,
    ground_truth_inputs: torch.Tensor,
    initial_phase: torch.Tensor,
    targets: torch.Tensor,
    *,
    initial_predecessor_targets: torch.Tensor,
    normalization: object,
    phase_advance_cap: float,
    joint_limits: torch.Tensor,
    envelope_contract: PhysicalEnvelopeObjective,
    global_batch_ordinals: torch.Tensor,
    rank: int,
    world_size: int,
    kinematics: nn.Module | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> RolloutResult:
```

Initialize `reached = initial_predecessor_targets`. At each step append
`physical_envelope_risks(prediction, reached, ...)`, then assign
`reached = prediction` without `detach`, `clone().detach`, NumPy conversion, or
in-place overwrite. Stack each family `[T,B]`, flatten time-major, and define
ordinal `time_index * (local_batch * world_size) + global_batch_ordinal` before
calling `global_max_plus_tail`. Assert the flattened global ordinals sort to
exact `0..T*global_batch-1`. Add the unflattened differentiable `[T,B]` tensors
to `RolloutResult.envelope_risks`; this makes the reached-state contract
directly testable and must not create a second or detached computation path.

- [ ] **Step 4: Make sequence discovery require a fitted predecessor**

Return a sequence record containing both `indices` and `predecessor_index`.
Require exact same clip/lane and `first_center - 1`; reject duplicates. Update
the deterministic sequence sampler to sample sequence-record positions without
changing its permutation algorithm. Batch `initial_predecessor_targets` from
the predecessor row's normalized `y`.

- [ ] **Step 5: Run recurrence, training, and runtime regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime
```

Expected: all tests pass; runtime output and thresholds remain unchanged.

- [ ] **Step 6: Commit Task 4**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_recurrence.py
git commit -m "feat: train PFNN rollout envelopes"
```

---

### Task 5: Bind objective, pair population, resume, reports, and promotion

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `sonic/python/mm_sonic/evaluate_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Consumes: exact objective contract, fitted pair receipt, target-feasibility audit, sampler state, and unchanged fitted report.
- Produces: checkpoint v6, one-step report v2, promotion receipt v3, and fail-closed safe reload/resume/evaluator ordering.

- [ ] **Step 1: Write RED schema and pre-restore tamper tests**

```python
def test_v6_checkpoint_binds_envelope_objective_pairs_and_feasibility_before_restore(self):
    checkpoint = self._write_v6_checkpoint()
    for field in ("physical_envelope_objective", "fitted_pair_receipt", "target_envelope_audit"):
        tampered = self._tamper_and_rehash_checkpoint(checkpoint, field)
        with self.subTest(field=field), self.assertRaises(ValueError):
            load_checkpoint(
                tampered,
                expected_dataset_digest=self.dataset_digest,
                expected_kinematic_signature_sha256=self.signature,
            )
    with self.assertRaisesRegex(ValueError, "schema"):
        load_checkpoint(self._write_v5_checkpoint(), expected_dataset_digest=self.dataset_digest,
                        expected_kinematic_signature_sha256=self.signature)


def test_active_pair_or_objective_mismatch_rejects_before_model_and_adam_restore(self):
    with mock.patch.object(PhaseFunctionedNetwork, "load_state_dict") as model_restore, \
         mock.patch.object(torch.optim.Adam, "load_state_dict") as adam_restore:
        with self.assertRaisesRegex(ValueError, "active physical envelope"):
            train_cli.train(self._resume_arguments_with_pair_digest("0" * 64))
    model_restore.assert_not_called()
    adam_restore.assert_not_called()
```

Add report/receipt tampering tests for every objective constant, pair digest,
target audit including the exact tolerance/unclamped minimum, envelope metric,
checkpoint schema, one-step schema, and v2/v3 promotion schema. Assert the
checkpoint compact receipt has no full-record list. Every rejection spy must
show model/Adam restoration and the known-terrain callback were never opened.
Add `test_dataset_file_snapshot_persists_hash_size_and_mtime_ns_atomically`:
construct the exact six relative paths used by the recurrence-v2 canary with
distinct bytes and nanosecond mtimes, assert canonical path order and field
values, atomically persist/reload identical JSON, and prove a byte, size, or
mtime-only mutation changes the snapshot digest. Missing/extra controlled-path
checks belong to Task 7 preflight, not the generic snapshot function.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_v6_checkpoint_binds_envelope_objective_pairs_and_feasibility_before_restore \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_active_pair_or_objective_mismatch_rejects_before_model_and_adam_restore
```

Expected: v5 still loads or new fields/schema validation are absent.

- [ ] **Step 3: Advance and validate exact schemas**

Set:

```python
CHECKPOINT_SCHEMA = "mm-sonic-terrain-pfnn-checkpoint/v6"
ONE_STEP_REPORT_SCHEMA = "mm-sonic-terrain-pfnn-one-step-report/v2"
PIPELINE_PROMOTION_RECEIPT_SCHEMA = "mm-sonic-pipeline-promotion-receipt/v3"
```

Checkpoint v6 adds exact top-level keys
`physical_envelope_objective`, `fitted_pair_receipt`, and
`target_envelope_audit`. Add `joint_step_envelope`,
`joint_limit_envelope`, and `phase_envelope` to the exact loss-weight map at
weight `1.0`. Safe load validates all types, constants, finite values, digests,
compact pair counts/digests, and audit tolerance/margins before building a
model. It does not deserialize full pair records from the checkpoint because
none are stored.

- [ ] **Step 4: Add canonical per-file snapshot persistence**

Add to `train_terrain_pfnn.py`:

```python
def dataset_file_snapshot(dataset_root: Path) -> dict[str, object]:
    root = dataset_root.resolve(strict=True)
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("dataset file changed while hashing")
        records.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        })
    if not records:
        raise ValueError("dataset snapshot cannot be empty")
    base = {"schema": "mm-sonic-dataset-file-snapshot/v1", "files": records}
    return {**base, "snapshot_sha256": _canonical_sha256(base)}


def write_dataset_file_snapshot(dataset_root: Path, destination: Path) -> None:
    payload = dataset_file_snapshot(dataset_root)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
```

- [ ] **Step 5: Bind resume before restoration**

Construct the fitted subset, both independently derived complete pair-record
sequences, compact pair receipt, feasibility audit, pair sampler, rollout
sequences, and active objective before model creation. Require the two full
record sequences to match exactly, then compare the freshly compacted receipt
with the checkpoint before model/Adam restoration.
Compare saved and active values before calling `restore_training_state`.
Retain the existing exact one-step `epoch/global_offset` and deterministic
sequence cursor validation; additionally require their active populations to
match the pair and sequence counts/digests.

- [ ] **Step 6: Advance reports and promotion receipt**

Before constructing one-step report v2, convert each detached scalar tensor
once with `.to(device="cpu", dtype=torch.float64).item()`, validate that its
exact type is `float` and it is finite, and assign it to the correspondingly
named `*_value` below. Compute maxima and counts from the complete fitted-pair
report, not a last minibatch. The report records:

```python
{
    "physical_envelope_objective": objective_payload,
    "physical_envelope_objective_sha256": objective_digest,
    "fitted_pair_receipt": pair_receipt,
    "fitted_pair_receipt_sha256": pair_receipt["receipt_sha256"],
    "target_envelope_audit": target_audit,
    "envelope_metrics": {
        "pair_count": fitted_pair_count,
        "joint_step_maximum_risk": joint_step_maximum_risk_value,
        "joint_step_cvar": joint_step_cvar_value,
        "joint_step_tail_count": joint_step_tail_count,
        "joint_step_active_count": joint_step_tail_active_count,
        "joint_step_positive_tail_mean": joint_step_positive_tail_mean_value,
        "joint_step_family_loss": joint_step_family_loss_value,
        "joint_limit_maximum_risk": joint_limit_maximum_risk_value,
        "joint_limit_cvar": joint_limit_cvar_value,
        "joint_limit_tail_count": joint_limit_tail_count,
        "joint_limit_active_count": joint_limit_tail_active_count,
        "joint_limit_positive_tail_mean": joint_limit_positive_tail_mean_value,
        "joint_limit_family_loss": joint_limit_family_loss_value,
        "phase_maximum_risk": phase_maximum_risk_value,
        "phase_cvar": phase_cvar_value,
        "phase_tail_count": phase_tail_count,
        "phase_active_count": phase_tail_active_count,
        "phase_positive_tail_mean": phase_positive_tail_mean_value,
        "phase_family_loss": phase_family_loss_value,
        "maximum_joint_step_rad": maximum_joint_step_rad_value,
        "maximum_joint_limit_excess_rad": maximum_joint_limit_excess_rad_value,
        "minimum_phase_advance_unclamped_rad": minimum_phase_advance_unclamped_rad_value,
        "maximum_phase_advance_rad": maximum_phase_advance_rad_value,
        "joint_step_objective_active_count": joint_step_objective_active_count,
        "joint_limit_objective_active_count": joint_limit_objective_active_count,
        "phase_objective_active_count": phase_objective_active_count,
        "joint_step_runtime_failure_count": joint_step_runtime_failure_count,
        "joint_limit_runtime_failure_count": joint_limit_runtime_failure_count,
        "phase_runtime_failure_count": phase_runtime_failure_count,
    },
}
```

Require each count's exact type to be `int` and each to be nonnegative. For
this one-step report, `pair_count`, objective-active counts, tail counts, and
runtime-failure counts all count canonical predecessor/current pairs; they
never count joints or hinge sides. Metric-stream fields produced by rollout
use the same names suffixed `_pair_time_count` and count flattened `[T,B]`
pair-time elements, never pairs alone, joints, sequences, or hinge sides.
Promotion receipt v3 and the evaluator request bind the exact objective and
pair digests. Keep the unchanged fitted-transition report v1 nested and
separately hashed. Any missing/rejected fitted report still blocks the callback
before receipt construction. These diagnostic values never replace the
unchanged fitted-transition acceptance thresholds.

- [ ] **Step 7: Run focused GREEN suites**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime
```

Expected: every v6/v3 round-trip and tamper test passes.

- [ ] **Step 8: Commit Task 5**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/evaluate_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_runtime.py
git commit -m "feat: bind PFNN envelope training receipts"
```

---

### Task 6: Review the source correction and prove the complete pre-run gate

**Files:**
- Modify only for RED-backed review fixes: files from Tasks 1-5
- Write ignored report: `.superpowers/sdd/physical-envelope-objective-source-report.md`

**Interfaces:**
- Consumes: committed Tasks 1-5.
- Produces: independently reviewed source range and complete green pre-run evidence.

- [ ] **Step 1: Run the complete PFNN suite with the dependency split**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_sources \
  tests.python.test_terrain_pfnn_phase

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_dataset \
  tests.python.test_terrain_pfnn_kinematics \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime
```

Expected: all tests pass in their dependency-specific environments.

- [ ] **Step 2: Run static verification**

```bash
git diff --check
find sonic/python/mm_sonic/terrain_pfnn -maxdepth 1 -name '*.py' -print | sort | \
  xargs sonic/.torch-mm-venv/bin/python -m py_compile
printf '%s\n' \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/evaluate_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_runtime.py | \
  xargs sonic/.torch-mm-venv/bin/python -m py_compile
```

Expected: every command exits 0 with no output.

- [ ] **Step 3: Recompute the real fitted target-feasibility audit read-only**

Use the accepted recurrence-v2 dataset and exact 2,048-row selection without
training. Require these observed bounds or stricter:

```text
maximum target joint step       <= 0.19434923564685413 rad
minimum native-limit clearance  >= 0.03785574245610318 rad
maximum target phase advance    <= 0.34906582224089533 rad
minimum target phase unclamped   >= -0.00000095367431640625 rad
objective joint onset            = 0.225 rad
objective native inset           = 0.020 rad
objective upper phase boundary   = 0.503598733361343 rad
```

Verify `phase_audit_negative_tolerance_rad == 9.5367431640625e-7`, the
unclamped minimum is persisted before the audit-only clamp, receipt digests
twice, and no validation/test row was loaded into the selection or audit.

- [ ] **Step 4: Request independent task review**

Generate a review package from `d94b443` through the Task 5 source head. Give
the reviewer this plan, the design, TDD report, and diff package. Require both
spec compliance and code quality verdicts. Fix every Critical/Important item
with a new focused RED test, rerun its covering suite, and re-review.

- [ ] **Step 5: Record pre-run authorization**

Write exact commands, test counts/timings, commit range, review verdict,
objective digest, pair receipt/count/classes, target audit, dataset digest,
and confirmation that no run artifact was created. Record the frozen source
commit and hashes of every Task 1-5 source/test file. The one controlled run is
not authorized until this report is complete and review is clean.

---

### Task 7: Execute exactly one corrected recurrence-v2 run and gate it

**Files:**
- Generate ignored artifacts: `sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1/`
- Write ignored report: `.superpowers/sdd/physical-envelope-objective-artifact-report.md`
- Write ignored snapshots: `.superpowers/sdd/physical-envelope-objective-dataset-before.json`, `.superpowers/sdd/physical-envelope-objective-dataset-after.json`
- Do not modify source or tests after the sole corrected training command starts

**Interfaces:**
- Consumes: reviewed v6 source, accepted recurrence-v2 dataset, canonical G1 MJCF, and exact known-terrain evaluator.
- Produces on success: accepted v6 candidate, fitted report, exact 600-tick JSON, v3 promotion receipt, and immutable best.

- [ ] **Step 1: Persist before/after snapshots around exact dataset resume**

Implement and test before Task 6 a pure
`dataset_file_snapshot(dataset_root: Path) -> dict[str, object]` and atomic
`write_dataset_file_snapshot(dataset_root, destination)`. The exact JSON schema
is `mm-sonic-dataset-file-snapshot/v1`; `files` is a relative-path-sorted list
of exact `{path, sha256, size, mtime_ns}` records and `snapshot_sha256` hashes
the canonical JSON excluding itself. For this dataset require exactly:

```text
manifest.json
normalization.npz
test/shard_00000.npz
train/shard_00000.npz
train/shard_00001.npz
validation/shard_00000.npz
```

Persist the before JSON, run the exact resume, then persist the after JSON:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -c \
  'from pathlib import Path; from mm_sonic.train_terrain_pfnn import write_dataset_file_snapshot; write_dataset_file_snapshot(Path("sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2"), Path(".superpowers/sdd/physical-envelope-objective-dataset-before.json"))'

PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2 \
  --terrain-family-limit 2 --resume

PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -c \
  'from pathlib import Path; from mm_sonic.train_terrain_pfnn import write_dataset_file_snapshot; write_dataset_file_snapshot(Path("sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2"), Path(".superpowers/sdd/physical-envelope-objective-dataset-after.json"))'
```

Require byte-identical canonical entries and equal `snapshot_sha256` values;
also record the separate SHA-256 of each persisted JSON. Verify dataset digest
`9e304bafe780aa694467e01912b7a04a67a0526d3c81a612a9618658ba4bdd48`,
v2 lane fields, split isolation, and train-only normalization. Hashing the test
shard as bytes is allowed; do not deserialize sealed test examples or evaluate
them.

- [ ] **Step 2: Confirm the new output path is absent and GPU 1 is available**

```bash
test ! -e sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

If the output exists, stop and resolve provenance; do not overwrite or resume.
Reconfirm the frozen source commit/hashes from Task 6 and record the exact UTC
freeze timestamp. From immediately before Step 3 starts, the correction cycle
has no authority to modify or commit source/tests.

- [ ] **Step 3: Run the one and only controlled training attempt**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.train_terrain_pfnn \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1 \
  --overfit-samples 2048 --steps 4000 --seed 7 \
  --rollout-finetune-frames 16 --rollout-finetune-steps 256
```

Expected before terrain evaluation: finite base/envelope losses, v6 candidate,
safe reload, exact fixed-sample reproduction, exact pair/objective receipts,
accepted 2,011-pair fitted-transition report, and no `best.pt` yet.

- [ ] **Step 4: Apply the first-failure stop rule**

If training, safe reload, fixed-sample reproduction, target audit, or fitted
gate fails, record exact clip/lane/predecessor/current center/field/joint/value/
limit and all receipt digests, then stop. Do not launch the evaluator, train
again, resume, change a margin/tail/weight/threshold, patch output, or edit
source/tests. A representation defect is evidence for a separately approved
future correction cycle, not authorization for an in-cycle patch.

- [ ] **Step 5: Run the exact 600-tick verifier only after fitted acceptance**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.evaluate_terrain_pfnn \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1/checkpoint-step-00004256.pt \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --split train --closed-loop-seconds 20 --promote-pipeline-best
```

Expected: 600 finite committed ticks, zero holds/reversal/freeze, every
unchanged raw/root/joint/collision/stance gate, supported
plateau-flank-turnaround-return traversal, exact fixed/fitted/objective/pair
bindings, immutable `best.pt`, and v3 receipt.

- [ ] **Step 6: Verify promoted artifacts**

Safely reload candidate and best; re-evaluate all fitted pairs and the bound
600-tick receipt; verify candidate/best SHA relationships, objective and pair
digests, `0444` best permissions, atomic receipt, and absence of test-derived
state. Repeat the complete split PFNN suite, `py_compile`, and `git diff --check`.

- [ ] **Step 7: Write the artifact report without source mutation**

Record exact commands, outputs, timings, loss curves, target/objective/fitted
metrics, hashes, permissions, callback count, and first-failure absence or
presence. Include both persisted snapshot JSON hashes/digests and exact
per-file equality. Generated artifacts remain ignored. Create no source/test
commit; only ignored evidence is authorized in this correction cycle.

---

### Task 8: Re-review original Task 7 after an accepted artifact

**Files:**
- Do not modify correction source/test files
- Update only after approval: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: original Task 7 brief/report, both correction designs/plans, source/artifact reports, and full diff from `6af614f`.
- Produces: independent Task 7 spec-compliance and code-quality approval.

- [ ] **Step 1: Stop if Task 7 artifact gate did not pass**

Require accepted fitted report, accepted 600-tick JSON, immutable best, valid v3
receipt, safe reload, and complete post-artifact tests. If any is absent, Task 8
does not begin.

- [ ] **Step 2: Generate the review package**

Generate a full package from `6af614f` through the final source head. Include:

```text
.superpowers/sdd/task-7-brief.md
.superpowers/sdd/task-7-report.md
docs/superpowers/specs/2026-08-07-g1-terrain-pfnn-recurrence-correction-design.md
docs/superpowers/plans/2026-08-07-g1-terrain-pfnn-recurrence-correction.md
docs/superpowers/specs/2026-08-08-g1-terrain-pfnn-physical-envelope-objective-design.md
docs/superpowers/plans/2026-08-08-g1-terrain-pfnn-physical-envelope-objective.md
.superpowers/sdd/physical-envelope-objective-source-report.md
.superpowers/sdd/physical-envelope-objective-artifact-report.md
```

- [ ] **Step 3: Obtain both review verdicts without changing frozen source**

Require explicit `Spec compliance: approved` and `Code quality: approved`.
If either reviewer finds a Critical/Important issue, record exact file/line,
evidence, proposed RED, and the future correction required, then stop this
cycle. Do not write the RED, implement a fix, change thresholds, or retrain.
Proceed only when both verdicts approve the unchanged frozen source.

- [ ] **Step 4: Record durable progress only after approval**

Append the exact objective-correction commit range, artifact hashes, and Task 7
review verdict to `.superpowers/sdd/progress.md`. Do not mark original Tasks 8
or 9 complete.

---

### Task 9: Resume original Tasks 8 and 9, ending with the interactive visual drive

**Files:**
- Follow exact original Task 8/9 files in `docs/superpowers/plans/2026-08-07-g1-terrain-pfnn.md`
- Create/modify only the viewer, full-corpus artifacts, and text evidence authorized there

**Interfaces:**
- Consumes: independently approved Task 7 and immutable pipeline best.
- Produces: three-hill viewer, validation-selected full model, one sealed-test result, headless canary, results report, and recorded interactive acceptance.

- [ ] **Step 1: Execute original Task 8 exactly**

Build `hill_map.py`, `terrain_pfnn_viewer.py`, and their tests test-first. Use
the accepted physical-envelope pipeline best for the 600-step headless canary:

```bash
PYNPUT_BACKEND=dummy PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.terrain_pfnn_viewer \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1/best.pt \
  --no-viewer --smoke-steps 600 --trace-every 30
```

Require finite output, no holds, and no IK/lock/portal state before Task 8 review.

- [ ] **Step 2: Execute original Task 9 through validation selection**

Run the complete suites, build the full sealed dataset, train with eight GPUs,
and select only on validation exactly as the original plan specifies. Carry
checkpoint v6/objective/pair receipts through full training. If validation or
fitted envelopes fail, stop without opening sealed test.

- [ ] **Step 3: Open the sealed test exactly once and run the headless canary**

Only after validation acceptance, execute the original sealed-test command and
the 1,200-step no-viewer three-hill command. Record every segment and global
gate. Do not rerun sealed test for tuning.

- [ ] **Step 4: Perform the final interactive visual drive**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m mm_sonic.terrain_pfnn_viewer \
  --checkpoint sonic/runs/terrain-pfnn-v1/full-training/best.pt
```

Physically hold and release W/A/S/D to verify flat response, uphill traversal,
summit, downhill traversal, direction changes while on a flank, both crossing
directions, stop/restart, and resumed flat walking. Watch the rendered feet,
root, posture, and contact diagnostics for sliding, penetration, freeze,
reversal, discontinuity, or hold. Record exact visual observations and trace
timestamps. Do not enable IK or alter the checkpoint.

- [ ] **Step 5: Complete original Task 9 evidence and review**

Write `docs/research/terrain-pfnn-v1-results.md` with dataset/checkpoint/source
digests, every automated/sealed threshold, headless traces, and interactive
observations. Set status `Implemented` only if all automated, sealed, headless,
and interactive gates pass; otherwise use `Prototype evaluated` and list exact
failures. Run final verification, confirm no run artifact is staged, commit only
the authorized text evidence, and request final whole-branch review.

---

## Plan completion handoff

Implementation begins with Task 1 under subagent-driven development. The
single controlled run is not authorized until Tasks 1-6 are independently
green and reviewed. A failed Task 7 artifact ends this correction cycle. A
successful Task 7 proceeds continuously through Task 7 re-review, original
Tasks 8/9, and the final interactive visual drive.
