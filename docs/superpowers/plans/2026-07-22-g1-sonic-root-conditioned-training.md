# G1 SONIC Root-Conditioned Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the released SONIC G1 motion encoder and G1 kinematic decoder to encode and reconstruct the canonical root trajectory while the control decoder and all non-G1 components remain frozen.

**Architecture:** Training adds a separate `[N,10,3]` tokenizer term derived from the existing first-reference-heading root transform. A deterministic checkpoint migrator expands the G1 encoder's first layer frame-by-frame and the G1 kinematic decoder's final layer frame-by-frame, copying every released parameter and zero-initializing only new root rows/columns. Stage A runs the existing SONIC trainer with only the G1 encoder and G1 kinematic decoder trainable; deployment export retains the multimode superset and appends the root term as values `[1762,1792)`.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, Isaac Lab, GEAR SONIC universal-token module, ONNX export, pytest.

## Global Constraints

- GEAR source identity is commit `294110cedba01ad764f1e268d57ddf7c1bbf9523`; perform work in the clean implementation worktree created for the root-stream plan.
- Download the official checkpoint with `python download_from_hf.py --training --no-smpl`; source checkpoint path is `sonic_release/last.pt`.
- Training logical root term is `motion_root_position_refheading_mf_nonflat` with shape `[num_envs,10,3]`.
- Its values are the position slice `root_transforms_relative_to_first_frame[...,0:3]`: local delta XY and absolute physical pelvis Z.
- G1 encoder input per frame changes from 64 to 67 values; flattened learned-layer input changes from 640 to 670.
- G1 kinematic decoder output per frame changes from 64 to 67 values; flattened learned-layer output changes from 640 to 670.
- Released 64-value per-frame order remains `joint_pos[29], joint_vel[29], orientation_6d[6]`; root XYZ is appended as values 64 through 66 of each frame inside the G1 network.
- Exported multimode ONNX input preserves all existing 1,762 superset values and appends root values at `[1762,1792)`.
- New encoder columns and new kinematic-decoder rows/biases initialize to exact zero; all copied values must be bit-equal to the released checkpoint.
- The G1 dynamic/control decoder, teleop encoder, SMPL encoder, quantizer, and running statistics remain frozen in Stage A; the critic follows the existing trainer because it is not part of the deployed control path.
- Stage A actor observations contain no privileged measured root pose, terrain height map, or global robot XY.
- Stage A data mixes official SONIC robot motions and physical-pelvis Motion Matching references; every source file and generated manifest is SHA-256 pinned.
- Stage B PPO terrain fine-tuning is out of scope until the evaluation plan emits `stage_b_required: true`.

---

## File Map

- Modify `gear_sonic/envs/manager_env/mdp/observations.py`: root-position observation function and tokenizer config field.
- Create `gear_sonic/config/manager_env/observations/terms/motion_root_position_refheading_mf_nonflat.yaml`: Hydra observation term.
- Create `gear_sonic/config/manager_env/observations/tokenizer/unitoken_all_noz_root.yaml`: tokenizer schema with root appended.
- Modify `gear_sonic/trl/modules/universal_token_modules.py`: per-decoder freeze support and trainable-module audit.
- Modify `gear_sonic/trl/losses/token_losses.py`: three-key G1 reconstruction/FK compatibility.
- Create `gear_sonic/config/actor_critic/encoders/g1_root_mf_mlp.yaml`: root-conditioned G1 encoder definition.
- Create `gear_sonic/config/actor_critic/decoders/g1_root_kin_mf_mlp.yaml`: root-reconstructing kinematic decoder definition.
- Create `gear_sonic/config/actor_critic/universal_token/all_mlp_v1_root_stage_a.yaml`: full model with only G1 encoder and G1 kinematic decoder trainable.
- Create `gear_sonic/config/aux_losses/terms/g1_root_recon.yaml`: aligned reconstruction loss.
- Create `gear_sonic/config/aux_losses/universal_token/g1_root_stage_a.yaml`: Stage-A loss selection.
- Create `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_root_stage_a.yaml`: training experiment.
- Create `gear_sonic/tools/expand_sonic_root_checkpoint.py`: deterministic checkpoint migration utility.
- Create `gear_sonic/tools/build_root_conditioning_dataset.py`: immutable mixed-dataset manifest/materializer.
- Create `gear_sonic/tests/test_root_conditioning.py`: pure observation, loss, freeze, and checkpoint tests.
- Create `gear_sonic/tests/test_root_dataset.py`: dataset identity tests.
- Create `gear_sonic/tools/verify_root_onnx.py`: ONNX layout, parity, and sensitivity verifier.
- Create `docs/root_conditioned_stage_a.md`: exact training/export/verification runbook and artifact identities.

---

### Task 1: Add the Training-Side Root Observation

**Files:**
- Modify: `gear_sonic/envs/manager_env/mdp/observations.py`
- Create: `gear_sonic/config/manager_env/observations/terms/motion_root_position_refheading_mf_nonflat.yaml`
- Create: `gear_sonic/config/manager_env/observations/tokenizer/unitoken_all_noz_root.yaml`
- Create: `gear_sonic/tests/test_root_conditioning.py`

**Interfaces:**
- Consumes: `TrackingCommand.root_transforms_relative_to_first_frame` shaped `[N,F,9]`.
- Produces: `motion_root_position_refheading_mf(env, command_name, non_flatten=False)` returning `[N,F,3]` when non-flat and `[N,F*3]` otherwise.

- [ ] **Step 1: Write failing shape/value tests using a fake command manager**

```python
def test_root_position_observation_is_position_slice_and_preserves_absolute_z():
    transforms = torch.arange(2 * 10 * 9, dtype=torch.float32).reshape(2, 10, 9)
    env = fake_env(root_transforms=transforms)
    actual = observations.motion_root_position_refheading_mf(env, "motion", non_flatten=True)
    torch.testing.assert_close(actual, transforms[..., :3], rtol=0.0, atol=0.0)
    assert actual.shape == (2, 10, 3)

def test_root_position_observation_flattens_only_when_requested():
    transforms = torch.randn(2, 10, 9)
    env = fake_env(root_transforms=transforms)
    assert observations.motion_root_position_refheading_mf(env, "motion").shape == (2, 30)
```

- [ ] **Step 2: Run the tests and confirm the missing function**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py -q`

Expected: failure reports that `motion_root_position_refheading_mf` is absent.

- [ ] **Step 3: Implement the observation**

```python
def motion_root_position_refheading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten: bool = False
) -> torch.Tensor:
    """Root delta XY and absolute pelvis Z in the first reference heading frame."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    position = command.root_transforms_relative_to_first_frame[..., :3]
    if non_flatten:
        return position.reshape(env.num_envs, command.num_future_frames, 3)
    return position.reshape(env.num_envs, command.num_future_frames * 3)
```

Add `motion_root_position_refheading_mf_nonflat = None` to `TokenizerCfg`. The term file is:

```yaml
motion_root_position_refheading_mf_nonflat:
  _target_: isaaclab.managers.ObservationTermCfg
  func: gear_sonic.envs.manager_env.mdp:motion_root_position_refheading_mf
  params:
    command_name: "motion"
    non_flatten: true
```

Copy `unitoken_all_noz.yaml` and append the new term after `motion_anchor_ori_b_mf_nonflat`; add additive uniform noise `[-0.02,0.02]` to XY and Z together.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py -q`

Expected: observation tests pass.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/envs/manager_env/mdp/observations.py gear_sonic/config/manager_env/observations/terms/motion_root_position_refheading_mf_nonflat.yaml gear_sonic/config/manager_env/observations/tokenizer/unitoken_all_noz_root.yaml gear_sonic/tests/test_root_conditioning.py
git commit -m "feat: add SONIC root trajectory observation"
```

### Task 2: Make the Stage-A Trainable Set Explicit

**Files:**
- Modify: `gear_sonic/trl/modules/universal_token_modules.py`
- Modify: `gear_sonic/tests/test_root_conditioning.py`

**Interfaces:**
- Consumes: optional `freeze: true` on individual decoder configs, matching existing per-encoder freeze semantics.
- Produces: frozen decoder parameters and `trainable_parameter_groups() -> dict[str, int]` for a fail-closed audit.

- [ ] **Step 1: Write failing freeze-contract tests**

```python
def test_per_decoder_freeze_leaves_only_g1_encoder_and_g1_kin_trainable():
    module = make_root_module(g1_dyn_freeze=True, g1_kin_freeze=False)
    groups = module.trainable_parameter_groups()
    assert groups["encoders.g1"] > 0
    assert groups["decoders.g1_kin"] > 0
    assert groups["decoders.g1_dyn"] == 0
    assert groups["encoders.teleop"] == 0
    assert groups["encoders.smpl"] == 0
    assert groups["quantizer"] == 0
```

- [ ] **Step 2: Run and observe the missing audit method**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py::test_per_decoder_freeze_leaves_only_g1_encoder_and_g1_kin_trainable -q`

Expected: failure reports `UniversalTokenModule` has no `trainable_parameter_groups`.

- [ ] **Step 3: Implement symmetric decoder freezing and audit**

After constructing each decoder:

```python
if decoder_config.get("freeze", False):
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    logger.info(f"Froze decoder: {decoder_name}")
```

Add the audit:

```python
def trainable_parameter_groups(self) -> dict[str, int]:
    groups = {
        **{f"encoders.{name}": sum(p.numel() for p in module.parameters() if p.requires_grad)
           for name, module in self.encoders.items()},
        **{f"decoders.{name}": sum(p.numel() for p in module.parameters() if p.requires_grad)
           for name, module in self.decoders.items()},
    }
    groups["quantizer"] = 0 if self.quantizer is None else sum(
        p.numel() for p in self.quantizer.parameters() if p.requires_grad
    )
    return groups
```

- [ ] **Step 4: Run module tests**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py -q`

Expected: all freeze and observation tests pass.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/trl/modules/universal_token_modules.py gear_sonic/tests/test_root_conditioning.py
git commit -m "feat: freeze SONIC decoders individually"
```

### Task 3: Expand the Released Checkpoint With Exact Framewise Remapping

**Files:**
- Create: `gear_sonic/tools/expand_sonic_root_checkpoint.py`
- Modify: `gear_sonic/tests/test_root_conditioning.py`

**Interfaces:**
- Consumes: a checkpoint containing exactly one state dictionary under `actor_model_state_dict` or `policy_state_dict` and uniquely identified G1 encoder input / G1 kinematic decoder output tensors.
- Produces: an expanded checkpoint, `root_expansion.json`, and SHA-256 values; source checkpoint is never overwritten.

- [ ] **Step 1: Write synthetic bit-copy tests**

```python
def test_expand_encoder_weight_remaps_each_frame_and_zeroes_root_columns():
    old = torch.arange(4 * 640, dtype=torch.float32).reshape(4, 640)
    new = expand_framewise_input(old, frames=10, old_width=64, added_width=3)
    assert new.shape == (4, 670)
    for frame in range(10):
        torch.testing.assert_close(new[:, frame * 67:frame * 67 + 64], old[:, frame * 64:(frame + 1) * 64], rtol=0, atol=0)
        assert torch.count_nonzero(new[:, frame * 67 + 64:(frame + 1) * 67]) == 0

def test_expand_decoder_output_remaps_rows_bias_and_zeroes_root_outputs():
    weight = torch.arange(640 * 3, dtype=torch.float32).reshape(640, 3)
    bias = torch.arange(640, dtype=torch.float32)
    new_weight, new_bias = expand_framewise_output(weight, bias, 10, 64, 3)
    assert new_weight.shape == (670, 3)
    assert new_bias.shape == (670,)
    for frame in range(10):
        torch.testing.assert_close(new_weight[frame * 67:frame * 67 + 64], weight[frame * 64:(frame + 1) * 64], rtol=0, atol=0)
        assert torch.count_nonzero(new_weight[frame * 67 + 64:(frame + 1) * 67]) == 0
        assert torch.count_nonzero(new_bias[frame * 67 + 64:(frame + 1) * 67]) == 0
```

- [ ] **Step 2: Run and confirm the missing utility**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py -q`

Expected: import fails for `gear_sonic.tools.expand_sonic_root_checkpoint`.

- [ ] **Step 3: Implement framewise expansion helpers**

```python
def expand_framewise_input(weight, *, frames: int, old_width: int, added_width: int):
    if tuple(weight.shape[1:]) != (frames * old_width,):
        raise ValueError("G1 encoder first-layer shape changed")
    new_width = old_width + added_width
    output = weight.new_zeros((weight.shape[0], frames * new_width))
    for frame in range(frames):
        output[:, frame * new_width:frame * new_width + old_width] = weight[:, frame * old_width:(frame + 1) * old_width]
    return output

def expand_framewise_output(weight, bias, frames: int, old_width: int, added_width: int):
    if weight.shape[0] != frames * old_width or bias.shape != (frames * old_width,):
        raise ValueError("G1 kinematic decoder output shape changed")
    new_width = old_width + added_width
    out_weight = weight.new_zeros((frames * new_width, weight.shape[1]))
    out_bias = bias.new_zeros((frames * new_width,))
    for frame in range(frames):
        old = slice(frame * old_width, (frame + 1) * old_width)
        new = slice(frame * new_width, frame * new_width + old_width)
        out_weight[new] = weight[old]
        out_bias[new] = bias[old]
    return out_weight, out_bias
```

The CLI uses exact suffixes `actor_module.encoders.g1.module.0.weight`, `actor_module.decoders.g1_kin.module.8.weight`, and `.bias`; it fails unless each suffix matches exactly one key with shapes `[2048,640]`, `[640,512]`, and `[640]`. It deep-copies the checkpoint, writes only the destination path, records source/destination tensor hashes and checkpoint hashes, and refuses an existing destination unless `--replace` is passed.

- [ ] **Step 4: Run synthetic tests and migrate the official checkpoint**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py -q`

Expected: all tests pass.

Run: `python download_from_hf.py --training --no-smpl`

Run: `python gear_sonic/tools/expand_sonic_root_checkpoint.py --source sonic_release/last.pt --output sonic_root_stage_a/initial.pt --report sonic_root_stage_a/root_expansion.json`

Expected: command exits zero and reports `encoder_input: 640 -> 670`, `g1_kin_output: 640 -> 670`, and `new_values_nonzero: 0`.

- [ ] **Step 5: Commit the utility and tests, not downloaded artifacts**

```bash
git add gear_sonic/tools/expand_sonic_root_checkpoint.py gear_sonic/tests/test_root_conditioning.py
git commit -m "feat: expand SONIC checkpoint for root conditioning"
```

### Task 4: Configure Root Reconstruction and the Stage-A Frozen Model

**Files:**
- Modify: `gear_sonic/trl/losses/token_losses.py`
- Create: `gear_sonic/config/actor_critic/encoders/g1_root_mf_mlp.yaml`
- Create: `gear_sonic/config/actor_critic/decoders/g1_root_kin_mf_mlp.yaml`
- Create: `gear_sonic/config/actor_critic/universal_token/all_mlp_v1_root_stage_a.yaml`
- Create: `gear_sonic/config/aux_losses/terms/g1_root_recon.yaml`
- Create: `gear_sonic/config/aux_losses/universal_token/g1_root_stage_a.yaml`
- Create: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_root_stage_a.yaml`
- Modify: `gear_sonic/tests/test_root_conditioning.py`

**Interfaces:**
- G1 encoder inputs: `command_multi_future_nonflat`, `motion_anchor_ori_b_mf_nonflat`, `motion_root_position_refheading_mf_nonflat`.
- G1 kinematic decoder outputs: the same three terms in the same order.
- Stage-A trainable groups: `encoders.g1` and `decoders.g1_kin` only.

- [ ] **Step 1: Add failing loss/config tests**

```python
def test_root_reconstruction_loss_includes_all_three_terms():
    target = make_tokenizer_obs(batch=2)
    prediction = {key: value.clone() for key, value in target.items()}
    prediction["motion_root_position_refheading_mf_nonflat"] += 1.0
    loss = G1ReconLoss()(dict(
        tokenizer_obs=target,
        decoded_outputs={"g1_kin": prediction},
        decoders_cfg={"g1_kin": {"outputs": list(target)}},
    ))
    assert loss > 0

def test_fk_converter_ignores_root_channel_but_accepts_three_key_decoder():
    output = make_three_key_decoder_output()
    position, rotation = decoder_output_to_egocentric_transforms(output, root_decoder_cfg(), fake_humanoid())
    assert position.shape[-1] == 3
    assert rotation.shape[-1] == 6
```

- [ ] **Step 2: Extend the legacy G1 decoder-key branch**

In every exact-key branch that currently accepts the two released G1 outputs, accept the three-key set and continue using joint positions plus orientation for FK. Root is supervised separately by `G1ReconLossAligned`.

```python
g1_qpos_keys = {"command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"}
if set(output_keys) in (
    g1_qpos_keys,
    g1_qpos_keys | {"motion_root_position_refheading_mf_nonflat"},
):
    # existing released qpos/orientation conversion remains unchanged
```

- [ ] **Step 3: Add encoder, decoder, and loss configs**

The encoder config copies `g1_mf_mlp.yaml` and uses the three inputs. The kinematic decoder copies `g1_kin_mf_mlp.yaml` and uses the three outputs. `g1_root_recon.yaml` instantiates the existing `G1ReconLoss` with MSE; that loss reads the expanded `g1_kin.outputs` list from `decoders_cfg`, so no extra loss-input contract is introduced. The Stage-A auxiliary config includes only `g1_root_recon` with coefficient `1.0`.

The universal-token config instantiates every released encoder/decoder for export compatibility, sets `freeze: true` on teleop, SMPL, and `g1_dyn`, sets `freeze_quantizer: true`, and leaves G1 plus `g1_kin` trainable. Its G1 sample probability is `1.0`; teleop and SMPL are `0.0`; `optimize_encoders_ratio_for_CHIP: true` prevents automatic secondary activation.

- [ ] **Step 4: Add the Stage-A experiment config**

Copy `sonic_release.yaml`, override tokenizer with `unitoken_all_noz_root`, actor-critic with `all_mlp_v1_root_stage_a`, and auxiliary losses with `g1_root_stage_a`. Keep `checkpoint: null` in the file so the caller must pass the expanded checkpoint explicitly; this prevents an accidental load of the unexpanded release checkpoint. Set:

```yaml
checkpoint: null
manager_env:
  commands:
    motion:
      num_future_frames: 10
      dt_future_ref_frames: 0.1
```

At initialization, assert the trainable audit has positive counts only for `encoders.g1` and `decoders.g1_kin`; raise `RuntimeError` before rollout otherwise.

- [ ] **Step 5: Run tests and a one-environment construction smoke test**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py -q`

Expected: all tests pass.

Run: `WANDB_MODE=disabled python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_root_stage_a num_envs=1 headless=True ++algo.config.max_steps=1`

Expected: model constructs with G1 input 670, G1 kinematic output 670, prints the exact trainable audit, and completes one update without a dimension or checkpoint error.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic/trl/losses/token_losses.py gear_sonic/config/actor_critic/encoders/g1_root_mf_mlp.yaml gear_sonic/config/actor_critic/decoders/g1_root_kin_mf_mlp.yaml gear_sonic/config/actor_critic/universal_token/all_mlp_v1_root_stage_a.yaml gear_sonic/config/aux_losses/terms/g1_root_recon.yaml gear_sonic/config/aux_losses/universal_token/g1_root_stage_a.yaml gear_sonic/config/exp/manager/universal_token/all_modes/sonic_root_stage_a.yaml gear_sonic/tests/test_root_conditioning.py
git commit -m "feat: configure frozen SONIC root Stage A"
```

### Task 5: Build a Hashed Mixed Motion Dataset

**Files:**
- Create: `gear_sonic/tools/build_root_conditioning_dataset.py`
- Create: `gear_sonic/tests/test_root_dataset.py`

**Interfaces:**
- Consumes: `--sonic-root` containing official robot-motion PKLs and repeated `--mm-bundle` Motion Matching run bundles containing canonical CSVs plus `mm_root_diagnostic.csv`.
- Produces: an output directory of read-only official-motion symlinks and normalized physical-pelvis MM reference directories plus `root_conditioning_manifest.json`; duplicate names receive a deterministic 12-character source-hash suffix.

- [ ] **Step 1: Write failing identity and rejection tests**

```python
def test_manifest_pins_every_source_and_marks_origin(tmp_path):
    sonic, mm_bundle = make_dataset_sources(tmp_path)
    manifest = build_dataset(sonic, [mm_bundle], tmp_path / "mixed")
    assert {entry["origin"] for entry in manifest["motions"]} == {"sonic", "motion-matching"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["motions"])
    assert manifest["motion_matching_fraction"] > 0.0

def test_rejects_mm_reference_with_zero_or_missing_body_position(tmp_path):
    mm = make_mm_bundle(tmp_path, physical_pelvis=np.zeros((21, 3), np.float32))
    with pytest.raises(ValueError, match="physical pelvis"):
        build_dataset(make_sonic_source(tmp_path), [mm], tmp_path / "mixed")
```

- [ ] **Step 2: Implement deterministic materialization**

For official data, accept only regular `.pkl` files, hash each file, and create relative read-only symlinks. For each MM bundle, validate the canonical joint/quaternion CSVs and `mm_root_diagnostic.csv`, extract columns `physical_pelvis_x/y/z`, reject non-finite or all-zero physical pelvis data, and materialize a new reference directory whose `body_pos.csv` contains those exact parsed float32 values. Hash `joint_pos.csv`, `joint_vel.csv`, `body_quat.csv`, generated `body_pos.csv`, and `metadata.txt` in sorted order. Write canonical sorted JSON with schema `sonic-root-conditioning-dataset/v1`, source paths, hashes, row counts, origin, and aggregate fractions.

- [ ] **Step 3: Run tests**

Run: `python -m pytest gear_sonic/tests/test_root_dataset.py -q`

Expected: all tests pass.

- [ ] **Step 4: Materialize the initial dataset**

Run: `python download_from_hf.py --sample`

Run: `python gear_sonic/tools/build_root_conditioning_dataset.py --sonic-root sample_data/robot_filtered --mm-bundle /home/ubuntu/mm-sonic-curb-low-diagnostic-20260722/manual-sonic/manual-20260722T162024737809Z-3083126 --output sonic_root_stage_a/motions --manifest sonic_root_stage_a/root_conditioning_manifest.json`

Expected: command exits zero, records both origins, and reports at least one Motion Matching clip with nonzero pelvis-height variance.

- [ ] **Step 5: Commit the tool and tests**

```bash
git add gear_sonic/tools/build_root_conditioning_dataset.py gear_sonic/tests/test_root_dataset.py
git commit -m "feat: build hashed root conditioning dataset"
```

### Task 6: Train, Export, and Prove Initial Parity Plus Learned Sensitivity

**Files:**
- Create: `gear_sonic/tools/verify_root_onnx.py`
- Create: `docs/root_conditioned_stage_a.md`
- Modify: `gear_sonic/tests/test_root_conditioning.py`

**Interfaces:**
- Consumes: released ONNX, expanded initial checkpoint, trained checkpoint, exported root encoder ONNX, and root-conditioned observation config.
- Produces: `root_stage_a_verification.json` with tensor-copy, initial token/action parity, learned sensitivity, flat probes, model hashes, and config hash.

- [ ] **Step 1: Write verifier tests with tiny ONNX fixtures**

```python
def test_zero_root_columns_preserve_initial_tokens_and_actions(tmp_path):
    released, expanded = make_linear_onnx_pair(tmp_path, append_zero_columns=30)
    report = compare_initial_models(released, expanded, samples=20, seed=20260722)
    assert report["max_token_abs_delta"] == 0.0

def test_sensitivity_requires_nonzero_response_to_root_only_change(tmp_path):
    model = make_root_sensitive_onnx(tmp_path)
    report = root_sensitivity(model, samples=20, root_offsets=(1762, 1792), seed=20260722)
    assert report["max_token_abs_delta"] > 1e-5
```

- [ ] **Step 2: Implement verification gates**

`verify_root_onnx.py` loads models with ONNX Runtime, checks exact input/output names and shapes, hashes every file, generates 20 seeded valid G1 samples, and runs:

```python
INITIAL_TOKEN_ATOL = 0.0
INITIAL_ACTION_ATOL = 1.0e-6
LEARNED_ROOT_TOKEN_MIN = 1.0e-5
FLAT_ACTION_ATOL = 5.0e-3
ROOT_RANGE = (1762, 1792)
```

Initial expanded models must match released tokens exactly and released decoder actions within `1e-6` when root columns are zero. The trained model must exceed `1e-5` token change under root-only perturbations while joint/orientation inputs remain bit-identical. Ten flat-root probes at constant Z and zero delta XY must keep action deltas below `5e-3` relative to the released model.

- [ ] **Step 3: Run unit tests**

Run: `python -m pytest gear_sonic/tests/test_root_conditioning.py gear_sonic/tests/test_root_dataset.py -q`

Expected: all tests pass.

- [ ] **Step 4: Run Stage-A training**

Run: `WANDB_MODE=offline python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_root_stage_a num_envs=4096 headless=True ++manager_env.commands.motion.motion_lib_cfg.motion_file=sonic_root_stage_a/motions +checkpoint=sonic_root_stage_a/initial.pt`

Expected: training logs show only G1 encoder and G1 kinematic decoder gradients, root reconstruction loss decreases, and a checkpoint is saved. Stop at the first checkpoint whose held-out root reconstruction MSE is at most 50% of its initial value; record the exact step rather than selecting by terrain outcome.

- [ ] **Step 5: Export ONNX from the selected checkpoint**

Run: `python gear_sonic/eval_agent_trl.py +exp=manager/universal_token/all_modes/sonic_root_stage_a +checkpoint=sonic_root_stage_a/selected.pt +headless=True ++num_envs=1 +export_onnx_only=true`

Expected: encoder ONNX input shape is `[1,1792]`, token output shape is `[1,64]`, and decoder input/output shapes match the released control decoder.

- [ ] **Step 6: Run all parity and sensitivity gates**

Run: `python gear_sonic/tools/verify_root_onnx.py --released-encoder gear_sonic_deploy/policy/release/model_encoder.onnx --released-decoder gear_sonic_deploy/policy/release/model_decoder.onnx --initial-checkpoint sonic_root_stage_a/initial.pt --trained-encoder sonic_root_stage_a/exported/model_encoder.onnx --trained-decoder sonic_root_stage_a/exported/model_decoder.onnx --observation-config gear_sonic_deploy/policy/release/observation_config_root_conditioned.yaml --output sonic_root_stage_a/root_stage_a_verification.json`

Expected: every initial parity, learned sensitivity, dimension, range, hash, and flat-probe gate is `pass`.

- [ ] **Step 7: Document and commit reproducible evidence**

`docs/root_conditioned_stage_a.md` records GEAR SHA, CUDA/GPU identity, source and expanded checkpoint hashes, dataset manifest hash, seed, exact command, selected step, ONNX/config hashes, and the complete verification JSON path. Do not commit checkpoints or generated datasets.

```bash
git add gear_sonic/tools/verify_root_onnx.py gear_sonic/tests/test_root_conditioning.py docs/root_conditioned_stage_a.md
git commit -m "test: qualify SONIC root-conditioned Stage A"
```
