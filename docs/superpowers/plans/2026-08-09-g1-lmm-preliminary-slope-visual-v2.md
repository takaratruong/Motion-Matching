# G1 LMM Preliminary Slope Visual V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one immutable, explicitly post-hoc learned exact-route slope visual whose only V1 gate difference is a 2.000 mm maximum non-root local translation.

**Architecture:** Extend only the isolated preliminary slope module and its focused tests. Add a V2 policy that re-evaluates completed decompressor metrics without mutating the production 1.000 mm gate, then reuse the projector-free stepper, full-route gate, authenticated loader, and MuJoCo viewer under a separate schema/output/claim.

**Tech Stack:** Python 3.12, NumPy 2.4, deterministic PyTorch 2.4 CUDA, MuJoCo, unittest, existing `resources.g1_lmm.training` stages.

## Global Constraints

- Preserve V1 rejection and production 1.000 mm gate byte-for-byte.
- V2 local translation maximum is exactly 0.002 m; every other threshold is unchanged.
- Keep seeds 1234/1235/1236, 1000/100000/100000 steps, batch 32, LR 0.001, and no projector.
- Use a distinct immutable V2 directory, persistent `O_EXCL` claim, schema, label, and receipt.
- One GPU3 run; first failure rejects without binaries and no retry or further tuning.

---

### Task 1: Add the isolated V2 policy and contract

**Files:**
- Modify: `sonic/python/mm_sonic/preliminary_learned_slope.py`
- Test: `tests/python/test_preliminary_learned_slope.py`

**Interfaces:**
- Consumes: V1 config, decompressor gate dictionary, projector-free training/viewer paths.
- Produces: `visual_v2_decompressor_gate(gate)`, fixed V2 output/claim/label, and `train-v2`, `smoke-v2`, `view-v2` commands.

- [ ] **Step 1: Write and run focused RED tests.** Prove V1 rejects 1.492458 mm, V2 accepts it at exactly 2.000 mm, V2 rejects 2.000001 mm or any other failed metric, paths/claims differ, projector paths remain absent, and loader/view requires the V2 receipt.

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_preliminary_learned_slope
```

- [ ] **Step 2: Implement the minimal copy-based V2 reducer.** Recompute `fitted.accepted` with the existing joint/FK/sole/contact/finite thresholds and `local_translation_max_error_m <= 0.002`; set outer acceptance only with no withheld rows. Use it only in V2 after the unchanged training stage returns. Bind a distinct schema, exact post-hoc label, `model-v2-visual`, and separate claim. Never edit `resources/g1_lmm/training.py`.

- [ ] **Step 3: Run GREEN and static checks.**

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_preliminary_learned_slope
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile sonic/python/mm_sonic/preliminary_learned_slope.py tests/python/test_preliminary_learned_slope.py
git diff --check
```

- [ ] **Step 4: Commit only the two focused files.**

```bash
git add sonic/python/mm_sonic/preliminary_learned_slope.py tests/python/test_preliminary_learned_slope.py
git commit -m "feat: add preliminary slope visual v2"
```

### Task 2: Execute the sole V2 fit

**Files:**
- Runtime output only: `sonic/runs/g1-lmm-authored-slope-preliminary/model-v2-visual/`

**Interfaces:**
- Consumes: committed `train-v2` command and authenticated 595-row source.
- Produces: one immutable rejection receipt, or latent/decompressor/stepper plus receipt and manifest.

- [ ] **Step 1: Preflight.** Require idle physical GPU3 UUID `GPU-87fb0777-4169-d4e9-12cc-382bdf7c0730`, absent V2 output/claim, and unchanged V1 receipt SHA `2e4e77a279812bbbee952a752447df16c67a6ba56f9ebb82fc342b54c6c83765`.

- [ ] **Step 2: Launch exactly once.**

```bash
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:resources:sonic/python /home/ubuntu/projects/foundation_stereo/.venv/bin/python -m mm_sonic.preliminary_learned_slope train-v2 --output sonic/runs/g1-lmm-authored-slope-preliminary/model-v2-visual
```

- [ ] **Step 3: Verify the terminal receipt.** Require exact bindings, no projector, V2 decompressor acceptance, stepper acceptance, full 595-row rollout acceptance, and exact artifact tree. Any rejection ends the visual experiment with no retry.

### Task 3: Smoke and open the learned viewer

**Files:**
- No further source changes expected.

- [ ] **Step 1: Run fresh-process smoke.**

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m mm_sonic.preliminary_learned_slope smoke-v2 --model sonic/runs/g1-lmm-authored-slope-preliminary/model-v2-visual
```

- [ ] **Step 2: Launch MuJoCo.**

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m mm_sonic.preliminary_learned_slope view-v2 --model sonic/runs/g1-lmm-authored-slope-preliminary/model-v2-visual
```

Verify the visible post-hoc/nonaccepted label, `W` learned advance, release bitwise pause, and `X`/Escape exit.
