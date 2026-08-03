# G1 Terrain-Conformal Swing Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify a deterministic offline optimizer for terrain-conformal G1 mid-foot swing trajectories.

**Architecture:** A pure Torch module validates one contiguous swing, evaluates batched mid-foot/toe/heel terrain costs, and performs seeded MPPI updates while fixing liftoff and touchdown exactly. A separate resource script applies it to one saved GRAIL route phase and emits numerical and visual evidence; the live matcher remains unchanged.

**Tech Stack:** Python 3, PyTorch, unittest, authenticated GRAIL height grids, MuJoCo/Pillow for generated audit artifacts.

## Global Constraints

- Preserve source timing, support sequence, liftoff, and touchdown.
- Do not modify the live matcher, Sonic, tracking, or the dirty landing-bridge/FK files.
- Fail closed on invalid terrain samples and never return a higher-cost path.
- Use float32 tensors on the caller-owned device and an explicit deterministic seed.

---

### Task 1: Typed swing objective

**Files:**
- Create: `sonic/python/mm_sonic/torch_terrain_conformal_swing.py`
- Create: `tests/python/test_sonic_torch_terrain_conformal_swing.py`

**Interfaces:**
- Produces: `TerrainConformalSwingConfig`, `TerrainConformalSwingCost`, and `terrain_conformal_swing_cost(paths, raw_path, sample_surface, toe_offset_xy, heel_offset_xy, config)`.

- [ ] **Step 1: Write failing contract and objective tests**

Add tests that require immutable finite configuration values, one batched
terrain callback invocation, exact zero clearance cost on a high flat swing,
positive clearance/edge costs when toe or heel crosses a 0.18 m riser, and a
positive second-difference cost for a kinked path.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B -m unittest \
  tests.python.test_sonic_torch_terrain_conformal_swing -v
```

Expected: import failure because the new module does not exist.

- [ ] **Step 3: Implement the minimal typed objective**

Validate `(batch, frames, 3)` candidate paths, one `(frames, 3)` raw path,
float32/device ownership, two `(2,)` offsets, and callback output shape. Sample
mid/toe/heel positions in one callback call. Return scalar tensors per batch
for reference, smoothness, clearance, edge, endpoint, and total cost.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused test command and commit only the new module and test with:

```bash
git commit -m "feat: score terrain-conformal swing paths"
```

### Task 2: Deterministic MPPI optimizer

**Files:**
- Modify: `sonic/python/mm_sonic/torch_terrain_conformal_swing.py`
- Modify: `tests/python/test_sonic_torch_terrain_conformal_swing.py`

**Interfaces:**
- Consumes: the Task 1 objective.
- Produces: `TerrainConformalSwingResult` and `optimize_terrain_conformal_swing(raw_path, swing_mask, sample_surface, toe_offset_xy, heel_offset_xy, config, seed)`.

- [ ] **Step 1: Write failing optimization tests**

Require bitwise-fixed non-swing samples and endpoints, byte-identical repeated
results for one seed, strict raw-cost improvement on a synthetic stair-riser
path, no regression on a flat already-clear path, and `ContractError` for a
disjoint swing or invalid callback output.

- [ ] **Step 2: Verify RED**

Run the focused unittest command. Expected: failure because the optimizer and
result type are absent.

- [ ] **Step 3: Implement the minimal optimizer**

Create a generator on the input device, sample `sample_count` perturbations,
smooth them temporally with a fixed normalized five-tap kernel, zero the
endpoint perturbations, clamp XYZ perturbations to configured bounds, evaluate
the batch, and apply a softmin-weighted update for `iteration_count` steps.
Restore all non-swing and endpoint values after every update. Return the raw
path unless final total cost is finite and strictly smaller.

- [ ] **Step 4: Verify GREEN, repeat determinism, and commit**

Run the focused tests twice and compare the test-reported deterministic tensor
digest. Commit with:

```bash
git commit -m "feat: optimize terrain-conformal swing paths"
```

### Task 3: Real GRAIL phase evidence

**Files:**
- Create: `resources/run_g1_torch_terrain_conformal_swing.py`
- Create: `tests/python/test_run_g1_torch_terrain_conformal_swing.py`

**Interfaces:**
- Consumes: a saved horizon-route `arrays.npz`, the terrain dataset/config, one foot, and a source-frame interval.
- Produces: canonical JSON metrics plus a generated contact-path sheet under `build/`.

- [ ] **Step 1: Write failing CLI tests**

Require explicit dataset/config/route arrays/output/device/foot/frame bounds,
reject output overwrite, and prove importing the module has no MuJoCo window or
Sonic dependency.

- [ ] **Step 2: Verify RED**

Run the CLI test module. Expected: import failure because the runner is absent.

- [ ] **Step 3: Implement the offline runner**

Load exact saved qpos, compute ankle-body transforms through the existing clean
FK surface, construct mid/toe/heel paths, optimize the chosen swing, and write
raw/optimized costs, clearance violation count, endpoint error, acceleration,
seed, and deterministic SHA-256. Save the paths needed for a separate MuJoCo
audit without starting a viewer.

- [ ] **Step 4: Verify and run the real ablation**

Run focused tests, then run the CLI on the first visibly failing swing from
`build/emitted-preview-v43-turn-warp-previewed-wide`. Require exact endpoint
preservation and fewer toe/heel clearance violations before rendering and
inspecting the result.

- [ ] **Step 5: Commit or reject**

If the real path qualifies, commit the runner/tests and record its artifact
identity. If it does not, document the measured failure and leave the live
matcher unchanged.
