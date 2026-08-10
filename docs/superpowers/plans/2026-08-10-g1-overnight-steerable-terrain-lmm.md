# G1 Overnight Steerable Terrain LMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a terrain-aware learned G1 pose generator over the broad GRAIL/Takara walking bank and expose it through a steerable hybrid motion matcher in MuJoCo.

**Architecture:** Load the existing 3,970,932-row 25 Hz canonical bank, rebuild the Orange Duck 31-D query features, and train streaming autoencoders without copying the complete corpus to GPU. Runtime performs exact cKDTree terrain/command search and uses the selected row's learned latent plus decoder for every displayed pose. A learned short-horizon stepper is optional; range-safe successors are the required fallback.

**Tech Stack:** Python 3.11, NumPy, SciPy cKDTree, PyTorch CUDA, MuJoCo, pynput/evdev, unittest.

## Global Constraints

- Preserve all current flat-v3, preliminary-slope, Task2 and Task3 behavior; add new modules rather than weakening their schemas or gates.
- Canonical source is `/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate`: 3,970,932 rows, 15,815 ranges, 25 Hz, 31 bones, two contacts and four terrain channels.
- Use matching horizons `(8, 17, 25)` and `dt=0.04` exactly for this bank.
- Source-held-out evaluation is by complete ranges; no temporal derivative, successor or feature horizon crosses a range boundary.
- Runtime label is `HYBRID TERRAIN LMM POC (EXACT SEARCH + LEARNED GENERATOR; SUPPORTED TERRAIN ONLY)`.
- Learned projector is absent. Exact search is a deliberate part of this PoC.
- The PFNN supplement's 17 released training clips are fit-only; held-out metrics come from the primary Takara/GRAIL split and are not a PFNN-generalization claim.
- Every feature or bugfix follows RED -> GREEN -> focused regression.
- Do not stage the existing concurrent modifications to validator/oracle files.

---

### Task 1: Broad corpus adapter and cache

**Files:**
- Create: `sonic/python/mm_sonic/hybrid_terrain_lmm_data.py`
- Create: `tests/python/test_hybrid_terrain_lmm_data.py`

**Interfaces:**
- Produces `HybridCorpus`, `load_hybrid_corpus(root: Path)`, `build_hybrid_cache(root: Path, output: Path)`, and `load_hybrid_cache(output: Path)`.
- `HybridCorpus` contains the canonical `ArtifactSet`, `FeatureSet`, range family IDs, deterministic train/evaluation masks, range lookup and immutable manifest receipt.

- [ ] **Step 1: Write failing synthetic range tests**

```python
def test_split_is_range_complete_and_features_do_not_cross_ranges():
    corpus = build_synthetic_hybrid_corpus()
    assert not np.any(corpus.train_mask & corpus.evaluation_mask)
    assert all(np.all(corpus.evaluation_mask[s:e]) or not np.any(corpus.evaluation_mask[s:e])
               for s, e in zip(corpus.artifacts.range_starts, corpus.artifacts.range_stops))
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_hybrid_terrain_lmm_data`

Expected: import failure because the adapter does not exist.

- [ ] **Step 3: Implement authenticated loading and feature construction**

Use `read_holden_database`, `read_terrain_sidecar`, `read_support_sidecar` and `build_matching_features(artifacts, 25.0, (8,17,25))`. Verify database and sidecar SHA-256 values from `manifest.json`, exactly 3,970,932 rows, skeleton parents, 15,815 ranges and finite active terrain channels. Derive family from `motion_banks` range lists. Hold out every tenth sorted GRAIL range within each family; hold out the final non-overlapping 10% of Takara.

- [ ] **Step 4: Implement atomic cache**

Write `features.npy`, offsets/scales, family/range arrays, masks and canonical JSON manifest under a temporary sibling, fsync, then rename. Reopen every member before publication. Cache references the immutable external `database.bin` and terrain/support files by path, size and SHA rather than duplicating 6.5 GB.

- [ ] **Step 5: Run GREEN and real preflight**

Run the focused test, then:

```bash
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m mm_sonic.hybrid_terrain_lmm_data build \
  --source /home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate \
  --output sonic/runs/g1-hybrid-terrain-lmm/corpus-v1
```

Expected: accepted receipt with 3,970,932 rows, all four terrain families and nonempty disjoint train/evaluation masks.

### Task 2: Streaming learned generator training and evaluation

**Files:**
- Create: `sonic/python/mm_sonic/hybrid_terrain_lmm_training.py`
- Create: `tests/python/test_hybrid_terrain_lmm_training.py`

**Interfaces:**
- Consumes `HybridCorpus`/cache.
- Produces `HybridModelConfig`, `train_hybrid_generator`, `evaluate_hybrid_generator`, `load_hybrid_generator`, `latent.npy`, `model.pt`, `training.json`, `evaluation.json`, and `manifest.json`.

- [ ] **Step 1: Write RED tests for streaming batches and held-out gates**

```python
def test_batches_are_family_balanced_and_never_use_evaluation_rows():
    rows = sample_training_rows(corpus, batch_size=16, generator=np.random.default_rng(7))
    assert np.all(corpus.train_mask[rows])
    assert set(corpus.family_ids[rows]) == set(corpus.family_names)
```

Also test immutable output, deterministic tiny training, checkpoint reload equality, practical gate calculations, nonfinite rejection and exact model/corpus hash binding.

- [ ] **Step 2: Run RED**

Run the focused unittest; expect missing-module failure.

- [ ] **Step 3: Implement the model and streaming optimizer**

Implement deterministic compressor `908 -> width -> width -> width -> latent` and decompressor `31+latent -> width -> width -> 458`. Batch only selected CPU rows onto CUDA. Optimize normalized SmoothL1 pose/velocity output, BCE contact logits and `1e-4 * mean(latent**2)`. Ensure one deterministic shuffled coverage pass over every training row before random family-balanced steps.

- [ ] **Step 4: Implement chunked full evaluation**

Encode/decode train and source-held-out rows in chunks. Compute joint geodesic MAE/p95, local-position p95, FK p95, support-foot p95 and bilateral contact F1. Use the practical gates in the design. Save all row latents only after train and evaluation outputs are finite.

Implementation variance recorded after the full run: the canonical learned
metrics passed, but the original all-row native-limit gate did not.  The
exhaustive diagnostic found 6,127 learned row-aligned violations.  Do not mark
the model artifact itself green for that gate.  The accepted PoC boundary is
the authenticated scripted MuJoCo runtime, which retries exact candidates and
requires every committed pose to pass native limits with zero clamp/fallback.

- [ ] **Step 5: Launch bounded variants on separate GPUs**

Run latent32/width512 on GPU5 and latent64/width512 on GPU6 with unique output paths. Start with 20,000 post-coverage steps. If both are red, run latent64/width1024 on GPU7. Select by held-out metrics; refit the chosen architecture on all rows for 30,000 steps while retaining the source-held-out selection receipt.

### Task 3: Exact-search hybrid runtime core

**Files:**
- Create: `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py`
- Create: `tests/python/test_hybrid_terrain_lmm_runtime.py`

**Interfaces:**
- Consumes cache and learned generator.
- Produces `TerrainAuthority`, `HybridMatcher`, `HybridRuntimeState`, `CommandState`, and `run_headless_smoke`.

- [ ] **Step 1: Write RED tests**

Test cKDTree/brute-force parity, stable tie resolution, range-safe successor behavior, terrain channel overwrite, nonmonotonic range switches, support-relative root placement, atomic hold on invalid decode and keyboard/gamepad command normalization.

- [ ] **Step 2: Implement exact candidate search**

Build the tree over range-safe normalized rows. Search 32 candidates, expand to 64/128 when required, rescore in float64, and tie-break by lowest row. Search on command/terrain events and every 100 ms. Query pose channels come from the accepted row; trajectory/facing from the filtered command; terrain channels from the live `TerrainAuthority`.

Search scoring uses normalized feature distance plus the frozen same-range
transition penalty.  Range-safe successor preference is the deterministic
next-row advancement between search events, not a third score term.

- [ ] **Step 3: Implement learned decode transaction**

Reseed from selected features+latent, overwrite live terrain, decode, convert with the existing exact Holden-to-native helper, place root from `live_support + selected_clearance`, validate finite/joint/step bounds and atomically commit. On failure, use the selected canonical source pose visibly and increment `fallback_count`.

- [ ] **Step 4: Headless GREEN**

Run at least 1,000 scripted frames covering idle/W/A/D/stop and two terrain classes. Require no NaN or range crossing, two source ranges, nonzero learned decode count, nonzero terrain variance and sampled tree/brute parity.

### Task 4: MuJoCo viewer and controls

**Files:**
- Create: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py`
- Create: `tests/python/test_hybrid_terrain_lmm_viewer.py`

**Interfaces:**
- Consumes Task 3 runtime.
- Produces one CLI with `smoke` and `view` subcommands.

- [ ] **Step 1: Write RED tests for input and overlay**

Require W/S speed, A/D steering, Space stop, R reset, evdev axes with 0.2 deadzone, neutral fallback on disconnect, and overlay fields for family/range/row/search distance/terrain/decode/fallback/support.

- [ ] **Step 2: Implement terrain/query/render authority**

Load one existing scene heightfield/mesh through a single adapter used by both search queries and MuJoCo rendering. Add `--scene` and reset without rebuilding the model generator.

- [ ] **Step 3: Implement interactive loop**

Use pynput keyboard and optional evdev joystick. Render at 60 Hz while stepping the 25 Hz matcher on a fixed accumulator. Do not mutate runtime state on pause/reset failures.

- [ ] **Step 4: Verify headless and live**

Run `smoke --frames 1000`, then launch `view`, drive W/A/D over the scene, capture one screenshot and write a runtime receipt.

### Task 5: Integration review and morning handoff

**Files:**
- Create: `docs/superpowers/results/2026-08-10-g1-hybrid-terrain-lmm.md`

- [ ] **Step 1: Run focused suites and static checks**

Run all four new unittest modules, `py_compile`, Ruff, and `git diff --check`.

- [ ] **Step 2: Independently verify artifacts**

Recompute corpus/model hashes, reload in a fresh process, run the 1,000-frame headless command, and confirm the live viewer loads the same model.

- [ ] **Step 3: Record honest coverage**

Report included/excluded GRAIL counts, PFNN bridge status, train/evaluation metrics, selected architecture, runtime support/fallback counts, screenshot and exact launch commands. Never call unsupported terrain generalization green.

- [ ] **Step 4: Review and commit only scoped files**

Request code review, fix Critical/Important findings, re-run affected tests, then commit without staging the concurrent validator/oracle edits.
