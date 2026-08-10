# Full Walking Task 5 Report

Status: implementation complete; real GPU execution deliberately not started.

## Scope

Implemented Plan Task 5:

- exact 60 Hz selection/validation, one-time test, and all-eligible-row refit;
- latent-32/width-512 default ABI (latent-64/width-512 remains the sole bounded
  alternate);
- deterministic shuffled complete coverage followed by precomputed hierarchical
  terrain/canonical-source/speed/turn/contact/transition sampling;
- immutable corpus/split/lane/model/test/refit provenance;
- streaming latent publication through the existing hybrid trainer; and
- legacy 25 Hz training compatibility.

The Task 4 corpus commit `b424020` was cherry-picked as local commit `954fdc2`
so the implementation could be checked against the final mmap accessor rather
than a speculative interface.

## TDD record

Initial focused RED:

```text
ModuleNotFoundError: No module named
'mm_sonic.full_walking_terrain_lmm_training'
```

The shared trainer RED separately proved that `HybridModelConfig(dt=1/60)` was
rejected by the legacy-only `dt=0.04` guard. Later focused REDs proved the
validation-red test-split guard, final Task 4 eligible-mask semantics,
no-replace model publication, full-walking model schema, explicit validation
receipt, hierarchical sampling receipt, and external one-time-test authority
were each absent before implementation.

## Implementation

Created:

- `sonic/python/mm_sonic/full_walking_terrain_lmm_training.py`
- `tests/python/test_full_walking_terrain_lmm_training.py`

Modified:

- `sonic/python/mm_sonic/hybrid_terrain_lmm_training.py`
- `tests/python/test_hybrid_terrain_lmm_training.py`

Key contracts:

- The wrapper rejects anything except `fps=60.0`, horizons `(20,40,60)`, the
  canonical 31-bone G1 hierarchy, 31-D matching features, 36-D terrain grid,
  908-D compressor input, 31-plus-latent decoder input, and 458-D target.
- Selection trains only on `eligible_mask & train_mask`; its held-out receipt is
  explicitly labeled `validation`, and a validation-red model cannot consume
  test rows.
- The test receipt is canonical JSON, exclusive/no-replace, bound to the exact
  selection manifest and validation receipt, and must resolve as an external
  hash authority when a refit model is loaded.
- Refit trains and normalizes the learned compressor/target only over the Task 4
  all-eligible fit mask, binds exact selection and test hashes, and retains the
  Task 4 train-only 31-D feature normalization through the corpus manifest.
- Full model manifests use `g1-full-walking-terrain-lmm-model/v1`; legacy models
  retain `g1-hybrid-terrain-lmm-manifest/v1` and exact 25 Hz behavior.
- Model publication now uses Linux atomic no-replace rename plus parent-directory
  fsync, closing the existing check/replace race.

## Verification evidence

Focused and legacy training:

```text
15 passed in 5.92s
```

Common contracts/corpus/combined/training regression set:

```text
40 passed in 22.49s
```

Final Task 4 mmap accessor smoke:

```text
{'rows': 160, 'mmap': 'memmap', 'eligible': 128,
 'train': 40, 'validation': 32, 'leaves': 12}
```

The production CUDA interpreter imports the module and renders the exact
`select|test|refit` CLI help with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. That
environment does not contain `pytest` (`No module named pytest`), so CPU
synthetic tests ran under the repository's `diffsim` interpreter, which has
PyTorch and pytest. No shared environment was modified.

## Execution blockers outside Task 5

- No real GPU training was launched, as explicitly required by the dispatch.
- Task 4 reported that the current Task 1 lane publication omits
  `ArtifactSet.terrain_features` and `terrain_support`; reopened real slope
  lanes therefore contain zero arrays despite nonzero `terrain_grid`. Real
  selection/refit must wait for the corrected lane ABI and republished corpus.
- The real inventory test was not repeated here because its metadata scan did
  not finish inside the tool's yielded command window; Task 1 already recorded
  that authority test. All Task 5-focused and non-real common regressions above
  completed normally.
