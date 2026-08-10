# Task 2 Report: Build and atomically publish the two-range terrain bundle

Status: DONE

Logical base commit: `738d3e2` (`fix(data): bind authored slope exterior policy`)

Task commit: this commit (`feat(data): publish authored-slope LMM bundle`)

## Scope delivered

- Added exact half-open `HoldenClip` slicing and per-source 60 Hz dynamics so
  the flat and authored-slope clips are differentiated, contact-filtered, and
  future-clamped independently before their ranges are joined.
- Built the immutable 851-row partition `[0,256), [256,851)`. Task 2 consumes
  Task 1's validated 598-row provisional source only through `source[3:]`,
  publishing exactly 595 authored-slope rows with no operation across row 256.
- Jointly normalized the 31-dimensional LMM features. Terrain dimensions
  27--30 are active and each has scale `0.040610138326883316`.
- Added the mandatory, mutually exclusive `--authored-slope-terrain` CLI mode
  and exact authentication of flat data, four GRAIL inputs, and the canonical
  G1 XML before payload decoding. Flat-only and general modes reject terrain
  options before loading inputs.
- Added the exact `g1-lmm-terrain-data/v1` manifest, validation document,
  source receipt/ranges, support and terrain sidecars, and the single
  `authored-slope` scene pack.
- Added a dedicated hybrid-tree publisher with pre-callback, post-callback,
  and under-parent-lock validation, exact-node rejection, fsync durability,
  atomic exchange, and rollback. Existing flat/general publishers were not
  weakened.
- Preserved Task 1's calibrated exterior height through rasterization. The
  published heightfield exterior is exactly
  `-0.012000000104308128` metres; legacy surfaces without that explicit
  property retain their existing zero exterior.

## TDD evidence

The implementation was developed through focused RED/GREEN checkpoints:

- Database partition tests were RED on missing slice/per-source-dynamics
  helpers, then GREEN after implementing the isolated two-range assembly.
- The real candidate integration test was RED on the missing LMM terrain
  assembly seam, then GREEN with 851 rows and exact frozen ranges.
- Exact mesh-membership accounting was RED on a missing surface-membership
  API, then GREEN with mesh/extension counts recomputed from live queries.
- Publisher tests were RED before the dedicated hybrid publisher existed,
  then GREEN for exact-tree, mutation, special-node, locked-parent, exchange,
  fsync, and rollback cases.
- CLI tests were RED before the terrain mode/required-path dispatch existed,
  then GREEN for mutual exclusion, mode isolation, authentication order, and
  publisher dispatch.
- The explicit calibrated-raster regression was RED with an observed `0.0`
  exterior instead of `-0.012000000104308128`, then GREEN after preserving
  the source surface's explicit exterior policy.

Final focused authentication-order regression:

```text
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_build_cli.BuildCliTests.test_authored_slope_authenticates_all_cli_inputs_before_payload_loading
```

Result: 1 test ran in 0.001 s, PASS.

Fresh prescribed GREEN command:

```text
PYTHONPATH=.:resources:sonic/python /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_database_builder tests.python.test_artifacts tests.python.test_build_cli tests.python.test_terrain
```

Result: 156 tests ran in 140.296 s, PASS.

## Published canary evidence

Atomic publication command completed with:

```text
BUILT g1-lmm-terrain-data/v1 frames=851 clips=2 scenes=1
```

Published path:
`sonic/runs/g1-lmm-terrain-60hz/data-v1-task2-candidate`

- `manifest.json`: `0aa8a3b9bc21335c4b159a3b314204dbab53696b3b23daefad77a5ca9ab1ead9`
- `database.bin`: `966d0a64cb5f748157fa5840781388d12ab604d6b4553fcff1b1a05d42256156`
- `features.bin`: `e3d887be8e577eb2c9268c8bcc28efce509cdf2597dac397d2cbe37b0f73a563`
- Exact database dimensions: 851 frames, 31 bones, 2 contacts.
- Exact feature dimensions: 851 rows, 31 features, 32 latent dimensions.
- Temporal partition: boundary row 256; derivatives and contacts are
  per-source; future features clamp at the range stop; cross-range operations
  are zero.
- Terrain query counts: feature mesh/extension `1922/1053`, support
  `1132/653`, route `374/221`.
- Continuity rejections: native `0`, local `0`, union `0`; the only dropped
  fragment is Task 1's frozen provisional prefix `[0,3)`.

## Exact published tree

```text
database.bin
features.bin
manifest.json
scenes/authored-slope/scene.json
scenes/authored-slope/terrain.bin
scenes/authored-slope/terrain.obj
scenes/authored-slope/walkability.bin
scenes/index.json
terrain_features.bin
terrain_support.bin
validation.json
```

## Verification and scope audit

- `py_compile` over all five modified production modules: PASS.
- `git diff --check`: PASS.
- Fresh atomic publication and exact 11-file tree audit: PASS.
- The first 256 flat database/feature rows remain bit-identical to the
  authenticated flat-v3 rebuild in the real integration test.
- Existing flat-v3 and general builder tests in the prescribed suite pass.
- The staged commit contains only the nine Task 2 implementation/test files
  plus this task-specific report.
- The independently owned learned-slope prototype files were not modified or
  staged.

## Handoff

Task 2's publisher performs its own complete validation three times. The
separate Task 3 standalone validator remains the next independent release
gate and is intentionally outside this commit.

Concerns: None.
