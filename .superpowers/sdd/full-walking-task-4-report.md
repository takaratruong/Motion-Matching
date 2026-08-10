# Full Walking Task 4 Report

Status: implementation complete; independent review pending

## Scope

Implemented Plan Task 4 in:

- `sonic/python/mm_sonic/full_walking_terrain_lmm_corpus.py`
- `tests/python/test_full_walking_terrain_lmm_corpus.py`

The implementation consumes the hardened Task 1 per-range source-ID ABI from
commit `f69efe86f69a61df4b9b3f72d3f44b2a0cf8bf39` and requires the corrected
lane-v2 terrain-sidecar ABI from commit `d21bee6`.

## Delivered behavior

- Independently authenticates every lane against the exact inventory and split
  ledger, then recomputes range-local 60 Hz velocities, angular velocities,
  contacts, quaternion continuity, terrain alignment, published source-map
  hashes, and exact source-local frame bounds.
- Packs ranges in stable
  `(family, canonical_source_id, mirror_of, range_id)` order into read-only
  NumPy mmap members while retaining exact lane/range shard bindings.
- Publishes articulated motion, 4-D terrain, 3-D support, 36-D terrain grid,
  raw and normalized 31-D features, exact row metadata, source interpolation
  maps, clean/usable eligibility, train/validation/test masks, source-balanced
  CSR views, range-safe successors, and canonical local `SE(2)` root deltas.
- Fits normalization only on clean/usable training rows with float64
  accumulation and transforms all rows without crossing a range horizon.
- Binds an authenticated four-scene pack containing `flat-standard`,
  `grail-curb-default`, `ramp-10-up-down`, and `stairs-standard`.
- Uses resumable durable member staging, prepublication validation, directory
  fsync, and Linux atomic no-replace rename. A validation failure leaves no
  published output and retains an authenticated resumable staging tree.
- Provides authenticated `build`, `verify`, and `reproduce` commands.
  Reproduction compares every relative path, size, and SHA-256, durably
  publishes its receipt, then removes only its authenticated scratch corpus.

## TDD evidence

The initial focused run failed during collection because
`mm_sonic.full_walking_terrain_lmm_corpus` did not exist. Subsequent regression
REDs covered warning-free quaternion derivatives, optional nested source-map
rate metadata used by real Takara receipts, mandatory source-local frame-map
bounds, the exact 4-D-to-36-D terrain-column relationship, prepublication
validation/resume, and coherent
metadata/shard/request mutations. A later round-trip RED exposed that lane-v1
publication omitted the required 4-D terrain and 3-D support channels; lane-v2
now persists, authenticates, tampers-checks, and exactly reopens both arrays.

Fresh focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q -W error \
  tests/python/test_full_walking_terrain_lmm_corpus.py

8 passed in 15.93s
```

Fresh Task 1 plus Task 4 aggregate verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q -W error \
  tests/python/test_full_walking_terrain_lmm_contracts.py \
  tests/python/test_full_walking_terrain_lmm_inventory.py \
  tests/python/test_full_walking_terrain_lmm_corpus.py

22 passed in 129.84s
```

Static verification:

```text
ruff check sonic/python/mm_sonic/full_walking_terrain_lmm_corpus.py \
  tests/python/test_full_walking_terrain_lmm_corpus.py
All checks passed!

ruff format --check sonic/python/mm_sonic/full_walking_terrain_lmm_corpus.py \
  tests/python/test_full_walking_terrain_lmm_corpus.py
2 files already formatted

python -m py_compile <production> <test>
exit 0

python -m mm_sonic.full_walking_terrain_lmm_corpus --help
exit 0; build|verify|reproduce present

git diff --check
exit 0
```

## Superseded real authority smoke

The independent lane verifier initially passed the two lane-v1 supplemental
artifacts against `inventory-v2.json` and `split-ledger-v2.json`:

- Takara: 41,835 rows, 1 range, manifest
  `ccd75b54201154e7628f37a6c1513afc56477d929235dd041572f5d0b177a566`
- missing GRAIL slopes: 13,754 rows, 23 ranges, manifest
  `6cf803cb687b487344b7245483ad2ff9bef53978c5141762d8e9b2a487778dc3`

Those manifests are now explicitly invalid for the formal corpus: reopening
proved that lane-v1 had replaced terrain features/support with zeros. They
must be republished under lane-v2 and pass a nonzero exact reopen before any
real all-lane assembly.

## Remaining integration dependency

All inherited-GRAIL, supplemental, Takara, and PFNN lanes must be available as
strict lane-v2 artifacts with bounded source-local maps. Therefore the real
all-lane corpus has not yet been assembled or reproduced; Task 8 must run
those commands only after every corrected immutable manifest exists.
