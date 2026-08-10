# Full Walking Task 4 Report

Status: implementation complete; independent Critical/Important review PASS

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
  contacts, quaternion continuity, terrain alignment, exact source-kind clocks,
  and exact source-local frame bounds. The frozen lane manifest set, all 16,999
  ranges, and their packed order are rechecked by both build and standalone
  verification.
- Publishes a terminal-outcome ledger with exact authenticated PFNN gait
  exclusion blocks, short fragments, retarget-continuity boundaries, and
  family-by-reason block/row counts.
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
  fsync, and Linux atomic no-replace rename. The external resume journal remains
  durable through the rename, resumed arrays are deterministically regenerated
  before reuse, and a validation failure leaves no published output.
- Provides authenticated `build`, `verify`, and `reproduce` commands.
  Reproduction compares every relative path, size, and SHA-256, durably
  publishes its receipt, then removes only its authenticated scratch corpus.

## TDD evidence

The initial focused run failed during collection because
`mm_sonic.full_walking_terrain_lmm_corpus` did not exist. Subsequent regression
REDs covered warning-free quaternion derivatives, optional nested source-map
rate metadata used by real Takara receipts, mandatory source-local frame-map
bounds, the exact G1TF 4-D-to-36-D terrain-column relationship, independently
source-bound native PFNN fit-identity semantics, prepublication validation/resume,
coherently rehashed but wrong source maps, dropped secondary ranges, standalone
fullness verification, exact terminal reasons, receipt placement, and coherent
metadata/shard/request mutations. A later round-trip RED exposed that lane-v1
publication omitted the required 4-D terrain and 3-D support channels; lane-v2
now persists, authenticates, tampers-checks, and exactly reopens both arrays.

Fresh focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q -W error \
  tests/python/test_full_walking_terrain_lmm_corpus.py

15 passed in 23.65s
```

Fresh Task 1 plus Task 4 aggregate verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q -W error \
  tests/python/test_full_walking_terrain_lmm_contracts.py \
  tests/python/test_full_walking_terrain_lmm_inventory.py \
  tests/python/test_full_walking_terrain_lmm_corpus.py

29 passed in 66.58s
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

## Real authority verification

The formal frozen lane set is pinned to:

- inherited GRAIL: manifest `e7cf6a83c65f642a69cf7e1f90b822a0e37cf4a5e8db0fe72298e1d2bfc19d78`;
- missing GRAIL slopes: manifest `21e504ee7f2ec83921174a5c541ec7e0a19b43655c4cf9f7e6650edbcd0f22c8`;
- Takara: manifest `f8436be15845b1f42693d1161e453a8b628d53ebd5685480e066d3e9cbe4074b`; and
- PFNN v3: manifest `ceff0acb02ac1d189eac9da96cc3a21bbe88fe65259afb85bbc488dc79d706b7`.

PFNN v3 passed full strict lane verification at 246,163 rows and 1,161 ranges,
including all 44 four-frame ranges. Its independently reconstructed terminal
aggregate exactly matches Task 3 receipts: crawl 18, crouch 58, jog 232, jump
204, run 98, short fragment 425, and retarget-continuity boundary 2,221.
The four formal lanes total 9,758,524 rows and 16,999 ranges.

## Remaining integration dependency

The real all-lane corpus build/reproduction and downstream training remain
Task 8 operations. They must use Task 4 HEAD
`14729ed2cb3785298d5868340d4af32e172adfdb`; earlier Task 4 checkpoints do not
contain the authenticated terminal-outcome member or the final standalone
coverage gate.
