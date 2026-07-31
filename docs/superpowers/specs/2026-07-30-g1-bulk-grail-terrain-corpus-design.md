# G1 Bulk GRAIL Terrain Corpus Design

Date: 2026-07-30

Status: Approved

Branch: `research/g1-torch-terrain-kinematics`

Supersedes the four-record GRAIL scope in
`2026-07-30-g1-expanded-stair-curb-corpus-design.md`.

## Goal

Test vanilla dense terrain-aware motion matching against the actual local GRAIL
coverage: 1,769 curb recordings, 6,094 `stair_p1` recordings, and 6,094
`stair_p2` recordings. This phase is kinematic only. It does not run SONIC,
physics, obstacle avoidance, learned depth, or tracking.

The first interactive artifact is a deterministic, representative curb-and-
stair subset. The same code must then scale to all 13,957 recordings without
changing source registration or matcher semantics.

## Evidence and Constraints

- Every sampled source has 250 frames at 25 Hz, 29 G1 DOFs, one paired object
  record, and one paired USD.
- The dataset is pinned to Hugging Face revision
  `943946a972d5de2eb0d2ff214b236d0e43575fd7` by
  `/home/ubuntu/datasets/GRAIL/g1_mm_inventory.json`.
- That inventory contains robot and USD hashes for the requested partitions.
  It mistakenly lists curb `recon` files instead of curb `objects` files, so
  paired-file validation must be authoritative and curb object hashes must be
  computed locally.
- Recorded terrain poses are effectively static but not bitwise constant.
  Across the requested corpus, maximum translation excursion is below 1.4 cm
  and maximum rotation excursion is below 0.005 degrees. The loader uses the
  first recorded pose and rejects excursions above 2 cm or 2 degrees.
- Motion conversion takes about 42 ms per clip. The current row-by-row terrain
  rasterizer takes 2--15 seconds per clip and is the build bottleneck.
- The host has 1.5 TiB RAM and each available L40S has 46 GB VRAM. A full
  native corpus and dense feature database fit, but serial construction and
  thousands of redundant Python point queries do not meet the iteration goal.
- GRAIL is strongly forward-biased. All curb recordings and 11,934 of 12,188
  stair recordings are predominantly forward; 201 stair recordings are
  predominantly lateral and none are predominantly backward.

## Considered Approaches

### 1. Expand the hardcoded source registry

This preserves the existing publisher but would place roughly 42,000 hashes in
Python source and still rasterize serially. It is reproducible but unmaintainable
and too slow.

### 2. Discover files directly on every build

This is simple but silently accepts dataset drift and recomputes hashes for
about 22 GB on every run. It does not meet the fail-closed source contract.

### 3. Pinned inventory plus paired-file validation

This is the selected approach. A bulk index reads the pinned inventory,
requires exact robot/object/USD stems, validates inventory-backed sizes and
hashes, computes only missing curb-object hashes, and emits stable short logical
identities. Selection and publication consume this index rather than a Python
whitelist.

For terrain construction, the first implementation retains the existing
per-clip grid and dense feature semantics but replaces the rasterizer's
grid-point/triangle scan with triangle-bounded rasterization. This minimizes
semantic change and supports both a quick representative corpus and the full
corpus. Precomputed consolidated terrain-feature sidecars are a later
optimization only if measured loading or GPU allocation, rather than
rasterization, becomes the next bottleneck.

## Source Index

Create a focused bulk-index module with immutable records:

- dataset revision and inventory SHA-256;
- partition (`curb`, `stair_p1`, or `stair_p2`);
- exact source stem;
- robot, object, and USD paths;
- byte sizes and SHA-256 values; and
- a stable logical name formed from partition plus a digest prefix, never the
  potentially overlong source stem.

The index rejects:

- an unsupported inventory schema or repository identity;
- a missing, symlinked, or out-of-root path;
- duplicate inventory paths;
- a robot stem without exactly one paired object and USD;
- a size or hash mismatch;
- duplicate logical identities; or
- an unrequested partition.

The checked index is cached as a small generated JSON report under `build/`.
It is evidence, not source authority; every build rechecks the pinned inventory
identity and selected file identities.

## Deterministic Selection

The command line supports:

- `--partitions curb stair_p1 stair_p2`;
- `--limit-per-partition N`, where omission means all records; and
- `--selection-seed`, defaulting to zero.

Selection is deterministic and order-independent. It divides each partition's
digest-sorted inventory into equal intervals and selects one digest-ranked
record from each interval. The representative build initially selects the same
bounded count from each partition, so the much larger stair partitions cannot
erase curb coverage. The manifest records the total available and selected
counts per partition.

The selector does not claim motion-semantic stratification. The existing
direction audit is reported separately, and the full build remains the
authoritative coverage test.

## Generalized GRAIL Record Loader

Replace the four-name-only loader boundary with a public single-source loader.
The existing `load_pinned_sources()` remains as a compatibility wrapper for the
qualified small corpus.

The generalized loader keeps the exact 250-frame, 25 Hz, joint-order,
finite-array, quaternion, scale, and paired-identity checks. For the recorded
object trajectory it:

1. normalizes and unrolls every quaternion;
2. measures maximum translation and shortest-arc rotation from frame zero;
3. rejects excursions above 0.02 m or 2 degrees; and
4. freezes frame-zero position, quaternion, and scale as the static terrain
   transform.

The accepted excursion metrics are published for audit.

## Faster Exact Terrain Rasterization

The optimized rasterizer preserves the existing public contract:

- float32 origin and cell size;
- the same inclusive grid dimensions;
- exterior height zero;
- maximum Z for overlapping projected triangles;
- closed barycentric boundaries with the existing tolerance; and
- float32 output heights.

Instead of asking every grid node to scan all triangle bounds, it visits each
nondegenerate triangle, computes its clipped grid-index bounding box, evaluates
barycentric heights only inside that box, and maximum-reduces into the grid.

Tests compare the optimized result to the current reference implementation on:

- the synthetic two-level surface;
- overlapping triangles;
- vertical and projected-degenerate triangles;
- boundary and reversed-winding triangles;
- randomized finite meshes; and
- real curb, procedural stair, and high-polygon stair sources.

The reference implementation remains privately available during qualification.
A real-source benchmark must show identical float32 grid values and materially
lower construction time before the optimized path becomes the default.

## Publisher and Artifacts

Add a separate bulk CLI so the qualified small and expanded publishers remain
unchanged:

```text
resources/build_g1_torch_grail_corpus.py
build/torch-grail-terrain-<selection>/
  manifest.json
  flat/motion.npz
  terrain/grail/<logical-name>/motion.npz
  terrain/grail/<logical-name>/terrain.npz
```

The publisher reuses the current 50 Hz conversion, terrain alignment, admission
oracle, deterministic NPZ serialization, validation, fsync, and atomic
replacement behavior. Candidate failures are recorded and do not abort other
records. A failed publication never replaces the last valid corpus.

The initial representative build targets enough clips to expose varied curb,
ascent, descent, turning, and lateral candidates while remaining playable
within one iteration. Counts increase only after measured build, load, GPU
memory, and search latency remain acceptable.

## Interactive Evaluation

The existing MuJoCo kinematic viewer remains the human-facing test. Its query
scene is one authenticated stair grid, controls remain WASD, Backspace resets,
and no physics or tracking process runs.

Qualification has three stages:

1. build and load a small smoke selection from all three partitions;
2. build and interactively test a representative balanced selection; and
3. build all 13,957 records and rerun the same commands.

Automated diagnostics record selected source/frame, command progress, terrain
feature cost, discontinuity/jerk, matcher latency, freezes, and source-family
usage. The adversarial routes remain side approach, cross-tread motion, turns
on stairs, diagonal ascent/descent, side exits, reversals, and stop/restart.

## Interpretation

- If the representative and full corpora improve the behavior, the next work is
  matcher tuning and depth-to-terrain replacement.
- If only the full corpus improves it, data volume and retrieval scaling are
  central.
- If neither improves backward or lateral behavior, the measured direction
  audit makes missing motion semantics the primary explanation; mirroring,
  direction-conditioned augmentation, or new data becomes the next experiment.
- SONIC tracking is deliberately deferred until the kinematic reference is
  good enough to track.

