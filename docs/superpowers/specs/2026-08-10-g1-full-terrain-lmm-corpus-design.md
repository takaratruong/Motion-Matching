# Full PFNN, GRAIL, and Takara Terrain-LMM Corpus Design

Date: 2026-08-10

Status: awaiting written-spec review

## Objective

Build and verify a resumable, sharded 60 Hz Unitree G1 motion corpus for
terrain-aware Learned Motion Matching (LMM). The corpus is built from all
admitted PFNN walking and terrain motion, all admitted GRAIL curb, slope, and
stair motion, and the complete Takara walk. Source families are processed in
parallel but emitted through one canonical G1 row contract.

This design covers data inventory, processing, terrain pairing, verification,
normalization, and publication of the combined corpus index. It does not train
the LMM networks or implement the interactive matcher. Training begins only
after this corpus has a complete verification receipt.

The small PFNN and GRAIL subsets already used in canaries are processor smoke
fixtures. They do not define or limit the final corpus.

## Admitted Sources

### PFNN

The PFNN lane attempts every released walking or terrain-motion source used by
the upstream PFNN database, including its official mirrors:

- `LocomotionFlat*`;
- `WalkingUpSteps*`;
- walking and terrain portions of `NewCaptures*`;
- matching `.gait`, `.phase`, and footstep sidecars;
- `patches.npz` and its source heightmap coordinates;
- all 13 released smooth PFNN heightmaps; and
- the upstream flat, rocky, beam, and jumpy terrain-fitting semantics.

Rows whose authoritative gait label is run, jog, crouch, crawl, or jump-only
are not admitted to the walking view. Stand, walk, stand/walk blends, and
terrain transitions are admitted. Excluded rows remain counted in the source
receipt; they are never silently discarded.

The local release contains approximately 931,980 motion frames at 120 Hz,
which is approximately 466,008 rows under PFNN's official 60 Hz downsample
before gait and continuity filtering. The one-frame rest source is inventory
metadata, not a motion range.

PFNN BVH motion is retargeted to canonical G1 using the current source-grounded
retarget contract: the 5.6444 position scale, authenticated GMR and retarget
project commits, canonical 29-joint ABI, and the exact 18-field receipt. The
existing 21-window retarget set remains a smoke and regression fixture. Full
processing retargets every admitted source rather than stopping at those
windows.

### GRAIL

The GRAIL lane admits exactly these classes:

| Class | Paired clips |
| --- | ---: |
| curb | 1,769 |
| slope | 1,880 |
| stair_p1 | 6,094 |
| stair_p2 | 6,094 |
| **Total** | **15,837** |

Sitting, pickup-ground, pickup-table, and other GRAIL activities are excluded.

Every admitted GRAIL robot motion contains 250 native-G1 frames at 25 Hz and
is paired by exact stem with a USD terrain asset. Duration-preserving 60 Hz
conversion produces 598 rows per clip, for a nominal 9,470,526 GRAIL rows.
Robot motion, USD, object trajectory, reconstruction data when present, and
metadata when present are separately hashed before deserialization.

The GRAIL classes require separate placement adapters:

- curb and slope bind robot, USD, reconstruction, metadata, and the recorded
  object trajectory;
- stair-p1 binds robot, USD, metadata, and the recorded object trajectory and
  defines placement without assuming a missing reconstruction file; and
- stair-p2 binds robot, USD, and the recorded object trajectory and defines
  placement without assuming missing reconstruction or metadata files.

The object trajectory is authoritative evidence. A terrain object that moves
during a clip is represented as time-varying terrain and tagged `dynamic`;
it is not silently treated as the static reconstruction. Static and dynamic
terrain rows remain separately selectable in the combined index.

### Takara

The Takara lane admits the complete native-G1 flat walk: 34,863 frames at
50 Hz. Duration-preserving 60 Hz conversion produces 41,835 rows. The remap,
source arrays, skeleton, and frame rate are authenticated through the existing
Takara source adapter. Its terrain surface is canonical flat zero.

### Nominal corpus size

Before PFNN gait/continuity filtering and quality classification, the three
families contain approximately 9,978,369 rows at 60 Hz. Processing must be
streaming and sharded; one monolithic in-memory artifact is out of scope.

## Considered Architectures

### A. Source-native processors into canonical immutable shards — selected

PFNN, GRAIL, and Takara retain their correct source-specific loaders and
terrain rules, but emit identical canonical G1 motion and terrain arrays.
Each source is independently receipted, and deterministic source groups are
packed into immutable shards after verification. A streaming index provides a
logical combined corpus without copying all rows into memory.

This approach preserves provenance, supports resumption and parallel work,
and lets one bad source fail without blocking other families.

### B. Convert every terrain to one raster format before motion processing

This simplifies later sampling but loses exact PFNN fitted-surface semantics,
GRAIL mesh and object-pose evidence, and class-specific placement diagnostics.
It also makes a rasterization bug contaminate every downstream artifact. This
approach is rejected.

### C. Keep sources native and compute motion/features on demand during training

This minimizes intermediate storage but repeats expensive retargeting, USD
decoding, FK, and terrain queries during every experiment. It is slower,
harder to reproduce, and unsuitable for approximately ten million rows. This
approach is rejected.

## Parallel Processing Architecture

### Inventory phase

One read-only inventory command scans all configured roots and writes a
canonical source ledger. Every expected identity appears exactly once with:

- source family and class;
- stable source identity, with mirrors linked to their canonical identity;
- all input paths, sizes, and SHA-256 digests;
- source rate, frame count, and expected 60 Hz count;
- terrain asset and placement-input identities;
- gait/phase/footstep sidecars where applicable; and
- an initial status of `pending`.

The inventory is immutable for one corpus build. Changed input bytes create a
new build identity rather than mutating an existing ledger.

### Independent worker lanes

The orchestration launches these lanes concurrently:

1. PFNN flat and transition sources;
2. PFNN terrain sources;
3. GRAIL curb;
4. GRAIL slope;
5. GRAIL stair-p1;
6. GRAIL stair-p2; and
7. Takara walk.

Each lane uses a bounded process pool selected from available CPU and memory.
No worker writes shared array files. A worker writes only a unique temporary
source directory, fsyncs it, validates it, and atomically publishes a source
receipt. Interrupted temporary outputs are never treated as complete.

GRAIL sources are packed in deterministic, sorted groups of at most 64 clips
after per-source verification. PFNN uses one logical motion source per shard
because source lengths vary substantially. Takara uses one shard. A failed
source does not fail the worker pool or stop unrelated lanes.

### Two-pass publication

Pass one emits raw, unnormalized canonical rows and per-shard streaming
statistics. After every source has a terminal receipt, the indexer computes
global feature offsets and scales over the frozen train split of the default
`clean + usable` view using numerically stable streaming accumulation. A
separate strict view contains only `clean` rows but uses the same frozen
normalization. `quarantine` rows never influence normalization. Pass two
publishes normalized matching features or applies the frozen normalization
lazily through the index.

No derivative, contact filter, feature horizon, or training window crosses a
source, continuity, terrain-fit, mirror, or shard range boundary.

## Canonical Motion Row

All admitted rows use the existing 31-bone G1 hierarchy:

- bone 0 is the synthetic planar `Simulation` root;
- bones 1-30 are the canonical G1 bodies;
- local rotations are normalized WXYZ quaternions;
- local position, rotation, linear velocity, and angular velocity are stored;
- contacts are `[left_foot, right_foot]`;
- range and source-map provenance is explicit; and
- time is exactly 60 Hz.

The row also stores source family, activity class, terrain class, static or
dynamic terrain status, mirror identity, quality tier, and source-local frame
mapping.

PFNN 120 Hz and Takara 50 Hz inputs are resampled by timestamp with linear
vector interpolation and sign-continuous quaternion SLERP. GRAIL 25 Hz is
converted the same way. Each 60 Hz row stores left source index, right source
index, and interpolation alpha.

### Support-relative canonicalization

Before different families are mixed, each pose is made relative to its paired
support surface. The source terrain height at the motion root is subtracted
from authored vertical placement, then velocities, contacts, and matching
features are recomputed. World terrain height is added back by runtime
placement, not memorized as clip-specific elevation in the latent pose.

This operation uses the same source terrain adapter as feature construction.
Flat Takara support is zero. Missing or nonfinite support is a structural
failure, not a fabricated flat fallback.

## Canonical Terrain Contract

Terrain is a first-class per-row input in both PFNN and GRAIL.

Every terrain adapter provides:

- support height at arbitrary planar points;
- optional surface gradient/normal;
- validity and walkability classification;
- source asset and placement identity;
- per-frame object transform for dynamic GRAIL terrain; and
- a deterministic rendering/raster artifact for later runtime use.

PFNN adapters evaluate the authenticated upstream fitted surface: selected
heightmap patch, contact alignment, and its residual fit. Full-corpus PFNN
processing implements each upstream terrain family rather than applying the
rocky fitter to flat, beam, or jumpy sources.

GRAIL adapters ray-query the authenticated paired USD support surface under
the authoritative per-clip placement and object trajectory. Stair and curb
vertical faces do not become support heights; the adapter selects the upper
walkable surface for each planar query and retains discontinuity and
walkability diagnostics.

### Stored terrain observations

Each row stores three terrain representations:

1. the four LMM centerline heights at 0.25, 0.50, 0.75, and 1.00 m ahead;
2. a PFNN-style 3-lane by 12-point terrain grid as a versioned sidecar; and
3. root and foot support heights used for placement and diagnostics.

The first LMM uses only the four centerline heights. Therefore its search
feature vector remains the established 31 dimensions: Orange Duck's 27 motion
and trajectory features plus four terrain heights. The richer 36-height grid
is retained so later terrain models can use it without reprocessing the source
corpus; it does not silently change the initial LMM network ABI.

## Matching-Feature Contract

The initial terrain LMM feature vector is:

| Dimensions | Meaning |
| --- | --- |
| 0-5 | left and right foot positions in the root frame |
| 6-11 | left and right foot velocities in the root frame |
| 12-14 | hip velocity in the root frame |
| 15-20 | future planar root positions at 20, 40, and 60 frames |
| 21-26 | future facing directions at 20, 40, and 60 frames |
| 27-30 | relative centerline terrain heights at 0.25-1.00 m |

The original Orange Duck matcher has the first 27 dimensions. Terrain support
adds dimensions 27-30. PFNN phase, gait, source class, and the 36-height sidecar
remain selection/provenance data and are not hidden additions to this vector.

Raw feature rows are stored before normalization. Training and validation are
split by canonical source or terrain identity before statistics are computed;
mirrors and variants of one identity cannot straddle splits. Feature scales
must be finite. A constant dimension is explicitly disabled rather than
divided by a small invented scale.

## Verification and Quality Classification

### Structural gates

A source is structurally rejected if any of these fail:

- expected source or terrain bytes are missing or their digest changes;
- decoding, skeleton ABI, units, frame rate, or source mapping is invalid;
- motion or terrain arrays contain nonfinite values;
- quaternion norm differs from one by more than `1e-5`, or independent FK
  reconstruction differs from the canonical conversion by more than
  `1e-5 m`;
- support queries needed by admitted motion are undefined;
- source, terrain, or object placement cannot be coherently paired; or
- a temporal operation crosses a declared range boundary.

Structural thresholds protect data meaning, not learned sub-millimeter output.

### Quality tiers

Structurally valid ranges receive one of three quality tiers. Metrics use the
canonical XML's eight ankle-roll foot sphere-bottom probes and the paired
terrain support surface. Planted drift begins on the seventh consecutive
contact frame and measures maximum terrain-tangent displacement from that
onset. Thresholds are:

| Metric | `clean` | `usable` |
| --- | ---: | ---: |
| maximum joint step | `<= 0.25 rad/frame` | `<= 0.50 rad/frame` |
| maximum planar root step | `<= 0.05 m/frame` | `<= 0.10 m/frame` |
| maximum root rotation step | `<= 0.15 rad/frame` | `<= 0.30 rad/frame` |
| maximum foot penetration | `<= 0.02 m` | `<= 0.05 m` |
| maximum forbidden-body penetration | `<= 0.01 m` | `<= 0.03 m` |
| maximum planted support gap | `<= 0.03 m` | `<= 0.06 m` |
| maximum planted-foot drift | `<= 0.03 m` | `<= 0.06 m` |

Both `clean` and `usable` walking ranges require at least one accepted contact
run for each foot. A structurally valid range exceeding any `usable` threshold
or lacking bilateral contact evidence is `quarantine`. Static versus dynamic
terrain is an independent tag and does not by itself change the tier.

The tiers mean:

- `clean`: included in the default and strict training views;
- `usable`: included in the default view with explicit centimeter-scale
  contact, collision, or continuity warnings; or
- `quarantine`: processed and indexed for audit, but excluded from default
  training until its named issue is reviewed.

Quality metrics include joint-limit clearance, maximum adjacent joint/root
step, bilateral contact counts, planted-foot drift, foot penetration and
support gap in centimeters, forbidden-body contact, terrain relief, terrain
feature variance, and static/dynamic object motion. The receipt records raw
metrics and thresholds. It does not claim sub-millimeter learned accuracy.

Processing everything means every inventory item gets an accepted,
quarantined, or rejected terminal receipt. It does not mean corrupt data is
silently trained on, and it does not permit unreported skips.

A source receipt is `accepted` when at least one structurally valid range is
`clean` or `usable`; it is `quarantined` when all structurally valid ranges are
`quarantine`; and it is `rejected` only on a structural failure that prevents
canonical decoding. The receipt always lists every derived range and tier.

### Independent verification

The verifier reopens published bytes rather than trusting worker-owned arrays.
It recomputes:

- source and terrain digests;
- 60 Hz row maps and duration;
- canonical FK and per-range velocities;
- contacts and support-relative placement;
- the four LMM terrain heights and sampled 36-height sidecar rows;
- raw 31-dimensional matching features;
- range ownership and boundary safety; and
- per-shard statistics and receipt totals.

Full numeric checks stream over every row. Visual checks sample at least one
source per family, GRAIL category, PFNN terrain family, quality tier, and
terrain-relief quartile. Visual checks compare source playback, canonical G1
playback, and rendered terrain registration.

At corpus level, the verifier requires:

- inventory identities equal accepted plus quarantined plus rejected
  identities, with zero missing outcomes;
- deterministic split ownership and no source/mirror leakage;
- exact row/range totals matching source receipts;
- finite global normalization and active nonflat terrain channels;
- no successor or feature horizon crossing a range;
- deterministic hashes for rebuilt fixture shards; and
- a published report of all exclusions and quality warnings by family.

## Artifacts

The build publishes:

- immutable source inventory and build configuration;
- per-source receipts;
- immutable motion/terrain shards;
- source-map and range index;
- raw 31-dimensional features;
- four-height terrain sidecars;
- PFNN-style 36-height sidecars;
- root/foot support sidecars;
- streaming-statistics receipts;
- frozen train/validation/test split and normalization;
- default (`clean + usable`), strict-clean, and quarantine dataset views; and
- one combined verification report and content-addressed corpus manifest.

Large binary data remains outside Git. Git contains processors, schemas,
tests, small fixtures, manifests, and reports.

## Error Handling and Resumption

Workers claim sources through exclusive, content-addressed staging paths.
Successful source and shard outputs are write-once. A restart reads terminal
receipts and resumes only pending or interrupted identities. It never deletes
or overwrites an accepted output.

Worker exceptions are serialized with source identity, processor version,
input digests, stage, and a bounded error message. The orchestrator continues
other sources and returns a nonzero final status if any source lacks a terminal
receipt. Retrying a rejected source requires a new build identity or an
explicit retry ledger; it cannot rewrite the original evidence.

## Testing Strategy

Development begins with small authenticated fixtures from every lane:

- current PFNN flat and rocky retargets plus one beam/jumpy semantic fixture;
- one GRAIL curb, slope, stair-p1, and stair-p2 pair;
- the Takara walk fixture;
- static and moving GRAIL object trajectories; and
- PFNN heightmap, patch, mirror, and gait exclusions.

Tests cover source authentication, resampling, terrain placement, dynamic
object transforms, support-relative pose conversion, contacts, feature parity,
range isolation, structural rejection, quality-tier assignment, atomic
publication, crash resumption, deterministic sharding, streaming statistics,
split leakage, and independent revalidation.

Before the full build launches, two independent fixture builds must be
byte-identical. After launch, per-lane progress reports provide attempted,
accepted, quarantined, rejected, and pending counts plus processed rows and
throughput. The final corpus cannot be called verified until every inventory
identity has a terminal outcome and the independent corpus verifier passes.

## Completion Claim

Completion means:

> All admitted PFNN walking/terrain sources, GRAIL curb/slope/stair sources,
> and the Takara walk were attempted through parallel, resumable processors;
> every source has explicit provenance and a terminal quality result; and the
> resulting shared 60 Hz G1 terrain-LMM corpus and combined index passed
> independent verification.

It does not mean the LMM has been trained, that every source is clean, that the
learned projector works, or that the final model generalizes to unseen terrain.
