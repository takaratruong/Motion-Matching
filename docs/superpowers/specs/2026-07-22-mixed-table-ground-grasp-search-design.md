# Mixed Table/Ground Grasp Search Design

## Goal

Extend the standalone grasp-trajectory viewer from the GRAIL
`pickup_table` corpus to one combined `pickup_table` plus `pickup_ground`
corpus. The requested world-space grasp remains the only retrieval anchor:
grasp height, position, and orientation determine which motions are compatible.
There is no Table/Ground mode switch and no category threshold in retrieval.

The current table-only artifact and its frozen controller gates remain
unchanged. The combined artifact is built at a new path and is used by the
standalone viewer only until separately promoted.

## Corpus Evidence

The local release contains 2,991 ten-second tabletop pickup clips across 685
objects and 1,613 fifteen-second ground pickup clips across 631 objects. A scan
of the first hand-contact sample found these source object-center height ranges:

- `pickup_table`: 0.2363 through 1.0096 m;
- `pickup_ground`: 0.0067 through 0.2999 m.

The small 0.2363--0.2999 m overlap is meaningful. A fixed category cutoff would
discard plausible candidates near low tables or tall ground objects. The
existing grasp alignment preserves world Y and already measures the residual
Contact wrist height, so continuous pose matching is the correct selector.

## Source Ingestion

The builder accepts one or more `--source-root` arguments. A single-root build
retains the existing tabletop schema and output structure. The combined target
passes the local `pickup_table` and `pickup_ground` roots and writes to
`build/smart-pickup/table-ground-pack`.

Sequence parsing accepts exactly these forms:

```text
pickup_table__<object-id>__<three-digit-index>
pickup_ground__<object-id>__<three-digit-index>
```

The object identity remains `<object-id>`, without the source category. Thus,
the same physical object represented in both corpora stays in one database or
held-out partition and cannot leak across the evaluation split. Sequence IDs
remain globally unique and lexicographically ordered.

Tabletop metadata continues to require `object_name`, `table_pos`,
`table_quat`, and `table_size`. Ground metadata requires exactly
`object_name`. During ground ingestion, the builder supplies a canonical
virtual floor slab in the existing support fields:

```text
source Z-up position: [0, 0, -0.02]
source Z-up size:     [20, 20, 0.04]
source rotation:      identity
```

After the existing Z-up-to-Y-up conversion, the slab top is exactly Y=0. The
slab is support metadata for feature construction and source labeling; it is
not added to full-skeleton collision geometry in the viewer. The source GRAIL
ground motions are already physics validated, and yaw/XZ grasp alignment does
not alter their recorded vertical placement.

The schema-v1 binary layout remains unchanged. For a multi-root build, the
manifest keeps the required absolute `source_root` as the common parent of all
roots and adds sorted `source_roots`. Each manifest clip adds `source_category`
with value `pickup_table` or `pickup_ground`. Existing validators accept these
additive fields and explicitly validate them when present. A single-root build
omits the additive fields and preserves its existing manifest representation.

Object dimensions are sampled from one deterministic USD representative per
object and source category. When the same object ID occurs in both pickup
categories, its two representative bounds must agree within the existing 1 mm
tolerance. This retains one physical-object identity without silently applying
the wrong geometry.

## Retrieval and Trajectory Construction

Search remains grasp-centric and independent of object identity. For every
clip, search first reconstructs only its recorded Contact wrist pose. It applies
the existing upright yaw/XZ alignment and computes Contact height and optional
orientation residuals. A clip exceeding the existing 0.12 m position envelope
or orientation envelope is rejected before allocating its complete trajectory.

Only compatible clips are expanded from source clip start through final
contiguous Lift. This is important for the combined corpus because a ground
approach can be approximately 12 seconds before Contact. The selected path
therefore shows the full recorded walk/kneel approach, Reach, Contact, and
complete Lift while continuing to omit Hold.

No category penalty, category quota, or hard height threshold enters cost or
membership. Near the natural height overlap, table and ground clips may both
appear and are ranked by the same grasp-pose residual. `HandTrajectory` carries
a diagnostic `SupportKind` derived by matching the exact canonical virtual-floor
dimensions and top height so the UI can label results `TABLE` or `GROUND`; this
label never gates retrieval.

IK behavior is unchanged: no correction through Reach entry, smooth correction
from Reach to Contact, and full correction through Lift. Root/body alignment
remains yaw plus X/Z only, preserving the source body's vertical grounding.

## Viewer Scene and Collision Semantics

The viewer starts from a tabletop clip and retains the existing table and flat
ground grid. The object/grasp can be moved continuously. At tabletop height,
ground clips normally fail the Contact-height envelope. As the grasp is lowered,
ground clips become compatible naturally. A low target left under the visible
table is physically invalid and may be rejected by table collision; the user
can move it clear of the table for a ground-pickup query.

All compatible motions still receive active-arm IK and full shaped-skeleton
collision checks against the target object and visible table. The virtual floor
slab is not submitted to the existing sphere/capsule collision routine because
recorded feet intentionally contact the floor. No floor-penetration correction
is introduced in this baseline.

Enter remains the only expensive recomputation trigger. Object controls,
stale-search indication, option cycling, position-only mode, and rejected-path
visibility remain unchanged. The option line adds the source support label.

## Memory and Runtime Bounds

The combined corpus must not restore the DCV instability caused by retaining
unused arrays. The viewer adds a focused trajectory-database loader for the
existing binary layout. It retains only fields required by grasp search,
shaping, collision, labeling, and scene construction and seeks over velocities,
hand DOFs, contact arrays, and other unused frame data. It validates header
counts, exact expected byte size, retained array shapes, quaternions, range
bounds, phases, and end-of-file position.

Full skeleton pose arrays continue to be released after collision validation
and regenerated only for the selected option. Background hand paths retain a
fixed drawing stride while the selected path and skeleton use every source
sample.

The acceptance bound for the new live process is steady-state RSS no greater
than the approximately 1.1 GB observed for the previous table-only viewer.
Exactly one viewer process may run, and its on-disk and running executable
hashes must match.

## Failure Handling

- Missing modalities across either source root fail discovery before building.
- Unsupported sequence prefixes or malformed indices fail explicitly.
- Ground metadata containing partial tabletop fields is rejected as ambiguous.
- Duplicate sequence IDs across roots fail explicitly.
- Conflicting source categories for one sequence fail explicitly.
- Clip-level conversion/phase failures remain reviewed rejections under the
  existing `--allow-rejections` policy.
- An empty combined database or a split without both nonempty partitions fails
  publication.
- A malformed or truncated compact-loaded binary fails before opening a window.
- Zero compatible or zero collision-safe options remains a valid interactive
  viewer state.

## Verification

Automated tests must prove:

- the parser accepts only exact table and ground sequence IDs;
- repeated roots discover and deterministically order both corpora;
- duplicate IDs and missing modalities across roots fail;
- tabletop metadata remains strict and ground metadata produces a floor whose
  canonical top is Y=0;
- shared object IDs across categories remain in one split partition;
- single-root tabletop fixture output remains compatible;
- Contact height rejects clearly incompatible clips before path extraction;
- no source-category term affects compatible membership or ranking;
- overlap-height table and ground clips can coexist in one ranked result set;
- ground extraction spans clip start through final contiguous Lift;
- the focused loader agrees with the full loader for every retained field and
  rejects malformed size/range/phase/quaternion fixtures;
- the viewer starts from a tabletop clip, labels support kind, retains
  Enter-only search, and remains independent of mesh, terrain, diffusion, and
  controller code.

Full-corpus verification builds the new combined pack without replacing the
reviewed table-only pack, validates its manifest/artifacts, runs the standalone
viewer tests and release build, then replaces only the exact old viewer process.
