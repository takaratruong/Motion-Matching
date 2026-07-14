# G1 Certified Clearance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sampled G1 sphere, capsule, and swept-foot clearance with a deterministic, mathematically conservative G1HF/v2 certificate that fails closed at the terrain boundary and under bounded-work exhaustion.

**Architecture:** One strict-floating-point translation unit computes signed vertical clearance for a finite capsule centerline against each fixed-diagonal terrain triangle. A sphere is the zero-length capsule case, and each linearly swept foot sphere is the same capsule primitive. The kernel reduces a capsule/triangle pair to eight triangular patches on a Minkowski-difference prism boundary, minimizes each patch analytically with outward-rounded interval guards, and uses bounded deterministic subdivision only for numerically degenerate patches. Swing lift selection belongs to the controller transaction: it stages a finite immutable ladder of actual sole commands, runs the production IK/orientation/FK path for each entry, certifies the resulting world-space sphere endpoints, and commits the first already-evaluated passing pose.

**Tech Stack:** C++17; existing `g1_ik.h`, `terrain_runtime.h`, `vec.h`, and `array.h`; IEEE-754 binary32/binary64; a non-inline `g1_clearance.cpp` compiled with strict FP; standalone strict, fast-math-caller, sanitizer, and parity test executables.

## Global Constraints

- Do not edit the active `docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md` on this design branch. Section 15 records the read-only reconciliation with integrated Task 2.
- Consume the existing G1HF/v2 node, cell, diagonal, query-domain, output-rounding, and exterior semantics. Do not introduce bilinear terrain interpolation or a second heightfield format.
- The authoritative cell triangles are closed `T0=(p00,p10,p11)` for `tx >= tz` and `T1=(p00,p11,p01)` for `tx < tz`; evaluating both closed triangles on their shared diagonal is permitted because their heights agree there.
- Never use `heightfield_sample_v2` returning a finite exterior value as proof that a query was in-domain or that the field was valid.
- Full projected sphere/capsule footprint containment in the inclusive authoritative node rectangle is required. Exact tangency is in-domain; any proven excursion or unresolved boundary comparison fails closed.
- Store safety bounds as binary64. Do not round a safety decision through binary32.
- The certified arithmetic and sole-command materializer must not be compiled under `-ffast-math`, reassociation, contraction, or LTO into a fast-math caller.
- The strict TU requires round-to-nearest **and gradual underflow** for binary32/binary64. An inherited x86 FTZ/DAZ mode or a portable volatile denormal probe failure is `ArithmeticFailure`, never an implicit arithmetic variant.
- A lift certificate may use only the actual four world-space sphere centers produced by the staged production controller path. No clearance planner may predict those centers as `sphere_y+lift`, infer them from the sole command, or rely on monotonicity across lift candidates.
- Exactly `dt = 1.0f / 25.0f` remains the swing-history timing contract. Call the Task 2 producer helper `g1_ik_dt_is_exact_25_hz`; do not add a second tolerant comparison.
- The immutable swing ladder is the 41 canonical binary32 values `RN32(i/500 m)` for `i=0..40`; its exact endpoints are `+0.0f` and `0.08f`. If no staged entry certifies, request safe stop without estimating an unrepresented required lift.
- A returned error or finite rejected candidate cannot mutate swing history, accepted pose arrays, accepted clearance, support state, matcher state, or simulation state.
- Do not allocate or iterate in proportion to unchecked field dimensions, radius, segment length, or float-derived sample counts.

---

## 1. Scope and Non-Goals

### In scope

- Checked signed point clearance against an in-domain G1HF/v2 triangle.
- Certified signed vertical clearance of a sphere and a finite capsule segment.
- Continuous linear sweep of each of the four configured G1 foot collision spheres.
- Four-sphere foot aggregation, thigh/shin capsule aggregation, and pose-level minimum diagnostics.
- Finite staged-controller lift selection against a clearance target that interpolates from planted to swing clearance.
- Explicit `OutsideDomain`, bounded-work, numerical-certification, invalid-input, and invalid-field outcomes.
- Deterministic witness selection, diagnostic work counters, strict/release parity, and transactional state ownership.

### Non-goals

- Euclidean separation distance. This plan preserves the existing signed **vertical** clearance definition: lower body height minus authoritative terrain height at the same XZ coordinate.
- Arbitrary curved center trajectories. The fixed-update contract is a linear path between prior accepted and actual staged/FK current sphere centers.
- Inferring rigid-foot interpolation from quaternions between updates. The four configured sphere-center paths are the owned sweep geometry.
- Rebuilding G1HF/G1WM artifacts, changing terrain rendering, changing the matcher, changing support retargeting, or changing the controller root.
- Treating the published exterior height as physical terrain outside the authoritative rectangle.
- A generic collision library. The implementation is intentionally specialized to a segment swept by a sphere over a fixed-diagonal heightfield.

---

## 2. File Map and Ownership

- Create `g1_clearance.h`: public status, result, witness, budget, point/sphere/capsule/foot/pose, history, command-materializer, and actual-center swing-validation declarations. It contains no certified arithmetic implementation.
- Create `g1_clearance.cpp`: strict-FP environment validation, sole-command materialization, certified endpoint expansions, terrain enumeration, interval arithmetic, prism-patch solver, bounded fallback, aggregators, and actual-center swing validation. For representable point queries it consumes Task 2's `G1SurfaceQueryStatus`/`g1_surface_query_v2` fail-closed status before reconstructing the continuous triangle certificate.
- Create `tests/cpp/test_g1_clearance.cpp`: analytic, adversarial, transaction, budget, strict-FP, and parity tests.
- Modify later `g1_ik_runtime.h` in Task 7: own the immutable lift ladder and staged controller transaction, consume strict-kernel statuses and binary64 lower bounds, and never reimplement clearance.
- Modify later controller/test build commands: compile `g1_clearance.cpp` separately without fast math, compile callers with their existing flags, then link the two objects.
- Modify later stop-reason/checker work only if it needs to distinguish `outside-domain`, `budget-exceeded`, or `uncertified-clearance`; do not launder these outcomes into a successful clearance.

Ownership is one-way:

```text
baseline controller state + immutable lift-bit ladder
                    |
                    v
 staged production sole command + IK/orientation/FK
                    |
        actual world-space sphere-center bits
                    |
                    v
       strict g1_clearance.cpp transaction
                    |
      certified status + actual-center margin
                    |
 first passing staged pose or rollback/safe-stop
                    |
       accepted-only swing-history commit
```

The clearance kernel never receives controller state and cannot mutate history.

---

## 3. Public Status and Data Contract

Add these declarations to `g1_clearance.h`:

```cpp
#pragma once

#include "g1_ik.h"

#include <cstdint>

enum G1ClearanceStatus : uint8_t
{
    G1ClearanceOk = 0,
    G1ClearanceOutsideDomain,
    G1ClearanceBudgetExceeded,
    G1ClearanceUncertified,
    G1ClearanceInvalidInput,
    G1ClearanceInvalidField,
    G1ClearanceArithmeticFailure
};

struct G1ClearanceWitness
{
    double body_x = 0.0;
    double body_y = 0.0;
    double body_z = 0.0;
    double surface_x = 0.0;
    double surface_y = 0.0;
    double surface_z = 0.0;
    double segment_parameter = 0.0;
    double terrain_weight_0 = 0.0;
    double terrain_weight_1 = 0.0;
    double terrain_weight_2 = 0.0;
    uint32_t primitive_index = 0;
    int32_t cell_x = 0;
    int32_t cell_z = 0;
    uint32_t terrain_triangle_index = 0;
    uint32_t patch_index = 0;
    uint32_t candidate_kind = 0;
    uint32_t candidate_subindex = 0;
};

struct G1ClearanceWork
{
    uint32_t point_queries = 0;
    uint32_t cells_visited = 0;
    uint32_t primitive_triangle_pairs = 0;
    uint32_t face_patches = 0;
    uint32_t candidate_tests = 0;
    uint32_t subdivision_nodes = 0;
};

struct G1ClearanceResult
{
    // Guaranteed lower_bound_m <= the true signed vertical clearance.
    // witness_upper_m outwardly encloses the value of the feasible parameter
    // witness and is therefore >= the true minimum. Safety decisions use
    // lower_bound_m only.
    double lower_bound_m = 0.0;
    double witness_upper_m = 0.0;
    G1ClearanceWitness witness;
    G1ClearanceWork work;
};

struct G1ClearanceBudget
{
    uint32_t maximum_point_queries;
    uint32_t maximum_cells;
    uint32_t maximum_primitive_triangle_pairs;
    uint32_t maximum_face_patches;
    uint32_t maximum_candidate_tests;
    uint32_t maximum_subdivision_nodes;
};

constexpr uint32_t G1ClearanceMaximumPointQueriesPerPose = 16;
constexpr uint32_t G1ClearanceMaximumCellsPerPrimitive = 512;
constexpr uint32_t G1ClearanceMaximumCellsPerPose = 2048;
constexpr uint32_t G1ClearanceMaximumCellsPerSwingFoot = 256;
constexpr uint32_t G1ClearanceMaximumPairsPerPrimitive = 1024;
constexpr uint32_t G1ClearanceMaximumPairsPerPose = 4096;
constexpr uint32_t G1ClearanceMaximumPairsPerSwingFoot = 512;
constexpr uint32_t G1ClearancePatchesPerPair = 8;
constexpr uint32_t G1ClearanceCandidatesPerPair = 32;
constexpr uint32_t G1ClearanceMaximumSubdivisionNodes = 8192;
constexpr double G1ClearanceMaximumCertificateWidthM = 1.0e-6;

G1ClearanceBudget g1_pose_clearance_budget();
G1ClearanceBudget g1_swing_foot_clearance_budget();

G1ClearanceStatus g1_apply_swing_lift_y(
    float& output_y,
    float input_y,
    float lift_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_point_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 point,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_sphere_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 center,
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_capsule_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_foot_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const vec3 sphere_centers[4],
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_swept_foot_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const vec3 previous_centers[4],
    const vec3 current_centers[4],
    float radius_m,
    char* error,
    int error_capacity);
```

Later pose and swing structs use `double` for every clearance or margin involved in a safety decision. The clearance layer validates actual endpoints and does not own a predicted-lift plan:

```cpp
struct G1SwingClearanceValidation
{
    double lower_margin_m = 0.0;
    double witness_upper_m = 0.0;
    bool sweep_evaluated = false;
    G1ClearanceWork work;
};

struct G1LegClearance
{
    G1ClearanceResult knee;
    G1ClearanceResult ankle;
    G1ClearanceResult toe;
    G1ClearanceResult foot;
    G1ClearanceResult thigh;
    G1ClearanceResult shin;
    G1ClearanceResult minimum;
};

struct G1PoseClearance
{
    G1ClearanceResult hips;
    G1LegClearance left;
    G1LegClearance right;
    G1ClearanceResult minimum;
};

G1ClearanceStatus g1_measure_leg_clearance(
    G1LegClearance& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_measure_pose_clearance(
    G1PoseClearance& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity);
```

### Status ownership

| Status | Meaning | Task 6/7 action |
|---|---|---|
| `Ok` | Bounds and witness are valid and within certification width | A staged swing candidate passes clearance only when `lower_bound_m >= 0` |
| `OutsideDomain` | The complete projected body/sweep is not proven inside the inclusive rectangle | Reject this finite stage and continue the fixed ladder; safe-stop if none passes |
| `BudgetExceeded` | A preflight or shared work budget would be exceeded | Reject this finite stage and continue the fixed ladder; safe-stop if none passes |
| `Uncertified` | Finite geometry could not reach the `1e-6 m` certificate-width bound | Reject this finite stage and continue the fixed ladder; safe-stop if none passes |
| `InvalidInput` | Nonfinite/invalid body, radius, configuration, exact-dt, history, or over-factory budget input | Controlled diagnostic and normal cleanup |
| `InvalidField` | Not a structurally valid G1HF/v2 field or a visited cell has invalid heights | Controlled diagnostic and normal cleanup |
| `ArithmeticFailure` | The rounding/gradual-underflow environment is unsupported or a checked strict-FP operation could not produce a finite enclosure | Controlled diagnostic and normal cleanup |

The result output is transactional: construct a local candidate and assign it only for `G1ClearanceOk`. `G1ClearanceBudget` is an immutable set of caller-selected limits, not a mutable ledger; each top-level call owns a private ledger and returns consumed work only through a successful result. Composite calls pass that private ledger through internal `_with_ledger` helpers, so a late failure cannot mutate either the public output or the caller's limits. Error text is diagnostic only and is not used to infer status.

The two budget factories initialize every field explicitly. The pose factory permits 16 point queries, 2048 total cell visits, 4096 pairs, 32768 face patches, 131072 analytic candidate tests, and 8192 subdivision nodes. The swing-foot factory covers one actual four-sphere sweep and permits zero point queries, 256 cell visits, 512 pairs, 4096 face-patch evaluations, 16384 analytic candidate tests, and 8192 subdivision nodes. These are absolute per-call ceilings, not suggestions. A caller-provided `limits` value may tighten any field but may not exceed the factory value for that entry-point family. Standalone point/sphere/capsule/foot and leg/pose measurement use the pose ceiling; swept-foot and actual-center swing validation use the swing ceiling. The runtime ladder's separate fixed 41-stage bound is specified in Section 9 and is not a caller-enlargeable clearance-budget field.

Validate every limit field with ordinary bounded comparisons before checked-work products or ledger creation. If any requested field exceeds its family factory—including `UINT32_MAX`—return `G1ClearanceInvalidInput` transactionally; do not clamp, wrap, or accept an enlarged budget. Zero and other smaller values are valid tightening requests and can subsequently produce `BudgetExceeded`/`Uncertified`. Fixed per-primitive caps are enforced independently and cannot be raised through either factory. Tests compare factory structs field-by-field so an implementation cannot silently inflate a factory to legitimize a hostile request.

`g1_measure_leg_clearance` and `g1_measure_pose_clearance` validate array shapes, indices, local geometry, transforms, and every input before assigning their aggregate structs. Each component result's `work` is the delta consumed by that component. The aggregate `minimum.work` is the complete shared-ledger total. Its lower bound is the minimum component lower bound, while its witness upper and coordinates come from the minimum feasible component witness under the key order below; these two component identities may differ. The pose call uses the exact Y-up transforms and local capsule/foot geometry already owned by `g1_ik.h` and does not duplicate configuration constants.

### Internal certified endpoint contract

The public `vec3` API is a wrapper, not the representation used by the proof. Define these private strict-TU concepts (the `G1Interval` definition is in Section 6):

```cpp
struct G1EndpointSourceKey
{
    uint32_t source_kind;       // 0 public capsule, 1 swing endpoint
    uint32_t primitive_index;
    uint32_t semantic_ordinal;  // canonical endpoint, or prior/current
    uint32_t original_x_bits;
    uint32_t original_y_bits;
    uint32_t original_z_bits;
};

struct G1ExactY
{
    // Canonical nonoverlapping expansion whose exact-real sum is endpoint Y.
    double terms[4];
    uint32_t count;
    G1Interval enclosure;
};

struct G1CertifiedEndpoint
{
    double x;                   // exact promotion of canonical binary32 X
    double z;                   // exact promotion of canonical binary32 Z
    G1ExactY y;
    G1EndpointSourceKey source_key;
};
```

`g1_capsule_clearance` validates/canonicalizes its public binary32 endpoints, promotes XZ exactly, creates a one-term exact Y expansion, derives the outward enclosure, and calls an internal `g1_capsule_clearance_certified`. For endpoint reversal parity, transform each canonical float bit pattern to IEEE total-order form (`negative ? ~bits : bits ^ 0x80000000`), sort by the `(x,y,z)` ordered-bit tuple, then assign public semantic ordinals `0,1`. No proof code converts a `G1CertifiedEndpoint` back to `vec3`.

Actual-center swing validation uses source kind 1, the sphere primitive index, semantic ordinal `0=prior`/`1=current`, and the canonical bits of the prior accepted and staged-FK centers. Form adjusted Y only with error-free `TwoDiff(double(A.y),double(P))` and `TwoDiff(double(B.y),double(W))`. Here `B` is already the controller-produced world-space sphere center; the clearance layer receives no lift key and performs no endpoint prediction. Sum each adjusted expansion outward to obtain `y.enclosure`; never cast either adjusted difference to binary32. Canonicalize the two certified endpoints by `source_key` before patch construction, so patch order and witness keys depend only on actual endpoint bits. The exact expansion defines the real segment used by witnesses; its interval drives conservative lower arithmetic.

`g1_apply_swing_lift_y` is a separate non-inline transactional sole-command helper, not an endpoint-proof constructor. It performs the Section 6 arithmetic-environment check; validates/canonicalizes runtime `input_y`; requires finite, nonnegative `lift_m <= 0.08f` and canonicalizes either zero sign to `+0.0f`; materializes `RN32(double(input_y)+double(lift_m))` with one strict binary64-to-binary32 round; canonicalizes a zero result to `+0.0f`; and rejects a nonzero subnormal or nonfinite result. Positive subnormal lift inputs retain their bits during the operation, but the helper does **not** promise that every positive lift succeeds: for example, `input_y=+0.0f` plus `denorm_min` materializes a forbidden nonzero-subnormal command and returns `ArithmeticFailure` with `output_y` unchanged. Production ladder entries above zero are normal. Invalid arguments return `InvalidInput`, while an unrepresentable materialized result returns `ArithmeticFailure`; either leaves `output_y` unchanged. The staged controller must call this helper for the actual desired sole command and record its output bits before IK.

### Bound and witness tie-breaking

The global lower bound and witness are deliberately independent:

- Aggregate `lower_bound_m` as the minimum of every patch lower bound.
- Aggregate `witness_upper_m` as the minimum value among all certified feasible witness candidates.
- The witness coordinates always belong to `witness_upper_m`; they do not claim to realize the lower bound.
- If two witness upper values have identical binary64 bits, choose the lexicographically smaller stable key:

```text
(primitive_index, cell_z, cell_x, terrain_triangle_index,
 patch_index, candidate_kind, candidate_subindex)
```

- Traverse cells in `z`, then `x`; terrain triangle `T0`, then `T1`; patches `0..7`; interior before edges `0..2`; edge stationary candidate before its lower and upper endpoints.
- Canonicalize signed zero, build `G1CertifiedEndpoint` values, and order them by `G1EndpointSourceKey` before forming patches. The public wrapper's source-key construction makes reversal bit-identical; swing ordering remains stable even though adjusted Y is an exact expansion with no binary32 encoding.

Candidate kinds are fixed: `0=face-interior`, `1=edge`, `2=fallback-node`, `3=point`. For kind 1, `candidate_subindex=3*edge_index+local`, where local `0=stationary`, `1=lower endpoint`, and `2=upper endpoint`. For kind 2 it is the deterministic subdivision-node preorder index; for kind 0 or 3 it is zero. These numeric values are part of parity output and must not depend on pointer order.

The internal witness proof record's `segment_parameter`, canonical endpoint source keys, and three nonnegative terrain weights are the authoritative feasibility data. The weights are binary64 values interpreted as exact reals; normalize them by their positive exact-real sum, which defines barycentrics that sum to one without relying on a rounded `1-w0-w1`. Require `t in [0,1]` and a positive finite weight sum before accepting a witness. Reconstruct centerline Y from the two `G1ExactY` expansions at that exact parameter and enclose it outward; reconstruct XZ from the promoted exact coordinates. Then require the horizontal disk inequality by an outward upper enclosure, choose a downward-enclosed square-root magnitude so the sphere offset is inside the closed ball, and upward-enclose the resulting vertical value. The public parameter/XYZ members are deterministic binary64 diagnostics reconstructed from the same internal source record; they are never used to re-prove or tighten the bound. Point calls use `t=0` and the selected triangle weights. Primitive indices are assigned before traversal: a standalone primitive is zero and a standalone foot uses spheres `0..3`; a standalone leg uses knee `0`, ankle `1`, toe `2`, foot spheres `3..6`, thigh `7`, shin `8`; a pose uses Hips `0`, left knee/ankle/toe `1..3`, left foot spheres `4..7`, left thigh/shin `8..9`, right knee/ankle/toe `10..12`, right foot spheres `13..16`, and right thigh/shin `17..18`.

### Float output semantics

- `lower_bound_m`, `witness_upper_m`, pose clearances, and swing margins remain `double` through controller decisions and logging.
- Print these fields with a double conversion such as `%.17g`; do not first store them in `float`.
- If a legacy visualization requires a float lower bound, provide a strict-TU helper that rounds toward negative infinity. If a nonpositive value would become zero or a subnormal under FTZ, return `-FLT_MIN`; if a positive value would become subnormal, returning `0.0f` is conservative. That float is display-only and cannot feed acceptance.

---

## 4. Exact Capsule-Triangle Geometry

Let `T` be one closed terrain triangle, `S=[A,B]` a capsule centerline, and `r>0` its radius. The required signed vertical clearance is:

```text
minimum body Y minus terrain Y at identical XZ
```

For `s in S`, `p in T`, and a sphere offset `e` with `|e| <= r`, define `d=s-p`. Equal body/terrain XZ requires `e.xz=-d.xz`. For a fixed feasible `d`, the lowest sphere offset is:

```text
e.y = -sqrt(r*r - dot(d.xz, d.xz))
```

Thus the exact real-geometry objective is:

```text
C(S,T,r) = min d.y - sqrt(r*r - |d.xz|^2)
             d in D=S-T, |d.xz| <= r
```

`D=S-T` is the affine image of `triangle x segment`, a triangular prism. With terrain vertices `P0,P1,P2`, define:

```text
ai = A - Pi
bi = B - Pi
```

Here `A.y`/`B.y` mean the exact-real sums of their `G1ExactY` expansions. Difference-space XZ coordinates come from promoted exact endpoint/terrain coordinates; each difference-space Y retains an exact expansion, its outward interval, and its endpoint source key. This representation change does not alter `D`, its boundary, or the eight-patch proof.

The image of the prism boundary is covered by exactly these eight triangular patches:

```text
patch 0: (a0, a1, a2)
patch 1: (b0, b1, b2)

for terrain edges (0,1), (1,2), (2,0):
    patch: (ai, aj, bj)
    patch: (ai, bj, bi)
```

No orientation or outward-normal test is necessary because all eight patches are evaluated.

### Coverage proof

Parameterize the source prism by two terrain barycentrics and one segment parameter. If its affine map to `D` has rank three, an interior feasible point cannot minimize because decreasing only `d.y` decreases the objective while keeping the disk constraint unchanged; the minimum lies on the prism boundary. If the map has rank below three—including `A==B`, a segment parallel to the terrain plane, or a fully coplanar segment—each nontrivial affine fiber through a source-prism point reaches the source boundary, so every point in `D` has a boundary preimage. Therefore the eight patches cover a minimizer in every rank case.

A sphere calls the same function with `A==B`. A sphere center moving linearly from prior to current sweeps exactly `S plus Ball(r)`, so continuous swing clearance is the same finite-capsule problem.

---

## 5. Projected Patch and Edge Minimization

For a patch whose XZ projection has a determinant interval that excludes zero, the exact-real patch has a plane:

```text
y = a*x + b*z + c
```

over the projected triangle intersected with the closed disk `x*x+z*z <= r*r`. The objective is:

```text
f(x,z) = a*x + b*z + c - sqrt(r*r - x*x - z*z)
```

It is convex. Its minimum is exhausted by one interior stationary candidate plus the three projected patch edges.

Compute outward intervals for `a`, `b`, and `c` from the endpoint Y expansions; every displayed formula below is interval-evaluated for a lower bound. Feasible upper witnesses instead reconstruct the exact-real source expansion at the selected parameter and round only the final enclosure/diagnostic. The projected determinant and disk clipping depend only on exact promoted XZ, so adjusted-Y intervals never change patch topology or patch order.

### Interior candidate

```text
n = sqrt(1 + a*a + b*b)
x_star = -r*a/n
z_star = -r*b/n
f_star = c - r*n
```

- If outward interval barycentrics certify `(x_star,z_star)` inside the closed projected patch, `f_star` is the patch minimum candidate.
- If they certify it outside, only patch edges can minimize.
- If membership straddles zero, include an outward lower enclosure of `c-r*n` as a conservative patch lower bound and obtain feasible upper witnesses from the edges or bounded subdivision.

### Edge candidate

For a projected edge with nonzero horizontal length, choose unit direction `e`, write its line as `u=q+e*s` with `dot(q,e)=0`, and let:

```text
R = sqrt(r*r - dot(q,q))
```

Intersect the edge's ordered `s` interval with `[-R,+R]`. Along this line the 3D edge height is `C+m*s`, so:

```text
g(s) = C + m*s - sqrt(R*R - s*s)
s_star = -m*R / sqrt(1 + m*m)
```

Clamp `s_star` to the feasible interval and evaluate it. Also interval-evaluate both feasible endpoints so uncertain clamp classification cannot omit the minimum. When the projected edge length is zero, XZ is constant and the smaller endpoint Y is the exact edge minimizer.

Every determinant, disk intersection, barycentric sign, and clamp comparison is three-way: certified inside, certified outside, or uncertain. Exact zero/tangency is retained as inside. Lower-bound evaluation uses the outward disk/edge intersection; an upper witness uses only an inward-certified scalar parameter and the Section 3 feasibility check. No geometric epsilon converts an uncertain sign into inside or outside. A denominator or radicand interval straddling its invalid boundary routes to the bounded fallback.

A disk-circle arc strictly inside a patch cannot minimize: moving radially inward makes the negative square-root term smaller faster than the finite affine term can increase. Circle/triangle endpoints are already included by the edge intervals.

### Feasible witness reconstruction

Store, for each patch vertex, the source triple `(certified endpoint source key, segment parameter, terrain weights)`. Projected patch barycentrics reconstruct both a centerline parameter and normalized terrain weights without collapsing endpoint Y expansions. Accept them as a witness only through the feasibility procedure in Section 3. Set diagnostic `body_x=surface_x` and `body_z=surface_z` from one shared rounding of the reconstructed terrain point; obtain body Y with a downward-enclosed sphere magnitude so the exact parameter witness stays inside the capsule. Upward-evaluate body Y minus the continuous fixed-diagonal plane Y at that same parameter point for `witness_upper_m`; do not derive a witness from a lower-bound-only interval. The separate output-rounding guard below applies only to the certified lower bound, so witness feasibility is never changed by quantizing its XZ coordinates or adjusted Y to binary32.

---

## 6. Outward Intervals, Degeneracies, and Surface Rounding

### Required arithmetic environment

The certified kernel uses IEEE round-to-nearest binary64 operations followed by `nextafter` to enclose each exact operation. It uses only `+`, `-`, `*`, `/`, and `sqrt`.

`g1_clearance.cpp` begins with:

```cpp
#ifdef __FAST_MATH__
#error "g1_clearance.cpp must be compiled without fast math"
#endif

#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif

static_assert(sizeof(float) == 4, "G1 clearance requires binary32");
static_assert(sizeof(double) == 8, "G1 clearance requires binary64");
static_assert(std::numeric_limits<float>::is_iec559,
              "G1 clearance requires IEEE-754 float");
static_assert(std::numeric_limits<double>::is_iec559,
              "G1 clearance requires IEEE-754 double");
static_assert(std::numeric_limits<float>::has_denorm == std::denorm_present,
              "G1 clearance requires binary32 gradual underflow");
static_assert(std::numeric_limits<double>::has_denorm == std::denorm_present,
              "G1 clearance requires binary64 gradual underflow");
```

Implement one non-inline strict-TU `g1_clearance_arithmetic_environment_is_supported()` and call it before input-dependent arithmetic in every public `G1ClearanceStatus` entry. Pure budget factories and history reset/commit only copy/validate binary32 state and do not invoke the certified kernel, so their existing non-status contracts remain environment-independent. The check performs all of these steps:

1. Require `std::fegetround() == FE_TONEAREST`.
2. On x86/SSE, read MXCSR with `_mm_getcsr()` and reject bit 15 (FTZ) or bit 6 (DAZ). Use the literal masks so the DAZ check does not depend on optional vendor macros.
3. Run no-inline volatile binary32 and binary64 probes. For each type, multiply `numeric_limits<T>::min()` by `0.5`, add `denorm_min()` to itself, and copy the materialized bits with `memcpy`. Require binary32 results `0x00400000` and `0x00000002`, and binary64 results `0x0008000000000000` and `0x0000000000000002`. The multiply catches flushed results and the addition catches denormal operands even on targets without MXCSR access.

A failed check returns `G1ClearanceArithmeticFailure` transactionally before ledger mutation or field access. The check observes but never changes the environment. Build the strict object with `-frounding-math` as well as contraction and fast-math disabled. Link without `-ffast-math`; on toolchains that add a fast-math startup object at link time, a fast link driver could enable FTZ/DAZ process-wide even though `g1_clearance.cpp` itself was compiled strictly.

Represent an interval as:

```cpp
struct G1Interval
{
    double lower;
    double upper;
};
```

Each primitive operation materializes its binary64 result, rejects nonfinite endpoints, then expands outward by one `nextafter` step. Interval multiplication evaluates all four endpoint products. Division rejects a denominator interval containing zero. Square root rejects a negative upper endpoint, clamps a lower endpoint that is at most zero to exact zero, and expands the upper result toward positive infinity. A feasible witness uses the downward square-root endpoint and rechecks `rho_squared + vertical_offset_squared <= r_squared` with an outward upper interval; if that check is inconclusive, it is not a witness. Compile with FP contraction disabled so an operation assumed by the proof cannot become a fused/reassociated expression.

### Terrain height and float-output semantics

The geometric quantity bracketed by `lower_bound_m` and `witness_upper_m` is defined against the continuous affine plane induced by the stored binary32 node heights promoted to binary64. The certified kernel never minimizes a collection of rounded point samples. `heightfield_sample_v2` is nevertheless allowed to round its binary64 plane result to binary32 and to canonicalize every binary32 subnormal or signed zero to `+0.0f`, so a point diagnostic must not claim more clearance than that sibling query reports at the same representable XZ.

For each visited triangle, derive a mandatory conservative output guard from its vertex-height range. It is at least one full binary32 ULP at the greatest finite magnitude in that range and at least `FLT_MIN` whenever the range contains or can round through the subnormal band; also include the outward binary64 interpolation-error enclosure. Compute the ULP on both sides of every crossed binade boundary and take the maximum. Subtract this guard from the continuous lower bound only. This covers upward height rounding and subnormal-to-zero canonicalization; a bare `ulp(0)` is specifically insufficient. At every representable point probe with valid producer sample `q`, enforce the observable invariant `lower_bound_m <= double(body_y) - double(q.height)`. The feasible witness and `witness_upper_m` remain values on the continuous triangle, so both public bounds still bracket one well-defined continuous minimum. The guard is one-sided safety slack, not a claim that a float-rounded point-sample surface is continuous.

The mandatory guard is never replaced by a tighter candidate-local estimate. If it is nonfinite, exceeds `G1ClearanceMaximumCertificateWidthM`, or prevents `witness_upper_m-lower_bound_m <= 1e-6`, return `Uncertified` with output unchanged. This deliberately makes high-binade fields fail closed even when one sampled value happens to be exact. A legacy float display uses the downward conversion rule from Section 3 and is not a second geometric result.

### Uncertain or projected-degenerate patch

For a patch triangle with vertices `v0,v1,v2`, a valid coarse lower bound is:

```text
min(v0.y, v1.y, v2.y)
    - sqrt(r*r - rho_min*rho_min)
```

where `rho_min` is the lower endpoint of a certified interval for the minimum horizontal distance from the origin to the projected closed triangle. Use the lower enclosure of vertex Y and the upper enclosure of the square root. Skip the patch only when the distance interval's lower endpoint is strictly greater than the upper enclosure of `r`; retain exact tangency. The bound is safe because Y is affine over the patch and the square-root term is maximized at the minimum horizontal radius, even when those extrema occur at different points.

Each fallback node carries all three exact-real source records `(endpoint source key, segment_parameter, terrain_weights)`, not only rounded difference-space vertices. Its Y bounds are rebuilt from endpoint expansions at every split. Attempt a feasible upper witness at every node using the interval-certified closest projected point; reject that attempt if the disk or normalized-weight checks are inconclusive. When the coarse lower and best feasible upper differ by more than `1e-6 m`, choose the longest edge by the outward upper enclosure of squared 3D difference-space length, breaking overlap/ties by edge index `0,1,2`. Split its source parameters at the exact dyadic midpoint, reconstruct both child triangles from source records, and push children in `(lower_bound, stable_node_key)` order. This longest-edge bisection makes the difference-space diameter converge; if all edge lengths are certified zero, the coarse and feasible values must coincide or the call fails `ArithmeticFailure` rather than looping. Stop with `G1ClearanceUncertified` before consuming subdivision node `8193`. Rank-zero and rank-one XZ projections are therefore handled without dividing by a near-zero plane determinant.

### Certification invariant

For every top-level `Ok` result:

```text
lower_bound_m <= true minimum <= witness_upper_m
witness_upper_m - lower_bound_m <= 1e-6
```

If the interval remains wider, return `Uncertified`; never replace the lower bound with the witness or midpoint merely to obtain a narrow diagnostic.

---

## 7. Exact G1HF/v2 Bounds and Triangle Enumeration

At every public status entry, first run the Section 6 arithmetic-environment check. Every entry that accepts a budget then validates the entire caller limit struct against its family factory before field access or ledger creation; `g1_apply_swing_lift_y` instead performs the scalar validation specified in Section 3 and never creates a ledger. After those preconditions, geometry entries:

1. Require `field.version==2` and `terrain_heightfield_is_queryable(field)`.
2. Require every body coordinate to pass Task 2's `g1_ik_vec3_is_runtime_value`; canonicalize signed zero. Nonzero binary32 subnormals are `InvalidInput`, including Y.
3. Require finite positive-normal radius for sphere/capsule calls.
4. Compute the full projected AABB with outward binary64 intervals:

```text
minimum_x = min(A.x,B.x) - r
maximum_x = max(A.x,B.x) + r
minimum_z = min(A.z,B.z) - r
maximum_z = max(A.z,B.z) + r
```

5. Compute authoritative maximum X/Z with the exact operation sequence used by `terrain_v2_locate_cell`: first materialize the binary64 product `(count-1)*double(cell_size)`, then materialize `double(origin)+product`. Disable contraction and compare the resulting binary64 boundary by bits in the boundary fixtures.
6. Form every footprint minimum and maximum as an error-free two-component expansion of the promoted binary32 endpoint and radius (`TwoDiff` for a lower extreme, `TwoSum` for an upper extreme). Compare expansions directly: lower must be greater than or equal to the exact binary64 origin and upper must be less than or equal to the exact binary64 maximum. Equality succeeds. A strict excursion returns `OutsideDomain`; a nonfinite expansion or comparison that cannot be established returns `ArithmeticFailure`. Do **not** quantize a footprint extreme to binary32 for this proof: the v2 maximum node need not itself be binary32-representable, and outward float rounding would incorrectly reject exact tangency there.
7. Derive conservative cell-index spans directly from the expansion endpoints with outward binary64 division/floor intervals. Compare floored endpoints against `0` and `count-1` in binary64 and prove they fit before any integer cast; perform the final conversion through checked `uint64_t`, never a float-derived unchecked `int`. Include both neighboring cells when a bound is exactly on a grid line or its floor interval is ambiguous; duplicate closed-edge evaluation is safe. Clamp indices only after Step 6 proves the complete footprint is in-domain. Use `terrain_v2_locate_cell` unchanged for the binary32 point API and as a fixture cross-check, not as the continuous-footprint containment predicate.
8. Use checked `uint64_t` multiplication for the rectangular cell count. Return `BudgetExceeded` before entering a loop if the primitive would exceed 512 cells or 1024 terrain-triangle pairs.
9. For each cell, call `terrain_v2_cell_heights`; a bad selected cell is `InvalidField`, not exterior terrain.
10. Construct source-coordinate vertices with the exact promoted origin/index/cell rule and emit:

```text
T0 = (p00, p10, p11)
T1 = (p00, p11, p01)
```

Point clearance first calls `g1_surface_query_v2` after separately validating input and field. Map `G1SurfaceQueryOutside` to `G1ClearanceOutsideDomain`; after those prechecks, map `G1SurfaceQueryInvalid` to `InvalidField` unless the strict reconstruction identifies an arithmetic failure. On `G1SurfaceQueryValid`, use the same checked locate/cell-height sequence to compute the selected continuous triangle plane in the strict kernel. Apply the height-output guard to its lower bound and use the unguarded continuous value as its feasible upper witness. A test at every representable fixture probe compares this continuous value with the producer sample within the declared output-rounding guard. Neither path consults `exterior_height` as geometry.

The clearance tests must build two otherwise identical fields with different exterior heights and require bit-identical in-domain results plus identical `OutsideDomain` statuses.

---

## 8. Work Budgets, Complexity, and Caching

The public budget is an immutable tightening request under an absolute factory cap. One private ledger is shared by all internal primitives in a pose or swing-foot call; every charge is checked before work occurs, and counters fail rather than wrap. A public standalone primitive also creates a private ledger, so the same implementation path is tested without exposing partially consumed state on failure. Over-factory validation happens before this ledger exists.

| Scope | Hard cap |
|---|---:|
| Checked point queries per pose | 16 |
| Broad-phase cells per primitive | 512 |
| Broad-phase cell visits per complete pose | 2048 |
| Broad-phase cell visits per actual swing-foot validation | 256 |
| Primitive/terrain-triangle pairs per primitive | 1024 |
| Primitive/terrain-triangle pairs per complete pose | 4096 |
| Primitive/terrain-triangle pairs across four spheres of one swing foot | 512 |
| Prism patches per pair | 8 |
| Analytic candidate tests per pair | 32 |
| Subdivision nodes per top-level call | 8192 |

Hard analytic work bounds are:

```text
complete pose: 4096 * 32 = 131072 candidate tests
one actual four-sphere sweep: 512 * 32 = 16384 candidate tests
all 41 staged ladder entries for one foot: at most 41 * 16384 = 671744 candidate tests
fallback: at most 8192 additional subdivision nodes per strict validation
```

Counter units are exact:

- One `cell` charge means one unique `(primitive,cell_z,cell_x)` broad-phase entry inserted into one actual-center validation's call-local cache. A different primitive crossing the same terrain cell is a separate visit. Terrain heights may be physically deduplicated behind that accounting.
- One `primitive_triangle_pair` charge means one primitive/closed-triangle pair; both fixed-diagonal triangles are separate pairs.
- One `face_patch` charge means one of the eight top-level prism-boundary triangles evaluated for one pair.
- One `candidate_test` charge means one analytic solver unit: the face stationary classification or one complete edge minimization. An edge unit includes its stationary value and two clipped endpoints, so there are at most four units per patch and exactly 32 preflightable units per pair.
- One `subdivision_node` charge covers one fallback child node, including its constant-size coarse bound, closest-feasible-point witness attempt, and deterministic split. Fallback work is not charged again as analytic candidate work.

The pose and swing factories set immutable shared ceilings from Section 3; effective limits are exactly the validated caller values, never `max(caller,factory)`. Fixed per-primitive limits of 512 cells and 1024 pairs are checked before those shared limits. Preflight the complete rectangular cell count, its two-triangle pair count, eight patches per pair, and four analytic units per patch with checked `uint64_t` arithmetic before the first terrain load or ledger mutation. Subdivision is charged incrementally because it is data-dependent; reaching its cap without the required width returns `Uncertified`, while a caller-supplied cap too small for already preflightable fixed work returns `BudgetExceeded`.

At the published `0.02 m` cell size, expected broad-phase counts are:

- stationary `r=0.02` sphere: approximately 8 to 18 terrain-triangle pairs;
- ordinary four-sphere fixed-update sweep: approximately 80 to 160 pairs;
- complete thigh/shin/foot pose: commonly 500 to 1500 pairs;
- one ordinary staged swing candidate: approximately 2500 to 5000 candidate tests;
- a lift-needed selection through ladder index `i`: approximately `(i+1)*2500` to `(i+1)*5000` candidate tests;
- pose diagnostics: commonly 16000 to 50000 candidate tests.

Each ladder entry has separately staged IK/FK endpoints, including potentially different XZ bits, so terrain spans and pairs are recomputed for that entry. Never reuse the old assumption that lift changes only endpoint Y. The runtime sums successful per-candidate `G1ClearanceWork` fields with checked `uint64_t` intermediates for diagnostics; the immutable 41-entry ladder and per-call factory caps bound total work even when every candidate fails.

Do not cache across candidates or scene generations. Do not allocate a field-sized acceleration structure in Task 5.

---

## 9. Actual-Center Swing Certification and Staged Lift Ladder

Add checked history operations:

```cpp
struct G1SwingHistory
{
    bool initialized = false;
    vec3 previous_sphere_centers[4];
};

bool g1_swing_history_reset(
    G1SwingHistory& output,
    const vec3 sphere_centers[4],
    char* error,
    int error_capacity);

bool g1_swing_history_commit(
    G1SwingHistory& history,
    const vec3 accepted_sphere_centers[4],
    char* error,
    int error_capacity);

G1ClearanceStatus g1_swing_clearance_validate(
    G1SwingClearanceValidation& output,
    const G1ClearanceBudget& limits,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const vec3 final_sphere_centers[4],
    bool recorded_contact,
    float dt,
    char* error,
    int error_capacity);
```

Validation canonicalizes all four actual centers into a local candidate before assignment. Commit requires initialized history. A failed call leaves every prior component unchanged.

For one sphere, let `A` be its prior accepted center, `B` its actual staged-FK center, `P` planted clearance, and `W` swing clearance. The target interpolates linearly, so subtract its endpoints from the actual centerline endpoints:

```text
A_adjusted.xz = exact_promote(A.xz)
A_adjusted.y  = exact_expansion(TwoDiff(double(A.y), double(P)))
B_adjusted.xz = exact_promote(B.xz)
B_adjusted.y  = exact_expansion(TwoDiff(double(B.y), double(W)))
```

Only the target subtraction is represented as an exact-real expansion; it must never pass through a binary32 adjusted-Y temporary. The certified capsule clearance of `[A_adjusted,B_adjusted]` against zero is exactly the continuous target-subtracted margin for the actual staged controller endpoint. Aggregate all four adjusted certified capsules. `g1_swing_clearance_validate` has no lift argument: every staged candidate and the defensive final check use `L=+0` because `B` already contains all controller and IK effects. Require initialized history, runtime prior/current centers, valid configuration, and `g1_ik_dt_is_exact_25_hz(dt)` before the contact shortcut. Recorded contact returns `Ok` with `sweep_evaluated=false`; otherwise a nonnegative `lower_margin_m` is the only clearance pass.

The controller policy in `g1_ik_runtime.h` is an immutable 2 mm ladder represented by checked-in bits, never by fast-math multiplication:

```cpp
constexpr uint32_t G1SwingLiftCandidateCount = 41;
constexpr uint32_t G1SwingNoCandidate = UINT32_MAX;
constexpr uint32_t G1SwingLiftCandidateBits[G1SwingLiftCandidateCount] = {
    0x00000000u, 0x3b03126fu, 0x3b83126fu, 0x3bc49ba6u,
    0x3c03126fu, 0x3c23d70au, 0x3c449ba6u, 0x3c656042u,
    0x3c83126fu, 0x3c9374bcu, 0x3ca3d70au, 0x3cb43958u,
    0x3cc49ba6u, 0x3cd4fdf4u, 0x3ce56042u, 0x3cf5c28fu,
    0x3d03126fu, 0x3d0b4396u, 0x3d1374bcu, 0x3d1ba5e3u,
    0x3d23d70au, 0x3d2c0831u, 0x3d343958u, 0x3d3c6a7fu,
    0x3d449ba6u, 0x3d4ccccdu, 0x3d54fdf4u, 0x3d5d2f1bu,
    0x3d656042u, 0x3d6d9168u, 0x3d75c28fu, 0x3d7df3b6u,
    0x3d83126fu, 0x3d872b02u, 0x3d8b4396u, 0x3d8f5c29u,
    0x3d9374bcu, 0x3d978d50u, 0x3d9ba5e3u, 0x3d9fbe77u,
    0x3da3d70au,
};

struct G1SwingCandidateDiagnostic
{
    uint32_t candidate_index = G1SwingNoCandidate;
    uint32_t lift_bits = 0;
    uint32_t materialized_command_y_bits = 0;
    uint32_t actual_sphere_center_bits[4][3] = {}; // [sphere][x,y,z], raw FK call-boundary bits
    G1ClearanceStatus clearance_status = G1ClearanceInvalidInput;
    bool controller_constraints_passed = false;
    bool clearance_certified = false;
    double lower_margin_m = 0.0;
    double witness_upper_margin_m = 0.0;
    G1ClearanceWork clearance_work;
};

struct G1SwingSelectionDiagnostic
{
    uint32_t candidates_evaluated = 0;
    uint32_t selected_index = G1SwingNoCandidate;
    G1SwingCandidateDiagnostic selected;
    G1ClearanceWork total_clearance_work;
};

#if defined(G1_IK_ENABLE_TEST_SEAMS)
bool g1_ik_stage_swing_candidate_for_test(
    G1SwingCandidateDiagnostic& output,
    const G1IkState& state,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> scratch_rotations,
    const slice1d<int> parents,
    const heightfield& field,
    uint32_t foot_index,
    uint32_t candidate_index,
    float dt,
    char* error,
    int error_capacity);
#endif
```

Entry `i` is exactly `RN32(i/500 m)`, so entries `0` and `40` are exact `+0.0f` and `0.08f`. The checked-in table is the production source of truth; an independent exact-rational test oracle verifies all 41 bits and strict increase. Two millimetres is below the existing 5 mm end-effector residual scale and keeps the worst case to 41 actual controller stages, close to the prior bounded work envelope. This is only a command-resolution and work-budget choice. The design claims no continuous minimum, no monotonic post-IK response, and no guarantee that a pass between ladder entries will be represented; absence of a passing listed entry fails closed.

Recorded contact does not enter the ladder: it leaves `candidates_evaluated=0` and `selected_index=G1SwingNoCandidate`, with the recorded-contact flag distinguishing that shortcut from an all-41-fail safe stop. The sole production engine for one swing entry is the private `g1_ik_stage_swing_candidate`; both the full selector and the test-only wrapper call it. For each non-contact foot, the runtime selection transaction does the following:

1. `static_assert` the table count, endpoint bits, and strict unsigned-bit order. Validate state, shapes, field, exact dt, and `g1_foot_runtime_config_validate(config)`, including that `config.max_swing_lift_m` bits equal table entry 40, before mutating any caller-owned value. Snapshot one immutable per-foot scratch input. Failed ladder entries never feed rotations, targets, normals, or state into later entries.
2. Visit indices `0..40` in that exact order. Reconstruct the listed lift by bits and call `g1_apply_swing_lift_y` on the **actual** `desired_sole_center.y`; record the lift and materialized command bits. No fast-math caller spells the addition.
3. Starting from a fresh copy of the same scratch input, run the identical production branch logic, invoking the bounded position-IK and orientation functions in the identical order whenever their production predicates are true, with the identical reach, correction, residual, and invariance checks. There is no proxy solve or clearance-only pose.
4. Run forward kinematics on that staged pose, compute the four configured world-space foot-sphere centers, record all twelve raw bits at the validation call boundary, then call `g1_swing_clearance_validate` on those same values with a fresh per-call swing budget. The strict public-input contract canonicalizes signed zero internally; the diagnostic does not rewrite what FK produced.
5. A candidate passes only when every controller constraint passes, clearance status is `Ok`, and its binary64 lower margin is nonnegative. A finite controller rejection, negative `Ok` margin, `OutsideDomain`, `BudgetExceeded`, or `Uncertified` rejects only that candidate and advances to the next index. `InvalidInput`, `InvalidField`, or `ArithmeticFailure` aborts the frame transaction unchanged.
6. Select the first passing candidate, move its already-evaluated staged rotations/result into the outer scratch pose, and copy its public diagnostic. Do not rerun IK to apply the selection. If all 41 candidates are finitely rejected, assign a safe-stop frame result with `candidates_evaluated=41`, `selected_index=G1SwingNoCandidate`, default ignored selected fields, and no accepted pose/history/state mutation. Do not fabricate a continuous required-lift field.

The two feet compose sequentially only in the outer scratch pose; each foot's failed stages remain isolated. After both feet are staged, recompute final FK once, require every selected foot's four center bit triples to match its stored staged endpoints, and run a mandatory fresh `g1_swing_clearance_validate` with `L=+0` for each foot. Any mismatch or failed defensive certificate rolls back the entire frame. This recheck is a defensive invariant for future cross-foot changes, not the first proof of a predicted endpoint.

`G1FootFrameResult::swing_selection` lets separate-object integration tests observe selected index, candidate lift bits, actual materialized sole-command bits, and actual staged/FK endpoint bits without exposing `G1CertifiedEndpoint` or solver records. If the existing `applied_swing_lift_m` log field is retained during schema migration, it is reconstructed by bit copy from `swing_selection.selected.lift_bits`, is diagnostic only, and must never apply a second correction or rerun the solve. Under `G1_IK_ENABLE_TEST_SEAMS` only, `g1_ik_stage_swing_candidate_for_test` takes the normal frame inputs plus `foot_index` and `candidate_index`, calls `g1_ik_stage_swing_candidate`, discards the private staged rotations, and returns only `G1SwingCandidateDiagnostic`. A finite candidate rejection returns `true` with its diagnostic; a global input/field/arithmetic error returns `false` with `output` unchanged. Tests probe all indices with that wrapper, derive the first passing index from real staged outcomes, and compare the full selector's public diagnostic. Release builds contain neither the wrapper declaration nor symbol; no private geometry or mutable controller state crosses the seam.

---

## 10. Transaction and Safe-Stop Semantics

- Geometry functions take no history or controller state.
- `g1_apply_swing_lift_y` changes only its scalar output and only on `Ok`; the controller uses it solely to materialize each staged desired-sole command.
- `g1_swing_clearance_validate` accepts `const G1SwingHistory&`, receives only actual FK sphere centers, and assigns only a local `G1SwingClearanceValidation` on `Ok`.
- The lock observer may advance only in the outer scratch transaction according to its existing policy. Every ladder stage copies that same scratch input; a rejected stage cannot advance locks, swing history, or the next stage's pose.
- The first passing stage is moved into the outer scratch pose without rerunning IK. Task 7 commits final rendered rotations and sphere centers only after endpoint-bit equality, the defensive actual-center certificate, planted thresholds, pose-capsule thresholds, and all other IK bounds are accepted.
- Candidate-local `OutsideDomain`, `BudgetExceeded`, or `Uncertified` leaves that validation output unchanged and rejects that stage. Exhausting the finite ladder assigns only a safe-stop frame diagnostic; it does not assign an accepted pose or infer an unrepresented lift.
- `InvalidInput`, `InvalidField`, or `ArithmeticFailure` return through controlled diagnostic and normal cleanup.
- On finite rejection, accepted local/global pose arrays, accepted clearance, and both histories remain bit-identical to the prior accepted state. Public diagnostics identify only the selected stage or the all-fail sentinel; private rejected poses are never exposed as if accepted.
- On a returned controlled error, public outputs and the complete `G1IkState` argument remain unchanged.

Add a distinct stop reason for unresolved certified clearance if later checker evidence cannot represent `OutsideDomain` or `BudgetExceeded`. Do not report a fabricated continuous lift requirement; log the selected ladder index/lift bits or `G1SwingNoCandidate`.

---

## 11. Strict-FP Build and Link Contract

`g1_clearance.cpp`, including the sole-command materializer, is non-inline and compiled separately. The release controller and staged IK caller may retain `-ffast-math`; the kernel may not. The immutable ladder is reconstructed only by the existing `memcpy`-based bit helper, never arithmetic or type-punning, and all lift addition occurs inside the strict helper.

### Strict unit test

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp -o /tmp/g1_clearance_strict.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance_strict.o
g++ /tmp/test_g1_clearance_strict.o /tmp/g1_clearance_strict.o \
  -o /tmp/test_g1_clearance_strict
/tmp/test_g1_clearance_strict
```

### Fast-math caller with strict kernel

```bash
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  -c tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance_release.o
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. \
  -c g1_clearance.cpp -o /tmp/g1_clearance_release_kernel.o
g++ /tmp/test_g1_clearance_release.o \
  /tmp/g1_clearance_release_kernel.o \
  -o /tmp/test_g1_clearance_release
/tmp/test_g1_clearance_release
```

The final link command intentionally has no `-ffast-math`. Preserve that separation in build scripts: fast math is a compile-only caller flag, because some GCC-family drivers link a startup object that enables FTZ/DAZ when the flag reaches the link step. The release executable's first clearance test records MXCSR where available and runs the portable gradual-underflow probe before comparing parity output.

Separate-object runtime tests compile `tests/cpp/test_g1_ik.cpp` with `-DG1_IK_ENABLE_TEST_SEAMS`, compile `g1_clearance.cpp` with the same strict kernel flags above, and link the two objects with a strict link driver. Repeat with a fast-math test caller. The macro exposes only the diagnostic wrapper in Section 9; the production controller build omits it, and a negative compile that names the wrapper without the macro must fail. Tests never include `g1_clearance.cpp` directly or inspect private endpoint/solver types.

### Sanitizer caller and kernel

```bash
g++ -std=c++17 -O1 -g -fno-fast-math -ffp-contract=off \
  -frounding-math \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c g1_clearance.cpp -o /tmp/g1_clearance_san_kernel.o
g++ -std=c++17 -O1 -g -fno-fast-math \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance_san.o
g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all \
  /tmp/test_g1_clearance_san.o /tmp/g1_clearance_san_kernel.o \
  -o /tmp/test_g1_clearance_san
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/test_g1_clearance_san
```

### Controller implication

Replace every later one-command controller build with separate objects:

```bash
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  -c controller.cpp -o /tmp/controller_g1_ik.o
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG \
  -I. -c g1_clearance.cpp -o /tmp/g1_clearance_controller.o
g++ /tmp/controller_g1_ik.o /tmp/g1_clearance_controller.o \
  -o /tmp/controller_g1_ik \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Do not add `-flto` or pass `-ffast-math` to a final link command. Add a negative compile guard that compiling `g1_clearance.cpp` with `-ffast-math` fails at the `#error`, and run fixture J in every linked executable.

---

## 12. TDD RED Fixture Matrix

All tests use `check`, not `assert`, so `-DNDEBUG` cannot remove them. Each failure-path test seeds outputs and history with distinct finite bit patterns and verifies no mutation.

### A. Smooth-plane sphere false-clear

Build a `9x9` field with binary32 `cell=0.02f`. Choose a positive binary32 height increment `delta` nearest `sqrt(3)*cell` with its low three mantissa bits cleared; verify that every `float(i)*delta`, `i=0..8`, is exact, and store that value in every node of column `i`. The resulting continuous v2 surface is exactly coplanar with binary64 slope `g=double(delta)/double(cell)`, which is close to `sqrt(3)` without pretending that an irrational slope or decimal inputs are exactly binary32.

Place a sphere with promoted radius `r=double(0.02f)` at an interior grid node. Set center Y by the specified binary32 construction near `h(center.x)+0.039`, then compute the oracle from the **actual promoted input bits**:

```text
oracle = double(center.y) - h(center.x) - r*sqrt(1 + g*g)
```

Require `oracle` to be negative and within `2e-6` of `-0.001`, require the returned interval to contain that oracle, and require width at most `1e-6`. Independently run the rejected lattice formula on the same promoted bits and require it to report positive clearance near `+0.00436 m`; do not hard-code an ideal-decimal value as the certified oracle.

### B. Finite capsule plane oracle

On a multi-cell coplanar field `h=g*x+k*z+b`, compare against:

```text
min(L(A),L(B)) - r*sqrt(1 + g*g + k*k)
L(P) = P.y - g*P.x - k*P.z - b
```

Cover vertical, oblique, terrain-coplanar, zero-length, and reversed endpoints. Reversal must produce bit-identical bounds, work, and witness key.

### C. Fixed diagonal and constrained edge

Use one cell with heights `{h00=0,h10=2,h01=4,h11=10}`. Test support points strictly inside each fixed triangle. At center XZ `(0.5,0.5)` and radius `0.05`, both unconstrained plane support points cross the diagonal, so the constrained optimum lies on the diagonal with support term:

```text
r*sqrt(1 + (10/sqrt(2))^2) = r*sqrt(51)
```

Require the interval to contain the corresponding analytic value. A bilinear surface or opposite diagonal must fail.

### D. Exact bounds and no exterior fallback

- A sphere whose footprint is exactly tangent to min X/Z and max X/Z succeeds.
- Repeat max-edge tangency on an axis whose materialized binary64 maximum node is not binary32-representable; this catches accidental footprint quantization through `terrain_v2_locate_cell`.
- Moving its center one binary32 ULP outward returns `OutsideDomain` and preserves output.
- Positive/negative nonzero subnormal XZ input returns `InvalidInput`.
- v1, wrong shape, null storage, and an invalid selected height return `InvalidField`.
- Two fields differing only in exterior height produce bit-identical in-domain results and identical outside statuses.

### E. Actual staged swing ladder

Reuse the near-`sqrt(3)` exactly coplanar terrain, but do not construct a lifted-sphere oracle. For the strict-kernel test, pass named prior and **actual current** four-sphere center bits directly to `g1_swing_clearance_validate`; compare the target-subtracted sweep with a high-precision oracle based on those endpoint bits. Repeat with only one sphere obstructed and require the public witness primitive index to identify it.

For the controller integration fixture, use `G1_IK_ENABLE_TEST_SEAMS` to run the real staging function for indices `0..40` from the same immutable input. Require at least one finite rejection before a pass, record the first candidate whose controller constraints and actual-center lower margin pass, then run the full selector and require:

- indices were attempted contiguously from zero through the selected index;
- `candidates_evaluated == selected_index+1`;
- selected lift bits, all twelve actual sphere-center bits, binary64 margins, and work match that probed stage, while materialized command-Y bits also match an independent public-helper call on the fixture's actual unlifted desired sole Y and selected lift bits;
- rebuilding the four `vec3` values from the diagnostic endpoint bits and calling the separate strict validation API reproduces the selected clearance status and margin bits;
- final output rotations produce those same sphere-center bits without another IK call;
- the mandatory fresh `L=+0` defensive validation passes.

Use a valid wall fixture for which all 41 real stages finitely reject. Require exact order through index 40, `G1SwingNoCandidate`, safe-stop, and bit-identical accepted rotations, state, histories, support, matcher, and simulation values. This fixture makes no claim about a continuous lift above or between ladder entries.

Keep the prior rounding trap as two explicitly separate public regressions. First, call the actual sole-command helper with input-Y bits `B=0x3d0f5c28` and non-ladder test lift `L=0x31000001`; require materialized command bits `0x3d0f5c29`, while `L=0x31000000` materializes back to `B`. This proves command rounding only and is not a selection oracle. Second, feed actual current sphere-center Y bits `C=0x3d0f5c29`, `W=0x3c75c28f`, `r=0x3ca3d70a`, and flat terrain `h=0x30000000` directly to actual-center validation. Make the prior endpoint safely higher and require the exact endpoint margin:

```text
double(C) - double(W) - double(r) - double(h)
         = +4.656612873077392578125e-10

double(float(C - W)) - double(r) - double(h)
         = -4.656612873077392578125e-10
```

Require the fixture's complete terrain/output guard to remain below the positive margin and the public validation lower bound to stay nonnegative. No test reads private `G1CertifiedEndpoint`, source-key, expansion, or solver fields. The paired tests fail an independently rounded controller addition or a binary32 adjusted target subtraction without pretending that the command-Y bits predict post-IK sphere-center bits.

### F. Rank and projection degeneracies

Exercise `A==B`, segment parallel to terrain, segment in the terrain plane, vertical projected patches, zero-horizontal-length edges, exact disk tangency, and a stationary point whose barycentric interval straddles a patch edge. Require either an `Ok` interval of width at most `1e-6` or the exact fail-closed `Uncertified` status at the subdivision cap; success with a wider interval is forbidden.

### G. Count and arithmetic adversaries

- Decode `cell=0x3b1efa48`, start `(0,0)`, and stop `(0x3b6e7765,0x36723088)`. The removed float lattice computes `spacing=float(0.5*cell)`, `float(length/spacing)==3.0f`, and centerline samples `q0=(0,0)`, `q1=(0x3a9efa44,0x35a175b0)`, `q2=(0x3b1efa44,0x362175b0)`, and `q3=stop`. Its first two rounded gaps are `0.001212903429334061 m`, strictly greater than half-cell `0.0012129032984375954 m`.
- Build an explicit `5x5` fixed-diagonal tent with origin bits X=`0xbb8b1b00`, Z=`0xbb9ef53c`, the same cell, node `(2,2)` height `0x3b83126f` (`0.004f`), and every other node `+0.0f`. Use radius bits `0x381efa48` (`cell/64`) and constant endpoint Y bits `0x3b543300`. The apex is `(0.0006064511835575104, 0.0000006016343832015991)`, inside the first old gap and only `1.51e-10 m` from the segment.
- Embed a test-local copy of the removed `segment_steps`/radial lattice, not production code. Its radial spacing admits only the center offset and its four samples return minimum clearance greater than `+0.00019 m` (oracle approximately `+0.00019999896`). The apex supplies a feasible capsule witness below `-0.00079 m` (oracle approximately `-0.00080000030`). Require the certified call to return `Ok`, width at most `1e-6`, and `witness_upper_m < -0.00079`; this proves an old positive versus certified negative result on the fixed diagonal.
- Minimum-positive-normal cell size, `FLT_MAX` endpoints/radius, endpoint subtraction overflow, and poisoned previous history must return promptly without conversion UB or loops.
- An AABB spanning exactly 512 cells/1024 closed-triangle pairs may proceed at the fixed caps; 513 cells/1026 pairs returns `BudgetExceeded` before the first terrain load.
- Give the 512-cell fixture a caller limit of only 1023 pairs and require `BudgetExceeded` before pair 1. This tests the pair boundary without inventing an odd pair count that fixed two-triangle cells cannot produce.
- For both budget families, increment each factory field one at a time and also set it to `UINT32_MAX`; every call returns `InvalidInput` before field access with output/history/limits unchanged. Exact factory structs and tightened values remain accepted as inputs.
- Exhausting candidate or subdivision budget returns `BudgetExceeded`/`Uncertified` with no counter wrap.

### H. Exact timing and cap boundaries

- `1.0f/25.0f` succeeds.
- Both adjacent `nextafter` values fail transactionally.
- Independently derive `RN32(i/500)` for all 41 indices: assert index 0 is exact positive zero; for indices `1..40`, decode each table value and its adjacent binary32 values as integer-mantissa/power-of-two rationals, cross-multiply their distances to `i/500` with checked `uint64_t`, and apply ties-to-even. Require the checked-in table to match bit-for-bit, increase strictly in index order, and end at exact `0x3da3d70a` (`0.08f`).
- Require the validated configuration maximum bits to equal entry 40. Reject an out-of-range index transactionally. The production selector evaluates no unlisted float and always includes index 40 before declaring all-fail; there is no `+1e-6` cap hole or hidden candidate 41.

### I. Build and parity

- Construct parity inputs from named `uint32_t` float bits rather than fast-math-compiled decimal/square-root expressions. The strict and fast-math-caller executables emit a compact line containing status, bound bits, witness bits/key, and all work counters; compare the lines byte-for-byte.
- For runtime ladder tests, require deterministic candidate order and repeated-run diagnostics within each build. Strict and release controller math may produce different actual IK endpoint bits; each build must expose and certify its own endpoints, while identical endpoint-bit inputs to the strict kernel remain byte-identical across callers.
- A direct `-ffast-math -c g1_clearance.cpp` command must fail and mention the certified-kernel guard.
- Strict, release-caller, ASan/UBSan, and repeated-run hashes all pass.

### J. Arithmetic-environment rejection and restoration

- Seed the lift helper's scalar output and every geometry/swing status output plus surrounding history with distinct bits. Under an RAII guard that snapshots `fegetround`, set `FE_UPWARD` and `FE_DOWNWARD` in turn; table-drive every public `G1ClearanceStatus` entry and require `ArithmeticFailure`, no ledger/field access, and no output/history mutation. Restore the original mode before ordinary numeric assertions and prove one subsequent valid call succeeds.
- On x86/SSE, use a second RAII guard that snapshots the exact MXCSR word. Set FTZ bit 15, DAZ bit 6, and both bits in separate cases; require the same transactional `ArithmeticFailure` from a geometry call and a swing call. Restore the original MXCSR word on every scope exit, including failed checks, and verify it bit-for-bit before continuing. Other targets skip only direct MXCSR mutation, never the portable baseline probe.
- Exercise the portable volatile multiply/add probes in the normal environment and require their four expected subnormal bit encodings. Tests must never leave the process rounding mode or denormal mode changed for later parity executables.
- With the normal gradual-underflow environment, pass `input_y=+0.0f` and positive `lift_m=denorm_min` to `g1_apply_swing_lift_y`; require transactional `ArithmeticFailure` because the materialized command is a forbidden nonzero subnormal. Separately require the `0x31000001` normal-result command fixture from E to succeed. Positive subnormal input is preserved during evaluation, not promised to yield a valid command.

### K. Mandatory float-output guard

- At representable XZ, use `h00=1.0f`, `h10=nextafterf(1.0f,+infinity)`, and `tx=0.75f` to force upward binary32 height rounding. For this and every case below, obtain `G1SurfaceSample q` from `g1_surface_query_v2` and assert `result.lower_bound_m <= double(body_y)-double(q.height)`.
- Cover zero/FTZ semantics with `h00=-FLT_MIN,h10=+0.0f,tx=0.5f` (a negative subnormal continuous height canonicalized upward to `+0.0f`) and the positive mirror `h00=+FLT_MIN`. A guard based only on `ulp(0)` must fail these REDs.
- Cross both sides of the `1.0f` and `2.0f` binade boundaries using `{nextafterf(B,0),B}` vertex pairs and probes on each triangle/diagonal side. Require the guard to use the larger adjacent ULP and preserve the producer-height inequality.
- Use a flat `16.0f` field, whose mandatory full ULP exceeds `1e-6`. Point, sphere, and capsule calls must return `Uncertified` with seeded outputs unchanged; exact local interpolation is not permission to discard the mandatory guard.

---

## 13. Staged Implementation Tasks and Review Gates

### Task 1: Lock the public status, ownership, and strict-TU boundary

**Files:**
- Create: `g1_clearance.h`
- Create: `g1_clearance.cpp`
- Create: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces the exact enums, structs, constants, and declarations from Section 3.
- Produces a strict-TU compile guard and stub status-name helper.

- [ ] Write compile-time tests for enum values, binary64 bound types, exact factory constants, the `g1_apply_swing_lift_y` signature, `has_denorm==denorm_present` for both types, and non-copying const-history signatures.
- [ ] Compile the test before creating the files; require a missing-header RED.
- [ ] Add RED fixture J for `FE_UPWARD`, `FE_DOWNWARD`, FTZ, DAZ, portable denormal bits, environment restoration, and transactional outputs.
- [ ] Add over-factory and `UINT32_MAX` REDs for every pose/swing budget field.
- [ ] Add the declarations, immutable factory-ceiling validation, strict-TU `__FAST_MATH__`/IEEE/gradual-underflow guards, and no-inline runtime environment probe.
- [ ] Compile the strict caller/kernel objects successfully.
- [ ] Compile `g1_clearance.cpp` with `-ffast-math`; require the intentional guard failure.
- [ ] Link the fast-math caller object with a strict link driver and prove MXCSR remains gradual-underflow before the first call.
- [ ] Commit only the three Task 1 files with `feat: define certified G1 clearance contract`.

**Review gate:** No implementation is accepted while a public safety field is `float`, a non-`Ok` result can partially assign output, a caller can enlarge a factory cap, FTZ/DAZ can enter certified arithmetic, or the strict kernel can inline into the caller.

### Task 2: Add checked v2 domain spans and terrain triangles

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces internal `g1_clearance_domain_contains`, checked cell-span enumeration, and exact `T0/T1` vertex construction.
- Produces checked point clearance.

- [ ] Add RED tests D and K, the noncoplanar point probes from C, malformed visited heights, and unchanged outputs.
- [ ] Implement structural/query-domain validation using existing bit helpers.
- [ ] Implement outward footprint/domain comparison with exact tangency handling.
- [ ] Preflight cell-span products and budgets before loops.
- [ ] Emit exact fixed-diagonal triangles, mandatory per-triangle output guards, and checked point results satisfying the producer-height inequality.
- [ ] Run strict, fast-math-caller, and sanitizer tests.
- [ ] Commit with `feat: enumerate checked G1HF v2 clearance triangles`.

**Review gate:** Exterior height cannot influence status or in-domain output; exact maximum edges, both triangle halves, upward/FTZ output rounding, binade transitions, and mandatory-guard `Uncertified` behavior must be demonstrated by tests.

### Task 3: Implement outward intervals and analytic patch solver

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces strict interval operations, `G1ExactY`/`G1CertifiedEndpoint`, projected face coefficients, interior stationary solver, disk-clipped 3D edge solver, feasible witness reconstruction, and stable source/candidate keys.

- [ ] Add RED tests for public exact-Y behavior, endpoint reversal, analytic face interior, each edge clamp branch, circle tangency, vertical projection, membership uncertainty, witness feasibility, and stable public witness keys. Do not reach into private endpoint/source records from the separately compiled test.
- [ ] Implement outward interval primitives with nonfinite/zero-denominator rejection.
- [ ] Implement `TwoSum`/`TwoDiff` expansion canonicalization, endpoint wrappers, and source-key ordering from canonical actual endpoint bits with no adjusted-Y downcasts.
- [ ] Implement face stationary formulas and interval barycentric classification.
- [ ] Implement the edge formulas and vertical-edge case.
- [ ] Implement lower/witness aggregation with the Section 3 key order.
- [ ] Verify every `Ok` result encloses a high-precision fixture oracle within `1e-6`.
- [ ] Commit with `feat: certify projected capsule patch minima`.

**Review gate:** The reviewer must derive the stationary and edge formulas independently, verify that no circle-only candidate is missing, and trace every patch/witness Y back to an exact endpoint expansion and stable source key.

### Task 4: Compose the eight-patch capsule kernel and bounded fallback

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces `g1_capsule_clearance` and `g1_sphere_clearance` with exact prism-boundary coverage.

- [ ] Add RED fixtures A, B, F, the explicit G tent/old-lattice sign reversal, K sphere/capsule cases, and endpoint-reversal parity.
- [ ] Construct the two caps and six side triangles in the fixed patch order.
- [ ] Evaluate all patches without outward-normal classification.
- [ ] Implement the coarse patch lower bound and deterministic longest-edge subdivision.
- [ ] Enforce certificate width and subdivision budgets.
- [ ] Add the mandatory per-triangle v2 height-rounding guard; guard width above `1e-6` returns `Uncertified` unchanged.
- [ ] Run strict/release/sanitizer tests and repeated hash parity.
- [ ] Commit with `feat: certify G1 sphere and capsule clearance`.

**Review gate:** The reviewer must check the rank-three boundary argument and rank-deficient fiber argument. The named tent must be old-lattice positive and certified-witness negative; sampled radial or centerline production code is an automatic rejection.

### Task 5: Add foot and pose aggregation under shared budgets

**Files:**
- Modify: `g1_clearance.h`
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces four-sphere foot clearance and binary64 `G1LegClearance`/`G1PoseClearance` diagnostics.

- [ ] Add RED tests for the one-worst-sphere public witness primitive index, both legs, every diagnostic member, tightened shared-pair exhaustion, over-factory/`UINT32_MAX` rejection, and transactional pose failure.
- [ ] Aggregate lower and witness values independently as specified.
- [ ] Thread one checked budget through every point/sphere/capsule primitive.
- [ ] Preserve exact G1 local geometry and Y-up world transforms from `g1_ik.h`.
- [ ] Run all focused modes.
- [ ] Commit with `feat: aggregate certified G1 pose clearance`.

**Review gate:** Every logged threshold source must be a binary64 lower bound, no caller limit may exceed the family factory, and a late right-leg error must leave the entire caller output unchanged.

### Task 6: Add actual-center swept-foot certification

**Files:**
- Modify: `g1_clearance.h`
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces checked swing history, strict sole-command materialization, four-capsule validation of actual world-space endpoints, and defensive `L=+0` revalidation.

- [ ] Add the strict-kernel portions of RED fixture E, including the separate `0x31000001` command-materialization and actual-endpoint adjusted-downcast traps, plus G, exact-dt/contact-shortcut validation, failed reset/commit, subnormal-command rejection, and rejected-history snapshots.
- [ ] Implement transactional `g1_apply_swing_lift_y` as the one strict round of the actual sole command; document and test that a positive subnormal lift input can still yield rejected non-runtime output.
- [ ] Build prior/current `G1CertifiedEndpoint` Y only as `TwoDiff(actual_y,target_y)`, never call the public `vec3` capsule wrapper internally, and never round the adjusted difference.
- [ ] Implement exact-dt validation before contact branching.
- [ ] Implement `g1_swing_clearance_validate` for four actual final centers with a fresh one-sweep budget and no lift argument or prediction.
- [ ] Run all focused modes and compare strict/release result lines byte-for-byte.
- [ ] Commit with `feat: certify actual G1 swing endpoints`.

**Review gate:** Tests observe only public results. The command helper must materialize `0x3d0f5c29` for the named non-ladder input, actual endpoint subtraction must retain the positive exact margin, and no strict-kernel API may accept a lift key as a proxy for post-IK sphere endpoints.

### Task 7: Stage and select actual controller candidates

**Files:**
- Modify later: `g1_ik_runtime.h`
- Modify later: `controller.cpp`
- Modify later: every terrain-IK native build command/checker affected by the new object
- Test: `tests/cpp/test_g1_ik.cpp`
- Test: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Consumes Task 6's strict materializer, actual-center validator, and history APIs.
- Produces the immutable 41-entry ladder, diagnostic-only test seam, first-passing staged pose selection, defensive recheck, and controller rejection/cleanup without changing matching, support, or accepted history ownership.

- [ ] Add integration RED tests for the exact 41-entry bit table/endpoints/order, out-of-range index, materialized command bits, actual sphere endpoint bits, every status mapping, and accepted/rejected snapshots.
- [ ] Under `G1_IK_ENABLE_TEST_SEAMS`, probe every real staged candidate in a shallow-obstacle fixture; require the full selector to choose the first passing probe with identical diagnostic bits and `candidates_evaluated=selected_index+1`.
- [ ] Add an all-41-fail wall fixture requiring `G1SwingNoCandidate`, safe-stop, and bit-identical accepted pose/state/history. Require selected output rotations and endpoint bits to equal the probed first-passing stage; inspect the call graph to prove selected IK is not rerun.
- [ ] Replace header-only calls with the public non-inline API.
- [ ] Implement the one private production function `g1_ik_stage_swing_candidate`: copy the same scratch input, bit-load the indexed lift, call `g1_apply_swing_lift_y` on the actual desired sole target, run the existing bounded IK/orientation/FK path, record actual endpoint bits, and validate those endpoints. The test seam must forward to this function rather than reproduce it.
- [ ] Iterate indices `0..40`, move the first passing staged pose into outer scratch without rerunning IK, and never spell `target_y + lift` in the fast-math caller. Do not use binary search, predecessor inference, or a continuous required-lift field.
- [ ] After both feet compose, require stored/final endpoint-bit equality and rerun the mandatory fresh actual-center `L=+0` certificate before any accepted commit.
- [ ] Add selected-index/lift/command/endpoint diagnostics to `G1FootFrameResult`; update later logs/checkers to consume those fields instead of treating a continuous lift requirement as measured evidence.
- [ ] Update strict, release, sanitizer, and controller link commands exactly as Section 11.
- [ ] Add the negative fast-math-kernel build guard and arithmetic-environment matrix to verification; link without a fast-math startup object.
- [ ] Run the complete verification matrix below.
- [ ] Commit only integration-owned files with `build: link certified G1 clearance kernel`.

**Review gate:** Inspect the staged-function call graph, selected-pose move, final endpoint-bit check, link command, object flags, and entry MXCSR/environment evidence. A predicted `sphere_y+lift`, private-field test assertion, rerun-to-apply path, noncontiguous ladder visit, fast kernel flag, or inherited FTZ/DAZ is an automatic rejection.

---

## 14. Verification Matrix

Run after each relevant task and in full after Task 7:

| Mode | Caller flags | Kernel flags | Required evidence |
|---|---|---|---|
| Strict | `-O2 -Wall -Wextra -Werror -pedantic` | same plus `-fno-fast-math -ffp-contract=off -frounding-math` | Exit 0, no warnings; nearest/gradual probes pass |
| Release caller | `-O3 -ffast-math -DNDEBUG` | `-O3 -fno-fast-math -ffp-contract=off -frounding-math -DNDEBUG` | Strict link driver; entry MXCSR gradual; identical endpoint-bit kernel fixtures byte-equal to strict |
| Sanitizer | `-O1 -g`, ASan/UBSan with halt-on-error | strict FP plus same sanitizers | Exit 0, no report |
| Negative build | any | `-ffast-math` | Compile fails at `__FAST_MATH__` guard |
| FP environment | strict and release objects | strict kernel | `FE_UPWARD`, `FE_DOWNWARD`, FTZ, and DAZ each return transactional `ArithmeticFailure`; RAII restores exact environment |
| Determinism | repeat each controller build 20 times | strict kernel | Identical per-build ladder order, selected diagnostic, status/bound/witness/work hash |
| Transaction | strict and release | strict kernel | Seeded accepted pose/state/history exact after each rejected stage, all-41 fail, and all non-`Ok` paths |
| Budget | strict and sanitizer | strict kernel | Exact/tightened limits accepted; every factory+1 and `UINT32_MAX` field is transactional `InvalidInput` before work |
| Endpoint precision | strict and release | strict kernel | Non-ladder `L=0x31000001` materializes command Y `0x3d0f5c29`; separately supplied actual endpoint Y has positive exact adjusted margin while float-adjusted is negative; public reversal output matches |
| Staged ladder | strict and release controller builds | strict kernel | All 41 exact bits/order pass; selector equals first real staged pass; selected command/endpoint bits match; all-fail rolls back and safe-stops |
| Output rounding | strict and sanitizer | strict kernel | Upward, zero/subnormal, and binade probes obey producer-height lower inequality; guard `>1e-6` is unchanged-output `Uncertified` |
| Spacing ridge | strict and release | strict kernel | Local removed lattice `>+0.00019`; certified tent witness `<-0.00079` |
| Geometry | all successful modes | strict kernel | Analytic oracle enclosed; width `<=1e-6` |

Final source/order guards:

```bash
! rg -n 'ceil\(|radial_steps|segment_steps|half.*cell.*sample' \
  g1_clearance.cpp g1_clearance.h
rg -n '#error.*fast math|has_denorm|_mm_getcsr|G1CertifiedEndpoint|g1_apply_swing_lift_y|TwoDiff|G1ClearancePatchesPerPair' \
  g1_clearance.cpp g1_clearance.h
rg -n 'G1SwingLiftCandidateBits|G1SwingCandidateDiagnostic|actual_sphere_center_bits|g1_ik_stage_swing_candidate_for_test' \
  g1_ik_runtime.h tests/cpp/test_g1_ik.cpp
! rg -n 'ordered.*lift|sphere.*\+.*lift|required_lift_m' \
  g1_clearance.cpp g1_clearance.h g1_ik_runtime.h
git diff --check
```

Expected: no production lattice sample-count or predictive-lift code exists; strict-FP/gradual-underflow, actual-endpoint, mandatory-output-guard, immutable-ladder, and absolute-budget guards are present; every test mode exits zero; identical endpoint-bit kernel parity lines match byte-for-byte; every environment mutation is restored; and the worktree is clean after task-scoped commits.

---

## 15. Reconciliation Gate Before Editing the Active Plan

Read-only reconciliation against integrated Task 2 commit `6287a0e` locks these producer names: `G1SurfaceQueryStatus`, `g1_surface_query_v2`, `g1_ik_vec3_is_runtime_value`, `g1_ik_dt_is_exact_25_hz`, and `g1_foot_runtime_config_validate`. This design consumes those names directly and preserves their fail-closed `Valid`/`Outside`/`Invalid` distinction; it does not add an adapter that restores exterior fallback or tolerant `25 Hz` checks. The active terrain-IK plan remains untouched on this branch. At implementation start, confirm that later integration has not renamed these producers before changing either plan.

The same read-only review marks the active plan's pre-IK predictive swing-planner call, direct desired-sole-Y lift increment, and continuous-required-lift checker assertions as superseded contracts. When the approved replacement is applied, remove those snippets and migrate logs/checkers to the selected ladder diagnostic; do not adapt the direct addition or preserve it beside the staged path. Until that replacement gate, the protected active plan remains unchanged.

Before replacing active Task 5, require two approvals:

1. **Geometry review:** prism proof, eight-patch construction, analytic stationary/edge formulas, degeneracy fallback, and sqrt(3) RED are accepted.
2. **Runtime review:** separate strict-FP object with nearest/gradual environment checks, finite staged controller ladder, actual FK endpoint diagnostics/certificates, absolute budget ceilings, mandatory output guards, status mapping, selected-stage reuse, accepted-only history commit, and controller build implications are accepted.

Only then replace the sampled Task 5 text and update all downstream one-command build invocations.
