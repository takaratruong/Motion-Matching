# G1 Certified Clearance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sampled G1 sphere, capsule, and swept-foot clearance with a deterministic, mathematically conservative G1HF/v2 certificate that fails closed at the terrain boundary and under bounded-work exhaustion.

**Architecture:** One strict-floating-point translation unit computes signed vertical clearance for a finite capsule centerline against each fixed-diagonal terrain triangle. A sphere is the zero-length capsule case, and each linearly swept foot sphere is the same capsule primitive. The kernel reduces a capsule/triangle pair to eight triangular patches on a Minkowski-difference prism boundary, minimizes each patch analytically with outward-rounded interval guards, and uses bounded deterministic subdivision only for numerically degenerate patches.

**Tech Stack:** C++17; existing `g1_ik.h`, `terrain_runtime.h`, `vec.h`, and `array.h`; IEEE-754 binary32/binary64; a non-inline `g1_clearance.cpp` compiled with strict FP; standalone strict, fast-math-caller, sanitizer, and parity test executables.

## Global Constraints

- Do not edit the active `docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md` on this design branch. Section 15 records the read-only reconciliation with integrated Task 2.
- Consume the existing G1HF/v2 node, cell, diagonal, query-domain, output-rounding, and exterior semantics. Do not introduce bilinear terrain interpolation or a second heightfield format.
- The authoritative cell triangles are closed `T0=(p00,p10,p11)` for `tx >= tz` and `T1=(p00,p11,p01)` for `tx < tz`; evaluating both closed triangles on their shared diagonal is permitted because their heights agree there.
- Never use `heightfield_sample_v2` returning a finite exterior value as proof that a query was in-domain or that the field was valid.
- Full projected sphere/capsule footprint containment in the inclusive authoritative node rectangle is required. Exact tangency is in-domain; any proven excursion or unresolved boundary comparison fails closed.
- Store safety bounds as binary64. Do not round a safety decision through binary32.
- The certified arithmetic and ordered-lift search must not be compiled under `-ffast-math`, reassociation, contraction, or LTO into a fast-math caller.
- Exactly `dt = 1.0f / 25.0f` remains the swing-history timing contract. Call the Task 2 producer helper `g1_ik_dt_is_exact_25_hz`; do not add a second tolerant comparison.
- `0.08f` is the exact maximum swing-only endpoint lift. Failure to certify the corrected sweep at exactly that value requests safe stop.
- A returned error or finite rejected candidate cannot mutate swing history, accepted pose arrays, accepted clearance, support state, matcher state, or simulation state.
- Do not allocate or iterate in proportion to unchecked field dimensions, radius, segment length, or float-derived sample counts.

---

## 1. Scope and Non-Goals

### In scope

- Checked signed point clearance against an in-domain G1HF/v2 triangle.
- Certified signed vertical clearance of a sphere and a finite capsule segment.
- Continuous linear sweep of each of the four configured G1 foot collision spheres.
- Four-sphere foot aggregation, thigh/shin capsule aggregation, and pose-level minimum diagnostics.
- Endpoint lift planning against a clearance target that interpolates from planted to swing clearance.
- Explicit `OutsideDomain`, bounded-work, numerical-certification, invalid-input, and invalid-field outcomes.
- Deterministic witness selection, diagnostic work counters, strict/release parity, and transactional state ownership.

### Non-goals

- Euclidean separation distance. This plan preserves the existing signed **vertical** clearance definition: lower body height minus authoritative terrain height at the same XZ coordinate.
- Arbitrary curved center trajectories. The fixed-update contract is a linear path between prior accepted and current candidate sphere centers.
- Inferring rigid-foot interpolation from quaternions between updates. The four configured sphere-center paths are the owned sweep geometry.
- Rebuilding G1HF/G1WM artifacts, changing terrain rendering, changing the matcher, changing support retargeting, or changing the controller root.
- Treating the published exterior height as physical terrain outside the authoritative rectangle.
- A generic collision library. The implementation is intentionally specialized to a segment swept by a sphere over a fixed-diagonal heightfield.

---

## 2. File Map and Ownership

- Create `g1_clearance.h`: public status, result, witness, budget, point/sphere/capsule/foot/pose, history, and swing-plan declarations. It contains no certified arithmetic implementation.
- Create `g1_clearance.cpp`: strict-FP terrain enumeration, interval arithmetic, prism-patch solver, bounded fallback, aggregators, and ordered-float lift search. For representable point queries it consumes Task 2's `G1SurfaceQueryStatus`/`g1_surface_query_v2` fail-closed status before reconstructing the continuous triangle certificate.
- Create `tests/cpp/test_g1_clearance.cpp`: analytic, adversarial, transaction, budget, strict-FP, and parity tests.
- Modify the later Task 6 `g1_ik_runtime.h`: consume statuses and binary64 lower bounds; do not reimplement clearance.
- Modify later controller/test build commands: compile `g1_clearance.cpp` separately without fast math, compile callers with their existing flags, then link the two objects.
- Modify later stop-reason/checker work only if it needs to distinguish `outside-domain`, `budget-exceeded`, or `uncertified-clearance`; do not launder these outcomes into a successful clearance.

Ownership is one-way:

```text
validated scene terrain + immutable candidate geometry
                    |
                    v
       strict g1_clearance.cpp transaction
                    |
          status + candidate result
                    |
       Task 6/7 accept or reject candidate
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
    uint32_t lift_evaluations = 0;
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
    uint32_t maximum_lift_evaluations;
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
constexpr uint32_t G1ClearanceMaximumLiftEvaluations = 32;
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

Later pose and swing structs use `double` for every clearance or margin involved in a safety decision. Lift values remain binary32 because they are applied to the existing binary32 pose:

```cpp
struct G1SwingClearancePlan
{
    double baseline_lower_margin_m = 0.0;
    double corrected_lower_margin_m = 0.0;
    double corrected_witness_upper_margin_m = 0.0;
    float required_lift_m = 0.0f;
    float applied_lift_m = 0.0f;
    bool sweep_evaluated = false;
    bool required_lift_certified = false;
    bool safe_stop_requested = false;
    G1ClearanceWork work;
};

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
| `Ok` | Bounds and witness are valid and within certification width | Compare `lower_bound_m` with the required threshold |
| `OutsideDomain` | The complete projected body/sweep is not proven inside the inclusive rectangle | Reject finite candidate, request safe stop, preserve history/output |
| `BudgetExceeded` | A preflight or shared work budget would be exceeded | Reject finite candidate, request safe stop, preserve history/output |
| `Uncertified` | Finite geometry could not reach the `1e-6 m` certificate-width bound | Reject finite candidate, request safe stop, preserve history/output |
| `InvalidInput` | Nonfinite/invalid body, radius, configuration, exact-dt, or history input | Controlled diagnostic and normal cleanup |
| `InvalidField` | Not a structurally valid G1HF/v2 field or a visited cell has invalid heights | Controlled diagnostic and normal cleanup |
| `ArithmeticFailure` | A checked strict-FP operation could not produce a finite enclosure | Controlled diagnostic and normal cleanup |

The result output is transactional: construct a local candidate and assign it only for `G1ClearanceOk`. `G1ClearanceBudget` is an immutable set of caller-selected limits, not a mutable ledger; each top-level call owns a private ledger and returns consumed work only through a successful result. Composite calls pass that private ledger through internal `_with_ledger` helpers, so a late failure cannot mutate either the public output or the caller's limits. Error text is diagnostic only and is not used to infer status.

The two budget factories initialize every field explicitly. The pose factory permits 16 point queries, 2048 total cell visits, 4096 pairs, 32768 face patches, 131072 analytic candidate tests, 8192 subdivision nodes, and zero lift evaluations. The swing-foot factory permits zero point queries, 256 cached cell visits, 512 cached pairs, 131072 face-patch evaluations, 524288 analytic candidate tests, 8192 subdivision nodes, and 32 distinct lift evaluations. Per-primitive caps are checked in addition to these shared totals.

`g1_measure_leg_clearance` and `g1_measure_pose_clearance` validate array shapes, indices, local geometry, transforms, and every input before assigning their aggregate structs. Each component result's `work` is the delta consumed by that component. The aggregate `minimum.work` is the complete shared-ledger total. Its lower bound is the minimum component lower bound, while its witness upper and coordinates come from the minimum feasible component witness under the key order below; these two component identities may differ. The pose call uses the exact Y-up transforms and local capsule/foot geometry already owned by `g1_ik.h` and does not duplicate configuration constants.

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
- Canonicalize signed zero and canonicalize capsule endpoints into lexicographic binary32-bit order before forming patches. Reversing a capsule must therefore produce bit-identical bounds, work counts, and witness tie-breaking.

Candidate kinds are fixed: `0=face-interior`, `1=edge`, `2=fallback-node`, `3=point`. For kind 1, `candidate_subindex=3*edge_index+local`, where local `0=stationary`, `1=lower endpoint`, and `2=upper endpoint`. For kind 2 it is the deterministic subdivision-node preorder index; for kind 0 or 3 it is zero. These numeric values are part of parity output and must not depend on pointer order.

The witness's `segment_parameter` and three nonnegative terrain weights are the authoritative feasibility record. They are binary64 values interpreted as exact reals; normalize the weights by their positive exact-real sum, which defines barycentrics that sum to one without relying on a rounded `1-w0-w1`. Require `t in [0,1]` and a positive finite weight sum before accepting a witness. Reconstruct that exact affine source point with intervals, require the horizontal disk inequality by an outward upper enclosure, choose a downward-enclosed square-root magnitude so the sphere offset is inside the closed ball, and upward-enclose the resulting vertical value. The XYZ members are deterministic nearest-binary64 diagnostics reconstructed from the same parameters; they are never used to re-prove or tighten the bound. Point calls use `t=0` and the selected triangle weights. Primitive indices are assigned before traversal: a standalone primitive is zero and a standalone foot uses spheres `0..3`; a standalone leg uses knee `0`, ankle `1`, toe `2`, foot spheres `3..6`, thigh `7`, shin `8`; a pose uses Hips `0`, left knee/ankle/toe `1..3`, left foot spheres `4..7`, left thigh/shin `8..9`, right knee/ankle/toe `10..12`, right foot spheres `13..16`, and right thigh/shin `17..18`.

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

For a patch whose XZ projection has a determinant interval that excludes zero, solve its plane as:

```text
y = a*x + b*z + c
```

over the projected triangle intersected with the closed disk `x*x+z*z <= r*r`. The objective is:

```text
f(x,z) = a*x + b*z + c - sqrt(r*r - x*x - z*z)
```

It is convex. Its minimum is exhausted by one interior stationary candidate plus the three projected patch edges.

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

Store, for each patch vertex, the source pair `(segment parameter, terrain weights)`. Projected patch barycentrics reconstruct both a centerline parameter and normalized terrain weights. Accept them as a witness only through the feasibility procedure in Section 3. Set diagnostic `body_x=surface_x` and `body_z=surface_z` from one shared rounding of the reconstructed terrain point; obtain body Y with a downward-enclosed sphere magnitude so the exact parameter witness stays inside the capsule. Upward-evaluate body Y minus the continuous fixed-diagonal plane Y at that same parameter point for `witness_upper_m`; do not derive a witness from a lower-bound-only interval. The separate output-rounding guard below applies only to the certified lower bound, so witness feasibility is never changed by quantizing its XZ coordinates to binary32.

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
#include <limits>

static_assert(sizeof(float) == 4, "G1 clearance requires binary32");
static_assert(sizeof(double) == 8, "G1 clearance requires binary64");
static_assert(std::numeric_limits<float>::is_iec559,
              "G1 clearance requires IEEE-754 float");
static_assert(std::numeric_limits<double>::is_iec559,
              "G1 clearance requires IEEE-754 double");
```

Every public entry checks `std::fegetround() == FE_TONEAREST`; any other runtime rounding mode returns `G1ClearanceArithmeticFailure` transactionally. Build the strict object with `-frounding-math` as well as contraction and fast-math disabled. The implementation never changes the caller's floating-point environment.

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

For each visited triangle, derive a conservative output guard from its vertex-height range. It is at least one full binary32 ULP at the greatest finite magnitude in that range and at least `FLT_MIN` whenever the range contains or can round through the subnormal band; also include the outward binary64 interpolation-error enclosure. Subtract this guard from the continuous lower bound only. This covers upward height rounding and subnormal-to-zero canonicalization; a bare `ulp(0)` is specifically insufficient. The feasible witness and `witness_upper_m` remain values on the continuous triangle, so both public bounds still bracket one well-defined continuous minimum. The guard is one-sided safety slack, not a claim that a float-rounded point-sample surface is continuous.

If the guard itself is nonfinite or already exceeds the maximum certificate width, the call may refine to a tighter per-candidate output-error enclosure; if it still cannot establish `witness_upper_m-lower_bound_m <= 1e-6`, return `Uncertified`. Never drop the guard to force success. A legacy float display uses the downward conversion rule from Section 3 and is not a second geometric result.

### Uncertain or projected-degenerate patch

For a patch triangle with vertices `v0,v1,v2`, a valid coarse lower bound is:

```text
min(v0.y, v1.y, v2.y)
    - sqrt(r*r - rho_min*rho_min)
```

where `rho_min` is the lower endpoint of a certified interval for the minimum horizontal distance from the origin to the projected closed triangle. Use the lower enclosure of vertex Y and the upper enclosure of the square root. Skip the patch only when the distance interval's lower endpoint is strictly greater than the upper enclosure of `r`; retain exact tangency. The bound is safe because Y is affine over the patch and the square-root term is maximized at the minimum horizontal radius, even when those extrema occur at different points.

Each fallback node carries all three exact-real source pairs `(segment_parameter, terrain_weights)`, not only rounded difference-space vertices. Attempt a feasible upper witness at every node using the interval-certified closest projected point; reject that attempt if the disk or normalized-weight checks are inconclusive. When the coarse lower and best feasible upper differ by more than `1e-6 m`, choose the longest edge by the outward upper enclosure of squared 3D difference-space length, breaking overlap/ties by edge index `0,1,2`. Split its source parameters at the exact dyadic midpoint, reconstruct both child triangles from source parameters, and push children in `(lower_bound, stable_node_key)` order. This longest-edge bisection makes the difference-space diameter converge; if all edge lengths are certified zero, the coarse and feasible values must coincide or the call fails `ArithmeticFailure` rather than looping. Stop with `G1ClearanceUncertified` before consuming subdivision node `8193`. Rank-zero and rank-one XZ projections are therefore handled without dividing by a near-zero plane determinant.

### Certification invariant

For every top-level `Ok` result:

```text
lower_bound_m <= true minimum <= witness_upper_m
witness_upper_m - lower_bound_m <= 1e-6
```

If the interval remains wider, return `Uncertified`; never replace the lower bound with the witness or midpoint merely to obtain a narrow diagnostic.

---

## 7. Exact G1HF/v2 Bounds and Triangle Enumeration

At every public entry:

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

The public budget is an immutable cap set. One private ledger is shared by all internal primitives in a pose or swing-foot call; every charge is checked before work occurs, and counters fail rather than wrap. A public standalone primitive also creates a private ledger, so the same implementation path is tested without exposing partially consumed state on failure.

| Scope | Hard cap |
|---|---:|
| Checked point queries per pose | 16 |
| Broad-phase cells per primitive | 512 |
| Broad-phase cell visits per complete pose | 2048 |
| Cached broad-phase cell visits per swing foot | 256 |
| Primitive/terrain-triangle pairs per primitive | 1024 |
| Primitive/terrain-triangle pairs per complete pose | 4096 |
| Primitive/terrain-triangle pairs across four spheres of one swing foot | 512 |
| Prism patches per pair | 8 |
| Analytic candidate tests per pair | 32 |
| Ordered-float lift evaluations | 32 |
| Subdivision nodes per top-level call | 8192 |

Hard analytic work bounds are:

```text
complete pose: 4096 * 32 = 131072 candidate tests
one lift-searched swing foot: 512 * 32 * 32 = 524288 candidate tests
fallback: at most 8192 additional subdivision nodes
```

Counter units are exact:

- One `cell` charge means one unique `(primitive,cell_z,cell_x)` broad-phase entry inserted into the call-local cache. Reusing that entry at another lift does not charge it again; a different primitive crossing the same terrain cell is a separate visit. Terrain heights may be physically deduplicated behind that accounting.
- One `primitive_triangle_pair` charge means one cached primitive/closed-triangle pair; both fixed-diagonal triangles are separate pairs.
- One `face_patch` charge means one of the eight top-level prism-boundary triangles evaluated for one pair at one lift.
- One `candidate_test` charge means one analytic solver unit: the face stationary classification or one complete edge minimization. An edge unit includes its stationary value and two clipped endpoints, so there are at most four units per patch and exactly 32 preflightable units per pair.
- One `subdivision_node` charge covers one fallback child node, including its constant-size coarse bound, closest-feasible-point witness attempt, and deterministic split. Fallback work is not charged again as analytic candidate work.
- One `lift_evaluation` charge means one distinct binary32 lift key. Results are memoized; looking up an already evaluated endpoint is not a new evaluation.

The pose and swing factories set shared totals from Section 3. Fixed per-primitive limits of 512 cells and 1024 pairs are checked before those shared limits. Preflight the complete rectangular cell count, its two-triangle pair count, eight patches per pair, and four analytic units per patch with checked `uint64_t` arithmetic before the first terrain load or ledger mutation. Subdivision is charged incrementally because it is data-dependent; reaching its cap without the required width returns `Uncertified`, while a caller-supplied cap too small for already preflightable fixed work returns `BudgetExceeded`.

At the published `0.02 m` cell size, expected broad-phase counts are:

- stationary `r=0.02` sphere: approximately 8 to 18 terrain-triangle pairs;
- ordinary four-sphere fixed-update sweep: approximately 80 to 160 pairs;
- complete thigh/shin/foot pose: commonly 500 to 1500 pairs;
- clear swing accepted at zero lift: approximately 2500 to 5000 candidate tests;
- lift-needed swing: commonly 65000 to 165000 candidate tests;
- pose diagnostics: commonly 16000 to 50000 candidate tests.

Before ordered-lift search, cache a bounded list of `(primitive,cell,triangle)` references and their exact terrain vertices. Lift changes only endpoint Y, so no XZ span or terrain enumeration is repeated. The cache cannot contain more than 512 pairs and is discarded with the local planning transaction.

Do not cache across scene generations. Do not allocate a field-sized acceleration structure in Task 5.

---

## 9. Certified Swing Lift and History

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

G1ClearanceStatus g1_swing_clearance_plan(
    G1SwingClearancePlan& output,
    const G1ClearanceBudget& limits,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const vec3 current_sphere_centers[4],
    bool recorded_contact,
    float dt,
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

Both functions validate all four centers into a local candidate before assignment. Commit requires initialized history. A failed call leaves every prior component unchanged.

For one sphere, let `A` be its prior accepted center, `B` its current candidate center, `P` planted clearance, `W` swing clearance, and `L` endpoint lift. Target clearance and lift both interpolate linearly, so subtract the target from centerline Y:

```text
A_adjusted = A; A_adjusted.y -= P
B_lifted.y = canonical_binary32(round_to_nearest_binary32(
                   double(B.y) + double(L)))
B_adjusted = B; B_adjusted.y = double(B_lifted.y) - double(W)
```

Materialize `B_lifted.y` through `g1_apply_swing_lift_y` inside the strict TU and reject a nonfinite or non-runtime result. Task 6/7 uses that same non-inline helper when applying the chosen lift to its scratch sole target. This models the actual binary32 endpoint application rather than assuming that adding `L` produces an exact real displacement. Subtract `P`/`W` with outward binary64 arithmetic. The certified capsule clearance of `[A_adjusted,B_adjusted]` against zero is then exactly the continuous corrected margin for that materialized endpoint. Aggregate all four adjusted capsules.

All three binary64 fields in `G1SwingClearancePlan` are target-subtracted margins, not raw sole clearances: `baseline_lower_margin_m` is the `L=0` lower bound, and the two corrected fields are the lower/witness bounds at `applied_lift_m`. On the contact shortcut they are canonical `+0.0` and ignored because `sweep_evaluated=false`. `work` is the total private-ledger work over distinct cached lift evaluations.

Lift search operates on positive finite binary32 bit keys:

1. Require history initialized, all prior/current centers to pass `g1_ik_vec3_is_runtime_value`, `g1_foot_runtime_config_validate` to accept the configuration, and `g1_ik_dt_is_exact_25_hz(dt)` before checking contact state.
2. If recorded contact is true, return `Ok` with a zero-lift plan, `sweep_evaluated=false`, and `required_lift_certified=true`; post-solve validation still validates every input, including exact dt, before the same marked shortcut.
3. Evaluate `L=0.0f`. Propagate any non-`Ok` evaluator status without assigning output. If its lower margin is nonnegative, return a certified zero lift.
4. Evaluate `L=config.max_swing_lift_m`, whose bits must equal exact `0.08f`. Propagate a non-`Ok` evaluator status. If its valid lower margin is negative, return `Ok` with both lift fields equal to the tested cap, `required_lift_certified=false`, and `safe_stop_requested=true`. This means "the cap did not certify," not that an unrepresented larger requirement was measured.
5. Otherwise maintain a not-certified low key and a directly certified-safe high key. Bisect the integer bit keys; evaluate and memoize each new midpoint. A midpoint with `lower_margin_m >= 0` becomes high, and any other valid midpoint becomes low. A non-`Ok` midpoint propagates transactionally rather than being treated as unsafe.
6. The inclusive range from `+0.0f` to `0.08f` needs at most 30 midpoint halvings, so the two endpoint evaluations plus those midpoints fit exactly in the 32-distinct-key cap. At adjacency, return the cached high key with `required_lift_certified=true`; its cached predecessor is the low key and did not certify. Do not spend two hidden 33rd/34th re-evaluations. If adjacency was not reached within the cap, return `Uncertified`.
7. Post-solve validation is a separate top-level call with a fresh swing budget. It uses prior accepted centers and actual final centers, subtracts planted/swing endpoint targets in the same way, and requires a nonnegative lower margin.

The materialized binary32 endpoint height is nondecreasing in positive ordered-float `L`; every sweep parameter therefore receives a nonnegative, nondecreasing endpoint displacement. A directly nonnegative lower bound certifies that key and the true geometry at every larger key. The numerical evaluator may conservatively fail to certify a safe midpoint; in that case the search can over-lift by an unresolved certification band, not necessarily only one ULP. It can never return an under-lift: the selected high key itself has a cached nonnegative lower bound for its materialized endpoint. Tests claim only that the predecessor did not certify, not that it was proven to penetrate.

---

## 10. Transaction and Safe-Stop Semantics

- Geometry functions take no history or controller state.
- Swing planning accepts `const G1SwingHistory&` and writes only a local `G1SwingClearancePlan` before success assignment.
- Task 6 may advance its explicitly owned lock observer according to its existing policy, but it cannot commit swing history.
- Task 7 commits final rendered sphere centers only after the candidate pose, post-solve certified margin, planted thresholds, pose-capsule thresholds, and all other IK bounds are accepted.
- `OutsideDomain`, `BudgetExceeded`, or `Uncertified` leave the plan output unchanged; Task 6/7 maps the status itself to finite rejection and safe stop. A valid `Ok` plan whose tested `0.08f` cap has a negative lower margin is assigned with `required_lift_certified=false` and `safe_stop_requested=true`.
- `InvalidInput`, `InvalidField`, or `ArithmeticFailure` return through controlled diagnostic and normal cleanup.
- On finite rejection, accepted local/global pose arrays, accepted clearance, and both histories remain bit-identical to the prior accepted state. Only an assigned `Ok` cap-failure plan is loggable through the public plan; a non-`Ok` scratch result is not exposed as if certified.
- On a returned controlled error, public outputs and the complete `G1IkState` argument remain unchanged.

Add a distinct stop reason for unresolved certified clearance if later checker evidence cannot represent `OutsideDomain` or `BudgetExceeded`. Do not report `swing-lift` with fabricated `required_lift_m > 0.08f` evidence.

---

## 11. Strict-FP Build and Link Contract

`g1_clearance.cpp` and the ordered-lift implementation are non-inline and compiled separately. The release caller may retain `-ffast-math`; the kernel may not.

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

Do not add `-flto`. Add a negative compile guard that compiling `g1_clearance.cpp` with `-ffast-math` fails at the `#error`.

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

### E. Continuous swept lift

Reuse the near-`sqrt(3)` exactly coplanar fixture. Construct previous/current binary32 center Ys near clearances `+0.005 m` and `-0.001 m` at the same XZ. With binary32 planted/swing targets near `0.005` and `0.015`, compute the exact-real endpoint threshold `R` from the promoted input bits; require it to be within `2e-6` of `0.016 m`. Require:

- returned lift at least the smallest certified float at or above the promoted-bit oracle `R`;
- corrected lower margin nonnegative;
- predecessor float not certified;
- the old approximate `0.010641016 m` lift rejected.

Repeat with only one of four spheres obstructed and require provenance for that sphere.

### F. Rank and projection degeneracies

Exercise `A==B`, segment parallel to terrain, segment in the terrain plane, vertical projected patches, zero-horizontal-length edges, exact disk tangency, and a stationary point whose barycentric interval straddles a patch edge. Require either an `Ok` interval of width at most `1e-6` or the exact fail-closed `Uncertified` status at the subdivision cap; success with a wider interval is forbidden.

### G. Count and arithmetic adversaries

- Use the existing half-cell regression bits `cell=0x3b1efa48`, stop-X `0x3b6e7765`, stop-Z `0x36723088` to prove no sampled-step API remains and the exact capsule sees the whole segment.
- Minimum-positive-normal cell size, `FLT_MAX` endpoints/radius, endpoint subtraction overflow, and poisoned previous history must return promptly without conversion UB or loops.
- An AABB spanning exactly 512 cells/1024 closed-triangle pairs may proceed at the fixed caps; 513 cells/1026 pairs returns `BudgetExceeded` before the first terrain load.
- Give the 512-cell fixture a caller limit of only 1023 pairs and require `BudgetExceeded` before pair 1. This tests the pair boundary without inventing an odd pair count that fixed two-triangle cells cannot produce.
- Exhausting candidate or subdivision budget returns `BudgetExceeded`/`Uncertified` with no counter wrap.

### H. Exact timing and cap boundaries

- `1.0f/25.0f` succeeds.
- Both adjacent `nextafter` values fail transactionally.
- Keep the configuration maximum at exact `0.08f`. Construct promoted-bit endpoint thresholds immediately below/at the cap and require success, then one ordered float above the cap and require the assigned cap-failure safe-stop plan even though the deficit is below `1e-6`.
- Exactly `0.08f` is evaluated; no `+1e-6` acceptance hole is allowed.

### I. Build and parity

- Construct parity inputs from named `uint32_t` float bits rather than fast-math-compiled decimal/square-root expressions. The strict and fast-math-caller executables emit a compact line containing status, bound bits, witness bits/key, and all work counters; compare the lines byte-for-byte.
- A direct `-ffast-math -c g1_clearance.cpp` command must fail and mention the certified-kernel guard.
- Strict, release-caller, ASan/UBSan, and repeated-run hashes all pass.

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

- [ ] Write compile-time tests for enum values, binary64 bound types, budget constants, and non-copying const-history signatures.
- [ ] Compile the test before creating the files; require a missing-header RED.
- [ ] Add the declarations and strict-TU `__FAST_MATH__`/IEEE guards.
- [ ] Compile the strict caller/kernel objects successfully.
- [ ] Compile `g1_clearance.cpp` with `-ffast-math`; require the intentional guard failure.
- [ ] Commit only the three Task 1 files with `feat: define certified G1 clearance contract`.

**Review gate:** No implementation is accepted while a public safety field is `float`, a non-`Ok` result can partially assign output, or the strict kernel can inline into the caller.

### Task 2: Add checked v2 domain spans and terrain triangles

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces internal `g1_clearance_domain_contains`, checked cell-span enumeration, and exact `T0/T1` vertex construction.
- Produces checked point clearance.

- [ ] Add RED tests D, the noncoplanar point probes from C, malformed visited heights, and unchanged outputs.
- [ ] Implement structural/query-domain validation using existing bit helpers.
- [ ] Implement outward footprint/domain comparison with exact tangency handling.
- [ ] Preflight cell-span products and budgets before loops.
- [ ] Emit exact fixed-diagonal triangles and checked point results.
- [ ] Run strict, fast-math-caller, and sanitizer tests.
- [ ] Commit with `feat: enumerate checked G1HF v2 clearance triangles`.

**Review gate:** Exterior height cannot influence status or in-domain output; exact maximum edges and both triangle halves must be demonstrated by tests.

### Task 3: Implement outward intervals and analytic patch solver

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces strict interval operations, projected face coefficients, interior stationary solver, disk-clipped 3D edge solver, feasible witness reconstruction, and stable candidate keys.

- [ ] Add RED tests for analytic face interior, each edge clamp branch, circle tangency, vertical projection, membership uncertainty, and witness feasibility.
- [ ] Implement outward interval primitives with nonfinite/zero-denominator rejection.
- [ ] Implement face stationary formulas and interval barycentric classification.
- [ ] Implement the edge formulas and vertical-edge case.
- [ ] Implement lower/witness aggregation with the Section 3 key order.
- [ ] Verify every `Ok` result encloses a high-precision fixture oracle within `1e-6`.
- [ ] Commit with `feat: certify projected capsule patch minima`.

**Review gate:** The reviewer must derive the stationary and edge formulas independently and verify that no circle-only candidate is missing.

### Task 4: Compose the eight-patch capsule kernel and bounded fallback

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces `g1_capsule_clearance` and `g1_sphere_clearance` with exact prism-boundary coverage.

- [ ] Add RED fixtures A, B, F, and endpoint-reversal parity.
- [ ] Construct the two caps and six side triangles in the fixed patch order.
- [ ] Evaluate all patches without outward-normal classification.
- [ ] Implement the coarse patch lower bound and deterministic longest-edge subdivision.
- [ ] Enforce certificate width and subdivision budgets.
- [ ] Add the per-triangle v2 height-rounding guard.
- [ ] Run strict/release/sanitizer tests and repeated hash parity.
- [ ] Commit with `feat: certify G1 sphere and capsule clearance`.

**Review gate:** The reviewer must check the rank-three boundary argument and rank-deficient fiber argument. A sampled radial or centerline lattice is an automatic rejection.

### Task 5: Add foot and pose aggregation under shared budgets

**Files:**
- Modify: `g1_clearance.h`
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces four-sphere foot clearance and binary64 `G1LegClearance`/`G1PoseClearance` diagnostics.

- [ ] Add RED tests for one-worst-sphere provenance, both legs, every diagnostic member, shared pair-budget exhaustion, and transactional pose failure.
- [ ] Aggregate lower and witness values independently as specified.
- [ ] Thread one checked budget through every point/sphere/capsule primitive.
- [ ] Preserve exact G1 local geometry and Y-up world transforms from `g1_ik.h`.
- [ ] Run all focused modes.
- [ ] Commit with `feat: aggregate certified G1 pose clearance`.

**Review gate:** Every logged threshold source must be a binary64 lower bound, and a late right-leg error must leave the entire caller output unchanged.

### Task 6: Add continuous swept-foot lift planning

**Files:**
- Modify: `g1_clearance.h`
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces checked swing history, four-capsule sweep, ordered-float endpoint lift, and actual post-solve validation.

- [ ] Add RED fixtures E, G, H, contact-shortcut validation, failed reset/commit, and rejected-history snapshots.
- [ ] Cache bounded XZ terrain pairs once per four-sphere plan.
- [ ] Implement adjusted endpoint target geometry.
- [ ] Route scratch endpoint-Y lift application through `g1_apply_swing_lift_y` and test both rounding directions.
- [ ] Implement exact-dt validation before contact branching.
- [ ] Implement the 0/max/ordered-bit lift search and predecessor verification.
- [ ] Implement post-solve validation with actual final centers.
- [ ] Run all focused modes and compare strict/release result lines byte-for-byte.
- [ ] Commit with `feat: plan certified G1 swing clearance`.

**Review gate:** The wall/cap rejection must leave history exact, and the sqrt(3) fixture must require approximately `0.016 m`, not the old sampled value.

### Task 7: Integrate separate-object builds and downstream status handling

**Files:**
- Modify later: `g1_ik_runtime.h`
- Modify later: `controller.cpp`
- Modify later: every terrain-IK native build command/checker affected by the new object
- Test: `tests/cpp/test_g1_ik.cpp`
- Test: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Consumes all prior Task 5 APIs.
- Produces controller rejection/cleanup behavior without changing matching, support, or accepted history ownership.

- [ ] Add integration RED tests for every status mapping and accepted/rejected snapshots.
- [ ] Replace header-only calls with the public non-inline API.
- [ ] Apply planned lift to the scratch sole target through `g1_apply_swing_lift_y`; never recompute `input_y + lift` in the fast-math caller.
- [ ] Update strict, release, sanitizer, and controller link commands exactly as Section 11.
- [ ] Add the negative fast-math-kernel build guard to verification.
- [ ] Run the complete verification matrix below.
- [ ] Commit only integration-owned files with `build: link certified G1 clearance kernel`.

**Review gate:** Inspect the final link command and object flags from build logs. Passing numerical tests does not compensate for compiling the kernel under fast math.

---

## 14. Verification Matrix

Run after each relevant task and in full after Task 7:

| Mode | Caller flags | Kernel flags | Required evidence |
|---|---|---|---|
| Strict | `-O2 -Wall -Wextra -Werror -pedantic` | same plus `-fno-fast-math -ffp-contract=off -frounding-math` | Exit 0, no warnings |
| Release caller | `-O3 -ffast-math -DNDEBUG` | `-O3 -fno-fast-math -ffp-contract=off -frounding-math -DNDEBUG` | Exit 0; result line byte-equal to strict |
| Sanitizer | `-O1 -g`, ASan/UBSan with halt-on-error | strict FP plus same sanitizers | Exit 0, no report |
| Negative build | any | `-ffast-math` | Compile fails at `__FAST_MATH__` guard |
| Determinism | repeat release caller 20 times | strict kernel | Identical status/bound/witness/work hash |
| Transaction | strict and release | strict kernel | Seeded outputs/history exact after all non-`Ok` paths |
| Budget | strict and sanitizer | strict kernel | Exact cap succeeds; cap+1 fails before work |
| Geometry | all successful modes | strict kernel | Analytic oracle enclosed; width `<=1e-6` |

Final source/order guards:

```bash
! rg -n 'ceil\(|radial_steps|segment_steps|half.*cell.*sample' \
  g1_clearance.cpp g1_clearance.h
rg -n '#error.*fast math|G1ClearancePatchesPerPair|G1ClearanceMaximumLiftEvaluations' \
  g1_clearance.cpp g1_clearance.h
git diff --check
```

Expected: no lattice sample-count code exists, the strict-FP and budget guards are present, every test mode exits zero, parity lines match byte-for-byte, and the worktree is clean after task-scoped commits.

---

## 15. Reconciliation Gate Before Editing the Active Plan

Read-only reconciliation against integrated Task 2 commit `6287a0e` locks these producer names: `G1SurfaceQueryStatus`, `g1_surface_query_v2`, `g1_ik_vec3_is_runtime_value`, `g1_ik_dt_is_exact_25_hz`, and `g1_foot_runtime_config_validate`. This design consumes those names directly and preserves their fail-closed `Valid`/`Outside`/`Invalid` distinction; it does not add an adapter that restores exterior fallback or tolerant `25 Hz` checks. The active terrain-IK plan remains untouched on this branch. At implementation start, confirm that later integration has not renamed these producers before changing either plan.

Before replacing active Task 5, require two approvals:

1. **Geometry review:** prism proof, eight-patch construction, analytic stationary/edge formulas, degeneracy fallback, and sqrt(3) RED are accepted.
2. **Runtime review:** separate strict-FP object, hard budgets, status mapping, accepted-only history commit, and controller build implications are accepted.

Only then replace the sampled Task 5 text and update all downstream one-command build invocations.
