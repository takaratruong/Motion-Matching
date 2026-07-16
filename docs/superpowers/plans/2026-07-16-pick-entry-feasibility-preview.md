# Pickup Entry Feasibility Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select once between the two exact standoff-preserving pickup arc slots by previewing their hard path feasibility and current live-snapshot match readiness through a const runtime-owned API.

**Architecture:** Refactor the pickup matcher around one request-independent structured evaluator while keeping `select_whole_clip` behavior byte-for-byte compatible at its public boundary. `InteractionRuntime::preview_pick` will rigidly map a copy of one real live-flat snapshot to a planar prospective root, build the normal query and matcher context from exact registry-owned data, and return path and readiness facts without mutation. The placement controller will construct both exact arc slots, preview them in fixed `Plus`, `Minus` order from one provider sample, freeze the deterministic eligible winner, and record an independently reconstructable evidence contract.

**Tech Stack:** C++17 matcher/runtime/controller code, native 25 Hz scheduler, Python `unittest` evidence validation, GNU Make normal and `-O3 -DNDEBUG -ffast-math` builds, Raylib/X11 graphical acceptance.

## Global Constraints

- Preserve native exact 25 Hz controller/runtime/evidence/video scheduling; do not add a runtime update, alternate-rate callback, or `dt` to preview.
- Preserve both arc invariants, the `0.15 m` Reach neighborhood, `[0.35 m, 0.45 m]` standoff, `0.25 m` matcher root-correction limit, `25 degree` matcher yaw limit, and every existing collision, cost, IK, playback, attachment, placement, and controller movement threshold.
- Preview consumes only an actual live-flat `LocomotionSnapshot`, planar prospective X/Z/world yaw, exact target handle, and affordance ID. It must not accept or synthesize canonical data, external `QueryInput`/`MatchInput`, matcher config, request ID, target snapshot, affordance snapshot, or candidate authority.
- Preview is `const`, request-free, reservation-free, attachment-free, event-free, and mutation-free across runtime state/diagnostics/pose/optionals, both registries, target ownership/generation, caller inputs, scheduler edges, request sequence, and simulation/displayed roots.
- Evaluate slots in stable `Plus`, then `Minus` order from one common snapshot. Hard feasibility filters eligibility; hand score ranks only slots that are both path-feasible and match-ready. Preserve the exact Right=`Plus`, Left=`Minus` bit-equal tie rule.
- Generate both previews inside the existing ordinary live-flat provider callback on its one local snapshot, atomically commit the pair, return that same snapshot to the scheduler, and let steering consume only the prior completed epoch on the next native tick.
- Bound zero-input preview settling with its own counter against existing `kPlacementAutodemoMaximumWalkTicks == 250U`; never reuse paused `walk_ticks`, add another timer, or surface no-path/deadline failure before recording the final complete pair.
- Freeze one slot before slot-specific navigation. Keep ordinary left-stick navigation and one-way braking; never retry Interact, switch after freeze/brake, or use preview output as Preflight execution authority.
- Preserve all dirty Task 8 work. Before each edit, inspect only the intended tracked-file diff; never reset, overwrite, stage, or reformat unrelated hunks.
- Never access, stat, hash, execute, modify, stage, or delete the protected repository-root artifact `interaction_query_probe`. Use only `git status --short --untracked-files=no`; safe builds may use `build/task12/interaction_query_probe_safe` through existing Make targets.
- Do not create an intermediate product commit. Leave the completed product files inside the open Task 8 review unit for its final verified commit boundary.
- Remove temporary matcher/predicate/controller diagnostics after their bounded use. Do not publish successful evidence from a diagnostic-only or canonical-equivalent probe.

## File and Responsibility Map

- `interaction_matcher.h`: retain the public `select_whole_clip` contract; declare the internal structured hard-feasibility/readiness result shared with runtime.
- `interaction_matcher.cpp`: factor request-independent context validation and deterministic whole-clip evaluation without changing normal selection ordering or reasons.
- `interaction_runtime.h`: declare `PickEntryRoot`, `PickEntryPreview`, and the exact public const `InteractionRuntime::preview_pick` API.
- `interaction_runtime.cpp`: implement planar rigid snapshot mapping, exact registry/query assembly, const preview, snapshot fingerprinting support, and reuse of the shared matcher evaluator.
- `tests/cpp/interaction_runtime_fixture.h`: add exact snapshot/runtime/registry comparison and prospective-root fixtures without weakening existing tests.
- `tests/cpp/test_interaction_matcher.cpp`: prove hard-filter boundaries, first-feasible metadata, readiness separation, and legacy selection parity.
- `tests/cpp/test_interaction_runtime.cpp`: prove rigid mapping, API rejection, repeated determinism, normal-Preflight parity, and complete no-mutation behavior.
- `tests/cpp/test_pick_entry_preview_fast_math.cpp`: small assertion-independent release-fast-math canary for mapper, matcher feasibility, and const preview.
- `controller.cpp`: construct and retain both arc slots, perform same-snapshot preview epochs, freeze selection, navigate/brake to the frozen slot, and emit evidence.
- `tests/python/test_playable_placement_evidence.py`: migrate the exact JSON schema, independently recompute both slots/outcomes/winner, and enforce timing, provider, mutation, retry, and freeze policy.
- `tests/python/test_interaction_gate1.py`: prove the new fast-math and oracle Make targets compile/run the intended sources and never acquire the protected probe dependency.
- `Makefile`: add focused normal/fast-math preview targets while preserving the dirty placement gate and safe protected-artifact flow.
- `.superpowers/sdd/task-8-report.md`: update only after all verification and the real graphical gate; no product commit is created here.

---

### Task 1: Share Matcher Hard Feasibility Without Changing Selection

**Files:**
- Modify: `interaction_matcher.h:65-83`
- Modify: `interaction_matcher.cpp:31-55,100-146,490-688`
- Modify: `tests/cpp/test_interaction_matcher.cpp`

**Interfaces:**
- Consumes: existing `MatchInput`, `MatchConfig`, `MatchCandidate`, `MatchResult`, deterministic clip/entry order, and unchanged collision/correction/cost predicates.
- Produces: `matcher_detail::PickEvaluationInput`, `matcher_detail::PickEvaluation`, and `matcher_detail::evaluate_pick_entries(const PickEvaluationInput&, const MatchConfig&)`; `select_whole_clip(const MatchInput&, const MatchConfig&)` remains unchanged and delegates only after its existing request/ownership validation.

- [ ] **Step 1: Add failing structured-evaluation tests before changing matcher code**

Add these named tests to `tests/cpp/test_interaction_matcher.cpp` and call them from `main()`:

```cpp
void test_shared_pick_evaluation_preserves_select_behavior();
void test_shared_pick_evaluation_separates_path_from_cost_readiness();
void test_shared_pick_evaluation_preserves_hard_reason_order();
void test_shared_pick_evaluation_reports_first_feasible_entry();
void test_shared_pick_evaluation_preserves_match_reason_priority_and_cost_availability();
```

Build an internal input from each existing `MatchInput` fixture without its request:

```cpp
matcher_detail::PickEvaluationInput evaluation_input(const MatchInput& input) {
    return {
        input.database,
        input.features,
        input.query,
        input.locomotion,
        input.target,
        input.affordance,
    };
}
```

The tests must assert all of the following with exact values:

```cpp
const MatchInput accepted_input = valid_input();
const auto accepted = matcher_detail::evaluate_pick_entries(
    evaluation_input(accepted_input), MatchConfig{});
assert(accepted.path_feasible);
assert(accepted.path_reason == Reason::None);
assert(accepted.match_ready);
assert(accepted.match_reason == Reason::None);
assert(accepted.selection.accepted);
assert(accepted.feasible_entry_frame == accepted.selection.candidate.entry_frame);
assert(accepted.contact_frame == accepted.selection.candidate.contact_frame);

const MatchInput over_cost_input = high_cost_input();
const auto over_cost = matcher_detail::evaluate_pick_entries(
    evaluation_input(over_cost_input), MatchConfig{});
assert(over_cost.path_feasible);
assert(over_cost.path_reason == Reason::None);
assert(!over_cost.match_ready);
assert(over_cost.match_reason == Reason::PoorMatch);
assert(over_cost.feasible_entry_frame >= 0);
assert(over_cost.contact_frame >= 0);
assert(over_cost.total_cost_available);
```

For `valid_input`, `wrong_hand_input`, `out_of_range_input`, `excessive_root_input`, `blocked_table_input`, `high_cost_input`, malformed feature storage, and mixed primary-Reach/earlier-Approach outcomes, assert that `select_whole_clip` retains its existing `accepted`, `reason`, and complete accepted `MatchCandidate` values. Add combined-invalid rows with null database plus invalid request, null features plus stale target/request, and valid pack plus invalid request; require the existing precedence `PackUnavailable` before request validation. Assert hard reason order exactly `OutOfRange`, `CorrectionLimit`, `BlockedPath`, `NoCandidate`; `PoorMatch` is a readiness reason after an entry reaches cost evaluation, not a hard-path reason.

Add three explicit no-winner rows: an entry that passes all hard path filters but has invalid feature/cost storage; over-cost-only entries; and a mixed invalid-feature plus finite-over-cost set. The first row must be path-feasible with first entry/contact metadata, `total_cost_available=false`, and legacy `match_reason=OutOfRange`. The over-cost-only row must have `total_cost_available=true` and `match_reason=PoorMatch`. The mixed row must preserve the legacy aggregate priority `OutOfRange` ahead of `PoorMatch`, even though a finite over-cost exists; assert `evaluation.match_reason == evaluation.selection.reason == Reason::OutOfRange`.

- [ ] **Step 2: Run matcher RED and verify the missing internal API is the only failure**

Run:

```bash
make build/tests/test_interaction_matcher
```

Expected: compile fails because `matcher_detail::PickEvaluationInput`, `PickEvaluation`, and `evaluate_pick_entries` do not exist. A changed legacy assertion, link error, or collision-boundary failure is not an acceptable RED result.

- [ ] **Step 3: Declare the minimal internal structured interface**

In `interaction_matcher.h`, add this internal namespace without changing `MatchInput`, `MatchResult`, or `select_whole_clip`:

```cpp
namespace matcher_detail {

struct PickEvaluationInput {
    const Database* database = nullptr;
    const Features* features = nullptr;
    NormalizedQuery query{};
    LocomotionSnapshot locomotion{};
    InteractionTarget target{};
    GraspAffordance affordance{};
};

struct PickEvaluation {
    bool path_feasible = false;
    bool match_ready = false;
    Reason path_reason = Reason::None;
    Reason match_reason = Reason::None;
    int32_t feasible_entry_frame = -1;
    int32_t contact_frame = -1;
    bool total_cost_available = false;
    float total_cost = 0.0F;
    MatchResult selection{};
};

PickEvaluation evaluate_pick_entries(
    const PickEvaluationInput& input,
    const MatchConfig& config);

}  // namespace matcher_detail
```

`feasible_entry_frame` and `contact_frame` are captured immediately from the first entry that passes request-independent context, correction, hand compatibility, and all path predicates, before feature lookup or cost validity, and are never overwritten. Independently, `total_cost_available` becomes true only after a finite cost exists; `total_cost` tracks the same deterministic minimum finite cost used by normal selection (accepted best when ready, otherwise best finite over-cost). `selection.candidate` remains valid only for the cost-ranked accepted winner. The availability bit prevents a valid zero cost from being confused with the default and never changes the configured maximum-cost decision.

- [ ] **Step 4: Factor request-independent validation and entry evaluation**

In `interaction_matcher.cpp`, split the current request checks into:

```cpp
std::optional<Reason> validate_pick_context(
    const matcher_detail::PickEvaluationInput& input);
std::optional<Reason> validate_request(const MatchInput& input);
matcher_detail::PickEvaluationInput pick_evaluation_input(
    const MatchInput& input);
```

`validate_request` keeps the exact nonzero ID/generation/request checks, exact handle/affordance match, and Targeted-owner behavior, then calls `validate_pick_context`. `validate_pick_context` checks database/features availability, exact authored affordance content, finite target/table/affordance geometry, positive dimensions, and rejects Attached/Held; preview's stricter `Free` requirement remains in runtime before calling it.

Refactor candidate evaluation so the first hard correction/compatibility/path pass records feasibility and frames before feature lookup or cost validation:

```cpp
struct CandidateEvaluation {
    CandidateStatus status = CandidateStatus::OutOfRange;
    bool path_feasible = false;
    MatchCandidate candidate{};
};

// Immediately after correction, hand, and path predicates pass, before
// feature lookup/cost validity:
evaluation.path_feasible = true;
evaluation.candidate.clip = frames.clip;
evaluation.candidate.entry_frame = entry_frame;
evaluation.candidate.contact_frame = frames.contact;
// Later, set total_cost_available only for a finite computed cost. Preserve
// OutOfRange for invalid feature storage, PoorMatch for a finite over-limit
// cost, and Accepted only for the unchanged configured threshold.
```

`evaluate_pick_entries` must run the existing deterministic primary-Reach pass and only then the earlier-Approach fallback. It records the first hard-path-feasible entry/contact in that evaluation order even if feature storage or cost is subsequently invalid, cost-ranks finite candidates exactly as today, and returns:

```cpp
path_feasible = any entry passed every hard correction/compatibility/path predicate;
path_reason = path_feasible ? Reason::None : aggregate_hard_reason(failures);
match_ready = best.has_value();
selection = legacy_match_result(best, failures);
match_reason = match_ready ? Reason::None : selection.reason;
```

Do not let a `saw_finite_over_cost` shortcut override legacy failure aggregation. The unchanged aggregate order still makes over-cost-only inputs `PoorMatch`, but mixed `OutOfRange` plus `PoorMatch` remains `OutOfRange`; `CorrectionLimit`/`BlockedPath` precedence remains whatever the existing selector already records for the same failure set.

Keep legacy `select_whole_clip` as:

```cpp
MatchResult select_whole_clip(const MatchInput& input, const MatchConfig& config) {
    if (input.database == nullptr || input.features == nullptr) {
        return reject(Reason::PackUnavailable);
    }
    if (const std::optional<Reason> invalid = validate_request(input)) {
        return reject(*invalid);
    }
    return matcher_detail::evaluate_pick_entries(
        pick_evaluation_input(input), config).selection;
}
```

The explicit pack-null guard stays ahead of `validate_request` exactly as in the current selector; do not rely on the shared context validator for that public reason precedence. `pick_evaluation_input` is a field-for-field aggregate copy of `database`, `features`, `query`, `locomotion`, `target`, and `affordance`; it has no request argument or request field. The wrapper must reproduce the old aggregate reason when no candidate is accepted, even where preview's separated `path_reason`/`match_reason` is more informative.

- [ ] **Step 5: Prove exact collision-boundary reuse and legacy GREEN**

Extend the existing root/table, hand/table, and pre-contact hand/object boundary tests to assert both the normal selector reason and structured hard feasibility on the same input object. Do not duplicate or expose collision helpers.

Run:

```bash
make build/tests/test_interaction_matcher
./build/tests/test_interaction_matcher
```

Expected: exit 0; all existing matcher assertions and the five new structured-evaluation tests pass with unchanged threshold literals.

- [ ] **Step 6: Independent matcher checkpoint**

Have a fresh reviewer compare `select_whole_clip` results before/after for every existing fixture and inspect the diff for duplicated path predicates, reordered clip/entry loops, changed `MatchConfig` values, or preview-specific threshold overrides. Do not continue until the reviewer confirms one evaluator owns correction/path/cost decisions and the public selector behavior is unchanged. Do not commit.

---

### Task 2: Map a Live Snapshot to a Prospective Planar Root

**Files:**
- Modify: `interaction_runtime.h:123-151`
- Modify: `interaction_runtime.cpp:13-390`
- Modify: `tests/cpp/interaction_runtime_fixture.h`
- Modify: `tests/cpp/test_interaction_runtime.cpp`

**Interfaces:**
- Consumes: `LocomotionSnapshot`, root bone `g1_skeleton::Simulation`, `Pose`, `world_pose`, quaternion/vector operations, and the three-scalar `PickEntryRoot` declared in this task.
- Produces: internal `runtime_detail::PickSnapshotMap map_pick_entry_snapshot(const LocomotionSnapshot&, PickEntryRoot)` and `uint64_t locomotion_snapshot_fingerprint(const LocomotionSnapshot&)`; Task 3 consumes only the mapped copy/result, while the public controller-facing API remains `preview_pick`.

- [ ] **Step 1: Write RED tests for every mapped and unchanged snapshot channel**

Declare the public planar value type in `interaction_runtime.h` exactly as approved:

```cpp
struct PickEntryRoot {
    float world_x = 0.0F;
    float world_z = 0.0F;
    float world_yaw_radians = 0.0F;
};
```

Add and register these tests in `tests/cpp/test_interaction_runtime.cpp`:

```cpp
void test_pick_snapshot_map_is_a_rigid_planar_world_map();
void test_pick_snapshot_map_preserves_local_channels_and_input();
void test_pick_snapshot_map_rejects_every_nonfinite_or_nonunit_input();
void test_pick_snapshot_fingerprint_is_fieldwise_and_canonical();
```

Create a nontrivial fixture snapshot: nonzero root Y, pitch/roll plus yaw, distinct root linear/angular velocity, nonzero local transforms and velocities for every bone, three distinct future roots, hand DOF/velocities, and both foot contacts. Preserve `const LocomotionSnapshot original = snapshot` before mapping.

For `slot = {2.25F, -1.75F, 0.70F}`, independently derive `q_delta`, `p_slot`, and `t_delta`, then assert:

```cpp
const auto mapped = runtime_detail::map_pick_entry_snapshot(snapshot, slot);
assert(mapped.accepted);
assert(exact(snapshot, original));
assert(near(mapped.snapshot.pose.positions[root].x, slot.world_x));
assert(near(mapped.snapshot.pose.positions[root].y, original.pose.positions[root].y));
assert(near(mapped.snapshot.pose.positions[root].z, slot.world_z));
assert(near(world_yaw(mapped.snapshot.pose.rotations[root]), slot.world_yaw_radians));

const WorldPose before = world_pose(original.pose);
const WorldPose after = world_pose(mapped.snapshot.pose);
for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
    assert(near(after.positions[bone], t_delta + quat_mul_vec3(q_delta, before.positions[bone])));
    assert(near_sign(after.rotations[bone], quat_mul(q_delta, before.rotations[bone])));
    assert(near(after.velocities[bone], quat_mul_vec3(q_delta, before.velocities[bone])));
    assert(near(after.angular_velocities[bone], quat_mul_vec3(q_delta, before.angular_velocities[bone])));
}
```

Assert every non-root local pose channel, every hand scalar, and both contacts are exactly unchanged; future positions use the same rigid map and future rotations are left-multiplied by `q_delta`. Add one subtest for each prospective X/Z/yaw non-finite value, each snapshot vector/quaternion/scalar non-finite family, non-unit root rotation, and undefined live yaw. Construct positive/negative infinity and quiet/signaling NaN test values from explicit IEEE-754 bit patterns with `std::memcpy`, and run the same rejection table in the release-fast-math canary. Rejection must leave the input exactly unchanged.

- [ ] **Step 2: Run mapper RED**

Run:

```bash
make build/tests/test_interaction_runtime
```

Expected: compile fails only because `PickEntryRoot`, `runtime_detail::PickSnapshotMap`, `map_pick_entry_snapshot`, and `locomotion_snapshot_fingerprint` are missing.

- [ ] **Step 3: Implement the copy-only planar rigid map**

Declare the testable internal result in `interaction_runtime.h` under `runtime_detail`:

```cpp
struct PickSnapshotMap {
    bool accepted = false;
    Reason reason = Reason::None;
    LocomotionSnapshot snapshot{};
};

PickSnapshotMap map_pick_entry_snapshot(
    const LocomotionSnapshot& live_flat_snapshot,
    PickEntryRoot prospective_root);
uint64_t locomotion_snapshot_fingerprint(
    const LocomotionSnapshot& snapshot);
```

Implement finiteness checks through this bitwise helper before any floating-point comparison, quaternion operation, or arithmetic; do not rely on `std::isfinite`, `value == value`, or optimizer-sensitive ordered comparisons under `-ffast-math`:

```cpp
bool finite_float_bits(float value) noexcept {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) != 0x7F800000U;
}
```

Apply `finite_float_bits` field-by-field to every prospective scalar and snapshot float. Only after that preflight succeeds, implement the map in `interaction_runtime.cpp` without normalizing invalid input:

```cpp
LocomotionSnapshot mapped = live_flat_snapshot;
const vec3 p_live = mapped.pose.positions[root];
const quat q_live = mapped.pose.rotations[root];
const float live_yaw = world_yaw(q_live);
const float delta_yaw = shortest_angle(
    prospective_root.world_yaw_radians - live_yaw);
const quat q_delta = quat_from_angle_axis(
    delta_yaw, vec3(0.0F, 1.0F, 0.0F));
const vec3 p_slot(
    prospective_root.world_x, p_live.y, prospective_root.world_z);
const vec3 t_delta = p_slot - quat_mul_vec3(q_delta, p_live);

mapped.pose.positions[root] = p_slot;
mapped.pose.rotations[root] = quat_mul(q_delta, q_live);
mapped.pose.velocities[root] = quat_mul_vec3(
    q_delta, mapped.pose.velocities[root]);
mapped.pose.angular_velocities[root] = quat_mul_vec3(
    q_delta, mapped.pose.angular_velocities[root]);
for (size_t i = 0; i < mapped.future_root_positions.size(); ++i) {
    mapped.future_root_positions[i] = t_delta + quat_mul_vec3(
        q_delta, mapped.future_root_positions[i]);
    mapped.future_root_rotations[i] = quat_mul(
        q_delta, mapped.future_root_rotations[i]);
}
```

Validate every snapshot scalar/vector/quaternion for finiteness with the bitwise helper and validate the live root quaternion against the existing unit tolerance before mapping; do not introduce stricter normalization checks for child-local or future channels than the approved contract. Re-run bitwise finiteness checks on derived `live_yaw`, `delta_yaw`, `q_delta`, `p_slot`, and `t_delta` before using them. After mapping, independently reconstruct both world poses and verify all world transforms/velocities, root Y/pitch-roll preservation, relative geometry, future deltas, and exact input equality within the existing pose/quaternion tolerances. Return `accepted=false`, `reason=Reason::OutOfRange`, and a value-initialized output snapshot rather than a partially mapped snapshot when input or a mapping invariant is invalid; matcher correction/path failures retain their own later reasons.

- [ ] **Step 4: Implement the canonical field-by-field snapshot digest**

Use an explicit FNV-1a-style byte stream over fields in declaration/index order. Canonicalize float negative zero to positive zero and choose a deterministic quaternion sign before hashing; never hash object addresses, aggregate object bytes, or padding:

```cpp
void hash_quat(CanonicalHash& hash, quat value) {
    if (value.w < 0.0F ||
        (value.w == 0.0F && std::tie(value.x, value.y, value.z) <
                              std::tuple(0.0F, 0.0F, 0.0F))) {
        value = -value;
    }
    hash.scalar(value.w);
    hash.scalar(value.x);
    hash.scalar(value.y);
    hash.scalar(value.z);
}
```

Hash all pose positions, velocities, rotations, angular velocities, hand DOF/velocities, contacts, future positions, and future rotations. Tests must show quaternion sign flips and negative-zero substitutions preserve the fingerprint, while changing any semantic field changes it.

- [ ] **Step 5: Run mapper GREEN**

Run:

```bash
make build/tests/test_interaction_runtime
./build/tests/test_interaction_runtime
```

Expected: exit 0; all existing runtime tests plus the four mapper/fingerprint tests pass.

- [ ] **Step 6: Independent rigid-map checkpoint**

Have a fresh reviewer verify the implementation maps only the copied root and future-root world channels, preserves child-local/hand/contact data exactly, rotates velocities, keeps root Y and pitch/roll, checks the untouched caller input, and does not use a canonical pose. Do not continue on a component-wise root rewrite that fails the reconstructed-world invariants. Do not commit.

---

### Task 3: Add Const Runtime-Owned Pickup Preview and No-Mutation Proof

**Files:**
- Modify: `interaction_runtime.h:123-216`
- Modify: `interaction_runtime.cpp:148-205,595-750,1200-1390`
- Modify: `tests/cpp/interaction_runtime_fixture.h`
- Modify: `tests/cpp/test_interaction_runtime.cpp`
- Create: `tests/cpp/test_pick_entry_preview_fast_math.cpp`
- Create: `tests/cpp/test_pick_entry_oracle.cpp`
- Modify: `Makefile:51-165,218-289,304-324`
- Modify: `tests/python/test_interaction_gate1.py` (`ReleaseFastMathMakefileTests`)

**Interfaces:**
- Consumes: Task 1 `matcher_detail::evaluate_pick_entries`, Task 2 mapper/fingerprint, existing `make_query_input`, query normalization, exact `TargetRegistry` lookup, runtime database/features/config, and actual live-flat snapshots.
- Produces: exact public `PickEntryPreview InteractionRuntime::preview_pick(const LocomotionSnapshot&, PickEntryRoot, TargetHandle, uint32_t) const`; controller code may consume only this method and its result.

- [ ] **Step 1: Add the exact public API and rejection RED tests**

Declare the approved result in `interaction_runtime.h`:

```cpp
struct PickEntryPreview {
    bool path_feasible = false;
    bool match_ready = false;
    Reason path_reason = Reason::None;
    Reason match_reason = Reason::None;
    PickEntryRoot prospective_root{};
    int32_t feasible_entry_frame = -1;
    int32_t contact_frame = -1;
    float total_cost = 0.0F;
    MatchCandidate match_candidate{};
};

PickEntryPreview preview_pick(
    const LocomotionSnapshot& live_flat_snapshot,
    PickEntryRoot prospective_root,
    TargetHandle target,
    uint32_t affordance_id) const;
```

Add and register these tests in `tests/cpp/test_interaction_runtime.cpp`:

```cpp
void test_pick_preview_public_api_reports_path_and_match_separately();
void test_pick_preview_rejects_invalid_runtime_target_and_root_inputs();
void test_pick_preview_is_deterministic_and_const_on_every_outcome();
void test_pick_preview_matches_normal_preflight_for_same_realized_snapshot();
void test_pick_preview_never_reserves_or_constructs_request_authority();
void test_pick_preview_rejection_reason_mapping_is_exact();
```

Cover disabled and non-Locomotion runtime; missing database/features/registry; zero/stale generation; wrong affordance; Free versus Targeted/Attached/Held state; each non-finite root scalar; invalid snapshot; hard `BlockedPath`; accepted; finite over-cost `PoorMatch`; and malformed feature storage that remains `OutOfRange`/`PackUnavailable` rather than being relabeled.

Table-drive the exact pre-evaluation mapping in `test_pick_preview_rejection_reason_mapping_is_exact`:

| Condition | `path_reason` and `match_reason` |
|---|---|
| missing database, features, or registry; disabled runtime with no pack | `PackUnavailable` |
| non-`Locomotion` runtime with dependencies present | `TargetUnavailable` |
| target ID absent/zero, affordance absent/zero/wrong, or target state `Targeted`/`Attached`/`Held` | `TargetUnavailable` |
| target ID exists but supplied generation differs from `find_by_id(id)->handle.generation` | `TargetChanged` |
| nonfinite/invalid snapshot, nonfinite prospective X/Z/yaw, non-unit live root, undefined yaw, or mapping invariant failure | `OutOfRange` |

Assert both booleans remain false, both reasons are bit-equal to the table, candidate/frames/cost remain default, and no registry/runtime state changes for every row. Do not let lookup ordering turn stale generation into `TargetUnavailable`.

- [ ] **Step 2: Capture exact before/after mutation snapshots in the fixture**

Add fixture helpers that compare, field by field:

```cpp
struct RuntimeObservation {
    RuntimeState state{};
    RuntimeDiagnostics diagnostics{};
    std::vector<InteractionTarget> targets{};
    std::vector<PlacementSurface> surfaces{};
    LocomotionSnapshot caller_snapshot{};
};

RuntimeObservation observe(
    const InteractionRuntime& runtime,
    const RuntimeFixture& fixture,
    const LocomotionSnapshot& caller);
void assert_exact(const RuntimeObservation& before,
                  const RuntimeObservation& after);
void assert_same_next_update_after_preview(
    RuntimeFixture& previewed,
    RuntimeFixture& untouched_control,
    const RuntimeInput& input);
```

Populate `targets` and `surfaces` by copying every field of the fixture's known handles through the registries' public `find` APIs; do not require either registry class to be copyable. Include target handle/generation/state/owner/reservation, surface handle/generation/state/owner/reservation, runtime diagnostics, and every exposed pose/object field. `observe` stays read-only. Where private runtime optionals cannot be observed directly, `assert_same_next_update_after_preview` advances the previewed and untouched control fixtures once with exactly equal `RuntimeInput` values and compares the complete outputs/candidate/event behavior. Repeat preview twice on success, rejection, and an injected exception path.

- [ ] **Step 3: Run preview RED**

Run:

```bash
make build/tests/test_interaction_runtime
```

Expected: compile fails because `PickEntryPreview` and `InteractionRuntime::preview_pick` are missing. Existing runtime tests must still compile up to that missing interface.

- [ ] **Step 4: Factor one runtime-owned pickup context builder**

In the private section of `InteractionRuntime` in `interaction_runtime.h`, declare one query assembly result/member; implement it in `interaction_runtime.cpp` for use by preview and normal Preflight:

```cpp
struct PickEvaluationBuild {
    bool accepted = false;
    Reason reason = Reason::None;
    matcher_detail::PickEvaluationInput input{};
};

PickEvaluationBuild build_pick_evaluation(
    const LocomotionSnapshot& locomotion,
    TargetHandle target_handle,
    uint32_t affordance_id,
    bool require_free) const;
```

The builder performs lookup in this exact order: reject zero/absent ID as `TargetUnavailable`; call `registry_->find_by_id(target_handle.id)`; report `TargetChanged` if the ID exists but generation differs; then resolve the exact nonzero authored affordance ID. Only `require_free == true` enforces `state == Free && owner_request == 0`, reporting missing/wrong/non-Free target or affordance as `TargetUnavailable`. Normal Preflight calls with `require_free == false` only after its existing outer request/reservation validation has proved the exact `Targeted` target and owner; the shared builder must accept that validated state and must not reapply the preview-only Free rule. It reads only runtime-owned database/features/config, builds `QueryInput` with existing `make_query_input`, calls `build_raw_query`/`normalize_query`, and never accepts a caller target/config/query/request/candidate.

- [ ] **Step 5: Implement `preview_pick` as a const composition**

Implement in this order:

```cpp
PickEntryPreview InteractionRuntime::preview_pick(
    const LocomotionSnapshot& live_flat_snapshot,
    PickEntryRoot prospective_root,
    TargetHandle target,
    uint32_t affordance_id) const {
    PickEntryPreview preview{};
    preview.prospective_root = prospective_root;
    if (database_ == nullptr || features_ == nullptr || registry_ == nullptr) {
        preview.path_reason = Reason::PackUnavailable;
        preview.match_reason = Reason::PackUnavailable;
        return preview;
    }
    if (state_ != RuntimeState::Locomotion) {
        preview.path_reason = Reason::TargetUnavailable;
        preview.match_reason = Reason::TargetUnavailable;
        return preview;
    }
    const auto mapped = runtime_detail::map_pick_entry_snapshot(
        live_flat_snapshot, prospective_root);
    if (!mapped.accepted) {
        preview.path_reason = mapped.reason;
        preview.match_reason = mapped.reason;
        return preview;
    }
    const PickEvaluationBuild built = build_pick_evaluation(
        mapped.snapshot, target, affordance_id, true);
    if (!built.accepted) {
        preview.path_reason = built.reason;
        preview.match_reason = built.reason;
        return preview;
    }
    const auto evaluated = matcher_detail::evaluate_pick_entries(
        built.input, config_.matcher);
    preview.path_feasible = evaluated.path_feasible;
    preview.match_ready = evaluated.match_ready;
    preview.path_reason = evaluated.path_reason;
    preview.match_reason = evaluated.match_reason;
    preview.feasible_entry_frame = evaluated.feasible_entry_frame;
    preview.contact_frame = evaluated.contact_frame;
    preview.total_cost = evaluated.total_cost_available
        ? evaluated.total_cost
        : 0.0F;
    if (evaluated.match_ready) {
        preview.match_candidate = evaluated.selection.candidate;
    }
    return preview;
}
```

Set `path_reason == None` iff feasible and `match_reason == None` iff ready. Require `match_ready => path_feasible`. Preserve first-hard-path entry/contact even when feature/cost is invalid; copy a finite cost only when the internal availability bit is true and copy `match_candidate` only when ready. Because the approved public struct has no cost-availability boolean, controller evidence claims `_cost_available` conservatively only for `match_ready` or `match_reason == PoorMatch`; mixed legacy-priority failures may carry a diagnostic numeric cost without advertising it as available. Let unexpected exceptions propagate to the controller only after the tests prove exact no-mutation rollback; do not catch and relabel them `PoorMatch`.

- [ ] **Step 6: Prove normal Preflight parity and request ownership separation**

For one realized mapped live snapshot, compare preview against the subsequent ordinary Interact/Preflight matcher result. Assert accepted/reason and the complete `match_candidate`, including its own entry/contact/cost, match when ready. Add a targeted parity row that calls the shared builder with `require_free=true` before Interact, observes a Free target, then lets ordinary Preflight reserve that exact target and calls the builder with `require_free=false`; require the latter to accept the correctly `Targeted`/owned target. A builder that rejects normal Preflight merely because it is no longer Free fails the test. Separately assert preview's feasible entry/contact remain the first hard-path-passing entry even if its feature/cost is invalid or a later accepted candidate wins; assert `total_cost` follows the evaluator's independent finite-cost availability/minimum semantics and is not assumed to belong to those first frames. Also assert preview leaves target `Free` with owner `0`, request IDs unchanged, diagnostics unchanged, and no pose ownership; only ordinary Preflight may reserve and populate request/candidate/player state.

- [ ] **Step 7: Write Make dry-run RED tests, then add normal and release-fast-math build seams**

Create `tests/cpp/test_pick_entry_preview_fast_math.cpp` with explicit `require(bool, const char*)` checks that remain active under `NDEBUG`. Cover a nontrivial rigid map, bit-pattern infinity/quiet-NaN/signaling-NaN rejection for snapshot and prospective-root fields, blocked versus feasible path, feasible over-cost `PoorMatch`, repeated preview equality, and unchanged registry/runtime observations. This canary must fail if mapper validation is replaced with `std::isfinite` behavior that `-ffast-math` optimizes away.

First add these unconditionally runnable methods to `ReleaseFastMathMakefileTests` in `tests/python/test_interaction_gate1.py`:

```python
def test_pick_entry_preview_release_fast_math_target_compiles_and_runs(self):
def test_safe_suite_executes_pick_entry_preview_release_fast_math(self):
def test_pick_entry_oracle_target_compiles_without_probe_dependency(self):
```

Use the class's existing `make -Bn` helper. The first two tests require exactly one compile of `tests/cpp/test_pick_entry_preview_fast_math.cpp` to `build/tests/test_pick_entry_preview_release_fast_math` with `-O3 -DNDEBUG -ffast-math -I.`, all `INTERACTION_RUNTIME_SOURCES`, and exactly one execution with `--fast-math-canary`. The oracle test dry-runs `build/tests/test_pick_entry_oracle`, requires exactly one normal compile of `tests/cpp/test_pick_entry_oracle.cpp` plus all runtime sources to that safe build path, and rejects any dependency/command containing the protected repository-root probe name.

Run RED before editing `Makefile`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests.test_pick_entry_preview_release_fast_math_target_compiles_and_runs \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests.test_safe_suite_executes_pick_entry_preview_release_fast_math \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests.test_pick_entry_oracle_target_compiles_without_probe_dependency -v
```

Expected: the dry-run assertions fail because the new Make targets/rules do not exist; a skipped test or unrelated Make parse failure is not acceptable.

Add Make variables and targets:

```make
PICK_ENTRY_PREVIEW_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_pick_entry_preview_release_fast_math
PICK_ENTRY_ORACLE_TEST := $(CPP_TEST_DIR)/test_pick_entry_oracle

$(PICK_ENTRY_PREVIEW_FAST_MATH_TEST): \
  tests/cpp/test_pick_entry_preview_fast_math.cpp \
  $(INTERACTION_RUNTIME_SOURCES) \
  interaction_runtime.h interaction_matcher.h interaction_pose.h \
  | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math \
	  tests/cpp/test_pick_entry_preview_fast_math.cpp \
	  $(INTERACTION_RUNTIME_SOURCES) -o $@

$(PICK_ENTRY_ORACLE_TEST): tests/cpp/test_pick_entry_oracle.cpp \
  tests/cpp/interaction_runtime_fixture.h $(INTERACTION_RUNTIME_SOURCES) \
  interaction_runtime.h interaction_matcher.h interaction_pose.h \
  | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_pick_entry_oracle.cpp \
	  $(INTERACTION_RUNTIME_SOURCES) -o $@

.PHONY: test-pick-entry-preview-release-fast-math
test-pick-entry-preview-release-fast-math: $(PICK_ENTRY_PREVIEW_FAST_MATH_TEST)
	$(PICK_ENTRY_PREVIEW_FAST_MATH_TEST) --fast-math-canary
```

Add the target to `test-interaction-safe` without changing the protected safe-query-probe dependency, then rerun the same three Python methods and require GREEN before compiling either new C++ target.

- [ ] **Step 8: Add a current-pack diagnostic oracle without using the protected artifact**

Create `tests/cpp/test_pick_entry_oracle.cpp`, a read-only test executable that loads an explicit pack directory argument, builds the approved canonical-equivalent diagnostic snapshot and exact target through existing tracked APIs, and calls `preview_pick` for `R`, `P_minus`, and `P_plus`. It must assert:

```text
R:       ready, entry 114, contact 139, cost 0.666622
P_minus: path_reason BlockedPath
P_plus:  ready, entry 114, contact 139, cost 0.718336
```

Use a `2e-6` cost tolerance and print one compact result line per root. Build only `build/tests/test_pick_entry_oracle`; do not depend on, execute, or inspect the protected repository-root artifact.

- [ ] **Step 9: Run focused GREEN in normal and fast-math modes**

Run:

```bash
make build/tests/test_interaction_matcher build/tests/test_interaction_runtime
./build/tests/test_interaction_matcher
./build/tests/test_interaction_runtime
make test-pick-entry-preview-release-fast-math
make build/tests/test_pick_entry_oracle
./build/tests/test_pick_entry_oracle resources/g1_interaction
```

Expected: every command exits 0; normal and fast-math results agree, legacy runtime lifecycle remains green, and the oracle prints only the preregistered values. If the pack is not yet built, defer only the last oracle invocation to Task 7; do not substitute canonical data into production preview.

- [ ] **Step 10: Independent runtime ownership checkpoint**

Have a fresh reviewer inspect the public signature, const implementation, registry/query ownership, same evaluator/config use, all rejection reasons, exception/no-mutation proof, Preflight parity, and fast-math canary. Reject any caller-supplied `MatchInput`, request ID, canonical fallback, config override, reservation, or hidden runtime update. Do not commit.

---

### Task 4: Construct Both Exact Arc Slots Without Selecting One

**Files:**
- Modify: `controller.cpp` (`ControllerPlacementAutodemoState`, `make_placement_autodemo_interaction_waypoint`, pickup waypoint geometry validation)
- Modify: `tests/python/test_playable_placement_evidence.py` (controller geometry/source-policy tests)

**Interfaces:**
- Consumes: the existing Reach waypoint `R`, resolved target transform/footprint, existing clearance chord, hand preference, and the approved standoff/rotation helpers.
- Produces: a fixed ordered pair `PlacementPickEntrySlots{Plus, Minus}`. This task constructs and validates candidates only; it does not sample locomotion, call preview, navigate, reserve, or choose a winner.

- [ ] **Step 1: Write RED tests for the exact two-slot geometry contract**

Add and register these unconditionally runnable tests on `Task8PlacementPolicyTests` in `tests/python/test_playable_placement_evidence.py`; do not place them on the real-artifact-gated `PlayablePlacementEvidenceTests` class:

```python
def test_pick_entry_constructs_both_ordered_exact_arc_slots(self):
def test_pick_entry_slots_are_yaw_equivariant_and_hand_scores_only_rank(self):
def test_pick_entry_slot_construction_fails_closed_for_each_invalid_input(self):
```

The first test must independently evaluate the controller geometry for an asymmetric target yaw and assert exactly two identities in order `Plus`, `Minus`. For each slot, assert the preserved radius/standoff, shared clearance chord, world X/Z, root yaw, unchanged waypoint height, unchanged target-relative hand/contact rotation, and a finite hand score. Assert `Plus` is the positive signed rotation from `R` around the target and `Minus` is the negative signed rotation; do not infer identity from left/right hand score.

The equivariance test rotates and translates the whole fixture, then proves both positions and rotations undergo the same rigid world transform while identities and tie policy remain unchanged. Give both slots bit-equal hand scores and assert the data still remains ordered `Plus`, `Minus`; selection belongs to Task 5.

Table-drive invalid chord, zero/negative/nonfinite standoff, nonfinite target/waypoint transforms, a chord greater than the diameter, invalid quaternion norm, and a result that leaves the approved `[0.35 m, 0.45 m]` band. Every invalid row must fail the construction as a unit; it must never emit only one slot or clamp the input.

- [ ] **Step 2: Run the geometry RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_constructs_both_ordered_exact_arc_slots \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_slots_are_yaw_equivariant_and_hand_scores_only_rank \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_slot_construction_fails_closed_for_each_invalid_input -v
```

Expected: the tests fail because the controller still collapses the two arc solutions to one hand-selected waypoint. A failure caused by changed geometry constants or fixture drift is not acceptable.

- [ ] **Step 3: Add explicit fixed slot identities and storage**

In `controller.cpp`, add these controller-local types next to `ControllerPlacementAutodemoState`:

```cpp
enum class PlacementPickEntrySlotIdentity : uint8_t { Plus, Minus };

struct PlacementPickEntrySlot {
    PlacementPickEntrySlotIdentity identity{};
    interaction::Transform waypoint{};
    interaction::PickEntryRoot prospective_root{};
    float hand_score = 0.0F;
};

struct PlacementPickEntrySlots {
    std::array<PlacementPickEntrySlot, 2> ordered{};
    float clearance_chord_m = 0.0F;
    float preserved_standoff_m = 0.0F;
};

PlacementPickEntrySlots make_placement_autodemo_interaction_slots(
    const interaction::Transform& reach_waypoint,
    const interaction::InteractionTarget& target);
```

Store the pair in `ControllerPlacementAutodemoState`; replace `make_placement_autodemo_interaction_waypoint` with `make_placement_autodemo_interaction_slots`, and remove the pre-preview single-slot `interaction_waypoint` winner. Keep the downstream selected waypoint unset until Task 5 freezes it.

- [ ] **Step 4: Generate both solutions from the existing invariant-preserving math**

Use the existing oriented-support and robust component/sign rotation code once to validate inputs and derive the radial X/Z vector. Then generate both candidates without branching on the active hand:

```cpp
const vec3 object = target.object_world.position;
const vec3 reach_radius(
    reach_waypoint.position.x - object.x,
    0.0F,
    reach_waypoint.position.z - object.z);
const float radius = length(reach_radius);
const float half_ratio = clearance_chord_m / (2.0F * radius);
const float arc = 2.0F * std::asin(half_ratio);

const auto make_slot = [&](PlacementPickEntrySlotIdentity identity, float sign) {
    const vec3 radial = quat_mul_vec3(
        quat_from_angle_axis(sign * arc, vec3(0.0F, 1.0F, 0.0F)),
        reach_radius);
    const vec3 position(
        object.x + radial.x,
        stable_reach_waypoint.position.y,
        object.z + radial.z);
    const quat rotation = stable_reach_waypoint.rotation;
    const float hand_score = dot(
        position - stable_reach_waypoint.position,
        hand_lateral);
    return PlacementPickEntrySlot{
        identity,
        interaction::Transform{position, rotation},
        interaction::PickEntryRoot{
            position.x,
            position.z,
            autodemo_yaw_radians(stable_reach_waypoint.rotation)},
        hand_score};
};

return PlacementPickEntrySlots{
    std::array<PlacementPickEntrySlot, 2>{
        make_slot(PlacementPickEntrySlotIdentity::Plus, +1.0F),
        make_slot(PlacementPickEntrySlotIdentity::Minus, -1.0F)},
    clearance_chord_m,
    radius};
```

Reject before storing either slot unless every input/result is finite, the quaternion checks pass, `clearance_chord_m <= 2 * radius`, both derived chords match the existing clearance within the existing geometric tolerance, both radii equal `radius`, both standoffs remain within `[0.35 m, 0.45 m]`, both copy `R.y` and `R.rotation` within `2e-5`, and both prospective yaws equal `world_yaw(R.rotation)`. Do not rotate the waypoint quaternion with the arc, change constants, add a lateral translation, or recompute the target/Reach waypoint per slot.

- [ ] **Step 5: Run geometry GREEN and an independent arc checkpoint**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_constructs_both_ordered_exact_arc_slots \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_slots_are_yaw_equivariant_and_hand_scores_only_rank \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_slot_construction_fails_closed_for_each_invalid_input -v
make gate-place-headless
```

Expected: the three tests and the existing headless placement gate pass. Have a fresh reviewer independently recompute `P_plus` and `P_minus` from `R`, target center, chord, and radius for an unrotated and a yawed fixture, then inspect that no score or preview outcome participates in construction. Do not continue or commit if either solution is mirrored, reordered, clamped, or partially accepted.

---

### Task 5: Preview One Common Live Snapshot, Select Deterministically, and Freeze Once

**Files:**
- Modify: `controller.cpp` (`ControllerPlacementAutodemoState`, placement pickup input block, scheduler locomotion-provider lambda, evidence counters)
- Modify: `tests/python/test_playable_placement_evidence.py` (selector and source-policy tests)

**Interfaces:**
- Consumes: the fixed `Plus`, `Minus` pair from Task 4, one ordinary live-flat locomotion-provider sample, and two `InteractionRuntime::preview_pick` results.
- Produces: one frozen slot before slot-specific navigation plus counters/timing facts for Task 6. Preview output remains advisory; normal Interact/Preflight reconstructs its own request and matcher input.

- [ ] **Step 1: Write RED tests for filtering, common-snapshot use, defer/fail, and freeze**

Add and register these unconditionally runnable tests on `Task8PlacementPolicyTests`; do not place them on `PlayablePlacementEvidenceTests`, whose artifact guard would turn RED/GREEN into skips:

```python
def test_pick_entry_selector_filters_before_hand_score_and_ties(self):
def test_pick_entry_preview_uses_one_live_snapshot_in_fixed_order(self):
def test_pick_entry_preview_runs_inside_provider_and_consumes_prior_epoch(self):
def test_pick_entry_preview_deadline_advances_while_walk_ticks_pause(self):
def test_pick_entry_selection_defers_freezes_once_and_never_retries(self):
def test_pick_entry_selection_scope_is_runtime_owned_and_live_only(self):
```

Drive the pure selection truth table with all combinations of `path_feasible`, `match_ready`, unequal score, and bit-equal score. Assert:

```text
eligible(slot) = slot.preview.path_feasible && slot.preview.match_ready
zero eligible and no path-feasible slot -> fail on the first consume of that complete pair, with no further preview or motion and only the evidence-presentation handoff before visibility
zero eligible and at least one path-feasible slot -> defer at zero input
one eligible -> that slot, regardless of the other slot's score
two eligible -> greater hand score; bit-equal score -> Right selects Plus, Left selects Minus
```

Use a scripted sequence `Plus PoorMatch / Minus BlockedPath`, then `Plus ready / Minus BlockedPath`. Assert both calls in an epoch receive the same snapshot fingerprint, call order is `Plus`, `Minus`, the first epoch does not navigate or submit Interact, and the second freezes `Plus`. The callback-handoff test must prove the pair is generated and atomically stored inside the one existing locomotion-provider callback, that callback returns the exact same snapshot value to the scheduler/resolver, and steering can consume only the prior complete epoch on the following native tick. Continue ticking with an artificially better `Minus`; assert the selected identity, waypoint, freeze tick, and prospective root never change, Interact is submitted exactly once, and preview is never called after freeze.

For the deadline test, hold stick input at zero and hold `walk_ticks` constant while emitting path-feasible/not-ready pairs. Assert the separate preview elapsed counter advances once per normal 25 Hz provider callback, allows a ready pair at the bound, and otherwise fails closed at `kPlacementAutodemoMaximumWalkTicks == 250U`; it must not inherit the paused `walk_ticks` value or add an alternate timer/update.

The source-policy test must reject controller source that constructs `QueryInput`, `MatchInput`, `InteractionRequest`, canonical pose/features, matcher config, or calls `select_whole_clip`; it must also reject direct writes to the runtime/root or registry from the preview block.

- [ ] **Step 2: Run the controller-policy RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_selector_filters_before_hand_score_and_ties \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_preview_uses_one_live_snapshot_in_fixed_order \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_preview_runs_inside_provider_and_consumes_prior_epoch \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_preview_deadline_advances_while_walk_ticks_pause \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_selection_defers_freezes_once_and_never_retries \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_selection_scope_is_runtime_owned_and_live_only -v
```

Expected: the tests fail because the paired preview epoch, eligibility filter, and freeze state do not yet exist.

- [ ] **Step 3: Add one explicit selector with the approved tie rule**

Add this controller-local function:

```cpp
std::optional<size_t> choose_placement_pick_entry_slot(
    const PlacementPickEntrySlots& slots,
    const std::array<interaction::PickEntryPreview, 2>& previews,
    interaction::Hand active_hand);
```

Implement only this decision order:

```cpp
std::array<size_t, 2> eligible{};
size_t count = 0;
for (size_t i = 0; i < 2; ++i) {
    if (previews[i].path_feasible && previews[i].match_ready) {
        eligible[count++] = i;
    }
}
if (count == 0) return std::nullopt;
if (count == 1) return eligible[0];
if (slots.ordered[0].hand_score > slots.ordered[1].hand_score) return 0;
if (slots.ordered[1].hand_score > slots.ordered[0].hand_score) return 1;
return active_hand == interaction::Hand::Right ? 0 : 1;
```

Validate finite hand scores when the pair is built. Do not use total match cost, entry/contact frame, evaluation order, path reason severity, or world side as a secondary rank key.

- [ ] **Step 4: Add auditable preview/freeze state**

Extend `ControllerPlacementAutodemoState` with exact state needed by the controller and evidence writer:

```cpp
PlacementPickEntrySlots pick_entry_slots{};
std::array<interaction::PickEntryPreview, 2> pick_entry_previews{};
int64_t pick_preview_epoch = -1;
int64_t pick_preview_consumed_epoch = -1;
uint64_t pick_preview_epoch_count = 0;
uint64_t pick_preview_snapshot_fingerprint = 0;
uint64_t runtime_pick_preview_calls = 0;
uint64_t pick_preview_mutation_count = 0;
uint32_t pick_preview_elapsed_ticks = 0;
std::optional<size_t> frozen_pick_entry_index{};
std::optional<uint64_t> pick_entry_freeze_tick{};
uint64_t pick_interact_submission_count = 0;
uint64_t pick_reservation_transition_count = 0;
PlacementPickPreviewFailure pending_pick_preview_failure =
    PlacementPickPreviewFailure::None;
bool pick_preview_failure_recorded = false;
```

Declare `enum class PlacementPickPreviewFailure : uint8_t { None, NoPath, Deadline };` beside the state. Initialize current/consumed epochs to `-1`, count/fingerprint/counters to zero, the failure to `None`, and optionals disengaged. Evidence maps a disengaged freeze tick to `-1`; the first completed pair becomes epoch `0`. `runtime_pick_preview_calls` is pickup-only and must remain distinct from the existing place-staging `runtime_preview_calls` field/counter. Increment the reservation-transition counter only from observed normal runtime lifecycle transitions, never from preview. Do not expose or store a pointer/reference to the provider snapshot beyond the callback.

- [ ] **Step 5: Generate and commit the pair inside the existing provider callback**

Keep the current `make_flat_controller_pose`/`expand_flat_controller_pose` capture once before `interaction_scheduler.tick`. Keep construction of the local `LocomotionSnapshot snapshot` and its three future `trajectory_positions`/`trajectory_rotations` inside the existing locomotion-provider callback passed to that tick. After that callback has generated its one local `snapshot`, run the paired preview there only when placement mode has reached the common `R` pre-entry point, no slot is frozen, and no failure is pending. Do not call, factor, or invoke the provider outside `interaction_scheduler.tick`; do not create an extra sample or callback. Keep the pickup demo's canonical branch unchanged and reject any canonical branch in placement mode.

Inside that existing callback, before `return snapshot`, execute:

```cpp
const uint64_t fingerprint =
    interaction::runtime_detail::locomotion_snapshot_fingerprint(snapshot);
const PlacementPickPreviewObservation runtime_before =
    capture_placement_pick_preview_observation(
        interaction_runtime,
        interaction_registry,
        interaction_surface_registry,
        interaction_scene_target_handle);

std::array<interaction::PickEntryPreview, 2> previews{};
for (size_t i = 0; i < 2; ++i) {  // fixed Plus, then Minus
    previews[i] = interaction_runtime.preview_pick(
        snapshot,
        placement_autodemo_state.pick_entry_slots
            .ordered[i].prospective_root,
        interaction_scene_target_handle,
        interaction_authored_target.affordances.front().id);
}

if (capture_placement_pick_preview_observation(
        interaction_runtime,
        interaction_registry,
        interaction_surface_registry,
        interaction_scene_target_handle) != runtime_before) {
    ++placement_autodemo_state.pick_preview_mutation_count;
    throw std::runtime_error("pickup preview mutated protected state");
}

// One atomic commit only after both const calls and the no-mutation check.
placement_autodemo_state.pick_entry_previews = previews;
placement_autodemo_state.pick_preview_snapshot_fingerprint = fingerprint;
placement_autodemo_state.pick_preview_epoch = static_cast<int64_t>(
    placement_autodemo_state.pick_preview_epoch_count);
++placement_autodemo_state.pick_preview_epoch_count;
placement_autodemo_state.runtime_pick_preview_calls += 2U;
++placement_autodemo_state.pick_preview_elapsed_ticks;
return snapshot;  // the exact value previewed, for this ordinary scheduler tick
```

Define `PlacementPickPreviewObservation` beside the state as fieldwise copies of `InteractionRuntime::state()`, diagnostics, the exact target, known placement surface, request sequence, displayed/simulation roots, and pending input edges; exclude only the controller's observational pickup-preview counters that are committed after the comparison. Compare fields explicitly rather than aggregate bytes. If a call throws, surface the error and keep input zero; do not accept a partial epoch. The local `snapshot` is constructed once, both calls take it by const reference, the resolver/runtime receives the same returned value, and `runtime_pick_preview_calls` advances only by two with a complete committed pair.

- [ ] **Step 6: Consume only the prior complete epoch and bound zero-input settling**

At the placement input/steering block, which runs before the current tick's scheduler callback, inspect a pair only when `pick_preview_epoch > pick_preview_consumed_epoch`, then set the consumed epoch. This fixed one-tick handoff prevents steering from observing a half-built/current-callback pair:

1. If neither stored preview is path-feasible, set `pending_pick_preview_failure=NoPath`; the complete pair, fingerprint, epoch, elapsed count, and two calls have already been committed on the prior tick.
2. Otherwise call `choose_placement_pick_entry_slot`. If eligible, freeze the index/identity/waypoint/root and record the freeze tick before any slot-specific navigation; check eligibility before the deadline so a ready pair produced at tick 250 may win.
3. If at least one path is feasible but no slot is ready, keep stick and Interact input exactly zero. If `pick_preview_elapsed_ticks >= kPlacementAutodemoMaximumWalkTicks` (the existing exact `250U` bound), set `pending_pick_preview_failure=Deadline`; do not use `walk_ticks`, which pauses during zero input.
4. When failure becomes pending, keep input zero and gate off further preview. Construct the one terminal evidence row from the pending failure and already committed last pair with its serialized JSON field `"pick_preview_failure_recorded": true`, even though controller state is still false during construction/write. Publish that row once; only after the write/flush succeeds set controller state `pick_preview_failure_recorded=true`, which permits visible `NoPath`/`Deadline` failure on the next input phase. If publication fails, publish no partial row, leave controller state false, and take the existing evidence-write error path. Never emit a second failure row or run another preview while presenting failure.
5. On a frozen winner, run the existing left-stick navigation and one-way braking against only that waypoint. Submit one Interact edge through the existing input path and increment `pick_interact_submission_count`; never retry on reject, switch slots, or inject a request directly.

Increment `pick_preview_elapsed_ticks` exactly once per committed provider-callback epoch from the first preview until freeze/failure; never reset it from `walk_ticks` or derive it from stick magnitude. Normal `InteractionRuntime::update` remains exactly once per native tick in its existing location. Preview gets no `dt`, and no extra provider sample, runtime update, registry operation, or scheduler edge is added.

- [ ] **Step 7: Run controller GREEN and independent state-machine review**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_selector_filters_before_hand_score_and_ties \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_preview_uses_one_live_snapshot_in_fixed_order \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_preview_runs_inside_provider_and_consumes_prior_epoch \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_preview_deadline_advances_while_walk_ticks_pause \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_selection_defers_freezes_once_and_never_retries \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_pick_entry_selection_scope_is_runtime_owned_and_live_only -v
make build/tests/test_interaction_runtime
./build/tests/test_interaction_runtime
make gate-place-headless
```

Expected: all focused tests, the runtime suite, and headless gate pass. Have a fresh reviewer trace three timelines—both blocked, transient `PoorMatch` through the independent 250-tick bound, and both ready/tied—and confirm preview executes inside one provider callback, the callback returns that exact snapshot, steering consumes only the prior committed epoch, two calls occur in fixed order, zero motion persists while deferred even with paused `walk_ticks`, failure retains the last pair, freeze precedes navigation, one normal Interact edge occurs, and no post-freeze preview/retry/fallback exists. Do not commit.

---

### Task 6: Migrate Evidence to a Fixed Pair and Independently Validate Selection

**Files:**
- Modify: `controller.cpp` (placement evidence row construction and final summary)
- Modify: `tests/python/test_playable_placement_evidence.py` (`FIELD_TYPES`, synthetic records, renderer/validator, corruption tests)

**Interfaces:**
- Consumes: the exact pair geometry, complete paired preview epochs, frozen selection, normal runtime lifecycle counters, and the existing per-tick placement evidence stream.
- Produces: one strict ordered JSON schema and validator that reconstructs both arc slots and the winner without trusting controller-selected/eligible booleans.

- [ ] **Step 1: Write the schema migration and corruption RED tests first**

Add and register these synthetic schema/corruption tests on `PlacementEvidenceValidatorUnitTests`; do not place them on the artifact-gated `PlayablePlacementEvidenceTests` class:

```python
def test_pick_entry_evidence_reconstructs_both_slots_and_winner(self):
def test_pick_entry_evidence_preserves_transient_poor_match_epochs(self):
def test_pick_entry_evidence_rejects_slot_geometry_order_and_score_corruption(self):
def test_pick_entry_evidence_rejects_outcome_eligibility_and_selection_corruption(self):
def test_pick_entry_evidence_rejects_snapshot_mutation_retry_and_timing_corruption(self):
def test_pick_entry_evidence_requires_callback_handoff_and_bounded_elapsed(self):
def test_pick_entry_evidence_requires_current_pack_plus_selection(self):
```

Migrate every existing synthetic fixture row rather than adding permissive defaults in the validator. Include at least: pre-preview sentinels, one provider-callback-committed deferred `PoorMatch` epoch, a prior-epoch consume row, frozen selection, navigation, braking, one Interact edge, normal Preflight/reservation, playback/contact, placement, and terminal success. Add separate valid no-path and 250-tick deadline fixture tails whose last complete pair is recorded before failure. The old single `interaction_waypoint` fixture must not remain authoritative.

Each corruption test changes exactly one fact in an otherwise valid record sequence and asserts a targeted validator message. Cover swapped identities/evaluation order, a sign-flipped arc, changed standoff/chord/rotation/height, zero or changed epoch fingerprint, score-ranked infeasible slot, false eligibility, wrong tie winner, different fingerprints within an epoch, preview mutation, partial/two-call mismatch, same-tick current-epoch steering, a provider/callback count mismatch, stalled elapsed ticks while `walk_ticks` is paused, elapsed greater than 250, failure before its pair, a terminal failure row with `pick_preview_failure_recorded=false`, any row/preview after the true terminal marker, navigation before freeze, selection change after freeze, post-freeze preview, multiple Interact edges, preview reservation, and retry/fallback after rejection. Signed-zero and quaternion-sign canonicalization belong to the C++ digest test where the underlying snapshot fields are available.

- [ ] **Step 2: Run evidence RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_reconstructs_both_slots_and_winner \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_preserves_transient_poor_match_epochs \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_rejects_slot_geometry_order_and_score_corruption \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_rejects_outcome_eligibility_and_selection_corruption \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_rejects_snapshot_mutation_retry_and_timing_corruption \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_requires_callback_handoff_and_bounded_elapsed \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_pick_entry_evidence_requires_current_pack_plus_selection -v
```

Expected: failures identify missing fixed-pair fields/validation. Do not weaken strict field equality or accept legacy rows to obtain RED/GREEN.

- [ ] **Step 3: Extend `FIELD_TYPES` in one fixed order with no optional fields**

Append these global fields in this exact order to `FIELD_TYPES` and the controller writer:

| Field | JSON type | Pre-preview sentinel/invariant |
|---|---:|---|
| `pick_preview_epoch` | integer | `-1`, then nondecreasing |
| `pick_preview_consumed_epoch` | integer | `-1`, consumes only a prior committed epoch |
| `pick_preview_snapshot_fingerprint` | integer | `0`, then nonzero |
| `pick_preview_snapshot_source` | string | `"unset"`, then exactly `"live_flat"` |
| `pick_preview_epoch_count` | integer | `0`, increments once per complete pair |
| `runtime_pick_preview_calls` | integer | `0`, equals `2 * pick_preview_epoch_count`; distinct from place `runtime_preview_calls` |
| `pick_preview_mutation_count` | integer | always `0` |
| `pick_preview_elapsed_ticks` | integer | `0`, advances per committed epoch; maximum `250` |
| `pick_preview_pending_failure` | string | `"none"`, then optionally `"no_path"` or `"deadline"` |
| `pick_preview_failure_recorded` | boolean | `false` on ordinary rows; exactly `true` in the sole terminal no-path/deadline row |
| `pick_entry_selected_slot` | string | `"none"`, then `"Plus"` or `"Minus"` |
| `pick_entry_selection_frozen` | boolean | `false`, then monotonic `true` |
| `pick_entry_freeze_tick` | integer | `-1`, then exact first frozen tick |
| `pick_interact_submission_count` | integer | `0`, terminal exactly `1` |
| `pick_reservation_transition_count` | integer | `0` through preview; normal lifecycle only |

For each prefix in exact order `pick_plus`, then `pick_minus`, append this exact suffix order:

| Suffix | JSON type | Sentinel/meaning |
|---|---:|---|
| `_identity` | string | fixed `"Plus"` / `"Minus"` |
| `_evaluation_order` | integer | fixed `0` / `1` |
| `_waypoint_position` | array[3] number | finite exact candidate |
| `_waypoint_rotation` | array[4] number | finite unit quaternion |
| `_prospective_world_x` | number | finite candidate X |
| `_prospective_world_z` | number | finite candidate Z |
| `_prospective_world_yaw_radians` | number | finite candidate yaw |
| `_clearance_chord_m` | number | common approved chord |
| `_preserved_standoff_m` | number | common radius in `[0.35, 0.45]` |
| `_hand_score` | number | finite; ranks eligible slots only |
| `_path_feasible` | boolean | `false` before first epoch |
| `_match_ready` | boolean | `false` before first epoch |
| `_path_reason` | string | `"NoCandidate"` sentinel, then enum spelling |
| `_match_reason` | string | `"NoCandidate"` sentinel, then enum spelling |
| `_feasible_entry_frame` | integer | `-1` unless hard feasible metadata exists |
| `_contact_frame` | integer | `-1` unless hard feasible metadata exists |
| `_cost_available` | boolean | conservative public availability: ready or `PoorMatch` |
| `_total_cost` | number | `0.0` sentinel; finite when available |
| `_eligible` | boolean | derived path-feasible AND match-ready |
| `_selected` | boolean | exactly one only after freeze |

Keep the existing strict renderer behavior: every row has every field, exact scalar/array types are checked before semantic validation, booleans are not accepted as integers, and nonfinite numbers are rejected. The controller must spell `Reason` values through one exhaustive converter with no default success case.

- [ ] **Step 4: Emit immutable pair geometry and complete epoch outcomes**

Write both slot identities/geometry/scores from the stored Task 4 pair on every row. Before the first complete preview epoch, emit only the documented outcome sentinels. After each provider callback completes both calls, atomically replace both outcome groups, fingerprint, epoch, elapsed count, and pickup-only call counter in the same evidence row; never mix a new Plus result with an old Minus result. `pick_preview_consumed_epoch` must lag the callback's newly committed epoch until the next native input/steering tick.

Derive `_eligible` in the writer as `path_feasible && match_ready`, but treat it as redundant audit data. Since `PickEntryPreview` intentionally has no public availability bit, set `_cost_available` only for ready or `PoorMatch` results and never infer it merely from a nonzero numeric cost. Set `_selected` only from the frozen index. Once frozen, retain the pair, last complete outcomes, selected identity, prospective root, fingerprint, and freeze tick unchanged through terminal success. For `no_path`/`deadline`, construct exactly one zero-input row with the last complete pair, pending failure, and JSON `pick_preview_failure_recorded=true`; atomically publish it, then flip the controller state's same-named flag only on successful write/flush and surface failure. A failed write leaves state false and publishes no row. Emit observed Interact/reservation counters from normal controller/runtime edges, not inferred terminal state.

- [ ] **Step 5: Independently reconstruct slots, epochs, and the winner in Python**

Add these validator helpers with no call into controller code:

```python
def _expected_pick_entry_slots(record):
    # Recompute Plus and Minus from R, target transform, chord, and standoff.

def _pick_entry_eligible(record, prefix):
    return record[f"{prefix}_path_feasible"] and record[f"{prefix}_match_ready"]

def _expected_pick_entry_winner(record):
    # Filter by eligibility, rank finite hand score, apply exact hand tie rule.
```

For every row, independently enforce:

- identities/order are always Plus then Minus and both exact geometries match `_expected_pick_entry_slots`, including radius, signed arc, chord, height, quaternion, and prospective yaw;
- outcomes change only as a complete pair inside one ordinary provider callback, one epoch uses one nonzero live-flat fingerprint, `runtime_pick_preview_calls` equals exactly two per epoch and remains independent of place `runtime_preview_calls`, and mutation count stays zero;
- steering consumes only an epoch committed by the preceding callback/tick; elapsed ticks advance with every deferred pair even while `walk_ticks` and input remain zero, never exceed 250, and accept a ready pair at the bound before declaring deadline;
- `path_reason == None` iff path-feasible, `match_reason == None` iff match-ready, a first hard-path pass always supplies feasible/contact frames even when its feature/cost fails, `PoorMatch` may defer only with path feasible, mixed legacy-priority `OutOfRange` is not relabeled, and cost obeys the conservative public availability guard;
- the recorded eligibility booleans equal recomputation, neither-path-feasible and deadline failures retain their last complete pair and publish exactly one terminal row with `pick_preview_failure_recorded == true` before becoming visible, no row after that terminal marker and no further preview exists, and a path-feasible/not-ready epoch keeps stick and Interact input zero until the existing 250-tick deadline;
- the recorded winner equals independent filtering/ranking/tie reconstruction, freezes before the first slot-navigation/brake row, never changes, and receives exactly one later normal Interact edge;
- preview never coincides with reservation/owner/attachment/request-sequence mutation; exactly one normal pick-resolver call follows freeze, the first reservation transition follows normal Preflight, and actual Preflight reaches `Align`; reject post-freeze preview, slot switch, retry, or fallback;
- controller/runtime ticks and evidence/video samples remain on the existing exact 25 Hz schedule.

Do not trust `_eligible`, `_selected`, controller summary text, or successful final state as proof of these relations.

- [ ] **Step 6: Pin the preregistered current-pack outcome in the real-evidence path**

When validating a real current-pack run, require this live pair and selected slot:

```text
Minus:   path_feasible false, path_reason BlockedPath, selected false
Plus:    path_feasible true, match_ready true,
         entry 114, contact 139, finite total_cost <= unchanged maximum_cost,
         selected true
```

Require exact frame/reason/identity equality and validate the live cost only as finite, nonnegative, available, and within the unchanged configured maximum. Live-flat cost may differ with the walking/settling pose. Keep this check behind the existing real-evidence/current-pack discriminator so synthetic unit fixtures can test other valid outcomes. The exact diagnostic costs belong only to the canonical-equivalent direct oracle in Task 3/Task 7 Step 5; never impose them on real live evidence or silently select Minus when the named pack disagrees.

- [ ] **Step 7: Run full evidence GREEN and independent validator review**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest tests.python.test_playable_placement_evidence -v
```

Expected: the full module passes, including all legacy timing/placement/attachment/video assertions and the seven new migration/corruption tests. Have a fresh reviewer start from raw synthetic JSON and recompute both arc roots, callback/consumed epoch handoff, fingerprint grouping, independent elapsed bound, eligibility, score/tie winner, failure-record boundary, freeze boundary, Interact count, and reservation timing without consulting controller-selected fields. Reject optional fields, permissive legacy fallback, rounded fingerprints, mixed epochs, or validation that merely repeats writer booleans. Do not commit.

---

### Task 7: Run Normal, Fast-Math, Regression, and Real 25 Hz Gates

**Files:**
- Verify only: all modified product/test files and generated evidence/video through their tracked Make targets
- Modify after successful verification: `.superpowers/sdd/task-8-report.md` (commands, outputs, evidence paths, independent reviews)
- Do not modify: protected repository-root artifact `interaction_query_probe`

**Interfaces:**
- Consumes: Tasks 1–6 as one uncommitted Task 8 review unit and the documented current interaction pack/display environment.
- Produces: reproducible verification evidence and a reviewed handoff. This task does not tune thresholds, manufacture evidence, or create an intermediate product commit.

- [ ] **Step 1: Review the complete diff before running gates**

Run only tracked-status/diff commands that cannot enumerate the protected untracked artifact:

```bash
git status --short --untracked-files=no
git diff --check
git diff -- interaction_matcher.h interaction_matcher.cpp interaction_runtime.h interaction_runtime.cpp controller.cpp Makefile tests/cpp/interaction_runtime_fixture.h tests/cpp/test_interaction_matcher.cpp tests/cpp/test_interaction_runtime.cpp tests/cpp/test_pick_entry_preview_fast_math.cpp tests/cpp/test_pick_entry_oracle.cpp tests/python/test_interaction_gate1.py tests/python/test_playable_placement_evidence.py
```

Confirm every hunk belongs to Tasks 1–6 or preserved dirty Task 8 work. Do not reset/reformat unrelated hunks, run an untracked-file listing, stage files, or inspect the protected repository-root artifact.

- [ ] **Step 2: Run focused normal C++ and controller/evidence checks**

Run from the repository root:

```bash
make build/tests/test_interaction_matcher build/tests/test_interaction_runtime build/tests/test_pick_entry_oracle
./build/tests/test_interaction_matcher
./build/tests/test_interaction_runtime
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests.test_pick_entry_preview_release_fast_math_target_compiles_and_runs \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests.test_safe_suite_executes_pick_entry_preview_release_fast_math \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests.test_pick_entry_oracle_target_compiles_without_probe_dependency -v
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest tests.python.test_playable_placement_evidence -v
make gate-place-headless
```

Expected: all commands exit 0. The Make dry-run tests prove the new rules compile/run the intended sources and keep the oracle isolated from the protected probe dependency. The headless gate recompiles the controller path and retains existing placement state/timing checks. A compile warning/error, changed legacy matcher result, runtime mutation assertion, skipped dry-run test, or validator relaxation blocks progression.

- [ ] **Step 3: Run the assertion-independent release-fast-math preview canary**

Run:

```bash
make test-pick-entry-preview-release-fast-math
```

Expected: the dedicated target compiles with `-O3 -DNDEBUG -ffast-math`, executes its explicit non-`assert` failure checks, and exits 0. Record its compiler command and output in the Task 8 report. Do not substitute the normal debug binary or treat disabled assertions as coverage.

- [ ] **Step 4: Run interaction and placement regressions through safe targets**

Run:

```bash
make test-interaction-safe
make gate-place-headless
```

Expected: both exit 0. `test-interaction-safe` must continue using `build/task12/interaction_query_probe_safe` through the existing Make dependency; do not invoke or inspect any same-named repository-root artifact. Investigate any lifecycle, carry, place, query, timing, or evidence regression before the real gate.

- [ ] **Step 5: Verify the preregistered pack oracle**

Run:

```bash
./build/tests/test_pick_entry_oracle resources/g1_interaction
```

Expected output consists only of three compact result lines and matches:

```text
R accepted entry=114 contact=139 cost=0.666622
Minus rejected path_reason=BlockedPath
Plus accepted entry=114 contact=139 cost=0.718336
```

The executable enforces `2e-6` cost tolerance. If the pack is absent or differs, stop and report the pack identity/problem; do not retune costs, collision bounds, slot geometry, or evidence expectations.

- [ ] **Step 6: Run the real graphical 25 Hz placement gate**

Require the caller's documented Grail and robot XML paths, then run the existing real gate exactly once per investigation cycle:

```bash
GRAIL_ROOT="${GRAIL_ROOT:?GRAIL_ROOT must name the Grail checkout}" \
G1_XML="${G1_XML:?G1_XML must name the G1 XML}" \
DISPLAY="${DISPLAY:-:1}" \
PATH="$GRAIL_ROOT/.venv/bin:$PATH" \
make gate-playable-placement
```

Expected: real Raylib/X11 rendering succeeds at exact 25 Hz, the controller completes one pickup and placement, the gate captures its ordinary evidence/video artifacts, and the strict validator passes. This is the acceptance gate; a diagnostic-only, canonical-equivalent, headless, or mocked run cannot replace it.

Inspect the validator summary and retained evidence through the existing gate output paths. Require one live-flat fingerprint per paired epoch, Plus then Minus calls, no preview mutation/reservation, `Minus=BlockedPath`, ready Plus at frames `114/139` with finite available cost within the unchanged matcher maximum, freeze to Plus before slot navigation, one Interact edge, normal reservation/Preflight/playback/attachment/place lifecycle, and exact 25 Hz tick/evidence/video cadence. Do not compare the live-flat cost to the canonical-equivalent oracle's exact diagnostic cost.

- [ ] **Step 7: Perform final independent reviews**

Before declaring success, obtain and record these independent checkpoints:

1. Matcher/runtime review: first-hard-path frame metadata, explicit finite-cost availability, unchanged mixed-failure priority, shared hard predicate ownership, legacy selection parity, bitwise fast-math-safe mapping validation, const/no-mutation proof, and normal/fast-math agreement.
2. Geometry/controller review: exact Plus/Minus arcs, paired calls inside the one provider callback, identical returned snapshot, prior-epoch steering handoff, independent 250-tick elapsed bound, eligibility-before-score, hand tie rule, last-pair-before-failure evidence, freeze-before-motion, and one Interact/no retry.
3. Evidence/Make review: strict migrated schema, independent reconstruction, pickup/place preview counter separation, current-pack values, callback/fingerprint/mutation/reservation/timing enforcement, and dry-run dependency assertions for both new targets.
4. Visual real-gate review: target remains visible, pickup follows the selected feasible side, contact/attachment/lift/carry/place appear coherent, and video duration/cadence agree with evidence.

Any review finding returns to the smallest owning RED test and repeats all downstream gates; do not patch around the validator or accept a verbal exception.

- [ ] **Step 8: Update the Task 8 report and leave one review unit uncommitted**

After every command above is freshly green, update `.superpowers/sdd/task-8-report.md` with exact commands, exit status, compiler mode, current pack path/identity, three oracle lines, real evidence/video paths, selected slot and preview outcomes, lifecycle/timing counters, and named independent review results. Remove temporary diagnostics and re-run:

```bash
git diff --check
git status --short --untracked-files=no
```

Expected: only intended tracked Task 8 files are reported; no temporary diagnostic source/output remains in tracked paths. Do not stage or create an intermediate product commit. Hand the single dirty review unit back for the existing Task 8 final verified commit boundary.

---
