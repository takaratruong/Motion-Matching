# G1 Bounded Candidate Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the legacy motion-match result as slot zero, then lazily enumerate and dual-certify at most six better strict-recovery transitions plus the incumbent so IK-off and IK-on choose the same first physically valid candidate within `K = 8`.

**Architecture:** Keep the existing `database_search` call and its public cost/provenance owner intact in the fast controller translation unit. Add one separately compiled strict-FP recovery provider that owns raw-query normalization, seeded row/AABB scoring, both incumbent admission bounds, and a fixed top-six set; after a finite slot-zero rejection, the frame coordinator evaluates each fixed record from fresh preallocated transaction-local state copies through common, raw, and IK branches. Commit only the first dual-certified record, persist the successful IK branch's `G1IkState` in both modes, project either the canonical raw or IK public pose by mode, and publish the complete first failing scratch only when the bounded set is exhausted.

**Tech Stack:** C++17; IEEE-754 binary32; existing small/large motion-database AABBs; strict-FP standalone objects; existing G1 footprint, root-reach, swing, IK, FK, clearance, frame-transaction, CSV, Gate E, and Gate L infrastructure; Python 3 runtime-log assertions.

## Global Constraints

- Implement the frozen design at `docs/superpowers/specs/2026-07-16-g1-bounded-candidate-certification-design.md`; its required SHA-256 is `10c21cb51a351b9cc2f4c1e03d58472e485bba36d0449967de4e08e33a7ec6dd`.
- Do not start source execution unless certified clamp-boundary commit `29ef804`, certified root-frontier commit `f6a6448`, and frozen design commit `07f9c7c` are ancestors of `HEAD` (`29ef804` descends the other two), or while another worker has uncommitted ownership of `ik.h`, `g1_ik.h`, `g1_ik_root_reach.cpp`, `tests/cpp/test_g1_ik.cpp`, or `tests/cpp/g1_root_reach_live_fixture_bits.h`. This plan never edits those five files and preserves their certified behavior.
- `K` is exactly 8: one unchanged legacy slot zero, at most six distinct strict-recovery transitions, and one incumbent continuation, with the incumbent omitted when slot zero already attempted it.
- Keep exactly one production `database_search` call. It remains the sole slot-zero query, normalization, exclusions, range choice, selected frame, transition flag, public `incumbent_cost`, and public `selected_cost` owner.
- If slot zero dual-accepts or globally errors, recovery enumeration and recovery evaluation are both zero. Only a finite slot-zero rejection on a scheduled-search frame may invoke exactly one additional accelerated traversal.
- Matching-disabled and non-search frames attempt only the incumbent. A scheduled legacy incumbent is not retried, although a finite rejection may still trigger one strict traversal for admissible transitions.
- Recovery owns all 31 raw-query normalization operations. Exact `FLT_MAX` scale bits materialize positive zero; every other scale is positive finite; all raw query/offset values and all subtraction/division intermediates are finite.
- Recovery-transition scoring starts at the exact transition-cost word. Private incumbent scoring starts at positive zero. Each dimension executes a separate binary32 subtract, multiply, and add in ascending order; never construct a zero-seeded transition sum, base cost, score delta, or score window.
- The same strict owner evaluates all large/small AABB lower bounds and pruning. Use the existing `BOUND_LR_SIZE = 64` and `BOUND_SM_SIZE = 16` hierarchy; no brute-force production scan and no repeated single-best search are permitted.
- Admit a transition only when it passes unchanged range-end/surrounding exclusions, differs from slot zero and the incumbent, is finite/nonnegative and strictly below the private finite incumbent score, and—unless the public word is exact `FLT_MAX`—is also strictly below public `incumbent_cost`. Equality fails.
- Retain at most six transitions ordered by numeric strict recovery score then database frame. A later better row evicts the current worst. Emit the ordered transitions and then the deduplicated incumbent.
- A recovery win publishes its strict score in the existing `selected_cost`; a legacy win keeps legacy cost bits. At end of animation, private incumbent score never replaces either public `FLT_MAX` sentinel.
- Every record begins from a fresh deep copy of one immutable post-input/post-matcher baseline. Evaluate common stages once, raw/no-IK first, and IK only after raw acceptance. A raw failure never executes IK; no stage executes after the first dual acceptance.
- Contact state is neither an eligibility gate nor a rank input. A valid immediate release can win; an invalid double-support candidate must fail its physical certificate.
- Existing `state.ik` is the only persistent hidden-certification owner and always receives the successful IK branch's next `G1IkState`. No accepted-state field, log column, or persistent hidden-pose/certificate owner is added.
- With IK off, accepted pose/result/clearance owners remain the canonical disabled/raw projection while hidden `state.ik` persists. With IK on, the existing public IK owners receive the successful IK products.
- The IK-off CSV suffix is exactly `motion_match_ik_diagnostic{}`. It cannot read `state.ik`, locks, swing history, root reach, lift, residual, hidden IK result/pose, or hidden clearance; its text defaults remain `none`/`invalid-input` and swing indices remain `UINT32_MAX`.
- Preserve the complete first failing `G1FrameTransactionScratch` by value. Later attempts cannot overwrite it. Exhaustion publishes that scratch through the existing footprint/IK/checkpoint authenticator, latches once, increments presentation once, and leaves accepted common, route, pose, and hidden IK state unchanged.
- Any provider, common, raw, IK, saved-scratch, or provenance global error aborts immediately and is never downgraded to a finite retry.
- Preserve exact `dt` bits `0x3d23d70a` at recovery, common, raw, and IK boundaries. Travel command and desired heading snapshots are immutable across attempts; no candidate rotation derives heading.
- Do not change any footprint, root-reach, shell, residual, correction, swing, defensive-clearance, or whole-pose threshold: in particular retain the 5 cm root-reach adjustment cap, 5 mm contact residual limit, 41 exact swing-lift words through the 8 cm maximum, and existing `-0.005 m` toe/foot and `-0.01 m` whole-pose clearance limits. Preserve contact derivation, terrain normal/sole/lock semantics, finite/global classification, every nested work budget, feature weights, route, heading policy, and exact 25 Hz sample rate; introduce no 60 Hz assumption, substep, or resampling.
- Do not modify `motion_match_log.h`, `resources/check_g1_runtime_log.py`, any terrain/database artifact, scene, schema, threshold, route, mesh, renderer, or `resources/g1_visualizer_process.py`.
- The fixed-capacity candidate trace and exhaustive oracle exist only under test seams, fail rather than truncate a ninth record, are written after selection, and never become a selection input or production log field.
- The authentic low-curb oracle must operate on the same in-memory `database` object and exact recovery request used by the live controller process. Cross-build diagnostic score words are density evidence only and are never hard-coded as recovery expectations.
- Compile `g1_clearance.cpp`, `g1_ik_root_reach.cpp`, and the new `g1_candidate_recovery.cpp` with `-fno-fast-math -ffp-contract=off -frounding-math`. Callers may use `-ffast-math`; final links may not. Do not use LTO.
- A separate zero-rejection 32-frame IK-off/IK-on low-curb gate is mandatory before any 800- or 832-frame command. The 800-frame release logs contain exactly 800 rows. Performance runs discard exactly 32 warm-up transactions and measure the following 800 transactions only.
- Each of three IK-on performance runs must have mean transaction time at most 40 ms and nearest-rank p99 at most 80 ms. Timing surrounds only `g1_frame_transaction_run`, is stored outside deterministic logs, and cannot influence choice.
- Build and certify disposable binaries only. Do not discover, signal, replace, restart, or otherwise touch the currently running visualizer.
- Preserve unrelated work. Before each commit, stage only the files named in that task and inspect `git diff --cached --name-only`.

## File Structure

- Create `g1_candidate_recovery.h`: fixed capacities, recovery request/record/set/work types, production provider signature, and seam-only audit/exhaustive declarations.
- Create `g1_candidate_recovery.cpp`: the only strict owner of recovery normalization, seeded scoring, AABB pruning, two-bound admission, top-six retention, and the seam-only exhaustive traversal.
- Create `tests/cpp/test_g1_candidate_recovery.cpp`: strict provider/oracle fixtures, fixed-bit arithmetic inversions, malformed inputs, exclusions, capacity, work, strict/fast caller parity, and authentic request hooks.
- Create `tests/cpp/compile_g1_candidate_recovery_production.cpp`: positive no-seam signature/link fixture.
- Create `tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp`: compile-time privacy fixture for the exhaustive/audit interface.
- Modify `g1_frame_transaction.h`: candidate stages, branch scratch, fixed workspace, recovery-provider dependency, bounded coordinator, hidden-state commit, first-failure ownership, trace types, preflight, validation, reset, and publication.
- Modify `scene_switch.h`: extend runtime preflight, reset validation, candidate isolation, atomic swap, and rollback from two controller states to all five runtime states.
- Modify `controller.cpp`: split matcher preparation from candidate application; implement common/raw/IK/finalize stages; pass the strict provider; add canonical disabled log projection; and add seam-only trace/timing output after the transaction.
- Create `g1_candidate_certification_trace.h`: seam-only TSV serialization, post-transaction same-matrix exhaustive comparison, and timing-file support; an empty no-seam preprocessor branch exposes no type or symbol.
- Modify `tests/cpp/test_g1_frame_transaction.cpp`: generic A/B/C coordinator, ordering, poison, hidden ownership, full first-failure, global-error, capacity, and exact-`dt` tests.
- Modify `tests/cpp/test_scene_switch.cpp`: five-state reset/switch validation, ten-way live/candidate storage isolation, atomic swap, and failure rollback.
- Modify `tests/cpp/test_g1_frame_transaction_production.cpp`: real runner A/B/C, genuine swing rejection, legacy slot-zero identity, provider laziness, work counters, mode parity, source/closure/lexical guards, and no-seam integration.
- Modify `tests/cpp/test_g1_controller_logging.cpp`: canonical disabled projection, hidden-state poison, schema identity, selected-cost/provenance, and accepted-state digest ownership.
- Verify unchanged `tests/cpp/compile_g1_frame_transaction_runner_negative.cpp`: retain all existing forbidden context modes; the stage-runner type is unchanged, so this fixture requires no edit.
- Verify only `database.h`, `g1_ik.h`, `g1_ik_root_reach.cpp`, `g1_ik_runtime.h`, `g1_clearance.h`, `g1_clearance.cpp`, `motion_match_log.h`, `resources/check_g1_runtime_log.py`, and all terrain assets. Leave the running process entirely undiscovered and untouched.

---

### Task 1: Add the Strict Bounded Recovery Provider

**Files:**
- Create: `g1_candidate_recovery.h`
- Create: `g1_candidate_recovery.cpp`
- Create: `tests/cpp/test_g1_candidate_recovery.cpp`
- Create: `tests/cpp/compile_g1_candidate_recovery_production.cpp`
- Create: `tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp`
- Reference only: `database.h`
- Reference only: `docs/superpowers/specs/2026-07-16-g1-bounded-candidate-certification-design.md`

**Interfaces:**
- Consumes: immutable `database` feature/range/bound arrays, an exact 31-word raw query snapshot, incumbent/legacy frames, transition cost, public incumbent cost, and unchanged exclusion counts.
- Produces exactly these public types and signatures:

```cpp
static constexpr uint32_t G1RecoveryFeatureCount = 31U;
static constexpr uint32_t G1CandidateAttemptCapacity = 8U;
static constexpr uint32_t G1RecoveryTransitionCapacity = 6U;
static constexpr uint32_t G1RecoveryTailCapacity = 7U;

enum G1CandidateKind : uint32_t
{
    G1CandidateLegacy = 0U,
    G1CandidateRecoveryTransition,
    G1CandidateIncumbent,
};

struct G1CandidateRecord
{
    G1CandidateKind kind = G1CandidateIncumbent;
    int selected_frame = -1;
    int executed_frame = -1;
    int source_range = -1;
    float selected_cost = FLT_MAX;
    uint32_t recovery_rank = UINT32_MAX;
    bool transitioned = false;
};

struct G1RecoveryRequest
{
    const database* db = nullptr;
    float raw_query[G1RecoveryFeatureCount] = {};
    int incumbent_frame = -1;
    int legacy_selected_frame = -1;
    float transition_cost = 0.0f;
    float public_incumbent_cost = FLT_MAX;
    int ignore_range_end = 20;
    int ignore_surrounding = 20;
};

struct G1RecoveryWork
{
    uint32_t accelerated_traversals = 0U;
    uint32_t large_bounds_tested = 0U;
    uint32_t small_bounds_tested = 0U;
    uint32_t rows_tested = 0U;
    uint32_t full_scores_materialized = 0U;
};

struct G1RecoveryCandidateSet
{
    G1CandidateRecord records[G1CandidateAttemptCapacity] = {};
    uint32_t count = 0U;
    G1RecoveryWork work;
};

enum G1RecoveryProviderStatus : uint32_t
{
    G1RecoveryProviderOk = 0U,
    G1RecoveryProviderGlobalError,
};

using G1RecoveryProvider = G1RecoveryProviderStatus (*)(
    G1RecoveryCandidateSet&,
    const G1RecoveryRequest&,
    char*,
    int);

G1RecoveryProviderStatus g1_recovery_candidates_build(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    char* error,
    int error_capacity);
```

- Under `G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM` only, produces these additional declarations; none is declared in a production include:

```cpp
struct G1RecoveryProviderAudit
{
    uint32_t normalized_query_bits[G1RecoveryFeatureCount] = {};
    uint32_t recovery_incumbent_score_bits = 0U;
    uint32_t exhaustive_rows_tested = 0U;
};

G1RecoveryProviderStatus g1_recovery_candidates_build_audited(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryProviderAudit& audit,
    char* error,
    int error_capacity);

G1RecoveryProviderStatus g1_recovery_candidates_exhaustive_for_test(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryProviderAudit& audit,
    char* error,
    int error_capacity);
```
- `G1RecoveryCandidateSet::records` has eight physical entries so a hostile seam can represent `count == 8` without memory corruption; production output validation permits at most seven tail records.

- [ ] **Step 1: Freeze the execution base and legacy evidence before the first feature edit**

Run from the worktree root after the clamp-boundary owner has committed and released the five excluded files:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert
mkdir -p "$out/base/build" "$out/base/logs" \
  "$out/base/current-frontier"
test "$(sha256sum docs/superpowers/specs/2026-07-16-g1-bounded-candidate-certification-design.md | cut -d' ' -f1)" = \
  10c21cb51a351b9cc2f4c1e03d58472e485bba36d0449967de4e08e33a7ec6dd
git merge-base --is-ancestor f6a6448 HEAD
git merge-base --is-ancestor 07f9c7c HEAD
git merge-base --is-ancestor 29ef804 HEAD
git diff --exit-code -- \
  ik.h g1_ik.h g1_ik_root_reach.cpp tests/cpp/test_g1_ik.cpp \
  tests/cpp/g1_root_reach_live_fixture_bits.h
git rev-parse HEAD > "$out/base/execution-base.txt"
git status --short > "$out/base/status.before"
sha256sum motion_match_log.h resources/check_g1_runtime_log.py \
  > "$out/base/immutable-files.before.sha256"

g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o "$out/base/build/clearance.o"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_ik_root_reach.cpp \
  -o "$out/base/build/root-reach.o"
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src -c controller.cpp \
  -o "$out/base/build/controller.o"
g++ "$out/base/build/controller.o" "$out/base/build/clearance.o" \
  "$out/base/build/root-reach.o" -o "$out/base/controller" \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
for ik in 0 1; do
  DISPLAY=:1 MM_IK="$ik" "$out/base/controller" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene grail-curb-low --test-mode route \
    --test-route curb-forward --test-frames 32 --test-heading forward \
    --terrain-weight 4 --log "$out/base/logs/low-curb-${ik}.csv"
done

# Freeze only short current-frontier behavior evidence before feature work.
# The first 800-frame current-frontier run is forbidden until Task 6's
# ZERO_REJECTION_32.PASS marker exists.
gate_l_routes=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  grail-curb-low:curb-forward
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
)
for specification in "${gate_l_routes[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  stem="${scene}__${route}"
  for ik in 0 1; do
    DISPLAY=:1 MM_IK="$ik" "$out/base/controller" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene "$scene" --test-mode route --test-route "$route" \
      --test-frames 32 --test-heading forward --terrain-weight 4 \
      --log "$out/base/current-frontier/forward32-${stem}-${ik}.csv"
  done
done

flat_cases=(
  flat-positive-z:forward:forward
  flat-positive-z:backward:backward
  flat-positive-z:positive-x:left
  flat-positive-z:negative-x:right
  flat-positive-x:positive-x:forward
  flat-positive-x:negative-x:backward
  flat-positive-x:forward:right
  flat-positive-x:backward:left
)
for specification in "${flat_cases[@]}"; do
  IFS=: read -r route heading relative <<<"$specification"
  stem="${route}__${heading}__${relative}"
  DISPLAY=:1 MM_IK=0 "$out/base/controller" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene stairs-shallow --test-mode route --test-route "$route" \
    --test-frames 100 --test-heading "$heading" --terrain-weight 4 \
    --log "$out/base/current-frontier/flat-${stem}.csv"
done
sha256sum "$out/base/controller" \
  "$out/base/current-frontier"/*.csv \
  > "$out/base/current-frontier/SHA256SUMS"
sha256sum -c "$out/base/current-frontier/SHA256SUMS"
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv, struct
root = "/tmp/g1-bounded-candidate-cert/base/logs"
with open(f"{root}/low-curb-0.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 32
row = rows[6]
bits = lambda text: struct.unpack("<I", struct.pack("<f", float(text)))[0]
assert row["query_database_frame"] == "272446", row
assert row["selected_database_frame"] == "428956", row
assert row["source_range"] == "1647", row
assert bits(row["selected_cost"]) == 0x40250630, row
assert row["query_bits_hex"] == (
    "3de11a0a3d574158bd3b1ecabe0478ec3d573390bc5b68fabb8c82fcb9a49830"
    "b86f6000bb04f968ba8ea77cba905e00be23d1cebd34d0483d851794bd34d7ba3d"
    "c62330bde623773e683e17be3fe67f3ebd5aa9be3635e23f7bea21be64911f3f79"
    "8a8ebe7176033f78c7f600000000000000003db7f28400000000")
print("VALID frozen legacy row-6 owner")
PY
```

Expected: every command exits `0`; the spec hash matches; all five excluded files are clean; the neutral link succeeds; the IK-off baseline freezes the authoritative row-6 query, selected frame/range, and `0x40250630` public score; and the frozen controller, ten 32-frame forward captures, and eight 100-frame flat captures authenticate as one current-frontier evidence set. No 800-frame process has run. Keep this `/tmp` tree through Tasks 1–6.

- [ ] **Step 2: Write the provider RED tests and compile-only ownership fixtures**

Create `tests/cpp/test_g1_candidate_recovery.cpp` with a local `CHECK`, `float_bits`, `float_from_bits`, a 31-feature synthetic `database` builder, and these exact test entry points:

```cpp
static void test_strict_normalization_owns_all_31_words();
static void test_seeded_transition_and_zero_seeded_incumbent_are_distinct();
static void test_both_incumbent_admission_bounds_are_strict();
static void test_top_six_ties_eviction_dedup_and_incumbent_order();
static void test_exclusions_sequential_idle_and_end_of_animation();
static void test_accelerated_provider_matches_exhaustive_oracle();
static void test_malformed_alias_overflow_and_capacity_fail_closed();

int main(int argc, char** argv)
{
    test_strict_normalization_owns_all_31_words();
    test_seeded_transition_and_zero_seeded_incumbent_are_distinct();
    test_both_incumbent_admission_bounds_are_strict();
    test_top_six_ties_eviction_dedup_and_incumbent_order();
    test_exclusions_sequential_idle_and_end_of_animation();
    test_accelerated_provider_matches_exhaustive_oracle();
    test_malformed_alias_overflow_and_capacity_fail_closed();
    if (argc == 2 && std::strcmp(argv[1], "--parity") == 0) {
        print_provider_parity_transcript();
    }
    return 0;
}
```

The normalization test must use 31 distinct ordinary query/offset/scale words, one exact `0x7f7fffff` disabled scale, and strict/fast callers. Require positive-zero output for the disabled dimension and exact audited words for every other dimension. Independently pass `+Inf`, `-Inf`, and NaN query/offset values; positive/negative infinity, NaN, `+0`, `-0`, and negative enabled scales; a subtraction overflow; and a division overflow. Each returns `G1RecoveryProviderGlobalError`, leaves a poison-filled output unchanged, and emits no list.

The two-bound test uses one-ULP-below/equal/one-ULP-above values for each bound with the other bound loose. It also uses exactly:

```cpp
const float private_score = float_from_bits(UINT32_C(0x4040097d));
const float public_score = float_from_bits(UINT32_C(0x404005f1));
const float gap_candidate = float_from_bits(UINT32_C(0x40400700));
```

Require `gap_candidate < private_score` but exclusion by the lower public owner. Re-run with exact public `FLT_MAX`; require admission against the finite private owner and no synthesized public finite value.

The seeded-order test constructs query positive zeros and sign-negates the following exact difference words so `query - database` recreates them. Place A and B at the sixth retained boundary behind five unambiguously lower production rows, use transition seed `0x3f800000`, and require B retained/A evicted:

```text
A: 3e76b806 3ea08e4a bed358ea bef09450 bf6f7616 bffbad3a 3ca7c2d5 be90bb1d bf462b72 bebc2ed2 3eb6d0d9 bf52f8b9 bebcdda3 3e44dcbc 3f489f8e 3f3e1335 bed05f70 3d8da082 be870002 3e12312d bf26d190 bf309386 3c7106e2 bcb0dfde 3e95b2a6 bf8594b6 3f1a3b4e be757c29 3e9389b0 3f54c596 3eb57e61
B: 3f21629e 3f585617 3faea056 bf1fba50 3f038e09 3d84b2a9 bcb7213c bf1c6c11 3fb3b6b2 3da1eb7d bf0437fd 3cdbcd5c be41121a 3e581e78 3edcba12 3f656afc bdb8ec54 3ec1cd11 bf68ed43 3e4af82f bdf8d97a bf777d29 3ed3b084 3f8d4823 be2999bd bf05d2f2 3e338467 3eb032f5 3f4fedb4 3d78fac2 beae08de
```

Assert test-only zero sums A/B are `0x413e77bb`/`0x413e77bc`, production seeded scores are `0x414e77bc`/`0x414e77bb`, and no provider output/audit field stores either zero-seeded transition sum.

The ranking tests cover zero through eight otherwise-admissible rows, equal-score frame ties, late eviction, range boundaries, the 20-frame range-end and surrounding exclusions, slot-zero/incumbent deduplication, clamped `+1` execution, idle transition seed, and both end-of-animation outcomes. Every accelerated set must bit-match `g1_recovery_candidates_exhaustive_for_test` and have `accelerated_traversals == 1`, fixed storage, at most six transition records, a final incumbent only when needed, and total tail count at most seven.

Create `compile_g1_candidate_recovery_production.cpp` with `static_assert(std::is_same<G1RecoveryProvider, decltype(&g1_recovery_candidates_build)>::value, "production recovery provider signature changed")` and one well-typed call. Create `compile_g1_candidate_recovery_seam_negative.cpp` that calls `g1_recovery_candidates_exhaustive_for_test`; without the seam macro this source must fail during compilation because the identifier is not declared.

Run RED:

```bash
set -o pipefail
mkdir -p /tmp/g1-bounded-candidate-cert/task1
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
     -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM -I. \
     tests/cpp/test_g1_candidate_recovery.cpp \
     -o /tmp/g1-bounded-candidate-cert/task1/provider-red \
     2>/tmp/g1-bounded-candidate-cert/task1/provider-red.stderr; then
  echo 'ERROR: provider RED unexpectedly compiled' >&2
  exit 1
fi
rg 'g1_candidate_recovery|No such file|not declared' \
  /tmp/g1-bounded-candidate-cert/task1/provider-red.stderr
```

Expected: compilation fails only because the new provider contract/definitions do not exist.

- [ ] **Step 3: Implement the strict provider with one accelerated traversal**

Add the exact Task 1 interfaces to `g1_candidate_recovery.h`. Put `#if defined(__FAST_MATH__) #error "g1_candidate_recovery.cpp requires strict floating-point compilation" #endif` at the top of `g1_candidate_recovery.cpp`. Implement private checked helpers with these responsibilities and names:

```cpp
struct G1RecoveryStrictAuditSink
{
    uint32_t normalized_query_bits[G1RecoveryFeatureCount] = {};
    uint32_t recovery_incumbent_score_bits = 0U;
    uint32_t exhaustive_rows_tested = 0U;
};

static bool g1_recovery_request_is_valid(
    const G1RecoveryRequest&, char*, int);
static bool g1_recovery_normalize_query(
    float (&normalized)[G1RecoveryFeatureCount],
    const G1RecoveryRequest&,
    G1RecoveryStrictAuditSink*);
static bool g1_recovery_score_row(
    float& score, const database&, int frame,
    const float (&normalized)[G1RecoveryFeatureCount], float seed);
static bool g1_recovery_score_bound(
    float& score, const slice2d<float> minimum,
    const slice2d<float> maximum, int box,
    const float (&normalized)[G1RecoveryFeatureCount], float seed,
    float strict_limit, bool equality_loses);
static bool g1_recovery_record_less(
    const G1CandidateRecord&, const G1CandidateRecord&);
static bool g1_recovery_insert_top_six(
    G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t& retained_count, const G1CandidateRecord&);
```

`G1RecoveryStrictAuditSink` is a private `g1_candidate_recovery.cpp` type compiled in both modes. Seam wrappers copy its words into public `G1RecoveryProviderAudit`; production helpers never depend on a seam-only type.

For every arithmetic operation, assign subtraction, multiplication, and addition to separate `float` lvalues in source order and reject a nonfinite result before the next operation. Compare scale bits to `UINT32_C(0x7f7fffff)` before arithmetic and write `0.0f` directly for disabled dimensions. Compute the private incumbent score completely from `0.0f` before traversal.

Validate the complete database view before traversal: 31 feature/offset/scale columns, consistent frame/range/contact/pose extents, ordered in-bounds ranges, correctly sized small/large bound arrays, finite ordered bound endpoints, finite row features as encountered, valid incumbent/legacy frames, nonoverlapping request/output storage, and counter arithmetic that cannot overflow. Traverse each validated range with the same global 64/16 box indices as `motion_matching_search`. Use private/public incumbent bounds and, after six retained records, the worst retained score as pruning limits. Prune equality against either incumbent because equality cannot be admitted; do not prune equality against the worst top-six score because a lower database frame can win the tie. Fully materialize every retained row score. After sorting, assign recovery ranks `0..count-1`, append one ordinary incumbent record when `legacy_selected_frame != incumbent_frame`, and validate the complete set transactionally before assigning `output`.

Implement `g1_recovery_candidates_exhaustive_for_test` under the seam macro as a range-by-range row scan using the same strict normalization/row-score owner but no AABB calls. It is test evidence only, never called by `g1_recovery_candidates_build`, and reports `accelerated_traversals == 0`.

- [ ] **Step 4: Run provider GREEN, strict/fast parity, sanitizer, privacy, and no-fast-math gates**

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task1
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math)
san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all)

g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM -I. \
  -c g1_candidate_recovery.cpp -o "$out/recovery-seam.o"
for mode in strict fast; do
  flags=(-O2 -Wall -Wextra -Werror -pedantic)
  test "$mode" = fast && flags=(-O3 -ffast-math -DNDEBUG)
  g++ -std=c++17 "${flags[@]}" \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM -I. \
    -c tests/cpp/test_g1_candidate_recovery.cpp \
    -o "$out/provider-${mode}.o"
  g++ "$out/provider-${mode}.o" "$out/recovery-seam.o" \
    -o "$out/provider-${mode}"
  "$out/provider-${mode}"
  "$out/provider-${mode}" --parity > "$out/provider-${mode}.txt"
done
cmp "$out/provider-strict.txt" "$out/provider-fast.txt"

g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM -I. \
  -c g1_candidate_recovery.cpp -o "$out/recovery-san.o"
g++ "${san[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM -I. \
  -c tests/cpp/test_g1_candidate_recovery.cpp \
  -o "$out/provider-san.o"
g++ "${san[@]}" "$out/provider-san.o" "$out/recovery-san.o" \
  -o "$out/provider-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$out/provider-san"

if g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
     -c g1_candidate_recovery.cpp -o "$out/forbidden-fast.o" \
     2>"$out/forbidden-fast.stderr"; then
  echo 'ERROR: strict recovery owner accepted fast math' >&2
  exit 1
fi
rg 'FAST_MATH|fast-math|strict' "$out/forbidden-fast.stderr"

g++ "${strict[@]}" -I. -c g1_candidate_recovery.cpp \
  -o "$out/recovery-production.o"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/compile_g1_candidate_recovery_production.cpp \
  -o "$out/production-caller.o"
g++ "$out/production-caller.o" "$out/recovery-production.o" \
  -o "$out/production-link"
"$out/production-link"
! nm -C "$out/recovery-production.o" | \
  rg 'exhaustive_for_test|ProviderAudit|build_audited'
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
     -c tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp \
     -o "$out/seam-negative.o" 2>"$out/seam-negative.stderr"; then
  echo 'ERROR: recovery test seam compiled in production mode' >&2
  exit 1
fi
rg 'g1_recovery_candidates_exhaustive_for_test|not declared' \
  "$out/seam-negative.stderr"
```

Expected: strict and fast callers pass and emit byte-identical transcripts; accelerated and exhaustive records match bit-for-bit; sanitizers are silent; the strict owner rejects fast compilation; production links; and no seam symbol survives the no-seam object.

- [ ] **Step 5: Commit the independently reviewable provider**

```bash
git add g1_candidate_recovery.h g1_candidate_recovery.cpp \
  tests/cpp/test_g1_candidate_recovery.cpp \
  tests/cpp/compile_g1_candidate_recovery_production.cpp \
  tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: add strict bounded recovery provider"
```

Expected staged paths: exactly the five Task 1 files.

---

### Task 2: Implement the Bounded Dual-Certification Coordinator and Hidden IK Ownership

**Files:**
- Modify: `g1_frame_transaction.h:46-258,668-1378,2450-3051`
- Modify: `scene_switch.h:20-506`
- Modify: `tests/cpp/test_g1_frame_transaction.cpp:1-4187`
- Modify: `tests/cpp/test_scene_switch.cpp:1-850`
- Verify only: `tests/cpp/compile_g1_frame_transaction_runner_negative.cpp`
- Reference only: `g1_controller_state.h`
- Consume: Task 1 provider contract and implementation

**Interfaces:**
- Retains the stage-runner function type:

```cpp
using G1FrameStageRunner = G1FrameStageOutcome (*)(
    G1FrameTransactionStage,
    g1_controller_state&,
    G1FrameTransactionScratch&,
    const G1FrameExternalInputs&,
    char*,
    int);
```

- Replaces the twelve-stage enum with this exact phase order:

```cpp
enum G1FrameTransactionStage
{
    G1FrameStageInputRouteCommand = 0,
    G1FrameStageMatcherSearch,
    G1FrameStageCandidateApply,
    G1FrameStageInertialization,
    G1FrameStageSimulationUpdate,
    G1FrameStageSupportObservation,
    G1FrameStageSupportRetarget,
    G1FrameStageContactUpdate,
    G1FrameStageFootprintObservation,
    G1FrameStageRawBegin,
    G1FrameStageRawFirstFoot,
    G1FrameStageRawSecondFoot,
    G1FrameStageRawFinalFk,
    G1FrameStageRawPoseCertificate,
    G1FrameStageIkBegin,
    G1FrameStageIkFirstFoot,
    G1FrameStageIkSecondFoot,
    G1FrameStageIkFinalFk,
    G1FrameStageIkPoseCertificate,
    G1FrameStageAcceptedFinalize,
    G1FrameStageCount,
};
```

- Adds selector-local workspace storage to `G1FrameRuntime`, not to `g1_controller_state`:

```cpp
struct G1FrameCandidateWorkspace
{
    g1_controller_state common_state;
    g1_controller_state raw_state;
    g1_controller_state ik_state;
};

struct G1FrameRuntime
{
    g1_controller_state accepted_state;
    g1_controller_state working_state;
    G1FrameCandidateWorkspace candidates;
    G1FramePublication publication;
    G1FrameAcceptedDiagnostic accepted_diagnostic;
};
```

- Adds exact branch scratch without adding accepted-state fields:

```cpp
enum G1FrameCertificateBranch : uint32_t
{
    G1FrameCertificateNone = 0U,
    G1FrameCertificateRaw,
    G1FrameCertificateIk,
};

struct G1FrameBranchCertificateScratch
{
    G1IkFrameTransaction ik_transaction;
    G1ClearanceStatus pose_status = G1ClearanceInvalidInput;
    G1PoseClearance pose_clearance;
};
```

`G1FrameTransactionScratch` gains `slot_zero_record`, `active_candidate`, `matching_scheduled`, `legacy_search_performed`, `transition_cost`, `recovery_request_ready`, `recovery_request`, `raw_certificate`, `ik_certificate`, and `rejection_branch`. Replace the old single `ik_transaction`, `pose_status`, and `pose_clearance` owners with the two branch owners everywhere.

- Extends `g1_frame_transaction_run` with the strict dependency immediately after the stage runner:

```cpp
static inline G1FrameTransactionStatus g1_frame_transaction_run(
    G1FrameRuntime& runtime,
    G1FrameStageRunner run_stage,
    G1RecoveryProvider recovery_provider,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity);
```

- Under `G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM` only, produces:

```cpp
enum G1CandidateScoreOwner : uint32_t
{
    G1CandidateScoreUnassigned = 0U,
    G1CandidateScoreLegacy,
    G1CandidateScoreStrictRecovery,
    G1CandidateScoreIncumbent,
};

enum G1CandidateDisposition : uint32_t
{
    G1CandidateDispositionNotRun = 0U,
    G1CandidateDispositionAccepted,
    G1CandidateDispositionFiniteRejected,
    G1CandidateDispositionGlobalError,
};

struct G1CandidateAttemptTraceRecord
{
    G1CandidateRecord candidate;
    G1CandidateScoreOwner score_owner = G1CandidateScoreUnassigned;
    G1CandidateDisposition common = G1CandidateDispositionNotRun;
    G1CandidateDisposition raw = G1CandidateDispositionNotRun;
    G1CandidateDisposition ik = G1CandidateDispositionNotRun;
    G1FrameRejectionStage rejection_stage = G1FrameRejectNone;
    G1IkStopReason stop_reason = G1IkStopNone;
};

struct G1CandidateCertificationTrace
{
    G1CandidateAttemptTraceRecord attempts[G1CandidateAttemptCapacity] = {};
    uint32_t attempt_count = 0U;
    uint32_t legacy_traversals = 0U;
    uint32_t recovery_provider_calls = 0U;
    uint32_t common_evaluations = 0U;
    uint32_t raw_evaluations = 0U;
    uint32_t ik_evaluations = 0U;
    bool recovery_request_available = false;
    G1RecoveryRequest recovery_request;
    G1RecoveryCandidateSet recovery_set;
};
```

`G1FrameTransactionTestSeam` gains only `G1CandidateCertificationTrace* certification_trace = nullptr;`. The trace is reset at transaction entry and appended write-only by the coordinator.

- [ ] **Step 1: Write RED tests for attempt order, capacity, isolation, and mode-independent hidden ownership**

Extend the existing generic transaction fixture so reset allocates all five controller-state owners (`accepted`, `working`, `common`, `raw`, and `ik`) with pairwise-disjoint arrays. Add a deterministic stub stage runner and provider with these records:

```cpp
static constexpr G1CandidateRecord CandidateA = {
    G1CandidateLegacy, 64, 65, 1, 1.25f, UINT32_MAX, true};
static constexpr G1CandidateRecord CandidateB = {
    G1CandidateRecoveryTransition, 96, 97, 2, 1.50f, 0U, true};
static constexpr G1CandidateRecord CandidateC = {
    G1CandidateRecoveryTransition, 128, 129, 3, 1.75f, 1U, true};
static constexpr G1CandidateRecord Incumbent = {
    G1CandidateIncumbent, 32, 33, 0, 2.0f, UINT32_MAX, false};
```

The stub matcher publishes A as slot zero and a valid immutable request. The provider publishes B, C, then Incumbent and increments a test counter. Candidate A's common/raw phases pass and IK phase finite-rejects; B passes all phases; C and Incumbent are configured to fail the test if entered.

Add these exact test functions and call every one from the default full-test `main` path:

```cpp
static void test_abc_selects_b_in_both_modes_and_stops_before_c();
static void test_raw_rejection_short_circuits_ik_and_continues();
static void test_raw_pass_ik_reject_cannot_be_selected();
static void test_attempt_order_and_incumbent_deduplication();
static void test_rejected_attempt_storage_cannot_seed_next_attempt();
static void test_contact_phase_is_not_a_rank_or_admission_gate();
static void test_hidden_ik_commits_in_both_public_modes();
static void test_reset_switch_and_multiframe_hidden_state_are_paired();
static void test_exact_dt_intent_and_heading_are_immutable_per_attempt();
static void test_prior_safe_stop_latch_is_shared_and_consumed_once_on_winner();
static void test_prefix_camera_route_and_command_work_runs_once();
static void test_lazy_provider_and_attempt_work_bounds();
static void test_hypothetical_ninth_attempt_is_global_error();
static void test_scheduled_legacy_slot_zero_has_legacy_score_owner();
static void test_unscheduled_slot_zero_has_incumbent_score_owner();
static void test_matching_disabled_slot_zero_has_incumbent_score_owner();
```

In `tests/cpp/test_scene_switch.cpp`, add these exact lifecycle tests and call all six from `main`:

```cpp
static void test_five_state_reset_is_valid_equal_and_pairwise_disjoint();
static void test_five_state_live_preflight_rejects_each_array_alias();
static void test_five_state_candidate_isolation_covers_all_ten_states();
static void test_five_state_success_swaps_every_state_atomically();
static void test_five_state_switch_failure_preserves_every_live_owner();
static void test_five_state_reset_failure_preserves_every_live_owner();
```

Extend the existing test `state_storage_snapshot` into `runtime_storage_snapshot` with `digest[5]` and `data[5][49]`/`bytes[5][49]`, indexed exactly as accepted, working, common, raw, IK. `capture_active()` snapshots all five logical digests, all 245 heap ranges, publication, accepted diagnostic, scene/model/index, and counters; `active_matches_snapshot()` compares every one. Before every failure case, poison each state with a distinct valid `scene_frame`, `route_frames`, simulation X/Z, and hidden-IK release-frame value so accidentally resetting, partially swapping, or cross-copying any workspace is observable.

On successful reset/switch, require all five states independently pass `check_reset_state_member`, all ten within-runtime address pairs differ, all ten within-runtime heap-set pairs are disjoint, all five logical states are equal, and every route/IK/publication owner is canonical. For candidate isolation, hold five live plus five candidate states simultaneously and require all 45 pairwise heap-set comparisons disjoint; then alias one range at a time for each of the 25 live/candidate state pairs and require precommit rejection. On scene-load, reset, candidate-validation, unallocated-model, allocated-model, and malformed-config failures, require the complete five-state snapshot and storage addresses unchanged. On success, require all five candidate states installed before the old model unload callback, and require every old state storage set moved into the temporary old runtime rather than leaked or retained by the active runtime.

For A/B/C, run once with `external.tuning.ik_enabled = false` and once with `true`. Require identical query, selected/executed frame, range, source, strict score bits, transition flag, support, simulation XZ, route, requested/applied velocity, desired/predicted heading, and complete committed `state.ik`. Require raw visible arrays and canonical `ik_frame` in the off run; require IK arrays/result in the on run. Require A's score owner to be `G1CandidateScoreLegacy`, B's to be `G1CandidateScoreStrictRecovery`, trace dispositions `A=(accepted,accepted,finite-rejected)`, `B=(accepted,accepted,accepted)`, exactly two attempts, one provider call, two common/raw evaluations, two IK evaluations, and no C materialization.

The three slot-zero ownership tests must independently exercise (a) a scheduled legacy search, (b) a frame with no scheduled search, and (c) matching disabled. Transaction-entry trace reset leaves `score_owner == G1CandidateScoreUnassigned`; the coordinator must overwrite it before the first attempt. Require legacy/legacy in case (a), incumbent/incumbent in cases (b) and (c), and fail if any attempted trace record retains the unassigned poison. The default therefore cannot make an omitted incumbent assignment pass.

For raw rejection, reject A at `G1FrameStageRawPoseCertificate` and assert its IK stage counters remain zero while B executes. Require `g1_frame_candidate_dual_outcome_is_valid(G1CandidateDispositionAccepted, G1CandidateDispositionFiniteRejected, G1CandidateDispositionAccepted) == false`.

For poison isolation, write distinct noncanonical values to every scalar, array tail, support/contact owner, transition offset, route field, IK lock/history, certificate, and diagnostic during rejected A. Compare the complete B result with a direct B-only run from the same baseline using existing logical digests and per-array storage identities.

For an incoming safe-stop latch, require all attempted records to observe the same `prior_safe_stop_latched`, frozen applied intent, force-search bit, and route cursor. Rejecting A cannot consume or clear it; the selected B finalize consumes it once through the existing one-frame frozen-route rule. Count input command/route sampling, camera update, accepted finalize, presentation, and scene/route lifecycle: prefix work runs once, private attempts run none of it, success publishes once, and finite exhaustion performs only the existing one rejection publication.

For work bounds, cover accepted/global-error slot zero, finite scheduled slot zero, matching disabled, no scheduled search, provider global error, success before the tail, all seven tail records, and a hostile provider result with `count == 8`. Assert one legacy traversal only on scheduled search, provider calls `0/1` as specified, at most eight attempts, one common/raw/IK per eligible attempt, no IK after raw fail, and no stage after success.

Run RED against Task 1:

```bash
set -o pipefail
out=/tmp/g1-bounded-candidate-cert/task2
mkdir -p "$out"
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
     -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM -I. \
     -c tests/cpp/test_g1_frame_transaction.cpp \
     -o "$out/frame-red.o" 2>"$out/frame-red.stderr"; then
  echo 'ERROR: coordinator RED unexpectedly compiled' >&2
  exit 1
fi
rg 'G1FrameStageRawBegin|G1RecoveryProvider|G1FrameCandidateWorkspace|no member' \
  "$out/frame-red.stderr"
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
     -c tests/cpp/test_scene_switch.cpp \
     -o "$out/scene-switch-red.o" 2>"$out/scene-switch-red.stderr"; then
  echo 'ERROR: five-state scene-switch RED unexpectedly compiled' >&2
  exit 1
fi
rg 'candidates|common_state|raw_state|ik_state|no member' \
  "$out/scene-switch-red.stderr"
```

Expected: both compilations fail on the missing expanded coordinator/workspace interface or five-state scene lifecycle, not an unrelated warning.

- [ ] **Step 2: Add fixed workspace allocation, disjointness, and preflight ownership**

Include `g1_candidate_recovery.h` from `g1_frame_transaction.h`. Add the exact workspace and scratch types above. Generalize the current pair-storage preflight to collect ranges for all five states and require:

```cpp
const g1_controller_state* states[5] = {
    &runtime.accepted_state,
    &runtime.working_state,
    &runtime.candidates.common_state,
    &runtime.candidates.raw_state,
    &runtime.candidates.ik_state,
};
for (uint32_t i = 0U; i < 5U; ++i) {
    for (uint32_t j = i + 1U; j < 5U; ++j) {
        if (states[i] == states[j]) return false;
    }
}
```

Implement all ten pairwise state-object address comparisons; do not express this as chained C++ inequality. Because the five states are members of `G1FrameRuntime`, their object extents necessarily lie inside the enclosing runtime and must not be tested for disjointness from their parent. Collect only each state's heap-backed array storage ranges; require those ranges pairwise disjoint across all five states and disjoint from the enclosing runtime's fixed-object extent, every non-array fixed member, `G1FrameExternalInputs`, error storage, database/support/scene sources, and the optional seam/trace object. Do not allocate during a frame.

In `g1_frame_runtime_reset`, call `g1_controller_state_reset_configured` exactly five times into a temporary candidate runtime, assign the same mode-independent reset configuration to all five, verify all five logical states equal, then swap each state into output. Scene switch continues to call this reset path, so paired modes initialize identical hidden IK and workspace shapes.

Update `scene_switch.h` in the same task. `scene_frame_runtime_live_storage_preflight` must collect five state range sets and validate each against sources/error/fixed objects. `scene_frame_runtime_reset_candidate_is_valid` must independently validate accepted, working, common, raw, and IK and require all five logically equal. `scene_frame_runtime_candidate_is_isolated` must collect ten state range sets (five live, five candidate), require all ten state addresses distinct, all 45 heap-set pairs disjoint, and every heap set disjoint from both runtime fixed extents, config, extras, error, database, support, and both scenes. `scene_frame_runtime_swap` must call `g1_controller_state_swap` for accepted, working, `candidates.common_state`, `candidates.raw_state`, and `candidates.ik_state` before swapping publication/diagnostic. `scene_reset_current` and `scene_switch_transaction` keep their current precommit/failure order; no live owner changes until the complete five-state candidate and model are valid.

- [ ] **Step 3: Implement the bounded phase runner and fresh-copy protocol**

Add these exact internal helpers:

```cpp
enum G1CandidateEvaluationOutcome : uint32_t
{
    G1CandidateDualAccepted = 0U,
    G1CandidateFiniteRejected,
    G1CandidateGlobalError,
};

static inline G1FrameStageOutcome g1_frame_run_one_stage(
    G1FrameTransactionStage stage,
    G1FrameStageRunner run_stage,
    g1_controller_state& state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity);

static inline G1CandidateEvaluationOutcome g1_frame_candidate_evaluate(
    G1FrameCandidateWorkspace& workspace,
    G1FrameTransactionScratch& candidate_scratch,
    const g1_controller_state& immutable_baseline,
    const G1FrameTransactionScratch& prefix_scratch,
    const G1CandidateRecord& record,
    G1FrameStageRunner run_stage,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    G1CandidateAttemptTraceRecord* trace_record,
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity);
```

`g1_frame_candidate_evaluate` performs exactly this sequence. The immutable prefix scratch carries the one incoming safe-stop snapshot, route/command/heading snapshot, and camera result into every fresh attempt; no candidate may resample or consume them:

```text
deep-copy immutable_baseline -> common_state
value-copy prefix_scratch -> fresh candidate_scratch
bind active_candidate
CandidateApply through FootprintObservation on common_state
deep-copy common_state -> raw_state
RawBegin through RawPoseCertificate on raw_state
if raw accepted: deep-copy common_state -> ik_state
IkBegin through IkPoseCertificate on ik_state
return DualAccepted only after both branch ranges accept
```

Every deep copy uses `g1_controller_state_copy` and therefore overwrites all array and scalar owners. A finite stage stops only that record; a global stage stops the outer frame. Invoke the existing test hook immediately after each continued real stage, preserving the current real-outcome-before-hook rule.

Add `g1_frame_candidate_dual_outcome_is_valid(common, raw, ik)` and require common/raw/IK all `G1CandidateDispositionAccepted`; raw finite plus IK accepted is invalid even for a forged seam result. Keep `G1CandidateDualAccepted`, `G1CandidateFiniteRejected`, and `G1CandidateGlobalError` exclusively in `G1CandidateEvaluationOutcome`; no disposition enumerator reuses those identifiers.

- [ ] **Step 4: Implement lazy tail construction, first-success commit, and hidden IK projection**

Run `InputRouteCommand` and `MatcherSearch` once on `runtime.working_state`; this becomes the immutable candidate baseline. Before evaluating slot zero, explicitly assign both candidate and score owner: a scheduled legacy result uses `G1CandidateLegacy` plus `G1CandidateScoreLegacy`; no scheduled search and matching-disabled frames use `G1CandidateIncumbent` plus `G1CandidateScoreIncumbent`. Never derive this owner from the record's default, and reject `G1CandidateScoreUnassigned` before evaluation. Attempt `prefix_scratch.slot_zero_record` first. Only after its finite scheduled-search rejection call `recovery_provider` once, validate count/order/kinds/ranks/dedup/cost/provenance/work, and iterate the returned tail. Assign every recovery transition `G1CandidateScoreStrictRecovery` and a tail incumbent `G1CandidateScoreIncumbent` before its attempt.

On dual acceptance, perform:

```cpp
g1_controller_state& visible = external.tuning.ik_enabled
    ? runtime.candidates.ik_state
    : runtime.candidates.raw_state;
if (!g1_controller_state_copy(
        runtime.working_state, visible, error, error_capacity)) {
    return G1FrameTransactionGlobalError;
}
runtime.working_state.ik = runtime.candidates.ik_state.ik;
```

Then run `G1FrameStageAcceptedFinalize` exactly once on `runtime.working_state`, validate accepted diagnostic/provenance plus hidden/public ownership, swap accepted/working, and publish once. The raw state already carries canonical disabled `ik_frame`, raw local/global pose in both IK pose sets, and raw clearance in both clearance owners; assigning only hidden `.ik` cannot leak hidden pose/result fields. The IK state carries the existing enabled products.

On a finite attempt, value-copy the whole scratch into `first_failure_scratch` only when no earlier failure is saved. Task 4 strengthens its publication authentication; Task 2 must already delay all public writes until success or exhaustion.

- [ ] **Step 5: Run generic coordinator GREEN and strict/fast/sanitizer matrices**

Build Task 1's strict provider once and link the generic test with its stub provider (the real provider object remains linked to authenticate the function type):

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task2
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp -o "$out/recovery.o"
for mode in strict fast; do
  flags=("${strict[@]}")
  test "$mode" = fast && flags=("${fast[@]}")
  for name in g1_frame_transaction scene_switch; do
    g++ "${flags[@]}" \
      -c "tests/cpp/test_${name}.cpp" -o "$out/${name}-${mode}.o"
    g++ "$out/${name}-${mode}.o" "$out/clearance.o" \
      "$out/root-reach.o" "$out/recovery.o" \
      -o "$out/${name}-${mode}"
    "$out/${name}-${mode}"
  done
done

san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all)
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp -o "$out/clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_ik_root_reach.cpp -o "$out/root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_candidate_recovery.cpp -o "$out/recovery-san.o"
for name in g1_frame_transaction scene_switch; do
  g++ "${san[@]}" -I. -c "tests/cpp/test_${name}.cpp" \
    -o "$out/${name}-san.o"
  g++ "${san[@]}" "$out/${name}-san.o" "$out/clearance-san.o" \
    "$out/root-reach-san.o" "$out/recovery-san.o" \
    -o "$out/${name}-san"
  ASAN_OPTIONS=detect_leaks=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$out/${name}-san"
done
```

Expected: all coordinator and five-state scene lifecycle fixtures pass under strict and fast callers; sanitizers are silent; all five reset/switch owners are valid and isolated; every failure preserves the complete live five-state runtime; and test traces show no recovery/evaluation after success or global error.

- [ ] **Step 6: Commit the coordinator and hidden-state owner**

```bash
git add g1_frame_transaction.h scene_switch.h \
  tests/cpp/test_g1_frame_transaction.cpp tests/cpp/test_scene_switch.cpp
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: coordinate bounded dual candidate certification"
```

Expected staged paths: exactly the four modified Task 2 files. The unchanged compile-negative runner remains verification input for Tasks 3 and 5 and is not staged. The production controller is integrated in Task 3; this commit's acceptance gate is the generic coordinator plus scene-switch suites, which do not compile `controller.cpp`.

---

### Task 3: Integrate the Real Controller Runner and Preserve Legacy Slot Zero

**Files:**
- Modify: `controller.cpp:3540-4350,5240-5360`
- Modify: `tests/cpp/test_g1_frame_transaction_production.cpp:1-2920`
- Verify only: `database.h`
- Verify only: `g1_ik.h`
- Verify only: `g1_ik_root_reach.cpp`
- Verify only: `tests/cpp/test_g1_ik.cpp`
- Consume: Tasks 1-2

**Interfaces and ownership:**
- `G1FrameStageMatcherSearch` is preparation only. It constructs the exact 31-word raw query, terrain query, query frame/range, matching/search decision, incumbent/public costs, transition-cost word, and one legacy `G1CandidateRecord`. It calls the one existing `::database_search` only when the existing scheduled-search predicate is true. It does not transition a pose, advance the database frame, decrement either search timer, or read `ik_enabled`.
- `G1FrameStageCandidateApply` is the sole per-record matcher mutation. For a transition record it copies the selected database pose into the existing `trns_*` arrays, calls `g1_runner_pose_transition`, sets `frame_index = selected_frame`, and sets `transitioned = true`. For the incumbent it leaves the selected pose/current transition provenance unchanged and sets `transitioned = false`. It then projects the existing timer tail from the same post-matcher baseline, advances to the record's authenticated `executed_frame`, and fills `curr_*` pose/contact arrays. Each private projection performs the same one-frame timer arithmetic, rejected copies are discarded, and the selected outer state therefore contains exactly one timer decrement.
- A legacy record retains the exact legacy `selected_cost`, `incumbent_cost`, `selected_terrain_error`, selected frame, transition bit, and source range. A recovery-transition record replaces only existing `selected_cost` with the record's strict score and recomputes existing selected terrain error at that frame. An incumbent record preserves the existing public sentinel semantics described in Task 1.
- `G1FrameStageFootprintObservation` ends after the existing footprint certificate. It no longer begins IK.
- Raw stages call the existing IK transaction APIs with the literal branch flag `false`; IK stages call them with the literal branch flag `true`. Both use exact `external.tuning.dt`, the same footprint, candidate contacts, terrain, parents, and immutable baseline `state.ik`. The branch flag never comes from `external.tuning.ik_enabled`.
- `G1FrameStageAcceptedFinalize` is the only candidate stage that builds the accepted diagnostic and increments scene/route lifecycle. Candidate application and private certificate stages never publish or advance presentation/route lifecycle.
- `main` calls `g1_frame_transaction_run(frame_runtime, ::g1_controller_frame_stage_run, ::g1_recovery_candidates_build, external, test_seam_pointer, error, sizeof(error))` in seam builds and the identical call without `test_seam_pointer` in production. No wrapper reimplements, pre-normalizes, or rescans recovery.

- [ ] **Step 1: Add RED production-runner tests for real A/B/C behavior and a genuine swing rejection**

Extend the existing 160-frame, one-range production fixture rather than replacing its inherited cases. Reserve incumbent selected/executed frames `32/33`, A `64/65`, B `96/97`, and C `128/129`; all remain more than 20 frames apart and more than 20 frames from the range end. Use a normalized positive-zero query, transition cost `1.0f`, A's row distance `0.0f`, B's first enabled feature difference `0.5f`, C's first enabled feature difference `0.75f`, and incumbent's first enabled feature difference `2.0f`. Rebuild the real small/large bounds. This makes A the genuine legacy result, admits B then C below the private/public incumbent, and keeps the record order independent from contacts.

At A's executed frame, create a valid raw/no-IK candidate but a real IK swing-history boundary: left contact releases, right contact remains planted, and the incoming authenticated left-foot swing history contains no admissible sample. Require the unchanged IK code to produce `G1IkStopNoSwingCandidate`; do not inject a runner return code. At B, use a valid immediate contact release/landing arrangement that passes both branches. Configure C to fail the fixture if any stage executes. Add these exact tests:

```cpp
static void test_real_abc_selects_b_in_both_modes();
static void test_real_a_raw_passes_and_ik_reports_no_swing_candidate();
static void test_real_b_contact_release_is_not_an_eligibility_gate();
static void test_real_c_never_executes_after_b_accepts();
static void test_real_incumbent_runs_last_with_clamped_plus_one();
static void test_real_legacy_incumbent_is_not_retried();
static void test_real_provider_global_error_rolls_back_every_owner();
```

For both `ik_enabled` values, require A then B, selected frame `96`, executed frame `97`, source range `0`, identical query/cost/provenance/common/hidden state, and one outer lifecycle increment. Require A raw accepted and IK finite-rejected with `no-swing-candidate`; B common/raw/IK accepted; C untouched. Require B's public `selected_cost` bits to equal its strict record and be numerically below the unchanged public incumbent. Require raw public pose/clearance and a canonical `ik_frame` off, IK public pose/result/clearance on, and identical successful `state.ik` in both.

For the incumbent fallback, make A/B/C finite-reject in increasing rank and require selected provenance to remain the incumbent's current frame, execution to be exactly ordinary clamped `+1`, and one incumbent attempt. Repeat at end-of-animation and require public `incumbent_cost` and `selected_cost` to retain exact `FLT_MAX` words.

Run RED against the intentionally stale post-Task-2 controller. At this boundary, `g1_frame_transaction.h` already has the new stage enum and provider parameter while `controller.cpp` still names the old stages and old call signature, so controller compilation itself is the expected RED:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task3-red
rm -rf "$out"
mkdir -p "$out"
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
warn=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)

g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp -o "$out/recovery.o"
if g++ "${warn[@]}" "${rayinc[@]}" \
     -DG1_CONTROLLER_NO_MAIN -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
     -c controller.cpp -o "$out/controller-runner.o" \
     2>"$out/controller.stderr"; then
  echo 'ERROR: stale post-Task-2 controller unexpectedly compiled' >&2
  exit 1
fi
rg 'G1FrameStage(FirstFootIk|SecondFootIk|FinalFk|PoseCertificate)|g1_frame_transaction_run|too few arguments|not declared' \
  "$out/controller.stderr"
```

Expected: all three strict dependencies compile; only the stale controller fails, on removed stage names or the added recovery-provider argument. After Steps 2-3 restore the controller build, the new production A/B/C assertions remain RED until the real runner behavior is complete.

- [ ] **Step 2: Split matcher preparation from candidate application without changing slot zero**

Move no query expression and no legacy search expression into the strict provider. In `G1FrameStageMatcherSearch`:

1. Call `g1_runner_build_query` once and snapshot all 31 raw words before any legacy normalization.
2. Compute query frame/range, `next`, end-of-animation, the existing scheduled predicate, public incumbent cost, selected terrain error, and the exact idle transition-cost word exactly where they are currently computed.
3. Call the existing `::database_search` exactly once only on a scheduled search. Form `scratch.slot_zero_record` from its returned selected frame/cost and the existing active-range helper; use the exact clamped `+1` frame for `executed_frame`.
4. When scheduled, fill `scratch.recovery_request` from the same `external.db`, raw-query snapshot, pre-transition incumbent frame, returned legacy selected frame, transition cost, public incumbent cost, and literal unchanged exclusions `20/20`; set `recovery_request_ready = true`. Do not call the provider here.
5. When matching is disabled or search is not scheduled, form a single incumbent slot zero and leave `recovery_request_ready = false`.

Implement `G1FrameStageCandidateApply` as the exact former transition/advance tail. Validate candidate kind, frames, source range, cost owner, recovery rank, and clamped execution before mutation. For every candidate copy, decrement `search_timer` once from the identical baseline; when the outer scheduled search reset both timers, also decrement `force_search_timer` once exactly as today. Rejected projections disappear, so the selected committed state contains one—not rank-many—timer decrement. Never increment scene, route, presentation, or trace counters here.

Add a source guard that parses the production body and requires:

```text
database_search calls in controller.cpp production code: 1
database_search calls inside G1FrameStageMatcherSearch: 1
database_search calls inside all other stage cases: 0
g1_recovery_candidates_build calls inside stage runner: 0
```

The guard must ignore comments/string literals but not preprocessor-visible production code.

- [ ] **Step 3: Implement common, raw, IK, and finalize stages using unchanged physical APIs**

Keep inertialization through footprint observation behavior and thresholds byte-for-byte, but make each case operate on the candidate-local state selected by the coordinator. Split the old combined IK path into two structurally parallel blocks:

```cpp
G1FrameStageOutcome g1_runner_certificate_begin(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    G1FrameCertificateBranch branch_kind,
    bool enabled,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);

G1FrameStageOutcome g1_runner_certificate_foot(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    G1FrameCertificateBranch branch_kind,
    int foot,
    bool enabled,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);

G1FrameStageOutcome g1_runner_certificate_finish(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    bool enabled,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);

G1FrameStageOutcome g1_runner_pose_certificate(
    g1_controller_state& state,
    G1FrameBranchCertificateScratch& branch,
    G1FrameCertificateBranch branch_kind,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);
```

The raw cases pass `false`; the IK cases pass `true`. `finish` uses the existing `g1_ik_frame_finish`, checked FK, final-pose copy, and candidate-state/result authentication. Pose certification uses the unchanged clearance budget and thresholds. Record branch identity before creating a rejection so Task 4 can authenticate the correct producer. A finite common/raw/IK failure returns immediately; an unexpected status or any invalid result is global.

In `G1FrameStageAcceptedFinalize`, attach the selected footprint and mode-visible clearance/pose owners, construct the existing diagnostic from the selected scratch, and apply the exact scene/route increments once. Do not place this logic in either branch certificate.

- [ ] **Step 4: Add baseline identity, public-cost, closure, and no-seam regressions**

Add these exact production tests:

```cpp
static void test_legacy_slot_zero_acceptance_has_exact_pre_feature_public_owners();
static void test_legacy_slot_zero_acceptance_never_materializes_recovery();
static void test_matching_disabled_and_unscheduled_frames_attempt_only_incumbent();
static void test_end_of_animation_public_sentinels_do_not_leak_private_score();
static void test_stage_runner_trusted_call_closure_is_exact();
static void test_controller_contains_one_production_database_search_call();
static void test_controller_has_no_candidate_contact_rank_gate();
static void test_controller_has_no_persistent_dual_certificate_owner();
static void test_production_no_seam_links_with_strict_recovery_provider();
```

For a normal accepted slot-zero synthetic frame, compare every legacy matcher/common public field and exact cost/provenance bit with the Task 1 fixture before and after integration, under strict and fast callers. Require legacy/recovery counters `(1,0)`, one attempt, and no recovery set materialization. Separately compare rows `0..5` of the authentic Task 1 low-curb baseline with a new disposable IK-off build for query, selected/database/source frames, transition, costs, terrain, support, simulation, route, travel, and heading fields; do not compare hidden `state.ik` to the old IK-off state digest.

Update inherited stage-name, sole-runner declaration, trusted no-main call closure, candidate-audit post-publication, safe-stop, dirty-copy rollback, and negative-context guards to the expanded signature. Explicitly reject a provider pointer derived from opaque void, publication, or mutable accepted-state context. No guard may be deleted merely because its old stage count changed.

- [ ] **Step 5: Run the real integration matrix under strict, fast, sanitizer, and production closure**

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task3
rm -rf "$out"
mkdir -p "$out"
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
warn=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)

g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp -o "$out/recovery.o"
for flavor in strict fast; do
  flags=("${warn[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" "${rayinc[@]}" \
    -DG1_CONTROLLER_NO_MAIN -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$out/controller-runner-${flavor}.o"
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$out/production-${flavor}.o"
  g++ "$out/production-${flavor}.o" \
    "$out/controller-runner-${flavor}.o" "$out/clearance.o" \
    "$out/root-reach.o" "$out/recovery.o" "${raylib[@]}" \
    -o "$out/production-${flavor}"
  "$out/production-${flavor}"

done

g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_clearance.cpp -o "$out/clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_ik_root_reach.cpp -o "$out/root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_candidate_recovery.cpp -o "$out/recovery-san.o"
g++ "${san[@]}" "${rayinc[@]}" \
  -DG1_CONTROLLER_NO_MAIN -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$out/controller-runner-san.o"
g++ "${san[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o "$out/production-san.o"
g++ "${san[@]}" "$out/production-san.o" "$out/controller-runner-san.o" \
  "$out/clearance-san.o" "$out/root-reach-san.o" "$out/recovery-san.o" \
  "${raylib[@]}" -o "$out/production-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$out/production-san"

g++ "${fast[@]}" "${rayinc[@]}" -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -c controller.cpp -o "$out/controller-production.o"
g++ "$out/controller-production.o" "$out/clearance.o" \
  "$out/root-reach.o" "$out/recovery.o" "${raylib[@]}" \
  -o "$out/controller-production"
DISPLAY=:1 MM_IK=0 "$out/controller-production" \
  --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
  --terrain-scene grail-curb-low --test-mode route \
  --test-route curb-forward --test-frames 6 --test-heading forward \
  --terrain-weight 4 --log "$out/slot-zero-six.csv"
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv
from resources.check_g1_runtime_log import GATE_E_PAIR_INVARIANT_GROUPS
base = "/tmp/g1-bounded-candidate-cert/base/logs/low-curb-0.csv"
new = "/tmp/g1-bounded-candidate-cert/task3/slot-zero-six.csv"
with open(base, newline="", encoding="utf-8") as stream:
    before = list(csv.DictReader(stream))[:6]
with open(new, newline="", encoding="utf-8") as stream:
    after = list(csv.DictReader(stream))
assert len(before) == len(after) == 6
for index, (old, current) in enumerate(zip(before, after)):
    for group, names in GATE_E_PAIR_INVARIANT_GROUPS:
        for name in names:
            assert old[name] == current[name], (index, group, name)
print("VALID six-frame legacy slot-zero public baseline")
PY
test "$(rg -o '::database_search[[:space:]]*\\(' controller.cpp | wc -l)" -eq 1
! nm -C "$out/controller-production" | \
  rg 'CandidateCertificationTrace|exhaustive_for_test|ProviderAudit|test_seam'
for mode in VOID_CONTEXT PUBLICATION_CONTEXT ACCEPTED_CONTEXT; do
  if g++ "${warn[@]}" -D"G1_FRAME_NEGATIVE_${mode}" \
       -c tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
       -o "$out/runner-negative-${mode}.o" \
       2>"$out/runner-negative-${mode}.stderr"; then
    echo "ERROR: forbidden ${mode} runner context compiled" >&2
    exit 1
  fi
  rg 'G1FrameStageRunner|convert|conversion|argument' \
    "$out/runner-negative-${mode}.stderr"
done

sha256sum -c /tmp/g1-bounded-candidate-cert/base/immutable-files.before.sha256
git diff --check
```

Expected: the controller build is restored; A/B/C passes in both modes; all strict/fast public transcripts match; the real A rejection is `no-swing-candidate`; sanitizers are silent; production has exactly one legacy search and no seam symbol; negative contexts fail at compile time; immutable schema/checker hashes remain `OK`.

- [ ] **Step 6: Commit only the real controller integration**

```bash
git add controller.cpp tests/cpp/test_g1_frame_transaction_production.cpp
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: integrate bounded candidate certification"
```

Expected staged paths: exactly the two Task 3 files. Do not stage any root-frontier file, artifact, schema, checker, or generated fixture.

---

### Task 4: Authenticate Full First-Failure Publication and Canonical IK-Off Logging

**Files:**
- Modify: `g1_frame_transaction.h:1379-2450,2678-2914`
- Modify: `controller.cpp:1347-1945`
- Modify: `tests/cpp/test_g1_frame_transaction.cpp`
- Modify: `tests/cpp/test_g1_controller_logging.cpp:1-497`
- Verify only: `motion_match_log.h`
- Verify only: `resources/check_g1_runtime_log.py`
- Consume: Tasks 1-3

**Interfaces:**
- `g1_frame_publish_finite_rejection(runtime, first_failure_scratch, external, true)` remains the sole finite publisher and is called only once, after candidate exhaustion.
- `g1_frame_rejection_matches_scratch_transaction` selects `raw_certificate.ik_transaction` or `ik_certificate.ik_transaction` from `scratch.rejection_branch` and authenticates the saved footprint, checkpoint, IK result, and pose certificate against that same saved scratch.
- Adds one logger that has no state parameter:

```cpp
static void g1_log_canonical_disabled_ik(
    motion_match_ik_diagnostic& output)
{
    output = motion_match_ik_diagnostic{};
}
```

- Changes the suffix builder to receive the immutable process mode explicitly:

```cpp
static bool g1_build_task7_log_suffix(
    motion_match_log_row& log_row,
    const g1_controller_state& accepted_state,
    const G1FrameAcceptedDiagnostic& accepted_diagnostic,
    const G1FramePublication& publication,
    bool ik_enabled,
    char* error,
    int error_capacity);
```

- [ ] **Step 1: Write RED tests for complete first-failure evidence and later-attempt noninterference**

Add these exact generic tests:

```cpp
static void test_exhaustion_publishes_complete_first_failing_scratch_once();
static void test_later_failure_cannot_replace_first_branch_or_reason();
static void test_saved_footprint_mutation_is_global_error();
static void test_saved_ik_transaction_mutation_is_global_error();
static void test_diagnostic_only_failure_copy_cannot_publish();
static void test_global_error_after_finite_attempt_publishes_no_finite_evidence();
static void test_success_after_private_failures_publishes_no_rejection();
```

Give every assertion introduced by these seven tests a message beginning with the exact prefix `first-failure-red:`. Add a `run_first_failure_tests()` helper that calls exactly these seven functions. Change the test entry point to `int main(int argc, char** argv)` with this contract:

```cpp
if (argc == 2 && std::strcmp(argv[1], "--first-failure-red") == 0) {
    std::fputs("G1_FIRST_FAILURE_RED_SELECTED\n", stderr);
    run_first_failure_tests();
    return 0;
}
if (argc != 1) {
    return 2;
}
run_all_inherited_and_task2_tests();
run_first_failure_tests();
return 0;
```

`run_all_inherited_and_task2_tests()` denotes the existing default-main call sequence after Task 2, factored without changing its order. The selector branch calls none of those inherited tests. The normal no-argument path still runs every inherited, Task 2, and new Task 4 test.

Use three finite records with distinct requested intent bits, footprint digests, IK checkpoints, pose work, rejection stages, and stop reasons. Require the published result to equal the first record's complete value snapshot, not its diagnostic alone. On exhaustion assert:

```cpp
status == G1FrameTransactionFiniteRejected
accepted digest unchanged
accepted storage identities unchanged
accepted route/common/pose/state.ik unchanged
publication.presentation_frame == external.input.presentation_frame
publication.ik_safe_stop_latched == true
one finite publisher call
one presentation row
no accepted candidate provenance
```

Copy the first scratch, flip one saved footprint bit, and require authenticated publication failure. Restore it, flip the saved IK transaction's `next_foot`, staged provenance, candidate result, and candidate state independently, and require failure for each. Construct a scratch containing only the correct `G1FrameRejectionDiagnostic` with canonical/default producer fields and require failure. After one finite A, inject a global error in B; require the incoming publication, accepted diagnostic, accepted state, hidden IK, and latch to remain exactly as before the outer call.

Prove the transaction behavior itself is RED against the completed Task 3 implementation by running only the dedicated new-test selector:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task4/transaction-red
rm -rf "$out"
mkdir -p "$out"
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp -o "$out/recovery.o"
if ! g++ "${strict[@]}" -c tests/cpp/test_g1_frame_transaction.cpp \
     -o "$out/frame.o" 2>"$out/frame.compile.stderr"; then
  echo 'ERROR: first-failure RED test source did not compile' >&2
  exit 1
fi
if ! g++ "$out/frame.o" "$out/clearance.o" "$out/root-reach.o" \
     "$out/recovery.o" -o "$out/frame" \
     2>"$out/frame.link.stderr"; then
  echo 'ERROR: first-failure RED test executable did not link' >&2
  exit 1
fi
if "$out/frame" --first-failure-red >"$out/frame.output" 2>&1; then
  echo 'ERROR: first-failure transaction RED unexpectedly passed' >&2
  exit 1
fi
test "$(rg -c '^G1_FIRST_FAILURE_RED_SELECTED$' \
  "$out/frame.output")" -eq 1
rg '^G1 frame transaction test failed: first-failure-red:' \
  "$out/frame.output"
```

Expected: every compilation and the link succeed; the selector sentinel appears exactly once; and the executable fails in an assertion bearing the unique new-test prefix. Because this selector invokes only the seven Task 4 tests, neither a compile failure nor an inherited-test failure can satisfy RED. The later no-argument GREEN and full-suite invocations still execute all tests.

- [ ] **Step 2: Write RED logging tests for a poisoned hidden IK owner**

In `tests/cpp/test_g1_controller_logging.cpp`, add:

```cpp
static void test_disabled_projection_is_exact_default_under_hidden_poison();
static void test_enabled_projection_still_uses_certified_ik_products();
static void test_disabled_projection_has_no_hidden_owner_source_path();
static void test_log_schema_and_runtime_checker_hashes_are_unchanged();
static void test_recovery_selected_cost_uses_strict_word_and_checker_rule();
static void test_end_of_animation_incumbent_keeps_public_sentinels();
```

Build two valid accepted states whose common/raw pose and public projection are bit-identical. In one, poison every hidden lock flag/point/velocity, release frame, swing-history point, baseline sole normal, root-reach field, swing lift/result, residual, IK clearance witness/work counter, and hidden IK local/global pose/result value. Call the suffix builder with `ik_enabled == false` and compare every logical member with a separately value-initialized `motion_match_ik_diagnostic{}`; also require byte-identical serialized CSV suffixes for both states. Do not compare struct padding with `memcmp`. Require `candidate_clearance_status == "invalid-input"`, both swing statuses `"invalid-input"`, both selected indices `UINT32_MAX`, stop reason `"none"`, and every numeric/flag field zero.

Tokenize `controller.cpp` using the existing production source parser and require `g1_log_canonical_disabled_ik` to have exactly one output reference, no `g1_controller_state` parameter, and no identifiers `ik`, `state`, `lock`, `swing`, `root_reach`, `clearance`, or `g1_log_accepted_ik`. Require the suffix builder to call the accepted-IK logger only in the `ik_enabled` branch and the canonical logger only in the opposite branch.

Record the pre-task hashes from Task 1 and require the current `motion_match_log.h` and `resources/check_g1_runtime_log.py` hashes to match them byte-for-byte.

Add `emit_bounded_fixtures(const char* directory)` to the logging test only. It must construct authentic accepted state/diagnostic/publication triples and call the production row builder/writer for exactly three one-row files: `legacy.csv`, `recovery.csv`, and `incumbent.csv`. The legacy row retains its legacy cost bits; the recovery row sets existing `selected_cost` to the admitted strict record word and a numerically larger non-sentinel public `incumbent_cost`; the end-of-animation incumbent row keeps both public words at exact `FLT_MAX`. Invoke it only for `argc == 3 && std::strcmp(argv[1], "--emit-bounded-fixtures") == 0`; reject every other non-default argument. No test writes CSV fields directly.

Run RED:

```bash
set -o pipefail
out=/tmp/g1-bounded-candidate-cert/task4
mkdir -p "$out"
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
     "${rayinc[@]}" \
     -c tests/cpp/test_g1_controller_logging.cpp \
     -o "$out/logging-red.o" 2>"$out/logging-red.stderr"; then
  echo 'ERROR: logging RED unexpectedly compiled' >&2
  exit 1
fi
rg 'g1_log_canonical_disabled_ik|g1_build_task7_log_suffix|no matching' \
  "$out/logging-red.stderr"
```

Expected: new logging tests fail because the canonical state-free projection and explicit mode parameter do not exist.

- [ ] **Step 3: Strengthen scratch authentication and exhaustion publication**

Update `g1_frame_rejection_matches_scratch_transaction` so footprint evidence always compares with `scratch.footprint`; IK evidence requires `rejection_branch == G1FrameCertificateIk` and derives from `scratch.ik_certificate.ik_transaction` at the exact begin/foot checkpoint; raw pose evidence requires `rejection_branch == G1FrameCertificateRaw` and exact raw pose status/clearance; IK pose evidence requires `rejection_branch == G1FrameCertificateIk` and exact IK pose status/clearance. Reject an unknown branch, a diagnostic without its producer, an unavailable checkpoint, or any mismatch.

Keep the first failure as a plain value object local to `g1_frame_transaction_run`:

```cpp
G1FrameTransactionScratch first_failure_scratch;
bool first_failure_available = false;
```

Assign it only under `if (!first_failure_available)`. If a later record succeeds, never call the finite publisher. If the provider/tail exhausts, call the publisher exactly once with `authenticate_scratch_transaction == true`; if authentication fails, return `G1FrameTransactionGlobalError` without changing accepted/public owners.

Update accepted-success validation to authenticate the hidden successful IK transaction independently from the public mode projection. In both modes, require `working_state.ik` equal to `scratch.ik_certificate.ik_transaction.candidate_state`. With IK off, require canonical `ik_frame`, raw pose equality across all eight public pose arrays, and raw clearance equality; with IK on, require public IK result/poses/clearance equal the successful IK branch.

- [ ] **Step 4: Implement the state-free disabled logger and preserve enabled semantics**

Add `g1_log_canonical_disabled_ik` exactly as declared. In `g1_build_task7_log_suffix`, build query bits first, then:

```cpp
if (ik_enabled) {
    if (!g1_log_accepted_ik(
            log_row.ik, accepted_state, error, error_capacity)) {
        return false;
    }
} else {
    g1_log_canonical_disabled_ik(log_row.ik);
}
```

Continue to call `g1_log_directional` from the common accepted state because footprint, rejection, command, support, and accepted digest are not part of the disabled IK suffix. Pass `log_context.ik_enabled` from main. Do not modify the row type, header writer, column order, checker, or thresholds.

- [ ] **Step 5: Run focused publication/logging GREEN in strict, fast, and sanitized callers**

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task4
mkdir -p "$out"
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp -o "$out/clearance.o"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_candidate_recovery.cpp -o "$out/recovery.o"
for mode in strict fast; do
  flags=(-O2 -Wall -Wextra -Werror -pedantic)
  test "$mode" = fast && flags=(-O3 -ffast-math -DNDEBUG)
  g++ -std=c++17 "${flags[@]}" -I. \
    -c tests/cpp/test_g1_frame_transaction.cpp \
    -o "$out/g1_frame_transaction-${mode}.o"
  g++ "$out/g1_frame_transaction-${mode}.o" "$out/clearance.o" \
    "$out/root-reach.o" "$out/recovery.o" \
    -o "$out/g1_frame_transaction-${mode}"
  "$out/g1_frame_transaction-${mode}"

  g++ -std=c++17 "${flags[@]}" -I. "${rayinc[@]}" \
    -c tests/cpp/test_g1_controller_logging.cpp \
    -o "$out/g1_controller_logging-${mode}.o"
  g++ "$out/g1_controller_logging-${mode}.o" "$out/clearance.o" \
    "$out/root-reach.o" "$out/recovery.o" "${raylib[@]}" \
    -o "$out/g1_controller_logging-${mode}"
  "$out/g1_controller_logging-${mode}"
done

mkdir -p "$out/fixtures"
"$out/g1_controller_logging-strict" \
  --emit-bounded-fixtures "$out/fixtures"
for csv in legacy recovery incumbent; do
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$out/fixtures/${csv}.csv"
done

sha256sum -c /tmp/g1-bounded-candidate-cert/base/immutable-files.before.sha256
git diff --check
```

Run the same two tests under sanitizers:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task4
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -I.)
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_clearance.cpp -o "$out/clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_ik_root_reach.cpp -o "$out/root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_candidate_recovery.cpp -o "$out/recovery-san.o"
g++ "${san[@]}" -c tests/cpp/test_g1_frame_transaction.cpp \
  -o "$out/frame-san.o"
g++ "${san[@]}" "$out/frame-san.o" "$out/clearance-san.o" \
  "$out/root-reach-san.o" "$out/recovery-san.o" -o "$out/frame-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$out/frame-san"
g++ "${san[@]}" "${rayinc[@]}" \
  -c tests/cpp/test_g1_controller_logging.cpp -o "$out/logging-san.o"
g++ "${san[@]}" "$out/logging-san.o" "$out/clearance-san.o" \
  "$out/root-reach-san.o" "$out/recovery-san.o" "${raylib[@]}" \
  -o "$out/logging-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$out/logging-san"
```

Expected: strict/fast tests and all three unchanged-checker fixture invocations exit `0`; the recovery row passes the existing beat-incumbent rule without the four-ULP exception; the incumbent row retains exact sentinels; sanitizers are silent; the first producer scratch is authenticated; canonical disabled output survives every hidden poison; and both immutable hashes report `OK`.

- [ ] **Step 6: Commit failure publication and logging projection**

```bash
git add g1_frame_transaction.h controller.cpp \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_controller_logging.cpp
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: publish bounded candidate outcomes atomically"
```

Expected staged paths: exactly the four Task 4 files; neither schema nor checker is staged.

---

### Task 5: Add Same-Build Certification Trace and Isolated Transaction Timing

**Files:**
- Create: `g1_candidate_certification_trace.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_frame_transaction_production.cpp`
- Modify: `tests/cpp/test_g1_controller_logging.cpp`
- Verify only: `motion_match_log.h`
- Verify only: `resources/check_g1_runtime_log.py`
- Consume: Tasks 1-4

**Interfaces:**
- `g1_candidate_certification_trace.h` has no declaration, type, data object, or function in a normal build. Its implementation exists only when both `G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM` and `G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM` are defined.
- Under both seams, expose exactly:

```cpp
struct G1CandidateTraceFile
{
    std::FILE* stream = nullptr;
    bool header_written = false;
};

static inline bool g1_candidate_trace_open(
    G1CandidateTraceFile& file,
    const char* path,
    char* error,
    int error_capacity);

static inline bool g1_candidate_trace_append_after_transaction(
    G1CandidateTraceFile& file,
    uint32_t presentation_frame,
    const G1CandidateCertificationTrace& trace,
    char* error,
    int error_capacity);

static inline void g1_candidate_trace_close(
    G1CandidateTraceFile& file);
```

- `g1_candidate_trace_append_after_transaction` calls `g1_recovery_candidates_exhaustive_for_test` only when the transaction trace says a recovery request/set exists. It passes `trace.recovery_request` unchanged, so the `database*`, 31 raw-query words, frames, transition/public-cost bits, and exclusions are the exact live values from that process. It compares the complete accelerated tail with the exhaustive tail by count, kind, selected/executed frame, source range, exact cost bits, recovery rank, transition flag, and ordering before writing. A mismatch is a global test-process failure, never a candidate retry or selection input.
- The TSV has this exact header and one row for slot zero plus every materialized tail record, including unattempted records after success:

```text
presentation_frame\tcandidate_slot\tcandidate_kind\tscore_owner\tselected_frame\texecuted_frame\tsource_range\tcost_bits_hex\trecovery_rank\tattempted\tcommon\traw\tik\trejection_stage\tstop_reason\tlegacy_traversals\trecovery_provider_calls\trecovery_traversals\taccelerated_count\texhaustive_count\toracle_equal
```

Disposition text is exactly `not-run`, `accepted`, `finite-rejected`, or `global-error`; candidate kind is `legacy`, `strict-recovery`, or `incumbent`; stop/rejection text uses existing enum string owners. `cost_bits_hex` is eight lowercase hex digits without floating-point formatting. `oracle_equal` is `1` only after the complete set comparison; a row without recovery uses `1` with both counts zero.
- Under `G1_FRAME_TRANSACTION_BENCHMARK` only, `controller.cpp` uses `std::chrono::steady_clock` immediately before and after only `g1_frame_transaction_run`, appends one integer nanosecond duration per completed transaction to `MM_TRANSACTION_TIMINGS`, and performs all formatting/file I/O after the end timestamp. Timing data never enters `G1FrameExternalInputs`, state, scratch, trace, candidate provider, CSV, or selection.
- Under the dual test seams only, main reads `MM_CANDIDATE_TRACE`, opens it before the route loop, passes a write-only trace pointer through `G1FrameTransactionTestSeam`, and calls the trace appender only after `g1_frame_transaction_run` returns. There is no production default path and no trace output unless the environment variable is nonempty.

- [ ] **Step 1: Write RED tests for full-set oracle equality and post-selection isolation**

Add these exact production tests:

```cpp
static void test_trace_serializes_slot_zero_complete_tail_and_attempts();
static void test_trace_exhaustive_oracle_uses_exact_live_request_and_database();
static void test_trace_fails_instead_of_truncating_ninth_record();
static void test_trace_is_written_only_after_candidate_selection();
static void test_exhaustive_oracle_result_cannot_change_selected_candidate();
static void test_unattempted_tail_records_are_not_run_in_trace();
static void test_trace_cost_words_are_hex_not_reformatted_floats();
static void test_trace_slot_zero_uses_authenticated_score_owner();
static void test_benchmark_timestamp_surrounds_only_outer_transaction();
```

Use a synthetic accelerated tail with six transitions plus the incumbent. Accept the second transition, then require TSV rows for slot zero and the full seven-record tail: the first three slots are attempted with real dispositions, remaining slots are `attempted=0` and all branches `not-run`. Mutate one exhaustive frame, cost bit, source range, order, rank, and count independently and require the trace append to fail before writing any partial transaction block. Run three slot-zero-only cases: scheduled legacy has `attempts[0].score_owner == G1CandidateScoreLegacy` and text `legacy`; unscheduled search and matching-disabled each have incumbent kind, `attempts[0].score_owner == G1CandidateScoreIncumbent`, and text `incumbent`. A mismatch between candidate kind and authenticated score owner fails serialization rather than being relabeled.

To prove the oracle is post-selection only, snapshot accepted state, selected record, attempt trace, provider counters, and complete output bytes after the transaction. Append the authentic snapshot once, then pass copied completed traces with (a) one accelerated-set cost bit changed and (b) a copied request corrupted so the exhaustive call globally errors. The authentic append succeeds; the two copied trace appends fail before writing a block; the already-selected state/output snapshot remains byte-identical throughout. The production coordinator and provider signatures contain no oracle parameter or trace-return path.

Add a compile-only no-seam source assertion inside the logging/source test: preprocess `controller.cpp` without either seam and require no token `G1CandidateTraceFile`, `MM_CANDIDATE_TRACE`, `g1_candidate_trace_append_after_transaction`, or `g1_recovery_candidates_exhaustive_for_test` in the preprocessed translation unit.

Run RED:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task5-red
rm -rf "$out"
mkdir -p "$out"
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
     -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
     -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
     -c tests/cpp/test_g1_frame_transaction_production.cpp \
     -o "$out/trace-red.o" 2>"$out/trace-red.stderr"; then
  echo 'ERROR: trace RED unexpectedly compiled' >&2
  exit 1
fi
rg 'g1_candidate_certification_trace|G1CandidateTraceFile|not declared|No such file' \
  "$out/trace-red.stderr"
```

Expected: failure is the missing trace interface, not an inherited controller or strict-kernel error.

- [ ] **Step 2: Implement transactional TSV serialization and exact same-build comparison**

Create the seam-only header. Validate an absolute, nonempty path; reject an already-open file; open with `"wb"`; and make close idempotent. Build one transaction's lines into a fixed local character buffer sized from `G1CandidateAttemptCapacity`, validate every `snprintf` result, and call `fwrite` only after the full oracle comparison and full formatting succeed. Flush after a complete block. Never allocate proportional to the database or candidate count.

Construct the complete trace sequence as:

```text
slot 0: trace.attempts[0].candidate and trace.attempts[0].score_owner
tail:   trace.recovery_set.records[0..count-1]
```

Map slot-zero score-owner text only from the authenticated `trace.attempts[0].score_owner`; never hard-code `legacy`. Require legacy kind/legacy owner or incumbent kind/incumbent owner and reject every other slot-zero kind/owner pairing. Map tail dispositions from attempted records by exact candidate equality; tail entries after first acceptance remain not-run. Reject duplicate candidates, missing attempted prefixes, a tail count above seven, an attempt count above eight, recovery counters outside `(0 or 1)`, a recovery set without its request, a request without the exact database pointer, and any set/oracle mismatch. The serializer may read selection output but cannot mutate runtime, scratch, request, set, state, or provider counters.

Place the exhaustive call in this post-transaction header path. Keep `g1_recovery_candidates_build` free of exhaustive references; add both a source and `nm -C` guard for that separation.

- [ ] **Step 3: Add opt-in live trace and benchmark plumbing after the transaction call**

In controller main, create seam/timing contexts before the loop only under their macros. Use environment variables only to choose output files; absence means disabled. Around the existing call, the benchmark block must have this exact shape:

```cpp
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
const auto transaction_begin = std::chrono::steady_clock::now();
#endif
const G1FrameTransactionStatus status = ::g1_frame_transaction_run(
    frame_runtime,
    ::g1_controller_frame_stage_run,
    ::g1_recovery_candidates_build,
    external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    test_seam_pointer,
#endif
    error,
    sizeof(error));
#if defined(G1_FRAME_TRANSACTION_BENCHMARK)
const auto transaction_end = std::chrono::steady_clock::now();
transaction_timings.append(transaction_end - transaction_begin);
#endif
```

After status handling and before the next outer frame begins, append the certification trace using the presentation frame belonging to that transaction. The appender must not run inside the coordinator or runner. Close both output streams on every normal/global-error cleanup path using the existing cleanup ownership, without changing runtime log flush/close behavior.

The benchmark writer outputs the exact header `presentation_frame\tduration_ns` and one row per completed transaction. Tests parse source tokens to require no render, log-row build, `fwrite`, `fflush`, or candidate-trace call between the two timestamps.

- [ ] **Step 4: Run seam equality, no-seam privacy, strict/fast parity, and recorded-switch gates**

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task5
rm -rf "$out"
mkdir -p "$out"
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math \
  -frecord-gcc-switches -I.)
warn=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)

g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/recovery-seam.o"
for flavor in strict fast; do
  flags=("${warn[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" "${rayinc[@]}" \
    -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$out/controller-${flavor}.o"
  g++ "${flags[@]}" \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$out/trace-test-${flavor}.o"
  g++ "$out/trace-test-${flavor}.o" "$out/controller-${flavor}.o" \
    "$out/clearance.o" "$out/root-reach.o" "$out/recovery-seam.o" \
    "${raylib[@]}" -o "$out/trace-test-${flavor}"
  "$out/trace-test-${flavor}" --trace-transcript \
    >"$out/trace-${flavor}.tsv"
done
cmp "$out/trace-strict.tsv" "$out/trace-fast.tsv"

g++ "${strict[@]}" -c g1_candidate_recovery.cpp \
  -o "$out/recovery-production.o"
g++ "${fast[@]}" "${rayinc[@]}" -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -c controller.cpp -o "$out/controller-production.o"
g++ "$out/controller-production.o" "$out/clearance.o" \
  "$out/root-reach.o" "$out/recovery-production.o" "${raylib[@]}" \
  -o "$out/controller-production"
! nm -C "$out/controller-production" | \
  rg 'CandidateTrace|CertificationTrace|exhaustive_for_test|ProviderAudit'
! strings "$out/controller-production" | \
  rg 'MM_CANDIDATE_TRACE|oracle_equal|recovery_traversals'
! nm -C "$out/recovery-production.o" | rg 'exhaustive_for_test'

g++ "${fast[@]}" "${rayinc[@]}" -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -DG1_FRAME_TRANSACTION_BENCHMARK -c controller.cpp \
  -o "$out/controller-benchmark.o"
g++ "$out/controller-benchmark.o" "$out/clearance.o" \
  "$out/root-reach.o" "$out/recovery-production.o" "${raylib[@]}" \
  -o "$out/controller-benchmark"

for object in clearance root-reach recovery-production; do
  readelf -p .GCC.command.line "$out/${object}.o" \
    >"$out/${object}.switches"
  rg -- '-fno-fast-math' "$out/${object}.switches"
  rg -- '-ffp-contract=off' "$out/${object}.switches"
  rg -- '-frounding-math' "$out/${object}.switches"
  ! rg -- '(^| )-ffast-math( |$)' "$out/${object}.switches"
  ! readelf -SW "$out/${object}.o" | rg '\.gnu\.lto'
done
sha256sum -c /tmp/g1-bounded-candidate-cert/base/immutable-files.before.sha256
git diff --check
```

Run the trace integration under the same sanitizers:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task5
san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_clearance.cpp -o "$out/trace-clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_ik_root_reach.cpp -o "$out/trace-root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/trace-recovery-san.o"
g++ "${san[@]}" "${rayinc[@]}" -DG1_CONTROLLER_NO_MAIN \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$out/trace-controller-san.o"
g++ "${san[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o "$out/trace-test-san.o"
g++ "${san[@]}" "$out/trace-test-san.o" \
  "$out/trace-controller-san.o" "$out/trace-clearance-san.o" \
  "$out/trace-root-reach-san.o" "$out/trace-recovery-san.o" \
  "${raylib[@]}" -o "$out/trace-test-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$out/trace-test-san" --trace-transcript > "$out/trace-san.tsv"
```

Expected: same-build accelerated/exhaustive sets are bit-identical; strict/fast trace transcripts match; mismatches fail before a partial block; no seam symbol/string survives production; the benchmark build contains timing only; strict switches are recorded and there is no LTO; sanitizers and immutable hashes are clean.

- [ ] **Step 5: Run the complete focused source suite before live certification**

Run the exact Task 1 provider, Task 2 generic coordinator, Task 3 real production runner, and Task 4 logging/publication command blocks again, then execute this inherited dependency-mapped matrix. The three implementation objects remain strict; only test/controller callers vary. Every final link is neutral:

```bash
set -euo pipefail
out=/tmp/g1-bounded-candidate-cert/task5-full
rm -rf "$out"
mkdir -p "$out"
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
warn=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)

g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-reach-seam.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp -o "$out/recovery.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/recovery-seam.o"

plain=(
  cleanup_runtime g1_candidate_audit g1_clearance g1_command_runtime
  g1_controller_state g1_footprint_runtime g1_frame_transaction
  g1_skeleton motion_match_log route_runtime scene_runtime scene_switch
  support_matching support_runtime terrain_database terrain_runtime
)
embedded_controller=(g1_candidate_audit_controller g1_controller_logging)

for flavor in strict fast; do
  flags=("${warn[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")

  for name in "${plain[@]}"; do
    g++ "${flags[@]}" -c "tests/cpp/test_${name}.cpp" \
      -o "$out/${name}-${flavor}.o"
    g++ "$out/${name}-${flavor}.o" "$out/clearance.o" \
      "$out/root-reach.o" "$out/recovery.o" "${raylib[@]}" \
      -o "$out/${name}-${flavor}"
    "$out/${name}-${flavor}"
    case "$name" in
      cleanup_runtime|route_runtime|support_matching)
        "$out/${name}-${flavor}" --controller controller.cpp
        ;;
    esac
    if test "$name" = route_runtime; then
      "$out/${name}-${flavor}" --route-header route_runtime.h
    fi
    if test "$name" = g1_footprint_runtime; then
      "$out/${name}-${flavor}" --parity \
        > "$out/footprint-${flavor}.txt"
    fi
  done

  for name in "${embedded_controller[@]}"; do
    g++ "${flags[@]}" "${rayinc[@]}" \
      -c "tests/cpp/test_${name}.cpp" \
      -o "$out/${name}-${flavor}.o"
    g++ "$out/${name}-${flavor}.o" "$out/clearance.o" \
      "$out/root-reach.o" "$out/recovery.o" "${raylib[@]}" \
      -o "$out/${name}-${flavor}"
    "$out/${name}-${flavor}"
  done

  g++ "${flags[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
    -c tests/cpp/test_g1_ik.cpp -o "$out/g1_ik-${flavor}.o"
  g++ "$out/g1_ik-${flavor}.o" "$out/clearance.o" \
    "$out/root-reach-seam.o" "$out/recovery.o" \
    -o "$out/g1_ik-${flavor}"
  "$out/g1_ik-${flavor}"
  "$out/g1_ik-${flavor}" --parity > "$out/ik-${flavor}.txt"

  g++ "${flags[@]}" "${rayinc[@]}" -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$out/controller-runner-${flavor}.o"
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$out/frame-production-${flavor}.o"
  g++ "$out/frame-production-${flavor}.o" \
    "$out/controller-runner-${flavor}.o" "$out/clearance.o" \
    "$out/root-reach.o" "$out/recovery-seam.o" "${raylib[@]}" \
    -o "$out/frame-production-${flavor}"
  "$out/frame-production-${flavor}"
done

cmp "$out/footprint-strict.txt" "$out/footprint-fast.txt"
cmp "$out/ik-strict.txt" "$out/ik-fast.txt"

g++ "${warn[@]}" -c tests/cpp/compile_g1_candidate_recovery_production.cpp \
  -o "$out/recovery-positive.o"
g++ "$out/recovery-positive.o" "$out/recovery.o" \
  -o "$out/recovery-positive"
"$out/recovery-positive"
g++ "${warn[@]}" -c tests/cpp/compile_g1_ik_production.cpp \
  -o "$out/ik-positive.o"
g++ "$out/ik-positive.o" "$out/clearance.o" "$out/root-reach.o" \
  -o "$out/ik-positive"
"$out/ik-positive"

if g++ "${warn[@]}" \
     -c tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp \
     -o "$out/recovery-negative.o" 2>"$out/recovery-negative.stderr"; then
  echo 'ERROR: recovery seam negative compiled' >&2
  exit 1
fi
rg 'g1_recovery_candidates_exhaustive_for_test|not declared' \
  "$out/recovery-negative.stderr"
if g++ "${warn[@]}" -c tests/cpp/compile_g1_ik_seam_negative.cpp \
     -o "$out/ik-negative.o" 2>"$out/ik-negative.stderr"; then
  echo 'ERROR: IK seam negative compiled' >&2
  exit 1
fi
rg 'for_test|not declared' "$out/ik-negative.stderr"
for mode in VOID_CONTEXT PUBLICATION_CONTEXT ACCEPTED_CONTEXT; do
  if g++ "${warn[@]}" -D"G1_FRAME_NEGATIVE_${mode}" \
       -c tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
       -o "$out/runner-negative-${mode}.o" \
       2>"$out/runner-negative-${mode}.stderr"; then
    echo "ERROR: forbidden ${mode} runner context compiled" >&2
    exit 1
  fi
  rg 'G1FrameStageRunner|convert|conversion|argument' \
    "$out/runner-negative-${mode}.stderr"
done

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests
sha256sum -c /tmp/g1-bounded-candidate-cert/base/immutable-files.before.sha256
git diff --check
```

Expected: all 20 named native test programs pass in both caller modes; the production runner passes with both seams; controller/route/header source modes pass; IK and footprint parity transcripts are byte-identical; both positive production fixtures run; all five compile-only privacy/context cases fail for their intended identifier/type reason; Python is green; immutable hashes and `git diff --check` pass. Do not narrow the matrix after a failure.

- [ ] **Step 6: Commit only opt-in evidence instrumentation**

```bash
git add g1_candidate_certification_trace.h controller.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp \
  tests/cpp/test_g1_controller_logging.cpp
git diff --cached --check
git diff --cached --name-only
git commit -m "test: trace bounded candidate certification"
```

Expected staged paths: exactly the four Task 5 files. The no-seam runtime behavior and production CSV schema remain unchanged.

---

### Task 6: Certify the Authentic Low-Curb Recovery, Performance, Gate E, and Gate L

**Files:**
- Verify only: all source and test files from Tasks 1-5
- Verify only: `/tmp/g1-terrain-footprint-runtime-v1`
- Verify only as asset evidence, never as a behavior oracle: `/tmp/g1-footprint-prechange-oracle`
- Consume only after the 32-frame marker: Task 1's hashed current-frontier controller and captures under `/tmp/g1-bounded-candidate-cert/base/current-frontier/`
- Generate only: `/tmp/g1-bounded-candidate-cert/live/`
- Do not modify or commit any repository file in this task
- Consume: Tasks 1-5 and the mandatory Task 1 execution-base evidence

**Live-build contract:**
- Build three disposable executables: dual-seam trace, no-seam release, and no-seam benchmark. Each uses separately compiled strict `g1_clearance.cpp`, `g1_ik_root_reach.cpp`, and `g1_candidate_recovery.cpp`; a fast controller caller; a neutral final link; and no LTO.
- The trace executable uses the same loaded in-memory motion `database` for production accelerated recovery and the post-transaction exhaustive oracle. Never transfer a score/list between separately compiled binaries as an acceptance oracle.
- The release executable produces the unchanged production CSV only. The benchmark executable adds only the transaction timing file.
- Every Gate L physical/property checker consumes a no-seam release CSV. A separate trace process reruns the exact deterministic inputs into a production-schema CSV and TSV; byte-compare the release and trace CSVs before using that TSV for same-build oracle or recovery-boundary evidence.
- Never run an 800- or 832-frame command until the paired 32-frame trace/runtime prerequisite creates `/tmp/g1-bounded-candidate-cert/live/ZERO_REJECTION_32.PASS`.
- After that marker, the hashed Task 1 controller is the only separate-build behavior baseline. For each compared run, authenticate the candidate's first recovery presentation frame from its same-process TSV. Compare every `GATE_L_ORACLE_INVARIANTS` field through the exact prefix before that frame; require full-run equality when no recovery occurs. The exact same-build accelerated/exhaustive trace remains authoritative for every recovery row.
- The older `/tmp/g1-footprint-prechange-oracle` corpus is retained only to prove those assets were not modified. Never pass one of its paths to any behavior checker.
- Never inspect, discover, signal, replace, restart, attach to, or otherwise touch the running visualizer. All commands below invoke unique disposable paths directly; none uses `ps`, `pgrep`, `pidof`, `kill`, `pkill`, `systemctl`, or a visualizer helper.

- [ ] **Step 1: Build and authenticate the three disposable executables**

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
rm -rf "$root"
mkdir -p "$root/build" "$root/trace32" "$root/low-curb" \
  "$root/performance" "$root/gate-e" "$root/gate-l"
test "$(sha256sum docs/superpowers/specs/2026-07-16-g1-bounded-candidate-certification-design.md | cut -d' ' -f1)" = \
  10c21cb51a351b9cc2f4c1e03d58472e485bba36d0449967de4e08e33a7ec6dd
sha256sum -c /tmp/g1-bounded-candidate-cert/base/immutable-files.before.sha256
sha256sum -c \
  /tmp/g1-bounded-candidate-cert/base/current-frontier/SHA256SUMS
(cd /tmp/g1-footprint-prechange-oracle && sha256sum -c SHA256SUMS)

strict=(-std=c++17 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)

g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/build/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/build/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp \
  -o "$root/build/recovery-production.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/build/recovery-seam.o"

g++ "${fast[@]}" "${rayinc[@]}" \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$root/build/controller-trace.o"
g++ "$root/build/controller-trace.o" "$root/build/clearance.o" \
  "$root/build/root-reach.o" "$root/build/recovery-seam.o" \
  "${raylib[@]}" -o "$root/controller-trace"

g++ "${fast[@]}" "${rayinc[@]}" -c controller.cpp \
  -o "$root/build/controller-release.o"
g++ "$root/build/controller-release.o" "$root/build/clearance.o" \
  "$root/build/root-reach.o" "$root/build/recovery-production.o" \
  "${raylib[@]}" -o "$root/controller-release"

g++ "${fast[@]}" "${rayinc[@]}" \
  -DG1_FRAME_TRANSACTION_BENCHMARK -c controller.cpp \
  -o "$root/build/controller-benchmark.o"
g++ "$root/build/controller-benchmark.o" "$root/build/clearance.o" \
  "$root/build/root-reach.o" "$root/build/recovery-production.o" \
  "${raylib[@]}" -o "$root/controller-benchmark"

for object in clearance root-reach recovery-production recovery-seam; do
  readelf -p .GCC.command.line "$root/build/${object}.o" \
    >"$root/build/${object}.switches"
  rg -- '-fno-fast-math' "$root/build/${object}.switches"
  rg -- '-ffp-contract=off' "$root/build/${object}.switches"
  rg -- '-frounding-math' "$root/build/${object}.switches"
  ! rg -- '(^| )-ffast-math( |$)' "$root/build/${object}.switches"
  ! readelf -SW "$root/build/${object}.o" | rg '\.gnu\.lto'
done
! nm -C "$root/controller-release" | \
  rg 'CandidateTrace|CertificationTrace|exhaustive_for_test|ProviderAudit'
! strings "$root/controller-release" | \
  rg 'MM_CANDIDATE_TRACE|MM_TRANSACTION_TIMINGS|oracle_equal'
! nm -C "$root/controller-benchmark" | \
  rg 'CandidateTrace|CertificationTrace|exhaustive_for_test|ProviderAudit'
sha256sum "$root/controller-trace" "$root/controller-release" \
  "$root/controller-benchmark" > "$root/BINARIES.sha256"
```

Expected: all three neutral links succeed, all strict objects record all three strict flags and no fast/LTO flag, release contains no test-seam/timing string, benchmark contains no candidate trace/oracle, the Task 1 current-frontier controller/captures and older asset-only corpus authenticate unchanged, and no running process has been queried or changed.

- [ ] **Step 2: Pass the mandatory paired 32-frame same-build prerequisite**

Run only 32-frame processes in this step:

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
rm -f "$root/ZERO_REJECTION_32.PASS"
for ik in 0 1; do
  DISPLAY=:1 MM_IK="$ik" \
    MM_CANDIDATE_TRACE="$root/trace32/candidates-${ik}.tsv" \
    "$root/controller-trace" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene grail-curb-low --test-mode route \
    --test-route curb-forward --test-frames 32 --test-heading forward \
    --terrain-weight 4 --log "$root/trace32/runtime-${ik}.csv"
done
cmp "$root/trace32/candidates-0.tsv" \
  "$root/trace32/candidates-1.tsv"

/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv
import struct
from pathlib import Path
from resources.check_g1_runtime_log import (
    GATE_E_PAIR_INVARIANT_GROUPS,
    check_rows,
    read_rows,
)

root = Path("/tmp/g1-bounded-candidate-cert/live")
off = read_rows(root / "trace32/runtime-0.csv")
on = read_rows(root / "trace32/runtime-1.csv")
assert len(off) == len(on) == 32
check_rows(off)
check_rows(on, allow_ik=True)
for index, (left, right) in enumerate(zip(off, on)):
    assert left["ik_enabled"] == "0"
    assert right["ik_enabled"] == "1"
    assert left["frame_rejected"] == right["frame_rejected"] == "0"
    assert left["ik_safe_stop_latched"] == right["ik_safe_stop_latched"] == "0"
    for _, names in GATE_E_PAIR_INVARIANT_GROUPS:
        for name in names:
            assert left[name] == right[name], (index, name, left[name], right[name])

row = off[6]
bits = lambda text: struct.unpack("<I", struct.pack("<f", float(text)))[0]
assert row["query_database_frame"] == "272446"
assert row["selected_database_frame"] == on[6]["selected_database_frame"]
assert row["frame_rejected"] == "0"
assert row["ik_safe_stop_latched"] == "0"

with (root / "trace32/candidates-0.tsv").open(
        newline="", encoding="utf-8") as stream:
    trace = list(csv.DictReader(stream, delimiter="\t"))
rows = [value for value in trace if value["presentation_frame"] == "6"]
assert rows
slot0 = rows[0]
assert slot0["candidate_slot"] == "0"
assert slot0["candidate_kind"] == "legacy"
assert slot0["score_owner"] == "legacy"
assert slot0["selected_frame"] == "428956"
assert slot0["source_range"] == "1647"
assert slot0["cost_bits_hex"] == "40250630"
assert slot0["attempted"] == "1"
assert slot0["common"] == slot0["raw"] == "accepted"
assert slot0["ik"] == "finite-rejected"
assert slot0["stop_reason"] == "no-swing-candidate"
assert slot0["legacy_traversals"] == "1"
assert slot0["recovery_provider_calls"] == "1"
assert slot0["recovery_traversals"] == "1"
assert all(value["oracle_equal"] == "1" for value in rows)
assert all(value["accelerated_count"] == value["exhaustive_count"] for value in rows)

tail = rows[1:]
assert 1 <= len(tail) <= 7
strict_frames = [value["selected_frame"] for value in tail
                 if value["candidate_kind"] == "strict-recovery"]
assert len(strict_frames) == len(set(strict_frames))
assert "428956" not in strict_frames and "272446" not in strict_frames
accepted = [value for value in rows[1:]
            if value["attempted"] == "1" and
               value["common"] == value["raw"] == value["ik"] == "accepted"]
assert len(accepted) == 1
winner = accepted[0]
winner_slot = int(winner["candidate_slot"])
for earlier in rows[:winner_slot]:
    assert earlier["attempted"] == "1"
    assert "finite-rejected" in (
        earlier["common"], earlier["raw"], earlier["ik"])
for later in rows[winner_slot + 1:]:
    assert later["attempted"] == "0"
    assert later["common"] == later["raw"] == later["ik"] == "not-run"
assert row["selected_database_frame"] == winner["selected_frame"]
if winner["candidate_kind"] == "strict-recovery":
    assert bits(row["selected_cost"]) == int(winner["cost_bits_hex"], 16)
    assert float(row["selected_cost"]) < float(row["incumbent_cost"])
else:
    assert winner["candidate_kind"] == "incumbent"
print("VALID paired 32-frame bounded-candidate prerequisite")
(root / "ZERO_REJECTION_32.PASS").write_text("PASS\n", encoding="ascii")
PY
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
```

Expected: the two traces are byte-identical; row 6 preserves the exact legacy slot-zero owner then chooses the first later dual-certified bounded record; the complete accelerated set equals the same-process exhaustive set without hard-coded recovery scores; every earlier attempt failed; no later attempt ran; both public rows select the same record and contain no private rejection/safe-stop; all 32 paired rows pass canonical/runtime and pair-invariant checks. The marker is created only after every assertion passes.

- [ ] **Step 3: Run exact 800-row low-curb release logs and three 32+800 timing trials**

Every long command begins by checking the prerequisite marker:

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
for ik in 0 1; do
  DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene grail-curb-low --test-mode route \
    --test-route curb-forward --test-frames 800 --test-heading forward \
    --terrain-weight 4 --log "$root/low-curb/800-${ik}.csv"
done
read -r end_x end_z <<<"$(jq -r \
  '.routes[] | select(.id == "curb-forward") | .waypoints_xz[-1] | @tsv' \
  /tmp/g1-terrain-footprint-runtime-v1/scenes/grail-curb-low/scene.json)"
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py "$root/low-curb/800-1.csv" \
  --gate-l --compare-ik-off "$root/low-curb/800-0.csv" \
  --expected-end-x "$end_x" --expected-end-z "$end_z" \
  --expected-heading forward --require-multilevel
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
from resources.check_g1_runtime_log import read_rows
root = "/tmp/g1-bounded-candidate-cert/live/low-curb"
for ik in (0, 1):
    rows = read_rows(f"{root}/800-{ik}.csv")
    assert len(rows) == 800
    assert all(row["frame_rejected"] == "0" for row in rows)
    assert all(row["ik_safe_stop_latched"] == "0" for row in rows)
    assert any(row["route_complete"] == "1" for row in rows)
print("VALID exact 800-row low-curb release logs")
PY

for run in 1 2 3; do
  test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
  DISPLAY=:1 MM_IK=1 \
    MM_TRANSACTION_TIMINGS="$root/performance/run-${run}.tsv" \
    "$root/controller-benchmark" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene grail-curb-low --test-mode route \
    --test-route curb-forward --test-frames 832 --test-heading forward \
    --terrain-weight 4 --log "$root/performance/run-${run}.csv"
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv
import math
import statistics
from pathlib import Path
root = Path("/tmp/g1-bounded-candidate-cert/live/performance")
for run in (1, 2, 3):
    with (root / f"run-{run}.tsv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 832
    measured = [int(row["duration_ns"]) for row in rows[32:]]
    assert len(measured) == 800
    assert all(value > 0 for value in measured)
    mean_ms = statistics.fmean(measured) / 1_000_000.0
    ordered = sorted(measured)
    p99_ms = ordered[math.ceil(0.99 * len(ordered)) - 1] / 1_000_000.0
    assert mean_ms <= 40.0, (run, mean_ms)
    assert p99_ms <= 80.0, (run, p99_ms)
    print(f"VALID performance run={run} mean_ms={mean_ms:.6f} p99_ms={p99_ms:.6f}")
PY
```

Expected: release logs contain exactly 800 accepted rows, route completion, exact heading and existing physical/work bounds; all three timing files contain 832 transactions, discard exactly the first 32, and independently pass mean `<= 40 ms` and nearest-rank p99 `<= 80 ms` over the following 800. Timing excludes startup, rendering, logging, and flush.

- [ ] **Step 4: Re-run the nine normal, four stress, and two blocked Gate-E/Gate-D-IK pairs**

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
normal=(
  grail-curb-low:curb-forward
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  stairs-unseen-variable:ascent-landing-descent
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
  cross-slope-05:forward-cross-slope
  cross-slope-10:forward-cross-slope
  mixed-multilevel:full-course
)
for specification in "${normal[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  stem="${scene}__${route}"
  for ik in 0 1; do
    DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene "$scene" --test-mode route --test-route "$route" \
      --test-frames 800 --test-heading forward --terrain-weight 4 \
      --log "$root/gate-e/normal-${stem}-${ik}.csv"
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "$root/gate-e/normal-${stem}-1.csv" --gate-e \
    --compare-ik-off "$root/gate-e/normal-${stem}-0.csv" \
    --expected-scene "$scene" --expected-route "$route"
done

stress=(
  grail-curb-default:curb-forward
  grail-curb-medium:curb-forward
  grail-curb-high:curb-forward
  ramp-15-stress:up-landing-down
)
for specification in "${stress[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  stem="${scene}__${route}"
  for ik in 0 1; do
    DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene "$scene" --test-mode route --test-route "$route" \
      --test-frames 800 --test-heading forward --terrain-weight 4 \
      --log "$root/gate-e/stress-${stem}-${ik}.csv"
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "$root/gate-e/stress-${stem}-1.csv" --gate-e-stress \
    --compare-ik-off "$root/gate-e/stress-${stem}-0.csv" \
    --expected-scene "$scene" --expected-route "$route"
done

for route in wall-safe-stop ramp-safe-stop; do
  for ik in 0 1; do
    DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene blocked-course --test-mode route --test-route "$route" \
      --test-frames 600 --test-heading forward --terrain-weight 4 \
      --log "$root/gate-e/blocked-${route}-${ik}.csv"
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "$root/gate-e/blocked-${route}-0.csv" --gate-d
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "$root/gate-e/blocked-${route}-1.csv" --gate-d-ik \
    --compare-ik-off "$root/gate-e/blocked-${route}-0.csv" \
    --expected-scene blocked-course --expected-route "$route"
done
```

Expected: 9 normal Gate E pairs pass and complete, 4 stress pairs retain their checker-classified traverse/safe-stop branch, and both blocked controls plus Gate-D-IK pairs retain the exact obstruction/rejection signature and atomic tail. Every pre-stop pair invariant—including query/frames/costs, terrain, support, simulation, route, travel, and heading—remains byte-identical; IK-off suffixes remain canonical.

- [ ] **Step 5: Re-run the complete 30-cell Gate L matrix plus lateral, exit, tangent, and flat gates**

Run the full physical quality gate only on no-seam `controller-release` logs.
For every identical mode/cell input, run a second `controller-trace` process to
produce a disposable production-schema CSV plus its TSV. Require the release
and trace CSV files to be byte-identical before the TSV may locate a recovery
boundary. Thus the seam process supplies post-selection oracle evidence but is
never the only certified runtime. The comparison helper then authenticates
every same-build accelerated/exhaustive row, discovers the first recovery frame
from the TSV rather than from behavior differences, and never reads the older
allowlisted oracle corpus:

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
base=/tmp/g1-bounded-candidate-cert/base
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
sha256sum -c "$base/current-frontier/SHA256SUMS"

compare_frontier() {
  /home/ubuntu/miniconda3/envs/diffsim/bin/python - \
    "$1" "$2" "$3" <<'PY'
import csv
import sys
from pathlib import Path
from resources.check_g1_runtime_log import (
    GATE_L_ORACLE_INVARIANTS,
    read_rows,
)

current_path, frontier_path, trace_path = map(Path, sys.argv[1:4])
current = read_rows(current_path)
frontier = read_rows(frontier_path)
assert len(current) == len(frontier) and current
with trace_path.open(newline="", encoding="utf-8") as stream:
    trace = list(csv.DictReader(stream, delimiter="\t"))
assert trace
assert all(row["oracle_equal"] == "1" for row in trace)

recovery_frames = set()
for row in trace:
    calls = int(row["recovery_provider_calls"])
    traversals = int(row["recovery_traversals"])
    accelerated = int(row["accelerated_count"])
    exhaustive = int(row["exhaustive_count"])
    assert calls in (0, 1)
    if calls == 0:
        assert traversals == accelerated == exhaustive == 0
    else:
        assert traversals == 1
        assert accelerated == exhaustive
        recovery_frames.add(int(row["presentation_frame"]))

frame_to_index = {}
for index, row in enumerate(current):
    frame = int(row["frame"])
    assert frame not in frame_to_index
    frame_to_index[frame] = index
assert list(frame_to_index) == list(range(len(current)))
trace_frames = {int(row["presentation_frame"]) for row in trace}
assert trace_frames == set(frame_to_index)

if recovery_frames:
    first_recovery = min(recovery_frames)
    assert first_recovery in frame_to_index
    stop = frame_to_index[first_recovery]
    mode = f"prefix-before-recovery-{first_recovery}"
else:
    stop = len(current)
    mode = "full-zero-recovery"

for index, (candidate, baseline) in enumerate(
        zip(current[:stop], frontier[:stop])):
    for name in GATE_L_ORACLE_INVARIANTS:
        assert candidate[name] == baseline[name], (
            index, name, candidate[name], baseline[name])
print(
    f"VALID current-frontier comparison mode={mode} rows={stop} "
    f"candidate={current_path.name}")
PY
}

verify_frontier_capture() {
  /home/ubuntu/miniconda3/envs/diffsim/bin/python - \
    "$1" "$2" <<'PY'
import sys
from resources.check_g1_runtime_log import (
    GATE_L_ORACLE_INVARIANTS,
    read_rows,
)

long_rows = read_rows(sys.argv[1])
captured = read_rows(sys.argv[2])
assert len(long_rows) == 800 and len(captured) == 32
for index, (current, frozen) in enumerate(zip(long_rows[:32], captured)):
    for name in GATE_L_ORACLE_INVARIANTS:
        assert current[name] == frozen[name], (index, name)
print("VALID frozen current-frontier 32-frame capture")
PY
}

check_flat_l4_contract() {
  /home/ubuntu/miniconda3/envs/diffsim/bin/python - \
    "$1" "$2" "$3" "$4" <<'PY'
import sys
from resources.check_g1_runtime_log import (
    _check_disabled_ik_is_canonical,
    _gate_l_heading,
    _gate_l_surface_split,
    _integer,
    _relative_travel_direction,
    _require_full_runtime_header,
    check_rows,
    read_rows,
)

path, route, heading, expected_relative = sys.argv[1:5]
rows = read_rows(path)
_require_full_runtime_header(rows)
check_rows(rows, allow_ik=True)
assert len(rows) == 100
assert {(row["scene_id"], row["route"]) for row in rows} == {
    ("stairs-shallow", route)}
assert all(row["mode"] == "route" for row in rows)
assert len({_integer(row, "scene_generation", index)
            for index, row in enumerate(rows)}) == 1
_gate_l_heading(rows, heading)
assert all(_integer(row, "ik_enabled", index) == 0 and
           _integer(row, "ik_applied", index) == 0
           for index, row in enumerate(rows))
for index, row in enumerate(rows):
    _check_disabled_ik_is_canonical(row, index)
assert all(_integer(row, name, index) == 0
           for index, row in enumerate(rows)
           for name in (
               "footprint_blocked", "blocked", "ik_safe_stop_requested",
               "ik_safe_stop_latched", "frame_rejected"))
assert all(row["footprint_status"] == "invalid-input" for row in rows)
assert _integer(rows[-1], "route_complete", 99) == 1
maximum_split = _gate_l_surface_split(rows, False)
assert maximum_split < .04
relative, distance = _relative_travel_direction(rows, heading)
assert relative == expected_relative
print(
    f"VALID Gate L4 properties route={route} heading={heading} "
    f"relative={relative} distance={distance:.6f} "
    f"maximum_split={maximum_split:.6f}")
PY
}

quality="$root/gate-l/quality-verdicts.tsv"
: > "$quality"
routes=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  grail-curb-low:curb-forward
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
)
headings=(
  forward backward positive-x negative-x
  diagonal-positive-x diagonal-negative-x
)
for specification in "${routes[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  read -r end_x end_z <<<"$(jq -r --arg route "$route" \
    '.routes[] | select(.id == $route) | .waypoints_xz[-1] | @tsv' \
    "/tmp/g1-terrain-footprint-runtime-v1/scenes/${scene}/scene.json")"
  for heading in "${headings[@]}"; do
    stem="${scene}__${route}__${heading}"
    off="$root/gate-l/${stem}-off.csv"
    on="$root/gate-l/${stem}-on.csv"
    for ik in 0 1; do
      log="$off"; test "$ik" = 1 && log="$on"
      trace="$root/gate-l/${stem}-${ik}.tsv"
      trace_runtime="$root/gate-l/trace-runtime-${stem}-${ik}.csv"
      frontier="$root/gate-l/frontier-${stem}-${ik}.csv"
      DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
        --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
        --terrain-scene "$scene" --test-mode route --test-route "$route" \
        --test-frames 800 --test-heading "$heading" --terrain-weight 4 \
        --log "$log"
      DISPLAY=:1 MM_IK="$ik" MM_CANDIDATE_TRACE="$trace" \
        "$root/controller-trace" \
        --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
        --terrain-scene "$scene" --test-mode route --test-route "$route" \
        --test-frames 800 --test-heading "$heading" --terrain-weight 4 \
        --log "$trace_runtime"
      cmp "$log" "$trace_runtime"
      DISPLAY=:1 MM_IK="$ik" "$base/controller" \
        --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
        --terrain-scene "$scene" --test-mode route --test-route "$route" \
        --test-frames 800 --test-heading "$heading" --terrain-weight 4 \
        --log "$frontier"
      compare_frontier "$log" "$frontier" "$trace"
      if test "$heading" = forward; then
        verify_frontier_capture "$frontier" \
          "$base/current-frontier/forward32-${scene}__${route}-${ik}.csv"
      fi
    done
    extra=()
    if test "$heading" != forward; then
      extra=(--compare-forward \
        "$root/gate-l/${scene}__${route}__forward-on.csv")
    fi
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      resources/check_g1_runtime_log.py "$on" --gate-l \
      --compare-ik-off "$off" "${extra[@]}" \
      --expected-end-x "$end_x" --expected-end-z "$end_z" \
      --expected-heading "$heading" --require-multilevel
    printf '%s\tPASS\n' "$stem" >> "$quality"
  done
done
test "$(wc -l < "$quality")" -eq 30
test "$(awk -F '\t' '$2 == "PASS" {n++} END {print n+0}' "$quality")" -eq 30

/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  "$root/gate-l/stairs-standard__ascent-landing-descent__positive-x-on.csv" \
  --gate-l2-pair \
  "$root/gate-l/stairs-standard__ascent-landing-descent__negative-x-on.csv"

for ik in 0 1; do
  release="$root/gate-l/landing-exit-stress-${ik}.csv"
  trace_runtime="$root/gate-l/trace-runtime-landing-exit-stress-${ik}.csv"
  trace="$root/gate-l/landing-exit-stress-${ik}.tsv"
  DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene stairs-standard --test-mode route \
    --test-route landing-side-exit-stress --test-frames 800 \
    --test-heading positive-x --terrain-weight 4 \
    --log "$release"
  DISPLAY=:1 MM_IK="$ik" MM_CANDIDATE_TRACE="$trace" \
    "$root/controller-trace" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene stairs-standard --test-mode route \
    --test-route landing-side-exit-stress --test-frames 800 \
    --test-heading positive-x --terrain-weight 4 \
    --log "$trace_runtime"
  cmp "$release" "$trace_runtime"
  DISPLAY=:1 MM_IK="$ik" "$base/controller" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene stairs-standard --test-mode route \
    --test-route landing-side-exit-stress --test-frames 800 \
    --test-heading positive-x --terrain-weight 4 \
    --log "$root/gate-l/frontier-landing-exit-stress-${ik}.csv"
  compare_frontier \
    "$release" \
    "$root/gate-l/frontier-landing-exit-stress-${ik}.csv" \
    "$trace"
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  "$root/gate-l/landing-exit-stress-1.csv" --gate-l2-exit-stress \
  --compare-ik-off "$root/gate-l/landing-exit-stress-0.csv"

for ik in 0 1; do
  release="$root/gate-l/tangent-${ik}.csv"
  trace_runtime="$root/gate-l/trace-runtime-tangent-${ik}.csv"
  trace="$root/gate-l/tangent-${ik}.tsv"
  DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene mixed-multilevel --test-mode route \
    --test-route tangent-level-boundary --test-frames 800 \
    --test-heading positive-x --terrain-weight 4 \
    --log "$release"
  DISPLAY=:1 MM_IK="$ik" MM_CANDIDATE_TRACE="$trace" \
    "$root/controller-trace" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene mixed-multilevel --test-mode route \
    --test-route tangent-level-boundary --test-frames 800 \
    --test-heading positive-x --terrain-weight 4 \
    --log "$trace_runtime"
  cmp "$release" "$trace_runtime"
  DISPLAY=:1 MM_IK="$ik" "$base/controller" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene mixed-multilevel --test-mode route \
    --test-route tangent-level-boundary --test-frames 800 \
    --test-heading positive-x --terrain-weight 4 \
    --log "$root/gate-l/frontier-tangent-${ik}.csv"
  compare_frontier \
    "$release" \
    "$root/gate-l/frontier-tangent-${ik}.csv" \
    "$trace"
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py "$root/gate-l/tangent-1.csv" --gate-l \
  --compare-ik-off "$root/gate-l/tangent-0.csv" \
  --expected-end-x 0.62 --expected-end-z 6.0 \
  --expected-heading positive-x --require-multilevel

flat_cases=(
  flat-positive-z:forward:forward
  flat-positive-z:backward:backward
  flat-positive-z:positive-x:left
  flat-positive-z:negative-x:right
  flat-positive-x:positive-x:forward
  flat-positive-x:negative-x:backward
  flat-positive-x:forward:right
  flat-positive-x:backward:left
)
for specification in "${flat_cases[@]}"; do
  IFS=: read -r route heading relative <<<"$specification"
  stem="${route}__${heading}__${relative}"
  final="$root/gate-l/flat-${stem}.csv"
  baseline="$base/current-frontier/flat-${stem}.csv"
  trace="$root/gate-l/flat-${stem}.tsv"
  trace_runtime="$root/gate-l/trace-runtime-flat-${stem}.csv"
  DISPLAY=:1 MM_IK=0 "$root/controller-release" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene stairs-shallow --test-mode route --test-route "$route" \
    --test-frames 100 --test-heading "$heading" --terrain-weight 4 \
    --log "$final"
  DISPLAY=:1 MM_IK=0 MM_CANDIDATE_TRACE="$trace" \
    "$root/controller-trace" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene stairs-shallow --test-mode route --test-route "$route" \
    --test-frames 100 --test-heading "$heading" --terrain-weight 4 \
    --log "$trace_runtime"
  cmp "$final" "$trace_runtime"
  check_flat_l4_contract "$final" "$route" "$heading" "$relative"
  compare_frontier "$final" "$baseline" "$trace"
done
```

Expected: every physical Gate L, L2, exit, tangent, and flat checker consumes only no-seam release CSVs. For both modes of all 30 cells, both exit modes, both tangent modes, and all eight flat cases, the separately run trace binary emits a byte-identical production CSV before its TSV is trusted; therefore no test-only column or seam-only behavior enters certification. All 30 paired full Gate L physical cells pass; both lateral directions pass the L2 relation; exit stress retains its allowed branch; tangent traversal passes; and all eight flat cases reproduce the complete Gate L4 property contract. Every candidate trace proves exact same-build accelerated/exhaustive equality. Each release/current-frontier pair is equal over all `GATE_L_ORACLE_INVARIANTS` rows when recovery is absent, or over the exact prefix before the trace-authenticated first recovery frame otherwise. The five forward current-frontier runs also reproduce their hashed 32-frame captures. No gate is weakened or replaced with safety-only, and no stale pre-frontier behavior path is consumed.

- [ ] **Step 6: Audit evidence, repository scope, commits, and the untouched production surface**

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
sha256sum -c /tmp/g1-bounded-candidate-cert/base/immutable-files.before.sha256
sha256sum -c \
  /tmp/g1-bounded-candidate-cert/base/current-frontier/SHA256SUMS
(cd /tmp/g1-footprint-prechange-oracle && sha256sum -c SHA256SUMS)
test "$(sha256sum docs/superpowers/specs/2026-07-16-g1-bounded-candidate-certification-design.md | cut -d' ' -f1)" = \
  10c21cb51a351b9cc2f4c1e03d58472e485bba36d0449967de4e08e33a7ec6dd
git diff --check
git status --short > "$root/status.after"
cmp /tmp/g1-bounded-candidate-cert/base/status.before \
  "$root/status.after"
git log --oneline --decorate \
  "$(cat /tmp/g1-bounded-candidate-cert/base/execution-base.txt)..HEAD"
git diff --name-only \
  "$(cat /tmp/g1-bounded-candidate-cert/base/execution-base.txt)..HEAD" \
  | sort > "$root/changed-paths.txt"
expected="$root/expected-paths.txt"
printf '%s\n' \
  controller.cpp \
  g1_candidate_certification_trace.h \
  g1_candidate_recovery.cpp \
  g1_candidate_recovery.h \
  g1_frame_transaction.h \
  scene_switch.h \
  tests/cpp/compile_g1_candidate_recovery_production.cpp \
  tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp \
  tests/cpp/test_g1_candidate_recovery.cpp \
  tests/cpp/test_g1_controller_logging.cpp \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp \
  tests/cpp/test_scene_switch.cpp \
  | sort > "$expected"
cmp "$expected" "$root/changed-paths.txt"
test "$(git log --format=%s \
  "$(cat /tmp/g1-bounded-candidate-cert/base/execution-base.txt)..HEAD" \
  | rg -c '^(feat: add strict bounded recovery provider|feat: coordinate bounded dual candidate certification|feat: publish bounded candidate outcomes atomically|feat: integrate bounded candidate certification|test: trace bounded candidate certification)$')" -eq 5
```

Review the 32-frame trace, exact 800-row release result, three performance summaries, 9 normal/4 stress/2 blocked Gate reports, 30 Gate L verdicts, trace-bounded current-frontier comparison output, L2/exit/tangent/flat reports, native/Python test logs, recorded compiler switches, binary hashes, and five task commits. Resolve any Critical or Important code-review finding test-first and rerun the affected focused matrix plus every live gate whose ownership could change.

Expected: evidence hashes verify; changed source paths are exactly the thirteen planned paths; exactly five task commits have the planned subjects; the final worktree status exactly matches the recorded pre-implementation unrelated status; the unchanged compile-negative runner has executed in the later verification matrices; schema, checker, spec, terrain, current-frontier captures/controller, and asset-only prechange corpus are unchanged; `git diff --check` is silent; no repository evidence artifact was created; and the running visualizer was neither discovered nor touched. There is deliberately no Task 6 commit.

---
