# G1 Inertialized Candidate Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unchanged G1 SONIC Stage A matcher reject an otherwise raw-safe database candidate when its exact next state-dependent inertialized joint pose violates the authenticated joint contract, then select the lowest-cost dynamically safe candidate.

**Architecture:** Add a generic tri-state validation hook at the branch-and-bound leaf, while keeping the prior frame as an explicit and independent 20-frame neighborhood center. The G1 runtime previews ordinary continuation and transition candidates with one shared transition-plus-inertialization primitive; the real MM adapter binds those poses to the existing structured joint projector and publishes strict audit diagnostics. The hard live projection, search costs, raw feasibility masks, controller, assets, policy, and seven Stage A gates remain unchanged.

**Tech Stack:** C++17 header-only motion matcher/runtime, bundled C++ MM server, JSON Schema draft-07, Python 3.10 dataclasses and `unittest`, GNU Make, ASan/UBSan, ONNX Runtime G1 controller, immutable Stage A artifact pipeline.

## Global Constraints

- Implementation base is reviewed commit `e8e7d8d6388579439709be28d213c090732f65c0`; the approved design commit is `6fd0de89bd7d028f4492a4ace7aff226c5e692f4`.
- Do not clip, clamp, sanitize, blend, interpolate, or otherwise repair an invalid emitted joint value.
- Do not widen or replace the registered G1 joint limits.
- Do not change the database, feasibility sidecars, GEAR checkout, policy, commands, scenes, feature weights, transition cost, search timing, inertialization halflife, 20-frame neighborhood, Stage A thresholds, or seven gate definitions.
- Preserve the feature query, range clamp, branch-and-bound arithmetic, strict-less-than tie behavior, and database iteration order.
- Keep the raw `joint_feasibility` hello identity byte-for-byte stable.
- Only a well-formed `SonicJointProjectionLimit` diagnostic is a recoverable candidate rejection; every other preview failure is a fatal integration error.
- Keep `sonic_project_pose()` as the authoritative live boundary and never retry after that live boundary fails.
- Candidate preview must not mutate live runtime state, database state, output result, timers, or diagnostics before an accepted choice is committed.
- Failed runtime steps and failed protocol generation remain transactional.
- The exact dynamic exhaustion message is `no inertialized-joint-safe database candidate`; the exact existing zero-dynamic-rejection message remains `no joint-limit-safe database candidate`.
- Use the detached Reliable Claude release at exact SHA `11f44df060fa011db504199917beb3e8bb5200dd` with plugin `0.1.1+codex.20260716220019`; do not mutate a runtime serving a nonterminal job.
- Do not push, merge, publish, tag, or activate releases. Preserve all prior releases and scientific outputs for rollback.
- Run every scientific trial in a fresh output root, and accept a Stage A pass only when all seven unchanged gates pass and the immutable inventory verifies.

---

### Task 1: Generic Cost-Ordered Validated Database Search

**Files:**
- Modify: `database.h:796-1002`
- Test: `tests/cpp/test_terrain_database.cpp`

**Interfaces:**
- Consumes: existing normalized query, transition cost, candidate mask, range-end exclusion, surrounding-frame exclusion, and strict branch-and-bound cost comparison.
- Produces:

```cpp
enum database_candidate_verdict
{
    DatabaseCandidateAccept,
    DatabaseCandidateReject,
    DatabaseCandidateFatal
};

enum database_search_status
{
    DatabaseSearchComplete,
    DatabaseSearchInvalidInput,
    DatabaseSearchCandidateFatal
};

struct database_candidate_validator
{
    void* context = nullptr;
    database_candidate_verdict (*evaluate)(void*, int) = nullptr;
};

constexpr int DatabaseUseIncumbentNeighborhood = -2;

database_search_status database_search_validated(
    int& best_index,
    float& best_cost,
    const database& db,
    const slice1d<float> query,
    float transition_cost,
    int ignore_range_end,
    int ignore_surrounding,
    const unsigned char* candidate_mask,
    int candidate_mask_count,
    int neighborhood_center,
    const database_candidate_validator* validator);
```

The existing `motion_matching_search()` and `database_search()` signatures remain source-compatible wrappers with exactly their previous behavior. A `nullptr` validator accepts every eligible leaf. `DatabaseUseIncumbentNeighborhood` tells the wrapper to preserve the historical use of the incoming incumbent as the neighborhood center; the G1 validated call passes the prior frame explicitly.

- [ ] **Step 1: Write the failing search-order and neighborhood tests**

Add named tests to `tests/cpp/test_terrain_database.cpp` using the file's existing synthetic database/query helpers:

```cpp
struct candidate_script
{
    int calls[16] = {};
    int count = 0;
    int rejected = -1;
    int fatal = -1;
};

static database_candidate_verdict scripted_candidate(void* raw, int frame)
{
    candidate_script& script = *static_cast<candidate_script*>(raw);
    script.calls[script.count++] = frame;
    if (frame == script.fatal) return DatabaseCandidateFatal;
    if (frame == script.rejected) return DatabaseCandidateReject;
    return DatabaseCandidateAccept;
}
```

Cover all of these assertions with deterministic feature costs:

```cpp
// Cheapest frame 40 rejects, so frame 70 becomes the accepted minimum.
assert(status == DatabaseSearchComplete);
assert(best_index == 70);
assert(best_cost == accepted_cost_70);
assert(script.calls[0] == 40);

// A rejected leaf never lowers the pruning bound: a later cost between the
// rejected cost and the incumbent cost is still evaluated and accepted.
assert(script.calls[1] == 70);

// Fatal validation terminates without publishing a winner.
assert(status == DatabaseSearchCandidateFatal);
assert(best_index == -1);
assert(best_cost == FLT_MAX);

// An unavailable incumbent does not erase the explicit prior-frame center.
// Frames [prior-20, prior+20] are never presented to the validator.
assert(!contains(script.calls, script.count, prior_frame + 1));

// Equal-cost accepted candidates retain the earlier database frame.
assert(best_index == first_equal_cost_frame);
```

Also call the existing unvalidated overload on the same fixture and assert its prior selected frame and cost remain unchanged.

- [ ] **Step 2: Run the focused test and witness RED**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database
```

Expected: compilation fails because `database_candidate_verdict`, `database_candidate_validator`, and `database_search_validated()` do not exist. Record the exact compiler output in the task log before changing production code.

- [ ] **Step 3: Implement one shared validated branch-and-bound core**

Refactor the current search body, without changing its normalization or cost expressions, so leaf promotion is exactly:

```cpp
if (candidate_cost < best_cost)
{
    const database_candidate_verdict verdict = validator == nullptr
        ? DatabaseCandidateAccept
        : validator->evaluate(validator->context, candidate_index);
    if (verdict == DatabaseCandidateFatal)
    {
        best_index = -1;
        best_cost = FLT_MAX;
        return DatabaseSearchCandidateFatal;
    }
    if (verdict == DatabaseCandidateAccept)
    {
        best_index = candidate_index;
        best_cost = candidate_cost;
    }
}
```

Validate mask length, validator function pointer, and neighborhood center before search. Return `DatabaseSearchInvalidInput` with `best_index = -1` and `best_cost = FLT_MAX` on malformed input. Keep the incumbent's precomputed cost only when it exists and is prevalidated; use `neighborhood_center` solely for surrounding exclusion. Have both old wrappers call this core with `validator == nullptr` and their historical neighborhood semantics.

- [ ] **Step 4: Run the focused test and witness GREEN**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database
/tmp/test_terrain_database
```

Expected: exit `0` with no warning output.

- [ ] **Step 5: Commit the independently reviewed search primitive**

```bash
git add database.h tests/cpp/test_terrain_database.cpp
git commit -m "feat: validate motion match candidates in cost order"
```

### Task 2: Side-Effect-Free G1 Pose Preview and Runtime Scheduling

**Files:**
- Modify: `sonic/cpp/g1_runtime.h:17-65,180-325,875-987`
- Test: `tests/cpp/test_g1_runtime.cpp`

**Interfaces:**
- Consumes: Task 1's `database_search_validated()`, current runtime pose/offset state, database source pose arrays, `database_trajectory_index_clamp()`, and the unchanged transition/update formulas.
- Produces:

```cpp
enum g1_runtime_joint_preview_verdict
{
    G1RuntimeJointPreviewAccept,
    G1RuntimeJointPreviewRejectLimit,
    G1RuntimeJointPreviewFatal
};

struct g1_runtime_candidate_preview_diagnostic
{
    int candidate_preview_count = 0;
    int candidate_limit_rejection_count = 0;
    int first_rejected_database_frame = -1;
    int first_rejected_joint_index = -1;
    float first_rejected_joint_position = 0.0f;
};

struct g1_runtime_joint_preview_validator
{
    void* context = nullptr;
    g1_runtime_joint_preview_verdict (*evaluate)(
        void*,
        int selected_database_frame,
        int emitted_database_frame,
        slice1d<quat> local_rotations,
        slice1d<vec3> local_angular_velocities,
        int& rejected_joint_index,
        float& rejected_joint_position,
        char* error,
        int capacity) = nullptr;
};
```

Add `g1_runtime_candidate_preview_diagnostic candidate_preview;` to `g1_runtime_step_result`. Add a validator-aware `g1_runtime_step()` overload; retain the existing overload and route it through an accept-all/disabled preview path that preserves old non-Sonic behavior.

- [ ] **Step 1: Write RED tests for shared pose parity and transactional preview**

Add focused fixtures that snapshot every live runtime array and scalar before preview. Exercise both no-transition continuation and a transition. Assert:

```cpp
assert(std::memcmp(
    preview_rotations.data,
    live_pre_support_rotations.data,
    sizeof(quat) * preview_rotations.size) == 0);
assert(std::memcmp(
    preview_angular_velocities.data,
    live_pre_support_angular_velocities.data,
    sizeof(vec3) * preview_angular_velocities.size) == 0);
assert(runtime_state_equal(before, after_preview));
```

Add a scripted validator that rejects the ordinary successor, accepts a later search candidate, and records each `(selected, emitted)` pair. Assert ordinary rejection forces search even with `matching_enabled == false`, the accepted result is selected, counts are `2` and `1`, and the first rejection fields identify the ordinary selected frame and scripted joint/value.

Add separate cases proving:

```cpp
// A valid ordinary preview remains the incumbent during scheduled search.
assert(result.selected_database_frame == prior_frame);

// A fatal ordinary preview aborts and preserves all state/result bytes.
assert(!ok);
assert(runtime_state_equal(before, state));
assert(step_result_equal(result_before, result));

// All dynamic candidates rejected uses the new exact message.
assert(std::strcmp(error, "no inertialized-joint-safe database candidate") == 0);

// Search exhaustion with zero dynamic rejection retains the old message.
assert(std::strcmp(error, "no joint-limit-safe database candidate") == 0);
```

- [ ] **Step 2: Run the runtime test and witness RED**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime
```

Expected: compilation fails on the absent preview types, result field, and validator-aware runtime overload.

- [ ] **Step 3: Factor and use one transition-plus-update primitive**

Introduce an internal pose-workspace view holding pose, offsets, and transition root transforms. Implement one `g1_runtime_pose_advance()` that conditionally calls the existing `inertialize_pose_transition()` and always calls the existing `inertialize_pose_update()` on the range-clamped emitted successor. Its live call receives the cloned `next` arrays; its preview call receives preallocated scratch arrays reset from the same read-only baseline. Do not duplicate either formula and do not allocate inside the per-candidate validator call.

The scheduling sequence must be exactly:

```cpp
preview ordinary successor;
if (raw ordinary successor is unsafe || ordinary preview rejects ||
    ordinary successor is end-of-range)
    force search;
else
    cache ordinary preview as a prevalidated incumbent;

database_search_validated(
    best_index, best_cost, db, query, transition_cost, 20, 20,
    feasibility == nullptr ? nullptr : feasibility->search_safe,
    feasibility == nullptr ? 0 : feasibility->count,
    prior_index,
    &database_validator);

apply g1_runtime_pose_advance() once to cloned live state for accepted choice;
```

Increment preview count only when the external validator is invoked. Cache and reuse the ordinary verdict if the prior frame becomes the incumbent. On the first limit rejection, copy the exact selected frame, structured joint index, and position; do not overwrite it on later rejects. Convert Task 1 fatal status and every validator fatal verdict into a transactional integration failure using the callback's rendered error. Give the dynamic exhaustion message priority whenever rejection count is positive.

- [ ] **Step 4: Run the runtime test and witness GREEN**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime
/tmp/test_g1_runtime
```

Expected: exit `0`, exact message assertions pass, preview/live pose bits match, and all state snapshots remain unchanged on rejected/fatal paths.

- [ ] **Step 5: Commit the runtime preview boundary**

```bash
git add sonic/cpp/g1_runtime.h tests/cpp/test_g1_runtime.cpp
git commit -m "feat: preview inertialized G1 match candidates"
```

### Task 3: Bind Real Candidate Preview to the Structured G1 Joint Contract

**Files:**
- Modify: `sonic/cpp/mm_chunk_server.cpp:178-665`
- Test: `tests/python/test_mm_chunk_server.py`

**Interfaces:**
- Consumes: Task 2's validator callback, the real adapter's loaded `contract_`, and `sonic_project_joint_state()` from `sonic/cpp/g1_joint_projection.h`.
- Produces: a private static callback on `mm_real_adapter` that returns `G1RuntimeJointPreviewAccept`, `G1RuntimeJointPreviewRejectLimit`, or `G1RuntimeJointPreviewFatal` from structured projection state; fake-server operation continues through the validator-disabled runtime path.

- [ ] **Step 1: Write RED real-adapter validator classification tests**

Extend the server test harness to compile a focused C++ probe that can call the real adapter's validator through its runtime boundary. Use the existing authenticated contract fixture and constructed local pose arrays to cover:

```cpp
// Valid projection.
assert(verdict == G1RuntimeJointPreviewAccept);
assert(rejected_joint_index == -1);
assert(rejected_joint_position == 0.0f);

// Genuine registered limit failure.
assert(verdict == G1RuntimeJointPreviewRejectLimit);
assert(rejected_joint_index == expected_row);
assert(rejected_joint_position < contract[expected_row].lower ||
       rejected_joint_position > contract[expected_row].upper);

// Every other failure is fatal.
assert(shape_verdict == G1RuntimeJointPreviewFatal);
assert(singular_verdict == G1RuntimeJointPreviewFatal);
assert(residual_verdict == G1RuntimeJointPreviewFatal);
assert(input_verdict == G1RuntimeJointPreviewFatal);
assert(velocity_verdict == G1RuntimeJointPreviewFatal);
assert(malformed_limit_verdict == G1RuntimeJointPreviewFatal);
```

The malformed-limit cases must alter a test-local diagnostic so that its row is out of range, a numeric member is non-finite, bounds are unordered, or the position lies inside the reported interval. Assert each retains a nonempty structured error. Also assert the fake endpoint still generates a chunk without loading a real contract.

- [ ] **Step 2: Run the focused server test and witness RED**

Run:

```bash
env PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_mm_chunk_server
```

Expected: the new real validator probe fails to compile or its classification assertions fail because the adapter has not bound a preview validator.

- [ ] **Step 3: Implement fail-closed structured classification and bind it**

Add a private callback whose projection call is structurally equivalent to:

```cpp
float positions[SonicG1JointCount] = {};
float velocities[SonicG1JointCount] = {};
float residuals[SonicG1JointCount] = {};
sonic_joint_projection_diagnostic projection;
const bool projected = sonic_project_joint_state(
    positions, velocities, residuals, projection, adapter.contract_,
    local_rotations, local_angular_velocities, error, capacity);
if (projected)
{
    if (projection.failure != SonicJointProjectionValid)
        return G1RuntimeJointPreviewFatal;
    return G1RuntimeJointPreviewAccept;
}
const int row = projection.row;
const bool well_formed_limit =
    projection.failure == SonicJointProjectionLimit &&
    row >= 0 && row < SonicG1JointCount &&
    std::isfinite(projection.position) &&
    std::isfinite(projection.lower) &&
    std::isfinite(projection.upper) &&
    projection.lower < projection.upper &&
    (projection.position < projection.lower ||
     projection.position > projection.upper) &&
    projection.lower == adapter.contract_[row].lower &&
    projection.upper == adapter.contract_[row].upper;
if (!well_formed_limit)
    return G1RuntimeJointPreviewFatal;
rejected_joint_index = row;
rejected_joint_position = projection.position;
return G1RuntimeJointPreviewRejectLimit;
```

If a malformed result did not render an error, render a deterministic integration detail before returning fatal. In `mm_real_adapter::advance()`, construct the validator with `context = this`, pass it to the Task 2 overload, and leave `observe_boundary()` and its unconditional `sonic_project_pose()` call untouched. Do not bind this callback in `mm_fake_adapter`.

- [ ] **Step 4: Run focused projection and server tests and witness GREEN**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_projection.cpp -o /tmp/test_g1_joint_projection
/tmp/test_g1_joint_projection
env PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_mm_chunk_server
```

Expected: both commands exit `0`; real limit failures reject candidates, malformed/non-limit failures abort, and fake generation remains green.

- [ ] **Step 5: Commit the real contract binding**

```bash
git add sonic/cpp/mm_chunk_server.cpp tests/python/test_mm_chunk_server.py
git commit -m "feat: validate G1 previews against joint contract"
```

### Task 4: Publish Strict Per-Step Candidate Audit Evidence

**Files:**
- Modify: `sonic/cpp/mm_chunk_protocol.h:103-145`
- Modify: `sonic/cpp/mm_chunk_server.cpp:500-560,1090-1240`
- Modify: `sonic/schemas/mm_chunk_v1.schema.json`
- Modify: `sonic/python/mm_sonic/schema.py:39-65,524-700`
- Test: `tests/cpp/test_mm_chunk_protocol.cpp`
- Test: `tests/python/test_mm_chunk_server.py`
- Test: `tests/python/test_sonic_schema.py`

**Interfaces:**
- Consumes: Task 2's `g1_runtime_candidate_preview_diagnostic` and the already loaded contract used by Task 3.
- Produces five required ten-element `mm-chunk/v1` arrays on C++ JSON output and immutable Python `SourceChunk` values:

```text
candidate_preview_count
candidate_limit_rejection_count
first_rejected_database_frame
first_rejected_joint_index
first_rejected_joint_position
```

- [ ] **Step 1: Write RED C++ protocol and JSON-writer tests**

Extend `mm_chunk_step_diagnostic` equality and the protocol fixtures. Assert successful real and fake chunks serialize all five exact keys and ten values. Cover zero and positive rejection invariants:

```cpp
assert(step.candidate_preview_count >= 0);
assert(step.candidate_limit_rejection_count >= 0);
assert(step.candidate_limit_rejection_count <= step.candidate_preview_count);

// No rejection sentinel.
assert(step.first_rejected_database_frame == -1);
assert(step.first_rejected_joint_index == -1);
assert(step.first_rejected_joint_position == 0.0f);

// Positive rejection evidence.
assert(step.first_rejected_database_frame >= 0);
assert(step.first_rejected_joint_index >= 0);
assert(step.first_rejected_joint_index < SonicG1JointCount);
assert(std::isfinite(step.first_rejected_joint_position));
assert(step.first_rejected_joint_position < contract[row].lower ||
       step.first_rejected_joint_position > contract[row].upper);
assert(step.selected_database_frame != step.first_rejected_database_frame);
```

Add transactional writer cases for negative counts, rejections exceeding previews, wrong zero sentinels, positive rejection with bad indices/non-finite or in-range position, and selected frame equal to first rejected frame. Each must fail without publishing partial JSON.

- [ ] **Step 2: Write RED strict schema and Python parser tests**

Update valid source-chunk fixtures in `tests/python/test_sonic_schema.py`, then add one-field mutation tests for missing/extra keys, wrong array shapes, booleans in integer arrays, negative counts, count inversion, inconsistent sentinels, out-of-range indices, non-finite rejected position, contract-relative in-range position, and selected/rejected frame equality. The valid positive case must assert immutable owned arrays:

```python
assert chunk.candidate_preview_count.shape == (10,)
assert chunk.candidate_preview_count.dtype.kind in "iu"
assert not chunk.candidate_preview_count.flags.writeable
assert chunk.candidate_limit_rejection_count[2] == 1
assert chunk.first_rejected_database_frame[2] == 927
assert chunk.first_rejected_joint_index[2] == expected_left_ankle_row
assert np.float32(chunk.first_rejected_joint_position[2]).view(np.uint32) \
    == np.float32(-0.27224052).view(np.uint32)
```

- [ ] **Step 3: Run protocol/schema tests and witness RED**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_mm_chunk_protocol.cpp -o /tmp/test_mm_chunk_protocol
env PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_sonic_schema tests.python.test_mm_chunk_server
```

Expected: C++ compilation and strict Python parsing fail on the absent diagnostic members and required schema keys.

- [ ] **Step 4: Implement the atomic strict protocol extension**

Add the five scalar members to `mm_chunk_step_diagnostic`, copy Task 2 diagnostics in the real adapter, and set the fake adapter to `0, 0, -1, -1, 0.0f`. Add all fields to equality, C++ contract validation, and the JSON writer. Add required arrays with exact lengths/types to `mm_chunk_v1.schema.json`.

In Python, add the five fields to `SourceChunk`, the exact-key set, and parsing. Use `_integer_array()`/`_float_array()` for owned immutable arrays, then enforce per step:

```python
if previews < 0 or rejections < 0 or rejections > previews:
    raise ContractError("candidate preview counts are inconsistent")
if rejections == 0:
    if frame != -1 or joint != -1 or position != 0.0:
        raise ContractError("candidate rejection sentinels are inconsistent")
else:
    if frame < 0 or joint < 0 or joint >= _JOINT_COUNT:
        raise ContractError("candidate rejection indices are invalid")
    lower = contract.rows[joint].lower
    upper = contract.rows[joint].upper
    if not math.isfinite(position) or lower <= position <= upper:
        raise ContractError("candidate rejected position is not outside its contract")
    if selected_frame == frame:
        raise ContractError("selected frame equals first rejected frame")
```

Use the contract's existing source-row order/accessor rather than introducing a second limit table. Keep hello and `joint_feasibility` JSON unchanged.

- [ ] **Step 5: Run protocol/schema tests and witness GREEN**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_mm_chunk_protocol.cpp -o /tmp/test_mm_chunk_protocol
/tmp/test_mm_chunk_protocol
env PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_sonic_schema tests.python.test_mm_chunk_server
```

Expected: all focused tests pass; old/new endpoint mixtures fail strict key parsing; fake diagnostics use exact sentinels; real positive evidence is contract-relative.

- [ ] **Step 6: Commit the bundled wire-contract change**

```bash
git add sonic/cpp/mm_chunk_protocol.h sonic/cpp/mm_chunk_server.cpp \
  sonic/schemas/mm_chunk_v1.schema.json sonic/python/mm_sonic/schema.py \
  tests/cpp/test_mm_chunk_protocol.cpp tests/python/test_mm_chunk_server.py \
  tests/python/test_sonic_schema.py
git commit -m "feat: publish candidate preview diagnostics"
```

### Task 5: Classify Dynamic Exhaustion and Prove the Recorded Step-52 Boundary

**Files:**
- Modify: `sonic/python/mm_sonic/cli.py:149-204,4190-4210`
- Test: `tests/python/test_sonic_cli.py:2780-2925`
- Create (controller-owned, outside repository): `/home/ubuntu/reliable-claude-projects/motion-matching/evaluators/evaluate_g1_candidate_preview.py`
- Create (controller-owned, outside repository): `/home/ubuntu/reliable-claude-projects/motion-matching/evaluators/g1_candidate_preview.cpp`

**Interfaces:**
- Consumes: the exact runtime error from Task 2, the real adapter from Task 3, the diagnostic wire fields from Task 4, immutable G1 database/sidecars/contract assets, and the prior localized step-50 state.
- Produces: exact scientific ownership for remote dynamic exhaustion and a protected warning-strict executable replay whose single success line binds source and asset hashes.

- [ ] **Step 1: Write RED verdict ownership tests**

Add the second exact constant and predicate:

```python
_REMOTE_NO_INERTIALIZED_SAFE_CANDIDATE_MESSAGE = (
    "no inertialized-joint-safe database candidate"
)

def _is_remote_no_inertialized_safe_candidate(error: BaseException) -> bool:
    return (
        isinstance(error, _RemoteMMError)
        and error.code == "generation_failed"
        and error.message == _REMOTE_NO_INERTIALIZED_SAFE_CANDIDATE_MESSAGE
    )
```

Before implementing it, add table-driven tests showing the exact remote code/message is scientific while these remain integration-owned: wrong code, prefix, suffix, case change, underscore/hyphen lookalike, local stderr only, non-remote exception, and the existing raw-safe message with altered text. Retain all existing live joint-limit and raw-candidate ownership tests.

- [ ] **Step 2: Run the verdict test and witness RED**

Run:

```bash
env PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_sonic_cli
```

Expected: the new exact message is not recognized as a scientific remote generation failure.

- [ ] **Step 3: Implement only the exact verdict extension**

Add `_is_remote_no_inertialized_safe_candidate()` beside the existing two remote predicates and include it in the same `generation_failed` scientific branch. Do not broaden regexes, accept stderr, or alter ownership for final live projection failures.

- [ ] **Step 4: Run the verdict test and witness GREEN**

Run the Step 2 command again.

Expected: all CLI tests pass, including wrong-code and lookalike integration cases.

- [ ] **Step 5: Write the protected real-artifact replay before running it**

Create the controller-owned C++ probe and Python compiler/runner using `apply_patch`, not repository edits. The Python evaluator must compile with `/usr/bin/g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic`, a temporary output directory, explicit `--repo`, and timeouts. The C++ probe must load the immutable published database, terrain/support sidecars, G1 contract, raw/search masks, and exact known-good request prefix; it must run through the recorded state sequence rather than constructing a synthetic offset.

At the 50/51/52 boundary, assert all of the following:

```cpp
assert(step50.query_database_frame == 50);
assert(step51.selected_database_frame == 865);
assert(step51_emitted_database_frame == 866);
assert(step52.candidate_preview.candidate_limit_rejection_count > 0);
assert(step52.candidate_preview.first_rejected_database_frame == 0);
assert(step52.candidate_preview.first_rejected_joint_index == left_ankle_roll_row);
assert(replay_validator.saw_rejection(927, left_ankle_roll_row));
assert(float_bits(replay_validator.rejected_position(927)) ==
       float_bits(-0.27224052f));
assert(step52.selected_database_frame != 927);
assert(final_projection_succeeds);
assert(accepted_preview_rotation_and_velocity_bits_equal_live);
```

The evaluator must print exactly one replay success record with the selected replacement frame, emitted frame, rejected position bits, live projected value, database SHA-256, raw/search mask SHA-256, contract SHA-256, runtime SHA-256, and probe SHA-256. Any missing asset, hash mismatch, warning, nonzero exit, extra output, or assertion fails closed.

- [ ] **Step 6: Run the protected replay and witness the real boundary**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  /home/ubuntu/reliable-claude-projects/motion-matching/evaluators/evaluate_g1_candidate_preview.py \
  --repo /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline
```

Expected: exit `0`; candidate `927` is rejected at the exact recorded float bits, the next cost-ordered accepted candidate becomes live, and its hard live projection is joint-safe. If a different immutable asset hash is observed, stop the replay as an integration failure rather than updating an expected hash silently.

- [ ] **Step 7: Commit repository-owned verdict behavior**

```bash
git add sonic/python/mm_sonic/cli.py tests/python/test_sonic_cli.py
git commit -m "feat: classify inertialized candidate exhaustion"
```

Record the controller-owned evaluator SHA-256 values in the final result document; do not add the evaluator files to the product repository.

### Task 6: Full Qualification, Immutable Stage A Trial, and Research Handoff

**Files:**
- Create: `docs/superpowers/results/2026-07-17-g1-inertialized-candidate-preview.md`
- Modify: `/home/ubuntu/reliable-claude-projects/motion-matching/FRESH_SESSION_HANDOFF.md`
- Verify without modifying: all product files, published assets, GEAR checkout, prior run roots, Reliable Claude release/rollback roots.

**Interfaces:**
- Consumes: Tasks 1-5, the protected replay, supported Python 3.10 environment, current real-server artifacts, immutable GPU-0 Stage A command, and the existing seven-gate evaluator.
- Produces: warning-strict/sanitizer/full-suite evidence, two stable hello identities, one fresh immutable Stage A run, verified inventory hashes, a truthful result document, and a fresh-session handoff.

- [ ] **Step 1: Run all warning-strict focused C++ binaries**

Run each compile followed by its executable:

```bash
for name in test_terrain_database test_g1_runtime test_g1_joint_projection \
  test_g1_joint_feasibility test_mm_chunk_protocol
do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/${name}.cpp" -o "/tmp/${name}"
  "/tmp/${name}"
done
```

Expected: all five compile warning-free and exit `0`.

- [ ] **Step 2: Run runtime sanitizers**

Run:

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime_san
/tmp/test_g1_runtime_san
```

Expected: exit `0` with no ASan/UBSan diagnostic.

- [ ] **Step 3: Rebuild public C++ endpoints from clean sources**

Run:

```bash
make -C sonic/cpp clean
make -C sonic/cpp mm_chunk_server route_schedule_cli g1_project_pose_cli
```

Expected: all three targets build successfully; record their SHA-256 values and build commit identity.

- [ ] **Step 4: Run the real server and complete supported Python catalog**

Set the projection CLI to the just-built absolute path and run:

```bash
env PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  SONIC_PROJECT_CLI=/home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/build/g1_project_pose_cli \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_mm_chunk_server \
  tests.python.test_sonic_artifacts \
  tests.python.test_sonic_cli \
  tests.python.test_sonic_commands \
  tests.python.test_sonic_coordinator \
  tests.python.test_sonic_external \
  tests.python.test_sonic_gated_sim \
  tests.python.test_sonic_joints \
  tests.python.test_sonic_metrics \
  tests.python.test_sonic_process \
  tests.python.test_sonic_reference \
  tests.python.test_sonic_resample \
  tests.python.test_sonic_runtime_parity \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_schema \
  tests.python.test_sonic_timeline \
  tests.python.test_sonic_timing \
  tests.python.test_sonic_transform \
  tests.python.test_sonic_zmq_v1
```

Expected: the complete supported catalog passes with only its ten established optional/environment skips; any additional skip, warning, hang, or failure is investigated before scientific execution.

- [ ] **Step 4a: Close any exact downstream interval exposed by qualification**

Preserve a terminal scientific run unchanged. If endpoint-safe source data is
rejected by the existing 25-to-50 Hz bridge, pin the first observed endpoints,
velocities, binary32 `dt`, interpolated value, joint row, and limits in a RED
test. Extend the real preview predicate to project the shared live left boundary
once, project each scratch right boundary, and evaluate the exact Hermite
midpoint in double precision. Endpoint/midpoint registered-limit failures are
candidate rejections; malformed shared state or interpolation is fatal. Re-run
the protected real validator before another immutable trial.

The first qualification trial at
`known-good-stream-20260717T132232556131Z-206cc7ad` exercised this branch:
chunk 6 target joint 17 interpolated to `-0.2662098130672067` from legal 25 Hz
endpoints. It remains immutable evidence rather than being overwritten or
reclassified.

- [ ] **Step 5: Re-run protected evaluators and prove immutable identities**

Run the prior certificate, projection, masked-search, verdict, evidence-fence, environment, LFS-boundary, scene-resolution, and new candidate-preview evaluators. Launch the freshly built real server twice in separate processes and capture strict hello JSON. Assert both hello objects are equal and retain:

```text
frame_count=459682
raw_safe_count=458619
raw_unsafe_count=1063
search_safe_count=458274
mask_sha256=cce20d9b5d2dd4ed4013a1d6416f3402e564b013aebdfefc15bd1a468b494e25
```

Also assert GEAR remains clean at `60de0df7`, Reliable Claude source remains clean at `11f44df060fa011db504199917beb3e8bb5200dd`, the rollback release still exists, and all prior scientific inventory files retain their recorded hashes.

- [ ] **Step 6: Commit qualification-ready repository state before Stage A**

Run `git diff --check`, inspect `git status --short`, review every commit since `6fd0de8`, and obtain a fresh code/spec review. Resolve only reviewed in-scope defects and rerun affected tests. Then create the qualification commit if documentation or test-only adjustments remain:

```bash
git add docs/superpowers/results/2026-07-17-g1-inertialized-candidate-preview.md
git commit -m "docs: qualify G1 inertialized candidate preview"
```

Do not amend scientific implementation commits after the immutable Stage A run begins.

- [ ] **Step 7: Run the unchanged GPU-0 Stage A command in a fresh root**

Run exactly:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m mm_sonic.cli stage-a \
  --mode known-good-stream \
  --gear-checkout /tmp/groot-wbc-plan-inspect \
  --policy /tmp/groot-wbc-plan-inspect/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx \
  --observation-config /tmp/groot-wbc-plan-inspect/gear_sonic_deploy/policy/release/observation_config.yaml \
  --source-mjcf /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/runs/stage-a-inertialized-candidate-preview-integrated-20260717
```

Expected success: command exits `0`, gate 4 retains all `601` required reference frames without a joint-contract violation, all seven unchanged gates pass, and final inventory verification returns `True`.

If the run truthfully fails, finalize that unique run unchanged, verify its inventory, localize the next first failing boundary, write the observed scientific result, and design the next single intervention without altering this trial or its thresholds.

- [ ] **Step 8: Finalize evidence and the fresh-session handoff**

Write the result document with RED/GREEN evidence, exact commands and counts, commit chain, compiler/runtime versions, protected evaluator hashes/output, two hello payload hashes, candidate-preview/rejection totals, step-52 replacement selection, all seven gate verdicts, manifest/evidence/inventory SHA-256 values, `verify_run_inventory(...)` result, rollback point, and any next blocker.

Update `/home/ubuntu/reliable-claude-projects/motion-matching/FRESH_SESSION_HANDOFF.md` with the current branch/HEAD, clean/dirty status, Reliable Claude release and rollback roots, immutable assets, final run path, exact first failure or seven-gate pass, and the next authorized action. Use `apply_patch` for both documents, run `git diff --check`, commit repository-owned result changes, and leave the product worktree clean.

- [ ] **Step 9: Obtain final qualification review**

Give a fresh reviewer the approved design, this plan, complete commit range, protected evaluator outputs, test logs, immutable Stage A evidence, and result document. Require explicit classification of Critical/Important/Minor findings. Resolve all Critical and Important findings with witnessed regression tests and rerun the relevant qualification; record any accepted Minor finding without concealing it.
