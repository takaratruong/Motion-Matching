# G1 Joint-Feasibility-Masked Motion Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Prevent the matcher from selecting or advancing into database frames whose projected 29-joint G1 state violates the authenticated joint contract, while retaining the live inertialized projection as the final hard safety gate.

**Architecture:** Build one immutable feasibility certificate immediately after database and joint-contract validation. The certificate classifies every raw database frame through the same projection math used at runtime, derives a successor-safe search mask, binds the mask to a deterministic SHA-256 identity, and threads that mask through ordinary database search and progression. The server publishes the certificate identity in hello; Python validates and cross-binds it before accepting any motion evidence.

**Tech Stack:** C++17 header-only runtime code, existing animation database and SHA-256 helpers, JSONL protocol, Python 3.10 dataclasses and strict schema parsing, unittest, real G1 terrain database.

## Global Constraints

- Base all work on motion-matching commit 05014911828c3b52b38a1651fecf3760ef8f03ee.
- Treat only structured JointProjectionLimit failures as maskable. Shape, contract, input, singularity, residual, velocity, and non-finite failures abort certificate construction.
- Build the certificate once during server initialization. Never scan the full database inside a frame step.
- A searchable frame is safe only when its emitted pose and required successor are raw-safe inside the same animation range; clamp range-end successor exactly as runtime progression does.
- Preserve motion_matching_search and the existing unmasked database_search behavior for callers outside the authenticated G1 server path.
- Preserve sonic_project_pose public behavior, exact legacy error strings, and transactional output.
- Preserve the live inertialized call to sonic_project_pose as a hard gate after selection.
- Return the exact structured server error no joint-limit-safe database candidate only when masked search has no finite candidate.
- Treat that exact generation failure as scientific in Python; treat lookalike text and certificate mismatches as integration failures.
- Do not change protocol version 1 or relax any existing hello key-set checks.

---

### Task 1: Expose structured joint-projection diagnostics without changing legacy behavior

**Files:**
- Modify: sonic/cpp/g1_joint_projection.h
- Modify: tests/cpp/test_g1_joint_projection.cpp

**Interfaces:**

~~~cpp
enum sonic_joint_projection_failure {
    SonicJointProjectionValid = 0,
    SonicJointProjectionShape,
    SonicJointProjectionContract,
    SonicJointProjectionInput,
    SonicJointProjectionSingular,
    SonicJointProjectionResidual,
    SonicJointProjectionLimit,
    SonicJointProjectionVelocity,
};

struct sonic_joint_projection_diagnostic {
    sonic_joint_projection_failure failure = SonicJointProjectionValid;
    int row = -1;
    float position = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
};

bool sonic_project_joint_state(
    float (&positions)[SonicG1JointCount],
    float (&velocities)[SonicG1JointCount],
    float (&residuals)[SonicG1JointCount],
    sonic_joint_projection_diagnostic& diagnostic,
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    char* error,
    int error_capacity);
~~~

- [ ] **Step 1: Write RED diagnostic tests**

Add tests that independently trigger a malformed slice, bad contract row, non-finite quaternion, excessive off-axis residual, lower-limit violation, upper-limit violation, and non-finite angular velocity. For every case assert the exact enum; for limit failures assert row, position, lower, and upper. Assert all three output arrays and the diagnostic remain transactional until failure is published.

Also retain assertions for the exact existing sonic_project_pose messages, including the left_ankle_roll limit message observed by Stage A.

- [ ] **Step 2: Run the projection test RED**

~~~bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_projection.cpp \
  -o /tmp/test_g1_joint_projection
~~~

Expected: compilation fails because the structured API does not exist.

- [ ] **Step 3: Extract the joint-only transactional core**

Move the existing per-row extraction, twist, residual, limit, and velocity checks into sonic_project_joint_state. Populate a local candidate diagnostic and local output arrays; copy them to the caller only on success. Assign SonicJointProjectionLimit only at the two inclusive range comparisons.

Make sonic_project_pose call the new core, then perform its existing pelvis/global-shape checks and publish sonic_projected_pose only after every check passes. Do not rewrite formulas or error text.

- [ ] **Step 4: Run projection GREEN**

~~~bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_projection.cpp \
  -o /tmp/test_g1_joint_projection
/tmp/test_g1_joint_projection
~~~

Expected: every legacy and structured projection test passes.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/cpp/g1_joint_projection.h tests/cpp/test_g1_joint_projection.cpp
git commit -m "refactor: expose joint projection diagnostics"
~~~

---

### Task 2: Build a deterministic authenticated feasibility certificate

**Files:**
- Create: sonic/cpp/g1_joint_feasibility.h
- Create: tests/cpp/test_g1_joint_feasibility.cpp
- Modify: sonic/cpp/Makefile

**Interfaces:**

~~~cpp
struct sonic_joint_feasibility_certificate {
    array1d<unsigned char> raw_safe;
    array1d<unsigned char> search_safe;
    int frame_count = 0;
    int raw_safe_count = 0;
    int raw_unsafe_count = 0;
    int search_safe_count = 0;
    int joint_limit_violation_count[SonicG1JointCount] = {};
    char mask_sha256[65] = {};
};

bool sonic_build_joint_feasibility_certificate(
    sonic_joint_feasibility_certificate& out,
    const database& db,
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    char* error,
    int error_capacity);
~~~

- [ ] **Step 1: Write RED certificate tests**

Create a four-frame synthetic database with two animation ranges. Cover:

- every raw-safe frame produces raw_safe value 1;
- a single structured limit failure produces value 0 and increments exactly its joint row;
- a residual or malformed-input failure aborts the entire build transactionally;
- a safe frame immediately before an unsafe successor becomes search-unsafe;
- the last frame in a range clamps to itself rather than crossing ranges;
- an unsafe frame 0 aborts certificate construction;
- an all-zero search mask aborts certificate construction;
- counts reconcile exactly with frame_count;
- two identical builds produce the same lowercase 64-hex SHA-256;
- changing one mask byte changes the digest.

- [ ] **Step 2: Run certificate RED**

~~~bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_feasibility.cpp \
  -o /tmp/test_g1_joint_feasibility
~~~

Expected: compilation fails because the feasibility header does not exist.

- [ ] **Step 3: Implement the one-time scan**

For each database frame, reconstruct its local rotations and angular velocities from database arrays using the same slices and indexing convention as the runtime selected pose. Call sonic_project_joint_state once. Mark only SonicJointProjectionLimit as raw-unsafe; return false for every other failure with frame and projection context in the error.

After the raw pass, derive search_safe by applying the runtime successor rule within each database range. A frame is search-safe only if both the selected frame and the emitted successor are raw-safe.

Hash this exact byte stream with the existing sonic/cpp/sha256.h implementation:

1. ASCII schema tag g1-joint-feasibility-certificate/v1 with no terminator.
2. frame_count as one unsigned 64-bit little-endian integer.
3. every raw_safe byte in frame order.
4. every search_safe byte in frame order.

Reject negative frame counts before encoding. Require frame 0 to be raw-safe and at least one frame to be search-safe. Finish into lowercase hexadecimal and publish the candidate object only after all checks, counts, and the digest are complete.

- [ ] **Step 4: Run certificate GREEN and projection regression**

~~~bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_feasibility.cpp \
  -o /tmp/test_g1_joint_feasibility
/tmp/test_g1_joint_feasibility
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_projection.cpp \
  -o /tmp/test_g1_joint_projection
/tmp/test_g1_joint_projection
~~~

Expected: both binaries pass.

- [ ] **Step 5: Register the production dependency and commit**

Add g1_joint_feasibility.h and sha256.h to the mm_chunk_server dependency list in sonic/cpp/Makefile.

~~~bash
git add sonic/cpp/g1_joint_feasibility.h tests/cpp/test_g1_joint_feasibility.cpp \
  sonic/cpp/Makefile
git commit -m "feat: certify G1 database joint feasibility"
~~~

---

### Task 3: Constrain search and progression to certified frames

**Files:**
- Modify: database.h
- Modify: sonic/cpp/g1_runtime.h
- Modify: tests/cpp/test_terrain_database.cpp
- Modify: tests/cpp/test_g1_runtime.cpp

**Interfaces:**

~~~cpp
void database_search(
    int& best_index,
    float& best_cost,
    const database& db,
    slice1d<float> query,
    float transition_cost = 0.0f,
    int ignore_range_end = 20,
    int ignore_surrounding = 20,
    const unsigned char* candidate_mask = nullptr,
    int candidate_mask_count = 0);

struct g1_runtime_frame_feasibility {
    const unsigned char* raw_safe = nullptr;
    const unsigned char* search_safe = nullptr;
    int count = 0;
};
~~~

- [ ] **Step 1: Write RED masked-search tests**

In test_terrain_database.cpp, construct a cost ordering in which the cheapest frame is masked out and the next finite frame is selected. Assert:

- masked candidates are never evaluated as winners;
- a valid incumbent that is masked out is not retained;
- a null mask preserves the exact old result through the old API;
- a count mismatch leaves no selected index;
- an all-zero mask leaves best_index equal to -1.

- [ ] **Step 2: Write RED runtime progression tests**

In test_g1_runtime.cpp, supply a small certificate and assert:

- ordinary continuation proceeds when the current successor is raw-safe;
- an unsafe successor forces search before pose publication;
- search cannot choose a search-unsafe frame even when it has lowest cost;
- all-zero search_safe fails with the exact error and leaves controller state/result unchanged;
- an invalid mask shape fails as integration error;
- the existing live inertialized joint projection still rejects an unsafe post-blend state.

- [ ] **Step 3: Run both test binaries RED**

~~~bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_database.cpp \
  -o /tmp/test_terrain_database
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp \
  -o /tmp/test_g1_runtime
~~~

Expected: compilation or assertions fail because masked search and runtime feasibility do not exist.

- [ ] **Step 4: Add the masked database primitive**

Add trailing optional candidate-mask pointer/count arguments to motion_matching_search and database_search without changing their existing argument order or distance arithmetic. Existing callers therefore compile to the exact null-mask path. When a mask is supplied, reject a count mismatch by leaving no selected index; skip a masked incumbent and skip each masked leaf before cost evaluation. Binary-byte validation remains a one-time certificate-build invariant, not a hot-path full scan.

When no allowed finite candidate exists, leave best_index equal to -1. The G1 runtime converts only that valid masked-search outcome to the exact registered error. Preserve old unmasked selection behavior for all existing callers.

- [ ] **Step 5: Thread feasibility through the renderer-free runtime**

Add a feasibility parameter to the authenticated runtime overload and retain an old overload that supplies no mask for existing tests/callers. Validate the certificate shape before cloning state.

Before ordinary one-frame progression, inspect the exact successor the runtime will emit. If it is raw-unsafe, force the existing search path. Pass search_safe to database_search. If masked search returns no index, render the exact no joint-limit-safe database candidate error. After selection and inertialization, retain the existing sonic_project_pose call as the final projection gate.

Do not silently skip frames, cross animation ranges, or modify transition costs.

- [ ] **Step 6: Run unit GREEN**

~~~bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_database.cpp \
  -o /tmp/test_terrain_database
/tmp/test_terrain_database
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp \
  -o /tmp/test_g1_runtime
/tmp/test_g1_runtime
~~~

Expected: both binaries pass, including all prior unmasked and runtime tests.

- [ ] **Step 7: Commit**

~~~bash
git add database.h sonic/cpp/g1_runtime.h \
  tests/cpp/test_terrain_database.cpp tests/cpp/test_g1_runtime.cpp
git commit -m "feat: mask unsafe G1 matcher candidates"
~~~

---

### Task 4: Own and publish the certificate in the chunk server

**Files:**
- Modify: sonic/cpp/mm_chunk_server.cpp
- Modify: sonic/cpp/Makefile
- Modify: tests/python/test_mm_chunk_server.py

**Protocol identity:** joint_feasibility has exactly these seven keys: schema, frame_count, raw_safe_count, raw_unsafe_count, search_safe_count, mask_sha256, and joint_limit_violation_count. schema is the exact string g1-joint-feasibility-certificate/v1; joint_limit_violation_count contains exactly 29 nonnegative integers in registered source-joint order.

- [ ] **Step 1: Write RED real and fake server tests**

Extend the hello assertions to require the exact seven-key nested object. For the fake adapter, require a deterministic all-safe identity whose schema, frame_count, counts, 29 zeros, and digest are stable across launches.

For the real adapter, add source/behavior tests that require certificate construction after database and contract validation, mask ownership for the full server lifetime, and the feasibility argument on every g1_runtime_step call.

Add negative hello fixtures for missing/extra nested keys, wrong schema, non-integer counts, count mismatch, wrong joint-count length, uppercase or malformed digest, and inconsistent frame_count.

- [ ] **Step 2: Run server tests RED**

~~~bash
make -C sonic/cpp clean
make -C sonic/cpp mm_chunk_server
PYTHONPATH=sonic/python SONIC_MM_SERVER=sonic/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_mm_chunk_server -v
~~~

Expected: hello assertions fail because joint_feasibility is absent.

- [ ] **Step 3: Build and own the certificate during initialization**

Add sonic_joint_feasibility_certificate to the real adapter. Build it only after database shape validation and joint-contract loading succeed. Reject startup on any non-limit projection failure or count/hash inconsistency. Pass non-owning slices into each runtime step; the adapter-owned arrays outlive every request.

Use a deterministic synthetic certificate for fake mode and compute its digest through the same helper, rather than hard-coding an unauthenticated string.

- [ ] **Step 4: Publish strict hello identity**

Write joint_feasibility as a nested key in the existing data object. Emit all seven approved keys in fixed order, counts as JSON integers, exactly 29 violation counts, and mask_sha256 as a string. Keep protocol_version equal to 1 and leave every existing identity field unchanged.

- [ ] **Step 5: Run server GREEN**

~~~bash
make -C sonic/cpp clean
make -C sonic/cpp mm_chunk_server
PYTHONPATH=sonic/python SONIC_MM_SERVER=sonic/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_mm_chunk_server -v
~~~

Expected: fake and real-source server tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add sonic/cpp/mm_chunk_server.cpp sonic/cpp/Makefile \
  tests/python/test_mm_chunk_server.py
git commit -m "feat: publish authenticated joint feasibility"
~~~

---

### Task 5: Strictly bind feasibility identity through Python orchestration

**Files:**
- Modify: sonic/python/mm_sonic/schema.py
- Modify: sonic/python/mm_sonic/scene.py
- Modify: sonic/python/mm_sonic/cli.py
- Modify: tests/python/test_sonic_schema.py
- Modify: tests/python/test_sonic_scene.py
- Modify: tests/python/test_sonic_cli.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class JointFeasibilityIdentity:
    schema: str
    frame_count: int
    raw_safe_count: int
    raw_unsafe_count: int
    search_safe_count: int
    joint_limit_violation_count: tuple[int, ...]
    mask_sha256: str

def parse_joint_feasibility_identity(value: object) -> JointFeasibilityIdentity:
    ...
~~~

- [ ] **Step 1: Write RED schema tests**

Require exact key equality; exact schema equality; bool rejection for integer fields; nonnegative counts; frame_count equality to raw_safe_count plus raw_unsafe_count; positive search_safe_count no greater than raw_safe_count; exactly 29 nonnegative per-joint counts; sum of violation counts equal to raw_unsafe_count; and lowercase full-match [0-9a-f]{64} mask digest validation.

Test a valid real-shaped identity with frame_count 459682, raw_safe_count 458619, raw_unsafe_count 1063, and a 29-element violation vector summing to 1063.

- [ ] **Step 2: Write RED scene and CLI binding tests**

Require scene hello verification to parse the identity and reject malformed values before any rollout.

In CLI tests, launch distinct preflight and run hello fixtures and assert:

- exact identities permit rollout;
- any count, violation-vector, or digest drift fails as integration before evidence acceptance;
- the accepted identity is retained in basis/evidence metadata;
- exact structured generation_failed error no joint-limit-safe database candidate yields the registered scientific verdict;
- the same words in stderr, a different status, or a prefixed/suffixed message remain integration failures.

- [ ] **Step 3: Run Python tests RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_schema \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_cli -v
~~~

Expected: parser/import and binding assertions fail.

- [ ] **Step 4: Implement strict parsing and cross-binding**

Parse joint_feasibility beside the existing hello identities. Use exact key-set comparison and explicit type checks before constructing the frozen dataclass. Do not accept coercion, unknown keys, uppercase hashes, or inconsistent totals.

Store the preflight identity in coordinator state. When the real server hello arrives, compare the full dataclass for equality before allowing chunks or basis evidence. Preserve the matched identity in the run manifest, committed basis metadata, and gate evidence.

Classify only the structured protocol envelope whose status is generation_failed and whose error equals no joint-limit-safe database candidate as scientific. All parse failures, identity drift, and text lookalikes are integration failures.

- [ ] **Step 5: Run Python GREEN**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_schema \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_cli -v
~~~

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add sonic/python/mm_sonic/schema.py sonic/python/mm_sonic/scene.py \
  sonic/python/mm_sonic/cli.py tests/python/test_sonic_schema.py \
  tests/python/test_sonic_scene.py tests/python/test_sonic_cli.py
git commit -m "feat: bind joint feasibility evidence"
~~~

---

### Task 6: Qualify focused C++, full-suite Python, and real Stage A behavior

**Files:**
- Create: docs/superpowers/results/2026-07-17-g1-joint-feasibility-mask.md
- Modify: reliable-claude-projects/motion-matching/FRESH_SESSION_HANDOFF.md outside this repository after commits are final

- [ ] **Step 1: Run the affected warning-strict C++ matrix**

~~~bash
set -euo pipefail
for name in test_g1_joint_projection test_g1_joint_feasibility \
  test_terrain_database test_g1_runtime; do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/$name.cpp" -o "/tmp/$name"
  "/tmp/$name"
done
~~~

Expected: all four affected C++ binaries compile and pass.

- [ ] **Step 2: Run the protected warning-strict 18-module Python catalog**

~~~bash
PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  /home/ubuntu/reliable-claude-projects/motion-matching/verifiers/stage-a-integration-env-py310-20260717/venv/bin/python \
  -B -W error -m unittest \
  tests.python.test_sonic_artifacts tests.python.test_sonic_cli \
  tests.python.test_sonic_commands tests.python.test_sonic_coordinator \
  tests.python.test_sonic_external tests.python.test_sonic_gated_sim \
  tests.python.test_sonic_joints tests.python.test_sonic_metrics \
  tests.python.test_sonic_process tests.python.test_sonic_reference \
  tests.python.test_sonic_resample tests.python.test_sonic_runtime_parity \
  tests.python.test_sonic_scene tests.python.test_sonic_schema \
  tests.python.test_sonic_timeline tests.python.test_sonic_timing \
  tests.python.test_sonic_transform tests.python.test_sonic_zmq_v1
~~~

Expected: the complete protected catalog passes; record exact count, skips, and duration.

- [ ] **Step 3: Inspect the real hello certificate**

Start sonic/build/mm_chunk_server against the registered real database and issue one hello request. Record and reconcile:

- frame_count equals 459682;
- raw_unsafe_count equals 1063;
- raw_safe_count equals 458619;
- the 29 violation counts sum to 1063;
- left_ankle_roll contributes 938 violations;
- search_safe_count is positive and no greater than raw_safe_count;
- mask_sha256 is stable across two fresh server launches.

Any mismatch blocks Stage A and requires diagnosis; do not rewrite expected evidence silently.

- [ ] **Step 4: Run unchanged Stage A**

Use the same known-good command, scene inputs, and environment as stage-a-joint-limit-scientific-integrated-20260717, changing only the output root to:

~~~text
sonic/runs/stage-a-joint-feasibility-integrated-20260717
~~~

Qualification accepts either truthful terminal result, but scientific-hypothesis success is narrower:

- The intervention hypothesis succeeds only if gate 4 contains all 601 required frames and the prior 866-to-867 raw-limit failure is absent.
- An exact scientific no-candidate or later inertialized-limit failure is recorded as the next scientific result, not as a pass.

A projection limit failure after masked selection, any certificate mismatch, or any protocol/schema failure is an implementation defect and blocks completion.

- [ ] **Step 5: Record exact evidence**

Write the result document with:

- base and final commit SHAs;
- every focused RED/GREEN command and observed result;
- full C++ and Python counts/duration;
- real certificate counts, 29-value vector, and digest;
- two-launch digest stability;
- Stage A output path, exit code, gate verdicts, and first structured failure if any;
- confirmation that the live inertialized projection gate remained enabled;
- git status and runtime-survivor check.

- [ ] **Step 6: Commit qualification evidence**

~~~bash
git add docs/superpowers/results/2026-07-17-g1-joint-feasibility-mask.md
git commit -m "docs: record G1 feasibility qualification"
git diff --check
git status --short --branch
~~~

Expected: the motion worktree is clean after the result commit. Update the external fresh-session handoff with the final SHAs and evidence paths; do not push or merge unless separately authorized.
