# G1 Unscheduled Lazy Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute this plan task by task. Preserve the
> RED evidence, stop at every named live gate, and request independent review
> before advancing.

**Goal:** When a matching-enabled frame does not schedule legacy search and
its incumbent slot zero finitely fails hidden IK, lazily evaluate the existing
strict recovery tail once in the same transaction, without a second legacy
search, a larger candidate set, or any weakened physical rule.

**Architecture:** `controller.cpp` prepares one exact recovery request on every
matching-enabled matcher prefix but continues to call `database_search` only
when the existing schedule says to search. `g1_frame_transaction.h`
authenticates either the existing scheduled request grammar or the new
equal-frame unscheduled grammar, then calls the unchanged strict provider only
after a finite slot-zero result. `g1_candidate_certification_trace.h` accepts
and audits both grammars against the same-build exhaustive oracle. The first
dual-certified candidate still commits atomically through the existing raw/IK
projection path.

**Tech stack:** C++17, exact binary32 request words, strict-FP recovery/
clearance/root-reach objects, fast-math controller callers, existing Raylib
headless route runner on disposable `DISPLAY=:1`, TSV candidate evidence,
production CSV checkers, and Python `unittest`.

## Frozen inputs and constraints

- Implement the approved design at
  `docs/superpowers/specs/2026-07-17-g1-unscheduled-lazy-recovery-design.md`.
  Its SHA-256 is
  `0b3972bd3196589b0a5b07bb5da4991c706da3dfe2b6840217378838d1940eb6`.
- This plan itself must be committed with subject
  `docs: plan unscheduled lazy recovery` before Task 1 begins. Task 1
  authenticates the committed blob dynamically so the plan does not contain a
  self-referential commit hash.
- The five bounded-candidate commits `7c098c9`, `a8c24c8`, `975206a`,
  `6c83cfb`, and `2ac258d`, plus design commit `6568694`, must remain ancestors
  of `HEAD`.
- Simulation and source motion stay exactly 25 Hz. Do not resample, substep,
  or introduce any 60 Hz assumption.
- Heading and travel direction remain independent. Do not rotate heading
  toward motion, including for forward, backward, lateral, or diagonal routes.
- Do not modify search cadence, feature weights, database data, terrain data,
  scene routes, renderer, mesh, schema, log checker, footprint rules,
  clearance tolerances, root-reach cap, swing ladder, contact thresholds, or
  any IK threshold.
- Do not add a raw-only fallback, pose hold, second `database_search`,
  lookahead, retry loop, provider call after a global error, or ninth attempt.
- `G1CandidateAttemptCapacity == 8`,
  `G1RecoveryTransitionCapacity == 6`, and
  `G1RecoveryTailCapacity == 7` remain unchanged. Equal incumbent and legacy
  exclusions make the unscheduled provider tail at most six records, so slot
  zero plus tail is at most seven attempts.
- Matching-disabled `G1_TestSequential` execution never prepares a live
  request and never calls recovery. An accepted slot zero in any mode performs
  zero provider work.
- `database_search` remains the only legacy traversal and the only owner of a
  scheduled slot-zero legacy score. The strict provider remains the only owner
  of recovery scores.
- Do not modify `g1_candidate_recovery.h` or
  `g1_candidate_recovery.cpp`; their existing equal-frame behavior is a tested
  dependency of this repair. If that dependency fails, stop and return the
  design for review rather than expanding provider scope implicitly.
- `motion_match_log.h`, `resources/check_g1_runtime_log.py`, all terrain and
  database artifacts, and `resources/g1_visualizer_process.py` are immutable.
- Do not inspect, discover, query, signal, attach to, replace, restart, or
  otherwise touch the currently running visualizer. In particular, do not use
  `ps`, `pgrep`, `pidof`, `pkill`, `systemctl`, or the visualizer helper. Every
  live command below invokes a uniquely named disposable binary directly on
  `DISPLAY=:1`. The user-requested candidate checkpoint may later signal only
  the new PID captured directly from its own launch, after reauthenticating
  that PID's start time, executable, inode, and command line; it never searches
  for any process.
- Preserve unrelated work. Before every commit, stage only the paths named by
  that task and inspect `git diff --cached --name-only`.

## Planned source scope

- Modify `tests/cpp/test_g1_frame_transaction.cpp`: generic RED fixtures for
  matching-enabled unscheduled recovery, laziness, exact request binding,
  exhaustion, global error, disabled mode, and unchanged work bounds.
- Modify `tests/cpp/test_g1_frame_transaction_production.cpp`: real matcher
  prefix/source ownership tests, unscheduled trace grammar fixtures, negative
  mutations, and a focused RED selector.
- Modify `controller.cpp`: prepare a request for every matching-enabled prefix
  and accept strict-tail provenance for both scheduled and unscheduled forms.
- Modify `g1_frame_transaction.h`: authenticate the two request forms and
  permit one provider call after a matching-enabled finite slot-zero result.
- Modify `g1_candidate_certification_trace.h`: serialize and exhaustively audit
  scheduled-legacy and unscheduled-incumbent recovery grammars.
- Verify only `g1_candidate_recovery.h`, `g1_candidate_recovery.cpp`,
  `motion_match_log.h`, `resources/check_g1_runtime_log.py`, all other source,
  terrain/database assets, and visualizer files.

---

### Task 1: Implement the Equal-Frame Unscheduled Recovery Route Test-First

**Files:**

- Modify: `tests/cpp/test_g1_frame_transaction.cpp`
- Modify: `tests/cpp/test_g1_frame_transaction_production.cpp`
- Modify: `g1_frame_transaction.h`
- Modify: `controller.cpp`
- Modify: `g1_candidate_certification_trace.h`
- Verify only: `g1_candidate_recovery.h`
- Verify only: `g1_candidate_recovery.cpp`

**Runtime contract:**

- On every matching-enabled matcher prefix,
  `recovery_request_ready == true` and the request is a bit-exact snapshot of
  the existing query, transition cost, public incumbent cost, source database,
  and exclusions.
- `matching_scheduled` and `legacy_search_performed` remain true only when the
  existing schedule ran `database_search`.
- Scheduled request: request incumbent is the immutable baseline and request
  legacy selection is the legacy slot-zero selection.
- Unscheduled request: immutable baseline, incumbent-owned slot zero, request
  incumbent, and request legacy selection are the same frame.
- Slot zero evaluates once. Acceptance returns immediately; global error
  returns immediately; finite matching-disabled failure publishes immediately;
  finite matching-enabled failure authenticates and calls the provider once.
- A later tail acceptance commits normally. Tail exhaustion publishes the
  original slot-zero failure. A malformed request/set or provider error is a
  nonpublishing global error.

- [ ] **Step 1: Authenticate the base and record immutable owners**

Run from the worktree root:

```bash
set -euo pipefail
out=/tmp/g1-unscheduled-lazy-recovery/task1
rm -rf "$out"
mkdir -p "$out/red" "$out/green" "$out/full"
test "$(sha256sum docs/superpowers/specs/2026-07-17-g1-unscheduled-lazy-recovery-design.md | cut -d' ' -f1)" = \
  0b3972bd3196589b0a5b07bb5da4991c706da3dfe2b6840217378838d1940eb6
for commit in 7c098c9 a8c24c8 975206a 6c83cfb 2ac258d 6568694; do
  git merge-base --is-ancestor "$commit" HEAD
done
plan=docs/superpowers/plans/2026-07-17-g1-unscheduled-lazy-recovery.md
plan_commit=$(git log -1 --format=%H -- "$plan")
test -n "$plan_commit"
test "$(git show -s --format=%s "$plan_commit")" = \
  'docs: plan unscheduled lazy recovery'
git merge-base --is-ancestor "$plan_commit" HEAD
test "$(git hash-object "$plan")" = \
  "$(git rev-parse "$plan_commit:$plan")"
test -z "$(git status --short -- "$plan")"
git diff --exit-code -- g1_candidate_recovery.h g1_candidate_recovery.cpp \
  motion_match_log.h resources/check_g1_runtime_log.py
sha256sum g1_candidate_recovery.h g1_candidate_recovery.cpp \
  motion_match_log.h resources/check_g1_runtime_log.py \
  > "$out/immutable.before.sha256"
git status --short > "$out/status.before"
git rev-parse HEAD > "$out/base.txt"
terrain=/tmp/g1-terrain-footprint-runtime-v1
test -d "$terrain"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$out/terrain.before.sha256"
test -s "$out/terrain.before.sha256"
(
  cd "$terrain"
  sha256sum -c "$out/terrain.before.sha256"
)
```

Expected: the approved design, committed plan, and all prerequisite commits
authenticate; the provider and production log/checker owners are clean and
hashed; the sorted external terrain/database package manifest verifies.

- [ ] **Step 2: Write the generic coordinator RED tests**

In `tests/cpp/test_g1_frame_transaction.cpp`, first change only test code:

1. Add `run_unscheduled_lazy_recovery_red_tests()` and call it both from the
   ordinary suite and from a new selector:

```cpp
if (argc == 2 &&
    std::strcmp(argv[1], "--unscheduled-lazy-recovery-red") == 0) {
    std::fputs("G1_UNSCHEDULED_LAZY_RECOVERY_RED_SELECTED\n", stderr);
    run_unscheduled_lazy_recovery_red_tests();
    return 0;
}
```

2. Add `test_unscheduled_incumbent_finite_rejection_uses_lazy_recovery()`:
   disable scheduled search, reject `Incumbent` at
   `G1FrameStageIkPoseCertificate`, return a valid one-record strict tail
   containing `CandidateB`, and allow B to dual-accept. Its primary `check`
   message is exactly
   `unscheduled finite incumbent uses one lazy strict tail and commits first dual winner`.
   Require:

```text
transaction status                  Accepted
legacy traversals                   0
provider calls/recovery traversals  1 / 1
attempt count                       2
attempt 0 kind/owner                incumbent / incumbent
attempt 0 disposition               common accepted, raw accepted, IK finite
attempt 1 kind/owner                strict-recovery / strict-recovery
attempt 1 disposition               dual accepted
request incumbent/legacy frame      Incumbent.selected_frame / same frame
accepted frame                      CandidateB.executed_frame
public rejection                    absent
```

3. Add focused tests proving:

   - an accepted unscheduled incumbent performs zero provider calls;
   - an unscheduled provider global error is a nonpublishing global error;
   - an exhausted unscheduled six-record strict tail preserves the exact first
     incumbent rejection and stays at seven total attempts;
   - matching-disabled finite failure performs zero provider work;
   - mutation of either equal request frame, database pointer, transition-cost
     word, incumbent-cost word, any of 31 query words, slot-zero owner, rank,
     ordering, or duplicate record causes a global error before a tail attempt.

4. Do not yet change the generic matcher fixture's current
   `recovery_request_ready = scheduled` behavior. That missing behavior is the
   intended RED.

- [ ] **Step 3: Write the production-prefix and trace RED tests**

In `tests/cpp/test_g1_frame_transaction_production.cpp`, first change only test
code:

1. Add two independent selectors guarded by both existing test seams, so a
   failure in one ownership boundary cannot hide the other RED:

```cpp
if (argc == 2 &&
    std::strcmp(argv[1], "--unscheduled-prefix-red") == 0) {
    std::fputs("G1_UNSCHEDULED_PREFIX_RED_SELECTED\n", stderr);
    run_unscheduled_prefix_red_tests();
    return 0;
}
if (argc == 2 &&
    std::strcmp(argv[1], "--unscheduled-trace-red") == 0) {
    std::fputs("G1_UNSCHEDULED_TRACE_RED_SELECTED\n", stderr);
    run_unscheduled_trace_red_tests();
    return 0;
}
```

   The prefix selector owns the real matcher/source tests and its primary RED
   message is exactly
   `matching-enabled unscheduled finite incumbent prepares and uses an equal-frame recovery request`.
   The trace selector owns only the synthetic serializer/oracle tests and its
   primary RED message is exactly
   `trace accepts exact unscheduled incumbent recovery grammar`. Neither emits
   the ordinary scheduled transcript.
2. Split the existing
   `test_matching_disabled_and_unscheduled_frames_attempt_only_incumbent()`
   expectation: accepted unscheduled and matching-disabled frames still attempt
   only slot zero and call no provider; a finite matching-enabled unscheduled
   frame must call the provider, whereas a matching-disabled one must not.
3. Add a source ownership test requiring exactly one production
   `database_search(` token, that call remaining inside `if (state.searched)`,
   and request preparation guarded by `matching_enabled`. Scope its heading
   guard to the changed matcher/recovery blocks: they may snapshot the existing
   requested heading but must not derive or assign heading from
   `commanded_velocity`, `move_stick`, or `desired_velocity`. Do not globally
   forbid the intentional live non-strafe and route-without-override heading
   paths already owned elsewhere. Retain
   `test_controller_publishes_independent_travel_and_heading()` and the
   production heading-override/strafe matrices as the behavior oracles.
4. Extend `CandidateTraceFixture` with a helper that constructs the legal
   unscheduled form:

```text
legacy_traversals                         0
recovery_provider_calls                   1
request incumbent/legacy-selected frame  10 / 10
slot zero                                 incumbent frame 10, incumbent owner
slot-zero cost                            exact public incumbent-cost word
slot-zero outcome                         common/raw accepted, IK finite
provider tail                             six strict transitions, no incumbent
attempted prefix                          slot zero then first strict winner
```

5. Assert serialization produces the header plus six complete tail rows and
   one slot-zero row, `legacy_traversals=0`, `provider_calls=1`,
   `recovery_traversals=1`, and exact accelerated/exhaustive equality on every
   row. Keep the scheduled transcript byte-for-byte unchanged.
6. Replace the now-obsolete negative mutation “a request and tail follow an
   incumbent slot zero” with exact invalid forms: unequal request frames,
   request/slot-zero mismatch, public-cost mismatch, legacy traversal drift,
   legacy score owner on an incumbent, incumbent score owner on a legacy
   record, database-pointer substitution, one-bit mutation of each selected
   query-word case, duplicate strict frame, and malformed recovery rank. Each
   must fail before writing any bytes for that transaction. The query mutation
   fixture must cover all 31 positions, one transaction attempt at a time.

- [ ] **Step 4: Capture the intended RED in strict and fast callers**

Build the existing strict dependency objects and both test programs. The new
focused selector must compile, print its unique sentinel, and exit nonzero on
the missing route; the failure must not be an unrelated compiler warning or a
sanitizer failure.

```bash
set -euo pipefail
out=/tmp/g1-unscheduled-lazy-recovery/task1
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread \
  -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/red/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/red/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/red/recovery-seam.o"

for flavor in strict fast; do
  flags=("${strict[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction.cpp \
    -o "$out/red/frame-${flavor}.o"
  g++ "$out/red/frame-${flavor}.o" "$out/red/clearance.o" \
    "$out/red/root-reach.o" "$out/red/recovery-seam.o" \
    -o "$out/red/frame-${flavor}"
  if "$out/red/frame-${flavor}" --unscheduled-lazy-recovery-red \
       >"$out/red/frame-${flavor}.stdout" \
       2>"$out/red/frame-${flavor}.stderr"; then
    echo "ERROR: generic $flavor RED unexpectedly passed" >&2
    exit 1
  fi
  rg 'G1_UNSCHEDULED_LAZY_RECOVERY_RED_SELECTED' \
    "$out/red/frame-${flavor}.stderr"
  rg 'unscheduled finite incumbent uses one lazy strict tail and commits first dual winner' \
    "$out/red/frame-${flavor}.stderr"

  g++ "${flags[@]}" "${rayinc[@]}" -D_DEFAULT_SOURCE \
    -DPLATFORM_DESKTOP -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$out/red/controller-${flavor}.o"
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$out/red/production-${flavor}.o"
  g++ "$out/red/production-${flavor}.o" \
    "$out/red/controller-${flavor}.o" "$out/red/clearance.o" \
    "$out/red/root-reach.o" "$out/red/recovery-seam.o" \
    "${raylib[@]}" \
    -o "$out/red/production-${flavor}"
  for selector in prefix trace; do
    if "$out/red/production-${flavor}" \
         "--unscheduled-${selector}-red" \
         >"$out/red/production-${flavor}-${selector}.stdout" \
         2>"$out/red/production-${flavor}-${selector}.stderr"; then
      echo "ERROR: production $flavor $selector RED unexpectedly passed" >&2
      exit 1
    fi
  done
  rg 'G1_UNSCHEDULED_PREFIX_RED_SELECTED' \
    "$out/red/production-${flavor}-prefix.stderr"
  rg 'matching-enabled unscheduled finite incumbent prepares and uses an equal-frame recovery request' \
    "$out/red/production-${flavor}-prefix.stderr"
  rg 'G1_UNSCHEDULED_TRACE_RED_SELECTED' \
    "$out/red/production-${flavor}-trace.stderr"
  rg 'trace accepts exact unscheduled incumbent recovery grammar' \
    "$out/red/production-${flavor}-trace.stderr"
done
sha256sum "$out"/red/*.stderr > "$out/red/SHA256SUMS"
```

Expected: both generic runs and all four independent production-prefix/trace
runs fail on their missing unscheduled contracts and retain their unique
sentinel. Record the assertion text and hashes in the task report before
editing product code.

- [ ] **Step 5: Prepare the exact request without changing search cadence**

In the real `G1FrameStageMatcherSearch` branch of `controller.cpp`:

1. Keep `state.searched` and the single `database_search` block unchanged.
2. After constructing slot zero, reset the request unconditionally and make
   readiness depend on matching mode, not scheduling:

```cpp
scratch.matching_scheduled = state.searched;
scratch.legacy_search_performed = state.searched;
scratch.recovery_request = G1RecoveryRequest{};
scratch.recovery_request_ready = matching_enabled;
if (matching_enabled) {
    scratch.recovery_request.db = external.db;
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        scratch.recovery_request.raw_query[feature] =
            scratch.query[feature];
    }
    scratch.recovery_request.incumbent_frame = prior_index;
    scratch.recovery_request.legacy_selected_frame = slot_selected;
    scratch.recovery_request.transition_cost = scratch.transition_cost;
    scratch.recovery_request.public_incumbent_cost =
        state.incumbent_cost;
    scratch.recovery_request.ignore_range_end = 20;
    scratch.recovery_request.ignore_surrounding = 20;
}
```

On an unscheduled frame `slot_selected == prior_index`, giving the required
equal-frame request without a second traversal.

3. In `G1FrameStageCandidateApply`, compute the authorization explicitly:

```cpp
const bool matching_enabled =
    external.tuning.mode != G1_TestSequential;
const bool recovery_context_valid =
    matching_enabled &&
    scratch.recovery_request_ready &&
    scratch.matching_scheduled ==
        scratch.legacy_search_performed;
```

   A strict recovery transition requires `recovery_context_valid` plus every
   existing selected/executed/source/rank/cost/exclusion check; it must not
   require both scheduled flags to be true. A legacy candidate continues to
   require true/true scheduled flags. An incumbent uses request-owned frame and
   cost only when `recovery_context_valid`; otherwise it uses slot-zero frame
   and cost. This authorizes true/true scheduled recovery and false/false
   matching-enabled unscheduled recovery, while false/false sequential mode
   remains unauthorized for any tail.

- [ ] **Step 6: Authenticate both request forms and invoke recovery lazily**

In `g1_frame_transaction.h`:

1. Change the prefix invariant to:

```cpp
if (prefix_scratch.matching_scheduled != matching_scheduled ||
    prefix_scratch.legacy_search_performed != matching_scheduled ||
    prefix_scratch.recovery_request_ready != matching_enabled) {
    return G1FrameTransactionGlobalError;
}
```

2. Rename the request authenticator's `legacy_record` parameter to
   `slot_zero` and retain every exact pointer, query, cost, finiteness,
   exclusion, range, and immutable-baseline check. Replace the scheduled-only
   prefix with two explicit forms:

```cpp
const bool scheduled_form =
    prefix_scratch.matching_scheduled &&
    prefix_scratch.legacy_search_performed &&
    slot_zero.kind == G1CandidateLegacy &&
    request.legacy_selected_frame == slot_zero.selected_frame;

const bool unscheduled_form =
    !prefix_scratch.matching_scheduled &&
    !prefix_scratch.legacy_search_performed &&
    slot_zero.kind == G1CandidateIncumbent &&
    !slot_zero.transitioned &&
    slot_zero.selected_frame == immutable_baseline.frame_index &&
    request.legacy_selected_frame == request.incumbent_frame &&
    request.legacy_selected_frame == slot_zero.selected_frame;
```

Require `recovery_request_ready` and exactly one of those forms. Keep
`request.incumbent_frame == immutable_baseline.frame_index` common to both.

3. After slot-zero finite rejection, replace the current
   `if (!matching_scheduled)` early publication with:

```cpp
if (!matching_enabled) {
    return publish_first_failure();
}
```

Then run the same request authentication, exactly one provider call, whole-set
validation, and ordered tail loop already used for scheduled recovery. Do not
add a second provider path or duplicate loop.

4. Keep trace accounting exact: scheduled recovery has
   `legacy_traversals=1`; unscheduled recovery has `legacy_traversals=0`; both
   have provider calls `1` and provider work `accelerated_traversals=1`.

5. Complete the test fixture's GREEN side in
   `tests/cpp/test_g1_frame_transaction.cpp`: set
   `recovery_request_ready = matching_enabled`, reset the request on every
   matcher prefix, fill all request fields whenever matching is enabled, and
   choose `legacy_selected_frame = scratch.slot_zero_record.selected_frame`.
   Thus its unscheduled form binds incumbent/legacy/slot zero to `Incumbent`,
   while sequential mode retains an exact reset request and false readiness.

- [ ] **Step 7: Extend trace grammar without weakening invalid-state checks**

In `g1_candidate_certification_trace.h`, retain all existing enum, unused-slot,
attempt-prefix, disposition, alias, buffer, and exhaustive-equality checks.
Replace the scheduled-only request binding with:

```cpp
const bool scheduled_recovery =
    trace.legacy_traversals == 1U &&
    slot_zero.candidate.kind == G1CandidateLegacy &&
    slot_zero.score_owner == G1CandidateScoreLegacy &&
    trace.recovery_request.legacy_selected_frame ==
        slot_zero.candidate.selected_frame;

const bool unscheduled_recovery =
    trace.legacy_traversals == 0U &&
    slot_zero.candidate.kind == G1CandidateIncumbent &&
    slot_zero.score_owner == G1CandidateScoreIncumbent &&
    !slot_zero.candidate.transitioned &&
    trace.recovery_request.incumbent_frame ==
        slot_zero.candidate.selected_frame &&
    trace.recovery_request.legacy_selected_frame ==
        slot_zero.candidate.selected_frame &&
    g1_frame_float_bits_equal(
        trace.recovery_request.public_incumbent_cost,
        slot_zero.candidate.selected_cost);
```

If a request is available, require a finite slot zero and exactly one legal
form. Preserve the rule that a finite legacy slot zero cannot omit its request.
A finite incumbent without a request remains serializable because this seam
does not carry matching mode and matching-disabled execution is legal; the
controller/coordinator prevents matching-enabled omission.

Pass `trace.recovery_request` unchanged to
`g1_recovery_candidates_exhaustive_for_test` only after the transaction is
complete. Do not let the exhaustive result influence selection.

- [ ] **Step 8: Run focused GREEN in strict, fast, and sanitizer builds**

Use the same dependency separation as RED, rerun both focused selectors and
ordinary suites, then require transcript parity:

```bash
set -euo pipefail
out=/tmp/g1-unscheduled-lazy-recovery/task1
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread \
  -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/green/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/green/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/green/recovery-seam.o"

for flavor in strict fast; do
  flags=("${strict[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction.cpp \
    -o "$out/green/frame-${flavor}.o"
  g++ "$out/green/frame-${flavor}.o" "$out/green/clearance.o" \
    "$out/green/root-reach.o" "$out/green/recovery-seam.o" \
    -o "$out/green/frame-${flavor}"
  "$out/green/frame-${flavor}" --unscheduled-lazy-recovery-red
  "$out/green/frame-${flavor}"

  g++ "${flags[@]}" "${rayinc[@]}" -D_DEFAULT_SOURCE \
    -DPLATFORM_DESKTOP -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$out/green/controller-${flavor}.o"
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$out/green/production-${flavor}.o"
  g++ "$out/green/production-${flavor}.o" \
    "$out/green/controller-${flavor}.o" "$out/green/clearance.o" \
    "$out/green/root-reach.o" "$out/green/recovery-seam.o" \
    "${raylib[@]}" \
    -o "$out/green/production-${flavor}"
  "$out/green/production-${flavor}" --unscheduled-prefix-red
  "$out/green/production-${flavor}" --unscheduled-trace-red
  "$out/green/production-${flavor}"
  "$out/green/production-${flavor}" --trace-transcript \
    > "$out/green/trace-${flavor}.tsv"
done
cmp "$out/green/trace-strict.tsv" "$out/green/trace-fast.tsv"

san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -I.)
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_clearance.cpp -o "$out/green/clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_ik_root_reach.cpp -o "$out/green/root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/green/recovery-san.o"
g++ "${san[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction.cpp \
  -o "$out/green/frame-san.o"
g++ "${san[@]}" "$out/green/frame-san.o" \
  "$out/green/clearance-san.o" "$out/green/root-reach-san.o" \
  "$out/green/recovery-san.o" -o "$out/green/frame-san"
g++ "${san[@]}" "${rayinc[@]}" -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -DG1_CONTROLLER_NO_MAIN \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$out/green/controller-san.o"
g++ "${san[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o "$out/green/production-san.o"
g++ "${san[@]}" "$out/green/production-san.o" \
  "$out/green/controller-san.o" "$out/green/clearance-san.o" \
  "$out/green/root-reach-san.o" "$out/green/recovery-san.o" \
  "${raylib[@]}" \
  -o "$out/green/production-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$out/green/frame-san" --unscheduled-lazy-recovery-red
for selector in prefix trace; do
  ASAN_OPTIONS=detect_leaks=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$out/green/production-san" "--unscheduled-${selector}-red"
done
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$out/green/production-san" --trace-transcript \
  > "$out/green/trace-san.tsv"
cmp "$out/green/trace-strict.tsv" "$out/green/trace-san.tsv"
```

Expected: focused and ordinary suites pass under both callers, sanitizers are
silent, and the existing scheduled transcript remains byte-identical across
strict, fast, and sanitizer builds. The new unscheduled fixture independently
proves its seven-row candidate grammar and exact oracle equality.

- [ ] **Step 9: Re-run no-seam privacy, the inherited full native matrix, and Python regression**

Authenticate both frozen command owners:

```bash
set -euo pipefail
test "$(sha256sum docs/superpowers/plans/2026-07-16-g1-bounded-candidate-certification.md | cut -d' ' -f1)" = \
  e58873a2b6f30a7fefe7853e5b2cf7f2f15117fb7c71f5d9f0ba026170e08d89
test "$(sha256sum .superpowers/sdd/task-5-brief.md | cut -d' ' -f1)" = \
  cea15199d8b0f25f39b9ce9aeefd50262a632a61bcb98f401edf2a06d23f63a6
mkdir -p /tmp/g1-unscheduled-lazy-recovery/task1/full/provider
```

Run Task 1 Step 4 of the authenticated committed bounded-candidate plan's
complete strict-provider command block exactly, substituting only:

```text
out=/tmp/g1-unscheduled-lazy-recovery/task1/full/provider
```

This independently rechecks equal-frame provider behavior under strict and
fast callers, exact accelerated/exhaustive parity, sanitizer safety,
forbidden-fast compilation, production privacy, and the seam-negative owner.

Next run `.superpowers/sdd/task-5-brief.md` Step 4 exactly, substituting only:

```text
out=/tmp/g1-unscheduled-lazy-recovery/task1/full/task5-step4
```

Then run its Step 5 command block with a different output root so its leading
`rm -rf` cannot delete Step 4 evidence:

```text
out=/tmp/g1-unscheduled-lazy-recovery/task1/full/task5-step5
```

The pinned interpreter has no `pytest`. In that Step 5 block, replace exactly
the unavailable line

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests
```

with the repository's canonical available suite at the same position under
the same `set -e` owner:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v \
  2>&1 | tee \
  /tmp/g1-unscheduled-lazy-recovery/task1/full/task5-step5/python-unittest.log
```

Do not run the missing `pytest` command before or after this replacement. The
three separate roots must retain all provider, seam/privacy, and full-matrix
evidence. Together the blocks include:

- all 20 named strict and fast native programs;
- strict/fast footprint and IK transcript parity;
- production-positive recovery and IK links;
- all five intended compile-negative privacy contexts;
- no-seam controller preprocessing;
- production and benchmark `nm -C`/`strings` privacy checks;
- strict-object switch and no-LTO authentication; and
- all 386 Python `unittest` cases.

Finally reauthenticate immutable source and the exact sorted external terrain
package, including additions and deletions:

```bash
set -euo pipefail
sha256sum -c /tmp/g1-unscheduled-lazy-recovery/task1/immutable.before.sha256
terrain=/tmp/g1-terrain-footprint-runtime-v1
current=/tmp/g1-unscheduled-lazy-recovery/task1/full/terrain.current.sha256
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256 \
  "$current"
(
  cd "$terrain"
  sha256sum -c \
    /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256
)
git diff --check
```

Expected: all native programs pass in both flavors, intended negative compiles
fail only for their hidden symbols/types, production binaries contain no trace
or exhaustive-oracle surface, all 386 Python tests pass, immutable owners
authenticate, and `git diff --check` is silent.

- [ ] **Step 10: Review and commit only the five implementation paths**

Before committing, request an independent specification review against the
approved design and a separate code-quality review. Resolve every Critical or
Important finding test-first and rerun Steps 8–9 for any affected ownership
surface. Then:

```bash
git add controller.cpp g1_frame_transaction.h \
  g1_candidate_certification_trace.h \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 5
git diff --cached --name-only
git commit -m "fix: recover unscheduled incumbent rejection"
git status --short
```

Expected: exactly five implementation/test paths are committed; provider,
physics, terrain, database, log schema/checker, and visualizer files remain
unchanged.

---

### Task 2: Prove the Canonical Frame-17 Recovery Before Any Long Run

**Files:**

- Verify only: all Task 1 source/test paths
- Verify only: `/tmp/g1-terrain-footprint-runtime-v1`
- Generate only: `/tmp/g1-unscheduled-lazy-recovery/live32/`
- Do not modify or commit any repository file

**Gate rule:** Do not run 800 or 832 frames in this task. If the authentic
equal-frame request contains no dual-certified winner in its unchanged strict
tail, stop with exact trace evidence and return to design review. Do not change
capacity, search cadence, provider ranking, or physical thresholds.

- [ ] **Step 1: Build and authenticate disposable trace and release binaries**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
rm -rf "$root"
mkdir -p "$root/build" "$root/run18" "$root/run32"
terrain=/tmp/g1-terrain-footprint-runtime-v1
current="$root/terrain.before-live32.sha256"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256 \
  "$current"
(
  cd "$terrain"
  sha256sum -c \
    /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256
)
strict=(-std=c++17 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread \
  -ldl -lrt -lX11)

g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/build/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp \
  -o "$root/build/root-reach.o"
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

for object in clearance root-reach recovery-production recovery-seam; do
  readelf -p .GCC.command.line "$root/build/${object}.o" \
    > "$root/build/${object}.switches"
  rg -- '-fno-fast-math' "$root/build/${object}.switches"
  rg -- '-ffp-contract=off' "$root/build/${object}.switches"
  rg -- '-frounding-math' "$root/build/${object}.switches"
  ! rg -- '(^| )-ffast-math( |$)' "$root/build/${object}.switches"
  ! readelf -SW "$root/build/${object}.o" | rg '\.gnu\.lto'
done
! nm -C "$root/controller-release" | \
  rg 'CandidateTrace|CertificationTrace|exhaustive_for_test|ProviderAudit'
! strings "$root/controller-release" | \
  rg 'MM_CANDIDATE_TRACE|oracle_equal|recovery_traversals'
sha256sum "$root/controller-trace" "$root/controller-release" \
  > "$root/BINARIES.sha256"
```

Expected: strict owners have all strict flags and no fast/LTO flags; the trace
binary has both seams; the release binary exposes neither seam; no running
process was queried.

- [ ] **Step 2: Run paired 18-frame trace and release processes**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
for frames in 18 32; do
  dir="$root/run${frames}"
  for ik in 0 1; do
    DISPLAY=:1 MM_IK="$ik" \
      MM_CANDIDATE_TRACE="$dir/candidates-${ik}.tsv" \
      "$root/controller-trace" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene grail-curb-low --test-mode route \
      --test-route curb-forward --test-frames "$frames" \
      --test-heading forward --terrain-weight 4 \
      --log "$dir/trace-runtime-${ik}.csv"
    DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene grail-curb-low --test-mode route \
      --test-route curb-forward --test-frames "$frames" \
      --test-heading forward --terrain-weight 4 \
      --log "$dir/release-${ik}.csv"
    cmp "$dir/trace-runtime-${ik}.csv" "$dir/release-${ik}.csv"
  done
  cmp "$dir/candidates-0.tsv" "$dir/candidates-1.tsv"
done
```

Expected: release and trace builds produce byte-identical production CSVs in
each mode; IK-off and IK-on candidate traces are byte-identical for 18 and 32
frames.

- [ ] **Step 3: Authenticate the exact frame-17 boundary and create the marker**

Write no repository script. Run this assertion directly:

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
rm -f "$root/ZERO_REJECTION_32.PASS"
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv
import hashlib
import struct
from pathlib import Path
from resources.check_g1_runtime_log import (
    GATE_E_PAIR_INVARIANT_GROUPS,
    check_rows,
    read_rows,
)

root = Path('/tmp/g1-unscheduled-lazy-recovery/live32')
for frames in (18, 32):
    directory = root / f'run{frames}'
    off = read_rows(directory / 'release-0.csv')
    on = read_rows(directory / 'release-1.csv')
    assert len(off) == len(on) == frames
    check_rows(off)
    check_rows(on, allow_ik=True)
    for index, (left, right) in enumerate(zip(off, on)):
        assert left['frame_rejected'] == right['frame_rejected'] == '0'
        assert left['ik_safe_stop_latched'] == \
               right['ik_safe_stop_latched'] == '0'
        for _, names in GATE_E_PAIR_INVARIANT_GROUPS:
            for name in names:
                assert left[name] == right[name], \
                    (frames, index, name, left[name], right[name])

    with (directory / 'candidates-0.tsv').open(
            newline='', encoding='utf-8') as stream:
        trace = list(csv.DictReader(stream, delimiter='\t'))
    frame17 = [row for row in trace if row['presentation_frame'] == '17']
    assert 2 <= len(frame17) <= 7
    slot0 = frame17[0]
    assert slot0['candidate_slot'] == '0'
    assert slot0['candidate_kind'] == 'incumbent'
    assert slot0['score_owner'] == 'incumbent'
    assert slot0['selected_frame'] == '117711'
    assert slot0['executed_frame'] == '117712'
    assert slot0['source_range'] == '402'
    assert slot0['cost_bits_hex'] == '40f14831'
    assert slot0['attempted'] == '1'
    assert slot0['common'] == slot0['raw'] == 'accepted'
    assert slot0['ik'] == 'finite-rejected'
    assert slot0['rejection_stage'] == 'ik-candidate'
    assert slot0['stop_reason'] == 'no-swing-candidate'
    assert slot0['legacy_traversals'] == '0'
    assert slot0['recovery_provider_calls'] == '1'
    assert slot0['recovery_traversals'] == '1'
    assert all(row['oracle_equal'] == '1' for row in frame17)
    assert all(row['accelerated_count'] == row['exhaustive_count']
               for row in frame17)

    tail = frame17[1:]
    assert 1 <= len(tail) <= 6
    assert all(row['candidate_kind'] == 'strict-recovery' for row in tail)
    frames_seen = [row['selected_frame'] for row in tail]
    assert len(frames_seen) == len(set(frames_seen))
    assert '117711' not in frames_seen
    attempted = [row for row in tail if row['attempted'] == '1']
    accepted = [row for row in attempted
                if row['common'] == row['raw'] == row['ik'] == 'accepted']
    assert len(accepted) == 1
    winner = accepted[0]
    winner_slot = int(winner['candidate_slot'])
    assert winner['score_owner'] == 'strict-recovery'
    assert all(row['attempted'] == '1' for row in frame17[:winner_slot + 1])
    assert all(row['attempted'] == '0' for row in frame17[winner_slot + 1:])
    selected = off[17]['selected_database_frame']
    assert selected == on[17]['selected_database_frame']
    assert selected == winner['selected_frame']
    winner_cost_bits = int(winner['cost_bits_hex'], 16)
    f32_bits = lambda text: struct.unpack(
        '<I', struct.pack('<f', float(text)))[0]
    assert f32_bits(off[17]['selected_cost']) == winner_cost_bits
    assert f32_bits(on[17]['selected_cost']) == winner_cost_bits

    digest = hashlib.sha256(
        (directory / 'candidates-0.tsv').read_bytes()).hexdigest()
    (directory / 'TRACE.sha256').write_text(
        f'{digest}  candidates-0.tsv\n', encoding='ascii')
    (directory / 'FRAME17_WINNER.txt').write_text(
        f"slot={winner['candidate_slot']} "
        f"selected={winner['selected_frame']} "
        f"executed={winner['executed_frame']} "
        f"cost_bits={winner['cost_bits_hex']}\n",
        encoding='ascii')

(root / 'ZERO_REJECTION_32.PASS').write_text('PASS\n', encoding='ascii')
PY
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
```

Expected: the formerly failing unscheduled incumbent at presentation frame 17
authenticates exactly, the unchanged tail contains exactly one first
dual-certified winner, the public frame matches that winner in both modes, all
18/32 rows remain rejection-free and unlatch, and the marker is created only
after every assertion passes.

- [ ] **Step 4: Independently review the short live proof**

Have a fresh reviewer inspect the source diff, RED/GREEN evidence, both frame-17
traces, release/trace CSV parity, exact provider/exhaustive equality, and marker
creation logic. If the reviewer finds a code issue, return to Task 1 test-first
and invalidate this entire evidence directory. If the exact tail simply has no
winner, report `NEEDS_CONTEXT` with the authentic tail; do not start Task 3.

- [ ] **Step 5: Launch the reviewed candidate for the user's visual inspection**

This is a disposable inspection checkpoint, not packaging or replacement. It
launches only the already-hashed no-seam Task 2 release binary as a second
window. It never queries or modifies the old visualizer.

First reauthenticate the binary and external terrain package, then launch the
candidate directly and capture only the PID returned by that launch:

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
terrain=/tmp/g1-terrain-footprint-runtime-v1
current="$root/terrain.before-candidate-visualizer.sha256"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256 \
  "$current"
candidate=$(readlink -f "$root/controller-release")
test -x "$candidate"
stdout="$root/candidate-visualizer.stdout"
stderr="$root/candidate-visualizer.stderr"
test ! -e "$root/CANDIDATE_VISUALIZER.RUNNING"
rm -f "$root/CANDIDATE_VISUALIZER_CLOSED.PASS"
nohup env DISPLAY=:1 MM_IK=1 MM_STRAFE=1 \
  "$candidate" \
  --terrain-dir "$terrain" \
  --terrain-scene mixed-multilevel \
  --test-mode live --terrain-weight 4 \
  </dev/null >"$stdout" 2>"$stderr" &
candidate_pid=$!
test "$candidate_pid" -gt 0
candidate_start=''
for attempt in $(seq 1 100); do
  if test -r "/proc/$candidate_pid/stat"; then
    candidate_start=$(
      sed -E 's/^[0-9]+ \(.*\) //' "/proc/$candidate_pid/stat" | \
        awk '{print $20}'
    )
    test -n "$candidate_start" && break
  fi
  sleep 0.05
done
case "$candidate_start" in
  ''|*[!0-9]*) exit 1 ;;
esac
candidate_ready=false
for attempt in $(seq 1 100); do
  if test -r "/proc/$candidate_pid/stat" && \
     test -e "/proc/$candidate_pid/exe"; then
    observed_start=$(
      sed -E 's/^[0-9]+ \(.*\) //' "/proc/$candidate_pid/stat" | \
        awk '{print $20}'
    )
    observed_exe=$(readlink -f "/proc/$candidate_pid/exe" || true)
    if test "$observed_start" = "$candidate_start" && \
       test "$observed_exe" = "$candidate"; then
      candidate_ready=true
      break
    fi
  fi
  sleep 0.05
done
test "$candidate_ready" = true
sleep 2
kill -0 "$candidate_pid"

candidate_inode=$(stat -Lc '%d:%i' "/proc/$candidate_pid/exe")
candidate_cmdline=$(
  sha256sum "/proc/$candidate_pid/cmdline" | cut -d' ' -f1
)
test "$(sed -E 's/^[0-9]+ \(.*\) //' "/proc/$candidate_pid/stat" | awk '{print $20}')" = \
  "$candidate_start"
test "$(readlink -f "/proc/$candidate_pid/exe")" = "$candidate"
printf '%s\t%s\t%s\t%s\t%s\n' \
  "$candidate_pid" "$candidate_start" "$candidate" \
  "$candidate_inode" "$candidate_cmdline" \
  > "$root/CANDIDATE_VISUALIZER.identity"
printf 'RUNNING\n' > "$root/CANDIDATE_VISUALIZER.RUNNING"
```

Expected: a new interactive skeleton window opens on `mixed-multilevel` with
IK enabled and strafe/independent-heading control enabled. Tell the user this
is the new candidate window and wait for an explicit visual verdict. Do not
start Task 3 while the marker says `RUNNING`. Do not use `ps`, `pgrep`,
`pidof`, `xdotool`, a visualizer helper, or any window/process discovery.

- [ ] **Step 6: Close only the authenticated candidate after the user's verdict**

If the user reports a visible regression, save the exact report and return to
the relevant test-first task; do not enter long certification. If the user
accepts the checkpoint—or explicitly asks to continue—reauthenticate every
captured identity field before signaling only that candidate PID:

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
IFS=$'\t' read -r candidate_pid candidate_start candidate \
  candidate_inode candidate_cmdline \
  < "$root/CANDIDATE_VISUALIZER.identity"
case "$candidate_pid" in
  ''|*[!0-9]*) exit 1 ;;
esac
case "$candidate_start" in
  ''|*[!0-9]*) exit 1 ;;
esac

if test -e "/proc/$candidate_pid/exe"; then
  current_start=$(
    sed -E 's/^[0-9]+ \(.*\) //' "/proc/$candidate_pid/stat" | \
      awk '{print $20}'
  )
  current_exe=$(readlink -f "/proc/$candidate_pid/exe")
  current_inode=$(stat -Lc '%d:%i' "/proc/$candidate_pid/exe")
  current_cmdline=$(
    sha256sum "/proc/$candidate_pid/cmdline" | cut -d' ' -f1
  )
  test "$current_start" = "$candidate_start"
  test "$current_exe" = "$candidate"
  test "$current_inode" = "$candidate_inode"
  test "$current_cmdline" = "$candidate_cmdline"
  kill -TERM "$candidate_pid"
  for attempt in $(seq 1 100); do
    test ! -e "/proc/$candidate_pid" && break
    sleep 0.05
  done
  if test -e "/proc/$candidate_pid"; then
    test "$(sed -E 's/^[0-9]+ \(.*\) //' "/proc/$candidate_pid/stat" | awk '{print $20}')" = \
      "$candidate_start"
    test "$(readlink -f "/proc/$candidate_pid/exe")" = "$candidate"
    test "$(stat -Lc '%d:%i' "/proc/$candidate_pid/exe")" = \
      "$candidate_inode"
    test "$(sha256sum "/proc/$candidate_pid/cmdline" | cut -d' ' -f1)" = \
      "$candidate_cmdline"
    kill -KILL "$candidate_pid"
    for attempt in $(seq 1 100); do
      test ! -e "/proc/$candidate_pid" && break
      sleep 0.05
    done
  fi
fi
test ! -e "/proc/$candidate_pid"
rm -f "$root/CANDIDATE_VISUALIZER.RUNNING"
printf 'PASS\n' > "$root/CANDIDATE_VISUALIZER_CLOSED.PASS"

terrain=/tmp/g1-terrain-footprint-runtime-v1
current="$root/terrain.after-candidate-visualizer.sha256"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256 \
  "$current"
(
  cd "$terrain"
  sha256sum -c \
    /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256
)
```

Expected: only the directly launched candidate is closed, its exact identity
cannot have changed before the signal, the old visualizer was never queried or
touched, and the terrain/database package remains byte-identical. Long timing
and certification may now begin.

---

### Task 3: Restart the Complete Live Certification from Step 1

**Files:**

- Verify only: all committed source, tests, design, and this plan
- Verify only: `/tmp/g1-terrain-footprint-runtime-v1`
- Generate only: `/tmp/g1-bounded-candidate-cert/live/`
- Do not modify or commit any repository file

**Prerequisites:**

- `/tmp/g1-unscheduled-lazy-recovery/live32/ZERO_REJECTION_32.PASS` contains
  exactly `PASS`.
- `/tmp/g1-unscheduled-lazy-recovery/live32/CANDIDATE_VISUALIZER_CLOSED.PASS`
  contains exactly `PASS`, after the user's explicit visual verdict.
- Task 1 has passed independent specification and code-quality review.
- Task 2 has passed independent live-evidence review.

- [ ] **Step 1: Execute the frozen full-certification brief from the beginning**

Authenticate and then execute `.superpowers/sdd/task-6-brief.md` (SHA-256
`2bb19ccb6bb2c56950179f7139a78888c9c0c975da6d65841bf6b567cbe0946a`)
from its Step 1 through Step 5, literally and in order. It is the exact command
owner for:

1. three disposable strict/fast/no-LTO builds and production privacy;
2. a fresh paired 32-frame prerequisite and its own marker;
3. exact 800-row low-curb logs and three 832-transaction timing trials with 32
   warm-up transactions discarded;
4. the 9 normal, 4 stress, and 2 blocked Gate E matrix; and
5. all 30 Gate L cells, paired L2 ascent/descent, landing-exit stress,
   tangential multilevel traversal, and eight heading/travel-independent flat
   cases.

Before invoking its first command, run:

```bash
set -euo pipefail
test "$(cat /tmp/g1-unscheduled-lazy-recovery/live32/ZERO_REJECTION_32.PASS)" = PASS
test "$(cat /tmp/g1-unscheduled-lazy-recovery/live32/CANDIDATE_VISUALIZER_CLOSED.PASS)" = PASS
test ! -e /tmp/g1-unscheduled-lazy-recovery/live32/CANDIDATE_VISUALIZER.RUNNING
test "$(sha256sum .superpowers/sdd/task-6-brief.md | cut -d' ' -f1)" = \
  2bb19ccb6bb2c56950179f7139a78888c9c0c975da6d65841bf6b567cbe0946a
terrain=/tmp/g1-terrain-footprint-runtime-v1
current=/tmp/g1-unscheduled-lazy-recovery/terrain.before-full-cert.sha256
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256 \
  "$current"
git status --porcelain=v1 > /tmp/g1-unscheduled-lazy-recovery/pre-full-status.txt
git rev-parse HEAD > /tmp/g1-unscheduled-lazy-recovery/certification-head.txt
```

Do not reuse the failed Task 6 binaries or marker. Its Step 1 deliberately
removes `/tmp/g1-bounded-candidate-cert/live` and rebuilds everything. Do not
skip its short prerequisite merely because Task 2 passed.

Expected: every original behavior, performance, deterministic-pair, physical,
multilevel, lateral, tangent, slope/stair/edge, same-build oracle, and privacy
gate passes without a source or threshold change.

- [ ] **Step 2: Replace only the obsolete repository-scope closure**

The frozen brief's Step 6 expected-path list predates the approved repair and
Task 5's reviewed dependent test migrations. Run all of its Step 6 hash,
binary, immutable-file, current-frontier, asset, log, and evidence checks, but
replace only its old changed-path/commit-count block with this exact closure:

```bash
set -euo pipefail
root=/tmp/g1-bounded-candidate-cert/live
base=$(cat /tmp/g1-bounded-candidate-cert/base/execution-base.txt)
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
test "$(git rev-parse HEAD)" = \
  "$(cat /tmp/g1-unscheduled-lazy-recovery/certification-head.txt)"
terrain=/tmp/g1-terrain-footprint-runtime-v1
current="$root/terrain.after-full-cert.sha256"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256 \
  "$current"
(
  cd "$terrain"
  sha256sum -c \
    /tmp/g1-unscheduled-lazy-recovery/task1/terrain.before.sha256
)
git diff --name-only "$base"..HEAD | sort > "$root/changed-paths.txt"
expected="$root/expected-paths.txt"
printf '%s\n' \
  controller.cpp \
  docs/superpowers/plans/2026-07-17-g1-unscheduled-lazy-recovery.md \
  docs/superpowers/specs/2026-07-17-g1-unscheduled-lazy-recovery-design.md \
  g1_candidate_certification_trace.h \
  g1_candidate_recovery.cpp \
  g1_candidate_recovery.h \
  g1_frame_transaction.h \
  scene_switch.h \
  tests/cpp/compile_g1_candidate_recovery_production.cpp \
  tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp \
  tests/cpp/test_cleanup_runtime.cpp \
  tests/cpp/test_g1_candidate_audit_controller.cpp \
  tests/cpp/test_g1_candidate_recovery.cpp \
  tests/cpp/test_g1_controller_logging.cpp \
  tests/cpp/test_g1_controller_state.cpp \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp \
  tests/cpp/test_route_runtime.cpp \
  tests/cpp/test_scene_switch.cpp \
  tests/cpp/test_support_matching.cpp \
  tests/cpp/test_terrain_runtime.cpp \
  | sort > "$expected"
cmp "$expected" "$root/changed-paths.txt"
subjects="$root/expected-subjects.txt"
printf '%s\n' \
  'feat: add strict bounded recovery provider' \
  'feat: coordinate bounded dual candidate certification' \
  'feat: integrate bounded candidate certification' \
  'feat: publish bounded candidate outcomes atomically' \
  'test: trace bounded candidate certification' \
  'docs: design unscheduled lazy recovery' \
  'docs: plan unscheduled lazy recovery' \
  'fix: recover unscheduled incumbent rejection' \
  | sort > "$subjects"
git log --format=%s "$base"..HEAD | sort > "$root/actual-subjects.txt"
cmp "$subjects" "$root/actual-subjects.txt"
cmp /tmp/g1-unscheduled-lazy-recovery/pre-full-status.txt \
  <(git status --porcelain=v1)
git diff --check
```

Expected: the exact 21-path and eight-commit repair history is present; no
certification run changed the repository; immutable source/assets and all
binary/evidence hashes still authenticate.

- [ ] **Step 3: Final independent certification review and handoff**

A fresh reviewer must inspect:

- the canonical frame-17 winner and full accelerated/exhaustive equality;
- exact 800-row low-curb logs in both modes;
- all three timing summaries (`mean <= 40 ms`, nearest-rank `p99 <= 80 ms`);
- all Gate E and Gate L verdicts, including lateral, diagonal, half-support,
  slope/stair, tangent, landing-exit, and multilevel cases;
- release/trace production-CSV byte equality wherever required;
- normal-build privacy and immutable hashes;
- exact repository scope and unchanged worktree status; and
- absence of any interaction with the old visualizer.

Resolve no finding by weakening a gate. A production change invalidates every
live artifact owned by that change and restarts the affected certification from
Task 1 or Task 2 as appropriate.

Expected: a reviewer returns PASS with zero unresolved Critical or Important
findings. Only then is this repair certified and eligible for the separate
packaging/visualizer-replacement task. This plan itself does not launch,
inspect, stop, or replace the running visualizer and does not add the G1 mesh.

## Completion criteria

- The generic and production RED selectors fail before product edits for the
  intended missing unscheduled route, then pass after implementation.
- Scheduled matching, accepted unscheduled matching, and matching-disabled
  execution retain their existing behavior and provenance.
- There is still one production `database_search`, no search-period change,
  and no heading/travel coupling.
- Canonical frame 17 recovers with `legacy_traversals=0`, one provider call,
  exact accelerated/exhaustive equality, and a first strict dual-certified
  winner; neither mode rejects or latches in the paired 32-frame run.
- After that short gate, the user visually inspects the separately launched
  hashed no-seam candidate on `mixed-multilevel`; only its directly captured
  and reauthenticated PID is closed before performance certification.
- Strict, fast, sanitizer, privacy, full native, and Python regression matrices
  pass.
- Full low-curb performance, Gate E, and Gate L certification passes without a
  threshold, checker, terrain, database, or schema change.
- Repository scope is exactly the planned 21 paths and eight commits relative
  to the frozen execution base.
- The old visualizer remains entirely untouched until a later, separately
  reviewed packaging task.
