# G1 Best-Effort IK Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept the already-certified raw motion-matching pose when terrain IK alone finite-rejects, publish an authenticated `off`/`applied`/`skipped` cleanup disposition, and expose that disposition in the unchanged runtime log schema and visualizer.

**Architecture:** Preserve the existing common → raw → IK stage order and every physical certificate. Common or raw finite failures still enter bounded recovery; global errors still abort. Once raw is certified, an authenticated IK finite rejection becomes a terminal accepted candidate whose public pose and hidden IK state come from the raw branch. A certified IK branch remains the public owner only with IK enabled and remains the hidden-state owner in IK-off mode. Accepted diagnostics own the cleanup disposition/reason; CSV and overlay projection read that accepted owner rather than failed working storage.

**Tech Stack:** C++17; existing fixed-capacity frame transaction and recovery provider; strict-FP clearance/root-reach/recovery/trace objects; Raylib overlay; unchanged 305-column CSV; Python 3 runtime checker; exact float32 25 Hz route gate.

## Global Constraints

- Implement the frozen design at `docs/superpowers/specs/2026-07-17-g1-best-effort-ik-cleanup-design.md`; its required SHA-256 is `56792b9d954ec01011e45dfef6af19c71a7e1ce6158b9a6ddc8334938536af85`.
- Simulation cadence remains exact float32 `0.04` seconds (25 Hz).
- Terrain assets and motion data remain separate.
- Heading and travel direction remain independent.
- Do not change a footprint, clearance, contact, reach, landing-patch, swing, solver, terrain, motion-matching, or candidate-ranking threshold.
- Do not change candidate order, `K = 8`, recovery scoring, recovery capacity, database data, terrain scenes, route geometry, or sample rate.
- Common-stage and raw-certificate finite failures remain candidate failures. Recovery and exhausted-frame safe-stop behavior remain fail closed.
- A global error at any stage remains a global error and cannot be downgraded to a cleanup skip.
- An IK cleanup skip is legal only after the same candidate's raw branch fully certifies and the IK finite rejection is authenticated against its branch-local transaction with a non-`none` `G1IkStopReason`.
- A skipped frame publishes the exact raw public pose, does not call recovery, does not publish `frame_rejected`, does not latch a safe stop, and does not advance failed IK state.
- When IK is disabled, the public pose and disposition are `raw`/`off`. A successfully certified hidden IK branch may still advance the existing hidden `G1IkState`; a failed hidden branch may not.
- Keep the CSV header, column count, column order, and `motion_match_log.h` byte-for-byte unchanged. The checker change is grammar only.
- Accepted skipped rows use exactly: `ik_enabled=1`, `ik_applied=0`, `ik_safe_stop_requested=0`, authenticated non-`none` `ik_stop_reason`, `ik_candidate_rejected=1`, `frame_rejected=0`, and `ik_safe_stop_latched=0`.
- The overlay reads only `frame_runtime.accepted_diagnostic` and shows green `IK ACTIVE`, amber `IK SKIPPED — <reason>`, or gray `IK OFF`.
- Compile `g1_clearance.cpp`, `g1_ik_root_reach.cpp`, `g1_candidate_recovery.cpp`, and seam-only `g1_candidate_certification_trace.cpp` with `-fno-fast-math -ffp-contract=off -frounding-math`. Do not use LTO. Final links are neutral.
- Preserve unrelated work. Before every commit, stage only task-owned files, run `git diff --cached --check`, and inspect `git diff --cached --name-only`.
- Commit and push every coherent reviewed checkpoint immediately to `checkpoint/g1-footprint-task6`.
- Do not inspect, signal, terminate, or replace the currently running corrected-mesh visualizer. Short tests use disposable binaries; the final candidate is launched directly as a separate reviewed process.
- Keep disposable user-facing videos only in `/home/ubuntu/projects/mm_verifications/`; delete each after the user verifies it.

## File Structure

- Modify `g1_frame_transaction.h`: cleanup disposition type, accepted diagnostic ownership, authenticated skip outcome, raw/IK commit selection, hidden-state rules, equality/validity/success checks, and test-seam outcome grammar.
- Modify `tests/cpp/test_g1_frame_transaction.cpp`: generic RED/GREEN transaction cases for skip, recovery suppression, hidden-state isolation, off/applied projection, authentication, and unchanged failure behavior.
- Modify `controller.cpp`: production finalize owner, raw rendering for skips, accepted-diagnostic hash ownership, accepted-log projection, and three-state overlay.
- Modify `g1_candidate_certification_trace.cpp`: treat raw-accepted/IK-finite as unique terminal cleanup acceptance, never as a recovery-triggering finite failure.
- Modify `tests/cpp/test_g1_frame_transaction_production.cpp`: real-runner skip, trace grammar, raw publication, hidden-state, overlay source, and no-recovery coverage.
- Modify `tests/cpp/test_g1_controller_logging.cpp`: exact off/applied/skipped suffixes, diagnostic hash ownership, schema identity, and forged-combination guards.
- Modify `resources/check_g1_runtime_log.py`: authenticate the accepted skipped grammar while retaining all existing rejected-row and disabled-IK rules.
- Modify `tests/python/test_runtime_log.py`: accepted skipped positives and forged mixture negatives.
- Verify unchanged `motion_match_log.h`, all terrain/database assets, all IK/clearance thresholds, and the G1 mesh asset/export pipeline.

---

### Task 1: Implement Terminal Raw Acceptance and Its Visual Signal

**Files:**
- Modify: `g1_frame_transaction.h`
- Modify: `controller.cpp`
- Modify: `g1_candidate_certification_trace.cpp`
- Modify: `tests/cpp/test_g1_frame_transaction.cpp`
- Modify: `tests/cpp/test_g1_frame_transaction_production.cpp`
- Reference only: `docs/superpowers/specs/2026-07-17-g1-best-effort-ik-cleanup-design.md`

- [ ] **Step 1: Freeze the base and add a focused RED selector**

Run:

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task1
rm -rf "$root"
mkdir -p "$root"
test "$(sha256sum docs/superpowers/specs/2026-07-17-g1-best-effort-ik-cleanup-design.md | cut -d' ' -f1)" = \
  56792b9d954ec01011e45dfef6af19c71a7e1ce6158b9a6ddc8334938536af85
git rev-parse HEAD > "$root/base.txt"
git status --short > "$root/status.before"
sha256sum motion_match_log.h resources/database.bin \
  resources/g1_mesh/g1_raylib.glb > "$root/immutable.before.sha256"
```

In `tests/cpp/test_g1_frame_transaction.cpp`, add a `--best-effort-ik-red` selector and focused tests that express all of these failures before production edits:

1. raw-safe plus IK-finite returns `G1FrameTransactionAccepted`, publishes raw pose arrays, creates no rejection/latch, and records enabled cleanup as skipped with the exact authenticated reason;
2. the same attempt invokes the recovery provider zero times and runs no later candidate;
3. failed IK state poison does not reach `runtime.accepted_state.ik` or any public pose array;
4. missing, `none`, wrong-branch, or mutated IK rejection evidence returns global error;
5. raw finite failure still recovers/fails closed and IK global error still aborts atomically;
6. IK-off publishes raw with `off`; IK-on dual success publishes IK with `applied`;
7. accepted-diagnostic equality, validity, digest, dirty-storage reset, and hostile-finalize tests own both new fields.

In `tests/cpp/test_g1_frame_transaction_production.cpp`, add a
`--best-effort-ik-production-red` selector proving the same raw/IK ownership
with the real runner, plus terminal candidate-trace grammar and the three
accepted-state overlay labels. The production selector must fail on the old
source independently of the generic selector.

Compile the old source with the new tests:

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task1
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery.o"
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction.cpp -o "$root/frame-red.o"
g++ "$root/frame-red.o" "$root/clearance.o" "$root/root-reach.o" \
  "$root/recovery.o" -o "$root/frame-red"
if "$root/frame-red" --best-effort-ik-red \
    >"$root/red.stdout" 2>"$root/red.stderr"; then
  echo 'RED selector unexpectedly passed' >&2
  exit 1
fi
rg 'G1_BEST_EFFORT_IK_RED_SELECTED' "$root/red.stderr"

rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp -o "$root/trace-red.o"
g++ "${strict[@]}" "${rayinc[@]}" -DG1_CONTROLLER_NO_MAIN \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$root/controller-red.o"
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o "$root/production-red.o"
g++ "$root/production-red.o" "$root/controller-red.o" \
  "$root/trace-red.o" "$root/clearance.o" "$root/root-reach.o" \
  "$root/recovery.o" "${raylib[@]}" -o "$root/production-red"
if "$root/production-red" --best-effort-ik-production-red \
    >"$root/production-red.stdout" 2>"$root/production-red.stderr"; then
  echo 'production RED selector unexpectedly passed' >&2
  exit 1
fi
rg 'G1_BEST_EFFORT_IK_PRODUCTION_RED_SELECTED' \
  "$root/production-red.stderr"
```

Expected: both builds succeed; both selectors are proven active and fail
because current IK finite rejection still rejects/recovers rather than
accepting raw, and the current trace/UI have no terminal cleanup grammar.

- [ ] **Step 2: Add explicit accepted cleanup ownership**

In `g1_frame_transaction.h`:

- add a closed enum with exactly `off`, `applied`, and `skipped` values;
- add disposition and `G1IkStopReason` fields to `G1FrameAcceptedDiagnostic` and transaction scratch ownership needed to finalize a selected candidate;
- make unready diagnostic canonical zero/default storage;
- require `(off, none)` only when IK is disabled, `(applied, none)` only when IK is enabled and the visible IK certificate is complete, and `(skipped, non-none)` only when IK is enabled and the raw certificate is complete;
- include both fields in equality, validity, success matching, logical digests, poison/reset, and hostile-output tests.

Do not infer cleanup state from `ik_enabled`, `ik_frame.applied`, or mutable working storage after publication.

- [ ] **Step 3: Convert only authenticated IK finite rejection into acceptance**

Refactor candidate evaluation so outcomes distinguish:

- common/raw finite rejection;
- fully IK-certified acceptance;
- raw-certified/IK-finite cleanup acceptance;
- global error.

For raw-certified/IK-finite:

- authenticate `candidate_scratch.rejection` with `g1_frame_candidate_failure_is_authentic`;
- require `rejection_branch == G1FrameCertificateIk`, a legal IK rejection stage, and non-`none` reason;
- return terminal accepted outcome without retaining first failure or invoking recovery;
- preserve trace disposition as common=`accepted`, raw=`accepted`, IK=`finite-rejected` with exact stage/reason.

In the commit path:

- choose IK visible state only for enabled, fully certified `applied`;
- choose exact raw state for `off` and `skipped`;
- copy successful hidden IK state for dual-certified `applied` and `off`;
- retain the raw/baseline hidden IK state for an IK-finite cleanup acceptance in either enabled or disabled mode;
- finalize and validate the certificate corresponding to the visible state;
- publish canonical no-rejection state and stop before every recovery/later candidate.

In `G1FrameStageAcceptedFinalize`, select the raw certificate for `off` and
`skipped`, select the IK certificate for `applied`, snapshot `rendered` from
the public arrays already chosen by the coordinator, and fill the accepted
cleanup fields only from authenticated scratch ownership. Add those fields to
the controller's accepted-diagnostic logical hash.

In `g1_candidate_certification_trace.cpp`, classify common=`accepted`,
raw=`accepted`, IK=`finite-rejected` with canonical stage/reason as a unique
terminal cleanup acceptance. It increments the normal evaluation counters,
cannot request/call recovery, need not exhaust a tail, preserves the existing
serialized columns, and cannot be followed by another attempted candidate.
Common/raw finite attempts retain their existing recovery/exhaustion grammar.

In the overlay, read only `frame_runtime.accepted_diagnostic` and draw green
`IK ACTIVE`, amber `IK SKIPPED — <reason>`, or gray `IK OFF` directly below
`Transactional terrain IK`. Move existing help lines down without changing
bindings.

- [ ] **Step 4: Run focused and full generic GREEN in strict and fast callers**

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task1
for mode in strict fast; do
  if test "$mode" = strict; then
    flags=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
      -fno-fast-math -ffp-contract=off -frounding-math -I.)
  else
    flags=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
  fi
  g++ "${flags[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction.cpp \
    -o "$root/frame-${mode}.o"
  g++ "$root/frame-${mode}.o" "$root/clearance.o" \
    "$root/root-reach.o" "$root/recovery.o" \
    -o "$root/frame-${mode}"
  "$root/frame-${mode}" --best-effort-ik-red
  "$root/frame-${mode}"
done
git diff --check
```

Expected: focused selector and complete inherited suite pass in strict and fast callers; no recovery occurs after an IK-only finite failure; common/raw failures and global errors retain old behavior.

- [ ] **Step 5: Run production, trace, overlay, and sanitizer GREEN**

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task1
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp -o "$root/trace-owner.o"
for mode in strict fast; do
  if test "$mode" = strict; then
    flags=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
      -fno-fast-math -ffp-contract=off -frounding-math -I.)
  else
    flags=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
  fi
  g++ "${flags[@]}" "${rayinc[@]}" \
    -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$root/controller-${mode}.o"
  g++ "${flags[@]}" \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$root/production-${mode}.o"
  g++ "$root/production-${mode}.o" "$root/controller-${mode}.o" \
    "$root/trace-owner.o" "$root/clearance.o" \
    "$root/root-reach.o" "$root/recovery.o" \
    "${raylib[@]}" -o "$root/production-${mode}"
  "$root/production-${mode}" --best-effort-ik-production-red
  "$root/production-${mode}"
  "$root/production-${mode}" --trace-transcript \
    > "$root/trace-${mode}.tsv"
done
cmp "$root/trace-strict.tsv" "$root/trace-fast.tsv"

san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-divide-by-zero,float-cast-overflow \
  -fno-sanitize-recover=all -I.)
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp -o "$root/trace-owner-san.o"
g++ "${san[@]}" "${rayinc[@]}" \
  -DG1_CONTROLLER_NO_MAIN \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$root/controller-san.o"
g++ "${san[@]}" \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o "$root/production-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_clearance.cpp -o "$root/clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_ik_root_reach.cpp -o "$root/root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery-san.o"
g++ "${san[@]}" "$root/production-san.o" "$root/controller-san.o" \
  "$root/trace-owner-san.o" "$root/clearance-san.o" \
  "$root/root-reach-san.o" "$root/recovery-san.o" \
  "${raylib[@]}" -o "$root/production-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$root/production-san" --best-effort-ik-production-red
git diff --check
```

Expected: both focused selectors and inherited suites pass, trace transcripts
match byte-for-byte, sanitizer is silent, and source guards prove all three
labels read accepted diagnostics.

- [ ] **Step 6: Review, commit, and push Task 1**

Have a fresh task reviewer inspect the design conformance, both authentic RED
selectors, exact public/hidden owners, failure authentication, terminal trace
grammar, overlay ownership, and all generic/production regressions. Resolve
every Critical/Important finding and rerun Steps 4–5.

```bash
git add g1_frame_transaction.h controller.cpp \
  g1_candidate_certification_trace.cpp \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 5
git commit -m "feat: accept and display best-effort IK cleanup"
git push checkpoint HEAD:g1-footprint-task6
```

Expected staged paths: exactly the five Task 1 files; remote push succeeds.

---

### Task 2: Authenticate the Unchanged CSV Schema for Skipped Cleanup

**Files:**
- Modify: `controller.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/cpp/test_g1_controller_logging.cpp`
- Modify: `tests/python/test_runtime_log.py`
- Verify unchanged: `motion_match_log.h`

- [ ] **Step 1: Freeze the schema and write RED fixtures**

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task2
rm -rf "$root"
mkdir -p "$root"
sha256sum motion_match_log.h > "$root/schema.before.sha256"
python3 - <<'PY'
from resources.check_g1_runtime_log import RUNTIME_COLUMNS
assert len(RUNTIME_COLUMNS) == 305
PY
```

Add focused C++ and Python tests for:

- accepted `off`, `applied`, and `skipped` rows exactly matching the design table;
- skipped suffix starts from canonical disabled output and adds only the accepted reason, candidate-rejected flag, and authoritative horizontal simulation speed computed from the accepted raw state;
- skipped rows have canonical rejection fields, `frame_rejected=0`, and no latch;
- checker rejects skipped grammar with `ik_enabled=0`, `reason=none`, `ik_applied=1`, safe-stop requested, candidate-rejected=0, frame rejection/latch, invalid reason, or noncanonical failed-branch payload;
- checker still rejects an unsafe/raw rejected row relabeled as skipped;
- header text/hash, 305-column count, order, and `motion_match_log.h` hash do not change.

Add a `--best-effort-ik-logging-red` C++ selector and one focused Python test
named `RuntimeLogTests.test_best_effort_ik_accepted_grammar`. Before
checker/projection edits, run:

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task2
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/red-clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/red-root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/red-recovery.o"
g++ "${strict[@]}" "${rayinc[@]}" \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_controller_logging.cpp \
  -o "$root/logging-red.o"
g++ "$root/logging-red.o" "$root/red-clearance.o" \
  "$root/red-root-reach.o" "$root/red-recovery.o" \
  "${raylib[@]}" -o "$root/logging-red"
if "$root/logging-red" --best-effort-ik-logging-red \
    >"$root/cpp-red.stdout" 2>"$root/cpp-red.stderr"; then
  echo 'logging RED selector unexpectedly passed' >&2
  exit 1
fi
rg 'G1_BEST_EFFORT_IK_LOGGING_RED_SELECTED' "$root/cpp-red.stderr"
if /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
    tests.python.test_runtime_log.RuntimeLogTests.test_best_effort_ik_accepted_grammar \
    >"$root/python-red.stdout" 2>"$root/python-red.stderr"; then
  echo 'checker RED test unexpectedly passed' >&2
  exit 1
fi
rg 'accepted IK stop reason is not none|accepted IK candidate is marked rejected' \
  "$root/python-red.stdout" "$root/python-red.stderr"
```

Expected: both selectors are active and fail because the old projection and
accepted-row checker reject the approved skipped combination.

- [ ] **Step 2: Project accepted suffix from the disposition owner**

Change `g1_build_task7_log_suffix`/its helper so it authenticates `accepted_diagnostic` and emits:

- off: exact canonical disabled suffix;
- applied: existing fully certified accepted IK suffix;
- skipped: canonical non-applied suffix with `safe_stop_requested=0`, diagnostic stop reason, and `candidate_rejected=1`.

Never read failed IK candidate arrays, failed hidden state, or transient rejection scratch for a skipped accepted row.

- [ ] **Step 3: Update only the checker grammar**

Refactor `_check_accepted_status_schema` into explicit off/applied/skipped accepted combinations. Require the skipped combination to be IK-enabled, non-applied, non-safe-stop, candidate-rejected, non-rejected/unlatched, and to use a legal non-`none` IK stop reason. Retain `_check_disabled_ik_is_canonical`, `_check_rejection_schema`, all thresholds, and every other gate unchanged.

- [ ] **Step 4: Run C++ strict/fast, focused/full Python, sanitizer, and schema gates**

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task2
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery.o"
for mode in strict fast; do
  if test "$mode" = strict; then
    flags=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
      -fno-fast-math -ffp-contract=off -frounding-math -I.)
  else
    flags=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
  fi
  g++ "${flags[@]}" "${rayinc[@]}" \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_controller_logging.cpp \
    -o "$root/logging-${mode}.o"
  g++ "$root/logging-${mode}.o" "$root/clearance.o" \
    "$root/root-reach.o" "$root/recovery.o" "${raylib[@]}" \
    -o "$root/logging-${mode}"
  "$root/logging-${mode}"
done

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v 2>&1 | tee "$root/runtime-log-tests.txt"
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v \
  2>&1 | tee "$root/python-all.txt"
rg '^OK$' "$root/runtime-log-tests.txt"
rg '^OK$' "$root/python-all.txt"
sha256sum -c "$root/schema.before.sha256"
python3 - <<'PY'
from resources.check_g1_runtime_log import RUNTIME_COLUMNS
assert len(RUNTIME_COLUMNS) == 305
PY
git diff --check
```

Run the exact sanitizer block:

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/task2
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-divide-by-zero,float-cast-overflow \
  -fno-sanitize-recover=all -I.)
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_clearance.cpp -o "$root/clearance-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -c g1_ik_root_reach.cpp -o "$root/root-reach-san.o"
g++ "${san[@]}" -fno-fast-math -ffp-contract=off -frounding-math \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery-san.o"
g++ "${san[@]}" "${rayinc[@]}" \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_controller_logging.cpp \
  -o "$root/logging-san.o"
g++ "${san[@]}" "$root/logging-san.o" "$root/clearance-san.o" \
  "$root/root-reach-san.o" "$root/recovery-san.o" \
  "${raylib[@]}" -o "$root/logging-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$root/logging-san" --best-effort-ik-logging-red
```

Expected: all authentic rows pass, every forged mixture fails for the intended
reason, full Python is green, sanitizer is silent, and schema hash/count are
unchanged.

- [ ] **Step 5: Review, commit, and push Task 2**

```bash
git add controller.cpp resources/check_g1_runtime_log.py \
  tests/cpp/test_g1_controller_logging.cpp \
  tests/python/test_runtime_log.py
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 4
git commit -m "feat: authenticate skipped IK cleanup logs"
git push checkpoint HEAD:g1-footprint-task6
```

Expected: fresh review reports no Critical/Important finding; schema/header is absent from the staged paths; push succeeds.

---

### Task 3: Prove the 32-Frame Terrain Gate and Launch a Visual Candidate

**Files:**
- Generate only: `/tmp/g1-best-effort-ik-cleanup/gate32/`
- Generate only after gate: `/home/ubuntu/projects/mm_verifications/g1_best_effort_ik_mixed_multilevel.mp4`
- Do not modify or commit repository files in this task

- [ ] **Step 1: Run the complete focused/full source verification**

Rerun Tasks 1–2 strict/fast/full tests, then:

```bash
set -euo pipefail
sha256sum -c /tmp/g1-best-effort-ik-cleanup/task1/immutable.before.sha256
git diff --check
test -z "$(git status --short)"
git fetch checkpoint g1-footprint-task6
test "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)"
test "$(sha256sum docs/superpowers/specs/2026-07-17-g1-best-effort-ik-cleanup-design.md | cut -d' ' -f1)" = \
  56792b9d954ec01011e45dfef6af19c71a7e1ce6158b9a6ddc8334938536af85
```

- [ ] **Step 2: Build disposable seam and no-seam binaries**

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/gate32
rm -rf "$root"
mkdir -p "$root/build"
strict=(-std=c++17 -O3 -DNDEBUG -fno-fast-math \
  -ffp-contract=off -frounding-math -frecord-gcc-switches -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/build/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp \
  -o "$root/build/root-reach.o"
g++ "${strict[@]}" -c g1_candidate_recovery.cpp \
  -o "$root/build/recovery-production.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/build/recovery-seam.o"
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp \
  -o "$root/build/trace-owner.o"
g++ "${fast[@]}" "${rayinc[@]}" \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$root/build/controller-trace.o"
g++ "$root/build/controller-trace.o" "$root/build/trace-owner.o" \
  "$root/build/clearance.o" "$root/build/root-reach.o" \
  "$root/build/recovery-seam.o" "${raylib[@]}" \
  -o "$root/controller-trace"
g++ "${fast[@]}" "${rayinc[@]}" -c controller.cpp \
  -o "$root/build/controller-release.o"
g++ "$root/build/controller-release.o" "$root/build/clearance.o" \
  "$root/build/root-reach.o" "$root/build/recovery-production.o" \
  "${raylib[@]}" -o "$root/controller-release"
for object in clearance root-reach recovery-production recovery-seam \
    trace-owner; do
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
  rg 'MM_CANDIDATE_TRACE|oracle_equal'
sha256sum "$root/controller-trace" "$root/controller-release" \
  > "$root/BINARIES.sha256"
```

Expected: both binaries link neutrally without LTO; all four physical owners
and the trace owner record strict switches; only the seam binary exposes the
trace/oracle surface.

- [ ] **Step 3: Run paired 32-frame exact-25-Hz mixed-multilevel routes**

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/gate32
rm -f "$root/BEST_EFFORT_IK_32.PASS"
unset MM_G1_MESH_PATH
for ik in 0 1; do
  DISPLAY=:1 MM_IK="$ik" \
    MM_CANDIDATE_TRACE="$root/candidates-${ik}.tsv" \
    "$root/controller-trace" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene mixed-multilevel --test-mode route \
    --test-route full-course --test-frames 32 \
    --test-heading forward --terrain-weight 4 \
    --log "$root/trace-${ik}.csv"
  DISPLAY=:1 MM_IK="$ik" "$root/controller-release" \
    --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
    --terrain-scene mixed-multilevel --test-mode route \
    --test-route full-course --test-frames 32 \
    --test-heading forward --terrain-weight 4 \
    --log "$root/release-${ik}.csv"
  cmp "$root/trace-${ik}.csv" "$root/release-${ik}.csv"
done
```

Authenticate the exact approved behavior and create the marker only after all
assertions pass:

```bash
cmp "$root/candidates-0.tsv" "$root/candidates-1.tsv"
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv
import struct
from pathlib import Path
from resources.check_g1_runtime_log import check_rows, read_rows

root = Path('/tmp/g1-best-effort-ik-cleanup/gate32')
off = read_rows(root / 'release-0.csv')
on = read_rows(root / 'release-1.csv')
assert len(off) == len(on) == 32
check_rows(off)
check_rows(on, allow_ik=True)
f32 = lambda text: struct.pack('<f', float(text))
dt = struct.pack('<f', 0.04)

for index, row in enumerate(off):
    assert f32(row['fixed_dt']) == dt
    assert row['ik_enabled'] == '0'
    assert row['ik_applied'] == '0'
    assert row['ik_stop_reason'] == 'none'
    assert row['ik_candidate_rejected'] == '0'
    assert row['frame_rejected'] == '0'
    assert row['ik_safe_stop_latched'] == '0'

skipped = []
for index, row in enumerate(on):
    assert f32(row['fixed_dt']) == dt
    assert row['ik_enabled'] == '1'
    assert row['frame_rejected'] == '0'
    assert row['ik_safe_stop_latched'] == '0'
    applied = row['ik_applied'] == '1'
    is_skip = (row['ik_applied'] == '0' and
               row['ik_safe_stop_requested'] == '0' and
               row['ik_stop_reason'] != 'none' and
               row['ik_candidate_rejected'] == '1')
    assert applied != is_skip, (index, row['ik_stop_reason'])
    if applied:
        assert row['ik_stop_reason'] == 'none'
        assert row['ik_candidate_rejected'] == '0'
    else:
        skipped.append(index)
        assert float(row['rendered_min_clearance']) >= -0.01

for index in (19, 23, 25, 30, 31):
    row = on[index]
    assert row['frame_rejected'] == '0'
    assert row['ik_applied'] == '1' or index in skipped

with (root / 'candidates-1.tsv').open(
        newline='', encoding='utf-8') as stream:
    trace = list(csv.DictReader(stream, delimiter='\t'))

def frame_rows(frame):
    return [row for row in trace
            if int(row['presentation_frame']) == frame]

frame18 = frame_rows(18)
attempted18 = [row for row in frame18 if row['attempted'] == '1']
assert len(attempted18) >= 2
assert attempted18[0]['raw'] == 'finite-rejected'
winner18 = attempted18[-1]
assert winner18['common'] == winner18['raw'] == 'accepted'
assert winner18['ik'] in ('accepted', 'finite-rejected')
assert on[18]['selected_database_frame'] == winner18['selected_frame']
assert winner18['selected_frame'] != attempted18[0]['selected_frame']

for frame in skipped:
    rows = frame_rows(frame)
    attempted = [row for row in rows if row['attempted'] == '1']
    assert attempted
    terminal = attempted[-1]
    assert terminal['common'] == terminal['raw'] == 'accepted'
    assert terminal['ik'] == 'finite-rejected'
    assert terminal['stop_reason'] == on[frame]['ik_stop_reason']
    terminal_slot = int(terminal['candidate_slot'])
    assert all(row['attempted'] == '0'
               for row in rows[terminal_slot + 1:])
    if terminal_slot == 0:
        assert terminal['recovery_provider_calls'] == '0'

(root / 'BEST_EFFORT_IK_32.PASS').write_text(
    'PASS\n', encoding='ascii')
PY
test "$(cat "$root/BEST_EFFORT_IK_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
```

Expected: exactly 32 exact-25-Hz rows per mode; no rejection/latch; frame 18
recovers instead of publishing its unsafe raw slot zero; frames 19/23/25/30/31
apply or explicitly skip; every skip is authenticated and terminal; release
and seam CSVs are byte-identical.

- [ ] **Step 4: Independent gate review**

Have a fresh reviewer inspect source diffs, RED/GREEN evidence, both 32-row CSVs, trace authentication, frame 18 recovery, frames 19/23/25/30/31 cleanup outcomes, release/seam parity, binary flags, unchanged schema, and marker creation. Any code finding invalidates the marker and returns to the owning task.

- [ ] **Step 5: Launch the reviewed IK-on visualizer and make one disposable video**

Without querying or touching the currently running process, launch the exact hashed no-seam gate binary as a separate candidate:

```bash
set -euo pipefail
root=/tmp/g1-best-effort-ik-cleanup/gate32
test "$(cat "$root/BEST_EFFORT_IK_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
mkdir -p /home/ubuntu/projects/mm_verifications
unset MM_G1_MESH_PATH
nohup env DISPLAY=:1 MM_IK=1 MM_STRAFE=1 \
  "$root/controller-release" \
  --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
  --terrain-scene mixed-multilevel --test-mode live --terrain-weight 4 \
  >"$root/candidate.stdout" 2>"$root/candidate.stderr" </dev/null &
printf '%s\n' "$!" > "$root/candidate.pid"
sleep 2
geometry=$(DISPLAY=:1 xdpyinfo | awk '/dimensions:/{print $2; exit}')
test -n "$geometry"
ffmpeg -y -f x11grab -framerate 25 -video_size "$geometry" \
  -i :1.0+0,0 -t 12 -c:v libx264 -preset veryfast -crf 22 \
  -pix_fmt yuv420p \
  /home/ubuntu/projects/mm_verifications/g1_best_effort_ik_mixed_multilevel.mp4
test -s /home/ubuntu/projects/mm_verifications/g1_best_effort_ik_mixed_multilevel.mp4
```

Capture a short verification video to `/home/ubuntu/projects/mm_verifications/g1_best_effort_ik_mixed_multilevel.mp4` showing the G1 mesh and visible ACTIVE/SKIPPED state. Keep only that one new verification artifact and delete it after the user approves it.

- [ ] **Step 6: Resume bounded-candidate Task 6 only after this marker**

Do not run any 800/832-frame command before `BEST_EFFORT_IK_32.PASS` exists and is independently reviewed. Reconcile the long certification against the new reviewed HEAD, compile/link the strict out-of-line trace owner only into the seam binary, authenticate the current checker/mesh hashes, unset `MM_G1_MESH_PATH` for disposable runs, and audit historical reviewed commit ranges separately as recorded by the Task 6 reconciliation.
