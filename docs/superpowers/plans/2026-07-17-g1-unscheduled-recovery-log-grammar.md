# G1 Unscheduled Recovery Log Grammar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the canonical production-log checker the exact matching-enabled unscheduled-recovery grammar without changing the controller, 305-column schema, or motion behavior.

**Architecture:** Keep `searched` bound to the one scheduled legacy `database_search` and `transitioned` bound to the accepted frame change. Use the already-authenticated `matching_enabled` field to distinguish recovery-capable unscheduled rows from matching-disabled rows, require strict cost improvement for unscheduled recovery, and retain the existing four-ULP exception only for scheduled legacy transitions. C++ logging tests construct the authentic production row; Python tests own the row grammar and rejected-baseline transition count.

**Tech Stack:** C++17, Python 3.12 `unittest`, GCC strict/fast/ASan+UBSan callers, existing Raylib link dependencies, exact CSV/TSV evidence.

## Global Constraints

- Exact runtime rate remains 25 Hz (`fixed_dt` binary32 `0.04`).
- Modify exactly `resources/check_g1_runtime_log.py`, `tests/python/test_runtime_log.py`, and `tests/cpp/test_g1_controller_logging.cpp`.
- `motion_match_log.h`, `controller.cpp`, `g1_frame_transaction.h`, provider/trace owners, terrain/database assets, thresholds, and CSV column order remain byte-unchanged.
- `searched=1` means scheduled legacy `database_search`; never set it for strict recovery.
- An unscheduled recovery is exactly `matching_enabled=1`, `searched=0`, `transitioned=1`, with `selected_cost < incumbent_cost`.
- Matching-disabled rows require `searched=0` and `transitioned=0`.
- A rejected publication may repeat the last accepted transition bit, but it is not counted as a new transition event.
- Do not weaken selected/query/current/range, cost, finite-value, rejection, or lifecycle checks.
- Do not inspect, discover, signal, replace, or restart any existing visualizer or process. Never use `ps`, `pgrep`, `pidof`, `pkill`, `systemctl`, window discovery, or a visualizer helper.
- This plan does not launch a visualizer. The known frame-18 rejection/latch remains a separate runtime blocker.

---

### Task 1: Implement the Mode-Aware Recovery Grammar Test-First

**Files:**

- Modify: `tests/python/test_runtime_log.py`
- Modify: `tests/cpp/test_g1_controller_logging.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Verify only: `motion_match_log.h`
- Verify only: `controller.cpp`
- Verify only: `g1_frame_transaction.h`
- Verify only: `g1_candidate_recovery.h`
- Verify only: `g1_candidate_recovery.cpp`
- Verify only: `g1_candidate_certification_trace.h`
- Verify only: `g1_candidate_certification_trace.cpp`

**Interfaces:**

- Consumes `check_rows(rows, *, allow_ik=False)` and its existing return object.
- Produces the unchanged `check_rows` signature with corrected recovery grammar and accepted-transition count.
- Produces no production ABI, CSV, controller, provider, or trace change.

- [ ] **Step 1: Authenticate the design/base and freeze non-owned sources**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-recovery-log-grammar
rm -rf "$root"
mkdir -p "$root/red" "$root/green" "$root/full"
test "$(git rev-parse HEAD^)" = c69d097a566283fac26720c401eef9aac1375b1c
test "$(git show -s --format=%s HEAD)" = \
  'docs: plan unscheduled recovery log grammar'
spec=docs/superpowers/specs/2026-07-17-g1-unscheduled-recovery-log-grammar-design.md
test "$(sha256sum "$spec" | cut -d' ' -f1)" = \
  c19e7890678a64876bd9f80130f56e1cc383631e214c788658ef6e07887dc2f3
git diff --exit-code
sha256sum controller.cpp g1_frame_transaction.h \
  g1_candidate_recovery.h g1_candidate_recovery.cpp \
  g1_candidate_certification_trace.h g1_candidate_certification_trace.cpp \
  motion_match_log.h > "$root/immutable.before.sha256"
(
  cd /tmp/g1-terrain-footprint-runtime-v1
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --
) > "$root/terrain.before.sha256"
test "$(sha256sum /tmp/g1-unscheduled-lazy-recovery/live32/controller-release | cut -d' ' -f1)" = \
  6873c8d4239e4b1f682122cdb69161e21a11b34e43522d45f886b5f9b2fd548d
```

Expected: clean reviewed base; exact design; immutable production owners,
terrain package, and release evidence authenticate.

- [ ] **Step 2: Extend existing Python tests to create RED without changing the 386-test count**

In `RuntimeLogTests.test_accepts_selected_frame_and_source_range_for_transition`,
retain the scheduled case and add:

```python
        unscheduled = check_rows([row(
            0, 21, matching_enabled=1, searched=0, transitioned=1,
            query_database_frame=10, query_range=0,
            selected_database_frame=20, source_range=1, range=1,
            incumbent_cost=2.0, selected_cost=1.0,
        )])
        self.assertEqual(unscheduled["transitions"], 1)

        for searched, transitioned in ((1, 0), (0, 1), (1, 1)):
            values = row(
                0, 21, matching_enabled=0,
                searched=searched, transitioned=transitioned,
                query_database_frame=10,
                selected_database_frame=(20 if transitioned else 10),
                source_range=(1 if transitioned else 0),
                range=(1 if transitioned else 0),
                incumbent_cost=2.0,
                selected_cost=(1.0 if transitioned else 2.0),
            )
            with self.subTest(
                    searched=searched, transitioned=transitioned):
                with self.assertRaisesRegex(
                        ValueError, "matching is disabled"):
                    check_rows([values])
```

In `test_rejects_transition_that_does_not_beat_incumbent`, retain the existing
searched rejection and add:

```python
        for selected in (
                1.0,
                float32_offset(1.0, 1),
                float32_offset(1.0, 4)):
            values = row(
                0, 21, matching_enabled=1, searched=0, transitioned=1,
                query_database_frame=10, query_range=0,
                selected_database_frame=20, source_range=1, range=1,
                incumbent_cost=1.0, selected_cost=selected,
            )
            with self.subTest(unscheduled_selected=selected):
                with self.assertRaisesRegex(
                        ValueError,
                        "unscheduled recovery did not strictly beat incumbent"):
                    check_rows([values])
```

In `test_rejected_transition_freezes_accepted_query_cursor`, strengthen the
scheduled baseline with:

```python
        self.assertEqual(
            check_rows(rows, allow_ik=True)["transitions"], 1)
```

Then add an unscheduled copy inside the same method:

```python
        unscheduled = [dict(item) for item in rows]
        for item in unscheduled[1:]:
            item["matching_enabled"] = "1"
            item["searched"] = "0"
        self.assertEqual(
            check_rows(unscheduled, allow_ik=True),
            {"frames": 3, "transitions": 1})
```

Do not change
`test_accepts_transition_cost_within_four_ulps_of_incumbent`; it proves the
legacy four-ULP allowance remains scheduled-only.

- [ ] **Step 3: Add the authentic C++ producer-to-log RED**

Replace `test_log_schema_and_runtime_checker_hashes_are_unchanged` with a
schema-only hash lock:

```cpp
static void test_log_schema_hash_is_unchanged()
{
    std::string observed;
    char error[512] = {};
    logging_check(
        sha256_file_hex(
            observed,
            "motion_match_log.h",
            error,
            static_cast<int>(sizeof(error))),
        error);
    logging_check(
        observed ==
            "470dcd5978fe6f03e9398dc845c5454f70d5a6f1958e54014dc0eda69d9c5586",
        "the 305-column production log schema remains byte unchanged");
}
```

Add this test after `build_bounded_row`:

```cpp
static void test_unscheduled_recovery_row_uses_mode_aware_checker_grammar()
{
    fixture value;
    configure_real_candidate_fixture(value, true, false);
    value.external.input.presentation_frame = 0;
    value.external.tuning.ik_enabled = false;
    value.runtime.accepted_state.search_timer =
        value.runtime.accepted_state.search_time;
    value.db.contact_states(RealIncumbentExecutedFrame, 0) = false;
    value.db.contact_states(RealIncumbentExecutedFrame, 1) = true;
    real_unscheduled_provider_calls = 0;
    g1_frame_recovery_request_reset(real_unscheduled_provider_request);

    G1CandidateCertificationTrace trace;
    char error[1024] = {};
    logging_check(
        run_real_candidate_fixture(
            value,
            trace,
            real_unscheduled_recording_provider,
            nullptr,
            error,
            static_cast<int>(sizeof(error))) ==
            G1FrameTransactionAccepted,
        error);
    logging_check(
        trace.attempt_count >= 2U &&
            trace.legacy_traversals == 0U &&
            trace.recovery_provider_calls == 1U &&
            trace.attempts[0].candidate.kind == G1CandidateIncumbent,
        "authentic unscheduled fixture reaches strict recovery");
    const G1CandidateAttemptTraceRecord& winner =
        trace.attempts[trace.attempt_count - 1U];
    logging_check(
        winner.candidate.kind == G1CandidateRecoveryTransition &&
            winner.score_owner == G1CandidateScoreStrictRecovery &&
            winner.common == G1CandidateDispositionAccepted &&
            winner.raw == G1CandidateDispositionAccepted &&
            winner.ik == G1CandidateDispositionAccepted,
        "authentic unscheduled fixture dual-certifies its strict winner");

    motion_match_log_row row;
    logging_check(
        build_bounded_row(
            row, value, false, error, static_cast<int>(sizeof(error))),
        error);
    logging_check(
        row.matching_enabled && !row.searched && row.transitioned &&
            row.selected_database_frame == winner.candidate.selected_frame &&
            row.database_frame == winner.candidate.executed_frame &&
            terrain_float_bits(row.selected_cost) ==
                terrain_float_bits(winner.candidate.selected_cost) &&
            row.selected_cost < row.incumbent_cost,
        "production row preserves exact unscheduled strict winner provenance");

    const std::string checker =
        read_source_file("resources/check_g1_runtime_log.py");
    logging_check(
        checker.find("unscheduled_recovery = (") != std::string::npos &&
            checker.find(
                "if (searched or transitioned) and not matching_enabled:") !=
                std::string::npos &&
            checker.find(
                "unscheduled recovery did not strictly beat incumbent") !=
                std::string::npos,
        "canonical checker owns the exact mode-aware recovery grammar");
}
```

Replace the old main call and add the new one:

```cpp
    test_log_schema_hash_is_unchanged();
    test_unscheduled_recovery_row_uses_mode_aware_checker_grammar();
```

- [ ] **Step 4: Run focused RED and preserve intended failures**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-recovery-log-grammar
python=/home/ubuntu/miniconda3/envs/diffsim/bin/python
set +e
"$python" -m unittest -v \
  tests.python.test_runtime_log.RuntimeLogTests.test_accepts_selected_frame_and_source_range_for_transition \
  tests.python.test_runtime_log.RuntimeLogTests.test_rejects_transition_that_does_not_beat_incumbent \
  tests.python.test_runtime_log.RuntimeLogTests.test_rejected_transition_freezes_accepted_query_cursor \
  >"$root/red/python.stdout" 2>"$root/red/python.stderr"
python_status=$?
set -e
test "$python_status" -ne 0
rg 'transition without search|matching is disabled|transitions' \
  "$root/red/python.stdout" "$root/red/python.stderr"

strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/red/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/red/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/red/recovery.o"
g++ "${strict[@]}" "${rayinc[@]}" \
  -c tests/cpp/test_g1_controller_logging.cpp -o "$root/red/logging.o"
g++ "$root/red/logging.o" "$root/red/clearance.o" \
  "$root/red/root-reach.o" "$root/red/recovery.o" "${raylib[@]}" \
  -o "$root/red/logging"
set +e
"$root/red/logging" >"$root/red/logging.stdout" 2>"$root/red/logging.stderr"
logging_status=$?
set -e
test "$logging_status" -ne 0
rg 'canonical checker owns the exact mode-aware recovery grammar' \
  "$root/red/logging.stderr"
sha256sum tests/python/test_runtime_log.py \
  tests/cpp/test_g1_controller_logging.cpp > "$root/red/test-state.sha256"
```

Expected: Python fails on the obsolete implication/count; C++ fails only on
the absent checker grammar.

- [ ] **Step 5: Implement the minimal checker correction**

In `resources/check_g1_runtime_log.py::check_rows`, add before the row loop:

```python
    accepted_transitions = 0
```

Immediately after parsing `searched` and `transitioned`, add:

```python
        matching_enabled = _integer(row, "matching_enabled", index)
```

Replace the unconditional `transitioned and not searched` rejection with:

```python
        if matching_enabled not in (0, 1):
            raise ValueError(
                f"row {index}: matching_enabled must be 0 or 1")
        if (searched or transitioned) and not matching_enabled:
            raise ValueError(
                f"row {index}: search or transition while matching is disabled")
        unscheduled_recovery = (
            matching_enabled == 1 and searched == 0 and transitioned == 1)
```

Replace the transitioned-cost branch with:

```python
        if transitioned and not selected < incumbent:
            if unscheduled_recovery:
                raise ValueError(
                    f"row {index}: unscheduled recovery did not strictly "
                    "beat incumbent cost")
            cost_ulps = _float32_ulp_distance(selected, incumbent, index)
            if cost_ulps > 4:
                raise ValueError(
                    f"row {index}: transition did not beat incumbent cost "
                    f"and differs by {cost_ulps} float32 ULPs")
```

After full validation of each row, count only accepted transitions:

```python
        if transitioned and not row_rejected_for_lifecycle:
            accepted_transitions += 1
```

Return it:

```python
    return {
        "frames": len(rows),
        "transitions": accepted_transitions,
    }
```

Do not change any schema/type constant.

- [ ] **Step 6: Run focused GREEN in Python and strict/fast C++ callers**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-recovery-log-grammar
python=/home/ubuntu/miniconda3/envs/diffsim/bin/python
"$python" -m unittest -v \
  tests.python.test_runtime_log.RuntimeLogTests.test_accepts_selected_frame_and_source_range_for_transition \
  tests.python.test_runtime_log.RuntimeLogTests.test_rejects_transition_that_does_not_beat_incumbent \
  tests.python.test_runtime_log.RuntimeLogTests.test_accepts_transition_cost_within_four_ulps_of_incumbent \
  tests.python.test_runtime_log.RuntimeLogTests.test_rejected_transition_freezes_accepted_query_cursor \
  2>&1 | tee "$root/green/python-focused.log"

strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
for mode in strict fast; do
  if test "$mode" = strict; then
    flags=("${strict[@]}")
  else
    flags=("${fast[@]}")
  fi
  out="$root/green/$mode"
  mkdir -p "$out"
  g++ "${flags[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
  g++ "${flags[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
  g++ "${flags[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c g1_candidate_recovery.cpp -o "$out/recovery.o"
  g++ "${flags[@]}" "${rayinc[@]}" \
    -c tests/cpp/test_g1_controller_logging.cpp -o "$out/logging.o"
  g++ "$out/logging.o" "$out/clearance.o" "$out/root-reach.o" \
    "$out/recovery.o" "${raylib[@]}" -o "$out/logging"
  "$out/logging"
done
```

Expected: all focused tests and both authentic C++ callers pass.

- [ ] **Step 7: Run sanitizer, full Python, immutable, and hygiene gates**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-recovery-log-grammar
out="$root/full"
mkdir -p "$out"
san=(-std=c++17 -O1 -g -Wall -Wextra -Werror -pedantic \
  -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-divide-by-zero,float-cast-overflow \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${san[@]}" -c g1_clearance.cpp -o "$out/clearance-san.o"
g++ "${san[@]}" -c g1_ik_root_reach.cpp -o "$out/root-reach-san.o"
g++ "${san[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$out/recovery-san.o"
g++ "${san[@]}" "${rayinc[@]}" \
  -c tests/cpp/test_g1_controller_logging.cpp -o "$out/logging-san.o"
g++ "${san[@]}" "$out/logging-san.o" "$out/clearance-san.o" \
  "$out/root-reach-san.o" "$out/recovery-san.o" "${raylib[@]}" \
  -o "$out/logging-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$out/logging-san"

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v \
  2>&1 | tee "$out/python-unittest.log"
rg '^Ran 386 tests in ' "$out/python-unittest.log"
rg '^OK$' "$out/python-unittest.log"

sha256sum -c "$root/immutable.before.sha256"
(
  cd /tmp/g1-terrain-footprint-runtime-v1
  sha256sum -c "$root/terrain.before.sha256"
)
test "$(python3 - <<'PY'
from resources.check_g1_runtime_log import GATE_A_COLUMNS, RUNTIME_COLUMNS
print(len(GATE_A_COLUMNS), len(RUNTIME_COLUMNS))
PY
)" = '62 305'
git diff --check
test -z "$(git diff --name-only -- controller.cpp g1_frame_transaction.h \
  g1_candidate_recovery.h g1_candidate_recovery.cpp \
  g1_candidate_certification_trace.h g1_candidate_certification_trace.cpp \
  motion_match_log.h)"
```

Expected: sanitizer silent; all 386 tests pass; exact 62/305 schema width and
immutable source/assets remain unchanged.

- [ ] **Step 8: Write the evidence report and commit exactly three paths**

Write ignored report
`.superpowers/sdd/unscheduled-recovery-log-grammar-task-1-report.md` with RED
statuses/hashes, focused strict/fast results, sanitizer result, full Python
count/runtime/log SHA, immutable/terrain results, exact checker SHA, unchanged
schema width and log-header SHA, and confirmation of no visualizer/process
operation.

```bash
set -euo pipefail
git diff --check
git add resources/check_g1_runtime_log.py \
  tests/python/test_runtime_log.py \
  tests/cpp/test_g1_controller_logging.cpp
test "$(git diff --cached --name-only | LC_ALL=C sort)" = \
"resources/check_g1_runtime_log.py
tests/cpp/test_g1_controller_logging.cpp
tests/python/test_runtime_log.py"
git diff --cached --check
git commit -m "fix: validate unscheduled recovery log grammar"
git status --porcelain=v1
```

Expected: one exact three-path implementation commit and a clean tracked tree.

- [ ] **Step 9: Independent review**

The reviewer authenticates the design/plan, RED evidence, exact three-path
scope, unchanged 305-column schema, authentic producer row, complete flag/cost
grammar, rejected-baseline transition count, focused/full/sanitizer results,
immutable owners, and absence of visualizer/process interaction.

Critical or Important findings block Task 2 and are fixed test-first before a
fresh review. The required verdict is explicit PASS/BLOCK with Critical,
Important, and Minor counts.

---

### Task 2: Re-run Captured Evidence and Isolate the Runtime Liveness Blocker

**Files:**

- Create ignored report: `.superpowers/sdd/unscheduled-recovery-log-grammar-task-2-report.md`
- Verify only: `/tmp/g1-unscheduled-lazy-recovery/live32/**`
- Modify no source.

**Interfaces:**

- Consumes reviewed `check_rows` from Task 1.
- Consumes byte-authenticated binaries/evidence built from unchanged production source.
- Produces frame-17 checker/recovery PASS and precise frame-18 runtime BLOCK.
- Produces no launch marker.

- [ ] **Step 1: Reauthenticate unchanged production binaries and pair evidence**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
git diff --exit-code 2f61b2691273227bad31eec0b791ec277d6184f0..HEAD -- \
  controller.cpp g1_frame_transaction.h g1_candidate_recovery.h \
  g1_candidate_recovery.cpp g1_candidate_certification_trace.h \
  g1_candidate_certification_trace.cpp motion_match_log.h
test "$(sha256sum "$root/controller-release" | cut -d' ' -f1)" = \
  6873c8d4239e4b1f682122cdb69161e21a11b34e43522d45f886b5f9b2fd548d
sha256sum -c "$root/BINARIES.sha256"
cmp "$root/run18/trace-runtime-0.csv" "$root/run18/release-0.csv"
cmp "$root/run18/trace-runtime-1.csv" "$root/run18/release-1.csv"
cmp "$root/run32/trace-runtime-0.csv" "$root/run32/release-0.csv"
cmp "$root/run32/trace-runtime-1.csv" "$root/run32/release-1.csv"
cmp "$root/run18/candidates-0.tsv" "$root/run18/candidates-1.tsv"
cmp "$root/run32/candidates-0.tsv" "$root/run32/candidates-1.tsv"
```

Expected: production sources, binaries, and all six pairs remain exact.

- [ ] **Step 2: Run repaired checker and exact frame-17 assertions**

```bash
set -euo pipefail
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv
import struct
from pathlib import Path
from resources.check_g1_runtime_log import check_rows

root = Path('/tmp/g1-unscheduled-lazy-recovery/live32')
for frames in (18, 32):
    directory = root / f'run{frames}'
    runtime = {}
    for ik in (0, 1):
        with (directory / f'release-{ik}.csv').open(newline='') as stream:
            runtime[ik] = list(csv.DictReader(stream))
        assert check_rows(runtime[ik], allow_ik=bool(ik))['frames'] == frames
    with (directory / 'candidates-0.tsv').open(newline='') as stream:
        trace = list(csv.DictReader(stream, delimiter='\t'))
    frame17 = [row for row in trace if row['presentation_frame'] == '17']
    assert 2 <= len(frame17) <= 7
    incumbent = frame17[0]
    assert incumbent['candidate_slot'] == '0'
    assert incumbent['candidate_kind'] == incumbent['score_owner'] == 'incumbent'
    assert incumbent['selected_frame'] == '117711'
    assert incumbent['executed_frame'] == '117712'
    assert incumbent['source_range'] == '402'
    assert incumbent['cost_bits_hex'] == '40f14831'
    assert incumbent['attempted'] == '1'
    assert incumbent['common'] == incumbent['raw'] == 'accepted'
    assert incumbent['ik'] == 'finite-rejected'
    assert incumbent['rejection_stage'] == 'ik-candidate'
    assert incumbent['stop_reason'] == 'no-swing-candidate'
    assert incumbent['legacy_traversals'] == '0'
    assert incumbent['recovery_provider_calls'] == '1'
    assert incumbent['recovery_traversals'] == '1'
    assert all(row['oracle_equal'] == '1' for row in frame17)
    assert all(row['accelerated_count'] == row['exhaustive_count']
               for row in frame17)
    tail = frame17[1:]
    attempted = [row for row in tail if row['attempted'] == '1']
    winners = [row for row in attempted
               if row['common'] == row['raw'] == row['ik'] == 'accepted']
    assert len(winners) == 1 and attempted[-1] is winners[0]
    winner = winners[0]
    public = runtime[0][17]
    assert public['matching_enabled'] == '1'
    assert public['searched'] == '0'
    assert public['transitioned'] == '1'
    assert public['selected_database_frame'] == winner['selected_frame']
    assert public['database_frame'] == winner['executed_frame']
    assert struct.pack('>f', float(public['selected_cost'])).hex() == \
        winner['cost_bits_hex']
    assert runtime[1][17]['selected_database_frame'] == winner['selected_frame']
print('FRAME17_CHECKER_AND_RECOVERY_PASS')
PY
```

Expected: `FRAME17_CHECKER_AND_RECOVERY_PASS`.

- [ ] **Step 3: Prove the separate frame-18 liveness blocker**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
set +e
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY' \
  >"$root/LIVENESS.stdout" 2>"$root/LIVENESS.stderr"
import csv
from pathlib import Path

root = Path('/tmp/g1-unscheduled-lazy-recovery/live32')
for frames in (18, 32):
    for ik in (0, 1):
        path = root / f'run{frames}' / f'release-{ik}.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            if row['frame_rejected'] != '0' or row['ik_safe_stop_latched'] != '0':
                raise ValueError(
                    f'{path}: row {index}: rejection={row["frame_rejected"]} '
                    f'stage={row["frame_rejection_stage"]} '
                    f'latch={row["ik_safe_stop_latched"]}')
PY
liveness_status=$?
set -e
test "$liveness_status" -ne 0
rg 'row 18: rejection=1 stage=pose-certificate latch=1' \
  "$root/LIVENESS.stderr"
test ! -e "$root/ZERO_REJECTION_32.PASS"
test ! -e "$root/CANDIDATE_VISUALIZER.RUNNING"
```

Expected: exact runtime BLOCK at row 18, with no PASS or launch marker.

- [ ] **Step 4: Report and hand off to the separate liveness design**

Write
`.superpowers/sdd/unscheduled-recovery-log-grammar-task-2-report.md` with
unchanged production/binary authentication, all six pair hashes, repaired
checker results, exact frame-17 proof, exact first frame-18 rejection/latch and
zero-candidate-tail evidence, and no visualizer/process interaction.

Final verdict: `CHECKER_PASS_RUNTIME_BLOCK`. The next design starts from frame
18 having a physically invalid incumbent continuation and an empty strict
recovery set (`accelerated_count=0`, `exhaustive_count=0`).
