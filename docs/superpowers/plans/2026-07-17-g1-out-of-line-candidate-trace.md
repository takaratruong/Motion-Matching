# Out-of-Line Candidate Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the fast no-seam controller bit-for-bit while moving candidate-trace validation, serialization, and oracle work into a strict out-of-line object so fast trace and release runs emit byte-identical production CSVs.

**Architecture:** Keep the seam-only trace file type and three function declarations in `g1_candidate_certification_trace.h`. Move the exact existing function bodies into a new strictly compiled `g1_candidate_certification_trace.cpp`; the trace executable links that object, while the release executable neither compiles nor links it. `controller.cpp` and every production math/control path remain unchanged.

**Tech Stack:** C++17, GCC 13, raylib 5.0, Bash, Python 3.12 `unittest`, ASan/UBSan, ELF `nm`/`readelf`/`strings`, Git.

## Global Constraints

- Exact runtime rate remains 25 Hz.
- Preserve `-ffast-math` behavior for the no-seam release controller.
- Do not change matcher cadence, thresholds, candidate capacity, heading/travel independence, terrain/database assets, or production CSV schema.
- Do not relax byte equality or omit query, selected-cost, or accepted-state fields.
- The trace process keeps the same loaded-database accelerated/exhaustive oracle; the release exposes no trace/oracle symbol or string.
- Do not inspect, discover, query, signal, replace, restart, or otherwise touch the existing visualizer. Never use `ps`, `pgrep`, `pidof`, `pkill`, `systemctl`, or a visualizer helper.
- All live processes use unique disposable paths directly on `DISPLAY=:1`.
- Stage and commit only task-owned paths. Preserve unrelated work.

## File map

- Modify `g1_candidate_certification_trace.h`: seam-only interface and `G1CandidateTraceFile`; no function body.
- Create `g1_candidate_certification_trace.cpp`: strict trace implementation, validation, exhaustive oracle, fixed-buffer TSV writer, close logic.
- Modify `tests/cpp/test_g1_frame_transaction_production.cpp`: focused RED selector and exact source/linkage ownership guard.
- Verify only `controller.cpp`, `g1_frame_transaction.h`, `g1_candidate_recovery.h/.cpp`, log/checker owners, and all terrain/database assets.

---

### Task 1: Split Candidate Trace Behind a Strict Out-of-Line Boundary

**Files:**

- Modify: `tests/cpp/test_g1_frame_transaction_production.cpp`
- Modify: `g1_candidate_certification_trace.h`
- Create: `g1_candidate_certification_trace.cpp`
- Verify only: `controller.cpp`
- Verify only: `g1_frame_transaction.h`
- Verify only: `g1_candidate_recovery.h`
- Verify only: `g1_candidate_recovery.cpp`

**Interfaces:**

- Consumes existing exact signatures:

```cpp
bool g1_candidate_trace_open(
    G1CandidateTraceFile& file,
    const char* path,
    char* error,
    int error_capacity);

bool g1_candidate_trace_append_after_transaction(
    G1CandidateTraceFile& file,
    uint32_t presentation_frame,
    const G1CandidateCertificationTrace& trace,
    char* error,
    int error_capacity);

void g1_candidate_trace_close(G1CandidateTraceFile& file);
```

- Produces one strictly compiled implementation object that satisfies those
  external declarations. No production API or runtime state changes.

- [ ] **Step 1: Authenticate the approved base and preserve live RED evidence**

Run from the worktree root:

```bash
set -euo pipefail
root=/tmp/g1-out-of-line-candidate-trace
rm -rf "$root"
mkdir -p "$root/red" "$root/green" "$root/full"
plan=docs/superpowers/plans/2026-07-17-g1-out-of-line-candidate-trace.md
plan_commit=$(git log -1 --format=%H -- "$plan")
test -n "$plan_commit"
test "$(git show -s --format=%s "$plan_commit")" = \
  'docs: plan out-of-line candidate trace'
test "$(git rev-parse HEAD)" = "$plan_commit"
test "$(git hash-object "$plan")" = \
  "$(git rev-parse "$plan_commit:$plan")"
test "$(sha256sum docs/superpowers/specs/2026-07-17-g1-out-of-line-candidate-trace-design.md | cut -d' ' -f1)" = \
  e64d4173bdc57da1cd600686c3272aa91cc939c16db8722232a3d40dae1466ea
git diff --exit-code -- controller.cpp g1_frame_transaction.h \
  g1_candidate_recovery.h g1_candidate_recovery.cpp \
  g1_candidate_certification_trace.h \
  tests/cpp/test_g1_frame_transaction_production.cpp
test "$(sha256sum /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/release-1.csv | cut -d' ' -f1)" = \
  22a1566608e11e9c567b3358b25c9a49b6c7d6b772b834581280155959a91be6
test "$(sha256sum /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/trace-enabled.csv | cut -d' ' -f1)" = \
  34395a7199195382bf9f9e0f4993a2a42ca2a52b1168fb92d5a36292bb9c4760
! cmp /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/release-1.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/trace-enabled.csv
cmp /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/release-1.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/release-2.csv
cmp /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/trace-enabled.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/parity-probe/trace-disabled.csv
cmp /tmp/g1-unscheduled-lazy-recovery/live32/macro-probe/none.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/macro-probe/frame.csv
cmp /tmp/g1-unscheduled-lazy-recovery/live32/macro-probe/none.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/macro-probe/recovery.csv
! cmp /tmp/g1-unscheduled-lazy-recovery/live32/macro-probe/none.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/macro-probe/both.csv
cmp /tmp/g1-unscheduled-lazy-recovery/live32/strict-parity-probe/none.csv \
  /tmp/g1-unscheduled-lazy-recovery/live32/strict-parity-probe/both.csv
sha256sum controller.cpp g1_frame_transaction.h \
  g1_candidate_recovery.h g1_candidate_recovery.cpp \
  motion_match_log.h resources/check_g1_runtime_log.py \
  > "$root/immutable.before.sha256"
(
  cd /tmp/g1-terrain-footprint-runtime-v1
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --
) > "$root/terrain.before.sha256"
```

Expected: the committed repair/spec authenticate; fast release and trace are
individually deterministic but cross-build different; either macro alone is
neutral; strict cross-build output is identical; immutable owners are hashed.

- [ ] **Step 2: Write the focused source/linkage RED test**

In `tests/cpp/test_g1_frame_transaction_production.cpp`, add this function in
the dual-seam trace-test section:

```cpp
static void test_candidate_trace_implementation_is_out_of_line()
{
    const std::string interface_source =
        read_source_file("g1_candidate_certification_trace.h");
    check(interface_source.find(
              "static inline bool g1_candidate_trace_open") ==
              std::string::npos &&
          interface_source.find(
              "static inline bool "
              "g1_candidate_trace_append_after_transaction") ==
              std::string::npos &&
          interface_source.find(
              "static inline void g1_candidate_trace_close") ==
              std::string::npos,
          "candidate trace implementation is out of line from the fast controller owner");

    const std::string implementation_source =
        read_source_file("g1_candidate_certification_trace.cpp");
    check(implementation_source.find(
              "bool g1_candidate_trace_open(") != std::string::npos &&
          implementation_source.find(
              "bool g1_candidate_trace_append_after_transaction(") !=
              std::string::npos &&
          implementation_source.find(
              "void g1_candidate_trace_close(") != std::string::npos &&
          implementation_source.find(
              "g1_recovery_candidates_exhaustive_for_test(") !=
              std::string::npos,
          "strict candidate trace owner contains all definitions and the same-build oracle");

    const std::string controller_source =
        read_source_file("controller.cpp");
    check(controller_source.find(
              "#include \"g1_candidate_certification_trace.h\"") !=
              std::string::npos &&
          controller_source.find(
              "g1_candidate_certification_trace.cpp") ==
              std::string::npos,
          "fast controller consumes only the candidate trace interface");
}
```

Add a focused runner:

```cpp
static void run_candidate_trace_linkage_red_tests()
{
    test_candidate_trace_implementation_is_out_of_line();
}
```

Add this selector before `--trace-transcript`:

```cpp
if (argc == 2 &&
    std::strcmp(argv[1], "--candidate-trace-linkage-red") == 0) {
    std::fputs("G1_CANDIDATE_TRACE_LINKAGE_RED_SELECTED\n", stderr);
    run_candidate_trace_linkage_red_tests();
    return 0;
}
```

Also call `test_candidate_trace_implementation_is_out_of_line()` from
`run_candidate_trace_tests()` so the ordinary suite retains the guard.

- [ ] **Step 3: Run RED in strict and fast callers before product edits**

```bash
set -euo pipefail
root=/tmp/g1-out-of-line-candidate-trace/red
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery-seam.o"
for flavor in strict fast; do
  flags=("${strict[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" "${rayinc[@]}" \
    -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$root/controller-$flavor.o"
  g++ "${flags[@]}" \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$root/test-$flavor.o"
  g++ "$root/test-$flavor.o" "$root/controller-$flavor.o" \
    "$root/clearance.o" "$root/root-reach.o" "$root/recovery-seam.o" \
    "${raylib[@]}" -o "$root/test-$flavor"
  if "$root/test-$flavor" --candidate-trace-linkage-red \
      >"$root/$flavor.stdout" 2>"$root/$flavor.stderr"; then
    echo "ERROR: $flavor linkage RED unexpectedly passed" >&2
    exit 1
  fi
  test "$(rg -c '^G1_CANDIDATE_TRACE_LINKAGE_RED_SELECTED$' \
      "$root/$flavor.stderr")" -eq 1
  rg 'candidate trace implementation is out of line from the fast controller owner' \
    "$root/$flavor.stderr"
done
sha256sum "$root/strict.stderr" "$root/fast.stderr" \
  > "$root/RED.sha256"
```

Expected: both executables build and fail only on the unique new ownership
assertion; no product path has changed.

- [ ] **Step 4: Move the exact implementation out of line**

Replace `g1_candidate_certification_trace.h` with this interface shape under
its existing dual-seam conditional:

```cpp
#pragma once

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && \
    defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)

#include "g1_frame_transaction.h"

#include <cstdint>
#include <cstdio>

struct G1CandidateTraceFile
{
    std::FILE* stream = nullptr;
    bool header_written = false;
};

bool g1_candidate_trace_open(
    G1CandidateTraceFile& file,
    const char* path,
    char* error,
    int error_capacity);

bool g1_candidate_trace_append_after_transaction(
    G1CandidateTraceFile& file,
    uint32_t presentation_frame,
    const G1CandidateCertificationTrace& trace,
    char* error,
    int error_capacity);

void g1_candidate_trace_close(G1CandidateTraceFile& file);

#endif
```

Create `g1_candidate_certification_trace.cpp` with this preamble:

```cpp
#if !defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) || \
    !defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
#error "candidate trace implementation requires both test seams"
#endif

#include "g1_candidate_certification_trace.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
```

After that preamble, move the three function bodies from committed
`d9d58d9:g1_candidate_certification_trace.h` exactly, in order, changing only:

```text
static inline bool g1_candidate_trace_open                  -> bool g1_candidate_trace_open
static inline bool g1_candidate_trace_append_after_transaction -> bool g1_candidate_trace_append_after_transaction
static inline void g1_candidate_trace_close                 -> void g1_candidate_trace_close
```

Do not change a validation condition, message, field order, format string,
oracle call, buffer size, write, flush, reset, or close operation. Do not edit
`controller.cpp`.

Update the two existing source-ownership checks in
`tests/cpp/test_g1_frame_transaction_production.cpp` to follow the moved owner:

```cpp
const std::string implementation_source =
    read_source_file("g1_candidate_certification_trace.cpp");
const std::size_t oracle_call = implementation_source.find(
    "g1_recovery_candidates_exhaustive_for_test(");
check(oracle_call != std::string::npos &&
          implementation_source.find(
              "trace.recovery_request", oracle_call) !=
              std::string::npos,
      "the same-build oracle call receives trace.recovery_request directly");
```

In `test_trace_append_is_after_completed_transaction()`, replace
`trace_header_source` with:

```cpp
const std::string trace_implementation_source =
    read_source_file("g1_candidate_certification_trace.cpp");
```

Use `trace_implementation_source` for the two existing
`g1_frame_rejection_stage_name` call-count checks and the existing absence
checks for the duplicated stage mapping and `rejection_stage_text`. Do not
change their expected counts or messages.

- [ ] **Step 5: Build the strict owner and run focused GREEN**

```bash
set -euo pipefail
root=/tmp/g1-out-of-line-candidate-trace/green
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math \
  -frecord-gcc-switches -I.)
fast=(-std=c++17 -O3 -ffast-math -DNDEBUG -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${strict[@]}" -c g1_clearance.cpp -o "$root/clearance.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$root/root-reach.o"
g++ "${strict[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery-seam.o"
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp -o "$root/trace-owner.o"
readelf -p .GCC.command.line "$root/trace-owner.o" > "$root/trace-owner.switches"
rg -- '-fno-fast-math' "$root/trace-owner.switches"
rg -- '-ffp-contract=off' "$root/trace-owner.switches"
rg -- '-frounding-math' "$root/trace-owner.switches"
! rg -- '(^| )-ffast-math( |$)' "$root/trace-owner.switches"
! readelf -SW "$root/trace-owner.o" | rg '\.gnu\.lto'

for flavor in strict fast; do
  flags=("${strict[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" "${rayinc[@]}" \
    -DG1_CONTROLLER_NO_MAIN \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c controller.cpp -o "$root/controller-$flavor.o"
  g++ "${flags[@]}" \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "$root/test-$flavor.o"
  g++ "$root/test-$flavor.o" "$root/controller-$flavor.o" \
    "$root/clearance.o" "$root/root-reach.o" "$root/recovery-seam.o" \
    "$root/trace-owner.o" "${raylib[@]}" -o "$root/test-$flavor"
  "$root/test-$flavor" --candidate-trace-linkage-red
  "$root/test-$flavor" --unscheduled-trace-red
  "$root/test-$flavor" --trace-transcript \
    > "$root/transcript-$flavor.tsv"
  "$root/test-$flavor"
done
cmp "$root/transcript-strict.tsv" "$root/transcript-fast.tsv"
test "$(sha256sum "$root/transcript-strict.tsv" | cut -d' ' -f1)" = \
  1ff4504e24d43225217ae7a885ca3484bbfaccbb57a2c5c3780497ea9c273fe7
```

Expected: strict ownership flags authenticate; focused and ordinary suites
pass in both callers; frozen scheduled transcript bytes remain exact.

- [ ] **Step 6: Run sanitizer, privacy, full native, and Python gates**

Execute this exact full native block first:

```bash
set -euo pipefail
out=/tmp/g1-out-of-line-candidate-trace/full/native
mkdir -p "$out"
strict=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
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
g++ "${strict[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp -o "$out/trace-owner.o"

for flavor in strict fast; do
  flags=("${strict[@]}")
  test "$flavor" = fast && flags=("${fast[@]}")
  g++ "${flags[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
    -c tests/cpp/test_g1_candidate_recovery.cpp \
    -o "$out/provider-$flavor.o"
  g++ "$out/provider-$flavor.o" "$out/recovery-seam.o" \
    -o "$out/provider-$flavor"
  "$out/provider-$flavor" --parity > "$out/provider-$flavor.txt"
done
cmp "$out/provider-strict.txt" "$out/provider-fast.txt"

plain=(
  cleanup_runtime g1_candidate_audit g1_clearance g1_command_runtime
  g1_controller_state g1_footprint_runtime g1_frame_transaction
  g1_skeleton motion_match_log route_runtime scene_runtime scene_switch
  support_matching support_runtime terrain_database terrain_runtime
)
embedded_controller=(g1_candidate_audit_controller g1_controller_logging)

for flavor in strict fast; do
  flags=("${strict[@]}")
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
    "$out/root-reach.o" "$out/recovery-seam.o" \
    "$out/trace-owner.o" "${raylib[@]}" \
    -o "$out/frame-production-${flavor}"
  "$out/frame-production-${flavor}"
done

cmp "$out/footprint-strict.txt" "$out/footprint-fast.txt"
cmp "$out/ik-strict.txt" "$out/ik-fast.txt"

g++ "${strict[@]}" \
  -c tests/cpp/compile_g1_candidate_recovery_production.cpp \
  -o "$out/recovery-positive.o"
g++ "$out/recovery-positive.o" "$out/recovery.o" \
  -o "$out/recovery-positive"
"$out/recovery-positive"
g++ "${strict[@]}" -c tests/cpp/compile_g1_ik_production.cpp \
  -o "$out/ik-positive.o"
g++ "$out/ik-positive.o" "$out/clearance.o" "$out/root-reach.o" \
  -o "$out/ik-positive"
"$out/ik-positive"

if g++ "${strict[@]}" \
    -c tests/cpp/compile_g1_candidate_recovery_seam_negative.cpp \
    -o "$out/recovery-negative.o" 2>"$out/recovery-negative.stderr"; then
  echo 'ERROR: recovery seam negative compiled' >&2
  exit 1
fi
rg 'g1_recovery_candidates_exhaustive_for_test|not declared' \
  "$out/recovery-negative.stderr"
if g++ "${strict[@]}" -c tests/cpp/compile_g1_ik_seam_negative.cpp \
    -o "$out/ik-negative.o" 2>"$out/ik-negative.stderr"; then
  echo 'ERROR: IK seam negative compiled' >&2
  exit 1
fi
rg 'for_test|not declared' "$out/ik-negative.stderr"
for mode in VOID_CONTEXT PUBLICATION_CONTEXT ACCEPTED_CONTEXT; do
  if g++ "${strict[@]}" -D"G1_FRAME_NEGATIVE_${mode}" \
      -c tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
      -o "$out/runner-negative-${mode}.o" \
      2>"$out/runner-negative-${mode}.stderr"; then
    echo "ERROR: forbidden ${mode} runner context compiled" >&2
    exit 1
  fi
  rg 'G1FrameStageRunner|convert|conversion|argument' \
    "$out/runner-negative-${mode}.stderr"
done
```

Then run the sanitizer, strict-owner negative, Python, and immutable blocks:

```bash
set -euo pipefail
root=/tmp/g1-out-of-line-candidate-trace/full
san=(-std=c++17 -O1 -g -Wall -Wextra -Werror -pedantic \
  -fno-omit-frame-pointer -fsanitize=address,undefined,float-divide-by-zero,float-cast-overflow \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
rayinc=(-I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src)
raylib=(-L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11)
g++ "${san[@]}" -c g1_clearance.cpp -o "$root/clearance-san.o"
g++ "${san[@]}" -c g1_ik_root_reach.cpp -o "$root/root-reach-san.o"
g++ "${san[@]}" -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_recovery.cpp -o "$root/recovery-san.o"
g++ "${san[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp -o "$root/trace-owner-san.o"
g++ "${san[@]}" "${rayinc[@]}" -DG1_CONTROLLER_NO_MAIN \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$root/controller-san.o"
g++ "${san[@]}" -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o "$root/test-san.o"
g++ "${san[@]}" "$root/test-san.o" "$root/controller-san.o" \
  "$root/clearance-san.o" "$root/root-reach-san.o" \
  "$root/recovery-san.o" "$root/trace-owner-san.o" \
  "${raylib[@]}" -o "$root/test-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$root/test-san" --candidate-trace-linkage-red
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$root/test-san" --unscheduled-trace-red
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$root/test-san" --trace-transcript > "$root/transcript-san.tsv"
test "$(sha256sum "$root/transcript-san.tsv" | cut -d' ' -f1)" = \
  1ff4504e24d43225217ae7a885ca3484bbfaccbb57a2c5c3780497ea9c273fe7

if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    -c g1_candidate_certification_trace.cpp \
    -o "$root/trace-owner-negative.o" 2>"$root/trace-owner-negative.stderr"; then
  echo 'ERROR: trace owner compiled without seam macros' >&2
  exit 1
fi
rg 'requires both test seams' "$root/trace-owner-negative.stderr"

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v \
  2>&1 | tee "$root/python-unittest.log"
rg '^Ran 386 tests in ' "$root/python-unittest.log"
rg '^OK$' "$root/python-unittest.log"
sha256sum -c /tmp/g1-out-of-line-candidate-trace/immutable.before.sha256
(
  cd /tmp/g1-terrain-footprint-runtime-v1
  sha256sum -c /tmp/g1-out-of-line-candidate-trace/terrain.before.sha256
)
git diff --check
```

Expected: sanitizer is silent; the implementation cannot compile without both
seams; all native and 386 Python tests pass; immutable source/assets match.

- [ ] **Step 7: Independent review and exact commit**

Have a fresh reviewer check the design, RED evidence, mechanical body move,
external linkage, strict owner flags, unchanged controller, frozen transcript,
full tests, and privacy. Resolve every Critical or Important finding test-first.
Then:

```bash
set -euo pipefail
git diff --check
test -z "$(git diff --name-only -- controller.cpp g1_frame_transaction.h \
  g1_candidate_recovery.h g1_candidate_recovery.cpp \
  motion_match_log.h resources/check_g1_runtime_log.py)"
git add g1_candidate_certification_trace.h \
  g1_candidate_certification_trace.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp
printf '%s\n' \
  g1_candidate_certification_trace.cpp \
  g1_candidate_certification_trace.h \
  tests/cpp/test_g1_frame_transaction_production.cpp \
  | sort > /tmp/g1-out-of-line-candidate-trace/expected-staged.txt
git diff --cached --name-only | sort \
  > /tmp/g1-out-of-line-candidate-trace/actual-staged.txt
cmp /tmp/g1-out-of-line-candidate-trace/expected-staged.txt \
  /tmp/g1-out-of-line-candidate-trace/actual-staged.txt
git diff --cached --check
git commit -m 'fix: isolate candidate trace instrumentation'
git status --short
```

Expected: exactly the interface, strict implementation, and production test
are committed; controller/provider/log/terrain owners remain unchanged.

---

### Task 2: Rebuild the Fast Live Gate and Launch the Candidate

**Files:**

- Verify only: all committed Task 1 paths
- Generate only: `/tmp/g1-unscheduled-lazy-recovery/live32/`
- Do not modify or commit a repository file

**Interfaces:**

- Consumes the strict trace implementation object and the unchanged fast
  controller source.
- Produces exact paired 18/32 evidence and, only after PASS, one separately
  launched no-seam candidate window.

- [ ] **Step 1: Rebuild trace and release from scratch**

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
rm -rf "$root"
mkdir -p "$root/build" "$root/run18" "$root/run32"
terrain=/tmp/g1-terrain-footprint-runtime-v1
current="$root/terrain.before-live32.sha256"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-out-of-line-candidate-trace/terrain.before.sha256 "$current"
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
g++ "${strict[@]}" \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c g1_candidate_certification_trace.cpp \
  -o "$root/build/candidate-trace-owner.o"
g++ "${fast[@]}" "${rayinc[@]}" \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -DG1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM \
  -c controller.cpp -o "$root/build/controller-trace.o"
g++ "$root/build/controller-trace.o" "$root/build/clearance.o" \
  "$root/build/root-reach.o" "$root/build/recovery-seam.o" \
  "$root/build/candidate-trace-owner.o" "${raylib[@]}" \
  -o "$root/controller-trace"
g++ "${fast[@]}" "${rayinc[@]}" -c controller.cpp \
  -o "$root/build/controller-release.o"
g++ "$root/build/controller-release.o" "$root/build/clearance.o" \
  "$root/build/root-reach.o" "$root/build/recovery-production.o" \
  "${raylib[@]}" -o "$root/controller-release"

for object in clearance root-reach recovery-production recovery-seam \
    candidate-trace-owner; do
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
test "$(sha256sum "$root/controller-release" | cut -d' ' -f1)" = \
  6873c8d4239e4b1f682122cdb69161e21a11b34e43522d45f886b5f9b2fd548d
```

Expected: the trace owner and all numerical owners are strict/no-LTO; only the
trace link contains the trace object; release privacy passes and its binary
hash is unchanged from the pre-split build.

- [ ] **Step 2: Run paired 18/32 and exact frame-17 assertions**

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
    seen = [row['selected_frame'] for row in tail]
    assert len(seen) == len(set(seen))
    assert '117711' not in seen
    attempted = [row for row in tail if row['attempted'] == '1']
    accepted = [row for row in attempted
                if row['common'] == row['raw'] == row['ik'] == 'accepted']
    assert len(accepted) == 1
    winner = accepted[0]
    winner_slot = int(winner['candidate_slot'])
    assert winner['score_owner'] == 'strict-recovery'
    assert all(row['attempted'] == '1'
               for row in frame17[:winner_slot + 1])
    assert all(row['attempted'] == '0'
               for row in frame17[winner_slot + 1:])
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

(root / 'ZERO_REJECTION_32.PASS').write_text('PASS\n', encoding='ascii')
PY
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
```

Required results:

```text
trace production CSV == no-seam release CSV for IK 0 and IK 1
candidate trace IK 0 == IK 1 for 18 and 32 frames
all production rows rejection-free and unlatch
frame 17 slot zero: incumbent / incumbent
selected -> executed: 117711 -> 117712
source range: 402
cost bits: 40f14831
legacy traversals: 0
provider calls / recovery traversals: 1 / 1
accelerated candidate list == exhaustive candidate list
exactly one first attempted strict dual winner
public selected frame and cost bits == that winner in both modes
```

If any condition fails, do not launch, weaken, or continue to long
certification.

- [ ] **Step 3: Independent evidence review**

A fresh reviewer authenticates source HEAD, release hash, strict trace-owner
flags, release privacy, all four CSV comparisons, both trace comparisons,
frame-17 provenance, oracle equality, and marker creation. Critical or
Important findings stop the launch.

- [ ] **Step 4: Launch the separate candidate visualizer**

After the reviewer returns PASS, run:

```bash
set -euo pipefail
root=/tmp/g1-unscheduled-lazy-recovery/live32
test "$(cat "$root/ZERO_REJECTION_32.PASS")" = PASS
sha256sum -c "$root/BINARIES.sha256"
terrain=/tmp/g1-terrain-footprint-runtime-v1
current="$root/terrain.before-candidate-visualizer.sha256"
(
  cd "$terrain"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --
) > "$current"
cmp /tmp/g1-out-of-line-candidate-trace/terrain.before.sha256 "$current"
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

Expected: only the PID returned directly by the new launch is authenticated;
the new `mixed-multilevel` skeleton window runs with IK and independent-heading
strafe input enabled. Tell the user it is the candidate and wait for their
visual verdict before closing it or starting long certification.

## Final scope amendment

Relative to the frozen bounded-candidate execution base, final closure now
expects the former 21 paths plus exactly:

```text
docs/superpowers/plans/2026-07-17-g1-out-of-line-candidate-trace.md
docs/superpowers/specs/2026-07-17-g1-out-of-line-candidate-trace-design.md
g1_candidate_certification_trace.cpp
```

It also expects the former eight commit subjects plus exactly:

```text
docs: design out-of-line candidate trace
docs: plan out-of-line candidate trace
fix: isolate candidate trace instrumentation
```

No terrain/database, threshold, schema/checker, controller, or visualizer file
is added by this amendment.
