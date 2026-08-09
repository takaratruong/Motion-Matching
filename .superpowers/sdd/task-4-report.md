# Task 4 report: Load and drive the G1 LMM flat canary

## Result

Implemented an explicit `--locomotion-engine ordinary|lmm` controller mode.
Ordinary remains the default and retains its existing runtime call. LMM requires
the authenticated Task 1 `g1-lmm-flat-data/v2` bundle and an accepted Task 3
`g1-lmm-model/v1` bundle before the graphics window or controller loop can
start; no ordinary fallback is available after LMM is selected.

The model loader accepts the exact Task 3 publisher manifest shape, binds the
raw Task 1 manifest digest and both data artifact digests, reauthenticates the
live data/model artifact sizes and SHA-256 values, and parses the little-endian
Orange Duck ABI with exact dimensions and EOF. Evaluation buffers are allocated
only after all data/model artifact authentication and all four runtime binaries
parse successfully.

The recurrent 31D normalized feature vector, 32D latent vector, stepper count,
and commit count now live in `g1_controller_state` and participate in reset,
shape validation, deep clone, and swap. One LMM frame performs this order on a
clone: normalized query, transactional projection, projected-terrain overwrite
at indices 27:31, one stepper evaluation, non-mutating provisional decode,
candidate-root terrain sample/overwrite, final decode from the original root,
fixed-point/mechanical/finite validation, and one live-state swap. LMM does not
advance or publish an ordinary database frame identity and does not invoke
inertialization, support retargeting, IK, pose damping, smoothing, adjustment,
or clamping.

Runtime results expose engine ownership, raw and normalized 31D queries,
projector cost, provisional/final pose and root diagnostics, and accepted
stepper/commit counters. The controller log mode and UI identify the active
engine; LMM log rows publish `-1` for database identity and mark ordinary pose
corrections disabled.

## Brief path correction

The regenerated brief named `sonic/cpp/g1_controller_state.h`, which does not
exist. Per the parent interface decision, Task 4 modified the existing
repository-root `g1_controller_state.h` and did not create a duplicate header.

## TDD evidence

The initial RED command (direct compilation because this repository has no
CMake project or `build/` tree) was:

```text
g++ -std=c++17 -I. tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm_red
```

It failed on the intended missing `g1_lmm_dimensions_valid`,
`projector_cost_normalized`, `lmm_features`, `lmm_latent`,
`lmm_commit_count`, and `lmm_stepper_count` contracts. A second RED failed on
the missing authenticated bundle type/loader, and a third RED failed on the
missing `g1_runtime_step_lmm_normalized` transaction and engine diagnostics.

GREEN coverage uses a production-shaped synthetic model: full-size
decompressor `63->512->458`, stepper `63->512->512->63`, projector
`31->512->512->512->512->63`, and a `3853x32` latent table. It verifies the
canonical Task 1 bundle binding, exact normalized distance, projected and
candidate terrain overwrite order, exactly one step/commit, provisional
scratch isolation, raw/normalized parity, and no database identity. A NaN
projector after command/query preparation leaves command, recurrent, pose,
root, contact, terrain-bearing features, counters, and caller result bitwise
unchanged.

Negative authentication tests cover the bound data-manifest digest, both data
artifact digests, all six model artifact digests, a live data artifact change,
and a self-consistent model hash/size update containing trailing network bytes.
Every authentication rejection occurs with zero evaluation-buffer allocations.

## Verification

All final focused checks passed:

```text
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm_strict && \
  /tmp/test_g1_lmm_strict

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime_strict && \
  /tmp/test_g1_runtime_strict

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state_strict && \
  /tmp/test_g1_controller_state_strict

/tmp/test_g1_runtime_strict \
  --flat-bundle sonic/runs/g1-lmm-flat-60hz/data

g++ -std=c++17 -O3 -ffast-math -DNDEBUG -w -I. \
  tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm_fast && \
  /tmp/test_g1_lmm_fast
```

The desktop controller also passed a complete link against the checked-out
Raylib static library and an independent `-fsyntax-only` compile. Its invalid
CLI gate returned exit 2 with the exact usage, and the real LMM launch returned
exit 2 before `InitWindow` because
`sonic/runs/g1-lmm-flat-60hz/model/manifest.json` does not exist.

## Deferred real interactive gate

Task 3 intentionally published only rejected training/evaluation receipts: the
real decompressor gate failed and there is no accepted manifest or runtime
binary set. Therefore the requested 20-second interactive LMM drive/video and
mechanical receipt were not run. This is the expected fail-closed stopping
condition from the brief, not an ordinary fallback or a synthetic acceptance
claim.
