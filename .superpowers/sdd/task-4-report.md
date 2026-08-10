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

## Review remediation

The three Important review findings were reproduced and corrected in a
separate RED-to-GREEN cycle.

The projector RED added failures at the final projected feature and final
latent output. The original helper rejected the NaN only after mutating earlier
caller outputs and its evaluation layers:

```text
G1 LMM test failed: late projector rejection leaves all outputs and evaluation state bitwise unchanged
```

Projector inference now uses a private evaluation candidate plus private
feature/latent/cost/transition candidates. It swaps and copies into caller
state only after every output and derived cost is finite. The negative test
hashes transition, cost, both output arrays, the evaluation layer count, every
layer shape, and every layer byte before and after each late rejection.

The loader RED changed the public seam so no caller-populated
`motion_pack_manifest` could satisfy authentication; compilation failed on the
removed trusted metadata argument. The loader now calls
`motion_manifest_load_and_verify` on the observed Task 1 directory itself,
then binds the model manifest to the SHA-256 of that verified raw v2 manifest
and its authenticated database/features digests. The positive synthetic model
fixture is bound to the real production-shaped Task 1 v2 bundle. Arbitrary raw
manifest text with manually plausible artifact files rejects before any
evaluation allocation, as do a copied live-data size tamper and all prior
digest/ABI tampers.

The controller transaction RED failed to compile on the missing staged gait,
gait-velocity, route-waypoint, and camera-azimuth request contract. These
values are now derived in locals. LMM applies them only to its existing cloned
runtime candidate, so the runtime's final state swap commits them together
with an accepted tick. Ordinary mode retains its prior live-field behavior.
The controller's compile-time camera scripting is staged through the same
contract. A late-projector-NaN rejection test changes all four candidate
values and verifies the live `desired_gait`, `desired_gait_velocity`,
`route_waypoint`, and `camera_azimuth` remain bitwise unchanged; the accepted
case verifies they commit atomically.

Fresh remediation verification passed:

```text
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm_remediation_strict && \
  /tmp/test_g1_lmm_remediation_strict

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime_remediation_strict && \
  /tmp/test_g1_runtime_remediation_strict && \
  /tmp/test_g1_runtime_remediation_strict \
    --flat-bundle sonic/runs/g1-lmm-flat-60hz/data

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state_remediation_strict && \
  /tmp/test_g1_controller_state_remediation_strict

g++ -std=c++17 -O3 -ffast-math -DNDEBUG -w -I. \
  tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm_remediation_fast && \
  /tmp/test_g1_lmm_remediation_fast

g++ -std=c++17 -fsyntax-only -I. \
  -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp

g++ -std=c++17 -O2 -I. -I/home/ubuntu/apps/raylib/src \
  -I/home/ubuntu/apps/raygui/src controller.cpp \
  /home/ubuntu/apps/raylib/src/libraylib.a \
  -lGL -lm -lpthread -ldl -lrt -lX11 \
  -o /tmp/controller_task4_remediation
```

The linked controller still fails closed before window creation when launched
with the canonical Task 1 bundle because the accepted Task 3 model manifest is
absent:

```text
G1_TERRAIN_DIR=sonic/runs/g1-lmm-flat-60hz/data \
  /tmp/controller_task4_remediation --locomotion-engine lmm
G1 LMM model error: ./sonic/runs/g1-lmm-flat-60hz/model/manifest.json: cannot open for SHA-256 (No such file or directory)
```

No real interactive claim was made.

## Canonical v3 data binding migration (2026-08-09)

The Task 4 loader now consumes only the canonical Task 1
`g1-lmm-flat-data/v3` identity. Its production-shaped synthetic model binds the
observed v3 manifest digest and the canonical database/features digests, and
its latent table has exactly `256x32` values to match the walk-only database.
A model manifest that still declares a v2 data binding rejects before any of
the three evaluation buffers are allocated. The existing data-manifest,
data-artifact, model-artifact, live-file, ABI, late-NaN, one-stepper/one-commit,
and full transaction-isolation gates continue to pass against v3.

Only the finite ordinary source proof changed duration: it now visits the 256
source rows once without a seam. The learned recurrent LMM acceptance contract
remains a 600-tick transactional interactive gate and is not bounded by latent
table row count. No accepted Task 3 model is published, so this migration did
not launch the viewer or claim the deferred 600-tick interactive result.

## Kinematics-model data gate remediation (2026-08-09)

Because the LMM loader internally invokes the flat-data parser before model
artifact parsing or evaluation allocation, it now inherits the exact Task 1
`kinematics_model` receipt gate. A copied canonical data directory with that
top-level field removed rejects with zero evaluation allocations and a
kinematics-specific error. The positive synthetic model dynamically binds the
new observed raw-manifest digest; no model-binding constant or accepted model
artifact was synthesized. The deferred learned 600-tick viewer gate remains
unchanged and was not launched.

Fresh strict `test_g1_lmm`, `test_g1_runtime`, and
`test_g1_controller_state` runs passed, as did controller syntax and link.

## Accepted canary model-scope binding (2026-08-09)

The runtime model manifest now requires the exact top-level key/value
`"model_scope":"single-clip-overfit-canary"`. The key participates in the
manifest's exact-key set, is parsed before any network evaluation allocation,
and is published only with the fully authenticated model candidate. The scope
is retained in `g1_lmm_model_bundle` and its transactional swap, logged once
after successful startup authentication, and shown in the existing LMM
controller panel. It was deliberately not added to the stable per-frame CSV
or runtime-diagnostic ABI.

The compile-stage RED was the new accepted-bundle assertion failing because
`g1_lmm_model_bundle` had no `model_scope` member:

```text
tests/cpp/test_g1_lmm.cpp:717:20: error: 'struct g1_lmm_model_bundle'
has no member named 'model_scope'
```

The GREEN fixture emits the exact accepted scope by default. Missing, wrong
string, wrong type, and an extra scope-related top-level key each reject with
zero evaluation allocations; no rejected candidate publishes scope metadata.
A failed reload preserves an already authenticated scope and latent storage,
and a direct metadata-only swap test proves scope exchanges symmetrically with
the data digest, authentication flag, and allocation count. The controller
source contract also checks the two lifetime-safe diagnostic uses (startup
log and overlay).

Fresh verification passed:

```text
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_lmm.cpp -o /tmp/test_g1_lmm_scope_strict && \
  /tmp/test_g1_lmm_scope_strict

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp -o /tmp/test_g1_runtime_scope_strict && \
  /tmp/test_g1_runtime_scope_strict && \
  /tmp/test_g1_runtime_scope_strict \
    --flat-bundle sonic/runs/g1-lmm-flat-60hz/data-v3

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state_scope_strict && \
  /tmp/test_g1_controller_state_scope_strict

g++ -std=c++17 -fsyntax-only -I. \
  -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp

g++ -std=c++17 -O2 -I. -I/home/ubuntu/apps/raylib/src \
  -I/home/ubuntu/apps/raygui/src controller.cpp \
  /home/ubuntu/apps/raylib/src/libraylib.a \
  -lGL -lm -lpthread -ldl -lrt -lX11 \
  -o /tmp/controller_model_scope

git diff --check
```

The Task 3 Python publisher changes were owned by the concurrent training
lane and were not staged or modified here. No viewer or model executable was
launched; the deferred learned 600-tick interactive gate remains unchanged.

## Ordinary overlay geometry remediation (2026-08-09)

Review found that the canary-scope overlay had unconditionally expanded the
learned-motion-matching panel from 40 to 65 pixels and moved the controls from
`y=380` to `y=405`, including in ordinary mode. The focused RED failed with:

```text
controller reset test failed: ordinary keeps the 40-pixel LMM panel while LMM alone expands it
```

The layout now selects `40/380` for ordinary mode and `65/405` only for LMM.
The authenticated scope remains visible only on the LMM path. The focused
controller-state test, controller syntax and full desktop link, and
`git diff --check` passed. No viewer or GPU/model process was launched or
modified.
