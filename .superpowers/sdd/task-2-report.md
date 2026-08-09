# Task 2 report: Run the G1 ordinary matcher at 60 Hz

## Result

Migrated the ordinary G1 controller, runtime/query path, route/footprint/IK/
clearance gates, scene manifest, and runtime log contract from exact float32
25 Hz to exact float32 60 Hz. The shared G1 rate predicate requires the
binary32 `1/60` bits `0x3c888889`; stale 25 Hz manifests fail before
controller or IK state allocation.

Trajectory horizons now derive from an explicit validated rate. G1 binds
60 Hz to `[20,40,60]`; the shipped 23-bone asset formats and call path remain
unchanged. The ordinary database search retains Orange Duck's stock 20-frame
ignore-range-end and surrounding margins.

Each ordinary-runtime result and CSV row now carries the raw 31D query and its
shared normalized 31D snapshot. The runtime-log checker requires exact 60 Hz
for runtime rows and changes the 15-second Gate A default from 375 to 900 rows.

## Flat bundle adapter

Added a fail-closed adapter for the intentionally minimal flat bundle. It
accepts only `g1-lmm-flat-data/v2`, authenticates `database.bin` and
`features.bin`, checks the exact rate/horizon/skeleton/features/zero-terrain
claims, and independently verifies:

- complete 3,853-row coverage by 13 ranges of at least 61 frames;
- the little-endian range and source-map SHA-256 preimages;
- source reconstruction indices inside each retained range;
- the continuity receipt and all in-range local-rotation steps at
  `<=0.25 rad/frame`; and
- the stored feature normalization and four exactly-zero terrain dimensions.

The adapter creates only an in-memory zero-height flat scene/query context; it
does not require or synthesize GRAIL v2 sidecars. The unsafe unsplit v1 bundle,
stale rate, altered continuity threshold, altered source-map digest, and
nonzero terrain claim all reject before runtime state.

The corrected immutable bundle at
`sonic/runs/g1-lmm-flat-60hz/data` passes a 600-frame ordinary replay with
matching enabled: zero holds/resets, finite poses, maximum joint step within
`0.25 rad`, and raw-to-normalized feature parity within `1e-6`.

## Additional G1 entry points

The chunk server's movement/turn reset probes now use exact `1/60`, and its
actual advance path consumes the 60 Hz default `g1_runtime_config`. Its
serialized `source_rate_hz=25` is intentionally unchanged because that field
describes the separate legacy source-chunk wire artifact, not controller
execution cadence; a focused source assertion locks this distinction.

The paired route command compiler and route-schedule CLI now run at 60 Hz.
Their chunk width changes from 10 to 24 source intervals, preserving the
physical 0.4-second command duration. Route speed and waypoints are unchanged.
The regenerated 12-second flat command receipt is
`8cb01ff2398dac23df585b3970571c03d51215fe6ea25f662fbbc7462f765d9d`.
The real curb route contains 408 60 Hz frames and all three registered-route
C++/Python parity cases pass.

## TDD evidence

RED was observed before production edits for stale 8/17/25 horizons, the 25 Hz
runtime default, missing normalized-query logging, missing flat adapter, 0.04
chunk-server probes, the 0.04/25 Hz route CLI, and the 10-interval/25 Hz command
receipt. GREEN covers exact 60/25 acceptance/rejection, analytical 20/40/60
feature parity, normalized-query parity at `1e-6`, checked feature loading,
continuity range boundaries, the real v2 replay, and the five tamper negatives.

## Verification

There is no CMake build directory, so scoped C++ targets and entry points were
compiled directly with
`g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I.`.

Final focused verification comprises three scoped C++ tests, controller syntax,
strict chunk-server and route-CLI builds, the 600-frame real v2 replay, five
negative bundles, 15 command/chunk-server Python tests, three real route parity
tests, and Python bytecode validation of the runtime-log checker.

Only Task 2 source/tests and this report are staged; concurrent Task 1 and
Task 3 edits in the shared worktree are excluded.

## Independent-review remediation (2026-08-09)

The continuity gate now evaluates all 31 local rotations, including bone 0
(`Simulation`). The 600-frame ordinary replay uses the same complete bone set,
so its joint-step evidence cannot mirror the former root omission. A focused
root-only `0.30 rad` tamper failed RED because the validator started at bone 1;
after the correction it rejects with `bone=0`, while discontinuities exactly
at retained-range boundaries remain admissible.

The flat adapter now retains each manifest artifact `size_bytes` receipt and
compares it with the actual `database.bin` and `features.bin` file sizes before
SHA-256 authentication. The real-v2 replay test copies the immutable bundle,
changes only the database size receipt by one byte, and requires a size-mismatch
rejection. This negative failed RED because the parser previously discarded the
field and passed GREEN after size authentication was added.

Fresh focused verification directly compiled the terrain-database,
support-matching, and G1-runtime C++ tests under the strict C++17 warning
contract. All three source tests passed; the canonical v2 600-frame replay also
passed with complete 31-bone continuity evidence and the new size-only negative.
The missing chunk-server and route-CLI executables were directly compiled into
the expected narrow test location, after which the focused Python command and
chunk-server suites ran 41 tests: 39 passed and the two guarded external-terrain
tests skipped because `SONIC_TERRAIN_DIR` was not supplied.

## Canonical v3 walk-only migration (2026-08-09)

Task 2 now accepts only the Task 1 walk-only `g1-lmm-flat-data/v3` identity at
`sonic/runs/g1-lmm-flat-60hz/data-v3`. The runtime pins and reauthenticates all
three canonical SHA-256 values before controller state:

- manifest: `db01f5bfb7641333b3e40ea4a5d1eb655131c7bc72681d2738fde2dbf9298709`;
- `database.bin`: `13b368759c22ff3427d937a86fd9399cd6e80646e5f288e80da01bd988daaaad`;
- `features.bin`: `7c35809e1dd5ea14bd56b0f607cb9da22b3464ebbece01a895050372530b40df`.

The flat adapter requires exactly 256 rows, one complete `[0,256)` flat-walk
range, one authenticated 512-frame/120 Hz source receipt, and the absolute
source map `7659,7661,...,8169`. It also requires the v3 nearest-mode Orange
Duck contact contract (116 left / 117 right contact rows and its bilateral run
receipt), zero continuity rejections/drops, `[20,40,60]` horizons, and the
existing exact-zero terrain slice. The prior v2 bundle rejects, as does a
semantically identical v3 manifest whose raw bytes no longer match the
canonical digest. Artifact size and live SHA checks remain active.

The ordinary proof no longer asks a 256-row source clip to sustain a 600-frame
loop. It owns row 0 after reset and performs exactly 255 matching-disabled
advances through rows 1..255. Every call asserts the query/selected identity is
the prior row, the emitted identity is exactly prior+1, and there is no search,
transition, hold, reset, wrap, or seam. The replay stops without calling the
clamped end row again. Finite pose, all-31-bone `<=0.25 rad` continuity, and
raw/normalized 31D query parity at `<=1e-6` remain required. Orange Duck's
ordinary search implementation still uses the stock 20-frame end and
surrounding margins; the source replay disables search rather than changing
those margins.

The real v3 database exposed a numeric mismatch in the independent continuity
receipt check: the old C++ path multiplied slightly non-unit binary32
quaternions directly, while Task 1 normalizes and computes the geodesic in
float64. A focused RED reproduced a nominal `0.10 rad` step as approximately
`0.100002 rad`. The validator now normalizes through a float64 dot/norm
calculation before `acos`, matching the independent Task 1 definition while
retaining the hard `0.25 rad` rejection and bone-0 coverage.

RED first failed both runtime and LMM tests on the missing v3 manifest shape;
after structural migration, the semantic-whitespace tamper remained RED until
the canonical raw-manifest pin was added. The quaternion-normalization RED was
observed independently in `test_terrain_database`.

Fresh focused GREEN compiled with the strict C++17 warning contract and passed
`test_terrain_database`, `test_support_matching`, `test_g1_runtime` (ordinary,
canonical v3, and v2 rejection modes), `test_g1_controller_state`, and
`test_g1_lmm`; `controller.cpp` also passed its Raylib/Raygui syntax check.
The accepted learned model is still absent, so no viewer was launched.
