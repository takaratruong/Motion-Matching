# Task 5 implementation report

Status: complete and verified.

Branch: `g1-tabletop-placement`

Base: `00241f645038410406fca7389a5c946c88e3a91b`

Commit message: `feat: coordinate carry to place lifecycle`

The final commit hash is reported in the parent handoff. It cannot be embedded
in the report contained by that same commit without changing the commit hash.

## RED evidence

The first runtime test build failed as intended:

```text
make build/tests/test_interaction_runtime
```

Exit 2. The compiler reported the missing `RuntimeInput::place_request`,
`RuntimeConfig::place`, `RuntimeState::{PlacePreflight,PlaceAlign,PlaceReplay,
PlaceRelease}`, placement-enabled constructor, `preview_place`, and nested place
diagnostics.

The focused controller diagnostic build also failed as intended:

```text
make build/tests/test_interaction_place_controller
```

Exit 2. `PlaceStep` lacked exact source progress and requested/applied root,
yaw, hand-position, and hand-orientation correction fields.

After the first implementation compile succeeded, the runtime test failed at
`first.ready`. Diagnostic evidence was:

```text
mode=RecordedPlace root_error=0.258824 current_root=(0.258824,2)
staging_root=(0,2)
```

The production selector was correct; the new synthetic recorded-place fixture
kept a 0.10 m authored x offset through release instead of reaching the
destination. The fixture trajectory was corrected to decay to zero at release.

Adversarial test development exposed further fixture/build issues:

- A replacement surface moved its plane without moving its support volume and
  was correctly rejected by surface validation. The test now moves both.
- The original pickup fixture rests on the support plane for several
  post-contact frames, so reversed placement was correctly rejected as blocked.
  A reverse-only fixture now lifts immediately after contact.
- The first full safe run linked `test_interaction_matcher` against the shared
  fixture after placement initialization had been added to the base factory,
  producing an undefined `PlacementSurfaceRegistry::upsert` reference. The
  placement setup is now isolated in `make_place_runtime_fixture`, so legacy
  matcher/playback/carry tests retain their original dependency closure.

No adversarial RED required weakening production gates.

## Implementation

- Preserved `RuntimeState::Carry == 6` and appended the four Place states at
  values 7 through 10.
- Added the placement-enabled runtime constructor while retaining the legacy
  pickup constructor. Both validate the complete Place controller and IK
  configuration before registry mutation.
- Added a const, mutation-free placement preview backed by one private pure
  match-input builder. It re-fetches the exact Held owner/target generation,
  grasp, destination surface generation, and affordance, and supplies the exact
  runtime timing/match/IK configuration.
- Carry Interact freezes one exact pose/object/input/preview and publishes one
  PlacePreflight tick. The next tick rebuilds and compares the complete private
  input, canonical candidate, selection ID, and IK fingerprint before beginning
  playback. Pre-begin rejection republishes the frozen pair and retains the
  original Carry controller.
- PlaceAlign and PlaceReplay update the controller before evaluating cancel, so
  the last pre-commit sample cancels while exact/after-commit cancels are
  ignored. Post-begin failures seed a fresh Carry controller from the last-safe
  place pair.
- Release re-fetches the exact surface/affordance, repeats actual fit and sweep
  gates, and performs one atomic `commit_place` with the original pickup owner
  plus destination support context. Failed transactions remain Held and recover
  to Carry; success publishes the incremented Free generation and retracts to
  the final stop sample once before Locomotion.
- Extended nested runtime and PlaceStep diagnostics with exact source progress,
  preview/config/IK identity, goal/fit/sweep/support errors, and requested/applied
  root, yaw, hand-position, and hand-orientation corrections. Corrections are
  zero after release acknowledgement.
- Updated debug text, exhaustive runtime probe state naming, and the complete
  four-file placement source/header Make closure, including collision support.

## Adversarial coverage

The runtime test now covers:

- deterministic pure preview before/during/after Carry, legacy missing
  placement dependencies, and exact collapsed success lifecycle;
- immediate ordinary re-pick of the returned Free handle using destination
  support context rather than the source table;
- both-constructor validation for every Place/IK field family, hard caps,
  next-float-above caps, exact caps, and valid tighter request limits;
- non-default timing, match, IK, and behavioral release threshold forwarding;
- every IK scalar/iteration fingerprint and selection-ID perturbation plus
  cross-runtime stale selection rejection;
- missing/zero-ID/wrong-held requests, far/unready staging, stale far selection,
  stale target/library/surface snapshots, candidate rejection, and unchanged
  preflight Carry continuity;
- cancellation after four Align ticks, default and non-default release-position
  failures, surface replacement, and generation-overflow atomic rejection with
  exact last-safe publication and fresh-Carry seam;
- before/exact/after commit cancellation for recorded and reverse candidates;
- Reset and duplicate Place edges in every Place state;
- recorded/reverse deterministic replay, monotonic exact source diagnostics,
  zeroed post-release corrections, and one final stop sample.

## Changed files

- `interaction_runtime.h`
- `interaction_runtime.cpp`
- `interaction_place_controller.h`
- `interaction_place_controller.cpp`
- `interaction_debug_draw.h`
- `interaction_runtime_probe.cpp`
- `tests/cpp/interaction_runtime_fixture.h`
- `tests/cpp/test_interaction_runtime.cpp`
- `tests/cpp/test_interaction_place_controller.cpp`
- `Makefile`
- `.superpowers/sdd/task-5-report.md`

## Verification

All final commands exited 0:

```text
make build/tests/test_interaction_runtime \
  build/tests/test_interaction_controller_adapter \
  interaction_runtime_probe controller

build/tests/test_interaction_runtime
build/tests/test_interaction_controller_adapter

make test-interaction-safe
```

The safe suite reported 263 Python tests passing with 3 environment-dependent
skips, then passed all C++ interaction tests and selection/release/carry
fast-math gates. The desktop controller build emitted only pre-existing warnings
from raygui/array file-reading code.

## Self-review

The final audit covered every Place transition, output authority flag,
attachment/registry mutation, target generation and support update, frozen versus
reconstructed Carry continuity, candidate/config/IK identity, cancellation
boundary, release transaction terminal, final retraction sample, and Make link
closure. Dead release-attempt bookkeeping and an unused include were removed.
Placement fixture setup was isolated from non-runtime tests after the full-suite
link RED. No blocking finding remains.

The protected repository-root `interaction_query_probe` artifact was never
accessed; verification used only the safe build output selected by the Make
target.
