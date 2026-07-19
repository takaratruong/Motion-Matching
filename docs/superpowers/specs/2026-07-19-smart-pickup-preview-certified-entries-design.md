# Smart Pickup Preview-Certified Entries Design

## Purpose

An interaction affordance can expose multiple authored entry transforms, but
the current Smart Pickup chooses the nearest geometry-valid entry before it
checks whether the live pose can transition into that entry's recorded
motion. It freezes that winner, walks there, and previews only that one entry.
If the preview is hard-rejected, the attempt fails even when another clear
authored entry would match.

The next Smart Pickup level must treat motion compatibility as entry
eligibility. Pressing `F` should select a reachable, collision-clear,
motion-compatible entry deterministically, then keep the one-way frozen funnel
and normal runtime preflight authority.

## Approaches Considered

1. Preview every geometry-eligible entry immediately after activation, then
   freeze the best certified entry before locomotion. This is selected. It
   follows the Smart Object pattern of selecting an interaction position and
   animation together before navigation.
2. Walk to the nearest geometry-only entry and preview alternates only if its
   final preview fails. This looks smaller, but it causes visible reversal,
   requires FinalPreview-to-Approach loops and deadline resets, and recomputes
   reachability from a changed root.
3. Sample an approach with a learned/diffusion model. This may later expand
   the entry distribution, but deterministic authored entries must first have
   correct eligibility, ranking, provenance, and failure labels.

## State Model

Append `SlotSelectionPreview = 7` to `PickAssistState` without renumbering
existing values. The state sequence becomes:

```text
Idle
  -> SlotSelectionPreview
  -> SlotApproach
  -> Settling
  -> FinalPreview
  -> ReadyToSubmit
  -> Submitted
```

`begin` maps and geometry-filters every authored entry, snapshots target,
affordance, obstacles, and deterministic ranking keys, and enters
`SlotSelectionPreview`. It does not yet set the frozen slot, selected slot ID,
or final preview root.

The activation post-step still calls `begin` and `observe` exactly once. That
observation returns stationary braking plus a batch of preview requests. No
matcher callback runs on the activation tick. On the next fixed tick, the
controller evaluates the whole batch against one live flat-locomotion
snapshot and returns it in one observation.

If one or more entries certify, the assist freezes exactly one deterministic
winner and enters `SlotApproach`. There is no alternate-slot switch after this
freeze. Approach, Settling, the singular frozen-slot FinalPreview, request
extraction, runtime Preflight, attachment, and Carry then retain their current
one-way semantics.

## Entry Ranking

`PickSlotSelection::ordered` remains in authored order for provenance and
diagnostics. Add `ranked_eligible_indices`, containing only geometry-clear
entries sorted by the existing integer tuple:

```text
(route millimetres, heading milliradians, slot ID)
```

Legacy `selected_index` remains the first geometry-only index so existing
diagnostics and callers do not silently change meaning. The assist's actual
frozen entry gets a separate `frozen_slot_index` diagnostic.

Preview readiness is an eligibility filter, not a new float ranking term. If
several entries certify, choose the first certified index in
`ranked_eligible_indices`. Do not rank on full-pack floating-point cost: all
ready candidates already passed the configured threshold, and integer
geometry ordering remains deterministic under normal and fast-math builds.

## Batch Preview API

Add these value types:

```cpp
struct PickAssistPreviewRequest {
    uint32_t slot_id = 0U;
    PickEntryRoot root{};
};

struct PickAssistPreviewResult {
    PickAssistPreviewRequest request{};
    std::optional<PickEntryPreview> preview{};
};
```

Replace the singular preview fields with one protocol used by both selection
and final certification:

```cpp
// PickAssistOutput
std::vector<PickAssistPreviewRequest> preview_requests{};

// PickAssistObservation
std::vector<PickAssistPreviewResult> preview_results{};
uint64_t preview_snapshot_fingerprint = 0U;
```

FinalPreview emits and consumes a one-element batch for the frozen slot. This
avoids two nearly identical preview protocols.

`SmartPickupController::post_step` evaluates each request in the previous
output, in request order, using the exact same `live_flat_snapshot`. It echoes
the request identity into each result and sets the one batch fingerprint to
the already-computed live snapshot fingerprint. The callback type,
controller/backend method signatures, `InteractionRuntime::preview_pick`, and
`PickRequest` remain unchanged.

## Certification and Provenance

A selection result is certified only when all of these hold:

- the result batch is complete and in the exact outstanding-request order;
- request slot ID still identifies the same authored entry;
- request root and returned prospective root are finite and bit-equal to the
  mapped authored root;
- the frozen target handle/generation, affordance ID, slot order/metadata, and
  obstacle snapshot remain unchanged;
- the current route to that entry remains collision-clear;
- preview fingerprint exactly equals the observation snapshot fingerprint;
- `path_feasible`, `path_reason == None`, `match_ready`, and
  `match_reason == None`.

The authoritative selection provenance is:

```text
(target handle and generation,
 affordance ID,
 slot ID,
 mapped-root float bits,
 live-snapshot fingerprint)
```

The match candidate, clip, entry/contact frames, and cost remain diagnostics.
They are not cached as an allowlist, added to `PickRequest`, or trusted by
runtime Preflight. Runtime evaluates the submitted target/affordance again
from its authoritative live snapshot.

## Retry and Failure Semantics

An incomplete batch is atomic: select nothing and request the entire batch
again. Never choose from a partial response.

If no entry certifies but at least one exact result is physically feasible and
reports retryable `PoorMatch`, remain stationary and request all eligible
entries again on the next tick. Track whether PoorMatch has occurred.

Use a selection-preview counter bounded by the existing
`maximum_arrival_ticks`; reset the ordinary arrival counter when an entry is
frozen. Deadline failure is `PoorMatch` if any retryable cost failure was seen,
otherwise `ArrivalDeadline`.

Append `SelectionPreviewRejected = 15` to `PickAssistReason` without
renumbering existing values. A complete batch with no certified or retryable
entry fails with that reason. Wrong count/order, duplicate slot, wrong root, wrong
fingerprint, non-finite values, or changed target/slot provenance also fail
closed; they never trigger alternate selection from untrusted results.

Once an entry freezes, a hard singular FinalPreview rejection is terminal.
There is no post-freeze ping-pong, no switch after request extraction, and no
switch after runtime submission.

## Call-Count Contract

Let `K` be the number of geometry-eligible entries, `E` the number of
selection-preview epochs, and `F` the number of frozen FinalPreview attempts:

- activation tick: 0 matcher callbacks;
- each selection epoch: exactly K callbacks in ranked order on one snapshot;
- SlotApproach and Settling: 0 callbacks;
- each FinalPreview attempt: exactly 1 callback for the frozen entry;
- Failed or Submitted: 0 callbacks.

Total callbacks before submission are exactly `E*K + F`. An empty callback
produces zero calls and an incomplete epoch; it never yields a partial winner.
This slice does not move matching to a worker thread or cache the result.

## Diagnostics and UI

Diagnostics retain the geometry-only selection and add:

- actual `frozen_slot_index` and selected slot ID;
- one selection-preview diagnostic per eligible entry;
- requested slot/root and returned prospective root;
- observation and preview fingerprints;
- availability, root/fingerprint equality, path/match outcome, frames, and
  cost;
- selection epoch/call counts and PoorMatch-seen state.

The existing `SMART PICKUP AUTO` banner covers SlotSelectionPreview as an
automatic, WASD/`X`-cancellable phase. A hard batch failure uses the existing
truthful `PICKUP FAILED: <reason>` presentation.

## Verification

Implementation is split into reviewed TDD slices:

1. Slot ranking: authored order is preserved, ranked indices include only
   eligible entries, tuple ties are deterministic, and normal/fast-math builds
   agree.
2. Assist state/API: begin requests K previews without freezing; nearest
   incompatible plus farther ready freezes the farther entry; two ready
   entries follow the geometry tuple even if motion costs reverse; incomplete,
   misordered, duplicate, wrong-root, wrong-fingerprint, mutation, retry, and
   deadline cases are exact; the frozen entry never switches.
3. Controller batching: activation makes zero callback calls, next tick makes
   K in exact order with bit-identical snapshot/fingerprint, retry makes K,
   FinalPreview makes one, submission makes zero, and one observation occurs
   per tick.
4. Live-flat/full-pack gate: a geometry-nearest non-ready entry and a farther
   ready entry select the farther one, then pass ordinary runtime Preflight to
   attachment/Carry without trusted clip authority.
5. Existing scene/extractor tests remain unchanged to prove authored slot IDs,
   order, and source provenance still come from the interaction data.

All control/runtime clocks remain exactly 25 Hz, flat locomotion remains the
only movement source, and terrain, playback, Carry, placement, and learned
models remain out of scope.
