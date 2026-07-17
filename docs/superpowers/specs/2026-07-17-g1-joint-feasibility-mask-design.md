# G1 Joint-Feasibility-Masked Motion Matching Design

**Date:** 2026-07-17

**Status:** Approved written design; implementation plan linked

**Base:** `67eb47a20623f9e2ccd2a2a35bc3e9fa2d04afa8`

**Implementation branch:** `g1-sonic-scene-aware-baseline`

## Purpose

The first unchanged Stage A trial proved that the authenticated terrain-aware
motion database contains poses that cannot be represented within the registered
G1 joint contract. At chunk 5, the matcher selected database frame 866 and then
advanced to frame 867. Frame 867 projects the left ankle roll to
`-0.307408422`, outside `[-0.261799991, 0.261799991]`.

The controller now reports that event truthfully as a scientific failure. This
intervention asks the next narrow question:

> Can the existing matcher complete its reference using only database
> transitions whose selected and immediately emitted raw poses satisfy the
> authenticated G1 joint contract?

The intervention constrains matching. It does not repair, clip, or relabel an
invalid pose.

## Evidence and scope

The authenticated database contains 459,682 frames. A complete read-only scan
found 1,063 raw limit-invalid frames (`0.231%`) in 88 source ranges. The exact
failure neighborhood is:

- frame 866: raw left ankle roll `-0.190088332`, valid;
- frames 867 through 872: raw left ankle roll invalid;
- frame 873: raw pose valid.

Filtering only the frame returned by search is insufficient because the
runtime advances one database frame before emitting the new boundary. Dropping
every affected source range is also unsuitable: it would discard 39,182 frames
(`8.52%`) and the entire Takara flat source range.

## Goals

- Derive a deterministic raw-pose feasibility mask from the authenticated
  database and registered joint contract at server load.
- Prevent ordinary range progression from entering an unsafe raw pose.
- Exclude search candidates whose transition target or immediate emitted
  successor is unsafe.
- Preserve the existing database and sidecar bytes and their identities.
- Expose the derived mask identity and counts through the authenticated MM
  protocol and Stage A evidence.
- Keep final projection of the actual inertialized pose as an uncompromised hard
  gate.
- Rerun the unchanged Stage A command after all protected and repository tests
  pass.

## Non-goals

This change does not:

- clip, clamp, sanitize, or interpolate joint values into range;
- widen the MJCF or registered joint limits;
- remove or rewrite source frames in the published database;
- silently omit a frame after it has been emitted;
- alter feature weights, terrain costs, transition costs, commands, scenes,
  policy weights, SONIC observations, or Stage A thresholds;
- guarantee that inertialization between individually valid raw poses is valid;
- implement next-best search after an inertialized transition fails;
- claim a scientific pass unless all seven Stage A gates pass.

## Selected approach

The MM server derives two immutable bit masks after loading and authenticating
the database and joint contract:

1. `raw_safe[i]`: database frame `i` has an exact, finite, on-axis joint
   projection and every projected joint lies within its registered range.
2. `search_safe[i]`: both frame `i` and the range-clamped one-step successor of
   `i` are `raw_safe`.

The runtime uses `raw_safe` to guard ordinary progression and `search_safe` to
filter search candidates. The database remains byte-identical; feasibility is
a deterministic runtime derivative of already authenticated inputs.

This approach is preferred over artifact fragmentation because it preserves
the flat Takara corpus and avoids changing source-manifest topology before the
scientific hypothesis is tested. It is preferred over full transition
enumeration because it closes the observed raw-successor defect without
changing the matcher's cost ordering or adding a state-dependent retry search.

## Architecture

### 1. Structured joint projection

The joint-position extraction currently embedded in `sonic_project_pose()` is
factored behind one internal projection primitive. The primitive returns
structured status, including the failure category and source joint index,
without parsing human-readable error strings.

The categories are:

- valid;
- registered joint-limit violation;
- singular twist;
- off-axis or reconstruction violation;
- non-finite or malformed input.

`sonic_project_pose()` continues to render its existing exact error messages
from that structured result. Runtime behavior and the remote scientific error
therefore remain compatible.

Only a registered joint-limit violation makes a database frame maskable.
Singular, off-axis, reconstruction, non-finite, or shape failures remain
artifact-integrity failures and stop server loading. This prevents the mask
from hiding a corrupt or semantically incompatible database.

### 2. Immutable feasibility certificate

After database, terrain sidecars, matching features, and joint contract are
authenticated, the server projects every database frame's local rotations with
the same primitive used for emitted poses.

The certificate contains:

- one byte per frame for `raw_safe`;
- one byte per frame for `search_safe`;
- total, raw-safe, raw-unsafe, and search-safe frame counts;
- per-joint raw-limit-violation counts;
- SHA-256 of a canonical certificate payload containing the schema tag,
  database frame count, `raw_safe`, and `search_safe` bytes.

The canonical schema tag is
`g1-joint-feasibility-certificate/v1`. Integer fields use fixed-width
little-endian encoding in the hashed payload. The certificate is immutable for
the lifetime of the loaded server and is swapped with the same resource
boundary as the database.

Load succeeds only when:

- mask lengths exactly equal `database.nframes()`;
- frame 0 is raw-safe, so reset can emit a valid initial boundary;
- at least one search-safe frame exists;
- every mask byte is exactly zero or one;
- all non-limit projection failures are absent.

### 3. Filtered search without cost changes

`motion_matching_search()` and `database_search()` gain an optional candidate
mask. Existing callers that omit it retain byte-for-byte search semantics.
When supplied, the leaf scan skips a frame before cost evaluation unless its
mask byte is one.

The branch-and-bound boxes remain unchanged. Their bounds may cover skipped
frames, which can reduce pruning efficiency but cannot change the minimum among
allowed frames. Mask shape and binary values are validated once when the
immutable certificate is constructed; the G1 search wrapper passes only that
prevalidated slice, so no full-mask scan occurs in the per-step hot path.

If no allowed candidate exists, search returns no index and the G1 runtime
fails generation with an explicit `no joint-limit-safe database candidate`
error. It never falls back to an unsafe incumbent.

### 4. Progression boundary

Before advancing the current database frame, the runtime computes its normal
range-clamped successor. An unsafe successor is treated like the end of an
animation for search scheduling:

- incumbent cost becomes unavailable;
- search is forced even when the ordinary timer has not expired;
- the candidate search uses `search_safe`;
- the selected target and its emitted successor must both be raw-safe.

This closes the observed 866-to-867 transition. Range clamps remain authoritative;
the mask does not permit cross-range progression.

The selected raw target can still influence inertialization. After the step,
the server projects the actual inertialized state through the existing strict
boundary. If that projection violates a limit, candidate generation fails
scientifically and the active transactional state is unchanged. No alternate
candidate is attempted in this design.

### 5. Protocol and evidence

The strict MM hello artifact identity gains a `joint_feasibility` object with:

- `schema`;
- `frame_count`;
- `raw_safe_count`;
- `raw_unsafe_count`;
- `search_safe_count`;
- `mask_sha256`;
- a fixed 29-element per-source-joint violation-count array.

Python schema validation requires exact keys, exact integer domains, exact
joint ordering, count conservation, and a lowercase SHA-256 value. Scene
identity verification binds this object across preflight and the real server.
The complete object is retained in the run manifest and gate evidence so a
Stage A result identifies the derived candidate population.

Adding the object is a protocol-schema change, not a protocol-version change:
both bundled endpoints are updated atomically, and strict old/new mismatches
fail integration preflight.

## Error ownership

- Certificate construction or non-limit raw projection failure:
  integration/configuration failure before a trial.
- No search-safe candidate for the current query:
  scientific `generation_failed` during the MM gate. The server renders the
  exact fixed message `no joint-limit-safe database candidate`, and the Python
  verdict boundary recognizes only that anchored message with the structured
  `generation_failed` code. Lookalikes and other generation errors remain
  integration failures.
- Actual inertialized pose outside registered limits:
  scientific `generation_failed`, unchanged from the repaired verdict path.
- Malformed mask identity or cross-process mismatch:
  integration failure.

No failure in this design is converted into a pass or hidden by a skipped
output frame.

## Testing

Implementation is test-first and includes:

1. Projection-unit RED tests for valid, lower-limit, upper-limit, singular,
   off-axis, and non-finite frames, proving structured and rendered results
   agree.
2. Certificate RED tests for exact counts, per-joint counts, deterministic
   hash, frame-0 safety, malformed masks, and non-limit fail-closed behavior.
3. Search RED tests proving an invalid lowest-cost frame is skipped while
   unmasked callers retain existing selection.
4. Runtime RED tests reproducing a safe selected frame with an unsafe immediate
   successor and proving search is forced before progression.
5. Protocol/schema RED tests for exact certificate identity and mismatch
   rejection, plus CLI verdict tests for the exact search-exhaustion error and
   integration-owned lookalikes.
6. The existing joint-limit verdict protected evaluator, all focused C++ and
   Python suites, and the complete warning-strict catalog.
7. A fresh full-database certificate scan whose counts are reconciled against
   the prior 1,063-frame audit.
8. The unchanged GPU-0 Stage A command in a new output root.

## Scientific acceptance

The implementation experiment succeeds at its narrow hypothesis only if the
exact prior 866-to-867 raw-limit failure is absent and the gate-4 reference
contains all 601 required frames. If a different raw, inertialized, protocol,
or search-exhaustion failure appears, it is recorded as the next result rather
than repaired during the same trial.

Stage A passes only if gates 1 through 7 all pass under the existing thresholds.
If gate 4 passes, the unchanged controller proceeds to the three dynamic gates;
their result is reported without tuning or threshold changes.

## Rollback and provenance

The implementation remains on the existing isolated research branch. The
published motion database, pinned GEAR checkout, policy, and active Reliable
Claude release are never mutated. Every trial uses a new immutable output root.
Rollback is the parent commit before the feasibility-mask implementation; the
original scientific-failure runs remain immutable evidence.
