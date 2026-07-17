# G1 Best-Effort IK Cleanup Design

## Goal

Make terrain IK an optional cleanup of an already safe raw motion-matching
pose. A finite IK failure must not stop locomotion when the raw pose has
passed the existing footprint and pose-clearance certificates. The running
visualizer must show whether IK was applied, skipped, or disabled.

This is a narrow unblocker before the separate G1 mesh-visualization task. It
does not tune motion matching, terrain, thresholds, or IK geometry.

## Evidence and Root Cause

The current transaction accepts a candidate only when both its raw and IK
branches certify. On the 32-frame mixed-multilevel route, the raw branch is
safe on frames 19, 23, 25, 30, and 31, but the IK branch finite-rejects. The
transaction then rejects and latches the whole frame. Six range-diverse
recovery candidates still fail correlated physical IK/landing checks at frame
19, so additional candidate search is not a liveness solution.

The architectural fault is that optional pose cleanup is acting as a safety
gate. The raw footprint and pose-clearance certificates are the correct safety
boundary; IK may improve a safe pose but may not invalidate it merely because
the cleanup solver cannot produce a certified result.

## Accepted Cleanup State

`G1FrameAcceptedDiagnostic` will carry an explicit cleanup disposition and
stop reason:

- `off`: IK was disabled for this accepted frame; reason is `none`.
- `applied`: IK was enabled and its complete branch, including IK pose
  clearance, certified; reason is `none`.
- `skipped`: IK was enabled, the raw branch certified, and the IK branch
  finite-rejected; reason is the authenticated non-`none`
  `G1IkStopReason` from that IK attempt.

The disposition is part of accepted publication state. It is not inferred
from transient working memory, and equality, validity, copy, and hashing tests
must include it.

## Candidate Transaction

Candidate evaluation remains ordered as common stages, raw certificate, then
the existing IK branch. The hidden IK branch may continue to run when IK is
disabled so existing dual-branch diagnostics and parity coverage remain
available; the accepted disposition is still `off` and the published pose is
raw.

The outcomes are:

1. A common-stage or raw-certificate finite rejection remains a candidate
   rejection. Recovery search proceeds exactly as it does now, and exhaustion
   still publishes an atomic rejected frame and safe-stop latch.
2. Once the raw branch certifies, an IK finite rejection is converted into an
   authenticated cleanup skip. The candidate is accepted immediately using
   the exact raw state. It does not start recovery, publish `frame_rejected`,
   request a safe stop, or set the next-frame latch.
3. A fully certified IK branch is published only when IK is enabled. When IK
   is disabled, the exact raw state is published even if the hidden IK branch
   certifies.
4. Any global error remains a global error. An IK finite rejection that cannot
   provide a valid branch-local rejection record and reason is a global error,
   not an unauthenticated skip.

On a skipped frame, the failed IK branch does not advance persistent IK state.
The next frame retries cleanup from the last accepted safe state.

No clearance, contact, reach, landing-patch, footprint, or motion-matching
threshold changes are permitted.

## Logging

The existing 305-column CSV schema remains byte-for-byte unchanged. Accepted
rows use the existing columns as follows:

| Cleanup state | `ik_applied` | `ik_safe_stop_requested` | `ik_stop_reason` | `ik_candidate_rejected` | `frame_rejected` | `ik_safe_stop_latched` |
|---|---:|---:|---|---:|---:|---:|
| off | 0 | 0 | `none` | 0 | 0 | 0 |
| applied | 1 | 0 | `none` | 0 | 0 | 0 |
| skipped | 0 | 0 | authenticated reason | 1 | 0 | 0 |

Rejected common/raw frames keep the existing rejected-row grammar. The log
checker must accept the new `skipped` combination only when `ik_enabled=1`,
and must continue rejecting inconsistent combinations.

## Visual Signal

The overlay beneath `Transactional terrain IK` will show one accepted-state
label:

- green: `IK ACTIVE`
- amber: `IK SKIPPED — <reason>` using `g1_ik_stop_reason_name`
- gray: `IK OFF`

The label reads only `frame_runtime.accepted_diagnostic`, so a failed working
attempt cannot flicker or misreport the pose that is actually rendered. A
skipped frame renders the certified raw pose.

## Tests and Runtime Gate

Implementation follows test-first development.

The transaction tests must first demonstrate the old failure and then prove:

- raw-safe plus IK-finite produces an accepted raw frame with `skipped` and
  the exact reason;
- skipped acceptance does not call recovery, reject the frame, latch a stop,
  or mutate failed IK state into the accepted state;
- common/raw finite failure still recovers or fails closed;
- IK global failure still aborts;
- IK-off publishes raw with `off`;
- certified IK-on publishes IK with `applied`;
- strict and compatible-fast candidate traces remain identical.

Logging tests must prove all three accepted-row combinations and reject
forged mixtures without changing the 305-column schema. The controller build
test must prove all three literal labels are wired to accepted diagnostics.

The finite integration gate is the existing 32-frame, exact-25-Hz,
mixed-multilevel IK-on route. It passes only when:

- all 32 frames advance without a global error;
- no frame is rejected or latched solely because IK cleanup failed;
- every skipped frame has a safe raw pose and an authenticated reason;
- the previously unsafe raw frame 18 still recovers rather than being
  published raw;
- accepted rows at former blockers 19, 23, 25, 30, and 31 either apply IK or
  explicitly report a skip;
- the visualizer renders the same accepted status represented in its log.

After this gate is independently reviewed, the checkpoint is pushed and a
new IK-on visualizer is launched. The next separate milestone is loading the
G1 robot mesh to inspect ankle and foot alignment; this design makes no mesh
changes.

## Global Constraints

- Simulation cadence remains exact float32 `0.04` seconds (25 Hz).
- Terrain assets and motion data remain separate.
- Heading and travel direction remain independent.
- The running old visualizer is not inspected, signaled, or terminated.
- Every coherent test-backed checkpoint is committed and pushed to
  `checkpoint/g1-footprint-task6` immediately.
