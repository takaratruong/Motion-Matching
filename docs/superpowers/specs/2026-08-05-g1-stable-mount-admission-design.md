# G1 Stable Mount Admission Design

## Goal

Prevent horizontal-grid step-up seeds from ending in an aerial or
single-support pose while preserving the existing terrain-placement search.

## Problem

The phase search ranks source windows by contact-signature similarity,
coverage, heading, clearance, and stance error. It does not require a mount
window to end at a usable splice boundary. Consequently, a terrain-valid
window can contain a long unsupported interval or stop while the swing foot is
still raised. Holding that final frame in the seed viewer exposes the invalid
boundary as a step in midair.

## Design

Apply a support-boundary admission gate only to `mount` task windows before
contact-signature scoring:

- Reject a window with more than four consecutive frames where neither foot is
  supported.
- Reject a window with fewer than eight consecutive double-support frames at
  its end.
- Leave `interior` and `dismount` task windows unchanged.
- Continue using the existing terrain-placement certification and ranking for
  admitted mount windows.
- If no admitted terrain-valid mount exists, report the phase as infeasible.
  Never substitute an unsupported seed merely to increase grid coverage.

At 50 Hz, the allowed unsupported run is at most 80 ms and the required
terminal landing is at least 160 ms.

## Components and Data Flow

`_stable_mount_window` receives a `RawMotionWindow` and the source clip support
mask. It returns a Boolean admission decision. `_scan_one` builds both the
normal source-window set and the admitted mount subset. Task scoring uses the
subset for mount phases and the complete set for all other phases.

The output schema remains unchanged. A post-run audit recomputes unsupported
runs and terminal double-support lengths from every selected mount and must
find no contract violations.

## Validation

Unit tests cover rejection of long aerial intervals, rejection of
single-support endings, acceptance of a brief flight followed by a stable
landing, and proof that interior tasks retain ordinary walking windows. The
complete 20 cm grid is then regenerated, audited, rendered, and played as a
multi-lane mount playlist.

