# G1 Semantic Dismount Admission Design

## Goal

Ensure horizontal-grid step-down seeds contain an actual controlled terrain
descent and supported landing instead of flat walking, prolonged flight, or an
unfinished aerial step.

## Root Cause

The phase search matches relative contact signatures and path coverage but
does not require dismount semantics. Existing selections therefore include
clips with no source contact-height drop, flights of 25–29 frames, and
unsupported terminal poses.

## Admission Contract

Apply a hard gate only to `dismount` source windows:

- The window begins with at least 8 consecutive frames where either foot is
  supported.
- A later contact is at least 0.12 m below the initial supported surface.
- Both feet establish contacts within 0.04 m of the lower landing surface.
- No unsupported run exceeds 20 frames.
- The window ends with at least 8 consecutive frames where either foot is
  supported.

The terminal pose may be single support. Requiring terminal double support
would reject the previously approved curb descent, which lands both feet
sequentially and continues naturally.

## Data Flow

`_semantic_dismount_window` evaluates a `RawMotionWindow` against the source
support and per-foot surface-height arrays. `_scan_one` routes only dismount
tasks through the admitted subset. Mount routing retains the stable-boundary
gate and interior tasks retain the complete source-window set.

## Failure Behavior and Validation

If no admitted window survives terrain placement, the dismount is infeasible;
the system must not substitute flat or unsupported motion. Unit tests encode
the approved descent profile and reject flat motion, excessive flight, a
one-foot landing, and unsupported endings. The complete 20 cm grid is rerun,
audited directly from source profiles, and displayed as a multi-clip playlist.

