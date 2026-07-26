# Tabletop-Only Reach Viewer Design

## Goal

Show only the corrected `tabletop_soma` reach corpus so its inbound approaches
can be reviewed without candidates from the original reach dataset.

## Design

Build a separate reach pack from `tabletop-review-v2` and
`tabletop-annotations-v2.json`. The pack contains the 222 accepted captured
tabletop reaches and their 222 mirrored right-hand variants. The combined
`reach-pack-v5` remains unchanged.

Launch the existing coverage viewer against this tabletop-only pack. Playback
retains the current one-way 3.6-second start-to-contact behavior and holds at
contact.

## Verification

Confirm the new manifest contains 444 total reaches, all sequence IDs begin
with `tabletop_soma/`, the viewer tests pass, and the running viewer loads the
new pack.
