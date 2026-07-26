# SOMA Tabletop Reach Corpus Extension Design

## Goal

Add the 12 continuous standing tabletop recordings in
`/home/ubuntu/Downloads/takara_tabletop_delivery/03_retargeted_soma_csv/`
to the reusable G1 reach corpus. Each source file contains approximately 20
left-hand reach-and-return cycles. The new data must use the existing reach
construction pipeline rather than introducing another segmentation or clip
representation.

## Source Contract

The source directory contains wrist-bias-corrected SOMA CSV files recorded at
100 Hz. A trusted source adapter will validate the exact 36-column schema:

- `Frame`;
- root translation XYZ in centimeters;
- intrinsic ZYX root Euler rotation in degrees, stored in XYZ columns; and
- the established 29 G1 joint DOFs in degrees.

The adapter will require finite numeric values, contiguous zero-based frame
indices, a positive source rate, unique sequence IDs, and the exact expected
joint ordering. It will convert root translation to meters, root rotation to a
normalized quaternion, and joint angles to radians, producing the same
validated `qpos` source record consumed by the existing canonical converter.
Every source record will retain its file path and SHA-256 provenance.

Sequence IDs will use a `tabletop_soma/` namespace so they cannot collide with
the original reach recordings. The source is left-hand authored; the existing
bilateral mirror stage remains responsible for right-hand alternatives.

## Existing Segmentation and Clip Semantics

No new splitter will be added. After conversion and resampling to 25 Hz, the
new sources pass unchanged through the existing wrist-trajectory proposal,
annotation, and clip-construction pipeline used for the original corpus.

For every detected neutral-to-reach-to-neutral cycle:

1. the departure frame begins the reusable outbound approach;
2. the confirmed grab frame is the contact anchor;
3. the recorded samples after contact form the paired pull-back to neutral; and
4. the contact index separates the two portions inside one canonical reach
   record.

Runtime reach search consumes only departure through contact. Successful
pickup uses the same clip's post-contact samples for the recorded pull-back.
The existing 3.6-second outbound cap, approach-direction derivation,
return-stop detection, manual annotation model, and mirroring behavior remain
unchanged.

## Review and Publication Flow

The tabletop sources receive their own review directory and annotation
document. This preserves the checksums and 192 accepted annotations in the
existing `review-v2` corpus.

The build layer will accept more than one validated review-and-annotation
input. It will:

1. build captured canonical reaches from each input;
2. reject duplicate reach and proposal IDs;
3. combine the captured reaches;
4. apply bilateral mirroring once to the combined captured set; and
5. assemble one deterministic reach pack with per-source provenance.

The initial combined output is `build/g1-reaches/reach-pack-v4`. Existing
packs, reviews, annotations, and generated reports remain untouched. The
manifest will report captured, mirrored, paired-return, unavailable-return,
and per-corpus counts.

## Validation and Visualization

Focused tests will cover:

- SOMA schema, units, ZYX Euler conversion, source rate, and file hashing;
- rejection of malformed headers, discontinuous frames, non-finite values,
  duplicate sequence IDs, and wrong joint order;
- feeding parsed SOMA records through the existing 25 Hz canonical converter;
- recovering multiple proposals from one continuous source;
- preserving the existing departure/contact/return construction;
- combining multiple review corpora without invalidating old annotations;
- duplicate-ID rejection and deterministic combined pack output; and
- bilateral captured-left/mirrored-right provenance.

The real-data run will report proposal counts per file. Counts far from the
expected approximately 20 cycles will be inspected in the existing annotation
viewer rather than fixed by equal-duration splitting. After annotation, the
combined pack will be loaded in the current reach coverage viewer to confirm
that the new tabletop trajectories, complete outbound motion, and paired
returns are selectable.

## Scope

This extension changes source ingestion and multi-corpus publication only.
It does not alter the existing segmentation algorithm, contact/return
semantics, IK, collision policy, orientation mode, locomotion, or diffusion
stitching.

## Success Criteria

The work is complete when all 12 SOMA sources convert with traceable
provenance, each continuous file yields plausible independent reach proposals,
confirmed clips preserve both outbound and paired pull-back samples, a
combined v4 pack contains the previous accepted reaches plus the new tabletop
reaches and mirrors, all protected tests pass, and the current viewer can
search and animate the combined corpus.
