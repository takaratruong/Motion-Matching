# PFNN Terrain Transfer Gate Design

## Goal

Prove that one released PFNN terrain-motion example transfers to the native
29-DoF Unitree G1 before adding GRAIL data or training another controller.  The
proof must show the approved G1 retarget on the exact terrain surface produced
by PFNN's preprocessing, not terrain inferred from the G1 motion.

## Scope

The gate uses `WalkingUpSteps01_000.bvh`, source frames 8160 through 8279 at
120 Hz.  Those are the frames whose retargeted motion the user already
approved.  The terrain fit is the rank-zero fit for the enclosing PFNN
footstep cycle beginning at source frame 8135 and ending at source frame 8229.
The fit's expanded source window covers the complete displayed interval.

This gate does not build a corpus, import GRAIL, train PFNN, alter runtime
control, or modify acceptance thresholds.  Those tasks remain blocked until
the transfer gate passes.

## Source Authority

The implementation uses the released files under
`/home/ubuntu/datasets/pfnn/pfnn`:

- `generate_patches.py`, including random seed 2, 128 by 128 patches, and the
  ordered list of source heightmaps;
- `generate_database.py`, including position scale 5.6444, 60 Hz preprocessing,
  `patchfunc`, contact detection, terrain ranking, vertical alignment, and the
  linear RBF residual fit;
- `data/animations/WalkingUpSteps01_000.bvh` and its footsteps file;
- every released heightmap named by `generate_patches.py`.

Derived files carry SHA-256 receipts for all inputs and the selected patch.
The implementation reproduces only the requested fit; it does not run the
multi-hour full database build.

## Terrain Extraction

The extractor deterministically regenerates PFNN's patch candidates in their
original order.  It loads the selected motion interval using PFNN's own BVH,
animation, quaternion, contact, and RBF utilities.  It then runs the original
terrain-error terms for the enclosing footstep cycle and chooses the same
rank-zero patch as `np.argsort(terrain_error)[0]`.

The saved terrain artifact contains:

- the selected raw 128 by 128 patch and its source heightmap/crop transform;
- the horizontal contact centroid used by PFNN;
- the base vertical alignment;
- the fitted linear RBF residual parameters;
- a dense evaluation grid covering the displayed motion and its foot probes;
- source hashes and numeric fit diagnostics.

The authoritative terrain query is the fitted continuous PFNN function:
bilinear patch sampling, then vertical alignment, then the RBF residual.  Both
contact metrics and the rendered mesh use evaluations of this function.

## Coordinates and Units

PFNN evaluates horizontal coordinates `(x, z)` and vertical height `y` after
multiplying BVH positions by 5.6444.  GMR's pinned Nokov loader applies the
matrix

```
[[1, 0, 0],
 [0, 0,-1],
 [0, 1, 0]]
```

and divides by 100 before the existing retarget adapter restores the PFNN
scale.  Therefore a fitted PFNN point `(x, y, z)` is rendered in the G1 world
as `(x, -z, y) / 100`.  Tests verify this transform against source and GMR root
samples; no fitted terrain is rescaled to the G1 morphology.

## Retarget Comparison

The first comparison preserves the already-approved GMR joint solution and
removes only the flat-ground postprocess.  It places that unmodified motion on
the exact fitted PFNN terrain and measures both G1 heel and toe probes against
the continuous terrain function on PFNN stance frames.

If the comparison fails, the diagnostic must be preserved before any
correction.  A later terrain-contact correction may change root height and leg
joints offline, but it may not change time, horizontal root trajectory,
terrain, upper-body joints, joint limits, or the source receipt.  No corrective
solver is part of the initial visualization step.

## Visualization

The viewer triangulates a dense sample of the fitted PFNN function and adds it
to the pinned G1 MuJoCo model.  The full fitted surface is rendered as one
continuous mesh.  It must not use the existing `auto-steps` platform builder.

Four sole probes are overlaid per frame:

- green when a stance probe is within 2 cm of the fitted surface;
- red when it penetrates by more than 1 cm;
- yellow when a stance probe floats by more than 2 cm;
- blue for a swing probe.

The viewer retains pause, single-frame seek, restart, camera follow, and escape
controls.  It prints the current frame, stance state, and each sole-to-terrain
gap so a visual mismatch is traceable to exact numbers.

## Acceptance

The transfer gate passes only when all of the following are true:

1. Patch generation and selection reproduce across two fresh processes with
   identical receipts and evaluated grids.
2. Terrain queries used by the report and viewer agree within 1 mm over every
   rendered sole probe.
3. The G1 artifact remains finite, at 120 Hz, within native joint limits, and
   has no joint step above 0.25 radians.
4. Every stance probe penetrates by no more than 1 cm and at least one heel or
   toe probe for each stance foot is within 2 cm of the surface.
5. The user visually confirms that the G1 motion follows the rendered PFNN
   surface without twisting, obvious floating, or obvious clipping.

A failed condition produces a rejected report with the first failing frame,
foot, probe, signed gap, source hashes, and selected-patch receipt.  Failure
does not silently substitute another patch or generate terrain from the G1.

## Test Strategy

Tests are written before implementation and cover deterministic patch
selection, original PFNN bilinear sampling, RBF residual reproduction,
footstep-cycle selection, coordinate conversion, mesh/query agreement,
artifact tampering, stance-gap classification, and rejection of the
`auto-steps` terrain mode as transfer evidence.  The focused automated suite
is followed by two-process extraction, one real retarget comparison, and the
interactive viewer gate.
