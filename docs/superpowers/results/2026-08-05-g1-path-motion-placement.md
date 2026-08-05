# G1 directed-path rigid motion placement result

Date: 2026-08-05

## Question

Given only a directed path and a target terrain height grid, can the system
search the complete GRAIL terrain corpus for an existing motion window that
can be rigidly placed on the path without joint edits?

The first decisive case is the approved horizontal traversal across
`grail-stair_p1-db7949fce1b2e48d39f4`.

## Search contract

- Search all height-grid GRAIL clips without a source family, clip, or frame
  override.
- Use approximate nominal footprints only to form a contact-height query.
- Validate the selected raw motion using its actual foot contacts and sole
  geometry.
- Permit only global yaw, XY translation, and root-Z translation.
- Reject candidates for heading, lateral-path, stance-height, or sole
  penetration violations.
- Report uncovered path intervals instead of hiding a gap.

## Command

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_path_motion_placement.py \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --target-scene grail-stair_p1-db7949fce1b2e48d39f4 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --path-start -1.2795985755 0.9682251052 \
  --path-stop 1.2590216406 0.9682251052 \
  --step-width-m 0.20 \
  --workers 32 \
  --coarse-results 2000 \
  --placement-shortlist 800 \
  --output build/g1-path-motion-placement-horizontal-auto-v7
```

## Result

The search scanned 12,645 terrain clips and retained 198,503 diverse coarse
windows before shortlist selection. It automatically selected:

- Source: `grail-curb-89af6ff090a6439ba88f`
- Raw frames: `[6, 211)`
- Path length: 2.5386202161 m
- Motion coverage: 2.5503714302 m
- Uncovered intervals: none
- Joint positions edited: false
- Heading error p95: 0.1545728157 rad (8.86 degrees)
- Maximum root lateral error: 0.1316746597 m
- Maximum stance-height error: 0.0189284161 m
- Minimum sole clearance: -0.0188302201 m

The contact-height sequence contains the intended low/high alternation and
returns to two low contacts at the dismount:

```text
0.000, 0.185, 0.000, 0.186, 0.000, 0.000 m
```

The automatically selected result is the same raw motion previously found by
manual inspection, but the automatic command receives no source identity.

## Determinism

An independent second full-corpus run to
`build/g1-path-motion-placement-horizontal-auto-v8` selected the same source
and frames and produced byte-identical artifacts:

```text
placements.json  109bd9775537775a2df0a2643b1db17d1c22632af382fe3eb596fd086c20aafd
coverage.json    caea669c6109f7335bace8f49d01a7be10ddf9cce7c4d6675197f474f0649733
traversal.npz    478bbb427b16541bfc7f3e20d209e1702d3e06169b0dddb85e00fb02365dd0c6
```

## Interpretation

This is a positive answer for the first horizontal-path feasibility test:
path/contact retrieval can find and rigidly place an existing GRAIL motion
without IK, joint edits, semantic clip labels, or a manually selected source.

It is not yet evidence for arbitrary-angle generalization or a complete path
graph. The next work is to:

1. run the same path-only search at 45 degrees and head-on;
2. add compatibility edges and repeated-window chaining for paths longer than
   one raw window;
3. validate splice pose, velocity, support, and contact continuity;
4. report explicit graph gaps where the corpus has no compatible transition.

