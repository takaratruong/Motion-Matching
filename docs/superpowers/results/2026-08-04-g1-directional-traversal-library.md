# G1 directional stair traversal library

## Outcome

The offline pipeline now produces contact-valid G1 stair traversals at three
approach angles and their time-reversed descents:

| primitive | frames | max joint step | max root step | unsupported |
|---|---:|---:|---:|---:|
| horizontal ascent/descent | 91 | 0.291 rad | 0.019 m | 0 |
| 45-degree ascent/descent | 91 | 0.238 rad | 0.020 m | 0 |
| head-on ascent/descent | 96 | 0.458 rad | 0.032 m | 0 |

The pipeline also produces an elevated horizontal-to-45-degree transition. It
uses an ARDY-generated motion prior, an explicit two-step contact schedule,
terrain IK, and temporal repair. Its validated metrics are:

- 50 frames
- 0 unsupported frames
- 0.291 rad maximum joint step
- 0.028 m maximum root step
- 0.0071 m maximum stance contact error
- -0.0229 m minimum sole clearance
- 5 minimum supported sole samples

Finally, the horizontal ascent, elevated turn, and diagonal descent compose
into a 230-frame end-to-end route with 0 unsupported frames, 0.0071 m maximum
stance error, -0.0233 m minimum sole clearance, and at least 5 supported sole
samples. The route is at:

`build/g1-traversal-library/horizontal-turn-diagonal-route/traversal.npz`

## What the experiments established

A nearest compatible rigid splice is not sufficient for an elevated direction
change. The closest valid horizontal and diagonal support states are about
15.5 cm apart, and translating the outgoing suffix invalidates its later
contacts.

Raw ARDY output is also not terrain-valid by itself. All eight sampled motions
had unsupported or penetrating frames; the best raw sample still had 8
unsupported frames and 0.177 m sole penetration. ARDY is useful as a whole-body
motion prior, but terrain validity must come from an explicit contact schedule
and terrain projection.

This agrees with the broader structure used in terrain trajectory optimization:
motion/phase proposals and terrain/contact feasibility are separate constraints.
Relevant references are the official [ARDY repository](https://github.com/nv-tlabs/ardy),
[TAMOLS](https://arxiv.org/abs/2206.14049), and
[Phase-Guided Terrain Traversal](https://openreview.net/pdf?id=iDyNnoitXA).

## Reproduction

The official ARDY repository was qualified separately at
`/home/ubuntu/projects/ardy`, commit
`693f74d13b3d04a0a22ce127ee79c929dd89756b`. ARDY is only needed to resample
the motion prior. The checked-in terrain bridge consumes its CSV output and
runs in the motion-matching Torch environment.

Generate the horizontal ascent:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_supported_step_up_search.py \
  --source-dataset build/torch-grail-terrain-full-v1 \
  --source-clip grail-stair_p2-f408d0e157701b54dede \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --minimum-rise-m 0.28 --maximum-rise-m 0.42 \
  --traversal-angle-deg 0 --maximum-placements-per-event 1 \
  --output build/g1-step-up-search/horizontal-current-check
```

Use the same command with `--traversal-angle-deg 45` and output
`build/g1-step-up-search/angle-45-current` for the diagonal ascent.

Project the selected ARDY prior onto the terrain:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_ardy_terrain_bridge.py \
  --ardy-qpos-csv build/g1-traversal-library/ardy-horizontal-to-diagonal/relative-samples/relative-samples_07.csv \
  --incoming build/g1-step-up-search/horizontal-current-check/step-up.npz \
  --outgoing build/g1-step-up-search/angle-45-current/step-up.npz \
  --incoming-frame 88 --outgoing-frame 90 \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-traversal-library/ardy-horizontal-to-diagonal/terrain-bridge
```

Repair the two endpoint IK branches:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_temporal_traversal_repair.py \
  --input build/g1-traversal-library/ardy-horizontal-to-diagonal/terrain-bridge/transition.npz \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --window-radius-frames 14 \
  --output build/g1-traversal-library/ardy-horizontal-to-diagonal/final
```

Compose the complete route:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_compose_traversals.py \
  --segment 'build/g1-step-up-search/horizontal-current-check/step-up.npz@0:89' \
  --segment build/g1-traversal-library/ardy-horizontal-to-diagonal/final/traversal.npz \
  --segment build/g1-traversal-library/diagonal-descent/traversal.npz \
  --output build/g1-traversal-library/horizontal-turn-diagonal-route
```

Validate every scheduled contact and sole sample in the composed route:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_validate_traversal.py \
  --input build/g1-traversal-library/horizontal-turn-diagonal-route/traversal.npz \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-traversal-library/horizontal-turn-diagonal-route/validation.json
```

Render a contact sheet or launch passive playback:

```bash
MUJOCO_GL=egl PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_stair_pivot_viewer.py \
  --connector build/g1-traversal-library/horizontal-turn-diagonal-route/traversal.npz \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --loop
```

## Remaining limitation

The head-on primitive is valid but temporally weaker than the horizontal and
45-degree primitives. Its source placement begins too deep in the target stair
for clean regeneration, so the current version uses temporal repair. The next
library-quality improvement is to search a new head-on placement/source event
rather than increasing repair strength.
