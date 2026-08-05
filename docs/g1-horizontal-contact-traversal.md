# G1 horizontal contact traversal

This experiment is the first reproducible 30 Hz kinematic baseline across
the target staircase. Sparse proxy keyframes constrain the released
MotionBricks G1 backbone; the generated intervals are then corrected with
contact-phase stance anchors and full-sole footprint checks:

1. supported step-up from the ground;
2. same-heading uneven locomotion with one foot on each stair level;
3. a declared MotionBricks drop from the unsupported side;
4. stable double-support landing on the ground.

The checked-in reference artifact is
`build/g1-traversal-library/horizontal-full/motionbricks-keyframes-v27`.
It contains 180 frames (6.0 seconds), including a 24-frame (0.8 second)
declared drop interval. Its generator gates full-sole penetration at 25 mm,
stance slide at 10 mm, and endpoint root error at 100 mm.

## Rebuild

From the repository worktree:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONPATH=sonic/python:/home/ubuntu/projects/gear-sonic-worktrees/motionbricks-contact-transition/motionbricks:/home/ubuntu/projects/gear-sonic-worktrees/motionbricks-contact-transition/motionbricks/.deps/python \
sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_motionbricks_terrain_task_actor.py \
  --motionbricks /home/ubuntu/projects/gear-sonic-worktrees/motionbricks-contact-transition/motionbricks \
  --proxy-route build/g1-traversal-library/horizontal-full/motionbricks-contact-exit-v4/traversal.npz \
  --target-dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output build/g1-traversal-library/horizontal-full/motionbricks-keyframes-v27
```

The output directory contains `traversal.npz`, `metrics.json`,
`candidate-metrics.json`, and `proxy-plan.json`.

The proxy input is also checked in at
`build/g1-traversal-library/horizontal-full/motionbricks-contact-exit-v4/traversal.npz`.
Its SHA-256 is
`391c81d2a0744663c40ab11ef57f3caca3be2fb24c02eee223501ab90a686550`.
The reference traversal SHA-256 is
`a6410c768828ac8909976266264241efe19a559f6757368bc7adf41e28611c8e`;
the reference metrics SHA-256 is
`8474330894335388467026e60488dc8faab2a5d016d920869307a8e730dea7cb`.
The generator is deterministic with MotionBricks seed 7 on this cluster.

## Play

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  resources/run_g1_stair_pivot_viewer.py \
  --connector build/g1-traversal-library/horizontal-full/motionbricks-keyframes-v27/traversal.npz \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --frames-per-second 30 \
  --loop
```

Controls: Space pauses, the arrow keys step, Backspace rewinds, and `X`
exits.

## Current qualification

The generated route has 5.49 mm maximum scheduled-stance slide, 24.07 mm
worst sole penetration, and six of seven sole samples supported in the worst
stance.
The legacy traversal validator still rejects generated routes because they
have no source-frame provenance and because it requires every sole sample to
touch one height; a single safe overhang above the lower tread is therefore
reported as 159 mm of "complete sole contact error." The largest remaining
motion-quality defect is a 0.69 rad knee step at the first support switch.
MotionBricks' generated contact channels do not independently validate the
schedule: a thresholded interpretation labels most generated interval frames
as flight. The current projection therefore treats the larger of each foot's
heel/toe channel as a phase hint and always schedules one stance foot outside
the explicit drop. This is a visual/kinematic baseline, not a controller-ready
contact certificate.
