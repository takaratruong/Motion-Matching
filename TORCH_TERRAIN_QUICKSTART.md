# Native Torch Stair Kinematics Quickstart

This experiment answers one narrow question before depth learning or SONIC
tracking: can privileged local height observations make the native 50 Hz Torch
motion matcher select useful stair kinematics?

It is deliberately kinematic. It never calls a physics step or a SONIC policy.

## 1. Build the pinned five-clip dataset

```bash
cd /home/ubuntu/projects/motion-matching

python3 resources/build_g1_torch_stair_slice.py \
  --grail-root /home/ubuntu/datasets/GRAIL/data/stair_p1 \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --flat-motion /home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz \
  --output build/torch-stair-small
```

The output is a deterministic native Z-up, wxyz, 50 Hz motion folder containing
the Takara flat control and four explicitly pinned GRAIL recordings. Generated
data stays under ignored `build/`.

## 2. Run flat, legacy, and dense conditions

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m \
  mm_sonic.torch_terrain_rollout \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --conditions flat legacy dense \
  --device cuda \
  --output build/torch-stair-small-results
```

Each condition writes:

- `rollout.npz`: saved 50 Hz kinematics and diagnostics, with no pickle fields;
- `metrics.json`: hashes, acceptance inputs, and latency percentiles;
- `events.jsonl`: search and transition events; and
- `resolved_config.json`: the derived command, ascent reference, and identities.

When flat and dense are both requested, the parent output also receives
`acceptance.json`.

## 3. Inspect the dense rollout

```bash
MPLBACKEND=TkAgg PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.torch_terrain_viewer \
  --run build/torch-stair-small-results/dense \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
```

Controls:

- `Space`: pause or resume;
- `Left` / `Right`: move through saved frames;
- `R`: restart; and
- `Escape`: close.

The viewer authenticates the rollout, dataset manifest, motion, and height grid
before creating a window. It is not a second matcher and only calls MuJoCo
forward kinematics to draw saved root/joint states.

## 4. Record an MP4

```bash
MPLBACKEND=Agg PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.torch_terrain_viewer \
  --run build/torch-stair-small-results/dense \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --output-mp4 build/torch-stair-small-results/dense.mp4
```

For a single diagnostic image, replace `--output-mp4 ...` with
`--frame 150 --output-png build/dense-frame-150.png`.

## Interpretation

The strict acceptance rule is intentionally harder than “the root went up.”
Dense must select stair motion by the pre-riser deadline, achieve reference
progress and height gain, preserve foot clearance, and beat flat by landing
success or penetration integral. A failure is evidence about the kinematic
matcher/transition representation; it is not evidence about SONIC stability.
