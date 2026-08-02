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

## 4. Drive the matcher live in native MuJoCo

```bash
DISPLAY=:1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -B -m \
  mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-stair-small \
  --config sonic/configs/experiments/torch_stair_small.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda
```

Click the MuJoCo window, then use:

- `W`, `A`, `S`, `D`: command matcher-world forward, left, backward, and right;
- `Space`: stop;
- `Backspace`: reset the matcher and rendered robot; and
- `X`: exit.

This is a live kinematic experiment. The Torch matcher supplies every root and
joint state. MuJoCo renders the authenticated stair heightfield, copies each
state into `qpos`, and runs forward kinematics only. It does not integrate
physics and does not launch SONIC, GEAR, or a tracking policy.

To drive the qualified stable-endpoint multi-horizon terrain matcher instead:

```bash
DISPLAY=:1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -B -m \
  mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:0 \
  --multi-horizon
```

The controls are unchanged. The right overlay shows the committed horizon,
stable endpoint, and entry/outcome costs. A coverage failure remains visible in
the window instead of closing it; change command or press Backspace to retry.

## 5. Record an MP4

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
