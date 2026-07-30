# Torch Motion Matching + SONIC Tracker Quickstart

This branch runs flat G1 locomotion with:

1. a native 50 Hz Takara motion folder;
2. an in-process PyTorch/CUDA motion matcher; and
3. the released GEAR SONIC tracker.

It does **not** start `mm_chunk_server`, load the terrain-aware matcher, or
require terrain features. The streamed SONIC reference stays in native
IsaacLab joint order. Internally, startup alone converts the initial joint pose
into MuJoCo order.

## Qualified revision

```text
repository: https://github.com/takaratruong/Motion-Matching.git
branch:     research/g1-low-latency-driver
implementation baseline: 2e6e17ed033676c508b0a2a84cd8bb2df6e49d57
```

Clone or update without merging it into another motion-matching branch:

```bash
git clone https://github.com/takaratruong/Motion-Matching.git
cd Motion-Matching
git fetch origin research/g1-low-latency-driver
git switch --track origin/research/g1-low-latency-driver
```

For an existing clone:

```bash
git fetch checkpoint research/g1-low-latency-driver
git switch research/g1-low-latency-driver
git pull --ff-only checkpoint research/g1-low-latency-driver
```

## Required inputs

The verified launch uses these external inputs:

| Input | Verified cluster path |
|---|---|
| Takara motions | `/home/ubuntu/Downloads/takara_walk_50hz.npz_v0` |
| GEAR checkout | `/home/ubuntu/projects/gear-sonic-worktrees/simulation-lowstate-wait-v6` |
| GEAR commit | `294110cedba01ad764f1e268d57ddf7c1bbf9523` |
| SONIC runtime | `/home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f` |
| Qualified flat source run | `/home/ubuntu/mm-flat-walk-stage-b-r16-hold-resume.tym9Fc/stage-b/stage-b-20260718T005901613783Z-8b63d477` |
| Existing environment root | `/home/ubuntu/projects/motion-matching/resources/g1_terrain` |

The runtime directory must contain:

```text
model_decoder.onnx
model_encoder.onnx
observation_config.yaml
```

The source run supplies the authenticated flat scene and
`known-good/reference-base`. The `--terrain-dir` argument remains part of the
existing child-process environment contract, but this Torch matcher launch
uses the flat scene, sets terrain weight to zero, and reads no terrain features.

The motion directory may contain one or more recursively discovered
`motion.npz` files. Each clip must be native Takara 50 Hz data with 29 joints,
30 bodies, Z-up vectors, and wxyz quaternions.

## Python environment

The qualified cluster already has:

```text
sonic/.torch-mm-venv
Python 3.10
Torch 2.13.0+cu130
CUDA on NVIDIA L40S
MuJoCo, NumPy, pyzmq, and the SONIC integration dependencies
```

Check it:

```bash
sonic/.torch-mm-venv/bin/python - <<'PY'
import torch
import mujoco
import zmq

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mujoco", mujoco.__version__)
print("zmq", zmq.__version__)
PY
```

For another prepared SONIC/IsaacLab cluster, create an isolated environment
without altering the existing SONIC environment:

```bash
python3.10 -m venv --system-site-packages sonic/.torch-mm-venv
sonic/.torch-mm-venv/bin/python -m pip install --upgrade pip
sonic/.torch-mm-venv/bin/python -m pip install -e './sonic[integration,test,torch-mm]'
```

The external GEAR executable, DDS/Unitree runtime, policy assets, X11 display,
and GPU driver must already be installed. This repository does not build or
download those qualified external artifacts.

## Verify before launching

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -q \
  tests.python.test_sonic_torch_motion_data \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_sonic \
  tests.python.test_sonic_manual_demo
```

The tests cover strict motion loading, shared 27-value features, exact CUDA
search, inertialized 46-frame output, transactional SONIC publication, and
Ctrl/input supersession retry.

## Verified cluster launch

Run from the repository root:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.manual_demo \
  --mode interactive \
  --input-source x11 \
  --onscreen \
  --motion-backend torch \
  --motions-dir /home/ubuntu/Downloads/takara_walk_50hz.npz_v0 \
  --torch-device cuda \
  --scene-id sonic-flat-baseline \
  --route-id flat-12s \
  --terrain-weight 0 \
  --chunks 300000 \
  --gear-checkout /home/ubuntu/projects/gear-sonic-worktrees/simulation-lowstate-wait-v6 \
  --source-run /home/ubuntu/mm-flat-walk-stage-b-r16-hold-resume.tym9Fc/stage-b/stage-b-20260718T005901613783Z-8b63d477 \
  --runtime /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root /home/ubuntu/mm-sonic-torch-live
```

Click the MuJoCo viewer when prompted. A ready launch prints lines similar to:

```text
Loading Torch matcher on cuda
Starting GEAR and publishing Torch rows 0..45
FOCUS THE MUJOCO VIEWER
LIVE TORCH MM
SONIC first action ready
SONIC policy command received
torch boundary 00000
```

## Portable launch template

Set paths for the new cluster, then use the same command:

```bash
export MOTIONS_DIR=/absolute/path/to/takara_motion_folder
export GEAR_CHECKOUT=/absolute/path/to/qualified/GEAR
export SONIC_RUNTIME=/absolute/path/to/sonic_runtime
export SOURCE_RUN=/absolute/path/to/qualified_flat_source_run
export ENVIRONMENT_ROOT=/absolute/path/to/existing_environment_root
export OUTPUT_ROOT=/absolute/path/to/output

test "$(git -C "$GEAR_CHECKOUT" rev-parse HEAD)" = \
  294110cedba01ad764f1e268d57ddf7c1bbf9523
test -x "$GEAR_CHECKOUT/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
test -f "$SONIC_RUNTIME/model_decoder.onnx"
test -f "$SONIC_RUNTIME/model_encoder.onnx"
test -f "$SONIC_RUNTIME/observation_config.yaml"
test -d "$SOURCE_RUN/known-good/reference-base"

PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.manual_demo \
  --mode interactive --input-source x11 --onscreen \
  --motion-backend torch --motions-dir "$MOTIONS_DIR" --torch-device cuda \
  --scene-id sonic-flat-baseline --route-id flat-12s --terrain-weight 0 \
  --chunks 300000 \
  --gear-checkout "$GEAR_CHECKOUT" \
  --source-run "$SOURCE_RUN" \
  --runtime "$SONIC_RUNTIME" \
  --terrain-dir "$ENVIRONMENT_ROOT" \
  --output-root "$OUTPUT_ROOT"
```

## Controls

| Keys | Action |
|---|---|
| `W` / `S` | Forward / backward |
| `A` / `D` | Left / right movement |
| `Left Ctrl` + arrows | Strafe/facing control |
| `Space` | Stop |
| `Backspace` | Tear down and start a fresh episode |
| `X` | Exit cleanly |

Every committed 20 ms boundary publishes an overlapping 46-row motion window.
Physics remains fenced until GEAR acknowledges the window’s final row. A newer
key revision can supersede an unsent command; the launcher retries that same
boundary instead of treating normal Ctrl transitions as a crash.

## Troubleshooting

### `GEAR commit mismatch`

Use `simulation-lowstate-wait-v6` at:

```text
294110cedba01ad764f1e268d57ddf7c1bbf9523
```

The older `simulation-control-gate` worktree has a different commit and is
rejected intentionally.

### Viewer waits at `FOCUS THE MUJOCO VIEWER`

Click inside the MuJoCo window once. The X11 controller binds only after it
observes the focused viewer.

### CUDA is unavailable

Check `torch.cuda.is_available()` in the isolated environment. `--torch-device
cpu` is useful for contract testing, but the interactive qualified path uses
CUDA.

### A prior run may still exist

Inspect without killing anything:

```bash
pgrep -af 'mm_sonic.manual_demo|mm_sonic.gated_sim|g1_deploy_onnx_ref|mm_chunk_server'
```

Prefer `X` or Backspace for managed cleanup. After a normal exit, the command
should return no manual-demo, gated-simulator, or GEAR tracker process. The
Torch launch should never show an `mm_chunk_server` process.

### Retained evidence

Each run is written below `--output-root`. It retains simulator/GEAR logs and
the exact transmitted 46-row windows, which are the first place to inspect
after a tracker, publication, or physics-gate failure.

