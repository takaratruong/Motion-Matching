# Motion Matching to GEAR-SONIC

This directory is an isolated integration package for driving the pinned
GEAR-SONIC deployment from the motion-matching runtime. It does not vendor or
modify GEAR-SONIC, model checkpoints, terrain artifacts, generated references,
or run evidence.

Every run must supply the GEAR checkout, policy checkpoint, observation
configuration, optional encoder checkpoint, terrain directory, and source G1
MJCF explicitly. `mm_sonic.external` resolves and validates those paths,
confines outputs away from input trees, verifies the checkout against
`configs/gear_sonic.lock.json`, and records deterministic SHA-256 identities.

Python packaging is rooted here rather than at the repository root. A local
environment may be created at `sonic/.venv`; builds and run artifacts belong in
the ignored `sonic/build` and `sonic/runs` directories.

```bash
python -m venv sonic/.venv
sonic/.venv/bin/pip install -e 'sonic[integration]'
PYTHONPATH=sonic/python sonic/.venv/bin/python -m unittest \
  tests.python.test_sonic_external -v
```

## In-process Torch motion matcher runtime

The in-process Torch motion matcher runs in its own candidate-local virtual
environment, `sonic/.torch-mm-venv`, so that the optional Torch dependency never
enters the baseline import paths or the shared `sonic/.venv`.

> **Warning:** `sonic/.venv` is a shared symlink used by the baseline runtime
> and must never be created, deleted, or otherwise modified by the Torch matcher
> workflow. Always target the separate `sonic/.torch-mm-venv` directory, which is
> git-ignored.

Create and verify the isolated environment:

```bash
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m venv \
  --system-site-packages sonic/.torch-mm-venv
sonic/.torch-mm-venv/bin/pip install -e 'sonic[integration,torch-mm]'
sonic/.torch-mm-venv/bin/python - <<'PY'
import sys
import mujoco
import numpy
import torch
import zmq
print(sys.version.split()[0])
print(torch.__version__)
print(torch.cuda.is_available())
print(numpy.__version__, mujoco.__version__, zmq.__version__)
PY
```

Expected: Python `3.10.x`, Torch `2.13.0+cu130`, `True`, and successful imports.
Confirm `readlink -f sonic/.venv` is unchanged before and after. Run the focused
native Takara loader tests with this interpreter; they import only NumPy and
never Torch:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_torch_motion_data
```

The `torch-mm` extra declares only `torch>=2.13`; the native Takara loader in
`mm_sonic.torch_motion_data` itself imports no Torch, so baseline test discovery
never depends on the optional runtime.

## Privileged G1 terrain motion matching

The globally informed kinematic bootstrap combines the 173-clip Takara/BONES
flat bank with the clean 500-clip G1 stair archive. It supports two paths:

- `CarpetTerrainMotionMatcher` performs rolling full-bank search using future
  surface heights beneath both feet at 0/15/30/45 frames, matching the terrain
  feature used by Learned Motion Matching.
- `build_target_specific_warped_library` compiles a mechanically audited G1
  source traverse onto an exact known target staircase. Leave-one-target-out
  evaluation removes the target motion entirely before online search.

Run the current scripted diagonal-entry evaluation on one L40S:

```bash
sbatch sonic/run_carpet_terrain_eval.sbatch
```

Launch the dedicated clean-kinematic Switch-controller viewer:

```bash
sbatch sonic/run_privileged_warped_viewer.sbatch
```

Launch the globally informed four-way playground instead:

```bash
sbatch sonic/run_privileged_course_viewer.sbatch
```

That scene has a finite raised platform with a staircase on every side.  The
up/down target clips are both removed from retrieval; separately warped and
mechanically audited G1 sources are shared across all four placements.  Global
character position and command direction select a nearby physical portal,
while ordinary pose/trajectory matching selects the motion.  The platform
therefore tests repeated flat→stair→flat transfer and the same stair animation
at four world orientations, rather than replaying one globally fixed route.

Run the scripted west-ascent/platform/east-descent qualification with:

```bash
sbatch sonic/run_privileged_course_eval.sbatch
```

The viewer receipt is authoritative. It reports the Torch backend,
`takara_ball`, robot-local left-stick travel, independent right-stick facing,
no command-side root-translation odometry, target exclusion, and the exact
warped source. This remains a globally privileged one-way stair scene, not a
SONIC physics result. The main entry points are
`mm_sonic.privileged_terrain_matcher`, `mm_sonic.privileged_terrain_course`,
`mm_sonic.carpet_terrain_matcher`,
`mm_sonic.evaluate_privileged_online_matcher`, and
`mm_sonic.privileged_terrain_interactive_viewer`.

The `integration` extra installs the three pinned upstream runtime dependencies
(`scipy==1.15.3`, `PyYAML==6.0.3`, `cyclonedds==0.10.2`) alongside `mujoco` and
`pyzmq`. It intentionally does **not** declare a PyPI `unitree_sdk2py`: the
pinned Unitree Python SDK is resolved from
`external_dependencies/unitree_sdk2_python` inside the authenticated
`--gear-checkout`, so callers never supply a Unitree `PYTHONPATH` and the
checkout is never mutated.

## Stage A integration gate

The Stage A CLI is non-interactive. All external inputs and the isolated output
root are mandatory command-line paths; `SONIC_*` environment variables are not
used as input defaults. `--encoder` is optional only for policies whose
authenticated observation contract has no encoder.

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli preflight \
  --gear-checkout /read-only/GR00T-WholeBodyControl \
  --policy /read-only/model_decoder.onnx \
  --observation-config /read-only/observation_config.yaml \
  --encoder /read-only/model_encoder.onnx \
  --source-mjcf /read-only/g1_29dof.xml \
  --terrain-dir /read-only/g1_terrain \
  --output-root sonic/runs

PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-a \
  --mode known-good-stream \
  --gear-checkout /read-only/GR00T-WholeBodyControl \
  --policy /read-only/model_decoder.onnx \
  --observation-config /read-only/observation_config.yaml \
  --encoder /read-only/model_encoder.onnx \
  --source-mjcf /read-only/g1_29dof.xml \
  --terrain-dir /read-only/g1_terrain \
  --output-root sonic/runs
```

The three Stage A modes execute ordered prefixes of the registered seven-gate
contract. Only `known-good-stream` can create a complete Stage A pass: it binds
the file dynamic metric, stream delivery audit, and stream dynamic result to
the same policy, encoder, observation configuration, external commit, model,
generated flat scene, initial qpos, and canonical reference hashes. Later
stages must call `mm_sonic.metrics.validate_stage_a_prerequisite` on the
retained `stage-a-evidence.json`; a file-only result is intentionally rejected.
The validator authenticates the sealed inventory first, requires the evidence
registry hash to match the current canonical `stage_a.json`, and reconstructs
the exact raw `stage-a`/`known-good-stream` option set plus its sealed canonical
invocation cwd. Only argv-relative tokens are resolved against that cwd before
checking external paths, output-root placement, safe environment, ordered
passing gates, literal
per-gate hash claims, primary output digests, merged identity, metrics,
top-level outputs, and terminal `complete`/pass outcome. The stream-delivery
primary output is canonicalized again and its embedded delivery-audit digest is
recomputed after removing only the digest and publication-summary fields. A
stale registry or a self-consistent but forged evidence claim is not a
prerequisite.

Exit status is `0` for the requested completed gate prefix, `2` for a
configuration or integration failure, `3` for a scientific failure, and `4`
when a required external checkpoint, Git-LFS payload, dependency, execution
provider, or GPU is absent. Exit `4` writes immutable, result-free `not_run`
evidence whenever the explicit inputs make safe output creation possible. It
is never a feasibility result. Raw runs remain under ignored `sonic/runs`.

### Known-good timing and scoring contract

The pinned GEAR policy runs at wall-clock 50 Hz while the official MuJoCo
backend advances independently at 500 Hz. Stage A therefore never derives a
policy-row count or target/state alignment from simulator-step counts.

Both known-good modes cold-start GEAR only through authenticated
`WAIT_FOR_CONTROL`. File mode resets frame zero and arms playback there. Stream
mode enables ZMQ there and archives four independently labelled preload
transcripts: one `readiness` publication for frame 0, 22 `logical`
publications covering frames `1..440`, one `padding` publication for transport
indices `441..486`, and one `receipt_fence` publication at index 487. Padding
and the receipt fence repeat canonical pose 440 and are never scored. Every
publication requires exact, separately ranged pinned Start, processing, merge,
and `*** End of ZMQ decoding processing ***` lines. Arbitrary pinned decoder
diagnostics may occur between the merge and End lines and are SHA-256 bound.
Run-local CSV projection renders every binary32 value as an exact promoted
binary64 decimal, so GEAR's `std::stod` file path and the stream payload receive
identical numeric values rather than merely values that round back to binary32.
Publication N+1's authenticated Start-through-End completion event, rather
than its boundary snapshot alone, is the causal fence for publication N.

The `ZMQ STREAMING MODE: ENABLED` line alone is not preparation. Immediately
after its raw UTF-8 byte boundary, Stage A sends one `qe` PTY write and requires
the exact ordered acknowledgements `Delta heading left: 0.1 rad` and
`Delta heading right: 0 rad`. This fences the handler's reset tail and restores
net heading delta to zero before frame 0 can be published. GEAR stdout is
written and flushed to `dynamic/stream/gear.stdout` before those identical raw
bytes enter the stdout-only observer (stderr is archived separately), so every
recorded offset is an archive byte offset, including non-ASCII or split startup
output. Before CONTROL, the auditor
reopens that archive and writes
`dynamic/stream/preload-consumer-transcript.json`, binding the post-enable
fence and all 25 ordered sets of independent event-byte ranges to a frozen
stdout-prefix length and SHA-256. Every intervening region and the required
decoder tail after the final merge line are also range-hashed, while remaining
free to contain the pinned source's diagnostics. Later CONTROL output may
append to stdout without changing that authenticated prefix.

Before CONTROL, Stage A proves the target, `q.csv`, and `base_quat.csv` logs
contain zero data rows. It stops GEAR in WAIT, resets a distinct scored MuJoCo
epoch, and advances one pre-CONTROL step to publish fresh LowState. The reset
reuses the already authenticated same-scene simulator backend and DDS publisher
while rotating the physical log epoch; rebuilding the process-global Unitree
channel at this boundary is forbidden. It then
resumes once, activates CONTROL without stopping on the transition marker, and
advances physics in small increments until the target log is the exact
canonical 441-row sequence `0..440`. Overshoot, omission, duplication,
truncation, inode replacement, timeout, or a post-stop row fails integration.

Tracking uses the 441 same-CONTROL-tick GEAR `q.csv` and `base_quat.csv` rows,
after validating exact pinned headers, indices, shared timestamp prefixes,
strictly increasing monotonic time, joint permutation, finite values, and unit
base quaternions. MuJoCo state/contact logs remain physical cadence evidence,
not policy-aligned tracking rows. Evidence reports the one-step pre-CONTROL
prime, CONTROL-active drive steps/duration, and total scored-epoch
steps/duration separately; retained state/contact row counts cover the full
scored log epoch, including the prime.

## Stage B motion-matching qualification

Stage B replaces the known-good pose input with the registered flat
motion-matching route. It accepts only an immutable passing
`known-good-stream` Stage A evidence file whose inventory, invocation,
identities, gate outputs, and registry are revalidated against the current
code. It then executes the same four prerequisite gates, constructs the
601-frame motion-matching reference, transports frames `0..600` through the
pinned GEAR ZMQ decoder, and scores the official G1 SONIC dynamics.

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-b \
  --stage-a-evidence /read-only/stage-a-evidence.json \
  --gear-checkout /read-only/GR00T-WholeBodyControl \
  --policy /read-only/model_decoder.onnx \
  --observation-config /read-only/observation_config.yaml \
  --encoder /read-only/model_encoder.onnx \
  --source-mjcf /read-only/g1_29dof.xml \
  --terrain-dir /read-only/g1_terrain \
  --output-root sonic/runs/stage-b
```

The registered contract in `configs/experiments/stage_b.json` is exact: 30
accepted 20-frame commands, 601 authoritative target rows, and 12.000 seconds
of CONTROL-active physics. At the official 0.005-second simulator step this is
2,400 CONTROL steps. A one-step scored prime makes the retained epoch 2,401
steps, with exactly 2,401 contact rows and 600 state rows at 50 Hz. Complete but
truncated JSONL is rejected by validating every step number and sampling time,
not merely by counting syntactically valid records.

GEAR remains a wall-clock 50 Hz controller. For Stage B only, the simulator
adapter omits its redundant per-step wall-clock sleep and advances physics in
row-paced batches. The registered 16-row control lead ensures all 2,400
CONTROL steps finish before frame 600 is emitted. As soon as a newline-complete
601st target row exists, the process group is stopped before the full target
audit; early terminal arrival, overshoot, content mismatch, inode replacement,
or any post-stop write fails integration. The terminal fence records simulator
counters immediately before the stop, requires an identical post-stop and
final snapshot, and proves that requested, CONTROL-active, and required steps
were already equal before GEAR stopped. No simulator step may be used to fill
the duration after that boundary. Stage A retains its default paced simulator
behavior.

The generated run-local scene is bound semantically, not by a path-sensitive
digest alone. Stage B verifies the complete generated and included XML against
the authenticated Stage A scene after normalizing only the run-local include
path, and binds the registered allowed feet, forbidden body groups, and terrain
geom IDs. A pass additionally requires the exact command/frame/duration/log
coverage checks, no forbidden contacts, pelvis height and uprightness limits,
and joint/pelvis tracking ratios relative to the authenticated known-good
baseline. Secondary drift, clearance, scuff, impulse, and timing measurements
remain diagnostics and cannot override those registered primary gates.

## Opt-in in-process Torch motion matcher

For a clone-to-launch guide, see
[`TORCH_SONIC_QUICKSTART.md`](../TORCH_SONIC_QUICKSTART.md).

The flat interactive demo can load a native 50 Hz Takara motion folder and run
the matcher directly on CUDA, without starting `mm_chunk_server`:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.manual_demo \
  --mode interactive --input-source x11 --onscreen \
  --motion-backend torch \
  --motions-dir /home/ubuntu/Downloads/takara_walk_50hz.npz_v0 \
  --torch-device cuda \
  --scene-id sonic-flat-baseline --route-id flat-12s --terrain-weight 0 \
  --chunks 3000 \
  --gear-checkout /home/ubuntu/projects/gear-sonic-worktrees/simulation-lowstate-wait-v6 \
  --source-run /home/ubuntu/mm-flat-walk-stage-b-r16-hold-resume.tym9Fc/stage-b/stage-b-20260718T005901613783Z-8b63d477 \
  --runtime /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root /home/ubuntu/mm-sonic-torch-live
```

The matcher publishes overlapping 46-row windows. Physics remains fenced until
GEAR acknowledges the final row, after which exactly one 20 ms policy interval
is released. W/A/S/D move, Space stops, Backspace restarts, and X exits.

## Audited canonical terrain corpus

The phase-1 corpus tool is available as
`python -m mm_sonic.terrain_oracle.corpus_cli` (or the lazy
`mm_sonic.terrain_oracle.corpus_main` entry point). Its commands are
`inventory`, `import`, `audit`, `render-audit`, `coverage`, and `freeze`.
The normal publication sequence is:

```bash
PYTHONPATH=sonic/python:. \
  /move/u/justingu/miniconda3/envs/isaac6_test/bin/python -B \
  -m mm_sonic.terrain_oracle.corpus_cli import \
  --fixture CANONICAL_FIXTURE --output RAW_CORPUS

PYTHONPATH=sonic/python:. \
  /move/u/justingu/miniconda3/envs/isaac6_test/bin/python -B \
  -m mm_sonic.terrain_oracle.corpus_cli audit \
  --corpus RAW_CORPUS --model G1_MODEL.xml --output AUDITED_CORPUS

PYTHONPATH=sonic/python:. \
  /move/u/justingu/miniconda3/envs/isaac6_test/bin/python -B \
  -m mm_sonic.terrain_oracle.corpus_cli render-audit \
  --corpus AUDITED_CORPUS \
  --output AUDITED_CORPUS/render-audit

PYTHONPATH=sonic/python:. \
  /move/u/justingu/miniconda3/envs/isaac6_test/bin/python -B \
  -m mm_sonic.terrain_oracle.corpus_cli coverage \
  --corpus AUDITED_CORPUS --output AUDITED_CORPUS/coverage.json

PYTHONPATH=sonic/python:. \
  /move/u/justingu/miniconda3/envs/isaac6_test/bin/python -B \
  -m mm_sonic.terrain_oracle.corpus_cli freeze \
  --corpus AUDITED_CORPUS --output FROZEN_CORPUS
```

`audit` republishes canonical clip and mesh bytes with exactly one mechanical
report per clip. `render-audit` invokes the fixed package-owned MuJoCo renderer
with `shell=False`; production evidence cannot be supplied by an external
renderer. It records one H.264 video/contact-overlay PNG pair for every accepted
`[start,end)` interval, one contact sheet covering the complete interval set,
and one full video for every recomputed
`(source, action_class, terrain, direction)` stratum. Receipts bind clip,
source, model, terrain, mesh, normalized invocation, executable, and output
hashes. `coverage` is rebuilt from the public phase-1 coverage authority.
`freeze` accepts no missing, extra, stale, malformed, or hash-consistent but
mismatched audit/render/coverage evidence. The frozen release carries an
authenticated compiled G1 MJB and reloads it to prove the same structural model
hash, so the release does not depend on the XML's external `meshdir`.

The exact real inventory command reported 173 flat clips, 18 Justin clips, and
489 clean GRAIL pairs: 166 `c490_stair_p1`, 174 `c490_stair_p2`, 77
`c490_slope`, and 72 `c490_curb`. The sealed inventory file SHA-256 is
`a7d6157116eec567b5a4df2daf21c95f4a6363b8dc8c6dfe1a8a5c6652c6063c`;
its recomputed content hash is
`901dd172146444f18e77d8c427c910bfef0f0f7935fab543a7e245c7713a21eb`.
The selected LAFAN G1 release contains 40 strict numeric CSVs at immutable
revision `ce1572906efe6157840e8474d5a0d7aa87481e74`; every downloaded CSV,
README, and LICENSE metadata sidecar records that commit. The README assigns
the LAFAN1 motion data to `CC-BY-NC-ND-4.0`, while the separate repository/code
LICENSE is `BSD-3-Clause`. The LAFAN inventory file SHA-256 is
`1b1f8097c532a08c26e39e7f504f7bb2f4e887636129c59f95930ea0ef41d844`;
its recomputed content hash is
`98493a9a0443ed2b9151a1de4b5084ebe3283657f4f055333a92ba724588820c`.
The two file hashes are code-anchored phase-1 trust roots: real import rejects
even a self-consistent replacement inventory before any GRAIL pickle can be
deserialized. Flat, Justin, and LAFAN adapters parse private no-follow source
snapshots while retaining the original authority paths; GRAIL records are
reconstructed from the anchored inventory and their exact PKL, USD,
`clips.json`, frame-count, and terrain-pose authorities are rechecked before
deserialization.

Directory publications use canonical relative paths, fsync-backed staging,
exclusive no-replace destination claims, and authenticated logical-completion
markers. Linux `renameat2` is used when the filesystem supports it; the NFS
fallback is intentionally described as logically complete, not kernel-atomic.
Existing destinations are never overwritten, and readers reject incomplete,
changed, or still-writer-owned trees.
Contract failures—including malformed arguments—print an actionable error and
return status 2 without a Python traceback; `--help` returns status 0.

The Torch demo defaults to `--torch-trajectory-model takara_ball` and
`--torch-max-source-joint-step-rad 0.35`, matching offline corpus generation;
pass `--torch-trajectory-model legacy` only for an explicit A/B.
Its periodic terminal diagnostic prints raw stick velocity, filtered ball
velocity, root-to-ball separation, and clamp displacement for live controller
inspection.

On the `slam` workstation, the same configuration is packaged as:

```bash
sonic/launch_takara_ball_interactive.sh
```

Run it from the graphical desktop/session where `DISPLAY` and the controller
mapping are available.  It deliberately refuses a headless shell.  The default
responsive horizon is five source intervals (0.2 seconds); set
`TAKARA_RESPONSIVE_SOURCE_INTERVALS=10` for the more conservative 0.4-second
variant.  All external paths can be overridden with the `TAKARA_*` variables
declared at the top of the launcher.

## Hybrid flat↔Justin-stair kinematic motion-matching viewer

The first terrain viewer isolates motion-matching quality from SONIC tracking
and dynamics. It uses the full 173-clip Takara/BONES bank for unrestricted
flat locomotion, hands off to the fixed-world Justin archive before stair
contact, and returns to flat motion matching at a double-support landing. It
writes those clean kinematics directly into the articulated G1 MuJoCo model
and calls forward kinematics only:

```bash
sonic/launch_terrain_mm_kinematic_interactive.sh
```

The default frontend is the same tokenized browser controller used by the
SONIC interactive evaluation. The launcher prints an SSH tunnel command and a
`localhost` URL. Open that URL locally, select the Switch controller in the
Gamepad panel, verify or change its four axis indices, and press **Start
simulation**. Both sticks remain native analog inputs; the browser exposes raw
axes and deadzone control. Its reset, center, and end buttons control this
kinematic session.

An X11 keyboard fallback remains available with
`TERRAIN_MM_FRONTEND=x11`. In that mode `W/A/S/D` requests robot-local travel,
`Ctrl` plus the arrow keys provides an independent facing stick, arrows without
`Ctrl` orbit the camera, `Shift` selects slower walking, `Space` stops at
double support, `Backspace` restarts, and `X` exits.

The cyan path is the critically damped command-ball intention re-centred at
the current robot, orange is the selected clean motion's future, and red is
the realized root trail. Flat matching uses 96,012 searchable frames. A stair
handoff is considered only inside a pre-contact approach window and is gated
by fixed-world root pose, commanded path/facing, source pose/velocity, and
planted-foot geometry. Once committed, stair search runs every five 50 Hz
frames with the 10 mm planted-foot execution gate. The return to flat waits
for double support after the recorded exit window and inertializes into a
new flat match at the landing's world pose and height.

This viewer is explicitly not a tracker or stability test. It uses orientation
to express robot-local commands but stores no global root position or
translation odometry as a policy or joystick input. World position is used
inside this kinematic test only to align the clean motion to the fixed visible
staircase. Override the flat bank or Torch device with
`TERRAIN_MM_FLAT_MOTIONS` and `TERRAIN_MM_FLAT_DEVICE`; CPU is the default and
runs comfortably faster than the 50 Hz viewer deadline.

The same real catalog can be checked without a display:

```bash
PYTHONPATH=sonic/python \
  /move/u/justingu/miniconda3/envs/isaac6_test/bin/python \
  -m mm_sonic.terrain_interactive_viewer --headless-frames 300
```

## Optional BONES walking support bank

The single TakaraWalk source does not physically cover every lateral,
backward, turning, and start/stop request. A curated BONES bank can be
exported from the clean, retargeted 50 Hz G1 kinematics without using the
short/reset-heavy BONES rollout episodes as command labels:

```bash
cd /move/u/bodow/Projects/Motion-Matching-takara-corpus
PYTHONPATH=sonic/python \
  /move/u/justingu/miniconda3/envs/env_isaaclab/bin/python \
  -m mm_sonic.export_bones_motion_bank \
  --source-zarr /move/data/bones/g1/zarr/locomotion_50hz.zarr \
  --include-legacy-takara-npz \
    /move/u/justingu/rmr_tracking/motions/isaac6/takara_walk_50hz.npz \
  --output-dir \
    /move/data/terrain-aware/motion-matching/takara_bones_walk_support_v2_startstop
```

The default export contains 86 BONES base clips and their 86 explicit mirrors,
plus the clean TakaraWalk clip when `--include-legacy-takara-npz` is supplied.
Seventy-seven of those base clips (154 after mirroring) contain start or stop
transitions spanning forward, backward, lateral, diagonal, and turn-then-walk
motion; the remainder supplies loops, arcs, and in-place turns. It writes the
same native
`*/motion.npz` layout consumed by `--motions-dir`, plus a provenance
manifest. The exporter is read-only with respect to BONES and refuses to
overwrite an existing output directory.

Use BONES as a *kinematic support bank*, keep the generated joystick and
persistent-ball streams as the training commands, and run a clean physical
tracker gate before any noisy collection. The existing BONES rollout zarr is
not a substitute: those episodes are only about one second long and their
command channels were reconstructed from realized motion rather than retained
from a human joystick.

## Takara two-stick flat-data corpus

The offline corpus path turns deterministic, human-like two-stick traces into
trackable Takara references and then uses Justin's SONIC data collector to add
the recovery noise used by TML-BeyondMimic. The left stick requests planar
travel in the robot's live root-yaw frame: pushing forward remains robot
forward even after a 180-degree turn. The right stick requests facing in the
controller frame; a centered right stick retains the last facing request.
This conversion needs orientation but never root position or translation
odometry. The 60 base traces include starts/stops, speed
changes, forward/backward reversals, lateral toggles, smooth arcs, circles,
slaloms, zigzags, independent travel/facing, in-place spins, and seeded
human-like random stick movements.  Every base trace is followed by an exact
sagittal mirror, producing 120 balanced 12-second clips at 50 Hz.

The zarr retains both the operator request and the generated result:

- raw left/right stick and button channels;
- robot-local requested velocity, its matcher-world realization, facing, and
  the raw integrated path;
- the persistent critically damped Takara simulation-ball position, velocity,
  facing, and exact yaw rate used by motion matching;
- selected source frames, search costs, transition flags, root-adjustment
  magnitudes, and safety-clamp activity;
- clean joint and full-body kinematics;
- intended/realized Task-4 diagnostics and the direct H24 Task12 command:
  filtered-ball `[local_vx, local_vy, yaw_rate]` at +6/+12/+18/+24 frames.

The ball is controller state, not odometry. At reset it starts at the robot,
then its velocity and facing are critically damped toward the two live stick
requests using Takara's C++ constants (`0.27 s` half-lives). Search sees the
ball's predicted positions/facing at H15/H30/H45. The character root is gently
adjusted toward the ball (velocity-bounded `0.10 s` position and `0.20 s`
rotation half-lives) and is finally clamped to `0.15 m` / `pi/2`. This uses no
world localization, map, or remembered robot root position at policy
inference. Task12 is position-free: the policy receives four future local
velocity/yaw-rate knots from the controller filter. The motion-matching
generator uses only its current generated root yaw to rotate the operator's
local left-stick velocity internally.

The source safety filter rejects any candidate whose published 46-frame
window contains either a joint change larger than `0.35 rad` in one 20 ms
step or a stored joint speed above the equivalent `17.5 rad/s`. Checking both
also catches a dangerous candidate first frame whose incoming pose step lies
just outside its window. This matters because Takara's flat source contains a
few very fast arm segments that are invisible to the pelvis/foot search
features and are unsuitable for the physical tracker.

The legacy Takara NPZ has a misleading `body_quat_w` field name: its stored
values are XYZW.  `offline_corpus` converts this explicitly to the canonical
WXYZ convention at ingestion.  All output zarrs and Justin pipeline manifests
are WXYZ.

Run the stages in order:

```bash
# 1. Generate the CPU-only kinematic smoke and prepare its tracker bundle.
sbatch --partition=move sonic/run_offline_corpus_smoke.sbatch

# 2. Require clean physical tracking and inspect paths/gait before scaling.
sbatch --partition=move sonic/run_clean_tracker_gate.sbatch
sbatch --partition=move sonic/run_clean_tracker_review.sbatch

# 3. Generate all 60 base traces plus exact mirrors.
sbatch --partition=move sonic/run_full_corpus.sbatch

# 4. Collect four disjoint, mirror-paired noisy shards in parallel.
sbatch --partition=move --array=0-3%4 \
  sonic/run_noisy_collection_shard.sbatch

# 5. Join the retained command stream to each physical rollout.
sbatch --partition=move --array=0-3%4 \
  sonic/run_annotate_noisy_shard.sbatch
```

Stages 1 and 3 request no GPU. Only physical SONIC tracking/rendering and noisy
rollout collection consume GPU lanes.

For robustness to deliberately bad joystick timing, the smooth corpus has a
separate additive abrupt-command bank.  It contains 102 canonical traces and
their exact sagittal mirrors, with one-frame travel reversals, heading
reversals, simultaneous two-stick changes, stop/restart, and speed jumps at
varied gait phases.  `data/command_event_mask` labels the exact hard frames so
training can oversample their surrounding windows without overweighting every
held frame of each clip.  Keep this as an explicitly weighted supplement to
the smooth omnidirectional bank rather than replacing normal steering data.

```bash
# Materialize the 204-clip clean abrupt bank and all 12 tracker inputs.
sbatch sonic/run_abrupt_corpus.sbatch

# Qualify one shard, then collect the complete bank with two concurrent L40s.
sbatch sonic/run_noisy_collection_abrupt_canary.sbatch
sbatch sonic/run_noisy_collection_abrupt.sbatch

# CPU-only Task4/Task12 and event-mask sidecars after collection finishes.
sbatch sonic/run_annotate_abrupt_shards.sbatch
```

The supplied checkpoint bundle must pair Justin's Takara weights with the
configuration used by his successful evaluations:

```text
weights: /move/u/justingu/Projects/grail-stairs/sonic_port/takara/ft_runs/poc/last.pt
config:  /move/u/justingu/Projects/grail-stairs/sonic_port/takara/ft_runs/poc/config.yaml
```

The physical tracker receives no global root position or odometry command.
Task 4 remains a diagnostic in the current physical pelvis-yaw frame. The
training Task12 is copied directly from the retained filtered-ball intention,
not reconstructed from noisy realized motion. The raw joystick is retained
separately. Sharding keeps every base/mirror pair together, and all collectors
use the real flat rigid slab from Justin's Takara assets.

For interactive inspection after GPUs are available, the existing C++/SONIC
interactive path already owns the same persistent simulation ball and supports
keyboard/gamepad input. The offline Torch port intentionally uses the same
spring and adjustment constants. Its motion-matching mode uses robot-local
left-stick velocity and caps run speed at `0.72/0.55/0.55 m/s`
(forward/lateral/back), matching the corpus rather than the older
`0.9/0.6/0.6 m/s` viewer defaults. Interactive joystick behavior is therefore
a direct sanity check of corpus-generation semantics rather than a separate
controller.
