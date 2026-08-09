# Native G1 challenging-terrain collection

This branch replaces the rejected narrow-stair spatial warps as the source of
new, exotic terrain motion.  It trains a dynamically valid G1 controller on
broad procedural terrain and records the resulting simulated states as clean
kinematics for later SONIC tracking and diffusion-policy training.

## Runtime interface

The policy receives only quantities available on the robot:

- robot-local joystick command `(forward velocity, lateral velocity, yaw rate)`;
- base linear and angular velocity;
- projected gravity;
- joint position and velocity;
- previous action;
- a robot-centred 17x11 local height scan.

It does **not** receive global root position, global yaw, the terrain class, the
terrain seed, or a pre-authored route.  Commands are therefore centred on the
current robot at every policy step and remain body-relative after the robot
turns.

## Terrain distribution

`mm_sonic.g1_hct_omni_env` generates 8x8 m tiles in these families:

| Family | Share | Difficulty |
|---|---:|---|
| flat / lightly rough | 11.25% | 0-5 mm |
| random rough | 11.25% | +/-25 mm |
| smooth slopes, up/down | 11.25% | 2-20% grade |
| rough slopes, up/down | 11.25% | 2-20% grade plus +/-25 mm |
| scattered obstacles | 22.5% | 5-50 mm, 0.2-0.8 m wide |
| correlated rolling hills | 22.5% | 50-300 mm amplitude |
| stairs, up/down | 10% | 50-180 mm risers, 0.3 m treads |

The stock IsaacLab difficulty curriculum is disabled.  It promotes examples by
net displacement from the spawn point, which incorrectly demotes valid spins,
stops, reversals, arcs, and out-and-back paths.  All ten difficulty rows are
instead populated uniformly throughout the continuation run.

## Command distribution

Commands are resampled every 1.5-4 seconds.  The continuous ranges are:

- forward/backward: `[-0.8, 1.0] m/s`;
- lateral: `[-0.5, 0.5] m/s`;
- yaw rate: `[-1.0, 1.0] rad/s`.

Each component is independently set exactly to zero with probability 0.5.  This
produces real coverage of straight walking, backward walking, pure strafe, pure
spin, translation-plus-yaw, and complete stop/start transitions rather than
hoping that a continuous sampler lands near those cases.  Resampling is abrupt
on purpose: the collected controller should not fall when a human changes a
stick quickly.

A soft arm-deviation cost of `-0.30` suppresses a repeated high-energy hand
swing while preserving enough arm motion for balance.

## Training

The initial policy is Justin's upright G1 rough-terrain checkpoint:

`/move/u/justingu/Projects/paper_repro_night/baseline/logs/rsl_rl/g1_rough/2026-07-16_16-37-15_gait_Aexp/model_2999.pt`

The checkpoint uses RSL-RL's `log_std` parameterization.  The runner declares
that explicitly and loads strictly; it never performs a permissive partial
load.

Phase one broadened the command and terrain family distributions.  Phase two
continues from phase one on a fresh terrain seed with uniform difficulty.  Both
use 4096 environments, one GPU, 4 CPUs, and 24 GB host RAM.  The scripts are:

- `sonic/run_train_g1_hct_omni.sbatch`
- `sonic/python/mm_sonic/train_g1_hct_omni.py`
- `sonic/python/mm_sonic/g1_hct_omni_env.py`

## Evaluation and visual review

`evaluate_g1_hct_omni.py` assigns ten deterministic 12-second joystick programs
across 200 environments at difficulty row 9/9, using terrain seeds not used by
either training phase.  Every terrain column receives every program:

1. steady forward;
2. steady backward;
3. left strafe;
4. right strafe;
5. left diagonal arc;
6. right diagonal arc;
7. spin and spin reversal;
8. forward, centred stick, backward, centred stick;
9. abrupt forward/left/back/right/diagonal sequence;
10. alternating diagonal/yaw zigzag.

The archive records root pose, all joint states, actual body-relative velocity,
commands, actions, both ankle trajectories and contacts, episode/reset state,
and the moving height scan.  A reset, episode-counter drop, or 0.5 m one-step
teleport invalidates an environment before any tracking metric is interpreted.

`evaluate_g1_hct_omni.py` also exports the exact triangle mesh authored by the
Isaac terrain generator and each environment's world origin.  The mesh is
shared across environments rather than duplicated in every clip.

`render_g1_hct_rollout.py` replays the recorded root and joint state on the
articulated G1 mesh in MuJoCo and crops the exact Isaac terrain around that
route.  Old archives without the mesh retain the height-scan fallback.  The
renderer deliberately does not report cross-asset sole clearance: MuJoCo and
Isaac use different collision envelopes, so that number was systematically
biased even on ordinary flat contact.

No motion enters the accepted corpus until dense videos and metrics reject:

- falls, resets, teleports, or freezes;
- visible foot penetration or hovering;
- stance gliding and foot crossing;
- abrupt pose jolts;
- failure to follow translation or yaw commands;
- violent or repetitive arm motion.

The previous extreme-angle narrow-stair and side-on warp videos are explicitly
not accepted examples under this bar.

## Result and model selection

The current balanced source checkpoint is:

`/move/data/terrain-aware/g1-hct-omni/train/logs/rsl_rl/g1_hct_omni/2026-08-08_21-58-52_omni_hct_uniform_v2/model_4997.pt`

Its exact-mesh, unseen-seed-44 evaluation is:

`/move/data/terrain-aware/g1-hct-omni/eval/uniform_v2_4997_seed44_exact_full.npz`

It completed all 200 difficulty-9/9, 12-second cases without a reset or
teleport.  Aggregate mixed command RMSE was 0.450, the maximum physical XY step
was 34.3 mm, maximum joint step was 0.593 rad, and arm deviation RMS was 0.0149
rad.  All 20 stair cases survived, although stairs remain the weakest guidance
family (0.670 ascent and 0.753 descent mixed RMSE).  Right strafe/right arc also
remain less accurate than their left counterparts.

Phase one tracked commands better (0.424 mixed RMSE) but failed two of the same
200 exact-terrain cases and had a larger 0.723 rad joint step.  It was therefore
not selected as the collection source.

The planted-reward refinement (`omni_hct_planted_v3`) increased command and
anti-slide weights, reduced the air-time incentive, and evaluated preserved
checkpoints 5050, 5100, 5150, 5200, and 5496.  It was rejected: the final model
fell to 199/200, command RMSE worsened to 0.529, stance motion and arm motion
increased, and no intermediate checkpoint dominated `model_4997.pt`.  These
files are retained as negative experimental evidence, not promoted models.

Dense exact-terrain review videos are in:

`/move/data/terrain-aware/g1-hct-omni/video/uniform_v2_4997_seed44_exact_review`

The exact geometry removes the false smeared-riser penetration seen in the old
height-scan renders.  Diagonal stair contacts are continuous and land on
treads.  Pure side-on descent is still visibly leaned-back and kicky, so the
corpus is a robust dynamically generated terrain bank, not a claim that every
command/terrain pairing is MotionBricks-quality animation.

## Packaged 11-second corpus

`package_g1_hct_clips.py` drops the one-second neutral lead-in and writes one
clip per clean environment.  The verified collection is:

`/move/data/terrain-aware/g1-hct-omni/collection/uniform_v2_4997_seed44_200x11s`

It contains exactly 200 NPZ clips, 550 frames / 11 seconds each at 50 Hz, plus
one shared exact `terrain.npz` and `manifest.json`.  Total size is 246 MB.  Each
clip preserves root and joint state, local velocities, actions, robot-local
joystick commands, ankle trajectories and contacts, local height scans,
terrain/program labels, source environment, and the exact terrain origin.  The
program distribution is exactly 20 clips each for forward, backward, both
strafes, both diagonal arcs, spin/reversal, forward-stop-backward, abrupt omni,
and zigzag/yaw.
