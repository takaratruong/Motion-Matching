# G1 Torch Stair Live Kinematic Viewer Design

## Goal

Provide a native MuJoCo window in which an operator can drive the small
five-clip, 50 Hz Torch terrain matcher over the authenticated stair heightfield
with live keyboard input. This experiment evaluates motion-matched kinematics
only.

## Scientific Boundary

- Run the dense privileged-height matcher from the existing
  `torch-stair-small` experiment.
- Do not launch SONIC, GEAR, a policy, a simulator coordinator, or any transport.
- Do not call `mj_step`, integrate forces, or use MuJoCo state estimation.
- MuJoCo is only the renderer and forward-kinematics engine: each accepted
  matcher result is copied into `data.qpos`, followed by `mj_forward`.
- Use the existing authenticated G1 joint permutation, dataset manifest, query
  height grid, and matcher-to-scene alignment.

## Operator Contract

- `W` requests forward motion at the experiment's derived reference speed.
- `S` requests backward motion at the same speed.
- `A` and `D` request lateral motion.
- Simultaneous orthogonal keys are normalized so diagonals do not exceed the
  reference speed.
- `Space` requests zero velocity.
- `Backspace` resets the matcher and rendered state.
- `X` exits cleanly.
- Commands are sampled every displayed 20 ms tick. The viewer may render more
  slowly under load, but it must never invent intermediate matcher states.

## Rendering

Build one temporary MuJoCo scene before opening the viewer. The scene includes:

- the pinned `g1_29dof.xml` model;
- a static mesh generated from the authenticated query height grid;
- the matched robot root and 29 joint positions; and
- a small visual marker at each of the 91 dense terrain sample locations.

The render loop writes the latest matched root pose and source-ordered joint
state to `qpos`, calls `mj_forward`, updates the marker positions, and
synchronizes the passive viewer. An overlay reports the command, selected
clip/frame, motion and terrain costs, matcher-step latency, and the explicit
`KINEMATIC ONLY / NO PHYSICS / NO SONIC` status.

## Structure

- Add `mm_sonic.torch_terrain_live_viewer` as a separate executable module.
  Keep the authenticated saved-rollout viewer unchanged.
- Put key-to-command and matcher-state-to-MuJoCo conversion in small,
  deterministic functions that can be tested without opening a window.
- Reuse `resolve_stair_config`, `TerrainFeatureExtension`,
  `TorchMotionMatcher`, `X11KeyStateProvider`, and the pinned joint
  permutation. Do not create a second matcher implementation.
- Generate the temporary XML/mesh under an owned temporary directory and remove
  it on normal or exceptional exit.

## Failure Handling

Reject missing or unauthenticated dataset/config/model inputs before opening a
window. Reject any G1 model whose `nq` is not 36. Treat non-finite matcher
states, wrong joint shapes, or a failed viewer close as fatal integration
errors. Always release X11 key grabs and close the viewer in `finally`.

## Verification

- Unit-test keyboard combinations, diagonal normalization, stop/reset/exit
  edges, joint permutation, and root quaternion placement.
- Test the scene construction and one kinematic application with MuJoCo,
  asserting that `mj_forward` updates body positions without `mj_step`.
- Run a short headless CUDA matcher smoke against the five-clip stair dataset.
- Launch on `DISPLAY=:1`, confirm a MuJoCo window exists and the process remains
  live for operator control.

