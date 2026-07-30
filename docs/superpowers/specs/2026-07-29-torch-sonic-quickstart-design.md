# Torch Motion Matching + SONIC Quickstart Design

## Goal

Add one self-contained quickstart for running the in-process PyTorch motion
matcher with the released SONIC tracker. The guide must let a researcher repeat
the verified flat-locomotion demo without reading the terrain pipeline or the
C++ motion-matching integration.

## Scope

The guide covers:

- the native 50 Hz Takara motion-folder input;
- the isolated Torch runtime;
- the qualified GEAR/SONIC tracker, policy, encoder, and observation assets;
- the exact verified cluster launch;
- a portable launch template using path placeholders;
- controls, expected readiness messages, restart, and shutdown;
- focused verification commands; and
- concise troubleshooting for identity, X11 focus, dependency, and process
  cleanup failures.

The guide does not cover terrain-aware motion matching, depth observations,
training, the C++ matcher server, obstacle avoidance, or scientific
qualification.

## Location and Structure

Create `TORCH_SONIC_QUICKSTART.md` at the repository root so it is visible
immediately after cloning. Keep the existing `sonic/README.md` launch section
as the detailed subsystem reference and link to it from the quickstart.

The quickstart order is:

1. architecture and non-requirements;
2. prerequisites and required assets;
3. checkout and Torch environment;
4. focused verification;
5. exact cluster launch;
6. portable launch template;
7. controls and expected startup;
8. restart/shutdown;
9. troubleshooting.

## Correctness Requirements

- Pin the implementation branch `research/g1-low-latency-driver`.
- Include the verified commit at the time the guide is published.
- Use `simulation-lowstate-wait-v6`, whose authenticated GEAR commit matches
  the simulator contract.
- Use `sonic/.torch-mm-venv/bin/python`.
- Require `--motion-backend torch`, the motion folder, CUDA selection, the flat
  scene, and terrain weight zero.
- State that no `mm_chunk_server` process is required.
- Preserve native IsaacLab joint order in streamed references; mention the
  MuJoCo permutation only as an internal initial-state concern.
- Describe the 46-row acknowledged publication and one-interval physics gate
  at a high level.

## Verification

Before publishing, run the focused Torch, SONIC adapter, manual-demo, gear
action, and simulation-gate test set. Check every documented absolute cluster
path exists, validate the launch parser, scan for placeholders accidentally
left in the exact cluster command, and run `git diff --check`.

