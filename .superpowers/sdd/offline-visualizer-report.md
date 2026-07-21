# Offline G1 Overlap Visualizer Report

Status: DONE

Base SHA: `2683f6a6eb1855695607b13fa082c416a94d5c46`

## Contract

`G1OVLP01` schema version 1 is strict little-endian. Its header records version 1, 25 Hz, 80 frames, 31 bones, two foot contacts, the canonical G1 skeleton signature, and the exact canonical bone-name order. The payload is, in order: positions `[80,31,3]`, velocities `[80,31,3]`, angular velocities `[80,31,3]`, WXYZ rotations `[80,31,4]`, and two binary foot contacts `[80,2]`.

The native loader rejects wrong magic/version/rate/sizes/signature/order, truncated payloads, nonfinite floats, non-unit quaternions, non-binary contacts, and trailing bytes. `Player::advance_25hz()` advances exactly one 25 Hz frame; it supports looping and restart.

`MM_G1_OFFLINE_OVERLAP=/path/to/pose.bin` loads a looping visual-only player. The controller runs all normal interaction/state/attachment/grasp validation first, then replaces only the final pose used to write local arrays, FK, camera, and mesh rendering. It does not feed the offline pose into interaction state or pickup physics.

## Export commands

The current dataset has no canonical-local-offset array, so reconstruct its ground-truth row with the interaction pack that produced `build/g1-overlap/dataset.npz`:

```bash
INTERACTION_PACK=/absolute/path/to/the/matching/g1_interaction_pack
test -f "$INTERACTION_PACK/interaction_database.bin"
python tools/export_g1_overlap_pose.py \
  --dataset build/g1-overlap/dataset.npz \
  --interaction-pack "$INTERACTION_PACK" \
  --split train --row 0 \
  --align-root-position 0 0 0 \
  --align-root-yaw-degrees 0 \
  --output build/g1-overlap/offline-overlap-train-0.bin
```

If the parallel dataset build supplies `canonical_local_positions` (or `local_positions`), omit `--interaction-pack`:

```bash
python tools/export_g1_overlap_pose.py \
  --dataset build/g1-overlap/dataset.npz \
  --split train --row 0 \
  --align-root-position 0 0 0 \
  --align-root-yaw-degrees 0 \
  --output build/g1-overlap/offline-overlap-train-0.bin
```

For a later generated `[80,195]` NPZ, provide the required frozen offsets separately because channels 2–30 offsets are intentionally absent from the 195-channel format:

```bash
python tools/export_g1_overlap_pose.py \
  --generated build/g1-overlap/generated.npz --generated-key generated \
  --local-positions build/g1-overlap/local-positions.npz --local-positions-key local_positions \
  --align-root-position 0 0 0 \
  --align-root-yaw-degrees 0 \
  --output build/g1-overlap/offline-overlap-generated.bin
```

The declared alignment moves frame-0 Simulation root position to `--align-root-position`, rotates the Simulation root position/derivatives by `--align-root-yaw-degrees`, and pre-multiplies its WXYZ root rotation. Child locals remain local, so playback is already in the file's declared world frame and the controller applies no hidden teleport/alignment.

## Verification

```text
make build/tests/test_interaction_offline_overlap && build/tests/test_interaction_offline_overlap  PASS
python -m unittest tests.python.test_export_g1_overlap_pose -v                         PASS (4 tests)
make build/tests/test_interaction_native_g1_bridge && build/tests/test_interaction_native_g1_bridge  PASS
make controller EXT=.offline-overlap-final                                              PASS; not launched
```

A headless generated-NPZ smoke export was loaded by the C++ reader and reported `80 1.25 0.9 -2.5`, verifying the requested frame-0 world alignment. No controller, GUI, X11 capture, pickup physics, attachment, or interaction runtime test path was launched.
