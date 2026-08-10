# G1 LMM Preliminary Slope Visual V2 Design

Date: 2026-08-09

Status: approved for implementation

## Objective

Produce one small learned exact-route terrain visualization in MuJoCo. The G1
must advance through the authenticated authored GRAIL slope using the learned
decompressor and recurrent stepper when `W` is held, and pause bitwise when
`W` is released.

This is a deadline-oriented visual experiment. It is not an accepted terrain
model, a terrain-generalization result, or evidence of general command
coverage.

## Why V2 Exists

The immutable V1 fit stopped after 100,000 decompressor steps because its
worst non-root local bone-translation error was 1.492458 mm against the
production 1.000 mm limit. That metric is a decoder/skeleton-stretch guard,
not terrain registration accuracy. Every other decompressor gate passed:
maximum joint error was 0.007690 rad, whole-body FK error was 5.068 mm, sole
error was 4.948 mm, and both contact F1 scores were 1.0.

For this visual-only experiment, the local bone-translation maximum is allowed
to be at most 2.000 mm. The production terrain canary retains its 1.000 mm
limit unchanged. This exception is explicitly post-hoc and may not be used to
claim numerical acceptance.

## Frozen Experiment

V2 retains V1's exact authenticated 595-row slope source, data assembly,
network dimensions, Orange Duck objective, deterministic CUDA configuration,
GPU3 identity, optimizer, learning rate, batch size, seeds, and budgets:

- 1,000-step 64-row capacity preflight;
- 100,000 decompressor steps;
- 100,000 recurrent-stepper steps;
- seed 1234 with stage seeds 1235 and 1236;
- AdamW at learning rate 0.001, batch size 32;
- 32 latent dimensions; and
- no projector in configuration, training, artifacts, or runtime.

The sole numerical change is the visual-only local bone-translation gate from
1.000 mm to 2.000 mm. All joint, FK, sole, contact, finiteness, stepper, and
full-route thresholds remain byte-for-byte equivalent to V1.

## Immutable Outputs and Labels

V1 and its rejection receipt remain untouched. V2 uses a distinct fixed output
directory and persistent exclusive-fit claim. Its manifest, receipt, viewer,
and overlay must display:

`PRELIMINARY V2 LEARNED EXACT-ROUTE OVERFIT (POST-HOC VISUAL GATE; NOT ACCEPTED/NO GENERALIZATION)`

No V2 artifact may use an accepted production schema or scope.

## Transaction and Stop Rules

The run stops and publishes only a rejection receipt on the first exception,
nonfinite value, failed 64-row preflight, failed decompressor gate, failed
stepper gate, failed export/reload check, or failed 595-row rollout. It never
trains or loads a projector.

If the decompressor passes every unchanged gate plus the 2.000 mm visual-only
translation limit, V2 may train the unchanged stepper. Publication requires a
fresh-process reload and a complete 595-row learned rollout that passes the
existing joint, planar-root, contact, terrain-resampling, and bitwise-pause
checks. There is one V2 GPU run and no retry, seed change, further threshold
change, or budget tuning.

## Viewer

The MuJoCo viewer uses the authenticated captured G1 XML/assets and exact green
slope surface. Each held-`W` tick performs one learned recurrent step and an
authoritative authored-route terrain resample before final decode. Releasing
`W` leaves learned state, pose, row, and counters bitwise unchanged. `X` or
Escape exits. The preliminary/post-hoc label remains visible throughout.

## Verification

Implementation is test-first. Tests must prove V1 remains immutable and still
rejects at 1.000 mm, V2 alone uses 2.000 mm, all other configuration and gates
are unchanged, the output/claim is exclusive, rejected outputs contain no
model binaries, accepted preliminary outputs contain exactly latent,
decompressor, stepper, receipt, and manifest artifacts, and the viewer reruns
the full-route gate before opening a display.
